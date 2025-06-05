#include <torch/types.h>
#include <torch/torch.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/amd_detail/amd_hip_fp16.h>

#include <rocwmma/rocwmma.hpp>
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

using rocwmma::accumulator;
using rocwmma::col_major;
using rocwmma::matrix_a;
using rocwmma::matrix_b;
using rocwmma::row_major;

using rocwmma::bfloat16_t;
using rocwmma::float16_t;
using rocwmma::float32_t;

#define ComputeType_Out float

#define USE_HALF 0

#if USE_HALF
#define MAX_NUM 30000.0 // 65504.0
#define ComputeType float16_t
#define AT_PTR_TYPE at::Half
#define TORCH_DTYPE torch::kFloat16
#else

#define MAX_NUM INFINITY
#define ComputeType bfloat16_t
#define AT_PTR_TYPE at::BFloat16
#define TORCH_DTYPE torch::kBFloat16

#endif

constexpr int ROCWMMA_M = 16;
constexpr int ROCWMMA_N = 16;
constexpr int ROCWMMA_K = 16;

//constexpr int N_WAVES = 16;
constexpr int WAVE_SIZE = 32;

using float8 = __attribute__((__vector_size__(8 * sizeof(float)))) float;
using float_v16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
using float_v4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

using bit16_t = uint16_t;
using bit16x4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
typedef bit16x4 bhalf4;

using bit16x8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
union b16x8_u {
    bit16x8 u16x8;
    bhalf4 xy[2];
};
typedef b16x8_u bhalf8;

using bit16x16 =
    __attribute__((__vector_size__(16 * sizeof(uint16_t)))) uint16_t;
union b16x16_u {
    bit16x16 u16x16;
    bhalf8 xy[2];
};
typedef b16x16_u bhalf16;

typedef float8 fp32_frag;
typedef bhalf16 bf16_frag;

//typedef uint16_t bf16_frag __attribute__((ext_vector_type(16)));
//typedef float fp32_frag __attribute__((ext_vector_type(8)));
//typedef uint16_t bhalf4 __attribute__((ext_vector_type(4)));
//typedef uint16_t bhalf8 __attribute__((ext_vector_type(8)));
//typedef uint16_t bhalf16 __attribute__((ext_vector_type(16)));
#define HALF16(pointer) (reinterpret_cast<bhalf16 *>((void *)&(pointer))[0])
#define HALF8(pointer) (reinterpret_cast<bhalf8 *>((void *)&(pointer))[0])
//typedef float float8 __attribute__((ext_vector_type(8)));
//typedef float float_v4 __attribute__((ext_vector_type(4)));
//typedef float float_v16 __attribute__((ext_vector_type(16)));
#define FLOAT8(pointer) (reinterpret_cast<float8 *>((void *)&(pointer))[0])
#define FLOATV16(pointer) (reinterpret_cast<float_v16 *>((void *)&(pointer))[0])
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

__device__ __forceinline__ bhalf4 f32_to_bf16_4(const float_v4& val)
{
    bhalf4 ret;

    for (int i = 0; i < 4; i++) {
        union fcvt {
            uint32_t u32;
            float f32;
        } u;

        u.f32 = val[i];
        u.u32 += 0x7fff + ((u.u32 >> 16) & 1);  // BF16 RNE with no nan/inf check
        ret[i] = uint16_t(u.u32 >> 16);
    }

    return ret;
}


__device__ __forceinline__ uint16_t f32_to_bf16(float val)
{
    uint16_t res = 0;
    union
    {
        float val_f32;
        uint32_t val_u32;
    } u = {val};

    u.val_u32 += 0x7fff + ((u.val_u32 >> 16) & 1);
    res = uint16_t(u.val_u32 >> 16);

    return res;
}

__device__ __forceinline__ bhalf8 f32_to_bf16_8(const float8& inp) {
    bhalf8 ret;
    for (int i = 0; i < 8; i++) {
        union fcvt {
            uint32_t u32;
            float f32;
        } u;
        u.f32 = inp[i];
        u.u32 += 0x7fff + ((u.u32 >> 16) & 1);  // BF16 RNE with no nan/inf check
        ret.u16x8[i] = uint16_t(u.u32 >> 16);
    }
    return ret;
}

__device__ __forceinline__ bhalf16 f32_to_bf16_16(const float_v16& inp) {
    bhalf16 ret;
    for (int i = 0; i < 16; i++) {
        union fcvt {
            uint32_t u32;
            float f32;
        } u;
        u.f32 = inp[i];
        u.u32 += 0x7fff + ((u.u32 >> 16) & 1);  // BF16 RNE with no nan/inf check
        ret.u16x16[i] = uint16_t(u.u32 >> 16);
    }
    return ret;
}

__device__ __forceinline__ float bf16_to_f32(uint16_t val)
{
    union
    {
        float val_f32;
        uint32_t val_u32;
    } u;
    u.val_u32 = val << 16;
    return u.val_f32;
}

//================================ Matrix multiplication ===============================
// C = A @ (B^T)
template <int N_WAVES>
__device__ void mul_A_BT(
    ComputeType *__restrict__ A,
    ComputeType *__restrict__ B,
    ComputeType_Out *__restrict__ C,
    int lda, int ldb, int ldc, // ldq, ld_kv, bc
    int m, int n, int k, // br, bc d
    const float scale)
{
    bf16_frag fragA[2];
    bf16_frag fragB[2];

    const int wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
    const int lane_id = threadIdx.x % WAVE_SIZE;
    const int wmma_lane = (threadIdx.x % 16);

    for (int wave_off = 0; wave_off < ((m * n) / (ROCWMMA_M * ROCWMMA_N) + N_WAVES - 1) / N_WAVES; wave_off++)
    {
        int wave_xy = __builtin_amdgcn_readfirstlane(wave_id + wave_off * N_WAVES); // N_WAVES=16

        int wave_x = __builtin_amdgcn_readfirstlane(wave_xy % (n / ROCWMMA_N)); // wave_xy & (128/16)
        int wave_y = __builtin_amdgcn_readfirstlane(wave_xy / (n / ROCWMMA_N)); // wave_xy % (128/16)

        int blk_x = __builtin_amdgcn_readfirstlane(wave_x * ROCWMMA_N);
        int blk_y = __builtin_amdgcn_readfirstlane(wave_y * ROCWMMA_M);
        if ((blk_x < n) && (blk_y < m))
        {
            fp32_frag fragACC = {};
            for (int i = 0; i < k; i += ROCWMMA_K * 2)
            {
                fragA[0] = HALF16((A + (blk_y * lda + i))[wmma_lane * lda]); // k
                fragB[0] = HALF16((B + (blk_x * ldb + i))[wmma_lane * ldb]); // k

                fragA[1] = HALF16((A + (blk_y * lda + i + ROCWMMA_K))[wmma_lane * lda]);
                fragB[1] = HALF16((B + (blk_x * ldb + i + ROCWMMA_K))[wmma_lane * ldb]);
                // fragA[2] = HALF16((A + (blk_y * k + i + 2*ROCWMMA_K))[wmma_lane * k]);
                // fragB[2] = HALF16((B + (blk_x * k + i + 2*ROCWMMA_K))[wmma_lane * k]);
                // fragA[3] = HALF16((A + (blk_y * k + i + 3*ROCWMMA_K))[wmma_lane * k]);
                // fragB[3] = HALF16((B + (blk_x * k + i + 3*ROCWMMA_K))[wmma_lane * k]);

                fragACC = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(fragA[0].u16x16, fragB[0].u16x16, fragACC);
                fragACC = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(fragA[1].u16x16, fragB[1].u16x16, fragACC);
                // fragACC = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(fragA[2], fragB[2], fragACC);
                // fragACC = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(fragA[3], fragB[3], fragACC);
            }
            //fragACC = fragACC * scale;
            __syncthreads();
            for (int ele = 0; ele < 8; ++ele)
            {
                const int r = ele * 2 + (lane_id / 16);
                (C + (blk_y * ldc + blk_x))[r * ldc + wmma_lane] = fragACC[ele] * scale; // n
            }
        }
    }
    // asm volatile("s_sleep 0");
}


template <int N_WAVES>
__device__ void mul_add_A_B(
    ComputeType *__restrict__ A,
    ComputeType *__restrict__ B,
    ComputeType_Out *__restrict__ C,
    int lda, int ldb, int ldc, // 2*bc, ld_kv, d
    const int m, const int n, const int k) // br, d, bc
{

    rocwmma::fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, row_major> fragA[2];
    rocwmma::fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, row_major> fragB[2];
    rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType_Out> fragC;
    //rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t> fragACC;

    const int wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
    //const int lane_id = threadIdx.x % WAVE_SIZE;
    //const int wmma_lane = (threadIdx.x % 16);

    for (int wave_off = 0; wave_off < ((m * n) / (ROCWMMA_M * ROCWMMA_N) + N_WAVES - 1) / N_WAVES; wave_off++)
    {
        int wave_xy = __builtin_amdgcn_readfirstlane(wave_id + wave_off * N_WAVES);

        int wave_x = __builtin_amdgcn_readfirstlane(wave_xy % (n / ROCWMMA_N));
        int wave_y = __builtin_amdgcn_readfirstlane(wave_xy / (n / ROCWMMA_N));

        int blk_x = __builtin_amdgcn_readfirstlane(wave_x * ROCWMMA_N);
        int blk_y = __builtin_amdgcn_readfirstlane(wave_y * ROCWMMA_M);
        if ((blk_x < n) && (blk_y < m))
        {
            rocwmma::load_matrix_sync(fragC, C + (blk_y * ldc + blk_x), ldc, rocwmma::mem_row_major); //n
            //rocwmma::fill_fragment(fragACC, (float32_t)0.0);
            for (int i = 0; i < k; i += ROCWMMA_K * 2)
            {
                rocwmma::load_matrix_sync(fragA[0], A + (blk_y * lda + i), lda); //k
                rocwmma::load_matrix_sync(fragB[0], B + (i * ldb + blk_x), ldb); //n
                rocwmma::load_matrix_sync(fragA[1], A + (blk_y * lda + (i + 1 * ROCWMMA_K)), lda);
                rocwmma::load_matrix_sync(fragB[1], B + ((i + 1 * ROCWMMA_K) * ldb + blk_x), ldb);
                // rocwmma::load_matrix_sync(fragA[2], A + (blk_y * k + (i + 2 * ROCWMMA_K)), k);
                // rocwmma::load_matrix_sync(fragB[2], B + ((i + 2 * ROCWMMA_K) * n + blk_x), n);
                // rocwmma::load_matrix_sync(fragA[3], A + (blk_y * k + (i + 3 * ROCWMMA_K)), k);
                // rocwmma::load_matrix_sync(fragB[3], B + ((i + 3 * ROCWMMA_K) * n + blk_x), n);

                rocwmma::mma_sync(fragC, fragA[0], fragB[0], fragC);
                rocwmma::mma_sync(fragC, fragA[1], fragB[1], fragC);
                // rocwmma::mma_sync(fragACC, fragA[2], fragB[2], fragACC);
                // rocwmma::mma_sync(fragACC, fragA[3], fragB[3], fragACC);
            }

            //for (int i = 0; i < fragC.num_elements; ++i)
            //{
            //    fragC.x[i] = fragACC.x[i] + fragC.x[i];
            //}
            rocwmma::store_matrix_sync(C + (blk_y * ldc + blk_x), fragC, ldc, rocwmma::mem_row_major); //n
        }
    }
    //__syncthreads();
}

// C = AB + C
template <int N_WAVES>
__device__ void mul_add_A_B_mask_k(
    ComputeType *__restrict__ A,
    ComputeType *__restrict__ B,
    ComputeType_Out *__restrict__ C,
    int lda, int ldb, int ldc,
    const int m, const int n, const int k, const int mask_k_start)
{

    rocwmma::fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, row_major> fragA[2];
    rocwmma::fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, row_major> fragB[2];
    rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType_Out> fragC;
    //rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t> fragACC;

    const int wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
    const int tid = threadIdx.x % (WAVE_SIZE);
    const int wmma_lane = (threadIdx.x % 16);
    for (int wave_off = 0; wave_off < ((m * n) / (ROCWMMA_M * ROCWMMA_N) + N_WAVES - 1) / N_WAVES; wave_off++)
    {
        int wave_xy = __builtin_amdgcn_readfirstlane(wave_id + wave_off * N_WAVES);

        int wave_x = __builtin_amdgcn_readfirstlane(wave_xy % (n / ROCWMMA_N));
        int wave_y = __builtin_amdgcn_readfirstlane(wave_xy / (n / ROCWMMA_N));

        int blk_x = __builtin_amdgcn_readfirstlane(wave_x * ROCWMMA_N);
        int blk_y = __builtin_amdgcn_readfirstlane(wave_y * ROCWMMA_M);

        int wmma_k_end = (mask_k_start / (ROCWMMA_K * 2)) * ROCWMMA_K * 2;

        if ((blk_x < n) && (blk_y < m))
        {
            rocwmma::load_matrix_sync(fragC, C + (blk_y * ldc + blk_x), ldc, rocwmma::mem_row_major); //n
            //rocwmma::fill_fragment(fragACC, (float32_t)0.0);
            for (int i = 0; i < wmma_k_end; i += ROCWMMA_K * 2)
            {
                rocwmma::load_matrix_sync(fragA[0], A + (blk_y * lda + i), lda); //k
                rocwmma::load_matrix_sync(fragB[1], B + ((i + 1 * ROCWMMA_K) * ldb + blk_x), ldb);

                rocwmma::load_matrix_sync(fragA[1], A + (blk_y * lda + (i + 1 * ROCWMMA_K)), lda);
                rocwmma::load_matrix_sync(fragB[0], B + (i * ldb + blk_x), ldb);  //n

                rocwmma::mma_sync(fragC, fragA[0], fragB[0], fragC);
                rocwmma::mma_sync(fragC, fragA[1], fragB[1], fragC);
            }
            //for (int i = 0; i < fragC.num_elements; ++i)
            //{
            //    fragC.x[i] = fragACC.x[i] + fragC.x[i];
            //}
            rocwmma::store_matrix_sync(C + (blk_y * ldc + blk_x), fragC, ldc, rocwmma::mem_row_major);

            {
                for (int y = blk_y; y < blk_y + ROCWMMA_M; y += WAVE_SIZE/ROCWMMA_M)
                {
                    int x = blk_x + (tid % ROCWMMA_N);
                    float32_t acc0 = (0);
                    float32_t acc1 = (0);
                    for (int i = wmma_k_end; i < mask_k_start; i++)
                    {
                        acc0 += A[y * lda + i] * B[i * ldb + x];  // k n
                        acc1 += A[(y + 1) * lda + i] * B[i * ldb + x];
                    }
                    C[y * ldc + x] = C[y * ldc + x] + acc0;
                    C[(y + 1) * ldc + x] = C[(y + 1) * ldc + x] + acc1; // n
                }
            }
        }
    }
    //__syncthreads();
}

// =========================================================================================

template <bool pad_mask, bool causal, int N_WAVES>
__global__ void
__launch_bounds__(WAVE_SIZE * N_WAVES)
    fwd_kernel(
        ComputeType *__restrict__ q,
        ComputeType *__restrict__ k,
        ComputeType *__restrict__ v,
        ComputeType *__restrict__ o,
        float *__restrict__ L,
        const int Tr, const int Tc, const int Br, const int Bc,
        const int nq, const int nkv,
        const int d,
        const int64_t q_stride0,const int64_t q_stride1,const int64_t q_stride2,
        const int64_t kv_stride0,const int64_t kv_stride1,const int64_t kv_stride2,
        const int L_stride_b, const int L_stride_h,
        const float32_t scale, const bool permute_NH)
{

    int q_offset = blockIdx.x * q_stride0 + blockIdx.y * q_stride1;
    int kv_offset = blockIdx.x * kv_stride0 + blockIdx.y * kv_stride1;

    int ld_q = q_stride2; //d;
    int ld_kv = kv_stride2;
    if(permute_NH)
    {
        q_offset = blockIdx.x * q_stride0 + blockIdx.y * q_stride2;
        kv_offset = blockIdx.x * kv_stride0 + blockIdx.y * kv_stride2;
        ld_q = q_stride1; //h * d;
        ld_kv = kv_stride1;
    }

    const int L_offset = blockIdx.x * L_stride_b + blockIdx.y * L_stride_h;

    const int Tr_i = blockIdx.z;
    if (Tr_i >= Tr)
        return;
    const int ele_y = __builtin_amdgcn_readfirstlane(Tr_i * Br);
    const int yb = __builtin_amdgcn_readfirstlane(ele_y + Br);
    const int tx = threadIdx.x;

    extern __shared__ char sram[];

//    uintptr_t raw_base = reinterpret_cast<uintptr_t>(sram);
//    uintptr_t aligned_base = (raw_base + 63) & ~uintptr_t(63);
//    uintptr_t offset_Si = 0;
//    uintptr_t offset_Oi = offset_Si + sizeof(ComputeType_Out) * Br * Bc;

//    ComputeType_Out* __restrict__ Si = reinterpret_cast<ComputeType_Out*>(aligned_base + offset_Si);       // Br * Bc
//    ComputeType_Out* __restrict__ Oi = reinterpret_cast<ComputeType_Out*>(aligned_base + offset_Oi); // Br * d
//    float_v16* Oi_vec = reinterpret_cast<float_v16*>(Oi);
    ComputeType_Out* __restrict__ Si = reinterpret_cast<ComputeType_Out*>(&sram[0]);       // Br * Bc
    ComputeType_Out* __restrict__ Oi = reinterpret_cast<ComputeType_Out*>(&sram[sizeof(ComputeType_Out) * Br * Bc]); // Br * d
    // ComputeType *__restrict__ Qi = &sram[Br * Bc + Br * d]; // Br * d
    // ComputeType *__restrict__ Vj = &sram[Br * Bc + Br * d]; // Bc * d


    if (tx < Br) // Br = 64
    {
        int Oi_row = tx * d;
#pragma unroll 4
        for (int i = 0; i < d; i += 16) // d = 128
        {
            // Load Qi into sram, fill 0 to Oi
            // Qi[tx * d + i] = q[q_offset + Tr_i * Br * d + tx * d + i];
            // FLOAT8(Qi[tx * d + i]) = FLOAT8((&(q[q_offset + Tr_i * Br * d]))[tx * d + i]);
            FLOATV16(Oi[Oi_row + i]) = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        }
        // #pragma unroll 32
        // for (int i = 0; i < d; i++)
        {
            // pre-scale, Si=(Q @ K^T)*scale
            // Qi[tx * d + i] *= scale * 1.442695f; // 1/ln2=1.442695, exp(x)=exp2f((1/ln2)*x)
        }
        // #pragma unroll 4
        // for (int i = 0; i < Bc; i++)
        // {
        //     FLOAT8(Si[tx * d + i]) = {0, 0, 0, 0, 0, 0, 0, 0};
        // }
    }
//    const int chunk_size_Oi = 8;
//    int num_th_row_Oi = d / chunk_size_Oi; // 128 / 8 = 16
//    int Oi_row = tx / num_th_row_Oi;
//    int Oi_col = (tx % num_th_row_Oi) * chunk_size_Oi;
//    int Oi_idx = Oi_row * d + Oi_col;
//
//    if (tx < num_th_row_Oi * Br)
//    {
//        FLOAT8(Oi[Oi_idx]) = {0, 0, 0, 0, 0, 0, 0, 0};
//    }
//    __syncthreads();

    ComputeType *__restrict__ Qi = &q[q_offset + (Tr_i * Br) * ld_q];
    // ComputeType *__restrict__ Oi = &o[q_offset + Tr_i * Br * d];

    float row_max_old = -FLT_MAX;
    float l_i = 0;

    for (int j = 0; j < Tc; j++)
    {
        ComputeType *__restrict__ Kj = &k[kv_offset + (j * Bc) * ld_kv];
        ComputeType *__restrict__ Vj = &v[kv_offset + (j * Bc) * ld_kv];
        int ele_x = j * Bc;
        int xr = ele_x + Bc;
        float row_max_new = -FLT_MAX; // mij
        float row_sum = 0.0f;
        float rowmax_diff_exp = 0.0f; // Sij - mij
        //------------ Sij = Qi @ Kj^T
        if constexpr (!causal)
        {
            mul_A_BT<N_WAVES>(Qi, Kj, Si, ld_q, ld_kv, Bc, Br, Bc, d, scale);
        }
        else
        {
            if (ele_y >= ele_x)
            {
                mul_A_BT<N_WAVES>(Qi, Kj, Si,ld_q, ld_kv, Bc,  Br, Bc, d, scale);
                __syncthreads();
            }
            if ((ele_y < ele_x + Bc - 1) && (tx < Br))
            {
#pragma unroll 32
                for (int i = 0; i < Bc; i++)
                {
                    if (i >= tx + (ele_y - ele_x + 1))
                        Si[tx * Bc + i] = -FLT_MAX;
                }
            }
        }
        __syncthreads();
        //------------
        if constexpr (pad_mask)
        {
            if (unlikely((xr > nkv) && (tx < Br)))
            {
#pragma unroll 32
                for (int i = nkv - ele_x; i < Bc; i++)
                    Si[tx * Bc + i] = -FLT_MAX;
            }

            if (unlikely((yb > nq) && (tx < Bc)))
            {
#pragma unroll 32
                for (int i = nq - ele_y; i < Br; i++)
                    Si[i * Bc + tx] = -FLT_MAX;
            }
            __syncthreads();
        }
        //------------

        float val32 = row_max_new;
        const int chunk_size_Si = 8;
        int num_th_row_Si = Bc / chunk_size_Si; // 128 / 8 = 16
        int qk_row = tx / num_th_row_Si;
        int qk_col = (tx % num_th_row_Si) * chunk_size_Si;
        int qk_idx = qk_row * Bc + qk_col;
//        int qk_idx2 = qk_row * Bc * 2 + qk_col * chunk_size_Si;
        float local_row_sum = 0.0f;

        if (tx < num_th_row_Si * Br)
        {
            float8 val_f32 = FLOAT8(Si[qk_idx]);

            for (int k = 0; k < 8; ++k)
                val32 = fmaxf(val32, val_f32[k]);

            for (int mask = 8; mask > 0; mask >>= 1) 
                val32 = fmaxf(val32, __shfl_xor(val32, mask, 16));
//            float row_max = __shfl(val32, 0, 16);

            row_max_new = val32;

            row_max_new = max(row_max_old, row_max_new);
            //rowmax_diff_exp = exp2f(row_max_old - row_max_new);
            rowmax_diff_exp = expf(row_max_old - row_max_new);
            row_max_old = row_max_new;

//// Si - mi
//            val_f32 = val_f32 - row_max_new;
//#pragma unroll // exp but using exp2 instead.
//            for (int k = 0; k < 8; ++k)
//                //val_f32[j] = exp2f(val_f32[j]);
//                val_f32[k] = expf(val_f32[k]);

////#pragma unroll
//            for (int k = 0; k < 8; ++k) {
//                local_row_sum += val_f32[k];
//            }
//
//            __syncthreads();
//            for (int mask = 8; mask > 0; mask >>= 1) {
//                local_row_sum += __shfl_xor(local_row_sum, mask);
//            }
//            __syncthreads();
//            row_sum = local_row_sum;
//
//            FLOAT8(Si[qk_idx]) = val_f32;

        }
//        __syncthreads();

        if (tx < Br)
        {
////// --------------------- find every row max val in Si[Br * Bc]
//////            float val32 = row_max_new;
//////#pragma unroll 2
//////            for (int i = 0; i < Bc; i += 16)
//////            {
//////                float_v16 val_f32 = FLOATV16(Si[(tx * Bc) + i]); 
//////#pragma unroll
//////                for (int k = 0; k < 16; k++)
//////                    val32 = max(val32, val_f32[k]); // V_PK_MAX_F16
//////            }
////
//////--------------------Calc Pi = exp(Si - mi) and rowsum 
#pragma unroll 4
            for (int i = 0; i < Bc; i += 16)
            {
                float_v16 val_f32 = FLOATV16(Si[(tx * Bc) + i]);
// Si - mi
                val_f32 = val_f32 - row_max_new;
#pragma unroll // exp but using exp2 instead.
                for (int k = 0; k < 16; k++) {
//                    //val_f32[j] = exp2f(val_f32[j]);
                    val_f32[k] = expf(val_f32[k]);
                }
//
#pragma unroll // calc rowsum
                for (int k = 0; k < 16; k++)
                    row_sum += val_f32[k];

                bhalf16 val;
                val = f32_to_bf16_16(val_f32);
////#pragma unroll
////                for (int k = 0; k < 16; k++)
////                    val[k] = f32_to_bf16(val_f32[k]);
////////////                bhalf4* val_ptr = reinterpret_cast<bhalf4*>(&val);
////////////
////////////                const float* val_f32_raw = reinterpret_cast<const float*>(&val_f32);
////////////#pragma unroll
////////////                for (int k = 0; k < 4; k++)
////////////                    val_ptr[k] = f32_to_bf16_4(*reinterpret_cast<const float_v4*>(&val_f32_raw[k * 4]));
//////////
                // write back
                HALF16((reinterpret_cast<ComputeType*>(Si))[(tx * Bc * 2) + i]) = val;
            }
            l_i = rowmax_diff_exp * l_i + row_sum;
#pragma unroll 4
            for (int i = 0; i < d; i += 16)
            {
                float_v16 val_f32 = FLOATV16(Oi[(tx * d) + i]);

                val_f32 *= rowmax_diff_exp;

                FLOATV16(Oi[(tx * d) + i]) = val_f32;
            }
// --------------------- 
        }
        __syncthreads();

//        int chunk_size_Oi = 16;
//        int num_th_row_Oi = d / chunk_size_Oi; // 128 / 16 = 8
//        int Oi_row = tx / num_th_row_Oi;
//        int Oi_col = (tx % num_th_row_Oi) * chunk_size_Oi;
//        int Oi_idx = Oi_row * d + Oi_col;
//        int vec_idx = Oi_idx / 16;
//
//        if (tx < num_th_row_Oi * Br) {
//            float_v16 val_f32 = Oi_vec[vec_idx];
//            if (Tr_i == 0 && tx < 16) {
//                printf("Tc[%d] tx=%d Oi_row=%d, Oi_col=%d, vec_idx=%d, r_maxdiff=%f\n", j, tx, Oi_row, Oi_col, vec_idx, rowmax_diff_exp);
//            }
//                        printf("\n");
//                    }
//                }
//            val_f32 *= rowmax_diff_exp;
//            Oi_vec[vec_idx] = val_f32;
//
//            if (tx == 0) {
//                printf("Oi_vec addr = %p (mod 64 = %lu)\n", Oi_vec, reinterpret_cast<uintptr_t>(Oi_vec) % 64);
//                printf("rowmax_diff_exp = %f\n", rowmax_diff_exp);
//                if (vec_idx == 0) {
//                    for (int i = 0; i < 16; ++i) {
//                        printf("tx=%d val[%d]=%f\n", tx, i, val_f32[i]);
//                    }
//                }
//            }
//
//            val_f32 *= rowmax_diff_exp;
//            Oi_vec[vec_idx] = val_f32;
//        }
//        __syncthreads();
//
//        int qk_idx2 = qk_row * num_th_row_Si + tx % num_th_row_Si;
// 
//        if (tx < num_th_row_Si * Br)
//        {
//            float8 val_f32 = FLOAT8(Si[qk_idx]);
//            bhalf8 val;
//            val = f32_to_bf16_8(val_f32);
//            HALF8((reinterpret_cast<ComputeType*>(Si))[qk_idx2]) = val;
//#pragma unroll
//            for (int k = 0; k < 8; k++)
//                val[k] = f32_to_bf16(val_f32[k]);
//
//            reinterpret_cast<bhalf8*>(Si)[qk_idx2] = val;
//        }
//        __syncthreads();


//        if (tx < Br && tx % 2 == 1)
//        {
//            for (int i = 0; i < Bc; i += 16) {
//                HALF16((reinterpret_cast<ComputeType*>(Si))[(tx * Bc) + i]) = HALF16((reinterpret_cast<ComputeType*>(Si))[(tx * Bc * 2) + i]);
//            }
//        }
//        __syncthreads();
//            
//        if (tx < Br && tx % 2 == 0)
//        {
//            for (int i = 0; i < Bc; i += 16) {
//                HALF16((reinterpret_cast<ComputeType*>(Si))[(tx * Bc) + i]) = HALF16((reinterpret_cast<ComputeType*>(Si))[(tx * Bc * 2) + i]);
//            }
//        }
//        __syncthreads();

        if constexpr (!pad_mask)
            mul_add_A_B<N_WAVES>(reinterpret_cast<ComputeType*>(Si), Vj, Oi,   2*Bc,ld_kv,d,   Br, d, Bc);
        else
        {
            if (unlikely(xr > nkv))
            {
                mul_add_A_B_mask_k<N_WAVES>(reinterpret_cast<ComputeType*>(Si), Vj, Oi,   2*Bc,ld_kv,d,  Br, d, Bc, Bc - (xr - nkv));
            }
            else
            {
                mul_add_A_B<N_WAVES>(reinterpret_cast<ComputeType*>(Si), Vj, Oi,   2*Bc,ld_kv,d,    Br, d, Bc);
            }
        }

        __syncthreads();
    }

    if (tx < Br)
    {
// #pragma unroll 32
//         for (int i = 0; i < d; i++)
//             Oi[tx * d + i] = Oi[tx * d + i] / l_i;

//------------------------ Calc: Oi /= li  Write back: Oi
// #pragma unroll 4
            for (int i = 0; i < d; i += 16)
            {
                float_v16 val_f32 = FLOATV16(Oi[(tx * d) + i]);

                val_f32 = val_f32 / l_i;

                bhalf16 val;
                val = f32_to_bf16_16(val_f32);
//#pragma unroll
//                for (int k = 0; k < 16; k++)
//                    val.u16x16[k] = f32_to_bf16(val_f32[k]);

//                bhalf4* val_ptr = reinterpret_cast<bhalf4*>(&val);
//
//                const float* val_f32_raw = reinterpret_cast<const float*>(&val_f32);
//#pragma unroll
//                for (int k = 0; k < 4; k++)
//                    val_ptr[k] = f32_to_bf16_4(*reinterpret_cast<const float_v4*>(&val_f32_raw[k * 4]));

                //HALF8(Oi[(tx * d) + i]) = val;
                HALF16((&(o[q_offset + (Tr_i * Br) * ld_q]))[tx * ld_q + i]) = val;
            }

// #pragma unroll 4
//         for (int i = 0; i < d; i += 16)
//             // o[q_offset + Tr_i * Br * d + tx * d + i] = Oi[tx * d + i];
//             FLOAT8((&(o[q_offset + Tr_i * Br * d]))[tx * d + i]) = FLOAT8(Oi[tx * d + i]);

        l_i = row_max_old + logf(l_i);
        //l_i = row_max_old + log2f(l_i);
        L[L_offset + Tr_i * Br + tx] = l_i;
    }
}
// =================================================================================


template <bool pad_mask, bool causal, int N_WAVES>
__global__ void
__launch_bounds__(WAVE_SIZE *N_WAVES)
bwd_kernel(
    ComputeType *__restrict__ q,  // [(b*h) x N x d]
    ComputeType *__restrict__ k,  // [(b*h) x N x d]
    ComputeType *__restrict__ v,  // [(b*h) x N x d]
    ComputeType *__restrict__ O,  // [(b*h) x N x d]
    ComputeType *__restrict__ dO, // [(b*h) x N x d]
    ComputeType *__restrict__ dQ, // [(b*h) x N x d]
    ComputeType *__restrict__ dK, // [(b*h) x N x d]
    ComputeType *__restrict__ dV, // [(b*h) x N x d]
    float *__restrict__ Di,       // [(b*h) * N]
    float *__restrict__ L,        // [(b*h) * N]
    const int Tr, const int Tc,
    const int Br, const int Bc,
    const int nq, const int nkv,
    const int d,
    const int64_t Q_O_dO_stride_0, const int64_t Q_O_dO_stride_1, const int64_t Q_O_dO_stride_2,
    const int64_t kvDkv_stride_0, const int64_t kvdKv_stride_1, const int64_t kvdKv_stride_2,
    const int64_t L_stride_b, const int64_t L_stride_h,
    const float32_t scale, const bool permute_NH
    )

{
    
}

// =================================================================================

std::vector<torch::Tensor> forward_bf16(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    const int Br, const int Bc,
    const bool causal,
    const float scale, const bool permute_NH
)
{

    auto q_pad = q;
    auto k_pad = k;
    auto v_pad = v;

    const int b = q.size(0);
    const int h = permute_NH ? q.size(2):q.size(1);
    const int n = permute_NH ? q.size(1):q.size(2);
    const int d = q.size(3);

    const int n_kv = permute_NH ? k.size(1):k.size(2);
    

    int Nq_pad_sz = (Br - (n % Br)) % Br;
    int Nkv_pad_sz = (Bc - (n_kv % Bc)) % Bc;
    int d_pad_sz = ((ROCWMMA_K * 2) - (d % (ROCWMMA_K * 2))) % (ROCWMMA_K * 2);

    const bool pad_mask = Nq_pad_sz || Nkv_pad_sz;

    if (Nq_pad_sz || d_pad_sz)
    {
        if(permute_NH)
            q_pad = torch::nn::functional::pad(q_pad, torch::nn::functional::PadFuncOptions({0, d_pad_sz, 0,0, 0, Nq_pad_sz}));
        else
            q_pad = torch::nn::functional::pad(q_pad, torch::nn::functional::PadFuncOptions({0, d_pad_sz, 0, Nq_pad_sz})); 
    }
    // if (Nkv_pad_sz || d_pad_sz)
    if (d_pad_sz)
    {
        k_pad = torch::nn::functional::pad(k_pad, torch::nn::functional::PadFuncOptions({0, d_pad_sz}));
        v_pad = torch::nn::functional::pad(v_pad, torch::nn::functional::PadFuncOptions({0, d_pad_sz}));
    }
    if (q_pad.stride(-1) != 1)
        q_pad = q_pad.contiguous();

    if (k_pad.stride(-1) != 1)
        k_pad = k_pad.contiguous();

    if (v_pad.stride(-1) != 1)
        v_pad = v_pad.contiguous();

    const int Tr = ceil((float)(n+Nq_pad_sz) / Br); // seqlen_q / Br # of iters for total rows
    const int Tc = ceil((float)(n_kv+Nkv_pad_sz) / Bc); // seqlen_kv / Bc

    // auto opt = torch::TensorOptions().dtype(TORCH_DTYPE).device(torch::kCUDA);
    auto O = torch::zeros_like(q_pad);

    auto opt2 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto L = torch::zeros({b, h, n + Nq_pad_sz}, opt2);

    int N_WAVES = 16;
    if(d + d_pad_sz == 128)
         N_WAVES = 32;

    auto blockDim = dim3(WAVE_SIZE * N_WAVES); // 32 * 16 = 512
    int nblk = b * h * Tr;
    int trPad = 96 - (nblk % 96); // TODO: 96 CU only for gfx1100

    auto gridDim = dim3(b, h, Tr + trPad);

    const int sram_sz =
        (Br * Bc) * sizeof(ComputeType_Out)               // Si
        + (Br * (d + d_pad_sz)) * sizeof(ComputeType_Out) // Oi
        // + Br * (d + d_pad_sz) * sizeof(ComputeType) // Qi
        // + Bc * (d + d_pad_sz) * sizeof(ComputeType) // Vj
        ;
 
#define para_fwd                                      \
        (ComputeType *)q_pad.data_ptr<AT_PTR_TYPE>(), \
        (ComputeType *)k_pad.data_ptr<AT_PTR_TYPE>(), \
        (ComputeType *)v_pad.data_ptr<AT_PTR_TYPE>(), \
        (ComputeType *)O.data_ptr<AT_PTR_TYPE>(),     \
        (float *)L.data_ptr<float>(),                 \
        Tr, Tc, Br, Bc,                               \
        n + Nq_pad_sz, n_kv + Nkv_pad_sz,             \
        d + d_pad_sz,                                 \
        q_pad.stride(0),q_pad.stride(1),q_pad.stride(2),\
        k_pad.stride(0),k_pad.stride(1),k_pad.stride(2),\
        L.stride(0), L.stride(1),                     \
        scale, permute_NH

    cudaError_t err = cudaGetLastError();

    if (N_WAVES == 32)
    {
        constexpr int NW = 32;
        if (!pad_mask && !causal)
            fwd_kernel<false, false,NW><<<gridDim, blockDim, sram_sz>>>(para_fwd);
        else if (pad_mask && causal)
            fwd_kernel<true, true,NW><<<gridDim, blockDim, sram_sz>>>(para_fwd);
        else if (!pad_mask && causal)
            fwd_kernel<false, true,NW><<<gridDim, blockDim, sram_sz>>>(para_fwd);
        else if (pad_mask && !causal)
            fwd_kernel<true, false,NW><<<gridDim, blockDim, sram_sz>>>(para_fwd);
    }
    else if (N_WAVES == 16)
    {
        constexpr int NW = 16;
        if (!pad_mask && !causal)
            fwd_kernel<false, false,NW><<<gridDim, blockDim, sram_sz>>>(para_fwd);
        else if (pad_mask && causal)
            fwd_kernel<true, true,NW><<<gridDim, blockDim, sram_sz>>>(para_fwd);
        else if (!pad_mask && causal)
            fwd_kernel<false, true,NW><<<gridDim, blockDim, sram_sz>>>(para_fwd);
        else if (pad_mask && !causal)
            fwd_kernel<true, false,NW><<<gridDim, blockDim, sram_sz>>>(para_fwd);
    }


    err = cudaGetLastError();
    if (err != hipSuccess)
    {
        printf("=============== Kernel Launch Failed !!! =============\r\n");
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        printf("Br:%d, Bc:%d \r\n", Br, Bc);
        printf("Tr:%d, Tc:%d \r\n", Tr, Tc);
        printf("B:%d, H:%d, Qn:%d, KVn:%d, d:%d \r\n", b, h, n, n_kv, d);
        printf("SRAM Requirements:%d \r\n", sram_sz);
    }


    auto O_fwd = permute_NH ? 
                    O.index({"...",
                          torch::indexing::Slice(torch::indexing::None, n),
                          torch::indexing::Slice(torch::indexing::None, torch::indexing::None),
                          torch::indexing::Slice(torch::indexing::None, d)})
                    : 
                    O.index({"...",
                          torch::indexing::Slice(torch::indexing::None, n),
                          torch::indexing::Slice(torch::indexing::None, d)});

    return {O_fwd, q_pad, k_pad, v_pad, O, L};
}

std::vector<torch::Tensor> backward_bf16(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor dO,
    torch::Tensor L,
    const int act_n,
    const int act_nkv,
    const int act_d,
    const int Br,
    const int Bc,
    const bool causal,
    const float scale, const bool permute_NH)
{
    auto opt = torch::TensorOptions().dtype(TORCH_DTYPE).device(torch::kCUDA);
    auto dQ = torch::zeros_like(Q, opt);
    auto dK = torch::zeros_like(K, opt);
    auto dV = torch::zeros_like(V, opt);

    return {dQ, dK, dV};
}
