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
#define FLOAT4(pointer) (reinterpret_cast<float_v4 *>(&(pointer))[0])

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


__device__ __forceinline__ bhalf8 f32_to_bf16_8(const float8& inp)
{
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

__device__ __forceinline__ float to_float_b16(const bit16_t& inp)
{
    union tmpcvt {
      bit16_t u;
      _Float16 f;
      __hip_bfloat16 b;
    } t16;
    t16.u = inp;
    return __bfloat162float(t16.b);
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
    //bf16_frag fragA[2];
    //bf16_frag fragB[2];
    rocwmma::fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, row_major> fragA[2];
    rocwmma::fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, col_major> fragB[2];
    rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType_Out> fragACC;

    const int wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
    //const int lane_id = threadIdx.x % WAVE_SIZE;
    //const int row_id = lane_id / 16;
    //const int wmma_lane = (threadIdx.x % 16);

    for (int wave_off = 0; wave_off < ((m * n) / (ROCWMMA_M * ROCWMMA_N) + N_WAVES - 1) / N_WAVES; wave_off++)
    {
        int wave_xy = __builtin_amdgcn_readfirstlane(wave_id + wave_off * N_WAVES); // N_WAVES=16

        int wave_x = __builtin_amdgcn_readfirstlane(wave_xy % (n / ROCWMMA_N)); // wave_xy & (128/16)
        int wave_y = __builtin_amdgcn_readfirstlane(wave_xy / (n / ROCWMMA_N)); // wave_xy % (128/16)

        int blk_x = __builtin_amdgcn_readfirstlane(wave_x * ROCWMMA_N);
        int blk_y = __builtin_amdgcn_readfirstlane(wave_y * ROCWMMA_M);
        if ((blk_x < n) && (blk_y < m))
        {
            //fp32_frag fragACC = {};
            rocwmma::fill_fragment(fragACC, (float32_t)0.0);
            for (int i = 0; i < k; i += ROCWMMA_K * 2)
            {
                //fragA[0] = HALF16((A + (blk_y * lda + i))[wmma_lane * lda]); // k
                //fragB[0] = HALF16((B + (blk_x * ldb + i))[wmma_lane * ldb]); // k

                //fragA[1] = HALF16((A + (blk_y * lda + i + ROCWMMA_K))[wmma_lane * lda]);
                //fragB[1] = HALF16((B + (blk_x * ldb + i + ROCWMMA_K))[wmma_lane * ldb]);
                // fragA[2] = HALF16((A + (blk_y * k + i + 2*ROCWMMA_K))[wmma_lane * k]);
                // fragB[2] = HALF16((B + (blk_x * k + i + 2*ROCWMMA_K))[wmma_lane * k]);
                // fragA[3] = HALF16((A + (blk_y * k + i + 3*ROCWMMA_K))[wmma_lane * k]);
                // fragB[3] = HALF16((B + (blk_x * k + i + 3*ROCWMMA_K))[wmma_lane * k]);

                rocwmma::load_matrix_sync(fragA[0], A + (blk_y * lda + i), lda); //k
                rocwmma::load_matrix_sync(fragB[0], B + (blk_x * ldb + i), ldb); //n

                rocwmma::load_matrix_sync(fragA[1], A + (blk_y * lda + (i + 1 * ROCWMMA_K)), lda);
                rocwmma::load_matrix_sync(fragB[1], B + (blk_x * ldb + (i + 1 * ROCWMMA_K)), ldb);

                rocwmma::mma_sync(fragACC, fragA[0], fragB[0], fragACC);
                rocwmma::mma_sync(fragACC, fragA[1], fragB[1], fragACC);
                //fragACC = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(fragA[0].u16x16, fragB[0].u16x16, fragACC);
                //fragACC = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(fragA[1].u16x16, fragB[1].u16x16, fragACC);
                // fragACC = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(fragA[2], fragB[2], fragACC);
                // fragACC = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(fragA[3], fragB[3], fragACC);
            }
            //fragACC = fragACC * scale;
            //__syncthreads();

            for (int i = 0; i < fragACC.num_elements; ++i)
            {
                fragACC.x[i] = fragACC.x[i] * scale;
            }
            rocwmma::store_matrix_sync(C + (blk_y * ldc + blk_x), fragACC, ldc, rocwmma::mem_row_major); //n
            //for (int ele = 0; ele < 8; ++ele)
            //{
            //    const int r = ele * 2 + row_id;
            //    (C + (blk_y * ldc + blk_x))[r * ldc + wmma_lane] = fragACC[ele] * scale; // n
            //}
        }
    }
    // asm volatile("s_sleep 0");
}


template <int N_WAVES>
__device__ void mul_add_A_B(
    ComputeType *__restrict__ A,
    ComputeType *__restrict__ B,
    ComputeType_Out *__restrict__ C,
    int lda, int ldb, int ldc, // bc, N, d
    const int m, const int n, const int k) // br, d, bc
{
    //bf16_frag fragA[2];
    //bf16_frag fragB[2];
    //fp32_frag fragC;
    rocwmma::fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, row_major> fragA[2];
    rocwmma::fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, col_major> fragB[2];
    rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType_Out> fragC;

    const int wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
    //const int lane_id = threadIdx.x % WAVE_SIZE;
    //const int row_id = lane_id / 16;
    //const int wmma_lane = (threadIdx.x % 16);

    //if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
	// printf("[mul_add_A_B] lda: %d ldb: %d ldc: %d\n", lda, ldb, ldc); 

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
            //fp32_frag fragACC = {};
            //for (int ele = 0; ele < 8; ++ele)
            //{
            //    const int r = ele * 2 + row_id;
            //    fragC[ele] = (C + (blk_y * ldc + blk_x))[r * ldc + wmma_lane]; // n
            //}

            for (int i = 0; i < k; i += ROCWMMA_K * 2)
            {
                //fragA[0] = HALF16((A + (blk_y * lda + i))[wmma_lane * lda]); // k
                //fragB[0] = HALF16((B + (blk_x * ldb + i))[wmma_lane * ldb]); // k

                //fragA[1] = HALF16((A + (blk_y * lda + i + ROCWMMA_K))[wmma_lane * lda]);
                //fragB[1] = HALF16((B + (blk_x * ldb + i + ROCWMMA_K))[wmma_lane * ldb]);

                //fragC = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(fragA[0].u16x16, fragB[0].u16x16, fragC);
                //fragC = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(fragA[1].u16x16, fragB[1].u16x16, fragC);
                rocwmma::load_matrix_sync(fragA[0], A + (blk_y * lda + i), lda); //k
                rocwmma::load_matrix_sync(fragB[0], B + (blk_x * ldb + i), ldb); //n
                //rocwmma::load_matrix_sync(fragB[0], B + (i * ldb + blk_x), ldb); //n
                rocwmma::load_matrix_sync(fragA[1], A + (blk_y * lda + (i + 1 * ROCWMMA_K)), lda);
                rocwmma::load_matrix_sync(fragB[1], B + (blk_x * ldb + (i + 1 * ROCWMMA_K)), ldb);
                //rocwmma::load_matrix_sync(fragB[1], B + ((i + 1 * ROCWMMA_K) * ldb + blk_x), ldb);
                // rocwmma::load_matrix_sync(fragA[2], A + (blk_y * k + (i + 2 * ROCWMMA_K)), k);
                // rocwmma::load_matrix_sync(fragB[2], B + ((i + 2 * ROCWMMA_K) * n + blk_x), n);
                // rocwmma::load_matrix_sync(fragA[3], A + (blk_y * k + (i + 3 * ROCWMMA_K)), k);
                // rocwmma::load_matrix_sync(fragB[3], B + ((i + 3 * ROCWMMA_K) * n + blk_x), n);

                rocwmma::mma_sync(fragC, fragA[0], fragB[0], fragC);
                rocwmma::mma_sync(fragC, fragA[1], fragB[1], fragC);
                // rocwmma::mma_sync(fragACC, fragA[2], fragB[2], fragACC);
                // rocwmma::mma_sync(fragACC, fragA[3], fragB[3], fragACC);
            }
            //__syncthreads();
            //for (int ele = 0; ele < 8; ++ele)
            //{
            //    const int r = ele * 2 + row_id;
            //    (C + (blk_y * ldc + blk_x))[r * ldc + wmma_lane] += fragC[ele]; // n
            //}
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
    rocwmma::fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, col_major> fragB[2];
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
                rocwmma::load_matrix_sync(fragB[0], B + (blk_x * ldb + i), ldb); //n
                //rocwmma::load_matrix_sync(fragB[1], B + ((i + 1 * ROCWMMA_K) * ldb + blk_x), ldb);

                rocwmma::load_matrix_sync(fragA[1], A + (blk_y * lda + (i + 1 * ROCWMMA_K)), lda);
                rocwmma::load_matrix_sync(fragB[1], B + (blk_x * ldb + (i + 1 * ROCWMMA_K)), ldb); //n
                //rocwmma::load_matrix_sync(fragB[0], B + (i * ldb + blk_x), ldb);  //n

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
        const int64_t kv_stride0,const int64_t kv_stride1,const int64_t kv_stride2, const int64_t v_stride2,
        const int L_stride_b, const int L_stride_h,
        const float32_t scale, const bool permute_NH)
{

    int q_offset = blockIdx.x * q_stride0 + blockIdx.y * q_stride1;
    int kv_offset = blockIdx.x * kv_stride0 + blockIdx.y * kv_stride1;

    int ld_q = q_stride2; //d;
    int ld_kv = kv_stride2;
    int ld_v = v_stride2;

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

    ComputeType *__restrict__ Qi = &q[q_offset + (Tr_i * Br) * ld_q];
    // ComputeType *__restrict__ Oi = &o[q_offset + Tr_i * Br * d];

    float row_max_old = -FLT_MAX;
    float l_i = 0;

    const int chunk_size_Si = 8;
    int num_th_row_Si = Bc / chunk_size_Si; // 128 / 8 = 16
    int qk_row = tx / num_th_row_Si;
    int qk_col = (tx % num_th_row_Si) * chunk_size_Si;
    int qk_idx = qk_row * Bc + qk_col;
    //int qk_idx2 = qk_row * Bc + qk_col;

    for (int j = 0; j < Tc; j++)
    {
        ComputeType *__restrict__ Kj = &k[kv_offset + (j * Bc) * ld_kv];
        ComputeType *__restrict__ Vj = &v[kv_offset + (j * Bc)];
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

        float local_row_sum = 0.0f;

        if (tx < num_th_row_Si * Br)
        {
            float8 val_f32 = FLOAT8(Si[qk_idx]);
            float local_row_max = row_max_new;

            for (int k = 0; k < 8; ++k)
                local_row_max = fmaxf(local_row_max, val_f32[k]);

            for (int mask = 8; mask > 0; mask >>= 1) 
                local_row_max = fmaxf(local_row_max, __shfl_xor(local_row_max, mask, 16));

            row_max_new = local_row_max;
            row_max_new = max(row_max_old, row_max_new);
            //rowmax_diff_exp = exp2f(row_max_old - row_max_new);
            rowmax_diff_exp = expf(row_max_old - row_max_new);
            row_max_old = row_max_new;

// Si - mi
            val_f32 = val_f32 - row_max_new;
#pragma unroll // exp but using exp2 instead.
            for (int k = 0; k < 8; ++k)
//                //val_f32[j] = exp2f(val_f32[j]);
                val_f32[k] = expf(val_f32[k]);

#pragma unroll
            for (int k = 0; k < 8; ++k)
                local_row_sum += val_f32[k];

            for (int mask = 8; mask > 0; mask >>= 1)
                local_row_sum += __shfl_xor(local_row_sum, mask, 16);

            row_sum = local_row_sum;

            bhalf8 val;
            val = f32_to_bf16_8(val_f32);

            HALF8((reinterpret_cast<ComputeType*>(Si))[qk_idx]) = val;

            l_i = rowmax_diff_exp * l_i + row_sum;

            val_f32 = FLOAT8(Oi[qk_idx]);

            val_f32 *= rowmax_diff_exp;

            FLOAT8(Oi[qk_idx]) = val_f32;
        }
        __syncthreads();


        if constexpr (!pad_mask)
	{
            mul_add_A_B<N_WAVES>(reinterpret_cast<ComputeType*>(Si), Vj, Oi, Bc, ld_v, d, Br, d, Bc);
	}
        else
        {
            if (unlikely(xr > nkv))
            {
                mul_add_A_B_mask_k<N_WAVES>(reinterpret_cast<ComputeType*>(Si), Vj, Oi, Bc, ld_v,d, Br, d, Bc, Bc - (xr - nkv));
            }
            else
            {
                mul_add_A_B<N_WAVES>(reinterpret_cast<ComputeType*>(Si), Vj, Oi, Bc, ld_v,d, Br, d, Bc);
            }
        }

        __syncthreads();
    }

    if (tx < num_th_row_Si * Br)
    {
        float8 val_f32 = FLOAT8(Oi[qk_idx]);

        val_f32 = val_f32 / l_i;

        bhalf8 val;
        val = f32_to_bf16_8(val_f32);
        HALF8((&(o[q_offset + (Tr_i * Br) * ld_q]))[qk_row * ld_q + qk_col]) = val;

        l_i = row_max_old + logf(l_i);
    }

    if (tx % 16 == 0)
        L[L_offset + Tr_i * Br + qk_row] = l_i;
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
        v_pad = torch::nn::functional::pad(v_pad, torch::nn::functional::PadFuncOptions({0, 0, 0, d_pad_sz}));
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
        v_pad.stride(2),\
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
