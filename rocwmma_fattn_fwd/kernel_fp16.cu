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

#define ComputeType_Out float32_t

#define USE_HALF 1

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


typedef _Float16 fp16_frag __attribute__((ext_vector_type(16)));
typedef float fp32_frag __attribute__((ext_vector_type(8)));
typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef _Float16 half16 __attribute__((ext_vector_type(16)));
#define HALF16(pointer) (reinterpret_cast<half16 *>((void *)&(pointer))[0])
#define HALF8(pointer) (reinterpret_cast<half8 *>((void *)&(pointer))[0])
typedef float float8 __attribute__((ext_vector_type(8)));
typedef float float_v16 __attribute__((ext_vector_type(16)));
#define FLOAT8(pointer) (reinterpret_cast<float8 *>((void *)&(pointer))[0])
#define FLOATV16(pointer) (reinterpret_cast<float_v16 *>((void *)&(pointer))[0])
#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

//================================ Matrix multiplication ===============================
// C = A @ (B^T)
template <int N_WAVES>
__device__ void mul_A_BT(
    ComputeType *__restrict__ A,
    ComputeType *__restrict__ B,
    ComputeType *__restrict__ C,
    int lda, int ldb, int ldc, // ld_qkv, ld_kqv, bc
    int m, int n, int k, // br, bc, d
    const float scale)
{

    fp16_frag fragA[2];
    fp16_frag fragB[2];

    const int wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
    const int lane_id = threadIdx.x % WAVE_SIZE;
    const int wmma_lane = (threadIdx.x % 16);


    for (int wave_off = 0; wave_off < ((m * n) / (ROCWMMA_M * ROCWMMA_N) + N_WAVES - 1) / N_WAVES; wave_off++)
    {
        int wave_xy = __builtin_amdgcn_readfirstlane(wave_id + wave_off * N_WAVES);

        int wave_x = __builtin_amdgcn_readfirstlane(wave_xy % (n / ROCWMMA_N));
        int wave_y = __builtin_amdgcn_readfirstlane(wave_xy / (n / ROCWMMA_N));

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

                fragACC = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fragA[0], fragB[0], fragACC);
                fragACC = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fragA[1], fragB[1], fragACC);
                // fragACC = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fragA[2], fragB[2], fragACC);
                // fragACC = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fragA[3], fragB[3], fragACC);
            }
            fragACC = fragACC * scale;
            __syncthreads();

            for (int ele = 0; ele < 8; ++ele)
            {
                const int r = ele * 2 + (lane_id / 16); // 0 2 4 14 / 1 
                (C + (blk_y * ldc + blk_x))[r * ldc + wmma_lane] = fragACC[ele]; // n
            }
        }
    }
    // asm volatile("s_sleep 0");
}

// C = A@B + C
template <int N_WAVES>
__device__ void mul_add_A_B(
    ComputeType *__restrict__ A,
    ComputeType *__restrict__ B,
    ComputeType_Out *__restrict__ C,
    int lda, int ldb, int ldc,
    const int m, const int n, const int k)
{

    rocwmma::fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, row_major> fragA[2];
    rocwmma::fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType, row_major> fragB[2];
    rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeType_Out> fragC;
    rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t> fragACC;

    const int wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);

    for (int wave_off = 0; wave_off < ((m * n) / (ROCWMMA_M * ROCWMMA_N) + N_WAVES - 1) / N_WAVES; wave_off++)
    {
        int wave_xy = __builtin_amdgcn_readfirstlane(wave_id + wave_off * N_WAVES);

        int wave_x = __builtin_amdgcn_readfirstlane(wave_xy % (n / ROCWMMA_N));
        int wave_y = __builtin_amdgcn_readfirstlane(wave_xy / (n / ROCWMMA_N));

        int blk_x = __builtin_amdgcn_readfirstlane(wave_x * ROCWMMA_N);
        int blk_y = __builtin_amdgcn_readfirstlane(wave_y * ROCWMMA_M);
        if ((blk_x < n) && (blk_y < m))
        {
            rocwmma::fill_fragment(fragACC, (float32_t)0.0);
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

                rocwmma::mma_sync(fragACC, fragA[0], fragB[0], fragACC);
                rocwmma::mma_sync(fragACC, fragA[1], fragB[1], fragACC);
                // rocwmma::mma_sync(fragACC, fragA[2], fragB[2], fragACC);
                // rocwmma::mma_sync(fragACC, fragA[3], fragB[3], fragACC);
            }
            rocwmma::load_matrix_sync(fragC, C + (blk_y * ldc + blk_x), ldc, rocwmma::mem_row_major); //n
            for (int i = 0; i < fragC.num_elements; ++i)
            {
                fragC.x[i] = fragACC.x[i] + fragC.x[i];
            }
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
    rocwmma::fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, float32_t> fragACC;

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
            rocwmma::fill_fragment(fragACC, (float32_t)0.0);
            for (int i = 0; i < wmma_k_end; i += ROCWMMA_K * 2)
            {
                rocwmma::load_matrix_sync(fragA[0], A + (blk_y * lda + i), lda); //k
                rocwmma::load_matrix_sync(fragB[0], B + (i * ldb + blk_x), ldb);  //n
                rocwmma::load_matrix_sync(fragA[1], A + (blk_y * lda + (i + 1 * ROCWMMA_K)), lda);
                rocwmma::load_matrix_sync(fragB[1], B + ((i + 1 * ROCWMMA_K) * ldb + blk_x), ldb);

                rocwmma::mma_sync(fragACC, fragA[0], fragB[0], fragACC);
                rocwmma::mma_sync(fragACC, fragA[1], fragB[1], fragACC);
            }
            rocwmma::load_matrix_sync(fragC, C + (blk_y * ldc + blk_x), ldc, rocwmma::mem_row_major); //n
            for (int i = 0; i < fragC.num_elements; ++i)
            {
                fragC.x[i] = fragACC.x[i] + fragC.x[i];
            }
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

    int ld_qkv = q_stride2; //d;
    if(permute_NH)
    {
        q_offset = blockIdx.x * q_stride0 + blockIdx.y * q_stride2;
        kv_offset = blockIdx.x * kv_stride0 + blockIdx.y * kv_stride2;
        ld_qkv = q_stride1; //h * d;
    }

    const int L_offset = blockIdx.x * L_stride_b + blockIdx.y * L_stride_h;

    const int Tr_i = blockIdx.z;
    if (Tr_i >= Tr)
        return;
    const int ele_y = __builtin_amdgcn_readfirstlane(Tr_i * Br);
    const int yb = __builtin_amdgcn_readfirstlane(ele_y + Br);
    const int tx = threadIdx.x;

    extern __shared__ char sram[];
    ComputeType* __restrict__ Si = reinterpret_cast<ComputeType*>(&sram[0]);       // Br * Bc
    ComputeType_Out* __restrict__ Oi = reinterpret_cast<ComputeType_Out*>(&sram[sizeof(ComputeType_Out) * Br * Bc]); // Br * d
    // ComputeType *__restrict__ Qi = &sram[Br * Bc + Br * d]; // Br * d
    // ComputeType *__restrict__ Vj = &sram[Br * Bc + Br * d]; // Bc * d

    if (tx < Br)
    {
#pragma unroll 4
        for (int i = 0; i < d; i += 16)
        {
            // Load Qi into sram, fill 0 to Oi
            // Qi[tx * d + i] = q[q_offset + Tr_i * Br * d + tx * d + i];
            // FLOAT8(Qi[tx * d + i]) = FLOAT8((&(q[q_offset + Tr_i * Br * d]))[tx * d + i]);
            FLOATV16(Oi[tx * d + i]) = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
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

    __syncthreads();

    ComputeType *__restrict__ Qi = &q[q_offset + (Tr_i * Br) * ld_qkv];
    // ComputeType *__restrict__ Oi = &o[q_offset + Tr_i * Br * d];

    float32_t row_max_old = -FLT_MAX;
    float32_t l_i = 0;

    for (int j = 0; j < Tc; j++)
    {

        ComputeType *__restrict__ Kj = &k[kv_offset + (j * Bc) * ld_qkv];
        ComputeType *__restrict__ Vj = &v[kv_offset + (j * Bc) * ld_qkv];
        int ele_x = j * Bc;
        int xr = ele_x + Bc;
        float32_t row_max_new = -FLT_MAX; // mij
        float32_t row_sum = 0;
        float32_t rowmax_diff_exp = 0; // Sij - mij
        //------------ Sij = Qi @ Kj^T
        if constexpr (!causal)
        {
            mul_A_BT<N_WAVES>(Qi, Kj, Si, ld_qkv, ld_qkv, Bc, Br, Bc, d, scale);
        }
        else
        {
            if (ele_y >= ele_x)
            {
                mul_A_BT<N_WAVES>(Qi, Kj, Si,ld_qkv, ld_qkv, Bc,  Br, Bc, d, scale);
                __syncthreads();
            }
            if ((ele_y < ele_x + Bc - 1) && (tx < Br))
            {
#pragma unroll 32
                for (int i = 0; i < Bc; i++)
                {
                    if (i >= tx + (ele_y - ele_x + 1))
                        Si[tx * Bc + i] = -MAX_NUM;
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
                    Si[tx * Bc + i] = -MAX_NUM;
            }

            if (unlikely((yb > nq) && (tx < Bc)))
            {
#pragma unroll 32
                for (int i = nq - ele_y; i < Br; i++)
                    Si[i * Bc + tx] = -MAX_NUM;
            }
            __syncthreads();
        }
        //------------

        if (tx < Br)
        {
// --------------------- find every row max val in Si[Br * Bc]
            float16_t val16 = row_max_new;
            //float32_t val32 = row_max_new;
#pragma unroll 2
            for (int i = 0; i < Bc; i += 16)
            {
                half16 val = HALF16(Si[(tx * Bc) + i]);
                //float_v16 val_f32 = FLOATV16(Si[(tx * Bc) + i]);

#pragma unroll
                for (int k = 0; k < 16; k++)
                    val16 = max(val16, val[k]); // V_PK_MAX_F16
                    //val32 = max(val32, val_f32[k]); // V_PK_MAX_F16
            }
            row_max_new = val16;
            //row_max_new = val32;

            row_max_new = max(row_max_old, row_max_new);
            rowmax_diff_exp = expf(row_max_old - row_max_new);
            row_max_old = row_max_new;

//--------------------Calc Pi = exp(Si - mi) and rowsum
#pragma unroll 4
            for (int i = 0; i < Bc; i += 16)
            {
                half16 val = HALF16(Si[(tx * Bc) + i]);
                float_v16 val_f32;
#pragma unroll // Load fp16 into VGPRs and convert to FP32
                for (int k = 0; k < 16; k++)
                    val_f32[k] = val[k];
// Si - mi
                val_f32 = val_f32 - row_max_new;
#pragma unroll // exp but using exp2 instead.
                for (int k = 0; k < 16; k++)
                    val_f32[k] = expf(val_f32[k]);

#pragma unroll // calc rowsum
                for (int k = 0; k < 16; k++)
                    row_sum += val_f32[k];

//                half16 val;
#pragma unroll // convert back to fp16
                for (int k = 0; k < 16; k++)
                    val[k] = (_Float16)(val_f32[k]);

               // write back
                HALF16(Si[(tx * Bc) + i]) = val;
            }
            l_i = rowmax_diff_exp * l_i + row_sum;

// --------------------- calc: Oi *= exp2f(row_max_old - row_max_new)
#pragma unroll 4
            for (int i = 0; i < d; i += 16)
            {
                //half16 val = HALF16(Oi[(tx * d) + i]); 
                float_v16 val_f32 = FLOATV16(Oi[(tx * d) + i]); 
                //val = val * rowmax_diff_exp; // V_PK_MUL_F16 
                val_f32 = val_f32 * rowmax_diff_exp; // V_PK_MUL_F16 
                //HALF16(Oi[(tx * d) + i]) = val;
                FLOATV16(Oi[(tx * d) + i]) = val_f32;
            }
// --------------------- 
        }
        __syncthreads();

        if constexpr (!pad_mask)
            mul_add_A_B<N_WAVES>(Si, Vj, Oi,   Bc,ld_qkv,d,   Br, d, Bc);
        else
        {
            if (unlikely(xr > nkv))
            {
                mul_add_A_B_mask_k<N_WAVES>(Si, Vj, Oi,   Bc,ld_qkv,d,  Br, d, Bc, Bc - (xr - nkv));
            }
            else
            {
                mul_add_A_B<N_WAVES>(Si, Vj, Oi,   Bc,ld_qkv,d,    Br, d, Bc);
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
#pragma unroll 4
            for (int i = 0; i < d; i += 16)
            {
                //half16 val = HALF16(Oi[(tx * d) + i]);
                float_v16 val_f32 = FLOATV16(Oi[(tx * d) + i]);
                // float8 val_f32;
// #pragma unroll
                // for (int j = 0; j < 8; j++)
                    // val_f32[j] = val[j];

                //val = val / l_i;
                val_f32 = val_f32 / l_i;

                half16 val;
#pragma unroll
                for (int j = 0; j < 16; j++)
                    val[j] = val_f32[j];
                    
                //HALF8(Oi[(tx * d) + i]) = val;
                HALF16((&(o[q_offset + (Tr_i * Br) * ld_qkv]))[tx * ld_qkv + i]) = val;
            }

// #pragma unroll 4
//         for (int i = 0; i < d; i += 16)
//             // o[q_offset + Tr_i * Br * d + tx * d + i] = Oi[tx * d + i];
//             FLOAT8((&(o[q_offset + Tr_i * Br * d]))[tx * d + i]) = FLOAT8(Oi[tx * d + i]);

        l_i = row_max_old + logf(l_i);
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

std::vector<torch::Tensor> forward_fp16(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    const int Br, const int Bc,
    const bool causal,
    const float scale, const bool permute_NH)
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
        k_pad = torch::nn::functional::pad(k_pad, torch::nn::functional::PadFuncOptions({0, d_pad_sz, 0, 0}));
        v_pad = torch::nn::functional::pad(v_pad, torch::nn::functional::PadFuncOptions({0, d_pad_sz, 0, 0}));
    }
    if (q_pad.stride(-1) != 1)
        q_pad = q_pad.contiguous();

    if (k_pad.stride(-1) != 1)
        k_pad = k_pad.contiguous();

    if (v_pad.stride(-1) != 1)
        v_pad = v_pad.contiguous();

    const int Tr = ceil((float)n / Br);
    const int Tc = ceil((float)n_kv / Bc);

    // auto opt = torch::TensorOptions().dtype(TORCH_DTYPE).device(torch::kCUDA);
    auto O = torch::zeros_like(q_pad);

    auto opt2 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto L = torch::zeros({b, h, n + Nq_pad_sz}, opt2);

    int N_WAVES = 16;
    // if(d + d_pad_sz == 128)
    //     N_WAVES = 32;

    auto blockDim = dim3(WAVE_SIZE * N_WAVES);
    int nblk = b * h * Tr;
    int trPad = 96 - (nblk % 96); // TODO: 96 CU only for gfx1100

    auto gridDim = dim3(b, h, Tr + trPad);

    const int sram_sz =
        Br * Bc * sizeof(ComputeType_Out)               // Si
        + Br * (d + d_pad_sz) * sizeof(ComputeType_Out) // Oi
        // + Br * (d + d_pad_sz) * sizeof(ComputeType) // Qi
        // + Bc * (d + d_pad_sz) * sizeof(ComputeType) // Vj
        ;

#define para_fwd                                      \
        (ComputeType *)q_pad.data_ptr<AT_PTR_TYPE>(), \
        (ComputeType *)k_pad.data_ptr<AT_PTR_TYPE>(), \
        (ComputeType *)v_pad.data_ptr<AT_PTR_TYPE>(), \
        (ComputeType *)O.data_ptr<AT_PTR_TYPE>(),     \
        (ComputeType_Out *)L.data_ptr<ComputeType_Out>(),                 \
        Tr, Tc, Br, Bc,                               \
        n, n_kv,                                      \
        d + d_pad_sz,                                 \
        q_pad.stride(0), q_pad.stride(1),q_pad.stride(2),\
        k_pad.stride(0), k_pad.stride(1),k_pad.stride(2),\
        L.stride(0), L.stride(1),                     \
        scale, permute_NH

    cudaError_t err = cudaGetLastError();

    if(N_WAVES == 32)
    {
        if (!pad_mask && !causal)
            fwd_kernel<false, false,32><<<gridDim, blockDim, sram_sz>>>(para_fwd);
        else if (pad_mask && causal)
            fwd_kernel<true, true,32><<<gridDim, blockDim, sram_sz>>>(para_fwd);
        else if (!pad_mask && causal)
            fwd_kernel<false, true,32><<<gridDim, blockDim, sram_sz>>>(para_fwd);
        else if (pad_mask && !causal)
            fwd_kernel<true, false,32><<<gridDim, blockDim, sram_sz>>>(para_fwd);
    }else if(N_WAVES == 16)
    {
        if (!pad_mask && !causal)
            fwd_kernel<false, false,16><<<gridDim, blockDim, sram_sz>>>(para_fwd);
        else if (pad_mask && causal)
            fwd_kernel<true, true,16><<<gridDim, blockDim, sram_sz>>>(para_fwd);
        else if (!pad_mask && causal)
            fwd_kernel<false, true,16><<<gridDim, blockDim, sram_sz>>>(para_fwd);
        else if (pad_mask && !causal)
            fwd_kernel<true, false,16><<<gridDim, blockDim, sram_sz>>>(para_fwd);
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

std::vector<torch::Tensor> backward_fp16(
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
