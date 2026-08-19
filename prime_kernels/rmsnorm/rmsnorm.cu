// Entry point, plus the bandwidth reference the benchmark measures against.
// Each kernel variant lives in src/kernels/rmsnorm_NN_*.cu.

#include <cassert>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "common.cuh"
#include "rmsnorm.cuh"

// Dispatch follows the measured B200 standings in RESULTS.md: 32 elements per
// thread below C=4096, shared-memory staging at 4096, and the two-pass kernel
// above that, where a thread's slice stops fitting in registers.
void fused_rmsnorm_residual_forward(
        __nv_fp8_e4m3* __restrict__ vals, __nv_fp8_e8m0* __restrict__ scales, float* rrms,
        const nv_bfloat16* branch, const nv_bfloat16* residual,
        const nv_bfloat16* weight,
        float epsilon, int N, int C, cudaStream_t stream)
{
    assert(C % 128 == 0);
    assert(N % 128 == 0);

    if (C <= 2048) {
        launch_rmsnorm_32pt(vals, scales, rrms, branch, residual, weight, epsilon, N, C, stream);
    } else if (C <= 4096) {
        launch_rmsnorm_tma_simple_swz(vals, scales, rrms, branch, residual, weight, epsilon, N, C, stream);
    } else {
        launch_rmsnorm_epi(vals, scales, rrms, branch, residual, weight, epsilon, N, C, stream);
    }
}

template<int BlockSize>
__global__ __launch_bounds__(BlockSize) void rmsnorm_bandwidth_ref_kernel(
        __nv_fp8_e4m3* __restrict__ vals,
        const nv_bfloat16* __restrict__ branch, const nv_bfloat16* __restrict__ residual,
        int N, int C) {
    using bf16x16 = GenericVector<nv_bfloat16, 16>;
    using fp8x16 = GenericVector<__nv_fp8_e4m3, 16>;

    const long idx = blockIdx.x;
    if (idx >= N) return;

    vals += C * idx;
    branch += C * idx;
    residual += C * idx;

    for (int c = threadIdx.x * bf16x16::size; c < C; c += BlockSize * bf16x16::size) {
        const bf16x16 a = bf16x16::load_cs(branch + c);
        const bf16x16 b = bf16x16::load_cs(residual + c);
        fp8x16 q;
        for (int k = 0; k < bf16x16::size; ++k) {
            auto raw = __nv_cvt_bfloat16raw_to_fp8(a[k] + b[k], __NV_SATFINITE, __NV_E4M3);
            q[k] = reinterpret_cast<__nv_fp8_e4m3&>(raw);
        }
        q.store(vals + c);
    }
}

void launch_rmsnorm_bandwidth(
        __nv_fp8_e4m3* vals, __nv_fp8_e8m0* scales, float* rrms,
        const nv_bfloat16* branch, const nv_bfloat16* residual,
        const nv_bfloat16* weight,
        float epsilon, int N, int C, cudaStream_t stream)
{
    constexpr int block_size = 128;
    rmsnorm_bandwidth_ref_kernel<block_size><<<N, block_size, 0, stream>>>(
            vals, branch, residual, N, C);
}

// ---------------------------------------------------------------------------
// Bisection ladder: copy -> copy_2pass -> copy_red -> loop
// ---------------------------------------------------------------------------
//
// `copy` runs at 2.1x the best real kernel, so the access pattern is not what
// limits us -- the work built on top of it is. These two intermediate kernels
// exist to say *which* work, by adding one thing at a time on the way from
// `copy` to `loop`, all at `loop`'s shape (one block per token, 128 threads,
// 16-element vectors). Read the deltas, not the absolute numbers:
//
//   copy       -> copy_2pass   the second read pass over `branch`
//   copy_2pass -> copy_red     the block reduction and its __syncthreads
//   copy_red   -> loop         quantization, e8m0 scales, and the scale stores
//
// Benchmark only; the output is meaningless and none of these are in the test
// table. `copy_red` skips `weight` too, so its delta to `loop` includes that
// one extra (small, cached) read.

// copy + a first pass over `branch`, so the byte traffic matches `loop`
// exactly, but the norm stays thread-local: no cross-thread reduce, no barrier.
