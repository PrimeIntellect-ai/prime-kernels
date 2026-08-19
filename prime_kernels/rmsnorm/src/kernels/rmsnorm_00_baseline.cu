// The naive baseline: one block per token and a strided loop over C.
// Two passes over `branch` -- once to accumulate the sum of squares,
// once to normalize and quantize.

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "common.cuh"

namespace cg = cooperative_groups;

namespace {

template<int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void fused_rmsnorm_residual_forward_bsl_kernel(
        __nv_fp8_e4m3* __restrict__ vals, __nv_fp8_e8m0* __restrict__ scales, float* __restrict__ rrms,
        const nv_bfloat16* __restrict__ branch, const nv_bfloat16* __restrict__ residual,
        const nv_bfloat16* __restrict__ weight, float epsilon,
        int N, int C) {
    using bf16x16 = GenericVector<nv_bfloat16, 16>;
    using fp8x16 = GenericVector<__nv_fp8_e4m3, 16>;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    const cg::thread_block block = cg::this_thread_block();
    const cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(block);
    const cg::thread_block_tile<2> quant_group = cg::tiled_partition<2>(block);
    long idx = blockIdx.x;
    if(idx >= N) return;

    __shared__ __align__(16) float sum_squared_com[NUM_WARPS];

    // adjust pointers to current token
    vals += C * idx;
    branch += C * idx;
    residual += C * idx;

    // rmsnorm is applied to branch; residual is added *after* normalization,
    // i.e. out = rmsnorm(branch) * weight + residual.
    float sum_squared = 0.f;

    for(int c = threadIdx.x * bf16x16::size; c < C; c += BLOCK_SIZE * bf16x16::size) {
        const bf16x16 in1 = bf16x16::load(branch + c);
        for(int k = 0; k < bf16x16::size; ++k) {
            sum_squared = fma(in1[k], in1[k], sum_squared);
        }
    }

    sum_squared = cg::reduce(warp, sum_squared, cg::plus<float>{}) / C;
    if(warp.thread_rank() == 0) {
        sum_squared_com[warp.meta_group_rank()] = sum_squared;
    }
    block.sync();
    sum_squared = cg::reduce(warp, warp.thread_rank() < NUM_WARPS ? sum_squared_com[warp.thread_rank()] : 0.0f, cg::plus<float>{});
    float s = rsqrtf(sum_squared + epsilon);

    for(int c = threadIdx.x * bf16x16::size; c < C; c += BLOCK_SIZE * bf16x16::size) {
        const bf16x16 in1 = bf16x16::load_cs(branch + c);
        const bf16x16 w = bf16x16::load(weight + c);
        const bf16x16 in2 = bf16x16::load_cs(residual + c);
        bf16x16 res_out;
        nv_bfloat16 local_max = __float2bfloat16(0.f);
        for(int k = 0; k < bf16x16::size; ++k) {
            float n = s * __bfloat162float(in1[k]); // normalized output
            // scale in bf16 (not fp32) to match the reference transformer implementation
            float nw = __bfloat162float(__float2bfloat16(n) * w[k]);
            // the residual add happens after the norm, in the input precision.
            nv_bfloat16 res = __float2bfloat16(nw + __bfloat162float(in2[k]));
            local_max = __hmax(local_max, __habs(res));
            res_out[k] = res;
        }

        nv_bfloat16 group_amax = cg::reduce(quant_group, local_max, cg::greater<nv_bfloat16>{});
        constexpr float one_over_448 = 1.f / 448.f;
        float scale_factor = __bfloat162float(group_amax) * one_over_448;
        __nv_fp8_storage_t sf = __nv_cvt_float_to_e8m0(scale_factor, __NV_SATFINITE, cudaRoundPosInf);
        // by construction, E8M0 can't be zero
        // V = 2^(SF - 127) => for 1.0 / V: SF' = 127 - SF; add 127 for bias
        nv_bfloat16 inv_sf = nv_bfloat16(__nv_cvt_e8m0_to_bf16raw(254-sf));

        fp8x16 quantized;
        for(int k = 0; k < bf16x16::size; ++k) {
            auto fp8_raw = __nv_cvt_bfloat16raw_to_fp8(res_out[k] * inv_sf, __NV_SATFINITE, __NV_E4M3);
            quantized[k] = reinterpret_cast<__nv_fp8_e4m3&>(fp8_raw);
        }

        quantized.store(vals + c);
        if( quant_group.thread_rank() == 0 ) {
            scales[get_sf_out_offset(idx, c / 32, C/128)] = reinterpret_cast<__nv_fp8_e8m0&>(sf);
        }
    }

    // cache the rrms for the backward pass later
    if(block.thread_rank() == 0) {
        rrms[idx] = s;
    }
}

void do_launch_rmsnorm_baseline(
        __nv_fp8_e4m3* vals, __nv_fp8_e8m0* scales, float* rrms,
        const nv_bfloat16* branch, const nv_bfloat16* residual,
        const nv_bfloat16* weight,
        float epsilon, int N, int C, cudaStream_t stream, int block_size)
{
    if(block_size == 64) {
        fused_rmsnorm_residual_forward_bsl_kernel<64><<<N, 64, 0, stream>>>(
            vals, scales, rrms, branch, residual, weight, epsilon, N, C);
    } else if(block_size == 128) {
        fused_rmsnorm_residual_forward_bsl_kernel<128><<<N, 128, 0, stream>>>(
            vals, scales, rrms, branch, residual, weight, epsilon, N, C);
    }
}
}

void launch_rmsnorm_loop(
        __nv_fp8_e4m3* vals, __nv_fp8_e8m0* scales, float* rrms,
        const nv_bfloat16* branch, const nv_bfloat16* residual,
        const nv_bfloat16* weight,
        float epsilon, int N, int C, cudaStream_t stream)
{
    int dev, sms, blk_64;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blk_64, fused_rmsnorm_residual_forward_bsl_kernel<64>, 64,0);

    // too little parallelism to fill the GPU with small blocks -> use larger block size with
    if (blk_64 * sms < N && C >= 768) {
        do_launch_rmsnorm_baseline(vals,  scales, rrms, branch, residual, weight, epsilon, N, C, stream, 128);
    } else {
        do_launch_rmsnorm_baseline(vals,  scales, rrms, branch, residual, weight, epsilon, N, C, stream, 64);
    }
}
