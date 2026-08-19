// Like the baseline, but with micro-optimizations that improve the quantization epilogue.

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "common.cuh"

namespace cg = cooperative_groups;

template<int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void fused_rmsnorm_residual_forward_epi_kernel(
        __nv_fp8_e4m3* __restrict__ vals, __nv_fp8_e8m0* __restrict__ scales, float* __restrict__ rrms,
        const nv_bfloat16* __restrict__ branch, const nv_bfloat16* __restrict__ residual,
        const nv_bfloat16* __restrict__ weight, float epsilon,
        int N, int C, float inv_c) {
    using bf16x16 = GenericVector<nv_bfloat16, 16>;
    using fp8x16 = GenericVector<__nv_fp8_e4m3, 16>;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    const cg::thread_block block = cg::this_thread_block();
    const cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(block);
    const cg::thread_block_tile<2> quant_group = cg::tiled_partition<2>(block);
    long idx = blockIdx.x;
    if(idx >= N) return;

    __shared__ __align__(16) float sum_squared_com[BLOCK_SIZE / WARP_SIZE];

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

    sum_squared = cg::reduce(warp, sum_squared, cg::plus<float>{}) * inv_c;
    if(warp.thread_rank() == 0) {
        sum_squared_com[warp.meta_group_rank()] = sum_squared;
    }
    static_assert(NUM_WARPS == 4 || NUM_WARPS == 2, "reduction tail handles 2 or 4 warps");
    block.sync();
    if constexpr (NUM_WARPS == 4) {
        float4 all_ss = *reinterpret_cast<float4*>(sum_squared_com);
        sum_squared = (all_ss.x + all_ss.y) + (all_ss.z + all_ss.w);
    } else if constexpr (NUM_WARPS == 2) {
        float2 all_ss = *reinterpret_cast<float2*>(sum_squared_com);
        sum_squared = all_ss.x + all_ss.y;
    }

    float s = rsqrtf(sum_squared + epsilon);

    for(int c = threadIdx.x * bf16x16::size; c < C; c += BLOCK_SIZE * bf16x16::size) {
        const bf16x16 in1 = bf16x16::load_cs(branch + c);
        const bf16x16 w = bf16x16::load(weight + c);
        const bf16x16 in2 = bf16x16::load_cs(residual + c);
        bf16x16 res_out;
        nv_bfloat162 local_maxes = __float22bfloat162_rn(make_float2(0.f, 0.f));
        for(int k = 0; k < bf16x16::size; k += 2) {
            // scale in fp32; the bf16 product is exact, so this is one fused op
            // and one multiply
            float2 sbw = {
                .x = s * fmul(in1[k + 0], w[k + 0]),
                .y = s * fmul(in1[k + 1], w[k + 1]),
            };
            __nv_bfloat162 res_bf16 = make_bfloat162(in2[k], in2[k+1]);
            // the residual add happens after the norm, in the input precision.
            nv_bfloat162 out = __float22bfloat162_rn(__fadd2_rn(sbw, __bfloat1622float2(res_bf16)));
            local_maxes = __hmax2(local_maxes, __habs2(out));
            res_out[k+0] = out.x;
            res_out[k+1] = out.y;
        }

        nv_bfloat16 local_max = __hmax(local_maxes.x, local_maxes.y);
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

void do_launch_rmsnorm_epi(
        __nv_fp8_e4m3* vals, __nv_fp8_e8m0* scales, float* rrms,
        const nv_bfloat16* branch, const nv_bfloat16* residual,
        const nv_bfloat16* weight,
        float epsilon, int N, int C, cudaStream_t stream, int block_size)
{
    if(block_size == 64) {
        fused_rmsnorm_residual_forward_epi_kernel<64><<<N, 64, 0, stream>>>(
            vals, scales, rrms, branch, residual, weight, epsilon, N, C, 1.f/C);
    } else if(block_size == 128) {
        fused_rmsnorm_residual_forward_epi_kernel<128><<<N, 128, 0, stream>>>(
            vals, scales, rrms, branch, residual, weight, epsilon, N, C, 1.f/C);
    }
}


void launch_rmsnorm_epi(
        __nv_fp8_e4m3* vals, __nv_fp8_e8m0* scales, float* rrms,
        const nv_bfloat16* branch, const nv_bfloat16* residual,
        const nv_bfloat16* weight,
        float epsilon, int N, int C, cudaStream_t stream)
{
    int dev, sms, blk_64;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blk_64, fused_rmsnorm_residual_forward_epi_kernel<64>, 64,0);

    // too little parallelism to fill the GPU with small blocks -> use larger block size with
    if (blk_64 * sms < N && C >= 768) {
        do_launch_rmsnorm_epi(vals,  scales, rrms, branch, residual, weight, epsilon, N, C, stream, 128);
    } else {
        do_launch_rmsnorm_epi(vals,  scales, rrms, branch, residual, weight, epsilon, N, C, stream, 64);
    }
}
