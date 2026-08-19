// Small-C variant: one 16-element vector per thread and no loop over C
// the number of channels is determined by the number of threads started.
// this allows us to keep everything in registers.

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cassert>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "common.cuh"

namespace cg = cooperative_groups;

// kernel for small channel dimension: no loop over C, each thread just processes one vector (16 elements)
template<int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void fused_rmsnorm_residual_forward_fixed_c_kernel(
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
    int tc =  threadIdx.x * bf16x16::size;
    vals += C * idx + tc;
    branch += C * idx + tc;
    residual += C * idx + tc;

    // rmsnorm is applied to branch; residual is added *after* normalization,
    // i.e. out = rmsnorm(branch) * weight + residual.
    float sum_squared = 0.f;

    bf16x16 in1;
    bf16x16 in2;
    bf16x16 w;
    if(tc < C) {
        in1 = bf16x16::load_cs(branch);
        for(int k = 0; k < bf16x16::size; ++k) {
            sum_squared = fma(in1[k], in1[k], sum_squared);
        }
        w = bf16x16::load(weight + tc);
        in2 = bf16x16::load_cs(residual);
    }

    sum_squared = cg::reduce(warp, sum_squared, cg::plus<float>{}) * inv_c;
    if(warp.thread_rank() == 0) {
        sum_squared_com[warp.meta_group_rank()] = sum_squared;
    }
    block.sync();

    sum_squared = cg::reduce(warp, warp.thread_rank() < NUM_WARPS ? sum_squared_com[warp.thread_rank()] : 0.0f, cg::plus<float>{});

    float s = rsqrtf(sum_squared + epsilon);

    if(tc < C) {
        bf16x16 res_out;
        nv_bfloat162 local_maxes = __float22bfloat162_rn(make_float2(0.f, 0.f));;
        for(int k = 0; k < bf16x16::size; k += 2) {
            float2 sbw = {
                .x = s * fmul(in1[k + 0], w[k + 0]),
                .y = s * fmul(in1[k + 1], w[k + 1]),
            };
            __nv_bfloat162 res_bf16 = make_bfloat162(in2[k], in2[k+1]);
            // the residual add happens after the norm, in the input precision.
            nv_bfloat162 out = __float22bfloat162_rn(__fadd2_rn(sbw,  __bfloat1622float2(res_bf16)));
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

        quantized.store(vals);
        if( quant_group.thread_rank() == 0 ) {
            scales[get_sf_out_offset(idx, tc / 32, C/128)] = reinterpret_cast<__nv_fp8_e8m0&>(sf);
        }
    }

    // cache the rrms for the backward pass later
    if(block.thread_rank() == 0) {
        rrms[idx] = s;
    }
}

template<int BlockSize>
void do_launch_rmsnorm_fixed_c(
        __nv_fp8_e4m3* vals, __nv_fp8_e8m0* scales, float* rrms,
        const nv_bfloat16* branch, const nv_bfloat16* residual,
        const nv_bfloat16* weight,
        float epsilon, int N, int C, cudaStream_t stream)
{
    fused_rmsnorm_residual_forward_fixed_c_kernel<BlockSize><<<N, BlockSize, 0, stream>>>(
            vals, scales, rrms, branch, residual, weight, epsilon, N, C, 1.f/C);
}
void launch_rmsnorm_fixed_c(
        __nv_fp8_e4m3* vals, __nv_fp8_e8m0* scales, float* rrms,
        const nv_bfloat16* branch, const nv_bfloat16* residual,
        const nv_bfloat16* weight,
        float epsilon, int N, int C, cudaStream_t stream)
{
    if (C <= 1024) {
        do_launch_rmsnorm_fixed_c<64>(vals, scales, rrms, branch, residual, weight, epsilon, N, C, stream);
    } else if (C <= 1536) {
        do_launch_rmsnorm_fixed_c<96>(vals, scales, rrms, branch, residual, weight, epsilon, N, C, stream);
    } else if (C <= 2048) {
        do_launch_rmsnorm_fixed_c<128>(vals, scales, rrms, branch, residual, weight, epsilon, N, C, stream);
    } else if (C <= 2560) {
        do_launch_rmsnorm_fixed_c<160>(vals, scales, rrms, branch, residual, weight, epsilon, N, C, stream);
    } else if (C <= 3072) {
        do_launch_rmsnorm_fixed_c<192>(vals, scales, rrms, branch, residual, weight, epsilon, N, C, stream);
    } else if (C <= 3584) {
        do_launch_rmsnorm_fixed_c<224>(vals, scales, rrms, branch, residual, weight, epsilon, N, C, stream);
    } else if (C <= 4096) {
        do_launch_rmsnorm_fixed_c<256>(vals, scales, rrms, branch, residual, weight, epsilon, N, C, stream);
    } else if (C <= 5120) {
        do_launch_rmsnorm_fixed_c<320>(vals, scales, rrms, branch, residual, weight, epsilon, N, C, stream);
    } else if (C <= 6144) {
        do_launch_rmsnorm_fixed_c<384>(vals, scales, rrms, branch, residual, weight, epsilon, N, C, stream);
    } else if (C <= 7168) {
        do_launch_rmsnorm_fixed_c<448>(vals, scales, rrms, branch, residual, weight, epsilon, N, C, stream);
    } else if (C <= 8192) {
        do_launch_rmsnorm_fixed_c<512>(vals, scales, rrms, branch, residual, weight, epsilon, N, C, stream);
    } else {
        std::exit(EXIT_FAILURE);
    }
}
