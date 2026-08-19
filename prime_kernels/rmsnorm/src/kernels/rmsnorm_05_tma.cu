// Rows staged in shared memory by TMA instead of held in registers. Every warp
// runs the whole fetch -> reduce -> quantize sequence, synchronized by
// __syncthreads(); the copy for the next token is issued a token ahead.
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include "common.cuh"
#include "rmsnorm.cuh"

// Swizzled so a thread can own 16 consecutive elements without the 2-way bank
// conflict that layout otherwise gives its two LDS.128s.
__device__ __forceinline__ constexpr int tma_swz(int e) {
    return e ^ (((e >> 6) & 7) << 3);
}

using cuTensorMapEncodeTiled_t = CUresult (*)(
        CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
        const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
        CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static cuTensorMapEncodeTiled_t tma_encode_fn() {
    static cuTensorMapEncodeTiled_t fn = nullptr;
    if (fn == nullptr) {
        void* p = nullptr;
        cudaDriverEntryPointQueryResult qr;
        cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000,
                                         cudaEnableDefault, &qr);
        fn = reinterpret_cast<cuTensorMapEncodeTiled_t>(p);
    }
    return fn;
}

// [., 64] view: `rows` is N*C/64 for the token buffers, C/64 for the weight
// row; `boxRows` is C/64 either way, so a tile is one token's row.
static CUtensorMap tma_make_map(const nv_bfloat16* ptr, long rows, int boxRows) {
    CUtensorMap map{};
    const cuuint64_t globalDim[2] = {64, static_cast<cuuint64_t>(rows)};
    const cuuint64_t globalStrides[1] = {128};   // bytes; one 64-element row
    const cuuint32_t boxDim[2] = {64, static_cast<cuuint32_t>(boxRows)};
    const cuuint32_t elementStrides[2] = {1, 1};

    // 64 * 2 = 128 B inner box, the SWIZZLE_128B limit; boxDim[1] <= 256 caps this
    // path at C <= 16384
    tma_encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
                    const_cast<nv_bfloat16*>(ptr), globalDim, globalStrides,
                    boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                    CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return map;
}

bool rmsnorm_tma_simple_swizzle_supported(int C) {
    // C % 512 keeps every stage base on a 1024 B swizzle atom; C/64 <= 256 is the
    // TMA box limit
    return C % 512 == 0 && C / 64 <= 256 && tma_encode_fn() != nullptr;
}

using block_barrier = cuda::barrier<cuda::thread_scope_block>;
using cuda::device::barrier_native_handle;
namespace ptx = cuda::ptx;

// Flat byte copy: `bytes` must be a multiple of 16 and both addresses aligned.
__device__ __forceinline__ void bulk_copy_g2s(void* smem, const void* gmem, uint32_t bytes, block_barrier& bar) {
    ptx::cp_async_bulk(ptx::space_shared, ptx::space_global, smem, gmem, bytes,
                       barrier_native_handle(bar));
}

__device__ __forceinline__ void arrive_with_tx(block_barrier& bar, int bytes) {
    (void)cuda::device::barrier_arrive_tx(bar, 1, bytes);
}

// ElemsPerThread is the width a thread owns; 16 halves the per-thread work of
// 8. Smem loads are always 16 B: vec.cuh's 32 B path is global-only.
template<int BlockSize, int ElemsPerThread, int Depth, bool Swizzle>
__global__ __launch_bounds__(BlockSize)
void fused_rmsnorm_residual_forward_tma_simple_kernel(
        __nv_fp8_e4m3* __restrict__ vals, __nv_fp8_e8m0* __restrict__ scales, float* __restrict__ rrms,
        const nv_bfloat16* __restrict__ branch, const nv_bfloat16* __restrict__ residual,
        const nv_bfloat16* __restrict__ weight, float epsilon,
        int N, int C, float inv_c,
        const __grid_constant__ CUtensorMap branch_map,
        const __grid_constant__ CUtensorMap res_map,
        const __grid_constant__ CUtensorMap wgt_map) {
    using bf16x8 = GenericVector<nv_bfloat16, 8>;
    // register container only -- never load()ed, since a 32 B GenericVector uses
    // the global-only ld.global.v8.u32 path and faults on shared memory
    using fp8_vec = GenericVector<__nv_fp8_storage_t, ElemsPerThread>;
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = BlockSize / WARP_SIZE;
    constexpr int EPT = ElemsPerThread;
    constexpr int VECS = EPT / 8;        // 16 B smem loads per thread per pass
    constexpr int LANES_PER_GROUP = 32 / EPT;   // threads sharing one e8m0 scale

    static_assert(Depth >= 2);
    static_assert(EPT == 8 || EPT == 16);

    __shared__ block_barrier tok_bar[Depth];
    __shared__ block_barrier wgt_bar;
    __shared__ __align__(16) float warp_sums[NUM_WARPS];

    // 1024 B: the swizzle is keyed on the absolute address within its 8-row
    // atom, so a stage starting mid-atom gets a rotated permutation. Stage bases
    // are C*2 apart, hence the C % 512 == 0 requirement on the swizzle path.
    extern __shared__ __align__(1024) char tma_simple_smem[];
    nv_bfloat16* branch_smem = reinterpret_cast<nv_bfloat16*>(tma_simple_smem);
    nv_bfloat16* res_smem = branch_smem + Depth * static_cast<size_t>(C);
    nv_bfloat16* wgt_smem = res_smem + Depth * static_cast<size_t>(C);

    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int bytes_per_row = C * sizeof(nv_bfloat16);

    if (tid == 0) {
        for (int d = 0; d < Depth; ++d) {
            init(&tok_bar[d], 1);
        }
        init(&wgt_bar, 1);
    }
    __syncthreads();

    // both inputs for one token go through a single barrier: the transaction
    // counts add up, so one arrive and one wait cover the pair
    const int tile_rows = C / 64;   // 64 bf16 = 128 B per row of the tiled view
    auto fetch_token = [&](long row, int stage) {
        if constexpr (Swizzle) {
            const int32_t coords[2] = {0, static_cast<int32_t>(row * tile_rows)};
            ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global,
                                      branch_smem + stage * C, &branch_map, coords,
                                      barrier_native_handle(tok_bar[stage]));
            ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global,
                                      res_smem + stage * C, &res_map, coords,
                                      barrier_native_handle(tok_bar[stage]));
        } else {
            bulk_copy_g2s(branch_smem + stage * C, branch + static_cast<long>(C) * row,
                          bytes_per_row, tok_bar[stage]);
            bulk_copy_g2s(res_smem + stage * C, residual + static_cast<long>(C) * row,
                          bytes_per_row, tok_bar[stage]);
        }
        arrive_with_tx(tok_bar[stage], 2 * bytes_per_row);
    };

    // Prologue: the weight row (loaded once for the whole block) and the first
    // Depth-1 tokens. From here the loop is always Depth-1 tokens ahead.
    if (tid == 0) {
        if constexpr (Swizzle) {
            const int32_t coords[2] = {0, 0};
            ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global,
                                      wgt_smem, &wgt_map, coords,
                                      barrier_native_handle(wgt_bar));
        } else {
            bulk_copy_g2s(wgt_smem, weight, bytes_per_row, wgt_bar);
        }
        arrive_with_tx(wgt_bar, bytes_per_row);
        for (int d = 0; d < Depth - 1; ++d) {
            const long row = blockIdx.x + static_cast<long>(d) * gridDim.x;
            if (row < N) {
                fetch_token(row, d);
            }
        }
    }
    wgt_bar.wait_parity(0);

    int stage = 0;
    int phase = 0;
    for (long idx = blockIdx.x; idx < N; idx += gridDim.x) {
        // the token Depth-1 iterations out, landing in the stage this block finished
        // with one iteration ago
        const long ahead = idx + static_cast<long>(Depth - 1) * gridDim.x;
        const int fill = (stage + Depth - 1) % Depth;

        // prefetch before the wait, so Depth-1 copies stay in flight; safe because the
        // barrier at the bottom of the loop is downstream of every read of the stage
        if (tid == 0 && ahead < N) {
            fetch_token(ahead, fill);
        }
        tok_bar[stage].wait_parity(phase);

        const nv_bfloat16* branch_row = branch_smem + stage * C;
        const nv_bfloat16* res_row = res_smem + stage * C;

        // logical element offset -> physical offset in the staged tile
        auto at = [](int e) { return Swizzle ? tma_swz(e) : e; };

        float sum_squared = 0.f;
        for (int c = tid * EPT; c < C; c += BlockSize * EPT) {
            for (int u = 0; u < VECS; ++u) {
                const bf16x8 v = bf16x8::load(branch_row + at(c + u * 8));
                for (int k = 0; k < bf16x8::size; ++k) {
                    sum_squared = fma(v[k], v[k], sum_squared);
                }
            }
        }

        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            sum_squared += __shfl_xor_sync(0xffffffff, sum_squared, off);
        }
        if (lane == 0) {
            warp_sums[warp_id] = sum_squared;
        }
        __syncthreads();

        // shuffles, not a loop over warp_sums; the xor steps span aligned groups of
        // NUM_WARPS lanes, so each group needs its own copy of the partials
        static_assert(NUM_WARPS <= WARP_SIZE && (NUM_WARPS & (NUM_WARPS - 1)) == 0);
        float norm = warp_sums[lane & (NUM_WARPS - 1)];
        for (int off = NUM_WARPS / 2; off > 0; off >>= 1) {
            norm += __shfl_xor_sync(0xffffffff, norm, off);
        }
        const float s = rsqrtf(norm * inv_c + epsilon);

        for (int c = tid * EPT; c < C; c += BlockSize * EPT) {
            // paired throughout, bit-identical to the scalar form
            nv_bfloat162 res_out[EPT / 2];
            nv_bfloat162 local_maxes = __float22bfloat162_rn(make_float2(0.f, 0.f));
            for (int u = 0; u < VECS; ++u) {
                const int e = at(c + u * 8);
                const bf16x8 branch_v = bf16x8::load(branch_row + e);
                const bf16x8 weight_v = bf16x8::load(wgt_smem + e);
                auto wt = [&](int k) { return weight_v[k]; };
                const bf16x8 res_v = bf16x8::load(res_row + e);
                for (int k = 0; k < bf16x8::size; k += 2) {
                    // scale in fp32; the bf16 product is exact, so this is one
                    // fused op and one multiply
                    const float2 sbw = {
                        .x = s * fmul(branch_v[k + 0], wt(k + 0)),
                        .y = s * fmul(branch_v[k + 1], wt(k + 1)),
                    };
                    const nv_bfloat162 res_v2 = make_bfloat162(res_v[k], res_v[k + 1]);
                    // the residual add happens after the norm, in the input precision.
                    // __fadd2_rn, not '+': the multiplies above must not contract into
                    // FFMAs with this add, or the rounding changes
                    const nv_bfloat162 out =
                            __float22bfloat162_rn(__fadd2_rn(sbw, __bfloat1622float2(res_v2)));
                    local_maxes = __hmax2(local_maxes, __habs2(out));
                    res_out[(u * 8 + k) / 2] = out;
                }
            }
            const nv_bfloat16 local_max = __hmax(local_maxes.x, local_maxes.y);

            // one scale per 32 elements, so LANES_PER_GROUP threads share it;
            // __reduce_max_sync lowers to CREDUX + ENDCOLLECTIVE, which is more
            // instructions and a stall site
            const unsigned my_mask =
                    ((1u << LANES_PER_GROUP) - 1u) << ((lane / LANES_PER_GROUP) * LANES_PER_GROUP);
            nv_bfloat16 group_amax = local_max;
            for (int off = 1; off < LANES_PER_GROUP; off <<= 1) {
                group_amax = __hmax(group_amax, __ushort_as_bfloat16(
                        __shfl_xor_sync(my_mask, __bfloat16_as_ushort(group_amax), off)));
            }

            constexpr float one_over_448 = 1.f / 448.f;
            float scale_factor = __bfloat162float(group_amax) * one_over_448;
            __nv_fp8_storage_t sf = __nv_cvt_float_to_e8m0(scale_factor, __NV_SATFINITE, cudaRoundPosInf);
            // by construction, E8M0 can't be zero
            // V = 2^(SF - 127) => for 1.0 / V: SF' = 127 - SF; add 127 for bias
            const nv_bfloat16 inv_sf_s = nv_bfloat16(__nv_cvt_e8m0_to_bf16raw(254 - sf));
            const nv_bfloat162 inv_sf = make_bfloat162(inv_sf_s, inv_sf_s);

            fp8_vec quantized;
            for (int k = 0; k < EPT / 2; ++k) {
                const nv_bfloat162 scaled = __hmul2(res_out[k], inv_sf);
                reinterpret_cast<__nv_fp8x2_storage_t*>(&quantized)[k] =
                        __nv_cvt_bfloat16raw2_to_fp8x2(static_cast<__nv_bfloat162_raw>(scaled),
                                                       __NV_SATFINITE, __NV_E4M3);
            }

            quantized.store(reinterpret_cast<__nv_fp8_storage_t*>(vals) + static_cast<long>(C) * idx + c);
            if (lane % LANES_PER_GROUP == 0) {
                scales[get_sf_out_offset(idx, c / 32, C / 128)] = reinterpret_cast<__nv_fp8_e8m0&>(sf);
            }
        }

        // cache the rrms for the backward pass later
        if (tid == 0) {
            rrms[idx] = s;
        }

        // each barrier completes once per Depth iterations, so the parity flips
        // exactly when the stage index wraps
        if (++stage == Depth) {
            stage = 0;
            phase ^= 1;
        }
        // frees this stage for the next iteration's copies, and warp_sums for its
        // reduction
        __syncthreads();
    }
}

// smem layout: branch[Depth][C] | residual[Depth][C] | weight[C]
size_t rmsnorm_tma_simple_smem_bytes(int C, int depth) {
    return (2 * static_cast<size_t>(depth) + 1) * static_cast<size_t>(C) * sizeof(nv_bfloat16);
}

// 10 bytes per column at the shallowest depth the dispatch falls back to, so
// the opt-in smem bounds C: ~10112 on sm_120, ~23168 on B200.
int rmsnorm_tma_simple_max_c() { return max_c_for_smem(10); }

static int tma_simple_max_smem() {
    static int max_smem = 0;
    if (max_smem == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    }
    return max_smem;
}

static int tma_simple_num_sms() {
    static int num_sms = 0;
    if (num_sms == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
    }
    return num_sms;
}

// raise the opt-in limit before the occupancy query, not just before the
// launch, or the query answers for the default 48 KB
template<typename Kernel>
static void tma_simple_raise_smem_limit(Kernel kernel, size_t smem) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem));
}

// encoding is a host call, so cache the maps on the (pointer, C) they were
// built for rather than rebuilding per launch
struct TmaMaps {
    CUtensorMap branch{}, residual{}, weight{};
    const void* branch_ptr = nullptr;
    const void* residual_ptr = nullptr;
    const void* weight_ptr = nullptr;
    long n = 0;
    int c = 0;
};

static const TmaMaps& tma_simple_maps(
        const nv_bfloat16* branch, const nv_bfloat16* residual,
        const nv_bfloat16* weight, int N, int C) {
    static TmaMaps cache;
    if (cache.branch_ptr != branch || cache.residual_ptr != residual ||
        cache.weight_ptr != weight || cache.n != N || cache.c != C) {
        const long rows = static_cast<long>(N) * C / 64;
        cache.branch = tma_make_map(branch, rows, C / 64);
        cache.residual = tma_make_map(residual, rows, C / 64);
        cache.weight = tma_make_map(weight, C / 64, C / 64);
        cache.branch_ptr = branch;
        cache.residual_ptr = residual;
        cache.weight_ptr = weight;
        cache.n = N;
        cache.c = C;
    }
    return cache;
}

template<int BlockSize, int ElemsPerThread, int Depth, bool Swizzle>
static void launch_tma_simple_sized(
        __nv_fp8_e4m3* vals, __nv_fp8_e8m0* scales, float* rrms,
        const nv_bfloat16* branch, const nv_bfloat16* residual,
        const nv_bfloat16* weight,
        float epsilon, int N, int C, cudaStream_t stream)
{
    auto kernel = fused_rmsnorm_residual_forward_tma_simple_kernel<
            BlockSize, ElemsPerThread, Depth, Swizzle>;
    const size_t smem = rmsnorm_tma_simple_smem_bytes(C, Depth);
    tma_simple_raise_smem_limit(kernel, smem);

    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, BlockSize, smem);

    int blocks = min(blocks_per_sm * tma_simple_num_sms(), N);

    if constexpr (Swizzle) {
        const TmaMaps& m = tma_simple_maps(branch, residual, weight, N, C);
        kernel<<<blocks, BlockSize, smem, stream>>>(
                vals, scales, rrms, branch, residual, weight, epsilon, N, C,
                1.f / float(C), m.branch, m.residual, m.weight);
    } else {
        const CUtensorMap unused{};
        kernel<<<blocks, BlockSize, smem, stream>>>(
                vals, scales, rrms, branch, residual, weight, epsilon, N, C,
                1.f / float(C), unused, unused, unused);
    }
}

// blocks of this shape that fit on an SM; 0 if the staging does not fit
template<int BlockSize, int ElemsPerThread, int Depth, bool Swizzle>
static int tma_simple_blocks_per_sm(int C) {
    const size_t smem = rmsnorm_tma_simple_smem_bytes(C, Depth);
    if (smem > static_cast<size_t>(tma_simple_max_smem())) {
        return 0;
    }
    auto kernel = fused_rmsnorm_residual_forward_tma_simple_kernel<
            BlockSize, ElemsPerThread, Depth, Swizzle>;
    tma_simple_raise_smem_limit(kernel, smem);

    int blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, BlockSize, smem);
    return blocks;
}

// Depth 3 keeps two copies outstanding instead of one, at 7*C of smem against
// 5*C. Take it only when it costs no blocks-per-SM.
template<int BlockSize, int ElemsPerThread, bool Swizzle>
static void launch_tma_simple_shape(
        __nv_fp8_e4m3* vals, __nv_fp8_e8m0* scales, float* rrms,
        const nv_bfloat16* branch, const nv_bfloat16* residual,
        const nv_bfloat16* weight,
        float epsilon, int N, int C, cudaStream_t stream)
{
    const int deep = tma_simple_blocks_per_sm<BlockSize, ElemsPerThread, 3, Swizzle>(C);
    const int shallow = tma_simple_blocks_per_sm<BlockSize, ElemsPerThread, 2, Swizzle>(C);

    if (deep > 0 && deep >= shallow) {
        launch_tma_simple_sized<BlockSize, ElemsPerThread, 3, Swizzle>(
                vals, scales, rrms, branch, residual, weight, epsilon, N, C, stream);
    } else {
        launch_tma_simple_sized<BlockSize, ElemsPerThread, 2, Swizzle>(
                vals, scales, rrms, branch, residual, weight, epsilon, N, C, stream);
    }
}

template<bool Swizzle>
static void launch_tma_simple_depth(
        __nv_fp8_e4m3* vals, __nv_fp8_e8m0* scales, float* rrms,
        const nv_bfloat16* branch, const nv_bfloat16* residual,
        const nv_bfloat16* weight,
        float epsilon, int N, int C, cudaStream_t stream)
{
    if (C < 128 * 16) {
        launch_tma_simple_shape<128, 8, Swizzle>(vals, scales, rrms, branch,
                                                   residual, weight, epsilon, N, C, stream);
    } else if (C < 256 * 16) {
        launch_tma_simple_shape<128, 16, Swizzle>(vals, scales, rrms, branch,
                                                    residual, weight, epsilon, N, C, stream);
    } else if (C < 512 * 16) {
        launch_tma_simple_shape<256, 16, Swizzle>(vals, scales, rrms, branch,
                                                    residual, weight, epsilon, N, C, stream);
    } else {
        launch_tma_simple_shape<512, 16, Swizzle>(vals, scales, rrms, branch,
                                                    residual, weight, epsilon, N, C, stream);
    }
}

template<bool Swizzle>
static void launch_tma_simple_impl(
        __nv_fp8_e4m3* vals, __nv_fp8_e8m0* scales, float* rrms,
        const nv_bfloat16* branch, const nv_bfloat16* residual,
        const nv_bfloat16* weight,
        float epsilon, int N, int C, cudaStream_t stream)
{
    launch_tma_simple_depth<Swizzle>(vals, scales, rrms, branch, residual,
                                                weight, epsilon, N, C, stream);
}

void launch_rmsnorm_tma_simple(
        __nv_fp8_e4m3* vals, __nv_fp8_e8m0* scales, float* rrms,
        const nv_bfloat16* branch, const nv_bfloat16* residual,
        const nv_bfloat16* weight,
        float epsilon, int N, int C, cudaStream_t stream)
{
    launch_tma_simple_impl<false>(vals, scales, rrms, branch, residual, weight,
                                         epsilon, N, C, stream);
}

void launch_rmsnorm_tma_simple_swz(
        __nv_fp8_e4m3* vals, __nv_fp8_e8m0* scales, float* rrms,
        const nv_bfloat16* branch, const nv_bfloat16* residual,
        const nv_bfloat16* weight,
        float epsilon, int N, int C, cudaStream_t stream)
{
    // C/64 > 256 exceeds the TMA box limit; fall back rather than fail
    if (rmsnorm_tma_simple_swizzle_supported(C)) {
        launch_tma_simple_impl<true>(vals, scales, rrms, branch, residual, weight,
                                            epsilon, N, C, stream);
    } else {
        launch_tma_simple_impl<false>(vals, scales, rrms, branch, residual, weight,
                                             epsilon, N, C, stream);
    }
}


