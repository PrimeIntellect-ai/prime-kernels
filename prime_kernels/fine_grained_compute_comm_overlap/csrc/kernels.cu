#include "kernels.cuh"

#include <cstdio>
#include <cuda_bf16.h>
#include <mma.h>
#include "tcgen05_prelude.cuh"
#include "tiled_pipeline.cuh"
#include "transport.cuh"
#include "sched.cuh"

namespace pi {
    namespace {
        __global__ void scatter_tiles_kernel(
            const uint8_t *__restrict__ src,
            const int64_t *__restrict__ hidden_peer_ptrs,
            const int64_t *__restrict__ flag_peer_ptrs,
            const int32_t *__restrict__ tile_peer_rank,
            const int32_t *__restrict__ tile_local_row_start,
            const int32_t *__restrict__ tile_peer_row_start,
            const int32_t *__restrict__ tile_valid_rows,
            const int32_t *__restrict__ tile_flag_index,
            int64_t n_tiles,
            int64_t row_bytes
        ) {
            for (int64_t tile=blockIdx.x; tile < n_tiles; tile += gridDim.x) {
                int32_t dest_rank = tile_peer_rank[tile];
                if (dest_rank < 0) continue;
                int64_t local_start = tile_local_row_start[tile];
                int64_t peer_start = tile_peer_row_start[tile];
                int64_t valid = tile_valid_rows[tile];
                int32_t flag_idx = tile_flag_index[tile];

                auto *dest_base = reinterpret_cast<uint8_t *>(hidden_peer_ptrs[dest_rank]);
                const uint4 *src_vec = reinterpret_cast<const uint4 *>(src + local_start*row_bytes);
                uint4 *dst_vec = reinterpret_cast<uint4 *>(dest_base + peer_start*row_bytes);
                int64_t n_vec = (valid*row_bytes)>>4;
                for (int64_t i=threadIdx.x; i < n_vec; i += blockDim.x)
                    dst_vec[i] = src_vec[i];
                __syncthreads();
                if (threadIdx.x == 0) {
                    __threadfence_system();
                    auto *flag_base = reinterpret_cast<int32_t *>(flag_peer_ptrs[dest_rank]);
                    atomicExch_system(&flag_base[flag_idx], 1);
                }
            }
        }

        __global__ void wait_tiles_kernel(
            int32_t *__restrict__ local_flag,
            const int32_t *__restrict__ tile_valid,
            int64_t n_tiles
        ) {
            for (int64_t t = blockIdx.x*(int64_t)blockDim.x + threadIdx.x; t < n_tiles;
                 t += (int64_t)gridDim.x*blockDim.x) {
                if (tile_valid[t] != 0) {
                    while (!atomicAdd_system(&local_flag[t], 0)) {
                    }
                }
            }
        }
        __global__ void wait_and_reduce_kernel(
            const __nv_bfloat16 *__restrict__ local_hidden,
            int32_t *__restrict__ local_flag,
            const float *__restrict__ routed_scores,
            __nv_bfloat16 *__restrict__ weighted_routed_out,
            const int32_t *__restrict__ dispatch_peer_rank,
            const int32_t *__restrict__ dispatch_local_row_start,
            const int32_t *__restrict__ dispatch_own_tile_ordinal,
            const int32_t *__restrict__ dispatch_valid_rows,
            int64_t n_dispatch_tiles,
            int block_m,
            int dim
        ) {
            for (int64_t tile=blockIdx.x; tile < n_dispatch_tiles; tile += gridDim.x) {
                int32_t dest_check = dispatch_peer_rank[tile];
                if (dest_check < 0) continue;
                int64_t local_start = dispatch_local_row_start[tile];
                int64_t ordinal = dispatch_own_tile_ordinal[tile];
                int32_t valid = dispatch_valid_rows[tile];
                __shared__ int ready;
                if (threadIdx.x == 0) {
                    while (!atomicAdd_system(&local_flag[ordinal], 0));
                    ready = 1;
                }
                __syncthreads();
                (void)ready;
                const __nv_bfloat16 *src_base = local_hidden + ordinal*block_m*(int64_t)dim;
                __nv_bfloat16 *dst_base = weighted_routed_out + local_start*(int64_t)dim;
                for (int row=0; row < valid; row++) {
                    float score = routed_scores[local_start + row];
                    const __nv_bfloat16 *src_row = src_base + (int64_t)row*dim;
                    __nv_bfloat16 *dst_row = dst_base + (int64_t)row*dim;
                    for (int col=threadIdx.x; col < dim; col += blockDim.x) {
                        float v = __bfloat162float(src_row[col])*score;
                        dst_row[col] = __float2bfloat16(v);
                    }
                }
            }
    }

    constexpr int FUSED_THREADS = 1024;
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr int FUSED_MAX_WARPS = FUSED_THREADS>>5;

    __device__ void bf16_gemm_bt_tile(
        const __nv_bfloat16 *__restrict__ A,
        const __nv_bfloat16 *__restrict__ B,
        __nv_bfloat16 *__restrict__ out,
        int M, int N, int K
    ) {
        using namespace nvcuda;
        __shared__ float store_buf[FUSED_MAX_WARPS][WMMA_M][WMMA_N];

        int warp_id = threadIdx.x>>5;
        int lane = 31&threadIdx.x;
        int num_warps = blockDim.x>>5;
        int m_tiles = M / WMMA_M;
        int n_tiles = N / WMMA_N;
        int total_tiles = m_tiles*n_tiles;
        for (int t=warp_id; t < total_tiles; t += num_warps) {
            int mt = t / n_tiles;
            int nt = t % n_tiles;
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
            for (int k0=0; k0 < K; k0 += WMMA_K) {
                const __nv_bfloat16 *a_ptr = A + (int64_t)(mt*WMMA_M)*K + k0;
                const __nv_bfloat16 *b_ptr = B + (int64_t)(nt*WMMA_N)*K + k0;
                wmma::load_matrix_sync(a_frag, a_ptr, K);
                wmma::load_matrix_sync(b_frag, b_ptr, K);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(&store_buf[warp_id][0][0], c_frag, WMMA_N, wmma::mem_row_major);
            __syncwarp();
            __nv_bfloat16 *out_tile = out + (int64_t)(mt*WMMA_M)*N + nt*WMMA_N;
            for (int i=lane; i < WMMA_M*WMMA_N; i += 32) {
                int r = i / WMMA_N, c = i % WMMA_N;
                out_tile[(int64_t)r*N + c] = __float2bfloat16(store_buf[warp_id][r][c]);
            }
            __syncwarp();
        }
    }

    constexpr int TCGEN05_BLOCK_M = 128;
    constexpr int TCGEN05_BLOCK_N = 128;
    constexpr int TCGEN05_BLOCK_K = 64;
    constexpr int TCGEN05_ACTIVE_WARPS = 4;

    __device__ void bf16_gemm_bt_tile_tcgen05(
        const CUtensorMap &A_tmap, int a_row,
        const CUtensorMap &B_tmap, int b_row,
        __nv_bfloat16 *__restrict__ out, int out_ld,
        int N, int K,
        char *__restrict__ smem_pool,
        uint32_t taddr
    ) {
        int warp_id = threadIdx.x>>5;
        __nv_bfloat16 *A_smem = reinterpret_cast<__nv_bfloat16 *>(smem_pool);
        __nv_bfloat16 *B_smem = A_smem + TCGEN05_BLOCK_M*TCGEN05_BLOCK_K;

        #pragma nv_diag_suppress static_var_with_dynamic_init
        __shared__ barrier mbar;

        if (threadIdx.x == 0) {
            mbar.init(1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        __syncthreads();

        int phase = 0;
        const uint32_t i_desc = tcgen05::encode_idesc_format_1(TCGEN05_BLOCK_M, TCGEN05_BLOCK_N);
        const int n_chunks = N / TCGEN05_BLOCK_N;
        const int num_iters = K / TCGEN05_BLOCK_K;

        for (int nc = 0; nc < n_chunks; nc++) {
            for (int iter_k = 0; iter_k < num_iters; iter_k++) {
                if (warp_id == 0 && threadIdx.x == 0) {
                    for (int k = 0; k < TCGEN05_BLOCK_K/8; k++) {
                        const int off_k8 = (iter_k*TCGEN05_BLOCK_K + k*8)/8;
                        cp_async::load3d(A_smem + k*TCGEN05_BLOCK_M*8, &A_tmap, *mbar, 0, a_row, off_k8);
                        cp_async::load3d(B_smem + k*TCGEN05_BLOCK_N*8, &B_tmap, *mbar, 0, b_row + nc*TCGEN05_BLOCK_N, off_k8);
                    }
                    constexpr uint32_t cp_size = (TCGEN05_BLOCK_M + TCGEN05_BLOCK_N)*TCGEN05_BLOCK_K*sizeof(__nv_bfloat16);
                    mbar.expect_nb(cp_size);
                }
                if (warp_id < TCGEN05_ACTIVE_WARPS) {
                    mbar.await(phase);
                    tcgen05::after_thread_sync();
                }
                phase ^= 1;

                if (warp_id == 0 && threadIdx.x == 0) {
                    tcgen05::mma_f16(taddr, tcgen05::encode_smem_desc(A_smem, TCGEN05_BLOCK_M),
                                      tcgen05::encode_smem_desc(B_smem, TCGEN05_BLOCK_N), i_desc, iter_k);
                    for (int k = 1; k < TCGEN05_BLOCK_K/16; k++) {
                        tcgen05::mma_f16(
                            taddr,
                            tcgen05::encode_smem_desc(A_smem + k*TCGEN05_BLOCK_M*16, TCGEN05_BLOCK_M),
                            tcgen05::encode_smem_desc(B_smem + k*TCGEN05_BLOCK_N*16, TCGEN05_BLOCK_N),
                            i_desc, 1);
                    }
                    tcgen05::commit_mbarrier(*mbar);
                }
                if (warp_id < TCGEN05_ACTIVE_WARPS) mbar.await(phase);
                phase ^= 1;
            }

            if (warp_id < TCGEN05_ACTIVE_WARPS) {
                tcgen05::after_thread_sync();
                for (int n = 0; n < TCGEN05_BLOCK_N/8; n++) {
                    float tmp[8];
                    tcgen05::ld_32x32b_x8(tmp, taddr + ((warp_id*32)<<16) + (n*8));
                    tcgen05::await_ld();

                    __nv_bfloat162 pk[4];
                    for (int i = 0; i < 4; i++)
                        pk[i] = __float22bfloat162_rn({tmp[i*2], tmp[i*2 + 1]});

                    __nv_bfloat16 *out_ptr = out + (int64_t)(warp_id*32 + (threadIdx.x&31))*out_ld + (nc*TCGEN05_BLOCK_N + n*8);
                    reinterpret_cast<int4 *>(out_ptr)[0] = reinterpret_cast<int4 *>(pk)[0];
                }
            }
            __syncthreads();
        }
    }

    // FFN compute policy for tile_pipeline_kernel_hull's consumer role (see tiled_pipeline.cuh):
    // up/gate/down projection via bf16_gemm_bt_tile_tcgen05, matching pi::tile_compute. This,
    // together with peer_store_transport (transport.cuh) and round_robin_scheduler (sched.cuh),
    // replaces the old monolithic fused_dispatch_ffn_kernel __global__ with the generic pipeline.
    //
    // ctx.dyn_smem's first 1024 bytes are reserved for the CTA-lifetime tensor-memory address
    // (tcgen05.alloc happens once in init(), reused by every exec() call, freed once in
    // epilogue() -- see bf16_gemm_bt_tile_tcgen05's docstring for why repeated alloc/dealloc per
    // call is illegal); the TMA staging pool for the GEMMs themselves starts right after, at byte
    // offset 1024 (kept 1024-aligned to match tcgen05_smem's own required alignment).
    struct ffn_dispatch_compute final {
        const __nv_bfloat16 *gate_proj;
        __nv_bfloat16 *expert_out;
        __nv_bfloat16 *act_scratch;
        __nv_bfloat16 *gate_scratch;
        const int32_t *recv_tile_valid;
        const int32_t *recv_tile_to_local_expert;

        CUtensorMap hidden_tmap;
        CUtensorMap up_tmap;
        CUtensorMap gate_tmap;
        CUtensorMap act_tmap;
        CUtensorMap down_tmap;

        int hidden_dim;
        int intermediate_dim;
        int block_m;

        __device__ bool valid(int64_t tile) const {
            return recv_tile_valid[tile] != 0;
        }

        __device__ void init(cta_block_ctx &ctx) const {
            int warp_id = threadIdx.x>>5;
            if (warp_id == 1)
                tcgen05::tmem_alloc(ctx.dyn_smem, TCGEN05_BLOCK_N);
            __syncthreads();
        }

        __device__ void exec(int64_t tile, cta_block_ctx &ctx) const {
            const uint32_t taddr = *static_cast<uint32_t *>(ctx.dyn_smem);
            char *smem_pool = static_cast<char *>(ctx.dyn_smem) + 1024;

            int32_t e = recv_tile_to_local_expert[tile];
            int slot = ctx.com_slot;
            __nv_bfloat16 *my_act = act_scratch + (int64_t)slot*block_m*intermediate_dim;
            __nv_bfloat16 *my_gate = gate_proj != nullptr ? gate_scratch + (int64_t)slot*block_m*intermediate_dim : nullptr;
            int a_row = (int)(tile*block_m);
            int b_row = e*intermediate_dim;

            bf16_gemm_bt_tile_tcgen05(hidden_tmap, a_row, up_tmap, b_row, my_act, intermediate_dim, intermediate_dim, hidden_dim, smem_pool, taddr);
            if (gate_proj != nullptr) {
                bf16_gemm_bt_tile_tcgen05(hidden_tmap, a_row, gate_tmap, b_row, my_gate, intermediate_dim, intermediate_dim, hidden_dim, smem_pool, taddr);
                __syncthreads();
                for (int i=threadIdx.x; i < block_m*intermediate_dim; i += blockDim.x) {
                    float g = __bfloat162float(my_gate[i]);
                    float u = __bfloat162float(my_act[i]);
                    float silu_g = g/(1.0f + __expf(-g));
                    my_act[i] = __float2bfloat16(silu_g*u);
                }
            } else {
                __syncthreads();
                for (int i=threadIdx.x; i < block_m*intermediate_dim; i += blockDim.x) {
                    float u = __bfloat162float(my_act[i]);
                    my_act[i] = __float2bfloat16(u/(1.0f + __expf(-u)));
                }
            }
            __syncthreads();

            __nv_bfloat16 *out_tile = expert_out + tile*(int64_t)block_m*hidden_dim;
            int act_row = slot*block_m;
            bf16_gemm_bt_tile_tcgen05(act_tmap, act_row, down_tmap, e*hidden_dim, out_tile, hidden_dim, hidden_dim, intermediate_dim, smem_pool, taddr);
            __syncthreads();
        }

        __device__ void epilogue(cta_block_ctx &ctx) const {
            int warp_id = threadIdx.x>>5;
            const uint32_t taddr = *static_cast<uint32_t *>(ctx.dyn_smem);
            if (warp_id == 0)
                tcgen05::tmem_free(taddr, TCGEN05_BLOCK_N);
        }
    };

    __device__ void bf16_gemm_nn_sum2_tile(
        const __nv_bfloat16 *__restrict__ A1,
        const __nv_bfloat16 *__restrict__ B1,
        int K1,
        const __nv_bfloat16 *__restrict__ A2,
        const __nv_bfloat16 *__restrict__ B2,
        int K2,
        __nv_bfloat16 *__restrict__ out,
        int M, int N
    ) {
        using namespace nvcuda;
        __shared__ float store_buf[FUSED_MAX_WARPS][WMMA_M][WMMA_N];

        int warp_id = threadIdx.x>>5;
        int lane = 31&threadIdx.x;
        int num_warps = blockDim.x>>5;
        int m_tiles = M / WMMA_M;
        int n_tiles = N / WMMA_N;
        int total_tiles = m_tiles*n_tiles;

        for (int t=warp_id; t < total_tiles; t += num_warps) {
            int mt = t / n_tiles;
            int nt = t % n_tiles;
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
            for (int k0=0; k0 < K1; k0 += WMMA_K) {
                const __nv_bfloat16 *a_ptr = A1 + (int64_t)(mt*WMMA_M)*K1 + k0;
                const __nv_bfloat16 *b_ptr = B1 + (int64_t)k0*N + nt*WMMA_N;
                wmma::load_matrix_sync(a_frag, a_ptr, K1);
                wmma::load_matrix_sync(b_frag, b_ptr, N);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            if (A2 != nullptr) {
                for (int k0=0; k0 < K2; k0 += WMMA_K) {
                    const __nv_bfloat16 *a_ptr = A2 + (int64_t)(mt*WMMA_M)*K2 + k0;
                    const __nv_bfloat16 *b_ptr = B2 + (int64_t)k0*N + nt*WMMA_N;
                    wmma::load_matrix_sync(a_frag, a_ptr, K2);
                    wmma::load_matrix_sync(b_frag, b_ptr, N);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }
            }
            wmma::store_matrix_sync(&store_buf[warp_id][0][0], c_frag, WMMA_N, wmma::mem_row_major);
            __syncwarp();
            __nv_bfloat16 *out_tile = out + (int64_t)(mt*WMMA_M)*N + nt*WMMA_N;
            for (int i=lane; i < WMMA_M*WMMA_N; i += 32) {
                int r = i / WMMA_N, c = i % WMMA_N;
                out_tile[(int64_t)r*N + c] = __float2bfloat16(store_buf[warp_id][r][c]);
            }
            __syncwarp();
        }
    }

    __device__ void bf16_gemm_tn_atomic_tile(
        const __nv_bfloat16 *__restrict__ A,
        const __nv_bfloat16 *__restrict__ B,
        float *__restrict__ out_fp32,
        int M, int N1, int N2
    ) {
        using namespace nvcuda;
        __shared__ float store_buf[FUSED_MAX_WARPS][WMMA_M][WMMA_N];

        int warp_id = threadIdx.x>>5;
        int lane = 31&threadIdx.x;
        int num_warps = blockDim.x>>5;
        int m_tiles = N1 / WMMA_M;
        int n_tiles = N2 / WMMA_N;
        int total_tiles = m_tiles*n_tiles;

        for (int t=warp_id; t < total_tiles; t += num_warps) {
            int mt = t / n_tiles;
            int nt = t % n_tiles;
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
            for (int k0=0; k0 < M; k0 += WMMA_K) {
                const __nv_bfloat16 *a_ptr = A + (int64_t)k0*N1 + mt*WMMA_M;
                const __nv_bfloat16 *b_ptr = B + (int64_t)k0*N2 + nt*WMMA_N;
                wmma::load_matrix_sync(a_frag, a_ptr, N1);
                wmma::load_matrix_sync(b_frag, b_ptr, N2);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(&store_buf[warp_id][0][0], c_frag, WMMA_N, wmma::mem_row_major);
            __syncwarp();
            float *out_tile = out_fp32 + (int64_t)(mt*WMMA_M)*N2 + nt*WMMA_N;
            for (int i=lane; i < WMMA_M*WMMA_N; i += 32) {
                int r = i / WMMA_N, c = i % WMMA_N;
                atomicAdd(&out_tile[(int64_t)r*N2 + c], store_buf[warp_id][r][c]);
            }
            __syncwarp();
        }
    }

    __global__ void fused_grad_combine_ffn_kernel(
        const uint8_t *__restrict__ src,
        const int64_t *__restrict__ hidden_peer_ptrs,
        const int64_t *__restrict__ flag_peer_ptrs,
        const int32_t *__restrict__ tile_peer_rank,
        const int32_t *__restrict__ tile_local_row_start,
        const int32_t *__restrict__ tile_peer_row_start,
        const int32_t *__restrict__ tile_valid_rows,
        const int32_t *__restrict__ tile_flag_index,
        int64_t n_dispatch_tiles,
        int64_t row_bytes,
        const __nv_bfloat16 *__restrict__ grad_expert_out_recv,
        int32_t *__restrict__ recv_flag,
        const int32_t *__restrict__ recv_tile_valid,
        const int32_t *__restrict__ recv_tile_to_local_expert,
        const __nv_bfloat16 *__restrict__ hidden_shadow,
        const __nv_bfloat16 *__restrict__ gate_proj,
        const __nv_bfloat16 *__restrict__ up_proj,
        const __nv_bfloat16 *__restrict__ down_proj,
        __nv_bfloat16 *__restrict__ grad_dispatch_hidden_out,
        __nv_bfloat16 *__restrict__ up_scratch,
        __nv_bfloat16 *__restrict__ gate_scratch,
        __nv_bfloat16 *__restrict__ grad_act_scratch,
        __nv_bfloat16 *__restrict__ act_scratch,
        float *__restrict__ grad_up_proj_fp32,
        float *__restrict__ grad_down_proj_fp32,
        float *__restrict__ grad_gate_proj_fp32,
        int64_t n_recv_tiles,
        int hidden_dim,
        int intermediate_dim,
        int block_m,
        int n_producer_blocks
    ) {
        if ((int)blockIdx.x < n_producer_blocks) {
            for (int64_t tile = blockIdx.x; tile < n_dispatch_tiles; tile += n_producer_blocks) {
                int32_t dest_rank = tile_peer_rank[tile];
                if (dest_rank < 0) continue;
                int64_t local_start = tile_local_row_start[tile];
                int64_t peer_start = tile_peer_row_start[tile];
                int64_t valid = tile_valid_rows[tile];
                int32_t flag_idx = tile_flag_index[tile];
                auto *dest_base = reinterpret_cast<uint8_t *>(hidden_peer_ptrs[dest_rank]);
                const uint4 *src_vec = reinterpret_cast<const uint4 *>(src + local_start*row_bytes);
                uint4 *dst_vec = reinterpret_cast<uint4 *>(dest_base + peer_start*row_bytes);
                int64_t n_vec = (valid*row_bytes)>>4;
                for (int64_t i=threadIdx.x; i < n_vec; i += blockDim.x)
                    dst_vec[i] = src_vec[i];
                __syncthreads();
                if (threadIdx.x == 0) {
                    __threadfence_system();
                    auto *flag_base = reinterpret_cast<int32_t *>(flag_peer_ptrs[dest_rank]);
                    atomicExch_system(&flag_base[flag_idx], 1);
                }
            }
        } else {
            int n_consumer_blocks = gridDim.x - n_producer_blocks;
            int slot = (int)blockIdx.x - n_producer_blocks;
            __nv_bfloat16 *my_up = up_scratch + (int64_t)slot*block_m*intermediate_dim;
            __nv_bfloat16 *my_gate = gate_proj != nullptr ? gate_scratch + (int64_t)slot*block_m*intermediate_dim : nullptr;
            __nv_bfloat16 *my_grad_act = grad_act_scratch + (int64_t)slot*block_m*intermediate_dim;
            __nv_bfloat16 *my_act = act_scratch + (int64_t)slot*block_m*intermediate_dim;

            for (int64_t tile = slot; tile < n_recv_tiles; tile += n_consumer_blocks) {
                if (recv_tile_valid[tile] == 0) continue;
                __shared__ int ready;
                if (threadIdx.x == 0) {
                    while (!atomicAdd_system(&recv_flag[tile], 0));
                    ready = 1;
                }
                __syncthreads();
                (void)ready;

                int32_t e = recv_tile_to_local_expert[tile];
                const __nv_bfloat16 *hidden_tile = hidden_shadow + tile*(int64_t)block_m*hidden_dim;
                const __nv_bfloat16 *grad_out_tile = grad_expert_out_recv + tile*(int64_t)block_m*hidden_dim;
                const __nv_bfloat16 *up_e = up_proj + (int64_t)e*intermediate_dim*hidden_dim;
                const __nv_bfloat16 *down_e = down_proj + (int64_t)e*hidden_dim*intermediate_dim;
                bf16_gemm_bt_tile(hidden_tile, up_e, my_up, block_m, intermediate_dim, hidden_dim);
                if (gate_proj != nullptr) {
                    const __nv_bfloat16 *gate_e = gate_proj + (int64_t)e*intermediate_dim*hidden_dim;
                    bf16_gemm_bt_tile(hidden_tile, gate_e, my_gate, block_m, intermediate_dim, hidden_dim);
                }
                __syncthreads();
                bf16_gemm_nn_sum2_tile(
                    grad_out_tile, down_e, hidden_dim,
                    nullptr, nullptr, 0,
                    my_grad_act, block_m, intermediate_dim
                );
                __syncthreads();
                if (gate_proj != nullptr) {
                    for (int i = threadIdx.x; i < block_m*intermediate_dim; i += blockDim.x) {
                        float g = __bfloat162float(my_gate[i]);
                        float u = __bfloat162float(my_up[i]);
                        float sig = 1.0f/(1.0f + __expf(-g));
                        my_act[i] = __float2bfloat16(u*g*sig);
                    }
                } else {
                    for (int i = threadIdx.x; i < block_m*intermediate_dim; i += blockDim.x) {
                        float u = __bfloat162float(my_up[i]);
                        float sig = 1.0f/(1.0f + __expf(-u));
                        my_act[i] = __float2bfloat16(u*sig);
                    }
                }
                __syncthreads();
                bf16_gemm_tn_atomic_tile(
                    grad_out_tile, my_act,
                    grad_down_proj_fp32 + (int64_t)e*hidden_dim*intermediate_dim,
                    block_m, hidden_dim, intermediate_dim
                );
                __syncthreads();
                if (gate_proj != nullptr) {
                    for (int i = threadIdx.x; i < block_m*intermediate_dim; i += blockDim.x) {
                        float g = __bfloat162float(my_gate[i]);
                        float u = __bfloat162float(my_up[i]);
                        float ga = __bfloat162float(my_grad_act[i]);
                        float sig = 1.0f/(1.0f + __expf(-g));
                        float silu_g = g*sig;
                        float dsilu_g = sig*(1.0f + g*(1.0f - sig));
                        my_up[i] = __float2bfloat16(ga*silu_g);
                        my_gate[i] = __float2bfloat16(ga*u*dsilu_g);
                    }
                } else {
                    for (int i = threadIdx.x; i < block_m*intermediate_dim; i += blockDim.x) {
                        float u = __bfloat162float(my_up[i]);
                        float ga = __bfloat162float(my_grad_act[i]);
                        float sig = 1.0f/(1.0f + __expf(-u));
                        float dsilu_u = sig*(1.0f + u*(1.0f - sig));
                        my_up[i] = __float2bfloat16(ga*dsilu_u);
                    }
                }
                __syncthreads();
                __nv_bfloat16 *grad_hidden_tile = grad_dispatch_hidden_out + tile*(int64_t)block_m*hidden_dim;
                bf16_gemm_nn_sum2_tile(
                    my_up, up_e, intermediate_dim,
                    gate_proj != nullptr ? my_gate : nullptr, gate_proj != nullptr ? gate_proj + (int64_t)e*intermediate_dim*hidden_dim : nullptr, intermediate_dim,
                    grad_hidden_tile, block_m, hidden_dim
                );
                __syncthreads();
                bf16_gemm_tn_atomic_tile(
                    my_up, hidden_tile,
                    grad_up_proj_fp32 + (int64_t)e*intermediate_dim*hidden_dim,
                    block_m, intermediate_dim, hidden_dim
                );
                if (gate_proj != nullptr) {
                    bf16_gemm_tn_atomic_tile(
                        my_gate, hidden_tile,
                        grad_gate_proj_fp32 + (int64_t)e*intermediate_dim*hidden_dim,
                        block_m, intermediate_dim, hidden_dim
                    );
                }
                __syncthreads();
            }
        }
    }
}

void launch_scatter_tiles(
    const uint8_t *src,
    const int64_t *hidden_peer_ptrs,
    const int64_t *flag_peer_ptrs,
    const int32_t *tile_peer_rank,
    const int32_t *tile_local_row_start,
    const int32_t *tile_peer_row_start,
    const int32_t *tile_valid_rows,
    const int32_t *tile_flag_index,
    int64_t n_tiles,
    int64_t row_bytes,
    int n_blocks,
    cudaStream_t stream
) {
    int blocks = n_tiles < n_blocks ? (int)n_tiles : n_blocks;
    if (blocks <= 0) return;
    scatter_tiles_kernel<<<blocks, 256, 0, stream>>>(
        src, hidden_peer_ptrs, flag_peer_ptrs, tile_peer_rank, tile_local_row_start,
        tile_peer_row_start, tile_valid_rows, tile_flag_index, n_tiles, row_bytes
    );
}

void launch_wait_tiles(
    int32_t *local_flag,
    const int32_t *tile_valid,
    int64_t n_tiles,
    cudaStream_t stream
) {
    if (n_tiles <= 0) return;
    int threads = 256;
    int64_t blocks64 = (n_tiles+threads - 1) / threads;
    int blocks = blocks64 > 2048 ? 2048 : (int)blocks64;
    wait_tiles_kernel<<<blocks, threads, 0, stream>>>(local_flag, tile_valid, n_tiles);
}

void launch_wait_and_reduce(
    const void *local_hidden_bf16,
    int32_t *local_flag,
    const float *routed_scores,
    void *weighted_routed_out_bf16,
    const int32_t *dispatch_peer_rank,
    const int32_t *dispatch_local_row_start,
    const int32_t *dispatch_own_tile_ordinal,
    const int32_t *dispatch_valid_rows,
    int64_t n_dispatch_tiles,
    int block_m,
    int dim,
    int n_blocks,
    cudaStream_t stream
) {
    int blocks = n_dispatch_tiles < n_blocks ? (int)n_dispatch_tiles : n_blocks;
    if (blocks <= 0) return;
    wait_and_reduce_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(local_hidden_bf16),
        local_flag,
        routed_scores,
        static_cast<__nv_bfloat16 *>(weighted_routed_out_bf16),
        dispatch_peer_rank,
        dispatch_local_row_start,
        dispatch_own_tile_ordinal,
        dispatch_valid_rows,
        n_dispatch_tiles,
        block_m,
        dim
    );
}

void launch_fused_dispatch_ffn(
    const uint8_t *src,
    const int64_t *hidden_peer_ptrs,
    const int64_t *flag_peer_ptrs,
    const int32_t *tile_peer_rank,
    const int32_t *tile_local_row_start,
    const int32_t *tile_peer_row_start,
    const int32_t *tile_valid_rows,
    const int32_t *tile_flag_index,
    int64_t n_dispatch_tiles,
    int64_t row_bytes,
    const void *recv_hidden_bf16,
    int32_t *recv_flag,
    const int32_t *recv_tile_valid,
    const int32_t *recv_tile_to_local_expert,
    const void *gate_proj_bf16,
    const void *up_proj_bf16,
    const void *down_proj_bf16,
    void *expert_out_bf16,
    void *act_scratch_bf16,
    void *gate_scratch_bf16,
    int64_t n_recv_tiles,
    int hidden_dim,
    int intermediate_dim,
    int block_m,
    int n_producer_blocks,
    int n_consumer_blocks,
    int64_t dispatch_capacity,
    int64_t num_local_experts,
    cudaStream_t stream
) {
    int total_blocks = n_producer_blocks + n_consumer_blocks;
    if (total_blocks <= 0) {
        return;
    }

    auto make_tmap = [](const char *name, const void *ptr, int64_t rows, int width) {
        return init_tmap_kmajor_3d(name, ptr, rows, width, TCGEN05_BLOCK_M, 8);
    };
    CUtensorMap hidden_tmap = make_tmap("fine_grained_compute_comm_overlap.hidden", recv_hidden_bf16, dispatch_capacity, hidden_dim);
    CUtensorMap up_tmap = make_tmap("fine_grained_compute_comm_overlap.up_proj", up_proj_bf16, num_local_experts*intermediate_dim, hidden_dim);
    CUtensorMap gate_tmap = gate_proj_bf16 != nullptr
        ? make_tmap("fine_grained_compute_comm_overlap.gate_proj", gate_proj_bf16, num_local_experts*intermediate_dim, hidden_dim)
        : CUtensorMap{};
    CUtensorMap act_tmap = make_tmap("fine_grained_compute_comm_overlap.act_scratch", act_scratch_bf16, (int64_t)n_consumer_blocks*block_m, intermediate_dim);
    CUtensorMap down_tmap = make_tmap("fine_grained_compute_comm_overlap.down_proj", down_proj_bf16, num_local_experts*hidden_dim, intermediate_dim);

    peer_store_transport transport{};
    transport.src = src;
    transport.hidden_peer_ptrs = hidden_peer_ptrs;
    transport.flag_peer_ptrs = flag_peer_ptrs;
    transport.tile_peer_rank = tile_peer_rank;
    transport.tile_local_row_start = tile_local_row_start;
    transport.tile_peer_row_start = tile_peer_row_start;
    transport.tile_valid_rows = tile_valid_rows;
    transport.tile_flag_index = tile_flag_index;
    transport.row_bytes = row_bytes;
    transport.recv_flag = recv_flag;

    round_robin_scheduler scheduler{};
    scheduler.n_send_tiles = n_dispatch_tiles;
    scheduler.n_recv_tiles = n_recv_tiles;

    ffn_dispatch_compute compute{};
    compute.gate_proj = static_cast<const __nv_bfloat16 *>(gate_proj_bf16);
    compute.expert_out = static_cast<__nv_bfloat16 *>(expert_out_bf16);
    compute.act_scratch = static_cast<__nv_bfloat16 *>(act_scratch_bf16);
    compute.gate_scratch = static_cast<__nv_bfloat16 *>(gate_scratch_bf16);
    compute.recv_tile_valid = recv_tile_valid;
    compute.recv_tile_to_local_expert = recv_tile_to_local_expert;
    compute.hidden_tmap = hidden_tmap;
    compute.up_tmap = up_tmap;
    compute.gate_tmap = gate_tmap;
    compute.act_tmap = act_tmap;
    compute.down_tmap = down_tmap;
    compute.hidden_dim = hidden_dim;
    compute.intermediate_dim = intermediate_dim;
    compute.block_m = block_m;

    constexpr size_t smem_size = 1024 + (size_t)(TCGEN05_BLOCK_M + TCGEN05_BLOCK_N)*TCGEN05_BLOCK_K*sizeof(__nv_bfloat16);
    cudaFuncSetAttribute( // always set mem via func set
        tile_pipeline_kernel_hull<peer_store_transport, ffn_dispatch_compute, round_robin_scheduler>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_size)
    );
    launch_tile_pipeline(
        transport, compute, scheduler,
        n_producer_blocks, n_consumer_blocks,
        FUSED_THREADS, smem_size, stream
    );
}

void launch_fused_grad_combine_ffn(
    const uint8_t *src,
    const int64_t *hidden_peer_ptrs,
    const int64_t *flag_peer_ptrs,
    const int32_t *tile_peer_rank,
    const int32_t *tile_local_row_start,
    const int32_t *tile_peer_row_start,
    const int32_t *tile_valid_rows,
    const int32_t *tile_flag_index,
    int64_t n_dispatch_tiles,
    int64_t row_bytes,
    const void *grad_expert_out_recv_bf16,
    int32_t *recv_flag,
    const int32_t *recv_tile_valid,
    const int32_t *recv_tile_to_local_expert,
    const void *hidden_shadow_bf16,
    const void *gate_proj_bf16,
    const void *up_proj_bf16,
    const void *down_proj_bf16,
    void *grad_dispatch_hidden_out_bf16,
    void *up_scratch_bf16,
    void *gate_scratch_bf16,
    void *grad_act_scratch_bf16,
    void *act_scratch_bf16,
    float *grad_up_proj_fp32,
    float *grad_down_proj_fp32,
    float *grad_gate_proj_fp32,
    int64_t n_recv_tiles,
    int hidden_dim,
    int intermediate_dim,
    int block_m,
    int n_producer_blocks,
    int n_consumer_blocks,
    cudaStream_t stream
) {
    int total_blocks = n_producer_blocks + n_consumer_blocks;
    if (total_blocks <= 0) {
        return;
    }
    fused_grad_combine_ffn_kernel<<<total_blocks, FUSED_THREADS, 0, stream>>>(
        src, hidden_peer_ptrs, flag_peer_ptrs, tile_peer_rank, tile_local_row_start,
        tile_peer_row_start, tile_valid_rows, tile_flag_index, n_dispatch_tiles, row_bytes,
        static_cast<const __nv_bfloat16 *>(grad_expert_out_recv_bf16),
        recv_flag,
        recv_tile_valid,
        recv_tile_to_local_expert,
        static_cast<const __nv_bfloat16 *>(hidden_shadow_bf16),
        static_cast<const __nv_bfloat16 *>(gate_proj_bf16),
        static_cast<const __nv_bfloat16 *>(up_proj_bf16),
        static_cast<const __nv_bfloat16 *>(down_proj_bf16),
        static_cast<__nv_bfloat16 *>(grad_dispatch_hidden_out_bf16),
        static_cast<__nv_bfloat16 *>(up_scratch_bf16),
        static_cast<__nv_bfloat16 *>(gate_scratch_bf16),
        static_cast<__nv_bfloat16 *>(grad_act_scratch_bf16),
        static_cast<__nv_bfloat16 *>(act_scratch_bf16),
        grad_up_proj_fp32,
        grad_down_proj_fp32,
        grad_gate_proj_fp32,
        n_recv_tiles,
        hidden_dim,
        intermediate_dim,
        block_m,
        n_producer_blocks
    );
}
}
