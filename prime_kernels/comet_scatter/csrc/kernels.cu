#include "kernels.cuh"

#include <cuda_bf16.h>
#include <mma.h>

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
                if (dest_rank < 0) {
                    continue;
                }
                int64_t local_start = tile_local_row_start[tile];
                int64_t peer_start = tile_peer_row_start[tile];
                int64_t valid = tile_valid_rows[tile];
                int32_t flag_idx = tile_flag_index[tile];

                auto *dest_base = reinterpret_cast<uint8_t *>(hidden_peer_ptrs[dest_rank]);
                const uint4 *src_vec = reinterpret_cast<const uint4 *>(src + local_start*row_bytes);
                uint4 *dst_vec = reinterpret_cast<uint4 *>(dest_base + peer_start*row_bytes);
                int64_t n_vec = (valid*row_bytes)>>4;
                for (int64_t i = threadIdx.x; i < n_vec; i += blockDim.x)
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
                    while (atomicAdd_system(&local_flag[t], 0) == 0) {
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

        int warp_id = threadIdx.>>5;
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

    __global__ void fused_dispatch_ffn_kernel(
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
        const __nv_bfloat16 *__restrict__ recv_hidden,
        int32_t *__restrict__ recv_flag,
        const int32_t *__restrict__ recv_tile_valid,
        const int32_t *__restrict__ recv_tile_to_local_expert,
        const __nv_bfloat16 *__restrict__ gate_proj,
        const __nv_bfloat16 *__restrict__ up_proj,
        const __nv_bfloat16 *__restrict__ down_proj,
        __nv_bfloat16 *__restrict__ expert_out,
        __nv_bfloat16 *__restrict__ act_scratch,
        __nv_bfloat16 *__restrict__ gate_scratch,
        int64_t n_recv_tiles,
        int hidden_dim,
        int intermediate_dim,
        int block_m,
        int n_producer_blocks,
        long long *__restrict__ block_start_clock,
        long long *__restrict__ block_end_clock
    ) {
        if (threadIdx.x == 0)
            block_start_clock[blockIdx.x] = clock64();

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
            __nv_bfloat16 *my_act = act_scratch + (int64_t)slot*block_m*intermediate_dim;
            __nv_bfloat16 *my_gate = gate_proj != nullptr ? gate_scratch + (int64_t)slot*block_m*intermediate_dim : nullptr;
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
                const __nv_bfloat16 *hidden_tile = recv_hidden + tile*(int64_t)block_m*hidden_dim;
                const __nv_bfloat16 *up_e = up_proj + (int64_t)e*intermediate_dim*hidden_dim;
                const __nv_bfloat16 *down_e = down_proj + (int64_t)e*hidden_dim*intermediate_dim;
                bf16_gemm_bt_tile(hidden_tile, up_e, my_act, block_m, intermediate_dim, hidden_dim);
                if (gate_proj != nullptr) {
                    const __nv_bfloat16 *gate_e = gate_proj + (int64_t)e*intermediate_dim*hidden_dim;
                    bf16_gemm_bt_tile(hidden_tile, gate_e, my_gate, block_m, intermediate_dim, hidden_dim);
                    __syncthreads();
                    for (int i=threadIdx.x; i < block_m*intermediate_dim; i += blockDim.x) {
                        float g = __bfloat162float(my_gate[i]);
                        float u = __bfloat162float(my_act[i]);
                        float silu_g = g / (1.0f + __expf(-g));
                        my_act[i] = __float2bfloat16(silu_g*u);
                    }
                } else {
                    __syncthreads();
                    for (int i=threadIdx.x; i < block_m*intermediate_dim; i += blockDim.x) {
                        float u = __bfloat162float(my_act[i]);
                        my_act[i] = __float2bfloat16(u / (1.0f + __expf(-u)));
                    }
                }
                __syncthreads();

                __nv_bfloat16 *out_tile = expert_out + tile*(int64_t)block_m*hidden_dim;
                bf16_gemm_bt_tile(my_act, down_e, out_tile, block_m, hidden_dim, intermediate_dim);
                __syncthreads();
            }
        }
        __syncthreads();
        if (threadIdx.x == 0)
            block_end_clock[blockIdx.x] = clock64();
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
    long long *block_start_clock,
    long long *block_end_clock,
    cudaStream_t stream
) {
    int total_blocks = n_producer_blocks + n_consumer_blocks;
    if (total_blocks <= 0) {
        return;
    }
    fused_dispatch_ffn_kernel<<<total_blocks, FUSED_THREADS, 0, stream>>>(
        src, hidden_peer_ptrs, flag_peer_ptrs, tile_peer_rank, tile_local_row_start,
        tile_peer_row_start, tile_valid_rows, tile_flag_index, n_dispatch_tiles, row_bytes,
        static_cast<const __nv_bfloat16 *>(recv_hidden_bf16),
        recv_flag,
        recv_tile_valid,
        recv_tile_to_local_expert,
        static_cast<const __nv_bfloat16 *>(gate_proj_bf16),
        static_cast<const __nv_bfloat16 *>(up_proj_bf16),
        static_cast<const __nv_bfloat16 *>(down_proj_bf16),
        static_cast<__nv_bfloat16 *>(expert_out_bf16),
        static_cast<__nv_bfloat16 *>(act_scratch_bf16),
        static_cast<__nv_bfloat16 *>(gate_scratch_bf16),
        n_recv_tiles,
        hidden_dim,
        intermediate_dim,
        block_m,
        n_producer_blocks,
        block_start_clock,
        block_end_clock
    );
}
}
