#include "kernels.cuh"

#include <cuda_bf16.h>

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
            for (int64_t tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
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
                int64_t n_vec = (valid*row_bytes) / 16;

                for (int64_t i = threadIdx.x; i < n_vec; i += blockDim.x) {
                    dst_vec[i] = src_vec[i];
                }

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
            for (int64_t tile = blockIdx.x; tile < n_dispatch_tiles; tile += gridDim.x) {
                int32_t dest_check = dispatch_peer_rank[tile];
                if (dest_check < 0) {
                    continue;
                }
                int64_t local_start = dispatch_local_row_start[tile];
                int64_t ordinal = dispatch_own_tile_ordinal[tile];
                int32_t valid = dispatch_valid_rows[tile];

                __shared__ int ready;
                if (threadIdx.x == 0) {
                    while (atomicAdd_system(&local_flag[ordinal], 0) == 0) {
                    }
                    ready = 1;
                }
                __syncthreads();
                (void)ready;

                const __nv_bfloat16 *src_base = local_hidden + ordinal*block_m*(int64_t)dim;
                __nv_bfloat16 *dst_base = weighted_routed_out + local_start*(int64_t)dim;

                for (int row = 0; row < valid; row++) {
                    float score = routed_scores[local_start + row];
                    const __nv_bfloat16 *src_row = src_base + (int64_t)row*dim;
                    __nv_bfloat16 *dst_row = dst_base + (int64_t)row*dim;
                    for (int col = threadIdx.x; col < dim; col += blockDim.x) {
                        float v = __bfloat162float(src_row[col])*score;
                        dst_row[col] = __float2bfloat16(v);
                    }
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
    if (blocks <= 0) {
        return;
    }
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
    if (n_tiles <= 0) {
        return;
    }
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
    if (blocks <= 0) {
        return;
    }
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
}
