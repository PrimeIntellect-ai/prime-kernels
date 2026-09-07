#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace pi {
    extern void launch_scatter_tiles(
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
    );

    extern void launch_wait_tiles(
        int32_t *local_flag,
        const int32_t *tile_valid,
        int64_t n_tiles,
        cudaStream_t stream
    );

    extern void launch_wait_and_reduce(
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
    );

}
