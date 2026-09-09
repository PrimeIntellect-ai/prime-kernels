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

    extern void launch_fused_dispatch_ffn(
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
    );

    extern void launch_fused_grad_combine_ffn(
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
    );

}
