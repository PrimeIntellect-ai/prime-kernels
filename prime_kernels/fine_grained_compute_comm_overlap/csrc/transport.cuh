#pragma once

#include <cstdint>
#include <cuda_bf16.h>

#include "tiled_pipeline.cuh"

namespace pi {
    struct peer_store_transport final { // Direct p2p memory transpoer (NVLink/PCIE using symmetric mem)
        const uint8_t *src = nullptr;
        const int64_t *hidden_peer_ptrs = nullptr;
        const int64_t *flag_peer_ptrs = nullptr;
        const int32_t *tile_peer_rank = nullptr;
        const int32_t *tile_local_row_start = nullptr;
        const int32_t *tile_peer_row_start = nullptr;
        const int32_t *tile_valid_rows = nullptr;
        const int32_t *tile_flag_index = nullptr;
        int64_t row_bytes = 0;
        int32_t *recv_flag = nullptr;

        [[nodiscard]] __device__ __forceinline__ void prod_init([[maybe_unused]] int slot, [[maybe_unused]] cta_block_ctx &ctx) const { }

        [[nodiscard]] __device__ __forceinline__ bool send_valid(int64_t tile) const {
            return tile_peer_rank[tile] >= 0;
        }

        __device__ void post(int64_t tile, [[maybe_unused]] int slot, [[maybe_unused]] cta_block_ctx &ctx) const {
            int32_t dest_rank = tile_peer_rank[tile];
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
            __syncthreads();
        }

        __device__ __forceinline__ void prod_progress([[maybe_unused]] int slot, [[maybe_unused]] cta_block_ctx &ctx) const {
            // Not needed here - will be needed for RDMA
        }

        __device__ __forceinline__ void prod_epilogue([[maybe_unused]] int slot, [[maybe_unused]]cta_block_ctx &ctx) const {}

        __device__ __forceinline__ void acquire(int64_t tile, [[maybe_unused]] cta_block_ctx &ctx) const {
            if (threadIdx.x == 0)
                while (!atomicAdd_system(&recv_flag[tile], 0));
            __syncthreads();
        }
    };
}
