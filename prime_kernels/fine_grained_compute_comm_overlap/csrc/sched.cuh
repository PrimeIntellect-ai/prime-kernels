#pragma once

#include <cstdint>

namespace pi {
    struct round_robin_scheduler final {
        int64_t n_send_tiles = 0;
        int64_t n_recv_tiles = 0;
        struct ProdState {
            int64_t next;
            int64_t stride;
        };
        struct ComState {
            int64_t next;
            int64_t stride;
        };

        [[nodiscard]] __device__ __forceinline__ ProdState prod_init(int slot, int nblocks) const {
            return ProdState{static_cast<int64_t>(slot), static_cast<int64_t>(nblocks)};
        }

        [[nodiscard]] __device__ __forceinline__ int64_t next_send_tile(ProdState &state) const {
            if (state.next >= n_send_tiles) return -1;
            int64_t tile = state.next;
            state.next += state.stride;
            return tile;
        }

        [[nodiscard]] __device__ __forceinline__ ComState com_init(int slot, int nblocks) const {
            return ComState{static_cast<int64_t>(slot), static_cast<int64_t>(nblocks)};
        }

        [[nodiscard]] __device__ __forceinline__ int64_t next_compute_tile(ComState &state) const {
            if (state.next >= n_recv_tiles) return -1;
            int64_t tile = state.next;
            state.next += state.stride;
            return tile;
        }
    };
}
