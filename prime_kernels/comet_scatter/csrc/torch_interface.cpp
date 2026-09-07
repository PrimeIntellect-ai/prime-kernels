#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>

#include "kernels.cuh"

namespace pi {
    static void scatter_tiles_torch_stub(
        const torch::Tensor &src,
        const torch::Tensor &hidden_peer_ptrs,
        const torch::Tensor &flag_peer_ptrs,
        const torch::Tensor &tile_peer_rank,
        const torch::Tensor &tile_local_row_start,
        const torch::Tensor &tile_peer_row_start,
        const torch::Tensor &tile_valid_rows,
        const torch::Tensor &tile_flag_index,
        int64_t n_blocks
    ) {
        at::cuda::OptionalCUDAGuard device_guard{device_of(src)};
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        TORCH_CHECK(src.dim() == 2, "src must be a 2D (rows, dim) tensor");
        TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
        int64_t row_bytes = src.size(1) * src.element_size();
        TORCH_CHECK((row_bytes&15 )== 0, "src row width in bytes (", row_bytes, ") must be a multiple of 16");
        TORCH_CHECK(hidden_peer_ptrs.dtype() == torch::kInt64, "hidden_peer_ptrs must be int64");
        TORCH_CHECK(flag_peer_ptrs.dtype() == torch::kInt64, "flag_peer_ptrs must be int64");
        TORCH_CHECK(tile_peer_rank.dtype() == torch::kInt32, "tile schedule tensors must be int32");
        int64_t n_tiles = tile_peer_rank.numel();
        TORCH_CHECK(tile_local_row_start.numel() == n_tiles, "tile schedule tensors must have matching length");
        TORCH_CHECK(tile_peer_row_start.numel() == n_tiles, "tile schedule tensors must have matching length");
        TORCH_CHECK(tile_valid_rows.numel() == n_tiles, "tile schedule tensors must have matching length");
        TORCH_CHECK(tile_flag_index.numel() == n_tiles, "tile schedule tensors must have matching length");

        pi::launch_scatter_tiles(
            static_cast<const uint8_t *>(src.const_data_ptr()),
            static_cast<const int64_t *>(hidden_peer_ptrs.const_data_ptr()),
            static_cast<const int64_t *>(flag_peer_ptrs.const_data_ptr()),
            static_cast<const int32_t *>(tile_peer_rank.const_data_ptr()),
            static_cast<const int32_t *>(tile_local_row_start.const_data_ptr()),
            static_cast<const int32_t *>(tile_peer_row_start.const_data_ptr()),
            static_cast<const int32_t *>(tile_valid_rows.const_data_ptr()),
            static_cast<const int32_t *>(tile_flag_index.const_data_ptr()),
            n_tiles,
            row_bytes,
            static_cast<int>(n_blocks),
            stream
        );
    }

    static void wait_tiles_torch_stub(torch::Tensor local_flag, const torch::Tensor &tile_valid) {
        at::cuda::OptionalCUDAGuard device_guard{device_of(local_flag)};
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        TORCH_CHECK(local_flag.dtype() == torch::kInt32, "local_flag must be int32");
        TORCH_CHECK(tile_valid.dtype() == torch::kInt32, "tile_valid must be int32");
        TORCH_CHECK(local_flag.numel() == tile_valid.numel(), "local_flag and tile_valid must have the same length");

        pi::launch_wait_tiles(
            static_cast<int32_t *>(local_flag.mutable_data_ptr()),
            static_cast<const int32_t *>(tile_valid.const_data_ptr()),
            local_flag.numel(),
            stream
        );
    }

    static void wait_and_reduce_torch_stub(
        const torch::Tensor &local_hidden,
        torch::Tensor local_flag,
        const torch::Tensor &routed_scores,
        torch::Tensor weighted_routed_out,
        const torch::Tensor &dispatch_peer_rank,
        const torch::Tensor &dispatch_local_row_start,
        const torch::Tensor &dispatch_own_tile_ordinal,
        const torch::Tensor &dispatch_valid_rows,
        int64_t block_m,
        int64_t n_blocks
    ) {
        at::cuda::OptionalCUDAGuard device_guard{device_of(local_hidden)};
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        TORCH_CHECK(local_hidden.dtype() == torch::kBFloat16, "local_hidden must be bf16");
        TORCH_CHECK(weighted_routed_out.dtype() == torch::kBFloat16, "weighted_routed_out must be bf16");
        TORCH_CHECK(routed_scores.dtype() == torch::kFloat32, "routed_scores must be fp32");
        TORCH_CHECK(local_hidden.is_contiguous(), "local_hidden must be contiguous");
        TORCH_CHECK(weighted_routed_out.is_contiguous(), "weighted_routed_out must be contiguous");
        TORCH_CHECK(local_flag.dtype() == torch::kInt32, "local_flag must be int32");
        int64_t dim = local_hidden.size(1);
        TORCH_CHECK(weighted_routed_out.size(1) == dim, "local_hidden and weighted_routed_out must share dim");
        int64_t n_dispatch_tiles = dispatch_peer_rank.numel();

        pi::launch_wait_and_reduce(
            local_hidden.const_data_ptr(),
            static_cast<int32_t *>(local_flag.mutable_data_ptr()),
            static_cast<const float *>(routed_scores.const_data_ptr()),
            weighted_routed_out.mutable_data_ptr(),
            static_cast<const int32_t *>(dispatch_peer_rank.const_data_ptr()),
            static_cast<const int32_t *>(dispatch_local_row_start.const_data_ptr()),
            static_cast<const int32_t *>(dispatch_own_tile_ordinal.const_data_ptr()),
            static_cast<const int32_t *>(dispatch_valid_rows.const_data_ptr()),
            n_dispatch_tiles,
            static_cast<int>(block_m),
            static_cast<int>(dim),
            static_cast<int>(n_blocks),
            stream
        );
    }
}

TORCH_LIBRARY_FRAGMENT(prime_comet_scatter, m) {
    m.def("scatter_tiles("
        "Tensor src, "
        "Tensor hidden_peer_ptrs, "
        "Tensor flag_peer_ptrs, "
        "Tensor tile_peer_rank, "
        "Tensor tile_local_row_start, "
        "Tensor tile_peer_row_start, "
        "Tensor tile_valid_rows, "
        "Tensor tile_flag_index, "
        "int n_blocks"
        ") -> ()"
    );
    m.impl("scatter_tiles", torch::kCUDA, &pi::scatter_tiles_torch_stub);

    m.def("wait_tiles(Tensor(a!) local_flag, Tensor tile_valid) -> ()");
    m.impl("wait_tiles", torch::kCUDA, &pi::wait_tiles_torch_stub);

    m.def("wait_and_reduce("
        "Tensor local_hidden, "
        "Tensor(a!) local_flag, "
        "Tensor routed_scores, "
        "Tensor(b!) weighted_routed_out, "
        "Tensor dispatch_peer_rank, "
        "Tensor dispatch_local_row_start, "
        "Tensor dispatch_own_tile_ordinal, "
        "Tensor dispatch_valid_rows, "
        "int block_m, "
        "int n_blocks"
        ") -> ()"
    );
    m.impl("wait_and_reduce", torch::kCUDA, &pi::wait_and_reduce_torch_stub);
}

PYBIND11_MODULE(_C, m) {}
