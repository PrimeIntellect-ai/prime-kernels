#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>

#include "rmsnorm.cuh"

namespace pi {
    // The kernel writes MXFP8 scales as a sequence of 512 byte tiles, one per
    // 128 row x 128 column block -- the layout `get_sf_out_offset` in
    // src/include/common.cuh addresses and a tensor core GEMM consumes. N and C
    // being multiples of 128 makes that buffer exactly covered, so it needs no
    // zeroing.
    static int64_t scale_numel(int64_t N, int64_t C) {
        return (N/128) * (C/128) * 512;
    }

    static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_rmsnorm_residual_torch_stub(
        const torch::Tensor &branch,
        const torch::Tensor &residual,
        const torch::Tensor &weight,
        double epsilon
    ) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        at::cuda::OptionalCUDAGuard device_guard {device_of(branch)};
        TORCH_CHECK(branch.dim() == 2, "branch must be a 2D (N, C) tensor");
        const int64_t N = branch.size(0);
        const int64_t C = branch.size(1);
        // The dispatcher asserts both, and NDEBUG compiles those asserts out.
        TORCH_CHECK((N&127) == 0, "N (", N, ") must be a multiple of 128");
        TORCH_CHECK((C&127) == 0, "C (", C, ") must be a multiple of 128");
        TORCH_CHECK(residual.sizes() == branch.sizes(), "residual must have the same shape as branch");
        TORCH_CHECK(weight.dim() == 1 && weight.size(0) == C, "weight must be a 1D (C,) tensor");
        TORCH_CHECK(branch.is_contiguous());
        TORCH_CHECK(residual.is_contiguous());
        TORCH_CHECK(weight.is_contiguous());
        TORCH_CHECK(branch.dtype() == torch::kBFloat16);
        TORCH_CHECK(residual.dtype() == torch::kBFloat16);
        TORCH_CHECK(weight.dtype() == torch::kBFloat16);
        TORCH_CHECK(residual.device() == branch.device() && weight.device() == branch.device());
        auto opts = torch::TensorOptions().device(branch.device());
        torch::Tensor vals = torch::empty({N, C}, opts.dtype(torch::kFloat8_e4m3fn));
        torch::Tensor scales = torch::empty({scale_numel(N, C)}, opts.dtype(torch::kFloat8_e8m0fnu));
        torch::Tensor rrms = torch::empty({N}, opts.dtype(torch::kFloat32));
        fused_rmsnorm_residual_forward(
            static_cast<__nv_fp8_e4m3 *>(vals.data_ptr()),
            static_cast<__nv_fp8_e8m0 *>(scales.data_ptr()),
            static_cast<float *>(rrms.data_ptr()),
            static_cast<const nv_bfloat16 *>(branch.const_data_ptr()),
            static_cast<const nv_bfloat16 *>(residual.const_data_ptr()),
            static_cast<const nv_bfloat16 *>(weight.const_data_ptr()),
            epsilon,
            N,
            C,
            stream
        );
        return {vals, scales, rrms};
    }
}

TORCH_LIBRARY_FRAGMENT(prime_rmsnorm, m) {
    m.def("fused_rmsnorm_residual("
        "Tensor branch, "
        "Tensor residual, "
        "Tensor weight, "
        "float epsilon"
        ") -> (Tensor, Tensor, Tensor)"
    );
    m.impl("fused_rmsnorm_residual", torch::kCUDA, &pi::fused_rmsnorm_residual_torch_stub);
}

PYBIND11_MODULE(_C, m) {}
