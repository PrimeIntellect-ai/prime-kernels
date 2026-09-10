#include <ATen/ATen.h>
#include <torch/library.h>

#include <tuple>

namespace prime_kernels::nvfp4 {

std::tuple<at::Tensor, at::Tensor, at::Tensor>
quantize_activations_cuda(
    const at::Tensor& matrix,
    const at::Tensor& offsets,
    bool four_over_six);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
quantize_weights_cuda(const at::Tensor& weight_rows, bool four_over_six);

at::Tensor dequantize_activations_cuda(
    const at::Tensor& packed,
    const at::Tensor& block_scales,
    const at::Tensor& global_scales,
    const at::Tensor& offsets);

at::Tensor dequantize_weights_cuda(
    const at::Tensor& packed,
    const at::Tensor& block_scales,
    const at::Tensor& global_scales);

TORCH_LIBRARY_FRAGMENT(prime_rl, m) {
  m.def(
      "quantize_nvfp4_activations(Tensor matrix, Tensor offsets, bool four_over_six=False) -> "
      "(Tensor, Tensor, Tensor)");
  m.def(
      "quantize_nvfp4_weights(Tensor weight_rows, bool four_over_six=False) -> "
      "(Tensor, Tensor, Tensor)");
  m.def(
      "dequantize_nvfp4_activations(Tensor packed, Tensor block_scales, "
      "Tensor global_scales, Tensor offsets) -> Tensor");
  m.def(
      "dequantize_nvfp4_weights(Tensor packed, Tensor block_scales, "
      "Tensor global_scales) -> Tensor");
}

TORCH_LIBRARY_IMPL(prime_rl, CUDA, m) {
  m.impl("quantize_nvfp4_activations", quantize_activations_cuda);
  m.impl("quantize_nvfp4_weights", quantize_weights_cuda);
  m.impl("dequantize_nvfp4_activations", dequantize_activations_cuda);
  m.impl("dequantize_nvfp4_weights", dequantize_weights_cuda);
}

} // namespace prime_kernels::nvfp4
