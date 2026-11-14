#include <ATen/ATen.h>
#include <torch/library.h>

#include <tuple>

namespace prime_kernels::nvfp4 {

std::tuple<at::Tensor, at::Tensor, at::Tensor>
quantize_activations_cuda(
    const at::Tensor& matrix,
    const at::Tensor& offsets);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
quantize_weights_cuda(const at::Tensor& weight_rows);

at::Tensor dequantize_activations_cuda(
    const at::Tensor& packed,
    const at::Tensor& block_scales,
    const at::Tensor& global_scales,
    const at::Tensor& offsets);

at::Tensor dequantize_weights_cuda(
    const at::Tensor& packed,
    const at::Tensor& block_scales,
    const at::Tensor& global_scales);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mxfp8_stochastic_bf16_cuda(
    const at::Tensor& values, int64_t seed, int64_t call_rows, int64_t call_cols, bool rows_wise, bool cols_wise);

std::tuple<at::Tensor, at::Tensor> mxfp8_stochastic_nvfp4_activations_cuda(
    const at::Tensor& packed, const at::Tensor& block_scales, const at::Tensor& global_scales,
    const at::Tensor& offsets, int64_t seed, int64_t call);

std::tuple<at::Tensor, at::Tensor> mxfp8_stochastic_nvfp4_weights_cuda(
    const at::Tensor& packed, const at::Tensor& block_scales, const at::Tensor& global_scales, int64_t seed,
    int64_t call);

TORCH_LIBRARY_FRAGMENT(prime_rl, m) {
  m.def(
      "quantize_nvfp4_activations(Tensor matrix, Tensor offsets) -> (Tensor, Tensor, Tensor)");
  m.def(
      "quantize_nvfp4_weights(Tensor weight_rows) -> (Tensor, Tensor, Tensor)");
  m.def(
      "dequantize_nvfp4_activations(Tensor packed, Tensor block_scales, "
      "Tensor global_scales, Tensor offsets) -> Tensor");
  m.def(
      "dequantize_nvfp4_weights(Tensor packed, Tensor block_scales, "
      "Tensor global_scales) -> Tensor");
  m.def(
      "mxfp8_stochastic_bf16(Tensor values, int seed, int call_rows, int call_cols, bool rows_wise, "
      "bool cols_wise) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "mxfp8_stochastic_nvfp4_activations(Tensor packed, Tensor block_scales, Tensor global_scales, "
      "Tensor offsets, int seed, int call) -> (Tensor, Tensor)");
  m.def(
      "mxfp8_stochastic_nvfp4_weights(Tensor packed, Tensor block_scales, Tensor global_scales, int seed, "
      "int call) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(prime_rl, CUDA, m) {
  m.impl("quantize_nvfp4_activations", quantize_activations_cuda);
  m.impl("quantize_nvfp4_weights", quantize_weights_cuda);
  m.impl("dequantize_nvfp4_activations", dequantize_activations_cuda);
  m.impl("dequantize_nvfp4_weights", dequantize_weights_cuda);
  m.impl("mxfp8_stochastic_bf16", mxfp8_stochastic_bf16_cuda);
  m.impl("mxfp8_stochastic_nvfp4_activations", mxfp8_stochastic_nvfp4_activations_cuda);
  m.impl("mxfp8_stochastic_nvfp4_weights", mxfp8_stochastic_nvfp4_weights_cuda);
}

} // namespace prime_kernels::nvfp4
