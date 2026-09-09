#include "nvfp4_grouped.h"

#include <torch/library.h>

namespace prime_kernels::nvfp4 {

at::Tensor grouped_mm(
    const at::Tensor& activations,
    const at::Tensor& weight,
    const at::Tensor& activation_block_scales,
    const at::Tensor& weight_block_scales,
    const at::Tensor& offsets,
    const at::Tensor& activation_token_scales,
    const at::Tensor& weight_expert_scales) {
  return f4f4bf16_ultra_grouped_mm(
      activations,
      weight,
      activation_block_scales,
      weight_block_scales,
      offsets,
      activation_token_scales,
      weight_expert_scales);
}

TORCH_LIBRARY_FRAGMENT(prime_rl, m) {
  m.def(
      "grouped_nvfp4_gemm(Tensor activations, Tensor weight, "
      "Tensor activation_block_scales, Tensor weight_block_scales, "
      "Tensor offsets, Tensor activation_token_scales, "
      "Tensor weight_expert_scales) -> Tensor");
}

TORCH_LIBRARY_IMPL(prime_rl, CUDA, m) {
  m.impl("grouped_nvfp4_gemm", grouped_mm);
}

} // namespace prime_kernels::nvfp4
