/*
 * Portions of the Blackwell conversion and 4/6 quantization are adapted from
 * NVIDIA Transformer Engine, Copyright NVIDIA Corporation & affiliates, and
 * FlashInfer, Copyright (c) 2025 by FlashInfer team, under Apache License 2.0.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <type_traits>

namespace prime_kernels::nvfp4 {
namespace {

constexpr int kSfVectorSize = 16;
constexpr int kScaleRowTile = 128;
constexpr int kScaleColTile = 4;
constexpr int kQuantThreads = 256;
constexpr int kQuantScaleColsPerCta = 64;
constexpr int kQuantScaleTilesPerCta =
    kQuantScaleColsPerCta / kScaleColTile;
constexpr int kQuantScaleTileElements =
    kScaleRowTile * kQuantScaleColsPerCta;
constexpr int kQuantItemsPerThread =
    kQuantScaleTileElements / kQuantThreads;
constexpr int kDequantScaleColsPerCta = 32;
constexpr int kDequantScaleTilesPerCta =
    kDequantScaleColsPerCta / kScaleColTile;
constexpr int kDequantScaleTileElements =
    kScaleRowTile * kDequantScaleColsPerCta;
constexpr int kDequantItemsPerThread =
    kDequantScaleTileElements / kQuantThreads;
constexpr int kReductionThreads = 256;
constexpr float kGlobalScaleDenominator = 6.0f * 448.0f;

__host__ __device__ __forceinline__ int64_t round_up(
    int64_t value,
    int64_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

void check_bf16_matrix(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor.");
  TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be BF16.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

void check_quantized_tensors(
    const at::Tensor& packed,
    const at::Tensor& block_scales,
    const at::Tensor& global_scales) {
  TORCH_CHECK(packed.is_cuda(), "packed must be a CUDA tensor.");
  TORCH_CHECK(packed.scalar_type() == at::kByte, "packed must use byte storage.");
  TORCH_CHECK(packed.is_contiguous(), "packed must be contiguous.");
  TORCH_CHECK(
      block_scales.device() == packed.device() &&
          global_scales.device() == packed.device(),
      "quantized tensors must be on the same CUDA device.");
  TORCH_CHECK(
      block_scales.scalar_type() == at::kFloat8_e4m3fn,
      "block_scales must be FP8 E4M3.");
  TORCH_CHECK(
      global_scales.scalar_type() == at::kFloat,
      "global_scales must be FP32.");
  TORCH_CHECK(
      block_scales.is_contiguous() && global_scales.is_contiguous(),
      "quantized tensors must be contiguous.");
}

struct alignas(16) Bf16Block {
  uint4 vectors[2];
};

union FP4Block {
  uint64_t packed;
  __nv_fp4x4_e2m1 values[4];
};

__device__ __forceinline__ Bf16Block load_bf16_block(
    const __nv_bfloat16* input) {
  Bf16Block block;
  const auto* vectors = reinterpret_cast<const uint4*>(input);
  block.vectors[0] = __ldcg(vectors);
  block.vectors[1] = __ldcg(vectors + 1);
  return block;
}

__device__ __forceinline__ float block_amax(const Bf16Block& block) {
  const auto* pairs = reinterpret_cast<const uint32_t*>(block.vectors);
  uint32_t maximum = 0;
#pragma unroll
  for (int index = 0; index < 8; ++index) {
    asm volatile(
        "max.xorsign.abs.bf16x2 %0, %1, %2;"
        : "=r"(maximum)
        : "r"(maximum), "r"(pairs[index]));
  }
  const float2 values =
      __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&maximum));
  return fmaxf(fabsf(values.x), fabsf(values.y));
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return value;
}

__device__ __forceinline__ float block_max(
    float value,
    float* warp_maxima) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_max(value);
  if (lane == 0) {
    warp_maxima[warp] = value;
  }
  __syncthreads();
  if (warp == 0) {
    value = lane < (blockDim.x >> 5) ? warp_maxima[lane] : 0.0f;
    value = warp_max(value);
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

// Adapted from Transformer Engine's Blackwell BF16-to-E2M1 conversion. Keeping
// the coefficient in FP32 avoids an extra recipe-changing BF16 rounding.
__device__ __forceinline__ uint32_t pack_e2m1_8(
    uint64_t values_03,
    uint64_t values_47,
    float coefficient) {
  uint32_t output;
  asm volatile(
      "{\n"
      ".reg.b64 coefficient_2x;\n\t"
      "mov.b64 coefficient_2x, {%3, %3};\n\t"
      ".reg.b16 h0, h1, h2, h3, h4, h5, h6, h7;\n\t"
      "mov.b64 {h0, h1, h2, h3}, %1;\n\t"
      "mov.b64 {h4, h5, h6, h7}, %2;\n\t"
      ".reg.b32 v0, v1, v2, v3, v4, v5, v6, v7;\n\t"
      "cvt.f32.bf16 v0, h0;\n\t"
      "cvt.f32.bf16 v1, h1;\n\t"
      "cvt.f32.bf16 v2, h2;\n\t"
      "cvt.f32.bf16 v3, h3;\n\t"
      "cvt.f32.bf16 v4, h4;\n\t"
      "cvt.f32.bf16 v5, h5;\n\t"
      "cvt.f32.bf16 v6, h6;\n\t"
      "cvt.f32.bf16 v7, h7;\n\t"
      ".reg.b64 v01, v23, v45, v67;\n\t"
      "mov.b64 v01, {v0, v1};\n\t"
      "mov.b64 v23, {v2, v3};\n\t"
      "mov.b64 v45, {v4, v5};\n\t"
      "mov.b64 v67, {v6, v7};\n\t"
      "mul.f32x2 v01, v01, coefficient_2x;\n\t"
      "mul.f32x2 v23, v23, coefficient_2x;\n\t"
      "mul.f32x2 v45, v45, coefficient_2x;\n\t"
      "mul.f32x2 v67, v67, coefficient_2x;\n\t"
      "mov.b64 {v1, v0}, v01;\n\t"
      "mov.b64 {v3, v2}, v23;\n\t"
      "mov.b64 {v5, v4}, v45;\n\t"
      "mov.b64 {v7, v6}, v67;\n\t"
      ".reg.b8 f0, f1, f2, f3;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f0, v0, v1;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f1, v2, v3;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f2, v4, v5;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f3, v6, v7;\n\t"
      "mov.b32 %0, {f0, f1, f2, f3};\n\t"
      "}"
      : "=r"(output)
      : "l"(values_03), "l"(values_47), "f"(coefficient));
  return output;
}

__device__ __forceinline__ uint64_t pack_e2m1(
    const Bf16Block& block,
    float coefficient) {
  const auto* words = reinterpret_cast<const uint64_t*>(block.vectors);
  const uint32_t low = pack_e2m1_8(words[0], words[1], coefficient);
  const uint32_t high = pack_e2m1_8(words[2], words[3], coefficient);
  return static_cast<uint64_t>(low) |
      (static_cast<uint64_t>(high) << 32);
}

__device__ __forceinline__ Bf16Block dequantize_e2m1(
    uint64_t packed,
    float decode_scale) {
  FP4Block input;
  input.packed = packed;
  Bf16Block output;
  auto* output_pairs = reinterpret_cast<__nv_bfloat162*>(output.vectors);
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    const float4 values = static_cast<float4>(input.values[index]);
    output_pairs[index * 2] = __floats2bfloat162_rn(
        values.x * decode_scale,
        values.y * decode_scale);
    output_pairs[index * 2 + 1] = __floats2bfloat162_rn(
        values.z * decode_scale,
        values.w * decode_scale);
  }
  return output;
}

// IEEE-rounded reciprocals in the block math: Composer 2 found approximate division in NVFP4 quantization made RL
// diverge.
__device__ __forceinline__ float reciprocal(float value) {
  return __frcp_rn(value);
}

template <int kScaleCols>
__device__ __forceinline__ int local_scale_offset(
    int row_in_tile,
    int scale_column_in_cta) {
  return (scale_column_in_cta >> 2) * (kScaleRowTile * kScaleColTile) +
      ((row_in_tile & 31) << 4) |
      (((row_in_tile >> 5) & 3) << 2) |
      (scale_column_in_cta & 3);
}

__device__ __forceinline__ uint8_t quantize_block(
    const Bf16Block& input,
    float global_decode_scale,
    uint64_t* packed) {
  const float local_amax = block_amax(input);
  float scale_value = 0.0f;
  if (global_decode_scale > 0.0f && local_amax > 0.0f) {
    scale_value =
        local_amax * reciprocal(6.0f * global_decode_scale);
  }
  const __nv_fp8_e4m3 fp8_scale(scale_value);
  const float decode_scale =
      static_cast<float>(fp8_scale) * global_decode_scale;
  const float coefficient =
      decode_scale > 0.0f ? reciprocal(decode_scale) : 0.0f;
  *packed = pack_e2m1(input, coefficient);
  return fp8_scale.__x;
}

__device__ __forceinline__ float four_over_six_error(
    const Bf16Block& input,
    uint64_t packed,
    float block_scale,
    float global_amax) {
  FP4Block candidate;
  candidate.packed = packed;
  const auto* values = reinterpret_cast<const __nv_bfloat16*>(input.vectors);
  float error = 0.0f;
#pragma unroll
  for (int chunk = 0; chunk < 4; ++chunk) {
    const float4 decoded = static_cast<float4>(candidate.values[chunk]);
    const float elements[4] = {decoded.x, decoded.y, decoded.z, decoded.w};
#pragma unroll
    for (int index = 0; index < 4; ++index) {
      const float reconstructed = __fdiv_rn(
          __fmul_rn(__fmul_rn(elements[index], block_scale), global_amax),
          kGlobalScaleDenominator);
      const float difference = __fsub_rn(
          reconstructed, __bfloat162float(values[chunk * 4 + index]));
      error = __fadd_rn(error, fabsf(difference));
    }
  }
  return error;
}

// Activations only (weights never use 4/6): the stored decode scale is the row's amax / 2688, scored against the row
// amax.
__device__ __forceinline__ uint8_t quantize_block_four_over_six(
    const Bf16Block& input,
    float global_amax,
    float stored_decode_scale,
    uint64_t* packed) {
  const float local_amax = block_amax(input);
  if (local_amax == 0.0f) {
    *packed = 0;
    return 0;
  }
  const float encode_scale = reciprocal(stored_decode_scale);
  const float decode_scale = reciprocal(encode_scale);
  const float scale6_value = __fmul_rn(
      encode_scale, __fmul_rn(local_amax, reciprocal(6.0f)));
  const __nv_fp8_e4m3 scale6(scale6_value);
  const __nv_fp8_e4m3 scale4(__fmul_rn(scale6_value, 1.5f));
  const float sf6 = static_cast<float>(scale6);
  const float sf4 = static_cast<float>(scale4);
  const uint64_t packed6 = pack_e2m1(
      input, reciprocal(__fmul_rn(sf6, decode_scale)));
  const uint64_t packed4 = pack_e2m1(
      input, reciprocal(__fmul_rn(sf4, decode_scale)));
  const float error6 = four_over_six_error(input, packed6, sf6, global_amax);
  const float error4 = four_over_six_error(input, packed4, sf4, global_amax);
  const bool use_four = error4 < error6;
  *packed = use_four ? packed4 : packed6;
  return use_four ? scale4.__x : scale6.__x;
}

__device__ __forceinline__ bool activation_tile_metadata(
    int tile,
    const int32_t* offsets,
    int group_count,
    int* global_row_start,
    int* valid_rows) {
  int group_start = 0;
  int tile_start = 0;
  for (int group = 0; group < group_count; ++group) {
    const int group_end = offsets[group];
    const int group_rows = group_end - group_start;
    const int group_tiles =
        (group_rows + kScaleRowTile - 1) / kScaleRowTile;
    if (tile < tile_start + group_tiles) {
      const int tile_in_group = tile - tile_start;
      *global_row_start =
          group_start + tile_in_group * kScaleRowTile;
      *valid_rows = min(
          kScaleRowTile,
          group_end - *global_row_start);
      return true;
    }
    group_start = group_end;
    tile_start += group_tiles;
  }
  return false;
}

__global__ void activation_amax_kernel(
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ global_scales,
    float* __restrict__ row_amax,
    int rows,
    int contraction_size) {
  __shared__ float warp_maxima[kReductionThreads / 32];
  const int scale_columns = contraction_size / kSfVectorSize;
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    float local_amax = 0.0f;
    for (int scale_column = threadIdx.x;
         scale_column < scale_columns;
         scale_column += blockDim.x) {
      const int64_t input_offset =
          static_cast<int64_t>(row) * contraction_size +
          scale_column * kSfVectorSize;
      local_amax = fmaxf(
          local_amax,
          block_amax(load_bf16_block(input + input_offset)));
    }
    const float maximum = block_max(local_amax, warp_maxima);
    if (threadIdx.x == 0) {
      row_amax[row] = maximum;
      global_scales[row] = __fmul_rn(maximum, 1.0f / kGlobalScaleDenominator);
    }
    __syncthreads();
  }
}

__global__ void weight_amax_kernel(
    const __nv_bfloat16* __restrict__ input,
    float* __restrict__ expert_amax,
    int64_t elements_per_expert,
    int blocks_per_expert) {
  __shared__ float warp_maxima[kReductionThreads / 32];
  const int expert = blockIdx.x / blocks_per_expert;
  const int expert_block = blockIdx.x % blocks_per_expert;
  const int64_t scale_blocks = elements_per_expert / kSfVectorSize;
  float local_amax = 0.0f;
  for (int64_t scale_block =
           static_cast<int64_t>(expert_block) * blockDim.x + threadIdx.x;
       scale_block < scale_blocks;
       scale_block += static_cast<int64_t>(blocks_per_expert) * blockDim.x) {
    const int64_t input_offset =
        static_cast<int64_t>(expert) * elements_per_expert +
        scale_block * kSfVectorSize;
    local_amax = fmaxf(
        local_amax,
        block_amax(load_bf16_block(input + input_offset)));
  }
  const float maximum = block_max(local_amax, warp_maxima);
  if (threadIdx.x == 0) {
    atomicMax(
        reinterpret_cast<unsigned int*>(expert_amax + expert),
        __float_as_uint(maximum));
  }
}

__global__ void finalize_weight_scales_kernel(
    const float* __restrict__ expert_amax,
    float* __restrict__ global_scales,
    int groups) {
  const int expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert < groups) {
    const float maximum = expert_amax[expert];
    global_scales[expert] =
        maximum > 0.0f ? __fdiv_rn(maximum, kGlobalScaleDenominator) : 0.0f;
  }
}

__global__ void quantize_activations_tiled_kernel(
    const __nv_bfloat16* __restrict__ input,
    const int32_t* __restrict__ offsets,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ block_scales,
    const float* __restrict__ global_scales,
    const float* __restrict__ row_amax,
    int contraction_size,
    int group_count,
    int scale_column_tiles) {
  __shared__ alignas(16) uint8_t scale_tile[kQuantScaleTileElements];
  __shared__ int global_row_start;
  __shared__ int valid_rows;
  __shared__ int active;

  if (threadIdx.x == 0) {
    active = activation_tile_metadata(
        blockIdx.x,
        offsets,
        group_count,
        &global_row_start,
        &valid_rows);
  }
  __syncthreads();
  if (!active) {
    return;
  }

#pragma unroll
  for (int item = 0; item < kQuantItemsPerThread; ++item) {
    const int tile_index = threadIdx.x + item * kQuantThreads;
    const int row_in_tile = tile_index / kQuantScaleColsPerCta;
    const int column_in_tile = tile_index % kQuantScaleColsPerCta;
    const int scale_column =
        blockIdx.y * kQuantScaleColsPerCta + column_in_tile;
    uint8_t scale = 0;
    if (row_in_tile < valid_rows &&
        scale_column < contraction_size / kSfVectorSize) {
      const int row = global_row_start + row_in_tile;
      const int64_t input_offset =
          static_cast<int64_t>(row) * contraction_size +
          scale_column * kSfVectorSize;
      uint64_t packed_value;
      scale = quantize_block_four_over_six(
          load_bf16_block(input + input_offset),
          row_amax[row],
          global_scales[row],
          &packed_value);
      const int64_t packed_offset =
          static_cast<int64_t>(row) * (contraction_size / 2) +
          scale_column * 8;
      *reinterpret_cast<uint64_t*>(packed + packed_offset) =
          packed_value;
    }
    scale_tile[local_scale_offset<kQuantScaleColsPerCta>(
        row_in_tile,
        column_in_tile)] =
        scale;
  }
  __syncthreads();

  const int first_scale_tile =
      blockIdx.y * kQuantScaleTilesPerCta;
  const int valid_scale_tiles =
      min(
          kQuantScaleTilesPerCta,
          scale_column_tiles - first_scale_tile);
  const int scale_vectors =
      valid_scale_tiles * kScaleRowTile * kScaleColTile /
      static_cast<int>(sizeof(uint4));
  for (int vector = threadIdx.x;
       vector < scale_vectors;
       vector += blockDim.x) {
    const int64_t tile_offset =
        (static_cast<int64_t>(blockIdx.x) * scale_column_tiles +
         first_scale_tile) *
        (kScaleRowTile * kScaleColTile);
    reinterpret_cast<uint4*>(block_scales + tile_offset)[vector] =
        reinterpret_cast<const uint4*>(scale_tile)[vector];
  }
}

__global__ void quantize_weights_tiled_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ global_scales,
    uint8_t* __restrict__ packed,
    uint8_t* __restrict__ block_scales,
    int output_size,
    int contraction_size,
    int output_row_tiles,
    int scale_column_tiles) {
  __shared__ alignas(16) uint8_t scale_tile[kQuantScaleTileElements];
  const int expert = blockIdx.x / output_row_tiles;
  const int output_tile = blockIdx.x % output_row_tiles;
  const int output_row_start = output_tile * kScaleRowTile;

#pragma unroll
  for (int item = 0; item < kQuantItemsPerThread; ++item) {
    const int tile_index = threadIdx.x + item * kQuantThreads;
    const int row_in_tile = tile_index / kQuantScaleColsPerCta;
    const int column_in_tile = tile_index % kQuantScaleColsPerCta;
    const int output_row = output_row_start + row_in_tile;
    const int scale_column =
        blockIdx.y * kQuantScaleColsPerCta + column_in_tile;
    uint8_t scale = 0;
    if (output_row < output_size &&
        scale_column < contraction_size / kSfVectorSize) {
      const int64_t row =
          static_cast<int64_t>(expert) * output_size + output_row;
      const int64_t input_offset =
          row * contraction_size + scale_column * kSfVectorSize;
      uint64_t packed_value;
      scale = quantize_block(
          load_bf16_block(input + input_offset),
          global_scales[expert],
          &packed_value);
      const int64_t packed_offset =
          row * (contraction_size / 2) + scale_column * 8;
      *reinterpret_cast<uint64_t*>(packed + packed_offset) =
          packed_value;
    }
    scale_tile[local_scale_offset<kQuantScaleColsPerCta>(
        row_in_tile,
        column_in_tile)] =
        scale;
  }
  __syncthreads();

  const int first_scale_tile =
      blockIdx.y * kQuantScaleTilesPerCta;
  const int valid_scale_tiles =
      min(
          kQuantScaleTilesPerCta,
          scale_column_tiles - first_scale_tile);
  const int scale_vectors =
      valid_scale_tiles * kScaleRowTile * kScaleColTile /
      static_cast<int>(sizeof(uint4));
  for (int vector = threadIdx.x;
       vector < scale_vectors;
       vector += blockDim.x) {
    const int64_t tile_offset =
        (static_cast<int64_t>(blockIdx.x) * scale_column_tiles +
         first_scale_tile) *
        (kScaleRowTile * kScaleColTile);
    reinterpret_cast<uint4*>(block_scales + tile_offset)[vector] =
        reinterpret_cast<const uint4*>(scale_tile)[vector];
  }
}

__global__ void dequantize_activations_tiled_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ block_scales,
    const float* __restrict__ global_scales,
    const int32_t* __restrict__ offsets,
    __nv_bfloat16* __restrict__ output,
    int contraction_size,
    int group_count,
    int scale_column_tiles) {
  __shared__ alignas(16) uint8_t scale_tile[kDequantScaleTileElements];
  __shared__ int global_row_start;
  __shared__ int valid_rows;
  __shared__ int active;

  if (threadIdx.x == 0) {
    active = activation_tile_metadata(
        blockIdx.x,
        offsets,
        group_count,
        &global_row_start,
        &valid_rows);
  }
  const int first_scale_tile =
      blockIdx.y * kDequantScaleTilesPerCta;
  const int valid_scale_tiles =
      min(
          kDequantScaleTilesPerCta,
          scale_column_tiles - first_scale_tile);
  if (threadIdx.x <
      valid_scale_tiles * kScaleRowTile * kScaleColTile /
          static_cast<int>(sizeof(uint4))) {
    const int64_t tile_offset =
        (static_cast<int64_t>(blockIdx.x) * scale_column_tiles +
         first_scale_tile) *
        (kScaleRowTile * kScaleColTile);
    reinterpret_cast<uint4*>(scale_tile)[threadIdx.x] =
        reinterpret_cast<const uint4*>(
            block_scales + tile_offset)[threadIdx.x];
  }
  __syncthreads();
  if (!active) {
    return;
  }

#pragma unroll
  for (int item = 0; item < kDequantItemsPerThread; ++item) {
    const int tile_index = threadIdx.x + item * kQuantThreads;
    const int row_in_tile = tile_index / kDequantScaleColsPerCta;
    const int column_in_tile = tile_index % kDequantScaleColsPerCta;
    const int scale_column =
        blockIdx.y * kDequantScaleColsPerCta + column_in_tile;
    if (row_in_tile < valid_rows &&
        scale_column < contraction_size / kSfVectorSize) {
      const int row = global_row_start + row_in_tile;
      const int64_t packed_offset =
          static_cast<int64_t>(row) * (contraction_size / 2) +
          scale_column * 8;
      const uint64_t packed_value =
          *reinterpret_cast<const uint64_t*>(packed + packed_offset);
      const uint8_t scale_bits =
          scale_tile[local_scale_offset<kDequantScaleColsPerCta>(
              row_in_tile,
              column_in_tile)];
      __nv_fp8_e4m3 scale;
      scale.__x = scale_bits;
      const float decode_scale =
          static_cast<float>(scale) * global_scales[row];
      const Bf16Block values =
          dequantize_e2m1(packed_value, decode_scale);
      const int64_t output_offset =
          static_cast<int64_t>(row) * contraction_size +
          scale_column * kSfVectorSize;
      auto* output_vectors =
          reinterpret_cast<uint4*>(output + output_offset);
      output_vectors[0] = values.vectors[0];
      output_vectors[1] = values.vectors[1];
    }
  }
}

__global__ void dequantize_weights_tiled_kernel(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ block_scales,
    const float* __restrict__ global_scales,
    __nv_bfloat16* __restrict__ output,
    int output_size,
    int contraction_size,
    int output_row_tiles,
    int scale_column_tiles) {
  __shared__ alignas(16) uint8_t scale_tile[kDequantScaleTileElements];
  const int expert = blockIdx.x / output_row_tiles;
  const int output_tile = blockIdx.x % output_row_tiles;
  const int output_row_start = output_tile * kScaleRowTile;

  const int first_scale_tile =
      blockIdx.y * kDequantScaleTilesPerCta;
  const int valid_scale_tiles =
      min(
          kDequantScaleTilesPerCta,
          scale_column_tiles - first_scale_tile);
  if (threadIdx.x <
      valid_scale_tiles * kScaleRowTile * kScaleColTile /
          static_cast<int>(sizeof(uint4))) {
    const int64_t tile_offset =
        (static_cast<int64_t>(blockIdx.x) * scale_column_tiles +
         first_scale_tile) *
        (kScaleRowTile * kScaleColTile);
    reinterpret_cast<uint4*>(scale_tile)[threadIdx.x] =
        reinterpret_cast<const uint4*>(
            block_scales + tile_offset)[threadIdx.x];
  }
  __syncthreads();

#pragma unroll
  for (int item = 0; item < kDequantItemsPerThread; ++item) {
    const int tile_index = threadIdx.x + item * kQuantThreads;
    const int row_in_tile = tile_index / kDequantScaleColsPerCta;
    const int column_in_tile = tile_index % kDequantScaleColsPerCta;
    const int output_row = output_row_start + row_in_tile;
    const int scale_column =
        blockIdx.y * kDequantScaleColsPerCta + column_in_tile;
    if (output_row < output_size &&
        scale_column < contraction_size / kSfVectorSize) {
      const int64_t row =
          static_cast<int64_t>(expert) * output_size + output_row;
      const int64_t packed_offset =
          row * (contraction_size / 2) + scale_column * 8;
      const uint64_t packed_value =
          *reinterpret_cast<const uint64_t*>(packed + packed_offset);
      const uint8_t scale_bits =
          scale_tile[local_scale_offset<kDequantScaleColsPerCta>(
              row_in_tile,
              column_in_tile)];
      __nv_fp8_e4m3 scale;
      scale.__x = scale_bits;
      const float decode_scale =
          static_cast<float>(scale) * global_scales[expert];
      const Bf16Block values =
          dequantize_e2m1(packed_value, decode_scale);
      const int64_t output_offset =
          row * contraction_size + scale_column * kSfVectorSize;
      auto* output_vectors =
          reinterpret_cast<uint4*>(output + output_offset);
      output_vectors[0] = values.vectors[0];
      output_vectors[1] = values.vectors[1];
    }
  }
}

int weight_amax_blocks_per_expert(
    int device,
    int groups,
    int64_t scale_blocks_per_expert) {
  const auto* properties = at::cuda::getDeviceProperties(device);
  const int target = std::max(
      1,
      (properties->multiProcessorCount * 4 + groups - 1) / groups);
  const int available = std::max<int64_t>(
      1,
      (scale_blocks_per_expert + kReductionThreads - 1) /
          kReductionThreads);
  return std::min(256, std::min(target, available));
}

int activation_amax_threads(int contraction_size) {
  const int scale_columns = contraction_size / kSfVectorSize;
  return std::min(
      kReductionThreads,
      std::max(32, static_cast<int>(round_up(scale_columns, 32))));
}

int activation_amax_blocks(
    int device,
    int rows,
    int threads,
    int contraction_size) {
  if (contraction_size >= 2688) {
    return rows;
  }
  const auto* properties = at::cuda::getDeviceProperties(device);
  const int blocks_per_sm = std::max(1, 1024 / threads);
  return std::min(rows, properties->multiProcessorCount * blocks_per_sm);
}


// ── MXFP8 with stochastic rounding, for the backward's gradient GEMMs ────────
//
// One pass per operand: a 32 x 128 tile of the source (BF16, or an NVFP4 operand dequantized exactly as
// dequantize_e2m1 does) is staged in shared memory, then quantized to MXFP8 (E4M3 values, E8M0 scales per 32) along
// its rows and/or its columns, so a gradient's row-wise and column-wise forms come from one read and neither needs a
// transpose pass. Scales are torchao's RCEIL scales, computed exactly from the float bits. Rounding is stochastic:
// each value rounds down or up to its E4M3 neighbours with probability in proportion to its distance from the other,
// from 16 random bits drawn by Philox4x32-10 keyed by the seed and counted by (output index / 8, call). stochastic.py
// holds the same algorithm in PyTorch as the reference, so both give the same bits for the same seed and call.

constexpr int kMxBlock = 32;
constexpr int kSrTileRows = 32;
constexpr int kSrTileCols = 128;
constexpr int kSrPitch = kSrTileCols + 8;

__device__ __forceinline__ uint4 philox4x32_10(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3, uint32_t k0,
                                               uint32_t k1) {
#pragma unroll
  for (int round = 0; round < 10; ++round) {
    const uint32_t lo0 = 0xD2511F53u * c0, hi0 = __umulhi(0xD2511F53u, c0);
    const uint32_t lo1 = 0xCD9E8D57u * c2, hi1 = __umulhi(0xCD9E8D57u, c2);
    c0 = hi1 ^ c1 ^ k0;
    c1 = lo1;
    c2 = hi0 ^ c3 ^ k1;
    c3 = lo0;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  return make_uint4(c0, c1, c2, c3);
}

// 16 random 16-bit draws for the outputs [start, start + 16), start a multiple of 16
__device__ __forceinline__ void random16x16(int64_t start, uint32_t call, uint32_t seed_lo, uint32_t seed_hi,
                                            uint32_t* draws) {
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const uint64_t counter = static_cast<uint64_t>(start) / 8 + half;
    const uint4 words = philox4x32_10(static_cast<uint32_t>(counter), static_cast<uint32_t>(counter >> 32), call,
                                      0u, seed_lo, seed_hi);
    const uint32_t word[4] = {words.x, words.y, words.z, words.w};
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      draws[half * 8 + index] = (word[index / 2] >> (16 * (index % 2))) & 0xFFFFu;
    }
  }
}

// torchao's RCEIL scale: ceil(log2(amax / 448)) + 127, clamped to [0, 254] (255 for NaN)
__device__ __forceinline__ int e8m0_rceil(float amax) {
  const uint32_t bits = __float_as_uint(__fdiv_rn(amax, 448.0f));
  const int exponent = (bits >> 23) & 0xff;
  const uint32_t mantissa = bits & 0x7fffff;
  if (exponent == 0xff) {
    return mantissa ? 255 : 254;
  }
  if (exponent == 0 && mantissa == 0) {
    return 0;
  }
  const int ceil_log2 = exponent ? exponent - 127 + (mantissa != 0) : -149 + (32 - __clz(mantissa - 1));
  return min(max(ceil_log2, -127), 127) + 127;
}

__device__ __forceinline__ float e8m0_reciprocal(int exponent) {
  if (exponent == 0) {
    return 1.0f;
  }
  if (exponent == 254) {
    return __uint_as_float(1u << 22);  // 2^-127
  }
  if (exponent == 255) {
    return __uint_as_float(0x7fc00000u);
  }
  return __uint_as_float(static_cast<uint32_t>(254 - exponent) << 23);
}

__device__ __forceinline__ uint8_t stochastic_e4m3(float value, float reciprocal_scale, uint32_t draw) {
  float scaled = value * reciprocal_scale;
  if (scaled != scaled) {
    return 0x7f;
  }
  scaled = scaled > 448.0f ? 448.0f : (scaled < -448.0f ? -448.0f : scaled);
  const float magnitude = fabsf(scaled);
  // The E4M3 step of the value's binade (subnormal below 2^-6)
  const int exponent = max(static_cast<int>((__float_as_uint(magnitude) >> 23) & 0xff) - 127, -6);
  const float step = __uint_as_float(static_cast<uint32_t>(exponent - 3 + 127) << 23);
  const float inverse = __uint_as_float(static_cast<uint32_t>(127 - (exponent - 3)) << 23);
  const float noise = static_cast<float>(draw) * 0x1p-16f;
  const float rounded = fminf(floorf(__fadd_rn(__fmul_rn(magnitude, inverse), noise)) * step, 448.0f);
  return __nv_cvt_float_to_fp8(copysignf(rounded, scaled), __NV_SATFINITE, __NV_E4M3);
}

// Sources: write 16 consecutive values (row, col .. col + 16) of matrix z as BF16 into dst
struct Bf16Source {
  const __nv_bfloat16* values;
  int64_t rows, cols;
  __device__ void load16(int z, int row, int col, __nv_bfloat16* dst) const {
    const auto* src = reinterpret_cast<const uint4*>(values + (static_cast<int64_t>(z) * rows + row) * cols + col);
    reinterpret_cast<uint4*>(dst)[0] = __ldcg(src);
    reinterpret_cast<uint4*>(dst)[1] = __ldcg(src + 1);
  }
};

__device__ __forceinline__ int64_t swizzled_scale_offset(int64_t tile, int row_in_tile, int scale_col,
                                                         int scale_column_tiles) {
  return (tile * scale_column_tiles + (scale_col >> 2)) * (kScaleRowTile * kScaleColTile) +
         (((row_in_tile & 31) << 4) | (((row_in_tile >> 5) & 3) << 2) | (scale_col & 3));
}

__device__ __forceinline__ void store_dequantized(uint64_t packed, uint8_t scale_bits, float global_scale,
                                                  __nv_bfloat16* dst) {
  __nv_fp8_e4m3 scale;
  scale.__x = scale_bits;
  const Bf16Block values = dequantize_e2m1(packed, static_cast<float>(scale) * global_scale);
  reinterpret_cast<uint4*>(dst)[0] = values.vectors[0];
  reinterpret_cast<uint4*>(dst)[1] = values.vectors[1];
}

// Activations [M, K] with per-token global scales and block scales in row tiles padded per group
struct NVFP4ActivationSource {
  const uint8_t* packed;
  const uint8_t* block_scales;
  const float* global_scales;
  const int32_t* offsets;
  int groups, contraction, scale_column_tiles;
  __device__ void load16(int, int row, int col, __nv_bfloat16* dst) const {
    int group_start = 0, tile_start = 0;
    for (int group = 0; group < groups; ++group) {
      const int group_end = offsets[group];
      if (row < group_end) {
        break;
      }
      tile_start += (group_end - group_start + kScaleRowTile - 1) / kScaleRowTile;
      group_start = group_end;
    }
    const int row_in_group = row - group_start;
    const int64_t tile = tile_start + row_in_group / kScaleRowTile;
    const uint64_t value =
        *reinterpret_cast<const uint64_t*>(packed + static_cast<int64_t>(row) * (contraction / 2) + col / 2);
    const uint8_t scale =
        block_scales[swizzled_scale_offset(tile, row_in_group % kScaleRowTile, col / kSfVectorSize,
                                           scale_column_tiles)];
    store_dequantized(value, scale, global_scales[row], dst);
  }
};

// Expert weights stored [G, N, K] (rows N, contraction K) with per-expert global scales, rows padded to 128
struct NVFP4WeightSource {
  const uint8_t* packed;
  const uint8_t* block_scales;
  const float* global_scales;
  int output_size, contraction, output_row_tiles, scale_column_tiles;
  __device__ void load16(int expert, int row, int col, __nv_bfloat16* dst) const {
    const uint64_t value = *reinterpret_cast<const uint64_t*>(
        packed + (static_cast<int64_t>(expert) * output_size + row) * (contraction / 2) + col / 2);
    const int64_t tile = static_cast<int64_t>(expert) * output_row_tiles + row / kScaleRowTile;
    const uint8_t scale =
        block_scales[swizzled_scale_offset(tile, row % kScaleRowTile, col / kSfVectorSize, scale_column_tiles)];
    store_dequantized(value, scale, global_scales[expert], dst);
  }
};

struct SrOutputs {
  uint8_t* rows_data;    // [Z, rows, cols]
  uint8_t* rows_scales;  // [Z, rows, cols / 32]
  uint8_t* cols_data;    // [Z, cols, rows]
  uint8_t* cols_scales;  // [Z, cols, rows / 32]
};

__device__ __forceinline__ void quantize16(const float* values, float amax, int64_t start, uint32_t call,
                                           uint32_t seed_lo, uint32_t seed_hi, uint8_t* data, uint8_t* scale,
                                           bool write_scale) {
  const int exponent = e8m0_rceil(amax);
  const float reciprocal_scale = e8m0_reciprocal(exponent);
  uint32_t draws[16];
  random16x16(start, call, seed_lo, seed_hi, draws);
  alignas(16) uint8_t bytes[16];
#pragma unroll
  for (int index = 0; index < 16; ++index) {
    bytes[index] = stochastic_e4m3(values[index], reciprocal_scale, draws[index]);
  }
  *reinterpret_cast<uint4*>(data) = *reinterpret_cast<const uint4*>(bytes);
  if (write_scale) {
    *scale = static_cast<uint8_t>(exponent);
  }
}

template <class Source, bool kRows, bool kCols>
__global__ void mxfp8_stochastic_kernel(Source source, SrOutputs out, int rows, int cols, uint32_t seed_lo,
                                        uint32_t seed_hi, uint32_t call_rows, uint32_t call_cols) {
  __shared__ alignas(16) __nv_bfloat16 tile[kSrTileRows][kSrPitch];
  const int z = blockIdx.z;
  const int row0 = blockIdx.y * kSrTileRows;
  const int col0 = blockIdx.x * kSrTileCols;
  {
    const int row = threadIdx.x / 8, chunk = (threadIdx.x % 8) * 16;
    if (col0 + chunk < cols) {
      source.load16(z, row0 + row, col0 + chunk, &tile[row][chunk]);
    } else {
      reinterpret_cast<uint4*>(&tile[row][chunk])[0] = make_uint4(0, 0, 0, 0);
      reinterpret_cast<uint4*>(&tile[row][chunk])[1] = make_uint4(0, 0, 0, 0);
    }
  }
  __syncthreads();
  const int half = threadIdx.x % 2;
  float values[16];
  if constexpr (kRows) {
    // Two threads per 32-wide block of a row
    const int block = threadIdx.x / 2, row = block / 4, col = (block % 4) * kMxBlock + half * 16;
    float amax = 0.0f;
#pragma unroll
    for (int index = 0; index < 16; ++index) {
      values[index] = __bfloat162float(tile[row][col + index]);
      amax = fmaxf(amax, fabsf(values[index]));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    if (col0 + col < cols) {
      const int64_t out_row = static_cast<int64_t>(z) * rows + row0 + row;
      const int64_t start = out_row * cols + col0 + col;
      quantize16(values, amax, start, call_rows, seed_lo, seed_hi, out.rows_data + start,
                 out.rows_scales + out_row * (cols / kMxBlock) + (col0 + col) / kMxBlock, half == 0);
    }
  }
  if constexpr (kCols) {
    // Two threads per column, 16 rows each
    const int col = threadIdx.x / 2, row = half * 16;
    float amax = 0.0f;
#pragma unroll
    for (int index = 0; index < 16; ++index) {
      values[index] = __bfloat162float(tile[row + index][col]);
      amax = fmaxf(amax, fabsf(values[index]));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    if (col0 + col < cols) {
      const int64_t out_row = static_cast<int64_t>(z) * cols + col0 + col;
      const int64_t start = out_row * rows + row0 + row;
      quantize16(values, amax, start, call_cols, seed_lo, seed_hi, out.cols_data + start,
                 out.cols_scales + out_row * (rows / kMxBlock) + row0 / kMxBlock, half == 0);
    }
  }
}

void check_blackwell() {
  const auto* properties = at::cuda::getDeviceProperties(at::cuda::current_device());
  TORCH_CHECK(properties->major == 10 && (properties->minor == 0 || properties->minor == 3),
              "prime-kernels NVFP4/MXFP8 quantization requires SM100 or SM103, but the current device is SM",
              properties->major, properties->minor, ".");
}

template <class Source, bool kRows, bool kCols>
void launch_stochastic(const Source& source, const SrOutputs& out, int64_t z, int64_t rows, int64_t cols, int64_t seed,
                       int64_t call_rows, int64_t call_cols, const at::Tensor& like) {
  if (rows == 0 || cols == 0 || z == 0) {
    return;
  }
  const dim3 grid((cols + kSrTileCols - 1) / kSrTileCols, rows / kSrTileRows, z);
  mxfp8_stochastic_kernel<Source, kRows, kCols><<<grid, 256, 0, at::cuda::getCurrentCUDAStream(like.get_device())>>>(
      source, out, rows, cols, static_cast<uint32_t>(static_cast<uint64_t>(seed)),
      static_cast<uint32_t>(static_cast<uint64_t>(seed) >> 32), static_cast<uint32_t>(call_rows),
      static_cast<uint32_t>(call_cols));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

uint8_t* byte_ptr(const at::Tensor& tensor) {
  return tensor.numel() ? reinterpret_cast<uint8_t*>(tensor.data_ptr()) : nullptr;
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor>
quantize_activations_cuda(
    const at::Tensor& matrix,
    const at::Tensor& offsets) {
  check_bf16_matrix(matrix, "matrix");
  TORCH_CHECK(matrix.dim() == 2, "matrix must be 2D.");
  TORCH_CHECK(
      matrix.size(1) > 0 && matrix.size(1) % 32 == 0,
      "matrix's contraction dimension must be a positive multiple of 32.");
  TORCH_CHECK(offsets.is_cuda(), "offsets must be a CUDA tensor.");
  TORCH_CHECK(offsets.scalar_type() == at::kInt, "offsets must be int32.");
  TORCH_CHECK(
      offsets.dim() == 1 && offsets.numel() > 0,
      "offsets must be a non-empty 1D tensor.");
  TORCH_CHECK(offsets.is_contiguous(), "offsets must be contiguous.");
  TORCH_CHECK(
      offsets.device() == matrix.device(),
      "matrix and offsets must be on the same device.");

  c10::cuda::CUDAGuard device_guard(matrix.device());
  check_blackwell();

  const int64_t rows = matrix.size(0);
  const int64_t contraction_size = matrix.size(1);
  const int64_t group_count = offsets.numel();
  const int64_t scale_columns =
      round_up(contraction_size / kSfVectorSize, kScaleColTile);
  const int64_t scale_column_tiles =
      scale_columns / kScaleColTile;
  const int64_t scale_column_ctas =
      round_up(scale_column_tiles, kQuantScaleTilesPerCta) /
      kQuantScaleTilesPerCta;
  const int64_t padded_scale_rows =
      round_up(rows + group_count * (kScaleRowTile - 1), kScaleRowTile);
  const int64_t scale_row_tiles =
      padded_scale_rows / kScaleRowTile;

  auto packed = at::empty(
      {rows, contraction_size / 2},
      matrix.options().dtype(at::kByte));
  auto block_scales = at::empty(
      {padded_scale_rows, scale_columns},
      matrix.options().dtype(at::kFloat8_e4m3fn));
  auto global_scales =
      at::empty({rows}, matrix.options().dtype(at::kFloat));
  auto row_amax = at::empty_like(global_scales);

  if (rows > 0) {
    const int device = matrix.get_device();
    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(device);
    const int amax_threads =
        activation_amax_threads(contraction_size);
    const int amax_blocks =
        activation_amax_blocks(
            device,
            rows,
            amax_threads,
            contraction_size);
    activation_amax_kernel<<<amax_blocks, amax_threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(matrix.data_ptr()),
        global_scales.mutable_data_ptr<float>(),
        row_amax.mutable_data_ptr<float>(),
        rows,
        contraction_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const dim3 grid(scale_row_tiles, scale_column_ctas);
    quantize_activations_tiled_kernel<<<grid, kQuantThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(matrix.data_ptr()),
        offsets.const_data_ptr<int32_t>(),
        packed.mutable_data_ptr<uint8_t>(),
        reinterpret_cast<uint8_t*>(block_scales.mutable_data_ptr()),
        global_scales.const_data_ptr<float>(),
        row_amax.const_data_ptr<float>(),
        contraction_size,
        group_count,
        scale_column_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {packed, block_scales, global_scales};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
quantize_weights_cuda(const at::Tensor& weight_rows) {
  check_bf16_matrix(weight_rows, "weight_rows");
  TORCH_CHECK(weight_rows.dim() == 3, "weight_rows must be 3D.");
  TORCH_CHECK(weight_rows.size(0) > 0, "weight_rows must contain an expert.");
  TORCH_CHECK(
      weight_rows.size(2) > 0 && weight_rows.size(2) % 32 == 0,
      "weight_rows' contraction dimension must be a positive multiple of 32.");

  c10::cuda::CUDAGuard device_guard(weight_rows.device());
  check_blackwell();

  const int64_t groups = weight_rows.size(0);
  const int64_t output_size = weight_rows.size(1);
  const int64_t contraction_size = weight_rows.size(2);
  const int64_t padded_output_size =
      round_up(output_size, kScaleRowTile);
  const int64_t output_row_tiles =
      padded_output_size / kScaleRowTile;
  const int64_t scale_columns =
      round_up(contraction_size / kSfVectorSize, kScaleColTile);
  const int64_t scale_column_tiles =
      scale_columns / kScaleColTile;
  const int64_t scale_column_ctas =
      round_up(scale_column_tiles, kQuantScaleTilesPerCta) /
      kQuantScaleTilesPerCta;

  auto packed = at::empty(
      {groups, output_size, contraction_size / 2},
      weight_rows.options().dtype(at::kByte));
  auto block_scales = at::empty(
      {groups, padded_output_size * scale_columns},
      weight_rows.options().dtype(at::kFloat8_e4m3fn));
  auto global_scales =
      at::empty({groups}, weight_rows.options().dtype(at::kFloat));

  if (output_size > 0) {
    const int device = weight_rows.get_device();
    const int64_t scale_blocks_per_expert =
        output_size * contraction_size / kSfVectorSize;
    const int blocks_per_expert = weight_amax_blocks_per_expert(
        device,
        groups,
        scale_blocks_per_expert);
    auto expert_amax =
        at::zeros({groups}, weight_rows.options().dtype(at::kFloat));
    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(device);
    weight_amax_kernel
        <<<groups * blocks_per_expert, kReductionThreads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                weight_rows.data_ptr()),
            expert_amax.mutable_data_ptr<float>(),
            output_size * contraction_size,
            blocks_per_expert);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    finalize_weight_scales_kernel<<<1, 256, 0, stream>>>(
        expert_amax.const_data_ptr<float>(),
        global_scales.mutable_data_ptr<float>(),
        groups);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const dim3 grid(groups * output_row_tiles, scale_column_ctas);
    quantize_weights_tiled_kernel<<<grid, kQuantThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(weight_rows.data_ptr()),
        global_scales.const_data_ptr<float>(),
        packed.mutable_data_ptr<uint8_t>(),
        reinterpret_cast<uint8_t*>(block_scales.mutable_data_ptr()),
        output_size,
        contraction_size,
        output_row_tiles,
        scale_column_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {packed, block_scales, global_scales};
}

at::Tensor dequantize_activations_cuda(
    const at::Tensor& packed,
    const at::Tensor& block_scales,
    const at::Tensor& global_scales,
    const at::Tensor& offsets) {
  check_quantized_tensors(packed, block_scales, global_scales);
  TORCH_CHECK(packed.dim() == 2, "packed activations must be 2D.");
  TORCH_CHECK(
      global_scales.dim() == 1 &&
          global_scales.size(0) == packed.size(0),
      "activation global_scales must contain one value per row.");
  TORCH_CHECK(
      offsets.is_cuda() && offsets.scalar_type() == at::kInt &&
          offsets.dim() == 1 && offsets.numel() > 0 &&
          offsets.is_contiguous() && offsets.device() == packed.device(),
      "offsets must be a non-empty contiguous CUDA int32 tensor on the "
      "same device.");

  c10::cuda::CUDAGuard device_guard(packed.device());
  check_blackwell();

  const int64_t rows = packed.size(0);
  const int64_t contraction_size = packed.size(1) * 2;
  const int64_t scale_columns =
      round_up(contraction_size / kSfVectorSize, kScaleColTile);
  const int64_t scale_column_tiles =
      scale_columns / kScaleColTile;
  const int64_t scale_column_ctas =
      round_up(scale_column_tiles, kDequantScaleTilesPerCta) /
      kDequantScaleTilesPerCta;
  TORCH_CHECK(
      block_scales.numel() % (scale_columns * kScaleRowTile) == 0,
      "activation block_scales have an invalid swizzled shape.");
  const int64_t scale_row_tiles =
      block_scales.numel() / (scale_columns * kScaleRowTile);
  auto output = at::empty(
      {rows, contraction_size},
      packed.options().dtype(at::kBFloat16));
  if (rows > 0) {
    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(packed.get_device());
    const dim3 grid(scale_row_tiles, scale_column_ctas);
    dequantize_activations_tiled_kernel
        <<<grid, kQuantThreads, 0, stream>>>(
            packed.const_data_ptr<uint8_t>(),
            reinterpret_cast<const uint8_t*>(block_scales.data_ptr()),
            global_scales.const_data_ptr<float>(),
            offsets.const_data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(
                output.mutable_data_ptr()),
            contraction_size,
            offsets.numel(),
            scale_column_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

at::Tensor dequantize_weights_cuda(
    const at::Tensor& packed,
    const at::Tensor& block_scales,
    const at::Tensor& global_scales) {
  check_quantized_tensors(packed, block_scales, global_scales);
  TORCH_CHECK(packed.dim() == 3, "packed weights must be 3D.");
  TORCH_CHECK(
      global_scales.dim() == 1 &&
          global_scales.size(0) == packed.size(0),
      "weight global_scales must contain one value per expert.");

  c10::cuda::CUDAGuard device_guard(packed.device());
  check_blackwell();

  const int64_t groups = packed.size(0);
  const int64_t output_size = packed.size(1);
  const int64_t contraction_size = packed.size(2) * 2;
  const int64_t padded_output_size =
      round_up(output_size, kScaleRowTile);
  const int64_t output_row_tiles =
      padded_output_size / kScaleRowTile;
  const int64_t scale_columns =
      round_up(contraction_size / kSfVectorSize, kScaleColTile);
  const int64_t scale_column_tiles =
      scale_columns / kScaleColTile;
  const int64_t scale_column_ctas =
      round_up(scale_column_tiles, kDequantScaleTilesPerCta) /
      kDequantScaleTilesPerCta;
  TORCH_CHECK(
      block_scales.numel() ==
          groups * padded_output_size * scale_columns,
      "weight block_scales have an invalid swizzled shape.");
  auto output = at::empty(
      {groups, output_size, contraction_size},
      packed.options().dtype(at::kBFloat16));
  if (output_size > 0) {
    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(packed.get_device());
    const dim3 grid(
        groups * output_row_tiles,
        scale_column_ctas);
    dequantize_weights_tiled_kernel
        <<<grid, kQuantThreads, 0, stream>>>(
            packed.const_data_ptr<uint8_t>(),
            reinterpret_cast<const uint8_t*>(block_scales.data_ptr()),
            global_scales.const_data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(
                output.mutable_data_ptr()),
            output_size,
            contraction_size,
            output_row_tiles,
            scale_column_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mxfp8_stochastic_bf16_cuda(
    const at::Tensor& values, int64_t seed, int64_t call_rows, int64_t call_cols, bool rows_wise, bool cols_wise) {
  check_bf16_matrix(values, "values");
  TORCH_CHECK(values.dim() == 2, "values must be 2D.");
  c10::cuda::CUDAGuard device_guard(values.device());
  check_blackwell();
  const int64_t rows = values.size(0), cols = values.size(1);
  TORCH_CHECK(rows % kMxBlock == 0 && cols % kMxBlock == 0, "both dims must be multiples of 32.");
  const auto bytes = values.options().dtype(at::kByte);
  auto rows_data = at::empty({rows_wise ? rows : 0, cols}, bytes);
  auto rows_scales = at::empty({rows_wise ? rows : 0, cols / kMxBlock}, bytes);
  auto cols_data = at::empty({cols_wise ? cols : 0, rows}, bytes);
  auto cols_scales = at::empty({cols_wise ? cols : 0, rows / kMxBlock}, bytes);
  const Bf16Source source{reinterpret_cast<const __nv_bfloat16*>(values.const_data_ptr()), rows, cols};
  const SrOutputs out{byte_ptr(rows_data), byte_ptr(rows_scales), byte_ptr(cols_data), byte_ptr(cols_scales)};
  if (rows_wise && cols_wise) {
    launch_stochastic<Bf16Source, true, true>(source, out, 1, rows, cols, seed, call_rows, call_cols, values);
  } else if (rows_wise) {
    launch_stochastic<Bf16Source, true, false>(source, out, 1, rows, cols, seed, call_rows, call_cols, values);
  } else if (cols_wise) {
    launch_stochastic<Bf16Source, false, true>(source, out, 1, rows, cols, seed, call_rows, call_cols, values);
  }
  return {rows_data, rows_scales, cols_data, cols_scales};
}

std::tuple<at::Tensor, at::Tensor> mxfp8_stochastic_nvfp4_activations_cuda(
    const at::Tensor& packed, const at::Tensor& block_scales, const at::Tensor& global_scales,
    const at::Tensor& offsets, int64_t seed, int64_t call) {
  check_quantized_tensors(packed, block_scales, global_scales);
  TORCH_CHECK(packed.dim() == 2, "packed activations must be 2D.");
  TORCH_CHECK(offsets.is_cuda() && offsets.scalar_type() == at::kInt && offsets.is_contiguous() && offsets.numel() > 0,
              "offsets must be a non-empty contiguous CUDA int32 tensor.");
  c10::cuda::CUDAGuard device_guard(packed.device());
  check_blackwell();
  const int64_t rows = packed.size(0), contraction = packed.size(1) * 2;
  TORCH_CHECK(rows % kMxBlock == 0, "the token count must be a multiple of 32.");
  TORCH_CHECK(contraction % kSfVectorSize == 0, "the contraction dim must be a multiple of 16.");
  const int64_t scale_column_tiles = round_up(contraction / kSfVectorSize, kScaleColTile) / kScaleColTile;
  const auto bytes = packed.options().dtype(at::kByte);
  auto data = at::empty({contraction, rows}, bytes);
  auto scales = at::empty({contraction, rows / kMxBlock}, bytes);
  const NVFP4ActivationSource source{packed.const_data_ptr<uint8_t>(),
                                     reinterpret_cast<const uint8_t*>(block_scales.data_ptr()),
                                     global_scales.const_data_ptr<float>(),
                                     offsets.const_data_ptr<int32_t>(),
                                     static_cast<int>(offsets.numel()),
                                     static_cast<int>(contraction),
                                     static_cast<int>(scale_column_tiles)};
  launch_stochastic<NVFP4ActivationSource, false, true>(source, SrOutputs{nullptr, nullptr, byte_ptr(data),
                                                                          byte_ptr(scales)},
                                                        1, rows, contraction, seed, call, call, packed);
  return {data, scales};
}

std::tuple<at::Tensor, at::Tensor> mxfp8_stochastic_nvfp4_weights_cuda(
    const at::Tensor& packed, const at::Tensor& block_scales, const at::Tensor& global_scales, int64_t seed,
    int64_t call) {
  check_quantized_tensors(packed, block_scales, global_scales);
  TORCH_CHECK(packed.dim() == 3, "packed weights must be 3D.");
  c10::cuda::CUDAGuard device_guard(packed.device());
  check_blackwell();
  const int64_t groups = packed.size(0), output_size = packed.size(1), contraction = packed.size(2) * 2;
  TORCH_CHECK(output_size % kMxBlock == 0, "the weights' output dim must be a multiple of 32.");
  TORCH_CHECK(contraction % kSfVectorSize == 0, "the contraction dim must be a multiple of 16.");
  const int64_t output_row_tiles = round_up(output_size, kScaleRowTile) / kScaleRowTile;
  const int64_t scale_column_tiles = round_up(contraction / kSfVectorSize, kScaleColTile) / kScaleColTile;
  const auto bytes = packed.options().dtype(at::kByte);
  auto data = at::empty({groups, contraction, output_size}, bytes);
  auto scales = at::empty({groups, contraction, output_size / kMxBlock}, bytes);
  const NVFP4WeightSource source{packed.const_data_ptr<uint8_t>(),
                                 reinterpret_cast<const uint8_t*>(block_scales.data_ptr()),
                                 global_scales.const_data_ptr<float>(),
                                 static_cast<int>(output_size),
                                 static_cast<int>(contraction),
                                 static_cast<int>(output_row_tiles),
                                 static_cast<int>(scale_column_tiles)};
  launch_stochastic<NVFP4WeightSource, false, true>(source, SrOutputs{nullptr, nullptr, byte_ptr(data),
                                                                      byte_ptr(scales)},
                                                    groups, output_size, contraction, seed, call, call, packed);
  return {data, scales};
}

} // namespace prime_kernels::nvfp4
