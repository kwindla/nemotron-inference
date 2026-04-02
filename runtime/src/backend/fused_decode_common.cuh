#pragma once

#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstddef>
#include <cstdint>

namespace nemotron::fused_decode {

constexpr int kThreadsPerBlock = 256;
constexpr std::size_t kNvfp4BlockWidth = 16;
constexpr float kNvfp4Fp4MaxFinite = 6.0f;
constexpr float kNvfp4Fp8E4M3MaxFinite = 448.0f;
constexpr float kNvfp4MinScale = 1.0f / 1024.0f;

struct Nvfp4WeightView {
  const std::uint8_t* packed_data = nullptr;
  const std::uint8_t* block_scales_data = nullptr;
  const float* tensor_scale_data = nullptr;
  std::size_t output_rows = 0;
  std::size_t input_cols = 0;

  __host__ __device__ bool valid() const {
    return packed_data != nullptr &&
           block_scales_data != nullptr &&
           tensor_scale_data != nullptr &&
           output_rows != 0 &&
           input_cols != 0 &&
           (input_cols % 16) == 0;
  }
};

__device__ inline float Sigmoid(float value) {
  if (value >= 0.0f) {
    const float exp_neg = expf(-value);
    return 1.0f / (1.0f + exp_neg);
  }
  const float exp_pos = expf(value);
  return exp_pos / (1.0f + exp_pos);
}

__device__ inline float SiLU(float value) {
  return value * Sigmoid(value);
}

__device__ inline float Softplus(float value) {
  if (value > 20.0f) {
    return value;
  }
  if (value < -20.0f) {
    return expf(value);
  }
  return log1pf(expf(value));
}

__device__ inline float Relu2(float value) {
  return value > 0.0f ? value * value : 0.0f;
}

__device__ inline float DecodeFp4(std::uint8_t raw_nibble) {
  __nv_fp4_e2m1 value;
  value.__x = raw_nibble & 0x0F;
  return static_cast<float>(value);
}

__device__ inline float DecodeFp8(std::uint8_t raw_byte) {
  __nv_fp8_e4m3 value;
  value.__x = raw_byte;
  return static_cast<float>(value);
}

__device__ inline float ClampNvfp4Scale(float value) {
  if (!isfinite(value) || value < kNvfp4MinScale) {
    return kNvfp4MinScale;
  }
  return value;
}

__device__ inline std::uint8_t EncodeFp4(float value) {
  return static_cast<std::uint8_t>(
             __nv_cvt_float_to_fp4(value, __NV_E2M1, cudaRoundNearest)) &
         0x0F;
}

__device__ inline std::uint8_t EncodeFp8Scale(float value) {
  return static_cast<std::uint8_t>(
      __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

__device__ inline float QuantizeDequantizeNvfp4Scalar(float value, float scale) {
  if (!(scale > 0.0f) || !isfinite(value)) {
    return 0.0f;
  }
  return DecodeFp4(EncodeFp4(value / scale)) * scale;
}

__device__ inline void QuantizeDequantizeNvfp4Row(
    const float* input,
    float* output,
    std::size_t cols) {
  if (input == nullptr || output == nullptr || cols == 0 || (cols % kNvfp4BlockWidth) != 0) {
    return;
  }

  float global_max_abs = 0.0f;
  for (std::size_t col = 0; col < cols; ++col) {
    global_max_abs = fmaxf(global_max_abs, fabsf(input[col]));
  }

  float tensor_scale = 1.0f;
  if (global_max_abs > (kNvfp4Fp4MaxFinite * kNvfp4Fp8E4M3MaxFinite)) {
    tensor_scale = ClampNvfp4Scale(
        global_max_abs / (kNvfp4Fp4MaxFinite * kNvfp4Fp8E4M3MaxFinite));
  }

  for (std::size_t block = 0; block < cols / kNvfp4BlockWidth; ++block) {
    const std::size_t block_offset = block * kNvfp4BlockWidth;
    float block_max_abs = 0.0f;
    for (std::size_t i = 0; i < kNvfp4BlockWidth; ++i) {
      block_max_abs = fmaxf(block_max_abs, fabsf(input[block_offset + i]));
    }

    float block_scale = 1.0f;
    if (block_max_abs > 0.0f) {
      block_scale = ClampNvfp4Scale(
          block_max_abs / (kNvfp4Fp4MaxFinite * tensor_scale));
    }
    const std::uint8_t encoded_block_scale = EncodeFp8Scale(block_scale);
    const float quantized_block_scale = DecodeFp8(encoded_block_scale);
    const float pack_scale = tensor_scale * block_scale;
    const float dequant_scale = tensor_scale * quantized_block_scale;
    for (std::size_t i = 0; i < kNvfp4BlockWidth; ++i) {
      const std::uint8_t encoded_value =
          EncodeFp4(input[block_offset + i] / pack_scale);
      output[block_offset + i] = DecodeFp4(encoded_value) * dequant_scale;
    }
  }
}

__device__ inline float Nvfp4RowMajorDot(
    const float* activations,
    const Nvfp4WeightView& weight,
    std::size_t row_index) {
  const std::size_t blocks_per_row = weight.input_cols / 16;
  const std::size_t packed_row_offset = row_index * (weight.input_cols / 2);
  const std::size_t scale_row_offset = row_index * blocks_per_row;
  const float tensor_scale = *weight.tensor_scale_data;
  double accum = 0.0;
  std::size_t packed_index = packed_row_offset;
  for (std::size_t block = 0; block < blocks_per_row; ++block) {
    const float block_scale =
        DecodeFp8(weight.block_scales_data[scale_row_offset + block]) * tensor_scale;
    const std::size_t col_start = block * 16;
    for (std::size_t offset = 0; offset < 16; offset += 2) {
      const std::uint8_t byte = weight.packed_data[packed_index++];
      const float w0 = DecodeFp4(byte & 0x0F) * block_scale;
      const float w1 = DecodeFp4((byte >> 4) & 0x0F) * block_scale;
      accum += static_cast<double>(activations[col_start + offset]) *
               static_cast<double>(w0);
      accum += static_cast<double>(activations[col_start + offset + 1]) *
               static_cast<double>(w1);
    }
  }
  return static_cast<float>(accum);
}

}  // namespace nemotron::fused_decode
