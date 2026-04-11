#include "nemotron/device_nvfp4_matrix.h"
#include "nemotron/device_tensor.h"
#include "nemotron/fused_moe_prefill.h"
#include "nemotron/nvfp4_scale_layout.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using nemotron::BuildNvfp4ExecutionScaleLayout;
using nemotron::DeviceNvfp4Matrix;
using nemotron::DeviceTensorBf16;
using nemotron::DeviceTensorFp32;
using nemotron::Nvfp4ExecutionScaleLayout;
using nemotron::Nvfp4ScaleLayout;
using nemotron::RunP5NativeDirectPackOracleForTesting;
using nemotron::SwizzleRowMajorNvfp4ScalesForExecution;

constexpr int kTileRows = 128;
constexpr int kTileCols = 128;
constexpr int kThreadsPerBlock = 256;
constexpr std::size_t kBlockWidth = 16;
constexpr std::size_t kBlocksPerRow = kTileCols / kBlockWidth;
constexpr std::size_t kPackedBytesPerRow = kTileCols / 2u;
constexpr std::size_t kPackedBytesPerBlock = kBlockWidth / 2u;
constexpr std::size_t kScaleCount = kTileRows * kBlocksPerRow;
constexpr std::size_t kTileElements = kTileRows * kTileCols;
constexpr float kNvfp4Fp4MaxFinite = 6.0f;
constexpr float kNvfp4MinScale = 1.0f / 1024.0f;
constexpr Nvfp4ScaleLayout kScaleLayout = Nvfp4ScaleLayout::kSwizzled128x4;

__host__ __device__ inline float relu2(float value) {
  return value > 0.0f ? value * value : 0.0f;
}

__host__ __device__ inline float clamp_nvfp4_scale(float value) {
  if (!isfinite(value) || value < kNvfp4MinScale) {
    return kNvfp4MinScale;
  }
  return value;
}

__host__ __device__ inline std::size_t round_up_to_4(std::size_t value) {
  return (value + 3u) & ~std::size_t{3u};
}

__host__ __device__ inline std::size_t execution_scale_offset(
    std::size_t row,
    std::size_t block_col,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout) {
  const std::size_t num_k_tiles = padded_blocks_per_row / 4u;
  const std::size_t k_tile = block_col / 4u;
  const std::size_t inner_k = block_col & 3u;
  switch (scale_layout) {
    case Nvfp4ScaleLayout::kSwizzled128x4: {
      const std::size_t m_tile = row / 128u;
      const std::size_t outer_m = row & 31u;
      const std::size_t inner_m = (row >> 5u) & 3u;
      return ((((m_tile * num_k_tiles) + k_tile) << 9u) |
              (outer_m << 4u) |
              (inner_m << 2u) |
              inner_k);
    }
    case Nvfp4ScaleLayout::kSwizzled8x4: {
      const std::size_t m_tile = row / 8u;
      const std::size_t inner_m = row & 7u;
      return (((m_tile * num_k_tiles) + k_tile) << 5u) |
             (inner_m << 2u) |
             inner_k;
    }
  }
  return 0u;
}

__device__ inline std::uint8_t encode_fp4_device(float value) {
  return static_cast<std::uint8_t>(
             __nv_cvt_float_to_fp4(value, __NV_E2M1, cudaRoundNearest)) &
         0x0Fu;
}

__device__ inline std::uint8_t encode_fp8_scale_device(float value) {
  return static_cast<std::uint8_t>(
      __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

template <bool kLegacyZeroBlockScale>
__device__ inline void pack_block_outputs(
    const float* activated_block,
    std::size_t row,
    std::size_t block,
    int valid_rows,
    std::size_t padded_blocks_per_row,
    float* activation_output_scale,
    std::uint8_t* packed_data,
    std::uint8_t* block_scales_data,
    std::uint8_t* matmul_block_scales_data) {
  const std::size_t scale_index = row * kBlocksPerRow + block;
  const std::size_t matmul_scale_index =
      execution_scale_offset(row, block, padded_blocks_per_row, kScaleLayout);
  const std::size_t packed_offset = row * kPackedBytesPerRow + block * kPackedBytesPerBlock;

  if (static_cast<int>(row) >= valid_rows) {
    activation_output_scale[scale_index] = 0.0f;
    block_scales_data[scale_index] = 0u;
    matmul_block_scales_data[matmul_scale_index] = 0u;
    for (std::size_t pair = 0; pair < kPackedBytesPerBlock; ++pair) {
      packed_data[packed_offset + pair] = 0u;
    }
    return;
  }

  float block_max_abs = 0.0f;
  for (std::size_t i = 0; i < kBlockWidth; ++i) {
    block_max_abs = fmaxf(block_max_abs, activated_block[i]);
  }

  float dequant_scale = 0.0f;
  if constexpr (kLegacyZeroBlockScale) {
    dequant_scale = block_max_abs > 0.0f
                        ? clamp_nvfp4_scale(block_max_abs / kNvfp4Fp4MaxFinite)
                        : 1.0f;
  } else {
    dequant_scale = clamp_nvfp4_scale(block_max_abs / kNvfp4Fp4MaxFinite);
  }
  const std::uint8_t encoded_scale = encode_fp8_scale_device(dequant_scale);

  activation_output_scale[scale_index] = dequant_scale;
  block_scales_data[scale_index] = encoded_scale;
  matmul_block_scales_data[matmul_scale_index] = encoded_scale;

  for (std::size_t pair = 0; pair < kPackedBytesPerBlock; ++pair) {
    const std::uint8_t lhs = encode_fp4_device(activated_block[pair * 2u] / dequant_scale);
    const std::uint8_t rhs =
        encode_fp4_device(activated_block[pair * 2u + 1u] / dequant_scale);
    packed_data[packed_offset + pair] =
        static_cast<std::uint8_t>((lhs & 0x0Fu) | ((rhs & 0x0Fu) << 4u));
  }
}

struct SharedStagingStorage {
  union {
    float activated[kTileElements];
    std::uint8_t raw[kTileElements * sizeof(float)];
  };
};

__global__ void TestStagedFp4PackKernel(
    const float* input,
    int valid_rows,
    std::size_t padded_blocks_per_row,
    float* activation_output_scale,
    std::uint8_t* packed_data,
    std::uint8_t* block_scales_data,
    std::uint8_t* matmul_block_scales_data,
    float* tensor_scale_data,
    float* per_row_tensor_scales) {
  __shared__ SharedStagingStorage staging;

  for (std::size_t row = threadIdx.x; row < kTileRows; row += blockDim.x) {
    per_row_tensor_scales[row] = 1.0f;
  }
  if (threadIdx.x == 0) {
    *tensor_scale_data = 1.0f;
  }

  // Stage the full tile after activation using the aliased-storage model from the probe.
  for (std::size_t index = threadIdx.x; index < kTileElements; index += blockDim.x) {
    const std::size_t row = index / kTileCols;
    staging.activated[index] = static_cast<int>(row) < valid_rows ? relu2(input[index]) : 0.0f;
  }
  __syncthreads();

  for (std::size_t row_block = threadIdx.x; row_block < kScaleCount; row_block += blockDim.x) {
    const std::size_t row = row_block / kBlocksPerRow;
    const std::size_t block = row_block % kBlocksPerRow;
    pack_block_outputs<false>(
        staging.activated + (row * kTileCols) + (block * kBlockWidth),
        row,
        block,
        valid_rows,
        padded_blocks_per_row,
        activation_output_scale,
        packed_data,
        block_scales_data,
        matmul_block_scales_data);
  }
}

__device__ inline void run_scaled_staged_pack(
    SharedStagingStorage& staging,
    const float* input,
    const float* per_row_tensor_scales_input,
    int valid_rows,
    std::size_t padded_blocks_per_row,
    float* activation_output_scale,
    std::uint8_t* packed_data,
    std::uint8_t* block_scales_data,
    std::uint8_t* matmul_block_scales_data,
    float* tensor_scale_data,
    float* per_row_tensor_scales) {
  constexpr float weight_tensor_scale = 1.0f;

  for (std::size_t row = threadIdx.x; row < kTileRows; row += blockDim.x) {
    const float row_scale =
        per_row_tensor_scales_input != nullptr ? per_row_tensor_scales_input[row] : 1.0f;
    per_row_tensor_scales[row] = row_scale;
  }
  if (threadIdx.x == 0) {
    *tensor_scale_data = weight_tensor_scale;
  }
  __syncthreads();

  for (std::size_t index = threadIdx.x; index < kTileElements; index += blockDim.x) {
    const std::size_t row = index / kTileCols;
    const float row_scale =
        per_row_tensor_scales_input != nullptr ? per_row_tensor_scales_input[row] : 1.0f;
    staging.activated[index] = static_cast<int>(row) < valid_rows
                                   ? relu2(input[index] * row_scale * weight_tensor_scale)
                                   : 0.0f;
  }
  __syncthreads();

  for (std::size_t row_block = threadIdx.x; row_block < kScaleCount; row_block += blockDim.x) {
    const std::size_t row = row_block / kBlocksPerRow;
    const std::size_t block = row_block % kBlocksPerRow;
    pack_block_outputs<false>(
        staging.activated + (row * kTileCols) + (block * kBlockWidth),
        row,
        block,
        valid_rows,
        padded_blocks_per_row,
        activation_output_scale,
        packed_data,
        block_scales_data,
        matmul_block_scales_data);
  }
}

__global__ void TestScaledStagedFp4PackKernel(
    const float* input,
    const float* per_row_tensor_scales_input,
    int valid_rows,
    std::size_t padded_blocks_per_row,
    float* activation_output_scale,
    std::uint8_t* packed_data,
    std::uint8_t* block_scales_data,
    std::uint8_t* matmul_block_scales_data,
    float* tensor_scale_data,
    float* per_row_tensor_scales) {
  __shared__ SharedStagingStorage staging;
  run_scaled_staged_pack(
      staging,
      input,
      per_row_tensor_scales_input,
      valid_rows,
      padded_blocks_per_row,
      activation_output_scale,
      packed_data,
      block_scales_data,
      matmul_block_scales_data,
      tensor_scale_data,
      per_row_tensor_scales);
}

__global__ void TestWarpLocalFp4PackKernel(
    const float* input,
    const float* per_row_tensor_scales_input,
    int valid_rows,
    std::size_t padded_blocks_per_row,
    float* activation_output_scale,
    std::uint8_t* packed_data,
    std::uint8_t* block_scales_data,
    std::uint8_t* matmul_block_scales_data,
    float* tensor_scale_data,
    float* per_row_tensor_scales) {
  for (std::size_t row = threadIdx.x; row < kTileRows; row += blockDim.x) {
    per_row_tensor_scales[row] =
        per_row_tensor_scales_input != nullptr ? per_row_tensor_scales_input[row] : 1.0f;
  }
  if (threadIdx.x == 0) {
    *tensor_scale_data = 1.0f;
  }
  __syncthreads();

  constexpr int kPassRows = 32;
  constexpr int kPassCols = 128;
  constexpr int kLTBH[16] = {0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1};
  constexpr int kLTTG[16] = {0, 2, 0, 2, 1, 3, 1, 3, 0, 2, 0, 2, 1, 3, 1, 3};
  constexpr int kLTMH[16] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
  constexpr int kBEMh0[8] = {0, 4, 1, 5, 2, 6, 3, 7};
  constexpr int kBEMh1[8] = {8, 12, 9, 13, 10, 14, 11, 15};

  for (int pass = 0; pass < 4; ++pass) {
    const int row_base = pass * kPassRows;
    const int pass_valid_rows = min(valid_rows - row_base, kPassRows);
    if (pass_valid_rows <= 0) {
      for (std::size_t row_block = threadIdx.x;
           row_block < static_cast<std::size_t>(kPassRows) * kBlocksPerRow;
           row_block += blockDim.x) {
        const std::size_t row = row_base + row_block / kBlocksPerRow;
        const std::size_t block = row_block % kBlocksPerRow;
        if (row >= kTileRows) {
          continue;
        }
        const std::size_t scale_index = row * kBlocksPerRow + block;
        const std::size_t matmul_scale_index =
            execution_scale_offset(row, block, padded_blocks_per_row, kScaleLayout);
        const std::size_t packed_offset =
            row * kPackedBytesPerRow + block * kPackedBytesPerBlock;
        activation_output_scale[scale_index] = 0.0f;
        block_scales_data[scale_index] = 0u;
        matmul_block_scales_data[matmul_scale_index] = 0u;
        for (std::size_t p = 0; p < kPackedBytesPerBlock; ++p) {
          packed_data[packed_offset + p] = 0u;
        }
      }
      __syncthreads();
      continue;
    }

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x & 31;
    const int g_m = warp_id & 3;
    const int g_n = warp_id / 4;
    const int q_val = lane_id / 4;
    const int r_val = lane_id & 3;
    const unsigned int subgroup_mask = 0x11111111u << r_val;

    const int n_coords[4] = {
        16 * g_n + 2 * r_val,
        16 * g_n + 2 * r_val + 1,
        16 * g_n + 8 + 2 * r_val,
        16 * g_n + 9 + 2 * r_val};

    constexpr float weight_tensor_scale = 1.0f;
    float activated[16];
    for (int i = 0; i < 16; ++i) {
      const int bh = kLTBH[i];
      const int tg = kLTTG[i];
      const int mh = kLTMH[i];
      const int m_coord =
          (bh == 0 ? 0 : 64) + 16 * g_m + (mh == 0 ? q_val : 8 + q_val);
      const int n_coord = n_coords[tg];
      const int abs_row = row_base + n_coord;

      activated[i] = 0.0f;
      if (n_coord < pass_valid_rows &&
          m_coord < kPassCols &&
          abs_row < static_cast<int>(kTileRows)) {
        const float row_scale =
            per_row_tensor_scales_input != nullptr ? per_row_tensor_scales_input[abs_row] : 1.0f;
        activated[i] =
            relu2(input[abs_row * kTileCols + m_coord] * row_scale * weight_tensor_scale);
      }
    }

    for (int b = 0; b < 8; ++b) {
      const int bh = b / 4;
      const int tg = b & 3;
      const float val_mh0 = activated[kBEMh0[b]];
      const float val_mh1 = activated[kBEMh1[b]];

      const int n_coord = n_coords[tg];
      const int abs_row = row_base + n_coord;
      const int block_start_m = (bh == 0 ? 0 : 64) + 16 * g_m;

      if (block_start_m + static_cast<int>(kBlockWidth) > kPassCols) {
        continue;
      }
      if (abs_row >= static_cast<int>(kTileRows)) {
        continue;
      }

      const std::size_t block_index = static_cast<std::size_t>(block_start_m) / kBlockWidth;
      const std::size_t scale_index = static_cast<std::size_t>(abs_row) * kBlocksPerRow + block_index;
      const std::size_t matmul_scale_index = execution_scale_offset(
          static_cast<std::size_t>(abs_row), block_index, padded_blocks_per_row, kScaleLayout);
      const std::size_t packed_offset_base =
          static_cast<std::size_t>(abs_row) * kPackedBytesPerRow +
          static_cast<std::size_t>(block_start_m) / 2u;

      if (n_coord >= pass_valid_rows) {
        if (q_val == 0) {
          activation_output_scale[scale_index] = 0.0f;
          block_scales_data[scale_index] = 0u;
          matmul_block_scales_data[matmul_scale_index] = 0u;
        }
        if ((q_val & 1) == 0) {
          const int k = q_val / 2;
          packed_data[packed_offset_base + static_cast<std::size_t>(k)] = 0u;
          packed_data[packed_offset_base + 4u + static_cast<std::size_t>(k)] = 0u;
        }
        continue;
      }

      float local_max = fmaxf(val_mh0, val_mh1);
      float partner = __shfl_xor_sync(subgroup_mask, local_max, 4);
      local_max = fmaxf(local_max, partner);
      partner = __shfl_xor_sync(subgroup_mask, local_max, 8);
      local_max = fmaxf(local_max, partner);
      partner = __shfl_xor_sync(subgroup_mask, local_max, 16);
      local_max = fmaxf(local_max, partner);

      float block_scale = 1.0f;
      if (local_max > 0.0f) {
        block_scale = clamp_nvfp4_scale(local_max / kNvfp4Fp4MaxFinite);
      }
      const std::uint8_t encoded_scale = encode_fp8_scale_device(block_scale);

      const std::uint8_t nibble_mh0 = encode_fp4_device(val_mh0 / block_scale);
      const std::uint8_t nibble_mh1 = encode_fp4_device(val_mh1 / block_scale);
      const std::uint8_t partner_nibble_mh0 = static_cast<std::uint8_t>(
          __shfl_down_sync(subgroup_mask, static_cast<int>(nibble_mh0), 4));
      const std::uint8_t partner_nibble_mh1 = static_cast<std::uint8_t>(
          __shfl_down_sync(subgroup_mask, static_cast<int>(nibble_mh1), 4));

      if (q_val == 0) {
        activation_output_scale[scale_index] = block_scale;
        block_scales_data[scale_index] = encoded_scale;
        matmul_block_scales_data[matmul_scale_index] = encoded_scale;
      }

      if ((q_val & 1) == 0) {
        const int k = q_val / 2;
        packed_data[packed_offset_base + static_cast<std::size_t>(k)] =
            static_cast<std::uint8_t>(
                (nibble_mh0 & 0x0Fu) | ((partner_nibble_mh0 & 0x0Fu) << 4u));
        packed_data[packed_offset_base + 4u + static_cast<std::size_t>(k)] =
            static_cast<std::uint8_t>(
                (nibble_mh1 & 0x0Fu) | ((partner_nibble_mh1 & 0x0Fu) << 4u));
      }
    }

    __syncthreads();
  }
}

__global__ void TestLegacyBf16Relu2PackKernel(
    const __nv_bfloat16* input,
    int valid_rows,
    std::size_t padded_blocks_per_row,
    float* activation_output_scale,
    std::uint8_t* packed_data,
    std::uint8_t* block_scales_data,
    std::uint8_t* matmul_block_scales_data,
    float* tensor_scale_data,
    float* per_row_tensor_scales) {
  for (std::size_t row = threadIdx.x; row < kTileRows; row += blockDim.x) {
    per_row_tensor_scales[row] = 1.0f;
  }
  if (threadIdx.x == 0) {
    *tensor_scale_data = 1.0f;
  }
  __syncthreads();

  for (std::size_t row_block = threadIdx.x; row_block < kScaleCount; row_block += blockDim.x) {
    const std::size_t row = row_block / kBlocksPerRow;
    const std::size_t block = row_block % kBlocksPerRow;
    float activated_block[kBlockWidth];
    const std::size_t input_offset = row * kTileCols + block * kBlockWidth;
    for (std::size_t i = 0; i < kBlockWidth; ++i) {
      activated_block[i] =
          static_cast<int>(row) < valid_rows ? relu2(__bfloat162float(input[input_offset + i]))
                                             : 0.0f;
    }
    pack_block_outputs<true>(
        activated_block,
        row,
        block,
        valid_rows,
        padded_blocks_per_row,
        activation_output_scale,
        packed_data,
        block_scales_data,
        matmul_block_scales_data);
  }
}

struct HostPackOutputs {
  std::vector<std::uint8_t> packed;
  std::vector<std::uint8_t> block_scales;
  std::vector<std::uint8_t> matmul_block_scales;
  std::vector<float> activation_output_scale;
  std::vector<float> per_row_tensor_scales;
  float tensor_scale = 0.0f;
};

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool check_cuda(cudaError_t status, const std::string& where) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA error at " << where << ": " << cudaGetErrorString(status) << "\n";
    return false;
  }
  return true;
}

bool has_cuda_device() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

std::uint32_t float_bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool compare_byte_vectors(
    const std::vector<std::uint8_t>& actual,
    const std::vector<std::uint8_t>& expected,
    const std::string& label) {
  if (!expect(actual.size() == expected.size(), label + " size mismatch")) {
    return false;
  }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      std::cerr << "FAIL: " << label << " mismatch at byte " << i << " actual="
                << static_cast<unsigned int>(actual[i]) << " expected="
                << static_cast<unsigned int>(expected[i]) << "\n";
      return false;
    }
  }
  return true;
}

bool compare_float_vectors_bitwise(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    const std::string& label) {
  if (!expect(actual.size() == expected.size(), label + " size mismatch")) {
    return false;
  }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (float_bits(actual[i]) != float_bits(expected[i])) {
      std::cerr << "FAIL: " << label << " mismatch at element " << i << " actual_bits=0x"
                << std::hex << float_bits(actual[i]) << " expected_bits=0x"
                << float_bits(expected[i]) << std::dec << "\n";
      return false;
    }
  }
  return true;
}

bool compare_float_scalar_bitwise(float actual, float expected, const std::string& label) {
  if (float_bits(actual) != float_bits(expected)) {
    std::cerr << "FAIL: " << label << " mismatch actual_bits=0x" << std::hex
              << float_bits(actual) << " expected_bits=0x" << float_bits(expected)
              << std::dec << "\n";
    return false;
  }
  return true;
}

float decode_positive_fp8_value(std::uint8_t code) {
  if (code == 0u) {
    return 0.0f;
  }
  const std::uint8_t exponent = static_cast<std::uint8_t>((code >> 3u) & 0x0Fu);
  const std::uint8_t mantissa = static_cast<std::uint8_t>(code & 0x07u);
  if (exponent == 0u) {
    return std::ldexp(static_cast<float>(mantissa), -9);
  }
  return std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, static_cast<int>(exponent) - 7);
}

const std::array<float, 0x7F>& positive_fp8_finite_values() {
  static const std::array<float, 0x7F> values = []() {
    std::array<float, 0x7F> table{};
    for (std::size_t code = 0; code <= 0x7Eu; ++code) {
      table[code] = decode_positive_fp8_value(static_cast<std::uint8_t>(code));
    }
    return table;
  }();
  return values;
}

std::uint8_t encode_fp8_scale_reference(float value) {
  if (!(value > 0.0f) || !std::isfinite(value)) {
    return 0u;
  }
  const auto& candidates = positive_fp8_finite_values();
  float best_error = std::numeric_limits<float>::infinity();
  std::uint8_t best_code = 0u;
  for (std::size_t code = 0; code <= 0x7Eu; ++code) {
    const float error = std::fabs(candidates[code] - value);
    if (error < best_error) {
      best_error = error;
      best_code = static_cast<std::uint8_t>(code);
      continue;
    }
    if (error == best_error &&
        ((code & 1u) == 0u) &&
        ((best_code & 1u) != 0u)) {
      best_code = static_cast<std::uint8_t>(code);
    }
  }
  return best_code;
}

std::uint8_t encode_fp4_reference(float value) {
  static constexpr std::array<float, 8> kPositiveValues = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  if (!(value > 0.0f) || !std::isfinite(value)) {
    return 0u;
  }
  float best_error = std::numeric_limits<float>::infinity();
  std::uint8_t best_code = 0u;
  for (std::size_t code = 0; code < kPositiveValues.size(); ++code) {
    const float error = std::fabs(kPositiveValues[code] - value);
    if (error < best_error) {
      best_error = error;
      best_code = static_cast<std::uint8_t>(code);
      continue;
    }
    if (error == best_error &&
        ((code & 1u) == 0u) &&
        ((best_code & 1u) != 0u)) {
      best_code = static_cast<std::uint8_t>(code);
    }
  }
  return best_code;
}

HostPackOutputs compute_reference_outputs(
    const std::vector<float>& input,
    int valid_rows,
    bool legacy_zero_block_scale) {
  HostPackOutputs output;
  output.packed.assign(kTileRows * kPackedBytesPerRow, 0u);
  output.block_scales.assign(kScaleCount, 0u);
  output.activation_output_scale.assign(kScaleCount, 0.0f);
  output.per_row_tensor_scales.assign(kTileRows, 1.0f);
  output.tensor_scale = 1.0f;

  for (int row = 0; row < kTileRows; ++row) {
    if (row >= valid_rows) {
      continue;
    }
    for (std::size_t block = 0; block < kBlocksPerRow; ++block) {
      const std::size_t input_offset = static_cast<std::size_t>(row) * kTileCols + block * kBlockWidth;
      const std::size_t scale_index = static_cast<std::size_t>(row) * kBlocksPerRow + block;
      const std::size_t packed_offset =
          static_cast<std::size_t>(row) * kPackedBytesPerRow + block * kPackedBytesPerBlock;
      float activated_block[kBlockWidth];
      float block_max_abs = 0.0f;
      for (std::size_t i = 0; i < kBlockWidth; ++i) {
        activated_block[i] = relu2(input[input_offset + i]);
        block_max_abs = std::max(block_max_abs, activated_block[i]);
      }

      float dequant_scale = 0.0f;
      if (legacy_zero_block_scale) {
        dequant_scale =
            block_max_abs > 0.0f ? clamp_nvfp4_scale(block_max_abs / kNvfp4Fp4MaxFinite) : 1.0f;
      } else {
        dequant_scale = clamp_nvfp4_scale(block_max_abs / kNvfp4Fp4MaxFinite);
      }
      output.activation_output_scale[scale_index] = dequant_scale;
      output.block_scales[scale_index] = encode_fp8_scale_reference(dequant_scale);

      for (std::size_t pair = 0; pair < kPackedBytesPerBlock; ++pair) {
        const std::uint8_t lhs = encode_fp4_reference(activated_block[pair * 2u] / dequant_scale);
        const std::uint8_t rhs =
            encode_fp4_reference(activated_block[pair * 2u + 1u] / dequant_scale);
        output.packed[packed_offset + pair] =
            static_cast<std::uint8_t>((lhs & 0x0Fu) | ((rhs & 0x0Fu) << 4u));
      }
    }
  }

  output.matmul_block_scales = SwizzleRowMajorNvfp4ScalesForExecution(
      output.block_scales.data(),
      kTileRows,
      kTileCols,
      kScaleLayout);
  return output;
}

HostPackOutputs compute_reference_outputs_with_per_row_scales(
    const std::vector<float>& input,
    const std::vector<float>& per_row_tensor_scales,
    int valid_rows,
    bool legacy_zero_block_scale) {
  HostPackOutputs output;
  output.packed.assign(kTileRows * kPackedBytesPerRow, 0u);
  output.block_scales.assign(kScaleCount, 0u);
  output.activation_output_scale.assign(kScaleCount, 0.0f);
  output.per_row_tensor_scales = per_row_tensor_scales;
  output.tensor_scale = 1.0f;

  for (int row = 0; row < kTileRows; ++row) {
    if (row >= valid_rows) {
      continue;
    }
    for (std::size_t block = 0; block < kBlocksPerRow; ++block) {
      const std::size_t input_offset =
          static_cast<std::size_t>(row) * kTileCols + block * kBlockWidth;
      const std::size_t scale_index = static_cast<std::size_t>(row) * kBlocksPerRow + block;
      const std::size_t packed_offset =
          static_cast<std::size_t>(row) * kPackedBytesPerRow + block * kPackedBytesPerBlock;
      float activated_block[kBlockWidth];
      float block_max_abs = 0.0f;
      for (std::size_t i = 0; i < kBlockWidth; ++i) {
        activated_block[i] =
            relu2(input[input_offset + i] * output.per_row_tensor_scales[row] * output.tensor_scale);
        block_max_abs = std::max(block_max_abs, activated_block[i]);
      }

      float dequant_scale = 0.0f;
      if (legacy_zero_block_scale) {
        dequant_scale =
            block_max_abs > 0.0f ? clamp_nvfp4_scale(block_max_abs / kNvfp4Fp4MaxFinite) : 1.0f;
      } else {
        dequant_scale = clamp_nvfp4_scale(block_max_abs / kNvfp4Fp4MaxFinite);
      }
      output.activation_output_scale[scale_index] = dequant_scale;
      output.block_scales[scale_index] = encode_fp8_scale_reference(dequant_scale);

      for (std::size_t pair = 0; pair < kPackedBytesPerBlock; ++pair) {
        const std::uint8_t lhs = encode_fp4_reference(activated_block[pair * 2u] / dequant_scale);
        const std::uint8_t rhs =
            encode_fp4_reference(activated_block[pair * 2u + 1u] / dequant_scale);
        output.packed[packed_offset + pair] =
            static_cast<std::uint8_t>((lhs & 0x0Fu) | ((rhs & 0x0Fu) << 4u));
      }
    }
  }

  output.matmul_block_scales = SwizzleRowMajorNvfp4ScalesForExecution(
      output.block_scales.data(),
      kTileRows,
      kTileCols,
      kScaleLayout);
  return output;
}

std::vector<float> make_mixed_input(bool bf16_exact) {
  std::vector<float> values(kTileElements, 0.0f);
  for (int row = 0; row < kTileRows; ++row) {
    for (std::size_t block = 0; block < kBlocksPerRow; ++block) {
      for (std::size_t offset = 0; offset < kBlockWidth; ++offset) {
        const std::size_t col = block * kBlockWidth + offset;
        float value = 0.0f;
        if (offset == 0u) {
          value = bf16_exact
                      ? (0.5f + 0.125f * static_cast<float>((row + static_cast<int>(block)) % 5))
                      : (0.43f + 0.097f * static_cast<float>((row + static_cast<int>(block)) % 5));
        } else {
          const int selector = (row * 19 + static_cast<int>(col) * 7 + static_cast<int>(offset)) % 9;
          if (bf16_exact) {
            switch (selector) {
              case 0:
                value = 0.0f;
                break;
              case 1:
                value = -0.25f * static_cast<float>((row % 5) + 1);
                break;
              case 2:
                value = 0.125f * static_cast<float>((static_cast<int>(col) % 7) + 1);
                break;
              case 3:
                value = -0.5f * static_cast<float>((static_cast<int>(block) % 3) + 1);
                break;
              case 4:
                value = 0.75f;
                break;
              case 5:
                value = -1.0f;
                break;
              case 6:
                value = 0.3125f * static_cast<float>((row % 4) + 1);
                break;
              case 7:
                value = -0.1875f * static_cast<float>((static_cast<int>(col) % 4) + 1);
                break;
              default:
                value = 0.0f;
                break;
            }
          } else {
            switch (selector) {
              case 0:
                value = 0.0f;
                break;
              case 1:
                value = -0.21f * static_cast<float>((row % 5) + 1);
                break;
              case 2:
                value = 0.17f * static_cast<float>((static_cast<int>(col) % 7) + 1);
                break;
              case 3:
                value = -0.46f * static_cast<float>((static_cast<int>(block) % 3) + 1);
                break;
              case 4:
                value = 0.81f;
                break;
              case 5:
                value = -1.13f;
                break;
              case 6:
                value = 0.29f * static_cast<float>((row % 4) + 1);
                break;
              case 7:
                value = -0.14f * static_cast<float>((static_cast<int>(col) % 4) + 1);
                break;
              default:
                value = 0.0f;
                break;
            }
          }
        }
        values[static_cast<std::size_t>(row) * kTileCols + col] = value;
      }
    }
  }
  return values;
}

std::vector<float> make_large_input(bool bf16_exact) {
  std::vector<float> values(kTileElements, 0.0f);
  for (int row = 0; row < kTileRows; ++row) {
    for (std::size_t block = 0; block < kBlocksPerRow; ++block) {
      for (std::size_t offset = 0; offset < kBlockWidth; ++offset) {
        const std::size_t col = block * kBlockWidth + offset;
        float value = 0.0f;
        if (offset == 0u || offset == 5u) {
          value = bf16_exact
                      ? (64.0f + 8.0f * static_cast<float>((row + static_cast<int>(block) + static_cast<int>(offset)) % 4))
                      : (63.5f + 7.25f * static_cast<float>((row + static_cast<int>(block) + static_cast<int>(offset)) % 4));
        } else if (offset == 1u) {
          value = bf16_exact ? -32.0f : -31.75f;
        } else if (offset == 2u) {
          value = bf16_exact ? 4.0f : 3.875f;
        } else if (offset == 3u) {
          value = 0.0f;
        } else {
          const float magnitude = bf16_exact
                                      ? (0.5f * static_cast<float>(((row + static_cast<int>(col)) % 6) + 1))
                                      : (0.47f * static_cast<float>(((row + static_cast<int>(col)) % 6) + 1));
          value = ((row + static_cast<int>(col)) % 2 == 0) ? magnitude : -magnitude;
        }
        values[static_cast<std::size_t>(row) * kTileCols + col] = value;
      }
    }
  }
  return values;
}

std::vector<float> make_non_uniform_per_row_scales() {
  std::vector<float> values(kTileRows, 1.0f);
  for (int row = 0; row < kTileRows; ++row) {
    values[row] = 0.5f + 0.1f * static_cast<float>(row % 8);
  }
  return values;
}

bool copy_per_row_scales(const DeviceNvfp4Matrix& pack, std::vector<float>* output) {
  output->assign(kTileRows, 0.0f);
  return check_cuda(
      cudaMemcpy(
          output->data(),
          pack.per_row_tensor_scales(),
          output->size() * sizeof(float),
          cudaMemcpyDeviceToHost),
      "cudaMemcpy(per_row_tensor_scales)");
}

HostPackOutputs run_staged_kernel(
    const std::vector<float>& input,
    int valid_rows,
    const Nvfp4ExecutionScaleLayout& scale_layout,
    bool* ok) {
  HostPackOutputs output;
  *ok = false;

  auto device_input = DeviceTensorFp32::Create({kTileRows, kTileCols});
  auto device_activation_scales = DeviceTensorFp32::Create({kTileRows, kBlocksPerRow});
  auto device_pack = DeviceNvfp4Matrix::Create(kTileRows, kTileCols, kScaleLayout);
  if (!device_input || !device_input->valid() ||
      !device_activation_scales || !device_activation_scales->valid() ||
      !device_pack || !device_pack->valid()) {
    return output;
  }

  if (!device_input->CopyFromHost(input.data(), input.size())) {
    return output;
  }

  TestStagedFp4PackKernel<<<1, kThreadsPerBlock>>>(
      device_input->data(),
      valid_rows,
      scale_layout.padded_blocks_per_row,
      device_activation_scales->data(),
      const_cast<std::uint8_t*>(device_pack->packed_data()),
      const_cast<std::uint8_t*>(device_pack->block_scales_data()),
      const_cast<std::uint8_t*>(device_pack->matmul_block_scales_data()),
      const_cast<float*>(device_pack->device_tensor_scale_ptr()),
      const_cast<float*>(device_pack->per_row_tensor_scales()));
  if (!check_cuda(cudaGetLastError(), "TestStagedFp4PackKernel launch") ||
      !check_cuda(cudaDeviceSynchronize(), "TestStagedFp4PackKernel sync")) {
    return output;
  }

  output.activation_output_scale.assign(kScaleCount, 0.0f);
  if (!device_pack->CopyPackedToHost(&output.packed) ||
      !device_pack->CopyBlockScalesToHost(&output.block_scales) ||
      !device_pack->CopyMatmulBlockScalesToHost(&output.matmul_block_scales) ||
      !device_pack->CopyTensorScaleToHost(&output.tensor_scale) ||
      !device_activation_scales->CopyToHost(
          output.activation_output_scale.data(),
          output.activation_output_scale.size()) ||
      !copy_per_row_scales(*device_pack, &output.per_row_tensor_scales)) {
    return output;
  }

  *ok = true;
  return output;
}

HostPackOutputs run_staged_kernel_with_per_row_scales(
    const std::vector<float>& input,
    const std::vector<float>& per_row_tensor_scales_input,
    int valid_rows,
    const Nvfp4ExecutionScaleLayout& scale_layout,
    bool* ok) {
  HostPackOutputs output;
  *ok = false;

  auto device_input = DeviceTensorFp32::Create({kTileRows, kTileCols});
  auto device_per_row_tensor_scales = DeviceTensorFp32::Create({kTileRows});
  auto device_activation_scales = DeviceTensorFp32::Create({kTileRows, kBlocksPerRow});
  auto device_pack = DeviceNvfp4Matrix::Create(kTileRows, kTileCols, kScaleLayout);
  if (!device_input || !device_input->valid() ||
      !device_per_row_tensor_scales || !device_per_row_tensor_scales->valid() ||
      !device_activation_scales || !device_activation_scales->valid() ||
      !device_pack || !device_pack->valid()) {
    return output;
  }

  if (!device_input->CopyFromHost(input.data(), input.size()) ||
      !device_per_row_tensor_scales->CopyFromHost(
          per_row_tensor_scales_input.data(),
          per_row_tensor_scales_input.size())) {
    return output;
  }

  TestScaledStagedFp4PackKernel<<<1, kThreadsPerBlock>>>(
      device_input->data(),
      device_per_row_tensor_scales->data(),
      valid_rows,
      scale_layout.padded_blocks_per_row,
      device_activation_scales->data(),
      const_cast<std::uint8_t*>(device_pack->packed_data()),
      const_cast<std::uint8_t*>(device_pack->block_scales_data()),
      const_cast<std::uint8_t*>(device_pack->matmul_block_scales_data()),
      const_cast<float*>(device_pack->device_tensor_scale_ptr()),
      const_cast<float*>(device_pack->per_row_tensor_scales()));
  if (!check_cuda(cudaGetLastError(), "TestScaledStagedFp4PackKernel launch") ||
      !check_cuda(cudaDeviceSynchronize(), "TestScaledStagedFp4PackKernel sync")) {
    return output;
  }

  output.activation_output_scale.assign(kScaleCount, 0.0f);
  if (!device_pack->CopyPackedToHost(&output.packed) ||
      !device_pack->CopyBlockScalesToHost(&output.block_scales) ||
      !device_pack->CopyMatmulBlockScalesToHost(&output.matmul_block_scales) ||
      !device_pack->CopyTensorScaleToHost(&output.tensor_scale) ||
      !device_activation_scales->CopyToHost(
          output.activation_output_scale.data(),
          output.activation_output_scale.size()) ||
      !copy_per_row_scales(*device_pack, &output.per_row_tensor_scales)) {
    return output;
  }

  *ok = true;
  return output;
}

HostPackOutputs run_live_p5_native_kernel(
    const std::vector<float>& input,
    const std::vector<float>& per_row_tensor_scales_input,
    int valid_rows,
    const Nvfp4ExecutionScaleLayout& scale_layout,
    bool* ok) {
  HostPackOutputs output;
  *ok = false;

  auto device_input = DeviceTensorFp32::Create({kTileRows, kTileCols});
  auto device_per_row_tensor_scales = DeviceTensorFp32::Create({kTileRows});
  auto device_activation_scales = DeviceTensorFp32::Create({kTileRows, kBlocksPerRow});
  auto device_pack = DeviceNvfp4Matrix::Create(kTileRows, kTileCols, kScaleLayout);
  if (!device_input || !device_input->valid() ||
      !device_per_row_tensor_scales || !device_per_row_tensor_scales->valid() ||
      !device_activation_scales || !device_activation_scales->valid() ||
      !device_pack || !device_pack->valid()) {
    return output;
  }

  if (!device_input->CopyFromHost(input.data(), input.size()) ||
      !device_per_row_tensor_scales->CopyFromHost(
          per_row_tensor_scales_input.data(),
          per_row_tensor_scales_input.size())) {
    return output;
  }

  if (!RunP5NativeDirectPackOracleForTesting(
      device_input->data(),
      device_per_row_tensor_scales->data(),
      valid_rows,
      scale_layout.padded_blocks_per_row,
      kScaleLayout,
      device_activation_scales->data(),
      const_cast<std::uint8_t*>(device_pack->packed_data()),
      const_cast<std::uint8_t*>(device_pack->block_scales_data()),
      const_cast<std::uint8_t*>(device_pack->matmul_block_scales_data()),
      const_cast<float*>(device_pack->device_tensor_scale_ptr()),
      const_cast<float*>(device_pack->per_row_tensor_scales()))) {
    return output;
  }

  output.activation_output_scale.assign(kScaleCount, 0.0f);
  if (!device_pack->CopyPackedToHost(&output.packed) ||
      !device_pack->CopyBlockScalesToHost(&output.block_scales) ||
      !device_pack->CopyMatmulBlockScalesToHost(&output.matmul_block_scales) ||
      !device_pack->CopyTensorScaleToHost(&output.tensor_scale) ||
      !device_activation_scales->CopyToHost(
          output.activation_output_scale.data(),
          output.activation_output_scale.size()) ||
      !copy_per_row_scales(*device_pack, &output.per_row_tensor_scales)) {
    return output;
  }

  *ok = true;
  return output;
}

HostPackOutputs run_legacy_kernel(
    const std::vector<float>& input,
    int valid_rows,
    const Nvfp4ExecutionScaleLayout& scale_layout,
    bool* ok) {
  HostPackOutputs output;
  *ok = false;

  auto device_input = DeviceTensorBf16::Create({kTileRows, kTileCols});
  auto device_activation_scales = DeviceTensorFp32::Create({kTileRows, kBlocksPerRow});
  auto device_pack = DeviceNvfp4Matrix::Create(kTileRows, kTileCols, kScaleLayout);
  if (!device_input || !device_input->valid() ||
      !device_activation_scales || !device_activation_scales->valid() ||
      !device_pack || !device_pack->valid()) {
    return output;
  }

  std::vector<__nv_bfloat16> bf16_input(kTileElements);
  for (std::size_t i = 0; i < input.size(); ++i) {
    bf16_input[i] = __nv_bfloat16(input[i]);
  }
  if (!device_input->CopyFromHost(bf16_input.data(), bf16_input.size())) {
    return output;
  }

  TestLegacyBf16Relu2PackKernel<<<1, kThreadsPerBlock>>>(
      device_input->data(),
      valid_rows,
      scale_layout.padded_blocks_per_row,
      device_activation_scales->data(),
      const_cast<std::uint8_t*>(device_pack->packed_data()),
      const_cast<std::uint8_t*>(device_pack->block_scales_data()),
      const_cast<std::uint8_t*>(device_pack->matmul_block_scales_data()),
      const_cast<float*>(device_pack->device_tensor_scale_ptr()),
      const_cast<float*>(device_pack->per_row_tensor_scales()));
  if (!check_cuda(cudaGetLastError(), "TestLegacyBf16Relu2PackKernel launch") ||
      !check_cuda(cudaDeviceSynchronize(), "TestLegacyBf16Relu2PackKernel sync")) {
    return output;
  }

  output.activation_output_scale.assign(kScaleCount, 0.0f);
  if (!device_pack->CopyPackedToHost(&output.packed) ||
      !device_pack->CopyBlockScalesToHost(&output.block_scales) ||
      !device_pack->CopyMatmulBlockScalesToHost(&output.matmul_block_scales) ||
      !device_pack->CopyTensorScaleToHost(&output.tensor_scale) ||
      !device_activation_scales->CopyToHost(
          output.activation_output_scale.data(),
          output.activation_output_scale.size()) ||
      !copy_per_row_scales(*device_pack, &output.per_row_tensor_scales)) {
    return output;
  }

  *ok = true;
  return output;
}

bool verify_phase1_contract_outputs(
    const HostPackOutputs& actual,
    const HostPackOutputs& expected,
    const std::string& label) {
  return compare_byte_vectors(actual.packed, expected.packed, label + " packed_data") &&
         compare_byte_vectors(
             actual.block_scales,
             expected.block_scales,
             label + " block_scales_data") &&
         compare_byte_vectors(
             actual.matmul_block_scales,
             expected.matmul_block_scales,
             label + " matmul_block_scales_data") &&
         compare_float_vectors_bitwise(
             actual.activation_output_scale,
             expected.activation_output_scale,
             label + " activation_output_scale") &&
         compare_float_scalar_bitwise(actual.tensor_scale, 1.0f, label + " tensor_scale_data") &&
         compare_float_vectors_bitwise(
             actual.per_row_tensor_scales,
             std::vector<float>(kTileRows, 1.0f),
             label + " per_row_tensor_scales");
}

bool compare_all_outputs_bitwise(
    const HostPackOutputs& actual,
    const HostPackOutputs& expected,
    const std::string& label) {
  return compare_byte_vectors(actual.packed, expected.packed, label + " packed_data") &&
         compare_byte_vectors(
             actual.block_scales,
             expected.block_scales,
             label + " block_scales_data") &&
         compare_byte_vectors(
             actual.matmul_block_scales,
             expected.matmul_block_scales,
             label + " matmul_block_scales_data") &&
         compare_float_vectors_bitwise(
             actual.activation_output_scale,
             expected.activation_output_scale,
             label + " activation_output_scale") &&
         compare_float_scalar_bitwise(
             actual.tensor_scale,
             expected.tensor_scale,
             label + " tensor_scale_data") &&
         compare_float_vectors_bitwise(
             actual.per_row_tensor_scales,
             expected.per_row_tensor_scales,
             label + " per_row_tensor_scales");
}

bool test_staged_matches_cpu_reference(
    const std::vector<float>& input,
    int valid_rows,
    const std::string& label,
    const Nvfp4ExecutionScaleLayout& scale_layout) {
  bool ok = false;
  const HostPackOutputs gpu = run_staged_kernel(input, valid_rows, scale_layout, &ok);
  if (!expect(ok, label + " staged kernel execution should succeed")) {
    return false;
  }
  const HostPackOutputs reference = compute_reference_outputs(input, valid_rows, false);
  return verify_phase1_contract_outputs(gpu, reference, label);
}

bool test_direct_output_compare(
    const std::vector<float>& input,
    int valid_rows,
    const std::string& label,
    const Nvfp4ExecutionScaleLayout& scale_layout) {
  bool staged_ok = false;
  bool legacy_ok = false;
  const HostPackOutputs staged = run_staged_kernel(input, valid_rows, scale_layout, &staged_ok);
  const HostPackOutputs legacy = run_legacy_kernel(input, valid_rows, scale_layout, &legacy_ok);
  if (!expect(staged_ok, label + " staged kernel execution should succeed") ||
      !expect(legacy_ok, label + " legacy kernel execution should succeed")) {
    return false;
  }

  return compare_byte_vectors(staged.packed, legacy.packed, label + " packed_data") &&
         compare_byte_vectors(
             staged.block_scales,
             legacy.block_scales,
             label + " block_scales_data") &&
         compare_byte_vectors(
             staged.matmul_block_scales,
             legacy.matmul_block_scales,
             label + " matmul_block_scales_data") &&
         compare_float_vectors_bitwise(
             staged.activation_output_scale,
             legacy.activation_output_scale,
             label + " activation_output_scale");
}

bool test_live_p5_native_matches_staged(const Nvfp4ExecutionScaleLayout& scale_layout) {
  const std::vector<float> input = make_mixed_input(false);
  const std::vector<float> per_row_tensor_scales = make_non_uniform_per_row_scales();
  const std::vector<int> valid_rows_cases = {kTileRows, 1, 7, 64, 127};

  for (int valid_rows : valid_rows_cases) {
    const std::string label =
        "live_p5_native_matches_staged_valid_rows_" + std::to_string(valid_rows);
    bool staged_ok = false;
    bool live_ok = false;
    const HostPackOutputs staged = run_staged_kernel_with_per_row_scales(
        input,
        per_row_tensor_scales,
        valid_rows,
        scale_layout,
        &staged_ok);
    const HostPackOutputs live = run_live_p5_native_kernel(
        input,
        per_row_tensor_scales,
        valid_rows,
        scale_layout,
        &live_ok);
    if (!expect(staged_ok, label + " staged kernel execution should succeed") ||
        !expect(live_ok, label + " live P5 kernel execution should succeed")) {
      return false;
    }

    const HostPackOutputs reference = compute_reference_outputs_with_per_row_scales(
        input,
        per_row_tensor_scales,
        valid_rows,
        false);
    if (!compare_all_outputs_bitwise(staged, reference, label + " staged_reference") ||
        !compare_all_outputs_bitwise(live, reference, label + " live_reference") ||
        !compare_all_outputs_bitwise(live, staged, label + " live_staged")) {
      return false;
    }
  }

  return true;
}

}  // namespace

int main() {
  if (!has_cuda_device()) {
    std::cout << "staged_fp4_pack_test: SKIP (no CUDA device available)\n";
    return 0;
  }

  const auto scale_layout = BuildNvfp4ExecutionScaleLayout(kTileRows, kTileCols, kScaleLayout);
  if (!expect(scale_layout.has_value() && scale_layout->valid(),
              "execution scale layout should be valid for a 128x128 tile")) {
    return 1;
  }
  if (!expect(scale_layout->padded_blocks_per_row == round_up_to_4(kBlocksPerRow),
              "padded blocks per row should match the 128x4 swizzle contract")) {
    return 1;
  }

  const std::vector<int> partial_rows = {1, 7, 64, 127};

  if (!test_staged_matches_cpu_reference(
          make_mixed_input(false),
          kTileRows,
          "full_128x128_mixed_fp32",
          *scale_layout)) {
    return 1;
  }
  for (int valid_rows : partial_rows) {
    if (!test_staged_matches_cpu_reference(
            make_mixed_input(false),
            valid_rows,
            "partial_mixed_valid_rows_" + std::to_string(valid_rows),
            *scale_layout)) {
      return 1;
    }
  }
  if (!test_staged_matches_cpu_reference(
          make_large_input(false),
          kTileRows,
          "full_128x128_large_values",
          *scale_layout)) {
    return 1;
  }

  // The direct old/new harness keeps one positive value in every active 16-wide block so the
  // legacy routed path's zero-block special-case does not dominate the comparison.
  if (!test_direct_output_compare(
          make_mixed_input(true),
          kTileRows,
          "direct_compare_full_bf16_exact_mixed",
          *scale_layout)) {
    return 1;
  }
  for (int valid_rows : partial_rows) {
    if (!test_direct_output_compare(
            make_mixed_input(true),
            valid_rows,
            "direct_compare_partial_valid_rows_" + std::to_string(valid_rows),
            *scale_layout)) {
      return 1;
    }
  }
  if (!test_direct_output_compare(
          make_large_input(true),
          kTileRows,
          "direct_compare_full_bf16_exact_large",
          *scale_layout)) {
    return 1;
  }
  if (!test_live_p5_native_matches_staged(*scale_layout)) {
    return 1;
  }

  std::cout << "staged_fp4_pack_test: PASS\n";
  return 0;
}
