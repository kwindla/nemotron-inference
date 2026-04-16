#include "nemotron/fused_moe_prefill.h"

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

constexpr const char* kSourceRoot = NEMOTRON_SOURCE_ROOT;
constexpr int kNumRows = 128;
constexpr int kHiddenSize = 256;
constexpr int kInterSize = 256;
constexpr int kNvfp4BlockWidth = 16;
constexpr int kPackedRowBytes = kHiddenSize / 2;
constexpr int kScaleBytesPerRow = kHiddenSize / kNvfp4BlockWidth;
constexpr int kNanoNumRows = 128;
constexpr int kNanoHiddenSize = 2688;
constexpr int kNanoInterSize = 1920;
constexpr int kNanoPackedRowBytes = kNanoHiddenSize / 2;
constexpr int kNanoScaleBytesPerRow = kNanoHiddenSize / kNvfp4BlockWidth;
constexpr float kNvfp4Fp4MaxFinite = 6.0f;
constexpr float kNvfp4ActivationMaxFinite = 6.0f * 448.0f;
constexpr float kNvfp4MinScale = 1.0f / 1024.0f;

using nemotron::RunNanoP1DirectPackKernelForTesting;

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  bool Allocate(std::size_t count) {
    if (count == 0) {
      return true;
    }
    count_ = count;
    return cudaMalloc(&data_, count * sizeof(T)) == cudaSuccess;
  }

  T* data() const { return data_; }
  std::size_t size() const { return count_; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

struct DirectPackOutputs {
  std::vector<std::uint8_t> packed_bytes;
  std::vector<std::uint8_t> block_scales;
  std::vector<std::uint8_t> matmul_block_scales;
  std::vector<float> activation_output_scales;
};

bool CheckCuda(cudaError_t status, const char* what) {
  if (status == cudaSuccess) {
    return true;
  }
  std::printf(
      "nano_p1_direct_pack_oracle_test: CUDA failure at %s: %s\n",
      what,
      cudaGetErrorString(status));
  return false;
}

bool HasCudaDevice() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

bool BuildSourcePath(const char* relative_path, char* out, std::size_t out_size) {
  if (std::snprintf(out, out_size, "%s/%s", kSourceRoot, relative_path) >=
      static_cast<int>(out_size)) {
    return false;
  }
  return true;
}

bool ReadBinaryFile(const char* relative_path, std::vector<std::uint8_t>* bytes) {
  char full_path[1024];
  if (!BuildSourcePath(relative_path, full_path, sizeof(full_path))) {
    std::printf("nano_p1_direct_pack_oracle_test: path too long for %s\n", relative_path);
    return false;
  }
  std::ifstream input(full_path, std::ios::binary);
  if (!input) {
    std::printf("nano_p1_direct_pack_oracle_test: failed to open %s\n", full_path);
    return false;
  }
  input.seekg(0, std::ios::end);
  const std::streamsize size = input.tellg();
  if (size < 0) {
    std::printf("nano_p1_direct_pack_oracle_test: failed to stat %s\n", full_path);
    return false;
  }
  input.seekg(0, std::ios::beg);
  bytes->assign(static_cast<std::size_t>(size), 0u);
  if (size > 0) {
    input.read(reinterpret_cast<char*>(bytes->data()), size);
    if (!input) {
      std::printf("nano_p1_direct_pack_oracle_test: failed to read %s\n", full_path);
      return false;
    }
  }
  return true;
}

template <typename T>
bool ReadTypedFile(
    const char* relative_path,
    std::size_t expected_count,
    std::vector<T>* values) {
  std::vector<std::uint8_t> bytes;
  if (!ReadBinaryFile(relative_path, &bytes)) {
    return false;
  }
  if (bytes.size() != expected_count * sizeof(T)) {
    std::printf(
        "nano_p1_direct_pack_oracle_test: size mismatch for %s: got=%zu expected=%zu\n",
        relative_path,
        bytes.size(),
        expected_count * sizeof(T));
    return false;
  }
  values->assign(expected_count, T{});
  if (!bytes.empty()) {
    std::memcpy(values->data(), bytes.data(), bytes.size());
  }
  return true;
}

std::size_t RoundUp(std::size_t value, std::size_t alignment) {
  return alignment == 0 ? value : ((value + alignment - 1u) / alignment) * alignment;
}

std::size_t ExecutionScaleOffset(
    std::size_t row,
    std::size_t block_col,
    std::size_t padded_blocks_per_row) {
  const std::size_t num_k_tiles = padded_blocks_per_row / 4u;
  const std::size_t k_tile = block_col / 4u;
  const std::size_t inner_k = block_col & 3u;
  const std::size_t m_tile = row / 128u;
  const std::size_t outer_m = row & 31u;
  const std::size_t inner_m = (row >> 5u) & 3u;
  return ((((m_tile * num_k_tiles) + k_tile) << 9u) |
          (outer_m << 4u) |
          (inner_m << 2u) |
          inner_k);
}

std::vector<std::uint8_t> SwizzleRowMajorScalesForExecution(
    const std::vector<std::uint8_t>& row_major_scales,
    int rows,
    int cols) {
  const std::size_t logical_blocks_per_row = static_cast<std::size_t>(cols / kNvfp4BlockWidth);
  const std::size_t padded_blocks_per_row = RoundUp(logical_blocks_per_row, 4u);
  std::vector<std::uint8_t> execution_scales(
      static_cast<std::size_t>(rows) * padded_blocks_per_row,
      std::uint8_t{0});
  for (int row = 0; row < rows; ++row) {
    for (std::size_t block = 0; block < logical_blocks_per_row; ++block) {
      execution_scales[ExecutionScaleOffset(
          static_cast<std::size_t>(row),
          block,
          padded_blocks_per_row)] =
          row_major_scales[static_cast<std::size_t>(row) * logical_blocks_per_row + block];
    }
  }
  return execution_scales;
}

float DecodeFp4(std::uint8_t raw_nibble) {
  __nv_fp4_e2m1 value;
  value.__x = raw_nibble & 0x0Fu;
  return static_cast<float>(value);
}

float DecodeFp8(std::uint8_t raw_byte) {
  __nv_fp8_e4m3 value;
  value.__x = raw_byte;
  return static_cast<float>(value);
}

std::uint8_t EncodeFp4(float value) {
  return static_cast<std::uint8_t>(
             __nv_cvt_float_to_fp4(value, __NV_E2M1, cudaRoundNearest)) &
         0x0Fu;
}

std::uint8_t EncodeFp8(float value) {
  return static_cast<std::uint8_t>(
      __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

float ClampNvfp4Scale(float value) {
  if (!std::isfinite(value) || value < kNvfp4MinScale) {
    return kNvfp4MinScale;
  }
  return value;
}

float ClampNvfp4TensorScale(float value) {
  if (!std::isfinite(value) || value < kNvfp4MinScale) {
    return kNvfp4MinScale;
  }
  return value;
}

enum class ScaleOverrideMode {
  kCaptured = 0,
  kUnit = 1,
  kZero = 2,
};

enum class ScaleOverrideRegion {
  kAll = 0,
  kLo4 = 1,
  kHi4 = 2,
  kSlot0 = 3,
  kSlot1 = 4,
  kSlot2 = 5,
  kSlot3 = 6,
  kSlot4 = 7,
  kSlot5 = 8,
  kSlot6 = 9,
  kSlot7 = 10,
};

ScaleOverrideMode ParseScaleOverrideMode(const char* env_name) {
  const char* value = std::getenv(env_name);
  if (value == nullptr || value[0] == '\0' || std::strcmp(value, "captured") == 0) {
    return ScaleOverrideMode::kCaptured;
  }
  if (std::strcmp(value, "unit") == 0 || std::strcmp(value, "ones") == 0) {
    return ScaleOverrideMode::kUnit;
  }
  if (std::strcmp(value, "zero") == 0 || std::strcmp(value, "zeros") == 0) {
    return ScaleOverrideMode::kZero;
  }
  std::printf(
      "nano_p1_direct_pack_oracle_test: unknown %s=%s, expected captured|unit|zero\n",
      env_name,
      value);
  return ScaleOverrideMode::kCaptured;
}

const char* ScaleOverrideModeName(ScaleOverrideMode mode) {
  switch (mode) {
    case ScaleOverrideMode::kCaptured:
      return "captured";
    case ScaleOverrideMode::kUnit:
      return "unit";
    case ScaleOverrideMode::kZero:
      return "zero";
  }
  return "unknown";
}

ScaleOverrideRegion ParseScaleOverrideRegion(const char* env_name) {
  const char* value = std::getenv(env_name);
  if (value == nullptr || value[0] == '\0' || std::strcmp(value, "all") == 0) {
    return ScaleOverrideRegion::kAll;
  }
  if (std::strcmp(value, "lo4") == 0 || std::strcmp(value, "low4") == 0) {
    return ScaleOverrideRegion::kLo4;
  }
  if (std::strcmp(value, "hi4") == 0 || std::strcmp(value, "high4") == 0) {
    return ScaleOverrideRegion::kHi4;
  }
  if (std::strcmp(value, "slot0") == 0) {
    return ScaleOverrideRegion::kSlot0;
  }
  if (std::strcmp(value, "slot1") == 0) {
    return ScaleOverrideRegion::kSlot1;
  }
  if (std::strcmp(value, "slot2") == 0) {
    return ScaleOverrideRegion::kSlot2;
  }
  if (std::strcmp(value, "slot3") == 0) {
    return ScaleOverrideRegion::kSlot3;
  }
  if (std::strcmp(value, "slot4") == 0) {
    return ScaleOverrideRegion::kSlot4;
  }
  if (std::strcmp(value, "slot5") == 0) {
    return ScaleOverrideRegion::kSlot5;
  }
  if (std::strcmp(value, "slot6") == 0) {
    return ScaleOverrideRegion::kSlot6;
  }
  if (std::strcmp(value, "slot7") == 0) {
    return ScaleOverrideRegion::kSlot7;
  }
  std::printf(
      "nano_p1_direct_pack_oracle_test: unknown %s=%s, expected all|lo4|hi4|slot0..slot7\n",
      env_name,
      value);
  return ScaleOverrideRegion::kAll;
}

const char* ScaleOverrideRegionName(ScaleOverrideRegion region) {
  switch (region) {
    case ScaleOverrideRegion::kAll:
      return "all";
    case ScaleOverrideRegion::kLo4:
      return "lo4";
    case ScaleOverrideRegion::kHi4:
      return "hi4";
    case ScaleOverrideRegion::kSlot0:
      return "slot0";
    case ScaleOverrideRegion::kSlot1:
      return "slot1";
    case ScaleOverrideRegion::kSlot2:
      return "slot2";
    case ScaleOverrideRegion::kSlot3:
      return "slot3";
    case ScaleOverrideRegion::kSlot4:
      return "slot4";
    case ScaleOverrideRegion::kSlot5:
      return "slot5";
    case ScaleOverrideRegion::kSlot6:
      return "slot6";
    case ScaleOverrideRegion::kSlot7:
      return "slot7";
  }
  return "unknown";
}

bool ScaleOverrideAppliesToBlock(std::size_t block, ScaleOverrideRegion region) {
  const std::size_t slot = block & 7u;
  switch (region) {
    case ScaleOverrideRegion::kAll:
      return true;
    case ScaleOverrideRegion::kLo4:
      return slot < 4u;
    case ScaleOverrideRegion::kHi4:
      return slot >= 4u;
    case ScaleOverrideRegion::kSlot0:
      return slot == 0u;
    case ScaleOverrideRegion::kSlot1:
      return slot == 1u;
    case ScaleOverrideRegion::kSlot2:
      return slot == 2u;
    case ScaleOverrideRegion::kSlot3:
      return slot == 3u;
    case ScaleOverrideRegion::kSlot4:
      return slot == 4u;
    case ScaleOverrideRegion::kSlot5:
      return slot == 5u;
    case ScaleOverrideRegion::kSlot6:
      return slot == 6u;
    case ScaleOverrideRegion::kSlot7:
      return slot == 7u;
  }
  return true;
}

void ApplyScaleOverride(
    std::vector<std::uint8_t>* scales,
    ScaleOverrideMode mode,
    int rows,
    int cols,
    ScaleOverrideRegion region) {
  if (scales == nullptr || mode == ScaleOverrideMode::kCaptured) {
    return;
  }
  const std::uint8_t fill =
      mode == ScaleOverrideMode::kUnit ? EncodeFp8(1.0f) : static_cast<std::uint8_t>(0u);
  const std::size_t logical_blocks_per_row = static_cast<std::size_t>(cols / kNvfp4BlockWidth);
  const std::size_t padded_blocks_per_row = RoundUp(logical_blocks_per_row, 4u);
  for (int row = 0; row < rows; ++row) {
    for (std::size_t block = 0; block < logical_blocks_per_row; ++block) {
      if (!ScaleOverrideAppliesToBlock(block, region)) {
        continue;
      }
      (*scales)[ExecutionScaleOffset(
          static_cast<std::size_t>(row),
          block,
          padded_blocks_per_row)] = fill;
    }
  }
}

float Relu2(float value) {
  return value > 0.0f ? value * value : 0.0f;
}

std::vector<float> DequantizeNvfp4Matrix(
    const std::vector<std::uint8_t>& packed,
    const std::vector<std::uint8_t>& row_major_scales,
    int rows,
    int cols) {
  std::vector<float> output(static_cast<std::size_t>(rows * cols), 0.0f);
  std::size_t packed_index = 0;
  std::size_t scale_index = 0;
  for (int row = 0; row < rows; ++row) {
    for (int block = 0; block < cols / kNvfp4BlockWidth; ++block) {
      const float block_scale = DecodeFp8(row_major_scales[scale_index++]);
      const int col_start = block * kNvfp4BlockWidth;
      for (int offset = 0; offset < kNvfp4BlockWidth; offset += 2) {
        const std::uint8_t byte = packed[packed_index++];
        output[static_cast<std::size_t>(row * cols + col_start + offset + 0)] =
            DecodeFp4(byte & 0x0Fu) * block_scale;
        output[static_cast<std::size_t>(row * cols + col_start + offset + 1)] =
            DecodeFp4((byte >> 4) & 0x0Fu) * block_scale;
      }
    }
  }
  return output;
}

std::vector<float> DequantizeNvfp4MatrixFromExecutionScales(
    const std::vector<std::uint8_t>& packed,
    const std::vector<std::uint8_t>& execution_scales,
    int rows,
    int cols) {
  std::vector<float> output(static_cast<std::size_t>(rows * cols), 0.0f);
  const std::size_t logical_blocks_per_row = static_cast<std::size_t>(cols / kNvfp4BlockWidth);
  const std::size_t padded_blocks_per_row = RoundUp(logical_blocks_per_row, 4u);
  std::size_t packed_index = 0;
  for (int row = 0; row < rows; ++row) {
    for (std::size_t block = 0; block < logical_blocks_per_row; ++block) {
      const float block_scale = DecodeFp8(
          execution_scales[ExecutionScaleOffset(
              static_cast<std::size_t>(row),
              block,
              padded_blocks_per_row)]);
      const int col_start = static_cast<int>(block * kNvfp4BlockWidth);
      for (int offset = 0; offset < kNvfp4BlockWidth; offset += 2) {
        const std::uint8_t byte = packed[packed_index++];
        output[static_cast<std::size_t>(row * cols + col_start + offset + 0)] =
            DecodeFp4(byte & 0x0Fu) * block_scale;
        output[static_cast<std::size_t>(row * cols + col_start + offset + 1)] =
            DecodeFp4((byte >> 4) & 0x0Fu) * block_scale;
      }
    }
  }
  return output;
}

DirectPackOutputs BuildDirectPackReference(
    const std::vector<std::uint8_t>& input_fp4,
    const std::vector<std::uint8_t>& weight_fp4,
    const std::vector<std::uint8_t>& input_sf_row_major,
    const std::vector<std::uint8_t>& weight_sf_row_major,
    float alpha) {
  const std::size_t blocks_per_row = static_cast<std::size_t>(kInterSize / kNvfp4BlockWidth);
  const std::size_t padded_blocks_per_row = RoundUp(blocks_per_row, 4u);
  const std::size_t packed_row_bytes = static_cast<std::size_t>(kInterSize / 2);

  DirectPackOutputs output;
  output.packed_bytes.assign(static_cast<std::size_t>(kNumRows) * packed_row_bytes, 0u);
  output.block_scales.assign(static_cast<std::size_t>(kNumRows) * padded_blocks_per_row, 0u);
  output.matmul_block_scales.assign(
      static_cast<std::size_t>(kNumRows) * padded_blocks_per_row,
      0u);
  output.activation_output_scales.assign(static_cast<std::size_t>(kNumRows), 0.0f);

  const std::vector<float> activations =
      DequantizeNvfp4Matrix(input_fp4, input_sf_row_major, kNumRows, kHiddenSize);
  const std::vector<float> weights =
      DequantizeNvfp4Matrix(weight_fp4, weight_sf_row_major, kInterSize, kHiddenSize);
  std::vector<float> activated(static_cast<std::size_t>(kNumRows * kInterSize), 0.0f);

  for (int row = 0; row < kNumRows; ++row) {
    float row_max_abs = 0.0f;
    const float* activation_row = activations.data() + static_cast<std::size_t>(row * kHiddenSize);
    for (int col = 0; col < kInterSize; ++col) {
      const float* weight_row = weights.data() + static_cast<std::size_t>(col * kHiddenSize);
      double accum = 0.0;
      for (int k = 0; k < kHiddenSize; ++k) {
        accum += static_cast<double>(activation_row[k]) *
                 static_cast<double>(weight_row[k]);
      }
      const float value = Relu2(alpha * static_cast<float>(accum));
      activated[static_cast<std::size_t>(row * kInterSize + col)] = value;
      row_max_abs = std::max(row_max_abs, std::fabs(value));
    }
    float row_scale = 1.0f;
    if (row_max_abs > kNvfp4ActivationMaxFinite) {
      row_scale = ClampNvfp4TensorScale(row_max_abs / kNvfp4ActivationMaxFinite);
    }
    output.activation_output_scales[static_cast<std::size_t>(row)] = row_scale;
  }

  for (int row = 0; row < kNumRows; ++row) {
    const float row_scale = output.activation_output_scales[static_cast<std::size_t>(row)];
    for (std::size_t block = 0; block < blocks_per_row; ++block) {
      const std::size_t col_base = block * kNvfp4BlockWidth;
      float block_max_abs = 0.0f;
      for (int offset = 0; offset < kNvfp4BlockWidth; ++offset) {
        const float value =
            activated[static_cast<std::size_t>(row * kInterSize) + col_base + static_cast<std::size_t>(offset)];
        block_max_abs = std::max(block_max_abs, std::fabs(value));
      }

      const float raw_block_scale =
          ClampNvfp4Scale(block_max_abs / kNvfp4Fp4MaxFinite);
      const float stabilized_block_scale = ClampNvfp4Scale(
          block_max_abs / (kNvfp4Fp4MaxFinite * row_scale));
      const std::size_t scale_offset = ExecutionScaleOffset(
          static_cast<std::size_t>(row),
          block,
          padded_blocks_per_row);
      output.block_scales[scale_offset] = EncodeFp8(raw_block_scale);
      output.matmul_block_scales[scale_offset] = EncodeFp8(stabilized_block_scale);

      const float pack_scale = row_scale * stabilized_block_scale;
      const std::size_t packed_offset =
          static_cast<std::size_t>(row) * packed_row_bytes + (col_base / 2u);
      for (int pair = 0; pair < (kNvfp4BlockWidth / 2); ++pair) {
        const float lhs =
            activated[static_cast<std::size_t>(row * kInterSize) + col_base + static_cast<std::size_t>(pair * 2 + 0)];
        const float rhs =
            activated[static_cast<std::size_t>(row * kInterSize) + col_base + static_cast<std::size_t>(pair * 2 + 1)];
        const std::uint8_t lhs_nibble = EncodeFp4(lhs / pack_scale);
        const std::uint8_t rhs_nibble = EncodeFp4(rhs / pack_scale);
        output.packed_bytes[packed_offset + static_cast<std::size_t>(pair)] =
            static_cast<std::uint8_t>(
                (lhs_nibble & 0x0Fu) | ((rhs_nibble & 0x0Fu) << 4u));
      }
    }
  }

  return output;
}

DirectPackOutputs BuildDirectPackReferenceForShape(
    const std::vector<std::uint8_t>& input_fp4,
    const std::vector<std::uint8_t>& weight_fp4,
    const std::vector<std::uint8_t>& input_sf_exec,
    const std::vector<std::uint8_t>& weight_sf_exec,
    int num_rows,
    int hidden_size,
    int inter_size,
    float alpha) {
  const std::size_t blocks_per_row =
      static_cast<std::size_t>(inter_size / kNvfp4BlockWidth);
  const std::size_t padded_blocks_per_row = RoundUp(blocks_per_row, 4u);
  const std::size_t packed_row_bytes = static_cast<std::size_t>(inter_size / 2);

  DirectPackOutputs output;
  output.packed_bytes.assign(static_cast<std::size_t>(num_rows) * packed_row_bytes, 0u);
  output.block_scales.assign(static_cast<std::size_t>(num_rows) * padded_blocks_per_row, 0u);
  output.matmul_block_scales.assign(
      static_cast<std::size_t>(num_rows) * padded_blocks_per_row,
      0u);
  output.activation_output_scales.assign(static_cast<std::size_t>(num_rows), 0.0f);

  const std::vector<float> activations =
      DequantizeNvfp4MatrixFromExecutionScales(input_fp4, input_sf_exec, num_rows, hidden_size);
  const std::vector<float> weights =
      DequantizeNvfp4MatrixFromExecutionScales(weight_fp4, weight_sf_exec, inter_size, hidden_size);
  std::vector<float> activated(static_cast<std::size_t>(num_rows * inter_size), 0.0f);

  for (int row = 0; row < num_rows; ++row) {
    float row_max_abs = 0.0f;
    const float* activation_row =
        activations.data() + static_cast<std::size_t>(row * hidden_size);
    for (int col = 0; col < inter_size; ++col) {
      const float* weight_row = weights.data() + static_cast<std::size_t>(col * hidden_size);
      double accum = 0.0;
      for (int k = 0; k < hidden_size; ++k) {
        accum += static_cast<double>(activation_row[k]) *
                 static_cast<double>(weight_row[k]);
      }
      const float value = Relu2(alpha * static_cast<float>(accum));
      activated[static_cast<std::size_t>(row * inter_size + col)] = value;
      row_max_abs = std::max(row_max_abs, std::fabs(value));
    }
    float row_scale = 1.0f;
    if (row_max_abs > kNvfp4ActivationMaxFinite) {
      row_scale = ClampNvfp4TensorScale(row_max_abs / kNvfp4ActivationMaxFinite);
    }
    output.activation_output_scales[static_cast<std::size_t>(row)] = row_scale;
  }

  for (int row = 0; row < num_rows; ++row) {
    const float row_scale = output.activation_output_scales[static_cast<std::size_t>(row)];
    for (std::size_t block = 0; block < blocks_per_row; ++block) {
      const std::size_t col_base = block * kNvfp4BlockWidth;
      float block_max_abs = 0.0f;
      for (int offset = 0; offset < kNvfp4BlockWidth; ++offset) {
        const float value = activated[static_cast<std::size_t>(row * inter_size) +
                                      col_base + static_cast<std::size_t>(offset)];
        block_max_abs = std::max(block_max_abs, std::fabs(value));
      }

      const float raw_block_scale =
          ClampNvfp4Scale(block_max_abs / kNvfp4Fp4MaxFinite);
      const float stabilized_block_scale = ClampNvfp4Scale(
          block_max_abs / (kNvfp4Fp4MaxFinite * row_scale));
      const std::size_t scale_offset = ExecutionScaleOffset(
          static_cast<std::size_t>(row),
          block,
          padded_blocks_per_row);
      output.block_scales[scale_offset] = EncodeFp8(raw_block_scale);
      output.matmul_block_scales[scale_offset] = EncodeFp8(stabilized_block_scale);

      const float pack_scale = row_scale * stabilized_block_scale;
      const std::size_t packed_offset =
          static_cast<std::size_t>(row) * packed_row_bytes + (col_base / 2u);
      for (int pair = 0; pair < (kNvfp4BlockWidth / 2); ++pair) {
        const float lhs = activated[static_cast<std::size_t>(row * inter_size) +
                                    col_base + static_cast<std::size_t>(pair * 2 + 0)];
        const float rhs = activated[static_cast<std::size_t>(row * inter_size) +
                                    col_base + static_cast<std::size_t>(pair * 2 + 1)];
        const std::uint8_t lhs_nibble = EncodeFp4(lhs / pack_scale);
        const std::uint8_t rhs_nibble = EncodeFp4(rhs / pack_scale);
        output.packed_bytes[packed_offset + static_cast<std::size_t>(pair)] =
            static_cast<std::uint8_t>(
                (lhs_nibble & 0x0Fu) | ((rhs_nibble & 0x0Fu) << 4u));
      }
    }
  }

  return output;
}

bool RunNanoP1DirectPackKernel(
    const std::vector<std::uint8_t>& input_fp4,
    const std::vector<std::uint8_t>& weight_fp4,
    const std::vector<std::uint8_t>& input_sf_exec,
    const std::vector<std::uint8_t>& weight_sf_exec,
    const std::vector<float>& g1_alphas,
    DirectPackOutputs* output) {
  const std::size_t blocks_per_row = static_cast<std::size_t>(kInterSize / kNvfp4BlockWidth);
  const std::size_t padded_blocks_per_row = RoundUp(blocks_per_row, 4u);

  DeviceBuffer<std::uint8_t> input_fp4_dev;
  DeviceBuffer<std::uint8_t> weight_fp4_dev;
  DeviceBuffer<std::uint8_t> input_sf_dev;
  DeviceBuffer<std::uint8_t> weight_sf_dev;
  DeviceBuffer<float> g1_alphas_dev;
  DeviceBuffer<std::uint8_t> packed_output_dev;
  DeviceBuffer<std::uint8_t> block_scales_output_dev;
  DeviceBuffer<std::uint8_t> matmul_block_scales_output_dev;
  DeviceBuffer<float> activation_output_scales_dev;

  if (!input_fp4_dev.Allocate(input_fp4.size()) ||
      !weight_fp4_dev.Allocate(weight_fp4.size()) ||
      !input_sf_dev.Allocate(input_sf_exec.size()) ||
      !weight_sf_dev.Allocate(weight_sf_exec.size()) ||
      !g1_alphas_dev.Allocate(g1_alphas.size()) ||
      !packed_output_dev.Allocate(static_cast<std::size_t>(kNumRows * (kInterSize / 2))) ||
      !block_scales_output_dev.Allocate(static_cast<std::size_t>(kNumRows) * padded_blocks_per_row) ||
      !matmul_block_scales_output_dev.Allocate(static_cast<std::size_t>(kNumRows) * padded_blocks_per_row) ||
      !activation_output_scales_dev.Allocate(static_cast<std::size_t>(kNumRows))) {
    std::printf("nano_p1_direct_pack_oracle_test: cudaMalloc failed\n");
    return false;
  }

  if (!CheckCuda(
          cudaMemcpy(
              input_fp4_dev.data(),
              input_fp4.data(),
              input_fp4.size(),
              cudaMemcpyHostToDevice),
          "copy input_fp4") ||
      !CheckCuda(
          cudaMemcpy(
              weight_fp4_dev.data(),
              weight_fp4.data(),
              weight_fp4.size(),
              cudaMemcpyHostToDevice),
          "copy weight_fp4") ||
      !CheckCuda(
          cudaMemcpy(
              input_sf_dev.data(),
              input_sf_exec.data(),
              input_sf_exec.size(),
              cudaMemcpyHostToDevice),
          "copy input_sf") ||
      !CheckCuda(
          cudaMemcpy(
              weight_sf_dev.data(),
              weight_sf_exec.data(),
              weight_sf_exec.size(),
              cudaMemcpyHostToDevice),
          "copy weight_sf") ||
      !CheckCuda(
          cudaMemcpy(
              g1_alphas_dev.data(),
              g1_alphas.data(),
              g1_alphas.size() * sizeof(float),
              cudaMemcpyHostToDevice),
          "copy g1_alphas")) {
    return false;
  }

  if (!RunNanoP1DirectPackKernelForTesting(
          input_fp4_dev.data(),
          weight_fp4_dev.data(),
          input_sf_dev.data(),
          weight_sf_dev.data(),
          packed_output_dev.data(),
          block_scales_output_dev.data(),
          matmul_block_scales_output_dev.data(),
          activation_output_scales_dev.data(),
          g1_alphas_dev.data(),
          kNumRows,
          kHiddenSize,
          kInterSize,
          nullptr)) {
    std::printf(
        "nano_p1_direct_pack_oracle_test: RunNanoP1DirectPackKernelForTesting returned false\n");
    return false;
  }

  output->packed_bytes.assign(packed_output_dev.size(), 0u);
  output->block_scales.assign(block_scales_output_dev.size(), 0u);
  output->matmul_block_scales.assign(matmul_block_scales_output_dev.size(), 0u);
  output->activation_output_scales.assign(activation_output_scales_dev.size(), 0.0f);
  if (!CheckCuda(
          cudaMemcpy(
              output->packed_bytes.data(),
              packed_output_dev.data(),
              output->packed_bytes.size(),
              cudaMemcpyDeviceToHost),
          "copy packed_output") ||
      !CheckCuda(
          cudaMemcpy(
              output->block_scales.data(),
              block_scales_output_dev.data(),
              output->block_scales.size(),
              cudaMemcpyDeviceToHost),
          "copy block_scales_output") ||
      !CheckCuda(
          cudaMemcpy(
              output->matmul_block_scales.data(),
              matmul_block_scales_output_dev.data(),
              output->matmul_block_scales.size(),
              cudaMemcpyDeviceToHost),
          "copy matmul_block_scales_output") ||
      !CheckCuda(
          cudaMemcpy(
              output->activation_output_scales.data(),
              activation_output_scales_dev.data(),
              output->activation_output_scales.size() * sizeof(float),
              cudaMemcpyDeviceToHost),
          "copy activation_output_scales")) {
    return false;
  }
  return true;
}

bool RunNanoP1DirectPackKernelForShape(
    const std::vector<std::uint8_t>& input_fp4,
    const std::vector<std::uint8_t>& weight_fp4,
    const std::vector<std::uint8_t>& input_sf_exec,
    const std::vector<std::uint8_t>& weight_sf_exec,
    const std::vector<float>& g1_alphas,
    int num_rows,
    int hidden_size,
    int inter_size,
    DirectPackOutputs* output) {
  const std::size_t blocks_per_row =
      static_cast<std::size_t>(inter_size / kNvfp4BlockWidth);
  const std::size_t padded_blocks_per_row = RoundUp(blocks_per_row, 4u);
  const std::size_t packed_row_bytes = static_cast<std::size_t>(inter_size / 2);
  const std::size_t scale_bytes_per_row =
      static_cast<std::size_t>(hidden_size / kNvfp4BlockWidth);

  if (input_fp4.size() != static_cast<std::size_t>(num_rows) * (hidden_size / 2) ||
      weight_fp4.size() != static_cast<std::size_t>(inter_size) * (hidden_size / 2) ||
      input_sf_exec.size() != static_cast<std::size_t>(num_rows) * scale_bytes_per_row ||
      weight_sf_exec.size() != static_cast<std::size_t>(inter_size) * scale_bytes_per_row ||
      g1_alphas.size() != 1) {
    std::printf(
        "nano_p1_direct_pack_oracle_test: invalid host tensor sizes for shape rows=%d hidden=%d inter=%d\n",
        num_rows,
        hidden_size,
        inter_size);
    return false;
  }

  DeviceBuffer<std::uint8_t> input_fp4_dev;
  DeviceBuffer<std::uint8_t> weight_fp4_dev;
  DeviceBuffer<std::uint8_t> input_sf_dev;
  DeviceBuffer<std::uint8_t> weight_sf_dev;
  DeviceBuffer<float> g1_alphas_dev;
  DeviceBuffer<std::uint8_t> packed_output_dev;
  DeviceBuffer<std::uint8_t> block_scales_output_dev;
  DeviceBuffer<std::uint8_t> matmul_block_scales_output_dev;
  DeviceBuffer<float> activation_output_scales_dev;

  if (!input_fp4_dev.Allocate(input_fp4.size()) ||
      !weight_fp4_dev.Allocate(weight_fp4.size()) ||
      !input_sf_dev.Allocate(input_sf_exec.size()) ||
      !weight_sf_dev.Allocate(weight_sf_exec.size()) ||
      !g1_alphas_dev.Allocate(g1_alphas.size()) ||
      !packed_output_dev.Allocate(static_cast<std::size_t>(num_rows) * packed_row_bytes) ||
      !block_scales_output_dev.Allocate(static_cast<std::size_t>(num_rows) * padded_blocks_per_row) ||
      !matmul_block_scales_output_dev.Allocate(static_cast<std::size_t>(num_rows) * padded_blocks_per_row) ||
      !activation_output_scales_dev.Allocate(static_cast<std::size_t>(num_rows))) {
    std::printf("nano_p1_direct_pack_oracle_test: cudaMalloc failed\n");
    return false;
  }

  if (!CheckCuda(
          cudaMemcpy(
              input_fp4_dev.data(),
              input_fp4.data(),
              input_fp4.size(),
              cudaMemcpyHostToDevice),
          "copy input_fp4") ||
      !CheckCuda(
          cudaMemcpy(
              weight_fp4_dev.data(),
              weight_fp4.data(),
              weight_fp4.size(),
              cudaMemcpyHostToDevice),
          "copy weight_fp4") ||
      !CheckCuda(
          cudaMemcpy(
              input_sf_dev.data(),
              input_sf_exec.data(),
              input_sf_exec.size(),
              cudaMemcpyHostToDevice),
          "copy input_sf") ||
      !CheckCuda(
          cudaMemcpy(
              weight_sf_dev.data(),
              weight_sf_exec.data(),
              weight_sf_exec.size(),
              cudaMemcpyHostToDevice),
          "copy weight_sf") ||
      !CheckCuda(
          cudaMemcpy(
              g1_alphas_dev.data(),
              g1_alphas.data(),
              g1_alphas.size() * sizeof(float),
              cudaMemcpyHostToDevice),
          "copy g1_alphas")) {
    return false;
  }

  if (!RunNanoP1DirectPackKernelForTesting(
          input_fp4_dev.data(),
          weight_fp4_dev.data(),
          input_sf_dev.data(),
          weight_sf_dev.data(),
          packed_output_dev.data(),
          block_scales_output_dev.data(),
          matmul_block_scales_output_dev.data(),
          activation_output_scales_dev.data(),
          g1_alphas_dev.data(),
          num_rows,
          hidden_size,
          inter_size,
          nullptr)) {
    std::printf(
        "nano_p1_direct_pack_oracle_test: RunNanoP1DirectPackKernelForTesting returned false\n");
    return false;
  }

  output->packed_bytes.assign(packed_output_dev.size(), 0u);
  output->block_scales.assign(block_scales_output_dev.size(), 0u);
  output->matmul_block_scales.assign(matmul_block_scales_output_dev.size(), 0u);
  output->activation_output_scales.assign(activation_output_scales_dev.size(), 0.0f);
  if (!CheckCuda(
          cudaMemcpy(
              output->packed_bytes.data(),
              packed_output_dev.data(),
              output->packed_bytes.size(),
              cudaMemcpyDeviceToHost),
          "copy packed_output") ||
      !CheckCuda(
          cudaMemcpy(
              output->block_scales.data(),
              block_scales_output_dev.data(),
              output->block_scales.size(),
              cudaMemcpyDeviceToHost),
          "copy block_scales_output") ||
      !CheckCuda(
          cudaMemcpy(
              output->matmul_block_scales.data(),
              matmul_block_scales_output_dev.data(),
              output->matmul_block_scales.size(),
              cudaMemcpyDeviceToHost),
          "copy matmul_block_scales_output") ||
      !CheckCuda(
          cudaMemcpy(
              output->activation_output_scales.data(),
              activation_output_scales_dev.data(),
              output->activation_output_scales.size() * sizeof(float),
              cudaMemcpyDeviceToHost),
          "copy activation_output_scales")) {
    return false;
  }
  return true;
}

std::size_t ReportByteMismatches(
    const char* label,
    const std::vector<std::uint8_t>& actual,
    const std::vector<std::uint8_t>& expected) {
  if (actual.size() != expected.size()) {
    std::printf(
        "nano_p1_direct_pack_oracle_test: %s size mismatch actual=%zu expected=%zu\n",
        label,
        actual.size(),
        expected.size());
    return expected.size() > actual.size() ? expected.size() - actual.size()
                                           : actual.size() - expected.size();
  }
  std::size_t mismatch_count = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] == expected[i]) {
      continue;
    }
    if (mismatch_count < 8) {
      std::printf(
          "nano_p1_direct_pack_oracle_test: %s mismatch[%zu] index=%zu actual=0x%02x expected=0x%02x\n",
          label,
          mismatch_count,
          i,
          actual[i],
          expected[i]);
    }
    ++mismatch_count;
  }
  std::printf(
      "nano_p1_direct_pack_oracle_test: %s %s mismatches=%zu\n",
      label,
      mismatch_count == 0 ? "PASS" : "FAIL",
      mismatch_count);
  return mismatch_count;
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0u;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::size_t ReportFloatBitMismatches(
    const char* label,
    const std::vector<float>& actual,
    const std::vector<float>& expected) {
  if (actual.size() != expected.size()) {
    std::printf(
        "nano_p1_direct_pack_oracle_test: %s size mismatch actual=%zu expected=%zu\n",
        label,
        actual.size(),
        expected.size());
    return expected.size() > actual.size() ? expected.size() - actual.size()
                                           : actual.size() - expected.size();
  }
  std::size_t mismatch_count = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (FloatBits(actual[i]) == FloatBits(expected[i])) {
      continue;
    }
    if (mismatch_count < 8) {
      std::printf(
          "nano_p1_direct_pack_oracle_test: %s mismatch[%zu] row=%zu actual=%g (0x%08x) expected=%g (0x%08x)\n",
          label,
          mismatch_count,
          i,
          actual[i],
          FloatBits(actual[i]),
          expected[i],
          FloatBits(expected[i]));
    }
    ++mismatch_count;
  }
  std::printf(
      "nano_p1_direct_pack_oracle_test: %s %s mismatches=%zu\n",
      label,
      mismatch_count == 0 ? "PASS" : "FAIL",
      mismatch_count);
  return mismatch_count;
}

bool RunPhase1AllOnes() {
  const std::uint8_t packed_one = static_cast<std::uint8_t>(
      EncodeFp4(1.0f) | (EncodeFp4(1.0f) << 4));
  const std::uint8_t scale_one = EncodeFp8(1.0f);

  std::vector<std::uint8_t> input_fp4(static_cast<std::size_t>(kNumRows * kPackedRowBytes), packed_one);
  std::vector<std::uint8_t> weight_fp4(static_cast<std::size_t>(kInterSize * kPackedRowBytes), packed_one);
  std::vector<std::uint8_t> input_sf_row_major(
      static_cast<std::size_t>(kNumRows * kScaleBytesPerRow),
      scale_one);
  std::vector<std::uint8_t> weight_sf_row_major(
      static_cast<std::size_t>(kInterSize * kScaleBytesPerRow),
      scale_one);
  const std::vector<std::uint8_t> input_sf_exec =
      SwizzleRowMajorScalesForExecution(input_sf_row_major, kNumRows, kHiddenSize);
  const std::vector<std::uint8_t> weight_sf_exec =
      SwizzleRowMajorScalesForExecution(weight_sf_row_major, kInterSize, kHiddenSize);
  std::vector<float> g1_alphas(1, 1.0f);

  DirectPackOutputs actual;
  if (!RunNanoP1DirectPackKernel(
          input_fp4,
          weight_fp4,
          input_sf_exec,
          weight_sf_exec,
          g1_alphas,
          &actual)) {
    return false;
  }

  const DirectPackOutputs expected = BuildDirectPackReference(
      input_fp4,
      weight_fp4,
      input_sf_row_major,
      weight_sf_row_major,
      g1_alphas[0]);

  const std::size_t packed_mismatches = ReportByteMismatches(
      "Phase 1 packed_bytes",
      actual.packed_bytes,
      expected.packed_bytes);
  const std::size_t block_scale_mismatches = ReportByteMismatches(
      "Phase 1 block_scales",
      actual.block_scales,
      expected.block_scales);
  const std::size_t matmul_scale_mismatches = ReportByteMismatches(
      "Phase 1 matmul_block_scales",
      actual.matmul_block_scales,
      expected.matmul_block_scales);
  const std::size_t activation_scale_mismatches = ReportFloatBitMismatches(
      "Phase 1 activation_output_scales",
      actual.activation_output_scales,
      expected.activation_output_scales);
  return packed_mismatches == 0 &&
         block_scale_mismatches == 0 &&
         matmul_scale_mismatches == 0 &&
         activation_scale_mismatches == 0;
}

void PrintPhase2Deferred() {
  std::printf(
      "nano_p1_direct_pack_oracle_test: Phase 2 DEFERRED (capture-backed direct-pack oracle is deferred to step 5; Phase 1 synthetic all-ones covers the fused NanoP1 direct-pack contract bitwise)\n");
}

bool RunPhase3NanoBucketK2688() {
  std::vector<std::uint8_t> input_fp4;
  std::vector<std::uint8_t> input_sf;
  std::vector<std::uint8_t> weight_fp4;
  std::vector<std::uint8_t> weight_sf;
  std::vector<float> g1_alphas;
  if (!ReadBinaryFile(
          "proj-2026-04-12-1022/trtllm_reference/golden_nano_k2688/input_fp4_permuted.bin",
          &input_fp4) ||
      !ReadBinaryFile(
          "proj-2026-04-12-1022/trtllm_reference/golden_nano_k2688/input_sf_permuted.bin",
          &input_sf) ||
      !ReadBinaryFile(
          "proj-2026-04-12-1022/trtllm_reference/golden_nano_k2688/inputs_w1_fp4.bin",
          &weight_fp4) ||
      !ReadBinaryFile(
          "proj-2026-04-12-1022/trtllm_reference/golden_nano_k2688/inputs_w1_sf.bin",
          &weight_sf) ||
      !ReadTypedFile(
          "proj-2026-04-12-1022/trtllm_reference/golden_nano_k2688/inputs_g1_alphas.bin",
          1,
          &g1_alphas)) {
    return false;
  }

  if (input_fp4.size() != static_cast<std::size_t>(kNanoNumRows * kNanoPackedRowBytes) ||
      input_sf.size() != static_cast<std::size_t>(kNanoNumRows * kNanoScaleBytesPerRow) ||
      weight_fp4.size() != static_cast<std::size_t>(kNanoInterSize * kNanoPackedRowBytes) ||
      weight_sf.size() != static_cast<std::size_t>(kNanoInterSize * kNanoScaleBytesPerRow)) {
    std::printf("nano_p1_direct_pack_oracle_test: Phase 3 unexpected tensor sizes\n");
    return false;
  }

  const ScaleOverrideMode input_scale_mode =
      ParseScaleOverrideMode("NEMOTRON_NANO_P1_DIRECT_PACK_INPUT_SCALE_MODE");
  const ScaleOverrideMode weight_scale_mode =
      ParseScaleOverrideMode("NEMOTRON_NANO_P1_DIRECT_PACK_WEIGHT_SCALE_MODE");
  const ScaleOverrideRegion input_scale_region =
      ParseScaleOverrideRegion("NEMOTRON_NANO_P1_DIRECT_PACK_INPUT_SCALE_REGION");
  const ScaleOverrideRegion weight_scale_region =
      ParseScaleOverrideRegion("NEMOTRON_NANO_P1_DIRECT_PACK_WEIGHT_SCALE_REGION");
  ApplyScaleOverride(
      &input_sf,
      input_scale_mode,
      kNanoNumRows,
      kNanoHiddenSize,
      input_scale_region);
  ApplyScaleOverride(
      &weight_sf,
      weight_scale_mode,
      kNanoInterSize,
      kNanoHiddenSize,
      weight_scale_region);
  std::printf(
      "nano_p1_direct_pack_oracle_test: Phase 3 scale_modes input=%s/%s weight=%s/%s\n",
      ScaleOverrideModeName(input_scale_mode),
      ScaleOverrideRegionName(input_scale_region),
      ScaleOverrideModeName(weight_scale_mode),
      ScaleOverrideRegionName(weight_scale_region));

  DirectPackOutputs actual;
  if (!RunNanoP1DirectPackKernelForShape(
          input_fp4,
          weight_fp4,
          input_sf,
          weight_sf,
          g1_alphas,
          kNanoNumRows,
          kNanoHiddenSize,
          kNanoInterSize,
          &actual)) {
    return false;
  }

  // `proj-2026-04-12-1022/trtllm_reference/NOTES.md` §5
  // "Convergence persists at Nano K=2688" explains why the BF16 mainloop gate
  // can use `tactic1.bin` as a mathematical-consistency oracle at this shape.
  // TRT-LLM still stops at the BF16 GEMM boundary, so the fused direct-pack
  // stage remains validated against this local exact host reference rather than
  // as a "same CollectiveBuilder instantiation as TRT-LLM P1" claim. That
  // sharper identity check remains step 4a's compile-time probe gate.
  const DirectPackOutputs expected = BuildDirectPackReferenceForShape(
      input_fp4,
      weight_fp4,
      input_sf,
      weight_sf,
      kNanoNumRows,
      kNanoHiddenSize,
      kNanoInterSize,
      g1_alphas[0]);

  const std::size_t packed_mismatches = ReportByteMismatches(
      "Phase 3 packed_bytes",
      actual.packed_bytes,
      expected.packed_bytes);
  const std::size_t block_scale_mismatches = ReportByteMismatches(
      "Phase 3 block_scales",
      actual.block_scales,
      expected.block_scales);
  const std::size_t matmul_scale_mismatches = ReportByteMismatches(
      "Phase 3 matmul_block_scales",
      actual.matmul_block_scales,
      expected.matmul_block_scales);
  const std::size_t activation_scale_mismatches = ReportFloatBitMismatches(
      "Phase 3 activation_output_scales",
      actual.activation_output_scales,
      expected.activation_output_scales);
  return packed_mismatches == 0 &&
         block_scale_mismatches == 0 &&
         matmul_scale_mismatches == 0 &&
         activation_scale_mismatches == 0;
}

}  // namespace

int main() {
  if (!HasCudaDevice()) {
    std::printf("nano_p1_direct_pack_oracle_test: SKIP (no CUDA device available)\n");
    return 0;
  }

  if (!RunPhase1AllOnes()) {
    return 1;
  }

  PrintPhase2Deferred();
  if (!RunPhase3NanoBucketK2688()) {
    return 1;
  }
  std::printf("nano_p1_direct_pack_oracle_test: PASS\n");
  return 0;
}
