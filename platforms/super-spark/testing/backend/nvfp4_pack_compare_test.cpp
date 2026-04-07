#include "nemotron/nvfp4_packing.h"
#include "nemotron/nvfp4_scale_layout.h"
#include "nemotron/device_nvfp4_matrix.h"
#include "nemotron/device_tensor.h"

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

using nemotron::HostNvfp4Matrix;
using nemotron::DeviceNvfp4Matrix;
using nemotron::DeviceTensorBf16;
using nemotron::DeviceTensorFp32;
using nemotron::PackDeviceRowMajorBf16ToNvfp4;
using nemotron::PackDeviceRowMajorFp32ToNvfp4;
using nemotron::PackRowMajorFp32ToNvfp4;

constexpr std::size_t kBlockWidth = 16;
constexpr float kFp4MaxFinite = 6.0f;
constexpr float kFp8E4M3MaxFinite = 448.0f;
constexpr float kMinScale = 1.0f / 1024.0f;
constexpr float kFp4PositiveValues[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

std::optional<std::size_t> parse_env_uint(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || std::string(value).empty()) {
    return std::nullopt;
  }
  try {
    return static_cast<std::size_t>(std::stoull(value));
  } catch (...) {
    return std::nullopt;
  }
}

std::vector<std::uint8_t> read_file_bytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::vector<float> read_float_file(const std::filesystem::path& path) {
  const std::vector<std::uint8_t> bytes = read_file_bytes(path);
  if (bytes.size() % sizeof(float) != 0) {
    return {};
  }
  std::vector<float> values(bytes.size() / sizeof(float), 0.0f);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

void write_file_bytes(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
}

void write_scalar_float(const std::filesystem::path& path, float value) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(&value), static_cast<std::streamsize>(sizeof(value)));
}

std::optional<float> parse_env_float(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || std::string(value).empty()) {
    return std::nullopt;
  }
  try {
    return std::stof(value);
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<std::string> parse_env_string(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || std::string(value).empty()) {
    return std::nullopt;
  }
  return std::string(value);
}

float clamp_scale(float value) {
  if (!std::isfinite(value) || value < kMinScale) {
    return kMinScale;
  }
  return value;
}

std::uint8_t encode_fp4_reference(float value) {
  const float magnitude = std::fabs(value);
  std::uint8_t best_index = 0;
  float best_error = std::numeric_limits<float>::infinity();
  for (std::uint8_t index = 0; index < 8; ++index) {
    const float error = std::fabs(magnitude - kFp4PositiveValues[index]);
    if (error < best_error) {
      best_error = error;
      best_index = index;
    }
  }
  return static_cast<std::uint8_t>(best_index | (value < 0.0f ? 0x8u : 0x0u));
}

std::uint8_t encode_fp8_scale(float value) {
  return static_cast<std::uint8_t>(
      __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

float decode_fp4(std::uint8_t code) {
  __nv_fp4_e2m1 value;
  value.__x = code & 0x0Fu;
  return static_cast<float>(value);
}

float decode_fp8(std::uint8_t code) {
  __nv_fp8_e4m3 value;
  value.__x = code;
  return static_cast<float>(value);
}

HostNvfp4Matrix pack_reference(const std::vector<float>& data, std::size_t rows, std::size_t cols) {
  HostNvfp4Matrix matrix;
  matrix.rows = rows;
  matrix.cols = cols;
  matrix.packed.resize((rows * cols) / 2u, 0u);
  matrix.block_scales.resize(rows * (cols / kBlockWidth), 0u);

  float global_max_abs = 0.0f;
  for (float value : data) {
    global_max_abs = std::max(global_max_abs, std::fabs(value));
  }
  if (global_max_abs > kFp4MaxFinite * kFp8E4M3MaxFinite) {
    matrix.tensor_scale = clamp_scale(global_max_abs / (kFp4MaxFinite * kFp8E4M3MaxFinite));
  } else {
    matrix.tensor_scale = 1.0f;
  }

  std::size_t packed_index = 0;
  std::size_t scale_index = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    const float* row_data = data.data() + (row * cols);
    for (std::size_t block = 0; block < cols / kBlockWidth; ++block) {
      const std::size_t block_offset = block * kBlockWidth;
      float block_max_abs = 0.0f;
      for (std::size_t i = 0; i < kBlockWidth; ++i) {
        block_max_abs = std::max(block_max_abs, std::fabs(row_data[block_offset + i]));
      }
      float block_scale = 1.0f;
      if (block_max_abs > 0.0f) {
        block_scale = clamp_scale(block_max_abs / (kFp4MaxFinite * matrix.tensor_scale));
      }
      matrix.block_scales[scale_index++] = encode_fp8_scale(block_scale);
      const float scale = matrix.tensor_scale * block_scale;
      for (std::size_t i = 0; i < kBlockWidth; i += 2) {
        const std::uint8_t lhs = encode_fp4_reference(row_data[block_offset + i] / scale);
        const std::uint8_t rhs = encode_fp4_reference(row_data[block_offset + i + 1] / scale);
        matrix.packed[packed_index++] = static_cast<std::uint8_t>(lhs | (rhs << 4));
      }
    }
  }
  return matrix;
}

std::vector<float> dequantize(const HostNvfp4Matrix& matrix) {
  std::vector<float> output(matrix.rows * matrix.cols, 0.0f);
  std::size_t packed_index = 0;
  std::size_t scale_index = 0;
  for (std::size_t row = 0; row < matrix.rows; ++row) {
    for (std::size_t block = 0; block < matrix.cols / kBlockWidth; ++block) {
      const float block_scale = decode_fp8(matrix.block_scales[scale_index++]) * matrix.tensor_scale;
      const std::size_t col_start = block * kBlockWidth;
      for (std::size_t offset = 0; offset < kBlockWidth; offset += 2) {
        const std::uint8_t byte = matrix.packed[packed_index++];
        output[row * matrix.cols + col_start + offset] = decode_fp4(byte & 0x0F) * block_scale;
        output[row * matrix.cols + col_start + offset + 1] = decode_fp4((byte >> 4) & 0x0F) * block_scale;
      }
    }
  }
  return output;
}

std::size_t count_byte_differences(const std::vector<std::uint8_t>& lhs, const std::vector<std::uint8_t>& rhs) {
  if (lhs.size() != rhs.size()) {
    return std::numeric_limits<std::size_t>::max();
  }
  std::size_t count = 0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i]) {
      ++count;
    }
  }
  return count;
}

float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return std::numeric_limits<float>::infinity();
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
  }
  return max_diff;
}

std::vector<__nv_bfloat16> float_vector_to_bf16(const std::vector<float>& input) {
  std::vector<__nv_bfloat16> output(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    output[i] = __float2bfloat16(input[i]);
  }
  return output;
}

std::vector<float> bf16_vector_to_float(const std::vector<__nv_bfloat16>& input) {
  std::vector<float> output(input.size(), 0.0f);
  for (std::size_t i = 0; i < input.size(); ++i) {
    output[i] = __bfloat162float(input[i]);
  }
  return output;
}

}  // namespace

int main() {
  const char* input_bin_env = std::getenv("NEMOTRON_NVFP4_PACK_COMPARE_INPUT_BIN");
  const auto rows = parse_env_uint("NEMOTRON_NVFP4_PACK_COMPARE_ROWS");
  const auto cols = parse_env_uint("NEMOTRON_NVFP4_PACK_COMPARE_COLS");
  const auto fixed_tensor_scale =
      parse_env_float("NEMOTRON_NVFP4_PACK_COMPARE_FIXED_TENSOR_SCALE");
  const auto runtime_packer =
      parse_env_string("NEMOTRON_NVFP4_PACK_COMPARE_RUNTIME_PACKER");
  const bool precast_bf16 =
      std::getenv("NEMOTRON_NVFP4_PACK_COMPARE_PRECAST_BF16") != nullptr;
  const char* dump_packed_env = std::getenv("NEMOTRON_NVFP4_PACK_COMPARE_DUMP_PACKED_BIN");
  const char* dump_raw_scales_env =
      std::getenv("NEMOTRON_NVFP4_PACK_COMPARE_DUMP_RAW_SCALES_BIN");
  const char* dump_swizzled_scales_env =
      std::getenv("NEMOTRON_NVFP4_PACK_COMPARE_DUMP_SWIZZLED_SCALES_BIN");
  const char* dump_tensor_scale_env =
      std::getenv("NEMOTRON_NVFP4_PACK_COMPARE_DUMP_TENSOR_SCALE_BIN");
  const char* dump_matmul_scales_env =
      std::getenv("NEMOTRON_NVFP4_PACK_COMPARE_DUMP_MATMUL_SCALES_BIN");
  if (input_bin_env == nullptr || std::string(input_bin_env).empty() || !rows.has_value() || !cols.has_value()) {
    std::cout << "nvfp4_pack_compare_test: SKIP (set NEMOTRON_NVFP4_PACK_COMPARE_INPUT_BIN, _ROWS, _COLS)\n";
    return 0;
  }

  const std::filesystem::path input_path(input_bin_env);
  std::vector<float> data = read_float_file(input_path);
  if (!expect(std::filesystem::exists(input_path), "input file should exist") ||
      !expect(data.size() == (*rows) * (*cols), "input file should match rows*cols")) {
    return 1;
  }

  if (precast_bf16) {
    for (float& value : data) {
      value = __bfloat162float(__float2bfloat16(value));
    }
  }

  const nemotron::Nvfp4PackOptions options =
      fixed_tensor_scale.has_value()
          ? nemotron::Nvfp4PackOptions{fixed_tensor_scale}
          : nemotron::Nvfp4PackOptions{};

  const std::string runtime_packer_name =
      runtime_packer.has_value() ? *runtime_packer : "host_fp32";
  std::vector<float> effective_reference_data = data;
  std::vector<std::uint8_t> runtime_packed_bytes;
  std::vector<std::uint8_t> runtime_block_scale_bytes;
  std::vector<std::uint8_t> runtime_matmul_scale_bytes;
  float runtime_tensor_scale = 0.0f;

  if (runtime_packer_name == "host_fp32") {
    const auto runtime_pack = PackRowMajorFp32ToNvfp4(
        data.data(),
        *rows,
        *cols,
        options);
    if (!expect(runtime_pack.has_value(), "host runtime packer should succeed")) {
      return 1;
    }
    runtime_packed_bytes = runtime_pack->packed;
    runtime_block_scale_bytes = runtime_pack->block_scales;
    runtime_tensor_scale = runtime_pack->tensor_scale;
    runtime_matmul_scale_bytes = nemotron::SwizzleRowMajorNvfp4ScalesForExecution(
        runtime_block_scale_bytes.data(),
        *rows,
        *cols);
  } else if (runtime_packer_name == "device_fp32") {
    auto source = DeviceTensorFp32::Create({*rows, *cols});
    if (!expect(source != nullptr && source->valid(), "device fp32 source should allocate") ||
        !expect(source->CopyFromHost(data.data(), data.size()), "device fp32 source upload should succeed")) {
      return 1;
    }
    std::unique_ptr<DeviceNvfp4Matrix> runtime_pack = PackDeviceRowMajorFp32ToNvfp4(*source, options);
    if (!expect(runtime_pack != nullptr && runtime_pack->valid(),
                "device fp32 runtime packer should succeed")) {
      return 1;
    }
    if (!expect(runtime_pack->CopyPackedToHost(&runtime_packed_bytes),
                "device fp32 packed bytes should copy back") ||
        !expect(runtime_pack->CopyBlockScalesToHost(&runtime_block_scale_bytes),
                "device fp32 block scales should copy back") ||
        !expect(runtime_pack->CopyMatmulBlockScalesToHost(&runtime_matmul_scale_bytes),
                "device fp32 matmul scales should copy back") ||
        !expect(runtime_pack->CopyTensorScaleToHost(&runtime_tensor_scale),
                "device fp32 tensor scale should copy back")) {
      return 1;
    }
  } else if (runtime_packer_name == "device_bf16") {
    std::vector<__nv_bfloat16> bf16_data = float_vector_to_bf16(data);
    effective_reference_data = bf16_vector_to_float(bf16_data);
    auto source = DeviceTensorBf16::Create({*rows, *cols});
    if (!expect(source != nullptr && source->valid(), "device bf16 source should allocate") ||
        !expect(source->CopyFromHost(bf16_data.data(), bf16_data.size()),
                "device bf16 source upload should succeed")) {
      return 1;
    }
    std::unique_ptr<DeviceNvfp4Matrix> runtime_pack = PackDeviceRowMajorBf16ToNvfp4(*source, options);
    if (!expect(runtime_pack != nullptr && runtime_pack->valid(),
                "device bf16 runtime packer should succeed")) {
      return 1;
    }
    if (!expect(runtime_pack->CopyPackedToHost(&runtime_packed_bytes),
                "device bf16 packed bytes should copy back") ||
        !expect(runtime_pack->CopyBlockScalesToHost(&runtime_block_scale_bytes),
                "device bf16 block scales should copy back") ||
        !expect(runtime_pack->CopyMatmulBlockScalesToHost(&runtime_matmul_scale_bytes),
                "device bf16 matmul scales should copy back") ||
        !expect(runtime_pack->CopyTensorScaleToHost(&runtime_tensor_scale),
                "device bf16 tensor scale should copy back")) {
      return 1;
    }
  } else {
    std::cerr << "FAIL: unsupported NEMOTRON_NVFP4_PACK_COMPARE_RUNTIME_PACKER="
              << runtime_packer_name << "\n";
    return 1;
  }

  const HostNvfp4Matrix reference_pack = pack_reference(effective_reference_data, *rows, *cols);

  const std::size_t packed_byte_diffs =
      count_byte_differences(runtime_packed_bytes, reference_pack.packed);
  const std::size_t scale_byte_diffs =
      count_byte_differences(runtime_block_scale_bytes, reference_pack.block_scales);
  const float tensor_scale_diff = std::fabs(runtime_tensor_scale - reference_pack.tensor_scale);
  const HostNvfp4Matrix runtime_pack_view{
      .rows = *rows,
      .cols = *cols,
      .packed = runtime_packed_bytes,
      .block_scales = runtime_block_scale_bytes,
      .tensor_scale = runtime_tensor_scale,
  };
  const auto runtime_dequant = dequantize(runtime_pack_view);
  const auto reference_dequant = dequantize(reference_pack);
  const float dequant_diff = max_abs_diff(runtime_dequant, reference_dequant);

  if (dump_packed_env != nullptr && std::string(dump_packed_env).size() != 0) {
    write_file_bytes(dump_packed_env, runtime_packed_bytes);
  }
  if (dump_raw_scales_env != nullptr && std::string(dump_raw_scales_env).size() != 0) {
    write_file_bytes(dump_raw_scales_env, runtime_block_scale_bytes);
  }
  if (dump_swizzled_scales_env != nullptr && std::string(dump_swizzled_scales_env).size() != 0) {
    const auto& swizzled = runtime_matmul_scale_bytes.empty()
                               ? nemotron::SwizzleRowMajorNvfp4ScalesForExecution(
                                     runtime_block_scale_bytes.data(),
                                     *rows,
                                     *cols)
                               : runtime_matmul_scale_bytes;
    write_file_bytes(dump_swizzled_scales_env, swizzled);
  }
  if (dump_tensor_scale_env != nullptr && std::string(dump_tensor_scale_env).size() != 0) {
    write_scalar_float(dump_tensor_scale_env, runtime_tensor_scale);
  }
  if (dump_matmul_scales_env != nullptr && std::string(dump_matmul_scales_env).size() != 0) {
    const auto& matmul_scales = runtime_matmul_scale_bytes.empty()
                                    ? nemotron::SwizzleRowMajorNvfp4ScalesForExecution(
                                          runtime_block_scale_bytes.data(),
                                          *rows,
                                          *cols)
                                    : runtime_matmul_scale_bytes;
    write_file_bytes(dump_matmul_scales_env, matmul_scales);
  }

  std::cout << "nvfp4_pack_compare_test:"
            << " rows=" << *rows
            << " cols=" << *cols
            << " runtime_packer=" << runtime_packer_name
            << " precast_bf16=" << (precast_bf16 ? 1 : 0)
            << " fixed_tensor_scale="
            << (fixed_tensor_scale.has_value() ? std::to_string(*fixed_tensor_scale) : "dynamic")
            << " packed_byte_diffs=" << packed_byte_diffs
            << " scale_byte_diffs=" << scale_byte_diffs
            << " tensor_scale_diff=" << tensor_scale_diff
            << " dequant_max_abs_diff=" << dequant_diff
            << "\n";
  return 0;
}
