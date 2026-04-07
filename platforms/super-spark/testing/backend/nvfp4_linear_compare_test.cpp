#include "nemotron/nvfp4_packing.h"

#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

using nemotron::PackRowMajorFp32ToNvfp4;

constexpr std::size_t kBlockWidth = 16;

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

bool parse_env_flag(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && std::string(value) == "1";
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

float read_scalar_float(const std::filesystem::path& path) {
  const std::vector<float> values = read_float_file(path);
  return values.size() == 1 ? values[0] : 0.0f;
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

std::vector<float> dequantize_nvfp4_matrix(
    const std::uint8_t* packed,
    std::size_t packed_nbytes,
    const std::uint8_t* block_scales,
    std::size_t block_scales_nbytes,
    float tensor_scale,
    std::size_t rows,
    std::size_t cols) {
  if (packed == nullptr ||
      block_scales == nullptr ||
      cols == 0 ||
      cols % kBlockWidth != 0 ||
      packed_nbytes != (rows * cols) / 2u ||
      block_scales_nbytes != rows * (cols / kBlockWidth)) {
    return {};
  }
  std::vector<float> output(rows * cols, 0.0f);
  std::size_t packed_index = 0;
  std::size_t scale_index = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t block = 0; block < cols / kBlockWidth; ++block) {
      const float block_scale = decode_fp8(block_scales[scale_index++]) * tensor_scale;
      const std::size_t col_start = block * kBlockWidth;
      for (std::size_t offset = 0; offset < kBlockWidth; offset += 2) {
        const std::uint8_t byte = packed[packed_index++];
        output[row * cols + col_start + offset] = decode_fp4(byte & 0x0F) * block_scale;
        output[row * cols + col_start + offset + 1] = decode_fp4((byte >> 4) & 0x0F) * block_scale;
      }
    }
  }
  return output;
}

std::vector<float> cpu_matmul_row_major(
    const std::vector<float>& activations,
    std::size_t rows,
    const std::vector<float>& weights,
    std::size_t output_rows,
    std::size_t input_cols) {
  std::vector<float> output(rows * output_rows, 0.0f);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t out = 0; out < output_rows; ++out) {
      float accum = 0.0f;
      for (std::size_t col = 0; col < input_cols; ++col) {
        accum += activations[row * input_cols + col] * weights[out * input_cols + col];
      }
      output[row * output_rows + out] = accum;
    }
  }
  return output;
}

void apply_relu2(std::vector<float>* values) {
  for (float& value : *values) {
    value = value > 0.0f ? value * value : 0.0f;
  }
}

float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return INFINITY;
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
  }
  return max_diff;
}

}  // namespace

int main() {
  const char* input_bin_env = std::getenv("NEMOTRON_NVFP4_LINEAR_COMPARE_INPUT_BIN");
  const char* weight_packed_env = std::getenv("NEMOTRON_NVFP4_LINEAR_COMPARE_WEIGHT_PACKED_BIN");
  const char* weight_scales_env = std::getenv("NEMOTRON_NVFP4_LINEAR_COMPARE_WEIGHT_SCALES_BIN");
  const char* weight_tensor_scale_env = std::getenv("NEMOTRON_NVFP4_LINEAR_COMPARE_WEIGHT_TENSOR_SCALE_BIN");
  const char* expected_output_env = std::getenv("NEMOTRON_NVFP4_LINEAR_COMPARE_EXPECTED_OUTPUT_BIN");
  const auto rows = parse_env_uint("NEMOTRON_NVFP4_LINEAR_COMPARE_ROWS");
  const auto input_cols = parse_env_uint("NEMOTRON_NVFP4_LINEAR_COMPARE_INPUT_COLS");
  const auto output_rows = parse_env_uint("NEMOTRON_NVFP4_LINEAR_COMPARE_OUTPUT_ROWS");
  const bool relu2 = parse_env_flag("NEMOTRON_NVFP4_LINEAR_COMPARE_RELU2");

  if (input_bin_env == nullptr || weight_packed_env == nullptr || weight_scales_env == nullptr ||
      weight_tensor_scale_env == nullptr || !rows.has_value() || !input_cols.has_value() ||
      !output_rows.has_value()) {
    std::cout << "nvfp4_linear_compare_test: SKIP (set input/weight env vars plus ROWS, INPUT_COLS, OUTPUT_ROWS)\n";
    return 0;
  }

  const std::vector<float> input = read_float_file(input_bin_env);
  const std::vector<std::uint8_t> weight_packed = read_file_bytes(weight_packed_env);
  const std::vector<std::uint8_t> weight_scales = read_file_bytes(weight_scales_env);
  const float weight_tensor_scale = read_scalar_float(weight_tensor_scale_env);
  if (!expect(input.size() == (*rows) * (*input_cols), "input size should match rows*input_cols") ||
      !expect(!weight_packed.empty(), "weight packed bytes should load") ||
      !expect(!weight_scales.empty(), "weight scale bytes should load")) {
    return 1;
  }

  const auto activation_pack = PackRowMajorFp32ToNvfp4(input.data(), *rows, *input_cols);
  if (!expect(activation_pack.has_value(), "activation pack should succeed")) {
    return 1;
  }
  const std::vector<float> activation_dequant =
      dequantize_nvfp4_matrix(
          activation_pack->packed_data(),
          activation_pack->packed_nbytes(),
          activation_pack->block_scales_data(),
          activation_pack->block_scales_nbytes(),
          activation_pack->tensor_scale,
          *rows,
          *input_cols);
  const std::vector<float> weight_dequant =
      dequantize_nvfp4_matrix(
          weight_packed.data(),
          weight_packed.size(),
          weight_scales.data(),
          weight_scales.size(),
          weight_tensor_scale,
          *output_rows,
          *input_cols);
  if (!expect(!activation_dequant.empty(), "activation dequant should succeed") ||
      !expect(!weight_dequant.empty(), "weight dequant should succeed")) {
    return 1;
  }

  std::vector<float> output =
      cpu_matmul_row_major(activation_dequant, *rows, weight_dequant, *output_rows, *input_cols);
  if (relu2) {
    apply_relu2(&output);
  }

  float expected_diff = 0.0f;
  if (expected_output_env != nullptr && std::string(expected_output_env).size() != 0) {
    const std::vector<float> expected = read_float_file(expected_output_env);
    if (!expect(expected.size() == output.size(), "expected output size should match actual output")) {
      return 1;
    }
    expected_diff = max_abs_diff(output, expected);
  }

  std::cout << "nvfp4_linear_compare_test:"
            << " rows=" << *rows
            << " input_cols=" << *input_cols
            << " output_rows=" << *output_rows
            << " relu2=" << (relu2 ? 1 : 0)
            << " expected_max_abs_diff=" << expected_diff
            << "\n";
  return 0;
}
