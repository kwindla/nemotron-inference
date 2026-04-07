#include "nemotron/cublaslt_gemm_plan.h"
#include "nemotron/cublaslt_handle.h"
#include "nemotron/dense_gemm_runner.h"
#include "nemotron/gemm_execution.h"
#include "nemotron/gemm_planner.h"

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

constexpr float kMinScale = 1.0f / 1024.0f;

using nemotron::BuildCublasLtGemmPlan;
using nemotron::BuildGemmLaunchPlan;
using nemotron::CublasLtHandle;
using nemotron::CublasLtGemmPlan;
using nemotron::DenseRowMajorHostResult;
using nemotron::DeviceTensorFp32;
using nemotron::GemmBackendKind;
using nemotron::GemmDescriptor;
using nemotron::GemmHeuristicCache;
using nemotron::GemmKernelFamily;
using nemotron::PrepareGemmExecution;
using nemotron::RunDenseRowMajorFp32;

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

float read_scalar_float(const std::filesystem::path& path) {
  const std::vector<float> values = read_float_file(path);
  return values.size() == 1 ? values[0] : 0.0f;
}

float clamp_scale(float value) {
  if (!std::isfinite(value) || value < kMinScale) {
    return kMinScale;
  }
  return value;
}

float decode_fp8(std::uint8_t raw_byte) {
  __nv_fp8_e4m3 value;
  value.__x = raw_byte;
  return static_cast<float>(value);
}

std::vector<float> quantize_fp32_to_scaled_fp8_roundtrip(
    const std::vector<float>& input,
    float input_scale) {
  const float scale = clamp_scale(input_scale);
  std::vector<float> output(input.size(), 0.0f);
  for (std::size_t i = 0; i < input.size(); ++i) {
    output[i] = decode_fp8(static_cast<std::uint8_t>(
                    __nv_cvt_float_to_fp8(input[i] / scale, __NV_SATFINITE, __NV_E4M3))) *
                scale;
  }
  return output;
}

std::vector<float> dequantize_scaled_fp8_weight(
    const std::vector<std::uint8_t>& packed_weight,
    float weight_scale) {
  std::vector<float> output(packed_weight.size(), 0.0f);
  for (std::size_t i = 0; i < packed_weight.size(); ++i) {
    output[i] = decode_fp8(packed_weight[i]) * weight_scale;
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

std::optional<CublasLtGemmPlan> build_runtime_gemm_plan(
    std::size_t output_rows,
    std::size_t input_cols,
    std::size_t rows,
    GemmHeuristicCache* heuristic_cache,
    const float* packed_weight_data) {
  GemmDescriptor descriptor;
  descriptor.tensor_name = "scaled_fp8_linear_compare";
  descriptor.op_class = "scaled_fp8_linear_compare";
  descriptor.kernel_family = GemmKernelFamily::kDenseRowMajor;
  descriptor.output_rows = output_rows;
  descriptor.input_cols = input_cols;
  descriptor.storage_dtype = "fp32";
  descriptor.compute_dtype = "fp32";
  descriptor.layout_tag = "row_major";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data = reinterpret_cast<const std::uint8_t*>(packed_weight_data);
  descriptor.packed_nbytes = output_rows * input_cols * sizeof(float);

  const auto launch_plan = BuildGemmLaunchPlan(descriptor, rows);
  if (!launch_plan.has_value()) {
    return std::nullopt;
  }
  const auto execution = PrepareGemmExecution(*launch_plan, heuristic_cache);
  if (!execution.has_value() || execution->backend_kind != GemmBackendKind::kCublasLtDense) {
    return std::nullopt;
  }
  return BuildCublasLtGemmPlan(*execution);
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
  const char* input_bin_env = std::getenv("NEMOTRON_SCALED_FP8_COMPARE_INPUT_BIN");
  const char* weight_fp8_env = std::getenv("NEMOTRON_SCALED_FP8_COMPARE_WEIGHT_FP8_BIN");
  const char* weight_scale_env = std::getenv("NEMOTRON_SCALED_FP8_COMPARE_WEIGHT_SCALE_BIN");
  const char* input_scale_env = std::getenv("NEMOTRON_SCALED_FP8_COMPARE_INPUT_SCALE_BIN");
  const char* expected_quantized_input_env =
      std::getenv("NEMOTRON_SCALED_FP8_COMPARE_EXPECTED_QUANTIZED_INPUT_BIN");
  const char* expected_weight_dequant_env =
      std::getenv("NEMOTRON_SCALED_FP8_COMPARE_EXPECTED_WEIGHT_DEQUANT_BIN");
  const char* expected_output_env = std::getenv("NEMOTRON_SCALED_FP8_COMPARE_EXPECTED_OUTPUT_BIN");
  const auto rows = parse_env_uint("NEMOTRON_SCALED_FP8_COMPARE_ROWS");
  const auto input_cols = parse_env_uint("NEMOTRON_SCALED_FP8_COMPARE_INPUT_COLS");
  const auto output_rows = parse_env_uint("NEMOTRON_SCALED_FP8_COMPARE_OUTPUT_ROWS");

  if (input_bin_env == nullptr || weight_fp8_env == nullptr || weight_scale_env == nullptr ||
      input_scale_env == nullptr || !rows.has_value() || !input_cols.has_value() ||
      !output_rows.has_value()) {
    std::cout << "scaled_fp8_linear_compare_test: SKIP (set input/weight env vars plus ROWS, INPUT_COLS, OUTPUT_ROWS)\n";
    return 0;
  }

  const std::vector<float> input = read_float_file(input_bin_env);
  const std::vector<std::uint8_t> weight_fp8 = read_file_bytes(weight_fp8_env);
  const float weight_scale = read_scalar_float(weight_scale_env);
  const float input_scale = read_scalar_float(input_scale_env);
  if (!expect(input.size() == (*rows) * (*input_cols), "input size should match rows*input_cols") ||
      !expect(weight_fp8.size() == (*output_rows) * (*input_cols), "weight size should match output_rows*input_cols") ||
      !expect(std::isfinite(weight_scale) && weight_scale > 0.0f, "weight scale should be finite and positive") ||
      !expect(std::isfinite(input_scale) && input_scale > 0.0f, "input scale should be finite and positive")) {
    return 1;
  }

  const std::vector<float> quantized_input =
      quantize_fp32_to_scaled_fp8_roundtrip(input, input_scale);
  const std::vector<float> weight_dequant =
      dequantize_scaled_fp8_weight(weight_fp8, weight_scale);
  const std::vector<float> output =
      cpu_matmul_row_major(quantized_input, *rows, weight_dequant, *output_rows, *input_cols);

  float quantized_input_diff = 0.0f;
  if (expected_quantized_input_env != nullptr && std::string(expected_quantized_input_env).size() != 0) {
    const std::vector<float> expected_quantized_input = read_float_file(expected_quantized_input_env);
    if (!expect(expected_quantized_input.size() == quantized_input.size(),
                "expected quantized input size should match actual quantized input")) {
      return 1;
    }
    quantized_input_diff = max_abs_diff(quantized_input, expected_quantized_input);
  }

  float weight_dequant_diff = 0.0f;
  if (expected_weight_dequant_env != nullptr && std::string(expected_weight_dequant_env).size() != 0) {
    const std::vector<float> expected_weight_dequant = read_float_file(expected_weight_dequant_env);
    if (!expect(expected_weight_dequant.size() == weight_dequant.size(),
                "expected weight dequant size should match actual weight dequant")) {
      return 1;
    }
    weight_dequant_diff = max_abs_diff(weight_dequant, expected_weight_dequant);
  }

  float output_diff = 0.0f;
  if (expected_output_env != nullptr && std::string(expected_output_env).size() != 0) {
    const std::vector<float> expected_output = read_float_file(expected_output_env);
    if (!expect(expected_output.size() == output.size(), "expected output size should match actual output")) {
      return 1;
    }
    output_diff = max_abs_diff(output, expected_output);
  }

  bool device_attempted = false;
  bool device_succeeded = false;
  float device_output_diff = 0.0f;
  if (expected_output_env != nullptr && std::string(expected_output_env).size() != 0) {
    auto handle = CublasLtHandle::Create();
    if (handle && handle->valid()) {
      GemmHeuristicCache heuristic_cache;
      const auto plan = build_runtime_gemm_plan(
          *output_rows,
          *input_cols,
          *rows,
          &heuristic_cache,
          weight_dequant.data());
      if (plan.has_value()) {
        device_attempted = true;
        const auto device_result = RunDenseRowMajorFp32(
            *handle,
            *plan,
            quantized_input.data(),
            *rows);
        if (device_result.has_value()) {
          const std::vector<float> expected_output = read_float_file(expected_output_env);
          if (!expect(expected_output.size() == device_result->output.size(),
                      "device expected output size should match actual output")) {
            return 1;
          }
          device_succeeded = true;
          device_output_diff = max_abs_diff(device_result->output, expected_output);
        }
      }
    }
  }

  std::cout << "scaled_fp8_linear_compare_test:"
            << " rows=" << *rows
            << " input_cols=" << *input_cols
            << " output_rows=" << *output_rows
            << " quantized_input_max_abs_diff=" << quantized_input_diff
            << " weight_dequant_max_abs_diff=" << weight_dequant_diff
            << " output_max_abs_diff=" << output_diff
            << " device_attempted=" << (device_attempted ? 1 : 0)
            << " device_succeeded=" << (device_succeeded ? 1 : 0)
            << " device_output_max_abs_diff=" << device_output_diff
            << "\n";
  return 0;
}
