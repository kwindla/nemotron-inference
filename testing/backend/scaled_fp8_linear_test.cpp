#include "nemotron/scaled_fp8_linear.h"
#include "nemotron/runtime_stats.h"

#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

using nemotron::CublasLtHandle;
using nemotron::DeviceTensorFp32;
using nemotron::GetRuntimeExecutionStatsSnapshot;
using nemotron::GemmHeuristicCache;
using nemotron::ResetRuntimeExecutionStats;
using nemotron::ScaledFp8LinearConfig;
using nemotron::ScaledFp8LinearOp;
using nemotron::DequantizeScaledFp8WeightToHostFp32;
using nemotron::QuantizeFp32ToScaledFp8RoundTrip;

class ScopedEnvVar {
 public:
  explicit ScopedEnvVar(const char* name) : name_(name) {
    if (const char* value = std::getenv(name_); value != nullptr) {
      original_value_ = value;
    }
  }

  ~ScopedEnvVar() {
    if (original_value_.has_value()) {
      setenv(name_, original_value_->c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }

  void Unset() const {
    unsetenv(name_);
  }

  void Set(const char* value) const {
    setenv(name_, value, 1);
  }

 private:
  const char* name_ = nullptr;
  std::optional<std::string> original_value_;
};

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

std::uint8_t encode_fp8(float value) {
  return static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

float decode_fp8(std::uint8_t raw_byte) {
  __nv_fp8_e4m3 value;
  value.__x = raw_byte;
  return static_cast<float>(value);
}

std::vector<float> cpu_reference(
    const std::vector<float>& activations,
    std::size_t rows,
    std::size_t input_cols,
    const std::vector<std::uint8_t>& weight_fp8,
    std::size_t output_rows,
    float weight_scale,
    float input_scale) {
  std::vector<float> output(rows * output_rows, 0.0f);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t out = 0; out < output_rows; ++out) {
      float accum = 0.0f;
      for (std::size_t col = 0; col < input_cols; ++col) {
        const float quantized_input = decode_fp8(encode_fp8(activations[row * input_cols + col] / input_scale)) * input_scale;
        const float weight = decode_fp8(weight_fp8[out * input_cols + col]) * weight_scale;
        accum += quantized_input * weight;
      }
      output[row * output_rows + out] = accum;
    }
  }
  return output;
}

bool nearly_equal(const std::vector<float>& lhs, const std::vector<float>& rhs, float tolerance) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
  }
  if (max_diff > tolerance) {
    std::cerr << "max_diff=" << max_diff << " exceeds tolerance=" << tolerance << "\n";
    return false;
  }
  return true;
}

std::vector<float> read_f32_file(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    return {};
  }
  stream.seekg(0, std::ios::end);
  const std::size_t bytes = static_cast<std::size_t>(stream.tellg());
  stream.seekg(0, std::ios::beg);
  if (bytes % sizeof(float) != 0) {
    return {};
  }
  std::vector<float> values(bytes / sizeof(float), 0.0f);
  stream.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
  return values;
}

std::vector<std::uint8_t> read_u8_file(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    return {};
  }
  stream.seekg(0, std::ios::end);
  const std::size_t bytes = static_cast<std::size_t>(stream.tellg());
  stream.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> values(bytes, 0u);
  stream.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
  return values;
}

float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return std::numeric_limits<float>::infinity();
  }
  float max_value = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_value = std::max(max_value, std::fabs(lhs[i] - rhs[i]));
  }
  return max_value;
}

bool run_small_scaled_fp8_case_and_check_stats(
    bool disable_dequantized_dense,
    std::uint64_t expected_dequantized_dense_success,
    std::uint64_t expected_reference_fallbacks,
    std::uint64_t expected_reference_fallbacks_expert_shared_up) {
  const auto handle = CublasLtHandle::Create();
  if (!handle || !handle->valid()) {
    std::cout << "scaled_fp8_linear_test: SKIP stats case (no CUDA device)\n";
    return true;
  }

  ScopedEnvVar native_fp8("NEMOTRON_ENABLE_EXPERIMENTAL_FP8_NATIVE");
  ScopedEnvVar disable_dequantized("NEMOTRON_DISABLE_SCALED_FP8_DEQUANTIZED_DENSE");
  ScopedEnvVar family_filter("NEMOTRON_EXPERIMENTAL_SCALED_FP8_SURFACE_FAMILY");
  ScopedEnvVar tensor_filter("NEMOTRON_EXPERIMENTAL_SCALED_FP8_SURFACE_TENSORS");
  native_fp8.Unset();
  disable_dequantized.Unset();
  family_filter.Unset();
  tensor_filter.Unset();
  if (disable_dequantized_dense) {
    disable_dequantized.Set("1");
  }

  ResetRuntimeExecutionStats();

  constexpr std::size_t kRows = 1;
  constexpr std::size_t kInputCols = 64;
  constexpr std::size_t kOutputRows = 32;
  constexpr float kWeightScale = 0.25f;
  constexpr float kInputScale = 0.125f;

  std::vector<float> activations(kRows * kInputCols, 0.0f);
  for (std::size_t i = 0; i < activations.size(); ++i) {
    activations[i] = (static_cast<float>((i * 11) % 19) - 9.0f) * 0.0625f;
  }

  std::vector<std::uint8_t> weights_fp8(kOutputRows * kInputCols, 0u);
  for (std::size_t i = 0; i < weights_fp8.size(); ++i) {
    const float value = (static_cast<float>((i * 5) % 17) - 8.0f) * 0.125f;
    weights_fp8[i] = encode_fp8(value);
  }

  ScaledFp8LinearConfig config;
  config.output_rows = kOutputRows;
  config.input_cols = kInputCols;
  config.packed_weight_data = weights_fp8.data();
  config.packed_weight_nbytes = weights_fp8.size();
  config.tensor_name = "model.layers.3.shared_experts.up_proj.weight";
  config.weight_scale = kWeightScale;
  config.input_scale = kInputScale;
  auto op = ScaledFp8LinearOp::Create(config);
  if (!expect(op != nullptr && op->valid(), "stats case scaled-fp8 op should build")) {
    return false;
  }

  auto input = DeviceTensorFp32::Create({kRows, kInputCols});
  auto output = DeviceTensorFp32::Create({kRows, kOutputRows});
  if (!expect(input != nullptr && input->valid(), "stats case input tensor should create") ||
      !expect(output != nullptr && output->valid(), "stats case output tensor should create") ||
      !expect(input->CopyFromHost(activations.data(), activations.size()), "stats case input should upload")) {
    return false;
  }

  GemmHeuristicCache heuristic_cache;
  if (!expect(op->Run(*handle, &heuristic_cache, *input, output.get()), "stats case op should execute")) {
    return false;
  }

  std::vector<float> actual(output->numel(), 0.0f);
  if (!expect(output->CopyToHost(actual.data(), actual.size()), "stats case output should download")) {
    return false;
  }

  const std::vector<float> expected =
      cpu_reference(activations, kRows, kInputCols, weights_fp8, kOutputRows, kWeightScale, kInputScale);
  if (!expect(
          nearly_equal(actual, expected, 1.0e-4f),
          "stats case scaled-fp8 output should match CPU reference")) {
    return false;
  }

  const auto stats = GetRuntimeExecutionStatsSnapshot();
  return expect(
             stats.scaled_fp8_dequantized_dense_success == expected_dequantized_dense_success,
             "scaled-fp8 dequantized dense success count should match") &&
         expect(
             stats.scaled_fp8_reference_fallbacks == expected_reference_fallbacks,
             "scaled-fp8 reference fallback count should match") &&
         expect(
             stats.scaled_fp8_reference_fallbacks_expert_shared_up ==
                 expected_reference_fallbacks_expert_shared_up,
             "scaled-fp8 shared-up fallback family count should match");
}

bool test_scaled_fp8_linear_matches_cpu_reference() {
  const auto handle = CublasLtHandle::Create();
  if (!handle || !handle->valid()) {
    std::cout << "scaled_fp8_linear_test: SKIP (no CUDA device)\n";
    return true;
  }

  constexpr std::size_t kRows = 16;
  constexpr std::size_t kInputCols = 64;
  constexpr std::size_t kOutputRows = 64;
  constexpr float kWeightScale = 0.25f;
  constexpr float kInputScale = 0.125f;

  std::vector<float> activations(kRows * kInputCols, 0.0f);
  for (std::size_t i = 0; i < activations.size(); ++i) {
    activations[i] = (static_cast<float>((i * 13) % 29) - 14.0f) * 0.0625f;
  }

  std::vector<std::uint8_t> weights_fp8(kOutputRows * kInputCols, 0u);
  for (std::size_t i = 0; i < weights_fp8.size(); ++i) {
    const float value = (static_cast<float>((i * 7) % 23) - 11.0f) * 0.125f;
    weights_fp8[i] = encode_fp8(value);
  }

  ScaledFp8LinearConfig config;
  config.output_rows = kOutputRows;
  config.input_cols = kInputCols;
  config.packed_weight_data = weights_fp8.data();
  config.packed_weight_nbytes = weights_fp8.size();
  config.weight_scale = kWeightScale;
  config.input_scale = kInputScale;
  auto op = ScaledFp8LinearOp::Create(config);
  if (!expect(op != nullptr && op->valid(), "scaled fp8 linear op should build")) {
    return false;
  }

  auto input = DeviceTensorFp32::Create({kRows, kInputCols});
  auto output = DeviceTensorFp32::Create({kRows, kOutputRows});
  if (!expect(input != nullptr && input->valid(), "input tensor should create") ||
      !expect(output != nullptr && output->valid(), "output tensor should create") ||
      !expect(input->CopyFromHost(activations.data(), activations.size()), "input tensor should upload")) {
    return false;
  }

  GemmHeuristicCache heuristic_cache;
  if (!expect(op->Run(*handle, &heuristic_cache, *input, output.get()), "scaled fp8 linear op should execute")) {
    return false;
  }

  std::vector<float> actual(output->numel(), 0.0f);
  if (!expect(output->CopyToHost(actual.data(), actual.size()), "output tensor should download")) {
    return false;
  }

  const std::vector<float> expected =
      cpu_reference(activations, kRows, kInputCols, weights_fp8, kOutputRows, kWeightScale, kInputScale);
  return expect(nearly_equal(actual, expected, 1.0e-4f), "scaled fp8 linear output should match CPU reference");
}

bool test_scaled_fp8_linear_defaults_to_dequantized_dense_single_token() {
  return run_small_scaled_fp8_case_and_check_stats(/*disable_dequantized_dense=*/false,
                                                   /*expected_dequantized_dense_success=*/1,
                                                   /*expected_reference_fallbacks=*/0,
                                                   /*expected_reference_fallbacks_expert_shared_up=*/0);
}

bool test_scaled_fp8_linear_disable_dequantized_dense_uses_reference_fallback() {
  return run_small_scaled_fp8_case_and_check_stats(/*disable_dequantized_dense=*/true,
                                                   /*expected_dequantized_dense_success=*/0,
                                                   /*expected_reference_fallbacks=*/1,
                                                   /*expected_reference_fallbacks_expert_shared_up=*/1);
}

bool test_scaled_fp8_linear_matches_oracle_fixture() {
  const auto handle = CublasLtHandle::Create();
  if (!handle || !handle->valid()) {
    std::cout << "scaled_fp8_linear_test: SKIP fixture (no CUDA device)\n";
    return true;
  }

  const std::filesystem::path fixture_root =
      "/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/oracle/expert_layer3_prefix_input_block_cuda";
  if (!std::filesystem::exists(fixture_root)) {
    std::cout << "scaled_fp8_linear_test: SKIP fixture (missing oracle fixture)\n";
    return true;
  }

  const auto activations = read_f32_file(fixture_root / "expected_norm_output_fp32.bin");
  const auto expected_quantized_input =
      read_f32_file(fixture_root / "expected_fc1_latent_quantized_input_fp32.bin");
  const auto expected_weight_dequant =
      read_f32_file(fixture_root / "expected_fc1_latent_weight_dequant_fp32.bin");
  const auto expected_output =
      read_f32_file(fixture_root / "expected_fc1_latent_output_fp32.bin");
  const auto weight_fp8 =
      read_u8_file(fixture_root / "fc1_latent_weight_fp8.bin");
  const auto weight_scale =
      read_f32_file(fixture_root / "fc1_latent_weight_scale_fp32.bin");
  const auto input_scale =
      read_f32_file(fixture_root / "fc1_latent_input_scale_fp32.bin");
  if (!expect(!activations.empty(), "fixture activations should load") ||
      !expect(!expected_quantized_input.empty(), "fixture quantized activations should load") ||
      !expect(!expected_weight_dequant.empty(), "fixture dequantized weights should load") ||
      !expect(!expected_output.empty(), "fixture output should load") ||
      !expect(!weight_fp8.empty(), "fixture weight fp8 should load") ||
      !expect(weight_scale.size() == 1, "fixture weight scale should load") ||
      !expect(input_scale.size() == 1, "fixture input scale should load")) {
    return false;
  }

  constexpr std::size_t kRows = 4;
  constexpr std::size_t kInputCols = 4096;
  constexpr std::size_t kOutputRows = 1024;
  if (!expect(activations.size() == kRows * kInputCols, "fixture activation shape should match") ||
      !expect(expected_quantized_input.size() == activations.size(), "fixture quantized input shape should match") ||
      !expect(weight_fp8.size() == kOutputRows * kInputCols, "fixture weight shape should match") ||
      !expect(expected_weight_dequant.size() == weight_fp8.size(), "fixture weight dequant shape should match") ||
      !expect(expected_output.size() == kRows * kOutputRows, "fixture output shape should match")) {
    return false;
  }

  ScaledFp8LinearConfig config;
  config.output_rows = kOutputRows;
  config.input_cols = kInputCols;
  config.packed_weight_data = weight_fp8.data();
  config.packed_weight_nbytes = weight_fp8.size();
  config.weight_scale = weight_scale[0];
  config.input_scale = input_scale[0];
  auto op = ScaledFp8LinearOp::Create(config);
  if (!expect(op != nullptr && op->valid(), "fixture scaled-fp8 op should build")) {
    return false;
  }

  const auto host_weight_dequant = DequantizeScaledFp8WeightToHostFp32(config);
  if (!expect(host_weight_dequant.has_value(), "fixture weight dequant should build")) {
    return false;
  }
  const float weight_diff = max_abs_diff(*host_weight_dequant, expected_weight_dequant);

  auto input = DeviceTensorFp32::Create({kRows, kInputCols});
  auto quantized_input = DeviceTensorFp32::Create({kRows, kInputCols});
  auto output = DeviceTensorFp32::Create({kRows, kOutputRows});
  if (!expect(input != nullptr && input->valid(), "fixture input tensor should create") ||
      !expect(quantized_input != nullptr && quantized_input->valid(), "fixture quantized tensor should create") ||
      !expect(output != nullptr && output->valid(), "fixture output tensor should create") ||
      !expect(input->CopyFromHost(activations.data(), activations.size()), "fixture activations should upload")) {
    return false;
  }

  if (!expect(
          QuantizeFp32ToScaledFp8RoundTrip(*input, config.input_scale, quantized_input.get()),
          "fixture activations should quantize")) {
    return false;
  }
  std::vector<float> actual_quantized_input(expected_quantized_input.size(), 0.0f);
  if (!expect(
          quantized_input->CopyToHost(actual_quantized_input.data(), actual_quantized_input.size()),
          "fixture quantized activations should download")) {
    return false;
  }
  const float quantized_input_diff = max_abs_diff(actual_quantized_input, expected_quantized_input);

  GemmHeuristicCache heuristic_cache;
  if (!expect(op->Run(*handle, &heuristic_cache, *input, output.get()), "fixture scaled-fp8 op should execute")) {
    return false;
  }
  std::vector<float> actual_output(expected_output.size(), 0.0f);
  if (!expect(output->CopyToHost(actual_output.data(), actual_output.size()), "fixture output should download")) {
    return false;
  }
  const float output_diff = max_abs_diff(actual_output, expected_output);
  std::cout << "scaled_fp8_linear_test: fixture_quantized_input_diff=" << quantized_input_diff
            << " fixture_weight_dequant_diff=" << weight_diff
            << " fixture_output_diff=" << output_diff << "\n";

  return expect(weight_diff <= 1.0e-6f, "fixture weight dequant should match oracle") &&
         expect(quantized_input_diff <= 1.0e-6f, "fixture quantized input should match oracle") &&
         expect(output_diff <= 1.0e-3f, "fixture scaled-fp8 output should match oracle");
}

}  // namespace

int main() {
  return (test_scaled_fp8_linear_defaults_to_dequantized_dense_single_token() &&
          test_scaled_fp8_linear_disable_dequantized_dense_uses_reference_fallback() &&
          test_scaled_fp8_linear_matches_cpu_reference() &&
          test_scaled_fp8_linear_matches_oracle_fixture())
             ? 0
             : 1;
}
