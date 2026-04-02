#include "nemotron/scaled_fp8_linear.h"

#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

using nemotron::CublasLtHandle;
using nemotron::DeviceTensorFp32;
using nemotron::GemmHeuristicCache;
using nemotron::ScaledFp8LinearConfig;
using nemotron::ScaledFp8LinearOp;

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
      double accum = 0.0;
      for (std::size_t col = 0; col < input_cols; ++col) {
        const float quantized_input = decode_fp8(encode_fp8(activations[row * input_cols + col] / input_scale)) * input_scale;
        const float weight = decode_fp8(weight_fp8[out * input_cols + col]) * weight_scale;
        accum += static_cast<double>(quantized_input) * static_cast<double>(weight);
      }
      output[row * output_rows + out] = static_cast<float>(accum);
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

}  // namespace

int main() {
  return test_scaled_fp8_linear_matches_cpu_reference() ? 0 : 1;
}
