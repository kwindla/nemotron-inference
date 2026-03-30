#include "nemotron/device_tensor.h"
#include "nemotron/primitive_ops.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

bool expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool nearly_equal(const std::vector<float>& lhs, const std::vector<float>& rhs, float tol) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    const float diff = lhs[i] - rhs[i];
    if (diff > tol || diff < -tol) {
      return false;
    }
  }
  return true;
}

std::vector<float> cpu_rms_norm(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    std::size_t rows,
    std::size_t hidden_size,
    float epsilon) {
  std::vector<float> output(input.size(), 0.0f);
  for (std::size_t row = 0; row < rows; ++row) {
    float sumsq = 0.0f;
    for (std::size_t col = 0; col < hidden_size; ++col) {
      const float value = input[row * hidden_size + col];
      sumsq += value * value;
    }
    const float inv_rms = 1.0f / std::sqrt((sumsq / static_cast<float>(hidden_size)) + epsilon);
    for (std::size_t col = 0; col < hidden_size; ++col) {
      output[row * hidden_size + col] =
          input[row * hidden_size + col] * inv_rms * weight[col];
    }
  }
  return output;
}

bool test_residual_add_matches_cpu_reference() {
  auto lhs = nemotron::DeviceTensorFp32::Create({2, 3});
  auto rhs = nemotron::DeviceTensorFp32::Create({2, 3});
  auto output = nemotron::DeviceTensorFp32::Create({2, 3});
  if (!lhs || !rhs || !output) {
    std::cout << "primitive_ops_test: SKIP (no CUDA device available)\n";
    return true;
  }

  const std::vector<float> lhs_host = {1.0f, -2.0f, 3.5f, 0.5f, 4.0f, -1.0f};
  const std::vector<float> rhs_host = {0.25f, 1.0f, -0.5f, 2.0f, -3.0f, 5.0f};
  const std::vector<float> expected = {1.25f, -1.0f, 3.0f, 2.5f, 1.0f, 4.0f};
  if (!expect(lhs->CopyFromHost(lhs_host.data(), lhs_host.size()), "lhs upload should succeed") ||
      !expect(rhs->CopyFromHost(rhs_host.data(), rhs_host.size()), "rhs upload should succeed") ||
      !expect(nemotron::ResidualAddFp32(*lhs, *rhs, output.get()), "residual add kernel should succeed")) {
    return false;
  }

  std::vector<float> actual(expected.size(), 0.0f);
  if (!expect(output->CopyToHost(actual.data(), actual.size()), "result download should succeed")) {
    return false;
  }
  return expect(nearly_equal(actual, expected, 1e-6f), "residual add should match CPU reference");
}

bool test_rms_norm_matches_cpu_reference() {
  auto input = nemotron::DeviceTensorFp32::Create({2, 4});
  auto weight = nemotron::DeviceTensorFp32::Create({4});
  auto output = nemotron::DeviceTensorFp32::Create({2, 4});
  if (!input || !weight || !output) {
    std::cout << "primitive_ops_test: SKIP (no CUDA device available)\n";
    return true;
  }

  const std::vector<float> input_host = {
      1.0f, -2.0f, 0.5f, 4.0f,
      -1.5f, 3.0f, 2.0f, -0.25f,
  };
  const std::vector<float> weight_host = {1.0f, 0.5f, 1.5f, 2.0f};
  constexpr float kEpsilon = 1e-5f;
  const std::vector<float> expected = cpu_rms_norm(input_host, weight_host, 2, 4, kEpsilon);

  if (!expect(input->CopyFromHost(input_host.data(), input_host.size()), "input upload should succeed") ||
      !expect(weight->CopyFromHost(weight_host.data(), weight_host.size()), "weight upload should succeed") ||
      !expect(nemotron::RmsNormFp32(*input, *weight, kEpsilon, output.get()), "rmsnorm kernel should succeed")) {
    return false;
  }

  std::vector<float> actual(expected.size(), 0.0f);
  if (!expect(output->CopyToHost(actual.data(), actual.size()), "output download should succeed")) {
    return false;
  }
  return expect(nearly_equal(actual, expected, 1e-4f), "rmsnorm should match CPU reference");
}

}  // namespace

int main() {
  if (!test_residual_add_matches_cpu_reference() ||
      !test_rms_norm_matches_cpu_reference()) {
    return 1;
  }
  std::cout << "primitive_ops_test: PASS\n";
  return 0;
}
