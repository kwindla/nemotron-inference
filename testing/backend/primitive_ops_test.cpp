#include "nemotron/device_tensor.h"
#include "nemotron/primitive_ops.h"

#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

bool expect(bool condition, const std::string& message) {
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

bool nearly_equal_relative(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    float rel_tol,
    float abs_tol) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    const float diff = std::fabs(lhs[i] - rhs[i]);
    const float limit = abs_tol + rel_tol * std::fabs(rhs[i]);
    if (diff > limit) {
      return false;
    }
  }
  return true;
}

__nv_bfloat16 to_bf16(float value) {
  return __float2bfloat16(value);
}

float from_bf16(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

std::vector<__nv_bfloat16> to_bf16_vector(const std::vector<float>& values) {
  std::vector<__nv_bfloat16> converted(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    converted[i] = to_bf16(values[i]);
  }
  return converted;
}

std::vector<float> from_bf16_vector(const std::vector<__nv_bfloat16>& values) {
  std::vector<float> converted(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    converted[i] = from_bf16(values[i]);
  }
  return converted;
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

struct FusedAddRmsNormCase {
  const char* name = "";
  std::size_t rows = 0;
  std::size_t hidden_size = 0;
  float epsilon = 0.0f;
  float hidden_scale = 1.0f;
  float residual_scale = 1.0f;
  float weight_scale = 1.0f;
  bool include_extremes = false;
};

void fill_fused_add_rms_norm_case(
    const FusedAddRmsNormCase& test_case,
    int seed,
    std::vector<float>* hidden,
    std::vector<float>* residual,
    std::vector<float>* weight) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> activation_dist(-1.0f, 1.0f);
  std::uniform_real_distribution<float> weight_dist(-1.5f, 1.5f);

  hidden->resize(test_case.rows * test_case.hidden_size);
  residual->resize(test_case.rows * test_case.hidden_size);
  weight->resize(test_case.hidden_size);
  for (float& value : *hidden) {
    value = activation_dist(rng) * test_case.hidden_scale;
  }
  for (float& value : *residual) {
    value = activation_dist(rng) * test_case.residual_scale;
  }
  for (float& value : *weight) {
    value = weight_dist(rng) * test_case.weight_scale;
  }

  if (test_case.include_extremes && !hidden->empty()) {
    (*hidden)[0] = 1.0e-3f;
    (*residual)[0] = -9.6e2f;
    (*weight)[0] = 1.75f;

    if (hidden->size() > 1) {
      (*hidden)[1] = -8.2e2f;
      (*residual)[1] = 2.5e-3f;
    }
    if (hidden->size() > 2) {
      (*hidden)[2] = 7.5e2f;
      (*residual)[2] = -7.45e2f;
    }
    if (weight->size() > 1) {
      (*weight)[1] = -0.125f;
    }
    if (weight->size() > 2) {
      (*weight)[2] = 3.0f;
    }
  }
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

bool test_fused_add_rms_norm_bf16_matches_fp32_reference() {
  const std::vector<FusedAddRmsNormCase> cases = {
      {"basic", 3, 64, 1e-5f, 1.25f, 0.75f, 1.5f, false},
      {"non_warp_multiple", 4, 37, 1e-5f, 2.0f, 1.5f, 0.8f, false},
      {"near_zero_epsilon", 2, 19, 1e-12f, 0.35f, 0.25f, 1.0f, false},
      {"large_small_and_looping", 2, 513, 1e-6f, 4.0f, 3.5f, 1.25f, true},
  };

  auto hidden_bf16 = nemotron::DeviceTensorBf16::Create({cases[0].rows, cases[0].hidden_size});
  auto residual_bf16 = nemotron::DeviceTensorBf16::Create({cases[0].rows, cases[0].hidden_size});
  auto output_bf16 = nemotron::DeviceTensorBf16::Create({cases[0].rows, cases[0].hidden_size});
  auto hidden_fp32 = nemotron::DeviceTensorFp32::Create({cases[0].rows, cases[0].hidden_size});
  auto residual_fp32 = nemotron::DeviceTensorFp32::Create({cases[0].rows, cases[0].hidden_size});
  auto residual_added_fp32 = nemotron::DeviceTensorFp32::Create({cases[0].rows, cases[0].hidden_size});
  auto output_fp32 = nemotron::DeviceTensorFp32::Create({cases[0].rows, cases[0].hidden_size});
  auto weight_fp32 = nemotron::DeviceTensorFp32::Create({cases[0].hidden_size});
  if (!hidden_bf16 || !residual_bf16 || !output_bf16 || !hidden_fp32 || !residual_fp32 ||
      !residual_added_fp32 || !output_fp32 || !weight_fp32) {
    std::cout << "primitive_ops_test: SKIP (no CUDA device available)\n";
    return true;
  }

  constexpr float kRelTol = 1.0e-2f;
  constexpr float kAbsTol = 5.0e-3f;

  for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
    const FusedAddRmsNormCase& test_case = cases[case_index];

    hidden_bf16 = nemotron::DeviceTensorBf16::Create({test_case.rows, test_case.hidden_size});
    residual_bf16 = nemotron::DeviceTensorBf16::Create({test_case.rows, test_case.hidden_size});
    output_bf16 = nemotron::DeviceTensorBf16::Create({test_case.rows, test_case.hidden_size});
    hidden_fp32 = nemotron::DeviceTensorFp32::Create({test_case.rows, test_case.hidden_size});
    residual_fp32 = nemotron::DeviceTensorFp32::Create({test_case.rows, test_case.hidden_size});
    residual_added_fp32 = nemotron::DeviceTensorFp32::Create({test_case.rows, test_case.hidden_size});
    output_fp32 = nemotron::DeviceTensorFp32::Create({test_case.rows, test_case.hidden_size});
    weight_fp32 = nemotron::DeviceTensorFp32::Create({test_case.hidden_size});
    if (!expect(
            hidden_bf16 && residual_bf16 && output_bf16 && hidden_fp32 && residual_fp32 &&
                residual_added_fp32 && output_fp32 && weight_fp32,
            std::string("case ") + test_case.name + ": device tensor allocation should succeed")) {
      return false;
    }

    std::vector<float> hidden_host;
    std::vector<float> residual_host;
    std::vector<float> weight_host;
    fill_fused_add_rms_norm_case(
        test_case,
        static_cast<int>(1234 + case_index * 97),
        &hidden_host,
        &residual_host,
        &weight_host);

    const std::vector<__nv_bfloat16> hidden_host_bf16 = to_bf16_vector(hidden_host);
    const std::vector<__nv_bfloat16> residual_host_bf16 = to_bf16_vector(residual_host);
    const std::vector<float> hidden_host_fp32 = from_bf16_vector(hidden_host_bf16);
    const std::vector<float> residual_host_fp32 = from_bf16_vector(residual_host_bf16);

    if (!expect(
            hidden_bf16->CopyFromHost(hidden_host_bf16.data(), hidden_host_bf16.size()),
            std::string("case ") + test_case.name + ": BF16 hidden upload should succeed") ||
        !expect(
            residual_bf16->CopyFromHost(residual_host_bf16.data(), residual_host_bf16.size()),
            std::string("case ") + test_case.name + ": BF16 residual upload should succeed") ||
        !expect(
            hidden_fp32->CopyFromHost(hidden_host_fp32.data(), hidden_host_fp32.size()),
            std::string("case ") + test_case.name + ": FP32 hidden upload should succeed") ||
        !expect(
            residual_fp32->CopyFromHost(residual_host_fp32.data(), residual_host_fp32.size()),
            std::string("case ") + test_case.name + ": FP32 residual upload should succeed") ||
        !expect(
            weight_fp32->CopyFromHost(weight_host.data(), weight_host.size()),
            std::string("case ") + test_case.name + ": weight upload should succeed")) {
      return false;
    }

    if (!expect(
            nemotron::ResidualAddFp32(*hidden_fp32, *residual_fp32, residual_added_fp32.get()),
            std::string("case ") + test_case.name + ": FP32 residual add should succeed") ||
        !expect(
            nemotron::RmsNormFp32(*residual_added_fp32, *weight_fp32, test_case.epsilon, output_fp32.get()),
            std::string("case ") + test_case.name + ": FP32 RMSNorm should succeed") ||
        !expect(
            nemotron::FusedAddRmsNormBf16(
                *hidden_bf16,
                residual_bf16.get(),
                *weight_fp32,
                test_case.epsilon,
                output_bf16.get()),
            std::string("case ") + test_case.name + ": fused BF16 add+RMSNorm should succeed")) {
      return false;
    }

    std::vector<float> expected_residual(hidden_host_fp32.size(), 0.0f);
    std::vector<float> expected_output(hidden_host_fp32.size(), 0.0f);
    if (!expect(
            residual_added_fp32->CopyToHost(expected_residual.data(), expected_residual.size()),
            std::string("case ") + test_case.name + ": expected residual download should succeed") ||
        !expect(
            output_fp32->CopyToHost(expected_output.data(), expected_output.size()),
            std::string("case ") + test_case.name + ": expected output download should succeed")) {
      return false;
    }

    std::vector<__nv_bfloat16> actual_residual_bf16(expected_residual.size());
    std::vector<__nv_bfloat16> actual_output_bf16(expected_output.size());
    if (!expect(
            residual_bf16->CopyToHost(actual_residual_bf16.data(), actual_residual_bf16.size()),
            std::string("case ") + test_case.name + ": fused residual download should succeed") ||
        !expect(
            output_bf16->CopyToHost(actual_output_bf16.data(), actual_output_bf16.size()),
            std::string("case ") + test_case.name + ": fused output download should succeed")) {
      return false;
    }

    const std::vector<float> actual_residual = from_bf16_vector(actual_residual_bf16);
    const std::vector<float> actual_output = from_bf16_vector(actual_output_bf16);
    if (!expect(
            nearly_equal_relative(actual_residual, expected_residual, kRelTol, kAbsTol),
            std::string("case ") + test_case.name +
                ": fused residual should match FP32 reference within BF16 tolerance") ||
        !expect(
            nearly_equal_relative(actual_output, expected_output, kRelTol, kAbsTol),
            std::string("case ") + test_case.name +
                ": fused output should match FP32 reference within BF16 tolerance")) {
      return false;
    }
  }

  return true;
}

}  // namespace

int main() {
  if (!test_residual_add_matches_cpu_reference() ||
      !test_rms_norm_matches_cpu_reference() ||
      !test_fused_add_rms_norm_bf16_matches_fp32_reference()) {
    return 1;
  }
  std::cout << "primitive_ops_test: PASS\n";
  return 0;
}
