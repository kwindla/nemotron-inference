#include "nemotron/mamba_gated_group_norm.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using nemotron::DeviceTensorFp32;
using nemotron::MambaGatedGroupNormParams;
using nemotron::RunMambaGatedGroupNorm;

constexpr std::size_t kIntermediateSize = 4096;
constexpr std::size_t kStateSize = 128;
constexpr std::size_t kNGroups = 8;
constexpr std::size_t kNumHeads = 64;
constexpr std::size_t kConvDim =
    kIntermediateSize + (2 * kNGroups * kStateSize);
constexpr std::size_t kProjectionSize =
    kIntermediateSize + kConvDim + kNumHeads;
constexpr float kMixerRmsEpsilon = 1.0e-5f;

bool HasCudaDevice() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

bool Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

float Pattern(
    std::size_t index,
    std::size_t multiplier,
    std::size_t modulus,
    float scale) {
  const std::size_t raw = ((index * multiplier) + 17) % modulus;
  return (static_cast<float>(raw) / static_cast<float>(modulus)) * scale -
         (scale * 0.5f);
}

float Sigmoid(float value) {
  if (value >= 0.0f) {
    const float exp_neg = std::exp(-value);
    return 1.0f / (1.0f + exp_neg);
  }
  const float exp_pos = std::exp(value);
  return exp_pos / (1.0f + exp_pos);
}

float SiLU(float value) {
  return value * Sigmoid(value);
}

float MaxAbsDiff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return INFINITY;
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
  }
  return max_diff;
}

void RunCpuReference(
    const std::vector<float>& projected,
    std::size_t token_count,
    const std::vector<float>& mixer_norm_weight,
    std::vector<float>* ssm_output) {
  const std::size_t mixer_group_size = kIntermediateSize / kNGroups;
  for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
    const float* gate = projected.data() + (token_index * kProjectionSize);
    float* output_row = ssm_output->data() + (token_index * kIntermediateSize);
    for (std::size_t group = 0; group < kNGroups; ++group) {
      const std::size_t begin = group * mixer_group_size;
      const std::size_t end = begin + mixer_group_size;
      float variance = 0.0f;
      for (std::size_t hidden_index = begin; hidden_index < end; ++hidden_index) {
        const float gated = output_row[hidden_index] * SiLU(gate[hidden_index]);
        output_row[hidden_index] = gated;
        variance += gated * gated;
      }
      variance /= static_cast<float>(mixer_group_size);
      const float rstd = 1.0f / std::sqrt(variance + kMixerRmsEpsilon);
      for (std::size_t hidden_index = begin; hidden_index < end; ++hidden_index) {
        output_row[hidden_index] =
            output_row[hidden_index] * rstd * mixer_norm_weight[hidden_index];
      }
    }
  }
}

bool RunParityCase(std::size_t token_count) {
  auto projected = DeviceTensorFp32::Create({token_count, kProjectionSize});
  auto ssm_output = DeviceTensorFp32::Create({token_count, kIntermediateSize});
  auto mixer_norm_weight = DeviceTensorFp32::Create({kIntermediateSize});
  if (!Expect(
          projected && projected->valid() &&
              ssm_output && ssm_output->valid() &&
              mixer_norm_weight && mixer_norm_weight->valid(),
          "device allocations should succeed")) {
    return false;
  }

  std::vector<float> projected_host(token_count * kProjectionSize, 0.0f);
  std::vector<float> ssm_output_host(token_count * kIntermediateSize, 0.0f);
  std::vector<float> mixer_norm_weight_host(kIntermediateSize, 0.0f);
  for (std::size_t i = 0; i < projected_host.size(); ++i) {
    projected_host[i] = Pattern(i, 37, 257, 0.8f);
  }
  for (std::size_t i = 0; i < ssm_output_host.size(); ++i) {
    ssm_output_host[i] = Pattern(i, 23, 211, 0.7f);
  }
  for (std::size_t i = 0; i < mixer_norm_weight_host.size(); ++i) {
    mixer_norm_weight_host[i] = 0.75f + Pattern(i, 11, 101, 0.08f);
  }

  if (!Expect(
          projected->CopyFromHost(projected_host.data(), projected_host.size()),
          "projected upload should succeed") ||
      !Expect(
          ssm_output->CopyFromHost(ssm_output_host.data(), ssm_output_host.size()),
          "ssm output upload should succeed") ||
      !Expect(
          mixer_norm_weight->CopyFromHost(
              mixer_norm_weight_host.data(),
              mixer_norm_weight_host.size()),
          "mixer norm weight upload should succeed")) {
    return false;
  }

  MambaGatedGroupNormParams params;
  params.intermediate_size = kIntermediateSize;
  params.n_groups = kNGroups;
  params.mixer_rms_epsilon = kMixerRmsEpsilon;
  params.mixer_norm_weight = mixer_norm_weight->data();
  if (!Expect(
          RunMambaGatedGroupNorm(params, *projected, ssm_output.get()),
          "RunMambaGatedGroupNorm should succeed") ||
      !Expect(
          cudaDeviceSynchronize() == cudaSuccess,
          "gated group norm kernel should complete")) {
    return false;
  }

  std::vector<float> actual_output(token_count * kIntermediateSize, 0.0f);
  if (!Expect(
          ssm_output->CopyToHost(actual_output.data(), actual_output.size()),
          "gated group norm output download should succeed")) {
    return false;
  }

  std::vector<float> expected_output = ssm_output_host;
  RunCpuReference(projected_host, token_count, mixer_norm_weight_host, &expected_output);

  const float output_diff = MaxAbsDiff(actual_output, expected_output);
  if (!Expect(output_diff <= 1.0e-5f, "gated group norm output should match CPU reference")) {
    std::cerr << "output_diff=" << output_diff << "\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!HasCudaDevice()) {
    std::cout << "mamba_gated_group_norm_test: SKIP (no CUDA device)\n";
    return 0;
  }

  if (!RunParityCase(1) || !RunParityCase(17)) {
    return 1;
  }

  std::cout << "mamba_gated_group_norm_test: PASS\n";
  return 0;
}
