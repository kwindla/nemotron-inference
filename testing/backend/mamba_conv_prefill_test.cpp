#include "nemotron/mamba_conv_prefill.h"
#include "nemotron/request_context.h"

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
using nemotron::MambaConvPrefillParams;
using nemotron::RequestExecutionConfig;
using nemotron::RequestExecutionContext;
using nemotron::RunMambaConvPrefill;

constexpr std::size_t kIntermediateSize = 4096;
constexpr std::size_t kStateSize = 128;
constexpr std::size_t kNGroups = 8;
constexpr std::size_t kNumHeads = 64;
constexpr std::size_t kConvKernelSize = 4;
constexpr std::size_t kConvDim =
    kIntermediateSize + (2 * kNGroups * kStateSize);
constexpr std::size_t kProjectionSize =
    kIntermediateSize + kConvDim + kNumHeads;
constexpr std::size_t kConvStateOffsetElems = 13;
constexpr std::size_t kConvStatePaddingElems = 29;

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

void CopyLayerState(
    const std::vector<float>& layer_state,
    std::vector<float>* full_state) {
  float* state_begin = full_state->data() + kConvStateOffsetElems;
  std::copy(layer_state.begin(), layer_state.end(), state_begin);
}

void RunCpuReference(
    const std::vector<float>& projected,
    std::size_t token_count,
    const std::vector<float>& conv1d_weight,
    const std::vector<float>& conv1d_bias,
    const std::vector<float>& initial_layer_state,
    std::vector<float>* conv_output,
    std::vector<float>* final_layer_state) {
  std::vector<float> state = initial_layer_state;
  conv_output->assign(token_count * kConvDim, 0.0f);
  for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
    const float* conv_input =
        projected.data() + (token_index * kProjectionSize) + kIntermediateSize;
    for (std::size_t channel = 0; channel < kConvDim; ++channel) {
      float* state_row = state.data() + (channel * kConvKernelSize);
      state_row[0] = state_row[1];
      state_row[1] = state_row[2];
      state_row[2] = state_row[3];
      state_row[3] = conv_input[channel];

      const float* weight_row =
          conv1d_weight.data() + (channel * kConvKernelSize);
      float accum = conv1d_bias[channel];
      accum = std::fma(state_row[0], weight_row[0], accum);
      accum = std::fma(state_row[1], weight_row[1], accum);
      accum = std::fma(state_row[2], weight_row[2], accum);
      accum = std::fma(state_row[3], weight_row[3], accum);
      (*conv_output)[(token_index * kConvDim) + channel] = SiLU(accum);
    }
  }
  *final_layer_state = std::move(state);
}

bool RunParityCase(std::size_t token_count, bool use_external_initial_state) {
  const std::size_t conv_state_elems = kConvDim * kConvKernelSize;
  const std::size_t total_state_elems =
      kConvStateOffsetElems + conv_state_elems + kConvStatePaddingElems;

  RequestExecutionConfig config;
  config.hidden_size = 1;
  config.max_tokens = token_count;
  config.scratch_tokens = token_count;
  config.mamba_conv_state_bytes_fp32 = total_state_elems * sizeof(float);

  auto request_context = RequestExecutionContext::Create(config);
  auto projected = DeviceTensorFp32::Create({token_count, kProjectionSize});
  auto conv_output = DeviceTensorFp32::Create({token_count, kConvDim});
  auto conv1d_weight = DeviceTensorFp32::Create({kConvDim, kConvKernelSize});
  auto conv1d_bias = DeviceTensorFp32::Create({kConvDim});
  if (!Expect(
          request_context && request_context->valid() &&
              projected && projected->valid() && conv_output &&
              conv_output->valid() && conv1d_weight && conv1d_weight->valid() &&
              conv1d_bias && conv1d_bias->valid(),
          "device allocations should succeed")) {
    return false;
  }

  std::vector<float> projected_host(token_count * kProjectionSize, 0.0f);
  std::vector<float> conv1d_weight_host(conv_state_elems, 0.0f);
  std::vector<float> conv1d_bias_host(kConvDim, 0.0f);
  std::vector<float> initial_layer_state(conv_state_elems, 0.0f);
  std::vector<float> request_state_host(total_state_elems, -0.75f);

  for (std::size_t i = 0; i < projected_host.size(); ++i) {
    projected_host[i] = Pattern(i, 37, 257, 0.8f);
  }
  for (std::size_t i = 0; i < conv1d_weight_host.size(); ++i) {
    conv1d_weight_host[i] = Pattern(i, 19, 97, 0.3f);
  }
  for (std::size_t i = 0; i < conv1d_bias_host.size(); ++i) {
    conv1d_bias_host[i] = Pattern(i, 11, 53, 0.06f);
  }
  for (std::size_t i = 0; i < initial_layer_state.size(); ++i) {
    initial_layer_state[i] = Pattern(i, 23, 89, 0.5f);
  }

  std::vector<float> request_initial_layer_state(conv_state_elems, 0.0f);
  for (std::size_t i = 0; i < request_initial_layer_state.size(); ++i) {
    request_initial_layer_state[i] = Pattern(i + 7, 29, 101, 0.4f);
  }
  CopyLayerState(request_initial_layer_state, &request_state_host);
  if (!Expect(
          request_context->mamba_conv_state()->CopyFromHost(
              request_state_host.data(),
              request_state_host.size()),
          "request conv state upload should succeed") ||
      !Expect(
          projected->CopyFromHost(projected_host.data(), projected_host.size()),
          "projected upload should succeed") ||
      !Expect(
          conv1d_weight->CopyFromHost(
              conv1d_weight_host.data(),
              conv1d_weight_host.size()),
          "conv1d weight upload should succeed") ||
      !Expect(
          conv1d_bias->CopyFromHost(
              conv1d_bias_host.data(),
              conv1d_bias_host.size()),
          "conv1d bias upload should succeed")) {
    return false;
  }

  std::unique_ptr<DeviceTensorFp32> external_initial_state;
  const DeviceTensorFp32* initial_state_arg = nullptr;
  if (use_external_initial_state) {
    std::vector<float> external_state_host(total_state_elems, -1.25f);
    CopyLayerState(initial_layer_state, &external_state_host);
    external_initial_state = DeviceTensorFp32::Create({total_state_elems});
    if (!Expect(
            external_initial_state && external_initial_state->valid() &&
                external_initial_state->CopyFromHost(
                    external_state_host.data(),
                    external_state_host.size()),
            "external initial conv state upload should succeed")) {
      return false;
    }
    initial_state_arg = external_initial_state.get();
  } else {
    initial_layer_state = request_initial_layer_state;
  }

  MambaConvPrefillParams params;
  params.intermediate_size = kIntermediateSize;
  params.state_size = kStateSize;
  params.n_groups = kNGroups;
  params.conv_kernel_size = kConvKernelSize;
  params.conv_state_elems = conv_state_elems;
  params.final_conv_state =
      request_context->mamba_conv_state()->data() + kConvStateOffsetElems;
  params.initial_conv_state =
      (initial_state_arg != nullptr ? initial_state_arg->data()
                                    : request_context->mamba_conv_state()->data()) +
      kConvStateOffsetElems;
  params.conv1d_weight = conv1d_weight->data();
  params.conv1d_bias = conv1d_bias->data();

  if (!Expect(
          RunMambaConvPrefill(params, *projected, conv_output.get()),
          "RunMambaConvPrefill should succeed") ||
      !Expect(
          cudaDeviceSynchronize() == cudaSuccess,
          "conv prefill kernel should complete")) {
    return false;
  }

  std::vector<float> conv_output_host(token_count * kConvDim, 0.0f);
  std::vector<float> final_request_state(total_state_elems, 0.0f);
  if (!Expect(
          conv_output->CopyToHost(conv_output_host.data(), conv_output_host.size()),
          "conv output download should succeed") ||
      !Expect(
          request_context->mamba_conv_state()->CopyToHost(
              final_request_state.data(),
              final_request_state.size()),
          "request conv state download should succeed")) {
    return false;
  }

  std::vector<float> expected_conv_output;
  std::vector<float> expected_final_layer_state;
  RunCpuReference(
      projected_host,
      token_count,
      conv1d_weight_host,
      conv1d_bias_host,
      initial_layer_state,
      &expected_conv_output,
      &expected_final_layer_state);

  std::vector<float> expected_request_state = request_state_host;
  CopyLayerState(expected_final_layer_state, &expected_request_state);

  const float output_diff = MaxAbsDiff(conv_output_host, expected_conv_output);
  const float state_diff = MaxAbsDiff(final_request_state, expected_request_state);
  if (output_diff > 3.0e-5f || state_diff > 1.0e-6f) {
    std::cerr << "mamba_conv_prefill parity mismatch: token_count=" << token_count
              << " external_initial=" << (use_external_initial_state ? 1 : 0)
              << " output_diff=" << output_diff
              << " state_diff=" << state_diff << "\n";
    return false;
  }

  return true;
}

}  // namespace

int main() {
  if (!HasCudaDevice()) {
    std::cout << "mamba_conv_prefill_test: SKIP (no CUDA device)\n";
    return 0;
  }

  if (!RunParityCase(32, false) || !RunParityCase(257, true)) {
    return 1;
  }

  std::cout << "mamba_conv_prefill_test: PASS\n";
  return 0;
}
