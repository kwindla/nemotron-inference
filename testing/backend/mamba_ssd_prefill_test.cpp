#include "nemotron/mamba_ssd_prefill.h"
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
using nemotron::MambaSsdPrefillParams;
using nemotron::RequestExecutionConfig;
using nemotron::RequestExecutionContext;
using nemotron::RunMambaSsdPrefill;

constexpr std::size_t kIntermediateSize = 4096;
constexpr std::size_t kStateSize = 128;
constexpr std::size_t kNGroups = 8;
constexpr std::size_t kNumHeads = 64;
constexpr std::size_t kHeadDim = 64;
constexpr std::size_t kConvDim =
    kIntermediateSize + (2 * kNGroups * kStateSize);
constexpr std::size_t kProjectionSize =
    kIntermediateSize + kConvDim + kNumHeads;
constexpr std::size_t kSsmStateOffsetElems = 19;
constexpr std::size_t kSsmStatePaddingElems = 37;
constexpr float kTimeStepMin = 1.0e-3f;

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

float Softplus(float value) {
  if (value > 20.0f) {
    return value;
  }
  if (value < -20.0f) {
    return std::exp(value);
  }
  return std::log1p(std::exp(value));
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
  float* state_begin = full_state->data() + kSsmStateOffsetElems;
  std::copy(layer_state.begin(), layer_state.end(), state_begin);
}

void RunCpuReference(
    const std::vector<float>& projected,
    const std::vector<float>& conv_output,
    std::size_t token_count,
    const std::vector<float>& A_log,
    const std::vector<float>& D,
    const std::vector<float>& dt_bias,
    const std::vector<float>& initial_layer_state,
    std::vector<float>* ssm_output,
    std::vector<float>* final_layer_state) {
  std::vector<float> state = initial_layer_state;
  ssm_output->assign(token_count * kIntermediateSize, 0.0f);

  const std::size_t group_width = kNumHeads / kNGroups;
  const std::size_t grouped_state_span = kNGroups * kStateSize;
  const std::size_t dt_offset = kProjectionSize - kNumHeads;
  for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
    const float* projected_row = projected.data() + (token_index * kProjectionSize);
    const float* conv_row = conv_output.data() + (token_index * kConvDim);
    float* output_row = ssm_output->data() + (token_index * kIntermediateSize);
    for (std::size_t head = 0; head < kNumHeads; ++head) {
      const std::size_t group = head / group_width;
      const float A = -std::exp(A_log[head]);
      const float D_value = D[head];
      const float dt = std::max(
          Softplus(projected_row[dt_offset + head] + dt_bias[head]),
          kTimeStepMin);
      const float decay = std::exp(dt * A);
      const float* B_row = conv_row + kIntermediateSize + (group * kStateSize);
      const float* C_row =
          conv_row + kIntermediateSize + grouped_state_span + (group * kStateSize);
      for (std::size_t dim = 0; dim < kHeadDim; ++dim) {
        const std::size_t hidden_index = (head * kHeadDim) + dim;
        const float hidden_value = conv_row[hidden_index];
        const float input_term = dt * hidden_value;
        float accum = 0.0f;
        float* state_row = state.data() + (hidden_index * kStateSize);
        for (std::size_t state_index = 0; state_index < kStateSize; ++state_index) {
          const float next =
              std::fma(input_term, B_row[state_index], state_row[state_index] * decay);
          state_row[state_index] = next;
          accum = std::fma(next, C_row[state_index], accum);
        }
        output_row[hidden_index] = accum + (hidden_value * D_value);
      }
    }
  }

  *final_layer_state = std::move(state);
}

bool RunParityCase(std::size_t token_count, bool use_external_initial_state) {
  const std::size_t ssm_state_elems = kIntermediateSize * kStateSize;
  const std::size_t total_state_elems =
      kSsmStateOffsetElems + ssm_state_elems + kSsmStatePaddingElems;

  RequestExecutionConfig config;
  config.hidden_size = 1;
  config.max_tokens = token_count;
  config.scratch_tokens = token_count;
  config.mamba_state_bytes_fp32 = total_state_elems * sizeof(float);

  auto request_context = RequestExecutionContext::Create(config);
  auto projected = DeviceTensorFp32::Create({token_count, kProjectionSize});
  auto conv_output = DeviceTensorFp32::Create({token_count, kConvDim});
  auto ssm_output = DeviceTensorFp32::Create({token_count, kIntermediateSize});
  auto A_log = DeviceTensorFp32::Create({kNumHeads});
  auto D = DeviceTensorFp32::Create({kNumHeads});
  auto dt_bias = DeviceTensorFp32::Create({kNumHeads});
  if (!Expect(
          request_context && request_context->valid() &&
              projected && projected->valid() &&
              conv_output && conv_output->valid() &&
              ssm_output && ssm_output->valid() &&
              A_log && A_log->valid() &&
              D && D->valid() &&
              dt_bias && dt_bias->valid(),
          "device allocations should succeed")) {
    return false;
  }

  std::vector<float> projected_host(token_count * kProjectionSize, 0.0f);
  std::vector<float> conv_output_host(token_count * kConvDim, 0.0f);
  std::vector<float> A_log_host(kNumHeads, 0.0f);
  std::vector<float> D_host(kNumHeads, 0.0f);
  std::vector<float> dt_bias_host(kNumHeads, 0.0f);
  std::vector<float> initial_layer_state(ssm_state_elems, 0.0f);
  std::vector<float> request_state_host(total_state_elems, -0.75f);

  for (std::size_t i = 0; i < projected_host.size(); ++i) {
    projected_host[i] = Pattern(i, 37, 257, 0.8f);
  }
  for (std::size_t i = 0; i < conv_output_host.size(); ++i) {
    conv_output_host[i] = Pattern(i, 19, 211, 0.7f);
  }
  for (std::size_t i = 0; i < A_log_host.size(); ++i) {
    A_log_host[i] = Pattern(i, 5, 43, 0.5f);
    D_host[i] = Pattern(i, 7, 59, 0.25f);
    dt_bias_host[i] = Pattern(i, 11, 67, 0.3f);
  }
  for (std::size_t i = 0; i < initial_layer_state.size(); ++i) {
    initial_layer_state[i] = Pattern(i, 23, 89, 0.4f);
  }

  std::vector<float> request_initial_layer_state(ssm_state_elems, 0.0f);
  for (std::size_t i = 0; i < request_initial_layer_state.size(); ++i) {
    request_initial_layer_state[i] = Pattern(i + 7, 29, 101, 0.35f);
  }
  CopyLayerState(request_initial_layer_state, &request_state_host);
  if (!Expect(
          request_context->mamba_state()->CopyFromHost(
              request_state_host.data(),
              request_state_host.size()),
          "request ssm state upload should succeed") ||
      !Expect(
          projected->CopyFromHost(projected_host.data(), projected_host.size()),
          "projected upload should succeed") ||
      !Expect(
          conv_output->CopyFromHost(conv_output_host.data(), conv_output_host.size()),
          "conv output upload should succeed") ||
      !Expect(
          A_log->CopyFromHost(A_log_host.data(), A_log_host.size()),
          "A_log upload should succeed") ||
      !Expect(
          D->CopyFromHost(D_host.data(), D_host.size()),
          "D upload should succeed") ||
      !Expect(
          dt_bias->CopyFromHost(dt_bias_host.data(), dt_bias_host.size()),
          "dt_bias upload should succeed")) {
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
            "external initial ssm state upload should succeed")) {
      return false;
    }
    initial_state_arg = external_initial_state.get();
  } else {
    initial_layer_state = request_initial_layer_state;
  }

  MambaSsdPrefillParams params;
  params.intermediate_size = kIntermediateSize;
  params.num_heads = kNumHeads;
  params.head_dim = kHeadDim;
  params.state_size = kStateSize;
  params.n_groups = kNGroups;
  params.ssm_state_elems = ssm_state_elems;
  params.final_ssm_state =
      request_context->mamba_state()->data() + kSsmStateOffsetElems;
  params.initial_ssm_state =
      (initial_state_arg != nullptr ? initial_state_arg->data()
                                    : request_context->mamba_state()->data()) +
      kSsmStateOffsetElems;
  params.time_step_min = kTimeStepMin;
  params.A_log = A_log->data();
  params.D = D->data();
  params.dt_bias = dt_bias->data();

  if (!Expect(
          RunMambaSsdPrefill(params, *projected, *conv_output, ssm_output.get()),
          "RunMambaSsdPrefill should succeed") ||
      !Expect(
          cudaDeviceSynchronize() == cudaSuccess,
          "ssd prefill kernel should complete")) {
    return false;
  }

  std::vector<float> ssm_output_host(token_count * kIntermediateSize, 0.0f);
  std::vector<float> final_request_state(total_state_elems, 0.0f);
  if (!Expect(
          ssm_output->CopyToHost(ssm_output_host.data(), ssm_output_host.size()),
          "ssd output download should succeed") ||
      !Expect(
          request_context->mamba_state()->CopyToHost(
              final_request_state.data(),
              final_request_state.size()),
          "request ssm state download should succeed")) {
    return false;
  }

  std::vector<float> expected_ssm_output;
  std::vector<float> expected_final_layer_state;
  RunCpuReference(
      projected_host,
      conv_output_host,
      token_count,
      A_log_host,
      D_host,
      dt_bias_host,
      initial_layer_state,
      &expected_ssm_output,
      &expected_final_layer_state);

  std::vector<float> expected_request_state = request_state_host;
  CopyLayerState(expected_final_layer_state, &expected_request_state);

  const float output_diff = MaxAbsDiff(ssm_output_host, expected_ssm_output);
  const float state_diff = MaxAbsDiff(final_request_state, expected_request_state);
  if (output_diff > 2.0e-4f || state_diff > 2.0e-5f) {
    std::cerr << "mamba_ssd_prefill parity mismatch: token_count=" << token_count
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
    std::cout << "mamba_ssd_prefill_test: SKIP (no CUDA device)\n";
    return 0;
  }

  if (!RunParityCase(32, false) || !RunParityCase(257, true)) {
    return 1;
  }

  std::cout << "mamba_ssd_prefill_test: PASS\n";
  return 0;
}
