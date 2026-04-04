#include "nemotron/mamba_ssd_prefill.h"

#include <cuda_runtime.h>

#include <cstddef>

#include "fused_decode_common.cuh"

namespace nemotron {
namespace {

constexpr unsigned kHiddenThreadsPerBlock = 64;
constexpr std::size_t kMaxSupportedStateSize = 128;

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

template <int kStateSize>
__global__ void MambaSsdPrefillFixedKernel(
    const float* __restrict__ projected,
    std::size_t projected_stride,
    const float* __restrict__ conv_output,
    std::size_t conv_output_stride,
    std::size_t token_count,
    std::size_t intermediate_size,
    std::size_t num_heads,
    std::size_t head_dim,
    std::size_t n_groups,
    float time_step_min,
    const float* __restrict__ A_log,
    const float* __restrict__ D,
    const float* __restrict__ dt_bias,
    const float* __restrict__ initial_ssm_state,
    float* __restrict__ final_ssm_state,
    float* __restrict__ ssm_output) {
  const std::size_t hidden_index =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (hidden_index >= intermediate_size) {
    return;
  }

  const std::size_t head = hidden_index / head_dim;
  const std::size_t group_width = num_heads / n_groups;
  const std::size_t group = head / group_width;
  const std::size_t grouped_state_span = n_groups * kStateSize;
  const std::size_t b_offset = intermediate_size + (group * kStateSize);
  const std::size_t c_offset = intermediate_size + grouped_state_span + (group * kStateSize);
  const std::size_t dt_offset = projected_stride - num_heads;

  float state[kStateSize];
  const float* initial_state_row = initial_ssm_state + (hidden_index * kStateSize);
#pragma unroll
  for (int state_index = 0; state_index < kStateSize; ++state_index) {
    state[state_index] = initial_state_row[state_index];
  }

  const float A = -expf(A_log[head]);
  const float D_value = D[head];
  const float dt_bias_value = dt_bias[head];

  for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
    const float* conv_row = conv_output + (token_index * conv_output_stride);
    const float hidden_value = conv_row[hidden_index];
    const float dt = fmaxf(
        fused_decode::Softplus(projected[(token_index * projected_stride) + dt_offset + head] +
                               dt_bias_value),
        time_step_min);
    const float decay = expf(dt * A);
    const float input_term = dt * hidden_value;
    const float* B_row = conv_row + b_offset;
    const float* C_row = conv_row + c_offset;

    float accum = 0.0f;
#pragma unroll
    for (int state_index = 0; state_index < kStateSize; ++state_index) {
      const float next =
          fmaf(input_term, B_row[state_index], state[state_index] * decay);
      state[state_index] = next;
      accum = fmaf(next, C_row[state_index], accum);
    }
    ssm_output[(token_index * intermediate_size) + hidden_index] =
        accum + (hidden_value * D_value);
  }

  float* final_state_row = final_ssm_state + (hidden_index * kStateSize);
#pragma unroll
  for (int state_index = 0; state_index < kStateSize; ++state_index) {
    final_state_row[state_index] = state[state_index];
  }
}

__global__ void MambaSsdPrefillGenericKernel(
    const float* __restrict__ projected,
    std::size_t projected_stride,
    const float* __restrict__ conv_output,
    std::size_t conv_output_stride,
    std::size_t token_count,
    std::size_t intermediate_size,
    std::size_t num_heads,
    std::size_t head_dim,
    std::size_t state_size,
    std::size_t n_groups,
    float time_step_min,
    const float* __restrict__ A_log,
    const float* __restrict__ D,
    const float* __restrict__ dt_bias,
    const float* __restrict__ initial_ssm_state,
    float* __restrict__ final_ssm_state,
    float* __restrict__ ssm_output) {
  const std::size_t hidden_index =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (hidden_index >= intermediate_size) {
    return;
  }

  const std::size_t head = hidden_index / head_dim;
  const std::size_t group_width = num_heads / n_groups;
  const std::size_t group = head / group_width;
  const std::size_t grouped_state_span = n_groups * state_size;
  const std::size_t b_offset = intermediate_size + (group * state_size);
  const std::size_t c_offset = intermediate_size + grouped_state_span + (group * state_size);
  const std::size_t dt_offset = projected_stride - num_heads;

  float state[kMaxSupportedStateSize];
  const float* initial_state_row = initial_ssm_state + (hidden_index * state_size);
  for (std::size_t state_index = 0; state_index < state_size; ++state_index) {
    state[state_index] = initial_state_row[state_index];
  }

  const float A = -expf(A_log[head]);
  const float D_value = D[head];
  const float dt_bias_value = dt_bias[head];

  for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
    const float* conv_row = conv_output + (token_index * conv_output_stride);
    const float hidden_value = conv_row[hidden_index];
    const float dt = fmaxf(
        fused_decode::Softplus(projected[(token_index * projected_stride) + dt_offset + head] +
                               dt_bias_value),
        time_step_min);
    const float decay = expf(dt * A);
    const float input_term = dt * hidden_value;
    const float* B_row = conv_row + b_offset;
    const float* C_row = conv_row + c_offset;

    float accum = 0.0f;
    for (std::size_t state_index = 0; state_index < state_size; ++state_index) {
      const float next =
          fmaf(input_term, B_row[state_index], state[state_index] * decay);
      state[state_index] = next;
      accum = fmaf(next, C_row[state_index], accum);
    }
    ssm_output[(token_index * intermediate_size) + hidden_index] =
        accum + (hidden_value * D_value);
  }

  float* final_state_row = final_ssm_state + (hidden_index * state_size);
  for (std::size_t state_index = 0; state_index < state_size; ++state_index) {
    final_state_row[state_index] = state[state_index];
  }
}

}  // namespace

bool RunMambaSsdPrefill(
    const MambaSsdPrefillParams& params,
    const DeviceTensorFp32& projected,
    const DeviceTensorFp32& conv_output,
    DeviceTensorFp32* ssm_output) {
  if (!projected.valid() ||
      projected.shape().size() != 2 ||
      projected.shape()[0] == 0 ||
      !conv_output.valid() ||
      conv_output.shape().size() != 2 ||
      ssm_output == nullptr ||
      !ssm_output->valid() ||
      ssm_output->shape().size() != 2 ||
      params.intermediate_size == 0 ||
      params.num_heads == 0 ||
      params.head_dim == 0 ||
      params.state_size == 0 ||
      params.state_size > kMaxSupportedStateSize ||
      params.n_groups == 0 ||
      params.ssm_state_elems == 0 ||
      params.final_ssm_state == nullptr ||
      params.initial_ssm_state == nullptr ||
      params.A_log == nullptr ||
      params.D == nullptr ||
      params.dt_bias == nullptr) {
    return false;
  }

  if (params.intermediate_size != params.num_heads * params.head_dim ||
      (params.num_heads % params.n_groups) != 0) {
    return false;
  }

  const std::size_t token_count = projected.shape()[0];
  const std::size_t conv_dim =
      params.intermediate_size + (2 * params.n_groups * params.state_size);
  const std::size_t ssm_state_elems = params.intermediate_size * params.state_size;
  if (projected.shape()[1] < params.num_heads ||
      conv_output.shape()[0] != token_count ||
      conv_output.shape()[1] != conv_dim ||
      ssm_output->shape()[0] != token_count ||
      ssm_output->shape()[1] != params.intermediate_size ||
      params.ssm_state_elems != ssm_state_elems) {
    return false;
  }

  const dim3 block(kHiddenThreadsPerBlock);
  const dim3 grid(static_cast<unsigned>((params.intermediate_size + block.x - 1) / block.x));
  switch (params.state_size) {
    case 128:
      MambaSsdPrefillFixedKernel<128><<<grid, block>>>(
          projected.data(),
          projected.shape()[1],
          conv_output.data(),
          conv_output.shape()[1],
          token_count,
          params.intermediate_size,
          params.num_heads,
          params.head_dim,
          params.n_groups,
          params.time_step_min,
          params.A_log,
          params.D,
          params.dt_bias,
          params.initial_ssm_state,
          params.final_ssm_state,
          ssm_output->data());
      break;
    default:
      MambaSsdPrefillGenericKernel<<<grid, block>>>(
          projected.data(),
          projected.shape()[1],
          conv_output.data(),
          conv_output.shape()[1],
          token_count,
          params.intermediate_size,
          params.num_heads,
          params.head_dim,
          params.state_size,
          params.n_groups,
          params.time_step_min,
          params.A_log,
          params.D,
          params.dt_bias,
          params.initial_ssm_state,
          params.final_ssm_state,
          ssm_output->data());
      break;
  }

  return CheckCuda(cudaGetLastError());
}

}  // namespace nemotron
