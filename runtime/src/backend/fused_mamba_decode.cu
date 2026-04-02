#include "nemotron/fused_mamba_decode.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "fused_decode_common.cuh"

namespace nemotron {
namespace {

constexpr int kMaxMambaGroups = 32;

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

bool EnvEnabled(const char* env_var) {
  const char* value = std::getenv(env_var);
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

__global__ void FusedMambaDecodeKernel(
    FusedMambaLayerParams params,
    const float* projected,
    float* conv_state,
    float* ssm_state,
    float* scan_output) {
  extern __shared__ float conv_output[];
  __shared__ float group_sums[kMaxMambaGroups];

  const std::size_t tid = threadIdx.x;
  const std::size_t conv_dim =
      params.intermediate_size + (2 * params.n_groups * params.state_size);
  const std::size_t group_width = params.num_heads / params.n_groups;
  const std::size_t mixer_group_size = params.intermediate_size / params.n_groups;
  const float* gate = projected;
  const float* conv_input = projected + params.intermediate_size;
  const float* dt_pre = conv_input + conv_dim;
  float* layer_conv_state = conv_state + params.conv_state_offset_elems;
  float* layer_ssm_state = ssm_state + params.ssm_state_offset_elems;

  for (std::size_t channel = tid; channel < conv_dim; channel += blockDim.x) {
    float* state_row = layer_conv_state + (channel * params.conv_kernel_size);
    if (params.conv_kernel_size > 1) {
      for (std::size_t tap = 0; tap + 1 < params.conv_kernel_size; ++tap) {
        state_row[tap] = state_row[tap + 1];
      }
    }
    state_row[params.conv_kernel_size - 1] = conv_input[channel];

    float accum = params.conv1d_bias[channel];
    const float* weight_row = params.conv1d_weight + (channel * params.conv_kernel_size);
    for (std::size_t tap = 0; tap < params.conv_kernel_size; ++tap) {
      accum += state_row[tap] * weight_row[tap];
    }
    conv_output[channel] = fused_decode::SiLU(accum);
  }
  __syncthreads();

  for (std::size_t hidden_index = tid; hidden_index < params.intermediate_size;
       hidden_index += blockDim.x) {
    const std::size_t head = hidden_index / params.head_dim;
    const std::size_t group = head / group_width;
    const float hidden_value = conv_output[hidden_index];
    const float dt = fmaxf(
        fused_decode::Softplus(dt_pre[head] + params.dt_bias[head]),
        params.time_step_min);
    const float A = -expf(params.A_log[head]);
    const float D = params.D[head];
    const float decay = expf(dt * A);
    const float* B_head =
        conv_output + params.intermediate_size + (group * params.state_size);
    const float* C_head =
        conv_output + params.intermediate_size +
        (params.n_groups * params.state_size) + (group * params.state_size);
    float* state_row = layer_ssm_state + (hidden_index * params.state_size);
    float accum = 0.0f;
    for (std::size_t state = 0; state < params.state_size; ++state) {
      const float next = state_row[state] * decay + (dt * B_head[state] * hidden_value);
      state_row[state] = next;
      accum += next * C_head[state];
    }
    scan_output[hidden_index] = accum + (hidden_value * D);
  }
  __syncthreads();

  if (tid == 0) {
    for (std::size_t group = 0; group < params.n_groups; ++group) {
      const std::size_t begin = group * mixer_group_size;
      const std::size_t end = begin + mixer_group_size;
      float variance = 0.0f;
      for (std::size_t i = begin; i < end; ++i) {
        const float gated = scan_output[i] * fused_decode::SiLU(gate[i]);
        variance += gated * gated;
        scan_output[i] = gated;
      }
      group_sums[group] = variance;
    }
  }
  __syncthreads();

  for (std::size_t hidden_index = tid; hidden_index < params.intermediate_size;
       hidden_index += blockDim.x) {
    const std::size_t group = hidden_index / mixer_group_size;
    const float variance =
        group_sums[group] / static_cast<float>(mixer_group_size);
    const float rstd = rsqrtf(variance + params.mixer_rms_epsilon);
    scan_output[hidden_index] =
        scan_output[hidden_index] * rstd * params.mixer_norm_weight[hidden_index];
  }
}

}  // namespace

bool FusedMambaDecodeEnabled() {
  return EnvEnabled("NEMOTRON_FORWARD_FUSED_MAMBA_DECODE");
}

bool RunFusedMambaDecode(
    const FusedMambaLayerParams& params,
    RequestExecutionContext& request_context,
    const DeviceTensorFp32& projected,
    DeviceTensorFp32* scan_output) {
  if (!FusedMambaDecodeEnabled() ||
      !request_context.valid() ||
      request_context.mamba_conv_state() == nullptr ||
      request_context.mamba_state() == nullptr ||
      !request_context.mamba_conv_state()->valid() ||
      !request_context.mamba_state()->valid() ||
      !projected.valid() ||
      projected.shape().size() != 2 ||
      projected.shape()[0] != 1 ||
      scan_output == nullptr ||
      !scan_output->valid() ||
      scan_output->shape().size() != 2 ||
      scan_output->shape()[0] != 1 ||
      scan_output->shape()[1] != params.intermediate_size ||
      params.intermediate_size == 0 ||
      params.num_heads == 0 ||
      params.head_dim == 0 ||
      params.state_size == 0 ||
      params.n_groups == 0 ||
      params.n_groups > kMaxMambaGroups ||
      params.conv_kernel_size == 0 ||
      params.mixer_norm_weight == nullptr ||
      params.conv1d_weight == nullptr ||
      params.conv1d_bias == nullptr ||
      params.A_log == nullptr ||
      params.D == nullptr ||
      params.dt_bias == nullptr) {
    return false;
  }

  const std::size_t conv_dim =
      params.intermediate_size + (2 * params.n_groups * params.state_size);
  const std::size_t projection_size =
      params.intermediate_size + conv_dim + params.num_heads;
  if (projected.shape()[1] != projection_size ||
      params.intermediate_size != params.num_heads * params.head_dim ||
      (params.intermediate_size % params.n_groups) != 0 ||
      (params.num_heads % params.n_groups) != 0) {
    return false;
  }

  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(1);
  const std::size_t shared_bytes = conv_dim * sizeof(float);
  FusedMambaDecodeKernel<<<grid, block, shared_bytes>>>(
      params,
      projected.data(),
      request_context.mamba_conv_state()->data(),
      request_context.mamba_state()->data(),
      scan_output->data());
  return CheckCuda(cudaGetLastError());
}

}  // namespace nemotron
