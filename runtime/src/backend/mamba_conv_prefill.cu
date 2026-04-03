#include "nemotron/mamba_conv_prefill.h"

#include <cuda_runtime.h>

#include <cstddef>

#include "fused_decode_common.cuh"

namespace nemotron {
namespace {

constexpr unsigned kChannelsPerBlock = 64;
constexpr std::size_t kMaxSupportedConvKernelSize = 16;

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

template <int kKernelSize>
__global__ void MambaConvPrefillFixedKernel(
    const float* __restrict__ projected,
    std::size_t projected_stride,
    std::size_t token_count,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    const float* __restrict__ conv1d_weight,
    const float* __restrict__ conv1d_bias,
    const float* __restrict__ initial_conv_state,
    float* __restrict__ final_conv_state,
    float* __restrict__ conv_output) {
  const std::size_t channel =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (channel >= conv_dim) {
    return;
  }

  const float* weight_row = conv1d_weight + (channel * kKernelSize);
  const float* initial_row = initial_conv_state + (channel * kKernelSize);
  float* final_row = final_conv_state + (channel * kKernelSize);
  float state[kKernelSize];
#pragma unroll
  for (int tap = 0; tap < kKernelSize; ++tap) {
    state[tap] = initial_row[tap];
  }

  const float bias = conv1d_bias[channel];
  for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
    const float next_value =
        projected[(token_index * projected_stride) + intermediate_size + channel];
    if constexpr (kKernelSize > 1) {
      #pragma unroll
      for (int tap = 0; tap + 1 < kKernelSize; ++tap) {
        state[tap] = state[tap + 1];
      }
    }
    state[kKernelSize - 1] = next_value;

    float accum = bias;
#pragma unroll
    for (int tap = 0; tap < kKernelSize; ++tap) {
      accum = fmaf(state[tap], weight_row[tap], accum);
    }
    conv_output[(token_index * conv_dim) + channel] = fused_decode::SiLU(accum);
  }

#pragma unroll
  for (int tap = 0; tap < kKernelSize; ++tap) {
    final_row[tap] = state[tap];
  }
}

__global__ void MambaConvPrefillGenericKernel(
    const float* __restrict__ projected,
    std::size_t projected_stride,
    std::size_t token_count,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t conv_kernel_size,
    const float* __restrict__ conv1d_weight,
    const float* __restrict__ conv1d_bias,
    const float* __restrict__ initial_conv_state,
    float* __restrict__ final_conv_state,
    float* __restrict__ conv_output) {
  const std::size_t channel =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (channel >= conv_dim) {
    return;
  }

  const float* weight_row = conv1d_weight + (channel * conv_kernel_size);
  const float* initial_row = initial_conv_state + (channel * conv_kernel_size);
  float* final_row = final_conv_state + (channel * conv_kernel_size);
  float state[kMaxSupportedConvKernelSize];
#pragma unroll
  for (std::size_t tap = 0; tap < kMaxSupportedConvKernelSize; ++tap) {
    state[tap] = 0.0f;
  }
  for (std::size_t tap = 0; tap < conv_kernel_size; ++tap) {
    state[tap] = initial_row[tap];
  }

  const float bias = conv1d_bias[channel];
  for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
    const float next_value =
        projected[(token_index * projected_stride) + intermediate_size + channel];
    for (std::size_t tap = 0; tap + 1 < conv_kernel_size; ++tap) {
      state[tap] = state[tap + 1];
    }
    state[conv_kernel_size - 1] = next_value;

    float accum = bias;
    for (std::size_t tap = 0; tap < conv_kernel_size; ++tap) {
      accum = fmaf(state[tap], weight_row[tap], accum);
    }
    conv_output[(token_index * conv_dim) + channel] = fused_decode::SiLU(accum);
  }

  for (std::size_t tap = 0; tap < conv_kernel_size; ++tap) {
    final_row[tap] = state[tap];
  }
}

}  // namespace

bool RunMambaConvPrefill(
    const MambaConvPrefillParams& params,
    RequestExecutionContext& request_context,
    const DeviceTensorFp32& projected,
    DeviceTensorFp32* conv_output,
    const DeviceTensorFp32* initial_conv_state) {
  if (!request_context.valid() ||
      request_context.mamba_conv_state() == nullptr ||
      !request_context.mamba_conv_state()->valid() ||
      !projected.valid() ||
      projected.shape().size() != 2 ||
      projected.shape()[0] == 0 ||
      conv_output == nullptr ||
      !conv_output->valid() ||
      conv_output->shape().size() != 2 ||
      params.intermediate_size == 0 ||
      params.state_size == 0 ||
      params.n_groups == 0 ||
      params.conv_kernel_size == 0 ||
      params.conv_kernel_size > kMaxSupportedConvKernelSize ||
      params.conv1d_weight == nullptr ||
      params.conv1d_bias == nullptr ||
      (initial_conv_state != nullptr &&
       (!initial_conv_state->valid() || initial_conv_state->shape().empty()))) {
    return false;
  }

  const std::size_t conv_dim =
      params.intermediate_size + (2 * params.n_groups * params.state_size);
  const std::size_t conv_state_elems = conv_dim * params.conv_kernel_size;
  const std::size_t required_projection_cols = params.intermediate_size + conv_dim;
  if (projected.shape()[1] < required_projection_cols ||
      conv_output->shape()[0] != projected.shape()[0] ||
      conv_output->shape()[1] != conv_dim ||
      request_context.mamba_conv_state()->numel() <
          params.conv_state_offset_elems + conv_state_elems) {
    return false;
  }

  if (initial_conv_state != nullptr &&
      initial_conv_state->numel() < params.conv_state_offset_elems + conv_state_elems) {
    return false;
  }

  float* layer_conv_state =
      request_context.mamba_conv_state()->data() + params.conv_state_offset_elems;
  const float* initial_layer_conv_state =
      (initial_conv_state != nullptr ? initial_conv_state->data()
                                     : request_context.mamba_conv_state()->data()) +
      params.conv_state_offset_elems;

  const dim3 block(kChannelsPerBlock);
  const dim3 grid(static_cast<unsigned>((conv_dim + block.x - 1) / block.x));
  switch (params.conv_kernel_size) {
    case 4:
      MambaConvPrefillFixedKernel<4><<<grid, block>>>(
          projected.data(),
          projected.shape()[1],
          projected.shape()[0],
          params.intermediate_size,
          conv_dim,
          params.conv1d_weight,
          params.conv1d_bias,
          initial_layer_conv_state,
          layer_conv_state,
          conv_output->data());
      break;
    case 3:
      MambaConvPrefillFixedKernel<3><<<grid, block>>>(
          projected.data(),
          projected.shape()[1],
          projected.shape()[0],
          params.intermediate_size,
          conv_dim,
          params.conv1d_weight,
          params.conv1d_bias,
          initial_layer_conv_state,
          layer_conv_state,
          conv_output->data());
      break;
    case 2:
      MambaConvPrefillFixedKernel<2><<<grid, block>>>(
          projected.data(),
          projected.shape()[1],
          projected.shape()[0],
          params.intermediate_size,
          conv_dim,
          params.conv1d_weight,
          params.conv1d_bias,
          initial_layer_conv_state,
          layer_conv_state,
          conv_output->data());
      break;
    case 1:
      MambaConvPrefillFixedKernel<1><<<grid, block>>>(
          projected.data(),
          projected.shape()[1],
          projected.shape()[0],
          params.intermediate_size,
          conv_dim,
          params.conv1d_weight,
          params.conv1d_bias,
          initial_layer_conv_state,
          layer_conv_state,
          conv_output->data());
      break;
    default:
      MambaConvPrefillGenericKernel<<<grid, block>>>(
          projected.data(),
          projected.shape()[1],
          projected.shape()[0],
          params.intermediate_size,
          conv_dim,
          params.conv_kernel_size,
          params.conv1d_weight,
          params.conv1d_bias,
          initial_layer_conv_state,
          layer_conv_state,
          conv_output->data());
      break;
  }

  return CheckCuda(cudaGetLastError());
}

}  // namespace nemotron
