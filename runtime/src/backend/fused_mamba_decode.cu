#include "nemotron/fused_mamba_decode.h"

#include <cuda_runtime.h>

#include <algorithm>

#include "fused_decode_common.cuh"

namespace nemotron {
namespace {

constexpr int kMaxMambaGroups = 32;
constexpr unsigned kWarpSize = 32;
constexpr unsigned kWarpsPerBlock = fused_decode::kThreadsPerBlock / kWarpSize;

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

__device__ __forceinline__ float WarpReduceSum(float value) {
  for (int offset = static_cast<int>(kWarpSize) / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
  }
  return value;
}

__device__ __forceinline__ void ShiftConvState(
    float* state_row,
    std::size_t conv_kernel_size,
    float next_value) {
  if (conv_kernel_size == 4) {
    state_row[0] = state_row[1];
    state_row[1] = state_row[2];
    state_row[2] = state_row[3];
    state_row[3] = next_value;
    return;
  }

  if (conv_kernel_size > 1) {
    for (std::size_t tap = 0; tap + 1 < conv_kernel_size; ++tap) {
      state_row[tap] = state_row[tap + 1];
    }
  }
  state_row[conv_kernel_size - 1] = next_value;
}

__device__ __forceinline__ float ComputeConvOutputChannel(
    const FusedMambaLayerParams& params,
    const float* layer_conv_state,
    std::size_t channel) {
  const float* state_row = layer_conv_state + (channel * params.conv_kernel_size);
  const float* weight_row = params.conv1d_weight + (channel * params.conv_kernel_size);
  float accum = params.conv1d_bias[channel];
  if (params.conv_kernel_size == 4) {
    accum = fmaf(state_row[0], weight_row[0], accum);
    accum = fmaf(state_row[1], weight_row[1], accum);
    accum = fmaf(state_row[2], weight_row[2], accum);
    accum = fmaf(state_row[3], weight_row[3], accum);
    return fused_decode::SiLU(accum);
  }

  for (std::size_t tap = 0; tap < params.conv_kernel_size; ++tap) {
    accum = fmaf(state_row[tap], weight_row[tap], accum);
  }
  return fused_decode::SiLU(accum);
}

__global__ void UpdateMambaConvStateKernel(
    FusedMambaLayerParams params,
    const float* __restrict__ projected,
    float* __restrict__ conv_state) {
  const std::size_t tid = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const std::size_t conv_dim =
      params.intermediate_size + (2 * params.n_groups * params.state_size);
  const float* conv_input = projected + params.intermediate_size;
  float* layer_conv_state = conv_state;

  for (std::size_t channel = tid; channel < conv_dim; channel += stride) {
    float* state_row = layer_conv_state + (channel * params.conv_kernel_size);
    ShiftConvState(state_row, params.conv_kernel_size, conv_input[channel]);
  }
}

__global__ void FusedMambaDecodeHeadKernel(
    FusedMambaLayerParams params,
    const float* __restrict__ projected,
    const float* __restrict__ conv_state,
    float* __restrict__ ssm_state,
    float* __restrict__ scan_output) {
  extern __shared__ float shared_bc[];
  __shared__ float shared_dt;
  __shared__ float shared_decay;
  __shared__ float shared_d;

  const std::size_t head = blockIdx.x;
  if (head >= params.num_heads) {
    return;
  }

  const std::size_t tid = threadIdx.x;
  const std::size_t lane = tid % kWarpSize;
  const std::size_t warp = tid / kWarpSize;
  const std::size_t group_width = params.num_heads / params.n_groups;
  const std::size_t group = head / group_width;
  const std::size_t bc_elements = 2 * params.state_size;
  const std::size_t b_base = params.intermediate_size + (group * params.state_size);
  const std::size_t c_base =
      params.intermediate_size + (params.n_groups * params.state_size) + (group * params.state_size);
  const std::size_t conv_dim =
      params.intermediate_size + (2 * params.n_groups * params.state_size);
  const float* dt_pre = projected + params.intermediate_size + conv_dim;
  const float* layer_conv_state = conv_state;
  float* layer_ssm_state = ssm_state;

  for (std::size_t idx = tid; idx < bc_elements; idx += blockDim.x) {
    const bool is_b = idx < params.state_size;
    const std::size_t channel = is_b ? (b_base + idx) : (c_base + (idx - params.state_size));
    shared_bc[idx] = ComputeConvOutputChannel(params, layer_conv_state, channel);
  }

  if (tid == 0) {
    const float dt = fmaxf(
        fused_decode::Softplus(dt_pre[head] + params.dt_bias[head]),
        params.time_step_min);
    const float A = -expf(params.A_log[head]);
    shared_dt = dt;
    shared_decay = expf(dt * A);
    shared_d = params.D[head];
  }
  __syncthreads();

  const std::size_t head_base = head * params.head_dim;
  const float* shared_b = shared_bc;
  const float* shared_c = shared_bc + params.state_size;
  for (std::size_t hidden_offset = warp; hidden_offset < params.head_dim; hidden_offset += kWarpsPerBlock) {
    const std::size_t hidden_index = head_base + hidden_offset;
    float hidden_value = 0.0f;
    if (lane == 0) {
      hidden_value = ComputeConvOutputChannel(params, layer_conv_state, hidden_index);
    }
    hidden_value = __shfl_sync(0xFFFFFFFFu, hidden_value, 0);
    const float input_term = shared_dt * hidden_value;

    float partial = 0.0f;
    float* state_row = layer_ssm_state + (hidden_index * params.state_size);
    for (std::size_t state = lane; state < params.state_size; state += kWarpSize) {
      const float next = fmaf(input_term, shared_b[state], state_row[state] * shared_decay);
      state_row[state] = next;
      partial = fmaf(next, shared_c[state], partial);
    }
    partial = WarpReduceSum(partial);
    if (lane == 0) {
      scan_output[hidden_index] = partial + (hidden_value * shared_d);
    }
  }
}

__global__ void FusedMambaDecodeGroupNormKernel(
    FusedMambaLayerParams params,
    const float* __restrict__ projected,
    float* __restrict__ scan_output) {
  __shared__ float warp_sums[kWarpsPerBlock];
  __shared__ float shared_rstd;

  const std::size_t group = blockIdx.x;
  if (group >= params.n_groups) {
    return;
  }

  const std::size_t tid = threadIdx.x;
  const std::size_t lane = tid % kWarpSize;
  const std::size_t warp = tid / kWarpSize;
  const std::size_t mixer_group_size = params.intermediate_size / params.n_groups;
  const std::size_t begin = group * mixer_group_size;
  const std::size_t end = begin + mixer_group_size;
  const float* gate = projected;

  float variance_sum = 0.0f;
  for (std::size_t hidden_index = begin + tid; hidden_index < end; hidden_index += blockDim.x) {
    const float gated = scan_output[hidden_index] * fused_decode::SiLU(gate[hidden_index]);
    scan_output[hidden_index] = gated;
    variance_sum = fmaf(gated, gated, variance_sum);
  }

  variance_sum = WarpReduceSum(variance_sum);
  if (lane == 0) {
    warp_sums[warp] = variance_sum;
  }
  __syncthreads();

  if (warp == 0) {
    float block_sum = lane < kWarpsPerBlock ? warp_sums[lane] : 0.0f;
    block_sum = WarpReduceSum(block_sum);
    if (lane == 0) {
      const float variance = block_sum / static_cast<float>(mixer_group_size);
      shared_rstd = rsqrtf(variance + params.mixer_rms_epsilon);
    }
  }
  __syncthreads();

  for (std::size_t hidden_index = begin + tid; hidden_index < end; hidden_index += blockDim.x) {
    scan_output[hidden_index] =
        scan_output[hidden_index] * shared_rstd * params.mixer_norm_weight[hidden_index];
  }
}

}  // namespace

bool RunFusedMambaDecode(
    const FusedMambaLayerParams& params,
    const DeviceTensorFp32& projected,
    DeviceTensorFp32* scan_output) {
  if (!projected.valid() ||
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
      params.conv_state_elems == 0 ||
      params.ssm_state_elems == 0 ||
      params.conv_state == nullptr ||
      params.ssm_state == nullptr ||
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
  const std::size_t conv_state_elems = conv_dim * params.conv_kernel_size;
  const std::size_t ssm_state_elems =
      params.num_heads * params.head_dim * params.state_size;
  if (projected.shape()[1] != projection_size ||
      params.intermediate_size != params.num_heads * params.head_dim ||
      (params.intermediate_size % params.n_groups) != 0 ||
      (params.num_heads % params.n_groups) != 0 ||
      params.conv_state_elems != conv_state_elems ||
      params.ssm_state_elems != ssm_state_elems) {
    return false;
  }

  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 conv_grid(
      static_cast<unsigned>((conv_dim + block.x - 1) / block.x));
  UpdateMambaConvStateKernel<<<conv_grid, block>>>(
      params,
      projected.data(),
      params.conv_state);
  if (!CheckCuda(cudaGetLastError())) {
    return false;
  }

  const dim3 head_grid(static_cast<unsigned>(params.num_heads));
  const std::size_t head_shared_bytes = 2 * params.state_size * sizeof(float);
  FusedMambaDecodeHeadKernel<<<head_grid, block, head_shared_bytes>>>(
      params,
      projected.data(),
      params.conv_state,
      params.ssm_state,
      scan_output->data());
  if (!CheckCuda(cudaGetLastError())) {
    return false;
  }

  const dim3 norm_grid(static_cast<unsigned>(params.n_groups));
  FusedMambaDecodeGroupNormKernel<<<norm_grid, block>>>(
      params,
      projected.data(),
      scan_output->data());
  return CheckCuda(cudaGetLastError());
}

}  // namespace nemotron
