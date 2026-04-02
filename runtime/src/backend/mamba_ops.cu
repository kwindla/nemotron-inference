#include "nemotron/mamba_ops.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>

namespace nemotron {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kDecodeConvThreads = 128;
constexpr int kDecodeSsmThreads = 64;

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

__device__ __forceinline__ float SigmoidDevice(float value) {
  if (value >= 0.0f) {
    const float exp_neg = expf(-value);
    return 1.0f / (1.0f + exp_neg);
  }
  const float exp_pos = expf(value);
  return exp_pos / (1.0f + exp_pos);
}

__device__ __forceinline__ float SiLUDevice(float value) {
  return value * SigmoidDevice(value);
}

__device__ __forceinline__ float SoftplusDevice(float value) {
  if (value > 20.0f) {
    return value;
  }
  if (value < -20.0f) {
    return expf(value);
  }
  return log1pf(expf(value));
}

__device__ __forceinline__ float LoadMambaValue(const float* input, std::size_t index) {
  return input[index];
}

__device__ __forceinline__ float LoadMambaValue(const __nv_bfloat16* input, std::size_t index) {
  return __bfloat162float(input[index]);
}

__device__ __forceinline__ void StoreMambaValue(float* output, std::size_t index, float value) {
  output[index] = value;
}

__device__ __forceinline__ void StoreMambaValue(
    __nv_bfloat16* output,
    std::size_t index,
    float value) {
  output[index] = __float2bfloat16(value);
}

template <int kWidth, typename ProjectedT, typename OutputT>
__global__ __launch_bounds__(kDecodeConvThreads) void MambaDecodeCausalConv1dUpdateKernel(
    const ProjectedT* projected,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    const float* conv_weight,
    const float* conv_bias,
    float* conv_state,
    OutputT* conv_output) {
  const std::size_t channel = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (channel >= conv_dim) {
    return;
  }

  const float x_value = LoadMambaValue(projected, intermediate_size + channel);
  float* state_row = conv_state + channel * kWidth;
  const float* weight_row = conv_weight + channel * kWidth;

  float state_values[kWidth];
#pragma unroll
  for (int tap = 0; tap < kWidth; ++tap) {
    state_values[tap] = state_row[tap];
  }
#pragma unroll
  for (int tap = 0; tap + 1 < kWidth; ++tap) {
    state_values[tap] = state_values[tap + 1];
    state_row[tap] = state_values[tap];
  }
  state_values[kWidth - 1] = x_value;
  state_row[kWidth - 1] = x_value;

  float accum = conv_bias[channel];
#pragma unroll
  for (int tap = 0; tap < kWidth; ++tap) {
    accum += state_values[tap] * weight_row[tap];
  }
  StoreMambaValue(conv_output, channel, SiLUDevice(accum));
}

template <typename ProjectedT, typename ConvOutputT, typename OutputT>
__global__ __launch_bounds__(kDecodeSsmThreads) void MambaSelectiveStateUpdateDecodeKernel(
    const ProjectedT* projected,
    const ConvOutputT* conv_output,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t num_heads,
    std::size_t head_dim,
    std::size_t state_size,
    std::size_t n_groups,
    float time_step_min,
    const float* a_log,
    const float* d,
    const float* dt_bias,
    float* ssm_state,
    OutputT* gated_output) {
  const std::size_t head = static_cast<std::size_t>(blockIdx.x);
  if (head >= num_heads) {
    return;
  }

  const std::size_t group_width = num_heads / n_groups;
  const std::size_t group = head / group_width;
  const std::size_t b_offset = intermediate_size + group * state_size;
  const std::size_t c_offset =
      intermediate_size + (n_groups * state_size) + group * state_size;
  const std::size_t hidden_begin = head * head_dim;

  extern __shared__ float shared_state[];
  float* shared_b = shared_state;
  float* shared_c = shared_state + state_size;
  __shared__ float shared_dt;
  __shared__ float shared_decay;
  __shared__ float shared_d;

  for (std::size_t state = threadIdx.x; state < state_size; state += blockDim.x) {
    shared_b[state] = LoadMambaValue(conv_output, b_offset + state);
    shared_c[state] = LoadMambaValue(conv_output, c_offset + state);
  }
  if (threadIdx.x == 0) {
    const float dt_base =
        LoadMambaValue(projected, intermediate_size + conv_dim + head) + dt_bias[head];
    shared_dt = fmaxf(SoftplusDevice(dt_base), time_step_min);
    shared_decay = expf(shared_dt * (-expf(a_log[head])));
    shared_d = d[head];
  }
  __syncthreads();

  const float dt = shared_dt;
  const float decay = shared_decay;
  const float d_value = shared_d;
  for (std::size_t local_hidden = threadIdx.x; local_hidden < head_dim; local_hidden += blockDim.x) {
    const std::size_t hidden_index = hidden_begin + local_hidden;
    if (hidden_index >= intermediate_size) {
      continue;
    }

    const float hidden_value = LoadMambaValue(conv_output, hidden_index);
    const float gate_value = SiLUDevice(LoadMambaValue(projected, hidden_index));
    const float dt_hidden = dt * hidden_value;
    float* state_row = ssm_state + hidden_index * state_size;
    float accum = 0.0f;
#pragma unroll 4
    for (std::size_t state = 0; state < state_size; ++state) {
      const float next = state_row[state] * decay + (shared_b[state] * dt_hidden);
      state_row[state] = next;
      accum += next * shared_c[state];
    }
    const float y_value = accum + (hidden_value * d_value);
    StoreMambaValue(gated_output, hidden_index, y_value * gate_value);
  }
}

template <typename ProjectedT, typename OutputT>
__global__ void MambaConv1dSiluUpdateKernel(
    const ProjectedT* projected,
    std::size_t token_count,
    std::size_t projection_size,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t conv_kernel_size,
    const float* conv_weight,
    const float* conv_bias,
    float* conv_state,
    OutputT* conv_output) {
  const std::size_t channel = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (channel >= conv_dim) {
    return;
  }

  float* state_row = conv_state + channel * conv_kernel_size;
  const float* weight_row = conv_weight + channel * conv_kernel_size;
  for (std::size_t token = 0; token < token_count; ++token) {
    const float conv_input =
        LoadMambaValue(projected, token * projection_size + intermediate_size + channel);
    if (conv_kernel_size > 1) {
      for (std::size_t tap = 0; tap + 1 < conv_kernel_size; ++tap) {
        state_row[tap] = state_row[tap + 1];
      }
    }
    state_row[conv_kernel_size - 1] = conv_input;

    float accum = conv_bias[channel];
    for (std::size_t tap = 0; tap < conv_kernel_size; ++tap) {
      accum += state_row[tap] * weight_row[tap];
    }
    StoreMambaValue(conv_output, token * conv_dim + channel, SiLUDevice(accum));
  }
}

template <typename ProjectedT, typename ConvOutputT, typename OutputT>
__global__ void MambaSsmUpdateKernel(
    const ProjectedT* projected,
    const ConvOutputT* conv_output,
    std::size_t token_count,
    std::size_t projection_size,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t num_heads,
    std::size_t head_dim,
    std::size_t state_size,
    std::size_t n_groups,
    float time_step_min,
    const float* a_log,
    const float* d,
    const float* dt_bias,
    float* ssm_state,
    OutputT* y_output) {
  const std::size_t hidden_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (hidden_index >= intermediate_size) {
    return;
  }

  const std::size_t head = hidden_index / head_dim;
  const std::size_t group_width = num_heads / n_groups;
  const std::size_t group = head / group_width;
  const float a = -expf(a_log[head]);
  const float d_value = d[head];
  float* state_row = ssm_state + hidden_index * state_size;

  for (std::size_t token = 0; token < token_count; ++token) {
    const float hidden_value = LoadMambaValue(conv_output, token * conv_dim + hidden_index);
    const float dt_base =
        LoadMambaValue(projected, token * projection_size + intermediate_size + conv_dim + head) +
        dt_bias[head];
    const float dt = fmaxf(SoftplusDevice(dt_base), time_step_min);
    const float decay = expf(dt * a);
    const std::size_t grouped_b_offset =
        token * conv_dim + intermediate_size + group * state_size;
    const std::size_t grouped_c_offset =
        token * conv_dim + intermediate_size + n_groups * state_size + group * state_size;

    float accum = 0.0f;
    for (std::size_t state = 0; state < state_size; ++state) {
      const float next = state_row[state] * decay +
                         (dt * LoadMambaValue(conv_output, grouped_b_offset + state) * hidden_value);
      state_row[state] = next;
      accum += next * LoadMambaValue(conv_output, grouped_c_offset + state);
    }
    StoreMambaValue(
        y_output,
        token * intermediate_size + hidden_index,
        accum + hidden_value * d_value);
  }
}

template <typename YOutputT, typename ProjectedT, typename OutputT>
__global__ void GroupedRmsNormGatedKernel(
    const YOutputT* y_output,
    const ProjectedT* projected,
    const float* mixer_norm_weight,
    std::size_t rows,
    std::size_t intermediate_size,
    std::size_t mixer_group_size,
    std::size_t projection_size,
    float epsilon,
    OutputT* output) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.x);
  const std::size_t group = static_cast<std::size_t>(blockIdx.y);
  if (row >= rows) {
    return;
  }

  extern __shared__ float shared_sum[];
  const std::size_t begin = group * mixer_group_size;
  const std::size_t end = begin + mixer_group_size;
  const std::size_t row_output_offset = row * intermediate_size;
  const std::size_t row_proj_offset = row * projection_size;

  float local_sum = 0.0f;
  for (std::size_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
    const float gated =
        LoadMambaValue(y_output, row_output_offset + i) *
        SiLUDevice(LoadMambaValue(projected, row_proj_offset + i));
    local_sum += gated * gated;
  }
  shared_sum[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv_rms =
      rsqrtf((shared_sum[0] / static_cast<float>(mixer_group_size)) + epsilon);
  for (std::size_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
    const float gated =
        LoadMambaValue(y_output, row_output_offset + i) *
        SiLUDevice(LoadMambaValue(projected, row_proj_offset + i));
    StoreMambaValue(output, row_output_offset + i, gated * inv_rms * mixer_norm_weight[i]);
  }
}

template <typename InputT, typename OutputT>
__global__ void GroupedRmsNormKernel(
    const InputT* input,
    const float* mixer_norm_weight,
    std::size_t rows,
    std::size_t intermediate_size,
    std::size_t mixer_group_size,
    float epsilon,
    OutputT* output) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.x);
  const std::size_t group = static_cast<std::size_t>(blockIdx.y);
  if (row >= rows) {
    return;
  }

  extern __shared__ float shared_sum[];
  const std::size_t begin = group * mixer_group_size;
  const std::size_t end = begin + mixer_group_size;
  const std::size_t row_offset = row * intermediate_size;

  float local_sum = 0.0f;
  for (std::size_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
    const float value = LoadMambaValue(input, row_offset + i);
    local_sum += value * value;
  }
  shared_sum[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv_rms =
      rsqrtf((shared_sum[0] / static_cast<float>(mixer_group_size)) + epsilon);
  for (std::size_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
    const float value = LoadMambaValue(input, row_offset + i);
    StoreMambaValue(output, row_offset + i, value * inv_rms * mixer_norm_weight[i]);
  }
}

template <typename ProjectedT, typename OutputT>
bool LaunchMambaDecodeCausalConv1dUpdate(
    const ProjectedT* projected,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t conv_kernel_size,
    const float* conv_weight,
    const float* conv_bias,
    float* conv_state,
    OutputT* conv_output,
    cudaStream_t stream) {
  const int grid_size =
      static_cast<int>((conv_dim + kDecodeConvThreads - 1u) / kDecodeConvThreads);
  switch (conv_kernel_size) {
    case 2:
      MambaDecodeCausalConv1dUpdateKernel<2><<<grid_size, kDecodeConvThreads, 0, stream>>>(
          projected,
          intermediate_size,
          conv_dim,
          conv_weight,
          conv_bias,
          conv_state,
          conv_output);
      break;
    case 3:
      MambaDecodeCausalConv1dUpdateKernel<3><<<grid_size, kDecodeConvThreads, 0, stream>>>(
          projected,
          intermediate_size,
          conv_dim,
          conv_weight,
          conv_bias,
          conv_state,
          conv_output);
      break;
    case 4:
      MambaDecodeCausalConv1dUpdateKernel<4><<<grid_size, kDecodeConvThreads, 0, stream>>>(
          projected,
          intermediate_size,
          conv_dim,
          conv_weight,
          conv_bias,
          conv_state,
          conv_output);
      break;
    case 5:
      MambaDecodeCausalConv1dUpdateKernel<5><<<grid_size, kDecodeConvThreads, 0, stream>>>(
          projected,
          intermediate_size,
          conv_dim,
          conv_weight,
          conv_bias,
          conv_state,
          conv_output);
      break;
    default:
      return false;
  }
  return CheckCuda(cudaGetLastError());
}

template <typename ProjectedT, typename ConvOutputT, typename OutputT>
bool LaunchMambaSelectiveStateUpdateDecode(
    const ProjectedT* projected,
    const ConvOutputT* conv_output,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t num_heads,
    std::size_t head_dim,
    std::size_t state_size,
    std::size_t n_groups,
    float time_step_min,
    const float* a_log,
    const float* d,
    const float* dt_bias,
    float* ssm_state,
    OutputT* gated_output,
    cudaStream_t stream) {
  MambaSelectiveStateUpdateDecodeKernel<<<static_cast<unsigned int>(num_heads),
                                          kDecodeSsmThreads,
                                          (2 * state_size) * sizeof(float),
                                          stream>>>(
      projected,
      conv_output,
      intermediate_size,
      conv_dim,
      num_heads,
      head_dim,
      state_size,
      n_groups,
      time_step_min,
      a_log,
      d,
      dt_bias,
      ssm_state,
      gated_output);
  return CheckCuda(cudaGetLastError());
}

template <typename ProjectedT, typename OutputT>
__global__ void MambaDecodeStepFusedKernel(
    const ProjectedT* projected,
    std::size_t projection_size,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t num_heads,
    std::size_t head_dim,
    std::size_t state_size,
    std::size_t n_groups,
    std::size_t conv_kernel_size,
    float time_step_min,
    float mixer_rms_epsilon,
    std::size_t conv_state_offset_elems,
    std::size_t ssm_state_offset_elems,
    const float* conv_weight,
    const float* conv_bias,
    const float* a_log,
    const float* d,
    const float* dt_bias,
    const float* mixer_norm_weight,
    float* conv_state,
    float* ssm_state,
    OutputT* output) {
  const std::size_t group = static_cast<std::size_t>(blockIdx.x);
  if (group >= n_groups) {
    return;
  }

  const std::size_t group_hidden_size = intermediate_size / n_groups;
  const std::size_t hidden_begin = group * group_hidden_size;
  const std::size_t b_begin = intermediate_size + group * state_size;
  const std::size_t c_begin = intermediate_size + (n_groups * state_size) + group * state_size;
  float* conv_state_base = conv_state + conv_state_offset_elems;
  float* ssm_state_base = ssm_state + ssm_state_offset_elems;

  extern __shared__ float shared_storage[];
  float* shared_b = shared_storage;
  float* shared_c = shared_b + state_size;
  float* shared_gated = shared_c + state_size;
  float* shared_sum = shared_gated + group_hidden_size;

  for (std::size_t local_state = threadIdx.x; local_state < state_size; local_state += blockDim.x) {
    const std::size_t b_channel = b_begin + local_state;
    float* b_state_row = conv_state_base + b_channel * conv_kernel_size;
    const float* b_weight_row = conv_weight + b_channel * conv_kernel_size;
    const float b_input = LoadMambaValue(projected, intermediate_size + b_channel);
    if (conv_kernel_size > 1) {
      for (std::size_t tap = 0; tap + 1 < conv_kernel_size; ++tap) {
        b_state_row[tap] = b_state_row[tap + 1];
      }
    }
    b_state_row[conv_kernel_size - 1] = b_input;

    float b_accum = conv_bias[b_channel];
    for (std::size_t tap = 0; tap < conv_kernel_size; ++tap) {
      b_accum += b_state_row[tap] * b_weight_row[tap];
    }
    shared_b[local_state] = SiLUDevice(b_accum);

    const std::size_t c_channel = c_begin + local_state;
    float* c_state_row = conv_state_base + c_channel * conv_kernel_size;
    const float* c_weight_row = conv_weight + c_channel * conv_kernel_size;
    const float c_input = LoadMambaValue(projected, intermediate_size + c_channel);
    if (conv_kernel_size > 1) {
      for (std::size_t tap = 0; tap + 1 < conv_kernel_size; ++tap) {
        c_state_row[tap] = c_state_row[tap + 1];
      }
    }
    c_state_row[conv_kernel_size - 1] = c_input;

    float c_accum = conv_bias[c_channel];
    for (std::size_t tap = 0; tap < conv_kernel_size; ++tap) {
      c_accum += c_state_row[tap] * c_weight_row[tap];
    }
    shared_c[local_state] = SiLUDevice(c_accum);
  }
  __syncthreads();

  float local_sum = 0.0f;
  for (std::size_t local_hidden = threadIdx.x;
       local_hidden < group_hidden_size;
       local_hidden += blockDim.x) {
    const std::size_t hidden_index = hidden_begin + local_hidden;
    float* hidden_conv_state_row = conv_state_base + hidden_index * conv_kernel_size;
    const float* hidden_weight_row = conv_weight + hidden_index * conv_kernel_size;
    const float hidden_input = LoadMambaValue(projected, intermediate_size + hidden_index);
    if (conv_kernel_size > 1) {
      for (std::size_t tap = 0; tap + 1 < conv_kernel_size; ++tap) {
        hidden_conv_state_row[tap] = hidden_conv_state_row[tap + 1];
      }
    }
    hidden_conv_state_row[conv_kernel_size - 1] = hidden_input;

    float hidden_accum = conv_bias[hidden_index];
    for (std::size_t tap = 0; tap < conv_kernel_size; ++tap) {
      hidden_accum += hidden_conv_state_row[tap] * hidden_weight_row[tap];
    }
    const float hidden_value = SiLUDevice(hidden_accum);
    const std::size_t head = hidden_index / head_dim;
    const float a = -expf(a_log[head]);
    const float d_value = d[head];
    const float dt_base =
        LoadMambaValue(projected, intermediate_size + conv_dim + head) + dt_bias[head];
    const float dt = fmaxf(SoftplusDevice(dt_base), time_step_min);
    const float decay = expf(dt * a);

    float* state_row = ssm_state_base + hidden_index * state_size;
    float y_accum = 0.0f;
    for (std::size_t state = 0; state < state_size; ++state) {
      const float next =
          state_row[state] * decay + (dt * shared_b[state] * hidden_value);
      state_row[state] = next;
      y_accum += next * shared_c[state];
    }
    const float y_value = y_accum + hidden_value * d_value;
    const float gated = y_value * SiLUDevice(LoadMambaValue(projected, hidden_index));
    shared_gated[local_hidden] = gated;
    local_sum += gated * gated;
  }

  shared_sum[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv_rms =
      rsqrtf((shared_sum[0] / static_cast<float>(group_hidden_size)) + mixer_rms_epsilon);
  for (std::size_t local_hidden = threadIdx.x;
       local_hidden < group_hidden_size;
       local_hidden += blockDim.x) {
    const std::size_t hidden_index = hidden_begin + local_hidden;
    StoreMambaValue(
        output,
        hidden_index,
        shared_gated[local_hidden] * inv_rms * mixer_norm_weight[hidden_index]);
  }
}

}  // namespace

bool MambaCausalConv1dUpdateDecodeFp32(
    const DeviceTensorFp32& projected,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t conv_kernel_size,
    std::size_t conv_state_offset_elems,
    const DeviceTensorFp32& conv_weight,
    const DeviceTensorFp32& conv_bias,
    DeviceTensorFp32* conv_state,
    DeviceTensorFp32* conv_output,
    cudaStream_t stream) {
  if (!projected.valid() ||
      !conv_weight.valid() ||
      !conv_bias.valid() ||
      conv_state == nullptr ||
      !conv_state->valid() ||
      conv_output == nullptr ||
      !conv_output->valid() ||
      projected.shape().size() != 2 ||
      projected.shape()[0] != 1 ||
      conv_output->shape().size() != 2 ||
      conv_output->shape()[0] != 1 ||
      conv_output->shape()[1] != conv_dim ||
      conv_bias.shape().size() != 1 ||
      conv_weight.shape().size() != 1 ||
      conv_weight.numel() != conv_dim * conv_kernel_size ||
      conv_bias.numel() != conv_dim ||
      conv_kernel_size < 2 ||
      conv_state_offset_elems + (conv_dim * conv_kernel_size) > conv_state->numel()) {
    return false;
  }

  return LaunchMambaDecodeCausalConv1dUpdate(
      projected.data(),
      intermediate_size,
      conv_dim,
      conv_kernel_size,
      conv_weight.data(),
      conv_bias.data(),
      conv_state->data() + conv_state_offset_elems,
      conv_output->data(),
      stream);
}

bool MambaCausalConv1dUpdateDecodeBf16(
    const DeviceTensorBf16& projected,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t conv_kernel_size,
    std::size_t conv_state_offset_elems,
    const DeviceTensorFp32& conv_weight,
    const DeviceTensorFp32& conv_bias,
    DeviceTensorFp32* conv_state,
    DeviceTensorBf16* conv_output,
    cudaStream_t stream) {
  if (!projected.valid() ||
      !conv_weight.valid() ||
      !conv_bias.valid() ||
      conv_state == nullptr ||
      !conv_state->valid() ||
      conv_output == nullptr ||
      !conv_output->valid() ||
      projected.shape().size() != 2 ||
      projected.shape()[0] != 1 ||
      conv_output->shape().size() != 2 ||
      conv_output->shape()[0] != 1 ||
      conv_output->shape()[1] != conv_dim ||
      conv_bias.shape().size() != 1 ||
      conv_weight.shape().size() != 1 ||
      conv_weight.numel() != conv_dim * conv_kernel_size ||
      conv_bias.numel() != conv_dim ||
      conv_kernel_size < 2 ||
      conv_state_offset_elems + (conv_dim * conv_kernel_size) > conv_state->numel()) {
    return false;
  }

  return LaunchMambaDecodeCausalConv1dUpdate(
      projected.data(),
      intermediate_size,
      conv_dim,
      conv_kernel_size,
      conv_weight.data(),
      conv_bias.data(),
      conv_state->data() + conv_state_offset_elems,
      conv_output->data(),
      stream);
}

bool MambaConv1dSiluUpdateFp32(
    const DeviceTensorFp32& projected,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t conv_kernel_size,
    std::size_t conv_state_offset_elems,
    const DeviceTensorFp32& conv_weight,
    const DeviceTensorFp32& conv_bias,
    DeviceTensorFp32* conv_state,
    DeviceTensorFp32* conv_output,
    cudaStream_t stream) {
  if (!projected.valid() ||
      !conv_weight.valid() ||
      !conv_bias.valid() ||
      conv_state == nullptr ||
      !conv_state->valid() ||
      conv_output == nullptr ||
      !conv_output->valid() ||
      projected.shape().size() != 2 ||
      conv_output->shape().size() != 2 ||
      conv_bias.shape().size() != 1 ||
      conv_weight.shape().size() != 1 ||
      conv_output->shape()[0] != projected.shape()[0] ||
      conv_output->shape()[1] != conv_dim ||
      conv_weight.numel() != conv_dim * conv_kernel_size ||
      conv_bias.numel() != conv_dim ||
      conv_state_offset_elems + (conv_dim * conv_kernel_size) > conv_state->numel()) {
    return false;
  }

  const std::size_t projection_size = projected.shape()[1];
  constexpr int kBlockSize = 256;
  const int grid_size = static_cast<int>((conv_dim + kBlockSize - 1u) / kBlockSize);
  MambaConv1dSiluUpdateKernel<<<grid_size, kBlockSize, 0, stream>>>(
      projected.data(),
      projected.shape()[0],
      projection_size,
      intermediate_size,
      conv_dim,
      conv_kernel_size,
      conv_weight.data(),
      conv_bias.data(),
      conv_state->data() + conv_state_offset_elems,
      conv_output->data());
  return CheckCuda(cudaGetLastError());
}

bool MambaConv1dSiluUpdateBf16(
    const DeviceTensorBf16& projected,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t conv_kernel_size,
    std::size_t conv_state_offset_elems,
    const DeviceTensorFp32& conv_weight,
    const DeviceTensorFp32& conv_bias,
    DeviceTensorFp32* conv_state,
    DeviceTensorBf16* conv_output,
    cudaStream_t stream) {
  if (!projected.valid() ||
      !conv_weight.valid() ||
      !conv_bias.valid() ||
      conv_state == nullptr ||
      !conv_state->valid() ||
      conv_output == nullptr ||
      !conv_output->valid() ||
      projected.shape().size() != 2 ||
      conv_output->shape().size() != 2 ||
      conv_bias.shape().size() != 1 ||
      conv_weight.shape().size() != 1 ||
      conv_output->shape()[0] != projected.shape()[0] ||
      conv_output->shape()[1] != conv_dim ||
      conv_weight.numel() != conv_dim * conv_kernel_size ||
      conv_bias.numel() != conv_dim ||
      conv_state_offset_elems + (conv_dim * conv_kernel_size) > conv_state->numel()) {
    return false;
  }

  const std::size_t projection_size = projected.shape()[1];
  constexpr int kBlockSize = 256;
  const int grid_size = static_cast<int>((conv_dim + kBlockSize - 1u) / kBlockSize);
  MambaConv1dSiluUpdateKernel<<<grid_size, kBlockSize, 0, stream>>>(
      projected.data(),
      projected.shape()[0],
      projection_size,
      intermediate_size,
      conv_dim,
      conv_kernel_size,
      conv_weight.data(),
      conv_bias.data(),
      conv_state->data() + conv_state_offset_elems,
      conv_output->data());
  return CheckCuda(cudaGetLastError());
}

bool MambaSsmUpdateFp32(
    const DeviceTensorFp32& projected,
    const DeviceTensorFp32& conv_output,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t num_heads,
    std::size_t head_dim,
    std::size_t state_size,
    std::size_t n_groups,
    float time_step_min,
    std::size_t ssm_state_offset_elems,
    const DeviceTensorFp32& a_log,
    const DeviceTensorFp32& d,
    const DeviceTensorFp32& dt_bias,
    DeviceTensorFp32* ssm_state,
    DeviceTensorFp32* y_output,
    cudaStream_t stream) {
  if (!projected.valid() ||
      !conv_output.valid() ||
      !a_log.valid() ||
      !d.valid() ||
      !dt_bias.valid() ||
      ssm_state == nullptr ||
      !ssm_state->valid() ||
      y_output == nullptr ||
      !y_output->valid() ||
      projected.shape().size() != 2 ||
      conv_output.shape().size() != 2 ||
      y_output->shape().size() != 2 ||
      projected.shape()[0] != conv_output.shape()[0] ||
      projected.shape()[0] != y_output->shape()[0] ||
      conv_output.shape()[1] != conv_dim ||
      y_output->shape()[1] != intermediate_size ||
      a_log.numel() != num_heads ||
      d.numel() != num_heads ||
      dt_bias.numel() != num_heads ||
      time_step_min <= 0.0f ||
      ssm_state_offset_elems + (intermediate_size * state_size) > ssm_state->numel()) {
    return false;
  }

  constexpr int kBlockSize = 256;
  const int grid_size = static_cast<int>((intermediate_size + kBlockSize - 1u) / kBlockSize);
  MambaSsmUpdateKernel<<<grid_size, kBlockSize, 0, stream>>>(
      projected.data(),
      conv_output.data(),
      projected.shape()[0],
      projected.shape()[1],
      intermediate_size,
      conv_dim,
      num_heads,
      head_dim,
      state_size,
      n_groups,
      time_step_min,
      a_log.data(),
      d.data(),
      dt_bias.data(),
      ssm_state->data() + ssm_state_offset_elems,
      y_output->data());
  return CheckCuda(cudaGetLastError());
}

bool MambaSsmUpdateBf16(
    const DeviceTensorBf16& projected,
    const DeviceTensorBf16& conv_output,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t num_heads,
    std::size_t head_dim,
    std::size_t state_size,
    std::size_t n_groups,
    float time_step_min,
    std::size_t ssm_state_offset_elems,
    const DeviceTensorFp32& a_log,
    const DeviceTensorFp32& d,
    const DeviceTensorFp32& dt_bias,
    DeviceTensorFp32* ssm_state,
    DeviceTensorBf16* y_output,
    cudaStream_t stream) {
  if (!projected.valid() ||
      !conv_output.valid() ||
      !a_log.valid() ||
      !d.valid() ||
      !dt_bias.valid() ||
      ssm_state == nullptr ||
      !ssm_state->valid() ||
      y_output == nullptr ||
      !y_output->valid() ||
      projected.shape().size() != 2 ||
      conv_output.shape().size() != 2 ||
      y_output->shape().size() != 2 ||
      projected.shape()[0] != conv_output.shape()[0] ||
      projected.shape()[0] != y_output->shape()[0] ||
      conv_output.shape()[1] != conv_dim ||
      y_output->shape()[1] != intermediate_size ||
      a_log.numel() != num_heads ||
      d.numel() != num_heads ||
      dt_bias.numel() != num_heads ||
      time_step_min <= 0.0f ||
      ssm_state_offset_elems + (intermediate_size * state_size) > ssm_state->numel()) {
    return false;
  }

  constexpr int kBlockSize = 256;
  const int grid_size = static_cast<int>((intermediate_size + kBlockSize - 1u) / kBlockSize);
  MambaSsmUpdateKernel<<<grid_size, kBlockSize, 0, stream>>>(
      projected.data(),
      conv_output.data(),
      projected.shape()[0],
      projected.shape()[1],
      intermediate_size,
      conv_dim,
      num_heads,
      head_dim,
      state_size,
      n_groups,
      time_step_min,
      a_log.data(),
      d.data(),
      dt_bias.data(),
      ssm_state->data() + ssm_state_offset_elems,
      y_output->data());
  return CheckCuda(cudaGetLastError());
}

bool MambaSelectiveStateUpdateDecodeFp32(
    const DeviceTensorFp32& projected,
    const DeviceTensorFp32& conv_output,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t num_heads,
    std::size_t head_dim,
    std::size_t state_size,
    std::size_t n_groups,
    float time_step_min,
    std::size_t ssm_state_offset_elems,
    const DeviceTensorFp32& a_log,
    const DeviceTensorFp32& d,
    const DeviceTensorFp32& dt_bias,
    DeviceTensorFp32* ssm_state,
    DeviceTensorFp32* gated_output,
    cudaStream_t stream) {
  if (!projected.valid() ||
      !conv_output.valid() ||
      !a_log.valid() ||
      !d.valid() ||
      !dt_bias.valid() ||
      ssm_state == nullptr ||
      !ssm_state->valid() ||
      gated_output == nullptr ||
      !gated_output->valid() ||
      projected.shape().size() != 2 ||
      projected.shape()[0] != 1 ||
      conv_output.shape().size() != 2 ||
      conv_output.shape()[0] != 1 ||
      conv_output.shape()[1] != conv_dim ||
      gated_output->shape().size() != 2 ||
      gated_output->shape()[0] != 1 ||
      gated_output->shape()[1] != intermediate_size ||
      num_heads == 0 ||
      n_groups == 0 ||
      num_heads % n_groups != 0 ||
      intermediate_size != num_heads * head_dim ||
      a_log.numel() != num_heads ||
      d.numel() != num_heads ||
      dt_bias.numel() != num_heads ||
      time_step_min <= 0.0f ||
      ssm_state_offset_elems + (intermediate_size * state_size) > ssm_state->numel()) {
    return false;
  }

  return LaunchMambaSelectiveStateUpdateDecode(
      projected.data(),
      conv_output.data(),
      intermediate_size,
      conv_dim,
      num_heads,
      head_dim,
      state_size,
      n_groups,
      time_step_min,
      a_log.data(),
      d.data(),
      dt_bias.data(),
      ssm_state->data() + ssm_state_offset_elems,
      gated_output->data(),
      stream);
}

bool MambaSelectiveStateUpdateDecodeBf16(
    const DeviceTensorBf16& projected,
    const DeviceTensorBf16& conv_output,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t num_heads,
    std::size_t head_dim,
    std::size_t state_size,
    std::size_t n_groups,
    float time_step_min,
    std::size_t ssm_state_offset_elems,
    const DeviceTensorFp32& a_log,
    const DeviceTensorFp32& d,
    const DeviceTensorFp32& dt_bias,
    DeviceTensorFp32* ssm_state,
    DeviceTensorBf16* gated_output,
    cudaStream_t stream) {
  if (!projected.valid() ||
      !conv_output.valid() ||
      !a_log.valid() ||
      !d.valid() ||
      !dt_bias.valid() ||
      ssm_state == nullptr ||
      !ssm_state->valid() ||
      gated_output == nullptr ||
      !gated_output->valid() ||
      projected.shape().size() != 2 ||
      projected.shape()[0] != 1 ||
      conv_output.shape().size() != 2 ||
      conv_output.shape()[0] != 1 ||
      conv_output.shape()[1] != conv_dim ||
      gated_output->shape().size() != 2 ||
      gated_output->shape()[0] != 1 ||
      gated_output->shape()[1] != intermediate_size ||
      num_heads == 0 ||
      n_groups == 0 ||
      num_heads % n_groups != 0 ||
      intermediate_size != num_heads * head_dim ||
      a_log.numel() != num_heads ||
      d.numel() != num_heads ||
      dt_bias.numel() != num_heads ||
      time_step_min <= 0.0f ||
      ssm_state_offset_elems + (intermediate_size * state_size) > ssm_state->numel()) {
    return false;
  }

  return LaunchMambaSelectiveStateUpdateDecode(
      projected.data(),
      conv_output.data(),
      intermediate_size,
      conv_dim,
      num_heads,
      head_dim,
      state_size,
      n_groups,
      time_step_min,
      a_log.data(),
      d.data(),
      dt_bias.data(),
      ssm_state->data() + ssm_state_offset_elems,
      gated_output->data(),
      stream);
}

bool MambaDecodeStepFusedFp32(
    const DeviceTensorFp32& projected,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t num_heads,
    std::size_t head_dim,
    std::size_t state_size,
    std::size_t n_groups,
    std::size_t conv_kernel_size,
    float time_step_min,
    float mixer_rms_epsilon,
    std::size_t conv_state_offset_elems,
    std::size_t ssm_state_offset_elems,
    const DeviceTensorFp32& conv_weight,
    const DeviceTensorFp32& conv_bias,
    const DeviceTensorFp32& a_log,
    const DeviceTensorFp32& d,
    const DeviceTensorFp32& dt_bias,
    const DeviceTensorFp32& mixer_norm_weight,
    DeviceTensorFp32* conv_state,
    DeviceTensorFp32* ssm_state,
    DeviceTensorFp32* output,
    cudaStream_t stream) {
  if (!projected.valid() ||
      !conv_weight.valid() ||
      !conv_bias.valid() ||
      !a_log.valid() ||
      !d.valid() ||
      !dt_bias.valid() ||
      !mixer_norm_weight.valid() ||
      conv_state == nullptr ||
      !conv_state->valid() ||
      ssm_state == nullptr ||
      !ssm_state->valid() ||
      output == nullptr ||
      !output->valid() ||
      projected.shape().size() != 2 ||
      projected.shape()[0] != 1 ||
      output->shape().size() != 2 ||
      output->shape()[0] != 1 ||
      output->shape()[1] != intermediate_size ||
      conv_weight.shape().size() != 1 ||
      conv_bias.shape().size() != 1 ||
      a_log.shape().size() != 1 ||
      d.shape().size() != 1 ||
      dt_bias.shape().size() != 1 ||
      mixer_norm_weight.shape().size() != 1 ||
      conv_weight.numel() != conv_dim * conv_kernel_size ||
      conv_bias.numel() != conv_dim ||
      a_log.numel() != num_heads ||
      d.numel() != num_heads ||
      dt_bias.numel() != num_heads ||
      mixer_norm_weight.numel() != intermediate_size ||
      n_groups == 0 ||
      intermediate_size % n_groups != 0 ||
      num_heads % n_groups != 0 ||
      conv_kernel_size == 0 ||
      time_step_min <= 0.0f ||
      mixer_rms_epsilon <= 0.0f ||
      conv_state_offset_elems + (conv_dim * conv_kernel_size) > conv_state->numel() ||
      ssm_state_offset_elems + (intermediate_size * state_size) > ssm_state->numel()) {
    return false;
  }

  const std::size_t projection_size = projected.shape()[1];
  const std::size_t shared_floats =
      (2 * state_size) + (intermediate_size / n_groups) + kThreadsPerBlock;
  MambaDecodeStepFusedKernel<<<static_cast<unsigned int>(n_groups),
                               kThreadsPerBlock,
                               shared_floats * sizeof(float),
                               stream>>>(
      projected.data(),
      projection_size,
      intermediate_size,
      conv_dim,
      num_heads,
      head_dim,
      state_size,
      n_groups,
      conv_kernel_size,
      time_step_min,
      mixer_rms_epsilon,
      conv_state_offset_elems,
      ssm_state_offset_elems,
      conv_weight.data(),
      conv_bias.data(),
      a_log.data(),
      d.data(),
      dt_bias.data(),
      mixer_norm_weight.data(),
      conv_state->data(),
      ssm_state->data(),
      output->data());
  return CheckCuda(cudaGetLastError());
}

bool MambaDecodeStepFusedBf16(
    const DeviceTensorBf16& projected,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t num_heads,
    std::size_t head_dim,
    std::size_t state_size,
    std::size_t n_groups,
    std::size_t conv_kernel_size,
    float time_step_min,
    float mixer_rms_epsilon,
    std::size_t conv_state_offset_elems,
    std::size_t ssm_state_offset_elems,
    const DeviceTensorFp32& conv_weight,
    const DeviceTensorFp32& conv_bias,
    const DeviceTensorFp32& a_log,
    const DeviceTensorFp32& d,
    const DeviceTensorFp32& dt_bias,
    const DeviceTensorFp32& mixer_norm_weight,
    DeviceTensorFp32* conv_state,
    DeviceTensorFp32* ssm_state,
    DeviceTensorBf16* output,
    cudaStream_t stream) {
  if (!projected.valid() ||
      !conv_weight.valid() ||
      !conv_bias.valid() ||
      !a_log.valid() ||
      !d.valid() ||
      !dt_bias.valid() ||
      !mixer_norm_weight.valid() ||
      conv_state == nullptr ||
      !conv_state->valid() ||
      ssm_state == nullptr ||
      !ssm_state->valid() ||
      output == nullptr ||
      !output->valid() ||
      projected.shape().size() != 2 ||
      projected.shape()[0] != 1 ||
      output->shape().size() != 2 ||
      output->shape()[0] != 1 ||
      output->shape()[1] != intermediate_size ||
      conv_weight.shape().size() != 1 ||
      conv_bias.shape().size() != 1 ||
      a_log.shape().size() != 1 ||
      d.shape().size() != 1 ||
      dt_bias.shape().size() != 1 ||
      mixer_norm_weight.shape().size() != 1 ||
      conv_weight.numel() != conv_dim * conv_kernel_size ||
      conv_bias.numel() != conv_dim ||
      a_log.numel() != num_heads ||
      d.numel() != num_heads ||
      dt_bias.numel() != num_heads ||
      mixer_norm_weight.numel() != intermediate_size ||
      n_groups == 0 ||
      intermediate_size % n_groups != 0 ||
      num_heads % n_groups != 0 ||
      conv_kernel_size == 0 ||
      time_step_min <= 0.0f ||
      mixer_rms_epsilon <= 0.0f ||
      conv_state_offset_elems + (conv_dim * conv_kernel_size) > conv_state->numel() ||
      ssm_state_offset_elems + (intermediate_size * state_size) > ssm_state->numel()) {
    return false;
  }

  const std::size_t projection_size = projected.shape()[1];
  const std::size_t shared_floats =
      (2 * state_size) + (intermediate_size / n_groups) + kThreadsPerBlock;
  MambaDecodeStepFusedKernel<<<static_cast<unsigned int>(n_groups),
                               kThreadsPerBlock,
                               shared_floats * sizeof(float),
                               stream>>>(
      projected.data(),
      projection_size,
      intermediate_size,
      conv_dim,
      num_heads,
      head_dim,
      state_size,
      n_groups,
      conv_kernel_size,
      time_step_min,
      mixer_rms_epsilon,
      conv_state_offset_elems,
      ssm_state_offset_elems,
      conv_weight.data(),
      conv_bias.data(),
      a_log.data(),
      d.data(),
      dt_bias.data(),
      mixer_norm_weight.data(),
      conv_state->data(),
      ssm_state->data(),
      output->data());
  return CheckCuda(cudaGetLastError());
}

bool GroupedRmsNormFp32(
    const DeviceTensorFp32& input,
    const DeviceTensorFp32& mixer_norm_weight,
    std::size_t n_groups,
    float epsilon,
    DeviceTensorFp32* output,
    cudaStream_t stream) {
  if (!input.valid() ||
      !mixer_norm_weight.valid() ||
      output == nullptr ||
      !output->valid() ||
      input.shape().size() != 2 ||
      output->shape() != input.shape() ||
      mixer_norm_weight.shape().size() != 1 ||
      mixer_norm_weight.shape()[0] != input.shape()[1] ||
      n_groups == 0 ||
      input.shape()[1] % n_groups != 0 ||
      epsilon <= 0.0f) {
    return false;
  }

  const std::size_t rows = input.shape()[0];
  const std::size_t intermediate_size = input.shape()[1];
  const std::size_t mixer_group_size = intermediate_size / n_groups;
  const dim3 grid(static_cast<unsigned int>(rows), static_cast<unsigned int>(n_groups));
  const dim3 block(kThreadsPerBlock);
  GroupedRmsNormKernel<<<grid, block, sizeof(float) * kThreadsPerBlock, stream>>>(
      input.data(),
      mixer_norm_weight.data(),
      rows,
      intermediate_size,
      mixer_group_size,
      epsilon,
      output->data());
  return CheckCuda(cudaGetLastError());
}

bool GroupedRmsNormBf16(
    const DeviceTensorBf16& input,
    const DeviceTensorFp32& mixer_norm_weight,
    std::size_t n_groups,
    float epsilon,
    DeviceTensorBf16* output,
    cudaStream_t stream) {
  if (!input.valid() ||
      !mixer_norm_weight.valid() ||
      output == nullptr ||
      !output->valid() ||
      input.shape().size() != 2 ||
      output->shape() != input.shape() ||
      mixer_norm_weight.shape().size() != 1 ||
      mixer_norm_weight.shape()[0] != input.shape()[1] ||
      n_groups == 0 ||
      input.shape()[1] % n_groups != 0 ||
      epsilon <= 0.0f) {
    return false;
  }

  const std::size_t rows = input.shape()[0];
  const std::size_t intermediate_size = input.shape()[1];
  const std::size_t mixer_group_size = intermediate_size / n_groups;
  const dim3 grid(static_cast<unsigned int>(rows), static_cast<unsigned int>(n_groups));
  const dim3 block(kThreadsPerBlock);
  GroupedRmsNormKernel<<<grid, block, sizeof(float) * kThreadsPerBlock, stream>>>(
      input.data(),
      mixer_norm_weight.data(),
      rows,
      intermediate_size,
      mixer_group_size,
      epsilon,
      output->data());
  return CheckCuda(cudaGetLastError());
}

bool GroupedRmsNormGatedFp32(
    const DeviceTensorFp32& y_output,
    const DeviceTensorFp32& projected,
    const DeviceTensorFp32& mixer_norm_weight,
    std::size_t n_groups,
    float epsilon,
    DeviceTensorFp32* output,
    cudaStream_t stream) {
  if (!y_output.valid() ||
      !projected.valid() ||
      !mixer_norm_weight.valid() ||
      output == nullptr ||
      !output->valid() ||
      y_output.shape().size() != 2 ||
      projected.shape().size() != 2 ||
      output->shape() != y_output.shape() ||
      y_output.shape()[0] != projected.shape()[0] ||
      mixer_norm_weight.shape().size() != 1 ||
      mixer_norm_weight.shape()[0] != y_output.shape()[1] ||
      n_groups == 0 ||
      y_output.shape()[1] % n_groups != 0 ||
      epsilon <= 0.0f) {
    return false;
  }

  const std::size_t rows = y_output.shape()[0];
  const std::size_t intermediate_size = y_output.shape()[1];
  const std::size_t mixer_group_size = intermediate_size / n_groups;
  const dim3 grid(static_cast<unsigned int>(rows), static_cast<unsigned int>(n_groups));
  const dim3 block(kThreadsPerBlock);
  GroupedRmsNormGatedKernel<<<grid, block, sizeof(float) * kThreadsPerBlock, stream>>>(
      y_output.data(),
      projected.data(),
      mixer_norm_weight.data(),
      rows,
      intermediate_size,
      mixer_group_size,
      projected.shape()[1],
      epsilon,
      output->data());
  return CheckCuda(cudaGetLastError());
}

bool GroupedRmsNormGatedBf16(
    const DeviceTensorBf16& y_output,
    const DeviceTensorBf16& projected,
    const DeviceTensorFp32& mixer_norm_weight,
    std::size_t n_groups,
    float epsilon,
    DeviceTensorBf16* output,
    cudaStream_t stream) {
  if (!y_output.valid() ||
      !projected.valid() ||
      !mixer_norm_weight.valid() ||
      output == nullptr ||
      !output->valid() ||
      y_output.shape().size() != 2 ||
      projected.shape().size() != 2 ||
      output->shape() != y_output.shape() ||
      y_output.shape()[0] != projected.shape()[0] ||
      mixer_norm_weight.shape().size() != 1 ||
      mixer_norm_weight.shape()[0] != y_output.shape()[1] ||
      n_groups == 0 ||
      y_output.shape()[1] % n_groups != 0 ||
      epsilon <= 0.0f) {
    return false;
  }

  const std::size_t rows = y_output.shape()[0];
  const std::size_t intermediate_size = y_output.shape()[1];
  const std::size_t mixer_group_size = intermediate_size / n_groups;
  const dim3 grid(static_cast<unsigned int>(rows), static_cast<unsigned int>(n_groups));
  const dim3 block(kThreadsPerBlock);
  GroupedRmsNormGatedKernel<<<grid, block, sizeof(float) * kThreadsPerBlock, stream>>>(
      y_output.data(),
      projected.data(),
      mixer_norm_weight.data(),
      rows,
      intermediate_size,
      mixer_group_size,
      projected.shape()[1],
      epsilon,
      output->data());
  return CheckCuda(cudaGetLastError());
}

}  // namespace nemotron
