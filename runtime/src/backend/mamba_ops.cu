#include "nemotron/mamba_ops.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>

namespace nemotron {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kDecodeConvThreads = 128;
constexpr int kDecodeSsmThreads = 64;
constexpr int kMambaChunkSize = 128;
constexpr int kMambaFixedHeads = 128;
constexpr int kMambaFixedHeadDim = 64;
constexpr int kMambaFixedStateSize = 128;
constexpr int kMambaFixedGroups = 8;
constexpr int kMambaFixedIntermediate = kMambaFixedHeads * kMambaFixedHeadDim;
constexpr int kChunkCumsumTileHeads = 8;
constexpr int kChunkStateNTile = 64;
constexpr int kChunkScanQTile = 64;
constexpr int kStatePassingTile = 512;

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

__device__ __forceinline__ std::size_t ChunkHeadOffset(
    std::size_t chunk,
    std::size_t head,
    std::size_t chunk_size) {
  return ((chunk * kMambaFixedHeads) + head) * chunk_size;
}

__device__ __forceinline__ int ChunkLengthDevice(
    std::size_t token_count,
    std::size_t chunk,
    std::size_t chunk_size) {
  const std::size_t start = chunk * chunk_size;
  if (start >= token_count) {
    return 0;
  }
  const std::size_t remaining = token_count - start;
  return static_cast<int>(remaining < chunk_size ? remaining : chunk_size);
}

template <int Q, int TileHeads>
__global__ __launch_bounds__(128) void ChunkCumsumKernel(
    const __nv_bfloat16* projected,
    std::size_t token_count,
    std::size_t projection_size,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    const float* a_log,
    const float* dt_bias,
    float* dt_chunk,
    float* dA_cumsum) {
  extern __shared__ float shared_dt[];

  const int chunk = blockIdx.x;
  const int head_base = blockIdx.y * TileHeads;
  const int tid = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  const int threads = static_cast<int>(blockDim.x * blockDim.y);

  for (int linear = tid; linear < TileHeads * Q; linear += threads) {
    const int head_local = linear / Q;
    const int q = linear % Q;
    const int head = head_base + head_local;
    float dt = 0.0f;
    if (head < kMambaFixedHeads) {
      const std::size_t token = static_cast<std::size_t>(chunk) * Q + q;
      if (token < token_count) {
        const std::size_t dt_index =
            token * projection_size + intermediate_size + conv_dim + head;
        dt = SoftplusDevice(__bfloat162float(projected[dt_index]) + dt_bias[head]);
      }
    }
    shared_dt[linear] = dt;
  }
  __syncthreads();

  if (tid >= TileHeads) {
    return;
  }

  const int head = head_base + tid;
  if (head >= kMambaFixedHeads) {
    return;
  }

  const float a = -expf(a_log[head]);
  const std::size_t base = ChunkHeadOffset(chunk, head, Q);
  float cumsum = 0.0f;
  const int row_offset = tid * Q;
  for (int q = 0; q < Q; ++q) {
    const float dt = shared_dt[row_offset + q];
    cumsum += dt * a;
    dt_chunk[base + q] = dt;
    dA_cumsum[base + q] = cumsum;
  }
}

template <int Q, int P, int N, int G, int NTile>
__global__ __launch_bounds__(128) void ChunkStateKernel(
    const __nv_bfloat16* conv_output,
    std::size_t token_count,
    std::size_t conv_dim,
    std::size_t intermediate_size,
    const float* dt_chunk,
    const float* dA_cumsum,
    float* state_scratch) {
  __shared__ float shared_scale[Q];

  const int blocks_per_head = N / NTile;
  const int head = blockIdx.x / blocks_per_head;
  const int n_tile = (blockIdx.x % blocks_per_head) * NTile;
  const int chunk = blockIdx.y;
  const int tid =
      static_cast<int>((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);
  const int threads = static_cast<int>(blockDim.x * blockDim.y * blockDim.z);
  const int chunk_length = ChunkLengthDevice(token_count, chunk, Q);
  const int group = head / (kMambaFixedHeads / G);
  const std::size_t dt_base = ChunkHeadOffset(chunk, head, Q);
  const float lambda_last = chunk_length > 0 ? dA_cumsum[dt_base + chunk_length - 1] : 0.0f;

  for (int q = tid; q < Q; q += threads) {
    float scale = 0.0f;
    if (q < chunk_length) {
      scale = expf(lambda_last - dA_cumsum[dt_base + q]) * dt_chunk[dt_base + q];
    }
    shared_scale[q] = scale;
  }
  __syncthreads();

  for (int linear = tid; linear < P * NTile; linear += threads) {
    const int p = linear / NTile;
    const int n = n_tile + (linear % NTile);
    float accum = 0.0f;
    for (int q = 0; q < chunk_length; ++q) {
      const std::size_t token = static_cast<std::size_t>(chunk) * Q + q;
      const float x = __bfloat162float(conv_output[token * conv_dim + head * P + p]);
      const float b =
          __bfloat162float(conv_output[token * conv_dim + intermediate_size + group * N + n]);
      accum += shared_scale[q] * b * x;
    }
    state_scratch[(((static_cast<std::size_t>(chunk) * kMambaFixedHeads + head) * P + p) * N) + n] =
        accum;
  }
}

template <int Q, int H, int P, int N, int Tile>
__global__ __launch_bounds__(128) void StatePassingKernel(
    std::size_t token_count,
    std::size_t chunk_count,
    const float* dA_cumsum,
    float* state_scratch,
    float* ssm_state) {
  constexpr int kThreads = 128;
  constexpr int kValuesPerThread = Tile / kThreads;
  static_assert(Tile % kThreads == 0, "State passing tile must map evenly onto the block.");

  const int slices_per_head = (P * N) / Tile;
  const int head = blockIdx.x / slices_per_head;
  const int slice = blockIdx.x % slices_per_head;
  const int tid = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  const std::size_t head_state_base = static_cast<std::size_t>(head) * P * N;
  const std::size_t slice_base = static_cast<std::size_t>(slice) * Tile;

  float current[kValuesPerThread];
#pragma unroll
  for (int i = 0; i < kValuesPerThread; ++i) {
    const std::size_t offset = slice_base + tid + i * kThreads;
    current[i] = ssm_state[head_state_base + offset];
  }

  for (std::size_t chunk = 0; chunk < chunk_count; ++chunk) {
    const int chunk_length = ChunkLengthDevice(token_count, chunk, Q);
    const float decay =
        chunk_length > 0 ? expf(dA_cumsum[ChunkHeadOffset(chunk, head, Q) + chunk_length - 1])
                         : 1.0f;
    const std::size_t scratch_base =
        ((chunk * H + head) * P * N) + slice_base;
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i) {
      const std::size_t offset = tid + i * kThreads;
      const float delta = state_scratch[scratch_base + offset];
      state_scratch[scratch_base + offset] = current[i];
      current[i] = decay * current[i] + delta;
    }
  }

#pragma unroll
  for (int i = 0; i < kValuesPerThread; ++i) {
    const std::size_t offset = slice_base + tid + i * kThreads;
    ssm_state[head_state_base + offset] = current[i];
  }
}

template <int Q, int G, int N, int TileQ>
__global__ __launch_bounds__(128) void BmmChunkKernel(
    const __nv_bfloat16* conv_output,
    std::size_t token_count,
    std::size_t conv_dim,
    std::size_t intermediate_size,
    float* cb_chunk) {
  const int tiles_per_group = (Q / TileQ) * (Q / TileQ);
  const int group = blockIdx.x / tiles_per_group;
  const int local_tile = blockIdx.x % tiles_per_group;
  const int qi_start = (local_tile / (Q / TileQ)) * TileQ;
  const int qj_start = (local_tile % (Q / TileQ)) * TileQ;
  const int chunk = blockIdx.y;
  const int tid =
      static_cast<int>((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);
  const int threads = static_cast<int>(blockDim.x * blockDim.y * blockDim.z);
  const int chunk_length = ChunkLengthDevice(token_count, chunk, Q);

  for (int linear = tid; linear < TileQ * TileQ; linear += threads) {
    const int qi = qi_start + (linear / TileQ);
    const int qj = qj_start + (linear % TileQ);
    float accum = 0.0f;
    if (qi < chunk_length && qj < chunk_length) {
      const std::size_t token_i = static_cast<std::size_t>(chunk) * Q + qi;
      const std::size_t token_j = static_cast<std::size_t>(chunk) * Q + qj;
      const std::size_t c_base =
          token_i * conv_dim + intermediate_size + G * N + group * N;
      const std::size_t b_base = token_j * conv_dim + intermediate_size + group * N;
      for (int n = 0; n < N; ++n) {
        accum += __bfloat162float(conv_output[c_base + n]) *
                 __bfloat162float(conv_output[b_base + n]);
      }
    }
    cb_chunk[(((static_cast<std::size_t>(chunk) * G + group) * Q + qi) * Q) + qj] = accum;
  }
}

template <int Q, int H, int P, int N, int G, int TileQ>
__global__ __launch_bounds__(128) void ChunkScanKernel(
    const __nv_bfloat16* conv_output,
    std::size_t token_count,
    std::size_t conv_dim,
    std::size_t intermediate_size,
    const float* d_values,
    const float* dt_chunk,
    const float* dA_cumsum,
    const float* boundary_state,
    const float* cb_chunk,
    __nv_bfloat16* y_output) {
  extern __shared__ unsigned char shared_bytes[];
  __nv_bfloat16* shared_x = reinterpret_cast<__nv_bfloat16*>(shared_bytes);
  float* shared_dt = reinterpret_cast<float*>(shared_x + (Q * P));
  float* shared_dA = shared_dt + Q;

  const int head = blockIdx.x / (Q / TileQ);
  const int q_tile = blockIdx.x % (Q / TileQ);
  const int q_start = q_tile * TileQ;
  const int chunk = blockIdx.y;
  const int group = head / (H / G);
  const int tid = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
  const int threads = static_cast<int>(blockDim.x * blockDim.y);
  const int chunk_length = ChunkLengthDevice(token_count, chunk, Q);
  const std::size_t chunk_head_base = ChunkHeadOffset(chunk, head, Q);

  for (int linear = tid; linear < Q * P; linear += threads) {
    const int q = linear / P;
    const int p = linear % P;
    __nv_bfloat16 value = __float2bfloat16(0.0f);
    if (q < chunk_length) {
      const std::size_t token = static_cast<std::size_t>(chunk) * Q + q;
      value = conv_output[token * conv_dim + head * P + p];
    }
    shared_x[linear] = value;
  }
  for (int q = tid; q < Q; q += threads) {
    const float dA = dA_cumsum[chunk_head_base + q];
    shared_dA[q] = dA;
    shared_dt[q] = q < chunk_length ? expf(-dA) * dt_chunk[chunk_head_base + q] : 0.0f;
  }
  __syncthreads();

  for (int linear = tid; linear < TileQ * P; linear += threads) {
    const int qi_local = linear / P;
    const int p = linear % P;
    const int qi = q_start + qi_local;
    if (qi >= chunk_length) {
      continue;
    }

    const std::size_t token = static_cast<std::size_t>(chunk) * Q + qi;
    const float scale_i = expf(shared_dA[qi]);
    const std::size_t c_base =
        token * conv_dim + intermediate_size + G * N + group * N;
    const std::size_t boundary_base =
        (((static_cast<std::size_t>(chunk) * H + head) * P + p) * N);
    float accum = 0.0f;
    for (int n = 0; n < N; ++n) {
      accum += __bfloat162float(conv_output[c_base + n]) * boundary_state[boundary_base + n];
    }
    accum *= scale_i;

    const std::size_t cb_row_base =
        (((static_cast<std::size_t>(chunk) * G + group) * Q + qi) * Q);
    for (int qj = 0; qj <= qi; ++qj) {
      accum += scale_i * shared_dt[qj] * cb_chunk[cb_row_base + qj] *
               __bfloat162float(shared_x[qj * P + p]);
    }

    accum += d_values[head] * __bfloat162float(shared_x[qi * P + p]);
    y_output[token * intermediate_size + head * P + p] = __float2bfloat16(accum);
  }
}

template <int kWidth, typename ProjectedT, typename ConvStateT, typename OutputT>
__global__ __launch_bounds__(kDecodeConvThreads) void MambaDecodeCausalConv1dUpdateKernel(
    const ProjectedT* projected,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    const float* conv_weight,
    const float* conv_bias,
    ConvStateT* conv_state,
    OutputT* conv_output) {
  const std::size_t channel = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (channel >= conv_dim) {
    return;
  }

  const float x_value = LoadMambaValue(projected, intermediate_size + channel);
  ConvStateT* state_row = conv_state + channel * kWidth;
  const float* weight_row = conv_weight + channel * kWidth;

  float state_values[kWidth];
#pragma unroll
  for (int tap = 0; tap < kWidth; ++tap) {
    state_values[tap] = LoadMambaValue(state_row, tap);
  }
#pragma unroll
  for (int tap = 0; tap + 1 < kWidth; ++tap) {
    state_values[tap] = state_values[tap + 1];
    StoreMambaValue(state_row, tap, state_values[tap]);
  }
  state_values[kWidth - 1] = x_value;
  StoreMambaValue(state_row, kWidth - 1, x_value);

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
  (void)time_step_min;
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
    shared_dt = SoftplusDevice(dt_base);
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

template <typename ProjectedT, typename ConvStateT, typename OutputT>
__global__ void MambaConv1dSiluUpdateKernel(
    const ProjectedT* projected,
    std::size_t token_count,
    std::size_t projection_size,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t conv_kernel_size,
    const float* conv_weight,
    const float* conv_bias,
    ConvStateT* conv_state,
    OutputT* conv_output) {
  const std::size_t channel = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (channel >= conv_dim) {
    return;
  }

  ConvStateT* state_row = conv_state + channel * conv_kernel_size;
  const float* weight_row = conv_weight + channel * conv_kernel_size;
  for (std::size_t token = 0; token < token_count; ++token) {
    const float conv_input =
        LoadMambaValue(projected, token * projection_size + intermediate_size + channel);
    if (conv_kernel_size > 1) {
      for (std::size_t tap = 0; tap + 1 < conv_kernel_size; ++tap) {
        StoreMambaValue(state_row, tap, LoadMambaValue(state_row, tap + 1));
      }
    }
    StoreMambaValue(state_row, conv_kernel_size - 1, conv_input);

    float accum = conv_bias[channel];
    for (std::size_t tap = 0; tap < conv_kernel_size; ++tap) {
      accum += LoadMambaValue(state_row, tap) * weight_row[tap];
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
  (void)time_step_min;
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
    const float dt = SoftplusDevice(dt_base);
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

template <typename ProjectedT, typename ConvStateT, typename OutputT>
bool LaunchMambaDecodeCausalConv1dUpdate(
    const ProjectedT* projected,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t conv_kernel_size,
    const float* conv_weight,
    const float* conv_bias,
    ConvStateT* conv_state,
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

template <typename ProjectedT, typename ConvStateT, typename OutputT>
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
    ConvStateT* conv_state,
    float* ssm_state,
    OutputT* output) {
  (void)time_step_min;
  const std::size_t group = static_cast<std::size_t>(blockIdx.x);
  if (group >= n_groups) {
    return;
  }

  const std::size_t group_hidden_size = intermediate_size / n_groups;
  const std::size_t hidden_begin = group * group_hidden_size;
  const std::size_t b_begin = intermediate_size + group * state_size;
  const std::size_t c_begin = intermediate_size + (n_groups * state_size) + group * state_size;
  ConvStateT* conv_state_base = conv_state + conv_state_offset_elems;
  float* ssm_state_base = ssm_state + ssm_state_offset_elems;

  extern __shared__ float shared_storage[];
  float* shared_b = shared_storage;
  float* shared_c = shared_b + state_size;
  float* shared_gated = shared_c + state_size;
  float* shared_sum = shared_gated + group_hidden_size;

  for (std::size_t local_state = threadIdx.x; local_state < state_size; local_state += blockDim.x) {
    const std::size_t b_channel = b_begin + local_state;
    ConvStateT* b_state_row = conv_state_base + b_channel * conv_kernel_size;
    const float* b_weight_row = conv_weight + b_channel * conv_kernel_size;
    const float b_input = LoadMambaValue(projected, intermediate_size + b_channel);
    if (conv_kernel_size > 1) {
      for (std::size_t tap = 0; tap + 1 < conv_kernel_size; ++tap) {
        StoreMambaValue(b_state_row, tap, LoadMambaValue(b_state_row, tap + 1));
      }
    }
    StoreMambaValue(b_state_row, conv_kernel_size - 1, b_input);

    float b_accum = conv_bias[b_channel];
    for (std::size_t tap = 0; tap < conv_kernel_size; ++tap) {
      b_accum += LoadMambaValue(b_state_row, tap) * b_weight_row[tap];
    }
    shared_b[local_state] = SiLUDevice(b_accum);

    const std::size_t c_channel = c_begin + local_state;
    ConvStateT* c_state_row = conv_state_base + c_channel * conv_kernel_size;
    const float* c_weight_row = conv_weight + c_channel * conv_kernel_size;
    const float c_input = LoadMambaValue(projected, intermediate_size + c_channel);
    if (conv_kernel_size > 1) {
      for (std::size_t tap = 0; tap + 1 < conv_kernel_size; ++tap) {
        StoreMambaValue(c_state_row, tap, LoadMambaValue(c_state_row, tap + 1));
      }
    }
    StoreMambaValue(c_state_row, conv_kernel_size - 1, c_input);

    float c_accum = conv_bias[c_channel];
    for (std::size_t tap = 0; tap < conv_kernel_size; ++tap) {
      c_accum += LoadMambaValue(c_state_row, tap) * c_weight_row[tap];
    }
    shared_c[local_state] = SiLUDevice(c_accum);
  }
  __syncthreads();

  float local_sum = 0.0f;
  for (std::size_t local_hidden = threadIdx.x;
       local_hidden < group_hidden_size;
       local_hidden += blockDim.x) {
    const std::size_t hidden_index = hidden_begin + local_hidden;
    ConvStateT* hidden_conv_state_row = conv_state_base + hidden_index * conv_kernel_size;
    const float* hidden_weight_row = conv_weight + hidden_index * conv_kernel_size;
    const float hidden_input = LoadMambaValue(projected, intermediate_size + hidden_index);
    if (conv_kernel_size > 1) {
      for (std::size_t tap = 0; tap + 1 < conv_kernel_size; ++tap) {
        StoreMambaValue(
            hidden_conv_state_row,
            tap,
            LoadMambaValue(hidden_conv_state_row, tap + 1));
      }
    }
    StoreMambaValue(hidden_conv_state_row, conv_kernel_size - 1, hidden_input);

    float hidden_accum = conv_bias[hidden_index];
    for (std::size_t tap = 0; tap < conv_kernel_size; ++tap) {
      hidden_accum += LoadMambaValue(hidden_conv_state_row, tap) * hidden_weight_row[tap];
    }
    const float hidden_value = SiLUDevice(hidden_accum);
    const std::size_t head = hidden_index / head_dim;
    const float a = -expf(a_log[head]);
    const float d_value = d[head];
    const float dt_base =
        LoadMambaValue(projected, intermediate_size + conv_dim + head) + dt_bias[head];
    const float dt = SoftplusDevice(dt_base);
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
    DeviceTensorBf16* conv_state,
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
    DeviceTensorBf16* conv_state,
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
    DeviceTensorBf16* conv_state,
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
    DeviceTensorBf16* conv_state,
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

bool MambaChunkedScanPrefillBf16(
    const DeviceTensorBf16& projected,
    const DeviceTensorBf16& conv_output,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t num_heads,
    std::size_t head_dim,
    std::size_t state_size,
    std::size_t n_groups,
    std::size_t chunk_size,
    std::size_t ssm_state_offset_elems,
    const DeviceTensorFp32& a_log,
    const DeviceTensorFp32& d,
    const DeviceTensorFp32& dt_bias,
    DeviceTensorFp32* ssm_state,
    DeviceTensorBf16* y_output,
    MambaChunkScanWorkspace* workspace,
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
      workspace == nullptr ||
      workspace->dt_chunk == nullptr ||
      workspace->dA_cumsum == nullptr ||
      workspace->state_scratch == nullptr ||
      workspace->cb_chunk == nullptr ||
      !workspace->dt_chunk->valid() ||
      !workspace->dA_cumsum->valid() ||
      !workspace->state_scratch->valid() ||
      !workspace->cb_chunk->valid() ||
      projected.shape().size() != 2 ||
      conv_output.shape().size() != 2 ||
      y_output->shape().size() != 2 ||
      projected.shape()[0] == 0 ||
      projected.shape()[0] != conv_output.shape()[0] ||
      projected.shape()[0] != y_output->shape()[0] ||
      projected.shape()[1] != intermediate_size + conv_dim + num_heads ||
      conv_output.shape()[1] != conv_dim ||
      y_output->shape()[1] != intermediate_size ||
      intermediate_size != kMambaFixedIntermediate ||
      num_heads != kMambaFixedHeads ||
      head_dim != kMambaFixedHeadDim ||
      state_size != kMambaFixedStateSize ||
      n_groups != kMambaFixedGroups ||
      chunk_size != kMambaChunkSize ||
      conv_dim != (kMambaFixedIntermediate + 2 * kMambaFixedGroups * kMambaFixedStateSize) ||
      a_log.numel() != num_heads ||
      d.numel() != num_heads ||
      dt_bias.numel() != num_heads ||
      ssm_state_offset_elems + (intermediate_size * state_size) > ssm_state->numel() ||
      workspace->chunk_size != chunk_size ||
      workspace->capacity_tokens < projected.shape()[0]) {
    return false;
  }

  const std::size_t token_count = projected.shape()[0];
  const std::size_t chunk_count = (token_count + chunk_size - 1) / chunk_size;
  const std::size_t dt_required = chunk_count * num_heads * chunk_size;
  const std::size_t state_required = chunk_count * num_heads * head_dim * state_size;
  const std::size_t cb_required = chunk_count * n_groups * chunk_size * chunk_size;
  if (workspace->dt_chunk->numel() < dt_required ||
      workspace->dA_cumsum->numel() < dt_required ||
      workspace->state_scratch->numel() < state_required ||
      workspace->cb_chunk->numel() < cb_required) {
    return false;
  }

  const dim3 chunk_cumsum_grid(
      static_cast<unsigned int>(chunk_count),
      static_cast<unsigned int>(kMambaFixedHeads / kChunkCumsumTileHeads),
      1);
  const dim3 chunk_cumsum_block(32, 4, 1);
  ChunkCumsumKernel<kMambaChunkSize, kChunkCumsumTileHeads>
      <<<chunk_cumsum_grid,
         chunk_cumsum_block,
         sizeof(float) * kChunkCumsumTileHeads * kMambaChunkSize,
         stream>>>(
          projected.data(),
          token_count,
          projected.shape()[1],
          intermediate_size,
          conv_dim,
          a_log.data(),
          dt_bias.data(),
          workspace->dt_chunk->data(),
          workspace->dA_cumsum->data());
  if (!CheckCuda(cudaGetLastError())) {
    return false;
  }

  const dim3 chunk_state_grid(
      static_cast<unsigned int>(kMambaFixedHeads * (kMambaFixedStateSize / kChunkStateNTile)),
      static_cast<unsigned int>(chunk_count),
      1);
  const dim3 chunk_state_block(32, 2, 2);
  ChunkStateKernel<
      kMambaChunkSize,
      kMambaFixedHeadDim,
      kMambaFixedStateSize,
      kMambaFixedGroups,
      kChunkStateNTile>
      <<<chunk_state_grid, chunk_state_block, sizeof(float) * kMambaChunkSize, stream>>>(
          conv_output.data(),
          token_count,
          conv_dim,
          intermediate_size,
          workspace->dt_chunk->data(),
          workspace->dA_cumsum->data(),
          workspace->state_scratch->data());
  if (!CheckCuda(cudaGetLastError())) {
    return false;
  }

  const dim3 state_passing_grid(
      static_cast<unsigned int>(
          kMambaFixedHeads * ((kMambaFixedHeadDim * kMambaFixedStateSize) / kStatePassingTile)),
      1,
      1);
  const dim3 state_passing_block(32, 4, 1);
  StatePassingKernel<
      kMambaChunkSize,
      kMambaFixedHeads,
      kMambaFixedHeadDim,
      kMambaFixedStateSize,
      kStatePassingTile>
      <<<state_passing_grid, state_passing_block, 0, stream>>>(
          token_count,
          chunk_count,
          workspace->dA_cumsum->data(),
          workspace->state_scratch->data(),
          ssm_state->data() + ssm_state_offset_elems);
  if (!CheckCuda(cudaGetLastError())) {
    return false;
  }

  const dim3 bmm_chunk_grid(
      static_cast<unsigned int>(
          kMambaFixedGroups * (kMambaChunkSize / kChunkScanQTile) * (kMambaChunkSize / kChunkScanQTile)),
      static_cast<unsigned int>(chunk_count),
      1);
  const dim3 bmm_chunk_block(32, 2, 2);
  BmmChunkKernel<
      kMambaChunkSize,
      kMambaFixedGroups,
      kMambaFixedStateSize,
      kChunkScanQTile>
      <<<bmm_chunk_grid, bmm_chunk_block, 0, stream>>>(
          conv_output.data(),
          token_count,
          conv_dim,
          intermediate_size,
          workspace->cb_chunk->data());
  if (!CheckCuda(cudaGetLastError())) {
    return false;
  }

  const dim3 chunk_scan_grid(
      static_cast<unsigned int>(kMambaFixedHeads * (kMambaChunkSize / kChunkScanQTile)),
      static_cast<unsigned int>(chunk_count),
      1);
  const dim3 chunk_scan_block(32, 4, 1);
  const std::size_t chunk_scan_shared =
      (sizeof(__nv_bfloat16) * kMambaChunkSize * kMambaFixedHeadDim) +
      (sizeof(float) * kMambaChunkSize * 2);
  ChunkScanKernel<
      kMambaChunkSize,
      kMambaFixedHeads,
      kMambaFixedHeadDim,
      kMambaFixedStateSize,
      kMambaFixedGroups,
      kChunkScanQTile>
      <<<chunk_scan_grid, chunk_scan_block, chunk_scan_shared, stream>>>(
          conv_output.data(),
          token_count,
          conv_dim,
          intermediate_size,
          d.data(),
          workspace->dt_chunk->data(),
          workspace->dA_cumsum->data(),
          workspace->state_scratch->data(),
          workspace->cb_chunk->data(),
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
    DeviceTensorBf16* conv_state,
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
    DeviceTensorBf16* conv_state,
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
