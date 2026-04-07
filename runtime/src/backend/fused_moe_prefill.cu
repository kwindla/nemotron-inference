#include "nemotron/fused_moe_prefill.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <limits>

#include "nemotron/device_nvfp4_matrix.h"
#include "fused_decode_common.cuh"

namespace nemotron {
namespace {

namespace wmma = nvcuda::wmma;

constexpr int kGroupedTokenTile = static_cast<int>(kMoeLaunchPlanTokenTile);
constexpr int kPlannedOutputTile = 32;
constexpr int kPlannedThreadsPerBlock = 64;
constexpr int kPlannedWmmaTileM = 16;
constexpr int kPlannedWmmaTileN = 16;
constexpr int kPlannedWmmaTileK = 16;
constexpr int kPlannedWmmaWarpsPerBlock = 2;
constexpr int kContiguousSmallOutputTile = 32;
constexpr int kContiguousMediumOutputTile = 64;
constexpr int kContiguousLargeOutputTile = 128;
constexpr int kContiguousSmallThreadsPerBlock = 64;
constexpr int kContiguousMediumThreadsPerBlock = 128;
constexpr int kContiguousLargeThreadsPerBlock = 256;
constexpr float kNvfp4ActivationMaxFinite = 6.0f * 448.0f;
constexpr float kNvfp4MinTensorScale = 1.0f / 1024.0f;
constexpr std::size_t kNvfp4ScaleBlockTile = 4u;

static_assert(kGroupedTokenTile == static_cast<int>(kMoeLaunchPlanTokenTile));

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

bool ValidFusedNvfp4WeightView(const FusedNvfp4WeightView& weight) {
  return weight.packed_data != nullptr &&
         weight.block_scales_data != nullptr &&
         weight.tensor_scale_data != nullptr &&
         weight.output_rows > 0 &&
         weight.input_cols > 0;
}

__host__ __device__ fused_decode::Nvfp4WeightView MakeDeviceWeightView(
    const FusedNvfp4WeightView& weight) {
  return fused_decode::Nvfp4WeightView{
      weight.packed_data,
      weight.block_scales_data,
      weight.tensor_scale_data,
      weight.output_rows,
      weight.input_cols,
  };
}

std::size_t CeilDiv(std::size_t numerator, std::size_t denominator) {
  return (numerator + denominator - 1u) / denominator;
}

__host__ __device__ std::size_t RoundUp(std::size_t value, std::size_t alignment) {
  return alignment == 0 ? value : ((value + alignment - 1u) / alignment) * alignment;
}

__host__ __device__ std::size_t RowTile(Nvfp4ScaleLayout scale_layout) {
  switch (scale_layout) {
    case Nvfp4ScaleLayout::kSwizzled128x4:
      return 128u;
    case Nvfp4ScaleLayout::kSwizzled8x4:
      return 8u;
  }
  return 128u;
}

__host__ __device__ std::size_t ExecutionScaleOffset(
    std::size_t row,
    std::size_t block_col,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout) {
  const std::size_t num_k_tiles = padded_blocks_per_row / kNvfp4ScaleBlockTile;
  const std::size_t k_tile = block_col / kNvfp4ScaleBlockTile;
  const std::size_t inner_k = block_col & 3u;
  switch (scale_layout) {
    case Nvfp4ScaleLayout::kSwizzled128x4: {
      const std::size_t m_tile = row / 128u;
      const std::size_t outer_m = row & 31u;
      const std::size_t inner_m = (row >> 5u) & 3u;
      return ((((m_tile * num_k_tiles) + k_tile) << 9u) |
              (outer_m << 4u) |
              (inner_m << 2u) |
              inner_k);
    }
    case Nvfp4ScaleLayout::kSwizzled8x4: {
      const std::size_t m_tile = row / 8u;
      const std::size_t inner_m = row & 7u;
      return (((m_tile * num_k_tiles) + k_tile) << 5u) |
             (inner_m << 2u) |
             inner_k;
    }
  }
  return 0;
}

int GetMultiProcessorCount() {
  int device = 0;
  if (!CheckCuda(cudaGetDevice(&device))) {
    return 0;
  }
  int multiprocessor_count = 0;
  if (!CheckCuda(cudaDeviceGetAttribute(
          &multiprocessor_count, cudaDevAttrMultiProcessorCount, device))) {
    return 0;
  }
  return multiprocessor_count;
}

int SelectContiguousOutputTile(std::size_t output_rows, std::size_t input_row_count) {
  const int multiprocessor_count = GetMultiProcessorCount();
  if (multiprocessor_count <= 0) {
    return kContiguousMediumOutputTile;
  }
  const std::size_t input_row_tiles =
      CeilDiv(input_row_count, static_cast<std::size_t>(kPlannedWmmaTileM));
  const auto meets_target = [&](int output_tile) {
    const std::size_t output_row_tiles =
        CeilDiv(output_rows, static_cast<std::size_t>(output_tile));
    return output_row_tiles * input_row_tiles >=
           static_cast<std::size_t>(multiprocessor_count);
  };
  if (meets_target(kContiguousLargeOutputTile)) {
    return kContiguousLargeOutputTile;
  }
  if (meets_target(kContiguousMediumOutputTile)) {
    return kContiguousMediumOutputTile;
  }
  return kContiguousSmallOutputTile;
}

__global__ void ZeroBufferKernel(float* data, std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  data[index] = 0.0f;
}

__global__ void ZeroBf16BufferKernel(__nv_bfloat16* data, std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  data[index] = __float2bfloat16(0.0f);
}

__global__ void GatherRowsKernel(
    const float* input,
    const int* row_indices,
    const int* active_output_rows,
    float* output,
    std::size_t output_row_capacity,
    std::size_t input_rows,
    std::size_t cols) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = output_row_capacity * cols;
  if (index >= count) {
    return;
  }

  const std::size_t row = index / cols;
  if (active_output_rows != nullptr &&
      row >= static_cast<std::size_t>(active_output_rows[0])) {
    return;
  }
  const std::size_t col = index % cols;
  const int input_row = row_indices[row];
  if (input_row < 0 || static_cast<std::size_t>(input_row) >= input_rows) {
    output[index] = 0.0f;
    return;
  }
  output[index] = input[static_cast<std::size_t>(input_row) * cols + col];
}

__global__ void QuantizeDequantizeRowsKernel(
    float* data,
    const int* active_row_count,
    std::size_t row_capacity,
    std::size_t cols) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.x);
  if (row >= row_capacity) {
    return;
  }
  if (active_row_count != nullptr &&
      row >= static_cast<std::size_t>(active_row_count[0])) {
    return;
  }
  fused_decode::QuantizeDequantizeNvfp4Row(
      data + row * cols,
      data + row * cols,
      cols);
}

__global__ void Relu2Kernel(float* data, std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  data[index] = fused_decode::Relu2(data[index]);
}

__global__ void Relu2RowsKernel(
    float* data,
    const int* active_row_count,
    std::size_t row_capacity,
    std::size_t cols) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = row_capacity * cols;
  if (index >= count) {
    return;
  }
  const std::size_t row = index / cols;
  if (active_row_count != nullptr &&
      row >= static_cast<std::size_t>(active_row_count[0])) {
    return;
  }
  data[index] = fused_decode::Relu2(data[index]);
}

__device__ float ClampNvfp4TensorScale(float value) {
  if (!isfinite(value) || value < kNvfp4MinTensorScale) {
    return kNvfp4MinTensorScale;
  }
  return value;
}

__global__ void ComputeExpertActivationScalesKernel(
    const float* input,
    const int* expert_first_token_offsets,
    std::size_t n_experts,
    std::size_t cols,
    float* output_scales) {
  const std::size_t expert_index = static_cast<std::size_t>(blockIdx.x);
  if (input == nullptr ||
      expert_first_token_offsets == nullptr ||
      output_scales == nullptr ||
      expert_index >= n_experts) {
    return;
  }

  const int row_begin = expert_first_token_offsets[expert_index];
  const int row_end = expert_first_token_offsets[expert_index + 1];
  if (row_end <= row_begin) {
    if (threadIdx.x == 0) {
      output_scales[expert_index] = 1.0f;
    }
    return;
  }

  float thread_max_abs = 0.0f;
  const std::size_t row_count = static_cast<std::size_t>(row_end - row_begin);
  const std::size_t numel = row_count * cols;
  for (std::size_t linear_index = threadIdx.x;
       linear_index < numel;
       linear_index += blockDim.x) {
    const std::size_t row = linear_index / cols;
    const std::size_t col = linear_index % cols;
    const float value = input[(static_cast<std::size_t>(row_begin) + row) * cols + col];
    const float abs_value = fabsf(value);
    if (abs_value > thread_max_abs) {
      thread_max_abs = abs_value;
    }
  }

  __shared__ float shared_max_abs[fused_decode::kThreadsPerBlock];
  shared_max_abs[threadIdx.x] = thread_max_abs;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
    if (threadIdx.x < stride &&
        shared_max_abs[threadIdx.x + stride] > shared_max_abs[threadIdx.x]) {
      shared_max_abs[threadIdx.x] = shared_max_abs[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float tensor_scale = 1.0f;
    if (shared_max_abs[0] > kNvfp4ActivationMaxFinite) {
      tensor_scale = ClampNvfp4TensorScale(shared_max_abs[0] / kNvfp4ActivationMaxFinite);
    }
    output_scales[expert_index] = tensor_scale;
  }
}

__global__ void ComputeExpertActivationScalesBf16Kernel(
    const __nv_bfloat16* input,
    const int* expert_first_token_offsets,
    std::size_t n_experts,
    std::size_t cols,
    float* output_scales) {
  const std::size_t expert_index = static_cast<std::size_t>(blockIdx.x);
  if (input == nullptr ||
      expert_first_token_offsets == nullptr ||
      output_scales == nullptr ||
      expert_index >= n_experts) {
    return;
  }

  const int row_begin = expert_first_token_offsets[expert_index];
  const int row_end = expert_first_token_offsets[expert_index + 1];
  if (row_end <= row_begin) {
    if (threadIdx.x == 0) {
      output_scales[expert_index] = 1.0f;
    }
    return;
  }

  float thread_max_abs = 0.0f;
  const std::size_t row_count = static_cast<std::size_t>(row_end - row_begin);
  const std::size_t numel = row_count * cols;
  for (std::size_t linear_index = threadIdx.x;
       linear_index < numel;
       linear_index += blockDim.x) {
    const std::size_t row = linear_index / cols;
    const std::size_t col = linear_index % cols;
    const float value = fused_decode::Relu2(__bfloat162float(
        input[(static_cast<std::size_t>(row_begin) + row) * cols + col]));
    const float abs_value = value;
    if (abs_value > thread_max_abs) {
      thread_max_abs = abs_value;
    }
  }

  __shared__ float shared_max_abs[fused_decode::kThreadsPerBlock];
  shared_max_abs[threadIdx.x] = thread_max_abs;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
    if (threadIdx.x < stride &&
        shared_max_abs[threadIdx.x + stride] > shared_max_abs[threadIdx.x]) {
      shared_max_abs[threadIdx.x] = shared_max_abs[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float tensor_scale = 1.0f;
    if (shared_max_abs[0] > kNvfp4ActivationMaxFinite) {
      tensor_scale = ClampNvfp4TensorScale(shared_max_abs[0] / kNvfp4ActivationMaxFinite);
    }
    output_scales[expert_index] = tensor_scale;
  }
}

__global__ void MaxBitsToTensorScalesKernel(
    const unsigned int* max_bits,
    std::size_t count,
    float* output_scales) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const float max_abs = __uint_as_float(max_bits[index]);
  float tensor_scale = 1.0f;
  if (max_abs > kNvfp4ActivationMaxFinite) {
    tensor_scale = ClampNvfp4TensorScale(max_abs / kNvfp4ActivationMaxFinite);
  }
  output_scales[index] = tensor_scale;
}

__global__ void Nvfp4GroupedExpertMatVecRowsKernel(
    const float* input,
    const int* expert_offsets,
    std::size_t n_experts,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  __shared__ float partial_sums[kGroupedTokenTile * fused_decode::kThreadsPerBlock];

  const std::size_t expert_index = static_cast<std::size_t>(blockIdx.x) / output_rows_per_expert;
  const std::size_t output_row = static_cast<std::size_t>(blockIdx.x) % output_rows_per_expert;
  if (expert_index >= n_experts) {
    return;
  }

  const int begin_row = expert_offsets[expert_index];
  const int end_row = expert_offsets[expert_index + 1];
  if (begin_row >= end_row) {
    return;
  }

  const fused_decode::Nvfp4WeightView weight =
      MakeDeviceWeightView(weights[expert_index]);
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t packed_row_offset = output_row * (weight.input_cols / 2);
  const std::size_t scale_row_offset = output_row * blocks_per_row;
  const float tensor_scale = *weight.tensor_scale_data;

  for (int tile_begin = begin_row; tile_begin < end_row; tile_begin += kGroupedTokenTile) {
    const int remaining_rows = end_row - tile_begin;
    const int valid_rows =
        remaining_rows < kGroupedTokenTile ? remaining_rows : kGroupedTokenTile;
    float accum[kGroupedTokenTile] = {0.0f};

    for (std::size_t pair_index = static_cast<std::size_t>(threadIdx.x);
         pair_index < pairs_per_row;
         pair_index += blockDim.x) {
      const std::size_t block = pair_index / 8u;
      const std::size_t pair_in_block = pair_index % 8u;
      const std::size_t col = block * fused_decode::kNvfp4BlockWidth + pair_in_block * 2u;
      const float block_scale =
          fused_decode::DecodeFp8(weight.block_scales_data[scale_row_offset + block]) * tensor_scale;
      const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
      const float w0 = fused_decode::DecodeFp4(packed & 0x0Fu) * block_scale;
      const float w1 = fused_decode::DecodeFp4((packed >> 4) & 0x0Fu) * block_scale;
      for (int tile_row = 0; tile_row < valid_rows; ++tile_row) {
        const float* input_row =
            input + static_cast<std::size_t>(tile_begin + tile_row) * weight.input_cols;
        accum[tile_row] += input_row[col] * w0;
        accum[tile_row] += input_row[col + 1] * w1;
      }
    }

    for (int tile_row = 0; tile_row < kGroupedTokenTile; ++tile_row) {
      partial_sums[tile_row * blockDim.x + threadIdx.x] =
          tile_row < valid_rows ? accum[tile_row] : 0.0;
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        for (int tile_row = 0; tile_row < valid_rows; ++tile_row) {
          partial_sums[tile_row * blockDim.x + threadIdx.x] +=
              partial_sums[tile_row * blockDim.x + threadIdx.x + stride];
        }
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      for (int tile_row = 0; tile_row < valid_rows; ++tile_row) {
        output[static_cast<std::size_t>(tile_begin + tile_row) * output_rows_per_expert +
               output_row] = partial_sums[tile_row * blockDim.x];
      }
    }
    __syncthreads();
  }
}

__global__ void Nvfp4LaunchPlannedExpertMatVecRowsKernel(
    const float* input,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  __shared__ __nv_bfloat16 a_tile[kPlannedOutputTile][kPlannedWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kPlannedWmmaTileK][kPlannedWmmaTileM];
  __shared__ float c_tile[kPlannedOutputTile][kPlannedWmmaTileM];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  const int output_row_base = static_cast<int>(blockIdx.x) * kPlannedOutputTile;
  if (expert_index < 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert ||
      valid_rows <= 0 ||
      warp_id >= kPlannedWmmaWarpsPerBlock) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kPlannedOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kPlannedOutputTile));
  const float tensor_scale = *weight.tensor_scale_data;

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols;
       k_base += static_cast<std::size_t>(kPlannedWmmaTileK)) {
    for (int linear_index = tid;
         linear_index < (kPlannedOutputTile * kPlannedWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kPlannedWmmaTileK;
      const int tile_k = linear_index % kPlannedWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * blocks_per_row;
        const float block_scale =
            fused_decode::DecodeFp8(weight.block_scales_data[scale_row_offset + block]) *
            tensor_scale;
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }
    for (int linear_index = tid;
         linear_index < (kPlannedWmmaTileK * kPlannedWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kPlannedWmmaTileM;
      const int tile_token = linear_index % kPlannedWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const float input_value =
            input[input_row * weight.input_cols + (k_base + static_cast<std::size_t>(tile_k))];
        value = __float2bfloat16(input_value);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<
        wmma::matrix_a,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kPlannedWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kPlannedWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kPlannedWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kPlannedWmmaTileN;
  wmma::store_matrix_sync(
      &c_tile[warp_row][0], c_frag, kPlannedWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  for (int linear_index = tid;
       linear_index < (output_rows_this_tile * valid_rows);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_output_row = linear_index / valid_rows;
    const int tile_token = linear_index % valid_rows;
    const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
    const std::size_t output_row = static_cast<std::size_t>(output_row_base + tile_output_row);
    output[input_row * output_rows_per_expert + output_row] =
        c_tile[tile_output_row][tile_token];
  }
}

__global__ void Nvfp4LaunchPlannedExpertMatVecRowsBf16Kernel(
    const float* input,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    __nv_bfloat16* output) {
  __shared__ __nv_bfloat16 a_tile[kPlannedOutputTile][kPlannedWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kPlannedWmmaTileK][kPlannedWmmaTileM];
  __shared__ float c_tile[kPlannedOutputTile][kPlannedWmmaTileM];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  const int output_row_base = static_cast<int>(blockIdx.x) * kPlannedOutputTile;
  if (expert_index < 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert ||
      valid_rows <= 0 ||
      warp_id >= kPlannedWmmaWarpsPerBlock) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kPlannedOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kPlannedOutputTile));
  const float tensor_scale = *weight.tensor_scale_data;

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols;
       k_base += static_cast<std::size_t>(kPlannedWmmaTileK)) {
    for (int linear_index = tid;
         linear_index < (kPlannedOutputTile * kPlannedWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kPlannedWmmaTileK;
      const int tile_k = linear_index % kPlannedWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * blocks_per_row;
        const float block_scale =
            fused_decode::DecodeFp8(weight.block_scales_data[scale_row_offset + block]) *
            tensor_scale;
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }
    for (int linear_index = tid;
         linear_index < (kPlannedWmmaTileK * kPlannedWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kPlannedWmmaTileM;
      const int tile_token = linear_index % kPlannedWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const float input_value =
            input[input_row * weight.input_cols + (k_base + static_cast<std::size_t>(tile_k))];
        value = __float2bfloat16(input_value);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<
        wmma::matrix_a,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kPlannedWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kPlannedWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kPlannedWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kPlannedWmmaTileN;
  wmma::store_matrix_sync(
      &c_tile[warp_row][0], c_frag, kPlannedWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  for (int linear_index = tid;
       linear_index < (output_rows_this_tile * valid_rows);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_output_row = linear_index / valid_rows;
    const int tile_token = linear_index % valid_rows;
    const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
    const std::size_t output_row = static_cast<std::size_t>(output_row_base + tile_output_row);
    output[input_row * output_rows_per_expert + output_row] =
        __float2bfloat16(c_tile[tile_output_row][tile_token]);
  }
}

__global__ void Nvfp4LaunchPlannedExpertMatVecRelu2MaxAbsKernel(
    const float* input,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    unsigned int* expert_max_bits) {
  __shared__ __nv_bfloat16 a_tile[kPlannedOutputTile][kPlannedWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kPlannedWmmaTileK][kPlannedWmmaTileM];
  __shared__ float c_tile[kPlannedOutputTile][kPlannedWmmaTileM];
  __shared__ float shared_max[kPlannedThreadsPerBlock];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count || expert_max_bits == nullptr) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int output_row_base = static_cast<int>(blockIdx.x) * kPlannedOutputTile;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert ||
      warp_id >= kPlannedWmmaWarpsPerBlock) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kPlannedOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kPlannedOutputTile));
  const float tensor_scale = *weight.tensor_scale_data;

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols;
       k_base += static_cast<std::size_t>(kPlannedWmmaTileK)) {
    for (int linear_index = tid;
         linear_index < (kPlannedOutputTile * kPlannedWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kPlannedWmmaTileK;
      const int tile_k = linear_index % kPlannedWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * blocks_per_row;
        const float block_scale =
            fused_decode::DecodeFp8(weight.block_scales_data[scale_row_offset + block]) *
            tensor_scale;
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }
    for (int linear_index = tid;
         linear_index < (kPlannedWmmaTileK * kPlannedWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kPlannedWmmaTileM;
      const int tile_token = linear_index % kPlannedWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const float input_value =
            input[input_row * weight.input_cols + (k_base + static_cast<std::size_t>(tile_k))];
        value = __float2bfloat16(input_value);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<
        wmma::matrix_a,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kPlannedWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kPlannedWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kPlannedWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kPlannedWmmaTileN;
  wmma::store_matrix_sync(
      &c_tile[warp_row][0], c_frag, kPlannedWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  float thread_max = 0.0f;
  for (int linear_index = tid;
       linear_index < (output_rows_this_tile * valid_rows);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_output_row = linear_index / valid_rows;
    const int tile_token = linear_index % valid_rows;
    const float value = fused_decode::Relu2(c_tile[tile_output_row][tile_token]);
    thread_max = fmaxf(thread_max, value);
  }

  shared_max[tid] = thread_max;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
    if (threadIdx.x < stride &&
        shared_max[threadIdx.x + stride] > shared_max[threadIdx.x]) {
      shared_max[threadIdx.x] = shared_max[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicMax(&expert_max_bits[expert_index], __float_as_uint(shared_max[0]));
  }
}

__global__ void Nvfp4LaunchPlannedExpertMatVecRelu2PackKernel(
    const float* input,
    const float* output_expert_tensor_scales,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    std::size_t output_row_capacity,
    Nvfp4ScaleLayout output_scale_layout,
    std::uint8_t* output_packed,
    std::uint8_t* output_block_scales,
    std::uint8_t* output_matmul_scales) {
  __shared__ __nv_bfloat16 a_tile[kPlannedOutputTile][kPlannedWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kPlannedWmmaTileK][kPlannedWmmaTileM];
  __shared__ float c_tile[kPlannedOutputTile][kPlannedWmmaTileM];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count ||
      output_expert_tensor_scales == nullptr ||
      output_packed == nullptr ||
      output_block_scales == nullptr ||
      output_matmul_scales == nullptr) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int lane_id = tid & 31;
  const int output_row_base = static_cast<int>(blockIdx.x) * kPlannedOutputTile;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert ||
      warp_id >= kPlannedWmmaWarpsPerBlock) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t pairs_per_input_row = weight.input_cols / 2;
  const std::size_t input_blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kPlannedOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kPlannedOutputTile));
  const float weight_tensor_scale = *weight.tensor_scale_data;

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols;
       k_base += static_cast<std::size_t>(kPlannedWmmaTileK)) {
    for (int linear_index = tid;
         linear_index < (kPlannedOutputTile * kPlannedWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kPlannedWmmaTileK;
      const int tile_k = linear_index % kPlannedWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_input_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * input_blocks_per_row;
        const float block_scale =
            fused_decode::DecodeFp8(weight.block_scales_data[scale_row_offset + block]) *
            weight_tensor_scale;
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }
    for (int linear_index = tid;
         linear_index < (kPlannedWmmaTileK * kPlannedWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kPlannedWmmaTileM;
      const int tile_token = linear_index % kPlannedWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const float input_value =
            input[input_row * weight.input_cols + (k_base + static_cast<std::size_t>(tile_k))];
        value = __float2bfloat16(input_value);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<
        wmma::matrix_a,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kPlannedWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kPlannedWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kPlannedWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kPlannedWmmaTileN;
  wmma::store_matrix_sync(
      &c_tile[warp_row][0], c_frag, kPlannedWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  if (lane_id >= valid_rows || warp_row >= output_rows_this_tile) {
    return;
  }

  const std::size_t output_row = static_cast<std::size_t>(row_start + lane_id);
  if (output_row >= output_row_capacity) {
    return;
  }
  const std::size_t blocks_per_output_row =
      output_rows_per_expert / fused_decode::kNvfp4BlockWidth;
  const std::size_t padded_blocks_per_row =
      RoundUp(blocks_per_output_row, kNvfp4ScaleBlockTile);
  const std::size_t block_col =
      static_cast<std::size_t>(output_row_base + warp_row) / fused_decode::kNvfp4BlockWidth;
  const float tensor_scale = output_expert_tensor_scales[expert_index];

  float activated[16];
  float block_max_abs = 0.0f;
  for (int col = 0; col < 16; ++col) {
    const float value = fused_decode::Relu2(c_tile[warp_row + col][lane_id]);
    activated[col] = value;
    if (value > block_max_abs) {
      block_max_abs = value;
    }
  }

  float block_scale = 1.0f;
  if (block_max_abs > 0.0f) {
    block_scale = fused_decode::ClampNvfp4Scale(
        block_max_abs / (fused_decode::kNvfp4Fp4MaxFinite * tensor_scale));
  }
  const std::uint8_t encoded_block_scale = fused_decode::EncodeFp8Scale(block_scale);
  const std::size_t block_scale_offset = output_row * blocks_per_output_row + block_col;
  output_block_scales[block_scale_offset] = encoded_block_scale;
  output_matmul_scales[ExecutionScaleOffset(
      output_row, block_col, padded_blocks_per_row, output_scale_layout)] = encoded_block_scale;

  const float pack_scale = tensor_scale * block_scale;
  const std::size_t packed_row_offset =
      output_row * (output_rows_per_expert / 2u) +
      block_col * (fused_decode::kNvfp4BlockWidth / 2u);
  for (int pair = 0; pair < 8; ++pair) {
    const std::uint8_t lhs = fused_decode::EncodeFp4(activated[pair * 2] / pack_scale);
    const std::uint8_t rhs = fused_decode::EncodeFp4(activated[pair * 2 + 1] / pack_scale);
    output_packed[packed_row_offset + static_cast<std::size_t>(pair)] =
        static_cast<std::uint8_t>((lhs & 0x0Fu) | ((rhs & 0x0Fu) << 4u));
  }
}

__global__ void Nvfp4LaunchPlannedPackedInputExpertMatVecRowsKernel(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const float* input_tensor_scale_data,
    const float* input_expert_tensor_scales,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  __shared__ __nv_bfloat16 a_tile[kPlannedOutputTile][kPlannedWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kPlannedWmmaTileK][kPlannedWmmaTileM];
  __shared__ float c_tile[kPlannedOutputTile][kPlannedWmmaTileM];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count ||
      packed_input == nullptr ||
      input_block_scales == nullptr ||
      input_tensor_scale_data == nullptr) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int output_row_base = static_cast<int>(blockIdx.x) * kPlannedOutputTile;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert ||
      warp_id >= kPlannedWmmaWarpsPerBlock) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kPlannedOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kPlannedOutputTile));
  const float weight_tensor_scale = *weight.tensor_scale_data;
  const float input_tensor_scale =
      input_expert_tensor_scales != nullptr ? input_expert_tensor_scales[expert_index]
                                            : *input_tensor_scale_data;

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols;
       k_base += static_cast<std::size_t>(kPlannedWmmaTileK)) {
    for (int linear_index = tid;
         linear_index < (kPlannedOutputTile * kPlannedWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kPlannedWmmaTileK;
      const int tile_k = linear_index % kPlannedWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * blocks_per_row;
        const float block_scale =
            fused_decode::DecodeFp8(weight.block_scales_data[scale_row_offset + block]) *
            weight_tensor_scale;
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }
    for (int linear_index = tid;
         linear_index < (kPlannedWmmaTileK * kPlannedWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kPlannedWmmaTileM;
      const int tile_token = linear_index % kPlannedWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset = input_row * pairs_per_row;
        const std::size_t scale_row_offset = input_row * blocks_per_row;
        const float block_scale =
            fused_decode::DecodeFp8(input_block_scales[scale_row_offset + block]) *
            input_tensor_scale;
        const std::uint8_t packed = packed_input[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<
        wmma::matrix_a,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kPlannedWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kPlannedWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kPlannedWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kPlannedWmmaTileN;
  wmma::store_matrix_sync(
      &c_tile[warp_row][0], c_frag, kPlannedWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  for (int linear_index = tid;
       linear_index < (output_rows_this_tile * valid_rows);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_output_row = linear_index / valid_rows;
    const int tile_token = linear_index % valid_rows;
    const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
    const std::size_t output_row = static_cast<std::size_t>(output_row_base + tile_output_row);
    output[input_row * output_rows_per_expert + output_row] =
        c_tile[tile_output_row][tile_token];
  }
}

__global__ void Nvfp4LaunchPlannedPackedInputExpertMatVecRelu2MaxAbsKernel(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const float* input_tensor_scale_data,
    const float* input_expert_tensor_scales,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_m_limits,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    unsigned int* expert_max_bits) {
  __shared__ __nv_bfloat16 a_tile[kPlannedOutputTile][kPlannedWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kPlannedWmmaTileK][kPlannedWmmaTileM];
  __shared__ float c_tile[kPlannedOutputTile][kPlannedWmmaTileM];
  __shared__ float shared_max[kPlannedThreadsPerBlock];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count ||
      packed_input == nullptr ||
      input_block_scales == nullptr ||
      input_tensor_scale_data == nullptr ||
      expert_max_bits == nullptr) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int output_row_base = static_cast<int>(blockIdx.x) * kPlannedOutputTile;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_index * kGroupedTokenTile;
  const int m_limit = cta_m_limits[cta_index];
  const int valid_rows =
      max(0, min(m_limit - row_start, kGroupedTokenTile));
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert ||
      warp_id >= kPlannedWmmaWarpsPerBlock) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kPlannedOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kPlannedOutputTile));
  const float weight_tensor_scale = *weight.tensor_scale_data;
  const float input_tensor_scale =
      input_expert_tensor_scales != nullptr ? input_expert_tensor_scales[expert_index]
                                            : *input_tensor_scale_data;

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols;
       k_base += static_cast<std::size_t>(kPlannedWmmaTileK)) {
    for (int linear_index = tid;
         linear_index < (kPlannedOutputTile * kPlannedWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kPlannedWmmaTileK;
      const int tile_k = linear_index % kPlannedWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * blocks_per_row;
        const float block_scale =
            fused_decode::DecodeFp8(weight.block_scales_data[scale_row_offset + block]) *
            weight_tensor_scale;
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }
    for (int linear_index = tid;
         linear_index < (kPlannedWmmaTileK * kPlannedWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kPlannedWmmaTileM;
      const int tile_token = linear_index % kPlannedWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset = input_row * pairs_per_row;
        const std::size_t scale_row_offset = input_row * blocks_per_row;
        const float block_scale =
            fused_decode::DecodeFp8(input_block_scales[scale_row_offset + block]) *
            input_tensor_scale;
        const std::uint8_t packed = packed_input[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<
        wmma::matrix_a,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kPlannedWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kPlannedWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kPlannedWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kPlannedWmmaTileN;
  wmma::store_matrix_sync(
      &c_tile[warp_row][0], c_frag, kPlannedWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  float thread_max = 0.0f;
  for (int linear_index = tid;
       linear_index < (output_rows_this_tile * valid_rows);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_output_row = linear_index / valid_rows;
    const int tile_token = linear_index % valid_rows;
    const float value = fused_decode::Relu2(c_tile[tile_output_row][tile_token]);
    thread_max = fmaxf(thread_max, value);
  }

  shared_max[tid] = thread_max;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
    if (threadIdx.x < stride &&
        shared_max[threadIdx.x + stride] > shared_max[threadIdx.x]) {
      shared_max[threadIdx.x] = shared_max[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicMax(&expert_max_bits[expert_index], __float_as_uint(shared_max[0]));
  }
}

__global__ void Nvfp4LaunchPlannedPackedInputExpertMatVecRelu2PackKernel(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const float* input_tensor_scale_data,
    const float* input_expert_tensor_scales,
    const float* output_expert_tensor_scales,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_m_limits,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    std::size_t output_row_capacity,
    Nvfp4ScaleLayout output_scale_layout,
    std::uint8_t* output_packed,
    std::uint8_t* output_block_scales,
    std::uint8_t* output_matmul_scales) {
  __shared__ __nv_bfloat16 a_tile[kPlannedOutputTile][kPlannedWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kPlannedWmmaTileK][kPlannedWmmaTileM];
  __shared__ float c_tile[kPlannedOutputTile][kPlannedWmmaTileM];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count ||
      packed_input == nullptr ||
      input_block_scales == nullptr ||
      input_tensor_scale_data == nullptr ||
      output_expert_tensor_scales == nullptr ||
      output_packed == nullptr ||
      output_block_scales == nullptr ||
      output_matmul_scales == nullptr) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int lane_id = tid & 31;
  const int output_row_base = static_cast<int>(blockIdx.x) * kPlannedOutputTile;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_index * kGroupedTokenTile;
  const int m_limit = cta_m_limits[cta_index];
  const int valid_rows =
      max(0, min(m_limit - row_start, kGroupedTokenTile));
  if (expert_index < 0 ||
      valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert ||
      warp_id >= kPlannedWmmaWarpsPerBlock) {
    return;
  }

  const FusedNvfp4WeightView weight = weights[expert_index];
  const std::size_t pairs_per_input_row = weight.input_cols / 2;
  const std::size_t input_blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kPlannedOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kPlannedOutputTile));
  const float weight_tensor_scale = *weight.tensor_scale_data;
  const float input_tensor_scale =
      input_expert_tensor_scales != nullptr ? input_expert_tensor_scales[expert_index]
                                            : *input_tensor_scale_data;

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols;
       k_base += static_cast<std::size_t>(kPlannedWmmaTileK)) {
    for (int linear_index = tid;
         linear_index < (kPlannedOutputTile * kPlannedWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kPlannedWmmaTileK;
      const int tile_k = linear_index % kPlannedWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_input_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * input_blocks_per_row;
        const float block_scale =
            fused_decode::DecodeFp8(weight.block_scales_data[scale_row_offset + block]) *
            weight_tensor_scale;
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }
    for (int linear_index = tid;
         linear_index < (kPlannedWmmaTileK * kPlannedWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kPlannedWmmaTileM;
      const int tile_token = linear_index % kPlannedWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset = input_row * pairs_per_input_row;
        const std::size_t scale_row_offset = input_row * input_blocks_per_row;
        const float block_scale =
            fused_decode::DecodeFp8(input_block_scales[scale_row_offset + block]) *
            input_tensor_scale;
        const std::uint8_t packed = packed_input[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<
        wmma::matrix_a,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kPlannedWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kPlannedWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kPlannedWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kPlannedWmmaTileN;
  wmma::store_matrix_sync(
      &c_tile[warp_row][0], c_frag, kPlannedWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  if (lane_id >= valid_rows || warp_row >= output_rows_this_tile) {
    return;
  }

  const std::size_t output_row = static_cast<std::size_t>(row_start + lane_id);
  if (output_row >= output_row_capacity) {
    return;
  }
  const std::size_t blocks_per_output_row =
      output_rows_per_expert / fused_decode::kNvfp4BlockWidth;
  const std::size_t padded_rows = RoundUp(output_row_capacity, RowTile(output_scale_layout));
  const std::size_t padded_blocks_per_row =
      RoundUp(blocks_per_output_row, kNvfp4ScaleBlockTile);
  const std::size_t block_col =
      static_cast<std::size_t>(output_row_base + warp_row) / fused_decode::kNvfp4BlockWidth;
  const float tensor_scale = output_expert_tensor_scales[expert_index];

  float activated[16];
  float block_max_abs = 0.0f;
  for (int col = 0; col < 16; ++col) {
    const float value = fused_decode::Relu2(c_tile[warp_row + col][lane_id]);
    activated[col] = value;
    if (value > block_max_abs) {
      block_max_abs = value;
    }
  }

  float block_scale = 1.0f;
  if (block_max_abs > 0.0f) {
    block_scale = fused_decode::ClampNvfp4Scale(
        block_max_abs / (fused_decode::kNvfp4Fp4MaxFinite * tensor_scale));
  }
  const std::uint8_t encoded_block_scale = fused_decode::EncodeFp8Scale(block_scale);
  const std::size_t block_scale_offset = output_row * blocks_per_output_row + block_col;
  output_block_scales[block_scale_offset] = encoded_block_scale;
  output_matmul_scales[ExecutionScaleOffset(
      output_row, block_col, padded_blocks_per_row, output_scale_layout)] = encoded_block_scale;

  const float pack_scale = tensor_scale * block_scale;
  const std::size_t packed_row_offset =
      output_row * (output_rows_per_expert / 2u) +
      block_col * (fused_decode::kNvfp4BlockWidth / 2u);
  for (int pair = 0; pair < 8; ++pair) {
    const std::uint8_t lhs = fused_decode::EncodeFp4(activated[pair * 2] / pack_scale);
    const std::uint8_t rhs = fused_decode::EncodeFp4(activated[pair * 2 + 1] / pack_scale);
    output_packed[packed_row_offset + static_cast<std::size_t>(pair)] =
        static_cast<std::uint8_t>((lhs & 0x0Fu) | ((rhs & 0x0Fu) << 4u));
  }
}

template <bool kGroupedRows>
__global__ void Nvfp4MatVecRowsKernel(
    const float* input,
    std::size_t input_row_count,
    const int* expert_offsets,
    int expert_index,
    fused_decode::Nvfp4WeightView weight,
    float* output) {
  __shared__ float partial_sums[kGroupedTokenTile * fused_decode::kThreadsPerBlock];

  const std::size_t output_row = static_cast<std::size_t>(blockIdx.x);
  if (output_row >= weight.output_rows) {
    return;
  }

  int begin_row = 0;
  int end_row = static_cast<int>(input_row_count);
  if constexpr (kGroupedRows) {
    begin_row = expert_offsets[expert_index];
    end_row = expert_offsets[expert_index + 1];
  }
  if (begin_row >= end_row) {
    return;
  }

  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t packed_row_offset = output_row * (weight.input_cols / 2);
  const std::size_t scale_row_offset = output_row * blocks_per_row;
  const float tensor_scale = *weight.tensor_scale_data;

  for (int tile_begin = begin_row; tile_begin < end_row; tile_begin += kGroupedTokenTile) {
    const int remaining_rows = end_row - tile_begin;
    const int valid_rows =
        remaining_rows < kGroupedTokenTile ? remaining_rows : kGroupedTokenTile;
    float accum[kGroupedTokenTile] = {0.0f};

    for (std::size_t pair_index = static_cast<std::size_t>(threadIdx.x);
         pair_index < pairs_per_row;
         pair_index += blockDim.x) {
      const std::size_t block = pair_index / 8u;
      const std::size_t pair_in_block = pair_index % 8u;
      const std::size_t col = block * fused_decode::kNvfp4BlockWidth + pair_in_block * 2u;
      const float block_scale =
          fused_decode::DecodeFp8(weight.block_scales_data[scale_row_offset + block]) * tensor_scale;
      const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
      const float w0 = fused_decode::DecodeFp4(packed & 0x0Fu) * block_scale;
      const float w1 = fused_decode::DecodeFp4((packed >> 4) & 0x0Fu) * block_scale;
      for (int tile_row = 0; tile_row < valid_rows; ++tile_row) {
        const float* input_row =
            input + static_cast<std::size_t>(tile_begin + tile_row) * weight.input_cols;
        accum[tile_row] += input_row[col] * w0;
        accum[tile_row] += input_row[col + 1] * w1;
      }
    }

    for (int tile_row = 0; tile_row < kGroupedTokenTile; ++tile_row) {
      partial_sums[tile_row * blockDim.x + threadIdx.x] =
          tile_row < valid_rows ? accum[tile_row] : 0.0;
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        for (int tile_row = 0; tile_row < valid_rows; ++tile_row) {
          partial_sums[tile_row * blockDim.x + threadIdx.x] +=
              partial_sums[tile_row * blockDim.x + threadIdx.x + stride];
        }
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      for (int tile_row = 0; tile_row < valid_rows; ++tile_row) {
        output[static_cast<std::size_t>(tile_begin + tile_row) * weight.output_rows + output_row] =
            partial_sums[tile_row * blockDim.x];
      }
    }
    __syncthreads();
  }
}

template <int kOutputTile, int kThreadsPerBlock>
__global__ void Nvfp4ContiguousWmmaMatVecRowsKernel(
    const float* input,
    std::size_t input_row_count,
    FusedNvfp4WeightView weight,
    float* output) {
  constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;

  __shared__ __nv_bfloat16 a_tile[kOutputTile][kPlannedWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kPlannedWmmaTileK][kPlannedWmmaTileM];
  __shared__ float c_tile[kOutputTile][kPlannedWmmaTileM];

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / 32;
  const int output_row_base = static_cast<int>(blockIdx.x) * kOutputTile;
  const int row_start = static_cast<int>(blockIdx.y) * kPlannedWmmaTileM;
  if (warp_id >= kWarpsPerBlock ||
      static_cast<std::size_t>(output_row_base) >= weight.output_rows ||
      static_cast<std::size_t>(row_start) >= input_row_count) {
    return;
  }

  const std::size_t remaining_input_rows =
      input_row_count - static_cast<std::size_t>(row_start);
  const int valid_rows = static_cast<int>(
      remaining_input_rows < static_cast<std::size_t>(kPlannedWmmaTileM)
          ? remaining_input_rows
          : static_cast<std::size_t>(kPlannedWmmaTileM));
  const std::size_t remaining_output_rows =
      weight.output_rows - static_cast<std::size_t>(output_row_base);
  const int output_rows_this_tile = static_cast<int>(
      remaining_output_rows < static_cast<std::size_t>(kOutputTile)
          ? remaining_output_rows
          : static_cast<std::size_t>(kOutputTile));
    const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const float tensor_scale = *weight.tensor_scale_data;

  wmma::fragment<
      wmma::accumulator,
      kPlannedWmmaTileM,
      kPlannedWmmaTileN,
      kPlannedWmmaTileK,
      float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols;
       k_base += static_cast<std::size_t>(kPlannedWmmaTileK)) {
    for (int linear_index = tid;
         linear_index < (kOutputTile * kPlannedWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kPlannedWmmaTileK;
      const int tile_k = linear_index % kPlannedWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * blocks_per_row;
        const float block_scale =
            fused_decode::DecodeFp8(weight.block_scales_data[scale_row_offset + block]) *
            tensor_scale;
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(fused_decode::DecodeFp4(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }
    for (int linear_index = tid;
         linear_index < (kPlannedWmmaTileK * kPlannedWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kPlannedWmmaTileM;
      const int tile_token = linear_index % kPlannedWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const float input_value =
            input[input_row * weight.input_cols + (k_base + static_cast<std::size_t>(tile_k))];
        value = __float2bfloat16(input_value);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<
        wmma::matrix_a,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        a_frag;
    wmma::fragment<
        wmma::matrix_b,
        kPlannedWmmaTileM,
        kPlannedWmmaTileN,
        kPlannedWmmaTileK,
        __nv_bfloat16,
        wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kPlannedWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kPlannedWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kPlannedWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kPlannedWmmaTileN;
  wmma::store_matrix_sync(
      &c_tile[warp_row][0], c_frag, kPlannedWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  for (int linear_index = tid;
       linear_index < (output_rows_this_tile * valid_rows);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_output_row = linear_index / valid_rows;
    const int tile_token = linear_index % valid_rows;
    const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
    const std::size_t output_row = static_cast<std::size_t>(output_row_base + tile_output_row);
    output[input_row * weight.output_rows + output_row] =
        c_tile[tile_output_row][tile_token];
  }
}

__global__ void ReduceSelectionOutputsKernel(
    const float* grouped_output,
    const int* selection_to_sorted,
    const int* sorted_to_permuted,
    const float* selected_weights,
    std::size_t token_count,
    std::size_t top_k,
    std::size_t hidden_size,
    float* output,
    float* routed_output) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = token_count * hidden_size;
  if (index >= count) {
    return;
  }

  const std::size_t token_index = index / hidden_size;
  const std::size_t hidden_index = index % hidden_size;
  double accum = 0.0;
  for (std::size_t slot = 0; slot < top_k; ++slot) {
    const std::size_t selection_index = token_index * top_k + slot;
    const int sorted_index = selection_to_sorted[selection_index];
    if (sorted_index < 0) {
      continue;
    }
    const int output_row =
        sorted_to_permuted != nullptr ? sorted_to_permuted[sorted_index] : sorted_index;
    if (output_row < 0) {
      continue;
    }
    const std::size_t grouped_offset =
        static_cast<std::size_t>(output_row) * hidden_size + hidden_index;
    accum += static_cast<double>(selected_weights[selection_index]) *
             static_cast<double>(grouped_output[grouped_offset]);
  }
  const float reduced = static_cast<float>(accum);
  output[index] = reduced;
  if (routed_output != nullptr) {
    routed_output[index] = reduced;
  }
}

__global__ void AccumulateSharedOutputKernel(
    const float* shared_output,
    std::size_t count,
    float* output,
    float* shared_output_copy) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const float value = shared_output[index];
  output[index] += value;
  if (shared_output_copy != nullptr) {
    shared_output_copy[index] = value;
  }
}

bool LaunchZeroBuffer(float* data, std::size_t count) {
  if (data == nullptr || count == 0) {
    return true;
  }
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1u) / block.x));
  ZeroBufferKernel<<<grid, block>>>(data, count);
  return CheckCuda(cudaGetLastError());
}

bool LaunchZeroBf16Buffer(__nv_bfloat16* data, std::size_t count) {
  if (data == nullptr || count == 0) {
    return true;
  }
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1u) / block.x));
  ZeroBf16BufferKernel<<<grid, block>>>(data, count);
  return CheckCuda(cudaGetLastError());
}

bool ClearDeviceNvfp4Matrix(DeviceNvfp4Matrix* matrix, float tensor_scale = 1.0f) {
  if (matrix == nullptr || !matrix->valid()) {
    return false;
  }
  return CheckCuda(cudaMemset(
             const_cast<std::uint8_t*>(matrix->packed_data()), 0, matrix->packed_nbytes())) &&
         CheckCuda(cudaMemset(
             const_cast<std::uint8_t*>(matrix->block_scales_data()),
             0,
             matrix->block_scales_nbytes())) &&
         CheckCuda(cudaMemset(
             const_cast<std::uint8_t*>(matrix->matmul_block_scales_data()),
             0,
             matrix->matmul_block_scales_nbytes())) &&
         CheckCuda(cudaMemcpy(
             const_cast<std::uint8_t*>(matrix->tensor_scale_data()),
             &tensor_scale,
             sizeof(tensor_scale),
             cudaMemcpyHostToDevice));
}

bool LaunchGatherRows(
    const float* input,
    const int* row_indices,
    const int* active_output_rows,
    std::size_t output_row_capacity,
    std::size_t input_rows,
    std::size_t cols,
    float* output) {
  const std::size_t count = output_row_capacity * cols;
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1u) / block.x));
  GatherRowsKernel<<<grid, block>>>(
      input,
      row_indices,
      active_output_rows,
      output,
      output_row_capacity,
      input_rows,
      cols);
  return CheckCuda(cudaGetLastError());
}

bool LaunchComputeExpertActivationScales(
    const float* input,
    const int* expert_first_token_offsets,
    std::size_t n_experts,
    std::size_t cols,
    float* output_scales) {
  if (input == nullptr ||
      expert_first_token_offsets == nullptr ||
      output_scales == nullptr ||
      n_experts == 0 ||
      cols == 0) {
    return false;
  }
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(n_experts));
  ComputeExpertActivationScalesKernel<<<grid, block>>>(
      input,
      expert_first_token_offsets,
      n_experts,
      cols,
      output_scales);
  return CheckCuda(cudaGetLastError());
}

bool LaunchComputeExpertActivationScalesBf16(
    const __nv_bfloat16* input,
    const int* expert_first_token_offsets,
    std::size_t n_experts,
    std::size_t cols,
    float* output_scales) {
  if (input == nullptr ||
      expert_first_token_offsets == nullptr ||
      n_experts == 0 ||
      cols == 0 ||
      output_scales == nullptr) {
    return false;
  }
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(n_experts));
  ComputeExpertActivationScalesBf16Kernel<<<grid, block>>>(
      input,
      expert_first_token_offsets,
      n_experts,
      cols,
      output_scales);
  return CheckCuda(cudaGetLastError());
}

bool LaunchMaxBitsToTensorScales(
    const unsigned int* max_bits,
    std::size_t count,
    float* output_scales) {
  if (max_bits == nullptr || output_scales == nullptr || count == 0) {
    return false;
  }
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1u) / block.x));
  MaxBitsToTensorScalesKernel<<<grid, block>>>(max_bits, count, output_scales);
  return CheckCuda(cudaGetLastError());
}

bool LaunchQuantizeDequantizeRows(
    float* data,
    const int* active_row_count,
    std::size_t row_capacity,
    std::size_t cols) {
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(row_capacity));
  QuantizeDequantizeRowsKernel<<<grid, block>>>(
      data,
      active_row_count,
      row_capacity,
      cols);
  return CheckCuda(cudaGetLastError());
}

bool LaunchRelu2(float* data, std::size_t count) {
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1u) / block.x));
  Relu2Kernel<<<grid, block>>>(data, count);
  return CheckCuda(cudaGetLastError());
}

bool LaunchRelu2Rows(
    float* data,
    const int* active_row_count,
    std::size_t row_capacity,
    std::size_t cols) {
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(((row_capacity * cols) + block.x - 1u) / block.x));
  Relu2RowsKernel<<<grid, block>>>(
      data,
      active_row_count,
      row_capacity,
      cols);
  return CheckCuda(cudaGetLastError());
}

bool LaunchGroupedMatVec(
    const float* input,
    const int* expert_offsets,
    std::size_t n_experts,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  if (weights == nullptr || n_experts == 0 || output_rows_per_expert == 0) {
    return false;
  }
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(n_experts * output_rows_per_expert));
  Nvfp4GroupedExpertMatVecRowsKernel<<<grid, block>>>(
      input,
      expert_offsets,
      n_experts,
      weights,
      output_rows_per_expert,
      output);
  return CheckCuda(cudaGetLastError());
}

bool LaunchContiguousMatVec(
    const float* input,
    std::size_t input_row_count,
    const FusedNvfp4WeightView& weight,
    float* output) {
  if (!ValidFusedNvfp4WeightView(weight)) {
    return false;
  }
  if (input_row_count == 0) {
    return true;
  }
  const int output_tile = SelectContiguousOutputTile(weight.output_rows, input_row_count);
  const std::size_t input_row_tile_count =
      CeilDiv(input_row_count, static_cast<std::size_t>(kPlannedWmmaTileM));

  if (output_tile == kContiguousLargeOutputTile) {
    const dim3 block(kContiguousLargeThreadsPerBlock);
    const dim3 grid(
        static_cast<unsigned int>(CeilDiv(
            weight.output_rows,
            static_cast<std::size_t>(kContiguousLargeOutputTile))),
        static_cast<unsigned int>(input_row_tile_count));
    Nvfp4ContiguousWmmaMatVecRowsKernel<
        kContiguousLargeOutputTile,
        kContiguousLargeThreadsPerBlock><<<grid, block>>>(
        input,
        input_row_count,
        weight,
        output);
    return CheckCuda(cudaGetLastError());
  }

  if (output_tile == kContiguousMediumOutputTile) {
    const dim3 block(kContiguousMediumThreadsPerBlock);
    const dim3 grid(
        static_cast<unsigned int>(CeilDiv(
            weight.output_rows,
            static_cast<std::size_t>(kContiguousMediumOutputTile))),
        static_cast<unsigned int>(input_row_tile_count));
    Nvfp4ContiguousWmmaMatVecRowsKernel<
        kContiguousMediumOutputTile,
        kContiguousMediumThreadsPerBlock><<<grid, block>>>(
        input,
        input_row_count,
        weight,
        output);
    return CheckCuda(cudaGetLastError());
  }

  const dim3 block(kContiguousSmallThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(CeilDiv(
          weight.output_rows,
          static_cast<std::size_t>(kContiguousSmallOutputTile))),
      static_cast<unsigned int>(input_row_tile_count));
  Nvfp4ContiguousWmmaMatVecRowsKernel<
      kContiguousSmallOutputTile,
      kContiguousSmallThreadsPerBlock><<<grid, block>>>(
      input,
      input_row_count,
      weight,
      output);
  return CheckCuda(cudaGetLastError());
}

bool LaunchPlannedMatVec(
    const float* input,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  if (launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0) {
    return false;
  }
  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(
          launch_plan->n_experts(),
          active_selection_count,
          launch_plan->selected_token_tile());
  if (!current_cta_capacity.has_value() ||
      *current_cta_capacity == 0 ||
      *current_cta_capacity > launch_plan->cta_capacity()) {
    return false;
  }
  const std::size_t output_row_tile_count =
      (output_rows_per_expert + static_cast<std::size_t>(kPlannedOutputTile) - 1u) /
      static_cast<std::size_t>(kPlannedOutputTile);
  if (output_row_tile_count == 0) {
    return false;
  }
  const dim3 block(kPlannedThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  Nvfp4LaunchPlannedExpertMatVecRowsKernel<<<grid, block>>>(
      input,
      launch_plan->num_non_exiting_ctas(),
      launch_plan->cta_expert_ids(),
      launch_plan->cta_row_starts(),
      launch_plan->cta_valid_rows(),
      weights,
      output_rows_per_expert,
      output);
  return CheckCuda(cudaGetLastError());
}

bool LaunchPlannedMatVecBf16(
    const float* input,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    __nv_bfloat16* output) {
  if (input == nullptr ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output == nullptr) {
    return false;
  }
  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(
          launch_plan->n_experts(),
          active_selection_count,
          launch_plan->selected_token_tile());
  if (!current_cta_capacity.has_value() ||
      *current_cta_capacity == 0 ||
      *current_cta_capacity > launch_plan->cta_capacity()) {
    return false;
  }
  const std::size_t output_row_tile_count =
      (output_rows_per_expert + static_cast<std::size_t>(kPlannedOutputTile) - 1u) /
      static_cast<std::size_t>(kPlannedOutputTile);
  if (output_row_tile_count == 0) {
    return false;
  }
  const dim3 block(kPlannedThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  Nvfp4LaunchPlannedExpertMatVecRowsBf16Kernel<<<grid, block>>>(
      input,
      launch_plan->num_non_exiting_ctas(),
      launch_plan->cta_expert_ids(),
      launch_plan->cta_row_starts(),
      launch_plan->cta_valid_rows(),
      weights,
      output_rows_per_expert,
      output);
  return CheckCuda(cudaGetLastError());
}

bool LaunchPlannedMatVecRelu2ExpertScales(
    const float* input,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output_expert_scales) {
  if (input == nullptr ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output_expert_scales == nullptr) {
    return false;
  }
  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(
          launch_plan->n_experts(),
          active_selection_count,
          launch_plan->selected_token_tile());
  if (!current_cta_capacity.has_value() ||
      *current_cta_capacity == 0 ||
      *current_cta_capacity > launch_plan->cta_capacity()) {
    return false;
  }
  const std::size_t output_row_tile_count =
      (output_rows_per_expert + static_cast<std::size_t>(kPlannedOutputTile) - 1u) /
      static_cast<std::size_t>(kPlannedOutputTile);
  if (output_row_tile_count == 0) {
    return false;
  }
  if (!CheckCuda(cudaMemset(
          reinterpret_cast<void*>(output_expert_scales),
          0,
          launch_plan->n_experts() * sizeof(float)))) {
    return false;
  }
  const dim3 block(kPlannedThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  Nvfp4LaunchPlannedExpertMatVecRelu2MaxAbsKernel<<<grid, block>>>(
      input,
      launch_plan->num_non_exiting_ctas(),
      launch_plan->cta_expert_ids(),
      launch_plan->cta_row_starts(),
      launch_plan->cta_valid_rows(),
      weights,
      output_rows_per_expert,
      reinterpret_cast<unsigned int*>(output_expert_scales));
  return CheckCuda(cudaGetLastError()) &&
         LaunchMaxBitsToTensorScales(
             reinterpret_cast<const unsigned int*>(output_expert_scales),
             launch_plan->n_experts(),
             output_expert_scales);
}

bool LaunchPlannedMatVecRelu2Pack(
    const float* input,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    const float* output_expert_scales,
    DeviceNvfp4Matrix* output_pack) {
  if (input == nullptr ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output_expert_scales == nullptr ||
      output_pack == nullptr ||
      !output_pack->valid() ||
      output_pack->rows() < launch_plan->padded_row_capacity() ||
      output_pack->cols() != output_rows_per_expert) {
    return false;
  }
  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(
          launch_plan->n_experts(),
          active_selection_count,
          launch_plan->selected_token_tile());
  if (!current_cta_capacity.has_value() ||
      *current_cta_capacity == 0 ||
      *current_cta_capacity > launch_plan->cta_capacity()) {
    return false;
  }
  const std::size_t output_row_tile_count =
      (output_rows_per_expert + static_cast<std::size_t>(kPlannedOutputTile) - 1u) /
      static_cast<std::size_t>(kPlannedOutputTile);
  if (output_row_tile_count == 0 || !ClearDeviceNvfp4Matrix(output_pack)) {
    return false;
  }
  const dim3 block(kPlannedThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  Nvfp4LaunchPlannedExpertMatVecRelu2PackKernel<<<grid, block>>>(
      input,
      output_expert_scales,
      launch_plan->num_non_exiting_ctas(),
      launch_plan->cta_expert_ids(),
      launch_plan->cta_row_starts(),
      launch_plan->cta_valid_rows(),
      weights,
      output_rows_per_expert,
      output_pack->rows(),
      output_pack->scale_layout(),
      const_cast<std::uint8_t*>(output_pack->packed_data()),
      const_cast<std::uint8_t*>(output_pack->block_scales_data()),
      const_cast<std::uint8_t*>(output_pack->matmul_block_scales_data()));
  return CheckCuda(cudaGetLastError());
}

bool LaunchPlannedPackedInputMatVec(
    const DeviceNvfp4Matrix& input_pack,
    const float* input_expert_tensor_scales,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  if (!input_pack.valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output == nullptr) {
    return false;
  }
  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(
          launch_plan->n_experts(),
          active_selection_count,
          launch_plan->selected_token_tile());
  if (!current_cta_capacity.has_value() ||
      *current_cta_capacity == 0 ||
      *current_cta_capacity > launch_plan->cta_capacity()) {
    return false;
  }
  const std::size_t output_row_tile_count =
      (output_rows_per_expert + static_cast<std::size_t>(kPlannedOutputTile) - 1u) /
      static_cast<std::size_t>(kPlannedOutputTile);
  if (output_row_tile_count == 0) {
    return false;
  }
  const dim3 block(kPlannedThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  Nvfp4LaunchPlannedPackedInputExpertMatVecRowsKernel<<<grid, block>>>(
      input_pack.packed_data(),
      input_pack.block_scales_data(),
      input_pack.device_tensor_scale_ptr(),
      input_expert_tensor_scales,
      launch_plan->num_non_exiting_ctas(),
      launch_plan->cta_expert_ids(),
      launch_plan->cta_row_starts(),
      launch_plan->cta_valid_rows(),
      weights,
      output_rows_per_expert,
      output);
  return CheckCuda(cudaGetLastError());
}

[[maybe_unused]] bool LaunchPlannedPackedInputMatVecRelu2ExpertScales(
    const DeviceNvfp4Matrix& input_pack,
    const float* input_expert_tensor_scales,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output_expert_scales) {
  if (!input_pack.valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output_expert_scales == nullptr) {
    return false;
  }
  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(launch_plan->n_experts(), active_selection_count);
  if (!current_cta_capacity.has_value() ||
      *current_cta_capacity == 0 ||
      *current_cta_capacity > launch_plan->cta_capacity()) {
    return false;
  }
  const std::size_t output_row_tile_count =
      (output_rows_per_expert + static_cast<std::size_t>(kPlannedOutputTile) - 1u) /
      static_cast<std::size_t>(kPlannedOutputTile);
  if (output_row_tile_count == 0) {
    return false;
  }
  if (!CheckCuda(cudaMemset(
          reinterpret_cast<void*>(output_expert_scales),
          0,
          launch_plan->n_experts() * sizeof(float)))) {
    return false;
  }
  const dim3 block(kPlannedThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  Nvfp4LaunchPlannedPackedInputExpertMatVecRelu2MaxAbsKernel<<<grid, block>>>(
      input_pack.packed_data(),
      input_pack.block_scales_data(),
      input_pack.device_tensor_scale_ptr(),
      input_expert_tensor_scales,
      launch_plan->num_non_exiting_ctas(),
      launch_plan->cta_idx_xy_to_batch_idx(),
      launch_plan->cta_idx_xy_to_mn_limit(),
      weights,
      output_rows_per_expert,
      reinterpret_cast<unsigned int*>(output_expert_scales));
  return CheckCuda(cudaGetLastError()) &&
         LaunchMaxBitsToTensorScales(
             reinterpret_cast<const unsigned int*>(output_expert_scales),
             launch_plan->n_experts(),
             output_expert_scales);
}

[[maybe_unused]] bool LaunchPlannedPackedInputMatVecRelu2Pack(
    const DeviceNvfp4Matrix& input_pack,
    const float* input_expert_tensor_scales,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    const float* output_expert_scales,
    DeviceNvfp4Matrix* output_pack) {
  if (!input_pack.valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output_expert_scales == nullptr ||
      output_pack == nullptr ||
      !output_pack->valid() ||
      output_pack->rows() < launch_plan->padded_row_capacity() ||
      output_pack->cols() != output_rows_per_expert) {
    return false;
  }
  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(launch_plan->n_experts(), active_selection_count);
  if (!current_cta_capacity.has_value() ||
      *current_cta_capacity == 0 ||
      *current_cta_capacity > launch_plan->cta_capacity()) {
    return false;
  }
  const std::size_t output_row_tile_count =
      (output_rows_per_expert + static_cast<std::size_t>(kPlannedOutputTile) - 1u) /
      static_cast<std::size_t>(kPlannedOutputTile);
  if (output_row_tile_count == 0 || !ClearDeviceNvfp4Matrix(output_pack)) {
    return false;
  }
  const dim3 block(kPlannedThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  Nvfp4LaunchPlannedPackedInputExpertMatVecRelu2PackKernel<<<grid, block>>>(
      input_pack.packed_data(),
      input_pack.block_scales_data(),
      input_pack.device_tensor_scale_ptr(),
      input_expert_tensor_scales,
      output_expert_scales,
      launch_plan->num_non_exiting_ctas(),
      launch_plan->cta_idx_xy_to_batch_idx(),
      launch_plan->cta_idx_xy_to_mn_limit(),
      weights,
      output_rows_per_expert,
      output_pack->rows(),
      output_pack->scale_layout(),
      const_cast<std::uint8_t*>(output_pack->packed_data()),
      const_cast<std::uint8_t*>(output_pack->block_scales_data()),
      const_cast<std::uint8_t*>(output_pack->matmul_block_scales_data()));
  return CheckCuda(cudaGetLastError());
}

bool LaunchReduceSelectionOutputs(
    const float* grouped_output,
    const int* selection_to_sorted,
    const int* sorted_to_permuted,
    const float* selected_weights,
    std::size_t token_count,
    std::size_t top_k,
    std::size_t hidden_size,
    float* output,
    float* routed_output) {
  const std::size_t count = token_count * hidden_size;
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1u) / block.x));
  ReduceSelectionOutputsKernel<<<grid, block>>>(
      grouped_output,
      selection_to_sorted,
      sorted_to_permuted,
      selected_weights,
      token_count,
      top_k,
      hidden_size,
      output,
      routed_output);
  return CheckCuda(cudaGetLastError());
}

bool LaunchAccumulateSharedOutput(
    const float* shared_output,
    std::size_t count,
    float* output,
    float* shared_output_copy) {
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1u) / block.x));
  AccumulateSharedOutputKernel<<<grid, block>>>(
      shared_output,
      count,
      output,
      shared_output_copy);
  return CheckCuda(cudaGetLastError());
}

}  // namespace

bool RunGroupedNvfp4ExpertMatVec(
    const float* input,
    const int* expert_offsets,
    std::size_t n_experts,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  if (input == nullptr ||
      expert_offsets == nullptr ||
      n_experts == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output == nullptr) {
    return false;
  }
  return LaunchGroupedMatVec(
      input,
      expert_offsets,
      n_experts,
      weights,
      output_rows_per_expert,
      output);
}

bool RunLaunchPlannedNvfp4ExpertMatVec(
    const float* input,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  if (input == nullptr ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output == nullptr) {
    return false;
  }
  return LaunchPlannedMatVec(
      input,
      launch_plan,
      active_selection_count,
      weights,
      output_rows_per_expert,
      output);
}

bool RunLaunchPlannedNvfp4ExpertMatVecBf16(
    const float* input,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    __nv_bfloat16* output) {
  if (input == nullptr ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      active_selection_count == 0 ||
      weights == nullptr ||
      output_rows_per_expert == 0 ||
      output == nullptr) {
    return false;
  }
  return LaunchPlannedMatVecBf16(
      input,
      launch_plan,
      active_selection_count,
      weights,
      output_rows_per_expert,
      output);
}

bool RunFusedMoePrefill(const FusedMoePrefillParams& params) {
  const std::size_t selection_count = params.token_count * params.top_k;
  const float* fc1_input_expert_scales = params.fc1_expert_activation_scales;
  DeviceNvfp4Matrix* gemm1_output =
      params.gemm1_output != nullptr ? params.gemm1_output : params.fc2_grouped_pack;
  float* gemm1_output_scale =
      params.gemm1_output_scale != nullptr ? params.gemm1_output_scale
                                           : params.fc2_expert_activation_scales;
  float* activation_output_scale =
      params.activation_output_scale != nullptr ? params.activation_output_scale
                                                : gemm1_output_scale;
  const bool use_legacy_packed_fc1 =
      params.fc1_grouped_pack != nullptr && fc1_input_expert_scales != nullptr;
  const bool use_grouped_gemm1_output =
      gemm1_output != nullptr &&
      gemm1_output_scale != nullptr &&
      activation_output_scale != nullptr;
  const bool use_packed_fc2 =
      gemm1_output != nullptr && activation_output_scale != nullptr;

  if (params.token_count == 0 ||
      params.hidden_size == 0 ||
      params.hidden_size % fused_decode::kNvfp4BlockWidth != 0 ||
      params.routed_expert_intermediate_size == 0 ||
      params.routed_expert_intermediate_size % fused_decode::kNvfp4BlockWidth != 0 ||
      params.shared_expert_intermediate_size == 0 ||
      params.shared_expert_intermediate_size % fused_decode::kNvfp4BlockWidth != 0 ||
      params.n_routed_experts == 0 ||
      params.top_k == 0 ||
      params.top_k > params.n_routed_experts ||
      !ValidFusedNvfp4WeightView(params.shared_up) ||
      !ValidFusedNvfp4WeightView(params.shared_down) ||
      params.routed_up_device == nullptr ||
      params.routed_down_device == nullptr ||
      params.selected_indices == nullptr ||
      params.selected_weights == nullptr ||
      params.input == nullptr ||
      params.normalized == nullptr ||
      params.routing == nullptr ||
      params.launch_plan == nullptr ||
      params.routed_gather_scratch == nullptr ||
      (!use_grouped_gemm1_output && params.routed_up_scratch == nullptr) ||
      (use_grouped_gemm1_output && params.gemm1_output_bf16 == nullptr) ||
      params.shared_up_scratch == nullptr ||
      params.output == nullptr) {
    return false;
  }

  if (!params.routing->valid() ||
      params.routing->n_experts() != params.n_routed_experts ||
      params.routing->selection_count() < selection_count ||
      !params.launch_plan->valid() ||
      params.launch_plan->n_experts() != params.n_routed_experts ||
      params.launch_plan->selection_count() < selection_count) {
    return false;
  }

  if (params.token_count >
          (std::numeric_limits<std::size_t>::max() / params.top_k) ||
      params.token_count >
          (std::numeric_limits<std::size_t>::max() / params.hidden_size) ||
      params.token_count >
          static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()) ||
      params.shared_up.input_cols != params.hidden_size ||
      params.shared_up.output_rows != params.shared_expert_intermediate_size ||
      params.shared_down.input_cols != params.shared_expert_intermediate_size ||
      params.shared_down.output_rows != params.hidden_size) {
    return false;
  }

  const std::size_t token_hidden_count = params.token_count * params.hidden_size;
  const std::size_t shared_intermediate_count =
      params.token_count * params.shared_expert_intermediate_size;
  const auto current_padded_row_capacity =
      DeviceMoeLaunchPlan::PaddedRowCapacity(params.n_routed_experts, selection_count);
  if (!current_padded_row_capacity.has_value()) {
    return false;
  }

  if (!RunDeviceExpertRouting(
          params.selected_indices,
          params.selected_weights,
          params.token_count,
          params.top_k,
          params.routing) ||
      !BuildDeviceMoeLaunchPlan(*params.routing, selection_count, params.launch_plan) ||
      !LaunchZeroBuffer(params.output, token_hidden_count) ||
      !LaunchZeroBuffer(params.routed_output, token_hidden_count) ||
      !LaunchZeroBuffer(params.shared_output, token_hidden_count)) {
    return false;
  }

  if (!LaunchGatherRows(
          params.normalized,
          params.launch_plan->permuted_idx_to_token_idx(),
          params.launch_plan->total_num_padded_tokens(),
          params.launch_plan->padded_row_capacity(),
          params.token_count,
          params.hidden_size,
          params.routed_gather_scratch) ||
      (use_legacy_packed_fc1 && !use_grouped_gemm1_output &&
       (!LaunchComputeExpertActivationScales(
            params.routed_gather_scratch,
            params.launch_plan->expert_first_token_offsets(),
            params.n_routed_experts,
            params.hidden_size,
            const_cast<float*>(fc1_input_expert_scales)) ||
        !PackDeviceRowMajorFp32ToNvfp4PerExpert(
            params.routed_gather_scratch,
            params.launch_plan->padded_row_capacity(),
            params.hidden_size,
            params.launch_plan->expert_first_token_offsets(),
            params.n_routed_experts,
            fc1_input_expert_scales,
            params.fc1_grouped_pack))) ||
      (!use_legacy_packed_fc1 && !use_grouped_gemm1_output &&
       !LaunchQuantizeDequantizeRows(
           params.routed_gather_scratch,
           params.launch_plan->total_num_padded_tokens(),
           params.launch_plan->padded_row_capacity(),
           params.hidden_size))) {
    return false;
  }

  if (use_grouped_gemm1_output) {
    if (!LaunchZeroBf16Buffer(
            params.gemm1_output_bf16,
            params.launch_plan->padded_row_capacity() *
                params.routed_expert_intermediate_size) ||
        !RunLaunchPlannedNvfp4ExpertMatVecBf16(
            params.routed_gather_scratch,
            params.launch_plan,
            selection_count,
            params.routed_up_device,
            params.routed_expert_intermediate_size,
            params.gemm1_output_bf16) ||
        !LaunchComputeExpertActivationScalesBf16(
            params.gemm1_output_bf16,
            params.launch_plan->expert_first_token_offsets(),
            params.n_routed_experts,
            params.routed_expert_intermediate_size,
            activation_output_scale) ||
        !PackDeviceRowMajorBf16ToNvfp4PerExpert(
            params.gemm1_output_bf16,
            params.launch_plan->padded_row_capacity(),
            params.routed_expert_intermediate_size,
            params.launch_plan->expert_first_token_offsets(),
            params.n_routed_experts,
            activation_output_scale,
            gemm1_output) ||
        !LaunchPlannedPackedInputMatVec(
            *gemm1_output,
            activation_output_scale,
            params.launch_plan,
            selection_count,
            params.routed_down_device,
            params.hidden_size,
            params.routed_gather_scratch)) {
      return false;
    }
  } else {
    if (!LaunchZeroBuffer(
            params.routed_up_scratch,
            *current_padded_row_capacity * params.routed_expert_intermediate_size) ||
        !(use_legacy_packed_fc1
              ? LaunchPlannedPackedInputMatVec(
                    *params.fc1_grouped_pack,
                    fc1_input_expert_scales,
                    params.launch_plan,
                    selection_count,
                    params.routed_up_device,
                    params.routed_expert_intermediate_size,
                    params.routed_up_scratch)
              : RunLaunchPlannedNvfp4ExpertMatVec(
                    params.routed_gather_scratch,
                    params.launch_plan,
                    selection_count,
                    params.routed_up_device,
                    params.routed_expert_intermediate_size,
                    params.routed_up_scratch)) ||
        !LaunchRelu2Rows(
            params.routed_up_scratch,
            params.launch_plan->total_num_padded_tokens(),
            params.launch_plan->padded_row_capacity(),
            params.routed_expert_intermediate_size) ||
        (activation_output_scale != nullptr &&
         !LaunchComputeExpertActivationScales(
             params.routed_up_scratch,
             params.launch_plan->expert_first_token_offsets(),
             params.n_routed_experts,
             params.routed_expert_intermediate_size,
             activation_output_scale)) ||
        (gemm1_output != nullptr &&
         !PackDeviceRowMajorFp32ToNvfp4PerExpert(
             params.routed_up_scratch,
             params.launch_plan->padded_row_capacity(),
             params.routed_expert_intermediate_size,
             params.launch_plan->expert_first_token_offsets(),
             params.n_routed_experts,
             activation_output_scale,
             gemm1_output)) ||
        (gemm1_output == nullptr &&
         !LaunchQuantizeDequantizeRows(
             params.routed_up_scratch,
             params.launch_plan->total_num_padded_tokens(),
             params.launch_plan->padded_row_capacity(),
             params.routed_expert_intermediate_size))) {
      return false;
    }

    if (!(use_packed_fc2
              ? LaunchPlannedPackedInputMatVec(
                    *gemm1_output,
                    activation_output_scale,
                    params.launch_plan,
                    selection_count,
                    params.routed_down_device,
                    params.hidden_size,
                    params.routed_gather_scratch)
              : RunLaunchPlannedNvfp4ExpertMatVec(
                    params.routed_up_scratch,
                    params.launch_plan,
                    selection_count,
                    params.routed_down_device,
                    params.hidden_size,
                    params.routed_gather_scratch))) {
      return false;
    }
  }

  if (!LaunchReduceSelectionOutputs(
          params.routed_gather_scratch,
          params.routing->selection_to_sorted(),
          params.launch_plan->sorted_to_permuted_indices(),
          params.selected_weights,
          params.token_count,
          params.top_k,
          params.hidden_size,
          params.output,
          params.routed_output)) {
    return false;
  }

  if (!CheckCuda(cudaMemcpyAsync(
          params.routed_gather_scratch,
          params.normalized,
          token_hidden_count * sizeof(float),
          cudaMemcpyDeviceToDevice)) ||
      !LaunchQuantizeDequantizeRows(
          params.routed_gather_scratch,
          nullptr,
          params.token_count,
          params.hidden_size) ||
      !LaunchContiguousMatVec(
          params.routed_gather_scratch,
          params.token_count,
          params.shared_up,
          params.shared_up_scratch) ||
      !LaunchRelu2(params.shared_up_scratch, shared_intermediate_count) ||
      !LaunchQuantizeDequantizeRows(
          params.shared_up_scratch,
          nullptr,
          params.token_count,
          params.shared_expert_intermediate_size) ||
      !LaunchContiguousMatVec(
          params.shared_up_scratch,
          params.token_count,
          params.shared_down,
          params.routed_gather_scratch) ||
      !LaunchAccumulateSharedOutput(
          params.routed_gather_scratch,
          token_hidden_count,
          params.output,
          params.shared_output)) {
    return false;
  }

  return CheckCuda(cudaGetLastError());
}

}  // namespace nemotron
