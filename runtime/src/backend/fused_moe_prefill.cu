#include "nemotron/fused_moe_prefill.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>

#include "fused_decode_common.cuh"

namespace nemotron {
namespace {

constexpr int kGroupedTokenTile = 8;
constexpr int kPlannedOutputTile = 4;
constexpr int kPlannedPairsPerStep = 32;
constexpr int kPlannedThreadsPerBlock = kPlannedOutputTile * 32;

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

__global__ void ZeroBufferKernel(float* data, std::size_t count) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  data[index] = 0.0f;
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

__global__ void Nvfp4GroupedExpertMatVecRowsKernel(
    const float* input,
    const int* expert_offsets,
    std::size_t n_experts,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  __shared__ double partial_sums[kGroupedTokenTile * fused_decode::kThreadsPerBlock];

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
    double accum[kGroupedTokenTile] = {0.0};

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
        accum[tile_row] += static_cast<double>(input_row[col]) * static_cast<double>(w0);
        accum[tile_row] += static_cast<double>(input_row[col + 1]) * static_cast<double>(w1);
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
               output_row] = static_cast<float>(partial_sums[tile_row * blockDim.x]);
      }
    }
    __syncthreads();
  }
}

__global__ void Nvfp4LaunchPlannedExpertMatVecRowsKernel(
    const float* input,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  __shared__ float input_tile[kGroupedTokenTile][kPlannedPairsPerStep * 2];

  const int cta_index = static_cast<int>(blockIdx.x);
  const int warp_index = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count) {
    return;
  }

  const int output_row_tile = static_cast<int>(blockIdx.y);
  const int output_row = output_row_tile * kPlannedOutputTile + warp_index;
  const int expert_index = cta_expert_ids[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  const int padded_row_base =
      cta_index * static_cast<int>(kMoeLaunchPlanTokenTile);
  if (expert_index < 0 ||
      warp_index >= kPlannedOutputTile ||
      static_cast<std::size_t>(output_row) >= output_rows_per_expert ||
      valid_rows <= 0) {
    return;
  }

  const fused_decode::Nvfp4WeightView weight =
      MakeDeviceWeightView(weights[expert_index]);
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t packed_row_offset = static_cast<std::size_t>(output_row) * pairs_per_row;
  const std::size_t scale_row_offset = static_cast<std::size_t>(output_row) * blocks_per_row;
  const float tensor_scale = *weight.tensor_scale_data;
  double accum[kGroupedTokenTile] = {0.0};

  for (std::size_t pair_base = 0; pair_base < pairs_per_row;
       pair_base += static_cast<std::size_t>(kPlannedPairsPerStep)) {
    const std::size_t remaining_pairs = pairs_per_row - pair_base;
    const int pairs_this_step = static_cast<int>(
        remaining_pairs < static_cast<std::size_t>(kPlannedPairsPerStep)
            ? remaining_pairs
            : static_cast<std::size_t>(kPlannedPairsPerStep));
    const int values_this_step = valid_rows * pairs_this_step * 2;
    for (int linear_index = static_cast<int>(threadIdx.x);
         linear_index < values_this_step;
         linear_index += static_cast<int>(blockDim.x)) {
      const int token_index = linear_index / (pairs_this_step * 2);
      const int within_token = linear_index % (pairs_this_step * 2);
      const int pair_offset = within_token / 2;
      const int value_offset = within_token % 2;
      const std::size_t input_row =
          static_cast<std::size_t>(padded_row_base + token_index);
      const std::size_t col =
          (pair_base + static_cast<std::size_t>(pair_offset)) * 2u +
          static_cast<std::size_t>(value_offset);
      input_tile[token_index][within_token] =
          input[input_row * weight.input_cols + col];
    }
    __syncthreads();

    const std::size_t pair_index = pair_base + static_cast<std::size_t>(lane);
    if (lane < pairs_this_step) {
      const std::size_t block = pair_index / 8u;
      const float block_scale =
          fused_decode::DecodeFp8(weight.block_scales_data[scale_row_offset + block]) *
          tensor_scale;
      const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
      const float w0 = fused_decode::DecodeFp4(packed & 0x0Fu) * block_scale;
      const float w1 = fused_decode::DecodeFp4((packed >> 4) & 0x0Fu) * block_scale;
      for (int token_index = 0; token_index < valid_rows; ++token_index) {
        const int value_index = lane * 2;
        accum[token_index] +=
            static_cast<double>(input_tile[token_index][value_index]) *
            static_cast<double>(w0);
        accum[token_index] +=
            static_cast<double>(input_tile[token_index][value_index + 1]) *
            static_cast<double>(w1);
      }
    }
    __syncthreads();
  }

  for (int token_index = 0; token_index < valid_rows; ++token_index) {
    for (int stride = 16; stride > 0; stride >>= 1) {
      accum[token_index] += __shfl_down_sync(0xffffffffu, accum[token_index], stride);
    }
  }

  if (lane == 0) {
    for (int token_index = 0; token_index < valid_rows; ++token_index) {
      const std::size_t input_row =
          static_cast<std::size_t>(padded_row_base + token_index);
      output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row)] =
          static_cast<float>(accum[token_index]);
    }
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
  __shared__ double partial_sums[kGroupedTokenTile * fused_decode::kThreadsPerBlock];

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
    double accum[kGroupedTokenTile] = {0.0};

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
        accum[tile_row] += static_cast<double>(input_row[col]) * static_cast<double>(w0);
        accum[tile_row] += static_cast<double>(input_row[col + 1]) * static_cast<double>(w1);
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
            static_cast<float>(partial_sums[tile_row * blockDim.x]);
      }
    }
    __syncthreads();
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
    const int padded_index = sorted_to_permuted[sorted_index];
    if (padded_index < 0) {
      continue;
    }
    const std::size_t grouped_offset =
        static_cast<std::size_t>(padded_index) * hidden_size + hidden_index;
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
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(weight.output_rows));
  Nvfp4MatVecRowsKernel<false><<<grid, block>>>(
      input,
      input_row_count,
      nullptr,
      0,
      MakeDeviceWeightView(weight),
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
  const dim3 block(kPlannedThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(*current_cta_capacity),
      static_cast<unsigned int>(output_row_tile_count));
  Nvfp4LaunchPlannedExpertMatVecRowsKernel<<<grid, block>>>(
      input,
      launch_plan->cta_count(),
      launch_plan->cta_expert_ids(),
      launch_plan->cta_valid_rows(),
      weights,
      output_rows_per_expert,
      output);
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

bool RunFusedMoePrefill(const FusedMoePrefillParams& params) {
  const std::size_t selection_count = params.token_count * params.top_k;

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
      params.routed_up_scratch == nullptr ||
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
  const auto padded_selection_capacity =
      DeviceMoeLaunchPlan::PaddedRowCapacity(params.n_routed_experts, selection_count);
  if (!padded_selection_capacity.has_value()) {
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
      !LaunchZeroBuffer(params.shared_output, token_hidden_count) ||
      !LaunchGatherRows(
          params.normalized,
          params.launch_plan->permuted_token_indices(),
          params.launch_plan->total_padded_rows(),
          *padded_selection_capacity,
          params.token_count,
          params.hidden_size,
          params.routed_gather_scratch) ||
      !LaunchQuantizeDequantizeRows(
          params.routed_gather_scratch,
          params.launch_plan->total_padded_rows(),
          *padded_selection_capacity,
          params.hidden_size)) {
    return false;
  }

  if (!RunLaunchPlannedNvfp4ExpertMatVec(
          params.routed_gather_scratch,
          params.launch_plan,
          selection_count,
          params.routed_up_device,
          params.routed_expert_intermediate_size,
          params.routed_up_scratch)) {
    return false;
  }

  if (!LaunchRelu2Rows(
          params.routed_up_scratch,
          params.launch_plan->total_padded_rows(),
          *padded_selection_capacity,
          params.routed_expert_intermediate_size) ||
      !LaunchQuantizeDequantizeRows(
          params.routed_up_scratch,
          params.launch_plan->total_padded_rows(),
          *padded_selection_capacity,
          params.routed_expert_intermediate_size)) {
    return false;
  }

  if (!RunLaunchPlannedNvfp4ExpertMatVec(
          params.routed_up_scratch,
          params.launch_plan,
          selection_count,
          params.routed_down_device,
          params.hidden_size,
          params.routed_gather_scratch)) {
    return false;
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
