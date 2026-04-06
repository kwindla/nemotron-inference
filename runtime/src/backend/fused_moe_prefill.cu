#include "nemotron/fused_moe_prefill.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>

#include "fused_decode_common.cuh"

namespace nemotron {
namespace {

constexpr int kMaxSelectedExperts = 32;
constexpr int kMaxRoutedExperts = 1024;

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

bool ValidFusedNvfp4WeightView(const FusedNvfp4WeightView& weight) {
  return weight.packed_data != nullptr &&
         weight.block_scales_data != nullptr &&
         weight.matmul_block_scales_data != nullptr &&
         weight.tensor_scale_data != nullptr &&
         weight.output_rows > 0 &&
         weight.input_cols > 0;
}

__device__ fused_decode::Nvfp4WeightView MakeDeviceWeightView(
    const FusedNvfp4WeightView& weight) {
  return fused_decode::Nvfp4WeightView{
      weight.packed_data,
      weight.block_scales_data,
      weight.tensor_scale_data,
      weight.output_rows,
      weight.input_cols,
  };
}

__global__ void FusedMoePrefillKernel(FusedMoePrefillParams params) {
  extern __shared__ float shared_storage[];
  __shared__ int selected_indices[kMaxSelectedExperts];
  __shared__ float selected_weights[kMaxSelectedExperts];

  const std::size_t expert_buffer_size =
      params.routed_expert_intermediate_size > params.shared_expert_intermediate_size
          ? params.routed_expert_intermediate_size
          : params.shared_expert_intermediate_size;
  float* expert_buffer = shared_storage;
  float* quantized_input = expert_buffer + expert_buffer_size;

  const std::size_t token_index = static_cast<std::size_t>(blockIdx.x);
  if (token_index >= params.token_count) {
    return;
  }

  const std::size_t tid = static_cast<std::size_t>(threadIdx.x);
  const std::size_t hidden_offset = token_index * params.hidden_size;
  const float* normalized_row = params.normalized + hidden_offset;
  float* output_row = params.output + hidden_offset;
  float* routed_output_row =
      params.routed_output != nullptr ? params.routed_output + hidden_offset : nullptr;
  float* shared_output_row =
      params.shared_output != nullptr ? params.shared_output + hidden_offset : nullptr;
  const std::size_t selection_offset = token_index * params.top_k;

  for (std::size_t slot = tid; slot < params.top_k; slot += blockDim.x) {
    selected_indices[slot] = params.selected_indices[selection_offset + slot];
    selected_weights[slot] = params.selected_weights[selection_offset + slot];
  }
  fused_decode::QuantizeDequantizeNvfp4Row(
      normalized_row,
      quantized_input,
      params.hidden_size);
  __syncthreads();

  for (std::size_t column = tid; column < params.hidden_size; column += blockDim.x) {
    output_row[column] = 0.0f;
    if (routed_output_row != nullptr) {
      routed_output_row[column] = 0.0f;
    }
    if (shared_output_row != nullptr) {
      shared_output_row[column] = 0.0f;
    }
  }
  __syncthreads();

  for (std::size_t slot = 0; slot < params.top_k; ++slot) {
    const int expert_index = selected_indices[slot];
    if (expert_index < 0 ||
        static_cast<std::size_t>(expert_index) >= params.n_routed_experts) {
      continue;
    }

    const fused_decode::Nvfp4WeightView up_view =
        MakeDeviceWeightView(params.routed_up[expert_index]);
    const fused_decode::Nvfp4WeightView down_view =
        MakeDeviceWeightView(params.routed_down[expert_index]);

    for (std::size_t row = tid;
         row < params.routed_expert_intermediate_size;
         row += blockDim.x) {
      expert_buffer[row] = fused_decode::Nvfp4RowMajorDot(quantized_input, up_view, row);
    }
    __syncthreads();

    for (std::size_t row = tid;
         row < params.routed_expert_intermediate_size;
         row += blockDim.x) {
      expert_buffer[row] = fused_decode::Relu2(expert_buffer[row]);
    }
    __syncthreads();
    fused_decode::QuantizeDequantizeNvfp4Row(
        expert_buffer,
        expert_buffer,
        params.routed_expert_intermediate_size);
    __syncthreads();

    for (std::size_t column = tid; column < params.hidden_size; column += blockDim.x) {
      const float contribution =
          fused_decode::Nvfp4RowMajorDot(expert_buffer, down_view, column);
      const float weighted_contribution =
          __fmul_rn(selected_weights[slot], contribution);
      output_row[column] = __fadd_rn(output_row[column], weighted_contribution);
      if (routed_output_row != nullptr) {
        routed_output_row[column] = __fadd_rn(
            routed_output_row[column],
            weighted_contribution);
      }
    }
    __syncthreads();
  }

  const fused_decode::Nvfp4WeightView shared_up_view =
      MakeDeviceWeightView(params.shared_up);
  const fused_decode::Nvfp4WeightView shared_down_view =
      MakeDeviceWeightView(params.shared_down);
  for (std::size_t row = tid;
       row < params.shared_expert_intermediate_size;
       row += blockDim.x) {
    expert_buffer[row] = fused_decode::Nvfp4RowMajorDot(
        quantized_input,
        shared_up_view,
        row);
  }
  __syncthreads();

  for (std::size_t row = tid;
       row < params.shared_expert_intermediate_size;
       row += blockDim.x) {
    expert_buffer[row] = fused_decode::Relu2(expert_buffer[row]);
  }
  __syncthreads();
  fused_decode::QuantizeDequantizeNvfp4Row(
      expert_buffer,
      expert_buffer,
      params.shared_expert_intermediate_size);
  __syncthreads();

  for (std::size_t column = tid; column < params.hidden_size; column += blockDim.x) {
    const float contribution =
        fused_decode::Nvfp4RowMajorDot(expert_buffer, shared_down_view, column);
    output_row[column] = __fadd_rn(output_row[column], contribution);
    if (shared_output_row != nullptr) {
      shared_output_row[column] = contribution;
    }
  }
}

}  // namespace

bool RunFusedMoePrefill(const FusedMoePrefillParams& params) {
  if (params.token_count == 0 ||
      params.hidden_size == 0 ||
      params.hidden_size % fused_decode::kNvfp4BlockWidth != 0 ||
      params.routed_expert_intermediate_size == 0 ||
      params.routed_expert_intermediate_size % fused_decode::kNvfp4BlockWidth != 0 ||
      params.shared_expert_intermediate_size == 0 ||
      params.shared_expert_intermediate_size % fused_decode::kNvfp4BlockWidth != 0 ||
      params.n_routed_experts == 0 ||
      params.n_routed_experts > kMaxRoutedExperts ||
      params.top_k == 0 ||
      params.top_k > params.n_routed_experts ||
      params.top_k > kMaxSelectedExperts ||
      !ValidFusedNvfp4WeightView(params.shared_up) ||
      !ValidFusedNvfp4WeightView(params.shared_down) ||
      params.routed_up == nullptr ||
      params.routed_down == nullptr ||
      params.selected_indices == nullptr ||
      params.selected_weights == nullptr ||
      params.input == nullptr ||
      params.normalized == nullptr ||
      params.output == nullptr) {
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

  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(params.token_count));
  const std::size_t shared_bytes =
      (std::max(params.routed_expert_intermediate_size,
                params.shared_expert_intermediate_size) +
       params.hidden_size) *
      sizeof(float);
  FusedMoePrefillKernel<<<grid, block, shared_bytes>>>(params);
  return CheckCuda(cudaGetLastError());
}

}  // namespace nemotron
