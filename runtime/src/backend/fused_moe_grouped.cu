#include "nemotron/fused_moe_grouped.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "fused_decode_common.cuh"

namespace nemotron {
namespace {

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

template <typename T>
bool CopyDeviceBufferToHost(
    const T* device_data,
    std::size_t count,
    std::vector<T>* output) {
  if (device_data == nullptr || output == nullptr) {
    return false;
  }
  output->assign(count, T{});
  return count == 0 ||
         CheckCuda(cudaMemcpy(
             output->data(),
             device_data,
             count * sizeof(T),
             cudaMemcpyDeviceToHost));
}

bool ValidWeightArrayPointer(const FusedNvfp4WeightView* weights) {
  return weights != nullptr;
}

bool ValidConfig(const FusedGroupedMoeConfig& config) {
  return config.hidden_size > 0 &&
         config.routed_expert_intermediate_size > 0 &&
         config.n_routed_experts > 0 &&
         config.top_k > 0 &&
         config.tile_size > 0 &&
         (config.hidden_size % fused_decode::kNvfp4BlockWidth) == 0 &&
         (config.routed_expert_intermediate_size %
              fused_decode::kNvfp4BlockWidth) == 0 &&
         config.n_routed_experts <=
             static_cast<std::size_t>(std::numeric_limits<int>::max());
}

std::size_t CeilDiv(std::size_t numerator, std::size_t denominator) {
  return (numerator + denominator - 1) / denominator;
}

__device__ fused_decode::Nvfp4WeightView MakeDecodeWeightView(
    const FusedNvfp4WeightView& weight) {
  return fused_decode::Nvfp4WeightView{
      weight.packed_data,
      weight.block_scales_data,
      weight.tensor_scale_data,
      weight.output_rows,
      weight.input_cols,
  };
}

__global__ void FusedGroupedMoeKernel(
    FusedGroupedMoeParams params,
    FusedGroupedMoeConfig config) {
  extern __shared__ float shared_storage[];
  float* gathered_row = shared_storage;
  float* intermediate = gathered_row + config.hidden_size;

  const int active_slot = static_cast<int>(blockIdx.y);
  const int expert_id = params.active_expert_ids[active_slot];
  if (expert_id < 0) {
    return;
  }

  const int expert_begin = params.expert_offsets[expert_id];
  const int expert_end = params.expert_offsets[expert_id + 1];
  if (expert_begin < 0 || expert_end <= expert_begin) {
    return;
  }

  const std::size_t expert_token_count =
      static_cast<std::size_t>(expert_end - expert_begin);
  const std::size_t tile_begin =
      static_cast<std::size_t>(blockIdx.x) * config.tile_size;
  if (tile_begin >= expert_token_count) {
    return;
  }

  const fused_decode::Nvfp4WeightView up_view =
      MakeDecodeWeightView(params.routed_up[expert_id]);
  const fused_decode::Nvfp4WeightView down_view =
      MakeDecodeWeightView(params.routed_down[expert_id]);
  if (!up_view.valid() || !down_view.valid()) {
    return;
  }

  for (std::size_t tile_offset = 0; tile_offset < config.tile_size; ++tile_offset) {
    const std::size_t expert_token_offset = tile_begin + tile_offset;
    if (expert_token_offset >= expert_token_count) {
      break;
    }

    const std::size_t selection_index =
        static_cast<std::size_t>(expert_begin) + expert_token_offset;
    const int token_index = params.sorted_token_indices[selection_index];
    if (token_index < 0 ||
        static_cast<std::size_t>(token_index) >= params.token_count) {
      continue;
    }
    const float route_weight = params.sorted_token_weights[selection_index];
    const std::size_t token_row_offset =
        static_cast<std::size_t>(token_index) * config.hidden_size;

    for (std::size_t column = threadIdx.x; column < config.hidden_size;
         column += blockDim.x) {
      gathered_row[column] =
          __bfloat162float(params.normalized[token_row_offset + column]);
    }
    __syncthreads();

    for (std::size_t row = threadIdx.x;
         row < config.routed_expert_intermediate_size;
         row += blockDim.x) {
      intermediate[row] = fused_decode::Nvfp4RowMajorDot(gathered_row, up_view, row);
    }
    __syncthreads();

    for (std::size_t row = threadIdx.x;
         row < config.routed_expert_intermediate_size;
         row += blockDim.x) {
      intermediate[row] = fused_decode::Relu2(intermediate[row]);
    }
    __syncthreads();

    for (std::size_t column = threadIdx.x; column < config.hidden_size;
         column += blockDim.x) {
      const float contribution =
          fused_decode::Nvfp4RowMajorDot(intermediate, down_view, column);
      atomicAdd(
          params.output + token_row_offset + column,
          route_weight * contribution);
    }
    __syncthreads();
  }
}

}  // namespace

bool RunFusedGroupedMoe(
    const FusedGroupedMoeParams& params,
    const FusedGroupedMoeConfig& config) {
  if (!ValidConfig(config) ||
      params.token_count == 0 ||
      params.selection_count == 0 ||
      params.normalized == nullptr ||
      params.expert_offsets == nullptr ||
      params.sorted_token_indices == nullptr ||
      params.sorted_token_weights == nullptr ||
      params.active_expert_count == nullptr ||
      params.active_expert_ids == nullptr ||
      !ValidWeightArrayPointer(params.routed_up) ||
      !ValidWeightArrayPointer(params.routed_down) ||
      params.output == nullptr ||
      params.token_count >
          (std::numeric_limits<std::size_t>::max() / config.top_k) ||
      params.selection_count != (params.token_count * config.top_k)) {
    return false;
  }

  int active_expert_count = 0;
  if (!CheckCuda(cudaMemcpy(
          &active_expert_count,
          params.active_expert_count,
          sizeof(active_expert_count),
          cudaMemcpyDeviceToHost)) ||
      active_expert_count < 0 ||
      static_cast<std::size_t>(active_expert_count) > config.n_routed_experts) {
    return false;
  }

  if (!CheckCuda(cudaMemset(
          params.output,
          0,
          params.token_count * config.hidden_size * sizeof(float)))) {
    return false;
  }

  if (active_expert_count == 0) {
    return true;
  }

  std::vector<int> expert_offsets_host;
  if (!CopyDeviceBufferToHost(
          params.expert_offsets,
          config.n_routed_experts + 1,
          &expert_offsets_host) ||
      expert_offsets_host.size() != config.n_routed_experts + 1 ||
      expert_offsets_host.front() != 0 ||
      expert_offsets_host.back() < 0 ||
      static_cast<std::size_t>(expert_offsets_host.back()) > params.selection_count) {
    return false;
  }

  std::size_t max_expert_tokens = 0;
  for (std::size_t expert_index = 0; expert_index < config.n_routed_experts; ++expert_index) {
    const int expert_begin = expert_offsets_host[expert_index];
    const int expert_end = expert_offsets_host[expert_index + 1];
    if (expert_begin > expert_end || expert_begin < 0) {
      return false;
    }
    max_expert_tokens = std::max(
        max_expert_tokens,
        static_cast<std::size_t>(expert_end - expert_begin));
  }

  if (max_expert_tokens == 0) {
    return true;
  }

  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(CeilDiv(max_expert_tokens, config.tile_size)),
      static_cast<unsigned int>(active_expert_count));
  const std::size_t shared_storage_bytes =
      (config.hidden_size + config.routed_expert_intermediate_size) *
      sizeof(float);

  FusedGroupedMoeKernel<<<grid, block, shared_storage_bytes>>>(params, config);
  return CheckCuda(cudaGetLastError());
}

}  // namespace nemotron
