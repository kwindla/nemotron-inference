#include "nemotron/fused_moe_decode.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "fused_decode_common.cuh"

namespace nemotron {
namespace {

constexpr int kMaxSelectedExperts = 32;
constexpr int kMaxExpertGroups = 64;
constexpr int kMaxRoutedExperts = 1024;
constexpr int kWarpExpertSelectionThreads = 32;
constexpr int kExpertsPerWarpSelectionThread = 4;
constexpr int kWarpExpertSelectionCapacity =
    kWarpExpertSelectionThreads * kExpertsPerWarpSelectionThread;
constexpr unsigned int kFullWarpMask = 0xffffffffu;

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

bool EnvEnabled(const char* env_var) {
  const char* value = std::getenv(env_var);
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

__device__ __forceinline__ bool ExpertSelectionCandidateIsBetter(
    float candidate_value,
    int candidate_expert,
    float current_value,
    int current_expert) {
  if (candidate_expert < 0) {
    return false;
  }
  if (candidate_value > current_value) {
    return true;
  }
  if (candidate_value < current_value) {
    return false;
  }
  if (!(candidate_value == current_value)) {
    return false;
  }
  if (current_expert < 0) {
    return false;
  }
  return candidate_expert < current_expert;
}

__device__ void SelectTopExpertsOneToken(
    const FusedMoeDirectLayerParams& params,
    const float* router_logits,
    int* selected_indices,
    float* selected_weights) {
  float scores[kMaxRoutedExperts];
  float choice_scores[kMaxRoutedExperts];
  float group_scores[kMaxExpertGroups];
  int ranked_groups[kMaxExpertGroups];
  float best_values[kMaxSelectedExperts];
  int best_indices[kMaxSelectedExperts];

  for (std::size_t i = 0; i < params.n_routed_experts; ++i) {
    scores[i] = fused_decode::Sigmoid(router_logits[i]);
    choice_scores[i] = scores[i] + params.correction_bias[i];
  }

  const std::size_t group_size = params.n_routed_experts / params.n_group;
  for (std::size_t group = 0; group < params.n_group; ++group) {
    float top1 = -INFINITY;
    float top2 = -INFINITY;
    for (std::size_t i = 0; i < group_size; ++i) {
      const float value = choice_scores[group * group_size + i];
      if (value > top1) {
        top2 = top1;
        top1 = value;
      } else if (value > top2) {
        top2 = value;
      }
    }
    if (!isfinite(top2)) {
      top2 = top1;
    }
    group_scores[group] = top1 + top2;
    ranked_groups[group] = static_cast<int>(group);
  }

  for (std::size_t i = 0; i < params.n_group; ++i) {
    for (std::size_t j = i + 1; j < params.n_group; ++j) {
      if (group_scores[ranked_groups[j]] > group_scores[ranked_groups[i]]) {
        const int swap_index = ranked_groups[i];
        ranked_groups[i] = ranked_groups[j];
        ranked_groups[j] = swap_index;
      }
    }
  }

  bool group_selected[kMaxExpertGroups];
  for (std::size_t group = 0; group < params.n_group; ++group) {
    group_selected[group] = false;
  }
  const std::size_t selected_group_count =
      params.topk_group < params.n_group ? params.topk_group : params.n_group;
  for (std::size_t i = 0; i < selected_group_count; ++i) {
    group_selected[ranked_groups[i]] = true;
  }

  for (std::size_t slot = 0; slot < params.top_k; ++slot) {
    best_values[slot] = -INFINITY;
    best_indices[slot] = -1;
  }
  for (std::size_t expert = 0; expert < params.n_routed_experts; ++expert) {
    const std::size_t group = expert / group_size;
    const float value = group_selected[group] ? choice_scores[expert] : 0.0f;
    std::size_t insert_slot = params.top_k;
    for (std::size_t slot = 0; slot < params.top_k; ++slot) {
      if (value > best_values[slot]) {
        insert_slot = slot;
        break;
      }
    }
    if (insert_slot == params.top_k) {
      continue;
    }
    for (std::size_t slot = params.top_k; slot > insert_slot + 1; --slot) {
      best_values[slot - 1] = best_values[slot - 2];
      best_indices[slot - 1] = best_indices[slot - 2];
    }
    best_values[insert_slot] = value;
    best_indices[insert_slot] = static_cast<int>(expert);
  }

  float weight_sum = 0.0f;
  for (std::size_t slot = 0; slot < params.top_k; ++slot) {
    selected_indices[slot] = best_indices[slot];
    const float weight =
        best_indices[slot] >= 0 ? scores[best_indices[slot]] : 0.0f;
    selected_weights[slot] = weight;
    weight_sum += weight;
  }

  if (params.norm_topk_prob) {
    const float denominator = weight_sum + 1.0e-20f;
    for (std::size_t slot = 0; slot < params.top_k; ++slot) {
      selected_weights[slot] /= denominator;
    }
  }
  for (std::size_t slot = 0; slot < params.top_k; ++slot) {
    selected_weights[slot] *= params.routed_scaling_factor;
  }
}

__global__ void FusedMoeDirectDecodeKernel(
    FusedMoeDirectLayerParams params,
    const float* input,
    const float* normalized,
    const float* router_logits,
    float* output) {
  extern __shared__ float shared_storage[];
  __shared__ int selected_indices[kMaxSelectedExperts];
  __shared__ float selected_weights[kMaxSelectedExperts];
  float* expert_buffer = shared_storage;
  float* quantized_input =
      expert_buffer +
      max(params.routed_expert_intermediate_size, params.shared_expert_intermediate_size);

  const std::size_t tid = threadIdx.x;
  if (tid == 0) {
    if (params.selected_indices != nullptr && params.selected_weights != nullptr) {
      for (std::size_t slot = 0; slot < params.top_k; ++slot) {
        selected_indices[slot] = params.selected_indices[slot];
        selected_weights[slot] = params.selected_weights[slot];
      }
    } else {
      SelectTopExpertsOneToken(params, router_logits, selected_indices, selected_weights);
    }
    fused_decode::QuantizeDequantizeNvfp4Row(
        normalized,
        quantized_input,
        params.hidden_size);
  }
  __syncthreads();

  for (std::size_t column = tid; column < params.hidden_size; column += blockDim.x) {
    output[column] = input[column];
    if (params.routed_output != nullptr) {
      params.routed_output[column] = 0.0f;
    }
    if (params.shared_output != nullptr) {
      params.shared_output[column] = 0.0f;
    }
  }
  __syncthreads();

  for (std::size_t slot = 0; slot < params.top_k; ++slot) {
    const int expert_index = selected_indices[slot];
    if (expert_index < 0) {
      continue;
    }
    const fused_decode::Nvfp4WeightView up_view{
        params.routed_up[expert_index].packed_data,
        params.routed_up[expert_index].block_scales_data,
        params.routed_up[expert_index].tensor_scale_data,
        params.routed_up[expert_index].output_rows,
        params.routed_up[expert_index].input_cols,
    };
    const fused_decode::Nvfp4WeightView down_view{
        params.routed_down[expert_index].packed_data,
        params.routed_down[expert_index].block_scales_data,
        params.routed_down[expert_index].tensor_scale_data,
        params.routed_down[expert_index].output_rows,
        params.routed_down[expert_index].input_cols,
    };
    for (std::size_t row = tid; row < params.routed_expert_intermediate_size;
         row += blockDim.x) {
      expert_buffer[row] = fused_decode::Nvfp4RowMajorDot(quantized_input, up_view, row);
    }
    __syncthreads();
    if (tid == 0) {
      for (std::size_t row = 0; row < params.routed_expert_intermediate_size; ++row) {
        expert_buffer[row] = fused_decode::Relu2(expert_buffer[row]);
      }
      fused_decode::QuantizeDequantizeNvfp4Row(
          expert_buffer,
          expert_buffer,
          params.routed_expert_intermediate_size);
    }
    __syncthreads();
    for (std::size_t column = tid; column < params.hidden_size; column += blockDim.x) {
      const float contribution =
          fused_decode::Nvfp4RowMajorDot(expert_buffer, down_view, column);
      const float weighted_contribution = __fmul_rn(selected_weights[slot], contribution);
      output[column] = __fadd_rn(output[column], weighted_contribution);
      if (params.routed_output != nullptr) {
        params.routed_output[column] = __fadd_rn(
            params.routed_output[column],
            weighted_contribution);
      }
    }
    __syncthreads();
  }

  const fused_decode::Nvfp4WeightView shared_up_view{
      params.shared_up.packed_data,
      params.shared_up.block_scales_data,
      params.shared_up.tensor_scale_data,
      params.shared_up.output_rows,
      params.shared_up.input_cols,
  };
  const fused_decode::Nvfp4WeightView shared_down_view{
      params.shared_down.packed_data,
      params.shared_down.block_scales_data,
      params.shared_down.tensor_scale_data,
      params.shared_down.output_rows,
      params.shared_down.input_cols,
  };
  for (std::size_t row = tid; row < params.shared_expert_intermediate_size;
       row += blockDim.x) {
    expert_buffer[row] = fused_decode::Nvfp4RowMajorDot(quantized_input, shared_up_view, row);
  }
  __syncthreads();
  if (tid == 0) {
    for (std::size_t row = 0; row < params.shared_expert_intermediate_size; ++row) {
      expert_buffer[row] = fused_decode::Relu2(expert_buffer[row]);
    }
    fused_decode::QuantizeDequantizeNvfp4Row(
        expert_buffer,
        expert_buffer,
        params.shared_expert_intermediate_size);
  }
  __syncthreads();
  for (std::size_t column = tid; column < params.hidden_size; column += blockDim.x) {
    const float shared_contribution =
        fused_decode::Nvfp4RowMajorDot(expert_buffer, shared_down_view, column);
    output[column] = __fadd_rn(output[column], shared_contribution);
    if (params.shared_output != nullptr) {
      params.shared_output[column] = shared_contribution;
    }
  }
}

__global__ void DeviceExpertSelectionLegacyKernel(
    FusedMoeDirectLayerParams params,
    const float* router_logits,
    int* selected_indices,
    float* selected_weights) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  SelectTopExpertsOneToken(params, router_logits, selected_indices, selected_weights);
}

__global__ void DeviceExpertSelectionKernel(
    FusedMoeDirectLayerParams params,
    const float* router_logits,
    int* selected_indices,
    float* selected_weights) {
  if (blockIdx.x != 0 || threadIdx.x >= kWarpExpertSelectionThreads) {
    return;
  }

  const int lane = static_cast<int>(threadIdx.x);
  const int n_routed_experts = static_cast<int>(params.n_routed_experts);
  float raw_scores[kExpertsPerWarpSelectionThread];
  float choice_scores[kExpertsPerWarpSelectionThread];
  int expert_indices[kExpertsPerWarpSelectionThread];

#pragma unroll
  for (int local_slot = 0; local_slot < kExpertsPerWarpSelectionThread; ++local_slot) {
    const int expert = lane * kExpertsPerWarpSelectionThread + local_slot;
    if (expert < n_routed_experts) {
      expert_indices[local_slot] = expert;
      const float raw_score = fused_decode::Sigmoid(router_logits[expert]);
      raw_scores[local_slot] = raw_score;
      choice_scores[local_slot] = raw_score + params.correction_bias[expert];
    } else {
      expert_indices[local_slot] = -1;
      raw_scores[local_slot] = 0.0f;
      choice_scores[local_slot] = -INFINITY;
    }
  }

  for (int selected_slot = 0; selected_slot < static_cast<int>(params.top_k);
       ++selected_slot) {
    float local_best_choice = -INFINITY;
    float local_best_raw = 0.0f;
    int local_best_expert = -1;
    int local_best_lane = lane;
    int local_best_slot = -1;

#pragma unroll
    for (int local_slot = 0; local_slot < kExpertsPerWarpSelectionThread; ++local_slot) {
      if (ExpertSelectionCandidateIsBetter(
              choice_scores[local_slot],
              expert_indices[local_slot],
              local_best_choice,
              local_best_expert)) {
        local_best_choice = choice_scores[local_slot];
        local_best_raw = raw_scores[local_slot];
        local_best_expert = expert_indices[local_slot];
        local_best_slot = local_slot;
      }
    }

    for (int offset = kWarpExpertSelectionThreads / 2; offset > 0; offset >>= 1) {
      const float other_choice =
          __shfl_down_sync(kFullWarpMask, local_best_choice, offset);
      const float other_raw = __shfl_down_sync(kFullWarpMask, local_best_raw, offset);
      const int other_expert = __shfl_down_sync(kFullWarpMask, local_best_expert, offset);
      const int other_lane = __shfl_down_sync(kFullWarpMask, local_best_lane, offset);
      const int other_slot = __shfl_down_sync(kFullWarpMask, local_best_slot, offset);
      if (ExpertSelectionCandidateIsBetter(
              other_choice,
              other_expert,
              local_best_choice,
              local_best_expert)) {
        local_best_choice = other_choice;
        local_best_raw = other_raw;
        local_best_expert = other_expert;
        local_best_lane = other_lane;
        local_best_slot = other_slot;
      }
    }

    const int winning_expert = __shfl_sync(kFullWarpMask, local_best_expert, 0);
    const float winning_raw = __shfl_sync(kFullWarpMask, local_best_raw, 0);
    const int winning_lane = __shfl_sync(kFullWarpMask, local_best_lane, 0);
    const int winning_local_slot = __shfl_sync(kFullWarpMask, local_best_slot, 0);

    if (lane == winning_lane && winning_local_slot >= 0) {
      expert_indices[winning_local_slot] = -1;
      choice_scores[winning_local_slot] = -INFINITY;
    }

    if (lane == 0) {
      selected_indices[selected_slot] = winning_expert;
      selected_weights[selected_slot] = winning_expert >= 0 ? winning_raw : 0.0f;
    }
  }

  if (lane == 0) {
    float weight_sum = 0.0f;
    for (int slot = 0; slot < static_cast<int>(params.top_k); ++slot) {
      weight_sum += selected_weights[slot];
    }
    if (params.norm_topk_prob) {
      const float denominator = weight_sum + 1.0e-20f;
      for (int slot = 0; slot < static_cast<int>(params.top_k); ++slot) {
        selected_weights[slot] /= denominator;
      }
    }
    for (int slot = 0; slot < static_cast<int>(params.top_k); ++slot) {
      selected_weights[slot] =
          selected_weights[slot] * params.routed_scaling_factor;
    }
  }
}

__global__ void AccumulateScaledByDeviceWeightKernel(
    const float* input,
    const float* scales_device,
    std::size_t scale_index,
    float* output,
    std::size_t count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  output[index] = __fadd_rn(output[index], __fmul_rn(input[index], scales_device[scale_index]));
}

}  // namespace

bool FusedMoeDecodeEnabled() {
  return EnvEnabled("NEMOTRON_FORWARD_FUSED_MOE_DECODE");
}

bool RunDeviceExpertSelection(
    const DeviceTensorFp32& router_logits,
    const DeviceTensorFp32& correction_bias,
    std::size_t n_routed_experts,
    std::size_t top_k,
    std::size_t n_group,
    std::size_t topk_group,
    float routed_scaling_factor,
    bool norm_topk_prob,
    int* selected_indices,
    float* selected_weights) {
  if (!router_logits.valid() ||
      !correction_bias.valid() ||
      selected_indices == nullptr ||
      selected_weights == nullptr ||
      router_logits.shape().size() != 2 ||
      router_logits.shape()[0] != 1 ||
      router_logits.shape()[1] != n_routed_experts ||
      correction_bias.shape().size() != 1 ||
      correction_bias.shape()[0] != n_routed_experts ||
      n_routed_experts == 0 ||
      n_routed_experts > kMaxRoutedExperts ||
      top_k == 0 ||
      top_k > n_routed_experts ||
      top_k > kMaxSelectedExperts ||
      n_group == 0 ||
      n_group > n_routed_experts ||
      n_group > kMaxExpertGroups ||
      topk_group == 0 ||
      (n_routed_experts % n_group) != 0) {
    return false;
  }

  FusedMoeDirectLayerParams params;
  params.n_routed_experts = n_routed_experts;
  params.top_k = top_k;
  params.n_group = n_group;
  params.topk_group = topk_group;
  params.routed_scaling_factor = routed_scaling_factor;
  params.norm_topk_prob = norm_topk_prob;
  params.correction_bias = correction_bias.data();

  const bool force_legacy = EnvEnabled("NEMOTRON_FORWARD_EXPERT_SELECT_LEGACY");
  const bool use_warp_selection =
      !force_legacy &&
      n_group == 1 &&
      topk_group == 1 &&
      n_routed_experts <= static_cast<std::size_t>(kWarpExpertSelectionCapacity);

  if (use_warp_selection) {
    DeviceExpertSelectionKernel<<<1, kWarpExpertSelectionThreads>>>(
        params,
        router_logits.data(),
        selected_indices,
        selected_weights);
  } else {
    DeviceExpertSelectionLegacyKernel<<<1, 1>>>(
        params,
        router_logits.data(),
        selected_indices,
        selected_weights);
  }
  return CheckCuda(cudaGetLastError());
}

bool AccumulateScaledFp32ByDeviceWeight(
    const DeviceTensorFp32& input,
    const float* scales_device,
    std::size_t scale_index,
    DeviceTensorFp32* output) {
  if (output == nullptr ||
      !input.valid() ||
      !output->valid() ||
      scales_device == nullptr ||
      input.shape() != output->shape()) {
    return false;
  }

  const std::size_t count = input.numel();
  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1) / block.x));
  AccumulateScaledByDeviceWeightKernel<<<grid, block>>>(
      input.data(),
      scales_device,
      scale_index,
      output->data(),
      count);
  return CheckCuda(cudaGetLastError());
}

bool RunFusedMoeDirectDecode(
    const FusedMoeDirectLayerParams& params,
    const DeviceTensorFp32& input,
    const DeviceTensorFp32& normalized,
    const DeviceTensorFp32& router_logits,
    DeviceTensorFp32* output) {
  if (!FusedMoeDecodeEnabled() ||
      !input.valid() ||
      !normalized.valid() ||
      !router_logits.valid() ||
      output == nullptr ||
      !output->valid() ||
      input.shape().size() != 2 ||
      normalized.shape() != input.shape() ||
      output->shape() != input.shape() ||
      router_logits.shape().size() != 2 ||
      input.shape()[0] != 1 ||
      router_logits.shape()[0] != 1 ||
      router_logits.shape()[1] != params.n_routed_experts ||
      params.hidden_size == 0 ||
      params.routed_expert_intermediate_size == 0 ||
      params.shared_expert_intermediate_size == 0 ||
      params.n_routed_experts == 0 ||
      params.top_k == 0 ||
      params.top_k > kMaxSelectedExperts ||
      params.n_group == 0 ||
      params.n_group > kMaxExpertGroups ||
      params.topk_group == 0 ||
      params.n_routed_experts > kMaxRoutedExperts ||
      params.routed_up == nullptr ||
      params.routed_down == nullptr ||
      params.correction_bias == nullptr) {
    return false;
  }

  if (input.shape()[1] != params.hidden_size ||
      params.shared_up.input_cols != params.hidden_size ||
      params.shared_down.output_rows != params.hidden_size) {
    return false;
  }

  const dim3 block(fused_decode::kThreadsPerBlock);
  const dim3 grid(1);
  const std::size_t shared_bytes =
      (std::max(params.routed_expert_intermediate_size,
                params.shared_expert_intermediate_size) +
       params.hidden_size) *
      sizeof(float);
  FusedMoeDirectDecodeKernel<<<grid, block, shared_bytes>>>(
      params,
      input.data(),
      normalized.data(),
      router_logits.data(),
      output->data());
  return CheckCuda(cudaGetLastError());
}

}  // namespace nemotron
