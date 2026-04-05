#pragma once

#include <cstddef>

#include <cuda_bf16.h>

#include "nemotron/fused_moe_decode.h"

namespace nemotron {

inline constexpr std::size_t kFusedGroupedMoeNanoHiddenSize = 2688;
inline constexpr std::size_t kFusedGroupedMoeNanoRoutedIntermediateSize = 1856;
inline constexpr std::size_t kFusedGroupedMoeNanoRoutedExperts = 128;
inline constexpr std::size_t kFusedGroupedMoeNanoTopK = 6;

struct FusedGroupedMoeConfig {
  std::size_t hidden_size = kFusedGroupedMoeNanoHiddenSize;
  std::size_t routed_expert_intermediate_size =
      kFusedGroupedMoeNanoRoutedIntermediateSize;
  std::size_t n_routed_experts = kFusedGroupedMoeNanoRoutedExperts;
  std::size_t top_k = kFusedGroupedMoeNanoTopK;
  std::size_t tile_size = 1;
};

struct FusedGroupedMoeParams {
  std::size_t token_count = 0;
  std::size_t selection_count = 0;
  const __nv_bfloat16* normalized = nullptr;
  const int* expert_offsets = nullptr;
  const int* sorted_token_indices = nullptr;
  const float* sorted_token_weights = nullptr;
  const int* active_expert_count = nullptr;
  const int* active_expert_ids = nullptr;
  const FusedNvfp4WeightView* routed_up = nullptr;
  const FusedNvfp4WeightView* routed_down = nullptr;
  float* output = nullptr;
};

bool RunFusedGroupedMoe(
    const FusedGroupedMoeParams& params,
    const FusedGroupedMoeConfig& config = {});

}  // namespace nemotron
