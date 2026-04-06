#pragma once

#include <cstddef>

#include "nemotron/fused_moe_decode.h"

namespace nemotron {

struct FusedMoePrefillParams {
  std::size_t token_count = 0;
  std::size_t hidden_size = 0;
  std::size_t routed_expert_intermediate_size = 0;
  std::size_t shared_expert_intermediate_size = 0;
  std::size_t n_routed_experts = 0;
  std::size_t top_k = 0;
  FusedNvfp4WeightView shared_up;
  FusedNvfp4WeightView shared_down;
  const FusedNvfp4WeightView* routed_up = nullptr;
  const FusedNvfp4WeightView* routed_down = nullptr;
  const int* selected_indices = nullptr;
  const float* selected_weights = nullptr;
  const float* input = nullptr;
  const float* normalized = nullptr;
  float* output = nullptr;
  float* routed_output = nullptr;  // Optional.
  float* shared_output = nullptr;  // Optional.
};

bool RunFusedMoePrefill(const FusedMoePrefillParams& params);

}  // namespace nemotron
