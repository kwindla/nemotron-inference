#pragma once

#include <cstddef>

#include "nemotron/expert_routing_device.h"
#include "nemotron/fused_moe_decode.h"
#include "nemotron/moe_launch_plan_device.h"

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
  const FusedNvfp4WeightView* routed_up_device = nullptr;
  const FusedNvfp4WeightView* routed_down_device = nullptr;
  const int* selected_indices = nullptr;
  const float* selected_weights = nullptr;
  const float* input = nullptr;  // Reserved for future fused epilog variants.
  const float* normalized = nullptr;
  DeviceExpertRouting* routing = nullptr;
  DeviceMoeLaunchPlan* launch_plan = nullptr;
  float* routed_gather_scratch = nullptr;  // selection_count x hidden_size
  float* routed_up_scratch = nullptr;      // selection_count x routed_expert_intermediate_size
  float* shared_up_scratch = nullptr;      // token_count x shared_expert_intermediate_size
  float* output = nullptr;  // Receives routed + shared expert contributions.
  float* routed_output = nullptr;  // Optional.
  float* shared_output = nullptr;  // Optional.
};

bool RunGroupedNvfp4ExpertMatVec(
    const float* input,
    const int* expert_offsets,
    std::size_t n_experts,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output);

bool RunLaunchPlannedNvfp4ExpertMatVec(
    const float* input,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output);

bool RunFusedMoePrefill(const FusedMoePrefillParams& params);

}  // namespace nemotron
