#pragma once

#include <cstddef>

#include "nemotron/fused_moe_decode.h"

namespace nemotron {

class CublasLtHandle;
class DeviceNvfp4Matrix;
class GemmHeuristicCache;
struct GemmDescriptor;

struct FusedMoePrefillParams {
  std::size_t token_count = 0;
  std::size_t selection_count = 0;
  std::size_t hidden_size = 0;
  std::size_t routed_expert_intermediate_size = 0;
  std::size_t shared_expert_intermediate_size = 0;
  std::size_t n_routed_experts = 0;
  std::size_t top_k = 0;
  CublasLtHandle* cublas_handle = nullptr;
  GemmHeuristicCache* heuristic_cache = nullptr;
  const GemmDescriptor* shared_up_descriptor = nullptr;
  const GemmDescriptor* shared_down_descriptor = nullptr;
  const GemmDescriptor* const* routed_up_descriptors = nullptr;
  const GemmDescriptor* const* routed_down_descriptors = nullptr;
  FusedNvfp4WeightView shared_up;
  FusedNvfp4WeightView shared_down;
  // Host-side arrays indexed by expert id. The views themselves point at
  // resident device weights.
  const FusedNvfp4WeightView* routed_up = nullptr;
  const FusedNvfp4WeightView* routed_down = nullptr;
  const float* input = nullptr;
  const float* normalized = nullptr;
  float* output = nullptr;
  float* routed_output = nullptr;
  float* gather_scratch = nullptr;
  float* expert_up_scratch = nullptr;
  float* shared_up_scratch = nullptr;
  DeviceNvfp4Matrix* gather_pack = nullptr;
  DeviceNvfp4Matrix* expert_up_pack = nullptr;
  DeviceNvfp4Matrix* shared_up_pack = nullptr;
  const int* expert_offsets = nullptr;
  const int* sorted_token_indices = nullptr;
  const float* sorted_token_weights = nullptr;
};

bool RunFusedMoePrefill(const FusedMoePrefillParams& params);

}  // namespace nemotron
