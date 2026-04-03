#pragma once

#include <cstddef>
#include <vector>

#include "nemotron/expert_layer.h"

namespace nemotron {

class MoeBackend {
 public:
  virtual ~MoeBackend() = default;

  virtual const char* Name() const = 0;

  virtual bool Supports(
      const ExpertLayerConfig& config,
      std::size_t token_count,
      int device_sm_version) const = 0;

  virtual bool PrepareWeights(
      const ExpertLayerConfig& config,
      const std::vector<ExpertWeightPair>& routed_weights,
      const GemmDescriptor* shared_up,
      const GemmDescriptor* shared_down) {
    (void)config;
    (void)routed_weights;
    (void)shared_up;
    (void)shared_down;
    return true;
  }

  virtual bool Run(
      CublasLtHandle& cublas_handle,
      GemmHeuristicCache* heuristic_cache,
      const ExpertLayerConfig& config,
      std::size_t token_count,
      const DeviceTensorFp32& input,
      const DeviceTensorFp32& normalized,
      const DeviceTensorFp32& router_logits,
      DeviceTensorFp32* output,
      ExpertLayerRunTrace* trace) = 0;
};

}  // namespace nemotron
