#pragma once

#include <cstddef>
#include <vector>

#include "nemotron/expert_layer.h"

namespace nemotron {

class DeviceNvfp4Weight;
class MonolithicNvfp4ExpertWeights;

struct MoeBackendPrepareContext {
  const ExpertLayerConfig* config = nullptr;
  const std::vector<ExpertWeightPair>* routed_weights = nullptr;
  const GemmDescriptor* shared_up = nullptr;
  const GemmDescriptor* shared_down = nullptr;
  const MonolithicNvfp4ExpertWeights* monolithic_up = nullptr;
  const MonolithicNvfp4ExpertWeights* monolithic_down = nullptr;
  const DeviceNvfp4Weight* shared_up_device = nullptr;
  const DeviceNvfp4Weight* shared_down_device = nullptr;
};

class MoeBackend {
 public:
  virtual ~MoeBackend() = default;

  virtual const char* Name() const = 0;

  virtual bool Supports(
      const ExpertLayerConfig& config,
      std::size_t token_count,
      int device_sm_version) const = 0;

  // Backends can use this load-time hook to read resident NVFP4 weights and
  // cache any backend-native representation needed by Run().
  virtual bool PrepareWeights(const MoeBackendPrepareContext& context) {
    (void)context;
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
      const int* topk_ids,
      const float* topk_weights,
      DeviceTensorFp32* output,
      ExpertLayerRunTrace* trace) = 0;
};

}  // namespace nemotron
