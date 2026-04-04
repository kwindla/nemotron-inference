#pragma once

#include <cstddef>
#include <vector>

#include "nemotron/expert_layer.h"
#include "nemotron/fused_moe_decode.h"

namespace nemotron {

inline bool MoeBackendHasValidWeightView(const FusedNvfp4WeightView& weight) {
  return weight.packed_data != nullptr &&
         weight.block_scales_data != nullptr &&
         weight.matmul_block_scales_data != nullptr &&
         weight.tensor_scale_data != nullptr &&
         weight.output_rows > 0 &&
         weight.input_cols > 0;
}

struct MoeBackendPrepareContext {
  const ExpertLayerConfig* config = nullptr;
  FusedNvfp4WeightView shared_up_view;
  FusedNvfp4WeightView shared_down_view;
  const std::vector<FusedNvfp4WeightView>* resident_routed_up_views = nullptr;
  const std::vector<FusedNvfp4WeightView>* resident_routed_down_views = nullptr;
  const FusedNvfp4WeightView* resident_routed_up_views_device = nullptr;
  const FusedNvfp4WeightView* resident_routed_down_views_device = nullptr;
};

struct MoeBackendPreparedWeights {
  bool prepared = false;
  FusedNvfp4WeightView shared_up;
  FusedNvfp4WeightView shared_down;
  std::vector<FusedNvfp4WeightView> routed_up_views;
  std::vector<FusedNvfp4WeightView> routed_down_views;
  const FusedNvfp4WeightView* routed_up_views_device = nullptr;
  const FusedNvfp4WeightView* routed_down_views_device = nullptr;
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

class PreparedResidentMoeBackend : public MoeBackend {
 public:
  bool PrepareWeights(const MoeBackendPrepareContext& context) final {
    prepared_weights_ = MoeBackendPreparedWeights{};
    if (!PrepareResidentWeights(context, &prepared_weights_)) {
      return false;
    }
    if (!PrepareBackendSpecificWeights(context, &prepared_weights_)) {
      prepared_weights_ = MoeBackendPreparedWeights{};
      return false;
    }
    prepared_weights_.prepared = true;
    return true;
  }

 protected:
  virtual bool PrepareBackendSpecificWeights(
      const MoeBackendPrepareContext& context,
      MoeBackendPreparedWeights* prepared_weights) {
    (void)context;
    (void)prepared_weights;
    return true;
  }

  const MoeBackendPreparedWeights& prepared_weights() const {
    return prepared_weights_;
  }

  static bool PrepareResidentWeights(
      const MoeBackendPrepareContext& context,
      MoeBackendPreparedWeights* prepared_weights) {
    if (prepared_weights == nullptr ||
        context.config == nullptr ||
        !MoeBackendHasValidWeightView(context.shared_up_view) ||
        !MoeBackendHasValidWeightView(context.shared_down_view) ||
        context.resident_routed_up_views == nullptr ||
        context.resident_routed_down_views == nullptr ||
        context.resident_routed_up_views->size() != context.config->n_routed_experts ||
        context.resident_routed_down_views->size() != context.config->n_routed_experts) {
      return false;
    }

    for (std::size_t expert_index = 0; expert_index < context.config->n_routed_experts; ++expert_index) {
      if (!MoeBackendHasValidWeightView((*context.resident_routed_up_views)[expert_index]) ||
          !MoeBackendHasValidWeightView((*context.resident_routed_down_views)[expert_index])) {
        return false;
      }
    }

    prepared_weights->shared_up = context.shared_up_view;
    prepared_weights->shared_down = context.shared_down_view;
    prepared_weights->routed_up_views = *context.resident_routed_up_views;
    prepared_weights->routed_down_views = *context.resident_routed_down_views;
    prepared_weights->routed_up_views_device = context.resident_routed_up_views_device;
    prepared_weights->routed_down_views_device = context.resident_routed_down_views_device;
    return true;
  }

 private:
  MoeBackendPreparedWeights prepared_weights_;
};

}  // namespace nemotron
