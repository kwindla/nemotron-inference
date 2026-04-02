#include "nemotron/gemm_planner.h"

namespace nemotron {

std::optional<GemmLaunchPlan> BuildGemmLaunchPlan(
    const GemmDescriptor& descriptor,
    std::size_t activation_rows) {
  if (activation_rows == 0 || !descriptor.packed_bytes().valid()) {
    return std::nullopt;
  }

  GemmLaunchPlan plan;
  plan.descriptor = &descriptor;
  plan.kernel_family = descriptor.kernel_family;
  plan.m = activation_rows;
  plan.n = descriptor.output_rows;
  plan.k = descriptor.input_cols;
  plan.tensor_name = descriptor.tensor_name;
  plan.storage_dtype = descriptor.storage_dtype;
  plan.compute_dtype = descriptor.compute_dtype;
  plan.uses_block_scales = descriptor.is_scaled();
  plan.packed_bytes = descriptor.packed_bytes();
  plan.block_scales_bytes = descriptor.block_scales_bytes();
  plan.tensor_scale_bytes = descriptor.tensor_scale_bytes();
  plan.heuristic_key =
      descriptor.heuristic_key_prefix() + "|m=" + std::to_string(plan.m);
  return plan;
}

bool GemmHeuristicCache::Store(const GemmLaunchPlan& launch_plan, std::uint64_t algorithm_id) {
  if (launch_plan.heuristic_key.empty()) {
    return false;
  }
  algorithm_ids_by_key_[launch_plan.heuristic_key] = algorithm_id;
  return true;
}

std::optional<std::uint64_t> GemmHeuristicCache::Lookup(const GemmLaunchPlan& launch_plan) const {
  auto it = algorithm_ids_by_key_.find(launch_plan.heuristic_key);
  if (it == algorithm_ids_by_key_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::size_t GemmHeuristicCache::size() const {
  return algorithm_ids_by_key_.size();
}

}  // namespace nemotron
