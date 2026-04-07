#include "nemotron/loader.h"

namespace nemotron {

LoaderPlan BuildLoaderPlan(
    const PackedModelManifest& manifest,
    const ServiceMemoryTarget& target,
    bool use_fp16_mamba_state,
    std::size_t reusable_node_metadata_bytes) {
  LoaderPlan plan;
  plan.issues = ValidateManifest(manifest);
  if (HasManifestErrors(plan.issues)) {
    return plan;
  }

  plan.valid = true;
  plan.tensor_count = manifest.tensors.size();
  plan.memory_profile.kv_bytes_per_token = manifest.runtime.kv_bytes_per_token;
  plan.memory_profile.current_mamba_state_bytes =
      use_fp16_mamba_state ? manifest.runtime.mamba_state_bytes_fp16
                           : manifest.runtime.mamba_state_bytes_fp32;
  plan.memory_profile.reusable_node_metadata_bytes = reusable_node_metadata_bytes;

  for (const TensorManifestEntry& tensor : manifest.tensors) {
    plan.total_packed_bytes += tensor.nbytes;
    plan.bytes_by_op_class[tensor.op_class] += tensor.nbytes;
    plan.bytes_by_file[tensor.packed_file] += tensor.nbytes;
    for (const TensorAuxiliaryManifest& auxiliary : tensor.auxiliaries) {
      plan.total_auxiliary_bytes += auxiliary.nbytes;
      plan.bytes_by_file[auxiliary.file] += auxiliary.nbytes;
    }
  }

  plan.memory_budget = BuildMemoryBudgetSummary(plan.memory_profile, target);
  return plan;
}

}  // namespace nemotron
