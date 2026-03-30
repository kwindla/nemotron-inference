#include "nemotron/loader.h"

#include <iostream>
#include <string>

namespace {

using nemotron::BuildLoaderPlan;
using nemotron::HasManifestErrors;
using nemotron::LoaderPlan;
using nemotron::ManifestValidationIssue;
using nemotron::ManifestIssueSeverity;
using nemotron::PackedModelManifest;
using nemotron::ServiceMemoryTarget;
using nemotron::TensorAuxiliaryManifest;
using nemotron::TensorManifestEntry;

constexpr std::size_t GiB(std::size_t value) {
  return value * 1024ull * 1024ull * 1024ull;
}

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

PackedModelManifest make_manifest() {
  PackedModelManifest manifest;
  manifest.schema_version = 1;
  manifest.runtime.model_id = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4";
  manifest.runtime.source_revision = "b1ffe499";
  manifest.runtime.tokenizer_revision = "b1ffe499";
  manifest.runtime.packer_version = "test-packer";
  manifest.runtime.gpu_family = "GB10";
  manifest.runtime.compute_capability = "12.1";
  manifest.runtime.kv_bytes_per_token = 4096;
  manifest.runtime.mamba_state_bytes_fp16 = 87162880;
  manifest.runtime.mamba_state_bytes_fp32 = 174325760;

  TensorManifestEntry dense;
  dense.name = "backbone.embeddings.weight";
  dense.op_class = "embedding";
  dense.logical_shape = {131072, 4096};
  dense.packed_shape = {131072, 4096};
  dense.storage_dtype = "bf16";
  dense.compute_dtype = "bf16";
  dense.layout_tag = "row_major";
  dense.alignment_bytes = 16;
  dense.packed_file = "weights/core.bin";
  dense.offset_bytes = 0;
  dense.nbytes = 4096;
  dense.source_tensor_name = dense.name;
  dense.checksum = "abc";
  manifest.tensors.push_back(dense);

  TensorManifestEntry scaled;
  scaled.name = "layers.12.experts.117.w1";
  scaled.op_class = "routed_expert";
  scaled.logical_shape = {2688, 1024};
  scaled.packed_shape = {2688, 1024};
  scaled.storage_dtype = "nvfp4_e2m1";
  scaled.compute_dtype = "fp32_accum";
  scaled.layout_tag = "cublaslt_fp4_tn_v1";
  scaled.alignment_bytes = 16;
  scaled.block_scale_mode = "vec16_e4m3";
  scaled.block_scale_dtype = "e4m3";
  scaled.tensor_scale_dtype = "fp32";
  scaled.packed_file = "weights/experts.bin";
  scaled.offset_bytes = 4096;
  scaled.nbytes = 2048;
  scaled.source_tensor_name = scaled.name;
  scaled.checksum = "def";
  scaled.auxiliaries.push_back(TensorAuxiliaryManifest{
      "block_scales",
      "weights/experts_scales.bin",
      0,
      512,
  });
  scaled.auxiliaries.push_back(TensorAuxiliaryManifest{
      "tensor_scale",
      "weights/experts_scales.bin",
      512,
      4,
  });
  manifest.tensors.push_back(scaled);

  return manifest;
}

bool test_loader_plan_builds_memory_profile_and_totals() {
  const auto manifest = make_manifest();
  ServiceMemoryTarget target;
  target.total_memory_bytes = GiB(128);
  target.weights_bytes = GiB(100);
  target.workspace_bytes = GiB(8);
  target.graph_bytes = GiB(4);
  target.safety_headroom_bytes = GiB(4);
  target.target_active_requests = 8;
  target.target_context_tokens = 65536;

  const LoaderPlan plan = BuildLoaderPlan(
      manifest,
      target,
      /*use_fp16_mamba_state=*/true,
      /*reusable_node_metadata_bytes=*/4096);

  return expect(plan.valid, "valid manifest should build a valid loader plan") &&
         expect(plan.tensor_count == 2, "loader plan should count tensors") &&
         expect(plan.total_packed_bytes == 6144, "loader plan should total packed tensor bytes") &&
         expect(plan.total_auxiliary_bytes == 516, "loader plan should total auxiliary bytes") &&
         expect(plan.bytes_by_op_class.at("embedding") == 4096, "embedding bytes should be tracked by op class") &&
         expect(plan.bytes_by_op_class.at("routed_expert") == 2048, "routed expert bytes should be tracked by op class") &&
         expect(plan.bytes_by_file.at("weights/experts_scales.bin") == 516, "auxiliary bytes should be tracked by file") &&
         expect(plan.memory_profile.current_mamba_state_bytes == 87162880,
                "loader plan should select the FP16 Mamba-state byte size") &&
         expect(plan.memory_budget.max_full_context_nodes == 28,
                "loader plan should integrate with runtime memory-budget planning");
}

bool test_duplicate_tensor_names_are_rejected() {
  auto manifest = make_manifest();
  manifest.tensors.push_back(manifest.tensors.front());

  ServiceMemoryTarget target;
  const LoaderPlan plan = BuildLoaderPlan(manifest, target, true, 4096);
  return expect(!plan.valid, "manifest with duplicate tensor names should be invalid") &&
         expect(HasManifestErrors(plan.issues), "duplicate tensor names should produce manifest errors");
}

bool test_scaled_tensor_requires_auxiliaries() {
  auto manifest = make_manifest();
  manifest.tensors[1].auxiliaries.clear();

  ServiceMemoryTarget target;
  const LoaderPlan plan = BuildLoaderPlan(manifest, target, true, 4096);
  return expect(!plan.valid, "scaled tensor without auxiliaries should be invalid") &&
         expect(HasManifestErrors(plan.issues), "missing scale auxiliaries should produce manifest errors");
}

bool test_manifest_rejects_missing_checksum() {
  auto manifest = make_manifest();
  manifest.tensors[0].checksum.clear();

  ServiceMemoryTarget target;
  const LoaderPlan plan = BuildLoaderPlan(manifest, target, false, 4096);
  return expect(!plan.valid, "missing checksum should invalidate the manifest") &&
         expect(HasManifestErrors(plan.issues), "missing checksum should produce manifest errors");
}

}  // namespace

int main() {
  const bool ok =
      test_loader_plan_builds_memory_profile_and_totals() &&
      test_duplicate_tensor_names_are_rejected() &&
      test_scaled_tensor_requires_auxiliaries() &&
      test_manifest_rejects_missing_checksum();

  if (!ok) {
    return 1;
  }
  std::cout << "loader_plan_test: PASS\n";
  return 0;
}
