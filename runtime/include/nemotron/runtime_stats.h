#pragma once

#include <cstdint>

namespace nemotron {

struct RuntimeExecutionStatsSnapshot {
  std::uint64_t dense_plan_cache_hits = 0;
  std::uint64_t dense_plan_build_failures = 0;
  std::uint64_t dense_plan_build_failures_attention = 0;
  std::uint64_t dense_plan_build_failures_expert = 0;
  std::uint64_t dense_plan_build_failures_other = 0;
  std::uint64_t dense_native_success = 0;
  std::uint64_t dense_native_success_attention = 0;
  std::uint64_t dense_native_success_expert = 0;
  std::uint64_t dense_native_success_other = 0;
  std::uint64_t dense_reference_fallbacks = 0;
  std::uint64_t dense_reference_fallbacks_attention = 0;
  std::uint64_t dense_reference_fallbacks_expert = 0;
  std::uint64_t dense_reference_fallbacks_other = 0;
  std::uint64_t dense_bf16_native_failures = 0;
  std::uint64_t dense_fp32_native_failures = 0;
  std::uint64_t scaled_fp8_plan_cache_hits = 0;
  std::uint64_t scaled_fp8_plan_build_failures = 0;
  std::uint64_t scaled_fp8_plan_build_failures_mamba_in_proj = 0;
  std::uint64_t scaled_fp8_plan_build_failures_mamba_out_proj = 0;
  std::uint64_t scaled_fp8_plan_build_failures_expert_fc1_latent = 0;
  std::uint64_t scaled_fp8_plan_build_failures_expert_shared_up = 0;
  std::uint64_t scaled_fp8_plan_build_failures_expert_shared_down = 0;
  std::uint64_t scaled_fp8_plan_build_failures_other = 0;
  std::uint64_t scaled_fp8_native_success = 0;
  std::uint64_t scaled_fp8_native_success_mamba_in_proj = 0;
  std::uint64_t scaled_fp8_native_success_mamba_out_proj = 0;
  std::uint64_t scaled_fp8_native_success_expert_fc1_latent = 0;
  std::uint64_t scaled_fp8_native_success_expert_shared_up = 0;
  std::uint64_t scaled_fp8_native_success_expert_shared_down = 0;
  std::uint64_t scaled_fp8_native_success_other = 0;
  std::uint64_t scaled_fp8_dequantized_dense_success = 0;
  std::uint64_t scaled_fp8_reference_fallbacks = 0;
  std::uint64_t scaled_fp8_reference_fallbacks_mamba_in_proj = 0;
  std::uint64_t scaled_fp8_reference_fallbacks_mamba_out_proj = 0;
  std::uint64_t scaled_fp8_reference_fallbacks_expert_fc1_latent = 0;
  std::uint64_t scaled_fp8_reference_fallbacks_expert_shared_up = 0;
  std::uint64_t scaled_fp8_reference_fallbacks_expert_shared_down = 0;
  std::uint64_t scaled_fp8_reference_fallbacks_other = 0;
  std::uint64_t attention_decode_plan_creates = 0;
  std::uint64_t attention_decode_plan_hits = 0;
  std::uint64_t attention_decode_device_page_table_copies = 0;
  std::uint64_t expert_selection_metadata_downloads = 0;
  std::uint64_t routed_lookup_repair_downloads = 0;
  std::uint64_t routed_lookup_repair_experts = 0;
  std::uint64_t routed_expert_materializations = 0;
  std::uint64_t flashinfer_routed_expert_uses = 0;
  std::uint64_t flashinfer_routed_expert_fallbacks = 0;
  std::uint64_t grouped_routed_expert_fastpath_uses = 0;
  std::uint64_t grouped_routed_expert_fastpath_fallbacks = 0;
  std::uint64_t grouped_routed_expert_prereq_fallbacks = 0;
  std::uint64_t grouped_routed_expert_lookup_fallbacks = 0;
  std::uint64_t grouped_routed_expert_plan_fallbacks = 0;
  std::uint64_t grouped_routed_expert_pack_fallbacks = 0;
  std::uint64_t grouped_routed_expert_matmul_fallbacks = 0;
  std::uint64_t grouped_routed_expert_merge_fallbacks = 0;
  std::uint64_t routed_expert_prefetch_layers = 0;
  std::uint64_t routed_expert_prefetch_experts = 0;
};

void ResetRuntimeExecutionStats();
RuntimeExecutionStatsSnapshot GetRuntimeExecutionStatsSnapshot();

enum class DenseRuntimeOpFamily {
  kAttention,
  kExpert,
  kOther,
};

enum class ScaledFp8RuntimeOpFamily {
  kMambaInProj,
  kMambaOutProj,
  kExpertFc1Latent,
  kExpertSharedUp,
  kExpertSharedDown,
  kOther,
};

void RecordDensePlanCacheHit();
void RecordDensePlanBuildFailure(DenseRuntimeOpFamily family);
void RecordDenseNativeSuccess(DenseRuntimeOpFamily family);
void RecordDenseReferenceFallback(DenseRuntimeOpFamily family);
void RecordDenseBf16NativeFailure();
void RecordDenseFp32NativeFailure();
void RecordScaledFp8PlanCacheHit();
void RecordScaledFp8PlanBuildFailure(ScaledFp8RuntimeOpFamily family);
void RecordScaledFp8NativeSuccess(ScaledFp8RuntimeOpFamily family);
void RecordScaledFp8DequantizedDenseSuccess();
void RecordScaledFp8ReferenceFallback(ScaledFp8RuntimeOpFamily family);
void RecordAttentionDecodePlanCreate();
void RecordAttentionDecodePlanHit();
void RecordAttentionDecodeDevicePageTableCopy();
void RecordExpertSelectionMetadataDownload();
void RecordRoutedLookupRepairDownload();
void RecordRoutedLookupRepairExperts(std::uint64_t count);
void RecordRoutedExpertMaterialization();
void RecordFlashInferRoutedExpertUse();
void RecordFlashInferRoutedExpertFallback();
void RecordGroupedRoutedExpertFastpathUse();
void RecordGroupedRoutedExpertFastpathFallback();
void RecordGroupedRoutedExpertPrereqFallback();
void RecordGroupedRoutedExpertLookupFallback();
void RecordGroupedRoutedExpertPlanFallback();
void RecordGroupedRoutedExpertPackFallback();
void RecordGroupedRoutedExpertMatmulFallback();
void RecordGroupedRoutedExpertMergeFallback();
void RecordRoutedExpertPrefetchLayer();
void RecordRoutedExpertPrefetchExperts(std::uint64_t count);

}  // namespace nemotron
