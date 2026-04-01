#include "nemotron/runtime_stats.h"

#include <atomic>

namespace nemotron {
namespace {

struct RuntimeExecutionStats {
  std::atomic<std::uint64_t> dense_plan_cache_hits{0};
  std::atomic<std::uint64_t> dense_plan_build_failures{0};
  std::atomic<std::uint64_t> dense_plan_build_failures_attention{0};
  std::atomic<std::uint64_t> dense_plan_build_failures_expert{0};
  std::atomic<std::uint64_t> dense_plan_build_failures_other{0};
  std::atomic<std::uint64_t> dense_native_success{0};
  std::atomic<std::uint64_t> dense_native_success_attention{0};
  std::atomic<std::uint64_t> dense_native_success_expert{0};
  std::atomic<std::uint64_t> dense_native_success_other{0};
  std::atomic<std::uint64_t> dense_reference_fallbacks{0};
  std::atomic<std::uint64_t> dense_reference_fallbacks_attention{0};
  std::atomic<std::uint64_t> dense_reference_fallbacks_expert{0};
  std::atomic<std::uint64_t> dense_reference_fallbacks_other{0};
  std::atomic<std::uint64_t> dense_bf16_native_failures{0};
  std::atomic<std::uint64_t> dense_fp32_native_failures{0};
  std::atomic<std::uint64_t> scaled_fp8_plan_cache_hits{0};
  std::atomic<std::uint64_t> scaled_fp8_plan_build_failures{0};
  std::atomic<std::uint64_t> scaled_fp8_plan_build_failures_mamba_in_proj{0};
  std::atomic<std::uint64_t> scaled_fp8_plan_build_failures_mamba_out_proj{0};
  std::atomic<std::uint64_t> scaled_fp8_plan_build_failures_expert_fc1_latent{0};
  std::atomic<std::uint64_t> scaled_fp8_plan_build_failures_expert_shared_up{0};
  std::atomic<std::uint64_t> scaled_fp8_plan_build_failures_expert_shared_down{0};
  std::atomic<std::uint64_t> scaled_fp8_plan_build_failures_other{0};
  std::atomic<std::uint64_t> scaled_fp8_native_success{0};
  std::atomic<std::uint64_t> scaled_fp8_native_success_mamba_in_proj{0};
  std::atomic<std::uint64_t> scaled_fp8_native_success_mamba_out_proj{0};
  std::atomic<std::uint64_t> scaled_fp8_native_success_expert_fc1_latent{0};
  std::atomic<std::uint64_t> scaled_fp8_native_success_expert_shared_up{0};
  std::atomic<std::uint64_t> scaled_fp8_native_success_expert_shared_down{0};
  std::atomic<std::uint64_t> scaled_fp8_native_success_other{0};
  std::atomic<std::uint64_t> scaled_fp8_dequantized_dense_success{0};
  std::atomic<std::uint64_t> scaled_fp8_reference_fallbacks{0};
  std::atomic<std::uint64_t> scaled_fp8_reference_fallbacks_mamba_in_proj{0};
  std::atomic<std::uint64_t> scaled_fp8_reference_fallbacks_mamba_out_proj{0};
  std::atomic<std::uint64_t> scaled_fp8_reference_fallbacks_expert_fc1_latent{0};
  std::atomic<std::uint64_t> scaled_fp8_reference_fallbacks_expert_shared_up{0};
  std::atomic<std::uint64_t> scaled_fp8_reference_fallbacks_expert_shared_down{0};
  std::atomic<std::uint64_t> scaled_fp8_reference_fallbacks_other{0};
  std::atomic<std::uint64_t> attention_decode_plan_creates{0};
  std::atomic<std::uint64_t> attention_decode_plan_hits{0};
  std::atomic<std::uint64_t> attention_decode_device_page_table_copies{0};
  std::atomic<std::uint64_t> expert_selection_metadata_downloads{0};
  std::atomic<std::uint64_t> routed_lookup_repair_downloads{0};
  std::atomic<std::uint64_t> routed_lookup_repair_experts{0};
  std::atomic<std::uint64_t> routed_expert_materializations{0};
  std::atomic<std::uint64_t> flashinfer_routed_expert_uses{0};
  std::atomic<std::uint64_t> flashinfer_routed_expert_fallbacks{0};
  std::atomic<std::uint64_t> grouped_routed_expert_fastpath_uses{0};
  std::atomic<std::uint64_t> moe_graph_captures{0};
  std::atomic<std::uint64_t> moe_graph_replays{0};
  std::atomic<std::uint64_t> grouped_routed_expert_fastpath_fallbacks{0};
  std::atomic<std::uint64_t> grouped_routed_expert_prereq_fallbacks{0};
  std::atomic<std::uint64_t> grouped_routed_expert_lookup_fallbacks{0};
  std::atomic<std::uint64_t> grouped_routed_expert_plan_fallbacks{0};
  std::atomic<std::uint64_t> grouped_routed_expert_pack_fallbacks{0};
  std::atomic<std::uint64_t> grouped_routed_expert_matmul_fallbacks{0};
  std::atomic<std::uint64_t> grouped_routed_expert_merge_fallbacks{0};
  std::atomic<std::uint64_t> routed_expert_prefetch_layers{0};
  std::atomic<std::uint64_t> routed_expert_prefetch_experts{0};
};

RuntimeExecutionStats& MutableRuntimeExecutionStats() {
  static RuntimeExecutionStats stats;
  return stats;
}

}  // namespace

void ResetRuntimeExecutionStats() {
  auto& stats = MutableRuntimeExecutionStats();
  stats.dense_plan_cache_hits.store(0, std::memory_order_relaxed);
  stats.dense_plan_build_failures.store(0, std::memory_order_relaxed);
  stats.dense_plan_build_failures_attention.store(0, std::memory_order_relaxed);
  stats.dense_plan_build_failures_expert.store(0, std::memory_order_relaxed);
  stats.dense_plan_build_failures_other.store(0, std::memory_order_relaxed);
  stats.dense_native_success.store(0, std::memory_order_relaxed);
  stats.dense_native_success_attention.store(0, std::memory_order_relaxed);
  stats.dense_native_success_expert.store(0, std::memory_order_relaxed);
  stats.dense_native_success_other.store(0, std::memory_order_relaxed);
  stats.dense_reference_fallbacks.store(0, std::memory_order_relaxed);
  stats.dense_reference_fallbacks_attention.store(0, std::memory_order_relaxed);
  stats.dense_reference_fallbacks_expert.store(0, std::memory_order_relaxed);
  stats.dense_reference_fallbacks_other.store(0, std::memory_order_relaxed);
  stats.dense_bf16_native_failures.store(0, std::memory_order_relaxed);
  stats.dense_fp32_native_failures.store(0, std::memory_order_relaxed);
  stats.scaled_fp8_plan_cache_hits.store(0, std::memory_order_relaxed);
  stats.scaled_fp8_plan_build_failures.store(0, std::memory_order_relaxed);
  stats.scaled_fp8_plan_build_failures_mamba_in_proj.store(0, std::memory_order_relaxed);
  stats.scaled_fp8_plan_build_failures_mamba_out_proj.store(0, std::memory_order_relaxed);
  stats.scaled_fp8_plan_build_failures_expert_fc1_latent.store(0, std::memory_order_relaxed);
  stats.scaled_fp8_plan_build_failures_expert_shared_up.store(0, std::memory_order_relaxed);
  stats.scaled_fp8_plan_build_failures_expert_shared_down.store(0, std::memory_order_relaxed);
  stats.scaled_fp8_plan_build_failures_other.store(0, std::memory_order_relaxed);
  stats.scaled_fp8_native_success.store(0, std::memory_order_relaxed);
  stats.scaled_fp8_native_success_mamba_in_proj.store(0, std::memory_order_relaxed);
  stats.scaled_fp8_native_success_mamba_out_proj.store(0, std::memory_order_relaxed);
  stats.scaled_fp8_native_success_expert_fc1_latent.store(0, std::memory_order_relaxed);
  stats.scaled_fp8_native_success_expert_shared_up.store(0, std::memory_order_relaxed);
  stats.scaled_fp8_native_success_expert_shared_down.store(0, std::memory_order_relaxed);
  stats.scaled_fp8_native_success_other.store(0, std::memory_order_relaxed);
  stats.scaled_fp8_dequantized_dense_success.store(0, std::memory_order_relaxed);
  stats.scaled_fp8_reference_fallbacks.store(0, std::memory_order_relaxed);
  stats.scaled_fp8_reference_fallbacks_mamba_in_proj.store(0, std::memory_order_relaxed);
  stats.scaled_fp8_reference_fallbacks_mamba_out_proj.store(0, std::memory_order_relaxed);
  stats.scaled_fp8_reference_fallbacks_expert_fc1_latent.store(0, std::memory_order_relaxed);
  stats.scaled_fp8_reference_fallbacks_expert_shared_up.store(0, std::memory_order_relaxed);
  stats.scaled_fp8_reference_fallbacks_expert_shared_down.store(0, std::memory_order_relaxed);
  stats.scaled_fp8_reference_fallbacks_other.store(0, std::memory_order_relaxed);
  stats.attention_decode_plan_creates.store(0, std::memory_order_relaxed);
  stats.attention_decode_plan_hits.store(0, std::memory_order_relaxed);
  stats.attention_decode_device_page_table_copies.store(0, std::memory_order_relaxed);
  stats.expert_selection_metadata_downloads.store(0, std::memory_order_relaxed);
  stats.routed_lookup_repair_downloads.store(0, std::memory_order_relaxed);
  stats.routed_lookup_repair_experts.store(0, std::memory_order_relaxed);
  stats.routed_expert_materializations.store(0, std::memory_order_relaxed);
  stats.flashinfer_routed_expert_uses.store(0, std::memory_order_relaxed);
  stats.flashinfer_routed_expert_fallbacks.store(0, std::memory_order_relaxed);
  stats.grouped_routed_expert_fastpath_uses.store(0, std::memory_order_relaxed);
  stats.moe_graph_captures.store(0, std::memory_order_relaxed);
  stats.moe_graph_replays.store(0, std::memory_order_relaxed);
  stats.grouped_routed_expert_fastpath_fallbacks.store(0, std::memory_order_relaxed);
  stats.grouped_routed_expert_prereq_fallbacks.store(0, std::memory_order_relaxed);
  stats.grouped_routed_expert_lookup_fallbacks.store(0, std::memory_order_relaxed);
  stats.grouped_routed_expert_plan_fallbacks.store(0, std::memory_order_relaxed);
  stats.grouped_routed_expert_pack_fallbacks.store(0, std::memory_order_relaxed);
  stats.grouped_routed_expert_matmul_fallbacks.store(0, std::memory_order_relaxed);
  stats.grouped_routed_expert_merge_fallbacks.store(0, std::memory_order_relaxed);
  stats.routed_expert_prefetch_layers.store(0, std::memory_order_relaxed);
  stats.routed_expert_prefetch_experts.store(0, std::memory_order_relaxed);
}

RuntimeExecutionStatsSnapshot GetRuntimeExecutionStatsSnapshot() {
  const auto& stats = MutableRuntimeExecutionStats();
  RuntimeExecutionStatsSnapshot snapshot;
  snapshot.dense_plan_cache_hits = stats.dense_plan_cache_hits.load(std::memory_order_relaxed);
  snapshot.dense_plan_build_failures =
      stats.dense_plan_build_failures.load(std::memory_order_relaxed);
  snapshot.dense_plan_build_failures_attention =
      stats.dense_plan_build_failures_attention.load(std::memory_order_relaxed);
  snapshot.dense_plan_build_failures_expert =
      stats.dense_plan_build_failures_expert.load(std::memory_order_relaxed);
  snapshot.dense_plan_build_failures_other =
      stats.dense_plan_build_failures_other.load(std::memory_order_relaxed);
  snapshot.dense_native_success = stats.dense_native_success.load(std::memory_order_relaxed);
  snapshot.dense_native_success_attention =
      stats.dense_native_success_attention.load(std::memory_order_relaxed);
  snapshot.dense_native_success_expert =
      stats.dense_native_success_expert.load(std::memory_order_relaxed);
  snapshot.dense_native_success_other =
      stats.dense_native_success_other.load(std::memory_order_relaxed);
  snapshot.dense_reference_fallbacks =
      stats.dense_reference_fallbacks.load(std::memory_order_relaxed);
  snapshot.dense_reference_fallbacks_attention =
      stats.dense_reference_fallbacks_attention.load(std::memory_order_relaxed);
  snapshot.dense_reference_fallbacks_expert =
      stats.dense_reference_fallbacks_expert.load(std::memory_order_relaxed);
  snapshot.dense_reference_fallbacks_other =
      stats.dense_reference_fallbacks_other.load(std::memory_order_relaxed);
  snapshot.dense_bf16_native_failures =
      stats.dense_bf16_native_failures.load(std::memory_order_relaxed);
  snapshot.dense_fp32_native_failures =
      stats.dense_fp32_native_failures.load(std::memory_order_relaxed);
  snapshot.scaled_fp8_plan_cache_hits =
      stats.scaled_fp8_plan_cache_hits.load(std::memory_order_relaxed);
  snapshot.scaled_fp8_plan_build_failures =
      stats.scaled_fp8_plan_build_failures.load(std::memory_order_relaxed);
  snapshot.scaled_fp8_plan_build_failures_mamba_in_proj =
      stats.scaled_fp8_plan_build_failures_mamba_in_proj.load(std::memory_order_relaxed);
  snapshot.scaled_fp8_plan_build_failures_mamba_out_proj =
      stats.scaled_fp8_plan_build_failures_mamba_out_proj.load(std::memory_order_relaxed);
  snapshot.scaled_fp8_plan_build_failures_expert_fc1_latent =
      stats.scaled_fp8_plan_build_failures_expert_fc1_latent.load(std::memory_order_relaxed);
  snapshot.scaled_fp8_plan_build_failures_expert_shared_up =
      stats.scaled_fp8_plan_build_failures_expert_shared_up.load(std::memory_order_relaxed);
  snapshot.scaled_fp8_plan_build_failures_expert_shared_down =
      stats.scaled_fp8_plan_build_failures_expert_shared_down.load(std::memory_order_relaxed);
  snapshot.scaled_fp8_plan_build_failures_other =
      stats.scaled_fp8_plan_build_failures_other.load(std::memory_order_relaxed);
  snapshot.scaled_fp8_native_success =
      stats.scaled_fp8_native_success.load(std::memory_order_relaxed);
  snapshot.scaled_fp8_native_success_mamba_in_proj =
      stats.scaled_fp8_native_success_mamba_in_proj.load(std::memory_order_relaxed);
  snapshot.scaled_fp8_native_success_mamba_out_proj =
      stats.scaled_fp8_native_success_mamba_out_proj.load(std::memory_order_relaxed);
  snapshot.scaled_fp8_native_success_expert_fc1_latent =
      stats.scaled_fp8_native_success_expert_fc1_latent.load(std::memory_order_relaxed);
  snapshot.scaled_fp8_native_success_expert_shared_up =
      stats.scaled_fp8_native_success_expert_shared_up.load(std::memory_order_relaxed);
  snapshot.scaled_fp8_native_success_expert_shared_down =
      stats.scaled_fp8_native_success_expert_shared_down.load(std::memory_order_relaxed);
  snapshot.scaled_fp8_native_success_other =
      stats.scaled_fp8_native_success_other.load(std::memory_order_relaxed);
  snapshot.scaled_fp8_dequantized_dense_success =
      stats.scaled_fp8_dequantized_dense_success.load(std::memory_order_relaxed);
  snapshot.scaled_fp8_reference_fallbacks =
      stats.scaled_fp8_reference_fallbacks.load(std::memory_order_relaxed);
  snapshot.scaled_fp8_reference_fallbacks_mamba_in_proj =
      stats.scaled_fp8_reference_fallbacks_mamba_in_proj.load(std::memory_order_relaxed);
  snapshot.scaled_fp8_reference_fallbacks_mamba_out_proj =
      stats.scaled_fp8_reference_fallbacks_mamba_out_proj.load(std::memory_order_relaxed);
  snapshot.scaled_fp8_reference_fallbacks_expert_fc1_latent =
      stats.scaled_fp8_reference_fallbacks_expert_fc1_latent.load(std::memory_order_relaxed);
  snapshot.scaled_fp8_reference_fallbacks_expert_shared_up =
      stats.scaled_fp8_reference_fallbacks_expert_shared_up.load(std::memory_order_relaxed);
  snapshot.scaled_fp8_reference_fallbacks_expert_shared_down =
      stats.scaled_fp8_reference_fallbacks_expert_shared_down.load(std::memory_order_relaxed);
  snapshot.scaled_fp8_reference_fallbacks_other =
      stats.scaled_fp8_reference_fallbacks_other.load(std::memory_order_relaxed);
  snapshot.attention_decode_plan_creates =
      stats.attention_decode_plan_creates.load(std::memory_order_relaxed);
  snapshot.attention_decode_plan_hits =
      stats.attention_decode_plan_hits.load(std::memory_order_relaxed);
  snapshot.attention_decode_device_page_table_copies =
      stats.attention_decode_device_page_table_copies.load(std::memory_order_relaxed);
  snapshot.expert_selection_metadata_downloads =
      stats.expert_selection_metadata_downloads.load(std::memory_order_relaxed);
  snapshot.routed_lookup_repair_downloads =
      stats.routed_lookup_repair_downloads.load(std::memory_order_relaxed);
  snapshot.routed_lookup_repair_experts =
      stats.routed_lookup_repair_experts.load(std::memory_order_relaxed);
  snapshot.routed_expert_materializations =
      stats.routed_expert_materializations.load(std::memory_order_relaxed);
  snapshot.flashinfer_routed_expert_uses =
      stats.flashinfer_routed_expert_uses.load(std::memory_order_relaxed);
  snapshot.flashinfer_routed_expert_fallbacks =
      stats.flashinfer_routed_expert_fallbacks.load(std::memory_order_relaxed);
  snapshot.grouped_routed_expert_fastpath_uses =
      stats.grouped_routed_expert_fastpath_uses.load(std::memory_order_relaxed);
  snapshot.moe_graph_captures = stats.moe_graph_captures.load(std::memory_order_relaxed);
  snapshot.moe_graph_replays = stats.moe_graph_replays.load(std::memory_order_relaxed);
  snapshot.grouped_routed_expert_fastpath_fallbacks =
      stats.grouped_routed_expert_fastpath_fallbacks.load(std::memory_order_relaxed);
  snapshot.grouped_routed_expert_prereq_fallbacks =
      stats.grouped_routed_expert_prereq_fallbacks.load(std::memory_order_relaxed);
  snapshot.grouped_routed_expert_lookup_fallbacks =
      stats.grouped_routed_expert_lookup_fallbacks.load(std::memory_order_relaxed);
  snapshot.grouped_routed_expert_plan_fallbacks =
      stats.grouped_routed_expert_plan_fallbacks.load(std::memory_order_relaxed);
  snapshot.grouped_routed_expert_pack_fallbacks =
      stats.grouped_routed_expert_pack_fallbacks.load(std::memory_order_relaxed);
  snapshot.grouped_routed_expert_matmul_fallbacks =
      stats.grouped_routed_expert_matmul_fallbacks.load(std::memory_order_relaxed);
  snapshot.grouped_routed_expert_merge_fallbacks =
      stats.grouped_routed_expert_merge_fallbacks.load(std::memory_order_relaxed);
  snapshot.routed_expert_prefetch_layers =
      stats.routed_expert_prefetch_layers.load(std::memory_order_relaxed);
  snapshot.routed_expert_prefetch_experts =
      stats.routed_expert_prefetch_experts.load(std::memory_order_relaxed);
  return snapshot;
}

void RecordDensePlanCacheHit() {
  MutableRuntimeExecutionStats().dense_plan_cache_hits.fetch_add(1, std::memory_order_relaxed);
}

void RecordDensePlanBuildFailure(DenseRuntimeOpFamily family) {
  auto& stats = MutableRuntimeExecutionStats();
  stats.dense_plan_build_failures.fetch_add(1, std::memory_order_relaxed);
  switch (family) {
    case DenseRuntimeOpFamily::kAttention:
      stats.dense_plan_build_failures_attention.fetch_add(1, std::memory_order_relaxed);
      break;
    case DenseRuntimeOpFamily::kExpert:
      stats.dense_plan_build_failures_expert.fetch_add(1, std::memory_order_relaxed);
      break;
    case DenseRuntimeOpFamily::kOther:
      stats.dense_plan_build_failures_other.fetch_add(1, std::memory_order_relaxed);
      break;
  }
}

void RecordDenseNativeSuccess(DenseRuntimeOpFamily family) {
  auto& stats = MutableRuntimeExecutionStats();
  stats.dense_native_success.fetch_add(1, std::memory_order_relaxed);
  switch (family) {
    case DenseRuntimeOpFamily::kAttention:
      stats.dense_native_success_attention.fetch_add(1, std::memory_order_relaxed);
      break;
    case DenseRuntimeOpFamily::kExpert:
      stats.dense_native_success_expert.fetch_add(1, std::memory_order_relaxed);
      break;
    case DenseRuntimeOpFamily::kOther:
      stats.dense_native_success_other.fetch_add(1, std::memory_order_relaxed);
      break;
  }
}

void RecordDenseReferenceFallback(DenseRuntimeOpFamily family) {
  auto& stats = MutableRuntimeExecutionStats();
  stats.dense_reference_fallbacks.fetch_add(1, std::memory_order_relaxed);
  switch (family) {
    case DenseRuntimeOpFamily::kAttention:
      stats.dense_reference_fallbacks_attention.fetch_add(1, std::memory_order_relaxed);
      break;
    case DenseRuntimeOpFamily::kExpert:
      stats.dense_reference_fallbacks_expert.fetch_add(1, std::memory_order_relaxed);
      break;
    case DenseRuntimeOpFamily::kOther:
      stats.dense_reference_fallbacks_other.fetch_add(1, std::memory_order_relaxed);
      break;
  }
}

void RecordDenseBf16NativeFailure() {
  MutableRuntimeExecutionStats().dense_bf16_native_failures.fetch_add(
      1, std::memory_order_relaxed);
}

void RecordDenseFp32NativeFailure() {
  MutableRuntimeExecutionStats().dense_fp32_native_failures.fetch_add(
      1, std::memory_order_relaxed);
}

void RecordScaledFp8PlanCacheHit() {
  MutableRuntimeExecutionStats().scaled_fp8_plan_cache_hits.fetch_add(1, std::memory_order_relaxed);
}

void RecordScaledFp8PlanBuildFailure(ScaledFp8RuntimeOpFamily family) {
  auto& stats = MutableRuntimeExecutionStats();
  stats.scaled_fp8_plan_build_failures.fetch_add(1, std::memory_order_relaxed);
  switch (family) {
    case ScaledFp8RuntimeOpFamily::kMambaInProj:
      stats.scaled_fp8_plan_build_failures_mamba_in_proj.fetch_add(
          1, std::memory_order_relaxed);
      break;
    case ScaledFp8RuntimeOpFamily::kMambaOutProj:
      stats.scaled_fp8_plan_build_failures_mamba_out_proj.fetch_add(
          1, std::memory_order_relaxed);
      break;
    case ScaledFp8RuntimeOpFamily::kExpertFc1Latent:
      stats.scaled_fp8_plan_build_failures_expert_fc1_latent.fetch_add(
          1, std::memory_order_relaxed);
      break;
    case ScaledFp8RuntimeOpFamily::kExpertSharedUp:
      stats.scaled_fp8_plan_build_failures_expert_shared_up.fetch_add(
          1, std::memory_order_relaxed);
      break;
    case ScaledFp8RuntimeOpFamily::kExpertSharedDown:
      stats.scaled_fp8_plan_build_failures_expert_shared_down.fetch_add(
          1, std::memory_order_relaxed);
      break;
    case ScaledFp8RuntimeOpFamily::kOther:
      stats.scaled_fp8_plan_build_failures_other.fetch_add(1, std::memory_order_relaxed);
      break;
  }
}

void RecordScaledFp8NativeSuccess(ScaledFp8RuntimeOpFamily family) {
  auto& stats = MutableRuntimeExecutionStats();
  stats.scaled_fp8_native_success.fetch_add(1, std::memory_order_relaxed);
  switch (family) {
    case ScaledFp8RuntimeOpFamily::kMambaInProj:
      stats.scaled_fp8_native_success_mamba_in_proj.fetch_add(1, std::memory_order_relaxed);
      break;
    case ScaledFp8RuntimeOpFamily::kMambaOutProj:
      stats.scaled_fp8_native_success_mamba_out_proj.fetch_add(1, std::memory_order_relaxed);
      break;
    case ScaledFp8RuntimeOpFamily::kExpertFc1Latent:
      stats.scaled_fp8_native_success_expert_fc1_latent.fetch_add(
          1, std::memory_order_relaxed);
      break;
    case ScaledFp8RuntimeOpFamily::kExpertSharedUp:
      stats.scaled_fp8_native_success_expert_shared_up.fetch_add(1, std::memory_order_relaxed);
      break;
    case ScaledFp8RuntimeOpFamily::kExpertSharedDown:
      stats.scaled_fp8_native_success_expert_shared_down.fetch_add(
          1, std::memory_order_relaxed);
      break;
    case ScaledFp8RuntimeOpFamily::kOther:
      stats.scaled_fp8_native_success_other.fetch_add(1, std::memory_order_relaxed);
      break;
  }
}

void RecordScaledFp8DequantizedDenseSuccess() {
  MutableRuntimeExecutionStats().scaled_fp8_dequantized_dense_success.fetch_add(
      1, std::memory_order_relaxed);
}

void RecordScaledFp8ReferenceFallback(ScaledFp8RuntimeOpFamily family) {
  auto& stats = MutableRuntimeExecutionStats();
  stats.scaled_fp8_reference_fallbacks.fetch_add(1, std::memory_order_relaxed);
  switch (family) {
    case ScaledFp8RuntimeOpFamily::kMambaInProj:
      stats.scaled_fp8_reference_fallbacks_mamba_in_proj.fetch_add(
          1, std::memory_order_relaxed);
      break;
    case ScaledFp8RuntimeOpFamily::kMambaOutProj:
      stats.scaled_fp8_reference_fallbacks_mamba_out_proj.fetch_add(
          1, std::memory_order_relaxed);
      break;
    case ScaledFp8RuntimeOpFamily::kExpertFc1Latent:
      stats.scaled_fp8_reference_fallbacks_expert_fc1_latent.fetch_add(
          1, std::memory_order_relaxed);
      break;
    case ScaledFp8RuntimeOpFamily::kExpertSharedUp:
      stats.scaled_fp8_reference_fallbacks_expert_shared_up.fetch_add(
          1, std::memory_order_relaxed);
      break;
    case ScaledFp8RuntimeOpFamily::kExpertSharedDown:
      stats.scaled_fp8_reference_fallbacks_expert_shared_down.fetch_add(
          1, std::memory_order_relaxed);
      break;
    case ScaledFp8RuntimeOpFamily::kOther:
      stats.scaled_fp8_reference_fallbacks_other.fetch_add(1, std::memory_order_relaxed);
      break;
  }
}

void RecordAttentionDecodePlanCreate() {
  MutableRuntimeExecutionStats().attention_decode_plan_creates.fetch_add(
      1, std::memory_order_relaxed);
}

void RecordAttentionDecodePlanHit() {
  MutableRuntimeExecutionStats().attention_decode_plan_hits.fetch_add(
      1, std::memory_order_relaxed);
}

void RecordAttentionDecodeDevicePageTableCopy() {
  MutableRuntimeExecutionStats().attention_decode_device_page_table_copies.fetch_add(
      1, std::memory_order_relaxed);
}

void RecordExpertSelectionMetadataDownload() {
  MutableRuntimeExecutionStats().expert_selection_metadata_downloads.fetch_add(
      1, std::memory_order_relaxed);
}

void RecordRoutedLookupRepairDownload() {
  MutableRuntimeExecutionStats().routed_lookup_repair_downloads.fetch_add(
      1, std::memory_order_relaxed);
}

void RecordRoutedLookupRepairExperts(std::uint64_t count) {
  MutableRuntimeExecutionStats().routed_lookup_repair_experts.fetch_add(
      count, std::memory_order_relaxed);
}

void RecordRoutedExpertMaterialization() {
  MutableRuntimeExecutionStats().routed_expert_materializations.fetch_add(
      1, std::memory_order_relaxed);
}

void RecordFlashInferRoutedExpertUse() {
  MutableRuntimeExecutionStats().flashinfer_routed_expert_uses.fetch_add(
      1, std::memory_order_relaxed);
}

void RecordFlashInferRoutedExpertFallback() {
  MutableRuntimeExecutionStats().flashinfer_routed_expert_fallbacks.fetch_add(
      1, std::memory_order_relaxed);
}

void RecordGroupedRoutedExpertFastpathUse() {
  MutableRuntimeExecutionStats().grouped_routed_expert_fastpath_uses.fetch_add(
      1, std::memory_order_relaxed);
}

void RecordMoeGraphCapture() {
  MutableRuntimeExecutionStats().moe_graph_captures.fetch_add(1, std::memory_order_relaxed);
}

void RecordMoeGraphReplay() {
  MutableRuntimeExecutionStats().moe_graph_replays.fetch_add(1, std::memory_order_relaxed);
}

void RecordGroupedRoutedExpertFastpathFallback() {
  MutableRuntimeExecutionStats().grouped_routed_expert_fastpath_fallbacks.fetch_add(
      1, std::memory_order_relaxed);
}

void RecordGroupedRoutedExpertPrereqFallback() {
  MutableRuntimeExecutionStats().grouped_routed_expert_prereq_fallbacks.fetch_add(
      1, std::memory_order_relaxed);
}

void RecordGroupedRoutedExpertLookupFallback() {
  MutableRuntimeExecutionStats().grouped_routed_expert_lookup_fallbacks.fetch_add(
      1, std::memory_order_relaxed);
}

void RecordGroupedRoutedExpertPlanFallback() {
  MutableRuntimeExecutionStats().grouped_routed_expert_plan_fallbacks.fetch_add(
      1, std::memory_order_relaxed);
}

void RecordGroupedRoutedExpertPackFallback() {
  MutableRuntimeExecutionStats().grouped_routed_expert_pack_fallbacks.fetch_add(
      1, std::memory_order_relaxed);
}

void RecordGroupedRoutedExpertMatmulFallback() {
  MutableRuntimeExecutionStats().grouped_routed_expert_matmul_fallbacks.fetch_add(
      1, std::memory_order_relaxed);
}

void RecordGroupedRoutedExpertMergeFallback() {
  MutableRuntimeExecutionStats().grouped_routed_expert_merge_fallbacks.fetch_add(
      1, std::memory_order_relaxed);
}

void RecordRoutedExpertPrefetchLayer() {
  MutableRuntimeExecutionStats().routed_expert_prefetch_layers.fetch_add(
      1, std::memory_order_relaxed);
}

void RecordRoutedExpertPrefetchExperts(std::uint64_t count) {
  MutableRuntimeExecutionStats().routed_expert_prefetch_experts.fetch_add(
      count, std::memory_order_relaxed);
}

}  // namespace nemotron
