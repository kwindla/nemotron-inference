#include "nemotron/device_tensor.h"
#include "nemotron/memory_budget.h"
#include "nemotron/model_cache.h"
#include "nemotron/runtime_stats.h"
#include "nemotron/runtime_environment.h"
#include "nemotron/single_token_forward_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t GiB(std::size_t value) {
  return value * 1024ull * 1024ull * 1024ull;
}

struct FixtureMetadata {
  std::optional<std::int32_t> selected_token_id;
};

struct BenchOptions {
  std::filesystem::path manifest_path;
  std::filesystem::path fixture_root =
      "/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/oracle/"
      "full_model_single_token_short_chat_cuda_v3";
  std::filesystem::path json_output;
  std::size_t warmup_iterations = 1;
  std::size_t measured_iterations = 3;
  std::size_t generate_tokens = 1;
  nemotron::ArtifactLoadMode artifact_load_mode = nemotron::ArtifactLoadMode::kMmap;
  bool mmap_prefetch = false;
  std::filesystem::path model_cache_path;
  std::filesystem::path write_model_cache_path;
  std::size_t safety_headroom_bytes = GiB(4);
  std::size_t warning_host_budget_bytes = GiB(20);
  bool abort_on_low_host_budget = false;
};

struct ProcessMemorySnapshot {
  std::size_t vm_rss_bytes = 0;
  std::size_t vm_hwm_bytes = 0;
  std::size_t vm_size_bytes = 0;
  bool valid = false;
};

struct CudaMemorySnapshot {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  bool valid = false;
};

struct BenchMemorySnapshot {
  std::string phase;
  ProcessMemorySnapshot process;
  nemotron::HostMemorySnapshot host;
  CudaMemorySnapshot cuda;
};

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream input(path);
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::optional<std::size_t> ParseJsonUintField(const std::string& json, const std::string& key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9]+)");
  std::smatch match;
  if (!std::regex_search(json, match, pattern) || match.size() != 2) {
    return std::nullopt;
  }
  try {
    return static_cast<std::size_t>(std::stoull(match[1].str()));
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<FixtureMetadata> LoadMetadata(const std::filesystem::path& root) {
  const std::string json = ReadTextFile(root / "metadata.json");
  FixtureMetadata metadata;
  const auto selected_token_id = ParseJsonUintField(json, "selected_token_id");
  if (selected_token_id.has_value()) {
    metadata.selected_token_id = static_cast<std::int32_t>(*selected_token_id);
  }
  return metadata;
}

std::size_t GiBFromEnv(const char* env_name, std::size_t default_value) {
  const char* raw = std::getenv(env_name);
  if (raw == nullptr || *raw == '\0') {
    return default_value;
  }
  try {
    return GiB(static_cast<std::size_t>(std::stoull(raw)));
  } catch (...) {
    return default_value;
  }
}

ProcessMemorySnapshot ReadProcessMemorySnapshot() {
  std::ifstream input("/proc/self/status");
  if (!input) {
    return {};
  }

  ProcessMemorySnapshot snapshot;
  std::string line;
  while (std::getline(input, line)) {
    auto parse_line = [&](const char* prefix, std::size_t* output) {
      if (line.rfind(prefix, 0) != 0 || output == nullptr) {
        return;
      }
      std::istringstream parser(line.substr(std::strlen(prefix)));
      std::size_t value_kib = 0;
      std::string unit;
      if (parser >> value_kib >> unit && unit == "kB") {
        *output = value_kib * 1024ull;
        snapshot.valid = true;
      }
    };
    parse_line("VmRSS:", &snapshot.vm_rss_bytes);
    parse_line("VmHWM:", &snapshot.vm_hwm_bytes);
    parse_line("VmSize:", &snapshot.vm_size_bytes);
  }
  return snapshot;
}

CudaMemorySnapshot ReadCudaMemorySnapshot() {
  CudaMemorySnapshot snapshot;
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
    snapshot.free_bytes = free_bytes;
    snapshot.total_bytes = total_bytes;
    snapshot.valid = true;
  }
  return snapshot;
}

BenchMemorySnapshot CaptureMemorySnapshot(const std::string& phase) {
  BenchMemorySnapshot snapshot;
  snapshot.phase = phase;
  snapshot.process = ReadProcessMemorySnapshot();
  snapshot.host = nemotron::ReadHostMemorySnapshotFromProcMeminfo("/proc/meminfo");
  snapshot.cuda = ReadCudaMemorySnapshot();
  return snapshot;
}

std::size_t HostBudgetBytes(const nemotron::HostMemorySnapshot& snapshot) {
  return snapshot.mem_available_bytes + snapshot.swap_free_bytes;
}

double BytesToGiB(std::size_t bytes) {
  return static_cast<double>(bytes) / static_cast<double>(GiB(1));
}

void EmitMemorySnapshot(const BenchMemorySnapshot& snapshot, const BenchOptions& options) {
  std::cerr << std::fixed << std::setprecision(3)
            << "memory_phase=" << snapshot.phase;
  if (snapshot.process.valid) {
    std::cerr << " proc_vm_rss_gib=" << BytesToGiB(snapshot.process.vm_rss_bytes)
              << " proc_vm_hwm_gib=" << BytesToGiB(snapshot.process.vm_hwm_bytes)
              << " proc_vm_size_gib=" << BytesToGiB(snapshot.process.vm_size_bytes);
  }
  if (snapshot.host.valid) {
    const std::size_t host_budget_bytes = HostBudgetBytes(snapshot.host);
    std::cerr << " host_mem_available_gib=" << BytesToGiB(snapshot.host.mem_available_bytes)
              << " host_swap_free_gib=" << BytesToGiB(snapshot.host.swap_free_bytes)
              << " host_budget_gib=" << BytesToGiB(host_budget_bytes);
    if (host_budget_bytes < options.warning_host_budget_bytes) {
      std::cerr << " warning=low_host_budget";
    }
  }
  if (snapshot.cuda.valid) {
    std::cerr << " cuda_free_gib=" << BytesToGiB(snapshot.cuda.free_bytes)
              << " cuda_total_gib=" << BytesToGiB(snapshot.cuda.total_bytes);
  }
  std::cerr << "\n" << std::flush;
}

nemotron::RuntimeBootstrapOptions MakeOptions(const BenchOptions& bench_options) {
  nemotron::RuntimeBootstrapOptions options;
  options.service_target.total_memory_bytes = GiB(128);
  options.service_target.weights_bytes = GiB(100);
  options.service_target.workspace_bytes = GiB(8);
  options.service_target.graph_bytes = GiB(4);
  options.service_target.safety_headroom_bytes = bench_options.safety_headroom_bytes;
  options.service_target.target_active_requests = 1;
  options.service_target.target_context_tokens = bench_options.generate_tokens;
  options.use_fp16_mamba_state = false;
  options.reusable_node_metadata_bytes = 4096;
  options.prefer_host_memory_snapshot = true;
  options.verify_manifest_files = false;
  options.materialize_weight_arena = false;
  options.artifact_load_mode = bench_options.artifact_load_mode;
  options.prefetch_mapped_artifacts =
      bench_options.artifact_load_mode == nemotron::ArtifactLoadMode::kMmap &&
      bench_options.mmap_prefetch;
  return options;
}

bool HasCudaDevice() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

std::int32_t Argmax(const std::vector<float>& values) {
  if (values.empty()) {
    return -1;
  }
  const auto it = std::max_element(values.begin(), values.end());
  return static_cast<std::int32_t>(std::distance(values.begin(), it));
}

double MeanMs(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double MaxMs(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  return *std::max_element(values.begin(), values.end());
}

double MinMs(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  return *std::min_element(values.begin(), values.end());
}

const char* LayerKindName(nemotron::ForwardLayerKind kind) {
  switch (kind) {
    case nemotron::ForwardLayerKind::kAttention:
      return "attention";
    case nemotron::ForwardLayerKind::kMamba:
      return "mamba";
    case nemotron::ForwardLayerKind::kExpert:
      return "expert";
  }
  return "unknown";
}

struct StepRuntimeStatsDelta {
  std::uint64_t expert_selection_metadata_downloads = 0;
  std::uint64_t routed_lookup_repair_downloads = 0;
  std::uint64_t routed_lookup_repair_experts = 0;
  std::uint64_t grouped_routed_expert_fastpath_uses = 0;
};

struct StepGraphReadinessSummary {
  std::uint64_t graph_safe_steps = 0;
  std::uint64_t max_graph_safe_streak = 0;
  std::uint64_t tail_graph_safe_streak = 0;
  std::int64_t first_graph_safe_tail_token_index = -1;
};

StepRuntimeStatsDelta ComputeStepRuntimeStatsDelta(
    const nemotron::RuntimeExecutionStatsSnapshot& before,
    const nemotron::RuntimeExecutionStatsSnapshot& after) {
  return StepRuntimeStatsDelta{
      after.expert_selection_metadata_downloads - before.expert_selection_metadata_downloads,
      after.routed_lookup_repair_downloads - before.routed_lookup_repair_downloads,
      after.routed_lookup_repair_experts - before.routed_lookup_repair_experts,
      after.grouped_routed_expert_fastpath_uses - before.grouped_routed_expert_fastpath_uses,
  };
}

bool IsGraphSafeStep(const StepRuntimeStatsDelta& stats) {
  return stats.expert_selection_metadata_downloads == 0 &&
         stats.routed_lookup_repair_downloads == 0 &&
         stats.routed_lookup_repair_experts == 0;
}

StepGraphReadinessSummary SummarizeGraphReadiness(
    const std::vector<StepRuntimeStatsDelta>& steps) {
  StepGraphReadinessSummary summary;
  std::uint64_t current_streak = 0;
  for (std::size_t i = 0; i < steps.size(); ++i) {
    if (!IsGraphSafeStep(steps[i])) {
      current_streak = 0;
      continue;
    }
    ++summary.graph_safe_steps;
    ++current_streak;
    summary.max_graph_safe_streak = std::max(summary.max_graph_safe_streak, current_streak);
  }
  for (std::size_t i = steps.size(); i > 0; --i) {
    if (!IsGraphSafeStep(steps[i - 1])) {
      break;
    }
    ++summary.tail_graph_safe_streak;
  }
  if (summary.tail_graph_safe_streak != 0) {
    summary.first_graph_safe_tail_token_index =
        static_cast<std::int64_t>(steps.size() - summary.tail_graph_safe_streak);
  }
  return summary;
}

bool WriteJsonReport(
    const BenchOptions& options,
    double environment_build_ms,
    double model_build_ms,
    const nemotron::SingleTokenForwardBuildReport& build_report,
    const nemotron::RuntimeExecutionStatsSnapshot& runtime_stats,
    const std::vector<BenchMemorySnapshot>& memory_snapshots,
    const std::vector<double>& warmup_ms,
    const std::vector<double>& hot_ms,
    const std::vector<std::vector<double>>& warmup_step_ms,
    const std::vector<std::vector<double>>& hot_step_ms,
    const std::vector<std::vector<StepRuntimeStatsDelta>>& warmup_step_runtime_stats,
    const std::vector<std::vector<StepRuntimeStatsDelta>>& hot_step_runtime_stats,
    const std::vector<StepGraphReadinessSummary>& warmup_graph_readiness,
    const std::vector<StepGraphReadinessSummary>& hot_graph_readiness,
    std::int32_t selected_token_id,
    std::int32_t predicted_token_id,
    const std::vector<std::int32_t>& generated_token_ids) {
  if (options.json_output.empty()) {
    return true;
  }
  if (!options.json_output.parent_path().empty()) {
    std::filesystem::create_directories(options.json_output.parent_path());
  }
  std::ofstream output(options.json_output);
  if (!output) {
    return false;
  }

  output << std::fixed << std::setprecision(6);
  output << "{\n";
  output << "  \"manifest_path\": \"" << options.manifest_path.string() << "\",\n";
  output << "  \"fixture_root\": \"" << options.fixture_root.string() << "\",\n";
  output << "  \"generate_tokens\": " << options.generate_tokens << ",\n";
  output << "  \"artifact_load_mode\": \"" << nemotron::ToString(options.artifact_load_mode)
         << "\",\n";
  output << "  \"mmap_prefetch\": " << (options.mmap_prefetch ? "true" : "false") << ",\n";
  output << "  \"model_cache_path\": \"" << options.model_cache_path.string() << "\",\n";
  output << "  \"write_model_cache_path\": \"" << options.write_model_cache_path.string() << "\",\n";
  output << "  \"environment_build_ms\": " << environment_build_ms << ",\n";
  output << "  \"model_build_ms\": " << model_build_ms << ",\n";
  output << "  \"runtime_stats\": {\n";
  output << "    \"dense_plan_cache_hits\": " << runtime_stats.dense_plan_cache_hits << ",\n";
  output << "    \"dense_plan_build_failures\": " << runtime_stats.dense_plan_build_failures << ",\n";
  output << "    \"dense_plan_build_failures_attention\": "
         << runtime_stats.dense_plan_build_failures_attention << ",\n";
  output << "    \"dense_plan_build_failures_expert\": "
         << runtime_stats.dense_plan_build_failures_expert << ",\n";
  output << "    \"dense_plan_build_failures_other\": "
         << runtime_stats.dense_plan_build_failures_other << ",\n";
  output << "    \"dense_native_success\": " << runtime_stats.dense_native_success << ",\n";
  output << "    \"dense_native_success_attention\": "
         << runtime_stats.dense_native_success_attention << ",\n";
  output << "    \"dense_native_success_expert\": "
         << runtime_stats.dense_native_success_expert << ",\n";
  output << "    \"dense_native_success_other\": "
         << runtime_stats.dense_native_success_other << ",\n";
  output << "    \"dense_reference_fallbacks\": " << runtime_stats.dense_reference_fallbacks << ",\n";
  output << "    \"dense_reference_fallbacks_attention\": "
         << runtime_stats.dense_reference_fallbacks_attention << ",\n";
  output << "    \"dense_reference_fallbacks_expert\": "
         << runtime_stats.dense_reference_fallbacks_expert << ",\n";
  output << "    \"dense_reference_fallbacks_other\": "
         << runtime_stats.dense_reference_fallbacks_other << ",\n";
  output << "    \"dense_bf16_native_failures\": " << runtime_stats.dense_bf16_native_failures << ",\n";
  output << "    \"dense_fp32_native_failures\": " << runtime_stats.dense_fp32_native_failures << ",\n";
  output << "    \"scaled_fp8_plan_cache_hits\": " << runtime_stats.scaled_fp8_plan_cache_hits << ",\n";
  output << "    \"scaled_fp8_plan_build_failures\": "
         << runtime_stats.scaled_fp8_plan_build_failures << ",\n";
  output << "    \"scaled_fp8_plan_build_failures_mamba_in_proj\": "
         << runtime_stats.scaled_fp8_plan_build_failures_mamba_in_proj << ",\n";
  output << "    \"scaled_fp8_plan_build_failures_mamba_out_proj\": "
         << runtime_stats.scaled_fp8_plan_build_failures_mamba_out_proj << ",\n";
  output << "    \"scaled_fp8_plan_build_failures_expert_fc1_latent\": "
         << runtime_stats.scaled_fp8_plan_build_failures_expert_fc1_latent << ",\n";
  output << "    \"scaled_fp8_plan_build_failures_expert_shared_up\": "
         << runtime_stats.scaled_fp8_plan_build_failures_expert_shared_up << ",\n";
  output << "    \"scaled_fp8_plan_build_failures_expert_shared_down\": "
         << runtime_stats.scaled_fp8_plan_build_failures_expert_shared_down << ",\n";
  output << "    \"scaled_fp8_plan_build_failures_other\": "
         << runtime_stats.scaled_fp8_plan_build_failures_other << ",\n";
  output << "    \"scaled_fp8_native_success\": " << runtime_stats.scaled_fp8_native_success << ",\n";
  output << "    \"scaled_fp8_native_success_mamba_in_proj\": "
         << runtime_stats.scaled_fp8_native_success_mamba_in_proj << ",\n";
  output << "    \"scaled_fp8_native_success_mamba_out_proj\": "
         << runtime_stats.scaled_fp8_native_success_mamba_out_proj << ",\n";
  output << "    \"scaled_fp8_native_success_expert_fc1_latent\": "
         << runtime_stats.scaled_fp8_native_success_expert_fc1_latent << ",\n";
  output << "    \"scaled_fp8_native_success_expert_shared_up\": "
         << runtime_stats.scaled_fp8_native_success_expert_shared_up << ",\n";
  output << "    \"scaled_fp8_native_success_expert_shared_down\": "
         << runtime_stats.scaled_fp8_native_success_expert_shared_down << ",\n";
  output << "    \"scaled_fp8_native_success_other\": "
         << runtime_stats.scaled_fp8_native_success_other << ",\n";
  output << "    \"scaled_fp8_dequantized_dense_success\": "
         << runtime_stats.scaled_fp8_dequantized_dense_success << ",\n";
  output << "    \"scaled_fp8_reference_fallbacks\": " << runtime_stats.scaled_fp8_reference_fallbacks << ",\n";
  output << "    \"scaled_fp8_reference_fallbacks_mamba_in_proj\": "
         << runtime_stats.scaled_fp8_reference_fallbacks_mamba_in_proj << ",\n";
  output << "    \"scaled_fp8_reference_fallbacks_mamba_out_proj\": "
         << runtime_stats.scaled_fp8_reference_fallbacks_mamba_out_proj << ",\n";
  output << "    \"scaled_fp8_reference_fallbacks_expert_fc1_latent\": "
         << runtime_stats.scaled_fp8_reference_fallbacks_expert_fc1_latent << ",\n";
  output << "    \"scaled_fp8_reference_fallbacks_expert_shared_up\": "
         << runtime_stats.scaled_fp8_reference_fallbacks_expert_shared_up << ",\n";
  output << "    \"scaled_fp8_reference_fallbacks_expert_shared_down\": "
         << runtime_stats.scaled_fp8_reference_fallbacks_expert_shared_down << ",\n";
  output << "    \"scaled_fp8_reference_fallbacks_other\": "
         << runtime_stats.scaled_fp8_reference_fallbacks_other << ",\n";
  output << "    \"attention_decode_plan_creates\": " << runtime_stats.attention_decode_plan_creates << ",\n";
  output << "    \"attention_decode_plan_hits\": " << runtime_stats.attention_decode_plan_hits << ",\n";
  output << "    \"attention_decode_device_page_table_copies\": "
         << runtime_stats.attention_decode_device_page_table_copies << ",\n";
  output << "    \"expert_selection_metadata_downloads\": "
         << runtime_stats.expert_selection_metadata_downloads << ",\n";
  output << "    \"routed_lookup_repair_downloads\": "
         << runtime_stats.routed_lookup_repair_downloads << ",\n";
  output << "    \"routed_lookup_repair_experts\": "
         << runtime_stats.routed_lookup_repair_experts << ",\n";
  output << "    \"routed_expert_materializations\": "
         << runtime_stats.routed_expert_materializations << ",\n";
  output << "    \"flashinfer_routed_expert_uses\": "
         << runtime_stats.flashinfer_routed_expert_uses << ",\n";
  output << "    \"flashinfer_routed_expert_fallbacks\": "
         << runtime_stats.flashinfer_routed_expert_fallbacks << ",\n";
  output << "    \"grouped_routed_expert_fastpath_uses\": "
         << runtime_stats.grouped_routed_expert_fastpath_uses << ",\n";
  output << "    \"moe_graph_captures\": " << runtime_stats.moe_graph_captures << ",\n";
  output << "    \"moe_graph_replays\": " << runtime_stats.moe_graph_replays << ",\n";
  output << "    \"forward_graph_captures\": " << runtime_stats.forward_graph_captures << ",\n";
  output << "    \"forward_graph_replays\": " << runtime_stats.forward_graph_replays << ",\n";
  output << "    \"grouped_routed_expert_fastpath_fallbacks\": "
         << runtime_stats.grouped_routed_expert_fastpath_fallbacks << ",\n";
  output << "    \"grouped_routed_expert_prereq_fallbacks\": "
         << runtime_stats.grouped_routed_expert_prereq_fallbacks << ",\n";
  output << "    \"grouped_routed_expert_lookup_fallbacks\": "
         << runtime_stats.grouped_routed_expert_lookup_fallbacks << ",\n";
  output << "    \"grouped_routed_expert_plan_fallbacks\": "
         << runtime_stats.grouped_routed_expert_plan_fallbacks << ",\n";
  output << "    \"grouped_routed_expert_pack_fallbacks\": "
         << runtime_stats.grouped_routed_expert_pack_fallbacks << ",\n";
  output << "    \"grouped_routed_expert_matmul_fallbacks\": "
         << runtime_stats.grouped_routed_expert_matmul_fallbacks << ",\n";
  output << "    \"grouped_routed_expert_merge_fallbacks\": "
         << runtime_stats.grouped_routed_expert_merge_fallbacks << ",\n";
  output << "    \"routed_expert_prefetch_layers\": "
         << runtime_stats.routed_expert_prefetch_layers << ",\n";
  output << "    \"routed_expert_prefetch_experts\": "
         << runtime_stats.routed_expert_prefetch_experts << "\n";
  output << "  },\n";
  output << "  \"memory_snapshots\": [\n";
  for (std::size_t i = 0; i < memory_snapshots.size(); ++i) {
    const auto& snapshot = memory_snapshots[i];
    output << "    {\"phase\": \"" << snapshot.phase << "\""
           << ", \"process_valid\": " << (snapshot.process.valid ? "true" : "false")
           << ", \"proc_vm_rss_bytes\": " << snapshot.process.vm_rss_bytes
           << ", \"proc_vm_hwm_bytes\": " << snapshot.process.vm_hwm_bytes
           << ", \"proc_vm_size_bytes\": " << snapshot.process.vm_size_bytes
           << ", \"host_valid\": " << (snapshot.host.valid ? "true" : "false")
           << ", \"host_mem_available_bytes\": " << snapshot.host.mem_available_bytes
           << ", \"host_swap_free_bytes\": " << snapshot.host.swap_free_bytes
           << ", \"host_budget_bytes\": " << HostBudgetBytes(snapshot.host)
           << ", \"cuda_valid\": " << (snapshot.cuda.valid ? "true" : "false")
           << ", \"cuda_free_bytes\": " << snapshot.cuda.free_bytes
           << ", \"cuda_total_bytes\": " << snapshot.cuda.total_bytes
           << "}";
    output << (i + 1 == memory_snapshots.size() ? "\n" : ",\n");
  }
  output << "  ],\n";
  output << "  \"build_phases\": [\n";
  for (std::size_t i = 0; i < build_report.phases.size(); ++i) {
    const auto& phase = build_report.phases[i];
    output << "    {\"name\": \"" << phase.name << "\", \"milliseconds\": " << phase.milliseconds << "}";
    output << (i + 1 == build_report.phases.size() ? "\n" : ",\n");
  }
  output << "  ],\n";
  output << "  \"build_layers\": [\n";
  for (std::size_t i = 0; i < build_report.layers.size(); ++i) {
    const auto& layer = build_report.layers[i];
    output << "    {\"layer_index\": " << layer.layer_index
           << ", \"kind\": \"" << LayerKindName(layer.kind)
           << "\", \"bindings_ms\": " << layer.bindings_ms
           << ", \"slice_create_ms\": " << layer.slice_create_ms
           << ", \"total_ms\": " << layer.total_ms
           << ", \"details\": [";
    for (std::size_t detail_index = 0; detail_index < layer.details.size(); ++detail_index) {
      if (detail_index != 0) {
        output << ", ";
      }
      const auto& detail = layer.details[detail_index];
      output << "{\"name\": \"" << detail.name
             << "\", \"milliseconds\": " << detail.milliseconds << "}";
    }
    output << "]}";
    output << (i + 1 == build_report.layers.size() ? "\n" : ",\n");
  }
  output << "  ],\n";
  output << "  \"warmup_iterations\": " << options.warmup_iterations << ",\n";
  output << "  \"measured_iterations\": " << options.measured_iterations << ",\n";
  output << "  \"warmup_mean_ms\": " << MeanMs(warmup_ms) << ",\n";
  output << "  \"warmup_min_ms\": " << MinMs(warmup_ms) << ",\n";
  output << "  \"warmup_max_ms\": " << MaxMs(warmup_ms) << ",\n";
  output << "  \"hot_mean_ms\": " << MeanMs(hot_ms) << ",\n";
  output << "  \"hot_min_ms\": " << MinMs(hot_ms) << ",\n";
  output << "  \"hot_max_ms\": " << MaxMs(hot_ms) << ",\n";
  output << "  \"selected_token_id\": " << selected_token_id << ",\n";
  output << "  \"predicted_token_id\": " << predicted_token_id << ",\n";
  output << "  \"generated_token_ids\": [";
  for (std::size_t i = 0; i < generated_token_ids.size(); ++i) {
    if (i != 0) {
      output << ", ";
    }
    output << generated_token_ids[i];
  }
  output << "],\n";
  output << "  \"warmup_ms\": [";
  for (std::size_t i = 0; i < warmup_ms.size(); ++i) {
    if (i != 0) {
      output << ", ";
    }
    output << warmup_ms[i];
  }
  output << "],\n";
  output << "  \"hot_ms\": [";
  for (std::size_t i = 0; i < hot_ms.size(); ++i) {
    if (i != 0) {
      output << ", ";
    }
    output << hot_ms[i];
  }
  output << "],\n";
  output << "  \"warmup_step_ms\": [";
  for (std::size_t i = 0; i < warmup_step_ms.size(); ++i) {
    if (i != 0) {
      output << ", ";
    }
    output << "[";
    for (std::size_t j = 0; j < warmup_step_ms[i].size(); ++j) {
      if (j != 0) {
        output << ", ";
      }
      output << warmup_step_ms[i][j];
    }
    output << "]";
  }
  output << "],\n";
  output << "  \"hot_step_ms\": [";
  for (std::size_t i = 0; i < hot_step_ms.size(); ++i) {
    if (i != 0) {
      output << ", ";
    }
    output << "[";
    for (std::size_t j = 0; j < hot_step_ms[i].size(); ++j) {
      if (j != 0) {
        output << ", ";
      }
      output << hot_step_ms[i][j];
    }
    output << "]";
  }
  output << "],\n";
  output << "  \"warmup_step_runtime_stats\": [";
  for (std::size_t i = 0; i < warmup_step_runtime_stats.size(); ++i) {
    if (i != 0) {
      output << ", ";
    }
    output << "[";
    for (std::size_t j = 0; j < warmup_step_runtime_stats[i].size(); ++j) {
      if (j != 0) {
        output << ", ";
      }
      const auto& stats = warmup_step_runtime_stats[i][j];
      output << "{\"expert_selection_metadata_downloads\": "
             << stats.expert_selection_metadata_downloads
             << ", \"routed_lookup_repair_downloads\": "
             << stats.routed_lookup_repair_downloads
             << ", \"routed_lookup_repair_experts\": "
             << stats.routed_lookup_repair_experts
             << ", \"grouped_routed_expert_fastpath_uses\": "
             << stats.grouped_routed_expert_fastpath_uses
             << "}";
    }
    output << "]";
  }
  output << "],\n";
  output << "  \"hot_step_runtime_stats\": [";
  for (std::size_t i = 0; i < hot_step_runtime_stats.size(); ++i) {
    if (i != 0) {
      output << ", ";
    }
    output << "[";
    for (std::size_t j = 0; j < hot_step_runtime_stats[i].size(); ++j) {
      if (j != 0) {
        output << ", ";
      }
      const auto& stats = hot_step_runtime_stats[i][j];
      output << "{\"expert_selection_metadata_downloads\": "
             << stats.expert_selection_metadata_downloads
             << ", \"routed_lookup_repair_downloads\": "
             << stats.routed_lookup_repair_downloads
             << ", \"routed_lookup_repair_experts\": "
             << stats.routed_lookup_repair_experts
             << ", \"grouped_routed_expert_fastpath_uses\": "
             << stats.grouped_routed_expert_fastpath_uses
             << "}";
    }
    output << "]";
  }
  output << "],\n";
  output << "  \"warmup_graph_readiness\": [";
  for (std::size_t i = 0; i < warmup_graph_readiness.size(); ++i) {
    if (i != 0) {
      output << ", ";
    }
    const auto& summary = warmup_graph_readiness[i];
    output << "{\"graph_safe_steps\": " << summary.graph_safe_steps
           << ", \"max_graph_safe_streak\": " << summary.max_graph_safe_streak
           << ", \"tail_graph_safe_streak\": " << summary.tail_graph_safe_streak
           << ", \"first_graph_safe_tail_token_index\": "
           << summary.first_graph_safe_tail_token_index
           << "}";
  }
  output << "],\n";
  output << "  \"hot_graph_readiness\": [";
  for (std::size_t i = 0; i < hot_graph_readiness.size(); ++i) {
    if (i != 0) {
      output << ", ";
    }
    const auto& summary = hot_graph_readiness[i];
    output << "{\"graph_safe_steps\": " << summary.graph_safe_steps
           << ", \"max_graph_safe_streak\": " << summary.max_graph_safe_streak
           << ", \"tail_graph_safe_streak\": " << summary.tail_graph_safe_streak
           << ", \"first_graph_safe_tail_token_index\": "
           << summary.first_graph_safe_tail_token_index
           << "}";
  }
  output << "]\n";
  output << "}\n";
  return true;
}

bool ParseArgs(int argc, char** argv, BenchOptions* options) {
  if (options == nullptr) {
    return false;
  }
  options->safety_headroom_bytes =
      GiBFromEnv("NEMOTRON_BENCH_SAFETY_HEADROOM_GIB", options->safety_headroom_bytes / GiB(1));
  options->warning_host_budget_bytes =
      GiBFromEnv(
          "NEMOTRON_BENCH_WARNING_HOST_BUDGET_GIB",
          options->warning_host_budget_bytes / GiB(1));
  options->abort_on_low_host_budget =
      std::getenv("NEMOTRON_BENCH_ABORT_ON_LOW_HOST_BUDGET") != nullptr;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--manifest" && i + 1 < argc) {
      options->manifest_path = argv[++i];
      continue;
    }
    if (arg == "--fixture-root" && i + 1 < argc) {
      options->fixture_root = argv[++i];
      continue;
    }
    if (arg == "--json-output" && i + 1 < argc) {
      options->json_output = argv[++i];
      continue;
    }
    if (arg == "--warmup" && i + 1 < argc) {
      options->warmup_iterations = static_cast<std::size_t>(std::stoull(argv[++i]));
      continue;
    }
    if (arg == "--iterations" && i + 1 < argc) {
      options->measured_iterations = static_cast<std::size_t>(std::stoull(argv[++i]));
      continue;
    }
    if (arg == "--generate-tokens" && i + 1 < argc) {
      options->generate_tokens = static_cast<std::size_t>(std::stoull(argv[++i]));
      continue;
    }
    if (arg == "--artifact-load-mode" && i + 1 < argc) {
      const std::string mode(argv[++i]);
      if (mode == "mmap") {
        options->artifact_load_mode = nemotron::ArtifactLoadMode::kMmap;
      } else if (mode == "readall") {
        options->artifact_load_mode = nemotron::ArtifactLoadMode::kReadAll;
      } else {
        std::cerr << "unknown artifact load mode: " << mode << "\n";
        return false;
      }
      continue;
    }
    if (arg == "--mmap-prefetch") {
      options->mmap_prefetch = true;
      continue;
    }
    if (arg == "--model-cache" && i + 1 < argc) {
      options->model_cache_path = argv[++i];
      continue;
    }
    if (arg == "--write-model-cache" && i + 1 < argc) {
      options->write_model_cache_path = argv[++i];
      continue;
    }
    std::cerr << "unknown argument: " << arg << "\n";
    return false;
  }

  if (options->manifest_path.empty()) {
    const char* manifest_env = std::getenv("NEMOTRON_FORWARD_MANIFEST");
    if (manifest_env != nullptr && std::string(manifest_env).size() != 0) {
      options->manifest_path = manifest_env;
    }
  }
  return !options->manifest_path.empty();
}

}  // namespace

int main(int argc, char** argv) {
  BenchOptions options;
  if (!ParseArgs(argc, argv, &options)) {
    std::cerr
        << "usage: single_token_decode_bench --manifest /path/to/manifest.json "
        << "[--fixture-root /path/to/oracle/root] [--warmup 1] [--iterations 3] "
        << "[--generate-tokens 1] "
        << "[--artifact-load-mode mmap|readall] "
        << "[--mmap-prefetch] "
        << "[--model-cache /path/to/model.cache] "
        << "[--write-model-cache /path/to/model.cache] "
        << "[--json-output /path/to/report.json]\n";
    return 2;
  }
  if (!HasCudaDevice()) {
    std::cerr << "single_token_decode_bench: no CUDA device available\n";
    return 1;
  }
  if (!std::filesystem::exists(options.manifest_path)) {
    std::cerr << "single_token_decode_bench: manifest does not exist: "
              << options.manifest_path << "\n";
    return 1;
  }

  const auto metadata = LoadMetadata(options.fixture_root);
  if (!metadata.has_value() || !metadata->selected_token_id.has_value()) {
    std::cerr << "single_token_decode_bench: failed to load selected_token_id from fixture root "
              << options.fixture_root << "\n";
    return 1;
  }

  std::vector<BenchMemorySnapshot> memory_snapshots;
  memory_snapshots.push_back(CaptureMemorySnapshot("pre_environment_build"));
  EmitMemorySnapshot(memory_snapshots.back(), options);
  if (memory_snapshots.back().host.valid &&
      HostBudgetBytes(memory_snapshots.back().host) < options.warning_host_budget_bytes) {
    std::cerr << "single_token_decode_bench: host budget "
              << BytesToGiB(HostBudgetBytes(memory_snapshots.back().host))
              << " GiB is below warning threshold "
              << BytesToGiB(options.warning_host_budget_bytes) << " GiB\n";
    if (options.abort_on_low_host_budget) {
      std::cerr << "single_token_decode_bench: aborting due to "
                   "NEMOTRON_BENCH_ABORT_ON_LOW_HOST_BUDGET\n";
      return 1;
    }
  }

  const auto environment_start = std::chrono::steady_clock::now();
  auto environment =
      nemotron::RuntimeEnvironment::BuildFromManifestFile(
          options.manifest_path,
          MakeOptions(options));
  const auto environment_end = std::chrono::steady_clock::now();
  if (!environment) {
    std::cerr << "single_token_decode_bench: runtime environment build failed\n";
    return 1;
  }
  const double environment_build_ms =
      std::chrono::duration<double, std::milli>(environment_end - environment_start).count();
  memory_snapshots.push_back(CaptureMemorySnapshot("post_environment_build"));
  EmitMemorySnapshot(memory_snapshots.back(), options);
  std::cerr << std::fixed << std::setprecision(3)
            << "environment_build_ms=" << environment_build_ms << "\n"
            << std::flush;

  nemotron::SingleTokenForwardConfig config = nemotron::KnownNemotron3Super120BA12BConfig();
  config.max_tokens = std::max<std::size_t>(1, options.generate_tokens);
  if (!options.write_model_cache_path.empty()) {
    nemotron::ModelCacheWriteReport cache_report;
    if (!nemotron::WriteDeterministicModelCache(
            *environment,
            config,
            options.write_model_cache_path,
            &cache_report)) {
      std::cerr << "single_token_decode_bench: model cache write failed\n";
      return 1;
    }
    std::cerr << "model_cache_entry_count=" << cache_report.entry_count << "\n" << std::flush;
    std::cerr << "model_cache_payload_nbytes=" << cache_report.payload_nbytes << "\n" << std::flush;
  }
  const auto model_start = std::chrono::steady_clock::now();
  auto model = options.model_cache_path.empty()
      ? nemotron::SingleTokenForwardModel::Create(*environment, config)
      : nemotron::SingleTokenForwardModel::CreateFromCache(
            *environment,
            config,
            options.model_cache_path);
  const auto model_end = std::chrono::steady_clock::now();
  if (!model || !model->valid()) {
    std::cerr << "single_token_decode_bench: forward model build failed\n";
    return 1;
  }
  const double model_build_ms =
      std::chrono::duration<double, std::milli>(model_end - model_start).count();
  memory_snapshots.push_back(CaptureMemorySnapshot("post_model_build"));
  EmitMemorySnapshot(memory_snapshots.back(), options);
  std::cerr << "model_build_ms=" << model_build_ms << "\n" << std::flush;
  for (const auto& phase : model->build_report().phases) {
    std::cerr << "build_phase_" << phase.name << "_ms=" << phase.milliseconds << "\n" << std::flush;
  }
  for (const auto& layer : model->build_report().layers) {
    std::cerr << "build_layer_" << layer.layer_index << "_" << LayerKindName(layer.kind)
              << "_bindings_ms=" << layer.bindings_ms
              << " slice_create_ms=" << layer.slice_create_ms
              << " total_ms=" << layer.total_ms << "\n"
              << std::flush;
  }
  std::cout << std::fixed << std::setprecision(3)
            << "environment_build_ms=" << environment_build_ms << "\n"
            << "model_build_ms=" << model_build_ms << "\n"
            << std::flush;

  nemotron::ResetRuntimeExecutionStats();

  std::vector<double> warmup_ms;
  std::vector<double> hot_ms;
  std::vector<std::vector<double>> warmup_step_ms;
  std::vector<std::vector<double>> hot_step_ms;
  std::vector<std::vector<StepRuntimeStatsDelta>> warmup_step_runtime_stats;
  std::vector<std::vector<StepRuntimeStatsDelta>> hot_step_runtime_stats;
  std::vector<StepGraphReadinessSummary> warmup_graph_readiness;
  std::vector<StepGraphReadinessSummary> hot_graph_readiness;
  warmup_ms.reserve(options.warmup_iterations);
  hot_ms.reserve(options.measured_iterations);
  warmup_step_ms.reserve(options.warmup_iterations);
  hot_step_ms.reserve(options.measured_iterations);
  warmup_step_runtime_stats.reserve(options.warmup_iterations);
  hot_step_runtime_stats.reserve(options.measured_iterations);
  warmup_graph_readiness.reserve(options.warmup_iterations);
  hot_graph_readiness.reserve(options.measured_iterations);
  std::vector<float> last_logits_host;
  std::vector<std::int32_t> last_generated_token_ids;

  const auto run_iteration =
      [&](std::vector<double>* timings,
          std::vector<std::vector<double>>* step_timings,
          std::vector<std::vector<StepRuntimeStatsDelta>>* step_runtime_stats,
          std::vector<StepGraphReadinessSummary>* graph_readiness,
          const char* phase,
          std::size_t index) -> bool {
    auto request_context = model->CreateRequestContext();
    auto logits = nemotron::DeviceTensorFp32::Create({1, config.vocab_size});
    if (!request_context || !request_context->valid() || !logits || !logits->valid()) {
      return false;
    }
    std::vector<double> per_step_ms;
    std::vector<StepRuntimeStatsDelta> per_step_runtime_stats;
    std::vector<std::int32_t> generated_token_ids;
    per_step_ms.reserve(options.generate_tokens);
    per_step_runtime_stats.reserve(options.generate_tokens);
    generated_token_ids.reserve(options.generate_tokens);
    std::int32_t current_token = *metadata->selected_token_id;
    const auto sequence_begin = std::chrono::steady_clock::now();
    for (std::size_t step = 0; step < options.generate_tokens; ++step) {
      cudaDeviceSynchronize();
      const auto begin = std::chrono::steady_clock::now();
      const auto runtime_stats_before = nemotron::GetRuntimeExecutionStatsSnapshot();
      const bool ok = model->RunDecodeStep(current_token, *request_context, logits.get());
      const auto sync_status = cudaDeviceSynchronize();
      const auto end = std::chrono::steady_clock::now();
      if (!ok || sync_status != cudaSuccess) {
        return false;
      }
      const double elapsed_ms =
          std::chrono::duration<double, std::milli>(end - begin).count();
      last_logits_host.assign(logits->numel(), 0.0f);
      if (!logits->CopyToHost(last_logits_host.data(), last_logits_host.size())) {
        return false;
      }
      const std::int32_t predicted = Argmax(last_logits_host);
      current_token = predicted;
      per_step_ms.push_back(elapsed_ms);
      per_step_runtime_stats.push_back(
          ComputeStepRuntimeStatsDelta(
              runtime_stats_before,
              nemotron::GetRuntimeExecutionStatsSnapshot()));
      generated_token_ids.push_back(predicted);
    }
    const auto sequence_end = std::chrono::steady_clock::now();
    const double sequence_elapsed_ms =
        std::chrono::duration<double, std::milli>(sequence_end - sequence_begin).count();
    timings->push_back(sequence_elapsed_ms);
    step_timings->push_back(per_step_ms);
    step_runtime_stats->push_back(std::move(per_step_runtime_stats));
    if (graph_readiness != nullptr) {
      graph_readiness->push_back(SummarizeGraphReadiness(step_runtime_stats->back()));
    }
    last_generated_token_ids = generated_token_ids;
    std::cerr << phase << "_" << index << "_ms=" << sequence_elapsed_ms << "\n" << std::flush;
    std::cout << phase << "_" << index << "_ms=" << sequence_elapsed_ms << "\n" << std::flush;
    return true;
  };

  for (std::size_t i = 0; i < options.warmup_iterations; ++i) {
    if (!run_iteration(
            &warmup_ms,
            &warmup_step_ms,
            &warmup_step_runtime_stats,
            &warmup_graph_readiness,
            "warmup",
            i)) {
      std::cerr << "single_token_decode_bench: warmup iteration failed\n";
      return 1;
    }
  }
  memory_snapshots.push_back(CaptureMemorySnapshot("post_warmup"));
  EmitMemorySnapshot(memory_snapshots.back(), options);
  for (std::size_t i = 0; i < options.measured_iterations; ++i) {
    if (!run_iteration(
            &hot_ms,
            &hot_step_ms,
            &hot_step_runtime_stats,
            &hot_graph_readiness,
            "hot",
            i)) {
      std::cerr << "single_token_decode_bench: measured iteration failed\n";
      return 1;
    }
  }
  memory_snapshots.push_back(CaptureMemorySnapshot("post_hot"));
  EmitMemorySnapshot(memory_snapshots.back(), options);
  const std::int32_t predicted_token_id = Argmax(last_logits_host);
  const auto runtime_stats = nemotron::GetRuntimeExecutionStatsSnapshot();

  if (!WriteJsonReport(
          options,
          environment_build_ms,
          model_build_ms,
          model->build_report(),
          runtime_stats,
          memory_snapshots,
          warmup_ms,
          hot_ms,
          warmup_step_ms,
          hot_step_ms,
          warmup_step_runtime_stats,
          hot_step_runtime_stats,
          warmup_graph_readiness,
          hot_graph_readiness,
          *metadata->selected_token_id,
          predicted_token_id,
          last_generated_token_ids)) {
    std::cerr << "single_token_decode_bench: failed to write json output\n";
    return 1;
  }

  std::cout << std::fixed << std::setprecision(3)
            << "environment_build_ms=" << environment_build_ms << "\n"
            << "model_build_ms=" << model_build_ms << "\n"
            << "warmup_mean_ms=" << MeanMs(warmup_ms) << "\n"
            << "hot_mean_ms=" << MeanMs(hot_ms) << "\n"
            << "hot_min_ms=" << MinMs(hot_ms) << "\n"
            << "hot_max_ms=" << MaxMs(hot_ms) << "\n"
            << "dense_plan_cache_hits=" << runtime_stats.dense_plan_cache_hits << "\n"
            << "dense_plan_build_failures=" << runtime_stats.dense_plan_build_failures << "\n"
            << "dense_plan_build_failures_attention="
            << runtime_stats.dense_plan_build_failures_attention << "\n"
            << "dense_plan_build_failures_expert="
            << runtime_stats.dense_plan_build_failures_expert << "\n"
            << "dense_plan_build_failures_other="
            << runtime_stats.dense_plan_build_failures_other << "\n"
            << "dense_native_success=" << runtime_stats.dense_native_success << "\n"
            << "dense_native_success_attention="
            << runtime_stats.dense_native_success_attention << "\n"
            << "dense_native_success_expert="
            << runtime_stats.dense_native_success_expert << "\n"
            << "dense_native_success_other="
            << runtime_stats.dense_native_success_other << "\n"
            << "dense_reference_fallbacks=" << runtime_stats.dense_reference_fallbacks << "\n"
            << "dense_reference_fallbacks_attention="
            << runtime_stats.dense_reference_fallbacks_attention << "\n"
            << "dense_reference_fallbacks_expert="
            << runtime_stats.dense_reference_fallbacks_expert << "\n"
            << "dense_reference_fallbacks_other="
            << runtime_stats.dense_reference_fallbacks_other << "\n"
            << "dense_bf16_native_failures=" << runtime_stats.dense_bf16_native_failures << "\n"
            << "dense_fp32_native_failures=" << runtime_stats.dense_fp32_native_failures << "\n"
            << "scaled_fp8_plan_cache_hits=" << runtime_stats.scaled_fp8_plan_cache_hits << "\n"
            << "scaled_fp8_plan_build_failures=" << runtime_stats.scaled_fp8_plan_build_failures << "\n"
            << "scaled_fp8_plan_build_failures_mamba_in_proj="
            << runtime_stats.scaled_fp8_plan_build_failures_mamba_in_proj << "\n"
            << "scaled_fp8_plan_build_failures_mamba_out_proj="
            << runtime_stats.scaled_fp8_plan_build_failures_mamba_out_proj << "\n"
            << "scaled_fp8_plan_build_failures_expert_fc1_latent="
            << runtime_stats.scaled_fp8_plan_build_failures_expert_fc1_latent << "\n"
            << "scaled_fp8_plan_build_failures_expert_shared_up="
            << runtime_stats.scaled_fp8_plan_build_failures_expert_shared_up << "\n"
            << "scaled_fp8_plan_build_failures_expert_shared_down="
            << runtime_stats.scaled_fp8_plan_build_failures_expert_shared_down << "\n"
            << "scaled_fp8_plan_build_failures_other="
            << runtime_stats.scaled_fp8_plan_build_failures_other << "\n"
            << "scaled_fp8_native_success=" << runtime_stats.scaled_fp8_native_success << "\n"
            << "scaled_fp8_native_success_mamba_in_proj="
            << runtime_stats.scaled_fp8_native_success_mamba_in_proj << "\n"
            << "scaled_fp8_native_success_mamba_out_proj="
            << runtime_stats.scaled_fp8_native_success_mamba_out_proj << "\n"
            << "scaled_fp8_native_success_expert_fc1_latent="
            << runtime_stats.scaled_fp8_native_success_expert_fc1_latent << "\n"
            << "scaled_fp8_native_success_expert_shared_up="
            << runtime_stats.scaled_fp8_native_success_expert_shared_up << "\n"
            << "scaled_fp8_native_success_expert_shared_down="
            << runtime_stats.scaled_fp8_native_success_expert_shared_down << "\n"
            << "scaled_fp8_native_success_other="
            << runtime_stats.scaled_fp8_native_success_other << "\n"
            << "scaled_fp8_dequantized_dense_success="
            << runtime_stats.scaled_fp8_dequantized_dense_success << "\n"
            << "scaled_fp8_reference_fallbacks=" << runtime_stats.scaled_fp8_reference_fallbacks << "\n"
            << "scaled_fp8_reference_fallbacks_mamba_in_proj="
            << runtime_stats.scaled_fp8_reference_fallbacks_mamba_in_proj << "\n"
            << "scaled_fp8_reference_fallbacks_mamba_out_proj="
            << runtime_stats.scaled_fp8_reference_fallbacks_mamba_out_proj << "\n"
            << "scaled_fp8_reference_fallbacks_expert_fc1_latent="
            << runtime_stats.scaled_fp8_reference_fallbacks_expert_fc1_latent << "\n"
            << "scaled_fp8_reference_fallbacks_expert_shared_up="
            << runtime_stats.scaled_fp8_reference_fallbacks_expert_shared_up << "\n"
            << "scaled_fp8_reference_fallbacks_expert_shared_down="
            << runtime_stats.scaled_fp8_reference_fallbacks_expert_shared_down << "\n"
            << "scaled_fp8_reference_fallbacks_other="
            << runtime_stats.scaled_fp8_reference_fallbacks_other << "\n"
            << "attention_decode_plan_creates=" << runtime_stats.attention_decode_plan_creates << "\n"
            << "attention_decode_plan_hits=" << runtime_stats.attention_decode_plan_hits << "\n"
            << "attention_decode_device_page_table_copies="
            << runtime_stats.attention_decode_device_page_table_copies << "\n"
            << "expert_selection_metadata_downloads="
            << runtime_stats.expert_selection_metadata_downloads << "\n"
            << "routed_lookup_repair_downloads="
            << runtime_stats.routed_lookup_repair_downloads << "\n"
            << "routed_lookup_repair_experts="
            << runtime_stats.routed_lookup_repair_experts << "\n"
            << "routed_expert_materializations="
            << runtime_stats.routed_expert_materializations << "\n"
            << "flashinfer_routed_expert_uses="
            << runtime_stats.flashinfer_routed_expert_uses << "\n"
            << "flashinfer_routed_expert_fallbacks="
            << runtime_stats.flashinfer_routed_expert_fallbacks << "\n"
            << "grouped_routed_expert_fastpath_uses="
            << runtime_stats.grouped_routed_expert_fastpath_uses << "\n"
            << "moe_graph_captures=" << runtime_stats.moe_graph_captures << "\n"
            << "moe_graph_replays=" << runtime_stats.moe_graph_replays << "\n"
            << "forward_graph_captures=" << runtime_stats.forward_graph_captures << "\n"
            << "forward_graph_replays=" << runtime_stats.forward_graph_replays << "\n"
            << "grouped_routed_expert_fastpath_fallbacks="
            << runtime_stats.grouped_routed_expert_fastpath_fallbacks << "\n"
            << "grouped_routed_expert_prereq_fallbacks="
            << runtime_stats.grouped_routed_expert_prereq_fallbacks << "\n"
            << "grouped_routed_expert_lookup_fallbacks="
            << runtime_stats.grouped_routed_expert_lookup_fallbacks << "\n"
            << "grouped_routed_expert_plan_fallbacks="
            << runtime_stats.grouped_routed_expert_plan_fallbacks << "\n"
            << "grouped_routed_expert_pack_fallbacks="
            << runtime_stats.grouped_routed_expert_pack_fallbacks << "\n"
            << "grouped_routed_expert_matmul_fallbacks="
            << runtime_stats.grouped_routed_expert_matmul_fallbacks << "\n"
            << "grouped_routed_expert_merge_fallbacks="
            << runtime_stats.grouped_routed_expert_merge_fallbacks << "\n"
            << "routed_expert_prefetch_layers="
            << runtime_stats.routed_expert_prefetch_layers << "\n"
            << "routed_expert_prefetch_experts="
            << runtime_stats.routed_expert_prefetch_experts << "\n";
  if (!hot_graph_readiness.empty()) {
    const auto& graph_summary = hot_graph_readiness.back();
    std::cout << "hot_graph_safe_steps=" << graph_summary.graph_safe_steps << "\n"
              << "hot_max_graph_safe_streak=" << graph_summary.max_graph_safe_streak << "\n"
              << "hot_tail_graph_safe_streak=" << graph_summary.tail_graph_safe_streak << "\n"
              << "hot_first_graph_safe_tail_token_index="
              << graph_summary.first_graph_safe_tail_token_index << "\n";
  }
  std::cout
            << "generate_tokens=" << options.generate_tokens << "\n"
            << "selected_token_id=" << *metadata->selected_token_id << "\n"
            << "predicted_token_id=" << predicted_token_id << "\n";
  return 0;
}
