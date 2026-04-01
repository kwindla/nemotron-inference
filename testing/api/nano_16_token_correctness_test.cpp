#include "nemotron/device_tensor.h"
#include "nemotron/linear_op_counters.h"
#include "nemotron/manifest.h"
#include "nemotron/runtime_environment.h"
#include "nemotron/single_token_forward_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char* kNanoModelId = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4";
constexpr std::size_t kPromptTokenCount = 16;
constexpr std::size_t kDecodeTokenCount = 16;

constexpr std::size_t GiB(std::size_t value) {
  return value * 1024ull * 1024ull * 1024ull;
}

const std::vector<std::int32_t>& FixedPromptTokenIds() {
  static const std::vector<std::int32_t> kPrompt = {
      10, 25708, 1010, 4568, 1584, 1261, 25399, 27089,
      1046, 50856, 15776, 10693, 1046, 11, 1010, 10,
  };
  return kPrompt;
}

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool has_cuda_device() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

bool EnvEnabledOrDefault(const char* env_var, bool default_enabled) {
  const char* value = std::getenv(env_var);
  if (value == nullptr) {
    return default_enabled;
  }
  return value[0] != '\0' && std::string(value) != "0";
}

struct LinearFallbackObservation {
  const char* counter_name = "";
  std::uint64_t count = 0;
};

struct LinearCounterSnapshot {
  std::uint64_t dense_fastpath_plan_success = 0;
  std::uint64_t dense_fastpath_plan_fail = 0;
  std::uint64_t dense_fastpath_execute = 0;
  std::uint64_t dense_fastpath_execute_fail = 0;
  std::uint64_t dense_reference_fallback = 0;

  std::uint64_t nvfp4_fastpath_plan_success = 0;
  std::uint64_t nvfp4_fastpath_plan_fail = 0;
  std::uint64_t nvfp4_fastpath_execute = 0;
  std::uint64_t nvfp4_fastpath_execute_fail = 0;
  std::uint64_t nvfp4_reference_fallback = 0;

  std::uint64_t scaled_fp8_fastpath_execute = 0;
  std::uint64_t scaled_fp8_reference_fallback = 0;
};

LinearCounterSnapshot SnapshotLinearOpCounters() {
  const auto& counters = nemotron::GetLinearOpCounters();
  LinearCounterSnapshot snapshot;
  snapshot.dense_fastpath_plan_success =
      counters.dense_fastpath_plan_success.load(std::memory_order_relaxed);
  snapshot.dense_fastpath_plan_fail =
      counters.dense_fastpath_plan_fail.load(std::memory_order_relaxed);
  snapshot.dense_fastpath_execute =
      counters.dense_fastpath_execute.load(std::memory_order_relaxed);
  snapshot.dense_fastpath_execute_fail =
      counters.dense_fastpath_execute_fail.load(std::memory_order_relaxed);
  snapshot.dense_reference_fallback =
      counters.dense_reference_fallback.load(std::memory_order_relaxed);

  snapshot.nvfp4_fastpath_plan_success =
      counters.nvfp4_fastpath_plan_success.load(std::memory_order_relaxed);
  snapshot.nvfp4_fastpath_plan_fail =
      counters.nvfp4_fastpath_plan_fail.load(std::memory_order_relaxed);
  snapshot.nvfp4_fastpath_execute =
      counters.nvfp4_fastpath_execute.load(std::memory_order_relaxed);
  snapshot.nvfp4_fastpath_execute_fail =
      counters.nvfp4_fastpath_execute_fail.load(std::memory_order_relaxed);
  snapshot.nvfp4_reference_fallback =
      counters.nvfp4_reference_fallback.load(std::memory_order_relaxed);

  snapshot.scaled_fp8_fastpath_execute =
      counters.scaled_fp8_fastpath_execute.load(std::memory_order_relaxed);
  snapshot.scaled_fp8_reference_fallback =
      counters.scaled_fp8_reference_fallback.load(std::memory_order_relaxed);
  return snapshot;
}

LinearCounterSnapshot AddLinearCounterSnapshots(
    const LinearCounterSnapshot& lhs,
    const LinearCounterSnapshot& rhs) {
  LinearCounterSnapshot sum;
  sum.dense_fastpath_plan_success =
      lhs.dense_fastpath_plan_success + rhs.dense_fastpath_plan_success;
  sum.dense_fastpath_plan_fail =
      lhs.dense_fastpath_plan_fail + rhs.dense_fastpath_plan_fail;
  sum.dense_fastpath_execute = lhs.dense_fastpath_execute + rhs.dense_fastpath_execute;
  sum.dense_fastpath_execute_fail =
      lhs.dense_fastpath_execute_fail + rhs.dense_fastpath_execute_fail;
  sum.dense_reference_fallback =
      lhs.dense_reference_fallback + rhs.dense_reference_fallback;

  sum.nvfp4_fastpath_plan_success =
      lhs.nvfp4_fastpath_plan_success + rhs.nvfp4_fastpath_plan_success;
  sum.nvfp4_fastpath_plan_fail =
      lhs.nvfp4_fastpath_plan_fail + rhs.nvfp4_fastpath_plan_fail;
  sum.nvfp4_fastpath_execute = lhs.nvfp4_fastpath_execute + rhs.nvfp4_fastpath_execute;
  sum.nvfp4_fastpath_execute_fail =
      lhs.nvfp4_fastpath_execute_fail + rhs.nvfp4_fastpath_execute_fail;
  sum.nvfp4_reference_fallback =
      lhs.nvfp4_reference_fallback + rhs.nvfp4_reference_fallback;

  sum.scaled_fp8_fastpath_execute =
      lhs.scaled_fp8_fastpath_execute + rhs.scaled_fp8_fastpath_execute;
  sum.scaled_fp8_reference_fallback =
      lhs.scaled_fp8_reference_fallback + rhs.scaled_fp8_reference_fallback;
  return sum;
}

void RestoreLinearOpCounters(const LinearCounterSnapshot& snapshot) {
  auto& counters = nemotron::GetLinearOpCounters();
  counters.dense_fastpath_plan_success.store(
      snapshot.dense_fastpath_plan_success,
      std::memory_order_relaxed);
  counters.dense_fastpath_plan_fail.store(
      snapshot.dense_fastpath_plan_fail,
      std::memory_order_relaxed);
  counters.dense_fastpath_execute.store(
      snapshot.dense_fastpath_execute,
      std::memory_order_relaxed);
  counters.dense_fastpath_execute_fail.store(
      snapshot.dense_fastpath_execute_fail,
      std::memory_order_relaxed);
  counters.dense_reference_fallback.store(
      snapshot.dense_reference_fallback,
      std::memory_order_relaxed);

  counters.nvfp4_fastpath_plan_success.store(
      snapshot.nvfp4_fastpath_plan_success,
      std::memory_order_relaxed);
  counters.nvfp4_fastpath_plan_fail.store(
      snapshot.nvfp4_fastpath_plan_fail,
      std::memory_order_relaxed);
  counters.nvfp4_fastpath_execute.store(
      snapshot.nvfp4_fastpath_execute,
      std::memory_order_relaxed);
  counters.nvfp4_fastpath_execute_fail.store(
      snapshot.nvfp4_fastpath_execute_fail,
      std::memory_order_relaxed);
  counters.nvfp4_reference_fallback.store(
      snapshot.nvfp4_reference_fallback,
      std::memory_order_relaxed);

  counters.scaled_fp8_fastpath_execute.store(
      snapshot.scaled_fp8_fastpath_execute,
      std::memory_order_relaxed);
  counters.scaled_fp8_reference_fallback.store(
      snapshot.scaled_fp8_reference_fallback,
      std::memory_order_relaxed);
}

void PrintLayerProbeLinearCounters(
    std::size_t layer_index,
    const LinearCounterSnapshot& snapshot) {
  std::cout << "nano_16_token_correctness_test: layer_probe layer_index="
            << layer_index
            << " linear_counters dense_ref=" << snapshot.dense_reference_fallback
            << " nvfp4_ref=" << snapshot.nvfp4_reference_fallback
            << " fp8_ref=" << snapshot.scaled_fp8_reference_fallback << "\n";
  std::cout.flush();
}

std::vector<LinearFallbackObservation> CollectLinearReferenceFallbacks() {
  const auto& counters = nemotron::GetLinearOpCounters();
  std::vector<LinearFallbackObservation> fallbacks;
  fallbacks.reserve(3);

  const auto append_if_nonzero = [&fallbacks](const char* counter_name, const auto& counter) {
    const std::uint64_t count = counter.load(std::memory_order_relaxed);
    if (count != 0) {
      fallbacks.push_back({counter_name, count});
    }
  };

  append_if_nonzero("dense_reference_fallback", counters.dense_reference_fallback);
  append_if_nonzero("nvfp4_reference_fallback", counters.nvfp4_reference_fallback);
  append_if_nonzero(
      "scaled_fp8_reference_fallback",
      counters.scaled_fp8_reference_fallback);
  return fallbacks;
}

void PrintUnexpectedLinearFallbackWarning(
    std::ostream& stream,
    const std::vector<LinearFallbackObservation>& fallbacks) {
  if (fallbacks.empty()) {
    return;
  }

  stream << "WARNING: nano_16_token_correctness_test: unexpected linear reference fallbacks "
         << "while NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH=1\n";
  for (const LinearFallbackObservation& fallback : fallbacks) {
    stream << "WARNING: nano_16_token_correctness_test: operator="
           << fallback.counter_name
           << " reference_fallback_count=" << fallback.count << "\n";
  }
}

class ScopedLinearCounterReport {
 public:
  ScopedLinearCounterReport(
      bool strict_linear_enabled,
      bool linear_device_fastpath_enabled)
      : strict_linear_enabled_(strict_linear_enabled),
        linear_device_fastpath_enabled_(linear_device_fastpath_enabled) {
    nemotron::ResetLinearOpCounters();
  }

  ~ScopedLinearCounterReport() {
    nemotron::PrintLinearOpCounterSummary(std::cout);
    std::cout.flush();

    if (!strict_linear_enabled_ || !linear_device_fastpath_enabled_) {
      return;
    }

    const std::vector<LinearFallbackObservation> fallbacks =
        CollectLinearReferenceFallbacks();
    if (fallbacks.empty()) {
      return;
    }

    PrintUnexpectedLinearFallbackWarning(std::cerr, fallbacks);
    std::cerr.flush();
  }

  ScopedLinearCounterReport(const ScopedLinearCounterReport&) = delete;
  ScopedLinearCounterReport& operator=(const ScopedLinearCounterReport&) = delete;

 private:
  bool strict_linear_enabled_ = false;
  bool linear_device_fastpath_enabled_ = false;
};

class ScopedEnvOverride {
 public:
  ScopedEnvOverride(const char* name, const char* value) : name_(name) {
    const char* existing = std::getenv(name_.c_str());
    if (existing != nullptr) {
      had_original_ = true;
      original_value_ = existing;
    }
    if (value == nullptr) {
      unsetenv(name_.c_str());
    } else {
      setenv(name_.c_str(), value, 1);
    }
  }

  ~ScopedEnvOverride() {
    if (had_original_) {
      setenv(name_.c_str(), original_value_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  ScopedEnvOverride(const ScopedEnvOverride&) = delete;
  ScopedEnvOverride& operator=(const ScopedEnvOverride&) = delete;

 private:
  std::string name_;
  std::string original_value_;
  bool had_original_ = false;
};

struct PrefillRouteConfig {
  const char* route_id = "";
  const char* description = "";
  bool fused_enabled = false;
  bool decode_consistent_prefill_enabled = false;
};

class ScopedRouteOverrides {
 public:
  explicit ScopedRouteOverrides(const PrefillRouteConfig& route)
      : fused_mamba_(
            "NEMOTRON_FORWARD_FUSED_MAMBA_DECODE",
            route.fused_enabled ? "1" : "0"),
        fused_moe_(
            "NEMOTRON_FORWARD_FUSED_MOE_DECODE",
            route.fused_enabled ? "1" : "0"),
        decode_consistent_prefill_(
            "NEMOTRON_FORWARD_DECODE_CONSISTENT_PREFILL",
            route.decode_consistent_prefill_enabled ? "1" : "0") {}

  ScopedRouteOverrides(const ScopedRouteOverrides&) = delete;
  ScopedRouteOverrides& operator=(const ScopedRouteOverrides&) = delete;

 private:
  ScopedEnvOverride fused_mamba_;
  ScopedEnvOverride fused_moe_;
  ScopedEnvOverride decode_consistent_prefill_;
};

constexpr PrefillRouteConfig kRouteA = {
    "A",
    "reference kernels + legacy multi-token prefill",
    false,
    false,
};
constexpr PrefillRouteConfig kRouteB = {
    "B",
    "reference kernels + decode-consistent prefill",
    false,
    true,
};
constexpr PrefillRouteConfig kRouteC = {
    "C",
    "fused kernels + decode-consistent prefill",
    true,
    true,
};

std::string PairId(
    const PrefillRouteConfig& lhs_route,
    const PrefillRouteConfig& rhs_route) {
  return std::string(lhs_route.route_id) + "<->" + rhs_route.route_id;
}

std::string RouteLabel(const PrefillRouteConfig& route) {
  return std::string("Route ") + route.route_id;
}

std::string RouteDetail(const PrefillRouteConfig& route) {
  return RouteLabel(route) + " (" + route.description + ")";
}

nemotron::RuntimeBootstrapOptions make_options() {
  nemotron::RuntimeBootstrapOptions options;
  options.service_target.total_memory_bytes = GiB(128);
  options.service_target.weights_bytes = GiB(100);
  options.service_target.workspace_bytes = GiB(8);
  options.service_target.graph_bytes = GiB(4);
  options.service_target.safety_headroom_bytes = GiB(4);
  options.service_target.target_active_requests = 1;
  options.service_target.target_context_tokens = kPromptTokenCount + kDecodeTokenCount;
  options.use_fp16_mamba_state = false;
  options.reusable_node_metadata_bytes = 4096;
  options.verify_manifest_files = false;
  options.materialize_weight_arena = false;
  return options;
}

std::optional<nemotron::SingleTokenForwardConfig> config_for_manifest(
    const nemotron::PackedModelManifest& manifest) {
  if (manifest.runtime.model_id == kNanoModelId) {
    return nemotron::KnownNemotron3Nano30BA3BConfig();
  }
  return std::nullopt;
}

std::optional<std::size_t> ParseEnvSizeT(const char* env_var) {
  const char* value = std::getenv(env_var);
  if (value == nullptr || value[0] == '\0') {
    return std::nullopt;
  }
  try {
    return static_cast<std::size_t>(std::stoull(value));
  } catch (...) {
    return std::nullopt;
  }
}

std::vector<float> CopyTensorToHost(const nemotron::DeviceTensorFp32& tensor) {
  std::vector<float> host(tensor.numel(), 0.0f);
  if (!tensor.CopyToHost(host.data(), host.size())) {
    return {};
  }
  return host;
}

std::vector<float> SliceRow(
    const std::vector<float>& values,
    std::size_t row_index,
    std::size_t row_width) {
  if (row_width == 0) {
    return {};
  }
  const std::size_t row_offset = row_index * row_width;
  if (row_offset + row_width > values.size()) {
    return {};
  }
  return std::vector<float>(
      values.begin() + static_cast<std::ptrdiff_t>(row_offset),
      values.begin() + static_cast<std::ptrdiff_t>(row_offset + row_width));
}

std::optional<std::int32_t> ArgMaxTokenId(const std::vector<float>& logits_row) {
  if (logits_row.empty()) {
    return std::nullopt;
  }
  const auto max_it = std::max_element(logits_row.begin(), logits_row.end());
  if (max_it == logits_row.end()) {
    return std::nullopt;
  }
  return static_cast<std::int32_t>(std::distance(logits_row.begin(), max_it));
}

float MaxAbsDiff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return std::numeric_limits<float>::infinity();
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
  }
  return max_diff;
}

std::string TopKSummary(const std::vector<float>& logits_row, std::size_t k) {
  if (logits_row.empty() || k == 0) {
    return "[]";
  }
  k = std::min(k, logits_row.size());
  std::vector<std::size_t> indices(logits_row.size(), 0);
  for (std::size_t i = 0; i < indices.size(); ++i) {
    indices[i] = i;
  }
  std::partial_sort(
      indices.begin(),
      indices.begin() + static_cast<std::ptrdiff_t>(k),
      indices.end(),
      [&logits_row](std::size_t lhs, std::size_t rhs) {
        if (logits_row[lhs] == logits_row[rhs]) {
          return lhs < rhs;
        }
        return logits_row[lhs] > logits_row[rhs];
      });
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < k; ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << indices[i] << ":" << logits_row[indices[i]];
  }
  out << "]";
  return out.str();
}

struct BoundaryObservation {
  std::int32_t selected_token_id = -1;
  std::vector<float> logits_row;
  std::vector<float> prompt_boundary_embedding_row;
  std::vector<float> prompt_boundary_final_hidden_normed_row;
  double elapsed_ms = 0.0;
  std::size_t sequence_length = 0;
  std::size_t decode_position = 0;
};

struct LayerObservation {
  std::size_t layer_index = 0;
  std::vector<float> hidden_row;
  double elapsed_ms = 0.0;
};

struct BoundaryComparison {
  const PrefillRouteConfig* lhs_route = nullptr;
  const PrefillRouteConfig* rhs_route = nullptr;
  std::int32_t lhs_token_id = -1;
  std::int32_t rhs_token_id = -1;
  float max_abs_diff = 0.0f;
};

struct LayerProbe {
  std::size_t layer_index = 0;
  bool coarse = false;
};

std::optional<std::vector<float>> ExtractTraceRow(
    const std::vector<float>& values,
    std::size_t expected_row_count,
    std::size_t row_index,
    std::size_t row_width) {
  if (row_width == 0 ||
      row_index >= expected_row_count ||
      values.size() != expected_row_count * row_width) {
    return std::nullopt;
  }
  const std::vector<float> row = SliceRow(values, row_index, row_width);
  if (row.size() != row_width) {
    return std::nullopt;
  }
  return row;
}

BoundaryComparison MakeBoundaryComparison(
    const PrefillRouteConfig& lhs_route,
    const BoundaryObservation& lhs_observation,
    const PrefillRouteConfig& rhs_route,
    const BoundaryObservation& rhs_observation) {
  BoundaryComparison comparison;
  comparison.lhs_route = &lhs_route;
  comparison.rhs_route = &rhs_route;
  comparison.lhs_token_id = lhs_observation.selected_token_id;
  comparison.rhs_token_id = rhs_observation.selected_token_id;
  comparison.max_abs_diff =
      MaxAbsDiff(lhs_observation.logits_row, rhs_observation.logits_row);
  return comparison;
}

bool HasSelectedTokenMismatch(const BoundaryComparison& comparison) {
  return comparison.lhs_token_id != comparison.rhs_token_id;
}

void PrintPrefillSummaryTable(
    const std::vector<BoundaryComparison>& comparisons) {
  const std::ios::fmtflags previous_flags = std::cout.flags();
  const std::streamsize previous_precision = std::cout.precision();

  std::cout << "nano_16_token_correctness_test: prefill route legend "
            << "A=(" << kRouteA.description << "), "
            << "B=(" << kRouteB.description << "), "
            << "C=(" << kRouteC.description << ")\n";
  std::cout << "nano_16_token_correctness_test: prefill comparison matrix\n";
  std::cout << std::left << std::setw(8) << "pair"
            << std::setw(12) << "lhs_token"
            << std::setw(12) << "rhs_token"
            << std::setw(16) << "max_abs_diff"
            << "token_match\n";
  std::cout << std::fixed << std::setprecision(6);
  for (const BoundaryComparison& comparison : comparisons) {
    std::cout << std::left << std::setw(8)
              << PairId(*comparison.lhs_route, *comparison.rhs_route)
              << std::setw(12) << comparison.lhs_token_id
              << std::setw(12) << comparison.rhs_token_id
              << std::setw(16) << comparison.max_abs_diff
              << (HasSelectedTokenMismatch(comparison) ? "no" : "yes")
              << "\n";
  }
  std::cout.flush();

  std::cout.flags(previous_flags);
  std::cout.precision(previous_precision);
}

std::optional<BoundaryObservation> RunPrefillBoundary(
    nemotron::SingleTokenForwardModel& model,
    const PrefillRouteConfig& route,
    nemotron::RequestExecutionContext& request_context,
    bool capture_trace_rows = false) {
  ScopedRouteOverrides route_overrides(route);
  (void)route_overrides;
  const auto& prompt_token_ids = FixedPromptTokenIds();

  auto logits =
      nemotron::DeviceTensorFp32::Create({kPromptTokenCount, model.config().vocab_size});
  if (logits == nullptr || !logits->valid()) {
    return std::nullopt;
  }
  nemotron::SingleTokenForwardTrace trace;
  nemotron::SingleTokenForwardTrace* trace_ptr = capture_trace_rows ? &trace : nullptr;

  std::cout << "nano_16_token_correctness_test: " << RouteDetail(route)
            << " prefill start prompt_tokens=" << kPromptTokenCount << "\n";
  std::cout.flush();
  const auto start = std::chrono::steady_clock::now();
  if (!model.RunPrefill(
          prompt_token_ids.data(),
          prompt_token_ids.size(),
          request_context,
          logits.get(),
          /*capture_layer_indices=*/{},
          trace_ptr)) {
    return std::nullopt;
  }
  const std::vector<float> host_logits = CopyTensorToHost(*logits);
  const auto end = std::chrono::steady_clock::now();
  const std::vector<float> final_row =
      SliceRow(host_logits, prompt_token_ids.size() - 1, model.config().vocab_size);
  const auto selected_token_id = ArgMaxTokenId(final_row);
  if (!selected_token_id.has_value()) {
    return std::nullopt;
  }

  BoundaryObservation observation;
  observation.selected_token_id = *selected_token_id;
  observation.logits_row = final_row;
  observation.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
  observation.sequence_length = request_context.sequence_length();
  observation.decode_position = request_context.decode_position();
  if (capture_trace_rows) {
    const std::size_t prompt_boundary_index = prompt_token_ids.size() - 1;
    const std::size_t hidden_size = model.config().hidden_size;
    auto embedding_row = ExtractTraceRow(
        trace.embedding_output,
        prompt_token_ids.size(),
        prompt_boundary_index,
        hidden_size);
    if (!embedding_row.has_value()) {
      std::cerr << "nano_16_token_correctness_test: failed to capture prompt-boundary embedding row "
                << "route=" << route.route_id
                << " trace_values=" << trace.embedding_output.size()
                << " hidden_size=" << hidden_size << "\n";
      return std::nullopt;
    }
    auto final_hidden_normed_row = ExtractTraceRow(
        trace.final_hidden_normed,
        prompt_token_ids.size(),
        prompt_boundary_index,
        hidden_size);
    if (!final_hidden_normed_row.has_value()) {
      std::cerr
          << "nano_16_token_correctness_test: failed to capture prompt-boundary final-norm row "
          << "route=" << route.route_id
          << " trace_values=" << trace.final_hidden_normed.size()
          << " hidden_size=" << hidden_size << "\n";
      return std::nullopt;
    }
    observation.prompt_boundary_embedding_row = std::move(*embedding_row);
    observation.prompt_boundary_final_hidden_normed_row =
        std::move(*final_hidden_normed_row);
  }

  std::cout << "nano_16_token_correctness_test: " << RouteDetail(route)
            << " prefill complete elapsed_ms=" << observation.elapsed_ms
            << " first_token=" << observation.selected_token_id << "\n";
  std::cout.flush();
  return observation;
}

void PrintTraceComparisonSummary(
    const PrefillRouteConfig& lhs_route,
    const BoundaryObservation& lhs_observation,
    const PrefillRouteConfig& rhs_route,
    const BoundaryObservation& rhs_observation) {
  const std::ios::fmtflags previous_flags = std::cout.flags();
  const std::streamsize previous_precision = std::cout.precision();

  // Compare the prompt-boundary row only; that's the row that drives the boundary logits.
  const float embedding_diff = MaxAbsDiff(
      lhs_observation.prompt_boundary_embedding_row,
      rhs_observation.prompt_boundary_embedding_row);
  const float final_norm_diff = MaxAbsDiff(
      lhs_observation.prompt_boundary_final_hidden_normed_row,
      rhs_observation.prompt_boundary_final_hidden_normed_row);
  std::cout << std::fixed << std::setprecision(6)
            << "nano_16_token_correctness_test: embedding "
            << PairId(lhs_route, rhs_route) << " max_abs_diff=" << embedding_diff
            << "  final_norm " << PairId(lhs_route, rhs_route)
            << " max_abs_diff=" << final_norm_diff << "\n";
  std::cout.flush();

  std::cout.flags(previous_flags);
  std::cout.precision(previous_precision);
}

std::optional<BoundaryObservation> RunContinuationBoundary(
    nemotron::SingleTokenForwardModel& model,
    const PrefillRouteConfig& route,
    nemotron::RequestExecutionContext& request_context,
    std::int32_t token_id,
    std::size_t token_index) {
  ScopedRouteOverrides route_overrides(route);
  (void)route_overrides;

  auto logits = nemotron::DeviceTensorFp32::Create({1, model.config().vocab_size});
  if (logits == nullptr || !logits->valid()) {
    return std::nullopt;
  }

  std::cout << "nano_16_token_correctness_test: " << RouteDetail(route)
            << " decode step " << token_index << "/" << (kDecodeTokenCount - 1)
            << " start consume_token=" << token_id << "\n";
  std::cout.flush();
  const auto start = std::chrono::steady_clock::now();
  if (!model.ContinueSingleToken(token_id, request_context, logits.get())) {
    return std::nullopt;
  }
  const std::vector<float> host_logits = CopyTensorToHost(*logits);
  const auto end = std::chrono::steady_clock::now();
  const auto selected_token_id = ArgMaxTokenId(host_logits);
  if (!selected_token_id.has_value()) {
    return std::nullopt;
  }

  BoundaryObservation observation;
  observation.selected_token_id = *selected_token_id;
  observation.logits_row = host_logits;
  observation.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
  observation.sequence_length = request_context.sequence_length();
  observation.decode_position = request_context.decode_position();

  std::cout << "nano_16_token_correctness_test: " << RouteDetail(route)
            << " decode step " << token_index << "/" << (kDecodeTokenCount - 1)
            << " complete elapsed_ms=" << observation.elapsed_ms
            << " next_token=" << observation.selected_token_id << "\n";
  std::cout.flush();
  return observation;
}

void PrintBoundaryMismatch(
    const char* phase_label,
    std::size_t token_index,
    const PrefillRouteConfig& lhs_route,
    const BoundaryObservation& lhs_observation,
    const PrefillRouteConfig& rhs_route,
    const BoundaryObservation& rhs_observation) {
  std::cerr << "nano_16_token_correctness_test: divergent " << phase_label
            << " boundary pair=" << PairId(lhs_route, rhs_route)
            << " index=" << token_index
            << " " << lhs_route.route_id << "_token=" << lhs_observation.selected_token_id
            << " " << rhs_route.route_id << "_token=" << rhs_observation.selected_token_id
            << " max_abs_diff="
            << MaxAbsDiff(lhs_observation.logits_row, rhs_observation.logits_row)
            << "\n";
  std::cerr << lhs_route.route_id << "_top5="
            << TopKSummary(lhs_observation.logits_row, 5) << "\n";
  std::cerr << rhs_route.route_id << "_top5="
            << TopKSummary(rhs_observation.logits_row, 5) << "\n";
  std::cerr << lhs_route.route_id << "_state=(sequence_length="
            << lhs_observation.sequence_length
            << ", decode_position=" << lhs_observation.decode_position << ")\n";
  std::cerr << rhs_route.route_id << "_state=(sequence_length="
            << rhs_observation.sequence_length
            << ", decode_position=" << rhs_observation.decode_position << ")\n";
}

std::vector<LayerProbe> BuildLayerProbes(
    nemotron::SingleTokenForwardModel& model,
    std::size_t probe_stride) {
  const auto& layers = model.plan().layers;
  std::vector<LayerProbe> probes;
  probes.reserve(layers.size());
  std::vector<bool> visited(layers.size(), false);

  auto add_probe = [&](std::size_t layer_position, bool coarse) {
    if (layer_position >= layers.size() || visited[layer_position]) {
      return;
    }
    visited[layer_position] = true;
    probes.push_back({layers[layer_position].layer_index, coarse});
  };

  for (std::size_t layer_position = 0; layer_position < layers.size();
       layer_position += probe_stride) {
    add_probe(layer_position, true);
  }
  if (!layers.empty()) {
    add_probe(layers.size() - 1, true);
  }
  for (std::size_t layer_position = 0; layer_position < layers.size(); ++layer_position) {
    add_probe(layer_position, false);
  }
  return probes;
}

std::optional<LayerObservation> RunLayerObservation(
    nemotron::SingleTokenForwardModel& model,
    const PrefillRouteConfig& route,
    std::size_t layer_index) {
  ScopedRouteOverrides route_overrides(route);
  (void)route_overrides;

  auto request_context = model.CreateRequestContext();
  if (request_context == nullptr || !request_context->valid()) {
    return std::nullopt;
  }
  auto logits = nemotron::DeviceTensorFp32::Create({kPromptTokenCount, model.config().vocab_size});
  if (logits == nullptr || !logits->valid()) {
    return std::nullopt;
  }

  nemotron::SingleTokenForwardTrace trace;
  std::cout << "nano_16_token_correctness_test: " << RouteDetail(route)
            << " layer compare start layer_index=" << layer_index << "\n";
  std::cout.flush();
  const auto start = std::chrono::steady_clock::now();
  if (!model.RunPrefill(
          FixedPromptTokenIds().data(),
          FixedPromptTokenIds().size(),
          *request_context,
          logits.get(),
          {layer_index},
          &trace,
          layer_index)) {
    return std::nullopt;
  }
  const auto end = std::chrono::steady_clock::now();

  for (auto it = trace.captured_layers.rbegin(); it != trace.captured_layers.rend(); ++it) {
    if (it->layer_index != layer_index) {
      continue;
    }
    LayerObservation observation;
    observation.layer_index = layer_index;
    observation.hidden_row = it->hidden;
    observation.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "nano_16_token_correctness_test: " << RouteDetail(route)
              << " layer compare complete layer_index=" << layer_index
              << " elapsed_ms=" << observation.elapsed_ms
              << " hidden_values=" << observation.hidden_row.size()
              << "\n";
    std::cout.flush();
    return observation;
  }
  return std::nullopt;
}

void ComparePromptLayer(
    nemotron::SingleTokenForwardModel& model,
    const PrefillRouteConfig& lhs_route,
    const PrefillRouteConfig& rhs_route,
    std::size_t layer_index) {
  const auto lhs_observation = RunLayerObservation(model, lhs_route, layer_index);
  const auto rhs_observation = RunLayerObservation(model, rhs_route, layer_index);
  if (!lhs_observation.has_value() || !rhs_observation.has_value()) {
    std::cerr << "nano_16_token_correctness_test: layer compare failed pair="
              << PairId(lhs_route, rhs_route)
              << " layer_index=" << layer_index << "\n";
    return;
  }
  std::cerr << "nano_16_token_correctness_test: layer compare pair="
            << PairId(lhs_route, rhs_route)
            << " layer_index=" << layer_index
            << " max_abs_diff="
            << MaxAbsDiff(lhs_observation->hidden_row, rhs_observation->hidden_row)
            << "\n";
}

void MaybeTraceDivergentPromptLayers(
    nemotron::SingleTokenForwardModel& model,
    const PrefillRouteConfig& lhs_route,
    const PrefillRouteConfig& rhs_route) {
  const bool strict_linear_enabled =
      EnvEnabledOrDefault("NEMOTRON_NANO_16_STRICT_LINEAR", false);
  const auto explicit_layer = ParseEnvSizeT("NEMOTRON_NANO_16_COMPARE_LAYER");
  if (explicit_layer.has_value()) {
    ComparePromptLayer(model, lhs_route, rhs_route, *explicit_layer);
    return;
  }
  const char* trace_divergence = std::getenv("NEMOTRON_NANO_16_TRACE_DIVERGENCE");
  if (trace_divergence == nullptr || std::string(trace_divergence) == "0") {
    return;
  }

  const auto& layers = model.plan().layers;
  if (layers.empty()) {
    return;
  }
  constexpr float kDivergenceTol = 1.0e-3f;
  const std::size_t probe_stride =
      std::max<std::size_t>(1, ParseEnvSizeT("NEMOTRON_NANO_16_PROBE_STRIDE").value_or(4));
  const std::vector<LayerProbe> probes = BuildLayerProbes(model, probe_stride);
  std::vector<std::size_t> divergent_layers;
  divergent_layers.reserve(layers.size());
  for (const LayerProbe& probe : probes) {
    const auto run_observation = [&](const PrefillRouteConfig& route) {
      if (!strict_linear_enabled) {
        return RunLayerObservation(model, route, probe.layer_index);
      }

      const LinearCounterSnapshot cumulative_before = SnapshotLinearOpCounters();
      nemotron::ResetLinearOpCounters();
      std::optional<LayerObservation> observation =
          RunLayerObservation(model, route, probe.layer_index);
      const LinearCounterSnapshot layer_delta = SnapshotLinearOpCounters();
      PrintLayerProbeLinearCounters(probe.layer_index, layer_delta);
      RestoreLinearOpCounters(AddLinearCounterSnapshots(cumulative_before, layer_delta));
      return observation;
    };

    const auto lhs_observation = run_observation(lhs_route);
    const auto rhs_observation = run_observation(rhs_route);
    if (!lhs_observation.has_value() || !rhs_observation.has_value()) {
      std::cerr << "nano_16_token_correctness_test: layer probe failed pair="
                << PairId(lhs_route, rhs_route)
                << " layer_index=" << probe.layer_index << "\n";
      return;
    }
    const float diff =
        MaxAbsDiff(lhs_observation->hidden_row, rhs_observation->hidden_row);
    std::cerr << "nano_16_token_correctness_test: layer probe pair="
              << PairId(lhs_route, rhs_route)
              << " phase=" << (probe.coarse ? "coarse" : "refine")
              << " layer_index=" << probe.layer_index
              << " max_abs_diff=" << diff << "\n";
    if (diff > kDivergenceTol) {
      divergent_layers.push_back(probe.layer_index);
    }
  }
  if (divergent_layers.empty()) {
    std::cerr << "nano_16_token_correctness_test: no divergent prompt-prefill layers pair="
              << PairId(lhs_route, rhs_route)
              << " probe_stride=" << probe_stride << "\n";
    return;
  }
  std::sort(divergent_layers.begin(), divergent_layers.end());
  divergent_layers.erase(
      std::unique(divergent_layers.begin(), divergent_layers.end()),
      divergent_layers.end());
  std::cerr << "nano_16_token_correctness_test: divergent prompt-prefill layers pair="
            << PairId(lhs_route, rhs_route)
            << " probe_stride=" << probe_stride
            << " layers=[";
  for (std::size_t i = 0; i < divergent_layers.size(); ++i) {
    if (i != 0) {
      std::cerr << ",";
    }
    std::cerr << divergent_layers[i];
  }
  std::cerr << "]\n";
}

bool run_nano_correctness_gate() {
  const char* manifest_env = std::getenv("NEMOTRON_FORWARD_MANIFEST");
  if (manifest_env == nullptr || std::string(manifest_env).empty()) {
    std::cout << "nano_16_token_correctness_test: SKIP (NEMOTRON_FORWARD_MANIFEST is unset)\n";
    return true;
  }
  if (!has_cuda_device()) {
    std::cout << "nano_16_token_correctness_test: SKIP (no CUDA device available)\n";
    return true;
  }

  const std::filesystem::path manifest_path(manifest_env);
  if (!std::filesystem::exists(manifest_path)) {
    std::cout << "nano_16_token_correctness_test: SKIP (manifest path does not exist)\n";
    return true;
  }

  const auto load_result = nemotron::LoadManifestFromJsonFile(manifest_path);
  if (!expect(load_result.ok, "manifest JSON should parse")) {
    return false;
  }
  if (load_result.manifest.runtime.model_id != kNanoModelId) {
    std::cout << "nano_16_token_correctness_test: SKIP (manifest model is not Nano)\n";
    return true;
  }

  auto environment =
      nemotron::RuntimeEnvironment::BuildFromManifestFile(manifest_path, make_options());
  if (!expect(static_cast<bool>(environment), "runtime environment should build")) {
    return false;
  }

  const auto config_opt = config_for_manifest(load_result.manifest);
  if (!expect(config_opt.has_value(), "Nano runtime config should resolve")) {
    return false;
  }
  nemotron::SingleTokenForwardConfig config = *config_opt;
  config.max_tokens = std::max<std::size_t>(config.max_tokens, kPromptTokenCount + kDecodeTokenCount);

  auto model = nemotron::SingleTokenForwardModel::Create(*environment, config);
  if (!expect(model != nullptr && model->valid(), "forward model should build")) {
    return false;
  }

  const bool strict_linear_enabled =
      EnvEnabledOrDefault("NEMOTRON_NANO_16_STRICT_LINEAR", false);
  const bool linear_device_fastpath_enabled =
      EnvEnabledOrDefault("NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH", false);
  const bool capture_embedding_trace =
      EnvEnabledOrDefault("NEMOTRON_NANO_16_TRACE_EMBEDDING", false);
  ScopedLinearCounterReport linear_counter_report(
      strict_linear_enabled,
      linear_device_fastpath_enabled);

  auto route_a_context = model->CreateRequestContext();
  auto route_b_context = model->CreateRequestContext();
  auto route_c_context = model->CreateRequestContext();
  if (!expect(
          route_a_context != nullptr && route_a_context->valid(),
          "Route A request context should create") ||
      !expect(
          route_b_context != nullptr && route_b_context->valid(),
          "Route B request context should create") ||
      !expect(
          route_c_context != nullptr && route_c_context->valid(),
          "Route C request context should create")) {
    return false;
  }

  const auto route_a_prefill =
      RunPrefillBoundary(*model, kRouteA, *route_a_context, capture_embedding_trace);
  const auto route_b_prefill =
      RunPrefillBoundary(*model, kRouteB, *route_b_context, capture_embedding_trace);
  const auto route_c_prefill =
      RunPrefillBoundary(*model, kRouteC, *route_c_context, capture_embedding_trace);
  if (!expect(route_a_prefill.has_value(), "Route A prefill boundary should succeed") ||
      !expect(route_b_prefill.has_value(), "Route B prefill boundary should succeed") ||
      !expect(route_c_prefill.has_value(), "Route C prefill boundary should succeed")) {
    return false;
  }

  const BoundaryComparison route_ab_comparison =
      MakeBoundaryComparison(kRouteA, *route_a_prefill, kRouteB, *route_b_prefill);
  const BoundaryComparison route_bc_comparison =
      MakeBoundaryComparison(kRouteB, *route_b_prefill, kRouteC, *route_c_prefill);
  const BoundaryComparison route_ac_comparison =
      MakeBoundaryComparison(kRouteA, *route_a_prefill, kRouteC, *route_c_prefill);
  PrintPrefillSummaryTable({route_ab_comparison, route_bc_comparison, route_ac_comparison});
  if (capture_embedding_trace) {
    PrintTraceComparisonSummary(kRouteA, *route_a_prefill, kRouteB, *route_b_prefill);
    PrintTraceComparisonSummary(kRouteB, *route_b_prefill, kRouteC, *route_c_prefill);
  }

  bool has_prefill_mismatch = false;
  if (HasSelectedTokenMismatch(route_ab_comparison)) {
    PrintBoundaryMismatch("prefill", 0, kRouteA, *route_a_prefill, kRouteB, *route_b_prefill);
    MaybeTraceDivergentPromptLayers(*model, kRouteA, kRouteB);
    has_prefill_mismatch = true;
  }
  if (HasSelectedTokenMismatch(route_bc_comparison)) {
    PrintBoundaryMismatch("prefill", 0, kRouteB, *route_b_prefill, kRouteC, *route_c_prefill);
    MaybeTraceDivergentPromptLayers(*model, kRouteB, kRouteC);
    has_prefill_mismatch = true;
  }
  if (has_prefill_mismatch) {
    return false;
  }

  std::int32_t route_a_token = route_a_prefill->selected_token_id;
  std::int32_t route_c_token = route_c_prefill->selected_token_id;
  std::vector<std::int32_t> generated_token_ids = {route_a_token};

  for (std::size_t token_index = 1; token_index < kDecodeTokenCount; ++token_index) {
    const auto route_a_step =
        RunContinuationBoundary(*model, kRouteA, *route_a_context, route_a_token, token_index);
    const auto route_c_step =
        RunContinuationBoundary(*model, kRouteC, *route_c_context, route_c_token, token_index);
    if (!expect(route_a_step.has_value(), "Route A continuation boundary should succeed") ||
        !expect(route_c_step.has_value(), "Route C continuation boundary should succeed")) {
      return false;
    }
    if (route_a_step->selected_token_id != route_c_step->selected_token_id) {
      PrintBoundaryMismatch("decode", token_index, kRouteA, *route_a_step, kRouteC, *route_c_step);
      std::cerr << "route_a_top1=[" << TopKSummary(route_a_step->logits_row, 1) << "]\n";
      std::cerr << "generated_tokens_before_divergence=[";
      for (std::size_t i = 0; i < generated_token_ids.size(); ++i) {
        if (i != 0) {
          std::cerr << ",";
        }
        std::cerr << generated_token_ids[i];
      }
      std::cerr << "]\n";
      MaybeTraceDivergentPromptLayers(*model, kRouteA, kRouteC);
      return false;
    }
    route_a_token = route_a_step->selected_token_id;
    route_c_token = route_c_step->selected_token_id;
    generated_token_ids.push_back(route_a_token);
  }

  std::cout << "nano_16_token_correctness_test: PASS generated_tokens=[";
  for (std::size_t i = 0; i < generated_token_ids.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << generated_token_ids[i];
  }
  std::cout << "]\n";
  return true;
}

}  // namespace

int main() {
  return run_nano_correctness_gate() ? 0 : 1;
}
