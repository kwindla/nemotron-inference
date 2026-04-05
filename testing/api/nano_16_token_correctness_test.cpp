#include "nemotron/device_tensor.h"
#include "nemotron/expert_staging_counters.h"
#include "nemotron/linear_op_counters.h"
#include "nemotron/manifest.h"
#include "nemotron/runtime_environment.h"
#include "nemotron/single_token_forward_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
constexpr std::size_t kParityContinuationStepCount = 3;

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

struct LinearCounterSnapshot {
  std::uint64_t dense_fastpath_plan_success = 0;
  std::uint64_t dense_fastpath_plan_fail = 0;
  std::uint64_t dense_fastpath_execute = 0;
  std::uint64_t dense_fastpath_execute_fail = 0;

  std::uint64_t nvfp4_fastpath_plan_success = 0;
  std::uint64_t nvfp4_fastpath_plan_fail = 0;
  std::uint64_t nvfp4_fastpath_execute = 0;
  std::uint64_t nvfp4_fastpath_execute_fail = 0;

  std::uint64_t scaled_fp8_fastpath_execute = 0;
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

  snapshot.nvfp4_fastpath_plan_success =
      counters.nvfp4_fastpath_plan_success.load(std::memory_order_relaxed);
  snapshot.nvfp4_fastpath_plan_fail =
      counters.nvfp4_fastpath_plan_fail.load(std::memory_order_relaxed);
  snapshot.nvfp4_fastpath_execute =
      counters.nvfp4_fastpath_execute.load(std::memory_order_relaxed);
  snapshot.nvfp4_fastpath_execute_fail =
      counters.nvfp4_fastpath_execute_fail.load(std::memory_order_relaxed);

  snapshot.scaled_fp8_fastpath_execute =
      counters.scaled_fp8_fastpath_execute.load(std::memory_order_relaxed);
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

  sum.nvfp4_fastpath_plan_success =
      lhs.nvfp4_fastpath_plan_success + rhs.nvfp4_fastpath_plan_success;
  sum.nvfp4_fastpath_plan_fail =
      lhs.nvfp4_fastpath_plan_fail + rhs.nvfp4_fastpath_plan_fail;
  sum.nvfp4_fastpath_execute = lhs.nvfp4_fastpath_execute + rhs.nvfp4_fastpath_execute;
  sum.nvfp4_fastpath_execute_fail =
      lhs.nvfp4_fastpath_execute_fail + rhs.nvfp4_fastpath_execute_fail;

  sum.scaled_fp8_fastpath_execute =
      lhs.scaled_fp8_fastpath_execute + rhs.scaled_fp8_fastpath_execute;
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

  counters.scaled_fp8_fastpath_execute.store(
      snapshot.scaled_fp8_fastpath_execute,
      std::memory_order_relaxed);
}

void PrintLayerProbeLinearCounters(
    std::size_t layer_index,
    const LinearCounterSnapshot& snapshot) {
  std::cout << "nano_16_token_correctness_test: layer_probe layer_index="
            << layer_index
            << " linear_counters dense_plan_fail=" << snapshot.dense_fastpath_plan_fail
            << " dense_execute_fail=" << snapshot.dense_fastpath_execute_fail
            << " nvfp4_plan_fail=" << snapshot.nvfp4_fastpath_plan_fail
            << " nvfp4_execute_fail=" << snapshot.nvfp4_fastpath_execute_fail << "\n";
  std::cout.flush();
}

struct ExpertStagingSnapshot {
  std::uint64_t total_bytes_uploaded = 0;
  std::uint64_t total_experts_staged = 0;
  std::uint64_t total_staging_calls = 0;
  std::uint64_t staging_elapsed_us = 0;
};

ExpertStagingSnapshot SnapshotExpertStagingCounters() {
  const auto& counters = nemotron::GetExpertStagingCounters();
  ExpertStagingSnapshot snapshot;
  snapshot.total_bytes_uploaded =
      counters.total_bytes_uploaded.load(std::memory_order_relaxed);
  snapshot.total_experts_staged =
      counters.total_experts_staged.load(std::memory_order_relaxed);
  snapshot.total_staging_calls =
      counters.total_staging_calls.load(std::memory_order_relaxed);
  snapshot.staging_elapsed_us =
      counters.staging_elapsed_us.load(std::memory_order_relaxed);
  return snapshot;
}

void PrintUnexpectedExpertStagingWarning(
    std::ostream& stream,
    const ExpertStagingSnapshot& snapshot) {
  if (snapshot.total_bytes_uploaded == 0) {
    return;
  }

  stream
      << "WARNING: nano_16_token_correctness_test: unexpected expert staging cost while "
      << "NEMOTRON_NANO_16_STRICT_EXPERT_STAGING=1"
      << " total_bytes_uploaded=" << snapshot.total_bytes_uploaded
      << " total_experts_staged=" << snapshot.total_experts_staged
      << " staging_elapsed_us=" << snapshot.staging_elapsed_us << "\n";
}

class ScopedExpertStagingCounterReport {
 public:
  explicit ScopedExpertStagingCounterReport(bool strict_expert_staging_enabled)
      : strict_expert_staging_enabled_(strict_expert_staging_enabled) {
    nemotron::ResetExpertStagingCounters();
  }

  ~ScopedExpertStagingCounterReport() {
    nemotron::PrintExpertStagingCounterSummary(std::cout);
    std::cout.flush();

    if (!strict_expert_staging_enabled_) {
      return;
    }

    const ExpertStagingSnapshot snapshot = SnapshotExpertStagingCounters();
    if (snapshot.total_bytes_uploaded == 0) {
      return;
    }

    PrintUnexpectedExpertStagingWarning(std::cerr, snapshot);
    std::cerr.flush();
  }

  ScopedExpertStagingCounterReport(const ScopedExpertStagingCounterReport&) = delete;
  ScopedExpertStagingCounterReport& operator=(const ScopedExpertStagingCounterReport&) =
      delete;

 private:
  bool strict_expert_staging_enabled_ = false;
};

class ScopedLinearCounterReport {
 public:
  ScopedLinearCounterReport() {
    nemotron::ResetLinearOpCounters();
  }

  ~ScopedLinearCounterReport() {
    nemotron::PrintLinearOpCounterSummary(std::cout);
    std::cout.flush();
  }

  ScopedLinearCounterReport(const ScopedLinearCounterReport&) = delete;
  ScopedLinearCounterReport& operator=(const ScopedLinearCounterReport&) = delete;
};

struct PrefillRouteConfig {
  const char* route_id = "";
  const char* description = "";
  bool sequential_prefill_enabled = false;
};

class ScopedRouteOverrides {
 public:
  explicit ScopedRouteOverrides(const PrefillRouteConfig& route) {
    (void)route;
  }

  ScopedRouteOverrides(const ScopedRouteOverrides&) = delete;
  ScopedRouteOverrides& operator=(const ScopedRouteOverrides&) = delete;
};

constexpr PrefillRouteConfig kRouteA = {
    "baseline",
    "sequential single-token contract replay",
    true,
};
constexpr PrefillRouteConfig kRouteC = {
    "native",
    "native batched prefill + decode",
    false,
};

std::string PairId(
    const PrefillRouteConfig& lhs_route,
    const PrefillRouteConfig& rhs_route) {
  return std::string(lhs_route.route_id) + "<->" + rhs_route.route_id;
}

std::string RouteLabel(const PrefillRouteConfig& route) {
  return route.route_id;
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

std::string Trim(std::string value) {
  const auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  const auto begin = std::find_if(value.begin(), value.end(), not_space);
  if (begin == value.end()) {
    return "";
  }
  const auto end = std::find_if(value.rbegin(), value.rend(), not_space).base();
  return std::string(begin, end);
}

std::string TrimTrailingComma(std::string value) {
  value = Trim(std::move(value));
  if (!value.empty() && value.back() == ',') {
    value.pop_back();
  }
  return Trim(std::move(value));
}

bool StartsWith(const std::string& value, const char* prefix) {
  return value.rfind(prefix, 0) == 0;
}

std::optional<std::string> ExtractJsonLineValue(
    const std::string& line,
    const char* key) {
  const std::string trimmed = Trim(line);
  if (!StartsWith(trimmed, key)) {
    return std::nullopt;
  }
  const std::size_t colon = trimmed.find(':');
  if (colon == std::string::npos) {
    return std::nullopt;
  }
  return TrimTrailingComma(trimmed.substr(colon + 1));
}

std::optional<std::int32_t> ParseInt32Value(const std::string& text) {
  try {
    const long long value = std::stoll(TrimTrailingComma(text));
    if (value < std::numeric_limits<std::int32_t>::min() ||
        value > std::numeric_limits<std::int32_t>::max()) {
      return std::nullopt;
    }
    return static_cast<std::int32_t>(value);
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<float> ParseFloatValue(const std::string& text) {
  try {
    return std::stof(TrimTrailingComma(text));
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<std::vector<std::int32_t>> ParseInt32ArrayValue(const std::string& text) {
  const std::string trimmed = TrimTrailingComma(text);
  const std::size_t open = trimmed.find('[');
  const std::size_t close = trimmed.rfind(']');
  if (open == std::string::npos || close == std::string::npos || close < open) {
    return std::nullopt;
  }

  std::string content = Trim(trimmed.substr(open + 1, close - open - 1));
  std::vector<std::int32_t> values;
  if (content.empty()) {
    return values;
  }

  while (!content.empty()) {
    const std::size_t comma = content.find(',');
    const std::string token =
        comma == std::string::npos ? content : content.substr(0, comma);
    const auto parsed = ParseInt32Value(token);
    if (!parsed.has_value()) {
      return std::nullopt;
    }
    values.push_back(*parsed);
    if (comma == std::string::npos) {
      break;
    }
    content = Trim(content.substr(comma + 1));
  }
  return values;
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

struct IndexedLogit {
  std::int32_t index = -1;
  float value = 0.0f;
};

std::optional<std::string> ExtractObjectFieldValue(
    const std::string& object_text,
    const std::string& key) {
  const std::size_t key_pos = object_text.find(key);
  if (key_pos == std::string::npos) {
    return std::nullopt;
  }
  const std::size_t colon = object_text.find(':', key_pos + key.size());
  if (colon == std::string::npos) {
    return std::nullopt;
  }
  const std::size_t comma = object_text.find(',', colon + 1);
  const std::size_t end = comma == std::string::npos ? object_text.size() : comma;
  return Trim(object_text.substr(colon + 1, end - colon - 1));
}

std::optional<std::vector<IndexedLogit>> ParseIndexedLogitArrayValue(
    const std::string& text) {
  const std::string trimmed = TrimTrailingComma(text);
  const std::size_t open = trimmed.find('[');
  const std::size_t close = trimmed.rfind(']');
  if (open == std::string::npos || close == std::string::npos || close < open) {
    return std::nullopt;
  }

  std::string content = Trim(trimmed.substr(open + 1, close - open - 1));
  std::vector<IndexedLogit> values;
  while (!content.empty()) {
    if (content.front() != '{') {
      return std::nullopt;
    }
    const std::size_t object_end = content.find('}');
    if (object_end == std::string::npos) {
      return std::nullopt;
    }
    const std::string object_text = content.substr(1, object_end - 1);
    const auto index_text = ExtractObjectFieldValue(object_text, "\"index\"");
    const auto value_text = ExtractObjectFieldValue(object_text, "\"value\"");
    if (!index_text.has_value() || !value_text.has_value()) {
      return std::nullopt;
    }
    const auto index = ParseInt32Value(*index_text);
    const auto value = ParseFloatValue(*value_text);
    if (!index.has_value() || !value.has_value()) {
      return std::nullopt;
    }
    values.push_back({*index, *value});
    content = Trim(content.substr(object_end + 1));
    if (!content.empty()) {
      if (content.front() != ',') {
        return std::nullopt;
      }
      content = Trim(content.substr(1));
    }
  }
  return values;
}

struct SavedPromptOracle {
  std::filesystem::path path;
  std::int32_t boundary_token_id = -1;
  std::vector<IndexedLogit> boundary_top5;
  std::vector<std::int32_t> generated_token_ids;
};

std::optional<SavedPromptOracle> LoadSavedPromptOracle(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    std::cerr << "nano_16_token_correctness_test: failed to open oracle JSON "
              << path.string() << "\n";
    return std::nullopt;
  }

  SavedPromptOracle oracle;
  oracle.path = path;
  bool found_boundary_token_id = false;
  bool found_boundary_top5 = false;
  bool found_generated_token_ids = false;

  std::string line;
  while (std::getline(input, line)) {
    if (const auto value = ExtractJsonLineValue(line, "\"boundary_token_id\"");
        value.has_value()) {
      const auto parsed = ParseInt32Value(*value);
      if (!parsed.has_value()) {
        std::cerr << "nano_16_token_correctness_test: failed to parse boundary_token_id from "
                  << path.string() << "\n";
        return std::nullopt;
      }
      oracle.boundary_token_id = *parsed;
      found_boundary_token_id = true;
      continue;
    }
    if (const auto value = ExtractJsonLineValue(line, "\"boundary_top5\"");
        value.has_value()) {
      const auto parsed = ParseIndexedLogitArrayValue(*value);
      if (!parsed.has_value()) {
        std::cerr << "nano_16_token_correctness_test: failed to parse boundary_top5 from "
                  << path.string() << "\n";
        return std::nullopt;
      }
      oracle.boundary_top5 = std::move(*parsed);
      found_boundary_top5 = true;
      continue;
    }
    if (const auto value = ExtractJsonLineValue(line, "\"generated_token_ids\"");
        value.has_value()) {
      const auto parsed = ParseInt32ArrayValue(*value);
      if (!parsed.has_value()) {
        std::cerr << "nano_16_token_correctness_test: failed to parse generated_token_ids from "
                  << path.string() << "\n";
        return std::nullopt;
      }
      oracle.generated_token_ids = std::move(*parsed);
      found_generated_token_ids = true;
    }
  }

  if (!input.good() && !input.eof()) {
    std::cerr << "nano_16_token_correctness_test: failed while reading oracle JSON "
              << path.string() << "\n";
    return std::nullopt;
  }
  if (!found_boundary_token_id || !found_boundary_top5 || !found_generated_token_ids) {
    std::cerr << "nano_16_token_correctness_test: oracle JSON missing required fields in "
              << path.string() << "\n";
    return std::nullopt;
  }
  if (oracle.boundary_top5.empty()) {
    std::cerr << "nano_16_token_correctness_test: oracle boundary_top5 is empty in "
              << path.string() << "\n";
    return std::nullopt;
  }
  if (oracle.generated_token_ids.size() != kDecodeTokenCount) {
    std::cerr << "nano_16_token_correctness_test: expected "
              << kDecodeTokenCount
              << " generated_token_ids entries but found "
              << oracle.generated_token_ids.size()
              << " in " << path.string() << "\n";
    return std::nullopt;
  }
  if (oracle.generated_token_ids.front() != oracle.boundary_token_id) {
    std::cerr << "nano_16_token_correctness_test: oracle boundary token "
              << oracle.boundary_token_id
              << " does not match generated_token_ids[0]="
              << oracle.generated_token_ids.front()
              << " in " << path.string() << "\n";
    return std::nullopt;
  }

  return oracle;
}

struct BoundaryObservation {
  std::int32_t selected_token_id = -1;
  std::vector<float> logits_row;
  std::vector<IndexedLogit> sparse_logits;
  std::vector<float> prompt_boundary_embedding_row;
  std::vector<float> prompt_boundary_final_hidden_normed_row;
  double elapsed_ms = 0.0;
  std::size_t sequence_length = 0;
  std::size_t decode_position = 0;
};

BoundaryObservation MakeOracleBoundaryObservation(const SavedPromptOracle& oracle) {
  BoundaryObservation observation;
  observation.selected_token_id = oracle.boundary_token_id;
  observation.sparse_logits = oracle.boundary_top5;
  observation.sequence_length = kPromptTokenCount;
  observation.decode_position = kPromptTokenCount - 1;
  return observation;
}

std::optional<float> FindIndexedLogitValue(
    const std::vector<IndexedLogit>& logits,
    std::int32_t index) {
  for (const IndexedLogit& logit : logits) {
    if (logit.index == index) {
      return logit.value;
    }
  }
  return std::nullopt;
}

float SparseVsDenseMaxAbsDiff(
    const std::vector<IndexedLogit>& sparse_logits,
    const std::vector<float>& dense_logits) {
  if (sparse_logits.empty() || dense_logits.empty()) {
    return std::numeric_limits<float>::infinity();
  }
  float max_diff = 0.0f;
  for (const IndexedLogit& logit : sparse_logits) {
    if (logit.index < 0 ||
        static_cast<std::size_t>(logit.index) >= dense_logits.size()) {
      return std::numeric_limits<float>::infinity();
    }
    max_diff = std::max(
        max_diff,
        std::fabs(logit.value - dense_logits[static_cast<std::size_t>(logit.index)]));
  }
  return max_diff;
}

float MaxAbsDiff(const BoundaryObservation& lhs, const BoundaryObservation& rhs) {
  if (!lhs.sparse_logits.empty() || !rhs.sparse_logits.empty()) {
    if (!lhs.sparse_logits.empty() && rhs.sparse_logits.empty()) {
      return SparseVsDenseMaxAbsDiff(lhs.sparse_logits, rhs.logits_row);
    }
    if (lhs.sparse_logits.empty() && !rhs.sparse_logits.empty()) {
      return SparseVsDenseMaxAbsDiff(rhs.sparse_logits, lhs.logits_row);
    }

    float max_diff = 0.0f;
    for (const IndexedLogit& logit : lhs.sparse_logits) {
      const auto rhs_value = FindIndexedLogitValue(rhs.sparse_logits, logit.index);
      if (!rhs_value.has_value()) {
        return std::numeric_limits<float>::infinity();
      }
      max_diff = std::max(max_diff, std::fabs(logit.value - *rhs_value));
    }
    for (const IndexedLogit& logit : rhs.sparse_logits) {
      const auto lhs_value = FindIndexedLogitValue(lhs.sparse_logits, logit.index);
      if (!lhs_value.has_value()) {
        return std::numeric_limits<float>::infinity();
      }
      max_diff = std::max(max_diff, std::fabs(logit.value - *lhs_value));
    }
    return max_diff;
  }

  return MaxAbsDiff(lhs.logits_row, rhs.logits_row);
}

std::string IndexedTopKSummary(const std::vector<IndexedLogit>& logits) {
  if (logits.empty()) {
    return "[]";
  }
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < logits.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << logits[i].index << ":" << logits[i].value;
  }
  out << "]";
  return out.str();
}

std::string BoundaryTopKSummary(const BoundaryObservation& observation) {
  if (!observation.sparse_logits.empty()) {
    return IndexedTopKSummary(observation.sparse_logits);
  }
  return TopKSummary(observation.logits_row, 5);
}

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
  comparison.max_abs_diff = MaxAbsDiff(lhs_observation, rhs_observation);
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
            << kRouteA.route_id << "=(" << kRouteA.description << "), "
            << kRouteC.route_id << "=(" << kRouteC.description << ")\n";
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

  std::cout << "nano_16_token_correctness_test: " << RouteDetail(route)
            << " prefill start prompt_tokens=" << kPromptTokenCount << "\n";
  std::cout.flush();
  const auto start = std::chrono::steady_clock::now();
  BoundaryObservation observation;

  if (route.sequential_prefill_enabled) {
    auto logits = nemotron::DeviceTensorFp32::Create({1, model.config().vocab_size});
    if (logits == nullptr || !logits->valid()) {
      return std::nullopt;
    }

    nemotron::SingleTokenForwardTrace final_trace;
    for (std::size_t token_index = 0; token_index < prompt_token_ids.size(); ++token_index) {
      nemotron::SingleTokenForwardTrace* trace_ptr =
          capture_trace_rows && token_index + 1 == prompt_token_ids.size()
              ? &final_trace
              : nullptr;
      const bool token_ok =
          token_index == 0
              ? model.RunSingleToken(
                    prompt_token_ids[token_index],
                    request_context,
                    logits.get(),
                    /*capture_layer_indices=*/{},
                    trace_ptr)
              : model.ContinueSingleToken(
                    prompt_token_ids[token_index],
                    request_context,
                    logits.get(),
                    /*capture_layer_indices=*/{},
                    trace_ptr);
      if (!token_ok) {
        return std::nullopt;
      }
    }

    observation.logits_row = CopyTensorToHost(*logits);
    if (capture_trace_rows) {
      observation.prompt_boundary_embedding_row = std::move(final_trace.embedding_output);
      observation.prompt_boundary_final_hidden_normed_row =
          std::move(final_trace.final_hidden_normed);
    }
  } else {
    auto logits =
        nemotron::DeviceTensorFp32::Create({kPromptTokenCount, model.config().vocab_size});
    if (logits == nullptr || !logits->valid()) {
      return std::nullopt;
    }
    nemotron::SingleTokenForwardTrace trace;
    nemotron::SingleTokenForwardTrace* trace_ptr = capture_trace_rows ? &trace : nullptr;
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
    observation.logits_row =
        SliceRow(host_logits, prompt_token_ids.size() - 1, model.config().vocab_size);
    if (capture_trace_rows) {
      const std::size_t prompt_boundary_index = prompt_token_ids.size() - 1;
      const std::size_t hidden_size = model.config().hidden_size;
      auto embedding_row = ExtractTraceRow(
          trace.embedding_output,
          prompt_token_ids.size(),
          prompt_boundary_index,
          hidden_size);
      if (!embedding_row.has_value()) {
        std::cerr
            << "nano_16_token_correctness_test: failed to capture prompt-boundary embedding row "
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
  }

  const auto end = std::chrono::steady_clock::now();
  const auto selected_token_id = ArgMaxTokenId(observation.logits_row);
  if (!selected_token_id.has_value()) {
    return std::nullopt;
  }

  observation.selected_token_id = *selected_token_id;
  observation.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
  observation.sequence_length = request_context.sequence_length();
  observation.decode_position = request_context.decode_position();

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
            << MaxAbsDiff(lhs_observation, rhs_observation)
            << "\n";
  std::cerr << lhs_route.route_id << "_top5="
            << BoundaryTopKSummary(lhs_observation) << "\n";
  std::cerr << rhs_route.route_id << "_top5="
            << BoundaryTopKSummary(rhs_observation) << "\n";
  std::cerr << lhs_route.route_id << "_state=(sequence_length="
            << lhs_observation.sequence_length
            << ", decode_position=" << lhs_observation.decode_position << ")\n";
  std::cerr << rhs_route.route_id << "_state=(sequence_length="
            << rhs_observation.sequence_length
            << ", decode_position=" << rhs_observation.decode_position << ")\n";
}

void PrintTokenSequence(std::ostream& stream, const std::vector<std::int32_t>& token_ids) {
  for (std::size_t i = 0; i < token_ids.size(); ++i) {
    if (i != 0) {
      stream << ",";
    }
    stream << token_ids[i];
  }
}

void PrintOracleDecodeMismatch(
    std::size_t token_index,
    std::int32_t consume_token_id,
    std::int32_t expected_token_id,
    const PrefillRouteConfig& route,
    const BoundaryObservation& observation,
    const std::vector<std::int32_t>& oracle_generated_token_ids) {
  std::cerr << "nano_16_token_correctness_test: divergent decode boundary pair="
            << PairId(kRouteA, route)
            << " index=" << token_index
            << " consume_token=" << consume_token_id
            << " oracle_token=" << expected_token_id
            << " " << route.route_id << "_token=" << observation.selected_token_id
            << "\n";
  std::cerr << route.route_id << "_top5="
            << BoundaryTopKSummary(observation) << "\n";
  std::cerr << route.route_id << "_state=(sequence_length="
            << observation.sequence_length
            << ", decode_position=" << observation.decode_position << ")\n";
  std::cerr << "oracle_generated_tokens_before_divergence=[";
  PrintTokenSequence(
      std::cerr,
      std::vector<std::int32_t>(
          oracle_generated_token_ids.begin(),
          oracle_generated_token_ids.begin() +
              static_cast<std::ptrdiff_t>(token_index)));
  std::cerr << "]\n";
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
  const std::size_t logit_rows = route.sequential_prefill_enabled ? 1 : kPromptTokenCount;
  auto logits = nemotron::DeviceTensorFp32::Create({logit_rows, model.config().vocab_size});
  if (logits == nullptr || !logits->valid()) {
    return std::nullopt;
  }

  nemotron::SingleTokenForwardTrace trace;
  std::cout << "nano_16_token_correctness_test: " << RouteDetail(route)
            << " layer compare start layer_index=" << layer_index << "\n";
  std::cout.flush();
  const auto start = std::chrono::steady_clock::now();
  if (route.sequential_prefill_enabled) {
    const auto& prompt_token_ids = FixedPromptTokenIds();
    for (std::size_t token_index = 0; token_index < prompt_token_ids.size(); ++token_index) {
      nemotron::SingleTokenForwardTrace* trace_ptr =
          token_index + 1 == prompt_token_ids.size() ? &trace : nullptr;
      const bool token_ok =
          token_index == 0
              ? model.RunSingleToken(
                    prompt_token_ids[token_index],
                    *request_context,
                    logits.get(),
                    {layer_index},
                    trace_ptr,
                    layer_index)
              : model.ContinueSingleToken(
                    prompt_token_ids[token_index],
                    *request_context,
                    logits.get(),
                    {layer_index},
                    trace_ptr,
                    layer_index);
      if (!token_ok) {
        return std::nullopt;
      }
    }
  } else if (!model.RunPrefill(
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
    const std::size_t hidden_size = model.config().hidden_size;
    if (hidden_size != 0 &&
        it->hidden.size() > hidden_size &&
        (it->hidden.size() % hidden_size) == 0) {
      observation.hidden_row.assign(
          it->hidden.end() - static_cast<std::ptrdiff_t>(hidden_size),
          it->hidden.end());
    } else {
      observation.hidden_row = it->hidden;
    }
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

  const bool strict_expert_staging_enabled =
      EnvEnabledOrDefault("NEMOTRON_NANO_16_STRICT_EXPERT_STAGING", false);
  const bool capture_embedding_trace =
      EnvEnabledOrDefault("NEMOTRON_NANO_16_TRACE_EMBEDDING", false);
  ScopedExpertStagingCounterReport expert_staging_counter_report(
      strict_expert_staging_enabled);
  ScopedLinearCounterReport linear_counter_report;

  const char* oracle_env = std::getenv("NEMOTRON_NANO_16_ORACLE_PATH");
  const bool using_saved_oracle = oracle_env != nullptr && oracle_env[0] != '\0';
  std::optional<SavedPromptOracle> saved_oracle;
  if (using_saved_oracle) {
    saved_oracle = LoadSavedPromptOracle(std::filesystem::path(oracle_env));
    if (!expect(saved_oracle.has_value(), "saved oracle JSON should load")) {
      return false;
    }
    std::cout << "nano_16_token_correctness_test: using saved oracle from "
              << saved_oracle->path.string() << "\n";
  } else {
    std::cout << "nano_16_token_correctness_test: running live baseline contract reference\n";
  }
  std::cout.flush();

  std::unique_ptr<nemotron::RequestExecutionContext> route_a_context;
  if (!using_saved_oracle) {
    route_a_context = model->CreateRequestContext();
  }
  auto route_c_context = model->CreateRequestContext();
  if ((!using_saved_oracle &&
       !expect(
           route_a_context != nullptr && route_a_context->valid(),
           "baseline request context should create")) ||
      !expect(
          route_c_context != nullptr && route_c_context->valid(),
          "native request context should create")) {
    return false;
  }

  std::optional<BoundaryObservation> route_a_live_prefill;
  if (!using_saved_oracle) {
    route_a_live_prefill =
        RunPrefillBoundary(*model, kRouteA, *route_a_context, capture_embedding_trace);
  }
  const auto route_c_prefill =
      RunPrefillBoundary(*model, kRouteC, *route_c_context, capture_embedding_trace);
  if ((!using_saved_oracle &&
       !expect(route_a_live_prefill.has_value(), "baseline prefill boundary should succeed")) ||
      !expect(route_c_prefill.has_value(), "native prefill boundary should succeed")) {
    return false;
  }
  const BoundaryObservation route_a_prefill =
      using_saved_oracle ? MakeOracleBoundaryObservation(*saved_oracle) : *route_a_live_prefill;

  const BoundaryComparison route_ac_comparison =
      MakeBoundaryComparison(kRouteA, route_a_prefill, kRouteC, *route_c_prefill);
  PrintPrefillSummaryTable({route_ac_comparison});
  if (capture_embedding_trace && !using_saved_oracle) {
    PrintTraceComparisonSummary(kRouteA, route_a_prefill, kRouteC, *route_c_prefill);
  }

  if (HasSelectedTokenMismatch(route_ac_comparison)) {
    PrintBoundaryMismatch("prefill", 0, kRouteA, route_a_prefill, kRouteC, *route_c_prefill);
    if (!using_saved_oracle) {
      MaybeTraceDivergentPromptLayers(*model, kRouteA, kRouteC);
    }
    return false;
  }

  if (using_saved_oracle) {
    const std::vector<std::int32_t>& generated_token_ids = saved_oracle->generated_token_ids;
    for (std::size_t token_index = 1; token_index < generated_token_ids.size(); ++token_index) {
      const std::int32_t consume_token = generated_token_ids[token_index - 1];
      const std::int32_t expected_token = generated_token_ids[token_index];
      const auto route_c_step = RunContinuationBoundary(
          *model,
          kRouteC,
          *route_c_context,
          consume_token,
          token_index);
      if (!expect(route_c_step.has_value(), "native continuation boundary should succeed")) {
        return false;
      }

      const bool route_c_match = route_c_step->selected_token_id == expected_token;
      if (route_c_match) {
        continue;
      }

      PrintOracleDecodeMismatch(
          token_index,
          consume_token,
          expected_token,
          kRouteC,
          *route_c_step,
          generated_token_ids);
      return false;
    }

    std::cout << "nano_16_token_correctness_test: PASS generated_tokens=[";
    PrintTokenSequence(std::cout, generated_token_ids);
    std::cout << "]\n";
    return true;
  }

  std::int32_t route_a_token = route_a_prefill.selected_token_id;
  std::int32_t route_c_token = route_c_prefill->selected_token_id;
  std::vector<std::int32_t> generated_token_ids = {route_a_token};

  for (std::size_t token_index = 1;
       token_index <= kParityContinuationStepCount;
       ++token_index) {
    const auto route_a_step =
        RunContinuationBoundary(*model, kRouteA, *route_a_context, route_a_token, token_index);
    const auto route_c_step =
        RunContinuationBoundary(*model, kRouteC, *route_c_context, route_c_token, token_index);
    if (!expect(route_a_step.has_value(), "baseline continuation boundary should succeed") ||
        !expect(route_c_step.has_value(), "native continuation boundary should succeed")) {
      return false;
    }
    if (route_a_step->selected_token_id != route_c_step->selected_token_id) {
      PrintBoundaryMismatch("decode", token_index, kRouteA, *route_a_step, kRouteC, *route_c_step);
      std::cerr << "baseline_top1=[" << TopKSummary(route_a_step->logits_row, 1) << "]\n";
      std::cerr << "generated_tokens_before_divergence=[";
      PrintTokenSequence(std::cerr, generated_token_ids);
      std::cerr << "]\n";
      MaybeTraceDivergentPromptLayers(*model, kRouteA, kRouteC);
      return false;
    }
    route_a_token = route_a_step->selected_token_id;
    route_c_token = route_c_step->selected_token_id;
    generated_token_ids.push_back(route_a_token);
  }

  std::cout << "nano_16_token_correctness_test: PASS generated_tokens=[";
  PrintTokenSequence(std::cout, generated_token_ids);
  std::cout << "]\n";
  return true;
}

}  // namespace

int main() {
  return run_nano_correctness_gate() ? 0 : 1;
}
