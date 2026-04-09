#include "nemotron/attention_layer.h"
#include "nemotron/device_tensor.h"
#include "nemotron/expert_layer.h"
#include "nemotron/expert_staging_counters.h"
#include "nemotron/linear_op_counters.h"
#include "nemotron/mamba_layer.h"
#include "nemotron/manifest.h"
#include "nemotron/runtime_environment.h"
#include "nemotron/single_token_forward_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr const char* kNanoModelId = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4";
constexpr const char* kSuperModelId = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4";

struct Options {
  std::filesystem::path manifest_path;
  std::size_t default_max_new_tokens = 4096;
  std::size_t target_context_tokens = 8192;
  bool prefix_cache_enabled = true;
  std::string tenant_namespace = "interactive-forward";
  std::string serializer_revision = "interactive-chat-v1";
};

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

struct ExpertStagingSnapshot {
  std::uint64_t total_bytes_uploaded = 0;
  std::uint64_t total_experts_staged = 0;
  std::uint64_t total_staging_calls = 0;
  std::uint64_t monolithic_layers = 0;
  std::uint64_t staging_elapsed_us = 0;
};

struct TurnCommand {
  std::size_t max_new_tokens = 0;
  std::vector<std::int32_t> eos_token_ids;
  std::vector<std::int32_t> prompt_token_ids;
};

struct TurnResponse {
  bool ok = false;
  std::string error;
  std::string conversation_id;
  bool prefix_cache_enabled = false;
  std::string cache_mode = "disabled";
  bool cache_lookup_hit = false;
  std::string cache_lookup_source = "none";
  bool cache_restore_ok = false;
  bool exact_cache_hit = false;
  std::size_t prompt_token_count = 0;
  std::size_t matched_prefix_tokens = 0;
  std::size_t prefill_token_count = 0;
  std::size_t generated_token_count = 0;
  bool hit_eos = false;
  bool hit_capacity_limit = false;
  std::size_t sequence_length_after = 0;
  std::size_t max_context_tokens = 0;
  double cache_lookup_ms = 0.0;
  double cache_restore_ms = 0.0;
  double prefill_ms = 0.0;
  double prompt_logits_copy_ms = 0.0;
  double first_token_select_ms = 0.0;
  double decode_total_ms = 0.0;
  double first_decode_step_ms = 0.0;
  double decode_mean_ms = 0.0;
  double decode_min_ms = 0.0;
  double decode_max_ms = 0.0;
  double total_ms = 0.0;
  double decode_tokens_per_second = 0.0;
  double total_generated_tokens_per_second = 0.0;
  std::vector<double> decode_step_ms;
  std::vector<std::int32_t> generated_token_ids;
  nemotron::AttentionLayerExecutionCounters attention_counters;
  nemotron::MambaLayerExecutionCounters mamba_counters;
  nemotron::ExpertLayerExecutionCounters expert_counters;
  LinearCounterSnapshot linear_counters;
  ExpertStagingSnapshot expert_staging_counters;
  std::size_t prefix_cache_current_bytes = 0;
  std::size_t prefix_cache_node_count = 0;
  std::size_t prefix_cache_global_root_count = 0;
};

struct ServerState {
  Options options;
  nemotron::PackedModelManifest manifest;
  std::unique_ptr<nemotron::RuntimeEnvironment> environment;
  std::unique_ptr<nemotron::SingleTokenForwardModel> model;
  std::size_t conversation_serial = 0;
};

bool ParseNonNegativeSizeT(const std::string& value, std::size_t* parsed_value) {
  if (parsed_value == nullptr) {
    return false;
  }
  try {
    *parsed_value = std::stoull(value);
    return true;
  } catch (...) {
    return false;
  }
}

bool ParseEnabledFlag(const std::string& value, bool* enabled) {
  if (enabled == nullptr) {
    return false;
  }
  if (value == "1" || value == "on" || value == "true" || value == "enabled") {
    *enabled = true;
    return true;
  }
  if (value == "0" || value == "off" || value == "false" || value == "disabled") {
    *enabled = false;
    return true;
  }
  return false;
}

bool ParseArgs(int argc, char** argv, Options* options) {
  if (options == nullptr) {
    return false;
  }
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--manifest") {
      if (i + 1 >= argc) {
        return false;
      }
      options->manifest_path = std::filesystem::path(argv[++i]);
    } else if (arg == "--max-new-tokens") {
      if (i + 1 >= argc ||
          !ParseNonNegativeSizeT(argv[++i], &options->default_max_new_tokens)) {
        return false;
      }
    } else if (arg == "--target-context-tokens") {
      if (i + 1 >= argc ||
          !ParseNonNegativeSizeT(argv[++i], &options->target_context_tokens)) {
        return false;
      }
    } else if (arg == "--prefix-cache") {
      if (i + 1 >= argc ||
          !ParseEnabledFlag(argv[++i], &options->prefix_cache_enabled)) {
        return false;
      }
    } else if (arg == "--tenant-namespace") {
      if (i + 1 >= argc) {
        return false;
      }
      options->tenant_namespace = argv[++i];
    } else if (arg == "--serializer-revision") {
      if (i + 1 >= argc) {
        return false;
      }
      options->serializer_revision = argv[++i];
    } else {
      return false;
    }
  }
  return !options->manifest_path.empty() && options->target_context_tokens != 0;
}

std::string EscapeJson(std::string_view text) {
  std::ostringstream escaped;
  for (const char ch : text) {
    switch (ch) {
      case '\\':
        escaped << "\\\\";
        break;
      case '"':
        escaped << "\\\"";
        break;
      case '\n':
        escaped << "\\n";
        break;
      case '\r':
        escaped << "\\r";
        break;
      case '\t':
        escaped << "\\t";
        break;
      default:
        escaped << ch;
        break;
    }
  }
  return escaped.str();
}

template <typename T>
void WriteJsonIntArray(std::ostream& output, const std::vector<T>& values) {
  output << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      output << ",";
    }
    output << values[i];
  }
  output << "]";
}

void WriteJsonDoubleArray(std::ostream& output, const std::vector<double>& values) {
  output << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      output << ",";
    }
    output << std::fixed << std::setprecision(6) << values[i];
  }
  output << "]";
}

std::vector<std::string> SplitTabFields(const std::string& line) {
  std::vector<std::string> fields;
  std::string current;
  for (const char ch : line) {
    if (ch == '\t') {
      fields.push_back(current);
      current.clear();
      continue;
    }
    current.push_back(ch);
  }
  fields.push_back(current);
  return fields;
}

std::optional<std::vector<std::int32_t>> ParseInt32Csv(const std::string& csv) {
  std::vector<std::int32_t> values;
  if (csv.empty()) {
    return values;
  }
  std::stringstream stream(csv);
  std::string part;
  while (std::getline(stream, part, ',')) {
    if (part.empty()) {
      return std::nullopt;
    }
    try {
      values.push_back(static_cast<std::int32_t>(std::stoll(part)));
    } catch (...) {
      return std::nullopt;
    }
  }
  return values;
}

LinearCounterSnapshot SnapshotLinearCounters() {
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

ExpertStagingSnapshot SnapshotExpertStagingCounters() {
  const auto& counters = nemotron::GetExpertStagingCounters();
  ExpertStagingSnapshot snapshot;
  snapshot.total_bytes_uploaded =
      counters.total_bytes_uploaded.load(std::memory_order_relaxed);
  snapshot.total_experts_staged =
      counters.total_experts_staged.load(std::memory_order_relaxed);
  snapshot.total_staging_calls =
      counters.total_staging_calls.load(std::memory_order_relaxed);
  snapshot.monolithic_layers =
      counters.monolithic_layers.load(std::memory_order_relaxed);
  snapshot.staging_elapsed_us =
      counters.staging_elapsed_us.load(std::memory_order_relaxed);
  return snapshot;
}

double ElapsedMs(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

std::vector<float> CopyTensorToHost(const nemotron::DeviceTensorFp32& tensor) {
  std::vector<float> host(tensor.numel(), 0.0f);
  if (!tensor.CopyToHost(host.data(), host.size())) {
    return {};
  }
  return host;
}

bool AllFinite(const std::vector<float>& values) {
  return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

bool ContainsTokenId(const std::vector<std::int32_t>& token_ids, std::int32_t token_id) {
  return std::find(token_ids.begin(), token_ids.end(), token_id) != token_ids.end();
}

std::optional<std::int32_t> ArgMaxTokenId(
    const std::vector<float>& logits,
    std::size_t row_index,
    std::size_t row_width) {
  if (row_width == 0 || logits.size() < ((row_index + 1) * row_width)) {
    return std::nullopt;
  }
  const std::size_t row_offset = row_index * row_width;
  auto best_it = logits.begin() + static_cast<std::ptrdiff_t>(row_offset);
  for (auto it = best_it + 1;
       it != logits.begin() + static_cast<std::ptrdiff_t>(row_offset + row_width);
       ++it) {
    if (*it > *best_it) {
      best_it = it;
    }
  }
  return static_cast<std::int32_t>(
      std::distance(logits.begin() + static_cast<std::ptrdiff_t>(row_offset), best_it));
}

std::optional<nemotron::SingleTokenForwardConfig> ConfigForManifest(
    const nemotron::PackedModelManifest& manifest) {
  if (manifest.runtime.model_id == kNanoModelId) {
    return nemotron::KnownNemotron3Nano30BA3BConfig();
  }
  if (manifest.runtime.model_id == kSuperModelId) {
    return nemotron::KnownNemotron3Super120BA12BConfig();
  }
  return std::nullopt;
}

nemotron::RuntimeBootstrapOptions MakeRuntimeOptions(std::size_t target_context_tokens) {
  auto GiB = [](std::size_t value) { return value * 1024ull * 1024ull * 1024ull; };
  nemotron::RuntimeBootstrapOptions options;
  options.service_target.total_memory_bytes = GiB(128);
  options.service_target.weights_bytes = GiB(100);
  options.service_target.workspace_bytes = GiB(8);
  options.service_target.graph_bytes = GiB(4);
  options.service_target.safety_headroom_bytes = GiB(4);
  options.service_target.target_active_requests = 1;
  options.service_target.target_context_tokens = target_context_tokens;
  options.use_fp16_mamba_state = false;
  options.reusable_node_metadata_bytes = 4096;
  options.verify_manifest_files = false;
  options.materialize_weight_arena = false;
  return options;
}

std::string CacheMatchSourceName(nemotron::CacheMatchSource source) {
  switch (source) {
    case nemotron::CacheMatchSource::kConversationCommittedHead:
      return "conversation_committed_head";
    case nemotron::CacheMatchSource::kGlobalRoot:
      return "global_root";
    case nemotron::CacheMatchSource::kNone:
      break;
  }
  return "none";
}

std::string CurrentConversationId(const ServerState& state) {
  return "interactive-" + std::to_string(state.conversation_serial);
}

std::string ReadyJson(const ServerState& state) {
  std::ostringstream output;
  output << "{"
         << "\"ok\":true,"
         << "\"status\":\"ready\","
         << "\"model_id\":\"" << EscapeJson(state.manifest.runtime.model_id) << "\","
         << "\"conversation_id\":\"" << EscapeJson(CurrentConversationId(state)) << "\","
         << "\"prefix_cache_enabled\":"
         << (state.environment->prefix_cache().enabled() ? "true" : "false") << ","
         << "\"target_context_tokens\":" << state.options.target_context_tokens
         << "}";
  return output.str();
}

std::string StatusJson(
    const std::string& status,
    const ServerState& state) {
  std::ostringstream output;
  output << "{"
         << "\"ok\":true,"
         << "\"status\":\"" << EscapeJson(status) << "\","
         << "\"conversation_id\":\"" << EscapeJson(CurrentConversationId(state)) << "\","
         << "\"prefix_cache_enabled\":"
         << (state.environment->prefix_cache().enabled() ? "true" : "false") << ","
         << "\"prefix_cache_current_bytes\":" << state.environment->prefix_cache().current_bytes() << ","
         << "\"prefix_cache_node_count\":" << state.environment->prefix_cache().node_count() << ","
         << "\"prefix_cache_global_root_count\":" << state.environment->prefix_cache().global_root_count()
         << "}";
  return output.str();
}

std::string ErrorJson(const std::string& error) {
  std::ostringstream output;
  output << "{"
         << "\"ok\":false,"
         << "\"error\":\"" << EscapeJson(error) << "\""
         << "}";
  return output.str();
}

std::string TurnResponseJson(const TurnResponse& response) {
  std::ostringstream output;
  output << "{"
         << "\"ok\":" << (response.ok ? "true" : "false") << ","
         << "\"conversation_id\":\"" << EscapeJson(response.conversation_id) << "\","
         << "\"prefix_cache_enabled\":"
         << (response.prefix_cache_enabled ? "true" : "false") << ","
         << "\"cache_mode\":\"" << EscapeJson(response.cache_mode) << "\","
         << "\"cache_lookup_hit\":" << (response.cache_lookup_hit ? "true" : "false") << ","
         << "\"cache_lookup_source\":\"" << EscapeJson(response.cache_lookup_source) << "\","
         << "\"cache_restore_ok\":" << (response.cache_restore_ok ? "true" : "false") << ","
         << "\"exact_cache_hit\":" << (response.exact_cache_hit ? "true" : "false") << ","
         << "\"prompt_token_count\":" << response.prompt_token_count << ","
         << "\"matched_prefix_tokens\":" << response.matched_prefix_tokens << ","
         << "\"prefill_token_count\":" << response.prefill_token_count << ","
         << "\"generated_token_count\":" << response.generated_token_count << ","
         << "\"hit_eos\":" << (response.hit_eos ? "true" : "false") << ","
         << "\"hit_capacity_limit\":" << (response.hit_capacity_limit ? "true" : "false") << ","
         << "\"sequence_length_after\":" << response.sequence_length_after << ","
         << "\"max_context_tokens\":" << response.max_context_tokens << ","
         << "\"cache_lookup_ms\":" << std::fixed << std::setprecision(6) << response.cache_lookup_ms << ","
         << "\"cache_restore_ms\":" << response.cache_restore_ms << ","
         << "\"prefill_ms\":" << response.prefill_ms << ","
         << "\"prompt_logits_copy_ms\":" << response.prompt_logits_copy_ms << ","
         << "\"first_token_select_ms\":" << response.first_token_select_ms << ","
         << "\"decode_total_ms\":" << response.decode_total_ms << ","
         << "\"first_decode_step_ms\":" << response.first_decode_step_ms << ","
         << "\"decode_mean_ms\":" << response.decode_mean_ms << ","
         << "\"decode_min_ms\":" << response.decode_min_ms << ","
         << "\"decode_max_ms\":" << response.decode_max_ms << ","
         << "\"total_ms\":" << response.total_ms << ","
         << "\"decode_tokens_per_second\":" << response.decode_tokens_per_second << ","
         << "\"total_generated_tokens_per_second\":" << response.total_generated_tokens_per_second << ","
         << "\"generated_token_ids\":";
  WriteJsonIntArray(output, response.generated_token_ids);
  output << ",\"decode_step_ms\":";
  WriteJsonDoubleArray(output, response.decode_step_ms);
  output << ",\"attention\":{"
         << "\"native_multi_token_runs\":" << response.attention_counters.native_multi_token_runs << ","
         << "\"native_multi_token_tokens\":" << response.attention_counters.native_multi_token_tokens << ","
         << "\"row_replay_runs\":" << response.attention_counters.row_replay_runs << ","
         << "\"row_replay_tokens\":" << response.attention_counters.row_replay_tokens
         << "},\"mamba\":{"
         << "\"native_multi_token_runs\":" << response.mamba_counters.native_multi_token_runs << ","
         << "\"native_multi_token_tokens\":" << response.mamba_counters.native_multi_token_tokens << ","
         << "\"row_replay_runs\":" << response.mamba_counters.row_replay_runs << ","
         << "\"row_replay_tokens\":" << response.mamba_counters.row_replay_tokens
         << "},\"expert\":{"
         << "\"native_multi_token_runs\":" << response.expert_counters.native_multi_token_runs << ","
         << "\"native_multi_token_tokens\":" << response.expert_counters.native_multi_token_tokens << ","
         << "\"row_replay_runs\":" << response.expert_counters.row_replay_runs << ","
         << "\"row_replay_tokens\":" << response.expert_counters.row_replay_tokens
         << "},\"linear\":{"
         << "\"dense_fastpath_plan_success\":" << response.linear_counters.dense_fastpath_plan_success << ","
         << "\"dense_fastpath_plan_fail\":" << response.linear_counters.dense_fastpath_plan_fail << ","
         << "\"dense_fastpath_execute\":" << response.linear_counters.dense_fastpath_execute << ","
         << "\"dense_fastpath_execute_fail\":" << response.linear_counters.dense_fastpath_execute_fail << ","
         << "\"nvfp4_fastpath_plan_success\":" << response.linear_counters.nvfp4_fastpath_plan_success << ","
         << "\"nvfp4_fastpath_plan_fail\":" << response.linear_counters.nvfp4_fastpath_plan_fail << ","
         << "\"nvfp4_fastpath_execute\":" << response.linear_counters.nvfp4_fastpath_execute << ","
         << "\"nvfp4_fastpath_execute_fail\":" << response.linear_counters.nvfp4_fastpath_execute_fail << ","
         << "\"scaled_fp8_fastpath_execute\":" << response.linear_counters.scaled_fp8_fastpath_execute
         << "},\"expert_staging\":{"
         << "\"total_bytes_uploaded\":" << response.expert_staging_counters.total_bytes_uploaded << ","
         << "\"total_experts_staged\":" << response.expert_staging_counters.total_experts_staged << ","
         << "\"total_staging_calls\":" << response.expert_staging_counters.total_staging_calls << ","
         << "\"monolithic_layers\":" << response.expert_staging_counters.monolithic_layers << ","
         << "\"staging_elapsed_us\":" << response.expert_staging_counters.staging_elapsed_us
         << "},\"prefix_cache\":{"
         << "\"current_bytes\":" << response.prefix_cache_current_bytes << ","
         << "\"node_count\":" << response.prefix_cache_node_count << ","
         << "\"global_root_count\":" << response.prefix_cache_global_root_count
         << "}";
  if (!response.ok) {
    output << ",\"error\":\"" << EscapeJson(response.error) << "\"";
  }
  output << "}";
  return output.str();
}

bool ExecuteTurn(
    ServerState* state,
    const TurnCommand& command,
    TurnResponse* response) {
  if (state == nullptr || response == nullptr) {
    return false;
  }

  response->conversation_id = CurrentConversationId(*state);
  response->prefix_cache_enabled = state->environment->prefix_cache().enabled();
  response->cache_mode = response->prefix_cache_enabled ? "cold" : "disabled";
  response->prompt_token_count = command.prompt_token_ids.size();
  response->max_context_tokens = state->model->plan().request_config.max_tokens;

  if (command.prompt_token_ids.empty()) {
    response->error = "prompt token ids must be non-empty";
    return false;
  }

  auto request_context = state->model->CreateRequestContext();
  if (request_context == nullptr || !request_context->valid()) {
    response->error = "request context creation failed";
    return false;
  }

  nemotron::ResetLinearOpCounters();
  nemotron::ResetExpertStagingCounters();
  nemotron::ResetAttentionLayerExecutionCounters();
  nemotron::ResetMambaLayerExecutionCounters();
  nemotron::ResetExpertLayerExecutionCounters();

  nemotron::SerializedPromptIdentity identity;
  identity.token_ids = command.prompt_token_ids;
  identity.tenant_namespace = state->options.tenant_namespace;
  identity.tokenizer_revision = state->manifest.runtime.tokenizer_revision;
  identity.serializer_revision = state->options.serializer_revision;
  identity.model_revision = state->manifest.runtime.model_id;

  const auto total_start = std::chrono::steady_clock::now();
  std::size_t matched_prefix_tokens = 0;
  const std::int32_t* prefill_token_ids = identity.token_ids.data();
  std::size_t prefill_token_count = identity.token_ids.size();
  bool resumed_from_cache = false;
  std::vector<float> boundary_logits_host;

  if (state->environment->prefix_cache().enabled()) {
    nemotron::CacheLookupRequest lookup_request;
    lookup_request.identity = identity;
    lookup_request.conversation_id = response->conversation_id;
    const auto lookup_start = std::chrono::steady_clock::now();
    const nemotron::CacheMatch match = state->environment->prefix_cache().Lookup(lookup_request);
    const auto lookup_end = std::chrono::steady_clock::now();
    response->cache_lookup_ms = ElapsedMs(lookup_start, lookup_end);
    response->cache_lookup_hit = match.hit();
    response->cache_lookup_source = CacheMatchSourceName(match.source);
    const bool allow_partial_conversation_resume =
        !(state->model->config().moe_prefill_window_tokens != 0 &&
          match.source == nemotron::CacheMatchSource::kConversationCommittedHead &&
          match.matched_token_count < identity.token_ids.size());
    if (match.hit() && match.matched_token_count > 0 && allow_partial_conversation_resume) {
      const auto restore_start = std::chrono::steady_clock::now();
      response->cache_restore_ok =
          state->environment->prefix_cache().RestoreMatchState(match, *request_context);
      const auto restore_end = std::chrono::steady_clock::now();
      response->cache_restore_ms = ElapsedMs(restore_start, restore_end);
      if (response->cache_restore_ok) {
        if (match.matched_token_count < identity.token_ids.size()) {
          resumed_from_cache = true;
          matched_prefix_tokens = match.matched_token_count;
          prefill_token_ids += matched_prefix_tokens;
          prefill_token_count -= matched_prefix_tokens;
          response->cache_mode = "resume";
        } else if (match.matched_token_count == identity.token_ids.size()) {
          const auto cached_boundary_logits =
              state->environment->prefix_cache().CopyBoundaryLogits(match.node_id);
          if (cached_boundary_logits.has_value() &&
              cached_boundary_logits->size() == state->model->config().vocab_size &&
              AllFinite(*cached_boundary_logits)) {
            boundary_logits_host = *cached_boundary_logits;
            matched_prefix_tokens = match.matched_token_count;
            response->exact_cache_hit = true;
            response->cache_mode = "exact_hit";
          } else {
            response->error = "exact cache hit is missing valid boundary logits";
            return false;
          }
        }
      }
    } else if (!allow_partial_conversation_resume) {
      response->cache_lookup_hit = false;
      response->cache_mode = "cold";
    }
  }

  response->matched_prefix_tokens = matched_prefix_tokens;
  response->prefill_token_count = response->exact_cache_hit ? 0 : prefill_token_count;

  if (!response->exact_cache_hit) {
    auto prompt_logits = nemotron::DeviceTensorFp32::Create(
        {prefill_token_count, state->model->config().vocab_size});
    if (prompt_logits == nullptr || !prompt_logits->valid()) {
      response->error = "prompt logits allocation failed";
      return false;
    }

    const auto prefill_start = std::chrono::steady_clock::now();
    const bool prefill_ok =
        resumed_from_cache
            ? state->model->ContinuePrefill(
                  prefill_token_ids,
                  prefill_token_count,
                  *request_context,
                  prompt_logits.get())
            : state->model->RunPrefill(
                  identity.token_ids.data(),
                  identity.token_ids.size(),
                  *request_context,
                  prompt_logits.get());
    const auto prefill_end = std::chrono::steady_clock::now();
    response->prefill_ms = ElapsedMs(prefill_start, prefill_end);
    if (!prefill_ok) {
      response->error = "prefill failed";
      return false;
    }

    const auto copy_start = std::chrono::steady_clock::now();
    boundary_logits_host = CopyTensorToHost(*prompt_logits);
    const auto copy_end = std::chrono::steady_clock::now();
    response->prompt_logits_copy_ms = ElapsedMs(copy_start, copy_end);
    if (boundary_logits_host.empty() || !AllFinite(boundary_logits_host)) {
      response->error = "prompt logits copy failed or produced invalid values";
      return false;
    }
  }

  const std::size_t logits_row_index = response->exact_cache_hit ? 0 : (prefill_token_count - 1);
  const auto select_start = std::chrono::steady_clock::now();
  const auto first_token = ArgMaxTokenId(
      boundary_logits_host,
      logits_row_index,
      state->model->config().vocab_size);
  const auto select_end = std::chrono::steady_clock::now();
  response->first_token_select_ms = ElapsedMs(select_start, select_end);
  if (!first_token.has_value()) {
    response->error = "failed to select the first decode token";
    return false;
  }

  if (command.max_new_tokens > 0) {
    auto step_logits = nemotron::DeviceTensorFp32::Create({1, state->model->config().vocab_size});
    if (step_logits == nullptr || !step_logits->valid()) {
      response->error = "decode-step logits allocation failed";
      return false;
    }

    std::int32_t token_id = *first_token;
    for (std::size_t step = 0; step < command.max_new_tokens; ++step) {
      if (request_context->sequence_length() >= request_context->config().max_tokens) {
        response->hit_capacity_limit = true;
        break;
      }

      response->generated_token_ids.push_back(token_id);
      const auto step_start = std::chrono::steady_clock::now();
      if (!state->model->ContinueSingleToken(token_id, *request_context, step_logits.get())) {
        response->error =
            "decode continuation failed at step " + std::to_string(step);
        return false;
      }

      boundary_logits_host = CopyTensorToHost(*step_logits);
      if (boundary_logits_host.empty() || !AllFinite(boundary_logits_host)) {
        response->error =
            "decode logits copy failed or produced invalid values at step " +
            std::to_string(step);
        return false;
      }

      if (!ContainsTokenId(command.eos_token_ids, token_id)) {
        const auto next_token = ArgMaxTokenId(
            boundary_logits_host, 0, state->model->config().vocab_size);
        if (!next_token.has_value()) {
          response->error =
              "failed to select the next token at step " + std::to_string(step);
          return false;
        }
        token_id = *next_token;
      } else {
        response->hit_eos = true;
      }

      const auto step_end = std::chrono::steady_clock::now();
      response->decode_step_ms.push_back(ElapsedMs(step_start, step_end));
      if (response->hit_eos) {
        break;
      }
    }
  }

  response->generated_token_count = response->generated_token_ids.size();
  response->sequence_length_after = request_context->sequence_length();
  response->decode_total_ms = std::accumulate(
      response->decode_step_ms.begin(), response->decode_step_ms.end(), 0.0);
  if (!response->decode_step_ms.empty()) {
    response->first_decode_step_ms = response->decode_step_ms.front();
    response->decode_mean_ms =
        response->decode_total_ms / static_cast<double>(response->decode_step_ms.size());
    response->decode_min_ms =
        *std::min_element(response->decode_step_ms.begin(), response->decode_step_ms.end());
    response->decode_max_ms =
        *std::max_element(response->decode_step_ms.begin(), response->decode_step_ms.end());
  }
  if (response->decode_total_ms > 0.0 && response->generated_token_count != 0) {
    response->decode_tokens_per_second =
        (static_cast<double>(response->generated_token_count) * 1000.0) /
        response->decode_total_ms;
  }

  if (state->environment->prefix_cache().enabled()) {
    nemotron::SerializedPromptIdentity committed_identity = identity;
    committed_identity.token_ids.insert(
        committed_identity.token_ids.end(),
        response->generated_token_ids.begin(),
        response->generated_token_ids.end());
    state->environment->prefix_cache().PublishConversationHeadSnapshot(
        response->conversation_id,
        committed_identity,
        *request_context,
        response->conversation_id + "/committed",
        &boundary_logits_host);
  }

  const auto total_end = std::chrono::steady_clock::now();
  response->total_ms = ElapsedMs(total_start, total_end);
  if (response->total_ms > 0.0 && response->generated_token_count != 0) {
    response->total_generated_tokens_per_second =
        (static_cast<double>(response->generated_token_count) * 1000.0) /
        response->total_ms;
  }

  response->attention_counters = nemotron::GetAttentionLayerExecutionCounters();
  response->mamba_counters = nemotron::GetMambaLayerExecutionCounters();
  response->expert_counters = nemotron::GetExpertLayerExecutionCounters();
  response->linear_counters = SnapshotLinearCounters();
  response->expert_staging_counters = SnapshotExpertStagingCounters();
  response->prefix_cache_current_bytes = state->environment->prefix_cache().current_bytes();
  response->prefix_cache_node_count = state->environment->prefix_cache().node_count();
  response->prefix_cache_global_root_count =
      state->environment->prefix_cache().global_root_count();
  response->ok = true;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::ios::sync_with_stdio(false);

  Options options;
  if (!ParseArgs(argc, argv, &options)) {
    std::cerr
        << "Usage: " << argv[0]
        << " --manifest <path> [--max-new-tokens <count>] "
        << "[--target-context-tokens <count>] [--prefix-cache <on|off>] "
        << "[--tenant-namespace <value>] [--serializer-revision <value>]\n";
    return 1;
  }

  if (!std::filesystem::exists(options.manifest_path)) {
    std::cerr << "nemotron_interactive_forward_server: manifest path does not exist\n";
    return 1;
  }

  const auto load_result = nemotron::LoadManifestFromJsonFile(options.manifest_path);
  if (!load_result.ok) {
    std::cerr << "nemotron_interactive_forward_server: manifest parse failed\n";
    return 1;
  }

  const auto config_opt = ConfigForManifest(load_result.manifest);
  if (!config_opt.has_value()) {
    std::cerr << "nemotron_interactive_forward_server: unsupported model_id="
              << load_result.manifest.runtime.model_id << "\n";
    return 1;
  }

  nemotron::SingleTokenForwardConfig config = *config_opt;
  config.max_tokens = std::max(config.max_tokens, options.target_context_tokens);

  auto environment = nemotron::RuntimeEnvironment::BuildFromManifestFile(
      options.manifest_path,
      MakeRuntimeOptions(options.target_context_tokens));
  if (environment == nullptr) {
    std::cerr << "nemotron_interactive_forward_server: runtime environment build failed\n";
    return 1;
  }
  environment->prefix_cache().SetEnabled(options.prefix_cache_enabled);

  auto model = nemotron::SingleTokenForwardModel::Create(*environment, config);
  if (model == nullptr || !model->valid()) {
    std::cerr << "nemotron_interactive_forward_server: forward model build failed\n";
    return 1;
  }

  ServerState state;
  state.options = options;
  state.manifest = load_result.manifest;
  state.environment = std::move(environment);
  state.model = std::move(model);

  std::cout << ReadyJson(state) << "\n";
  std::cout.flush();

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) {
      continue;
    }
    if (line == "EXIT") {
      std::cout << StatusJson("bye", state) << "\n";
      std::cout.flush();
      break;
    }
    if (line == "RESET") {
      state.environment->prefix_cache().Clear();
      ++state.conversation_serial;
      std::cout << StatusJson("reset", state) << "\n";
      std::cout.flush();
      continue;
    }

    const std::vector<std::string> fields = SplitTabFields(line);
    if (fields.empty()) {
      std::cout << ErrorJson("empty command") << "\n";
      std::cout.flush();
      continue;
    }

    if (fields[0] == "SET_PREFIX_CACHE") {
      if (fields.size() != 2) {
        std::cout << ErrorJson("SET_PREFIX_CACHE requires exactly one flag field") << "\n";
        std::cout.flush();
        continue;
      }
      bool enabled = false;
      if (!ParseEnabledFlag(fields[1], &enabled)) {
        std::cout << ErrorJson("invalid prefix-cache flag") << "\n";
        std::cout.flush();
        continue;
      }
      state.environment->prefix_cache().SetEnabled(enabled);
      std::cout << StatusJson("prefix_cache_updated", state) << "\n";
      std::cout.flush();
      continue;
    }

    if (fields[0] == "TURN") {
      if (fields.size() != 4) {
        std::cout << ErrorJson("TURN requires max_new_tokens, eos_csv, and prompt_csv fields") << "\n";
        std::cout.flush();
        continue;
      }

      TurnCommand command;
      if (!ParseNonNegativeSizeT(fields[1], &command.max_new_tokens)) {
        std::cout << ErrorJson("invalid max_new_tokens") << "\n";
        std::cout.flush();
        continue;
      }
      const auto eos_token_ids = ParseInt32Csv(fields[2]);
      const auto prompt_token_ids = ParseInt32Csv(fields[3]);
      if (!eos_token_ids.has_value() || !prompt_token_ids.has_value()) {
        std::cout << ErrorJson("failed to parse TURN csv payload") << "\n";
        std::cout.flush();
        continue;
      }
      command.eos_token_ids = *eos_token_ids;
      command.prompt_token_ids = *prompt_token_ids;

      TurnResponse response;
      if (!ExecuteTurn(&state, command, &response) && response.error.empty()) {
        response.error = "turn execution failed";
      }
      std::cout << TurnResponseJson(response) << "\n";
      std::cout.flush();
      continue;
    }

    std::cout << ErrorJson("unknown command") << "\n";
    std::cout.flush();
  }

  return 0;
}
