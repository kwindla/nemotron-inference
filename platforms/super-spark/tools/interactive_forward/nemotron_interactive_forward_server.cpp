#include "nemotron/device_tensor.h"
#include "nemotron/manifest.h"
#include "nemotron/runtime_environment.h"
#include "nemotron/runtime_stats.h"
#include "nemotron/single_token_forward_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
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
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr const char* kSupportedModelId = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4";

constexpr std::size_t GiB(std::size_t value) {
  return value * 1024ull * 1024ull * 1024ull;
}

struct Options {
  std::filesystem::path manifest_path;
  std::size_t default_max_new_tokens = 2048;
  std::size_t target_context_tokens = 8192;
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
  bool prefix_cache_supported = false;
  std::size_t prompt_token_count = 0;
  std::size_t prefill_token_count = 0;
  std::size_t generated_token_count = 0;
  bool hit_eos = false;
  bool hit_capacity_limit = false;
  std::size_t sequence_length_after = 0;
  std::size_t max_context_tokens = 0;
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
  nemotron::RuntimeExecutionStatsSnapshot runtime_stats;
};

struct ReadyState {
  bool ok = false;
  std::string error;
  std::string model_id;
  double environment_build_ms = 0.0;
  double model_build_ms = 0.0;
};

struct ServerState {
  Options options;
  nemotron::PackedModelManifest manifest;
  std::unique_ptr<nemotron::RuntimeEnvironment> environment;
  std::unique_ptr<nemotron::SingleTokenForwardModel> model;
  std::size_t conversation_serial = 0;
  double environment_build_ms = 0.0;
  double model_build_ms = 0.0;
};

double ElapsedMs(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

bool HasCudaDevice() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

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

void WriteRuntimeStatsJson(
    std::ostream& output,
    const nemotron::RuntimeExecutionStatsSnapshot& stats) {
  output << "{";
  output << "\"dense_native_success\":" << stats.dense_native_success << ",";
  output << "\"dense_reference_fallbacks\":" << stats.dense_reference_fallbacks << ",";
  output << "\"scaled_fp8_native_success\":" << stats.scaled_fp8_native_success << ",";
  output << "\"scaled_fp8_reference_fallbacks\":" << stats.scaled_fp8_reference_fallbacks << ",";
  output << "\"attention_decode_plan_creates\":" << stats.attention_decode_plan_creates << ",";
  output << "\"attention_decode_plan_hits\":" << stats.attention_decode_plan_hits << ",";
  output << "\"expert_selection_metadata_downloads\":"
         << stats.expert_selection_metadata_downloads << ",";
  output << "\"flashinfer_routed_expert_uses\":" << stats.flashinfer_routed_expert_uses << ",";
  output << "\"flashinfer_routed_expert_fallbacks\":" << stats.flashinfer_routed_expert_fallbacks
         << ",";
  output << "\"grouped_routed_expert_fastpath_uses\":"
         << stats.grouped_routed_expert_fastpath_uses << ",";
  output << "\"grouped_routed_expert_fastpath_fallbacks\":"
         << stats.grouped_routed_expert_fastpath_fallbacks << ",";
  output << "\"grouped_routed_expert_prereq_fallbacks\":"
         << stats.grouped_routed_expert_prereq_fallbacks << ",";
  output << "\"forward_graph_captures\":" << stats.forward_graph_captures << ",";
  output << "\"forward_graph_replays\":" << stats.forward_graph_replays << ",";
  output << "\"moe_graph_captures\":" << stats.moe_graph_captures << ",";
  output << "\"moe_graph_replays\":" << stats.moe_graph_replays << ",";
  output << "\"routed_expert_prefetch_layers\":" << stats.routed_expert_prefetch_layers << ",";
  output << "\"routed_expert_prefetch_experts\":" << stats.routed_expert_prefetch_experts;
  output << "}";
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

std::optional<TurnCommand> ParseTurnCommand(
    const std::vector<std::string>& fields,
    std::size_t default_max_new_tokens) {
  if (fields.size() != 4) {
    return std::nullopt;
  }
  TurnCommand command;
  if (!ParseNonNegativeSizeT(fields[1], &command.max_new_tokens)) {
    return std::nullopt;
  }
  if (command.max_new_tokens == 0) {
    command.max_new_tokens = default_max_new_tokens;
  }
  const auto eos_token_ids = ParseInt32Csv(fields[2]);
  const auto prompt_token_ids = ParseInt32Csv(fields[3]);
  if (!eos_token_ids.has_value() || !prompt_token_ids.has_value()) {
    return std::nullopt;
  }
  command.eos_token_ids = std::move(*eos_token_ids);
  command.prompt_token_ids = std::move(*prompt_token_ids);
  return command;
}

bool IsEosToken(std::int32_t token_id, const std::vector<std::int32_t>& eos_token_ids) {
  return std::find(eos_token_ids.begin(), eos_token_ids.end(), token_id) != eos_token_ids.end();
}

std::int32_t Argmax(const std::vector<float>& values) {
  if (values.empty()) {
    return -1;
  }
  std::size_t best_index = 0;
  for (std::size_t i = 1; i < values.size(); ++i) {
    if (values[i] > values[best_index]) {
      best_index = i;
    }
  }
  if (best_index > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    return -1;
  }
  return static_cast<std::int32_t>(best_index);
}

bool CopyLastPromptLogitsRow(
    const nemotron::DeviceTensorFp32& logits,
    std::size_t row_count,
    std::size_t vocab_size,
    std::vector<float>* host_values) {
  if (host_values == nullptr || row_count == 0 || vocab_size == 0) {
    return false;
  }
  host_values->assign(vocab_size, 0.0f);
  const std::size_t row_offset = (row_count - 1) * vocab_size;
  return cudaMemcpy(
             host_values->data(),
             logits.data() + row_offset,
             vocab_size * sizeof(float),
             cudaMemcpyDeviceToHost) == cudaSuccess;
}

nemotron::RuntimeBootstrapOptions MakeBootstrapOptions(const Options& options) {
  nemotron::RuntimeBootstrapOptions bootstrap;
  bootstrap.service_target.total_memory_bytes = GiB(128);
  bootstrap.service_target.weights_bytes = GiB(100);
  bootstrap.service_target.workspace_bytes = GiB(8);
  bootstrap.service_target.graph_bytes = 0;
  bootstrap.service_target.safety_headroom_bytes = GiB(4);
  bootstrap.service_target.target_active_requests = 1;
  bootstrap.service_target.target_context_tokens = options.target_context_tokens;
  bootstrap.use_fp16_mamba_state = false;
  bootstrap.reusable_node_metadata_bytes = 4096;
  bootstrap.verify_manifest_files = false;
  bootstrap.materialize_weight_arena = false;
  return bootstrap;
}

std::optional<nemotron::SingleTokenForwardConfig> ConfigForManifest(
    const nemotron::PackedModelManifest& manifest) {
  if (manifest.runtime.model_id != kSupportedModelId) {
    return std::nullopt;
  }
  return nemotron::KnownNemotron3Super120BA12BConfig();
}

ReadyState InitializeServer(const Options& options, ServerState* state) {
  ReadyState ready;
  if (state == nullptr) {
    ready.error = "internal error: null server state";
    return ready;
  }
  if (!HasCudaDevice()) {
    ready.error = "no CUDA device available";
    return ready;
  }
  const auto manifest_result = nemotron::LoadManifestFromJsonFile(options.manifest_path);
  if (!manifest_result.ok) {
    ready.error = "failed to load manifest: " + options.manifest_path.string();
    return ready;
  }
  auto config = ConfigForManifest(manifest_result.manifest);
  if (!config.has_value()) {
    ready.error = "unsupported manifest model_id: " + manifest_result.manifest.runtime.model_id;
    return ready;
  }
  config->max_tokens = options.target_context_tokens;

  setenv("NEMOTRON_PREFIX_CACHE", "0", 1);
  setenv("NEMOTRON_DISABLE_CUDA_GRAPH_FORWARD", "1", 1);

  const auto environment_start = std::chrono::steady_clock::now();
  auto environment = nemotron::RuntimeEnvironment::BuildFromManifestFile(
      options.manifest_path,
      MakeBootstrapOptions(options));
  const auto environment_end = std::chrono::steady_clock::now();
  if (!environment) {
    ready.error = "runtime environment build failed";
    return ready;
  }

  const auto model_start = std::chrono::steady_clock::now();
  auto model = nemotron::SingleTokenForwardModel::Create(*environment, *config);
  const auto model_end = std::chrono::steady_clock::now();
  if (!model || !model->valid()) {
    ready.error = "forward model build failed";
    return ready;
  }

  state->options = options;
  state->manifest = manifest_result.manifest;
  state->environment = std::move(environment);
  state->model = std::move(model);
  state->conversation_serial = 0;
  state->environment_build_ms = ElapsedMs(environment_start, environment_end);
  state->model_build_ms = ElapsedMs(model_start, model_end);

  ready.ok = true;
  ready.model_id = state->manifest.runtime.model_id;
  ready.environment_build_ms = state->environment_build_ms;
  ready.model_build_ms = state->model_build_ms;
  return ready;
}

std::string ConversationId(const ServerState& state) {
  return "interactive-" + std::to_string(state.conversation_serial);
}

TurnResponse ExecuteTurn(const TurnCommand& command, ServerState& state) {
  TurnResponse response;
  response.conversation_id = ConversationId(state);
  response.prefix_cache_supported = false;
  response.prompt_token_count = command.prompt_token_ids.size();
  response.prefill_token_count = command.prompt_token_ids.size();

  if (!state.model || !state.model->valid()) {
    response.error = "forward model is not available";
    return response;
  }
  if (command.prompt_token_ids.empty()) {
    response.error = "prompt token list is empty";
    return response;
  }

  auto request_context = state.model->CreateRequestContext();
  if (!request_context || !request_context->valid()) {
    response.error = "failed to create request context";
    return response;
  }
  response.max_context_tokens = request_context->config().max_tokens;
  if (command.prompt_token_ids.size() > response.max_context_tokens) {
    response.error =
        "prompt token count exceeds target context tokens: prompt=" +
        std::to_string(command.prompt_token_ids.size()) +
        " target_context_tokens=" + std::to_string(response.max_context_tokens);
    return response;
  }

  auto prompt_logits = nemotron::DeviceTensorFp32::Create(
      {command.prompt_token_ids.size(), state.model->config().vocab_size});
  auto decode_logits = nemotron::DeviceTensorFp32::Create({1, state.model->config().vocab_size});
  if (!prompt_logits || !prompt_logits->valid() || !decode_logits || !decode_logits->valid()) {
    response.error = "failed to allocate logits tensors";
    return response;
  }

  nemotron::ResetRuntimeExecutionStats();
  std::vector<float> logits_host(state.model->config().vocab_size, 0.0f);

  const auto total_start = std::chrono::steady_clock::now();
  cudaDeviceSynchronize();
  const auto prefill_start = std::chrono::steady_clock::now();
  const bool prefill_ok = state.model->RunPrefill(
      command.prompt_token_ids.data(),
      command.prompt_token_ids.size(),
      *request_context,
      prompt_logits.get());
  const auto prefill_sync = cudaDeviceSynchronize();
  const auto prefill_end = std::chrono::steady_clock::now();
  if (!prefill_ok || prefill_sync != cudaSuccess) {
    response.error = "RunPrefill failed";
    return response;
  }
  response.prefill_ms = ElapsedMs(prefill_start, prefill_end);

  if (command.max_new_tokens != 0) {
    const auto copy_start = std::chrono::steady_clock::now();
    if (!CopyLastPromptLogitsRow(
            *prompt_logits,
            command.prompt_token_ids.size(),
            state.model->config().vocab_size,
            &logits_host)) {
      response.error = "failed to copy final prefill logits row";
      return response;
    }
    const auto copy_end = std::chrono::steady_clock::now();
    response.prompt_logits_copy_ms = ElapsedMs(copy_start, copy_end);

    const auto select_start = std::chrono::steady_clock::now();
    const std::int32_t first_token = Argmax(logits_host);
    const auto select_end = std::chrono::steady_clock::now();
    response.first_token_select_ms = ElapsedMs(select_start, select_end);
    if (first_token < 0) {
      response.error = "failed to select first generated token";
      return response;
    }
    response.generated_token_ids.push_back(first_token);
    response.hit_eos = IsEosToken(first_token, command.eos_token_ids);

    std::int32_t current_token = first_token;
    while (!response.hit_eos && response.generated_token_ids.size() < command.max_new_tokens) {
      if (command.prompt_token_ids.size() + response.generated_token_ids.size() >=
          response.max_context_tokens) {
        response.hit_capacity_limit = true;
        break;
      }
      cudaDeviceSynchronize();
      const auto decode_start = std::chrono::steady_clock::now();
      const bool decode_ok =
          state.model->RunDecodeStep(current_token, *request_context, decode_logits.get());
      const auto decode_sync = cudaDeviceSynchronize();
      const auto decode_end = std::chrono::steady_clock::now();
      if (!decode_ok || decode_sync != cudaSuccess) {
        response.error = "RunDecodeStep failed";
        return response;
      }
      const double decode_step_ms = ElapsedMs(decode_start, decode_end);
      response.decode_step_ms.push_back(decode_step_ms);
      if (!decode_logits->CopyToHost(logits_host.data(), logits_host.size())) {
        response.error = "failed to copy decode logits";
        return response;
      }
      current_token = Argmax(logits_host);
      if (current_token < 0) {
        response.error = "failed to select decode token";
        return response;
      }
      response.generated_token_ids.push_back(current_token);
      response.hit_eos = IsEosToken(current_token, command.eos_token_ids);
    }
  }

  const auto total_end = std::chrono::steady_clock::now();
  response.total_ms = ElapsedMs(total_start, total_end);
  response.generated_token_count = response.generated_token_ids.size();
  response.sequence_length_after = command.prompt_token_ids.size() + response.generated_token_count;
  response.runtime_stats = nemotron::GetRuntimeExecutionStatsSnapshot();

  if (!response.decode_step_ms.empty()) {
    response.first_decode_step_ms = response.decode_step_ms.front();
    response.decode_total_ms = 0.0;
    response.decode_min_ms = response.decode_step_ms.front();
    response.decode_max_ms = response.decode_step_ms.front();
    for (const double elapsed_ms : response.decode_step_ms) {
      response.decode_total_ms += elapsed_ms;
      response.decode_min_ms = std::min(response.decode_min_ms, elapsed_ms);
      response.decode_max_ms = std::max(response.decode_max_ms, elapsed_ms);
    }
    response.decode_mean_ms = response.decode_total_ms / response.decode_step_ms.size();
    response.decode_tokens_per_second =
        1000.0 * static_cast<double>(response.decode_step_ms.size()) / response.decode_total_ms;
  }
  if (response.generated_token_count != 0 && response.total_ms > 0.0) {
    response.total_generated_tokens_per_second =
        1000.0 * static_cast<double>(response.generated_token_count) / response.total_ms;
  }

  response.ok = true;
  return response;
}

void EmitReadyJson(const ReadyState& ready, const ServerState& state) {
  std::cout << "{";
  std::cout << "\"ok\":" << (ready.ok ? "true" : "false") << ",";
  if (ready.ok) {
    std::cout << "\"status\":\"ready\",";
    std::cout << "\"model_id\":\"" << EscapeJson(ready.model_id) << "\",";
    std::cout << "\"conversation_id\":\"" << EscapeJson(ConversationId(state)) << "\",";
    std::cout << "\"prefix_cache_supported\":false,";
    std::cout << "\"target_context_tokens\":" << state.options.target_context_tokens << ",";
    std::cout << "\"default_max_new_tokens\":" << state.options.default_max_new_tokens << ",";
    std::cout << "\"environment_build_ms\":" << std::fixed << std::setprecision(6)
              << ready.environment_build_ms << ",";
    std::cout << "\"model_build_ms\":" << std::fixed << std::setprecision(6)
              << ready.model_build_ms;
  } else {
    std::cout << "\"error\":\"" << EscapeJson(ready.error) << "\"";
  }
  std::cout << "}\n" << std::flush;
}

void EmitStatusJson(const ServerState& state) {
  std::cout << "{";
  std::cout << "\"ok\":true,";
  std::cout << "\"status\":\"ready\",";
  std::cout << "\"conversation_id\":\"" << EscapeJson(ConversationId(state)) << "\",";
  std::cout << "\"prefix_cache_supported\":false";
  std::cout << "}\n" << std::flush;
}

void EmitTurnJson(const TurnResponse& response) {
  std::cout << "{";
  std::cout << "\"ok\":" << (response.ok ? "true" : "false");
  if (!response.ok) {
    std::cout << ",\"error\":\"" << EscapeJson(response.error) << "\"";
    std::cout << "}\n" << std::flush;
    return;
  }
  std::cout << ",\"status\":\"turn_complete\"";
  std::cout << ",\"conversation_id\":\"" << EscapeJson(response.conversation_id) << "\"";
  std::cout << ",\"prefix_cache_supported\":false";
  std::cout << ",\"prompt_token_count\":" << response.prompt_token_count;
  std::cout << ",\"prefill_token_count\":" << response.prefill_token_count;
  std::cout << ",\"generated_token_count\":" << response.generated_token_count;
  std::cout << ",\"hit_eos\":" << (response.hit_eos ? "true" : "false");
  std::cout << ",\"hit_capacity_limit\":" << (response.hit_capacity_limit ? "true" : "false");
  std::cout << ",\"sequence_length_after\":" << response.sequence_length_after;
  std::cout << ",\"max_context_tokens\":" << response.max_context_tokens;
  std::cout << ",\"prefill_ms\":" << std::fixed << std::setprecision(6) << response.prefill_ms;
  std::cout << ",\"prompt_logits_copy_ms\":" << std::fixed << std::setprecision(6)
            << response.prompt_logits_copy_ms;
  std::cout << ",\"first_token_select_ms\":" << std::fixed << std::setprecision(6)
            << response.first_token_select_ms;
  std::cout << ",\"decode_total_ms\":" << std::fixed << std::setprecision(6)
            << response.decode_total_ms;
  std::cout << ",\"first_decode_step_ms\":" << std::fixed << std::setprecision(6)
            << response.first_decode_step_ms;
  std::cout << ",\"decode_mean_ms\":" << std::fixed << std::setprecision(6)
            << response.decode_mean_ms;
  std::cout << ",\"decode_min_ms\":" << std::fixed << std::setprecision(6)
            << response.decode_min_ms;
  std::cout << ",\"decode_max_ms\":" << std::fixed << std::setprecision(6)
            << response.decode_max_ms;
  std::cout << ",\"total_ms\":" << std::fixed << std::setprecision(6) << response.total_ms;
  std::cout << ",\"decode_tokens_per_second\":" << std::fixed << std::setprecision(6)
            << response.decode_tokens_per_second;
  std::cout << ",\"total_generated_tokens_per_second\":" << std::fixed << std::setprecision(6)
            << response.total_generated_tokens_per_second;
  std::cout << ",\"decode_step_ms\":";
  WriteJsonDoubleArray(std::cout, response.decode_step_ms);
  std::cout << ",\"generated_token_ids\":";
  WriteJsonIntArray(std::cout, response.generated_token_ids);
  std::cout << ",\"runtime_stats\":";
  WriteRuntimeStatsJson(std::cout, response.runtime_stats);
  std::cout << "}\n" << std::flush;
}

void EmitErrorJson(const std::string& error) {
  std::cout << "{\"ok\":false,\"error\":\"" << EscapeJson(error) << "\"}\n" << std::flush;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseArgs(argc, argv, &options)) {
    EmitErrorJson(
        "usage: nemotron_interactive_forward_server --manifest <path> "
        "[--max-new-tokens N] [--target-context-tokens N]");
    return 2;
  }

  ServerState state;
  const ReadyState ready = InitializeServer(options, &state);
  EmitReadyJson(ready, state);
  if (!ready.ok) {
    return 1;
  }

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line == "EXIT") {
      std::cout << "{\"ok\":true,\"status\":\"bye\"}\n" << std::flush;
      return 0;
    }
    if (line == "RESET") {
      ++state.conversation_serial;
      EmitStatusJson(state);
      continue;
    }

    const std::vector<std::string> fields = SplitTabFields(line);
    if (!fields.empty() && fields.front() == "TURN") {
      const auto command = ParseTurnCommand(fields, state.options.default_max_new_tokens);
      if (!command.has_value()) {
        EmitErrorJson("invalid TURN command");
        continue;
      }
      EmitTurnJson(ExecuteTurn(*command, state));
      continue;
    }

    EmitErrorJson("unknown command");
  }

  return 0;
}
