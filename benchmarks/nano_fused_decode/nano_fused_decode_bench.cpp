#include "nemotron/cudnn_handle.h"
#include "nemotron/device_argmax.h"
#include "nemotron/device_tensor.h"
#include "nemotron/expert_staging_counters.h"
#include "nemotron/linear_op_counters.h"
#include "nemotron/linear_op_trace.h"
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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char* kNanoModelId = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4";
constexpr std::size_t kPromptTokenCount = 16;
constexpr std::size_t kDefaultDecodeTokenCount = 16;

enum class BenchmarkMode {
  kPhased,
  kSteadyState,
  kCachedHead,
  kProfileReady,
};

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

struct BenchmarkOptions {
  std::optional<std::filesystem::path> manifest_path;
  std::optional<std::filesystem::path> json_output_path;
  BenchmarkMode mode = BenchmarkMode::kPhased;
  std::size_t decode_token_count = kDefaultDecodeTokenCount;
  std::size_t warmup_iterations = 1;
  std::size_t hot_iterations = 3;
  bool strict_linear = false;
};

struct EnvironmentInfo {
  std::string device_name;
  int runtime_version = 0;
  int driver_version = 0;
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
  std::size_t total_global_mem_bytes = 0;
};

struct BenchmarkResult {
  std::string mode = "phased";
  std::string model_id;
  std::string manifest_path;
  std::size_t prompt_token_count = 0;
  std::size_t max_new_tokens = 0;
  std::size_t warmup_iterations = 0;
  std::size_t hot_iterations = 0;
  std::size_t decode_token_count = 0;
  std::size_t generated_token_count = 0;
  bool fused_mamba_enabled = true;
  bool fused_moe_enabled = true;
  bool linear_device_fastpath_enabled = false;
  bool device_token_select_enabled = true;
  bool cudnn_fe_available = false;
  double cold_total_ms = 0.0;
  double cold_prefill_ms = 0.0;
  double cold_first_token_ms = 0.0;
  double cold_steady_state_mean_ms = 0.0;
  double cold_steady_state_min_ms = 0.0;
  double cold_steady_state_max_ms = 0.0;
  double hot_total_mean_ms = 0.0;
  double hot_total_min_ms = 0.0;
  double hot_total_max_ms = 0.0;
  double hot_prefill_mean_ms = 0.0;
  double hot_first_token_mean_ms = 0.0;
  double hot_steady_state_mean_ms = 0.0;
  double hot_steady_state_min_ms = 0.0;
  double hot_steady_state_max_ms = 0.0;
  double steady_state_prefill_ms = 0.0;
  double steady_state_generated_tokens_per_second = 0.0;
  double full_decode_tokens_per_second = 0.0;
  std::vector<double> steady_state_step_ms;
  std::vector<std::int32_t> generated_token_ids;
};

struct LinearFallbackObservation {
  const char* counter_name = "";
  std::uint64_t count = 0;
};

bool CheckCuda(cudaError_t status, const char* message) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << "nano_fused_decode_bench: " << message << ": "
            << cudaGetErrorString(status) << "\n";
  return false;
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

const char* EnabledStatus(bool enabled) {
  return enabled ? "enabled" : "disabled";
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

void PrintUnexpectedLinearFallbacks(
    std::ostream& stream,
    const std::vector<LinearFallbackObservation>& fallbacks) {
  if (fallbacks.empty()) {
    return;
  }

  stream << "nano_fused_decode_bench: unexpected linear reference fallbacks while "
         << "NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH=1\n";
  for (const LinearFallbackObservation& fallback : fallbacks) {
    stream << "nano_fused_decode_bench: operator=" << fallback.counter_name
           << " reference_fallback_count=" << fallback.count << "\n";
  }
}

struct DeviceTokenBuffer {
  ~DeviceTokenBuffer() {
    if (data != nullptr) {
      cudaFree(data);
    }
  }

  bool Allocate() {
    return CheckCuda(
        cudaMalloc(reinterpret_cast<void**>(&data), sizeof(std::int32_t)),
        "cudaMalloc device token buffer failed");
  }

  std::int32_t* data = nullptr;
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

nemotron::RuntimeBootstrapOptions make_options() {
  nemotron::RuntimeBootstrapOptions options;
  options.service_target.total_memory_bytes = GiB(128);
  options.service_target.weights_bytes = GiB(100);
  options.service_target.workspace_bytes = GiB(8);
  options.service_target.graph_bytes = GiB(4);
  options.service_target.safety_headroom_bytes = GiB(4);
  options.service_target.target_active_requests = 1;
  options.service_target.target_context_tokens = kPromptTokenCount + kDefaultDecodeTokenCount;
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

std::optional<std::int32_t> SelectContinuationTokenId(
    const nemotron::DeviceTensorFp32& logits_row,
    bool device_token_select_enabled,
    std::int32_t* device_token_id) {
  if (device_token_select_enabled) {
    if (device_token_id == nullptr) {
      std::cerr << "nano_fused_decode_bench: missing device token buffer\n";
      return std::nullopt;
    }
    if (!nemotron::DeviceArgmax(logits_row, device_token_id)) {
      std::cerr << "nano_fused_decode_bench: DeviceArgmax failed\n";
      return std::nullopt;
    }
    std::int32_t host_token_id = -1;
    if (!CheckCuda(
            cudaMemcpy(
                &host_token_id,
                device_token_id,
                sizeof(host_token_id),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy device token id failed")) {
      return std::nullopt;
    }
    return host_token_id;
  }

  const std::vector<float> host_logits = CopyTensorToHost(logits_row);
  if (host_logits.empty() && logits_row.numel() != 0) {
    std::cerr << "nano_fused_decode_bench: failed to copy logits to host\n";
    return std::nullopt;
  }
  return ArgMaxTokenId(host_logits);
}

std::string EscapeJson(const std::string& text) {
  std::string escaped;
  escaped.reserve(text.size() + 8);
  for (char ch : text) {
    switch (ch) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      default:
        escaped.push_back(ch);
        break;
    }
  }
  return escaped;
}

const char* BenchmarkModeCliName(BenchmarkMode mode) {
  switch (mode) {
    case BenchmarkMode::kSteadyState:
      return "steady-state";
    case BenchmarkMode::kCachedHead:
      return "cached-head";
    case BenchmarkMode::kProfileReady:
      return "profile-ready";
    case BenchmarkMode::kPhased:
      return "phased";
  }
  return "phased";
}

const char* BenchmarkModeJsonName(BenchmarkMode mode) {
  switch (mode) {
    case BenchmarkMode::kSteadyState:
      return "steady_state";
    case BenchmarkMode::kCachedHead:
      return "cached_head";
    case BenchmarkMode::kProfileReady:
      return "profile_ready";
    case BenchmarkMode::kPhased:
      return "phased";
  }
  return "phased";
}

bool ParseBenchmarkMode(const std::string& value, BenchmarkMode* mode) {
  if (mode == nullptr) {
    return false;
  }
  if (value == "phased") {
    *mode = BenchmarkMode::kPhased;
    return true;
  }
  if (value == "steady-state" || value == "steady_state") {
    *mode = BenchmarkMode::kSteadyState;
    return true;
  }
  if (value == "cached-head" || value == "cached_head") {
    *mode = BenchmarkMode::kCachedHead;
    return true;
  }
  if (value == "profile-ready" || value == "profile_ready") {
    *mode = BenchmarkMode::kProfileReady;
    return true;
  }
  return false;
}

bool ParsePositiveSizeT(const std::string& value, std::size_t* parsed_value) {
  if (parsed_value == nullptr) {
    return false;
  }
  try {
    const auto parsed = std::stoull(value);
    if (parsed == 0) {
      return false;
    }
    *parsed_value = parsed;
    return true;
  } catch (...) {
    return false;
  }
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

bool TryParseInlineOptionValue(
    const std::string& arg,
    const char* prefix,
    std::string* value) {
  const std::string prefix_string(prefix);
  if (arg.rfind(prefix_string, 0) != 0) {
    return false;
  }
  if (value != nullptr) {
    *value = arg.substr(prefix_string.size());
  }
  return true;
}

void WriteJsonDoubleArray(std::ostream& output, const std::vector<double>& values) {
  output << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      output << ", ";
    }
    output << std::fixed << std::setprecision(6) << values[i];
  }
  output << "]";
}

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n"
      << "  --manifest <path>           Manifest path. Defaults to NEMOTRON_FORWARD_MANIFEST.\n"
      << "  --mode <phased|steady-state|cached-head|profile-ready>\n"
      << "                             Benchmark mode. Default: phased\n"
      << "  --strict-linear            Fail if linear fastpath was requested but reference fallbacks occurred.\n"
      << "  --warmup <count>            Warmup iterations. Default: 1\n"
      << "  --iterations <count>        Timed hot iterations. Default: 3\n"
      << "  --decode-tokens <count>     Timed ContinueSingleToken steps for --mode=steady-state\n"
      << "                             and --mode=cached-head. Default: 16\n"
      << "  --json-output <path>        Write JSON output.\n";
}

bool ParseArgs(int argc, char** argv, BenchmarkOptions* options) {
  if (options == nullptr) {
    return false;
  }
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    std::string inline_value;
    if (arg == "--manifest") {
      if (i + 1 >= argc) {
        return false;
      }
      options->manifest_path = std::filesystem::path(argv[++i]);
    } else if (TryParseInlineOptionValue(arg, "--mode=", &inline_value)) {
      if (!ParseBenchmarkMode(inline_value, &options->mode)) {
        return false;
      }
    } else if (arg == "--mode") {
      if (i + 1 >= argc || !ParseBenchmarkMode(argv[++i], &options->mode)) {
        return false;
      }
    } else if (arg == "--strict-linear") {
      options->strict_linear = true;
    } else if (arg == "--warmup") {
      if (i + 1 >= argc) {
        return false;
      }
      if (!ParseNonNegativeSizeT(argv[++i], &options->warmup_iterations)) {
        return false;
      }
    } else if (arg == "--iterations") {
      if (i + 1 >= argc) {
        return false;
      }
      if (!ParseNonNegativeSizeT(argv[++i], &options->hot_iterations)) {
        return false;
      }
    } else if (TryParseInlineOptionValue(arg, "--decode-tokens=", &inline_value)) {
      if (!ParsePositiveSizeT(inline_value, &options->decode_token_count)) {
        return false;
      }
    } else if (arg == "--decode-tokens") {
      if (i + 1 >= argc || !ParsePositiveSizeT(argv[++i], &options->decode_token_count)) {
        return false;
      }
    } else if (arg == "--json-output") {
      if (i + 1 >= argc) {
        return false;
      }
      options->json_output_path = std::filesystem::path(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      return false;
    }
  }
  return true;
}

std::optional<EnvironmentInfo> QueryEnvironmentInfo() {
  EnvironmentInfo info;
  int device_index = 0;
  cudaDeviceProp prop{};
  if (!CheckCuda(cudaGetDevice(&device_index), "cudaGetDevice failed") ||
      !CheckCuda(cudaGetDeviceProperties(&prop, device_index), "cudaGetDeviceProperties failed") ||
      !CheckCuda(cudaRuntimeGetVersion(&info.runtime_version), "cudaRuntimeGetVersion failed") ||
      !CheckCuda(cudaDriverGetVersion(&info.driver_version), "cudaDriverGetVersion failed")) {
    return std::nullopt;
  }
  info.device_name = prop.name;
  info.compute_capability_major = prop.major;
  info.compute_capability_minor = prop.minor;
  info.total_global_mem_bytes = static_cast<std::size_t>(prop.totalGlobalMem);
  return info;
}

struct TimedPhasedRun {
  double total_ms = 0.0;
  double prefill_ms = 0.0;
  double first_token_ms = 0.0;
  std::vector<double> steady_state_step_ms;
  std::vector<std::int32_t> generated_token_ids;
};

struct TimedSteadyStateRun {
  double prefill_ms = 0.0;
  std::vector<double> steady_state_step_ms;
  std::vector<std::int32_t> generated_token_ids;
};

struct TimedProfileReadyRun {
  double prefill_ms = 0.0;
  double decode_step_ms = 0.0;
  std::vector<std::int32_t> generated_token_ids;
};

std::optional<TimedPhasedRun> RunTimedPhasedDecode(
    nemotron::SingleTokenForwardModel& model,
    const std::vector<std::int32_t>& prompt_token_ids,
    const std::string& run_label,
    bool device_token_select_enabled,
    std::int32_t* device_token_id) {
  auto request_context = model.CreateRequestContext();
  if (request_context == nullptr || !request_context->valid()) {
    std::cerr << "nano_fused_decode_bench: failed to create request context\n";
    return std::nullopt;
  }
  auto prompt_logits =
      nemotron::DeviceTensorFp32::Create({prompt_token_ids.size(), model.config().vocab_size});
  if (prompt_logits == nullptr || !prompt_logits->valid()) {
    std::cerr << "nano_fused_decode_bench: failed to allocate prompt logits\n";
    return std::nullopt;
  }
  auto step_logits = nemotron::DeviceTensorFp32::Create({1, model.config().vocab_size});
  if (step_logits == nullptr || !step_logits->valid()) {
    std::cerr << "nano_fused_decode_bench: failed to allocate step logits\n";
    return std::nullopt;
  }

  TimedPhasedRun run;
  run.generated_token_ids.reserve(kDefaultDecodeTokenCount);
  run.steady_state_step_ms.reserve(
      kDefaultDecodeTokenCount > 0 ? (kDefaultDecodeTokenCount - 1) : 0);

  std::cout << "nano_fused_decode_bench: " << run_label
            << " prefill start prompt_tokens=" << prompt_token_ids.size() << "\n";
  std::cout.flush();
  if (!CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize before prefill failed")) {
    return std::nullopt;
  }
  const auto total_start = std::chrono::steady_clock::now();
  const auto prefill_start = total_start;
  if (!model.RunPrefill(
          prompt_token_ids.data(),
          prompt_token_ids.size(),
          *request_context,
          prompt_logits.get())) {
    std::cerr << "nano_fused_decode_bench: RunPrefill failed\n";
    return std::nullopt;
  }
  if (!CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after prefill failed")) {
    return std::nullopt;
  }
  const auto prefill_end = std::chrono::steady_clock::now();
  run.prefill_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();

  const std::vector<float> prompt_host = CopyTensorToHost(*prompt_logits);
  const std::vector<float> prompt_final_row =
      SliceRow(prompt_host, prompt_token_ids.size() - 1, model.config().vocab_size);
  const auto first_token = ArgMaxTokenId(prompt_final_row);
  if (!first_token.has_value()) {
    std::cerr << "nano_fused_decode_bench: failed to select first token\n";
    return std::nullopt;
  }
  const auto first_token_end = std::chrono::steady_clock::now();
  run.first_token_ms =
      std::chrono::duration<double, std::milli>(first_token_end - total_start).count();
  run.generated_token_ids.push_back(*first_token);
  std::cout << "nano_fused_decode_bench: " << run_label
            << " prefill complete elapsed_ms=" << run.prefill_ms
            << " first_token_ms=" << run.first_token_ms
            << " token=" << *first_token << "\n";
  std::cout.flush();

  std::int32_t token_id = *first_token;
  for (std::size_t token_index = 1; token_index < kDefaultDecodeTokenCount; ++token_index) {
    std::cout << "nano_fused_decode_bench: " << run_label
              << " steady step " << token_index << "/" << (kDefaultDecodeTokenCount - 1)
              << " start consume_token=" << token_id << "\n";
    std::cout.flush();
    if (!CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize before decode step failed")) {
      return std::nullopt;
    }
    const auto step_start = std::chrono::steady_clock::now();
    if (!model.ContinueSingleToken(token_id, *request_context, step_logits.get())) {
      std::cerr << "nano_fused_decode_bench: ContinueSingleToken failed at step "
                << token_index << "\n";
      return std::nullopt;
    }
    if (!device_token_select_enabled &&
        !CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after decode step failed")) {
      return std::nullopt;
    }
    const auto next_token =
        SelectContinuationTokenId(*step_logits, device_token_select_enabled, device_token_id);
    if (!next_token.has_value()) {
      std::cerr << "nano_fused_decode_bench: failed to select token at step "
                << token_index << "\n";
      return std::nullopt;
    }
    const auto step_end = std::chrono::steady_clock::now();
    const double step_ms =
        std::chrono::duration<double, std::milli>(step_end - step_start).count();
    run.steady_state_step_ms.push_back(step_ms);
    run.generated_token_ids.push_back(*next_token);
    std::cout << "nano_fused_decode_bench: " << run_label
              << " steady step " << token_index << "/" << (kDefaultDecodeTokenCount - 1)
              << " elapsed_ms=" << step_ms
              << " next_token=" << *next_token << "\n";
    std::cout.flush();
    token_id = *next_token;
  }

  const auto total_end = std::chrono::steady_clock::now();
  run.total_ms =
      std::chrono::duration<double, std::milli>(total_end - total_start).count();
  return run;
}

std::optional<TimedSteadyStateRun> RunTimedSteadyStateDecode(
    nemotron::SingleTokenForwardModel& model,
    const std::vector<std::int32_t>& prompt_token_ids,
    std::size_t decode_token_count,
    const std::string& run_label,
    bool device_token_select_enabled,
    std::int32_t* device_token_id) {
  auto request_context = model.CreateRequestContext();
  if (request_context == nullptr || !request_context->valid()) {
    std::cerr << "nano_fused_decode_bench: failed to create request context\n";
    return std::nullopt;
  }
  auto prompt_logits =
      nemotron::DeviceTensorFp32::Create({prompt_token_ids.size(), model.config().vocab_size});
  if (prompt_logits == nullptr || !prompt_logits->valid()) {
    std::cerr << "nano_fused_decode_bench: failed to allocate prompt logits\n";
    return std::nullopt;
  }
  auto step_logits = nemotron::DeviceTensorFp32::Create({1, model.config().vocab_size});
  if (step_logits == nullptr || !step_logits->valid()) {
    std::cerr << "nano_fused_decode_bench: failed to allocate step logits\n";
    return std::nullopt;
  }

  TimedSteadyStateRun run;
  run.generated_token_ids.reserve(decode_token_count);
  run.steady_state_step_ms.reserve(decode_token_count);

  std::cout << "nano_fused_decode_bench: " << run_label
            << " prefill start prompt_tokens=" << prompt_token_ids.size()
            << " decode_tokens=" << decode_token_count << "\n";
  std::cout.flush();
  if (!CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize before prefill failed")) {
    return std::nullopt;
  }
  const auto prefill_start = std::chrono::steady_clock::now();
  if (!model.RunPrefill(
          prompt_token_ids.data(),
          prompt_token_ids.size(),
          *request_context,
          prompt_logits.get())) {
    std::cerr << "nano_fused_decode_bench: RunPrefill failed\n";
    return std::nullopt;
  }
  if (!CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after prefill failed")) {
    return std::nullopt;
  }
  const auto prefill_end = std::chrono::steady_clock::now();
  run.prefill_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();

  const std::vector<float> prompt_host = CopyTensorToHost(*prompt_logits);
  const std::vector<float> prompt_final_row =
      SliceRow(prompt_host, prompt_token_ids.size() - 1, model.config().vocab_size);
  const auto initial_token = ArgMaxTokenId(prompt_final_row);
  if (!initial_token.has_value()) {
    std::cerr << "nano_fused_decode_bench: failed to select steady-state seed token\n";
    return std::nullopt;
  }

  std::cout << "nano_fused_decode_bench: " << run_label
            << " prefill complete elapsed_ms=" << run.prefill_ms
            << " seed_token=" << *initial_token << "\n";
  std::cout.flush();

  std::int32_t token_id = *initial_token;
  for (std::size_t token_index = 0; token_index < decode_token_count; ++token_index) {
    const std::size_t step_number = token_index + 1;
    std::cout << "nano_fused_decode_bench: " << run_label
              << " decode step " << step_number << "/" << decode_token_count
              << " start consume_token=" << token_id << "\n";
    std::cout.flush();
    if (!CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize before decode step failed")) {
      return std::nullopt;
    }
    const auto step_start = std::chrono::steady_clock::now();
    if (!model.ContinueSingleToken(token_id, *request_context, step_logits.get())) {
      std::cerr << "nano_fused_decode_bench: ContinueSingleToken failed at step "
                << step_number << "\n";
      return std::nullopt;
    }
    if (!device_token_select_enabled &&
        !CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after decode step failed")) {
      return std::nullopt;
    }
    const auto next_token =
        SelectContinuationTokenId(*step_logits, device_token_select_enabled, device_token_id);
    if (!next_token.has_value()) {
      std::cerr << "nano_fused_decode_bench: failed to select token at step "
                << step_number << "\n";
      return std::nullopt;
    }
    const auto step_end = std::chrono::steady_clock::now();
    const double step_ms =
        std::chrono::duration<double, std::milli>(step_end - step_start).count();
    run.generated_token_ids.push_back(token_id);
    run.steady_state_step_ms.push_back(step_ms);
    std::cout << "nano_fused_decode_bench: " << run_label
              << " decode step " << step_number << "/" << decode_token_count
              << " elapsed_ms=" << step_ms
              << " next_token=" << *next_token << "\n";
    std::cout.flush();
    token_id = *next_token;
  }

  return run;
}

std::optional<TimedProfileReadyRun> RunProfileReadyDecode(
    nemotron::SingleTokenForwardModel& model,
    const std::vector<std::int32_t>& prompt_token_ids,
    bool device_token_select_enabled,
    std::int32_t* device_token_id) {
  auto request_context = model.CreateRequestContext();
  if (request_context == nullptr || !request_context->valid()) {
    std::cerr << "nano_fused_decode_bench: failed to create request context\n";
    return std::nullopt;
  }
  auto prompt_logits =
      nemotron::DeviceTensorFp32::Create({prompt_token_ids.size(), model.config().vocab_size});
  if (prompt_logits == nullptr || !prompt_logits->valid()) {
    std::cerr << "nano_fused_decode_bench: failed to allocate prompt logits\n";
    return std::nullopt;
  }
  auto step_logits = nemotron::DeviceTensorFp32::Create({1, model.config().vocab_size});
  if (step_logits == nullptr || !step_logits->valid()) {
    std::cerr << "nano_fused_decode_bench: failed to allocate step logits\n";
    return std::nullopt;
  }

  TimedProfileReadyRun run;
  run.generated_token_ids.reserve(2);

  if (!CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize before prefill failed")) {
    return std::nullopt;
  }
  std::cout << "===PREFILL_START===\n";
  std::cout.flush();
  const auto prefill_start = std::chrono::steady_clock::now();
  if (!model.RunPrefill(
          prompt_token_ids.data(),
          prompt_token_ids.size(),
          *request_context,
          prompt_logits.get())) {
    std::cerr << "nano_fused_decode_bench: RunPrefill failed\n";
    return std::nullopt;
  }
  if (!CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after prefill failed")) {
    return std::nullopt;
  }
  const auto prefill_end = std::chrono::steady_clock::now();
  run.prefill_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
  std::cout << "===PREFILL_END===\n";
  std::cout.flush();

  const std::vector<float> prompt_host = CopyTensorToHost(*prompt_logits);
  const std::vector<float> prompt_final_row =
      SliceRow(prompt_host, prompt_token_ids.size() - 1, model.config().vocab_size);
  const auto initial_token = ArgMaxTokenId(prompt_final_row);
  if (!initial_token.has_value()) {
    std::cerr << "nano_fused_decode_bench: failed to select profile-ready seed token\n";
    return std::nullopt;
  }
  run.generated_token_ids.push_back(*initial_token);

  if (!CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize before decode step failed")) {
    return std::nullopt;
  }
  std::cout << "===DECODE_START===\n";
  std::cout.flush();
  const auto decode_start = std::chrono::steady_clock::now();
  if (!model.ContinueSingleToken(*initial_token, *request_context, step_logits.get())) {
    std::cerr << "nano_fused_decode_bench: ContinueSingleToken failed at step 1\n";
    return std::nullopt;
  }
  const auto next_token =
      SelectContinuationTokenId(*step_logits, device_token_select_enabled, device_token_id);
  if (!next_token.has_value()) {
    std::cerr << "nano_fused_decode_bench: failed to select token at step 1\n";
    return std::nullopt;
  }
  if (!CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after decode step failed")) {
    return std::nullopt;
  }
  const auto decode_end = std::chrono::steady_clock::now();
  run.decode_step_ms =
      std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
  std::cout << "===DECODE_END===\n";
  std::cout.flush();
  run.generated_token_ids.push_back(*next_token);

  return run;
}

double Mean(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double MinOrZero(const std::vector<double>& values) {
  return values.empty() ? 0.0 : *std::min_element(values.begin(), values.end());
}

double MaxOrZero(const std::vector<double>& values) {
  return values.empty() ? 0.0 : *std::max_element(values.begin(), values.end());
}

bool WriteJson(
    const std::filesystem::path& output_path,
    const EnvironmentInfo& environment,
    const BenchmarkResult& result,
    const std::vector<std::int32_t>& prompt_token_ids) {
  if (!output_path.parent_path().empty()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  std::ofstream output(output_path);
  if (!output) {
    std::cerr << "nano_fused_decode_bench: failed to open JSON output path\n";
    return false;
  }

  output << "{\n";
  output << "  \"environment\": {\n";
  output << "    \"device_name\": \"" << EscapeJson(environment.device_name) << "\",\n";
  output << "    \"runtime_version\": " << environment.runtime_version << ",\n";
  output << "    \"driver_version\": " << environment.driver_version << ",\n";
  output << "    \"compute_capability\": \"" << environment.compute_capability_major
         << "." << environment.compute_capability_minor << "\",\n";
  output << "    \"total_global_mem_bytes\": " << environment.total_global_mem_bytes << "\n";
  output << "  },\n";
  output << "  \"benchmark\": {\n";
  output << "    \"name\": \"nano_fused_decode_16tok\",\n";
  output << "    \"mode\": \"" << EscapeJson(result.mode) << "\",\n";
  output << "    \"model_id\": \"" << EscapeJson(result.model_id) << "\",\n";
  output << "    \"manifest_path\": \"" << EscapeJson(result.manifest_path) << "\",\n";
  output << "    \"prompt_token_count\": " << result.prompt_token_count << ",\n";
  output << "    \"max_new_tokens\": " << result.max_new_tokens << ",\n";
  output << "    \"warmup_iterations\": " << result.warmup_iterations << ",\n";
  output << "    \"hot_iterations\": " << result.hot_iterations << ",\n";
  output << "    \"decode_token_count\": " << result.decode_token_count << ",\n";
  output << "    \"generated_token_count\": " << result.generated_token_count << ",\n";
  output << "    \"fused_mamba_enabled\": " << (result.fused_mamba_enabled ? "true" : "false") << ",\n";
  output << "    \"fused_moe_enabled\": " << (result.fused_moe_enabled ? "true" : "false") << ",\n";
  output << "    \"linear_device_fastpath_enabled\": "
         << (result.linear_device_fastpath_enabled ? "true" : "false") << ",\n";
  output << "    \"device_token_select_enabled\": "
         << (result.device_token_select_enabled ? "true" : "false") << ",\n";
  output << "    \"cold_total_ms\": " << std::fixed << std::setprecision(6) << result.cold_total_ms << ",\n";
  output << "    \"cold_prefill_ms\": " << result.cold_prefill_ms << ",\n";
  output << "    \"cold_first_token_ms\": " << result.cold_first_token_ms << ",\n";
  output << "    \"cold_steady_state_mean_ms\": " << result.cold_steady_state_mean_ms << ",\n";
  output << "    \"cold_steady_state_min_ms\": " << result.cold_steady_state_min_ms << ",\n";
  output << "    \"cold_steady_state_max_ms\": " << result.cold_steady_state_max_ms << ",\n";
  output << "    \"hot_total_mean_ms\": " << result.hot_total_mean_ms << ",\n";
  output << "    \"hot_total_min_ms\": " << result.hot_total_min_ms << ",\n";
  output << "    \"hot_total_max_ms\": " << result.hot_total_max_ms << ",\n";
  output << "    \"hot_prefill_mean_ms\": " << result.hot_prefill_mean_ms << ",\n";
  output << "    \"hot_first_token_mean_ms\": " << result.hot_first_token_mean_ms << ",\n";
  output << "    \"hot_steady_state_mean_ms\": " << result.hot_steady_state_mean_ms << ",\n";
  output << "    \"hot_steady_state_min_ms\": " << result.hot_steady_state_min_ms << ",\n";
  output << "    \"hot_steady_state_max_ms\": " << result.hot_steady_state_max_ms << ",\n";
  output << "    \"steady_state_prefill_ms\": " << result.steady_state_prefill_ms << ",\n";
  output << "    \"steady_state_generated_tokens_per_second\": "
         << result.steady_state_generated_tokens_per_second << ",\n";
  output << "    \"full_decode_tokens_per_second\": "
         << result.full_decode_tokens_per_second << "\n";
  output << "  },\n";
  output << "  \"prompt_token_ids\": [";
  for (std::size_t i = 0; i < prompt_token_ids.size(); ++i) {
    if (i != 0) {
      output << ", ";
    }
    output << prompt_token_ids[i];
  }
  output << "],\n";
  output << "  \"generated_token_ids\": [";
  for (std::size_t i = 0; i < result.generated_token_ids.size(); ++i) {
    if (i != 0) {
      output << ", ";
    }
    output << result.generated_token_ids[i];
  }
  output << "],\n";
  output << "  \"steady_state_step_ms\": ";
  WriteJsonDoubleArray(output, result.steady_state_step_ms);
  output << ",\n";
  output << "  \"cudnn_fe_available\": "
         << (result.cudnn_fe_available ? "true" : "false") << ",\n";
  const auto& counters = nemotron::GetLinearOpCounters();
  output << "  \"linear_op_counters\": {\n";
  output << "    \"dense_fastpath_plan_success\": "
         << counters.dense_fastpath_plan_success.load(std::memory_order_relaxed) << ",\n";
  output << "    \"dense_fastpath_plan_fail\": "
         << counters.dense_fastpath_plan_fail.load(std::memory_order_relaxed) << ",\n";
  output << "    \"dense_fastpath_execute\": "
         << counters.dense_fastpath_execute.load(std::memory_order_relaxed) << ",\n";
  output << "    \"dense_fastpath_execute_fail\": "
         << counters.dense_fastpath_execute_fail.load(std::memory_order_relaxed) << ",\n";
  output << "    \"dense_reference_fallback\": "
         << counters.dense_reference_fallback.load(std::memory_order_relaxed) << ",\n";
  output << "    \"nvfp4_fastpath_plan_success\": "
         << counters.nvfp4_fastpath_plan_success.load(std::memory_order_relaxed) << ",\n";
  output << "    \"nvfp4_fastpath_plan_fail\": "
         << counters.nvfp4_fastpath_plan_fail.load(std::memory_order_relaxed) << ",\n";
  output << "    \"nvfp4_fastpath_execute\": "
         << counters.nvfp4_fastpath_execute.load(std::memory_order_relaxed) << ",\n";
  output << "    \"nvfp4_fastpath_execute_fail\": "
         << counters.nvfp4_fastpath_execute_fail.load(std::memory_order_relaxed) << ",\n";
  output << "    \"nvfp4_reference_fallback\": "
         << counters.nvfp4_reference_fallback.load(std::memory_order_relaxed) << ",\n";
  output << "    \"scaled_fp8_fastpath_execute\": "
         << counters.scaled_fp8_fastpath_execute.load(std::memory_order_relaxed) << ",\n";
  output << "    \"scaled_fp8_reference_fallback\": "
         << counters.scaled_fp8_reference_fallback.load(std::memory_order_relaxed) << "\n";
  output << "  },\n";
  const auto& expert_staging_counters = nemotron::GetExpertStagingCounters();
  output << "  \"expert_staging_counters\": {\n";
  output << "    \"total_bytes_uploaded\": "
         << expert_staging_counters.total_bytes_uploaded.load(std::memory_order_relaxed) << ",\n";
  output << "    \"total_experts_staged\": "
         << expert_staging_counters.total_experts_staged.load(std::memory_order_relaxed) << ",\n";
  output << "    \"total_staging_calls\": "
         << expert_staging_counters.total_staging_calls.load(std::memory_order_relaxed) << ",\n";
  output << "    \"staging_elapsed_us\": "
         << expert_staging_counters.staging_elapsed_us.load(std::memory_order_relaxed) << "\n";
  output << "  }\n";
  output << "}\n";
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  BenchmarkOptions options;
  if (!ParseArgs(argc, argv, &options)) {
    PrintUsage(argv[0]);
    return 1;
  }
  if (!has_cuda_device()) {
    std::cerr << "nano_fused_decode_bench: no CUDA device available\n";
    return 1;
  }
  const auto environment_info = QueryEnvironmentInfo();
  if (!environment_info.has_value()) {
    return 1;
  }
  const bool cudnn_fe_available = []() {
    auto cudnn_handle = nemotron::CudnnHandle::Create();
    return cudnn_handle != nullptr && cudnn_handle->valid();
  }();
  const bool device_token_select_enabled =
      EnvEnabledOrDefault("NEMOTRON_FORWARD_DEVICE_TOKEN_SELECT", true);

  if (!options.manifest_path.has_value()) {
    const char* manifest_env = std::getenv("NEMOTRON_FORWARD_MANIFEST");
    if (manifest_env != nullptr && manifest_env[0] != '\0') {
      options.manifest_path = std::filesystem::path(manifest_env);
    }
  }
  if (!options.manifest_path.has_value()) {
    std::cerr << "nano_fused_decode_bench: manifest path is required\n";
    return 1;
  }
  if (!std::filesystem::exists(*options.manifest_path)) {
    std::cerr << "nano_fused_decode_bench: manifest path does not exist\n";
    return 1;
  }

  const auto load_result = nemotron::LoadManifestFromJsonFile(*options.manifest_path);
  if (!load_result.ok) {
    std::cerr << "nano_fused_decode_bench: manifest parse failed\n";
    return 1;
  }
  if (load_result.manifest.runtime.model_id != kNanoModelId) {
    std::cerr << "nano_fused_decode_bench: benchmark only supports Nano manifest\n";
    return 1;
  }
  const auto config_opt = config_for_manifest(load_result.manifest);
  if (!config_opt.has_value()) {
    std::cerr << "nano_fused_decode_bench: failed to resolve Nano runtime config\n";
    return 1;
  }

  auto runtime_options = make_options();
  runtime_options.service_target.target_context_tokens =
      kPromptTokenCount + std::max(kDefaultDecodeTokenCount, options.decode_token_count);

  auto runtime_environment = nemotron::RuntimeEnvironment::BuildFromManifestFile(
      *options.manifest_path, runtime_options);
  if (!runtime_environment) {
    std::cerr << "nano_fused_decode_bench: runtime environment build failed\n";
    return 1;
  }

  nemotron::SingleTokenForwardConfig config = *config_opt;
  config.max_tokens = std::max<std::size_t>(
      config.max_tokens,
      kPromptTokenCount + std::max(kDefaultDecodeTokenCount, options.decode_token_count));

  auto model = nemotron::SingleTokenForwardModel::Create(*runtime_environment, config);
  if (model == nullptr || !model->valid()) {
    std::cerr << "nano_fused_decode_bench: forward model build failed\n";
    return 1;
  }

  ScopedEnvOverride fused_mamba("NEMOTRON_FORWARD_FUSED_MAMBA_DECODE", "1");
  ScopedEnvOverride fused_moe("NEMOTRON_FORWARD_FUSED_MOE_DECODE", "1");

  const std::vector<std::int32_t>& prompt_token_ids = FixedPromptTokenIds();
  nemotron::ResetLinearOpCounters();
  nemotron::ResetLinearOpTrace();
  nemotron::ResetExpertStagingCounters();
  DeviceTokenBuffer device_token_buffer;
  if (device_token_select_enabled && !device_token_buffer.Allocate()) {
    return 1;
  }

  BenchmarkResult result;
  result.mode = BenchmarkModeJsonName(options.mode);
  result.model_id = load_result.manifest.runtime.model_id;
  result.manifest_path = options.manifest_path->string();
  result.prompt_token_count = prompt_token_ids.size();
  result.decode_token_count =
      (options.mode == BenchmarkMode::kSteadyState ||
       options.mode == BenchmarkMode::kCachedHead)
          ? options.decode_token_count
          : (options.mode == BenchmarkMode::kProfileReady ? 1 : kDefaultDecodeTokenCount);
  result.linear_device_fastpath_enabled =
      EnvEnabledOrDefault("NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH", false);
  result.device_token_select_enabled = device_token_select_enabled;
  result.cudnn_fe_available = cudnn_fe_available;

  if (options.mode == BenchmarkMode::kSteadyState ||
      options.mode == BenchmarkMode::kCachedHead) {
    const char* mode_name = BenchmarkModeCliName(options.mode);
    std::cout << "nano_fused_decode_bench: starting mode=" << mode_name
              << " prompt_tokens=" << prompt_token_ids.size()
              << " decode_tokens=" << options.decode_token_count
              << " device_token_select=" << EnabledStatus(device_token_select_enabled) << "\n";
    std::cout.flush();
    const auto steady_state_run = RunTimedSteadyStateDecode(
        *model,
        prompt_token_ids,
        options.decode_token_count,
        mode_name,
        device_token_select_enabled,
        device_token_buffer.data);
    if (!steady_state_run.has_value()) {
      return 1;
    }

    result.max_new_tokens = options.decode_token_count;
    result.warmup_iterations = 0;
    result.hot_iterations = 0;
    result.generated_token_count = steady_state_run->generated_token_ids.size();
    result.generated_token_ids = steady_state_run->generated_token_ids;
    result.steady_state_prefill_ms = steady_state_run->prefill_ms;
    result.steady_state_step_ms = steady_state_run->steady_state_step_ms;
    result.hot_steady_state_mean_ms = Mean(steady_state_run->steady_state_step_ms);
    result.hot_steady_state_min_ms = MinOrZero(steady_state_run->steady_state_step_ms);
    result.hot_steady_state_max_ms = MaxOrZero(steady_state_run->steady_state_step_ms);
    if (result.hot_steady_state_mean_ms > 0.0) {
      result.steady_state_generated_tokens_per_second =
          1000.0 / result.hot_steady_state_mean_ms;
    }

    std::cout << "nano_fused_decode_bench: mode=" << mode_name
              << " prompt_tokens=" << result.prompt_token_count
              << " decode_tokens=" << result.decode_token_count
              << " prefill_ms=" << std::fixed << std::setprecision(3)
              << result.steady_state_prefill_ms
              << " decode_mean_ms=" << result.hot_steady_state_mean_ms
              << " decode_min_ms=" << result.hot_steady_state_min_ms
              << " decode_max_ms=" << result.hot_steady_state_max_ms
              << " device_token_select=" << EnabledStatus(result.device_token_select_enabled)
              << " decode_tokens_per_second="
              << result.steady_state_generated_tokens_per_second
              << "\n";
    std::cout << "nano_fused_decode_bench: mode=" << mode_name << " per_step_ms=";
    for (std::size_t i = 0; i < result.steady_state_step_ms.size(); ++i) {
      if (i != 0) {
        std::cout << ",";
      }
      std::cout << std::fixed << std::setprecision(3) << result.steady_state_step_ms[i];
    }
    std::cout << "\n";
  } else if (options.mode == BenchmarkMode::kProfileReady) {
    std::cout << "nano_fused_decode_bench: starting mode=profile-ready"
              << " prompt_tokens=" << prompt_token_ids.size()
              << " decode_tokens=1"
              << " device_token_select=" << EnabledStatus(device_token_select_enabled) << "\n";
    std::cout.flush();
    const auto profile_ready_run = RunProfileReadyDecode(
        *model,
        prompt_token_ids,
        device_token_select_enabled,
        device_token_buffer.data);
    if (!profile_ready_run.has_value()) {
      return 1;
    }

    result.max_new_tokens = profile_ready_run->generated_token_ids.size();
    result.warmup_iterations = 0;
    result.hot_iterations = 0;
    result.generated_token_count = profile_ready_run->generated_token_ids.size();
    result.generated_token_ids = profile_ready_run->generated_token_ids;
    result.steady_state_prefill_ms = profile_ready_run->prefill_ms;
    result.steady_state_step_ms = {profile_ready_run->decode_step_ms};
    result.hot_steady_state_mean_ms = profile_ready_run->decode_step_ms;
    result.hot_steady_state_min_ms = profile_ready_run->decode_step_ms;
    result.hot_steady_state_max_ms = profile_ready_run->decode_step_ms;
    if (result.hot_steady_state_mean_ms > 0.0) {
      result.steady_state_generated_tokens_per_second =
          1000.0 / result.hot_steady_state_mean_ms;
    }
    const double total_profile_ready_ms =
        profile_ready_run->prefill_ms + profile_ready_run->decode_step_ms;
    if (total_profile_ready_ms > 0.0) {
      result.full_decode_tokens_per_second =
          (static_cast<double>(result.generated_token_count) * 1000.0) /
          total_profile_ready_ms;
    }

    std::cout << "nano_fused_decode_bench: mode=profile-ready"
              << " prompt_tokens=" << result.prompt_token_count
              << " decode_tokens=" << result.decode_token_count
              << " prefill_ms=" << std::fixed << std::setprecision(3)
              << result.steady_state_prefill_ms
              << " decode_ms=" << result.hot_steady_state_mean_ms
              << " device_token_select=" << EnabledStatus(result.device_token_select_enabled)
              << "\n";
  } else {
    std::cout << "nano_fused_decode_bench: starting cold run"
              << " prompt_tokens=" << prompt_token_ids.size()
              << " max_new_tokens=" << kDefaultDecodeTokenCount
              << " device_token_select=" << EnabledStatus(device_token_select_enabled) << "\n";
    std::cout.flush();
    const auto cold_run = RunTimedPhasedDecode(
        *model,
        prompt_token_ids,
        "cold",
        device_token_select_enabled,
        device_token_buffer.data);
    if (!cold_run.has_value()) {
      return 1;
    }
    std::cout << "nano_fused_decode_bench: completed cold run"
              << " total_ms=" << cold_run->total_ms
              << " prefill_ms=" << cold_run->prefill_ms
              << " first_token_ms=" << cold_run->first_token_ms
              << " steady_state_mean_ms=" << Mean(cold_run->steady_state_step_ms)
              << "\n";
    std::cout.flush();

    for (std::size_t iteration = 0; iteration < options.warmup_iterations; ++iteration) {
      const std::string run_label =
          "warmup " + std::to_string(iteration + 1) + "/" +
          std::to_string(options.warmup_iterations);
      std::cout << "nano_fused_decode_bench: " << run_label << " start\n";
      std::cout.flush();
      const auto warmup_run = RunTimedPhasedDecode(
          *model,
          prompt_token_ids,
          run_label,
          device_token_select_enabled,
          device_token_buffer.data);
      if (!warmup_run.has_value()) {
        return 1;
      }
      std::cout << "nano_fused_decode_bench: " << run_label
                << " total_ms=" << warmup_run->total_ms
                << " steady_state_mean_ms=" << Mean(warmup_run->steady_state_step_ms)
                << "\n";
      std::cout.flush();
    }

    std::vector<double> hot_total_ms;
    std::vector<double> hot_prefill_ms;
    std::vector<double> hot_first_token_ms;
    std::vector<double> hot_steady_state_mean_ms;
    std::vector<double> hot_steady_state_all_steps;
    std::vector<std::int32_t> last_generated_token_ids;
    for (std::size_t iteration = 0; iteration < options.hot_iterations; ++iteration) {
      const std::string run_label =
          "hot " + std::to_string(iteration + 1) + "/" + std::to_string(options.hot_iterations);
      std::cout << "nano_fused_decode_bench: " << run_label << " start\n";
      std::cout.flush();
      const auto hot_run = RunTimedPhasedDecode(
          *model,
          prompt_token_ids,
          run_label,
          device_token_select_enabled,
          device_token_buffer.data);
      if (!hot_run.has_value()) {
        return 1;
      }
      hot_total_ms.push_back(hot_run->total_ms);
      hot_prefill_ms.push_back(hot_run->prefill_ms);
      hot_first_token_ms.push_back(hot_run->first_token_ms);
      hot_steady_state_mean_ms.push_back(Mean(hot_run->steady_state_step_ms));
      hot_steady_state_all_steps.insert(
          hot_steady_state_all_steps.end(),
          hot_run->steady_state_step_ms.begin(),
          hot_run->steady_state_step_ms.end());
      last_generated_token_ids = hot_run->generated_token_ids;
      std::cout << "nano_fused_decode_bench: " << run_label
                << " total_ms=" << hot_run->total_ms
                << " prefill_ms=" << hot_run->prefill_ms
                << " first_token_ms=" << hot_run->first_token_ms
                << " steady_state_mean_ms=" << Mean(hot_run->steady_state_step_ms)
                << "\n";
      std::cout.flush();
    }

    result.max_new_tokens = kDefaultDecodeTokenCount;
    result.warmup_iterations = options.warmup_iterations;
    result.hot_iterations = options.hot_iterations;
    result.generated_token_count =
        !last_generated_token_ids.empty() ? last_generated_token_ids.size()
                                          : cold_run->generated_token_ids.size();
    result.generated_token_ids =
        !last_generated_token_ids.empty() ? last_generated_token_ids
                                          : cold_run->generated_token_ids;
    result.cold_total_ms = cold_run->total_ms;
    result.cold_prefill_ms = cold_run->prefill_ms;
    result.cold_first_token_ms = cold_run->first_token_ms;
    result.cold_steady_state_mean_ms = Mean(cold_run->steady_state_step_ms);
    result.cold_steady_state_min_ms = MinOrZero(cold_run->steady_state_step_ms);
    result.cold_steady_state_max_ms = MaxOrZero(cold_run->steady_state_step_ms);
    result.hot_total_mean_ms = Mean(hot_total_ms);
    result.hot_total_min_ms = MinOrZero(hot_total_ms);
    result.hot_total_max_ms = MaxOrZero(hot_total_ms);
    result.hot_prefill_mean_ms = Mean(hot_prefill_ms);
    result.hot_first_token_mean_ms = Mean(hot_first_token_ms);
    result.hot_steady_state_mean_ms = Mean(hot_steady_state_mean_ms);
    result.hot_steady_state_min_ms = MinOrZero(hot_steady_state_all_steps);
    result.hot_steady_state_max_ms = MaxOrZero(hot_steady_state_all_steps);
    if (result.hot_steady_state_mean_ms > 0.0) {
      result.steady_state_generated_tokens_per_second =
          1000.0 / result.hot_steady_state_mean_ms;
    }
    if (result.hot_total_mean_ms > 0.0) {
      result.full_decode_tokens_per_second =
          (static_cast<double>(result.generated_token_count) * 1000.0) / result.hot_total_mean_ms;
    }

    std::cout << "nano_fused_decode_bench: mode=phased"
              << " prompt_tokens=" << result.prompt_token_count
              << " generated_tokens=" << result.generated_token_count
              << " cold_prefill_ms=" << std::fixed << std::setprecision(3)
              << result.cold_prefill_ms
              << " cold_first_token_ms=" << result.cold_first_token_ms
              << " cold_steady_state_mean_ms=" << result.cold_steady_state_mean_ms
              << " hot_prefill_mean_ms=" << result.hot_prefill_mean_ms
              << " hot_first_token_mean_ms=" << result.hot_first_token_mean_ms
              << " hot_steady_state_mean_ms=" << result.hot_steady_state_mean_ms
              << " device_token_select=" << EnabledStatus(result.device_token_select_enabled)
              << " steady_state_generated_tokens_per_second="
              << result.steady_state_generated_tokens_per_second
              << "\n";
  }

  nemotron::PrintLinearOpCounterSummary(std::cout);
  nemotron::PrintExpertStagingCounterSummary(std::cout);
  std::cout.flush();

  if (options.json_output_path.has_value() &&
      !WriteJson(*options.json_output_path, *environment_info, result, prompt_token_ids)) {
    return 1;
  }

  if (options.strict_linear) {
    bool strict_linear_failed = false;
    if (result.linear_device_fastpath_enabled) {
      const std::vector<LinearFallbackObservation> fallbacks =
          CollectLinearReferenceFallbacks();
      if (!fallbacks.empty()) {
        PrintUnexpectedLinearFallbacks(std::cerr, fallbacks);
        strict_linear_failed = true;
      }
    }
    nemotron::PrintLinearOpTraceSummary(std::cerr);
    std::cerr.flush();
    if (strict_linear_failed) {
      return 1;
    }
  }
  return 0;
}
