#include "nemotron/attention_layer.h"
#include "nemotron/device_argmax.h"
#include "nemotron/device_tensor.h"
#include "nemotron/expert_layer.h"
#include "nemotron/manifest.h"
#include "nemotron/mamba_layer.h"
#include "nemotron/prefix_cache.h"
#include "nemotron/runtime_environment.h"
#include "nemotron/single_token_forward_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char* kNanoModelId = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4";
constexpr std::size_t kDefaultIterations = 5;
constexpr std::size_t kDefaultWarmupIterations = 1;
constexpr std::size_t kDefaultTailTokenCount = 32;
constexpr std::size_t kFirstDecodeTokenCount = 1;
constexpr std::size_t kPrefixLengths[] = {256, 1024, 4096};

constexpr std::size_t GiB(std::size_t value) {
  return value * 1024ull * 1024ull * 1024ull;
}

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

 private:
  std::string name_;
  bool had_original_ = false;
  std::string original_value_;
};

bool CheckCuda(cudaError_t status, const char* message) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << "nano_prefix_cache_ttft_bench: " << message << ": "
            << cudaGetErrorString(status) << "\n";
  return false;
}

bool HasCudaDevice() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

struct BenchmarkOptions {
  std::size_t warmup_iterations = kDefaultWarmupIterations;
  std::size_t measured_iterations = kDefaultIterations;
  std::size_t tail_token_count = kDefaultTailTokenCount;
  std::size_t moe_prefill_window_tokens = 0;
  std::size_t target_active_requests = 1;
  bool skip_commit_snapshot = false;
  std::vector<std::size_t> prefix_lengths;
  std::vector<std::string> case_filters;
};

enum class Scenario {
  kCold,
  kCommittedHead,
  kGlobalRoot,
};

struct CaseSpec {
  Scenario scenario = Scenario::kCold;
  std::size_t prefix_token_count = 0;
  std::size_t tail_token_count = 0;
};

struct IterationMetrics {
  double cold_ttft_ms = 0.0;
  std::optional<double> restore_latency_ms;
  std::optional<double> tail_prefill_latency_ms;
  std::optional<double> first_token_decode_latency_ms;
  std::optional<double> commit_latency_ms;
  std::optional<double> hot_prefix_ttft_ms;
  std::optional<double> speedup_factor;
  std::optional<double> prefix_snapshot_size_bytes;
  std::optional<double> matched_prefix_token_count;
  std::optional<double> attention_native_multi_token_runs;
  std::optional<double> attention_native_multi_token_tokens;
  std::optional<double> attention_row_replay_runs;
  std::optional<double> attention_row_replay_tokens;
  std::optional<double> expert_native_multi_token_runs;
  std::optional<double> expert_native_multi_token_tokens;
  std::optional<double> expert_row_replay_runs;
  std::optional<double> expert_row_replay_tokens;
  std::optional<double> mamba_native_multi_token_runs;
  std::optional<double> mamba_native_multi_token_tokens;
  std::optional<double> mamba_row_replay_runs;
  std::optional<double> mamba_row_replay_tokens;
};

struct MetricSummary {
  double median = 0.0;
  double p95 = 0.0;
};

struct CaseSummary {
  CaseSpec spec;
  MetricSummary cold_ttft_ms;
  std::optional<MetricSummary> restore_latency_ms;
  std::optional<MetricSummary> tail_prefill_latency_ms;
  std::optional<MetricSummary> first_token_decode_latency_ms;
  std::optional<MetricSummary> commit_latency_ms;
  std::optional<MetricSummary> hot_prefix_ttft_ms;
  std::optional<MetricSummary> speedup_factor;
  std::optional<MetricSummary> prefix_snapshot_size_bytes;
  std::optional<MetricSummary> matched_prefix_token_count;
  std::optional<MetricSummary> attention_native_multi_token_runs;
  std::optional<MetricSummary> attention_native_multi_token_tokens;
  std::optional<MetricSummary> attention_row_replay_runs;
  std::optional<MetricSummary> attention_row_replay_tokens;
  std::optional<MetricSummary> expert_native_multi_token_runs;
  std::optional<MetricSummary> expert_native_multi_token_tokens;
  std::optional<MetricSummary> expert_row_replay_runs;
  std::optional<MetricSummary> expert_row_replay_tokens;
  std::optional<MetricSummary> mamba_native_multi_token_runs;
  std::optional<MetricSummary> mamba_native_multi_token_tokens;
  std::optional<MetricSummary> mamba_row_replay_runs;
  std::optional<MetricSummary> mamba_row_replay_tokens;
};

struct LayerExecutionSnapshot {
  double native_multi_token_runs = 0.0;
  double native_multi_token_tokens = 0.0;
  double row_replay_runs = 0.0;
  double row_replay_tokens = 0.0;
};

struct BenchmarkExecutionSnapshot {
  LayerExecutionSnapshot attention;
  LayerExecutionSnapshot expert;
  LayerExecutionSnapshot mamba;
};

struct ColdPassResult {
  double prefill_ms = 0.0;
  double first_token_decode_ms = 0.0;
  double cold_ttft_ms = 0.0;
  std::int32_t generated_token_id = -1;
  std::vector<float> boundary_logits;
};

struct SeededCacheState {
  nemotron::PrefixNodeId node_id = 0;
  std::size_t snapshot_size_bytes = 0;
};

class DeviceTokenBuffer {
 public:
  ~DeviceTokenBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  bool Allocate() {
    return CheckCuda(
        cudaMalloc(reinterpret_cast<void**>(&data_), sizeof(std::int32_t)),
        "cudaMalloc device token buffer failed");
  }

  std::int32_t* data() const {
    return data_;
  }

 private:
  std::int32_t* data_ = nullptr;
};

class CudaEventTimer {
 public:
  CudaEventTimer() {
    valid_ =
        CheckCuda(cudaEventCreate(&start_), "cudaEventCreate start failed") &&
        CheckCuda(cudaEventCreate(&stop_), "cudaEventCreate stop failed");
    if (!valid_) {
      if (start_ != nullptr) {
        cudaEventDestroy(start_);
        start_ = nullptr;
      }
      if (stop_ != nullptr) {
        cudaEventDestroy(stop_);
        stop_ = nullptr;
      }
    }
  }

  ~CudaEventTimer() {
    if (start_ != nullptr) {
      cudaEventDestroy(start_);
    }
    if (stop_ != nullptr) {
      cudaEventDestroy(stop_);
    }
  }

  bool valid() const {
    return valid_;
  }

  template <typename Fn>
  bool Measure(const char* label, Fn&& fn, double* elapsed_ms) {
    if (!valid_ || elapsed_ms == nullptr) {
      return false;
    }
    if (!CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize before timing failed")) {
      return false;
    }
    if (!CheckCuda(cudaEventRecord(start_), "cudaEventRecord start failed")) {
      return false;
    }
    if (!fn()) {
      std::cerr << "nano_prefix_cache_ttft_bench: timed operation failed: " << label << "\n";
      return false;
    }
    if (!CheckCuda(cudaEventRecord(stop_), "cudaEventRecord stop failed") ||
        !CheckCuda(cudaEventSynchronize(stop_), "cudaEventSynchronize stop failed")) {
      return false;
    }
    float elapsed = 0.0f;
    if (!CheckCuda(cudaEventElapsedTime(&elapsed, start_, stop_), "cudaEventElapsedTime failed")) {
      return false;
    }
    *elapsed_ms = static_cast<double>(elapsed);
    return true;
  }

 private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
  bool valid_ = false;
};

bool ParsePositiveSizeT(const char* text, std::size_t* value_out) {
  if (text == nullptr || value_out == nullptr || text[0] == '\0') {
    return false;
  }
  char* parse_end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(text, &parse_end, 10);
  if (errno != 0 || parse_end == text || (parse_end != nullptr && *parse_end != '\0') || parsed == 0) {
    return false;
  }
  *value_out = static_cast<std::size_t>(parsed);
  return true;
}

bool ParseNonNegativeSizeT(const char* text, std::size_t* value_out) {
  if (text == nullptr || value_out == nullptr || text[0] == '\0') {
    return false;
  }
  char* parse_end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(text, &parse_end, 10);
  if (errno != 0 || parse_end == text || (parse_end != nullptr && *parse_end != '\0')) {
    return false;
  }
  *value_out = static_cast<std::size_t>(parsed);
  return true;
}

void PrintUsage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0
      << " [--warmup <count>] [--iterations <count>] [--tail-token-count <count>]"
      << " [--moe-prefill-window-tokens <count>] [--target-active-requests <count>]"
      << " [--prefix-length <count>] [--case <case-name>] [--skip-commit-snapshot]\n"
      << "  Manifest path is read from NEMOTRON_FORWARD_MANIFEST.\n"
      << "  --warmup <count>      Discarded warmup iterations per case. Default: 1\n"
      << "  --iterations <count>  Measured iterations per case (must be >= 5). Default: 5\n"
      << "  --tail-token-count <count>\n"
      << "                        Tail token count for resumed-prefix cases. Default: 32\n"
      << "  --moe-prefill-window-tokens <count>\n"
      << "                        Explicit SingleTokenForwardConfig.moe_prefill_window_tokens.\n"
      << "                        Default: 0 (use the runtime production default).\n"
      << "  --target-active-requests <count>\n"
      << "                        Runtime bootstrap target active requests. Default: 1\n"
      << "  --prefix-length <count>\n"
      << "                        Prefix length to benchmark. Repeatable. Defaults: 256, 1024, 4096\n"
      << "  --case <case-name>    Run only the named benchmark case. Repeatable.\n"
      << "  --skip-commit-snapshot\n"
      << "                        Skip publishing the next committed snapshot after resumed decode.\n";
}

bool ParseArgs(int argc, char** argv, BenchmarkOptions* options) {
  if (options == nullptr) {
    return false;
  }
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--warmup") {
      if (i + 1 >= argc || !ParseNonNegativeSizeT(argv[++i], &options->warmup_iterations)) {
        return false;
      }
    } else if (arg == "--iterations") {
      if (i + 1 >= argc || !ParsePositiveSizeT(argv[++i], &options->measured_iterations)) {
        return false;
      }
    } else if (arg == "--tail-token-count") {
      if (i + 1 >= argc || !ParsePositiveSizeT(argv[++i], &options->tail_token_count)) {
        return false;
      }
    } else if (arg == "--moe-prefill-window-tokens") {
      if (i + 1 >= argc ||
          !ParseNonNegativeSizeT(argv[++i], &options->moe_prefill_window_tokens)) {
        return false;
      }
    } else if (arg == "--target-active-requests") {
      if (i + 1 >= argc || !ParsePositiveSizeT(argv[++i], &options->target_active_requests)) {
        return false;
      }
    } else if (arg == "--prefix-length") {
      std::size_t prefix_length = 0;
      if (i + 1 >= argc || !ParsePositiveSizeT(argv[++i], &prefix_length)) {
        return false;
      }
      options->prefix_lengths.push_back(prefix_length);
    } else if (arg == "--case") {
      if (i + 1 >= argc) {
        return false;
      }
      options->case_filters.push_back(argv[++i]);
    } else if (arg == "--skip-commit-snapshot") {
      options->skip_commit_snapshot = true;
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      return false;
    }
  }
  return options->measured_iterations >= kDefaultIterations;
}

struct ResolvedMoePrefillSettings {
  std::size_t cli_window_tokens = 0;
  std::size_t runtime_capacity_tokens = 0;
  std::size_t runtime_window_tokens = 1;
};

std::optional<ResolvedMoePrefillSettings> ResolveMoePrefillSettingsForHeader(
    const nemotron::SingleTokenForwardModel& model) {
  auto request_context = model.CreateRequestContext();
  if (request_context == nullptr || !request_context->valid()) {
    std::cerr << "nano_prefix_cache_ttft_bench: failed to create request context for header"
              << " MoE prefill resolution\n";
    return std::nullopt;
  }

  ResolvedMoePrefillSettings resolved;
  resolved.cli_window_tokens = model.config().moe_prefill_window_tokens;
  resolved.runtime_capacity_tokens = request_context->config().moe_prefill_capacity_tokens;
  if (const nemotron::MoePrefillWorkspace* workspace = request_context->moe_prefill_workspace();
      workspace != nullptr && workspace->valid() && workspace->token_capacity() != 0) {
    resolved.runtime_window_tokens = workspace->token_capacity();
  } else if (resolved.cli_window_tokens != 0) {
    resolved.runtime_window_tokens = resolved.cli_window_tokens;
  }
  return resolved;
}

nemotron::RuntimeBootstrapOptions MakeOptions(
    std::size_t max_context_tokens,
    std::size_t target_active_requests) {
  nemotron::RuntimeBootstrapOptions options;
  options.service_target.total_memory_bytes = GiB(32);
  options.service_target.weights_bytes = GiB(20);
  options.service_target.workspace_bytes = GiB(1);
  options.service_target.graph_bytes = 512ULL * 1024 * 1024;
  options.service_target.safety_headroom_bytes = GiB(1);
  options.service_target.target_active_requests = target_active_requests;
  options.service_target.target_context_tokens = max_context_tokens;
  options.use_fp16_mamba_state = false;
  options.reusable_node_metadata_bytes = 4096;
  options.verify_manifest_files = false;
  return options;
}

std::optional<nemotron::SingleTokenForwardConfig> ConfigForManifest(
    const nemotron::PackedModelManifest& manifest) {
  if (manifest.runtime.model_id == kNanoModelId) {
    return nemotron::KnownNemotron3Nano30BA3BConfig();
  }
  return std::nullopt;
}

const char* ScenarioName(Scenario scenario) {
  switch (scenario) {
    case Scenario::kCold:
      return "cold_prefill";
    case Scenario::kCommittedHead:
      return "cached_committed_head";
    case Scenario::kGlobalRoot:
      return "cached_global_root";
  }
  return "unknown";
}

const char* MatchSourceName(nemotron::CacheMatchSource source) {
  switch (source) {
    case nemotron::CacheMatchSource::kNone:
      return "none";
    case nemotron::CacheMatchSource::kConversationCommittedHead:
      return "conversation_committed_head";
    case nemotron::CacheMatchSource::kGlobalRoot:
      return "global_root";
  }
  return "unknown";
}

std::string CaseName(const CaseSpec& spec) {
  std::string name = ScenarioName(spec.scenario);
  name += "_prefix";
  name += std::to_string(spec.prefix_token_count);
  if (spec.tail_token_count != 0) {
    name += "_tail";
    name += std::to_string(spec.tail_token_count);
  }
  return name;
}

bool CaseSelected(
    const BenchmarkOptions& options,
    const CaseSpec& spec) {
  if (options.case_filters.empty()) {
    return true;
  }
  const std::string name = CaseName(spec);
  return std::find(options.case_filters.begin(), options.case_filters.end(), name) !=
         options.case_filters.end();
}

std::vector<std::int32_t> MakeTokenIds(
    std::size_t count,
    std::size_t seed,
    std::size_t vocab_size) {
  std::vector<std::int32_t> token_ids;
  token_ids.reserve(count);
  const std::size_t usable_vocab = vocab_size > 512 ? vocab_size - 256 : vocab_size;
  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t value = usable_vocab == 0 ? 0 : ((seed + (i * 97)) % usable_vocab);
    token_ids.push_back(static_cast<std::int32_t>(value + 128));
  }
  return token_ids;
}

std::vector<std::int32_t> ConcatTokenIds(
    const std::vector<std::int32_t>& prefix,
    const std::vector<std::int32_t>& tail) {
  std::vector<std::int32_t> combined;
  combined.reserve(prefix.size() + tail.size());
  combined.insert(combined.end(), prefix.begin(), prefix.end());
  combined.insert(combined.end(), tail.begin(), tail.end());
  return combined;
}

nemotron::SerializedPromptIdentity MakeIdentity(
    const std::vector<std::int32_t>& token_ids,
    const std::string& model_id) {
  nemotron::SerializedPromptIdentity identity;
  identity.token_ids = token_ids;
  identity.tenant_namespace = "nano-prefix-cache-ttft";
  identity.tokenizer_revision = "nano-prefix-cache-ttft";
  identity.serializer_revision = "nano-prefix-cache-ttft";
  identity.model_revision = model_id;
  identity.reasoning_mode = false;
  return identity;
}

std::unique_ptr<nemotron::DeviceTensorFp32> CreateLastRowView(nemotron::DeviceTensorFp32& logits) {
  if (!logits.valid() || logits.shape().size() != 2 || logits.shape()[0] == 0) {
    return nullptr;
  }
  const std::size_t row_width = logits.shape()[1];
  const std::size_t row_index = logits.shape()[0] - 1;
  return nemotron::DeviceTensorFp32::CreateView(
      {1, row_width},
      logits.data() + (row_index * row_width));
}

std::vector<float> CopyTensorToHost(const nemotron::DeviceTensorFp32& tensor) {
  std::vector<float> host(tensor.numel(), 0.0f);
  if (!tensor.CopyToHost(host.data(), host.size())) {
    return {};
  }
  return host;
}

std::optional<std::int32_t> SelectTokenId(
    const nemotron::DeviceTensorFp32& logits_row,
    std::int32_t* device_token_id) {
  if (device_token_id == nullptr) {
    return std::nullopt;
  }
  if (!nemotron::DeviceArgmax(logits_row, device_token_id)) {
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

double Percentile(std::vector<double> values, double percentile) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double scaled_index =
      (percentile / 100.0) * static_cast<double>(values.size() - 1);
  const std::size_t lower_index = static_cast<std::size_t>(std::floor(scaled_index));
  const std::size_t upper_index = static_cast<std::size_t>(std::ceil(scaled_index));
  if (lower_index == upper_index) {
    return values[lower_index];
  }
  const double fraction = scaled_index - static_cast<double>(lower_index);
  return values[lower_index] + fraction * (values[upper_index] - values[lower_index]);
}

MetricSummary SummarizeMetric(const std::vector<double>& values) {
  return MetricSummary{
      Percentile(values, 50.0),
      Percentile(values, 95.0),
  };
}

void ResetLayerExecutionCounters() {
  nemotron::ResetAttentionLayerExecutionCounters();
  nemotron::ResetExpertLayerExecutionCounters();
  nemotron::ResetMambaLayerExecutionCounters();
}

BenchmarkExecutionSnapshot CollectLayerExecutionCounters() {
  const nemotron::AttentionLayerExecutionCounters attention =
      nemotron::GetAttentionLayerExecutionCounters();
  const nemotron::ExpertLayerExecutionCounters expert =
      nemotron::GetExpertLayerExecutionCounters();
  const nemotron::MambaLayerExecutionCounters mamba =
      nemotron::GetMambaLayerExecutionCounters();
  return BenchmarkExecutionSnapshot{
      LayerExecutionSnapshot{
          static_cast<double>(attention.native_multi_token_runs),
          static_cast<double>(attention.native_multi_token_tokens),
          static_cast<double>(attention.row_replay_runs),
          static_cast<double>(attention.row_replay_tokens),
      },
      LayerExecutionSnapshot{
          static_cast<double>(expert.native_multi_token_runs),
          static_cast<double>(expert.native_multi_token_tokens),
          static_cast<double>(expert.row_replay_runs),
          static_cast<double>(expert.row_replay_tokens),
      },
      LayerExecutionSnapshot{
          static_cast<double>(mamba.native_multi_token_runs),
          static_cast<double>(mamba.native_multi_token_tokens),
          static_cast<double>(mamba.row_replay_runs),
          static_cast<double>(mamba.row_replay_tokens),
      },
  };
}

void RecordLayerExecutionMetrics(
    const BenchmarkExecutionSnapshot& counters,
    IterationMetrics* metrics) {
  if (metrics == nullptr) {
    return;
  }
  metrics->attention_native_multi_token_runs = counters.attention.native_multi_token_runs;
  metrics->attention_native_multi_token_tokens = counters.attention.native_multi_token_tokens;
  metrics->attention_row_replay_runs = counters.attention.row_replay_runs;
  metrics->attention_row_replay_tokens = counters.attention.row_replay_tokens;
  metrics->expert_native_multi_token_runs = counters.expert.native_multi_token_runs;
  metrics->expert_native_multi_token_tokens = counters.expert.native_multi_token_tokens;
  metrics->expert_row_replay_runs = counters.expert.row_replay_runs;
  metrics->expert_row_replay_tokens = counters.expert.row_replay_tokens;
  metrics->mamba_native_multi_token_runs = counters.mamba.native_multi_token_runs;
  metrics->mamba_native_multi_token_tokens = counters.mamba.native_multi_token_tokens;
  metrics->mamba_row_replay_runs = counters.mamba.row_replay_runs;
  metrics->mamba_row_replay_tokens = counters.mamba.row_replay_tokens;
}

template <typename Accessor>
std::vector<double> CollectOptionalMetrics(
    const std::vector<IterationMetrics>& iterations,
    Accessor accessor) {
  std::vector<double> values;
  values.reserve(iterations.size());
  for (const IterationMetrics& iteration : iterations) {
    const std::optional<double> value = accessor(iteration);
    if (value.has_value()) {
      values.push_back(*value);
    }
  }
  return values;
}

std::optional<ColdPassResult> RunColdPass(
    nemotron::SingleTokenForwardModel& model,
    const std::vector<std::int32_t>& prompt_token_ids,
    std::int32_t* device_token_id) {
  auto request_context = model.CreateRequestContext();
  if (request_context == nullptr || !request_context->valid()) {
    std::cerr << "nano_prefix_cache_ttft_bench: failed to create request context\n";
    return std::nullopt;
  }
  auto prompt_logits =
      nemotron::DeviceTensorFp32::Create({1, model.config().vocab_size});
  auto step_logits = nemotron::DeviceTensorFp32::Create({1, model.config().vocab_size});
  if (prompt_logits == nullptr || !prompt_logits->valid() ||
      step_logits == nullptr || !step_logits->valid()) {
    std::cerr << "nano_prefix_cache_ttft_bench: failed to allocate cold logits buffers\n";
    return std::nullopt;
  }

  CudaEventTimer timer;
  if (!timer.valid()) {
    return std::nullopt;
  }

  ColdPassResult result;
  if (!timer.Measure(
          "cold RunPrefill",
          [&]() {
            return model.RunPrefill(
                prompt_token_ids.data(),
                prompt_token_ids.size(),
                *request_context,
                prompt_logits.get());
          },
          &result.prefill_ms)) {
    return std::nullopt;
  }

  const auto first_token_id = SelectTokenId(*prompt_logits, device_token_id);
  if (!first_token_id.has_value()) {
    std::cerr << "nano_prefix_cache_ttft_bench: failed to select cold first token\n";
    return std::nullopt;
  }

  if (!timer.Measure(
          "cold ContinueSingleToken",
          [&]() {
            return model.ContinueSingleToken(
                *first_token_id,
                *request_context,
                step_logits.get());
          },
          &result.first_token_decode_ms)) {
    return std::nullopt;
  }

  result.generated_token_id = *first_token_id;
  result.boundary_logits = CopyTensorToHost(*step_logits);
  if (result.boundary_logits.empty()) {
    std::cerr << "nano_prefix_cache_ttft_bench: failed to copy cold decode logits\n";
    return std::nullopt;
  }
  result.cold_ttft_ms = result.prefill_ms + result.first_token_decode_ms;
  return result;
}

std::optional<SeededCacheState> SeedPrefixCache(
    Scenario scenario,
    nemotron::SingleTokenForwardModel& model,
    nemotron::PrefixCache& prefix_cache,
    const nemotron::SerializedPromptIdentity& prefix_identity,
    const std::string& conversation_id) {
  auto request_context = model.CreateRequestContext();
  if (request_context == nullptr || !request_context->valid()) {
    std::cerr << "nano_prefix_cache_ttft_bench: failed to create cache seed request context\n";
    return std::nullopt;
  }
  auto prompt_logits =
      nemotron::DeviceTensorFp32::Create({1, model.config().vocab_size});
  if (prompt_logits == nullptr || !prompt_logits->valid()) {
    std::cerr << "nano_prefix_cache_ttft_bench: failed to allocate cache seed prompt logits\n";
    return std::nullopt;
  }
  if (!model.RunPrefill(
          prefix_identity.token_ids.data(),
          prefix_identity.token_ids.size(),
          *request_context,
          prompt_logits.get())) {
    std::cerr << "nano_prefix_cache_ttft_bench: cache seed RunPrefill failed\n";
    return std::nullopt;
  }

  std::vector<float> boundary_logits = CopyTensorToHost(*prompt_logits);
  if (boundary_logits.empty()) {
    std::cerr << "nano_prefix_cache_ttft_bench: failed to copy cache seed boundary logits\n";
    return std::nullopt;
  }

  nemotron::PrefixNodeId node_id = 0;
  if (scenario == Scenario::kCommittedHead) {
    node_id = prefix_cache.PublishConversationHeadSnapshot(
        conversation_id,
        prefix_identity,
        *request_context,
        conversation_id + "/seed",
        &boundary_logits);
  } else if (scenario == Scenario::kGlobalRoot) {
    node_id = prefix_cache.PublishGlobalRootSnapshot(
        prefix_identity,
        *request_context,
        conversation_id + "/root",
        &boundary_logits);
  }

  if (node_id == 0) {
    std::cerr << "nano_prefix_cache_ttft_bench: failed to publish cache seed node\n";
    return std::nullopt;
  }
  const auto entry = prefix_cache.Describe(node_id);
  if (!entry.has_value()) {
    std::cerr << "nano_prefix_cache_ttft_bench: failed to describe cache seed node\n";
    return std::nullopt;
  }
  return SeededCacheState{
      node_id,
      entry->total_bytes,
  };
}

std::optional<IterationMetrics> RunResumeIteration(
    Scenario scenario,
    nemotron::SingleTokenForwardModel& model,
    nemotron::PrefixCache& prefix_cache,
    const std::string& model_id,
    const std::vector<std::int32_t>& prefix_token_ids,
    const std::vector<std::int32_t>& tail_token_ids,
    std::size_t iteration_index,
    bool skip_commit_snapshot,
    std::int32_t* device_token_id) {
  if (scenario != Scenario::kCommittedHead && scenario != Scenario::kGlobalRoot) {
    return std::nullopt;
  }

  const std::vector<std::int32_t> full_prompt_token_ids =
      ConcatTokenIds(prefix_token_ids, tail_token_ids);
  prefix_cache.Clear();
  const auto cold = RunColdPass(model, full_prompt_token_ids, device_token_id);
  if (!cold.has_value()) {
    return std::nullopt;
  }

  const std::string conversation_id =
      std::string(ScenarioName(scenario)) + "_iter_" + std::to_string(iteration_index);
  const auto prefix_identity = MakeIdentity(prefix_token_ids, model_id);
  const auto seeded_cache =
      SeedPrefixCache(scenario, model, prefix_cache, prefix_identity, conversation_id);
  if (!seeded_cache.has_value()) {
    return std::nullopt;
  }

  const auto full_identity = MakeIdentity(full_prompt_token_ids, model_id);
  nemotron::CacheLookupRequest lookup_request;
  lookup_request.identity = full_identity;
  lookup_request.conversation_id = conversation_id;
  const nemotron::CacheMatch match = prefix_cache.Lookup(lookup_request);
  const nemotron::CacheMatchSource expected_source =
      scenario == Scenario::kCommittedHead
          ? nemotron::CacheMatchSource::kConversationCommittedHead
          : nemotron::CacheMatchSource::kGlobalRoot;
  if (!match.hit() ||
      match.source != expected_source ||
      match.matched_token_count != prefix_token_ids.size()) {
    std::cerr << "nano_prefix_cache_ttft_bench: unexpected cache lookup result"
              << " expected_source=" << MatchSourceName(expected_source)
              << " actual_source=" << MatchSourceName(match.source)
              << " expected_matched_tokens=" << prefix_token_ids.size()
              << " actual_matched_tokens=" << match.matched_token_count
              << "\n";
    return std::nullopt;
  }

  auto request_context = model.CreateRequestContext();
  if (request_context == nullptr || !request_context->valid()) {
    std::cerr << "nano_prefix_cache_ttft_bench: failed to create resume request context\n";
    return std::nullopt;
  }

  auto tail_logits =
      nemotron::DeviceTensorFp32::Create({1, model.config().vocab_size});
  auto step_logits = nemotron::DeviceTensorFp32::Create({1, model.config().vocab_size});
  if (tail_logits == nullptr || !tail_logits->valid() ||
      step_logits == nullptr || !step_logits->valid()) {
    std::cerr << "nano_prefix_cache_ttft_bench: failed to allocate resume logits buffers\n";
    return std::nullopt;
  }

  CudaEventTimer timer;
  if (!timer.valid()) {
    return std::nullopt;
  }

  IterationMetrics metrics;
  metrics.cold_ttft_ms = cold->cold_ttft_ms;
  if (!timer.Measure(
          "RestoreMatchState",
          [&]() { return prefix_cache.RestoreMatchState(match, *request_context); },
          &metrics.restore_latency_ms.emplace())) {
    return std::nullopt;
  }

  if (!timer.Measure(
          "ContinuePrefill tail",
          [&]() {
            return model.ContinuePrefill(
                tail_token_ids.data(),
                tail_token_ids.size(),
                *request_context,
                tail_logits.get());
          },
          &metrics.tail_prefill_latency_ms.emplace())) {
    return std::nullopt;
  }

  const auto first_token_id = SelectTokenId(*tail_logits, device_token_id);
  if (!first_token_id.has_value()) {
    std::cerr << "nano_prefix_cache_ttft_bench: failed to select resumed first token\n";
    return std::nullopt;
  }

  if (!timer.Measure(
          "ContinueSingleToken first decode",
          [&]() {
            return model.ContinueSingleToken(
                *first_token_id,
                *request_context,
                step_logits.get());
          },
          &metrics.first_token_decode_latency_ms.emplace())) {
    return std::nullopt;
  }

  std::vector<float> boundary_logits = CopyTensorToHost(*step_logits);
  if (boundary_logits.empty()) {
    std::cerr << "nano_prefix_cache_ttft_bench: failed to copy resumed decode logits\n";
    return std::nullopt;
  }

  if (!skip_commit_snapshot) {
    auto committed_identity = full_identity;
    committed_identity.token_ids.push_back(*first_token_id);
    if (!timer.Measure(
            "PublishConversationHeadSnapshot",
            [&]() {
              return prefix_cache.PublishConversationHeadSnapshot(
                         conversation_id,
                         committed_identity,
                         *request_context,
                         conversation_id + "/committed",
                         &boundary_logits) != 0;
            },
            &metrics.commit_latency_ms.emplace())) {
      return std::nullopt;
    }
  }

  metrics.hot_prefix_ttft_ms =
      *metrics.restore_latency_ms +
      *metrics.tail_prefill_latency_ms +
      *metrics.first_token_decode_latency_ms;
  if (*metrics.hot_prefix_ttft_ms > 0.0) {
    metrics.speedup_factor = metrics.cold_ttft_ms / *metrics.hot_prefix_ttft_ms;
  } else {
    metrics.speedup_factor = 0.0;
  }
  metrics.prefix_snapshot_size_bytes =
      static_cast<double>(seeded_cache->snapshot_size_bytes);
  metrics.matched_prefix_token_count =
      static_cast<double>(match.matched_token_count);
  return metrics;
}

CaseSummary SummarizeCase(
    const CaseSpec& spec,
    const std::vector<IterationMetrics>& iterations) {
  std::vector<double> cold_ttft_ms;
  cold_ttft_ms.reserve(iterations.size());
  for (const IterationMetrics& iteration : iterations) {
    cold_ttft_ms.push_back(iteration.cold_ttft_ms);
  }

  CaseSummary summary;
  summary.spec = spec;
  summary.cold_ttft_ms = SummarizeMetric(cold_ttft_ms);

  const auto restore_ms = CollectOptionalMetrics(
      iterations,
      [](const IterationMetrics& metrics) { return metrics.restore_latency_ms; });
  if (!restore_ms.empty()) {
    summary.restore_latency_ms = SummarizeMetric(restore_ms);
  }
  const auto tail_prefill_ms = CollectOptionalMetrics(
      iterations,
      [](const IterationMetrics& metrics) { return metrics.tail_prefill_latency_ms; });
  if (!tail_prefill_ms.empty()) {
    summary.tail_prefill_latency_ms = SummarizeMetric(tail_prefill_ms);
  }
  const auto decode_ms = CollectOptionalMetrics(
      iterations,
      [](const IterationMetrics& metrics) { return metrics.first_token_decode_latency_ms; });
  if (!decode_ms.empty()) {
    summary.first_token_decode_latency_ms = SummarizeMetric(decode_ms);
  }
  const auto commit_ms = CollectOptionalMetrics(
      iterations,
      [](const IterationMetrics& metrics) { return metrics.commit_latency_ms; });
  if (!commit_ms.empty()) {
    summary.commit_latency_ms = SummarizeMetric(commit_ms);
  }
  const auto hot_ttft_ms = CollectOptionalMetrics(
      iterations,
      [](const IterationMetrics& metrics) { return metrics.hot_prefix_ttft_ms; });
  if (!hot_ttft_ms.empty()) {
    summary.hot_prefix_ttft_ms = SummarizeMetric(hot_ttft_ms);
  }
  const auto speedup = CollectOptionalMetrics(
      iterations,
      [](const IterationMetrics& metrics) { return metrics.speedup_factor; });
  if (!speedup.empty()) {
    summary.speedup_factor = SummarizeMetric(speedup);
  }
  const auto snapshot_bytes = CollectOptionalMetrics(
      iterations,
      [](const IterationMetrics& metrics) { return metrics.prefix_snapshot_size_bytes; });
  if (!snapshot_bytes.empty()) {
    summary.prefix_snapshot_size_bytes = SummarizeMetric(snapshot_bytes);
  }
  const auto matched_tokens = CollectOptionalMetrics(
      iterations,
      [](const IterationMetrics& metrics) { return metrics.matched_prefix_token_count; });
  if (!matched_tokens.empty()) {
    summary.matched_prefix_token_count = SummarizeMetric(matched_tokens);
  }
  const auto attention_native_runs = CollectOptionalMetrics(
      iterations,
      [](const IterationMetrics& metrics) { return metrics.attention_native_multi_token_runs; });
  if (!attention_native_runs.empty()) {
    summary.attention_native_multi_token_runs = SummarizeMetric(attention_native_runs);
  }
  const auto attention_native_tokens = CollectOptionalMetrics(
      iterations,
      [](const IterationMetrics& metrics) { return metrics.attention_native_multi_token_tokens; });
  if (!attention_native_tokens.empty()) {
    summary.attention_native_multi_token_tokens = SummarizeMetric(attention_native_tokens);
  }
  const auto attention_row_replay_runs = CollectOptionalMetrics(
      iterations,
      [](const IterationMetrics& metrics) { return metrics.attention_row_replay_runs; });
  if (!attention_row_replay_runs.empty()) {
    summary.attention_row_replay_runs = SummarizeMetric(attention_row_replay_runs);
  }
  const auto attention_row_replay_tokens = CollectOptionalMetrics(
      iterations,
      [](const IterationMetrics& metrics) { return metrics.attention_row_replay_tokens; });
  if (!attention_row_replay_tokens.empty()) {
    summary.attention_row_replay_tokens = SummarizeMetric(attention_row_replay_tokens);
  }
  const auto expert_native_runs = CollectOptionalMetrics(
      iterations,
      [](const IterationMetrics& metrics) { return metrics.expert_native_multi_token_runs; });
  if (!expert_native_runs.empty()) {
    summary.expert_native_multi_token_runs = SummarizeMetric(expert_native_runs);
  }
  const auto expert_native_tokens = CollectOptionalMetrics(
      iterations,
      [](const IterationMetrics& metrics) { return metrics.expert_native_multi_token_tokens; });
  if (!expert_native_tokens.empty()) {
    summary.expert_native_multi_token_tokens = SummarizeMetric(expert_native_tokens);
  }
  const auto expert_row_replay_runs = CollectOptionalMetrics(
      iterations,
      [](const IterationMetrics& metrics) { return metrics.expert_row_replay_runs; });
  if (!expert_row_replay_runs.empty()) {
    summary.expert_row_replay_runs = SummarizeMetric(expert_row_replay_runs);
  }
  const auto expert_row_replay_tokens = CollectOptionalMetrics(
      iterations,
      [](const IterationMetrics& metrics) { return metrics.expert_row_replay_tokens; });
  if (!expert_row_replay_tokens.empty()) {
    summary.expert_row_replay_tokens = SummarizeMetric(expert_row_replay_tokens);
  }
  const auto mamba_native_runs = CollectOptionalMetrics(
      iterations,
      [](const IterationMetrics& metrics) { return metrics.mamba_native_multi_token_runs; });
  if (!mamba_native_runs.empty()) {
    summary.mamba_native_multi_token_runs = SummarizeMetric(mamba_native_runs);
  }
  const auto mamba_native_tokens = CollectOptionalMetrics(
      iterations,
      [](const IterationMetrics& metrics) { return metrics.mamba_native_multi_token_tokens; });
  if (!mamba_native_tokens.empty()) {
    summary.mamba_native_multi_token_tokens = SummarizeMetric(mamba_native_tokens);
  }
  const auto mamba_row_replay_runs = CollectOptionalMetrics(
      iterations,
      [](const IterationMetrics& metrics) { return metrics.mamba_row_replay_runs; });
  if (!mamba_row_replay_runs.empty()) {
    summary.mamba_row_replay_runs = SummarizeMetric(mamba_row_replay_runs);
  }
  const auto mamba_row_replay_tokens = CollectOptionalMetrics(
      iterations,
      [](const IterationMetrics& metrics) { return metrics.mamba_row_replay_tokens; });
  if (!mamba_row_replay_tokens.empty()) {
    summary.mamba_row_replay_tokens = SummarizeMetric(mamba_row_replay_tokens);
  }
  return summary;
}

void PrintOptionalMetric(
    const char* label,
    const std::optional<MetricSummary>& summary,
    const char* unit,
    bool integer_values = false) {
  std::cout << "  " << std::left << std::setw(28) << label << ": ";
  if (!summary.has_value()) {
    std::cout << "n/a\n";
    return;
  }
  if (integer_values) {
    std::cout << "median=" << static_cast<long long>(std::llround(summary->median))
              << unit
              << " p95=" << static_cast<long long>(std::llround(summary->p95))
              << unit
              << "\n";
    return;
  }
  std::cout << std::fixed << std::setprecision(3)
            << "median=" << summary->median << unit
            << " p95=" << summary->p95 << unit
            << "\n";
}

void PrintMetric(
    const char* label,
    const MetricSummary& summary,
    const char* unit,
    bool integer_values = false) {
  PrintOptionalMetric(label, std::optional<MetricSummary>(summary), unit, integer_values);
}

void PrintSummary(const CaseSummary& summary) {
  const std::size_t total_prompt_tokens =
      summary.spec.prefix_token_count + summary.spec.tail_token_count;
  std::cout << "\nCase: " << CaseName(summary.spec)
            << " (total_prompt_tokens=" << total_prompt_tokens << ")\n";
  PrintMetric("cold TTFT", summary.cold_ttft_ms, " ms");
  PrintOptionalMetric("restore latency", summary.restore_latency_ms, " ms");
  PrintOptionalMetric("tail prefill latency", summary.tail_prefill_latency_ms, " ms");
  PrintOptionalMetric("first-token decode latency", summary.first_token_decode_latency_ms, " ms");
  PrintOptionalMetric("commit/snapshot latency", summary.commit_latency_ms, " ms");
  PrintOptionalMetric("hot-prefix TTFT", summary.hot_prefix_ttft_ms, " ms");
  PrintOptionalMetric("speedup factor", summary.speedup_factor, "x");
  PrintOptionalMetric("prefix snapshot size", summary.prefix_snapshot_size_bytes, " bytes", true);
  PrintOptionalMetric("matched prefix token count", summary.matched_prefix_token_count, " tokens", true);
  PrintOptionalMetric(
      "attention native multi-token runs",
      summary.attention_native_multi_token_runs,
      " runs",
      true);
  PrintOptionalMetric(
      "attention native multi-token tokens",
      summary.attention_native_multi_token_tokens,
      " tokens",
      true);
  PrintOptionalMetric(
      "attention row replay runs",
      summary.attention_row_replay_runs,
      " runs",
      true);
  PrintOptionalMetric(
      "attention row replay tokens",
      summary.attention_row_replay_tokens,
      " tokens",
      true);
  PrintOptionalMetric(
      "expert native multi-token runs",
      summary.expert_native_multi_token_runs,
      " runs",
      true);
  PrintOptionalMetric(
      "expert native multi-token tokens",
      summary.expert_native_multi_token_tokens,
      " tokens",
      true);
  PrintOptionalMetric(
      "expert row replay runs",
      summary.expert_row_replay_runs,
      " runs",
      true);
  PrintOptionalMetric(
      "expert row replay tokens",
      summary.expert_row_replay_tokens,
      " tokens",
      true);
  PrintOptionalMetric(
      "mamba native multi-token runs",
      summary.mamba_native_multi_token_runs,
      " runs",
      true);
  PrintOptionalMetric(
      "mamba native multi-token tokens",
      summary.mamba_native_multi_token_tokens,
      " tokens",
      true);
  PrintOptionalMetric(
      "mamba row replay runs",
      summary.mamba_row_replay_runs,
      " runs",
      true);
  PrintOptionalMetric(
      "mamba row replay tokens",
      summary.mamba_row_replay_tokens,
      " tokens",
      true);
}

}  // namespace

int main(int argc, char** argv) {
  BenchmarkOptions options;
  if (!ParseArgs(argc, argv, &options)) {
    PrintUsage(argv[0]);
    return 1;
  }

  if (!HasCudaDevice()) {
    std::cout << "nano_prefix_cache_ttft_bench: skipped (no CUDA device available)\n";
    return 0;
  }

  const char* manifest_env = std::getenv("NEMOTRON_FORWARD_MANIFEST");
  if (manifest_env == nullptr || manifest_env[0] == '\0') {
    std::cout << "nano_prefix_cache_ttft_bench: skipped (NEMOTRON_FORWARD_MANIFEST is not set)\n";
    return 0;
  }

  const std::filesystem::path manifest_path(manifest_env);
  if (!std::filesystem::exists(manifest_path)) {
    std::cout << "nano_prefix_cache_ttft_bench: skipped (manifest not found: "
              << manifest_path.string() << ")\n";
    return 0;
  }

  const auto load_result = nemotron::LoadManifestFromJsonFile(manifest_path);
  if (!load_result.ok) {
    std::cerr << "nano_prefix_cache_ttft_bench: failed to parse manifest: "
              << manifest_path << "\n";
    return 1;
  }
  if (load_result.manifest.runtime.model_id != kNanoModelId) {
    std::cerr << "nano_prefix_cache_ttft_bench: benchmark only supports Nano manifest\n";
    return 1;
  }

  const auto config_opt = ConfigForManifest(load_result.manifest);
  if (!config_opt.has_value()) {
    std::cerr << "nano_prefix_cache_ttft_bench: failed to resolve Nano runtime config\n";
    return 1;
  }

  std::vector<std::size_t> prefix_lengths = options.prefix_lengths;
  if (prefix_lengths.empty()) {
    prefix_lengths.assign(
        std::begin(kPrefixLengths),
        std::end(kPrefixLengths));
  }
  std::sort(prefix_lengths.begin(), prefix_lengths.end());
  prefix_lengths.erase(
      std::unique(prefix_lengths.begin(), prefix_lengths.end()),
      prefix_lengths.end());

  const std::size_t max_total_prompt_tokens =
      prefix_lengths.back() +
      options.tail_token_count;
  auto runtime_options =
      MakeOptions(
          max_total_prompt_tokens + kFirstDecodeTokenCount,
          options.target_active_requests);
  auto runtime_environment =
      nemotron::RuntimeEnvironment::BuildFromManifestFile(manifest_path, runtime_options);
  if (runtime_environment == nullptr) {
    std::cerr << "nano_prefix_cache_ttft_bench: runtime environment build failed\n";
    return 1;
  }
  if (!runtime_environment->prefix_cache().enabled()) {
    std::cerr << "nano_prefix_cache_ttft_bench: prefix cache disabled by runtime config\n";
    return 1;
  }

  nemotron::SingleTokenForwardConfig config = *config_opt;
  config.max_tokens =
      std::max(config.max_tokens, max_total_prompt_tokens + kFirstDecodeTokenCount);
  config.moe_prefill_window_tokens = options.moe_prefill_window_tokens;
  auto model = nemotron::SingleTokenForwardModel::Create(*runtime_environment, config);
  if (model == nullptr || !model->valid()) {
    std::cerr << "nano_prefix_cache_ttft_bench: forward model build failed\n";
    return 1;
  }

  DeviceTokenBuffer device_token_buffer;
  if (!device_token_buffer.Allocate()) {
    return 1;
  }
  const auto resolved_moe_prefill = ResolveMoePrefillSettingsForHeader(*model);
  if (!resolved_moe_prefill.has_value()) {
    return 1;
  }

  std::vector<CaseSpec> cases;
  for (const std::size_t prefix_length : prefix_lengths) {
    const CaseSpec spec{Scenario::kCold, prefix_length, 0};
    if (CaseSelected(options, spec)) {
      cases.push_back(spec);
    }
  }
  for (const std::size_t prefix_length : prefix_lengths) {
    const CaseSpec spec{Scenario::kCommittedHead, prefix_length, options.tail_token_count};
    if (CaseSelected(options, spec)) {
      cases.push_back(spec);
    }
  }
  for (const std::size_t prefix_length : prefix_lengths) {
    const CaseSpec spec{Scenario::kGlobalRoot, prefix_length, options.tail_token_count};
    if (CaseSelected(options, spec)) {
      cases.push_back(spec);
    }
  }
  if (cases.empty()) {
    std::cerr << "nano_prefix_cache_ttft_bench: no cases selected";
    if (!options.case_filters.empty()) {
      std::cerr << " (requested:";
      for (const std::string& name : options.case_filters) {
        std::cerr << " " << name;
      }
      std::cerr << ")";
    }
    std::cerr << "\n";
    return 1;
  }

  std::cout << std::unitbuf;
  std::cout << "nano_prefix_cache_ttft_bench: manifest=" << manifest_path
            << " model_id=" << load_result.manifest.runtime.model_id
            << " warmup_iterations=" << options.warmup_iterations
            << " measured_iterations=" << options.measured_iterations
            << " tail_token_count=" << options.tail_token_count
            << " target_active_requests=" << options.target_active_requests
            << " moe_prefill_window_tokens_cli=" << resolved_moe_prefill->cli_window_tokens
            << " resolved_runtime_moe_prefill_capacity_tokens="
            << resolved_moe_prefill->runtime_capacity_tokens
            << " resolved_runtime_moe_prefill_window_tokens="
            << resolved_moe_prefill->runtime_window_tokens
            << "\n";

  for (const CaseSpec& spec : cases) {
    const std::string case_name = CaseName(spec);
    std::cout << "nano_prefix_cache_ttft_bench: running " << case_name << "\n";

    const std::vector<std::int32_t> prefix_token_ids =
        MakeTokenIds(spec.prefix_token_count, spec.prefix_token_count * 13, model->config().vocab_size);
    const std::vector<std::int32_t> tail_token_ids =
        MakeTokenIds(spec.tail_token_count, spec.prefix_token_count * 29 + 7, model->config().vocab_size);

    std::vector<IterationMetrics> measurements;
    measurements.reserve(options.measured_iterations);

    const std::size_t total_iterations = options.warmup_iterations + options.measured_iterations;
    for (std::size_t iteration = 0; iteration < total_iterations; ++iteration) {
      const bool is_warmup = iteration < options.warmup_iterations;
      std::cout << "  iteration " << (iteration + 1) << "/" << total_iterations
                << (is_warmup ? " warmup" : " measure") << "\n";
      ResetLayerExecutionCounters();

      std::optional<IterationMetrics> metrics;
      if (spec.scenario == Scenario::kCold) {
        const auto cold = RunColdPass(*model, prefix_token_ids, device_token_buffer.data());
        if (!cold.has_value()) {
          return 1;
        }
        IterationMetrics cold_metrics;
        cold_metrics.cold_ttft_ms = cold->cold_ttft_ms;
        metrics = cold_metrics;
      } else {
        metrics = RunResumeIteration(
            spec.scenario,
            *model,
            runtime_environment->prefix_cache(),
            load_result.manifest.runtime.model_id,
            prefix_token_ids,
            tail_token_ids,
            iteration,
            options.skip_commit_snapshot,
            device_token_buffer.data());
        if (!metrics.has_value()) {
          return 1;
        }
      }
      RecordLayerExecutionMetrics(CollectLayerExecutionCounters(), &*metrics);

      if (!is_warmup) {
        measurements.push_back(*metrics);
      }
    }

    const CaseSummary summary = SummarizeCase(spec, measurements);
    PrintSummary(summary);
    runtime_environment->prefix_cache().Clear();
  }

  return 0;
}
