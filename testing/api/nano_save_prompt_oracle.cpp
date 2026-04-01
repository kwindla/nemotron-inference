#include "nemotron/device_tensor.h"
#include "nemotron/manifest.h"
#include "nemotron/runtime_environment.h"
#include "nemotron/single_token_forward_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
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

struct OracleOptions {
  std::optional<std::filesystem::path> output_path;
};

struct IndexedLogit {
  std::int32_t index = -1;
  float value = 0.0f;
};

struct BoundaryObservation {
  std::int32_t selected_token_id = -1;
  float max_logit = 0.0f;
  std::vector<IndexedLogit> top5;
};

struct OracleRecord {
  std::string model_id;
  std::vector<std::int32_t> prompt_token_ids;
  std::size_t prompt_token_count = 0;
  std::size_t decode_token_count = 0;
  std::int32_t boundary_token_id = -1;
  std::vector<IndexedLogit> boundary_top5;
  float boundary_max_logit = 0.0f;
  std::vector<std::int32_t> generated_token_ids;
  std::string route;
  std::string route_description;
  std::string timestamp_utc;
};

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

std::vector<IndexedLogit> TopKSummary(const std::vector<float>& logits_row, std::size_t k) {
  if (logits_row.empty() || k == 0) {
    return {};
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

  std::vector<IndexedLogit> summary;
  summary.reserve(k);
  for (std::size_t i = 0; i < k; ++i) {
    summary.push_back({
        static_cast<std::int32_t>(indices[i]),
        logits_row[indices[i]],
    });
  }
  return summary;
}

std::optional<BoundaryObservation> RunPrefillBoundary(
    nemotron::SingleTokenForwardModel& model,
    nemotron::RequestExecutionContext& request_context) {
  auto logits =
      nemotron::DeviceTensorFp32::Create({kPromptTokenCount, model.config().vocab_size});
  if (logits == nullptr || !logits->valid()) {
    return std::nullopt;
  }

  if (!model.RunPrefill(
          FixedPromptTokenIds().data(),
          FixedPromptTokenIds().size(),
          request_context,
          logits.get(),
          /*capture_layer_indices=*/{},
          /*trace=*/nullptr)) {
    return std::nullopt;
  }

  const std::vector<float> host_logits = CopyTensorToHost(*logits);
  if (host_logits.size() != logits->numel()) {
    return std::nullopt;
  }
  const std::vector<float> final_row =
      SliceRow(host_logits, FixedPromptTokenIds().size() - 1, model.config().vocab_size);
  const auto selected_token_id = ArgMaxTokenId(final_row);
  if (!selected_token_id.has_value()) {
    return std::nullopt;
  }

  BoundaryObservation observation;
  observation.selected_token_id = *selected_token_id;
  observation.top5 = TopKSummary(final_row, 5);
  if (observation.top5.empty()) {
    return std::nullopt;
  }
  observation.max_logit = observation.top5.front().value;
  return observation;
}

std::optional<std::int32_t> RunContinuationStep(
    nemotron::SingleTokenForwardModel& model,
    nemotron::RequestExecutionContext& request_context,
    std::int32_t token_id) {
  auto logits = nemotron::DeviceTensorFp32::Create({1, model.config().vocab_size});
  if (logits == nullptr || !logits->valid()) {
    return std::nullopt;
  }
  if (!model.ContinueSingleToken(token_id, request_context, logits.get())) {
    return std::nullopt;
  }
  const std::vector<float> host_logits = CopyTensorToHost(*logits);
  if (host_logits.size() != logits->numel()) {
    return std::nullopt;
  }
  return ArgMaxTokenId(host_logits);
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

std::string FormatFloat(float value) {
  std::ostringstream stream;
  stream << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
  return stream.str();
}

std::tm UtcTime(std::time_t timestamp) {
  std::tm utc_time{};
#if defined(_WIN32)
  gmtime_s(&utc_time, &timestamp);
#else
  gmtime_r(&timestamp, &utc_time);
#endif
  return utc_time;
}

std::string FormatIso8601Utc(std::chrono::system_clock::time_point timestamp) {
  const std::time_t seconds = std::chrono::system_clock::to_time_t(timestamp);
  const std::tm utc_time = UtcTime(seconds);
  std::ostringstream output;
  output << std::put_time(&utc_time, "%Y-%m-%dT%H:%M:%SZ");
  return output.str();
}

std::string FormatFilenameTimestampUtc(std::chrono::system_clock::time_point timestamp) {
  const std::time_t seconds = std::chrono::system_clock::to_time_t(timestamp);
  const std::tm utc_time = UtcTime(seconds);
  std::ostringstream output;
  output << std::put_time(&utc_time, "%Y%m%dT%H%M%SZ");
  return output.str();
}

std::filesystem::path DefaultOutputPath(std::chrono::system_clock::time_point timestamp) {
  return std::filesystem::path("artifacts") / "oracles" /
         ("nano_16_token_oracle_" + FormatFilenameTimestampUtc(timestamp) + ".json");
}

std::optional<OracleOptions> ParseArguments(int argc, char** argv) {
  OracleOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--output") {
      if (i + 1 >= argc) {
        std::cerr << "nano_save_prompt_oracle: --output requires a path\n";
        return std::nullopt;
      }
      options.output_path = std::filesystem::path(argv[++i]);
      continue;
    }
    std::cerr << "nano_save_prompt_oracle: unknown argument: " << arg << "\n";
    return std::nullopt;
  }
  return options;
}

bool WriteOracleJson(const std::filesystem::path& path, const OracleRecord& oracle) {
  if (!path.parent_path().empty()) {
    std::error_code create_error;
    std::filesystem::create_directories(path.parent_path(), create_error);
    if (create_error) {
      std::cerr << "nano_save_prompt_oracle: failed to create output directory "
                << path.parent_path().string() << ": " << create_error.message() << "\n";
      return false;
    }
  }

  std::ofstream output(path);
  if (!output) {
    std::cerr << "nano_save_prompt_oracle: failed to open output file "
              << path.string() << "\n";
    return false;
  }

  output << "{\n";
  output << "  \"model_id\": \"" << EscapeJson(oracle.model_id) << "\",\n";
  output << "  \"prompt_token_ids\": [";
  for (std::size_t i = 0; i < oracle.prompt_token_ids.size(); ++i) {
    if (i != 0) {
      output << ", ";
    }
    output << oracle.prompt_token_ids[i];
  }
  output << "],\n";
  output << "  \"prompt_token_count\": " << oracle.prompt_token_count << ",\n";
  output << "  \"decode_token_count\": " << oracle.decode_token_count << ",\n";
  output << "  \"boundary_token_id\": " << oracle.boundary_token_id << ",\n";
  output << "  \"boundary_top5\": [";
  for (std::size_t i = 0; i < oracle.boundary_top5.size(); ++i) {
    if (i != 0) {
      output << ", ";
    }
    output << "{\"index\": " << oracle.boundary_top5[i].index
           << ", \"value\": " << FormatFloat(oracle.boundary_top5[i].value) << "}";
  }
  output << "],\n";
  output << "  \"boundary_max_logit\": " << FormatFloat(oracle.boundary_max_logit) << ",\n";
  output << "  \"generated_token_ids\": [";
  for (std::size_t i = 0; i < oracle.generated_token_ids.size(); ++i) {
    if (i != 0) {
      output << ", ";
    }
    output << oracle.generated_token_ids[i];
  }
  output << "],\n";
  output << "  \"route\": \"" << EscapeJson(oracle.route) << "\",\n";
  output << "  \"route_description\": \"" << EscapeJson(oracle.route_description) << "\",\n";
  output << "  \"timestamp_utc\": \"" << EscapeJson(oracle.timestamp_utc) << "\"\n";
  output << "}\n";

  if (!output.good()) {
    std::cerr << "nano_save_prompt_oracle: failed while writing output file "
              << path.string() << "\n";
    return false;
  }
  return true;
}

bool run_nano_save_prompt_oracle(const OracleOptions& options) {
  const char* manifest_env = std::getenv("NEMOTRON_FORWARD_MANIFEST");
  if (manifest_env == nullptr || std::string(manifest_env).empty()) {
    std::cout << "nano_save_prompt_oracle: SKIP (NEMOTRON_FORWARD_MANIFEST is unset)\n";
    return true;
  }
  if (!has_cuda_device()) {
    std::cout << "nano_save_prompt_oracle: SKIP (no CUDA device available)\n";
    return true;
  }

  const std::filesystem::path manifest_path(manifest_env);
  if (!std::filesystem::exists(manifest_path)) {
    std::cout << "nano_save_prompt_oracle: SKIP (manifest path does not exist)\n";
    return true;
  }

  const auto load_result = nemotron::LoadManifestFromJsonFile(manifest_path);
  if (!expect(load_result.ok, "manifest JSON should parse")) {
    return false;
  }
  if (load_result.manifest.runtime.model_id != kNanoModelId) {
    std::cout << "nano_save_prompt_oracle: SKIP (manifest model is not Nano)\n";
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
  config.max_tokens =
      std::max<std::size_t>(config.max_tokens, kPromptTokenCount + kDecodeTokenCount);

  auto model = nemotron::SingleTokenForwardModel::Create(*environment, config);
  if (!expect(model != nullptr && model->valid(), "forward model should build")) {
    return false;
  }

  auto request_context = model->CreateRequestContext();
  if (!expect(
          request_context != nullptr && request_context->valid(),
          "Route A request context should create")) {
    return false;
  }

  ScopedRouteOverrides route_overrides(kRouteA);
  (void)route_overrides;

  const auto prefill = RunPrefillBoundary(*model, *request_context);
  if (!expect(prefill.has_value(), "Route A prefill boundary should succeed")) {
    return false;
  }

  OracleRecord oracle;
  const auto now = std::chrono::system_clock::now();
  oracle.model_id = load_result.manifest.runtime.model_id;
  oracle.prompt_token_ids = FixedPromptTokenIds();
  oracle.prompt_token_count = kPromptTokenCount;
  oracle.decode_token_count = kDecodeTokenCount;
  oracle.boundary_token_id = prefill->selected_token_id;
  oracle.boundary_top5 = prefill->top5;
  oracle.boundary_max_logit = prefill->max_logit;
  oracle.generated_token_ids = {prefill->selected_token_id};
  oracle.route = kRouteA.route_id;
  oracle.route_description = kRouteA.description;
  oracle.timestamp_utc = FormatIso8601Utc(now);

  std::int32_t token_id = prefill->selected_token_id;
  for (std::size_t token_index = 1; token_index < kDecodeTokenCount; ++token_index) {
    const auto next_token_id = RunContinuationStep(*model, *request_context, token_id);
    if (!expect(next_token_id.has_value(), "Route A continuation step should succeed")) {
      return false;
    }
    token_id = *next_token_id;
    oracle.generated_token_ids.push_back(token_id);
  }

  if (!expect(
          oracle.generated_token_ids.size() == kDecodeTokenCount,
          "generated token count should match decode token count")) {
    return false;
  }

  const std::filesystem::path output_path =
      options.output_path.value_or(DefaultOutputPath(now));
  if (!WriteOracleJson(output_path, oracle)) {
    return false;
  }

  std::cout << "nano_save_prompt_oracle: wrote oracle to "
            << output_path.string()
            << " boundary_token_id=" << oracle.boundary_token_id
            << " generated_token_count=" << oracle.generated_token_ids.size() << "\n";
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const auto options = ParseArguments(argc, argv);
  if (!options.has_value()) {
    return 1;
  }
  return run_nano_save_prompt_oracle(*options) ? 0 : 1;
}
