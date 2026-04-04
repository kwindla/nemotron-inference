#include "nemotron/device_tensor.h"
#include "nemotron/manifest.h"
#include "nemotron/runtime_environment.h"
#include "nemotron/single_token_forward_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
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

#if !defined(_WIN32)
extern char** environ;
#endif

namespace {

constexpr const char* kNanoModelId = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4";
constexpr std::size_t kDefaultPromptTokenCount = 16;
constexpr std::size_t kDefaultDecodeTokenCount = 16;

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
};

class ScopedRouteOverrides {
 public:
  explicit ScopedRouteOverrides(const PrefillRouteConfig& route)
      : fused_mamba_(
            "NEMOTRON_FORWARD_FUSED_MAMBA_DECODE",
            route.fused_enabled ? "1" : "0"),
        fused_moe_(
            "NEMOTRON_FORWARD_FUSED_MOE_DECODE",
            route.fused_enabled ? "1" : "0") {}

  ScopedRouteOverrides(const ScopedRouteOverrides&) = delete;
  ScopedRouteOverrides& operator=(const ScopedRouteOverrides&) = delete;

 private:
  ScopedEnvOverride fused_mamba_;
  ScopedEnvOverride fused_moe_;
};

constexpr PrefillRouteConfig kRouteA = {
    "A",
    "batched prefill + reference decode",
    false,
};

struct OracleOptions {
  std::optional<std::filesystem::path> output_path;
  std::vector<std::int32_t> prompt_token_ids = FixedPromptTokenIds();
  std::size_t decode_token_count = kDefaultDecodeTokenCount;
  bool include_deep_regression_payload = false;
};

struct IndexedLogit {
  std::int32_t index = -1;
  float value = 0.0f;
};

struct BoundaryObservation {
  std::int32_t selected_token_id = -1;
  float max_logit = 0.0f;
  std::vector<IndexedLogit> top5;
  std::vector<float> boundary_logits;
};

struct BackendFlag {
  std::string name;
  std::string value;
};

struct DeepRegressionPayload {
  std::string format = "boundary_logits_fp32_v1";
  std::size_t vocab_size = 0;
  std::vector<float> boundary_logits;
};

struct OracleRecord {
  std::string model_id;
  std::string manifest_path;
  std::string build_dir;
  std::vector<BackendFlag> backend_flags;
  std::string git_revision;
  std::vector<std::int32_t> prompt_token_ids;
  std::size_t prompt_token_count = 0;
  std::size_t decode_token_count = 0;
  std::int32_t boundary_token_id = -1;
  std::vector<IndexedLogit> boundary_top5;
  float boundary_max_logit = 0.0f;
  std::vector<std::int32_t> generated_token_ids;
  std::optional<DeepRegressionPayload> deep_regression_payload;
  std::string route;
  std::string route_description;
  std::string timestamp_utc;
};

bool StartsWith(std::string_view text, std::string_view prefix) {
  return text.substr(0, prefix.size()) == prefix;
}

std::string AbsolutePathString(const std::filesystem::path& path) {
  std::error_code error;
  const std::filesystem::path absolute_path = std::filesystem::absolute(path, error);
  if (error) {
    return path.string();
  }
  return absolute_path.lexically_normal().string();
}

std::optional<std::filesystem::path> ExecutablePath() {
#if defined(__linux__)
  std::error_code error;
  const std::filesystem::path path = std::filesystem::read_symlink("/proc/self/exe", error);
  if (error) {
    return std::nullopt;
  }
  return path.lexically_normal();
#else
  return std::nullopt;
#endif
}

std::string ResolveBuildDir() {
  const char* build_dir_env = std::getenv("NEMOTRON_ORACLE_BUILD_DIR");
  if (build_dir_env != nullptr && build_dir_env[0] != '\0') {
    return AbsolutePathString(std::filesystem::path(build_dir_env));
  }
  const auto executable_path = ExecutablePath();
  if (!executable_path.has_value() || executable_path->parent_path().filename() != "testing") {
    return "";
  }
  return AbsolutePathString(executable_path->parent_path().parent_path());
}

std::vector<BackendFlag> CollectBackendFlags() {
  std::vector<BackendFlag> flags;
#if !defined(_WIN32)
  if (environ != nullptr) {
    for (char** entry = environ; *entry != nullptr; ++entry) {
      const std::string_view env_entry(*entry);
      const std::size_t separator = env_entry.find('=');
      if (separator == std::string_view::npos) {
        continue;
      }
      const std::string_view name = env_entry.substr(0, separator);
      if (!StartsWith(name, "NEMOTRON_FORWARD_")) {
        continue;
      }
      const std::string_view value = env_entry.substr(separator + 1);
      if (value.empty()) {
        continue;
      }
      flags.push_back({std::string(name), std::string(value)});
    }
  }
#endif
  std::sort(
      flags.begin(),
      flags.end(),
      [](const BackendFlag& lhs, const BackendFlag& rhs) { return lhs.name < rhs.name; });
  return flags;
}

std::string TrimWhitespace(std::string value) {
  while (!value.empty()) {
    const char ch = value.back();
    if (ch != '\n' && ch != '\r' && ch != ' ' && ch != '\t') {
      break;
    }
    value.pop_back();
  }
  return value;
}

std::string ShellQuote(std::string_view text) {
  std::string quoted = "'";
  for (const char ch : text) {
    if (ch == '\'') {
      quoted += "'\\''";
      continue;
    }
    quoted.push_back(ch);
  }
  quoted.push_back('\'');
  return quoted;
}

std::optional<std::string> RunCommandCapture(const std::string& command) {
  std::array<char, 256> buffer{};
  std::string output;
  FILE* pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) {
    return std::nullopt;
  }
  while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output.append(buffer.data());
  }
  if (pclose(pipe) != 0) {
    return std::nullopt;
  }
  return TrimWhitespace(output);
}

std::string ResolveGitRevision() {
  const char* git_revision_env = std::getenv("NEMOTRON_GIT_REVISION");
  if (git_revision_env != nullptr && git_revision_env[0] != '\0') {
    return git_revision_env;
  }

  const char* repo_root_env = std::getenv("NEMOTRON_REPO_ROOT");
  if (repo_root_env != nullptr && repo_root_env[0] != '\0') {
    const auto revision =
        RunCommandCapture("git -C " + ShellQuote(repo_root_env) + " rev-parse HEAD 2>/dev/null");
    if (revision.has_value()) {
      return *revision;
    }
  }

  const auto revision = RunCommandCapture("git rev-parse HEAD 2>/dev/null");
  return revision.value_or("");
}

nemotron::RuntimeBootstrapOptions make_options(std::size_t target_context_tokens) {
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
    nemotron::RequestExecutionContext& request_context,
    const std::vector<std::int32_t>& prompt_token_ids) {
  if (prompt_token_ids.empty()) {
    return std::nullopt;
  }

  auto logits = nemotron::DeviceTensorFp32::Create(
      {prompt_token_ids.size(), model.config().vocab_size});
  if (logits == nullptr || !logits->valid()) {
    return std::nullopt;
  }

  if (!model.RunPrefill(
          prompt_token_ids.data(),
          prompt_token_ids.size(),
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
      SliceRow(host_logits, prompt_token_ids.size() - 1, model.config().vocab_size);
  const auto selected_token_id = ArgMaxTokenId(final_row);
  if (!selected_token_id.has_value()) {
    return std::nullopt;
  }

  BoundaryObservation observation;
  observation.selected_token_id = *selected_token_id;
  observation.top5 = TopKSummary(final_row, 5);
  observation.boundary_logits = final_row;
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

std::filesystem::path DefaultOutputPath(
    std::size_t prompt_token_count,
    std::chrono::system_clock::time_point timestamp) {
  return std::filesystem::path("artifacts") / "oracles" /
         ("nano_" + std::to_string(prompt_token_count) + "_token_oracle_" +
          FormatFilenameTimestampUtc(timestamp) + ".json");
}

std::optional<std::vector<std::int32_t>> ParsePromptTokenIdsText(std::string text) {
  for (char& ch : text) {
    switch (ch) {
      case '[':
      case ']':
      case ',':
      case '\n':
      case '\r':
      case '\t':
        ch = ' ';
        break;
      default:
        break;
    }
  }

  std::istringstream stream(text);
  std::vector<std::int32_t> token_ids;
  long long token_id = 0;
  while (stream >> token_id) {
    if (token_id < 0 || token_id > std::numeric_limits<std::int32_t>::max()) {
      return std::nullopt;
    }
    token_ids.push_back(static_cast<std::int32_t>(token_id));
  }

  if (!stream.eof()) {
    return std::nullopt;
  }
  if (token_ids.empty()) {
    return std::nullopt;
  }
  return token_ids;
}

std::optional<std::vector<std::int32_t>> ReadPromptTokenIdsFile(
    const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    return std::nullopt;
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  if (!input.good() && !input.eof()) {
    return std::nullopt;
  }
  return ParsePromptTokenIdsText(buffer.str());
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
    if (arg == "--prompt-token-ids") {
      if (i + 1 >= argc) {
        std::cerr << "nano_save_prompt_oracle: --prompt-token-ids requires a value\n";
        return std::nullopt;
      }
      const auto token_ids = ParsePromptTokenIdsText(argv[++i]);
      if (!token_ids.has_value()) {
        std::cerr << "nano_save_prompt_oracle: failed to parse --prompt-token-ids\n";
        return std::nullopt;
      }
      options.prompt_token_ids = *token_ids;
      continue;
    }
    if (arg == "--prompt-token-ids-file") {
      if (i + 1 >= argc) {
        std::cerr << "nano_save_prompt_oracle: --prompt-token-ids-file requires a path\n";
        return std::nullopt;
      }
      const auto token_ids = ReadPromptTokenIdsFile(std::filesystem::path(argv[++i]));
      if (!token_ids.has_value()) {
        std::cerr << "nano_save_prompt_oracle: failed to read --prompt-token-ids-file\n";
        return std::nullopt;
      }
      options.prompt_token_ids = *token_ids;
      continue;
    }
    if (arg == "--decode-token-count") {
      if (i + 1 >= argc) {
        std::cerr << "nano_save_prompt_oracle: --decode-token-count requires a value\n";
        return std::nullopt;
      }
      try {
        options.decode_token_count = static_cast<std::size_t>(std::stoull(argv[++i]));
      } catch (...) {
        std::cerr << "nano_save_prompt_oracle: invalid --decode-token-count\n";
        return std::nullopt;
      }
      if (options.decode_token_count == 0) {
        std::cerr << "nano_save_prompt_oracle: --decode-token-count must be positive\n";
        return std::nullopt;
      }
      continue;
    }
    if (arg == "--include-deep-regression-payload") {
      options.include_deep_regression_payload = true;
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
  output << "  \"manifest_path\": \"" << EscapeJson(oracle.manifest_path) << "\",\n";
  output << "  \"build_dir\": \"" << EscapeJson(oracle.build_dir) << "\",\n";
  output << "  \"backend_flags\": {";
  for (std::size_t i = 0; i < oracle.backend_flags.size(); ++i) {
    if (i == 0) {
      output << "\n";
    } else {
      output << ",\n";
    }
    output << "    \"" << EscapeJson(oracle.backend_flags[i].name) << "\": \""
           << EscapeJson(oracle.backend_flags[i].value) << "\"";
  }
  if (oracle.backend_flags.empty()) {
    output << "},\n";
  } else {
    output << "\n  },\n";
  }
  output << "  \"git_revision\": \"" << EscapeJson(oracle.git_revision) << "\",\n";
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
  if (oracle.deep_regression_payload.has_value()) {
    output << "  \"deep_regression_payload\": {\n";
    output << "    \"format\": \""
           << EscapeJson(oracle.deep_regression_payload->format) << "\",\n";
    output << "    \"vocab_size\": " << oracle.deep_regression_payload->vocab_size << ",\n";
    output << "    \"boundary_logits\": [";
    for (std::size_t i = 0; i < oracle.deep_regression_payload->boundary_logits.size(); ++i) {
      if (i != 0) {
        output << ", ";
      }
      output << FormatFloat(oracle.deep_regression_payload->boundary_logits[i]);
    }
    output << "]\n";
    output << "  },\n";
  }
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
  if (!expect(!options.prompt_token_ids.empty(), "prompt token IDs should be non-empty")) {
    return false;
  }

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
      nemotron::RuntimeEnvironment::BuildFromManifestFile(
          manifest_path,
          make_options(options.prompt_token_ids.size() + options.decode_token_count));
  if (!expect(static_cast<bool>(environment), "runtime environment should build")) {
    return false;
  }

  const auto config_opt = config_for_manifest(load_result.manifest);
  if (!expect(config_opt.has_value(), "Nano runtime config should resolve")) {
    return false;
  }
  nemotron::SingleTokenForwardConfig config = *config_opt;
  config.max_tokens =
      std::max<std::size_t>(
          config.max_tokens,
          options.prompt_token_ids.size() + options.decode_token_count);

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

  const auto prefill = RunPrefillBoundary(*model, *request_context, options.prompt_token_ids);
  if (!expect(prefill.has_value(), "Route A prefill boundary should succeed")) {
    return false;
  }

  OracleRecord oracle;
  const auto now = std::chrono::system_clock::now();
  oracle.model_id = load_result.manifest.runtime.model_id;
  oracle.manifest_path = AbsolutePathString(manifest_path);
  oracle.build_dir = ResolveBuildDir();
  oracle.backend_flags = CollectBackendFlags();
  oracle.git_revision = ResolveGitRevision();
  oracle.prompt_token_ids = options.prompt_token_ids;
  oracle.prompt_token_count = options.prompt_token_ids.size();
  oracle.decode_token_count = options.decode_token_count;
  oracle.boundary_token_id = prefill->selected_token_id;
  oracle.boundary_top5 = prefill->top5;
  oracle.boundary_max_logit = prefill->max_logit;
  oracle.generated_token_ids = {prefill->selected_token_id};
  if (options.include_deep_regression_payload) {
    oracle.deep_regression_payload = DeepRegressionPayload{
        "boundary_logits_fp32_v1",
        model->config().vocab_size,
        prefill->boundary_logits,
    };
  }
  oracle.route = kRouteA.route_id;
  oracle.route_description = kRouteA.description;
  oracle.timestamp_utc = FormatIso8601Utc(now);

  std::int32_t token_id = prefill->selected_token_id;
  for (std::size_t token_index = 1; token_index < options.decode_token_count; ++token_index) {
    const auto next_token_id = RunContinuationStep(*model, *request_context, token_id);
    if (!expect(next_token_id.has_value(), "Route A continuation step should succeed")) {
      return false;
    }
    token_id = *next_token_id;
    oracle.generated_token_ids.push_back(token_id);
  }

  if (!expect(
          oracle.generated_token_ids.size() == options.decode_token_count,
          "generated token count should match decode token count")) {
    return false;
  }

  const std::filesystem::path output_path =
      options.output_path.value_or(DefaultOutputPath(options.prompt_token_ids.size(), now));
  if (!WriteOracleJson(output_path, oracle)) {
    return false;
  }

  std::cout << "nano_save_prompt_oracle: wrote oracle to "
            << output_path.string()
            << " prompt_token_count=" << oracle.prompt_token_count
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
