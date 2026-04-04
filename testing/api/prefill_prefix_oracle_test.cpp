#include "nemotron/device_tensor.h"
#include "nemotron/runtime_environment.h"
#include "nemotron/single_token_forward_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr float kPrefillAbsTol = 2.0e-1f;
constexpr float kPrefillRelL2Tol = 1.0e-1f;

constexpr std::size_t GiB(std::size_t value) {
  return value * 1024ull * 1024ull * 1024ull;
}

struct FixtureMetadata {
  std::size_t hidden_size = 0;
  std::size_t runtime_token_count = 0;
  std::vector<std::int32_t> runtime_token_ids;
  std::vector<std::size_t> capture_layers;
  std::size_t stop_layer = 0;
};

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

std::vector<std::uint8_t> read_file_bytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream input(path);
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::vector<float> read_float_file(const std::filesystem::path& path) {
  const std::vector<std::uint8_t> bytes = read_file_bytes(path);
  if (bytes.size() % sizeof(float) != 0) {
    return {};
  }
  std::vector<float> values(bytes.size() / sizeof(float), 0.0f);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

std::optional<std::size_t> parse_json_uint_field(const std::string& json, const std::string& key) {
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

std::optional<std::vector<std::size_t>> parse_json_uint_array(const std::string& json, const std::string& key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
  std::smatch match;
  if (!std::regex_search(json, match, pattern) || match.size() != 2) {
    return std::nullopt;
  }
  std::vector<std::size_t> values;
  const std::regex number_pattern("([0-9]+)");
  const std::string payload = match[1].str();
  for (auto it = std::sregex_iterator(payload.begin(), payload.end(), number_pattern);
       it != std::sregex_iterator();
       ++it) {
    try {
      values.push_back(static_cast<std::size_t>(std::stoull((*it)[1].str())));
    } catch (...) {
      return std::nullopt;
    }
  }
  return values;
}

std::optional<std::vector<std::int32_t>> parse_json_int32_array(const std::string& json, const std::string& key) {
  const auto uint_values = parse_json_uint_array(json, key);
  if (!uint_values.has_value()) {
    return std::nullopt;
  }
  std::vector<std::int32_t> values;
  values.reserve(uint_values->size());
  for (const std::size_t value : *uint_values) {
    if (value > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
      return std::nullopt;
    }
    values.push_back(static_cast<std::int32_t>(value));
  }
  return values;
}

std::optional<FixtureMetadata> load_metadata(const std::filesystem::path& root) {
  const std::string json = read_text_file(root / "metadata.json");
  FixtureMetadata metadata;
  const auto hidden_size = parse_json_uint_field(json, "hidden_size");
  const auto runtime_token_count = parse_json_uint_field(json, "runtime_token_count");
  const auto runtime_token_ids = parse_json_int32_array(json, "runtime_token_ids");
  const auto capture_layers = parse_json_uint_array(json, "capture_layers");
  const auto stop_layer = parse_json_uint_field(json, "stop_layer");
  if (!hidden_size || !runtime_token_count || !runtime_token_ids || !capture_layers || !stop_layer) {
    return std::nullopt;
  }
  metadata.hidden_size = *hidden_size;
  metadata.runtime_token_count = *runtime_token_count;
  metadata.runtime_token_ids = std::move(*runtime_token_ids);
  metadata.capture_layers = std::move(*capture_layers);
  metadata.stop_layer = *stop_layer;
  return metadata;
}

float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return INFINITY;
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
  }
  return max_diff;
}

float relative_l2_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size() || lhs.empty()) {
    return INFINITY;
  }
  double diff_sq = 0.0;
  double ref_sq = 0.0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    const double diff = static_cast<double>(lhs[i]) - static_cast<double>(rhs[i]);
    diff_sq += diff * diff;
    ref_sq += static_cast<double>(rhs[i]) * static_cast<double>(rhs[i]);
  }
  if (ref_sq == 0.0) {
    return diff_sq == 0.0 ? 0.0f : INFINITY;
  }
  return static_cast<float>(std::sqrt(diff_sq / ref_sq));
}

nemotron::RuntimeBootstrapOptions make_options() {
  nemotron::RuntimeBootstrapOptions options;
  options.service_target.total_memory_bytes = GiB(128);
  options.service_target.weights_bytes = GiB(100);
  options.service_target.workspace_bytes = GiB(8);
  options.service_target.graph_bytes = GiB(4);
  options.service_target.safety_headroom_bytes = GiB(4);
  options.service_target.target_active_requests = 1;
  options.service_target.target_context_tokens = 8;
  options.use_fp16_mamba_state = false;
  options.reusable_node_metadata_bytes = 4096;
  options.verify_manifest_files = false;
  options.materialize_weight_arena = false;
  return options;
}

bool run_prefill_prefix_oracle() {
  const bool debug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  const char* manifest_env = std::getenv("NEMOTRON_FORWARD_MANIFEST");
  if (manifest_env == nullptr || std::string(manifest_env).empty()) {
    std::cout << "prefill_prefix_oracle_test: SKIP (NEMOTRON_FORWARD_MANIFEST is unset)\n";
    return true;
  }
  if (!has_cuda_device()) {
    std::cout << "prefill_prefix_oracle_test: SKIP (no CUDA device available)\n";
    return true;
  }

  const std::filesystem::path fixture_root(NEMOTRON_PREFILL_PREFIX_ORACLE_FIXTURE_ROOT);
  const auto metadata = load_metadata(fixture_root);
  if (!expect(metadata.has_value(), "prefill prefix oracle metadata should load")) {
    return false;
  }

  const std::filesystem::path manifest_path(manifest_env);
  if (!std::filesystem::exists(manifest_path)) {
    std::cout << "prefill_prefix_oracle_test: SKIP (manifest path does not exist)\n";
    return true;
  }

  const auto environment = nemotron::RuntimeEnvironment::BuildFromManifestFile(manifest_path, make_options());
  if (!expect(static_cast<bool>(environment), "runtime environment should build for prefill prefix oracle")) {
    return false;
  }

  nemotron::SingleTokenForwardConfig config =
      nemotron::KnownNemotron3Nano30BA3BConfig();
  config.max_tokens = metadata->runtime_token_count;
  auto model = nemotron::SingleTokenForwardModel::Create(*environment, config);
  if (!expect(model != nullptr && model->valid(), "forward model should build for prefill prefix oracle")) {
    return false;
  }

  auto request_context = model->CreateRequestContext();
  if (!expect(request_context != nullptr && request_context->valid(), "request context should create")) {
    return false;
  }

  auto logits = nemotron::DeviceTensorFp32::Create({metadata->runtime_token_count, config.vocab_size});
  if (!expect(logits != nullptr && logits->valid(), "logits tensor should create")) {
    return false;
  }

  nemotron::SingleTokenForwardTrace trace;
  if (!expect(
          model->RunPrefill(
              metadata->runtime_token_ids.data(),
              metadata->runtime_token_ids.size(),
              *request_context,
              logits.get(),
              metadata->capture_layers,
              &trace,
              metadata->stop_layer),
          "prefill forward path should execute against the real manifest")) {
    return false;
  }

  const std::vector<float> expected_embedding =
      read_float_file(fixture_root / "expected_embedding_output_fp32.bin");
  const float embedding_diff = max_abs_diff(trace.embedding_output, expected_embedding);
  if (debug) {
    std::cerr << "prefill_prefix_oracle_test: embedding max_abs_diff="
              << embedding_diff << "\n";
  }
  if (!expect(
          trace.embedding_output.size() == expected_embedding.size(),
          "captured embedding output size should match fixture") ||
      !expect(
          embedding_diff <= 2.0e-3f,
          "captured embedding output should match fixture; max_abs_diff=" + std::to_string(embedding_diff))) {
    return false;
  }

  std::unordered_map<std::size_t, std::vector<float>> actual_layers;
  for (const auto& captured : trace.captured_layers) {
    actual_layers.emplace(captured.layer_index, captured.hidden);
  }

  for (const std::size_t layer_index : metadata->capture_layers) {
    const auto it = actual_layers.find(layer_index);
    if (!expect(it != actual_layers.end(), "captured layer should be present")) {
      return false;
    }
    const std::filesystem::path path =
        fixture_root / ("expected_layer_" + std::to_string(layer_index / 100 % 10) +
                        std::to_string(layer_index / 10 % 10) +
                        std::to_string(layer_index % 10) + "_output_fp32.bin");
    const std::vector<float> expected = read_float_file(path);
    const float layer_diff = max_abs_diff(it->second, expected);
    const float layer_rel_l2 = relative_l2_diff(it->second, expected);
    if (debug) {
      std::cerr << "prefill_prefix_oracle_test: layer_index=" << layer_index
                << " max_abs_diff=" << layer_diff
                << " rel_l2=" << layer_rel_l2 << "\n";
    }
    if (!expect(
            it->second.size() == expected.size(),
            "captured layer size should match oracle fixture")) {
      std::cerr << "layer_index=" << layer_index
                << " actual_size=" << it->second.size()
                << " expected_size=" << expected.size() << "\n";
      return false;
    }
    if (layer_diff > kPrefillAbsTol && layer_rel_l2 > kPrefillRelL2Tol) {
      std::cerr << "prefill_prefix_oracle_test: layer_index=" << layer_index
                << " max_abs_diff=" << layer_diff
                << " rel_l2=" << layer_rel_l2 << "\n";
      expect(false, "captured layer output should match oracle fixture");
      return false;
    }
  }

  const std::vector<float> expected_final_hidden =
      read_float_file(fixture_root / "expected_final_hidden_fp32.bin");
  const float final_hidden_diff = max_abs_diff(trace.final_hidden, expected_final_hidden);
  const float final_hidden_rel_l2 = relative_l2_diff(trace.final_hidden, expected_final_hidden);
  if (debug) {
    std::cerr << "prefill_prefix_oracle_test: final_hidden max_abs_diff="
              << final_hidden_diff
              << " rel_l2=" << final_hidden_rel_l2 << "\n";
  }
  if (!expect(
          trace.final_hidden.size() == expected_final_hidden.size(),
          "captured final hidden size should match fixture")) {
    std::cerr << "final_hidden actual_size=" << trace.final_hidden.size()
              << " expected_size=" << expected_final_hidden.size() << "\n";
    return false;
  }
  if (final_hidden_diff > kPrefillAbsTol && final_hidden_rel_l2 > kPrefillRelL2Tol) {
    std::cerr << "prefill_prefix_oracle_test: final_hidden max_abs_diff="
              << final_hidden_diff
              << " rel_l2=" << final_hidden_rel_l2 << "\n";
    expect(false, "captured final hidden should match fixture");
    return false;
  }

  return true;
}

}  // namespace

int main() {
  return run_prefill_prefix_oracle() ? 0 : 1;
}
