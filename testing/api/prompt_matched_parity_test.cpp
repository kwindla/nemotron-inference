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
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr std::size_t kExpectedPromptTokenCount = 40;
constexpr std::int32_t kExpectedFirstGeneratedTokenId = 4268;
constexpr float kPrefillAbsTol = 3.0e-3f;
constexpr float kPrefillRelL2Tol = 2.0e-2f;

constexpr std::size_t GiB(std::size_t value) {
  return value * 1024ull * 1024ull * 1024ull;
}

struct PromptMatchedOracle {
  std::vector<std::int32_t> generated_token_ids;
  std::optional<std::int32_t> first_generated_token_id;
};

struct PrefillFixtureMetadata {
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

std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream input(path);
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::vector<std::uint8_t> read_file_bytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
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

bool write_float_file(const std::filesystem::path& path, const std::vector<float>& values) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    return false;
  }
  output.write(
      reinterpret_cast<const char*>(values.data()),
      static_cast<std::streamsize>(values.size() * sizeof(float)));
  return output.good();
}

bool write_text_file(const std::filesystem::path& path, const std::string& text) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream output(path);
  if (!output) {
    return false;
  }
  output << text;
  return output.good();
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

std::optional<std::size_t> parse_env_uint(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || std::string(value).empty()) {
    return std::nullopt;
  }
  try {
    return static_cast<std::size_t>(std::stoull(value));
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<std::vector<std::size_t>> parse_env_uint_list(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || std::string(value).empty()) {
    return std::nullopt;
  }
  std::vector<std::size_t> values;
  std::string token;
  std::stringstream stream(value);
  while (std::getline(stream, token, ',')) {
    if (token.empty()) {
      continue;
    }
    try {
      values.push_back(static_cast<std::size_t>(std::stoull(token)));
    } catch (...) {
      return std::nullopt;
    }
  }
  return values;
}

std::optional<std::vector<std::int32_t>> parse_json_int32_array_file(const std::filesystem::path& path) {
  const std::string json = read_text_file(path);
  std::vector<std::int32_t> values;
  const std::regex number_pattern("([0-9]+)");
  for (auto it = std::sregex_iterator(json.begin(), json.end(), number_pattern);
       it != std::sregex_iterator();
       ++it) {
    try {
      const std::size_t value = static_cast<std::size_t>(std::stoull((*it)[1].str()));
      if (value > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        return std::nullopt;
      }
      values.push_back(static_cast<std::int32_t>(value));
    } catch (...) {
      return std::nullopt;
    }
  }
  if (values.empty()) {
    return std::nullopt;
  }
  return values;
}

std::optional<PromptMatchedOracle> load_prompt_matched_oracle(const std::filesystem::path& fixture_root) {
  const std::string metadata_json = read_text_file(fixture_root / "metadata.json");
  PromptMatchedOracle oracle;
  const auto generated_token_ids =
      parse_json_int32_array(metadata_json, "prompt_matched_vllm_oracle_generated_token_ids");
  const auto first_generated_token_id =
      parse_json_uint_field(metadata_json, "prompt_matched_vllm_oracle_first_generated_token_id");
  if (generated_token_ids.has_value()) {
    oracle.generated_token_ids = *generated_token_ids;
  }
  if (first_generated_token_id.has_value()) {
    if (*first_generated_token_id >
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
      return std::nullopt;
    }
    oracle.first_generated_token_id = static_cast<std::int32_t>(*first_generated_token_id);
  }
  if (oracle.generated_token_ids.empty() && !oracle.first_generated_token_id.has_value()) {
    return std::nullopt;
  }
  if (oracle.generated_token_ids.empty() && oracle.first_generated_token_id.has_value()) {
    oracle.generated_token_ids.push_back(*oracle.first_generated_token_id);
  }
  if (oracle.first_generated_token_id.has_value() &&
      oracle.generated_token_ids.front() != *oracle.first_generated_token_id) {
    return std::nullopt;
  }
  return oracle;
}

std::optional<PrefillFixtureMetadata> load_prefill_fixture_metadata(const std::filesystem::path& root) {
  const std::string metadata_json = read_text_file(root / "metadata.json");
  PrefillFixtureMetadata metadata;
  const auto hidden_size = parse_json_uint_field(metadata_json, "hidden_size");
  const auto runtime_token_count = parse_json_uint_field(metadata_json, "runtime_token_count");
  const auto runtime_token_ids = parse_json_int32_array(metadata_json, "runtime_token_ids");
  const auto capture_layers = parse_json_uint_array(metadata_json, "capture_layers");
  const auto stop_layer = parse_json_uint_field(metadata_json, "stop_layer");
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

std::int32_t argmax(const std::vector<float>& values) {
  if (values.empty()) {
    return -1;
  }
  std::size_t best_index = 0;
  for (std::size_t index = 1; index < values.size(); ++index) {
    if (values[index] > values[best_index]) {
      best_index = index;
    }
  }
  if (best_index > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    return -1;
  }
  return static_cast<std::int32_t>(best_index);
}

std::int32_t argmax_last_row(
    const std::vector<float>& logits,
    std::size_t row_count,
    std::size_t vocab_size) {
  if (row_count == 0 || vocab_size == 0 || logits.size() != row_count * vocab_size) {
    return -1;
  }
  const auto row_begin = logits.begin() + static_cast<std::ptrdiff_t>((row_count - 1) * vocab_size);
  std::size_t best_index = 0;
  float best_value = *row_begin;
  for (std::size_t index = 1; index < vocab_size; ++index) {
    const float value = *(row_begin + static_cast<std::ptrdiff_t>(index));
    if (value > best_value) {
      best_value = value;
      best_index = index;
    }
  }
  if (best_index > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    return -1;
  }
  return static_cast<std::int32_t>(best_index);
}

std::string layer_file_stem(std::size_t layer_index) {
  return std::to_string(layer_index / 100 % 10) +
         std::to_string(layer_index / 10 % 10) +
         std::to_string(layer_index % 10);
}

std::string index_file_stem(std::size_t index) {
  return std::to_string(index / 100 % 10) +
         std::to_string(index / 10 % 10) +
         std::to_string(index % 10);
}

float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return INFINITY;
  }
  float max_diff = 0.0f;
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    max_diff = std::max(max_diff, std::fabs(lhs[index] - rhs[index]));
  }
  return max_diff;
}

float relative_l2_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size() || lhs.empty()) {
    return INFINITY;
  }
  double diff_sq = 0.0;
  double ref_sq = 0.0;
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    const double diff = static_cast<double>(lhs[index]) - static_cast<double>(rhs[index]);
    diff_sq += diff * diff;
    ref_sq += static_cast<double>(rhs[index]) * static_cast<double>(rhs[index]);
  }
  if (ref_sq == 0.0) {
    return diff_sq == 0.0 ? 0.0f : INFINITY;
  }
  return static_cast<float>(std::sqrt(diff_sq / ref_sq));
}

bool dump_trace_outputs(
    const nemotron::SingleTokenForwardTrace& trace,
    const std::vector<float>& logits,
    const std::filesystem::path& dump_root) {
  std::filesystem::create_directories(dump_root);
  if (!write_float_file(dump_root / "embedding_output_fp32.bin", trace.embedding_output)) {
    return false;
  }
  for (const auto& captured : trace.captured_layers) {
    if (!write_float_file(
            dump_root / ("captured_layer_" + layer_file_stem(captured.layer_index) + "_output_fp32.bin"),
            captured.hidden)) {
      return false;
    }
  }
  if (!write_float_file(dump_root / "final_hidden_fp32.bin", trace.final_hidden)) {
    return false;
  }
  if (!write_float_file(dump_root / "final_hidden_normed_fp32.bin", trace.final_hidden_normed)) {
    return false;
  }
  if (!write_float_file(dump_root / "prefill_logits_fp32.bin", logits)) {
    return false;
  }
  if (!trace.logits.empty() && !write_float_file(dump_root / "trace_logits_fp32.bin", trace.logits)) {
    return false;
  }
  return true;
}

bool dump_decode_step_outputs(
    std::size_t generated_index,
    std::int32_t input_token_id,
    std::int32_t predicted_token_id,
    std::int32_t expected_token_id,
    const std::vector<float>& logits,
    const nemotron::SingleTokenForwardTrace* trace,
    const std::filesystem::path& dump_root) {
  std::filesystem::create_directories(dump_root);
  const std::string step_stem = index_file_stem(generated_index);
  if (!write_float_file(
          dump_root / ("decode_step_" + step_stem + "_logits_fp32.bin"),
          logits)) {
    return false;
  }
  std::ostringstream summary;
  summary << "generated_index=" << generated_index << "\n"
          << "input_token_id=" << input_token_id << "\n"
          << "predicted_token_id=" << predicted_token_id << "\n"
          << "expected_token_id=" << expected_token_id << "\n";
  if (!write_text_file(
          dump_root / ("decode_step_" + step_stem + "_summary.txt"),
          summary.str())) {
    return false;
  }
  if (trace != nullptr &&
      !dump_trace_outputs(
          *trace,
          logits,
          dump_root / ("decode_step_" + step_stem))) {
    return false;
  }
  return true;
}

bool compare_prefill_trace_against_fixture(
    const nemotron::SingleTokenForwardTrace& trace,
    const PrefillFixtureMetadata& metadata,
    const std::filesystem::path& fixture_root,
    bool debug) {
  const std::vector<float> expected_embedding =
      read_float_file(fixture_root / "expected_embedding_output_fp32.bin");
  const float embedding_diff = max_abs_diff(trace.embedding_output, expected_embedding);
  const float embedding_rel_l2 = relative_l2_diff(trace.embedding_output, expected_embedding);
  std::cerr << "prompt_matched_parity_test: compare embedding max_abs_diff="
            << embedding_diff << " rel_l2=" << embedding_rel_l2 << "\n";
  if (!expect(
          trace.embedding_output.size() == expected_embedding.size(),
          "compare fixture embedding output size should match trace")) {
    return false;
  }
  if (embedding_diff > kPrefillAbsTol && embedding_rel_l2 > kPrefillRelL2Tol) {
    expect(false, "compare fixture embedding output should match trace");
    return false;
  }

  std::unordered_map<std::size_t, std::vector<float>> actual_layers;
  for (const auto& captured : trace.captured_layers) {
    actual_layers.emplace(captured.layer_index, captured.hidden);
  }

  bool layer_ok = true;
  std::optional<std::size_t> first_failing_layer;
  float first_failing_diff = 0.0f;
  float first_failing_rel_l2 = 0.0f;
  std::size_t worst_layer = 0;
  float worst_diff = 0.0f;
  float worst_rel_l2 = 0.0f;
  for (const std::size_t layer_index : metadata.capture_layers) {
    const auto it = actual_layers.find(layer_index);
    if (!expect(it != actual_layers.end(), "compare fixture captured layer should be present")) {
      return false;
    }
    const std::vector<float> expected =
        read_float_file(fixture_root / ("expected_layer_" + layer_file_stem(layer_index) + "_output_fp32.bin"));
    const float layer_diff = max_abs_diff(it->second, expected);
    const float layer_rel_l2 = relative_l2_diff(it->second, expected);
    if (layer_diff > worst_diff ||
        (layer_diff == worst_diff && layer_rel_l2 > worst_rel_l2)) {
      worst_layer = layer_index;
      worst_diff = layer_diff;
      worst_rel_l2 = layer_rel_l2;
    }
    if (debug) {
      std::cerr << "prompt_matched_parity_test: compare layer_index=" << layer_index
                << " max_abs_diff=" << layer_diff
                << " rel_l2=" << layer_rel_l2 << "\n";
    }
    if (layer_diff > kPrefillAbsTol && layer_rel_l2 > kPrefillRelL2Tol) {
      if (!first_failing_layer.has_value()) {
        first_failing_layer = layer_index;
        first_failing_diff = layer_diff;
        first_failing_rel_l2 = layer_rel_l2;
      }
      layer_ok = false;
    }
  }
  if (!layer_ok) {
    std::cerr << "prompt_matched_parity_test: compare first_failing_layer="
              << *first_failing_layer
              << " max_abs_diff=" << first_failing_diff
              << " rel_l2=" << first_failing_rel_l2
              << " worst_layer_index=" << worst_layer
              << " worst_layer_max_abs_diff=" << worst_diff
              << " worst_layer_rel_l2=" << worst_rel_l2 << "\n";
    expect(false, "compare fixture captured layers should match trace");
    return false;
  }
  std::cerr << "prompt_matched_parity_test: compare layers ok"
            << " worst_layer_index=" << worst_layer
            << " worst_layer_max_abs_diff=" << worst_diff
            << " worst_layer_rel_l2=" << worst_rel_l2 << "\n";

  const std::vector<float> expected_final_hidden =
      read_float_file(fixture_root / "expected_final_hidden_fp32.bin");
  const float final_hidden_diff = max_abs_diff(trace.final_hidden, expected_final_hidden);
  const float final_hidden_rel_l2 = relative_l2_diff(trace.final_hidden, expected_final_hidden);
  std::cerr << "prompt_matched_parity_test: compare final_hidden max_abs_diff="
            << final_hidden_diff << " rel_l2=" << final_hidden_rel_l2 << "\n";
  if (!expect(
          trace.final_hidden.size() == expected_final_hidden.size(),
          "compare fixture final hidden size should match trace")) {
    return false;
  }
  if (final_hidden_diff > kPrefillAbsTol && final_hidden_rel_l2 > kPrefillRelL2Tol) {
    expect(false, "compare fixture final hidden should match trace");
    return false;
  }
  return true;
}

bool print_token_diagnostic(
    std::size_t generated_index,
    const char* stage,
    std::int32_t predicted_token_id,
    std::int32_t expected_token_id) {
  const bool matches = predicted_token_id == expected_token_id;
  std::cout << "prompt_matched_parity_test: generated_index=" << generated_index
            << " stage=" << stage
            << " predicted_token=" << predicted_token_id
            << " expected_token=" << expected_token_id
            << " " << (matches ? "match" : "mismatch") << "\n";
  return matches;
}

bool run_prompt_matched_parity() {
  const bool debug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  const bool trace_only = std::getenv("NEMOTRON_PROMPT_MATCHED_TRACE_ONLY") != nullptr;
  const bool dump_decode_trace =
      std::getenv("NEMOTRON_PROMPT_MATCHED_DUMP_DECODE_TRACE") != nullptr;
  const auto token_limit = parse_env_uint("NEMOTRON_PROMPT_MATCHED_TOKEN_LIMIT");
  const auto capture_layers_override = parse_env_uint_list("NEMOTRON_PROMPT_MATCHED_CAPTURE_LAYERS");
  const auto stop_layer_override = parse_env_uint("NEMOTRON_PROMPT_MATCHED_STOP_LAYER");
  const char* dump_root_env = std::getenv("NEMOTRON_PROMPT_MATCHED_DUMP_ROOT");
  const char* compare_fixture_env =
      std::getenv("NEMOTRON_PROMPT_MATCHED_COMPARE_PREFILL_FIXTURE_ROOT");
  const char* manifest_env = std::getenv("NEMOTRON_FORWARD_MANIFEST");
  if (manifest_env == nullptr || std::string(manifest_env).empty()) {
    std::cout << "prompt_matched_parity_test: SKIP (NEMOTRON_FORWARD_MANIFEST is unset)\n";
    return true;
  }
  if (!has_cuda_device()) {
    std::cout << "prompt_matched_parity_test: SKIP (no CUDA device available)\n";
    return true;
  }

  const std::filesystem::path manifest_path(manifest_env);
  if (!std::filesystem::exists(manifest_path)) {
    std::cout << "prompt_matched_parity_test: SKIP (manifest path does not exist)\n";
    return true;
  }

  const std::filesystem::path fixture_root(NEMOTRON_PROMPT_MATCHED_PARITY_FIXTURE_ROOT);
  const auto prompt_token_ids = parse_json_int32_array_file(fixture_root / "prompt_token_ids.json");
  if (!expect(prompt_token_ids.has_value(), "prompt token IDs should load")) {
    return false;
  }
  if (!expect(
          prompt_token_ids->size() == kExpectedPromptTokenCount,
          "prompt token IDs should contain the full 40-token prompt")) {
    std::cerr << "prompt_matched_parity_test: prompt_token_count=" << prompt_token_ids->size() << "\n";
    return false;
  }
  std::vector<std::int32_t> active_prompt_token_ids = *prompt_token_ids;
  if (token_limit.has_value()) {
    if (!expect(*token_limit != 0, "prompt token limit should be non-zero")) {
      return false;
    }
    if (!expect(
            *token_limit <= active_prompt_token_ids.size(),
            "prompt token limit should not exceed prompt length")) {
      return false;
    }
    active_prompt_token_ids.resize(*token_limit);
  }

  const auto oracle = load_prompt_matched_oracle(fixture_root);
  if (!expect(oracle.has_value(), "prompt-matched vLLM oracle should load")) {
    return false;
  }
  if (!expect(
          !oracle->generated_token_ids.empty(),
          "prompt-matched vLLM oracle should contain at least one generated token")) {
    return false;
  }
  if (!expect(
          oracle->generated_token_ids.front() == kExpectedFirstGeneratedTokenId,
          "first prompt-matched vLLM oracle token should match the checked-in contract")) {
    return false;
  }
  if (!trace_only &&
      !expect(
          active_prompt_token_ids.size() == prompt_token_ids->size(),
          "prompt token limit requires NEMOTRON_PROMPT_MATCHED_TRACE_ONLY=1")) {
    return false;
  }

  std::optional<PrefillFixtureMetadata> compare_fixture_metadata;
  std::filesystem::path compare_fixture_root;
  if (compare_fixture_env != nullptr && std::string(compare_fixture_env).size() != 0) {
    compare_fixture_root = std::filesystem::path(compare_fixture_env);
    compare_fixture_metadata = load_prefill_fixture_metadata(compare_fixture_root);
    if (!expect(compare_fixture_metadata.has_value(), "compare prefill fixture metadata should load")) {
      return false;
    }
    if (!expect(
            compare_fixture_metadata->runtime_token_ids == active_prompt_token_ids,
            "compare prefill fixture tokens should match active prompt prefix")) {
      return false;
    }
  }

  std::vector<std::size_t> capture_layers =
      capture_layers_override.has_value()
          ? *capture_layers_override
          : (compare_fixture_metadata.has_value() ? compare_fixture_metadata->capture_layers
                                                  : std::vector<std::size_t>{});
  const std::optional<std::size_t> effective_stop_layer =
      stop_layer_override.has_value()
          ? stop_layer_override
          : (compare_fixture_metadata.has_value()
                 ? std::optional<std::size_t>(compare_fixture_metadata->stop_layer)
                 : std::nullopt);

  const std::size_t total_context_tokens = active_prompt_token_ids.size() + oracle->generated_token_ids.size();
  const auto environment =
      nemotron::RuntimeEnvironment::BuildFromManifestFile(manifest_path, make_options(total_context_tokens));
  if (!expect(static_cast<bool>(environment), "runtime environment should build for prompt-matched parity")) {
    return false;
  }

  nemotron::SingleTokenForwardConfig config = nemotron::KnownNemotron3Super120BA12BConfig();
  config.max_tokens = total_context_tokens;
  auto model = nemotron::SingleTokenForwardModel::Create(*environment, config);
  if (!expect(model != nullptr && model->valid(), "forward model should build for prompt-matched parity")) {
    return false;
  }

  auto request_context = model->CreateRequestContext();
  if (!expect(request_context != nullptr && request_context->valid(), "request context should create")) {
    return false;
  }

  auto prefill_logits =
      nemotron::DeviceTensorFp32::Create({active_prompt_token_ids.size(), config.vocab_size});
  if (!expect(prefill_logits != nullptr && prefill_logits->valid(), "prefill logits tensor should create")) {
    return false;
  }

  nemotron::SingleTokenForwardTrace prefill_trace;
  nemotron::SingleTokenForwardTrace* prefill_trace_ptr =
      (trace_only || dump_root_env != nullptr || compare_fixture_metadata.has_value() ||
       !capture_layers.empty() || effective_stop_layer.has_value())
          ? &prefill_trace
          : nullptr;
  if (!expect(
          model->RunPrefill(
              active_prompt_token_ids.data(),
              active_prompt_token_ids.size(),
              *request_context,
              prefill_logits.get(),
              capture_layers,
              prefill_trace_ptr,
              effective_stop_layer),
          "prompt prefill should execute against the real manifest")) {
    return false;
  }

  std::vector<float> prefill_logits_host(prefill_logits->numel(), 0.0f);
  if (!expect(
          prefill_logits->CopyToHost(prefill_logits_host.data(), prefill_logits_host.size()),
          "prefill logits should download")) {
    return false;
  }

  // Diagnostic: check logits quality across rows
  {
    const std::size_t vocab = config.vocab_size;
    const std::size_t nrows = active_prompt_token_ids.size();
    for (std::size_t row : {std::size_t(0), nrows / 2, nrows - 1}) {
      if (row >= nrows) continue;
      const float* row_ptr = prefill_logits_host.data() + row * vocab;
      float max_val = row_ptr[0];
      float min_val = row_ptr[0];
      std::int32_t row_argmax = 0;
      int nan_count = 0;
      for (std::size_t i = 0; i < vocab; ++i) {
        if (row_ptr[i] != row_ptr[i]) { ++nan_count; continue; }  // NaN check
        if (row_ptr[i] > max_val) { max_val = row_ptr[i]; row_argmax = static_cast<std::int32_t>(i); }
        if (row_ptr[i] < min_val) { min_val = row_ptr[i]; }
      }
      std::cout << "prompt_matched_parity_test: logits_row=" << row
                << " argmax=" << row_argmax
                << " max=" << max_val
                << " min=" << min_val
                << " nan_count=" << nan_count << "\n";
    }
  }

  if (dump_root_env != nullptr && std::string(dump_root_env).size() != 0) {
    if (prefill_trace_ptr == nullptr) {
      expect(false, "prompt trace dump requires trace capture");
      return false;
    }
    if (!dump_trace_outputs(*prefill_trace_ptr, prefill_logits_host, std::filesystem::path(dump_root_env))) {
      expect(false, "prompt trace dump should succeed");
      return false;
    }
  }

  if (compare_fixture_metadata.has_value()) {
    if (prefill_trace_ptr == nullptr) {
      expect(false, "compare prefill fixture requires trace capture");
      return false;
    }
    if (!compare_prefill_trace_against_fixture(
            *prefill_trace_ptr,
            *compare_fixture_metadata,
            compare_fixture_root,
            debug)) {
      return false;
    }
  }

  if (trace_only) {
    std::cout << "prompt_matched_parity_test: trace_only_complete token_count="
              << active_prompt_token_ids.size();
    if (effective_stop_layer.has_value()) {
      std::cout << " stop_layer=" << *effective_stop_layer;
    }
    std::cout << "\n";
    return true;
  }

  const std::int32_t first_predicted_token =
      argmax_last_row(prefill_logits_host, active_prompt_token_ids.size(), config.vocab_size);
  if (!expect(first_predicted_token >= 0, "prefill should produce a valid first generated token")) {
    return false;
  }
  if (!print_token_diagnostic(
          0,
          "prefill",
          first_predicted_token,
          oracle->generated_token_ids.front())) {
    return false;
  }

  std::int32_t current_token = first_predicted_token;
  auto decode_logits = nemotron::DeviceTensorFp32::Create({1, config.vocab_size});
  if (!expect(decode_logits != nullptr && decode_logits->valid(), "decode logits tensor should create")) {
    return false;
  }
  for (std::size_t generated_index = 1; generated_index < oracle->generated_token_ids.size(); ++generated_index) {
    const std::int32_t input_token_for_step = current_token;
    nemotron::SingleTokenForwardTrace decode_trace;
    nemotron::SingleTokenForwardTrace* decode_trace_ptr =
        (dump_root_env != nullptr && std::string(dump_root_env).size() != 0 && dump_decode_trace)
            ? &decode_trace
            : nullptr;
    if (!expect(
            model->RunDecodeStep(
                current_token,
                *request_context,
                decode_logits.get(),
                decode_trace_ptr != nullptr ? capture_layers : std::vector<std::size_t>{},
                decode_trace_ptr,
                decode_trace_ptr != nullptr ? effective_stop_layer : std::nullopt),
            "decode step should execute against the real manifest")) {
      return false;
    }

    std::vector<float> decode_logits_host(decode_logits->numel(), 0.0f);
    if (!expect(
            decode_logits->CopyToHost(decode_logits_host.data(), decode_logits_host.size()),
            "decode logits should download")) {
      return false;
    }

    current_token = argmax(decode_logits_host);
    if (!expect(current_token >= 0, "decode step should produce a valid generated token")) {
      return false;
    }
    if (dump_root_env != nullptr && std::string(dump_root_env).size() != 0) {
      if (!dump_decode_step_outputs(
              generated_index,
              input_token_for_step,
              current_token,
              oracle->generated_token_ids[generated_index],
              decode_logits_host,
              decode_trace_ptr,
              std::filesystem::path(dump_root_env))) {
        expect(false, "prompt decode dump should succeed");
        return false;
      }
    }
    if (!print_token_diagnostic(
            generated_index,
            "decode",
            current_token,
            oracle->generated_token_ids[generated_index])) {
      return false;
    }
  }

  return true;
}

}  // namespace

int main() {
  return run_prompt_matched_parity() ? 0 : 1;
}
