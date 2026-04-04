#include "nemotron/device_tensor.h"
#include "nemotron/runtime_environment.h"
#include "nemotron/single_token_forward_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <regex>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kExpectedPromptTokenCount = 40;
constexpr std::int32_t kExpectedFirstGeneratedTokenId = 4268;

constexpr std::size_t GiB(std::size_t value) {
  return value * 1024ull * 1024ull * 1024ull;
}

struct PromptMatchedOracle {
  std::vector<std::int32_t> generated_token_ids;
  std::optional<std::int32_t> first_generated_token_id;
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

  const std::size_t total_context_tokens = prompt_token_ids->size() + oracle->generated_token_ids.size();
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
      nemotron::DeviceTensorFp32::Create({prompt_token_ids->size(), config.vocab_size});
  if (!expect(prefill_logits != nullptr && prefill_logits->valid(), "prefill logits tensor should create")) {
    return false;
  }

  if (!expect(
          model->RunPrefill(
              prompt_token_ids->data(),
              prompt_token_ids->size(),
              *request_context,
              prefill_logits.get()),
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
    const std::size_t nrows = prompt_token_ids->size();
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

  const std::int32_t first_predicted_token =
      argmax_last_row(prefill_logits_host, prompt_token_ids->size(), config.vocab_size);
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
    if (!expect(
            model->RunDecodeStep(current_token, *request_context, decode_logits.get()),
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
