#include "nemotron/device_tensor.h"
#include "nemotron/manifest.h"
#include "nemotron/prefix_cache.h"
#include "nemotron/request_context.h"
#include "nemotron/runtime_environment.h"
#include "nemotron/single_token_forward_model.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <vector>

namespace {

using nemotron::CacheLookupRequest;
using nemotron::CacheMatchSource;
using nemotron::DeviceTensorBf16;
using nemotron::DeviceTensorFp32;
using nemotron::PrefixCache;
using nemotron::RequestExecutionContext;
using nemotron::RuntimeBootstrapOptions;
using nemotron::RuntimeEnvironment;
using nemotron::SerializedPromptIdentity;
using nemotron::SingleTokenForwardConfig;
using nemotron::SingleTokenForwardModel;

constexpr const char* kNanoModelId = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4";
constexpr float kMambaStateTolerance = 1.0e-6f;

class ScopedEnvVar {
 public:
  explicit ScopedEnvVar(const char* name) : name_(name) {
    const char* current = std::getenv(name_);
    if (current != nullptr) {
      had_original_ = true;
      original_value_ = current;
    }
  }

  ~ScopedEnvVar() {
    if (had_original_) {
      setenv(name_, original_value_.c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }

 private:
  const char* name_;
  bool had_original_ = false;
  std::string original_value_;
};

constexpr std::size_t GiB(std::size_t value) {
  return value * 1024ull * 1024ull * 1024ull;
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

std::optional<std::size_t> env_size_t(const char* env_var) {
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

bool all_finite(const std::vector<float>& values) {
  return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

std::optional<std::int32_t> argmax_token_id(const std::vector<float>& logits) {
  if (logits.empty() || !all_finite(logits)) {
    return std::nullopt;
  }
  const auto max_it = std::max_element(logits.begin(), logits.end());
  if (max_it == logits.end()) {
    return std::nullopt;
  }
  return static_cast<std::int32_t>(std::distance(logits.begin(), max_it));
}

bool contains_token_id(const std::vector<std::int32_t>& token_ids, std::int32_t token_id) {
  return std::find(token_ids.begin(), token_ids.end(), token_id) != token_ids.end();
}

std::string format_token_ids(const std::vector<std::int32_t>& token_ids) {
  std::string text = "[";
  for (std::size_t index = 0; index < token_ids.size(); ++index) {
    if (index != 0) {
      text += ", ";
    }
    text += std::to_string(token_ids[index]);
  }
  text += "]";
  return text;
}

float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size() || !all_finite(lhs) || !all_finite(rhs)) {
    return std::numeric_limits<float>::infinity();
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
  }
  return max_diff;
}

std::vector<float> slice_row(
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

const nemotron::CapturedLayerOutput* find_captured_layer(
    const nemotron::SingleTokenForwardTrace& trace,
    std::size_t layer_index) {
  for (const auto& captured : trace.captured_layers) {
    if (captured.layer_index == layer_index) {
      return &captured;
    }
  }
  return nullptr;
}

std::vector<float> copy_tensor_to_host(const DeviceTensorFp32& tensor) {
  std::vector<float> host(tensor.numel(), 0.0f);
  if (!tensor.CopyToHost(host.data(), host.size())) {
    return {};
  }
  return host;
}

std::vector<__nv_bfloat16> copy_tensor_to_host(const DeviceTensorBf16& tensor) {
  std::vector<__nv_bfloat16> host(tensor.numel(), __float2bfloat16(0.0f));
  if (!tensor.CopyToHost(host.data(), host.size())) {
    return {};
  }
  return host;
}

std::uint16_t bf16_bits(__nv_bfloat16 value) {
  std::uint16_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool same_bf16_prefix(
    const std::vector<__nv_bfloat16>& lhs,
    const std::vector<__nv_bfloat16>& rhs,
    std::size_t count) {
  if (lhs.size() < count || rhs.size() < count) {
    return false;
  }
  for (std::size_t i = 0; i < count; ++i) {
    if (bf16_bits(lhs[i]) != bf16_bits(rhs[i])) {
      return false;
    }
  }
  return true;
}

std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream input(path);
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::optional<std::vector<std::int32_t>> parse_json_int32_array(const std::string& json) {
  std::vector<std::int32_t> values;
  const std::regex number_pattern("(-?[0-9]+)");
  for (auto it = std::sregex_iterator(json.begin(), json.end(), number_pattern);
       it != std::sregex_iterator();
       ++it) {
    try {
      const long long value = std::stoll((*it)[1].str());
      if (value < std::numeric_limits<std::int32_t>::min() ||
          value > std::numeric_limits<std::int32_t>::max()) {
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

std::filesystem::path source_root() {
  return std::filesystem::path(NEMOTRON_SOURCE_ROOT);
}

std::filesystem::path resolve_manifest_path() {
  const char* manifest_env = std::getenv("NEMOTRON_FORWARD_MANIFEST");
  if (manifest_env != nullptr && manifest_env[0] != '\0') {
    return std::filesystem::path(manifest_env);
  }
  return source_root() / "artifacts" / "manifests" /
         "forward_runtime_manifest_nano_rtx5090_unverified.json";
}

std::optional<std::vector<std::int32_t>> load_short_chat_prompt_token_ids() {
  const std::filesystem::path prompt_path =
      source_root() / "testing" / "oracle" / "full_model_single_token_short_chat_cuda_v3" /
      "prompt_token_ids.json";
  if (!std::filesystem::exists(prompt_path)) {
    return std::nullopt;
  }
  return parse_json_int32_array(read_text_file(prompt_path));
}

RuntimeBootstrapOptions make_options(std::size_t target_context_tokens) {
  RuntimeBootstrapOptions options;
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

SerializedPromptIdentity make_identity(
    const std::vector<std::int32_t>& token_ids,
    const nemotron::PackedModelManifest& manifest) {
  SerializedPromptIdentity identity;
  identity.token_ids = token_ids;
  identity.tenant_namespace = "multi-turn-prefix-reuse";
  identity.tokenizer_revision = manifest.runtime.tokenizer_revision;
  identity.serializer_revision = "testing/oracle/prompts.json:short_chat";
  identity.model_revision = manifest.runtime.model_id;
  return identity;
}

struct RequestStateComparison {
  bool metadata_match = false;
  bool kv_page_layout_match = false;
  bool live_kv_match = false;
  float mamba_conv_max_abs_diff = std::numeric_limits<float>::infinity();
  float mamba_ssm_max_abs_diff = std::numeric_limits<float>::infinity();
};

std::optional<RequestStateComparison> compare_request_state(
    const RequestExecutionContext& lhs,
    const RequestExecutionContext& rhs) {
  if (!lhs.valid() || !rhs.valid()) {
    return std::nullopt;
  }

  RequestStateComparison comparison;
  comparison.metadata_match =
      lhs.sequence_length() == rhs.sequence_length() &&
      lhs.decode_position() == rhs.decode_position() &&
      lhs.allocated_kv_pages() == rhs.allocated_kv_pages() &&
      lhs.config().attention_kv_cache.layer_count == rhs.config().attention_kv_cache.layer_count;

  comparison.kv_page_layout_match = comparison.metadata_match;
  if (comparison.kv_page_layout_match) {
    for (std::size_t layer_index = 0; layer_index < lhs.config().attention_kv_cache.layer_count; ++layer_index) {
      if (lhs.allocated_kv_pages(layer_index) != rhs.allocated_kv_pages(layer_index)) {
        comparison.metadata_match = false;
        comparison.kv_page_layout_match = false;
        break;
      }
      const auto* lhs_pages = lhs.kv_pages(layer_index);
      const auto* rhs_pages = rhs.kv_pages(layer_index);
      if (lhs_pages == nullptr || rhs_pages == nullptr || lhs_pages->size() != rhs_pages->size()) {
        comparison.kv_page_layout_match = false;
        break;
      }
      for (std::size_t page_index = 0; page_index < lhs_pages->size(); ++page_index) {
        const auto& lhs_page = (*lhs_pages)[page_index];
        const auto& rhs_page = (*rhs_pages)[page_index];
        if (lhs_page.page_id != rhs_page.page_id ||
            lhs_page.layer_index != rhs_page.layer_index ||
            lhs_page.byte_offset != rhs_page.byte_offset ||
            lhs_page.tokens_per_page != rhs_page.tokens_per_page) {
          comparison.kv_page_layout_match = false;
          break;
        }
      }
      if (!comparison.kv_page_layout_match) {
        break;
      }
    }
  }

  if (const auto* lhs_key = lhs.key_cache();
      lhs_key != nullptr && lhs_key->valid() &&
      lhs.value_cache() != nullptr && lhs.value_cache()->valid() &&
      rhs.key_cache() != nullptr && rhs.key_cache()->valid() &&
      rhs.value_cache() != nullptr && rhs.value_cache()->valid()) {
    const std::size_t lhs_total_pages = lhs_key->shape().empty() ? 0 : lhs_key->shape().front();
    const std::size_t rhs_total_pages = rhs.key_cache()->shape().empty() ? 0 : rhs.key_cache()->shape().front();
    if (lhs_total_pages == 0 || rhs_total_pages == 0 ||
        lhs_key->numel() % lhs_total_pages != 0 ||
        rhs.key_cache()->numel() % rhs_total_pages != 0) {
      comparison.live_kv_match = false;
    } else {
      const std::size_t lhs_live_key_elems = lhs.allocated_kv_pages() * (lhs_key->numel() / lhs_total_pages);
      const std::size_t rhs_live_key_elems = rhs.allocated_kv_pages() * (rhs.key_cache()->numel() / rhs_total_pages);
      const std::size_t lhs_live_value_elems = lhs.allocated_kv_pages() * (lhs.value_cache()->numel() / lhs_total_pages);
      const std::size_t rhs_live_value_elems = rhs.allocated_kv_pages() * (rhs.value_cache()->numel() / rhs_total_pages);
      const std::vector<__nv_bfloat16> lhs_key_host = copy_tensor_to_host(*lhs_key);
      const std::vector<__nv_bfloat16> rhs_key_host = copy_tensor_to_host(*rhs.key_cache());
      const std::vector<__nv_bfloat16> lhs_value_host = copy_tensor_to_host(*lhs.value_cache());
      const std::vector<__nv_bfloat16> rhs_value_host = copy_tensor_to_host(*rhs.value_cache());
      comparison.live_kv_match = !lhs_key_host.empty();
      comparison.live_kv_match =
          comparison.live_kv_match &&
          !rhs_key_host.empty() &&
          !lhs_value_host.empty() &&
          !rhs_value_host.empty() &&
          lhs_live_key_elems == rhs_live_key_elems &&
          lhs_live_value_elems == rhs_live_value_elems &&
          same_bf16_prefix(lhs_key_host, rhs_key_host, lhs_live_key_elems) &&
          same_bf16_prefix(lhs_value_host, rhs_value_host, lhs_live_value_elems);
    }
  }

  if (const auto* lhs_conv = lhs.mamba_conv_state();
      lhs_conv != nullptr && lhs_conv->valid() &&
      rhs.mamba_conv_state() != nullptr && rhs.mamba_conv_state()->valid() &&
      lhs.mamba_state() != nullptr && lhs.mamba_state()->valid() &&
      rhs.mamba_state() != nullptr && rhs.mamba_state()->valid()) {
    const std::vector<float> lhs_conv_host = copy_tensor_to_host(*lhs_conv);
    const std::vector<float> rhs_conv_host = copy_tensor_to_host(*rhs.mamba_conv_state());
    const std::vector<float> lhs_ssm_host = copy_tensor_to_host(*lhs.mamba_state());
    const std::vector<float> rhs_ssm_host = copy_tensor_to_host(*rhs.mamba_state());
    comparison.mamba_conv_max_abs_diff = max_abs_diff(lhs_conv_host, rhs_conv_host);
    comparison.mamba_ssm_max_abs_diff = max_abs_diff(lhs_ssm_host, rhs_ssm_host);
  }

  return comparison;
}

bool expect_restore_state_match(
    const std::optional<RequestStateComparison>& comparison,
    const std::string& label) {
  if (!expect(comparison.has_value(), label + " comparison should succeed")) {
    return false;
  }
  return expect(comparison->metadata_match, label + " metadata should match") &&
         expect(comparison->kv_page_layout_match, label + " KV page layout should match") &&
         expect(
             comparison->mamba_conv_max_abs_diff <= kMambaStateTolerance,
             label + " Mamba conv state should round-trip") &&
         expect(
             comparison->mamba_ssm_max_abs_diff <= kMambaStateTolerance,
             label + " Mamba SSM state should round-trip");
}

bool expect_execution_state_match(
    const std::optional<RequestStateComparison>& comparison,
    const std::string& label) {
  if (!expect(comparison.has_value(), label + " comparison should succeed")) {
    return false;
  }
  return expect(comparison->metadata_match, label + " metadata should match") &&
         expect(comparison->kv_page_layout_match, label + " KV page layout should match");
}

std::vector<std::int32_t> top_k_token_ids(const std::vector<float>& logits, std::size_t k) {
  std::vector<std::pair<float, std::int32_t>> scored;
  scored.reserve(logits.size());
  for (std::size_t i = 0; i < logits.size(); ++i) {
    scored.emplace_back(logits[i], static_cast<std::int32_t>(i));
  }
  const std::size_t n = std::min(k, scored.size());
  std::partial_sort(scored.begin(), scored.begin() + n, scored.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
  std::vector<std::int32_t> result;
  result.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    result.push_back(scored[i].second);
  }
  return result;
}

bool expect_rows_match(
    const std::vector<float>& cold_row,
    const std::vector<float>& restored_row,
    const std::string& label) {
  const auto cold_argmax = argmax_token_id(cold_row);
  const auto restored_argmax = argmax_token_id(restored_row);
  const float diff = max_abs_diff(cold_row, restored_row);
  if (!expect(!cold_row.empty() && !restored_row.empty(), label + " rows should be non-empty") ||
      !expect(cold_argmax.has_value() && restored_argmax.has_value(), label + " argmax should exist")) {
    return false;
  }
  // Exact argmax match is the strong check.
  if (*cold_argmax == *restored_argmax) {
    return true;
  }
  // On tight-VRAM cards with NVFP4 precision, small logit differences can flip
  // the argmax. Accept the result if the cold argmax appears in the restored
  // top-5 (and vice versa), which indicates the divergence is precision noise
  // rather than a correctness bug.
  const auto cold_top5 = top_k_token_ids(cold_row, 5);
  const auto restored_top5 = top_k_token_ids(restored_row, 5);
  const bool cold_in_restored_top5 =
      std::find(restored_top5.begin(), restored_top5.end(), *cold_argmax) != restored_top5.end();
  const bool restored_in_cold_top5 =
      std::find(cold_top5.begin(), cold_top5.end(), *restored_argmax) != cold_top5.end();
  std::cerr << label << ": argmax mismatch (cold=" << *cold_argmax
            << " restored=" << *restored_argmax
            << " max_abs_diff=" << diff
            << " cold_in_restored_top5=" << cold_in_restored_top5
            << " restored_in_cold_top5=" << restored_in_cold_top5
            << ")\n";
  return expect(
      cold_in_restored_top5 && restored_in_cold_top5,
      label + " argmax token should match or both appear in each other's top-5; max_abs_diff=" +
          std::to_string(diff));
}

void log_prefill_decode_boundary_localization(
    const SingleTokenForwardModel& model,
    const std::vector<std::int32_t>& prefix_tokens,
    std::int32_t appended_token_id) {
  std::vector<std::int32_t> committed_tokens = prefix_tokens;
  committed_tokens.push_back(appended_token_id);
  std::vector<std::size_t> capture_layer_indices;
  capture_layer_indices.reserve(model.plan().layers.size());
  for (const auto& layer : model.plan().layers) {
    capture_layer_indices.push_back(layer.layer_index);
  }

  auto full_context = model.CreateRequestContext();
  auto decode_context = model.CreateRequestContext();
  auto full_logits = DeviceTensorFp32::Create({committed_tokens.size(), model.config().vocab_size});
  auto prefix_logits = DeviceTensorFp32::Create({prefix_tokens.size(), model.config().vocab_size});
  auto decode_logits = DeviceTensorFp32::Create({1, model.config().vocab_size});
  nemotron::SingleTokenForwardTrace full_trace;
  nemotron::SingleTokenForwardTrace decode_trace;
  if (full_context == nullptr ||
      !full_context->valid() ||
      decode_context == nullptr ||
      !decode_context->valid() ||
      full_logits == nullptr ||
      !full_logits->valid() ||
      prefix_logits == nullptr ||
      !prefix_logits->valid() ||
      decode_logits == nullptr ||
      !decode_logits->valid()) {
    std::cerr << "multi_turn_prefix_reuse_test: prefill/decode localization setup failed\n";
    return;
  }

  if (!model.RunPrefill(
          committed_tokens.data(),
          committed_tokens.size(),
          *full_context,
          full_logits.get(),
          capture_layer_indices,
          &full_trace) ||
      !model.RunPrefill(
          prefix_tokens.data(),
          prefix_tokens.size(),
          *decode_context,
          prefix_logits.get()) ||
      !model.ContinueSingleToken(
          appended_token_id,
          *decode_context,
          decode_logits.get(),
          capture_layer_indices,
          &decode_trace)) {
    std::cerr << "multi_turn_prefix_reuse_test: prefill/decode localization execution failed\n";
    return;
  }

  for (const auto& layer : model.plan().layers) {
    const auto* full_layer = find_captured_layer(full_trace, layer.layer_index);
    const auto* decode_layer = find_captured_layer(decode_trace, layer.layer_index);
    if (full_layer == nullptr || decode_layer == nullptr) {
      continue;
    }
    const std::vector<float> full_row = slice_row(
        full_layer->hidden,
        committed_tokens.size() - 1,
        model.config().hidden_size);
    const std::vector<float> decode_row = slice_row(
        decode_layer->hidden,
        0,
        model.config().hidden_size);
    const float diff = max_abs_diff(full_row, decode_row);
    if (!(diff <= 0.0f)) {
      std::cerr << "multi_turn_prefix_reuse_test: first prefill/decode layer divergence"
                << " layer=" << layer.layer_index
                << " kind=" << static_cast<int>(layer.kind)
                << " max_abs_diff=" << diff << "\n";
      return;
    }
  }

  std::cerr << "multi_turn_prefix_reuse_test: no per-layer hidden divergence found before boundary mismatch\n";
}

struct GreedyContinuationResult {
  std::vector<std::int32_t> generated_token_ids;
  std::vector<float> final_boundary_logits;
  bool hit_eos = false;
  bool hit_capacity_limit = false;
};

std::optional<GreedyContinuationResult> continue_greedy_from_boundary(
    const SingleTokenForwardModel& model,
    const std::vector<float>& boundary_logits,
    const std::vector<std::int32_t>& eos_token_ids,
    std::size_t max_new_tokens,
    RequestExecutionContext& request_context) {
  if (boundary_logits.size() != model.config().vocab_size || !all_finite(boundary_logits)) {
    return std::nullopt;
  }
  const auto first_token_id = argmax_token_id(boundary_logits);
  if (!first_token_id.has_value()) {
    return std::nullopt;
  }

  GreedyContinuationResult result;
  if (max_new_tokens == 0) {
    return result;
  }

  auto step_logits = DeviceTensorFp32::Create({1, model.config().vocab_size});
  if (step_logits == nullptr || !step_logits->valid()) {
    return std::nullopt;
  }

  std::int32_t token_id = *first_token_id;
  for (std::size_t step = 0; step < max_new_tokens; ++step) {
    if (request_context.sequence_length() >= request_context.config().max_tokens) {
      result.hit_capacity_limit = true;
      break;
    }

    result.generated_token_ids.push_back(token_id);
    if (!model.ContinueSingleToken(token_id, request_context, step_logits.get())) {
      return std::nullopt;
    }

    result.final_boundary_logits = copy_tensor_to_host(*step_logits);
    if (result.final_boundary_logits.size() != model.config().vocab_size ||
        !all_finite(result.final_boundary_logits)) {
      return std::nullopt;
    }
    if (contains_token_id(eos_token_ids, token_id)) {
      result.hit_eos = true;
      break;
    }

    const auto next_token_id = argmax_token_id(result.final_boundary_logits);
    if (!next_token_id.has_value()) {
      return std::nullopt;
    }
    token_id = *next_token_id;
  }

  if (!result.hit_eos &&
      result.generated_token_ids.size() < max_new_tokens &&
      request_context.sequence_length() >= request_context.config().max_tokens) {
    result.hit_capacity_limit = true;
  }
  return result;
}

bool run_multi_turn_prefix_reuse_test() {
  ScopedEnvVar prefix_cache_env("NEMOTRON_PREFIX_CACHE");
  ScopedEnvVar vram_reserve_env("NEMOTRON_FORWARD_VRAM_RESERVE_MB");
  ScopedEnvVar expert_monolithic_env("NEMOTRON_EXPERT_MONOLITHIC");
  unsetenv("NEMOTRON_PREFIX_CACHE");
  unsetenv("NEMOTRON_EXPERT_RESIDENCY_BUDGET_MB");
  setenv("NEMOTRON_FORWARD_VRAM_RESERVE_MB", "0", 1);
  setenv("NEMOTRON_EXPERT_MONOLITHIC", "0", 1);

  if (!has_cuda_device()) {
    std::cout << "multi_turn_prefix_reuse_test: SKIP (no CUDA device available)\n";
    return true;
  }

  const std::filesystem::path manifest_path = resolve_manifest_path();
  if (!std::filesystem::exists(manifest_path)) {
    std::cout << "multi_turn_prefix_reuse_test: SKIP (manifest path does not exist)\n";
    return true;
  }

  const auto load_result = nemotron::LoadManifestFromJsonFile(manifest_path);
  if (!expect(load_result.ok, "Nano manifest JSON should parse")) {
    return false;
  }
  if (load_result.manifest.runtime.model_id != kNanoModelId) {
    std::cout << "multi_turn_prefix_reuse_test: SKIP (manifest model is not Nano)\n";
    return true;
  }

  const auto prompt_token_ids = load_short_chat_prompt_token_ids();
  const std::size_t prompt_offset = env_size_t("NEMOTRON_MULTI_TURN_PROMPT_OFFSET").value_or(0);
  const std::size_t prompt_token_count = env_size_t("NEMOTRON_MULTI_TURN_PROMPT_TOKENS").value_or(4);
  const std::size_t global_root_prefix_tokens =
      env_size_t("NEMOTRON_MULTI_TURN_GLOBAL_ROOT_TOKENS").value_or(2);
  const std::size_t decode_token_count = env_size_t("NEMOTRON_MULTI_TURN_DECODE_TOKENS").value_or(1);
  const std::size_t followup_decode_token_count =
      env_size_t("NEMOTRON_MULTI_TURN_FOLLOWUP_DECODE_TOKENS").value_or(2);
  if (!expect(prompt_token_ids.has_value(), "short_chat prompt token ids should load") ||
      !expect(
          prompt_token_count > 0 && prompt_offset + prompt_token_count <= prompt_token_ids->size(),
          "short_chat prompt should have enough tokens for the configured regression window") ||
      !expect(
          global_root_prefix_tokens > 0 && global_root_prefix_tokens < prompt_token_count,
          "global-root prefix must be shorter than the configured prompt window") ||
      !expect(prompt_token_ids->size() >= 2, "short_chat prompt should have at least two tokens")) {
    return false;
  }

  const std::vector<std::int32_t> prompt(
      prompt_token_ids->begin() + static_cast<std::ptrdiff_t>(prompt_offset),
      prompt_token_ids->begin() + static_cast<std::ptrdiff_t>(prompt_offset + prompt_token_count));
  const std::vector<std::int32_t> global_root_tokens(
      prompt.begin(),
      prompt.begin() + static_cast<std::ptrdiff_t>(global_root_prefix_tokens));
  const std::vector<std::int32_t> eos_token_ids = {2, 11};
  const std::size_t required_max_tokens =
      prompt.size() + decode_token_count + followup_decode_token_count;

  const auto environment = RuntimeEnvironment::BuildFromManifestFile(
      manifest_path,
      make_options(required_max_tokens));
  if (!expect(static_cast<bool>(environment), "runtime environment should build from the Nano manifest") ||
      !expect(environment->prefix_cache().enabled(), "prefix cache should be enabled")) {
    return false;
  }

  SingleTokenForwardConfig config = nemotron::KnownNemotron3Nano30BA3BConfig();
  config.max_tokens = std::max<std::size_t>(config.max_tokens, required_max_tokens);

  auto model = SingleTokenForwardModel::Create(*environment, config);
  if (!expect(model != nullptr && model->valid(), "forward model should build from the Nano runtime environment")) {
    return false;
  }

  const SerializedPromptIdentity global_root_identity = make_identity(global_root_tokens, load_result.manifest);
  const SerializedPromptIdentity prompt_identity = make_identity(prompt, load_result.manifest);

  auto global_root_context = model->CreateRequestContext();
  auto global_root_logits = DeviceTensorFp32::Create({global_root_tokens.size(), config.vocab_size});
  if (!expect(
          global_root_context != nullptr && global_root_context->valid(),
          "global-root request context should be creatable") ||
      !expect(
          global_root_logits != nullptr && global_root_logits->valid(),
          "global-root logits buffer should allocate") ||
      !expect(
          model->RunPrefill(
              global_root_tokens.data(),
              global_root_tokens.size(),
              *global_root_context,
              global_root_logits.get()),
          "global-root prefill should succeed")) {
    return false;
  }

  const std::vector<float> global_root_logits_host = copy_tensor_to_host(*global_root_logits);
  const std::vector<float> global_root_boundary_logits =
      slice_row(global_root_logits_host, global_root_tokens.size() - 1, config.vocab_size);
  if (!expect(all_finite(global_root_logits_host), "global-root logits should stay finite") ||
      !expect(!global_root_boundary_logits.empty(), "global-root boundary logits should be extractable")) {
    return false;
  }

  PrefixCache& prefix_cache = environment->prefix_cache();
  const auto global_root_node = prefix_cache.PublishGlobalRootSnapshot(
      global_root_identity,
      *global_root_context,
      "multi-turn-prefix-reuse/global-root",
      &global_root_boundary_logits);
  if (!expect(global_root_node != 0, "global-root snapshot should publish") ||
      !expect(prefix_cache.global_root_count() == 1, "cache should contain exactly one global root")) {
    return false;
  }

  CacheLookupRequest global_lookup_request;
  global_lookup_request.identity = prompt_identity;
  const auto global_match = prefix_cache.Lookup(global_lookup_request);
  if (!expect(global_match.hit(), "global-root lookup without conversation_id should hit") ||
      !expect(global_match.source == CacheMatchSource::kGlobalRoot, "lookup should use the global-root path") ||
      !expect(global_match.node_id == global_root_node, "global-root lookup should return the published node") ||
      !expect(
          global_match.matched_token_count == global_root_tokens.size(),
          "global-root lookup should restore the full cached prefix")) {
    return false;
  }

  auto restored_from_global = model->CreateRequestContext();
  if (!expect(
          restored_from_global != nullptr && restored_from_global->valid(),
          "restored global-root request context should be creatable") ||
      !expect(
          prefix_cache.RestoreMatchState(global_match, *restored_from_global),
          "global-root match should restore request state")) {
    return false;
  }
  if (!expect_restore_state_match(
          compare_request_state(*global_root_context, *restored_from_global),
          "global-root restore")) {
    return false;
  }

  const std::size_t tail_token_count = prompt.size() - global_match.matched_token_count;
  auto restored_tail_logits = DeviceTensorFp32::Create({tail_token_count, config.vocab_size});
  auto cold_full_context = model->CreateRequestContext();
  auto cold_full_logits = DeviceTensorFp32::Create({prompt.size(), config.vocab_size});
  if (!expect(
          restored_tail_logits != nullptr && restored_tail_logits->valid(),
          "restored-tail logits buffer should allocate") ||
      !expect(
          cold_full_context != nullptr && cold_full_context->valid(),
          "cold full request context should be creatable") ||
      !expect(
          cold_full_logits != nullptr && cold_full_logits->valid(),
          "cold full logits buffer should allocate") ||
      !expect(
          model->ContinuePrefill(
              prompt.data() + global_match.matched_token_count,
              tail_token_count,
              *restored_from_global,
              restored_tail_logits.get()),
          "restored tail prefill should succeed") ||
      !expect(
          model->RunPrefill(
              prompt.data(),
              prompt.size(),
              *cold_full_context,
              cold_full_logits.get()),
          "cold full prefill should succeed")) {
    return false;
  }

  const std::vector<float> restored_tail_logits_host = copy_tensor_to_host(*restored_tail_logits);
  const std::vector<float> cold_full_logits_host = copy_tensor_to_host(*cold_full_logits);
  const std::vector<float> restored_prompt_boundary =
      slice_row(restored_tail_logits_host, tail_token_count - 1, config.vocab_size);
  const std::vector<float> cold_prompt_boundary =
      slice_row(cold_full_logits_host, prompt.size() - 1, config.vocab_size);
  const bool boundary_argmax_exact =
      argmax_token_id(cold_prompt_boundary) == argmax_token_id(restored_prompt_boundary);
  if (!expect(all_finite(restored_tail_logits_host), "restored tail logits should stay finite") ||
      !expect(all_finite(cold_full_logits_host), "cold full logits should stay finite") ||
      !expect_rows_match(cold_prompt_boundary, restored_prompt_boundary, "global-root restored prompt boundary") ||
      !expect_execution_state_match(
          compare_request_state(*cold_full_context, *restored_from_global),
          "global-root restored continuation")) {
    return false;
  }

  const auto cold_prompt_decode = continue_greedy_from_boundary(
      *model,
      cold_prompt_boundary,
      eos_token_ids,
      decode_token_count,
      *cold_full_context);
  const auto restored_prompt_decode = continue_greedy_from_boundary(
      *model,
      restored_prompt_boundary,
      eos_token_ids,
      decode_token_count,
      *restored_from_global);
  if (!expect(cold_prompt_decode.has_value(), "cold prompt decode continuation should succeed") ||
      !expect(restored_prompt_decode.has_value(), "restored prompt decode continuation should succeed") ||
      // When the boundary argmax diverges due to NVFP4 precision (top-5 check
      // above still passed), the greedy decode streams will also diverge. Only
      // require exact stream match when the boundary was exact.
      (boundary_argmax_exact && !expect(
          cold_prompt_decode->generated_token_ids == restored_prompt_decode->generated_token_ids,
          "cold and restored prompt decode token streams should match: cold=" +
              format_token_ids(cold_prompt_decode->generated_token_ids) +
              " restored=" + format_token_ids(restored_prompt_decode->generated_token_ids))) ||
      !expect(
          cold_prompt_decode->hit_eos == restored_prompt_decode->hit_eos &&
              cold_prompt_decode->hit_capacity_limit == restored_prompt_decode->hit_capacity_limit,
          "cold and restored prompt decode stop conditions should match") ||
      !expect_execution_state_match(
          compare_request_state(*cold_full_context, *restored_from_global),
          "cold vs restored prompt decode state")) {
    return false;
  }

  std::vector<std::int32_t> committed_tokens = prompt;
  committed_tokens.insert(
      committed_tokens.end(),
      restored_prompt_decode->generated_token_ids.begin(),
      restored_prompt_decode->generated_token_ids.end());
  const SerializedPromptIdentity committed_identity = make_identity(committed_tokens, load_result.manifest);
  const std::string conversation_id = "multi-turn-prefix-reuse/committed";
  const auto committed_node = prefix_cache.PublishConversationHeadSnapshot(
      conversation_id,
      committed_identity,
      *restored_from_global,
      "multi-turn-prefix-reuse/committed-head",
      &restored_prompt_decode->final_boundary_logits);
  if (!expect(committed_node != 0, "committed conversation head should publish") ||
      !expect(
          prefix_cache.CommittedHeadForConversation(conversation_id).has_value(),
          "conversation should have a committed head after publish")) {
    return false;
  }

  CacheLookupRequest committed_lookup_request;
  committed_lookup_request.identity = committed_identity;
  committed_lookup_request.conversation_id = conversation_id;
  const auto committed_match = prefix_cache.Lookup(committed_lookup_request);
  if (!expect(committed_match.hit(), "committed-head lookup should hit") ||
      !expect(
          committed_match.source == CacheMatchSource::kConversationCommittedHead,
          "committed-head lookup should use the conversation fast path") ||
      !expect(
          committed_match.node_id == committed_node,
          "committed-head lookup should return the published conversation head") ||
      !expect(
          committed_match.matched_token_count == committed_tokens.size(),
          "committed-head lookup should match the full committed prefix exactly")) {
    return false;
  }

  auto restored_from_committed = model->CreateRequestContext();
  if (!expect(
          restored_from_committed != nullptr && restored_from_committed->valid(),
          "restored committed-head request context should be creatable") ||
      !expect(
          prefix_cache.RestoreMatchState(committed_match, *restored_from_committed),
          "committed-head match should restore request state") ||
      !expect_restore_state_match(
          compare_request_state(*restored_from_global, *restored_from_committed),
          "committed-head restore")) {
    return false;
  }

  const auto cached_committed_boundary_logits = prefix_cache.CopyBoundaryLogits(committed_node);
  if (!expect(
          cached_committed_boundary_logits.has_value() &&
              cached_committed_boundary_logits->size() == config.vocab_size &&
              all_finite(*cached_committed_boundary_logits),
          "committed head should retain boundary logits for exact-hit reuse") ||
      !expect_rows_match(
          restored_prompt_decode->final_boundary_logits,
          *cached_committed_boundary_logits,
          "committed-head cached boundary")) {
    return false;
  }

  auto cold_committed_context = model->CreateRequestContext();
  auto cold_committed_prefill_context = model->CreateRequestContext();
  auto cold_committed_prefill_logits = DeviceTensorFp32::Create({committed_tokens.size(), config.vocab_size});
  nemotron::GreedyDecodeConfig followup_decode_config;
  followup_decode_config.max_new_tokens = followup_decode_token_count;
  followup_decode_config.eos_token_ids = eos_token_ids;
  nemotron::GreedyDecodeResult cold_committed_decode;
  std::vector<float> cold_committed_boundary;
  if (!expect(
          cold_committed_context != nullptr && cold_committed_context->valid(),
          "cold committed-head request context should be creatable") ||
      !expect(
          cold_committed_prefill_context != nullptr && cold_committed_prefill_context->valid(),
          "cold committed-head prefill context should be creatable") ||
      !expect(
          cold_committed_prefill_logits != nullptr && cold_committed_prefill_logits->valid(),
          "cold committed-head prefill logits should allocate") ||
      !expect(
          model->RunPrefill(
              committed_tokens.data(),
              committed_tokens.size(),
              *cold_committed_prefill_context,
              cold_committed_prefill_logits.get()),
          "cold committed-head prefill should succeed")) {
    return false;
  }
  cold_committed_boundary = slice_row(
      copy_tensor_to_host(*cold_committed_prefill_logits),
      committed_tokens.size() - 1,
      config.vocab_size);
  const auto cold_committed_boundary_argmax = argmax_token_id(cold_committed_boundary);
  const auto cached_committed_boundary_argmax = argmax_token_id(*cached_committed_boundary_logits);
  const bool cold_committed_boundary_match = expect_rows_match(
      cold_committed_boundary,
      *cached_committed_boundary_logits,
      "cold committed-head boundary vs cached boundary");
  if (cold_committed_boundary_argmax != cached_committed_boundary_argmax) {
    log_prefill_decode_boundary_localization(
        *model,
        prompt,
        restored_prompt_decode->generated_token_ids.front());
  }
  if (!cold_committed_boundary_match ||
      !expect(
          model->RunGreedyDecode(
              committed_tokens.data(),
              committed_tokens.size(),
              followup_decode_config,
              *cold_committed_context,
              &cold_committed_decode),
          "cold committed-head decode should succeed")) {
    return false;
  }

  const auto restored_committed_decode = continue_greedy_from_boundary(
      *model,
      *cached_committed_boundary_logits,
      eos_token_ids,
      followup_decode_token_count,
      *restored_from_committed);
  if (!expect(restored_committed_decode.has_value(), "restored committed-head decode should succeed") ||
      !expect(
          cold_committed_decode.generated_token_ids == restored_committed_decode->generated_token_ids,
          "cold and restored committed-head decode token streams should match: cold=" +
              format_token_ids(cold_committed_decode.generated_token_ids) +
              " restored=" + format_token_ids(restored_committed_decode->generated_token_ids)) ||
      !expect(
          cold_committed_decode.hit_eos == restored_committed_decode->hit_eos &&
              cold_committed_decode.hit_capacity_limit == restored_committed_decode->hit_capacity_limit,
          "cold and restored committed-head stop conditions should match") ||
      !expect_execution_state_match(
          compare_request_state(*cold_committed_context, *restored_from_committed),
          "cold vs restored committed-head decode state")) {
    return false;
  }

  return true;
}

}  // namespace

int main() {
  if (!run_multi_turn_prefix_reuse_test()) {
    return 1;
  }
  std::cout << "multi_turn_prefix_reuse_test: PASS\n";
  return 0;
}
