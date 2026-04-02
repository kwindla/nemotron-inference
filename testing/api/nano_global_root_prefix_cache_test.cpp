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
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

using nemotron::CacheLookupRequest;
using nemotron::CacheMatchSource;
using nemotron::PrefixCache;
using nemotron::RequestExecutionContext;
using nemotron::RuntimeBootstrapOptions;
using nemotron::RuntimeEnvironment;
using nemotron::SerializedPromptIdentity;
using nemotron::SingleTokenForwardConfig;
using nemotron::SingleTokenForwardModel;

constexpr const char* kNanoModelId = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4";
constexpr float kLogitTolerance = 1.0e-3f;
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

std::vector<float> copy_tensor_to_host(const nemotron::DeviceTensorFp32& tensor) {
  std::vector<float> host(tensor.numel(), 0.0f);
  if (!tensor.CopyToHost(host.data(), host.size())) {
    return {};
  }
  return host;
}

std::vector<__nv_bfloat16> copy_tensor_to_host(const nemotron::DeviceTensorBf16& tensor) {
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

RuntimeBootstrapOptions make_options() {
  RuntimeBootstrapOptions options;
  options.service_target.total_memory_bytes = GiB(128);
  options.service_target.weights_bytes = GiB(100);
  options.service_target.workspace_bytes = GiB(8);
  options.service_target.graph_bytes = GiB(4);
  options.service_target.safety_headroom_bytes = GiB(4);
  options.service_target.target_active_requests = 1;
  options.service_target.target_context_tokens = 1;
  options.use_fp16_mamba_state = false;
  options.reusable_node_metadata_bytes = 4096;
  options.verify_manifest_files = false;
  options.materialize_weight_arena = false;
  return options;
}

SerializedPromptIdentity make_identity(
    const std::vector<std::int32_t>& token_ids,
    const std::string& model_id) {
  SerializedPromptIdentity identity;
  identity.token_ids = token_ids;
  identity.tenant_namespace = "nano-global-root-tenant";
  identity.tokenizer_revision = "nano-global-root-tokenizer";
  identity.serializer_revision = "nano-global-root-serializer";
  identity.model_revision = model_id;
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

struct BoundaryComparison {
  CacheMatchSource source = CacheMatchSource::kNone;
  std::size_t matched_token_count = 0;
  std::vector<float> full_row;
  std::vector<float> restored_row;
  bool request_state_match = false;
};

std::optional<BoundaryComparison> compare_global_root_resume_rows(
    const SingleTokenForwardModel& model,
    PrefixCache& prefix_cache,
    const SerializedPromptIdentity& identity,
    const std::string& conversation_id,
    const SingleTokenForwardConfig& config) {
  auto full_context = model.CreateRequestContext();
  auto restored_context = model.CreateRequestContext();
  if (full_context == nullptr || !full_context->valid() ||
      restored_context == nullptr || !restored_context->valid()) {
    return std::nullopt;
  }

  auto full_logits = nemotron::DeviceTensorFp32::Create({identity.token_ids.size(), config.vocab_size});
  if (full_logits == nullptr || !full_logits->valid()) {
    return std::nullopt;
  }

  CacheLookupRequest lookup_request;
  lookup_request.identity = identity;
  lookup_request.conversation_id = conversation_id;
  const auto match = prefix_cache.Lookup(lookup_request);
  if (!match.hit() ||
      match.source != CacheMatchSource::kGlobalRoot ||
      match.matched_token_count == 0 ||
      match.matched_token_count >= identity.token_ids.size() ||
      !prefix_cache.RestoreMatchState(match, *restored_context)) {
    return std::nullopt;
  }

  const std::size_t suffix_tokens = identity.token_ids.size() - match.matched_token_count;
  auto suffix_logits = nemotron::DeviceTensorFp32::Create({suffix_tokens, config.vocab_size});
  if (suffix_logits == nullptr || !suffix_logits->valid()) {
    return std::nullopt;
  }

  if (!model.RunPrefill(
          identity.token_ids.data(),
          identity.token_ids.size(),
          *full_context,
          full_logits.get()) ||
      !model.ContinuePrefill(
          identity.token_ids.data() + match.matched_token_count,
          suffix_tokens,
          *restored_context,
          suffix_logits.get())) {
    return std::nullopt;
  }

  const std::vector<float> full_host = copy_tensor_to_host(*full_logits);
  const std::vector<float> suffix_host = copy_tensor_to_host(*suffix_logits);
  if (!all_finite(full_host) || !all_finite(suffix_host)) {
    return std::nullopt;
  }

  const auto state_comparison = compare_request_state(*full_context, *restored_context);
  if (!state_comparison.has_value()) {
    return std::nullopt;
  }

  BoundaryComparison comparison;
  comparison.source = match.source;
  comparison.matched_token_count = match.matched_token_count;
  comparison.full_row = slice_row(full_host, identity.token_ids.size() - 1, config.vocab_size);
  comparison.restored_row = slice_row(suffix_host, suffix_tokens - 1, config.vocab_size);
  comparison.request_state_match =
      state_comparison->metadata_match &&
      state_comparison->kv_page_layout_match &&
      state_comparison->live_kv_match &&
      state_comparison->mamba_conv_max_abs_diff <= kMambaStateTolerance &&
      state_comparison->mamba_ssm_max_abs_diff <= kMambaStateTolerance;
  if (comparison.full_row.empty() || comparison.restored_row.empty()) {
    return std::nullopt;
  }
  return comparison;
}

bool run_nano_global_root_prefix_cache_test() {
  ScopedEnvVar prefix_cache_env("NEMOTRON_PREFIX_CACHE");
  unsetenv("NEMOTRON_PREFIX_CACHE");

  const char* manifest_env = std::getenv("NEMOTRON_FORWARD_MANIFEST");
  if (manifest_env == nullptr || manifest_env[0] == '\0') {
    std::cout << "nano_global_root_prefix_cache_test: SKIP (NEMOTRON_FORWARD_MANIFEST is unset)\n";
    return true;
  }
  if (!has_cuda_device()) {
    std::cout << "nano_global_root_prefix_cache_test: SKIP (no CUDA device available)\n";
    return true;
  }

  const std::filesystem::path manifest_path(manifest_env);
  if (!std::filesystem::exists(manifest_path)) {
    std::cout << "nano_global_root_prefix_cache_test: SKIP (manifest path does not exist)\n";
    return true;
  }

  const auto load_result = nemotron::LoadManifestFromJsonFile(manifest_path);
  if (!expect(load_result.ok, "Nano manifest JSON should parse")) {
    return false;
  }
  if (load_result.manifest.runtime.model_id != kNanoModelId) {
    std::cout << "nano_global_root_prefix_cache_test: SKIP (manifest model is not Nano)\n";
    return true;
  }

  const auto environment = RuntimeEnvironment::BuildFromManifestFile(manifest_path, make_options());
  if (!expect(static_cast<bool>(environment), "runtime environment should build from the Nano manifest") ||
      !expect(environment->prefix_cache().enabled(), "prefix cache should be enabled for the global-root test")) {
    return false;
  }

  SingleTokenForwardConfig config = nemotron::KnownNemotron3Nano30BA3BConfig();
  const std::vector<std::int32_t> system_tokens = {
      101, 102, 103, 104, 105, 106, 107, 108, 109,
      110, 112, 113, 114, 115, 116, 117, 118};
  const std::vector<std::int32_t> user_a_tokens = {201, 202, 203};
  const std::vector<std::int32_t> user_b_tokens = {301, 302, 303, 304};

  nemotron::GreedyDecodeConfig decode_config;
  decode_config.max_new_tokens = 2;
  decode_config.eos_token_ids = {2, 11};
  config.max_tokens = std::max<std::size_t>(
      config.max_tokens,
      system_tokens.size() + std::max(user_a_tokens.size(), user_b_tokens.size()) + decode_config.max_new_tokens + 1);

  auto model = SingleTokenForwardModel::Create(*environment, config);
  if (!expect(model != nullptr && model->valid(), "forward model should build from the Nano runtime environment")) {
    return false;
  }

  auto system_context = model->CreateRequestContext();
  auto system_logits = nemotron::DeviceTensorFp32::Create({system_tokens.size(), config.vocab_size});
  if (!expect(system_context != nullptr && system_context->valid(), "system-prefix request context should be creatable") ||
      !expect(system_logits != nullptr && system_logits->valid(), "system-prefix logits buffer should allocate") ||
      !expect(
          model->RunPrefill(system_tokens.data(), system_tokens.size(), *system_context, system_logits.get()),
          "system-prefix prefill should succeed")) {
    return false;
  }

  const std::vector<float> system_logits_host = copy_tensor_to_host(*system_logits);
  const std::vector<float> system_boundary_logits =
      slice_row(system_logits_host, system_tokens.size() - 1, config.vocab_size);
  if (!expect(all_finite(system_logits_host), "system-prefix logits should stay finite") ||
      !expect(!system_boundary_logits.empty(), "system-prefix boundary logits should be extractable")) {
    return false;
  }

  const SerializedPromptIdentity system_identity =
      make_identity(system_tokens, load_result.manifest.runtime.model_id);
  const auto global_root_node = environment->prefix_cache().PublishGlobalRootSnapshot(
      system_identity,
      *system_context,
      "nano-global-root/system",
      &system_boundary_logits);
  if (!expect(global_root_node != 0, "system-prefix prefill should publish a global root snapshot") ||
      !expect(environment->prefix_cache().global_root_count() == 1, "cache should contain one global root")) {
    return false;
  }
  const auto global_root_view = environment->prefix_cache().Describe(global_root_node);
  if (!expect(
          global_root_view.has_value() &&
              global_root_view->is_global_root &&
              global_root_view->has_boundary_logits &&
              global_root_view->boundary_logits_count == config.vocab_size &&
              global_root_view->identity.token_ids == system_tokens,
          "published node should be the system-prefix global root with boundary logits")) {
    return false;
  }

  std::vector<std::int32_t> identity_a_tokens = system_tokens;
  identity_a_tokens.insert(identity_a_tokens.end(), user_a_tokens.begin(), user_a_tokens.end());
  const SerializedPromptIdentity identity_a =
      make_identity(identity_a_tokens, load_result.manifest.runtime.model_id);
  const std::string conversation_a_id = "nano-global-root-a";

  CacheLookupRequest lookup_a_request;
  lookup_a_request.identity = identity_a;
  lookup_a_request.conversation_id = conversation_a_id;
  const auto lookup_a = environment->prefix_cache().Lookup(lookup_a_request);
  if (!expect(lookup_a.hit(), "conversation A should find a reusable prefix") ||
      !expect(lookup_a.source == CacheMatchSource::kGlobalRoot, "conversation A should reuse the shared global root") ||
      !expect(
          lookup_a.matched_token_count == system_tokens.size(),
          "conversation A should match exactly the system-prefix token count")) {
    return false;
  }

  const auto boundary_comparison =
      compare_global_root_resume_rows(*model, environment->prefix_cache(), identity_a, conversation_a_id, config);
  if (!expect(boundary_comparison.has_value(), "global-root prefill comparison should succeed")) {
    return false;
  }
  const auto full_argmax = argmax_token_id(boundary_comparison->full_row);
  const auto restored_argmax = argmax_token_id(boundary_comparison->restored_row);
  if (!expect(
          boundary_comparison->source == CacheMatchSource::kGlobalRoot,
          "prefill comparison should restore from the global root") ||
      !expect(
          boundary_comparison->matched_token_count == system_tokens.size(),
          "prefill comparison should restore the full system prefix") ||
      !expect(
          full_argmax.has_value() && restored_argmax.has_value(),
          "prefill comparison should produce valid argmax tokens") ||
      !expect(
          *full_argmax == *restored_argmax,
          "global-root restore and full-prefill should agree on the next-token argmax") ||
      !expect(
          max_abs_diff(boundary_comparison->full_row, boundary_comparison->restored_row) <= kLogitTolerance,
          "global-root restore and full-prefill logits should match within tolerance") ||
      !expect(
          boundary_comparison->request_state_match,
          "global-root restore and full-prefill request state should match")) {
    return false;
  }

  auto conversation_a_context = model->CreateRequestContext();
  if (!expect(
          conversation_a_context != nullptr && conversation_a_context->valid(),
          "conversation A request context should be creatable")) {
    return false;
  }
  nemotron::GreedyDecodeResult conversation_a_result;
  std::size_t conversation_a_match = 0;
  if (!expect(
          model->RunGreedyConversationTurn(
              identity_a,
              conversation_a_id,
              decode_config,
              *conversation_a_context,
              &conversation_a_result,
              &conversation_a_match),
          "conversation A turn should execute successfully")) {
    return false;
  }
  if (!expect(
          conversation_a_match == system_tokens.size(),
          "conversation A should reuse the global root inside RunGreedyConversationTurn") ||
      !expect(
          !conversation_a_result.generated_token_ids.empty(),
          "conversation A should generate at least one token")) {
    return false;
  }

  std::vector<std::int32_t> identity_b_tokens = system_tokens;
  identity_b_tokens.insert(identity_b_tokens.end(), user_b_tokens.begin(), user_b_tokens.end());
  const SerializedPromptIdentity identity_b =
      make_identity(identity_b_tokens, load_result.manifest.runtime.model_id);
  const std::string conversation_b_id = "nano-global-root-b";

  CacheLookupRequest lookup_b_request;
  lookup_b_request.identity = identity_b;
  lookup_b_request.conversation_id = conversation_b_id;
  const auto lookup_b = environment->prefix_cache().Lookup(lookup_b_request);
  if (!expect(lookup_b.hit(), "conversation B should find a reusable prefix") ||
      !expect(lookup_b.source == CacheMatchSource::kGlobalRoot, "conversation B should still reuse the shared global root") ||
      !expect(
          lookup_b.matched_token_count == system_tokens.size(),
          "conversation B should match exactly the system-prefix token count")) {
    return false;
  }

  auto conversation_b_context = model->CreateRequestContext();
  if (!expect(
          conversation_b_context != nullptr && conversation_b_context->valid(),
          "conversation B request context should be creatable")) {
    return false;
  }
  nemotron::GreedyDecodeResult conversation_b_result;
  std::size_t conversation_b_match = 0;
  if (!expect(
          model->RunGreedyConversationTurn(
              identity_b,
              conversation_b_id,
              decode_config,
              *conversation_b_context,
              &conversation_b_result,
              &conversation_b_match),
          "conversation B turn should execute successfully")) {
    return false;
  }
  if (!expect(
          conversation_b_match == system_tokens.size(),
          "conversation B should reuse the global root inside RunGreedyConversationTurn") ||
      !expect(
          !conversation_b_result.generated_token_ids.empty(),
          "conversation B should generate at least one token")) {
    return false;
  }
  if (!expect(
          environment->prefix_cache().CommittedHeadForConversation(conversation_a_id).has_value(),
          "conversation A should publish a committed head") ||
      !expect(
          environment->prefix_cache().CommittedHeadForConversation(conversation_b_id).has_value(),
          "conversation B should publish a committed head") ||
      !expect(
          environment->prefix_cache().global_root_count() == 1,
          "committed-head publication should not replace the shared global root")) {
    return false;
  }

  environment->prefix_cache().SetEnabled(false);
  if (!expect(!environment->prefix_cache().enabled(), "prefix cache should disable for the uncached reference run")) {
    return false;
  }

  auto conversation_a_uncached_context = model->CreateRequestContext();
  if (!expect(
          conversation_a_uncached_context != nullptr && conversation_a_uncached_context->valid(),
          "uncached conversation A request context should be creatable")) {
    return false;
  }
  nemotron::GreedyDecodeResult conversation_a_uncached_result;
  std::size_t conversation_a_uncached_match = 0;
  if (!expect(
          model->RunGreedyConversationTurn(
              identity_a,
              "nano-global-root-a-uncached",
              decode_config,
              *conversation_a_uncached_context,
              &conversation_a_uncached_result,
              &conversation_a_uncached_match),
          "uncached conversation A reference run should execute successfully")) {
    return false;
  }
  if (!expect(
          conversation_a_uncached_match == 0,
          "uncached conversation A reference run should not reuse any prefix") ||
      !expect(
          conversation_a_result.generated_token_ids == conversation_a_uncached_result.generated_token_ids,
          "cached and uncached conversation A should generate identical token ids") ||
      !expect(
          conversation_a_result.hit_eos == conversation_a_uncached_result.hit_eos &&
              conversation_a_result.hit_capacity_limit == conversation_a_uncached_result.hit_capacity_limit,
          "cached and uncached conversation A should share the same stop conditions")) {
    return false;
  }

  const auto final_state_comparison =
      compare_request_state(*conversation_a_context, *conversation_a_uncached_context);
  if (!expect(final_state_comparison.has_value(), "final request-state comparison should succeed")) {
    return false;
  }
  return expect(
             final_state_comparison->metadata_match,
             "cached and uncached conversation A should end with matching request metadata") &&
         expect(
             final_state_comparison->kv_page_layout_match,
             "cached and uncached conversation A should preserve KV page ordering") &&
         expect(
             final_state_comparison->live_kv_match,
             "cached and uncached conversation A should end with identical live KV tensors") &&
         expect(
             final_state_comparison->mamba_conv_max_abs_diff <= kMambaStateTolerance,
             "cached and uncached conversation A should end with matching Mamba conv state") &&
         expect(
             final_state_comparison->mamba_ssm_max_abs_diff <= kMambaStateTolerance,
             "cached and uncached conversation A should end with matching Mamba SSM state");
}

}  // namespace

int main() {
  if (!run_nano_global_root_prefix_cache_test()) {
    return 1;
  }
  std::cout << "nano_global_root_prefix_cache_test: PASS\n";
  return 0;
}
