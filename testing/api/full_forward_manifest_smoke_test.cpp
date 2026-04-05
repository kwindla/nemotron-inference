#include "nemotron/attention_layer.h"
#include "nemotron/artifact_loader.h"
#include "nemotron/device_tensor.h"
#include "nemotron/embedding_catalog.h"
#include "nemotron/expert_layer.h"
#include "nemotron/gemm_catalog.h"
#include "nemotron/kernel_catalog.h"
#include "nemotron/loader.h"
#include "nemotron/mamba_layer.h"
#include "nemotron/manifest.h"
#include "nemotron/model_schedule.h"
#include "nemotron/runtime_environment.h"
#include "nemotron/single_token_forward_model.h"
#include "nemotron/tensor_catalog.h"
#include "nemotron/weight_arena.h"
#include "nemotron/weight_arena_plan.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

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
  if (lhs.size() != rhs.size()) {
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

void print_issues(
    const std::string& stage,
    const std::vector<nemotron::ManifestValidationIssue>& issues) {
  if (issues.empty()) {
    std::cerr << stage << ": no issues were recorded\n";
    return;
  }
  std::cerr << stage << ": first issues:\n";
  const std::size_t max_issues = std::min<std::size_t>(issues.size(), 10);
  for (std::size_t index = 0; index < max_issues; ++index) {
    const auto& issue = issues[index];
    std::cerr << "  [" << index << "] "
              << (issue.tensor_name.empty() ? "<global>" : issue.tensor_name)
              << ": " << issue.message << "\n";
  }
}

bool has_cuda_device() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

bool should_build_forward_model() {
  const char* build_model_env = std::getenv("NEMOTRON_FORWARD_BUILD_MODEL");
  return build_model_env != nullptr && std::string(build_model_env) == "1";
}

std::optional<nemotron::SingleTokenForwardConfig> config_for_manifest(
    const nemotron::PackedModelManifest& manifest) {
  if (manifest.runtime.model_id == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4") {
    return nemotron::KnownNemotron3Nano30BA3BConfig();
  }
  if (manifest.runtime.model_id == "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4") {
    return nemotron::KnownNemotron3Super120BA12BConfig();
  }
  return std::nullopt;
}

template <typename DescriptorT, typename LookupFn>
const DescriptorT* find_global_descriptor_by_candidates(
    const std::vector<nemotron::GlobalTensorBinding>& bindings,
    nemotron::ModelGlobalRole role,
    LookupFn lookup,
    const std::vector<std::string>& candidates) {
  for (const auto& binding : bindings) {
    if (binding.role != role) {
      continue;
    }
    if (const DescriptorT* descriptor = lookup(binding.tensor_name); descriptor != nullptr) {
      return descriptor;
    }
  }
  for (const std::string& candidate : candidates) {
    if (const DescriptorT* descriptor = lookup(candidate); descriptor != nullptr) {
      return descriptor;
    }
  }
  return nullptr;
}

nemotron::RuntimeBootstrapOptions make_options() {
  nemotron::RuntimeBootstrapOptions options;
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

nemotron::SerializedPromptIdentity make_identity(
    const std::vector<std::int32_t>& token_ids,
    const std::string& model_id) {
  nemotron::SerializedPromptIdentity identity;
  identity.token_ids = token_ids;
  identity.tenant_namespace = "smoke-tenant";
  identity.tokenizer_revision = "smoke-tokenizer";
  identity.serializer_revision = "smoke-serializer";
  identity.model_revision = model_id;
  return identity;
}

bool env_enabled(const char* env_var) {
  const char* value = std::getenv(env_var);
  return value != nullptr && std::string(value) != "0";
}

struct SplitPrefillComparison {
  std::vector<float> full_row;
  std::vector<float> split_row;
  bool request_state_match = false;
};

struct BoundaryOnlyPrefillComparison {
  std::vector<float> full_row;
  std::vector<float> boundary_row;
  bool request_state_match = false;
};

std::optional<SplitPrefillComparison> compare_split_prefill_rows(
    const nemotron::SingleTokenForwardModel& model,
    const std::vector<std::int32_t>& token_ids,
    std::size_t reused_prefix_tokens,
    const nemotron::SingleTokenForwardConfig& config,
    std::optional<std::size_t> stop_layer_index = std::nullopt) {
  if (reused_prefix_tokens == 0 || reused_prefix_tokens >= token_ids.size()) {
    return std::nullopt;
  }

  const std::size_t suffix_tokens = token_ids.size() - reused_prefix_tokens;
  auto full_context = model.CreateRequestContext();
  auto split_context = model.CreateRequestContext();
  if (full_context == nullptr || !full_context->valid() ||
      split_context == nullptr || !split_context->valid()) {
    return std::nullopt;
  }

  auto full_logits = nemotron::DeviceTensorFp32::Create({token_ids.size(), config.vocab_size});
  auto prefix_logits = nemotron::DeviceTensorFp32::Create({reused_prefix_tokens, config.vocab_size});
  auto suffix_logits = nemotron::DeviceTensorFp32::Create({suffix_tokens, config.vocab_size});
  if (full_logits == nullptr || !full_logits->valid() ||
      prefix_logits == nullptr || !prefix_logits->valid() ||
      suffix_logits == nullptr || !suffix_logits->valid()) {
    return std::nullopt;
  }

  nemotron::SingleTokenForwardTrace full_trace;
  nemotron::SingleTokenForwardTrace split_trace;
  const std::vector<std::size_t> capture_layer_indices =
      stop_layer_index.has_value() ? std::vector<std::size_t>{*stop_layer_index} : std::vector<std::size_t>{};
  nemotron::SingleTokenForwardTrace* full_trace_ptr =
      stop_layer_index.has_value() ? &full_trace : nullptr;
  nemotron::SingleTokenForwardTrace* split_trace_ptr =
      stop_layer_index.has_value() ? &split_trace : nullptr;

  if (!model.RunPrefill(
          token_ids.data(),
          token_ids.size(),
          *full_context,
          full_logits.get(),
          capture_layer_indices,
          full_trace_ptr,
          stop_layer_index) ||
      !model.RunPrefill(
          token_ids.data(),
          reused_prefix_tokens,
          *split_context,
          prefix_logits.get(),
          {},
          nullptr,
          stop_layer_index) ||
      !model.ContinuePrefill(
          token_ids.data() + reused_prefix_tokens,
          suffix_tokens,
          *split_context,
          suffix_logits.get(),
          capture_layer_indices,
          split_trace_ptr,
          stop_layer_index)) {
    return std::nullopt;
  }

  SplitPrefillComparison comparison;
  comparison.request_state_match =
      full_context->sequence_length() == split_context->sequence_length() &&
      full_context->decode_position() == split_context->decode_position();

  if (stop_layer_index.has_value()) {
    if (full_trace.captured_layers.size() != 1 || split_trace.captured_layers.size() != 1) {
      return std::nullopt;
    }
    comparison.full_row = slice_row(
        full_trace.captured_layers.front().hidden,
        token_ids.size() - 1,
        config.hidden_size);
    comparison.split_row = slice_row(
        split_trace.captured_layers.front().hidden,
        suffix_tokens - 1,
        config.hidden_size);
  } else {
    const std::vector<float> full_host = copy_tensor_to_host(*full_logits);
    const std::vector<float> split_host = copy_tensor_to_host(*suffix_logits);
    comparison.full_row = slice_row(full_host, token_ids.size() - 1, config.vocab_size);
    comparison.split_row = slice_row(split_host, suffix_tokens - 1, config.vocab_size);
  }

  if (comparison.full_row.empty() || comparison.split_row.empty()) {
    return std::nullopt;
  }
  return comparison;
}

std::optional<BoundaryOnlyPrefillComparison> compare_boundary_only_prefill_row(
    const nemotron::SingleTokenForwardModel& model,
    const std::vector<std::int32_t>& token_ids,
    const nemotron::SingleTokenForwardConfig& config) {
  if (token_ids.size() <= 1) {
    return std::nullopt;
  }

  auto full_context = model.CreateRequestContext();
  auto boundary_context = model.CreateRequestContext();
  if (full_context == nullptr || !full_context->valid() ||
      boundary_context == nullptr || !boundary_context->valid()) {
    return std::nullopt;
  }

  auto full_logits = nemotron::DeviceTensorFp32::Create({token_ids.size(), config.vocab_size});
  auto boundary_logits = nemotron::DeviceTensorFp32::Create({1, config.vocab_size});
  if (full_logits == nullptr || !full_logits->valid() ||
      boundary_logits == nullptr || !boundary_logits->valid()) {
    return std::nullopt;
  }

  if (!model.RunPrefill(token_ids.data(), token_ids.size(), *full_context, full_logits.get()) ||
      !model.RunPrefill(token_ids.data(), token_ids.size(), *boundary_context, boundary_logits.get())) {
    return std::nullopt;
  }

  BoundaryOnlyPrefillComparison comparison;
  comparison.request_state_match =
      full_context->sequence_length() == boundary_context->sequence_length() &&
      full_context->decode_position() == boundary_context->decode_position();

  const std::vector<float> full_logits_host = copy_tensor_to_host(*full_logits);
  comparison.full_row = slice_row(
      full_logits_host,
      token_ids.size() - 1,
      config.vocab_size);
  comparison.boundary_row = copy_tensor_to_host(*boundary_logits);
  return comparison.full_row.empty() || comparison.boundary_row.empty()
             ? std::nullopt
             : std::optional<BoundaryOnlyPrefillComparison>(std::move(comparison));
}

void maybe_print_first_split_prefill_divergent_layer(
    const nemotron::SingleTokenForwardModel& model,
    const std::vector<std::int32_t>& token_ids,
    std::size_t reused_prefix_tokens,
    const nemotron::SingleTokenForwardConfig& config) {
  if (!env_enabled("NEMOTRON_FORWARD_TRACE_DIVERGENCE")) {
    return;
  }

  const auto& layers = model.plan().layers;
  if (layers.empty()) {
    std::cerr << "split_prefill_diagnostic: model plan has no layers\n";
    return;
  }

  std::size_t low = 0;
  std::size_t high = layers.size() - 1;
  std::optional<std::size_t> first_mismatch_index;
  while (low <= high) {
    const std::size_t mid = low + ((high - low) / 2u);
    const auto comparison =
        compare_split_prefill_rows(model, token_ids, reused_prefix_tokens, config, layers[mid].layer_index);
    if (!comparison.has_value()) {
      std::cerr << "split_prefill_diagnostic: comparison failed at layer_index="
                << layers[mid].layer_index << "\n";
      return;
    }
    const auto full_argmax = argmax_token_id(comparison->full_row);
    const auto split_argmax = argmax_token_id(comparison->split_row);
    const bool mismatch =
        !comparison->request_state_match ||
        !full_argmax.has_value() ||
        !split_argmax.has_value() ||
        *full_argmax != *split_argmax ||
        max_abs_diff(comparison->full_row, comparison->split_row) > 1.0e-3f;
    if (mismatch) {
      first_mismatch_index = mid;
      if (mid == 0) {
        break;
      }
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  }

  if (!first_mismatch_index.has_value()) {
    std::cerr << "split_prefill_diagnostic: no divergent layer found before final logits\n";
    return;
  }

  const auto& entry = layers[*first_mismatch_index];
  const auto comparison =
      compare_split_prefill_rows(model, token_ids, reused_prefix_tokens, config, entry.layer_index);
  if (!comparison.has_value()) {
    std::cerr << "split_prefill_diagnostic: failed to reproduce divergence at layer_index="
              << entry.layer_index << "\n";
    return;
  }
  const auto full_argmax = argmax_token_id(comparison->full_row);
  const auto split_argmax = argmax_token_id(comparison->split_row);
  std::cerr << "split_prefill_diagnostic: first_divergent_layer_index=" << entry.layer_index
            << " kind=" << static_cast<int>(entry.kind)
            << " max_abs_diff=" << max_abs_diff(comparison->full_row, comparison->split_row)
            << " request_state_match=" << comparison->request_state_match;
  if (full_argmax.has_value() && split_argmax.has_value()) {
    std::cerr << " full_argmax=" << *full_argmax
              << " split_argmax=" << *split_argmax;
  }
  std::cerr << "\n";
}

bool run_full_forward_manifest_smoke() {
  const char* manifest_env = std::getenv("NEMOTRON_FORWARD_MANIFEST");
  if (manifest_env == nullptr || std::string(manifest_env).empty()) {
    std::cout << "full_forward_manifest_smoke_test: SKIP (NEMOTRON_FORWARD_MANIFEST is unset)\n";
    return true;
  }
  if (!has_cuda_device()) {
    std::cout << "full_forward_manifest_smoke_test: SKIP (no CUDA device available)\n";
    return true;
  }

  const std::filesystem::path manifest_path(manifest_env);
  if (!std::filesystem::exists(manifest_path)) {
    std::cout << "full_forward_manifest_smoke_test: SKIP (manifest path does not exist)\n";
    return true;
  }

  const auto load_result = nemotron::LoadManifestFromJsonFile(manifest_path);
  if (!expect(load_result.ok, "manifest JSON should parse")) {
    print_issues("manifest_load", load_result.issues);
    return false;
  }

  auto artifact_loader = nemotron::ArtifactLoader::OpenVerifiedWithMode(
      load_result.manifest,
      manifest_path,
      make_options().artifact_load_mode);
  if (!expect(static_cast<bool>(artifact_loader), "artifact loader should open the generated manifest")) {
    return false;
  }

  const nemotron::TensorCatalog tensor_catalog =
      nemotron::BuildTensorCatalog(load_result.manifest, *artifact_loader);
  if (!expect(tensor_catalog.valid(), "tensor catalog should build from the generated manifest")) {
    print_issues("tensor_catalog", tensor_catalog.issues());
    return false;
  }

  const nemotron::KernelCatalog kernel_catalog =
      nemotron::BuildKernelCatalog(tensor_catalog);
  if (!expect(kernel_catalog.valid(), "kernel catalog should build from the generated manifest")) {
    print_issues("kernel_catalog", kernel_catalog.issues());
    return false;
  }

  const nemotron::GemmCatalog gemm_catalog = nemotron::BuildGemmCatalog(kernel_catalog);
  if (!expect(gemm_catalog.valid(), "GEMM catalog should build from the generated manifest")) {
    print_issues("gemm_catalog", gemm_catalog.issues());
    return false;
  }

  const nemotron::EmbeddingCatalog embedding_catalog =
      nemotron::BuildEmbeddingCatalog(kernel_catalog);
  if (!expect(embedding_catalog.valid(), "embedding catalog should build from the generated manifest")) {
    print_issues("embedding_catalog", embedding_catalog.issues());
    return false;
  }

  const nemotron::ModelSchedule model_schedule =
      nemotron::BuildModelSchedule(load_result.manifest);
  if (!expect(model_schedule.valid(), "model schedule should build from the generated manifest")) {
    print_issues("model_schedule", model_schedule.issues());
    return false;
  }

  auto options = make_options();
  const nemotron::LoaderPlan loader_plan = nemotron::BuildLoaderPlan(
      load_result.manifest,
      options.service_target,
      options.use_fp16_mamba_state,
      options.reusable_node_metadata_bytes);
  if (!expect(loader_plan.valid, "loader plan should build from the generated manifest")) {
    print_issues("loader_plan", loader_plan.issues);
    return false;
  }
  const auto environment =
      nemotron::RuntimeEnvironment::BuildFromManifestFile(manifest_path, options);
  if (!expect(static_cast<bool>(environment), "runtime environment should build from the forward manifest")) {
    return false;
  }
  if (!should_build_forward_model()) {
    std::cout << "full_forward_manifest_smoke_test: PASS (runtime environment built)\n";
    return true;
  }

  const auto config_opt = config_for_manifest(load_result.manifest);
  if (!expect(config_opt.has_value(), "manifest smoke test requires a known runtime config")) {
    std::cerr << "forward_model_debug: unsupported model_id="
              << load_result.manifest.runtime.model_id << "\n";
    return false;
  }
  nemotron::SingleTokenForwardConfig config = *config_opt;
  config.max_tokens = std::max<std::size_t>(config.max_tokens, 6);
  auto model = nemotron::SingleTokenForwardModel::Create(*environment, config);
  if (!expect(model != nullptr && model->valid(), "forward model should build from the runtime environment")) {
    const auto plan = nemotron::BuildSingleTokenForwardPlan(*environment->model_schedule(), config);
    if (!plan.has_value()) {
      std::cerr << "forward_model_debug: forward plan construction failed\n";
      return false;
    }
    const auto* embedding = find_global_descriptor_by_candidates<nemotron::EmbeddingDescriptor>(
        environment->model_schedule()->global_bindings(),
        nemotron::ModelGlobalRole::kEmbedding,
        [&](const std::string& name) { return environment->embedding_catalog()->FindDescriptor(name); },
        {"backbone.embeddings.weight", "embeddings.weight"});
    const auto* lm_head = find_global_descriptor_by_candidates<nemotron::GemmDescriptor>(
        environment->model_schedule()->global_bindings(),
        nemotron::ModelGlobalRole::kLogits,
        [&](const std::string& name) { return environment->gemm_catalog()->FindDescriptor(name); },
        {"lm_head.weight", "logits.weight", "output.weight"});
    const auto* final_norm = find_global_descriptor_by_candidates<nemotron::KernelTensorDescriptor>(
        environment->model_schedule()->global_bindings(),
        nemotron::ModelGlobalRole::kFinalNorm,
        [&](const std::string& name) { return environment->kernel_catalog()->FindTensor(name); },
        {"backbone.norm_f.weight", "norm_f.weight", "backbone.final_norm.weight", "final_norm.weight"});
    std::cerr << "forward_model_debug: embedding=" << (embedding ? "ok" : "missing")
              << " lm_head=" << (lm_head ? "ok" : "missing")
              << " final_norm=" << (final_norm ? "ok" : "missing") << "\n";
    if (embedding != nullptr) {
      std::cerr << "forward_model_debug: embedding_shape="
                << embedding->vocab_size << "x" << embedding->embedding_dim
                << " storage=" << embedding->storage_dtype << "\n";
    }
    if (lm_head != nullptr) {
      std::cerr << "forward_model_debug: lm_head_shape="
                << lm_head->output_rows << "x" << lm_head->input_cols
                << " family=" << static_cast<int>(lm_head->kernel_family)
                << " storage=" << lm_head->storage_dtype << "\n";
    }
    if (final_norm != nullptr && !final_norm->logical_shape.empty()) {
      std::cerr << "forward_model_debug: final_norm_dim="
                << final_norm->logical_shape.front()
                << " storage=" << final_norm->storage_dtype << "\n";
    }
    bool binding_failure = false;
    for (const auto& layer_entry : plan->layers) {
      const auto* layer = environment->model_schedule()->FindLayer(layer_entry.layer_index);
      if (layer == nullptr) {
        std::cerr << "forward_model_debug: missing schedule layer " << layer_entry.layer_index << "\n";
        binding_failure = true;
        break;
      }
      bool bindings_ok = false;
      switch (layer_entry.kind) {
        case nemotron::ForwardLayerKind::kAttention:
          bindings_ok =
              nemotron::BuildAttentionLayerBindings(
                  *layer,
                  *environment->kernel_catalog(),
                  *environment->gemm_catalog())
                  .has_value();
          break;
        case nemotron::ForwardLayerKind::kMamba:
          bindings_ok =
              nemotron::BuildMambaLayerBindings(
                  *layer,
                  *environment->kernel_catalog(),
                  *environment->gemm_catalog())
                  .has_value();
          break;
        case nemotron::ForwardLayerKind::kExpert:
          bindings_ok =
              nemotron::BuildExpertLayerBindings(
                  *layer,
                  *environment->kernel_catalog(),
                  *environment->gemm_catalog(),
                  config.n_routed_experts)
                  .has_value();
          break;
      }
      if (!bindings_ok) {
        std::cerr << "forward_model_debug: bindings failed for layer "
                  << layer_entry.layer_index << " kind="
                  << static_cast<int>(layer_entry.kind) << "\n";
        binding_failure = true;
        break;
      }
    }
    if (!binding_failure) {
      auto cublas = nemotron::CublasLtHandle::Create();
      std::cerr << "forward_model_debug: all layer bindings validated"
                << " cublas=" << ((cublas != nullptr && cublas->valid()) ? "ok" : "failed");
      if (plan->attention_layer_count != 0) {
        auto cudnn = nemotron::CudnnHandle::Create();
        std::cerr << " cudnn=" << ((cudnn != nullptr && cudnn->valid()) ? "ok" : "failed");
      }
      std::cerr << "\n";
    }
    return false;
  }

  auto request_context = model->CreateRequestContext();
  if (!expect(request_context != nullptr && request_context->valid(), "request context should be creatable")) {
    return false;
  }

  nemotron::GreedyDecodeConfig decode_config;
  decode_config.max_new_tokens = 2;
  decode_config.eos_token_ids = {2, 11};
  nemotron::GreedyDecodeResult decode_result;
  const std::int32_t prompt_token_ids[] = {1};
  if (!expect(
          model->RunGreedyDecode(
              prompt_token_ids,
              std::size(prompt_token_ids),
              decode_config,
              *request_context,
              &decode_result),
          "greedy decode loop should execute successfully")) {
    return false;
  }
  if (!expect(
          !decode_result.generated_token_ids.empty(),
          "greedy decode loop should emit at least one token when capacity permits")) {
    return false;
  }
  if (!expect(
          decode_result.generated_token_ids.size() <= decode_config.max_new_tokens,
          "greedy decode loop should not exceed the requested decode budget")) {
    return false;
  }
  if (!expect(
          request_context->sequence_length() ==
              (std::size(prompt_token_ids) + decode_result.generated_token_ids.size()),
          "request sequence length should include the consumed generated tokens")) {
    return false;
  }
  if (!expect(
          request_context->decode_position() == request_context->sequence_length(),
          "decode position should stay aligned with the committed sequence length")) {
    return false;
  }

  const std::vector<std::int32_t> boundary_only_prompt = {1, 7, 9, 13};
  const auto boundary_only_comparison =
      compare_boundary_only_prefill_row(*model, boundary_only_prompt, config);
  if (!expect(
          boundary_only_comparison.has_value(),
          "boundary-only prefill logits comparison should succeed") ||
      !expect(
          boundary_only_comparison->request_state_match,
          "boundary-only prefill should preserve request state") ||
      !expect(
          max_abs_diff(
              boundary_only_comparison->full_row,
              boundary_only_comparison->boundary_row) <= 1.0e-3f,
          "boundary-only prefill row should match the full prefill boundary row")) {
    return false;
  }

  if (!decode_result.hit_eos) {
    if (!expect(
            decode_result.generated_token_ids.size() == decode_config.max_new_tokens,
            "non-EOS decode should run to the requested decode budget")) {
      return false;
    }
  } else {
    const std::int32_t emitted_eos = decode_result.generated_token_ids.back();
    if (!expect(
            emitted_eos == 2 || emitted_eos == 11,
            "EOS termination should use the configured Nano EOS ids")) {
      return false;
    }
  }

  auto turn1_context = model->CreateRequestContext();
  auto turn2_cached_context = model->CreateRequestContext();
  auto turn2_baseline_context = model->CreateRequestContext();
  if (!expect(
          turn1_context != nullptr && turn1_context->valid(),
          "cached turn-1 request context should be creatable") ||
      !expect(
          turn2_cached_context != nullptr && turn2_cached_context->valid(),
          "cached turn-2 request context should be creatable") ||
      !expect(
          turn2_baseline_context != nullptr && turn2_baseline_context->valid(),
          "baseline turn-2 request context should be creatable")) {
    return false;
  }

  const std::string conversation_id = "full-forward-manifest-smoke";
  const auto turn1_identity =
      make_identity(std::vector<std::int32_t>{std::begin(prompt_token_ids), std::end(prompt_token_ids)},
                    load_result.manifest.runtime.model_id);
  nemotron::GreedyDecodeConfig turn_decode_config;
  turn_decode_config.max_new_tokens = 1;
  turn_decode_config.eos_token_ids = {2, 11};
  nemotron::GreedyDecodeResult turn1_result;
  std::size_t turn1_match = 0;
  if (!expect(
          model->RunGreedyConversationTurn(
              turn1_identity,
              conversation_id,
              turn_decode_config,
              *turn1_context,
              &turn1_result,
              &turn1_match),
          "cached turn-1 execution should succeed")) {
    return false;
  }
  if (!expect(
          turn1_match == 0,
          "turn-1 execution should cold-start without a reusable committed head") ||
      !expect(
          turn1_result.generated_token_ids.size() == 1,
          "turn-1 execution should emit one token for the follow-up turn")) {
    return false;
  }
  const auto turn1_committed_head = environment->prefix_cache().CommittedHeadForConversation(conversation_id);
  if (!expect(
          turn1_committed_head.has_value(),
          "turn-1 execution should publish a committed conversation head")) {
    return false;
  }

  auto turn2_identity = turn1_identity;
  turn2_identity.token_ids.insert(
      turn2_identity.token_ids.end(),
      turn1_result.generated_token_ids.begin(),
      turn1_result.generated_token_ids.end());
  turn2_identity.token_ids.push_back(17);
  const std::size_t turn2_reused_prefix_tokens =
      turn1_identity.token_ids.size() + turn1_result.generated_token_ids.size();
  const std::size_t turn2_suffix_tokens =
      turn2_identity.token_ids.size() - turn2_reused_prefix_tokens;
  auto turn2_full_prefill_context = model->CreateRequestContext();
  auto turn2_split_prefill_context = model->CreateRequestContext();
  if (!expect(
          turn2_full_prefill_context != nullptr && turn2_full_prefill_context->valid(),
          "turn-2 full-prefill request context should be creatable") ||
      !expect(
          turn2_split_prefill_context != nullptr && turn2_split_prefill_context->valid(),
          "turn-2 split-prefill request context should be creatable")) {
    return false;
  }
  auto turn2_full_prefill_logits =
      nemotron::DeviceTensorFp32::Create({turn2_identity.token_ids.size(), config.vocab_size});
  auto turn2_prefix_prefill_logits =
      nemotron::DeviceTensorFp32::Create({turn2_reused_prefix_tokens, config.vocab_size});
  auto turn2_suffix_prefill_logits =
      nemotron::DeviceTensorFp32::Create({turn2_suffix_tokens, config.vocab_size});
  if (!expect(
          turn2_full_prefill_logits != nullptr && turn2_full_prefill_logits->valid(),
          "turn-2 full-prefill logits buffer should allocate") ||
      !expect(
          turn2_prefix_prefill_logits != nullptr && turn2_prefix_prefill_logits->valid(),
          "turn-2 prefix-prefill logits buffer should allocate") ||
      !expect(
          turn2_suffix_prefill_logits != nullptr && turn2_suffix_prefill_logits->valid(),
          "turn-2 suffix-prefill logits buffer should allocate")) {
    return false;
  }
  if (!expect(
          model->RunPrefill(
              turn2_identity.token_ids.data(),
              turn2_identity.token_ids.size(),
              *turn2_full_prefill_context,
              turn2_full_prefill_logits.get()),
          "turn-2 full prefill should succeed") ||
      !expect(
          model->RunPrefill(
              turn2_identity.token_ids.data(),
              turn2_reused_prefix_tokens,
              *turn2_split_prefill_context,
              turn2_prefix_prefill_logits.get()),
          "turn-2 reused-prefix prefill should succeed") ||
      !expect(
          model->ContinuePrefill(
              turn2_identity.token_ids.data() + turn2_reused_prefix_tokens,
              turn2_suffix_tokens,
              *turn2_split_prefill_context,
              turn2_suffix_prefill_logits.get()),
          "turn-2 suffix continuation prefill should succeed")) {
    return false;
  }
  const std::vector<float> turn2_full_prefill_host = copy_tensor_to_host(*turn2_full_prefill_logits);
  const std::vector<float> turn2_suffix_prefill_host = copy_tensor_to_host(*turn2_suffix_prefill_logits);
  if (!expect(
          all_finite(turn2_full_prefill_host) && all_finite(turn2_suffix_prefill_host),
          "turn-2 prefill logits should stay finite")) {
    return false;
  }
  const std::vector<float> turn2_full_last_row =
      slice_row(turn2_full_prefill_host, turn2_identity.token_ids.size() - 1, config.vocab_size);
  const std::vector<float> turn2_suffix_last_row =
      slice_row(turn2_suffix_prefill_host, turn2_suffix_tokens - 1, config.vocab_size);
  const auto turn2_full_argmax = argmax_token_id(turn2_full_last_row);
  const auto turn2_suffix_argmax = argmax_token_id(turn2_suffix_last_row);
  const bool turn2_split_prefill_ok =
      expect(
          !turn2_full_last_row.empty() && !turn2_suffix_last_row.empty(),
          "turn-2 prefill row slicing should succeed") &&
      expect(
          turn2_full_argmax.has_value() && turn2_suffix_argmax.has_value(),
          "turn-2 prefill argmax selection should succeed") &&
      expect(
          *turn2_full_argmax == *turn2_suffix_argmax,
          "turn-2 full-prefill and split-prefill next-token argmax should match") &&
      expect(
          max_abs_diff(turn2_full_last_row, turn2_suffix_last_row) <= 1.0e-3f,
          "turn-2 full-prefill and split-prefill logits should match within tolerance") &&
      expect(
          turn2_full_prefill_context->sequence_length() == turn2_split_prefill_context->sequence_length() &&
              turn2_full_prefill_context->decode_position() == turn2_split_prefill_context->decode_position(),
          "turn-2 full-prefill and split-prefill request state should match");
  if (!turn2_split_prefill_ok) {
    maybe_print_first_split_prefill_divergent_layer(
        *model,
        turn2_identity.token_ids,
        turn2_reused_prefix_tokens,
        config);
    return false;
  }
  nemotron::GreedyDecodeResult turn2_cached_result;
  nemotron::GreedyDecodeResult turn2_baseline_result;
  std::size_t turn2_match = 0;
  if (!expect(
          model->RunGreedyConversationTurn(
              turn2_identity,
              conversation_id,
              turn_decode_config,
              *turn2_cached_context,
              &turn2_cached_result,
              &turn2_match),
          "cached turn-2 execution should succeed") ||
      !expect(
          model->RunGreedyDecode(
              turn2_identity.token_ids.data(),
              turn2_identity.token_ids.size(),
              turn_decode_config,
              *turn2_baseline_context,
              &turn2_baseline_result),
          "baseline turn-2 execution should succeed")) {
    return false;
  }
  if (!expect(
          turn2_match == turn2_reused_prefix_tokens,
          "turn-2 execution should reuse the committed head from turn 1") ||
      !expect(
          turn2_cached_result.generated_token_ids == turn2_baseline_result.generated_token_ids,
          "cached turn-2 decode should match the baseline decode tokens") ||
      !expect(
          turn2_cached_result.hit_eos == turn2_baseline_result.hit_eos &&
              turn2_cached_result.hit_capacity_limit == turn2_baseline_result.hit_capacity_limit,
          "cached turn-2 stop conditions should match the baseline decode path") ||
      !expect(
          turn2_cached_context->sequence_length() == turn2_baseline_context->sequence_length() &&
              turn2_cached_context->decode_position() == turn2_baseline_context->decode_position(),
          "cached turn-2 request state should match the baseline decode state")) {
    return false;
  }
  const auto turn2_committed_head = environment->prefix_cache().CommittedHeadForConversation(conversation_id);
  if (!expect(
          turn2_committed_head.has_value(),
          "turn-2 execution should refresh the committed head")) {
    return false;
  }
  const auto turn2_committed_view = environment->prefix_cache().Describe(*turn2_committed_head);
  if (!expect(
          turn2_committed_view.has_value() &&
              turn2_committed_view->has_boundary_logits &&
              turn2_committed_view->boundary_logits_count == config.vocab_size &&
              turn2_committed_view->identity.token_ids.size() ==
                  turn2_identity.token_ids.size() + turn2_cached_result.generated_token_ids.size(),
          "turn-2 committed head should extend the prompt with generated tokens")) {
    return false;
  }

  auto turn3_cached_context = model->CreateRequestContext();
  auto turn3_baseline_context = model->CreateRequestContext();
  if (!expect(
          turn3_cached_context != nullptr && turn3_cached_context->valid(),
          "cached turn-3 request context should be creatable") ||
      !expect(
          turn3_baseline_context != nullptr && turn3_baseline_context->valid(),
          "baseline turn-3 request context should be creatable")) {
    return false;
  }
  const auto turn3_identity = turn2_committed_view->identity;
  nemotron::GreedyDecodeResult turn3_cached_result;
  nemotron::GreedyDecodeResult turn3_baseline_result;
  std::size_t turn3_match = 0;
  if (!expect(
          model->RunGreedyConversationTurn(
              turn3_identity,
              conversation_id,
              turn_decode_config,
              *turn3_cached_context,
              &turn3_cached_result,
              &turn3_match),
          "cached turn-3 exact-hit execution should succeed") ||
      !expect(
          model->RunGreedyDecode(
              turn3_identity.token_ids.data(),
              turn3_identity.token_ids.size(),
              turn_decode_config,
              *turn3_baseline_context,
              &turn3_baseline_result),
          "baseline turn-3 exact-hit execution should succeed")) {
    return false;
  }
  if (!expect(
          turn3_match == turn3_identity.token_ids.size(),
          "turn-3 execution should reuse the exact committed head without prompt replay") ||
      !expect(
          turn3_cached_result.generated_token_ids == turn3_baseline_result.generated_token_ids,
          "cached turn-3 decode should match the baseline decode tokens") ||
      !expect(
          turn3_cached_result.hit_eos == turn3_baseline_result.hit_eos &&
              turn3_cached_result.hit_capacity_limit == turn3_baseline_result.hit_capacity_limit,
          "cached turn-3 stop conditions should match the baseline decode path") ||
      !expect(
          turn3_cached_context->sequence_length() == turn3_baseline_context->sequence_length() &&
              turn3_cached_context->decode_position() == turn3_baseline_context->decode_position(),
          "cached turn-3 request state should match the baseline decode state")) {
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!run_full_forward_manifest_smoke()) {
    return 1;
  }
  std::cout << "full_forward_manifest_smoke_test: PASS\n";
  return 0;
}
