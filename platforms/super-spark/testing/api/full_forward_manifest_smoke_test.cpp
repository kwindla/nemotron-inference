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

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// EVIDENCE SURFACE: manifest/runtime smoke test. This is NOT a parity oracle.

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

  const nemotron::SingleTokenForwardConfig config =
      nemotron::KnownNemotron3Super120BA12BConfig();
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
  auto logits = nemotron::DeviceTensorFp32::Create({1, config.vocab_size});
  if (!expect(request_context != nullptr && request_context->valid(), "request context should be creatable") ||
      !expect(logits != nullptr && logits->valid(), "logits tensor should be creatable")) {
    return false;
  }

  if (!expect(model->RunSingleToken(0, *request_context, logits.get()), "single-token forward execution should succeed")) {
    return false;
  }

  std::vector<float> host_logits(logits->numel(), 0.0f);
  if (!expect(logits->CopyToHost(host_logits.data(), host_logits.size()), "logits should copy back to host")) {
    return false;
  }
  for (float value : host_logits) {
    if (!std::isfinite(value)) {
      return expect(false, "logits should be finite");
    }
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
