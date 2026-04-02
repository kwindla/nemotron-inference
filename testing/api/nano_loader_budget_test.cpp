#include "nemotron/loader.h"
#include "nemotron/manifest.h"
#include "nemotron/memory_budget.h"
#include "nemotron/runtime_environment.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

using nemotron::BuildLoaderPlan;
using nemotron::BytesForExactPrefixNode;
using nemotron::LoadManifestFromJsonFile;
using nemotron::LoaderPlan;
using nemotron::PackedModelManifest;
using nemotron::RuntimeBootstrapOptions;
using nemotron::RuntimeEnvironment;
using nemotron::RuntimeMemoryProfile;
using nemotron::ServiceMemoryTarget;

constexpr const char* kNanoModelId = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4";
constexpr std::size_t kNanoKvBytesPerToken = 6144;
constexpr std::size_t kNanoMambaStateBytesFp16 = 25247744;
constexpr std::size_t kNanoMambaStateBytesFp32 = 50495488;
constexpr std::size_t kTargetContextTokens = 4096;
constexpr std::size_t kTargetActiveRequests = 4;
constexpr std::size_t kReusableNodeMetadataBytes = 4096;

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

constexpr std::size_t MiB(std::size_t value) {
  return value * 1024ull * 1024ull;
}

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

std::size_t RoundUp(std::size_t value, std::size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  const std::size_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  return value + (alignment - remainder);
}

std::filesystem::path default_manifest_path() {
  return std::filesystem::path(NEMOTRON_SOURCE_ROOT) /
         "artifacts" /
         "manifests" /
         "forward_runtime_manifest_nano_rtx5090_unverified.json";
}

std::filesystem::path resolve_manifest_path() {
  const char* manifest_env = std::getenv("NEMOTRON_FORWARD_MANIFEST");
  if (manifest_env != nullptr && manifest_env[0] != '\0') {
    return std::filesystem::path(manifest_env);
  }
  return default_manifest_path();
}

std::size_t manifest_weights_bytes_rounded_up(const PackedModelManifest& manifest) {
  std::size_t total_bytes = 0;
  for (const auto& tensor : manifest.tensors) {
    total_bytes += tensor.nbytes;
    for (const auto& auxiliary : tensor.auxiliaries) {
      total_bytes += auxiliary.nbytes;
    }
  }
  return RoundUp(total_bytes, MiB(1));
}

ServiceMemoryTarget make_rtx5090_target(const PackedModelManifest& manifest) {
  ServiceMemoryTarget target;
  target.total_memory_bytes = GiB(32);
  target.weights_bytes = manifest_weights_bytes_rounded_up(manifest);
  target.workspace_bytes = GiB(1);
  target.graph_bytes = MiB(512);
  target.safety_headroom_bytes = GiB(1);
  target.target_active_requests = kTargetActiveRequests;
  target.target_context_tokens = kTargetContextTokens;
  return target;
}

RuntimeBootstrapOptions make_options(const PackedModelManifest& manifest) {
  RuntimeBootstrapOptions options;
  options.service_target = make_rtx5090_target(manifest);
  options.use_fp16_mamba_state = true;
  options.reusable_node_metadata_bytes = kReusableNodeMetadataBytes;
  options.verify_manifest_files = false;
  options.materialize_weight_arena = false;
  return options;
}

void print_loader_plan_summary(const LoaderPlan& loader_plan, const ServiceMemoryTarget& target) {
  std::cout << "nano_loader_budget_test: summary"
            << " weights_bytes=" << target.weights_bytes
            << " active_kv_bytes=" << loader_plan.memory_budget.active_kv_bytes
            << " active_mamba_bytes=" << loader_plan.memory_budget.active_mamba_bytes
            << " shared_cache_budget_bytes=" << loader_plan.memory_budget.shared_cache_budget_bytes
            << " max_4k_full_context_nodes=" << loader_plan.memory_budget.max_full_context_nodes
            << "\n";
}

bool run_nano_loader_budget_test() {
  ScopedEnvVar prefix_cache_env("NEMOTRON_PREFIX_CACHE");
  unsetenv("NEMOTRON_PREFIX_CACHE");

  const std::filesystem::path manifest_path = resolve_manifest_path();
  if (!std::filesystem::exists(manifest_path)) {
    std::cout << "nano_loader_budget_test: SKIP (generated Nano manifest not found)\n";
    return true;
  }

  const auto load_result = LoadManifestFromJsonFile(manifest_path);
  if (!expect(load_result.ok, "Nano manifest JSON should parse")) {
    return false;
  }
  if (load_result.manifest.runtime.model_id != kNanoModelId) {
    std::cout << "nano_loader_budget_test: SKIP (manifest model is not Nano)\n";
    return true;
  }

  const PackedModelManifest& manifest = load_result.manifest;
  const ServiceMemoryTarget target = make_rtx5090_target(manifest);
  const RuntimeBootstrapOptions options = make_options(manifest);

  RuntimeMemoryProfile expected_profile;
  expected_profile.kv_bytes_per_token = kNanoKvBytesPerToken;
  expected_profile.current_mamba_state_bytes = kNanoMambaStateBytesFp16;
  expected_profile.reusable_node_metadata_bytes = kReusableNodeMetadataBytes;

  const LoaderPlan loader_plan = BuildLoaderPlan(
      manifest,
      target,
      options.use_fp16_mamba_state,
      options.reusable_node_metadata_bytes);
  print_loader_plan_summary(loader_plan, target);

  if (!expect(manifest.runtime.kv_bytes_per_token == kNanoKvBytesPerToken,
              "Nano manifest should carry the derived KV bytes per token") ||
      !expect(manifest.runtime.mamba_state_bytes_fp16 == kNanoMambaStateBytesFp16,
              "Nano manifest should carry the derived FP16 Mamba-state bytes") ||
      !expect(manifest.runtime.mamba_state_bytes_fp32 == kNanoMambaStateBytesFp32,
              "Nano manifest should carry the derived FP32 Mamba-state bytes") ||
      !expect(loader_plan.valid, "loader plan should build from the generated Nano manifest") ||
      !expect(loader_plan.memory_profile.kv_bytes_per_token == expected_profile.kv_bytes_per_token,
              "loader plan should use the manifest KV runtime profile") ||
      !expect(loader_plan.memory_profile.current_mamba_state_bytes == expected_profile.current_mamba_state_bytes,
              "loader plan should use the FP16 Mamba-state runtime profile") ||
      !expect(loader_plan.memory_budget.fits, "RTX 5090 target should fit the Nano working set") ||
      !expect(
          loader_plan.memory_budget.bytes_per_full_context_node ==
              BytesForExactPrefixNode(expected_profile, kTargetContextTokens),
          "bytes per 4K full-context node should match the prefix-node helper")) {
    return false;
  }

  const auto environment =
      RuntimeEnvironment::BuildFromManifestFile(manifest_path, options);
  return expect(static_cast<bool>(environment), "runtime environment should build from the Nano manifest") &&
         expect(environment->loader_plan().valid, "runtime environment should retain a valid loader plan") &&
         expect(environment->loader_plan().memory_budget.shared_cache_budget_bytes ==
                    loader_plan.memory_budget.shared_cache_budget_bytes,
                "manifest-backed runtime build should preserve the loader shared-cache budget") &&
         expect(environment->effective_shared_cache_budget_bytes() ==
                    loader_plan.memory_budget.shared_cache_budget_bytes,
                "effective runtime shared-cache budget should match the loader plan") &&
         expect(environment->prefix_cache().max_bytes() ==
                    loader_plan.memory_budget.shared_cache_budget_bytes,
                "prefix cache budget should match the loader shared-cache budget") &&
         expect(environment->reusable_state_arena().max_bytes() ==
                    loader_plan.memory_budget.shared_cache_budget_bytes,
                "reusable state arena budget should match the loader shared-cache budget");
}

}  // namespace

int main() {
  if (!run_nano_loader_budget_test()) {
    return 1;
  }
  std::cout << "nano_loader_budget_test: PASS\n";
  return 0;
}
