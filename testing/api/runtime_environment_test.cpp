#include "nemotron/gemm_planner.h"
#include "nemotron/model_schedule.h"
#include "nemotron/runtime_environment.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

namespace {

using nemotron::PackedModelManifest;
using nemotron::RuntimeBootstrapOptions;
using nemotron::RuntimeEnvironment;
using nemotron::ServiceMemoryTarget;
using nemotron::TensorAuxiliaryManifest;
using nemotron::TensorManifestEntry;

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

class TempFile {
 public:
  TempFile() {
    path_ = std::filesystem::temp_directory_path() /
            ("nemotron_runtime_environment_meminfo_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)) + ".txt");
  }

  ~TempFile() {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }

  const std::filesystem::path& path() const {
    return path_;
  }

 private:
  std::filesystem::path path_;
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

PackedModelManifest make_manifest() {
  PackedModelManifest manifest;
  manifest.schema_version = 1;
  manifest.runtime.model_id = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4";
  manifest.runtime.source_revision = "b1ffe499";
  manifest.runtime.tokenizer_revision = "b1ffe499";
  manifest.runtime.packer_version = "test-packer";
  manifest.runtime.gpu_family = "GB10";
  manifest.runtime.compute_capability = "12.1";
  manifest.runtime.kv_bytes_per_token = 4096;
  manifest.runtime.mamba_state_bytes_fp16 = 87162880;
  manifest.runtime.mamba_state_bytes_fp32 = 174325760;

  TensorManifestEntry dense;
  dense.name = "backbone.embeddings.weight";
  dense.op_class = "embedding";
  dense.logical_shape = {131072, 4096};
  dense.packed_shape = {131072, 4096};
  dense.storage_dtype = "bf16";
  dense.compute_dtype = "bf16";
  dense.layout_tag = "row_major";
  dense.alignment_bytes = 16;
  dense.packed_file = "weights/core.bin";
  dense.nbytes = 4096;
  dense.source_tensor_name = dense.name;
  dense.checksum = "abc";
  manifest.tensors.push_back(dense);

  TensorManifestEntry scaled;
  scaled.name = "layers.12.experts.117.w1";
  scaled.op_class = "routed_expert";
  scaled.logical_shape = {2688, 1024};
  scaled.packed_shape = {2688, 1024};
  scaled.storage_dtype = "nvfp4_e2m1";
  scaled.compute_dtype = "fp32_accum";
  scaled.layout_tag = "cublaslt_fp4_tn_v1";
  scaled.alignment_bytes = 16;
  scaled.block_scale_mode = "vec16_e4m3";
  scaled.block_scale_dtype = "e4m3";
  scaled.tensor_scale_dtype = "fp32";
  scaled.packed_file = "weights/experts.bin";
  scaled.nbytes = 2048;
  scaled.source_tensor_name = scaled.name;
  scaled.checksum = "def";
  scaled.auxiliaries.push_back(TensorAuxiliaryManifest{
      "block_scales",
      "weights/experts_scales.bin",
      0,
      512,
  });
  scaled.auxiliaries.push_back(TensorAuxiliaryManifest{
      "tensor_scale",
      "weights/experts_scales.bin",
      512,
      4,
  });
  manifest.tensors.push_back(scaled);

  return manifest;
}

RuntimeBootstrapOptions make_options() {
  RuntimeBootstrapOptions options;
  options.service_target.total_memory_bytes = GiB(128);
  options.service_target.weights_bytes = GiB(100);
  options.service_target.workspace_bytes = GiB(8);
  options.service_target.graph_bytes = GiB(4);
  options.service_target.safety_headroom_bytes = GiB(4);
  options.service_target.target_active_requests = 8;
  options.service_target.target_context_tokens = 65536;
  options.use_fp16_mamba_state = true;
  options.reusable_node_metadata_bytes = 4096;
  return options;
}

bool test_runtime_environment_uses_loader_budget_when_enabled() {
  ScopedEnvVar scoped("NEMOTRON_PREFIX_CACHE");
  unsetenv("NEMOTRON_PREFIX_CACHE");

  const auto environment = RuntimeEnvironment::Build(make_manifest(), make_options());
  return expect(static_cast<bool>(environment), "runtime environment should build for a valid manifest") &&
         expect(environment->config().prefix_cache_enabled, "cache should be enabled by default") &&
         expect(!environment->has_artifact_loader(), "in-memory bootstrap should not create an artifact loader") &&
         expect(!environment->has_tensor_catalog(), "in-memory bootstrap should not create a tensor catalog") &&
         expect(environment->has_model_schedule(), "in-memory bootstrap should create a model schedule from manifest metadata") &&
         expect(environment->model_schedule() != nullptr, "model schedule accessor should expose the schedule") &&
         expect(environment->model_schedule()->ordered_layers().size() == 1,
                "test manifest should currently expose one scheduled layer") &&
         expect(environment->model_schedule()->routed_expert_layer_count() == 1,
                "test manifest layer should be classified as routed-expert-backed") &&
         expect(!environment->has_weight_arena_plan(), "in-memory bootstrap should not create a weight arena plan") &&
         expect(!environment->has_weight_arena(), "in-memory bootstrap should not create a materialized weight arena") &&
         expect(!environment->has_kernel_catalog(), "in-memory bootstrap should not create a kernel catalog") &&
         expect(!environment->has_gemm_catalog(), "in-memory bootstrap should not create a GEMM catalog") &&
         expect(!environment->has_embedding_catalog(), "in-memory bootstrap should not create an embedding catalog") &&
         expect(environment->has_gemm_heuristic_cache(), "runtime environment should create a service-level GEMM heuristic cache") &&
         expect(environment->gemm_heuristic_cache() != nullptr, "heuristic-cache accessor should expose the service-level cache") &&
         expect(environment->gemm_heuristic_cache()->size() == 0, "new runtime environment should start with an empty GEMM heuristic cache") &&
         expect(environment->prefix_cache().enabled(), "prefix cache should start enabled") &&
         expect(environment->effective_shared_cache_budget_bytes() ==
                    environment->loader_plan().memory_budget.shared_cache_budget_bytes,
                "effective cache budget should match loader plan when cache is enabled") &&
         expect(environment->prefix_cache().max_bytes() == environment->effective_shared_cache_budget_bytes(),
                "prefix cache capacity should track the effective shared-cache budget") &&
         expect(environment->reusable_state_arena().max_bytes() == environment->effective_shared_cache_budget_bytes(),
                "state arena capacity should track the effective shared-cache budget");
}

bool test_runtime_environment_zeroes_cache_budget_when_disabled() {
  ScopedEnvVar scoped("NEMOTRON_PREFIX_CACHE");
  setenv("NEMOTRON_PREFIX_CACHE", "0", 1);

  const auto environment = RuntimeEnvironment::Build(make_manifest(), make_options());
  return expect(static_cast<bool>(environment), "runtime environment should still build when cache is disabled") &&
         expect(!environment->config().prefix_cache_enabled, "env should disable prefix caching") &&
         expect(!environment->prefix_cache().enabled(), "prefix cache object should be disabled") &&
         expect(environment->effective_shared_cache_budget_bytes() == 0,
                "effective shared-cache budget should be zero when caching is disabled") &&
         expect(environment->prefix_cache().max_bytes() == 0, "disabled cache should be constructed with zero budget") &&
         expect(environment->reusable_state_arena().max_bytes() == 0, "disabled cache should not reserve state-arena bytes");
}

bool test_runtime_environment_rejects_invalid_manifest() {
  auto manifest = make_manifest();
  manifest.tensors[0].checksum.clear();

  const auto environment = RuntimeEnvironment::Build(manifest, make_options());
  return expect(!environment, "invalid manifest should prevent environment construction");
}

bool test_runtime_environment_can_clamp_budget_with_proc_meminfo_signal() {
  ScopedEnvVar scoped("NEMOTRON_PREFIX_CACHE");
  unsetenv("NEMOTRON_PREFIX_CACHE");

  TempFile meminfo;
  std::ofstream output(meminfo.path());
  output
      << "MemAvailable:    6291456 kB\n"
      << "SwapFree:        2097152 kB\n";
  output.close();

  auto options = make_options();
  options.prefer_host_memory_snapshot = true;
  options.proc_meminfo_path = meminfo.path();

  const auto environment = RuntimeEnvironment::Build(make_manifest(), options);
  return expect(static_cast<bool>(environment), "runtime environment should still build with a meminfo clamp") &&
         expect(environment->loader_plan().memory_budget.used_host_memory_snapshot,
                "loader plan should report using the proc-meminfo snapshot") &&
         expect(environment->loader_plan().memory_budget.planning_total_memory_bytes == GiB(8),
                "planning total should clamp to MemAvailable + SwapFree") &&
         expect(!environment->loader_plan().memory_budget.fits,
                "the clamped budget should not fit the large service target") &&
         expect(environment->effective_shared_cache_budget_bytes() == 0,
                "no shared cache budget should remain when the clamped host budget does not fit");
}

}  // namespace

int main() {
  const bool ok =
      test_runtime_environment_uses_loader_budget_when_enabled() &&
      test_runtime_environment_zeroes_cache_budget_when_disabled() &&
      test_runtime_environment_rejects_invalid_manifest() &&
      test_runtime_environment_can_clamp_budget_with_proc_meminfo_signal();

  if (!ok) {
    return 1;
  }
  std::cout << "runtime_environment_test: PASS\n";
  return 0;
}
