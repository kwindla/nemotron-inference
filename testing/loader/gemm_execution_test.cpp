#include "nemotron/artifact_loader.h"
#include "nemotron/gemm_catalog.h"
#include "nemotron/gemm_execution.h"
#include "nemotron/gemm_planner.h"
#include "nemotron/kernel_catalog.h"
#include "nemotron/manifest.h"
#include "nemotron/tensor_catalog.h"
#include "nemotron/weight_arena.h"
#include "nemotron/weight_arena_plan.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace {

using nemotron::ArtifactLoader;
using nemotron::BuildGemmCatalog;
using nemotron::BuildGemmLaunchPlan;
using nemotron::BuildKernelCatalog;
using nemotron::BuildTensorCatalog;
using nemotron::BuildWeightArenaPlan;
using nemotron::GemmBackendKind;
using nemotron::GemmCatalog;
using nemotron::GemmHeuristicCache;
using nemotron::KernelCatalog;
using nemotron::LoadVerifiedManifestFromJsonFile;
using nemotron::ManifestLoadResult;
using nemotron::PrepareGemmExecution;
using nemotron::TensorCatalog;
using nemotron::WeightArena;

class TempDir {
 public:
  TempDir() {
    path_ = std::filesystem::temp_directory_path() /
            ("nemotron_gemm_execution_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
    std::filesystem::remove_all(path_);
    std::filesystem::create_directories(path_);
  }

  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }

  const std::filesystem::path& path() const {
    return path_;
  }

 private:
  std::filesystem::path path_;
};

struct LoadedGemmCatalog {
  ManifestLoadResult manifest_result;
  std::unique_ptr<ArtifactLoader> loader;
  TensorCatalog tensor_catalog;
  std::unique_ptr<WeightArena> arena;
  KernelCatalog kernel_catalog;
  GemmCatalog gemm_catalog;
};

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

std::string fnv1a64_hex(const std::string& bytes) {
  constexpr std::uint64_t kFnvOffsetBasis = 1469598103934665603ull;
  constexpr std::uint64_t kFnvPrime = 1099511628211ull;
  std::uint64_t hash = kFnvOffsetBasis;
  for (unsigned char ch : bytes) {
    hash ^= ch;
    hash *= kFnvPrime;
  }
  std::ostringstream oss;
  oss << std::hex << std::nouppercase << std::setfill('0') << std::setw(16) << hash;
  return oss.str();
}

void write_file(const std::filesystem::path& path, const std::string& contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  output << contents;
}

std::string make_manifest_json(
    const std::string& dense_file,
    const std::string& dense_checksum,
    const std::string& expert_file,
    const std::string& expert_checksum,
    const std::string& auxiliary_file) {
  std::ostringstream oss;
  oss
      << "{\n"
      << "  \"schema_version\": 1,\n"
      << "  \"model_id\": \"nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4\",\n"
      << "  \"source_revision\": \"b1ffe499\",\n"
      << "  \"tokenizer_revision\": \"b1ffe499\",\n"
      << "  \"packer_version\": \"test-packer\",\n"
      << "  \"target_platform\": {\n"
      << "    \"gpu_family\": \"GB10\",\n"
      << "    \"compute_capability\": \"12.1\"\n"
      << "  },\n"
      << "  \"runtime_profile\": {\n"
      << "    \"kv_bytes_per_token\": 4096,\n"
      << "    \"mamba_state_bytes_fp16\": 87162880,\n"
      << "    \"mamba_state_bytes_fp32\": 174325760\n"
      << "  },\n"
      << "  \"tensors\": [\n"
      << "    {\n"
      << "      \"name\": \"mlp.up_proj\",\n"
      << "      \"op_class\": \"dense_linear\",\n"
      << "      \"logical_shape\": [4, 2],\n"
      << "      \"packed_shape\": [4, 2],\n"
      << "      \"storage_dtype\": \"bf16\",\n"
      << "      \"compute_dtype\": \"bf16\",\n"
      << "      \"layout_tag\": \"row_major\",\n"
      << "      \"alignment_bytes\": 16,\n"
      << "      \"packed_file\": \"" << dense_file << "\",\n"
      << "      \"offset_bytes\": 0,\n"
      << "      \"nbytes\": 8,\n"
      << "      \"source_tensor_name\": \"mlp.up_proj\",\n"
      << "      \"checksum\": \"fnv1a64:" << dense_checksum << "\"\n"
      << "    },\n"
      << "    {\n"
      << "      \"name\": \"layers.12.experts.117.w1\",\n"
      << "      \"op_class\": \"routed_expert\",\n"
      << "      \"logical_shape\": [2688, 1024],\n"
      << "      \"packed_shape\": [2688, 1024],\n"
      << "      \"storage_dtype\": \"nvfp4_e2m1\",\n"
      << "      \"compute_dtype\": \"fp32_accum\",\n"
      << "      \"layout_tag\": \"cublaslt_fp4_tn_v1\",\n"
      << "      \"alignment_bytes\": 16,\n"
      << "      \"block_scale_mode\": \"vec16_e4m3\",\n"
      << "      \"block_scale_dtype\": \"e4m3\",\n"
      << "      \"tensor_scale_dtype\": \"fp32\",\n"
      << "      \"packed_file\": \"" << expert_file << "\",\n"
      << "      \"offset_bytes\": 2,\n"
      << "      \"nbytes\": 5,\n"
      << "      \"auxiliaries\": [\n"
      << "        {\n"
      << "          \"name\": \"block_scales\",\n"
      << "          \"file\": \"" << auxiliary_file << "\",\n"
      << "          \"offset_bytes\": 0,\n"
      << "          \"nbytes\": 4\n"
      << "        },\n"
      << "        {\n"
      << "          \"name\": \"tensor_scale\",\n"
      << "          \"file\": \"" << auxiliary_file << "\",\n"
      << "          \"offset_bytes\": 4,\n"
      << "          \"nbytes\": 4\n"
      << "        }\n"
      << "      ],\n"
      << "      \"source_tensor_name\": \"layers.12.experts.117.w1\",\n"
      << "      \"checksum\": \"fnv1a64:" << expert_checksum << "\"\n"
      << "    }\n"
      << "  ]\n"
      << "}\n";
  return oss.str();
}

LoadedGemmCatalog load_gemm_catalog(const std::filesystem::path& manifest_path) {
  LoadedGemmCatalog loaded;
  loaded.manifest_result = LoadVerifiedManifestFromJsonFile(manifest_path);
  if (!loaded.manifest_result.ok) {
    return loaded;
  }
  loaded.loader = ArtifactLoader::OpenVerified(loaded.manifest_result.manifest, manifest_path);
  if (!loaded.loader) {
    return loaded;
  }
  loaded.tensor_catalog = BuildTensorCatalog(loaded.manifest_result.manifest, *loaded.loader);
  if (!loaded.tensor_catalog.valid()) {
    return loaded;
  }
  const auto plan = BuildWeightArenaPlan(loaded.tensor_catalog);
  if (!plan.valid()) {
    return loaded;
  }
  loaded.arena = WeightArena::CreateFromPlan(plan);
  if (!loaded.arena) {
    return loaded;
  }
  loaded.kernel_catalog = BuildKernelCatalog(loaded.tensor_catalog, *loaded.arena);
  if (!loaded.kernel_catalog.valid()) {
    return loaded;
  }
  loaded.gemm_catalog = BuildGemmCatalog(loaded.kernel_catalog);
  return loaded;
}

bool test_prepare_gemm_execution_uses_cache_and_selects_backend() {
  TempDir temp_dir;
  const std::filesystem::path dense_path = temp_dir.path() / "weights" / "dense.bin";
  const std::filesystem::path expert_path = temp_dir.path() / "weights" / "experts.bin";
  const std::filesystem::path auxiliary_path = temp_dir.path() / "weights" / "experts_scales.bin";
  write_file(dense_path, "ABCDEFGH");
  write_file(expert_path, "abcdefghij");
  write_file(auxiliary_path, "scale123");

  const std::filesystem::path manifest_path = temp_dir.path() / "manifest.json";
  write_file(
      manifest_path,
      make_manifest_json(
          "weights/dense.bin",
          fnv1a64_hex("ABCDEFGH"),
          "weights/experts.bin",
          fnv1a64_hex("cdefg"),
          "weights/experts_scales.bin"));

  const LoadedGemmCatalog loaded = load_gemm_catalog(manifest_path);
  if (!expect(loaded.gemm_catalog.valid(), "GEMM catalog should be valid before execution prep")) {
    return false;
  }

  const auto* dense = loaded.gemm_catalog.FindDescriptor("mlp.up_proj");
  const auto* expert = loaded.gemm_catalog.FindDescriptor("layers.12.experts.117.w1");
  if (!expect(dense != nullptr && expert != nullptr, "execution prep requires both descriptors")) {
    return false;
  }

  const auto dense_plan = BuildGemmLaunchPlan(*dense, 16);
  const auto expert_plan = BuildGemmLaunchPlan(*expert, 16);
  if (!expect(dense_plan.has_value() && expert_plan.has_value(), "execution prep requires valid launch plans")) {
    return false;
  }

  GemmHeuristicCache cache;
  const auto first_dense = PrepareGemmExecution(*dense_plan, &cache);
  const auto second_dense = PrepareGemmExecution(*dense_plan, &cache);
  const auto expert_execution = PrepareGemmExecution(*expert_plan, &cache);
  return expect(first_dense.has_value(), "first dense execution should prepare successfully") &&
         expect(second_dense.has_value(), "second dense execution should prepare successfully") &&
         expect(expert_execution.has_value(), "scaled execution should prepare successfully") &&
         expect(first_dense->backend_kind == GemmBackendKind::kCublasLtDense,
                "dense execution should select the dense backend kind") &&
         expect(expert_execution->backend_kind == GemmBackendKind::kCublasLtNvfp4BlockScaled,
                "scaled execution should select the NVFP4 backend kind") &&
         expect(!first_dense->algorithm_from_cache, "first dense execution should populate the cache") &&
         expect(second_dense->algorithm_from_cache, "second dense execution should reuse the cached algorithm") &&
         expect(first_dense->algorithm_id == second_dense->algorithm_id,
                "cached dense execution should reuse the same algorithm ID") &&
         expect(expert_execution->requires_block_scales && expert_execution->requires_tensor_scale,
                "scaled execution should declare its scale dependencies") &&
         expect(cache.size() == 2, "cache should contain one dense and one scaled execution key");
}

bool test_prepare_gemm_execution_rejects_scaled_launch_without_scales() {
  nemotron::GemmLaunchPlan launch_plan;
  launch_plan.kernel_family = nemotron::GemmKernelFamily::kCublasLtNvfp4BlockScaled;
  launch_plan.m = 8;
  launch_plan.n = 4;
  launch_plan.k = 2;
  static const std::uint8_t packed_bytes[8] = {};
  launch_plan.packed_bytes = nemotron::ByteRangeView{packed_bytes, sizeof(packed_bytes)};
  launch_plan.heuristic_key = "nvfp4|test";

  return expect(!PrepareGemmExecution(launch_plan, nullptr).has_value(),
                "scaled execution should reject launch plans without scale metadata");
}

}  // namespace

int main() {
  const bool ok =
      test_prepare_gemm_execution_uses_cache_and_selects_backend() &&
      test_prepare_gemm_execution_rejects_scaled_launch_without_scales();

  if (!ok) {
    return 1;
  }
  std::cout << "gemm_execution_test: PASS\n";
  return 0;
}
