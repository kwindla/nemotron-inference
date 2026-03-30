#include "nemotron/artifact_loader.h"
#include "nemotron/gemm_catalog.h"
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
using nemotron::GemmCatalog;
using nemotron::GemmHeuristicCache;
using nemotron::GemmKernelFamily;
using nemotron::KernelCatalog;
using nemotron::LoadVerifiedManifestFromJsonFile;
using nemotron::ManifestLoadResult;
using nemotron::TensorCatalog;
using nemotron::WeightArena;

class TempDir {
 public:
  TempDir() {
    path_ = std::filesystem::temp_directory_path() /
            ("nemotron_gemm_planner_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
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

bool test_gemm_planner_builds_launch_plans_and_cache_keys() {
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
  const auto* dense = loaded.gemm_catalog.FindDescriptor("mlp.up_proj");
  const auto* expert = loaded.gemm_catalog.FindDescriptor("layers.12.experts.117.w1");
  if (!expect(loaded.gemm_catalog.valid(), "GEMM catalog should be valid before launch planning")) {
    return false;
  }
  if (!expect(dense != nullptr && expert != nullptr, "launch planner needs both dense and scaled descriptors")) {
    return false;
  }

  const auto dense_plan = BuildGemmLaunchPlan(*dense, 32);
  const auto expert_plan = BuildGemmLaunchPlan(*expert, 32);
  return expect(dense_plan.has_value(), "dense descriptor should produce a launch plan") &&
         expect(expert_plan.has_value(), "scaled descriptor should produce a launch plan") &&
         expect(dense_plan->m == 32 && dense_plan->n == 4 && dense_plan->k == 2,
                "dense launch plan should compute M/N/K from descriptor shape") &&
         expect(expert_plan->m == 32 && expert_plan->n == 2688 && expert_plan->k == 1024,
                "scaled launch plan should compute M/N/K from descriptor shape") &&
         expect(!dense_plan->uses_block_scales, "dense launch plan should not use block scales") &&
         expect(expert_plan->uses_block_scales, "scaled launch plan should use block scales") &&
         expect(!dense_plan->heuristic_key.empty(), "dense launch plan should expose a heuristic key") &&
         expect(!expert_plan->heuristic_key.empty(), "scaled launch plan should expose a heuristic key") &&
         expect(dense_plan->heuristic_key != expert_plan->heuristic_key,
                "different GEMM families should produce different heuristic keys");
}

bool test_gemm_heuristic_cache_stores_by_launch_key() {
  nemotron::GemmDescriptor descriptor;
  descriptor.tensor_name = "mlp.up_proj";
  descriptor.op_class = "dense_linear";
  descriptor.kernel_family = GemmKernelFamily::kDenseRowMajor;
  descriptor.output_rows = 4;
  descriptor.input_cols = 2;
  descriptor.storage_dtype = "bf16";
  descriptor.compute_dtype = "bf16";
  descriptor.layout_tag = "row_major";
  descriptor.alignment_bytes = 16;
  static const std::uint8_t bytes[8] = {};
  descriptor.packed_data = bytes;
  descriptor.packed_nbytes = sizeof(bytes);

  const auto launch_plan = BuildGemmLaunchPlan(descriptor, 16);
  if (!expect(launch_plan.has_value(), "heuristic cache test requires a valid launch plan")) {
    return false;
  }

  GemmHeuristicCache cache;
  return expect(cache.Store(*launch_plan, 42), "cache should store an algorithm ID for a valid launch plan") &&
         expect(cache.size() == 1, "cache should report one stored heuristic") &&
         expect(cache.Lookup(*launch_plan).has_value(), "cache should find a stored heuristic") &&
         expect(cache.Lookup(*launch_plan).value() == 42, "cache should return the stored algorithm ID");
}

bool test_gemm_planner_rejects_zero_row_launches() {
  nemotron::GemmDescriptor descriptor;
  descriptor.tensor_name = "mlp.up_proj";
  descriptor.op_class = "dense_linear";
  descriptor.kernel_family = GemmKernelFamily::kDenseRowMajor;
  descriptor.output_rows = 4;
  descriptor.input_cols = 2;
  descriptor.storage_dtype = "bf16";
  descriptor.compute_dtype = "bf16";
  descriptor.layout_tag = "row_major";
  static const std::uint8_t bytes[8] = {};
  descriptor.packed_data = bytes;
  descriptor.packed_nbytes = sizeof(bytes);

  return expect(!BuildGemmLaunchPlan(descriptor, 0).has_value(),
                "zero activation rows should not produce a launch plan");
}

}  // namespace

int main() {
  const bool ok =
      test_gemm_planner_builds_launch_plans_and_cache_keys() &&
      test_gemm_heuristic_cache_stores_by_launch_key() &&
      test_gemm_planner_rejects_zero_row_launches();

  if (!ok) {
    return 1;
  }
  std::cout << "gemm_planner_test: PASS\n";
  return 0;
}
