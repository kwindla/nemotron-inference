#include "nemotron/artifact_loader.h"
#include "nemotron/gemm_catalog.h"
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
using nemotron::BuildKernelCatalog;
using nemotron::BuildTensorCatalog;
using nemotron::BuildWeightArenaPlan;
using nemotron::GemmCatalog;
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
            ("nemotron_gemm_catalog_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
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

struct LoadedKernelCatalog {
  ManifestLoadResult manifest_result;
  std::unique_ptr<ArtifactLoader> loader;
  TensorCatalog tensor_catalog;
  std::unique_ptr<WeightArena> arena;
  KernelCatalog kernel_catalog;
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

std::string make_valid_manifest_json(
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

std::string make_invalid_layout_manifest_json(
    const std::string& dense_file,
    const std::string& dense_checksum) {
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
      << "      \"name\": \"router.weight\",\n"
      << "      \"op_class\": \"dense_linear\",\n"
      << "      \"logical_shape\": [4, 2],\n"
      << "      \"packed_shape\": [4, 2],\n"
      << "      \"storage_dtype\": \"bf16\",\n"
      << "      \"compute_dtype\": \"bf16\",\n"
      << "      \"layout_tag\": \"unknown_layout_v0\",\n"
      << "      \"alignment_bytes\": 16,\n"
      << "      \"packed_file\": \"" << dense_file << "\",\n"
      << "      \"offset_bytes\": 0,\n"
      << "      \"nbytes\": 8,\n"
      << "      \"source_tensor_name\": \"router.weight\",\n"
      << "      \"checksum\": \"fnv1a64:" << dense_checksum << "\"\n"
      << "    }\n"
      << "  ]\n"
      << "}\n";
  return oss.str();
}

LoadedKernelCatalog load_kernel_catalog(const std::filesystem::path& manifest_path) {
  LoadedKernelCatalog loaded;
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
  return loaded;
}

bool test_gemm_catalog_classifies_dense_and_scaled_weights() {
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
      make_valid_manifest_json(
          "weights/dense.bin",
          fnv1a64_hex("ABCDEFGH"),
          "weights/experts.bin",
          fnv1a64_hex("cdefg"),
          "weights/experts_scales.bin"));

  const LoadedKernelCatalog loaded = load_kernel_catalog(manifest_path);
  const GemmCatalog gemm_catalog = BuildGemmCatalog(loaded.kernel_catalog);
  const auto* dense = gemm_catalog.FindDescriptor("mlp.up_proj");
  const auto* expert = gemm_catalog.FindDescriptor("layers.12.experts.117.w1");
  return expect(loaded.manifest_result.ok, "manifest should verify for GEMM catalog build") &&
         expect(loaded.kernel_catalog.valid(), "kernel catalog should be valid for GEMM catalog build") &&
         expect(gemm_catalog.valid(), "GEMM catalog should build for supported row-major and NVFP4 weights") &&
         expect(gemm_catalog.descriptors().size() == 2, "GEMM catalog should include both 2D non-embedding weights") &&
         expect(dense != nullptr, "dense row-major weight should be indexed") &&
         expect(expert != nullptr, "scaled routed-expert weight should be indexed") &&
         expect(dense->kernel_family == GemmKernelFamily::kDenseRowMajor, "dense row-major weight should use dense GEMM family") &&
         expect(expert->kernel_family == GemmKernelFamily::kCublasLtNvfp4BlockScaled,
                "scaled routed-expert weight should use NVFP4 GEMM family") &&
         expect(!dense->is_scaled(), "dense row-major descriptor should not expose scale metadata") &&
         expect(expert->is_scaled(), "NVFP4 descriptor should expose scale metadata") &&
         expect(dense->output_rows == 4 && dense->input_cols == 2, "dense GEMM descriptor should preserve logical shape") &&
         expect(!dense->heuristic_key_prefix().empty(), "dense GEMM descriptor should expose a heuristic key prefix") &&
         expect(gemm_catalog.indices_by_family().count(static_cast<int>(GemmKernelFamily::kDenseRowMajor)) == 1,
                "GEMM catalog should index dense descriptors by family") &&
         expect(gemm_catalog.indices_by_family().count(static_cast<int>(GemmKernelFamily::kCublasLtNvfp4BlockScaled)) == 1,
                "GEMM catalog should index NVFP4 descriptors by family");
}

bool test_gemm_catalog_rejects_unknown_gemm_layouts() {
  TempDir temp_dir;
  const std::filesystem::path dense_path = temp_dir.path() / "weights" / "router.bin";
  write_file(dense_path, "ABCDEFGH");

  const std::filesystem::path manifest_path = temp_dir.path() / "manifest.json";
  write_file(
      manifest_path,
      make_invalid_layout_manifest_json("weights/router.bin", fnv1a64_hex("ABCDEFGH")));

  const LoadedKernelCatalog loaded = load_kernel_catalog(manifest_path);
  const GemmCatalog gemm_catalog = BuildGemmCatalog(loaded.kernel_catalog);
  return expect(loaded.manifest_result.ok, "manifest should still verify before GEMM-layout validation") &&
         expect(loaded.kernel_catalog.valid(), "kernel catalog should remain valid before GEMM-layout validation") &&
         expect(!gemm_catalog.valid(), "unknown 2D GEMM layouts should invalidate the first GEMM catalog layer");
}

}  // namespace

int main() {
  const bool ok =
      test_gemm_catalog_classifies_dense_and_scaled_weights() &&
      test_gemm_catalog_rejects_unknown_gemm_layouts();

  if (!ok) {
    return 1;
  }
  std::cout << "gemm_catalog_test: PASS\n";
  return 0;
}
