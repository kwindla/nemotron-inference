#include "nemotron/artifact_loader.h"
#include "nemotron/cublaslt_gemm_plan.h"
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
using nemotron::BuildCublasLtGemmPlan;
using nemotron::CublasLtContract;
using nemotron::CublasLtMatrixOrder;
using nemotron::BuildGemmCatalog;
using nemotron::BuildGemmLaunchPlan;
using nemotron::BuildKernelCatalog;
using nemotron::BuildTensorCatalog;
using nemotron::BuildWeightArenaPlan;
using nemotron::CublasLtScaleMode;
using nemotron::CublasLtTransform;
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
            ("nemotron_cublaslt_gemm_plan_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
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

bool test_cublaslt_gemm_plan_builds_for_dense_and_nvfp4() {
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
  if (!expect(loaded.gemm_catalog.valid(), "GEMM catalog should be valid before cublasLt plan build")) {
    return false;
  }

  const auto dense_launch =
      BuildGemmLaunchPlan(*loaded.gemm_catalog.FindDescriptor("mlp.up_proj"), 16);
  const auto expert_launch =
      BuildGemmLaunchPlan(*loaded.gemm_catalog.FindDescriptor("layers.12.experts.117.w1"), 16);
  GemmHeuristicCache cache;
  const auto dense_execution = PrepareGemmExecution(*dense_launch, &cache);
  const auto expert_execution = PrepareGemmExecution(*expert_launch, &cache);
  const auto dense_plan = BuildCublasLtGemmPlan(*dense_execution);
  const auto expert_plan = BuildCublasLtGemmPlan(*expert_execution);
  return expect(dense_plan.has_value(), "dense execution should produce a cublasLt plan") &&
         expect(expert_plan.has_value(), "scaled execution should produce a cublasLt plan") &&
         expect(dense_plan->contract == CublasLtContract::kRowMajorA_N_RowMajorB_T,
                "dense plan should encode the validated row-major NT contract") &&
         expect(expert_plan->contract == CublasLtContract::kRowMajorA_N_RowMajorB_T,
                "NVFP4 plan should encode the validated row-major NT contract") &&
         expect(dense_plan->transform_a == CublasLtTransform::kNone, "dense plan should keep A untransposed") &&
         expect(dense_plan->transform_b == CublasLtTransform::kTranspose, "dense plan should transpose the packed weight operand") &&
         expect(dense_plan->order_a == CublasLtMatrixOrder::kRowMajor &&
                    dense_plan->order_b == CublasLtMatrixOrder::kRowMajor &&
                    dense_plan->order_c == CublasLtMatrixOrder::kRowMajor,
                "dense plan should keep row-major operand orders") &&
         expect(expert_plan->order_a == CublasLtMatrixOrder::kRowMajor &&
                    expert_plan->order_b == CublasLtMatrixOrder::kRowMajor &&
                    expert_plan->order_c == CublasLtMatrixOrder::kRowMajor,
                "NVFP4 plan should keep row-major operand orders") &&
         expect(dense_plan->scale_mode == CublasLtScaleMode::kNone, "dense plan should not use matrix scaling") &&
         expect(expert_plan->scale_mode == CublasLtScaleMode::kVec16UE4M3, "NVFP4 plan should use vec16 UE4M3 scaling") &&
         expect(dense_plan->lda == 2 && dense_plan->ldb == 2 && dense_plan->ldc == 4,
                "dense cublasLt plan should derive row-major leading dimensions") &&
         expect(expert_plan->lda == 1024 && expert_plan->ldb == 1024 && expert_plan->ldc == 2688,
                "NVFP4 cublasLt plan should derive leading dimensions from logical shape") &&
         expect(dense_plan->packed_alignment_ok, "dense plan should satisfy the alignment contract") &&
         expect(expert_plan->packed_alignment_ok, "NVFP4 plan should satisfy the alignment contract") &&
         expect(expert_plan->block_scales_alignment_ok, "NVFP4 block scales should satisfy the alignment contract") &&
         expect(expert_plan->tensor_scale_alignment_ok, "NVFP4 tensor scale should satisfy the alignment contract");
}

bool test_cublaslt_gemm_plan_rejects_misaligned_dense_operand() {
  nemotron::PreparedGemmExecution execution;
  execution.backend_kind = nemotron::GemmBackendKind::kCublasLtDense;
  static const std::uint8_t bytes[17] = {};
  execution.launch_plan.m = 8;
  execution.launch_plan.n = 4;
  execution.launch_plan.k = 2;
  execution.launch_plan.packed_bytes = nemotron::ByteRangeView{bytes + 1, 8};
  execution.launch_plan.heuristic_key = "dense|test";
  return expect(!BuildCublasLtGemmPlan(execution).has_value(),
                "misaligned packed weights should reject the cublasLt plan");
}

}  // namespace

int main() {
  const bool ok =
      test_cublaslt_gemm_plan_builds_for_dense_and_nvfp4() &&
      test_cublaslt_gemm_plan_rejects_misaligned_dense_operand();

  if (!ok) {
    return 1;
  }
  std::cout << "cublaslt_gemm_plan_test: PASS\n";
  return 0;
}
