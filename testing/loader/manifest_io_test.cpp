#include "nemotron/tensor_catalog.h"
#include "nemotron/artifact_loader.h"
#include "nemotron/embedding_catalog.h"
#include "nemotron/gemm_catalog.h"
#include "nemotron/gemm_planner.h"
#include "nemotron/kernel_catalog.h"
#include "nemotron/manifest.h"
#include "nemotron/model_schedule.h"
#include "nemotron/runtime_environment.h"
#include "nemotron/weight_arena.h"
#include "nemotron/weight_arena_plan.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

using nemotron::ArtifactLoadMode;
using nemotron::LoadVerifiedManifestFromJsonFile;
using nemotron::ManifestLoadResult;
using nemotron::RuntimeBootstrapOptions;
using nemotron::RuntimeEnvironment;

class TempDir {
 public:
  TempDir() {
    path_ = std::filesystem::temp_directory_path() /
            ("nemotron_manifest_io_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
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
    const std::string& embedding_file,
    const std::string& embedding_checksum,
    const std::string& packed_file,
    const std::string& checksum,
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
      << "      \"name\": \"backbone.embeddings.weight\",\n"
      << "      \"op_class\": \"embedding\",\n"
      << "      \"logical_shape\": [4, 3],\n"
      << "      \"packed_shape\": [4, 3],\n"
      << "      \"storage_dtype\": \"fp32\",\n"
      << "      \"compute_dtype\": \"fp32\",\n"
      << "      \"layout_tag\": \"row_major\",\n"
      << "      \"alignment_bytes\": 16,\n"
      << "      \"packed_file\": \"" << embedding_file << "\",\n"
      << "      \"offset_bytes\": 0,\n"
      << "      \"nbytes\": 48,\n"
      << "      \"source_tensor_name\": \"backbone.embeddings.weight\",\n"
      << "      \"checksum\": \"fnv1a64:" << embedding_checksum << "\"\n"
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
      << "      \"packed_file\": \"" << packed_file << "\",\n"
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
      << "      \"checksum\": \"fnv1a64:" << checksum << "\"\n"
      << "    }\n"
      << "  ]\n"
      << "}\n";
  return oss.str();
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

bool test_load_verified_manifest_from_file() {
  TempDir temp_dir;
  const std::filesystem::path packed_path = temp_dir.path() / "weights" / "experts.bin";
  const std::filesystem::path embedding_path = temp_dir.path() / "weights" / "embedding.bin";
  const std::filesystem::path auxiliary_path = temp_dir.path() / "weights" / "experts_scales.bin";
  const std::string packed_bytes = "0123456789";
  const std::string verified_slice = packed_bytes.substr(2, 5);
  const std::vector<float> embedding_values = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
      10.0f, 11.0f, 12.0f,
  };
  const std::string embedding_bytes(
      reinterpret_cast<const char*>(embedding_values.data()),
      embedding_values.size() * sizeof(float));
  write_file(packed_path, packed_bytes);
  write_file(embedding_path, embedding_bytes);
  write_file(auxiliary_path, "aux01234");

  const std::filesystem::path manifest_path = temp_dir.path() / "manifest.json";
  write_file(
      manifest_path,
      make_manifest_json(
          "weights/embedding.bin",
          fnv1a64_hex(embedding_bytes),
          "weights/experts.bin",
          fnv1a64_hex(verified_slice),
          "weights/experts_scales.bin"));

  const ManifestLoadResult result = LoadVerifiedManifestFromJsonFile(manifest_path);
  return expect(result.ok, "valid manifest should load and verify from disk") &&
         expect(result.manifest.tensors.size() == 2, "loaded manifest should contain both embedding and expert tensors") &&
         expect(result.manifest.tensors[1].packed_file == "weights/experts.bin", "packed file path should round-trip") &&
         expect(result.issues.empty(), "verified manifest should not produce issues");
}

bool test_missing_packed_file_fails_verification() {
  TempDir temp_dir;
  const std::filesystem::path auxiliary_path = temp_dir.path() / "weights" / "experts_scales.bin";
  const std::filesystem::path embedding_path = temp_dir.path() / "weights" / "embedding.bin";
  const std::vector<float> embedding_values = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
      10.0f, 11.0f, 12.0f,
  };
  const std::string embedding_bytes(
      reinterpret_cast<const char*>(embedding_values.data()),
      embedding_values.size() * sizeof(float));
  write_file(embedding_path, embedding_bytes);
  write_file(auxiliary_path, "aux01234");

  const std::filesystem::path manifest_path = temp_dir.path() / "manifest.json";
  write_file(
      manifest_path,
      make_manifest_json(
          "weights/embedding.bin",
          fnv1a64_hex(embedding_bytes),
          "weights/missing.bin",
          "0000000000000000",
          "weights/experts_scales.bin"));

  const ManifestLoadResult result = LoadVerifiedManifestFromJsonFile(manifest_path);
  return expect(!result.ok, "missing packed file should fail verification") &&
         expect(!result.issues.empty(), "verification failure should report issues");
}

bool test_runtime_environment_builds_from_manifest_file() {
  TempDir temp_dir;
  const std::filesystem::path packed_path = temp_dir.path() / "weights" / "experts.bin";
  const std::filesystem::path embedding_path = temp_dir.path() / "weights" / "embedding.bin";
  const std::filesystem::path auxiliary_path = temp_dir.path() / "weights" / "experts_scales.bin";
  const std::string packed_bytes = "abcdefghij";
  const std::vector<float> embedding_values = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
      10.0f, 11.0f, 12.0f,
  };
  const std::string embedding_bytes(
      reinterpret_cast<const char*>(embedding_values.data()),
      embedding_values.size() * sizeof(float));
  write_file(packed_path, packed_bytes);
  write_file(embedding_path, embedding_bytes);
  write_file(auxiliary_path, "aux12345");

  const std::filesystem::path manifest_path = temp_dir.path() / "manifest.json";
  write_file(
      manifest_path,
      make_manifest_json(
          "weights/embedding.bin",
          fnv1a64_hex(embedding_bytes),
          "weights/experts.bin",
          fnv1a64_hex(packed_bytes.substr(2, 5)),
          "weights/experts_scales.bin"));

  const auto environment = RuntimeEnvironment::BuildFromManifestFile(manifest_path, make_options());
  return expect(static_cast<bool>(environment), "runtime environment should build from a verified manifest file") &&
         expect(environment->loader_plan().valid, "loader plan should be valid from manifest file") &&
         expect(environment->has_artifact_loader(), "manifest-backed bootstrap should retain an artifact loader") &&
         expect(environment->artifact_loader() != nullptr, "artifact loader accessor should expose the retained loader") &&
         expect(environment->artifact_loader()->load_mode() == ArtifactLoadMode::kMmap,
                "default manifest-backed bootstrap should use mmap loading") &&
         expect(environment->has_tensor_catalog(), "manifest-backed bootstrap should retain a tensor catalog") &&
         expect(environment->tensor_catalog() != nullptr, "tensor catalog accessor should expose the retained catalog") &&
         expect(environment->tensor_catalog()->valid(), "retained tensor catalog should be valid") &&
         expect(environment->tensor_catalog()->FindTensor("layers.12.experts.117.w1") != nullptr,
                "manifest-backed environment should index the mapped tensor in its catalog") &&
         expect(environment->has_model_schedule(), "manifest-backed bootstrap should retain a model schedule") &&
         expect(environment->model_schedule() != nullptr, "model schedule accessor should expose the retained schedule") &&
         expect(environment->model_schedule()->valid(), "retained model schedule should be valid") &&
         expect(environment->model_schedule()->ordered_layers().size() == 1,
                "test manifest should expose one scheduled layer") &&
         expect(environment->model_schedule()->routed_expert_layer_count() == 1,
                "scheduled layer should be classified as routed-expert-backed") &&
         expect(environment->tensor_catalog()
                        ->FindTensor("layers.12.experts.117.w1")
                        ->FindAuxiliary(nemotron::TensorAuxiliaryRole::kBlockScales) != nullptr,
                "tensor catalog should classify block scale metadata by role") &&
         expect(environment->has_weight_arena_plan(), "manifest-backed bootstrap should retain a weight arena plan") &&
         expect(environment->weight_arena_plan() != nullptr, "weight arena plan accessor should expose the retained plan") &&
         expect(environment->weight_arena_plan()->valid(), "retained weight arena plan should be valid") &&
         expect(environment->weight_arena_plan()->FindTensor("layers.12.experts.117.w1") != nullptr,
                "weight arena plan should index the mapped tensor") &&
         expect(environment->has_weight_arena(), "manifest-backed bootstrap should materialize a weight arena") &&
         expect(environment->weight_arena() != nullptr, "weight arena accessor should expose the materialized arena") &&
         expect(environment->weight_arena()->valid(), "materialized weight arena should be valid") &&
         expect(environment->weight_arena()->FindTensor("layers.12.experts.117.w1").has_value(),
                "weight arena should expose copied tensor bytes") &&
         expect(environment->has_kernel_catalog(), "manifest-backed bootstrap should retain a kernel catalog") &&
         expect(environment->kernel_catalog() != nullptr, "kernel catalog accessor should expose the retained catalog") &&
         expect(environment->kernel_catalog()->valid(), "retained kernel catalog should be valid") &&
         expect(environment->kernel_catalog()->FindTensor("layers.12.experts.117.w1") != nullptr,
                "kernel catalog should expose the routed expert descriptor") &&
         expect(environment->has_gemm_catalog(), "manifest-backed bootstrap should retain a GEMM catalog") &&
         expect(environment->gemm_catalog() != nullptr, "GEMM catalog accessor should expose the retained catalog") &&
         expect(environment->gemm_catalog()->valid(), "retained GEMM catalog should be valid") &&
         expect(environment->gemm_catalog()->FindDescriptor("layers.12.experts.117.w1") != nullptr,
                "GEMM catalog should expose the routed expert GEMM descriptor") &&
         expect(environment->has_embedding_catalog(), "manifest-backed bootstrap should retain an embedding catalog") &&
         expect(environment->embedding_catalog() != nullptr, "embedding catalog accessor should expose the retained catalog") &&
         expect(environment->embedding_catalog()->valid(), "retained embedding catalog should be valid") &&
         expect(environment->embedding_catalog()->FindDescriptor("backbone.embeddings.weight") != nullptr,
                "embedding catalog should expose the embedding descriptor") &&
         expect(environment->has_gemm_heuristic_cache(), "manifest-backed bootstrap should create a GEMM heuristic cache") &&
         expect(environment->gemm_heuristic_cache() != nullptr, "heuristic-cache accessor should expose the runtime cache") &&
         expect(environment->gemm_heuristic_cache()->size() == 0, "new manifest-backed environment should start with an empty GEMM heuristic cache") &&
         expect(environment->artifact_loader()->FindTensor("layers.12.experts.117.w1").has_value(),
                "manifest-backed environment should expose mapped tensor views") &&
         expect(environment->prefix_cache().max_bytes() == environment->effective_shared_cache_budget_bytes(),
                "bootstrap from file should preserve effective cache budgeting");
}

bool test_runtime_environment_builds_from_manifest_file_with_read_all_loader() {
  TempDir temp_dir;
  const std::filesystem::path packed_path = temp_dir.path() / "weights" / "experts.bin";
  const std::filesystem::path embedding_path = temp_dir.path() / "weights" / "embedding.bin";
  const std::filesystem::path auxiliary_path = temp_dir.path() / "weights" / "experts_scales.bin";
  const std::string packed_bytes = "abcdefghij";
  const std::vector<float> embedding_values = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
      10.0f, 11.0f, 12.0f,
  };
  const std::string embedding_bytes(
      reinterpret_cast<const char*>(embedding_values.data()),
      embedding_values.size() * sizeof(float));
  write_file(packed_path, packed_bytes);
  write_file(embedding_path, embedding_bytes);
  write_file(auxiliary_path, "aux12345");

  const std::filesystem::path manifest_path = temp_dir.path() / "manifest.json";
  write_file(
      manifest_path,
      make_manifest_json(
          "weights/embedding.bin",
          fnv1a64_hex(embedding_bytes),
          "weights/experts.bin",
          fnv1a64_hex(packed_bytes.substr(2, 5)),
          "weights/experts_scales.bin"));

  auto options = make_options();
  options.artifact_load_mode = ArtifactLoadMode::kReadAll;
  const auto environment = RuntimeEnvironment::BuildFromManifestFile(manifest_path, options);
  return expect(static_cast<bool>(environment), "runtime environment should build from a verified manifest file with eager loading") &&
         expect(environment->has_artifact_loader(), "manifest-backed eager bootstrap should retain an artifact loader") &&
         expect(environment->artifact_loader() != nullptr, "artifact loader accessor should expose the eager loader") &&
         expect(environment->artifact_loader()->load_mode() == ArtifactLoadMode::kReadAll,
                "manifest-backed bootstrap should honor eager-read loading") &&
         expect(environment->artifact_loader()->loaded_file_count() == 3,
                "eager-read loader should retain all unique files");
}

bool test_runtime_environment_can_skip_weight_arena_materialization() {
  TempDir temp_dir;
  const std::filesystem::path packed_path = temp_dir.path() / "weights" / "experts.bin";
  const std::filesystem::path embedding_path = temp_dir.path() / "weights" / "embedding.bin";
  const std::filesystem::path auxiliary_path = temp_dir.path() / "weights" / "experts_scales.bin";
  const std::string packed_bytes = "abcdefghij";
  const std::vector<float> embedding_values = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
      10.0f, 11.0f, 12.0f,
  };
  const std::string embedding_bytes(
      reinterpret_cast<const char*>(embedding_values.data()),
      embedding_values.size() * sizeof(float));
  write_file(packed_path, packed_bytes);
  write_file(embedding_path, embedding_bytes);
  write_file(auxiliary_path, "aux12345");

  const std::filesystem::path manifest_path = temp_dir.path() / "manifest.json";
  write_file(
      manifest_path,
      make_manifest_json(
          "weights/embedding.bin",
          fnv1a64_hex(embedding_bytes),
          "weights/experts.bin",
          fnv1a64_hex(packed_bytes.substr(2, 5)),
          "weights/experts_scales.bin"));

  auto options = make_options();
  options.materialize_weight_arena = false;
  const auto environment = RuntimeEnvironment::BuildFromManifestFile(manifest_path, options);
  return expect(static_cast<bool>(environment), "runtime environment should build from a manifest without weight-arena materialization") &&
         expect(environment->has_artifact_loader(), "no-copy manifest-backed bootstrap should retain an artifact loader") &&
         expect(environment->has_tensor_catalog(), "no-copy manifest-backed bootstrap should retain a tensor catalog") &&
         expect(environment->has_kernel_catalog(), "no-copy manifest-backed bootstrap should still retain a kernel catalog") &&
         expect(environment->kernel_catalog() != nullptr, "kernel catalog accessor should expose the retained catalog") &&
         expect(environment->kernel_catalog()->valid(), "kernel catalog built from mapped bytes should be valid") &&
         expect(environment->kernel_catalog()->FindTensor("layers.12.experts.117.w1") != nullptr,
                "no-copy kernel catalog should expose the routed expert descriptor") &&
         expect(!environment->has_weight_arena_plan(), "no-copy manifest-backed bootstrap should skip the weight arena plan") &&
         expect(!environment->has_weight_arena(), "no-copy manifest-backed bootstrap should skip the materialized weight arena");
}

bool test_runtime_environment_can_skip_manifest_verification() {
  TempDir temp_dir;
  const std::filesystem::path packed_path = temp_dir.path() / "weights" / "experts.bin";
  const std::filesystem::path embedding_path = temp_dir.path() / "weights" / "embedding.bin";
  const std::filesystem::path auxiliary_path = temp_dir.path() / "weights" / "experts_scales.bin";
  const std::string packed_bytes = "abcdefghij";
  const std::vector<float> embedding_values = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
      10.0f, 11.0f, 12.0f,
  };
  const std::string embedding_bytes(
      reinterpret_cast<const char*>(embedding_values.data()),
      embedding_values.size() * sizeof(float));
  write_file(packed_path, packed_bytes);
  write_file(embedding_path, embedding_bytes);
  write_file(auxiliary_path, "aux12345");

  const std::filesystem::path manifest_path = temp_dir.path() / "manifest.json";
  write_file(
      manifest_path,
      make_manifest_json(
          "weights/embedding.bin",
          "0000000000000000",
          "weights/experts.bin",
          "0000000000000000",
          "weights/experts_scales.bin"));

  auto options = make_options();
  options.verify_manifest_files = false;
  const auto environment = RuntimeEnvironment::BuildFromManifestFile(manifest_path, options);
  return expect(static_cast<bool>(environment), "runtime environment should build when manifest verification is explicitly disabled") &&
         expect(environment->has_artifact_loader(), "unchecked manifest bootstrap should still retain an artifact loader") &&
         expect(environment->artifact_loader() != nullptr, "artifact loader accessor should expose the unchecked loader") &&
         expect(environment->artifact_loader()->load_mode() == ArtifactLoadMode::kMmap,
                "unchecked manifest bootstrap should preserve the selected artifact load mode");
}

}  // namespace

int main() {
  const bool ok =
      test_load_verified_manifest_from_file() &&
      test_missing_packed_file_fails_verification() &&
      test_runtime_environment_builds_from_manifest_file() &&
      test_runtime_environment_builds_from_manifest_file_with_read_all_loader() &&
      test_runtime_environment_can_skip_weight_arena_materialization() &&
      test_runtime_environment_can_skip_manifest_verification();

  if (!ok) {
    return 1;
  }
  std::cout << "manifest_io_test: PASS\n";
  return 0;
}
