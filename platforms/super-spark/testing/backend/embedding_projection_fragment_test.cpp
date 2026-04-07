#include "nemotron/artifact_loader.h"
#include "nemotron/cublaslt_gemm_plan.h"
#include "nemotron/cublaslt_handle.h"
#include "nemotron/dense_weight.h"
#include "nemotron/device_tensor.h"
#include "nemotron/embedding_catalog.h"
#include "nemotron/embedding_projection_fragment.h"
#include "nemotron/embedding_table.h"
#include "nemotron/gemm_catalog.h"
#include "nemotron/gemm_execution.h"
#include "nemotron/gemm_planner.h"
#include "nemotron/kernel_catalog.h"
#include "nemotron/manifest.h"
#include "nemotron/tensor_catalog.h"
#include "nemotron/weight_arena.h"
#include "nemotron/weight_arena_plan.h"

#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using nemotron::ArtifactLoader;
using nemotron::BuildCublasLtGemmPlan;
using nemotron::BuildEmbeddingCatalog;
using nemotron::BuildGemmCatalog;
using nemotron::BuildGemmLaunchPlan;
using nemotron::BuildKernelCatalog;
using nemotron::BuildTensorCatalog;
using nemotron::BuildWeightArenaPlan;
using nemotron::CublasLtHandle;
using nemotron::DeviceDenseWeightFp32;
using nemotron::DeviceEmbeddingTableFp32;
using nemotron::DeviceTensorFp32;
using nemotron::EmbeddingCatalog;
using nemotron::GemmCatalog;
using nemotron::GemmHeuristicCache;
using nemotron::KernelCatalog;
using nemotron::LoadVerifiedManifestFromJsonFile;
using nemotron::ManifestLoadResult;
using nemotron::PrepareGemmExecution;
using nemotron::RunEmbeddingThenDenseProjectionFp32;
using nemotron::TensorCatalog;
using nemotron::WeightArena;

class TempDir {
 public:
  TempDir() {
    path_ = std::filesystem::temp_directory_path() /
            ("nemotron_embedding_projection_fragment_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
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

struct LoadedOperatorCatalogs {
  ManifestLoadResult manifest_result;
  std::unique_ptr<ArtifactLoader> loader;
  TensorCatalog tensor_catalog;
  std::unique_ptr<WeightArena> arena;
  KernelCatalog kernel_catalog;
  EmbeddingCatalog embedding_catalog;
  GemmCatalog gemm_catalog;
};

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

template <typename T>
std::string bytes_from_vector(const std::vector<T>& values) {
  return std::string(
      reinterpret_cast<const char*>(values.data()),
      values.size() * sizeof(T));
}

void write_file(const std::filesystem::path& path, const std::string& contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
}

std::string make_manifest_json(
    const std::string& embedding_file,
    const std::string& embedding_checksum,
    const std::string& projection_file,
    const std::string& projection_checksum) {
  std::ostringstream oss;
  oss
      << "{\n"
      << "  \"schema_version\": 1,\n"
      << "  \"model_id\": \"embedding-projection-fragment-test\",\n"
      << "  \"source_revision\": \"test\",\n"
      << "  \"tokenizer_revision\": \"test\",\n"
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
      << "      \"name\": \"lm_head.weight\",\n"
      << "      \"op_class\": \"dense_linear\",\n"
      << "      \"logical_shape\": [2, 3],\n"
      << "      \"packed_shape\": [2, 3],\n"
      << "      \"storage_dtype\": \"fp32\",\n"
      << "      \"compute_dtype\": \"fp32\",\n"
      << "      \"layout_tag\": \"row_major\",\n"
      << "      \"alignment_bytes\": 16,\n"
      << "      \"packed_file\": \"" << projection_file << "\",\n"
      << "      \"offset_bytes\": 0,\n"
      << "      \"nbytes\": 24,\n"
      << "      \"source_tensor_name\": \"lm_head.weight\",\n"
      << "      \"checksum\": \"fnv1a64:" << projection_checksum << "\"\n"
      << "    }\n"
      << "  ]\n"
      << "}\n";
  return oss.str();
}

LoadedOperatorCatalogs load_operator_catalogs(const std::filesystem::path& manifest_path) {
  LoadedOperatorCatalogs loaded;
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
  loaded.embedding_catalog = BuildEmbeddingCatalog(loaded.kernel_catalog);
  if (!loaded.embedding_catalog.valid()) {
    return loaded;
  }
  loaded.gemm_catalog = BuildGemmCatalog(loaded.kernel_catalog);
  return loaded;
}

std::vector<float> cpu_reference(
    const std::vector<float>& embedding_weights,
    const std::vector<std::int32_t>& token_ids,
    const std::vector<float>& projection_weights,
    std::size_t embedding_dim,
    std::size_t output_dim) {
  std::vector<float> output(token_ids.size() * output_dim, 0.0f);
  for (std::size_t token_index = 0; token_index < token_ids.size(); ++token_index) {
    const std::size_t token_id = static_cast<std::size_t>(token_ids[token_index]);
    for (std::size_t out = 0; out < output_dim; ++out) {
      float acc = 0.0f;
      for (std::size_t dim = 0; dim < embedding_dim; ++dim) {
        const float embedding = embedding_weights[token_id * embedding_dim + dim];
        const float weight = projection_weights[out * embedding_dim + dim];
        acc += embedding * weight;
      }
      output[token_index * output_dim + out] = acc;
    }
  }
  return output;
}

bool nearly_equal(const std::vector<float>& lhs, const std::vector<float>& rhs, float tol) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    const float diff = lhs[i] - rhs[i];
    if (diff > tol || diff < -tol) {
      return false;
    }
  }
  return true;
}

bool test_embedding_projection_fragment_matches_cpu_reference() {
  if (!has_cuda_device()) {
    std::cout << "embedding_projection_fragment_test: SKIP (no CUDA device available)\n";
    return true;
  }

  const auto handle = CublasLtHandle::Create();
  if (!handle || !handle->valid()) {
    std::cout << "embedding_projection_fragment_test: SKIP (cublasLt unavailable)\n";
    return true;
  }

  TempDir temp_dir;
  const std::vector<float> embedding_weights = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
      10.0f, 11.0f, 12.0f,
  };
  const std::vector<float> projection_weights = {
      1.0f, 0.0f, -1.0f,
      0.5f, 1.0f, 2.0f,
  };
  const std::string embedding_bytes = bytes_from_vector(embedding_weights);
  const std::string projection_bytes = bytes_from_vector(projection_weights);
  write_file(temp_dir.path() / "weights" / "embedding.bin", embedding_bytes);
  write_file(temp_dir.path() / "weights" / "projection.bin", projection_bytes);
  write_file(
      temp_dir.path() / "manifest.json",
      make_manifest_json(
          "weights/embedding.bin",
          fnv1a64_hex(embedding_bytes),
          "weights/projection.bin",
          fnv1a64_hex(projection_bytes)));

  const LoadedOperatorCatalogs loaded = load_operator_catalogs(temp_dir.path() / "manifest.json");
  if (!expect(loaded.embedding_catalog.valid(), "embedding catalog should load before fragment execution")) {
    return false;
  }
  if (!expect(loaded.gemm_catalog.valid(), "GEMM catalog should load before fragment execution")) {
    return false;
  }

  const auto* embedding = loaded.embedding_catalog.FindDescriptor("backbone.embeddings.weight");
  const auto* projection = loaded.gemm_catalog.FindDescriptor("lm_head.weight");
  if (!expect(embedding != nullptr, "embedding descriptor should exist before fragment execution")) {
    return false;
  }
  if (!expect(projection != nullptr, "projection descriptor should exist before fragment execution")) {
    return false;
  }

  auto table = DeviceEmbeddingTableFp32::Upload(*embedding);
  if (!expect(static_cast<bool>(table), "embedding table upload should succeed before fragment execution")) {
    return false;
  }
  auto projection_weight = DeviceDenseWeightFp32::Upload(*projection);
  if (!expect(static_cast<bool>(projection_weight), "projection weight upload should succeed before fragment execution")) {
    return false;
  }

  const auto launch_plan = BuildGemmLaunchPlan(*projection, 2);
  if (!expect(launch_plan.has_value(), "projection launch plan should build")) {
    return false;
  }
  GemmHeuristicCache cache;
  const auto execution = PrepareGemmExecution(*launch_plan, &cache);
  if (!expect(execution.has_value(), "projection execution should prepare")) {
    return false;
  }
  const auto cublaslt_plan = BuildCublasLtGemmPlan(*execution);
  if (!expect(cublaslt_plan.has_value(), "projection cublasLt plan should build")) {
    return false;
  }

  const std::vector<std::int32_t> token_ids = {2, 0};
  auto output = DeviceTensorFp32::Create({token_ids.size(), projection->output_rows});
  if (!expect(output && output->valid(), "fragment output tensor should allocate")) {
    return false;
  }

  const auto stats = RunEmbeddingThenDenseProjectionFp32(
      *table,
      token_ids.data(),
      token_ids.size(),
      *projection_weight,
      *handle,
      *cublaslt_plan,
      output.get());
  if (!expect(stats.has_value(), "embedding projection fragment should execute successfully")) {
    return false;
  }

  std::vector<float> host_output(output->numel(), 0.0f);
  if (!expect(output->CopyToHost(host_output.data(), host_output.size()), "fragment output should download")) {
    return false;
  }

  const auto reference = cpu_reference(
      embedding_weights,
      token_ids,
      projection_weights,
      embedding->embedding_dim,
      projection->output_rows);
  return expect(stats->token_count == token_ids.size() && stats->embedding_dim == embedding->embedding_dim,
                "fragment should report the expected embedding shape") &&
         expect(stats->projection.rows == token_ids.size() && stats->projection.cols == projection->output_rows,
                "fragment should report the expected projection shape") &&
         expect(nearly_equal(host_output, reference, 1e-4f),
                "embedding projection fragment should match the CPU reference");
}

}  // namespace

int main() {
  if (!test_embedding_projection_fragment_matches_cpu_reference()) {
    return 1;
  }
  std::cout << "embedding_projection_fragment_test: PASS\n";
  return 0;
}
