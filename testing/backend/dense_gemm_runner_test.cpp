#include "nemotron/artifact_loader.h"
#include "nemotron/cublaslt_gemm_plan.h"
#include "nemotron/cublaslt_handle.h"
#include "nemotron/dense_weight.h"
#include "nemotron/dense_gemm_runner.h"
#include "nemotron/device_tensor.h"
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
#include <vector>

namespace {

using nemotron::ArtifactLoader;
using nemotron::BuildCublasLtGemmPlan;
using nemotron::BuildGemmCatalog;
using nemotron::BuildGemmLaunchPlan;
using nemotron::BuildKernelCatalog;
using nemotron::BuildTensorCatalog;
using nemotron::BuildWeightArenaPlan;
using nemotron::CublasLtHandle;
using nemotron::DeviceDenseWeightFp32;
using nemotron::DeviceTensorFp32;
using nemotron::GemmCatalog;
using nemotron::GemmHeuristicCache;
using nemotron::KernelCatalog;
using nemotron::LoadVerifiedManifestFromJsonFile;
using nemotron::ManifestLoadResult;
using nemotron::PrepareGemmExecution;
using nemotron::RunDenseRowMajorFp32;
using nemotron::RunDenseRowMajorFp32ToDevice;
using nemotron::TensorCatalog;
using nemotron::WeightArena;

class TempDir {
 public:
  TempDir() {
    path_ = std::filesystem::temp_directory_path() /
            ("nemotron_dense_gemm_runner_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
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

struct LoadedDenseCatalog {
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
    const std::string& dense_file,
    const std::string& dense_checksum) {
  std::ostringstream oss;
  oss
      << "{\n"
      << "  \"schema_version\": 1,\n"
      << "  \"model_id\": \"dense-gemm-test\",\n"
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
      << "      \"name\": \"mlp.up_proj\",\n"
      << "      \"op_class\": \"dense_linear\",\n"
      << "      \"logical_shape\": [2, 3],\n"
      << "      \"packed_shape\": [2, 3],\n"
      << "      \"storage_dtype\": \"fp32\",\n"
      << "      \"compute_dtype\": \"fp32\",\n"
      << "      \"layout_tag\": \"row_major\",\n"
      << "      \"alignment_bytes\": 16,\n"
      << "      \"packed_file\": \"" << dense_file << "\",\n"
      << "      \"offset_bytes\": 0,\n"
      << "      \"nbytes\": 24,\n"
      << "      \"source_tensor_name\": \"mlp.up_proj\",\n"
      << "      \"checksum\": \"fnv1a64:" << dense_checksum << "\"\n"
      << "    }\n"
      << "  ]\n"
      << "}\n";
  return oss.str();
}

LoadedDenseCatalog load_dense_catalog(const std::filesystem::path& manifest_path) {
  LoadedDenseCatalog loaded;
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

std::vector<float> cpu_reference(
    const std::vector<float>& activations,
    std::size_t m,
    const std::vector<float>& weights,
    std::size_t n,
    std::size_t k) {
  std::vector<float> output(m * n, 0.0f);
  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t out = 0; out < n; ++out) {
      float acc = 0.0f;
      for (std::size_t col = 0; col < k; ++col) {
        acc += activations[row * k + col] * weights[out * k + col];
      }
      output[row * n + out] = acc;
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

bool test_dense_gemm_runner_executes_against_cpu_reference() {
  const auto handle = CublasLtHandle::Create();
  if (!handle || !handle->valid()) {
    std::cout << "dense_gemm_runner_test: SKIP (no CUDA device or cublasLt unavailable)\n";
    return true;
  }

  TempDir temp_dir;
  const std::filesystem::path dense_path = temp_dir.path() / "weights" / "dense.bin";
  const std::vector<float> weights = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
  };
  const std::string weight_bytes = bytes_from_vector(weights);
  write_file(dense_path, weight_bytes);

  const std::filesystem::path manifest_path = temp_dir.path() / "manifest.json";
  write_file(
      manifest_path,
      make_manifest_json("weights/dense.bin", fnv1a64_hex(weight_bytes)));

  const LoadedDenseCatalog loaded = load_dense_catalog(manifest_path);
  if (!expect(loaded.gemm_catalog.valid(), "dense GEMM catalog should be valid before execution")) {
    return false;
  }

  const auto* dense = loaded.gemm_catalog.FindDescriptor("mlp.up_proj");
  if (!expect(dense != nullptr, "dense GEMM descriptor should exist")) {
    return false;
  }

  const std::vector<float> activations = {
      1.0f, 0.0f, -1.0f,
      2.0f, 1.0f, 0.0f,
  };
  const auto launch_plan = BuildGemmLaunchPlan(*dense, 2);
  if (!expect(launch_plan.has_value(), "dense execution requires a valid launch plan")) {
    return false;
  }

  GemmHeuristicCache cache;
  const auto execution = PrepareGemmExecution(*launch_plan, &cache);
  if (!expect(execution.has_value(), "dense execution should prepare successfully")) {
    return false;
  }
  const auto cublaslt_plan = BuildCublasLtGemmPlan(*execution);
  if (!expect(cublaslt_plan.has_value(), "dense execution should build a cublasLt plan")) {
    return false;
  }

  const auto result = RunDenseRowMajorFp32(*handle, *cublaslt_plan, activations.data(), 2);
  if (!expect(result.has_value(), "dense GEMM runner should execute successfully")) {
    return false;
  }

  const auto reference = cpu_reference(activations, 2, weights, 2, 3);
  return expect(result->rows == 2 && result->cols == 2, "dense GEMM runner should report the correct output shape") &&
         expect(result->heuristic_count > 0, "dense GEMM runner should find at least one cuBLASLt heuristic") &&
         expect(nearly_equal(result->output, reference, 1e-4f),
                "dense GEMM runner output should match the CPU reference");
}

bool test_dense_gemm_runner_device_tensor_path_executes_against_cpu_reference() {
  const auto handle = CublasLtHandle::Create();
  if (!handle || !handle->valid()) {
    std::cout << "dense_gemm_runner_test: SKIP (no CUDA device or cublasLt unavailable)\n";
    return true;
  }

  TempDir temp_dir;
  const std::filesystem::path dense_path = temp_dir.path() / "weights" / "dense.bin";
  const std::vector<float> weights = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
  };
  const std::string weight_bytes = bytes_from_vector(weights);
  write_file(dense_path, weight_bytes);

  const std::filesystem::path manifest_path = temp_dir.path() / "manifest.json";
  write_file(
      manifest_path,
      make_manifest_json("weights/dense.bin", fnv1a64_hex(weight_bytes)));

  const LoadedDenseCatalog loaded = load_dense_catalog(manifest_path);
  if (!expect(loaded.gemm_catalog.valid(), "dense GEMM catalog should be valid before device execution")) {
    return false;
  }

  const auto* dense = loaded.gemm_catalog.FindDescriptor("mlp.up_proj");
  if (!expect(dense != nullptr, "dense GEMM descriptor should exist before device execution")) {
    return false;
  }

  const std::vector<float> host_activations = {
      1.0f, 0.0f, -1.0f,
      2.0f, 1.0f, 0.0f,
  };
  auto device_activations = DeviceTensorFp32::Create({2, 3});
  auto device_output = DeviceTensorFp32::Create({2, 2});
  if (!expect(device_activations && device_output, "device tensors should allocate successfully")) {
    return false;
  }
  if (!expect(device_activations->CopyFromHost(host_activations.data(), host_activations.size()),
              "device activations should upload successfully")) {
    return false;
  }

  const auto launch_plan = BuildGemmLaunchPlan(*dense, 2);
  if (!expect(launch_plan.has_value(), "device execution requires a valid launch plan")) {
    return false;
  }

  GemmHeuristicCache cache;
  const auto execution = PrepareGemmExecution(*launch_plan, &cache);
  if (!expect(execution.has_value(), "device execution should prepare successfully")) {
    return false;
  }
  const auto cublaslt_plan = BuildCublasLtGemmPlan(*execution);
  if (!expect(cublaslt_plan.has_value(), "device execution should build a cublasLt plan")) {
    return false;
  }

  auto device_weight = DeviceDenseWeightFp32::Upload(*dense);
  if (!expect(static_cast<bool>(device_weight), "dense weight upload should succeed before device execution")) {
    return false;
  }

  const auto stats = RunDenseRowMajorFp32ToDevice(
      *handle,
      *cublaslt_plan,
      *device_weight,
      *device_activations,
      device_output.get());
  if (!expect(stats.has_value(), "device tensor dense GEMM runner should execute successfully")) {
    return false;
  }

  std::vector<float> host_output(device_output->numel(), 0.0f);
  if (!expect(device_output->CopyToHost(host_output.data(), host_output.size()),
              "device output should download successfully")) {
    return false;
  }

  const auto reference = cpu_reference(host_activations, 2, weights, 2, 3);
  return expect(stats->rows == 2 && stats->cols == 2, "device tensor runner should report the correct output shape") &&
         expect(stats->heuristic_count > 0, "device tensor runner should find at least one cuBLASLt heuristic") &&
         expect(nearly_equal(host_output, reference, 1e-4f),
                "device tensor runner output should match the CPU reference");
}

bool test_dense_gemm_runner_rejects_non_fp32_dense_descriptor() {
  const auto handle = CublasLtHandle::Create();
  if (!handle || !handle->valid()) {
    std::cout << "dense_gemm_runner_test: SKIP (no CUDA device or cublasLt unavailable)\n";
    return true;
  }

  TempDir temp_dir;
  const std::filesystem::path dense_path = temp_dir.path() / "weights" / "dense.bin";
  const std::vector<float> weights = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
  };
  const std::string weight_bytes = bytes_from_vector(weights);
  write_file(dense_path, weight_bytes);

  const std::filesystem::path manifest_path = temp_dir.path() / "manifest.json";
  write_file(
      manifest_path,
      make_manifest_json("weights/dense.bin", fnv1a64_hex(weight_bytes)));

  const LoadedDenseCatalog loaded = load_dense_catalog(manifest_path);
  if (!expect(loaded.gemm_catalog.valid(), "dense GEMM catalog should be valid before rejection test")) {
    return false;
  }

  const auto* dense = loaded.gemm_catalog.FindDescriptor("mlp.up_proj");
  if (!expect(dense != nullptr, "dense GEMM descriptor should exist before rejection test")) {
    return false;
  }

  const auto launch_plan = BuildGemmLaunchPlan(*dense, 2);
  if (!expect(launch_plan.has_value(), "dense launch plan should exist before rejection test")) {
    return false;
  }

  GemmHeuristicCache cache;
  const auto execution = PrepareGemmExecution(*launch_plan, &cache);
  if (!expect(execution.has_value(), "dense execution should prepare before rejection test")) {
    return false;
  }

  const auto plan = BuildCublasLtGemmPlan(*execution);
  if (!expect(plan.has_value(), "plan build should succeed before dtype rejection")) {
    return false;
  }

  nemotron::GemmDescriptor unsupported_descriptor = *dense;
  unsupported_descriptor.storage_dtype = "bf16";
  unsupported_descriptor.compute_dtype = "bf16";
  auto unsupported_plan = *plan;
  unsupported_plan.execution.launch_plan.descriptor = &unsupported_descriptor;

  const float activations[6] = {};
  return expect(!RunDenseRowMajorFp32(*handle, unsupported_plan, activations, 2).has_value(),
                "first real dense GEMM runner should reject non-fp32 dense descriptors");
}

}  // namespace

int main() {
  const bool ok =
      test_dense_gemm_runner_executes_against_cpu_reference() &&
      test_dense_gemm_runner_device_tensor_path_executes_against_cpu_reference() &&
      test_dense_gemm_runner_rejects_non_fp32_dense_descriptor();

  if (!ok) {
    return 1;
  }
  std::cout << "dense_gemm_runner_test: PASS\n";
  return 0;
}
