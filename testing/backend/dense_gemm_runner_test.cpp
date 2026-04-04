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

#include <cuda_bf16.h>

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
using nemotron::DeviceDenseWeightBf16;
using nemotron::DeviceDenseWeightFp32;
using nemotron::DeviceTensorBf16;
using nemotron::DeviceTensorFp32;
using nemotron::GemmCatalog;
using nemotron::GemmHeuristicCache;
using nemotron::KernelCatalog;
using nemotron::LoadVerifiedManifestFromJsonFile;
using nemotron::ManifestLoadResult;
using nemotron::PrepareGemmExecution;
using nemotron::RunDenseRowMajorBf16ToDevice;
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
    const std::string& dense_checksum,
    const std::string& storage_dtype = "fp32",
    const std::string& compute_dtype = "fp32",
    std::size_t nbytes = 24) {
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
      << "      \"storage_dtype\": \"" << storage_dtype << "\",\n"
      << "      \"compute_dtype\": \"" << compute_dtype << "\",\n"
      << "      \"layout_tag\": \"row_major\",\n"
      << "      \"alignment_bytes\": 16,\n"
      << "      \"packed_file\": \"" << dense_file << "\",\n"
      << "      \"offset_bytes\": 0,\n"
      << "      \"nbytes\": " << nbytes << ",\n"
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

std::vector<__nv_bfloat16> to_bf16_vector(const std::vector<float>& values) {
  std::vector<__nv_bfloat16> converted(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    converted[i] = __float2bfloat16(values[i]);
  }
  return converted;
}

std::vector<float> from_bf16_vector(const std::vector<__nv_bfloat16>& values) {
  std::vector<float> converted(values.size(), 0.0f);
  for (std::size_t i = 0; i < values.size(); ++i) {
    converted[i] = __bfloat162float(values[i]);
  }
  return converted;
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

bool test_dense_gemm_runner_bf16_path_matches_fp32_reference() {
  const auto handle = CublasLtHandle::Create();
  if (!handle || !handle->valid()) {
    std::cout << "dense_gemm_runner_test: SKIP (no CUDA device or cublasLt unavailable)\n";
    return true;
  }

  TempDir temp_dir;
  const std::vector<float> weights = {
      1.15625f, -2.03125f, 3.1875f,
      -4.28125f, 5.40625f, -6.53125f,
  };
  const std::vector<float> activations = {
      0.9375f, -1.28125f, 2.46875f,
      -3.5625f, 4.6875f, -5.8125f,
  };
  const auto weights_bf16 = to_bf16_vector(weights);
  const auto activations_bf16 = to_bf16_vector(activations);
  const std::vector<float> rounded_weights = from_bf16_vector(weights_bf16);
  const std::vector<float> rounded_activations = from_bf16_vector(activations_bf16);

  const std::string weight_bytes_bf16 = bytes_from_vector(weights_bf16);
  write_file(temp_dir.path() / "weights" / "dense_bf16.bin", weight_bytes_bf16);
  write_file(
      temp_dir.path() / "manifest_bf16.json",
      make_manifest_json(
          "weights/dense_bf16.bin",
          fnv1a64_hex(weight_bytes_bf16),
          "bf16",
          "bf16",
          weight_bytes_bf16.size()));

  const std::string weight_bytes_fp32 = bytes_from_vector(rounded_weights);
  write_file(temp_dir.path() / "weights" / "dense_fp32.bin", weight_bytes_fp32);
  write_file(
      temp_dir.path() / "manifest_fp32.json",
      make_manifest_json(
          "weights/dense_fp32.bin",
          fnv1a64_hex(weight_bytes_fp32),
          "fp32",
          "fp32",
          weight_bytes_fp32.size()));

  const LoadedDenseCatalog loaded_bf16 = load_dense_catalog(temp_dir.path() / "manifest_bf16.json");
  const LoadedDenseCatalog loaded_fp32 = load_dense_catalog(temp_dir.path() / "manifest_fp32.json");
  if (!expect(loaded_bf16.gemm_catalog.valid(), "bf16 dense catalog should load for bf16 GEMM test") ||
      !expect(loaded_fp32.gemm_catalog.valid(), "fp32 dense catalog should load for bf16 GEMM test")) {
    return false;
  }

  const auto* dense_bf16 = loaded_bf16.gemm_catalog.FindDescriptor("mlp.up_proj");
  const auto* dense_fp32 = loaded_fp32.gemm_catalog.FindDescriptor("mlp.up_proj");
  if (!expect(dense_bf16 != nullptr, "bf16 dense descriptor should exist for bf16 GEMM test") ||
      !expect(dense_fp32 != nullptr, "fp32 dense descriptor should exist for bf16 GEMM test")) {
    return false;
  }

  const auto launch_plan_bf16 = BuildGemmLaunchPlan(*dense_bf16, 2);
  const auto launch_plan_fp32 = BuildGemmLaunchPlan(*dense_fp32, 2);
  if (!expect(launch_plan_bf16.has_value(), "bf16 launch plan should build") ||
      !expect(launch_plan_fp32.has_value(), "fp32 launch plan should build")) {
    return false;
  }

  GemmHeuristicCache cache;
  const auto execution_bf16 = PrepareGemmExecution(*launch_plan_bf16, &cache);
  const auto execution_fp32 = PrepareGemmExecution(*launch_plan_fp32, &cache);
  if (!expect(execution_bf16.has_value(), "bf16 execution should prepare") ||
      !expect(execution_fp32.has_value(), "fp32 execution should prepare")) {
    return false;
  }

  const auto plan_bf16 = BuildCublasLtGemmPlan(*execution_bf16);
  const auto plan_fp32 = BuildCublasLtGemmPlan(*execution_fp32);
  if (!expect(plan_bf16.has_value(), "bf16 cublasLt plan should build") ||
      !expect(plan_fp32.has_value(), "fp32 cublasLt plan should build")) {
    return false;
  }

  auto device_weight_bf16 = DeviceDenseWeightBf16::Upload(*dense_bf16);
  auto device_activations_bf16 = DeviceTensorBf16::Create({2, 3});
  auto device_output_bf16 = DeviceTensorBf16::Create({2, 2});
  if (!expect(device_weight_bf16 && device_weight_bf16->valid(), "bf16 dense weight upload should succeed") ||
      !expect(device_activations_bf16 && device_activations_bf16->valid(), "bf16 activation tensor should allocate") ||
      !expect(device_output_bf16 && device_output_bf16->valid(), "bf16 output tensor should allocate")) {
    return false;
  }
  if (!expect(
          device_activations_bf16->CopyFromHost(activations_bf16.data(), activations_bf16.size()),
          "bf16 activations should upload")) {
    return false;
  }

  const auto bf16_stats = RunDenseRowMajorBf16ToDevice(
      *handle,
      *plan_bf16,
      *device_weight_bf16,
      *device_activations_bf16,
      device_output_bf16.get());
  if (!expect(bf16_stats.has_value(), "bf16 dense GEMM runner should execute successfully")) {
    return false;
  }

  std::vector<__nv_bfloat16> host_output_bf16(device_output_bf16->numel(), __float2bfloat16(0.0f));
  if (!expect(
          device_output_bf16->CopyToHost(host_output_bf16.data(), host_output_bf16.size()),
          "bf16 dense GEMM output should download")) {
    return false;
  }

  const auto fp32_result = RunDenseRowMajorFp32(*handle, *plan_fp32, rounded_activations.data(), 2);
  if (!expect(fp32_result.has_value(), "fp32 dense GEMM runner should execute for bf16 comparison")) {
    return false;
  }

  const std::vector<float> bf16_output_fp32 = from_bf16_vector(host_output_bf16);
  const std::vector<float> fp32_output_bf16_round_trip = from_bf16_vector(to_bf16_vector(fp32_result->output));
  const auto reference = from_bf16_vector(to_bf16_vector(cpu_reference(
      rounded_activations,
      2,
      rounded_weights,
      2,
      3)));
  return expect(
             bf16_stats->rows == 2 && bf16_stats->cols == 2,
             "bf16 dense GEMM runner should report the correct output shape") &&
         expect(
             bf16_stats->heuristic_count > 0,
             "bf16 dense GEMM runner should find at least one cuBLASLt heuristic") &&
         expect(
             nearly_equal(bf16_output_fp32, fp32_output_bf16_round_trip, 2.0e-2f),
             "bf16 dense GEMM output should match fp32 GEMM after bf16 round-trip") &&
         expect(
             nearly_equal(bf16_output_fp32, reference, 2.0e-2f),
             "bf16 dense GEMM output should match the bf16-rounded CPU reference");
}

}  // namespace

int main() {
  const bool ok =
      test_dense_gemm_runner_executes_against_cpu_reference() &&
      test_dense_gemm_runner_device_tensor_path_executes_against_cpu_reference() &&
      test_dense_gemm_runner_rejects_non_fp32_dense_descriptor() &&
      test_dense_gemm_runner_bf16_path_matches_fp32_reference();

  if (!ok) {
    return 1;
  }
  std::cout << "dense_gemm_runner_test: PASS\n";
  return 0;
}
