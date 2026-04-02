#include "nemotron/artifact_loader.h"
#include "nemotron/dense_weight.h"
#include "nemotron/gemm_catalog.h"
#include "nemotron/kernel_catalog.h"
#include "nemotron/manifest.h"
#include "nemotron/tensor_catalog.h"
#include "nemotron/weight_arena.h"
#include "nemotron/weight_arena_plan.h"

#include <cuda_bf16.h>
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
using nemotron::BuildGemmCatalog;
using nemotron::BuildKernelCatalog;
using nemotron::BuildTensorCatalog;
using nemotron::BuildWeightArenaPlan;
using nemotron::DeviceDenseWeightFp32;
using nemotron::GemmCatalog;
using nemotron::KernelCatalog;
using nemotron::LoadVerifiedManifestFromJsonFile;
using nemotron::ManifestLoadResult;
using nemotron::TensorCatalog;
using nemotron::WeightArena;

class TempDir {
 public:
  TempDir() {
    path_ = std::filesystem::temp_directory_path() /
            ("nemotron_dense_weight_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
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
    const std::string& dense_file,
    const std::string& dense_checksum) {
  std::ostringstream oss;
  oss
      << "{\n"
      << "  \"schema_version\": 1,\n"
      << "  \"model_id\": \"dense-weight-test\",\n"
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

bool test_dense_weight_upload_round_trips_descriptor_bytes() {
  if (!has_cuda_device()) {
    std::cout << "dense_weight_test: SKIP (no CUDA device available)\n";
    return true;
  }

  TempDir temp_dir;
  const std::vector<float> weights = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
  };
  const std::string weight_bytes = bytes_from_vector(weights);
  write_file(temp_dir.path() / "weights" / "dense.bin", weight_bytes);
  write_file(
      temp_dir.path() / "manifest.json",
      make_manifest_json("weights/dense.bin", fnv1a64_hex(weight_bytes)));

  const LoadedDenseCatalog loaded = load_dense_catalog(temp_dir.path() / "manifest.json");
  if (!expect(loaded.gemm_catalog.valid(), "dense GEMM catalog should load before upload")) {
    return false;
  }

  const auto* dense = loaded.gemm_catalog.FindDescriptor("mlp.up_proj");
  if (!expect(dense != nullptr, "dense GEMM descriptor should exist before upload")) {
    return false;
  }

  auto uploaded = DeviceDenseWeightFp32::Upload(*dense);
  if (!expect(static_cast<bool>(uploaded), "dense weight upload should succeed")) {
    return false;
  }
  if (!expect(uploaded->valid(), "uploaded dense weight should be valid")) {
    return false;
  }
  if (!expect(uploaded->output_rows() == 2 && uploaded->input_cols() == 3,
              "uploaded dense weight should preserve matrix shape")) {
    return false;
  }

  std::vector<float> round_trip(uploaded->numel(), 0.0f);
  if (!expect(
          cudaMemcpy(
              round_trip.data(),
              uploaded->data(),
              uploaded->numel() * sizeof(float),
              cudaMemcpyDeviceToHost) == cudaSuccess,
          "uploaded dense weight should copy back to host")) {
    return false;
  }

  return expect(nearly_equal(round_trip, weights, 1e-6f),
                "uploaded dense weight contents should match the source descriptor bytes");
}

bool test_dense_weight_upload_rejects_non_fp32_descriptor() {
  if (!has_cuda_device()) {
    std::cout << "dense_weight_test: SKIP (no CUDA device available)\n";
    return true;
  }

  TempDir temp_dir;
  const std::vector<float> weights = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
  };
  const std::string weight_bytes = bytes_from_vector(weights);
  write_file(temp_dir.path() / "weights" / "dense.bin", weight_bytes);
  write_file(
      temp_dir.path() / "manifest.json",
      make_manifest_json("weights/dense.bin", fnv1a64_hex(weight_bytes)));

  const LoadedDenseCatalog loaded = load_dense_catalog(temp_dir.path() / "manifest.json");
  if (!expect(loaded.gemm_catalog.valid(), "dense GEMM catalog should load before rejection test")) {
    return false;
  }

  const auto* dense = loaded.gemm_catalog.FindDescriptor("mlp.up_proj");
  if (!expect(dense != nullptr, "dense GEMM descriptor should exist before rejection test")) {
    return false;
  }

  nemotron::GemmDescriptor unsupported = *dense;
  unsupported.storage_dtype = "int8";
  unsupported.compute_dtype = "int8";
  return expect(!DeviceDenseWeightFp32::Upload(unsupported),
                "device-resident dense weight upload should reject unsupported descriptors");
}

bool test_dense_weight_upload_accepts_bf16_descriptor() {
  if (!has_cuda_device()) {
    std::cout << "dense_weight_test: SKIP (no CUDA device available)\n";
    return true;
  }

  const std::vector<float> source_fp32 = {
      1.5f, -2.0f, 3.25f,
      4.5f, -5.75f, 6.0f,
  };
  std::vector<__nv_bfloat16> source_bf16(source_fp32.size());
  for (std::size_t i = 0; i < source_fp32.size(); ++i) {
    source_bf16[i] = __float2bfloat16(source_fp32[i]);
  }

  nemotron::GemmDescriptor descriptor;
  descriptor.tensor_name = "bf16.up_proj";
  descriptor.op_class = "dense_linear";
  descriptor.kernel_family = nemotron::GemmKernelFamily::kDenseRowMajor;
  descriptor.output_rows = 2;
  descriptor.input_cols = 3;
  descriptor.storage_dtype = "bf16";
  descriptor.compute_dtype = "bf16";
  descriptor.layout_tag = "row_major";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data = reinterpret_cast<const std::uint8_t*>(source_bf16.data());
  descriptor.packed_nbytes = source_bf16.size() * sizeof(__nv_bfloat16);

  auto uploaded = DeviceDenseWeightFp32::Upload(descriptor);
  if (!expect(static_cast<bool>(uploaded), "BF16 dense weight upload should succeed")) {
    return false;
  }

  std::vector<float> round_trip(source_fp32.size(), 0.0f);
  if (!expect(
          cudaMemcpy(
              round_trip.data(),
              uploaded->data(),
              round_trip.size() * sizeof(float),
              cudaMemcpyDeviceToHost) == cudaSuccess,
          "BF16 uploaded dense weight should copy back to host")) {
    return false;
  }

  std::vector<float> bf16_rounded(source_fp32.size(), 0.0f);
  for (std::size_t i = 0; i < source_fp32.size(); ++i) {
    bf16_rounded[i] = __bfloat162float(source_bf16[i]);
  }
  return expect(
      nearly_equal(round_trip, bf16_rounded, 1.0e-6f),
      "BF16 dense weight upload should convert values to FP32");
}

bool test_dense_weight_view_aliases_existing_device_storage() {
  if (!has_cuda_device()) {
    std::cout << "dense_weight_test: SKIP (no CUDA device available)\n";
    return true;
  }

  const std::vector<float> source = {
      0.5f, -1.0f, 1.5f,
      2.0f, -2.5f, 3.0f,
  };

  nemotron::GemmDescriptor descriptor;
  descriptor.tensor_name = "fp32.view";
  descriptor.op_class = "dense_linear";
  descriptor.kernel_family = nemotron::GemmKernelFamily::kDenseRowMajor;
  descriptor.output_rows = 2;
  descriptor.input_cols = 3;
  descriptor.storage_dtype = "fp32";
  descriptor.compute_dtype = "fp32";
  descriptor.layout_tag = "row_major";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data = reinterpret_cast<const std::uint8_t*>(source.data());
  descriptor.packed_nbytes = source.size() * sizeof(float);

  auto uploaded = DeviceDenseWeightFp32::Upload(descriptor);
  if (!expect(static_cast<bool>(uploaded), "source dense upload should succeed")) {
    return false;
  }

  auto view = DeviceDenseWeightFp32::CreateView(2, 3, const_cast<float*>(uploaded->data()));
  if (!expect(static_cast<bool>(view) && view->valid(), "dense view should be valid")) {
    return false;
  }

  std::vector<float> round_trip(view->numel(), 0.0f);
  if (!expect(
          cudaMemcpy(
              round_trip.data(),
              view->data(),
              round_trip.size() * sizeof(float),
              cudaMemcpyDeviceToHost) == cudaSuccess,
          "dense view should copy back to host")) {
    return false;
  }

  return expect(nearly_equal(round_trip, source, 1.0e-6f),
                "dense view should alias the original device storage");
}

}  // namespace

int main() {
  const bool ok =
      test_dense_weight_upload_round_trips_descriptor_bytes() &&
      test_dense_weight_upload_rejects_non_fp32_descriptor() &&
      test_dense_weight_upload_accepts_bf16_descriptor() &&
      test_dense_weight_view_aliases_existing_device_storage();

  if (!ok) {
    return 1;
  }
  std::cout << "dense_weight_test: PASS\n";
  return 0;
}
