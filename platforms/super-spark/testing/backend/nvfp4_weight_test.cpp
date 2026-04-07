#include "nemotron/artifact_loader.h"
#include "nemotron/gemm_catalog.h"
#include "nemotron/kernel_catalog.h"
#include "nemotron/manifest.h"
#include "nemotron/nvfp4_scale_layout.h"
#include "nemotron/nvfp4_weight.h"
#include "nemotron/tensor_catalog.h"
#include "nemotron/weight_arena.h"
#include "nemotron/weight_arena_plan.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <algorithm>
#include <vector>

namespace {

using nemotron::ArtifactLoader;
using nemotron::BuildGemmCatalog;
using nemotron::BuildKernelCatalog;
using nemotron::BuildTensorCatalog;
using nemotron::BuildWeightArenaPlan;
using nemotron::DeviceNvfp4Weight;
using nemotron::GemmCatalog;
using nemotron::KernelCatalog;
using nemotron::LoadVerifiedManifestFromJsonFile;
using nemotron::ManifestLoadResult;
using nemotron::SwizzleRowMajorNvfp4ScalesForExecution;
using nemotron::TensorCatalog;
using nemotron::WeightArena;

class TempDir {
 public:
  TempDir() {
    path_ = std::filesystem::temp_directory_path() /
            ("nemotron_nvfp4_weight_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
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

struct LoadedScaledCatalog {
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
    const std::string& packed_file,
    const std::string& packed_checksum,
    const std::string& auxiliary_file,
    std::size_t packed_nbytes,
    std::size_t block_scales_nbytes,
    std::size_t tensor_scale_nbytes) {
  std::ostringstream oss;
  oss
      << "{\n"
      << "  \"schema_version\": 1,\n"
      << "  \"model_id\": \"nvfp4-weight-test\",\n"
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
      << "      \"name\": \"layers.12.experts.117.w1\",\n"
      << "      \"op_class\": \"routed_expert\",\n"
      << "      \"logical_shape\": [32, 16],\n"
      << "      \"packed_shape\": [32, 16],\n"
      << "      \"storage_dtype\": \"nvfp4_e2m1\",\n"
      << "      \"compute_dtype\": \"fp32_accum\",\n"
      << "      \"layout_tag\": \"cublaslt_fp4_tn_v1\",\n"
      << "      \"alignment_bytes\": 16,\n"
      << "      \"block_scale_mode\": \"vec16_e4m3\",\n"
      << "      \"block_scale_dtype\": \"e4m3\",\n"
      << "      \"tensor_scale_dtype\": \"fp32\",\n"
      << "      \"packed_file\": \"" << packed_file << "\",\n"
      << "      \"offset_bytes\": 0,\n"
      << "      \"nbytes\": " << packed_nbytes << ",\n"
      << "      \"auxiliaries\": [\n"
      << "        {\n"
      << "          \"name\": \"block_scales\",\n"
      << "          \"file\": \"" << auxiliary_file << "\",\n"
      << "          \"offset_bytes\": 0,\n"
      << "          \"nbytes\": " << block_scales_nbytes << "\n"
      << "        },\n"
      << "        {\n"
      << "          \"name\": \"tensor_scale\",\n"
      << "          \"file\": \"" << auxiliary_file << "\",\n"
      << "          \"offset_bytes\": " << block_scales_nbytes << ",\n"
      << "          \"nbytes\": " << tensor_scale_nbytes << "\n"
      << "        }\n"
      << "      ],\n"
      << "      \"source_tensor_name\": \"layers.12.experts.117.w1\",\n"
      << "      \"checksum\": \"fnv1a64:" << packed_checksum << "\"\n"
      << "    }\n"
      << "  ]\n"
      << "}\n";
  return oss.str();
}

LoadedScaledCatalog load_scaled_catalog(const std::filesystem::path& manifest_path) {
  LoadedScaledCatalog loaded;
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

bool test_nvfp4_weight_upload_round_trips_packed_and_scale_bytes() {
  if (!has_cuda_device()) {
    std::cout << "nvfp4_weight_test: SKIP (no CUDA device available)\n";
    return true;
  }

  TempDir temp_dir;
  const std::vector<std::uint8_t> packed = {
      0x11, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
      0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
  };
  const std::vector<std::uint8_t> block_scales = {
      0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f,
      0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
      0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f,
      0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57,
  };
  const std::vector<float> tensor_scale = {1.25f};
  const std::string packed_bytes = bytes_from_vector(packed);
  const std::string block_scale_bytes = bytes_from_vector(block_scales);
  const std::string tensor_scale_bytes = bytes_from_vector(tensor_scale);
  write_file(temp_dir.path() / "weights" / "experts.bin", packed_bytes);
  write_file(
      temp_dir.path() / "weights" / "experts_scales.bin",
      block_scale_bytes + tensor_scale_bytes);
  write_file(
      temp_dir.path() / "manifest.json",
      make_manifest_json(
          "weights/experts.bin",
          fnv1a64_hex(packed_bytes),
          "weights/experts_scales.bin",
          packed_bytes.size(),
          block_scale_bytes.size(),
          tensor_scale_bytes.size()));

  const LoadedScaledCatalog loaded = load_scaled_catalog(temp_dir.path() / "manifest.json");
  if (!expect(loaded.gemm_catalog.valid(), "scaled GEMM catalog should load before upload")) {
    return false;
  }

  const auto* scaled = loaded.gemm_catalog.FindDescriptor("layers.12.experts.117.w1");
  if (!expect(scaled != nullptr, "scaled GEMM descriptor should exist before upload")) {
    return false;
  }

  auto uploaded = DeviceNvfp4Weight::Upload(*scaled);
  if (!expect(static_cast<bool>(uploaded), "NVFP4 weight upload should succeed")) {
    return false;
  }
  if (!expect(uploaded->valid(), "uploaded NVFP4 weight should be valid")) {
    return false;
  }
  if (!expect(uploaded->output_rows() == 32 && uploaded->input_cols() == 16,
              "uploaded NVFP4 weight should preserve matrix shape")) {
    return false;
  }

  std::vector<std::uint8_t> packed_round_trip(uploaded->packed_nbytes(), 0);
  std::vector<std::uint8_t> block_scales_round_trip(uploaded->block_scales_nbytes(), 0);
  std::vector<std::uint8_t> matmul_block_scales_round_trip(uploaded->matmul_block_scales_nbytes(), 0);
  std::vector<std::uint8_t> tensor_scale_round_trip(uploaded->tensor_scale_nbytes(), 0);
  if (!expect(
          cudaMemcpy(
              packed_round_trip.data(),
              uploaded->packed_data(),
              uploaded->packed_nbytes(),
              cudaMemcpyDeviceToHost) == cudaSuccess,
          "uploaded NVFP4 packed buffer should copy back to host")) {
    return false;
  }
  if (!expect(
          cudaMemcpy(
              block_scales_round_trip.data(),
              uploaded->block_scales_data(),
              uploaded->block_scales_nbytes(),
              cudaMemcpyDeviceToHost) == cudaSuccess,
          "uploaded NVFP4 block scales should copy back to host")) {
    return false;
  }
  if (!expect(
          cudaMemcpy(
              tensor_scale_round_trip.data(),
              uploaded->tensor_scale_data(),
              uploaded->tensor_scale_nbytes(),
              cudaMemcpyDeviceToHost) == cudaSuccess,
          "uploaded NVFP4 tensor scale should copy back to host")) {
    return false;
  }
  if (!expect(
          cudaMemcpy(
              matmul_block_scales_round_trip.data(),
              uploaded->matmul_block_scales_data(),
              uploaded->matmul_block_scales_nbytes(),
              cudaMemcpyDeviceToHost) == cudaSuccess,
          "uploaded NVFP4 execution block scales should copy back to host")) {
    return false;
  }

  const std::vector<std::uint8_t> expected_matmul_block_scales =
      SwizzleRowMajorNvfp4ScalesForExecution(block_scales.data(), 32, 16);
  if (matmul_block_scales_round_trip != expected_matmul_block_scales) {
    std::cerr << "expected execution scales bytes=" << expected_matmul_block_scales.size()
              << " actual=" << matmul_block_scales_round_trip.size() << "\n";
    const std::size_t limit =
        std::min<std::size_t>(matmul_block_scales_round_trip.size(), expected_matmul_block_scales.size());
    for (std::size_t i = 0; i < limit; ++i) {
      if (matmul_block_scales_round_trip[i] != expected_matmul_block_scales[i]) {
        std::cerr << "first execution scale mismatch at " << i
                  << " actual=0x" << std::hex << static_cast<int>(matmul_block_scales_round_trip[i])
                  << " expected=0x" << static_cast<int>(expected_matmul_block_scales[i]) << std::dec << "\n";
        break;
      }
    }
  }

  return expect(
             packed_round_trip == packed,
             "uploaded NVFP4 packed bytes should match the source descriptor bytes") &&
         expect(
             block_scales_round_trip == block_scales,
             "uploaded NVFP4 block scales should match the source descriptor bytes") &&
         expect(
             matmul_block_scales_round_trip == expected_matmul_block_scales,
             "uploaded NVFP4 execution block scales should match the swizzled source descriptor bytes") &&
         expect(
             tensor_scale_round_trip ==
                 std::vector<std::uint8_t>(
                     reinterpret_cast<const std::uint8_t*>(tensor_scale_bytes.data()),
                     reinterpret_cast<const std::uint8_t*>(tensor_scale_bytes.data()) +
                         tensor_scale_bytes.size()),
             "uploaded NVFP4 tensor scale should match the source descriptor bytes");
}

bool test_nvfp4_weight_upload_rejects_descriptor_without_scale_bytes() {
  if (!has_cuda_device()) {
    std::cout << "nvfp4_weight_test: SKIP (no CUDA device available)\n";
    return true;
  }

  TempDir temp_dir;
  const std::vector<std::uint8_t> packed = {
      0x11, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
      0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
  };
  const std::vector<std::uint8_t> block_scales = {
      0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f,
      0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
      0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f,
      0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57,
  };
  const std::vector<float> tensor_scale = {1.25f};
  const std::string packed_bytes = bytes_from_vector(packed);
  const std::string block_scale_bytes = bytes_from_vector(block_scales);
  const std::string tensor_scale_bytes = bytes_from_vector(tensor_scale);
  write_file(temp_dir.path() / "weights" / "experts.bin", packed_bytes);
  write_file(
      temp_dir.path() / "weights" / "experts_scales.bin",
      block_scale_bytes + tensor_scale_bytes);
  write_file(
      temp_dir.path() / "manifest.json",
      make_manifest_json(
          "weights/experts.bin",
          fnv1a64_hex(packed_bytes),
          "weights/experts_scales.bin",
          packed_bytes.size(),
          block_scale_bytes.size(),
          tensor_scale_bytes.size()));

  const LoadedScaledCatalog loaded = load_scaled_catalog(temp_dir.path() / "manifest.json");
  if (!expect(loaded.gemm_catalog.valid(), "scaled GEMM catalog should load before rejection test")) {
    return false;
  }

  const auto* scaled = loaded.gemm_catalog.FindDescriptor("layers.12.experts.117.w1");
  if (!expect(scaled != nullptr, "scaled GEMM descriptor should exist before rejection test")) {
    return false;
  }

  nemotron::GemmDescriptor unsupported = *scaled;
  unsupported.block_scales_data = nullptr;
  unsupported.block_scales_nbytes = 0;
  return expect(!DeviceNvfp4Weight::Upload(unsupported),
                "device-resident NVFP4 weight upload should reject descriptors without scale bytes");
}

bool test_nvfp4_weight_view_allows_execution_scales_without_raw_block_scales() {
  if (!has_cuda_device()) {
    std::cout << "nvfp4_weight_test: SKIP (no CUDA device available)\n";
    return true;
  }

  std::uint8_t* packed_data = nullptr;
  std::uint8_t* matmul_block_scales = nullptr;
  std::uint8_t* tensor_scale = nullptr;
  constexpr std::size_t kPackedBytes = 16;
  constexpr std::size_t kMatmulScaleBytes = 32;
  constexpr std::size_t kTensorScaleBytes = sizeof(float);
  const bool allocated =
      cudaMalloc(reinterpret_cast<void**>(&packed_data), kPackedBytes) == cudaSuccess &&
      cudaMalloc(reinterpret_cast<void**>(&matmul_block_scales), kMatmulScaleBytes) == cudaSuccess &&
      cudaMalloc(reinterpret_cast<void**>(&tensor_scale), kTensorScaleBytes) == cudaSuccess;
  if (!allocated) {
    if (tensor_scale != nullptr) {
      cudaFree(tensor_scale);
    }
    if (matmul_block_scales != nullptr) {
      cudaFree(matmul_block_scales);
    }
    if (packed_data != nullptr) {
      cudaFree(packed_data);
    }
    return expect(false, "device allocations for NVFP4 view test should succeed");
  }

  auto view = DeviceNvfp4Weight::CreateView(
      32,
      16,
      packed_data,
      kPackedBytes,
      nullptr,
      0,
      matmul_block_scales,
      kMatmulScaleBytes,
      tensor_scale,
      kTensorScaleBytes);

  const bool ok =
      expect(static_cast<bool>(view), "NVFP4 view creation should allow missing raw block scales") &&
      expect(view->valid(), "NVFP4 view without raw block scales should still be valid") &&
      expect(view->block_scales_data() == nullptr, "NVFP4 view should expose null raw block scales") &&
      expect(view->block_scales_nbytes() == 0, "NVFP4 view should expose zero raw block scale bytes");

  cudaFree(tensor_scale);
  cudaFree(matmul_block_scales);
  cudaFree(packed_data);
  return ok;
}

}  // namespace

int main() {
  const bool ok =
      test_nvfp4_weight_upload_round_trips_packed_and_scale_bytes() &&
      test_nvfp4_weight_upload_rejects_descriptor_without_scale_bytes() &&
      test_nvfp4_weight_view_allows_execution_scales_without_raw_block_scales();

  if (!ok) {
    return 1;
  }
  std::cout << "nvfp4_weight_test: PASS\n";
  return 0;
}
