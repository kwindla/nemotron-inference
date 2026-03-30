#include "nemotron/artifact_loader.h"
#include "nemotron/cublaslt_gemm_plan.h"
#include "nemotron/cublaslt_handle.h"
#include "nemotron/device_tensor.h"
#include "nemotron/gemm_catalog.h"
#include "nemotron/gemm_execution.h"
#include "nemotron/gemm_planner.h"
#include "nemotron/kernel_catalog.h"
#include "nemotron/manifest.h"
#include "nemotron/nvfp4_gemm_runner.h"
#include "nemotron/nvfp4_weight.h"
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
using nemotron::DeviceNvfp4Weight;
using nemotron::DeviceTensorFp32;
using nemotron::GemmCatalog;
using nemotron::GemmDescriptor;
using nemotron::GemmHeuristicCache;
using nemotron::KernelCatalog;
using nemotron::LoadVerifiedManifestFromJsonFile;
using nemotron::MakeNvfp4PackedMatrixDeviceView;
using nemotron::ManifestLoadResult;
using nemotron::Nvfp4PackedMatrixDeviceView;
using nemotron::PrepareGemmExecution;
using nemotron::RunNvfp4RowMajorFp32AccumToDevice;
using nemotron::RunNvfp4RowMajorFp32SourceToDevice;
using nemotron::TensorCatalog;
using nemotron::WeightArena;

class TempDir {
 public:
  TempDir() {
    path_ = std::filesystem::temp_directory_path() /
            ("nemotron_nvfp4_gemm_runner_test_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
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
  output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
}

std::string float_bytes(float value) {
  return std::string(reinterpret_cast<const char*>(&value), sizeof(value));
}

std::string make_nvfp4_manifest_json(
    const std::string& packed_file,
    const std::string& block_scale_file,
    const std::string& tensor_scale_file,
    const std::string& packed_checksum) {
  std::ostringstream oss;
  oss
      << "{\n"
      << "  \"schema_version\": 1,\n"
      << "  \"model_id\": \"nvfp4-gemm-test\",\n"
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
      << "      \"logical_shape\": [64, 64],\n"
      << "      \"packed_shape\": [64, 64],\n"
      << "      \"storage_dtype\": \"nvfp4_e2m1\",\n"
      << "      \"compute_dtype\": \"fp32_accum\",\n"
      << "      \"layout_tag\": \"cublaslt_fp4_tn_v1\",\n"
      << "      \"alignment_bytes\": 16,\n"
      << "      \"block_scale_mode\": \"vec16_e4m3\",\n"
      << "      \"block_scale_dtype\": \"e4m3\",\n"
      << "      \"tensor_scale_dtype\": \"fp32\",\n"
      << "      \"packed_file\": \"" << packed_file << "\",\n"
      << "      \"offset_bytes\": 0,\n"
      << "      \"nbytes\": 2048,\n"
      << "      \"source_tensor_name\": \"mlp.up_proj\",\n"
      << "      \"auxiliaries\": [\n"
      << "        {\n"
      << "          \"name\": \"block_scales\",\n"
      << "          \"file\": \"" << block_scale_file << "\",\n"
      << "          \"offset_bytes\": 0,\n"
      << "          \"nbytes\": 256\n"
      << "        },\n"
      << "        {\n"
      << "          \"name\": \"tensor_scale\",\n"
      << "          \"file\": \"" << tensor_scale_file << "\",\n"
      << "          \"offset_bytes\": 0,\n"
      << "          \"nbytes\": 4\n"
      << "        }\n"
      << "      ],\n"
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

GemmDescriptor make_activation_descriptor(
    const std::string& packed_bytes,
    const std::string& block_scale_bytes,
    const std::string& tensor_scale_bytes) {
  GemmDescriptor descriptor;
  descriptor.tensor_name = "activation";
  descriptor.op_class = "activation";
  descriptor.kernel_family = nemotron::GemmKernelFamily::kCublasLtNvfp4BlockScaled;
  descriptor.output_rows = 64;
  descriptor.input_cols = 64;
  descriptor.storage_dtype = "nvfp4_e2m1";
  descriptor.compute_dtype = "fp32_accum";
  descriptor.layout_tag = "cublaslt_fp4_tn_v1";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data = reinterpret_cast<const std::uint8_t*>(packed_bytes.data());
  descriptor.packed_nbytes = packed_bytes.size();
  descriptor.block_scales_data = reinterpret_cast<const std::uint8_t*>(block_scale_bytes.data());
  descriptor.block_scales_nbytes = block_scale_bytes.size();
  descriptor.tensor_scale_data = reinterpret_cast<const std::uint8_t*>(tensor_scale_bytes.data());
  descriptor.tensor_scale_nbytes = tensor_scale_bytes.size();
  return descriptor;
}

bool all_zero(const std::vector<float>& values) {
  for (float value : values) {
    if (value != 0.0f) {
      return false;
    }
  }
  return true;
}

bool test_nvfp4_gemm_runner_executes_row_major_contract() {
  const auto handle = CublasLtHandle::Create();
  if (!handle || !handle->valid()) {
    std::cout << "nvfp4_gemm_runner_test: SKIP (no CUDA device or cublasLt unavailable)\n";
    return true;
  }

  TempDir temp_dir;
  const std::filesystem::path packed_path = temp_dir.path() / "weights" / "packed.bin";
  const std::filesystem::path block_scale_path = temp_dir.path() / "weights" / "block_scales.bin";
  const std::filesystem::path tensor_scale_path = temp_dir.path() / "weights" / "tensor_scale.bin";

  const std::string packed_bytes(2048, '\0');
  const std::string block_scale_bytes(256, static_cast<char>(0x38));
  const std::string tensor_scale_bytes = float_bytes(1.0f);

  write_file(packed_path, packed_bytes);
  write_file(block_scale_path, block_scale_bytes);
  write_file(tensor_scale_path, tensor_scale_bytes);

  const std::filesystem::path manifest_path = temp_dir.path() / "manifest.json";
  write_file(
      manifest_path,
      make_nvfp4_manifest_json(
          "weights/packed.bin",
          "weights/block_scales.bin",
          "weights/tensor_scale.bin",
          fnv1a64_hex(packed_bytes)));

  const LoadedScaledCatalog loaded = load_scaled_catalog(manifest_path);
  if (!expect(loaded.gemm_catalog.valid(), "scaled GEMM catalog should be valid before NVFP4 execution")) {
    return false;
  }

  const auto* weight_descriptor = loaded.gemm_catalog.FindDescriptor("mlp.up_proj");
  if (!expect(weight_descriptor != nullptr, "scaled GEMM descriptor should exist")) {
    return false;
  }

  auto weight = DeviceNvfp4Weight::Upload(*weight_descriptor);
  if (!expect(static_cast<bool>(weight) && weight->valid(), "NVFP4 weight upload should succeed")) {
    return false;
  }

  const GemmDescriptor activation_descriptor =
      make_activation_descriptor(packed_bytes, block_scale_bytes, tensor_scale_bytes);
  auto activation = DeviceNvfp4Weight::Upload(activation_descriptor);
  if (!expect(static_cast<bool>(activation) && activation->valid(), "NVFP4 activation upload should succeed")) {
    return false;
  }

  const auto launch_plan = BuildGemmLaunchPlan(*weight_descriptor, 64);
  if (!expect(launch_plan.has_value(), "NVFP4 launch plan should build")) {
    return false;
  }

  GemmHeuristicCache cache;
  const auto execution = PrepareGemmExecution(*launch_plan, &cache);
  if (!expect(execution.has_value(), "NVFP4 execution should prepare")) {
    return false;
  }

  const auto plan = BuildCublasLtGemmPlan(*execution);
  if (!expect(plan.has_value(), "NVFP4 cublasLt plan should build")) {
    return false;
  }

  auto output = DeviceTensorFp32::Create({64, 64});
  if (!expect(output && output->valid(), "NVFP4 output tensor should allocate")) {
    return false;
  }

  const Nvfp4PackedMatrixDeviceView activation_view = MakeNvfp4PackedMatrixDeviceView(*activation);
  const auto stats =
      RunNvfp4RowMajorFp32AccumToDevice(*handle, *plan, activation_view, *weight, output.get());
  if (!expect(stats.has_value(), "NVFP4 GEMM runner should execute successfully")) {
    return false;
  }

  std::vector<float> host_output(output->numel(), 1.0f);
  if (!expect(output->CopyToHost(host_output.data(), host_output.size()),
              "NVFP4 output should download successfully")) {
    return false;
  }

  return expect(stats->rows == 64 && stats->cols == 64, "NVFP4 GEMM runner should report the correct output shape") &&
         expect(stats->heuristic_count > 0, "NVFP4 GEMM runner should find at least one cuBLASLt heuristic") &&
         expect(all_zero(host_output), "zero-valued NVFP4 inputs should produce zero-valued FP32 output");
}

bool test_nvfp4_gemm_runner_rejects_dense_plan() {
  const auto handle = CublasLtHandle::Create();
  if (!handle || !handle->valid()) {
    std::cout << "nvfp4_gemm_runner_test: SKIP (no CUDA device or cublasLt unavailable)\n";
    return true;
  }

  nemotron::CublasLtGemmPlan plan;
  auto output = DeviceTensorFp32::Create({64, 64});
  GemmDescriptor activation_descriptor =
      make_activation_descriptor(std::string(2048, '\0'), std::string(256, static_cast<char>(0x38)), float_bytes(1.0f));
  auto activation = DeviceNvfp4Weight::Upload(activation_descriptor);
  auto weight = DeviceNvfp4Weight::Upload(activation_descriptor);
  if (!expect(output && activation && weight, "rejection test should allocate its inputs")) {
    return false;
  }

  return expect(
      !RunNvfp4RowMajorFp32AccumToDevice(
           *handle,
           plan,
           MakeNvfp4PackedMatrixDeviceView(*activation),
           *weight,
           output.get())
           .has_value(),
      "NVFP4 GEMM runner should reject plans that are not NVFP4 block-scaled plans");
}

bool test_nvfp4_gemm_runner_executes_from_device_fp32_source() {
  const auto handle = CublasLtHandle::Create();
  if (!handle || !handle->valid()) {
    std::cout << "nvfp4_gemm_runner_test: SKIP (no CUDA device or cublasLt unavailable)\n";
    return true;
  }

  TempDir temp_dir;
  const std::filesystem::path packed_path = temp_dir.path() / "weights" / "packed.bin";
  const std::filesystem::path block_scale_path = temp_dir.path() / "weights" / "block_scales.bin";
  const std::filesystem::path tensor_scale_path = temp_dir.path() / "weights" / "tensor_scale.bin";

  const std::string packed_bytes(2048, '\0');
  const std::string block_scale_bytes(256, static_cast<char>(0x38));
  const std::string tensor_scale_bytes = float_bytes(1.0f);

  write_file(packed_path, packed_bytes);
  write_file(block_scale_path, block_scale_bytes);
  write_file(tensor_scale_path, tensor_scale_bytes);

  const std::filesystem::path manifest_path = temp_dir.path() / "manifest.json";
  write_file(
      manifest_path,
      make_nvfp4_manifest_json(
          "weights/packed.bin",
          "weights/block_scales.bin",
          "weights/tensor_scale.bin",
          fnv1a64_hex(packed_bytes)));

  const LoadedScaledCatalog loaded = load_scaled_catalog(manifest_path);
  if (!expect(loaded.gemm_catalog.valid(), "scaled GEMM catalog should be valid before FP32-source execution")) {
    return false;
  }

  const auto* weight_descriptor = loaded.gemm_catalog.FindDescriptor("mlp.up_proj");
  if (!expect(weight_descriptor != nullptr, "scaled GEMM descriptor should exist")) {
    return false;
  }

  auto weight = DeviceNvfp4Weight::Upload(*weight_descriptor);
  if (!expect(static_cast<bool>(weight) && weight->valid(), "NVFP4 weight upload should succeed")) {
    return false;
  }

  const auto launch_plan = BuildGemmLaunchPlan(*weight_descriptor, 64);
  if (!expect(launch_plan.has_value(), "NVFP4 launch plan should build")) {
    return false;
  }

  GemmHeuristicCache cache;
  const auto execution = PrepareGemmExecution(*launch_plan, &cache);
  if (!expect(execution.has_value(), "NVFP4 execution should prepare")) {
    return false;
  }

  const auto plan = BuildCublasLtGemmPlan(*execution);
  if (!expect(plan.has_value(), "NVFP4 cublasLt plan should build")) {
    return false;
  }

  auto activation_source = DeviceTensorFp32::Create({64, 64});
  auto output = DeviceTensorFp32::Create({64, 64});
  if (!expect(activation_source && activation_source->valid() && output && output->valid(),
              "FP32-source test should allocate its tensors")) {
    return false;
  }
  if (!expect(activation_source->FillZero(), "FP32 activation source should zero-fill")) {
    return false;
  }

  const auto stats =
      RunNvfp4RowMajorFp32SourceToDevice(*handle, *plan, *activation_source, *weight, output.get());
  if (!expect(stats.has_value(), "FP32-source NVFP4 GEMM path should execute successfully")) {
    return false;
  }

  std::vector<float> host_output(output->numel(), 1.0f);
  if (!expect(output->CopyToHost(host_output.data(), host_output.size()),
              "FP32-source output should download successfully")) {
    return false;
  }

  return expect(stats->rows == 64 && stats->cols == 64,
                "FP32-source NVFP4 GEMM runner should report the correct output shape") &&
         expect(stats->heuristic_count > 0,
                "FP32-source NVFP4 GEMM runner should find at least one cuBLASLt heuristic") &&
         expect(all_zero(host_output),
                "zero-valued FP32 activations and zero-valued NVFP4 weights should produce zero output");
}

}  // namespace

int main() {
  const bool ok =
      test_nvfp4_gemm_runner_executes_row_major_contract() &&
      test_nvfp4_gemm_runner_rejects_dense_plan() &&
      test_nvfp4_gemm_runner_executes_from_device_fp32_source();

  if (!ok) {
    return 1;
  }
  std::cout << "nvfp4_gemm_runner_test: PASS\n";
  return 0;
}
