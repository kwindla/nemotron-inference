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
#include "nemotron/linear_reference_kernels.h"
#include "nemotron/nvfp4_packing.h"
#include "nemotron/nvfp4_weight.h"
#include "nemotron/tensor_catalog.h"
#include "nemotron/weight_arena.h"
#include "nemotron/weight_arena_plan.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
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
using nemotron::Nvfp4PackOptions;
using nemotron::Nvfp4PackedMatrixDeviceView;
using nemotron::PackRowMajorFp32ToNvfp4;
using nemotron::PrepareGemmExecution;
using nemotron::RunNvfp4RowMajorFp32AccumToDevice;
using nemotron::RunNvfp4RowMajorReferenceToDevice;
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

GemmDescriptor make_nvfp4_descriptor(
    std::size_t output_rows,
    std::size_t input_cols,
    const std::uint8_t* packed_data,
    std::size_t packed_nbytes,
    const std::uint8_t* block_scales_data,
    std::size_t block_scales_nbytes,
    const std::uint8_t* tensor_scale_data,
    std::size_t tensor_scale_nbytes) {
  GemmDescriptor descriptor;
  descriptor.tensor_name = "nvfp4_test_tensor";
  descriptor.op_class = "nvfp4_test_tensor";
  descriptor.kernel_family = nemotron::GemmKernelFamily::kCublasLtNvfp4BlockScaled;
  descriptor.output_rows = output_rows;
  descriptor.input_cols = input_cols;
  descriptor.storage_dtype = "nvfp4_e2m1";
  descriptor.compute_dtype = "fp32_accum";
  descriptor.layout_tag = "cublaslt_fp4_tn_v1";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data = packed_data;
  descriptor.packed_nbytes = packed_nbytes;
  descriptor.block_scales_data = block_scales_data;
  descriptor.block_scales_nbytes = block_scales_nbytes;
  descriptor.tensor_scale_data = tensor_scale_data;
  descriptor.tensor_scale_nbytes = tensor_scale_nbytes;
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

float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return std::numeric_limits<float>::infinity();
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
  }
  return max_diff;
}

std::size_t count_nonfinite(const std::vector<float>& values) {
  std::size_t count = 0;
  for (float value : values) {
    if (!std::isfinite(value)) {
      ++count;
    }
  }
  return count;
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

bool test_nvfp4_gemm_runner_matches_reference_for_nonzero_m1() {
  const auto handle = CublasLtHandle::Create();
  if (!handle || !handle->valid()) {
    std::cout << "nvfp4_gemm_runner_test: SKIP (no CUDA device or cublasLt unavailable)\n";
    return true;
  }

  constexpr std::size_t kRows = 1;
  constexpr std::size_t kInputCols = 5376;
  constexpr std::size_t kOutputRows = 4096;

  std::vector<float> weights(kOutputRows * kInputCols, 0.0f);
  for (std::size_t row = 0; row < kOutputRows; ++row) {
    for (std::size_t col = 0; col < kInputCols; ++col) {
      const int pattern = static_cast<int>((row * 17 + col * 13) % 29) - 14;
      weights[row * kInputCols + col] = static_cast<float>(pattern) * 0.0078125f;
    }
  }
  const auto packed_weights = PackRowMajorFp32ToNvfp4(weights.data(), kOutputRows, kInputCols);
  if (!expect(packed_weights.has_value() && packed_weights->valid(),
              "nonzero NVFP4 weight pack should succeed")) {
    return false;
  }

  const auto descriptor = make_nvfp4_descriptor(
      kOutputRows,
      kInputCols,
      packed_weights->packed_data(),
      packed_weights->packed_nbytes(),
      packed_weights->block_scales_data(),
      packed_weights->block_scales_nbytes(),
      packed_weights->tensor_scale_data(),
      packed_weights->tensor_scale_nbytes());
  auto weight = DeviceNvfp4Weight::Upload(descriptor);
  if (!expect(static_cast<bool>(weight) && weight->valid(), "nonzero NVFP4 weight upload should succeed")) {
    return false;
  }

  const auto launch_plan = BuildGemmLaunchPlan(descriptor, kRows);
  if (!expect(launch_plan.has_value(), "nonzero NVFP4 launch plan should build")) {
    return false;
  }

  GemmHeuristicCache cache;
  const auto execution = PrepareGemmExecution(*launch_plan, &cache);
  if (!expect(execution.has_value(), "nonzero NVFP4 execution should prepare")) {
    return false;
  }

  const auto plan = BuildCublasLtGemmPlan(*execution);
  if (!expect(plan.has_value(), "nonzero NVFP4 cublasLt plan should build")) {
    return false;
  }

  auto activation_source = DeviceTensorFp32::Create({kRows, kInputCols});
  auto fastpath_output = DeviceTensorFp32::Create({kRows, kOutputRows});
  auto reference_output = DeviceTensorFp32::Create({kRows, kOutputRows});
  if (!expect(
          activation_source && activation_source->valid() &&
              fastpath_output && fastpath_output->valid() &&
              reference_output && reference_output->valid(),
          "nonzero FP32-source test should allocate its tensors")) {
    return false;
  }

  std::vector<float> activations(kRows * kInputCols, 0.0f);
  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t col = 0; col < kInputCols; ++col) {
      const int pattern = static_cast<int>((row * 11 + col * 7) % 23) - 11;
      activations[row * kInputCols + col] =
          static_cast<float>(pattern) * 0.015625f;
    }
  }
  if (!expect(
          activation_source->CopyFromHost(activations.data(), activations.size()),
          "nonzero FP32 activation source should upload")) {
    return false;
  }

  Nvfp4PackOptions pack_options;
  pack_options.execution_scale_layout = nemotron::Nvfp4ScaleLayout::kSwizzled128x4;
  const auto fastpath_stats = RunNvfp4RowMajorFp32SourceToDevice(
      *handle,
      *plan,
      *activation_source,
      *weight,
      fastpath_output.get(),
      pack_options);
  if (!expect(fastpath_stats.has_value(), "nonzero FP32-source NVFP4 path should execute successfully")) {
    return false;
  }

  if (!expect(
          RunNvfp4RowMajorReferenceToDevice(
              *activation_source,
              *weight,
              reference_output.get(),
              pack_options),
          "nonzero FP32-source NVFP4 reference path should execute successfully")) {
    return false;
  }

  std::vector<float> fastpath_host(fastpath_output->numel(), 0.0f);
  std::vector<float> reference_host(reference_output->numel(), 0.0f);
  if (!expect(
          fastpath_output->CopyToHost(fastpath_host.data(), fastpath_host.size()),
          "nonzero fastpath output should download successfully") ||
      !expect(
          reference_output->CopyToHost(reference_host.data(), reference_host.size()),
          "nonzero reference output should download successfully")) {
    return false;
  }

  const std::size_t nonfinite_count = count_nonfinite(fastpath_host);
  if (!expect(
          nonfinite_count == 0,
          "nonzero FP32-source NVFP4 fastpath output should stay finite")) {
    std::cerr << "nonzero_m1_diagnostic: nonfinite_count=" << nonfinite_count << "\n";
    return false;
  }

  const float diff = max_abs_diff(fastpath_host, reference_host);
  if (!expect(
          diff <= 5.0e-2f,
          "nonzero FP32-source NVFP4 fastpath output should stay close to the reference path")) {
    std::cerr << "nonzero_m1_diagnostic: max_abs_diff=" << diff
              << " fastpath0=" << fastpath_host.front()
              << " reference0=" << reference_host.front()
              << "\n";
    return false;
  }

  return expect(
      fastpath_stats->rows == kRows && fastpath_stats->cols == kOutputRows,
      "nonzero FP32-source NVFP4 stats should report the expected output shape");
}

}  // namespace

int main() {
  const bool ok =
      test_nvfp4_gemm_runner_executes_row_major_contract() &&
      test_nvfp4_gemm_runner_rejects_dense_plan() &&
      test_nvfp4_gemm_runner_executes_from_device_fp32_source() &&
      test_nvfp4_gemm_runner_matches_reference_for_nonzero_m1();

  if (!ok) {
    return 1;
  }
  std::cout << "nvfp4_gemm_runner_test: PASS\n";
  return 0;
}
