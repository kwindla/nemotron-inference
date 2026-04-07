#include "nemotron/cublaslt_gemm_plan.h"
#include "nemotron/cublaslt_handle.h"
#include "nemotron/device_nvfp4_matrix.h"
#include "nemotron/device_tensor.h"
#include "nemotron/gemm_execution.h"
#include "nemotron/gemm_planner.h"
#include "nemotron/nvfp4_gemm_runner.h"
#include "nemotron/nvfp4_packing.h"
#include "nemotron/nvfp4_scale_layout.h"
#include "nemotron/nvfp4_weight.h"

#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <regex>
#include <string>
#include <vector>

namespace {

using nemotron::BuildCublasLtGemmPlan;
using nemotron::BuildGemmLaunchPlan;
using nemotron::CublasLtHandle;
using nemotron::DeviceNvfp4Weight;
using nemotron::DeviceNvfp4Matrix;
using nemotron::DeviceTensorFp32;
using nemotron::GemmBackendKind;
using nemotron::GemmDescriptor;
using nemotron::GemmHeuristicCache;
using nemotron::GemmKernelFamily;
using nemotron::MakeNvfp4PackedMatrixDeviceView;
using nemotron::Nvfp4PackOptions;
using nemotron::PackRowMajorFp32ToNvfp4;
using nemotron::PrepareGemmExecution;
using nemotron::RunNvfp4RowMajorFp32AccumToDevice;
using nemotron::SwizzleRowMajorNvfp4ScalesForExecution;

constexpr float kMaxAbsDiffTolerance = 1.0e-3f;
constexpr double kMeanAbsDiffTolerance = 1.0e-4;

struct OracleFixtureMetadata {
  std::filesystem::path fixture_dir;
  std::string fixture_name;
  std::string target_operator;
  std::string activation_packing_mode = "fixed_checkpoint_input_scale";
  std::string weight_tensor_scale_contract;
  bool weight_input_scale_present = false;
  std::size_t rows = 0;
  std::size_t input_cols = 0;
  std::size_t output_rows = 0;
};

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

std::vector<std::uint8_t> read_file_bytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream input(path);
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::vector<float> read_float_file(const std::filesystem::path& path) {
  const std::vector<std::uint8_t> bytes = read_file_bytes(path);
  if (bytes.size() % sizeof(float) != 0) {
    return {};
  }
  std::vector<float> values(bytes.size() / sizeof(float), 0.0f);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

std::optional<std::string> parse_json_string_field(const std::string& json, const std::string& key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
  std::smatch match;
  if (!std::regex_search(json, match, pattern) || match.size() != 2) {
    return std::nullopt;
  }
  return match[1].str();
}

std::optional<std::size_t> parse_json_uint_field(const std::string& json, const std::string& key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9]+)");
  std::smatch match;
  if (!std::regex_search(json, match, pattern) || match.size() != 2) {
    return std::nullopt;
  }
  try {
    return static_cast<std::size_t>(std::stoull(match[1].str()));
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<bool> parse_json_bool_field(const std::string& json, const std::string& key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*(true|false)");
  std::smatch match;
  if (!std::regex_search(json, match, pattern) || match.size() != 2) {
    return std::nullopt;
  }
  return match[1].str() == "true";
}

bool fixture_files_exist(const std::filesystem::path& fixture_dir) {
  return std::filesystem::exists(fixture_dir / "metadata.json") &&
         std::filesystem::exists(fixture_dir / "activations_fp32.bin") &&
         std::filesystem::exists(fixture_dir / "activation_tensor_scale.bin") &&
         std::filesystem::exists(fixture_dir / "weight_packed.bin") &&
         std::filesystem::exists(fixture_dir / "weight_block_scales.bin") &&
         std::filesystem::exists(fixture_dir / "weight_tensor_scale.bin") &&
         std::filesystem::exists(fixture_dir / "expected_output_fp32.bin");
}

std::optional<OracleFixtureMetadata> load_fixture_metadata(const std::filesystem::path& fixture_dir) {
  if (!fixture_files_exist(fixture_dir)) {
    return std::nullopt;
  }

  const std::string metadata_json = read_text_file(fixture_dir / "metadata.json");
  const std::optional<std::string> fixture_kind = parse_json_string_field(metadata_json, "fixture_kind");
  const std::optional<std::string> target_operator = parse_json_string_field(metadata_json, "target_operator");
  const std::optional<std::string> activation_packing_mode =
      parse_json_string_field(metadata_json, "activation_packing_mode");
  const std::optional<std::string> weight_tensor_scale_contract =
      parse_json_string_field(metadata_json, "weight_tensor_scale_contract");
  const std::optional<bool> weight_input_scale_present =
      parse_json_bool_field(metadata_json, "weight_input_scale_present");
  const std::optional<std::size_t> rows = parse_json_uint_field(metadata_json, "rows");
  const std::optional<std::size_t> input_cols = parse_json_uint_field(metadata_json, "input_cols");
  const std::optional<std::size_t> output_rows = parse_json_uint_field(metadata_json, "output_rows");
  if (!fixture_kind || *fixture_kind != "nvfp4_operator_oracle_v1" ||
      !target_operator || !rows || !input_cols || !output_rows) {
    return std::nullopt;
  }

  OracleFixtureMetadata metadata;
  metadata.fixture_dir = fixture_dir;
  metadata.fixture_name = fixture_dir.filename().string();
  metadata.target_operator = *target_operator;
  if (activation_packing_mode.has_value()) {
    metadata.activation_packing_mode = *activation_packing_mode;
  }
  if (weight_tensor_scale_contract.has_value()) {
    metadata.weight_tensor_scale_contract = *weight_tensor_scale_contract;
  }
  metadata.weight_input_scale_present = weight_input_scale_present.value_or(false);
  metadata.rows = *rows;
  metadata.input_cols = *input_cols;
  metadata.output_rows = *output_rows;
  return metadata;
}

std::vector<OracleFixtureMetadata> discover_fixture_metadata() {
  const std::filesystem::path fixture_root = NEMOTRON_NVFP4_ORACLE_FIXTURE_ROOT;
  std::vector<OracleFixtureMetadata> fixtures;
  if (!std::filesystem::exists(fixture_root)) {
    return fixtures;
  }

  for (const auto& entry : std::filesystem::directory_iterator(fixture_root)) {
    if (!entry.is_directory()) {
      continue;
    }
    const auto metadata = load_fixture_metadata(entry.path());
    if (metadata.has_value()) {
      fixtures.push_back(*metadata);
    }
  }

  std::sort(
      fixtures.begin(),
      fixtures.end(),
      [](const OracleFixtureMetadata& lhs, const OracleFixtureMetadata& rhs) {
        return lhs.fixture_name < rhs.fixture_name;
      });
  return fixtures;
}

float decode_fp4(std::uint8_t raw_nibble) {
  __nv_fp4_e2m1 value;
  value.__x = raw_nibble & 0x0F;
  return static_cast<float>(value);
}

float decode_fp8(std::uint8_t raw_byte) {
  __nv_fp8_e4m3 value;
  value.__x = raw_byte;
  return static_cast<float>(value);
}

std::vector<float> dequantize_nvfp4_matrix(
    const std::uint8_t* packed,
    std::size_t packed_nbytes,
    const std::uint8_t* block_scales,
    std::size_t block_scales_nbytes,
    float tensor_scale,
    std::size_t rows,
    std::size_t cols) {
  if (packed == nullptr ||
      block_scales == nullptr ||
      cols == 0 ||
      cols % 16 != 0 ||
      packed_nbytes != (rows * cols) / 2 ||
      block_scales_nbytes != rows * (cols / 16)) {
    return {};
  }

  std::vector<float> output(rows * cols, 0.0f);
  std::size_t packed_index = 0;
  std::size_t scale_index = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t block = 0; block < cols / 16; ++block) {
      const float block_scale = decode_fp8(block_scales[scale_index++]) * tensor_scale;
      const std::size_t col_start = block * 16;
      for (std::size_t offset = 0; offset < 16; offset += 2) {
        const std::uint8_t byte = packed[packed_index++];
        output[row * cols + col_start + offset] = decode_fp4(byte & 0x0F) * block_scale;
        output[row * cols + col_start + offset + 1] = decode_fp4((byte >> 4) & 0x0F) * block_scale;
      }
    }
  }
  return output;
}

std::vector<float> cpu_reference(
    const std::vector<float>& activations,
    std::size_t rows,
    const std::vector<float>& weights,
    std::size_t output_rows,
    std::size_t input_cols) {
  std::vector<float> output(rows * output_rows, 0.0f);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t out = 0; out < output_rows; ++out) {
      float accum = 0.0f;
      for (std::size_t col = 0; col < input_cols; ++col) {
        accum += activations[row * input_cols + col] * weights[out * input_cols + col];
      }
      output[row * output_rows + out] = accum;
    }
  }
  return output;
}

bool run_fixture(
    CublasLtHandle& handle,
    GemmHeuristicCache* cache,
    const OracleFixtureMetadata& metadata) {
  const std::vector<float> activations = read_float_file(metadata.fixture_dir / "activations_fp32.bin");
  const std::vector<float> activation_tensor_scale =
      read_float_file(metadata.fixture_dir / "activation_tensor_scale.bin");
  const std::vector<std::uint8_t> weight_packed = read_file_bytes(metadata.fixture_dir / "weight_packed.bin");
  const std::vector<std::uint8_t> weight_block_scales =
      read_file_bytes(metadata.fixture_dir / "weight_block_scales.bin");
  const std::vector<float> weight_tensor_scale =
      read_float_file(metadata.fixture_dir / "weight_tensor_scale.bin");
  const std::vector<float> fixture_expected_output =
      read_float_file(metadata.fixture_dir / "expected_output_fp32.bin");

  if (!expect(!activations.empty(), metadata.fixture_name + ": oracle activations should load")) {
    return false;
  }
  if (!expect(
          activation_tensor_scale.size() == 1,
          metadata.fixture_name + ": activation tensor scale should be scalar")) {
    return false;
  }
  if (!expect(
          weight_tensor_scale.size() == 1,
          metadata.fixture_name + ": weight tensor scale should be scalar")) {
    return false;
  }
  if (!expect(
          !metadata.weight_tensor_scale_contract.empty(),
          metadata.fixture_name + ": weight tensor scale contract should be recorded")) {
    return false;
  }
  const std::string expected_weight_scale_contract = metadata.weight_input_scale_present
                                                         ? "effective_tensor_scale = input_scale * weight_scale_2"
                                                         : "raw_weight_scale_2";
  if (!expect(
          metadata.weight_tensor_scale_contract == expected_weight_scale_contract,
          metadata.fixture_name + ": weight tensor scale contract should match fixture input-scale availability")) {
    return false;
  }
  if (!expect(
          metadata.input_cols > 0 && activations.size() % metadata.input_cols == 0,
          metadata.fixture_name + ": activation row count should be integral")) {
    return false;
  }

  const std::size_t rows = activations.size() / metadata.input_cols;
  if (!expect(rows == metadata.rows, metadata.fixture_name + ": metadata row count should match activation data")) {
    return false;
  }
  if (!expect(rows > 0, metadata.fixture_name + ": fixture should provide at least one activation row")) {
    return false;
  }
  if (!expect(
          fixture_expected_output.size() == rows * metadata.output_rows,
          metadata.fixture_name + ": expected output size should match metadata shape")) {
    return false;
  }

  GemmDescriptor weight_descriptor;
  weight_descriptor.tensor_name = metadata.target_operator;
  weight_descriptor.op_class = "dense_linear";
  weight_descriptor.kernel_family = GemmKernelFamily::kCublasLtNvfp4BlockScaled;
  weight_descriptor.output_rows = metadata.output_rows;
  weight_descriptor.input_cols = metadata.input_cols;
  weight_descriptor.storage_dtype = "nvfp4_e2m1";
  weight_descriptor.compute_dtype = "fp32_accum";
  weight_descriptor.layout_tag = "cublaslt_fp4_tn_v1";
  weight_descriptor.alignment_bytes = 16;
  weight_descriptor.packed_data = weight_packed.data();
  weight_descriptor.packed_nbytes = weight_packed.size();
  weight_descriptor.block_scales_data = weight_block_scales.data();
  weight_descriptor.block_scales_nbytes = weight_block_scales.size();
  weight_descriptor.tensor_scale_data =
      reinterpret_cast<const std::uint8_t*>(weight_tensor_scale.data());
  weight_descriptor.tensor_scale_nbytes = sizeof(float);

  const auto launch_plan = BuildGemmLaunchPlan(weight_descriptor, rows);
  if (!expect(launch_plan.has_value(), metadata.fixture_name + ": launch plan should build")) {
    return false;
  }

  auto execution = PrepareGemmExecution(*launch_plan, cache);
  if (!expect(execution.has_value(), metadata.fixture_name + ": execution should prepare")) {
    return false;
  }
  if (!expect(
          execution->backend_kind == GemmBackendKind::kCublasLtNvfp4BlockScaled,
          metadata.fixture_name + ": execution should target NVFP4 cublasLt")) {
    return false;
  }

  const auto plan = BuildCublasLtGemmPlan(*execution);
  if (!expect(plan.has_value(), metadata.fixture_name + ": cublasLt plan should build")) {
    return false;
  }

  auto weight = DeviceNvfp4Weight::Upload(weight_descriptor);
  auto activation_matrix = DeviceNvfp4Matrix::Create(rows, metadata.input_cols);
  auto output = DeviceTensorFp32::Create({rows, metadata.output_rows});
  if (!expect(weight && weight->valid(), metadata.fixture_name + ": weight upload should succeed") ||
      !expect(
          activation_matrix && activation_matrix->valid(),
          metadata.fixture_name + ": activation matrix should allocate") ||
      !expect(output && output->valid(), metadata.fixture_name + ": output tensor should allocate")) {
    return false;
  }

  Nvfp4PackOptions pack_options;
  if (metadata.activation_packing_mode == "fixed_checkpoint_input_scale") {
    pack_options.fixed_tensor_scale = activation_tensor_scale.front();
  } else if (metadata.activation_packing_mode != "dynamic_runtime") {
    return expect(false, metadata.fixture_name + ": unsupported activation packing mode");
  }
  const auto activation_host_packed =
      PackRowMajorFp32ToNvfp4(activations.data(), rows, metadata.input_cols, pack_options);
  if (!expect(
          activation_host_packed.has_value(),
          metadata.fixture_name + ": host activation packing should succeed")) {
    return false;
  }
  if (!expect(
          std::fabs(activation_host_packed->tensor_scale - activation_tensor_scale.front()) <= 1.0e-6f,
          metadata.fixture_name + ": packed activation tensor scale should match fixture metadata")) {
    return false;
  }
  const std::vector<std::uint8_t> activation_matmul_scales = SwizzleRowMajorNvfp4ScalesForExecution(
      activation_host_packed->block_scales.data(),
      rows,
      metadata.input_cols);
  if (!expect(
          !activation_matmul_scales.empty(),
          metadata.fixture_name + ": activation matmul scales should swizzle")) {
    return false;
  }
  if (!expect(
          cudaMemcpy(
              const_cast<std::uint8_t*>(activation_matrix->packed_data()),
              activation_host_packed->packed.data(),
              activation_host_packed->packed.size(),
              cudaMemcpyHostToDevice) == cudaSuccess,
          metadata.fixture_name + ": packed activations should upload") ||
      !expect(
          cudaMemcpy(
              const_cast<std::uint8_t*>(activation_matrix->block_scales_data()),
              activation_host_packed->block_scales.data(),
              activation_host_packed->block_scales.size(),
              cudaMemcpyHostToDevice) == cudaSuccess,
          metadata.fixture_name + ": activation block scales should upload") ||
      !expect(
          cudaMemcpy(
              const_cast<std::uint8_t*>(activation_matrix->matmul_block_scales_data()),
              activation_matmul_scales.data(),
              activation_matmul_scales.size(),
              cudaMemcpyHostToDevice) == cudaSuccess,
          metadata.fixture_name + ": activation matmul scales should upload") ||
      !expect(
          cudaMemcpy(
              const_cast<std::uint8_t*>(activation_matrix->tensor_scale_data()),
              &activation_host_packed->tensor_scale,
              sizeof(float),
              cudaMemcpyHostToDevice) == cudaSuccess,
          metadata.fixture_name + ": activation tensor scale should upload")) {
    return false;
  }

  const std::vector<float> activation_reference = dequantize_nvfp4_matrix(
      activation_host_packed->packed.data(),
      activation_host_packed->packed.size(),
      activation_host_packed->block_scales.data(),
      activation_host_packed->block_scales.size(),
      activation_host_packed->tensor_scale,
      rows,
      metadata.input_cols);
  const std::vector<float> weight_reference = dequantize_nvfp4_matrix(
      weight_packed.data(),
      weight_packed.size(),
      weight_block_scales.data(),
      weight_block_scales.size(),
      weight_tensor_scale.front(),
      metadata.output_rows,
      metadata.input_cols);
  if (!expect(!activation_reference.empty(), metadata.fixture_name + ": activation reference should decode")) {
    return false;
  }
  if (!expect(!weight_reference.empty(), metadata.fixture_name + ": weight reference should decode")) {
    return false;
  }

  const std::vector<float> expected_output =
      cpu_reference(activation_reference, rows, weight_reference, metadata.output_rows, metadata.input_cols);
  float fixture_reference_max_abs_diff = 0.0f;
  double fixture_reference_mean_abs_diff = 0.0;
  for (std::size_t i = 0; i < expected_output.size(); ++i) {
    const float abs_diff = std::fabs(expected_output[i] - fixture_expected_output[i]);
    fixture_reference_max_abs_diff = std::max(fixture_reference_max_abs_diff, abs_diff);
    fixture_reference_mean_abs_diff += abs_diff;
  }
  fixture_reference_mean_abs_diff /= static_cast<double>(expected_output.size());

  const auto stats =
      RunNvfp4RowMajorFp32AccumToDevice(
          handle,
          *plan,
          MakeNvfp4PackedMatrixDeviceView(*activation_matrix),
          *weight,
          output.get());
  if (!expect(stats.has_value(), metadata.fixture_name + ": NVFP4 runtime path should execute")) {
    return false;
  }

  std::vector<float> actual_output(expected_output.size(), 0.0f);
  if (!expect(output->CopyToHost(actual_output.data(), actual_output.size()),
              metadata.fixture_name + ": output should copy to host")) {
    return false;
  }

  float max_abs_diff = 0.0f;
  double mean_abs_diff = 0.0;
  float max_abs_output = 0.0f;
  for (std::size_t i = 0; i < actual_output.size(); ++i) {
    if (!std::isfinite(actual_output[i])) {
      std::cerr << "FAIL: " << metadata.fixture_name << ": output contains non-finite values at index " << i
                << "\n";
      return false;
    }
    max_abs_output = std::max(max_abs_output, std::fabs(actual_output[i]));
    const float abs_diff = std::fabs(actual_output[i] - fixture_expected_output[i]);
    max_abs_diff = std::max(max_abs_diff, abs_diff);
    mean_abs_diff += abs_diff;
  }
  mean_abs_diff /= static_cast<double>(actual_output.size());

  if (!expect(
          stats->rows == rows && stats->cols == metadata.output_rows,
          metadata.fixture_name + ": runtime stats should report the expected output shape")) {
    return false;
  }
  if (!expect(
          stats->heuristic_count > 0,
          metadata.fixture_name + ": runtime path should find a cublasLt heuristic")) {
    return false;
  }
  if (!expect(
          max_abs_diff <= kMaxAbsDiffTolerance,
          metadata.fixture_name + ": runtime max abs diff should stay within the validated tolerance")) {
    return false;
  }
  if (!expect(
          mean_abs_diff <= kMeanAbsDiffTolerance,
          metadata.fixture_name + ": runtime mean abs diff should stay within the validated tolerance")) {
    return false;
  }

  std::cout << metadata.fixture_name << ": target_operator=" << metadata.target_operator
            << " max_abs_diff=" << max_abs_diff << " mean_abs_diff=" << mean_abs_diff
            << " max_abs_output=" << max_abs_output
            << " cpu_fixture_max_abs_diff=" << fixture_reference_max_abs_diff
            << " cpu_fixture_mean_abs_diff=" << fixture_reference_mean_abs_diff << "\n";
  return true;
}

bool test_nvfp4_oracle_fixtures_match_checkpoint_reference() {
  const std::vector<OracleFixtureMetadata> fixtures = discover_fixture_metadata();
  if (fixtures.empty()) {
    std::cout << "nvfp4_oracle_fixture_test: SKIP (no fixtures present)\n";
    return true;
  }

  const auto handle = CublasLtHandle::Create();
  if (!handle || !handle->valid()) {
    std::cout << "nvfp4_oracle_fixture_test: SKIP (no CUDA device or cublasLt unavailable)\n";
    return true;
  }

  GemmHeuristicCache cache;
  for (const auto& fixture : fixtures) {
    if (!run_fixture(*handle, &cache, fixture)) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  if (!test_nvfp4_oracle_fixtures_match_checkpoint_reference()) {
    return 1;
  }
  std::cout << "nvfp4_oracle_fixture_test: PASS\n";
  return 0;
}
