#include "nemotron/artifact_loader.h"
#include "nemotron/gemm_catalog.h"
#include "nemotron/kernel_catalog.h"
#include "nemotron/mamba_layer.h"
#include "nemotron/mamba_ops.h"
#include "nemotron/manifest.h"
#include "nemotron/model_schedule.h"
#include "nemotron/primitive_ops.h"
#include "nemotron/scaled_fp8_linear.h"
#include "nemotron/tensor_catalog.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <string>
#include <vector>

namespace {

using nemotron::CublasLtHandle;
using nemotron::DeviceTensorFp32;
using nemotron::GemmDescriptor;
using nemotron::GemmKernelFamily;
using nemotron::GemmHeuristicCache;
using nemotron::KernelTensorDescriptor;
using nemotron::MambaLayerBindings;
using nemotron::MambaLayerConfig;
using nemotron::MambaLayerRunTrace;
using nemotron::MambaLayerSlice;
using nemotron::RequestExecutionConfig;
using nemotron::RequestExecutionContext;
using nemotron::ScaledFp8LinearConfig;
using nemotron::ScaledFp8LinearOp;
using nemotron::ArtifactLoadMode;
using nemotron::ArtifactLoader;
using nemotron::BuildGemmCatalog;
using nemotron::BuildKernelCatalog;
using nemotron::BuildMambaLayerBindings;
using nemotron::BuildModelSchedule;
using nemotron::BuildTensorCatalog;
using nemotron::GemmCatalog;
using nemotron::KernelCatalog;
using nemotron::LoadManifestFromJsonFile;
using nemotron::ModelSchedule;
using nemotron::TensorCatalog;

struct FixtureMetadata {
  std::size_t layer_index = 0;
  std::size_t hidden_size = 0;
  std::size_t intermediate_size = 0;
  std::size_t num_heads = 0;
  std::size_t head_dim = 0;
  std::size_t state_size = 0;
  std::size_t n_groups = 0;
  std::size_t conv_kernel_size = 0;
  float input_rms_epsilon = 0.0f;
  float mixer_rms_epsilon = 0.0f;
  float time_step_min = 0.0f;
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

std::filesystem::path resolve_fixture_root(const char* env_name, const char* default_root) {
  const char* override_root = std::getenv(env_name);
  if (override_root != nullptr && std::string(override_root).size() != 0) {
    return std::filesystem::path(override_root);
  }
  return std::filesystem::path(default_root);
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

std::optional<float> parse_json_float_field(const std::string& json, const std::string& key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*([-+0-9.eE]+)");
  std::smatch match;
  if (!std::regex_search(json, match, pattern) || match.size() != 2) {
    return std::nullopt;
  }
  try {
    return std::stof(match[1].str());
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<FixtureMetadata> load_metadata(const std::filesystem::path& root) {
  const std::string json = read_text_file(root / "metadata.json");
  FixtureMetadata metadata;
  const auto layer_index = parse_json_uint_field(json, "layer_index");
  const auto hidden_size = parse_json_uint_field(json, "hidden_size");
  const auto intermediate_size = parse_json_uint_field(json, "intermediate_size");
  const auto num_heads = parse_json_uint_field(json, "num_heads");
  const auto head_dim = parse_json_uint_field(json, "head_dim");
  const auto state_size = parse_json_uint_field(json, "state_size");
  const auto n_groups = parse_json_uint_field(json, "n_groups");
  const auto conv_kernel_size = parse_json_uint_field(json, "conv_kernel_size");
  const auto input_rms_epsilon = parse_json_float_field(json, "input_rms_epsilon");
  const auto mixer_rms_epsilon = parse_json_float_field(json, "mixer_rms_epsilon");
  const auto time_step_min = parse_json_float_field(json, "time_step_min");
  if (!layer_index || !hidden_size || !intermediate_size || !num_heads || !head_dim ||
      !state_size || !n_groups || !conv_kernel_size || !input_rms_epsilon ||
      !mixer_rms_epsilon || !time_step_min) {
    return std::nullopt;
  }
  metadata.layer_index = *layer_index;
  metadata.hidden_size = *hidden_size;
  metadata.intermediate_size = *intermediate_size;
  metadata.num_heads = *num_heads;
  metadata.head_dim = *head_dim;
  metadata.state_size = *state_size;
  metadata.n_groups = *n_groups;
  metadata.conv_kernel_size = *conv_kernel_size;
  metadata.input_rms_epsilon = *input_rms_epsilon;
  metadata.mixer_rms_epsilon = *mixer_rms_epsilon;
  metadata.time_step_min = *time_step_min;
  return metadata;
}

KernelTensorDescriptor make_fp32_descriptor(
    const std::string& name,
    const std::vector<float>& values,
    std::vector<std::size_t> shape) {
  KernelTensorDescriptor descriptor;
  descriptor.tensor_name = name;
  descriptor.op_class = "oracle";
  descriptor.logical_shape = std::move(shape);
  descriptor.packed_shape = descriptor.logical_shape;
  descriptor.storage_dtype = "fp32";
  descriptor.compute_dtype = "fp32";
  descriptor.layout_tag = "row_major";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data = reinterpret_cast<const std::uint8_t*>(values.data());
  descriptor.packed_nbytes = values.size() * sizeof(float);
  return descriptor;
}

KernelTensorDescriptor make_fp8_descriptor(
    const std::string& name,
    const std::vector<std::uint8_t>& bytes,
    std::vector<std::size_t> shape) {
  KernelTensorDescriptor descriptor;
  descriptor.tensor_name = name;
  descriptor.op_class = "oracle";
  descriptor.logical_shape = std::move(shape);
  descriptor.packed_shape = descriptor.logical_shape;
  descriptor.storage_dtype = "fp8e4m3";
  descriptor.compute_dtype = "fp32";
  descriptor.layout_tag = "row_major";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data = bytes.data();
  descriptor.packed_nbytes = bytes.size();
  return descriptor;
}

GemmDescriptor make_dense_descriptor(
    const std::string& name,
    const std::vector<float>& weights,
    std::size_t rows,
    std::size_t cols) {
  GemmDescriptor descriptor;
  descriptor.tensor_name = name;
  descriptor.op_class = "oracle";
  descriptor.kernel_family = GemmKernelFamily::kDenseRowMajor;
  descriptor.output_rows = rows;
  descriptor.input_cols = cols;
  descriptor.storage_dtype = "fp32";
  descriptor.compute_dtype = "fp32";
  descriptor.layout_tag = "row_major";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data = reinterpret_cast<const std::uint8_t*>(weights.data());
  descriptor.packed_nbytes = weights.size() * sizeof(float);
  return descriptor;
}

float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return INFINITY;
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
  }
  return max_diff;
}

std::size_t numel_from_shape(const std::vector<std::size_t>& shape) {
  if (shape.empty()) {
    return 1;
  }
  std::size_t total = 1;
  for (const std::size_t dim : shape) {
    total *= dim;
  }
  return total;
}

bool env_var_enabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && std::string(value).size() != 0 && std::string(value) != "0";
}

std::filesystem::path resolve_path_override(
    const char* env_name,
    const std::filesystem::path& default_path) {
  const char* override_path = std::getenv(env_name);
  if (override_path != nullptr && std::string(override_path).size() != 0) {
    return std::filesystem::path(override_path);
  }
  return default_path;
}

std::filesystem::path default_runtime_root() {
  return std::filesystem::path(NEMOTRON_MAMBA_LAYER_ORACLE_FIXTURE_ROOT)
      .parent_path()
      .parent_path()
      .parent_path();
}

std::filesystem::path default_repo_root() {
  return default_runtime_root().parent_path();
}

std::vector<float> slice_rows(
    const std::vector<float>& values,
    std::size_t row_count,
    std::size_t row_width) {
  const std::size_t elements = row_count * row_width;
  if (values.size() < elements) {
    return {};
  }
  return std::vector<float>(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(elements));
}

std::vector<float> subtract_vectors(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return {};
  }
  std::vector<float> output(lhs.size(), 0.0f);
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    output[i] = lhs[i] - rhs[i];
  }
  return output;
}

bool write_float_file(const std::filesystem::path& path, const std::vector<float>& values) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    return false;
  }
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
  return static_cast<bool>(output);
}

struct VectorDiffSummary {
  float max_abs_diff = 0.0f;
  double rel_l2 = 0.0;
  std::size_t max_index = 0;
  float lhs_value = 0.0f;
  float rhs_value = 0.0f;
  std::size_t row = 0;
  std::size_t col = 0;
};

VectorDiffSummary summarize_diff(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::size_t row_width = 0) {
  VectorDiffSummary summary;
  if (lhs.size() != rhs.size() || lhs.empty()) {
    summary.max_abs_diff = INFINITY;
    return summary;
  }

  double lhs_l2 = 0.0;
  double diff_l2 = 0.0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    const double diff = static_cast<double>(lhs[i]) - static_cast<double>(rhs[i]);
    const float abs_diff = static_cast<float>(std::fabs(diff));
    if (abs_diff > summary.max_abs_diff) {
      summary.max_abs_diff = abs_diff;
      summary.max_index = i;
      summary.lhs_value = lhs[i];
      summary.rhs_value = rhs[i];
    }
    lhs_l2 += static_cast<double>(lhs[i]) * static_cast<double>(lhs[i]);
    diff_l2 += diff * diff;
  }
  summary.rel_l2 =
      lhs_l2 == 0.0 ? std::sqrt(diff_l2) : (std::sqrt(diff_l2) / std::sqrt(lhs_l2));
  if (row_width != 0) {
    summary.row = summary.max_index / row_width;
    summary.col = summary.max_index % row_width;
  }
  return summary;
}

void print_diff_summary(
    std::ostream& output,
    const std::string& label,
    const VectorDiffSummary& summary) {
  output << label
         << ": max_abs_diff=" << summary.max_abs_diff
         << " rel_l2=" << summary.rel_l2
         << " max_index=" << summary.max_index
         << " row=" << summary.row
         << " col=" << summary.col
         << " lhs=" << summary.lhs_value
         << " rhs=" << summary.rhs_value
         << "\n";
}

std::optional<std::vector<float>> read_descriptor_host_fp32(
    const KernelTensorDescriptor& descriptor) {
  if (descriptor.packed_data == nullptr) {
    return std::nullopt;
  }
  const std::size_t count = numel_from_shape(descriptor.logical_shape);
  if (count == 0) {
    return std::nullopt;
  }

  std::vector<float> values(count, 0.0f);
  if (descriptor.storage_dtype == "fp32" ||
      descriptor.storage_dtype == "float32" ||
      descriptor.storage_dtype == "float") {
    if (descriptor.packed_nbytes != count * sizeof(float)) {
      return std::nullopt;
    }
    std::memcpy(values.data(), descriptor.packed_data, descriptor.packed_nbytes);
    return values;
  }
  if (descriptor.storage_dtype == "bf16" || descriptor.storage_dtype == "bfloat16") {
    if (descriptor.packed_nbytes != count * sizeof(__nv_bfloat16)) {
      return std::nullopt;
    }
    const auto* src = reinterpret_cast<const __nv_bfloat16*>(descriptor.packed_data);
    for (std::size_t i = 0; i < count; ++i) {
      values[i] = __bfloat162float(src[i]);
    }
    return values;
  }
  return std::nullopt;
}

std::optional<float> read_scalar_descriptor_fp32(
    const KernelTensorDescriptor& descriptor) {
  const auto values = read_descriptor_host_fp32(descriptor);
  if (!values.has_value() || values->size() != 1) {
    return std::nullopt;
  }
  return (*values)[0];
}

std::unique_ptr<DeviceTensorFp32> upload_descriptor_to_device_fp32_flat(
    const KernelTensorDescriptor& descriptor) {
  const auto host = read_descriptor_host_fp32(descriptor);
  if (!host.has_value()) {
    return nullptr;
  }
  auto device = DeviceTensorFp32::Create({host->size()});
  if (!device || !device->valid() ||
      !device->CopyFromHost(host->data(), host->size())) {
    return nullptr;
  }
  return device;
}

struct MambaReplayResult {
  std::vector<float> norm_output;
  std::vector<float> in_proj_output;
  std::vector<float> conv_output;
  std::vector<float> ssm_output;
  std::vector<float> scan_output;
  std::vector<float> out_proj_output;
  std::vector<float> residual_output;
  std::vector<float> final_conv_state;
  std::vector<float> final_ssm_state;
};

std::optional<MambaReplayResult> run_mamba_replay(
    CublasLtHandle& cublas,
    GemmHeuristicCache* heuristic_cache,
    const MambaLayerConfig& layer_config,
    const MambaLayerBindings& bindings,
    const std::vector<float>& input_hidden) {
  const std::size_t hidden_size = layer_config.hidden_size;
  if (hidden_size == 0 || input_hidden.size() % hidden_size != 0) {
    return std::nullopt;
  }
  const std::size_t input_rows = input_hidden.size() / hidden_size;
  const std::size_t conv_dim =
      layer_config.intermediate_size + (2 * layer_config.n_groups * layer_config.state_size);
  const std::size_t projection_size =
      layer_config.intermediate_size + conv_dim + layer_config.num_heads;
  const std::size_t conv_state_elems = conv_dim * layer_config.conv_kernel_size;
  const std::size_t ssm_state_elems =
      layer_config.num_heads * layer_config.head_dim * layer_config.state_size;

  auto slice = MambaLayerSlice::Create(layer_config, bindings);
  if (!slice || !slice->valid()) {
    return std::nullopt;
  }

  RequestExecutionConfig request_config;
  request_config.hidden_size = hidden_size;
  request_config.max_tokens = input_rows;
  request_config.scratch_tokens = input_rows;
  request_config.mamba_hidden_size = hidden_size;
  request_config.mamba_projection_size = projection_size;
  request_config.mamba_intermediate_size = layer_config.intermediate_size;
  request_config.mamba_conv_state_bytes_fp32 = conv_state_elems * sizeof(float);
  request_config.mamba_state_bytes_fp32 = ssm_state_elems * sizeof(float);

  std::vector<float> zero_conv_state(conv_state_elems, 0.0f);
  std::vector<float> zero_ssm_state(ssm_state_elems, 0.0f);

  auto request = RequestExecutionContext::Create(request_config);
  auto input = DeviceTensorFp32::Create({input_rows, hidden_size});
  auto output = DeviceTensorFp32::Create({input_rows, hidden_size});
  if (!request || !request->valid() ||
      !input || !input->valid() ||
      !output || !output->valid() ||
      !input->CopyFromHost(input_hidden.data(), input_hidden.size()) ||
      !request->mamba_conv_state()->CopyFromHost(
          zero_conv_state.data(),
          zero_conv_state.size()) ||
      !request->mamba_state()->CopyFromHost(
          zero_ssm_state.data(),
          zero_ssm_state.size())) {
    return std::nullopt;
  }

  MambaLayerRunTrace trace;
  if (!slice->Run(cublas, heuristic_cache, *request, *input, output.get(), &trace)) {
    return std::nullopt;
  }

  MambaReplayResult result;
  result.norm_output = trace.norm_output;
  result.in_proj_output = trace.in_proj_output;
  result.scan_output = trace.scan_output;
  result.out_proj_output = trace.projected_output;
  result.residual_output.resize(output->numel(), 0.0f);
  result.final_conv_state.resize(conv_state_elems, 0.0f);
  result.final_ssm_state.resize(ssm_state_elems, 0.0f);
  if (!output->CopyToHost(result.residual_output.data(), result.residual_output.size()) ||
      !request->mamba_conv_state()->CopyToHost(
          result.final_conv_state.data(),
          result.final_conv_state.size()) ||
      !request->mamba_state()->CopyToHost(
          result.final_ssm_state.data(),
          result.final_ssm_state.size())) {
    return std::nullopt;
  }

  auto mixer_norm_weight = upload_descriptor_to_device_fp32_flat(*bindings.mixer_norm_weight);
  auto conv1d_weight = upload_descriptor_to_device_fp32_flat(*bindings.conv1d_weight);
  auto conv1d_bias = upload_descriptor_to_device_fp32_flat(*bindings.conv1d_bias);
  auto a_log = upload_descriptor_to_device_fp32_flat(*bindings.A_log);
  auto d = upload_descriptor_to_device_fp32_flat(*bindings.D);
  auto dt_bias = upload_descriptor_to_device_fp32_flat(*bindings.dt_bias);
  if (!mixer_norm_weight || !conv1d_weight || !conv1d_bias || !a_log || !d || !dt_bias) {
    return std::nullopt;
  }

  auto manual_request = RequestExecutionContext::Create(request_config);
  auto projected = DeviceTensorFp32::Create({input_rows, projection_size});
  auto conv_output = DeviceTensorFp32::Create({input_rows, conv_dim});
  auto y_output = DeviceTensorFp32::Create({input_rows, layer_config.intermediate_size});
  auto grouped_output = DeviceTensorFp32::Create({input_rows, layer_config.intermediate_size});
  if (!manual_request || !manual_request->valid() ||
      !projected || !projected->valid() ||
      !conv_output || !conv_output->valid() ||
      !y_output || !y_output->valid() ||
      !grouped_output || !grouped_output->valid() ||
      !manual_request->mamba_conv_state()->CopyFromHost(
          zero_conv_state.data(),
          zero_conv_state.size()) ||
      !manual_request->mamba_state()->CopyFromHost(
          zero_ssm_state.data(),
          zero_ssm_state.size()) ||
      !projected->CopyFromHost(trace.in_proj_output.data(), trace.in_proj_output.size())) {
    return std::nullopt;
  }

  if (!nemotron::MambaConv1dSiluUpdateFp32(
          *projected,
          layer_config.intermediate_size,
          conv_dim,
          layer_config.conv_kernel_size,
          layer_config.conv_state_offset_elems,
          *conv1d_weight,
          *conv1d_bias,
          manual_request->mamba_conv_state(),
          conv_output.get()) ||
      !nemotron::MambaSsmUpdateFp32(
          *projected,
          *conv_output,
          layer_config.intermediate_size,
          conv_dim,
          layer_config.num_heads,
          layer_config.head_dim,
          layer_config.state_size,
          layer_config.n_groups,
          layer_config.time_step_min,
          layer_config.ssm_state_offset_elems,
          *a_log,
          *d,
          *dt_bias,
          manual_request->mamba_state(),
          y_output.get()) ||
      !nemotron::GroupedRmsNormGatedFp32(
          *y_output,
          *projected,
          *mixer_norm_weight,
          layer_config.n_groups,
          layer_config.mixer_rms_epsilon,
          grouped_output.get())) {
    return std::nullopt;
  }

  result.conv_output.resize(conv_output->numel(), 0.0f);
  result.ssm_output.resize(y_output->numel(), 0.0f);
  std::vector<float> manual_grouped_output(grouped_output->numel(), 0.0f);
  std::vector<float> manual_conv_state(conv_state_elems, 0.0f);
  std::vector<float> manual_ssm_state(ssm_state_elems, 0.0f);
  if (!conv_output->CopyToHost(result.conv_output.data(), result.conv_output.size()) ||
      !y_output->CopyToHost(result.ssm_output.data(), result.ssm_output.size()) ||
      !grouped_output->CopyToHost(manual_grouped_output.data(), manual_grouped_output.size()) ||
      !manual_request->mamba_conv_state()->CopyToHost(
          manual_conv_state.data(),
          manual_conv_state.size()) ||
      !manual_request->mamba_state()->CopyToHost(
          manual_ssm_state.data(),
          manual_ssm_state.size())) {
    return std::nullopt;
  }

  if (max_abs_diff(manual_grouped_output, result.scan_output) > 1.0e-6f ||
      max_abs_diff(manual_conv_state, result.final_conv_state) > 1.0e-6f ||
      max_abs_diff(manual_ssm_state, result.final_ssm_state) > 1.0e-6f) {
    return std::nullopt;
  }

  return result;
}

struct PrefixOracleReference {
  FixtureMetadata metadata;
  std::vector<float> input_hidden;
  std::vector<float> expected_norm_output;
  std::vector<float> expected_in_proj_output;
  std::vector<float> expected_updated_conv_state;
  std::vector<float> expected_next_ssm_state;
  std::vector<float> expected_final_output;
};

std::optional<PrefixOracleReference> load_prefix_oracle_reference(
    const std::filesystem::path& root) {
  const auto metadata = load_metadata(root);
  if (!metadata.has_value()) {
    return std::nullopt;
  }
  PrefixOracleReference reference;
  reference.metadata = *metadata;
  reference.input_hidden = read_float_file(root / "input_hidden_fp32.bin");
  reference.expected_norm_output = read_float_file(root / "expected_norm_output_fp32.bin");
  reference.expected_in_proj_output = read_float_file(root / "expected_in_proj_output_fp32.bin");
  reference.expected_updated_conv_state =
      read_float_file(root / "expected_updated_conv_state_fp32.bin");
  reference.expected_next_ssm_state =
      read_float_file(root / "expected_next_ssm_state_fp32.bin");
  reference.expected_final_output = read_float_file(root / "expected_final_output_fp32.bin");
  return reference;
}

bool verify_prefix_oracle_against_replay(
    CublasLtHandle& cublas,
    GemmHeuristicCache* heuristic_cache,
    const MambaLayerConfig& layer_config,
    const MambaLayerBindings& bindings,
    const PrefixOracleReference& oracle,
    std::ostream& output) {
  const auto replay =
      run_mamba_replay(cublas, heuristic_cache, layer_config, bindings, oracle.input_hidden);
  if (!expect(replay.has_value(), "prefix oracle replay should execute")) {
    return false;
  }
  const VectorDiffSummary norm_diff =
      summarize_diff(replay->norm_output, oracle.expected_norm_output, layer_config.hidden_size);
  const VectorDiffSummary in_proj_diff =
      summarize_diff(replay->in_proj_output, oracle.expected_in_proj_output);
  const VectorDiffSummary conv_state_diff =
      summarize_diff(replay->final_conv_state, oracle.expected_updated_conv_state);
  const VectorDiffSummary ssm_state_diff =
      summarize_diff(replay->final_ssm_state, oracle.expected_next_ssm_state);
  const VectorDiffSummary output_diff =
      summarize_diff(replay->residual_output, oracle.expected_final_output, layer_config.hidden_size);
  print_diff_summary(output, "prefix_oracle.norm_output", norm_diff);
  print_diff_summary(output, "prefix_oracle.in_proj_output", in_proj_diff);
  print_diff_summary(output, "prefix_oracle.conv_state", conv_state_diff);
  print_diff_summary(output, "prefix_oracle.ssm_state", ssm_state_diff);
  print_diff_summary(output, "prefix_oracle.residual_output", output_diff);
  return expect(norm_diff.max_abs_diff <= 1.0e-6f, "prefix oracle norm output should match") &&
         expect(in_proj_diff.max_abs_diff <= 2.0e-3f, "prefix oracle in-proj output should match") &&
         expect(output_diff.max_abs_diff <= 2.0e-3f, "prefix oracle residual output should match");
}

bool run_mamba_layer_vllm_microrepro() {
  const auto cublas = CublasLtHandle::Create();
  if (!cublas || !cublas->valid()) {
    std::cout << "mamba_layer_oracle_test: SKIP (no CUDA device)\n";
    return true;
  }

  const std::filesystem::path runtime_root = default_runtime_root();
  const std::filesystem::path repo_root = default_repo_root();
  const std::filesystem::path manifest_path = resolve_path_override(
      "NEMOTRON_FORWARD_MANIFEST",
      runtime_root / "artifacts/manifests/forward_runtime_manifest_unverified.json");
  const std::filesystem::path input_path = resolve_path_override(
      "NEMOTRON_MAMBA_LAYER_INPUT_FP32",
      repo_root / "proj-2026-04-05-0516/runtime-prefill-dump/embedding_output_fp32.bin");
  const std::filesystem::path runtime_reference_path = resolve_path_override(
      "NEMOTRON_MAMBA_LAYER_RUNTIME_REFERENCE_FP32",
      repo_root / "proj-2026-04-05-0516/runtime-prefill-dump/captured_layer_000_output_fp32.bin");
  const std::filesystem::path vllm_reference_path = resolve_path_override(
      "NEMOTRON_MAMBA_LAYER_VLLM_REFERENCE_FP32",
      runtime_root / "artifacts/debug/vllm_trace_fullprompt/layer_000_output_fp32.bin");
  const std::filesystem::path dump_root = resolve_path_override(
      "NEMOTRON_MAMBA_LAYER_DUMP_ROOT",
      repo_root / "proj-2026-04-05-0516/mamba-layer0-vllm-microrepro");
  const std::filesystem::path prefix_oracle_root = resolve_path_override(
      "NEMOTRON_MAMBA_LAYER_PREFIX_ORACLE_ROOT",
      repo_root / "proj-2026-04-04-1657/layer0-prefix-oracle.zero");

  if (!expect(std::filesystem::exists(manifest_path), "manifest path should exist") ||
      !expect(std::filesystem::exists(input_path), "microrepro input path should exist") ||
      !expect(
          std::filesystem::exists(runtime_reference_path),
          "microrepro runtime reference path should exist") ||
      !expect(
          std::filesystem::exists(vllm_reference_path),
          "microrepro vLLM reference path should exist") ||
      !expect(
          std::filesystem::exists(prefix_oracle_root),
          "prefix oracle fixture root should exist")) {
    return false;
  }

  const auto manifest_result = LoadManifestFromJsonFile(manifest_path);
  if (!expect(manifest_result.ok, "microrepro manifest should parse")) {
    return false;
  }
  auto artifact_loader = ArtifactLoader::OpenVerifiedWithMode(
      manifest_result.manifest,
      manifest_path,
      ArtifactLoadMode::kMmap);
  if (!expect(static_cast<bool>(artifact_loader), "artifact loader should open for microrepro")) {
    return false;
  }

  const TensorCatalog tensor_catalog =
      BuildTensorCatalog(manifest_result.manifest, *artifact_loader);
  const KernelCatalog kernel_catalog = BuildKernelCatalog(tensor_catalog);
  const GemmCatalog gemm_catalog = BuildGemmCatalog(kernel_catalog);
  const ModelSchedule model_schedule = BuildModelSchedule(manifest_result.manifest);
  if (!expect(tensor_catalog.valid(), "tensor catalog should build for microrepro") ||
      !expect(kernel_catalog.valid(), "kernel catalog should build for microrepro") ||
      !expect(gemm_catalog.valid(), "gemm catalog should build for microrepro") ||
      !expect(model_schedule.valid(), "model schedule should build for microrepro")) {
    return false;
  }

  const auto* layer0 = model_schedule.FindLayer(0);
  if (!expect(layer0 != nullptr, "model schedule should expose layer 0") ||
      !expect(layer0->has_mamba, "layer 0 should be a Mamba layer")) {
    return false;
  }

  const auto bindings = BuildMambaLayerBindings(*layer0, kernel_catalog, gemm_catalog);
  if (!expect(bindings.has_value(), "layer-0 Mamba bindings should build from the manifest")) {
    return false;
  }

  constexpr std::size_t kLayerIndex = 0;
  constexpr std::size_t kHiddenSize = 4096;
  constexpr std::size_t kIntermediateSize = 8192;
  constexpr std::size_t kNumHeads = 128;
  constexpr std::size_t kHeadDim = 64;
  constexpr std::size_t kStateSize = 128;
  constexpr std::size_t kNGroups = 8;
  constexpr std::size_t kConvKernelSize = 4;
  constexpr float kLayerNormEpsilon = 1.0e-5f;
  constexpr float kTimeStepMin = 1.0e-3f;
  const std::size_t conv_dim = kIntermediateSize + (2 * kNGroups * kStateSize);
  if (!expect(
          numel_from_shape(bindings->input_norm_weight->logical_shape) == kHiddenSize,
          "layer-0 input norm shape should match expected hidden size") ||
      !expect(
          numel_from_shape(bindings->mixer_norm_weight->logical_shape) == kIntermediateSize,
          "layer-0 mixer norm shape should match expected intermediate size") ||
      !expect(
          numel_from_shape(bindings->A_log->logical_shape) == kNumHeads,
          "layer-0 A_log shape should match expected head count") ||
      !expect(
          numel_from_shape(bindings->conv1d_bias->logical_shape) == conv_dim,
          "layer-0 conv bias shape should match expected conv dim")) {
    return false;
  }

  MambaLayerConfig layer_config;
  layer_config.layer_index = kLayerIndex;
  layer_config.hidden_size = kHiddenSize;
  layer_config.intermediate_size = kIntermediateSize;
  layer_config.num_heads = kNumHeads;
  layer_config.head_dim = kHeadDim;
  layer_config.state_size = kStateSize;
  layer_config.n_groups = kNGroups;
  layer_config.conv_kernel_size = kConvKernelSize;
  layer_config.conv_state_offset_elems = 0;
  layer_config.ssm_state_offset_elems = 0;
  layer_config.input_rms_epsilon = kLayerNormEpsilon;
  layer_config.mixer_rms_epsilon = kLayerNormEpsilon;
  layer_config.time_step_min = kTimeStepMin;

  const std::vector<float> input_hidden = read_float_file(input_path);
  const std::vector<float> runtime_reference = read_float_file(runtime_reference_path);
  const std::vector<float> vllm_reference = read_float_file(vllm_reference_path);
  const auto prefix_oracle = load_prefix_oracle_reference(prefix_oracle_root);
  if (!expect(prefix_oracle.has_value(), "prefix oracle fixture should load")) {
    return false;
  }
  if (!expect(
          input_hidden.size() == runtime_reference.size() &&
              input_hidden.size() == vllm_reference.size(),
          "input and reference tensors should match size")) {
    return false;
  }

  std::ostringstream summary;
  summary << "mamba_layer_vllm_microrepro\n"
          << "manifest=" << manifest_path << "\n"
          << "input=" << input_path << "\n"
          << "runtime_reference=" << runtime_reference_path << "\n"
          << "vllm_reference=" << vllm_reference_path << "\n"
          << "dump_root=" << dump_root << "\n";

  GemmHeuristicCache heuristic_cache;
  if (!verify_prefix_oracle_against_replay(
          *cublas,
          &heuristic_cache,
          layer_config,
          *bindings,
          *prefix_oracle,
          summary)) {
    return false;
  }

  const auto replay =
      run_mamba_replay(*cublas, &heuristic_cache, layer_config, *bindings, input_hidden);
  if (!expect(replay.has_value(), "manifest-backed layer-0 replay should execute")) {
    return false;
  }

  const VectorDiffSummary replay_vs_runtime =
      summarize_diff(replay->residual_output, runtime_reference, layer_config.hidden_size);
  const VectorDiffSummary replay_vs_vllm =
      summarize_diff(replay->residual_output, vllm_reference, layer_config.hidden_size);
  const std::vector<float> inferred_runtime_out_proj =
      subtract_vectors(runtime_reference, input_hidden);
  const std::vector<float> inferred_vllm_out_proj =
      subtract_vectors(vllm_reference, input_hidden);
  const VectorDiffSummary out_proj_vs_runtime =
      summarize_diff(replay->out_proj_output, inferred_runtime_out_proj, layer_config.hidden_size);
  const VectorDiffSummary out_proj_vs_vllm =
      summarize_diff(replay->out_proj_output, inferred_vllm_out_proj, layer_config.hidden_size);

  std::vector<float> rowwise_final_max_diff(
      replay->residual_output.size() / layer_config.hidden_size,
      0.0f);
  for (std::size_t row = 0; row < rowwise_final_max_diff.size(); ++row) {
    const std::size_t begin = row * layer_config.hidden_size;
    const std::size_t end = begin + layer_config.hidden_size;
    float row_max = 0.0f;
    for (std::size_t i = begin; i < end; ++i) {
      row_max = std::max(
          row_max,
          std::fabs(replay->residual_output[i] - vllm_reference[i]));
    }
    rowwise_final_max_diff[row] = row_max;
  }
  const auto max_row_iter =
      std::max_element(rowwise_final_max_diff.begin(), rowwise_final_max_diff.end());
  const std::size_t max_row =
      static_cast<std::size_t>(std::distance(rowwise_final_max_diff.begin(), max_row_iter));

  if (!expect(
          replay_vs_runtime.max_abs_diff <= 1.0e-6f,
          "manifest-backed layer-0 replay should match the captured runtime output") ||
      !expect(
          out_proj_vs_runtime.max_abs_diff <= 1.0e-6f,
          "manifest-backed out-proj output should match the inferred runtime out-proj output")) {
    return false;
  }

  const bool residual_add_excluded =
      std::fabs(replay_vs_vllm.max_abs_diff - out_proj_vs_vllm.max_abs_diff) <= 1.0e-6f &&
      replay_vs_vllm.max_index == out_proj_vs_vllm.max_index;
  const bool drift_grows_beyond_conv_horizon =
      rowwise_final_max_diff.size() > kConvKernelSize &&
      *max_row_iter > rowwise_final_max_diff[0] &&
      max_row >= (kConvKernelSize * 2);

  print_diff_summary(summary, "manifest_replay.residual_vs_runtime", replay_vs_runtime);
  print_diff_summary(summary, "manifest_replay.residual_vs_vllm", replay_vs_vllm);
  print_diff_summary(summary, "manifest_replay.out_proj_vs_runtime", out_proj_vs_runtime);
  print_diff_summary(summary, "manifest_replay.out_proj_vs_vllm", out_proj_vs_vllm);
  summary << "rowwise_final_vllm_max_abs_diff";
  for (std::size_t row = 0; row < rowwise_final_max_diff.size(); ++row) {
    summary << (row == 0 ? ": " : ", ") << row << "=" << rowwise_final_max_diff[row];
  }
  summary << "\n"
          << "inference.residual_add_excluded="
          << (residual_add_excluded ? "true" : "false") << "\n"
          << "inference.max_row=" << max_row << "\n"
          << "inference.drift_grows_beyond_conv_horizon="
          << (drift_grows_beyond_conv_horizon ? "true" : "false") << "\n"
          << "inference.likely_source="
          << (residual_add_excluded && drift_grows_beyond_conv_horizon
                  ? "ssm_update_or_grouped_norm_gating (stateful path, SSM most likely)"
                  : "pre_out_proj_path (needs vLLM sub-op trace for tighter isolation)")
          << "\n";

  std::filesystem::create_directories(dump_root);
  if (!write_float_file(dump_root / "norm_output_fp32.bin", replay->norm_output) ||
      !write_float_file(dump_root / "in_proj_output_fp32.bin", replay->in_proj_output) ||
      !write_float_file(dump_root / "conv_output_fp32.bin", replay->conv_output) ||
      !write_float_file(dump_root / "ssm_output_fp32.bin", replay->ssm_output) ||
      !write_float_file(dump_root / "scan_output_fp32.bin", replay->scan_output) ||
      !write_float_file(dump_root / "out_proj_output_fp32.bin", replay->out_proj_output) ||
      !write_float_file(dump_root / "residual_output_fp32.bin", replay->residual_output) ||
      !write_float_file(
          dump_root / "runtime_reference_out_proj_output_fp32.bin",
          inferred_runtime_out_proj) ||
      !write_float_file(
          dump_root / "vllm_reference_out_proj_output_fp32.bin",
          inferred_vllm_out_proj) ||
      !write_float_file(dump_root / "final_conv_state_fp32.bin", replay->final_conv_state) ||
      !write_float_file(dump_root / "final_ssm_state_fp32.bin", replay->final_ssm_state)) {
    return expect(false, "microrepro output dumps should write");
  }
  std::ofstream summary_output(dump_root / "summary.txt");
  if (!summary_output) {
    return expect(false, "microrepro summary should open");
  }
  summary_output << summary.str();
  summary_output.close();
  std::cout << summary.str();
  return true;
}

bool run_mamba_layer_fixture() {
  const auto cublas = CublasLtHandle::Create();
  if (!cublas || !cublas->valid()) {
    std::cout << "mamba_layer_oracle_test: SKIP (no CUDA device)\n";
    return true;
  }

  const std::filesystem::path fixture_root = resolve_fixture_root(
      "NEMOTRON_MAMBA_LAYER_ORACLE_FIXTURE_ROOT_OVERRIDE",
      NEMOTRON_MAMBA_LAYER_ORACLE_FIXTURE_ROOT);
  const auto metadata = load_metadata(fixture_root);
  if (!expect(metadata.has_value(), "mamba layer oracle fixture metadata should load")) {
    return false;
  }

  const std::vector<float> input_hidden = read_float_file(fixture_root / "input_hidden_fp32.bin");
  const std::vector<float> initial_conv_state = read_float_file(fixture_root / "initial_conv_state_fp32.bin");
  const std::vector<float> initial_ssm_state = read_float_file(fixture_root / "initial_ssm_state_fp32.bin");
  const std::vector<float> input_norm_weight = read_float_file(fixture_root / "input_norm_weight_fp32.bin");
  const std::vector<float> mixer_norm_weight = read_float_file(fixture_root / "mixer_norm_weight_fp32.bin");
  const bool has_in_proj_fp8 = std::filesystem::exists(fixture_root / "in_proj_weight_fp8.bin");
  const std::vector<std::uint8_t> in_proj_weight_fp8 =
      has_in_proj_fp8 ? read_file_bytes(fixture_root / "in_proj_weight_fp8.bin") : std::vector<std::uint8_t>{};
  const std::vector<float> in_proj_weight_fp32 =
      !has_in_proj_fp8 ? read_float_file(fixture_root / "in_proj_weight_fp32.bin") : std::vector<float>{};
  const std::vector<float> in_proj_weight_scale =
      has_in_proj_fp8 ? read_float_file(fixture_root / "in_proj_weight_scale_fp32.bin") : std::vector<float>{};
  const std::vector<float> in_proj_input_scale =
      has_in_proj_fp8 ? read_float_file(fixture_root / "in_proj_input_scale_fp32.bin") : std::vector<float>{};
  const std::vector<float> conv1d_weight = read_float_file(fixture_root / "conv1d_weight_fp32.bin");
  const std::vector<float> conv1d_bias = read_float_file(fixture_root / "conv1d_bias_fp32.bin");
  const std::vector<float> A_log = read_float_file(fixture_root / "A_log_fp32.bin");
  const std::vector<float> D = read_float_file(fixture_root / "D_fp32.bin");
  const std::vector<float> dt_bias = read_float_file(fixture_root / "dt_bias_fp32.bin");
  const bool has_out_proj_fp8 = std::filesystem::exists(fixture_root / "out_proj_weight_fp8.bin");
  const std::vector<std::uint8_t> out_proj_weight_fp8 =
      has_out_proj_fp8 ? read_file_bytes(fixture_root / "out_proj_weight_fp8.bin") : std::vector<std::uint8_t>{};
  const std::vector<float> out_proj_weight_fp32 =
      !has_out_proj_fp8 ? read_float_file(fixture_root / "out_proj_weight_fp32.bin") : std::vector<float>{};
  const std::vector<float> out_proj_weight_scale =
      has_out_proj_fp8 ? read_float_file(fixture_root / "out_proj_weight_scale_fp32.bin") : std::vector<float>{};
  const std::vector<float> out_proj_input_scale =
      has_out_proj_fp8 ? read_float_file(fixture_root / "out_proj_input_scale_fp32.bin") : std::vector<float>{};
  const std::vector<float> expected_norm_output = read_float_file(fixture_root / "expected_norm_output_fp32.bin");
  const std::vector<float> expected_in_proj_output = read_float_file(fixture_root / "expected_in_proj_output_fp32.bin");
  const std::vector<float> expected_next_ssm_state = read_float_file(fixture_root / "expected_next_ssm_state_fp32.bin");
  const std::vector<float> expected_updated_conv_state = read_float_file(fixture_root / "expected_updated_conv_state_fp32.bin");
  const std::vector<float> expected_final_output = read_float_file(fixture_root / "expected_final_output_fp32.bin");

  const std::size_t conv_dim = metadata->intermediate_size + (2 * metadata->n_groups * metadata->state_size);
  const std::size_t conv_state_elems = conv_dim * metadata->conv_kernel_size;
  const std::size_t ssm_state_elems = metadata->num_heads * metadata->head_dim * metadata->state_size;
  const std::size_t in_proj_rows = metadata->intermediate_size + conv_dim + metadata->num_heads;
  const bool input_hidden_rows_ok =
      metadata->hidden_size != 0 && input_hidden.size() % metadata->hidden_size == 0;
  const std::size_t input_rows =
      input_hidden_rows_ok ? (input_hidden.size() / metadata->hidden_size) : 0;

  if (!expect(input_hidden_rows_ok && input_rows != 0, "input hidden rows should match metadata") ||
      !expect(initial_conv_state.size() == conv_state_elems, "conv state size should match metadata") ||
      !expect(initial_ssm_state.size() == ssm_state_elems, "ssm state size should match metadata") ||
      !expect(input_norm_weight.size() == metadata->hidden_size, "input norm size should match metadata") ||
      !expect(mixer_norm_weight.size() == metadata->intermediate_size, "mixer norm size should match metadata") ||
      !expect(
          (has_in_proj_fp8 &&
           in_proj_weight_fp8.size() == in_proj_rows * metadata->hidden_size &&
           in_proj_weight_scale.size() == 1 &&
           in_proj_input_scale.size() == 1) ||
              (!has_in_proj_fp8 &&
               in_proj_weight_fp32.size() == in_proj_rows * metadata->hidden_size),
          "in_proj weights should match metadata") ||
      !expect(conv1d_weight.size() == conv_state_elems, "conv1d weight size should match metadata") ||
      !expect(conv1d_bias.size() == conv_dim, "conv1d bias size should match metadata") ||
      !expect(A_log.size() == metadata->num_heads, "A_log size should match metadata") ||
      !expect(D.size() == metadata->num_heads, "D size should match metadata") ||
      !expect(dt_bias.size() == metadata->num_heads, "dt_bias size should match metadata") ||
      !expect(
          (has_out_proj_fp8 &&
           out_proj_weight_fp8.size() == metadata->hidden_size * metadata->intermediate_size &&
           out_proj_weight_scale.size() == 1 &&
           out_proj_input_scale.size() == 1) ||
              (!has_out_proj_fp8 &&
               out_proj_weight_fp32.size() == metadata->hidden_size * metadata->intermediate_size),
          "out_proj weights should match metadata") ||
      !expect(expected_next_ssm_state.size() == ssm_state_elems, "expected ssm state should match metadata") ||
      !expect(expected_updated_conv_state.size() == conv_state_elems, "expected conv state should match metadata") ||
      !expect(expected_norm_output.size() == input_rows * metadata->hidden_size, "expected norm output should match metadata") ||
      !expect(expected_in_proj_output.size() == input_rows * in_proj_rows, "expected in_proj output should match metadata") ||
      !expect(expected_final_output.size() == input_rows * metadata->hidden_size, "expected final output should match metadata")) {
    return false;
  }

  const std::string layer_prefix = "backbone.layers." + std::to_string(metadata->layer_index);
  const std::string mixer_prefix = layer_prefix + ".mixer";
  const auto input_norm_descriptor =
      make_fp32_descriptor(layer_prefix + ".norm.weight", input_norm_weight, {metadata->hidden_size});
  const auto mixer_norm_descriptor =
      make_fp32_descriptor(mixer_prefix + ".norm.weight", mixer_norm_weight, {metadata->intermediate_size});
  const auto in_proj_fp8_weight_descriptor =
      has_in_proj_fp8
          ? std::make_optional(
                make_fp8_descriptor(mixer_prefix + ".in_proj.weight", in_proj_weight_fp8, {in_proj_rows, metadata->hidden_size}))
          : std::nullopt;
  const auto in_proj_dense_weight_descriptor =
      !has_in_proj_fp8
          ? std::make_optional(
                make_dense_descriptor(mixer_prefix + ".in_proj.weight", in_proj_weight_fp32, in_proj_rows, metadata->hidden_size))
          : std::nullopt;
  const auto in_proj_weight_scale_descriptor =
      has_in_proj_fp8
          ? std::make_optional(make_fp32_descriptor(mixer_prefix + ".in_proj.weight_scale", in_proj_weight_scale, {1}))
          : std::nullopt;
  const auto in_proj_input_scale_descriptor =
      has_in_proj_fp8
          ? std::make_optional(make_fp32_descriptor(mixer_prefix + ".in_proj.input_scale", in_proj_input_scale, {1}))
          : std::nullopt;
  const auto conv1d_weight_descriptor =
      make_fp32_descriptor(mixer_prefix + ".conv1d.weight", conv1d_weight, {conv_dim, 1, metadata->conv_kernel_size});
  const auto conv1d_bias_descriptor =
      make_fp32_descriptor(mixer_prefix + ".conv1d.bias", conv1d_bias, {conv_dim});
  const auto A_log_descriptor =
      make_fp32_descriptor(mixer_prefix + ".A_log", A_log, {metadata->num_heads});
  const auto D_descriptor =
      make_fp32_descriptor(mixer_prefix + ".D", D, {metadata->num_heads});
  const auto dt_bias_descriptor =
      make_fp32_descriptor(mixer_prefix + ".dt_bias", dt_bias, {metadata->num_heads});
  const auto out_proj_fp8_weight_descriptor =
      has_out_proj_fp8
          ? std::make_optional(
                make_fp8_descriptor(mixer_prefix + ".out_proj.weight", out_proj_weight_fp8, {metadata->hidden_size, metadata->intermediate_size}))
          : std::nullopt;
  const auto out_proj_dense_weight_descriptor =
      !has_out_proj_fp8
          ? std::make_optional(
                make_dense_descriptor(mixer_prefix + ".out_proj.weight", out_proj_weight_fp32, metadata->hidden_size, metadata->intermediate_size))
          : std::nullopt;
  const auto out_proj_weight_scale_descriptor =
      has_out_proj_fp8
          ? std::make_optional(make_fp32_descriptor(mixer_prefix + ".out_proj.weight_scale", out_proj_weight_scale, {1}))
          : std::nullopt;
  const auto out_proj_input_scale_descriptor =
      has_out_proj_fp8
          ? std::make_optional(make_fp32_descriptor(mixer_prefix + ".out_proj.input_scale", out_proj_input_scale, {1}))
          : std::nullopt;

  MambaLayerBindings bindings;
  bindings.input_norm_weight = &input_norm_descriptor;
  bindings.mixer_norm_weight = &mixer_norm_descriptor;
  bindings.in_proj_gemm_weight =
      in_proj_dense_weight_descriptor.has_value() ? &*in_proj_dense_weight_descriptor : nullptr;
  bindings.in_proj_kernel_weight =
      in_proj_fp8_weight_descriptor.has_value() ? &*in_proj_fp8_weight_descriptor : nullptr;
  bindings.in_proj_weight_scale =
      in_proj_weight_scale_descriptor.has_value() ? &*in_proj_weight_scale_descriptor : nullptr;
  bindings.in_proj_input_scale =
      in_proj_input_scale_descriptor.has_value() ? &*in_proj_input_scale_descriptor : nullptr;
  bindings.conv1d_weight = &conv1d_weight_descriptor;
  bindings.conv1d_bias = &conv1d_bias_descriptor;
  bindings.A_log = &A_log_descriptor;
  bindings.D = &D_descriptor;
  bindings.dt_bias = &dt_bias_descriptor;
  bindings.out_proj_gemm_weight =
      out_proj_dense_weight_descriptor.has_value() ? &*out_proj_dense_weight_descriptor : nullptr;
  bindings.out_proj_kernel_weight =
      out_proj_fp8_weight_descriptor.has_value() ? &*out_proj_fp8_weight_descriptor : nullptr;
  bindings.out_proj_weight_scale =
      out_proj_weight_scale_descriptor.has_value() ? &*out_proj_weight_scale_descriptor : nullptr;
  bindings.out_proj_input_scale =
      out_proj_input_scale_descriptor.has_value() ? &*out_proj_input_scale_descriptor : nullptr;

  MambaLayerConfig layer_config;
  layer_config.layer_index = metadata->layer_index;
  layer_config.hidden_size = metadata->hidden_size;
  layer_config.intermediate_size = metadata->intermediate_size;
  layer_config.num_heads = metadata->num_heads;
  layer_config.head_dim = metadata->head_dim;
  layer_config.state_size = metadata->state_size;
  layer_config.n_groups = metadata->n_groups;
  layer_config.conv_kernel_size = metadata->conv_kernel_size;
  layer_config.input_rms_epsilon = metadata->input_rms_epsilon;
  layer_config.mixer_rms_epsilon = metadata->mixer_rms_epsilon;
  layer_config.time_step_min = metadata->time_step_min;
  layer_config.conv_state_offset_elems = 0;
  layer_config.ssm_state_offset_elems = 0;

  auto slice = MambaLayerSlice::Create(layer_config, bindings);
  if (!expect(slice != nullptr && slice->valid(), "mamba slice should build for oracle fixture")) {
    return false;
  }

  RequestExecutionConfig request_config;
  request_config.hidden_size = metadata->hidden_size;
  request_config.max_tokens = input_rows;
  request_config.scratch_tokens = input_rows;
  request_config.mamba_hidden_size = metadata->hidden_size;
  request_config.mamba_projection_size =
      metadata->intermediate_size + conv_dim + metadata->num_heads;
  request_config.mamba_intermediate_size = metadata->intermediate_size;
  request_config.mamba_conv_state_bytes_fp32 = conv_state_elems * sizeof(float);
  request_config.mamba_state_bytes_fp32 = ssm_state_elems * sizeof(float);

  auto request = RequestExecutionContext::Create(request_config);
  auto input = DeviceTensorFp32::Create({input_rows, metadata->hidden_size});
  auto output = DeviceTensorFp32::Create({input_rows, metadata->hidden_size});
  if (!expect(request != nullptr && request->valid(), "request context should create for mamba oracle") ||
      !expect(input != nullptr && input->valid(), "input tensor should create for mamba oracle") ||
      !expect(output != nullptr && output->valid(), "output tensor should create for mamba oracle") ||
      !expect(input->CopyFromHost(input_hidden.data(), input_hidden.size()), "fixture input should upload") ||
      !expect(
          request->mamba_conv_state()->CopyFromHost(initial_conv_state.data(), initial_conv_state.size()),
          "initial conv state should upload") ||
      !expect(
          request->mamba_state()->CopyFromHost(initial_ssm_state.data(), initial_ssm_state.size()),
          "initial ssm state should upload")) {
    return false;
  }

  GemmHeuristicCache heuristic_cache;
  MambaLayerRunTrace trace;
  if (!expect(
          slice->Run(*cublas, &heuristic_cache, *request, *input, output.get(), &trace),
          "mamba oracle slice should execute")) {
    return false;
  }

  std::vector<float> actual_output(expected_final_output.size(), 0.0f);
  std::vector<float> actual_conv_state(expected_updated_conv_state.size(), 0.0f);
  std::vector<float> actual_ssm_state(expected_next_ssm_state.size(), 0.0f);
  if (!expect(output->CopyToHost(actual_output.data(), actual_output.size()), "final output should download") ||
      !expect(
          request->mamba_conv_state()->CopyToHost(actual_conv_state.data(), actual_conv_state.size()),
          "conv state should download") ||
      !expect(
          request->mamba_state()->CopyToHost(actual_ssm_state.data(), actual_ssm_state.size()),
          "ssm state should download")) {
    return false;
  }

  const float output_diff = max_abs_diff(actual_output, expected_final_output);
  const float conv_state_diff = max_abs_diff(actual_conv_state, expected_updated_conv_state);
  const float ssm_state_diff = max_abs_diff(actual_ssm_state, expected_next_ssm_state);
  const float norm_output_diff = max_abs_diff(trace.norm_output, expected_norm_output);
  const float in_proj_output_diff = max_abs_diff(trace.in_proj_output, expected_in_proj_output);
  std::optional<float> direct_in_proj_from_oracle_norm_diff;
  std::optional<float> direct_in_proj_from_runtime_norm_vs_trace_diff;
  std::optional<float> direct_in_proj_from_runtime_norm_vs_expected_diff;
  if (has_in_proj_fp8 &&
      !trace.norm_output.empty() &&
      !trace.in_proj_output.empty() &&
      in_proj_weight_scale.size() == 1 &&
      in_proj_input_scale.size() == 1) {
    ScaledFp8LinearConfig replay_config;
    replay_config.output_rows = in_proj_rows;
    replay_config.input_cols = metadata->hidden_size;
    replay_config.packed_weight_data = in_proj_weight_fp8.data();
    replay_config.packed_weight_nbytes = in_proj_weight_fp8.size();
    replay_config.tensor_name = mixer_prefix + ".in_proj.weight";
    replay_config.weight_scale = in_proj_weight_scale[0];
    replay_config.input_scale = in_proj_input_scale[0];
    auto replay_op = ScaledFp8LinearOp::Create(replay_config);
    auto replay_input = DeviceTensorFp32::Create({input_rows, metadata->hidden_size});
    auto replay_output = DeviceTensorFp32::Create({input_rows, in_proj_rows});
    if (expect(replay_op != nullptr && replay_op->valid(), "mamba in-proj replay op should build") &&
        expect(replay_input != nullptr && replay_input->valid(), "mamba in-proj replay input should create") &&
        expect(replay_output != nullptr && replay_output->valid(), "mamba in-proj replay output should create")) {
      std::vector<float> replay_host(expected_in_proj_output.size(), 0.0f);
      if (expect(
              replay_input->CopyFromHost(expected_norm_output.data(), expected_norm_output.size()),
              "mamba in-proj replay oracle norm should upload") &&
          expect(
              replay_op->Run(*cublas, &heuristic_cache, *replay_input, replay_output.get()),
              "mamba in-proj replay from oracle norm should execute") &&
          expect(
              replay_output->CopyToHost(replay_host.data(), replay_host.size()),
              "mamba in-proj replay from oracle norm should download")) {
        direct_in_proj_from_oracle_norm_diff = max_abs_diff(replay_host, expected_in_proj_output);
      }
      if (expect(
              replay_input->CopyFromHost(trace.norm_output.data(), trace.norm_output.size()),
              "mamba in-proj replay runtime norm should upload") &&
          expect(
              replay_op->Run(*cublas, &heuristic_cache, *replay_input, replay_output.get()),
              "mamba in-proj replay from runtime norm should execute") &&
          expect(
              replay_output->CopyToHost(replay_host.data(), replay_host.size()),
              "mamba in-proj replay from runtime norm should download")) {
        direct_in_proj_from_runtime_norm_vs_trace_diff = max_abs_diff(replay_host, trace.in_proj_output);
        direct_in_proj_from_runtime_norm_vs_expected_diff = max_abs_diff(replay_host, expected_in_proj_output);
      }
    }
  }
  if (!expect(output_diff <= 2.0e-3f, "final output should match oracle") ||
      !expect(conv_state_diff <= 2.0e-5f, "conv state should match oracle") ||
      !expect(ssm_state_diff <= 5.0e-5f, "ssm state should match oracle")) {
    std::cerr << "norm_output_diff=" << norm_output_diff
              << " in_proj_output_diff=" << in_proj_output_diff
              << " direct_in_proj_from_oracle_norm_diff="
              << direct_in_proj_from_oracle_norm_diff.value_or(-1.0f)
              << " direct_in_proj_from_runtime_norm_vs_trace_diff="
              << direct_in_proj_from_runtime_norm_vs_trace_diff.value_or(-1.0f)
              << " direct_in_proj_from_runtime_norm_vs_expected_diff="
              << direct_in_proj_from_runtime_norm_vs_expected_diff.value_or(-1.0f)
              << " output_diff=" << output_diff
              << " conv_state_diff=" << conv_state_diff
              << " ssm_state_diff=" << ssm_state_diff << "\n";
    return false;
  }

  constexpr std::size_t kBatchTokens = 3;
  std::vector<float> batch_input(kBatchTokens * metadata->hidden_size, 0.0f);
  for (std::size_t token = 0; token < kBatchTokens; ++token) {
    for (std::size_t i = 0; i < metadata->hidden_size; ++i) {
      batch_input[token * metadata->hidden_size + i] =
          input_hidden[i] + (0.001f * static_cast<float>(token)) - (0.00005f * static_cast<float>(i % 11));
    }
  }

  RequestExecutionConfig batch_request_config = request_config;
  batch_request_config.max_tokens = kBatchTokens;
  batch_request_config.scratch_tokens = kBatchTokens;
  auto batch_request = RequestExecutionContext::Create(batch_request_config);
  auto batch_input_tensor = DeviceTensorFp32::Create({kBatchTokens, metadata->hidden_size});
  auto batch_output_tensor = DeviceTensorFp32::Create({kBatchTokens, metadata->hidden_size});
  if (!expect(batch_request != nullptr && batch_request->valid(), "batched request context should create for mamba oracle") ||
      !expect(batch_input_tensor != nullptr && batch_input_tensor->valid(), "batched input tensor should create for mamba oracle") ||
      !expect(batch_output_tensor != nullptr && batch_output_tensor->valid(), "batched output tensor should create for mamba oracle") ||
      !expect(batch_input_tensor->CopyFromHost(batch_input.data(), batch_input.size()), "batched fixture input should upload") ||
      !expect(
          batch_request->mamba_conv_state()->CopyFromHost(initial_conv_state.data(), initial_conv_state.size()),
          "batched initial conv state should upload") ||
      !expect(
          batch_request->mamba_state()->CopyFromHost(initial_ssm_state.data(), initial_ssm_state.size()),
          "batched initial ssm state should upload")) {
    return false;
  }

  if (!expect(
          slice->Run(*cublas, &heuristic_cache, *batch_request, *batch_input_tensor, batch_output_tensor.get()),
          "batched mamba slice should execute")) {
    return false;
  }

  auto sequential_request = RequestExecutionContext::Create(request_config);
  auto sequential_input_tensor = DeviceTensorFp32::Create({1, metadata->hidden_size});
  auto sequential_output_tensor = DeviceTensorFp32::Create({1, metadata->hidden_size});
  if (!expect(sequential_request != nullptr && sequential_request->valid(), "sequential request context should create for mamba oracle") ||
      !expect(sequential_input_tensor != nullptr && sequential_input_tensor->valid(), "sequential input tensor should create for mamba oracle") ||
      !expect(sequential_output_tensor != nullptr && sequential_output_tensor->valid(), "sequential output tensor should create for mamba oracle") ||
      !expect(
          sequential_request->mamba_conv_state()->CopyFromHost(initial_conv_state.data(), initial_conv_state.size()),
          "sequential initial conv state should upload") ||
      !expect(
          sequential_request->mamba_state()->CopyFromHost(initial_ssm_state.data(), initial_ssm_state.size()),
          "sequential initial ssm state should upload")) {
    return false;
  }

  std::vector<float> sequential_outputs(batch_input.size(), 0.0f);
  std::vector<float> single_row(metadata->hidden_size, 0.0f);
  std::vector<float> single_output(metadata->hidden_size, 0.0f);
  for (std::size_t token = 0; token < kBatchTokens; ++token) {
    std::copy_n(
        batch_input.data() + (token * metadata->hidden_size),
        metadata->hidden_size,
        single_row.data());
    if (!expect(
            sequential_input_tensor->CopyFromHost(single_row.data(), single_row.size()),
            "sequential row input should upload") ||
        !expect(
            slice->Run(
                *cublas,
                &heuristic_cache,
                *sequential_request,
                *sequential_input_tensor,
                sequential_output_tensor.get()),
            "sequential mamba slice should execute")) {
      return false;
    }
    if (!expect(
            sequential_output_tensor->CopyToHost(single_output.data(), single_output.size()),
            "sequential row output should download")) {
      return false;
    }
    std::copy(
        single_output.begin(),
        single_output.end(),
        sequential_outputs.begin() + (token * metadata->hidden_size));
  }

  std::vector<float> batch_outputs(batch_input.size(), 0.0f);
  std::vector<float> batch_conv_state(initial_conv_state.size(), 0.0f);
  std::vector<float> batch_ssm_state(initial_ssm_state.size(), 0.0f);
  std::vector<float> sequential_conv_state(initial_conv_state.size(), 0.0f);
  std::vector<float> sequential_ssm_state(initial_ssm_state.size(), 0.0f);
  if (!expect(
          batch_output_tensor->CopyToHost(batch_outputs.data(), batch_outputs.size()),
          "batched mamba output should download") ||
      !expect(
          batch_request->mamba_conv_state()->CopyToHost(batch_conv_state.data(), batch_conv_state.size()),
          "batched conv state should download") ||
      !expect(
          batch_request->mamba_state()->CopyToHost(batch_ssm_state.data(), batch_ssm_state.size()),
          "batched ssm state should download") ||
      !expect(
          sequential_request->mamba_conv_state()->CopyToHost(
              sequential_conv_state.data(),
              sequential_conv_state.size()),
          "sequential conv state should download") ||
      !expect(
          sequential_request->mamba_state()->CopyToHost(
              sequential_ssm_state.data(),
              sequential_ssm_state.size()),
          "sequential ssm state should download")) {
    return false;
  }

  if (!expect(
          max_abs_diff(batch_outputs, sequential_outputs) <= 2.0e-3f,
          "batched Mamba outputs should match repeated single-row execution") ||
      !expect(
          max_abs_diff(batch_conv_state, sequential_conv_state) <= 1.0e-5f,
          "batched Mamba conv state should match repeated single-row execution") ||
      !expect(
          max_abs_diff(batch_ssm_state, sequential_ssm_state) <= 5.0e-5f,
          "batched Mamba SSM state should match repeated single-row execution")) {
    return false;
  }

  return true;
}

}  // namespace

int main() {
  if (env_var_enabled("NEMOTRON_MAMBA_LAYER_VLLM_MICROREPRO")) {
    return run_mamba_layer_vllm_microrepro() ? 0 : 1;
  }
  return run_mamba_layer_fixture() ? 0 : 1;
}
