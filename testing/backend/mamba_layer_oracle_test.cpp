#include "nemotron/mamba_layer.h"

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

  if (!expect(input_hidden.size() == metadata->hidden_size, "input hidden size should match metadata") ||
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
      !expect(expected_norm_output.size() == metadata->hidden_size, "expected norm output should match metadata") ||
      !expect(expected_in_proj_output.size() == in_proj_rows, "expected in_proj output should match metadata") ||
      !expect(expected_final_output.size() == metadata->hidden_size, "expected final output should match metadata")) {
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
  request_config.max_tokens = 1;
  request_config.scratch_tokens = 1;
  request_config.mamba_conv_state_bytes_fp32 = conv_state_elems * sizeof(float);
  request_config.mamba_state_bytes_fp32 = ssm_state_elems * sizeof(float);

  auto request = RequestExecutionContext::Create(request_config);
  auto input = DeviceTensorFp32::Create({1, metadata->hidden_size});
  auto output = DeviceTensorFp32::Create({1, metadata->hidden_size});
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
  if (!expect(output_diff <= 2.0e-3f, "final output should match oracle") ||
      !expect(conv_state_diff <= 2.0e-5f, "conv state should match oracle") ||
      !expect(ssm_state_diff <= 5.0e-5f, "ssm state should match oracle")) {
    std::cerr << "norm_output_diff=" << norm_output_diff
              << " in_proj_output_diff=" << in_proj_output_diff
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
  return run_mamba_layer_fixture() ? 0 : 1;
}
