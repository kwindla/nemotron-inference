#include "nemotron/device_tensor.h"
#include "nemotron/attention_layer.h"
#include "nemotron/cublaslt_handle.h"
#include "nemotron/linear_op.h"
#include "nemotron/manifest.h"
#include "nemotron/primitive_ops.h"
#include "nemotron/request_context.h"
#include "nemotron/runtime_environment.h"
#include "nemotron/single_token_forward_model.h"
#include "nemotron/state_snapshot.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <vector>

namespace {

using nemotron::DeviceTensorBf16;
using nemotron::DeviceTensorFp32;
using nemotron::ForwardLayerKind;
using nemotron::RequestExecutionContext;
using nemotron::RuntimeBootstrapOptions;
using nemotron::RuntimeEnvironment;
using nemotron::SingleTokenForwardConfig;
using nemotron::SingleTokenForwardModel;

constexpr const char* kNanoModelId = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4";
constexpr float kHiddenTolerance = 1.0e-5f;
constexpr float kMambaStateTolerance = 1.0e-6f;
constexpr std::size_t kStableProjectionRows = 32;

class ScopedEnvVar {
 public:
  explicit ScopedEnvVar(const char* name) : name_(name) {
    const char* current = std::getenv(name_);
    if (current != nullptr) {
      had_original_ = true;
      original_value_ = current;
    }
  }

  ~ScopedEnvVar() {
    if (had_original_) {
      setenv(name_, original_value_.c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }

 private:
  const char* name_;
  bool had_original_ = false;
  std::string original_value_;
};

constexpr std::size_t GiB(std::size_t value) {
  return value * 1024ull * 1024ull * 1024ull;
}

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

std::optional<std::size_t> env_size_t(const char* env_var) {
  const char* value = std::getenv(env_var);
  if (value == nullptr || value[0] == '\0') {
    return std::nullopt;
  }
  try {
    return static_cast<std::size_t>(std::stoull(value));
  } catch (...) {
    return std::nullopt;
  }
}

bool all_finite(const std::vector<float>& values) {
  return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

float max_abs_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size() || !all_finite(lhs) || !all_finite(rhs)) {
    return std::numeric_limits<float>::infinity();
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
  }
  return max_diff;
}

std::vector<float> slice_row(
    const std::vector<float>& values,
    std::size_t row_index,
    std::size_t row_width) {
  if (row_width == 0) {
    return {};
  }
  const std::size_t row_offset = row_index * row_width;
  if (row_offset + row_width > values.size()) {
    return {};
  }
  return std::vector<float>(
      values.begin() + static_cast<std::ptrdiff_t>(row_offset),
      values.begin() + static_cast<std::ptrdiff_t>(row_offset + row_width));
}

std::vector<float> copy_tensor_to_host(const DeviceTensorFp32& tensor) {
  std::vector<float> host(tensor.numel(), 0.0f);
  if (!tensor.CopyToHost(host.data(), host.size())) {
    return {};
  }
  return host;
}

std::vector<__nv_bfloat16> copy_tensor_to_host(const DeviceTensorBf16& tensor) {
  std::vector<__nv_bfloat16> host(tensor.numel(), __float2bfloat16(0.0f));
  if (!tensor.CopyToHost(host.data(), host.size())) {
    return {};
  }
  return host;
}

std::uint16_t bf16_bits(__nv_bfloat16 value) {
  std::uint16_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

__nv_bfloat16 bf16_from_bits(std::uint16_t bits) {
  __nv_bfloat16 value = __float2bfloat16(0.0f);
  std::memcpy(&value, &bits, sizeof(bits));
  return value;
}

std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream input(path);
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::optional<std::vector<std::int32_t>> parse_json_int32_array(const std::string& json) {
  std::vector<std::int32_t> values;
  const std::regex number_pattern("(-?[0-9]+)");
  for (auto it = std::sregex_iterator(json.begin(), json.end(), number_pattern);
       it != std::sregex_iterator();
       ++it) {
    try {
      const long long value = std::stoll((*it)[1].str());
      if (value < std::numeric_limits<std::int32_t>::min() ||
          value > std::numeric_limits<std::int32_t>::max()) {
        return std::nullopt;
      }
      values.push_back(static_cast<std::int32_t>(value));
    } catch (...) {
      return std::nullopt;
    }
  }
  if (values.empty()) {
    return std::nullopt;
  }
  return values;
}

std::filesystem::path source_root() {
  return std::filesystem::path(NEMOTRON_SOURCE_ROOT);
}

std::filesystem::path resolve_manifest_path() {
  const char* manifest_env = std::getenv("NEMOTRON_FORWARD_MANIFEST");
  if (manifest_env != nullptr && manifest_env[0] != '\0') {
    return std::filesystem::path(manifest_env);
  }
  return source_root() / "artifacts" / "manifests" /
         "forward_runtime_manifest_nano_rtx5090_unverified.json";
}

std::optional<std::vector<std::int32_t>> load_short_chat_prompt_token_ids() {
  const std::filesystem::path prompt_path =
      source_root() / "testing" / "oracle" / "full_model_single_token_short_chat_cuda_v3" /
      "prompt_token_ids.json";
  if (!std::filesystem::exists(prompt_path)) {
    return std::nullopt;
  }
  return parse_json_int32_array(read_text_file(prompt_path));
}

RuntimeBootstrapOptions make_options(std::size_t target_context_tokens) {
  RuntimeBootstrapOptions options;
  options.service_target.total_memory_bytes = GiB(128);
  options.service_target.weights_bytes = GiB(100);
  options.service_target.workspace_bytes = GiB(8);
  options.service_target.graph_bytes = GiB(4);
  options.service_target.safety_headroom_bytes = GiB(4);
  options.service_target.target_active_requests = 1;
  options.service_target.target_context_tokens = target_context_tokens;
  options.use_fp16_mamba_state = false;
  options.reusable_node_metadata_bytes = 4096;
  options.verify_manifest_files = false;
  options.materialize_weight_arena = false;
  return options;
}

std::unique_ptr<DeviceTensorFp32> make_boundary_logits_buffer(const SingleTokenForwardConfig& config) {
  return DeviceTensorFp32::Create({1, config.vocab_size});
}

struct AttentionWindow {
  std::size_t pre_attention_layer_index = 0;
  std::size_t attention_layer_index = 0;
};

std::optional<AttentionWindow> find_first_attention_window(const SingleTokenForwardModel& model) {
  const auto& layers = model.plan().layers;
  for (std::size_t i = 0; i < layers.size(); ++i) {
    if (layers[i].kind != ForwardLayerKind::kAttention) {
      continue;
    }
    if (i == 0) {
      return std::nullopt;
    }
    return AttentionWindow{layers[i - 1].layer_index, layers[i].layer_index};
  }
  return std::nullopt;
}

struct LayerKvRangeComparison {
  bool valid = false;
  bool key_match = false;
  bool value_match = false;
  std::optional<std::size_t> first_key_token;
  std::optional<std::size_t> first_key_elem_in_token;
  std::optional<float> first_key_lhs_value;
  std::optional<float> first_key_rhs_value;
  std::optional<std::size_t> first_value_token;
  std::optional<std::size_t> first_value_elem_in_token;
  std::optional<float> first_value_lhs_value;
  std::optional<float> first_value_rhs_value;
};

struct Bf16RowRangeComparison {
  bool valid = false;
  bool match = false;
  std::optional<std::size_t> first_lhs_row;
  std::optional<std::size_t> first_rhs_row;
  std::optional<std::size_t> first_elem_in_row;
  std::optional<float> first_lhs_value;
  std::optional<float> first_rhs_value;
};

struct AttentionProjectionHarness {
  std::unique_ptr<nemotron::CublasLtHandle> cublas;
  nemotron::GemmHeuristicCache* heuristic_cache = nullptr;
  std::unique_ptr<DeviceTensorFp32> norm_weight;
  std::unique_ptr<nemotron::UploadedLinearOp> k_proj;
  std::unique_ptr<nemotron::UploadedLinearOp> v_proj;
};

struct AttentionProjectionOutputs {
  std::vector<__nv_bfloat16> normalized;
  std::vector<__nv_bfloat16> k;
  std::vector<__nv_bfloat16> v;
};

std::vector<__nv_bfloat16> bf16_vector_from_bits(const std::vector<std::uint16_t>& bits) {
  std::vector<__nv_bfloat16> values(bits.size(), __float2bfloat16(0.0f));
  for (std::size_t i = 0; i < bits.size(); ++i) {
    values[i] = bf16_from_bits(bits[i]);
  }
  return values;
}

std::unique_ptr<DeviceTensorBf16> make_device_bf16_tensor_from_bits(
    std::vector<std::size_t> shape,
    const std::vector<std::uint16_t>& bits) {
  auto tensor = DeviceTensorBf16::Create(std::move(shape));
  if (tensor == nullptr || !tensor->valid() || tensor->numel() != bits.size()) {
    return nullptr;
  }
  const auto host_values = bf16_vector_from_bits(bits);
  if (!tensor->CopyFromHost(host_values.data(), host_values.size())) {
    return nullptr;
  }
  return tensor;
}

std::optional<Bf16RowRangeComparison> compare_bf16_row_range(
    const std::vector<__nv_bfloat16>& lhs,
    const std::vector<__nv_bfloat16>& rhs,
    std::size_t row_width,
    std::size_t lhs_row_start,
    std::size_t rhs_row_start,
    std::size_t row_count) {
  Bf16RowRangeComparison comparison;
  if (row_width == 0 ||
      lhs_row_start + row_count > lhs.size() / row_width ||
      rhs_row_start + row_count > rhs.size() / row_width) {
    return std::nullopt;
  }
  comparison.valid = true;
  comparison.match = true;
  for (std::size_t row = 0; row < row_count; ++row) {
    const std::size_t lhs_offset = (lhs_row_start + row) * row_width;
    const std::size_t rhs_offset = (rhs_row_start + row) * row_width;
    for (std::size_t elem = 0; elem < row_width; ++elem) {
      if (bf16_bits(lhs[lhs_offset + elem]) != bf16_bits(rhs[rhs_offset + elem])) {
        comparison.match = false;
        comparison.first_lhs_row = lhs_row_start + row;
        comparison.first_rhs_row = rhs_row_start + row;
        comparison.first_elem_in_row = elem;
        comparison.first_lhs_value = __bfloat162float(lhs[lhs_offset + elem]);
        comparison.first_rhs_value = __bfloat162float(rhs[rhs_offset + elem]);
        return comparison;
      }
    }
  }
  return comparison;
}

std::string describe_bf16_row_range_comparison(const Bf16RowRangeComparison& comparison) {
  if (!comparison.valid) {
    return "invalid";
  }
  std::string text = "match=" + std::string(comparison.match ? "true" : "false");
  if (comparison.first_lhs_row.has_value()) {
    text += " lhs_row=" + std::to_string(*comparison.first_lhs_row) +
            " rhs_row=" + std::to_string(*comparison.first_rhs_row) +
            " elem=" + std::to_string(*comparison.first_elem_in_row) +
            " lhs=" + std::to_string(*comparison.first_lhs_value) +
            " rhs=" + std::to_string(*comparison.first_rhs_value);
  }
  return text;
}

std::optional<AttentionProjectionHarness> build_attention_projection_harness(
    nemotron::RuntimeEnvironment& environment,
    const SingleTokenForwardConfig& config,
    std::size_t attention_layer_index) {
  if (!environment.has_model_schedule() ||
      !environment.has_kernel_catalog() ||
      !environment.has_gemm_catalog()) {
    return std::nullopt;
  }
  const auto* layer = environment.model_schedule()->FindLayer(attention_layer_index);
  if (layer == nullptr) {
    return std::nullopt;
  }
  const auto bindings = nemotron::BuildAttentionLayerBindings(
      *layer,
      *environment.kernel_catalog(),
      *environment.gemm_catalog());
  if (!bindings.has_value()) {
    return std::nullopt;
  }

  AttentionProjectionHarness harness;
  harness.cublas = nemotron::CublasLtHandle::Create();
  harness.heuristic_cache = environment.gemm_heuristic_cache();
  harness.norm_weight = nemotron::UploadVectorWeightToDeviceFp32(*bindings->norm_weight);
  harness.k_proj = nemotron::UploadedLinearOp::Create(*bindings->k_proj);
  harness.v_proj = nemotron::UploadedLinearOp::Create(*bindings->v_proj);
  if (harness.cublas == nullptr || !harness.cublas->valid() ||
      harness.norm_weight == nullptr || !harness.norm_weight->valid() ||
      harness.k_proj == nullptr || !harness.k_proj->valid() ||
      harness.v_proj == nullptr || !harness.v_proj->valid()) {
    return std::nullopt;
  }
  if (harness.k_proj->kernel_family() != nemotron::GemmKernelFamily::kDenseRowMajor ||
      harness.v_proj->kernel_family() != nemotron::GemmKernelFamily::kDenseRowMajor) {
    return std::nullopt;
  }
  if (harness.k_proj->input_cols() != config.hidden_size ||
      harness.v_proj->input_cols() != config.hidden_size) {
    return std::nullopt;
  }
  return harness;
}

std::optional<AttentionProjectionOutputs> run_attention_kv_projections(
    const AttentionProjectionHarness& harness,
    const SingleTokenForwardConfig& config,
    const nemotron::CapturedLayerOutput& captured,
    std::size_t token_count) {
  if (captured.hidden_delta_bf16_bits.size() != token_count * config.hidden_size ||
      captured.residual_accum_bf16_bits.size() != token_count * config.hidden_size) {
    return std::nullopt;
  }
  auto hidden_input = make_device_bf16_tensor_from_bits(
      {token_count, config.hidden_size},
      captured.hidden_delta_bf16_bits);
  auto residual = make_device_bf16_tensor_from_bits(
      {token_count, config.hidden_size},
      captured.residual_accum_bf16_bits);
  auto normalized = DeviceTensorBf16::Create({token_count, config.hidden_size});
  const std::size_t kv_width = config.attention_kv_head_count * config.attention_head_dim;
  auto k = DeviceTensorBf16::Create({token_count, kv_width});
  auto v = DeviceTensorBf16::Create({token_count, kv_width});
  if (hidden_input == nullptr || !hidden_input->valid() ||
      residual == nullptr || !residual->valid() ||
      normalized == nullptr || !normalized->valid() ||
      k == nullptr || !k->valid() ||
      v == nullptr || !v->valid()) {
    return std::nullopt;
  }

  const bool ok =
      nemotron::FusedAddRmsNormBf16(
          *hidden_input,
          residual.get(),
          *harness.norm_weight,
          config.layer_norm_epsilon,
          normalized.get()) &&
      nemotron::RunStableRowBf16Linear(
          *harness.k_proj,
          *harness.cublas,
          harness.heuristic_cache,
          *normalized,
          k.get(),
          kStableProjectionRows) &&
      nemotron::RunStableRowBf16Linear(
          *harness.v_proj,
          *harness.cublas,
          harness.heuristic_cache,
          *normalized,
          v.get(),
          kStableProjectionRows) &&
      cudaStreamSynchronize(nullptr) == cudaSuccess;
  if (!ok) {
    return std::nullopt;
  }

  AttentionProjectionOutputs outputs;
  outputs.normalized = copy_tensor_to_host(*normalized);
  outputs.k = copy_tensor_to_host(*k);
  outputs.v = copy_tensor_to_host(*v);
  if (outputs.normalized.empty() || outputs.k.empty() || outputs.v.empty()) {
    return std::nullopt;
  }
  return outputs;
}

nemotron::CapturedLayerOutput zero_pad_captured_layer_output(
    const nemotron::CapturedLayerOutput& captured,
    std::size_t source_token_count,
    std::size_t padded_token_count,
    std::size_t hidden_size) {
  nemotron::CapturedLayerOutput padded = captured;
  const std::size_t source_elems = source_token_count * hidden_size;
  const std::size_t padded_elems = padded_token_count * hidden_size;
  padded.hidden.resize(padded_elems, 0.0f);
  padded.hidden_delta_bf16_bits.resize(padded_elems, 0);
  padded.residual_accum_bf16_bits.resize(padded_elems, 0);
  if (captured.hidden.size() > source_elems) {
    padded.hidden.assign(captured.hidden.begin(), captured.hidden.begin() + source_elems);
    padded.hidden.resize(padded_elems, 0.0f);
  }
  if (captured.hidden_delta_bf16_bits.size() > source_elems) {
    padded.hidden_delta_bf16_bits.assign(
        captured.hidden_delta_bf16_bits.begin(),
        captured.hidden_delta_bf16_bits.begin() + static_cast<std::ptrdiff_t>(source_elems));
    padded.hidden_delta_bf16_bits.resize(padded_elems, 0);
  }
  if (captured.residual_accum_bf16_bits.size() > source_elems) {
    padded.residual_accum_bf16_bits.assign(
        captured.residual_accum_bf16_bits.begin(),
        captured.residual_accum_bf16_bits.begin() + static_cast<std::ptrdiff_t>(source_elems));
    padded.residual_accum_bf16_bits.resize(padded_elems, 0);
  }
  return padded;
}

std::optional<LayerKvRangeComparison> compare_layer_kv_token_range(
    const RequestExecutionContext& lhs,
    const RequestExecutionContext& rhs,
    std::size_t layer_index,
    std::size_t token_start,
    std::size_t token_count) {
  if (!lhs.valid() || !rhs.valid() ||
      lhs.key_cache() == nullptr || rhs.key_cache() == nullptr ||
      lhs.value_cache() == nullptr || rhs.value_cache() == nullptr ||
      !lhs.key_cache()->valid() || !rhs.key_cache()->valid() ||
      !lhs.value_cache()->valid() || !rhs.value_cache()->valid()) {
    return std::nullopt;
  }
  if (layer_index >= lhs.config().attention_kv_cache.layer_count ||
      layer_index >= rhs.config().attention_kv_cache.layer_count) {
    return std::nullopt;
  }
  if (lhs.sequence_length() < token_start + token_count ||
      rhs.sequence_length() < token_start + token_count) {
    return std::nullopt;
  }

  const std::size_t lhs_total_pages =
      lhs.key_cache()->shape().empty() ? 0 : lhs.key_cache()->shape().front();
  const std::size_t rhs_total_pages =
      rhs.key_cache()->shape().empty() ? 0 : rhs.key_cache()->shape().front();
  const std::size_t tokens_per_page = lhs.config().attention_kv_cache.tokens_per_page;
  if (lhs_total_pages == 0 || rhs_total_pages == 0 || tokens_per_page == 0 ||
      lhs.key_cache()->numel() % lhs_total_pages != 0 ||
      rhs.key_cache()->numel() % rhs_total_pages != 0) {
    return std::nullopt;
  }

  const std::size_t lhs_page_elems = lhs.key_cache()->numel() / lhs_total_pages;
  const std::size_t rhs_page_elems = rhs.key_cache()->numel() / rhs_total_pages;
  if (lhs_page_elems != rhs_page_elems || lhs_page_elems % tokens_per_page != 0) {
    return std::nullopt;
  }
  const std::size_t token_elems = lhs_page_elems / tokens_per_page;
  if (token_elems == 0) {
    return std::nullopt;
  }

  const auto* lhs_pages = lhs.kv_pages(layer_index);
  const auto* rhs_pages = rhs.kv_pages(layer_index);
  if (lhs_pages == nullptr || rhs_pages == nullptr) {
    return std::nullopt;
  }

  const std::vector<__nv_bfloat16> lhs_key = copy_tensor_to_host(*lhs.key_cache());
  const std::vector<__nv_bfloat16> rhs_key = copy_tensor_to_host(*rhs.key_cache());
  const std::vector<__nv_bfloat16> lhs_value = copy_tensor_to_host(*lhs.value_cache());
  const std::vector<__nv_bfloat16> rhs_value = copy_tensor_to_host(*rhs.value_cache());
  if (lhs_key.empty() || rhs_key.empty() || lhs_value.empty() || rhs_value.empty()) {
    return std::nullopt;
  }

  LayerKvRangeComparison comparison;
  comparison.valid = true;
  comparison.key_match = true;
  comparison.value_match = true;
  for (std::size_t token_index = token_start; token_index < token_start + token_count; ++token_index) {
    const std::size_t page_index = token_index / tokens_per_page;
    const std::size_t token_in_page = token_index % tokens_per_page;
    if (page_index >= lhs_pages->size() || page_index >= rhs_pages->size()) {
      return std::nullopt;
    }
    const std::size_t lhs_token_offset =
        ((*lhs_pages)[page_index].page_id * lhs_page_elems) + (token_in_page * token_elems);
    const std::size_t rhs_token_offset =
        ((*rhs_pages)[page_index].page_id * rhs_page_elems) + (token_in_page * token_elems);
    for (std::size_t elem = 0; elem < token_elems; ++elem) {
      if (comparison.key_match &&
          bf16_bits(lhs_key[lhs_token_offset + elem]) != bf16_bits(rhs_key[rhs_token_offset + elem])) {
        comparison.key_match = false;
        comparison.first_key_token = token_index;
        comparison.first_key_elem_in_token = elem;
        comparison.first_key_lhs_value = __bfloat162float(lhs_key[lhs_token_offset + elem]);
        comparison.first_key_rhs_value = __bfloat162float(rhs_key[rhs_token_offset + elem]);
      }
      if (comparison.value_match &&
          bf16_bits(lhs_value[lhs_token_offset + elem]) != bf16_bits(rhs_value[rhs_token_offset + elem])) {
        comparison.value_match = false;
        comparison.first_value_token = token_index;
        comparison.first_value_elem_in_token = elem;
        comparison.first_value_lhs_value = __bfloat162float(lhs_value[lhs_token_offset + elem]);
        comparison.first_value_rhs_value = __bfloat162float(rhs_value[rhs_token_offset + elem]);
      }
      if (!comparison.key_match && !comparison.value_match) {
        break;
      }
    }
    if (!comparison.key_match && !comparison.value_match) {
      break;
    }
  }
  return comparison;
}

std::string describe_layer_kv_range_comparison(const LayerKvRangeComparison& comparison) {
  if (!comparison.valid) {
    return "invalid";
  }
  std::string text = "key_match=" + std::string(comparison.key_match ? "true" : "false") +
                     " value_match=" + std::string(comparison.value_match ? "true" : "false");
  if (comparison.first_key_token.has_value()) {
    text += " first_key_token=" + std::to_string(*comparison.first_key_token) +
            " first_key_elem=" + std::to_string(*comparison.first_key_elem_in_token) +
            " first_key_lhs=" + std::to_string(*comparison.first_key_lhs_value) +
            " first_key_rhs=" + std::to_string(*comparison.first_key_rhs_value);
  }
  if (comparison.first_value_token.has_value()) {
    text += " first_value_token=" + std::to_string(*comparison.first_value_token) +
            " first_value_elem=" + std::to_string(*comparison.first_value_elem_in_token) +
            " first_value_lhs=" + std::to_string(*comparison.first_value_lhs_value) +
            " first_value_rhs=" + std::to_string(*comparison.first_value_rhs_value);
  }
  return text;
}

bool expect_hidden_row_match(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    float tolerance,
    const std::string& label) {
  const float diff = max_abs_diff(lhs, rhs);
  std::cerr << "split_prefill_localization_test: " << label
            << " max_abs_diff=" << diff << "\n";
  return expect(
      diff <= tolerance,
      label + " should match within tolerance; max_abs_diff=" + std::to_string(diff));
}

bool expect_layer_kv_match(
    const std::optional<LayerKvRangeComparison>& comparison,
    const std::string& label) {
  if (!expect(comparison.has_value(), label + " comparison should succeed")) {
    return false;
  }
  std::cerr << "split_prefill_localization_test: " << label
            << " " << describe_layer_kv_range_comparison(*comparison) << "\n";
  return expect(
      comparison->key_match && comparison->value_match,
      label + " should match exactly: " + describe_layer_kv_range_comparison(*comparison));
}

bool expect_bf16_row_range_match(
    const std::optional<Bf16RowRangeComparison>& comparison,
    const std::string& label) {
  if (!expect(comparison.has_value(), label + " comparison should succeed")) {
    return false;
  }
  std::cerr << "split_prefill_localization_test: " << label
            << " " << describe_bf16_row_range_comparison(*comparison) << "\n";
  return expect(
      comparison->match,
      label + " should match exactly: " + describe_bf16_row_range_comparison(*comparison));
}

bool run_split_prefill_localization_test() {
  ScopedEnvVar prefix_cache_env("NEMOTRON_PREFIX_CACHE");
  ScopedEnvVar vram_reserve_env("NEMOTRON_FORWARD_VRAM_RESERVE_MB");
  ScopedEnvVar expert_monolithic_env("NEMOTRON_EXPERT_MONOLITHIC");
  ScopedEnvVar expert_residency_budget_env("NEMOTRON_EXPERT_RESIDENCY_BUDGET_MB");
  unsetenv("NEMOTRON_PREFIX_CACHE");
  unsetenv("NEMOTRON_EXPERT_RESIDENCY_BUDGET_MB");
  setenv("NEMOTRON_FORWARD_VRAM_RESERVE_MB", "0", 1);
  setenv("NEMOTRON_EXPERT_MONOLITHIC", "0", 1);

  if (!has_cuda_device()) {
    std::cout << "split_prefill_localization_test: SKIP (no CUDA device available)\n";
    return true;
  }

  const std::filesystem::path manifest_path = resolve_manifest_path();
  if (!std::filesystem::exists(manifest_path)) {
    std::cout << "split_prefill_localization_test: SKIP (manifest path does not exist)\n";
    return true;
  }

  const auto load_result = nemotron::LoadManifestFromJsonFile(manifest_path);
  if (!expect(load_result.ok, "Nano manifest JSON should parse")) {
    return false;
  }
  if (load_result.manifest.runtime.model_id != kNanoModelId) {
    std::cout << "split_prefill_localization_test: SKIP (manifest model is not Nano)\n";
    return true;
  }

  const auto prompt_token_ids = load_short_chat_prompt_token_ids();
  const std::size_t prompt_offset = env_size_t("NEMOTRON_MULTI_TURN_PROMPT_OFFSET").value_or(0);
  const std::size_t prompt_token_count = env_size_t("NEMOTRON_MULTI_TURN_PROMPT_TOKENS").value_or(4);
  const std::size_t prefix_token_count =
      env_size_t("NEMOTRON_MULTI_TURN_GLOBAL_ROOT_TOKENS").value_or(2);
  if (!expect(prompt_token_ids.has_value(), "short_chat prompt token ids should load") ||
      !expect(
          prompt_token_count > 0 && prompt_offset + prompt_token_count <= prompt_token_ids->size(),
          "short_chat prompt should have enough tokens for the configured regression window") ||
      !expect(
          prefix_token_count > 0 && prefix_token_count < prompt_token_count,
          "split-prefill prefix must be shorter than the configured prompt window")) {
    return false;
  }

  const std::vector<std::int32_t> prompt(
      prompt_token_ids->begin() + static_cast<std::ptrdiff_t>(prompt_offset),
      prompt_token_ids->begin() + static_cast<std::ptrdiff_t>(prompt_offset + prompt_token_count));
  const std::size_t suffix_token_count = prompt.size() - prefix_token_count;

  const auto environment = RuntimeEnvironment::BuildFromManifestFile(
      manifest_path,
      make_options(prompt.size()));
  if (!expect(static_cast<bool>(environment), "runtime environment should build from the Nano manifest")) {
    return false;
  }

  SingleTokenForwardConfig config = nemotron::KnownNemotron3Nano30BA3BConfig();
  config.max_tokens = std::max<std::size_t>(config.max_tokens, prompt.size());

  auto model = SingleTokenForwardModel::Create(*environment, config);
  if (!expect(model != nullptr && model->valid(), "forward model should build from the Nano runtime environment")) {
    return false;
  }

  const auto attention_window = find_first_attention_window(*model);
  if (!expect(attention_window.has_value(), "model should expose a first attention layer")) {
    return false;
  }
  std::cerr << "split_prefill_localization_test: pre_attention_layer="
            << attention_window->pre_attention_layer_index
            << " attention_layer=" << attention_window->attention_layer_index << "\n";

  bool ok = true;

  // Compare the prefix-token hidden rows immediately before the first
  // attention layer. If these diverge, the bug is upstream of attention and
  // restore.
  auto full_pre_context = model->CreateRequestContext();
  auto prefix_pre_context = model->CreateRequestContext();
  auto full_pre_logits = make_boundary_logits_buffer(config);
  auto prefix_pre_logits = make_boundary_logits_buffer(config);
  nemotron::SingleTokenForwardTrace full_pre_trace;
  nemotron::SingleTokenForwardTrace prefix_pre_trace;
  ok = expect(
           full_pre_context != nullptr && full_pre_context->valid() &&
               prefix_pre_context != nullptr && prefix_pre_context->valid() &&
               full_pre_logits != nullptr && full_pre_logits->valid() &&
               prefix_pre_logits != nullptr && prefix_pre_logits->valid(),
           "pre-attention trace contexts should allocate") &&
       ok;
  if (ok) {
    ok = expect(
             model->RunPrefill(
                 prompt.data(),
                 prompt.size(),
                 *full_pre_context,
                 full_pre_logits.get(),
                 {attention_window->pre_attention_layer_index},
                 &full_pre_trace,
                 attention_window->pre_attention_layer_index),
             "full prompt pre-attention trace should succeed") &&
         ok;
    ok = expect(
             model->RunPrefill(
                 prompt.data(),
                 prefix_token_count,
                 *prefix_pre_context,
                 prefix_pre_logits.get(),
                 {attention_window->pre_attention_layer_index},
                 &prefix_pre_trace,
                 attention_window->pre_attention_layer_index),
             "prefix-only pre-attention trace should succeed") &&
         ok;
  }
  if (ok) {
    ok = expect(
             full_pre_trace.captured_layers.size() == 1 &&
                 prefix_pre_trace.captured_layers.size() == 1,
             "pre-attention traces should capture exactly one layer") &&
         ok;
  }
  if (ok) {
    for (std::size_t row_index = 0; row_index < prefix_token_count; ++row_index) {
      const std::vector<float> full_row = slice_row(
          full_pre_trace.captured_layers.front().hidden,
          row_index,
          config.hidden_size);
      const std::vector<float> prefix_row = slice_row(
          prefix_pre_trace.captured_layers.front().hidden,
          row_index,
          config.hidden_size);
      ok = expect_hidden_row_match(
               full_row,
               prefix_row,
               kHiddenTolerance,
               "pre-attention prefix row " + std::to_string(row_index)) &&
           ok;
    }
  }
  if (ok) {
    const auto attention_projection_harness = build_attention_projection_harness(
        *environment,
        config,
        attention_window->attention_layer_index);
    ok = expect(
             attention_projection_harness.has_value(),
             "attention projection harness should build") &&
         ok;
    if (ok) {
      const auto full_projections = run_attention_kv_projections(
          *attention_projection_harness,
          config,
          full_pre_trace.captured_layers.front(),
          prompt.size());
      const auto prefix_projections = run_attention_kv_projections(
          *attention_projection_harness,
          config,
          prefix_pre_trace.captured_layers.front(),
          prefix_token_count);
      const auto padded_prefix_capture = zero_pad_captured_layer_output(
          prefix_pre_trace.captured_layers.front(),
          prefix_token_count,
          prompt.size(),
          config.hidden_size);
      const auto padded_prefix_projections = run_attention_kv_projections(
          *attention_projection_harness,
          config,
          padded_prefix_capture,
          prompt.size());
      ok = expect(full_projections.has_value(), "full pre-attention projections should succeed") && ok;
      ok = expect(prefix_projections.has_value(), "prefix pre-attention projections should succeed") && ok;
      ok = expect(
               padded_prefix_projections.has_value(),
               "padded prefix pre-attention projections should succeed") &&
           ok;
      if (ok) {
        const auto norm_compare = compare_bf16_row_range(
            full_projections->normalized,
            prefix_projections->normalized,
            config.hidden_size,
            0,
            0,
            prefix_token_count);
        const auto k_compare = compare_bf16_row_range(
            full_projections->k,
            prefix_projections->k,
            config.attention_kv_head_count * config.attention_head_dim,
            0,
            0,
            prefix_token_count);
        const auto v_compare = compare_bf16_row_range(
            full_projections->v,
            prefix_projections->v,
            config.attention_kv_head_count * config.attention_head_dim,
            0,
            0,
            prefix_token_count);
        const auto padded_norm_compare = compare_bf16_row_range(
            full_projections->normalized,
            padded_prefix_projections->normalized,
            config.hidden_size,
            0,
            0,
            prefix_token_count);
        const auto padded_k_compare = compare_bf16_row_range(
            full_projections->k,
            padded_prefix_projections->k,
            config.attention_kv_head_count * config.attention_head_dim,
            0,
            0,
            prefix_token_count);
        const auto padded_v_compare = compare_bf16_row_range(
            full_projections->v,
            padded_prefix_projections->v,
            config.attention_kv_head_count * config.attention_head_dim,
            0,
            0,
            prefix_token_count);
        const bool norm_match_ok =
            expect_bf16_row_range_match(norm_compare, "pre-scatter normalized prefix rows");
        const bool k_match_ok =
            expect_bf16_row_range_match(k_compare, "pre-scatter key prefix rows");
        const bool v_match_ok =
            expect_bf16_row_range_match(v_compare, "pre-scatter value prefix rows");
        const bool padded_norm_match_ok = expect_bf16_row_range_match(
            padded_norm_compare,
            "pre-scatter normalized padded-prefix rows");
        const bool padded_k_match_ok = expect_bf16_row_range_match(
            padded_k_compare,
            "pre-scatter key padded-prefix rows");
        const bool padded_v_match_ok = expect_bf16_row_range_match(
            padded_v_compare,
            "pre-scatter value padded-prefix rows");
        ok = norm_match_ok &&
             k_match_ok &&
             v_match_ok &&
             padded_norm_match_ok &&
             padded_k_match_ok &&
             padded_v_match_ok &&
             ok;
      }
    }
  }

  // Compare direct snapshot/restore of the prefix state before any tail tokens
  // are run. If this fails, restore is suspect.
  auto snapshot_source_context = model->CreateRequestContext();
  auto snapshot_prefix_logits = make_boundary_logits_buffer(config);
  ok = expect(
           snapshot_source_context != nullptr && snapshot_source_context->valid() &&
               snapshot_prefix_logits != nullptr && snapshot_prefix_logits->valid(),
           "snapshot source context should allocate") &&
       ok;
  if (ok) {
    ok = expect(
             model->RunPrefill(
                 prompt.data(),
                 prefix_token_count,
                 *snapshot_source_context,
                 snapshot_prefix_logits.get()),
             "snapshot source prefix prefill should succeed") &&
         ok;
  }

  std::optional<nemotron::ReusableStateDescriptor> snapshot_descriptor;
  auto restored_prefix_context = model->CreateRequestContext();
  if (ok) {
    snapshot_descriptor = nemotron::SnapshotRequestState(
        environment->reusable_state_arena(),
        *snapshot_source_context,
        "split-prefill-localization");
    ok = expect(snapshot_descriptor.has_value(), "snapshot descriptor should allocate") &&
         expect(
             restored_prefix_context != nullptr && restored_prefix_context->valid(),
             "restored prefix context should allocate") &&
         ok;
  }
  if (ok) {
    ok = expect(
             nemotron::RestoreRequestState(
                 environment->reusable_state_arena(),
                 *snapshot_descriptor,
                 prefix_token_count,
                 *restored_prefix_context),
             "prefix snapshot restore should succeed") &&
         ok;
  }
  if (ok) {
    const auto restored_prefix_kv = compare_layer_kv_token_range(
        *snapshot_source_context,
        *restored_prefix_context,
        attention_window->attention_layer_index,
        0,
        prefix_token_count);
    ok = expect_layer_kv_match(restored_prefix_kv, "restored-prefix attention KV") && ok;

    const std::vector<float> source_conv = copy_tensor_to_host(*snapshot_source_context->mamba_conv_state());
    const std::vector<float> restored_conv = copy_tensor_to_host(*restored_prefix_context->mamba_conv_state());
    const std::vector<float> source_ssm = copy_tensor_to_host(*snapshot_source_context->mamba_state());
    const std::vector<float> restored_ssm = copy_tensor_to_host(*restored_prefix_context->mamba_state());
    const float conv_diff = max_abs_diff(source_conv, restored_conv);
    const float ssm_diff = max_abs_diff(source_ssm, restored_ssm);
    std::cerr << "split_prefill_localization_test: restored-prefix mamba"
              << " conv_max_abs_diff=" << conv_diff
              << " ssm_max_abs_diff=" << ssm_diff << "\n";
    ok = expect(
             conv_diff <= kMambaStateTolerance,
             "restored prefix conv state should round-trip exactly enough; max_abs_diff=" +
                 std::to_string(conv_diff)) &&
         ok;
    ok = expect(
             ssm_diff <= kMambaStateTolerance,
             "restored prefix SSM state should round-trip exactly enough; max_abs_diff=" +
                 std::to_string(ssm_diff)) &&
         ok;
  }

  auto full_context = model->CreateRequestContext();
  auto full_logits = make_boundary_logits_buffer(config);
  auto split_live_context = model->CreateRequestContext();
  auto split_live_prefix_logits = make_boundary_logits_buffer(config);
  auto split_live_tail_logits = make_boundary_logits_buffer(config);
  auto split_restored_context = model->CreateRequestContext();
  auto split_restored_tail_logits = make_boundary_logits_buffer(config);
  ok = expect(
           full_context != nullptr && full_context->valid() &&
               full_logits != nullptr && full_logits->valid() &&
               split_live_context != nullptr && split_live_context->valid() &&
               split_live_prefix_logits != nullptr && split_live_prefix_logits->valid() &&
               split_live_tail_logits != nullptr && split_live_tail_logits->valid() &&
               split_restored_context != nullptr && split_restored_context->valid() &&
               split_restored_tail_logits != nullptr && split_restored_tail_logits->valid(),
           "full and split continuation contexts should allocate") &&
       ok;
  if (ok) {
    ok = expect(
             model->RunPrefill(
                 prompt.data(),
                 prompt.size(),
                 *full_context,
                 full_logits.get()),
             "full prompt prefill should succeed") &&
         ok;
    ok = expect(
             model->RunPrefill(
                 prompt.data(),
                 prefix_token_count,
                 *split_live_context,
                 split_live_prefix_logits.get()),
             "split-live prefix prefill should succeed") &&
         ok;
    ok = expect(
             model->ContinuePrefill(
                 prompt.data() + prefix_token_count,
                 suffix_token_count,
                 *split_live_context,
                 split_live_tail_logits.get()),
             "split-live tail prefill should succeed") &&
         ok;
    ok = expect(
             nemotron::RestoreRequestState(
                 environment->reusable_state_arena(),
                 *snapshot_descriptor,
                 prefix_token_count,
                 *split_restored_context),
             "split-restored prefix restore should succeed") &&
         ok;
    ok = expect(
             model->ContinuePrefill(
                 prompt.data() + prefix_token_count,
                 suffix_token_count,
                 *split_restored_context,
                 split_restored_tail_logits.get()),
             "split-restored tail prefill should succeed") &&
         ok;
  }

  if (ok) {
    const auto full_vs_prefix_only_prefix_kv = compare_layer_kv_token_range(
        *full_context,
        *snapshot_source_context,
        attention_window->attention_layer_index,
        0,
        prefix_token_count);
    const auto full_vs_restored_prefix_only_kv = compare_layer_kv_token_range(
        *full_context,
        *restored_prefix_context,
        attention_window->attention_layer_index,
        0,
        prefix_token_count);
    const auto full_vs_live_prefix_kv = compare_layer_kv_token_range(
        *full_context,
        *split_live_context,
        attention_window->attention_layer_index,
        0,
        prefix_token_count);
    const auto full_vs_live_suffix_kv = compare_layer_kv_token_range(
        *full_context,
        *split_live_context,
        attention_window->attention_layer_index,
        prefix_token_count,
        suffix_token_count);
    const auto full_vs_restored_prefix_kv = compare_layer_kv_token_range(
        *full_context,
        *split_restored_context,
        attention_window->attention_layer_index,
        0,
        prefix_token_count);
    const auto full_vs_restored_suffix_kv = compare_layer_kv_token_range(
        *full_context,
        *split_restored_context,
        attention_window->attention_layer_index,
        prefix_token_count,
        suffix_token_count);
    const auto live_vs_restored_full_kv = compare_layer_kv_token_range(
        *split_live_context,
        *split_restored_context,
        attention_window->attention_layer_index,
        0,
        prompt.size());
    ok = expect_layer_kv_match(
             full_vs_prefix_only_prefix_kv,
             "full vs prefix-only prefix attention KV") &&
         ok;
    ok = expect_layer_kv_match(
             full_vs_restored_prefix_only_kv,
             "full vs restored-prefix-only attention KV") &&
         ok;
    ok = expect_layer_kv_match(full_vs_live_prefix_kv, "full vs split-live prefix attention KV") && ok;
    ok = expect_layer_kv_match(full_vs_live_suffix_kv, "full vs split-live suffix attention KV") && ok;
    ok = expect_layer_kv_match(full_vs_restored_prefix_kv, "full vs split-restored prefix attention KV") && ok;
    ok = expect_layer_kv_match(full_vs_restored_suffix_kv, "full vs split-restored suffix attention KV") && ok;
    ok = expect_layer_kv_match(live_vs_restored_full_kv, "split-live vs split-restored full attention KV") && ok;
  }

  auto full_layer5_context = model->CreateRequestContext();
  auto full_layer5_logits = make_boundary_logits_buffer(config);
  auto split_live_layer5_context = model->CreateRequestContext();
  auto split_live_layer5_prefix_logits = make_boundary_logits_buffer(config);
  auto split_live_layer5_tail_logits = make_boundary_logits_buffer(config);
  auto split_restored_layer5_context = model->CreateRequestContext();
  auto split_restored_layer5_tail_logits = make_boundary_logits_buffer(config);
  nemotron::SingleTokenForwardTrace full_layer5_trace;
  nemotron::SingleTokenForwardTrace split_live_layer5_trace;
  nemotron::SingleTokenForwardTrace split_restored_layer5_trace;
  ok = expect(
           full_layer5_context != nullptr && full_layer5_context->valid() &&
               full_layer5_logits != nullptr && full_layer5_logits->valid() &&
               split_live_layer5_context != nullptr && split_live_layer5_context->valid() &&
               split_live_layer5_prefix_logits != nullptr && split_live_layer5_prefix_logits->valid() &&
               split_live_layer5_tail_logits != nullptr && split_live_layer5_tail_logits->valid() &&
               split_restored_layer5_context != nullptr && split_restored_layer5_context->valid() &&
               split_restored_layer5_tail_logits != nullptr && split_restored_layer5_tail_logits->valid(),
           "layer-5 trace contexts should allocate") &&
       ok;
  if (ok) {
    ok = expect(
             model->RunPrefill(
                 prompt.data(),
                 prompt.size(),
                 *full_layer5_context,
                 full_layer5_logits.get(),
                 {attention_window->attention_layer_index},
                 &full_layer5_trace,
                 attention_window->attention_layer_index),
             "full prompt layer-5 trace should succeed") &&
         ok;
    ok = expect(
             model->RunPrefill(
                 prompt.data(),
                 prefix_token_count,
                 *split_live_layer5_context,
                 split_live_layer5_prefix_logits.get()),
             "split-live layer-5 prefix prefill should succeed") &&
         ok;
    ok = expect(
             model->ContinuePrefill(
                 prompt.data() + prefix_token_count,
                 suffix_token_count,
                 *split_live_layer5_context,
                 split_live_layer5_tail_logits.get(),
                 {attention_window->attention_layer_index},
                 &split_live_layer5_trace,
                 attention_window->attention_layer_index),
             "split-live layer-5 tail trace should succeed") &&
         ok;
    ok = expect(
             nemotron::RestoreRequestState(
                 environment->reusable_state_arena(),
                 *snapshot_descriptor,
                 prefix_token_count,
                 *split_restored_layer5_context),
             "split-restored layer-5 prefix restore should succeed") &&
         ok;
    ok = expect(
             model->ContinuePrefill(
                 prompt.data() + prefix_token_count,
                 suffix_token_count,
                 *split_restored_layer5_context,
                 split_restored_layer5_tail_logits.get(),
                 {attention_window->attention_layer_index},
                 &split_restored_layer5_trace,
                 attention_window->attention_layer_index),
             "split-restored layer-5 tail trace should succeed") &&
         ok;
  }
  if (ok) {
    ok = expect(
             full_layer5_trace.captured_layers.size() == 1 &&
                 split_live_layer5_trace.captured_layers.size() == 1 &&
                 split_restored_layer5_trace.captured_layers.size() == 1,
             "layer-5 traces should capture exactly one layer each") &&
         ok;
  }
  if (ok) {
    const auto& full_hidden = full_layer5_trace.captured_layers.front().hidden;
    const auto& split_live_hidden = split_live_layer5_trace.captured_layers.front().hidden;
    const auto& split_restored_hidden = split_restored_layer5_trace.captured_layers.front().hidden;
    for (std::size_t suffix_row = 0; suffix_row < suffix_token_count; ++suffix_row) {
      const std::vector<float> full_row = slice_row(
          full_hidden,
          prefix_token_count + suffix_row,
          config.hidden_size);
      const std::vector<float> live_row = slice_row(
          split_live_hidden,
          suffix_row,
          config.hidden_size);
      const std::vector<float> restored_row = slice_row(
          split_restored_hidden,
          suffix_row,
          config.hidden_size);
      ok = expect_hidden_row_match(
               full_row,
               live_row,
               kHiddenTolerance,
               "layer-5 full vs split-live suffix row " + std::to_string(suffix_row)) &&
           ok;
      ok = expect_hidden_row_match(
               full_row,
               restored_row,
               kHiddenTolerance,
               "layer-5 full vs split-restored suffix row " + std::to_string(suffix_row)) &&
           ok;
      ok = expect_hidden_row_match(
               live_row,
               restored_row,
               kHiddenTolerance,
               "layer-5 split-live vs split-restored suffix row " + std::to_string(suffix_row)) &&
           ok;
    }
  }

  return ok;
}

}  // namespace

int main() {
  if (!run_split_prefill_localization_test()) {
    return 1;
  }
  std::cout << "split_prefill_localization_test: PASS\n";
  return 0;
}
