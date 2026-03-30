#include "nemotron/expert_layer.h"
#include "nemotron/linear_op.h"

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
using nemotron::ExpertLayerBindings;
using nemotron::ExpertLayerConfig;
using nemotron::ExpertLayerRunTrace;
using nemotron::ExpertLayerSlice;
using nemotron::ExpertSelection;
using nemotron::GemmDescriptor;
using nemotron::GemmHeuristicCache;
using nemotron::GemmKernelFamily;
using nemotron::KernelTensorDescriptor;
using nemotron::ScaledFp8LinearConfig;
using nemotron::ScaledFp8LinearOp;
using nemotron::UploadedLinearOp;

struct FixtureMetadata {
  std::size_t layer_index = 0;
  std::size_t input_rows = 1;
  std::size_t hidden_size = 0;
  std::size_t moe_latent_size = 0;
  std::size_t routed_expert_intermediate_size = 0;
  std::size_t shared_expert_intermediate_size = 0;
  std::size_t n_routed_experts = 0;
  std::size_t top_k = 0;
  std::size_t n_group = 0;
  std::size_t topk_group = 0;
  float routed_scaling_factor = 0.0f;
  float rms_epsilon = 0.0f;
  bool norm_topk_prob = false;
  std::string fc1_latent_family = "scaled_fp8";
  std::string shared_up_family = "scaled_fp8";
  std::string shared_down_family;
};

struct RoutedExpertFixture {
  std::int32_t expert_index = -1;
  std::vector<std::uint8_t> up_weight_packed;
  std::vector<std::uint8_t> up_weight_block_scales;
  std::vector<float> up_weight_tensor_scale;
  std::vector<std::uint8_t> down_weight_packed;
  std::vector<std::uint8_t> down_weight_block_scales;
  std::vector<float> down_weight_tensor_scale;
  GemmDescriptor up_descriptor;
  GemmDescriptor down_descriptor;
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

void write_float_file(const std::filesystem::path& path, const std::vector<float>& values) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
}

std::filesystem::path resolve_fixture_root(const char* env_name, const char* default_root) {
  const char* override_root = std::getenv(env_name);
  if (override_root != nullptr && std::string(override_root).size() != 0) {
    return std::filesystem::path(override_root);
  }
  return std::filesystem::path(default_root);
}

std::vector<std::int32_t> read_int32_file(const std::filesystem::path& path) {
  const std::vector<std::uint8_t> bytes = read_file_bytes(path);
  if (bytes.size() % sizeof(std::int32_t) != 0) {
    return {};
  }
  std::vector<std::int32_t> values(bytes.size() / sizeof(std::int32_t), 0);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
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

std::optional<bool> parse_json_bool_field(const std::string& json, const std::string& key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*(true|false)");
  std::smatch match;
  if (!std::regex_search(json, match, pattern) || match.size() != 2) {
    return std::nullopt;
  }
  return match[1].str() == "true";
}

std::optional<std::string> parse_json_string_field(const std::string& json, const std::string& key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
  std::smatch match;
  if (!std::regex_search(json, match, pattern) || match.size() != 2) {
    return std::nullopt;
  }
  return match[1].str();
}

std::optional<FixtureMetadata> load_metadata(const std::filesystem::path& root) {
  const std::string json = read_text_file(root / "metadata.json");
  FixtureMetadata metadata;
  const auto layer_index = parse_json_uint_field(json, "layer_index");
  const auto input_rows = parse_json_uint_field(json, "input_rows");
  const auto hidden_size = parse_json_uint_field(json, "hidden_size");
  const auto moe_latent_size = parse_json_uint_field(json, "moe_latent_size");
  const auto routed_expert_intermediate_size = parse_json_uint_field(json, "routed_expert_intermediate_size");
  const auto shared_expert_intermediate_size = parse_json_uint_field(json, "shared_expert_intermediate_size");
  const auto n_routed_experts = parse_json_uint_field(json, "n_routed_experts");
  const auto top_k = parse_json_uint_field(json, "top_k");
  const auto n_group = parse_json_uint_field(json, "n_group");
  const auto topk_group = parse_json_uint_field(json, "topk_group");
  const auto routed_scaling_factor = parse_json_float_field(json, "routed_scaling_factor");
  const auto rms_epsilon = parse_json_float_field(json, "rms_epsilon");
  const auto norm_topk_prob = parse_json_bool_field(json, "norm_topk_prob");
  const auto fc1_latent_family = parse_json_string_field(json, "fc1_latent_family");
  const auto shared_up_family = parse_json_string_field(json, "shared_up_family");
  const auto shared_down_family = parse_json_string_field(json, "shared_down_family");
  if (!layer_index || !hidden_size || !moe_latent_size || !routed_expert_intermediate_size ||
      !shared_expert_intermediate_size || !n_routed_experts || !top_k || !n_group ||
      !topk_group || !routed_scaling_factor || !rms_epsilon || !norm_topk_prob ||
      !shared_down_family) {
    return std::nullopt;
  }
  metadata.layer_index = *layer_index;
  metadata.input_rows = input_rows.value_or(1);
  metadata.hidden_size = *hidden_size;
  metadata.moe_latent_size = *moe_latent_size;
  metadata.routed_expert_intermediate_size = *routed_expert_intermediate_size;
  metadata.shared_expert_intermediate_size = *shared_expert_intermediate_size;
  metadata.n_routed_experts = *n_routed_experts;
  metadata.top_k = *top_k;
  metadata.n_group = *n_group;
  metadata.topk_group = *topk_group;
  metadata.routed_scaling_factor = *routed_scaling_factor;
  metadata.rms_epsilon = *rms_epsilon;
  metadata.norm_topk_prob = *norm_topk_prob;
  if (fc1_latent_family) {
    metadata.fc1_latent_family = *fc1_latent_family;
  }
  if (shared_up_family) {
    metadata.shared_up_family = *shared_up_family;
  }
  metadata.shared_down_family = *shared_down_family;
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
  descriptor.storage_dtype = "fp8_e4m3fn";
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

GemmDescriptor make_nvfp4_descriptor(
    const std::string& name,
    const std::vector<std::uint8_t>& packed,
    const std::vector<std::uint8_t>& block_scales,
    const std::vector<float>& tensor_scale,
    std::size_t rows,
    std::size_t cols) {
  GemmDescriptor descriptor;
  descriptor.tensor_name = name;
  descriptor.op_class = "oracle";
  descriptor.kernel_family = GemmKernelFamily::kCublasLtNvfp4BlockScaled;
  descriptor.output_rows = rows;
  descriptor.input_cols = cols;
  descriptor.storage_dtype = "nvfp4_e2m1";
  descriptor.compute_dtype = "fp32_accum";
  descriptor.layout_tag = "cublaslt_fp4_tn_v1";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data = packed.data();
  descriptor.packed_nbytes = packed.size();
  descriptor.block_scales_data = block_scales.data();
  descriptor.block_scales_nbytes = block_scales.size();
  descriptor.tensor_scale_data = reinterpret_cast<const std::uint8_t*>(tensor_scale.data());
  descriptor.tensor_scale_nbytes = tensor_scale.size() * sizeof(float);
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

float relative_l2_diff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size() || lhs.empty()) {
    return INFINITY;
  }
  double diff_sq = 0.0;
  double ref_sq = 0.0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    const double diff = static_cast<double>(lhs[i]) - static_cast<double>(rhs[i]);
    diff_sq += diff * diff;
    ref_sq += static_cast<double>(rhs[i]) * static_cast<double>(rhs[i]);
  }
  if (ref_sq == 0.0) {
    return diff_sq == 0.0 ? 0.0f : INFINITY;
  }
  return static_cast<float>(std::sqrt(diff_sq / ref_sq));
}

std::vector<ExpertSelection> sorted_selections(std::vector<ExpertSelection> selections) {
  std::sort(
      selections.begin(),
      selections.end(),
      [](const ExpertSelection& lhs, const ExpertSelection& rhs) {
        return lhs.expert_index < rhs.expert_index;
      });
  return selections;
}

bool run_expert_layer_fixture() {
  const char* dump_root_env = std::getenv("NEMOTRON_EXPERT_LAYER_DUMP_ROOT");
  const bool debug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  if (debug) {
    std::cerr << "expert_layer_oracle_test: dump_root_env="
              << (dump_root_env == nullptr ? "<unset>" : dump_root_env) << "\n";
  }
  const auto cublas = CublasLtHandle::Create();
  if (!cublas || !cublas->valid()) {
    std::cout << "expert_layer_oracle_test: SKIP (no CUDA device)\n";
    return true;
  }

  const std::filesystem::path fixture_root = resolve_fixture_root(
      "NEMOTRON_EXPERT_LAYER_ORACLE_FIXTURE_ROOT_OVERRIDE",
      NEMOTRON_EXPERT_LAYER_ORACLE_FIXTURE_ROOT);
  const auto metadata = load_metadata(fixture_root);
  if (!expect(metadata.has_value(), "expert layer oracle fixture metadata should load")) {
    return false;
  }

  const std::vector<float> input_hidden = read_float_file(fixture_root / "input_hidden_fp32.bin");
  const std::vector<float> expected_norm_output =
      read_float_file(fixture_root / "expected_norm_output_fp32.bin");
  const std::vector<float> norm_weight = read_float_file(fixture_root / "norm_weight_fp32.bin");
  const std::vector<float> gate_weight = read_float_file(fixture_root / "gate_weight_fp32.bin");
  const std::vector<float> gate_score_correction_bias =
      read_float_file(fixture_root / "gate_score_correction_bias_fp32.bin");
  const std::vector<std::uint8_t> fc1_latent_weight =
      read_file_bytes(fixture_root / "fc1_latent_weight_fp8.bin");
  const std::vector<float> fc1_latent_weight_fp32 =
      read_float_file(fixture_root / "fc1_latent_weight_fp32.bin");
  const std::vector<float> fc1_latent_weight_scale =
      read_float_file(fixture_root / "fc1_latent_weight_scale_fp32.bin");
  const std::vector<float> fc1_latent_input_scale =
      read_float_file(fixture_root / "fc1_latent_input_scale_fp32.bin");
  const std::vector<float> fc2_latent_weight =
      read_float_file(fixture_root / "fc2_latent_weight_fp32.bin");
  const std::vector<std::uint8_t> shared_up_weight =
      read_file_bytes(fixture_root / "shared_up_weight_fp8.bin");
  const std::vector<float> shared_up_weight_fp32 =
      read_float_file(fixture_root / "shared_up_weight_fp32.bin");
  const std::vector<float> shared_up_weight_scale =
      read_float_file(fixture_root / "shared_up_weight_scale_fp32.bin");
  const std::vector<float> shared_up_input_scale =
      read_float_file(fixture_root / "shared_up_input_scale_fp32.bin");
  const std::vector<std::uint8_t> shared_down_weight_packed =
      read_file_bytes(fixture_root / "shared_down_weight_packed.bin");
  const std::vector<std::uint8_t> shared_down_weight_block_scales =
      read_file_bytes(fixture_root / "shared_down_weight_block_scales.bin");
  const std::vector<float> shared_down_weight_tensor_scale =
      read_float_file(fixture_root / "shared_down_weight_tensor_scale_fp32.bin");
  const std::vector<std::uint8_t> shared_down_weight_fp8 =
      read_file_bytes(fixture_root / "shared_down_weight_fp8.bin");
  const std::vector<float> shared_down_weight_fp32 =
      read_float_file(fixture_root / "shared_down_weight_fp32.bin");
  const std::vector<float> shared_down_weight_scale =
      read_float_file(fixture_root / "shared_down_weight_scale_fp32.bin");
  const std::vector<float> shared_down_input_scale =
      read_float_file(fixture_root / "shared_down_input_scale_fp32.bin");
  const std::vector<float> expected_router_logits =
      read_float_file(fixture_root / "expected_router_logits_fp32.bin");
  const std::vector<float> expected_fc1_latent_output =
      read_float_file(fixture_root / "expected_fc1_latent_output_fp32.bin");
  const std::vector<float> expected_routed_latent_output =
      read_float_file(fixture_root / "expected_routed_latent_output_fp32.bin");
  const std::vector<float> expected_shared_output =
      read_float_file(fixture_root / "expected_shared_output_fp32.bin");
  const std::vector<float> expected_mixer_output =
      read_float_file(fixture_root / "expected_mixer_output_fp32.bin");
  const std::vector<float> expected_final_output =
      read_float_file(fixture_root / "expected_final_output_fp32.bin");
  const std::vector<std::int32_t> expected_selected_expert_indices =
      read_int32_file(fixture_root / "expected_selected_expert_indices_u32.bin");
  const std::vector<float> expected_selected_expert_weights =
      read_float_file(fixture_root / "expected_selected_expert_weights_fp32.bin");
  const std::string layer_prefix = "backbone.layers." + std::to_string(metadata->layer_index);
  const std::string mixer_prefix = layer_prefix + ".mixer";

  if (!expect(input_hidden.size() == metadata->input_rows * metadata->hidden_size, "input hidden size should match metadata") ||
      !expect(norm_weight.size() == metadata->hidden_size, "norm weight size should match metadata") ||
      !expect(
          expected_norm_output.empty() ||
              expected_norm_output.size() == metadata->input_rows * metadata->hidden_size,
          "expected norm output size should match metadata when present") ||
      !expect(gate_weight.size() == metadata->n_routed_experts * metadata->hidden_size, "gate weight size should match metadata") ||
      !expect(gate_score_correction_bias.size() == metadata->n_routed_experts, "gate correction bias size should match metadata") ||
      !expect(
          (metadata->fc1_latent_family == "scaled_fp8" &&
           fc1_latent_weight.size() == metadata->moe_latent_size * metadata->hidden_size &&
           fc1_latent_weight_scale.size() == 1 &&
           fc1_latent_input_scale.size() == 1) ||
              (metadata->fc1_latent_family == "dense" &&
               fc1_latent_weight_fp32.size() == metadata->moe_latent_size * metadata->hidden_size),
          "fc1 latent weights should match metadata") ||
      !expect(fc2_latent_weight.size() == metadata->hidden_size * metadata->moe_latent_size, "fc2 latent weight size should match metadata") ||
      !expect(
          (metadata->shared_up_family == "scaled_fp8" &&
           shared_up_weight.size() == metadata->shared_expert_intermediate_size * metadata->hidden_size &&
           shared_up_weight_scale.size() == 1 &&
           shared_up_input_scale.size() == 1) ||
              (metadata->shared_up_family == "dense" &&
               shared_up_weight_fp32.size() == metadata->shared_expert_intermediate_size * metadata->hidden_size),
          "shared up weights should match metadata") ||
      !expect(
          (metadata->shared_down_family == "nvfp4" &&
           !shared_down_weight_packed.empty() &&
           !shared_down_weight_block_scales.empty() &&
           shared_down_weight_tensor_scale.size() == 1) ||
              (metadata->shared_down_family == "scaled_fp8" &&
               shared_down_weight_fp8.size() == metadata->hidden_size * metadata->shared_expert_intermediate_size &&
               shared_down_weight_scale.size() == 1 &&
               shared_down_input_scale.size() == 1) ||
              (metadata->shared_down_family == "dense" &&
               shared_down_weight_fp32.size() == metadata->hidden_size * metadata->shared_expert_intermediate_size),
          "shared down weights should match metadata") ||
      !expect(expected_router_logits.size() == metadata->input_rows * metadata->n_routed_experts, "expected router logits size should match metadata") ||
      !expect(expected_fc1_latent_output.size() == metadata->input_rows * metadata->moe_latent_size, "expected fc1 latent output size should match metadata") ||
      !expect(expected_routed_latent_output.size() == metadata->input_rows * metadata->moe_latent_size, "expected routed latent output size should match metadata") ||
      !expect(expected_shared_output.size() == metadata->input_rows * metadata->hidden_size, "expected shared output size should match metadata") ||
      !expect(expected_mixer_output.size() == metadata->input_rows * metadata->hidden_size, "expected mixer output size should match metadata") ||
      !expect(expected_final_output.size() == metadata->input_rows * metadata->hidden_size, "expected final output size should match metadata") ||
      !expect(expected_selected_expert_indices.size() == metadata->input_rows * metadata->top_k, "selected expert count should match top-k") ||
      !expect(expected_selected_expert_weights.size() == metadata->input_rows * metadata->top_k, "selected expert weights size should match top-k")) {
    return false;
  }

  const auto input_norm_descriptor =
      make_fp32_descriptor(layer_prefix + ".norm.weight", norm_weight, {metadata->hidden_size});
  const auto gate_weight_descriptor =
      make_dense_descriptor(mixer_prefix + ".gate.weight", gate_weight, metadata->n_routed_experts, metadata->hidden_size);
  const auto gate_score_correction_bias_descriptor =
      make_fp32_descriptor(mixer_prefix + ".gate.e_score_correction_bias", gate_score_correction_bias, {metadata->n_routed_experts});
  const auto fc1_latent_weight_descriptor =
      make_fp8_descriptor(mixer_prefix + ".fc1_latent_proj.weight", fc1_latent_weight, {metadata->moe_latent_size, metadata->hidden_size});
  const auto fc1_latent_dense_descriptor =
      make_dense_descriptor(mixer_prefix + ".fc1_latent_proj.weight", fc1_latent_weight_fp32, metadata->moe_latent_size, metadata->hidden_size);
  const auto fc1_latent_weight_scale_descriptor =
      make_fp32_descriptor(mixer_prefix + ".fc1_latent_proj.weight_scale", fc1_latent_weight_scale, {1});
  const auto fc1_latent_input_scale_descriptor =
      make_fp32_descriptor(mixer_prefix + ".fc1_latent_proj.input_scale", fc1_latent_input_scale, {1});
  const auto fc2_latent_weight_descriptor =
      make_dense_descriptor(mixer_prefix + ".fc2_latent_proj.weight", fc2_latent_weight, metadata->hidden_size, metadata->moe_latent_size);
  const auto shared_up_weight_descriptor =
      make_fp8_descriptor(mixer_prefix + ".shared_experts.up_proj.weight", shared_up_weight, {metadata->shared_expert_intermediate_size, metadata->hidden_size});
  const auto shared_up_dense_descriptor =
      make_dense_descriptor(
          mixer_prefix + ".shared_experts.up_proj.weight",
          shared_up_weight_fp32,
          metadata->shared_expert_intermediate_size,
          metadata->hidden_size);
  const auto shared_up_weight_scale_descriptor =
      make_fp32_descriptor(mixer_prefix + ".shared_experts.up_proj.weight_scale", shared_up_weight_scale, {1});
  const auto shared_up_input_scale_descriptor =
      make_fp32_descriptor(mixer_prefix + ".shared_experts.up_proj.input_scale", shared_up_input_scale, {1});
  const auto shared_down_nvfp4_descriptor = make_nvfp4_descriptor(
      mixer_prefix + ".shared_experts.down_proj.weight",
      shared_down_weight_packed,
      shared_down_weight_block_scales,
      shared_down_weight_tensor_scale,
      metadata->hidden_size,
      metadata->shared_expert_intermediate_size);
  const auto shared_down_fp8_descriptor =
      make_fp8_descriptor(
          mixer_prefix + ".shared_experts.down_proj.weight",
          shared_down_weight_fp8,
          {metadata->hidden_size, metadata->shared_expert_intermediate_size});
  const auto shared_down_dense_descriptor =
      make_dense_descriptor(
          mixer_prefix + ".shared_experts.down_proj.weight",
          shared_down_weight_fp32,
          metadata->hidden_size,
          metadata->shared_expert_intermediate_size);
  const auto shared_down_weight_scale_descriptor =
      make_fp32_descriptor(mixer_prefix + ".shared_experts.down_proj.weight_scale", shared_down_weight_scale, {1});
  const auto shared_down_input_scale_descriptor =
      make_fp32_descriptor(mixer_prefix + ".shared_experts.down_proj.input_scale", shared_down_input_scale, {1});

  std::vector<RoutedExpertFixture> routed_experts;
  routed_experts.reserve(expected_selected_expert_indices.size());
  ExpertLayerBindings bindings;
  bindings.input_norm_weight = &input_norm_descriptor;
  bindings.gate_weight = &gate_weight_descriptor;
  bindings.gate_score_correction_bias = &gate_score_correction_bias_descriptor;
  if (metadata->fc1_latent_family == "scaled_fp8") {
    bindings.fc1_latent_kernel_weight = &fc1_latent_weight_descriptor;
    bindings.fc1_latent_weight_scale = &fc1_latent_weight_scale_descriptor;
    bindings.fc1_latent_input_scale = &fc1_latent_input_scale_descriptor;
  } else if (metadata->fc1_latent_family == "dense") {
    bindings.fc1_latent_gemm_weight = &fc1_latent_dense_descriptor;
  } else {
    return expect(false, "fc1 latent family should be supported");
  }
  bindings.fc2_latent_weight = &fc2_latent_weight_descriptor;
  if (metadata->shared_up_family == "scaled_fp8") {
    bindings.shared_up_kernel_weight = &shared_up_weight_descriptor;
    bindings.shared_up_weight_scale = &shared_up_weight_scale_descriptor;
    bindings.shared_up_input_scale = &shared_up_input_scale_descriptor;
  } else if (metadata->shared_up_family == "dense") {
    bindings.shared_up_gemm_weight = &shared_up_dense_descriptor;
  } else {
    return expect(false, "shared up family should be supported");
  }
  if (metadata->shared_down_family == "nvfp4") {
    bindings.shared_down_gemm_weight = &shared_down_nvfp4_descriptor;
  } else if (metadata->shared_down_family == "scaled_fp8") {
    bindings.shared_down_kernel_weight = &shared_down_fp8_descriptor;
    bindings.shared_down_weight_scale = &shared_down_weight_scale_descriptor;
    bindings.shared_down_input_scale = &shared_down_input_scale_descriptor;
    if (std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr) {
      std::cerr << "expert_layer_oracle_test: layer=" << metadata->layer_index
                << " shared_down_family=scaled_fp8"
                << " weight_bytes=" << shared_down_weight_fp8.size()
                << " weight_scale_count=" << shared_down_weight_scale.size()
                << " input_scale_count=" << shared_down_input_scale.size()
                << " storage_dtype=" << shared_down_fp8_descriptor.storage_dtype
                << "\n";
    }
  } else if (metadata->shared_down_family == "dense") {
    bindings.shared_down_gemm_weight = &shared_down_dense_descriptor;
  } else {
    return expect(false, "shared down family should be supported");
  }
  bindings.routed_experts.resize(metadata->n_routed_experts);

  std::vector<bool> routed_expert_needed(metadata->n_routed_experts, false);
  for (std::int32_t expert_index : expected_selected_expert_indices) {
    if (expert_index < 0 ||
        static_cast<std::size_t>(expert_index) >= metadata->n_routed_experts) {
      return expect(false, "selected expert index should be in range");
    }
    routed_expert_needed[static_cast<std::size_t>(expert_index)] = true;
  }

  for (std::size_t expert_index = 0;
       expert_index < routed_expert_needed.size();
       ++expert_index) {
    if (!routed_expert_needed[expert_index]) {
      continue;
    }
    const std::string expert_dir_name =
        std::string("expert_") +
        (expert_index < 100 ? (expert_index < 10 ? "00" : "0") : "") +
        std::to_string(expert_index);
    const std::filesystem::path expert_dir = fixture_root / expert_dir_name;
    RoutedExpertFixture expert;
    expert.expert_index = static_cast<std::int32_t>(expert_index);
    expert.up_weight_packed = read_file_bytes(expert_dir / "up_proj_weight_packed.bin");
    expert.up_weight_block_scales = read_file_bytes(expert_dir / "up_proj_weight_block_scales.bin");
    expert.up_weight_tensor_scale = read_float_file(expert_dir / "up_proj_weight_tensor_scale_fp32.bin");
    expert.down_weight_packed = read_file_bytes(expert_dir / "down_proj_weight_packed.bin");
    expert.down_weight_block_scales = read_file_bytes(expert_dir / "down_proj_weight_block_scales.bin");
    expert.down_weight_tensor_scale = read_float_file(expert_dir / "down_proj_weight_tensor_scale_fp32.bin");
    if (!expect(expert.up_weight_tensor_scale.size() == 1, "expert up tensor scale should be scalar") ||
        !expect(expert.down_weight_tensor_scale.size() == 1, "expert down tensor scale should be scalar")) {
      return false;
    }
    expert.up_descriptor = make_nvfp4_descriptor(
        mixer_prefix + ".experts." + std::to_string(expert_index) + ".up_proj.weight",
        expert.up_weight_packed,
        expert.up_weight_block_scales,
        expert.up_weight_tensor_scale,
        metadata->routed_expert_intermediate_size,
        metadata->moe_latent_size);
    expert.down_descriptor = make_nvfp4_descriptor(
        mixer_prefix + ".experts." + std::to_string(expert_index) + ".down_proj.weight",
        expert.down_weight_packed,
        expert.down_weight_block_scales,
        expert.down_weight_tensor_scale,
        metadata->moe_latent_size,
        metadata->routed_expert_intermediate_size);
    routed_experts.push_back(std::move(expert));
  }

  for (RoutedExpertFixture& expert : routed_experts) {
    bindings.routed_experts[static_cast<std::size_t>(expert.expert_index)].up_proj = &expert.up_descriptor;
    bindings.routed_experts[static_cast<std::size_t>(expert.expert_index)].down_proj = &expert.down_descriptor;
  }

  ExpertLayerConfig layer_config;
  layer_config.layer_index = metadata->layer_index;
  layer_config.hidden_size = metadata->hidden_size;
  layer_config.moe_latent_size = metadata->moe_latent_size;
  layer_config.routed_expert_intermediate_size = metadata->routed_expert_intermediate_size;
  layer_config.shared_expert_intermediate_size = metadata->shared_expert_intermediate_size;
  layer_config.n_routed_experts = metadata->n_routed_experts;
  layer_config.top_k = metadata->top_k;
  layer_config.n_group = metadata->n_group;
  layer_config.topk_group = metadata->topk_group;
  layer_config.routed_scaling_factor = metadata->routed_scaling_factor;
  layer_config.norm_topk_prob = metadata->norm_topk_prob;
  layer_config.rms_epsilon = metadata->rms_epsilon;

  auto slice = ExpertLayerSlice::Create(layer_config, bindings);
  if (!expect(slice != nullptr && slice->valid(), "expert layer slice should be created")) {
    return false;
  }

  auto input = DeviceTensorFp32::Create({metadata->input_rows, metadata->hidden_size});
  auto output = DeviceTensorFp32::Create({metadata->input_rows, metadata->hidden_size});
  if (!expect(input != nullptr && output != nullptr, "device tensors should be allocated")) {
    return false;
  }
  if (!expect(input->CopyFromHost(input_hidden.data(), input_hidden.size()), "input hidden should upload")) {
    return false;
  }

  GemmHeuristicCache heuristic_cache;
  if (!expect(
          slice->Run(*cublas, &heuristic_cache, *input, output.get(), nullptr),
          "expert layer slice should run")) {
    return false;
  }

  std::vector<float> output_host(output->numel(), 0.0f);
  if (!expect(output->CopyToHost(output_host.data(), output_host.size()), "final output should download")) {
    return false;
  }

  auto row_input = DeviceTensorFp32::Create({1, metadata->hidden_size});
  auto row_output = DeviceTensorFp32::Create({1, metadata->hidden_size});
  if (!expect(row_input != nullptr && row_output != nullptr, "row device tensors should allocate")) {
    return false;
  }

  float max_selection_weight_diff = 0.0f;
  float norm_diff = 0.0f;
  float router_diff = 0.0f;
  float fc1_latent_diff = 0.0f;
  float routed_diff = 0.0f;
  float routed_activated_hidden_diff = 0.0f;
  std::size_t routed_activated_hidden_expert_index = std::numeric_limits<std::size_t>::max();
  float routed_expert_output_diff = 0.0f;
  std::size_t routed_expert_output_expert_index = std::numeric_limits<std::size_t>::max();
  float routed_contribution_diff = 0.0f;
  std::size_t routed_contribution_expert_index = std::numeric_limits<std::size_t>::max();
  float projected_routed_diff = 0.0f;
  float shared_diff = 0.0f;
  float mixer_diff = 0.0f;
  float final_diff = max_abs_diff(output_host, expected_final_output);
  float batch_vs_sequential_diff = 0.0f;
  float direct_gate_diff = 0.0f;
  float direct_fc1_diff = 0.0f;
  float direct_fc2_diff = 0.0f;
  float projected_routed_rel_l2 = 0.0f;
  float mixer_rel_l2 = 0.0f;
  float final_rel_l2 = relative_l2_diff(output_host, expected_final_output);
  std::vector<float> sequential_output(output_host.size(), 0.0f);
  std::vector<float> row_input_host(metadata->hidden_size, 0.0f);
  std::vector<float> row_output_host(metadata->hidden_size, 0.0f);
  for (std::size_t row = 0; row < metadata->input_rows; ++row) {
    const std::size_t hidden_offset = row * metadata->hidden_size;
    const std::size_t expert_offset = row * metadata->top_k;
    const std::size_t router_offset = row * metadata->n_routed_experts;
    const std::size_t latent_offset = row * metadata->moe_latent_size;

    std::copy_n(
        input_hidden.data() + hidden_offset,
        metadata->hidden_size,
        row_input_host.data());
    if (!expect(
            row_input->CopyFromHost(row_input_host.data(), row_input_host.size()),
            "row input should upload")) {
      return false;
    }

    ExpertLayerRunTrace trace;
    if (!expect(
            slice->Run(*cublas, &heuristic_cache, *row_input, row_output.get(), &trace),
            "row expert layer slice should run")) {
      return false;
    }
    if (!expect(
            row_output->CopyToHost(row_output_host.data(), row_output_host.size()),
            "row final output should download")) {
      return false;
    }
    if (dump_root_env != nullptr && std::string(dump_root_env).size() != 0 && row == 0) {
      const std::filesystem::path dump_root(dump_root_env);
      write_float_file(dump_root / "trace_latent_output_fp32.bin", trace.latent_output);
      write_float_file(dump_root / "trace_routed_latent_output_fp32.bin", trace.routed_latent_output);
      write_float_file(dump_root / "trace_projected_routed_output_fp32.bin", trace.projected_routed_output);
      write_float_file(dump_root / "trace_shared_output_fp32.bin", trace.shared_output);
      write_float_file(dump_root / "trace_mixer_output_fp32.bin", trace.mixer_output);
      for (std::size_t slot = 0; slot < trace.routed_expert_order.size(); ++slot) {
        const std::string slot_stem = std::string(slot < 10 ? "0" : "") + std::to_string(slot);
        write_float_file(
            dump_root / ("trace_routed_activated_hidden_slot" + slot_stem + "_fp32.bin"),
            std::vector<float>(
                trace.routed_expert_activated_hidden.begin() +
                    slot * metadata->routed_expert_intermediate_size,
                trace.routed_expert_activated_hidden.begin() +
                    (slot + 1) * metadata->routed_expert_intermediate_size));
        write_float_file(
            dump_root / ("trace_routed_expert_output_slot" + slot_stem + "_fp32.bin"),
            std::vector<float>(
                trace.routed_expert_outputs.begin() +
                    slot * metadata->moe_latent_size,
                trace.routed_expert_outputs.begin() +
                    (slot + 1) * metadata->moe_latent_size));
        write_float_file(
            dump_root / ("trace_routed_contribution_slot" + slot_stem + "_fp32.bin"),
            std::vector<float>(
                trace.routed_expert_weighted_contributions.begin() +
                    slot * metadata->moe_latent_size,
                trace.routed_expert_weighted_contributions.begin() +
                    (slot + 1) * metadata->moe_latent_size));
      }
    }
    std::copy(
        row_output_host.begin(),
        row_output_host.end(),
        sequential_output.begin() + hidden_offset);

    std::vector<ExpertSelection> expected_selections;
    expected_selections.reserve(metadata->top_k);
    for (std::size_t i = 0; i < metadata->top_k; ++i) {
      expected_selections.push_back(
          ExpertSelection{
              static_cast<std::size_t>(expected_selected_expert_indices[expert_offset + i]),
              expected_selected_expert_weights[expert_offset + i],
          });
    }

    const auto actual_sorted = sorted_selections(trace.selected_experts);
    const auto expected_sorted = sorted_selections(expected_selections);
    if (!expect(
            actual_sorted.size() == expected_sorted.size(),
            "selected expert counts should match")) {
      return false;
    }
    for (std::size_t i = 0; i < actual_sorted.size(); ++i) {
      if (!expect(
              actual_sorted[i].expert_index == expected_sorted[i].expert_index,
              "selected expert indices should match")) {
        return false;
      }
      max_selection_weight_diff = std::max(
          max_selection_weight_diff,
          std::fabs(actual_sorted[i].weight - expected_sorted[i].weight));
    }

    router_diff = std::max(
        router_diff,
        max_abs_diff(
            trace.router_logits,
            std::vector<float>(
                expected_router_logits.begin() + router_offset,
                expected_router_logits.begin() + router_offset + metadata->n_routed_experts)));
    if (!expected_norm_output.empty()) {
      norm_diff = std::max(
          norm_diff,
          max_abs_diff(
              trace.normalized_input,
              std::vector<float>(
                  expected_norm_output.begin() + hidden_offset,
                  expected_norm_output.begin() + hidden_offset + metadata->hidden_size)));
    }
    fc1_latent_diff = std::max(
        fc1_latent_diff,
        max_abs_diff(
            trace.latent_output,
            std::vector<float>(
                expected_fc1_latent_output.begin() + latent_offset,
                expected_fc1_latent_output.begin() + latent_offset + metadata->moe_latent_size)));
    routed_diff = std::max(
        routed_diff,
        max_abs_diff(
            trace.routed_latent_output,
            std::vector<float>(
                expected_routed_latent_output.begin() + latent_offset,
                expected_routed_latent_output.begin() + latent_offset + metadata->moe_latent_size)));
    if (!trace.routed_expert_order.empty() &&
        trace.routed_expert_weighted_contributions.size() ==
            trace.routed_expert_order.size() * metadata->moe_latent_size) {
      for (std::size_t slot = 0; slot < metadata->top_k; ++slot) {
        const std::string row_stem = std::string(row < 10 ? "0" : "") + std::to_string(row);
        const std::string slot_stem = std::string(slot < 10 ? "0" : "") + std::to_string(slot);
        const std::filesystem::path hidden_path =
            fixture_root /
            ("expected_routed_activated_hidden_row" + row_stem +
             "_slot" + slot_stem +
             "_fp32.bin");
        const std::filesystem::path output_path =
            fixture_root /
            ("expected_routed_expert_output_row" + row_stem +
             "_slot" + slot_stem +
             "_fp32.bin");
        const std::filesystem::path contribution_path =
            fixture_root /
            ("expected_routed_contribution_row" + row_stem +
             "_slot" + slot_stem +
             "_fp32.bin");
        if (!std::filesystem::exists(contribution_path)) {
          continue;
        }
        const std::size_t expected_expert_index =
            static_cast<std::size_t>(expected_selected_expert_indices[expert_offset + slot]);
        auto actual_it = std::find(
            trace.routed_expert_order.begin(),
            trace.routed_expert_order.end(),
            expected_expert_index);
        if (!expect(actual_it != trace.routed_expert_order.end(),
                    "expected routed contribution expert should be present in trace order")) {
          return false;
        }
        const std::size_t actual_slot =
            static_cast<std::size_t>(actual_it - trace.routed_expert_order.begin());
        if (std::filesystem::exists(hidden_path) &&
            trace.routed_expert_activated_hidden.size() ==
                trace.routed_expert_order.size() * metadata->routed_expert_intermediate_size) {
          const std::vector<float> expected_hidden = read_float_file(hidden_path);
          const float hidden_diff =
              max_abs_diff(
                  std::vector<float>(
                      trace.routed_expert_activated_hidden.begin() +
                          actual_slot * metadata->routed_expert_intermediate_size,
                      trace.routed_expert_activated_hidden.begin() +
                          (actual_slot + 1) * metadata->routed_expert_intermediate_size),
                  expected_hidden);
          if (hidden_diff > routed_activated_hidden_diff) {
            routed_activated_hidden_diff = hidden_diff;
            routed_activated_hidden_expert_index = expected_expert_index;
          }
        }
        if (std::filesystem::exists(output_path) &&
            trace.routed_expert_outputs.size() ==
                trace.routed_expert_order.size() * metadata->moe_latent_size) {
          const std::vector<float> expected_output = read_float_file(output_path);
          const float output_diff =
              max_abs_diff(
                  std::vector<float>(
                      trace.routed_expert_outputs.begin() +
                          actual_slot * metadata->moe_latent_size,
                      trace.routed_expert_outputs.begin() +
                          (actual_slot + 1) * metadata->moe_latent_size),
                  expected_output);
          if (output_diff > routed_expert_output_diff) {
            routed_expert_output_diff = output_diff;
            routed_expert_output_expert_index = expected_expert_index;
          }
        }
        const std::vector<float> expected_contribution = read_float_file(contribution_path);
        const float contribution_diff =
            max_abs_diff(
                std::vector<float>(
                    trace.routed_expert_weighted_contributions.begin() +
                        actual_slot * metadata->moe_latent_size,
                    trace.routed_expert_weighted_contributions.begin() +
                        (actual_slot + 1) * metadata->moe_latent_size),
                expected_contribution);
        if (contribution_diff > routed_contribution_diff) {
          routed_contribution_diff = contribution_diff;
          routed_contribution_expert_index = expected_expert_index;
        }
      }
    }
    std::vector<float> expected_projected_routed(metadata->hidden_size, 0.0f);
    for (std::size_t i = 0; i < metadata->hidden_size; ++i) {
      expected_projected_routed[i] =
          expected_mixer_output[hidden_offset + i] - expected_shared_output[hidden_offset + i];
    }
    projected_routed_diff = std::max(
        projected_routed_diff,
        max_abs_diff(trace.projected_routed_output, expected_projected_routed));
    projected_routed_rel_l2 = std::max(
        projected_routed_rel_l2,
        relative_l2_diff(trace.projected_routed_output, expected_projected_routed));
    shared_diff = std::max(
        shared_diff,
        max_abs_diff(
            trace.shared_output,
            std::vector<float>(
                expected_shared_output.begin() + hidden_offset,
                expected_shared_output.begin() + hidden_offset + metadata->hidden_size)));
    mixer_diff = std::max(
        mixer_diff,
        max_abs_diff(
            trace.mixer_output,
            std::vector<float>(
                expected_mixer_output.begin() + hidden_offset,
                expected_mixer_output.begin() + hidden_offset + metadata->hidden_size)));
    mixer_rel_l2 = std::max(
        mixer_rel_l2,
        relative_l2_diff(
            trace.mixer_output,
            std::vector<float>(
                expected_mixer_output.begin() + hidden_offset,
                expected_mixer_output.begin() + hidden_offset + metadata->hidden_size)));
    final_diff = std::max(
        final_diff,
        max_abs_diff(
            row_output_host,
            std::vector<float>(
                expected_final_output.begin() + hidden_offset,
                expected_final_output.begin() + hidden_offset + metadata->hidden_size)));
    final_rel_l2 = std::max(
        final_rel_l2,
        relative_l2_diff(
            row_output_host,
            std::vector<float>(
                expected_final_output.begin() + hidden_offset,
                expected_final_output.begin() + hidden_offset + metadata->hidden_size)));
  }
  batch_vs_sequential_diff = max_abs_diff(output_host, sequential_output);

  if (!expected_norm_output.empty()) {
    auto exact_norm_input =
        DeviceTensorFp32::Create({metadata->input_rows, metadata->hidden_size});
    auto exact_gate_output =
        DeviceTensorFp32::Create({metadata->input_rows, metadata->n_routed_experts});
    auto exact_fc1_output =
        DeviceTensorFp32::Create({metadata->input_rows, metadata->moe_latent_size});
    auto exact_routed_input =
        DeviceTensorFp32::Create({metadata->input_rows, metadata->moe_latent_size});
    auto exact_fc2_output =
        DeviceTensorFp32::Create({metadata->input_rows, metadata->hidden_size});
    auto direct_gate = UploadedLinearOp::Create(gate_weight_descriptor);
    auto direct_fc2 = UploadedLinearOp::Create(fc2_latent_weight_descriptor);
    std::unique_ptr<UploadedLinearOp> direct_fc1_dense;
    std::unique_ptr<ScaledFp8LinearOp> direct_fc1_fp8;
    if (metadata->fc1_latent_family == "scaled_fp8") {
      ScaledFp8LinearConfig direct_fc1_config;
      direct_fc1_config.output_rows = metadata->moe_latent_size;
      direct_fc1_config.input_cols = metadata->hidden_size;
      direct_fc1_config.packed_weight_data = fc1_latent_weight.data();
      direct_fc1_config.packed_weight_nbytes = fc1_latent_weight.size();
      direct_fc1_config.weight_scale = fc1_latent_weight_scale[0];
      direct_fc1_config.input_scale = fc1_latent_input_scale[0];
      direct_fc1_fp8 = ScaledFp8LinearOp::Create(direct_fc1_config);
    } else {
      direct_fc1_dense = UploadedLinearOp::Create(fc1_latent_dense_descriptor);
    }
    if (!expect(
            exact_norm_input != nullptr &&
                exact_gate_output != nullptr &&
                exact_fc1_output != nullptr &&
                exact_routed_input != nullptr &&
                exact_fc2_output != nullptr &&
                direct_gate != nullptr &&
                direct_gate->valid() &&
                ((direct_fc1_fp8 != nullptr && direct_fc1_fp8->valid()) ||
                 (direct_fc1_dense != nullptr && direct_fc1_dense->valid())) &&
                direct_fc2 != nullptr &&
                direct_fc2->valid(),
            "exact-input component operators should build")) {
      return false;
    }
    if (!expect(
            exact_norm_input->CopyFromHost(
                expected_norm_output.data(),
                expected_norm_output.size()),
            "exact norm input should upload") ||
        !expect(
            exact_routed_input->CopyFromHost(
                expected_routed_latent_output.data(),
                expected_routed_latent_output.size()),
            "exact routed latent input should upload") ||
        !expect(
            direct_gate->Run(*cublas, &heuristic_cache, *exact_norm_input, exact_gate_output.get()),
            "exact-input gate projection should run") ||
        !expect(
            (direct_fc1_fp8 != nullptr
                 ? direct_fc1_fp8->Run(*cublas, &heuristic_cache, *exact_norm_input, exact_fc1_output.get())
                 : direct_fc1_dense->Run(*cublas, &heuristic_cache, *exact_norm_input, exact_fc1_output.get())),
            "exact-input fc1 projection should run") ||
        !expect(
            direct_fc2->Run(*cublas, &heuristic_cache, *exact_routed_input, exact_fc2_output.get()),
            "exact-input fc2 projection should run")) {
      return false;
    }
    std::vector<float> exact_gate_host(expected_router_logits.size(), 0.0f);
    std::vector<float> exact_fc1_host(expected_fc1_latent_output.size(), 0.0f);
    std::vector<float> exact_fc2_host(metadata->input_rows * metadata->hidden_size, 0.0f);
    std::vector<float> expected_projected_routed_full(metadata->input_rows * metadata->hidden_size, 0.0f);
    for (std::size_t i = 0; i < expected_projected_routed_full.size(); ++i) {
      expected_projected_routed_full[i] = expected_mixer_output[i] - expected_shared_output[i];
    }
    if (!expect(
            exact_gate_output->CopyToHost(exact_gate_host.data(), exact_gate_host.size()),
            "exact gate output should download") ||
        !expect(
            exact_fc1_output->CopyToHost(exact_fc1_host.data(), exact_fc1_host.size()),
            "exact fc1 output should download") ||
        !expect(
            exact_fc2_output->CopyToHost(exact_fc2_host.data(), exact_fc2_host.size()),
            "exact fc2 output should download")) {
      return false;
    }
    direct_gate_diff = max_abs_diff(exact_gate_host, expected_router_logits);
    direct_fc1_diff = max_abs_diff(exact_fc1_host, expected_fc1_latent_output);
    direct_fc2_diff = max_abs_diff(exact_fc2_host, expected_projected_routed_full);
  }

  std::cout << "expert_layer_oracle_test:"
            << " norm_diff=" << norm_diff
            << " router_diff=" << router_diff
            << " selection_weight_diff=" << max_selection_weight_diff
            << " fc1_latent_diff=" << fc1_latent_diff
            << " routed_diff=" << routed_diff
            << " routed_activated_hidden_diff=" << routed_activated_hidden_diff
            << " routed_activated_hidden_expert_index="
            << (routed_activated_hidden_expert_index == std::numeric_limits<std::size_t>::max()
                    ? static_cast<std::size_t>(0)
                    : routed_activated_hidden_expert_index)
            << " routed_expert_output_diff=" << routed_expert_output_diff
            << " routed_expert_output_expert_index="
            << (routed_expert_output_expert_index == std::numeric_limits<std::size_t>::max()
                    ? static_cast<std::size_t>(0)
                    : routed_expert_output_expert_index)
            << " routed_contribution_diff=" << routed_contribution_diff
            << " routed_contribution_expert_index="
            << (routed_contribution_expert_index == std::numeric_limits<std::size_t>::max()
                    ? static_cast<std::size_t>(0)
                    : routed_contribution_expert_index)
            << " projected_routed_diff=" << projected_routed_diff
            << " projected_routed_rel_l2=" << projected_routed_rel_l2
            << " shared_diff=" << shared_diff
            << " mixer_diff=" << mixer_diff
            << " mixer_rel_l2=" << mixer_rel_l2
            << " batch_vs_sequential_diff=" << batch_vs_sequential_diff
            << " direct_gate_diff=" << direct_gate_diff
            << " direct_fc1_diff=" << direct_fc1_diff
            << " direct_fc2_diff=" << direct_fc2_diff
            << " final_diff=" << final_diff
            << " final_rel_l2=" << final_rel_l2
            << "\n";

  if (!expect(
          expected_norm_output.empty() || norm_diff <= 1.0e-5f,
          "norm output should match oracle when present") ||
      !expect(router_diff <= 2.0e-5f, "router logits should match oracle") ||
      !expect(max_selection_weight_diff <= 1.0e-5f, "selected expert weights should match oracle") ||
      !expect(fc1_latent_diff <= 1.0e-3f, "fc1 latent output should match oracle") ||
      !expect(routed_diff <= 1.0e-3f, "routed latent output should match oracle") ||
      !expect(routed_activated_hidden_diff <= 1.0e-3f, "routed expert activated hidden should match oracle") ||
      !expect(routed_expert_output_diff <= 1.0e-3f, "routed expert outputs should match oracle") ||
      !expect(routed_contribution_diff <= 1.0e-3f, "routed expert contributions should match oracle") ||
      !expect(projected_routed_diff <= 1.0e-3f, "projected routed output should match oracle") ||
      !expect(shared_diff <= 1.0e-3f, "shared output should match oracle") ||
      !expect(mixer_diff <= 1.0e-3f, "mixer output should match oracle") ||
      !expect(batch_vs_sequential_diff <= 1.0e-3f, "batched and row-by-row expert outputs should match oracle") ||
      !expect(final_diff <= 1.0e-3f, "final output should match oracle")) {
    return false;
  }

  const bool require_batched_follow_up =
      std::string(NEMOTRON_EXPERT_LAYER_ORACLE_FIXTURE_ROOT).find("runtime_input") == std::string::npos;

  if (metadata->input_rows != 1) {
    return true;
  }

  constexpr std::size_t kBatchTokens = 3;
  std::vector<float> batch_input(kBatchTokens * metadata->hidden_size, 0.0f);
  for (std::size_t token = 0; token < kBatchTokens; ++token) {
    for (std::size_t i = 0; i < metadata->hidden_size; ++i) {
      batch_input[token * metadata->hidden_size + i] =
          input_hidden[i] + (0.001f * static_cast<float>(token)) - (0.00003f * static_cast<float>(i % 13));
    }
  }

  auto batch_input_tensor = DeviceTensorFp32::Create({kBatchTokens, metadata->hidden_size});
  auto batch_output_tensor = DeviceTensorFp32::Create({kBatchTokens, metadata->hidden_size});
  const bool batch_setup_ok =
      expect(batch_input_tensor != nullptr && batch_input_tensor->valid(), "batched expert input tensor should allocate") &&
      expect(batch_output_tensor != nullptr && batch_output_tensor->valid(), "batched expert output tensor should allocate") &&
      expect(batch_input_tensor->CopyFromHost(batch_input.data(), batch_input.size()), "batched expert input should upload");
  const bool batch_run_ok =
      batch_setup_ok &&
      slice->Run(*cublas, &heuristic_cache, *batch_input_tensor, batch_output_tensor.get(), nullptr);
  if (!batch_setup_ok ||
      (!batch_run_ok &&
       require_batched_follow_up &&
       !expect(false, "batched expert slice should run"))) {
    return false;
  }
  if (!batch_run_ok) {
    std::cout << "expert_layer_oracle_test: note=batch_follow_up_skipped_for_runtime_input_fixture\n";
    return true;
  }

  auto sequential_input_tensor = DeviceTensorFp32::Create({1, metadata->hidden_size});
  auto sequential_output_tensor = DeviceTensorFp32::Create({1, metadata->hidden_size});
  if (!expect(sequential_input_tensor != nullptr && sequential_input_tensor->valid(), "sequential expert input tensor should allocate") ||
      !expect(sequential_output_tensor != nullptr && sequential_output_tensor->valid(), "sequential expert output tensor should allocate")) {
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
            "sequential expert row input should upload") ||
        !expect(
            slice->Run(
                *cublas,
                &heuristic_cache,
                *sequential_input_tensor,
                sequential_output_tensor.get(),
                nullptr),
            "sequential expert slice should run") ||
        !expect(
            sequential_output_tensor->CopyToHost(single_output.data(), single_output.size()),
            "sequential expert row output should download")) {
      return false;
    }
    std::copy(
        single_output.begin(),
        single_output.end(),
        sequential_outputs.begin() + (token * metadata->hidden_size));
  }

  std::vector<float> batch_outputs(batch_input.size(), 0.0f);
  if (!expect(
          batch_output_tensor->CopyToHost(batch_outputs.data(), batch_outputs.size()),
          "batched expert output should download")) {
    return false;
  }

  return expect(
      max_abs_diff(batch_outputs, sequential_outputs) <= 1.0e-3f,
      "batched expert outputs should match repeated single-row execution");
}

}  // namespace

int main() {
  return run_expert_layer_fixture() ? 0 : 1;
}
