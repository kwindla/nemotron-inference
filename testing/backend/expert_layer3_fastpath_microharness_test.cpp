#include "nemotron/expert_layer.h"
#include "nemotron/flashinfer_moe_backend.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <regex>
#include <string>
#include <vector>

namespace {

using nemotron::CublasLtHandle;
using nemotron::DeviceTensorFp32;
using nemotron::ExpertLayerBindings;
using nemotron::ExpertLayerConfig;
using nemotron::ExpertLayerSlice;
using nemotron::GemmDescriptor;
using nemotron::GemmHeuristicCache;
using nemotron::GemmKernelFamily;
using nemotron::KernelTensorDescriptor;

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
  std::string fc2_latent_family = "dense";
  std::string shared_up_family = "scaled_fp8";
  std::string shared_down_family;
  std::string routed_nvfp4_activation_packing_mode;
  std::string routed_nvfp4_weight_tensor_scale_contract;
  std::string shared_down_nvfp4_activation_packing_mode;
  std::string shared_down_nvfp4_weight_tensor_scale_contract;
  bool shared_down_nvfp4_input_scale_present = false;
};

struct RoutedExpertFixture {
  std::int32_t expert_index = -1;
  std::vector<std::uint8_t> up_weight_packed;
  std::vector<std::uint8_t> up_weight_block_scales;
  std::vector<float> up_weight_tensor_scale;
  std::vector<float> up_input_scale;
  std::vector<std::uint8_t> down_weight_packed;
  std::vector<std::uint8_t> down_weight_block_scales;
  std::vector<float> down_weight_tensor_scale;
  std::vector<float> down_input_scale;
  GemmDescriptor up_descriptor;
  GemmDescriptor down_descriptor;
  KernelTensorDescriptor up_input_scale_descriptor;
  KernelTensorDescriptor down_input_scale_descriptor;
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
  return std::vector<std::uint8_t>(
      std::istreambuf_iterator<char>(input),
      std::istreambuf_iterator<char>());
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

std::vector<std::int32_t> read_int32_file(const std::filesystem::path& path) {
  const std::vector<std::uint8_t> bytes = read_file_bytes(path);
  if (bytes.size() % sizeof(std::int32_t) != 0) {
    return {};
  }
  std::vector<std::int32_t> values(bytes.size() / sizeof(std::int32_t), 0);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

void write_float_file(const std::filesystem::path& path, const std::vector<float>& values) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream output(path, std::ios::binary);
  output.write(
      reinterpret_cast<const char*>(values.data()),
      static_cast<std::streamsize>(values.size() * sizeof(float)));
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

std::optional<float> parse_routed_expert_input_scale(
    const std::string& json,
    std::size_t expert_index,
    const char* proj_name) {
  const std::string selected_anchor = "\"selected_experts\"";
  const std::size_t selected_pos = json.find(selected_anchor);
  if (selected_pos == std::string::npos) {
    return std::nullopt;
  }
  const std::string expert_anchor = "\"" + std::to_string(expert_index) + "\"";
  const std::size_t expert_pos = json.find(expert_anchor, selected_pos);
  if (expert_pos == std::string::npos) {
    return std::nullopt;
  }
  const std::string proj_anchor = "\"" + std::string(proj_name) + "\"";
  const std::size_t proj_pos = json.find(proj_anchor, expert_pos);
  if (proj_pos == std::string::npos) {
    return std::nullopt;
  }
  const std::string tail = json.substr(proj_pos, 2048);
  return parse_json_float_field(tail, "input_scale");
}

std::optional<FixtureMetadata> load_metadata(const std::filesystem::path& root) {
  const std::string json = read_text_file(root / "metadata.json");
  FixtureMetadata metadata;
  const auto layer_index = parse_json_uint_field(json, "layer_index");
  const auto input_rows = parse_json_uint_field(json, "input_rows");
  const auto hidden_size = parse_json_uint_field(json, "hidden_size");
  const auto moe_latent_size = parse_json_uint_field(json, "moe_latent_size");
  const auto routed_expert_intermediate_size =
      parse_json_uint_field(json, "routed_expert_intermediate_size");
  const auto shared_expert_intermediate_size =
      parse_json_uint_field(json, "shared_expert_intermediate_size");
  const auto n_routed_experts = parse_json_uint_field(json, "n_routed_experts");
  const auto top_k = parse_json_uint_field(json, "top_k");
  const auto n_group = parse_json_uint_field(json, "n_group");
  const auto topk_group = parse_json_uint_field(json, "topk_group");
  const auto routed_scaling_factor = parse_json_float_field(json, "routed_scaling_factor");
  const auto rms_epsilon = parse_json_float_field(json, "rms_epsilon");
  const auto norm_topk_prob = parse_json_bool_field(json, "norm_topk_prob");
  const auto fc1_latent_family = parse_json_string_field(json, "fc1_latent_family");
  const auto fc2_latent_family = parse_json_string_field(json, "fc2_latent_family");
  const auto shared_up_family = parse_json_string_field(json, "shared_up_family");
  const auto shared_down_family = parse_json_string_field(json, "shared_down_family");
  const auto routed_nvfp4_activation_packing_mode =
      parse_json_string_field(json, "routed_nvfp4_activation_packing_mode");
  const auto routed_nvfp4_weight_tensor_scale_contract =
      parse_json_string_field(json, "routed_nvfp4_weight_tensor_scale_contract");
  const auto shared_down_nvfp4_activation_packing_mode =
      parse_json_string_field(json, "shared_down_nvfp4_activation_packing_mode");
  const auto shared_down_nvfp4_weight_tensor_scale_contract =
      parse_json_string_field(json, "shared_down_nvfp4_weight_tensor_scale_contract");
  const auto shared_down_nvfp4_input_scale_present =
      parse_json_bool_field(json, "shared_down_nvfp4_input_scale_present");
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
  if (fc2_latent_family) {
    metadata.fc2_latent_family = *fc2_latent_family;
  }
  if (shared_up_family) {
    metadata.shared_up_family = *shared_up_family;
  }
  metadata.shared_down_family = *shared_down_family;
  if (routed_nvfp4_activation_packing_mode) {
    metadata.routed_nvfp4_activation_packing_mode = *routed_nvfp4_activation_packing_mode;
  }
  if (routed_nvfp4_weight_tensor_scale_contract) {
    metadata.routed_nvfp4_weight_tensor_scale_contract =
        *routed_nvfp4_weight_tensor_scale_contract;
  }
  if (shared_down_nvfp4_activation_packing_mode) {
    metadata.shared_down_nvfp4_activation_packing_mode =
        *shared_down_nvfp4_activation_packing_mode;
  }
  if (shared_down_nvfp4_weight_tensor_scale_contract) {
    metadata.shared_down_nvfp4_weight_tensor_scale_contract =
        *shared_down_nvfp4_weight_tensor_scale_contract;
  }
  if (shared_down_nvfp4_input_scale_present) {
    metadata.shared_down_nvfp4_input_scale_present =
        *shared_down_nvfp4_input_scale_present;
  }
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
    return std::numeric_limits<float>::infinity();
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
  }
  return max_diff;
}

void print_diff_report(
    const std::string& comparison,
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float tolerance) {
  const float diff = max_abs_diff(actual, expected);
  std::cerr << "comparison: " << comparison << "\n";
  std::cerr << "max_abs_diff: " << diff << "\n";
  std::cerr << "within_tolerance: " << (diff <= tolerance ? "yes" : "no") << "\n";
}

bool run_expert_layer3_fastpath_microharness() {
  constexpr float kTolerance = 1.0e-3f;

  const auto cublas = CublasLtHandle::Create();
  if (!cublas || !cublas->valid()) {
    std::cout << "expert_layer3_fastpath_microharness_test: SKIP (no CUDA device)\n";
    return true;
  }

  const std::filesystem::path fixture_root = resolve_fixture_root(
      "NEMOTRON_EXPERT_LAYER_ORACLE_FIXTURE_ROOT_OVERRIDE",
      NEMOTRON_EXPERT_LAYER_ORACLE_FIXTURE_ROOT);
  const auto metadata = load_metadata(fixture_root);
  if (!expect(metadata.has_value(), "expert layer fixture metadata should load")) {
    return false;
  }

  const std::vector<float> input_hidden = read_float_file(fixture_root / "input_hidden_fp32.bin");
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
  const std::vector<std::uint8_t> fc2_latent_weight_fp8 =
      read_file_bytes(fixture_root / "fc2_latent_weight_fp8.bin");
  const std::vector<float> fc2_latent_weight_scale =
      read_float_file(fixture_root / "fc2_latent_weight_scale_fp32.bin");
  const std::vector<float> fc2_latent_input_scale =
      read_float_file(fixture_root / "fc2_latent_input_scale_fp32.bin");
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
  const std::vector<float> expected_final_output =
      read_float_file(fixture_root / "expected_final_output_fp32.bin");
  const std::vector<std::int32_t> expected_selected_expert_indices =
      read_int32_file(fixture_root / "expected_selected_expert_indices_u32.bin");
  const std::string metadata_json = read_text_file(fixture_root / "metadata.json");

  const bool routed_contract_is_effective =
      metadata->routed_nvfp4_weight_tensor_scale_contract ==
      "effective_tensor_scale = input_scale * weight_scale_2";
  const bool routed_contract_is_raw =
      metadata->routed_nvfp4_weight_tensor_scale_contract == "raw_weight_scale_2";
  if (!expect(
          input_hidden.size() == metadata->input_rows * metadata->hidden_size,
          "input hidden size should match metadata") ||
      !expect(norm_weight.size() == metadata->hidden_size, "norm weight size should match metadata") ||
      !expect(
          gate_weight.size() == metadata->n_routed_experts * metadata->hidden_size,
          "gate weight size should match metadata") ||
      !expect(
          gate_score_correction_bias.size() == metadata->n_routed_experts,
          "gate correction bias size should match metadata") ||
      !expect(
          metadata->routed_nvfp4_activation_packing_mode == "dynamic_runtime",
          "routed NVFP4 activation packing mode should be dynamic_runtime") ||
      !expect(
          routed_contract_is_effective || routed_contract_is_raw,
          "routed NVFP4 tensor scale contract should be supported") ||
      !expect(
          expected_selected_expert_indices.size() == metadata->input_rows * metadata->top_k,
          "selected expert count should match top-k") ||
      !expect(
          expected_final_output.size() == metadata->input_rows * metadata->hidden_size,
          "expected final output size should match metadata") ||
      !expect(
          (metadata->fc1_latent_family == "scaled_fp8" &&
           fc1_latent_weight.size() == metadata->moe_latent_size * metadata->hidden_size &&
           fc1_latent_weight_scale.size() == 1 &&
           fc1_latent_input_scale.size() == 1) ||
              (metadata->fc1_latent_family == "dense" &&
               fc1_latent_weight_fp32.size() ==
                   metadata->moe_latent_size * metadata->hidden_size),
          "fc1 latent weights should match metadata") ||
      !expect(
          (metadata->fc2_latent_family == "scaled_fp8" &&
           fc2_latent_weight_fp8.size() == metadata->hidden_size * metadata->moe_latent_size &&
           fc2_latent_weight_scale.size() == 1 &&
           fc2_latent_input_scale.size() == 1) ||
              (metadata->fc2_latent_family == "dense" &&
               fc2_latent_weight.size() ==
                   metadata->hidden_size * metadata->moe_latent_size),
          "fc2 latent weights should match metadata") ||
      !expect(
          (metadata->shared_up_family == "scaled_fp8" &&
           shared_up_weight.size() ==
               metadata->shared_expert_intermediate_size * metadata->hidden_size &&
           shared_up_weight_scale.size() == 1 &&
           shared_up_input_scale.size() == 1) ||
              (metadata->shared_up_family == "dense" &&
               shared_up_weight_fp32.size() ==
                   metadata->shared_expert_intermediate_size * metadata->hidden_size),
          "shared up weights should match metadata") ||
      !expect(
          (metadata->shared_down_family == "nvfp4" &&
           !shared_down_weight_packed.empty() &&
           !shared_down_weight_block_scales.empty() &&
           shared_down_weight_tensor_scale.size() == 1) ||
              (metadata->shared_down_family == "scaled_fp8" &&
               shared_down_weight_fp8.size() ==
                   metadata->hidden_size * metadata->shared_expert_intermediate_size &&
               shared_down_weight_scale.size() == 1 &&
               shared_down_input_scale.size() == 1) ||
              (metadata->shared_down_family == "dense" &&
               shared_down_weight_fp32.size() ==
                   metadata->hidden_size * metadata->shared_expert_intermediate_size),
          "shared down weights should match metadata")) {
    return false;
  }

  if (metadata->shared_down_family == "nvfp4") {
    const std::string expected_shared_down_contract =
        metadata->shared_down_nvfp4_input_scale_present
            ? "effective_tensor_scale = input_scale * weight_scale_2"
            : "raw_weight_scale_2";
    if (!expect(
            metadata->shared_down_nvfp4_activation_packing_mode == "dynamic_runtime",
            "shared down NVFP4 activation packing mode should be dynamic_runtime") ||
        !expect(
            metadata->shared_down_nvfp4_weight_tensor_scale_contract ==
                expected_shared_down_contract,
            "shared down NVFP4 tensor scale contract should match input-scale availability")) {
      return false;
    }
  }

  const std::string layer_prefix = "backbone.layers." + std::to_string(metadata->layer_index);
  const std::string mixer_prefix = layer_prefix + ".mixer";
  const auto input_norm_descriptor =
      make_fp32_descriptor(layer_prefix + ".norm.weight", norm_weight, {metadata->hidden_size});
  const auto gate_weight_descriptor =
      make_dense_descriptor(
          mixer_prefix + ".gate.weight",
          gate_weight,
          metadata->n_routed_experts,
          metadata->hidden_size);
  const auto gate_score_correction_bias_descriptor =
      make_fp32_descriptor(
          mixer_prefix + ".gate.e_score_correction_bias",
          gate_score_correction_bias,
          {metadata->n_routed_experts});
  const auto fc1_latent_weight_descriptor =
      make_fp8_descriptor(
          mixer_prefix + ".fc1_latent_proj.weight",
          fc1_latent_weight,
          {metadata->moe_latent_size, metadata->hidden_size});
  const auto fc1_latent_dense_descriptor =
      make_dense_descriptor(
          mixer_prefix + ".fc1_latent_proj.weight",
          fc1_latent_weight_fp32,
          metadata->moe_latent_size,
          metadata->hidden_size);
  const auto fc1_latent_weight_scale_descriptor =
      make_fp32_descriptor(
          mixer_prefix + ".fc1_latent_proj.weight_scale",
          fc1_latent_weight_scale,
          {1});
  const auto fc1_latent_input_scale_descriptor =
      make_fp32_descriptor(
          mixer_prefix + ".fc1_latent_proj.input_scale",
          fc1_latent_input_scale,
          {1});
  const auto fc2_latent_weight_descriptor =
      make_dense_descriptor(
          mixer_prefix + ".fc2_latent_proj.weight",
          fc2_latent_weight,
          metadata->hidden_size,
          metadata->moe_latent_size);
  const auto fc2_latent_fp8_descriptor =
      make_fp8_descriptor(
          mixer_prefix + ".fc2_latent_proj.weight",
          fc2_latent_weight_fp8,
          {metadata->hidden_size, metadata->moe_latent_size});
  const auto fc2_latent_weight_scale_descriptor =
      make_fp32_descriptor(
          mixer_prefix + ".fc2_latent_proj.weight_scale",
          fc2_latent_weight_scale,
          {1});
  const auto fc2_latent_input_scale_descriptor =
      make_fp32_descriptor(
          mixer_prefix + ".fc2_latent_proj.input_scale",
          fc2_latent_input_scale,
          {1});
  const auto shared_up_weight_descriptor =
      make_fp8_descriptor(
          mixer_prefix + ".shared_experts.up_proj.weight",
          shared_up_weight,
          {metadata->shared_expert_intermediate_size, metadata->hidden_size});
  const auto shared_up_dense_descriptor =
      make_dense_descriptor(
          mixer_prefix + ".shared_experts.up_proj.weight",
          shared_up_weight_fp32,
          metadata->shared_expert_intermediate_size,
          metadata->hidden_size);
  const auto shared_up_weight_scale_descriptor =
      make_fp32_descriptor(
          mixer_prefix + ".shared_experts.up_proj.weight_scale",
          shared_up_weight_scale,
          {1});
  const auto shared_up_input_scale_descriptor =
      make_fp32_descriptor(
          mixer_prefix + ".shared_experts.up_proj.input_scale",
          shared_up_input_scale,
          {1});
  const auto shared_down_nvfp4_descriptor =
      make_nvfp4_descriptor(
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
      make_fp32_descriptor(
          mixer_prefix + ".shared_experts.down_proj.weight_scale",
          shared_down_weight_scale,
          {1});
  const auto shared_down_input_scale_descriptor =
      make_fp32_descriptor(
          mixer_prefix + ".shared_experts.down_proj.input_scale",
          shared_down_input_scale,
          {1});

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
  if (metadata->fc2_latent_family == "scaled_fp8") {
    bindings.fc2_latent_kernel_weight = &fc2_latent_fp8_descriptor;
    bindings.fc2_latent_weight_scale = &fc2_latent_weight_scale_descriptor;
    bindings.fc2_latent_input_scale = &fc2_latent_input_scale_descriptor;
  } else if (metadata->fc2_latent_family == "dense") {
    bindings.fc2_latent_gemm_weight = &fc2_latent_weight_descriptor;
  } else {
    return expect(false, "fc2 latent family should be supported");
  }
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
  } else if (metadata->shared_down_family == "dense") {
    bindings.shared_down_gemm_weight = &shared_down_dense_descriptor;
  } else {
    return expect(false, "shared down family should be supported");
  }
  bindings.routed_experts.resize(metadata->n_routed_experts);

  std::vector<bool> routed_expert_needed(metadata->n_routed_experts, false);
  for (std::int32_t expert_index : expected_selected_expert_indices) {
    if (!expect(
            expert_index >= 0 &&
                static_cast<std::size_t>(expert_index) < metadata->n_routed_experts,
            "selected expert index should be in range")) {
      return false;
    }
    routed_expert_needed[static_cast<std::size_t>(expert_index)] = true;
  }

  for (std::size_t expert_index = 0; expert_index < routed_expert_needed.size(); ++expert_index) {
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
    expert.up_weight_tensor_scale =
        read_float_file(expert_dir / "up_proj_weight_tensor_scale_fp32.bin");
    expert.down_weight_packed = read_file_bytes(expert_dir / "down_proj_weight_packed.bin");
    expert.down_weight_block_scales =
        read_file_bytes(expert_dir / "down_proj_weight_block_scales.bin");
    expert.down_weight_tensor_scale =
        read_float_file(expert_dir / "down_proj_weight_tensor_scale_fp32.bin");
    if (routed_contract_is_raw) {
      const auto up_input_scale =
          parse_routed_expert_input_scale(metadata_json, expert_index, "up_proj");
      const auto down_input_scale =
          parse_routed_expert_input_scale(metadata_json, expert_index, "down_proj");
      if (!expect(
              up_input_scale.has_value() && down_input_scale.has_value(),
              "raw routed NVFP4 fixture should expose per-expert input scales")) {
        return false;
      }
      expert.up_input_scale = {*up_input_scale};
      expert.down_input_scale = {*down_input_scale};
    }
    if (!expect(
            expert.up_weight_tensor_scale.size() == 1,
            "expert up tensor scale should be scalar") ||
        !expect(
            expert.down_weight_tensor_scale.size() == 1,
            "expert down tensor scale should be scalar") ||
        !expect(
            !routed_contract_is_raw || expert.up_input_scale.size() == 1,
            "raw routed expert up input scale should be scalar") ||
        !expect(
            !routed_contract_is_raw || expert.down_input_scale.size() == 1,
            "raw routed expert down input scale should be scalar")) {
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
    if (routed_contract_is_raw) {
      expert.up_input_scale_descriptor = make_fp32_descriptor(
          mixer_prefix + ".experts." + std::to_string(expert_index) + ".up_proj.input_scale",
          expert.up_input_scale,
          {1});
      expert.down_input_scale_descriptor = make_fp32_descriptor(
          mixer_prefix + ".experts." + std::to_string(expert_index) + ".down_proj.input_scale",
          expert.down_input_scale,
          {1});
    }
    routed_experts.push_back(std::move(expert));
  }

  for (RoutedExpertFixture& expert : routed_experts) {
    auto& bound_expert = bindings.routed_experts[static_cast<std::size_t>(expert.expert_index)];
    bound_expert.up_proj = &expert.up_descriptor;
    bound_expert.down_proj = &expert.down_descriptor;
    if (routed_contract_is_raw) {
      bound_expert.up_input_scale = &expert.up_input_scale_descriptor;
      bound_expert.down_input_scale = &expert.down_input_scale_descriptor;
    }
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

  std::unique_ptr<ExpertLayerSlice> slice = ExpertLayerSlice::Create(layer_config, bindings);
  if (!expect(slice != nullptr && slice->valid(), "expert layer slice should be created")) {
    return false;
  }

  const std::string serving_path_identity = nemotron::GetServingPathIdentity();
  std::cerr << "serving_path_identity: " << serving_path_identity << "\n";
  if (!expect(
          serving_path_identity == "direct_custom_grouped_fused_nvfp4",
          "serving path identity should match direct custom grouped fused NVFP4")) {
    return false;
  }
  std::cerr << "graph_replay: disabled\n";

  auto batch_input = DeviceTensorFp32::Create({metadata->input_rows, metadata->hidden_size});
  auto batch_output = DeviceTensorFp32::Create({metadata->input_rows, metadata->hidden_size});
  if (!expect(batch_input != nullptr && batch_output != nullptr, "batched device tensors should allocate") ||
      !expect(
          batch_input->CopyFromHost(input_hidden.data(), input_hidden.size()),
          "batched input hidden should upload")) {
    return false;
  }

  GemmHeuristicCache heuristic_cache;
  if (!expect(
          slice->Run(*cublas, &heuristic_cache, *batch_input, batch_output.get(), nullptr),
          "batched fastpath expert layer run should succeed")) {
    return false;
  }

  std::vector<float> batch_output_host(batch_output->numel(), 0.0f);
  if (!expect(
          batch_output->CopyToHost(batch_output_host.data(), batch_output_host.size()),
          "batched output should download")) {
    return false;
  }

  auto row_input = DeviceTensorFp32::Create({1, metadata->hidden_size});
  auto row_output = DeviceTensorFp32::Create({1, metadata->hidden_size});
  if (!expect(row_input != nullptr && row_output != nullptr, "row device tensors should allocate")) {
    return false;
  }

  std::vector<float> sequential_output_host(input_hidden.size(), 0.0f);
  std::vector<float> row_input_host(metadata->hidden_size, 0.0f);
  std::vector<float> row_output_host(metadata->hidden_size, 0.0f);
  for (std::size_t row = 0; row < metadata->input_rows; ++row) {
    std::copy_n(
        input_hidden.data() + (row * metadata->hidden_size),
        metadata->hidden_size,
        row_input_host.data());
    if (!expect(
            row_input->CopyFromHost(row_input_host.data(), row_input_host.size()),
            "row input hidden should upload") ||
        !expect(
            slice->Run(*cublas, &heuristic_cache, *row_input, row_output.get(), nullptr),
            "sequential fastpath expert layer run should succeed") ||
        !expect(
            row_output->CopyToHost(row_output_host.data(), row_output_host.size()),
            "row output should download")) {
      return false;
    }
    std::copy(
        row_output_host.begin(),
        row_output_host.end(),
        sequential_output_host.begin() + (row * metadata->hidden_size));
  }

  std::cerr << "grouped_fastpath: all tokens used direct fused path\n";
  std::cerr << std::fixed << std::setprecision(9);
  print_diff_report("batched_vs_oracle_expected", batch_output_host, expected_final_output, kTolerance);
  print_diff_report(
      "sequential_vs_oracle_expected",
      sequential_output_host,
      expected_final_output,
      kTolerance);
  std::cerr << "comparison: batched_vs_sequential\n";
  std::cerr << "max_abs_diff: " << max_abs_diff(batch_output_host, sequential_output_host) << "\n";

  const char* dump_root_env = std::getenv("NEMOTRON_EXPERT_LAYER_DUMP_ROOT");
  if (dump_root_env != nullptr && std::string(dump_root_env).size() != 0) {
    const std::filesystem::path dump_root(dump_root_env);
    const std::filesystem::path batch_dump_path =
        dump_root / "expert_layer3_fastpath_microharness_batch_output_fp32.bin";
    const std::filesystem::path sequential_dump_path =
        dump_root / "expert_layer3_fastpath_microharness_sequential_output_fp32.bin";
    write_float_file(batch_dump_path, batch_output_host);
    write_float_file(sequential_dump_path, sequential_output_host);
    std::cerr << "dumped_output: " << batch_dump_path << "\n";
    std::cerr << "dumped_output: " << sequential_dump_path << "\n";
  }

  return true;
}

}  // namespace

int main() {
  return run_expert_layer3_fastpath_microharness() ? 0 : 1;
}
