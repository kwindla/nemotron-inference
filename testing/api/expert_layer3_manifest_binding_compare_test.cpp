#include "nemotron/expert_layer.h"
#include "nemotron/runtime_environment.h"
#include "nemotron/single_token_forward_model.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <string>
#include <vector>

namespace {

constexpr std::size_t GiB(std::size_t value) {
  return value * 1024ull * 1024ull * 1024ull;
}

struct FixtureMetadata {
  std::size_t layer_index = 0;
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
  return std::string(
      std::istreambuf_iterator<char>(input),
      std::istreambuf_iterator<char>());
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

std::optional<std::size_t> parse_json_uint_field(
    const std::string& json,
    const std::string& key) {
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

std::optional<FixtureMetadata> load_metadata(
    const std::filesystem::path& root) {
  const std::string json = read_text_file(root / "metadata.json");
  const auto layer_index = parse_json_uint_field(json, "layer_index");
  if (!layer_index) {
    return std::nullopt;
  }
  FixtureMetadata metadata;
  metadata.layer_index = *layer_index;
  return metadata;
}

float max_abs_diff(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return INFINITY;
  }
  float diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    diff = std::max(diff, std::fabs(lhs[i] - rhs[i]));
  }
  return diff;
}

nemotron::RuntimeBootstrapOptions make_options() {
  nemotron::RuntimeBootstrapOptions options;
  options.service_target.total_memory_bytes = GiB(128);
  options.service_target.weights_bytes = GiB(100);
  options.service_target.workspace_bytes = GiB(8);
  options.service_target.graph_bytes = GiB(4);
  options.service_target.safety_headroom_bytes = GiB(4);
  options.service_target.target_active_requests = 1;
  options.service_target.target_context_tokens = 8;
  options.use_fp16_mamba_state = false;
  options.reusable_node_metadata_bytes = 4096;
  options.verify_manifest_files = false;
  options.materialize_weight_arena = false;
  return options;
}

std::optional<std::vector<float>> read_kernel_descriptor_fp32(
    const nemotron::KernelTensorDescriptor& descriptor) {
  if (descriptor.packed_data == nullptr) {
    return std::nullopt;
  }
  std::size_t count = 1;
  for (std::size_t dim : descriptor.logical_shape) {
    count *= dim;
  }
  std::vector<float> output(count, 0.0f);
  if (descriptor.storage_dtype == "fp32" ||
      descriptor.storage_dtype == "float32" ||
      descriptor.storage_dtype == "float") {
    if (descriptor.packed_nbytes != count * sizeof(float)) {
      return std::nullopt;
    }
    std::memcpy(output.data(), descriptor.packed_data, descriptor.packed_nbytes);
    return output;
  }
  if (descriptor.storage_dtype == "bf16" ||
      descriptor.storage_dtype == "bfloat16") {
    if (descriptor.packed_nbytes != count * sizeof(__nv_bfloat16)) {
      return std::nullopt;
    }
    const auto* src =
        reinterpret_cast<const __nv_bfloat16*>(descriptor.packed_data);
    for (std::size_t i = 0; i < count; ++i) {
      output[i] = __bfloat162float(src[i]);
    }
    return output;
  }
  return std::nullopt;
}

std::optional<std::vector<float>> read_gemm_descriptor_fp32(
    const nemotron::GemmDescriptor& descriptor) {
  if (descriptor.packed_data == nullptr) {
    return std::nullopt;
  }
  const std::size_t count = descriptor.output_rows * descriptor.input_cols;
  std::vector<float> output(count, 0.0f);
  if (descriptor.storage_dtype == "fp32" ||
      descriptor.storage_dtype == "float32" ||
      descriptor.storage_dtype == "float") {
    if (descriptor.packed_nbytes != count * sizeof(float)) {
      return std::nullopt;
    }
    std::memcpy(output.data(), descriptor.packed_data, descriptor.packed_nbytes);
    return output;
  }
  if (descriptor.storage_dtype == "bf16" ||
      descriptor.storage_dtype == "bfloat16") {
    if (descriptor.packed_nbytes != count * sizeof(__nv_bfloat16)) {
      return std::nullopt;
    }
    const auto* src =
        reinterpret_cast<const __nv_bfloat16*>(descriptor.packed_data);
    for (std::size_t i = 0; i < count; ++i) {
      output[i] = __bfloat162float(src[i]);
    }
    return output;
  }
  if (descriptor.storage_dtype == "fp8_e4m3fn" ||
      descriptor.storage_dtype == "fp8_e4m3") {
    if (descriptor.packed_nbytes != count * sizeof(__nv_fp8_e4m3)) {
      return std::nullopt;
    }
    const auto* src =
        reinterpret_cast<const __nv_fp8_e4m3*>(descriptor.packed_data);
    for (std::size_t i = 0; i < count; ++i) {
      output[i] = static_cast<float>(src[i]);
    }
    return output;
  }
  return std::nullopt;
}

std::optional<std::vector<float>> read_raw_fp32_bytes(
    const std::uint8_t* data,
    std::size_t nbytes) {
  if (data == nullptr || nbytes % sizeof(float) != 0) {
    return std::nullopt;
  }
  std::vector<float> output(nbytes / sizeof(float), 0.0f);
  std::memcpy(output.data(), data, nbytes);
  return output;
}

bool compare_exact_bytes(
    const std::string& label,
    const std::uint8_t* actual,
    std::size_t actual_nbytes,
    const std::vector<std::uint8_t>& expected) {
  if (!expect(actual != nullptr, label + " should have bytes")) {
    return false;
  }
  if (!expect(
          actual_nbytes == expected.size(),
          label + " byte count should match fixture")) {
    return false;
  }
  const bool same =
      std::memcmp(actual, expected.data(), expected.size()) == 0;
  return expect(same, label + " bytes should match fixture");
}

bool compare_float_vectors(
    const std::string& label,
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float tolerance) {
  const float diff = max_abs_diff(actual, expected);
  std::cout << "expert_layer3_manifest_binding_compare_test: "
            << label << "_diff=" << diff << "\n";
  return expect(diff <= tolerance, label + " should match fixture");
}

bool run_expert_layer3_manifest_binding_compare() {
  const char* manifest_env = std::getenv("NEMOTRON_FORWARD_MANIFEST");
  if (manifest_env == nullptr || std::string(manifest_env).empty()) {
    std::cout << "expert_layer3_manifest_binding_compare_test: SKIP (NEMOTRON_FORWARD_MANIFEST is unset)\n";
    return true;
  }
  const std::filesystem::path manifest_path(manifest_env);
  if (!std::filesystem::exists(manifest_path)) {
    std::cout << "expert_layer3_manifest_binding_compare_test: SKIP (manifest path does not exist)\n";
    return true;
  }

  const std::filesystem::path fixture_root(
      NEMOTRON_EXPERT_LAYER_PREFIX_INPUT_FIXTURE_ROOT);
  const auto metadata = load_metadata(fixture_root);
  if (!expect(metadata.has_value(), "fixture metadata should load")) {
    return false;
  }

  const auto environment =
      nemotron::RuntimeEnvironment::BuildFromManifestFile(
          manifest_path,
          make_options());
  if (!expect(static_cast<bool>(environment), "runtime environment should build") ||
      !expect(environment->has_model_schedule(), "model schedule should exist") ||
      !expect(environment->has_kernel_catalog(), "kernel catalog should exist") ||
      !expect(environment->has_gemm_catalog(), "gemm catalog should exist")) {
    return false;
  }

  const auto* layer =
      environment->model_schedule()->FindLayer(metadata->layer_index);
  if (!expect(layer != nullptr, "target layer should exist")) {
    return false;
  }

  const auto bindings = nemotron::BuildExpertLayerBindings(
      *layer,
      *environment->kernel_catalog(),
      *environment->gemm_catalog(),
      nemotron::KnownNemotron3Super120BA12BConfig().n_routed_experts);
  if (!expect(bindings.has_value(), "expert bindings should build")) {
    return false;
  }

  if (!expect(bindings->gate_weight != nullptr, "gate weight binding should exist") ||
      !expect(bindings->gate_score_correction_bias != nullptr, "gate bias binding should exist") ||
      !expect(bindings->fc1_latent_kernel_weight != nullptr, "fc1 latent kernel binding should exist") ||
      !expect(bindings->fc1_latent_weight_scale != nullptr, "fc1 latent weight scale should exist") ||
      !expect(bindings->fc1_latent_input_scale != nullptr, "fc1 latent input scale should exist") ||
      !expect(bindings->fc2_latent_weight != nullptr, "fc2 latent binding should exist") ||
      !expect(bindings->shared_up_kernel_weight != nullptr, "shared up kernel binding should exist") ||
      !expect(bindings->shared_up_weight_scale != nullptr, "shared up weight scale should exist") ||
      !expect(bindings->shared_up_input_scale != nullptr, "shared up input scale should exist") ||
      !expect(bindings->shared_down_kernel_weight != nullptr, "shared down kernel binding should exist") ||
      !expect(bindings->shared_down_weight_scale != nullptr, "shared down weight scale should exist") ||
      !expect(bindings->shared_down_input_scale != nullptr, "shared down input scale should exist")) {
    return false;
  }

  if (!compare_float_vectors(
          "norm_weight",
          *read_kernel_descriptor_fp32(*bindings->input_norm_weight),
          read_float_file(fixture_root / "norm_weight_fp32.bin"),
          0.0f) ||
      !compare_float_vectors(
          "gate_weight",
          *read_gemm_descriptor_fp32(*bindings->gate_weight),
          read_float_file(fixture_root / "gate_weight_fp32.bin"),
          0.0f) ||
      !compare_float_vectors(
          "gate_score_correction_bias",
          *read_kernel_descriptor_fp32(*bindings->gate_score_correction_bias),
          read_float_file(fixture_root / "gate_score_correction_bias_fp32.bin"),
          0.0f) ||
      !compare_exact_bytes(
          "fc1_latent_weight",
          bindings->fc1_latent_kernel_weight->packed_data,
          bindings->fc1_latent_kernel_weight->packed_nbytes,
          read_file_bytes(fixture_root / "fc1_latent_weight_fp8.bin")) ||
      !compare_float_vectors(
          "fc1_latent_weight_scale",
          *read_kernel_descriptor_fp32(*bindings->fc1_latent_weight_scale),
          read_float_file(fixture_root / "fc1_latent_weight_scale_fp32.bin"),
          0.0f) ||
      !compare_float_vectors(
          "fc1_latent_input_scale",
          *read_kernel_descriptor_fp32(*bindings->fc1_latent_input_scale),
          read_float_file(fixture_root / "fc1_latent_input_scale_fp32.bin"),
          0.0f) ||
      !compare_float_vectors(
          "fc2_latent_weight",
          *read_gemm_descriptor_fp32(*bindings->fc2_latent_weight),
          read_float_file(fixture_root / "fc2_latent_weight_fp32.bin"),
          0.0f) ||
      !compare_exact_bytes(
          "shared_up_weight",
          bindings->shared_up_kernel_weight->packed_data,
          bindings->shared_up_kernel_weight->packed_nbytes,
          read_file_bytes(fixture_root / "shared_up_weight_fp8.bin")) ||
      !compare_float_vectors(
          "shared_up_weight_scale",
          *read_kernel_descriptor_fp32(*bindings->shared_up_weight_scale),
          read_float_file(fixture_root / "shared_up_weight_scale_fp32.bin"),
          0.0f) ||
      !compare_float_vectors(
          "shared_up_input_scale",
          *read_kernel_descriptor_fp32(*bindings->shared_up_input_scale),
          read_float_file(fixture_root / "shared_up_input_scale_fp32.bin"),
          0.0f) ||
      !compare_exact_bytes(
          "shared_down_weight",
          bindings->shared_down_kernel_weight->packed_data,
          bindings->shared_down_kernel_weight->packed_nbytes,
          read_file_bytes(fixture_root / "shared_down_weight_fp8.bin")) ||
      !compare_float_vectors(
          "shared_down_weight_scale",
          *read_kernel_descriptor_fp32(*bindings->shared_down_weight_scale),
          read_float_file(fixture_root / "shared_down_weight_scale_fp32.bin"),
          0.0f) ||
      !compare_float_vectors(
          "shared_down_input_scale",
          *read_kernel_descriptor_fp32(*bindings->shared_down_input_scale),
          read_float_file(fixture_root / "shared_down_input_scale_fp32.bin"),
          0.0f)) {
    return false;
  }

  std::vector<int> expert_ids;
  for (const auto& entry : std::filesystem::directory_iterator(fixture_root)) {
    if (!entry.is_directory()) {
      continue;
    }
    const std::string name = entry.path().filename().string();
    if (name.rfind("expert_", 0) != 0) {
      continue;
    }
    expert_ids.push_back(std::stoi(name.substr(7)));
  }
  std::sort(expert_ids.begin(), expert_ids.end());

  for (int expert_id : expert_ids) {
    const auto& pair =
        bindings->routed_experts[static_cast<std::size_t>(expert_id)];
    if (!expect(pair.up_proj != nullptr, "routed expert up binding should exist") ||
        !expect(pair.down_proj != nullptr, "routed expert down binding should exist")) {
      return false;
    }
    const std::filesystem::path expert_dir =
        fixture_root /
        (std::string("expert_") +
         (expert_id < 100 ? (expert_id < 10 ? "00" : "0") : "") +
         std::to_string(expert_id));
    if (!compare_exact_bytes(
            "expert_" + std::to_string(expert_id) + "_up_weight",
            pair.up_proj->packed_data,
            pair.up_proj->packed_nbytes,
            read_file_bytes(expert_dir / "up_proj_weight_packed.bin")) ||
        !compare_exact_bytes(
            "expert_" + std::to_string(expert_id) + "_up_scales",
            pair.up_proj->block_scales_data,
            pair.up_proj->block_scales_nbytes,
            read_file_bytes(expert_dir / "up_proj_weight_block_scales.bin")) ||
        !compare_float_vectors(
            "expert_" + std::to_string(expert_id) + "_up_tensor_scale",
            *read_raw_fp32_bytes(
                pair.up_proj->tensor_scale_data,
                pair.up_proj->tensor_scale_nbytes),
            read_float_file(expert_dir / "up_proj_weight_tensor_scale_fp32.bin"),
            0.0f) ||
        !compare_exact_bytes(
            "expert_" + std::to_string(expert_id) + "_down_weight",
            pair.down_proj->packed_data,
            pair.down_proj->packed_nbytes,
            read_file_bytes(expert_dir / "down_proj_weight_packed.bin")) ||
        !compare_exact_bytes(
            "expert_" + std::to_string(expert_id) + "_down_scales",
            pair.down_proj->block_scales_data,
            pair.down_proj->block_scales_nbytes,
            read_file_bytes(expert_dir / "down_proj_weight_block_scales.bin")) ||
        !compare_float_vectors(
            "expert_" + std::to_string(expert_id) + "_down_tensor_scale",
            *read_raw_fp32_bytes(
                pair.down_proj->tensor_scale_data,
                pair.down_proj->tensor_scale_nbytes),
            read_float_file(expert_dir / "down_proj_weight_tensor_scale_fp32.bin"),
            0.0f)) {
      return false;
    }
  }

  return true;
}

}  // namespace

int main() {
  return run_expert_layer3_manifest_binding_compare() ? 0 : 1;
}
