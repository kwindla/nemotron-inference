#include "nemotron/expert_layer.h"
#include "nemotron/flashinfer_moe_backend.h"
#include "nemotron/runtime_environment.h"

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
#include <sstream>
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
using nemotron::RequestExecutionConfig;
using nemotron::RequestExecutionContext;
using nemotron::RuntimeBootstrapOptions;
using nemotron::RuntimeEnvironment;

constexpr std::size_t GiB(std::size_t value) {
  return value * 1024ull * 1024ull * 1024ull;
}

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

class ScopedStreamRedirect {
 public:
  ScopedStreamRedirect(std::ostream& stream, std::streambuf* new_buffer)
      : stream_(stream), old_buffer_(stream.rdbuf(new_buffer)) {}

  ~ScopedStreamRedirect() { stream_.rdbuf(old_buffer_); }

  ScopedStreamRedirect(const ScopedStreamRedirect&) = delete;
  ScopedStreamRedirect& operator=(const ScopedStreamRedirect&) = delete;

 private:
  std::ostream& stream_;
  std::streambuf* old_buffer_ = nullptr;
};

bool is_supported_nvfp4_activation_packing_mode(std::string_view mode) {
  return mode == "dynamic_runtime" || mode == "fixed_checkpoint_input_scale";
}

bool is_supported_nvfp4_tensor_scale_contract(std::string_view contract) {
  return contract == "raw_weight_scale_2" ||
         contract == "effective_tensor_scale = input_scale * weight_scale_2";
}

struct RuntimeScaleContractObservation {
  bool up_prepack_seen = false;
  bool up_packed_seen = false;
  bool down_prepack_seen = false;
  bool down_packed_seen = false;
  std::int32_t expert_index = -1;
  float up_input_scale = 0.0f;
  float up_tensor_scale = 0.0f;
  float latent_tensor_scale = 0.0f;
  float runtime_expected_latent_tensor_scale = 0.0f;
  bool runtime_latent_match = false;
  float runtime_latent_normalized_checkpoint_tensor_scale_candidate = 0.0f;
  float runtime_latent_inverse_candidate = 0.0f;
  bool runtime_latent_inverse_match = false;
  float runtime_up_actual_fused_alpha = 0.0f;
  float runtime_up_expected_fused_alpha = 0.0f;
  bool runtime_up_fused_alpha_match = false;
  float down_input_scale = 0.0f;
  float down_tensor_scale = 0.0f;
  float down_act_tensor_scale = 0.0f;
  float runtime_expected_down_act_tensor_scale = 0.0f;
  bool runtime_down_match = false;
  float runtime_down_normalized_checkpoint_tensor_scale_candidate = 0.0f;
  float runtime_down_inverse_candidate = 0.0f;
  bool runtime_down_inverse_match = false;
  float runtime_down_actual_fused_alpha = 0.0f;
  float runtime_down_expected_fused_alpha = 0.0f;
  bool runtime_down_fused_alpha_match = false;
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

template <typename T>
std::vector<T> slice_vector(
    const std::vector<T>& values,
    std::size_t offset,
    std::size_t count) {
  if (offset > values.size() || count > values.size() - offset) {
    return {};
  }
  return std::vector<T>(values.begin() + offset, values.begin() + offset + count);
}

std::string two_digit_stem(std::size_t value) {
  return std::string(value < 10 ? "0" : "") + std::to_string(value);
}

RuntimeBootstrapOptions make_runtime_bootstrap_options() {
  RuntimeBootstrapOptions options;
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

bool scale_contract_matches(float actual, float expected) {
  if (!std::isfinite(actual) || !std::isfinite(expected)) {
    return false;
  }
  const float tolerance =
      std::max(1.0e-6f, 1.0e-6f * std::max(std::fabs(actual), std::fabs(expected)));
  return std::fabs(actual - expected) <= tolerance;
}

bool parse_runtime_scale_contract_log(
    const std::string& log,
    std::size_t top_k,
    std::vector<RuntimeScaleContractObservation>* observations) {
  if (observations == nullptr) {
    return false;
  }
  observations->assign(top_k, RuntimeScaleContractObservation{});

  const std::regex up_prepack_pattern(
      R"(scale_contract_up_prepack token=([0-9]+) slot=([0-9]+) expert=(-?[0-9]+) up_input_scale=([-+0-9.eE]+) up_tensor_scale=([-+0-9.eE]+))");
  const std::regex up_packed_pattern(
      R"(scale_contract_up_packed token=([0-9]+) slot=([0-9]+) expert=(-?[0-9]+) actual_kernel_tensor_scale=([-+0-9.eE]+) raw_checkpoint_input_scale=([-+0-9.eE]+) raw_checkpoint_input_scale_match=(yes|no) normalized_checkpoint_tensor_scale_candidate=([-+0-9.eE]+) activation_global_scale_inverse=([-+0-9.eE]+) activation_global_scale_inverse_match=(yes|no) actual_fused_alpha=([-+0-9.eE]+) expected_fused_alpha=([-+0-9.eE]+) fused_alpha_match=(yes|no))");
  const std::regex down_prepack_pattern(
      R"(scale_contract_down_prepack token=([0-9]+) slot=([0-9]+) expert=(-?[0-9]+) down_input_scale=([-+0-9.eE]+) down_tensor_scale=([-+0-9.eE]+))");
  const std::regex down_packed_pattern(
      R"(scale_contract_down_packed token=([0-9]+) slot=([0-9]+) expert=(-?[0-9]+) actual_kernel_tensor_scale=([-+0-9.eE]+) raw_checkpoint_input_scale=([-+0-9.eE]+) raw_checkpoint_input_scale_match=(yes|no) normalized_checkpoint_tensor_scale_candidate=([-+0-9.eE]+) activation_global_scale_inverse=([-+0-9.eE]+) activation_global_scale_inverse_match=(yes|no) actual_fused_alpha=([-+0-9.eE]+) expected_fused_alpha=([-+0-9.eE]+) fused_alpha_match=(yes|no))");

  auto parse_slot =
      [top_k](const std::smatch& match) -> std::optional<std::size_t> {
    try {
      const std::size_t slot = static_cast<std::size_t>(std::stoull(match[2].str()));
      if (slot >= top_k) {
        return std::nullopt;
      }
      return slot;
    } catch (...) {
      return std::nullopt;
    }
  };
  auto parse_expert = [](const std::smatch& match) -> std::optional<std::int32_t> {
    try {
      return static_cast<std::int32_t>(std::stoi(match[3].str()));
    } catch (...) {
      return std::nullopt;
    }
  };
  auto parse_float_group =
      [](const std::smatch& match, std::size_t group_index) -> std::optional<float> {
    try {
      return std::stof(match[group_index].str());
    } catch (...) {
      return std::nullopt;
    }
  };

  std::istringstream input(log);
  std::string line;
  while (std::getline(input, line)) {
    std::smatch match;
    if (std::regex_search(line, match, up_prepack_pattern)) {
      const auto slot = parse_slot(match);
      const auto expert_index = parse_expert(match);
      const auto up_input_scale = parse_float_group(match, 4);
      const auto up_tensor_scale = parse_float_group(match, 5);
      if (!slot.has_value() ||
          !expert_index.has_value() ||
          !up_input_scale.has_value() ||
          !up_tensor_scale.has_value()) {
        return false;
      }
      auto& observation = (*observations)[*slot];
      observation.up_prepack_seen = true;
      observation.expert_index = *expert_index;
      observation.up_input_scale = *up_input_scale;
      observation.up_tensor_scale = *up_tensor_scale;
      continue;
    }
    if (std::regex_search(line, match, up_packed_pattern)) {
      const auto slot = parse_slot(match);
      const auto expert_index = parse_expert(match);
      const auto latent_tensor_scale = parse_float_group(match, 4);
      const auto runtime_expected_latent_tensor_scale = parse_float_group(match, 5);
      const auto runtime_latent_normalized_checkpoint_tensor_scale_candidate =
          parse_float_group(match, 7);
      const auto runtime_latent_inverse_candidate = parse_float_group(match, 8);
      const auto runtime_up_actual_fused_alpha = parse_float_group(match, 10);
      const auto runtime_up_expected_fused_alpha = parse_float_group(match, 11);
      if (!slot.has_value() ||
          !expert_index.has_value() ||
          !latent_tensor_scale.has_value() ||
          !runtime_expected_latent_tensor_scale.has_value() ||
          !runtime_latent_normalized_checkpoint_tensor_scale_candidate.has_value() ||
          !runtime_latent_inverse_candidate.has_value() ||
          !runtime_up_actual_fused_alpha.has_value() ||
          !runtime_up_expected_fused_alpha.has_value()) {
        return false;
      }
      auto& observation = (*observations)[*slot];
      observation.up_packed_seen = true;
      observation.expert_index = *expert_index;
      observation.latent_tensor_scale = *latent_tensor_scale;
      observation.runtime_expected_latent_tensor_scale =
          *runtime_expected_latent_tensor_scale;
      observation.runtime_latent_match = match[6].str() == "yes";
      observation.runtime_latent_normalized_checkpoint_tensor_scale_candidate =
          *runtime_latent_normalized_checkpoint_tensor_scale_candidate;
      observation.runtime_latent_inverse_candidate = *runtime_latent_inverse_candidate;
      observation.runtime_latent_inverse_match = match[9].str() == "yes";
      observation.runtime_up_actual_fused_alpha = *runtime_up_actual_fused_alpha;
      observation.runtime_up_expected_fused_alpha = *runtime_up_expected_fused_alpha;
      observation.runtime_up_fused_alpha_match = match[12].str() == "yes";
      continue;
    }
    if (std::regex_search(line, match, down_prepack_pattern)) {
      const auto slot = parse_slot(match);
      const auto expert_index = parse_expert(match);
      const auto down_input_scale = parse_float_group(match, 4);
      const auto down_tensor_scale = parse_float_group(match, 5);
      if (!slot.has_value() ||
          !expert_index.has_value() ||
          !down_input_scale.has_value() ||
          !down_tensor_scale.has_value()) {
        return false;
      }
      auto& observation = (*observations)[*slot];
      observation.down_prepack_seen = true;
      observation.expert_index = *expert_index;
      observation.down_input_scale = *down_input_scale;
      observation.down_tensor_scale = *down_tensor_scale;
      continue;
    }
    if (std::regex_search(line, match, down_packed_pattern)) {
      const auto slot = parse_slot(match);
      const auto expert_index = parse_expert(match);
      const auto down_act_tensor_scale = parse_float_group(match, 4);
      const auto runtime_expected_down_act_tensor_scale = parse_float_group(match, 5);
      const auto runtime_down_normalized_checkpoint_tensor_scale_candidate =
          parse_float_group(match, 7);
      const auto runtime_down_inverse_candidate = parse_float_group(match, 8);
      const auto runtime_down_actual_fused_alpha = parse_float_group(match, 10);
      const auto runtime_down_expected_fused_alpha = parse_float_group(match, 11);
      if (!slot.has_value() ||
          !expert_index.has_value() ||
          !down_act_tensor_scale.has_value() ||
          !runtime_expected_down_act_tensor_scale.has_value() ||
          !runtime_down_normalized_checkpoint_tensor_scale_candidate.has_value() ||
          !runtime_down_inverse_candidate.has_value() ||
          !runtime_down_actual_fused_alpha.has_value() ||
          !runtime_down_expected_fused_alpha.has_value()) {
        return false;
      }
      auto& observation = (*observations)[*slot];
      observation.down_packed_seen = true;
      observation.expert_index = *expert_index;
      observation.down_act_tensor_scale = *down_act_tensor_scale;
      observation.runtime_expected_down_act_tensor_scale =
          *runtime_expected_down_act_tensor_scale;
      observation.runtime_down_match = match[6].str() == "yes";
      observation.runtime_down_normalized_checkpoint_tensor_scale_candidate =
          *runtime_down_normalized_checkpoint_tensor_scale_candidate;
      observation.runtime_down_inverse_candidate = *runtime_down_inverse_candidate;
      observation.runtime_down_inverse_match = match[9].str() == "yes";
      observation.runtime_down_actual_fused_alpha = *runtime_down_actual_fused_alpha;
      observation.runtime_down_expected_fused_alpha = *runtime_down_expected_fused_alpha;
      observation.runtime_down_fused_alpha_match = match[12].str() == "yes";
      continue;
    }
  }

  return true;
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

std::optional<std::vector<float>> read_descriptor_fp32(const KernelTensorDescriptor& descriptor) {
  if (descriptor.packed_data == nullptr || descriptor.storage_dtype != "fp32") {
    return std::nullopt;
  }
  std::size_t count = 1;
  for (const std::size_t dim : descriptor.logical_shape) {
    count *= dim;
  }
  if (count == 0 || descriptor.packed_nbytes != count * sizeof(float)) {
    return std::nullopt;
  }
  std::vector<float> values(count, 0.0f);
  std::memcpy(values.data(), descriptor.packed_data, descriptor.packed_nbytes);
  return values;
}

std::optional<std::vector<float>> read_gemm_descriptor_fp32(const GemmDescriptor& descriptor) {
  if (descriptor.packed_data == nullptr) {
    return std::nullopt;
  }
  const std::size_t count = descriptor.output_rows * descriptor.input_cols;
  if (count == 0) {
    return std::nullopt;
  }
  std::vector<float> values(count, 0.0f);
  if (descriptor.storage_dtype == "fp32") {
    if (descriptor.packed_nbytes != count * sizeof(float)) {
      return std::nullopt;
    }
    std::memcpy(values.data(), descriptor.packed_data, descriptor.packed_nbytes);
    return values;
  }
  if (descriptor.storage_dtype == "bf16") {
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

std::optional<RequestExecutionConfig> make_expert_request_config(
    const FixtureMetadata& metadata,
    std::size_t max_tokens) {
  if (metadata.hidden_size == 0 ||
      max_tokens == 0 ||
      metadata.top_k == 0 ||
      metadata.routed_expert_intermediate_size == 0 ||
      metadata.moe_latent_size == 0 ||
      metadata.shared_expert_intermediate_size == 0 ||
      metadata.n_routed_experts == 0) {
    return std::nullopt;
  }
  RequestExecutionConfig config;
  config.hidden_size = metadata.hidden_size;
  config.max_tokens = max_tokens;
  config.scratch_tokens = max_tokens;
  config.expert_selection_capacity = max_tokens * metadata.top_k;
  config.expert_intermediate_scratch_numel =
      metadata.top_k * metadata.routed_expert_intermediate_size;
  config.expert_aux_scratch_numel =
      (4 * metadata.hidden_size) +
      (3 * metadata.moe_latent_size) +
      metadata.shared_expert_intermediate_size +
      metadata.n_routed_experts;
  return config;
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

struct DiffSummary {
  float max_abs_diff = std::numeric_limits<float>::infinity();
  std::size_t index = 0;
  float actual = 0.0f;
  float expected = 0.0f;
};

DiffSummary summarize_diff(
    const std::vector<float>& actual,
    const std::vector<float>& expected) {
  DiffSummary summary;
  if (actual.size() != expected.size()) {
    return summary;
  }
  summary.max_abs_diff = 0.0f;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const float diff = std::fabs(actual[i] - expected[i]);
    if (diff > summary.max_abs_diff) {
      summary.max_abs_diff = diff;
      summary.index = i;
      summary.actual = actual[i];
      summary.expected = expected[i];
    }
  }
  return summary;
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

void print_diff_summary(const std::string& comparison, float diff, float tolerance) {
  std::cerr << "comparison: " << comparison << "\n";
  std::cerr << "max_abs_diff: " << diff << "\n";
  std::cerr << "within_tolerance: " << (diff <= tolerance ? "yes" : "no") << "\n";
}

void print_diff_summary(
    const std::string& comparison,
    const DiffSummary& summary,
    std::size_t hidden_size,
    float tolerance) {
  std::cerr << "comparison: " << comparison << "\n";
  std::cerr << "max_abs_diff: " << summary.max_abs_diff << "\n";
  std::cerr << "within_tolerance: "
            << (summary.max_abs_diff <= tolerance ? "yes" : "no") << "\n";
  std::cerr << "first_bad_index: " << summary.index << "\n";
  std::cerr << "first_bad_row: "
            << (hidden_size == 0 ? 0 : summary.index / hidden_size) << "\n";
  std::cerr << "first_bad_col: "
            << (hidden_size == 0 ? 0 : summary.index % hidden_size) << "\n";
  std::cerr << "first_bad_actual: " << summary.actual << "\n";
  std::cerr << "first_bad_expected: " << summary.expected << "\n";
}

bool run_expert_layer3_fastpath_microharness() {
  constexpr float kTolerance = 1.0e-3f;
  if (::setenv("NEMOTRON_FORWARD_DEBUG", "1", 1) != 0) {
    return expect(false, "NEMOTRON_FORWARD_DEBUG should be set");
  }

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

  const char* input_hidden_override_env =
      std::getenv("NEMOTRON_EXPERT_LAYER_INPUT_HIDDEN_OVERRIDE");
  const std::filesystem::path input_hidden_path =
      (input_hidden_override_env != nullptr &&
       std::string(input_hidden_override_env).size() != 0)
          ? std::filesystem::path(input_hidden_override_env)
          : (fixture_root / "input_hidden_fp32.bin");
  const std::vector<float> input_hidden = read_float_file(input_hidden_path);
  std::cerr << "input_hidden_path: " << input_hidden_path << "\n";
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
  const std::vector<float> expected_fc1_latent_output =
      read_float_file(fixture_root / "expected_fc1_latent_output_fp32.bin");
  const std::vector<float> expected_routed_latent_output =
      read_float_file(fixture_root / "expected_routed_latent_output_fp32.bin");
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
          is_supported_nvfp4_activation_packing_mode(metadata->routed_nvfp4_activation_packing_mode),
          "routed NVFP4 activation packing mode should be supported") ||
      !expect(
          routed_contract_is_effective || routed_contract_is_raw,
          "routed NVFP4 tensor scale contract should be supported") ||
      !expect(
          expected_selected_expert_indices.size() == metadata->input_rows * metadata->top_k,
          "selected expert count should match top-k") ||
      !expect(
          expected_fc1_latent_output.size() == metadata->input_rows * metadata->moe_latent_size,
          "expected fc1 latent output size should match metadata") ||
      !expect(
          expected_routed_latent_output.size() ==
              metadata->input_rows * metadata->moe_latent_size,
          "expected routed latent output size should match metadata") ||
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
    if (!expect(
            is_supported_nvfp4_activation_packing_mode(
                metadata->shared_down_nvfp4_activation_packing_mode),
            "shared down NVFP4 activation packing mode should be supported") ||
        !expect(
            is_supported_nvfp4_tensor_scale_contract(
                metadata->shared_down_nvfp4_weight_tensor_scale_contract),
            "shared down NVFP4 tensor scale contract should be supported")) {
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
  const auto find_routed_expert_fixture =
      [&](std::int32_t expert_index) -> const RoutedExpertFixture* {
    const auto it = std::find_if(
        routed_experts.begin(),
        routed_experts.end(),
        [expert_index](const RoutedExpertFixture& expert) {
          return expert.expert_index == expert_index;
        });
    return it == routed_experts.end() ? nullptr : &(*it);
  };

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
  bool batch_run_ok = false;
  {
    std::ostringstream suppressed_debug_log;
    ScopedStreamRedirect redirect(std::cerr, suppressed_debug_log.rdbuf());
    batch_run_ok =
        slice->Run(*cublas, &heuristic_cache, *batch_input, batch_output.get(), nullptr);
  }
  if (!expect(
          batch_run_ok,
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
            "row input hidden should upload")) {
      return false;
    }
    bool sequential_run_ok = false;
    {
      std::ostringstream suppressed_debug_log;
      ScopedStreamRedirect redirect(std::cerr, suppressed_debug_log.rdbuf());
      sequential_run_ok =
          slice->Run(*cublas, &heuristic_cache, *row_input, row_output.get(), nullptr);
    }
    if (!expect(
            sequential_run_ok,
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

  const auto batch_request_config =
      make_expert_request_config(*metadata, metadata->input_rows);
  if (!expect(batch_request_config.has_value(), "batch request config should build")) {
    return false;
  }
  auto batch_request_context = RequestExecutionContext::Create(*batch_request_config);
  if (!expect(
          batch_request_context != nullptr && batch_request_context->valid(),
          "batch request context should allocate")) {
    return false;
  }
  auto batch_request_output =
      DeviceTensorFp32::Create({metadata->input_rows, metadata->hidden_size});
  if (!expect(
          batch_request_output != nullptr && batch_request_output->valid(),
          "batch request output tensor should allocate")) {
    return false;
  }
  bool batch_request_run_ok = false;
  {
    std::ostringstream suppressed_debug_log;
    ScopedStreamRedirect redirect(std::cerr, suppressed_debug_log.rdbuf());
    batch_request_run_ok = slice->RunWithRequestContext(
        *cublas,
        &heuristic_cache,
        *batch_request_context,
        *batch_input,
        batch_request_output.get(),
        nullptr);
  }
  if (!expect(
          batch_request_run_ok,
          "batched request-context fastpath expert layer run should succeed")) {
    return false;
  }
  std::vector<float> batch_request_output_host(batch_output->numel(), 0.0f);
  if (!expect(
          batch_request_output->CopyToHost(
              batch_request_output_host.data(),
              batch_request_output_host.size()),
          "batched request-context output should download")) {
    return false;
  }

  const auto row_request_config = make_expert_request_config(*metadata, 1);
  if (!expect(row_request_config.has_value(), "row request config should build")) {
    return false;
  }
  auto row_request_context = RequestExecutionContext::Create(*row_request_config);
  if (!expect(
          row_request_context != nullptr && row_request_context->valid(),
          "row request context should allocate")) {
    return false;
  }

  std::copy_n(input_hidden.data(), metadata->hidden_size, row_input_host.data());
  bool request_run_ok = false;
  std::ostringstream request_debug_capture;
  std::string request_debug_log;
  if (!expect(
          row_input->CopyFromHost(row_input_host.data(), row_input_host.size()),
          "row0 request-context input hidden should upload")) {
    return false;
  }
  {
    ScopedStreamRedirect redirect(std::cerr, request_debug_capture.rdbuf());
    request_run_ok = slice->RunWithRequestContext(
        *cublas,
        &heuristic_cache,
        *row_request_context,
        *row_input,
        row_output.get(),
        nullptr);
  }
  request_debug_log = request_debug_capture.str();
  std::cerr << request_debug_log;
  if (!expect(
          request_run_ok,
          "row0 request-context fastpath expert layer run should succeed")) {
    return false;
  }

  std::vector<float> row0_request_output(metadata->hidden_size, 0.0f);
  if (!expect(
          row_output->CopyToHost(row0_request_output.data(), row0_request_output.size()),
          "row0 request-context output should download")) {
    return false;
  }

  const std::size_t grouped_up_count =
      metadata->top_k * metadata->routed_expert_intermediate_size;
  if (!expect(
          row_request_context->expert_intermediate_scratch() != nullptr &&
              row_request_context->expert_intermediate_scratch()->valid() &&
              row_request_context->expert_intermediate_scratch()->numel() >= grouped_up_count,
          "row0 grouped-up scratch should be available")) {
    return false;
  }
  std::vector<float> grouped_up_host(grouped_up_count, 0.0f);
  if (!expect(
          row_request_context->expert_intermediate_scratch()->CopyToHost(
              grouped_up_host.data(),
              grouped_up_host.size()),
          "row0 grouped-up scratch should download")) {
    return false;
  }

  if (!expect(
          row_request_context->expert_aux_scratch() != nullptr &&
              row_request_context->expert_aux_scratch()->valid(),
          "row0 expert aux scratch should be available")) {
    return false;
  }
  std::vector<float> expert_aux_host(
      row_request_context->expert_aux_scratch()->numel(),
      0.0f);
  if (!expect(
          row_request_context->expert_aux_scratch()->CopyToHost(
              expert_aux_host.data(),
              expert_aux_host.size()),
          "row0 expert aux scratch should download")) {
    return false;
  }

  const auto* row0_selected_indices_host = row_request_context->expert_selection_indices_host();
  if (!expect(
          row0_selected_indices_host != nullptr &&
              row0_selected_indices_host->size() >= metadata->top_k,
          "row0 selected expert host buffer should be populated")) {
    return false;
  }

  const std::size_t aux_normalized_offset = metadata->n_routed_experts;
  const std::size_t aux_latent_offset = aux_normalized_offset + metadata->hidden_size;
  const std::size_t aux_routed_latent_offset =
      aux_latent_offset + metadata->moe_latent_size;
  const std::vector<float> row0_latent_host =
      slice_vector(expert_aux_host, aux_latent_offset, metadata->moe_latent_size);
  const std::vector<float> row0_routed_latent_host =
      slice_vector(expert_aux_host, aux_routed_latent_offset, metadata->moe_latent_size);
  const std::vector<float> expected_row0_fc1_latent =
      slice_vector(expected_fc1_latent_output, 0, metadata->moe_latent_size);
  const std::vector<float> expected_row0_routed_latent =
      slice_vector(expected_routed_latent_output, 0, metadata->moe_latent_size);
  const std::vector<float> expected_row0_final_output =
      slice_vector(expected_final_output, 0, metadata->hidden_size);
  if (!expect(
          row0_latent_host.size() == metadata->moe_latent_size,
          "row0 latent scratch view should match metadata") ||
      !expect(
          row0_routed_latent_host.size() == metadata->moe_latent_size,
          "row0 routed latent scratch view should match metadata") ||
      !expect(
          expected_row0_fc1_latent.size() == metadata->moe_latent_size,
          "row0 expected fc1 latent slice should match metadata") ||
      !expect(
          expected_row0_routed_latent.size() == metadata->moe_latent_size,
          "row0 expected routed latent slice should match metadata") ||
      !expect(
          expected_row0_final_output.size() == metadata->hidden_size,
          "row0 expected final output slice should match metadata")) {
    return false;
  }

  std::cerr << std::fixed << std::setprecision(9);
  std::cerr << "row00_request_selected_experts=[";
  for (std::size_t slot = 0; slot < metadata->top_k; ++slot) {
    if (slot > 0) {
      std::cerr << ",";
    }
    std::cerr << (*row0_selected_indices_host)[slot];
  }
  std::cerr << "]\n";

  std::vector<RuntimeScaleContractObservation> runtime_scale_observations;
  if (!expect(
          parse_runtime_scale_contract_log(
              request_debug_log,
              metadata->top_k,
              &runtime_scale_observations),
          "request-context scale contract debug log should parse")) {
    return false;
  }
  std::cerr << std::setprecision(std::numeric_limits<float>::max_digits10);
  bool scale_contract_ok = true;
  const auto record_scale_expect =
      [&](bool condition, const std::string& message) {
        if (!expect(condition, message)) {
          scale_contract_ok = false;
        }
      };
  for (std::size_t slot = 0; slot < metadata->top_k; ++slot) {
    const std::int32_t expert_index = (*row0_selected_indices_host)[slot];
    const auto expected_up_input_scale =
        parse_routed_expert_input_scale(
            metadata_json,
            static_cast<std::size_t>(expert_index),
            "up_proj");
    const auto expected_down_input_scale =
        parse_routed_expert_input_scale(
            metadata_json,
            static_cast<std::size_t>(expert_index),
            "down_proj");
    const RoutedExpertFixture* expert_fixture = find_routed_expert_fixture(expert_index);
    if (!expect(
            expected_up_input_scale.has_value() &&
                expected_down_input_scale.has_value(),
            "selected expert should expose fixture input scales") ||
        !expect(
            expert_fixture != nullptr,
            "selected expert fixture should be available")) {
      return false;
    }
    const float expected_latent_tensor_scale = *expected_up_input_scale;
    const float expected_down_act_tensor_scale = *expected_down_input_scale;
    const float expected_up_fused_alpha =
        expected_latent_tensor_scale * expert_fixture->up_weight_tensor_scale.front();
    const float expected_down_fused_alpha =
        expected_down_act_tensor_scale * expert_fixture->down_weight_tensor_scale.front();
    const RuntimeScaleContractObservation& runtime_observation =
        runtime_scale_observations[slot];
    const bool up_input_scale_match =
        runtime_observation.up_prepack_seen &&
        scale_contract_matches(
            runtime_observation.up_input_scale,
            *expected_up_input_scale);
    const bool latent_tensor_scale_match =
        runtime_observation.up_packed_seen &&
        scale_contract_matches(
            runtime_observation.latent_tensor_scale,
            expected_latent_tensor_scale);
    const bool down_input_scale_match =
        runtime_observation.down_prepack_seen &&
        scale_contract_matches(
            runtime_observation.down_input_scale,
            *expected_down_input_scale);
    const bool down_act_tensor_scale_match =
        runtime_observation.down_packed_seen &&
        scale_contract_matches(
            runtime_observation.down_act_tensor_scale,
            expected_down_act_tensor_scale);
    std::cerr << "scale_contract_expected: row=00 slot=" << two_digit_stem(slot)
              << " expert=" << expert_index
              << " expected_up_input_scale=" << *expected_up_input_scale
              << " expected_latent_tensor_scale=" << expected_latent_tensor_scale
              << " expected_up_fused_alpha=" << expected_up_fused_alpha
              << " expected_up_tensor_scale=" << expert_fixture->up_weight_tensor_scale.front()
              << " expected_down_input_scale=" << *expected_down_input_scale
              << " expected_down_act_tensor_scale="
              << expected_down_act_tensor_scale
              << " expected_down_fused_alpha=" << expected_down_fused_alpha
              << " expected_down_tensor_scale="
              << expert_fixture->down_weight_tensor_scale.front()
              << "\n";
    std::cerr << "scale_contract_check: row=00 slot=" << two_digit_stem(slot)
              << " expert=" << expert_index
              << " up_input_scale_match=" << (up_input_scale_match ? "yes" : "no")
              << " latent_tensor_scale_match="
              << (latent_tensor_scale_match ? "yes" : "no")
              << " down_input_scale_match=" << (down_input_scale_match ? "yes" : "no")
              << " down_act_tensor_scale_match="
              << (down_act_tensor_scale_match ? "yes" : "no")
              << "\n";
    record_scale_expect(
        runtime_observation.up_prepack_seen,
        "runtime should emit routed up prepack scale dump for each slot");
    record_scale_expect(
        runtime_observation.up_packed_seen,
        "runtime should emit routed up packed scale dump for each slot");
    record_scale_expect(
        runtime_observation.down_prepack_seen,
        "runtime should emit routed down prepack scale dump for each slot");
    record_scale_expect(
        runtime_observation.down_packed_seen,
        "runtime should emit routed down packed scale dump for each slot");
    record_scale_expect(
        runtime_observation.expert_index == expert_index,
        "runtime scale dump expert index should match selected expert");
    record_scale_expect(
        runtime_observation.runtime_latent_match,
        "runtime routed up raw checkpoint tensor-scale match flag should be yes");
    record_scale_expect(
        runtime_observation.runtime_down_match,
        "runtime routed down raw checkpoint tensor-scale match flag should be yes");
    record_scale_expect(
        scale_contract_matches(
            runtime_observation.runtime_expected_latent_tensor_scale,
            expected_latent_tensor_scale),
        "runtime expected routed up tensor scale should equal fixture checkpoint input scale");
    record_scale_expect(
        scale_contract_matches(
            runtime_observation.runtime_expected_down_act_tensor_scale,
            expected_down_act_tensor_scale),
        "runtime expected routed down tensor scale should equal fixture checkpoint input scale");
    record_scale_expect(
        up_input_scale_match,
        "gathered routed up input scale should match fixture checkpoint value");
    record_scale_expect(
        latent_tensor_scale_match,
        "latent tensor scale should equal routed up checkpoint input scale");
    record_scale_expect(
        down_input_scale_match,
        "gathered routed down input scale should match fixture checkpoint value");
    record_scale_expect(
        down_act_tensor_scale_match,
        "down activation tensor scale should equal routed down checkpoint input scale");
    record_scale_expect(
        runtime_observation.runtime_up_fused_alpha_match,
        "runtime routed up fused alpha should match checkpoint input_scale * weight_scale_2");
    record_scale_expect(
        runtime_observation.runtime_down_fused_alpha_match,
        "runtime routed down fused alpha should match checkpoint input_scale * weight_scale_2");
    record_scale_expect(
        scale_contract_matches(
            runtime_observation.runtime_up_expected_fused_alpha,
            expected_up_fused_alpha),
        "runtime expected routed up fused alpha should equal fixture checkpoint input_scale * weight_scale_2");
    record_scale_expect(
        scale_contract_matches(
            runtime_observation.runtime_down_expected_fused_alpha,
            expected_down_fused_alpha),
        "runtime expected routed down fused alpha should equal fixture checkpoint input_scale * weight_scale_2");
  }
  if (!scale_contract_ok) {
    return false;
  }
  std::cerr << std::fixed << std::setprecision(9);

  print_diff_report(
      "row00_fc1_latent_vs_oracle_expected",
      row0_latent_host,
      expected_row0_fc1_latent,
      kTolerance);

  float row0_grouped_up_diff = 0.0f;
  float row0_activated_hidden_diff = 0.0f;
  for (std::size_t oracle_slot = 0; oracle_slot < metadata->top_k; ++oracle_slot) {
    const std::int32_t expected_expert_index =
        expected_selected_expert_indices[oracle_slot];
    const auto actual_it = std::find(
        row0_selected_indices_host->begin(),
        row0_selected_indices_host->begin() + metadata->top_k,
        expected_expert_index);
    if (!expect(
            actual_it != row0_selected_indices_host->begin() + metadata->top_k,
            "row0 expected expert should be present in actual selected experts")) {
      return false;
    }
    const std::size_t actual_slot =
        static_cast<std::size_t>(actual_it - row0_selected_indices_host->begin());
    const std::string slot_stem = two_digit_stem(oracle_slot);
    const std::vector<float> actual_grouped_up = slice_vector(
        grouped_up_host,
        actual_slot * metadata->routed_expert_intermediate_size,
        metadata->routed_expert_intermediate_size);
    const std::vector<float> expected_grouped_up =
        read_float_file(
            fixture_root /
            ("expected_routed_up_proj_output_row00_slot" + slot_stem + "_fp32.bin"));
    const std::vector<float> expected_activated_hidden =
        read_float_file(
            fixture_root /
            ("expected_routed_activated_hidden_row00_slot" + slot_stem + "_fp32.bin"));
    if (!expect(
            actual_grouped_up.size() == metadata->routed_expert_intermediate_size,
            "row0 grouped-up slot shape should match metadata") ||
        !expect(
            expected_grouped_up.size() == metadata->routed_expert_intermediate_size,
            "row0 expected grouped-up slot shape should match metadata") ||
        !expect(
            expected_activated_hidden.size() ==
                metadata->routed_expert_intermediate_size,
            "row0 expected activated-hidden slot shape should match metadata")) {
      return false;
    }
    std::vector<float> actual_activated_hidden = actual_grouped_up;
    for (float& value : actual_activated_hidden) {
      value = value > 0.0f ? (value * value) : 0.0f;
    }
    const float grouped_up_diff =
        max_abs_diff(actual_grouped_up, expected_grouped_up);
    const float activated_hidden_diff =
        max_abs_diff(actual_activated_hidden, expected_activated_hidden);
    row0_grouped_up_diff = std::max(row0_grouped_up_diff, grouped_up_diff);
    row0_activated_hidden_diff =
        std::max(row0_activated_hidden_diff, activated_hidden_diff);
    std::cerr << "boundary_check: row=00 oracle_slot=" << slot_stem
              << " actual_slot=" << two_digit_stem(actual_slot)
              << " expert=" << expected_expert_index
              << " grouped_up_max_abs_diff=" << grouped_up_diff
              << " activated_hidden_max_abs_diff=" << activated_hidden_diff
              << "\n";
  }
  print_diff_summary(
      "row00_grouped_up_vs_oracle_expected",
      row0_grouped_up_diff,
      kTolerance);
  print_diff_summary(
      "row00_activated_hidden_vs_oracle_expected",
      row0_activated_hidden_diff,
      kTolerance);
  print_diff_report(
      "row00_routed_latent_vs_oracle_expected",
      row0_routed_latent_host,
      expected_row0_routed_latent,
      kTolerance);
  print_diff_report(
      "row00_request_context_final_vs_oracle_expected",
      row0_request_output,
      expected_row0_final_output,
      kTolerance);

  const bool compare_manifest_slice =
      std::getenv("NEMOTRON_COMPARE_MANIFEST_SLICE") != nullptr;
  if (compare_manifest_slice) {
    const char* manifest_env = std::getenv("NEMOTRON_FORWARD_MANIFEST");
    if (!expect(
            manifest_env != nullptr && std::string(manifest_env).size() != 0,
            "manifest slice compare requires NEMOTRON_FORWARD_MANIFEST")) {
      return false;
    }
    const std::filesystem::path manifest_path(manifest_env);
    if (!expect(
            std::filesystem::exists(manifest_path),
            "manifest slice compare manifest path should exist")) {
      return false;
    }

    const auto environment = RuntimeEnvironment::BuildFromManifestFile(
        manifest_path,
        make_runtime_bootstrap_options());
    if (!expect(
            static_cast<bool>(environment),
            "manifest-backed runtime environment should build") ||
        !expect(
            environment->has_model_schedule(),
            "manifest-backed runtime environment should expose model schedule") ||
        !expect(
            environment->has_kernel_catalog(),
            "manifest-backed runtime environment should expose kernel catalog") ||
        !expect(
            environment->has_gemm_catalog(),
            "manifest-backed runtime environment should expose gemm catalog")) {
      return false;
    }
    const auto* manifest_layer =
        environment->model_schedule()->FindLayer(metadata->layer_index);
    if (!expect(
            manifest_layer != nullptr,
            "manifest-backed model schedule should expose the target layer")) {
      return false;
    }
    const auto manifest_bindings = nemotron::BuildExpertLayerBindings(
        *manifest_layer,
        *environment->kernel_catalog(),
        *environment->gemm_catalog(),
        metadata->n_routed_experts);
    if (!expect(
            manifest_bindings.has_value(),
            "manifest-backed layer bindings should build")) {
      return false;
    }
    const auto manifest_gate_score_correction_bias = read_descriptor_fp32(
        *manifest_bindings->gate_score_correction_bias);
    const auto manifest_gate_weight =
        read_gemm_descriptor_fp32(*manifest_bindings->gate_weight);
    if (!expect(
            manifest_gate_score_correction_bias.has_value() &&
                manifest_gate_score_correction_bias->size() == gate_score_correction_bias.size(),
            "manifest-backed gate score correction bias should load")) {
      return false;
    }
    if (!expect(
            manifest_gate_weight.has_value() &&
                manifest_gate_weight->size() == gate_weight.size(),
            "manifest-backed gate weight should load")) {
      return false;
    }

    auto manifest_slice = ExpertLayerSlice::Create(layer_config, *manifest_bindings);
    if (!expect(
            manifest_slice != nullptr && manifest_slice->valid(),
            "manifest-backed expert layer slice should create")) {
      return false;
    }

    auto manifest_batch_output =
        DeviceTensorFp32::Create({metadata->input_rows, metadata->hidden_size});
    if (!expect(
            manifest_batch_output != nullptr && manifest_batch_output->valid(),
            "manifest-backed batched output tensor should allocate")) {
      return false;
    }
    bool manifest_batch_run_ok = false;
    {
      std::ostringstream suppressed_debug_log;
      ScopedStreamRedirect redirect(std::cerr, suppressed_debug_log.rdbuf());
      manifest_batch_run_ok = manifest_slice->Run(
          *cublas,
          &heuristic_cache,
          *batch_input,
          manifest_batch_output.get(),
          nullptr);
    }
    if (!expect(
            manifest_batch_run_ok,
            "manifest-backed batched expert layer run should succeed")) {
      return false;
    }
    std::vector<float> manifest_batch_output_host(batch_output_host.size(), 0.0f);
    if (!expect(
            manifest_batch_output->CopyToHost(
                manifest_batch_output_host.data(),
                manifest_batch_output_host.size()),
            "manifest-backed batched output should download")) {
      return false;
    }

    const auto manifest_request_config = make_expert_request_config(*metadata, 1);
    if (!expect(
            manifest_request_config.has_value(),
            "manifest-backed row request config should build")) {
      return false;
    }
    auto manifest_request_context =
        RequestExecutionContext::Create(*manifest_request_config);
    if (!expect(
            manifest_request_context != nullptr &&
                manifest_request_context->valid(),
            "manifest-backed row request context should allocate")) {
      return false;
    }
    if (!expect(
            row_input->CopyFromHost(row_input_host.data(), row_input_host.size()),
            "manifest-backed row input hidden should upload")) {
      return false;
    }
    bool manifest_request_run_ok = false;
    std::ostringstream manifest_request_debug_capture;
    {
      ScopedStreamRedirect redirect(std::cerr, manifest_request_debug_capture.rdbuf());
      manifest_request_run_ok = manifest_slice->RunWithRequestContext(
          *cublas,
          &heuristic_cache,
          *manifest_request_context,
          *row_input,
          row_output.get(),
          nullptr);
    }
    std::cerr << manifest_request_debug_capture.str();
    if (!expect(
            manifest_request_run_ok,
            "manifest-backed row request-context run should succeed") ||
        !expect(
            row_output->CopyToHost(
                row_output_host.data(),
                row_output_host.size()),
            "manifest-backed row request output should download")) {
      return false;
    }
    const std::vector<float> manifest_row0_request_output = row_output_host;

    struct RowRequestSnapshot {
      std::vector<std::int32_t> selected_indices;
      std::vector<float> selected_weights;
      std::vector<float> output;
      std::vector<float> grouped_up;
      std::vector<float> router_logits;
      std::vector<float> normalized;
    };

    const auto capture_row_request =
        [&](ExpertLayerSlice& target_slice,
            std::size_t row_index,
            const char* label,
            bool emit_debug_log,
            RowRequestSnapshot* snapshot) -> bool {
      if (snapshot == nullptr) {
        return expect(false, "row request snapshot output should be provided");
      }
      auto request_context = RequestExecutionContext::Create(*row_request_config);
      if (!expect(
              request_context != nullptr && request_context->valid(),
              std::string(label) + " row request context should allocate")) {
        return false;
      }
      std::copy_n(
          input_hidden.data() + (row_index * metadata->hidden_size),
          metadata->hidden_size,
          row_input_host.data());
      if (!expect(
              row_input->CopyFromHost(row_input_host.data(), row_input_host.size()),
              std::string(label) + " row input should upload")) {
        return false;
      }

      bool run_ok = false;
      std::ostringstream debug_capture;
      {
        ScopedStreamRedirect redirect(std::cerr, debug_capture.rdbuf());
        run_ok = target_slice.RunWithRequestContext(
            *cublas,
            &heuristic_cache,
            *request_context,
            *row_input,
            row_output.get(),
            nullptr);
      }
      if (emit_debug_log) {
        std::cerr << debug_capture.str();
      }
      if (!expect(run_ok, std::string(label) + " row request run should succeed") ||
          !expect(
              row_output->CopyToHost(row_output_host.data(), row_output_host.size()),
              std::string(label) + " row output should download")) {
        return false;
      }

      const auto* selected_indices_host = request_context->expert_selection_indices_host();
      const auto* selected_weights_host = request_context->expert_selection_weights_host();
      if (!expect(
              selected_indices_host != nullptr &&
                  selected_indices_host->size() >= metadata->top_k,
              std::string(label) + " selected indices should be populated") ||
          !expect(
              selected_weights_host != nullptr &&
                  selected_weights_host->size() >= metadata->top_k,
              std::string(label) + " selected weights should be populated") ||
          !expect(
              request_context->expert_intermediate_scratch() != nullptr &&
                  request_context->expert_intermediate_scratch()->valid() &&
                  request_context->expert_intermediate_scratch()->numel() >=
                      grouped_up_count,
              std::string(label) + " grouped-up scratch should be available")) {
        return false;
      }

      snapshot->selected_indices.assign(
          selected_indices_host->begin(),
          selected_indices_host->begin() + metadata->top_k);
      snapshot->selected_weights.assign(
          selected_weights_host->begin(),
          selected_weights_host->begin() + metadata->top_k);
      snapshot->output = row_output_host;
      snapshot->grouped_up.assign(grouped_up_count, 0.0f);
      if (!expect(
              request_context->expert_intermediate_scratch()->CopyToHost(
                  snapshot->grouped_up.data(),
                  snapshot->grouped_up.size()),
              std::string(label) + " grouped-up scratch should download")) {
        return false;
      }
      if (!expect(
              request_context->expert_aux_scratch() != nullptr &&
                  request_context->expert_aux_scratch()->valid() &&
                  request_context->expert_aux_scratch()->numel() >=
                      metadata->n_routed_experts,
              std::string(label) + " expert aux scratch should be available")) {
        return false;
      }
      std::vector<float> expert_aux_host(
          request_context->expert_aux_scratch()->numel(),
          0.0f);
      if (!expect(
              request_context->expert_aux_scratch()->CopyToHost(
                  expert_aux_host.data(),
                  expert_aux_host.size()),
              std::string(label) + " expert aux scratch should download")) {
        return false;
      }
      snapshot->router_logits = slice_vector(
          expert_aux_host,
          0,
          metadata->n_routed_experts);
      snapshot->normalized = slice_vector(
          expert_aux_host,
          metadata->n_routed_experts,
          metadata->hidden_size);
      return true;
    };

    const auto* manifest_selected_indices_host_ptr =
        manifest_request_context->expert_selection_indices_host();
    const auto* manifest_selected_weights_host_ptr =
        manifest_request_context->expert_selection_weights_host();
    const auto* fixture_selected_weights_host_ptr =
        row_request_context->expert_selection_weights_host();
    if (!expect(
            manifest_selected_indices_host_ptr != nullptr &&
                manifest_selected_indices_host_ptr->size() >= metadata->top_k,
            "manifest-backed selected expert indices should be populated") ||
        !expect(
            manifest_selected_weights_host_ptr != nullptr &&
                manifest_selected_weights_host_ptr->size() >= metadata->top_k,
            "manifest-backed selected expert weights should be populated") ||
        !expect(
            fixture_selected_weights_host_ptr != nullptr &&
                fixture_selected_weights_host_ptr->size() >= metadata->top_k,
            "fixture-backed selected expert weights should be populated")) {
      return false;
    }

    const std::vector<std::int32_t> fixture_selected_indices(
        row0_selected_indices_host->begin(),
        row0_selected_indices_host->begin() + metadata->top_k);
    const std::vector<float> fixture_selected_weights(
        fixture_selected_weights_host_ptr->begin(),
        fixture_selected_weights_host_ptr->begin() + metadata->top_k);
    const std::vector<std::int32_t> manifest_selected_indices(
        manifest_selected_indices_host_ptr->begin(),
        manifest_selected_indices_host_ptr->begin() + metadata->top_k);
    const std::vector<float> manifest_selected_weights(
        manifest_selected_weights_host_ptr->begin(),
        manifest_selected_weights_host_ptr->begin() + metadata->top_k);

    const auto print_selection_vector =
        [](const char* label, const auto& values) {
          std::cerr << label << "=[";
          for (std::size_t i = 0; i < values.size(); ++i) {
            if (i > 0) {
              std::cerr << ",";
            }
            std::cerr << values[i];
          }
          std::cerr << "]\n";
        };
    print_selection_vector("row00_fixture_selected_experts", fixture_selected_indices);
    print_selection_vector("row00_manifest_selected_experts", manifest_selected_indices);
    print_selection_vector("row00_fixture_selected_weights", fixture_selected_weights);
    print_selection_vector("row00_manifest_selected_weights", manifest_selected_weights);

    const bool selection_order_match =
        fixture_selected_indices == manifest_selected_indices;
    std::vector<std::int32_t> fixture_selected_sorted = fixture_selected_indices;
    std::vector<std::int32_t> manifest_selected_sorted = manifest_selected_indices;
    std::sort(fixture_selected_sorted.begin(), fixture_selected_sorted.end());
    std::sort(manifest_selected_sorted.begin(), manifest_selected_sorted.end());
    const bool selection_set_match =
        fixture_selected_sorted == manifest_selected_sorted;
    std::cerr << "selection_compare: order_match="
              << (selection_order_match ? "yes" : "no")
              << " set_match=" << (selection_set_match ? "yes" : "no")
              << "\n";

    const std::size_t manifest_grouped_up_count =
        metadata->top_k * metadata->routed_expert_intermediate_size;
    if (!expect(
            manifest_request_context->expert_intermediate_scratch() != nullptr &&
                manifest_request_context->expert_intermediate_scratch()->valid() &&
                manifest_request_context->expert_intermediate_scratch()->numel() >=
                    manifest_grouped_up_count,
            "manifest-backed grouped-up scratch should be available") ||
        !expect(
            manifest_request_context->expert_aux_scratch() != nullptr &&
                manifest_request_context->expert_aux_scratch()->valid(),
            "manifest-backed expert aux scratch should be available")) {
      return false;
    }
    std::vector<float> manifest_grouped_up_host(manifest_grouped_up_count, 0.0f);
    if (!expect(
            manifest_request_context->expert_intermediate_scratch()->CopyToHost(
                manifest_grouped_up_host.data(),
                manifest_grouped_up_host.size()),
            "manifest-backed grouped-up scratch should download")) {
      return false;
    }
    std::vector<float> manifest_expert_aux_host(
        manifest_request_context->expert_aux_scratch()->numel(),
        0.0f);
    if (!expect(
            manifest_request_context->expert_aux_scratch()->CopyToHost(
                manifest_expert_aux_host.data(),
                manifest_expert_aux_host.size()),
            "manifest-backed expert aux scratch should download")) {
      return false;
    }
    const std::vector<float> manifest_row0_latent_host =
        slice_vector(manifest_expert_aux_host, aux_latent_offset, metadata->moe_latent_size);
    const std::vector<float> manifest_row0_routed_latent_host = slice_vector(
        manifest_expert_aux_host,
        aux_routed_latent_offset,
        metadata->moe_latent_size);

    const DiffSummary manifest_batch_vs_fixture_summary =
        summarize_diff(manifest_batch_output_host, batch_output_host);
    std::cerr << "manifest_gate_weight_descriptor:"
              << " kernel_family=" << static_cast<int>(manifest_bindings->gate_weight->kernel_family)
              << " storage_dtype=" << manifest_bindings->gate_weight->storage_dtype
              << " compute_dtype=" << manifest_bindings->gate_weight->compute_dtype
              << " layout=" << manifest_bindings->gate_weight->layout_tag
              << "\n";
    print_diff_report(
        "manifest_gate_weight_vs_fixture",
        *manifest_gate_weight,
        gate_weight,
        kTolerance);
    print_diff_report(
        "manifest_gate_score_correction_bias_vs_fixture",
        *manifest_gate_score_correction_bias,
        gate_score_correction_bias,
        kTolerance);
    print_diff_summary(
        "manifest_batch_vs_fixture_batch",
        manifest_batch_vs_fixture_summary,
        metadata->hidden_size,
        kTolerance);
    print_diff_report(
        "manifest_row00_fc1_latent_vs_fixture",
        manifest_row0_latent_host,
        row0_latent_host,
        kTolerance);
    print_diff_report(
        "manifest_row00_routed_latent_vs_fixture",
        manifest_row0_routed_latent_host,
        row0_routed_latent_host,
        kTolerance);
    print_diff_report(
        "manifest_row00_final_vs_fixture",
        manifest_row0_request_output,
        row0_request_output,
        kTolerance);

    const float direct_grouped_up_diff =
        selection_order_match
            ? max_abs_diff(manifest_grouped_up_host, grouped_up_host)
            : std::numeric_limits<float>::infinity();
    print_diff_summary(
        "manifest_row00_grouped_up_direct_order_vs_fixture",
        direct_grouped_up_diff,
        kTolerance);

    float aligned_grouped_up_diff = 0.0f;
    float aligned_weight_diff = 0.0f;
    bool aligned_compare_possible = selection_set_match;
    for (std::size_t fixture_slot = 0; fixture_slot < metadata->top_k; ++fixture_slot) {
      const std::int32_t expert_index = fixture_selected_indices[fixture_slot];
      const auto manifest_it = std::find(
          manifest_selected_indices.begin(),
          manifest_selected_indices.end(),
          expert_index);
      if (manifest_it == manifest_selected_indices.end()) {
        aligned_compare_possible = false;
        break;
      }
      const std::size_t manifest_slot = static_cast<std::size_t>(
          manifest_it - manifest_selected_indices.begin());
      const std::vector<float> fixture_grouped_up = slice_vector(
          grouped_up_host,
          fixture_slot * metadata->routed_expert_intermediate_size,
          metadata->routed_expert_intermediate_size);
      const std::vector<float> manifest_grouped_up = slice_vector(
          manifest_grouped_up_host,
          manifest_slot * metadata->routed_expert_intermediate_size,
          metadata->routed_expert_intermediate_size);
      const float grouped_up_diff =
          max_abs_diff(manifest_grouped_up, fixture_grouped_up);
      aligned_grouped_up_diff =
          std::max(aligned_grouped_up_diff, grouped_up_diff);
      aligned_weight_diff = std::max(
          aligned_weight_diff,
          std::fabs(
              fixture_selected_weights[fixture_slot] -
              manifest_selected_weights[manifest_slot]));
      std::cerr << "manifest_fixture_boundary_check: fixture_slot="
                << two_digit_stem(fixture_slot)
                << " manifest_slot=" << two_digit_stem(manifest_slot)
                << " expert=" << expert_index
                << " grouped_up_max_abs_diff=" << grouped_up_diff
                << " weight_abs_diff="
                << std::fabs(
                       fixture_selected_weights[fixture_slot] -
                       manifest_selected_weights[manifest_slot])
                << "\n";
    }
    if (!aligned_compare_possible) {
      aligned_grouped_up_diff = std::numeric_limits<float>::infinity();
      aligned_weight_diff = std::numeric_limits<float>::infinity();
    }
    print_diff_summary(
        "manifest_row00_grouped_up_aligned_by_expert_vs_fixture",
        aligned_grouped_up_diff,
        kTolerance);
    print_diff_summary(
        "manifest_row00_selection_weight_aligned_by_expert_vs_fixture",
        aligned_weight_diff,
        kTolerance);

    if (manifest_batch_vs_fixture_summary.max_abs_diff > kTolerance) {
      const std::size_t worst_row =
          metadata->hidden_size == 0
              ? 0
              : manifest_batch_vs_fixture_summary.index / metadata->hidden_size;
      std::cerr << "manifest_fixture_worst_row=" << worst_row << "\n";

      RowRequestSnapshot fixture_worst_row;
      RowRequestSnapshot manifest_worst_row;
      if (!capture_row_request(
              *slice,
              worst_row,
              "fixture worst row",
              false,
              &fixture_worst_row) ||
          !capture_row_request(
              *manifest_slice,
              worst_row,
              "manifest worst row",
              false,
              &manifest_worst_row)) {
        return false;
      }

      print_selection_vector(
          "worst_row_fixture_selected_experts",
          fixture_worst_row.selected_indices);
      print_selection_vector(
          "worst_row_manifest_selected_experts",
          manifest_worst_row.selected_indices);
      print_selection_vector(
          "worst_row_fixture_selected_weights",
          fixture_worst_row.selected_weights);
      print_selection_vector(
          "worst_row_manifest_selected_weights",
          manifest_worst_row.selected_weights);

      const bool worst_row_order_match =
          fixture_worst_row.selected_indices == manifest_worst_row.selected_indices;
      std::vector<std::int32_t> worst_row_fixture_sorted =
          fixture_worst_row.selected_indices;
      std::vector<std::int32_t> worst_row_manifest_sorted =
          manifest_worst_row.selected_indices;
      std::sort(worst_row_fixture_sorted.begin(), worst_row_fixture_sorted.end());
      std::sort(worst_row_manifest_sorted.begin(), worst_row_manifest_sorted.end());
      const bool worst_row_set_match =
          worst_row_fixture_sorted == worst_row_manifest_sorted;
      std::cerr << "worst_row_selection_compare: order_match="
                << (worst_row_order_match ? "yes" : "no")
                << " set_match=" << (worst_row_set_match ? "yes" : "no")
                << "\n";
      print_diff_report(
          "manifest_worst_row_normalized_vs_fixture",
          manifest_worst_row.normalized,
          fixture_worst_row.normalized,
          kTolerance);
      print_diff_report(
          "manifest_worst_row_router_logits_vs_fixture",
          manifest_worst_row.router_logits,
          fixture_worst_row.router_logits,
          kTolerance);

      if (worst_row_set_match) {
        for (std::size_t slot = 0; slot < fixture_worst_row.selected_indices.size(); ++slot) {
          const std::int32_t expert_index = fixture_worst_row.selected_indices[slot];
          const auto manifest_it = std::find(
              manifest_worst_row.selected_indices.begin(),
              manifest_worst_row.selected_indices.end(),
              expert_index);
          if (manifest_it == manifest_worst_row.selected_indices.end()) {
            continue;
          }
          const std::size_t manifest_slot = static_cast<std::size_t>(
              manifest_it - manifest_worst_row.selected_indices.begin());
          std::cerr << "worst_row_router_logit_compare: expert=" << expert_index
                    << " fixture=" << fixture_worst_row.router_logits[expert_index]
                    << " manifest=" << manifest_worst_row.router_logits[expert_index]
                    << " weight_fixture=" << fixture_worst_row.selected_weights[slot]
                    << " weight_manifest="
                    << manifest_worst_row.selected_weights[manifest_slot]
                    << "\n";
        }
      } else {
        std::vector<std::int32_t> differing_experts = fixture_worst_row.selected_indices;
        differing_experts.insert(
            differing_experts.end(),
            manifest_worst_row.selected_indices.begin(),
            manifest_worst_row.selected_indices.end());
        std::sort(differing_experts.begin(), differing_experts.end());
        differing_experts.erase(
            std::unique(differing_experts.begin(), differing_experts.end()),
            differing_experts.end());
        for (const std::int32_t expert_index : differing_experts) {
          const bool in_fixture = std::find(
                                      fixture_worst_row.selected_indices.begin(),
                                      fixture_worst_row.selected_indices.end(),
                                      expert_index) !=
                                  fixture_worst_row.selected_indices.end();
          const bool in_manifest = std::find(
                                       manifest_worst_row.selected_indices.begin(),
                                       manifest_worst_row.selected_indices.end(),
                                       expert_index) !=
                                   manifest_worst_row.selected_indices.end();
          if (in_fixture == in_manifest) {
            continue;
          }
          std::cerr << "worst_row_router_logit_diff_expert: expert=" << expert_index
                    << " in_fixture=" << (in_fixture ? "yes" : "no")
                    << " in_manifest=" << (in_manifest ? "yes" : "no")
                    << " fixture_logit="
                    << fixture_worst_row.router_logits[expert_index]
                    << " manifest_logit="
                    << manifest_worst_row.router_logits[expert_index]
                    << " fixture_score="
                    << (1.0f / (1.0f + std::exp(-fixture_worst_row.router_logits[expert_index])) +
                        gate_score_correction_bias[expert_index])
                    << " manifest_score="
                    << (1.0f / (1.0f + std::exp(-manifest_worst_row.router_logits[expert_index])) +
                        (*manifest_gate_score_correction_bias)[expert_index])
                    << "\n";
        }
      }

      print_diff_report(
          "manifest_worst_row_final_vs_fixture",
          manifest_worst_row.output,
          fixture_worst_row.output,
          kTolerance);

      const float worst_row_direct_grouped_up_diff =
          worst_row_order_match
              ? max_abs_diff(
                    manifest_worst_row.grouped_up,
                    fixture_worst_row.grouped_up)
              : std::numeric_limits<float>::infinity();
      print_diff_summary(
          "manifest_worst_row_grouped_up_direct_order_vs_fixture",
          worst_row_direct_grouped_up_diff,
          kTolerance);

      float worst_row_aligned_grouped_up_diff = 0.0f;
      float worst_row_aligned_weight_diff = 0.0f;
      bool worst_row_aligned_compare_possible = worst_row_set_match;
      for (std::size_t fixture_slot = 0; fixture_slot < metadata->top_k; ++fixture_slot) {
        const std::int32_t expert_index = fixture_worst_row.selected_indices[fixture_slot];
        const auto manifest_it = std::find(
            manifest_worst_row.selected_indices.begin(),
            manifest_worst_row.selected_indices.end(),
            expert_index);
        if (manifest_it == manifest_worst_row.selected_indices.end()) {
          worst_row_aligned_compare_possible = false;
          break;
        }
        const std::size_t manifest_slot = static_cast<std::size_t>(
            manifest_it - manifest_worst_row.selected_indices.begin());
        const std::vector<float> fixture_grouped_up = slice_vector(
            fixture_worst_row.grouped_up,
            fixture_slot * metadata->routed_expert_intermediate_size,
            metadata->routed_expert_intermediate_size);
        const std::vector<float> manifest_grouped_up = slice_vector(
            manifest_worst_row.grouped_up,
            manifest_slot * metadata->routed_expert_intermediate_size,
            metadata->routed_expert_intermediate_size);
        const float grouped_up_diff =
            max_abs_diff(manifest_grouped_up, fixture_grouped_up);
        worst_row_aligned_grouped_up_diff =
            std::max(worst_row_aligned_grouped_up_diff, grouped_up_diff);
        worst_row_aligned_weight_diff = std::max(
            worst_row_aligned_weight_diff,
            std::fabs(
                fixture_worst_row.selected_weights[fixture_slot] -
                manifest_worst_row.selected_weights[manifest_slot]));
        std::cerr << "manifest_fixture_worst_row_boundary_check: fixture_slot="
                  << two_digit_stem(fixture_slot)
                  << " manifest_slot=" << two_digit_stem(manifest_slot)
                  << " expert=" << expert_index
                  << " grouped_up_max_abs_diff=" << grouped_up_diff
                  << " weight_abs_diff="
                  << std::fabs(
                         fixture_worst_row.selected_weights[fixture_slot] -
                         manifest_worst_row.selected_weights[manifest_slot])
                  << "\n";
      }
      if (!worst_row_aligned_compare_possible) {
        worst_row_aligned_grouped_up_diff = std::numeric_limits<float>::infinity();
        worst_row_aligned_weight_diff = std::numeric_limits<float>::infinity();
      }
      print_diff_summary(
          "manifest_worst_row_grouped_up_aligned_by_expert_vs_fixture",
          worst_row_aligned_grouped_up_diff,
          kTolerance);
      print_diff_summary(
          "manifest_worst_row_selection_weight_aligned_by_expert_vs_fixture",
          worst_row_aligned_weight_diff,
          kTolerance);
    }
  }

  std::cerr << "grouped_fastpath: all tokens used direct fused path\n";
  std::cerr << std::fixed << std::setprecision(9);
  print_diff_report("batched_vs_oracle_expected", batch_output_host, expected_final_output, kTolerance);
  print_diff_report(
      "batched_request_context_vs_oracle_expected",
      batch_request_output_host,
      expected_final_output,
      kTolerance);
  print_diff_report(
      "sequential_vs_oracle_expected",
      sequential_output_host,
      expected_final_output,
      kTolerance);
  std::cerr << "comparison: batched_vs_sequential\n";
  std::cerr << "max_abs_diff: " << max_abs_diff(batch_output_host, sequential_output_host) << "\n";
  std::cerr << "comparison: batched_request_context_vs_batched\n";
  std::cerr << "max_abs_diff: " << max_abs_diff(batch_request_output_host, batch_output_host) << "\n";

  const char* dump_root_env = std::getenv("NEMOTRON_EXPERT_LAYER_DUMP_ROOT");
  if (dump_root_env != nullptr && std::string(dump_root_env).size() != 0) {
    const std::filesystem::path dump_root(dump_root_env);
    const std::filesystem::path batch_dump_path =
        dump_root / "expert_layer3_fastpath_microharness_batch_output_fp32.bin";
    const std::filesystem::path batch_request_dump_path =
        dump_root / "expert_layer3_fastpath_microharness_batch_request_output_fp32.bin";
    const std::filesystem::path sequential_dump_path =
        dump_root / "expert_layer3_fastpath_microharness_sequential_output_fp32.bin";
    write_float_file(batch_dump_path, batch_output_host);
    write_float_file(batch_request_dump_path, batch_request_output_host);
    write_float_file(sequential_dump_path, sequential_output_host);
    std::cerr << "dumped_output: " << batch_dump_path << "\n";
    std::cerr << "dumped_output: " << batch_request_dump_path << "\n";
    std::cerr << "dumped_output: " << sequential_dump_path << "\n";
  }

  return true;
}

}  // namespace

int main() {
  return run_expert_layer3_fastpath_microharness() ? 0 : 1;
}
