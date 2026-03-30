#include "nemotron/cublaslt_handle.h"
#include "nemotron/device_tensor.h"
#include "nemotron/expert_layer.h"
#include "nemotron/runtime_environment.h"
#include "nemotron/single_token_forward_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <regex>
#include <string>
#include <vector>

namespace {

constexpr std::size_t GiB(std::size_t value) {
  return value * 1024ull * 1024ull * 1024ull;
}

struct FixtureMetadata {
  std::size_t hidden_size = 0;
  std::size_t runtime_token_count = 0;
};

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

std::optional<FixtureMetadata> load_metadata(const std::filesystem::path& root) {
  const std::string json = read_text_file(root / "metadata.json");
  FixtureMetadata metadata;
  const auto hidden_size = parse_json_uint_field(json, "hidden_size");
  const auto runtime_token_count = parse_json_uint_field(json, "runtime_token_count");
  if (!hidden_size || !runtime_token_count) {
    return std::nullopt;
  }
  metadata.hidden_size = *hidden_size;
  metadata.runtime_token_count = *runtime_token_count;
  return metadata;
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

bool run_expert_layer3_prefix_replay() {
  const bool debug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  const char* manifest_env = std::getenv("NEMOTRON_FORWARD_MANIFEST");
  if (manifest_env == nullptr || std::string(manifest_env).empty()) {
    std::cout << "expert_layer3_prefix_replay_test: SKIP (NEMOTRON_FORWARD_MANIFEST is unset)\n";
    return true;
  }
  if (!has_cuda_device()) {
    std::cout << "expert_layer3_prefix_replay_test: SKIP (no CUDA device available)\n";
    return true;
  }

  const std::filesystem::path fixture_root(
      NEMOTRON_PREFILL_PREFIX_ORACLE_FIXTURE_ROOT);
  const auto metadata = load_metadata(fixture_root);
  if (!expect(metadata.has_value(), "prefix replay metadata should load")) {
    return false;
  }

  const std::vector<float> layer2_input =
      read_float_file(fixture_root / "expected_layer_002_output_fp32.bin");
  const std::vector<float> layer3_expected =
      read_float_file(fixture_root / "expected_layer_003_output_fp32.bin");
  if (!expect(
          layer2_input.size() ==
              metadata->runtime_token_count * metadata->hidden_size,
          "layer2 oracle input size should match fixture metadata") ||
      !expect(
          layer3_expected.size() == layer2_input.size(),
          "layer3 oracle output size should match layer2 input")) {
    return false;
  }

  const std::filesystem::path manifest_path(manifest_env);
  if (!std::filesystem::exists(manifest_path)) {
    std::cout << "expert_layer3_prefix_replay_test: SKIP (manifest path does not exist)\n";
    return true;
  }

  const auto environment =
      nemotron::RuntimeEnvironment::BuildFromManifestFile(
          manifest_path,
          make_options());
  if (!expect(static_cast<bool>(environment), "runtime environment should build")) {
    return false;
  }
  if (!expect(environment->has_model_schedule(), "runtime environment should expose model schedule") ||
      !expect(environment->has_kernel_catalog(), "runtime environment should expose kernel catalog") ||
      !expect(environment->has_gemm_catalog(), "runtime environment should expose gemm catalog")) {
    return false;
  }

  const auto* layer = environment->model_schedule()->FindLayer(3);
  if (!expect(layer != nullptr, "layer 3 should exist in model schedule")) {
    return false;
  }

  const nemotron::SingleTokenForwardConfig config =
      nemotron::KnownNemotron3Super120BA12BConfig();
  const auto bindings = nemotron::BuildExpertLayerBindings(
      *layer,
      *environment->kernel_catalog(),
      *environment->gemm_catalog(),
      config.n_routed_experts);
  if (!expect(bindings.has_value(), "layer 3 expert bindings should build")) {
    return false;
  }

  nemotron::ExpertLayerConfig layer_config;
  layer_config.layer_index = 3;
  layer_config.hidden_size = config.hidden_size;
  layer_config.moe_latent_size = config.moe_latent_size;
  layer_config.routed_expert_intermediate_size = config.routed_expert_intermediate_size;
  layer_config.shared_expert_intermediate_size = config.shared_expert_intermediate_size;
  layer_config.n_routed_experts = config.n_routed_experts;
  layer_config.top_k = config.experts_per_token;
  layer_config.n_group = config.expert_n_group;
  layer_config.topk_group = config.expert_topk_group;
  layer_config.rms_epsilon = config.layer_norm_epsilon;
  layer_config.routed_scaling_factor = config.routed_scaling_factor;
  layer_config.norm_topk_prob = config.norm_topk_prob;

  auto slice = nemotron::ExpertLayerSlice::Create(layer_config, *bindings);
  if (!expect(slice != nullptr && slice->valid(), "layer 3 expert slice should create")) {
    return false;
  }

  auto cublas = nemotron::CublasLtHandle::Create();
  if (!expect(cublas != nullptr && cublas->valid(), "cublasLt handle should create")) {
    return false;
  }
  nemotron::GemmHeuristicCache heuristic_cache;

  auto batch_input_tensor =
      nemotron::DeviceTensorFp32::Create(
          {metadata->runtime_token_count, metadata->hidden_size});
  auto batch_output_tensor =
      nemotron::DeviceTensorFp32::Create(
          {metadata->runtime_token_count, metadata->hidden_size});
  if (!expect(batch_input_tensor != nullptr && batch_input_tensor->valid(), "batched input tensor should allocate") ||
      !expect(batch_output_tensor != nullptr && batch_output_tensor->valid(), "batched output tensor should allocate") ||
      !expect(
          batch_input_tensor->CopyFromHost(layer2_input.data(), layer2_input.size()),
          "batched layer2 input should upload") ||
      !expect(
          slice->Run(
              *cublas,
              &heuristic_cache,
              *batch_input_tensor,
              batch_output_tensor.get(),
              nullptr),
          "batched layer3 expert slice should run")) {
    return false;
  }

  std::vector<float> batch_output(layer3_expected.size(), 0.0f);
  if (!expect(
          batch_output_tensor->CopyToHost(batch_output.data(), batch_output.size()),
          "batched layer3 output should download")) {
    return false;
  }

  auto row_input_tensor =
      nemotron::DeviceTensorFp32::Create({1, metadata->hidden_size});
  auto row_output_tensor =
      nemotron::DeviceTensorFp32::Create({1, metadata->hidden_size});
  if (!expect(row_input_tensor != nullptr && row_input_tensor->valid(), "row input tensor should allocate") ||
      !expect(row_output_tensor != nullptr && row_output_tensor->valid(), "row output tensor should allocate")) {
    return false;
  }

  std::vector<float> sequential_output(layer3_expected.size(), 0.0f);
  std::vector<float> single_row_input(metadata->hidden_size, 0.0f);
  std::vector<float> single_row_output(metadata->hidden_size, 0.0f);
  for (std::size_t token = 0; token < metadata->runtime_token_count; ++token) {
    std::copy_n(
        layer2_input.data() + token * metadata->hidden_size,
        metadata->hidden_size,
        single_row_input.data());
    if (!expect(
            row_input_tensor->CopyFromHost(
                single_row_input.data(),
                single_row_input.size()),
            "row input should upload") ||
        !expect(
            slice->Run(
                *cublas,
                &heuristic_cache,
                *row_input_tensor,
                row_output_tensor.get(),
                nullptr),
            "row-by-row layer3 expert slice should run") ||
        !expect(
            row_output_tensor->CopyToHost(
                single_row_output.data(),
                single_row_output.size()),
            "row output should download")) {
      return false;
    }
    std::copy(
        single_row_output.begin(),
        single_row_output.end(),
        sequential_output.begin() + token * metadata->hidden_size);
  }

  const float batch_vs_expected = max_abs_diff(batch_output, layer3_expected);
  const float sequential_vs_expected = max_abs_diff(sequential_output, layer3_expected);
  const float batch_vs_sequential = max_abs_diff(batch_output, sequential_output);

  std::cout << "expert_layer3_prefix_replay_test:"
            << " batch_vs_expected=" << batch_vs_expected
            << " sequential_vs_expected=" << sequential_vs_expected
            << " batch_vs_sequential=" << batch_vs_sequential
            << "\n";

  if (debug) {
    std::cerr << "expert_layer3_prefix_replay_test:"
              << " batch_vs_expected=" << batch_vs_expected
              << " sequential_vs_expected=" << sequential_vs_expected
              << " batch_vs_sequential=" << batch_vs_sequential
              << "\n";
  }

  return expect(
             batch_vs_expected <= 3.0e-3f,
             "batched layer3 expert replay should match prefix oracle") &&
         expect(
             sequential_vs_expected <= 3.0e-3f,
             "row-by-row layer3 expert replay should match prefix oracle") &&
         expect(
             batch_vs_sequential <= 1.0e-3f,
             "batched and row-by-row layer3 expert replay should match");
}

}  // namespace

int main() {
  return run_expert_layer3_prefix_replay() ? 0 : 1;
}
