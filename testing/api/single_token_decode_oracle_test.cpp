#include "nemotron/device_tensor.h"
#include "nemotron/mamba_layer.h"
#include "nemotron/model_schedule.h"
#include "nemotron/runtime_environment.h"
#include "nemotron/single_token_forward_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr float kDecodeAbsTol = 3.0e-3f;
constexpr float kDecodeRelL2Tol = 2.0e-2f;
constexpr float kDecodeFunctionalLogitsRelL2Tol = 2.0e-1f;
constexpr std::size_t kDecodeTop5RequiredOverlap = 4;
constexpr std::size_t kDecodeTop10RequiredOverlap = 9;

constexpr std::size_t GiB(std::size_t value) {
  return value * 1024ull * 1024ull * 1024ull;
}

struct FixtureMetadata {
  std::size_t hidden_size = 0;
  std::size_t runtime_token_count = 0;
  std::vector<std::int32_t> runtime_token_ids;
  std::vector<std::size_t> capture_layers;
  std::size_t stop_layer = 0;
  std::size_t vocab_size = 0;
  std::optional<std::int32_t> selected_token_id;
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
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream input(path);
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

void write_float_file(const std::filesystem::path& path, const std::vector<float>& values) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
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

std::optional<std::size_t> parse_env_uint(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || std::string(value).empty()) {
    return std::nullopt;
  }
  try {
    return static_cast<std::size_t>(std::stoull(value));
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<std::vector<std::size_t>> parse_env_uint_list(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || std::string(value).empty()) {
    return std::nullopt;
  }
  std::vector<std::size_t> values;
  std::string token;
  std::stringstream stream(value);
  while (std::getline(stream, token, ',')) {
    if (token.empty()) {
      continue;
    }
    try {
      values.push_back(static_cast<std::size_t>(std::stoull(token)));
    } catch (...) {
      return std::nullopt;
    }
  }
  return values;
}

std::optional<std::vector<std::size_t>> parse_json_uint_array(const std::string& json, const std::string& key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
  std::smatch match;
  if (!std::regex_search(json, match, pattern) || match.size() != 2) {
    return std::nullopt;
  }
  std::vector<std::size_t> values;
  const std::regex number_pattern("([0-9]+)");
  const std::string payload = match[1].str();
  for (auto it = std::sregex_iterator(payload.begin(), payload.end(), number_pattern);
       it != std::sregex_iterator();
       ++it) {
    try {
      values.push_back(static_cast<std::size_t>(std::stoull((*it)[1].str())));
    } catch (...) {
      return std::nullopt;
    }
  }
  return values;
}

std::optional<std::vector<std::int32_t>> parse_json_int32_array(const std::string& json, const std::string& key) {
  const auto uint_values = parse_json_uint_array(json, key);
  if (!uint_values.has_value()) {
    return std::nullopt;
  }
  std::vector<std::int32_t> values;
  values.reserve(uint_values->size());
  for (const std::size_t value : *uint_values) {
    if (value > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
      return std::nullopt;
    }
    values.push_back(static_cast<std::int32_t>(value));
  }
  return values;
}

std::optional<FixtureMetadata> load_metadata(const std::filesystem::path& root) {
  const std::string json = read_text_file(root / "metadata.json");
  FixtureMetadata metadata;
  const auto hidden_size = parse_json_uint_field(json, "hidden_size");
  const auto runtime_token_count = parse_json_uint_field(json, "runtime_token_count");
  const auto runtime_token_ids = parse_json_int32_array(json, "runtime_token_ids");
  const auto capture_layers = parse_json_uint_array(json, "capture_layers");
  const auto stop_layer = parse_json_uint_field(json, "stop_layer");
  const auto vocab_size = parse_json_uint_field(json, "vocab_size");
  const auto selected_token_id = parse_json_uint_field(json, "selected_token_id");
  if (!hidden_size || !runtime_token_count || !runtime_token_ids || !capture_layers || !stop_layer || !vocab_size) {
    return std::nullopt;
  }
  metadata.hidden_size = *hidden_size;
  metadata.runtime_token_count = *runtime_token_count;
  metadata.runtime_token_ids = std::move(*runtime_token_ids);
  metadata.capture_layers = std::move(*capture_layers);
  metadata.stop_layer = *stop_layer;
  metadata.vocab_size = *vocab_size;
  if (selected_token_id.has_value()) {
    metadata.selected_token_id = static_cast<std::int32_t>(*selected_token_id);
  }
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

std::vector<std::size_t> top_k_indices(const std::vector<float>& values, std::size_t k) {
  if (values.empty() || k == 0) {
    return {};
  }
  k = std::min(k, values.size());
  std::vector<std::size_t> indices(values.size(), 0);
  for (std::size_t i = 0; i < indices.size(); ++i) {
    indices[i] = i;
  }
  std::partial_sort(
      indices.begin(),
      indices.begin() + static_cast<std::ptrdiff_t>(k),
      indices.end(),
      [&values](std::size_t lhs, std::size_t rhs) {
        if (values[lhs] == values[rhs]) {
          return lhs < rhs;
        }
        return values[lhs] > values[rhs];
      });
  indices.resize(k);
  return indices;
}

std::size_t count_index_overlap(
    const std::vector<std::size_t>& lhs,
    const std::vector<std::size_t>& rhs) {
  std::size_t overlap = 0;
  for (const std::size_t lhs_index : lhs) {
    if (std::find(rhs.begin(), rhs.end(), lhs_index) != rhs.end()) {
      ++overlap;
    }
  }
  return overlap;
}

struct FunctionalDecodeSummary {
  std::int32_t runtime_top1_token = -1;
  std::int32_t expected_top1_token = -1;
  std::size_t top5_overlap = 0;
  std::size_t top10_overlap = 0;
};

FunctionalDecodeSummary summarize_functional_decode(
    const std::vector<float>& actual_logits,
    const std::vector<float>& expected_logits) {
  FunctionalDecodeSummary summary;
  const auto actual_top1 = top_k_indices(actual_logits, 1);
  const auto expected_top1 = top_k_indices(expected_logits, 1);
  const auto actual_top5 = top_k_indices(actual_logits, 5);
  const auto expected_top5 = top_k_indices(expected_logits, 5);
  const auto actual_top10 = top_k_indices(actual_logits, 10);
  const auto expected_top10 = top_k_indices(expected_logits, 10);
  if (!actual_top1.empty()) {
    summary.runtime_top1_token = static_cast<std::int32_t>(actual_top1.front());
  }
  if (!expected_top1.empty()) {
    summary.expected_top1_token = static_cast<std::int32_t>(expected_top1.front());
  }
  summary.top5_overlap = count_index_overlap(actual_top5, expected_top5);
  summary.top10_overlap = count_index_overlap(actual_top10, expected_top10);
  return summary;
}

struct LayerComparisonSummary {
  bool ok = true;
  std::optional<std::size_t> first_failing_layer;
  float first_failing_layer_diff = 0.0f;
  float first_failing_layer_rel_l2 = 0.0f;
  std::size_t worst_layer_index = 0;
  float worst_layer_diff = 0.0f;
  float worst_layer_rel_l2 = 0.0f;
};

struct MambaStateComparisonSummary {
  bool any_compared = false;
  std::size_t worst_conv_layer_index = 0;
  float worst_conv_diff = 0.0f;
  float worst_conv_rel_l2 = 0.0f;
  std::size_t worst_ssm_layer_index = 0;
  float worst_ssm_diff = 0.0f;
  float worst_ssm_rel_l2 = 0.0f;
};

std::string layer_file_stem(std::size_t layer_index) {
  return std::to_string(layer_index / 100 % 10) +
         std::to_string(layer_index / 10 % 10) +
         std::to_string(layer_index % 10);
}

std::size_t MambaConvStateElemsPerLayer(const nemotron::SingleTokenForwardConfig& config);
std::size_t MambaSsmStateElemsPerLayer(const nemotron::SingleTokenForwardConfig& config);

std::optional<LayerComparisonSummary> CompareCapturedLayersUpTo(
    const nemotron::SingleTokenForwardTrace& trace,
    const FixtureMetadata& metadata,
    const std::filesystem::path& fixture_root,
    std::size_t effective_stop_layer,
    bool debug) {
  std::unordered_map<std::size_t, std::vector<float>> actual_layers;
  for (const auto& captured : trace.captured_layers) {
    actual_layers.emplace(captured.layer_index, captured.hidden);
  }

  LayerComparisonSummary summary;
  for (const std::size_t layer_index : metadata.capture_layers) {
    if (layer_index > effective_stop_layer) {
      continue;
    }
    const auto it = actual_layers.find(layer_index);
    if (it == actual_layers.end()) {
      std::cerr << "single_token_decode_oracle_test: missing captured layer " << layer_index << "\n";
      return std::nullopt;
    }
    const std::filesystem::path path =
        fixture_root / ("expected_layer_" + layer_file_stem(layer_index) + "_output_fp32.bin");
    const std::vector<float> expected = read_float_file(path);
    if (it->second.size() != expected.size()) {
      std::cerr << "single_token_decode_oracle_test: size mismatch at layer " << layer_index << "\n";
      return std::nullopt;
    }
    const float layer_diff = max_abs_diff(it->second, expected);
    const float layer_rel_l2 = relative_l2_diff(it->second, expected);
    if (debug) {
      std::cerr << "single_token_decode_oracle_test: layer_index=" << layer_index
                << " max_abs_diff=" << layer_diff
                << " rel_l2=" << layer_rel_l2 << "\n";
    }
    if (layer_diff > summary.worst_layer_diff ||
        (layer_diff == summary.worst_layer_diff && layer_rel_l2 > summary.worst_layer_rel_l2)) {
      summary.worst_layer_index = layer_index;
      summary.worst_layer_diff = layer_diff;
      summary.worst_layer_rel_l2 = layer_rel_l2;
    }
    if (!(layer_diff <= kDecodeAbsTol || layer_rel_l2 <= kDecodeRelL2Tol)) {
      summary.ok = false;
      if (!summary.first_failing_layer.has_value()) {
        summary.first_failing_layer = layer_index;
        summary.first_failing_layer_diff = layer_diff;
        summary.first_failing_layer_rel_l2 = layer_rel_l2;
      }
    }
  }
  return summary;
}

const nemotron::ForwardLayerPlanEntry* FindPlanLayer(
    const nemotron::SingleTokenForwardPlan& plan,
    std::size_t layer_index) {
  for (const auto& layer : plan.layers) {
    if (layer.layer_index == layer_index) {
      return &layer;
    }
  }
  return nullptr;
}

std::optional<MambaStateComparisonSummary> CompareCapturedMambaStatesUpTo(
    const nemotron::RequestExecutionContext& request_context,
    const nemotron::SingleTokenForwardPlan& plan,
    const nemotron::SingleTokenForwardConfig& config,
    const FixtureMetadata& metadata,
    const std::filesystem::path& fixture_root,
    std::size_t effective_stop_layer,
    bool debug) {
  const auto* full_conv = request_context.mamba_conv_state();
  const auto* full_ssm = request_context.mamba_state();
  if (full_conv == nullptr || !full_conv->valid() || full_ssm == nullptr || !full_ssm->valid()) {
    return MambaStateComparisonSummary{};
  }

  std::vector<float> full_conv_host(full_conv->numel(), 0.0f);
  std::vector<float> full_ssm_host(full_ssm->numel(), 0.0f);
  if (!full_conv->CopyToHost(full_conv_host.data(), full_conv_host.size()) ||
      !full_ssm->CopyToHost(full_ssm_host.data(), full_ssm_host.size())) {
    return std::nullopt;
  }

  const std::size_t conv_per_layer = MambaConvStateElemsPerLayer(config);
  const std::size_t ssm_per_layer = MambaSsmStateElemsPerLayer(config);
  MambaStateComparisonSummary summary;

  for (const std::size_t layer_index : metadata.capture_layers) {
    if (layer_index > effective_stop_layer) {
      continue;
    }

    const std::filesystem::path expected_conv_path =
        fixture_root / ("expected_layer_" + layer_file_stem(layer_index) + "_mamba_conv_state_fp32.bin");
    const std::filesystem::path expected_ssm_path =
        fixture_root / ("expected_layer_" + layer_file_stem(layer_index) + "_mamba_ssm_state_fp32.bin");
    if (!std::filesystem::exists(expected_conv_path) || !std::filesystem::exists(expected_ssm_path)) {
      continue;
    }

    const auto* layer = FindPlanLayer(plan, layer_index);
    if (layer == nullptr || layer->kind != nemotron::ForwardLayerKind::kMamba) {
      std::cerr << "single_token_decode_oracle_test: missing Mamba plan layer " << layer_index << "\n";
      return std::nullopt;
    }

    const std::size_t conv_start = layer->mamba_conv_state_offset_elems;
    const std::size_t conv_end = conv_start + conv_per_layer;
    const std::size_t ssm_start = layer->mamba_state_offset_elems;
    const std::size_t ssm_end = ssm_start + ssm_per_layer;
    if (conv_end > full_conv_host.size() || ssm_end > full_ssm_host.size()) {
      std::cerr << "single_token_decode_oracle_test: state slice overflow at layer " << layer_index << "\n";
      return std::nullopt;
    }

    const std::vector<float> expected_conv = read_float_file(expected_conv_path);
    const std::vector<float> expected_ssm = read_float_file(expected_ssm_path);
    const std::vector<float> actual_conv(
        full_conv_host.begin() + static_cast<std::ptrdiff_t>(conv_start),
        full_conv_host.begin() + static_cast<std::ptrdiff_t>(conv_end));
    const std::vector<float> actual_ssm(
        full_ssm_host.begin() + static_cast<std::ptrdiff_t>(ssm_start),
        full_ssm_host.begin() + static_cast<std::ptrdiff_t>(ssm_end));
    if (expected_conv.size() != actual_conv.size() || expected_ssm.size() != actual_ssm.size()) {
      std::cerr << "single_token_decode_oracle_test: Mamba state size mismatch at layer "
                << layer_index << "\n";
      return std::nullopt;
    }

    summary.any_compared = true;
    const float conv_diff = max_abs_diff(actual_conv, expected_conv);
    const float conv_rel_l2 = relative_l2_diff(actual_conv, expected_conv);
    const float ssm_diff = max_abs_diff(actual_ssm, expected_ssm);
    const float ssm_rel_l2 = relative_l2_diff(actual_ssm, expected_ssm);
    if (debug) {
      std::cerr << "single_token_decode_oracle_test: mamba_state layer_index=" << layer_index
                << " conv_max_abs_diff=" << conv_diff
                << " conv_rel_l2=" << conv_rel_l2
                << " ssm_max_abs_diff=" << ssm_diff
                << " ssm_rel_l2=" << ssm_rel_l2 << "\n";
    }
    if (conv_diff > summary.worst_conv_diff ||
        (conv_diff == summary.worst_conv_diff && conv_rel_l2 > summary.worst_conv_rel_l2)) {
      summary.worst_conv_layer_index = layer_index;
      summary.worst_conv_diff = conv_diff;
      summary.worst_conv_rel_l2 = conv_rel_l2;
    }
    if (ssm_diff > summary.worst_ssm_diff ||
        (ssm_diff == summary.worst_ssm_diff && ssm_rel_l2 > summary.worst_ssm_rel_l2)) {
      summary.worst_ssm_layer_index = layer_index;
      summary.worst_ssm_diff = ssm_diff;
      summary.worst_ssm_rel_l2 = ssm_rel_l2;
    }
  }

  return summary;
}

void DumpTraceOutputs(
    const nemotron::SingleTokenForwardTrace& trace,
    const std::filesystem::path& dump_root) {
  std::filesystem::create_directories(dump_root);
  if (!trace.embedding_output.empty()) {
    write_float_file(dump_root / "embedding_output_fp32.bin", trace.embedding_output);
  }
  for (const auto& captured : trace.captured_layers) {
    const std::string name = "layer_" + layer_file_stem(captured.layer_index) + "_output_fp32.bin";
    write_float_file(dump_root / name, captured.hidden);
  }
  if (!trace.final_hidden.empty()) {
    write_float_file(dump_root / "final_hidden_fp32.bin", trace.final_hidden);
  }
  if (!trace.final_hidden_normed.empty()) {
    write_float_file(dump_root / "final_hidden_normed_fp32.bin", trace.final_hidden_normed);
  }
  if (!trace.logits.empty()) {
    write_float_file(dump_root / "logits_fp32.bin", trace.logits);
  }
}

std::size_t MambaConvStateElemsPerLayer(const nemotron::SingleTokenForwardConfig& config) {
  const std::size_t conv_dim =
      config.mamba_intermediate_size + (2 * config.mamba_n_groups * config.mamba_state_size);
  return conv_dim * config.mamba_conv_kernel_size;
}

std::size_t MambaSsmStateElemsPerLayer(const nemotron::SingleTokenForwardConfig& config) {
  return config.mamba_num_heads * config.mamba_head_dim * config.mamba_state_size;
}

void DumpRequestState(
    const nemotron::RequestExecutionContext& request_context,
    const nemotron::SingleTokenForwardPlan& plan,
    const nemotron::SingleTokenForwardConfig& config,
    const std::filesystem::path& dump_root) {
  if (const auto* full_conv = request_context.mamba_conv_state();
      full_conv != nullptr && full_conv->valid()) {
    std::vector<float> host(full_conv->numel(), 0.0f);
    if (full_conv->CopyToHost(host.data(), host.size())) {
      write_float_file(dump_root / "mamba_conv_state_full_fp32.bin", host);
      const std::size_t per_layer = MambaConvStateElemsPerLayer(config);
      for (const auto& layer : plan.layers) {
        if (layer.kind != nemotron::ForwardLayerKind::kMamba) {
          continue;
        }
        const std::size_t start = layer.mamba_conv_state_offset_elems;
        const std::size_t end = start + per_layer;
        if (end > host.size()) {
          continue;
        }
        write_float_file(
            dump_root / ("mamba_layer_" + layer_file_stem(layer.layer_index) + "_conv_state_fp32.bin"),
            std::vector<float>(host.begin() + static_cast<std::ptrdiff_t>(start),
                               host.begin() + static_cast<std::ptrdiff_t>(end)));
      }
    }
  }
  if (const auto* full_ssm = request_context.mamba_state();
      full_ssm != nullptr && full_ssm->valid()) {
    std::vector<float> host(full_ssm->numel(), 0.0f);
    if (full_ssm->CopyToHost(host.data(), host.size())) {
      write_float_file(dump_root / "mamba_ssm_state_full_fp32.bin", host);
      const std::size_t per_layer = MambaSsmStateElemsPerLayer(config);
      for (const auto& layer : plan.layers) {
        if (layer.kind != nemotron::ForwardLayerKind::kMamba) {
          continue;
        }
        const std::size_t start = layer.mamba_state_offset_elems;
        const std::size_t end = start + per_layer;
        if (end > host.size()) {
          continue;
        }
        write_float_file(
            dump_root / ("mamba_layer_" + layer_file_stem(layer.layer_index) + "_ssm_state_fp32.bin"),
            std::vector<float>(host.begin() + static_cast<std::ptrdiff_t>(start),
                               host.begin() + static_cast<std::ptrdiff_t>(end)));
      }
    }
  }
}

nemotron::RuntimeBootstrapOptions make_options() {
  nemotron::RuntimeBootstrapOptions options;
  options.service_target.total_memory_bytes = GiB(128);
  options.service_target.weights_bytes = GiB(100);
  options.service_target.workspace_bytes = GiB(8);
  options.service_target.graph_bytes = GiB(4);
  options.service_target.safety_headroom_bytes = GiB(4);
  options.service_target.target_active_requests = 1;
  options.service_target.target_context_tokens = 1;
  options.use_fp16_mamba_state = false;
  options.reusable_node_metadata_bytes = 4096;
  options.verify_manifest_files = false;
  options.materialize_weight_arena = false;
  return options;
}

bool run_single_token_decode_oracle() {
  const bool debug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  const char* dump_root_env = std::getenv("NEMOTRON_SINGLE_TOKEN_DECODE_DUMP_ROOT");
  const char* manifest_env = std::getenv("NEMOTRON_FORWARD_MANIFEST");
  if (manifest_env == nullptr || std::string(manifest_env).empty()) {
    std::cout << "single_token_decode_oracle_test: SKIP (NEMOTRON_FORWARD_MANIFEST is unset)\n";
    return true;
  }
  if (!has_cuda_device()) {
    std::cout << "single_token_decode_oracle_test: SKIP (no CUDA device available)\n";
    return true;
  }

  const std::filesystem::path fixture_root(NEMOTRON_SINGLE_TOKEN_DECODE_ORACLE_FIXTURE_ROOT);
  const auto metadata = load_metadata(fixture_root);
  if (!expect(metadata.has_value(), "single-token decode oracle metadata should load")) {
    return false;
  }
  if (debug) {
    std::cerr << "single_token_decode_oracle_test: fixture_root=" << fixture_root
              << " capture_layer_count=" << metadata->capture_layers.size() << "\n";
  }
  if (!expect(metadata->selected_token_id.has_value(), "single-token fixture should include selected_token_id")) {
    return false;
  }
  const auto stop_layer_override = parse_env_uint("NEMOTRON_SINGLE_TOKEN_DECODE_STOP_LAYER");
  const auto stop_layer_sweep = parse_env_uint_list("NEMOTRON_SINGLE_TOKEN_DECODE_SWEEP_LAYERS");
  const std::size_t effective_stop_layer =
      stop_layer_override.has_value() ? *stop_layer_override : metadata->stop_layer;

  const std::filesystem::path manifest_path(manifest_env);
  if (!std::filesystem::exists(manifest_path)) {
    std::cout << "single_token_decode_oracle_test: SKIP (manifest path does not exist)\n";
    return true;
  }

  const auto environment = nemotron::RuntimeEnvironment::BuildFromManifestFile(manifest_path, make_options());
  if (!expect(static_cast<bool>(environment), "runtime environment should build for single-token decode oracle")) {
    return false;
  }

  nemotron::SingleTokenForwardConfig config = nemotron::KnownNemotron3Nano30BA3BConfig();
  config.max_tokens = 1;
  auto model = nemotron::SingleTokenForwardModel::Create(*environment, config);
  if (!expect(model != nullptr && model->valid(), "forward model should build for single-token decode oracle")) {
    return false;
  }

  auto request_context = model->CreateRequestContext();
  if (!expect(request_context != nullptr && request_context->valid(), "request context should create")) {
    return false;
  }

  auto logits = nemotron::DeviceTensorFp32::Create({1, config.vocab_size});
  if (!expect(logits != nullptr && logits->valid(), "logits tensor should create")) {
    return false;
  }

  nemotron::SingleTokenForwardTrace trace;
  if (stop_layer_sweep.has_value()) {
    bool sweep_ok = true;
    for (const std::size_t sweep_stop : *stop_layer_sweep) {
      auto sweep_context = model->CreateRequestContext();
      auto sweep_logits = nemotron::DeviceTensorFp32::Create({1, config.vocab_size});
      nemotron::SingleTokenForwardTrace sweep_trace;
      const bool ran =
          sweep_context != nullptr &&
          sweep_context->valid() &&
          sweep_logits != nullptr &&
          sweep_logits->valid() &&
          model->RunSingleToken(
              *metadata->selected_token_id,
              *sweep_context,
              sweep_logits.get(),
              metadata->capture_layers,
              &sweep_trace,
              sweep_stop);
      if (!ran) {
        std::cerr << "single_token_decode_oracle_test: sweep_stop=" << sweep_stop
                  << " execution_failed\n";
        sweep_ok = false;
        continue;
      }
      if (dump_root_env != nullptr && std::string(dump_root_env).size() != 0) {
        const auto dump_root = std::filesystem::path(dump_root_env) / ("stop_" + std::to_string(sweep_stop));
        DumpTraceOutputs(sweep_trace, dump_root);
        DumpRequestState(*sweep_context, model->plan(), config, dump_root);
      }
      const auto layer_summary =
          CompareCapturedLayersUpTo(sweep_trace, *metadata, fixture_root, sweep_stop, debug);
      const auto mamba_state_summary =
          CompareCapturedMambaStatesUpTo(*sweep_context, model->plan(), config, *metadata, fixture_root, sweep_stop, debug);
      if (!layer_summary.has_value()) {
        sweep_ok = false;
        continue;
      }
      if (!mamba_state_summary.has_value()) {
        sweep_ok = false;
        continue;
      }
      if (layer_summary->ok) {
        std::cerr << "single_token_decode_oracle_test: sweep_stop=" << sweep_stop
                  << " ok"
                  << " worst_layer_index=" << layer_summary->worst_layer_index
                  << " worst_layer_max_abs_diff=" << layer_summary->worst_layer_diff
                  << " worst_layer_rel_l2=" << layer_summary->worst_layer_rel_l2;
        if (mamba_state_summary->any_compared) {
          std::cerr << " worst_mamba_conv_layer=" << mamba_state_summary->worst_conv_layer_index
                    << " worst_mamba_conv_max_abs_diff=" << mamba_state_summary->worst_conv_diff
                    << " worst_mamba_conv_rel_l2=" << mamba_state_summary->worst_conv_rel_l2
                    << " worst_mamba_ssm_layer=" << mamba_state_summary->worst_ssm_layer_index
                    << " worst_mamba_ssm_max_abs_diff=" << mamba_state_summary->worst_ssm_diff
                    << " worst_mamba_ssm_rel_l2=" << mamba_state_summary->worst_ssm_rel_l2;
        }
        std::cerr << "\n";
      } else {
        std::cerr << "single_token_decode_oracle_test: sweep_stop=" << sweep_stop
                  << " first_failing_layer=" << *layer_summary->first_failing_layer
                  << " max_abs_diff=" << layer_summary->first_failing_layer_diff
                  << " rel_l2=" << layer_summary->first_failing_layer_rel_l2
                  << " worst_layer_index=" << layer_summary->worst_layer_index
                  << " worst_layer_max_abs_diff=" << layer_summary->worst_layer_diff
                  << " worst_layer_rel_l2=" << layer_summary->worst_layer_rel_l2;
        if (mamba_state_summary->any_compared) {
          std::cerr << " worst_mamba_conv_layer=" << mamba_state_summary->worst_conv_layer_index
                    << " worst_mamba_conv_max_abs_diff=" << mamba_state_summary->worst_conv_diff
                    << " worst_mamba_conv_rel_l2=" << mamba_state_summary->worst_conv_rel_l2
                    << " worst_mamba_ssm_layer=" << mamba_state_summary->worst_ssm_layer_index
                    << " worst_mamba_ssm_max_abs_diff=" << mamba_state_summary->worst_ssm_diff
                    << " worst_mamba_ssm_rel_l2=" << mamba_state_summary->worst_ssm_rel_l2;
        }
        std::cerr << "\n";
        sweep_ok = false;
      }
    }
    return sweep_ok;
  }

  if (!expect(
          model->RunSingleToken(
              *metadata->selected_token_id,
              *request_context,
              logits.get(),
              metadata->capture_layers,
              &trace,
              effective_stop_layer),
          "single-token forward path should execute against the real manifest")) {
    return false;
  }
  if (dump_root_env != nullptr && std::string(dump_root_env).size() != 0) {
    const auto dump_root = std::filesystem::path(dump_root_env);
    DumpTraceOutputs(trace, dump_root);
    DumpRequestState(*request_context, model->plan(), config, dump_root);
  }

  const std::vector<float> expected_embedding =
      read_float_file(fixture_root / "expected_embedding_output_fp32.bin");
  const float embedding_diff = max_abs_diff(trace.embedding_output, expected_embedding);
  const float embedding_rel_l2 = relative_l2_diff(trace.embedding_output, expected_embedding);
  if (debug) {
    std::cerr << "single_token_decode_oracle_test: embedding max_abs_diff="
              << embedding_diff << " rel_l2=" << embedding_rel_l2 << "\n";
  }
  if (!expect(trace.embedding_output.size() == expected_embedding.size(), "embedding output size should match fixture") ||
      !expect(
          embedding_diff <= kDecodeAbsTol || embedding_rel_l2 <= kDecodeRelL2Tol,
          "embedding output should match fixture")) {
    return false;
  }

  const auto layer_summary =
      CompareCapturedLayersUpTo(trace, *metadata, fixture_root, effective_stop_layer, debug);
  if (!expect(layer_summary.has_value(), "captured layers should compare against fixture")) {
    return false;
  }
  if (!layer_summary->ok) {
    std::cerr << "single_token_decode_oracle_test: first_failing_layer="
              << *layer_summary->first_failing_layer
              << " max_abs_diff=" << layer_summary->first_failing_layer_diff
              << " rel_l2=" << layer_summary->first_failing_layer_rel_l2
              << " worst_layer_index=" << layer_summary->worst_layer_index
              << " worst_layer_max_abs_diff=" << layer_summary->worst_layer_diff
              << " worst_layer_rel_l2=" << layer_summary->worst_layer_rel_l2 << "\n";
  }

  if (effective_stop_layer < metadata->stop_layer) {
    if (debug) {
      std::cerr << "single_token_decode_oracle_test: stop-layer override active, "
                << "skipping final hidden / norm / logits checks at layer "
                << effective_stop_layer << "\n";
    }
    return true;
  }

  const std::vector<float> expected_final_hidden =
      read_float_file(fixture_root / "expected_final_hidden_fp32.bin");
  const float final_hidden_diff = max_abs_diff(trace.final_hidden, expected_final_hidden);
  const float final_hidden_rel_l2 = relative_l2_diff(trace.final_hidden, expected_final_hidden);
  if (debug) {
    std::cerr << "single_token_decode_oracle_test: final_hidden max_abs_diff="
              << final_hidden_diff << " rel_l2=" << final_hidden_rel_l2 << "\n";
  }
  if (!expect(trace.final_hidden.size() == expected_final_hidden.size(), "final hidden size should match fixture")) {
    return false;
  }

  const std::vector<float> expected_final_hidden_normed =
      read_float_file(fixture_root / "expected_final_hidden_normed_fp32.bin");
  const float final_hidden_normed_diff =
      max_abs_diff(trace.final_hidden_normed, expected_final_hidden_normed);
  const float final_hidden_normed_rel_l2 =
      relative_l2_diff(trace.final_hidden_normed, expected_final_hidden_normed);
  if (debug) {
    std::cerr << "single_token_decode_oracle_test: final_hidden_normed max_abs_diff="
              << final_hidden_normed_diff << " rel_l2=" << final_hidden_normed_rel_l2 << "\n";
  }
  if (!expect(
          trace.final_hidden_normed.size() == expected_final_hidden_normed.size(),
          "final hidden normed size should match fixture")) {
    return false;
  }

  std::vector<float> actual_logits(logits->numel(), 0.0f);
  if (!expect(logits->CopyToHost(actual_logits.data(), actual_logits.size()), "logits should download")) {
    return false;
  }
  const std::vector<float> expected_logits =
      read_float_file(fixture_root / "expected_logits_fp32.bin");
  const float logits_diff = max_abs_diff(actual_logits, expected_logits);
  const float logits_rel_l2 = relative_l2_diff(actual_logits, expected_logits);
  if (debug) {
    std::cerr << "single_token_decode_oracle_test: logits max_abs_diff="
              << logits_diff << " rel_l2=" << logits_rel_l2 << "\n";
  }
  if (!expect(actual_logits.size() == expected_logits.size(), "logits size should match fixture")) {
    return false;
  }

  const FunctionalDecodeSummary functional =
      summarize_functional_decode(actual_logits, expected_logits);
  if (debug || !layer_summary->ok) {
    std::cerr << "single_token_decode_oracle_test: functional_decode"
              << " runtime_top1=" << functional.runtime_top1_token
              << " expected_top1=" << functional.expected_top1_token
              << " top5_overlap=" << functional.top5_overlap
              << " top10_overlap=" << functional.top10_overlap
              << " logits_rel_l2=" << logits_rel_l2
              << " final_hidden_rel_l2=" << final_hidden_rel_l2
              << " final_hidden_normed_rel_l2=" << final_hidden_normed_rel_l2 << "\n";
  }

  if (!expect(
          functional.expected_top1_token >= 0,
          "expected logits should produce a valid top-1 token") ||
      !expect(
          functional.runtime_top1_token == functional.expected_top1_token,
          "runtime top-1 token should match the oracle token") ||
      !expect(
          functional.top5_overlap >= kDecodeTop5RequiredOverlap,
          "runtime top-5 overlap should stay within the decode oracle envelope") ||
      !expect(
          functional.top10_overlap >= kDecodeTop10RequiredOverlap,
          "runtime top-10 overlap should stay within the decode oracle envelope") ||
      !expect(
          logits_rel_l2 <= kDecodeFunctionalLogitsRelL2Tol,
          "final logits rel_l2 should stay within the functional decode tripwire")) {
    return false;
  }

  return true;
}

}  // namespace

int main() {
  return run_single_token_decode_oracle() ? 0 : 1;
}
