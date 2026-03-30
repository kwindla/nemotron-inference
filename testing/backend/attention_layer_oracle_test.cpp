#include "nemotron/attention_layer.h"

#include <cuda_bf16.h>

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

using nemotron::AttentionKvCacheConfig;
using nemotron::AttentionLayerBindings;
using nemotron::AttentionLayerConfig;
using nemotron::AttentionLayerSlice;
using nemotron::CublasLtHandle;
using nemotron::CudnnHandle;
using nemotron::DeviceTensorFp32;
using nemotron::GemmDescriptor;
using nemotron::GemmHeuristicCache;
using nemotron::GemmKernelFamily;
using nemotron::KernelTensorDescriptor;
using nemotron::KvCacheDataType;
using nemotron::RequestExecutionConfig;
using nemotron::RequestExecutionContext;

struct FixtureMetadata {
  std::size_t layer_index = 0;
  std::size_t token_count = 0;
  std::size_t tokens_per_page = 0;
  std::size_t total_pages = 0;
  std::size_t hidden_size = 0;
  std::size_t query_head_count = 0;
  std::size_t kv_head_count = 0;
  std::size_t head_dim = 0;
  float rms_epsilon = 0.0f;
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
  const std::filesystem::path metadata_path = root / "metadata.json";
  if (!std::filesystem::exists(metadata_path)) {
    return std::nullopt;
  }
  const std::string json = read_text_file(metadata_path);
  FixtureMetadata metadata;
  const auto layer_index = parse_json_uint_field(json, "layer_index");
  const auto token_count = parse_json_uint_field(json, "token_count");
  const auto tokens_per_page = parse_json_uint_field(json, "tokens_per_page");
  const auto total_pages = parse_json_uint_field(json, "total_pages");
  const auto hidden_size = parse_json_uint_field(json, "hidden_size");
  const auto query_head_count = parse_json_uint_field(json, "query_head_count");
  const auto kv_head_count = parse_json_uint_field(json, "kv_head_count");
  const auto head_dim = parse_json_uint_field(json, "head_dim");
  const auto rms_epsilon = parse_json_float_field(json, "rms_epsilon");
  if (!layer_index || !token_count || !tokens_per_page || !total_pages || !hidden_size ||
      !query_head_count || !kv_head_count || !head_dim || !rms_epsilon) {
    return std::nullopt;
  }
  metadata.layer_index = *layer_index;
  metadata.token_count = *token_count;
  metadata.tokens_per_page = *tokens_per_page;
  metadata.total_pages = *total_pages;
  metadata.hidden_size = *hidden_size;
  metadata.query_head_count = *query_head_count;
  metadata.kv_head_count = *kv_head_count;
  metadata.head_dim = *head_dim;
  metadata.rms_epsilon = *rms_epsilon;
  return metadata;
}

GemmDescriptor make_dense_descriptor(
    const std::string& name,
    const std::vector<float>& weights,
    std::size_t rows,
    std::size_t cols) {
  GemmDescriptor descriptor;
  descriptor.tensor_name = name;
  descriptor.op_class = "dense";
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

KernelTensorDescriptor make_norm_descriptor(const std::vector<float>& weights) {
  KernelTensorDescriptor descriptor;
  descriptor.tensor_name = "backbone.layers.7.norm.weight";
  descriptor.op_class = "norm";
  descriptor.logical_shape = {weights.size()};
  descriptor.packed_shape = {weights.size()};
  descriptor.storage_dtype = "fp32";
  descriptor.compute_dtype = "fp32";
  descriptor.layout_tag = "row_major";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data = reinterpret_cast<const std::uint8_t*>(weights.data());
  descriptor.packed_nbytes = weights.size() * sizeof(float);
  return descriptor;
}

std::vector<float> bf16_to_float(const std::vector<__nv_bfloat16>& values) {
  std::vector<float> output(values.size(), 0.0f);
  for (std::size_t i = 0; i < values.size(); ++i) {
    output[i] = __bfloat162float(values[i]);
  }
  return output;
}

std::vector<float> gather_page_major_cache(
    const std::vector<float>& full_cache,
    const std::vector<nemotron::KvPageHandle>& pages,
    std::size_t kv_head_count,
    std::size_t tokens_per_page,
    std::size_t head_dim) {
  const std::size_t page_stride = kv_head_count * tokens_per_page * head_dim;
  std::vector<float> gathered(pages.size() * page_stride, 0.0f);
  for (std::size_t page_index = 0; page_index < pages.size(); ++page_index) {
    const std::size_t src_offset = pages[page_index].page_id * page_stride;
    const std::size_t dst_offset = page_index * page_stride;
    std::copy_n(full_cache.begin() + src_offset, page_stride, gathered.begin() + dst_offset);
  }
  return gathered;
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

bool run_attention_oracle_fixture() {
  const auto cublas = CublasLtHandle::Create();
  const auto cudnn = CudnnHandle::Create();
  if (!cublas || !cublas->valid() || !cudnn || !cudnn->valid() || cudnn->version() < 90500) {
    std::cout << "attention_layer_oracle_test: SKIP (no CUDA device or cuDNN >= 9.5 unavailable)\n";
    return true;
  }

  const std::filesystem::path fixture_root = resolve_fixture_root(
      "NEMOTRON_ATTENTION_ORACLE_FIXTURE_ROOT_OVERRIDE",
      NEMOTRON_ATTENTION_ORACLE_FIXTURE_ROOT);
  const auto metadata = load_metadata(fixture_root);
  if (!expect(metadata.has_value(), "attention oracle fixture metadata should load")) {
    return false;
  }

  const std::vector<float> input_hidden = read_float_file(fixture_root / "input_hidden_fp32.bin");
  const std::vector<float> norm_weight = read_float_file(fixture_root / "norm_weight_fp32.bin");
  const std::vector<float> q_weight = read_float_file(fixture_root / "q_proj_weight_fp32.bin");
  const std::vector<float> k_weight = read_float_file(fixture_root / "k_proj_weight_fp32.bin");
  const std::vector<float> v_weight = read_float_file(fixture_root / "v_proj_weight_fp32.bin");
  const std::vector<float> o_weight = read_float_file(fixture_root / "o_proj_weight_fp32.bin");
  const std::vector<float> expected_final_output = read_float_file(fixture_root / "expected_final_output_fp32.bin");
  const std::vector<float> expected_key_cache = read_float_file(fixture_root / "expected_key_cache_fp32.bin");
  const std::vector<float> expected_value_cache = read_float_file(fixture_root / "expected_value_cache_fp32.bin");

  if (!expect(input_hidden.size() == metadata->token_count * metadata->hidden_size, "input hidden size should match metadata") ||
      !expect(norm_weight.size() == metadata->hidden_size, "norm weight size should match metadata") ||
      !expect(q_weight.size() == metadata->hidden_size * metadata->hidden_size, "q weight size should match metadata") ||
      !expect(k_weight.size() == metadata->kv_head_count * metadata->head_dim * metadata->hidden_size, "k weight size should match metadata") ||
      !expect(v_weight.size() == metadata->kv_head_count * metadata->head_dim * metadata->hidden_size, "v weight size should match metadata") ||
      !expect(o_weight.size() == metadata->hidden_size * metadata->hidden_size, "o weight size should match metadata") ||
      !expect(expected_final_output.size() == input_hidden.size(), "expected final output size should match input size") ||
      !expect(
          expected_key_cache.size() == metadata->total_pages * metadata->kv_head_count * metadata->tokens_per_page * metadata->head_dim,
          "expected key cache size should match metadata") ||
      !expect(expected_value_cache.size() == expected_key_cache.size(), "expected value cache size should match metadata")) {
    return false;
  }

  AttentionLayerBindings bindings;
  const auto norm_descriptor = make_norm_descriptor(norm_weight);
  const auto q_descriptor = make_dense_descriptor(
      "backbone.layers.7.mixer.q_proj.weight",
      q_weight,
      metadata->hidden_size,
      metadata->hidden_size);
  const auto k_descriptor = make_dense_descriptor(
      "backbone.layers.7.mixer.k_proj.weight",
      k_weight,
      metadata->kv_head_count * metadata->head_dim,
      metadata->hidden_size);
  const auto v_descriptor = make_dense_descriptor(
      "backbone.layers.7.mixer.v_proj.weight",
      v_weight,
      metadata->kv_head_count * metadata->head_dim,
      metadata->hidden_size);
  const auto o_descriptor = make_dense_descriptor(
      "backbone.layers.7.mixer.o_proj.weight",
      o_weight,
      metadata->hidden_size,
      metadata->hidden_size);
  bindings.norm_weight = &norm_descriptor;
  bindings.q_proj = &q_descriptor;
  bindings.k_proj = &k_descriptor;
  bindings.v_proj = &v_descriptor;
  bindings.o_proj = &o_descriptor;

  AttentionLayerConfig layer_config;
  layer_config.layer_index = metadata->layer_index;
  layer_config.hidden_size = metadata->hidden_size;
  layer_config.query_head_count = metadata->query_head_count;
  layer_config.kv_head_count = metadata->kv_head_count;
  layer_config.head_dim = metadata->head_dim;
  layer_config.rms_epsilon = metadata->rms_epsilon;

  auto slice = AttentionLayerSlice::Create(layer_config, bindings);
  if (!expect(slice != nullptr && slice->valid(), "attention slice should build for oracle fixture")) {
    return false;
  }

  RequestExecutionConfig request_config;
  request_config.hidden_size = metadata->hidden_size;
  request_config.max_tokens = metadata->token_count;
  request_config.scratch_tokens = metadata->token_count;
  request_config.attention_kv_cache = AttentionKvCacheConfig{
      metadata->layer_index + 1,
      metadata->kv_head_count,
      metadata->head_dim,
      metadata->tokens_per_page,
      KvCacheDataType::kBf16,
  };
  request_config.attention_total_pages = metadata->total_pages * (metadata->layer_index + 1);

  auto request = RequestExecutionContext::Create(request_config);
  auto input = DeviceTensorFp32::Create({metadata->token_count, metadata->hidden_size});
  auto output = DeviceTensorFp32::Create({metadata->token_count, metadata->hidden_size});
  if (!expect(request != nullptr && request->valid(), "request context should create for attention oracle") ||
      !expect(input != nullptr && input->valid(), "input tensor should create for attention oracle") ||
      !expect(output != nullptr && output->valid(), "output tensor should create for attention oracle")) {
    return false;
  }
  if (!expect(input->CopyFromHost(input_hidden.data(), input_hidden.size()), "fixture input should upload")) {
    return false;
  }

  GemmHeuristicCache heuristic_cache;
  if (!expect(
          slice->Run(*cublas, *cudnn, &heuristic_cache, *request, *input, output.get()),
          "attention oracle slice should execute")) {
    return false;
  }

  std::vector<float> actual_output(expected_final_output.size(), 0.0f);
  if (!expect(output->CopyToHost(actual_output.data(), actual_output.size()), "final output should download")) {
    return false;
  }

  std::vector<__nv_bfloat16> key_cache_bf16(request->key_cache()->numel());
  std::vector<__nv_bfloat16> value_cache_bf16(request->value_cache()->numel());
  if (!expect(
          request->key_cache()->CopyToHost(key_cache_bf16.data(), key_cache_bf16.size()),
          "key cache should download") ||
      !expect(
          request->value_cache()->CopyToHost(value_cache_bf16.data(), value_cache_bf16.size()),
          "value cache should download")) {
    return false;
  }

  const std::vector<float> actual_key_cache = bf16_to_float(key_cache_bf16);
  const std::vector<float> actual_value_cache = bf16_to_float(value_cache_bf16);
  const auto* layer_pages = request->kv_pages(metadata->layer_index);
  if (!expect(layer_pages != nullptr, "layer page handles should be available") ||
      !expect(layer_pages->size() == metadata->total_pages, "layer page count should match metadata")) {
    return false;
  }
  const std::vector<float> actual_key_cache_layer = gather_page_major_cache(
      actual_key_cache,
      *layer_pages,
      metadata->kv_head_count,
      metadata->tokens_per_page,
      metadata->head_dim);
  const std::vector<float> actual_value_cache_layer = gather_page_major_cache(
      actual_value_cache,
      *layer_pages,
      metadata->kv_head_count,
      metadata->tokens_per_page,
      metadata->head_dim);

  const float output_diff = max_abs_diff(actual_output, expected_final_output);
  const float key_cache_diff = max_abs_diff(actual_key_cache_layer, expected_key_cache);
  const float value_cache_diff = max_abs_diff(actual_value_cache_layer, expected_value_cache);
  if (!expect(output_diff <= 2.5e-2f, "final output should match oracle within tolerance") ||
      !expect(key_cache_diff <= 1.0e-4f, "key cache should match oracle within tolerance") ||
      !expect(value_cache_diff <= 3.0e-4f, "value cache should match oracle within tolerance")) {
    std::cerr << "output_diff=" << output_diff
              << " key_cache_diff=" << key_cache_diff
              << " value_cache_diff=" << value_cache_diff
              << " layer_page_count=" << layer_pages->size()
              << " actual_key_cache_layer_size=" << actual_key_cache_layer.size()
              << " expected_key_cache_size=" << expected_key_cache.size()
              << "\n";
    for (std::size_t i = 0; i < layer_pages->size(); ++i) {
      std::cerr << "page[" << i << "]=" << (*layer_pages)[i].page_id << " ";
    }
    std::cerr << "\n";
    return false;
  }

  std::cout << "attention oracle test passed: output_diff=" << output_diff
            << " key_cache_diff=" << key_cache_diff
            << " value_cache_diff=" << value_cache_diff << "\n";
  return true;
}

}  // namespace

int main() {
  return run_attention_oracle_fixture() ? 0 : 1;
}
