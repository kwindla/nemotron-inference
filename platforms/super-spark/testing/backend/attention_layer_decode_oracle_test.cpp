#include "nemotron/attention_layer.h"
#include "nemotron/attention_layout.h"
#include "nemotron/cudnn_paged_attention.h"
#include "nemotron/paged_attention_plan.h"

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
using nemotron::CudnnPagedAttentionExecution;
using nemotron::CudnnPagedAttentionPlan;
using nemotron::DeviceTensorFp32;
using nemotron::DeviceTensorBf16;
using nemotron::GemmDescriptor;
using nemotron::GemmHeuristicCache;
using nemotron::GemmKernelFamily;
using nemotron::KernelTensorDescriptor;
using nemotron::KvCacheDataType;
using nemotron::PagedAttentionBatchPlan;
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

std::vector<__nv_bfloat16> float_to_bf16(const std::vector<float>& values) {
  std::vector<__nv_bfloat16> output(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    output[i] = __float2bfloat16(values[i]);
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

void scatter_page_major_cache_into_full_cache(
    const std::vector<float>& layer_cache,
    const std::vector<nemotron::KvPageHandle>& pages,
    std::size_t kv_head_count,
    std::size_t tokens_per_page,
    std::size_t head_dim,
    std::vector<float>* full_cache) {
  const std::size_t page_stride = kv_head_count * tokens_per_page * head_dim;
  for (std::size_t page_index = 0; page_index < pages.size(); ++page_index) {
    const std::size_t src_offset = page_index * page_stride;
    const std::size_t dst_offset = pages[page_index].page_id * page_stride;
    std::copy_n(layer_cache.begin() + src_offset, page_stride, full_cache->begin() + dst_offset);
  }
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

std::vector<float> last_row(const std::vector<float>& values, std::size_t row_width) {
  if (row_width == 0 || values.size() < row_width || values.size() % row_width != 0) {
    return {};
  }
  return std::vector<float>(values.end() - static_cast<std::ptrdiff_t>(row_width), values.end());
}

std::vector<float> prefix_only_page_major_cache(
    const std::vector<float>& full_cache,
    std::size_t prefix_token_count,
    std::size_t total_pages,
    std::size_t kv_head_count,
    std::size_t tokens_per_page,
    std::size_t head_dim) {
  std::vector<float> prefix_cache(full_cache.size(), 0.0f);
  const std::size_t token_stride = head_dim;
  for (std::size_t token = 0; token < prefix_token_count; ++token) {
    const std::size_t page_index = token / tokens_per_page;
    const std::size_t page_offset = token % tokens_per_page;
    for (std::size_t kv_head = 0; kv_head < kv_head_count; ++kv_head) {
      const std::size_t src_offset =
          (((page_index * kv_head_count) + kv_head) * tokens_per_page + page_offset) * token_stride;
      std::copy_n(
          full_cache.begin() + static_cast<std::ptrdiff_t>(src_offset),
          token_stride,
          prefix_cache.begin() + static_cast<std::ptrdiff_t>(src_offset));
    }
  }
  return prefix_cache;
}

float cache_used_token_diff(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::size_t used_token_count,
    std::size_t kv_head_count,
    std::size_t tokens_per_page,
    std::size_t head_dim) {
  const std::size_t token_stride = head_dim;
  float max_diff = 0.0f;
  for (std::size_t token = 0; token < used_token_count; ++token) {
    const std::size_t page_index = token / tokens_per_page;
    const std::size_t page_offset = token % tokens_per_page;
    for (std::size_t kv_head = 0; kv_head < kv_head_count; ++kv_head) {
      const std::size_t offset =
          (((page_index * kv_head_count) + kv_head) * tokens_per_page + page_offset) * token_stride;
      for (std::size_t dim = 0; dim < head_dim; ++dim) {
        max_diff = std::max(max_diff, std::fabs(lhs[offset + dim] - rhs[offset + dim]));
      }
    }
  }
  return max_diff;
}

float cache_padding_diff(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    std::size_t used_token_count,
    std::size_t total_pages,
    std::size_t kv_head_count,
    std::size_t tokens_per_page,
    std::size_t head_dim) {
  const std::size_t total_slots = total_pages * tokens_per_page;
  const std::size_t token_stride = head_dim;
  float max_diff = 0.0f;
  for (std::size_t token = used_token_count; token < total_slots; ++token) {
    const std::size_t page_index = token / tokens_per_page;
    const std::size_t page_offset = token % tokens_per_page;
    for (std::size_t kv_head = 0; kv_head < kv_head_count; ++kv_head) {
      const std::size_t offset =
          (((page_index * kv_head_count) + kv_head) * tokens_per_page + page_offset) * token_stride;
      for (std::size_t dim = 0; dim < head_dim; ++dim) {
        max_diff = std::max(max_diff, std::fabs(lhs[offset + dim] - rhs[offset + dim]));
      }
    }
  }
  return max_diff;
}

bool run_attention_decode_oracle_fixture() {
  const auto cublas = CublasLtHandle::Create();
  const auto cudnn = CudnnHandle::Create();
  if (!cublas || !cublas->valid() || !cudnn || !cudnn->valid() || cudnn->version() < 90500) {
    std::cout << "attention_layer_decode_oracle_test: SKIP (no CUDA device or cuDNN >= 9.5 unavailable)\n";
    return true;
  }

  const bool fixture_root_overridden = []() {
    const char* override_root = std::getenv("NEMOTRON_ATTENTION_ORACLE_FIXTURE_ROOT_OVERRIDE");
    return override_root != nullptr && std::string(override_root).size() != 0;
  }();
  const std::filesystem::path fixture_root = resolve_fixture_root(
      "NEMOTRON_ATTENTION_ORACLE_FIXTURE_ROOT_OVERRIDE",
      NEMOTRON_ATTENTION_ORACLE_FIXTURE_ROOT);
  const auto metadata = load_metadata(fixture_root);
  if (!expect(metadata.has_value(), "attention decode oracle fixture metadata should load")) {
    return false;
  }
  if (!expect(metadata->token_count > 0, "attention decode oracle fixture should contain at least one token")) {
    return false;
  }

  const std::size_t prefix_token_count = metadata->token_count - 1;
  const std::vector<float> input_hidden = read_float_file(fixture_root / "input_hidden_fp32.bin");
  const std::vector<float> norm_weight = read_float_file(fixture_root / "norm_weight_fp32.bin");
  const std::vector<float> q_weight = read_float_file(fixture_root / "q_proj_weight_fp32.bin");
  const std::vector<float> k_weight = read_float_file(fixture_root / "k_proj_weight_fp32.bin");
  const std::vector<float> v_weight = read_float_file(fixture_root / "v_proj_weight_fp32.bin");
  const std::vector<float> o_weight = read_float_file(fixture_root / "o_proj_weight_fp32.bin");
  const std::vector<float> expected_norm = read_float_file(fixture_root / "expected_norm_fp32.bin");
  const std::vector<float> expected_q = read_float_file(fixture_root / "expected_q_fp32.bin");
  const std::vector<float> expected_k = read_float_file(fixture_root / "expected_k_fp32.bin");
  const std::vector<float> expected_v = read_float_file(fixture_root / "expected_v_fp32.bin");
  const std::vector<float> expected_attention_output =
      read_float_file(fixture_root / "expected_attention_output_fp32.bin");
  const std::vector<float> expected_projected_output =
      read_float_file(fixture_root / "expected_projected_output_fp32.bin");
  const std::vector<float> expected_final_output =
      read_float_file(fixture_root / "expected_final_output_fp32.bin");
  const std::vector<float> expected_key_cache =
      read_float_file(fixture_root / "expected_key_cache_fp32.bin");
  const std::vector<float> expected_value_cache =
      read_float_file(fixture_root / "expected_value_cache_fp32.bin");

  const std::size_t hidden_size = metadata->hidden_size;
  const std::size_t kv_width = metadata->kv_head_count * metadata->head_dim;
  if (!expect(input_hidden.size() == metadata->token_count * hidden_size, "input hidden size should match metadata") ||
      !expect(expected_norm.size() == input_hidden.size(), "expected norm size should match input") ||
      !expect(expected_q.size() == input_hidden.size(), "expected q size should match input") ||
      !expect(expected_k.size() == metadata->token_count * kv_width, "expected k size should match metadata") ||
      !expect(expected_v.size() == metadata->token_count * kv_width, "expected v size should match metadata") ||
      !expect(expected_attention_output.size() == input_hidden.size(), "expected attention output size should match input") ||
      !expect(expected_projected_output.size() == input_hidden.size(), "expected projected output size should match input") ||
      !expect(expected_final_output.size() == input_hidden.size(), "expected final output size should match input")) {
    return false;
  }

  const std::vector<float> current_input = last_row(input_hidden, hidden_size);
  const std::vector<float> expected_norm_last = last_row(expected_norm, hidden_size);
  const std::vector<float> expected_q_last = last_row(expected_q, hidden_size);
  const std::vector<float> expected_k_last = last_row(expected_k, kv_width);
  const std::vector<float> expected_v_last = last_row(expected_v, kv_width);
  const std::vector<float> expected_attention_output_last = last_row(expected_attention_output, hidden_size);
  const std::vector<float> expected_projected_output_last = last_row(expected_projected_output, hidden_size);
  const std::vector<float> expected_final_output_last = last_row(expected_final_output, hidden_size);
  const std::vector<float> expected_prefix_key_cache = prefix_only_page_major_cache(
      expected_key_cache,
      prefix_token_count,
      metadata->total_pages,
      metadata->kv_head_count,
      metadata->tokens_per_page,
      metadata->head_dim);
  const std::vector<float> expected_prefix_value_cache = prefix_only_page_major_cache(
      expected_value_cache,
      prefix_token_count,
      metadata->total_pages,
      metadata->kv_head_count,
      metadata->tokens_per_page,
      metadata->head_dim);

  AttentionLayerBindings bindings;
  const auto norm_descriptor = make_norm_descriptor(norm_weight);
  const auto q_descriptor = make_dense_descriptor(
      "backbone.layers.7.mixer.q_proj.weight",
      q_weight,
      hidden_size,
      hidden_size);
  const auto k_descriptor = make_dense_descriptor(
      "backbone.layers.7.mixer.k_proj.weight",
      k_weight,
      kv_width,
      hidden_size);
  const auto v_descriptor = make_dense_descriptor(
      "backbone.layers.7.mixer.v_proj.weight",
      v_weight,
      kv_width,
      hidden_size);
  const auto o_descriptor = make_dense_descriptor(
      "backbone.layers.7.mixer.o_proj.weight",
      o_weight,
      hidden_size,
      hidden_size);
  bindings.norm_weight = &norm_descriptor;
  bindings.q_proj = &q_descriptor;
  bindings.k_proj = &k_descriptor;
  bindings.v_proj = &v_descriptor;
  bindings.o_proj = &o_descriptor;

  AttentionLayerConfig layer_config;
  layer_config.layer_index = metadata->layer_index;
  layer_config.hidden_size = hidden_size;
  layer_config.query_head_count = metadata->query_head_count;
  layer_config.kv_head_count = metadata->kv_head_count;
  layer_config.head_dim = metadata->head_dim;
  layer_config.rms_epsilon = metadata->rms_epsilon;

  auto slice = AttentionLayerSlice::Create(layer_config, bindings);
  if (!expect(slice != nullptr && slice->valid(), "attention decode slice should build")) {
    return false;
  }

  RequestExecutionConfig request_config;
  request_config.hidden_size = hidden_size;
  request_config.max_tokens = metadata->token_count;
  request_config.scratch_tokens = 1;
  request_config.attention_kv_cache = AttentionKvCacheConfig{
      metadata->layer_index + 1,
      metadata->kv_head_count,
      metadata->head_dim,
      metadata->tokens_per_page,
      KvCacheDataType::kBf16,
  };
  request_config.attention_total_pages = metadata->total_pages * (metadata->layer_index + 1);
  request_config.attention_query_head_count = metadata->query_head_count;
  request_config.attention_head_dim = metadata->head_dim;

  auto request = RequestExecutionContext::Create(request_config);
  auto input = DeviceTensorFp32::Create({1, hidden_size});
  auto output = DeviceTensorFp32::Create({1, hidden_size});
  if (!expect(request != nullptr && request->valid(), "request context should create for attention decode oracle") ||
      !expect(input != nullptr && input->valid(), "input tensor should create for attention decode oracle") ||
      !expect(output != nullptr && output->valid(), "output tensor should create for attention decode oracle")) {
    return false;
  }
  if (!expect(input->CopyFromHost(current_input.data(), current_input.size()), "current input should upload")) {
    return false;
  }
  if (!expect(request->SetSequenceLength(prefix_token_count), "request sequence length should seed the decode prefix")) {
    return false;
  }

  const auto* layer_pages = request->kv_pages(metadata->layer_index);
  if (!expect(layer_pages != nullptr, "layer page handles should be available after prefix setup")) {
    return false;
  }
  if (layer_pages->size() != metadata->total_pages) {
    if (!fixture_root_overridden) {
      std::cout
          << "attention_layer_decode_oracle_test: SKIP (default fixture is not decode-shaped; "
          << "set NEMOTRON_ATTENTION_ORACLE_FIXTURE_ROOT_OVERRIDE to a teacher-forced decode fixture)\n";
      return true;
    }
    if (!expect(false, "layer page count should match metadata")) {
      return false;
    }
  }
  if (!expect(layer_pages->size() == metadata->total_pages, "layer page count should match metadata")) {
    return false;
  }

  std::vector<float> key_cache_full(request->key_cache()->numel(), 0.0f);
  std::vector<float> value_cache_full(request->value_cache()->numel(), 0.0f);
  scatter_page_major_cache_into_full_cache(
      expected_prefix_key_cache,
      *layer_pages,
      metadata->kv_head_count,
      metadata->tokens_per_page,
      metadata->head_dim,
      &key_cache_full);
  scatter_page_major_cache_into_full_cache(
      expected_prefix_value_cache,
      *layer_pages,
      metadata->kv_head_count,
      metadata->tokens_per_page,
      metadata->head_dim,
      &value_cache_full);

  const std::vector<__nv_bfloat16> key_cache_prefix_bf16 = float_to_bf16(key_cache_full);
  const std::vector<__nv_bfloat16> value_cache_prefix_bf16 = float_to_bf16(value_cache_full);
  if (!expect(
          request->key_cache()->CopyFromHost(key_cache_prefix_bf16.data(), key_cache_prefix_bf16.size()),
          "prefix key cache should upload") ||
      !expect(
          request->value_cache()->CopyFromHost(value_cache_prefix_bf16.data(), value_cache_prefix_bf16.size()),
          "prefix value cache should upload")) {
    return false;
  }
  if (!expect(request->AdvanceDecodePosition(1), "decode position should advance for the current token")) {
    return false;
  }

  GemmHeuristicCache heuristic_cache;
  if (!expect(
          slice->Run(*cublas, *cudnn, &heuristic_cache, *request, *input, output.get()),
          "attention decode oracle slice should execute")) {
    return false;
  }

  std::vector<float> actual_output(expected_final_output_last.size(), 0.0f);
  std::vector<float> actual_norm(expected_norm_last.size(), 0.0f);
  std::vector<float> actual_q(expected_q_last.size(), 0.0f);
  std::vector<float> actual_k(expected_k_last.size(), 0.0f);
  std::vector<float> actual_v(expected_v_last.size(), 0.0f);
  std::vector<float> actual_attention_output(expected_attention_output_last.size(), 0.0f);
  std::vector<float> actual_projected_output(expected_projected_output_last.size(), 0.0f);
  if (!expect(output->CopyToHost(actual_output.data(), actual_output.size()), "final output should download") ||
      !expect(request->attention_normed_decode()->CopyToHost(actual_norm.data(), actual_norm.size()), "norm should download") ||
      !expect(request->attention_q_decode()->CopyToHost(actual_q.data(), actual_q.size()), "q should download") ||
      !expect(request->attention_k_decode()->CopyToHost(actual_k.data(), actual_k.size()), "k should download") ||
      !expect(request->attention_v_decode()->CopyToHost(actual_v.data(), actual_v.size()), "v should download") ||
      !expect(
          request->attention_output_fp32_decode()->CopyToHost(
              actual_attention_output.data(),
              actual_attention_output.size()),
          "attention output should download") ||
      !expect(
          request->attention_projected_decode()->CopyToHost(
              actual_projected_output.data(),
              actual_projected_output.size()),
          "projected output should download")) {
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

  const float norm_diff = max_abs_diff(actual_norm, expected_norm_last);
  const float q_diff = max_abs_diff(actual_q, expected_q_last);
  const float k_diff = max_abs_diff(actual_k, expected_k_last);
  const float v_diff = max_abs_diff(actual_v, expected_v_last);
  const float attention_output_diff =
      max_abs_diff(actual_attention_output, expected_attention_output_last);
  const float projected_output_diff =
      max_abs_diff(actual_projected_output, expected_projected_output_last);
  const float final_output_diff = max_abs_diff(actual_output, expected_final_output_last);
  const float key_cache_diff = max_abs_diff(actual_key_cache_layer, expected_key_cache);
  const float value_cache_diff = max_abs_diff(actual_value_cache_layer, expected_value_cache);
  const float key_cache_used_diff = cache_used_token_diff(
      actual_key_cache_layer,
      expected_key_cache,
      metadata->token_count,
      metadata->kv_head_count,
      metadata->tokens_per_page,
      metadata->head_dim);
  const float value_cache_used_diff = cache_used_token_diff(
      actual_value_cache_layer,
      expected_value_cache,
      metadata->token_count,
      metadata->kv_head_count,
      metadata->tokens_per_page,
      metadata->head_dim);
  const float key_cache_padding_diff = cache_padding_diff(
      actual_key_cache_layer,
      expected_key_cache,
      metadata->token_count,
      metadata->total_pages,
      metadata->kv_head_count,
      metadata->tokens_per_page,
      metadata->head_dim);
  const float value_cache_padding_diff = cache_padding_diff(
      actual_value_cache_layer,
      expected_value_cache,
      metadata->token_count,
      metadata->total_pages,
      metadata->kv_head_count,
      metadata->tokens_per_page,
      metadata->head_dim);

  float generic_attention_output_diff = INFINITY;
  {
    std::vector<nemotron::AttentionSequencePages> sequences = {
        nemotron::AttentionSequencePages{
            metadata->token_count,
            [&]() {
              std::vector<std::size_t> page_ids;
              page_ids.reserve(layer_pages->size());
              for (const nemotron::KvPageHandle& handle : *layer_pages) {
                page_ids.push_back(handle.page_id);
              }
              return page_ids;
            }(),
        },
    };
    auto batch_plan = nemotron::BuildPagedAttentionBatchPlan(
        request->config().attention_kv_cache,
        metadata->layer_index,
        sequences,
        nullptr);
    if (!expect(batch_plan.has_value(), "generic paged-attention plan should build")) {
      return false;
    }
    auto attention_config = nemotron::BuildCudnnPagedAttentionConfig(
        *batch_plan,
        metadata->query_head_count,
        1,
        request->config().attention_total_pages,
        0.0f,
        true,
        false);
    if (!expect(attention_config.has_value(), "generic cuDNN paged-attention config should build")) {
      return false;
    }
    auto attention_plan = CudnnPagedAttentionPlan::Create(*cudnn, *attention_config);
    if (!expect(attention_plan != nullptr && attention_plan->valid(), "generic cuDNN paged-attention plan should create")) {
      return false;
    }

    auto generic_query_bf16 = DeviceTensorBf16::Create(
        {1, metadata->query_head_count, 1, metadata->head_dim});
    auto generic_output_bf16 = DeviceTensorBf16::Create(
        {1, metadata->query_head_count, 1, metadata->head_dim});
    auto generic_attention_output = DeviceTensorFp32::Create({1, hidden_size});
    if (!expect(
            generic_query_bf16 != nullptr && generic_query_bf16->valid() &&
                generic_output_bf16 != nullptr && generic_output_bf16->valid() &&
                generic_attention_output != nullptr && generic_attention_output->valid(),
            "generic attention scratch should allocate")) {
      return false;
    }
    if (!expect(
            nemotron::ConvertRowMajorMatrixToAttentionBf16(
                *request->attention_q_decode(),
                1,
                metadata->query_head_count,
                metadata->head_dim,
                generic_query_bf16.get(),
                nullptr),
            "generic query layout conversion should succeed") ||
        !expect(generic_output_bf16->FillZero(), "generic output tensor should clear")) {
      return false;
    }

    if (!expect(
            request->EnsureAttentionAuxCapacity(
                batch_plan->sequence_lengths.size(),
                batch_plan->page_table.size()),
            "generic attention aux buffers should allocate")) {
      return false;
    }
    auto* seq_len_q = request->attention_seq_len_q_device();
    auto* seq_len_kv = request->attention_seq_len_kv_device();
    auto* page_table = request->attention_page_table_device();
    const std::vector<std::int32_t> seq_len_q_host(batch_plan->sequence_lengths.size(), 1);
    if (!expect(
            seq_len_q != nullptr && seq_len_kv != nullptr && page_table != nullptr,
            "generic attention aux buffers should be present") ||
        !expect(seq_len_q->CopyFromHost(seq_len_q_host), "generic seq_len_q should upload") ||
        !expect(seq_len_kv->CopyFromHost(batch_plan->sequence_lengths), "generic seq_len_kv should upload") ||
        !expect(page_table->CopyFromHost(batch_plan->page_table), "generic page table should upload")) {
      return false;
    }

    const CudnnPagedAttentionExecution generic_execution{
        generic_query_bf16->data(),
        request->key_cache_data(),
        request->value_cache_data(),
        seq_len_q->data(),
        seq_len_kv->data(),
        page_table->data(),
        page_table->data(),
        generic_output_bf16->data(),
        nullptr,
    };
    if (!expect(
            attention_plan->Execute(*cudnn, generic_execution, nullptr, true),
            "generic paged-attention execution should succeed") ||
        !expect(
            nemotron::ConvertAttentionBf16ToRowMajorMatrix(
                *generic_output_bf16,
                1,
                metadata->query_head_count,
                metadata->head_dim,
                generic_attention_output.get(),
                nullptr),
            "generic output conversion should succeed")) {
      return false;
    }
    std::vector<float> generic_attention_output_host(expected_attention_output_last.size(), 0.0f);
    if (!expect(
            generic_attention_output->CopyToHost(
                generic_attention_output_host.data(),
                generic_attention_output_host.size()),
            "generic attention output should download")) {
      return false;
    }
    generic_attention_output_diff =
        max_abs_diff(generic_attention_output_host, expected_attention_output_last);
  }

  std::cout << "attention decode oracle:"
            << " prefix_token_count=" << prefix_token_count
            << " norm_diff=" << norm_diff
            << " q_diff=" << q_diff
            << " k_diff=" << k_diff
            << " v_diff=" << v_diff
            << " attention_output_diff=" << attention_output_diff
            << " projected_output_diff=" << projected_output_diff
            << " final_output_diff=" << final_output_diff
            << " key_cache_diff=" << key_cache_diff
            << " value_cache_diff=" << value_cache_diff
            << " key_cache_used_diff=" << key_cache_used_diff
            << " value_cache_used_diff=" << value_cache_used_diff
            << " key_cache_padding_diff=" << key_cache_padding_diff
            << " value_cache_padding_diff=" << value_cache_padding_diff
            << " generic_attention_output_diff=" << generic_attention_output_diff
            << "\n";

  constexpr float kDecodeAttentionOutputTolerance = 5.0e-3f;
  const bool surfaces_ok =
      expect(norm_diff <= 1.0e-5f, "decode norm should match oracle within tolerance") &&
      expect(q_diff <= 1.0e-5f, "decode q should match oracle within tolerance") &&
      expect(k_diff <= 1.0e-5f, "decode k should match oracle within tolerance") &&
      expect(v_diff <= 1.0e-5f, "decode v should match oracle within tolerance") &&
      expect(
          attention_output_diff <= kDecodeAttentionOutputTolerance,
          "decode attention output should match oracle within tolerance") &&
      expect(
          generic_attention_output_diff <= kDecodeAttentionOutputTolerance,
          "generic paged attention output should match oracle within tolerance") &&
      expect(
          projected_output_diff <= 1.0e-3f,
          "decode projected output should match oracle within tolerance") &&
      expect(final_output_diff <= 1.0e-3f, "decode final output should match oracle within tolerance");
  const bool cache_ok =
      expect(key_cache_used_diff <= 1.0e-4f, "decode key cache used-token values should match oracle") &&
      expect(value_cache_used_diff <= 3.0e-4f, "decode value cache used-token values should match oracle");
  return surfaces_ok && cache_ok;
}

}  // namespace

int main() {
  return run_attention_decode_oracle_fixture() ? 0 : 1;
}
