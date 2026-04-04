#include "nemotron/attention_layer.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using nemotron::AttentionBackend;
using nemotron::AttentionBackendName;
using nemotron::AttentionBackendPolicy;
using nemotron::AttentionBackendSelectorConfig;
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

class ScopedEnvVar {
 public:
  ScopedEnvVar(const char* name, const char* value) : name_(name) {
    const char* existing = std::getenv(name_);
    if (existing != nullptr) {
      had_old_value_ = true;
      old_value_ = existing;
    }
    if (value != nullptr) {
      setenv(name_, value, 1);
    } else {
      unsetenv(name_);
    }
  }

  ~ScopedEnvVar() {
    if (had_old_value_) {
      setenv(name_, old_value_.c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }

  ScopedEnvVar(const ScopedEnvVar&) = delete;
  ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

 private:
  const char* name_ = nullptr;
  bool had_old_value_ = false;
  std::string old_value_;
};

template <typename T>
class AlignedHostArray {
 public:
  explicit AlignedHostArray(std::size_t count) : count_(count) {
    void* raw = nullptr;
    if (posix_memalign(&raw, 16, count * sizeof(T)) == 0) {
      data_ = reinterpret_cast<T*>(raw);
    }
  }

  ~AlignedHostArray() {
    std::free(data_);
  }

  AlignedHostArray(const AlignedHostArray&) = delete;
  AlignedHostArray& operator=(const AlignedHostArray&) = delete;

  T* data() const { return data_; }
  std::size_t count() const { return count_; }
  bool valid() const { return data_ != nullptr; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

std::vector<float> cpu_matmul_row_major(
    const std::vector<float>& activations,
    const std::vector<float>& weights,
    std::size_t rows,
    std::size_t k,
    std::size_t n) {
  std::vector<float> output(rows * n, 0.0f);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (std::size_t kk = 0; kk < k; ++kk) {
        sum += activations[row * k + kk] * weights[col * k + kk];
      }
      output[row * n + col] = sum;
    }
  }
  return output;
}

std::vector<float> cpu_rms_norm(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    std::size_t rows,
    std::size_t hidden_size,
    float epsilon) {
  std::vector<float> output(input.size(), 0.0f);
  for (std::size_t row = 0; row < rows; ++row) {
    float sumsq = 0.0f;
    for (std::size_t col = 0; col < hidden_size; ++col) {
      const float value = input[row * hidden_size + col];
      sumsq += value * value;
    }
    const float inv_rms = 1.0f / std::sqrt((sumsq / static_cast<float>(hidden_size)) + epsilon);
    for (std::size_t col = 0; col < hidden_size; ++col) {
      output[row * hidden_size + col] = input[row * hidden_size + col] * inv_rms * weight[col];
    }
  }
  return output;
}

std::vector<float> cpu_attention(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    std::size_t token_count,
    std::size_t query_head_count,
    std::size_t kv_head_count,
    std::size_t head_dim) {
  std::vector<float> output(token_count * query_head_count * head_dim, 0.0f);
  const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  if (kv_head_count == 0 || (query_head_count % kv_head_count) != 0) {
    return {};
  }
  const std::size_t queries_per_kv_head = query_head_count / kv_head_count;
  for (std::size_t token = 0; token < token_count; ++token) {
    for (std::size_t head = 0; head < query_head_count; ++head) {
      const std::size_t kv_head = head / queries_per_kv_head;
      std::vector<float> scores(token + 1, 0.0f);
      float max_score = -1e30f;
      for (std::size_t prev = 0; prev <= token; ++prev) {
        float score = 0.0f;
        for (std::size_t dim = 0; dim < head_dim; ++dim) {
          score += q[token * (query_head_count * head_dim) + head * head_dim + dim] *
                   k[prev * (kv_head_count * head_dim) + kv_head * head_dim + dim];
        }
        score *= attn_scale;
        scores[prev] = score;
        max_score = std::max(max_score, score);
      }
      float sum = 0.0f;
      for (float& score : scores) {
        score = std::exp(score - max_score);
        sum += score;
      }
      for (std::size_t prev = 0; prev <= token; ++prev) {
        const float weight = scores[prev] / sum;
        for (std::size_t dim = 0; dim < head_dim; ++dim) {
          output[token * (query_head_count * head_dim) + head * head_dim + dim] +=
              weight * v[prev * (kv_head_count * head_dim) + kv_head * head_dim + dim];
        }
      }
    }
  }
  return output;
}

bool nearly_equal(const std::vector<float>& lhs, const std::vector<float>& rhs, float tol) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  float max_abs_diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_abs_diff = std::max(max_abs_diff, std::fabs(lhs[i] - rhs[i]));
  }
  if (max_abs_diff > tol) {
    std::cerr << "max_abs_diff=" << max_abs_diff << " exceeds tol=" << tol << "\n";
    return false;
  }
  return true;
}

KernelTensorDescriptor make_norm_descriptor(const std::vector<float>& weights) {
  KernelTensorDescriptor descriptor;
  descriptor.tensor_name = "layers.0.input_norm.weight";
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

GemmDescriptor make_dense_descriptor(
    const std::string& name,
    const AlignedHostArray<float>& weights,
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
  descriptor.packed_nbytes = weights.count() * sizeof(float);
  return descriptor;
}

AttentionBackendSelectorConfig make_selector_config(
    std::size_t query_head_count,
    std::size_t kv_head_count,
    std::size_t head_dim,
    std::size_t tokens_per_page,
    std::size_t max_query_tokens,
    std::size_t max_kv_tokens,
    bool cudnn_available,
    long long cudnn_version) {
  AttentionBackendSelectorConfig config;
  config.cache_config = AttentionKvCacheConfig{
      1,
      kv_head_count,
      head_dim,
      tokens_per_page,
      KvCacheDataType::kBf16,
  };
  config.batch_size = 1;
  config.query_head_count = query_head_count;
  config.max_query_tokens = max_query_tokens;
  config.max_kv_tokens = max_kv_tokens;
  config.container_page_count =
      nemotron::RequiredPagesForTokens(config.cache_config, max_kv_tokens);
  config.page_table_entries =
      nemotron::RequiredPagesForTokens(config.cache_config, max_kv_tokens);
  config.cudnn_available = cudnn_available;
  config.cudnn_version = cudnn_version;
  config.causal = true;
  config.generate_stats = false;
  return config;
}

bool test_attention_backend_policy_ignores_legacy_env_overrides() {
  const ScopedEnvVar production_env("NEMOTRON_FORWARD_ATTENTION_PRODUCTION", "0");
  const ScopedEnvVar scalar_fallback_env("NEMOTRON_FORWARD_ATTENTION_SCALAR_FALLBACK", "1");

  AttentionBackendPolicy policy;
  const AttentionBackendSelectorConfig cudnn_config =
      make_selector_config(/*query_head_count=*/4, /*kv_head_count=*/2, /*head_dim=*/64, /*tokens_per_page=*/16,
                           /*max_query_tokens=*/2, /*max_kv_tokens=*/31, /*cudnn_available=*/true, /*cudnn_version=*/90500);
  const AttentionBackendSelectorConfig fallback_config =
      make_selector_config(/*query_head_count=*/4, /*kv_head_count=*/2, /*head_dim=*/64, /*tokens_per_page=*/16,
                           /*max_query_tokens=*/2, /*max_kv_tokens=*/31, /*cudnn_available=*/false, /*cudnn_version=*/0);

  const AttentionBackend selected_cudnn = policy.Select(cudnn_config, /*token_count=*/2);
  const AttentionBackend selected_fallback = policy.Select(fallback_config, /*token_count=*/2);
  return expect(selected_cudnn == AttentionBackend::kCudnnPaged,
                "legacy env vars should not block validated cuDNN selection") &&
         expect(selected_fallback == AttentionBackend::kDeviceFallback,
                "legacy env vars should not force scalar fallback over the deterministic device fallback path");
}

bool test_attention_backend_policy_supports_exact_nano_decode_shape() {
  AttentionBackendPolicy policy;
  const AttentionBackendSelectorConfig nano_config =
      make_selector_config(/*query_head_count=*/32, /*kv_head_count=*/2, /*head_dim=*/128, /*tokens_per_page=*/16,
                           /*max_query_tokens=*/1, /*max_kv_tokens=*/16, /*cudnn_available=*/false, /*cudnn_version=*/0);
  const AttentionBackendSelectorConfig generic_config =
      make_selector_config(/*query_head_count=*/8, /*kv_head_count=*/2, /*head_dim=*/128, /*tokens_per_page=*/16,
                           /*max_query_tokens=*/1, /*max_kv_tokens=*/16, /*cudnn_available=*/false, /*cudnn_version=*/0);

  return expect(
             policy.Supports(AttentionBackend::kNanoDecode, nano_config, /*token_count=*/1, /*device_sm=*/120),
             "Nano decode backend should support the measured exact Nano decode shape") &&
         expect(
             !policy.Supports(AttentionBackend::kNanoDecode, generic_config, /*token_count=*/1, /*device_sm=*/120),
             "Nano decode backend should reject non-Nano shapes") &&
         expect(
             !policy.Supports(AttentionBackend::kNanoDecode, nano_config, /*token_count=*/2, /*device_sm=*/120),
             "Nano decode backend should reject multi-token requests") &&
         expect(
             !policy.Supports(AttentionBackend::kNanoDecode, nano_config, /*token_count=*/1, /*device_sm=*/90),
             "Nano decode backend should stay fenced behind the required device capability");
}

bool test_attention_layer_slice_matches_cpu_reference() {
  const auto cublas = CublasLtHandle::Create();
  const auto cudnn = CudnnHandle::Create();
  if (!cublas || !cublas->valid() || !cudnn || !cudnn->valid() || cudnn->version() < 90500) {
    std::cout << "attention_layer_test: SKIP (no CUDA device or cuDNN >= 9.5 unavailable)\n";
    return true;
  }

  constexpr std::size_t kHidden = 128;
  constexpr std::size_t kQueryHeads = 2;
  constexpr std::size_t kKvHeads = 2;
  constexpr std::size_t kHeadDim = 64;
  constexpr std::size_t kTokens = 3;
  constexpr float kEpsilon = 1e-5f;

  std::vector<float> norm_weight_host(kHidden, 1.0f);
  AlignedHostArray<float> q_weight(kHidden * kHidden);
  AlignedHostArray<float> k_weight(kHidden * kHidden);
  AlignedHostArray<float> v_weight(kHidden * kHidden);
  AlignedHostArray<float> o_weight(kHidden * kHidden);
  if (!q_weight.valid() || !k_weight.valid() || !v_weight.valid() || !o_weight.valid()) {
    std::cout << "attention_layer_test: SKIP (aligned host allocation failed)\n";
    return true;
  }
  std::fill(q_weight.data(), q_weight.data() + q_weight.count(), 0.0f);
  std::fill(k_weight.data(), k_weight.data() + k_weight.count(), 0.0f);
  std::fill(v_weight.data(), v_weight.data() + v_weight.count(), 0.0f);
  std::fill(o_weight.data(), o_weight.data() + o_weight.count(), 0.0f);
  for (std::size_t i = 0; i < kHidden; ++i) {
    q_weight.data()[i * kHidden + i] = 1.0f;
    k_weight.data()[i * kHidden + i] = 1.0f;
    v_weight.data()[i * kHidden + i] = 1.0f;
    o_weight.data()[i * kHidden + i] = 1.0f;
  }

  AttentionLayerBindings bindings;
  const auto norm_descriptor = make_norm_descriptor(norm_weight_host);
  const auto q_descriptor = make_dense_descriptor("layers.0.self_attn.q_proj.weight", q_weight, kHidden, kHidden);
  const auto k_descriptor = make_dense_descriptor("layers.0.self_attn.k_proj.weight", k_weight, kHidden, kHidden);
  const auto v_descriptor = make_dense_descriptor("layers.0.self_attn.v_proj.weight", v_weight, kHidden, kHidden);
  const auto o_descriptor = make_dense_descriptor("layers.0.self_attn.o_proj.weight", o_weight, kHidden, kHidden);
  bindings.norm_weight = &norm_descriptor;
  bindings.q_proj = &q_descriptor;
  bindings.k_proj = &k_descriptor;
  bindings.v_proj = &v_descriptor;
  bindings.o_proj = &o_descriptor;

  AttentionLayerConfig layer_config;
  layer_config.layer_index = 0;
  layer_config.hidden_size = kHidden;
  layer_config.query_head_count = kQueryHeads;
  layer_config.kv_head_count = kKvHeads;
  layer_config.head_dim = kHeadDim;
  layer_config.rms_epsilon = kEpsilon;

  auto slice = AttentionLayerSlice::Create(layer_config, bindings);
  if (!expect(slice != nullptr && slice->valid(), "attention slice should build for valid synthetic weights")) {
    return false;
  }

  RequestExecutionConfig request_config;
  request_config.hidden_size = kHidden;
  request_config.max_tokens = 4;
  request_config.scratch_tokens = 4;
  request_config.attention_kv_cache = AttentionKvCacheConfig{
      1,
      kKvHeads,
      kHeadDim,
      16,
      KvCacheDataType::kBf16,
  };
  request_config.attention_total_pages = 4;
  auto request_context = RequestExecutionContext::Create(request_config);
  if (!expect(request_context != nullptr && request_context->valid(), "request context should build")) {
    return false;
  }

  auto input = DeviceTensorFp32::Create({kTokens, kHidden});
  auto output = DeviceTensorFp32::Create({kTokens, kHidden});
  if (!input || !output) {
    std::cout << "attention_layer_test: SKIP (no CUDA device available)\n";
    return true;
  }

  std::vector<float> input_host(kTokens * kHidden, 0.0f);
  for (std::size_t token = 0; token < kTokens; ++token) {
    for (std::size_t dim = 0; dim < kHidden; ++dim) {
      const float base = static_cast<float>((token + 1) * ((dim % 17) - 8));
      input_host[token * kHidden + dim] = base / 7.0f;
    }
  }
  if (!expect(input->CopyFromHost(input_host.data(), input_host.size()), "input upload should succeed")) {
    return false;
  }
  if (!expect(
          request_context->EnsureAttentionTokens(kTokens),
          "request context should reserve KV pages for the prefill token window")) {
    return false;
  }

  GemmHeuristicCache heuristic_cache;
  if (!expect(
          slice->Run(
              *cublas,
              *cudnn,
              &heuristic_cache,
              *request_context,
              0,
              kTokens,
              *input,
              output.get()),
              "attention slice should execute successfully")) {
    return false;
  }

  const std::vector<float> normed = cpu_rms_norm(input_host, norm_weight_host, kTokens, kHidden, kEpsilon);
  const std::vector<float> q_cpu = cpu_matmul_row_major(normed, std::vector<float>(q_weight.data(), q_weight.data() + q_weight.count()), kTokens, kHidden, kHidden);
  const std::vector<float> k_cpu = cpu_matmul_row_major(normed, std::vector<float>(k_weight.data(), k_weight.data() + k_weight.count()), kTokens, kHidden, kHidden);
  const std::vector<float> v_cpu = cpu_matmul_row_major(normed, std::vector<float>(v_weight.data(), v_weight.data() + v_weight.count()), kTokens, kHidden, kHidden);
  const std::vector<float> attention_cpu = cpu_attention(q_cpu, k_cpu, v_cpu, kTokens, kQueryHeads, kKvHeads, kHeadDim);
  const std::vector<float> projected_cpu = cpu_matmul_row_major(attention_cpu, std::vector<float>(o_weight.data(), o_weight.data() + o_weight.count()), kTokens, kHidden, kHidden);
  std::vector<float> expected(projected_cpu.size(), 0.0f);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    expected[i] = input_host[i] + projected_cpu[i];
  }

  std::vector<float> actual(expected.size(), 0.0f);
  if (!expect(output->CopyToHost(actual.data(), actual.size()), "output download should succeed")) {
    return false;
  }
  return expect(nearly_equal(actual, expected, 2e-2f), "attention slice should match CPU reference");
}

bool test_attention_layer_slice_continues_from_existing_prefix() {
  const auto cublas = CublasLtHandle::Create();
  const auto cudnn = CudnnHandle::Create();
  if (!cublas || !cublas->valid() || !cudnn) {
    std::cout << "attention_layer_test: SKIP (no CUDA device available)\n";
    return true;
  }

  constexpr std::size_t kHidden = 128;
  constexpr std::size_t kQueryHeads = 2;
  constexpr std::size_t kKvHeads = 2;
  constexpr std::size_t kHeadDim = 64;
  constexpr std::size_t kPrefixTokens = 2;
  constexpr std::size_t kDecodeTokens = 1;
  constexpr std::size_t kTotalTokens = kPrefixTokens + kDecodeTokens;
  constexpr float kEpsilon = 1e-5f;

  std::vector<float> norm_weight_host(kHidden, 1.0f);
  AlignedHostArray<float> q_weight(kHidden * kHidden);
  AlignedHostArray<float> k_weight(kHidden * kHidden);
  AlignedHostArray<float> v_weight(kHidden * kHidden);
  AlignedHostArray<float> o_weight(kHidden * kHidden);
  if (!q_weight.valid() || !k_weight.valid() || !v_weight.valid() || !o_weight.valid()) {
    std::cout << "attention_layer_test: SKIP (aligned host allocation failed)\n";
    return true;
  }
  std::fill(q_weight.data(), q_weight.data() + q_weight.count(), 0.0f);
  std::fill(k_weight.data(), k_weight.data() + k_weight.count(), 0.0f);
  std::fill(v_weight.data(), v_weight.data() + v_weight.count(), 0.0f);
  std::fill(o_weight.data(), o_weight.data() + o_weight.count(), 0.0f);
  for (std::size_t i = 0; i < kHidden; ++i) {
    q_weight.data()[i * kHidden + i] = 1.0f;
    k_weight.data()[i * kHidden + i] = 1.0f;
    v_weight.data()[i * kHidden + i] = 1.0f;
    o_weight.data()[i * kHidden + i] = 1.0f;
  }

  AttentionLayerBindings bindings;
  const auto norm_descriptor = make_norm_descriptor(norm_weight_host);
  const auto q_descriptor = make_dense_descriptor("layers.0.self_attn.q_proj.weight", q_weight, kHidden, kHidden);
  const auto k_descriptor = make_dense_descriptor("layers.0.self_attn.k_proj.weight", k_weight, kHidden, kHidden);
  const auto v_descriptor = make_dense_descriptor("layers.0.self_attn.v_proj.weight", v_weight, kHidden, kHidden);
  const auto o_descriptor = make_dense_descriptor("layers.0.self_attn.o_proj.weight", o_weight, kHidden, kHidden);
  bindings.norm_weight = &norm_descriptor;
  bindings.q_proj = &q_descriptor;
  bindings.k_proj = &k_descriptor;
  bindings.v_proj = &v_descriptor;
  bindings.o_proj = &o_descriptor;

  AttentionLayerConfig layer_config;
  layer_config.layer_index = 0;
  layer_config.hidden_size = kHidden;
  layer_config.query_head_count = kQueryHeads;
  layer_config.kv_head_count = kKvHeads;
  layer_config.head_dim = kHeadDim;
  layer_config.rms_epsilon = kEpsilon;

  auto slice = AttentionLayerSlice::Create(layer_config, bindings);
  if (!expect(slice != nullptr && slice->valid(), "attention slice should build for continuation test")) {
    return false;
  }

  RequestExecutionConfig request_config;
  request_config.hidden_size = kHidden;
  request_config.max_tokens = 4;
  request_config.scratch_tokens = 4;
  request_config.attention_kv_cache = AttentionKvCacheConfig{
      1,
      kKvHeads,
      kHeadDim,
      16,
      KvCacheDataType::kBf16,
  };
  request_config.attention_total_pages = 4;
  auto request_context = RequestExecutionContext::Create(request_config);
  if (!expect(request_context != nullptr && request_context->valid(), "continuation request context should build")) {
    return false;
  }

  std::vector<float> input_host(kTotalTokens * kHidden, 0.0f);
  for (std::size_t token = 0; token < kTotalTokens; ++token) {
    for (std::size_t dim = 0; dim < kHidden; ++dim) {
      const float base = static_cast<float>((token + 2) * ((dim % 19) - 9));
      input_host[token * kHidden + dim] = base / 11.0f;
    }
  }
  const std::vector<float> prefix_host(
      input_host.begin(),
      input_host.begin() + static_cast<std::ptrdiff_t>(kPrefixTokens * kHidden));
  const std::vector<float> decode_host(
      input_host.begin() + static_cast<std::ptrdiff_t>(kPrefixTokens * kHidden),
      input_host.end());

  auto prefix_input = DeviceTensorFp32::Create({kPrefixTokens, kHidden});
  auto prefix_output = DeviceTensorFp32::Create({kPrefixTokens, kHidden});
  auto decode_input = DeviceTensorFp32::Create({kDecodeTokens, kHidden});
  auto decode_output = DeviceTensorFp32::Create({kDecodeTokens, kHidden});
  if (!prefix_input || !prefix_output || !decode_input || !decode_output) {
    std::cout << "attention_layer_test: SKIP (no CUDA device available)\n";
    return true;
  }
  if (!expect(prefix_input->CopyFromHost(prefix_host.data(), prefix_host.size()), "prefix input upload should succeed") ||
      !expect(decode_input->CopyFromHost(decode_host.data(), decode_host.size()), "decode input upload should succeed")) {
    return false;
  }

  GemmHeuristicCache heuristic_cache;
  if (!expect(
          request_context->EnsureAttentionTokens(kPrefixTokens),
          "request context should reserve KV pages for the prefix")) {
    return false;
  }
  if (!expect(
          slice->Run(
              *cublas,
              *cudnn,
              &heuristic_cache,
              *request_context,
              0,
              kPrefixTokens,
              *prefix_input,
              prefix_output.get()),
          "prefix attention pass should succeed")) {
    return false;
  }
  if (!expect(
          request_context->SetSequenceLength(kPrefixTokens),
          "prefix pass should commit sequence length")) {
    return false;
  }
  if (!expect(
          request_context->EnsureAttentionTokens(kTotalTokens),
          "request context should reserve KV pages for the appended decode token")) {
    return false;
  }
  if (!expect(
          slice->Run(
              *cublas,
              *cudnn,
              &heuristic_cache,
              *request_context,
              kPrefixTokens,
              kTotalTokens,
              *decode_input,
              decode_output.get()),
          "decode continuation pass should succeed")) {
    return false;
  }
  if (!expect(
          request_context->AdvanceDecodePosition(kDecodeTokens),
          "decode continuation should advance request positions")) {
    return false;
  }

  const std::vector<float> normed = cpu_rms_norm(input_host, norm_weight_host, kTotalTokens, kHidden, kEpsilon);
  const std::vector<float> q_cpu = cpu_matmul_row_major(
      normed,
      std::vector<float>(q_weight.data(), q_weight.data() + q_weight.count()),
      kTotalTokens,
      kHidden,
      kHidden);
  const std::vector<float> k_cpu = cpu_matmul_row_major(
      normed,
      std::vector<float>(k_weight.data(), k_weight.data() + k_weight.count()),
      kTotalTokens,
      kHidden,
      kHidden);
  const std::vector<float> v_cpu = cpu_matmul_row_major(
      normed,
      std::vector<float>(v_weight.data(), v_weight.data() + v_weight.count()),
      kTotalTokens,
      kHidden,
      kHidden);
  const std::vector<float> attention_cpu =
      cpu_attention(q_cpu, k_cpu, v_cpu, kTotalTokens, kQueryHeads, kKvHeads, kHeadDim);
  const std::vector<float> projected_cpu = cpu_matmul_row_major(
      attention_cpu,
      std::vector<float>(o_weight.data(), o_weight.data() + o_weight.count()),
      kTotalTokens,
      kHidden,
      kHidden);
  std::vector<float> expected_decode(kDecodeTokens * kHidden, 0.0f);
  for (std::size_t i = 0; i < expected_decode.size(); ++i) {
    expected_decode[i] =
        decode_host[i] + projected_cpu[(kPrefixTokens * kHidden) + i];
  }

  std::vector<float> actual_decode(expected_decode.size(), 0.0f);
  if (!expect(
          decode_output->CopyToHost(actual_decode.data(), actual_decode.size()),
          "decode continuation output download should succeed")) {
    return false;
  }
  return expect(
      nearly_equal(actual_decode, expected_decode, 2e-2f),
      "attention continuation should match the CPU decode reference");
}

}  // namespace

int main() {
  if (!test_attention_backend_policy_ignores_legacy_env_overrides() ||
      !test_attention_backend_policy_supports_exact_nano_decode_shape() ||
      !test_attention_layer_slice_matches_cpu_reference() ||
      !test_attention_layer_slice_continues_from_existing_prefix()) {
    return 1;
  }
  std::cout << "attention_layer_test: PASS\n";
  return 0;
}
