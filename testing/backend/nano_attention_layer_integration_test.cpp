#include "nemotron/attention_layer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using nemotron::AttentionLayerBindings;
using nemotron::AttentionLayerConfig;
using nemotron::AttentionLayerExecutionCounters;
using nemotron::AttentionLayerSlice;
using nemotron::AttentionKvCacheConfig;
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
    const float inv_rms =
        1.0f / std::sqrt((sumsq / static_cast<float>(hidden_size)) + epsilon);
    for (std::size_t col = 0; col < hidden_size; ++col) {
      output[row * hidden_size + col] =
          input[row * hidden_size + col] * inv_rms * weight[col];
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

bool all_finite(const std::vector<float>& values) {
  return std::all_of(
      values.begin(),
      values.end(),
      [](float value) { return std::isfinite(value); });
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

RequestExecutionConfig make_request_config(
    std::size_t hidden_size,
    std::size_t max_tokens,
    std::size_t kv_head_count,
    std::size_t head_dim) {
  RequestExecutionConfig config;
  config.hidden_size = hidden_size;
  config.max_tokens = max_tokens;
  config.scratch_tokens = max_tokens;
  config.attention_kv_cache = AttentionKvCacheConfig{
      1,
      kv_head_count,
      head_dim,
      16,
      KvCacheDataType::kBf16,
  };
  config.attention_total_pages =
      nemotron::RequiredPagesForTokens(config.attention_kv_cache, max_tokens);
  return config;
}

std::vector<float> make_input_host(std::size_t token_count, std::size_t hidden_size) {
  std::vector<float> input(token_count * hidden_size, 0.0f);
  for (std::size_t token = 0; token < token_count; ++token) {
    for (std::size_t dim = 0; dim < hidden_size; ++dim) {
      input[token * hidden_size + dim] =
          (0.03f * static_cast<float>(token + 1)) +
          (0.01f * static_cast<float>((dim % 11) - 5));
    }
  }
  return input;
}

bool TestNanoAttentionLayerMatchesCpuReferenceSmallCase() {
  const auto cublas = CublasLtHandle::Create();
  const auto cudnn = CudnnHandle::Create();
  if (!cublas || !cublas->valid() || !cudnn) {
    std::cout << "nano_attention_layer_integration_test: SKIP (no CUDA device available)\n";
    return true;
  }

  constexpr std::size_t kHidden = 64;
  constexpr std::size_t kQueryHeads = 32;
  constexpr std::size_t kKvHeads = 2;
  constexpr std::size_t kHeadDim = 128;
  constexpr std::size_t kTokens = 8;
  constexpr float kEpsilon = 1e-5f;

  const std::size_t kQueryWidth = kQueryHeads * kHeadDim;
  const std::size_t kKvWidth = kKvHeads * kHeadDim;

  std::vector<float> norm_weight_host(kHidden, 1.0f);
  AlignedHostArray<float> q_weight(kQueryWidth * kHidden);
  AlignedHostArray<float> k_weight(kKvWidth * kHidden);
  AlignedHostArray<float> v_weight(kKvWidth * kHidden);
  AlignedHostArray<float> o_weight(kHidden * kQueryWidth);
  if (!q_weight.valid() || !k_weight.valid() || !v_weight.valid() || !o_weight.valid()) {
    std::cout << "nano_attention_layer_integration_test: SKIP (aligned host allocation failed)\n";
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
    o_weight.data()[i * kQueryWidth + i] = 1.0f;
  }

  AttentionLayerBindings bindings;
  const auto norm_descriptor = make_norm_descriptor(norm_weight_host);
  const auto q_descriptor =
      make_dense_descriptor("layers.0.self_attn.q_proj.weight", q_weight, kQueryWidth, kHidden);
  const auto k_descriptor =
      make_dense_descriptor("layers.0.self_attn.k_proj.weight", k_weight, kKvWidth, kHidden);
  const auto v_descriptor =
      make_dense_descriptor("layers.0.self_attn.v_proj.weight", v_weight, kKvWidth, kHidden);
  const auto o_descriptor =
      make_dense_descriptor("layers.0.self_attn.o_proj.weight", o_weight, kHidden, kQueryWidth);
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
  if (!expect(slice != nullptr && slice->valid(), "exact-geometry attention slice should build")) {
    return false;
  }

  auto request_context = RequestExecutionContext::Create(
      make_request_config(kHidden, kTokens, kKvHeads, kHeadDim));
  if (!expect(request_context != nullptr && request_context->valid(), "request context should build")) {
    return false;
  }

  auto input = DeviceTensorFp32::Create({kTokens, kHidden});
  auto output = DeviceTensorFp32::Create({kTokens, kHidden});
  if (!expect(input != nullptr && output != nullptr, "input/output tensors should create")) {
    return false;
  }

  const std::vector<float> input_host = make_input_host(kTokens, kHidden);
  if (!expect(input->CopyFromHost(input_host.data(), input_host.size()), "input upload should succeed") ||
      !expect(request_context->EnsureAttentionTokens(kTokens), "request context should reserve KV pages")) {
    return false;
  }

  nemotron::ResetAttentionLayerExecutionCounters();
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
          "attention layer should execute on exact Nano geometry")) {
    return false;
  }

  const std::vector<float> normed =
      cpu_rms_norm(input_host, norm_weight_host, kTokens, kHidden, kEpsilon);
  const std::vector<float> q_cpu = cpu_matmul_row_major(
      normed,
      std::vector<float>(q_weight.data(), q_weight.data() + q_weight.count()),
      kTokens,
      kHidden,
      kQueryWidth);
  const std::vector<float> k_cpu = cpu_matmul_row_major(
      normed,
      std::vector<float>(k_weight.data(), k_weight.data() + k_weight.count()),
      kTokens,
      kHidden,
      kKvWidth);
  const std::vector<float> v_cpu = cpu_matmul_row_major(
      normed,
      std::vector<float>(v_weight.data(), v_weight.data() + v_weight.count()),
      kTokens,
      kHidden,
      kKvWidth);
  const std::vector<float> attention_cpu =
      cpu_attention(q_cpu, k_cpu, v_cpu, kTokens, kQueryHeads, kKvHeads, kHeadDim);
  const std::vector<float> expected = cpu_matmul_row_major(
      attention_cpu,
      std::vector<float>(o_weight.data(), o_weight.data() + o_weight.count()),
      kTokens,
      kQueryWidth,
      kHidden);

  std::vector<float> actual(expected.size(), 0.0f);
  if (!expect(output->CopyToHost(actual.data(), actual.size()), "output download should succeed")) {
    return false;
  }

  const AttentionLayerExecutionCounters counters =
      nemotron::GetAttentionLayerExecutionCounters();
  return expect(
             counters.native_multi_token_runs == 1 &&
                 counters.native_multi_token_tokens == kTokens,
             "exact-geometry run should record one native multi-token execution") &&
         expect(
             nearly_equal(actual, expected, 3.0e-2f),
             "attention layer output should match CPU reference on exact Nano geometry");
}

bool TestNanoAttentionLayerChunksLargePrefill() {
  const auto cublas = CublasLtHandle::Create();
  const auto cudnn = CudnnHandle::Create();
  if (!cublas || !cublas->valid() || !cudnn) {
    std::cout << "nano_attention_layer_integration_test: SKIP (no CUDA device available)\n";
    return true;
  }

  constexpr std::size_t kHidden = 64;
  constexpr std::size_t kQueryHeads = 32;
  constexpr std::size_t kKvHeads = 2;
  constexpr std::size_t kHeadDim = 128;
  constexpr std::size_t kTokens = 1032;
  constexpr float kEpsilon = 1e-5f;

  const std::size_t kQueryWidth = kQueryHeads * kHeadDim;
  const std::size_t kKvWidth = kKvHeads * kHeadDim;

  std::vector<float> norm_weight_host(kHidden, 1.0f);
  AlignedHostArray<float> q_weight(kQueryWidth * kHidden);
  AlignedHostArray<float> k_weight(kKvWidth * kHidden);
  AlignedHostArray<float> v_weight(kKvWidth * kHidden);
  AlignedHostArray<float> o_weight(kHidden * kQueryWidth);
  if (!q_weight.valid() || !k_weight.valid() || !v_weight.valid() || !o_weight.valid()) {
    std::cout << "nano_attention_layer_integration_test: SKIP (aligned host allocation failed)\n";
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
    o_weight.data()[i * kQueryWidth + i] = 1.0f;
  }

  AttentionLayerBindings bindings;
  const auto norm_descriptor = make_norm_descriptor(norm_weight_host);
  const auto q_descriptor =
      make_dense_descriptor("layers.0.self_attn.q_proj.weight", q_weight, kQueryWidth, kHidden);
  const auto k_descriptor =
      make_dense_descriptor("layers.0.self_attn.k_proj.weight", k_weight, kKvWidth, kHidden);
  const auto v_descriptor =
      make_dense_descriptor("layers.0.self_attn.v_proj.weight", v_weight, kKvWidth, kHidden);
  const auto o_descriptor =
      make_dense_descriptor("layers.0.self_attn.o_proj.weight", o_weight, kHidden, kQueryWidth);
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
  if (!expect(slice != nullptr && slice->valid(), "chunking attention slice should build")) {
    return false;
  }

  auto request_context = RequestExecutionContext::Create(
      make_request_config(kHidden, kTokens, kKvHeads, kHeadDim));
  if (!expect(request_context != nullptr && request_context->valid(), "chunking request context should build")) {
    return false;
  }

  auto input = DeviceTensorFp32::Create({kTokens, kHidden});
  auto output = DeviceTensorFp32::Create({kTokens, kHidden});
  if (!expect(input != nullptr && output != nullptr, "chunking tensors should create")) {
    return false;
  }

  const std::vector<float> input_host = make_input_host(kTokens, kHidden);
  if (!expect(input->CopyFromHost(input_host.data(), input_host.size()), "chunking input upload should succeed") ||
      !expect(request_context->EnsureAttentionTokens(kTokens), "chunking request should reserve KV pages")) {
    return false;
  }

  nemotron::ResetAttentionLayerExecutionCounters();
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
          "large exact-geometry attention layer should execute")) {
    return false;
  }

  std::vector<float> actual(kTokens * kHidden, 0.0f);
  if (!expect(output->CopyToHost(actual.data(), actual.size()), "chunking output download should succeed")) {
    return false;
  }
  const AttentionLayerExecutionCounters counters =
      nemotron::GetAttentionLayerExecutionCounters();
  return expect(all_finite(actual), "chunked attention output should stay finite") &&
         expect(
             counters.native_multi_token_runs == 2 &&
                 counters.native_multi_token_tokens == kTokens,
             "large exact-geometry prefill should execute as two native multi-token chunks");
}

}  // namespace

int main() {
  if (!TestNanoAttentionLayerMatchesCpuReferenceSmallCase() ||
      !TestNanoAttentionLayerChunksLargePrefill()) {
    return 1;
  }
  std::cout << "nano_attention_layer_integration_test: PASS\n";
  return 0;
}
