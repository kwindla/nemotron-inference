#include "nemotron/cudnn_handle.h"
#include "nemotron/cudnn_paged_attention.h"
#include "nemotron/paged_attention_plan.h"
#include "nemotron/paged_kv_cache.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

using nemotron::AttentionKvCacheConfig;
using nemotron::AttentionSequencePages;
using nemotron::BuildCudnnPagedAttentionConfig;
using nemotron::BuildPagedAttentionBatchPlan;
using nemotron::CudnnHandle;
using nemotron::CudnnPagedAttentionExecution;
using nemotron::CudnnPagedAttentionPlan;
using nemotron::KvCacheDataType;
using nemotron::PagedKvCacheArena;

template <typename T>
class DeviceBuffer {
 public:
  static std::optional<DeviceBuffer> Create(std::size_t count) {
    if (count == 0) {
      return std::nullopt;
    }
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
      return std::nullopt;
    }

    DeviceBuffer buffer;
    buffer.count_ = count;
    if (cudaMalloc(reinterpret_cast<void**>(&buffer.data_), count * sizeof(T)) != cudaSuccess) {
      return std::nullopt;
    }
    return buffer;
  }

  DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), count_(other.count_) {
    other.data_ = nullptr;
    other.count_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    if (data_ != nullptr) {
      cudaFree(data_);
    }
    data_ = other.data_;
    count_ = other.count_;
    other.data_ = nullptr;
    other.count_ = 0;
    return *this;
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* data() const {
    return data_;
  }

  std::size_t count() const {
    return count_;
  }

  bool CopyFromHost(const std::vector<T>& host_values) {
    if (host_values.size() != count_) {
      return false;
    }
    return cudaMemcpy(
               data_,
               host_values.data(),
               count_ * sizeof(T),
               cudaMemcpyHostToDevice) == cudaSuccess &&
           cudaDeviceSynchronize() == cudaSuccess;
  }

  bool CopyToHost(std::vector<T>* host_values) const {
    if (host_values == nullptr || host_values->size() != count_) {
      return false;
    }
    return cudaMemcpy(
               host_values->data(),
               data_,
               count_ * sizeof(T),
               cudaMemcpyDeviceToHost) == cudaSuccess &&
           cudaDeviceSynchronize() == cudaSuccess;
  }

  bool FillZero() {
    return cudaMemset(data_, 0, count_ * sizeof(T)) == cudaSuccess &&
           cudaDeviceSynchronize() == cudaSuccess;
  }

 private:
  DeviceBuffer() = default;

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

__nv_bfloat16 to_bf16(float value) {
  return __float2bfloat16(value);
}

float from_bf16(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

std::vector<__nv_bfloat16> to_bf16_vector(const std::vector<float>& values) {
  std::vector<__nv_bfloat16> converted(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    converted[i] = to_bf16(values[i]);
  }
  return converted;
}

std::vector<float> from_bf16_vector(const std::vector<__nv_bfloat16>& values) {
  std::vector<float> converted(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    converted[i] = from_bf16(values[i]);
  }
  return converted;
}

std::size_t offset_4d(
    std::size_t i0,
    std::size_t i1,
    std::size_t i2,
    std::size_t i3,
    std::size_t d1,
    std::size_t d2,
    std::size_t d3) {
  return ((i0 * d1 + i1) * d2 + i2) * d3 + i3;
}

std::vector<float> cpu_paged_attention_reference(
    const std::vector<float>& query,
    const std::vector<float>& key_cache,
    const std::vector<float>& value_cache,
    const std::vector<std::int32_t>& seq_len_q,
    const std::vector<std::int32_t>& seq_len_kv,
    const std::vector<std::int32_t>& page_table,
    std::size_t batch_size,
    std::size_t query_head_count,
    std::size_t kv_head_count,
    std::size_t max_query_tokens,
    std::size_t max_pages_per_sequence,
    std::size_t tokens_per_page,
    std::size_t head_dim,
    float attn_scale,
    bool causal) {
  std::vector<float> output(batch_size * query_head_count * max_query_tokens * head_dim, 0.0f);

  for (std::size_t batch = 0; batch < batch_size; ++batch) {
    const std::size_t q_tokens = static_cast<std::size_t>(seq_len_q[batch]);
    const std::size_t kv_tokens = static_cast<std::size_t>(seq_len_kv[batch]);
    for (std::size_t head = 0; head < query_head_count; ++head) {
      const std::size_t kv_head = head % kv_head_count;
      for (std::size_t q_token = 0; q_token < q_tokens; ++q_token) {
        const std::size_t visible_kv_tokens = causal ? std::min(kv_tokens, q_token + 1) : kv_tokens;
        std::vector<float> scores(visible_kv_tokens, 0.0f);
        float max_score = -std::numeric_limits<float>::infinity();
        for (std::size_t kv_token = 0; kv_token < visible_kv_tokens; ++kv_token) {
          const std::size_t page_slot = kv_token / tokens_per_page;
          const std::size_t page_offset = kv_token % tokens_per_page;
          const std::int32_t page_id = page_table[batch * max_pages_per_sequence + page_slot];
          const std::size_t q_base = offset_4d(batch, head, q_token, 0, query_head_count, max_query_tokens, head_dim);
          const std::size_t k_base = offset_4d(
              static_cast<std::size_t>(page_id),
              kv_head,
              page_offset,
              0,
              kv_head_count,
              tokens_per_page,
              head_dim);

          float score = 0.0f;
          for (std::size_t dim = 0; dim < head_dim; ++dim) {
            score += query[q_base + dim] * key_cache[k_base + dim];
          }
          score *= attn_scale;
          scores[kv_token] = score;
          max_score = std::max(max_score, score);
        }

        float sum = 0.0f;
        for (float& score : scores) {
          score = std::exp(score - max_score);
          sum += score;
        }

        const std::size_t out_base = offset_4d(batch, head, q_token, 0, query_head_count, max_query_tokens, head_dim);
        for (std::size_t kv_token = 0; kv_token < visible_kv_tokens; ++kv_token) {
          const float weight = scores[kv_token] / sum;
          const std::size_t page_slot = kv_token / tokens_per_page;
          const std::size_t page_offset = kv_token % tokens_per_page;
          const std::int32_t page_id = page_table[batch * max_pages_per_sequence + page_slot];
          const std::size_t v_base = offset_4d(
              static_cast<std::size_t>(page_id),
              kv_head,
              page_offset,
              0,
              kv_head_count,
              tokens_per_page,
              head_dim);
          for (std::size_t dim = 0; dim < head_dim; ++dim) {
            output[out_base + dim] += weight * value_cache[v_base + dim];
          }
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

bool test_build_cudnn_paged_attention_config_from_batch_plan() {
  AttentionKvCacheConfig cache_config;
  cache_config.layer_count = 1;
  cache_config.kv_head_count = 2;
  cache_config.head_dim = 64;
  cache_config.tokens_per_page = 16;
  cache_config.dtype = KvCacheDataType::kBf16;

  const auto kv_plan = BuildPagedAttentionBatchPlan(
      cache_config,
      0,
      {
          AttentionSequencePages{16, {0}},
          AttentionSequencePages{31, {1, 2}},
      },
      nullptr);
  if (!expect(kv_plan.has_value(), "KV plan should build")) {
    return false;
  }

  const auto config = BuildCudnnPagedAttentionConfig(*kv_plan, 4, 1, 8);
  if (!expect(config.has_value(), "cuDNN config should build from KV plan")) {
    return false;
  }

  return expect(config->valid(), "cuDNN config should validate") &&
         expect(config->batch_size == 2, "batch size should match KV plan") &&
         expect(config->query_head_count == 4, "query head count should be preserved") &&
         expect(config->max_query_tokens == 1, "max query tokens should be preserved") &&
         expect(config->max_kv_tokens == 31, "max KV tokens should come from KV plan") &&
         expect(config->page_table_entries == 2, "page table entries should match KV plan") &&
         expect(config->container_page_count == 8, "container page count should be preserved") &&
         expect(std::fabs(config->attn_scale - 0.125f) < 1e-6f, "default attention scale should be 1/sqrt(head_dim)");
}

bool run_paged_attention_reference_case(
    std::size_t batch_size,
    std::size_t max_query_tokens,
    const std::vector<std::int32_t>& seq_len_q,
    const std::vector<std::int32_t>& seq_len_kv,
    bool causal,
    const std::string& case_name) {
  const auto handle = CudnnHandle::Create();
  if (!handle || !handle->valid() || handle->version() < 90500) {
    std::cout << "cudnn_paged_attention_test: SKIP (no CUDA device or cuDNN >= 9.5 unavailable)\n";
    return true;
  }

  AttentionKvCacheConfig cache_config;
  cache_config.layer_count = 1;
  cache_config.kv_head_count = 2;
  cache_config.head_dim = 64;
  cache_config.tokens_per_page = 16;
  cache_config.dtype = KvCacheDataType::kBf16;

  if (!expect(seq_len_q.size() == batch_size, case_name + ": seq_len_q size should match batch")) {
    return false;
  }
  if (!expect(seq_len_kv.size() == batch_size, case_name + ": seq_len_kv size should match batch")) {
    return false;
  }

  std::size_t total_pages = 0;
  std::vector<std::size_t> pages_per_sequence(batch_size, 0);
  for (std::size_t batch = 0; batch < batch_size; ++batch) {
    pages_per_sequence[batch] =
        (static_cast<std::size_t>(seq_len_kv[batch]) + cache_config.tokens_per_page - 1) / cache_config.tokens_per_page;
    total_pages += pages_per_sequence[batch];
  }

  auto arena = PagedKvCacheArena::Create(cache_config, total_pages);
  if (!expect(arena.has_value(), "paged KV arena should build")) {
    return false;
  }

  std::vector<AttentionSequencePages> sequences;
  sequences.reserve(batch_size);
  for (std::size_t batch = 0; batch < batch_size; ++batch) {
    const auto pages = arena->AllocatePages(0, pages_per_sequence[batch]);
    if (!expect(pages.has_value(), case_name + ": page allocation should succeed")) {
      return false;
    }
    AttentionSequencePages sequence;
    sequence.token_count = static_cast<std::size_t>(seq_len_kv[batch]);
    sequence.page_ids.reserve(pages_per_sequence[batch]);
    for (const auto& page : *pages) {
      sequence.page_ids.push_back(page.page_id);
    }
    sequences.push_back(std::move(sequence));
  }

  const auto kv_plan = BuildPagedAttentionBatchPlan(cache_config, 0, sequences, &*arena);
  if (!expect(kv_plan.has_value(), "KV plan should build against arena")) {
    return false;
  }

  const auto config = BuildCudnnPagedAttentionConfig(
      *kv_plan,
      2,
      max_query_tokens,
      arena->total_pages(),
      0.0f,
      causal,
      false);
  if (!expect(config.has_value(), case_name + ": cuDNN config should build")) {
    return false;
  }

  const auto plan = CudnnPagedAttentionPlan::Create(*handle, *config);
  if (!expect(plan != nullptr && plan->valid(), case_name + ": cuDNN paged attention plan should build")) {
    return false;
  }

  const std::size_t q_count =
      config->batch_size * config->query_head_count * config->max_query_tokens * cache_config.head_dim;
  const std::size_t kv_count =
      config->container_page_count * cache_config.kv_head_count * cache_config.tokens_per_page * cache_config.head_dim;

  std::vector<float> query_fp32(q_count);
  std::vector<float> key_fp32(kv_count);
  std::vector<float> value_fp32(kv_count);
  for (std::size_t i = 0; i < q_count; ++i) {
    query_fp32[i] = 0.25f * std::sin(static_cast<float>(i + 1) * 0.031f);
  }
  for (std::size_t i = 0; i < kv_count; ++i) {
    key_fp32[i] = 0.20f * std::cos(static_cast<float>(i + 3) * 0.019f);
    value_fp32[i] = 0.15f * std::sin(static_cast<float>(i + 5) * 0.023f);
  }

  const auto query_bf16 = to_bf16_vector(query_fp32);
  const auto key_bf16 = to_bf16_vector(key_fp32);
  const auto value_bf16 = to_bf16_vector(value_fp32);

  auto query_dev = DeviceBuffer<__nv_bfloat16>::Create(query_bf16.size());
  auto key_dev = DeviceBuffer<__nv_bfloat16>::Create(key_bf16.size());
  auto value_dev = DeviceBuffer<__nv_bfloat16>::Create(value_bf16.size());
  auto output_dev = DeviceBuffer<__nv_bfloat16>::Create(query_bf16.size());
  auto seq_len_q_dev = DeviceBuffer<std::int32_t>::Create(config->batch_size);
  auto seq_len_kv_dev = DeviceBuffer<std::int32_t>::Create(config->batch_size);
  auto page_table_k_dev = DeviceBuffer<std::int32_t>::Create(kv_plan->page_table.size());
  auto page_table_v_dev = DeviceBuffer<std::int32_t>::Create(kv_plan->page_table.size());

  if (!expect(
          query_dev.has_value() && key_dev.has_value() && value_dev.has_value() && output_dev.has_value() &&
              seq_len_q_dev.has_value() && seq_len_kv_dev.has_value() &&
              page_table_k_dev.has_value() && page_table_v_dev.has_value(),
          case_name + ": all device buffers should allocate")) {
    return false;
  }

  const bool copied =
      query_dev->CopyFromHost(query_bf16) &&
      key_dev->CopyFromHost(key_bf16) &&
      value_dev->CopyFromHost(value_bf16) &&
      output_dev->FillZero() &&
      seq_len_q_dev->CopyFromHost(seq_len_q) &&
      seq_len_kv_dev->CopyFromHost(seq_len_kv) &&
      page_table_k_dev->CopyFromHost(kv_plan->page_table) &&
      page_table_v_dev->CopyFromHost(kv_plan->page_table);
  if (!expect(copied, case_name + ": device inputs should upload")) {
    return false;
  }

  const CudnnPagedAttentionExecution execution = {
      query_dev->data(),
      key_dev->data(),
      value_dev->data(),
      seq_len_q_dev->data(),
      seq_len_kv_dev->data(),
      page_table_k_dev->data(),
      page_table_v_dev->data(),
      output_dev->data(),
      nullptr,
  };
  if (!expect(plan->Execute(*handle, execution), case_name + ": cuDNN paged attention execution should succeed")) {
    return false;
  }

  std::vector<__nv_bfloat16> output_bf16(query_bf16.size());
  if (!expect(output_dev->CopyToHost(&output_bf16), case_name + ": output should copy back from device")) {
    return false;
  }

  const std::vector<float> query_rounded = from_bf16_vector(query_bf16);
  const std::vector<float> key_rounded = from_bf16_vector(key_bf16);
  const std::vector<float> value_rounded = from_bf16_vector(value_bf16);
  const std::vector<float> output_rounded = from_bf16_vector(output_bf16);
  const std::vector<float> expected = cpu_paged_attention_reference(
      query_rounded,
      key_rounded,
      value_rounded,
      seq_len_q,
      seq_len_kv,
      kv_plan->page_table,
      config->batch_size,
      config->query_head_count,
      cache_config.kv_head_count,
      config->max_query_tokens,
      config->page_table_entries,
      cache_config.tokens_per_page,
      cache_config.head_dim,
      config->attn_scale,
      causal);

  return expect(
      nearly_equal(output_rounded, expected, 2.5e-2f),
      case_name + ": cuDNN paged attention output should match CPU reference");
}

bool test_cudnn_paged_attention_executes_decode_against_cpu_reference() {
  return run_paged_attention_reference_case(
      2,
      1,
      {1, 1},
      {17, 29},
      false,
      "decode_reference");
}

bool test_cudnn_paged_attention_executes_causal_prefill_against_cpu_reference() {
  return run_paged_attention_reference_case(
      2,
      4,
      {4, 3},
      {4, 3},
      true,
      "causal_prefill_reference");
}

}  // namespace

int main() {
  if (!test_build_cudnn_paged_attention_config_from_batch_plan()) {
    return 1;
  }
  if (!test_cudnn_paged_attention_executes_decode_against_cpu_reference()) {
    return 1;
  }
  if (!test_cudnn_paged_attention_executes_causal_prefill_against_cpu_reference()) {
    return 1;
  }
  std::cout << "cudnn_paged_attention_test: PASS\n";
  return 0;
}
