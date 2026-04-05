#pragma once

#include "nemotron/attention_device_fallback.h"
#include "nemotron/device_tensor.h"
#include "nemotron/paged_attention_plan.h"
#include "nemotron/paged_kv_cache.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace nemotron::attention_harness {

constexpr std::size_t kBatchSize = 1;
constexpr std::size_t kLayerIndex = 0;
constexpr std::size_t kQueryHeadCount = 32;
constexpr std::size_t kKvHeadCount = 2;
constexpr std::size_t kHeadDim = 128;
constexpr std::size_t kTokensPerPage = 16;
constexpr std::size_t kMaxNanoMultiTokenQueryTokens = 1024;

enum class AttentionExecutionBackend {
  kDeviceFallback = 0,
  kNanoMultiToken,
};

inline const char* AttentionExecutionBackendName(AttentionExecutionBackend backend) {
  switch (backend) {
    case AttentionExecutionBackend::kDeviceFallback:
      return "device_fallback";
    case AttentionExecutionBackend::kNanoMultiToken:
      return "nano_multi_token";
  }
  return "unknown";
}

struct CaseSpec {
  std::string name;
  std::string phase;
  std::size_t prefix_tokens = 0;
  std::size_t query_tokens = 0;
  std::size_t total_sequence_tokens = 0;
  bool causal = true;
};

class ScopedNvtxRange {
 public:
  ScopedNvtxRange(bool enabled, std::string label)
      : enabled_(enabled), label_(std::move(label)) {
    if (enabled_) {
      nvtxRangePushA(label_.c_str());
    }
  }

  ~ScopedNvtxRange() {
    if (enabled_) {
      nvtxRangePop();
    }
  }

  ScopedNvtxRange(const ScopedNvtxRange&) = delete;
  ScopedNvtxRange& operator=(const ScopedNvtxRange&) = delete;

 private:
  bool enabled_ = false;
  std::string label_;
};

class CudaEventTimer {
 public:
  CudaEventTimer() {
    valid_ =
        Check(cudaEventCreate(&start_), "cudaEventCreate(start)") &&
        Check(cudaEventCreate(&stop_), "cudaEventCreate(stop)");
    if (!valid_) {
      if (start_ != nullptr) {
        cudaEventDestroy(start_);
        start_ = nullptr;
      }
      if (stop_ != nullptr) {
        cudaEventDestroy(stop_);
        stop_ = nullptr;
      }
    }
  }

  ~CudaEventTimer() {
    if (start_ != nullptr) {
      cudaEventDestroy(start_);
    }
    if (stop_ != nullptr) {
      cudaEventDestroy(stop_);
    }
  }

  bool valid() const {
    return valid_;
  }

  template <typename Fn>
  bool Measure(Fn&& fn, double* elapsed_ms) {
    if (!valid_ || elapsed_ms == nullptr) {
      return false;
    }
    if (!Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(before)")) {
      return false;
    }
    if (!Check(cudaEventRecord(start_), "cudaEventRecord(start)")) {
      return false;
    }
    if (!fn()) {
      return false;
    }
    if (!Check(cudaEventRecord(stop_), "cudaEventRecord(stop)") ||
        !Check(cudaEventSynchronize(stop_), "cudaEventSynchronize(stop)")) {
      return false;
    }
    float elapsed = 0.0f;
    if (!Check(cudaEventElapsedTime(&elapsed, start_, stop_), "cudaEventElapsedTime")) {
      return false;
    }
    *elapsed_ms = static_cast<double>(elapsed);
    return true;
  }

  static bool Check(cudaError_t status, const char* label) {
    if (status == cudaSuccess) {
      return true;
    }
    std::cerr << "nano_paged_attention: " << label << " failed: "
              << cudaGetErrorString(status) << "\n";
    return false;
  }

 private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
  bool valid_ = false;
};

inline bool HasCudaDevice() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

inline std::uint16_t Bf16Bits(__nv_bfloat16 value) {
  std::uint16_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

inline float FromBf16(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

inline __nv_bfloat16 ToBf16(float value) {
  return __float2bfloat16(value);
}

inline std::vector<float> MakeDeterministicFloatData(
    std::size_t count,
    float scale,
    float primary_frequency,
    float secondary_frequency,
    float bias) {
  std::vector<float> values(count, 0.0f);
  for (std::size_t i = 0; i < count; ++i) {
    const float x = static_cast<float>(i + 1);
    values[i] =
        scale * std::sin(x * primary_frequency) +
        (0.5f * scale) * std::cos(x * secondary_frequency) +
        bias;
  }
  return values;
}

inline std::vector<__nv_bfloat16> ToBf16Vector(const std::vector<float>& values) {
  std::vector<__nv_bfloat16> converted(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    converted[i] = ToBf16(values[i]);
  }
  return converted;
}

inline std::vector<float> FromBf16Vector(const std::vector<__nv_bfloat16>& values) {
  std::vector<float> converted(values.size(), 0.0f);
  for (std::size_t i = 0; i < values.size(); ++i) {
    converted[i] = FromBf16(values[i]);
  }
  return converted;
}

inline std::size_t Offset4d(
    std::size_t i0,
    std::size_t i1,
    std::size_t i2,
    std::size_t i3,
    std::size_t d1,
    std::size_t d2,
    std::size_t d3) {
  return ((i0 * d1 + i1) * d2 + i2) * d3 + i3;
}

inline std::vector<__nv_bfloat16> PackQueryRowMajorToAttentionBf16(
    const std::vector<__nv_bfloat16>& row_major,
    std::size_t token_count) {
  std::vector<__nv_bfloat16> packed(kBatchSize * kQueryHeadCount * token_count * kHeadDim);
  for (std::size_t token = 0; token < token_count; ++token) {
    for (std::size_t head = 0; head < kQueryHeadCount; ++head) {
      for (std::size_t dim = 0; dim < kHeadDim; ++dim) {
        const std::size_t src = token * (kQueryHeadCount * kHeadDim) + head * kHeadDim + dim;
        const std::size_t dst = Offset4d(
            0,
            head,
            token,
            dim,
            kQueryHeadCount,
            token_count,
            kHeadDim);
        packed[dst] = row_major[src];
      }
    }
  }
  return packed;
}

inline std::vector<float> AttentionOutputToRowMajorFloat(
    const std::vector<__nv_bfloat16>& attention_layout,
    std::size_t token_count) {
  std::vector<float> row_major(token_count * kQueryHeadCount * kHeadDim, 0.0f);
  for (std::size_t token = 0; token < token_count; ++token) {
    for (std::size_t head = 0; head < kQueryHeadCount; ++head) {
      for (std::size_t dim = 0; dim < kHeadDim; ++dim) {
        const std::size_t dst = token * (kQueryHeadCount * kHeadDim) + head * kHeadDim + dim;
        const std::size_t src = Offset4d(
            0,
            head,
            token,
            dim,
            kQueryHeadCount,
            token_count,
            kHeadDim);
        row_major[dst] = FromBf16(attention_layout[src]);
      }
    }
  }
  return row_major;
}

inline bool ScatterHostRowMajorBf16ToPagedCache(
    const std::vector<__nv_bfloat16>& matrix,
    std::size_t sequence_start,
    std::size_t token_count,
    const std::vector<std::int32_t>& page_table,
    std::size_t max_pages_per_sequence,
    std::vector<__nv_bfloat16>* cache) {
  if (cache == nullptr) {
    return false;
  }
  for (std::size_t token = 0; token < token_count; ++token) {
    const std::size_t absolute_token = sequence_start + token;
    const std::size_t page_slot = absolute_token / kTokensPerPage;
    const std::size_t page_offset = absolute_token % kTokensPerPage;
    if (page_slot >= max_pages_per_sequence || page_slot >= page_table.size()) {
      return false;
    }
    const std::int32_t page_id = page_table[page_slot];
    if (page_id < 0) {
      return false;
    }
    for (std::size_t head = 0; head < kKvHeadCount; ++head) {
      for (std::size_t dim = 0; dim < kHeadDim; ++dim) {
        const std::size_t src = token * (kKvHeadCount * kHeadDim) + head * kHeadDim + dim;
        const std::size_t dst = Offset4d(
            static_cast<std::size_t>(page_id),
            head,
            page_offset,
            dim,
            kKvHeadCount,
            kTokensPerPage,
            kHeadDim);
        (*cache)[dst] = matrix[src];
      }
    }
  }
  return true;
}

inline std::vector<float> CpuPagedAttentionReference(
    const std::vector<__nv_bfloat16>& query_attention,
    const std::vector<__nv_bfloat16>& key_cache,
    const std::vector<__nv_bfloat16>& value_cache,
    const PagedAttentionBatchPlan& batch_plan,
    std::size_t query_tokens,
    std::size_t query_start,
    bool causal) {
  std::vector<__nv_bfloat16> output_attention(
      kBatchSize * kQueryHeadCount * query_tokens * kHeadDim,
      ToBf16(0.0f));
  const float attn_scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
  const std::size_t queries_per_kv_head = kQueryHeadCount / kKvHeadCount;
  const std::size_t kv_tokens = batch_plan.max_sequence_tokens;

  for (std::size_t head = 0; head < kQueryHeadCount; ++head) {
    const std::size_t kv_head = head / queries_per_kv_head;
    for (std::size_t q_token = 0; q_token < query_tokens; ++q_token) {
      const std::size_t visible_kv_tokens =
          causal ? std::min(kv_tokens, query_start + q_token + 1) : kv_tokens;
      std::vector<float> scores(visible_kv_tokens, 0.0f);
      float max_score = -std::numeric_limits<float>::infinity();
      for (std::size_t kv_token = 0; kv_token < visible_kv_tokens; ++kv_token) {
        const std::size_t page_slot = kv_token / kTokensPerPage;
        const std::size_t page_offset = kv_token % kTokensPerPage;
        const std::size_t page_id =
            static_cast<std::size_t>(batch_plan.page_table[page_slot]);
        const std::size_t q_base = Offset4d(
            0,
            head,
            q_token,
            0,
            kQueryHeadCount,
            query_tokens,
            kHeadDim);
        const std::size_t k_base = Offset4d(
            page_id,
            kv_head,
            page_offset,
            0,
            kKvHeadCount,
            kTokensPerPage,
            kHeadDim);
        float score = 0.0f;
        for (std::size_t dim = 0; dim < kHeadDim; ++dim) {
          score += FromBf16(query_attention[q_base + dim]) *
                   FromBf16(key_cache[k_base + dim]);
        }
        score *= attn_scale;
        scores[kv_token] = score;
        max_score = std::max(max_score, score);
      }

      float denom = 0.0f;
      for (float& score : scores) {
        score = std::exp(score - max_score);
        denom += score;
      }

      for (std::size_t dim = 0; dim < kHeadDim; ++dim) {
        float accum = 0.0f;
        for (std::size_t kv_token = 0; kv_token < visible_kv_tokens; ++kv_token) {
          const float weight = scores[kv_token] / denom;
          const std::size_t page_slot = kv_token / kTokensPerPage;
          const std::size_t page_offset = kv_token % kTokensPerPage;
          const std::size_t page_id =
              static_cast<std::size_t>(batch_plan.page_table[page_slot]);
          const std::size_t v_base = Offset4d(
              page_id,
              kv_head,
              page_offset,
              dim,
              kKvHeadCount,
              kTokensPerPage,
              kHeadDim);
          accum += weight * FromBf16(value_cache[v_base]);
        }
        const std::size_t out_base = Offset4d(
            0,
            head,
            q_token,
            dim,
            kQueryHeadCount,
            query_tokens,
            kHeadDim);
        output_attention[out_base] = ToBf16(accum);
      }
    }
  }

  return AttentionOutputToRowMajorFloat(output_attention, query_tokens);
}

inline float MaxAbsDiff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return std::numeric_limits<float>::infinity();
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
  }
  return max_diff;
}

class Runner {
 public:
  static std::unique_ptr<Runner> Create(
      const CaseSpec& spec,
      AttentionExecutionBackend backend = AttentionExecutionBackend::kNanoMultiToken) {
    auto runner = std::unique_ptr<Runner>(new Runner(spec, backend));
    if (!runner->Initialize()) {
      return nullptr;
    }
    return runner;
  }

  const CaseSpec& spec() const {
    return spec_;
  }

  std::size_t total_pages() const {
    return total_pages_;
  }

  bool RunOnce(bool emit_nvtx) {
    if (!SeedPrefixCache()) {
      return false;
    }
    if (!output_attention_ || !output_row_major_ ||
        !output_attention_->FillZero() ||
        !output_row_major_->FillZero()) {
      return false;
    }

    {
      ScopedNvtxRange range(emit_nvtx, spec_.name + ".scatter_kv");
      if (!ScatterRowMajorBf16ToPagedCacheBf16(
              *tail_key_row_major_,
              spec_.prefix_tokens,
              spec_.query_tokens,
              kKvHeadCount,
              kHeadDim,
              kTokensPerPage,
              page_table_->data(),
              batch_plan_.max_pages_per_sequence,
              key_cache_.get()) ||
          !ScatterRowMajorBf16ToPagedCacheBf16(
              *tail_value_row_major_,
              spec_.prefix_tokens,
              spec_.query_tokens,
              kKvHeadCount,
              kHeadDim,
              kTokensPerPage,
              page_table_->data(),
              batch_plan_.max_pages_per_sequence,
              value_cache_.get())) {
        return false;
      }
    }

    if (backend_ == AttentionExecutionBackend::kNanoMultiToken &&
        spec_.query_tokens > kMaxNanoMultiTokenQueryTokens) {
      return RunNanoMultiTokenChunked(emit_nvtx);
    }

    {
      ScopedNvtxRange range(emit_nvtx, spec_.name + ".query_pack");
      if (!ConvertRowMajorBf16ToAttentionQueryBf16(
              *query_row_major_,
              spec_.query_tokens,
              kQueryHeadCount,
              kHeadDim,
              query_attention_.get())) {
        return false;
      }
    }

    {
      ScopedNvtxRange range(emit_nvtx, spec_.name + ".attention");
      switch (backend_) {
        case AttentionExecutionBackend::kNanoMultiToken:
          if (!RunPagedAttentionNanoMultiToken(
                  *query_attention_,
                  *key_cache_,
                  *value_cache_,
                  cache_config_,
                  kBatchSize,
                  batch_plan_.max_pages_per_sequence,
                  page_table_->data(),
                  sequence_lengths_->data(),
                  query_sequence_lengths_->data(),
                  query_starts_->data(),
                  kQueryHeadCount,
                  spec_.query_tokens,
                  1.0f / std::sqrt(static_cast<float>(kHeadDim)),
                  spec_.causal,
                  output_attention_.get())) {
            return false;
          }
          break;
        case AttentionExecutionBackend::kDeviceFallback:
          if (!RunPagedAttentionDeviceFallback(
                  *query_attention_,
                  *key_cache_,
                  *value_cache_,
                  cache_config_,
                  kBatchSize,
                  batch_plan_.max_pages_per_sequence,
                  page_table_->data(),
                  sequence_lengths_->data(),
                  query_sequence_lengths_->data(),
                  query_starts_->data(),
                  kQueryHeadCount,
                  spec_.query_tokens,
                  1.0f / std::sqrt(static_cast<float>(kHeadDim)),
                  spec_.causal,
                  output_attention_.get())) {
            return false;
          }
          break;
      }
    }

    {
      ScopedNvtxRange range(emit_nvtx, spec_.name + ".output_unpack");
      if (!ConvertAttentionOutputBf16ToRowMajorBf16(
              *output_attention_,
              spec_.query_tokens,
              kQueryHeadCount,
              kHeadDim,
              output_row_major_.get())) {
        return false;
      }
    }

    return CudaEventTimer::Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(after pipeline)");
  }

  bool MeasureOnce(bool emit_nvtx, double* elapsed_ms) {
    CudaEventTimer timer;
    if (!timer.valid()) {
      return false;
    }
    ScopedNvtxRange range(emit_nvtx, spec_.name + ".pipeline");
    return timer.Measure([&]() { return RunOnce(emit_nvtx); }, elapsed_ms);
  }

  bool DownloadOutputRowMajor(std::vector<float>* output) const {
    if (output == nullptr || !output_row_major_) {
      return false;
    }
    std::vector<__nv_bfloat16> host(output_row_major_->numel(), ToBf16(0.0f));
    if (!output_row_major_->CopyToHost(host.data(), host.size())) {
      return false;
    }
    *output = FromBf16Vector(host);
    return true;
  }

  std::vector<float> ComputeReferenceRowMajor() const {
    return CpuPagedAttentionReference(
        host_query_attention_,
        host_full_key_cache_,
        host_full_value_cache_,
        batch_plan_,
        spec_.query_tokens,
        spec_.prefix_tokens,
        spec_.causal);
  }

 private:
  Runner(CaseSpec spec, AttentionExecutionBackend backend)
      : spec_(std::move(spec)), backend_(backend) {}

  bool Initialize() {
    if (!HasCudaDevice() ||
        spec_.query_tokens == 0 ||
        spec_.total_sequence_tokens == 0 ||
        spec_.total_sequence_tokens != spec_.prefix_tokens + spec_.query_tokens) {
      return false;
    }

    cache_config_.layer_count = 1;
    cache_config_.kv_head_count = kKvHeadCount;
    cache_config_.head_dim = kHeadDim;
    cache_config_.tokens_per_page = kTokensPerPage;
    cache_config_.dtype = KvCacheDataType::kBf16;

    total_pages_ = RequiredPagesForTokens(cache_config_, spec_.total_sequence_tokens);
    auto arena = PagedKvCacheArena::Create(cache_config_, total_pages_);
    if (!arena.has_value()) {
      return false;
    }

    auto pages = arena->AllocatePages(kLayerIndex, total_pages_);
    if (!pages.has_value()) {
      return false;
    }

    AttentionSequencePages sequence;
    sequence.token_count = spec_.total_sequence_tokens;
    sequence.page_ids.reserve(pages->size());
    for (const auto& page : *pages) {
      sequence.page_ids.push_back(page.page_id);
    }

    auto batch_plan = BuildPagedAttentionBatchPlan(
        cache_config_,
        kLayerIndex,
        {sequence},
        &*arena);
    if (!batch_plan.has_value()) {
      return false;
    }
    batch_plan_ = std::move(*batch_plan);

    const std::size_t query_row_major_count = spec_.query_tokens * kQueryHeadCount * kHeadDim;
    const std::size_t tail_kv_count = spec_.query_tokens * kKvHeadCount * kHeadDim;
    const std::size_t prefix_kv_count = spec_.prefix_tokens * kKvHeadCount * kHeadDim;
    const std::size_t cache_count =
        total_pages_ * kKvHeadCount * kTokensPerPage * kHeadDim;

    host_query_row_major_ = ToBf16Vector(
        MakeDeterministicFloatData(query_row_major_count, 0.27f, 0.011f, 0.017f, 0.01f));
    host_query_attention_ = PackQueryRowMajorToAttentionBf16(host_query_row_major_, spec_.query_tokens);
    host_tail_key_row_major_ = ToBf16Vector(
        MakeDeterministicFloatData(tail_kv_count, 0.21f, 0.013f, 0.019f, -0.02f));
    host_tail_value_row_major_ = ToBf16Vector(
        MakeDeterministicFloatData(tail_kv_count, 0.17f, 0.017f, 0.023f, 0.03f));
    host_full_key_cache_.assign(cache_count, ToBf16(0.0f));
    host_full_value_cache_.assign(cache_count, ToBf16(0.0f));

    if (spec_.prefix_tokens != 0) {
      host_prefix_key_row_major_ = ToBf16Vector(
          MakeDeterministicFloatData(prefix_kv_count, 0.19f, 0.007f, 0.021f, 0.015f));
      host_prefix_value_row_major_ = ToBf16Vector(
          MakeDeterministicFloatData(prefix_kv_count, 0.16f, 0.009f, 0.027f, -0.01f));
      if (!ScatterHostRowMajorBf16ToPagedCache(
              host_prefix_key_row_major_,
              0,
              spec_.prefix_tokens,
              batch_plan_.page_table,
              batch_plan_.max_pages_per_sequence,
              &host_full_key_cache_) ||
          !ScatterHostRowMajorBf16ToPagedCache(
              host_prefix_value_row_major_,
              0,
              spec_.prefix_tokens,
              batch_plan_.page_table,
              batch_plan_.max_pages_per_sequence,
              &host_full_value_cache_)) {
        return false;
      }
    }

    if (!ScatterHostRowMajorBf16ToPagedCache(
            host_tail_key_row_major_,
            spec_.prefix_tokens,
            spec_.query_tokens,
            batch_plan_.page_table,
            batch_plan_.max_pages_per_sequence,
            &host_full_key_cache_) ||
        !ScatterHostRowMajorBf16ToPagedCache(
            host_tail_value_row_major_,
            spec_.prefix_tokens,
            spec_.query_tokens,
            batch_plan_.page_table,
            batch_plan_.max_pages_per_sequence,
            &host_full_value_cache_)) {
      return false;
    }

    query_row_major_ = DeviceTensorBf16::Create({spec_.query_tokens, kQueryHeadCount * kHeadDim});
    query_attention_ = DeviceTensorBf16::Create({kBatchSize, kQueryHeadCount, spec_.query_tokens, kHeadDim});
    tail_key_row_major_ = DeviceTensorBf16::Create({spec_.query_tokens, kKvHeadCount * kHeadDim});
    tail_value_row_major_ = DeviceTensorBf16::Create({spec_.query_tokens, kKvHeadCount * kHeadDim});
    output_attention_ = DeviceTensorBf16::Create({kBatchSize, kQueryHeadCount, spec_.query_tokens, kHeadDim});
    output_row_major_ = DeviceTensorBf16::Create({spec_.query_tokens, kQueryHeadCount * kHeadDim});
    nano_query_attention_scratch_ = DeviceTensorBf16::Create(
        {kBatchSize, kQueryHeadCount, std::min(spec_.query_tokens, kMaxNanoMultiTokenQueryTokens), kHeadDim});
    nano_output_attention_scratch_ = DeviceTensorBf16::Create(
        {kBatchSize, kQueryHeadCount, std::min(spec_.query_tokens, kMaxNanoMultiTokenQueryTokens), kHeadDim});
    key_cache_ = DeviceTensorBf16::Create({total_pages_, kKvHeadCount, kTokensPerPage, kHeadDim});
    value_cache_ = DeviceTensorBf16::Create({total_pages_, kKvHeadCount, kTokensPerPage, kHeadDim});
    page_table_ = DeviceTensorInt32::Create({batch_plan_.page_table.size()});
    sequence_lengths_ = DeviceTensorInt32::Create({kBatchSize});
    query_sequence_lengths_ = DeviceTensorInt32::Create({kBatchSize});
    query_starts_ = DeviceTensorInt32::Create({kBatchSize});
    if (!query_row_major_ ||
        !query_attention_ ||
        !tail_key_row_major_ ||
        !tail_value_row_major_ ||
        !output_attention_ ||
        !output_row_major_ ||
        !nano_query_attention_scratch_ ||
        !nano_output_attention_scratch_ ||
        !key_cache_ ||
        !value_cache_ ||
        !page_table_ ||
        !sequence_lengths_ ||
        !query_sequence_lengths_ ||
        !query_starts_) {
      return false;
    }

    if (!query_row_major_->CopyFromHost(host_query_row_major_.data(), host_query_row_major_.size()) ||
        !tail_key_row_major_->CopyFromHost(host_tail_key_row_major_.data(), host_tail_key_row_major_.size()) ||
        !tail_value_row_major_->CopyFromHost(host_tail_value_row_major_.data(), host_tail_value_row_major_.size()) ||
        !page_table_->CopyFromHost(batch_plan_.page_table.data(), batch_plan_.page_table.size())) {
      return false;
    }

    const std::int32_t sequence_length = static_cast<std::int32_t>(spec_.total_sequence_tokens);
    const std::int32_t query_length = static_cast<std::int32_t>(spec_.query_tokens);
    const std::int32_t query_start = static_cast<std::int32_t>(spec_.prefix_tokens);
    if (!sequence_lengths_->CopyFromHost(&sequence_length, 1) ||
        !query_sequence_lengths_->CopyFromHost(&query_length, 1) ||
        !query_starts_->CopyFromHost(&query_start, 1)) {
      return false;
    }

    if (spec_.prefix_tokens != 0) {
      prefix_key_row_major_ = DeviceTensorBf16::Create({spec_.prefix_tokens, kKvHeadCount * kHeadDim});
      prefix_value_row_major_ = DeviceTensorBf16::Create({spec_.prefix_tokens, kKvHeadCount * kHeadDim});
      if (!prefix_key_row_major_ ||
          !prefix_value_row_major_ ||
          !prefix_key_row_major_->CopyFromHost(
              host_prefix_key_row_major_.data(),
              host_prefix_key_row_major_.size()) ||
          !prefix_value_row_major_->CopyFromHost(
              host_prefix_value_row_major_.data(),
              host_prefix_value_row_major_.size())) {
        return false;
      }
    }

    prefix_seeded_ = false;
    return true;
  }

  bool SeedPrefixCache() {
    if (!key_cache_->FillZero() || !value_cache_->FillZero()) {
      return false;
    }
    if (spec_.prefix_tokens == 0) {
      prefix_seeded_ = true;
      return true;
    }
    if (!prefix_key_row_major_ || !prefix_value_row_major_) {
      return false;
    }
    if (!ScatterRowMajorBf16ToPagedCacheBf16(
            *prefix_key_row_major_,
            0,
            spec_.prefix_tokens,
            kKvHeadCount,
            kHeadDim,
            kTokensPerPage,
            page_table_->data(),
            batch_plan_.max_pages_per_sequence,
            key_cache_.get()) ||
        !ScatterRowMajorBf16ToPagedCacheBf16(
            *prefix_value_row_major_,
            0,
            spec_.prefix_tokens,
            kKvHeadCount,
            kHeadDim,
            kTokensPerPage,
            page_table_->data(),
            batch_plan_.max_pages_per_sequence,
            value_cache_.get())) {
      return false;
    }
    prefix_seeded_ = true;
    return true;
  }

  bool RunNanoMultiTokenChunked(bool emit_nvtx) {
    if (!nano_query_attention_scratch_ || !nano_output_attention_scratch_) {
      return false;
    }

    const std::size_t row_width = kQueryHeadCount * kHeadDim;
    for (std::size_t chunk_start = 0; chunk_start < spec_.query_tokens;
         chunk_start += kMaxNanoMultiTokenQueryTokens) {
      const std::size_t chunk_tokens =
          std::min(kMaxNanoMultiTokenQueryTokens, spec_.query_tokens - chunk_start);
      const std::int32_t sequence_length =
          static_cast<std::int32_t>(spec_.prefix_tokens + chunk_start + chunk_tokens);
      const std::int32_t query_length = static_cast<std::int32_t>(chunk_tokens);
      const std::int32_t query_start =
          static_cast<std::int32_t>(spec_.prefix_tokens + chunk_start);

      auto query_chunk = DeviceTensorBf16::CreateView(
          {chunk_tokens, row_width},
          query_row_major_->data() + (chunk_start * row_width));
      auto output_chunk = DeviceTensorBf16::CreateView(
          {chunk_tokens, row_width},
          output_row_major_->data() + (chunk_start * row_width));
      auto query_attention_chunk = DeviceTensorBf16::CreateView(
          {kBatchSize, kQueryHeadCount, chunk_tokens, kHeadDim},
          nano_query_attention_scratch_->data());
      auto output_attention_chunk = DeviceTensorBf16::CreateView(
          {kBatchSize, kQueryHeadCount, chunk_tokens, kHeadDim},
          nano_output_attention_scratch_->data());
      if (query_chunk == nullptr ||
          output_chunk == nullptr ||
          query_attention_chunk == nullptr ||
          output_attention_chunk == nullptr ||
          !sequence_lengths_->CopyFromHost(&sequence_length, 1) ||
          !query_sequence_lengths_->CopyFromHost(&query_length, 1) ||
          !query_starts_->CopyFromHost(&query_start, 1) ||
          !output_attention_chunk->FillZero()) {
        return false;
      }

      {
        ScopedNvtxRange range(emit_nvtx, spec_.name + ".query_pack");
        if (!ConvertRowMajorBf16ToAttentionQueryBf16(
                *query_chunk,
                chunk_tokens,
                kQueryHeadCount,
                kHeadDim,
                query_attention_chunk.get())) {
          return false;
        }
      }

      {
        ScopedNvtxRange range(emit_nvtx, spec_.name + ".attention");
        if (!RunPagedAttentionNanoMultiToken(
                *query_attention_chunk,
                *key_cache_,
                *value_cache_,
                cache_config_,
                kBatchSize,
                batch_plan_.max_pages_per_sequence,
                page_table_->data(),
                sequence_lengths_->data(),
                query_sequence_lengths_->data(),
                query_starts_->data(),
                kQueryHeadCount,
                chunk_tokens,
                1.0f / std::sqrt(static_cast<float>(kHeadDim)),
                spec_.causal,
                output_attention_chunk.get())) {
          return false;
        }
      }

      {
        ScopedNvtxRange range(emit_nvtx, spec_.name + ".output_unpack");
        if (!ConvertAttentionOutputBf16ToRowMajorBf16(
                *output_attention_chunk,
                chunk_tokens,
                kQueryHeadCount,
                kHeadDim,
                output_chunk.get())) {
          return false;
        }
      }
    }
    return true;
  }

  CaseSpec spec_;
  AttentionExecutionBackend backend_ = AttentionExecutionBackend::kDeviceFallback;
  AttentionKvCacheConfig cache_config_{};
  PagedAttentionBatchPlan batch_plan_{};
  std::size_t total_pages_ = 0;
  bool prefix_seeded_ = false;

  std::vector<__nv_bfloat16> host_query_row_major_;
  std::vector<__nv_bfloat16> host_query_attention_;
  std::vector<__nv_bfloat16> host_prefix_key_row_major_;
  std::vector<__nv_bfloat16> host_prefix_value_row_major_;
  std::vector<__nv_bfloat16> host_tail_key_row_major_;
  std::vector<__nv_bfloat16> host_tail_value_row_major_;
  std::vector<__nv_bfloat16> host_full_key_cache_;
  std::vector<__nv_bfloat16> host_full_value_cache_;

  std::unique_ptr<DeviceTensorBf16> query_row_major_;
  std::unique_ptr<DeviceTensorBf16> query_attention_;
  std::unique_ptr<DeviceTensorBf16> prefix_key_row_major_;
  std::unique_ptr<DeviceTensorBf16> prefix_value_row_major_;
  std::unique_ptr<DeviceTensorBf16> tail_key_row_major_;
  std::unique_ptr<DeviceTensorBf16> tail_value_row_major_;
  std::unique_ptr<DeviceTensorBf16> output_attention_;
  std::unique_ptr<DeviceTensorBf16> output_row_major_;
  std::unique_ptr<DeviceTensorBf16> nano_query_attention_scratch_;
  std::unique_ptr<DeviceTensorBf16> nano_output_attention_scratch_;
  std::unique_ptr<DeviceTensorBf16> key_cache_;
  std::unique_ptr<DeviceTensorBf16> value_cache_;
  std::unique_ptr<DeviceTensorInt32> page_table_;
  std::unique_ptr<DeviceTensorInt32> sequence_lengths_;
  std::unique_ptr<DeviceTensorInt32> query_sequence_lengths_;
  std::unique_ptr<DeviceTensorInt32> query_starts_;
};

}  // namespace nemotron::attention_harness
