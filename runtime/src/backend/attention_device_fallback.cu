#include "nemotron/attention_device_fallback.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>

namespace nemotron {
namespace {

constexpr int kAttentionFallbackBlockSize = 1;
constexpr int kPagedAttentionDecodeProductionBlockSize = 256;
constexpr int kPagedAttentionDecodeWarpSize = 32;
constexpr int kPagedAttentionDecodeWarpsPerBlock =
    kPagedAttentionDecodeProductionBlockSize / kPagedAttentionDecodeWarpSize;
constexpr int kPagedAttentionDecodeQueryHeadsPerWarp = 2;
constexpr int kPagedAttentionDecodeQueryHeadsPerBlock =
    kPagedAttentionDecodeWarpsPerBlock * kPagedAttentionDecodeQueryHeadsPerWarp;
constexpr int kPagedAttentionDecodeHeadDim = 128;
constexpr int kPagedAttentionDecodeDimsPerLane =
    kPagedAttentionDecodeHeadDim / kPagedAttentionDecodeWarpSize;
constexpr unsigned int kPagedAttentionDecodeFullMask = 0xffffffffu;

static_assert(
    kPagedAttentionDecodeProductionBlockSize == 2 * kPagedAttentionDecodeHeadDim,
    "Decode block shape assumes one thread per staged K/V element.");
static_assert(
    kPagedAttentionDecodeHeadDim % kPagedAttentionDecodeWarpSize == 0,
    "Decode kernel requires a whole-number dim shard per lane.");

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

__device__ inline float WarpReduceSum(float value) {
  for (int offset = kPagedAttentionDecodeWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(kPagedAttentionDecodeFullMask, value, offset);
  }
  return value;
}

__global__ void ConvertRowMajorFp32ToAttentionQueryBf16Kernel(
    const float* input,
    std::size_t token_count,
    std::size_t query_head_count,
    std::size_t head_dim,
    __nv_bfloat16* output) {
  const std::size_t index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const std::size_t total = token_count * query_head_count * head_dim;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = index; i < total; i += stride) {
    const std::size_t token = i / (query_head_count * head_dim);
    const std::size_t rem = i % (query_head_count * head_dim);
    const std::size_t head = rem / head_dim;
    const std::size_t dim = rem % head_dim;
    const std::size_t dst = ((head * token_count) + token) * head_dim + dim;
    output[dst] = __float2bfloat16(input[i]);
  }
}

__global__ void ConvertAttentionOutputBf16ToRowMajorFp32Kernel(
    const __nv_bfloat16* input,
    std::size_t token_count,
    std::size_t query_head_count,
    std::size_t head_dim,
    float* output) {
  const std::size_t index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const std::size_t total = token_count * query_head_count * head_dim;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = index; i < total; i += stride) {
    const std::size_t token = i / (query_head_count * head_dim);
    const std::size_t rem = i % (query_head_count * head_dim);
    const std::size_t head = rem / head_dim;
    const std::size_t dim = rem % head_dim;
    const std::size_t src = ((head * token_count) + token) * head_dim + dim;
    output[i] = __bfloat162float(input[src]);
  }
}

__global__ void ScatterRowMajorFp32ToPagedCacheBf16Kernel(
    const float* matrix,
    std::size_t sequence_start,
    std::size_t token_count,
    std::size_t kv_head_count,
    std::size_t head_dim,
    std::size_t tokens_per_page,
    const std::int32_t* page_table,
    std::size_t max_pages_per_sequence,
    __nv_bfloat16* cache) {
  const std::size_t index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const std::size_t total = token_count * kv_head_count * head_dim;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = index; i < total; i += stride) {
    const std::size_t token = i / (kv_head_count * head_dim);
    const std::size_t rem = i % (kv_head_count * head_dim);
    const std::size_t head = rem / head_dim;
    const std::size_t dim = rem % head_dim;
    const std::size_t absolute_token = sequence_start + token;
    const std::size_t page_slot = absolute_token / tokens_per_page;
    const std::size_t page_offset = absolute_token % tokens_per_page;
    if (page_slot >= max_pages_per_sequence) {
      continue;
    }
    const std::int32_t page_id = page_table[page_slot];
    if (page_id < 0) {
      continue;
    }
    const std::size_t dst =
        ((((static_cast<std::size_t>(page_id) * kv_head_count) + head) * tokens_per_page) + page_offset) *
            head_dim +
        dim;
    cache[dst] = __float2bfloat16(matrix[i]);
  }
}

__global__ void PagedAttentionDeviceFallbackKernel(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache,
    std::size_t batch_size,
    std::size_t query_head_count,
    std::size_t kv_head_count,
    std::size_t max_query_tokens,
    std::size_t head_dim,
    std::size_t tokens_per_page,
    std::size_t max_pages_per_sequence,
    const std::int32_t* page_table,
    const std::int32_t* sequence_lengths,
    const std::int32_t* query_sequence_lengths,
    const std::int32_t* query_sequence_starts,
    float attn_scale,
    bool causal,
    __nv_bfloat16* output) {
  extern __shared__ float score_scratch[];
  const std::size_t vector_index = static_cast<std::size_t>(blockIdx.x);
  const std::size_t batch = vector_index / (query_head_count * max_query_tokens);
  if (batch >= batch_size) {
    return;
  }
  const std::size_t head_vector = vector_index % (query_head_count * max_query_tokens);
  const std::size_t head = head_vector / max_query_tokens;
  const std::size_t q_token = head_vector % max_query_tokens;

  const std::size_t q_tokens = static_cast<std::size_t>(query_sequence_lengths[batch]);
  if (q_token >= q_tokens) {
    return;
  }

  const std::size_t q_start = static_cast<std::size_t>(query_sequence_starts[batch]);
  const std::size_t kv_tokens = static_cast<std::size_t>(sequence_lengths[batch]);
  const std::size_t visible_kv_tokens =
      causal ? min(kv_tokens, q_start + q_token + 1u) : kv_tokens;
  const std::size_t q_base =
      (((batch * query_head_count) + head) * max_query_tokens + q_token) * head_dim;
  if (threadIdx.x != 0) {
    return;
  }

  if (visible_kv_tokens == 0) {
    for (std::size_t dim = 0; dim < head_dim; ++dim) {
      output[q_base + dim] = __float2bfloat16(0.0f);
    }
    return;
  }

  const std::size_t kv_head = head % kv_head_count;
  float max_score = -INFINITY;
  for (std::size_t kv_token = 0; kv_token < visible_kv_tokens; ++kv_token) {
    const std::size_t page_slot = kv_token / tokens_per_page;
    const std::size_t page_offset = kv_token % tokens_per_page;
    if (page_slot >= max_pages_per_sequence) {
      continue;
    }
    const std::int32_t page_id = page_table[batch * max_pages_per_sequence + page_slot];
    if (page_id < 0) {
      continue;
    }
    const std::size_t kv_base =
        ((((static_cast<std::size_t>(page_id) * kv_head_count) + kv_head) * tokens_per_page) + page_offset) *
        head_dim;

    float score = 0.0f;
    for (std::size_t dim = 0; dim < head_dim; ++dim) {
      score = __fadd_rn(
          score,
          __fmul_rn(
              __bfloat162float(query[q_base + dim]),
              __bfloat162float(key_cache[kv_base + dim])));
    }
    const float scaled_score = __fmul_rn(score, attn_scale);
    score_scratch[kv_token] = scaled_score;
    max_score = fmaxf(max_score, scaled_score);
  }

  double denom = 0.0;
  for (std::size_t kv_token = 0; kv_token < visible_kv_tokens; ++kv_token) {
    score_scratch[kv_token] =
        static_cast<float>(exp(static_cast<double>(score_scratch[kv_token] - max_score)));
    denom += static_cast<double>(score_scratch[kv_token]);
  }

  if (!(denom > 0.0)) {
    for (std::size_t dim = 0; dim < head_dim; ++dim) {
      output[q_base + dim] = __float2bfloat16(0.0f);
    }
    return;
  }

  for (std::size_t dim = 0; dim < head_dim; ++dim) {
    double accum = 0.0;
    for (std::size_t kv_token = 0; kv_token < visible_kv_tokens; ++kv_token) {
      const std::size_t page_slot = kv_token / tokens_per_page;
      const std::size_t page_offset = kv_token % tokens_per_page;
      if (page_slot >= max_pages_per_sequence) {
        continue;
      }
      const std::int32_t page_id = page_table[batch * max_pages_per_sequence + page_slot];
      if (page_id < 0) {
        continue;
      }
      const std::size_t kv_base =
          ((((static_cast<std::size_t>(page_id) * kv_head_count) + kv_head) * tokens_per_page) + page_offset) *
          head_dim;
      const double weight = static_cast<double>(score_scratch[kv_token]) / denom;
      accum += weight * static_cast<double>(__bfloat162float(value_cache[kv_base + dim]));
    }
    output[q_base + dim] = __float2bfloat16(static_cast<float>(accum));
  }
}

// Fixed-size decode kernel for Nano's 128-dim GQA layout. Each block owns one
// KV head, stages a single K/V vector in shared memory, and each warp handles
// up to two query heads from that KV group.
__global__ void PagedAttentionDecodeProductionKernel(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache,
    std::size_t batch_size,
    std::size_t query_head_count,
    std::size_t kv_head_count,
    std::size_t max_query_tokens,
    std::size_t head_dim,
    std::size_t tokens_per_page,
    std::size_t max_pages_per_sequence,
    const std::int32_t* page_table,
    const std::int32_t* sequence_lengths,
    const std::int32_t* query_sequence_lengths,
    const std::int32_t* query_sequence_starts,
    float attn_scale,
    bool causal,
    __nv_bfloat16* output) {
  __shared__ float shared_key[kPagedAttentionDecodeHeadDim];
  __shared__ float shared_value[kPagedAttentionDecodeHeadDim];
  __shared__ std::size_t shared_kv_base;
  __shared__ int shared_kv_valid;

  const std::size_t block_index = static_cast<std::size_t>(blockIdx.x);
  const std::size_t batch = block_index / kv_head_count;
  if (batch >= batch_size) {
    return;
  }

  const std::size_t kv_head = block_index % kv_head_count;
  const std::size_t q_tokens = static_cast<std::size_t>(query_sequence_lengths[batch]);
  if (q_tokens == 0) {
    return;
  }

  const std::size_t q_start = static_cast<std::size_t>(query_sequence_starts[batch]);
  const std::size_t kv_tokens = static_cast<std::size_t>(sequence_lengths[batch]);
  const std::size_t visible_kv_tokens =
      causal ? ((q_start + 1u) < kv_tokens ? (q_start + 1u) : kv_tokens) : kv_tokens;
  const std::size_t queries_per_kv_head = query_head_count / kv_head_count;
  const std::size_t query_head_group_start = kv_head * queries_per_kv_head;
  const std::size_t query_head_group_end = query_head_group_start + queries_per_kv_head;

  const std::size_t warp = static_cast<std::size_t>(threadIdx.x) / kPagedAttentionDecodeWarpSize;
  const std::size_t lane = static_cast<std::size_t>(threadIdx.x) % kPagedAttentionDecodeWarpSize;
  const std::size_t dim_base = lane * kPagedAttentionDecodeDimsPerLane;
  const std::size_t warp_query_head_start =
      query_head_group_start + (warp * kPagedAttentionDecodeQueryHeadsPerWarp);

  const std::size_t query_heads[kPagedAttentionDecodeQueryHeadsPerWarp] = {
      warp_query_head_start,
      warp_query_head_start + 1u};
  const bool head_valid[kPagedAttentionDecodeQueryHeadsPerWarp] = {
      query_heads[0] < query_head_group_end,
      query_heads[1] < query_head_group_end};

  float query_frag[kPagedAttentionDecodeQueryHeadsPerWarp][kPagedAttentionDecodeDimsPerLane] = {};
  float output_frag[kPagedAttentionDecodeQueryHeadsPerWarp][kPagedAttentionDecodeDimsPerLane] = {};
  float max_score[kPagedAttentionDecodeQueryHeadsPerWarp] = {-INFINITY, -INFINITY};
  float sum_exp[kPagedAttentionDecodeQueryHeadsPerWarp] = {0.0f, 0.0f};

  for (int head_slot = 0; head_slot < kPagedAttentionDecodeQueryHeadsPerWarp; ++head_slot) {
    if (!head_valid[head_slot]) {
      continue;
    }
    const std::size_t q_base =
        (((batch * query_head_count) + query_heads[head_slot]) * max_query_tokens) * head_dim;
    for (int dim_offset = 0; dim_offset < kPagedAttentionDecodeDimsPerLane; ++dim_offset) {
      query_frag[head_slot][dim_offset] =
          __bfloat162float(query[q_base + dim_base + static_cast<std::size_t>(dim_offset)]);
    }
  }

  for (std::size_t kv_token = 0; kv_token < visible_kv_tokens; ++kv_token) {
    if (threadIdx.x == 0) {
      shared_kv_valid = 0;
      const std::size_t page_slot = kv_token / tokens_per_page;
      const std::size_t page_offset = kv_token % tokens_per_page;
      if (page_slot < max_pages_per_sequence) {
        const std::int32_t page_id =
            page_table[batch * max_pages_per_sequence + page_slot];
        if (page_id >= 0) {
          shared_kv_base =
              ((((static_cast<std::size_t>(page_id) * kv_head_count) + kv_head) * tokens_per_page) + page_offset) *
              head_dim;
          shared_kv_valid = 1;
        }
      }
    }
    __syncthreads();

    if (shared_kv_valid != 0) {
      if (threadIdx.x < kPagedAttentionDecodeHeadDim) {
        shared_key[threadIdx.x] =
            __bfloat162float(key_cache[shared_kv_base + static_cast<std::size_t>(threadIdx.x)]);
      } else {
        const int value_dim = threadIdx.x - kPagedAttentionDecodeHeadDim;
        shared_value[value_dim] =
            __bfloat162float(value_cache[shared_kv_base + static_cast<std::size_t>(value_dim)]);
      }
      __syncthreads();

      float key_frag[kPagedAttentionDecodeDimsPerLane];
      float value_frag[kPagedAttentionDecodeDimsPerLane];
      for (int dim_offset = 0; dim_offset < kPagedAttentionDecodeDimsPerLane; ++dim_offset) {
        const std::size_t dim = dim_base + static_cast<std::size_t>(dim_offset);
        key_frag[dim_offset] = shared_key[dim];
        value_frag[dim_offset] = shared_value[dim];
      }

      for (int head_slot = 0; head_slot < kPagedAttentionDecodeQueryHeadsPerWarp; ++head_slot) {
        if (!head_valid[head_slot]) {
          continue;
        }

        float score = 0.0f;
        for (int dim_offset = 0; dim_offset < kPagedAttentionDecodeDimsPerLane; ++dim_offset) {
          score = __fmaf_rn(query_frag[head_slot][dim_offset], key_frag[dim_offset], score);
        }
        score = WarpReduceSum(score);
        score = __shfl_sync(kPagedAttentionDecodeFullMask, score, 0);

        const float scaled_score = __fmul_rn(score, attn_scale);
        const float new_max = fmaxf(max_score[head_slot], scaled_score);
        const float correction = expf(max_score[head_slot] - new_max);
        const float weight = expf(scaled_score - new_max);
        for (int dim_offset = 0; dim_offset < kPagedAttentionDecodeDimsPerLane; ++dim_offset) {
          output_frag[head_slot][dim_offset] =
              __fmaf_rn(weight, value_frag[dim_offset], output_frag[head_slot][dim_offset] * correction);
        }
        sum_exp[head_slot] = __fmaf_rn(sum_exp[head_slot], correction, weight);
        max_score[head_slot] = new_max;
      }
    }

    __syncthreads();
  }

  for (int head_slot = 0; head_slot < kPagedAttentionDecodeQueryHeadsPerWarp; ++head_slot) {
    if (!head_valid[head_slot]) {
      continue;
    }
    const float inv_sum = sum_exp[head_slot] > 0.0f ? (1.0f / sum_exp[head_slot]) : 0.0f;
    const std::size_t q_base =
        (((batch * query_head_count) + query_heads[head_slot]) * max_query_tokens) * head_dim;
    for (int dim_offset = 0; dim_offset < kPagedAttentionDecodeDimsPerLane; ++dim_offset) {
      output[q_base + dim_base + static_cast<std::size_t>(dim_offset)] =
          __float2bfloat16(output_frag[head_slot][dim_offset] * inv_sum);
    }
  }
}

}  // namespace

bool ConvertRowMajorFp32ToAttentionQueryBf16(
    const DeviceTensorFp32& matrix,
    std::size_t token_count,
    std::size_t query_head_count,
    std::size_t head_dim,
    DeviceTensorBf16* output) {
  if (!matrix.valid() ||
      output == nullptr ||
      !output->valid() ||
      matrix.shape().size() != 2 ||
      matrix.shape()[0] != token_count ||
      matrix.shape()[1] != query_head_count * head_dim ||
      output->numel() != token_count * query_head_count * head_dim) {
    return false;
  }

  const std::size_t total = token_count * query_head_count * head_dim;
  const int block_size = 256;
  const int grid_size =
      static_cast<int>((total + static_cast<std::size_t>(block_size) - 1u) / static_cast<std::size_t>(block_size));
  ConvertRowMajorFp32ToAttentionQueryBf16Kernel<<<grid_size, block_size>>>(
      matrix.data(),
      token_count,
      query_head_count,
      head_dim,
      output->data());
  return CheckCuda(cudaGetLastError());
}

bool ScatterRowMajorFp32ToPagedCacheBf16(
    const DeviceTensorFp32& matrix,
    std::size_t sequence_start,
    std::size_t token_count,
    std::size_t kv_head_count,
    std::size_t head_dim,
    std::size_t tokens_per_page,
    const std::int32_t* page_table_device,
    std::size_t max_pages_per_sequence,
    DeviceTensorBf16* cache) {
  if (!matrix.valid() ||
      cache == nullptr ||
      !cache->valid() ||
      page_table_device == nullptr ||
      matrix.shape().size() != 2 ||
      matrix.shape()[0] != token_count ||
      matrix.shape()[1] != kv_head_count * head_dim) {
    return false;
  }

  const std::size_t total = token_count * kv_head_count * head_dim;
  const int block_size = 256;
  const int grid_size =
      static_cast<int>((total + static_cast<std::size_t>(block_size) - 1u) / static_cast<std::size_t>(block_size));
  ScatterRowMajorFp32ToPagedCacheBf16Kernel<<<grid_size, block_size>>>(
      matrix.data(),
      sequence_start,
      token_count,
      kv_head_count,
      head_dim,
      tokens_per_page,
      page_table_device,
      max_pages_per_sequence,
      cache->data());
  return CheckCuda(cudaGetLastError());
}

bool RunPagedAttentionDeviceFallback(
    const DeviceTensorBf16& query,
    const DeviceTensorBf16& key_cache,
    const DeviceTensorBf16& value_cache,
    const AttentionKvCacheConfig& cache_config,
    std::size_t batch_size,
    std::size_t max_pages_per_sequence,
    const std::int32_t* page_table_device,
    const std::int32_t* sequence_lengths_device,
    const std::int32_t* query_sequence_lengths_device,
    const std::int32_t* query_sequence_starts_device,
    std::size_t query_head_count,
    std::size_t max_query_tokens,
    float attn_scale,
    bool causal,
    DeviceTensorBf16* output) {
  if (!query.valid() ||
      !key_cache.valid() ||
      !value_cache.valid() ||
      output == nullptr ||
      !output->valid() ||
      page_table_device == nullptr ||
      sequence_lengths_device == nullptr ||
      query_sequence_lengths_device == nullptr ||
      query_sequence_starts_device == nullptr ||
      batch_size == 0 ||
      query_head_count == 0 ||
      max_query_tokens == 0 ||
      cache_config.kv_head_count == 0 ||
      cache_config.head_dim == 0 ||
      max_pages_per_sequence == 0 ||
      query.numel() != batch_size * query_head_count * max_query_tokens * cache_config.head_dim ||
      output->numel() != query.numel()) {
    return false;
  }

  const std::size_t vector_count = batch_size * query_head_count * max_query_tokens;
  const std::size_t max_kv_tokens = max_pages_per_sequence * cache_config.tokens_per_page;
  const std::size_t shared_memory_bytes = max_kv_tokens * sizeof(float);
  PagedAttentionDeviceFallbackKernel<<<
      static_cast<unsigned int>(vector_count),
      kAttentionFallbackBlockSize,
      shared_memory_bytes>>>(
      query.data(),
      key_cache.data(),
      value_cache.data(),
      batch_size,
      query_head_count,
      cache_config.kv_head_count,
      max_query_tokens,
      cache_config.head_dim,
      cache_config.tokens_per_page,
      max_pages_per_sequence,
      page_table_device,
      sequence_lengths_device,
      query_sequence_lengths_device,
      query_sequence_starts_device,
      attn_scale,
      causal,
      output->data());
  return CheckCuda(cudaGetLastError());
}

bool RunPagedAttentionDecodeProduction(
    const DeviceTensorBf16& query,
    const DeviceTensorBf16& key_cache,
    const DeviceTensorBf16& value_cache,
    const AttentionKvCacheConfig& cache_config,
    std::size_t batch_size,
    std::size_t max_pages_per_sequence,
    const std::int32_t* page_table_device,
    const std::int32_t* sequence_lengths_device,
    const std::int32_t* query_sequence_lengths_device,
    const std::int32_t* query_sequence_starts_device,
    std::size_t query_head_count,
    std::size_t max_query_tokens,
    float attn_scale,
    bool causal,
    DeviceTensorBf16* output) {
  if (!query.valid() ||
      !key_cache.valid() ||
      !value_cache.valid() ||
      output == nullptr ||
      !output->valid() ||
      page_table_device == nullptr ||
      sequence_lengths_device == nullptr ||
      query_sequence_lengths_device == nullptr ||
      query_sequence_starts_device == nullptr ||
      batch_size == 0 ||
      query_head_count == 0 ||
      max_query_tokens != 1 ||
      cache_config.kv_head_count == 0 ||
      cache_config.head_dim != static_cast<std::size_t>(kPagedAttentionDecodeHeadDim) ||
      cache_config.tokens_per_page == 0 ||
      max_pages_per_sequence == 0 ||
      (query_head_count % cache_config.kv_head_count) != 0 ||
      (query_head_count / cache_config.kv_head_count) >
          static_cast<std::size_t>(kPagedAttentionDecodeQueryHeadsPerBlock) ||
      query.numel() != batch_size * query_head_count * max_query_tokens * cache_config.head_dim ||
      output->numel() != query.numel()) {
    return false;
  }

  const std::size_t block_count = batch_size * cache_config.kv_head_count;
  PagedAttentionDecodeProductionKernel<<<
      static_cast<unsigned int>(block_count),
      kPagedAttentionDecodeProductionBlockSize>>>(
      query.data(),
      key_cache.data(),
      value_cache.data(),
      batch_size,
      query_head_count,
      cache_config.kv_head_count,
      max_query_tokens,
      cache_config.head_dim,
      cache_config.tokens_per_page,
      max_pages_per_sequence,
      page_table_device,
      sequence_lengths_device,
      query_sequence_lengths_device,
      query_sequence_starts_device,
      attn_scale,
      causal,
      output->data());
  return CheckCuda(cudaGetLastError());
}

bool ConvertAttentionOutputBf16ToRowMajorFp32(
    const DeviceTensorBf16& tensor,
    std::size_t token_count,
    std::size_t query_head_count,
    std::size_t head_dim,
    DeviceTensorFp32* output) {
  if (!tensor.valid() ||
      output == nullptr ||
      !output->valid() ||
      output->shape().size() != 2 ||
      output->shape()[0] != token_count ||
      output->shape()[1] != query_head_count * head_dim ||
      tensor.numel() != token_count * query_head_count * head_dim) {
    return false;
  }

  const std::size_t total = token_count * query_head_count * head_dim;
  const int block_size = 256;
  const int grid_size =
      static_cast<int>((total + static_cast<std::size_t>(block_size) - 1u) / static_cast<std::size_t>(block_size));
  ConvertAttentionOutputBf16ToRowMajorFp32Kernel<<<grid_size, block_size>>>(
      tensor.data(),
      token_count,
      query_head_count,
      head_dim,
      output->data());
  return CheckCuda(cudaGetLastError());
}

}  // namespace nemotron
