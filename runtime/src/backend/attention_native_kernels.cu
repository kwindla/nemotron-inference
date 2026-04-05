#include "nemotron/attention_native_kernels.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>

namespace nemotron {
namespace {

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

constexpr int kPagedAttentionMultiTokenProductionBlockSize = 256;
constexpr int kPagedAttentionMultiTokenWarpSize = 32;
constexpr int kPagedAttentionMultiTokenWarpsPerBlock =
    kPagedAttentionMultiTokenProductionBlockSize / kPagedAttentionMultiTokenWarpSize;
constexpr int kPagedAttentionMultiTokenQueryHeadsPerWarp = 2;
constexpr int kPagedAttentionMultiTokenQueryHeadsPerBlock =
    kPagedAttentionMultiTokenWarpsPerBlock * kPagedAttentionMultiTokenQueryHeadsPerWarp;
constexpr int kPagedAttentionMultiTokenHeadDim = 128;
constexpr int kPagedAttentionMultiTokenDimsPerLane =
    kPagedAttentionMultiTokenHeadDim / kPagedAttentionMultiTokenWarpSize;
constexpr unsigned int kPagedAttentionMultiTokenFullMask = 0xffffffffu;
constexpr std::size_t kPagedAttentionMultiTokenMaxQueryTokens = 1024;

static_assert(
    kPagedAttentionDecodeProductionBlockSize == 2 * kPagedAttentionDecodeHeadDim,
    "Decode block shape assumes one thread per staged K/V element.");
static_assert(
    kPagedAttentionDecodeHeadDim % kPagedAttentionDecodeWarpSize == 0,
    "Decode kernel requires a whole-number dim shard per lane.");
static_assert(
    kPagedAttentionMultiTokenProductionBlockSize ==
        2 * kPagedAttentionMultiTokenHeadDim,
    "Multi-token block shape assumes one thread per staged K/V element.");
static_assert(
    kPagedAttentionMultiTokenHeadDim % kPagedAttentionMultiTokenWarpSize == 0,
    "Multi-token kernel requires a whole-number dim shard per lane.");

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

__device__ inline float WarpReduceSum(float value, unsigned int mask) {
  for (int offset = kPagedAttentionDecodeWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(mask, value, offset);
  }
  return value;
}

__global__ void ConvertRowMajorBf16ToAttentionQueryBf16Kernel(
    const __nv_bfloat16* input,
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
    output[dst] = input[i];
  }
}

__global__ void ConvertAttentionOutputBf16ToRowMajorBf16Kernel(
    const __nv_bfloat16* input,
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
    const std::size_t src = ((head * token_count) + token) * head_dim + dim;
    output[i] = input[src];
  }
}

__global__ void ScatterRowMajorBf16ToPagedCacheBf16Kernel(
    const __nv_bfloat16* matrix,
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
        ((((static_cast<std::size_t>(page_id) * kv_head_count) + head) * tokens_per_page) +
         page_offset) *
            head_dim +
        dim;
    cache[dst] = matrix[i];
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
        const std::int32_t page_id = page_table[batch * max_pages_per_sequence + page_slot];
        if (page_id >= 0) {
          shared_kv_base =
              ((((static_cast<std::size_t>(page_id) * kv_head_count) + kv_head) * tokens_per_page) +
               page_offset) *
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
        score = WarpReduceSum(score, kPagedAttentionDecodeFullMask);
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

// One block owns a single (batch, kv_head, query_token) slice, stages each
// visible K/V vector once in shared memory, and computes all query heads in
// that GQA group together.
__global__ void PagedAttentionNanoMultiTokenKernel(
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
  __shared__ float shared_key[kPagedAttentionMultiTokenHeadDim];
  __shared__ float shared_value[kPagedAttentionMultiTokenHeadDim];
  __shared__ std::size_t shared_kv_base;
  __shared__ int shared_kv_valid;

  const std::size_t block_index = static_cast<std::size_t>(blockIdx.x);
  const std::size_t batch = block_index / (kv_head_count * max_query_tokens);
  if (batch >= batch_size) {
    return;
  }

  const std::size_t batch_rem = block_index % (kv_head_count * max_query_tokens);
  const std::size_t kv_head = batch_rem / max_query_tokens;
  const std::size_t q_token = batch_rem % max_query_tokens;

  const std::size_t q_tokens = static_cast<std::size_t>(query_sequence_lengths[batch]);
  if (q_token >= q_tokens) {
    return;
  }

  const std::size_t q_start = static_cast<std::size_t>(query_sequence_starts[batch]);
  const std::size_t kv_tokens = static_cast<std::size_t>(sequence_lengths[batch]);
  const std::size_t visible_kv_tokens =
      causal ? min(kv_tokens, q_start + q_token + 1u) : kv_tokens;
  const std::size_t queries_per_kv_head = query_head_count / kv_head_count;
  const std::size_t query_head_group_start = kv_head * queries_per_kv_head;
  const std::size_t query_head_group_end = query_head_group_start + queries_per_kv_head;

  const std::size_t warp =
      static_cast<std::size_t>(threadIdx.x) / kPagedAttentionMultiTokenWarpSize;
  const std::size_t lane =
      static_cast<std::size_t>(threadIdx.x) % kPagedAttentionMultiTokenWarpSize;
  const std::size_t dim_base = lane * kPagedAttentionMultiTokenDimsPerLane;
  const std::size_t warp_query_head_start =
      query_head_group_start + (warp * kPagedAttentionMultiTokenQueryHeadsPerWarp);

  const std::size_t query_heads[kPagedAttentionMultiTokenQueryHeadsPerWarp] = {
      warp_query_head_start,
      warp_query_head_start + 1u};
  const bool head_valid[kPagedAttentionMultiTokenQueryHeadsPerWarp] = {
      query_heads[0] < query_head_group_end,
      query_heads[1] < query_head_group_end};

  float query_frag[kPagedAttentionMultiTokenQueryHeadsPerWarp]
                  [kPagedAttentionMultiTokenDimsPerLane] = {};
  float output_frag[kPagedAttentionMultiTokenQueryHeadsPerWarp]
                   [kPagedAttentionMultiTokenDimsPerLane] = {};
  float max_score[kPagedAttentionMultiTokenQueryHeadsPerWarp] = {-INFINITY, -INFINITY};
  float sum_exp[kPagedAttentionMultiTokenQueryHeadsPerWarp] = {0.0f, 0.0f};

  for (int head_slot = 0; head_slot < kPagedAttentionMultiTokenQueryHeadsPerWarp; ++head_slot) {
    if (!head_valid[head_slot]) {
      continue;
    }
    const std::size_t q_base =
        (((batch * query_head_count) + query_heads[head_slot]) * max_query_tokens + q_token) *
        head_dim;
    for (int dim_offset = 0; dim_offset < kPagedAttentionMultiTokenDimsPerLane; ++dim_offset) {
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
        const std::int32_t page_id = page_table[batch * max_pages_per_sequence + page_slot];
        if (page_id >= 0) {
          shared_kv_base =
              ((((static_cast<std::size_t>(page_id) * kv_head_count) + kv_head) * tokens_per_page) +
               page_offset) *
              head_dim;
          shared_kv_valid = 1;
        }
      }
    }
    __syncthreads();

    if (shared_kv_valid != 0) {
      if (threadIdx.x < kPagedAttentionMultiTokenHeadDim) {
        shared_key[threadIdx.x] =
            __bfloat162float(key_cache[shared_kv_base + static_cast<std::size_t>(threadIdx.x)]);
      } else {
        const int value_dim = threadIdx.x - kPagedAttentionMultiTokenHeadDim;
        shared_value[value_dim] =
            __bfloat162float(value_cache[shared_kv_base + static_cast<std::size_t>(value_dim)]);
      }
      __syncthreads();

      float key_frag[kPagedAttentionMultiTokenDimsPerLane];
      float value_frag[kPagedAttentionMultiTokenDimsPerLane];
      for (int dim_offset = 0; dim_offset < kPagedAttentionMultiTokenDimsPerLane; ++dim_offset) {
        const std::size_t dim = dim_base + static_cast<std::size_t>(dim_offset);
        key_frag[dim_offset] = shared_key[dim];
        value_frag[dim_offset] = shared_value[dim];
      }

      for (int head_slot = 0; head_slot < kPagedAttentionMultiTokenQueryHeadsPerWarp; ++head_slot) {
        if (!head_valid[head_slot]) {
          continue;
        }

        float score = 0.0f;
        for (int dim_offset = 0; dim_offset < kPagedAttentionMultiTokenDimsPerLane; ++dim_offset) {
          score = __fmaf_rn(query_frag[head_slot][dim_offset], key_frag[dim_offset], score);
        }
        score = WarpReduceSum(score, kPagedAttentionMultiTokenFullMask);
        score = __shfl_sync(kPagedAttentionMultiTokenFullMask, score, 0);

        const float scaled_score = __fmul_rn(score, attn_scale);
        const float new_max = fmaxf(max_score[head_slot], scaled_score);
        const float correction = expf(max_score[head_slot] - new_max);
        const float weight = expf(scaled_score - new_max);
        for (int dim_offset = 0; dim_offset < kPagedAttentionMultiTokenDimsPerLane; ++dim_offset) {
          output_frag[head_slot][dim_offset] =
              __fmaf_rn(
                  weight,
                  value_frag[dim_offset],
                  output_frag[head_slot][dim_offset] * correction);
        }
        sum_exp[head_slot] = __fmaf_rn(sum_exp[head_slot], correction, weight);
        max_score[head_slot] = new_max;
      }
    }

    __syncthreads();
  }

  for (int head_slot = 0; head_slot < kPagedAttentionMultiTokenQueryHeadsPerWarp; ++head_slot) {
    if (!head_valid[head_slot]) {
      continue;
    }
    const float inv_sum = sum_exp[head_slot] > 0.0f ? (1.0f / sum_exp[head_slot]) : 0.0f;
    const std::size_t q_base =
        (((batch * query_head_count) + query_heads[head_slot]) * max_query_tokens + q_token) *
        head_dim;
    for (int dim_offset = 0; dim_offset < kPagedAttentionMultiTokenDimsPerLane; ++dim_offset) {
      output[q_base + dim_base + static_cast<std::size_t>(dim_offset)] =
          __float2bfloat16(output_frag[head_slot][dim_offset] * inv_sum);
    }
  }
}

}  // namespace

bool ConvertRowMajorBf16ToAttentionQueryBf16(
    const DeviceTensorBf16& matrix,
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
      static_cast<int>((total + static_cast<std::size_t>(block_size) - 1u) /
                       static_cast<std::size_t>(block_size));
  ConvertRowMajorBf16ToAttentionQueryBf16Kernel<<<grid_size, block_size>>>(
      matrix.data(),
      token_count,
      query_head_count,
      head_dim,
      output->data());
  return CheckCuda(cudaGetLastError());
}

bool ScatterRowMajorBf16ToPagedCacheBf16(
    const DeviceTensorBf16& matrix,
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
      static_cast<int>((total + static_cast<std::size_t>(block_size) - 1u) /
                       static_cast<std::size_t>(block_size));
  ScatterRowMajorBf16ToPagedCacheBf16Kernel<<<grid_size, block_size>>>(
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

bool RunPagedAttentionNanoMultiToken(
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
      max_query_tokens <= 1 ||
      max_query_tokens > kPagedAttentionMultiTokenMaxQueryTokens ||
      cache_config.kv_head_count == 0 ||
      cache_config.head_dim != static_cast<std::size_t>(kPagedAttentionMultiTokenHeadDim) ||
      cache_config.tokens_per_page == 0 ||
      max_pages_per_sequence == 0 ||
      (query_head_count % cache_config.kv_head_count) != 0 ||
      (query_head_count / cache_config.kv_head_count) >
          static_cast<std::size_t>(kPagedAttentionMultiTokenQueryHeadsPerBlock) ||
      query.numel() != batch_size * query_head_count * max_query_tokens * cache_config.head_dim ||
      output->numel() != query.numel()) {
    return false;
  }

  const std::size_t block_count = batch_size * cache_config.kv_head_count * max_query_tokens;
  PagedAttentionNanoMultiTokenKernel<<<
      static_cast<unsigned int>(block_count),
      kPagedAttentionMultiTokenProductionBlockSize>>>(
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

bool ConvertAttentionOutputBf16ToRowMajorBf16(
    const DeviceTensorBf16& tensor,
    std::size_t token_count,
    std::size_t query_head_count,
    std::size_t head_dim,
    DeviceTensorBf16* output) {
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
      static_cast<int>((total + static_cast<std::size_t>(block_size) - 1u) /
                       static_cast<std::size_t>(block_size));
  ConvertAttentionOutputBf16ToRowMajorBf16Kernel<<<grid_size, block_size>>>(
      tensor.data(),
      token_count,
      query_head_count,
      head_dim,
      output->data());
  return CheckCuda(cudaGetLastError());
}

}  // namespace nemotron
