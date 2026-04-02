#pragma once

#include <cstddef>
#include <cstdint>

#include "nemotron/device_tensor.h"
#include "nemotron/paged_attention_plan.h"

namespace nemotron {

bool ConvertRowMajorFp32ToAttentionQueryBf16(
    const DeviceTensorFp32& matrix,
    std::size_t token_count,
    std::size_t query_head_count,
    std::size_t head_dim,
    DeviceTensorBf16* output);

bool ScatterRowMajorFp32ToPagedCacheBf16(
    const DeviceTensorFp32& matrix,
    std::size_t sequence_start,
    std::size_t token_count,
    std::size_t kv_head_count,
    std::size_t head_dim,
    std::size_t tokens_per_page,
    const std::int32_t* page_table_device,
    std::size_t max_pages_per_sequence,
    DeviceTensorBf16* cache);

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
    DeviceTensorBf16* output);

bool ConvertAttentionOutputBf16ToRowMajorFp32(
    const DeviceTensorBf16& tensor,
    std::size_t token_count,
    std::size_t query_head_count,
    std::size_t head_dim,
    DeviceTensorFp32* output);

}  // namespace nemotron
