#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "nemotron/device_tensor.h"

namespace nemotron {

bool ConvertRowMajorMatrixToAttentionBf16(
    const DeviceTensorFp32& matrix,
    std::size_t token_count,
    std::size_t head_count,
    std::size_t head_dim,
    DeviceTensorBf16* output,
    cudaStream_t stream = nullptr);

bool ConvertRowMajorMatrixToAttentionFp8E4M3(
    const DeviceTensorFp32& matrix,
    std::size_t token_count,
    std::size_t head_count,
    std::size_t head_dim,
    float scale,
    DeviceTensorFp8E4M3* output,
    cudaStream_t stream = nullptr);

bool ScatterRowMajorMatrixToPagedCacheBf16(
    const DeviceTensorFp32& matrix,
    std::size_t token_count,
    std::size_t start_token_index,
    std::size_t kv_head_count,
    std::size_t head_dim,
    std::size_t tokens_per_page,
    const std::int32_t* page_ids,
    std::size_t page_count,
    DeviceTensorBf16* cache,
    cudaStream_t stream = nullptr);

bool ScatterKvRowMajorMatricesToPagedCacheBf16(
    const DeviceTensorFp32& key_matrix,
    const DeviceTensorFp32& value_matrix,
    std::size_t token_count,
    std::size_t start_token_index,
    std::size_t kv_head_count,
    std::size_t head_dim,
    std::size_t tokens_per_page,
    const std::int32_t* page_ids,
    std::size_t page_count,
    DeviceTensorBf16* key_cache,
    DeviceTensorBf16* value_cache,
    cudaStream_t stream = nullptr);

bool ScatterKvRowMajorMatricesToPagedCacheFp8E4M3(
    const DeviceTensorFp32& key_matrix,
    const DeviceTensorFp32& value_matrix,
    std::size_t token_count,
    std::size_t start_token_index,
    std::size_t kv_head_count,
    std::size_t head_dim,
    std::size_t tokens_per_page,
    float key_scale,
    float value_scale,
    const std::int32_t* page_ids,
    std::size_t page_count,
    DeviceTensorFp8E4M3* key_cache,
    DeviceTensorFp8E4M3* value_cache,
    cudaStream_t stream = nullptr);

bool ConvertAttentionBf16ToRowMajorMatrix(
    const DeviceTensorBf16& tensor,
    std::size_t token_count,
    std::size_t head_count,
    std::size_t head_dim,
    DeviceTensorFp32* output,
    cudaStream_t stream = nullptr);

}  // namespace nemotron
