#include "nemotron/attention_layout.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>

namespace nemotron {
namespace {

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

float SanitizeFp8Scale(float scale) {
  constexpr float kMinScale = 1.0f / 1024.0f;
  return (!std::isfinite(scale) || scale < kMinScale) ? kMinScale : scale;
}

__global__ void RowMajorMatrixToAttentionBf16Kernel(
    const float* matrix,
    std::size_t token_count,
    std::size_t head_count,
    std::size_t head_dim,
    __nv_bfloat16* output) {
  const std::size_t linear_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = token_count * head_count * head_dim;
  if (linear_index >= total) {
    return;
  }

  const std::size_t dim = linear_index % head_dim;
  const std::size_t tmp = linear_index / head_dim;
  const std::size_t token = tmp % token_count;
  const std::size_t head = tmp / token_count;
  const std::size_t src = token * (head_count * head_dim) + head * head_dim + dim;
  output[linear_index] = __float2bfloat16(matrix[src]);
}

__global__ void RowMajorMatrixToAttentionFp8E4M3Kernel(
    const float* matrix,
    std::size_t token_count,
    std::size_t head_count,
    std::size_t head_dim,
    float scale,
    std::uint8_t* output) {
  const std::size_t linear_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = token_count * head_count * head_dim;
  if (linear_index >= total) {
    return;
  }

  const std::size_t dim = linear_index % head_dim;
  const std::size_t tmp = linear_index / head_dim;
  const std::size_t token = tmp % token_count;
  const std::size_t head = tmp / token_count;
  const std::size_t src = token * (head_count * head_dim) + head * head_dim + dim;
  output[linear_index] = static_cast<std::uint8_t>(
      __nv_cvt_float_to_fp8(matrix[src] / scale, __NV_SATFINITE, __NV_E4M3));
}

__global__ void ScatterRowMajorMatrixToPagedCacheBf16Kernel(
    const float* matrix,
    std::size_t token_count,
    std::size_t start_token_index,
    std::size_t kv_head_count,
    std::size_t head_dim,
    std::size_t tokens_per_page,
    const std::int32_t* page_ids,
    std::size_t page_count,
    __nv_bfloat16* cache) {
  const std::size_t linear_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = token_count * kv_head_count * head_dim;
  if (linear_index >= total) {
    return;
  }

  const std::size_t dim = linear_index % head_dim;
  const std::size_t tmp = linear_index / head_dim;
  const std::size_t head = tmp % kv_head_count;
  const std::size_t token = tmp / kv_head_count;
  const std::size_t absolute_token = start_token_index + token;
  const std::size_t page_slot = absolute_token / tokens_per_page;
  if (page_slot >= page_count) {
    return;
  }
  const std::int32_t page_id = page_ids[page_slot];
  if (page_id < 0) {
    return;
  }
  const std::size_t page_offset = absolute_token % tokens_per_page;
  const std::size_t src = token * (kv_head_count * head_dim) + head * head_dim + dim;
  const std::size_t dst =
      (((static_cast<std::size_t>(page_id) * kv_head_count) + head) * tokens_per_page +
       page_offset) *
          head_dim +
      dim;
  cache[dst] = __float2bfloat16(matrix[src]);
}

__global__ void ScatterKvRowMajorMatricesToPagedCacheBf16Kernel(
    const float* key_matrix,
    const float* value_matrix,
    std::size_t token_count,
    std::size_t start_token_index,
    std::size_t kv_head_count,
    std::size_t head_dim,
    std::size_t tokens_per_page,
    const std::int32_t* page_ids,
    std::size_t page_count,
    __nv_bfloat16* key_cache,
    __nv_bfloat16* value_cache) {
  const std::size_t linear_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = token_count * kv_head_count * head_dim;
  if (linear_index >= total) {
    return;
  }

  const std::size_t dim = linear_index % head_dim;
  const std::size_t tmp = linear_index / head_dim;
  const std::size_t head = tmp % kv_head_count;
  const std::size_t token = tmp / kv_head_count;
  const std::size_t absolute_token = start_token_index + token;
  const std::size_t page_slot = absolute_token / tokens_per_page;
  if (page_slot >= page_count) {
    return;
  }
  const std::int32_t page_id = page_ids[page_slot];
  if (page_id < 0) {
    return;
  }
  const std::size_t page_offset = absolute_token % tokens_per_page;
  const std::size_t src = token * (kv_head_count * head_dim) + head * head_dim + dim;
  const std::size_t dst =
      (((static_cast<std::size_t>(page_id) * kv_head_count) + head) * tokens_per_page +
       page_offset) *
          head_dim +
      dim;
  key_cache[dst] = __float2bfloat16(key_matrix[src]);
  value_cache[dst] = __float2bfloat16(value_matrix[src]);
}

__global__ void ScatterKvRowMajorMatricesToPagedCacheFp8E4M3Kernel(
    const float* key_matrix,
    const float* value_matrix,
    std::size_t token_count,
    std::size_t start_token_index,
    std::size_t kv_head_count,
    std::size_t head_dim,
    std::size_t tokens_per_page,
    float key_scale,
    float value_scale,
    const std::int32_t* page_ids,
    std::size_t page_count,
    std::uint8_t* key_cache,
    std::uint8_t* value_cache) {
  const std::size_t linear_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = token_count * kv_head_count * head_dim;
  if (linear_index >= total) {
    return;
  }

  const std::size_t dim = linear_index % head_dim;
  const std::size_t tmp = linear_index / head_dim;
  const std::size_t head = tmp % kv_head_count;
  const std::size_t token = tmp / kv_head_count;
  const std::size_t absolute_token = start_token_index + token;
  const std::size_t page_slot = absolute_token / tokens_per_page;
  if (page_slot >= page_count) {
    return;
  }
  const std::int32_t page_id = page_ids[page_slot];
  if (page_id < 0) {
    return;
  }
  const std::size_t page_offset = absolute_token % tokens_per_page;
  const std::size_t src = token * (kv_head_count * head_dim) + head * head_dim + dim;
  const std::size_t dst =
      (((static_cast<std::size_t>(page_id) * kv_head_count) + head) * tokens_per_page +
       page_offset) *
          head_dim +
      dim;
  key_cache[dst] = static_cast<std::uint8_t>(
      __nv_cvt_float_to_fp8(key_matrix[src] / key_scale, __NV_SATFINITE, __NV_E4M3));
  value_cache[dst] = static_cast<std::uint8_t>(
      __nv_cvt_float_to_fp8(value_matrix[src] / value_scale, __NV_SATFINITE, __NV_E4M3));
}

__global__ void AttentionBf16ToRowMajorMatrixKernel(
    const __nv_bfloat16* tensor,
    std::size_t token_count,
    std::size_t head_count,
    std::size_t head_dim,
    float* output) {
  const std::size_t linear_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = token_count * head_count * head_dim;
  if (linear_index >= total) {
    return;
  }

  const std::size_t dim = linear_index % head_dim;
  const std::size_t tmp = linear_index / head_dim;
  const std::size_t token = tmp % token_count;
  const std::size_t head = tmp / token_count;
  const std::size_t dst = token * (head_count * head_dim) + head * head_dim + dim;
  output[dst] = __bfloat162float(tensor[linear_index]);
}

bool ValidateAttentionMatrixShape(
    const DeviceTensorFp32& matrix,
    std::size_t token_count,
    std::size_t head_count,
    std::size_t head_dim) {
  return matrix.valid() &&
         matrix.shape().size() == 2 &&
         matrix.shape()[0] == token_count &&
         matrix.shape()[1] == head_count * head_dim;
}

bool ValidateAttentionTensorShape(
    const DeviceTensorBf16& tensor,
    std::size_t token_count,
    std::size_t head_count,
    std::size_t head_dim) {
  return tensor.valid() &&
         tensor.shape().size() == 4 &&
         tensor.shape()[0] == 1 &&
         tensor.shape()[1] == head_count &&
         tensor.shape()[2] == token_count &&
         tensor.shape()[3] == head_dim;
}

bool ValidateAttentionTensorShape(
    const DeviceTensorFp8E4M3& tensor,
    std::size_t token_count,
    std::size_t head_count,
    std::size_t head_dim) {
  return tensor.valid() &&
         tensor.shape().size() == 4 &&
         tensor.shape()[0] == 1 &&
         tensor.shape()[1] == head_count &&
         tensor.shape()[2] == token_count &&
         tensor.shape()[3] == head_dim;
}

}  // namespace

bool ConvertRowMajorMatrixToAttentionBf16(
    const DeviceTensorFp32& matrix,
    std::size_t token_count,
    std::size_t head_count,
    std::size_t head_dim,
    DeviceTensorBf16* output) {
  if (output == nullptr ||
      !ValidateAttentionMatrixShape(matrix, token_count, head_count, head_dim) ||
      !ValidateAttentionTensorShape(*output, token_count, head_count, head_dim)) {
    return false;
  }

  const std::size_t total = token_count * head_count * head_dim;
  constexpr int kBlockSize = 256;
  const int grid_size = static_cast<int>((total + kBlockSize - 1u) / kBlockSize);
  RowMajorMatrixToAttentionBf16Kernel<<<grid_size, kBlockSize>>>(
      matrix.data(),
      token_count,
      head_count,
      head_dim,
      output->data());
  return CheckCuda(cudaGetLastError());
}

bool ConvertRowMajorMatrixToAttentionFp8E4M3(
    const DeviceTensorFp32& matrix,
    std::size_t token_count,
    std::size_t head_count,
    std::size_t head_dim,
    float scale,
    DeviceTensorFp8E4M3* output) {
  if (output == nullptr ||
      !ValidateAttentionMatrixShape(matrix, token_count, head_count, head_dim) ||
      !ValidateAttentionTensorShape(*output, token_count, head_count, head_dim)) {
    return false;
  }

  const std::size_t total = token_count * head_count * head_dim;
  constexpr int kBlockSize = 256;
  const int grid_size = static_cast<int>((total + kBlockSize - 1u) / kBlockSize);
  RowMajorMatrixToAttentionFp8E4M3Kernel<<<grid_size, kBlockSize>>>(
      matrix.data(),
      token_count,
      head_count,
      head_dim,
      SanitizeFp8Scale(scale),
      output->data());
  return CheckCuda(cudaGetLastError());
}

bool ScatterRowMajorMatrixToPagedCacheBf16(
    const DeviceTensorFp32& matrix,
    std::size_t token_count,
    std::size_t start_token_index,
    std::size_t kv_head_count,
    std::size_t head_dim,
    std::size_t tokens_per_page,
    const std::int32_t* page_ids,
    std::size_t page_count,
    DeviceTensorBf16* cache) {
  if (page_ids == nullptr ||
      cache == nullptr ||
      tokens_per_page == 0 ||
      !ValidateAttentionMatrixShape(matrix, token_count, kv_head_count, head_dim) ||
      !cache->valid()) {
    return false;
  }

  const std::size_t total = token_count * kv_head_count * head_dim;
  constexpr int kBlockSize = 256;
  const int grid_size = static_cast<int>((total + kBlockSize - 1u) / kBlockSize);
  ScatterRowMajorMatrixToPagedCacheBf16Kernel<<<grid_size, kBlockSize>>>(
      matrix.data(),
      token_count,
      start_token_index,
      kv_head_count,
      head_dim,
      tokens_per_page,
      page_ids,
      page_count,
      cache->data());
  return CheckCuda(cudaGetLastError());
}

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
    DeviceTensorBf16* value_cache) {
  if (page_ids == nullptr ||
      key_cache == nullptr ||
      value_cache == nullptr ||
      tokens_per_page == 0 ||
      !ValidateAttentionMatrixShape(key_matrix, token_count, kv_head_count, head_dim) ||
      !ValidateAttentionMatrixShape(value_matrix, token_count, kv_head_count, head_dim) ||
      !key_cache->valid() ||
      !value_cache->valid()) {
    return false;
  }

  const std::size_t total = token_count * kv_head_count * head_dim;
  constexpr int kBlockSize = 256;
  const int grid_size = static_cast<int>((total + kBlockSize - 1u) / kBlockSize);
  ScatterKvRowMajorMatricesToPagedCacheBf16Kernel<<<grid_size, kBlockSize>>>(
      key_matrix.data(),
      value_matrix.data(),
      token_count,
      start_token_index,
      kv_head_count,
      head_dim,
      tokens_per_page,
      page_ids,
      page_count,
      key_cache->data(),
      value_cache->data());
  return CheckCuda(cudaGetLastError());
}

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
    DeviceTensorFp8E4M3* value_cache) {
  if (page_ids == nullptr ||
      key_cache == nullptr ||
      value_cache == nullptr ||
      tokens_per_page == 0 ||
      !ValidateAttentionMatrixShape(key_matrix, token_count, kv_head_count, head_dim) ||
      !ValidateAttentionMatrixShape(value_matrix, token_count, kv_head_count, head_dim) ||
      !key_cache->valid() ||
      !value_cache->valid()) {
    return false;
  }

  const std::size_t total = token_count * kv_head_count * head_dim;
  constexpr int kBlockSize = 256;
  const int grid_size = static_cast<int>((total + kBlockSize - 1u) / kBlockSize);
  ScatterKvRowMajorMatricesToPagedCacheFp8E4M3Kernel<<<grid_size, kBlockSize>>>(
      key_matrix.data(),
      value_matrix.data(),
      token_count,
      start_token_index,
      kv_head_count,
      head_dim,
      tokens_per_page,
      SanitizeFp8Scale(key_scale),
      SanitizeFp8Scale(value_scale),
      page_ids,
      page_count,
      key_cache->data(),
      value_cache->data());
  return CheckCuda(cudaGetLastError());
}

bool ConvertAttentionBf16ToRowMajorMatrix(
    const DeviceTensorBf16& tensor,
    std::size_t token_count,
    std::size_t head_count,
    std::size_t head_dim,
    DeviceTensorFp32* output) {
  if (output == nullptr ||
      !ValidateAttentionTensorShape(tensor, token_count, head_count, head_dim) ||
      !ValidateAttentionMatrixShape(*output, token_count, head_count, head_dim)) {
    return false;
  }

  const std::size_t total = token_count * head_count * head_dim;
  constexpr int kBlockSize = 256;
  const int grid_size = static_cast<int>((total + kBlockSize - 1u) / kBlockSize);
  AttentionBf16ToRowMajorMatrixKernel<<<grid_size, kBlockSize>>>(
      tensor.data(),
      token_count,
      head_count,
      head_dim,
      output->data());
  return CheckCuda(cudaGetLastError());
}

}  // namespace nemotron
