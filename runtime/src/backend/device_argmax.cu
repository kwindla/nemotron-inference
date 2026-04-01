#include "nemotron/device_argmax.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <limits>

namespace nemotron {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr std::int32_t kInvalidTokenIndex = INT32_MAX;

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

__device__ bool IsBetterCandidate(
    float candidate_value,
    std::int32_t candidate_index,
    float best_value,
    std::int32_t best_index) {
  return candidate_value > best_value ||
         (candidate_value == best_value && candidate_index < best_index);
}

__global__ void DeviceArgmaxKernel(
    const float* logits,
    std::size_t vocab_size,
    std::int32_t* token_id) {
  __shared__ float shared_values[kThreadsPerBlock];
  __shared__ std::int32_t shared_indices[kThreadsPerBlock];

  float local_best_value = -FLT_MAX;
  std::int32_t local_best_index = kInvalidTokenIndex;

  for (std::size_t index = static_cast<std::size_t>(threadIdx.x);
       index < vocab_size;
       index += blockDim.x) {
    const float value = logits[index];
    const std::int32_t candidate_index = static_cast<std::int32_t>(index);
    if (IsBetterCandidate(value, candidate_index, local_best_value, local_best_index)) {
      local_best_value = value;
      local_best_index = candidate_index;
    }
  }

  shared_values[threadIdx.x] = local_best_value;
  shared_indices[threadIdx.x] = local_best_index;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      const float candidate_value = shared_values[threadIdx.x + stride];
      const std::int32_t candidate_index = shared_indices[threadIdx.x + stride];
      if (IsBetterCandidate(
              candidate_value,
              candidate_index,
              shared_values[threadIdx.x],
              shared_indices[threadIdx.x])) {
        shared_values[threadIdx.x] = candidate_value;
        shared_indices[threadIdx.x] = candidate_index;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *token_id = shared_indices[0];
  }
}

}  // namespace

bool DeviceArgmax(const DeviceTensorFp32& logits_row, std::int32_t* device_token_id) {
  if (!logits_row.valid() || device_token_id == nullptr || logits_row.shape().size() != 2 ||
      logits_row.shape()[0] != 1) {
    return false;
  }

  const std::size_t vocab_size = logits_row.shape()[1];
  if (vocab_size == 0 || logits_row.numel() != vocab_size ||
      vocab_size > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    return false;
  }

  DeviceArgmaxKernel<<<1, kThreadsPerBlock>>>(logits_row.data(), vocab_size, device_token_id);
  return CheckCuda(cudaGetLastError()) && CheckCuda(cudaDeviceSynchronize());
}

}  // namespace nemotron
