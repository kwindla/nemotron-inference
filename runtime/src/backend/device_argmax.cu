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
constexpr std::size_t kNanoVocabSize = 131072;

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

template <std::size_t kStaticVocabSize>
__global__ void DeviceArgmaxKernel(
    const float* logits,
    std::size_t vocab_size,
    std::int32_t* token_id) {
  __shared__ float shared_values[kThreadsPerBlock];
  __shared__ std::int32_t shared_indices[kThreadsPerBlock];

  float local_best_value = -FLT_MAX;
  std::int32_t local_best_index = kInvalidTokenIndex;

  const std::size_t effective_vocab_size =
      kStaticVocabSize == 0 ? vocab_size : kStaticVocabSize;
  for (std::size_t index = static_cast<std::size_t>(threadIdx.x);
       index < effective_vocab_size;
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

bool LaunchDeviceArgmax(
    const float* logits,
    std::size_t vocab_size,
    std::int32_t* device_token_id) {
  if (logits == nullptr || device_token_id == nullptr || vocab_size == 0) {
    return false;
  }
  if (vocab_size == kNanoVocabSize) {
    DeviceArgmaxKernel<kNanoVocabSize><<<1, kThreadsPerBlock>>>(
        logits,
        vocab_size,
        device_token_id);
  } else {
    DeviceArgmaxKernel<0><<<1, kThreadsPerBlock>>>(
        logits,
        vocab_size,
        device_token_id);
  }
  return CheckCuda(cudaGetLastError());
}

bool ValidateLogitsMatrix(
    const DeviceTensorFp32& logits,
    std::size_t row_index) {
  return logits.valid() &&
         logits.shape().size() == 2 &&
         row_index < logits.shape()[0] &&
         logits.shape()[1] != 0 &&
         logits.shape()[1] <=
             static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
}

}  // namespace

bool DeviceArgmax(const DeviceTensorFp32& logits_row, std::int32_t* device_token_id) {
  if (!ValidateLogitsMatrix(logits_row, 0) ||
      device_token_id == nullptr ||
      logits_row.shape()[0] != 1) {
    return false;
  }

  const std::size_t vocab_size = logits_row.shape()[1];
  if (logits_row.numel() != vocab_size) {
    return false;
  }

  return LaunchDeviceArgmax(logits_row.data(), vocab_size, device_token_id);
}

bool DeviceArgmaxLastRow(const DeviceTensorFp32& logits, std::int32_t* device_token_id) {
  if (!ValidateLogitsMatrix(logits, logits.shape().empty() ? 0 : logits.shape()[0] - 1) ||
      device_token_id == nullptr) {
    return false;
  }

  const std::size_t vocab_size = logits.shape()[1];
  const std::size_t row_index = logits.shape()[0] - 1;
  const float* row_data = logits.data() + (row_index * vocab_size);
  return LaunchDeviceArgmax(row_data, vocab_size, device_token_id);
}

}  // namespace nemotron
