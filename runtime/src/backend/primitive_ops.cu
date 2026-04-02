#include "nemotron/primitive_ops.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace nemotron {
namespace {

constexpr int kThreadsPerBlock = 256;

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

__global__ void ResidualAddKernel(
    const float* lhs,
    const float* rhs,
    float* output,
    std::size_t count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  output[index] = lhs[index] + rhs[index];
}

__global__ void ResidualAddBf16Kernel(
    const __nv_bfloat16* lhs,
    const __nv_bfloat16* rhs,
    __nv_bfloat16* output,
    std::size_t count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const float lhs_value = __bfloat162float(lhs[index]);
  const float rhs_value = __bfloat162float(rhs[index]);
  output[index] = __float2bfloat16(lhs_value + rhs_value);
}

__global__ void RmsNormKernel(
    const float* input,
    const float* weight,
    float* output,
    std::size_t rows,
    std::size_t hidden_size,
    float epsilon) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  __shared__ float shared_sum[kThreadsPerBlock];
  float local_sum = 0.0f;
  const std::size_t row_offset = row * hidden_size;
  for (std::size_t column = threadIdx.x; column < hidden_size; column += blockDim.x) {
    const float value = input[row_offset + column];
    local_sum += value * value;
  }
  shared_sum[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv_rms = rsqrtf((shared_sum[0] / static_cast<float>(hidden_size)) + epsilon);
  for (std::size_t column = threadIdx.x; column < hidden_size; column += blockDim.x) {
    output[row_offset + column] = input[row_offset + column] * inv_rms * weight[column];
  }
}

__device__ float LoadNormValue(const float* input, std::size_t index) {
  return input[index];
}

__device__ float LoadNormValue(const __nv_bfloat16* input, std::size_t index) {
  return __bfloat162float(input[index]);
}

__device__ void StoreNormValue(float* output, std::size_t index, float value) {
  output[index] = value;
}

__device__ void StoreNormValue(__nv_bfloat16* output, std::size_t index, float value) {
  output[index] = __float2bfloat16(value);
}

template <typename InputT, typename OutputT>
__global__ void RmsNormTypedKernel(
    const InputT* input,
    const float* weight,
    OutputT* output,
    std::size_t rows,
    std::size_t hidden_size,
    float epsilon) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  __shared__ float shared_sum[kThreadsPerBlock];
  float local_sum = 0.0f;
  const std::size_t row_offset = row * hidden_size;
  for (std::size_t column = threadIdx.x; column < hidden_size; column += blockDim.x) {
    const float value = LoadNormValue(input, row_offset + column);
    local_sum += value * value;
  }
  shared_sum[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float inv_rms = rsqrtf((shared_sum[0] / static_cast<float>(hidden_size)) + epsilon);
  for (std::size_t column = threadIdx.x; column < hidden_size; column += blockDim.x) {
    const float value = LoadNormValue(input, row_offset + column) * inv_rms * weight[column];
    StoreNormValue(output, row_offset + column, value);
  }
}

bool HasCompatibleMatrixShape(
    const DeviceTensorFp32& input,
    const DeviceTensorFp32& output,
    std::size_t* rows,
    std::size_t* hidden_size) {
  if (!input.valid() || !output.valid() || input.shape().size() != 2 || output.shape() != input.shape()) {
    return false;
  }
  *rows = input.shape()[0];
  *hidden_size = input.shape()[1];
  return *rows != 0 && *hidden_size != 0;
}

bool HasCompatibleMatrixShape(
    const DeviceTensorBf16& input,
    const DeviceTensorBf16& output,
    std::size_t* rows,
    std::size_t* hidden_size) {
  if (!input.valid() || !output.valid() || input.shape().size() != 2 || output.shape() != input.shape()) {
    return false;
  }
  *rows = input.shape()[0];
  *hidden_size = input.shape()[1];
  return *rows != 0 && *hidden_size != 0;
}

}  // namespace

bool ResidualAddFp32(
    const DeviceTensorFp32& lhs,
    const DeviceTensorFp32& rhs,
    DeviceTensorFp32* output) {
  if (output == nullptr || !lhs.valid() || !rhs.valid() || !output->valid() ||
      lhs.shape() != rhs.shape() || lhs.shape() != output->shape()) {
    return false;
  }

  const std::size_t count = lhs.numel();
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1) / block.x));
  ResidualAddKernel<<<grid, block>>>(lhs.data(), rhs.data(), output->data(), count);
  return CheckCuda(cudaGetLastError());
}

bool ResidualAddBf16(
    const DeviceTensorBf16& lhs,
    const DeviceTensorBf16& rhs,
    DeviceTensorBf16* output) {
  if (output == nullptr || !lhs.valid() || !rhs.valid() || !output->valid() ||
      lhs.shape() != rhs.shape() || lhs.shape() != output->shape()) {
    return false;
  }

  const std::size_t count = lhs.numel();
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1) / block.x));
  ResidualAddBf16Kernel<<<grid, block>>>(lhs.data(), rhs.data(), output->data(), count);
  return CheckCuda(cudaGetLastError());
}

bool RmsNormFp32(
    const DeviceTensorFp32& input,
    const DeviceTensorFp32& weight,
    float epsilon,
    DeviceTensorFp32* output) {
  std::size_t rows = 0;
  std::size_t hidden_size = 0;
  if (output == nullptr || !HasCompatibleMatrixShape(input, *output, &rows, &hidden_size) ||
      !weight.valid() || weight.shape().size() != 1 || weight.shape()[0] != hidden_size || epsilon <= 0.0f) {
    return false;
  }

  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(rows));
  RmsNormKernel<<<grid, block>>>(input.data(), weight.data(), output->data(), rows, hidden_size, epsilon);
  return CheckCuda(cudaGetLastError());
}

bool RmsNormBf16(
    const DeviceTensorBf16& input,
    const DeviceTensorFp32& weight,
    float epsilon,
    DeviceTensorBf16* output) {
  std::size_t rows = 0;
  std::size_t hidden_size = 0;
  if (output == nullptr || !HasCompatibleMatrixShape(input, *output, &rows, &hidden_size) ||
      !weight.valid() || weight.shape().size() != 1 || weight.shape()[0] != hidden_size || epsilon <= 0.0f) {
    return false;
  }

  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(rows));
  RmsNormTypedKernel<<<grid, block>>>(
      input.data(),
      weight.data(),
      output->data(),
      rows,
      hidden_size,
      epsilon);
  return CheckCuda(cudaGetLastError());
}

}  // namespace nemotron
