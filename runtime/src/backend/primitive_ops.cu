#include "nemotron/primitive_ops.h"

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

__global__ void Relu2InPlaceKernel(
    float* data,
    std::size_t count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const float value = data[index];
  data[index] = value > 0.0f ? (value * value) : 0.0f;
}

__global__ void AccumulateScaledKernel(
    const float* input,
    float scale,
    float* output,
    std::size_t count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  output[index] += input[index] * scale;
}

__global__ void GatherRowsKernel(
    const float* input,
    const int* row_indices,
    float* output,
    std::size_t output_rows,
    std::size_t input_rows,
    std::size_t cols) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = output_rows * cols;
  if (index >= count) {
    return;
  }

  const std::size_t row = index / cols;
  const std::size_t col = index % cols;
  const int input_row = row_indices[row];
  if (input_row < 0 || static_cast<std::size_t>(input_row) >= input_rows) {
    output[index] = 0.0f;
    return;
  }
  output[index] = input[static_cast<std::size_t>(input_row) * cols + col];
}

__global__ void ScatterAddWeightedRowsKernel(
    const float* input,
    const int* row_indices,
    const float* row_weights,
    float* output,
    std::size_t input_rows,
    std::size_t output_rows,
    std::size_t cols) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = input_rows * cols;
  if (index >= count) {
    return;
  }

  const std::size_t row = index / cols;
  const std::size_t col = index % cols;
  const int output_row = row_indices[row];
  if (output_row < 0 || static_cast<std::size_t>(output_row) >= output_rows) {
    return;
  }
  output[static_cast<std::size_t>(output_row) * cols + col] +=
      input[index] * row_weights[row];
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

bool Relu2InPlaceFp32(DeviceTensorFp32* tensor) {
  if (tensor == nullptr || !tensor->valid()) {
    return false;
  }

  const std::size_t count = tensor->numel();
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1) / block.x));
  Relu2InPlaceKernel<<<grid, block>>>(tensor->data(), count);
  return CheckCuda(cudaGetLastError());
}

bool AccumulateScaledFp32(
    const DeviceTensorFp32& input,
    float scale,
    DeviceTensorFp32* output) {
  if (output == nullptr || !input.valid() || !output->valid() || input.shape() != output->shape()) {
    return false;
  }

  const std::size_t count = input.numel();
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1) / block.x));
  AccumulateScaledKernel<<<grid, block>>>(input.data(), scale, output->data(), count);
  return CheckCuda(cudaGetLastError());
}

bool GatherRowsFp32(
    const DeviceTensorFp32& input,
    const int* row_indices_device,
    DeviceTensorFp32* output) {
  if (output == nullptr ||
      !input.valid() ||
      !output->valid() ||
      row_indices_device == nullptr ||
      input.shape().size() != 2 ||
      output->shape().size() != 2 ||
      input.shape()[1] != output->shape()[1]) {
    return false;
  }

  const std::size_t output_rows = output->shape()[0];
  const std::size_t input_rows = input.shape()[0];
  const std::size_t cols = input.shape()[1];
  const std::size_t count = output->numel();
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1) / block.x));
  GatherRowsKernel<<<grid, block>>>(
      input.data(),
      row_indices_device,
      output->data(),
      output_rows,
      input_rows,
      cols);
  return CheckCuda(cudaGetLastError());
}

bool ScatterAddWeightedRowsFp32(
    const DeviceTensorFp32& input,
    const int* row_indices_device,
    const float* row_weights_device,
    DeviceTensorFp32* output) {
  if (output == nullptr ||
      !input.valid() ||
      !output->valid() ||
      row_indices_device == nullptr ||
      row_weights_device == nullptr ||
      input.shape().size() != 2 ||
      output->shape().size() != 2 ||
      input.shape()[1] != output->shape()[1]) {
    return false;
  }

  const std::size_t input_rows = input.shape()[0];
  const std::size_t output_rows = output->shape()[0];
  const std::size_t cols = input.shape()[1];
  const std::size_t count = input.numel();
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1) / block.x));
  ScatterAddWeightedRowsKernel<<<grid, block>>>(
      input.data(),
      row_indices_device,
      row_weights_device,
      output->data(),
      input_rows,
      output_rows,
      cols);
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

}  // namespace nemotron
