#include "nemotron/dense_gemm_runner.h"

#include <cuda_runtime.h>

namespace nemotron {
namespace {

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

__global__ void DenseRowMajorReferenceKernel(
    const float* activations,
    const float* weights,
    std::size_t rows,
    std::size_t cols,
    std::size_t input_cols,
    float* output) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  const std::size_t col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows || col >= cols) {
    return;
  }

  // Reference fallback: keep accumulation numerically stable so unsupported
  // cuBLASLt shapes do not reintroduce the host-side drift we already removed.
  double accum = 0.0;
  const std::size_t activation_offset = row * input_cols;
  const std::size_t weight_offset = col * input_cols;
  for (std::size_t k = 0; k < input_cols; ++k) {
    accum += static_cast<double>(activations[activation_offset + k]) *
             static_cast<double>(weights[weight_offset + k]);
  }
  output[row * cols + col] = static_cast<float>(accum);
}

}  // namespace

std::optional<DenseRowMajorDeviceStats> RunDenseRowMajorFp32ReferenceToDevice(
    const DeviceDenseWeightFp32& weights,
    const DeviceTensorFp32& activations,
    DeviceTensorFp32* output) {
  if (!weights.valid() || !activations.valid() || output == nullptr || !output->valid()) {
    return std::nullopt;
  }
  if (activations.shape().size() != 2 || output->shape().size() != 2) {
    return std::nullopt;
  }

  const std::size_t rows = activations.shape()[0];
  const std::size_t input_cols = activations.shape()[1];
  const std::size_t cols = weights.output_rows();
  if (rows == 0 ||
      input_cols == 0 ||
      cols == 0 ||
      weights.input_cols() != input_cols ||
      output->shape()[0] != rows ||
      output->shape()[1] != cols) {
    return std::nullopt;
  }

  constexpr dim3 kBlockDim(16, 16);
  const dim3 grid_dim(
      static_cast<unsigned int>((cols + kBlockDim.x - 1u) / kBlockDim.x),
      static_cast<unsigned int>((rows + kBlockDim.y - 1u) / kBlockDim.y));
  DenseRowMajorReferenceKernel<<<grid_dim, kBlockDim>>>(
      activations.data(),
      weights.data(),
      rows,
      cols,
      input_cols,
      output->data());
  if (!CheckCuda(cudaGetLastError())) {
    return std::nullopt;
  }

  return DenseRowMajorDeviceStats{
      rows,
      cols,
      0,
      0,
  };
}

}  // namespace nemotron
