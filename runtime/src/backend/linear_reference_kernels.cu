#include "nemotron/linear_reference_kernels.h"

#include <cuda_runtime.h>

#include "fused_decode_common.cuh"
#include "nemotron/device_nvfp4_matrix.h"

namespace nemotron {
namespace {

struct Nvfp4RowMajorView {
  const std::uint8_t* packed_data = nullptr;
  const std::uint8_t* block_scales_data = nullptr;
  const float* tensor_scale_data = nullptr;
  std::size_t rows = 0;
  std::size_t cols = 0;
};

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

__device__ inline float DecodeNvfp4RowMajorValue(
    const Nvfp4RowMajorView& view,
    std::size_t row,
    std::size_t col) {
  const std::size_t blocks_per_row = view.cols / fused_decode::kNvfp4BlockWidth;
  const std::size_t block = col / fused_decode::kNvfp4BlockWidth;
  const std::size_t offset_in_block = col % fused_decode::kNvfp4BlockWidth;
  const std::size_t packed_row_offset = row * (view.cols / 2);
  const std::size_t packed_index = packed_row_offset + (block * (fused_decode::kNvfp4BlockWidth / 2)) + (offset_in_block / 2);
  const std::uint8_t packed_byte = view.packed_data[packed_index];
  const std::uint8_t nibble =
      (offset_in_block & 1u) == 0u ? (packed_byte & 0x0F) : ((packed_byte >> 4) & 0x0F);
  const float tensor_scale = *view.tensor_scale_data;
  const float block_scale =
      fused_decode::DecodeFp8(view.block_scales_data[row * blocks_per_row + block]) * tensor_scale;
  return fused_decode::DecodeFp4(nibble) * block_scale;
}

__global__ void DenseRowMajorReferenceKernel(
    const float* activations,
    std::size_t rows,
    std::size_t input_cols,
    const float* weights,
    std::size_t output_rows,
    float* output) {
  const std::size_t out_col = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const std::size_t row = (static_cast<std::size_t>(blockIdx.y) * blockDim.y) + threadIdx.y;
  if (row >= rows || out_col >= output_rows) {
    return;
  }

  float accum = 0.0f;
  for (std::size_t col = 0; col < input_cols; ++col) {
    accum += activations[row * input_cols + col] *
             weights[out_col * input_cols + col];
  }
  output[row * output_rows + out_col] = accum;
}

__global__ void DenseRowMajorHighPrecisionReferenceKernel(
    const float* activations,
    std::size_t rows,
    std::size_t input_cols,
    const float* weights,
    std::size_t output_rows,
    float* output) {
  const std::size_t out_col = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const std::size_t row = (static_cast<std::size_t>(blockIdx.y) * blockDim.y) + threadIdx.y;
  if (row >= rows || out_col >= output_rows) {
    return;
  }

  double accum = 0.0;
  for (std::size_t col = 0; col < input_cols; ++col) {
    accum += static_cast<double>(activations[row * input_cols + col]) *
             static_cast<double>(weights[out_col * input_cols + col]);
  }
  output[row * output_rows + out_col] = static_cast<float>(accum);
}

__global__ void Nvfp4RowMajorReferenceKernel(
    Nvfp4RowMajorView activations,
    Nvfp4RowMajorView weights,
    float* output) {
  const std::size_t out_col = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const std::size_t row = (static_cast<std::size_t>(blockIdx.y) * blockDim.y) + threadIdx.y;
  if (row >= activations.rows || out_col >= weights.rows) {
    return;
  }

  double accum = 0.0;
  for (std::size_t col = 0; col < activations.cols; ++col) {
    const float a = DecodeNvfp4RowMajorValue(activations, row, col);
    const float b = DecodeNvfp4RowMajorValue(weights, out_col, col);
    accum += static_cast<double>(a) * static_cast<double>(b);
  }
  output[row * weights.rows + out_col] = static_cast<float>(accum);
}

}  // namespace

bool RunDenseRowMajorReferenceToDevice(
    const DeviceTensorFp32& activations,
    const DeviceDenseWeightFp32& weights,
    DeviceTensorFp32* output) {
  if (!activations.valid() ||
      !weights.valid() ||
      output == nullptr ||
      !output->valid() ||
      activations.shape().size() != 2 ||
      output->shape().size() != 2 ||
      activations.shape()[1] != weights.input_cols() ||
      output->shape()[0] != activations.shape()[0] ||
      output->shape()[1] != weights.output_rows()) {
    return false;
  }

  const dim3 block(16, 16);
  const dim3 grid(
      static_cast<unsigned int>((weights.output_rows() + block.x - 1u) / block.x),
      static_cast<unsigned int>((activations.shape()[0] + block.y - 1u) / block.y));
  DenseRowMajorReferenceKernel<<<grid, block>>>(
      activations.data(),
      activations.shape()[0],
      activations.shape()[1],
      weights.data(),
      weights.output_rows(),
      output->data());
  return CheckCuda(cudaGetLastError()) && CheckCuda(cudaDeviceSynchronize());
}

bool RunDenseRowMajorHighPrecisionReferenceToDevice(
    const DeviceTensorFp32& activations,
    const DeviceDenseWeightFp32& weights,
    DeviceTensorFp32* output) {
  if (!activations.valid() ||
      !weights.valid() ||
      output == nullptr ||
      !output->valid() ||
      activations.shape().size() != 2 ||
      output->shape().size() != 2 ||
      activations.shape()[1] != weights.input_cols() ||
      output->shape()[0] != activations.shape()[0] ||
      output->shape()[1] != weights.output_rows()) {
    return false;
  }

  const dim3 block(16, 16);
  const dim3 grid(
      static_cast<unsigned int>((weights.output_rows() + block.x - 1u) / block.x),
      static_cast<unsigned int>((activations.shape()[0] + block.y - 1u) / block.y));
  DenseRowMajorHighPrecisionReferenceKernel<<<grid, block>>>(
      activations.data(),
      activations.shape()[0],
      activations.shape()[1],
      weights.data(),
      weights.output_rows(),
      output->data());
  return CheckCuda(cudaGetLastError()) && CheckCuda(cudaDeviceSynchronize());
}

bool RunNvfp4RowMajorReferenceToDevice(
    const DeviceTensorFp32& activations,
    const DeviceNvfp4Weight& weights,
    DeviceTensorFp32* output,
    const Nvfp4PackOptions& pack_options) {
  if (!activations.valid() ||
      !weights.valid() ||
      output == nullptr ||
      !output->valid() ||
      activations.shape().size() != 2 ||
      output->shape().size() != 2 ||
      activations.shape()[1] != weights.input_cols() ||
      output->shape()[0] != activations.shape()[0] ||
      output->shape()[1] != weights.output_rows()) {
    return false;
  }

  auto packed_activations = PackDeviceRowMajorFp32ToNvfp4(activations, pack_options);
  if (!packed_activations || !packed_activations->valid()) {
    return false;
  }

  const Nvfp4RowMajorView activation_view{
      packed_activations->packed_data(),
      packed_activations->block_scales_data(),
      reinterpret_cast<const float*>(packed_activations->tensor_scale_data()),
      activations.shape()[0],
      activations.shape()[1],
  };
  const Nvfp4RowMajorView weight_view{
      weights.packed_data(),
      weights.block_scales_data(),
      reinterpret_cast<const float*>(weights.tensor_scale_data()),
      weights.output_rows(),
      weights.input_cols(),
  };

  const dim3 block(16, 16);
  const dim3 grid(
      static_cast<unsigned int>((weights.output_rows() + block.x - 1u) / block.x),
      static_cast<unsigned int>((activations.shape()[0] + block.y - 1u) / block.y));
  Nvfp4RowMajorReferenceKernel<<<grid, block>>>(
      activation_view,
      weight_view,
      output->data());
  return CheckCuda(cudaGetLastError()) && CheckCuda(cudaDeviceSynchronize());
}

}  // namespace nemotron
