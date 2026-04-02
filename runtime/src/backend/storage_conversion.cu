#include "storage_conversion.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace nemotron {
namespace {

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

__global__ void ScaleFp32Kernel(
    const float* input,
    std::size_t element_count,
    float scale,
    float* output) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  output[index] = input[index] * scale;
}

__global__ void ConvertBf16ToFp32Kernel(
    const __nv_bfloat16* input,
    std::size_t element_count,
    float scale,
    float* output) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  output[index] = __bfloat162float(input[index]) * scale;
}

__global__ void ConvertFp8E4M3ToFp32Kernel(
    const __nv_fp8_e4m3* input,
    std::size_t element_count,
    float scale,
    float* output) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  output[index] = static_cast<float>(input[index]) * scale;
}

__global__ void ConvertFp32ToBf16Kernel(
    const float* input,
    std::size_t element_count,
    __nv_bfloat16* output) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  output[index] = __float2bfloat16(input[index]);
}

__global__ void ConvertFp8E4M3ToBf16Kernel(
    const __nv_fp8_e4m3* input,
    std::size_t element_count,
    float scale,
    __nv_bfloat16* output) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  output[index] = __float2bfloat16(static_cast<float>(input[index]) * scale);
}

__global__ void QuantizeFp32ToFp8E4M3Kernel(
    const float* input,
    std::size_t element_count,
    float scale,
    std::uint8_t* output) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  const float normalized = input[index] / scale;
  output[index] = static_cast<std::uint8_t>(
      __nv_cvt_float_to_fp8(normalized, __NV_SATFINITE, __NV_E4M3));
}

__global__ void QuantizeFp32ToScaledFp8RoundTripBf16Kernel(
    const float* input,
    std::size_t element_count,
    float scale,
    __nv_bfloat16* output) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  const float normalized = input[index] / scale;
  const std::uint8_t raw = static_cast<std::uint8_t>(
      __nv_cvt_float_to_fp8(normalized, __NV_SATFINITE, __NV_E4M3));
  __nv_fp8_e4m3 quantized;
  quantized.__x = raw;
  output[index] = __float2bfloat16(static_cast<float>(quantized) * scale);
}

std::size_t SourceBytes(std::size_t element_count, PackedFloatStorage storage) {
  switch (storage) {
    case PackedFloatStorage::kFp32:
      return element_count * sizeof(float);
    case PackedFloatStorage::kBf16:
      return element_count * sizeof(__nv_bfloat16);
    case PackedFloatStorage::kFp8E4M3:
      return element_count * sizeof(__nv_fp8_e4m3);
  }
  return 0;
}

}  // namespace

bool UploadPackedFloatToDeviceFp32(
    const void* host_source,
    std::size_t element_count,
    PackedFloatStorage storage,
    float scale,
    float* device_output) {
  if (host_source == nullptr || element_count == 0 || device_output == nullptr) {
    return false;
  }

  constexpr int kBlockSize = 256;
  const int grid_size = static_cast<int>(
      (element_count + static_cast<std::size_t>(kBlockSize) - 1u) /
      static_cast<std::size_t>(kBlockSize));

  if (storage == PackedFloatStorage::kFp32 && scale == 1.0f) {
    return CheckCuda(cudaMemcpy(
        device_output,
        host_source,
        SourceBytes(element_count, storage),
        cudaMemcpyHostToDevice));
  }

  void* device_source = nullptr;
  const std::size_t source_bytes = SourceBytes(element_count, storage);
  if (source_bytes == 0) {
    return false;
  }
  if (!CheckCuda(cudaMalloc(&device_source, source_bytes))) {
    return false;
  }
  if (!CheckCuda(cudaMemcpy(device_source, host_source, source_bytes, cudaMemcpyHostToDevice))) {
    cudaFree(device_source);
    return false;
  }

  switch (storage) {
    case PackedFloatStorage::kFp32:
      ScaleFp32Kernel<<<grid_size, kBlockSize>>>(
          reinterpret_cast<const float*>(device_source),
          element_count,
          scale,
          device_output);
      break;
    case PackedFloatStorage::kBf16:
      ConvertBf16ToFp32Kernel<<<grid_size, kBlockSize>>>(
          reinterpret_cast<const __nv_bfloat16*>(device_source),
          element_count,
          scale,
          device_output);
      break;
    case PackedFloatStorage::kFp8E4M3:
      ConvertFp8E4M3ToFp32Kernel<<<grid_size, kBlockSize>>>(
          reinterpret_cast<const __nv_fp8_e4m3*>(device_source),
          element_count,
          scale,
          device_output);
      break;
  }

  const cudaError_t kernel_status = cudaGetLastError();
  const cudaError_t free_status = cudaFree(device_source);
  return CheckCuda(kernel_status) && CheckCuda(free_status);
}

bool UploadPackedFloatToDeviceBf16(
    const void* host_source,
    std::size_t element_count,
    PackedFloatStorage storage,
    float scale,
    __nv_bfloat16* device_output) {
  if (host_source == nullptr || element_count == 0 || device_output == nullptr) {
    return false;
  }

  constexpr int kBlockSize = 256;
  const int grid_size = static_cast<int>(
      (element_count + static_cast<std::size_t>(kBlockSize) - 1u) /
      static_cast<std::size_t>(kBlockSize));

  if (storage == PackedFloatStorage::kBf16 && scale == 1.0f) {
    return CheckCuda(cudaMemcpy(
        device_output,
        host_source,
        SourceBytes(element_count, storage),
        cudaMemcpyHostToDevice));
  }

  void* device_source = nullptr;
  const std::size_t source_bytes = SourceBytes(element_count, storage);
  if (source_bytes == 0) {
    return false;
  }
  if (!CheckCuda(cudaMalloc(&device_source, source_bytes))) {
    return false;
  }
  if (!CheckCuda(cudaMemcpy(device_source, host_source, source_bytes, cudaMemcpyHostToDevice))) {
    cudaFree(device_source);
    return false;
  }

  switch (storage) {
    case PackedFloatStorage::kFp32: {
      auto temp_fp32 = reinterpret_cast<float*>(device_source);
      ScaleFp32Kernel<<<grid_size, kBlockSize>>>(
          temp_fp32,
          element_count,
          scale,
          temp_fp32);
      if (!CheckCuda(cudaGetLastError())) {
        cudaFree(device_source);
        return false;
      }
      ConvertFp32ToBf16Kernel<<<grid_size, kBlockSize>>>(
          temp_fp32,
          element_count,
          device_output);
      break;
    }
    case PackedFloatStorage::kBf16: {
      float* temp_fp32 = nullptr;
      if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&temp_fp32), element_count * sizeof(float)))) {
        cudaFree(device_source);
        return false;
      }
      ConvertBf16ToFp32Kernel<<<grid_size, kBlockSize>>>(
          reinterpret_cast<const __nv_bfloat16*>(device_source),
          element_count,
          scale,
          temp_fp32);
      if (!CheckCuda(cudaGetLastError())) {
        cudaFree(temp_fp32);
        cudaFree(device_source);
        return false;
      }
      ConvertFp32ToBf16Kernel<<<grid_size, kBlockSize>>>(
          temp_fp32,
          element_count,
          device_output);
      const cudaError_t convert_status = cudaGetLastError();
      const cudaError_t temp_free_status = cudaFree(temp_fp32);
      if (!CheckCuda(convert_status) || !CheckCuda(temp_free_status)) {
        cudaFree(device_source);
        return false;
      }
      break;
    }
    case PackedFloatStorage::kFp8E4M3:
      ConvertFp8E4M3ToBf16Kernel<<<grid_size, kBlockSize>>>(
          reinterpret_cast<const __nv_fp8_e4m3*>(device_source),
          element_count,
          scale,
          device_output);
      break;
  }

  const cudaError_t kernel_status = cudaGetLastError();
  const cudaError_t free_status = cudaFree(device_source);
  return CheckCuda(kernel_status) && CheckCuda(free_status);
}

bool ConvertDeviceFp32ToBf16(
    const float* device_input,
    std::size_t element_count,
    __nv_bfloat16* device_output,
    cudaStream_t stream) {
  if (device_input == nullptr || element_count == 0 || device_output == nullptr) {
    return false;
  }

  constexpr int kBlockSize = 256;
  const int grid_size = static_cast<int>(
      (element_count + static_cast<std::size_t>(kBlockSize) - 1u) /
      static_cast<std::size_t>(kBlockSize));
  ConvertFp32ToBf16Kernel<<<grid_size, kBlockSize, 0, stream>>>(
      device_input,
      element_count,
      device_output);
  return CheckCuda(cudaGetLastError());
}

bool ConvertDeviceBf16ToFp32(
    const __nv_bfloat16* device_input,
    std::size_t element_count,
    float* device_output,
    cudaStream_t stream) {
  if (device_input == nullptr || element_count == 0 || device_output == nullptr) {
    return false;
  }

  constexpr int kBlockSize = 256;
  const int grid_size = static_cast<int>(
      (element_count + static_cast<std::size_t>(kBlockSize) - 1u) /
      static_cast<std::size_t>(kBlockSize));
  ConvertBf16ToFp32Kernel<<<grid_size, kBlockSize, 0, stream>>>(
      device_input,
      element_count,
      1.0f,
      device_output);
  return CheckCuda(cudaGetLastError());
}

bool QuantizeDeviceFp32ToScaledFp8RoundTripBf16(
    const float* device_input,
    std::size_t element_count,
    float input_scale,
    __nv_bfloat16* device_output,
    cudaStream_t stream) {
  if (device_input == nullptr || element_count == 0 || device_output == nullptr) {
    return false;
  }

  constexpr float kMinScale = 1.0f / 1024.0f;
  const float scale = (!std::isfinite(input_scale) || input_scale < kMinScale)
      ? kMinScale
      : input_scale;
  constexpr int kBlockSize = 256;
  const int grid_size = static_cast<int>(
      (element_count + static_cast<std::size_t>(kBlockSize) - 1u) /
      static_cast<std::size_t>(kBlockSize));
  QuantizeFp32ToScaledFp8RoundTripBf16Kernel<<<grid_size, kBlockSize, 0, stream>>>(
      device_input,
      element_count,
      scale,
      device_output);
  return CheckCuda(cudaGetLastError());
}

bool QuantizeDeviceFp32ToFp8E4M3(
    const float* device_input,
    std::size_t element_count,
    float input_scale,
    std::uint8_t* device_output,
    cudaStream_t stream) {
  if (device_input == nullptr || element_count == 0 || device_output == nullptr) {
    return false;
  }

  constexpr float kMinScale = 1.0f / 1024.0f;
  const float scale = (!std::isfinite(input_scale) || input_scale < kMinScale)
      ? kMinScale
      : input_scale;
  constexpr int kBlockSize = 256;
  const int grid_size = static_cast<int>(
      (element_count + static_cast<std::size_t>(kBlockSize) - 1u) /
      static_cast<std::size_t>(kBlockSize));
  QuantizeFp32ToFp8E4M3Kernel<<<grid_size, kBlockSize, 0, stream>>>(
      device_input,
      element_count,
      scale,
      device_output);
  return CheckCuda(cudaGetLastError());
}

}  // namespace nemotron
