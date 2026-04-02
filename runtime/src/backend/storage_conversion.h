#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>

namespace nemotron {

enum class PackedFloatStorage {
  kFp32,
  kBf16,
  kFp8E4M3,
};

bool UploadPackedFloatToDeviceFp32(
    const void* host_source,
    std::size_t element_count,
    PackedFloatStorage storage,
    float scale,
    float* device_output);

bool UploadPackedFloatToDeviceBf16(
    const void* host_source,
    std::size_t element_count,
    PackedFloatStorage storage,
    float scale,
    __nv_bfloat16* device_output);

bool ConvertDeviceFp32ToBf16(
    const float* device_input,
    std::size_t element_count,
    __nv_bfloat16* device_output);

bool ConvertDeviceBf16ToFp32(
    const __nv_bfloat16* device_input,
    std::size_t element_count,
    float* device_output);

bool QuantizeDeviceFp32ToScaledFp8RoundTripBf16(
    const float* device_input,
    std::size_t element_count,
    float input_scale,
    __nv_bfloat16* device_output);

bool QuantizeDeviceFp32ToFp8E4M3(
    const float* device_input,
    std::size_t element_count,
    float input_scale,
    std::uint8_t* device_output);

}  // namespace nemotron
