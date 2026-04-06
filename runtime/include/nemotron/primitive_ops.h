#pragma once

#include <cuda_runtime.h>

#include "nemotron/device_tensor.h"

namespace nemotron {

bool ResidualAddFp32(
    const DeviceTensorFp32& lhs,
    const DeviceTensorFp32& rhs,
    DeviceTensorFp32* output,
    cudaStream_t stream = nullptr);

bool ResidualAddBf16(
    const DeviceTensorBf16& lhs,
    const DeviceTensorBf16& rhs,
    DeviceTensorBf16* output,
    cudaStream_t stream = nullptr);

bool RmsNormFp32(
    const DeviceTensorFp32& input,
    const DeviceTensorFp32& weight,
    float epsilon,
    DeviceTensorFp32* output,
    cudaStream_t stream = nullptr);

bool RmsNormBf16(
    const DeviceTensorBf16& input,
    const DeviceTensorFp32& weight,
    float epsilon,
    DeviceTensorBf16* output,
    cudaStream_t stream = nullptr);

bool RmsNormFp32ToBf16(
    const DeviceTensorFp32& input,
    const DeviceTensorFp32& weight,
    float epsilon,
    DeviceTensorBf16* output,
    cudaStream_t stream = nullptr);

}  // namespace nemotron
