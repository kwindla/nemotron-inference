#pragma once

#include "nemotron/device_tensor.h"

namespace nemotron {

bool ResidualAddFp32(
    const DeviceTensorFp32& lhs,
    const DeviceTensorFp32& rhs,
    DeviceTensorFp32* output);

bool ResidualAddBf16(
    const DeviceTensorBf16& lhs,
    const DeviceTensorBf16& rhs,
    DeviceTensorBf16* output);

bool RmsNormFp32(
    const DeviceTensorFp32& input,
    const DeviceTensorFp32& weight,
    float epsilon,
    DeviceTensorFp32* output);

bool RmsNormBf16(
    const DeviceTensorBf16& input,
    const DeviceTensorFp32& weight,
    float epsilon,
    DeviceTensorBf16* output);

}  // namespace nemotron
