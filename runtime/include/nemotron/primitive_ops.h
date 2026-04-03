#pragma once

#include "nemotron/device_tensor.h"

namespace nemotron {

bool ResidualAddFp32(
    const DeviceTensorFp32& lhs,
    const DeviceTensorFp32& rhs,
    DeviceTensorFp32* output);

bool Relu2InPlaceFp32(DeviceTensorFp32* tensor);

bool AccumulateScaledFp32(
    const DeviceTensorFp32& input,
    float scale,
    DeviceTensorFp32* output);

bool GatherRowsFp32(
    const DeviceTensorFp32& input,
    const int* row_indices_device,
    DeviceTensorFp32* output);

bool ScatterAddWeightedRowsFp32(
    const DeviceTensorFp32& input,
    const int* row_indices_device,
    const float* row_weights_device,
    DeviceTensorFp32* output);

bool RmsNormFp32(
    const DeviceTensorFp32& input,
    const DeviceTensorFp32& weight,
    float epsilon,
    DeviceTensorFp32* output);

}  // namespace nemotron
