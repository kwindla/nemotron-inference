#pragma once

#include "nemotron/dense_weight.h"
#include "nemotron/device_tensor.h"
#include "nemotron/nvfp4_packing.h"
#include "nemotron/nvfp4_weight.h"

namespace nemotron {

bool RunDenseRowMajorReferenceToDevice(
    const DeviceTensorFp32& activations,
    const DeviceDenseWeightFp32& weights,
    DeviceTensorFp32* output);

bool RunDenseRowMajorHighPrecisionReferenceToDevice(
    const DeviceTensorFp32& activations,
    const DeviceDenseWeightFp32& weights,
    DeviceTensorFp32* output);

bool RunNvfp4RowMajorReferenceToDevice(
    const DeviceTensorFp32& activations,
    const DeviceNvfp4Weight& weights,
    DeviceTensorFp32* output,
    const Nvfp4PackOptions& pack_options = {});

}  // namespace nemotron
