#pragma once

#include "nemotron/device_tensor.h"

#include <cstdint>

namespace nemotron {

bool DeviceArgmax(const DeviceTensorFp32& logits_row, std::int32_t* device_token_id);
bool DeviceArgmaxLastRow(const DeviceTensorFp32& logits, std::int32_t* device_token_id);

}  // namespace nemotron
