#pragma once

#include <cstddef>

#include "nemotron/device_tensor.h"
#include "nemotron/request_context.h"

namespace nemotron {

struct MambaConvPrefillParams {
  std::size_t intermediate_size = 0;
  std::size_t state_size = 0;
  std::size_t n_groups = 0;
  std::size_t conv_kernel_size = 0;
  std::size_t conv_state_offset_elems = 0;
  const float* conv1d_weight = nullptr;
  const float* conv1d_bias = nullptr;
};

bool RunMambaConvPrefill(
    const MambaConvPrefillParams& params,
    RequestExecutionContext& request_context,
    const DeviceTensorFp32& projected,
    DeviceTensorFp32* conv_output,
    const DeviceTensorFp32* initial_conv_state = nullptr);

}  // namespace nemotron
