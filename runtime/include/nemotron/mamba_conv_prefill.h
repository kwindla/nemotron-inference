#pragma once

#include <cstddef>

#include "nemotron/device_tensor.h"

namespace nemotron {

struct MambaConvPrefillParams {
  std::size_t intermediate_size = 0;
  std::size_t state_size = 0;
  std::size_t n_groups = 0;
  std::size_t conv_kernel_size = 0;
  std::size_t conv_state_elems = 0;
  float* final_conv_state = nullptr;
  const float* initial_conv_state = nullptr;
  const float* conv1d_weight = nullptr;
  const float* conv1d_bias = nullptr;
};

bool RunMambaConvPrefill(
    const MambaConvPrefillParams& params,
    const DeviceTensorFp32& projected,
    DeviceTensorFp32* conv_output);

}  // namespace nemotron
