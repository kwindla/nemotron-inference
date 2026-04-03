#pragma once

#include <cstddef>

#include "nemotron/device_tensor.h"

namespace nemotron {

struct MambaGatedGroupNormParams {
  std::size_t intermediate_size = 0;
  std::size_t n_groups = 0;
  float mixer_rms_epsilon = 0.0f;
  const float* mixer_norm_weight = nullptr;
};

bool RunMambaGatedGroupNorm(
    const MambaGatedGroupNormParams& params,
    const DeviceTensorFp32& projected,
    DeviceTensorFp32* ssm_output);

}  // namespace nemotron
