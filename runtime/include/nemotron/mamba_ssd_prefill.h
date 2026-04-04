#pragma once

#include <cstddef>

#include "nemotron/device_tensor.h"

namespace nemotron {

struct MambaSsdPrefillParams {
  std::size_t intermediate_size = 0;
  std::size_t num_heads = 0;
  std::size_t head_dim = 0;
  std::size_t state_size = 0;
  std::size_t n_groups = 0;
  std::size_t ssm_state_elems = 0;
  float* final_ssm_state = nullptr;
  const float* initial_ssm_state = nullptr;
  float time_step_min = 0.0f;
  const float* A_log = nullptr;
  const float* D = nullptr;
  const float* dt_bias = nullptr;
};

bool RunMambaSsdPrefill(
    const MambaSsdPrefillParams& params,
    const DeviceTensorFp32& projected,
    const DeviceTensorFp32& conv_output,
    DeviceTensorFp32* ssm_output);

}  // namespace nemotron
