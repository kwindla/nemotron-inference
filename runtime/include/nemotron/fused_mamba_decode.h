#pragma once

#include <cstddef>

#include "nemotron/device_tensor.h"
#include "nemotron/request_context.h"

namespace nemotron {

struct FusedMambaLayerParams {
  std::size_t intermediate_size = 0;
  std::size_t num_heads = 0;
  std::size_t head_dim = 0;
  std::size_t state_size = 0;
  std::size_t n_groups = 0;
  std::size_t conv_kernel_size = 0;
  std::size_t conv_state_offset_elems = 0;
  std::size_t ssm_state_offset_elems = 0;
  float mixer_rms_epsilon = 0.0f;
  float time_step_min = 0.0f;
  const float* mixer_norm_weight = nullptr;
  const float* conv1d_weight = nullptr;
  const float* conv1d_bias = nullptr;
  const float* A_log = nullptr;
  const float* D = nullptr;
  const float* dt_bias = nullptr;
};

bool FusedMambaDecodeEnabled();

bool RunFusedMambaDecode(
    const FusedMambaLayerParams& params,
    RequestExecutionContext& request_context,
    const DeviceTensorFp32& projected,
    DeviceTensorFp32* scan_output);

}  // namespace nemotron
