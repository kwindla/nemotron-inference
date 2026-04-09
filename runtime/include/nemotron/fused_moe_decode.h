#pragma once

#include <cstddef>
#include <cstdint>

#include "nemotron/device_tensor.h"

namespace nemotron {

struct FusedNvfp4WeightView {
  const std::uint8_t* packed_data = nullptr;
  const std::uint8_t* block_scales_data = nullptr;
  const std::uint8_t* matmul_block_scales_data = nullptr;
  const float* tensor_scale_data = nullptr;
  std::size_t output_rows = 0;
  std::size_t input_cols = 0;
  const void* p5_tma_load_a = nullptr;
};

struct FusedMoeDirectLayerParams {
  std::size_t hidden_size = 0;
  std::size_t routed_expert_intermediate_size = 0;
  std::size_t shared_expert_intermediate_size = 0;
  std::size_t n_routed_experts = 0;
  std::size_t top_k = 0;
  std::size_t n_group = 0;
  std::size_t topk_group = 0;
  float routed_scaling_factor = 0.0f;
  bool norm_topk_prob = false;
  FusedNvfp4WeightView shared_up;
  FusedNvfp4WeightView shared_down;
  const FusedNvfp4WeightView* routed_up = nullptr;
  const FusedNvfp4WeightView* routed_down = nullptr;
  const float* correction_bias = nullptr;
  const int* selected_indices = nullptr;
  const float* selected_weights = nullptr;
  float* routed_output = nullptr;
  float* shared_output = nullptr;
};

bool FusedMoeDecodeEnabled();

bool RunDeviceExpertSelection(
    const DeviceTensorFp32& router_logits,
    const DeviceTensorFp32& correction_bias,
    std::size_t n_routed_experts,
    std::size_t top_k,
    std::size_t n_group,
    std::size_t topk_group,
    float routed_scaling_factor,
    bool norm_topk_prob,
    int* selected_indices,
    float* selected_weights);

bool AccumulateScaledFp32ByDeviceWeight(
    const DeviceTensorFp32& input,
    const float* scales_device,
    std::size_t scale_index,
    DeviceTensorFp32* output);

bool RunFusedMoeDirectDecode(
    const FusedMoeDirectLayerParams& params,
    const DeviceTensorFp32& input,
    const DeviceTensorFp32& normalized,
    const DeviceTensorFp32& router_logits,
    DeviceTensorFp32* output);

}  // namespace nemotron
