#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>

#include "nemotron/expert_routing_device.h"
#include "nemotron/fused_moe_decode.h"
#include "nemotron/moe_launch_plan_device.h"

namespace nemotron {

class DeviceNvfp4Matrix;

struct FusedMoePrefillParams {
  std::size_t token_count = 0;
  std::size_t hidden_size = 0;
  std::size_t routed_expert_intermediate_size = 0;
  std::size_t shared_expert_intermediate_size = 0;
  std::size_t n_routed_experts = 0;
  std::size_t top_k = 0;
  FusedNvfp4WeightView shared_up;
  FusedNvfp4WeightView shared_down;
  const FusedNvfp4WeightView* routed_up_device = nullptr;
  const FusedNvfp4WeightView* routed_down_device = nullptr;
  const int* selected_indices = nullptr;
  const float* selected_weights = nullptr;
  const float* input = nullptr;  // Reserved for future fused epilog variants.
  const float* normalized = nullptr;
  const DeviceNvfp4Matrix* normalized_pack = nullptr;  // Optional legacy global-scale FC1 source pack.
  DeviceExpertRouting* routing = nullptr;
  DeviceMoeLaunchPlan* launch_plan = nullptr;
  float* routed_gather_scratch = nullptr;  // padded_row_capacity x hidden_size
  float* fc1_expert_activation_scales = nullptr;  // Optional legacy FC1 input tensor scales.
  DeviceNvfp4Matrix* fc1_grouped_pack = nullptr;  // Optional legacy packed FC1 input contract.
  float* routed_up_scratch = nullptr;      // Optional fallback-only FP32 FC1->FC2 boundary.
  __nv_bfloat16* gemm1_output_bf16 = nullptr;  // TRT-style BF16 Gemm1 output buffer.
  float* fc2_expert_activation_scales = nullptr;  // Legacy alias for post-activation FC1 output scales.
  DeviceNvfp4Matrix* fc2_grouped_pack = nullptr;  // Legacy alias for packed routed FC2 input contract.
  DeviceNvfp4Matrix* gemm1_output = nullptr;  // TRT-style routed Gemm1 output contract, post-activation for ReLU2.
  float* gemm1_output_scale = nullptr;       // TRT-style routed Gemm1 output scalar scale contract.
  float* activation_output_scale = nullptr;  // TRT-style activation output scales; ReLU2 path aliases gemm1_output_scale.
  float* shared_up_scratch = nullptr;      // token_count x shared_expert_intermediate_size
  float* output = nullptr;  // Receives routed + shared expert contributions.
  float* routed_output = nullptr;  // Optional.
  float* shared_output = nullptr;  // Optional.
};

struct P13DebugTrace {
  int valid = 0;
  int row_start = -1;
  int valid_rows = -1;
  int output_row_base = -1;
  std::uint32_t a_regs[2][4] = {};
  std::uint32_t b_regs[2][2] = {};
  std::uint32_t a_scale_words[2] = {};
  std::uint32_t b_scale_words[2] = {};
  float block0_accum_regs[2][2][4] = {};
  float accum_regs[2][2][4] = {};
  int store_rows[16] = {};
  int store_cols[16] = {};
  int a_copy_rows[32] = {};
  int a_copy_cols[32] = {};
  std::uint8_t a_copy_raw[32] = {};
  int b_copy_rows[16] = {};
  int b_copy_cols[16] = {};
};

bool RunGroupedNvfp4ExpertMatVec(
    const float* input,
    const int* expert_offsets,
    std::size_t n_experts,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output);

bool RunLaunchPlannedNvfp4ExpertMatVec(
    const float* input,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output);

bool RunLaunchPlannedPackedNvfp4ExpertMatVecBf16(
    const DeviceNvfp4Matrix& input_pack,
    const DeviceMoeLaunchPlan* launch_plan,
    std::size_t dispatch_rows,
    std::size_t active_selection_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    __nv_bfloat16* output);

bool RunFusedMoePrefill(const FusedMoePrefillParams& params);

bool CopyP13DebugTrace(P13DebugTrace* out);
void ResetP13DebugTrace();

}  // namespace nemotron
