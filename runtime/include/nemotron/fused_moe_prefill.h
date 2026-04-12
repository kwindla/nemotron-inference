#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_bf16.h>

#include "nemotron/expert_routing_device.h"
#include "nemotron/fused_moe_decode.h"
#include "nemotron/moe_launch_plan_device.h"
#include "nemotron/nvfp4_scale_layout.h"

namespace nemotron {

class DeviceNvfp4Matrix;
class GemmHeuristicCache;

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
  const DeviceNvfp4Matrix* normalized_pack = nullptr;  // Default shared FC1 pack and optional packed routed FC1 source pack.
  DeviceNvfp4Matrix* shared_fc1_pack = nullptr;  // Alias for the shared FC1 input pack. Defaults to normalized_pack.
  DeviceExpertRouting* routing = nullptr;
  DeviceMoeLaunchPlan* launch_plan = nullptr;
  GemmHeuristicCache* heuristic_cache = nullptr;
  float* routed_gather_scratch = nullptr;  // padded_row_capacity x hidden_size
  float* fc1_expert_activation_scales = nullptr;  // Optional legacy FC1 input tensor scales.
  DeviceNvfp4Matrix* fc1_grouped_pack = nullptr;  // Optional legacy packed FC1 input contract.
  float* routed_up_scratch = nullptr;      // Optional fallback-only FP32 FC1->FC2 boundary.
  __nv_bfloat16* gemm1_output_bf16 = nullptr;  // TRT-style BF16 Gemm1 output buffer.
  float* fc2_expert_activation_scales = nullptr;  // Legacy alias for post-activation FC1 output scales.
  DeviceNvfp4Matrix* fc2_grouped_pack = nullptr;  // Legacy alias for packed routed FC2 input contract.
  DeviceNvfp4Matrix* shared_fc2_pack = nullptr;  // Alias for the shared FC2 input pack.
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
  std::uint32_t expected_b_regs[2][2] = {};
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
  std::uint8_t b_copy_raw[16] = {};
};

constexpr int kP1NativeFp4MmaTraceGroupCount = 8 * 2 * 32;
constexpr int kP1NativeFp4MmaTraceEntryCount =
    kP1NativeFp4MmaTraceGroupCount * 4;
constexpr int kP1NaturalFp4MmaTraceEntryCount = 128 * 32;

struct P1NativeFp4MmaTrace {
  int valid = 0;
  int group_output_cols[kP1NativeFp4MmaTraceGroupCount] = {};
  int group_match_counts[kP1NativeFp4MmaTraceGroupCount] = {};
  int group_warp_ids[kP1NativeFp4MmaTraceGroupCount] = {};
  int group_subtile_ids[kP1NativeFp4MmaTraceGroupCount] = {};
  int group_lane_ids[kP1NativeFp4MmaTraceGroupCount] = {};
  int output_cols[kP1NativeFp4MmaTraceEntryCount] = {};
  int token_rows[kP1NativeFp4MmaTraceEntryCount] = {};
  int physical_indices[kP1NativeFp4MmaTraceEntryCount] = {};
  int warp_ids[kP1NativeFp4MmaTraceEntryCount] = {};
  int subtile_ids[kP1NativeFp4MmaTraceEntryCount] = {};
  int lane_ids[kP1NativeFp4MmaTraceEntryCount] = {};
  int reg_ids[kP1NativeFp4MmaTraceEntryCount] = {};
  float values[kP1NativeFp4MmaTraceEntryCount] = {};
};

struct P1NaturalFp4MmaTrace {
  int valid = 0;
  int output_rows[kP1NaturalFp4MmaTraceEntryCount] = {};
  int token_rows[kP1NaturalFp4MmaTraceEntryCount] = {};
  int warp_ids[kP1NaturalFp4MmaTraceEntryCount] = {};
  int lane_ids[kP1NaturalFp4MmaTraceEntryCount] = {};
  int physical_indices[kP1NaturalFp4MmaTraceEntryCount] = {};
  int reg_ids[kP1NaturalFp4MmaTraceEntryCount] = {};
  int m_fragment_ids[kP1NaturalFp4MmaTraceEntryCount] = {};
  int n_fragment_ids[kP1NaturalFp4MmaTraceEntryCount] = {};
  float values[kP1NaturalFp4MmaTraceEntryCount] = {};
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

// Testing hooks: expose the current routed-profile selectors without requiring
// log scraping.
const char* SelectRoutedGemm1ProfileNameForTesting(std::size_t num_rows);
const char* ClassifyRoutedGemm1ProfileForTesting(std::size_t num_rows);
const char* SelectRoutedGemm2ProfileNameForTesting(std::size_t num_rows);
const char* ClassifyRoutedGemm2ProfileForTesting(std::size_t num_rows);

// Testing hook: runs the live P5 native direct-pack epilogue against a
// synthetic 128x128 accumulator tile laid out with the real CUTE
// `TracedP5AccumProfileLayout`.
bool RunP5NativeDirectPackOracleForTesting(
    const float* input,
    const float* per_row_tensor_scales_input,
    int valid_rows,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout,
    float* activation_output_scale,
    std::uint8_t* packed_data,
    std::uint8_t* block_scales_data,
    std::uint8_t* matmul_block_scales_data,
    float* tensor_scale_data,
    float* per_row_tensor_scales);

// Testing hook: validates the dormant generic direct-stage helpers against the
// production P13 dense store contract on a synthetic 32x128 tile. The generic
// path writes `generic_dense_output` through its BF16 debug-prepack surface,
// while `store_dense_output` comes from `StoreUnifiedRoutedFp4Output<P13,
// __nv_bfloat16>`.
bool RunP13GenericDirectStageOracleForTesting(
    const float* input,
    const float* per_row_tensor_scales_input,
    int valid_rows,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout,
    __nv_bfloat16* generic_dense_output,
    __nv_bfloat16* store_dense_output,
    float* activation_output_scale,
    std::uint8_t* packed_data,
    std::uint8_t* block_scales_data,
    std::uint8_t* matmul_block_scales_data,
    float* tensor_scale_data,
    float* per_row_tensor_scales);

bool CopyP1NativeFp4MmaTrace(P1NativeFp4MmaTrace* out);
void ResetP1NativeFp4MmaTrace();
bool CopyP1NaturalFp4MmaTrace(P1NaturalFp4MmaTrace* out);
void ResetP1NaturalFp4MmaTrace();

// Testing hook: runs the current traced P1 native FP4 MMA load+mma chain in
// isolation on a synthetic 16x64 activation tile and 128x64 weight tile,
// storing into a dense 16x128 output through a CUTE-derived coord map.
bool RunP1NativeFp4MmaOracleForTesting(
    const std::uint8_t* input_packed,
    const std::uint8_t* input_block_scales,
    const std::uint8_t* weight_packed,
    const std::uint8_t* weight_matmul_block_scales,
    float* dense_output);

// Testing hook: runs the natural traced P1 MMA contract end-to-end on a single
// 128x32x64 tile with A=weight(128x64), B=input(32x64), and writes the result
// out token-major as a 32x128 FP32 rectangle.
bool RunP1NaturalFp4MmaOracleForTesting(
    const std::uint8_t* weight_packed,
    const std::uint8_t* weight_matmul_block_scales,
    const std::uint8_t* input_packed,
    const std::uint8_t* input_block_scales,
    float* token_major_output);

}  // namespace nemotron
