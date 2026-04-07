#include "nemotron/cublaslt_gemm_plan.h"
#include "nemotron/cublaslt_handle.h"
#include "nemotron/gemm_catalog.h"
#include "nemotron/gemm_execution.h"
#include "nemotron/gemm_planner.h"
#include "nemotron/device_nvfp4_matrix.h"
#include "nemotron/device_tensor.h"
#include "nemotron/fused_moe_prefill.h"
#include "nemotron/monolithic_expert_weights.h"
#include "nemotron/nvfp4_gemm_runner.h"
#include "nemotron/nvfp4_packing.h"
#include "nemotron/nvfp4_weight.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

using nemotron::DeviceNvfp4Matrix;
using nemotron::DeviceTensorFp32;
using nemotron::FusedNvfp4WeightView;
using nemotron::MonolithicNvfp4ExpertWeights;
using nemotron::Nvfp4PackOptions;
using nemotron::Nvfp4ScaleLayout;
using nemotron::PackRowMajorFp32ToNvfp4;
using nemotron::BuildDeviceMoeLaunchPlan;
using nemotron::BuildDeviceMoeExactTaskMap;
using nemotron::DeviceExpertRouting;
using nemotron::DeviceMoeLaunchPlan;
using nemotron::BuildCublasLtGemmPlan;
using nemotron::BuildGemmLaunchPlan;
using nemotron::CublasLtHandle;
using nemotron::RunDeviceExpertRouting;
using nemotron::RunGroupedNvfp4ExpertMatVec;
using nemotron::RunLaunchPlannedNvfp4ExpertMatVec;
using nemotron::DeviceNvfp4Weight;
using nemotron::GemmDescriptor;
using nemotron::GemmHeuristicCache;
using nemotron::GemmKernelFamily;
using nemotron::HostNvfp4Matrix;
using nemotron::MakeNvfp4PackedMatrixDeviceView;
using nemotron::Nvfp4PackedMatrixDeviceView;
using nemotron::PrepareGemmExecution;
using nemotron::RunNvfp4RowMajorFp32AccumToDevice;
namespace wmma = nvcuda::wmma;

constexpr std::size_t kHiddenSize = 2688;
constexpr std::size_t kRoutedIntermediateSize = 1856;
constexpr std::size_t kRoutedExperts = 128;
constexpr std::size_t kTopK = 6;
constexpr double kNvfp4BytesPerWeight = 0.5625;
constexpr std::size_t kNvfp4BlockWidth = 16;
constexpr int kPermutedTokenTile = 8;
constexpr int kPermutedOutputTile = 4;
constexpr int kPermutedWideOutputTile = 8;
constexpr int kPermutedGroupedOutputTile = 32;
constexpr int kPermutedGroupedRowsPerWarp =
    kPermutedGroupedOutputTile / kPermutedWideOutputTile;
constexpr int kPermutedCoopOutputTile = 32;
constexpr int kPermutedCoopThreadsPerBlock =
    kPermutedTokenTile * kPermutedCoopOutputTile;
constexpr int kPermutedMicroTileOutputsPerThread = 2;
constexpr int kPermutedMicroTileThreadN =
    kPermutedCoopOutputTile / kPermutedMicroTileOutputsPerThread;
constexpr int kPermutedMicroTileThreadsPerBlock =
    kPermutedTokenTile * kPermutedMicroTileThreadN;
constexpr int kWmmaTileM = 16;
constexpr int kWmmaTileN = 16;
constexpr int kWmmaTileK = 16;
constexpr int kWmmaOutputTile = 32;
constexpr int kWmmaWarpsPerBlock = 2;
constexpr int kWmmaThreadsPerBlock = kWmmaWarpsPerBlock * 32;
constexpr int kPermutedPairsPerStep = 32;
constexpr int kPermutedThreadsPerBlock = kPermutedOutputTile * 32;
constexpr int kPermutedWideThreadsPerBlock = kPermutedWideOutputTile * 32;

struct BenchmarkCase {
  std::string name;
  std::size_t prefix_tokens = 0;
};

struct PackedInputView {
  const std::uint8_t* packed_data = nullptr;
  const std::uint8_t* block_scales_data = nullptr;
  const float* tensor_scale_data = nullptr;
  std::size_t rows = 0;
  std::size_t cols = 0;
};

struct BenchmarkOptions {
  std::vector<std::string> selected_case_names;
  std::size_t warmup_iterations = 1;
  std::size_t hot_iterations = 5;
};

struct BenchmarkResult {
  std::string variant_name;
  std::string case_name;
  std::size_t prefix_tokens = 0;
  std::size_t selection_count = 0;
  std::size_t padded_selection_count = 0;
  std::size_t active_experts = 0;
  double cold_ms = 0.0;
  double hot_mean_ms = 0.0;
  double hot_min_ms = 0.0;
  double hot_max_ms = 0.0;
  double hot_tflops = 0.0;
  double hot_weight_gib_per_s = 0.0;
  double max_abs_diff_vs_baseline = 0.0;
  std::size_t max_abs_diff_index = 0;
  float max_abs_diff_baseline_value = 0.0f;
  float max_abs_diff_variant_value = 0.0f;
};

struct UploadedWeights {
  std::unique_ptr<MonolithicNvfp4ExpertWeights> storage;
  std::vector<FusedNvfp4WeightView> views;
  std::optional<HostNvfp4Matrix> host_packed;
};

GemmDescriptor MakeNvfp4Descriptor(
    const std::string& tensor_name,
    const std::string& op_class,
    const HostNvfp4Matrix& packed_matrix) {
  GemmDescriptor descriptor;
  descriptor.tensor_name = tensor_name;
  descriptor.op_class = op_class;
  descriptor.kernel_family = GemmKernelFamily::kCublasLtNvfp4BlockScaled;
  descriptor.output_rows = packed_matrix.rows;
  descriptor.input_cols = packed_matrix.cols;
  descriptor.storage_dtype = "nvfp4_e2m1";
  descriptor.compute_dtype = "fp32_accum";
  descriptor.layout_tag = "cublaslt_fp4_tn_v1";
  descriptor.alignment_bytes = 16;
  descriptor.packed_data = packed_matrix.packed_data();
  descriptor.packed_nbytes = packed_matrix.packed_nbytes();
  descriptor.block_scales_data = packed_matrix.block_scales_data();
  descriptor.block_scales_nbytes = packed_matrix.block_scales_nbytes();
  descriptor.tensor_scale_data = packed_matrix.tensor_scale_data();
  descriptor.tensor_scale_nbytes = packed_matrix.tensor_scale_nbytes();
  return descriptor;
}

__device__ float DecodeFp4Nibble(std::uint8_t nibble) {
  __nv_fp4_e2m1 value;
  value.__x = nibble & 0x0F;
  return static_cast<float>(value);
}

__device__ float DecodeFp8Byte(std::uint8_t value) {
  __nv_fp8_e4m3 decoded;
  decoded.__x = value;
  return static_cast<float>(decoded);
}

__global__ void PermutedExpertRowCoopKernel(
    const float* input,
    const int* expert_offsets,
    const int* cta_expert_ids,
    const int* cta_row_starts,
    const int* cta_output_row_bases,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  __shared__ float input_tile[kPermutedTokenTile][kPermutedPairsPerStep * 2];

  const int warp_index = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int cta_index = static_cast<int>(blockIdx.x);
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int output_row_base = cta_output_row_bases[cta_index];
  const int output_row = output_row_base + warp_index;
  if (expert_index < 0 || static_cast<std::size_t>(expert_index) >= kRoutedExperts) {
    return;
  }
  if (warp_index >= kPermutedTokenTile ||
      static_cast<std::size_t>(output_row) >= output_rows_per_expert) {
    return;
  }

  const auto weight = weights[static_cast<std::size_t>(expert_index)];
  const int expert_end = expert_offsets[expert_index + 1];
  const int valid_rows = expert_end - row_start;
  if (valid_rows <= 0) {
    return;
  }
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row =
      weight.input_cols / kNvfp4BlockWidth;
  float accum[kPermutedTokenTile] = {0.0f};

  for (std::size_t pair_base = 0; pair_base < pairs_per_row;
       pair_base += kPermutedPairsPerStep) {
    const std::size_t remaining_pairs = pairs_per_row - pair_base;
    const int pairs_this_step = static_cast<int>(
        remaining_pairs < static_cast<std::size_t>(kPermutedPairsPerStep)
            ? remaining_pairs
            : static_cast<std::size_t>(kPermutedPairsPerStep));
    const int clamped_valid_rows =
        valid_rows < kPermutedTokenTile ? valid_rows : kPermutedTokenTile;
    const int values_this_step = clamped_valid_rows * pairs_this_step * 2;
    for (int linear_index = static_cast<int>(threadIdx.x);
         linear_index < values_this_step;
         linear_index += static_cast<int>(blockDim.x)) {
      const int token_index = linear_index / (pairs_this_step * 2);
      const int within_token = linear_index % (pairs_this_step * 2);
      const int pair_offset = within_token / 2;
      const int value_offset = within_token % 2;
      const std::size_t input_row = static_cast<std::size_t>(row_start + token_index);
      const std::size_t col =
          (pair_base + static_cast<std::size_t>(pair_offset)) * 2u +
          static_cast<std::size_t>(value_offset);
      input_tile[token_index][within_token] =
          input[input_row * weight.input_cols + col];
    }
    __syncthreads();

    if (static_cast<std::size_t>(output_row) < output_rows_per_expert) {
      const std::size_t packed_row_offset =
          static_cast<std::size_t>(output_row) * pairs_per_row;
      const std::size_t scale_row_offset =
          static_cast<std::size_t>(output_row) * blocks_per_row;
      const float tensor_scale = *weight.tensor_scale_data;
      const std::size_t pair_index = pair_base + static_cast<std::size_t>(lane);
      if (lane < pairs_this_step) {
        const std::size_t block = pair_index / 8u;
        const float block_scale =
            DecodeFp8Byte(weight.block_scales_data[scale_row_offset + block]) *
            tensor_scale;
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const float w0 = DecodeFp4Nibble(packed & 0x0Fu) * block_scale;
        const float w1 = DecodeFp4Nibble((packed >> 4) & 0x0Fu) * block_scale;
        for (int token_index = 0; token_index < clamped_valid_rows; ++token_index) {
          const int value_index = static_cast<int>(lane) * 2;
          accum[token_index] += input_tile[token_index][value_index] * w0;
          accum[token_index] += input_tile[token_index][value_index + 1] * w1;
        }
      }
    }
    __syncthreads();
  }

  if (static_cast<std::size_t>(output_row) >= output_rows_per_expert) {
    return;
  }

  const int clamped_valid_rows =
      valid_rows < kPermutedTokenTile ? valid_rows : kPermutedTokenTile;
  for (int token_index = 0; token_index < clamped_valid_rows; ++token_index) {
    for (int stride = 16; stride > 0; stride >>= 1) {
      accum[token_index] += __shfl_down_sync(0xffffffffu, accum[token_index], stride);
    }
  }

  if (lane == 0) {
    for (int token_index = 0; token_index < clamped_valid_rows; ++token_index) {
      const std::size_t input_row = static_cast<std::size_t>(row_start + token_index);
      output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row)] =
          accum[token_index];
    }
  }
}

__global__ void PermutedExpertWideRowCoopKernel(
    const float* input,
    const int* expert_offsets,
    const int* cta_expert_ids,
    const int* cta_row_starts,
    const int* cta_output_row_bases,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  __shared__ float input_tile[kPermutedTokenTile][kPermutedPairsPerStep * 2];

  const int warp_index = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int cta_index = static_cast<int>(blockIdx.x);
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int output_row_base = cta_output_row_bases[cta_index];
  const int output_row = output_row_base + warp_index;
  if (expert_index < 0 || static_cast<std::size_t>(expert_index) >= kRoutedExperts) {
    return;
  }
  if (warp_index >= kPermutedWideOutputTile ||
      static_cast<std::size_t>(output_row) >= output_rows_per_expert) {
    return;
  }

  const auto weight = weights[static_cast<std::size_t>(expert_index)];
  const int expert_end = expert_offsets[expert_index + 1];
  const int valid_rows = expert_end - row_start;
  if (valid_rows <= 0) {
    return;
  }
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row =
      weight.input_cols / kNvfp4BlockWidth;
  float accum[kPermutedTokenTile] = {0.0f};

  for (std::size_t pair_base = 0; pair_base < pairs_per_row;
       pair_base += kPermutedPairsPerStep) {
    const std::size_t remaining_pairs = pairs_per_row - pair_base;
    const int pairs_this_step = static_cast<int>(
        remaining_pairs < static_cast<std::size_t>(kPermutedPairsPerStep)
            ? remaining_pairs
            : static_cast<std::size_t>(kPermutedPairsPerStep));
    const int clamped_valid_rows =
        valid_rows < kPermutedTokenTile ? valid_rows : kPermutedTokenTile;
    const int values_this_step = clamped_valid_rows * pairs_this_step * 2;
    for (int linear_index = static_cast<int>(threadIdx.x);
         linear_index < values_this_step;
         linear_index += static_cast<int>(blockDim.x)) {
      const int token_index = linear_index / (pairs_this_step * 2);
      const int within_token = linear_index % (pairs_this_step * 2);
      const int pair_offset = within_token / 2;
      const int value_offset = within_token % 2;
      const std::size_t input_row = static_cast<std::size_t>(row_start + token_index);
      const std::size_t col =
          (pair_base + static_cast<std::size_t>(pair_offset)) * 2u +
          static_cast<std::size_t>(value_offset);
      input_tile[token_index][within_token] =
          input[input_row * weight.input_cols + col];
    }
    __syncthreads();

    if (static_cast<std::size_t>(output_row) < output_rows_per_expert) {
      const std::size_t packed_row_offset =
          static_cast<std::size_t>(output_row) * pairs_per_row;
      const std::size_t scale_row_offset =
          static_cast<std::size_t>(output_row) * blocks_per_row;
      const float tensor_scale = *weight.tensor_scale_data;
      const std::size_t pair_index = pair_base + static_cast<std::size_t>(lane);
      if (lane < pairs_this_step) {
        const std::size_t block = pair_index / 8u;
        const float block_scale =
            DecodeFp8Byte(weight.block_scales_data[scale_row_offset + block]) *
            tensor_scale;
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const float w0 = DecodeFp4Nibble(packed & 0x0Fu) * block_scale;
        const float w1 = DecodeFp4Nibble((packed >> 4) & 0x0Fu) * block_scale;
        for (int token_index = 0; token_index < clamped_valid_rows; ++token_index) {
          const int value_index = static_cast<int>(lane) * 2;
          accum[token_index] += input_tile[token_index][value_index] * w0;
          accum[token_index] += input_tile[token_index][value_index + 1] * w1;
        }
      }
    }
    __syncthreads();
  }

  if (static_cast<std::size_t>(output_row) >= output_rows_per_expert) {
    return;
  }

  const int clamped_valid_rows =
      valid_rows < kPermutedTokenTile ? valid_rows : kPermutedTokenTile;
  for (int token_index = 0; token_index < clamped_valid_rows; ++token_index) {
    for (int stride = 16; stride > 0; stride >>= 1) {
      accum[token_index] += __shfl_down_sync(0xffffffffu, accum[token_index], stride);
    }
  }

  if (lane == 0) {
    for (int token_index = 0; token_index < clamped_valid_rows; ++token_index) {
      const std::size_t input_row = static_cast<std::size_t>(row_start + token_index);
      output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row)] =
          accum[token_index];
    }
  }
}

__global__ void LaunchPlannedFp32AccumKernel(
    const float* input,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    int output_row_tile_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  __shared__ float input_tile[kPermutedTokenTile][kPermutedPairsPerStep * 2];

  const int task_index = static_cast<int>(blockIdx.x);
  const int warp_index = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int exact_cta_count = cta_count[0];
  const int exact_task_count = exact_cta_count * output_row_tile_count;
  if (task_index >= exact_task_count) {
    return;
  }

  const int cta_index = task_index / output_row_tile_count;
  const int output_row_tile = task_index % output_row_tile_count;
  const int output_row = output_row_tile * kPermutedOutputTile + warp_index;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  if (expert_index < 0 ||
      warp_index >= kPermutedOutputTile ||
      static_cast<std::size_t>(output_row) >= output_rows_per_expert ||
      valid_rows <= 0) {
    return;
  }

  const auto weight = weights[static_cast<std::size_t>(expert_index)];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / kNvfp4BlockWidth;
  const std::size_t packed_row_offset =
      static_cast<std::size_t>(output_row) * pairs_per_row;
  const std::size_t scale_row_offset =
      static_cast<std::size_t>(output_row) * blocks_per_row;
  const float tensor_scale = *weight.tensor_scale_data;
  float accum[kPermutedTokenTile] = {0.0f};

  for (std::size_t pair_base = 0; pair_base < pairs_per_row;
       pair_base += kPermutedPairsPerStep) {
    const std::size_t remaining_pairs = pairs_per_row - pair_base;
    const int pairs_this_step = static_cast<int>(
        remaining_pairs < static_cast<std::size_t>(kPermutedPairsPerStep)
            ? remaining_pairs
            : static_cast<std::size_t>(kPermutedPairsPerStep));
    const int values_this_step = valid_rows * pairs_this_step * 2;
    for (int linear_index = static_cast<int>(threadIdx.x);
         linear_index < values_this_step;
         linear_index += static_cast<int>(blockDim.x)) {
      const int token_index = linear_index / (pairs_this_step * 2);
      const int within_token = linear_index % (pairs_this_step * 2);
      const int pair_offset = within_token / 2;
      const int value_offset = within_token % 2;
      const std::size_t input_row =
          static_cast<std::size_t>(row_start + token_index);
      const std::size_t col =
          (pair_base + static_cast<std::size_t>(pair_offset)) * 2u +
          static_cast<std::size_t>(value_offset);
      input_tile[token_index][within_token] =
          input[input_row * weight.input_cols + col];
    }
    __syncthreads();

    const std::size_t pair_index = pair_base + static_cast<std::size_t>(lane);
    if (lane < pairs_this_step) {
      const std::size_t block = pair_index / 8u;
      const float block_scale =
          DecodeFp8Byte(weight.block_scales_data[scale_row_offset + block]) *
          tensor_scale;
      const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
      const float w0 = DecodeFp4Nibble(packed & 0x0Fu) * block_scale;
      const float w1 = DecodeFp4Nibble((packed >> 4) & 0x0Fu) * block_scale;
      for (int token_index = 0; token_index < valid_rows; ++token_index) {
        const int value_index = lane * 2;
        accum[token_index] += input_tile[token_index][value_index] * w0;
        accum[token_index] += input_tile[token_index][value_index + 1] * w1;
      }
    }
    __syncthreads();
  }

  for (int token_index = 0; token_index < valid_rows; ++token_index) {
    for (int stride = 16; stride > 0; stride >>= 1) {
      accum[token_index] += __shfl_down_sync(0xffffffffu, accum[token_index], stride);
    }
  }

  if (lane == 0) {
    for (int token_index = 0; token_index < valid_rows; ++token_index) {
      const std::size_t input_row =
          static_cast<std::size_t>(row_start + token_index);
      output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row)] =
          accum[token_index];
    }
  }
}

__global__ void LaunchPlannedWideFp32AccumKernel(
    const float* input,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    int output_row_tile_count,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  __shared__ float input_tile[kPermutedTokenTile][kPermutedPairsPerStep * 2];

  const int task_index = static_cast<int>(blockIdx.x);
  const int warp_index = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int exact_cta_count = cta_count[0];
  const int exact_task_count = exact_cta_count * output_row_tile_count;
  if (task_index >= exact_task_count) {
    return;
  }

  const int cta_index = task_index / output_row_tile_count;
  const int output_row_tile = task_index % output_row_tile_count;
  const int output_row = output_row_tile * kPermutedWideOutputTile + warp_index;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  if (expert_index < 0 ||
      warp_index >= kPermutedWideOutputTile ||
      static_cast<std::size_t>(output_row) >= output_rows_per_expert ||
      valid_rows <= 0) {
    return;
  }

  const auto weight = weights[static_cast<std::size_t>(expert_index)];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / kNvfp4BlockWidth;
  const std::size_t packed_row_offset =
      static_cast<std::size_t>(output_row) * pairs_per_row;
  const std::size_t scale_row_offset =
      static_cast<std::size_t>(output_row) * blocks_per_row;
  const float tensor_scale = *weight.tensor_scale_data;
  float accum[kPermutedTokenTile] = {0.0f};

  for (std::size_t pair_base = 0; pair_base < pairs_per_row;
       pair_base += kPermutedPairsPerStep) {
    const std::size_t remaining_pairs = pairs_per_row - pair_base;
    const int pairs_this_step = static_cast<int>(
        remaining_pairs < static_cast<std::size_t>(kPermutedPairsPerStep)
            ? remaining_pairs
            : static_cast<std::size_t>(kPermutedPairsPerStep));
    const int values_this_step = valid_rows * pairs_this_step * 2;
    for (int linear_index = static_cast<int>(threadIdx.x);
         linear_index < values_this_step;
         linear_index += static_cast<int>(blockDim.x)) {
      const int token_index = linear_index / (pairs_this_step * 2);
      const int within_token = linear_index % (pairs_this_step * 2);
      const int pair_offset = within_token / 2;
      const int value_offset = within_token % 2;
      const std::size_t input_row =
          static_cast<std::size_t>(row_start + token_index);
      const std::size_t col =
          (pair_base + static_cast<std::size_t>(pair_offset)) * 2u +
          static_cast<std::size_t>(value_offset);
      input_tile[token_index][within_token] =
          input[input_row * weight.input_cols + col];
    }
    __syncthreads();

    const std::size_t pair_index = pair_base + static_cast<std::size_t>(lane);
    if (lane < pairs_this_step) {
      const std::size_t block = pair_index / 8u;
      const float block_scale =
          DecodeFp8Byte(weight.block_scales_data[scale_row_offset + block]) *
          tensor_scale;
      const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
      const float w0 = DecodeFp4Nibble(packed & 0x0Fu) * block_scale;
      const float w1 = DecodeFp4Nibble((packed >> 4) & 0x0Fu) * block_scale;
      for (int token_index = 0; token_index < valid_rows; ++token_index) {
        const int value_index = lane * 2;
        accum[token_index] += input_tile[token_index][value_index] * w0;
        accum[token_index] += input_tile[token_index][value_index + 1] * w1;
      }
    }
    __syncthreads();
  }

  for (int token_index = 0; token_index < valid_rows; ++token_index) {
    for (int stride = 16; stride > 0; stride >>= 1) {
      accum[token_index] += __shfl_down_sync(0xffffffffu, accum[token_index], stride);
    }
  }

  if (lane == 0) {
    for (int token_index = 0; token_index < valid_rows; ++token_index) {
      const std::size_t input_row =
          static_cast<std::size_t>(row_start + token_index);
      output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row)] =
          accum[token_index];
    }
  }
}

__global__ void LaunchPlanExactTaskTile8Kernel(
    const float* input,
    const int* task_expert_ids,
    const int* task_row_starts,
    const int* task_valid_rows,
    const int* task_output_row_bases,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  __shared__ float input_tile[kPermutedTokenTile][kPermutedPairsPerStep * 2];

  const int task_index = static_cast<int>(blockIdx.x);
  const int warp_index = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int expert_index = task_expert_ids[task_index];
  const int row_start = task_row_starts[task_index];
  const int valid_rows = task_valid_rows[task_index];
  const int output_row = task_output_row_bases[task_index] + warp_index;
  if (expert_index < 0 ||
      warp_index >= kPermutedWideOutputTile ||
      static_cast<std::size_t>(output_row) >= output_rows_per_expert ||
      valid_rows <= 0) {
    return;
  }

  const auto weight = weights[static_cast<std::size_t>(expert_index)];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / kNvfp4BlockWidth;
  const std::size_t packed_row_offset =
      static_cast<std::size_t>(output_row) * pairs_per_row;
  const std::size_t scale_row_offset =
      static_cast<std::size_t>(output_row) * blocks_per_row;
  const float tensor_scale = *weight.tensor_scale_data;
  float accum[kPermutedTokenTile] = {0.0f};

  for (std::size_t pair_base = 0; pair_base < pairs_per_row;
       pair_base += kPermutedPairsPerStep) {
    const std::size_t remaining_pairs = pairs_per_row - pair_base;
    const int pairs_this_step = static_cast<int>(
        remaining_pairs < static_cast<std::size_t>(kPermutedPairsPerStep)
            ? remaining_pairs
            : static_cast<std::size_t>(kPermutedPairsPerStep));
    const int values_this_step = valid_rows * pairs_this_step * 2;
    for (int linear_index = static_cast<int>(threadIdx.x);
         linear_index < values_this_step;
         linear_index += static_cast<int>(blockDim.x)) {
      const int token_index = linear_index / (pairs_this_step * 2);
      const int within_token = linear_index % (pairs_this_step * 2);
      const int pair_offset = within_token / 2;
      const int value_offset = within_token % 2;
      const std::size_t input_row =
          static_cast<std::size_t>(row_start + token_index);
      const std::size_t col =
          (pair_base + static_cast<std::size_t>(pair_offset)) * 2u +
          static_cast<std::size_t>(value_offset);
      input_tile[token_index][within_token] =
          input[input_row * weight.input_cols + col];
    }
    __syncthreads();

    const std::size_t pair_index = pair_base + static_cast<std::size_t>(lane);
    if (lane < pairs_this_step) {
      const std::size_t block = pair_index / 8u;
      const float block_scale =
          DecodeFp8Byte(weight.block_scales_data[scale_row_offset + block]) *
          tensor_scale;
      const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
      const float w0 = DecodeFp4Nibble(packed & 0x0Fu) * block_scale;
      const float w1 = DecodeFp4Nibble((packed >> 4) & 0x0Fu) * block_scale;
      for (int token_index = 0; token_index < valid_rows; ++token_index) {
        const int value_index = lane * 2;
        accum[token_index] += input_tile[token_index][value_index] * w0;
        accum[token_index] += input_tile[token_index][value_index + 1] * w1;
      }
    }
    __syncthreads();
  }

  for (int token_index = 0; token_index < valid_rows; ++token_index) {
    for (int stride = 16; stride > 0; stride >>= 1) {
      accum[token_index] += __shfl_down_sync(0xffffffffu, accum[token_index], stride);
    }
  }

  if (lane == 0) {
    for (int token_index = 0; token_index < valid_rows; ++token_index) {
      const std::size_t input_row =
          static_cast<std::size_t>(row_start + token_index);
      output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row)] =
          accum[token_index];
    }
  }
}

__global__ void LaunchPlanGroupedTile32Kernel(
    const float* input,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  __shared__ float input_tile[kPermutedTokenTile][kPermutedPairsPerStep * 2];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int warp_index = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count) {
    return;
  }

  const int output_row_base =
      static_cast<int>(blockIdx.x) * kPermutedGroupedOutputTile + warp_index;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  if (expert_index < 0 || valid_rows <= 0) {
    return;
  }

  const auto weight = weights[static_cast<std::size_t>(expert_index)];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / kNvfp4BlockWidth;
  const float tensor_scale = *weight.tensor_scale_data;
  float accum[kPermutedGroupedRowsPerWarp][kPermutedTokenTile] = {{0.0f}};

  for (std::size_t pair_base = 0; pair_base < pairs_per_row;
       pair_base += kPermutedPairsPerStep) {
    const std::size_t remaining_pairs = pairs_per_row - pair_base;
    const int pairs_this_step = static_cast<int>(
        remaining_pairs < static_cast<std::size_t>(kPermutedPairsPerStep)
            ? remaining_pairs
            : static_cast<std::size_t>(kPermutedPairsPerStep));
    const int values_this_step = valid_rows * pairs_this_step * 2;
    for (int linear_index = static_cast<int>(threadIdx.x);
         linear_index < values_this_step;
         linear_index += static_cast<int>(blockDim.x)) {
      const int token_index = linear_index / (pairs_this_step * 2);
      const int within_token = linear_index % (pairs_this_step * 2);
      const int pair_offset = within_token / 2;
      const int value_offset = within_token % 2;
      const std::size_t input_row =
          static_cast<std::size_t>(row_start + token_index);
      const std::size_t col =
          (pair_base + static_cast<std::size_t>(pair_offset)) * 2u +
          static_cast<std::size_t>(value_offset);
      input_tile[token_index][within_token] =
          input[input_row * weight.input_cols + col];
    }
    __syncthreads();

    const std::size_t pair_index = pair_base + static_cast<std::size_t>(lane);
    if (lane < pairs_this_step) {
      const std::size_t block = pair_index / 8u;
      for (int row_group = 0; row_group < kPermutedGroupedRowsPerWarp; ++row_group) {
        const int output_row = output_row_base + row_group * kPermutedWideOutputTile;
        if (static_cast<std::size_t>(output_row) >= output_rows_per_expert) {
          continue;
        }
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * blocks_per_row;
        const float block_scale =
            DecodeFp8Byte(weight.block_scales_data[scale_row_offset + block]) *
            tensor_scale;
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const float w0 = DecodeFp4Nibble(packed & 0x0Fu) * block_scale;
        const float w1 = DecodeFp4Nibble((packed >> 4) & 0x0Fu) * block_scale;
        for (int token_index = 0; token_index < valid_rows; ++token_index) {
          const int value_index = lane * 2;
          accum[row_group][token_index] += input_tile[token_index][value_index] * w0;
          accum[row_group][token_index] += input_tile[token_index][value_index + 1] * w1;
        }
      }
    }
    __syncthreads();
  }

  for (int row_group = 0; row_group < kPermutedGroupedRowsPerWarp; ++row_group) {
    const int output_row = output_row_base + row_group * kPermutedWideOutputTile;
    if (static_cast<std::size_t>(output_row) >= output_rows_per_expert) {
      continue;
    }
    for (int token_index = 0; token_index < valid_rows; ++token_index) {
      for (int stride = 16; stride > 0; stride >>= 1) {
        accum[row_group][token_index] +=
            __shfl_down_sync(0xffffffffu, accum[row_group][token_index], stride);
      }
    }
    if (lane == 0) {
      for (int token_index = 0; token_index < valid_rows; ++token_index) {
        const std::size_t input_row =
            static_cast<std::size_t>(row_start + token_index);
        output[input_row * output_rows_per_expert + static_cast<std::size_t>(output_row)] =
            accum[row_group][token_index];
      }
    }
  }
}

__global__ void LaunchPlanCooperativeMxNKernel(
    const float* input,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  __shared__ float input_tile[kPermutedTokenTile][kPermutedPairsPerStep * 2];
  __shared__ float weight_tile[kPermutedCoopOutputTile][kPermutedPairsPerStep * 2];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int token_index = tid / kPermutedCoopOutputTile;
  const int output_lane = tid % kPermutedCoopOutputTile;
  const int output_row_base = static_cast<int>(blockIdx.x) * kPermutedCoopOutputTile;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  if (expert_index < 0 || valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert) {
    return;
  }

  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kPermutedCoopOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kPermutedCoopOutputTile));
  const auto weight = weights[static_cast<std::size_t>(expert_index)];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / kNvfp4BlockWidth;
  const float tensor_scale = *weight.tensor_scale_data;
  float accum = 0.0f;

  for (std::size_t pair_base = 0; pair_base < pairs_per_row;
       pair_base += kPermutedPairsPerStep) {
    const std::size_t remaining_pairs = pairs_per_row - pair_base;
    const int pairs_this_step = static_cast<int>(
        remaining_pairs < static_cast<std::size_t>(kPermutedPairsPerStep)
            ? remaining_pairs
            : static_cast<std::size_t>(kPermutedPairsPerStep));
    const int values_this_step = valid_rows * pairs_this_step * 2;
    for (int linear_index = tid;
         linear_index < values_this_step;
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_token = linear_index / (pairs_this_step * 2);
      const int within_token = linear_index % (pairs_this_step * 2);
      const int pair_offset = within_token / 2;
      const int value_offset = within_token % 2;
      const std::size_t input_row =
          static_cast<std::size_t>(row_start + tile_token);
      const std::size_t col =
          (pair_base + static_cast<std::size_t>(pair_offset)) * 2u +
          static_cast<std::size_t>(value_offset);
      input_tile[tile_token][within_token] =
          input[input_row * weight.input_cols + col];
    }

    const int packed_values_this_step = output_rows_this_tile * pairs_this_step;
    for (int linear_index = tid;
         linear_index < packed_values_this_step;
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / pairs_this_step;
      const int pair_offset = linear_index % pairs_this_step;
      const int output_row = output_row_base + tile_output_row;
      const std::size_t pair_index = pair_base + static_cast<std::size_t>(pair_offset);
      const std::size_t block = pair_index / 8u;
      const std::size_t packed_row_offset =
          static_cast<std::size_t>(output_row) * pairs_per_row;
      const std::size_t scale_row_offset =
          static_cast<std::size_t>(output_row) * blocks_per_row;
      const float block_scale =
          DecodeFp8Byte(weight.block_scales_data[scale_row_offset + block]) * tensor_scale;
      const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
      weight_tile[tile_output_row][pair_offset * 2] =
          DecodeFp4Nibble(packed & 0x0Fu) * block_scale;
      weight_tile[tile_output_row][pair_offset * 2 + 1] =
          DecodeFp4Nibble((packed >> 4) & 0x0Fu) * block_scale;
    }
    __syncthreads();

    if (token_index < valid_rows && output_lane < output_rows_this_tile) {
      for (int value_index = 0; value_index < (pairs_this_step * 2); ++value_index) {
        accum += input_tile[token_index][value_index] * weight_tile[output_lane][value_index];
      }
    }
    __syncthreads();
  }

  if (token_index < valid_rows && output_lane < output_rows_this_tile) {
    const std::size_t output_row = static_cast<std::size_t>(output_row_base + output_lane);
    const std::size_t input_row = static_cast<std::size_t>(row_start + token_index);
    output[input_row * output_rows_per_expert + output_row] = accum;
  }
}

__global__ void LaunchPlanMicroTile2Kernel(
    const float* input,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  __shared__ float input_tile[kPermutedTokenTile][kPermutedPairsPerStep * 2];
  __shared__ float weight_tile[kPermutedCoopOutputTile][kPermutedPairsPerStep * 2];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int token_index = tid / kPermutedMicroTileThreadN;
  const int output_pair_index = tid % kPermutedMicroTileThreadN;
  const int output_row_base = static_cast<int>(blockIdx.x) * kPermutedCoopOutputTile;
  const int output_lane0 = output_pair_index * kPermutedMicroTileOutputsPerThread;
  const int output_lane1 = output_lane0 + 1;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  if (expert_index < 0 || valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert) {
    return;
  }

  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kPermutedCoopOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kPermutedCoopOutputTile));
  const auto weight = weights[static_cast<std::size_t>(expert_index)];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / kNvfp4BlockWidth;
  const float tensor_scale = *weight.tensor_scale_data;
  float accum0 = 0.0f;
  float accum1 = 0.0f;

  for (std::size_t pair_base = 0; pair_base < pairs_per_row;
       pair_base += kPermutedPairsPerStep) {
    const std::size_t remaining_pairs = pairs_per_row - pair_base;
    const int pairs_this_step = static_cast<int>(
        remaining_pairs < static_cast<std::size_t>(kPermutedPairsPerStep)
            ? remaining_pairs
            : static_cast<std::size_t>(kPermutedPairsPerStep));
    const int values_this_step = valid_rows * pairs_this_step * 2;
    for (int linear_index = tid;
         linear_index < values_this_step;
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_token = linear_index / (pairs_this_step * 2);
      const int within_token = linear_index % (pairs_this_step * 2);
      const int pair_offset = within_token / 2;
      const int value_offset = within_token % 2;
      const std::size_t input_row =
          static_cast<std::size_t>(row_start + tile_token);
      const std::size_t col =
          (pair_base + static_cast<std::size_t>(pair_offset)) * 2u +
          static_cast<std::size_t>(value_offset);
      input_tile[tile_token][within_token] =
          input[input_row * weight.input_cols + col];
    }

    const int packed_values_this_step = output_rows_this_tile * pairs_this_step;
    for (int linear_index = tid;
         linear_index < packed_values_this_step;
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / pairs_this_step;
      const int pair_offset = linear_index % pairs_this_step;
      const int output_row = output_row_base + tile_output_row;
      const std::size_t pair_index = pair_base + static_cast<std::size_t>(pair_offset);
      const std::size_t block = pair_index / 8u;
      const std::size_t packed_row_offset =
          static_cast<std::size_t>(output_row) * pairs_per_row;
      const std::size_t scale_row_offset =
          static_cast<std::size_t>(output_row) * blocks_per_row;
      const float block_scale =
          DecodeFp8Byte(weight.block_scales_data[scale_row_offset + block]) * tensor_scale;
      const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
      weight_tile[tile_output_row][pair_offset * 2] =
          DecodeFp4Nibble(packed & 0x0Fu) * block_scale;
      weight_tile[tile_output_row][pair_offset * 2 + 1] =
          DecodeFp4Nibble((packed >> 4) & 0x0Fu) * block_scale;
    }
    __syncthreads();

    if (token_index < valid_rows) {
      for (int value_index = 0; value_index < (pairs_this_step * 2); ++value_index) {
        const float input_value = input_tile[token_index][value_index];
        if (output_lane0 < output_rows_this_tile) {
          accum0 += input_value * weight_tile[output_lane0][value_index];
        }
        if (output_lane1 < output_rows_this_tile) {
          accum1 += input_value * weight_tile[output_lane1][value_index];
        }
      }
    }
    __syncthreads();
  }

  if (token_index < valid_rows) {
    const std::size_t input_row = static_cast<std::size_t>(row_start + token_index);
    if (output_lane0 < output_rows_this_tile) {
      const std::size_t output_row = static_cast<std::size_t>(output_row_base + output_lane0);
      output[input_row * output_rows_per_expert + output_row] = accum0;
    }
    if (output_lane1 < output_rows_this_tile) {
      const std::size_t output_row = static_cast<std::size_t>(output_row_base + output_lane1);
      output[input_row * output_rows_per_expert + output_row] = accum1;
    }
  }
}

__global__ void LaunchPlanWmmaBf16Kernel(
    const float* input,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  __shared__ __nv_bfloat16 a_tile[kWmmaTileM][kWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kWmmaTileK][kWmmaOutputTile];
  __shared__ float c_tile[kWmmaTileM][kWmmaOutputTile];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count) {
    return;
  }

  const int warp_id = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int output_row_base = static_cast<int>(blockIdx.x) * kWmmaOutputTile;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  if (expert_index < 0 || valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert) {
    return;
  }

  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kWmmaOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kWmmaOutputTile));
  const auto weight = weights[static_cast<std::size_t>(expert_index)];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / kNvfp4BlockWidth;
  const float tensor_scale = *weight.tensor_scale_data;

  wmma::fragment<wmma::accumulator, kWmmaTileM, kWmmaTileN, kWmmaTileK, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols; k_base += kWmmaTileK) {
    for (int linear_index = static_cast<int>(threadIdx.x);
         linear_index < (kWmmaTileM * kWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_row = linear_index / kWmmaTileK;
      const int tile_col = linear_index % kWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_row < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_col)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_row);
        const float input_value =
            input[input_row * weight.input_cols + (k_base + static_cast<std::size_t>(tile_col))];
        value = __float2bfloat16(input_value);
      }
      a_tile[tile_row][tile_col] = value;
    }

    const int pairs_this_step = static_cast<int>(kWmmaTileK / 2);
    for (int linear_index = static_cast<int>(threadIdx.x);
         linear_index < (output_rows_this_tile * pairs_this_step);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / pairs_this_step;
      const int pair_offset = linear_index % pairs_this_step;
      const int output_row = output_row_base + tile_output_row;
      const std::size_t pair_index =
          (k_base / 2u) + static_cast<std::size_t>(pair_offset);
      const std::size_t block = pair_index / 8u;
      const std::size_t packed_row_offset =
          static_cast<std::size_t>(output_row) * pairs_per_row;
      const std::size_t scale_row_offset =
          static_cast<std::size_t>(output_row) * blocks_per_row;
      const float block_scale =
          DecodeFp8Byte(weight.block_scales_data[scale_row_offset + block]) * tensor_scale;
      const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
      b_tile[pair_offset * 2][tile_output_row] =
          __float2bfloat16(DecodeFp4Nibble(packed & 0x0Fu) * block_scale);
      b_tile[pair_offset * 2 + 1][tile_output_row] =
          __float2bfloat16(DecodeFp4Nibble((packed >> 4) & 0x0Fu) * block_scale);
    }
    for (int linear_index = static_cast<int>(threadIdx.x) + (output_rows_this_tile * pairs_this_step);
         linear_index < (kWmmaTileK * kWmmaOutputTile);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kWmmaOutputTile;
      const int tile_n = linear_index % kWmmaOutputTile;
      b_tile[tile_k][tile_n] = __float2bfloat16(0.0f);
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, kWmmaTileM, kWmmaTileN, kWmmaTileK, __nv_bfloat16, wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, kWmmaTileM, kWmmaTileN, kWmmaTileK, __nv_bfloat16, wmma::row_major>
        b_frag;
    const int warp_col = warp_id * kWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[0][0], kWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][warp_col], kWmmaOutputTile);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_col = warp_id * kWmmaTileN;
  wmma::store_matrix_sync(&c_tile[0][warp_col], c_frag, kWmmaOutputTile, wmma::mem_row_major);
  __syncthreads();

  for (int linear_index = static_cast<int>(threadIdx.x);
       linear_index < (valid_rows * output_rows_this_tile);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_row = linear_index / output_rows_this_tile;
    const int tile_col = linear_index % output_rows_this_tile;
    const std::size_t input_row = static_cast<std::size_t>(row_start + tile_row);
    const std::size_t output_row = static_cast<std::size_t>(output_row_base + tile_col);
    output[input_row * output_rows_per_expert + output_row] = c_tile[tile_row][tile_col];
  }
}

__global__ void LaunchPlanWmmaBf16TransposedKernel(
    const float* input,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  __shared__ __nv_bfloat16 a_tile[kWmmaOutputTile][kWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kWmmaTileK][kWmmaTileM];
  __shared__ float c_tile[kWmmaOutputTile][kWmmaTileM];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count) {
    return;
  }

  const int warp_id = static_cast<int>(threadIdx.x) / 32;
  const int output_row_base = static_cast<int>(blockIdx.x) * kWmmaOutputTile;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  if (expert_index < 0 || valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert ||
      warp_id >= kWmmaWarpsPerBlock) {
    return;
  }

  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kWmmaOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kWmmaOutputTile));
  const auto weight = weights[static_cast<std::size_t>(expert_index)];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / kNvfp4BlockWidth;
  const float tensor_scale = *weight.tensor_scale_data;

  wmma::fragment<wmma::accumulator, kWmmaTileM, kWmmaTileN, kWmmaTileK, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols; k_base += kWmmaTileK) {
    for (int linear_index = static_cast<int>(threadIdx.x);
         linear_index < (kWmmaOutputTile * kWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kWmmaTileK;
      const int tile_k = linear_index % kWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * blocks_per_row;
        const float block_scale =
            DecodeFp8Byte(weight.block_scales_data[scale_row_offset + block]) * tensor_scale;
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(DecodeFp4Nibble(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }

    for (int linear_index = static_cast<int>(threadIdx.x);
         linear_index < (kWmmaTileK * kWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kWmmaTileM;
      const int tile_token = linear_index % kWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const float input_value =
            input[input_row * weight.input_cols + (k_base + static_cast<std::size_t>(tile_k))];
        value = __float2bfloat16(input_value);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, kWmmaTileM, kWmmaTileN, kWmmaTileK, __nv_bfloat16, wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, kWmmaTileM, kWmmaTileN, kWmmaTileK, __nv_bfloat16, wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kWmmaTileN;
  wmma::store_matrix_sync(&c_tile[warp_row][0], c_frag, kWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  for (int linear_index = static_cast<int>(threadIdx.x);
       linear_index < (output_rows_this_tile * valid_rows);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_output_row = linear_index / valid_rows;
    const int tile_token = linear_index % valid_rows;
    const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
    const std::size_t output_row = static_cast<std::size_t>(output_row_base + tile_output_row);
    output[input_row * output_rows_per_expert + output_row] =
        c_tile[tile_output_row][tile_token];
  }
}

__global__ void LaunchPlanWmmaPackedInputTransposedKernel(
    const std::uint8_t* packed_input,
    const std::uint8_t* input_block_scales,
    const float* input_tensor_scale_data,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  __shared__ __nv_bfloat16 a_tile[kWmmaOutputTile][kWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kWmmaTileK][kWmmaTileM];
  __shared__ float c_tile[kWmmaOutputTile][kWmmaTileM];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count || packed_input == nullptr ||
      input_block_scales == nullptr || input_tensor_scale_data == nullptr) {
    return;
  }

  const int warp_id = static_cast<int>(threadIdx.x) / 32;
  const int output_row_base = static_cast<int>(blockIdx.x) * kWmmaOutputTile;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  if (expert_index < 0 || valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert ||
      warp_id >= kWmmaWarpsPerBlock) {
    return;
  }

  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kWmmaOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kWmmaOutputTile));
  const auto weight = weights[static_cast<std::size_t>(expert_index)];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / kNvfp4BlockWidth;
  const float weight_tensor_scale = *weight.tensor_scale_data;
  const float input_tensor_scale = *input_tensor_scale_data;

  wmma::fragment<wmma::accumulator, kWmmaTileM, kWmmaTileN, kWmmaTileK, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols; k_base += kWmmaTileK) {
    for (int linear_index = static_cast<int>(threadIdx.x);
         linear_index < (kWmmaOutputTile * kWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kWmmaTileK;
      const int tile_k = linear_index % kWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * pairs_per_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * blocks_per_row;
        const float block_scale =
            DecodeFp8Byte(weight.block_scales_data[scale_row_offset + block]) *
            weight_tensor_scale;
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(DecodeFp4Nibble(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }

    for (int linear_index = static_cast<int>(threadIdx.x);
         linear_index < (kWmmaTileK * kWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kWmmaTileM;
      const int tile_token = linear_index % kWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset = input_row * pairs_per_row;
        const std::size_t scale_row_offset = input_row * blocks_per_row;
        const float block_scale =
            DecodeFp8Byte(input_block_scales[scale_row_offset + block]) *
            input_tensor_scale;
        const std::uint8_t packed = packed_input[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(DecodeFp4Nibble(nibble) * block_scale);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, kWmmaTileM, kWmmaTileN, kWmmaTileK, __nv_bfloat16, wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, kWmmaTileM, kWmmaTileN, kWmmaTileK, __nv_bfloat16, wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kWmmaTileN;
  wmma::store_matrix_sync(&c_tile[warp_row][0], c_frag, kWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  for (int linear_index = static_cast<int>(threadIdx.x);
       linear_index < (output_rows_this_tile * valid_rows);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_output_row = linear_index / valid_rows;
    const int tile_token = linear_index % valid_rows;
    const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
    const std::size_t output_row = static_cast<std::size_t>(output_row_base + tile_output_row);
    output[input_row * output_rows_per_expert + output_row] =
        c_tile[tile_output_row][tile_token];
  }
}

__global__ void LaunchPlanWmmaPackedPerExpertInputTransposedKernel(
    const PackedInputView* packed_inputs,
    const int* expert_offsets,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  __shared__ __nv_bfloat16 a_tile[kWmmaOutputTile][kWmmaTileK];
  __shared__ __nv_bfloat16 b_tile[kWmmaTileK][kWmmaTileM];
  __shared__ float c_tile[kWmmaOutputTile][kWmmaTileM];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count || packed_inputs == nullptr ||
      expert_offsets == nullptr) {
    return;
  }

  const int warp_id = static_cast<int>(threadIdx.x) / 32;
  const int output_row_base = static_cast<int>(blockIdx.x) * kWmmaOutputTile;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  if (expert_index < 0 || valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert ||
      warp_id >= kWmmaWarpsPerBlock) {
    return;
  }

  const PackedInputView input_view = packed_inputs[expert_index];
  if (input_view.packed_data == nullptr ||
      input_view.block_scales_data == nullptr ||
      input_view.tensor_scale_data == nullptr ||
      input_view.cols == 0) {
    return;
  }
  const int expert_row_start = expert_offsets[expert_index];
  const int local_row_start = row_start - expert_row_start;
  if (local_row_start < 0) {
    return;
  }

  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kWmmaOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kWmmaOutputTile));
  const auto weight = weights[static_cast<std::size_t>(expert_index)];
  const std::size_t weight_pairs_per_row = weight.input_cols / 2;
  const std::size_t weight_blocks_per_row = weight.input_cols / kNvfp4BlockWidth;
  const std::size_t input_pairs_per_row = input_view.cols / 2;
  const std::size_t input_blocks_per_row = input_view.cols / kNvfp4BlockWidth;
  const float weight_tensor_scale = *weight.tensor_scale_data;
  const float input_tensor_scale = *input_view.tensor_scale_data;

  wmma::fragment<wmma::accumulator, kWmmaTileM, kWmmaTileN, kWmmaTileK, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (std::size_t k_base = 0; k_base < weight.input_cols; k_base += kWmmaTileK) {
    for (int linear_index = static_cast<int>(threadIdx.x);
         linear_index < (kWmmaOutputTile * kWmmaTileK);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / kWmmaTileK;
      const int tile_k = linear_index % kWmmaTileK;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_output_row < output_rows_this_tile &&
          (k_base + static_cast<std::size_t>(tile_k)) < weight.input_cols) {
        const int output_row = output_row_base + tile_output_row;
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset =
            static_cast<std::size_t>(output_row) * weight_pairs_per_row;
        const std::size_t scale_row_offset =
            static_cast<std::size_t>(output_row) * weight_blocks_per_row;
        const float block_scale =
            DecodeFp8Byte(weight.block_scales_data[scale_row_offset + block]) *
            weight_tensor_scale;
        const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(DecodeFp4Nibble(nibble) * block_scale);
      }
      a_tile[tile_output_row][tile_k] = value;
    }

    for (int linear_index = static_cast<int>(threadIdx.x);
         linear_index < (kWmmaTileK * kWmmaTileM);
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_k = linear_index / kWmmaTileM;
      const int tile_token = linear_index % kWmmaTileM;
      __nv_bfloat16 value = __float2bfloat16(0.0f);
      if (tile_token < valid_rows &&
          (k_base + static_cast<std::size_t>(tile_k)) < input_view.cols) {
        const std::size_t input_row =
            static_cast<std::size_t>(local_row_start + tile_token);
        const std::size_t pair_index =
            (k_base / 2u) + static_cast<std::size_t>(tile_k / 2);
        const std::size_t block = pair_index / 8u;
        const std::size_t packed_row_offset = input_row * input_pairs_per_row;
        const std::size_t scale_row_offset = input_row * input_blocks_per_row;
        const float block_scale =
            DecodeFp8Byte(input_view.block_scales_data[scale_row_offset + block]) *
            input_tensor_scale;
        const std::uint8_t packed = input_view.packed_data[packed_row_offset + pair_index];
        const std::uint8_t nibble =
            (tile_k & 1) == 0 ? (packed & 0x0Fu) : ((packed >> 4) & 0x0Fu);
        value = __float2bfloat16(DecodeFp4Nibble(nibble) * block_scale);
      }
      b_tile[tile_k][tile_token] = value;
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, kWmmaTileM, kWmmaTileN, kWmmaTileK, __nv_bfloat16, wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, kWmmaTileM, kWmmaTileN, kWmmaTileK, __nv_bfloat16, wmma::row_major>
        b_frag;
    const int warp_row = warp_id * kWmmaTileN;
    wmma::load_matrix_sync(a_frag, &a_tile[warp_row][0], kWmmaTileK);
    wmma::load_matrix_sync(b_frag, &b_tile[0][0], kWmmaTileM);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  const int warp_row = warp_id * kWmmaTileN;
  wmma::store_matrix_sync(&c_tile[warp_row][0], c_frag, kWmmaTileM, wmma::mem_row_major);
  __syncthreads();

  for (int linear_index = static_cast<int>(threadIdx.x);
       linear_index < (output_rows_this_tile * valid_rows);
       linear_index += static_cast<int>(blockDim.x)) {
    const int tile_output_row = linear_index / valid_rows;
    const int tile_token = linear_index % valid_rows;
    const std::size_t input_row = static_cast<std::size_t>(row_start + tile_token);
    const std::size_t output_row = static_cast<std::size_t>(output_row_base + tile_output_row);
    output[input_row * output_rows_per_expert + output_row] =
        c_tile[tile_output_row][tile_token];
  }
}

__global__ void LaunchPlanBf16ScalarKernel(
    const float* input,
    const int* cta_count,
    const int* cta_expert_ids,
    const int* cta_row_starts,
    const int* cta_valid_rows,
    const FusedNvfp4WeightView* weights,
    std::size_t output_rows_per_expert,
    float* output) {
  __shared__ __nv_bfloat16 input_tile[kPermutedTokenTile][kPermutedPairsPerStep * 2];
  __shared__ __nv_bfloat16 weight_tile[kPermutedCoopOutputTile][kPermutedPairsPerStep * 2];

  const int cta_index = static_cast<int>(blockIdx.y);
  const int exact_cta_count = cta_count[0];
  if (cta_index >= exact_cta_count) {
    return;
  }

  const int tid = static_cast<int>(threadIdx.x);
  const int token_index = tid / kPermutedCoopOutputTile;
  const int output_lane = tid % kPermutedCoopOutputTile;
  const int output_row_base = static_cast<int>(blockIdx.x) * kPermutedCoopOutputTile;
  const int expert_index = cta_expert_ids[cta_index];
  const int row_start = cta_row_starts[cta_index];
  const int valid_rows = cta_valid_rows[cta_index];
  if (expert_index < 0 || valid_rows <= 0 ||
      static_cast<std::size_t>(output_row_base) >= output_rows_per_expert) {
    return;
  }

  const int output_rows_this_tile = static_cast<int>(
      (output_rows_per_expert - static_cast<std::size_t>(output_row_base)) <
              static_cast<std::size_t>(kPermutedCoopOutputTile)
          ? (output_rows_per_expert - static_cast<std::size_t>(output_row_base))
          : static_cast<std::size_t>(kPermutedCoopOutputTile));
  const auto weight = weights[static_cast<std::size_t>(expert_index)];
  const std::size_t pairs_per_row = weight.input_cols / 2;
  const std::size_t blocks_per_row = weight.input_cols / kNvfp4BlockWidth;
  const float tensor_scale = *weight.tensor_scale_data;
  float accum = 0.0f;

  for (std::size_t pair_base = 0; pair_base < pairs_per_row;
       pair_base += kPermutedPairsPerStep) {
    const std::size_t remaining_pairs = pairs_per_row - pair_base;
    const int pairs_this_step = static_cast<int>(
        remaining_pairs < static_cast<std::size_t>(kPermutedPairsPerStep)
            ? remaining_pairs
            : static_cast<std::size_t>(kPermutedPairsPerStep));
    const int values_this_step = valid_rows * pairs_this_step * 2;
    for (int linear_index = tid;
         linear_index < values_this_step;
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_token = linear_index / (pairs_this_step * 2);
      const int within_token = linear_index % (pairs_this_step * 2);
      const int pair_offset = within_token / 2;
      const int value_offset = within_token % 2;
      const std::size_t input_row =
          static_cast<std::size_t>(row_start + tile_token);
      const std::size_t col =
          (pair_base + static_cast<std::size_t>(pair_offset)) * 2u +
          static_cast<std::size_t>(value_offset);
      input_tile[tile_token][within_token] =
          __float2bfloat16(input[input_row * weight.input_cols + col]);
    }

    const int packed_values_this_step = output_rows_this_tile * pairs_this_step;
    for (int linear_index = tid;
         linear_index < packed_values_this_step;
         linear_index += static_cast<int>(blockDim.x)) {
      const int tile_output_row = linear_index / pairs_this_step;
      const int pair_offset = linear_index % pairs_this_step;
      const int output_row = output_row_base + tile_output_row;
      const std::size_t pair_index =
          pair_base + static_cast<std::size_t>(pair_offset);
      const std::size_t block = pair_index / 8u;
      const std::size_t packed_row_offset =
          static_cast<std::size_t>(output_row) * pairs_per_row;
      const std::size_t scale_row_offset =
          static_cast<std::size_t>(output_row) * blocks_per_row;
      const float block_scale =
          DecodeFp8Byte(weight.block_scales_data[scale_row_offset + block]) * tensor_scale;
      const std::uint8_t packed = weight.packed_data[packed_row_offset + pair_index];
      weight_tile[tile_output_row][pair_offset * 2] =
          __float2bfloat16(DecodeFp4Nibble(packed & 0x0Fu) * block_scale);
      weight_tile[tile_output_row][pair_offset * 2 + 1] =
          __float2bfloat16(DecodeFp4Nibble((packed >> 4) & 0x0Fu) * block_scale);
    }
    __syncthreads();

    if (token_index < valid_rows && output_lane < output_rows_this_tile) {
      for (int value_index = 0; value_index < (pairs_this_step * 2); ++value_index) {
        accum += __bfloat162float(input_tile[token_index][value_index]) *
                 __bfloat162float(weight_tile[output_lane][value_index]);
      }
    }
    __syncthreads();
  }

  if (token_index < valid_rows && output_lane < output_rows_this_tile) {
    const std::size_t input_row = static_cast<std::size_t>(row_start + token_index);
    const std::size_t output_row = static_cast<std::size_t>(output_row_base + output_lane);
    output[input_row * output_rows_per_expert + output_row] = accum;
  }
}


__global__ void GatherPermutedRowsKernel(
    const float* input,
    const int* permuted_token_indices,
    float* output,
    std::size_t output_rows,
    std::size_t input_rows,
    std::size_t cols) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = output_rows * cols;
  if (index >= count) {
    return;
  }

  const std::size_t row = index / cols;
  const std::size_t col = index % cols;
  const int input_row = permuted_token_indices[row];
  if (input_row < 0 || static_cast<std::size_t>(input_row) >= input_rows) {
    output[index] = 0.0f;
    return;
  }
  output[index] = input[static_cast<std::size_t>(input_row) * cols + col];
}

template <typename T>
class DeviceArray {
 public:
  static std::unique_ptr<DeviceArray> CopyFromHost(const std::vector<T>& values) {
    if (values.empty()) {
      return nullptr;
    }
    T* data = nullptr;
    const std::size_t bytes = values.size() * sizeof(T);
    if (cudaMalloc(reinterpret_cast<void**>(&data), bytes) != cudaSuccess) {
      return nullptr;
    }
    if (cudaMemcpy(data, values.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
      cudaFree(data);
      return nullptr;
    }
    return std::unique_ptr<DeviceArray>(new DeviceArray(data));
  }

  ~DeviceArray() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;

  const T* data() const {
    return data_;
  }

 private:
  explicit DeviceArray(T* data) : data_(data) {}

  T* data_ = nullptr;
};

class ScopedCudaEventTimer {
 public:
  ScopedCudaEventTimer() {
    valid_ =
        cudaEventCreate(&start_) == cudaSuccess &&
        cudaEventCreate(&stop_) == cudaSuccess;
  }

  ~ScopedCudaEventTimer() {
    if (start_ != nullptr) {
      cudaEventDestroy(start_);
    }
    if (stop_ != nullptr) {
      cudaEventDestroy(stop_);
    }
  }

  bool valid() const {
    return valid_;
  }

  template <typename Callable>
  std::optional<double> Measure(Callable&& callable) {
    if (!valid_) {
      return std::nullopt;
    }
    if (cudaEventRecord(start_) != cudaSuccess) {
      return std::nullopt;
    }
    if (!callable()) {
      return std::nullopt;
    }
    if (cudaEventRecord(stop_) != cudaSuccess ||
        cudaEventSynchronize(stop_) != cudaSuccess) {
      return std::nullopt;
    }
    float elapsed_ms = 0.0f;
    if (cudaEventElapsedTime(&elapsed_ms, start_, stop_) != cudaSuccess) {
      return std::nullopt;
    }
    return static_cast<double>(elapsed_ms);
  }

 private:
  bool valid_ = false;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

bool HasCudaDevice() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

template <typename T>
bool CopyDeviceBufferToHost(const T* device_data, std::size_t count, std::vector<T>* host_values) {
  if (device_data == nullptr || host_values == nullptr) {
    return false;
  }
  host_values->assign(count, T{});
  return cudaMemcpy(
             host_values->data(),
             device_data,
             count * sizeof(T),
             cudaMemcpyDeviceToHost) == cudaSuccess;
}

bool LaunchGatherPermutedRows(
    const float* input,
    const int* permuted_token_indices,
    std::size_t output_rows,
    std::size_t input_rows,
    std::size_t cols,
    float* output) {
  if (input == nullptr ||
      permuted_token_indices == nullptr ||
      output == nullptr ||
      output_rows == 0 ||
      cols == 0) {
    return false;
  }
  const dim3 block(256);
  const dim3 grid(static_cast<unsigned int>(((output_rows * cols) + block.x - 1u) / block.x));
  GatherPermutedRowsKernel<<<grid, block>>>(
      input,
      permuted_token_indices,
      output,
      output_rows,
      input_rows,
      cols);
  return cudaGetLastError() == cudaSuccess;
}

std::vector<float> MakePatternedValues(
    std::size_t rows,
    std::size_t cols,
    int seed,
    float scale) {
  std::vector<float> values(rows * cols, 0.0f);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      const int raw =
          static_cast<int>(((row + 1) * (seed + 7)) + ((col + 5) * (seed + 13)));
      values[row * cols + col] =
          static_cast<float>((raw % 31) - 15) * scale +
          0.001f * static_cast<float>((row + col + static_cast<std::size_t>(seed)) % 5);
    }
  }
  return values;
}

const std::vector<BenchmarkCase>& DefaultCases() {
  static const std::vector<BenchmarkCase> kCases = {
      {"prefix4", 4},
      {"prefix128", 128},
      {"prefix4096", 4096},
  };
  return kCases;
}

bool ShouldRunCase(
    const BenchmarkOptions& options,
    const std::string& case_name) {
  return options.selected_case_names.empty() ||
         std::find(
             options.selected_case_names.begin(),
             options.selected_case_names.end(),
             case_name) != options.selected_case_names.end();
}

std::vector<int> BuildExpertOffsets(std::size_t selection_count) {
  std::vector<int> offsets(kRoutedExperts + 1, 0);
  const std::size_t base = selection_count / kRoutedExperts;
  const std::size_t remainder = selection_count % kRoutedExperts;
  int running = 0;
  for (std::size_t expert = 0; expert < kRoutedExperts; ++expert) {
    offsets[expert] = running;
    running += static_cast<int>(base + (expert < remainder ? 1 : 0));
  }
  offsets[kRoutedExperts] = running;
  return offsets;
}

std::size_t CountActiveExperts(const std::vector<int>& offsets) {
  std::size_t active = 0;
  for (std::size_t expert = 0; expert + 1 < offsets.size(); ++expert) {
    if (offsets[expert + 1] > offsets[expert]) {
      ++active;
    }
  }
  return active;
}

std::vector<int> BuildSelectedIndicesForOffsets(const std::vector<int>& offsets) {
  std::vector<int> selected_indices;
  if (offsets.size() < 2) {
    return selected_indices;
  }
  selected_indices.reserve(static_cast<std::size_t>(offsets.back()));
  for (std::size_t expert = 0; expert + 1 < offsets.size(); ++expert) {
    const int begin = offsets[expert];
    const int end = offsets[expert + 1];
    for (int index = begin; index < end; ++index) {
      (void)index;
      selected_indices.push_back(static_cast<int>(expert));
    }
  }
  return selected_indices;
}

void PopulateDiffSummary(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs,
    BenchmarkResult* result) {
  if (result == nullptr) {
    return;
  }
  if (lhs.size() != rhs.size()) {
    result->max_abs_diff_vs_baseline = std::numeric_limits<double>::infinity();
    return;
  }
  double max_abs_diff = 0.0;
  std::size_t max_abs_diff_index = 0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    const double abs_diff = static_cast<double>(std::abs(lhs[i] - rhs[i]));
    if (abs_diff > max_abs_diff) {
      max_abs_diff = abs_diff;
      max_abs_diff_index = i;
    }
  }
  result->max_abs_diff_vs_baseline = max_abs_diff;
  result->max_abs_diff_index = max_abs_diff_index;
  result->max_abs_diff_baseline_value = lhs[max_abs_diff_index];
  result->max_abs_diff_variant_value = rhs[max_abs_diff_index];
}

void BuildCtaTileMetadata(
    const std::vector<int>& padded_offsets,
    std::vector<int>* cta_expert_ids,
    std::vector<int>* cta_row_starts,
    std::vector<int>* cta_output_row_bases) {
  if (cta_expert_ids == nullptr || cta_row_starts == nullptr || cta_output_row_bases == nullptr) {
    return;
  }
  cta_expert_ids->clear();
  cta_row_starts->clear();
  cta_output_row_bases->clear();
  for (std::size_t expert = 0; expert + 1 < padded_offsets.size(); ++expert) {
    const int begin = padded_offsets[expert];
    const int end = padded_offsets[expert + 1];
    for (int row = begin; row < end; row += kPermutedTokenTile) {
      for (int output_row_base = 0;
           output_row_base < static_cast<int>(kRoutedIntermediateSize);
           output_row_base += kPermutedOutputTile) {
        cta_expert_ids->push_back(static_cast<int>(expert));
        cta_row_starts->push_back(row);
        cta_output_row_bases->push_back(output_row_base);
      }
    }
  }
}

void BuildWideCtaTileMetadata(
    const std::vector<int>& padded_offsets,
    std::vector<int>* cta_expert_ids,
    std::vector<int>* cta_row_starts,
    std::vector<int>* cta_output_row_bases) {
  if (cta_expert_ids == nullptr || cta_row_starts == nullptr || cta_output_row_bases == nullptr) {
    return;
  }
  cta_expert_ids->clear();
  cta_row_starts->clear();
  cta_output_row_bases->clear();
  for (std::size_t expert = 0; expert + 1 < padded_offsets.size(); ++expert) {
    const int begin = padded_offsets[expert];
    const int end = padded_offsets[expert + 1];
    for (int row = begin; row < end; row += kPermutedTokenTile) {
      for (int output_row_base = 0;
           output_row_base < static_cast<int>(kRoutedIntermediateSize);
           output_row_base += kPermutedWideOutputTile) {
        cta_expert_ids->push_back(static_cast<int>(expert));
        cta_row_starts->push_back(row);
        cta_output_row_bases->push_back(output_row_base);
      }
    }
  }
}

std::optional<UploadedWeights> BuildUploadedRoutedUpViews() {
  auto storage =
      MonolithicNvfp4ExpertWeights::Create(kRoutedExperts, kRoutedIntermediateSize, kHiddenSize);
  if (!storage || !storage->valid()) {
    return std::nullopt;
  }

  const auto source =
      MakePatternedValues(kRoutedIntermediateSize, kHiddenSize, 17, 0.015625f);
  const auto packed =
      PackRowMajorFp32ToNvfp4(source.data(), kRoutedIntermediateSize, kHiddenSize);
  if (!packed.has_value()) {
    return std::nullopt;
  }

  for (std::size_t expert = 0; expert < kRoutedExperts; ++expert) {
    if (!storage->UploadExpert(
            expert,
            packed->packed_data(),
            packed->packed_nbytes(),
            packed->block_scales_data(),
            packed->block_scales_nbytes(),
            reinterpret_cast<const float*>(packed->tensor_scale_data()))) {
      return std::nullopt;
    }
  }

  UploadedWeights uploaded;
  uploaded.views = storage->BuildAllViews();
  uploaded.host_packed = packed;
  uploaded.storage = std::move(storage);
  return uploaded;
}

std::optional<BenchmarkResult> RunCase(
    const BenchmarkCase& benchmark_case,
    const BenchmarkOptions& options,
    std::vector<float>* output_host = nullptr) {
  const std::size_t selection_count = benchmark_case.prefix_tokens * kTopK;
  const auto expert_offsets = BuildExpertOffsets(selection_count);
  const auto selected_indices_host = BuildSelectedIndicesForOffsets(expert_offsets);
  const std::vector<float> selected_weights_host(selection_count, 1.0f);
  const std::size_t active_experts = CountActiveExperts(expert_offsets);
  auto selected_indices_device = DeviceArray<int>::CopyFromHost(selected_indices_host);
  auto selected_weights_device = DeviceArray<float>::CopyFromHost(selected_weights_host);
  auto routing = DeviceExpertRouting::Create(kRoutedExperts, selection_count);
  if (selected_indices_device == nullptr ||
      selected_weights_device == nullptr ||
      routing == nullptr ||
      !routing->valid() ||
      !RunDeviceExpertRouting(
          selected_indices_device->data(),
          selected_weights_device->data(),
          benchmark_case.prefix_tokens,
          kTopK,
          routing.get()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  const auto routed_up = BuildUploadedRoutedUpViews();
  if (!routed_up.has_value()) {
    return std::nullopt;
  }
  auto routed_up_views_device =
      DeviceArray<FusedNvfp4WeightView>::CopyFromHost(routed_up->views);
  if (routed_up_views_device == nullptr) {
    return std::nullopt;
  }

  auto input = DeviceTensorFp32::Create({benchmark_case.prefix_tokens, kHiddenSize});
  auto compact_input = DeviceTensorFp32::Create({selection_count, kHiddenSize});
  auto output = DeviceTensorFp32::Create({selection_count, kRoutedIntermediateSize});
  if (input == nullptr || compact_input == nullptr || output == nullptr) {
    return std::nullopt;
  }

  const auto input_host =
      MakePatternedValues(benchmark_case.prefix_tokens, kHiddenSize, 23, 0.03125f);
  if (!input->CopyFromHost(input_host.data(), input_host.size())) {
    return std::nullopt;
  }
  if (!LaunchGatherPermutedRows(
          input->data(),
          routing->sorted_token_indices(),
          selection_count,
          benchmark_case.prefix_tokens,
          kHiddenSize,
          compact_input->data()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  ScopedCudaEventTimer timer;
  if (!timer.valid()) {
    return std::nullopt;
  }

  auto run_once = [&]() {
    return RunGroupedNvfp4ExpertMatVec(
        compact_input->data(),
        routing->expert_offsets(),
        kRoutedExperts,
        routed_up_views_device->data(),
        kRoutedIntermediateSize,
        output->data());
  };

  for (std::size_t iteration = 0; iteration < options.warmup_iterations; ++iteration) {
    const auto elapsed = timer.Measure(run_once);
    if (!elapsed.has_value()) {
      return std::nullopt;
    }
  }

  const auto cold_ms = timer.Measure(run_once);
  if (!cold_ms.has_value()) {
    return std::nullopt;
  }

  double hot_sum_ms = 0.0;
  double hot_min_ms = 0.0;
  double hot_max_ms = 0.0;
  for (std::size_t iteration = 0; iteration < options.hot_iterations; ++iteration) {
    const auto hot_elapsed = timer.Measure(run_once);
    if (!hot_elapsed.has_value()) {
      return std::nullopt;
    }
    const double elapsed_ms = *hot_elapsed;
    hot_sum_ms += elapsed_ms;
    if (iteration == 0 || elapsed_ms < hot_min_ms) {
      hot_min_ms = elapsed_ms;
    }
    if (iteration == 0 || elapsed_ms > hot_max_ms) {
      hot_max_ms = elapsed_ms;
    }
  }

  const double hot_mean_ms =
      hot_sum_ms / static_cast<double>(std::max<std::size_t>(1, options.hot_iterations));
  const double flops =
      2.0 * static_cast<double>(selection_count) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize);
  const double weight_bytes =
      static_cast<double>(active_experts) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize) *
      kNvfp4BytesPerWeight;

  BenchmarkResult result;
  result.variant_name = "baseline";
  result.case_name = benchmark_case.name;
  result.prefix_tokens = benchmark_case.prefix_tokens;
  result.selection_count = selection_count;
  result.padded_selection_count = selection_count;
  result.active_experts = active_experts;
  result.cold_ms = *cold_ms;
  result.hot_mean_ms = hot_mean_ms;
  result.hot_min_ms = hot_min_ms;
  result.hot_max_ms = hot_max_ms;
  result.hot_tflops =
      hot_mean_ms > 0.0 ? flops / (hot_mean_ms / 1000.0) / 1.0e12 : 0.0;
  result.hot_weight_gib_per_s =
      hot_mean_ms > 0.0 ? weight_bytes / (hot_mean_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0) : 0.0;
  if (output_host != nullptr) {
    output_host->resize(selection_count * kRoutedIntermediateSize);
    if (!output->CopyToHost(output_host->data(), output_host->size())) {
      return std::nullopt;
    }
  }
  return result;
}

std::optional<BenchmarkResult> RunRaggedRowCoopCase(
    const BenchmarkCase& benchmark_case,
    const BenchmarkOptions& options,
    std::vector<float>* output_host = nullptr) {
  const std::size_t selection_count = benchmark_case.prefix_tokens * kTopK;
  const auto expert_offsets = BuildExpertOffsets(selection_count);
  const auto selected_indices_host = BuildSelectedIndicesForOffsets(expert_offsets);
  const std::vector<float> selected_weights_host(selection_count, 1.0f);
  const std::size_t active_experts = CountActiveExperts(expert_offsets);
  auto selected_indices_device = DeviceArray<int>::CopyFromHost(selected_indices_host);
  auto selected_weights_device = DeviceArray<float>::CopyFromHost(selected_weights_host);
  auto routing = DeviceExpertRouting::Create(kRoutedExperts, selection_count);
  if (selected_indices_device == nullptr ||
      selected_weights_device == nullptr ||
      routing == nullptr ||
      !routing->valid() ||
      !RunDeviceExpertRouting(
          selected_indices_device->data(),
          selected_weights_device->data(),
          benchmark_case.prefix_tokens,
          kTopK,
          routing.get()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  std::vector<int> cta_expert_ids;
  std::vector<int> cta_row_starts;
  std::vector<int> cta_output_row_bases;
  BuildCtaTileMetadata(
      expert_offsets,
      &cta_expert_ids,
      &cta_row_starts,
      &cta_output_row_bases);

  auto expert_offsets_device = DeviceArray<int>::CopyFromHost(expert_offsets);
  auto cta_expert_ids_device = DeviceArray<int>::CopyFromHost(cta_expert_ids);
  auto cta_row_starts_device = DeviceArray<int>::CopyFromHost(cta_row_starts);
  auto cta_output_row_bases_device = DeviceArray<int>::CopyFromHost(cta_output_row_bases);
  if (expert_offsets_device == nullptr ||
      cta_expert_ids_device == nullptr ||
      cta_row_starts_device == nullptr ||
      cta_output_row_bases_device == nullptr) {
    return std::nullopt;
  }

  const auto routed_up = BuildUploadedRoutedUpViews();
  if (!routed_up.has_value()) {
    return std::nullopt;
  }
  auto routed_up_views_device =
      DeviceArray<FusedNvfp4WeightView>::CopyFromHost(routed_up->views);
  if (routed_up_views_device == nullptr) {
    return std::nullopt;
  }

  const auto input_host =
      MakePatternedValues(benchmark_case.prefix_tokens, kHiddenSize, 23, 0.03125f);
  auto input = DeviceTensorFp32::Create({benchmark_case.prefix_tokens, kHiddenSize});
  auto compact_input = DeviceTensorFp32::Create({selection_count, kHiddenSize});
  auto output = DeviceTensorFp32::Create({selection_count, kRoutedIntermediateSize});
  if (input == nullptr || compact_input == nullptr || output == nullptr) {
    return std::nullopt;
  }
  if (!input->CopyFromHost(input_host.data(), input_host.size())) {
    return std::nullopt;
  }
  if (!LaunchGatherPermutedRows(
          input->data(),
          routing->sorted_token_indices(),
          selection_count,
          benchmark_case.prefix_tokens,
          kHiddenSize,
          compact_input->data()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  ScopedCudaEventTimer timer;
  if (!timer.valid()) {
    return std::nullopt;
  }

  const dim3 block(kPermutedThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(cta_expert_ids.size()));
  auto run_once = [&]() {
    PermutedExpertRowCoopKernel<<<grid, block>>>(
        compact_input->data(),
        expert_offsets_device->data(),
        cta_expert_ids_device->data(),
        cta_row_starts_device->data(),
        cta_output_row_bases_device->data(),
        routed_up_views_device->data(),
        kRoutedIntermediateSize,
        output->data());
    return cudaGetLastError() == cudaSuccess;
  };

  for (std::size_t iteration = 0; iteration < options.warmup_iterations; ++iteration) {
    const auto elapsed = timer.Measure(run_once);
    if (!elapsed.has_value()) {
      return std::nullopt;
    }
  }

  const auto cold_ms = timer.Measure(run_once);
  if (!cold_ms.has_value()) {
    return std::nullopt;
  }

  double hot_sum_ms = 0.0;
  double hot_min_ms = 0.0;
  double hot_max_ms = 0.0;
  for (std::size_t iteration = 0; iteration < options.hot_iterations; ++iteration) {
    const auto hot_elapsed = timer.Measure(run_once);
    if (!hot_elapsed.has_value()) {
      return std::nullopt;
    }
    const double elapsed_ms = *hot_elapsed;
    hot_sum_ms += elapsed_ms;
    if (iteration == 0 || elapsed_ms < hot_min_ms) {
      hot_min_ms = elapsed_ms;
    }
    if (iteration == 0 || elapsed_ms > hot_max_ms) {
      hot_max_ms = elapsed_ms;
    }
  }

  const double hot_mean_ms =
      hot_sum_ms / static_cast<double>(std::max<std::size_t>(1, options.hot_iterations));
  const double logical_flops =
      2.0 * static_cast<double>(selection_count) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize);
  const double weight_bytes =
      static_cast<double>(active_experts) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize) *
      kNvfp4BytesPerWeight;

  BenchmarkResult result;
  result.variant_name = "ragged_row_coop";
  result.case_name = benchmark_case.name;
  result.prefix_tokens = benchmark_case.prefix_tokens;
  result.selection_count = selection_count;
  result.padded_selection_count = selection_count;
  result.active_experts = active_experts;
  result.cold_ms = *cold_ms;
  result.hot_mean_ms = hot_mean_ms;
  result.hot_min_ms = hot_min_ms;
  result.hot_max_ms = hot_max_ms;
  result.hot_tflops =
      hot_mean_ms > 0.0 ? logical_flops / (hot_mean_ms / 1000.0) / 1.0e12 : 0.0;
  result.hot_weight_gib_per_s =
      hot_mean_ms > 0.0 ? weight_bytes / (hot_mean_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0) : 0.0;
  if (output_host != nullptr) {
    output_host->resize(selection_count * kRoutedIntermediateSize);
    if (!output->CopyToHost(output_host->data(), output_host->size())) {
      return std::nullopt;
    }
  }
  return result;
}

std::optional<BenchmarkResult> RunRaggedWideRowCoopCase(
    const BenchmarkCase& benchmark_case,
    const BenchmarkOptions& options,
    std::vector<float>* output_host = nullptr) {
  const std::size_t selection_count = benchmark_case.prefix_tokens * kTopK;
  const auto expert_offsets = BuildExpertOffsets(selection_count);
  const auto selected_indices_host = BuildSelectedIndicesForOffsets(expert_offsets);
  const std::vector<float> selected_weights_host(selection_count, 1.0f);
  const std::size_t active_experts = CountActiveExperts(expert_offsets);
  auto selected_indices_device = DeviceArray<int>::CopyFromHost(selected_indices_host);
  auto selected_weights_device = DeviceArray<float>::CopyFromHost(selected_weights_host);
  auto routing = DeviceExpertRouting::Create(kRoutedExperts, selection_count);
  if (selected_indices_device == nullptr ||
      selected_weights_device == nullptr ||
      routing == nullptr ||
      !routing->valid() ||
      !RunDeviceExpertRouting(
          selected_indices_device->data(),
          selected_weights_device->data(),
          benchmark_case.prefix_tokens,
          kTopK,
          routing.get()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  std::vector<int> cta_expert_ids;
  std::vector<int> cta_row_starts;
  std::vector<int> cta_output_row_bases;
  BuildWideCtaTileMetadata(
      expert_offsets,
      &cta_expert_ids,
      &cta_row_starts,
      &cta_output_row_bases);

  auto expert_offsets_device = DeviceArray<int>::CopyFromHost(expert_offsets);
  auto cta_expert_ids_device = DeviceArray<int>::CopyFromHost(cta_expert_ids);
  auto cta_row_starts_device = DeviceArray<int>::CopyFromHost(cta_row_starts);
  auto cta_output_row_bases_device = DeviceArray<int>::CopyFromHost(cta_output_row_bases);
  if (expert_offsets_device == nullptr ||
      cta_expert_ids_device == nullptr ||
      cta_row_starts_device == nullptr ||
      cta_output_row_bases_device == nullptr) {
    return std::nullopt;
  }

  const auto routed_up = BuildUploadedRoutedUpViews();
  if (!routed_up.has_value()) {
    return std::nullopt;
  }
  auto routed_up_views_device =
      DeviceArray<FusedNvfp4WeightView>::CopyFromHost(routed_up->views);
  if (routed_up_views_device == nullptr) {
    return std::nullopt;
  }

  const auto input_host =
      MakePatternedValues(benchmark_case.prefix_tokens, kHiddenSize, 23, 0.03125f);
  auto input = DeviceTensorFp32::Create({benchmark_case.prefix_tokens, kHiddenSize});
  auto compact_input = DeviceTensorFp32::Create({selection_count, kHiddenSize});
  auto output = DeviceTensorFp32::Create({selection_count, kRoutedIntermediateSize});
  if (input == nullptr || compact_input == nullptr || output == nullptr) {
    return std::nullopt;
  }
  if (!input->CopyFromHost(input_host.data(), input_host.size())) {
    return std::nullopt;
  }
  if (!LaunchGatherPermutedRows(
          input->data(),
          routing->sorted_token_indices(),
          selection_count,
          benchmark_case.prefix_tokens,
          kHiddenSize,
          compact_input->data()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  ScopedCudaEventTimer timer;
  if (!timer.valid()) {
    return std::nullopt;
  }

  const dim3 block(kPermutedWideThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(cta_expert_ids.size()));
  auto run_once = [&]() {
    PermutedExpertWideRowCoopKernel<<<grid, block>>>(
        compact_input->data(),
        expert_offsets_device->data(),
        cta_expert_ids_device->data(),
        cta_row_starts_device->data(),
        cta_output_row_bases_device->data(),
        routed_up_views_device->data(),
        kRoutedIntermediateSize,
        output->data());
    return cudaGetLastError() == cudaSuccess;
  };

  for (std::size_t iteration = 0; iteration < options.warmup_iterations; ++iteration) {
    const auto elapsed = timer.Measure(run_once);
    if (!elapsed.has_value()) {
      return std::nullopt;
    }
  }

  const auto cold_ms = timer.Measure(run_once);
  if (!cold_ms.has_value()) {
    return std::nullopt;
  }

  double hot_sum_ms = 0.0;
  double hot_min_ms = 0.0;
  double hot_max_ms = 0.0;
  for (std::size_t iteration = 0; iteration < options.hot_iterations; ++iteration) {
    const auto hot_elapsed = timer.Measure(run_once);
    if (!hot_elapsed.has_value()) {
      return std::nullopt;
    }
    const double elapsed_ms = *hot_elapsed;
    hot_sum_ms += elapsed_ms;
    if (iteration == 0 || elapsed_ms < hot_min_ms) {
      hot_min_ms = elapsed_ms;
    }
    if (iteration == 0 || elapsed_ms > hot_max_ms) {
      hot_max_ms = elapsed_ms;
    }
  }

  const double hot_mean_ms =
      hot_sum_ms / static_cast<double>(std::max<std::size_t>(1, options.hot_iterations));
  const double logical_flops =
      2.0 * static_cast<double>(selection_count) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize);
  const double weight_bytes =
      static_cast<double>(active_experts) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize) *
      kNvfp4BytesPerWeight;

  BenchmarkResult result;
  result.variant_name = "ragged_row_coop_tile8";
  result.case_name = benchmark_case.name;
  result.prefix_tokens = benchmark_case.prefix_tokens;
  result.selection_count = selection_count;
  result.padded_selection_count = selection_count;
  result.active_experts = active_experts;
  result.cold_ms = *cold_ms;
  result.hot_mean_ms = hot_mean_ms;
  result.hot_min_ms = hot_min_ms;
  result.hot_max_ms = hot_max_ms;
  result.hot_tflops =
      hot_mean_ms > 0.0 ? logical_flops / (hot_mean_ms / 1000.0) / 1.0e12 : 0.0;
  result.hot_weight_gib_per_s =
      hot_mean_ms > 0.0 ? weight_bytes / (hot_mean_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0) : 0.0;
  if (output_host != nullptr) {
    output_host->resize(selection_count * kRoutedIntermediateSize);
    if (!output->CopyToHost(output_host->data(), output_host->size())) {
      return std::nullopt;
    }
  }
  return result;
}

std::optional<BenchmarkResult> RunLaunchPlanUpperBoundCase(
    const BenchmarkCase& benchmark_case,
    const BenchmarkOptions& options,
    std::vector<float>* output_host = nullptr) {
  const std::size_t selection_count = benchmark_case.prefix_tokens * kTopK;
  const auto expert_offsets = BuildExpertOffsets(selection_count);
  const std::size_t active_experts = CountActiveExperts(expert_offsets);
  const auto selected_indices_host = BuildSelectedIndicesForOffsets(expert_offsets);
  const std::vector<float> selected_weights_host(selection_count, 1.0f);

  auto selected_indices_device = DeviceArray<int>::CopyFromHost(selected_indices_host);
  auto selected_weights_device = DeviceArray<float>::CopyFromHost(selected_weights_host);
  auto routing = DeviceExpertRouting::Create(kRoutedExperts, selection_count);
  auto launch_plan = DeviceMoeLaunchPlan::Create(kRoutedExperts, selection_count);
  if (selected_indices_device == nullptr ||
      selected_weights_device == nullptr ||
      routing == nullptr ||
      !routing->valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      !RunDeviceExpertRouting(
          selected_indices_device->data(),
          selected_weights_device->data(),
          benchmark_case.prefix_tokens,
          kTopK,
          routing.get()) ||
      !BuildDeviceMoeLaunchPlan(*routing, selection_count, launch_plan.get()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  const auto routed_up = BuildUploadedRoutedUpViews();
  if (!routed_up.has_value()) {
    return std::nullopt;
  }
  auto routed_up_views_device =
      DeviceArray<FusedNvfp4WeightView>::CopyFromHost(routed_up->views);
  if (routed_up_views_device == nullptr) {
    return std::nullopt;
  }

  const auto input_host =
      MakePatternedValues(benchmark_case.prefix_tokens, kHiddenSize, 23, 0.03125f);
  auto input = DeviceTensorFp32::Create({benchmark_case.prefix_tokens, kHiddenSize});
  auto compact_input = DeviceTensorFp32::Create({selection_count, kHiddenSize});
  auto output = DeviceTensorFp32::Create({selection_count, kRoutedIntermediateSize});
  if (input == nullptr || compact_input == nullptr || output == nullptr) {
    return std::nullopt;
  }
  if (!input->CopyFromHost(input_host.data(), input_host.size())) {
    return std::nullopt;
  }
  if (!LaunchGatherPermutedRows(
          input->data(),
          routing->sorted_token_indices(),
          selection_count,
          benchmark_case.prefix_tokens,
          kHiddenSize,
          compact_input->data()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  ScopedCudaEventTimer timer;
  if (!timer.valid()) {
    return std::nullopt;
  }

  auto run_once = [&]() {
    return RunLaunchPlannedNvfp4ExpertMatVec(
        compact_input->data(),
        launch_plan.get(),
        selection_count,
        routed_up_views_device->data(),
        kRoutedIntermediateSize,
        output->data());
  };

  for (std::size_t iteration = 0; iteration < options.warmup_iterations; ++iteration) {
    const auto elapsed = timer.Measure(run_once);
    if (!elapsed.has_value()) {
      return std::nullopt;
    }
  }

  const auto cold_ms = timer.Measure(run_once);
  if (!cold_ms.has_value()) {
    return std::nullopt;
  }

  double hot_sum_ms = 0.0;
  double hot_min_ms = 0.0;
  double hot_max_ms = 0.0;
  for (std::size_t iteration = 0; iteration < options.hot_iterations; ++iteration) {
    const auto hot_elapsed = timer.Measure(run_once);
    if (!hot_elapsed.has_value()) {
      return std::nullopt;
    }
    const double elapsed_ms = *hot_elapsed;
    hot_sum_ms += elapsed_ms;
    if (iteration == 0 || elapsed_ms < hot_min_ms) {
      hot_min_ms = elapsed_ms;
    }
    if (iteration == 0 || elapsed_ms > hot_max_ms) {
      hot_max_ms = elapsed_ms;
    }
  }

  const double hot_mean_ms =
      hot_sum_ms / static_cast<double>(std::max<std::size_t>(1, options.hot_iterations));
  const double logical_flops =
      2.0 * static_cast<double>(selection_count) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize);
  const double weight_bytes =
      static_cast<double>(active_experts) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize) *
      kNvfp4BytesPerWeight;

  BenchmarkResult result;
  result.variant_name = "launch_plan_upper_bound";
  result.case_name = benchmark_case.name;
  result.prefix_tokens = benchmark_case.prefix_tokens;
  result.selection_count = selection_count;
  result.padded_selection_count = selection_count;
  result.active_experts = active_experts;
  result.cold_ms = *cold_ms;
  result.hot_mean_ms = hot_mean_ms;
  result.hot_min_ms = hot_min_ms;
  result.hot_max_ms = hot_max_ms;
  result.hot_tflops =
      hot_mean_ms > 0.0 ? logical_flops / (hot_mean_ms / 1000.0) / 1.0e12 : 0.0;
  result.hot_weight_gib_per_s =
      hot_mean_ms > 0.0 ? weight_bytes / (hot_mean_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0) : 0.0;
  if (output_host != nullptr) {
    output_host->resize(selection_count * kRoutedIntermediateSize);
    if (!output->CopyToHost(output_host->data(), output_host->size())) {
      return std::nullopt;
    }
  }
  return result;
}

std::optional<BenchmarkResult> RunLaunchPlanFp32AccumCase(
    const BenchmarkCase& benchmark_case,
    const BenchmarkOptions& options,
    std::vector<float>* output_host = nullptr) {
  const std::size_t selection_count = benchmark_case.prefix_tokens * kTopK;
  const auto expert_offsets = BuildExpertOffsets(selection_count);
  const std::size_t active_experts = CountActiveExperts(expert_offsets);
  const auto selected_indices_host = BuildSelectedIndicesForOffsets(expert_offsets);
  const std::vector<float> selected_weights_host(selection_count, 1.0f);

  auto selected_indices_device = DeviceArray<int>::CopyFromHost(selected_indices_host);
  auto selected_weights_device = DeviceArray<float>::CopyFromHost(selected_weights_host);
  auto routing = DeviceExpertRouting::Create(kRoutedExperts, selection_count);
  auto launch_plan = DeviceMoeLaunchPlan::Create(kRoutedExperts, selection_count);
  if (selected_indices_device == nullptr ||
      selected_weights_device == nullptr ||
      routing == nullptr ||
      !routing->valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      !RunDeviceExpertRouting(
          selected_indices_device->data(),
          selected_weights_device->data(),
          benchmark_case.prefix_tokens,
          kTopK,
          routing.get()) ||
      !BuildDeviceMoeLaunchPlan(*routing, selection_count, launch_plan.get()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  const auto routed_up = BuildUploadedRoutedUpViews();
  if (!routed_up.has_value()) {
    return std::nullopt;
  }
  auto routed_up_views_device =
      DeviceArray<FusedNvfp4WeightView>::CopyFromHost(routed_up->views);
  if (routed_up_views_device == nullptr) {
    return std::nullopt;
  }

  const auto input_host =
      MakePatternedValues(benchmark_case.prefix_tokens, kHiddenSize, 23, 0.03125f);
  auto input = DeviceTensorFp32::Create({benchmark_case.prefix_tokens, kHiddenSize});
  auto compact_input = DeviceTensorFp32::Create({selection_count, kHiddenSize});
  auto output = DeviceTensorFp32::Create({selection_count, kRoutedIntermediateSize});
  if (input == nullptr || compact_input == nullptr || output == nullptr) {
    return std::nullopt;
  }
  if (!input->CopyFromHost(input_host.data(), input_host.size())) {
    return std::nullopt;
  }
  if (!LaunchGatherPermutedRows(
          input->data(),
          routing->sorted_token_indices(),
          selection_count,
          benchmark_case.prefix_tokens,
          kHiddenSize,
          compact_input->data()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(kRoutedExperts, selection_count);
  if (!current_cta_capacity.has_value() || *current_cta_capacity == 0) {
    return std::nullopt;
  }
  const std::size_t output_row_tile_count =
      (kRoutedIntermediateSize + static_cast<std::size_t>(kPermutedOutputTile) - 1u) /
      static_cast<std::size_t>(kPermutedOutputTile);

  ScopedCudaEventTimer timer;
  if (!timer.valid()) {
    return std::nullopt;
  }

  const dim3 block(kPermutedThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(*current_cta_capacity * output_row_tile_count));
  auto run_once = [&]() {
    LaunchPlannedFp32AccumKernel<<<grid, block>>>(
        compact_input->data(),
        launch_plan->cta_count(),
        launch_plan->cta_expert_ids(),
        launch_plan->cta_row_starts(),
        launch_plan->cta_valid_rows(),
        static_cast<int>(output_row_tile_count),
        routed_up_views_device->data(),
        kRoutedIntermediateSize,
        output->data());
    return cudaGetLastError() == cudaSuccess;
  };

  for (std::size_t iteration = 0; iteration < options.warmup_iterations; ++iteration) {
    const auto elapsed = timer.Measure(run_once);
    if (!elapsed.has_value()) {
      return std::nullopt;
    }
  }

  const auto cold_ms = timer.Measure(run_once);
  if (!cold_ms.has_value()) {
    return std::nullopt;
  }

  double hot_sum_ms = 0.0;
  double hot_min_ms = 0.0;
  double hot_max_ms = 0.0;
  for (std::size_t iteration = 0; iteration < options.hot_iterations; ++iteration) {
    const auto hot_elapsed = timer.Measure(run_once);
    if (!hot_elapsed.has_value()) {
      return std::nullopt;
    }
    const double elapsed_ms = *hot_elapsed;
    hot_sum_ms += elapsed_ms;
    if (iteration == 0 || elapsed_ms < hot_min_ms) {
      hot_min_ms = elapsed_ms;
    }
    if (iteration == 0 || elapsed_ms > hot_max_ms) {
      hot_max_ms = elapsed_ms;
    }
  }

  const double hot_mean_ms =
      hot_sum_ms / static_cast<double>(std::max<std::size_t>(1, options.hot_iterations));
  const double logical_flops =
      2.0 * static_cast<double>(selection_count) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize);
  const double weight_bytes =
      static_cast<double>(active_experts) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize) *
      kNvfp4BytesPerWeight;

  BenchmarkResult result;
  result.variant_name = "launch_plan_flattened";
  result.case_name = benchmark_case.name;
  result.prefix_tokens = benchmark_case.prefix_tokens;
  result.selection_count = selection_count;
  result.padded_selection_count = selection_count;
  result.active_experts = active_experts;
  result.cold_ms = *cold_ms;
  result.hot_mean_ms = hot_mean_ms;
  result.hot_min_ms = hot_min_ms;
  result.hot_max_ms = hot_max_ms;
  result.hot_tflops =
      hot_mean_ms > 0.0 ? logical_flops / (hot_mean_ms / 1000.0) / 1.0e12 : 0.0;
  result.hot_weight_gib_per_s =
      hot_mean_ms > 0.0 ? weight_bytes / (hot_mean_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0) : 0.0;
  if (output_host != nullptr) {
    output_host->resize(selection_count * kRoutedIntermediateSize);
    if (!output->CopyToHost(output_host->data(), output_host->size())) {
      return std::nullopt;
    }
  }
  return result;
}

std::optional<BenchmarkResult> RunLaunchPlanWideFp32AccumCase(
    const BenchmarkCase& benchmark_case,
    const BenchmarkOptions& options,
    std::vector<float>* output_host = nullptr) {
  const std::size_t selection_count = benchmark_case.prefix_tokens * kTopK;
  const auto expert_offsets = BuildExpertOffsets(selection_count);
  const std::size_t active_experts = CountActiveExperts(expert_offsets);
  const auto selected_indices_host = BuildSelectedIndicesForOffsets(expert_offsets);
  const std::vector<float> selected_weights_host(selection_count, 1.0f);

  auto selected_indices_device = DeviceArray<int>::CopyFromHost(selected_indices_host);
  auto selected_weights_device = DeviceArray<float>::CopyFromHost(selected_weights_host);
  auto routing = DeviceExpertRouting::Create(kRoutedExperts, selection_count);
  auto launch_plan = DeviceMoeLaunchPlan::Create(kRoutedExperts, selection_count);
  if (selected_indices_device == nullptr ||
      selected_weights_device == nullptr ||
      routing == nullptr ||
      !routing->valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      !RunDeviceExpertRouting(
          selected_indices_device->data(),
          selected_weights_device->data(),
          benchmark_case.prefix_tokens,
          kTopK,
          routing.get()) ||
      !BuildDeviceMoeLaunchPlan(*routing, selection_count, launch_plan.get()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  const auto routed_up = BuildUploadedRoutedUpViews();
  if (!routed_up.has_value()) {
    return std::nullopt;
  }
  auto routed_up_views_device =
      DeviceArray<FusedNvfp4WeightView>::CopyFromHost(routed_up->views);
  if (routed_up_views_device == nullptr) {
    return std::nullopt;
  }

  const auto input_host =
      MakePatternedValues(benchmark_case.prefix_tokens, kHiddenSize, 23, 0.03125f);
  auto input = DeviceTensorFp32::Create({benchmark_case.prefix_tokens, kHiddenSize});
  auto compact_input = DeviceTensorFp32::Create({selection_count, kHiddenSize});
  auto output = DeviceTensorFp32::Create({selection_count, kRoutedIntermediateSize});
  if (input == nullptr || compact_input == nullptr || output == nullptr) {
    return std::nullopt;
  }
  if (!input->CopyFromHost(input_host.data(), input_host.size())) {
    return std::nullopt;
  }
  if (!LaunchGatherPermutedRows(
          input->data(),
          routing->sorted_token_indices(),
          selection_count,
          benchmark_case.prefix_tokens,
          kHiddenSize,
          compact_input->data()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(kRoutedExperts, selection_count);
  if (!current_cta_capacity.has_value() || *current_cta_capacity == 0) {
    return std::nullopt;
  }
  const std::size_t output_row_tile_count =
      (kRoutedIntermediateSize + static_cast<std::size_t>(kPermutedWideOutputTile) - 1u) /
      static_cast<std::size_t>(kPermutedWideOutputTile);

  ScopedCudaEventTimer timer;
  if (!timer.valid()) {
    return std::nullopt;
  }

  const dim3 block(kPermutedWideThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(*current_cta_capacity * output_row_tile_count));
  auto run_once = [&]() {
    LaunchPlannedWideFp32AccumKernel<<<grid, block>>>(
        compact_input->data(),
        launch_plan->cta_count(),
        launch_plan->cta_expert_ids(),
        launch_plan->cta_row_starts(),
        launch_plan->cta_valid_rows(),
        static_cast<int>(output_row_tile_count),
        routed_up_views_device->data(),
        kRoutedIntermediateSize,
        output->data());
    return cudaGetLastError() == cudaSuccess;
  };

  for (std::size_t iteration = 0; iteration < options.warmup_iterations; ++iteration) {
    const auto elapsed = timer.Measure(run_once);
    if (!elapsed.has_value()) {
      return std::nullopt;
    }
  }

  const auto cold_ms = timer.Measure(run_once);
  if (!cold_ms.has_value()) {
    return std::nullopt;
  }

  double hot_sum_ms = 0.0;
  double hot_min_ms = 0.0;
  double hot_max_ms = 0.0;
  for (std::size_t iteration = 0; iteration < options.hot_iterations; ++iteration) {
    const auto hot_elapsed = timer.Measure(run_once);
    if (!hot_elapsed.has_value()) {
      return std::nullopt;
    }
    const double elapsed_ms = *hot_elapsed;
    hot_sum_ms += elapsed_ms;
    if (iteration == 0 || elapsed_ms < hot_min_ms) {
      hot_min_ms = elapsed_ms;
    }
    if (iteration == 0 || elapsed_ms > hot_max_ms) {
      hot_max_ms = elapsed_ms;
    }
  }

  const double hot_mean_ms =
      hot_sum_ms / static_cast<double>(std::max<std::size_t>(1, options.hot_iterations));
  const double logical_flops =
      2.0 * static_cast<double>(selection_count) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize);
  const double weight_bytes =
      static_cast<double>(active_experts) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize) *
      kNvfp4BytesPerWeight;

  BenchmarkResult result;
  result.variant_name = "launch_plan_flattened_tile8";
  result.case_name = benchmark_case.name;
  result.prefix_tokens = benchmark_case.prefix_tokens;
  result.selection_count = selection_count;
  result.padded_selection_count = selection_count;
  result.active_experts = active_experts;
  result.cold_ms = *cold_ms;
  result.hot_mean_ms = hot_mean_ms;
  result.hot_min_ms = hot_min_ms;
  result.hot_max_ms = hot_max_ms;
  result.hot_tflops =
      hot_mean_ms > 0.0 ? logical_flops / (hot_mean_ms / 1000.0) / 1.0e12 : 0.0;
  result.hot_weight_gib_per_s =
      hot_mean_ms > 0.0 ? weight_bytes / (hot_mean_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0) : 0.0;
  if (output_host != nullptr) {
    output_host->resize(selection_count * kRoutedIntermediateSize);
    if (!output->CopyToHost(output_host->data(), output_host->size())) {
      return std::nullopt;
    }
  }
  return result;
}

std::optional<BenchmarkResult> RunLaunchPlanExactTaskTile8Case(
    const BenchmarkCase& benchmark_case,
    const BenchmarkOptions& options,
    std::vector<float>* output_host = nullptr) {
  const std::size_t selection_count = benchmark_case.prefix_tokens * kTopK;
  const auto expert_offsets = BuildExpertOffsets(selection_count);
  const std::size_t active_experts = CountActiveExperts(expert_offsets);
  const auto selected_indices_host = BuildSelectedIndicesForOffsets(expert_offsets);
  const std::vector<float> selected_weights_host(selection_count, 1.0f);

  auto selected_indices_device = DeviceArray<int>::CopyFromHost(selected_indices_host);
  auto selected_weights_device = DeviceArray<float>::CopyFromHost(selected_weights_host);
  auto routing = DeviceExpertRouting::Create(kRoutedExperts, selection_count);
  auto launch_plan =
      DeviceMoeLaunchPlan::Create(kRoutedExperts, selection_count, kRoutedIntermediateSize);
  if (selected_indices_device == nullptr ||
      selected_weights_device == nullptr ||
      routing == nullptr ||
      !routing->valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      !RunDeviceExpertRouting(
          selected_indices_device->data(),
          selected_weights_device->data(),
          benchmark_case.prefix_tokens,
          kTopK,
          routing.get()) ||
      !BuildDeviceMoeLaunchPlan(*routing, selection_count, launch_plan.get()) ||
      !BuildDeviceMoeExactTaskMap(
          kRoutedIntermediateSize,
          nemotron::kMoeLaunchPlanOutputTile,
          launch_plan.get()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  std::vector<int> task_count_host;
  if (!CopyDeviceBufferToHost(launch_plan->task_count(), 1, &task_count_host) ||
      task_count_host.empty() ||
      task_count_host[0] <= 0) {
    return std::nullopt;
  }

  const auto routed_up = BuildUploadedRoutedUpViews();
  if (!routed_up.has_value()) {
    return std::nullopt;
  }
  auto routed_up_views_device =
      DeviceArray<FusedNvfp4WeightView>::CopyFromHost(routed_up->views);
  if (routed_up_views_device == nullptr) {
    return std::nullopt;
  }

  const auto input_host =
      MakePatternedValues(benchmark_case.prefix_tokens, kHiddenSize, 23, 0.03125f);
  auto input = DeviceTensorFp32::Create({benchmark_case.prefix_tokens, kHiddenSize});
  auto compact_input = DeviceTensorFp32::Create({selection_count, kHiddenSize});
  auto output = DeviceTensorFp32::Create({selection_count, kRoutedIntermediateSize});
  if (input == nullptr || compact_input == nullptr || output == nullptr) {
    return std::nullopt;
  }
  if (!input->CopyFromHost(input_host.data(), input_host.size())) {
    return std::nullopt;
  }
  if (!LaunchGatherPermutedRows(
          input->data(),
          routing->sorted_token_indices(),
          selection_count,
          benchmark_case.prefix_tokens,
          kHiddenSize,
          compact_input->data()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  ScopedCudaEventTimer timer;
  if (!timer.valid()) {
    return std::nullopt;
  }

  const dim3 block(kPermutedWideThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(task_count_host[0]));
  auto run_once = [&]() {
    LaunchPlanExactTaskTile8Kernel<<<grid, block>>>(
        compact_input->data(),
        launch_plan->task_expert_ids(),
        launch_plan->task_row_starts(),
        launch_plan->task_valid_rows(),
        launch_plan->task_output_row_bases(),
        routed_up_views_device->data(),
        kRoutedIntermediateSize,
        output->data());
    return cudaGetLastError() == cudaSuccess;
  };

  for (std::size_t iteration = 0; iteration < options.warmup_iterations; ++iteration) {
    const auto elapsed = timer.Measure(run_once);
    if (!elapsed.has_value()) {
      return std::nullopt;
    }
  }

  const auto cold_ms = timer.Measure(run_once);
  if (!cold_ms.has_value()) {
    return std::nullopt;
  }

  double hot_sum_ms = 0.0;
  double hot_min_ms = 0.0;
  double hot_max_ms = 0.0;
  for (std::size_t iteration = 0; iteration < options.hot_iterations; ++iteration) {
    const auto hot_elapsed = timer.Measure(run_once);
    if (!hot_elapsed.has_value()) {
      return std::nullopt;
    }
    const double elapsed_ms = *hot_elapsed;
    hot_sum_ms += elapsed_ms;
    if (iteration == 0 || elapsed_ms < hot_min_ms) {
      hot_min_ms = elapsed_ms;
    }
    if (iteration == 0 || elapsed_ms > hot_max_ms) {
      hot_max_ms = elapsed_ms;
    }
  }

  const double hot_mean_ms =
      hot_sum_ms / static_cast<double>(std::max<std::size_t>(1, options.hot_iterations));
  const double logical_flops =
      2.0 * static_cast<double>(selection_count) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize);
  const double weight_bytes =
      static_cast<double>(active_experts) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize) *
      kNvfp4BytesPerWeight;

  BenchmarkResult result;
  result.variant_name = "launch_plan_exact_task_tile8";
  result.case_name = benchmark_case.name;
  result.prefix_tokens = benchmark_case.prefix_tokens;
  result.selection_count = selection_count;
  result.padded_selection_count = selection_count;
  result.active_experts = active_experts;
  result.cold_ms = *cold_ms;
  result.hot_mean_ms = hot_mean_ms;
  result.hot_min_ms = hot_min_ms;
  result.hot_max_ms = hot_max_ms;
  result.hot_tflops =
      hot_mean_ms > 0.0 ? logical_flops / (hot_mean_ms / 1000.0) / 1.0e12 : 0.0;
  result.hot_weight_gib_per_s =
      hot_mean_ms > 0.0 ? weight_bytes / (hot_mean_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0) : 0.0;
  if (output_host != nullptr) {
    output_host->resize(selection_count * kRoutedIntermediateSize);
    if (!output->CopyToHost(output_host->data(), output_host->size())) {
      return std::nullopt;
    }
  }
  return result;
}

std::optional<BenchmarkResult> RunLaunchPlanGroupedTile32Case(
    const BenchmarkCase& benchmark_case,
    const BenchmarkOptions& options,
    std::vector<float>* output_host = nullptr) {
  const std::size_t selection_count = benchmark_case.prefix_tokens * kTopK;
  const auto expert_offsets = BuildExpertOffsets(selection_count);
  const std::size_t active_experts = CountActiveExperts(expert_offsets);
  const auto selected_indices_host = BuildSelectedIndicesForOffsets(expert_offsets);
  const std::vector<float> selected_weights_host(selection_count, 1.0f);

  auto selected_indices_device = DeviceArray<int>::CopyFromHost(selected_indices_host);
  auto selected_weights_device = DeviceArray<float>::CopyFromHost(selected_weights_host);
  auto routing = DeviceExpertRouting::Create(kRoutedExperts, selection_count);
  auto launch_plan = DeviceMoeLaunchPlan::Create(kRoutedExperts, selection_count);
  if (selected_indices_device == nullptr ||
      selected_weights_device == nullptr ||
      routing == nullptr ||
      !routing->valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      !RunDeviceExpertRouting(
          selected_indices_device->data(),
          selected_weights_device->data(),
          benchmark_case.prefix_tokens,
          kTopK,
          routing.get()) ||
      !BuildDeviceMoeLaunchPlan(*routing, selection_count, launch_plan.get()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(kRoutedExperts, selection_count);
  if (!current_cta_capacity.has_value() || *current_cta_capacity == 0) {
    return std::nullopt;
  }

  const auto routed_up = BuildUploadedRoutedUpViews();
  if (!routed_up.has_value()) {
    return std::nullopt;
  }
  auto routed_up_views_device =
      DeviceArray<FusedNvfp4WeightView>::CopyFromHost(routed_up->views);
  if (routed_up_views_device == nullptr) {
    return std::nullopt;
  }

  const auto input_host =
      MakePatternedValues(benchmark_case.prefix_tokens, kHiddenSize, 23, 0.03125f);
  auto input = DeviceTensorFp32::Create({benchmark_case.prefix_tokens, kHiddenSize});
  auto compact_input = DeviceTensorFp32::Create({selection_count, kHiddenSize});
  auto output = DeviceTensorFp32::Create({selection_count, kRoutedIntermediateSize});
  if (input == nullptr || compact_input == nullptr || output == nullptr) {
    return std::nullopt;
  }
  if (!input->CopyFromHost(input_host.data(), input_host.size())) {
    return std::nullopt;
  }
  if (!LaunchGatherPermutedRows(
          input->data(),
          routing->sorted_token_indices(),
          selection_count,
          benchmark_case.prefix_tokens,
          kHiddenSize,
          compact_input->data()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  const std::size_t output_row_tile_count =
      (kRoutedIntermediateSize + static_cast<std::size_t>(kPermutedGroupedOutputTile) - 1u) /
      static_cast<std::size_t>(kPermutedGroupedOutputTile);

  ScopedCudaEventTimer timer;
  if (!timer.valid()) {
    return std::nullopt;
  }

  const dim3 block(kPermutedWideThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  auto run_once = [&]() {
    LaunchPlanGroupedTile32Kernel<<<grid, block>>>(
        compact_input->data(),
        launch_plan->cta_count(),
        launch_plan->cta_expert_ids(),
        launch_plan->cta_row_starts(),
        launch_plan->cta_valid_rows(),
        routed_up_views_device->data(),
        kRoutedIntermediateSize,
        output->data());
    return cudaGetLastError() == cudaSuccess;
  };

  for (std::size_t iteration = 0; iteration < options.warmup_iterations; ++iteration) {
    const auto elapsed = timer.Measure(run_once);
    if (!elapsed.has_value()) {
      return std::nullopt;
    }
  }

  const auto cold_ms = timer.Measure(run_once);
  if (!cold_ms.has_value()) {
    return std::nullopt;
  }

  double hot_sum_ms = 0.0;
  double hot_min_ms = 0.0;
  double hot_max_ms = 0.0;
  for (std::size_t iteration = 0; iteration < options.hot_iterations; ++iteration) {
    const auto hot_elapsed = timer.Measure(run_once);
    if (!hot_elapsed.has_value()) {
      return std::nullopt;
    }
    const double elapsed_ms = *hot_elapsed;
    hot_sum_ms += elapsed_ms;
    if (iteration == 0 || elapsed_ms < hot_min_ms) {
      hot_min_ms = elapsed_ms;
    }
    if (iteration == 0 || elapsed_ms > hot_max_ms) {
      hot_max_ms = elapsed_ms;
    }
  }

  const double hot_mean_ms =
      hot_sum_ms / static_cast<double>(std::max<std::size_t>(1, options.hot_iterations));
  const double logical_flops =
      2.0 * static_cast<double>(selection_count) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize);
  const double weight_bytes =
      static_cast<double>(active_experts) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize) *
      kNvfp4BytesPerWeight;

  BenchmarkResult result;
  result.variant_name = "launch_plan_grouped_tile32";
  result.case_name = benchmark_case.name;
  result.prefix_tokens = benchmark_case.prefix_tokens;
  result.selection_count = selection_count;
  result.padded_selection_count = selection_count;
  result.active_experts = active_experts;
  result.cold_ms = *cold_ms;
  result.hot_mean_ms = hot_mean_ms;
  result.hot_min_ms = hot_min_ms;
  result.hot_max_ms = hot_max_ms;
  result.hot_tflops =
      hot_mean_ms > 0.0 ? logical_flops / (hot_mean_ms / 1000.0) / 1.0e12 : 0.0;
  result.hot_weight_gib_per_s =
      hot_mean_ms > 0.0 ? weight_bytes / (hot_mean_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0) : 0.0;
  if (output_host != nullptr) {
    output_host->resize(selection_count * kRoutedIntermediateSize);
    if (!output->CopyToHost(output_host->data(), output_host->size())) {
      return std::nullopt;
    }
  }
  return result;
}

std::optional<BenchmarkResult> RunLaunchPlanCooperativeMxNCase(
    const BenchmarkCase& benchmark_case,
    const BenchmarkOptions& options,
    std::vector<float>* output_host = nullptr) {
  const std::size_t selection_count = benchmark_case.prefix_tokens * kTopK;
  const auto expert_offsets = BuildExpertOffsets(selection_count);
  const std::size_t active_experts = CountActiveExperts(expert_offsets);
  const auto selected_indices_host = BuildSelectedIndicesForOffsets(expert_offsets);
  const std::vector<float> selected_weights_host(selection_count, 1.0f);

  auto selected_indices_device = DeviceArray<int>::CopyFromHost(selected_indices_host);
  auto selected_weights_device = DeviceArray<float>::CopyFromHost(selected_weights_host);
  auto routing = DeviceExpertRouting::Create(kRoutedExperts, selection_count);
  auto launch_plan = DeviceMoeLaunchPlan::Create(kRoutedExperts, selection_count);
  if (selected_indices_device == nullptr ||
      selected_weights_device == nullptr ||
      routing == nullptr ||
      !routing->valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      !RunDeviceExpertRouting(
          selected_indices_device->data(),
          selected_weights_device->data(),
          benchmark_case.prefix_tokens,
          kTopK,
          routing.get()) ||
      !BuildDeviceMoeLaunchPlan(*routing, selection_count, launch_plan.get()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(kRoutedExperts, selection_count);
  if (!current_cta_capacity.has_value() || *current_cta_capacity == 0) {
    return std::nullopt;
  }

  const auto routed_up = BuildUploadedRoutedUpViews();
  if (!routed_up.has_value()) {
    return std::nullopt;
  }
  auto routed_up_views_device =
      DeviceArray<FusedNvfp4WeightView>::CopyFromHost(routed_up->views);
  if (routed_up_views_device == nullptr) {
    return std::nullopt;
  }

  const auto input_host =
      MakePatternedValues(benchmark_case.prefix_tokens, kHiddenSize, 23, 0.03125f);
  auto input = DeviceTensorFp32::Create({benchmark_case.prefix_tokens, kHiddenSize});
  auto compact_input = DeviceTensorFp32::Create({selection_count, kHiddenSize});
  auto output = DeviceTensorFp32::Create({selection_count, kRoutedIntermediateSize});
  if (input == nullptr || compact_input == nullptr || output == nullptr) {
    return std::nullopt;
  }
  if (!input->CopyFromHost(input_host.data(), input_host.size())) {
    return std::nullopt;
  }
  if (!LaunchGatherPermutedRows(
          input->data(),
          routing->sorted_token_indices(),
          selection_count,
          benchmark_case.prefix_tokens,
          kHiddenSize,
          compact_input->data()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  const std::size_t output_row_tile_count =
      (kRoutedIntermediateSize + static_cast<std::size_t>(kPermutedCoopOutputTile) - 1u) /
      static_cast<std::size_t>(kPermutedCoopOutputTile);

  ScopedCudaEventTimer timer;
  if (!timer.valid()) {
    return std::nullopt;
  }

  const dim3 block(kPermutedCoopThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  auto run_once = [&]() {
    LaunchPlanCooperativeMxNKernel<<<grid, block>>>(
        compact_input->data(),
        launch_plan->cta_count(),
        launch_plan->cta_expert_ids(),
        launch_plan->cta_row_starts(),
        launch_plan->cta_valid_rows(),
        routed_up_views_device->data(),
        kRoutedIntermediateSize,
        output->data());
    return cudaGetLastError() == cudaSuccess;
  };

  for (std::size_t iteration = 0; iteration < options.warmup_iterations; ++iteration) {
    const auto elapsed = timer.Measure(run_once);
    if (!elapsed.has_value()) {
      return std::nullopt;
    }
  }

  const auto cold_ms = timer.Measure(run_once);
  if (!cold_ms.has_value()) {
    return std::nullopt;
  }

  double hot_sum_ms = 0.0;
  double hot_min_ms = 0.0;
  double hot_max_ms = 0.0;
  for (std::size_t iteration = 0; iteration < options.hot_iterations; ++iteration) {
    const auto hot_elapsed = timer.Measure(run_once);
    if (!hot_elapsed.has_value()) {
      return std::nullopt;
    }
    const double elapsed_ms = *hot_elapsed;
    hot_sum_ms += elapsed_ms;
    if (iteration == 0 || elapsed_ms < hot_min_ms) {
      hot_min_ms = elapsed_ms;
    }
    if (iteration == 0 || elapsed_ms > hot_max_ms) {
      hot_max_ms = elapsed_ms;
    }
  }

  const double hot_mean_ms =
      hot_sum_ms / static_cast<double>(std::max<std::size_t>(1, options.hot_iterations));
  const double logical_flops =
      2.0 * static_cast<double>(selection_count) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize);
  const double weight_bytes =
      static_cast<double>(active_experts) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize) *
      kNvfp4BytesPerWeight;

  BenchmarkResult result;
  result.variant_name = "launch_plan_cooperative_m8n32";
  result.case_name = benchmark_case.name;
  result.prefix_tokens = benchmark_case.prefix_tokens;
  result.selection_count = selection_count;
  result.padded_selection_count = selection_count;
  result.active_experts = active_experts;
  result.cold_ms = *cold_ms;
  result.hot_mean_ms = hot_mean_ms;
  result.hot_min_ms = hot_min_ms;
  result.hot_max_ms = hot_max_ms;
  result.hot_tflops =
      hot_mean_ms > 0.0 ? logical_flops / (hot_mean_ms / 1000.0) / 1.0e12 : 0.0;
  result.hot_weight_gib_per_s =
      hot_mean_ms > 0.0 ? weight_bytes / (hot_mean_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0) : 0.0;
  if (output_host != nullptr) {
    output_host->resize(selection_count * kRoutedIntermediateSize);
    if (!output->CopyToHost(output_host->data(), output_host->size())) {
      return std::nullopt;
    }
  }
  return result;
}

std::optional<BenchmarkResult> RunLaunchPlanMicroTile2Case(
    const BenchmarkCase& benchmark_case,
    const BenchmarkOptions& options,
    std::vector<float>* output_host = nullptr) {
  const std::size_t selection_count = benchmark_case.prefix_tokens * kTopK;
  const auto expert_offsets = BuildExpertOffsets(selection_count);
  const std::size_t active_experts = CountActiveExperts(expert_offsets);
  const auto selected_indices_host = BuildSelectedIndicesForOffsets(expert_offsets);
  const std::vector<float> selected_weights_host(selection_count, 1.0f);

  auto selected_indices_device = DeviceArray<int>::CopyFromHost(selected_indices_host);
  auto selected_weights_device = DeviceArray<float>::CopyFromHost(selected_weights_host);
  auto routing = DeviceExpertRouting::Create(kRoutedExperts, selection_count);
  auto launch_plan = DeviceMoeLaunchPlan::Create(kRoutedExperts, selection_count);
  if (selected_indices_device == nullptr ||
      selected_weights_device == nullptr ||
      routing == nullptr ||
      !routing->valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      !RunDeviceExpertRouting(
          selected_indices_device->data(),
          selected_weights_device->data(),
          benchmark_case.prefix_tokens,
          kTopK,
          routing.get()) ||
      !BuildDeviceMoeLaunchPlan(*routing, selection_count, launch_plan.get()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(kRoutedExperts, selection_count);
  if (!current_cta_capacity.has_value() || *current_cta_capacity == 0) {
    return std::nullopt;
  }

  const auto routed_up = BuildUploadedRoutedUpViews();
  if (!routed_up.has_value()) {
    return std::nullopt;
  }
  auto routed_up_views_device =
      DeviceArray<FusedNvfp4WeightView>::CopyFromHost(routed_up->views);
  if (routed_up_views_device == nullptr) {
    return std::nullopt;
  }

  const auto input_host =
      MakePatternedValues(benchmark_case.prefix_tokens, kHiddenSize, 23, 0.03125f);
  auto input = DeviceTensorFp32::Create({benchmark_case.prefix_tokens, kHiddenSize});
  auto compact_input = DeviceTensorFp32::Create({selection_count, kHiddenSize});
  auto output = DeviceTensorFp32::Create({selection_count, kRoutedIntermediateSize});
  if (input == nullptr || compact_input == nullptr || output == nullptr) {
    return std::nullopt;
  }
  if (!input->CopyFromHost(input_host.data(), input_host.size())) {
    return std::nullopt;
  }
  if (!LaunchGatherPermutedRows(
          input->data(),
          routing->sorted_token_indices(),
          selection_count,
          benchmark_case.prefix_tokens,
          kHiddenSize,
          compact_input->data()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  const std::size_t output_row_tile_count =
      (kRoutedIntermediateSize + static_cast<std::size_t>(kPermutedCoopOutputTile) - 1u) /
      static_cast<std::size_t>(kPermutedCoopOutputTile);

  ScopedCudaEventTimer timer;
  if (!timer.valid()) {
    return std::nullopt;
  }

  const dim3 block(kPermutedMicroTileThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  auto run_once = [&]() {
    LaunchPlanMicroTile2Kernel<<<grid, block>>>(
        compact_input->data(),
        launch_plan->cta_count(),
        launch_plan->cta_expert_ids(),
        launch_plan->cta_row_starts(),
        launch_plan->cta_valid_rows(),
        routed_up_views_device->data(),
        kRoutedIntermediateSize,
        output->data());
    return cudaGetLastError() == cudaSuccess;
  };

  for (std::size_t iteration = 0; iteration < options.warmup_iterations; ++iteration) {
    const auto elapsed = timer.Measure(run_once);
    if (!elapsed.has_value()) {
      return std::nullopt;
    }
  }

  const auto cold_ms = timer.Measure(run_once);
  if (!cold_ms.has_value()) {
    return std::nullopt;
  }

  double hot_sum_ms = 0.0;
  double hot_min_ms = 0.0;
  double hot_max_ms = 0.0;
  for (std::size_t iteration = 0; iteration < options.hot_iterations; ++iteration) {
    const auto hot_elapsed = timer.Measure(run_once);
    if (!hot_elapsed.has_value()) {
      return std::nullopt;
    }
    const double elapsed_ms = *hot_elapsed;
    hot_sum_ms += elapsed_ms;
    if (iteration == 0 || elapsed_ms < hot_min_ms) {
      hot_min_ms = elapsed_ms;
    }
    if (iteration == 0 || elapsed_ms > hot_max_ms) {
      hot_max_ms = elapsed_ms;
    }
  }

  const double hot_mean_ms =
      hot_sum_ms / static_cast<double>(std::max<std::size_t>(1, options.hot_iterations));
  const double logical_flops =
      2.0 * static_cast<double>(selection_count) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize);
  const double weight_bytes =
      static_cast<double>(active_experts) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize) *
      kNvfp4BytesPerWeight;

  BenchmarkResult result;
  result.variant_name = "launch_plan_microtile2_m8n32";
  result.case_name = benchmark_case.name;
  result.prefix_tokens = benchmark_case.prefix_tokens;
  result.selection_count = selection_count;
  result.padded_selection_count = selection_count;
  result.active_experts = active_experts;
  result.cold_ms = *cold_ms;
  result.hot_mean_ms = hot_mean_ms;
  result.hot_min_ms = hot_min_ms;
  result.hot_max_ms = hot_max_ms;
  result.hot_tflops =
      hot_mean_ms > 0.0 ? logical_flops / (hot_mean_ms / 1000.0) / 1.0e12 : 0.0;
  result.hot_weight_gib_per_s =
      hot_mean_ms > 0.0 ? weight_bytes / (hot_mean_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0) : 0.0;
  if (output_host != nullptr) {
    output_host->resize(selection_count * kRoutedIntermediateSize);
    if (!output->CopyToHost(output_host->data(), output_host->size())) {
      return std::nullopt;
    }
  }
  return result;
}

std::optional<BenchmarkResult> RunLaunchPlanWmmaBf16Case(
    const BenchmarkCase& benchmark_case,
    const BenchmarkOptions& options,
    std::vector<float>* output_host = nullptr) {
  const std::size_t selection_count = benchmark_case.prefix_tokens * kTopK;
  const auto expert_offsets = BuildExpertOffsets(selection_count);
  const std::size_t active_experts = CountActiveExperts(expert_offsets);
  const auto selected_indices_host = BuildSelectedIndicesForOffsets(expert_offsets);
  const std::vector<float> selected_weights_host(selection_count, 1.0f);

  auto selected_indices_device = DeviceArray<int>::CopyFromHost(selected_indices_host);
  auto selected_weights_device = DeviceArray<float>::CopyFromHost(selected_weights_host);
  auto routing = DeviceExpertRouting::Create(kRoutedExperts, selection_count);
  auto launch_plan = DeviceMoeLaunchPlan::Create(kRoutedExperts, selection_count);
  if (selected_indices_device == nullptr ||
      selected_weights_device == nullptr ||
      routing == nullptr ||
      !routing->valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      !RunDeviceExpertRouting(
          selected_indices_device->data(),
          selected_weights_device->data(),
          benchmark_case.prefix_tokens,
          kTopK,
          routing.get()) ||
      !BuildDeviceMoeLaunchPlan(*routing, selection_count, launch_plan.get()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(kRoutedExperts, selection_count);
  if (!current_cta_capacity.has_value() || *current_cta_capacity == 0) {
    return std::nullopt;
  }

  const auto routed_up = BuildUploadedRoutedUpViews();
  if (!routed_up.has_value()) {
    return std::nullopt;
  }
  auto routed_up_views_device =
      DeviceArray<FusedNvfp4WeightView>::CopyFromHost(routed_up->views);
  if (routed_up_views_device == nullptr) {
    return std::nullopt;
  }

  const auto input_host =
      MakePatternedValues(benchmark_case.prefix_tokens, kHiddenSize, 23, 0.03125f);
  auto input = DeviceTensorFp32::Create({benchmark_case.prefix_tokens, kHiddenSize});
  auto compact_input = DeviceTensorFp32::Create({selection_count, kHiddenSize});
  auto output = DeviceTensorFp32::Create({selection_count, kRoutedIntermediateSize});
  if (input == nullptr || compact_input == nullptr || output == nullptr) {
    return std::nullopt;
  }
  if (!input->CopyFromHost(input_host.data(), input_host.size())) {
    return std::nullopt;
  }
  if (!LaunchGatherPermutedRows(
          input->data(),
          routing->sorted_token_indices(),
          selection_count,
          benchmark_case.prefix_tokens,
          kHiddenSize,
          compact_input->data()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  const std::size_t output_row_tile_count =
      (kRoutedIntermediateSize + static_cast<std::size_t>(kWmmaOutputTile) - 1u) /
      static_cast<std::size_t>(kWmmaOutputTile);

  ScopedCudaEventTimer timer;
  if (!timer.valid()) {
    return std::nullopt;
  }

  const dim3 block(kWmmaThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  auto run_once = [&]() {
    LaunchPlanWmmaBf16Kernel<<<grid, block>>>(
        compact_input->data(),
        launch_plan->cta_count(),
        launch_plan->cta_expert_ids(),
        launch_plan->cta_row_starts(),
        launch_plan->cta_valid_rows(),
        routed_up_views_device->data(),
        kRoutedIntermediateSize,
        output->data());
    return cudaGetLastError() == cudaSuccess;
  };

  for (std::size_t iteration = 0; iteration < options.warmup_iterations; ++iteration) {
    const auto elapsed = timer.Measure(run_once);
    if (!elapsed.has_value()) {
      return std::nullopt;
    }
  }

  const auto cold_ms = timer.Measure(run_once);
  if (!cold_ms.has_value()) {
    return std::nullopt;
  }

  double hot_sum_ms = 0.0;
  double hot_min_ms = 0.0;
  double hot_max_ms = 0.0;
  for (std::size_t iteration = 0; iteration < options.hot_iterations; ++iteration) {
    const auto hot_elapsed = timer.Measure(run_once);
    if (!hot_elapsed.has_value()) {
      return std::nullopt;
    }
    const double elapsed_ms = *hot_elapsed;
    hot_sum_ms += elapsed_ms;
    if (iteration == 0 || elapsed_ms < hot_min_ms) {
      hot_min_ms = elapsed_ms;
    }
    if (iteration == 0 || elapsed_ms > hot_max_ms) {
      hot_max_ms = elapsed_ms;
    }
  }

  const double hot_mean_ms =
      hot_sum_ms / static_cast<double>(std::max<std::size_t>(1, options.hot_iterations));
  const double logical_flops =
      2.0 * static_cast<double>(selection_count) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize);
  const double weight_bytes =
      static_cast<double>(active_experts) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize) *
      kNvfp4BytesPerWeight;

  BenchmarkResult result;
  result.variant_name = "launch_plan_wmma_bf16_m16n32k16";
  result.case_name = benchmark_case.name;
  result.prefix_tokens = benchmark_case.prefix_tokens;
  result.selection_count = selection_count;
  result.padded_selection_count = selection_count;
  result.active_experts = active_experts;
  result.cold_ms = *cold_ms;
  result.hot_mean_ms = hot_mean_ms;
  result.hot_min_ms = hot_min_ms;
  result.hot_max_ms = hot_max_ms;
  result.hot_tflops =
      hot_mean_ms > 0.0 ? logical_flops / (hot_mean_ms / 1000.0) / 1.0e12 : 0.0;
  result.hot_weight_gib_per_s =
      hot_mean_ms > 0.0 ? weight_bytes / (hot_mean_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0) : 0.0;
  if (output_host != nullptr) {
    output_host->resize(selection_count * kRoutedIntermediateSize);
    if (!output->CopyToHost(output_host->data(), output_host->size())) {
      return std::nullopt;
    }
  }
  return result;
}

std::optional<BenchmarkResult> RunLaunchPlanBf16ScalarCase(
    const BenchmarkCase& benchmark_case,
    const BenchmarkOptions& options,
    std::vector<float>* output_host = nullptr) {
  const std::size_t selection_count = benchmark_case.prefix_tokens * kTopK;
  const auto expert_offsets = BuildExpertOffsets(selection_count);
  const std::size_t active_experts = CountActiveExperts(expert_offsets);
  const auto selected_indices_host = BuildSelectedIndicesForOffsets(expert_offsets);
  const std::vector<float> selected_weights_host(selection_count, 1.0f);

  auto selected_indices_device = DeviceArray<int>::CopyFromHost(selected_indices_host);
  auto selected_weights_device = DeviceArray<float>::CopyFromHost(selected_weights_host);
  auto routing = DeviceExpertRouting::Create(kRoutedExperts, selection_count);
  auto launch_plan = DeviceMoeLaunchPlan::Create(kRoutedExperts, selection_count);
  if (selected_indices_device == nullptr ||
      selected_weights_device == nullptr ||
      routing == nullptr ||
      !routing->valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      !RunDeviceExpertRouting(
          selected_indices_device->data(),
          selected_weights_device->data(),
          benchmark_case.prefix_tokens,
          kTopK,
          routing.get()) ||
      !BuildDeviceMoeLaunchPlan(*routing, selection_count, launch_plan.get()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(kRoutedExperts, selection_count);
  if (!current_cta_capacity.has_value() || *current_cta_capacity == 0) {
    return std::nullopt;
  }

  const auto routed_up = BuildUploadedRoutedUpViews();
  if (!routed_up.has_value()) {
    return std::nullopt;
  }
  auto routed_up_views_device =
      DeviceArray<FusedNvfp4WeightView>::CopyFromHost(routed_up->views);
  if (routed_up_views_device == nullptr) {
    return std::nullopt;
  }

  const auto input_host =
      MakePatternedValues(benchmark_case.prefix_tokens, kHiddenSize, 23, 0.03125f);
  auto input = DeviceTensorFp32::Create({benchmark_case.prefix_tokens, kHiddenSize});
  auto compact_input = DeviceTensorFp32::Create({selection_count, kHiddenSize});
  auto output = DeviceTensorFp32::Create({selection_count, kRoutedIntermediateSize});
  if (input == nullptr || compact_input == nullptr || output == nullptr) {
    return std::nullopt;
  }
  if (!input->CopyFromHost(input_host.data(), input_host.size())) {
    return std::nullopt;
  }
  if (!LaunchGatherPermutedRows(
          input->data(),
          routing->sorted_token_indices(),
          selection_count,
          benchmark_case.prefix_tokens,
          kHiddenSize,
          compact_input->data()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  const std::size_t output_row_tile_count =
      (kRoutedIntermediateSize + static_cast<std::size_t>(kPermutedCoopOutputTile) - 1u) /
      static_cast<std::size_t>(kPermutedCoopOutputTile);

  ScopedCudaEventTimer timer;
  if (!timer.valid()) {
    return std::nullopt;
  }

  const dim3 block(kPermutedCoopThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  auto run_once = [&]() {
    LaunchPlanBf16ScalarKernel<<<grid, block>>>(
        compact_input->data(),
        launch_plan->cta_count(),
        launch_plan->cta_expert_ids(),
        launch_plan->cta_row_starts(),
        launch_plan->cta_valid_rows(),
        routed_up_views_device->data(),
        kRoutedIntermediateSize,
        output->data());
    return cudaGetLastError() == cudaSuccess;
  };

  for (std::size_t iteration = 0; iteration < options.warmup_iterations; ++iteration) {
    const auto elapsed = timer.Measure(run_once);
    if (!elapsed.has_value()) {
      return std::nullopt;
    }
  }

  const auto cold_ms = timer.Measure(run_once);
  if (!cold_ms.has_value()) {
    return std::nullopt;
  }

  double hot_sum_ms = 0.0;
  double hot_min_ms = 0.0;
  double hot_max_ms = 0.0;
  for (std::size_t iteration = 0; iteration < options.hot_iterations; ++iteration) {
    const auto hot_elapsed = timer.Measure(run_once);
    if (!hot_elapsed.has_value()) {
      return std::nullopt;
    }
    const double elapsed_ms = *hot_elapsed;
    hot_sum_ms += elapsed_ms;
    if (iteration == 0 || elapsed_ms < hot_min_ms) {
      hot_min_ms = elapsed_ms;
    }
    if (iteration == 0 || elapsed_ms > hot_max_ms) {
      hot_max_ms = elapsed_ms;
    }
  }

  const double hot_mean_ms =
      hot_sum_ms / static_cast<double>(std::max<std::size_t>(1, options.hot_iterations));
  const double logical_flops =
      2.0 * static_cast<double>(selection_count) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize);
  const double weight_bytes =
      static_cast<double>(active_experts) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize) *
      kNvfp4BytesPerWeight;

  BenchmarkResult result;
  result.variant_name = "launch_plan_bf16_scalar_m8n32";
  result.case_name = benchmark_case.name;
  result.prefix_tokens = benchmark_case.prefix_tokens;
  result.selection_count = selection_count;
  result.padded_selection_count = selection_count;
  result.active_experts = active_experts;
  result.cold_ms = *cold_ms;
  result.hot_mean_ms = hot_mean_ms;
  result.hot_min_ms = hot_min_ms;
  result.hot_max_ms = hot_max_ms;
  result.hot_tflops =
      hot_mean_ms > 0.0 ? logical_flops / (hot_mean_ms / 1000.0) / 1.0e12 : 0.0;
  result.hot_weight_gib_per_s =
      hot_mean_ms > 0.0 ? weight_bytes / (hot_mean_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0) : 0.0;
  if (output_host != nullptr) {
    output_host->resize(selection_count * kRoutedIntermediateSize);
    if (!output->CopyToHost(output_host->data(), output_host->size())) {
      return std::nullopt;
    }
  }
  return result;
}

std::optional<BenchmarkResult> RunLaunchPlanWmmaBf16TransposedCase(
    const BenchmarkCase& benchmark_case,
    const BenchmarkOptions& options,
    std::vector<float>* output_host = nullptr) {
  const std::size_t selection_count = benchmark_case.prefix_tokens * kTopK;
  const auto expert_offsets = BuildExpertOffsets(selection_count);
  const std::size_t active_experts = CountActiveExperts(expert_offsets);
  const auto selected_indices_host = BuildSelectedIndicesForOffsets(expert_offsets);
  const std::vector<float> selected_weights_host(selection_count, 1.0f);

  auto selected_indices_device = DeviceArray<int>::CopyFromHost(selected_indices_host);
  auto selected_weights_device = DeviceArray<float>::CopyFromHost(selected_weights_host);
  auto routing = DeviceExpertRouting::Create(kRoutedExperts, selection_count);
  auto launch_plan = DeviceMoeLaunchPlan::Create(kRoutedExperts, selection_count);
  if (selected_indices_device == nullptr ||
      selected_weights_device == nullptr ||
      routing == nullptr ||
      !routing->valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      !RunDeviceExpertRouting(
          selected_indices_device->data(),
          selected_weights_device->data(),
          benchmark_case.prefix_tokens,
          kTopK,
          routing.get()) ||
      !BuildDeviceMoeLaunchPlan(*routing, selection_count, launch_plan.get()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(kRoutedExperts, selection_count);
  if (!current_cta_capacity.has_value() || *current_cta_capacity == 0) {
    return std::nullopt;
  }

  const auto routed_up = BuildUploadedRoutedUpViews();
  if (!routed_up.has_value()) {
    return std::nullopt;
  }
  auto routed_up_views_device =
      DeviceArray<FusedNvfp4WeightView>::CopyFromHost(routed_up->views);
  if (routed_up_views_device == nullptr) {
    return std::nullopt;
  }

  const auto input_host =
      MakePatternedValues(benchmark_case.prefix_tokens, kHiddenSize, 23, 0.03125f);
  auto input = DeviceTensorFp32::Create({benchmark_case.prefix_tokens, kHiddenSize});
  auto compact_input = DeviceTensorFp32::Create({selection_count, kHiddenSize});
  auto output = DeviceTensorFp32::Create({selection_count, kRoutedIntermediateSize});
  if (input == nullptr || compact_input == nullptr || output == nullptr) {
    return std::nullopt;
  }
  if (!input->CopyFromHost(input_host.data(), input_host.size())) {
    return std::nullopt;
  }
  if (!LaunchGatherPermutedRows(
          input->data(),
          routing->sorted_token_indices(),
          selection_count,
          benchmark_case.prefix_tokens,
          kHiddenSize,
          compact_input->data()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  const std::size_t output_row_tile_count =
      (kRoutedIntermediateSize + static_cast<std::size_t>(kWmmaOutputTile) - 1u) /
      static_cast<std::size_t>(kWmmaOutputTile);

  ScopedCudaEventTimer timer;
  if (!timer.valid()) {
    return std::nullopt;
  }

  const dim3 block(kWmmaThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  auto run_once = [&]() {
    LaunchPlanWmmaBf16TransposedKernel<<<grid, block>>>(
        compact_input->data(),
        launch_plan->cta_count(),
        launch_plan->cta_expert_ids(),
        launch_plan->cta_row_starts(),
        launch_plan->cta_valid_rows(),
        routed_up_views_device->data(),
        kRoutedIntermediateSize,
        output->data());
    return cudaGetLastError() == cudaSuccess;
  };

  for (std::size_t iteration = 0; iteration < options.warmup_iterations; ++iteration) {
    const auto elapsed = timer.Measure(run_once);
    if (!elapsed.has_value()) {
      return std::nullopt;
    }
  }

  const auto cold_ms = timer.Measure(run_once);
  if (!cold_ms.has_value()) {
    return std::nullopt;
  }

  double hot_sum_ms = 0.0;
  double hot_min_ms = 0.0;
  double hot_max_ms = 0.0;
  for (std::size_t iteration = 0; iteration < options.hot_iterations; ++iteration) {
    const auto hot_elapsed = timer.Measure(run_once);
    if (!hot_elapsed.has_value()) {
      return std::nullopt;
    }
    const double elapsed_ms = *hot_elapsed;
    hot_sum_ms += elapsed_ms;
    if (iteration == 0 || elapsed_ms < hot_min_ms) {
      hot_min_ms = elapsed_ms;
    }
    if (iteration == 0 || elapsed_ms > hot_max_ms) {
      hot_max_ms = elapsed_ms;
    }
  }

  const double hot_mean_ms =
      hot_sum_ms / static_cast<double>(std::max<std::size_t>(1, options.hot_iterations));
  const double logical_flops =
      2.0 * static_cast<double>(selection_count) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize);
  const double weight_bytes =
      static_cast<double>(active_experts) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize) *
      kNvfp4BytesPerWeight;

  BenchmarkResult result;
  result.variant_name = "launch_plan_wmma_bf16_transposed_m16n32k16";
  result.case_name = benchmark_case.name;
  result.prefix_tokens = benchmark_case.prefix_tokens;
  result.selection_count = selection_count;
  result.padded_selection_count = selection_count;
  result.active_experts = active_experts;
  result.cold_ms = *cold_ms;
  result.hot_mean_ms = hot_mean_ms;
  result.hot_min_ms = hot_min_ms;
  result.hot_max_ms = hot_max_ms;
  result.hot_tflops =
      hot_mean_ms > 0.0 ? logical_flops / (hot_mean_ms / 1000.0) / 1.0e12 : 0.0;
  result.hot_weight_gib_per_s =
      hot_mean_ms > 0.0 ? weight_bytes / (hot_mean_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0) : 0.0;
  if (output_host != nullptr) {
    output_host->resize(selection_count * kRoutedIntermediateSize);
    if (!output->CopyToHost(output_host->data(), output_host->size())) {
      return std::nullopt;
    }
  }
  return result;
}

std::optional<BenchmarkResult> RunLaunchPlanWmmaPackedInputTransposedCase(
    const BenchmarkCase& benchmark_case,
    const BenchmarkOptions& options,
    std::vector<float>* output_host = nullptr) {
  const std::size_t selection_count = benchmark_case.prefix_tokens * kTopK;
  const auto expert_offsets = BuildExpertOffsets(selection_count);
  const std::size_t active_experts = CountActiveExperts(expert_offsets);
  const auto selected_indices_host = BuildSelectedIndicesForOffsets(expert_offsets);
  const std::vector<float> selected_weights_host(selection_count, 1.0f);

  auto selected_indices_device = DeviceArray<int>::CopyFromHost(selected_indices_host);
  auto selected_weights_device = DeviceArray<float>::CopyFromHost(selected_weights_host);
  auto routing = DeviceExpertRouting::Create(kRoutedExperts, selection_count);
  auto launch_plan = DeviceMoeLaunchPlan::Create(kRoutedExperts, selection_count);
  if (selected_indices_device == nullptr ||
      selected_weights_device == nullptr ||
      routing == nullptr ||
      !routing->valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      !RunDeviceExpertRouting(
          selected_indices_device->data(),
          selected_weights_device->data(),
          benchmark_case.prefix_tokens,
          kTopK,
          routing.get()) ||
      !BuildDeviceMoeLaunchPlan(*routing, selection_count, launch_plan.get()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(kRoutedExperts, selection_count);
  if (!current_cta_capacity.has_value() || *current_cta_capacity == 0) {
    return std::nullopt;
  }

  const auto routed_up = BuildUploadedRoutedUpViews();
  if (!routed_up.has_value()) {
    return std::nullopt;
  }
  auto routed_up_views_device =
      DeviceArray<FusedNvfp4WeightView>::CopyFromHost(routed_up->views);
  if (routed_up_views_device == nullptr) {
    return std::nullopt;
  }

  const auto input_host =
      MakePatternedValues(benchmark_case.prefix_tokens, kHiddenSize, 23, 0.03125f);
  auto input = DeviceTensorFp32::Create({benchmark_case.prefix_tokens, kHiddenSize});
  auto compact_input = DeviceTensorFp32::Create({selection_count, kHiddenSize});
  auto compact_input_pack =
      DeviceNvfp4Matrix::Create(selection_count, kHiddenSize, Nvfp4ScaleLayout::kSwizzled128x4);
  auto output = DeviceTensorFp32::Create({selection_count, kRoutedIntermediateSize});
  if (input == nullptr || compact_input == nullptr ||
      compact_input_pack == nullptr || output == nullptr) {
    return std::nullopt;
  }
  if (!input->CopyFromHost(input_host.data(), input_host.size())) {
    return std::nullopt;
  }
  if (!LaunchGatherPermutedRows(
          input->data(),
          routing->sorted_token_indices(),
          selection_count,
          benchmark_case.prefix_tokens,
          kHiddenSize,
          compact_input->data()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  Nvfp4PackOptions pack_options;
  pack_options.execution_scale_layout = Nvfp4ScaleLayout::kSwizzled128x4;
  if (!compact_input_pack->PackInto(*compact_input, pack_options) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  const std::size_t output_row_tile_count =
      (kRoutedIntermediateSize + static_cast<std::size_t>(kWmmaOutputTile) - 1u) /
      static_cast<std::size_t>(kWmmaOutputTile);

  ScopedCudaEventTimer timer;
  if (!timer.valid()) {
    return std::nullopt;
  }

  const dim3 block(kWmmaThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  auto run_once = [&]() {
    LaunchPlanWmmaPackedInputTransposedKernel<<<grid, block>>>(
        compact_input_pack->packed_data(),
        compact_input_pack->block_scales_data(),
        compact_input_pack->device_tensor_scale_ptr(),
        launch_plan->cta_count(),
        launch_plan->cta_expert_ids(),
        launch_plan->cta_row_starts(),
        launch_plan->cta_valid_rows(),
        routed_up_views_device->data(),
        kRoutedIntermediateSize,
        output->data());
    return cudaGetLastError() == cudaSuccess;
  };

  for (std::size_t iteration = 0; iteration < options.warmup_iterations; ++iteration) {
    const auto elapsed = timer.Measure(run_once);
    if (!elapsed.has_value()) {
      return std::nullopt;
    }
  }

  const auto cold_ms = timer.Measure(run_once);
  if (!cold_ms.has_value()) {
    return std::nullopt;
  }

  double hot_sum_ms = 0.0;
  double hot_min_ms = 0.0;
  double hot_max_ms = 0.0;
  for (std::size_t iteration = 0; iteration < options.hot_iterations; ++iteration) {
    const auto hot_elapsed = timer.Measure(run_once);
    if (!hot_elapsed.has_value()) {
      return std::nullopt;
    }
    const double elapsed_ms = *hot_elapsed;
    hot_sum_ms += elapsed_ms;
    if (iteration == 0 || elapsed_ms < hot_min_ms) {
      hot_min_ms = elapsed_ms;
    }
    if (iteration == 0 || elapsed_ms > hot_max_ms) {
      hot_max_ms = elapsed_ms;
    }
  }

  const double hot_mean_ms =
      hot_sum_ms / static_cast<double>(std::max<std::size_t>(1, options.hot_iterations));
  const double logical_flops =
      2.0 * static_cast<double>(selection_count) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize);
  const double weight_bytes =
      static_cast<double>(active_experts) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize) *
      kNvfp4BytesPerWeight;

  BenchmarkResult result;
  result.variant_name = "launch_plan_wmma_packed_input_transposed_m16n32k16";
  result.case_name = benchmark_case.name;
  result.prefix_tokens = benchmark_case.prefix_tokens;
  result.selection_count = selection_count;
  result.padded_selection_count = selection_count;
  result.active_experts = active_experts;
  result.cold_ms = *cold_ms;
  result.hot_mean_ms = hot_mean_ms;
  result.hot_min_ms = hot_min_ms;
  result.hot_max_ms = hot_max_ms;
  result.hot_tflops =
      hot_mean_ms > 0.0 ? logical_flops / (hot_mean_ms / 1000.0) / 1.0e12 : 0.0;
  result.hot_weight_gib_per_s =
      hot_mean_ms > 0.0 ? weight_bytes / (hot_mean_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0) : 0.0;
  if (output_host != nullptr) {
    output_host->resize(selection_count * kRoutedIntermediateSize);
    if (!output->CopyToHost(output_host->data(), output_host->size())) {
      return std::nullopt;
    }
  }
  return result;
}

std::optional<BenchmarkResult> RunLaunchPlanWmmaPerExpertPackedInputTransposedCase(
    const BenchmarkCase& benchmark_case,
    const BenchmarkOptions& options,
    std::vector<float>* output_host = nullptr) {
  const std::size_t selection_count = benchmark_case.prefix_tokens * kTopK;
  const auto expert_offsets = BuildExpertOffsets(selection_count);
  const std::size_t active_experts = CountActiveExperts(expert_offsets);
  const auto selected_indices_host = BuildSelectedIndicesForOffsets(expert_offsets);
  const std::vector<float> selected_weights_host(selection_count, 1.0f);

  auto selected_indices_device = DeviceArray<int>::CopyFromHost(selected_indices_host);
  auto selected_weights_device = DeviceArray<float>::CopyFromHost(selected_weights_host);
  auto expert_offsets_device = DeviceArray<int>::CopyFromHost(expert_offsets);
  auto routing = DeviceExpertRouting::Create(kRoutedExperts, selection_count);
  auto launch_plan = DeviceMoeLaunchPlan::Create(kRoutedExperts, selection_count);
  if (selected_indices_device == nullptr ||
      selected_weights_device == nullptr ||
      expert_offsets_device == nullptr ||
      routing == nullptr ||
      !routing->valid() ||
      launch_plan == nullptr ||
      !launch_plan->valid() ||
      !RunDeviceExpertRouting(
          selected_indices_device->data(),
          selected_weights_device->data(),
          benchmark_case.prefix_tokens,
          kTopK,
          routing.get()) ||
      !BuildDeviceMoeLaunchPlan(*routing, selection_count, launch_plan.get()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  const auto current_cta_capacity =
      DeviceMoeLaunchPlan::CtaCapacity(kRoutedExperts, selection_count);
  if (!current_cta_capacity.has_value() || *current_cta_capacity == 0) {
    return std::nullopt;
  }

  const auto routed_up = BuildUploadedRoutedUpViews();
  if (!routed_up.has_value()) {
    return std::nullopt;
  }
  auto routed_up_views_device =
      DeviceArray<FusedNvfp4WeightView>::CopyFromHost(routed_up->views);
  if (routed_up_views_device == nullptr) {
    return std::nullopt;
  }

  const auto input_host =
      MakePatternedValues(benchmark_case.prefix_tokens, kHiddenSize, 23, 0.03125f);
  auto input = DeviceTensorFp32::Create({benchmark_case.prefix_tokens, kHiddenSize});
  auto compact_input = DeviceTensorFp32::Create({selection_count, kHiddenSize});
  auto output = DeviceTensorFp32::Create({selection_count, kRoutedIntermediateSize});
  if (input == nullptr || compact_input == nullptr || output == nullptr) {
    return std::nullopt;
  }
  if (!input->CopyFromHost(input_host.data(), input_host.size())) {
    return std::nullopt;
  }
  if (!LaunchGatherPermutedRows(
          input->data(),
          routing->sorted_token_indices(),
          selection_count,
          benchmark_case.prefix_tokens,
          kHiddenSize,
          compact_input->data()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  Nvfp4PackOptions pack_options;
  pack_options.execution_scale_layout = Nvfp4ScaleLayout::kSwizzled128x4;
  std::vector<std::unique_ptr<DeviceNvfp4Matrix>> packed_storage(kRoutedExperts);
  std::vector<PackedInputView> packed_views(kRoutedExperts);
  for (std::size_t expert_index = 0; expert_index < kRoutedExperts; ++expert_index) {
    const int begin = expert_offsets[expert_index];
    const int end = expert_offsets[expert_index + 1];
    const std::size_t rows =
        end > begin ? static_cast<std::size_t>(end - begin) : 0u;
    if (rows == 0) {
      continue;
    }
    auto input_view = DeviceTensorFp32::CreateView(
        {rows, kHiddenSize},
        compact_input->data() + static_cast<std::size_t>(begin) * kHiddenSize);
    auto packed = DeviceNvfp4Matrix::Create(rows, kHiddenSize, Nvfp4ScaleLayout::kSwizzled128x4);
    if (input_view == nullptr ||
        packed == nullptr ||
        !packed->PackInto(*input_view, pack_options)) {
      return std::nullopt;
    }
    packed_views[expert_index] = PackedInputView{
        packed->packed_data(),
        packed->block_scales_data(),
        packed->device_tensor_scale_ptr(),
        rows,
        kHiddenSize,
    };
    packed_storage[expert_index] = std::move(packed);
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }
  auto packed_views_device = DeviceArray<PackedInputView>::CopyFromHost(packed_views);
  if (packed_views_device == nullptr) {
    return std::nullopt;
  }

  const std::size_t output_row_tile_count =
      (kRoutedIntermediateSize + static_cast<std::size_t>(kWmmaOutputTile) - 1u) /
      static_cast<std::size_t>(kWmmaOutputTile);

  ScopedCudaEventTimer timer;
  if (!timer.valid()) {
    return std::nullopt;
  }

  const dim3 block(kWmmaThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>(output_row_tile_count),
      static_cast<unsigned int>(*current_cta_capacity));
  auto run_once = [&]() {
    LaunchPlanWmmaPackedPerExpertInputTransposedKernel<<<grid, block>>>(
        packed_views_device->data(),
        expert_offsets_device->data(),
        launch_plan->cta_count(),
        launch_plan->cta_expert_ids(),
        launch_plan->cta_row_starts(),
        launch_plan->cta_valid_rows(),
        routed_up_views_device->data(),
        kRoutedIntermediateSize,
        output->data());
    return cudaGetLastError() == cudaSuccess;
  };

  for (std::size_t iteration = 0; iteration < options.warmup_iterations; ++iteration) {
    const auto elapsed = timer.Measure(run_once);
    if (!elapsed.has_value()) {
      return std::nullopt;
    }
  }

  const auto cold_ms = timer.Measure(run_once);
  if (!cold_ms.has_value()) {
    return std::nullopt;
  }

  double hot_sum_ms = 0.0;
  double hot_min_ms = 0.0;
  double hot_max_ms = 0.0;
  for (std::size_t iteration = 0; iteration < options.hot_iterations; ++iteration) {
    const auto hot_elapsed = timer.Measure(run_once);
    if (!hot_elapsed.has_value()) {
      return std::nullopt;
    }
    const double elapsed_ms = *hot_elapsed;
    hot_sum_ms += elapsed_ms;
    if (iteration == 0 || elapsed_ms < hot_min_ms) {
      hot_min_ms = elapsed_ms;
    }
    if (iteration == 0 || elapsed_ms > hot_max_ms) {
      hot_max_ms = elapsed_ms;
    }
  }

  const double hot_mean_ms =
      hot_sum_ms / static_cast<double>(std::max<std::size_t>(1, options.hot_iterations));
  const double logical_flops =
      2.0 * static_cast<double>(selection_count) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize);
  const double weight_bytes =
      static_cast<double>(active_experts) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize) *
      kNvfp4BytesPerWeight;

  BenchmarkResult result;
  result.variant_name = "launch_plan_wmma_per_expert_packed_input_transposed_m16n32k16";
  result.case_name = benchmark_case.name;
  result.prefix_tokens = benchmark_case.prefix_tokens;
  result.selection_count = selection_count;
  result.padded_selection_count = selection_count;
  result.active_experts = active_experts;
  result.cold_ms = *cold_ms;
  result.hot_mean_ms = hot_mean_ms;
  result.hot_min_ms = hot_min_ms;
  result.hot_max_ms = hot_max_ms;
  result.hot_tflops =
      hot_mean_ms > 0.0 ? logical_flops / (hot_mean_ms / 1000.0) / 1.0e12 : 0.0;
  result.hot_weight_gib_per_s =
      hot_mean_ms > 0.0 ? weight_bytes / (hot_mean_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0) : 0.0;
  if (output_host != nullptr) {
    output_host->resize(selection_count * kRoutedIntermediateSize);
    if (!output->CopyToHost(output_host->data(), output_host->size())) {
      return std::nullopt;
    }
  }
  return result;
}

std::optional<BenchmarkResult> RunReferenceNvfp4CublasLtExpertLoopCase(
    const BenchmarkCase& benchmark_case,
    const BenchmarkOptions& options,
    std::vector<float>* output_host = nullptr) {
  const std::size_t selection_count = benchmark_case.prefix_tokens * kTopK;
  const auto expert_offsets = BuildExpertOffsets(selection_count);
  const std::size_t active_experts = CountActiveExperts(expert_offsets);
  const auto selected_indices_host = BuildSelectedIndicesForOffsets(expert_offsets);
  const std::vector<float> selected_weights_host(selection_count, 1.0f);

  auto selected_indices_device = DeviceArray<int>::CopyFromHost(selected_indices_host);
  auto selected_weights_device = DeviceArray<float>::CopyFromHost(selected_weights_host);
  auto routing = DeviceExpertRouting::Create(kRoutedExperts, selection_count);
  if (selected_indices_device == nullptr ||
      selected_weights_device == nullptr ||
      routing == nullptr ||
      !routing->valid() ||
      !RunDeviceExpertRouting(
          selected_indices_device->data(),
          selected_weights_device->data(),
          benchmark_case.prefix_tokens,
          kTopK,
          routing.get()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  const auto routed_up = BuildUploadedRoutedUpViews();
  if (!routed_up.has_value() || !routed_up->host_packed.has_value()) {
    return std::nullopt;
  }

  const GemmDescriptor weight_descriptor =
      MakeNvfp4Descriptor("nano_routed_up.reference", "routed_up", *routed_up->host_packed);
  auto device_weight = DeviceNvfp4Weight::Upload(weight_descriptor);
  auto cublas_handle = CublasLtHandle::Create(64u * 1024u * 1024u);
  if (device_weight == nullptr || !device_weight->valid() ||
      cublas_handle == nullptr || !cublas_handle->valid()) {
    return std::nullopt;
  }

  const auto input_host =
      MakePatternedValues(benchmark_case.prefix_tokens, kHiddenSize, 23, 0.03125f);
  auto input = DeviceTensorFp32::Create({benchmark_case.prefix_tokens, kHiddenSize});
  auto compact_input = DeviceTensorFp32::Create({selection_count, kHiddenSize});
  auto output = DeviceTensorFp32::Create({selection_count, kRoutedIntermediateSize});
  if (input == nullptr || compact_input == nullptr || output == nullptr) {
    return std::nullopt;
  }
  if (!input->CopyFromHost(input_host.data(), input_host.size())) {
    return std::nullopt;
  }
  if (!LaunchGatherPermutedRows(
          input->data(),
          routing->sorted_token_indices(),
          selection_count,
          benchmark_case.prefix_tokens,
          kHiddenSize,
          compact_input->data()) ||
      cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  struct ExpertGemmWork {
    std::size_t rows = 0;
    std::unique_ptr<DeviceNvfp4Matrix> input_pack;
    std::unique_ptr<DeviceTensorFp32> output_view;
    const nemotron::CublasLtGemmPlan* plan = nullptr;
  };

  Nvfp4PackOptions pack_options;
  pack_options.execution_scale_layout = Nvfp4ScaleLayout::kSwizzled128x4;
  GemmHeuristicCache heuristic_cache;
  std::vector<std::size_t> cached_plan_rows;
  std::vector<std::unique_ptr<nemotron::CublasLtGemmPlan>> cached_plans;
  std::vector<ExpertGemmWork> work_items;
  work_items.reserve(active_experts);

  for (std::size_t expert_index = 0; expert_index < kRoutedExperts; ++expert_index) {
    const int begin = expert_offsets[expert_index];
    const int end = expert_offsets[expert_index + 1];
    const std::size_t rows =
        end > begin ? static_cast<std::size_t>(end - begin) : 0u;
    if (rows == 0) {
      continue;
    }

    auto input_view = DeviceTensorFp32::CreateView(
        {rows, kHiddenSize},
        compact_input->data() + static_cast<std::size_t>(begin) * kHiddenSize);
    auto output_view = DeviceTensorFp32::CreateView(
        {rows, kRoutedIntermediateSize},
        output->data() + static_cast<std::size_t>(begin) * kRoutedIntermediateSize);
    auto input_pack =
        DeviceNvfp4Matrix::Create(rows, kHiddenSize, Nvfp4ScaleLayout::kSwizzled128x4);
    if (input_view == nullptr || output_view == nullptr ||
        input_pack == nullptr || !input_pack->PackInto(*input_view, pack_options)) {
      return std::nullopt;
    }

    const nemotron::CublasLtGemmPlan* cached_plan = nullptr;
    for (std::size_t i = 0; i < cached_plan_rows.size(); ++i) {
      if (cached_plan_rows[i] == rows) {
        cached_plan = cached_plans[i].get();
        break;
      }
    }
    if (cached_plan == nullptr) {
      auto runtime_launch_plan = BuildGemmLaunchPlan(weight_descriptor, rows);
      if (!runtime_launch_plan.has_value()) {
        return std::nullopt;
      }
      const auto execution = PrepareGemmExecution(*runtime_launch_plan, &heuristic_cache);
      if (!execution.has_value()) {
        return std::nullopt;
      }
      const auto cublas_plan = BuildCublasLtGemmPlan(*execution);
      if (!cublas_plan.has_value()) {
        return std::nullopt;
      }
      cached_plan_rows.push_back(rows);
      cached_plans.push_back(
          std::make_unique<nemotron::CublasLtGemmPlan>(std::move(*cublas_plan)));
      cached_plan = cached_plans.back().get();
    }

    work_items.push_back(ExpertGemmWork{
        rows,
        std::move(input_pack),
        std::move(output_view),
        cached_plan,
    });
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    return std::nullopt;
  }

  ScopedCudaEventTimer timer;
  if (!timer.valid()) {
    return std::nullopt;
  }

  auto run_once = [&]() {
    for (const auto& work_item : work_items) {
      if (work_item.plan == nullptr || work_item.input_pack == nullptr ||
          work_item.output_view == nullptr) {
        return false;
      }
      const auto stats = RunNvfp4RowMajorFp32AccumToDevice(
          *cublas_handle,
          *work_item.plan,
          MakeNvfp4PackedMatrixDeviceView(*work_item.input_pack),
          *device_weight,
          work_item.output_view.get());
      if (!stats.has_value()) {
        return false;
      }
    }
    return true;
  };

  for (std::size_t iteration = 0; iteration < options.warmup_iterations; ++iteration) {
    const auto elapsed = timer.Measure(run_once);
    if (!elapsed.has_value()) {
      return std::nullopt;
    }
  }

  const auto cold_ms = timer.Measure(run_once);
  if (!cold_ms.has_value()) {
    return std::nullopt;
  }

  double hot_sum_ms = 0.0;
  double hot_min_ms = 0.0;
  double hot_max_ms = 0.0;
  for (std::size_t iteration = 0; iteration < options.hot_iterations; ++iteration) {
    const auto hot_elapsed = timer.Measure(run_once);
    if (!hot_elapsed.has_value()) {
      return std::nullopt;
    }
    const double elapsed_ms = *hot_elapsed;
    hot_sum_ms += elapsed_ms;
    if (iteration == 0 || elapsed_ms < hot_min_ms) {
      hot_min_ms = elapsed_ms;
    }
    if (iteration == 0 || elapsed_ms > hot_max_ms) {
      hot_max_ms = elapsed_ms;
    }
  }

  const double hot_mean_ms =
      hot_sum_ms / static_cast<double>(std::max<std::size_t>(1, options.hot_iterations));
  const double logical_flops =
      2.0 * static_cast<double>(selection_count) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize);
  const double weight_bytes =
      static_cast<double>(active_experts) *
      static_cast<double>(kHiddenSize) *
      static_cast<double>(kRoutedIntermediateSize) *
      kNvfp4BytesPerWeight;

  BenchmarkResult result;
  result.variant_name = "reference_cublaslt_nvfp4_expert_loop";
  result.case_name = benchmark_case.name;
  result.prefix_tokens = benchmark_case.prefix_tokens;
  result.selection_count = selection_count;
  result.padded_selection_count = selection_count;
  result.active_experts = active_experts;
  result.cold_ms = *cold_ms;
  result.hot_mean_ms = hot_mean_ms;
  result.hot_min_ms = hot_min_ms;
  result.hot_max_ms = hot_max_ms;
  result.hot_tflops =
      hot_mean_ms > 0.0 ? logical_flops / (hot_mean_ms / 1000.0) / 1.0e12 : 0.0;
  result.hot_weight_gib_per_s =
      hot_mean_ms > 0.0 ? weight_bytes / (hot_mean_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0) : 0.0;
  if (output_host != nullptr) {
    output_host->resize(selection_count * kRoutedIntermediateSize);
    if (!output->CopyToHost(output_host->data(), output_host->size())) {
      return std::nullopt;
    }
  }
  return result;
}

BenchmarkOptions ParseOptions(int argc, char** argv) {
  BenchmarkOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--warmup" && i + 1 < argc) {
      options.warmup_iterations = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
    } else if (arg == "--iterations" && i + 1 < argc) {
      options.hot_iterations = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
    } else if (arg == "--case" && i + 1 < argc) {
      options.selected_case_names.emplace_back(argv[++i]);
    }
  }
  return options;
}

void PrintResult(const BenchmarkResult& result) {
  std::cout << "Case: " << result.case_name
            << " variant=" << result.variant_name
            << " prefix_tokens=" << result.prefix_tokens
            << " selection_count=" << result.selection_count
            << " padded_selection_count=" << result.padded_selection_count
            << " active_experts=" << result.active_experts << "\n";
  std::cout << "  cold_ms=" << std::fixed << std::setprecision(3) << result.cold_ms
            << " hot_mean_ms=" << result.hot_mean_ms
            << " hot_min_ms=" << result.hot_min_ms
            << " hot_max_ms=" << result.hot_max_ms << "\n";
  std::cout << "  hot_tflops=" << std::setprecision(3) << result.hot_tflops
            << " hot_weight_gib_per_s=" << result.hot_weight_gib_per_s
            << " max_abs_diff_vs_baseline=" << result.max_abs_diff_vs_baseline;
  if (result.max_abs_diff_vs_baseline > 0.0) {
    std::cout << " diff_index=" << result.max_abs_diff_index
              << " baseline_value=" << result.max_abs_diff_baseline_value
              << " variant_value=" << result.max_abs_diff_variant_value;
  }
  std::cout << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  const BenchmarkOptions options = ParseOptions(argc, argv);
  if (!HasCudaDevice()) {
    std::cout << "nano_routed_up_bench: SKIP (no CUDA device)\n";
    return 0;
  }

  std::cout << "nano_routed_up_bench: hidden=" << kHiddenSize
            << " routed_intermediate=" << kRoutedIntermediateSize
            << " routed_experts=" << kRoutedExperts
            << " top_k=" << kTopK
            << " warmup_iterations=" << options.warmup_iterations
            << " hot_iterations=" << options.hot_iterations << "\n";

  bool ran_any = false;
  for (const auto& benchmark_case : DefaultCases()) {
    if (!ShouldRunCase(options, benchmark_case.name)) {
      continue;
    }
    ran_any = true;
    std::vector<float> baseline_output_host;
    const auto result = RunCase(benchmark_case, options, &baseline_output_host);
    if (!result.has_value()) {
      std::cerr << "nano_routed_up_bench: case failed: " << benchmark_case.name << "\n";
      return 1;
    }
    PrintResult(*result);

    std::vector<float> permuted_output_host;
    auto permuted_result = RunRaggedRowCoopCase(benchmark_case, options, &permuted_output_host);
    if (!permuted_result.has_value()) {
      std::cerr << "nano_routed_up_bench: ragged row-coop case failed: "
                << benchmark_case.name << "\n";
      return 1;
    }
    PopulateDiffSummary(baseline_output_host, permuted_output_host, &*permuted_result);
    PrintResult(*permuted_result);

    std::vector<float> wide_permuted_output_host;
    auto wide_permuted_result =
        RunRaggedWideRowCoopCase(benchmark_case, options, &wide_permuted_output_host);
    if (!wide_permuted_result.has_value()) {
      std::cerr << "nano_routed_up_bench: ragged wide row-coop case failed: "
                << benchmark_case.name << "\n";
      return 1;
    }
    PopulateDiffSummary(baseline_output_host, wide_permuted_output_host, &*wide_permuted_result);
    PrintResult(*wide_permuted_result);

    std::vector<float> launch_plan_output_host;
    auto launch_plan_result =
        RunLaunchPlanUpperBoundCase(benchmark_case, options, &launch_plan_output_host);
    if (!launch_plan_result.has_value()) {
      std::cerr << "nano_routed_up_bench: launch-plan upper-bound case failed: "
                << benchmark_case.name << "\n";
      return 1;
    }
    PopulateDiffSummary(baseline_output_host, launch_plan_output_host, &*launch_plan_result);
    PrintResult(*launch_plan_result);

    std::vector<float> fp32_accum_output_host;
    auto fp32_accum_result =
        RunLaunchPlanFp32AccumCase(benchmark_case, options, &fp32_accum_output_host);
    if (!fp32_accum_result.has_value()) {
      std::cerr << "nano_routed_up_bench: launch-plan fp32-accum case failed: "
                << benchmark_case.name << "\n";
      return 1;
    }
    PopulateDiffSummary(baseline_output_host, fp32_accum_output_host, &*fp32_accum_result);
    PrintResult(*fp32_accum_result);

    std::vector<float> wide_output_host;
    auto wide_result =
        RunLaunchPlanWideFp32AccumCase(benchmark_case, options, &wide_output_host);
    if (!wide_result.has_value()) {
      std::cerr << "nano_routed_up_bench: launch-plan wide fp32 case failed: "
                << benchmark_case.name << "\n";
      return 1;
    }
    PopulateDiffSummary(baseline_output_host, wide_output_host, &*wide_result);
    PrintResult(*wide_result);

    std::vector<float> exact_task_output_host;
    auto exact_task_result =
        RunLaunchPlanExactTaskTile8Case(benchmark_case, options, &exact_task_output_host);
    if (!exact_task_result.has_value()) {
      std::cerr << "nano_routed_up_bench: launch-plan exact-task tile8 case failed: "
                << benchmark_case.name << "\n";
      return 1;
    }
    PopulateDiffSummary(baseline_output_host, exact_task_output_host, &*exact_task_result);
    PrintResult(*exact_task_result);

    std::vector<float> grouped_tile_output_host;
    auto grouped_tile_result =
        RunLaunchPlanGroupedTile32Case(benchmark_case, options, &grouped_tile_output_host);
    if (!grouped_tile_result.has_value()) {
      std::cerr << "nano_routed_up_bench: launch-plan grouped tile32 case failed: "
                << benchmark_case.name << "\n";
      return 1;
    }
    PopulateDiffSummary(baseline_output_host, grouped_tile_output_host, &*grouped_tile_result);
    PrintResult(*grouped_tile_result);

    std::vector<float> cooperative_output_host;
    auto cooperative_result =
        RunLaunchPlanCooperativeMxNCase(benchmark_case, options, &cooperative_output_host);
    if (!cooperative_result.has_value()) {
      std::cerr << "nano_routed_up_bench: launch-plan cooperative m8n32 case failed: "
                << benchmark_case.name << "\n";
      return 1;
    }
    PopulateDiffSummary(baseline_output_host, cooperative_output_host, &*cooperative_result);
    PrintResult(*cooperative_result);

    std::vector<float> microtile_output_host;
    auto microtile_result =
        RunLaunchPlanMicroTile2Case(benchmark_case, options, &microtile_output_host);
    if (!microtile_result.has_value()) {
      std::cerr << "nano_routed_up_bench: launch-plan microtile2 m8n32 case failed: "
                << benchmark_case.name << "\n";
      return 1;
    }
    PopulateDiffSummary(baseline_output_host, microtile_output_host, &*microtile_result);
    PrintResult(*microtile_result);

    std::vector<float> wmma_output_host;
    auto wmma_result =
        RunLaunchPlanWmmaBf16Case(benchmark_case, options, &wmma_output_host);
    if (!wmma_result.has_value()) {
      std::cerr << "nano_routed_up_bench: launch-plan wmma bf16 case failed: "
                << benchmark_case.name << "\n";
      return 1;
    }
    PopulateDiffSummary(baseline_output_host, wmma_output_host, &*wmma_result);
    PrintResult(*wmma_result);

    std::vector<float> wmma_transposed_output_host;
    auto wmma_transposed_result =
        RunLaunchPlanWmmaBf16TransposedCase(benchmark_case, options, &wmma_transposed_output_host);
    if (!wmma_transposed_result.has_value()) {
      std::cerr << "nano_routed_up_bench: launch-plan wmma bf16 transposed case failed: "
                << benchmark_case.name << "\n";
      return 1;
    }
    PopulateDiffSummary(
        baseline_output_host, wmma_transposed_output_host, &*wmma_transposed_result);
    PrintResult(*wmma_transposed_result);

    std::vector<float> wmma_packed_input_output_host;
    auto wmma_packed_input_result = RunLaunchPlanWmmaPackedInputTransposedCase(
        benchmark_case, options, &wmma_packed_input_output_host);
    if (!wmma_packed_input_result.has_value()) {
      std::cerr << "nano_routed_up_bench: launch-plan wmma packed-input transposed case failed: "
                << benchmark_case.name << "\n";
      return 1;
    }
    PopulateDiffSummary(
        baseline_output_host,
        wmma_packed_input_output_host,
        &*wmma_packed_input_result);
    PrintResult(*wmma_packed_input_result);

    std::vector<float> wmma_per_expert_packed_input_output_host;
    auto wmma_per_expert_packed_input_result =
        RunLaunchPlanWmmaPerExpertPackedInputTransposedCase(
            benchmark_case, options, &wmma_per_expert_packed_input_output_host);
    if (!wmma_per_expert_packed_input_result.has_value()) {
      std::cerr << "nano_routed_up_bench: launch-plan wmma per-expert packed-input transposed case failed: "
                << benchmark_case.name << "\n";
      return 1;
    }
    PopulateDiffSummary(
        baseline_output_host,
        wmma_per_expert_packed_input_output_host,
        &*wmma_per_expert_packed_input_result);
    PrintResult(*wmma_per_expert_packed_input_result);

    std::vector<float> reference_cublaslt_output_host;
    auto reference_cublaslt_result = RunReferenceNvfp4CublasLtExpertLoopCase(
        benchmark_case, options, &reference_cublaslt_output_host);
    if (!reference_cublaslt_result.has_value()) {
      std::cerr << "nano_routed_up_bench: reference cublaslt nvfp4 expert-loop case failed: "
                << benchmark_case.name << "\n";
      return 1;
    }
    PopulateDiffSummary(
        baseline_output_host,
        reference_cublaslt_output_host,
        &*reference_cublaslt_result);
    PrintResult(*reference_cublaslt_result);

    std::vector<float> bf16_scalar_output_host;
    auto bf16_scalar_result =
        RunLaunchPlanBf16ScalarCase(benchmark_case, options, &bf16_scalar_output_host);
    if (!bf16_scalar_result.has_value()) {
      std::cerr << "nano_routed_up_bench: launch-plan bf16 scalar case failed: "
                << benchmark_case.name << "\n";
      return 1;
    }
    PopulateDiffSummary(baseline_output_host, bf16_scalar_output_host, &*bf16_scalar_result);
    PrintResult(*bf16_scalar_result);

  }

  if (!ran_any) {
    std::cerr << "nano_routed_up_bench: no cases selected\n";
    return 1;
  }

  return 0;
}
