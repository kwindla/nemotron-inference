#include "nemotron/expert_ops.h"

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>

#include "nemotron/nvfp4_scale_layout.h"

namespace nemotron {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr float kNegInf = -1.0e30f;
constexpr std::size_t kNvfp4BlockWidth = 16;
constexpr std::size_t kScaleRowTile = 128;
constexpr std::size_t kScaleBlockTile = 4;
constexpr float kFp4MaxFinite = 6.0f;
constexpr float kFp8E4M3MaxFinite = 448.0f;
constexpr float kMinScale = 1.0f / 1024.0f;

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

__device__ float Sigmoid(float value) {
  if (value >= 0.0f) {
    const float exp_neg = expf(-value);
    return 1.0f / (1.0f + exp_neg);
  }
  const float exp_pos = expf(value);
  return exp_pos / (1.0f + exp_pos);
}

__device__ float ClampScale(float value) {
  if (!isfinite(value) || value < kMinScale) {
    return kMinScale;
  }
  return value;
}

__device__ float DecodeFp4E2M1(std::uint8_t code) {
  const __half raw = static_cast<__half>(__nv_cvt_fp4_to_halfraw(
      static_cast<__nv_fp4_storage_t>(code & 0x0fu),
      __NV_E2M1));
  return __half2float(raw);
}

__device__ float DecodeFp8E4M3(std::uint8_t code) {
  const __half raw = static_cast<__half>(__nv_cvt_fp8_to_halfraw(
      static_cast<__nv_fp8_storage_t>(code),
      __NV_E4M3));
  return __half2float(raw);
}

__global__ void CopyRowKernel(
    const float* input,
    std::size_t input_cols,
    std::size_t row_index,
    float* output_row) {
  const std::size_t column = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= input_cols) {
    return;
  }
  output_row[column] = input[row_index * input_cols + column];
}

__global__ void WriteRowKernel(
    const float* input_row,
    std::size_t output_cols,
    std::size_t row_index,
    float* output) {
  const std::size_t column = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= output_cols) {
    return;
  }
  output[row_index * output_cols + column] = input_row[column];
}

__global__ void Relu2InPlaceKernel(float* data, std::size_t count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const float value = data[index];
  data[index] = value > 0.0f ? (value * value) : 0.0f;
}

__global__ void AddScaledKernel(
    const float* input,
    float scale,
    float* accumulator,
    std::size_t count) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  accumulator[index] += input[index] * scale;
}

__global__ void AddScaledRowKernel(
    const float* input_row,
    float scale,
    std::size_t row_index,
    std::size_t accumulator_cols,
    float* accumulator) {
  const std::size_t column = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column >= accumulator_cols) {
    return;
  }
  accumulator[row_index * accumulator_cols + column] += input_row[column] * scale;
}

__global__ void SelectTopExpertsKernel(
    const float* router_logits,
    const float* correction_bias,
    std::size_t rows,
    std::size_t expert_count,
    std::size_t n_group,
    std::size_t topk_group,
    std::size_t top_k,
    bool norm_topk_prob,
    float routed_scaling_factor,
    std::int32_t* selected_indices,
    float* selected_weights) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.x);
  if (row >= rows || threadIdx.x != 0 || expert_count == 0 || n_group == 0 || top_k == 0 ||
      expert_count % n_group != 0) {
    return;
  }

  constexpr std::size_t kMaxExperts = 1024;
  constexpr std::size_t kMaxGroups = 128;
  if (expert_count > kMaxExperts || n_group > kMaxGroups) {
    for (std::size_t i = 0; i < top_k; ++i) {
      selected_indices[row * top_k + i] = -1;
      selected_weights[row * top_k + i] = 0.0f;
    }
    return;
  }

  const float* row_logits = router_logits + (row * expert_count);
  const std::size_t group_size = expert_count / n_group;

  float scores[kMaxExperts];
  float scores_for_choice[kMaxExperts];
  for (std::size_t expert = 0; expert < expert_count; ++expert) {
    scores[expert] = Sigmoid(row_logits[expert]);
    scores_for_choice[expert] = scores[expert] + correction_bias[expert];
  }

  bool group_selected[kMaxGroups];
  for (std::size_t group = 0; group < n_group; ++group) {
    group_selected[group] = false;
  }

  float top_group_score[kMaxGroups];
  for (std::size_t group = 0; group < n_group; ++group) {
    float top1 = kNegInf;
    float top2 = kNegInf;
    const std::size_t group_offset = group * group_size;
    for (std::size_t index = 0; index < group_size; ++index) {
      const float value = scores_for_choice[group_offset + index];
      if (value > top1) {
        top2 = top1;
        top1 = value;
      } else if (value > top2) {
        top2 = value;
      }
    }
    if (!isfinite(top2)) {
      top2 = top1;
    }
    top_group_score[group] = top1 + top2;
  }

  const std::size_t group_pick_count = topk_group < n_group ? topk_group : n_group;
  for (std::size_t pick = 0; pick < group_pick_count; ++pick) {
      float best_value = kNegInf;
    std::size_t best_group = 0;
    for (std::size_t group = 0; group < n_group; ++group) {
      if (group_selected[group]) {
        continue;
      }
      if (top_group_score[group] > best_value) {
        best_value = top_group_score[group];
        best_group = group;
      }
    }
    group_selected[best_group] = true;
  }

  bool expert_taken[kMaxExperts];
  for (std::size_t expert = 0; expert < expert_count; ++expert) {
    expert_taken[expert] = false;
  }

  float weight_sum = 0.0f;
  const std::size_t expert_pick_count = top_k < expert_count ? top_k : expert_count;
  for (std::size_t pick = 0; pick < expert_pick_count; ++pick) {
    float best_value = kNegInf;
    std::size_t best_expert = 0;
    for (std::size_t expert = 0; expert < expert_count; ++expert) {
      if (expert_taken[expert]) {
        continue;
      }
      const std::size_t group = expert / group_size;
      const float masked = group_selected[group] ? scores_for_choice[expert] : 0.0f;
      if (masked > best_value) {
        best_value = masked;
        best_expert = expert;
      }
    }
    expert_taken[best_expert] = true;
    selected_indices[row * top_k + pick] = static_cast<std::int32_t>(best_expert);
    selected_weights[row * top_k + pick] = scores[best_expert];
    weight_sum += scores[best_expert];
  }

  if (norm_topk_prob) {
    const float denom = weight_sum + 1.0e-20f;
    for (std::size_t pick = 0; pick < expert_pick_count; ++pick) {
      selected_weights[row * top_k + pick] /= denom;
    }
  }
  for (std::size_t pick = 0; pick < expert_pick_count; ++pick) {
    selected_weights[row * top_k + pick] *= routed_scaling_factor;
  }
}

__global__ void ComputeRowTensorScalesKernel(
    const float* input_rows,
    const float* row_scales,
    std::size_t rows,
    std::size_t cols,
    float* tensor_scales) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  extern __shared__ float shared_max[];
  float local_max = 0.0f;
  const float row_scale = row_scales[row];
  const std::size_t row_offset = row * cols;
  for (std::size_t col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = input_rows[row_offset + col];
    value = value > 0.0f ? (value * value) : 0.0f;
    value *= row_scale;
    local_max = fmaxf(local_max, fabsf(value));
  }
  shared_max[threadIdx.x] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const float global_max_abs = shared_max[0];
    float tensor_scale = 1.0f;
    if (global_max_abs > kFp4MaxFinite * kFp8E4M3MaxFinite) {
      tensor_scale = ClampScale(global_max_abs / (kFp4MaxFinite * kFp8E4M3MaxFinite));
    }
    tensor_scales[row] = tensor_scale;
  }
}

__global__ void PackScaledRelu2RowsToNvfp4Kernel(
    const float* input_rows,
    const float* row_scales,
    const float* tensor_scales,
    std::size_t rows,
    std::size_t cols,
    std::uint8_t* packed,
    std::uint8_t* block_scales) {
  const std::size_t blocks_per_row = cols / kNvfp4BlockWidth;
  const std::size_t block_index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const std::size_t total_blocks = rows * blocks_per_row;
  if (block_index >= total_blocks) {
    return;
  }

  const std::size_t row = block_index / blocks_per_row;
  const std::size_t block = block_index % blocks_per_row;
  const std::size_t input_offset = row * cols + (block * kNvfp4BlockWidth);
  const std::size_t packed_offset = row * (cols / 2u) + (block * (kNvfp4BlockWidth / 2u));
  const float tensor_scale = tensor_scales[row];
  const float row_scale = row_scales[row];

  float block_max_abs = 0.0f;
  float transformed[kNvfp4BlockWidth];
  #pragma unroll
  for (std::size_t i = 0; i < kNvfp4BlockWidth; ++i) {
    float value = input_rows[input_offset + i];
    value = value > 0.0f ? (value * value) : 0.0f;
    value *= row_scale;
    transformed[i] = value;
    block_max_abs = fmaxf(block_max_abs, fabsf(value));
  }

  float block_scale = 1.0f;
  if (block_max_abs > 0.0f) {
    block_scale = ClampScale(block_max_abs / (kFp4MaxFinite * tensor_scale));
  }
  block_scales[block_index] = static_cast<std::uint8_t>(
      __nv_cvt_float_to_fp8(block_scale, __NV_SATFINITE, __NV_E4M3));

  const float scale = tensor_scale * block_scale;
  #pragma unroll
  for (std::size_t i = 0; i < kNvfp4BlockWidth; i += 2) {
    const float lhs = transformed[i] / scale;
    const float rhs = transformed[i + 1] / scale;
    const std::uint8_t lhs_fp4 = static_cast<std::uint8_t>(
                                     __nv_cvt_float_to_fp4(lhs, __NV_E2M1, cudaRoundNearest)) &
                                 0x0fu;
    const std::uint8_t rhs_fp4 = static_cast<std::uint8_t>(
                                     __nv_cvt_float_to_fp4(rhs, __NV_E2M1, cudaRoundNearest)) &
                                 0x0fu;
    packed[packed_offset + (i / 2u)] = static_cast<std::uint8_t>(lhs_fp4 | (rhs_fp4 << 4));
  }
}

__device__ std::size_t ExecutionScaleOffset(
    std::size_t row,
    std::size_t block_col,
    std::size_t padded_blocks_per_row) {
  const std::size_t num_k_tiles = padded_blocks_per_row / kScaleBlockTile;
  const std::size_t m_tile = row / kScaleRowTile;
  const std::size_t outer_m = row & 31u;
  const std::size_t inner_m = (row >> 5u) & 3u;
  const std::size_t k_tile = block_col / kScaleBlockTile;
  const std::size_t inner_k = block_col & 3u;
  return ((((m_tile * num_k_tiles) + k_tile) << 9u) |
          (outer_m << 4u) |
          (inner_m << 2u) |
          inner_k);
}

__global__ void SwizzlePerRowBlockScalesKernel(
    const std::uint8_t* row_major_scales,
    std::size_t rows,
    std::size_t logical_blocks_per_row,
    std::size_t padded_blocks_per_row,
    std::size_t matmul_bytes_per_row,
    std::uint8_t* matmul_scales) {
  const std::size_t scale_index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const std::size_t total_scales = rows * logical_blocks_per_row;
  if (scale_index >= total_scales) {
    return;
  }
  const std::size_t row = scale_index / logical_blocks_per_row;
  const std::size_t block_col = scale_index % logical_blocks_per_row;
  const std::size_t destination_offset =
      row * matmul_bytes_per_row + ExecutionScaleOffset(0, block_col, padded_blocks_per_row);
  matmul_scales[destination_offset] = row_major_scales[scale_index];
}

// Fused kernel: relu2 + tensor-scale reduction + FP4 packing + scale swizzle
// in one launch. One block per expert row.
__global__ void FusedRelu2PackRowsToNvfp4Kernel(
    const float* input_rows,
    const float* row_scales,
    std::size_t rows,
    std::size_t cols,
    float* tensor_scales,
    std::uint8_t* packed,
    std::uint8_t* block_scales,
    std::size_t packed_row_stride_bytes,
    std::size_t padded_blocks_per_row,
    std::size_t matmul_bytes_per_row,
    std::uint8_t* matmul_scales) {
  extern __shared__ float smem[];
  const std::size_t row = static_cast<std::size_t>(blockIdx.x);
  if (row >= rows) return;

  const std::size_t tid = threadIdx.x;
  const float row_scale = row_scales[row];
  const std::size_t row_offset = row * cols;
  const std::size_t blocks_per_row = cols / kNvfp4BlockWidth;
  const std::size_t packed_bytes_per_row = cols / 2u;
  std::uint8_t* row_packed = packed + row * packed_row_stride_bytes;
  std::uint8_t* row_matmul =
      matmul_scales != nullptr ? (matmul_scales + row * matmul_bytes_per_row) : nullptr;

  float local_max = 0.0f;
  for (std::size_t col = tid; col < cols; col += blockDim.x) {
    float value = input_rows[row_offset + col];
    value = value > 0.0f ? (value * value) : 0.0f;
    value *= row_scale;
    local_max = fmaxf(local_max, fabsf(value));
  }
  smem[tid] = local_max;
  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      smem[tid] = fmaxf(smem[tid], smem[tid + s]);
    }
    __syncthreads();
  }

  float tensor_scale = 1.0f;
  if (smem[0] > kFp4MaxFinite * kFp8E4M3MaxFinite) {
    tensor_scale = ClampScale(smem[0] / (kFp4MaxFinite * kFp8E4M3MaxFinite));
  }
  if (tid == 0) {
    tensor_scales[row] = tensor_scale;
    smem[0] = tensor_scale;
  }
  __syncthreads();
  tensor_scale = smem[0];

  if (packed_row_stride_bytes > packed_bytes_per_row) {
    for (std::size_t i = packed_bytes_per_row + tid; i < packed_row_stride_bytes; i += blockDim.x) {
      row_packed[i] = 0;
    }
  }

  if (row_matmul != nullptr) {
    for (std::size_t i = tid; i < matmul_bytes_per_row; i += blockDim.x) {
      row_matmul[i] = 0;
    }
    __syncthreads();
  }

  for (std::size_t block_index = tid; block_index < blocks_per_row;
       block_index += blockDim.x) {
    const std::size_t input_offset = row_offset + block_index * kNvfp4BlockWidth;
    const std::size_t packed_offset = block_index * (kNvfp4BlockWidth / 2u);
    const std::size_t scale_offset = row * blocks_per_row + block_index;

    float block_max_abs = 0.0f;
    float transformed[kNvfp4BlockWidth];
    #pragma unroll
    for (std::size_t i = 0; i < kNvfp4BlockWidth; ++i) {
      float value = input_rows[input_offset + i];
      value = value > 0.0f ? (value * value) : 0.0f;
      value *= row_scale;
      transformed[i] = value;
      block_max_abs = fmaxf(block_max_abs, fabsf(value));
    }

    float block_scale = 1.0f;
    if (block_max_abs > 0.0f) {
      block_scale = ClampScale(block_max_abs / (kFp4MaxFinite * tensor_scale));
    }
    const std::uint8_t block_scale_fp8 = static_cast<std::uint8_t>(
        __nv_cvt_float_to_fp8(block_scale, __NV_SATFINITE, __NV_E4M3));
    block_scales[scale_offset] = block_scale_fp8;

    const float scale = tensor_scale * block_scale;
    #pragma unroll
    for (std::size_t i = 0; i < kNvfp4BlockWidth; i += 2) {
      const float lhs = transformed[i] / scale;
      const float rhs = transformed[i + 1] / scale;
      const std::uint8_t lhs_fp4 = static_cast<std::uint8_t>(
                                       __nv_cvt_float_to_fp4(lhs, __NV_E2M1, cudaRoundNearest)) &
                                   0x0fu;
      const std::uint8_t rhs_fp4 = static_cast<std::uint8_t>(
                                       __nv_cvt_float_to_fp4(rhs, __NV_E2M1, cudaRoundNearest)) &
                                   0x0fu;
      row_packed[packed_offset + (i / 2u)] = static_cast<std::uint8_t>(lhs_fp4 | (rhs_fp4 << 4));
    }

    if (row_matmul != nullptr) {
      const std::size_t dest = ExecutionScaleOffset(0, block_index, padded_blocks_per_row);
      row_matmul[dest] = block_scale_fp8;
    }
  }
}

__global__ void WeightedSumRowsKernel(
    const float* input_rows,
    const float* row_scales,
    std::size_t rows,
    std::size_t cols,
    float* output_row) {
  const std::size_t col = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (col >= cols) {
    return;
  }
  float accum = 0.0f;
  for (std::size_t row = 0; row < rows; ++row) {
    accum += input_rows[row * cols + col] * row_scales[row];
  }
  output_row[col] = accum;
}

__global__ void GatherExpertSelectionLookupsKernel(
    const std::int32_t* selected_indices,
    std::size_t selection_count,
    std::size_t lookup_count,
    const void* const* packed_lookup,
    const void* const* matmul_scale_lookup,
    const float* tensor_scale_lookup,
    const void** selected_packed,
    const void** selected_matmul_scales,
    float* selected_tensor_scales) {
  const std::size_t index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (index >= selection_count) {
    return;
  }
  const std::int32_t expert_index = selected_indices[index];
  if (expert_index < 0 || static_cast<std::size_t>(expert_index) >= lookup_count) {
    selected_packed[index] = nullptr;
    selected_matmul_scales[index] = nullptr;
    selected_tensor_scales[index] = 0.0f;
    return;
  }
  const std::size_t lookup_index = static_cast<std::size_t>(expert_index);
  selected_packed[index] = packed_lookup[lookup_index];
  selected_matmul_scales[index] = matmul_scale_lookup[lookup_index];
  selected_tensor_scales[index] = tensor_scale_lookup[lookup_index];
}

__global__ void GatherExpertSelectionLookupsCheckedKernel(
    const std::int32_t* selected_indices,
    std::size_t selection_count,
    std::size_t lookup_count,
    const void* const* packed_lookup,
    const void* const* matmul_scale_lookup,
    const float* tensor_scale_lookup,
    const void** selected_packed,
    const void** selected_matmul_scales,
    float* selected_tensor_scales,
    std::uint32_t* missing_count,
    std::int32_t* missing_indices) {
  const std::size_t index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (index >= selection_count) {
    return;
  }
  const std::int32_t expert_index = selected_indices[index];
  if (expert_index < 0 || static_cast<std::size_t>(expert_index) >= lookup_count) {
    selected_packed[index] = nullptr;
    selected_matmul_scales[index] = nullptr;
    selected_tensor_scales[index] = 0.0f;
    const std::uint32_t missing_index = atomicAdd(missing_count, 1u);
    if (missing_indices != nullptr) {
      missing_indices[missing_index] = expert_index;
    }
    return;
  }

  const std::size_t lookup_index = static_cast<std::size_t>(expert_index);
  const void* packed = packed_lookup[lookup_index];
  const void* matmul_scale = matmul_scale_lookup[lookup_index];
  const float tensor_scale = tensor_scale_lookup[lookup_index];
  selected_packed[index] = packed;
  selected_matmul_scales[index] = matmul_scale;
  selected_tensor_scales[index] = tensor_scale;
  if (packed == nullptr || matmul_scale == nullptr || tensor_scale == 0.0f) {
    const std::uint32_t missing_index = atomicAdd(missing_count, 1u);
    if (missing_indices != nullptr) {
      missing_indices[missing_index] = expert_index;
    }
  }
}

// Merged up+down gather in a single kernel launch. Both use the same
// selected_indices but index different lookup tables.
__global__ void GatherExpertSelectionLookupsDualCheckedKernel(
    const std::int32_t* selected_indices,
    std::size_t selection_count,
    std::size_t lookup_count,
    const void* const* up_packed_lookup,
    const void* const* up_raw_scale_lookup,
    const void* const* up_matmul_scale_lookup,
    const float* up_tensor_scale_lookup,
    const void** selected_up_packed,
    const void** selected_up_raw_scales,
    const void** selected_up_matmul_scales,
    float* selected_up_tensor_scales,
    const void* const* down_packed_lookup,
    const void* const* down_raw_scale_lookup,
    const void* const* down_matmul_scale_lookup,
    const float* down_tensor_scale_lookup,
    const void** selected_down_packed,
    const void** selected_down_raw_scales,
    const void** selected_down_matmul_scales,
    float* selected_down_tensor_scales,
    std::uint32_t* missing_count,
    std::int32_t* missing_indices) {
  const std::size_t index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (index >= selection_count) {
    return;
  }
  const std::int32_t expert_index = selected_indices[index];
  if (expert_index < 0 || static_cast<std::size_t>(expert_index) >= lookup_count) {
    selected_up_packed[index] = nullptr;
    selected_up_raw_scales[index] = nullptr;
    selected_up_matmul_scales[index] = nullptr;
    selected_up_tensor_scales[index] = 0.0f;
    selected_down_packed[index] = nullptr;
    selected_down_raw_scales[index] = nullptr;
    selected_down_matmul_scales[index] = nullptr;
    selected_down_tensor_scales[index] = 0.0f;
    const std::uint32_t missing_index = atomicAdd(missing_count, 1u);
    if (missing_indices != nullptr) {
      missing_indices[missing_index] = expert_index;
    }
    return;
  }

  const std::size_t li = static_cast<std::size_t>(expert_index);
  const void* up_p = up_packed_lookup[li];
  const void* up_rs = up_raw_scale_lookup[li];
  const void* up_ms = up_matmul_scale_lookup[li];
  const float up_ts = up_tensor_scale_lookup[li];
  selected_up_packed[index] = up_p;
  selected_up_raw_scales[index] = up_rs;
  selected_up_matmul_scales[index] = up_ms;
  selected_up_tensor_scales[index] = up_ts;

  const void* down_p = down_packed_lookup[li];
  const void* down_rs = down_raw_scale_lookup[li];
  const void* down_ms = down_matmul_scale_lookup[li];
  const float down_ts = down_tensor_scale_lookup[li];
  selected_down_packed[index] = down_p;
  selected_down_raw_scales[index] = down_rs;
  selected_down_matmul_scales[index] = down_ms;
  selected_down_tensor_scales[index] = down_ts;

  if (up_p == nullptr || up_rs == nullptr || up_ms == nullptr || up_ts == 0.0f ||
      down_p == nullptr || down_rs == nullptr || down_ms == nullptr || down_ts == 0.0f) {
    const std::uint32_t missing_index = atomicAdd(missing_count, 1u);
    if (missing_indices != nullptr) {
      missing_indices[missing_index] = expert_index;
    }
  }
}

__global__ void FillPointerArrayKernel(
    const void* value,
    std::size_t count,
    const void** output) {
  const std::size_t index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (index >= count) {
    return;
  }
  output[index] = value;
}

__global__ void FillByteOffsetPointerArrayKernel(
    const std::uint8_t* base,
    std::size_t row_stride_bytes,
    std::size_t count,
    const void** output) {
  const std::size_t index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (index >= count) {
    return;
  }
  output[index] = base + (index * row_stride_bytes);
}

__global__ void FillOutputPointerArrayKernel(
    float* base,
    std::size_t row_stride_elems,
    std::size_t count,
    void** output) {
  const std::size_t index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (index >= count) {
    return;
  }
  output[index] = base + (index * row_stride_elems);
}

__global__ void ComputeGroupedUpPackScalesKernel(
    const float* activation_tensor_scale,
    const float* weight_tensor_scales,
    std::size_t count,
    float* output_row_scales) {
  const std::size_t index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (index >= count) {
    return;
  }
  output_row_scales[index] = (*activation_tensor_scale) * weight_tensor_scales[index];
}

__global__ void ComputeWeightedMergeScalesKernel(
    const float* activation_tensor_scales,
    const float* weight_tensor_scales,
    std::size_t count,
    float* output_row_scales) {
  const std::size_t index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (index >= count) {
    return;
  }
  output_row_scales[index] =
      activation_tensor_scales[index] * weight_tensor_scales[index];
}

__global__ void FusedRoutedUpProjPackedNvfp4SingleTokenKernel(
    const std::uint8_t* activation_packed,
    const std::uint8_t* activation_block_scales,
    const float* activation_tensor_scale,
    std::size_t input_cols,
    const void* const* weight_packed_ptrs,
    const void* const* weight_block_scale_ptrs,
    const float* weight_tensor_scales,
    std::size_t output_rows,
    float* output_rows_data) {
  const std::size_t output_row = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const std::size_t expert_row = static_cast<std::size_t>(blockIdx.y);
  if (output_row >= output_rows) {
    return;
  }

  extern __shared__ float shared_activation[];
  const std::size_t blocks_per_row = input_cols / kNvfp4BlockWidth;
  const auto* weight_packed =
      reinterpret_cast<const std::uint8_t*>(weight_packed_ptrs[expert_row]);
  const auto* weight_block_scales =
      reinterpret_cast<const std::uint8_t*>(weight_block_scale_ptrs[expert_row]);
  const float act_tensor_scale = *activation_tensor_scale;
  const float weight_tensor_scale = weight_tensor_scales[expert_row];
  const std::size_t packed_row_stride = input_cols / 2u;
  const std::size_t scale_row_stride = blocks_per_row;
  float accum = 0.0f;

  for (std::size_t block = 0; block < blocks_per_row; ++block) {
    if (threadIdx.x < kNvfp4BlockWidth) {
      const std::size_t packed_index = block * (kNvfp4BlockWidth / 2u) + (threadIdx.x / 2u);
      const std::uint8_t packed_pair = activation_packed[packed_index];
      const std::uint8_t nibble =
          (threadIdx.x & 1u) != 0u ? (packed_pair >> 4u) : (packed_pair & 0x0fu);
      const float block_scale = DecodeFp8E4M3(activation_block_scales[block]);
      shared_activation[threadIdx.x] = DecodeFp4E2M1(nibble) * block_scale * act_tensor_scale;
    }
    __syncthreads();

    const float weight_block_scale =
        DecodeFp8E4M3(weight_block_scales[output_row * scale_row_stride + block]) *
        weight_tensor_scale;
    const std::size_t packed_offset = output_row * packed_row_stride + block * (kNvfp4BlockWidth / 2u);
    #pragma unroll
    for (std::size_t inner = 0; inner < kNvfp4BlockWidth; inner += 2u) {
      const std::uint8_t packed_pair = weight_packed[packed_offset + (inner / 2u)];
      const float lhs = DecodeFp4E2M1(packed_pair & 0x0fu) * weight_block_scale;
      const float rhs = DecodeFp4E2M1((packed_pair >> 4u) & 0x0fu) * weight_block_scale;
      accum += shared_activation[inner] * lhs;
      accum += shared_activation[inner + 1u] * rhs;
    }
    __syncthreads();
  }

  output_rows_data[expert_row * output_rows + output_row] = accum;
}

__global__ void FusedRoutedDownProjWeightedPackedNvfp4SingleTokenKernel(
    const std::uint8_t* activation_rows_packed,
    const std::uint8_t* activation_rows_block_scales,
    const float* activation_row_tensor_scales,
    const float* selection_weights,
    std::size_t selection_count,
    std::size_t input_cols,
    std::size_t activation_rows_packed_row_stride_bytes,
    const void* const* weight_packed_ptrs,
    const void* const* weight_block_scale_ptrs,
    const float* weight_tensor_scales,
    std::size_t output_rows,
    float* output_row_data) {
  const std::size_t output_row = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (output_row >= output_rows) {
    return;
  }

  extern __shared__ float shared_activation[];
  const std::size_t blocks_per_row = input_cols / kNvfp4BlockWidth;
  const std::size_t activation_packed_row_stride =
      activation_rows_packed_row_stride_bytes == 0
          ? (input_cols / 2u)
          : activation_rows_packed_row_stride_bytes;
  const std::size_t weight_packed_row_stride = input_cols / 2u;
  const std::size_t scale_row_stride = blocks_per_row;
  float accum = 0.0f;

  for (std::size_t expert_row = 0; expert_row < selection_count; ++expert_row) {
    const auto* weight_packed =
        reinterpret_cast<const std::uint8_t*>(weight_packed_ptrs[expert_row]);
    const auto* weight_block_scales =
        reinterpret_cast<const std::uint8_t*>(weight_block_scale_ptrs[expert_row]);
    const float act_tensor_scale = activation_row_tensor_scales[expert_row];
    const float weight_tensor_scale = weight_tensor_scales[expert_row];
    const float expert_weight = selection_weights[expert_row];

    for (std::size_t block = 0; block < blocks_per_row; ++block) {
      if (threadIdx.x < kNvfp4BlockWidth) {
        const std::size_t packed_index =
            expert_row * activation_packed_row_stride +
            block * (kNvfp4BlockWidth / 2u) +
            (threadIdx.x / 2u);
        const std::uint8_t packed_pair = activation_rows_packed[packed_index];
        const std::uint8_t nibble =
            (threadIdx.x & 1u) != 0u ? (packed_pair >> 4u) : (packed_pair & 0x0fu);
        const float block_scale =
            DecodeFp8E4M3(activation_rows_block_scales[expert_row * scale_row_stride + block]);
        shared_activation[threadIdx.x] = DecodeFp4E2M1(nibble) * block_scale * act_tensor_scale;
      }
      __syncthreads();

      const float weight_block_scale =
          DecodeFp8E4M3(weight_block_scales[output_row * scale_row_stride + block]) *
          weight_tensor_scale;
      const std::size_t packed_offset =
          output_row * weight_packed_row_stride + block * (kNvfp4BlockWidth / 2u);
      #pragma unroll
      for (std::size_t inner = 0; inner < kNvfp4BlockWidth; inner += 2u) {
        const std::uint8_t packed_pair = weight_packed[packed_offset + (inner / 2u)];
        const float lhs = DecodeFp4E2M1(packed_pair & 0x0fu) * weight_block_scale;
        const float rhs = DecodeFp4E2M1((packed_pair >> 4u) & 0x0fu) * weight_block_scale;
        accum += expert_weight * shared_activation[inner] * lhs;
        accum += expert_weight * shared_activation[inner + 1u] * rhs;
      }
      __syncthreads();
    }
  }

  output_row_data[output_row] = accum;
}

bool HasSingleRowShape(const DeviceTensorFp32& tensor) {
  return tensor.valid() && tensor.shape().size() == 2 && tensor.shape()[0] == 1 && tensor.shape()[1] != 0;
}

}  // namespace

bool CopyRowFp32(
    const DeviceTensorFp32& input,
    std::size_t row_index,
    DeviceTensorFp32* output_row) {
  if (!input.valid() || output_row == nullptr || !output_row->valid() || input.shape().size() != 2 ||
      row_index >= input.shape()[0] || !HasSingleRowShape(*output_row) ||
      output_row->shape()[1] != input.shape()[1]) {
    return false;
  }
  const std::size_t cols = input.shape()[1];
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((cols + block.x - 1) / block.x));
  CopyRowKernel<<<grid, block>>>(input.data(), cols, row_index, output_row->data());
  return CheckCuda(cudaGetLastError());
}

bool WriteRowFp32(
    const DeviceTensorFp32& input_row,
    std::size_t row_index,
    DeviceTensorFp32* output) {
  if (!HasSingleRowShape(input_row) || output == nullptr || !output->valid() || output->shape().size() != 2 ||
      row_index >= output->shape()[0] || input_row.shape()[1] != output->shape()[1]) {
    return false;
  }
  const std::size_t cols = input_row.shape()[1];
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((cols + block.x - 1) / block.x));
  WriteRowKernel<<<grid, block>>>(input_row.data(), cols, row_index, output->data());
  return CheckCuda(cudaGetLastError());
}

bool Relu2InPlaceFp32(DeviceTensorFp32* tensor) {
  if (tensor == nullptr || !tensor->valid()) {
    return false;
  }
  const std::size_t count = tensor->numel();
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1) / block.x));
  Relu2InPlaceKernel<<<grid, block>>>(tensor->data(), count);
  return CheckCuda(cudaGetLastError());
}

bool AddScaledFp32(
    const DeviceTensorFp32& input,
    float scale,
    DeviceTensorFp32* accumulator) {
  if (!input.valid() || accumulator == nullptr || !accumulator->valid() || input.shape() != accumulator->shape()) {
    return false;
  }
  const std::size_t count = input.numel();
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1) / block.x));
  AddScaledKernel<<<grid, block>>>(input.data(), scale, accumulator->data(), count);
  return CheckCuda(cudaGetLastError());
}

bool AddScaledRowFp32(
    const DeviceTensorFp32& input_row,
    float scale,
    std::size_t row_index,
    DeviceTensorFp32* accumulator) {
  if (!HasSingleRowShape(input_row) || accumulator == nullptr || !accumulator->valid() ||
      accumulator->shape().size() != 2 || row_index >= accumulator->shape()[0] ||
      input_row.shape()[1] != accumulator->shape()[1]) {
    return false;
  }
  const std::size_t cols = input_row.shape()[1];
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((cols + block.x - 1) / block.x));
  AddScaledRowKernel<<<grid, block>>>(input_row.data(), scale, row_index, cols, accumulator->data());
  return CheckCuda(cudaGetLastError());
}

bool SelectTopExpertsFp32(
    const DeviceTensorFp32& router_logits,
    const DeviceTensorFp32& correction_bias,
    std::size_t n_group,
    std::size_t topk_group,
    std::size_t top_k,
    bool norm_topk_prob,
    float routed_scaling_factor,
    std::int32_t* selected_indices_device,
    float* selected_weights_device) {
  if (!router_logits.valid() || !correction_bias.valid() || router_logits.shape().size() != 2 ||
      correction_bias.shape().size() != 1 || correction_bias.shape()[0] != router_logits.shape()[1] ||
      selected_indices_device == nullptr || selected_weights_device == nullptr) {
    return false;
  }
  const std::size_t rows = router_logits.shape()[0];
  const std::size_t expert_count = router_logits.shape()[1];
  SelectTopExpertsKernel<<<static_cast<unsigned int>(rows), 1>>>(
      router_logits.data(),
      correction_bias.data(),
      rows,
      expert_count,
      n_group,
      topk_group,
      top_k,
      norm_topk_prob,
      routed_scaling_factor,
      selected_indices_device,
      selected_weights_device);
  return CheckCuda(cudaGetLastError());
}

bool GatherExpertSelectionLookups(
    const std::int32_t* selected_indices_device,
    std::size_t selection_count,
    std::size_t lookup_count,
    const DeviceBuffer<const void*>& packed_lookup,
    const DeviceBuffer<const void*>& matmul_scale_lookup,
    const DeviceBuffer<float>& tensor_scale_lookup,
    DeviceBuffer<const void*>* selected_packed_ptrs,
    DeviceBuffer<const void*>* selected_matmul_scale_ptrs,
    DeviceBuffer<float>* selected_tensor_scales) {
  if (selected_indices_device == nullptr ||
      selection_count == 0 ||
      lookup_count == 0 ||
      !packed_lookup.valid() ||
      !matmul_scale_lookup.valid() ||
      !tensor_scale_lookup.valid() ||
      packed_lookup.count() != matmul_scale_lookup.count() ||
      packed_lookup.count() != tensor_scale_lookup.count() ||
      selected_packed_ptrs == nullptr ||
      selected_matmul_scale_ptrs == nullptr ||
      selected_tensor_scales == nullptr) {
    return false;
  }
  if (!selected_packed_ptrs->Resize(selection_count) ||
      !selected_matmul_scale_ptrs->Resize(selection_count) ||
      !selected_tensor_scales->Resize(selection_count)) {
    return false;
  }
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((selection_count + block.x - 1) / block.x));
  GatherExpertSelectionLookupsKernel<<<grid, block>>>(
      selected_indices_device,
      selection_count,
      lookup_count,
      packed_lookup.data(),
      matmul_scale_lookup.data(),
      tensor_scale_lookup.data(),
      selected_packed_ptrs->data(),
      selected_matmul_scale_ptrs->data(),
      selected_tensor_scales->data());
  return CheckCuda(cudaGetLastError());
}

bool GatherExpertSelectionLookupsChecked(
    const std::int32_t* selected_indices_device,
    std::size_t selection_count,
    std::size_t lookup_count,
    const DeviceBuffer<const void*>& packed_lookup,
    const DeviceBuffer<const void*>& matmul_scale_lookup,
    const DeviceBuffer<float>& tensor_scale_lookup,
    DeviceBuffer<const void*>* selected_packed_ptrs,
    DeviceBuffer<const void*>* selected_matmul_scale_ptrs,
    DeviceBuffer<float>* selected_tensor_scales,
    DeviceBuffer<std::uint32_t>* missing_count,
    DeviceBuffer<std::int32_t>* missing_indices) {
  if (selected_indices_device == nullptr ||
      selection_count == 0 ||
      lookup_count == 0 ||
      !packed_lookup.valid() ||
      !matmul_scale_lookup.valid() ||
      !tensor_scale_lookup.valid() ||
      packed_lookup.count() != matmul_scale_lookup.count() ||
      packed_lookup.count() != tensor_scale_lookup.count() ||
      selected_packed_ptrs == nullptr ||
      selected_matmul_scale_ptrs == nullptr ||
      selected_tensor_scales == nullptr ||
      missing_count == nullptr ||
      missing_indices == nullptr) {
    return false;
  }
  if (!selected_packed_ptrs->Resize(selection_count) ||
      !selected_matmul_scale_ptrs->Resize(selection_count) ||
      !selected_tensor_scales->Resize(selection_count) ||
      !missing_count->Resize(1) ||
      !missing_indices->Resize(selection_count) ||
      !missing_count->FillZero()) {
    return false;
  }
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((selection_count + block.x - 1) / block.x));
  GatherExpertSelectionLookupsCheckedKernel<<<grid, block>>>(
      selected_indices_device,
      selection_count,
      lookup_count,
      packed_lookup.data(),
      matmul_scale_lookup.data(),
      tensor_scale_lookup.data(),
      selected_packed_ptrs->data(),
      selected_matmul_scale_ptrs->data(),
      selected_tensor_scales->data(),
      missing_count->data(),
      missing_indices->data());
  return CheckCuda(cudaGetLastError());
}

bool GatherExpertSelectionLookupsCheckedInPlace(
    const std::int32_t* selected_indices_device,
    std::size_t selection_count,
    std::size_t lookup_count,
    const DeviceBuffer<const void*>& packed_lookup,
    const DeviceBuffer<const void*>& matmul_scale_lookup,
    const DeviceBuffer<float>& tensor_scale_lookup,
    DeviceBuffer<const void*>& selected_packed_ptrs,
    DeviceBuffer<const void*>& selected_matmul_scale_ptrs,
    DeviceBuffer<float>& selected_tensor_scales,
    DeviceBuffer<std::uint32_t>& missing_count,
    DeviceBuffer<std::int32_t>& missing_indices) {
  if (selected_indices_device == nullptr ||
      selection_count == 0 ||
      lookup_count == 0 ||
      !packed_lookup.valid() ||
      !matmul_scale_lookup.valid() ||
      !tensor_scale_lookup.valid() ||
      packed_lookup.count() != matmul_scale_lookup.count() ||
      packed_lookup.count() != tensor_scale_lookup.count() ||
      !selected_packed_ptrs.valid() ||
      selected_packed_ptrs.count() < selection_count ||
      !selected_matmul_scale_ptrs.valid() ||
      selected_matmul_scale_ptrs.count() < selection_count ||
      !selected_tensor_scales.valid() ||
      selected_tensor_scales.count() < selection_count ||
      !missing_count.valid() ||
      missing_count.count() < 1 ||
      !missing_indices.valid() ||
      missing_indices.count() < selection_count) {
    return false;
  }
  if (!missing_count.FillZero()) {
    return false;
  }
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((selection_count + block.x - 1) / block.x));
  GatherExpertSelectionLookupsCheckedKernel<<<grid, block>>>(
      selected_indices_device,
      selection_count,
      lookup_count,
      packed_lookup.data(),
      matmul_scale_lookup.data(),
      tensor_scale_lookup.data(),
      selected_packed_ptrs.data(),
      selected_matmul_scale_ptrs.data(),
      selected_tensor_scales.data(),
      missing_count.data(),
      missing_indices.data());
  return CheckCuda(cudaGetLastError());
}

bool GatherExpertSelectionLookupsDualCheckedInPlace(
    const std::int32_t* selected_indices_device,
    std::size_t selection_count,
    std::size_t lookup_count,
    const DeviceBuffer<const void*>& up_packed_lookup,
    const DeviceBuffer<const void*>& up_raw_scale_lookup,
    const DeviceBuffer<const void*>& up_matmul_scale_lookup,
    const DeviceBuffer<float>& up_tensor_scale_lookup,
    DeviceBuffer<const void*>& selected_up_packed_ptrs,
    DeviceBuffer<const void*>& selected_up_raw_scale_ptrs,
    DeviceBuffer<const void*>& selected_up_matmul_scale_ptrs,
    DeviceBuffer<float>& selected_up_tensor_scales,
    const DeviceBuffer<const void*>& down_packed_lookup,
    const DeviceBuffer<const void*>& down_raw_scale_lookup,
    const DeviceBuffer<const void*>& down_matmul_scale_lookup,
    const DeviceBuffer<float>& down_tensor_scale_lookup,
    DeviceBuffer<const void*>& selected_down_packed_ptrs,
    DeviceBuffer<const void*>& selected_down_raw_scale_ptrs,
    DeviceBuffer<const void*>& selected_down_matmul_scale_ptrs,
    DeviceBuffer<float>& selected_down_tensor_scales,
    DeviceBuffer<std::uint32_t>& missing_count,
    DeviceBuffer<std::int32_t>& missing_indices,
    cudaStream_t stream) {
  if (selected_indices_device == nullptr ||
      selection_count == 0 || lookup_count == 0 ||
      !up_packed_lookup.valid() || !up_raw_scale_lookup.valid() ||
      !up_matmul_scale_lookup.valid() ||
      !up_tensor_scale_lookup.valid() ||
      !selected_up_packed_ptrs.valid() || selected_up_packed_ptrs.count() < selection_count ||
      !selected_up_raw_scale_ptrs.valid() || selected_up_raw_scale_ptrs.count() < selection_count ||
      !selected_up_matmul_scale_ptrs.valid() || selected_up_matmul_scale_ptrs.count() < selection_count ||
      !selected_up_tensor_scales.valid() || selected_up_tensor_scales.count() < selection_count ||
      !down_packed_lookup.valid() || !down_raw_scale_lookup.valid() ||
      !down_matmul_scale_lookup.valid() ||
      !down_tensor_scale_lookup.valid() ||
      !selected_down_packed_ptrs.valid() || selected_down_packed_ptrs.count() < selection_count ||
      !selected_down_raw_scale_ptrs.valid() || selected_down_raw_scale_ptrs.count() < selection_count ||
      !selected_down_matmul_scale_ptrs.valid() || selected_down_matmul_scale_ptrs.count() < selection_count ||
      !selected_down_tensor_scales.valid() || selected_down_tensor_scales.count() < selection_count ||
      !missing_count.valid() || missing_count.count() < 1 ||
      !missing_indices.valid() || missing_indices.count() < selection_count) {
    return false;
  }
  if (!missing_count.FillZeroAsync(stream)) {
    return false;
  }
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((selection_count + block.x - 1) / block.x));
  GatherExpertSelectionLookupsDualCheckedKernel<<<grid, block, 0, stream>>>(
      selected_indices_device,
      selection_count,
      lookup_count,
      up_packed_lookup.data(),
      up_raw_scale_lookup.data(),
      up_matmul_scale_lookup.data(),
      up_tensor_scale_lookup.data(),
      selected_up_packed_ptrs.data(),
      selected_up_raw_scale_ptrs.data(),
      selected_up_matmul_scale_ptrs.data(),
      selected_up_tensor_scales.data(),
      down_packed_lookup.data(),
      down_raw_scale_lookup.data(),
      down_matmul_scale_lookup.data(),
      down_tensor_scale_lookup.data(),
      selected_down_packed_ptrs.data(),
      selected_down_raw_scale_ptrs.data(),
      selected_down_matmul_scale_ptrs.data(),
      selected_down_tensor_scales.data(),
      missing_count.data(),
      missing_indices.data());
  return CheckCuda(cudaGetLastError());
}

bool FillDevicePointerArray(
    const void* value,
    std::size_t count,
    DeviceBuffer<const void*>* output) {
  if (output == nullptr) {
    return false;
  }
  if (count == 0) {
    return output->Resize(0);
  }
  if (!output->Resize(count)) {
    return false;
  }
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1) / block.x));
  FillPointerArrayKernel<<<grid, block>>>(value, count, output->data());
  return CheckCuda(cudaGetLastError());
}

bool FillDeviceByteOffsetPointerArray(
    const std::uint8_t* base,
    std::size_t row_stride_bytes,
    std::size_t count,
    DeviceBuffer<const void*>* output) {
  if (output == nullptr) {
    return false;
  }
  if (count == 0) {
    return output->Resize(0);
  }
  if (base == nullptr || row_stride_bytes == 0 || !output->Resize(count)) {
    return false;
  }
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1) / block.x));
  FillByteOffsetPointerArrayKernel<<<grid, block>>>(
      base,
      row_stride_bytes,
      count,
      output->data());
  return CheckCuda(cudaGetLastError());
}

bool FillDeviceOutputPointerArray(
    float* base,
    std::size_t row_stride_elems,
    std::size_t count,
    DeviceBuffer<void*>* output) {
  if (output == nullptr) {
    return false;
  }
  if (count == 0) {
    return output->Resize(0);
  }
  if (base == nullptr || row_stride_elems == 0 || !output->Resize(count)) {
    return false;
  }
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1) / block.x));
  FillOutputPointerArrayKernel<<<grid, block>>>(base, row_stride_elems, count, output->data());
  return CheckCuda(cudaGetLastError());
}

bool ComputeGroupedUpPackScales(
    const float* activation_tensor_scale_device,
    const DeviceBuffer<float>& weight_tensor_scales,
    DeviceBuffer<float>* output_row_scales) {
  if (activation_tensor_scale_device == nullptr ||
      !weight_tensor_scales.valid() ||
      output_row_scales == nullptr ||
      !output_row_scales->Resize(weight_tensor_scales.count())) {
    return false;
  }
  const std::size_t count = weight_tensor_scales.count();
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1) / block.x));
  ComputeGroupedUpPackScalesKernel<<<grid, block>>>(
      activation_tensor_scale_device,
      weight_tensor_scales.data(),
      count,
      output_row_scales->data());
  return CheckCuda(cudaGetLastError());
}

bool ComputeWeightedMergeScales(
    const DeviceBuffer<float>& activation_tensor_scales,
    const DeviceBuffer<float>& weight_tensor_scales,
    DeviceBuffer<float>* output_row_scales) {
  if (!activation_tensor_scales.valid() ||
      !weight_tensor_scales.valid() ||
      activation_tensor_scales.count() != weight_tensor_scales.count() ||
      output_row_scales == nullptr ||
      !output_row_scales->Resize(activation_tensor_scales.count())) {
    return false;
  }
  const std::size_t count = activation_tensor_scales.count();
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1) / block.x));
  ComputeWeightedMergeScalesKernel<<<grid, block>>>(
      activation_tensor_scales.data(),
      weight_tensor_scales.data(),
      count,
      output_row_scales->data());
  return CheckCuda(cudaGetLastError());
}

bool ScaleRelu2PackRowsToNvfp4(
    const DeviceTensorFp32& input_rows,
    const float* row_scales_device,
    DeviceBuffer<std::uint8_t>* packed,
    DeviceBuffer<std::uint8_t>* block_scales,
    DeviceBuffer<std::uint8_t>* matmul_block_scales,
    DeviceBuffer<float>* tensor_scales) {
  if (!input_rows.valid() ||
      input_rows.shape().size() != 2 ||
      row_scales_device == nullptr ||
      packed == nullptr ||
      block_scales == nullptr ||
      tensor_scales == nullptr) {
    return false;
  }
  const std::size_t rows = input_rows.shape()[0];
  const std::size_t cols = input_rows.shape()[1];
  if (rows == 0 || cols == 0 || cols % kNvfp4BlockWidth != 0) {
    return false;
  }

  const std::size_t blocks_per_row = cols / kNvfp4BlockWidth;
  const std::size_t packed_row_stride_bytes = cols / 2u;
  const std::size_t packed_bytes = rows * packed_row_stride_bytes;
  const std::size_t block_scale_bytes = rows * blocks_per_row;
  std::size_t matmul_bytes_per_row = 0;
  std::optional<Nvfp4ExecutionScaleLayout> layout;
  if (matmul_block_scales != nullptr) {
    matmul_bytes_per_row = ExecutionNvfp4ScaleBytes(1, cols);
    if (matmul_bytes_per_row == 0) {
      return false;
    }
    layout = BuildNvfp4ExecutionScaleLayout(1, cols);
    if (!layout.has_value()) {
      return false;
    }
  }

  if (!packed->Resize(packed_bytes) ||
      !block_scales->Resize(block_scale_bytes) ||
      !tensor_scales->Resize(rows)) {
    return false;
  }
  if (matmul_block_scales != nullptr &&
      !matmul_block_scales->Resize(rows * matmul_bytes_per_row)) {
    return false;
  }

  FusedRelu2PackRowsToNvfp4Kernel<<<static_cast<unsigned int>(rows), kThreadsPerBlock,
                                    sizeof(float) * kThreadsPerBlock>>>(
      input_rows.data(),
      row_scales_device,
      rows,
      cols,
      tensor_scales->data(),
      packed->data(),
      block_scales->data(),
      packed_row_stride_bytes,
      layout.has_value() ? layout->padded_blocks_per_row : 0,
      matmul_bytes_per_row,
      matmul_block_scales != nullptr ? matmul_block_scales->data() : nullptr);
  return CheckCuda(cudaGetLastError());
}

bool ScaleRelu2PackRowsToNvfp4InPlace(
    const DeviceTensorFp32& input_rows,
    const float* row_scales_device,
    DeviceBuffer<std::uint8_t>& packed,
    DeviceBuffer<std::uint8_t>& block_scales,
    DeviceBuffer<std::uint8_t>* matmul_block_scales,
    DeviceBuffer<float>& tensor_scales,
    std::size_t packed_row_stride_bytes,
    cudaStream_t stream) {
  if (!input_rows.valid() ||
      input_rows.shape().size() != 2 ||
      row_scales_device == nullptr) {
    return false;
  }
  const std::size_t rows = input_rows.shape()[0];
  const std::size_t cols = input_rows.shape()[1];
  if (rows == 0 || cols == 0 || cols % kNvfp4BlockWidth != 0) {
    return false;
  }

  const std::size_t blocks_per_row = cols / kNvfp4BlockWidth;
  const std::size_t packed_bytes_per_row = cols / 2u;
  if (packed_row_stride_bytes == 0) {
    packed_row_stride_bytes = packed_bytes_per_row;
  }
  if (packed_row_stride_bytes < packed_bytes_per_row) {
    return false;
  }
  const std::size_t packed_bytes = rows * packed_row_stride_bytes;
  const std::size_t block_scale_bytes = rows * blocks_per_row;
  std::size_t matmul_bytes_per_row = 0;
  std::optional<Nvfp4ExecutionScaleLayout> layout;
  if (matmul_block_scales != nullptr) {
    matmul_bytes_per_row = ExecutionNvfp4ScaleBytes(1, cols);
    if (matmul_bytes_per_row == 0) {
      return false;
    }
    layout = BuildNvfp4ExecutionScaleLayout(1, cols);
    if (!layout.has_value()) {
      return false;
    }
  }

  if (!packed.valid() || packed.count() < packed_bytes ||
      !block_scales.valid() || block_scales.count() < block_scale_bytes ||
      !tensor_scales.valid() || tensor_scales.count() < rows) {
    return false;
  }
  if (matmul_block_scales != nullptr &&
      (!matmul_block_scales->valid() ||
       matmul_block_scales->count() < rows * matmul_bytes_per_row)) {
    return false;
  }

  FusedRelu2PackRowsToNvfp4Kernel<<<static_cast<unsigned int>(rows), kThreadsPerBlock,
                                    sizeof(float) * kThreadsPerBlock, stream>>>(
      input_rows.data(),
      row_scales_device,
      rows,
      cols,
      tensor_scales.data(),
      packed.data(),
      block_scales.data(),
      packed_row_stride_bytes,
      layout.has_value() ? layout->padded_blocks_per_row : 0,
      matmul_bytes_per_row,
      matmul_block_scales != nullptr ? matmul_block_scales->data() : nullptr);
  return CheckCuda(cudaGetLastError());
}

bool WeightedSumRowsFp32(
    const DeviceTensorFp32& input_rows,
    const float* row_scales_device,
    DeviceTensorFp32* output_row) {
  if (!input_rows.valid() ||
      input_rows.shape().size() != 2 ||
      row_scales_device == nullptr ||
      output_row == nullptr ||
      !HasSingleRowShape(*output_row) ||
      output_row->shape()[1] != input_rows.shape()[1]) {
    return false;
  }
  const std::size_t rows = input_rows.shape()[0];
  const std::size_t cols = input_rows.shape()[1];
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((cols + block.x - 1) / block.x));
  WeightedSumRowsKernel<<<grid, block>>>(
      input_rows.data(),
      row_scales_device,
      rows,
      cols,
      output_row->data());
  return CheckCuda(cudaGetLastError());
}

bool FusedRoutedUpProjPackedNvfp4SingleToken(
    const std::uint8_t* activation_packed,
    const std::uint8_t* activation_block_scales,
    const float* activation_tensor_scale,
    std::size_t input_cols,
    const DeviceBuffer<const void*>& weight_packed_ptrs,
    const DeviceBuffer<const void*>& weight_block_scale_ptrs,
    const DeviceBuffer<float>& weight_tensor_scales,
    DeviceTensorFp32* output_rows,
    cudaStream_t stream) {
  if (activation_packed == nullptr ||
      activation_block_scales == nullptr ||
      activation_tensor_scale == nullptr ||
      input_cols == 0 ||
      input_cols % kNvfp4BlockWidth != 0 ||
      !weight_packed_ptrs.valid() ||
      !weight_block_scale_ptrs.valid() ||
      !weight_tensor_scales.valid() ||
      weight_packed_ptrs.count() == 0 ||
      weight_packed_ptrs.count() != weight_block_scale_ptrs.count() ||
      weight_packed_ptrs.count() != weight_tensor_scales.count() ||
      output_rows == nullptr ||
      !output_rows->valid() ||
      output_rows->shape().size() != 2 ||
      output_rows->shape()[0] != weight_packed_ptrs.count()) {
    return false;
  }
  const std::size_t output_row_count = output_rows->shape()[1];
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>((output_row_count + block.x - 1u) / block.x),
      static_cast<unsigned int>(weight_packed_ptrs.count()));
  const std::size_t shared_bytes = kNvfp4BlockWidth * sizeof(float);
  FusedRoutedUpProjPackedNvfp4SingleTokenKernel<<<grid, block, shared_bytes, stream>>>(
      activation_packed,
      activation_block_scales,
      activation_tensor_scale,
      input_cols,
      weight_packed_ptrs.data(),
      weight_block_scale_ptrs.data(),
      weight_tensor_scales.data(),
      output_row_count,
      output_rows->data());
  return CheckCuda(cudaGetLastError());
}

bool FusedRoutedDownProjWeightedPackedNvfp4SingleToken(
    const std::uint8_t* activation_rows_packed,
    const std::uint8_t* activation_rows_block_scales,
    const DeviceBuffer<float>& activation_row_tensor_scales,
    const float* selection_weights_device,
    std::size_t input_cols,
    const DeviceBuffer<const void*>& weight_packed_ptrs,
    const DeviceBuffer<const void*>& weight_block_scale_ptrs,
    const DeviceBuffer<float>& weight_tensor_scales,
    DeviceTensorFp32* output_row,
    std::size_t activation_rows_packed_row_stride_bytes,
    cudaStream_t stream) {
  if (activation_rows_packed == nullptr ||
      activation_rows_block_scales == nullptr ||
      selection_weights_device == nullptr ||
      input_cols == 0 ||
      input_cols % kNvfp4BlockWidth != 0 ||
      !activation_row_tensor_scales.valid() ||
      !weight_packed_ptrs.valid() ||
      !weight_block_scale_ptrs.valid() ||
      !weight_tensor_scales.valid() ||
      activation_row_tensor_scales.count() == 0 ||
      activation_row_tensor_scales.count() != weight_packed_ptrs.count() ||
      weight_packed_ptrs.count() != weight_block_scale_ptrs.count() ||
      weight_packed_ptrs.count() != weight_tensor_scales.count() ||
      output_row == nullptr ||
      !output_row->valid() ||
      !HasSingleRowShape(*output_row)) {
    return false;
  }
  const std::size_t tight_packed_row_stride_bytes = input_cols / 2u;
  if (activation_rows_packed_row_stride_bytes != 0 &&
      activation_rows_packed_row_stride_bytes < tight_packed_row_stride_bytes) {
    return false;
  }
  const std::size_t output_row_count = output_row->shape()[1];
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((output_row_count + block.x - 1u) / block.x));
  const std::size_t shared_bytes = kNvfp4BlockWidth * sizeof(float);
  FusedRoutedDownProjWeightedPackedNvfp4SingleTokenKernel<<<grid, block, shared_bytes, stream>>>(
      activation_rows_packed,
      activation_rows_block_scales,
      activation_row_tensor_scales.data(),
      selection_weights_device,
      weight_packed_ptrs.count(),
      input_cols,
      activation_rows_packed_row_stride_bytes,
      weight_packed_ptrs.data(),
      weight_block_scale_ptrs.data(),
      weight_tensor_scales.data(),
      output_row_count,
      output_row->data());
  return CheckCuda(cudaGetLastError());
}

__global__ void FillPointerArrayKernel(void** array, void* value, std::size_t count) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    array[i] = value;
  }
}

__global__ void BuildStridedPointerArrayKernel(
    void** array, char* base, std::size_t stride_bytes, std::size_t count) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    array[i] = base + i * stride_bytes;
  }
}

__global__ void BuildIndexedStridedPointerArrayKernel(
    void** array,
    char* base,
    const std::int32_t* selected_indices,
    std::size_t stride_bytes,
    std::size_t count) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const std::int32_t expert_index = selected_indices[i];
  array[i] = expert_index < 0 ? nullptr : (base + static_cast<std::size_t>(expert_index) * stride_bytes);
}

__global__ void GatherIndexedFloatsKernel(
    const float* values,
    std::size_t value_count,
    const std::int32_t* selected_indices,
    std::size_t count,
    float* output) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const std::int32_t expert_index = selected_indices[i];
  if (expert_index < 0 || static_cast<std::size_t>(expert_index) >= value_count) {
    output[i] = 0.0f;
    return;
  }
  output[i] = values[static_cast<std::size_t>(expert_index)];
}

bool FillDevicePointerArray(void** device_array, void* value, std::size_t count) {
  if (device_array == nullptr || count == 0) return false;
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1) / block.x));
  FillPointerArrayKernel<<<grid, block>>>(device_array, value, count);
  return CheckCuda(cudaGetLastError());
}

bool BuildStridedDevicePointerArray(void** device_array, void* base, std::size_t stride_bytes, std::size_t count) {
  if (device_array == nullptr || base == nullptr || count == 0) return false;
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1) / block.x));
  BuildStridedPointerArrayKernel<<<grid, block>>>(device_array, reinterpret_cast<char*>(base), stride_bytes, count);
  return CheckCuda(cudaGetLastError());
}

bool BuildStridedDevicePointerArray(
    void** device_array,
    void* base,
    const std::int32_t* selected_indices_device,
    std::size_t stride_bytes,
    std::size_t count,
    cudaStream_t stream) {
  if (device_array == nullptr ||
      base == nullptr ||
      selected_indices_device == nullptr ||
      count == 0) {
    return false;
  }
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1) / block.x));
  BuildIndexedStridedPointerArrayKernel<<<grid, block, 0, stream>>>(
      device_array,
      reinterpret_cast<char*>(base),
      selected_indices_device,
      stride_bytes,
      count);
  return CheckCuda(cudaGetLastError());
}

bool GatherIndexedFloatsInPlace(
    const float* values_device,
    std::size_t value_count,
    const std::int32_t* selected_indices_device,
    std::size_t count,
    DeviceBuffer<float>& output,
    cudaStream_t stream) {
  if (values_device == nullptr ||
      selected_indices_device == nullptr ||
      count == 0 ||
      !output.valid() ||
      output.count() < count) {
    return false;
  }
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1) / block.x));
  GatherIndexedFloatsKernel<<<grid, block, 0, stream>>>(
      values_device,
      value_count,
      selected_indices_device,
      count,
      output.data());
  return CheckCuda(cudaGetLastError());
}

__global__ void ScaleRowsByTensorScaleKernel(
    float* data,
    const float* act_tensor_scale,
    const float* weight_tensor_scales,
    std::size_t cols,
    std::size_t row_count) {
  const std::size_t col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t row = static_cast<std::size_t>(blockIdx.y);
  if (col >= cols || row >= row_count) {
    return;
  }
  const float scale = (*act_tensor_scale) * weight_tensor_scales[row];
  data[row * cols + col] *= scale;
}

bool ScaleRowsByTensorScaleFp32(
    DeviceTensorFp32* data,
    const float* act_tensor_scale_device,
    const float* weight_tensor_scales_device,
    std::size_t row_count) {
  if (data == nullptr || !data->valid() ||
      act_tensor_scale_device == nullptr ||
      weight_tensor_scales_device == nullptr ||
      data->shape().size() != 2 ||
      data->shape()[0] < row_count ||
      row_count == 0) {
    return false;
  }
  const std::size_t cols = data->shape()[1];
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(
      static_cast<unsigned int>((cols + block.x - 1) / block.x),
      static_cast<unsigned int>(row_count));
  ScaleRowsByTensorScaleKernel<<<grid, block>>>(
      data->data(), act_tensor_scale_device, weight_tensor_scales_device, cols, row_count);
  return CheckCuda(cudaGetLastError());
}

__global__ void ScaleWeightedAccumulateRowsKernel(
    const float* data,
    const float* act_tensor_scales,
    const float* weight_tensor_scales,
    const float* routing_weights,
    std::size_t cols,
    std::size_t row_count,
    float* accumulator) {
  const std::size_t col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (col >= cols) {
    return;
  }
  float sum = 0.0f;
  for (std::size_t row = 0; row < row_count; ++row) {
    const float scale = act_tensor_scales[row] * weight_tensor_scales[row] * routing_weights[row];
    sum += data[row * cols + col] * scale;
  }
  accumulator[col] += sum;
}

bool ScaleWeightedAccumulateRowsFp32(
    const DeviceTensorFp32& data,
    const float* act_tensor_scales_device,
    const float* weight_tensor_scales_device,
    const float* routing_weights_device,
    std::size_t row_count,
    DeviceTensorFp32* accumulator) {
  if (!data.valid() || accumulator == nullptr || !accumulator->valid() ||
      act_tensor_scales_device == nullptr ||
      weight_tensor_scales_device == nullptr || routing_weights_device == nullptr ||
      data.shape().size() != 2 || accumulator->shape().size() != 2 ||
      data.shape()[0] < row_count || row_count == 0 ||
      data.shape()[1] != accumulator->shape()[1]) {
    return false;
  }
  const std::size_t cols = data.shape()[1];
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>((cols + block.x - 1) / block.x));
  ScaleWeightedAccumulateRowsKernel<<<grid, block>>>(
      data.data(), act_tensor_scales_device, weight_tensor_scales_device,
      routing_weights_device, cols, row_count, accumulator->data());
  return CheckCuda(cudaGetLastError());
}

}  // namespace nemotron
