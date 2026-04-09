#include "nemotron/device_nvfp4_matrix.h"

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <device_types.h>

#include <cmath>
#include <optional>
#include <utility>
#include <vector>

#include "nemotron/moe_launch_plan_device.h"
#include "nemotron/nvfp4_scale_layout.h"
#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
#include "routed_p5_tma_descriptor.cuh"
#endif

namespace nemotron {
namespace {

constexpr std::size_t kBlockWidth = 16;
constexpr std::size_t kScaleBlockTile = 4;
constexpr float kFp4MaxFinite = 6.0f;
constexpr float kFp8E4M3MaxFinite = 448.0f;
constexpr float kMinScale = 1.0f / 1024.0f;

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

bool HasCudaDevice() {
  static const bool kHasCudaDevice = []() {
    int device_count = 0;
    return CheckCuda(cudaGetDeviceCount(&device_count)) && device_count > 0;
  }();
  return kHasCudaDevice;
}

std::size_t PackedBytes(std::size_t rows, std::size_t cols) {
  return (rows * cols + 1u) / 2u;
}

std::size_t BlockScaleBytes(std::size_t rows, std::size_t cols) {
  return cols % kBlockWidth == 0 ? rows * (cols / kBlockWidth) : 0;
}

std::size_t RoundUp(std::size_t value, std::size_t alignment) {
  return alignment == 0 ? value : ((value + alignment - 1u) / alignment) * alignment;
}

std::size_t RowTile(Nvfp4ScaleLayout scale_layout) {
  switch (scale_layout) {
    case Nvfp4ScaleLayout::kSwizzled128x4:
      return 128;
    case Nvfp4ScaleLayout::kSwizzled8x4:
      return 8;
  }
  return 128;
}

std::size_t MatmulScaleBytes(
    std::size_t rows,
    std::size_t cols,
    Nvfp4ScaleLayout scale_layout) {
  return cols % kBlockWidth == 0
             ? RoundUp(rows, RowTile(scale_layout)) *
                   RoundUp(cols / kBlockWidth, kScaleBlockTile)
             : 0;
}

__device__ float ClampScale(float value) {
  if (!isfinite(value) || value < kMinScale) {
    return kMinScale;
  }
  return value;
}

std::optional<float> NormalizeFixedTensorScale(const Nvfp4PackOptions& options) {
  if (!options.fixed_tensor_scale.has_value()) {
    return std::nullopt;
  }
  const float value = *options.fixed_tensor_scale;
  if (!std::isfinite(value) || value <= 0.0f) {
    return std::nullopt;
  }
  if (value < kMinScale) {
    return kMinScale;
  }
  return value;
}

__global__ void ComputeGlobalMaxAbsKernel(
    const float* source,
    std::size_t numel,
    unsigned int* global_max_bits) {
  std::size_t index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (; index < numel; index += stride) {
    const float value = source[index];
    const float abs_value = fabsf(value);
    atomicMax(global_max_bits, __float_as_uint(abs_value));
  }
}

__global__ void WriteTensorScaleKernel(
    const unsigned int* global_max_bits,
    float* tensor_scale_data) {
  const float global_max_abs = __uint_as_float(*global_max_bits);
  float tensor_scale = 1.0f;
  if (global_max_abs > kFp4MaxFinite * kFp8E4M3MaxFinite) {
    tensor_scale = ClampScale(global_max_abs / (kFp4MaxFinite * kFp8E4M3MaxFinite));
  }
  *tensor_scale_data = tensor_scale;
}

__global__ void MultiplyTensorScalesKernel(
    const float* activation_tensor_scale_device,
    const float* weight_tensor_scale_device,
    float* alpha_device) {
  if (activation_tensor_scale_device == nullptr ||
      weight_tensor_scale_device == nullptr ||
      alpha_device == nullptr) {
    return;
  }
  *alpha_device = (*activation_tensor_scale_device) * (*weight_tensor_scale_device);
}

__device__ std::size_t ExecutionScaleOffset(
    std::size_t row,
    std::size_t block_col,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout) {
  const std::size_t num_k_tiles = padded_blocks_per_row / kScaleBlockTile;
  const std::size_t k_tile = block_col / kScaleBlockTile;
  const std::size_t inner_k = block_col & 3u;
  switch (scale_layout) {
    case Nvfp4ScaleLayout::kSwizzled128x4: {
      const std::size_t m_tile = row / 128u;
      const std::size_t outer_m = row & 31u;
      const std::size_t inner_m = (row >> 5u) & 3u;
      return ((((m_tile * num_k_tiles) + k_tile) << 9u) |
              (outer_m << 4u) |
              (inner_m << 2u) |
              inner_k);
    }
    case Nvfp4ScaleLayout::kSwizzled8x4: {
      const std::size_t m_tile = row / 8u;
      const std::size_t inner_m = row & 7u;
      return (((m_tile * num_k_tiles) + k_tile) << 5u) |
             (inner_m << 2u) |
             inner_k;
    }
  }
  return 0;
}

__global__ void PackAndSwizzleRowMajorFp32ToNvfp4Kernel(
    const float* source,
    std::size_t rows,
    std::size_t cols,
    std::size_t padded_rows,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout,
    const float* tensor_scale_data,
    std::uint8_t* packed,
    std::uint8_t* block_scales,
    std::uint8_t* matmul_scales) {
  const std::size_t blocks_per_row = cols / kBlockWidth;
  const std::size_t swizzled_index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const std::size_t total_scale_entries = padded_rows * padded_blocks_per_row;
  if (swizzled_index >= total_scale_entries) {
    return;
  }

  const std::size_t row = swizzled_index / padded_blocks_per_row;
  const std::size_t block = swizzled_index % padded_blocks_per_row;
  const std::size_t destination_offset =
      ExecutionScaleOffset(row, block, padded_blocks_per_row, scale_layout);
  if (row >= rows || block >= blocks_per_row) {
    matmul_scales[destination_offset] = 0u;
    return;
  }

  const std::size_t block_index = row * blocks_per_row + block;
  const std::size_t input_offset = row * cols + (block * kBlockWidth);
  const std::size_t packed_offset = row * (cols / 2u) + (block * (kBlockWidth / 2u));
  const float tensor_scale = *tensor_scale_data;

  float block_max_abs = 0.0f;
  for (std::size_t i = 0; i < kBlockWidth; ++i) {
    const float value = source[input_offset + i];
    const float abs_value = value >= 0.0f ? value : -value;
    if (abs_value > block_max_abs) {
      block_max_abs = abs_value;
    }
  }

  float block_scale = 1.0f;
  if (block_max_abs > 0.0f) {
    block_scale = ClampScale(block_max_abs / (kFp4MaxFinite * tensor_scale));
  }
  const std::uint8_t block_scale_fp8 = static_cast<std::uint8_t>(
      __nv_cvt_float_to_fp8(block_scale, __NV_SATFINITE, __NV_E4M3));
      
  block_scales[block_index] = block_scale_fp8;

  matmul_scales[destination_offset] = block_scale_fp8;

  const float scale = tensor_scale * block_scale;
  for (std::size_t i = 0; i < kBlockWidth; i += 2) {
    const float lhs = source[input_offset + i] / scale;
    const float rhs = source[input_offset + i + 1] / scale;
    const std::uint8_t lhs_fp4 = static_cast<std::uint8_t>(
                                     __nv_cvt_float_to_fp4(lhs, __NV_E2M1, cudaRoundNearest)) &
                                 0x0fu;
    const std::uint8_t rhs_fp4 = static_cast<std::uint8_t>(
                                     __nv_cvt_float_to_fp4(rhs, __NV_E2M1, cudaRoundNearest)) &
                                 0x0fu;
    packed[packed_offset + (i / 2u)] = static_cast<std::uint8_t>(lhs_fp4 | (rhs_fp4 << 4));
  }
}

__global__ void PackAndSwizzleRowMajorFp32ToNvfp4PerExpertKernel(
    const float* source,
    std::size_t rows,
    std::size_t cols,
    std::size_t padded_rows,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout,
    const int* expert_first_token_offsets,
    int n_experts,
    const float* expert_tensor_scales,
    std::uint8_t* packed,
    std::uint8_t* block_scales,
    std::uint8_t* matmul_scales) {
  auto find_expert_for_row = [expert_first_token_offsets, n_experts](std::size_t row) {
    for (int expert_index = 0; expert_index < n_experts; ++expert_index) {
      const int begin = expert_first_token_offsets[expert_index];
      const int end = expert_first_token_offsets[expert_index + 1];
      if (static_cast<int>(row) >= begin && static_cast<int>(row) < end) {
        return expert_index;
      }
    }
    return -1;
  };

  const std::size_t blocks_per_row = cols / kBlockWidth;
  const std::size_t swizzled_index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const std::size_t total_scale_entries = padded_rows * padded_blocks_per_row;
  if (swizzled_index >= total_scale_entries) {
    return;
  }

  const std::size_t row = swizzled_index / padded_blocks_per_row;
  const std::size_t block = swizzled_index % padded_blocks_per_row;
  const std::size_t destination_offset =
      ExecutionScaleOffset(row, block, padded_blocks_per_row, scale_layout);

  if (row >= rows || block >= blocks_per_row) {
    matmul_scales[destination_offset] = 0u;
    return;
  }

  const int total_active_rows = expert_first_token_offsets[n_experts];
  const std::size_t block_index = row * blocks_per_row + block;
  const std::size_t packed_offset = row * (cols / 2u) + (block * (kBlockWidth / 2u));
  if (static_cast<int>(row) >= total_active_rows) {
    block_scales[block_index] = 0u;
    matmul_scales[destination_offset] = 0u;
    for (std::size_t i = 0; i < (kBlockWidth / 2u); ++i) {
      packed[packed_offset + i] = 0u;
    }
    return;
  }

  float tensor_scale = 1.0f;
  if (expert_tensor_scales != nullptr) {
    const int expert_index = find_expert_for_row(row);
    if (expert_index >= 0) {
      tensor_scale = expert_tensor_scales[expert_index];
    }
  }

  const std::size_t input_offset = row * cols + (block * kBlockWidth);
  float block_max_abs = 0.0f;
  for (std::size_t i = 0; i < kBlockWidth; ++i) {
    const float value = source[input_offset + i];
    const float abs_value = value >= 0.0f ? value : -value;
    if (abs_value > block_max_abs) {
      block_max_abs = abs_value;
    }
  }

  float block_scale = 1.0f;
  if (block_max_abs > 0.0f) {
    block_scale = ClampScale(block_max_abs / (kFp4MaxFinite * tensor_scale));
  }
  const std::uint8_t block_scale_fp8 = static_cast<std::uint8_t>(
      __nv_cvt_float_to_fp8(block_scale, __NV_SATFINITE, __NV_E4M3));
  block_scales[block_index] = block_scale_fp8;
  matmul_scales[destination_offset] = block_scale_fp8;

  const float scale = tensor_scale * block_scale;
  for (std::size_t i = 0; i < kBlockWidth; i += 2) {
    const float lhs = source[input_offset + i] / scale;
    const float rhs = source[input_offset + i + 1] / scale;
    const std::uint8_t lhs_fp4 = static_cast<std::uint8_t>(
                                     __nv_cvt_float_to_fp4(lhs, __NV_E2M1, cudaRoundNearest)) &
                                 0x0fu;
    const std::uint8_t rhs_fp4 = static_cast<std::uint8_t>(
                                     __nv_cvt_float_to_fp4(rhs, __NV_E2M1, cudaRoundNearest)) &
                                 0x0fu;
    packed[packed_offset + (i / 2u)] = static_cast<std::uint8_t>(lhs_fp4 | (rhs_fp4 << 4));
  }
}

__global__ void PackAndSwizzleRowMajorBf16ToNvfp4PerExpertKernel(
    const __nv_bfloat16* source,
    std::size_t rows,
    std::size_t cols,
    std::size_t padded_rows,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout,
    const int* expert_first_token_offsets,
    int n_experts,
    const float* expert_tensor_scales,
    float* output_dequant_scales,
    std::uint8_t* packed,
    std::uint8_t* block_scales,
    std::uint8_t* matmul_scales) {
  auto find_expert_for_row = [expert_first_token_offsets, n_experts](std::size_t row) {
    for (int expert_index = 0; expert_index < n_experts; ++expert_index) {
      const int begin = expert_first_token_offsets[expert_index];
      const int end = expert_first_token_offsets[expert_index + 1];
      if (static_cast<int>(row) >= begin && static_cast<int>(row) < end) {
        return expert_index;
      }
    }
    return -1;
  };

  const std::size_t blocks_per_row = cols / kBlockWidth;
  const std::size_t swizzled_index =
      (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const std::size_t total_scale_entries = padded_rows * padded_blocks_per_row;
  if (swizzled_index >= total_scale_entries) {
    return;
  }

  const std::size_t row = swizzled_index / padded_blocks_per_row;
  const std::size_t block = swizzled_index % padded_blocks_per_row;
  const std::size_t destination_offset =
      ExecutionScaleOffset(row, block, padded_blocks_per_row, scale_layout);

  if (row >= rows || block >= blocks_per_row) {
    matmul_scales[destination_offset] = 0u;
    return;
  }

  const int total_active_rows = expert_first_token_offsets[n_experts];
  const std::size_t block_index = row * blocks_per_row + block;
  const std::size_t packed_offset = row * (cols / 2u) + (block * (kBlockWidth / 2u));
  if (static_cast<int>(row) >= total_active_rows) {
    block_scales[block_index] = 0u;
    matmul_scales[destination_offset] = 0u;
    if (output_dequant_scales != nullptr) {
      output_dequant_scales[block_index] = 0.0f;
    }
    for (std::size_t i = 0; i < (kBlockWidth / 2u); ++i) {
      packed[packed_offset + i] = 0u;
    }
    return;
  }

  float tensor_scale = 1.0f;
  if (expert_tensor_scales != nullptr) {
    const int expert_index = find_expert_for_row(row);
    if (expert_index >= 0) {
      tensor_scale = expert_tensor_scales[expert_index];
    }
  }

  const std::size_t input_offset = row * cols + (block * kBlockWidth);
  float block_max_abs = 0.0f;
  for (std::size_t i = 0; i < kBlockWidth; ++i) {
    const float value = __bfloat162float(source[input_offset + i]);
    const float activated = value > 0.0f ? value * value : 0.0f;
    if (activated > block_max_abs) {
      block_max_abs = activated;
    }
  }

  float block_scale = 1.0f;
  if (block_max_abs > 0.0f) {
    block_scale = ClampScale(block_max_abs / (kFp4MaxFinite * tensor_scale));
  }
  const std::uint8_t block_scale_fp8 = static_cast<std::uint8_t>(
      __nv_cvt_float_to_fp8(block_scale, __NV_SATFINITE, __NV_E4M3));
  block_scales[block_index] = block_scale_fp8;
  matmul_scales[destination_offset] = block_scale_fp8;

  const float scale = tensor_scale * block_scale;
  if (output_dequant_scales != nullptr) {
    output_dequant_scales[block_index] = scale;
  }
  for (std::size_t i = 0; i < kBlockWidth; i += 2) {
    const float lhs_value = __bfloat162float(source[input_offset + i]);
    const float rhs_value = __bfloat162float(source[input_offset + i + 1]);
    const float lhs = (lhs_value > 0.0f ? lhs_value * lhs_value : 0.0f) / scale;
    const float rhs = (rhs_value > 0.0f ? rhs_value * rhs_value : 0.0f) / scale;
    const std::uint8_t lhs_fp4 = static_cast<std::uint8_t>(
                                     __nv_cvt_float_to_fp4(lhs, __NV_E2M1, cudaRoundNearest)) &
                                 0x0fu;
    const std::uint8_t rhs_fp4 = static_cast<std::uint8_t>(
                                     __nv_cvt_float_to_fp4(rhs, __NV_E2M1, cudaRoundNearest)) &
                                 0x0fu;
    packed[packed_offset + (i / 2u)] = static_cast<std::uint8_t>(lhs_fp4 | (rhs_fp4 << 4));
  }
}

__global__ void GatherPackedRowsKernel(
    const std::uint8_t* source_packed,
    std::size_t source_rows,
    std::size_t cols,
    const int* source_row_indices,
    std::size_t row_count,
    std::uint8_t* output_packed) {
  const std::size_t pairs_per_row = cols / 2u;
  const std::size_t linear_index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const std::size_t total_pairs = row_count * pairs_per_row;
  if (linear_index >= total_pairs) {
    return;
  }

  const std::size_t row = linear_index / pairs_per_row;
  const std::size_t pair = linear_index % pairs_per_row;
  const int source_row_index = source_row_indices[row];
  if (source_row_index < 0 || static_cast<std::size_t>(source_row_index) >= source_rows) {
    return;
  }
  output_packed[(row * pairs_per_row) + pair] =
      source_packed[static_cast<std::size_t>(source_row_index) * pairs_per_row + pair];
}

__global__ void GatherBlockScalesRowsKernel(
    const std::uint8_t* source_block_scales,
    std::size_t source_rows,
    std::size_t cols,
    std::size_t padded_rows,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout,
    const int* source_row_indices,
    std::size_t row_count,
    std::uint8_t* output_block_scales,
    std::uint8_t* output_matmul_scales) {
  const std::size_t blocks_per_row = cols / kBlockWidth;
  const std::size_t swizzled_index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const std::size_t total_scale_entries = padded_rows * padded_blocks_per_row;
  if (swizzled_index >= total_scale_entries) {
    return;
  }

  const std::size_t row = swizzled_index / padded_blocks_per_row;
  const std::size_t block = swizzled_index % padded_blocks_per_row;
  const std::size_t destination_offset =
      ExecutionScaleOffset(row, block, padded_blocks_per_row, scale_layout);

  if (row >= row_count || block >= blocks_per_row) {
    output_matmul_scales[destination_offset] = 0u;
    return;
  }

  const int source_row_index = source_row_indices[row];
  if (source_row_index < 0 || static_cast<std::size_t>(source_row_index) >= source_rows) {
    output_matmul_scales[destination_offset] = 0u;
    return;
  }

  const std::size_t source_offset = static_cast<std::size_t>(source_row_index) * blocks_per_row + block;
  const std::size_t output_offset = row * blocks_per_row + block;
  const std::uint8_t block_scale = source_block_scales[source_offset];
  output_block_scales[output_offset] = block_scale;
  output_matmul_scales[destination_offset] = block_scale;
}

}  // namespace

struct DeviceNvfp4Matrix::Impl {
  Nvfp4ScaleLayout scale_layout = Nvfp4ScaleLayout::kSwizzled128x4;
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::size_t packed_nbytes = 0;
  std::size_t block_scales_nbytes = 0;
  std::size_t matmul_block_scales_nbytes = 0;
  std::size_t tensor_scale_nbytes = 0;
  std::uint8_t* packed_data = nullptr;
  std::uint8_t* block_scales_data = nullptr;
  std::uint8_t* matmul_block_scales_data = nullptr;
  std::uint8_t* tensor_scale_data = nullptr;
  const DeviceMoeLaunchPlan* cached_p5_tma_b_launch_plan = nullptr;
  std::size_t cached_p5_tma_b_build_epoch = 0;
  void* cached_p5_tma_b_descriptors = nullptr;
};

std::unique_ptr<DeviceNvfp4Matrix> DeviceNvfp4Matrix::Create(
    std::size_t rows,
    std::size_t cols,
    Nvfp4ScaleLayout scale_layout) {
  if (rows == 0 || cols == 0 || cols % kBlockWidth != 0 || !HasCudaDevice()) {
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  impl->scale_layout = scale_layout;
  impl->rows = rows;
  impl->cols = cols;
  impl->packed_nbytes = PackedBytes(rows, cols);
  impl->block_scales_nbytes = BlockScaleBytes(rows, cols);
  impl->matmul_block_scales_nbytes = MatmulScaleBytes(rows, cols, scale_layout);
  impl->tensor_scale_nbytes = sizeof(float);

  if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&impl->packed_data), impl->packed_nbytes)) ||
      !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&impl->block_scales_data), impl->block_scales_nbytes)) ||
      !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&impl->matmul_block_scales_data), impl->matmul_block_scales_nbytes)) ||
      !CheckCuda(cudaMalloc(reinterpret_cast<void**>(&impl->tensor_scale_data), impl->tensor_scale_nbytes))) {
    if (impl->tensor_scale_data != nullptr) {
      cudaFree(impl->tensor_scale_data);
    }
    if (impl->matmul_block_scales_data != nullptr) {
      cudaFree(impl->matmul_block_scales_data);
    }
    if (impl->block_scales_data != nullptr) {
      cudaFree(impl->block_scales_data);
    }
    if (impl->packed_data != nullptr) {
      cudaFree(impl->packed_data);
    }
    return nullptr;
  }

  return std::unique_ptr<DeviceNvfp4Matrix>(new DeviceNvfp4Matrix(std::move(impl)));
}

DeviceNvfp4Matrix::DeviceNvfp4Matrix(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

DeviceNvfp4Matrix::DeviceNvfp4Matrix(DeviceNvfp4Matrix&&) noexcept = default;

DeviceNvfp4Matrix& DeviceNvfp4Matrix::operator=(DeviceNvfp4Matrix&&) noexcept = default;

DeviceNvfp4Matrix::~DeviceNvfp4Matrix() {
  if (!impl_) {
    return;
  }
  if (impl_->tensor_scale_data != nullptr) {
    cudaFree(impl_->tensor_scale_data);
  }
  if (impl_->cached_p5_tma_b_descriptors != nullptr) {
    cudaFree(impl_->cached_p5_tma_b_descriptors);
  }
  if (impl_->matmul_block_scales_data != nullptr) {
    cudaFree(impl_->matmul_block_scales_data);
  }
  if (impl_->block_scales_data != nullptr) {
    cudaFree(impl_->block_scales_data);
  }
  if (impl_->packed_data != nullptr) {
    cudaFree(impl_->packed_data);
  }
}

bool DeviceNvfp4Matrix::valid() const {
  return impl_ != nullptr &&
         impl_->rows > 0 &&
         impl_->cols > 0 &&
         impl_->packed_data != nullptr &&
         impl_->block_scales_data != nullptr &&
         impl_->matmul_block_scales_data != nullptr &&
         impl_->tensor_scale_data != nullptr;
}

std::size_t DeviceNvfp4Matrix::rows() const {
  return impl_ ? impl_->rows : 0;
}

std::size_t DeviceNvfp4Matrix::cols() const {
  return impl_ ? impl_->cols : 0;
}

std::size_t DeviceNvfp4Matrix::packed_nbytes() const {
  return impl_ ? impl_->packed_nbytes : 0;
}

std::size_t DeviceNvfp4Matrix::block_scales_nbytes() const {
  return impl_ ? impl_->block_scales_nbytes : 0;
}

std::size_t DeviceNvfp4Matrix::matmul_block_scales_nbytes() const {
  return impl_ ? impl_->matmul_block_scales_nbytes : 0;
}

std::size_t DeviceNvfp4Matrix::tensor_scale_nbytes() const {
  return impl_ ? impl_->tensor_scale_nbytes : 0;
}

float DeviceNvfp4Matrix::host_tensor_scale() const {
  float host_tensor_scale = 0.0f;
  if (!CopyTensorScaleToHost(&host_tensor_scale)) {
    return 0.0f;
  }
  return host_tensor_scale;
}

const float* DeviceNvfp4Matrix::device_tensor_scale_ptr() const {
  return impl_ ? reinterpret_cast<const float*>(impl_->tensor_scale_data) : nullptr;
}

const float* DeviceNvfp4Matrix::effective_device_tensor_scale_ptr(
    const Nvfp4PackOptions& options) const {
  if (options.fixed_tensor_scale.has_value() && options.fixed_tensor_scale_device != nullptr) {
    return options.fixed_tensor_scale_device;
  }
  return device_tensor_scale_ptr();
}

const std::uint8_t* DeviceNvfp4Matrix::packed_data() const {
  return impl_ ? impl_->packed_data : nullptr;
}

const std::uint8_t* DeviceNvfp4Matrix::block_scales_data() const {
  return impl_ ? impl_->block_scales_data : nullptr;
}

const std::uint8_t* DeviceNvfp4Matrix::matmul_block_scales_data() const {
  return impl_ ? impl_->matmul_block_scales_data : nullptr;
}

const std::uint8_t* DeviceNvfp4Matrix::tensor_scale_data() const {
  return impl_ ? impl_->tensor_scale_data : nullptr;
}

const void* DeviceNvfp4Matrix::p5_tma_load_b_descriptors(
    const DeviceMoeLaunchPlan& launch_plan) const {
#if defined(NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE)
  if (!valid() ||
      cols() == 0 ||
      (cols() % 128u) != 0 ||
      launch_plan.exact_cta_count_host() <= 0 ||
      launch_plan.cta_row_starts_host() == nullptr ||
      launch_plan.cta_valid_rows_host() == nullptr) {
    return nullptr;
  }

  if (impl_->cached_p5_tma_b_launch_plan == &launch_plan &&
      impl_->cached_p5_tma_b_build_epoch == launch_plan.build_epoch() &&
      impl_->cached_p5_tma_b_descriptors != nullptr) {
    return impl_->cached_p5_tma_b_descriptors;
  }

  std::vector<routed_p5_tma::P5TmaLoadB> host_descriptors;
  host_descriptors.reserve(static_cast<std::size_t>(launch_plan.exact_cta_count_host()));
  const std::size_t packed_row_bytes = cols() / 2u;
  const int32_t input_cols = static_cast<int32_t>(cols());
  const int64_t input_cols_stride = static_cast<int64_t>(cols());
  for (int cta_index = 0; cta_index < launch_plan.exact_cta_count_host(); ++cta_index) {
    const int row_start = launch_plan.cta_row_starts_host()[cta_index];
    const int valid_rows = launch_plan.cta_valid_rows_host()[cta_index];
    if (row_start < 0 ||
        valid_rows <= 0 ||
        static_cast<std::size_t>(row_start) + static_cast<std::size_t>(valid_rows) > rows()) {
      return nullptr;
    }

    const auto* cta_packed =
        packed_data() + static_cast<std::size_t>(row_start) * packed_row_bytes;
    auto tensor_b = cute::make_tensor(
        cute::make_gmem_ptr(cute::recast_ptr<routed_p5_tma::ElementAB>(cta_packed)),
        cute::make_layout(
            cute::make_shape(static_cast<int32_t>(valid_rows), input_cols, int32_t{1}),
            cute::make_stride(
                input_cols_stride,
                cute::Int<1>{},
                static_cast<int64_t>(valid_rows) * input_cols_stride)));
    host_descriptors.push_back(routed_p5_tma::MakeP5TmaLoadB(tensor_b));
  }

  void* device_descriptors = nullptr;
  const std::size_t descriptor_bytes =
      sizeof(routed_p5_tma::P5TmaLoadB) * host_descriptors.size();
  if (descriptor_bytes == 0 ||
      !CheckCuda(cudaMalloc(&device_descriptors, descriptor_bytes)) ||
      !CheckCuda(cudaMemcpy(
          device_descriptors,
          host_descriptors.data(),
          descriptor_bytes,
          cudaMemcpyHostToDevice))) {
    if (device_descriptors != nullptr) {
      cudaFree(device_descriptors);
    }
    return nullptr;
  }

  if (impl_->cached_p5_tma_b_descriptors != nullptr) {
    cudaFree(impl_->cached_p5_tma_b_descriptors);
  }
  impl_->cached_p5_tma_b_launch_plan = &launch_plan;
  impl_->cached_p5_tma_b_build_epoch = launch_plan.build_epoch();
  impl_->cached_p5_tma_b_descriptors = device_descriptors;
  return impl_->cached_p5_tma_b_descriptors;
#else
  (void) launch_plan;
  return nullptr;
#endif
}

Nvfp4ScaleLayout DeviceNvfp4Matrix::scale_layout() const {
  return impl_ ? impl_->scale_layout : Nvfp4ScaleLayout::kSwizzled128x4;
}

bool DeviceNvfp4Matrix::PackInto(
    const DeviceTensorFp32& source,
    const Nvfp4PackOptions& options) {
  if (!valid() || !source.valid() || source.shape().size() != 2) {
    return false;
  }

  const std::size_t rows = source.shape()[0];
  const std::size_t cols = source.shape()[1];
  const Nvfp4ScaleLayout scale_layout =
      ResolveActivationNvfp4ScaleLayout(rows, options.execution_scale_layout);
  if (rows > impl_->rows ||
      cols != impl_->cols ||
      scale_layout != impl_->scale_layout) {
    return false;
  }

  const auto layout = BuildNvfp4ExecutionScaleLayout(rows, cols, scale_layout);
  if (!layout.has_value()) {
    return false;
  }

  const std::optional<float> fixed_tensor_scale = NormalizeFixedTensorScale(options);
  if (options.fixed_tensor_scale.has_value() && !fixed_tensor_scale.has_value()) {
    return false;
  }

  auto* tensor_scale_data = reinterpret_cast<float*>(impl_->tensor_scale_data);
  const float* tensor_scale_data_for_pack = tensor_scale_data;
  if (fixed_tensor_scale.has_value()) {
    if (options.fixed_tensor_scale_device != nullptr) {
      tensor_scale_data_for_pack = options.fixed_tensor_scale_device;
    } else {
      const float host_tensor_scale = *fixed_tensor_scale;
      if (!CheckCuda(cudaMemcpy(
              tensor_scale_data,
              &host_tensor_scale,
              sizeof(host_tensor_scale),
              cudaMemcpyHostToDevice))) {
        return false;
      }
    }
  } else {
    auto* global_max_bits = reinterpret_cast<unsigned int*>(tensor_scale_data);
    if (!CheckCuda(cudaMemset(global_max_bits, 0, impl_->tensor_scale_nbytes))) {
      return false;
    }

    const std::size_t numel = rows * cols;
    constexpr std::size_t kReductionThreadsPerBlock = 128;
    const std::size_t reduction_grid_size =
        (numel + kReductionThreadsPerBlock - 1u) / kReductionThreadsPerBlock;
    ComputeGlobalMaxAbsKernel<<<static_cast<unsigned int>(reduction_grid_size), kReductionThreadsPerBlock>>>(
        source.data(),
        numel,
        global_max_bits);
    if (!CheckCuda(cudaGetLastError())) {
      return false;
    }

    WriteTensorScaleKernel<<<1, 1>>>(
        global_max_bits,
        tensor_scale_data);
    if (!CheckCuda(cudaGetLastError())) {
      return false;
    }
  }

  constexpr std::size_t kThreadsPerBlock = 128;
  const std::size_t total_scale_entries = layout->padded_rows * layout->padded_blocks_per_row;
  const std::size_t grid_size = (total_scale_entries + kThreadsPerBlock - 1u) / kThreadsPerBlock;

  PackAndSwizzleRowMajorFp32ToNvfp4Kernel<<<static_cast<unsigned int>(grid_size), kThreadsPerBlock>>>(
      source.data(),
      rows,
      cols,
      layout->padded_rows,
      layout->padded_blocks_per_row,
      scale_layout,
      tensor_scale_data_for_pack,
      impl_->packed_data,
      impl_->block_scales_data,
      impl_->matmul_block_scales_data);
  if (!CheckCuda(cudaGetLastError())) {
    return false;
  }

  return true;
}

bool DeviceNvfp4Matrix::CopyPackedToHost(std::vector<std::uint8_t>* output) const {
  if (!valid() || output == nullptr) {
    return false;
  }
  output->assign(packed_nbytes(), 0u);
  return CheckCuda(cudaMemcpy(output->data(), packed_data(), packed_nbytes(), cudaMemcpyDeviceToHost));
}

bool DeviceNvfp4Matrix::CopyBlockScalesToHost(std::vector<std::uint8_t>* output) const {
  if (!valid() || output == nullptr) {
    return false;
  }
  output->assign(block_scales_nbytes(), 0u);
  return CheckCuda(
      cudaMemcpy(output->data(), block_scales_data(), block_scales_nbytes(), cudaMemcpyDeviceToHost));
}

bool DeviceNvfp4Matrix::CopyMatmulBlockScalesToHost(std::vector<std::uint8_t>* output) const {
  if (!valid() || output == nullptr) {
    return false;
  }
  output->assign(matmul_block_scales_nbytes(), 0u);
  return CheckCuda(
      cudaMemcpy(
          output->data(),
          matmul_block_scales_data(),
          matmul_block_scales_nbytes(),
          cudaMemcpyDeviceToHost));
}

bool DeviceNvfp4Matrix::CopyTensorScaleToHost(float* output) const {
  if (!valid() || output == nullptr) {
    return false;
  }
  return CheckCuda(
      cudaMemcpy(output, impl_->tensor_scale_data, sizeof(*output), cudaMemcpyDeviceToHost));
}

std::unique_ptr<DeviceNvfp4Matrix> PackDeviceRowMajorFp32ToNvfp4(
    const DeviceTensorFp32& source,
    const Nvfp4PackOptions& options) {
  if (!source.valid() || source.shape().size() != 2) {
    return nullptr;
  }

  const std::size_t rows = source.shape()[0];
  const std::size_t cols = source.shape()[1];
  const Nvfp4ScaleLayout scale_layout =
      ResolveActivationNvfp4ScaleLayout(rows, options.execution_scale_layout);
  auto packed = DeviceNvfp4Matrix::Create(rows, cols, scale_layout);
  if (!packed || !packed->valid() || !packed->PackInto(source, options)) {
    return nullptr;
  }

  return packed;
}

bool PackDeviceRowMajorFp32ToNvfp4PerExpert(
    const float* source,
    std::size_t rows,
    std::size_t cols,
    const int* expert_first_token_offsets,
    std::size_t n_experts,
    const float* expert_tensor_scales,
    DeviceNvfp4Matrix* output) {
  if (source == nullptr ||
      expert_first_token_offsets == nullptr ||
      n_experts == 0 ||
      output == nullptr ||
      !output->valid() ||
      rows == 0 ||
      cols == 0 ||
      cols % kBlockWidth != 0 ||
      rows > output->rows() ||
      cols != output->cols()) {
    return false;
  }

  if (!CheckCuda(cudaMemset(
          const_cast<std::uint8_t*>(output->packed_data()),
          0,
          output->packed_nbytes())) ||
      !CheckCuda(cudaMemset(
          const_cast<std::uint8_t*>(output->block_scales_data()),
          0,
          output->block_scales_nbytes())) ||
      !CheckCuda(cudaMemset(
          const_cast<std::uint8_t*>(output->matmul_block_scales_data()),
          0,
          output->matmul_block_scales_nbytes()))) {
    return false;
  }

  const float one = 1.0f;
  if (!CheckCuda(cudaMemcpy(
          const_cast<std::uint8_t*>(output->tensor_scale_data()),
          &one,
          sizeof(one),
          cudaMemcpyHostToDevice))) {
    return false;
  }

  const std::size_t padded_rows = RoundUp(output->rows(), RowTile(output->scale_layout()));
  const std::size_t padded_blocks_per_row = RoundUp(cols / kBlockWidth, kScaleBlockTile);
  const std::size_t total_scale_entries = padded_rows * padded_blocks_per_row;
  const dim3 block(256);
  const dim3 grid(static_cast<unsigned int>((total_scale_entries + block.x - 1u) / block.x));
  PackAndSwizzleRowMajorFp32ToNvfp4PerExpertKernel<<<grid, block>>>(
      source,
      rows,
      cols,
      padded_rows,
      padded_blocks_per_row,
      output->scale_layout(),
      expert_first_token_offsets,
      static_cast<int>(n_experts),
      expert_tensor_scales,
      const_cast<std::uint8_t*>(output->packed_data()),
      const_cast<std::uint8_t*>(output->block_scales_data()),
      const_cast<std::uint8_t*>(output->matmul_block_scales_data()));
  return CheckCuda(cudaGetLastError());
}

bool PackDeviceRowMajorBf16ToNvfp4PerExpert(
    const __nv_bfloat16* source,
    std::size_t rows,
    std::size_t cols,
    const int* expert_first_token_offsets,
    std::size_t n_experts,
    const float* expert_tensor_scales,
    float* output_dequant_scales,
    DeviceNvfp4Matrix* output) {
  if (source == nullptr ||
      expert_first_token_offsets == nullptr ||
      n_experts == 0 ||
      output == nullptr ||
      !output->valid() ||
      rows == 0 ||
      cols == 0 ||
      cols % kBlockWidth != 0 ||
      rows > output->rows() ||
      cols != output->cols()) {
    return false;
  }

  if (!CheckCuda(cudaMemset(
          const_cast<std::uint8_t*>(output->packed_data()),
          0,
          output->packed_nbytes())) ||
      !CheckCuda(cudaMemset(
          const_cast<std::uint8_t*>(output->block_scales_data()),
          0,
          output->block_scales_nbytes())) ||
      !CheckCuda(cudaMemset(
          const_cast<std::uint8_t*>(output->matmul_block_scales_data()),
          0,
          output->matmul_block_scales_nbytes())) ||
      (output_dequant_scales != nullptr &&
       !CheckCuda(cudaMemset(
           output_dequant_scales,
           0,
           rows * (cols / kBlockWidth) * sizeof(float))))) {
    return false;
  }

  const float one = 1.0f;
  if (!CheckCuda(cudaMemcpy(
          const_cast<std::uint8_t*>(output->tensor_scale_data()),
          &one,
          sizeof(one),
          cudaMemcpyHostToDevice))) {
    return false;
  }

  const std::size_t padded_rows = RoundUp(output->rows(), RowTile(output->scale_layout()));
  const std::size_t padded_blocks_per_row = RoundUp(cols / kBlockWidth, kScaleBlockTile);
  const std::size_t total_scale_entries = padded_rows * padded_blocks_per_row;
  const dim3 block(256);
  const dim3 grid(static_cast<unsigned int>((total_scale_entries + block.x - 1u) / block.x));
  PackAndSwizzleRowMajorBf16ToNvfp4PerExpertKernel<<<grid, block>>>(
      source,
      rows,
      cols,
      padded_rows,
      padded_blocks_per_row,
      output->scale_layout(),
      expert_first_token_offsets,
      static_cast<int>(n_experts),
      expert_tensor_scales,
      output_dequant_scales,
      const_cast<std::uint8_t*>(output->packed_data()),
      const_cast<std::uint8_t*>(output->block_scales_data()),
      const_cast<std::uint8_t*>(output->matmul_block_scales_data()));
  return CheckCuda(cudaGetLastError());
}

bool GatherDeviceNvfp4Rows(
    const DeviceNvfp4Matrix& source,
    const int* source_row_indices,
    std::size_t row_count,
    DeviceNvfp4Matrix* output) {
  if (!source.valid() ||
      source_row_indices == nullptr ||
      row_count == 0 ||
      output == nullptr ||
      !output->valid() ||
      source.cols() != output->cols() ||
      source.scale_layout() != output->scale_layout() ||
      row_count > output->rows()) {
    return false;
  }

  if (!CheckCuda(cudaMemset(
          const_cast<std::uint8_t*>(output->packed_data()),
          0,
          output->packed_nbytes())) ||
      !CheckCuda(cudaMemset(
          const_cast<std::uint8_t*>(output->block_scales_data()),
          0,
          output->block_scales_nbytes())) ||
      !CheckCuda(cudaMemset(
          const_cast<std::uint8_t*>(output->matmul_block_scales_data()),
          0,
          output->matmul_block_scales_nbytes())) ||
      !CheckCuda(cudaMemcpy(
          const_cast<std::uint8_t*>(output->tensor_scale_data()),
          source.tensor_scale_data(),
          source.tensor_scale_nbytes(),
          cudaMemcpyDeviceToDevice))) {
    return false;
  }

  const std::size_t cols = source.cols();
  const std::size_t pairs_per_row = cols / 2u;
  const std::size_t total_pairs = row_count * pairs_per_row;
  const dim3 block(256);
  if (total_pairs > 0) {
    const dim3 grid(static_cast<unsigned int>((total_pairs + block.x - 1u) / block.x));
    GatherPackedRowsKernel<<<grid, block>>>(
        source.packed_data(),
        source.rows(),
        cols,
        source_row_indices,
        row_count,
        const_cast<std::uint8_t*>(output->packed_data()));
    if (!CheckCuda(cudaGetLastError())) {
      return false;
    }
  }

  const std::size_t padded_rows = RoundUp(output->rows(), RowTile(output->scale_layout()));
  const std::size_t padded_blocks_per_row = RoundUp(cols / kBlockWidth, kScaleBlockTile);
  const std::size_t total_scale_entries = padded_rows * padded_blocks_per_row;
  const dim3 scale_grid(static_cast<unsigned int>((total_scale_entries + block.x - 1u) / block.x));
  GatherBlockScalesRowsKernel<<<scale_grid, block>>>(
      source.block_scales_data(),
      source.rows(),
      cols,
      padded_rows,
      padded_blocks_per_row,
      output->scale_layout(),
      source_row_indices,
      row_count,
      const_cast<std::uint8_t*>(output->block_scales_data()),
      const_cast<std::uint8_t*>(output->matmul_block_scales_data()));
  return CheckCuda(cudaGetLastError());
}

bool MultiplyDeviceTensorScales(
    const float* activation_tensor_scale_device,
    const float* weight_tensor_scale_device,
    float* alpha_device) {
  if (activation_tensor_scale_device == nullptr ||
      weight_tensor_scale_device == nullptr ||
      alpha_device == nullptr) {
    return false;
  }
  MultiplyTensorScalesKernel<<<1, 1>>>(
      activation_tensor_scale_device,
      weight_tensor_scale_device,
      alpha_device);
  return CheckCuda(cudaGetLastError());
}

}  // namespace nemotron
