#include "nemotron/device_nvfp4_matrix.h"

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <device_types.h>

#include <utility>

#include "nemotron/nvfp4_scale_layout.h"

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

__global__ void PackRowMajorFp32ToNvfp4Kernel(
    const float* source,
    std::size_t rows,
    std::size_t cols,
    const float* tensor_scale_data,
    std::uint8_t* packed,
    std::uint8_t* block_scales) {
  const std::size_t blocks_per_row = cols / kBlockWidth;
  const std::size_t block_index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const std::size_t total_blocks = rows * blocks_per_row;
  if (block_index >= total_blocks) {
    return;
  }

  const std::size_t row = block_index / blocks_per_row;
  const std::size_t block = block_index % blocks_per_row;
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
  block_scales[block_index] = static_cast<std::uint8_t>(
      __nv_cvt_float_to_fp8(block_scale, __NV_SATFINITE, __NV_E4M3));

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

__global__ void SwizzleBlockScalesForMatmulKernel(
    const std::uint8_t* row_major_scales,
    std::size_t rows,
    std::size_t logical_blocks_per_row,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout,
    std::uint8_t* matmul_scales) {
  const std::size_t scale_index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const std::size_t total_scales = rows * logical_blocks_per_row;
  if (scale_index >= total_scales) {
    return;
  }

  const std::size_t row = scale_index / logical_blocks_per_row;
  const std::size_t block_col = scale_index % logical_blocks_per_row;
  const std::size_t destination_offset =
      ExecutionScaleOffset(row, block_col, padded_blocks_per_row, scale_layout);
  matmul_scales[destination_offset] = row_major_scales[scale_index];
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

Nvfp4ScaleLayout DeviceNvfp4Matrix::scale_layout() const {
  return impl_ ? impl_->scale_layout : Nvfp4ScaleLayout::kSwizzled128x4;
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
  return CheckCuda(cudaMemcpy(output, tensor_scale_data(), sizeof(float), cudaMemcpyDeviceToHost));
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
  if (!packed || !packed->valid()) {
    return nullptr;
  }

  const std::optional<float> fixed_tensor_scale = NormalizeFixedTensorScale(options);
  if (options.fixed_tensor_scale.has_value() && !fixed_tensor_scale.has_value()) {
    return nullptr;
  }

  if (fixed_tensor_scale.has_value()) {
    const float host_tensor_scale = *fixed_tensor_scale;
    if (!CheckCuda(
            cudaMemcpy(
                const_cast<std::uint8_t*>(packed->tensor_scale_data()),
                &host_tensor_scale,
                sizeof(host_tensor_scale),
                cudaMemcpyHostToDevice))) {
      return nullptr;
    }
  } else {
    auto* tensor_scale_data =
        reinterpret_cast<float*>(const_cast<std::uint8_t*>(packed->tensor_scale_data()));
    auto* global_max_bits = reinterpret_cast<unsigned int*>(tensor_scale_data);
    if (!CheckCuda(cudaMemset(global_max_bits, 0, packed->tensor_scale_nbytes()))) {
      return nullptr;
    }
    const std::size_t numel = rows * cols;
    constexpr std::size_t kThreadsPerBlock = 128;
    const std::size_t reduction_grid_size = (numel + kThreadsPerBlock - 1u) / kThreadsPerBlock;
    ComputeGlobalMaxAbsKernel<<<static_cast<unsigned int>(reduction_grid_size), kThreadsPerBlock>>>(
        source.data(),
        numel,
        global_max_bits);
    if (!CheckCuda(cudaGetLastError())) {
      return nullptr;
    }

    WriteTensorScaleKernel<<<1, 1>>>(
        global_max_bits,
        tensor_scale_data);
    if (!CheckCuda(cudaGetLastError())) {
      return nullptr;
    }
  }

  constexpr std::size_t kThreadsPerBlock = 128;
  const std::size_t total_blocks = rows * (cols / kBlockWidth);
  const std::size_t grid_size = (total_blocks + kThreadsPerBlock - 1u) / kThreadsPerBlock;
  PackRowMajorFp32ToNvfp4Kernel<<<static_cast<unsigned int>(grid_size), kThreadsPerBlock>>>(
      source.data(),
      rows,
      cols,
      reinterpret_cast<const float*>(packed->tensor_scale_data()),
      const_cast<std::uint8_t*>(packed->packed_data()),
      const_cast<std::uint8_t*>(packed->block_scales_data()));
  if (!CheckCuda(cudaGetLastError())) {
    return nullptr;
  }

  const auto layout = BuildNvfp4ExecutionScaleLayout(rows, cols, scale_layout);
  if (!layout.has_value()) {
    return nullptr;
  }
  if (!CheckCuda(cudaMemset(
          const_cast<std::uint8_t*>(packed->matmul_block_scales_data()),
          0,
          packed->matmul_block_scales_nbytes()))) {
    return nullptr;
  }
  SwizzleBlockScalesForMatmulKernel<<<static_cast<unsigned int>(grid_size), kThreadsPerBlock>>>(
      packed->block_scales_data(),
      rows,
      layout->logical_blocks_per_row,
      layout->padded_blocks_per_row,
      scale_layout,
      const_cast<std::uint8_t*>(packed->matmul_block_scales_data()));
  if (!CheckCuda(cudaGetLastError())) {
    return nullptr;
  }

  return packed;
}

}  // namespace nemotron
