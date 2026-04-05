#include "nemotron/device_nvfp4_matrix.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <device_types.h>

#include <utility>

#include "nemotron/nvfp4_scale_layout.h"

namespace nemotron {
namespace {

constexpr std::size_t kBlockWidth = 16;
constexpr std::size_t kScaleRowTile = 128;
constexpr std::size_t kScaleBlockTile = 4;
constexpr float kFp4MaxFinite = 6.0f;
constexpr float kFp8E4M3MaxFinite = 448.0f;
constexpr float kMinScale = 1.0f / 1024.0f;

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
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

std::size_t MatmulScaleBytes(std::size_t rows, std::size_t cols) {
  return cols % kBlockWidth == 0 ? RoundUp(rows, kScaleRowTile) * RoundUp(cols / kBlockWidth, kScaleBlockTile)
                                 : 0;
}

__device__ float ClampScale(float value) {
  if (!isfinite(value) || value < kMinScale) {
    return kMinScale;
  }
  return value;
}

__device__ float NormalizeFixedTensorScaleDevice(float value) {
  if (!isfinite(value) || value <= 0.0f) {
    return kMinScale;
  }
  return value;
}

__device__ float LoadSourceValue(const float* source, std::size_t index) {
  return source[index];
}

__device__ float LoadSourceValue(const __nv_bfloat16* source, std::size_t index) {
  return __bfloat162float(source[index]);
}

std::optional<float> NormalizeFixedTensorScale(const Nvfp4PackOptions& options) {
  if (!options.fixed_tensor_scale.has_value()) {
    return std::nullopt;
  }
  const float value = *options.fixed_tensor_scale;
  if (!std::isfinite(value) || value <= 0.0f) {
    return std::nullopt;
  }
  return value;
}

template <typename SourceT>
__global__ void ComputeGlobalMaxAbsKernel(
    const SourceT* source,
    std::size_t numel,
    unsigned int* global_max_bits) {
  std::size_t index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (; index < numel; index += stride) {
    const float value = LoadSourceValue(source, index);
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

__global__ void WriteFixedTensorScaleKernel(
    float* dst,
    float value) {
  *dst = value;
}

template <typename SourceT>
__global__ void PackRowMajorToNvfp4Kernel(
    const SourceT* source,
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
    const float value = LoadSourceValue(source, input_offset + i);
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
    const float lhs = LoadSourceValue(source, input_offset + i) / scale;
    const float rhs = LoadSourceValue(source, input_offset + i + 1) / scale;
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

__global__ void SwizzleBlockScalesForMatmulKernel(
    const std::uint8_t* row_major_scales,
    std::size_t rows,
    std::size_t logical_blocks_per_row,
    std::size_t padded_blocks_per_row,
    std::uint8_t* matmul_scales) {
  const std::size_t scale_index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const std::size_t total_scales = rows * logical_blocks_per_row;
  if (scale_index >= total_scales) {
    return;
  }

  const std::size_t row = scale_index / logical_blocks_per_row;
  const std::size_t block_col = scale_index % logical_blocks_per_row;
  const std::size_t destination_offset =
      ExecutionScaleOffset(row, block_col, padded_blocks_per_row);
  matmul_scales[destination_offset] = row_major_scales[scale_index];
}

// Fused kernel for M=1 single-row packing. Replaces ComputeGlobalMaxAbsKernel +
// WriteTensorScaleKernel + PackRowMajorFp32ToNvfp4Kernel +
// SwizzleBlockScalesForMatmulKernel in a single launch of one thread block.
template <typename SourceT>
__global__ void FusedPackSingleRowToNvfp4Kernel(
    const SourceT* source,
    std::size_t cols,
    float fixed_tensor_scale,
    std::uint8_t* packed,
    std::uint8_t* block_scales,
    std::size_t matmul_scales_nbytes,
    std::size_t padded_blocks_per_row,
    std::uint8_t* matmul_scales,
    float* tensor_scale_data) {
  extern __shared__ float smem[];
  const std::size_t tid = threadIdx.x;
  const std::size_t blocks_per_row = cols / kBlockWidth;

  float tensor_scale;
  if (fixed_tensor_scale > 0.0f) {
    tensor_scale = fixed_tensor_scale;
    if (tid == 0) {
      *tensor_scale_data = tensor_scale;
    }
  } else {
    float local_max = 0.0f;
    for (std::size_t i = tid; i < cols; i += blockDim.x) {
      const float val = fabsf(LoadSourceValue(source, i));
      if (val > local_max) local_max = val;
    }
    smem[tid] = local_max;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s && smem[tid + s] > smem[tid]) {
        smem[tid] = smem[tid + s];
      }
      __syncthreads();
    }
    const float global_max_abs = smem[0];
    tensor_scale = 1.0f;
    if (global_max_abs > kFp4MaxFinite * kFp8E4M3MaxFinite) {
      tensor_scale = ClampScale(global_max_abs / (kFp4MaxFinite * kFp8E4M3MaxFinite));
    }
    if (tid == 0) {
      *tensor_scale_data = tensor_scale;
      smem[0] = tensor_scale;
    }
    __syncthreads();
    tensor_scale = smem[0];
  }

  for (std::size_t i = tid; i < matmul_scales_nbytes; i += blockDim.x) {
    matmul_scales[i] = 0;
  }
  __syncthreads();

  for (std::size_t block_index = tid; block_index < blocks_per_row;
       block_index += blockDim.x) {
    const std::size_t input_offset = block_index * kBlockWidth;
    const std::size_t packed_offset = block_index * (kBlockWidth / 2u);

    float block_max_abs = 0.0f;
    for (std::size_t i = 0; i < kBlockWidth; ++i) {
      const float abs_value = fabsf(LoadSourceValue(source, input_offset + i));
      if (abs_value > block_max_abs) block_max_abs = abs_value;
    }

    float block_scale = 1.0f;
    if (block_max_abs > 0.0f) {
      block_scale = ClampScale(block_max_abs / (kFp4MaxFinite * tensor_scale));
    }
    const std::uint8_t block_scale_fp8 = static_cast<std::uint8_t>(
        __nv_cvt_float_to_fp8(block_scale, __NV_SATFINITE, __NV_E4M3));
    block_scales[block_index] = block_scale_fp8;

    const float scale = tensor_scale * block_scale;
    for (std::size_t i = 0; i < kBlockWidth; i += 2) {
      const float lhs = LoadSourceValue(source, input_offset + i) / scale;
      const float rhs = LoadSourceValue(source, input_offset + i + 1) / scale;
      const std::uint8_t lhs_fp4 = static_cast<std::uint8_t>(
                                       __nv_cvt_float_to_fp4(lhs, __NV_E2M1, cudaRoundNearest)) &
                                   0x0fu;
      const std::uint8_t rhs_fp4 = static_cast<std::uint8_t>(
                                       __nv_cvt_float_to_fp4(rhs, __NV_E2M1, cudaRoundNearest)) &
                                   0x0fu;
      packed[packed_offset + (i / 2u)] = static_cast<std::uint8_t>(lhs_fp4 | (rhs_fp4 << 4));
    }

    const std::size_t dest = ExecutionScaleOffset(0, block_index, padded_blocks_per_row);
    matmul_scales[dest] = block_scale_fp8;
  }
}

template <typename SourceT>
__global__ void PackLatentPerSelectedExpertToNvfp4Kernel(
    const SourceT* source_row,
    std::size_t selected_expert_count,
    std::size_t cols,
    const float* selected_expert_input_scales,
    std::size_t packed_row_bytes,
    std::size_t blocks_per_row,
    std::size_t padded_blocks_per_row,
    std::size_t matmul_bytes_per_row,
    std::uint8_t* packed,
    std::uint8_t* block_scales,
    std::uint8_t* matmul_scales,
    float* tensor_scales) {
  const std::size_t expert_row = static_cast<std::size_t>(blockIdx.x);
  if (expert_row >= selected_expert_count) {
    return;
  }

  const std::size_t tid = threadIdx.x;
  const float raw_input_scale = selected_expert_input_scales[expert_row];
  const float tensor_scale = NormalizeFixedTensorScaleDevice(
      (raw_input_scale > 0.0f && isfinite(raw_input_scale)) ? raw_input_scale
                                                            : kMinScale);
  std::uint8_t* row_packed = packed + (expert_row * packed_row_bytes);
  std::uint8_t* row_block_scales = block_scales + (expert_row * blocks_per_row);
  std::uint8_t* row_matmul_scales = matmul_scales + (expert_row * matmul_bytes_per_row);

  if (tid == 0) {
    tensor_scales[expert_row] = tensor_scale;
  }

  for (std::size_t i = tid; i < matmul_bytes_per_row; i += blockDim.x) {
    row_matmul_scales[i] = 0;
  }
  __syncthreads();

  for (std::size_t block_index = tid; block_index < blocks_per_row;
       block_index += blockDim.x) {
    const std::size_t input_offset = block_index * kBlockWidth;
    const std::size_t packed_offset = block_index * (kBlockWidth / 2u);

    float block_max_abs = 0.0f;
    for (std::size_t i = 0; i < kBlockWidth; ++i) {
      const float abs_value = fabsf(LoadSourceValue(source_row, input_offset + i));
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
    row_block_scales[block_index] = block_scale_fp8;

    const float scale = tensor_scale * block_scale;
    for (std::size_t i = 0; i < kBlockWidth; i += 2) {
      const float lhs = LoadSourceValue(source_row, input_offset + i) / scale;
      const float rhs = LoadSourceValue(source_row, input_offset + i + 1) / scale;
      const std::uint8_t lhs_fp4 = static_cast<std::uint8_t>(
                                       __nv_cvt_float_to_fp4(lhs, __NV_E2M1, cudaRoundNearest)) &
                                   0x0fu;
      const std::uint8_t rhs_fp4 = static_cast<std::uint8_t>(
                                       __nv_cvt_float_to_fp4(rhs, __NV_E2M1, cudaRoundNearest)) &
                                   0x0fu;
      row_packed[packed_offset + (i / 2u)] = static_cast<std::uint8_t>(lhs_fp4 | (rhs_fp4 << 4));
    }

    const std::size_t dest = ExecutionScaleOffset(0, block_index, padded_blocks_per_row);
    row_matmul_scales[dest] = block_scale_fp8;
  }
}

}  // namespace

struct DeviceNvfp4Matrix::Impl {
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

std::unique_ptr<DeviceNvfp4Matrix> DeviceNvfp4Matrix::Create(std::size_t rows, std::size_t cols) {
  if (rows == 0 || cols == 0 || cols % kBlockWidth != 0) {
    return nullptr;
  }

  int device_count = 0;
  if (!CheckCuda(cudaGetDeviceCount(&device_count)) || device_count <= 0) {
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  impl->rows = rows;
  impl->cols = cols;
  impl->packed_nbytes = PackedBytes(rows, cols);
  impl->block_scales_nbytes = BlockScaleBytes(rows, cols);
  impl->matmul_block_scales_nbytes = MatmulScaleBytes(rows, cols);
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
  auto packed = DeviceNvfp4Matrix::Create(rows, cols);
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
    unsigned int* global_max_bits = nullptr;
    if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&global_max_bits), sizeof(unsigned int)))) {
      return nullptr;
    }
    if (!CheckCuda(cudaMemset(global_max_bits, 0, sizeof(unsigned int)))) {
      cudaFree(global_max_bits);
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
      cudaFree(global_max_bits);
      return nullptr;
    }

    WriteTensorScaleKernel<<<1, 1>>>(
        global_max_bits,
        reinterpret_cast<float*>(const_cast<std::uint8_t*>(packed->tensor_scale_data())));
    if (!CheckCuda(cudaGetLastError())) {
      cudaFree(global_max_bits);
      return nullptr;
    }
    cudaFree(global_max_bits);
  }

  constexpr std::size_t kThreadsPerBlock = 128;
  const std::size_t total_blocks = rows * (cols / kBlockWidth);
  const std::size_t grid_size = (total_blocks + kThreadsPerBlock - 1u) / kThreadsPerBlock;
  PackRowMajorToNvfp4Kernel<<<static_cast<unsigned int>(grid_size), kThreadsPerBlock>>>(
      source.data(),
      rows,
      cols,
      reinterpret_cast<const float*>(packed->tensor_scale_data()),
      const_cast<std::uint8_t*>(packed->packed_data()),
      const_cast<std::uint8_t*>(packed->block_scales_data()));
  if (!CheckCuda(cudaGetLastError())) {
    return nullptr;
  }

  const auto layout = BuildNvfp4ExecutionScaleLayout(rows, cols);
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
      const_cast<std::uint8_t*>(packed->matmul_block_scales_data()));
  if (!CheckCuda(cudaGetLastError())) {
    return nullptr;
  }

  return packed;
}

std::unique_ptr<DeviceNvfp4Matrix> PackDeviceRowMajorBf16ToNvfp4(
    const DeviceTensorBf16& source,
    const Nvfp4PackOptions& options) {
  if (!source.valid() || source.shape().size() != 2) {
    return nullptr;
  }

  const std::size_t rows = source.shape()[0];
  const std::size_t cols = source.shape()[1];
  auto packed = DeviceNvfp4Matrix::Create(rows, cols);
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
    unsigned int* global_max_bits = nullptr;
    if (!CheckCuda(cudaMalloc(reinterpret_cast<void**>(&global_max_bits), sizeof(unsigned int)))) {
      return nullptr;
    }
    if (!CheckCuda(cudaMemset(global_max_bits, 0, sizeof(unsigned int)))) {
      cudaFree(global_max_bits);
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
      cudaFree(global_max_bits);
      return nullptr;
    }

    WriteTensorScaleKernel<<<1, 1>>>(
        global_max_bits,
        reinterpret_cast<float*>(const_cast<std::uint8_t*>(packed->tensor_scale_data())));
    if (!CheckCuda(cudaGetLastError())) {
      cudaFree(global_max_bits);
      return nullptr;
    }
    cudaFree(global_max_bits);
  }

  constexpr std::size_t kThreadsPerBlock = 128;
  const std::size_t total_blocks = rows * (cols / kBlockWidth);
  const std::size_t grid_size = (total_blocks + kThreadsPerBlock - 1u) / kThreadsPerBlock;
  PackRowMajorToNvfp4Kernel<<<static_cast<unsigned int>(grid_size), kThreadsPerBlock>>>(
      source.data(),
      rows,
      cols,
      reinterpret_cast<const float*>(packed->tensor_scale_data()),
      const_cast<std::uint8_t*>(packed->packed_data()),
      const_cast<std::uint8_t*>(packed->block_scales_data()));
  if (!CheckCuda(cudaGetLastError())) {
    return nullptr;
  }

  const auto layout = BuildNvfp4ExecutionScaleLayout(rows, cols);
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
      const_cast<std::uint8_t*>(packed->matmul_block_scales_data()));
  if (!CheckCuda(cudaGetLastError())) {
    return nullptr;
  }

  return packed;
}

bool PackDeviceRowMajorFp32ToNvfp4InPlace(
    const DeviceTensorFp32& source,
    const Nvfp4PackOptions& options,
    std::uint8_t* packed_data,
    std::uint8_t* block_scales_data,
    std::uint8_t* matmul_block_scales_data,
    float* tensor_scale_data,
    unsigned int* global_max_bits_scratch,
    cudaStream_t stream) {
  if (!source.valid() || source.shape().size() != 2 ||
      packed_data == nullptr || block_scales_data == nullptr ||
      matmul_block_scales_data == nullptr || tensor_scale_data == nullptr) {
    return false;
  }

  const std::size_t rows = source.shape()[0];
  const std::size_t cols = source.shape()[1];
  if (rows == 0 || cols == 0 || cols % kBlockWidth != 0) {
    return false;
  }

  const std::optional<float> fixed_tensor_scale = NormalizeFixedTensorScale(options);
  if (options.fixed_tensor_scale.has_value() && !fixed_tensor_scale.has_value()) {
    return false;
  }

  // Single-row fast path: one fused kernel replaces 4 separate launches + 2 memsets.
  if (rows == 1) {
    const auto layout = BuildNvfp4ExecutionScaleLayout(rows, cols);
    if (!layout.has_value()) {
      return false;
    }
    constexpr std::size_t kFusedThreads = 256;
    const float fixed_scale = fixed_tensor_scale.has_value() ? *fixed_tensor_scale : -1.0f;
    FusedPackSingleRowToNvfp4Kernel<<<1, kFusedThreads,
                                      kFusedThreads * sizeof(float), stream>>>(
        source.data(),
        cols,
        fixed_scale,
        packed_data,
        block_scales_data,
        MatmulScaleBytes(rows, cols),
        layout->padded_blocks_per_row,
        matmul_block_scales_data,
        tensor_scale_data);
    return CheckCuda(cudaGetLastError());
  }

  if (fixed_tensor_scale.has_value()) {
    WriteFixedTensorScaleKernel<<<1, 1, 0, stream>>>(tensor_scale_data, *fixed_tensor_scale);
    if (!CheckCuda(cudaGetLastError())) {
      return false;
    }
  } else {
    if (global_max_bits_scratch == nullptr) {
      return false;
    }
    if (!CheckCuda(cudaMemsetAsync(global_max_bits_scratch, 0, sizeof(unsigned int), stream))) {
      return false;
    }

    const std::size_t numel = rows * cols;
    constexpr std::size_t kThreadsPerBlock = 128;
    const std::size_t reduction_grid_size = (numel + kThreadsPerBlock - 1u) / kThreadsPerBlock;
    ComputeGlobalMaxAbsKernel<<<static_cast<unsigned int>(reduction_grid_size),
                                kThreadsPerBlock, 0, stream>>>(
        source.data(),
        numel,
        global_max_bits_scratch);
    if (!CheckCuda(cudaGetLastError())) {
      return false;
    }

    WriteTensorScaleKernel<<<1, 1, 0, stream>>>(
        global_max_bits_scratch,
        tensor_scale_data);
    if (!CheckCuda(cudaGetLastError())) {
      return false;
    }
  }

  constexpr std::size_t kThreadsPerBlock = 128;
  const std::size_t total_blocks = rows * (cols / kBlockWidth);
  const std::size_t grid_size = (total_blocks + kThreadsPerBlock - 1u) / kThreadsPerBlock;
  PackRowMajorToNvfp4Kernel<<<static_cast<unsigned int>(grid_size),
                              kThreadsPerBlock, 0, stream>>>(
      source.data(),
      rows,
      cols,
      tensor_scale_data,
      packed_data,
      block_scales_data);
  if (!CheckCuda(cudaGetLastError())) {
    return false;
  }

  const auto layout = BuildNvfp4ExecutionScaleLayout(rows, cols);
  if (!layout.has_value()) {
    return false;
  }
  if (!CheckCuda(cudaMemsetAsync(
          matmul_block_scales_data, 0, MatmulScaleBytes(rows, cols), stream))) {
    return false;
  }
  SwizzleBlockScalesForMatmulKernel<<<static_cast<unsigned int>(grid_size),
                                      kThreadsPerBlock, 0, stream>>>(
      block_scales_data,
      rows,
      layout->logical_blocks_per_row,
      layout->padded_blocks_per_row,
      matmul_block_scales_data);
  return CheckCuda(cudaGetLastError());
}

bool PackDeviceRowMajorBf16ToNvfp4InPlace(
    const DeviceTensorBf16& source,
    const Nvfp4PackOptions& options,
    std::uint8_t* packed_data,
    std::uint8_t* block_scales_data,
    std::uint8_t* matmul_block_scales_data,
    float* tensor_scale_data,
    unsigned int* global_max_bits_scratch,
    cudaStream_t stream) {
  if (!source.valid() || source.shape().size() != 2 ||
      packed_data == nullptr || block_scales_data == nullptr ||
      matmul_block_scales_data == nullptr || tensor_scale_data == nullptr) {
    return false;
  }

  const std::size_t rows = source.shape()[0];
  const std::size_t cols = source.shape()[1];
  if (rows == 0 || cols == 0 || cols % kBlockWidth != 0) {
    return false;
  }

  const std::optional<float> fixed_tensor_scale = NormalizeFixedTensorScale(options);
  if (options.fixed_tensor_scale.has_value() && !fixed_tensor_scale.has_value()) {
    return false;
  }

  if (rows == 1) {
    const auto layout = BuildNvfp4ExecutionScaleLayout(rows, cols);
    if (!layout.has_value()) {
      return false;
    }
    constexpr std::size_t kFusedThreads = 256;
    const float fixed_scale = fixed_tensor_scale.has_value() ? *fixed_tensor_scale : -1.0f;
    FusedPackSingleRowToNvfp4Kernel<<<1, kFusedThreads,
                                      kFusedThreads * sizeof(float), stream>>>(
        source.data(),
        cols,
        fixed_scale,
        packed_data,
        block_scales_data,
        MatmulScaleBytes(rows, cols),
        layout->padded_blocks_per_row,
        matmul_block_scales_data,
        tensor_scale_data);
    return CheckCuda(cudaGetLastError());
  }

  if (fixed_tensor_scale.has_value()) {
    WriteFixedTensorScaleKernel<<<1, 1, 0, stream>>>(tensor_scale_data, *fixed_tensor_scale);
    if (!CheckCuda(cudaGetLastError())) {
      return false;
    }
  } else {
    if (global_max_bits_scratch == nullptr) {
      return false;
    }
    if (!CheckCuda(cudaMemsetAsync(global_max_bits_scratch, 0, sizeof(unsigned int), stream))) {
      return false;
    }

    const std::size_t numel = rows * cols;
    constexpr std::size_t kThreadsPerBlock = 128;
    const std::size_t reduction_grid_size = (numel + kThreadsPerBlock - 1u) / kThreadsPerBlock;
    ComputeGlobalMaxAbsKernel<<<static_cast<unsigned int>(reduction_grid_size),
                                kThreadsPerBlock, 0, stream>>>(
        source.data(),
        numel,
        global_max_bits_scratch);
    if (!CheckCuda(cudaGetLastError())) {
      return false;
    }

    WriteTensorScaleKernel<<<1, 1, 0, stream>>>(
        global_max_bits_scratch,
        tensor_scale_data);
    if (!CheckCuda(cudaGetLastError())) {
      return false;
    }
  }

  constexpr std::size_t kThreadsPerBlock = 128;
  const std::size_t total_blocks = rows * (cols / kBlockWidth);
  const std::size_t grid_size = (total_blocks + kThreadsPerBlock - 1u) / kThreadsPerBlock;
  PackRowMajorToNvfp4Kernel<<<static_cast<unsigned int>(grid_size),
                              kThreadsPerBlock, 0, stream>>>(
      source.data(),
      rows,
      cols,
      tensor_scale_data,
      packed_data,
      block_scales_data);
  if (!CheckCuda(cudaGetLastError())) {
    return false;
  }

  const auto layout = BuildNvfp4ExecutionScaleLayout(rows, cols);
  if (!layout.has_value()) {
    return false;
  }
  if (!CheckCuda(cudaMemsetAsync(
          matmul_block_scales_data, 0, MatmulScaleBytes(rows, cols), stream))) {
    return false;
  }
  SwizzleBlockScalesForMatmulKernel<<<static_cast<unsigned int>(grid_size),
                                      kThreadsPerBlock, 0, stream>>>(
      block_scales_data,
      rows,
      layout->logical_blocks_per_row,
      layout->padded_blocks_per_row,
      matmul_block_scales_data);
  return CheckCuda(cudaGetLastError());
}

template <typename DeviceTensorT>
bool PackLatentPerSelectedExpertToNvfp4InPlaceImpl(
    const DeviceTensorT& source_row,
    std::size_t selected_expert_count,
    const float* selected_expert_input_scales,
    std::uint8_t* packed_data,
    std::uint8_t* block_scales_data,
    std::uint8_t* matmul_block_scales_data,
    float* tensor_scale_data,
    unsigned int* global_max_bits_scratch,
    cudaStream_t stream) {
  (void)global_max_bits_scratch;
  if (!source_row.valid() ||
      source_row.shape().size() != 2 ||
      source_row.shape()[0] != 1 ||
      selected_expert_count == 0 ||
      selected_expert_input_scales == nullptr ||
      packed_data == nullptr ||
      block_scales_data == nullptr ||
      matmul_block_scales_data == nullptr ||
      tensor_scale_data == nullptr) {
    return false;
  }

  const std::size_t cols = source_row.shape()[1];
  if (cols == 0 || cols % kBlockWidth != 0) {
    return false;
  }

  const auto layout = BuildNvfp4ExecutionScaleLayout(1, cols);
  if (!layout.has_value()) {
    return false;
  }

  constexpr std::size_t kThreadsPerBlock = 256;
  PackLatentPerSelectedExpertToNvfp4Kernel<<<
      static_cast<unsigned int>(selected_expert_count),
      kThreadsPerBlock,
      0,
      stream>>>(
      source_row.data(),
      selected_expert_count,
      cols,
      selected_expert_input_scales,
      PackedBytes(1, cols),
      cols / kBlockWidth,
      layout->padded_blocks_per_row,
      MatmulScaleBytes(1, cols),
      packed_data,
      block_scales_data,
      matmul_block_scales_data,
      tensor_scale_data);
  return CheckCuda(cudaGetLastError());
}

bool PackLatentPerSelectedExpertToNvfp4InPlace(
    const DeviceTensorFp32& source_row,
    std::size_t selected_expert_count,
    const float* selected_expert_input_scales,
    std::uint8_t* packed_data,
    std::uint8_t* block_scales_data,
    std::uint8_t* matmul_block_scales_data,
    float* tensor_scale_data,
    unsigned int* global_max_bits_scratch,
    cudaStream_t stream) {
  return PackLatentPerSelectedExpertToNvfp4InPlaceImpl(
      source_row,
      selected_expert_count,
      selected_expert_input_scales,
      packed_data,
      block_scales_data,
      matmul_block_scales_data,
      tensor_scale_data,
      global_max_bits_scratch,
      stream);
}

bool PackLatentPerSelectedExpertToNvfp4InPlace(
    const DeviceTensorBf16& source_row,
    std::size_t selected_expert_count,
    const float* selected_expert_input_scales,
    std::uint8_t* packed_data,
    std::uint8_t* block_scales_data,
    std::uint8_t* matmul_block_scales_data,
    float* tensor_scale_data,
    unsigned int* global_max_bits_scratch,
    cudaStream_t stream) {
  return PackLatentPerSelectedExpertToNvfp4InPlaceImpl(
      source_row,
      selected_expert_count,
      selected_expert_input_scales,
      packed_data,
      block_scales_data,
      matmul_block_scales_data,
      tensor_scale_data,
      global_max_bits_scratch,
      stream);
}

}  // namespace nemotron
