#include "nemotron/nvfp4_packing.h"

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <device_types.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace nemotron {
namespace {

constexpr std::size_t kBlockWidth = 16;
constexpr float kFp4MaxFinite = 6.0f;
constexpr float kFp8E4M3MaxFinite = 448.0f;
constexpr float kMinScale = 1.0f / 1024.0f;

float ClampScale(float value) {
  if (!std::isfinite(value) || value < kMinScale) {
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
  return ClampScale(value);
}

std::uint8_t EncodeFp4(float value) {
  return static_cast<std::uint8_t>(
             __nv_cvt_float_to_fp4(value, __NV_E2M1, cudaRoundNearest)) &
         0x0fu;
}

std::uint8_t EncodeFp8Scale(float value) {
  return static_cast<std::uint8_t>(
      __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

}  // namespace

bool HostNvfp4Matrix::valid() const {
  return rows > 0 &&
         cols > 0 &&
         cols % kBlockWidth == 0 &&
         !packed.empty() &&
         !block_scales.empty();
}

const std::uint8_t* HostNvfp4Matrix::packed_data() const {
  return packed.empty() ? nullptr : packed.data();
}

const std::uint8_t* HostNvfp4Matrix::block_scales_data() const {
  return block_scales.empty() ? nullptr : block_scales.data();
}

const std::uint8_t* HostNvfp4Matrix::tensor_scale_data() const {
  return reinterpret_cast<const std::uint8_t*>(&tensor_scale);
}

std::size_t HostNvfp4Matrix::packed_nbytes() const {
  return packed.size();
}

std::size_t HostNvfp4Matrix::block_scales_nbytes() const {
  return block_scales.size();
}

std::size_t HostNvfp4Matrix::tensor_scale_nbytes() const {
  return sizeof(tensor_scale);
}

std::size_t PackedFp4Bytes(std::size_t rows, std::size_t cols) {
  return (rows * cols + 1u) / 2u;
}

std::size_t RowMajorNvfp4ScaleBytes(std::size_t rows, std::size_t cols) {
  return cols % kBlockWidth == 0 ? rows * (cols / kBlockWidth) : 0;
}

std::optional<HostNvfp4Matrix> PackRowMajorFp32ToNvfp4(
    const float* data,
    std::size_t rows,
    std::size_t cols,
    const Nvfp4PackOptions& options) {
  if (data == nullptr || rows == 0 || cols == 0 || cols % kBlockWidth != 0) {
    return std::nullopt;
  }

  const std::optional<float> fixed_tensor_scale = NormalizeFixedTensorScale(options);
  if (options.fixed_tensor_scale.has_value() && !fixed_tensor_scale.has_value()) {
    return std::nullopt;
  }

  float global_max_abs = 0.0f;
  for (std::size_t i = 0; i < rows * cols; ++i) {
    const float value = data[i];
    if (!std::isfinite(value)) {
      return std::nullopt;
    }
    global_max_abs = std::max(global_max_abs, std::fabs(value));
  }

  HostNvfp4Matrix matrix;
  matrix.rows = rows;
  matrix.cols = cols;
  matrix.packed.resize(PackedFp4Bytes(rows, cols), 0u);
  matrix.block_scales.resize(RowMajorNvfp4ScaleBytes(rows, cols), 0u);

  if (fixed_tensor_scale.has_value()) {
    matrix.tensor_scale = *fixed_tensor_scale;
  } else if (global_max_abs > kFp4MaxFinite * kFp8E4M3MaxFinite) {
    matrix.tensor_scale =
        ClampScale(global_max_abs / (kFp4MaxFinite * kFp8E4M3MaxFinite));
  } else {
    matrix.tensor_scale = 1.0f;
  }

  std::size_t packed_index = 0;
  std::size_t scale_index = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    const float* row_data = data + (row * cols);
    for (std::size_t block = 0; block < cols / kBlockWidth; ++block) {
      const std::size_t block_offset = block * kBlockWidth;
      float block_max_abs = 0.0f;
      for (std::size_t i = 0; i < kBlockWidth; ++i) {
        block_max_abs = std::max(block_max_abs, std::fabs(row_data[block_offset + i]));
      }

      float block_scale = 1.0f;
      if (block_max_abs > 0.0f) {
        block_scale = ClampScale(block_max_abs / (kFp4MaxFinite * matrix.tensor_scale));
      }
      matrix.block_scales[scale_index++] = EncodeFp8Scale(block_scale);

      const float scale = matrix.tensor_scale * block_scale;
      for (std::size_t i = 0; i < kBlockWidth; i += 2) {
        const float lhs = row_data[block_offset + i] / scale;
        const float rhs = row_data[block_offset + i + 1] / scale;
        const std::uint8_t lhs_fp4 = EncodeFp4(lhs);
        const std::uint8_t rhs_fp4 = EncodeFp4(rhs);
        matrix.packed[packed_index++] = static_cast<std::uint8_t>(lhs_fp4 | (rhs_fp4 << 4));
      }
    }
  }

  return matrix;
}

}  // namespace nemotron
