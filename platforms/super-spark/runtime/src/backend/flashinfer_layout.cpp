#include "nemotron/flashinfer_layout.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#include <cuda_fp8.h>

namespace nemotron {
namespace {

constexpr std::array<std::size_t, 16> kSrcToDstBlk16RowMap = {
    0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15,
};

constexpr std::array<std::size_t, 32> kSrcToDstBlk32RowMap = {
    0,  8, 16, 24, 1,  9, 17, 25, 2,  10, 18, 26, 3,  11, 19, 27,
    4, 12, 20, 28, 5, 13, 21, 29, 6,  14, 22, 30, 7, 15, 23, 31,
};

constexpr std::size_t kNvfp4BlockWidth = 16;

std::size_t RoundUp(std::size_t value, std::size_t multiple) {
  if (multiple == 0) {
    return value;
  }
  const std::size_t remainder = value % multiple;
  return remainder == 0 ? value : (value + multiple - remainder);
}

}  // namespace

std::size_t GetFlashInferShuffleBlockSize(std::size_t epilogue_tile_m) {
  return epilogue_tile_m % 128 == 0 ? 32 : 16;
}

std::vector<std::size_t> BuildFlashInferShuffleMatrixARowIndices(
    std::size_t rows,
    std::size_t epilogue_tile_m) {
  const std::size_t block_size = GetFlashInferShuffleBlockSize(epilogue_tile_m);
  if (rows == 0 || rows % block_size != 0) {
    return {};
  }

  std::vector<std::size_t> row_indices(rows, 0);
  for (std::size_t old_row = 0; old_row < rows; ++old_row) {
    const std::size_t block_index = old_row / block_size;
    const std::size_t row_in_block = old_row % block_size;
    const std::size_t mapped_row_in_block =
        block_size == 16 ? kSrcToDstBlk16RowMap[row_in_block]
                         : kSrcToDstBlk32RowMap[row_in_block];
    const std::size_t new_row = block_index * block_size + mapped_row_in_block;
    row_indices[new_row] = old_row;
  }
  return row_indices;
}

bool FlashInferShuffleMatrixA(
    const std::uint8_t* input,
    std::size_t rows,
    std::size_t cols,
    std::size_t epilogue_tile_m,
    std::vector<std::uint8_t>* output) {
  if (input == nullptr || output == nullptr || rows == 0 || cols == 0) {
    return false;
  }
  const auto row_indices = BuildFlashInferShuffleMatrixARowIndices(rows, epilogue_tile_m);
  if (row_indices.empty()) {
    return false;
  }
  output->assign(rows * cols, 0);
  for (std::size_t new_row = 0; new_row < rows; ++new_row) {
    const std::size_t old_row = row_indices[new_row];
    std::memcpy(
        output->data() + (new_row * cols),
        input + (old_row * cols),
        cols);
  }
  return true;
}

bool FlashInferShuffleMatrixSfALinear(
    const std::uint8_t* input,
    std::size_t rows,
    std::size_t cols,
    std::size_t epilogue_tile_m,
    std::vector<std::uint8_t>* output) {
  if (cols == 0 || cols % 4 != 0 || rows % 128 != 0) {
    return false;
  }
  return FlashInferShuffleMatrixA(input, rows, cols, epilogue_tile_m, output);
}

bool FlashInferInterleaveBlockScales128x4(
    const std::uint8_t* input,
    std::size_t rows,
    std::size_t cols,
    std::vector<std::uint8_t>* output) {
  if (input == nullptr || output == nullptr || rows == 0 || cols == 0) {
    return false;
  }

  const std::size_t padded_rows = RoundUp(rows, 128);
  const std::size_t padded_cols = RoundUp(cols, 4);
  const std::size_t num_k_tiles = padded_cols / 4;
  output->assign(padded_rows * padded_cols, 0);

  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      const std::size_t inner_k = col % 4;
      const std::size_t inner_m = (row % 128) / 32;
      const std::size_t outer_m = row % 32;
      const std::size_t k_tile = col / 4;
      const std::size_t m_tile = row / 128;
      const std::size_t offset =
          ((((m_tile * num_k_tiles) + k_tile) * 32 + outer_m) * 4 + inner_m) * 4 + inner_k;
      (*output)[offset] = input[row * cols + col];
    }
  }
  return true;
}

bool FlashInferShuffleMatrixSfA(
    const std::uint8_t* input,
    std::size_t rows,
    std::size_t cols,
    std::size_t epilogue_tile_m,
    std::vector<std::uint8_t>* output) {
  if (output == nullptr) {
    return false;
  }
  std::vector<std::uint8_t> shuffled_linear;
  if (!FlashInferShuffleMatrixSfALinear(
          input,
          rows,
          cols,
          epilogue_tile_m,
          &shuffled_linear)) {
    return false;
  }
  return FlashInferInterleaveBlockScales128x4(
      shuffled_linear.data(),
      rows,
      cols,
      output);
}

bool FlashInferConvertToBlockLayout(
    const std::uint8_t* input,
    std::size_t rows,
    std::size_t cols,
    std::size_t block_k,
    std::vector<std::uint8_t>* output) {
  if (input == nullptr || output == nullptr || rows == 0 || cols == 0 ||
      block_k == 0 || cols % block_k != 0) {
    return false;
  }

  output->assign(rows * cols, 0);
  const std::size_t block_count = cols / block_k;
  std::size_t output_offset = 0;
  for (std::size_t block_index = 0; block_index < block_count; ++block_index) {
    const std::size_t block_col = block_index * block_k;
    for (std::size_t row = 0; row < rows; ++row) {
      std::memcpy(
          output->data() + output_offset,
          input + (row * cols) + block_col,
          block_k);
      output_offset += block_k;
    }
  }
  return true;
}

bool FlashInferPrepareShuffledBlockMajorWeight(
    const std::uint8_t* input,
    std::size_t rows,
    std::size_t cols,
    std::size_t epilogue_tile_m,
    std::size_t block_k,
    std::vector<std::uint8_t>* output) {
  std::vector<std::uint8_t> shuffled;
  if (!FlashInferShuffleMatrixA(input, rows, cols, epilogue_tile_m, &shuffled)) {
    return false;
  }
  return FlashInferConvertToBlockLayout(
      shuffled.data(),
      rows,
      cols,
      block_k,
      output);
}

std::optional<FlashInferPreparedNvfp4WeightHost> PrepareFlashInferNvfp4WeightHost(
    const GemmDescriptor& descriptor,
    std::optional<float> tensor_scale_override,
    std::size_t epilogue_tile_m) {
  if (descriptor.kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled ||
      descriptor.packed_data == nullptr ||
      descriptor.block_scales_data == nullptr ||
      descriptor.tensor_scale_data == nullptr ||
      descriptor.output_rows == 0 ||
      descriptor.input_cols == 0 ||
      descriptor.input_cols % 2 != 0 ||
      descriptor.input_cols % kNvfp4BlockWidth != 0 ||
      descriptor.tensor_scale_nbytes != sizeof(float)) {
    return std::nullopt;
  }

  const std::size_t packed_cols = descriptor.packed_nbytes / descriptor.output_rows;
  const std::size_t scale_cols = descriptor.block_scales_nbytes / descriptor.output_rows;
  if (packed_cols == 0 ||
      scale_cols == 0 ||
      packed_cols * descriptor.output_rows != descriptor.packed_nbytes ||
      scale_cols * descriptor.output_rows != descriptor.block_scales_nbytes ||
      scale_cols != descriptor.input_cols / kNvfp4BlockWidth) {
    return std::nullopt;
  }

  FlashInferPreparedNvfp4WeightHost prepared;
  prepared.output_rows = descriptor.output_rows;
  prepared.input_cols = descriptor.input_cols;
  if (tensor_scale_override.has_value()) {
    prepared.tensor_scale = *tensor_scale_override;
  } else {
    std::memcpy(
        &prepared.tensor_scale,
        descriptor.tensor_scale_data,
        sizeof(float));
  }
  if (!std::isfinite(prepared.tensor_scale) || prepared.tensor_scale <= 0.0f) {
    return std::nullopt;
  }

  if (!FlashInferShuffleMatrixA(
          descriptor.packed_data,
          descriptor.output_rows,
          packed_cols,
          epilogue_tile_m,
          &prepared.shuffled_packed_major_k)) {
    return std::nullopt;
  }
  if (!FlashInferShuffleMatrixSfALinear(
          descriptor.block_scales_data,
          descriptor.output_rows,
          scale_cols,
          epilogue_tile_m,
          &prepared.shuffled_scales_linear)) {
    return std::nullopt;
  }
  if (!FlashInferInterleaveBlockScales128x4(
          prepared.shuffled_scales_linear.data(),
          descriptor.output_rows,
          scale_cols,
          &prepared.shuffled_scales_128x4)) {
    return std::nullopt;
  }
  return prepared;
}

std::optional<NemotronFlashInferNvfp4WeightView> BuildFlashInferRawNvfp4WeightView(
    const GemmDescriptor& descriptor,
    std::optional<float> tensor_scale_override) {
  if (descriptor.kernel_family != GemmKernelFamily::kCublasLtNvfp4BlockScaled ||
      descriptor.packed_data == nullptr ||
      descriptor.block_scales_data == nullptr ||
      descriptor.tensor_scale_data == nullptr ||
      descriptor.output_rows == 0 ||
      descriptor.input_cols == 0 ||
      descriptor.input_cols % kNvfp4BlockWidth != 0 ||
      descriptor.tensor_scale_nbytes != sizeof(float)) {
    return std::nullopt;
  }

  const std::size_t scale_cols = descriptor.input_cols / kNvfp4BlockWidth;
  if (descriptor.block_scales_nbytes != descriptor.output_rows * scale_cols) {
    return std::nullopt;
  }

  float tensor_scale = 0.0f;
  if (tensor_scale_override.has_value()) {
    tensor_scale = *tensor_scale_override;
  } else {
    std::memcpy(&tensor_scale, descriptor.tensor_scale_data, sizeof(float));
  }
  if (!std::isfinite(tensor_scale) || tensor_scale <= 0.0f) {
    return std::nullopt;
  }

  NemotronFlashInferNvfp4WeightView view{};
  view.packed_data = descriptor.packed_data;
  view.packed_nbytes = descriptor.packed_nbytes;
  view.scale_data = descriptor.block_scales_data;
  view.scale_nbytes = descriptor.block_scales_nbytes;
  view.tensor_scale = tensor_scale;
  view.dequant_scale = 1.0f / tensor_scale;
  view.output_rows = descriptor.output_rows;
  view.input_cols = descriptor.input_cols;
  view.scale_rows = descriptor.output_rows;
  view.scale_cols = scale_cols;
  view.packed_layout = NEMOTRON_FLASHINFER_WEIGHT_LAYOUT_RAW_ROW_MAJOR;
  view.scale_layout = NEMOTRON_FLASHINFER_SCALE_LAYOUT_RAW_LINEAR;
  return view;
}

NemotronFlashInferNvfp4WeightView BuildFlashInferPreparedNvfp4WeightView(
    const FlashInferPreparedNvfp4WeightHost& prepared) {
  NemotronFlashInferNvfp4WeightView view{};
  view.packed_data = prepared.shuffled_packed_major_k.data();
  view.packed_nbytes = prepared.shuffled_packed_major_k.size();
  view.scale_data = prepared.shuffled_scales_128x4.data();
  view.scale_nbytes = prepared.shuffled_scales_128x4.size();
  view.tensor_scale = prepared.tensor_scale;
  view.dequant_scale = prepared.tensor_scale != 0.0f ? (1.0f / prepared.tensor_scale) : 0.0f;
  view.output_rows = prepared.output_rows;
  view.input_cols = prepared.input_cols;
  view.scale_rows = prepared.output_rows;
  view.scale_cols = prepared.input_cols / kNvfp4BlockWidth;
  view.packed_layout = NEMOTRON_FLASHINFER_WEIGHT_LAYOUT_SHUFFLED_MAJOR_K;
  view.scale_layout = NEMOTRON_FLASHINFER_SCALE_LAYOUT_SWIZZLED_128X4;
  return view;
}

}  // namespace nemotron
