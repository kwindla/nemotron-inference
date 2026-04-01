#include "nemotron/nvfp4_scale_layout.h"

namespace nemotron {
namespace {

constexpr std::size_t kBlockWidth = 16;
constexpr std::size_t kBlockTile = 4;

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

std::size_t ExecutionScaleOffset(
    std::size_t row,
    std::size_t block_col,
    std::size_t padded_blocks_per_row,
    Nvfp4ScaleLayout scale_layout) {
  const std::size_t num_k_tiles = padded_blocks_per_row / kBlockTile;
  const std::size_t k_tile = block_col / kBlockTile;
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

}  // namespace

const char* ToString(Nvfp4ScaleLayout layout) {
  switch (layout) {
    case Nvfp4ScaleLayout::kSwizzled128x4:
      return "swizzled_128x4";
    case Nvfp4ScaleLayout::kSwizzled8x4:
      return "swizzled_8x4";
  }
  return "unknown";
}

Nvfp4ScaleLayout ResolveActivationNvfp4ScaleLayout(
    std::size_t rows,
    std::optional<Nvfp4ScaleLayout> requested_layout) {
  if (requested_layout.has_value()) {
    return *requested_layout;
  }
  return rows <= 32 ? Nvfp4ScaleLayout::kSwizzled8x4
                    : Nvfp4ScaleLayout::kSwizzled128x4;
}

bool Nvfp4ExecutionScaleLayout::valid() const {
  return logical_rows > 0 &&
         logical_blocks_per_row > 0 &&
         padded_rows >= logical_rows &&
         padded_blocks_per_row >= logical_blocks_per_row &&
         padded_rows % RowTile(scale_layout) == 0 &&
         padded_blocks_per_row % kBlockTile == 0;
}

std::size_t Nvfp4ExecutionScaleLayout::nbytes() const {
  return valid() ? padded_rows * padded_blocks_per_row : 0;
}

std::optional<Nvfp4ExecutionScaleLayout> BuildNvfp4ExecutionScaleLayout(
    std::size_t rows,
    std::size_t cols,
    Nvfp4ScaleLayout scale_layout) {
  if (rows == 0 || cols == 0 || cols % kBlockWidth != 0) {
    return std::nullopt;
  }

  Nvfp4ExecutionScaleLayout layout;
  layout.scale_layout = scale_layout;
  layout.logical_rows = rows;
  layout.logical_blocks_per_row = cols / kBlockWidth;
  layout.padded_rows = RoundUp(rows, RowTile(scale_layout));
  layout.padded_blocks_per_row = RoundUp(layout.logical_blocks_per_row, kBlockTile);
  if (!layout.valid()) {
    return std::nullopt;
  }
  return layout;
}

std::size_t ExecutionNvfp4ScaleBytes(
    std::size_t rows,
    std::size_t cols,
    Nvfp4ScaleLayout scale_layout) {
  const auto layout = BuildNvfp4ExecutionScaleLayout(rows, cols, scale_layout);
  return layout ? layout->nbytes() : 0;
}

std::vector<std::uint8_t> SwizzleRowMajorNvfp4ScalesForExecution(
    const std::uint8_t* row_major_scales,
    std::size_t rows,
    std::size_t cols,
    Nvfp4ScaleLayout scale_layout) {
  const auto layout = BuildNvfp4ExecutionScaleLayout(rows, cols, scale_layout);
  if (!layout || row_major_scales == nullptr) {
    return {};
  }

  std::vector<std::uint8_t> swizzled(layout->nbytes(), 0u);
  for (std::size_t row = 0; row < layout->logical_rows; ++row) {
    for (std::size_t block_col = 0; block_col < layout->logical_blocks_per_row; ++block_col) {
      const std::size_t source_offset = row * layout->logical_blocks_per_row + block_col;
      const std::size_t destination_offset =
          ExecutionScaleOffset(
              row,
              block_col,
              layout->padded_blocks_per_row,
              layout->scale_layout);
      swizzled[destination_offset] = row_major_scales[source_offset];
    }
  }
  return swizzled;
}

}  // namespace nemotron
