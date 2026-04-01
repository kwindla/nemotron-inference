#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace nemotron {

enum class Nvfp4ScaleLayout {
  kSwizzled128x4,
  kSwizzled8x4,
};

const char* ToString(Nvfp4ScaleLayout layout);

Nvfp4ScaleLayout ResolveActivationNvfp4ScaleLayout(
    std::size_t rows,
    std::optional<Nvfp4ScaleLayout> requested_layout = std::nullopt);

struct Nvfp4ExecutionScaleLayout {
  Nvfp4ScaleLayout scale_layout = Nvfp4ScaleLayout::kSwizzled128x4;
  std::size_t logical_rows = 0;
  std::size_t logical_blocks_per_row = 0;
  std::size_t padded_rows = 0;
  std::size_t padded_blocks_per_row = 0;

  bool valid() const;
  std::size_t nbytes() const;
};

std::optional<Nvfp4ExecutionScaleLayout> BuildNvfp4ExecutionScaleLayout(
    std::size_t rows,
    std::size_t cols,
    Nvfp4ScaleLayout scale_layout = Nvfp4ScaleLayout::kSwizzled128x4);

std::size_t ExecutionNvfp4ScaleBytes(
    std::size_t rows,
    std::size_t cols,
    Nvfp4ScaleLayout scale_layout = Nvfp4ScaleLayout::kSwizzled128x4);

std::vector<std::uint8_t> SwizzleRowMajorNvfp4ScalesForExecution(
    const std::uint8_t* row_major_scales,
    std::size_t rows,
    std::size_t cols,
    Nvfp4ScaleLayout scale_layout = Nvfp4ScaleLayout::kSwizzled128x4);

}  // namespace nemotron
