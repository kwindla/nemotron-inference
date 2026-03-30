#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace nemotron {

struct Nvfp4ExecutionScaleLayout {
  std::size_t logical_rows = 0;
  std::size_t logical_blocks_per_row = 0;
  std::size_t padded_rows = 0;
  std::size_t padded_blocks_per_row = 0;

  bool valid() const;
  std::size_t nbytes() const;
};

std::optional<Nvfp4ExecutionScaleLayout> BuildNvfp4ExecutionScaleLayout(
    std::size_t rows,
    std::size_t cols);

std::size_t ExecutionNvfp4ScaleBytes(std::size_t rows, std::size_t cols);

std::vector<std::uint8_t> SwizzleRowMajorNvfp4ScalesForExecution(
    const std::uint8_t* row_major_scales,
    std::size_t rows,
    std::size_t cols);

}  // namespace nemotron
