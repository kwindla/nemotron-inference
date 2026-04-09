#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

#include "nemotron/routed_expert_runtime.h"

namespace nemotron {

struct PreparedNvfp4ExecutionWeightHostData {
  std::size_t output_rows = 0;
  std::size_t input_cols = 0;
  bool padded = false;
  std::vector<std::uint8_t> packed;
  std::vector<std::uint8_t> block_scales;
};

inline bool PrepareNvfp4WeightForExecution(
    std::string_view op_class,
    std::size_t logical_output_rows,
    std::size_t logical_input_cols,
    const std::uint8_t* host_packed,
    std::size_t packed_nbytes,
    const std::uint8_t* host_block_scales,
    std::size_t block_scales_nbytes,
    PreparedNvfp4ExecutionWeightHostData* output) {
  if (output == nullptr ||
      host_packed == nullptr ||
      host_block_scales == nullptr ||
      logical_output_rows == 0 ||
      logical_input_cols == 0 ||
      (logical_input_cols % kNvfp4ScaleBlockWidthRuntime) != 0) {
    return false;
  }

  const std::size_t logical_packed_nbytes = (logical_output_rows * logical_input_cols) / 2u;
  const std::size_t logical_block_scales_nbytes =
      logical_output_rows * (logical_input_cols / kNvfp4ScaleBlockWidthRuntime);
  if (packed_nbytes != logical_packed_nbytes ||
      block_scales_nbytes != logical_block_scales_nbytes) {
    return false;
  }

  const std::size_t execution_output_rows =
      IsRoutedExpertUpOpClass(op_class)
          ? DefaultRoutedExpertIntermediateSizePadded(logical_output_rows)
          : logical_output_rows;
  const std::size_t execution_input_cols =
      IsRoutedExpertDownOpClass(op_class)
          ? DefaultRoutedExpertIntermediateSizePadded(logical_input_cols)
          : logical_input_cols;

  output->output_rows = execution_output_rows;
  output->input_cols = execution_input_cols;
  output->padded =
      execution_output_rows != logical_output_rows ||
      execution_input_cols != logical_input_cols;
  output->packed.clear();
  output->block_scales.clear();

  if (!output->padded) {
    return true;
  }

  const std::size_t source_packed_row_bytes = logical_input_cols / 2u;
  const std::size_t source_blocks_per_row =
      logical_input_cols / kNvfp4ScaleBlockWidthRuntime;
  const std::size_t destination_packed_row_bytes = execution_input_cols / 2u;
  const std::size_t destination_blocks_per_row =
      execution_input_cols / kNvfp4ScaleBlockWidthRuntime;

  output->packed.assign((execution_output_rows * execution_input_cols) / 2u, 0u);
  output->block_scales.assign(
      execution_output_rows * destination_blocks_per_row,
      0u);

  for (std::size_t row = 0; row < logical_output_rows; ++row) {
    std::memcpy(
        output->packed.data() + (row * destination_packed_row_bytes),
        host_packed + (row * source_packed_row_bytes),
        source_packed_row_bytes);
    std::memcpy(
        output->block_scales.data() + (row * destination_blocks_per_row),
        host_block_scales + (row * source_blocks_per_row),
        source_blocks_per_row);
  }
  return true;
}

}  // namespace nemotron
