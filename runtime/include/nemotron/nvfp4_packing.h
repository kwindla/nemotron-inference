#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "nemotron/nvfp4_scale_layout.h"

namespace nemotron {

struct Nvfp4PackOptions {
  std::optional<float> fixed_tensor_scale;
  const float* fixed_tensor_scale_device = nullptr;
  std::optional<Nvfp4ScaleLayout> execution_scale_layout;
};

struct HostNvfp4Matrix {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<std::uint8_t> packed;
  std::vector<std::uint8_t> block_scales;
  float tensor_scale = 1.0f;

  bool valid() const;
  const std::uint8_t* packed_data() const;
  const std::uint8_t* block_scales_data() const;
  const std::uint8_t* tensor_scale_data() const;
  std::size_t packed_nbytes() const;
  std::size_t block_scales_nbytes() const;
  std::size_t tensor_scale_nbytes() const;
};

std::size_t PackedFp4Bytes(std::size_t rows, std::size_t cols);
std::size_t RowMajorNvfp4ScaleBytes(std::size_t rows, std::size_t cols);

std::optional<HostNvfp4Matrix> PackRowMajorFp32ToNvfp4(
    const float* data,
    std::size_t rows,
    std::size_t cols,
    const Nvfp4PackOptions& options = {});

}  // namespace nemotron
