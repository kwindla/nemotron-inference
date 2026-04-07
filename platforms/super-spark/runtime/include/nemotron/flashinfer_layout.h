#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "nemotron/flashinfer_moe_plugin_abi.h"
#include "nemotron/gemm_catalog.h"

namespace nemotron {

struct FlashInferPreparedNvfp4WeightHost {
  std::vector<std::uint8_t> shuffled_packed_major_k;
  std::vector<std::uint8_t> shuffled_scales_linear;
  std::vector<std::uint8_t> shuffled_scales_128x4;
  float tensor_scale = 0.0f;
  std::size_t output_rows = 0;
  std::size_t input_cols = 0;
};

std::size_t GetFlashInferShuffleBlockSize(std::size_t epilogue_tile_m);

std::vector<std::size_t> BuildFlashInferShuffleMatrixARowIndices(
    std::size_t rows,
    std::size_t epilogue_tile_m);

bool FlashInferShuffleMatrixA(
    const std::uint8_t* input,
    std::size_t rows,
    std::size_t cols,
    std::size_t epilogue_tile_m,
    std::vector<std::uint8_t>* output);

bool FlashInferShuffleMatrixSfALinear(
    const std::uint8_t* input,
    std::size_t rows,
    std::size_t cols,
    std::size_t epilogue_tile_m,
    std::vector<std::uint8_t>* output);

bool FlashInferInterleaveBlockScales128x4(
    const std::uint8_t* input,
    std::size_t rows,
    std::size_t cols,
    std::vector<std::uint8_t>* output);

bool FlashInferShuffleMatrixSfA(
    const std::uint8_t* input,
    std::size_t rows,
    std::size_t cols,
    std::size_t epilogue_tile_m,
    std::vector<std::uint8_t>* output);

bool FlashInferConvertToBlockLayout(
    const std::uint8_t* input,
    std::size_t rows,
    std::size_t cols,
    std::size_t block_k,
    std::vector<std::uint8_t>* output);

bool FlashInferPrepareShuffledBlockMajorWeight(
    const std::uint8_t* input,
    std::size_t rows,
    std::size_t cols,
    std::size_t epilogue_tile_m,
    std::size_t block_k,
    std::vector<std::uint8_t>* output);

std::optional<FlashInferPreparedNvfp4WeightHost> PrepareFlashInferNvfp4WeightHost(
    const GemmDescriptor& descriptor,
    std::optional<float> tensor_scale_override = std::nullopt,
    std::size_t epilogue_tile_m = 128);

std::optional<NemotronFlashInferNvfp4WeightView> BuildFlashInferRawNvfp4WeightView(
    const GemmDescriptor& descriptor,
    std::optional<float> tensor_scale_override = std::nullopt);

NemotronFlashInferNvfp4WeightView BuildFlashInferPreparedNvfp4WeightView(
    const FlashInferPreparedNvfp4WeightHost& prepared);

}  // namespace nemotron
