#include "nemotron/flashinfer_layout.h"

#include <cuda_fp8.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

void TestShuffleIndices16() {
  const auto indices = nemotron::BuildFlashInferShuffleMatrixARowIndices(16, 64);
  const std::vector<std::size_t> expected = {
      0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
  };
  assert(indices == expected);
}

void TestShuffleIndices32() {
  const auto indices = nemotron::BuildFlashInferShuffleMatrixARowIndices(32, 128);
  const std::vector<std::size_t> expected = {
      0, 4, 8, 12, 16, 20, 24, 28,
      1, 5, 9, 13, 17, 21, 25, 29,
      2, 6, 10, 14, 18, 22, 26, 30,
      3, 7, 11, 15, 19, 23, 27, 31,
  };
  assert(indices == expected);
}

void TestShuffleMatrixA() {
  std::vector<std::uint8_t> input(16 * 2, 0);
  for (std::size_t row = 0; row < 16; ++row) {
    input[row * 2] = static_cast<std::uint8_t>(row);
    input[row * 2 + 1] = static_cast<std::uint8_t>(row + 100);
  }
  std::vector<std::uint8_t> shuffled;
  const bool ok = nemotron::FlashInferShuffleMatrixA(
      input.data(),
      16,
      2,
      64,
      &shuffled);
  assert(ok);
  for (std::size_t row = 0; row < 16; ++row) {
    const std::size_t expected_old_row =
        std::vector<std::size_t>{0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15}[row];
    assert(shuffled[row * 2] == input[expected_old_row * 2]);
    assert(shuffled[row * 2 + 1] == input[expected_old_row * 2 + 1]);
  }
}

void TestConvertToBlockLayout() {
  const std::vector<std::uint8_t> input = {
      0, 1, 2, 3,
      4, 5, 6, 7,
  };
  std::vector<std::uint8_t> output;
  const bool ok = nemotron::FlashInferConvertToBlockLayout(
      input.data(),
      2,
      4,
      2,
      &output);
  assert(ok);
  const std::vector<std::uint8_t> expected = {
      0, 1,
      4, 5,
      2, 3,
      6, 7,
  };
  assert(output == expected);
}

void TestInterleaveBlockScales128x4() {
  std::vector<std::uint8_t> input(128 * 4, 0);
  for (std::size_t row = 0; row < 128; ++row) {
    for (std::size_t col = 0; col < 4; ++col) {
      input[row * 4 + col] = static_cast<std::uint8_t>((row + 1) * 10 + col);
    }
  }
  std::vector<std::uint8_t> output;
  const bool ok =
      nemotron::FlashInferInterleaveBlockScales128x4(input.data(), 128, 4, &output);
  assert(ok);
  assert(output.size() == input.size());
  assert(output[0] == input[0 * 4 + 0]);
  assert(output[4] == input[32 * 4 + 0]);
  assert(output[8] == input[64 * 4 + 0]);
  assert(output[12] == input[96 * 4 + 0]);
  assert(output[16] == input[1 * 4 + 0]);
}

void TestShuffleMatrixSfA() {
  std::vector<std::uint8_t> input(128 * 4, 0);
  for (std::size_t row = 0; row < 128; ++row) {
    for (std::size_t col = 0; col < 4; ++col) {
      input[row * 4 + col] = static_cast<std::uint8_t>(row);
    }
  }
  std::vector<std::uint8_t> output;
  const bool ok =
      nemotron::FlashInferShuffleMatrixSfA(input.data(), 128, 4, 128, &output);
  assert(ok);
  assert(output.size() == input.size());
  std::vector<std::uint8_t> expected_linear;
  const bool linear_ok =
      nemotron::FlashInferShuffleMatrixSfALinear(input.data(), 128, 4, 128, &expected_linear);
  assert(linear_ok);
  std::vector<std::uint8_t> expected_interleaved;
  const bool interleave_ok = nemotron::FlashInferInterleaveBlockScales128x4(
      expected_linear.data(), 128, 4, &expected_interleaved);
  assert(interleave_ok);
  assert(output == expected_interleaved);
}

void TestPrepareShuffledBlockMajorWeight() {
  std::vector<std::uint8_t> input(16 * 4, 0);
  for (std::size_t row = 0; row < 16; ++row) {
    for (std::size_t col = 0; col < 4; ++col) {
      input[row * 4 + col] = static_cast<std::uint8_t>(row * 4 + col);
    }
  }
  std::vector<std::uint8_t> output;
  const bool ok = nemotron::FlashInferPrepareShuffledBlockMajorWeight(
      input.data(),
      16,
      4,
      64,
      2,
      &output);
  assert(ok);
  assert(output.size() == input.size());
  assert(output[0] == 0);
  assert(output[1] == 1);
  assert(output[2] == 8);
  assert(output[3] == 9);
}

void TestPrepareFlashInferNvfp4WeightHost() {
  const std::size_t rows = 128;
  const std::size_t input_cols = 64;
  std::vector<std::uint8_t> packed(rows * (input_cols / 2), 0);
  for (std::size_t i = 0; i < packed.size(); ++i) {
    packed[i] = static_cast<std::uint8_t>(i & 0xFF);
  }
  std::vector<std::uint8_t> scales(rows * (input_cols / 16), 0);
  for (std::size_t i = 0; i < scales.size(); ++i) {
    scales[i] = static_cast<std::uint8_t>((i + 11) & 0xFF);
  }
  float tensor_scale_value = 0.25f;

  nemotron::GemmDescriptor descriptor;
  descriptor.kernel_family = nemotron::GemmKernelFamily::kCublasLtNvfp4BlockScaled;
  descriptor.output_rows = rows;
  descriptor.input_cols = input_cols;
  descriptor.packed_data = packed.data();
  descriptor.packed_nbytes = packed.size();
  descriptor.block_scales_data = scales.data();
  descriptor.block_scales_nbytes = scales.size();
  descriptor.tensor_scale_data = reinterpret_cast<const std::uint8_t*>(&tensor_scale_value);
  descriptor.tensor_scale_nbytes = sizeof(float);

  const auto prepared = nemotron::PrepareFlashInferNvfp4WeightHost(descriptor, 128);
  assert(prepared.has_value());
  assert(prepared->output_rows == rows);
  assert(prepared->input_cols == input_cols);
  assert(prepared->shuffled_packed_major_k.size() == packed.size());
  assert(prepared->shuffled_scales_linear.size() == scales.size());
  assert(prepared->shuffled_scales_128x4.size() == scales.size());
  assert(prepared->tensor_scale == tensor_scale_value);

  std::vector<std::uint8_t> expected_shuffled_packed;
  const bool packed_ok = nemotron::FlashInferShuffleMatrixA(
      packed.data(), rows, input_cols / 2, 128, &expected_shuffled_packed);
  assert(packed_ok);
  assert(prepared->shuffled_packed_major_k == expected_shuffled_packed);

  std::vector<std::uint8_t> expected_shuffled_scales;
  const bool scale_ok = nemotron::FlashInferShuffleMatrixSfA(
      scales.data(), rows, input_cols / 16, 128, &expected_shuffled_scales);
  assert(scale_ok);
  assert(prepared->shuffled_scales_128x4 == expected_shuffled_scales);

  const auto raw_view = nemotron::BuildFlashInferRawNvfp4WeightView(descriptor);
  assert(raw_view.has_value());
  assert(raw_view->packed_data == packed.data());
  assert(raw_view->scale_data == scales.data());
  assert(raw_view->packed_layout == NEMOTRON_FLASHINFER_WEIGHT_LAYOUT_RAW_ROW_MAJOR);
  assert(raw_view->scale_layout == NEMOTRON_FLASHINFER_SCALE_LAYOUT_RAW_LINEAR);
  assert(raw_view->tensor_scale == tensor_scale_value);
  assert(raw_view->dequant_scale == 4.0f);
  assert(raw_view->scale_rows == rows);
  assert(raw_view->scale_cols == input_cols / 16);

  const auto prepared_view = nemotron::BuildFlashInferPreparedNvfp4WeightView(*prepared);
  assert(prepared_view.packed_data == prepared->shuffled_packed_major_k.data());
  assert(prepared_view.scale_data == prepared->shuffled_scales_128x4.data());
  assert(prepared_view.packed_layout == NEMOTRON_FLASHINFER_WEIGHT_LAYOUT_SHUFFLED_MAJOR_K);
  assert(prepared_view.scale_layout == NEMOTRON_FLASHINFER_SCALE_LAYOUT_SWIZZLED_128X4);
  assert(prepared_view.dequant_scale == 4.0f);
  assert(prepared_view.scale_rows == rows);
  assert(prepared_view.scale_cols == input_cols / 16);
}

}  // namespace

int main() {
  TestShuffleIndices16();
  TestShuffleIndices32();
  TestShuffleMatrixA();
  TestConvertToBlockLayout();
  TestInterleaveBlockScales128x4();
  TestShuffleMatrixSfA();
  TestPrepareShuffledBlockMajorWeight();
  TestPrepareFlashInferNvfp4WeightHost();
  return 0;
}
