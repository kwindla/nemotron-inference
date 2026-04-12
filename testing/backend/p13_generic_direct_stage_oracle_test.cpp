#include "nemotron/device_nvfp4_matrix.h"
#include "nemotron/device_tensor.h"
#include "nemotron/fused_moe_prefill.h"
#include "nemotron/nvfp4_scale_layout.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

using nemotron::BuildNvfp4ExecutionScaleLayout;
using nemotron::DeviceNvfp4Matrix;
using nemotron::DeviceTensorBf16;
using nemotron::DeviceTensorFp32;
using nemotron::Nvfp4ExecutionScaleLayout;
using nemotron::Nvfp4ScaleLayout;
using nemotron::ResolveActivationNvfp4ScaleLayout;
using nemotron::RunP13GenericDirectStageOracleForTesting;
using nemotron::SwizzleRowMajorNvfp4ScalesForExecution;

constexpr int kRows = 32;
constexpr int kCols = 128;
constexpr std::size_t kBlockWidth = 16;
constexpr std::size_t kBlocksPerRow = kCols / kBlockWidth;
constexpr std::size_t kPackedBytesPerRow = kCols / 2u;
constexpr std::size_t kPackedBytesPerBlock = kBlockWidth / 2u;
constexpr std::size_t kScaleCount = kRows * kBlocksPerRow;
constexpr std::size_t kElementCount = kRows * kCols;
constexpr float kNvfp4Fp4MaxFinite = 6.0f;
constexpr float kNvfp4MinScale = 1.0f / 1024.0f;

struct OracleOutputs {
  std::vector<__nv_bfloat16> generic_dense;
  std::vector<__nv_bfloat16> store_dense;
  std::vector<std::uint8_t> packed;
  std::vector<std::uint8_t> block_scales;
  std::vector<std::uint8_t> matmul_block_scales;
  std::vector<float> activation_output_scale;
  std::vector<float> per_row_tensor_scales;
  float tensor_scale = 0.0f;
};

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool expect_divergence(bool equal, const std::string& message) {
  return expect(!equal, message);
}

bool check_cuda(cudaError_t status, const std::string& where) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA error at " << where << ": " << cudaGetErrorString(status) << "\n";
    return false;
  }
  return true;
}

bool has_cuda_device() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

std::uint32_t float_bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::uint16_t bf16_bits(__nv_bfloat16 value) {
  std::uint16_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool compare_byte_vectors(
    const std::vector<std::uint8_t>& actual,
    const std::vector<std::uint8_t>& expected,
    const std::string& label) {
  if (!expect(actual.size() == expected.size(), label + " size mismatch")) {
    return false;
  }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      std::cerr << "FAIL: " << label << " mismatch at byte " << i << " actual="
                << static_cast<unsigned int>(actual[i]) << " expected="
                << static_cast<unsigned int>(expected[i]) << "\n";
      return false;
    }
  }
  return true;
}

bool compare_float_vectors_bitwise(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    const std::string& label) {
  if (!expect(actual.size() == expected.size(), label + " size mismatch")) {
    return false;
  }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (float_bits(actual[i]) != float_bits(expected[i])) {
      std::cerr << "FAIL: " << label << " mismatch at element " << i
                << " actual_bits=0x" << std::hex << float_bits(actual[i])
                << " expected_bits=0x" << float_bits(expected[i]) << std::dec << "\n";
      return false;
    }
  }
  return true;
}

bool compare_bf16_vectors_bitwise(
    const std::vector<__nv_bfloat16>& actual,
    const std::vector<__nv_bfloat16>& expected,
    const std::string& label) {
  if (!expect(actual.size() == expected.size(), label + " size mismatch")) {
    return false;
  }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (bf16_bits(actual[i]) != bf16_bits(expected[i])) {
      std::cerr << "FAIL: " << label << " mismatch at element " << i
                << " actual_bits=0x" << std::hex << bf16_bits(actual[i])
                << " expected_bits=0x" << bf16_bits(expected[i]) << std::dec << "\n";
      return false;
    }
  }
  return true;
}

bool compare_float_scalar_bitwise(float actual, float expected, const std::string& label) {
  if (float_bits(actual) != float_bits(expected)) {
    std::cerr << "FAIL: " << label << " mismatch actual_bits=0x" << std::hex
              << float_bits(actual) << " expected_bits=0x" << float_bits(expected)
              << std::dec << "\n";
    return false;
  }
  return true;
}

bool equal_bf16_vectors_bitwise(
    const std::vector<__nv_bfloat16>& lhs,
    const std::vector<__nv_bfloat16>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (bf16_bits(lhs[i]) != bf16_bits(rhs[i])) {
      return false;
    }
  }
  return true;
}

bool equal_byte_vectors(
    const std::vector<std::uint8_t>& lhs,
    const std::vector<std::uint8_t>& rhs) {
  return lhs == rhs;
}

bool equal_float_vectors_bitwise(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (float_bits(lhs[i]) != float_bits(rhs[i])) {
      return false;
    }
  }
  return true;
}

float relu2(float value) {
  return value > 0.0f ? value * value : 0.0f;
}

float clamp_nvfp4_scale(float value) {
  if (!std::isfinite(value) || value < kNvfp4MinScale) {
    return kNvfp4MinScale;
  }
  return value;
}

float decode_positive_fp8_value(std::uint8_t code) {
  if (code == 0u) {
    return 0.0f;
  }
  const std::uint8_t exponent = static_cast<std::uint8_t>((code >> 3u) & 0x0Fu);
  const std::uint8_t mantissa = static_cast<std::uint8_t>(code & 0x07u);
  if (exponent == 0u) {
    return std::ldexp(static_cast<float>(mantissa), -9);
  }
  return std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, static_cast<int>(exponent) - 7);
}

const std::array<float, 0x7F>& positive_fp8_finite_values() {
  static const std::array<float, 0x7F> values = []() {
    std::array<float, 0x7F> table{};
    for (std::size_t code = 0; code <= 0x7Eu; ++code) {
      table[code] = decode_positive_fp8_value(static_cast<std::uint8_t>(code));
    }
    return table;
  }();
  return values;
}

std::uint8_t encode_fp8_scale_reference(float value) {
  if (!(value > 0.0f) || !std::isfinite(value)) {
    return 0u;
  }
  const auto& candidates = positive_fp8_finite_values();
  float best_error = std::numeric_limits<float>::infinity();
  std::uint8_t best_code = 0u;
  for (std::size_t code = 0; code <= 0x7Eu; ++code) {
    const float error = std::fabs(candidates[code] - value);
    if (error < best_error) {
      best_error = error;
      best_code = static_cast<std::uint8_t>(code);
      continue;
    }
    if (error == best_error && ((code & 1u) == 0u) && ((best_code & 1u) != 0u)) {
      best_code = static_cast<std::uint8_t>(code);
    }
  }
  return best_code;
}

std::uint8_t encode_fp4_reference(float value) {
  static constexpr std::array<float, 8> kPositiveValues = {
      0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  if (!(value > 0.0f) || !std::isfinite(value)) {
    return 0u;
  }
  float best_error = std::numeric_limits<float>::infinity();
  std::uint8_t best_code = 0u;
  for (std::size_t code = 0; code < kPositiveValues.size(); ++code) {
    const float error = std::fabs(kPositiveValues[code] - value);
    if (error < best_error) {
      best_error = error;
      best_code = static_cast<std::uint8_t>(code);
      continue;
    }
    if (error == best_error && ((code & 1u) == 0u) && ((best_code & 1u) != 0u)) {
      best_code = static_cast<std::uint8_t>(code);
    }
  }
  return best_code;
}

std::vector<float> make_input() {
  std::vector<float> values(kElementCount, 0.0f);
  for (int row = 0; row < kRows; ++row) {
    for (int col = 0; col < kCols; ++col) {
      const int selector = (row * 17 + col * 11) % 10;
      float value = 0.0f;
      switch (selector) {
        case 0:
          value = 0.0f;
          break;
        case 1:
          value = -0.35f * static_cast<float>((row % 5) + 1);
          break;
        case 2:
          value = 0.20f * static_cast<float>((col % 7) + 1);
          break;
        case 3:
          value = -0.90f;
          break;
        case 4:
          value = 0.75f + 0.05f * static_cast<float>(row % 3);
          break;
        case 5:
          value = -0.12f * static_cast<float>((col % 4) + 1);
          break;
        case 6:
          value = 1.25f;
          break;
        case 7:
          value = -1.75f;
          break;
        case 8:
          value = 0.03125f * static_cast<float>((row + col) % 9);
          break;
        default:
          value = 0.5f;
          break;
      }
      values[static_cast<std::size_t>(row) * kCols + static_cast<std::size_t>(col)] = value;
    }
  }
  return values;
}

std::vector<float> make_non_uniform_per_row_scales() {
  std::vector<float> values(kRows, 1.0f);
  for (int row = 0; row < kRows; ++row) {
    values[row] = 0.5f + 0.125f * static_cast<float>(row % 6);
  }
  return values;
}

std::vector<__nv_bfloat16> compute_expected_dense_bf16(
    const std::vector<float>& input,
    const std::vector<float>& per_row_tensor_scales,
    int valid_rows) {
  std::vector<__nv_bfloat16> output(kElementCount, __float2bfloat16(0.0f));
  for (int row = 0; row < valid_rows; ++row) {
    for (int col = 0; col < kCols; ++col) {
      const float value =
          input[static_cast<std::size_t>(row) * kCols + static_cast<std::size_t>(col)] *
          per_row_tensor_scales[static_cast<std::size_t>(row)];
      output[static_cast<std::size_t>(row) * kCols + static_cast<std::size_t>(col)] =
          __float2bfloat16(value);
    }
  }
  return output;
}

OracleOutputs compute_reference_pack_from_fp32(
    const std::vector<float>& input,
    const std::vector<float>& per_row_tensor_scales,
    int valid_rows,
    Nvfp4ScaleLayout scale_layout) {
  OracleOutputs output;
  output.packed.assign(kRows * kPackedBytesPerRow, 0u);
  output.block_scales.assign(kScaleCount, 0u);
  output.activation_output_scale.assign(kScaleCount, 0.0f);
  output.per_row_tensor_scales = per_row_tensor_scales;
  output.tensor_scale = 1.0f;

  for (int row = 0; row < valid_rows; ++row) {
    for (std::size_t block = 0; block < kBlocksPerRow; ++block) {
      const std::size_t input_offset =
          static_cast<std::size_t>(row) * kCols + block * kBlockWidth;
      const std::size_t scale_index = static_cast<std::size_t>(row) * kBlocksPerRow + block;
      const std::size_t packed_offset =
          static_cast<std::size_t>(row) * kPackedBytesPerRow + block * kPackedBytesPerBlock;
      float activated_block[kBlockWidth];
      float block_max_abs = 0.0f;
      for (std::size_t i = 0; i < kBlockWidth; ++i) {
        activated_block[i] =
            relu2(input[input_offset + i] * per_row_tensor_scales[static_cast<std::size_t>(row)]);
        block_max_abs = std::max(block_max_abs, activated_block[i]);
      }

      float dequant_scale = 1.0f;
      if (block_max_abs > 0.0f) {
        dequant_scale = clamp_nvfp4_scale(block_max_abs / kNvfp4Fp4MaxFinite);
      }
      output.activation_output_scale[scale_index] = dequant_scale;
      output.block_scales[scale_index] = encode_fp8_scale_reference(dequant_scale);

      for (std::size_t pair = 0; pair < kPackedBytesPerBlock; ++pair) {
        const std::uint8_t lhs = encode_fp4_reference(activated_block[pair * 2u] / dequant_scale);
        const std::uint8_t rhs =
            encode_fp4_reference(activated_block[pair * 2u + 1u] / dequant_scale);
        output.packed[packed_offset + pair] =
            static_cast<std::uint8_t>((lhs & 0x0Fu) | ((rhs & 0x0Fu) << 4u));
      }
    }
  }

  output.matmul_block_scales = SwizzleRowMajorNvfp4ScalesForExecution(
      output.block_scales.data(),
      kRows,
      kCols,
      scale_layout);
  return output;
}

bool copy_per_row_scales(const DeviceNvfp4Matrix& pack, std::vector<float>* output) {
  output->assign(kRows, 0.0f);
  return check_cuda(
      cudaMemcpy(
          output->data(),
          pack.per_row_tensor_scales(),
          output->size() * sizeof(float),
          cudaMemcpyDeviceToHost),
      "cudaMemcpy(per_row_tensor_scales)");
}

OracleOutputs run_p13_oracle(
    const std::vector<float>& input,
    const std::vector<float>& per_row_tensor_scales,
    int valid_rows,
    const Nvfp4ExecutionScaleLayout& scale_layout,
    bool* ok) {
  OracleOutputs output;
  *ok = false;

  auto device_input = DeviceTensorFp32::Create({kRows, kCols});
  auto device_per_row_tensor_scales = DeviceTensorFp32::Create({kRows});
  auto device_generic_dense = DeviceTensorBf16::Create({kRows, kCols});
  auto device_store_dense = DeviceTensorBf16::Create({kRows, kCols});
  auto device_activation_scales = DeviceTensorFp32::Create({kRows, kBlocksPerRow});
  auto device_pack = DeviceNvfp4Matrix::Create(kRows, kCols, scale_layout.scale_layout);
  if (!device_input || !device_input->valid() ||
      !device_per_row_tensor_scales || !device_per_row_tensor_scales->valid() ||
      !device_generic_dense || !device_generic_dense->valid() ||
      !device_store_dense || !device_store_dense->valid() ||
      !device_activation_scales || !device_activation_scales->valid() ||
      !device_pack || !device_pack->valid()) {
    return output;
  }

  if (!device_input->CopyFromHost(input.data(), input.size()) ||
      !device_per_row_tensor_scales->CopyFromHost(
          per_row_tensor_scales.data(),
          per_row_tensor_scales.size())) {
    return output;
  }

  if (!RunP13GenericDirectStageOracleForTesting(
          device_input->data(),
          device_per_row_tensor_scales->data(),
          valid_rows,
          scale_layout.padded_blocks_per_row,
          scale_layout.scale_layout,
          device_generic_dense->data(),
          device_store_dense->data(),
          device_activation_scales->data(),
          const_cast<std::uint8_t*>(device_pack->packed_data()),
          const_cast<std::uint8_t*>(device_pack->block_scales_data()),
          const_cast<std::uint8_t*>(device_pack->matmul_block_scales_data()),
          const_cast<float*>(device_pack->device_tensor_scale_ptr()),
          const_cast<float*>(device_pack->per_row_tensor_scales()))) {
    return output;
  }

  output.generic_dense.assign(kElementCount, __float2bfloat16(0.0f));
  output.store_dense.assign(kElementCount, __float2bfloat16(0.0f));
  output.activation_output_scale.assign(kScaleCount, 0.0f);
  if (!device_generic_dense->CopyToHost(output.generic_dense.data(), output.generic_dense.size()) ||
      !device_store_dense->CopyToHost(output.store_dense.data(), output.store_dense.size()) ||
      !device_pack->CopyPackedToHost(&output.packed) ||
      !device_pack->CopyBlockScalesToHost(&output.block_scales) ||
      !device_pack->CopyMatmulBlockScalesToHost(&output.matmul_block_scales) ||
      !device_pack->CopyTensorScaleToHost(&output.tensor_scale) ||
      !device_activation_scales->CopyToHost(
          output.activation_output_scale.data(),
          output.activation_output_scale.size()) ||
      !copy_per_row_scales(*device_pack, &output.per_row_tensor_scales)) {
    return output;
  }

  *ok = true;
  return output;
}

bool test_p13_generic_direct_stage_contract(
    const std::vector<float>& input,
    const std::vector<float>& per_row_tensor_scales,
    int valid_rows,
    const Nvfp4ExecutionScaleLayout& scale_layout) {
  const std::string label = "p13_generic_direct_valid_rows_" + std::to_string(valid_rows);
  bool ok = false;
  const OracleOutputs actual = run_p13_oracle(
      input,
      per_row_tensor_scales,
      valid_rows,
      scale_layout,
      &ok);
  if (!expect(ok, label + " oracle execution should succeed")) {
    return false;
  }

  const std::vector<__nv_bfloat16> expected_dense =
      compute_expected_dense_bf16(input, per_row_tensor_scales, valid_rows);
  const OracleOutputs expected_pack =
      compute_reference_pack_from_fp32(input, per_row_tensor_scales, valid_rows, scale_layout.scale_layout);

  bool passed = true;
  passed = compare_bf16_vectors_bitwise(
               actual.store_dense,
               expected_dense,
               label + " store_dense_reference") &&
           passed;
  if (valid_rows == kRows) {
    // The full 32x128 P13 case is intentionally a negative sentinel for the
    // current dormant generic direct stage. If that path is fixed later, these
    // divergence checks should be flipped to equality checks.
    passed = expect_divergence(
                 equal_bf16_vectors_bitwise(actual.generic_dense, actual.store_dense),
                 label + " generic_dense should diverge from the live P13 store contract") &&
             passed;
    passed = expect_divergence(
                 equal_bf16_vectors_bitwise(actual.generic_dense, expected_dense),
                 label + " generic_dense should diverge from the dense reference") &&
             passed;
    passed = expect_divergence(
                 equal_byte_vectors(actual.packed, expected_pack.packed),
                 label + " packed_data should diverge under the generic direct stage") &&
             passed;
    passed = expect_divergence(
                 equal_byte_vectors(actual.block_scales, expected_pack.block_scales),
                 label + " block_scales_data should diverge under the generic direct stage") &&
             passed;
    passed = expect_divergence(
                 equal_byte_vectors(
                     actual.matmul_block_scales,
                     expected_pack.matmul_block_scales),
                 label +
                     " matmul_block_scales_data should diverge under the generic direct stage") &&
             passed;
    passed = expect_divergence(
                 equal_float_vectors_bitwise(
                     actual.activation_output_scale,
                     expected_pack.activation_output_scale),
                 label +
                     " activation_output_scale should diverge under the generic direct stage") &&
             passed;
  }
  passed = compare_float_scalar_bitwise(
               actual.tensor_scale,
               expected_pack.tensor_scale,
               label + " tensor_scale_data") &&
           passed;
  passed = compare_float_vectors_bitwise(
               actual.per_row_tensor_scales,
               expected_pack.per_row_tensor_scales,
               label + " per_row_tensor_scales") &&
           passed;
  return passed;
}

}  // namespace

int main() {
  if (!has_cuda_device()) {
    std::cout << "p13_generic_direct_stage_oracle_test: SKIP (no CUDA device available)\n";
    return 0;
  }

  const Nvfp4ScaleLayout scale_layout_kind =
      ResolveActivationNvfp4ScaleLayout(kRows, std::nullopt);
  const auto scale_layout = BuildNvfp4ExecutionScaleLayout(kRows, kCols, scale_layout_kind);
  if (!expect(scale_layout.has_value() && scale_layout->valid(),
              "execution scale layout should be valid for a 32x128 tile")) {
    return 1;
  }

  const std::vector<float> input = make_input();
  const std::vector<float> per_row_tensor_scales = make_non_uniform_per_row_scales();
  const std::vector<int> valid_rows_cases = {kRows, 1, 7, 31};

  for (int valid_rows : valid_rows_cases) {
    if (!test_p13_generic_direct_stage_contract(
            input,
            per_row_tensor_scales,
            valid_rows,
            *scale_layout)) {
      return 1;
    }
  }

  std::cout << "p13_generic_direct_stage_oracle_test: PASS\n";
  return 0;
}
