// Self-contained host-side fp64 oracle for NVFP4 swap-false FC1 GEMM.
//
// Purpose: independent ground truth that does not depend on any GPU kernel,
// so we can detect bugs in either the BF16 reference path or the new direct
// FP4 kernel without trusting either of them as a reference.
//
// Algorithm matches what TRT-LLM / vLLM do at the math level for
// `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3`:
//   - dequant value = decode_fp4(nibble) * decode_fp8(per_block_scale) * per_tensor_scale
//   - GEMM(M, N, K) computed in fp64
//   - per_tensor scales already folded into the dequant (matches our project's
//     PackRowMajorFp32ToNvfp4 quantization formula)

#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include "nemotron/nvfp4_packing.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

float DecodeFp4Host(std::uint8_t raw_nibble) {
  __nv_fp4_e2m1 value;
  value.__x = raw_nibble & 0x0Fu;
  return static_cast<float>(value);
}

float DecodeFp8Host(std::uint8_t raw_byte) {
  __nv_fp8_e4m3 value;
  value.__x = raw_byte;
  return static_cast<float>(value);
}

std::vector<float> DequantizeNvfp4Matrix(
    const std::uint8_t* packed,
    const std::uint8_t* block_scales,
    float tensor_scale,
    std::size_t rows,
    std::size_t cols) {
  std::vector<float> output(rows * cols, 0.0f);
  if (packed == nullptr || block_scales == nullptr || cols == 0 || cols % 16u != 0) {
    return output;
  }
  std::size_t packed_index = 0;
  std::size_t scale_index = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t block = 0; block < cols / 16u; ++block) {
      const float effective_scale =
          DecodeFp8Host(block_scales[scale_index++]) * tensor_scale;
      const std::size_t col_start = block * 16u;
      for (std::size_t offset = 0; offset < 16u; offset += 2u) {
        const std::uint8_t byte = packed[packed_index++];
        output[row * cols + col_start + offset + 0u] =
            DecodeFp4Host(byte & 0x0Fu) * effective_scale;
        output[row * cols + col_start + offset + 1u] =
            DecodeFp4Host((byte >> 4) & 0x0Fu) * effective_scale;
      }
    }
  }
  return output;
}

// Host fp64 GEMM reference for FP4 row-major activation × col-major-of-N weight.
//   A: M × K dequantized fp32, row-major (a_dq[m*K + k])
//   B: N × K dequantized fp32, row-major (b_dq[n*K + k])  -- weight stored as
//      one row per output dim N, K columns each, matching FC1's contract.
//   C: M × N row-major (c[m*N + n] = sum_k A[m,k] * B[n,k])
// Computed in fp64 to keep accumulator precision well above FP4 quant noise.
std::vector<float> ComputeFp4MatVecHostReference(
    const std::vector<float>& a_dq,
    const std::vector<float>& b_dq,
    std::size_t m,
    std::size_t n,
    std::size_t k) {
  std::vector<float> output(m * n, 0.0f);
  if (a_dq.size() != m * k || b_dq.size() != n * k) {
    return output;
  }
  for (std::size_t mi = 0; mi < m; ++mi) {
    const float* a_row = a_dq.data() + mi * k;
    for (std::size_t ni = 0; ni < n; ++ni) {
      const float* b_row = b_dq.data() + ni * k;
      double acc = 0.0;
      for (std::size_t ki = 0; ki < k; ++ki) {
        acc += static_cast<double>(a_row[ki]) * static_cast<double>(b_row[ki]);
      }
      output[mi * n + ni] = static_cast<float>(acc);
    }
  }
  return output;
}

bool ExpectClose(
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    float abs_tolerance,
    float rel_tolerance,
    const char* label) {
  if (expected.size() != actual.size()) {
    std::cerr << "swap_false_fp4_oracle_test: " << label
              << " size mismatch expected=" << expected.size()
              << " actual=" << actual.size() << "\n";
    return false;
  }
  float max_abs_diff = 0.0f;
  std::size_t max_abs_index = 0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const float diff = std::fabs(expected[i] - actual[i]);
    if (diff > max_abs_diff) {
      max_abs_diff = diff;
      max_abs_index = i;
    }
    const float allowed = abs_tolerance + rel_tolerance * std::fabs(expected[i]);
    if (diff > allowed) {
      std::cerr << "swap_false_fp4_oracle_test: " << label
                << " mismatch at index=" << i
                << " expected=" << expected[i] << " actual=" << actual[i]
                << " diff=" << diff << " allowed=" << allowed << "\n";
      return false;
    }
  }
  std::cout << "swap_false_fp4_oracle_test: " << label
            << " ok max_abs_diff=" << max_abs_diff
            << " at index=" << max_abs_index << "\n";
  return true;
}

std::vector<float> MakeRandomMatrix(std::size_t rows, std::size_t cols, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
  std::vector<float> values(rows * cols);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = dist(rng);
  }
  return values;
}

bool test_dequant_known_values() {
  // Pack a single block of 16 known values that exactly hit FP4 representable
  // points so the dequant should be lossless. e2m1 representable magnitudes:
  // {0, 0.5, 1, 1.5, 2, 3, 4, 6}, both signs.
  // With these values, block_max = 6, block_scale ≈ 6 / (kFp4MaxFinite=6)
  // = 1.0, and the round-trip should be exact.
  constexpr std::size_t kRows = 1;
  constexpr std::size_t kCols = 16;
  const std::vector<float> values = {
      0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
      -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
  };
  const auto packed = nemotron::PackRowMajorFp32ToNvfp4(values.data(), kRows, kCols);
  if (!packed.has_value() || !packed->valid()) {
    std::cerr << "swap_false_fp4_oracle_test: known_values pack failed\n";
    return false;
  }
  const auto dequant = DequantizeNvfp4Matrix(
      packed->packed_data(),
      packed->block_scales_data(),
      packed->tensor_scale,
      kRows,
      kCols);
  return ExpectClose(values, dequant, 1.0e-6f, 0.0f, "dequant_known_values");
}

bool test_dequant_roundtrip() {
  // Pack a random fp32 matrix to FP4 and dequantize via the host helper. The
  // round-trip should reproduce the original values within FP4 quantization
  // noise. This is a sanity check that the dequant formula matches the packer
  // semantically; tolerances are loose enough that any structural bug (sign
  // flip, off-by-one block index, swapped scale, etc.) will still trip them.
  constexpr std::size_t kRows = 4;
  constexpr std::size_t kCols = 32;
  const auto values = MakeRandomMatrix(kRows, kCols, 0xC001CAFEull);
  const auto packed = nemotron::PackRowMajorFp32ToNvfp4(values.data(), kRows, kCols);
  if (!packed.has_value() || !packed->valid()) {
    std::cerr << "swap_false_fp4_oracle_test: roundtrip pack failed\n";
    return false;
  }
  const auto dequant = DequantizeNvfp4Matrix(
      packed->packed_data(),
      packed->block_scales_data(),
      packed->tensor_scale,
      kRows,
      kCols);
  return ExpectClose(values, dequant, 0.5f, 0.5f, "dequant_roundtrip");
}

bool test_host_gemm_matches_fp32_gemm() {
  // Random FC1-shaped GEMM at small dims. Pack both A and B to FP4, dequantize,
  // run host fp64 GEMM, compare against a fp32 GEMM on the original (not
  // quantized) values. The diff should be bounded by accumulated FP4
  // quantization noise; the bound here is generous because the point is to
  // catch *structural* errors in the host GEMM (e.g., wrong indexing, swapped
  // M/N), not to characterize quant noise.
  constexpr std::size_t kM = 4;
  constexpr std::size_t kN = 8;
  constexpr std::size_t kK = 256;
  const auto a_fp32 = MakeRandomMatrix(kM, kK, 0xA11CE001ull);
  const auto b_fp32 = MakeRandomMatrix(kN, kK, 0xB0B5BABEull);

  std::vector<float> expected_fp32(kM * kN, 0.0f);
  for (std::size_t mi = 0; mi < kM; ++mi) {
    for (std::size_t ni = 0; ni < kN; ++ni) {
      double acc = 0.0;
      for (std::size_t ki = 0; ki < kK; ++ki) {
        acc += static_cast<double>(a_fp32[mi * kK + ki])
             * static_cast<double>(b_fp32[ni * kK + ki]);
      }
      expected_fp32[mi * kN + ni] = static_cast<float>(acc);
    }
  }

  const auto a_packed = nemotron::PackRowMajorFp32ToNvfp4(a_fp32.data(), kM, kK);
  const auto b_packed = nemotron::PackRowMajorFp32ToNvfp4(b_fp32.data(), kN, kK);
  if (!a_packed.has_value() || !b_packed.has_value()) {
    std::cerr << "swap_false_fp4_oracle_test: gemm pack failed\n";
    return false;
  }
  const auto a_dq = DequantizeNvfp4Matrix(
      a_packed->packed_data(), a_packed->block_scales_data(),
      a_packed->tensor_scale, kM, kK);
  const auto b_dq = DequantizeNvfp4Matrix(
      b_packed->packed_data(), b_packed->block_scales_data(),
      b_packed->tensor_scale, kN, kK);
  const auto actual = ComputeFp4MatVecHostReference(a_dq, b_dq, kM, kN, kK);

  // sqrt(256) ~= 16; per-term noise (a*b) ~= 0.25 in this magnitude regime;
  // expected accumulated noise ~= 16 * 0.25 ~= 4 in magnitude space, so 8
  // absolute is comfortably above noise but well below any structural bug.
  return ExpectClose(expected_fp32, actual, 8.0f, 0.5f, "host_gemm_vs_fp32_gemm");
}

}  // namespace

int main() {
  bool ok = true;
  ok = test_dequant_known_values() && ok;
  ok = test_dequant_roundtrip() && ok;
  ok = test_host_gemm_matches_fp32_gemm() && ok;
  if (!ok) {
    std::cerr << "swap_false_fp4_oracle_test: FAILED\n";
    return 1;
  }
  std::cout << "swap_false_fp4_oracle_test: PASS\n";
  return 0;
}
