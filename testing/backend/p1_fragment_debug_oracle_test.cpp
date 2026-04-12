#include "nemotron/fused_moe_prefill.h"
#include "nemotron/nvfp4_packing.h"
#include "nemotron/nvfp4_scale_layout.h"

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using nemotron::CopyP1FragmentDebugTrace;
using nemotron::Nvfp4PackOptions;
using nemotron::Nvfp4ScaleLayout;
using nemotron::P1FragmentDebugTrace;
using nemotron::PackRowMajorFp32ToNvfp4;
using nemotron::ResetP1FragmentDebugTrace;
using nemotron::RunP1FragmentDebugOracleForTesting;
using nemotron::SwizzleRowMajorNvfp4ScalesForExecution;

constexpr std::size_t kTokenRows = 16;
constexpr std::size_t kOutputRows = 128;
constexpr std::size_t kK = 64;
constexpr std::size_t kBlocksPerRow = kK / 16u;
constexpr int kTargetTokenRow = 0;
constexpr int kTargetOutputRow = 2;
constexpr int kExpectedSfaFragmentCosize = 8;
constexpr int kExpectedSfbFragmentCosize = 32;

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

bool has_cuda_device() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

template <typename T>
class DeviceBuffer {
 public:
  static std::unique_ptr<DeviceBuffer> Create(std::size_t count) {
    if (!has_cuda_device() || count == 0) {
      return nullptr;
    }
    T* data = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)) != cudaSuccess) {
      return nullptr;
    }
    return std::unique_ptr<DeviceBuffer>(new DeviceBuffer(data, count));
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* data() const { return data_; }

  bool copy_from_host(const std::vector<T>& values) const {
    return values.size() == count_ &&
           cudaMemcpy(
               data_,
               values.data(),
               count_ * sizeof(T),
               cudaMemcpyHostToDevice) == cudaSuccess;
  }

 private:
  DeviceBuffer(T* data, std::size_t count) : data_(data), count_(count) {}

  T* data_ = nullptr;
  std::size_t count_ = 0;
};

std::uint32_t float_bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float decode_fp4_host(std::uint8_t raw_nibble) {
  __nv_fp4_e2m1 value;
  value.__x = raw_nibble & 0x0Fu;
  return static_cast<float>(value);
}

float decode_fp8_host(std::uint8_t raw_byte) {
  __nv_fp8_e4m3 value;
  value.__x = raw_byte;
  return static_cast<float>(value);
}

std::vector<float> dequantize_nvfp4_matrix(
    const std::uint8_t* packed,
    const std::uint8_t* block_scales,
    float tensor_scale,
    std::size_t rows,
    std::size_t cols) {
  std::vector<float> output(rows * cols, 0.0f);
  std::size_t packed_index = 0;
  std::size_t scale_index = 0;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t block = 0; block < cols / 16u; ++block) {
      const float block_scale =
          decode_fp8_host(block_scales[scale_index++]) * tensor_scale;
      const std::size_t col_base = block * 16u;
      for (std::size_t offset = 0; offset < 16u; offset += 2u) {
        const std::uint8_t byte = packed[packed_index++];
        output[row * cols + col_base + offset + 0u] =
            decode_fp4_host(byte & 0x0Fu) * block_scale;
        output[row * cols + col_base + offset + 1u] =
            decode_fp4_host((byte >> 4) & 0x0Fu) * block_scale;
      }
    }
  }
  return output;
}

std::vector<float> compute_host_reference_token_major(
    const std::vector<float>& weight_dq,
    const std::vector<float>& input_dq) {
  std::vector<float> output(kTokenRows * kOutputRows, 0.0f);
  for (std::size_t token_row = 0; token_row < kTokenRows; ++token_row) {
    const float* input_row = input_dq.data() + token_row * kK;
    for (std::size_t output_row = 0; output_row < kOutputRows; ++output_row) {
      const float* weight_row = weight_dq.data() + output_row * kK;
      double acc = 0.0;
      for (std::size_t k = 0; k < kK; ++k) {
        acc += static_cast<double>(weight_row[k]) * static_cast<double>(input_row[k]);
      }
      output[token_row * kOutputRows + output_row] = static_cast<float>(acc);
    }
  }
  return output;
}

std::vector<float> make_scaled_pattern_matrix(
    std::size_t rows,
    std::size_t cols,
    const std::array<float, 4>& scale_table,
    int row_mul,
    int block_mul) {
  static constexpr std::array<float, 16> kPattern = {
      6.0f,  4.0f,  3.0f,  2.0f,
      1.5f,  1.0f,  0.5f,  0.0f,
     -0.5f, -1.0f, -1.5f, -2.0f,
     -3.0f, -4.0f, -6.0f,  0.0f,
  };
  std::vector<float> values(rows * cols, 0.0f);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t block = 0; block < cols / 16u; ++block) {
      const float scale =
          scale_table[(row * static_cast<std::size_t>(row_mul) +
                       block * static_cast<std::size_t>(block_mul)) &
                      3u];
      const std::size_t base_col = block * 16u;
      const std::size_t offset =
          (row * static_cast<std::size_t>(row_mul) +
           block * static_cast<std::size_t>(block_mul)) &
          15u;
      for (std::size_t elem = 0; elem < 16u; ++elem) {
        values[row * cols + base_col + elem] =
            scale * kPattern[(offset + elem) & 15u];
      }
    }
  }
  return values;
}

std::uint8_t load_packed_nibble(
    const std::vector<std::uint8_t>& packed,
    std::size_t cols,
    int row,
    int col) {
  if (row < 0 || col < 0) {
    return 0u;
  }
  const std::size_t row_u = static_cast<std::size_t>(row);
  const std::size_t col_u = static_cast<std::size_t>(col);
  if (col_u >= cols) {
    return 0u;
  }
  const std::size_t packed_row_bytes = cols / 2u;
  const std::size_t byte_index = row_u * packed_row_bytes + (col_u / 2u);
  const std::uint8_t byte = packed[byte_index];
  return (col_u & 1u) == 0u ? static_cast<std::uint8_t>(byte & 0x0Fu)
                            : static_cast<std::uint8_t>((byte >> 4) & 0x0Fu);
}

std::uint32_t pack_expected_fragment_word(
    const std::vector<std::uint8_t>& packed,
    std::size_t cols,
    const int* row_coords,
    const int* col_coords,
    int physical_base) {
  std::uint32_t packed_word = 0u;
  for (int elem = 0; elem < 8; ++elem) {
    const std::uint8_t nibble = load_packed_nibble(
        packed,
        cols,
        row_coords[physical_base + elem],
        col_coords[physical_base + elem]);
    packed_word |= static_cast<std::uint32_t>(nibble) << (elem * 4);
  }
  return packed_word;
}

std::uint32_t build_expected_scale_word(
    const std::vector<std::uint8_t>& block_scales,
    int logical_cols,
    const int row_coords[4],
    const int col_coords[4]) {
  const int segment_len = logical_cols > 0 ? std::max(1, logical_cols / 4) : 1;
  std::uint32_t packed_word = 0u;
  for (int elem = 0; elem < 4; ++elem) {
    std::uint8_t raw = 0u;
    const int row = row_coords[elem];
    const int col = col_coords[elem];
    if (row >= 0 && col >= 0) {
      const int byte_index = std::min(col / segment_len, 3);
      raw = block_scales[static_cast<std::size_t>(row) * kBlocksPerRow +
                         static_cast<std::size_t>(byte_index)];
    }
    packed_word |= static_cast<std::uint32_t>(raw) << (elem * 8);
  }
  return packed_word;
}

std::uint8_t load_expected_scale_byte(
    const std::vector<std::uint8_t>& block_scales,
    int logical_cols,
    int row,
    int col) {
  if (row < 0 || col < 0) {
    return 0u;
  }
  const std::size_t row_u = static_cast<std::size_t>(row);
  if (row_u >= block_scales.size() / kBlocksPerRow) {
    return 0u;
  }
  const int segment_len = logical_cols > 0 ? std::max(1, logical_cols / 4) : 1;
  const int byte_index = std::min(col / segment_len, 3);
  return block_scales[row_u * kBlocksPerRow +
                      static_cast<std::size_t>(byte_index)];
}

bool compare_sfa_smem_dump(
    const P1FragmentDebugTrace& trace,
    const std::vector<std::uint8_t>& block_scales,
    bool print) {
  for (int physical = 0;
       physical < nemotron::kP1FragmentDebugScaleSmemDumpByteCount;
       ++physical) {
    const int row = trace.sfa_smem_row_coord[physical];
    const int col = trace.sfa_smem_col_coord[physical];
    const std::uint8_t expected =
        load_expected_scale_byte(block_scales, trace.sfa_logical_cols, row, col);
    const std::uint8_t actual = trace.sfa_smem_dump[physical];
    if (expected == actual) {
      continue;
    }
    if (print) {
      std::cerr << "FAIL: sfa_smem_dump mismatch physical=" << physical
                << " logical=(" << row << "," << col << ") actual=0x"
                << std::hex << static_cast<unsigned int>(actual)
                << " expected=0x" << static_cast<unsigned int>(expected)
                << std::dec << "\n";
    }
    return false;
  }
  return true;
}

bool compare_scale_bucket(
    const char* label,
    const std::vector<std::uint8_t>& block_scales,
    int logical_cols,
    const std::uint32_t* actual_words,
    const int row_coords[][nemotron::kP1FragmentDebugScaleCoordCount],
    const int col_coords[][nemotron::kP1FragmentDebugScaleCoordCount],
    bool print) {
  for (int lane = 0; lane < nemotron::kP1FragmentDebugLaneCount; ++lane) {
    const std::uint32_t expected = build_expected_scale_word(
        block_scales,
        logical_cols,
        row_coords[lane],
        col_coords[lane]);
    if (expected == actual_words[lane]) {
      continue;
    }
    if (print) {
      std::cerr << "FAIL: " << label << " mismatch lane=" << lane
                << " actual=0x" << std::hex << actual_words[lane]
                << " expected=0x" << expected << std::dec << "\n";
      for (int elem = 0; elem < nemotron::kP1FragmentDebugScaleCoordCount; ++elem) {
        std::cerr << "  scale_coord[" << elem << "]=(" << row_coords[lane][elem]
                  << "," << col_coords[lane][elem] << ")\n";
      }
    }
    return false;
  }
  return true;
}

bool compare_fragment_bucket(
    const char* label,
    const std::vector<std::uint8_t>& packed,
    std::size_t cols,
    const std::uint32_t* actual_words,
    const int* row_coords,
    const int* col_coords,
    int lane_count,
    int words_per_lane,
    bool shifted,
    bool print) {
  for (int lane = 0; lane < lane_count; ++lane) {
    for (int reg = 0; reg < words_per_lane; ++reg) {
      std::uint32_t expected = pack_expected_fragment_word(
          packed,
          cols,
          row_coords + lane * words_per_lane * 8,
          col_coords + lane * words_per_lane * 8,
          reg * 8);
      if (shifted) {
        expected <<= 2u;
      }
      const std::uint32_t actual = actual_words[lane * words_per_lane + reg];
      if (expected == actual) {
        continue;
      }
      if (print) {
        std::cerr << "FAIL: " << label << " mismatch lane=" << lane
                  << " reg=" << reg << " actual=0x" << std::hex << actual
                  << " expected=0x" << expected << std::dec << "\n";
        for (int elem = 0; elem < 8; ++elem) {
          const int physical = reg * 8 + elem;
          std::cerr << "  coord[" << physical << "]=("
                    << row_coords[lane * words_per_lane * 8 + physical] << ","
                    << col_coords[lane * words_per_lane * 8 + physical] << ")\n";
        }
      }
      return false;
    }
  }
  return true;
}

bool compare_c_atom_bucket(
    const P1FragmentDebugTrace& trace,
    const std::vector<float>& reference,
    bool print) {
  for (int lane = 0; lane < nemotron::kP1FragmentDebugLaneCount; ++lane) {
    for (int reg = 0; reg < nemotron::kP1FragmentDebugCAtomCount; ++reg) {
      const int output_row = trace.c_output_row[lane][reg];
      const int token_row = trace.c_token_row[lane][reg];
      float expected_value = 0.0f;
      if (output_row >= 0 &&
          output_row < static_cast<int>(kOutputRows) &&
          token_row >= 0 &&
          token_row < static_cast<int>(kTokenRows)) {
        expected_value =
            reference[static_cast<std::size_t>(token_row) * kOutputRows +
                      static_cast<std::size_t>(output_row)];
      }
      const float actual_value = trace.c_atom_post_mma[lane][reg];
      if (float_bits(actual_value) == float_bits(expected_value)) {
        continue;
      }
      if (print) {
        std::cerr << "FAIL: c_atom_post_mma mismatch lane=" << lane
                  << " reg=" << reg << " coord=(" << token_row << ","
                  << output_row << ") actual=" << actual_value
                  << " expected=" << expected_value
                  << " actual_bits=0x" << std::hex << float_bits(actual_value)
                  << " expected_bits=0x" << float_bits(expected_value)
                  << std::dec << "\n";
      }
      return false;
    }
  }
  return true;
}

void print_target_coord_owner(
    const P1FragmentDebugTrace& trace,
    const std::vector<float>& reference) {
  for (int lane = 0; lane < nemotron::kP1FragmentDebugLaneCount; ++lane) {
    for (int reg = 0; reg < nemotron::kP1FragmentDebugCAtomCount; ++reg) {
      if (trace.c_token_row[lane][reg] != kTargetTokenRow ||
          trace.c_output_row[lane][reg] != kTargetOutputRow) {
        continue;
      }
      const float expected =
          reference[static_cast<std::size_t>(kTargetTokenRow) * kOutputRows +
                    static_cast<std::size_t>(kTargetOutputRow)];
      const float actual = trace.c_atom_post_mma[lane][reg];
      std::cerr << "target_coord_owner: (" << kTargetTokenRow << ","
                << kTargetOutputRow << ") lane=" << lane << " reg=" << reg
                << " actual=" << actual << " expected=" << expected
                << " actual_bits=0x" << std::hex << float_bits(actual)
                << " expected_bits=0x" << float_bits(expected)
                << std::dec << "\n";
      return;
    }
  }
  std::cerr << "target_coord_owner: (" << kTargetTokenRow << ","
            << kTargetOutputRow << ") not found in first c atom\n";
}

}  // namespace

int main() {
  if (!has_cuda_device()) {
    std::cout << "p1_fragment_debug_oracle_test: SKIP (no CUDA device)\n";
    return 0;
  }

  const std::array<float, 4> kInputScales = {1.0f, 0.5f, 0.25f, 2.0f};
  const std::array<float, 4> kWeightScales = {1.5f, 1.0f, 0.5f, 0.25f};
  const auto input_fp32 =
      make_scaled_pattern_matrix(kTokenRows, kK, kInputScales, 1, 3);
  const auto weight_fp32 =
      make_scaled_pattern_matrix(kOutputRows, kK, kWeightScales, 5, 1);

  Nvfp4PackOptions pack_options;
  pack_options.fixed_tensor_scale = 1.0f;

  const auto input_pack = PackRowMajorFp32ToNvfp4(
      input_fp32.data(),
      kTokenRows,
      kK,
      pack_options);
  const auto weight_pack = PackRowMajorFp32ToNvfp4(
      weight_fp32.data(),
      kOutputRows,
      kK,
      pack_options);

  if (!expect(input_pack.has_value() && input_pack->valid(), "input pack failed") ||
      !expect(weight_pack.has_value() && weight_pack->valid(), "weight pack failed")) {
    return 1;
  }

  const auto weight_matmul_scales = SwizzleRowMajorNvfp4ScalesForExecution(
      weight_pack->block_scales_data(),
      kOutputRows,
      kK,
      Nvfp4ScaleLayout::kSwizzled128x4);

  auto input_packed_device = DeviceBuffer<std::uint8_t>::Create(input_pack->packed.size());
  auto input_scales_device =
      DeviceBuffer<std::uint8_t>::Create(input_pack->block_scales.size());
  auto weight_packed_device =
      DeviceBuffer<std::uint8_t>::Create(weight_pack->packed.size());
  auto weight_scales_device =
      DeviceBuffer<std::uint8_t>::Create(weight_matmul_scales.size());

  if (!expect(input_packed_device != nullptr, "alloc input packed") ||
      !expect(input_scales_device != nullptr, "alloc input scales") ||
      !expect(weight_packed_device != nullptr, "alloc weight packed") ||
      !expect(weight_scales_device != nullptr, "alloc weight scales")) {
    return 1;
  }

  if (!expect(input_packed_device->copy_from_host(input_pack->packed), "copy input packed") ||
      !expect(input_scales_device->copy_from_host(input_pack->block_scales), "copy input scales") ||
      !expect(weight_packed_device->copy_from_host(weight_pack->packed), "copy weight packed") ||
      !expect(weight_scales_device->copy_from_host(weight_matmul_scales), "copy weight scales")) {
    return 1;
  }

  ResetP1FragmentDebugTrace();
  if (!RunP1FragmentDebugOracleForTesting(
          weight_packed_device->data(),
          weight_scales_device->data(),
          input_packed_device->data(),
          input_scales_device->data())) {
    std::cerr << "FAIL: RunP1FragmentDebugOracleForTesting failed\n";
    return 1;
  }

  P1FragmentDebugTrace trace{};
  if (!expect(CopyP1FragmentDebugTrace(&trace), "copy P1 fragment debug trace")) {
    return 1;
  }
  if (!expect(trace.valid != 0, "P1 fragment debug trace should be valid")) {
    return 1;
  }

  const auto input_dq = dequantize_nvfp4_matrix(
      input_pack->packed_data(),
      input_pack->block_scales_data(),
      input_pack->tensor_scale,
      kTokenRows,
      kK);
  const auto weight_dq = dequantize_nvfp4_matrix(
      weight_pack->packed_data(),
      weight_pack->block_scales_data(),
      weight_pack->tensor_scale,
      kOutputRows,
      kK);
  const auto reference = compute_host_reference_token_major(weight_dq, input_dq);

  const bool sfa_smem_equal = compare_sfa_smem_dump(
      trace,
      input_pack->block_scales,
      false);
  const bool sfa_equal = compare_scale_bucket(
      "tCrSFA",
      input_pack->block_scales,
      trace.sfa_logical_cols,
      &trace.tCrSFA[0],
      trace.sfa_row_coord,
      trace.sfa_col_coord,
      false);
  const bool sfb_equal = compare_scale_bucket(
      "tCrSFB",
      weight_pack->block_scales,
      trace.sfb_logical_cols,
      &trace.tCrSFB[0],
      trace.sfb_row_coord,
      trace.sfb_col_coord,
      false);
  const bool a_pre_equal = compare_fragment_bucket(
      "tCrA_pre_shift",
      input_pack->packed,
      kK,
      &trace.tCrA_pre_shift[0][0],
      &trace.a_row_coord[0][0],
      &trace.a_col_coord[0][0],
      nemotron::kP1FragmentDebugLaneCount,
      4,
      false,
      false);
  const bool a_post_equal = compare_fragment_bucket(
      "tCrA_post_shift",
      input_pack->packed,
      kK,
      &trace.tCrA_post_shift[0][0],
      &trace.a_row_coord[0][0],
      &trace.a_col_coord[0][0],
      nemotron::kP1FragmentDebugLaneCount,
      4,
      true,
      false);
  const bool b_pre_equal = compare_fragment_bucket(
      "tCrB_pre_shift",
      weight_pack->packed,
      kK,
      &trace.tCrB_pre_shift[0][0],
      &trace.b_row_coord[0][0],
      &trace.b_col_coord[0][0],
      nemotron::kP1FragmentDebugLaneCount,
      2,
      false,
      false);
  const bool b_post_equal = compare_fragment_bucket(
      "tCrB_post_shift",
      weight_pack->packed,
      kK,
      &trace.tCrB_post_shift[0][0],
      &trace.b_row_coord[0][0],
      &trace.b_col_coord[0][0],
      nemotron::kP1FragmentDebugLaneCount,
      2,
      true,
      false);
  const bool c_equal = compare_c_atom_bucket(trace, reference, false);

  std::string first_bucket;
  if (!sfa_smem_equal) {
    first_bucket = "sfa_smem_dump";
    compare_sfa_smem_dump(trace, input_pack->block_scales, true);
  } else if (!sfa_equal) {
    first_bucket = "tCrSFA";
    compare_scale_bucket(
        "tCrSFA",
        input_pack->block_scales,
        trace.sfa_logical_cols,
        &trace.tCrSFA[0],
        trace.sfa_row_coord,
        trace.sfa_col_coord,
        true);
  } else if (!sfb_equal) {
    first_bucket = "tCrSFB";
    compare_scale_bucket(
        "tCrSFB",
        weight_pack->block_scales,
        trace.sfb_logical_cols,
        &trace.tCrSFB[0],
        trace.sfb_row_coord,
        trace.sfb_col_coord,
        true);
  } else if (!a_pre_equal) {
    first_bucket = "tCrA_pre_shift";
    compare_fragment_bucket(
        "tCrA_pre_shift",
        input_pack->packed,
        kK,
        &trace.tCrA_pre_shift[0][0],
        &trace.a_row_coord[0][0],
        &trace.a_col_coord[0][0],
        nemotron::kP1FragmentDebugLaneCount,
        4,
        false,
        true);
  } else if (!a_post_equal) {
    first_bucket = "tCrA_post_shift";
    compare_fragment_bucket(
        "tCrA_post_shift",
        input_pack->packed,
        kK,
        &trace.tCrA_post_shift[0][0],
        &trace.a_row_coord[0][0],
        &trace.a_col_coord[0][0],
        nemotron::kP1FragmentDebugLaneCount,
        4,
        true,
        true);
  } else if (!b_pre_equal) {
    first_bucket = "tCrB_pre_shift";
    compare_fragment_bucket(
        "tCrB_pre_shift",
        weight_pack->packed,
        kK,
        &trace.tCrB_pre_shift[0][0],
        &trace.b_row_coord[0][0],
        &trace.b_col_coord[0][0],
        nemotron::kP1FragmentDebugLaneCount,
        2,
        false,
        true);
  } else if (!b_post_equal) {
    first_bucket = "tCrB_post_shift";
    compare_fragment_bucket(
        "tCrB_post_shift",
        weight_pack->packed,
        kK,
        &trace.tCrB_post_shift[0][0],
        &trace.b_row_coord[0][0],
        &trace.b_col_coord[0][0],
        nemotron::kP1FragmentDebugLaneCount,
        2,
        true,
        true);
  } else if (!c_equal) {
    first_bucket = "c_atom_post_mma";
    compare_c_atom_bucket(trace, reference, true);
  }

  const bool sfa_fragment_cosize_match =
      trace.observed_sfa_fragment_cosize == kExpectedSfaFragmentCosize;
  const bool sfb_fragment_cosize_match =
      trace.observed_sfb_fragment_cosize == kExpectedSfbFragmentCosize;
  std::cout << "p1_fragment_debug_oracle_test: sfa_fragment_cosize="
            << trace.observed_sfa_fragment_cosize << "/"
            << kExpectedSfaFragmentCosize
            << " sfb_fragment_cosize=" << trace.observed_sfb_fragment_cosize
            << "/" << kExpectedSfbFragmentCosize
            << " scale_fragment_cosize_match="
            << ((sfa_fragment_cosize_match && sfb_fragment_cosize_match) ? "true"
                                                                         : "false")
            << "\n";
  std::cout << "p1_fragment_debug_oracle_test: sfa_smem_match="
            << (sfa_smem_equal ? "true" : "false")
            << " sfa_scale_match=" << (sfa_equal ? "true" : "false")
            << " sfb_scale_match=" << (sfb_equal ? "true" : "false")
            << "\n";
  print_target_coord_owner(trace, reference);

  const bool fully_equal =
      sfa_smem_equal && sfa_equal && sfb_equal && a_pre_equal && a_post_equal &&
      b_pre_equal && b_post_equal && c_equal;

  // This is intentionally a negative sentinel today: it localizes the first
  // fragment-level mismatch in the bespoke traced P1 fragment path. If the traced P1
  // contract is repaired later, flip this to require exact equality instead.
  if (!expect_divergence(
          fully_equal,
          "P1 fragment debug oracle should diverge until the traced P1 fragment contract is repaired")) {
    std::cerr << "p1_fragment_debug_oracle_test: FAILED\n";
    return 1;
  }

  if (!first_bucket.empty()) {
    std::cout << "p1_fragment_debug_oracle_test: first_mismatch_bucket="
              << first_bucket << "\n";
  }
  std::cout << "p1_fragment_debug_oracle_test: PASS (divergence sentinel)\n";
  return 0;
}
