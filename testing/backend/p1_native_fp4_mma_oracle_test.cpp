#include "nemotron/fused_moe_prefill.h"
#include "nemotron/nvfp4_packing.h"
#include "nemotron/nvfp4_scale_layout.h"

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using nemotron::CopyP1NativeFp4MmaTrace;
using nemotron::Nvfp4PackOptions;
using nemotron::Nvfp4ScaleLayout;
using nemotron::P1NativeFp4MmaTrace;
using nemotron::PackRowMajorFp32ToNvfp4;
using nemotron::ResetP1NativeFp4MmaTrace;
using nemotron::RunP1NativeFp4MmaOracleForTesting;
using nemotron::SwizzleRowMajorNvfp4ScalesForExecution;

constexpr std::size_t kRows = 16;
constexpr std::size_t kCols = 128;
constexpr std::size_t kK = 64;
constexpr std::size_t kBlocksPerRow = kK / 16u;
constexpr std::size_t kInputElementCount = kRows * kK;
constexpr std::size_t kWeightElementCount = kCols * kK;
constexpr std::size_t kOutputElementCount = kRows * kCols;

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
  std::size_t count() const { return count_; }

  bool copy_from_host(const std::vector<T>& values) const {
    return values.size() == count_ &&
           cudaMemcpy(
               data_,
               values.data(),
               count_ * sizeof(T),
               cudaMemcpyHostToDevice) == cudaSuccess;
  }

  bool copy_to_host(std::vector<T>* values) const {
    if (values == nullptr) {
      return false;
    }
    values->assign(count_, T{});
    return cudaMemcpy(
               values->data(),
               data_,
               count_ * sizeof(T),
               cudaMemcpyDeviceToHost) == cudaSuccess;
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

std::vector<float> compute_host_reference(
    const std::vector<float>& a_dq,
    const std::vector<float>& b_dq) {
  std::vector<float> output(kOutputElementCount, 0.0f);
  for (std::size_t row = 0; row < kRows; ++row) {
    const float* a_row = a_dq.data() + row * kK;
    for (std::size_t col = 0; col < kCols; ++col) {
      const float* b_row = b_dq.data() + col * kK;
      double acc = 0.0;
      for (std::size_t k = 0; k < kK; ++k) {
        acc += static_cast<double>(a_row[k]) * static_cast<double>(b_row[k]);
      }
      output[row * kCols + col] = static_cast<float>(acc);
    }
  }
  return output;
}

std::vector<float> make_exact_tile(std::size_t rows, std::size_t cols, int row_mul, int block_mul) {
  static constexpr std::array<float, 16> kPattern = {
      6.0f,  4.0f,  3.0f,  2.0f,
      1.5f,  1.0f,  0.5f,  0.0f,
     -0.5f, -1.0f, -1.5f, -2.0f,
     -3.0f, -4.0f, -6.0f,  0.0f,
  };
  std::vector<float> values(rows * cols, 0.0f);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      const std::size_t block_offset = col & 15u;
      const std::size_t block_index = col / 16u;
      const std::size_t pattern_index =
          (block_offset + row * static_cast<std::size_t>(row_mul) +
           block_index * static_cast<std::size_t>(block_mul)) &
          15u;
      values[row * cols + col] = kPattern[pattern_index];
    }
  }
  return values;
}

void print_trace_for_coord(
    const P1NativeFp4MmaTrace& trace,
    int token_row,
    int output_col) {
  bool found = false;
  for (int slot = 0; slot < nemotron::kP1NativeFp4MmaTraceEntryCount; ++slot) {
    if (trace.token_rows[slot] != token_row || trace.output_cols[slot] != output_col) {
      continue;
    }
    found = true;
    std::cerr << "trace: coord=(" << token_row << "," << output_col << ") warp="
              << trace.warp_ids[slot] << " subtile=" << trace.subtile_ids[slot]
              << " lane=" << trace.lane_ids[slot] << " reg=" << trace.reg_ids[slot]
              << " physical=" << trace.physical_indices[slot] << " value="
              << trace.values[slot] << " value_bits=0x" << std::hex
              << float_bits(trace.values[slot]) << std::dec << "\n";
  }
  if (!found) {
    std::cerr << "trace: coord=(" << token_row << "," << output_col
              << ") no mapped trace entry\n";
  }
}

void print_group_anomalies(const P1NativeFp4MmaTrace& trace) {
  int printed = 0;
  for (int group = 0; group < nemotron::kP1NativeFp4MmaTraceGroupCount; ++group) {
    if (trace.group_match_counts[group] == 4) {
      continue;
    }
    std::cerr << "group: warp=" << trace.group_warp_ids[group]
              << " subtile=" << trace.group_subtile_ids[group]
              << " lane=" << trace.group_lane_ids[group]
              << " expected_output_col=" << trace.group_output_cols[group]
              << " match_count=" << trace.group_match_counts[group] << "\n";
    if (++printed == 16) {
      break;
    }
  }
}

bool compare_bitwise(
    const std::vector<float>& actual,
    const std::vector<float>& expected) {
  if (!expect(actual.size() == expected.size(), "output size mismatch")) {
    return false;
  }

  P1NativeFp4MmaTrace trace{};
  bool have_trace = CopyP1NativeFp4MmaTrace(&trace);
  bool ok = true;
  int printed = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (float_bits(actual[i]) == float_bits(expected[i])) {
      continue;
    }
    ok = false;
    const int token_row = static_cast<int>(i / kCols);
    const int output_col = static_cast<int>(i % kCols);
    std::cerr << "FAIL: output mismatch at (" << token_row << "," << output_col
              << ") actual=" << actual[i] << " expected=" << expected[i]
              << " actual_bits=0x" << std::hex << float_bits(actual[i])
              << " expected_bits=0x" << float_bits(expected[i]) << std::dec << "\n";
    if (have_trace) {
      print_trace_for_coord(trace, token_row, output_col);
    }
    if (++printed == 16) {
      break;
    }
  }

  if (!ok && have_trace) {
    print_group_anomalies(trace);
  }
  return ok;
}

}  // namespace

int main() {
  if (!has_cuda_device()) {
    std::cerr << "SKIP: no CUDA device\n";
    return 0;
  }

  const auto input_fp32 = make_exact_tile(kRows, kK, 3, 1);
  const auto weight_fp32 = make_exact_tile(kCols, kK, 5, 3);

  Nvfp4PackOptions pack_options;
  pack_options.fixed_tensor_scale = 1.0f;

  const auto input_pack = PackRowMajorFp32ToNvfp4(
      input_fp32.data(),
      kRows,
      kK,
      pack_options);
  const auto weight_pack = PackRowMajorFp32ToNvfp4(
      weight_fp32.data(),
      kCols,
      kK,
      pack_options);

  if (!expect(input_pack.has_value() && input_pack->valid(), "input pack failed") ||
      !expect(weight_pack.has_value() && weight_pack->valid(), "weight pack failed")) {
    return 1;
  }

  const auto weight_matmul_scales = SwizzleRowMajorNvfp4ScalesForExecution(
      weight_pack->block_scales_data(),
      kCols,
      kK,
      Nvfp4ScaleLayout::kSwizzled128x4);

  auto input_packed_device = DeviceBuffer<std::uint8_t>::Create(input_pack->packed.size());
  auto input_scales_device =
      DeviceBuffer<std::uint8_t>::Create(input_pack->block_scales.size());
  auto weight_packed_device =
      DeviceBuffer<std::uint8_t>::Create(weight_pack->packed.size());
  auto weight_scales_device =
      DeviceBuffer<std::uint8_t>::Create(weight_matmul_scales.size());
  auto dense_output_device = DeviceBuffer<float>::Create(kOutputElementCount);

  if (!expect(input_packed_device != nullptr, "alloc input packed") ||
      !expect(input_scales_device != nullptr, "alloc input scales") ||
      !expect(weight_packed_device != nullptr, "alloc weight packed") ||
      !expect(weight_scales_device != nullptr, "alloc weight matmul scales") ||
      !expect(dense_output_device != nullptr, "alloc dense output")) {
    return 1;
  }

  if (!expect(input_packed_device->copy_from_host(input_pack->packed), "copy input packed") ||
      !expect(input_scales_device->copy_from_host(input_pack->block_scales), "copy input scales") ||
      !expect(weight_packed_device->copy_from_host(weight_pack->packed), "copy weight packed") ||
      !expect(weight_scales_device->copy_from_host(weight_matmul_scales), "copy weight scales")) {
    return 1;
  }

  ResetP1NativeFp4MmaTrace();
  if (!RunP1NativeFp4MmaOracleForTesting(
          input_packed_device->data(),
          input_scales_device->data(),
          weight_packed_device->data(),
          weight_scales_device->data(),
          dense_output_device->data())) {
    std::cerr << "FAIL: RunP1NativeFp4MmaOracleForTesting failed\n";
    return 1;
  }

  std::vector<float> dense_output;
  if (!expect(dense_output_device->copy_to_host(&dense_output), "copy dense output")) {
    return 1;
  }

  const auto input_dq = dequantize_nvfp4_matrix(
      input_pack->packed_data(),
      input_pack->block_scales_data(),
      input_pack->tensor_scale,
      kRows,
      kK);
  const auto weight_dq = dequantize_nvfp4_matrix(
      weight_pack->packed_data(),
      weight_pack->block_scales_data(),
      weight_pack->tensor_scale,
      kCols,
      kK);
  const auto reference = compute_host_reference(input_dq, weight_dq);
  const bool equal = compare_bitwise(dense_output, reference);

  // This Step 0 oracle is intentionally a negative sentinel today: it proves
  // the current traced P1 native FP4 MMA chain does not satisfy a CUTE-derived
  // store contract. If the contract is repaired later, flip this check to
  // require equality instead of divergence.
  if (!expect_divergence(
          equal,
          "current traced P1 native FP4 MMA oracle should diverge until the P1 contract is repaired")) {
    std::cerr << "p1_native_fp4_mma_oracle_test: FAILED\n";
    return 1;
  }

  std::cout << "p1_native_fp4_mma_oracle_test: PASS (divergence sentinel)\n";
  return 0;
}
