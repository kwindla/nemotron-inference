#include "nemotron/nvfp4_packing.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

using nemotron::HostNvfp4Matrix;
using nemotron::Nvfp4PackOptions;
using nemotron::PackRowMajorFp32ToNvfp4;
using nemotron::PackedFp4Bytes;
using nemotron::RowMajorNvfp4ScaleBytes;

bool expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool all_zero(const std::vector<std::uint8_t>& bytes) {
  return std::all_of(bytes.begin(), bytes.end(), [](std::uint8_t value) {
    return value == 0u;
  });
}

bool test_nvfp4_packing_rejects_invalid_inputs() {
  const float values[16] = {0.0f};
  if (!expect(!PackRowMajorFp32ToNvfp4(nullptr, 1, 16).has_value(),
              "null data should be rejected")) {
    return false;
  }
  if (!expect(!PackRowMajorFp32ToNvfp4(values, 0, 16).has_value(),
              "zero rows should be rejected")) {
    return false;
  }
  if (!expect(!PackRowMajorFp32ToNvfp4(values, 1, 15).has_value(),
              "columns must be divisible by 16")) {
    return false;
  }
  return true;
}

bool test_nvfp4_packing_produces_expected_buffer_sizes() {
  std::vector<float> values(32, 0.0f);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>((static_cast<int>(i) % 9) - 4) * 0.5f;
  }

  const auto packed = PackRowMajorFp32ToNvfp4(values.data(), 2, 16);
  if (!expect(packed.has_value(), "packing valid input should succeed")) {
    return false;
  }

  if (!expect(packed->valid(), "packed matrix should be structurally valid")) {
    return false;
  }
  if (!expect(packed->packed_nbytes() == PackedFp4Bytes(2, 16),
              "packed byte count should match rows*cols/2")) {
    return false;
  }
  if (!expect(packed->block_scales_nbytes() == RowMajorNvfp4ScaleBytes(2, 16),
              "block-scale byte count should match one scale per row block")) {
    return false;
  }
  if (!expect(packed->tensor_scale_nbytes() == sizeof(float),
              "tensor scale should remain a single fp32 scalar")) {
    return false;
  }
  if (!expect(packed->tensor_scale >= 1.0f,
              "moderate inputs should not require tensor down-scaling below one")) {
    return false;
  }
  return expect(!all_zero(packed->packed), "nonzero inputs should not pack to all zeros");
}

bool test_nvfp4_packing_raises_tensor_scale_for_large_values() {
  std::vector<float> values(16, 4096.0f);
  const auto packed = PackRowMajorFp32ToNvfp4(values.data(), 1, 16);
  if (!expect(packed.has_value(), "large finite inputs should still pack")) {
    return false;
  }

  if (!expect(packed->tensor_scale > 1.0f,
              "tensor scale should increase when a block exceeds fp4*fp8 range")) {
    return false;
  }
  return expect(!all_zero(packed->packed), "large inputs should still produce encoded fp4 bytes");
}

bool test_nvfp4_packing_honors_fixed_tensor_scale() {
  std::vector<float> values(16, 0.25f);
  Nvfp4PackOptions options;
  options.fixed_tensor_scale = 0.002197265625f;
  const auto packed = PackRowMajorFp32ToNvfp4(values.data(), 1, 16, options);
  if (!expect(packed.has_value(), "fixed-scale packing should succeed")) {
    return false;
  }
  return expect(
      packed->tensor_scale == *options.fixed_tensor_scale,
      "fixed tensor scale should be preserved exactly when valid");
}

}  // namespace

int main() {
  if (!test_nvfp4_packing_rejects_invalid_inputs() ||
      !test_nvfp4_packing_produces_expected_buffer_sizes() ||
      !test_nvfp4_packing_raises_tensor_scale_for_large_values() ||
      !test_nvfp4_packing_honors_fixed_tensor_scale()) {
    return 1;
  }
  std::cout << "nvfp4_packing_test: PASS\n";
  return 0;
}
