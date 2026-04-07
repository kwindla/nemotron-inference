#include "nemotron/device_nvfp4_matrix.h"
#include "nemotron/device_tensor.h"
#include "nemotron/nvfp4_packing.h"
#include "nemotron/nvfp4_scale_layout.h"

#include <iostream>
#include <vector>

namespace {

using nemotron::DeviceNvfp4Matrix;
using nemotron::DeviceTensorFp32;
using nemotron::Nvfp4PackOptions;
using nemotron::PackDeviceRowMajorFp32ToNvfp4;
using nemotron::PackRowMajorFp32ToNvfp4;
using nemotron::SwizzleRowMajorNvfp4ScalesForExecution;

bool expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool test_device_nvfp4_matrix_matches_host_packer_for_bounded_values() {
  auto source = DeviceTensorFp32::Create({2, 16});
  if (!source || !source->valid()) {
    std::cout << "device_nvfp4_matrix_test: SKIP (no CUDA device available)\n";
    return true;
  }

  std::vector<float> host_values(32, 0.0f);
  for (std::size_t i = 0; i < host_values.size(); ++i) {
    host_values[i] = static_cast<float>((static_cast<int>(i) % 11) - 5) * 0.25f;
  }
  if (!expect(source->CopyFromHost(host_values.data(), host_values.size()),
              "device source upload should succeed")) {
    return false;
  }

  const auto host_packed = PackRowMajorFp32ToNvfp4(host_values.data(), 2, 16);
  if (!expect(host_packed.has_value(), "host packer should succeed")) {
    return false;
  }

  auto device_packed = PackDeviceRowMajorFp32ToNvfp4(*source);
  if (!expect(device_packed != nullptr && device_packed->valid(),
              "device packer should succeed")) {
    return false;
  }

  std::vector<std::uint8_t> packed_bytes;
  std::vector<std::uint8_t> block_scale_bytes;
  float tensor_scale = 0.0f;
  if (!expect(device_packed->CopyPackedToHost(&packed_bytes),
              "packed bytes should copy back")) {
    return false;
  }
  if (!expect(device_packed->CopyBlockScalesToHost(&block_scale_bytes),
              "block scales should copy back")) {
    return false;
  }
  if (!expect(device_packed->CopyTensorScaleToHost(&tensor_scale),
              "tensor scale should copy back")) {
    return false;
  }

  if (!expect(packed_bytes == host_packed->packed,
              "device packed bytes should match host packer on bounded inputs")) {
    return false;
  }
  if (!expect(block_scale_bytes == host_packed->block_scales,
              "device block scales should match host packer on bounded inputs")) {
    return false;
  }
  return expect(tensor_scale == host_packed->tensor_scale,
                "device tensor scale should match host packer on bounded inputs");
}

bool test_device_nvfp4_matrix_matches_host_packer_for_large_values() {
  auto source = DeviceTensorFp32::Create({1, 16});
  if (!source || !source->valid()) {
    std::cout << "device_nvfp4_matrix_test: SKIP (no CUDA device available)\n";
    return true;
  }

  std::vector<float> host_values(16, 4096.0f);
  if (!expect(source->CopyFromHost(host_values.data(), host_values.size()),
              "large-value source upload should succeed")) {
    return false;
  }

  const auto host_packed = PackRowMajorFp32ToNvfp4(host_values.data(), 1, 16);
  if (!expect(host_packed.has_value(), "host packer should succeed for large values")) {
    return false;
  }
  if (!expect(host_packed->tensor_scale > 1.0f,
              "host packer should raise tensor scale for large values")) {
    return false;
  }

  auto device_packed = PackDeviceRowMajorFp32ToNvfp4(*source);
  if (!expect(device_packed != nullptr && device_packed->valid(),
              "device packer should succeed for large values")) {
    return false;
  }

  std::vector<std::uint8_t> packed_bytes;
  std::vector<std::uint8_t> block_scale_bytes;
  float tensor_scale = 0.0f;
  if (!expect(device_packed->CopyPackedToHost(&packed_bytes),
              "large-value packed bytes should copy back")) {
    return false;
  }
  if (!expect(device_packed->CopyBlockScalesToHost(&block_scale_bytes),
              "large-value block scales should copy back")) {
    return false;
  }
  if (!expect(device_packed->CopyTensorScaleToHost(&tensor_scale),
              "large-value tensor scale should copy back")) {
    return false;
  }

  if (!expect(packed_bytes == host_packed->packed,
              "device packed bytes should match host packer for large values")) {
    return false;
  }
  if (!expect(block_scale_bytes == host_packed->block_scales,
              "device block scales should match host packer for large values")) {
    return false;
  }
  return expect(tensor_scale == host_packed->tensor_scale,
                "device tensor scale should match host packer for large values");
}

bool test_device_nvfp4_matrix_rejects_invalid_shapes() {
  auto invalid_rank = DeviceTensorFp32::Create({32});
  if (!invalid_rank || !invalid_rank->valid()) {
    std::cout << "device_nvfp4_matrix_test: SKIP (no CUDA device available)\n";
    return true;
  }
  if (!expect(PackDeviceRowMajorFp32ToNvfp4(*invalid_rank) == nullptr,
              "rank-1 device tensor should be rejected")) {
    return false;
  }

  auto invalid_cols = DeviceTensorFp32::Create({1, 15});
  if (!expect(invalid_cols != nullptr && invalid_cols->valid(),
              "invalid-col tensor should still allocate")) {
    return false;
  }
  return expect(PackDeviceRowMajorFp32ToNvfp4(*invalid_cols) == nullptr,
                "column count not divisible by 16 should be rejected");
}

bool test_device_nvfp4_matrix_honors_fixed_tensor_scale() {
  auto source = DeviceTensorFp32::Create({1, 16});
  if (!source || !source->valid()) {
    std::cout << "device_nvfp4_matrix_test: SKIP (no CUDA device available)\n";
    return true;
  }

  std::vector<float> host_values(16, 0.25f);
  if (!expect(source->CopyFromHost(host_values.data(), host_values.size()),
              "fixed-scale source upload should succeed")) {
    return false;
  }

  Nvfp4PackOptions options;
  options.fixed_tensor_scale = 0.002197265625f;

  const auto host_packed = PackRowMajorFp32ToNvfp4(host_values.data(), 1, 16, options);
  if (!expect(host_packed.has_value(), "host fixed-scale packer should succeed")) {
    return false;
  }

  auto device_packed = PackDeviceRowMajorFp32ToNvfp4(*source, options);
  if (!expect(device_packed != nullptr && device_packed->valid(),
              "device fixed-scale packer should succeed")) {
    return false;
  }

  std::vector<std::uint8_t> packed_bytes;
  std::vector<std::uint8_t> block_scale_bytes;
  float tensor_scale = 0.0f;
  if (!expect(device_packed->CopyPackedToHost(&packed_bytes),
              "fixed-scale packed bytes should copy back")) {
    return false;
  }
  if (!expect(device_packed->CopyBlockScalesToHost(&block_scale_bytes),
              "fixed-scale block scales should copy back")) {
    return false;
  }
  if (!expect(device_packed->CopyTensorScaleToHost(&tensor_scale),
              "fixed-scale tensor scale should copy back")) {
    return false;
  }

  if (!expect(packed_bytes == host_packed->packed,
              "device fixed-scale packed bytes should match host packer")) {
    return false;
  }
  if (!expect(block_scale_bytes == host_packed->block_scales,
              "device fixed-scale block scales should match host packer")) {
    return false;
  }
  return expect(tensor_scale == host_packed->tensor_scale,
                "device fixed-scale tensor scale should match host packer");
}

bool test_device_nvfp4_matrix_exposes_swizzled_execution_scales() {
  auto source = DeviceTensorFp32::Create({16, 64});
  if (!source || !source->valid()) {
    std::cout << "device_nvfp4_matrix_test: SKIP (no CUDA device available)\n";
    return true;
  }

  std::vector<float> host_values(16 * 64, 0.0f);
  for (std::size_t i = 0; i < host_values.size(); ++i) {
    host_values[i] = static_cast<float>((static_cast<int>(i) % 13) - 6) * 0.125f;
  }
  if (!expect(source->CopyFromHost(host_values.data(), host_values.size()),
              "swizzle source upload should succeed")) {
    return false;
  }

  const auto host_packed = PackRowMajorFp32ToNvfp4(host_values.data(), 16, 64);
  if (!expect(host_packed.has_value(), "host packer should succeed for swizzle test")) {
    return false;
  }

  auto device_packed = PackDeviceRowMajorFp32ToNvfp4(*source);
  if (!expect(device_packed != nullptr && device_packed->valid(),
              "device packer should succeed for swizzle test")) {
    return false;
  }

  std::vector<std::uint8_t> execution_scales;
  if (!expect(device_packed->CopyMatmulBlockScalesToHost(&execution_scales),
              "device execution block scales should copy back")) {
    return false;
  }

  const std::vector<std::uint8_t> expected_execution_scales =
      SwizzleRowMajorNvfp4ScalesForExecution(host_packed->block_scales.data(), 16, 64);
  return expect(execution_scales == expected_execution_scales,
                "device execution block scales should match the swizzled host layout");
}

}  // namespace

int main() {
  if (!test_device_nvfp4_matrix_matches_host_packer_for_bounded_values() ||
      !test_device_nvfp4_matrix_matches_host_packer_for_large_values() ||
      !test_device_nvfp4_matrix_rejects_invalid_shapes() ||
      !test_device_nvfp4_matrix_honors_fixed_tensor_scale() ||
      !test_device_nvfp4_matrix_exposes_swizzled_execution_scales()) {
    return 1;
  }
  std::cout << "device_nvfp4_matrix_test: PASS\n";
  return 0;
}
