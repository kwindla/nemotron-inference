#include "nemotron/device_nvfp4_matrix.h"
#include "nemotron/device_tensor.h"
#include "nemotron/nvfp4_packing.h"
#include "nemotron/nvfp4_scale_layout.h"

#include <cuda_fp8.h>

#include <optional>
#include <iostream>
#include <vector>

namespace {

using nemotron::DeviceNvfp4Matrix;
using nemotron::DeviceTensorFp32;
using nemotron::DeviceTensorInt32;
using nemotron::GatherDeviceNvfp4Rows;
using nemotron::Nvfp4PackOptions;
using nemotron::Nvfp4ScaleLayout;
using nemotron::PackDeviceRowMajorFp32ToNvfp4;
using nemotron::PackDeviceRowMajorFp32ToNvfp4PerExpert;
using nemotron::PackRowMajorFp32ToNvfp4;
using nemotron::SwizzleRowMajorNvfp4ScalesForExecution;

bool expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

std::vector<std::uint8_t> slice_bytes(
    const std::vector<std::uint8_t>& values,
    std::size_t offset,
    std::size_t count) {
  if (offset + count > values.size()) {
    return {};
  }
  return std::vector<std::uint8_t>(values.begin() + static_cast<std::ptrdiff_t>(offset),
                                   values.begin() + static_cast<std::ptrdiff_t>(offset + count));
}

float decode_fp8(std::uint8_t raw_byte) {
  __nv_fp8_e4m3 value;
  value.__x = raw_byte;
  return static_cast<float>(value);
}

struct BatchDependentExample {
  std::vector<float> first_row;
  std::vector<float> second_row;
  float fixed_tensor_scale = 0.0f;
};

float effective_scale(const nemotron::HostNvfp4Matrix& matrix, std::size_t block_index) {
  return matrix.tensor_scale * decode_fp8(matrix.block_scales[block_index]);
}

std::optional<BatchDependentExample> find_batch_dependent_example() {
  const std::vector<float> base_values = {
      0.03125f, 0.046875f, 0.0625f, 0.09375f, 0.125f, 0.1875f, 0.25f,
      0.375f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 2.0f, 3.0f,
  };
  const std::vector<float> outlier_values = {
      2048.0f, 3072.0f, 4096.0f, 6144.0f, 8192.0f,
  };

  for (float base : base_values) {
    std::vector<float> first_row(16, 0.0f);
    for (std::size_t i = 0; i < first_row.size(); ++i) {
      const float sign = (i % 2 == 0) ? 1.0f : -1.0f;
      const float multiplier = 0.5f + (0.25f * static_cast<float>(i % 7));
      first_row[i] = sign * base * multiplier;
    }
    const auto single = PackRowMajorFp32ToNvfp4(first_row.data(), 1, 16);
    if (!single.has_value()) {
      continue;
    }

    for (float outlier : outlier_values) {
      std::vector<float> second_row(16, outlier);
      std::vector<float> batched_values = first_row;
      batched_values.insert(batched_values.end(), second_row.begin(), second_row.end());
      const auto batched = PackRowMajorFp32ToNvfp4(batched_values.data(), 2, 16);
      if (!batched.has_value()) {
        continue;
      }
      if (effective_scale(*single, 0) != effective_scale(*batched, 0) ||
          single->packed != std::vector<std::uint8_t>(batched->packed.begin(), batched->packed.begin() + 8)) {
        BatchDependentExample example;
        example.first_row = std::move(first_row);
        example.second_row = std::move(second_row);
        example.fixed_tensor_scale = 2.0f * batched->tensor_scale;
        return example;
      }
    }
  }
  return std::nullopt;
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
  auto source = DeviceTensorFp32::Create({64, 64});
  if (!source || !source->valid()) {
    std::cout << "device_nvfp4_matrix_test: SKIP (no CUDA device available)\n";
    return true;
  }

  std::vector<float> host_values(64 * 64, 0.0f);
  for (std::size_t i = 0; i < host_values.size(); ++i) {
    host_values[i] = static_cast<float>((static_cast<int>(i) % 13) - 6) * 0.125f;
  }
  if (!expect(source->CopyFromHost(host_values.data(), host_values.size()),
              "swizzle source upload should succeed")) {
    return false;
  }

  const auto host_packed = PackRowMajorFp32ToNvfp4(host_values.data(), 64, 64);
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
      SwizzleRowMajorNvfp4ScalesForExecution(
          host_packed->block_scales.data(),
          64,
          64,
          Nvfp4ScaleLayout::kSwizzled128x4);
  return expect(execution_scales == expected_execution_scales,
                "device execution block scales should match the swizzled host layout");
}

bool test_device_nvfp4_matrix_uses_8x4_execution_scales_for_small_m() {
  auto source = DeviceTensorFp32::Create({1, 64});
  if (!source || !source->valid()) {
    std::cout << "device_nvfp4_matrix_test: SKIP (no CUDA device available)\n";
    return true;
  }

  std::vector<float> host_values(64, 0.0f);
  for (std::size_t i = 0; i < host_values.size(); ++i) {
    host_values[i] = static_cast<float>((static_cast<int>(i) % 9) - 4) * 0.25f;
  }
  if (!expect(source->CopyFromHost(host_values.data(), host_values.size()),
              "small-M source upload should succeed")) {
    return false;
  }

  const auto host_packed = PackRowMajorFp32ToNvfp4(host_values.data(), 1, 64);
  if (!expect(host_packed.has_value(), "host packer should succeed for small-M test")) {
    return false;
  }

  auto device_packed = PackDeviceRowMajorFp32ToNvfp4(*source);
  if (!expect(device_packed != nullptr && device_packed->valid(),
              "small-M device packer should succeed")) {
    return false;
  }

  std::vector<std::uint8_t> execution_scales;
  if (!expect(device_packed->CopyMatmulBlockScalesToHost(&execution_scales),
              "small-M execution block scales should copy back")) {
    return false;
  }

  const std::vector<std::uint8_t> expected_execution_scales =
      SwizzleRowMajorNvfp4ScalesForExecution(
          host_packed->block_scales.data(),
          1,
          64,
          Nvfp4ScaleLayout::kSwizzled8x4);
  return expect(execution_scales == expected_execution_scales,
                "small-M activation packing should use the 8x4 execution scale layout");
}

bool test_device_nvfp4_matrix_default_pack_is_batch_dependent() {
  auto single = DeviceTensorFp32::Create({1, 16});
  auto batched = DeviceTensorFp32::Create({2, 16});
  if (!single || !single->valid() || !batched || !batched->valid()) {
    std::cout << "device_nvfp4_matrix_test: SKIP (no CUDA device available)\n";
    return true;
  }

  const auto example = find_batch_dependent_example();
  if (!expect(example.has_value(), "search should find a batch-dependent NVFP4 packing example")) {
    return false;
  }
  std::vector<float> single_values = example->first_row;
  std::vector<float> batched_values = example->first_row;
  batched_values.insert(batched_values.end(), example->second_row.begin(), example->second_row.end());

  if (!expect(single->CopyFromHost(single_values.data(), single_values.size()),
              "single-row source upload should succeed") ||
      !expect(batched->CopyFromHost(batched_values.data(), batched_values.size()),
              "batched source upload should succeed")) {
    return false;
  }

  auto single_packed = PackDeviceRowMajorFp32ToNvfp4(*single);
  auto batched_packed = PackDeviceRowMajorFp32ToNvfp4(*batched);
  if (!expect(single_packed != nullptr && single_packed->valid(),
              "single-row pack should succeed") ||
      !expect(batched_packed != nullptr && batched_packed->valid(),
              "batched pack should succeed")) {
    return false;
  }

  std::vector<std::uint8_t> single_packed_bytes;
  std::vector<std::uint8_t> batched_packed_bytes;
  std::vector<std::uint8_t> single_block_scales;
  std::vector<std::uint8_t> batched_block_scales;
  float single_tensor_scale = 0.0f;
  float batched_tensor_scale = 0.0f;
  if (!expect(single_packed->CopyPackedToHost(&single_packed_bytes),
              "single-row packed bytes should copy back") ||
      !expect(batched_packed->CopyPackedToHost(&batched_packed_bytes),
              "batched packed bytes should copy back") ||
      !expect(single_packed->CopyBlockScalesToHost(&single_block_scales),
              "single-row block scales should copy back") ||
      !expect(batched_packed->CopyBlockScalesToHost(&batched_block_scales),
              "batched block scales should copy back") ||
      !expect(single_packed->CopyTensorScaleToHost(&single_tensor_scale),
              "single-row tensor scale should copy back") ||
      !expect(batched_packed->CopyTensorScaleToHost(&batched_tensor_scale),
              "batched tensor scale should copy back")) {
    return false;
  }

  return expect(
             single_tensor_scale != batched_tensor_scale,
             "default activation packing should allow tensor scale to change with batch composition") &&
         expect(
             (single_tensor_scale * decode_fp8(single_block_scales.front())) !=
                 (batched_tensor_scale * decode_fp8(batched_block_scales.front())),
             "the same first row should use a different effective scale when packed alone vs in a larger batch");
}

bool test_device_nvfp4_matrix_fixed_tensor_scale_restores_batch_invariance() {
  auto single = DeviceTensorFp32::Create({1, 16});
  auto batched = DeviceTensorFp32::Create({2, 16});
  if (!single || !single->valid() || !batched || !batched->valid()) {
    std::cout << "device_nvfp4_matrix_test: SKIP (no CUDA device available)\n";
    return true;
  }

  const auto example = find_batch_dependent_example();
  if (!expect(example.has_value(), "search should find a batch-dependent example for fixed-scale validation")) {
    return false;
  }
  std::vector<float> single_values = example->first_row;
  std::vector<float> batched_values = example->first_row;
  batched_values.insert(batched_values.end(), example->second_row.begin(), example->second_row.end());

  if (!expect(single->CopyFromHost(single_values.data(), single_values.size()),
              "single-row fixed-scale source upload should succeed") ||
      !expect(batched->CopyFromHost(batched_values.data(), batched_values.size()),
              "batched fixed-scale source upload should succeed")) {
    return false;
  }

  Nvfp4PackOptions options;
  options.fixed_tensor_scale = example->fixed_tensor_scale;
  auto single_packed = PackDeviceRowMajorFp32ToNvfp4(*single, options);
  auto batched_packed = PackDeviceRowMajorFp32ToNvfp4(*batched, options);
  if (!expect(single_packed != nullptr && single_packed->valid(),
              "single-row fixed-scale pack should succeed") ||
      !expect(batched_packed != nullptr && batched_packed->valid(),
              "batched fixed-scale pack should succeed")) {
    return false;
  }

  std::vector<std::uint8_t> single_packed_bytes;
  std::vector<std::uint8_t> batched_packed_bytes;
  std::vector<std::uint8_t> single_block_scales;
  std::vector<std::uint8_t> batched_block_scales;
  float single_tensor_scale = 0.0f;
  float batched_tensor_scale = 0.0f;
  if (!expect(single_packed->CopyPackedToHost(&single_packed_bytes),
              "single-row fixed-scale packed bytes should copy back") ||
      !expect(batched_packed->CopyPackedToHost(&batched_packed_bytes),
              "batched fixed-scale packed bytes should copy back") ||
      !expect(single_packed->CopyBlockScalesToHost(&single_block_scales),
              "single-row fixed-scale block scales should copy back") ||
      !expect(batched_packed->CopyBlockScalesToHost(&batched_block_scales),
              "batched fixed-scale block scales should copy back") ||
      !expect(single_packed->CopyTensorScaleToHost(&single_tensor_scale),
              "single-row fixed-scale tensor scale should copy back") ||
      !expect(batched_packed->CopyTensorScaleToHost(&batched_tensor_scale),
              "batched fixed-scale tensor scale should copy back")) {
    return false;
  }

  return expect(
             single_tensor_scale == *options.fixed_tensor_scale &&
                 batched_tensor_scale == *options.fixed_tensor_scale,
             "fixed tensor scale should be preserved regardless of batch composition") &&
         expect(
             (single_tensor_scale * decode_fp8(single_block_scales.front())) ==
                 (batched_tensor_scale * decode_fp8(batched_block_scales.front())),
             "with a fixed tensor scale, the first-row effective scale should be batch invariant") &&
         expect(
             slice_bytes(single_packed_bytes, 0, 8) == slice_bytes(batched_packed_bytes, 0, 8),
             "with a fixed tensor scale, the first-row packed bytes should stay batch invariant") &&
         expect(
             slice_bytes(single_block_scales, 0, 1) == slice_bytes(batched_block_scales, 0, 1),
             "with a fixed tensor scale, the first-row block scale should be batch invariant");
}

bool test_device_nvfp4_matrix_per_expert_pack_honors_grouped_tensor_scales() {
  auto source = DeviceTensorFp32::Create({384, 16});
  auto expert_offsets = DeviceTensorInt32::Create({3});
  auto expert_scales = DeviceTensorFp32::Create({2, 1});
  if (!source || !source->valid() ||
      !expert_offsets || !expert_offsets->valid() ||
      !expert_scales || !expert_scales->valid()) {
    std::cout << "device_nvfp4_matrix_test: SKIP (no CUDA device available)\n";
    return true;
  }

  std::vector<float> host_values(384 * 16, 0.0f);
  for (std::size_t col = 0; col < 16; ++col) {
    const float sign = (col % 2 == 0) ? 1.0f : -1.0f;
    const float value = sign * (0.25f + (0.03125f * static_cast<float>(col % 5)));
    host_values[col] = value;
    host_values[(128 * 16) + col] = value;
  }
  const std::vector<int> host_offsets = {0, 128, 256};
  const std::vector<float> host_scales = {0.25f, 2.0f};
  if (!expect(source->CopyFromHost(host_values.data(), host_values.size()),
              "grouped source upload should succeed") ||
      !expect(expert_offsets->CopyFromHost(host_offsets.data(), host_offsets.size()),
              "expert offsets upload should succeed") ||
      !expect(expert_scales->CopyFromHost(host_scales.data(), host_scales.size()),
              "expert scales upload should succeed")) {
    return false;
  }

  auto packed = DeviceNvfp4Matrix::Create(384, 16, Nvfp4ScaleLayout::kSwizzled128x4);
  if (!expect(packed != nullptr && packed->valid(),
              "per-expert packed matrix should allocate")) {
    return false;
  }
  if (!expect(
          PackDeviceRowMajorFp32ToNvfp4PerExpert(
              source->data(),
              384,
              16,
              expert_offsets->data(),
              2,
              expert_scales->data(),
              packed.get()),
          "per-expert grouped pack should succeed")) {
    return false;
  }

  std::vector<std::uint8_t> packed_bytes;
  std::vector<std::uint8_t> block_scales;
  float tensor_scale = 0.0f;
  if (!expect(packed->CopyPackedToHost(&packed_bytes),
              "per-expert packed bytes should copy back") ||
      !expect(packed->CopyBlockScalesToHost(&block_scales),
              "per-expert block scales should copy back") ||
      !expect(packed->CopyTensorScaleToHost(&tensor_scale),
              "per-expert tensor scale should copy back")) {
    return false;
  }

  Nvfp4PackOptions first_options;
  first_options.fixed_tensor_scale = host_scales[0];
  Nvfp4PackOptions second_options;
  second_options.fixed_tensor_scale = host_scales[1];
  const auto first_row = PackRowMajorFp32ToNvfp4(host_values.data(), 1, 16, first_options);
  const auto second_row = PackRowMajorFp32ToNvfp4(host_values.data() + (128 * 16), 1, 16, second_options);
  if (!expect(first_row.has_value() && second_row.has_value(),
              "host row packers should succeed for per-expert validation")) {
    return false;
  }

  return expect(
             tensor_scale == 1.0f,
             "grouped per-expert pack should keep the matrix tensor scale neutral") &&
         expect(
             slice_bytes(packed_bytes, 0, 8) == first_row->packed,
             "expert 0 row should match the fixed-scale host packer") &&
         expect(
             slice_bytes(packed_bytes, 128 * 8, 8) == second_row->packed,
             "expert 1 row should match the fixed-scale host packer") &&
         expect(
             slice_bytes(block_scales, 0, 1) == first_row->block_scales,
             "expert 0 block scale should match the fixed-scale host packer") &&
         expect(
             slice_bytes(block_scales, 128, 1) == second_row->block_scales,
             "expert 1 block scale should match the fixed-scale host packer") &&
         expect(
             slice_bytes(packed_bytes, 256 * 8, 8) == std::vector<std::uint8_t>(8, 0u),
             "rows beyond the active grouped span should stay zeroed") &&
         expect(
             slice_bytes(block_scales, 256, 1) == std::vector<std::uint8_t>(1, 0u),
             "inactive grouped rows should keep zero block scales");
}

bool test_device_nvfp4_matrix_gathers_packed_rows_and_preserves_tensor_scale() {
  constexpr std::size_t kSourceRows = 128;
  auto source_tensor = DeviceTensorFp32::Create({kSourceRows, 16});
  auto gather_indices = DeviceTensorInt32::Create({6});
  if (!source_tensor || !source_tensor->valid() ||
      !gather_indices || !gather_indices->valid()) {
    std::cout << "device_nvfp4_matrix_test: SKIP (no CUDA device available)\n";
    return true;
  }

  std::vector<float> host_values(kSourceRows * 16, 0.0f);
  for (std::size_t row = 0; row < 4; ++row) {
    for (std::size_t col = 0; col < 16; ++col) {
      const float sign = ((row + col) % 2 == 0) ? 1.0f : -1.0f;
      host_values[row * 16 + col] = sign * (0.125f + 0.03125f * static_cast<float>(row + col));
    }
  }
  const std::vector<int> host_indices = {3, 1, -1, 0, 2, -1};
  if (!expect(source_tensor->CopyFromHost(host_values.data(), host_values.size()),
              "source tensor upload should succeed") ||
      !expect(gather_indices->CopyFromHost(host_indices.data(), host_indices.size()),
              "gather indices upload should succeed")) {
    return false;
  }

  Nvfp4PackOptions pack_options;
  pack_options.fixed_tensor_scale = 2.0f;
  auto source_pack = DeviceNvfp4Matrix::Create(kSourceRows, 16, Nvfp4ScaleLayout::kSwizzled128x4);
  auto gathered_pack = DeviceNvfp4Matrix::Create(6, 16, Nvfp4ScaleLayout::kSwizzled128x4);
  if (!expect(source_pack != nullptr && source_pack->valid(),
              "source pack should allocate") ||
      !expect(source_pack->PackInto(*source_tensor, pack_options),
              "source pack should succeed") ||
      !expect(gathered_pack != nullptr && gathered_pack->valid(),
              "gathered pack should allocate") ||
      !expect(GatherDeviceNvfp4Rows(*source_pack, gather_indices->data(), 6, gathered_pack.get()),
              "packed row gather should succeed")) {
    return false;
  }

  std::vector<std::uint8_t> source_packed_bytes;
  std::vector<std::uint8_t> source_block_scales;
  std::vector<std::uint8_t> gathered_packed_bytes;
  std::vector<std::uint8_t> gathered_block_scales;
  float source_tensor_scale = 0.0f;
  float gathered_tensor_scale = 0.0f;
  if (!expect(source_pack->CopyPackedToHost(&source_packed_bytes),
              "source packed bytes should copy back") ||
      !expect(source_pack->CopyBlockScalesToHost(&source_block_scales),
              "source block scales should copy back") ||
      !expect(gathered_pack->CopyPackedToHost(&gathered_packed_bytes),
              "gathered packed bytes should copy back") ||
      !expect(gathered_pack->CopyBlockScalesToHost(&gathered_block_scales),
              "gathered block scales should copy back") ||
      !expect(source_pack->CopyTensorScaleToHost(&source_tensor_scale),
              "source tensor scale should copy back") ||
      !expect(gathered_pack->CopyTensorScaleToHost(&gathered_tensor_scale),
              "gathered tensor scale should copy back")) {
    return false;
  }

  return expect(
             gathered_tensor_scale == source_tensor_scale,
             "gathered pack should preserve the source tensor scale") &&
         expect(
             slice_bytes(gathered_packed_bytes, 0, 8) == slice_bytes(source_packed_bytes, 3 * 8, 8),
             "gathered row 0 should copy source row 3 packed bytes") &&
         expect(
             slice_bytes(gathered_block_scales, 0, 1) == slice_bytes(source_block_scales, 3, 1),
             "gathered row 0 should copy source row 3 block scales") &&
         expect(
             slice_bytes(gathered_packed_bytes, 2 * 8, 8) == std::vector<std::uint8_t>(8, 0u),
             "invalid gathered rows should keep zero packed bytes") &&
         expect(
             slice_bytes(gathered_block_scales, 2, 1) == std::vector<std::uint8_t>(1, 0u),
             "invalid gathered rows should keep zero block scales");
}

}  // namespace

int main() {
  if (!test_device_nvfp4_matrix_matches_host_packer_for_bounded_values() ||
      !test_device_nvfp4_matrix_matches_host_packer_for_large_values() ||
      !test_device_nvfp4_matrix_rejects_invalid_shapes() ||
      !test_device_nvfp4_matrix_honors_fixed_tensor_scale() ||
      !test_device_nvfp4_matrix_exposes_swizzled_execution_scales() ||
      !test_device_nvfp4_matrix_uses_8x4_execution_scales_for_small_m() ||
      !test_device_nvfp4_matrix_default_pack_is_batch_dependent() ||
      !test_device_nvfp4_matrix_fixed_tensor_scale_restores_batch_invariance() ||
      !test_device_nvfp4_matrix_per_expert_pack_honors_grouped_tensor_scales() ||
      !test_device_nvfp4_matrix_gathers_packed_rows_and_preserves_tensor_scale()) {
    return 1;
  }
  std::cout << "device_nvfp4_matrix_test: PASS\n";
  return 0;
}
