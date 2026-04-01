#pragma once

#include <cstddef>
#include <cstdint>

#include "nemotron/device_buffer.h"
#include "nemotron/device_tensor.h"

namespace nemotron {

bool CopyRowFp32(
    const DeviceTensorFp32& input,
    std::size_t row_index,
    DeviceTensorFp32* output_row);

bool WriteRowFp32(
    const DeviceTensorFp32& input_row,
    std::size_t row_index,
    DeviceTensorFp32* output);

bool Relu2InPlaceFp32(DeviceTensorFp32* tensor);

bool AddScaledFp32(
    const DeviceTensorFp32& input,
    float scale,
    DeviceTensorFp32* accumulator);

bool AddScaledRowFp32(
    const DeviceTensorFp32& input_row,
    float scale,
    std::size_t row_index,
    DeviceTensorFp32* accumulator);

bool SelectTopExpertsFp32(
    const DeviceTensorFp32& router_logits,
    const DeviceTensorFp32& correction_bias,
    std::size_t n_group,
    std::size_t topk_group,
    std::size_t top_k,
    bool norm_topk_prob,
    float routed_scaling_factor,
    std::int32_t* selected_indices_device,
    float* selected_weights_device);

bool GatherExpertSelectionLookups(
    const std::int32_t* selected_indices_device,
    std::size_t selection_count,
    std::size_t lookup_count,
    const DeviceBuffer<const void*>& packed_lookup,
    const DeviceBuffer<const void*>& matmul_scale_lookup,
    const DeviceBuffer<float>& tensor_scale_lookup,
    DeviceBuffer<const void*>* selected_packed_ptrs,
    DeviceBuffer<const void*>* selected_matmul_scale_ptrs,
    DeviceBuffer<float>* selected_tensor_scales);

bool GatherExpertSelectionLookupsChecked(
    const std::int32_t* selected_indices_device,
    std::size_t selection_count,
    std::size_t lookup_count,
    const DeviceBuffer<const void*>& packed_lookup,
    const DeviceBuffer<const void*>& matmul_scale_lookup,
    const DeviceBuffer<float>& tensor_scale_lookup,
    DeviceBuffer<const void*>* selected_packed_ptrs,
    DeviceBuffer<const void*>* selected_matmul_scale_ptrs,
    DeviceBuffer<float>* selected_tensor_scales,
    DeviceBuffer<std::uint32_t>* missing_count,
    DeviceBuffer<std::int32_t>* missing_indices);

// Same as GatherExpertSelectionLookupsChecked but uses pre-allocated output
// buffers. All output DeviceBuffers must already be sized >= selection_count.
// Zero cudaMalloc on the hot path.
bool GatherExpertSelectionLookupsCheckedInPlace(
    const std::int32_t* selected_indices_device,
    std::size_t selection_count,
    std::size_t lookup_count,
    const DeviceBuffer<const void*>& packed_lookup,
    const DeviceBuffer<const void*>& matmul_scale_lookup,
    const DeviceBuffer<float>& tensor_scale_lookup,
    DeviceBuffer<const void*>& selected_packed_ptrs,
    DeviceBuffer<const void*>& selected_matmul_scale_ptrs,
    DeviceBuffer<float>& selected_tensor_scales,
    DeviceBuffer<std::uint32_t>& missing_count,
    DeviceBuffer<std::int32_t>& missing_indices);

bool FillDevicePointerArray(
    const void* value,
    std::size_t count,
    DeviceBuffer<const void*>* output);

bool FillDeviceByteOffsetPointerArray(
    const std::uint8_t* base,
    std::size_t row_stride_bytes,
    std::size_t count,
    DeviceBuffer<const void*>* output);

bool FillDeviceOutputPointerArray(
    float* base,
    std::size_t row_stride_elems,
    std::size_t count,
    DeviceBuffer<void*>* output);

bool ComputeGroupedUpPackScales(
    const float* activation_tensor_scale_device,
    const DeviceBuffer<float>& weight_tensor_scales,
    DeviceBuffer<float>* output_row_scales);

bool ComputeWeightedMergeScales(
    const float* selection_weights_device,
    const DeviceBuffer<float>& activation_tensor_scales,
    const DeviceBuffer<float>& weight_tensor_scales,
    DeviceBuffer<float>* output_row_scales);

bool ScaleRelu2PackRowsToNvfp4(
    const DeviceTensorFp32& input_rows,
    const float* row_scales_device,
    DeviceBuffer<std::uint8_t>* packed,
    DeviceBuffer<std::uint8_t>* block_scales,
    DeviceBuffer<std::uint8_t>* matmul_block_scales,
    DeviceBuffer<float>* tensor_scales);

// Same as ScaleRelu2PackRowsToNvfp4 but writes into pre-allocated buffers.
// No matmul_block_scales (not needed on the single-token hot path).
// Zero cudaMalloc on the hot path.
bool ScaleRelu2PackRowsToNvfp4InPlace(
    const DeviceTensorFp32& input_rows,
    const float* row_scales_device,
    DeviceBuffer<std::uint8_t>& packed,
    DeviceBuffer<std::uint8_t>& block_scales,
    DeviceBuffer<float>& tensor_scales);

bool WeightedSumRowsFp32(
    const DeviceTensorFp32& input_rows,
    const float* row_scales_device,
    DeviceTensorFp32* output_row);

bool FusedRoutedUpProjPackedNvfp4SingleToken(
    const std::uint8_t* activation_packed,
    const std::uint8_t* activation_block_scales,
    const float* activation_tensor_scale,
    std::size_t input_cols,
    const DeviceBuffer<const void*>& weight_packed_ptrs,
    const DeviceBuffer<const void*>& weight_block_scale_ptrs,
    const DeviceBuffer<float>& weight_tensor_scales,
    DeviceTensorFp32* output_rows);

bool FusedRoutedDownProjWeightedPackedNvfp4SingleToken(
    const std::uint8_t* activation_rows_packed,
    const std::uint8_t* activation_rows_block_scales,
    const DeviceBuffer<float>& activation_row_tensor_scales,
    const float* selection_weights_device,
    std::size_t input_cols,
    const DeviceBuffer<const void*>& weight_packed_ptrs,
    const DeviceBuffer<const void*>& weight_block_scale_ptrs,
    const DeviceBuffer<float>& weight_tensor_scales,
    DeviceTensorFp32* output_row);

// Fill a device array with the same pointer value. Used to build CUTLASS A pointer arrays
// where all groups share the same activation buffer.
bool FillDevicePointerArray(void** device_array, void* value, std::size_t count);

// Build a device array of strided pointers: device_array[i] = base + i * stride_bytes.
// Used to build CUTLASS C/D pointer arrays where each group's output is at a different row offset.
bool BuildStridedDevicePointerArray(void** device_array, void* base, std::size_t stride_bytes, std::size_t count);

// Post-GEMM per-row scaling: output[row][col] *= act_tensor_scale * weight_tensor_scales[row]
// All pointers are device-resident. Zero host involvement.
bool ScaleRowsByTensorScaleFp32(
    DeviceTensorFp32* data,
    const float* act_tensor_scale_device,
    const float* weight_tensor_scales_device,
    std::size_t row_count);

// Post-GEMM combined scale + weighted accumulate for down_proj:
// For each row i: accumulator[col] += data[i][col] * act_ts[i] * weight_ts[i] * routing_weight[i]
// act_tensor_scales_device is a per-row device array (one per expert).
// All pointers are device-resident. Zero host involvement.
bool ScaleWeightedAccumulateRowsFp32(
    const DeviceTensorFp32& data,
    const float* act_tensor_scales_device,
    const float* weight_tensor_scales_device,
    const float* routing_weights_device,
    std::size_t row_count,
    DeviceTensorFp32* accumulator);

}  // namespace nemotron
