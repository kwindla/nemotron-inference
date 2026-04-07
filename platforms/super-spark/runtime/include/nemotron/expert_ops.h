#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "nemotron/device_buffer.h"
#include "nemotron/device_tensor.h"

namespace nemotron {

bool CopyRowFp32(
    const DeviceTensorFp32& input,
    std::size_t row_index,
    DeviceTensorFp32* output_row,
    cudaStream_t stream = nullptr);

bool CopyRowBf16(
    const DeviceTensorBf16& input,
    std::size_t row_index,
    DeviceTensorBf16* output_row,
    cudaStream_t stream = nullptr);

bool WriteRowFp32(
    const DeviceTensorFp32& input_row,
    std::size_t row_index,
    DeviceTensorFp32* output);

bool Relu2InPlaceFp32(DeviceTensorFp32* tensor, cudaStream_t stream = nullptr);
bool Relu2InPlaceBf16(DeviceTensorBf16* tensor, cudaStream_t stream = nullptr);

bool AddScaledFp32(
    const DeviceTensorFp32& input,
    float scale,
    DeviceTensorFp32* accumulator,
    cudaStream_t stream = nullptr);

bool AddScaledBf16(
    const DeviceTensorBf16& input,
    float scale,
    DeviceTensorBf16* accumulator,
    cudaStream_t stream = nullptr);

bool AddScaledRowFp32(
    const DeviceTensorFp32& input_row,
    float scale,
    std::size_t row_index,
    DeviceTensorFp32* accumulator,
    cudaStream_t stream = nullptr);

bool AddScaledRowBf16(
    const DeviceTensorBf16& input_row,
    float scale,
    std::size_t row_index,
    DeviceTensorBf16* accumulator,
    cudaStream_t stream = nullptr);

// Grouped expert routing selection. Set NEMOTRON_DISABLE_PARALLEL_TOPK to
// force the scalar fallback kernel.
bool SelectTopExpertsFp32(
    const DeviceTensorFp32& router_logits,
    const DeviceTensorFp32& correction_bias,
    std::size_t n_group,
    std::size_t topk_group,
    std::size_t top_k,
    bool norm_topk_prob,
    float routed_scaling_factor,
    std::int32_t* selected_indices_device,
    float* selected_weights_device,
    cudaStream_t stream = nullptr);

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

// Merged up+down gather: gathers packed pointers, raw scale pointers, swizzled
// scale pointers, and tensor scales for both up/down in one pass.
bool GatherExpertSelectionLookupsDualCheckedInPlace(
    const std::int32_t* selected_indices_device,
    std::size_t selection_count,
    std::size_t lookup_count,
    const DeviceBuffer<const void*>& up_packed_lookup,
    const DeviceBuffer<const void*>& up_raw_scale_lookup,
    const DeviceBuffer<const void*>& up_matmul_scale_lookup,
    const DeviceBuffer<float>& up_tensor_scale_lookup,
    DeviceBuffer<const void*>& selected_up_packed_ptrs,
    DeviceBuffer<const void*>& selected_up_raw_scale_ptrs,
    DeviceBuffer<const void*>& selected_up_matmul_scale_ptrs,
    DeviceBuffer<float>& selected_up_tensor_scales,
    const DeviceBuffer<const void*>& down_packed_lookup,
    const DeviceBuffer<const void*>& down_raw_scale_lookup,
    const DeviceBuffer<const void*>& down_matmul_scale_lookup,
    const DeviceBuffer<float>& down_tensor_scale_lookup,
    DeviceBuffer<const void*>& selected_down_packed_ptrs,
    DeviceBuffer<const void*>& selected_down_raw_scale_ptrs,
    DeviceBuffer<const void*>& selected_down_matmul_scale_ptrs,
    DeviceBuffer<float>& selected_down_tensor_scales,
    DeviceBuffer<std::uint32_t>& missing_count,
    DeviceBuffer<std::int32_t>& missing_indices,
    cudaStream_t stream = nullptr);

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
    const DeviceBuffer<float>& activation_tensor_scales,
    const DeviceBuffer<float>& weight_tensor_scales,
    DeviceBuffer<float>* output_row_scales);

bool ComputeWeightedMergeScales(
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
// Optionally emits swizzled execution-layout block scales for CUTLASS.
// Zero cudaMalloc on the hot path.
bool ScaleRelu2PackRowsToNvfp4InPlace(
    const DeviceTensorFp32& input_rows,
    const float* row_scales_device,
    DeviceBuffer<std::uint8_t>& packed,
    DeviceBuffer<std::uint8_t>& block_scales,
    DeviceBuffer<std::uint8_t>* matmul_block_scales,
    DeviceBuffer<float>& tensor_scales,
    std::size_t packed_row_stride_bytes = 0,
    cudaStream_t stream = nullptr,
    const float* fixed_row_tensor_scales = nullptr);

bool WeightedSumRowsFp32(
    const DeviceTensorFp32& input_rows,
    const float* row_scales_device,
    DeviceTensorFp32* output_row);

bool FusedRoutedUpProjPackedNvfp4SingleToken(
    const std::uint8_t* activation_rows_packed,
    const std::uint8_t* activation_rows_block_scales,
    const DeviceBuffer<float>& activation_row_tensor_scales,
    std::size_t input_cols,
    const DeviceBuffer<const void*>& weight_packed_ptrs,
    const DeviceBuffer<const void*>& weight_block_scale_ptrs,
    const DeviceBuffer<float>& weight_tensor_scales,
    DeviceTensorFp32* output_rows,
    std::size_t activation_rows_packed_row_stride_bytes = 0,
    cudaStream_t stream = nullptr);

bool FusedRoutedUpProjPackedNvfp4SingleToken(
    const std::uint8_t* activation_rows_packed,
    const std::uint8_t* activation_rows_block_scales,
    const DeviceBuffer<float>& activation_row_tensor_scales,
    std::size_t input_cols,
    const std::uint8_t* contiguous_weight_packed_base,
    std::size_t weight_packed_stride_bytes,
    const std::uint8_t* contiguous_weight_scale_base,
    std::size_t weight_scale_stride_bytes,
    const float* contiguous_tensor_scales,
    const std::int32_t* selected_expert_indices,
    DeviceTensorFp32* output_rows,
    std::size_t activation_rows_packed_row_stride_bytes = 0,
    cudaStream_t stream = nullptr);

bool FusedRoutedDownProjPackedNvfp4SingleToken(
    const std::uint8_t* activation_rows_packed,
    const std::uint8_t* activation_rows_block_scales,
    const DeviceBuffer<float>& activation_row_tensor_scales,
    std::size_t input_cols,
    const DeviceBuffer<const void*>& weight_packed_ptrs,
    const DeviceBuffer<const void*>& weight_block_scale_ptrs,
    const DeviceBuffer<float>& weight_tensor_scales,
    DeviceTensorFp32* output_rows,
    std::size_t activation_rows_packed_row_stride_bytes = 0,
    cudaStream_t stream = nullptr);

bool FusedRoutedDownProjPackedNvfp4SingleToken(
    const std::uint8_t* activation_rows_packed,
    const std::uint8_t* activation_rows_block_scales,
    const DeviceBuffer<float>& activation_row_tensor_scales,
    std::size_t input_cols,
    const std::uint8_t* contiguous_weight_packed_base,
    std::size_t weight_packed_stride_bytes,
    const std::uint8_t* contiguous_weight_scale_base,
    std::size_t weight_scale_stride_bytes,
    const float* contiguous_tensor_scales,
    const std::int32_t* selected_expert_indices,
    DeviceTensorFp32* output_rows,
    std::size_t activation_rows_packed_row_stride_bytes = 0,
    cudaStream_t stream = nullptr);

// Fill a device array with the same pointer value. Used to build CUTLASS A pointer arrays
// where all groups share the same activation buffer.
bool FillDevicePointerArray(void** device_array, void* value, std::size_t count);

// Build a device array of strided pointers: device_array[i] = base + i * stride_bytes.
// Used to build CUTLASS C/D pointer arrays where each group's output is at a different row offset.
bool BuildStridedDevicePointerArray(void** device_array, void* base, std::size_t stride_bytes, std::size_t count);

// Indexed variant: device_array[i] = base + selected_indices_device[i] * stride_bytes.
// Used for contiguous expert-major weight stacks where each group selects a routed expert id.
bool BuildStridedDevicePointerArray(
    void** device_array,
    void* base,
    const std::int32_t* selected_indices_device,
    std::size_t stride_bytes,
    std::size_t count,
    cudaStream_t stream = nullptr);

// Gather FP32 values by selected expert id into a pre-allocated output buffer.
bool GatherIndexedFloatsInPlace(
    const float* values_device,
    std::size_t value_count,
    const std::int32_t* selected_indices_device,
    std::size_t count,
    DeviceBuffer<float>& output,
    cudaStream_t stream = nullptr);

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
    DeviceTensorFp32* accumulator,
    cudaStream_t stream = nullptr);

bool ScaleWeightedAccumulateRowsBf16(
    const DeviceTensorFp32& data,
    const float* act_tensor_scales_device,
    const float* weight_tensor_scales_device,
    const float* routing_weights_device,
    std::size_t row_count,
    DeviceTensorBf16* accumulator,
    cudaStream_t stream = nullptr);

}  // namespace nemotron
