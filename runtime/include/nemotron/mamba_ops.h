#pragma once

#include <cstddef>
#include <memory>

#include <cuda_runtime.h>

#include "nemotron/device_tensor.h"

namespace nemotron {

struct MambaChunkScanWorkspace {
  std::unique_ptr<DeviceTensorFp32> dt_chunk;
  std::unique_ptr<DeviceTensorFp32> dA_cumsum;
  std::unique_ptr<DeviceTensorFp32> state_scratch;
  std::unique_ptr<DeviceTensorFp32> cb_chunk;
  std::size_t capacity_tokens = 0;
  std::size_t chunk_size = 0;
};

bool MambaCausalConv1dUpdateDecodeFp32(
    const DeviceTensorFp32& projected,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t conv_kernel_size,
    std::size_t conv_state_offset_elems,
    const DeviceTensorFp32& conv_weight,
    const DeviceTensorFp32& conv_bias,
    DeviceTensorBf16* conv_state,
    DeviceTensorFp32* conv_output,
    cudaStream_t stream = nullptr);

bool MambaCausalConv1dUpdateDecodeBf16(
    const DeviceTensorBf16& projected,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t conv_kernel_size,
    std::size_t conv_state_offset_elems,
    const DeviceTensorFp32& conv_weight,
    const DeviceTensorFp32& conv_bias,
    DeviceTensorBf16* conv_state,
    DeviceTensorBf16* conv_output,
    cudaStream_t stream = nullptr);

bool MambaConv1dSiluUpdateFp32(
    const DeviceTensorFp32& projected,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t conv_kernel_size,
    std::size_t conv_state_offset_elems,
    const DeviceTensorFp32& conv_weight,
    const DeviceTensorFp32& conv_bias,
    DeviceTensorBf16* conv_state,
    DeviceTensorFp32* conv_output,
    cudaStream_t stream = nullptr);

bool MambaConv1dSiluUpdateBf16(
    const DeviceTensorBf16& projected,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t conv_kernel_size,
    std::size_t conv_state_offset_elems,
    const DeviceTensorFp32& conv_weight,
    const DeviceTensorFp32& conv_bias,
    DeviceTensorBf16* conv_state,
    DeviceTensorBf16* conv_output,
    cudaStream_t stream = nullptr);

bool MambaSsmUpdateFp32(
    const DeviceTensorFp32& projected,
    const DeviceTensorFp32& conv_output,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t num_heads,
    std::size_t head_dim,
    std::size_t state_size,
    std::size_t n_groups,
    float time_step_min,
    std::size_t ssm_state_offset_elems,
    const DeviceTensorFp32& a_log,
    const DeviceTensorFp32& d,
    const DeviceTensorFp32& dt_bias,
    DeviceTensorFp32* ssm_state,
    DeviceTensorFp32* y_output,
    cudaStream_t stream = nullptr);

bool MambaSsmUpdateBf16(
    const DeviceTensorBf16& projected,
    const DeviceTensorBf16& conv_output,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t num_heads,
    std::size_t head_dim,
    std::size_t state_size,
    std::size_t n_groups,
    float time_step_min,
    std::size_t ssm_state_offset_elems,
    const DeviceTensorFp32& a_log,
    const DeviceTensorFp32& d,
    const DeviceTensorFp32& dt_bias,
    DeviceTensorFp32* ssm_state,
    DeviceTensorBf16* y_output,
    cudaStream_t stream = nullptr);

bool MambaChunkedScanPrefillBf16(
    const DeviceTensorBf16& projected,
    const DeviceTensorBf16& conv_output,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t num_heads,
    std::size_t head_dim,
    std::size_t state_size,
    std::size_t n_groups,
    std::size_t chunk_size,
    std::size_t ssm_state_offset_elems,
    const DeviceTensorFp32& a_log,
    const DeviceTensorFp32& d,
    const DeviceTensorFp32& dt_bias,
    DeviceTensorFp32* ssm_state,
    DeviceTensorBf16* y_output,
    MambaChunkScanWorkspace* workspace,
    cudaStream_t stream = nullptr);

bool MambaChunkCumsumBf16(
    const DeviceTensorBf16& projected,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t num_heads,
    std::size_t chunk_size,
    const DeviceTensorFp32& a_log,
    const DeviceTensorFp32& dt_bias,
    DeviceTensorFp32* dt_chunk,
    DeviceTensorFp32* dA_cumsum,
    cudaStream_t stream = nullptr);

bool MambaChunkScanOnlyBf16(
    const DeviceTensorBf16& conv_output,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t num_heads,
    std::size_t head_dim,
    std::size_t state_size,
    std::size_t n_groups,
    std::size_t chunk_size,
    const DeviceTensorFp32& d,
    const DeviceTensorFp32& dt_chunk,
    const DeviceTensorFp32& dA_cumsum,
    const DeviceTensorFp32& boundary_state,
    const DeviceTensorFp32& cb_chunk,
    DeviceTensorBf16* y_output,
    cudaStream_t stream = nullptr);

bool MambaChunkedScanFromStage1Bf16(
    const DeviceTensorBf16& conv_output,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t num_heads,
    std::size_t head_dim,
    std::size_t state_size,
    std::size_t n_groups,
    std::size_t chunk_size,
    std::size_t ssm_state_offset_elems,
    const DeviceTensorFp32& d,
    const DeviceTensorFp32& dt_chunk,
    const DeviceTensorFp32& dA_cumsum,
    DeviceTensorFp32* ssm_state,
    DeviceTensorBf16* y_output,
    MambaChunkScanWorkspace* workspace,
    cudaStream_t stream = nullptr);

bool MambaSelectiveStateUpdateDecodeFp32(
    const DeviceTensorFp32& projected,
    const DeviceTensorFp32& conv_output,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t num_heads,
    std::size_t head_dim,
    std::size_t state_size,
    std::size_t n_groups,
    float time_step_min,
    std::size_t ssm_state_offset_elems,
    const DeviceTensorFp32& a_log,
    const DeviceTensorFp32& d,
    const DeviceTensorFp32& dt_bias,
    DeviceTensorFp32* ssm_state,
    DeviceTensorFp32* gated_output,
    cudaStream_t stream = nullptr);

bool MambaSelectiveStateUpdateDecodeBf16(
    const DeviceTensorBf16& projected,
    const DeviceTensorBf16& conv_output,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t num_heads,
    std::size_t head_dim,
    std::size_t state_size,
    std::size_t n_groups,
    float time_step_min,
    std::size_t ssm_state_offset_elems,
    const DeviceTensorFp32& a_log,
    const DeviceTensorFp32& d,
    const DeviceTensorFp32& dt_bias,
    DeviceTensorFp32* ssm_state,
    DeviceTensorBf16* gated_output,
    cudaStream_t stream = nullptr);

bool MambaDecodeStepFusedFp32(
    const DeviceTensorFp32& projected,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t num_heads,
    std::size_t head_dim,
    std::size_t state_size,
    std::size_t n_groups,
    std::size_t conv_kernel_size,
    float time_step_min,
    float mixer_rms_epsilon,
    std::size_t conv_state_offset_elems,
    std::size_t ssm_state_offset_elems,
    const DeviceTensorFp32& conv_weight,
    const DeviceTensorFp32& conv_bias,
    const DeviceTensorFp32& a_log,
    const DeviceTensorFp32& d,
    const DeviceTensorFp32& dt_bias,
    const DeviceTensorFp32& mixer_norm_weight,
    DeviceTensorBf16* conv_state,
    DeviceTensorFp32* ssm_state,
    DeviceTensorFp32* output,
    cudaStream_t stream = nullptr);

bool MambaDecodeStepFusedBf16(
    const DeviceTensorBf16& projected,
    std::size_t intermediate_size,
    std::size_t conv_dim,
    std::size_t num_heads,
    std::size_t head_dim,
    std::size_t state_size,
    std::size_t n_groups,
    std::size_t conv_kernel_size,
    float time_step_min,
    float mixer_rms_epsilon,
    std::size_t conv_state_offset_elems,
    std::size_t ssm_state_offset_elems,
    const DeviceTensorFp32& conv_weight,
    const DeviceTensorFp32& conv_bias,
    const DeviceTensorFp32& a_log,
    const DeviceTensorFp32& d,
    const DeviceTensorFp32& dt_bias,
    const DeviceTensorFp32& mixer_norm_weight,
    DeviceTensorBf16* conv_state,
    DeviceTensorFp32* ssm_state,
    DeviceTensorBf16* output,
    cudaStream_t stream = nullptr);

bool GroupedRmsNormGatedFp32(
    const DeviceTensorFp32& y_output,
    const DeviceTensorFp32& projected,
    const DeviceTensorFp32& mixer_norm_weight,
    std::size_t n_groups,
    float epsilon,
    DeviceTensorFp32* output,
    cudaStream_t stream = nullptr);

bool GroupedRmsNormGatedBf16(
    const DeviceTensorBf16& y_output,
    const DeviceTensorBf16& projected,
    const DeviceTensorFp32& mixer_norm_weight,
    std::size_t n_groups,
    float epsilon,
    DeviceTensorBf16* output,
    cudaStream_t stream = nullptr);

bool GroupedRmsNormFp32(
    const DeviceTensorFp32& input,
    const DeviceTensorFp32& mixer_norm_weight,
    std::size_t n_groups,
    float epsilon,
    DeviceTensorFp32* output,
    cudaStream_t stream = nullptr);

bool GroupedRmsNormBf16(
    const DeviceTensorBf16& input,
    const DeviceTensorFp32& mixer_norm_weight,
    std::size_t n_groups,
    float epsilon,
    DeviceTensorBf16* output,
    cudaStream_t stream = nullptr);

}  // namespace nemotron
