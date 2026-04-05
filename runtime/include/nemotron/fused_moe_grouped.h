#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "nemotron/device_nvfp4_matrix.h"
#include "nemotron/device_tensor.h"

namespace nemotron {

struct Nvfp4GroupedMoEWorkspace {
    std::size_t nbytes = 0;
    void* data = nullptr;
};

// Calculate required workspace size for the CUTLASS Grouped GEMM
std::size_t GetNvfp4GroupedMoEWorkspaceSize(int num_experts);

// Run grouped FP4 GEMM using SM120 tensor cores
bool RunNvfp4GroupedMoEFp32AccumToDevice(
    const std::uint8_t* const* packed_activations_device,
    const std::uint8_t* const* activation_scales_device,
    const float* const* activation_global_scale_device,
    const std::uint8_t* const* packed_weights_device,
    const std::uint8_t* const* weight_scales_device,
    const float* const* weight_global_scales_device,
    float* const* output_activations_device,
    const int32_t* expert_token_counts_device, // Size: num_experts
    int hidden_size,
    int intermediate_size,
    int num_experts,
    Nvfp4GroupedMoEWorkspace& workspace,
    cudaStream_t stream = 0);

}  // namespace nemotron
