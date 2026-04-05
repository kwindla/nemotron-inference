#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace nemotron {

// Workspace for CUTLASS SM120 grouped FP4×FP4 GEMM.
// Pre-allocate once per request context; reuse across expert layers.
struct GroupedMoeWorkspace {
  void* data = nullptr;
  std::size_t nbytes = 0;
};

// Calculate the required workspace for a grouped GEMM with up to
// max_experts groups. The workspace holds device-side pointer arrays,
// stride arrays, scale-factor layout arrays, alpha values, and any
// internal CUTLASS workspace.
std::size_t GroupedMoeWorkspaceBytes(int max_experts);

// Run a grouped FP4×FP4→FP32 GEMM on SM120 tensor cores.
// One kernel launch for all experts. Each expert has its own M dimension
// (token count) but shared N (output dim) and K (input dim).
//
// All pointer arguments are device pointers to arrays of length num_experts.
// Experts with zero tokens are skipped by CUTLASS internally.
//
// packed_A[i]:    packed FP4 activations for expert i, shape [M_i, K/2] bytes
// scales_A[i]:    swizzled block scales for A_i (UE4M3, 128x4 layout)
// global_A[i]:    pointer to single float tensor scale for A_i
// packed_B[i]:    packed FP4 weights for expert i, shape [N, K/2] bytes
// scales_B[i]:    swizzled block scales for B_i
// global_B[i]:    pointer to single float tensor scale for B_i
// output_D[i]:    FP32 output buffer for expert i, shape [M_i, N]
// token_counts[i]: number of tokens (M) for expert i
//
// K and N are uniform across all experts.
bool RunGroupedFp4Gemm(
    const std::uint8_t* const* packed_A,
    const std::uint8_t* const* scales_A,
    const float* const* global_A,
    const std::uint8_t* const* packed_B,
    const std::uint8_t* const* scales_B,
    const float* const* global_B,
    float* const* output_D,
    const std::int32_t* token_counts,
    int K,
    int N,
    int num_experts,
    GroupedMoeWorkspace& workspace,
    cudaStream_t stream = nullptr);

}  // namespace nemotron
