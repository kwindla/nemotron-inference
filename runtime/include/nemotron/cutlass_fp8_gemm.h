#pragma once

#include <cuda_runtime.h>

namespace nemotron {

// Returns true if the CUTLASS FP8 dense GEMM is available in this build.
bool CutlassFp8DenseGemmAvailable();

// Decode-focused CUTLASS FP8 dense GEMM for SM120/SM121.
//
// A (activations): FP8 E4M3, RowMajor, [M, K]
// B (weights):     FP8 E4M3, logical ColumnMajor, [N, K]
//                  Our stored row-major [N, K] layout matches this stride convention.
// D (output):      FP32, RowMajor, [M, N]
//
// The epilogue applies D = alpha * (A @ B), with beta fixed to 0.
// This minimal kernel is specialized for the decode path, currently only
// accepts M <= 16, and caches CUTLASS workspace/adapter state internally so
// repeated decode calls avoid per-call initialize-time allocation work.
bool RunCutlassFp8DenseGemm(
    int m,
    int n,
    int k,
    const void* fp8_a,
    const void* fp8_b,
    float alpha,
    float* output,
    cudaStream_t stream);

}  // namespace nemotron
