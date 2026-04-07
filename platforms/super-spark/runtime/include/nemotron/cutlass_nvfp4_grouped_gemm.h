#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace nemotron {

// Returns true if the CUTLASS NVFP4 grouped GEMM is available on this build.
bool CutlassNvfp4GroupedGemmAvailable();

// Pre-allocated plan for a CUTLASS NVFP4 grouped GEMM.
// Holds device-resident strides, SF layouts, problem sizes, and workspace
// that are constant across calls with the same (group_count, M, N, K).
// Created once during model construction, reused every token.
class CutlassNvfp4GroupedGemmPlan {
 public:
  static std::unique_ptr<CutlassNvfp4GroupedGemmPlan> Create(
      int group_count, int m, int n, int k);

  CutlassNvfp4GroupedGemmPlan(CutlassNvfp4GroupedGemmPlan&&) noexcept;
  CutlassNvfp4GroupedGemmPlan& operator=(CutlassNvfp4GroupedGemmPlan&&) noexcept;
  ~CutlassNvfp4GroupedGemmPlan();

  CutlassNvfp4GroupedGemmPlan(const CutlassNvfp4GroupedGemmPlan&) = delete;
  CutlassNvfp4GroupedGemmPlan& operator=(const CutlassNvfp4GroupedGemmPlan&) = delete;

  bool valid() const;
  int group_count() const;
  int m() const;
  int n() const;
  int k() const;

  // Run the grouped GEMM using pre-allocated metadata.
  // Only the pointer arrays (A, B, C, D, SFA, SFB), per-group alphas, and beta
  // change per call.
  // device_*_ptrs are device-resident arrays of group_count pointers.
  bool Run(
      const void* device_a_ptrs,
      const void* device_a_sf_ptrs,
      const void* device_b_ptrs,
      const void* device_b_sf_ptrs,
      const void* device_c_ptrs,
      void* device_d_ptrs,
      const float* device_alpha_values,
      float beta,
      cudaStream_t stream) const;

 private:
  struct Impl;
  explicit CutlassNvfp4GroupedGemmPlan(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// Run a grouped NVFP4 GEMM: group_count matmuls with the same M/N/K
// but different A/B/C/D pointers per group. All groups execute in one
// kernel launch.
//
// A (activations): NVFP4 block-scaled, RowMajor, [M, K] per group
// B (weights):     NVFP4 block-scaled, ColumnMajor, [N, K] per group
// D (output):      FP32, RowMajor, [M, N] per group
//
// Scale factors use the SM120 block-scaled 128x4 interleaved layout,
// which is identical to our existing SwizzleRowMajorNvfp4ScalesForExecution.
//
// host_a_ptrs, host_b_ptrs etc. are HOST arrays of group_count DEVICE pointers.
// host_a_sf_ptrs, host_b_sf_ptrs are HOST arrays of group_count DEVICE scale pointers.
//
// Each per-group alpha is applied as: D_i = alpha_i * (A_i @ B_i) + beta * C_i
// device_alpha_values is a device-resident array of group_count alpha scalars.
bool RunCutlassNvfp4GroupedGemm(
    int group_count,
    int m, int n, int k,
    const void* const* host_a_ptrs,
    const void* const* host_a_sf_ptrs,
    const void* const* host_b_ptrs,
    const void* const* host_b_sf_ptrs,
    const void* const* host_c_ptrs,
    void* const* host_d_ptrs,
    const float* device_alpha_values,
    float beta,
    cudaStream_t stream);

// Same as above but all pointer arrays are already on device.
// No H→D copies for pointer metadata — fully device-resident hot path.
// Each device_*_ptrs argument is a device-resident array of group_count
// device pointers (i.e., a pointer-to-pointer where the outer array is on device).
bool RunCutlassNvfp4GroupedGemmFromDevice(
    int group_count,
    int m, int n, int k,
    const void* device_a_ptrs,
    const void* device_a_sf_ptrs,
    const void* device_b_ptrs,
    const void* device_b_sf_ptrs,
    const void* device_c_ptrs,
    void* device_d_ptrs,
    const float* device_alpha_values,
    float beta,
    cudaStream_t stream);

}  // namespace nemotron
