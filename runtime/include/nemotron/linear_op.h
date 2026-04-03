#pragma once

#include <memory>
#include <optional>
#include <vector>

#include <cuda_runtime.h>

#include "nemotron/cublaslt_gemm_plan.h"
#include "nemotron/cublaslt_handle.h"
#include "nemotron/dense_weight.h"
#include "nemotron/dense_gemm_runner.h"
#include "nemotron/device_tensor.h"
#include "nemotron/gemm_execution.h"
#include "nemotron/gemm_planner.h"
#include "nemotron/nvfp4_gemm_runner.h"
#include "nemotron/nvfp4_weight.h"

namespace nemotron {

class UploadedLinearOp {
 public:
  static std::unique_ptr<UploadedLinearOp> Create(const GemmDescriptor& descriptor);
  static std::unique_ptr<UploadedLinearOp> CreateDenseView(
      const GemmDescriptor& descriptor,
      std::unique_ptr<DeviceDenseWeightFp32> weight_view);
  static std::unique_ptr<UploadedLinearOp> CreateDenseBf16View(
      const GemmDescriptor& descriptor,
      std::unique_ptr<DeviceTensorBf16> weight_view);
  static std::unique_ptr<UploadedLinearOp> CreateNvfp4View(
      const GemmDescriptor& descriptor,
      std::unique_ptr<DeviceNvfp4Weight> weight_view);

  UploadedLinearOp(UploadedLinearOp&&) noexcept;
  UploadedLinearOp& operator=(UploadedLinearOp&&) noexcept;
  ~UploadedLinearOp();

  UploadedLinearOp(const UploadedLinearOp&) = delete;
  UploadedLinearOp& operator=(const UploadedLinearOp&) = delete;

  bool valid() const;
  std::size_t output_rows() const;
  std::size_t input_cols() const;
  GemmKernelFamily kernel_family() const;
  const DeviceNvfp4Weight* nvfp4_weight() const;

  bool Run(
      CublasLtHandle& handle,
      GemmHeuristicCache* heuristic_cache,
      const DeviceTensorFp32& activations,
      DeviceTensorFp32* output,
      cudaStream_t stream = nullptr) const;

  bool Run(
      CublasLtHandle& handle,
      GemmHeuristicCache* heuristic_cache,
      const DeviceTensorBf16& activations,
      DeviceTensorFp32* output,
      cudaStream_t stream = nullptr) const;

  bool Run(
      CublasLtHandle& handle,
      GemmHeuristicCache* heuristic_cache,
      const DeviceTensorBf16& activations,
      DeviceTensorBf16* output,
      cudaStream_t stream = nullptr) const;

 private:
  struct Impl;

  explicit UploadedLinearOp(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

std::optional<std::vector<float>> ReadVectorWeightToHostFp32(
    const KernelTensorDescriptor& descriptor);
std::unique_ptr<DeviceTensorFp32> UploadVectorWeightToDeviceFp32(
    const KernelTensorDescriptor& descriptor);

}  // namespace nemotron
