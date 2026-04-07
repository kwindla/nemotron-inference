#pragma once

#include <functional>
#include <memory>
#include <optional>

#include <cuda_runtime.h>

#include "nemotron/cublaslt_handle.h"
#include "nemotron/gemm_catalog.h"
#include "nemotron/kernel_catalog.h"
#include "nemotron/linear_op.h"
#include "nemotron/model_schedule.h"
#include "nemotron/primitive_ops.h"
#include "nemotron/request_context.h"
#include "nemotron/scaled_fp8_linear.h"

namespace nemotron {

struct MambaLayerConfig {
  std::size_t layer_index = 0;
  std::size_t hidden_size = 0;
  std::size_t intermediate_size = 0;
  std::size_t num_heads = 0;
  std::size_t head_dim = 0;
  std::size_t state_size = 0;
  std::size_t n_groups = 0;
  std::size_t conv_kernel_size = 0;
  std::size_t conv_state_offset_elems = 0;
  std::size_t ssm_state_offset_elems = 0;
  float input_rms_epsilon = 1.0e-5f;
  float mixer_rms_epsilon = 1.0e-5f;
  float time_step_min = 0.0f;
};

struct MambaLayerBindings {
  const KernelTensorDescriptor* input_norm_weight = nullptr;
  const KernelTensorDescriptor* mixer_norm_weight = nullptr;
  const GemmDescriptor* in_proj_gemm_weight = nullptr;
  const KernelTensorDescriptor* in_proj_kernel_weight = nullptr;
  const KernelTensorDescriptor* in_proj_weight_scale = nullptr;
  const KernelTensorDescriptor* in_proj_input_scale = nullptr;
  const KernelTensorDescriptor* conv1d_weight = nullptr;
  const KernelTensorDescriptor* conv1d_bias = nullptr;
  const KernelTensorDescriptor* A_log = nullptr;
  const KernelTensorDescriptor* D = nullptr;
  const KernelTensorDescriptor* dt_bias = nullptr;
  const GemmDescriptor* out_proj_gemm_weight = nullptr;
  const KernelTensorDescriptor* out_proj_kernel_weight = nullptr;
  const KernelTensorDescriptor* out_proj_weight_scale = nullptr;
  const KernelTensorDescriptor* out_proj_input_scale = nullptr;
};

struct MambaLayerPreparedBindings {
  std::unique_ptr<DeviceTensorFp32> input_norm_weight;
  std::unique_ptr<DeviceTensorFp32> mixer_norm_weight;
  std::unique_ptr<DeviceTensorFp32> conv1d_weight;
  std::unique_ptr<DeviceTensorFp32> conv1d_bias;
  std::unique_ptr<DeviceTensorFp32> A_log;
  std::unique_ptr<DeviceTensorFp32> D;
  std::unique_ptr<DeviceTensorFp32> dt_bias;
  std::unique_ptr<UploadedLinearOp> in_proj_dense;
  std::unique_ptr<UploadedLinearOp> out_proj_dense;
  std::unique_ptr<ScaledFp8LinearOp> in_proj_scaled_fp8;
  std::unique_ptr<ScaledFp8LinearOp> out_proj_scaled_fp8;
};

struct MambaLayerRunTrace {
  std::vector<float> norm_output;
  std::vector<float> in_proj_output;
  std::vector<float> scan_output;
  std::vector<float> projected_output;
};

using MambaLayerBuildTimingSink = std::function<void(const char*, double)>;

std::optional<MambaLayerBindings> BuildMambaLayerBindings(
    const LayerScheduleEntry& layer,
    const KernelCatalog& kernel_catalog,
    const GemmCatalog& gemm_catalog);

class MambaLayerSlice {
 public:
  static std::unique_ptr<MambaLayerSlice> Create(
      const MambaLayerConfig& config,
      const MambaLayerBindings& bindings,
      MambaLayerBuildTimingSink timing_sink = {});
  static std::unique_ptr<MambaLayerSlice> CreatePrepared(
      const MambaLayerConfig& config,
      MambaLayerPreparedBindings bindings);

  MambaLayerSlice(MambaLayerSlice&&) noexcept;
  MambaLayerSlice& operator=(MambaLayerSlice&&) noexcept;
  ~MambaLayerSlice();

  MambaLayerSlice(const MambaLayerSlice&) = delete;
  MambaLayerSlice& operator=(const MambaLayerSlice&) = delete;

  bool valid() const;
  const MambaLayerConfig& config() const;

  bool Run(
      CublasLtHandle& cublas_handle,
      GemmHeuristicCache* heuristic_cache,
      RequestExecutionContext& request_context,
      const DeviceTensorFp32& input,
      DeviceTensorFp32* output,
      MambaLayerRunTrace* trace = nullptr,
      cudaStream_t stream = nullptr) const;
  bool Run(
      CublasLtHandle& cublas_handle,
      GemmHeuristicCache* heuristic_cache,
      RequestExecutionContext& request_context,
      const DeviceTensorBf16& input,
      DeviceTensorBf16* output,
      MambaLayerRunTrace* trace = nullptr,
      cudaStream_t stream = nullptr) const;

 private:
  struct Impl;

  explicit MambaLayerSlice(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace nemotron
