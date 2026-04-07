#pragma once

#include <memory>
#include <optional>

#include <cuda_runtime.h>

#include "nemotron/cublaslt_handle.h"
#include "nemotron/cudnn_handle.h"
#include "nemotron/gemm_catalog.h"
#include "nemotron/gemm_planner.h"
#include "nemotron/kernel_catalog.h"
#include "nemotron/linear_op.h"
#include "nemotron/model_schedule.h"
#include "nemotron/primitive_ops.h"
#include "nemotron/request_context.h"
#include "nemotron/scaled_fp8_linear.h"

namespace nemotron {

struct AttentionLayerConfig {
  std::size_t layer_index = 0;
  std::size_t hidden_size = 0;
  std::size_t query_head_count = 0;
  std::size_t kv_head_count = 0;
  std::size_t head_dim = 0;
  float rms_epsilon = 1e-5f;
};

struct AttentionLayerBindings {
  const KernelTensorDescriptor* norm_weight = nullptr;
  const GemmDescriptor* q_proj = nullptr;
  const GemmDescriptor* k_proj = nullptr;
  const GemmDescriptor* v_proj = nullptr;
  const GemmDescriptor* o_proj = nullptr;
  const KernelTensorDescriptor* o_proj_kernel_weight = nullptr;
  const KernelTensorDescriptor* o_proj_weight_scale = nullptr;
  const KernelTensorDescriptor* o_proj_input_scale = nullptr;
};

struct AttentionLayerPreparedBindings {
  std::unique_ptr<DeviceTensorFp32> norm_weight;
  std::unique_ptr<UploadedLinearOp> q_proj;
  std::unique_ptr<UploadedLinearOp> k_proj;
  std::unique_ptr<UploadedLinearOp> v_proj;
  std::unique_ptr<UploadedLinearOp> o_proj;
  std::unique_ptr<ScaledFp8LinearOp> o_proj_scaled_fp8;
};

std::optional<AttentionLayerBindings> BuildAttentionLayerBindings(
    const LayerScheduleEntry& layer,
    const KernelCatalog& kernel_catalog,
    const GemmCatalog& gemm_catalog);

class AttentionLayerSlice {
 public:
  static std::unique_ptr<AttentionLayerSlice> Create(
      const AttentionLayerConfig& config,
      const AttentionLayerBindings& bindings);
  static std::unique_ptr<AttentionLayerSlice> CreatePrepared(
      const AttentionLayerConfig& config,
      AttentionLayerPreparedBindings bindings);

  AttentionLayerSlice(AttentionLayerSlice&&) noexcept;
  AttentionLayerSlice& operator=(AttentionLayerSlice&&) noexcept;
  ~AttentionLayerSlice();

  AttentionLayerSlice(const AttentionLayerSlice&) = delete;
  AttentionLayerSlice& operator=(const AttentionLayerSlice&) = delete;

  bool valid() const;
  const AttentionLayerConfig& config() const;

  bool Run(
      CublasLtHandle& cublas_handle,
      const CudnnHandle& cudnn_handle,
      GemmHeuristicCache* heuristic_cache,
      RequestExecutionContext& request_context,
      const DeviceTensorFp32& input,
      DeviceTensorFp32* output,
      cudaStream_t stream = nullptr) const;
  bool Run(
      CublasLtHandle& cublas_handle,
      const CudnnHandle& cudnn_handle,
      GemmHeuristicCache* heuristic_cache,
      RequestExecutionContext& request_context,
      const DeviceTensorBf16& input,
      DeviceTensorBf16* output,
      cudaStream_t stream = nullptr) const;

 private:
  struct Impl;

  explicit AttentionLayerSlice(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace nemotron
