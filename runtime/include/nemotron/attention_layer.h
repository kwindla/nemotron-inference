#pragma once

#include <memory>
#include <optional>

#include "nemotron/cublaslt_handle.h"
#include "nemotron/cudnn_handle.h"
#include "nemotron/gemm_catalog.h"
#include "nemotron/gemm_planner.h"
#include "nemotron/kernel_catalog.h"
#include "nemotron/linear_op.h"
#include "nemotron/model_schedule.h"
#include "nemotron/primitive_ops.h"
#include "nemotron/request_context.h"

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
      std::size_t sequence_start,
      std::size_t total_sequence_length,
      const DeviceTensorFp32& input,
      DeviceTensorFp32* output) const;

 private:
  struct Impl;

  explicit AttentionLayerSlice(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace nemotron
