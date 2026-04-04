#pragma once

#include <cstddef>
#include <memory>
#include <optional>

#include "nemotron/cublaslt_handle.h"
#include "nemotron/gemm_catalog.h"
#include "nemotron/kernel_catalog.h"
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
  float time_step_min = 1.0e-3f;
};

struct MambaLayerStateLayout {
  std::size_t conv_state_offset_elems = 0;
  std::size_t conv_state_elems = 0;
  std::size_t ssm_state_offset_elems = 0;
  std::size_t ssm_state_elems = 0;

  bool valid() const {
    return conv_state_elems != 0 && ssm_state_elems != 0;
  }
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

struct MambaLayerExecutionCounters {
  std::size_t native_multi_token_runs = 0;
  std::size_t native_multi_token_tokens = 0;
  std::size_t row_replay_runs = 0;
  std::size_t row_replay_tokens = 0;
};

struct MambaLayerRunTrace {
  std::vector<float> norm_output;
  std::vector<float> in_proj_output;
  std::vector<float> scan_output;
  std::vector<float> projected_output;
};

void ResetMambaLayerExecutionCounters();
MambaLayerExecutionCounters GetMambaLayerExecutionCounters();

std::optional<MambaLayerBindings> BuildMambaLayerBindings(
    const LayerScheduleEntry& layer,
    const KernelCatalog& kernel_catalog,
    const GemmCatalog& gemm_catalog);

std::optional<MambaLayerStateLayout> BuildMambaLayerStateLayout(
    const MambaLayerConfig& config);

class MambaLayerSlice {
 public:
  static std::unique_ptr<MambaLayerSlice> Create(
      const MambaLayerConfig& config,
      const MambaLayerBindings& bindings);

  MambaLayerSlice(MambaLayerSlice&&) noexcept;
  MambaLayerSlice& operator=(MambaLayerSlice&&) noexcept;
  ~MambaLayerSlice();

  MambaLayerSlice(const MambaLayerSlice&) = delete;
  MambaLayerSlice& operator=(const MambaLayerSlice&) = delete;

  bool valid() const;
  const MambaLayerConfig& config() const;
  const MambaLayerStateLayout& state_layout() const;

  bool Run(
      CublasLtHandle& cublas_handle,
      GemmHeuristicCache* heuristic_cache,
      RequestExecutionContext& request_context,
      const DeviceTensorBf16& input,
      DeviceTensorBf16* residual,
      DeviceTensorBf16* output,
      MambaLayerRunTrace* trace = nullptr) const;

  bool Run(
      CublasLtHandle& cublas_handle,
      GemmHeuristicCache* heuristic_cache,
      RequestExecutionContext& request_context,
      const DeviceTensorFp32& input,
      DeviceTensorFp32* output,
      MambaLayerRunTrace* trace = nullptr) const;

 private:
  struct Impl;

  explicit MambaLayerSlice(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace nemotron
