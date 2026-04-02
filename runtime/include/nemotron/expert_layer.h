#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

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

struct ExpertLayerConfig {
  std::size_t layer_index = 0;
  std::size_t hidden_size = 0;
  std::size_t moe_latent_size = 0;
  std::size_t routed_expert_intermediate_size = 0;
  std::size_t shared_expert_intermediate_size = 0;
  std::size_t n_routed_experts = 0;
  std::size_t top_k = 0;
  std::size_t n_group = 1;
  std::size_t topk_group = 1;
  float rms_epsilon = 1.0e-5f;
  float routed_scaling_factor = 1.0f;
  bool norm_topk_prob = true;
};

struct ExpertWeightPair {
  const GemmDescriptor* up_proj = nullptr;
  const GemmDescriptor* down_proj = nullptr;
  const KernelTensorDescriptor* up_input_scale = nullptr;
  const KernelTensorDescriptor* down_input_scale = nullptr;
};

struct ExpertLayerBindings {
  const KernelTensorDescriptor* input_norm_weight = nullptr;
  const GemmDescriptor* gate_weight = nullptr;
  const KernelTensorDescriptor* gate_score_correction_bias = nullptr;
  const GemmDescriptor* fc1_latent_gemm_weight = nullptr;
  const KernelTensorDescriptor* fc1_latent_kernel_weight = nullptr;
  const KernelTensorDescriptor* fc1_latent_weight_scale = nullptr;
  const KernelTensorDescriptor* fc1_latent_input_scale = nullptr;
  const GemmDescriptor* fc2_latent_weight = nullptr;
  const GemmDescriptor* shared_up_gemm_weight = nullptr;
  const KernelTensorDescriptor* shared_up_kernel_weight = nullptr;
  const KernelTensorDescriptor* shared_up_weight_scale = nullptr;
  const KernelTensorDescriptor* shared_up_input_scale = nullptr;
  const GemmDescriptor* shared_down_gemm_weight = nullptr;
  const KernelTensorDescriptor* shared_down_kernel_weight = nullptr;
  const KernelTensorDescriptor* shared_down_weight_scale = nullptr;
  const KernelTensorDescriptor* shared_down_input_scale = nullptr;
  std::vector<ExpertWeightPair> routed_experts;
};

struct ExpertLayerPreparedExpert {
  GemmDescriptor up_descriptor;
  GemmDescriptor down_descriptor;
  std::optional<float> up_tensor_scale;
  std::optional<float> down_tensor_scale;
  std::optional<float> up_input_scale;
  std::optional<float> down_input_scale;
  std::unique_ptr<UploadedLinearOp> up_proj;
  std::unique_ptr<UploadedLinearOp> down_proj;
};

struct ExpertLayerPreparedBindings {
  std::unique_ptr<DeviceTensorFp32> input_norm_weight;
  std::unique_ptr<DeviceTensorFp32> gate_score_correction_bias_device;
  std::unique_ptr<UploadedLinearOp> gate_weight;
  std::unique_ptr<UploadedLinearOp> fc1_latent_dense;
  std::unique_ptr<ScaledFp8LinearOp> fc1_latent_scaled_fp8;
  std::unique_ptr<UploadedLinearOp> fc2_latent;
  std::unique_ptr<UploadedLinearOp> shared_up_dense;
  std::unique_ptr<ScaledFp8LinearOp> shared_up_scaled_fp8;
  std::unique_ptr<UploadedLinearOp> shared_down_dense;
  std::unique_ptr<ScaledFp8LinearOp> shared_down_scaled_fp8;
  std::unique_ptr<UploadedLinearOp> shared_down_nvfp4;
  std::vector<ExpertLayerPreparedExpert> routed_experts;
};

struct ExpertSelection {
  std::size_t expert_index = 0;
  float weight = 0.0f;
};

struct ExpertLayerRunTrace {
  std::vector<float> normalized_input;
  std::vector<float> router_logits;
  std::vector<ExpertSelection> selected_experts;
  std::vector<float> latent_output;
  std::vector<float> routed_latent_output;
  std::vector<std::size_t> routed_expert_order;
  std::vector<float> routed_expert_activated_hidden;
  std::vector<float> routed_expert_outputs;
  std::vector<float> routed_expert_weighted_contributions;
  std::vector<float> projected_routed_output;
  std::vector<float> shared_output;
  std::vector<float> mixer_output;
};

using ExpertLayerBuildTimingSink = std::function<void(const char*, double)>;

std::optional<ExpertLayerBindings> BuildExpertLayerBindings(
    const LayerScheduleEntry& layer,
    const KernelCatalog& kernel_catalog,
    const GemmCatalog& gemm_catalog,
    std::size_t routed_expert_count);

class ExpertLayerSlice {
 public:
  static std::unique_ptr<ExpertLayerSlice> Create(
      const ExpertLayerConfig& config,
      const ExpertLayerBindings& bindings,
      ExpertLayerBuildTimingSink timing_sink = {});
  static std::unique_ptr<ExpertLayerSlice> CreatePrepared(
      const ExpertLayerConfig& config,
      ExpertLayerPreparedBindings bindings);

  ExpertLayerSlice(ExpertLayerSlice&&) noexcept;
  ExpertLayerSlice& operator=(ExpertLayerSlice&&) noexcept;
  ~ExpertLayerSlice();

  ExpertLayerSlice(const ExpertLayerSlice&) = delete;
  ExpertLayerSlice& operator=(const ExpertLayerSlice&) = delete;

  bool valid() const;
  const ExpertLayerConfig& config() const;
  bool routed_experts_contiguous() const;

  bool Run(
      CublasLtHandle& cublas_handle,
      GemmHeuristicCache* heuristic_cache,
      const DeviceTensorFp32& input,
      DeviceTensorFp32* output,
      ExpertLayerRunTrace* trace = nullptr,
      cudaStream_t stream = nullptr) const;
  bool Run(
      CublasLtHandle& cublas_handle,
      GemmHeuristicCache* heuristic_cache,
      const DeviceTensorBf16& input,
      DeviceTensorBf16* output,
      ExpertLayerRunTrace* trace = nullptr,
      cudaStream_t stream = nullptr) const;

  bool RunWithRequestContext(
      CublasLtHandle& cublas_handle,
      GemmHeuristicCache* heuristic_cache,
      RequestExecutionContext& request_context,
      const DeviceTensorFp32& input,
      DeviceTensorFp32* output,
      ExpertLayerRunTrace* trace = nullptr,
      cudaStream_t stream = nullptr) const;
  bool RunWithRequestContext(
      CublasLtHandle& cublas_handle,
      GemmHeuristicCache* heuristic_cache,
      RequestExecutionContext& request_context,
      const DeviceTensorBf16& input,
      DeviceTensorBf16* output,
      ExpertLayerRunTrace* trace = nullptr,
      cudaStream_t stream = nullptr) const;

 private:
  struct Impl;

  explicit ExpertLayerSlice(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace nemotron
