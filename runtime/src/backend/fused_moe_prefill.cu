#include "nemotron/fused_moe_prefill.h"

#include <cuda_runtime.h>

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <vector>

#include "nemotron/gemm_execution.h"
#include "nemotron/gemm_planner.h"
#include "nemotron/nvfp4_gemm_runner.h"
#include "nemotron/nvfp4_packing.h"
#include "nemotron/nvfp4_scale_layout.h"
#include "nemotron/primitive_ops.h"

namespace nemotron {
namespace {

constexpr const char* kNvfp4ActivationTensorScaleEnvVar =
    "NEMOTRON_FORWARD_NVFP4_ACTIVATION_TENSOR_SCALE";

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

template <typename T>
bool CopyDeviceBufferToHost(
    const T* device_data,
    std::size_t count,
    std::vector<T>* output) {
  if (device_data == nullptr || output == nullptr) {
    return false;
  }
  output->assign(count, T{});
  return count == 0 ||
         CheckCuda(cudaMemcpy(
             output->data(),
             device_data,
             count * sizeof(T),
             cudaMemcpyDeviceToHost));
}

std::optional<float> ParsePositiveFloatEnv(const char* env_var) {
  const char* value = std::getenv(env_var);
  if (value == nullptr || value[0] == '\0') {
    return std::nullopt;
  }

  errno = 0;
  char* end = nullptr;
  const float parsed = std::strtof(value, &end);
  if (end == value || (end != nullptr && *end != '\0') || errno == ERANGE ||
      !std::isfinite(parsed) || parsed <= 0.0f) {
    return std::nullopt;
  }
  return parsed;
}

Nvfp4PackOptions RuntimeMoeNvfp4PackOptions() {
  Nvfp4PackOptions options;
  options.execution_scale_layout = Nvfp4ScaleLayout::kSwizzled128x4;
  if (const auto fixed_tensor_scale = ParsePositiveFloatEnv(kNvfp4ActivationTensorScaleEnvVar);
      fixed_tensor_scale.has_value()) {
    options.fixed_tensor_scale = *fixed_tensor_scale;
  }
  return options;
}

bool ValidFusedNvfp4WeightView(const FusedNvfp4WeightView& weight) {
  return weight.packed_data != nullptr &&
         weight.block_scales_data != nullptr &&
         weight.matmul_block_scales_data != nullptr &&
         weight.tensor_scale_data != nullptr &&
         weight.output_rows > 0 &&
         weight.input_cols > 0;
}

Nvfp4PackedMatrixDeviceView MakeNvfp4PackedMatrixDeviceView(
    const FusedNvfp4WeightView& weight) {
  return Nvfp4PackedMatrixDeviceView{
      weight.packed_data,
      PackedFp4Bytes(weight.output_rows, weight.input_cols),
      weight.matmul_block_scales_data,
      ExecutionNvfp4ScaleBytes(weight.output_rows, weight.input_cols),
      weight.tensor_scale_data,
      sizeof(float),
      weight.output_rows,
      weight.input_cols,
  };
}

bool DescriptorMatchesWeightView(
    const GemmDescriptor* descriptor,
    const FusedNvfp4WeightView& weight) {
  return descriptor != nullptr &&
         descriptor->kernel_family == GemmKernelFamily::kCublasLtNvfp4BlockScaled &&
         descriptor->output_rows == weight.output_rows &&
         descriptor->input_cols == weight.input_cols;
}

std::optional<CublasLtGemmPlan> BuildRuntimeNvfp4GemmPlan(
    const GemmDescriptor& descriptor,
    const Nvfp4PackedMatrixDeviceView& weight_view,
    std::size_t rows,
    GemmHeuristicCache* heuristic_cache) {
  auto runtime_launch_plan = BuildGemmLaunchPlan(descriptor, rows);
  if (!runtime_launch_plan.has_value()) {
    return std::nullopt;
  }
  runtime_launch_plan->packed_bytes = ByteRangeView{
      weight_view.packed_data,
      weight_view.packed_nbytes,
  };
  runtime_launch_plan->block_scales_bytes = ByteRangeView{
      weight_view.block_scales_data,
      weight_view.block_scales_nbytes,
  };
  runtime_launch_plan->tensor_scale_bytes = ByteRangeView{
      reinterpret_cast<const std::uint8_t*>(weight_view.tensor_scale_data),
      weight_view.tensor_scale_nbytes,
  };
  const auto execution = PrepareGemmExecution(*runtime_launch_plan, heuristic_cache);
  if (!execution.has_value()) {
    return std::nullopt;
  }
  return BuildCublasLtGemmPlan(*execution);
}

}  // namespace

bool RunFusedMoePrefill(const FusedMoePrefillParams& params) {
  if (params.cublas_handle == nullptr ||
      !params.cublas_handle->valid() ||
      params.token_count == 0 ||
      params.hidden_size == 0 ||
      params.routed_expert_intermediate_size == 0 ||
      params.shared_expert_intermediate_size == 0 ||
      params.n_routed_experts == 0 ||
      params.top_k == 0 ||
      params.token_count >
          (std::numeric_limits<std::size_t>::max() / params.top_k) ||
      params.selection_count < (params.token_count * params.top_k) ||
      !ValidFusedNvfp4WeightView(params.shared_up) ||
      !ValidFusedNvfp4WeightView(params.shared_down) ||
      !DescriptorMatchesWeightView(params.shared_up_descriptor, params.shared_up) ||
      !DescriptorMatchesWeightView(params.shared_down_descriptor, params.shared_down) ||
      params.routed_up_descriptors == nullptr ||
      params.routed_down_descriptors == nullptr ||
      params.routed_up == nullptr ||
      params.routed_down == nullptr ||
      params.input == nullptr ||
      params.normalized == nullptr ||
      params.output == nullptr ||
      params.routed_output == nullptr ||
      params.gather_scratch == nullptr ||
      params.expert_up_scratch == nullptr ||
      params.shared_up_scratch == nullptr ||
      params.expert_offsets == nullptr ||
      params.sorted_token_indices == nullptr ||
      params.sorted_token_weights == nullptr ||
      params.active_expert_count == nullptr ||
      params.active_expert_ids == nullptr) {
    return false;
  }

  auto input = DeviceTensorFp32::CreateView(
      {params.token_count, params.hidden_size},
      const_cast<float*>(params.input));
  auto normalized = DeviceTensorFp32::CreateView(
      {params.token_count, params.hidden_size},
      const_cast<float*>(params.normalized));
  auto output = DeviceTensorFp32::CreateView(
      {params.token_count, params.hidden_size},
      params.output);
  auto routed_output = DeviceTensorFp32::CreateView(
      {params.token_count, params.hidden_size},
      params.routed_output);
  auto shared_up = DeviceTensorFp32::CreateView(
      {params.token_count, params.shared_expert_intermediate_size},
      params.shared_up_scratch);
  if (!input || !normalized || !output || !routed_output || !shared_up) {
    return false;
  }

  int active_expert_count = 0;
  if (!CheckCuda(cudaMemcpy(
          &active_expert_count,
          params.active_expert_count,
          sizeof(active_expert_count),
          cudaMemcpyDeviceToHost)) ||
      active_expert_count < 0 ||
      static_cast<std::size_t>(active_expert_count) > params.n_routed_experts) {
    return false;
  }

  std::vector<int> expert_offsets_host;
  if (!CopyDeviceBufferToHost(
          params.expert_offsets,
          params.n_routed_experts + 1,
          &expert_offsets_host) ||
      expert_offsets_host.size() != params.n_routed_experts + 1) {
    return false;
  }

  const std::size_t selection_count = params.token_count * params.top_k;
  if (expert_offsets_host.front() != 0 ||
      expert_offsets_host.back() < 0 ||
      static_cast<std::size_t>(expert_offsets_host.back()) > selection_count) {
    return false;
  }
  for (std::size_t expert_index = 0; expert_index < params.n_routed_experts; ++expert_index) {
    if (expert_offsets_host[expert_index] > expert_offsets_host[expert_index + 1]) {
      return false;
    }
  }

  std::vector<int> active_expert_ids_host;
  if (active_expert_count > 0 &&
      (!CopyDeviceBufferToHost(
           params.active_expert_ids,
           static_cast<std::size_t>(active_expert_count),
           &active_expert_ids_host) ||
       active_expert_ids_host.size() != static_cast<std::size_t>(active_expert_count))) {
    return false;
  }

  const Nvfp4PackOptions pack_options = RuntimeMoeNvfp4PackOptions();
  for (int active_slot = 0; active_slot < active_expert_count; ++active_slot) {
    const int expert_id_int = active_expert_ids_host[active_slot];
    if (expert_id_int < 0) {
      return false;
    }
    const std::size_t expert_id = static_cast<std::size_t>(expert_id_int);
    if (expert_id >= params.n_routed_experts ||
        !ValidFusedNvfp4WeightView(params.routed_up[expert_id]) ||
        !ValidFusedNvfp4WeightView(params.routed_down[expert_id]) ||
        !DescriptorMatchesWeightView(
            params.routed_up_descriptors[expert_id],
            params.routed_up[expert_id]) ||
        !DescriptorMatchesWeightView(
            params.routed_down_descriptors[expert_id],
            params.routed_down[expert_id])) {
      return false;
    }

    const int expert_offset_int = expert_offsets_host[expert_id];
    const int next_expert_offset_int = expert_offsets_host[expert_id + 1];
    if (expert_offset_int < 0 ||
        next_expert_offset_int < expert_offset_int) {
      return false;
    }
    const std::size_t expert_offset = static_cast<std::size_t>(expert_offset_int);
    const std::size_t expert_token_count =
        static_cast<std::size_t>(next_expert_offset_int - expert_offset_int);
    if (expert_token_count == 0) {
      continue;
    }

    auto gather_view = DeviceTensorFp32::CreateView(
        {expert_token_count, params.hidden_size},
        params.gather_scratch + (expert_offset * params.hidden_size));
    auto expert_up_view = DeviceTensorFp32::CreateView(
        {expert_token_count, params.routed_expert_intermediate_size},
        params.expert_up_scratch +
            (expert_offset * params.routed_expert_intermediate_size));
    if (!gather_view || !expert_up_view) {
      return false;
    }

    const Nvfp4PackedMatrixDeviceView up_weight_view =
        MakeNvfp4PackedMatrixDeviceView(params.routed_up[expert_id]);
    const Nvfp4PackedMatrixDeviceView down_weight_view =
        MakeNvfp4PackedMatrixDeviceView(params.routed_down[expert_id]);
    const auto routed_up_plan = BuildRuntimeNvfp4GemmPlan(
        *params.routed_up_descriptors[expert_id],
        up_weight_view,
        expert_token_count,
        params.heuristic_cache);
    const auto routed_down_plan = BuildRuntimeNvfp4GemmPlan(
        *params.routed_down_descriptors[expert_id],
        down_weight_view,
        expert_token_count,
        params.heuristic_cache);
    if (!up_weight_view.valid() ||
        !down_weight_view.valid() ||
        !routed_up_plan.has_value() ||
        !routed_down_plan.has_value() ||
        !GatherRowsFp32(
             *normalized,
             params.sorted_token_indices + expert_offset,
             gather_view.get()) ||
        !RunNvfp4RowMajorFp32SourceToDevice(
             *params.cublas_handle,
             *routed_up_plan,
             *gather_view,
             up_weight_view,
             expert_up_view.get(),
             pack_options)
             .has_value() ||
        !Relu2InPlaceFp32(expert_up_view.get()) ||
        !RunNvfp4RowMajorFp32SourceToDevice(
             *params.cublas_handle,
             *routed_down_plan,
             *expert_up_view,
             down_weight_view,
             gather_view.get(),
             pack_options)
             .has_value() ||
        !ScatterAddWeightedRowsFp32(
             *gather_view,
             params.sorted_token_indices + expert_offset,
             params.sorted_token_weights + expert_offset,
             routed_output.get())) {
      return false;
    }
  }

  const Nvfp4PackedMatrixDeviceView shared_up_weight_view =
      MakeNvfp4PackedMatrixDeviceView(params.shared_up);
  const Nvfp4PackedMatrixDeviceView shared_down_weight_view =
      MakeNvfp4PackedMatrixDeviceView(params.shared_down);
  const auto shared_up_plan = BuildRuntimeNvfp4GemmPlan(
      *params.shared_up_descriptor,
      shared_up_weight_view,
      params.token_count,
      params.heuristic_cache);
  const auto shared_down_plan = BuildRuntimeNvfp4GemmPlan(
      *params.shared_down_descriptor,
      shared_down_weight_view,
      params.token_count,
      params.heuristic_cache);
  if (!shared_up_weight_view.valid() ||
      !shared_down_weight_view.valid() ||
      !shared_up_plan.has_value() ||
      !shared_down_plan.has_value() ||
      !RunNvfp4RowMajorFp32SourceToDevice(
           *params.cublas_handle,
           *shared_up_plan,
           *normalized,
           shared_up_weight_view,
           shared_up.get(),
           pack_options)
           .has_value() ||
      !Relu2InPlaceFp32(shared_up.get()) ||
      !RunNvfp4RowMajorFp32SourceToDevice(
           *params.cublas_handle,
           *shared_down_plan,
           *shared_up,
           shared_down_weight_view,
           output.get(),
           pack_options)
           .has_value() ||
      !ResidualAddFp32(*routed_output, *output, output.get()) ||
      !ResidualAddFp32(*input, *output, output.get())) {
    return false;
  }

  return true;
}

}  // namespace nemotron
