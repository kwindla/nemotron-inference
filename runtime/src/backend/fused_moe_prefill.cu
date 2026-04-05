#include "nemotron/fused_moe_prefill.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <vector>

#include "nemotron/gemm_execution.h"
#include "nemotron/fused_moe_grouped.h"
#include "nemotron/device_nvfp4_matrix.h"
#include "nemotron/gemm_planner.h"
#include "nemotron/nvfp4_gemm_runner.h"
#include "nemotron/nvfp4_packing.h"
#include "nemotron/nvfp4_scale_layout.h"
#include "nemotron/primitive_ops.h"

namespace nemotron {
namespace {

constexpr const char* kNvfp4ActivationTensorScaleEnvVar =
    "NEMOTRON_FORWARD_NVFP4_ACTIVATION_TENSOR_SCALE";
constexpr std::size_t kMinGroupedCutlassRows = 128;

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
  if (!normalized || !output || !routed_output || !shared_up) {
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

  // routed_up / routed_down are device arrays of FusedNvfp4WeightView.
  // Copy them to host so we can read per-expert views for GEMM plan building.
  std::vector<FusedNvfp4WeightView> routed_up_host;
  std::vector<FusedNvfp4WeightView> routed_down_host;
  if (!CopyDeviceBufferToHost(
          params.routed_up,
          params.n_routed_experts,
          &routed_up_host) ||
      !CopyDeviceBufferToHost(
          params.routed_down,
          params.n_routed_experts,
          &routed_down_host)) {
    return false;
  }

  const Nvfp4PackOptions pack_options = RuntimeMoeNvfp4PackOptions();

  std::vector<const std::uint8_t*> h_up_A_ptr(active_expert_count, nullptr);
  std::vector<const std::uint8_t*> h_up_A_scale(active_expert_count, nullptr);
  std::vector<const float*> h_up_A_global(active_expert_count, nullptr);
  std::vector<const std::uint8_t*> h_up_B_ptr(active_expert_count, nullptr);
  std::vector<const std::uint8_t*> h_up_B_scale(active_expert_count, nullptr);
  std::vector<const float*> h_up_B_global(active_expert_count, nullptr);
  std::vector<float*> h_up_D_ptr(active_expert_count, nullptr);
  std::vector<int32_t> h_expert_token_counts(active_expert_count, 0);

  std::vector<const std::uint8_t*> h_down_A_ptr(active_expert_count, nullptr);
  std::vector<const std::uint8_t*> h_down_A_scale(active_expert_count, nullptr);
  std::vector<const float*> h_down_A_global(active_expert_count, nullptr);
  std::vector<const std::uint8_t*> h_down_B_ptr(active_expert_count, nullptr);
  std::vector<const std::uint8_t*> h_down_B_scale(active_expert_count, nullptr);
  std::vector<const float*> h_down_B_global(active_expert_count, nullptr);
  std::vector<float*> h_down_D_ptr(active_expert_count, nullptr);

  std::uint8_t* pack_scratch_base = reinterpret_cast<std::uint8_t*>(params.nvfp4_pack_scratch);
  std::size_t pack_offset = 0;

  std::uint8_t* ws_base = reinterpret_cast<std::uint8_t*>(params.grouped_workspace_scratch);
  std::size_t ws_offset = 0;

  auto alloc_pack = [&](std::size_t bytes, std::size_t alignment = 128) -> std::uint8_t* {
      pack_offset = (pack_offset + alignment - 1) & ~(alignment - 1);
      std::uint8_t* ptr = pack_scratch_base + pack_offset;
      pack_offset += bytes;
      return ptr;
  };

  auto alloc_ws = [&](std::size_t bytes, std::size_t alignment = 16) -> std::uint8_t* {
      ws_offset = (ws_offset + alignment - 1) & ~(alignment - 1);
      std::uint8_t* ptr = ws_base + ws_offset;
      ws_offset += bytes;
      return ptr;
  };

  // Phase 1: Up Proj Packing
  for (int active_slot = 0; active_slot < active_expert_count; ++active_slot) {
    const int expert_id_int = active_expert_ids_host[active_slot];
    if (expert_id_int < 0) return false;
    const std::size_t expert_id = static_cast<std::size_t>(expert_id_int);

    const int expert_offset_int = expert_offsets_host[expert_id];
    const int next_expert_offset_int = expert_offsets_host[expert_id + 1];
    const std::size_t expert_offset = static_cast<std::size_t>(expert_offset_int);
    const std::size_t expert_token_count = static_cast<std::size_t>(next_expert_offset_int - expert_offset_int);
    if (expert_token_count == 0) continue;

    auto gather_view = DeviceTensorFp32::CreateView(
        {expert_token_count, params.hidden_size},
        params.gather_scratch + (expert_offset * params.hidden_size));
    
    if (!GatherRowsFp32(*normalized, params.sorted_token_indices + expert_offset, gather_view.get())) {
      return false;
    }

    std::size_t packed_bytes = expert_token_count * (params.hidden_size / 2);
    std::size_t padded_rows = (expert_token_count + 127) & ~127;
    std::size_t scale_bytes = (padded_rows / 128) * (params.hidden_size / 16) * 512;
    std::size_t block_scale_bytes = expert_token_count * (params.hidden_size / 16);

    std::uint8_t* packed_data = alloc_pack(packed_bytes);
    std::uint8_t* block_scales = alloc_pack(block_scale_bytes);
    std::uint8_t* matmul_scales = alloc_pack(scale_bytes);
    float* tensor_scale = reinterpret_cast<float*>(alloc_pack(sizeof(float), 4));

    if (!PackDeviceRowMajorFp32ToNvfp4Raw(
            *gather_view, packed_data, block_scales, matmul_scales, tensor_scale, pack_options)) {
        return false;
    }

    h_expert_token_counts[active_slot] = expert_token_count;
    h_up_A_ptr[active_slot] = packed_data;
    h_up_A_scale[active_slot] = matmul_scales;
    h_up_A_global[active_slot] = tensor_scale;

    h_up_B_ptr[active_slot] = routed_up_host[expert_id].packed_data;
    if (h_up_B_ptr[active_slot] == nullptr) {
      return false;
    }
    h_up_B_scale[active_slot] = routed_up_host[expert_id].matmul_block_scales_data;
    h_up_B_global[active_slot] = routed_up_host[expert_id].tensor_scale_data;

    h_up_D_ptr[active_slot] =
        params.expert_up_scratch + (expert_offset * params.routed_expert_intermediate_size);
  }

  std::vector<int> cutlass_slots;
  std::vector<int> fallback_slots;
  cutlass_slots.reserve(static_cast<std::size_t>(active_expert_count));
  fallback_slots.reserve(static_cast<std::size_t>(active_expert_count));
  for (int active_slot = 0; active_slot < active_expert_count; ++active_slot) {
    if (h_expert_token_counts[static_cast<std::size_t>(active_slot)] >=
        static_cast<int32_t>(kMinGroupedCutlassRows)) {
      cutlass_slots.push_back(active_slot);
    } else {
      fallback_slots.push_back(active_slot);
    }
  }

  auto copy_to_ws = [&](const auto& vec) {
      using T = typename std::decay<decltype(vec)>::type::value_type;
      std::size_t bytes = vec.size() * sizeof(T);
      T* d_ptr = reinterpret_cast<T*>(alloc_ws(bytes));
      if (bytes != 0) {
        cudaMemcpyAsync(d_ptr, vec.data(), bytes, cudaMemcpyHostToDevice, 0);
      }
      return d_ptr;
  };

  auto run_grouped_slots = [&](const auto& a_ptrs,
                               const auto& a_scales,
                               const auto& a_globals,
                               const auto& b_ptrs,
                               const auto& b_scales,
                               const auto& b_globals,
                               const auto& d_ptrs,
                               int k,
                               int n) {
    if (cutlass_slots.empty()) {
      return true;
    }

    auto subset = [&](const auto& values) {
      using T = typename std::decay<decltype(values)>::type::value_type;
      std::vector<T> filtered;
      filtered.reserve(cutlass_slots.size());
      for (int slot : cutlass_slots) {
        filtered.push_back(values[static_cast<std::size_t>(slot)]);
      }
      return filtered;
    };

    ws_offset = 0;
    const auto d_a_ptrs =
        reinterpret_cast<const std::uint8_t**>(copy_to_ws(subset(a_ptrs)));
    const auto d_a_scales =
        reinterpret_cast<const std::uint8_t**>(copy_to_ws(subset(a_scales)));
    const auto d_a_globals =
        reinterpret_cast<const float**>(copy_to_ws(subset(a_globals)));
    const auto d_b_ptrs =
        reinterpret_cast<const std::uint8_t**>(copy_to_ws(subset(b_ptrs)));
    const auto d_b_scales =
        reinterpret_cast<const std::uint8_t**>(copy_to_ws(subset(b_scales)));
    const auto d_b_globals =
        reinterpret_cast<const float**>(copy_to_ws(subset(b_globals)));
    const auto d_d_ptrs =
        reinterpret_cast<float**>(copy_to_ws(subset(d_ptrs)));
    const auto d_token_counts =
        reinterpret_cast<const int32_t*>(copy_to_ws(subset(h_expert_token_counts)));

    ws_offset = (ws_offset + 127) & ~127;
    GroupedMoeWorkspace cutlass_ws{
        ws_base + ws_offset,
        GroupedMoeWorkspaceBytes(static_cast<int>(cutlass_slots.size()))};
    return RunGroupedFp4Gemm(
        d_a_ptrs,
        d_a_scales,
        d_a_globals,
        d_b_ptrs,
        d_b_scales,
        d_b_globals,
        d_d_ptrs,
        d_token_counts,
        k,
        n,
        static_cast<int>(cutlass_slots.size()),
        cutlass_ws,
        nullptr);
  };

  auto run_fallback_slots = [&](std::size_t input_cols,
                                std::size_t output_cols,
                                float* source_base,
                                const auto& output_ptrs,
                                const auto& weights,
                                const auto descriptors) {
    for (int active_slot : fallback_slots) {
      const int expert_id_int = active_expert_ids_host[static_cast<std::size_t>(active_slot)];
      const std::size_t expert_id = static_cast<std::size_t>(expert_id_int);
      const int expert_offset_int = expert_offsets_host[expert_id];
      const int next_expert_offset_int = expert_offsets_host[expert_id + 1];
      const std::size_t expert_offset = static_cast<std::size_t>(expert_offset_int);
      const std::size_t expert_token_count =
          static_cast<std::size_t>(next_expert_offset_int - expert_offset_int);
      auto source_view = DeviceTensorFp32::CreateView(
          {expert_token_count, input_cols},
          source_base + (expert_offset * input_cols));
      auto output_view = DeviceTensorFp32::CreateView(
          {expert_token_count, output_cols},
          output_ptrs[static_cast<std::size_t>(active_slot)]);
      const auto weight_view = MakeNvfp4PackedMatrixDeviceView(weights[expert_id]);
      const auto plan = BuildRuntimeNvfp4GemmPlan(
          *descriptors[expert_id],
          weight_view,
          expert_token_count,
          params.heuristic_cache);
      if (!source_view ||
          !output_view ||
          !weight_view.valid() ||
          !plan.has_value() ||
          !RunNvfp4RowMajorFp32SourceToDevice(
              *params.cublas_handle,
              *plan,
              *source_view,
              weight_view,
              output_view.get(),
              pack_options,
              false)) {
        return false;
      }
    }
    return true;
  };

  // The current SM120 grouped FP4 kernel is numerically unstable below the
  // 128-row tile height covered by the existing CUTLASS smoke tests.
  if (!run_grouped_slots(
          h_up_A_ptr,
          h_up_A_scale,
          h_up_A_global,
          h_up_B_ptr,
          h_up_B_scale,
          h_up_B_global,
          h_up_D_ptr,
          static_cast<int>(params.hidden_size),
          static_cast<int>(params.routed_expert_intermediate_size)) ||
      !run_fallback_slots(
          params.hidden_size,
          params.routed_expert_intermediate_size,
          params.gather_scratch,
          h_up_D_ptr,
          routed_up_host,
          params.routed_up_descriptors)) {
    return false;
  }

  // Phase 2: Relu, Pack, and Down Proj setup
  ws_offset = 0;
  for (int active_slot = 0; active_slot < active_expert_count; ++active_slot) {
    const int expert_id_int = active_expert_ids_host[active_slot];
    const std::size_t expert_id = static_cast<std::size_t>(expert_id_int);
    const int expert_offset_int = expert_offsets_host[expert_id];
    const int next_expert_offset_int = expert_offsets_host[expert_id + 1];
    const std::size_t expert_offset = static_cast<std::size_t>(expert_offset_int);
    const std::size_t expert_token_count = static_cast<std::size_t>(next_expert_offset_int - expert_offset_int);
    if (expert_token_count == 0) continue;

    auto expert_up_view = DeviceTensorFp32::CreateView(
        {expert_token_count, params.routed_expert_intermediate_size},
        params.expert_up_scratch + (expert_offset * params.routed_expert_intermediate_size));

    if (!Relu2InPlaceFp32(expert_up_view.get())) {
      return false;
    }

    std::size_t packed_bytes = expert_token_count * (params.routed_expert_intermediate_size / 2);
    std::size_t padded_rows = (expert_token_count + 127) & ~127;
    std::size_t scale_bytes = (padded_rows / 128) * (params.routed_expert_intermediate_size / 16) * 512;
    std::size_t block_scale_bytes = expert_token_count * (params.routed_expert_intermediate_size / 16);

    std::uint8_t* packed_data = alloc_pack(packed_bytes);
    std::uint8_t* block_scales = alloc_pack(block_scale_bytes);
    std::uint8_t* matmul_scales = alloc_pack(scale_bytes);
    float* tensor_scale = reinterpret_cast<float*>(alloc_pack(sizeof(float), 4));

    if (!PackDeviceRowMajorFp32ToNvfp4Raw(
            *expert_up_view, packed_data, block_scales, matmul_scales, tensor_scale, pack_options)) {
        return false;
    }

    h_down_A_ptr[active_slot] = packed_data;
    h_down_A_scale[active_slot] = matmul_scales;
    h_down_A_global[active_slot] = tensor_scale;

    h_down_B_ptr[active_slot] = routed_down_host[expert_id].packed_data;
    if (h_down_B_ptr[active_slot] == nullptr) {
      return false;
    }
    h_down_B_scale[active_slot] = routed_down_host[expert_id].matmul_block_scales_data;
    h_down_B_global[active_slot] = routed_down_host[expert_id].tensor_scale_data;

    h_down_D_ptr[active_slot] = params.gather_scratch + (expert_offset * params.hidden_size);
  }

  if (!run_grouped_slots(
          h_down_A_ptr,
          h_down_A_scale,
          h_down_A_global,
          h_down_B_ptr,
          h_down_B_scale,
          h_down_B_global,
          h_down_D_ptr,
          static_cast<int>(params.routed_expert_intermediate_size),
          static_cast<int>(params.hidden_size)) ||
      !run_fallback_slots(
          params.routed_expert_intermediate_size,
          params.hidden_size,
          params.expert_up_scratch,
          h_down_D_ptr,
          routed_down_host,
          params.routed_down_descriptors)) {
    return false;
  }

  // Phase 3: ScatterAdd
  auto routed_output_view = DeviceTensorFp32::CreateView(
      {params.token_count, params.hidden_size},
      params.routed_output);
  if (!routed_output_view) return false;

  for (int active_slot = 0; active_slot < active_expert_count; ++active_slot) {
    const int expert_id_int = active_expert_ids_host[active_slot];
    const std::size_t expert_id = static_cast<std::size_t>(expert_id_int);
    
    const int expert_offset_int = expert_offsets_host[expert_id];
    const int next_expert_offset_int = expert_offsets_host[expert_id + 1];
    const std::size_t expert_offset = static_cast<std::size_t>(expert_offset_int);
    const std::size_t expert_token_count = static_cast<std::size_t>(next_expert_offset_int - expert_offset_int);
    if (expert_token_count == 0) continue;

    auto gather_view = DeviceTensorFp32::CreateView(
        {expert_token_count, params.hidden_size},
        params.gather_scratch + (expert_offset * params.hidden_size));

    if (!ScatterAddWeightedRowsFp32(
             *gather_view,
             params.sorted_token_indices + expert_offset,
             params.sorted_token_weights + expert_offset,
             routed_output_view.get())) {
      return false;
    }
  }
  
  if (cudaStreamSynchronize(0) != cudaSuccess) {
      return false;
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
      // The caller already updated the BF16 residual buffer at the layer entry.
      // This backend must return only the MoE delta, not input + delta.
      !ResidualAddFp32(*routed_output, *output, output.get())) {
    return false;
  }

  return true;
}

}  // namespace nemotron
