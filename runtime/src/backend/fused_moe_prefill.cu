#include "nemotron/fused_moe_prefill.h"

#include <cuda_runtime.h>

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <unordered_map>

#include "nemotron/gemm_execution.h"
#include "nemotron/gemm_planner.h"
#include "nemotron/nvfp4_gemm_runner.h"
#include "nemotron/nvfp4_packing.h"
#include "nemotron/nvfp4_scale_layout.h"
#include "nemotron/primitive_ops.h"

namespace nemotron {
namespace {

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

class PinnedHostIntBuffer {
 public:
  ~PinnedHostIntBuffer() {
    if (data_ != nullptr) {
      cudaFreeHost(data_);
    }
  }

  bool Reserve(std::size_t count) {
    if (count == 0) {
      return false;
    }
    if (count <= count_) {
      return true;
    }
    if (data_ != nullptr) {
      cudaFreeHost(data_);
      data_ = nullptr;
      count_ = 0;
    }
    if (!CheckCuda(cudaMallocHost(reinterpret_cast<void**>(&data_), count * sizeof(int)))) {
      return false;
    }
    count_ = count;
    return true;
  }

  int* data() const { return data_; }

 private:
  int* data_ = nullptr;
  std::size_t count_ = 0;
};

class AsyncExpertOffsetsCopy {
 public:
  ~AsyncExpertOffsetsCopy() {
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
    }
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  bool EnsureReady() {
    if (stream_ == nullptr &&
        !CheckCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking))) {
      return false;
    }
    if (event_ == nullptr &&
        !CheckCuda(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming))) {
      return false;
    }
    return true;
  }

  bool Synchronize() {
    return stream_ == nullptr || CheckCuda(cudaStreamSynchronize(stream_));
  }

  bool BeginCopy(const int* device_data, std::size_t count, int* host_data) {
    if (device_data == nullptr || host_data == nullptr || count == 0) {
      return false;
    }
    if (!EnsureReady() ||
        !Synchronize() ||
        !CheckCuda(cudaEventRecord(event_, nullptr)) ||
        !CheckCuda(cudaStreamWaitEvent(stream_, event_, 0)) ||
        !CheckCuda(cudaMemcpyAsync(
            host_data,
            device_data,
            count * sizeof(int),
            cudaMemcpyDeviceToHost,
            stream_))) {
      return false;
    }
    return true;
  }

 private:
  cudaStream_t stream_ = nullptr;
  cudaEvent_t event_ = nullptr;
};

PinnedHostIntBuffer& GetPinnedExpertOffsetsHostBuffer() {
  thread_local PinnedHostIntBuffer buffer;
  return buffer;
}

AsyncExpertOffsetsCopy& GetAsyncExpertOffsetsCopy() {
  thread_local AsyncExpertOffsetsCopy copy;
  return copy;
}

bool BeginAsyncExpertOffsetsCopy(const int* device_data, std::size_t count) {
  auto& host_buffer = GetPinnedExpertOffsetsHostBuffer();
  auto& async_copy = GetAsyncExpertOffsetsCopy();
  return host_buffer.Reserve(count) &&
         async_copy.BeginCopy(device_data, count, host_buffer.data());
}

bool WaitForAsyncExpertOffsetsCopy() {
  return GetAsyncExpertOffsetsCopy().Synchronize();
}

const int* PinnedExpertOffsetsHostData() {
  return GetPinnedExpertOffsetsHostBuffer().data();
}

std::string BuildRuntimeNvfp4PlanCacheKey(
    const GemmDescriptor& descriptor,
    std::size_t rows) {
  return descriptor.heuristic_key_prefix() + "|m=" + std::to_string(rows);
}

Nvfp4PackOptions RuntimeMoeNvfp4PackOptions() {
  Nvfp4PackOptions options;
  options.execution_scale_layout = Nvfp4ScaleLayout::kSwizzled128x4;
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

const CublasLtGemmPlan* GetOrCreateRuntimeNvfp4GemmPlan(
    std::unordered_map<std::string, CublasLtGemmPlan>* plan_cache,
    const GemmDescriptor& descriptor,
    const Nvfp4PackedMatrixDeviceView& weight_view,
    std::size_t rows,
    GemmHeuristicCache* heuristic_cache) {
  if (plan_cache == nullptr) {
    return nullptr;
  }
  const std::string key = BuildRuntimeNvfp4PlanCacheKey(descriptor, rows);
  auto cached = plan_cache->find(key);
  if (cached != plan_cache->end()) {
    return &cached->second;
  }
  auto plan = BuildRuntimeNvfp4GemmPlan(descriptor, weight_view, rows, heuristic_cache);
  if (!plan.has_value()) {
    return nullptr;
  }
  auto inserted = plan_cache->emplace(std::move(key), std::move(*plan));
  return &inserted.first->second;
}

}  // namespace

bool RunFusedMoePrefill(const FusedMoePrefillParams& params) {
  const std::size_t selection_count = params.token_count * params.top_k;
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
      params.gather_pack == nullptr ||
      params.expert_up_pack == nullptr ||
      params.shared_up_pack == nullptr ||
      !params.gather_pack->valid() ||
      !params.expert_up_pack->valid() ||
      !params.shared_up_pack->valid() ||
      params.gather_pack->rows() < selection_count ||
      params.gather_pack->cols() != params.hidden_size ||
      params.expert_up_pack->rows() < selection_count ||
      params.expert_up_pack->cols() != params.routed_expert_intermediate_size ||
      params.shared_up_pack->rows() < params.token_count ||
      params.shared_up_pack->cols() != params.shared_expert_intermediate_size ||
      params.expert_offsets == nullptr ||
      params.sorted_token_indices == nullptr ||
      params.sorted_token_weights == nullptr) {
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

  if (!BeginAsyncExpertOffsetsCopy(
          params.expert_offsets,
          params.n_routed_experts + 1)) {
    return false;
  }

  const Nvfp4PackOptions pack_options = RuntimeMoeNvfp4PackOptions();
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
      !params.gather_pack->PackInto(*normalized, pack_options) ||
      !RunNvfp4RowMajorFp32AccumToDevice(
           *params.cublas_handle,
           *shared_up_plan,
           MakeNvfp4PackedMatrixDeviceView(*params.gather_pack, params.token_count),
           params.gather_pack->device_tensor_scale_ptr(),
           shared_up_weight_view,
           shared_up_weight_view.tensor_scale_data,
           shared_up.get())
           .has_value() ||
      !Relu2InPlaceFp32(shared_up.get()) ||
      !params.shared_up_pack->PackInto(*shared_up, pack_options) ||
      !RunNvfp4RowMajorFp32AccumToDevice(
           *params.cublas_handle,
           *shared_down_plan,
           MakeNvfp4PackedMatrixDeviceView(*params.shared_up_pack, params.token_count),
           params.shared_up_pack->device_tensor_scale_ptr(),
           shared_down_weight_view,
           shared_down_weight_view.tensor_scale_data,
           output.get())
           .has_value()) {
    WaitForAsyncExpertOffsetsCopy();
    return false;
  }

  if (!WaitForAsyncExpertOffsetsCopy()) {
    return false;
  }
  const int* expert_offsets_host = PinnedExpertOffsetsHostData();
  if (expert_offsets_host == nullptr ||
      expert_offsets_host[0] != 0 ||
      expert_offsets_host[params.n_routed_experts] < 0 ||
      static_cast<std::size_t>(expert_offsets_host[params.n_routed_experts]) > selection_count) {
    return false;
  }

  std::unordered_map<std::string, CublasLtGemmPlan> routed_plan_cache;
  routed_plan_cache.reserve(params.n_routed_experts * 2);
  for (std::size_t expert_id = 0; expert_id < params.n_routed_experts; ++expert_id) {
    if (!ValidFusedNvfp4WeightView(params.routed_up[expert_id]) ||
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
    if (expert_offset_int < 0 || next_expert_offset_int < expert_offset_int) {
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
    const CublasLtGemmPlan* routed_up_plan = GetOrCreateRuntimeNvfp4GemmPlan(
        &routed_plan_cache,
        *params.routed_up_descriptors[expert_id],
        up_weight_view,
        expert_token_count,
        params.heuristic_cache);
    const CublasLtGemmPlan* routed_down_plan = GetOrCreateRuntimeNvfp4GemmPlan(
        &routed_plan_cache,
        *params.routed_down_descriptors[expert_id],
        down_weight_view,
        expert_token_count,
        params.heuristic_cache);
    if (!up_weight_view.valid() ||
        !down_weight_view.valid() ||
        routed_up_plan == nullptr ||
        routed_down_plan == nullptr ||
        !GatherRowsFp32(
             *normalized,
             params.sorted_token_indices + expert_offset,
             gather_view.get()) ||
        !params.gather_pack->PackInto(*gather_view, pack_options) ||
        !RunNvfp4RowMajorFp32AccumToDevice(
             *params.cublas_handle,
             *routed_up_plan,
             MakeNvfp4PackedMatrixDeviceView(*params.gather_pack, expert_token_count),
             params.gather_pack->device_tensor_scale_ptr(),
             up_weight_view,
             up_weight_view.tensor_scale_data,
             expert_up_view.get())
             .has_value() ||
        !Relu2InPlaceFp32(expert_up_view.get()) ||
        !params.expert_up_pack->PackInto(*expert_up_view, pack_options) ||
        !RunNvfp4RowMajorFp32AccumToDevice(
             *params.cublas_handle,
             *routed_down_plan,
             MakeNvfp4PackedMatrixDeviceView(*params.expert_up_pack, expert_token_count),
             params.expert_up_pack->device_tensor_scale_ptr(),
             down_weight_view,
             down_weight_view.tensor_scale_data,
             gather_view.get())
             .has_value() ||
        !ScatterAddWeightedRowsFp32(
             *gather_view,
             params.sorted_token_indices + expert_offset,
             params.sorted_token_weights + expert_offset,
             routed_output.get())) {
      return false;
    }
  }

  if (!ResidualAddFp32(*routed_output, *output, output.get())) {
    return false;
  }

  return true;
}

}  // namespace nemotron
