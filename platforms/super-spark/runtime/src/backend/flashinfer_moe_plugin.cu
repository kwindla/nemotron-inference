#include "nemotron/flashinfer_moe_plugin_abi.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "flashinfer/exception.h"
#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h"
#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/trtllm/gen/SfLayoutDecl.h"
#include "flashinfer/trtllm/fused_moe/DevKernel.h"
#include "flashinfer/trtllm/fused_moe/RoutingKernel.h"
#include "flashinfer/trtllm/fused_moe/runner.h"

namespace {

namespace btg = batchedGemm::trtllm::gen;
namespace trt_moe = tensorrt_llm::kernels::trtllmgen_moe::MoE;
namespace routing = moe::dev::routing;
namespace trt_routing = tensorrt_llm::kernels::trtllmgen_moe::Routing;
namespace trt_gemm2 = tensorrt_llm::kernels::trtllmgen_moe::Gemm2;

thread_local std::string g_last_error;

void SetLastError(const std::string& message) {
  g_last_error = message;
}

std::string FormatCudaError(const char* what, cudaError_t status) {
  std::ostringstream oss;
  oss << what << ": " << cudaGetErrorString(status);
  return oss.str();
}

void CheckCuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(FormatCudaError(what, status));
  }
}

template <typename T>
class DeviceAllocation {
 public:
  DeviceAllocation() = default;

  explicit DeviceAllocation(std::size_t count) { Reset(count); }

  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;

  DeviceAllocation(DeviceAllocation&& other) noexcept
      : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  DeviceAllocation& operator=(DeviceAllocation&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    Release();
    ptr_ = other.ptr_;
    count_ = other.count_;
    other.ptr_ = nullptr;
    other.count_ = 0;
    return *this;
  }

  ~DeviceAllocation() { Release(); }

  void Reset(std::size_t count) {
    Release();
    if (count == 0) {
      return;
    }
    CheckCuda(cudaMalloc(&ptr_, count * sizeof(T)), "cudaMalloc");
    count_ = count;
  }

  void Release() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
      count_ = 0;
    }
  }

  void CopyFromHost(const T* host, std::size_t count) {
    if (count > count_) {
      throw std::runtime_error("device allocation too small for host copy");
    }
    if (count == 0) {
      return;
    }
    CheckCuda(cudaMemcpy(ptr_, host, count * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
  }

  T* get() const { return ptr_; }
  std::size_t count() const { return count_; }

 private:
  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

__global__ void ConvertFp32ToBf16Kernel(
    const float* input,
    __nv_bfloat16* output,
    std::size_t count) {
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < count) {
    output[index] = __float2bfloat16_rn(input[index]);
  }
}

__global__ void ConvertBf16ToFp32Kernel(
    const __nv_bfloat16* input,
    float* output,
    std::size_t count) {
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < count) {
    output[index] = __bfloat162float(input[index]);
  }
}

__global__ void PackTopkIdsWeightsKernel(
    const std::int32_t* topk_ids,
    const float* topk_weights,
    std::int32_t* packed,
    std::size_t count) {
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < count) {
    const std::uint32_t expert = static_cast<std::uint32_t>(topk_ids[index]) & 0xffffu;
    const std::uint32_t weight =
        static_cast<std::uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(topk_weights[index])));
    packed[index] = static_cast<std::int32_t>((expert << 16) | weight);
  }
}

__global__ void Relu2InPlaceBf16Kernel(
    __nv_bfloat16* data,
    std::size_t row_stride,
    std::size_t max_rows) {
  const std::size_t row = blockIdx.y;
  if (row >= max_rows) {
    return;
  }
  const std::size_t col = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (col >= row_stride) {
    return;
  }
  const std::size_t index = (row * row_stride) + col;
  const float value = __bfloat162float(data[index]);
  const float relu = value > 0.0f ? value : 0.0f;
  data[index] = __float2bfloat16_rn(relu * relu);
}

std::size_t BytesPerExpert(const NemotronFlashInferNvfp4WeightView& view) {
  return view.packed_nbytes;
}

void ValidatePreparedWeightView(
    const NemotronFlashInferNvfp4WeightView& view,
    std::size_t expected_output_rows,
    std::size_t expected_input_cols,
    const char* which) {
  if (view.packed_data == nullptr || view.scale_data == nullptr) {
    throw std::runtime_error(std::string(which) + " view is missing packed or scale data");
  }
  if (view.packed_layout == NEMOTRON_FLASHINFER_WEIGHT_LAYOUT_RAW_ROW_MAJOR &&
      view.scale_layout == NEMOTRON_FLASHINFER_SCALE_LAYOUT_RAW_LINEAR) {
    throw std::runtime_error(
        std::string(which) +
        " uses the CUTLASS raw NVFP4 seam, but the current in-tree FlashInfer plugin still "
        "implements the older TRT split backend");
  }
  if (view.packed_layout != NEMOTRON_FLASHINFER_WEIGHT_LAYOUT_SHUFFLED_MAJOR_K) {
    throw std::runtime_error(std::string(which) + " packed layout is not the legacy shuffled MajorK surface");
  }
  if (view.scale_layout != NEMOTRON_FLASHINFER_SCALE_LAYOUT_SWIZZLED_128X4) {
    throw std::runtime_error(std::string(which) + " scale layout is not the legacy swizzled 128x4 surface");
  }
  if (view.output_rows != expected_output_rows || view.input_cols != expected_input_cols) {
    throw std::runtime_error(std::string(which) + " view shape does not match create params");
  }
}

template <typename ValueType>
std::vector<ValueType> StackHostBytes(
    const NemotronFlashInferNvfp4WeightView* views,
    std::size_t num_experts,
    bool use_scale_bytes) {
  std::vector<ValueType> stacked;
  if (num_experts == 0) {
    return stacked;
  }
  const std::size_t bytes_per_expert =
      use_scale_bytes ? views[0].scale_nbytes : views[0].packed_nbytes;
  stacked.resize((bytes_per_expert * num_experts) / sizeof(ValueType));
  std::byte* dst = reinterpret_cast<std::byte*>(stacked.data());
  for (std::size_t expert = 0; expert < num_experts; ++expert) {
    const auto& view = views[expert];
    const std::size_t bytes = use_scale_bytes ? view.scale_nbytes : view.packed_nbytes;
    if (bytes != bytes_per_expert) {
      throw std::runtime_error("prepared FlashInfer expert tensors must have uniform byte sizes");
    }
    const void* src = use_scale_bytes ? static_cast<const void*>(view.scale_data)
                                      : static_cast<const void*>(view.packed_data);
    std::memcpy(dst + (expert * bytes_per_expert), src, bytes_per_expert);
  }
  return stacked;
}

struct RoutedMoEHandleImpl {
  std::size_t layer_index = 0;
  std::size_t num_experts = 0;
  std::size_t hidden_size = 0;
  std::size_t intermediate_size = 0;
  std::size_t top_k = 0;
  std::size_t n_group = 0;
  std::size_t topk_group = 0;
  float routed_scaling_factor = 0.0f;
  trt_routing::RoutingMethodType routing_method = trt_routing::RoutingMethodType::DeepSeekV3;
  trt_moe::ActivationType activation_type = trt_moe::ActivationType::Relu2;
  int device_ordinal = 0;
  int tile_tokens_dim = 8;
  int64_t gemm1_config_index = -1;
  int64_t gemm2_config_index = -1;
  int total_max_padded_tokens = 0;

  std::unique_ptr<trt_gemm2::Runner> up_gemm_runner;
  std::unique_ptr<trt_gemm2::Runner> down_gemm_runner;

  DeviceAllocation<std::uint8_t> up_weights;
  DeviceAllocation<std::uint8_t> up_scales;
  DeviceAllocation<std::uint8_t> down_weights;
  DeviceAllocation<std::uint8_t> down_scales;
  DeviceAllocation<float> output1_scale_scalar;
  DeviceAllocation<float> output1_scale_gate_scalar;
  DeviceAllocation<float> output2_scale_scalar;

  DeviceAllocation<__nv_bfloat16> hidden_states_bf16;
  DeviceAllocation<std::int32_t> packed_topk;
  DeviceAllocation<std::int32_t> num_tokens_per_expert;
  DeviceAllocation<std::int32_t> total_num_padded_tokens;
  DeviceAllocation<std::int32_t> expanded_idx_to_permuted_idx;
  DeviceAllocation<std::int32_t> permuted_idx_to_token_idx;
  DeviceAllocation<std::int32_t> expert_count_histogram;
  DeviceAllocation<std::int32_t> cta_idx_xy_to_batch_idx;
  DeviceAllocation<std::int32_t> cta_idx_xy_to_mn_limit;
  DeviceAllocation<std::int32_t> num_non_exiting_ctas;
  DeviceAllocation<__nv_bfloat16> expert_weights_bf16;
  DeviceAllocation<__nv_bfloat16> output_bf16;
  DeviceAllocation<__nv_bfloat16> permuted_hidden_states_bf16;
  DeviceAllocation<__nv_bfloat16> gemm1_output_bf16;
  DeviceAllocation<__nv_bfloat16> gemm2_output_bf16;
  DeviceAllocation<std::uint8_t> bmm1_workspace;
  DeviceAllocation<std::uint8_t> bmm2_workspace;
};

void PrepareWeights(RoutedMoEHandleImpl& impl, const NemotronFlashInferRoutedMoECreateParams& params) {
  std::vector<float> output1_scalar(params.num_experts);
  std::vector<float> output1_gate_scalar(params.num_experts);
  std::vector<float> output2_scalar(params.num_experts);
  for (std::size_t expert = 0; expert < params.num_experts; ++expert) {
    ValidatePreparedWeightView(
        params.up_experts[expert],
        params.intermediate_size,
        params.moe_latent_size,
        "up expert");
    ValidatePreparedWeightView(
        params.down_experts[expert],
        params.moe_latent_size,
        params.intermediate_size,
        "down expert");
    output1_scalar[expert] = params.up_experts[expert].tensor_scale;
    output1_gate_scalar[expert] = params.up_experts[expert].tensor_scale;
    output2_scalar[expert] = params.down_experts[expert].tensor_scale;
  }

  auto up_weights_host =
      StackHostBytes<std::uint8_t>(params.up_experts, params.num_experts, false);
  auto up_scales_host =
      StackHostBytes<std::uint8_t>(params.up_experts, params.num_experts, true);
  auto down_weights_host =
      StackHostBytes<std::uint8_t>(params.down_experts, params.num_experts, false);
  auto down_scales_host =
      StackHostBytes<std::uint8_t>(params.down_experts, params.num_experts, true);

  impl.up_weights.Reset(up_weights_host.size());
  impl.up_weights.CopyFromHost(up_weights_host.data(), up_weights_host.size());
  impl.up_scales.Reset(up_scales_host.size());
  impl.up_scales.CopyFromHost(up_scales_host.data(), up_scales_host.size());
  impl.down_weights.Reset(down_weights_host.size());
  impl.down_weights.CopyFromHost(down_weights_host.data(), down_weights_host.size());
  impl.down_scales.Reset(down_scales_host.size());
  impl.down_scales.CopyFromHost(down_scales_host.data(), down_scales_host.size());

  impl.output1_scale_scalar.Reset(output1_scalar.size());
  impl.output1_scale_scalar.CopyFromHost(output1_scalar.data(), output1_scalar.size());
  impl.output1_scale_gate_scalar.Reset(output1_gate_scalar.size());
  impl.output1_scale_gate_scalar.CopyFromHost(
      output1_gate_scalar.data(),
      output1_gate_scalar.size());
  impl.output2_scale_scalar.Reset(output2_scalar.size());
  impl.output2_scale_scalar.CopyFromHost(output2_scalar.data(), output2_scalar.size());
}

void PrepareStaticWorkspace(RoutedMoEHandleImpl& impl) {
  std::string last_runner_error;
  for (const int candidate_tile : {8, 16, 32, 64}) {
    try {
      auto candidate_up_gemm = std::make_unique<trt_gemm2::Runner>(
          btg::Dtype::Bfloat16,
          btg::Dtype::MxE2m1,
          btg::Dtype::Bfloat16,
          false,
          candidate_tile,
          true,
          batchedGemm::gemm::MatrixLayout::MajorK);
      auto candidate_down_gemm = std::make_unique<trt_gemm2::Runner>(
          btg::Dtype::Bfloat16,
          btg::Dtype::MxE2m1,
          btg::Dtype::Bfloat16,
          false,
          candidate_tile,
          true,
          batchedGemm::gemm::MatrixLayout::MajorK);
      const auto candidate_gemm1_config = candidate_up_gemm->getDefaultValidConfigIndex(
          static_cast<int32_t>(impl.top_k),
          static_cast<int32_t>(impl.intermediate_size),
          static_cast<int32_t>(impl.hidden_size),
          static_cast<int32_t>(impl.num_experts),
          1);
      const auto candidate_gemm2_config = candidate_down_gemm->getDefaultValidConfigIndex(
          static_cast<int32_t>(impl.top_k),
          static_cast<int32_t>(impl.hidden_size),
          static_cast<int32_t>(impl.intermediate_size),
          static_cast<int32_t>(impl.num_experts),
          1);
      impl.tile_tokens_dim = candidate_tile;
      impl.gemm1_config_index = candidate_gemm1_config;
      impl.gemm2_config_index = candidate_gemm2_config;
      impl.up_gemm_runner = std::move(candidate_up_gemm);
      impl.down_gemm_runner = std::move(candidate_down_gemm);
      last_runner_error.clear();
      break;
    } catch (const std::exception& error) {
      last_runner_error = error.what();
    }
  }
  if (impl.up_gemm_runner == nullptr || impl.down_gemm_runner == nullptr ||
      impl.gemm1_config_index < 0 || impl.gemm2_config_index < 0) {
    throw std::runtime_error(
        last_runner_error.empty()
            ? "FlashInfer routed MoE failed to initialize any supported split BF16 tile"
            : "FlashInfer routed MoE split-runner init failed: " + last_runner_error);
  }

  const int max_num_padded_tokens = trt_routing::getMaxPermutedPaddedCount(
      1,
      static_cast<int32_t>(impl.top_k),
      static_cast<int32_t>(impl.num_experts),
      impl.tile_tokens_dim);
  const int max_num_ctas = trt_routing::getMaxNumCtasInBatchDim(
      1,
      static_cast<int32_t>(impl.top_k),
      static_cast<int32_t>(impl.num_experts),
      impl.tile_tokens_dim);
  const int expert_histogram_size =
      std::max(static_cast<int>(impl.num_experts) * 2, 256 * 2);

  impl.total_max_padded_tokens = max_num_padded_tokens;

  impl.hidden_states_bf16.Reset(impl.hidden_size);
  impl.packed_topk.Reset(impl.top_k);
  impl.num_tokens_per_expert.Reset(impl.num_experts);
  impl.total_num_padded_tokens.Reset(1);
  impl.expanded_idx_to_permuted_idx.Reset(impl.top_k);
  impl.permuted_idx_to_token_idx.Reset(max_num_padded_tokens);
  impl.expert_count_histogram.Reset(expert_histogram_size);
  impl.cta_idx_xy_to_batch_idx.Reset(max_num_ctas);
  impl.cta_idx_xy_to_mn_limit.Reset(max_num_ctas);
  impl.num_non_exiting_ctas.Reset(1);
  impl.expert_weights_bf16.Reset(impl.top_k);
  impl.output_bf16.Reset(impl.hidden_size);
  impl.permuted_hidden_states_bf16.Reset(max_num_padded_tokens * impl.hidden_size);
  impl.gemm1_output_bf16.Reset(max_num_padded_tokens * impl.intermediate_size);
  impl.gemm2_output_bf16.Reset(max_num_padded_tokens * impl.hidden_size);
  const auto bmm1_bytes = impl.up_gemm_runner->getWorkspaceSizeInBytes(
      static_cast<int32_t>(impl.top_k),
      static_cast<int32_t>(impl.intermediate_size),
      static_cast<int32_t>(impl.hidden_size),
      static_cast<int32_t>(impl.num_experts),
      1,
      static_cast<int32_t>(impl.gemm1_config_index));
  const auto bmm2_bytes = impl.down_gemm_runner->getWorkspaceSizeInBytes(
      static_cast<int32_t>(impl.top_k),
      static_cast<int32_t>(impl.hidden_size),
      static_cast<int32_t>(impl.intermediate_size),
      static_cast<int32_t>(impl.num_experts),
      1,
      static_cast<int32_t>(impl.gemm2_config_index));
  impl.bmm1_workspace.Reset(bmm1_bytes);
  impl.bmm2_workspace.Reset(bmm2_bytes);
}

void LaunchConvertFp32ToBf16(
    const float* input,
    __nv_bfloat16* output,
    std::size_t count,
    cudaStream_t stream) {
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
  ConvertFp32ToBf16Kernel<<<blocks, kThreads, 0, stream>>>(input, output, count);
}

void LaunchConvertBf16ToFp32(
    const __nv_bfloat16* input,
    float* output,
    std::size_t count,
    cudaStream_t stream) {
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
  ConvertBf16ToFp32Kernel<<<blocks, kThreads, 0, stream>>>(input, output, count);
}

void LaunchPackTopk(
    const std::int32_t* topk_ids,
    const float* topk_weights,
    std::int32_t* packed,
    std::size_t count,
    cudaStream_t stream) {
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
  PackTopkIdsWeightsKernel<<<blocks, kThreads, 0, stream>>>(topk_ids, topk_weights, packed, count);
}

}  // namespace

struct NemotronFlashInferRoutedMoEHandle {
  std::unique_ptr<RoutedMoEHandleImpl> impl;
};

extern "C" int nemotron_flashinfer_moe_abi_version() {
  return NEMOTRON_FLASHINFER_MOE_PLUGIN_ABI_VERSION;
}

extern "C" int nemotron_flashinfer_moe_backend_kind() {
  return NEMOTRON_FLASHINFER_BACKEND_KIND_TRT_SPLIT;
}

extern "C" const char* nemotron_flashinfer_moe_last_error() {
  return g_last_error.empty() ? nullptr : g_last_error.c_str();
}

extern "C" NemotronFlashInferRoutedMoEHandle* nemotron_flashinfer_routed_moe_create(
    const NemotronFlashInferRoutedMoECreateParams* params) {
  try {
    SetLastError("");
    if (params == nullptr) {
      throw std::runtime_error("missing FlashInfer create params");
    }
    if (params->num_experts == 0 || params->top_k == 0) {
      throw std::runtime_error("FlashInfer routed MoE requires non-zero expert and top-k counts");
    }
    if (params->up_experts == nullptr || params->down_experts == nullptr) {
      throw std::runtime_error("FlashInfer routed MoE requires prepared up/down expert views");
    }
    if (params->activation_type != 6 || params->routing_method_type != 2) {
      throw std::runtime_error("repo-local FlashInfer plugin currently supports DeepSeekV3 + Relu2 only");
    }

    int device_ordinal = 0;
    CheckCuda(cudaGetDevice(&device_ordinal), "cudaGetDevice");

    auto handle = std::make_unique<NemotronFlashInferRoutedMoEHandle>();
    handle->impl = std::make_unique<RoutedMoEHandleImpl>();
    handle->impl->layer_index = params->layer_index;
    handle->impl->num_experts = params->num_experts;
    // The routed FlashInfer backend only covers the latent-space expert subgraph.
    handle->impl->hidden_size = params->moe_latent_size;
    handle->impl->intermediate_size = params->intermediate_size;
    handle->impl->top_k = params->top_k;
    handle->impl->n_group = params->n_group;
    handle->impl->topk_group = params->topk_group;
    handle->impl->routed_scaling_factor = params->routed_scaling_factor;
    handle->impl->routing_method = trt_routing::RoutingMethodType::DeepSeekV3;
    handle->impl->activation_type = trt_moe::ActivationType::Relu2;
    handle->impl->device_ordinal = device_ordinal;

    PrepareWeights(*handle->impl, *params);
    PrepareStaticWorkspace(*handle->impl);
    return handle.release();
  } catch (const std::exception& error) {
    SetLastError(error.what());
    return nullptr;
  }
}

extern "C" void nemotron_flashinfer_routed_moe_destroy(
    NemotronFlashInferRoutedMoEHandle* handle) {
  delete handle;
}

extern "C" bool nemotron_flashinfer_routed_moe_run(
    NemotronFlashInferRoutedMoEHandle* handle,
    const NemotronFlashInferRoutedMoERunParams* params) {
  try {
    SetLastError("");
    if (handle == nullptr || handle->impl == nullptr || params == nullptr) {
      throw std::runtime_error("invalid FlashInfer routed MoE run handle or params");
    }
    auto& impl = *handle->impl;
    if (params->hidden_states_fp32 == nullptr ||
        params->topk_ids_device == nullptr ||
        params->topk_weights_device == nullptr ||
        params->output_fp32 == nullptr) {
      throw std::runtime_error("FlashInfer routed MoE run received null input or output pointers");
    }
    if (params->token_count != 1) {
      throw std::runtime_error("repo-local FlashInfer plugin currently supports single-token routed decode only");
    }

    cudaStream_t stream = params->cuda_stream != nullptr
                              ? static_cast<cudaStream_t>(params->cuda_stream)
                              : static_cast<cudaStream_t>(nullptr);

    LaunchConvertFp32ToBf16(
        params->hidden_states_fp32,
        impl.hidden_states_bf16.get(),
        impl.hidden_size,
        stream);
    LaunchPackTopk(
        params->topk_ids_device,
        params->topk_weights_device,
        impl.packed_topk.get(),
        impl.top_k,
        stream);
    CheckCuda(cudaGetLastError(), "FlashInfer pre-run kernels");

    routing::routingDeepSeek::Data routing_data;
    routing_data.mDtypeExpW = btg::Dtype::Bfloat16;
    routing_data.mUsePdl = true;
    routing_data.mPtrTopKPacked = impl.packed_topk.get();
    routing_data.mPtrExpertCounts = impl.expert_count_histogram.get();
    routing_data.mPtrPermutedIdxSize = impl.total_num_padded_tokens.get();
    routing_data.mPtrExpandedIdxToPermutedIdx = impl.expanded_idx_to_permuted_idx.get();
    routing_data.mPtrPermutedIdxToExpandedIdx = nullptr;
    routing_data.mPtrPermutedIdxToTokenIdx = impl.permuted_idx_to_token_idx.get();
    routing_data.mPtrTopKWeights = impl.expert_weights_bf16.get();
    routing_data.mPtrTopKIds = nullptr;
    routing_data.mPtrScores = nullptr;
    routing_data.mPtrRoutingBias = nullptr;
    routing_data.mPtrCtaIdxXyToBatchIdx = impl.cta_idx_xy_to_batch_idx.get();
    routing_data.mPtrCtaIdxXyToMnLimit = impl.cta_idx_xy_to_mn_limit.get();
    routing_data.mPtrNumNonExitingCtas = impl.num_non_exiting_ctas.get();
    routing_data.mNumTokens = 1;
    routing_data.mNumExperts = static_cast<int32_t>(impl.num_experts);
    routing_data.mNumExpertGroups = static_cast<int32_t>(impl.n_group);
    routing_data.mNumLimitedGroups = static_cast<int32_t>(impl.topk_group);
    routing_data.mTopK = static_cast<int32_t>(impl.top_k);
    routing_data.mPaddingLog2 = 0;
    int padding = impl.tile_tokens_dim;
    while (padding > 1) {
      routing_data.mPaddingLog2 += 1;
      padding >>= 1;
    }
    routing_data.mTileTokensDim = impl.tile_tokens_dim;
    routing_data.mLocalExpertsStartIdx = 0;
    routing_data.mLocalExpertsStrideLog2 = 0;
    routing_data.mNumLocalExperts = static_cast<int32_t>(impl.num_experts);
    routing_data.mHiddenDim = static_cast<int32_t>(impl.hidden_size);
    routing_data.mRouteScale = impl.routed_scaling_factor;
    routing_data.mUseRoutingSoftmax = false;
    routing::routingDeepSeek::run(routing_data, stream);

    moe::dev::permute::Data permute_data;
    permute_data.mDtypeElt = btg::Dtype::Bfloat16;
    permute_data.mUsePdl = true;
    permute_data.mUseDeepSeekFp8 = false;
    permute_data.inPtr = impl.hidden_states_bf16.get();
    permute_data.outPtr = impl.permuted_hidden_states_bf16.get();
    permute_data.inDqSfsPtr = nullptr;
    permute_data.outDqSfsPtr = nullptr;
    permute_data.expandedIdxToPermutedIdx = impl.expanded_idx_to_permuted_idx.get();
    permute_data.hiddenDim = static_cast<int32_t>(impl.hidden_size);
    permute_data.numTokens = 1;
    permute_data.topK = static_cast<int32_t>(impl.top_k);
    permute_data.totalNumPaddedTokens = impl.total_num_padded_tokens.get();
    moe::dev::permute::run(permute_data, stream);

    impl.up_gemm_runner->run(
        impl.permuted_hidden_states_bf16.get(),
        nullptr,
        impl.up_weights.get(),
        impl.up_scales.get(),
        impl.output1_scale_scalar.get(),
        nullptr,
        impl.gemm1_output_bf16.get(),
        nullptr,
        static_cast<int32_t>(impl.top_k),
        static_cast<int32_t>(impl.intermediate_size),
        static_cast<int32_t>(impl.hidden_size),
        static_cast<int32_t>(impl.num_experts),
        1,
        impl.num_non_exiting_ctas.get(),
        impl.total_num_padded_tokens.get(),
        impl.cta_idx_xy_to_batch_idx.get(),
        impl.cta_idx_xy_to_mn_limit.get(),
        impl.bmm1_workspace.get(),
        impl.device_ordinal,
        stream,
        static_cast<int32_t>(impl.gemm1_config_index),
        true);

    constexpr int kThreads = 256;
    const dim3 relu_grid(
        static_cast<unsigned int>((impl.intermediate_size + kThreads - 1) / kThreads),
        static_cast<unsigned int>(impl.total_max_padded_tokens));
    Relu2InPlaceBf16Kernel<<<relu_grid, kThreads, 0, stream>>>(
        impl.gemm1_output_bf16.get(),
        impl.intermediate_size,
        impl.total_max_padded_tokens);

    impl.down_gemm_runner->run(
        impl.gemm1_output_bf16.get(),
        nullptr,
        impl.down_weights.get(),
        impl.down_scales.get(),
        impl.output2_scale_scalar.get(),
        nullptr,
        impl.gemm2_output_bf16.get(),
        nullptr,
        static_cast<int32_t>(impl.top_k),
        static_cast<int32_t>(impl.hidden_size),
        static_cast<int32_t>(impl.intermediate_size),
        static_cast<int32_t>(impl.num_experts),
        1,
        impl.num_non_exiting_ctas.get(),
        impl.total_num_padded_tokens.get(),
        impl.cta_idx_xy_to_batch_idx.get(),
        impl.cta_idx_xy_to_mn_limit.get(),
        impl.bmm2_workspace.get(),
        impl.device_ordinal,
        stream,
        static_cast<int32_t>(impl.gemm2_config_index),
        true);

    moe::dev::finalize::Data finalize_data;
    finalize_data.mDtypeElt = btg::Dtype::Bfloat16;
    finalize_data.mDtypeExpW = btg::Dtype::Bfloat16;
    finalize_data.mUsePdl = true;
    finalize_data.mUseDeepSeekFp8 = false;
    finalize_data.inPtr = impl.gemm2_output_bf16.get();
    finalize_data.outPtr = impl.output_bf16.get();
    finalize_data.inDqSfsPtr = nullptr;
    finalize_data.outDqSfsPtr = nullptr;
    finalize_data.expertWeightsPtr = impl.expert_weights_bf16.get();
    finalize_data.expandedIdxToPermutedIdx = impl.expanded_idx_to_permuted_idx.get();
    finalize_data.numTokens = 1;
    finalize_data.numExperts = static_cast<int32_t>(impl.num_experts);
    finalize_data.topK = static_cast<int32_t>(impl.top_k);
    finalize_data.hiddenDim = static_cast<int32_t>(impl.hidden_size);
    finalize_data.hiddenDimPadded = static_cast<int32_t>(impl.hidden_size);
    finalize_data.totalNumPaddedTokens = impl.total_num_padded_tokens.get();
    moe::dev::finalize::run(finalize_data, stream);

    LaunchConvertBf16ToFp32(
        impl.output_bf16.get(),
        params->output_fp32,
        impl.hidden_size,
        stream);
    CheckCuda(cudaGetLastError(), "FlashInfer post-run kernels");
    return true;
  } catch (const std::exception& error) {
    SetLastError(error.what());
    return false;
  }
}
