#include "nemotron/attention_layer.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nemotron/cudnn_paged_attention.h"
#include "nemotron/paged_attention_plan.h"

namespace nemotron {
namespace {

template <typename T>
class DeviceBuffer {
 public:
  static std::optional<DeviceBuffer> Create(std::size_t count) {
    if (count == 0) {
      return std::nullopt;
    }
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
      return std::nullopt;
    }
    DeviceBuffer buffer;
    buffer.count_ = count;
    if (cudaMalloc(reinterpret_cast<void**>(&buffer.data_), count * sizeof(T)) != cudaSuccess) {
      return std::nullopt;
    }
    return buffer;
  }

  DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), count_(other.count_) {
    other.data_ = nullptr;
    other.count_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    if (data_ != nullptr) {
      cudaFree(data_);
    }
    data_ = other.data_;
    count_ = other.count_;
    other.data_ = nullptr;
    other.count_ = 0;
    return *this;
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* data() const { return data_; }
  std::size_t count() const { return count_; }

  bool CopyFromHost(const std::vector<T>& host_values) {
    return host_values.size() == count_ &&
           cudaMemcpy(data_, host_values.data(), count_ * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess &&
           cudaDeviceSynchronize() == cudaSuccess;
  }

 private:
  DeviceBuffer() = default;
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

bool ends_with(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

const KernelTensorDescriptor* FindKernelBinding(
    const LayerScheduleEntry& layer,
    const KernelCatalog& kernel_catalog,
    const std::vector<std::string>& suffixes) {
  for (const LayerTensorBinding& binding : layer.bindings) {
    for (const std::string& suffix : suffixes) {
      if (binding.local_name == suffix || ends_with(binding.local_name, suffix)) {
        return kernel_catalog.FindTensor(binding.tensor_name);
      }
    }
  }
  return nullptr;
}

const GemmDescriptor* FindGemmBinding(
    const LayerScheduleEntry& layer,
    const GemmCatalog& gemm_catalog,
    const std::vector<std::string>& suffixes) {
  for (const LayerTensorBinding& binding : layer.bindings) {
    for (const std::string& suffix : suffixes) {
      if (binding.local_name == suffix || ends_with(binding.local_name, suffix)) {
        return gemm_catalog.FindDescriptor(binding.tensor_name);
      }
    }
  }
  return nullptr;
}

std::vector<__nv_bfloat16> MatrixToAttentionQueryBf16(
    const std::vector<float>& matrix,
    std::size_t token_count,
    std::size_t query_head_count,
    std::size_t head_dim) {
  std::vector<__nv_bfloat16> output(token_count * query_head_count * head_dim);
  for (std::size_t token = 0; token < token_count; ++token) {
    for (std::size_t head = 0; head < query_head_count; ++head) {
      for (std::size_t dim = 0; dim < head_dim; ++dim) {
        const std::size_t src = token * (query_head_count * head_dim) + head * head_dim + dim;
        const std::size_t dst = ((head * token_count) + token) * head_dim + dim;
        output[dst] = __float2bfloat16(matrix[src]);
      }
    }
  }
  return output;
}

std::vector<float> AttentionOutputBf16ToMatrix(
    const std::vector<__nv_bfloat16>& tensor,
    std::size_t token_count,
    std::size_t query_head_count,
    std::size_t head_dim) {
  std::vector<float> output(token_count * query_head_count * head_dim, 0.0f);
  for (std::size_t token = 0; token < token_count; ++token) {
    for (std::size_t head = 0; head < query_head_count; ++head) {
      for (std::size_t dim = 0; dim < head_dim; ++dim) {
        const std::size_t dst = token * (query_head_count * head_dim) + head * head_dim + dim;
        const std::size_t src = ((head * token_count) + token) * head_dim + dim;
        output[dst] = __bfloat162float(tensor[src]);
      }
    }
  }
  return output;
}

bool ScatterMatrixIntoPagedCache(
    const std::vector<float>& matrix,
    std::size_t token_count,
    std::size_t kv_head_count,
    std::size_t head_dim,
    std::size_t tokens_per_page,
    const std::vector<KvPageHandle>& pages,
    std::vector<__nv_bfloat16>* cache_values) {
  if (cache_values == nullptr) {
    return false;
  }
  for (std::size_t token = 0; token < token_count; ++token) {
    const std::size_t page_slot = token / tokens_per_page;
    const std::size_t page_offset = token % tokens_per_page;
    if (page_slot >= pages.size()) {
      return false;
    }
    const std::size_t page_id = pages[page_slot].page_id;
    for (std::size_t head = 0; head < kv_head_count; ++head) {
      for (std::size_t dim = 0; dim < head_dim; ++dim) {
        const std::size_t src = token * (kv_head_count * head_dim) + head * head_dim + dim;
        const std::size_t dst =
            (((page_id * kv_head_count) + head) * tokens_per_page + page_offset) * head_dim + dim;
        (*cache_values)[dst] = __float2bfloat16(matrix[src]);
      }
    }
  }
  return true;
}

}  // namespace

struct AttentionLayerSlice::Impl {
  AttentionLayerConfig config;
  std::unique_ptr<DeviceTensorFp32> norm_weight;
  std::unique_ptr<UploadedLinearOp> q_proj;
  std::unique_ptr<UploadedLinearOp> k_proj;
  std::unique_ptr<UploadedLinearOp> v_proj;
  std::unique_ptr<UploadedLinearOp> o_proj;
};

std::optional<AttentionLayerBindings> BuildAttentionLayerBindings(
    const LayerScheduleEntry& layer,
    const KernelCatalog& kernel_catalog,
    const GemmCatalog& gemm_catalog) {
  AttentionLayerBindings bindings;
  bindings.norm_weight = FindKernelBinding(
      layer,
      kernel_catalog,
      {"norm.weight", "input_norm.weight", "attention_norm.weight"});
  bindings.q_proj = FindGemmBinding(layer, gemm_catalog, {"self_attn.q_proj.weight", "attention.q_proj.weight", "q_proj.weight"});
  bindings.k_proj = FindGemmBinding(layer, gemm_catalog, {"self_attn.k_proj.weight", "attention.k_proj.weight", "k_proj.weight"});
  bindings.v_proj = FindGemmBinding(layer, gemm_catalog, {"self_attn.v_proj.weight", "attention.v_proj.weight", "v_proj.weight"});
  bindings.o_proj = FindGemmBinding(layer, gemm_catalog, {"self_attn.o_proj.weight", "attention.o_proj.weight", "o_proj.weight"});
  if (bindings.norm_weight == nullptr ||
      bindings.q_proj == nullptr ||
      bindings.k_proj == nullptr ||
      bindings.v_proj == nullptr ||
      bindings.o_proj == nullptr) {
    return std::nullopt;
  }
  return bindings;
}

std::unique_ptr<AttentionLayerSlice> AttentionLayerSlice::Create(
    const AttentionLayerConfig& config,
    const AttentionLayerBindings& bindings) {
  if (config.hidden_size == 0 ||
      config.query_head_count == 0 ||
      config.kv_head_count == 0 ||
      config.head_dim == 0 ||
      config.hidden_size != config.query_head_count * config.head_dim ||
      config.rms_epsilon <= 0.0f ||
      bindings.norm_weight == nullptr ||
      bindings.q_proj == nullptr ||
      bindings.k_proj == nullptr ||
      bindings.v_proj == nullptr ||
      bindings.o_proj == nullptr) {
    return nullptr;
  }

  auto norm_weight = UploadVectorWeightToDeviceFp32(*bindings.norm_weight);
  auto q_proj = UploadedLinearOp::Create(*bindings.q_proj);
  auto k_proj = UploadedLinearOp::Create(*bindings.k_proj);
  auto v_proj = UploadedLinearOp::Create(*bindings.v_proj);
  auto o_proj = UploadedLinearOp::Create(*bindings.o_proj);
  if (!norm_weight || !q_proj || !k_proj || !v_proj || !o_proj) {
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  impl->config = config;
  impl->norm_weight = std::move(norm_weight);
  impl->q_proj = std::move(q_proj);
  impl->k_proj = std::move(k_proj);
  impl->v_proj = std::move(v_proj);
  impl->o_proj = std::move(o_proj);
  return std::unique_ptr<AttentionLayerSlice>(new AttentionLayerSlice(std::move(impl)));
}

AttentionLayerSlice::AttentionLayerSlice(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
AttentionLayerSlice::AttentionLayerSlice(AttentionLayerSlice&&) noexcept = default;
AttentionLayerSlice& AttentionLayerSlice::operator=(AttentionLayerSlice&&) noexcept = default;
AttentionLayerSlice::~AttentionLayerSlice() = default;

bool AttentionLayerSlice::valid() const {
  return impl_ != nullptr &&
         impl_->norm_weight != nullptr &&
         impl_->norm_weight->valid() &&
         impl_->q_proj != nullptr &&
         impl_->q_proj->valid() &&
         impl_->k_proj != nullptr &&
         impl_->k_proj->valid() &&
         impl_->v_proj != nullptr &&
         impl_->v_proj->valid() &&
         impl_->o_proj != nullptr &&
         impl_->o_proj->valid();
}

const AttentionLayerConfig& AttentionLayerSlice::config() const {
  return impl_->config;
}

bool AttentionLayerSlice::Run(
    CublasLtHandle& cublas_handle,
    const CudnnHandle& cudnn_handle,
    GemmHeuristicCache* heuristic_cache,
    RequestExecutionContext& request_context,
    const DeviceTensorFp32& input,
    DeviceTensorFp32* output) const {
  const bool debug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  if (!valid() ||
      !cublas_handle.valid() ||
      !cudnn_handle.valid() ||
      !request_context.valid() ||
      !input.valid() ||
      input.shape().size() != 2 ||
      input.shape()[1] != impl_->config.hidden_size ||
      output == nullptr ||
      !output->valid() ||
      output->shape() != input.shape()) {
    if (debug) {
      std::cout << "attention_layer: invalid run inputs or state\n";
    }
    return false;
  }

  const std::size_t token_count = input.shape()[0];
  if (token_count == 0 ||
      request_context.config().attention_kv_cache.kv_head_count != impl_->config.kv_head_count ||
      request_context.config().attention_kv_cache.head_dim != impl_->config.head_dim ||
      request_context.config().attention_kv_cache.layer_count <= impl_->config.layer_index) {
    if (debug) {
      std::cout << "attention_layer: request context KV config mismatch\n";
    }
    return false;
  }
  if (!request_context.SetSequenceLength(token_count)) {
    if (debug) {
      std::cout << "attention_layer: failed to set sequence length\n";
    }
    return false;
  }

  auto normed = DeviceTensorFp32::Create({token_count, impl_->config.hidden_size});
  auto q = DeviceTensorFp32::Create({token_count, impl_->config.query_head_count * impl_->config.head_dim});
  auto k = DeviceTensorFp32::Create({token_count, impl_->config.kv_head_count * impl_->config.head_dim});
  auto v = DeviceTensorFp32::Create({token_count, impl_->config.kv_head_count * impl_->config.head_dim});
  auto attn_output_fp32 = DeviceTensorFp32::Create({token_count, impl_->config.hidden_size});
  auto projected = DeviceTensorFp32::Create({token_count, impl_->config.hidden_size});
  auto query_bf16 = DeviceTensorBf16::Create({1, impl_->config.query_head_count, token_count, impl_->config.head_dim});
  auto output_bf16 = DeviceTensorBf16::Create({1, impl_->config.query_head_count, token_count, impl_->config.head_dim});
  if (!normed || !q || !k || !v || !attn_output_fp32 || !projected || !query_bf16 || !output_bf16) {
    if (debug) {
      std::cout << "attention_layer: scratch allocation failed\n";
    }
    return false;
  }

  const bool norm_ok =
      RmsNormFp32(input, *impl_->norm_weight, impl_->config.rms_epsilon, normed.get());
  const bool q_ok = norm_ok && impl_->q_proj->Run(cublas_handle, heuristic_cache, *normed, q.get());
  const bool k_ok = q_ok && impl_->k_proj->Run(cublas_handle, heuristic_cache, *normed, k.get());
  const bool v_ok = k_ok && impl_->v_proj->Run(cublas_handle, heuristic_cache, *normed, v.get());
  if (!norm_ok || !q_ok || !k_ok || !v_ok) {
    if (debug) {
      std::cout << "attention_layer: norm/qkv failed"
                << " norm_ok=" << norm_ok
                << " q_ok=" << q_ok
                << " k_ok=" << k_ok
                << " v_ok=" << v_ok << "\n";
    }
    return false;
  }

  std::vector<float> q_host(q->numel(), 0.0f);
  std::vector<float> k_host(k->numel(), 0.0f);
  std::vector<float> v_host(v->numel(), 0.0f);
  if (!q->CopyToHost(q_host.data(), q_host.size()) ||
      !k->CopyToHost(k_host.data(), k_host.size()) ||
      !v->CopyToHost(v_host.data(), v_host.size())) {
    if (debug) {
      std::cout << "attention_layer: failed to copy qkv to host\n";
    }
    return false;
  }

  const auto* layer_pages = request_context.kv_pages(impl_->config.layer_index);
  if (layer_pages == nullptr) {
    if (debug) {
      std::cout << "attention_layer: missing layer KV pages\n";
    }
    return false;
  }

  std::vector<__nv_bfloat16> key_cache_host(request_context.key_cache()->numel());
  std::vector<__nv_bfloat16> value_cache_host(request_context.value_cache()->numel());
  if (!request_context.key_cache()->CopyToHost(key_cache_host.data(), key_cache_host.size()) ||
      !request_context.value_cache()->CopyToHost(value_cache_host.data(), value_cache_host.size())) {
    if (debug) {
      std::cout << "attention_layer: failed to copy KV cache to host\n";
    }
    return false;
  }
  if (!ScatterMatrixIntoPagedCache(
          k_host,
          token_count,
          impl_->config.kv_head_count,
          impl_->config.head_dim,
          request_context.config().attention_kv_cache.tokens_per_page,
          *layer_pages,
          &key_cache_host) ||
      !ScatterMatrixIntoPagedCache(
          v_host,
          token_count,
          impl_->config.kv_head_count,
          impl_->config.head_dim,
          request_context.config().attention_kv_cache.tokens_per_page,
          *layer_pages,
          &value_cache_host)) {
    if (debug) {
      std::cout << "attention_layer: failed to scatter qkv into paged cache\n";
    }
    return false;
  }

  const std::vector<__nv_bfloat16> query_host = MatrixToAttentionQueryBf16(
      q_host,
      token_count,
      impl_->config.query_head_count,
      impl_->config.head_dim);
  if (!query_bf16->CopyFromHost(query_host.data(), query_host.size()) ||
      !output_bf16->FillZero() ||
      !request_context.key_cache()->CopyFromHost(key_cache_host.data(), key_cache_host.size()) ||
      !request_context.value_cache()->CopyFromHost(value_cache_host.data(), value_cache_host.size())) {
    if (debug) {
      std::cout << "attention_layer: failed to stage query or KV cache to device\n";
    }
    return false;
  }

  std::vector<AttentionSequencePages> sequences = {
      AttentionSequencePages{
          token_count,
          [&]() {
            std::vector<std::size_t> page_ids;
            page_ids.reserve(layer_pages->size());
            for (const KvPageHandle& handle : *layer_pages) {
              page_ids.push_back(handle.page_id);
            }
            return page_ids;
          }(),
      },
  };
  const auto batch_plan = BuildPagedAttentionBatchPlan(
      request_context.config().attention_kv_cache,
      impl_->config.layer_index,
      sequences,
      nullptr);
  if (!batch_plan.has_value()) {
    if (debug) {
      std::cout << "attention_layer: batch plan build failed\n";
    }
    return false;
  }
  const auto attention_config = BuildCudnnPagedAttentionConfig(
      *batch_plan,
      impl_->config.query_head_count,
      token_count,
      request_context.config().attention_total_pages,
      0.0f,
      true,
      false);
  if (!attention_config.has_value()) {
    if (debug) {
      std::cout << "attention_layer: cuDNN attention config build failed\n";
    }
    return false;
  }
  const auto attention_plan = CudnnPagedAttentionPlan::Create(cudnn_handle, *attention_config);
  if (!attention_plan || !attention_plan->valid()) {
    if (debug) {
      std::cout << "attention_layer: cuDNN attention plan creation failed\n";
    }
    return false;
  }

  auto seq_len_q = DeviceBuffer<std::int32_t>::Create(batch_plan->sequence_lengths.size());
  auto seq_len_kv = DeviceBuffer<std::int32_t>::Create(batch_plan->sequence_lengths.size());
  auto page_table_k = DeviceBuffer<std::int32_t>::Create(batch_plan->page_table.size());
  auto page_table_v = DeviceBuffer<std::int32_t>::Create(batch_plan->page_table.size());
  if (!seq_len_q.has_value() || !seq_len_kv.has_value() || !page_table_k.has_value() || !page_table_v.has_value()) {
    if (debug) {
      std::cout << "attention_layer: failed to allocate cuDNN auxiliary buffers\n";
    }
    return false;
  }
  if (!seq_len_q->CopyFromHost(batch_plan->sequence_lengths) ||
      !seq_len_kv->CopyFromHost(batch_plan->sequence_lengths) ||
      !page_table_k->CopyFromHost(batch_plan->page_table) ||
      !page_table_v->CopyFromHost(batch_plan->page_table)) {
    if (debug) {
      std::cout << "attention_layer: failed to upload cuDNN auxiliary buffers\n";
    }
    return false;
  }

  const CudnnPagedAttentionExecution execution{
      query_bf16->data(),
      request_context.key_cache()->data(),
      request_context.value_cache()->data(),
      seq_len_q->data(),
      seq_len_kv->data(),
      page_table_k->data(),
      page_table_v->data(),
      output_bf16->data(),
      nullptr,
  };
  if (!attention_plan->Execute(cudnn_handle, execution)) {
    if (debug) {
      std::cout << "attention_layer: cuDNN attention execute failed\n";
    }
    return false;
  }

  std::vector<__nv_bfloat16> attention_output_host(output_bf16->numel());
  if (!output_bf16->CopyToHost(attention_output_host.data(), attention_output_host.size())) {
    if (debug) {
      std::cout << "attention_layer: failed to copy attention output to host\n";
    }
    return false;
  }
  const std::vector<float> attention_output_matrix = AttentionOutputBf16ToMatrix(
      attention_output_host,
      token_count,
      impl_->config.query_head_count,
      impl_->config.head_dim);
  if (!attn_output_fp32->CopyFromHost(attention_output_matrix.data(), attention_output_matrix.size())) {
    if (debug) {
      std::cout << "attention_layer: failed to upload attention output matrix\n";
    }
    return false;
  }

  const bool o_ok = impl_->o_proj->Run(cublas_handle, heuristic_cache, *attn_output_fp32, projected.get());
  const bool residual_ok = o_ok && ResidualAddFp32(input, *projected, output);
  if (!o_ok || !residual_ok) {
    if (debug) {
      std::cout << "attention_layer: output projection or residual failed"
                << " o_ok=" << o_ok
                << " residual_ok=" << residual_ok << "\n";
    }
    return false;
  }
  return true;
}

}  // namespace nemotron
