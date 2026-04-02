#include "nemotron/attention_layer.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nemotron/attention_layout.h"
#include "nemotron/cudnn_paged_attention.h"
#include "nemotron/device_buffer.h"
#include "nemotron/paged_attention_plan.h"
#include "nemotron/runtime_stats.h"

namespace nemotron {
namespace {

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

}  // namespace

struct AttentionLayerSlice::Impl {
  AttentionLayerConfig config;
  std::unique_ptr<DeviceTensorFp32> norm_weight;
  std::unique_ptr<UploadedLinearOp> q_proj;
  std::unique_ptr<UploadedLinearOp> k_proj;
  std::unique_ptr<UploadedLinearOp> v_proj;
  std::unique_ptr<UploadedLinearOp> o_proj;
  mutable std::mutex decode_plan_mutex;
  mutable bool decode_plan_attempted = false;
  mutable std::size_t decode_plan_max_kv_tokens = 0;
  mutable std::size_t decode_plan_page_table_entries = 0;
  mutable std::unique_ptr<CudnnPagedAttentionPlan> decode_plan;
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
  AttentionLayerPreparedBindings prepared;
  prepared.norm_weight = std::move(norm_weight);
  prepared.q_proj = std::move(q_proj);
  prepared.k_proj = std::move(k_proj);
  prepared.v_proj = std::move(v_proj);
  prepared.o_proj = std::move(o_proj);
  return CreatePrepared(config, std::move(prepared));
}

std::unique_ptr<AttentionLayerSlice> AttentionLayerSlice::CreatePrepared(
    const AttentionLayerConfig& config,
    AttentionLayerPreparedBindings bindings) {
  if (config.hidden_size == 0 ||
      config.query_head_count == 0 ||
      config.kv_head_count == 0 ||
      config.head_dim == 0 ||
      config.hidden_size != config.query_head_count * config.head_dim ||
      config.rms_epsilon <= 0.0f ||
      !bindings.norm_weight || !bindings.norm_weight->valid() ||
      !bindings.q_proj || !bindings.q_proj->valid() ||
      !bindings.k_proj || !bindings.k_proj->valid() ||
      !bindings.v_proj || !bindings.v_proj->valid() ||
      !bindings.o_proj || !bindings.o_proj->valid()) {
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  impl->config = config;
  impl->norm_weight = std::move(bindings.norm_weight);
  impl->q_proj = std::move(bindings.q_proj);
  impl->k_proj = std::move(bindings.k_proj);
  impl->v_proj = std::move(bindings.v_proj);
  impl->o_proj = std::move(bindings.o_proj);
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
  const bool new_request = request_context.sequence_length() == 0;
  std::size_t cache_start_token = 0;
  std::size_t total_sequence_tokens = token_count;
  if (new_request) {
    if (!request_context.SetSequenceLength(token_count)) {
      if (debug) {
        std::cout << "attention_layer: failed to set sequence length\n";
      }
      return false;
    }
  } else {
    total_sequence_tokens = request_context.sequence_length();
    if (total_sequence_tokens < token_count) {
      if (debug) {
        std::cout << "attention_layer: decode sequence length smaller than token count\n";
      }
      return false;
    }
    cache_start_token = total_sequence_tokens - token_count;
    if (!request_context.EnsureAttentionTokens(total_sequence_tokens)) {
      if (debug) {
        std::cout << "attention_layer: failed to ensure attention tokens for decode append\n";
      }
      return false;
    }
  }

  std::unique_ptr<DeviceTensorFp32> normed_local;
  std::unique_ptr<DeviceTensorFp32> q_local;
  std::unique_ptr<DeviceTensorFp32> k_local;
  std::unique_ptr<DeviceTensorFp32> v_local;
  std::unique_ptr<DeviceTensorFp32> attn_output_fp32_local;
  std::unique_ptr<DeviceTensorFp32> projected_local;
  std::unique_ptr<DeviceTensorBf16> query_bf16_local;
  std::unique_ptr<DeviceTensorBf16> output_bf16_local;

  DeviceTensorFp32* normed = nullptr;
  DeviceTensorFp32* q = nullptr;
  DeviceTensorFp32* k = nullptr;
  DeviceTensorFp32* v = nullptr;
  DeviceTensorFp32* attn_output_fp32 = nullptr;
  DeviceTensorFp32* projected = nullptr;
  DeviceTensorBf16* query_bf16 = nullptr;
  DeviceTensorBf16* output_bf16 = nullptr;

  if (token_count == 1 &&
      request_context.attention_normed_decode() != nullptr &&
      request_context.attention_q_decode() != nullptr &&
      request_context.attention_k_decode() != nullptr &&
      request_context.attention_v_decode() != nullptr &&
      request_context.attention_output_fp32_decode() != nullptr &&
      request_context.attention_projected_decode() != nullptr &&
      request_context.attention_query_bf16_decode() != nullptr &&
      request_context.attention_output_bf16_decode() != nullptr) {
    normed = request_context.attention_normed_decode();
    q = request_context.attention_q_decode();
    k = request_context.attention_k_decode();
    v = request_context.attention_v_decode();
    attn_output_fp32 = request_context.attention_output_fp32_decode();
    projected = request_context.attention_projected_decode();
    query_bf16 = request_context.attention_query_bf16_decode();
    output_bf16 = request_context.attention_output_bf16_decode();
  } else {
    normed_local = DeviceTensorFp32::Create({token_count, impl_->config.hidden_size});
    q_local = DeviceTensorFp32::Create(
        {token_count, impl_->config.query_head_count * impl_->config.head_dim});
    k_local = DeviceTensorFp32::Create(
        {token_count, impl_->config.kv_head_count * impl_->config.head_dim});
    v_local = DeviceTensorFp32::Create(
        {token_count, impl_->config.kv_head_count * impl_->config.head_dim});
    attn_output_fp32_local = DeviceTensorFp32::Create({token_count, impl_->config.hidden_size});
    projected_local = DeviceTensorFp32::Create({token_count, impl_->config.hidden_size});
    query_bf16_local = DeviceTensorBf16::Create(
        {1, impl_->config.query_head_count, token_count, impl_->config.head_dim});
    output_bf16_local = DeviceTensorBf16::Create(
        {1, impl_->config.query_head_count, token_count, impl_->config.head_dim});
    normed = normed_local.get();
    q = q_local.get();
    k = k_local.get();
    v = v_local.get();
    attn_output_fp32 = attn_output_fp32_local.get();
    projected = projected_local.get();
    query_bf16 = query_bf16_local.get();
    output_bf16 = output_bf16_local.get();
  }
  if (normed == nullptr || q == nullptr || k == nullptr || v == nullptr ||
      attn_output_fp32 == nullptr || projected == nullptr ||
      query_bf16 == nullptr || output_bf16 == nullptr) {
    if (debug) {
      std::cout << "attention_layer: scratch allocation failed\n";
    }
    return false;
  }

  const bool norm_ok =
      RmsNormFp32(input, *impl_->norm_weight, impl_->config.rms_epsilon, normed);
  const bool q_ok = norm_ok && impl_->q_proj->Run(cublas_handle, heuristic_cache, *normed, q);
  const bool k_ok = q_ok && impl_->k_proj->Run(cublas_handle, heuristic_cache, *normed, k);
  const bool v_ok = k_ok && impl_->v_proj->Run(cublas_handle, heuristic_cache, *normed, v);
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

  const auto* layer_pages = request_context.kv_pages(impl_->config.layer_index);
  if (layer_pages == nullptr) {
    if (debug) {
      std::cout << "attention_layer: missing layer KV pages\n";
    }
    return false;
  }

  const DeviceBuffer<std::int32_t>* layer_page_ids_device =
      request_context.kv_page_ids_device(impl_->config.layer_index);
  if (layer_page_ids_device == nullptr ||
      layer_page_ids_device->count() < layer_pages->size()) {
    if (debug) {
      std::cout << "attention_layer: missing request-local layer page ids\n";
    }
    return false;
  }
  const std::size_t layer_page_id_count = layer_pages->size();
  const DeviceBuffer<std::int32_t>* decode_page_table_device = nullptr;
  if (token_count == 1) {
    const auto* fixed_decode_page_table =
        request_context.attention_decode_page_table_device(impl_->config.layer_index);
    if (fixed_decode_page_table != nullptr &&
        fixed_decode_page_table->count() >= layer_page_id_count) {
      decode_page_table_device = fixed_decode_page_table;
      layer_page_ids_device = fixed_decode_page_table;
    }
  }
  const bool query_layout_ok = ConvertRowMajorMatrixToAttentionBf16(
      *q,
      token_count,
      impl_->config.query_head_count,
      impl_->config.head_dim,
      query_bf16);
  const bool scatter_kv_ok = query_layout_ok && ScatterKvRowMajorMatricesToPagedCacheBf16(
      *k,
      *v,
      token_count,
      cache_start_token,
      impl_->config.kv_head_count,
      impl_->config.head_dim,
      request_context.config().attention_kv_cache.tokens_per_page,
      layer_page_ids_device->data(),
      layer_page_id_count,
      request_context.key_cache(),
      request_context.value_cache());
  if (!query_layout_ok || !scatter_kv_ok || !output_bf16->FillZero()) {
    if (debug) {
      std::cout << "attention_layer: failed to prepare device attention inputs"
                << " query_layout_ok=" << query_layout_ok
                << " scatter_kv_ok=" << scatter_kv_ok << "\n";
    }
    return false;
  }

  if (token_count == 1) {
    const std::size_t decode_max_kv_tokens = request_context.config().max_tokens;
    const std::size_t decode_page_table_entries =
        RequiredPagesForTokens(request_context.config().attention_kv_cache, decode_max_kv_tokens);
    const CudnnPagedAttentionPlan* decode_plan = nullptr;
    {
      std::lock_guard<std::mutex> lock(impl_->decode_plan_mutex);
      if (!impl_->decode_plan_attempted ||
          impl_->decode_plan_max_kv_tokens != decode_max_kv_tokens ||
          impl_->decode_plan_page_table_entries != decode_page_table_entries) {
        CudnnPagedAttentionConfig decode_config;
        decode_config.cache_config = request_context.config().attention_kv_cache;
        decode_config.batch_size = 1;
        decode_config.query_head_count = impl_->config.query_head_count;
        decode_config.max_query_tokens = 1;
        decode_config.max_kv_tokens = decode_max_kv_tokens;
        decode_config.container_page_count = request_context.config().attention_total_pages;
        decode_config.page_table_entries = decode_page_table_entries;
        decode_config.attn_scale = static_cast<float>(
            1.0 / std::sqrt(static_cast<double>(impl_->config.head_dim)));
        decode_config.causal = true;
        decode_config.generate_stats = false;
        impl_->decode_plan = CudnnPagedAttentionPlan::Create(cudnn_handle, decode_config);
        impl_->decode_plan_attempted = true;
        impl_->decode_plan_max_kv_tokens = decode_max_kv_tokens;
        impl_->decode_plan_page_table_entries = decode_page_table_entries;
        if (impl_->decode_plan && impl_->decode_plan->valid()) {
          RecordAttentionDecodePlanCreate();
        }
      } else if (impl_->decode_plan && impl_->decode_plan->valid()) {
        RecordAttentionDecodePlanHit();
      }
      if (impl_->decode_plan && impl_->decode_plan->valid()) {
        decode_plan = impl_->decode_plan.get();
      }
    }

    if (decode_plan != nullptr) {
      const auto* seq_len_q = request_context.attention_decode_seq_len_q_device();
      const auto* seq_len_kv = request_context.attention_decode_seq_len_kv_device();
      if (decode_page_table_device == nullptr) {
        decode_page_table_device =
            request_context.attention_decode_page_table_device(impl_->config.layer_index);
      }
      const bool decode_aux_ok =
          seq_len_q != nullptr &&
          seq_len_q->count() >= 1 &&
          seq_len_kv != nullptr &&
          seq_len_kv->count() >= 1 &&
          decode_page_table_device != nullptr &&
          decode_page_table_device->count() >= decode_page_table_entries;
      if (!decode_aux_ok) {
        if (debug) {
          std::cout << "attention_layer: decode metadata unavailable\n";
        }
        return false;
      }

      const CudnnPagedAttentionExecution execution{
          query_bf16->data(),
          request_context.key_cache()->data(),
          request_context.value_cache()->data(),
          seq_len_q->data(),
          seq_len_kv->data(),
          decode_page_table_device->data(),
          decode_page_table_device->data(),
          output_bf16->data(),
          nullptr,
      };
      if (!decode_plan->Execute(cudnn_handle, execution)) {
        if (debug) {
          std::cout << "attention_layer: cached decode attention execute failed\n";
        }
        return false;
      }

      if (!ConvertAttentionBf16ToRowMajorMatrix(
              *output_bf16,
              token_count,
              impl_->config.query_head_count,
              impl_->config.head_dim,
              attn_output_fp32)) {
        if (debug) {
          std::cout << "attention_layer: failed to convert cached decode output\n";
        }
        return false;
      }

      const bool o_ok =
          impl_->o_proj->Run(cublas_handle, heuristic_cache, *attn_output_fp32, projected);
      const bool residual_ok = o_ok && ResidualAddFp32(input, *projected, output);
      if (!o_ok || !residual_ok) {
        if (debug) {
          std::cout << "attention_layer: cached decode output projection or residual failed"
                    << " o_ok=" << o_ok
                    << " residual_ok=" << residual_ok << "\n";
        }
        return false;
      }
      return true;
    }
  }

  std::vector<AttentionSequencePages> sequences = {
      AttentionSequencePages{
          total_sequence_tokens,
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

  if (!request_context.EnsureAttentionAuxCapacity(
          batch_plan->sequence_lengths.size(),
          batch_plan->page_table.size())) {
    if (debug) {
      std::cout << "attention_layer: failed to allocate cuDNN auxiliary buffers\n";
    }
    return false;
  }
  auto* seq_len_q = request_context.attention_seq_len_q_device();
  auto* seq_len_kv = request_context.attention_seq_len_kv_device();
  auto* page_table = request_context.attention_page_table_device();
  const std::vector<std::int32_t> seq_len_q_host(
      batch_plan->sequence_lengths.size(),
      static_cast<std::int32_t>(token_count));
  if (seq_len_q == nullptr || seq_len_kv == nullptr || page_table == nullptr ||
      !seq_len_q->CopyFromHost(seq_len_q_host) ||
      !seq_len_kv->CopyFromHost(batch_plan->sequence_lengths) ||
      !page_table->CopyFromHost(batch_plan->page_table)) {
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
      page_table->data(),
      page_table->data(),
      output_bf16->data(),
      nullptr,
  };
  if (!attention_plan->Execute(cudnn_handle, execution)) {
    if (debug) {
      std::cout << "attention_layer: cuDNN attention execute failed\n";
    }
    return false;
  }

  if (!ConvertAttentionBf16ToRowMajorMatrix(
          *output_bf16,
          token_count,
          impl_->config.query_head_count,
          impl_->config.head_dim,
          attn_output_fp32)) {
    if (debug) {
      std::cout << "attention_layer: failed to convert attention output to row-major fp32\n";
    }
    return false;
  }

  const bool o_ok = impl_->o_proj->Run(cublas_handle, heuristic_cache, *attn_output_fp32, projected);
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
