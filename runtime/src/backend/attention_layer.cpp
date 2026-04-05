#include "nemotron/attention_layer.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nemotron/attention_native_kernels.h"
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

  bool CopyFromHost(const T* host_values, std::size_t count) {
    return host_values != nullptr &&
           count <= count_ &&
           cudaMemcpy(data_, host_values, count * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
  }

  bool CopyFromHost(const std::vector<T>& host_values) {
    return CopyFromHost(host_values.data(), host_values.size());
  }

  bool CopyFromHostAsync(const T* host_values, std::size_t count, cudaStream_t stream = nullptr) {
    return host_values != nullptr &&
           count <= count_ &&
           cudaMemcpyAsync(
               data_,
               host_values,
               count * sizeof(T),
               cudaMemcpyHostToDevice,
               stream) == cudaSuccess;
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

thread_local AttentionLayerExecutionCounters g_attention_layer_execution_counters;

void RecordAttentionNativeMultiTokenExecution(std::size_t token_count) {
  if (token_count <= 1) {
    return;
  }
  ++g_attention_layer_execution_counters.native_multi_token_runs;
  g_attention_layer_execution_counters.native_multi_token_tokens += token_count;
}

int GetCurrentDeviceSmVersion() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return 0;
  }

  int major = 0;
  int minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
    return 0;
  }
  return major * 10 + minor;
}

bool IsNanoDecodeShape(
    const AttentionLayerConfig& layer_config,
    const AttentionKvCacheConfig& cache_config,
    std::size_t batch_size,
    std::size_t token_count) {
  return batch_size == 1 &&
         layer_config.query_head_count == 32 &&
         token_count == 1 &&
         cache_config.dtype == KvCacheDataType::kBf16 &&
         cache_config.kv_head_count == 2 &&
         cache_config.head_dim == 128 &&
         cache_config.tokens_per_page == 16;
}

constexpr std::size_t kNanoMultiTokenMaxQueryTokens = 1024;

bool IsNanoMultiTokenShape(
    const AttentionLayerConfig& layer_config,
    const AttentionKvCacheConfig& cache_config,
    std::size_t batch_size,
    std::size_t token_count) {
  return batch_size >= 1 &&
         batch_size <= 4 &&
         layer_config.query_head_count == 32 &&
         token_count >= 2 &&
         token_count <= kNanoMultiTokenMaxQueryTokens &&
         cache_config.dtype == KvCacheDataType::kBf16 &&
         cache_config.kv_head_count == 2 &&
         cache_config.head_dim == 128 &&
         cache_config.tokens_per_page == 16;
}

bool SupportsNativeAttention(
    const AttentionLayerConfig& layer_config,
    const AttentionKvCacheConfig& cache_config,
    std::size_t batch_size,
    std::size_t token_count,
    int device_sm) {
  if (token_count == 1) {
    return device_sm >= 100 &&
           IsNanoDecodeShape(layer_config, cache_config, batch_size, token_count);
  }
  return device_sm >= 120 &&
         IsNanoMultiTokenShape(layer_config, cache_config, batch_size, token_count);
}

const char* NativeAttentionKernelName(std::size_t token_count) {
  return token_count == 1 ? "nano_decode" : "nano_multi_token";
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

void ResetAttentionLayerExecutionCounters() {
  g_attention_layer_execution_counters = AttentionLayerExecutionCounters{};
}

AttentionLayerExecutionCounters GetAttentionLayerExecutionCounters() {
  return g_attention_layer_execution_counters;
}

struct AttentionLayerSlice::Impl {
  AttentionLayerConfig config;
  std::unique_ptr<DeviceTensorFp32> norm_weight;
  std::unique_ptr<UploadedLinearOp> q_proj;
  std::unique_ptr<UploadedLinearOp> k_proj;
  std::unique_ptr<UploadedLinearOp> v_proj;
  std::unique_ptr<UploadedLinearOp> o_proj;
  std::unique_ptr<DeviceTensorBf16> normed_scratch;
  std::unique_ptr<DeviceTensorBf16> q_scratch;
  std::unique_ptr<DeviceTensorBf16> k_scratch;
  std::unique_ptr<DeviceTensorBf16> v_scratch;
  std::unique_ptr<DeviceTensorBf16> attn_output_scratch;
  std::unique_ptr<DeviceTensorBf16> query_bf16_scratch;
  std::unique_ptr<DeviceTensorBf16> attn_output_bf16_scratch;
  std::array<std::int32_t, 1> query_sequence_lengths_host{0};
  std::array<std::int32_t, 1> query_sequence_starts_host{0};
  PagedAttentionBatchPlan batch_plan;
  std::optional<DeviceBuffer<std::int32_t>> seq_len_q;
  std::optional<DeviceBuffer<std::int32_t>> seq_len_kv;
  std::optional<DeviceBuffer<std::int32_t>> query_starts;
  std::optional<DeviceBuffer<std::int32_t>> page_table_k;
  std::optional<DeviceBuffer<std::int32_t>> page_table_v;
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
  const bool debug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  const auto debug_fail = [&](const char* message) -> std::unique_ptr<AttentionLayerSlice> {
    if (debug) {
      std::cerr << "attention_layer: create failed for layer "
                << config.layer_index << ": " << message << "\n";
    }
    return nullptr;
  };
  const std::size_t query_width = config.query_head_count * config.head_dim;
  const std::size_t kv_width = config.kv_head_count * config.head_dim;
  if (config.hidden_size == 0 ||
      config.query_head_count == 0 ||
      config.kv_head_count == 0 ||
      config.head_dim == 0 ||
      config.rms_epsilon <= 0.0f ||
      bindings.norm_weight == nullptr ||
      bindings.q_proj == nullptr ||
      bindings.k_proj == nullptr ||
      bindings.v_proj == nullptr ||
      bindings.o_proj == nullptr) {
    return debug_fail("invalid config or missing bindings");
  }
  if (bindings.norm_weight->logical_shape.size() != 1 ||
      bindings.norm_weight->logical_shape[0] != config.hidden_size ||
      bindings.q_proj->input_cols != config.hidden_size ||
      bindings.q_proj->output_rows != query_width ||
      bindings.k_proj->input_cols != config.hidden_size ||
      bindings.k_proj->output_rows != kv_width ||
      bindings.v_proj->input_cols != config.hidden_size ||
      bindings.v_proj->output_rows != kv_width ||
      bindings.o_proj->input_cols != query_width ||
      bindings.o_proj->output_rows != config.hidden_size) {
    return debug_fail("shape validation failed");
  }

  auto norm_weight = UploadVectorWeightToDeviceFp32(*bindings.norm_weight);
  auto q_proj = UploadedLinearOp::Create(*bindings.q_proj);
  auto k_proj = UploadedLinearOp::Create(*bindings.k_proj);
  auto v_proj = UploadedLinearOp::Create(*bindings.v_proj);
  auto o_proj = UploadedLinearOp::Create(*bindings.o_proj);
  auto normed_scratch = DeviceTensorBf16::Create({1, config.hidden_size});
  auto q_scratch = DeviceTensorBf16::Create({1, query_width});
  auto k_scratch = DeviceTensorBf16::Create({1, kv_width});
  auto v_scratch = DeviceTensorBf16::Create({1, kv_width});
  auto attn_output_scratch = DeviceTensorBf16::Create({1, query_width});
  auto query_bf16_scratch =
      DeviceTensorBf16::Create({1, config.query_head_count, 1, config.head_dim});
  auto attn_output_bf16_scratch =
      DeviceTensorBf16::Create({1, config.query_head_count, 1, config.head_dim});
  if (!norm_weight ||
      !q_proj ||
      !k_proj ||
      !v_proj ||
      !o_proj ||
      !normed_scratch ||
      !q_scratch ||
      !k_scratch ||
      !v_scratch ||
      !attn_output_scratch ||
      !query_bf16_scratch ||
      !attn_output_bf16_scratch) {
    return debug_fail("weight materialization failed");
  }

  auto impl = std::make_unique<Impl>();
  impl->config = config;
  impl->norm_weight = std::move(norm_weight);
  impl->q_proj = std::move(q_proj);
  impl->k_proj = std::move(k_proj);
  impl->v_proj = std::move(v_proj);
  impl->o_proj = std::move(o_proj);
  impl->normed_scratch = std::move(normed_scratch);
  impl->q_scratch = std::move(q_scratch);
  impl->k_scratch = std::move(k_scratch);
  impl->v_scratch = std::move(v_scratch);
  impl->attn_output_scratch = std::move(attn_output_scratch);
  impl->query_bf16_scratch = std::move(query_bf16_scratch);
  impl->attn_output_bf16_scratch = std::move(attn_output_bf16_scratch);
  impl->seq_len_q = DeviceBuffer<std::int32_t>::Create(1);
  impl->seq_len_kv = DeviceBuffer<std::int32_t>::Create(1);
  impl->query_starts = DeviceBuffer<std::int32_t>::Create(1);
  impl->page_table_k = DeviceBuffer<std::int32_t>::Create(1);
  impl->page_table_v = DeviceBuffer<std::int32_t>::Create(1);
  impl->batch_plan.batch_size = 1;
  impl->batch_plan.sequence_lengths.resize(1);
  impl->batch_plan.pages_per_sequence.resize(1);
  if (!impl->seq_len_q.has_value() ||
      !impl->seq_len_kv.has_value() ||
      !impl->query_starts.has_value() ||
      !impl->page_table_k.has_value() ||
      !impl->page_table_v.has_value()) {
    return debug_fail("attention metadata buffer allocation failed");
  }
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
         impl_->o_proj->valid() &&
         impl_->normed_scratch != nullptr &&
         impl_->normed_scratch->valid() &&
         impl_->q_scratch != nullptr &&
         impl_->q_scratch->valid() &&
         impl_->k_scratch != nullptr &&
         impl_->k_scratch->valid() &&
         impl_->v_scratch != nullptr &&
         impl_->v_scratch->valid() &&
         impl_->attn_output_scratch != nullptr &&
         impl_->attn_output_scratch->valid() &&
         impl_->query_bf16_scratch != nullptr &&
         impl_->query_bf16_scratch->valid() &&
         impl_->attn_output_bf16_scratch != nullptr &&
         impl_->attn_output_bf16_scratch->valid() &&
         impl_->seq_len_q.has_value() &&
         impl_->seq_len_kv.has_value() &&
         impl_->query_starts.has_value() &&
         impl_->page_table_k.has_value() &&
         impl_->page_table_v.has_value();
}

const AttentionLayerConfig& AttentionLayerSlice::config() const {
  return impl_->config;
}

bool AttentionLayerSlice::Run(
    CublasLtHandle& cublas_handle,
    const CudnnHandle& cudnn_handle,
    GemmHeuristicCache* heuristic_cache,
    RequestExecutionContext& request_context,
    std::size_t sequence_start,
    std::size_t total_sequence_length,
    const DeviceTensorBf16& input,
    DeviceTensorBf16* residual,
    DeviceTensorBf16* output) const {
  (void)cudnn_handle;
  const bool debug = std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
  if (!valid() ||
      !cublas_handle.valid() ||
      !request_context.valid() ||
      !input.valid() ||
      input.shape().size() != 2 ||
      input.shape()[1] != impl_->config.hidden_size ||
      residual == nullptr ||
      !residual->valid() ||
      residual->shape() != input.shape() ||
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
      sequence_start + token_count != total_sequence_length ||
      total_sequence_length > request_context.config().max_tokens ||
      request_context.config().attention_kv_cache.kv_head_count != impl_->config.kv_head_count ||
      request_context.config().attention_kv_cache.head_dim != impl_->config.head_dim ||
      request_context.config().attention_kv_cache.layer_count <= impl_->config.layer_index) {
    if (debug) {
      std::cout << "attention_layer: request context KV config mismatch\n";
    }
    return false;
  }

  if (token_count > kNanoMultiTokenMaxQueryTokens &&
      impl_->config.query_head_count == 32 &&
      impl_->config.kv_head_count == 2 &&
      impl_->config.head_dim == 128 &&
      request_context.config().attention_kv_cache.tokens_per_page == 16) {
    const std::size_t hidden_size = impl_->config.hidden_size;
    for (std::size_t chunk_start = 0; chunk_start < token_count;
         chunk_start += kNanoMultiTokenMaxQueryTokens) {
      const std::size_t chunk_tokens =
          std::min(kNanoMultiTokenMaxQueryTokens, token_count - chunk_start);
      auto input_chunk = DeviceTensorBf16::CreateView(
          {chunk_tokens, hidden_size},
          input.data() + (chunk_start * hidden_size));
      auto residual_chunk = DeviceTensorBf16::CreateView(
          {chunk_tokens, hidden_size},
          residual->data() + (chunk_start * hidden_size));
      auto output_chunk = DeviceTensorBf16::CreateView(
          {chunk_tokens, hidden_size},
          output->data() + (chunk_start * hidden_size));
      if (input_chunk == nullptr ||
          residual_chunk == nullptr ||
          output_chunk == nullptr ||
          !Run(
              cublas_handle,
              cudnn_handle,
              heuristic_cache,
              request_context,
              sequence_start + chunk_start,
              sequence_start + chunk_start + chunk_tokens,
              *input_chunk,
              residual_chunk.get(),
              output_chunk.get())) {
        if (debug) {
          std::cout << "attention_layer: chunked multi-token execution failed"
                    << " chunk_start=" << chunk_start
                    << " chunk_tokens=" << chunk_tokens << "\n";
        }
        return false;
      }
    }
    return true;
  }

  if (token_count > 1) {
    RecordAttentionNativeMultiTokenExecution(token_count);
  }

  std::unique_ptr<DeviceTensorBf16> normed_owned;
  std::unique_ptr<DeviceTensorBf16> q_owned;
  std::unique_ptr<DeviceTensorBf16> k_owned;
  std::unique_ptr<DeviceTensorBf16> v_owned;
  std::unique_ptr<DeviceTensorBf16> attn_output_owned;
  std::unique_ptr<DeviceTensorBf16> query_bf16_owned;
  std::unique_ptr<DeviceTensorBf16> output_layout_owned;
  DeviceTensorBf16* normed = nullptr;
  DeviceTensorBf16* q = nullptr;
  DeviceTensorBf16* k = nullptr;
  DeviceTensorBf16* v = nullptr;
  DeviceTensorBf16* attn_output = nullptr;
  DeviceTensorBf16* query_bf16 = nullptr;
  DeviceTensorBf16* output_bf16 = nullptr;
  const bool use_decode_scratch = token_count == 1;
  if (use_decode_scratch) {
    normed = impl_->normed_scratch.get();
    q = impl_->q_scratch.get();
    k = impl_->k_scratch.get();
    v = impl_->v_scratch.get();
    attn_output = impl_->attn_output_scratch.get();
    query_bf16 = impl_->query_bf16_scratch.get();
    output_bf16 = impl_->attn_output_bf16_scratch.get();
  } else {
    normed_owned = DeviceTensorBf16::Create({token_count, impl_->config.hidden_size});
    q_owned = DeviceTensorBf16::Create({token_count, impl_->config.query_head_count * impl_->config.head_dim});
    k_owned = DeviceTensorBf16::Create({token_count, impl_->config.kv_head_count * impl_->config.head_dim});
    v_owned = DeviceTensorBf16::Create({token_count, impl_->config.kv_head_count * impl_->config.head_dim});
    attn_output_owned = DeviceTensorBf16::Create(
        {token_count, impl_->config.query_head_count * impl_->config.head_dim});
    query_bf16_owned =
        DeviceTensorBf16::Create({1, impl_->config.query_head_count, token_count, impl_->config.head_dim});
    output_layout_owned =
        DeviceTensorBf16::Create({1, impl_->config.query_head_count, token_count, impl_->config.head_dim});
    normed = normed_owned.get();
    q = q_owned.get();
    k = k_owned.get();
    v = v_owned.get();
    attn_output = attn_output_owned.get();
    query_bf16 = query_bf16_owned.get();
    output_bf16 = output_layout_owned.get();
  }
  if (!normed ||
      !q ||
      !k ||
      !v ||
      !attn_output ||
      !query_bf16 ||
      !output_bf16 ||
      !output_bf16->FillZero()) {
    if (debug) {
      std::cout << "attention_layer: scratch allocation failed\n";
    }
    return false;
  }

  const auto run_linear_bf16 = [&](const UploadedLinearOp& op,
                                   const DeviceTensorBf16& bf16_input,
                                   DeviceTensorBf16* bf16_output) -> bool {
    if (op.Run(cublas_handle, heuristic_cache, bf16_input, bf16_output)) {
      return true;
    }
    auto fp32_input = DeviceTensorFp32::Create(bf16_input.shape());
    auto fp32_output = DeviceTensorFp32::Create(bf16_output->shape());
    return fp32_input != nullptr &&
           fp32_output != nullptr &&
           CastTensorBf16ToFp32(bf16_input, fp32_input.get()) &&
           op.Run(cublas_handle, heuristic_cache, *fp32_input, fp32_output.get()) &&
           CastTensorFp32ToBf16(*fp32_output, bf16_output) &&
           cudaStreamSynchronize(nullptr) == cudaSuccess;
  };

  const bool norm_ok =
      FusedAddRmsNormBf16(input, residual, *impl_->norm_weight, impl_->config.rms_epsilon, normed);
  const bool q_ok = norm_ok && run_linear_bf16(*impl_->q_proj, *normed, q);
  const bool k_ok = q_ok && run_linear_bf16(*impl_->k_proj, *normed, k);
  const bool v_ok = k_ok && run_linear_bf16(*impl_->v_proj, *normed, v);
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

  const auto ensure_attention_metadata_capacity = [&]() -> bool {
    if (request_context.config().max_tokens == 0 ||
        request_context.config().attention_kv_cache.tokens_per_page == 0) {
      return false;
    }
    const std::size_t required_page_capacity = std::max<std::size_t>(
        1,
        RequiredPagesForTokens(
            request_context.config().attention_kv_cache,
            request_context.config().max_tokens));
    if (impl_->page_table_k.has_value() &&
        impl_->page_table_v.has_value() &&
        impl_->page_table_k->count() >= required_page_capacity &&
        impl_->page_table_v->count() >= required_page_capacity) {
      return true;
    }

    auto page_table_k = DeviceBuffer<std::int32_t>::Create(required_page_capacity);
    auto page_table_v = DeviceBuffer<std::int32_t>::Create(required_page_capacity);
    if (!page_table_k.has_value() || !page_table_v.has_value()) {
      return false;
    }

    impl_->page_table_k = std::move(page_table_k);
    impl_->page_table_v = std::move(page_table_v);
    impl_->batch_plan.page_table.reserve(required_page_capacity);
    return true;
  };
  if (!ensure_attention_metadata_capacity()) {
    if (debug) {
      std::cout << "attention_layer: failed to initialize persistent attention metadata\n";
    }
    return false;
  }
  const std::size_t required_pages = RequiredPagesForTokens(
      request_context.config().attention_kv_cache,
      total_sequence_length);
  if (layer_pages->size() < required_pages ||
      required_pages > impl_->page_table_k->count() ||
      required_pages > impl_->page_table_v->count()) {
    if (debug) {
      std::cout << "attention_layer: attention metadata capacity mismatch"
                << " required_pages=" << required_pages
                << " available_pages=" << impl_->page_table_k->count()
                << " layer_pages=" << layer_pages->size() << "\n";
    }
    return false;
  }

  impl_->query_sequence_lengths_host[0] = static_cast<std::int32_t>(token_count);
  impl_->query_sequence_starts_host[0] = static_cast<std::int32_t>(sequence_start);
  impl_->batch_plan.cache_config = request_context.config().attention_kv_cache;
  impl_->batch_plan.layer_index = impl_->config.layer_index;
  impl_->batch_plan.max_sequence_tokens = total_sequence_length;
  impl_->batch_plan.max_pages_per_sequence = required_pages;
  impl_->batch_plan.sequence_lengths[0] = static_cast<std::int32_t>(total_sequence_length);
  impl_->batch_plan.pages_per_sequence[0] = static_cast<std::int32_t>(required_pages);
  impl_->batch_plan.page_table.resize(required_pages);
  for (std::size_t page_index = 0; page_index < required_pages; ++page_index) {
    impl_->batch_plan.page_table[page_index] =
        static_cast<std::int32_t>((*layer_pages)[page_index].page_id);
  }
  if (!impl_->batch_plan.valid()) {
    if (debug) {
      std::cout << "attention_layer: persistent batch plan is invalid\n";
    }
    return false;
  }

  if (!ConvertRowMajorBf16ToAttentionQueryBf16(
          *q,
          token_count,
          impl_->config.query_head_count,
          impl_->config.head_dim,
          query_bf16)) {
    if (debug) {
      std::cout << "attention_layer: failed to convert query to BF16 layout\n";
    }
    return false;
  }

  if (!impl_->seq_len_q->CopyFromHostAsync(
          impl_->query_sequence_lengths_host.data(),
          impl_->query_sequence_lengths_host.size()) ||
      !impl_->seq_len_kv->CopyFromHostAsync(
          impl_->batch_plan.sequence_lengths.data(),
          impl_->batch_plan.sequence_lengths.size()) ||
      !impl_->query_starts->CopyFromHostAsync(
          impl_->query_sequence_starts_host.data(),
          impl_->query_sequence_starts_host.size()) ||
      !impl_->page_table_k->CopyFromHostAsync(
          impl_->batch_plan.page_table.data(),
          impl_->batch_plan.page_table.size()) ||
      !impl_->page_table_v->CopyFromHostAsync(
          impl_->batch_plan.page_table.data(),
          impl_->batch_plan.page_table.size())) {
    if (debug) {
      std::cout << "attention_layer: failed to upload attention auxiliary buffers\n";
    }
    return false;
  }

  if (!ScatterRowMajorBf16ToPagedCacheBf16(
          *k,
          sequence_start,
          token_count,
          impl_->config.kv_head_count,
          impl_->config.head_dim,
          request_context.config().attention_kv_cache.tokens_per_page,
          impl_->page_table_k->data(),
          impl_->batch_plan.max_pages_per_sequence,
          request_context.key_cache()) ||
      !ScatterRowMajorBf16ToPagedCacheBf16(
          *v,
          sequence_start,
          token_count,
          impl_->config.kv_head_count,
          impl_->config.head_dim,
          request_context.config().attention_kv_cache.tokens_per_page,
          impl_->page_table_v->data(),
          impl_->batch_plan.max_pages_per_sequence,
          request_context.value_cache())) {
    if (debug) {
      std::cout << "attention_layer: failed to scatter KV tensors into paged cache\n";
    }
    return false;
  }

  const int device_sm = GetCurrentDeviceSmVersion();
  if (!SupportsNativeAttention(
          impl_->config,
          request_context.config().attention_kv_cache,
          impl_->batch_plan.batch_size,
          token_count,
          device_sm)) {
    if (debug) {
      std::cout << "attention_layer: unsupported native attention shape"
                << " token_count=" << token_count
                << " device_sm=" << device_sm << "\n";
    }
    return false;
  }

  const float attn_scale = 1.0f / std::sqrt(static_cast<float>(impl_->config.head_dim));
  const bool attention_ok =
      token_count == 1
          ? RunPagedAttentionDecodeProduction(
                *query_bf16,
                *request_context.key_cache(),
                *request_context.value_cache(),
                request_context.config().attention_kv_cache,
                impl_->batch_plan.batch_size,
                impl_->batch_plan.max_pages_per_sequence,
                impl_->page_table_k->data(),
                impl_->seq_len_kv->data(),
                impl_->seq_len_q->data(),
                impl_->query_starts->data(),
                impl_->config.query_head_count,
                token_count,
                attn_scale,
                true,
                output_bf16)
          : RunPagedAttentionNanoMultiToken(
                *query_bf16,
                *request_context.key_cache(),
                *request_context.value_cache(),
                request_context.config().attention_kv_cache,
                impl_->batch_plan.batch_size,
                impl_->batch_plan.max_pages_per_sequence,
                impl_->page_table_k->data(),
                impl_->seq_len_kv->data(),
                impl_->seq_len_q->data(),
                impl_->query_starts->data(),
                impl_->config.query_head_count,
                token_count,
                attn_scale,
                true,
                output_bf16);
  if (!attention_ok) {
    if (debug) {
      std::cout << "attention_layer: native paged attention execution failed\n";
    }
    return false;
  }
  if (debug) {
    std::cout << "attention_layer: using " << NativeAttentionKernelName(token_count) << "\n";
  }

  if (!ConvertAttentionOutputBf16ToRowMajorBf16(
          *output_bf16,
          token_count,
          impl_->config.query_head_count,
          impl_->config.head_dim,
          attn_output)) {
    if (debug) {
      std::cout << "attention_layer: failed to convert attention output to row-major BF16\n";
    }
    return false;
  }

  const bool o_ok = run_linear_bf16(*impl_->o_proj, *attn_output, output);
  if (!o_ok) {
    if (debug) {
      std::cout << "attention_layer: output projection failed\n";
    }
    return false;
  }
  return cudaStreamSynchronize(nullptr) == cudaSuccess;
}

bool AttentionLayerSlice::Run(
    CublasLtHandle& cublas_handle,
    const CudnnHandle& cudnn_handle,
    GemmHeuristicCache* heuristic_cache,
    RequestExecutionContext& request_context,
    std::size_t sequence_start,
    std::size_t total_sequence_length,
    const DeviceTensorFp32& input,
    DeviceTensorFp32* output) const {
  auto input_bf16 = DeviceTensorBf16::Create(input.shape());
  auto residual_bf16 = DeviceTensorBf16::Create(input.shape());
  auto output_bf16 = DeviceTensorBf16::Create(input.shape());
  const bool ok =
      input_bf16 != nullptr &&
      residual_bf16 != nullptr &&
      output_bf16 != nullptr &&
      CastTensorFp32ToBf16(input, input_bf16.get()) &&
      residual_bf16->FillZero() &&
      Run(
          cublas_handle,
          cudnn_handle,
          heuristic_cache,
          request_context,
          sequence_start,
          total_sequence_length,
          *input_bf16,
          residual_bf16.get(),
          output_bf16.get()) &&
      CastTensorBf16ToFp32(*output_bf16, output);
  return ok && cudaStreamSynchronize(nullptr) == cudaSuccess;
}

}  // namespace nemotron
