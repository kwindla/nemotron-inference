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
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nemotron/attention_device_fallback.h"
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

bool DeviceAttentionCompareEnabled() {
  const char* active = std::getenv("NEMOTRON_FORWARD_COMPARE_DEVICE_ATTENTION_ACTIVE");
  return std::getenv("NEMOTRON_FORWARD_COMPARE_DEVICE_ATTENTION") != nullptr &&
         (active == nullptr || std::strcmp(active, "0") != 0);
}

bool DecodeScratchEnabled() {
  const char* value = std::getenv("NEMOTRON_FORWARD_DECODE_SCRATCH");
  return value == nullptr || (value[0] != '\0' && std::string(value) != "0");
}

bool LegacyAttentionPolicyEnvSet(const char* name) {
  return std::getenv(name) != nullptr;
}

void AcknowledgeLegacyAttentionPolicyEnvOverrides(bool debug) {
  static const bool production_env_set =
      LegacyAttentionPolicyEnvSet("NEMOTRON_FORWARD_ATTENTION_PRODUCTION");
  static const bool scalar_fallback_env_set =
      LegacyAttentionPolicyEnvSet("NEMOTRON_FORWARD_ATTENTION_SCALAR_FALLBACK");

  if (!debug || (!production_env_set && !scalar_fallback_env_set)) {
    return;
  }

  static bool warned = false;
  if (!warned) {
    warned = true;
    std::cerr
        << "attention_layer: ignoring legacy production selection env vars "
        << "(NEMOTRON_FORWARD_ATTENTION_PRODUCTION, "
        << "NEMOTRON_FORWARD_ATTENTION_SCALAR_FALLBACK); "
        << "backend selection is now deterministic from the validated config\n";
  }
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

bool IsNanoDecodeShape(const AttentionBackendSelectorConfig& config) {
  return config.batch_size == 1 &&
         config.query_head_count == 32 &&
         config.max_query_tokens == 1 &&
         config.cache_config.dtype == KvCacheDataType::kBf16 &&
         config.cache_config.kv_head_count == 2 &&
         config.cache_config.head_dim == 128 &&
         config.cache_config.tokens_per_page == 16 &&
         config.causal &&
         !config.generate_stats;
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

float MaxAbsDiff(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.size() != rhs.size()) {
    return std::numeric_limits<float>::infinity();
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
  }
  return max_diff;
}

std::vector<float> CopyTensorToHostFp32(const DeviceTensorBf16& tensor) {
  std::vector<__nv_bfloat16> host_bf16(tensor.numel());
  if (!tensor.CopyToHost(host_bf16.data(), host_bf16.size())) {
    return {};
  }

  std::vector<float> host_fp32(host_bf16.size(), 0.0f);
  for (std::size_t i = 0; i < host_bf16.size(); ++i) {
    host_fp32[i] = __bfloat162float(host_bf16[i]);
  }
  return host_fp32;
}

std::optional<std::size_t> QueryHeadToKvHead(
    std::size_t query_head,
    std::size_t query_head_count,
    std::size_t kv_head_count) {
  if (query_head_count == 0 ||
      kv_head_count == 0 ||
      (query_head_count % kv_head_count) != 0) {
    return std::nullopt;
  }
  const std::size_t queries_per_kv_head = query_head_count / kv_head_count;
  if (queries_per_kv_head == 0) {
    return std::nullopt;
  }
  const std::size_t kv_head = query_head / queries_per_kv_head;
  if (kv_head >= kv_head_count) {
    return std::nullopt;
  }
  return kv_head;
}

bool ScatterMatrixIntoPagedCache(
    const std::vector<float>& matrix,
    std::size_t sequence_start,
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
    const std::size_t absolute_token = sequence_start + token;
    const std::size_t page_slot = absolute_token / tokens_per_page;
    const std::size_t page_offset = absolute_token % tokens_per_page;
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

std::size_t Offset4d(
    std::size_t i0,
    std::size_t i1,
    std::size_t i2,
    std::size_t i3,
    std::size_t d1,
    std::size_t d2,
    std::size_t d3) {
  return ((i0 * d1 + i1) * d2 + i2) * d3 + i3;
}

std::optional<std::vector<__nv_bfloat16>> RunPagedAttentionHost(
    const std::vector<__nv_bfloat16>& query,
    const std::vector<__nv_bfloat16>& key_cache,
    const std::vector<__nv_bfloat16>& value_cache,
    const PagedAttentionBatchPlan& batch_plan,
    std::size_t query_head_count,
    const std::vector<std::int32_t>& query_sequence_lengths,
    const std::vector<std::int32_t>& query_sequence_starts,
    std::size_t max_query_tokens,
    float attn_scale,
    bool causal) {
  if (!batch_plan.valid() ||
      batch_plan.batch_size == 0 ||
      query_head_count == 0 ||
      max_query_tokens == 0 ||
      batch_plan.page_table.empty() ||
      query_sequence_lengths.size() != batch_plan.batch_size ||
      query_sequence_starts.size() != batch_plan.batch_size) {
    return std::nullopt;
  }

  const AttentionKvCacheConfig& cache_config = batch_plan.cache_config;
  const std::size_t expected_query_count =
      batch_plan.batch_size * query_head_count * max_query_tokens * cache_config.head_dim;
  if (query.size() != expected_query_count) {
    return std::nullopt;
  }

  std::size_t max_page_id = 0;
  bool found_page = false;
  for (std::int32_t page_id : batch_plan.page_table) {
    if (page_id < 0) {
      continue;
    }
    max_page_id = std::max(max_page_id, static_cast<std::size_t>(page_id));
    found_page = true;
  }
  if (!found_page) {
    return std::nullopt;
  }

  const std::size_t required_cache_count =
      (max_page_id + 1) * cache_config.kv_head_count * cache_config.tokens_per_page *
      cache_config.head_dim;
  if (key_cache.size() < required_cache_count || value_cache.size() < required_cache_count) {
    return std::nullopt;
  }

  std::vector<__nv_bfloat16> output(expected_query_count, __float2bfloat16(0.0f));
  for (std::size_t batch = 0; batch < batch_plan.batch_size; ++batch) {
    const std::size_t q_tokens = static_cast<std::size_t>(query_sequence_lengths[batch]);
    const std::size_t q_start = static_cast<std::size_t>(query_sequence_starts[batch]);
    const std::size_t kv_tokens = static_cast<std::size_t>(batch_plan.sequence_lengths[batch]);
    for (std::size_t head = 0; head < query_head_count; ++head) {
      const auto kv_head = QueryHeadToKvHead(
          head,
          query_head_count,
          cache_config.kv_head_count);
      if (!kv_head.has_value()) {
        return std::nullopt;
      }
      for (std::size_t q_token = 0; q_token < q_tokens; ++q_token) {
        const std::size_t visible_kv_tokens =
            causal ? std::min(kv_tokens, q_start + q_token + 1) : kv_tokens;
        if (visible_kv_tokens == 0) {
          continue;
        }

        std::vector<float> scores(visible_kv_tokens, 0.0f);
        float max_score = -std::numeric_limits<float>::infinity();
        for (std::size_t kv_token = 0; kv_token < visible_kv_tokens; ++kv_token) {
          const std::size_t page_slot = kv_token / cache_config.tokens_per_page;
          const std::size_t page_offset = kv_token % cache_config.tokens_per_page;
          const std::size_t page_index = batch * batch_plan.max_pages_per_sequence + page_slot;
          if (page_index >= batch_plan.page_table.size() || batch_plan.page_table[page_index] < 0) {
            return std::nullopt;
          }
          const std::size_t page_id = static_cast<std::size_t>(batch_plan.page_table[page_index]);
          const std::size_t q_base = Offset4d(
              batch,
              head,
              q_token,
              0,
              query_head_count,
              max_query_tokens,
              cache_config.head_dim);
          const std::size_t k_base = Offset4d(
              page_id,
              *kv_head,
              page_offset,
              0,
              cache_config.kv_head_count,
              cache_config.tokens_per_page,
              cache_config.head_dim);

          float score = 0.0f;
          for (std::size_t dim = 0; dim < cache_config.head_dim; ++dim) {
            score += __bfloat162float(query[q_base + dim]) *
                     __bfloat162float(key_cache[k_base + dim]);
          }
          score *= attn_scale;
          scores[kv_token] = score;
          max_score = std::max(max_score, score);
        }

        float denom = 0.0f;
        for (float& score : scores) {
          score = std::exp(score - max_score);
          denom += score;
        }
        if (!(denom > 0.0f)) {
          return std::nullopt;
        }

        const std::size_t out_base = Offset4d(
            batch,
            head,
            q_token,
            0,
            query_head_count,
            max_query_tokens,
            cache_config.head_dim);
        std::vector<float> accumulated(cache_config.head_dim, 0.0f);
        for (std::size_t kv_token = 0; kv_token < visible_kv_tokens; ++kv_token) {
          const float weight = scores[kv_token] / denom;
          const std::size_t page_slot = kv_token / cache_config.tokens_per_page;
          const std::size_t page_offset = kv_token % cache_config.tokens_per_page;
          const std::size_t page_index = batch * batch_plan.max_pages_per_sequence + page_slot;
          const std::size_t page_id = static_cast<std::size_t>(batch_plan.page_table[page_index]);
          const std::size_t v_base = Offset4d(
              page_id,
              *kv_head,
              page_offset,
              0,
              cache_config.kv_head_count,
              cache_config.tokens_per_page,
              cache_config.head_dim);
          for (std::size_t dim = 0; dim < cache_config.head_dim; ++dim) {
            accumulated[dim] += weight * __bfloat162float(value_cache[v_base + dim]);
          }
        }
        for (std::size_t dim = 0; dim < cache_config.head_dim; ++dim) {
          output[out_base + dim] = __float2bfloat16(accumulated[dim]);
        }
      }
    }
  }

  return output;
}

}  // namespace

const char* AttentionBackendName(AttentionBackend backend) {
  switch (backend) {
    case AttentionBackend::kCudnnPaged:
      return "cudnn_paged";
    case AttentionBackend::kNanoDecode:
      return "nano_decode";
    case AttentionBackend::kDeviceFallback:
      return "device_fallback";
    case AttentionBackend::kUnavailable:
      break;
  }
  return "unavailable";
}

bool AttentionBackendSelectorConfig::valid() const {
  if (!BuildAttentionKvPageGeometry(cache_config).has_value() ||
      batch_size == 0 ||
      query_head_count == 0 ||
      max_query_tokens == 0 ||
      max_kv_tokens == 0 ||
      container_page_count == 0 ||
      page_table_entries == 0) {
    return false;
  }

  const std::size_t required_pages = RequiredPagesForTokens(cache_config, max_kv_tokens);
  return required_pages > 0 &&
         page_table_entries >= required_pages &&
         container_page_count >= required_pages;
}

bool AttentionBackendPolicy::Supports(
    AttentionBackend backend,
    const AttentionBackendSelectorConfig& config,
    std::size_t token_count,
    int device_sm) const {
  if (!config.valid() ||
      token_count == 0 ||
      token_count > config.max_query_tokens ||
      config.cache_config.dtype != KvCacheDataType::kBf16) {
    return false;
  }

  switch (backend) {
    case AttentionBackend::kCudnnPaged:
      return config.cudnn_available && config.cudnn_version >= 90500;
    case AttentionBackend::kNanoDecode:
      return token_count == 1 && device_sm >= 100 && IsNanoDecodeShape(config);
    case AttentionBackend::kDeviceFallback:
      return !config.generate_stats;
    case AttentionBackend::kUnavailable:
      break;
  }
  return false;
}

AttentionBackend AttentionBackendPolicy::Select(
    const AttentionBackendSelectorConfig& config,
    std::size_t token_count) const {
  const int device_sm = GetCurrentDeviceSmVersion();
  constexpr std::array<AttentionBackend, 3> kPriorityOrder = {
      AttentionBackend::kCudnnPaged,
      AttentionBackend::kNanoDecode,
      AttentionBackend::kDeviceFallback,
  };
  for (const AttentionBackend backend : kPriorityOrder) {
    if (Supports(backend, config, token_count, device_sm)) {
      return backend;
    }
  }
  return AttentionBackend::kUnavailable;
}

struct AttentionLayerSlice::Impl {
  AttentionLayerConfig config;
  AttentionBackendPolicy backend_policy;
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
  AcknowledgeLegacyAttentionPolicyEnvOverrides(debug);
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
  const bool use_decode_scratch = token_count == 1 && DecodeScratchEnabled();
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
  if (layer_pages->size() != required_pages ||
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

  AttentionBackendSelectorConfig backend_selector_config;
  backend_selector_config.cache_config = request_context.config().attention_kv_cache;
  backend_selector_config.batch_size = impl_->batch_plan.batch_size;
  backend_selector_config.query_head_count = impl_->config.query_head_count;
  backend_selector_config.max_query_tokens = token_count;
  backend_selector_config.max_kv_tokens = total_sequence_length;
  backend_selector_config.container_page_count = request_context.config().attention_total_pages;
  backend_selector_config.page_table_entries = impl_->batch_plan.max_pages_per_sequence;
  backend_selector_config.cudnn_version = cudnn_handle.version();
  backend_selector_config.cudnn_available = cudnn_handle.valid();
  backend_selector_config.causal = true;
  backend_selector_config.generate_stats = false;
  if (!backend_selector_config.valid()) {
    if (debug) {
      std::cout << "attention_layer: invalid attention backend selector config\n";
    }
    return false;
  }

  const AttentionBackend selected_backend =
      impl_->backend_policy.Select(backend_selector_config, token_count);
  if (selected_backend == AttentionBackend::kUnavailable) {
    if (debug) {
      std::cout << "attention_layer: no supported attention backend for token_count="
                << token_count
                << " max_kv_tokens=" << total_sequence_length << "\n";
    }
    return false;
  }

  if (selected_backend == AttentionBackend::kCudnnPaged) {
    const auto attention_config = BuildCudnnPagedAttentionConfig(
        impl_->batch_plan,
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
    const auto attention_plan =
        CudnnPagedAttentionPlanCache::Global().GetOrCreate(cudnn_handle, *attention_config);
    if (!attention_plan || !attention_plan->valid()) {
      if (debug) {
        std::cout << "attention_layer: cuDNN attention plan lookup failed\n";
      }
      return false;
    }

    const CudnnPagedAttentionExecution execution{
        query_bf16->data(),
        request_context.key_cache()->data(),
        request_context.value_cache()->data(),
        impl_->seq_len_q->data(),
        impl_->seq_len_kv->data(),
        impl_->page_table_k->data(),
        impl_->page_table_v->data(),
        output_bf16->data(),
        nullptr,
    };
    if (!attention_plan->Execute(cudnn_handle, execution)) {
      if (debug) {
        std::cout << "attention_layer: cuDNN attention execute failed\n";
      }
      return false;
    }
    if (debug) {
      std::cout << "attention_layer: using " << AttentionBackendName(selected_backend) << "\n";
    }
  } else {
    const bool attention_ok =
        selected_backend == AttentionBackend::kNanoDecode
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
                  1.0f / std::sqrt(static_cast<float>(impl_->config.head_dim)),
                  true,
                  output_bf16)
            : RunPagedAttentionDeviceFallback(
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
                  1.0f / std::sqrt(static_cast<float>(impl_->config.head_dim)),
                  true,
                  output_bf16);
    if (!attention_ok) {
      if (debug) {
        std::cout << "attention_layer: device paged attention execution failed\n";
      }
      return false;
    }
    if (debug) {
      std::cout << "attention_layer: using " << AttentionBackendName(selected_backend) << "\n";
    }
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

  if (selected_backend != AttentionBackend::kCudnnPaged && DeviceAttentionCompareEnabled()) {
    std::vector<float> q_host = CopyTensorToHostFp32(*q);
    std::vector<__nv_bfloat16> key_cache_host(request_context.key_cache()->numel());
    std::vector<__nv_bfloat16> value_cache_host(request_context.value_cache()->numel());
    std::vector<float> device_attention_output = CopyTensorToHostFp32(*attn_output);
    if (!q_host.empty() &&
        !device_attention_output.empty() &&
        request_context.key_cache()->CopyToHost(key_cache_host.data(), key_cache_host.size()) &&
        request_context.value_cache()->CopyToHost(value_cache_host.data(), value_cache_host.size())) {
      const std::vector<__nv_bfloat16> query_host = MatrixToAttentionQueryBf16(
          q_host,
          token_count,
          impl_->config.query_head_count,
          impl_->config.head_dim);
      const auto attention_output_host = RunPagedAttentionHost(
          query_host,
          key_cache_host,
          value_cache_host,
          impl_->batch_plan,
          impl_->config.query_head_count,
          std::vector<std::int32_t>(
              impl_->query_sequence_lengths_host.begin(),
              impl_->query_sequence_lengths_host.end()),
          std::vector<std::int32_t>(
              impl_->query_sequence_starts_host.begin(),
              impl_->query_sequence_starts_host.end()),
          token_count,
          1.0f / std::sqrt(static_cast<float>(impl_->config.head_dim)),
          true);
      if (attention_output_host.has_value()) {
        const std::vector<float> reference_attention_output = AttentionOutputBf16ToMatrix(
            *attention_output_host,
            token_count,
            impl_->config.query_head_count,
            impl_->config.head_dim);
        const std::size_t row_width = impl_->config.query_head_count * impl_->config.head_dim;
        const std::size_t last_row_offset = (token_count - 1) * row_width;
        std::cout << "device_attention_compare: layer=" << impl_->config.layer_index
                  << " sequence_start=" << sequence_start
                  << " total_sequence_length=" << total_sequence_length
                  << " q_last0=" << q_host[last_row_offset]
                  << " device_last0=" << device_attention_output[last_row_offset]
                  << " host_last0=" << reference_attention_output[last_row_offset]
                  << " max_abs_diff="
                  << MaxAbsDiff(device_attention_output, reference_attention_output)
                  << "\n";
      }
    }
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
  auto delta_bf16 = DeviceTensorBf16::Create(input.shape());
  auto delta_fp32 = DeviceTensorFp32::Create(input.shape());
  const bool ok =
      input_bf16 != nullptr &&
      residual_bf16 != nullptr &&
      delta_bf16 != nullptr &&
      delta_fp32 != nullptr &&
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
          delta_bf16.get()) &&
      CastTensorBf16ToFp32(*delta_bf16, delta_fp32.get()) &&
      ResidualAddFp32(input, *delta_fp32, output);
  return ok && cudaStreamSynchronize(nullptr) == cudaSuccess;
}

}  // namespace nemotron
