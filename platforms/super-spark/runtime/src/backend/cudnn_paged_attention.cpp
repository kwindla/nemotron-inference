#include "nemotron/cudnn_paged_attention.h"

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

namespace fe = cudnn_frontend;

namespace nemotron {

namespace {

constexpr std::int64_t kQueryUid = 1;
constexpr std::int64_t kKeyUid = 2;
constexpr std::int64_t kValueUid = 3;
constexpr std::int64_t kOutputUid = 4;
constexpr std::int64_t kStatsUid = 5;
constexpr std::int64_t kSeqLenQUid = 6;
constexpr std::int64_t kSeqLenKvUid = 7;
constexpr std::int64_t kPageTableKUid = 8;
constexpr std::int64_t kPageTableVUid = 9;
constexpr std::int64_t kAmaxSUid = 10;
constexpr std::int64_t kAmaxOUid = 11;

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

std::optional<fe::DataType_t> ToCudnnFrontendDataType(KvCacheDataType dtype) {
  switch (dtype) {
    case KvCacheDataType::kFp16:
      return fe::DataType_t::HALF;
    case KvCacheDataType::kBf16:
      return fe::DataType_t::BFLOAT16;
    case KvCacheDataType::kFp8E4M3:
      return fe::DataType_t::FP8_E4M3;
  }
  return std::nullopt;
}

float SanitizeFp8Scale(float scale) {
  constexpr float kMinScale = 1.0f / 1024.0f;
  return (!std::isfinite(scale) || scale < kMinScale) ? kMinScale : scale;
}

bool UseFp8Attention(const CudnnPagedAttentionConfig& config) {
  return config.cache_config.dtype == KvCacheDataType::kFp8E4M3;
}

bool DebugAttentionEnabled() {
  return std::getenv("NEMOTRON_FORWARD_DEBUG") != nullptr;
}

void LogGraphStageFailure(
    const char* stage,
    fe::error_t status,
    const CudnnPagedAttentionConfig& config) {
  if (!DebugAttentionEnabled()) {
    return;
  }
  std::cout << "cudnn_paged_attention: " << stage << " failed: " << status.get_message()
            << " dtype=" << static_cast<int>(config.cache_config.dtype)
            << " batch_size=" << config.batch_size
            << " query_heads=" << config.query_head_count
            << " kv_heads=" << config.cache_config.kv_head_count
            << " max_query_tokens=" << config.max_query_tokens
            << " max_kv_tokens=" << config.max_kv_tokens
            << " head_dim=" << config.cache_config.head_dim
            << " tokens_per_page=" << config.cache_config.tokens_per_page
            << " page_table_entries=" << config.page_table_entries
            << "\n";
}

bool FitsInt64(std::size_t value) {
  return value <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
}

std::vector<std::int64_t> MakeDims(std::initializer_list<std::size_t> dims) {
  std::vector<std::int64_t> result;
  result.reserve(dims.size());
  for (const std::size_t dim : dims) {
    result.push_back(static_cast<std::int64_t>(dim));
  }
  return result;
}

}  // namespace

bool CudnnPagedAttentionConfig::valid() const {
  return BuildAttentionKvPageGeometry(cache_config).has_value() &&
         batch_size > 0 &&
         query_head_count > 0 &&
         max_query_tokens > 0 &&
         max_kv_tokens > 0 &&
         container_page_count > 0 &&
         page_table_entries > 0 &&
         FitsInt64(batch_size) &&
         FitsInt64(query_head_count) &&
         FitsInt64(cache_config.kv_head_count) &&
         FitsInt64(max_query_tokens) &&
         FitsInt64(max_kv_tokens) &&
         FitsInt64(cache_config.head_dim) &&
         FitsInt64(cache_config.tokens_per_page) &&
         FitsInt64(container_page_count) &&
         FitsInt64(page_table_entries) &&
         (!UseFp8Attention(*this) ||
          (std::isfinite(cache_config.q_scale) && cache_config.q_scale > 0.0f &&
           std::isfinite(cache_config.k_scale) && cache_config.k_scale > 0.0f &&
           std::isfinite(cache_config.v_scale) && cache_config.v_scale > 0.0f &&
           std::isfinite(cache_config.prob_scale) && cache_config.prob_scale > 0.0f));
}

std::optional<CudnnPagedAttentionConfig> BuildCudnnPagedAttentionConfig(
    const PagedAttentionBatchPlan& kv_plan,
    std::size_t query_head_count,
    std::size_t max_query_tokens,
    std::size_t container_page_count,
    float attn_scale,
    bool causal,
    bool generate_stats) {
  if (!kv_plan.valid() ||
      query_head_count == 0 ||
      max_query_tokens == 0 ||
      container_page_count == 0) {
    return std::nullopt;
  }

  std::size_t max_page_id = 0;
  bool has_page = false;
  for (const std::int32_t page_id : kv_plan.page_table) {
    if (page_id < 0) {
      continue;
    }
    has_page = true;
    max_page_id = std::max(max_page_id, static_cast<std::size_t>(page_id));
  }
  if (has_page && max_page_id >= container_page_count) {
    return std::nullopt;
  }

  CudnnPagedAttentionConfig config;
  config.cache_config = kv_plan.cache_config;
  config.batch_size = kv_plan.batch_size;
  config.query_head_count = query_head_count;
  config.max_query_tokens = max_query_tokens;
  config.max_kv_tokens = kv_plan.max_sequence_tokens;
  config.container_page_count = container_page_count;
  config.page_table_entries = kv_plan.max_pages_per_sequence;
  config.attn_scale = attn_scale > 0.0f
      ? attn_scale
      : static_cast<float>(1.0 / std::sqrt(static_cast<double>(config.cache_config.head_dim)));
  config.causal = causal;
  config.generate_stats = generate_stats;
  if (!config.valid()) {
    return std::nullopt;
  }
  return config;
}

struct CudnnPagedAttentionPlan::Impl {
  CudnnPagedAttentionConfig config;
  std::shared_ptr<fe::graph::Graph> graph;
  void* workspace = nullptr;
  std::size_t workspace_bytes = 0;
  void* amax_s = nullptr;
  void* amax_o = nullptr;
  bool valid = false;
};

std::unique_ptr<CudnnPagedAttentionPlan> CudnnPagedAttentionPlan::Create(
    const CudnnHandle& handle,
    const CudnnPagedAttentionConfig& config) {
  if (!handle.valid() || !config.valid() || handle.version() < 90500) {
    return nullptr;
  }

  const auto io_data_type = ToCudnnFrontendDataType(config.cache_config.dtype);
  if (!io_data_type.has_value()) {
    return nullptr;
  }
  const fe::DataType_t query_data_type =
      UseFp8Attention(config) ? fe::DataType_t::FP8_E4M3 : fe::DataType_t::BFLOAT16;

  auto impl = std::make_unique<Impl>();
  impl->config = config;
  impl->graph = std::make_shared<fe::graph::Graph>();
  impl->graph->set_io_data_type(query_data_type)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);

  auto q = impl->graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("Q")
                                   .set_uid(kQueryUid)
                                   .set_dim(MakeDims({
                                       config.batch_size,
                                       config.query_head_count,
                                       config.max_query_tokens,
                                       config.cache_config.head_dim,
                                   }))
                                   .set_stride(MakeDims({
                                       config.query_head_count * config.max_query_tokens * config.cache_config.head_dim,
                                       config.max_query_tokens * config.cache_config.head_dim,
                                       config.cache_config.head_dim,
                                       1,
                                   }))
                                   .set_data_type(query_data_type));

  auto k = impl->graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("K_container")
                                   .set_uid(kKeyUid)
                                   .set_dim(MakeDims({
                                       config.container_page_count,
                                       config.cache_config.kv_head_count,
                                       config.cache_config.tokens_per_page,
                                       config.cache_config.head_dim,
                                   }))
                                   .set_stride(MakeDims({
                                       config.cache_config.kv_head_count * config.cache_config.tokens_per_page *
                                           config.cache_config.head_dim,
                                       config.cache_config.tokens_per_page * config.cache_config.head_dim,
                                       config.cache_config.head_dim,
                                       1,
                                   }))
                                   .set_data_type(*io_data_type));

  auto v = impl->graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("V_container")
                                   .set_uid(kValueUid)
                                   .set_dim(MakeDims({
                                       config.container_page_count,
                                       config.cache_config.kv_head_count,
                                       config.cache_config.tokens_per_page,
                                       config.cache_config.head_dim,
                                   }))
                                   .set_stride(MakeDims({
                                       config.cache_config.kv_head_count * config.cache_config.tokens_per_page *
                                           config.cache_config.head_dim,
                                       config.cache_config.tokens_per_page * config.cache_config.head_dim,
                                       config.cache_config.head_dim,
                                       1,
                                   }))
                                   .set_data_type(*io_data_type));

  auto seq_q = impl->graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("seq_q")
                                       .set_uid(kSeqLenQUid)
                                       .set_dim({static_cast<std::int64_t>(config.batch_size), 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(fe::DataType_t::INT32));

  auto seq_kv = impl->graph->tensor(fe::graph::Tensor_attributes()
                                        .set_name("seq_kv")
                                        .set_uid(kSeqLenKvUid)
                                        .set_dim({static_cast<std::int64_t>(config.batch_size), 1, 1, 1})
                                        .set_stride({1, 1, 1, 1})
                                        .set_data_type(fe::DataType_t::INT32));

  auto page_table_k = impl->graph->tensor(fe::graph::Tensor_attributes()
                                              .set_name("page_table_k")
                                              .set_uid(kPageTableKUid)
                                              .set_dim({static_cast<std::int64_t>(config.batch_size),
                                                        1,
                                                        static_cast<std::int64_t>(config.page_table_entries),
                                                        1})
                                              .set_stride({static_cast<std::int64_t>(config.page_table_entries),
                                                           static_cast<std::int64_t>(config.page_table_entries),
                                                           1,
                                                           1})
                                              .set_data_type(fe::DataType_t::INT32));

  auto page_table_v = impl->graph->tensor(fe::graph::Tensor_attributes()
                                              .set_name("page_table_v")
                                              .set_uid(kPageTableVUid)
                                              .set_dim({static_cast<std::int64_t>(config.batch_size),
                                                        1,
                                                        static_cast<std::int64_t>(config.page_table_entries),
                                                        1})
                                              .set_stride({static_cast<std::int64_t>(config.page_table_entries),
                                                           static_cast<std::int64_t>(config.page_table_entries),
                                                           1,
                                                           1})
                                              .set_data_type(fe::DataType_t::INT32));

  auto sdpa_options = fe::graph::SDPA_attributes()
                          .set_name("nemotron_paged_attention")
                          .set_padding_mask(true)
                          .set_seq_len_q(seq_q)
                          .set_seq_len_kv(seq_kv)
                          .set_paged_attention_k_table(page_table_k)
                          .set_paged_attention_v_table(page_table_v)
                          .set_paged_attention_max_seq_len_kv(static_cast<int>(config.max_kv_tokens))
                          .set_generate_stats(config.generate_stats)
                          .set_attn_scale(config.attn_scale);

  if (config.causal) {
    // Single-token decode needs the causal band aligned to the newest KV position,
    // not the start of the cache. With q_len < kv_len, top-left alignment masks
    // almost the entire prefix and produces the wrong decode attention output.
    const fe::DiagonalAlignment_t diagonal_alignment =
        config.max_query_tokens < config.max_kv_tokens
            ? fe::DiagonalAlignment_t::BOTTOM_RIGHT
            : fe::DiagonalAlignment_t::TOP_LEFT;
    sdpa_options.set_diagonal_alignment(diagonal_alignment)
        .set_diagonal_band_right_bound(0);
  }

  std::shared_ptr<fe::graph::Tensor_attributes> o;
  std::shared_ptr<fe::graph::Tensor_attributes> stats;
  std::shared_ptr<fe::graph::Tensor_attributes> amax_s;
  std::shared_ptr<fe::graph::Tensor_attributes> amax_o;
  if (UseFp8Attention(config)) {
    const float q_scale = SanitizeFp8Scale(config.cache_config.q_scale);
    const float k_scale = SanitizeFp8Scale(config.cache_config.k_scale);
    const float v_scale = SanitizeFp8Scale(config.cache_config.v_scale);
    const float prob_scale = SanitizeFp8Scale(config.cache_config.prob_scale);
    auto descale_q = impl->graph->tensor(q_scale);
    auto descale_k = impl->graph->tensor(k_scale);
    auto descale_v = impl->graph->tensor(v_scale);
    auto descale_s = impl->graph->tensor(prob_scale);
    auto scale_s = impl->graph->tensor(1.0f / prob_scale);
    auto scale_o = impl->graph->tensor(1.0f);
    auto outputs = impl->graph->sdpa_fp8(
        q,
        k,
        v,
        descale_q,
        descale_k,
        descale_v,
        descale_s,
        scale_s,
        scale_o,
        std::move(sdpa_options));
    o = outputs[0];
    stats = outputs[1];
    amax_s = outputs[2];
    amax_o = outputs[3];
  } else {
    auto outputs = impl->graph->sdpa(q, k, v, std::move(sdpa_options));
    o = outputs[0];
    stats = outputs[1];
  }
  o->set_output(true)
      .set_uid(kOutputUid)
      .set_dim(MakeDims({
          config.batch_size,
          config.query_head_count,
          config.max_query_tokens,
          config.cache_config.head_dim,
      }))
      .set_stride(MakeDims({
          config.query_head_count * config.max_query_tokens * config.cache_config.head_dim,
          config.max_query_tokens * config.cache_config.head_dim,
          config.cache_config.head_dim,
          1,
      }))
      .set_data_type(fe::DataType_t::BFLOAT16);

  if (UseFp8Attention(config)) {
    if (amax_s == nullptr || amax_o == nullptr) {
      return nullptr;
    }
    amax_s->set_output(true)
        .set_uid(kAmaxSUid)
        .set_dim({1, 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_data_type(fe::DataType_t::FLOAT);
    amax_o->set_output(true)
        .set_uid(kAmaxOUid)
        .set_dim({1, 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_data_type(fe::DataType_t::FLOAT);
  }

  if (config.generate_stats) {
    if (stats == nullptr) {
      return nullptr;
    }
    stats->set_output(true).set_uid(kStatsUid).set_data_type(fe::DataType_t::FLOAT);
  }

  auto validate_status = impl->graph->validate();
  if (!validate_status.is_good()) {
    LogGraphStageFailure("validate", validate_status, config);
    return nullptr;
  }
  auto op_graph_status = impl->graph->build_operation_graph(
      reinterpret_cast<cudnnHandle_t>(handle.handle()));
  if (!op_graph_status.is_good()) {
    LogGraphStageFailure("build_operation_graph", op_graph_status, config);
    return nullptr;
  }
  const auto heur_modes = UseFp8Attention(config)
      ? std::vector<fe::HeurMode_t>{fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}
      : std::vector<fe::HeurMode_t>{fe::HeurMode_t::A};
  auto create_plans_status = impl->graph->create_execution_plans(heur_modes);
  if (!create_plans_status.is_good()) {
    LogGraphStageFailure("create_execution_plans", create_plans_status, config);
    return nullptr;
  }
  auto check_support_status = impl->graph->check_support(
      reinterpret_cast<cudnnHandle_t>(handle.handle()));
  if (!check_support_status.is_good()) {
    LogGraphStageFailure("check_support", check_support_status, config);
    return nullptr;
  }
  auto build_plans_status = impl->graph->build_plans(
      reinterpret_cast<cudnnHandle_t>(handle.handle()));
  if (!build_plans_status.is_good()) {
    LogGraphStageFailure("build_plans", build_plans_status, config);
    return nullptr;
  }

  std::int64_t workspace_bytes = 0;
  if (!impl->graph->get_workspace_size(workspace_bytes).is_good() || workspace_bytes < 0) {
    return nullptr;
  }

  impl->workspace_bytes = static_cast<std::size_t>(workspace_bytes);
  if (impl->workspace_bytes > 0) {
    if (!CheckCuda(cudaMalloc(&impl->workspace, impl->workspace_bytes))) {
      return nullptr;
    }
  }
  if (UseFp8Attention(config)) {
    if (!CheckCuda(cudaMalloc(&impl->amax_s, sizeof(float))) ||
        !CheckCuda(cudaMalloc(&impl->amax_o, sizeof(float)))) {
      if (impl->amax_s != nullptr) {
        cudaFree(impl->amax_s);
        impl->amax_s = nullptr;
      }
      if (impl->amax_o != nullptr) {
        cudaFree(impl->amax_o);
        impl->amax_o = nullptr;
      }
      if (impl->workspace != nullptr) {
        cudaFree(impl->workspace);
        impl->workspace = nullptr;
      }
      return nullptr;
    }
  }

  impl->valid = true;
  return std::unique_ptr<CudnnPagedAttentionPlan>(new CudnnPagedAttentionPlan(std::move(impl)));
}

CudnnPagedAttentionPlan::CudnnPagedAttentionPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

CudnnPagedAttentionPlan::CudnnPagedAttentionPlan(CudnnPagedAttentionPlan&&) noexcept = default;
CudnnPagedAttentionPlan& CudnnPagedAttentionPlan::operator=(CudnnPagedAttentionPlan&&) noexcept = default;

CudnnPagedAttentionPlan::~CudnnPagedAttentionPlan() {
  if (impl_ && impl_->workspace != nullptr) {
    cudaFree(impl_->workspace);
  }
  if (impl_ && impl_->amax_s != nullptr) {
    cudaFree(impl_->amax_s);
  }
  if (impl_ && impl_->amax_o != nullptr) {
    cudaFree(impl_->amax_o);
  }
}

bool CudnnPagedAttentionPlan::valid() const {
  return impl_ != nullptr && impl_->valid && impl_->graph != nullptr;
}

const CudnnPagedAttentionConfig& CudnnPagedAttentionPlan::config() const {
  return impl_->config;
}

std::size_t CudnnPagedAttentionPlan::workspace_bytes() const {
  return impl_ ? impl_->workspace_bytes : 0;
}

bool CudnnPagedAttentionPlan::Execute(
    const CudnnHandle& handle,
    const CudnnPagedAttentionExecution& execution,
    cudaStream_t stream,
    bool synchronize) const {
  if (!valid() ||
      !handle.valid() ||
      execution.query == nullptr ||
      execution.key_cache == nullptr ||
      execution.value_cache == nullptr ||
      execution.seq_len_q == nullptr ||
      execution.seq_len_kv == nullptr ||
      execution.page_table_k == nullptr ||
      execution.page_table_v == nullptr ||
      execution.output == nullptr ||
      (impl_->config.generate_stats && execution.stats == nullptr)) {
    return false;
  }

  std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
      {kQueryUid, const_cast<void*>(execution.query)},
      {kKeyUid, const_cast<void*>(execution.key_cache)},
      {kValueUid, const_cast<void*>(execution.value_cache)},
      {kSeqLenQUid, const_cast<std::int32_t*>(execution.seq_len_q)},
      {kSeqLenKvUid, const_cast<std::int32_t*>(execution.seq_len_kv)},
      {kPageTableKUid, const_cast<std::int32_t*>(execution.page_table_k)},
      {kPageTableVUid, const_cast<std::int32_t*>(execution.page_table_v)},
      {kOutputUid, execution.output},
  };
  if (impl_->config.generate_stats) {
    variant_pack[kStatsUid] = execution.stats;
  }
  if (UseFp8Attention(impl_->config)) {
    variant_pack[kAmaxSUid] = impl_->amax_s;
    variant_pack[kAmaxOUid] = impl_->amax_o;
  }

  if (cudnnSetStream(reinterpret_cast<cudnnHandle_t>(handle.handle()), stream) != CUDNN_STATUS_SUCCESS) {
    return false;
  }

  const bool ok = impl_->graph->execute(
             reinterpret_cast<cudnnHandle_t>(handle.handle()),
             variant_pack,
             impl_->workspace).is_good();
  if (!ok) {
    return false;
  }
  return !synchronize || CheckCuda(cudaStreamSynchronize(stream));
}

}  // namespace nemotron
