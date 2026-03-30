#include "nemotron/cudnn_paged_attention.h"

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

#include <cmath>
#include <cstdint>
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

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

std::optional<fe::DataType_t> ToCudnnFrontendDataType(KvCacheDataType dtype) {
  switch (dtype) {
    case KvCacheDataType::kFp16:
      return fe::DataType_t::HALF;
    case KvCacheDataType::kBf16:
      return fe::DataType_t::BFLOAT16;
  }
  return std::nullopt;
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
         FitsInt64(page_table_entries);
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

  auto impl = std::make_unique<Impl>();
  impl->config = config;
  impl->graph = std::make_shared<fe::graph::Graph>();
  impl->graph->set_io_data_type(*io_data_type)
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
                                   })));

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
                                   })));

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
                                   })));

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
    sdpa_options.set_diagonal_alignment(fe::DiagonalAlignment_t::TOP_LEFT)
        .set_diagonal_band_right_bound(0);
  }

  auto outputs = impl->graph->sdpa(q, k, v, std::move(sdpa_options));
  auto& o = outputs[0];
  auto& stats = outputs[1];
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
      }));

  if (config.generate_stats) {
    if (stats == nullptr) {
      return nullptr;
    }
    stats->set_output(true).set_uid(kStatsUid).set_data_type(fe::DataType_t::FLOAT);
  }

  if (!impl->graph->build(
          reinterpret_cast<cudnnHandle_t>(handle.handle()),
          {fe::HeurMode_t::A}).is_good()) {
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
    const CudnnPagedAttentionExecution& execution) const {
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

  return impl_->graph->execute(
             reinterpret_cast<cudnnHandle_t>(handle.handle()),
             variant_pack,
             impl_->workspace).is_good() &&
         CheckCuda(cudaDeviceSynchronize());
}

}  // namespace nemotron
