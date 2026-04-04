#include "nemotron/cudnn_paged_attention.h"

#include <memory>
#include <optional>

namespace nemotron {

bool CudnnPagedAttentionConfig::valid() const {
  return false;
}

bool CudnnPagedAttentionConfig::operator==(const CudnnPagedAttentionConfig& other) const {
  (void)other;
  return false;
}

std::size_t CudnnPagedAttentionConfigHash::operator()(
    const CudnnPagedAttentionConfig& config) const {
  (void)config;
  return 0;
}

std::optional<CudnnPagedAttentionConfig> BuildCudnnPagedAttentionConfig(
    const PagedAttentionBatchPlan&,
    std::size_t,
    std::size_t,
    std::size_t,
    float,
    bool,
    bool) {
  return std::nullopt;
}

struct CudnnPagedAttentionPlan::Impl {
  CudnnPagedAttentionConfig config;
};

std::unique_ptr<CudnnPagedAttentionPlan> CudnnPagedAttentionPlan::Create(
    const CudnnHandle&,
    const CudnnPagedAttentionConfig&) {
  return nullptr;
}

CudnnPagedAttentionPlan::CudnnPagedAttentionPlan(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

CudnnPagedAttentionPlan::CudnnPagedAttentionPlan(CudnnPagedAttentionPlan&&) noexcept = default;
CudnnPagedAttentionPlan& CudnnPagedAttentionPlan::operator=(CudnnPagedAttentionPlan&&) noexcept =
    default;
CudnnPagedAttentionPlan::~CudnnPagedAttentionPlan() = default;

bool CudnnPagedAttentionPlan::valid() const {
  return false;
}

const CudnnPagedAttentionConfig& CudnnPagedAttentionPlan::config() const {
  static const CudnnPagedAttentionConfig kInvalidConfig;
  return impl_ ? impl_->config : kInvalidConfig;
}

std::size_t CudnnPagedAttentionPlan::workspace_bytes() const {
  return 0;
}

bool CudnnPagedAttentionPlan::Execute(
    const CudnnHandle&,
    const CudnnPagedAttentionExecution&) const {
  return false;
}

CudnnPagedAttentionPlanCache& CudnnPagedAttentionPlanCache::Global() {
  static CudnnPagedAttentionPlanCache cache;
  return cache;
}

std::shared_ptr<const CudnnPagedAttentionPlan> CudnnPagedAttentionPlanCache::GetOrCreate(
    const CudnnHandle&,
    const CudnnPagedAttentionConfig&) {
  return nullptr;
}

CudnnPagedAttentionPlanCacheStats CudnnPagedAttentionPlanCache::stats() const {
  return CudnnPagedAttentionPlanCacheStats{};
}

void CudnnPagedAttentionPlanCache::Clear() {}

}  // namespace nemotron
