#include "nemotron/cudnn_handle.h"
#include "nemotron/cudnn_paged_attention.h"
#include "nemotron/paged_attention_plan.h"
#include "nemotron/paged_kv_cache.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <optional>

namespace {

using nemotron::AttentionKvCacheConfig;
using nemotron::AttentionSequencePages;
using nemotron::BuildCudnnPagedAttentionConfig;
using nemotron::BuildPagedAttentionBatchPlan;
using nemotron::CudnnHandle;
using nemotron::KvCacheDataType;

constexpr std::size_t kNanoQueryHeadCount = 32;
// Match the currently checked-in KnownNemotron3Nano30BA3BConfig() values.
constexpr std::size_t kNanoKvHeadCount = 2;
constexpr std::size_t kNanoHeadDim = 128;
constexpr std::size_t kNanoTokensPerPage = 16;
constexpr std::size_t kMaxQueryTokens = 1;
constexpr std::size_t kContainerPageCount = 2;
constexpr bool kCausal = true;

std::optional<nemotron::PagedAttentionBatchPlan> BuildNanoBatchPlan() {
  AttentionKvCacheConfig cache_config;
  cache_config.layer_count = 1;
  cache_config.kv_head_count = kNanoKvHeadCount;
  cache_config.head_dim = kNanoHeadDim;
  cache_config.tokens_per_page = kNanoTokensPerPage;
  cache_config.dtype = KvCacheDataType::kBf16;

  return BuildPagedAttentionBatchPlan(
      cache_config,
      0,
      {
          AttentionSequencePages{kNanoTokensPerPage, {0}},
          AttentionSequencePages{kNanoTokensPerPage * 2, {1, 2}},
      },
      nullptr);
}

const char* BoolString(bool value) {
  return value ? "true" : "false";
}

}  // namespace

int main() {
  const auto handle = CudnnHandle::Create();
  const bool handle_valid = handle != nullptr && handle->valid();
  const long long handle_version = handle != nullptr ? handle->version() : 0;

  std::cout << "CudnnHandle::Create(): valid=" << BoolString(handle_valid)
            << ", version=" << handle_version << "\n";

  const auto kv_plan = BuildNanoBatchPlan();
  if (!kv_plan.has_value()) {
    std::cout << "BuildPagedAttentionBatchPlan(...): failed\n";
  }

  const auto config = kv_plan.has_value()
                          ? BuildCudnnPagedAttentionConfig(
                                *kv_plan,
                                kNanoQueryHeadCount,
                                kMaxQueryTokens,
                                kContainerPageCount,
                                1.0f / std::sqrt(static_cast<float>(kNanoHeadDim)),
                                kCausal)
                          : std::optional<nemotron::CudnnPagedAttentionConfig>{};

  std::cout << "BuildCudnnPagedAttentionConfig(...): "
            << (config.has_value() ? "has value" : "no value") << "\n";

  std::cout << "cuDNN FE paged attention: "
            << ((handle_valid && config.has_value()) ? "AVAILABLE"
                                                     : "NOT AVAILABLE (stub build)")
            << "\n";
  return 0;
}
