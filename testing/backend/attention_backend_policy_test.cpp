#include "nemotron/attention_layer.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

using nemotron::AttentionBackend;
using nemotron::AttentionBackendPolicy;
using nemotron::AttentionBackendSelectorConfig;
using nemotron::AttentionKvCacheConfig;
using nemotron::KvCacheDataType;

class ScopedEnvVar {
 public:
  ScopedEnvVar(const char* name, const char* value) : name_(name) {
    const char* existing = std::getenv(name_);
    if (existing != nullptr) {
      had_old_value_ = true;
      old_value_ = existing;
    }
    if (value != nullptr) {
      setenv(name_, value, 1);
    } else {
      unsetenv(name_);
    }
  }

  ~ScopedEnvVar() {
    if (had_old_value_) {
      setenv(name_, old_value_.c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }

  ScopedEnvVar(const ScopedEnvVar&) = delete;
  ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

 private:
  const char* name_ = nullptr;
  bool had_old_value_ = false;
  std::string old_value_;
};

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

AttentionBackendSelectorConfig MakeSelectorConfig(
    std::size_t query_head_count,
    std::size_t kv_head_count,
    std::size_t head_dim,
    std::size_t tokens_per_page,
    std::size_t max_query_tokens,
    std::size_t max_kv_tokens,
    bool cudnn_available,
    long long cudnn_version) {
  AttentionBackendSelectorConfig config;
  config.cache_config = AttentionKvCacheConfig{
      1,
      kv_head_count,
      head_dim,
      tokens_per_page,
      KvCacheDataType::kBf16,
  };
  config.batch_size = 1;
  config.query_head_count = query_head_count;
  config.max_query_tokens = max_query_tokens;
  config.max_kv_tokens = max_kv_tokens;
  config.container_page_count =
      nemotron::RequiredPagesForTokens(config.cache_config, max_kv_tokens);
  config.page_table_entries =
      nemotron::RequiredPagesForTokens(config.cache_config, max_kv_tokens);
  config.cudnn_available = cudnn_available;
  config.cudnn_version = cudnn_version;
  config.causal = true;
  config.generate_stats = false;
  return config;
}

bool TestPolicyIgnoresLegacyEnvOverrides() {
  const ScopedEnvVar production_env("NEMOTRON_FORWARD_ATTENTION_PRODUCTION", "0");
  const ScopedEnvVar scalar_fallback_env("NEMOTRON_FORWARD_ATTENTION_SCALAR_FALLBACK", "1");

  AttentionBackendPolicy policy;
  const AttentionBackendSelectorConfig cudnn_config =
      MakeSelectorConfig(/*query_head_count=*/4, /*kv_head_count=*/2, /*head_dim=*/64, /*tokens_per_page=*/16,
                         /*max_query_tokens=*/2, /*max_kv_tokens=*/31, /*cudnn_available=*/true, /*cudnn_version=*/90500);
  const AttentionBackendSelectorConfig fallback_config =
      MakeSelectorConfig(/*query_head_count=*/4, /*kv_head_count=*/2, /*head_dim=*/64, /*tokens_per_page=*/16,
                         /*max_query_tokens=*/2, /*max_kv_tokens=*/31, /*cudnn_available=*/false, /*cudnn_version=*/0);

  return expect(
             policy.Select(cudnn_config, /*token_count=*/2) == AttentionBackend::kCudnnPaged,
             "legacy env vars should not block cuDNN selection when the config supports cuDNN") &&
         expect(
             policy.Select(fallback_config, /*token_count=*/2) == AttentionBackend::kDeviceFallback,
             "legacy env vars should not force the scalar fallback path");
}

bool TestPolicySupportsExactNanoDecodeShape() {
  AttentionBackendPolicy policy;
  const AttentionBackendSelectorConfig nano_config =
      MakeSelectorConfig(/*query_head_count=*/32, /*kv_head_count=*/2, /*head_dim=*/128, /*tokens_per_page=*/16,
                         /*max_query_tokens=*/1, /*max_kv_tokens=*/16, /*cudnn_available=*/false, /*cudnn_version=*/0);
  const AttentionBackendSelectorConfig generic_config =
      MakeSelectorConfig(/*query_head_count=*/8, /*kv_head_count=*/2, /*head_dim=*/128, /*tokens_per_page=*/16,
                         /*max_query_tokens=*/1, /*max_kv_tokens=*/16, /*cudnn_available=*/false, /*cudnn_version=*/0);

  return expect(
             policy.Supports(AttentionBackend::kNanoDecode, nano_config, /*token_count=*/1, /*device_sm=*/120),
             "Nano decode backend should support the exact measured Nano shape") &&
         expect(
             !policy.Supports(AttentionBackend::kNanoDecode, generic_config, /*token_count=*/1, /*device_sm=*/120),
             "Nano decode backend should reject non-Nano shapes") &&
         expect(
             !policy.Supports(AttentionBackend::kNanoDecode, nano_config, /*token_count=*/2, /*device_sm=*/120),
             "Nano decode backend should reject multi-token requests") &&
         expect(
             !policy.Supports(AttentionBackend::kNanoDecode, nano_config, /*token_count=*/1, /*device_sm=*/90),
             "Nano decode backend should reject insufficient device capability");
}

}  // namespace

int main() {
  if (!TestPolicyIgnoresLegacyEnvOverrides() ||
      !TestPolicySupportsExactNanoDecodeShape()) {
    return 1;
  }
  std::cout << "attention_backend_policy_test: PASS\n";
  return 0;
}
