#include "nemotron/prefix_cache.h"
#include "nemotron/reusable_state.h"
#include "nemotron/runtime_config.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using nemotron::CacheLookupRequest;
using nemotron::PrefixCache;
using nemotron::ReusableStateArena;
using nemotron::RuntimeConfig;
using nemotron::SerializedPromptIdentity;
using nemotron::TokenId;

class ScopedEnvVar {
 public:
  explicit ScopedEnvVar(const char* name) : name_(name) {
    const char* current = std::getenv(name_);
    if (current != nullptr) {
      had_original_ = true;
      original_value_ = current;
    }
  }

  ~ScopedEnvVar() {
    if (had_original_) {
      setenv(name_, original_value_.c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }

 private:
  const char* name_;
  bool had_original_ = false;
  std::string original_value_;
};

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

SerializedPromptIdentity make_identity(std::vector<TokenId> token_ids) {
  SerializedPromptIdentity identity;
  identity.token_ids = std::move(token_ids);
  identity.tenant_namespace = "tenant-a";
  identity.tokenizer_revision = "tok-r1";
  identity.serializer_revision = "chat-v1";
  identity.model_revision = "model-r1";
  return identity;
}

bool test_runtime_config_defaults_to_enabled() {
  ScopedEnvVar scoped("NEMOTRON_PREFIX_CACHE");
  unsetenv("NEMOTRON_PREFIX_CACHE");

  const RuntimeConfig config = nemotron::LoadRuntimeConfigFromEnv();
  return expect(config.prefix_cache_enabled, "prefix cache should be enabled by default") &&
         expect(config.issues.empty(), "default config should not produce issues");
}

bool test_runtime_config_disables_cache_from_env() {
  ScopedEnvVar scoped("NEMOTRON_PREFIX_CACHE");
  setenv("NEMOTRON_PREFIX_CACHE", "0", 1);

  const RuntimeConfig config = nemotron::LoadRuntimeConfigFromEnv();
  return expect(!config.prefix_cache_enabled, "NEMOTRON_PREFIX_CACHE=0 should disable prefix caching") &&
         expect(config.issues.empty(), "recognized disable value should not produce issues");
}

bool test_runtime_config_reports_invalid_value() {
  ScopedEnvVar scoped("NEMOTRON_PREFIX_CACHE");
  setenv("NEMOTRON_PREFIX_CACHE", "banana", 1);

  const RuntimeConfig config = nemotron::LoadRuntimeConfigFromEnv();
  return expect(config.prefix_cache_enabled, "invalid values should fall back to enabled") &&
         expect(config.issues.size() == 1, "invalid value should produce one config issue");
}

bool test_disabled_cache_bypasses_lookup_and_publish() {
  PrefixCache cache(/*max_bytes=*/1 << 20);
  cache.SetEnabled(false);

  const auto node_id = cache.PublishConversationHead(
      "conv-disabled",
      make_identity({1, 2, 3}),
      {});
  CacheLookupRequest request;
  request.identity = make_identity({1, 2, 3, 4});
  request.conversation_id = "conv-disabled";
  const auto match = cache.Lookup(request);

  return expect(!cache.enabled(), "cache should report disabled") &&
         expect(node_id == 0, "publish should be bypassed while disabled") &&
         expect(!match.hit(), "lookup should miss while disabled") &&
         expect(cache.node_count() == 0, "disabled cache should not retain nodes");
}

bool test_disabling_cache_clears_owned_state() {
  ReusableStateArena arena(/*max_bytes=*/1 << 20);
  PrefixCache cache(/*max_bytes=*/1 << 20, &arena);

  const auto descriptor = arena.AllocateDescriptor(1024, 2048, "runtime-config");
  const auto node_id = cache.PublishConversationHead(
      "conv-owned",
      make_identity({9, 9, 9}),
      descriptor);
  arena.Release(descriptor);

  cache.SetEnabled(false);

  return expect(node_id != 0, "initial publish should succeed before disabling") &&
         expect(!cache.enabled(), "cache should report disabled after SetEnabled(false)") &&
         expect(cache.node_count() == 0, "disabling should clear cached nodes") &&
         expect(arena.allocation_count() == 0, "disabling should release owned state");
}

bool test_runtime_config_applies_to_cache() {
  PrefixCache cache(/*max_bytes=*/1 << 20);
  RuntimeConfig config;
  config.prefix_cache_enabled = false;

  nemotron::ApplyRuntimeConfig(config, &cache);

  return expect(!cache.enabled(), "applying runtime config should disable the cache");
}

}  // namespace

int main() {
  const bool ok =
      test_runtime_config_defaults_to_enabled() &&
      test_runtime_config_disables_cache_from_env() &&
      test_runtime_config_reports_invalid_value() &&
      test_disabled_cache_bypasses_lookup_and_publish() &&
      test_disabling_cache_clears_owned_state() &&
      test_runtime_config_applies_to_cache();

  if (!ok) {
    return 1;
  }
  std::cout << "runtime_config_test: PASS\n";
  return 0;
}
