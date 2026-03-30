#include "nemotron/runtime_config.h"

#include "nemotron/prefix_cache.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <optional>
#include <string>

namespace nemotron {
namespace {

std::string lowercase_copy(std::string value) {
  std::transform(
      value.begin(),
      value.end(),
      value.begin(),
      [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return value;
}

std::optional<bool> parse_bool_env_value(const std::string& value) {
  const std::string lowered = lowercase_copy(value);
  if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on" || lowered == "enabled") {
    return true;
  }
  if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "off" || lowered == "disabled") {
    return false;
  }
  return std::nullopt;
}

}  // namespace

RuntimeConfig LoadRuntimeConfigFromEnv() {
  RuntimeConfig config;

  constexpr const char* kPrefixCacheEnvVar = "NEMOTRON_PREFIX_CACHE";
  const char* prefix_cache_env = std::getenv(kPrefixCacheEnvVar);
  if (prefix_cache_env == nullptr || prefix_cache_env[0] == '\0') {
    return config;
  }

  const std::string value(prefix_cache_env);
  const std::optional<bool> parsed = parse_bool_env_value(value);
  if (!parsed.has_value()) {
    config.issues.push_back(RuntimeConfigIssue{
        kPrefixCacheEnvVar,
        value,
        "unrecognized boolean value; expected one of 0/1, false/true, no/yes, or off/on",
    });
    return config;
  }

  config.prefix_cache_enabled = *parsed;
  return config;
}

void ApplyRuntimeConfig(const RuntimeConfig& config, PrefixCache* prefix_cache) {
  if (prefix_cache == nullptr) {
    return;
  }
  prefix_cache->SetEnabled(config.prefix_cache_enabled);
}

}  // namespace nemotron
