#pragma once

#include <string>
#include <vector>

namespace nemotron {

class PrefixCache;

struct RuntimeConfigIssue {
  std::string environment_variable;
  std::string value;
  std::string message;
};

struct RuntimeConfig {
  bool prefix_cache_enabled = true;
  std::vector<RuntimeConfigIssue> issues;
};

RuntimeConfig LoadRuntimeConfigFromEnv();
void ApplyRuntimeConfig(const RuntimeConfig& config, PrefixCache* prefix_cache);

}  // namespace nemotron
