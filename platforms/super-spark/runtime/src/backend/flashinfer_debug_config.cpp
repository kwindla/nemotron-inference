#include "tensorrt_llm/common/assert.h"

#include <cstdlib>

namespace {

bool InitCheckDebugEnabled() {
  const char* value = std::getenv("TLLM_DEBUG_MODE");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

}  // namespace

bool DebugConfig::isCheckDebugEnabled() {
  static const bool enabled = InitCheckDebugEnabled();
  return enabled;
}
