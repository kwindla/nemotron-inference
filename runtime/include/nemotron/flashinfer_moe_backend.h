#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "nemotron/device_tensor.h"
#include "nemotron/flashinfer_moe_plugin_abi.h"

namespace nemotron {

enum class RoutedMoEBackendKind {
  kCustomFused,
  kFlashInfer,
};

const char* ToString(RoutedMoEBackendKind kind);

struct FlashInferRoutedMoEAvailability {
  bool library_found = false;
  bool abi_compatible = false;
  bool available = false;
  int backend_kind = NEMOTRON_FLASHINFER_BACKEND_KIND_UNKNOWN;
  std::string library_path;
  std::string detail;
};

RoutedMoEBackendKind ResolveRequestedRoutedMoEBackend();
bool RoutedMoEBackendStrict();
FlashInferRoutedMoEAvailability GetFlashInferRoutedMoEAvailability();

class FlashInferRoutedMoEBackend {
 public:
  static std::unique_ptr<FlashInferRoutedMoEBackend> Create(
      const NemotronFlashInferRoutedMoECreateParams& params);

  FlashInferRoutedMoEBackend(FlashInferRoutedMoEBackend&&) noexcept;
  FlashInferRoutedMoEBackend& operator=(FlashInferRoutedMoEBackend&&) noexcept;
  ~FlashInferRoutedMoEBackend();

  FlashInferRoutedMoEBackend(const FlashInferRoutedMoEBackend&) = delete;
  FlashInferRoutedMoEBackend& operator=(const FlashInferRoutedMoEBackend&) = delete;

  bool valid() const;
  const FlashInferRoutedMoEAvailability& availability() const;
  bool Run(
      const DeviceTensorFp32& hidden_states_fp32,
      std::size_t token_count,
      const std::int32_t* topk_ids_device,
      const float* topk_weights_device,
      DeviceTensorFp32* output_fp32) const;

 private:
  struct Impl;

  explicit FlashInferRoutedMoEBackend(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace nemotron
