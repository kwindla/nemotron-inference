#include "nemotron/flashinfer_moe_backend.h"

#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cstdlib>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace nemotron {
namespace {

struct FlashInferPluginApi {
  void* handle = nullptr;
  NemotronFlashInferAbiVersionFn abi_version = nullptr;
  NemotronFlashInferBackendKindFn backend_kind = nullptr;
  NemotronFlashInferRoutedMoECreateFn create = nullptr;
  NemotronFlashInferRoutedMoEDestroyFn destroy = nullptr;
  NemotronFlashInferRoutedMoERunFn run = nullptr;
  NemotronFlashInferLastErrorFn last_error = nullptr;
  FlashInferRoutedMoEAvailability availability;
};

FlashInferPluginApi ProbeFlashInferPlugin() {
  FlashInferPluginApi api;
  std::string last_failure_detail;

  std::vector<std::string> candidates;
  if (const char* env = std::getenv("NEMOTRON_FLASHINFER_MOE_LIBRARY");
      env != nullptr && *env != '\0') {
    candidates.emplace_back(env);
  }
  candidates.emplace_back("libnemotron_flashinfer_moe.so");
  candidates.emplace_back("libflashinfer_moe_plugin.so");

  for (const std::string& candidate : candidates) {
    void* handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      const char* error = dlerror();
      last_failure_detail = error != nullptr
                                ? "dlopen(" + candidate + ") failed: " + std::string(error)
                                : "dlopen(" + candidate + ") failed";
      continue;
    }

    api.handle = handle;
    api.availability.library_found = true;
    api.availability.library_path = candidate;

    api.abi_version = reinterpret_cast<NemotronFlashInferAbiVersionFn>(
        dlsym(handle, "nemotron_flashinfer_moe_abi_version"));
    api.backend_kind = reinterpret_cast<NemotronFlashInferBackendKindFn>(
        dlsym(handle, "nemotron_flashinfer_moe_backend_kind"));
    api.create = reinterpret_cast<NemotronFlashInferRoutedMoECreateFn>(
        dlsym(handle, "nemotron_flashinfer_routed_moe_create"));
    api.destroy = reinterpret_cast<NemotronFlashInferRoutedMoEDestroyFn>(
        dlsym(handle, "nemotron_flashinfer_routed_moe_destroy"));
    api.run = reinterpret_cast<NemotronFlashInferRoutedMoERunFn>(
        dlsym(handle, "nemotron_flashinfer_routed_moe_run"));
    api.last_error = reinterpret_cast<NemotronFlashInferLastErrorFn>(
        dlsym(handle, "nemotron_flashinfer_moe_last_error"));

    if (api.abi_version == nullptr ||
        api.create == nullptr ||
        api.destroy == nullptr ||
        api.run == nullptr) {
      api.availability.detail =
          "plugin found but required routed MoE symbols are missing";
      dlclose(handle);
      api = FlashInferPluginApi{};
      last_failure_detail = "plugin found at " + candidate +
                            " but required routed MoE symbols are missing";
      continue;
    }

    if (api.abi_version() != NEMOTRON_FLASHINFER_MOE_PLUGIN_ABI_VERSION) {
      api.availability.detail = "plugin ABI version mismatch";
      dlclose(handle);
      api = FlashInferPluginApi{};
      last_failure_detail = "plugin ABI version mismatch at " + candidate;
      continue;
    }

    api.availability.abi_compatible = true;
    api.availability.available = true;
    api.availability.backend_kind = api.backend_kind != nullptr
                                        ? api.backend_kind()
                                        : NEMOTRON_FLASHINFER_BACKEND_KIND_UNKNOWN;
    api.availability.detail = "routed FlashInfer MoE plugin available";
    return api;
  }

  api.availability.detail = last_failure_detail.empty()
                                ? "no FlashInfer routed MoE plugin found; using custom fused backend"
                                : last_failure_detail;
  return api;
}

const FlashInferPluginApi& GetFlashInferPluginApi() {
  static const FlashInferPluginApi api = ProbeFlashInferPlugin();
  return api;
}

}  // namespace

const char* ToString(RoutedMoEBackendKind kind) {
  switch (kind) {
    case RoutedMoEBackendKind::kCustomFused:
      return "custom_fused";
    case RoutedMoEBackendKind::kFlashInfer:
      return "flashinfer";
  }
  return "unknown";
}

RoutedMoEBackendKind ResolveRequestedRoutedMoEBackend() {
  const char* env = std::getenv("NEMOTRON_ROUTED_MOE_BACKEND");
  if (env == nullptr || *env == '\0' || std::string(env) == "auto") {
    return GetFlashInferRoutedMoEAvailability().available
               ? RoutedMoEBackendKind::kFlashInfer
               : RoutedMoEBackendKind::kCustomFused;
  }
  const std::string value(env);
  if (value == "flashinfer") {
    return RoutedMoEBackendKind::kFlashInfer;
  }
  return RoutedMoEBackendKind::kCustomFused;
}

bool RoutedMoEBackendStrict() {
  const char* env = std::getenv("NEMOTRON_ROUTED_MOE_BACKEND_STRICT");
  return env != nullptr && *env != '\0';
}

FlashInferRoutedMoEAvailability GetFlashInferRoutedMoEAvailability() {
  return GetFlashInferPluginApi().availability;
}

struct FlashInferRoutedMoEBackend::Impl {
  FlashInferRoutedMoEAvailability availability;
  const FlashInferPluginApi* api = nullptr;
  NemotronFlashInferRoutedMoEHandle* handle = nullptr;
};

std::unique_ptr<FlashInferRoutedMoEBackend> FlashInferRoutedMoEBackend::Create(
    const NemotronFlashInferRoutedMoECreateParams& params) {
  const FlashInferPluginApi& api = GetFlashInferPluginApi();
  if (!api.availability.available) {
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  impl->availability = api.availability;
  impl->api = &api;
  impl->handle = api.create(&params);
  if (impl->handle == nullptr) {
    impl->availability.available = false;
    impl->availability.detail =
        api.last_error != nullptr && api.last_error() != nullptr
            ? api.last_error()
            : "FlashInfer routed MoE create failed";
    return std::unique_ptr<FlashInferRoutedMoEBackend>(
        new FlashInferRoutedMoEBackend(std::move(impl)));
  }
  return std::unique_ptr<FlashInferRoutedMoEBackend>(
      new FlashInferRoutedMoEBackend(std::move(impl)));
}

FlashInferRoutedMoEBackend::FlashInferRoutedMoEBackend(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

FlashInferRoutedMoEBackend::FlashInferRoutedMoEBackend(
    FlashInferRoutedMoEBackend&&) noexcept = default;
FlashInferRoutedMoEBackend& FlashInferRoutedMoEBackend::operator=(
    FlashInferRoutedMoEBackend&&) noexcept = default;

FlashInferRoutedMoEBackend::~FlashInferRoutedMoEBackend() {
  if (impl_ != nullptr &&
      impl_->api != nullptr &&
      impl_->api->destroy != nullptr &&
      impl_->handle != nullptr) {
    impl_->api->destroy(impl_->handle);
    impl_->handle = nullptr;
  }
}

bool FlashInferRoutedMoEBackend::valid() const {
  return impl_ != nullptr &&
         impl_->api != nullptr &&
         impl_->api->run != nullptr &&
         impl_->handle != nullptr;
}

const FlashInferRoutedMoEAvailability& FlashInferRoutedMoEBackend::availability() const {
  static const FlashInferRoutedMoEAvailability unavailable{};
  return impl_ != nullptr ? impl_->availability : unavailable;
}

bool FlashInferRoutedMoEBackend::Run(
    const DeviceTensorFp32& hidden_states_fp32,
    std::size_t token_count,
    const std::int32_t* topk_ids_device,
    const float* topk_weights_device,
    DeviceTensorFp32* output_fp32) const {
  if (!valid() ||
      !hidden_states_fp32.valid() ||
      topk_ids_device == nullptr ||
      topk_weights_device == nullptr ||
      output_fp32 == nullptr ||
      !output_fp32->valid()) {
    return false;
  }

  NemotronFlashInferRoutedMoERunParams params{};
  params.hidden_states_fp32 = hidden_states_fp32.data();
  params.token_count = token_count;
  params.topk_ids_device = topk_ids_device;
  params.topk_weights_device = topk_weights_device;
  params.output_fp32 = output_fp32->data();
  params.cuda_stream = nullptr;
  return impl_->api->run(impl_->handle, &params);
}

}  // namespace nemotron
