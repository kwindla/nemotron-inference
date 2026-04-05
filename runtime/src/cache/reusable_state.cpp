#include "nemotron/reusable_state.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <unordered_map>
#include <utility>

namespace nemotron {
namespace {

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

void ClearCudaErrorState() {
  static_cast<void>(cudaGetLastError());
}

bool HasCudaDevice() {
  int device_count = 0;
  return CheckCuda(cudaGetDeviceCount(&device_count)) && device_count > 0;
}

}  // namespace

struct ReusableStateArena::Impl {
  struct StateAllocation {
    ReusableStateId id = 0;
    ReusableStateKind kind = ReusableStateKind::kAttentionKv;
    std::string label;
    std::size_t bytes = 0;
    std::size_t ref_count = 0;
    void* device_storage = nullptr;
  };

  ReusableStateId next_id = 1;
  std::size_t current_bytes = 0;
  std::unordered_map<ReusableStateId, StateAllocation> allocations;
};

bool ReusableStateHandle::valid() const {
  return id != 0 && bytes != 0;
}

bool ReusableStateSnapshotLayout::valid() const {
  return prefix_token_count != 0 &&
         live_kv_pages_per_layer != 0 &&
         hidden_bytes != 0 &&
         residual_bytes != 0 &&
         BuildAttentionKvPageGeometry(attention_kv_cache).has_value();
}

bool ReusableStateDescriptor::valid() const {
  return kv_state.valid() && mamba_state.valid();
}

bool ReusableStateDescriptor::has_snapshot_layout() const {
  return snapshot_layout.valid();
}

std::size_t ReusableStateDescriptor::total_bytes() const {
  return kv_state.bytes + mamba_state.bytes;
}

ReusableStateArena::ReusableStateArena(std::size_t max_bytes)
    : max_bytes_(max_bytes), impl_(std::make_unique<Impl>()) {}

ReusableStateArena::~ReusableStateArena() {
  if (impl_ != nullptr) {
    for (auto& [_, allocation] : impl_->allocations) {
      if (allocation.device_storage != nullptr) {
        cudaFree(allocation.device_storage);
      }
    }
  }
}

ReusableStateArena::ReusableStateArena(ReusableStateArena&&) noexcept = default;
ReusableStateArena& ReusableStateArena::operator=(ReusableStateArena&&) noexcept = default;

ReusableStateHandle ReusableStateArena::Allocate(
    ReusableStateKind kind,
    std::size_t bytes,
    const std::string& label) {
  if (bytes == 0 || bytes > max_bytes_ || impl_->current_bytes + bytes > max_bytes_) {
    return {};
  }

  void* device_storage = nullptr;
  if (HasCudaDevice()) {
    if (!CheckCuda(cudaMalloc(&device_storage, bytes))) {
      // Snapshot publish may intentionally retry after a handled allocation
      // failure, so clear the sticky runtime error before returning.
      ClearCudaErrorState();
      return {};
    }
    if (!CheckCuda(cudaMemset(device_storage, 0, bytes))) {
      cudaFree(device_storage);
      ClearCudaErrorState();
      return {};
    }
  }

  ReusableStateHandle handle;
  handle.id = impl_->next_id++;
  handle.kind = kind;
  handle.bytes = bytes;

  impl_->allocations.emplace(handle.id, Impl::StateAllocation{
                                            handle.id,
                                            kind,
                                            label,
                                            bytes,
                                            1,
                                            device_storage,
                                        });
  impl_->current_bytes += bytes;
  return handle;
}

ReusableStateDescriptor ReusableStateArena::AllocateDescriptor(
    std::size_t kv_bytes,
    std::size_t mamba_bytes,
    const std::string& label) {
  const ReusableStateHandle kv_state = Allocate(
      ReusableStateKind::kAttentionKv,
      kv_bytes,
      label + "/kv");
  if (!kv_state.valid()) {
    return {};
  }

  const ReusableStateHandle mamba_state = Allocate(
      ReusableStateKind::kMambaRecurrent,
      mamba_bytes,
      label + "/mamba");
  if (!mamba_state.valid()) {
    Release(kv_state);
    return {};
  }

  return ReusableStateDescriptor{
      kv_state,
      mamba_state,
      {},
  };
}

bool ReusableStateArena::Retain(const ReusableStateHandle& handle) {
  if (!handle.valid()) {
    return false;
  }
  auto it = impl_->allocations.find(handle.id);
  if (it == impl_->allocations.end()) {
    return false;
  }
  if (it->second.kind != handle.kind || it->second.bytes != handle.bytes) {
    return false;
  }
  ++it->second.ref_count;
  return true;
}

bool ReusableStateArena::Retain(const ReusableStateDescriptor& descriptor) {
  if (!descriptor.valid()) {
    return false;
  }
  if (!Retain(descriptor.kv_state)) {
    return false;
  }
  if (!Retain(descriptor.mamba_state)) {
    Release(descriptor.kv_state);
    return false;
  }
  return true;
}

void ReusableStateArena::Release(const ReusableStateHandle& handle) {
  if (!handle.valid()) {
    return;
  }
  auto it = impl_->allocations.find(handle.id);
  if (it == impl_->allocations.end()) {
    return;
  }
  if (it->second.kind != handle.kind || it->second.bytes != handle.bytes) {
    return;
  }
  if (it->second.ref_count == 0) {
    return;
  }

  --it->second.ref_count;
  if (it->second.ref_count == 0) {
    if (it->second.device_storage != nullptr) {
      cudaFree(it->second.device_storage);
    }
    impl_->current_bytes -= it->second.bytes;
    impl_->allocations.erase(it);
  }
}

void ReusableStateArena::Release(const ReusableStateDescriptor& descriptor) {
  Release(descriptor.kv_state);
  Release(descriptor.mamba_state);
}

bool ReusableStateArena::CopyFromDevice(
    const ReusableStateHandle& handle,
    std::size_t offset_bytes,
    const void* device_src,
    std::size_t bytes) {
  if (!handle.valid() || device_src == nullptr || bytes == 0 || offset_bytes + bytes > handle.bytes) {
    return false;
  }
  auto it = impl_->allocations.find(handle.id);
  if (it == impl_->allocations.end() ||
      it->second.kind != handle.kind ||
      it->second.bytes != handle.bytes ||
      it->second.device_storage == nullptr) {
    return false;
  }
  void* device_dst = static_cast<std::byte*>(it->second.device_storage) + offset_bytes;
  return CheckCuda(cudaMemcpy(device_dst, device_src, bytes, cudaMemcpyDeviceToDevice));
}

bool ReusableStateArena::CopyToDevice(
    const ReusableStateHandle& handle,
    std::size_t offset_bytes,
    void* device_dst,
    std::size_t bytes) const {
  if (!handle.valid() || device_dst == nullptr || bytes == 0 || offset_bytes + bytes > handle.bytes) {
    return false;
  }
  auto it = impl_->allocations.find(handle.id);
  if (it == impl_->allocations.end() ||
      it->second.kind != handle.kind ||
      it->second.bytes != handle.bytes ||
      it->second.device_storage == nullptr) {
    return false;
  }
  const void* device_src =
      static_cast<const std::byte*>(it->second.device_storage) + offset_bytes;
  return CheckCuda(cudaMemcpy(device_dst, device_src, bytes, cudaMemcpyDeviceToDevice));
}

std::optional<ReusableStateView> ReusableStateArena::Describe(ReusableStateId id) const {
  auto it = impl_->allocations.find(id);
  if (it == impl_->allocations.end()) {
    return std::nullopt;
  }
  return ReusableStateView{
      it->second.id,
      it->second.kind,
      it->second.label,
      it->second.bytes,
      it->second.ref_count,
      it->second.device_storage != nullptr,
  };
}

std::size_t ReusableStateArena::current_bytes() const {
  return impl_->current_bytes;
}

std::size_t ReusableStateArena::max_bytes() const {
  return max_bytes_;
}

std::size_t ReusableStateArena::allocation_count() const {
  return impl_->allocations.size();
}

}  // namespace nemotron
