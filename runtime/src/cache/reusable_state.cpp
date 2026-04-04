#include "nemotron/reusable_state.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <map>
#include <unordered_map>
#include <utility>

namespace nemotron {
namespace {

bool CheckCuda(cudaError_t status) {
  return status == cudaSuccess;
}

bool HasCudaDevice() {
  int device_count = 0;
  return CheckCuda(cudaGetDeviceCount(&device_count)) && device_count > 0;
}

void* ByteOffset(void* base, std::size_t offset) {
  return static_cast<std::byte*>(base) + offset;
}

const void* ByteOffset(const void* base, std::size_t offset) {
  return static_cast<const std::byte*>(base) + offset;
}

std::map<std::size_t, std::size_t>::iterator FindFirstFit(
    std::map<std::size_t, std::size_t>& free_regions,
    std::size_t bytes) {
  for (auto it = free_regions.begin(); it != free_regions.end(); ++it) {
    if (it->second >= bytes) {
      return it;
    }
  }
  return free_regions.end();
}

void InsertFreeRegion(
    std::map<std::size_t, std::size_t>& free_regions,
    std::size_t offset,
    std::size_t bytes) {
  if (bytes == 0) {
    return;
  }

  auto next = free_regions.lower_bound(offset);
  if (next != free_regions.begin()) {
    auto prev = std::prev(next);
    if (prev->first + prev->second == offset) {
      offset = prev->first;
      bytes += prev->second;
      free_regions.erase(prev);
    }
  }

  next = free_regions.lower_bound(offset);
  if (next != free_regions.end() && offset + bytes == next->first) {
    bytes += next->second;
    free_regions.erase(next);
  }

  free_regions.emplace(offset, bytes);
}

}  // namespace

struct ReusableStateArena::Impl {
  struct StateAllocation {
    ReusableStateId id = 0;
    ReusableStateKind kind = ReusableStateKind::kAttentionKv;
    std::string label;
    std::size_t bytes = 0;
    std::size_t ref_count = 0;
    std::size_t slab_offset = 0;
  };

  ReusableStateId next_id = 1;
  std::size_t current_bytes = 0;
  void* device_slab = nullptr;
  bool device_storage_init_failed = false;
  std::unordered_map<ReusableStateId, StateAllocation> allocations;
  std::map<std::size_t, std::size_t> free_regions;
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
    : max_bytes_(max_bytes), impl_(std::make_unique<Impl>()) {
  if (max_bytes_ != 0) {
    impl_->free_regions.emplace(0, max_bytes_);
  }
}

ReusableStateArena::~ReusableStateArena() {
  if (impl_ != nullptr && impl_->device_slab != nullptr) {
    cudaFree(impl_->device_slab);
  }
}

ReusableStateArena::ReusableStateArena(ReusableStateArena&&) noexcept = default;
ReusableStateArena& ReusableStateArena::operator=(ReusableStateArena&&) noexcept = default;

ReusableStateHandle ReusableStateArena::Allocate(
    ReusableStateKind kind,
    std::size_t bytes,
    const std::string& label) {
  if (bytes == 0 ||
      bytes > max_bytes_ ||
      impl_->current_bytes + bytes > max_bytes_ ||
      impl_->device_storage_init_failed) {
    return {};
  }

  // Lazy slab allocation: defer cudaMalloc until the first real allocation so
  // the slab does not compete with model weight uploads for VRAM.
  if (impl_->device_slab == nullptr && !impl_->device_storage_init_failed &&
      max_bytes_ != 0 && HasCudaDevice()) {
    // Cap the slab to actual free VRAM minus headroom for execution scratch.
    constexpr std::size_t kExecutionHeadroomBytes = 512ull * 1024 * 1024;  // 512 MiB
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (CheckCuda(cudaMemGetInfo(&free_bytes, &total_bytes)) &&
        free_bytes > kExecutionHeadroomBytes) {
      const std::size_t available = free_bytes - kExecutionHeadroomBytes;
      if (available < max_bytes_) {
        max_bytes_ = available;
        impl_->free_regions.clear();
        impl_->free_regions.emplace(0, max_bytes_);
        impl_->current_bytes = 0;
      }
    }
    if (max_bytes_ < bytes || !CheckCuda(cudaMalloc(&impl_->device_slab, max_bytes_))) {
      impl_->device_storage_init_failed = true;
      return {};
    }
  }

  auto free_it = FindFirstFit(impl_->free_regions, bytes);
  if (free_it == impl_->free_regions.end()) {
    return {};
  }

  const std::size_t slab_offset = free_it->first;
  const std::size_t free_bytes = free_it->second;
  impl_->free_regions.erase(free_it);
  if (free_bytes > bytes) {
    impl_->free_regions.emplace(slab_offset + bytes, free_bytes - bytes);
  }

  if (impl_->device_slab != nullptr &&
      !CheckCuda(cudaMemset(ByteOffset(impl_->device_slab, slab_offset), 0, bytes))) {
    InsertFreeRegion(impl_->free_regions, slab_offset, bytes);
    return {};
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
                                            slab_offset,
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
    InsertFreeRegion(impl_->free_regions, it->second.slab_offset, it->second.bytes);
    impl_->current_bytes -= it->second.bytes;
    impl_->allocations.erase(it);

    // When the slab is completely empty, release it back to CUDA so execution
    // scratch and KV pages can use the VRAM. The slab will be lazily
    // re-allocated on the next Allocate() call.
    if (impl_->allocations.empty() && impl_->device_slab != nullptr) {
      cudaFree(impl_->device_slab);
      impl_->device_slab = nullptr;
    }
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
      impl_->device_slab == nullptr) {
    return false;
  }
  void* device_dst = ByteOffset(impl_->device_slab, it->second.slab_offset + offset_bytes);
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
      impl_->device_slab == nullptr) {
    return false;
  }
  const void* device_src = ByteOffset(impl_->device_slab, it->second.slab_offset + offset_bytes);
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
      impl_->device_slab != nullptr,
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
