#include "nemotron/reusable_state.h"

#include <unordered_map>
#include <utility>

namespace nemotron {

struct ReusableStateArena::Impl {
  struct StateAllocation {
    ReusableStateId id = 0;
    ReusableStateKind kind = ReusableStateKind::kAttentionKv;
    std::string label;
    std::size_t bytes = 0;
    std::size_t ref_count = 0;
  };

  ReusableStateId next_id = 1;
  std::size_t current_bytes = 0;
  std::unordered_map<ReusableStateId, StateAllocation> allocations;
};

bool ReusableStateHandle::valid() const {
  return id != 0 && bytes != 0;
}

bool ReusableStateDescriptor::valid() const {
  return kv_state.valid() && mamba_state.valid();
}

std::size_t ReusableStateDescriptor::total_bytes() const {
  return kv_state.bytes + mamba_state.bytes;
}

ReusableStateArena::ReusableStateArena(std::size_t max_bytes)
    : max_bytes_(max_bytes), impl_(std::make_unique<Impl>()) {}

ReusableStateArena::~ReusableStateArena() = default;
ReusableStateArena::ReusableStateArena(ReusableStateArena&&) noexcept = default;
ReusableStateArena& ReusableStateArena::operator=(ReusableStateArena&&) noexcept = default;

ReusableStateHandle ReusableStateArena::Allocate(
    ReusableStateKind kind,
    std::size_t bytes,
    const std::string& label) {
  if (bytes == 0 || impl_->current_bytes + bytes > max_bytes_) {
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
    impl_->current_bytes -= it->second.bytes;
    impl_->allocations.erase(it);
  }
}

void ReusableStateArena::Release(const ReusableStateDescriptor& descriptor) {
  Release(descriptor.kv_state);
  Release(descriptor.mamba_state);
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
