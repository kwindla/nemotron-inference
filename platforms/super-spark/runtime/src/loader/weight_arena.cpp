#include "nemotron/weight_arena.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace nemotron {
namespace {

bool IsPowerOfTwo(std::size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

std::size_t NormalizeAlignment(std::size_t alignment_bytes) {
  if (alignment_bytes == 0) {
    return alignof(std::max_align_t);
  }

  std::size_t normalized = alignment_bytes;
  if (normalized < sizeof(void*)) {
    normalized = sizeof(void*);
  }
  while (!IsPowerOfTwo(normalized)) {
    ++normalized;
  }
  return normalized;
}

bool RangeFits(std::size_t arena_size, std::size_t offset_bytes, std::size_t nbytes) {
  return offset_bytes <= arena_size && nbytes <= (arena_size - offset_bytes);
}

}  // namespace

struct WeightArena::Impl {
  ~Impl() {
    std::free(base);
  }

  WeightArenaPlan plan;
  std::uint8_t* base = nullptr;
  std::size_t size_bytes = 0;
  std::size_t base_alignment_bytes = 0;
  bool valid = false;
};

std::unique_ptr<WeightArena> WeightArena::CreateFromPlan(const WeightArenaPlan& plan) {
  if (!plan.valid()) {
    return nullptr;
  }

  auto impl = std::make_unique<Impl>();
  impl->plan = plan;
  impl->size_bytes = plan.total_arena_bytes();

  std::size_t max_alignment_bytes = alignof(std::max_align_t);
  for (const WeightArenaTensorPlacement& tensor : plan.tensors()) {
    max_alignment_bytes = std::max(max_alignment_bytes, NormalizeAlignment(tensor.packed_buffer.alignment_bytes));
    for (const WeightArenaBufferPlacement& auxiliary : tensor.auxiliary_buffers) {
      max_alignment_bytes = std::max(max_alignment_bytes, NormalizeAlignment(auxiliary.alignment_bytes));
    }
  }
  impl->base_alignment_bytes = max_alignment_bytes;

  if (impl->size_bytes > 0) {
    void* base = nullptr;
    if (posix_memalign(&base, impl->base_alignment_bytes, impl->size_bytes) != 0) {
      return nullptr;
    }
    impl->base = static_cast<std::uint8_t*>(base);
    std::memset(impl->base, 0, impl->size_bytes);
  }

  for (const WeightArenaTensorPlacement& tensor : plan.tensors()) {
    if (!tensor.packed_buffer.host_bytes.valid() ||
        tensor.packed_buffer.host_bytes.size != tensor.packed_buffer.nbytes ||
        !RangeFits(impl->size_bytes, tensor.packed_buffer.arena_offset_bytes, tensor.packed_buffer.nbytes)) {
      return nullptr;
    }
    std::memcpy(
        impl->base + tensor.packed_buffer.arena_offset_bytes,
        tensor.packed_buffer.host_bytes.data,
        tensor.packed_buffer.nbytes);

    for (const WeightArenaBufferPlacement& auxiliary : tensor.auxiliary_buffers) {
      if (!auxiliary.host_bytes.valid() ||
          auxiliary.host_bytes.size != auxiliary.nbytes ||
          !RangeFits(impl->size_bytes, auxiliary.arena_offset_bytes, auxiliary.nbytes)) {
        return nullptr;
      }
      std::memcpy(
          impl->base + auxiliary.arena_offset_bytes,
          auxiliary.host_bytes.data,
          auxiliary.nbytes);
    }
  }

  impl->valid = true;
  return std::unique_ptr<WeightArena>(new WeightArena(std::move(impl)));
}

WeightArena::WeightArena(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

WeightArena::WeightArena(WeightArena&&) noexcept = default;
WeightArena& WeightArena::operator=(WeightArena&&) noexcept = default;
WeightArena::~WeightArena() = default;

bool WeightArena::valid() const {
  return impl_ != nullptr && impl_->valid;
}

std::size_t WeightArena::size_bytes() const {
  return impl_->size_bytes;
}

std::size_t WeightArena::base_alignment_bytes() const {
  return impl_->base_alignment_bytes;
}

const WeightArenaPlan& WeightArena::plan() const {
  return impl_->plan;
}

std::optional<WeightArenaTensorView> WeightArena::FindTensor(const std::string& tensor_name) const {
  const WeightArenaTensorPlacement* tensor = impl_->plan.FindTensor(tensor_name);
  if (tensor == nullptr) {
    return std::nullopt;
  }

  WeightArenaTensorView view;
  view.placement = tensor;
  view.packed_bytes = ByteRangeView{
      impl_->base + tensor->packed_buffer.arena_offset_bytes,
      tensor->packed_buffer.nbytes,
  };
  view.auxiliary_buffers.reserve(tensor->auxiliary_buffers.size());
  for (const WeightArenaBufferPlacement& auxiliary : tensor->auxiliary_buffers) {
    view.auxiliary_buffers.push_back(WeightArenaBufferView{
        &auxiliary,
        ByteRangeView{
            impl_->base + auxiliary.arena_offset_bytes,
            auxiliary.nbytes,
        },
    });
  }
  return view;
}

}  // namespace nemotron
