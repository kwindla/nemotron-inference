#include "nemotron/reusable_state.h"

#include <iostream>
#include <string>

namespace {

using nemotron::ReusableStateArena;
using nemotron::ReusableStateKind;

bool expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool test_allocate_and_release_descriptor() {
  ReusableStateArena arena(/*max_bytes=*/8192);
  const auto descriptor = arena.AllocateDescriptor(1024, 2048, "descriptor-1");
  const auto kv_view = arena.Describe(descriptor.kv_state.id);
  const auto mamba_view = arena.Describe(descriptor.mamba_state.id);

  arena.Release(descriptor);

  return expect(descriptor.valid(), "descriptor allocation should succeed") &&
         expect(arena.max_bytes() == 8192, "arena max bytes should be reported") &&
         expect(kv_view.has_value() && kv_view->ref_count == 1, "KV allocation should start with ref_count 1") &&
         expect(mamba_view.has_value() && mamba_view->ref_count == 1, "Mamba allocation should start with ref_count 1") &&
         expect(arena.current_bytes() == 0, "releasing the descriptor should free all arena bytes") &&
         expect(arena.allocation_count() == 0, "releasing the descriptor should remove both allocations");
}

bool test_retain_increments_refcount_without_duplicate_bytes() {
  ReusableStateArena arena(/*max_bytes=*/8192);
  const auto descriptor = arena.AllocateDescriptor(1024, 2048, "descriptor-2");
  const std::size_t bytes_before_retain = arena.current_bytes();
  const bool retained = arena.Retain(descriptor);
  const auto kv_view = arena.Describe(descriptor.kv_state.id);
  const auto mamba_view = arena.Describe(descriptor.mamba_state.id);

  arena.Release(descriptor);
  const std::size_t bytes_after_one_release = arena.current_bytes();
  arena.Release(descriptor);

  return expect(retained, "retaining a valid descriptor should succeed") &&
         expect(bytes_before_retain == 3072, "arena bytes should equal the unique allocation bytes") &&
         expect(arena.current_bytes() == 0, "bytes should be zero after both retained references are released") &&
         expect(kv_view.has_value() && kv_view->ref_count == 2, "KV retain should increment ref_count") &&
         expect(mamba_view.has_value() && mamba_view->ref_count == 2, "Mamba retain should increment ref_count") &&
         expect(bytes_after_one_release == 3072, "releasing one of two references should not free the allocation");
}

bool test_capacity_limit_rejects_large_descriptor() {
  ReusableStateArena arena(/*max_bytes=*/4096);
  const auto descriptor = arena.AllocateDescriptor(3000, 2000, "descriptor-3");

  return expect(!descriptor.valid(), "descriptor allocation should fail when arena capacity is exceeded") &&
         expect(arena.current_bytes() == 0, "failed allocation should not consume bytes") &&
         expect(arena.allocation_count() == 0, "failed allocation should not leave partial state behind");
}

bool test_single_allocation_has_expected_kind_and_label() {
  ReusableStateArena arena(/*max_bytes=*/4096);
  const auto handle = arena.Allocate(ReusableStateKind::kAttentionKv, 1024, "kv-buffer");
  const auto view = arena.Describe(handle.id);
  arena.Release(handle);

  return expect(handle.valid(), "single allocation should succeed") &&
         expect(view.has_value(), "single allocation should be describable") &&
         expect(view->kind == ReusableStateKind::kAttentionKv, "kind should round-trip through the arena") &&
         expect(view->label == "kv-buffer", "label should round-trip through the arena");
}

bool test_released_capacity_is_reusable() {
  ReusableStateArena arena(/*max_bytes=*/3072);
  const auto first = arena.Allocate(ReusableStateKind::kAttentionKv, 1024, "slot-a");
  const auto second = arena.Allocate(ReusableStateKind::kMambaRecurrent, 2048, "slot-b");
  arena.Release(first);
  arena.Release(second);

  const auto recycled = arena.Allocate(ReusableStateKind::kAttentionKv, 3072, "slot-reused");
  const auto recycled_view = arena.Describe(recycled.id);
  arena.Release(recycled);

  return expect(first.valid() && second.valid(), "initial allocations should consume the full budget") &&
         expect(recycled.valid(), "released bytes should be reusable for a later allocation") &&
         expect(
             recycled_view.has_value() && recycled_view->bytes == 3072,
             "reused allocation should span the full arena budget") &&
         expect(arena.current_bytes() == 0, "releasing the reused allocation should return the arena to idle");
}

}  // namespace

int main() {
  const bool ok =
      test_allocate_and_release_descriptor() &&
      test_retain_increments_refcount_without_duplicate_bytes() &&
      test_capacity_limit_rejects_large_descriptor() &&
      test_single_allocation_has_expected_kind_and_label() &&
      test_released_capacity_is_reusable();

  if (!ok) {
    return 1;
  }
  std::cout << "reusable_state_test: PASS\n";
  return 0;
}
