#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include "nemotron/request_context.h"
#include "nemotron/reusable_state.h"

namespace nemotron {

std::size_t RequiredKvSnapshotBytes(const RequestExecutionContext& request_context);
std::size_t RequiredMambaSnapshotBytes(const RequestExecutionContext& request_context);

// Snapshot copies the live request state into arena-owned storage. The request
// context keeps ownership of live KV pages via its PagedKvCacheArena; restore
// allocates fresh live pages in the target request and copies the cached bytes
// back into them.
std::optional<ReusableStateDescriptor> SnapshotRequestState(
    ReusableStateArena& arena,
    const RequestExecutionContext& request_context,
    const std::string& label);

bool RestoreRequestState(
    const ReusableStateArena& arena,
    const ReusableStateDescriptor& descriptor,
    std::size_t expected_prefix_token_count,
    RequestExecutionContext& request_context);

}  // namespace nemotron
