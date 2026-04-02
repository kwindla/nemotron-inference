#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include "nemotron/request_context.h"
#include "nemotron/reusable_state.h"

namespace nemotron {

std::size_t RequiredKvSnapshotBytes(const RequestExecutionContext& request_context);
std::size_t RequiredMambaSnapshotBytes(const RequestExecutionContext& request_context);

std::optional<ReusableStateDescriptor> SnapshotRequestState(
    ReusableStateArena& arena,
    const RequestExecutionContext& request_context,
    const std::string& label);

bool RestoreRequestState(
    const ReusableStateArena& arena,
    const ReusableStateDescriptor& descriptor,
    std::size_t token_count,
    RequestExecutionContext& request_context);

}  // namespace nemotron
