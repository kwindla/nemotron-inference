# Plan: MoE Routing Contract Alignment

Project directory: `./proj-2026-04-03-1816`

## Goal

Make the MoE routing contract fully intentional, vLLM-aligned where appropriate, and free of accidental host-side leakage in the fast path.

The critical priority is to close accidental divergences now. The routing contract is already close in shape, but it still has multiple execution paths that reconstruct or compact routing on the host when the design should remain tensor-native.

## Scope

This review covers only MoE routing and the tensor contract between routing and expert execution:

- grouped top-k semantics
- routing tensor shapes and ownership
- host/device boundaries
- expert-major compaction paths
- debug and fallback leakage

It does not cover the broader MoE backend choice, weight layout, attention, Mamba, or cache control plane except where they affect routing ownership.

## Current State

The branch already has a real device routing path:

- `RunDeviceExpertSelection(...)` produces `selected_indices` and `selected_weights` on device in [fused_moe_decode.cu](/home/khkramer/src/nemotron-inference/runtime/src/backend/fused_moe_decode.cu:438).
- `ExpertLayerSlice::Run()` uses that device output to drive backend selection in [expert_layer.cpp](/home/khkramer/src/nemotron-inference/runtime/src/backend/expert_layer.cpp:3778).
- The unified backend surface passes the top-k tensors into backends instead of rebuilding routing from scratch in the common case in [expert_layer.cpp](/home/khkramer/src/nemotron-inference/runtime/src/backend/expert_layer.cpp:2441).

The current routing data structures are:

- per-token `topk_ids` and `topk_weights`
- expert-major routing tables in `DeviceExpertRouting`
- `active_expert_count` and `active_expert_ids` for dispatch compaction

That is the right general direction, but not yet a clean final contract.

## vLLM Reference Shape

vLLM’s grouped-top-k router returns tensor outputs directly:

- `topk_weights` as `float32`
- `topk_ids` as `int32`

The router logic in [grouped_topk_router.py](/home/khkramer/src/nemotron-inference/third_party/vllm/vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:29) computes:

- score activation from logits
- optional correction-bias adjustment for expert selection
- grouped top-k selection
- top-k expert selection
- optional renormalization
- routed scaling

The model wiring in [nemotron_h.py](/home/khkramer/src/nemotron-inference/third_party/vllm/vllm/model_executor/models/nemotron_h.py:127) then feeds those routing tensors directly into `SharedFusedMoE`.

That is the target shape: routing should emit tensors, and execution should consume tensors.

## Alignment

### What matches vLLM already

- We already keep routing on device in the fast path.
- We already use grouped top-k semantics for Nemotron H style routing.
- We already keep the routing interface as `ids + weights`, which matches the tensor-level contract used by vLLM.
- We already have a backend abstraction so routing is not hard-wired to one execution kernel.

### What is intentionally different

- Host-visible debug and trace paths are allowed.
- Fallback staging is allowed when the routing/expert residency contract cannot be satisfied.
- Expert-major compaction is allowed as a backend adapter if a backend truly needs it.

Those are acceptable only when they remain clearly outside the main fast path.

## Divergences

### Intentional divergences

- `RunFusedMoePrefill()` uses expert-major compaction and active-expert dispatch because the current backend expects it in [fused_moe_prefill.cu](/home/khkramer/src/nemotron-inference/runtime/src/backend/fused_moe_prefill.cu:117).
- The non-resident paths in [expert_layer.cpp](/home/khkramer/src/nemotron-inference/runtime/src/backend/expert_layer.cpp:2313) and [expert_layer.cpp](/home/khkramer/src/nemotron-inference/runtime/src/backend/expert_layer.cpp:3812) fall back to host staging and host-side routing reconstruction when residency is not available.
- Debug comparison paths intentionally copy routing data back to host for inspection.

These are acceptable only as fallback or debug scaffolding.

### Accidental divergences

- The prefill fast path still copies `active_expert_count` and `active_expert_ids` to host in [fused_moe_prefill.cu](/home/khkramer/src/nemotron-inference/runtime/src/backend/fused_moe_prefill.cu:220), even though the routing output itself already exists on device.
- The prefill path still reconstructs expert-major routing state on host with [BuildExpertRoutingTable](/home/khkramer/src/nemotron-inference/runtime/src/backend/expert_layer.cpp:1170), which is a contract leak unless the backend truly requires that representation.
- The direct decode fallback path still copies `topk_ids` to host in [expert_layer.cpp](/home/khkramer/src/nemotron-inference/runtime/src/backend/expert_layer.cpp:2087), then stages weights per expert when residency is missing.
- The routing selection implementation is duplicated across host and device code paths in [expert_layer.cpp](/home/khkramer/src/nemotron-inference/runtime/src/backend/expert_layer.cpp:441) and [fused_moe_decode.cu](/home/khkramer/src/nemotron-inference/runtime/src/backend/fused_moe_decode.cu:57), which increases the risk of semantic drift.
- Tie-breaking / deterministic ordering is not obviously pinned tightly enough to vLLM’s `sorted` behavior in grouped-top-k routing.

## What Needs To Be Closed

1. The main fast path should consume grouped-top-k tensors directly, not an expert-major host reconstruction.
2. Expert-major compaction should exist only as a backend adapter or fallback, not as the default routing contract.
3. The host/device routing semantics should be verified against vLLM for sigmoid + correction-bias + renormalization + routed-scaling behavior.
4. Deterministic ordering and tie handling should be pinned down so host and device implementations cannot diverge silently.
5. Any host copies in the hot path must be justified as unavoidable backend requirements, not historical leftovers.

## Concrete Implementation Tasks

- Define the canonical routing contract explicitly in one place, including:
  - tensor shapes for `topk_ids` and `topk_weights`
  - whether the contract is token-major or expert-major at each stage
  - ordering guarantees for selected experts
  - tie-breaking rules
  - renormalization and scaling semantics
- Remove host reconstruction from the fast prefill path unless the backend truly requires expert-major tables.
- If expert-major dispatch remains necessary, isolate it behind a backend-specific adapter and make the tensor contract stay canonical at the API boundary.
- Replace the current duplicate routing logic with one shared implementation or a single source-of-truth device kernel plus a validation-only host mirror.
- Add a parity test that compares our routing output against vLLM grouped-top-k for Nemotron Nano inputs and exercises:
  - grouped top-k
  - `e_score_correction_bias`
  - renormalization
  - routed scaling
  - deterministic tie cases
- Add a regression test that fails if the main MoE fast path starts copying routing state to host outside debug or fallback modes.
- Clarify whether `active_expert_count` / `active_expert_ids` are part of the long-term API or only a temporary dispatch adapter.

## Missing Evidence

- No direct test currently proves our grouped-top-k output matches vLLM on tie-sensitive cases.
- No direct test currently proves host and device selection paths stay semantically identical across all supported routing modes.
- No evidence yet that expert-major host compaction is still needed in the polished fast path once the backend surface is fully tensor-native.
- No benchmark evidence yet shows that routing compaction itself is worth optimizing independently of the backend that consumes it.

## Sequencing

1. Freeze the routing contract in tests and docs first.
2. Remove accidental host leakage from the fast path next.
3. Keep fallback staging only where residency or backend limitations force it.
4. Once the contract is clean, optimize routing-side performance only if profiling shows it is still a bottleneck.

## Review Notes

The important distinction is simple:

- `device top-k` and `ids + weights` is the contract we want.
- `expert-major compaction` is only an adapter, not the contract.

If that boundary stays blurred, routing will continue to drift away from the polished vLLM-like shape even if the kernels themselves are correct.
