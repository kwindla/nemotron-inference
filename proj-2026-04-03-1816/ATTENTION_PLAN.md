# Plan: Attention / Paged-KV Alignment Review

Project directory: `./proj-2026-04-03-1816`

**Superseded status (2026-04-04):** Background analysis only.
Follow `proj-2026-04-04-0133/PLAN.md` for active execution sequencing and
verification requirements. This file is not the authoritative source for the
current BF16 multi-token or local vLLM-on-RTX-5090 alignment work.

## Goal

Close all accidental divergences in the attention and paged-KV subsystem, while preserving only the divergences that are intentional for exact-prefix caching on Nano.

The critical priority now is not new optimization work. The critical priority is:

1. prove which gaps vs vLLM are still real
2. classify each gap as intentional or accidental
3. close the accidental gaps first
4. leave only the minimum intentional divergence required by the custom prefix-cache design

## Current State

- The current runtime has a real page-based KV cache, not a mock. Page geometry is explicit in [`paged_kv_cache.cpp`](/home/khkramer/src/nemotron-inference/runtime/src/attention/paged_kv_cache.cpp) and the request context owns per-layer page handles in [`request_context.cpp`](/home/khkramer/src/nemotron-inference/runtime/src/backend/request_context.cpp).
- The attention layer already does the right high-level thing: project Q/K/V, scatter K/V into the paged cache, run paged attention, then project out and add the residual in [`attention_layer.cpp`](/home/khkramer/src/nemotron-inference/runtime/src/backend/attention_layer.cpp).
- There are three execution paths today: cuDNN when a handle exists, a device fallback kernel, and a Nano-specific decode kernel behind an env gate in [`attention_device_fallback.cu`](/home/khkramer/src/nemotron-inference/runtime/src/backend/attention_device_fallback.cu).
- Cache-aware control flow is wired into the model path. Conversation turns restore prefix state, run only the uncached tail when needed, and publish committed-head snapshots in [`single_token_forward_model.cpp`](/home/khkramer/src/nemotron-inference/runtime/src/api/single_token_forward_model.cpp) and [`state_snapshot.cpp`](/home/khkramer/src/nemotron-inference/runtime/src/api/state_snapshot.cpp).

## Alignment With vLLM

- The page/block idea is aligned. vLLM’s KV cache interface is block based, with cache specs and page-size accounting in [`kv_cache_interface.py`](/home/khkramer/src/nemotron-inference/third_party/vllm/vllm/v1/kv_cache_interface.py).
- The backend-selection concept is aligned. vLLM picks an attention backend through an explicit selector and validates configuration up front in [`selector.py`](/home/khkramer/src/nemotron-inference/third_party/vllm/vllm/v1/attention/selector.py) and [`attention.py`](/home/khkramer/src/nemotron-inference/third_party/vllm/vllm/model_executor/layers/attention/attention.py).
- The cuDNN path is directionally aligned with vLLM’s backend model because it is a real backend implementation, not a one-off kernel hack. The config/build/execute split in [`cudnn_paged_attention.cpp`](/home/khkramer/src/nemotron-inference/runtime/src/backend/cudnn_paged_attention.cpp) is the right shape.
- The cache/control-plane integration is aligned with the project goal, not with vLLM’s exact mechanism. The branch already has a coherent state-restoration story for attention KV plus Mamba state in [`state_snapshot.cpp`](/home/khkramer/src/nemotron-inference/runtime/src/api/state_snapshot.cpp).

## Divergences

### Intentional

- Exact-prefix whole-state reuse is intentionally different from vLLM’s hash-based block reuse in [`prefix_caching.md`](/home/khkramer/src/nemotron-inference/third_party/vllm/docs/design/prefix_caching.md). We need precise reuse of KV plus recurrent state, so this is a justified divergence if it stays disciplined.
- The request-local page allocator is intentionally simpler than vLLM’s global block pool if the only supported reuse model is snapshot restore for a single request context.
- The Nano-specific decode kernel is intentionally shape-specific if it remains a fenced fast path for the one model shape that fits on this branch.

### Accidental

- Backend choice is still too ad hoc. vLLM uses a validated backend selector; this branch chooses cuDNN if the handle exists, otherwise falls into env-gated device code. That is not a polished backend policy.
- `CudnnPagedAttentionPlan::Create` is built in the hot path on every attention call instead of being cached or prevalidated once per layer/config. That is accidental churn and should be removed if the plan is stable across calls.
- `RequestExecutionContext::EnsureAttentionTokens()` reserves enough pages for the full current sequence length on every step, which couples live state growth to a request-local capacity model instead of a clean attention-cache policy. This is workable, but it is not yet a polished allocator design.
- `ResetForNewRequest()` releases pages back to the local arena, but there is still no shared cache manager or explicit eviction policy for attention pages. That is fine only if we deliberately keep the cache scoped to per-request snapshotting.
- The device fallback is a correct safety net, but the env gates around `ProductionAttentionEnabled`, `DecodeScratchEnabled`, and `ScalarAttentionFallbackForced` are policy leakage. Those flags should not be the long-term way to choose production code paths.
- The host comparison harness in [`attention_layer.cpp`](/home/khkramer/src/nemotron-inference/runtime/src/backend/attention_layer.cpp) is useful for debugging, but it must remain completely out of the production decision path.

## Component Review

### Paged KV Layout

- The layout is structurally sound: a page has a fixed token count, page handles carry page id and layer id, and the cache tensor shape is explicit.
- The missing polish is around invariants and ABI clarity. The snapshot code treats `attention_total_pages` and the page geometry as an implicit ABI for copy/restore, but that contract is not yet strongly tested.
- The page layout should be treated as part of the cache ABI, not just an implementation detail.

### Allocator Design

- The allocator is a request-local free list, not vLLM’s shared pool with block hashing, ref counts, and LRU-style eviction.
- That is acceptable only if we continue to treat prefix caching as exact-state replay rather than generic block reuse.
- The accidental gap is that the current allocator shape leaves lifetime and reuse policy under-specified. It should have explicit invariants for allocation, release, restore, and reset.

### Attention Backend Selection

- vLLM’s model layer asks for a backend through a capability-driven selector and validates the choice up front in [`attention.py`](/home/khkramer/src/nemotron-inference/third_party/vllm/vllm/model_executor/layers/attention/attention.py).
- This branch should move toward the same clarity. The backend decision needs to be explicit, deterministic, and benchmarkable.
- The current branching logic in [`attention_layer.cpp`](/home/khkramer/src/nemotron-inference/runtime/src/backend/attention_layer.cpp) should be replaced or wrapped by a real backend policy object, not spread across env vars and ad hoc `if` statements.

### cuDNN Path

- The cuDNN path is the most promising production backend for this branch because it already consumes the paged layout and a batch plan.
- The important missing piece is plan reuse and measurement. Right now the path is structurally correct but still too expensive to treat as a polished backend until we know the construction overhead and steady-state latency.
- If cuDNN wins only on some shapes, that should be encoded in selection logic, not discovered implicitly via the presence of a handle.

### Device Fallback Path

- The general device fallback is a necessary correctness backstop.
- The Nano decode kernel is a legitimate specialization, but it must stay subordinate to the same backend policy and be selected because it wins, not because an env var happened to be set.
- The fallback path should remain available for unsupported shapes, failed backend init, and debug verification. It should not remain a permanent policy layer.

### Control-Plane Integration With Caching

- The current control plane is closer to the project goal than vLLM’s generic block cache because it can restore and re-publish exact state.
- The accidental gap is that the page allocator and the snapshot system are only loosely tied together through request context ownership. That is enough for bring-up, but not enough for a polished implementation.
- The cache ABI needs explicit tests for exact hits, tail reuse, restore correctness, page release, and replay after partial-prefix matches.

## Prioritized Tasks

1. Introduce an explicit attention backend policy object with `Supports(...)` / `Select(...)` semantics, modeled after vLLM’s validate-and-select flow.
2. Cache or prebuild cuDNN attention plans per stable layer/config shape instead of recreating them on every call.
3. Tighten allocator invariants so allocation, restore, reset, and release are validated as a single contract.
4. Remove env-var policy leakage from hot attention code paths and replace it with benchmarked backend selection.
5. Keep the Nano-specific decode kernel only if it remains measurably better than the general device path and cuDNN for the relevant shapes.
6. Add explicit tests for snapshot restore, exact cache hits, page release, and invalid page-table / geometry rejection.

## Missing Measurements

- cuDNN plan build cost versus steady-state execution cost.
- cuDNN versus device fallback latency for `1`, `32`, `64`, `128`, and `256` token runs.
- Decode-kernel versus general-device versus cuDNN comparison on the exact Nano shapes this branch targets.
- Cost of page reservation and reset in conversation-turn reuse, including any allocator churn hidden by the current request-local model.
- Correctness on restored prefixes that exercise partial-page boundaries and committed-head replay.

## Sequencing Advice

- First make backend selection explicit and cache the cuDNN plan path.
- Then harden allocator invariants and page-table tests so the cache ABI is visible and stable.
- Then measure cuDNN, the Nano decode kernel, and the general fallback against the same workloads.
- Only after those measurements should any backend-specific cleanup or deletion happen.
