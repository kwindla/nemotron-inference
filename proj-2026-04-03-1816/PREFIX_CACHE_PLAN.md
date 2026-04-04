# Plan: Prefix Cache and Reusable State Alignment Review

Project directory: `./proj-2026-04-03-1816`

## Goal

Close accidental divergences in prefix reuse, reusable-state ownership, and page allocation on the Nano / RTX 5090 branch.

The critical priority is to keep the custom prefix-cache design where it is intentional, while removing implementation choices that are only present because the current code is still immature.

## Current State

The current implementation already has the right high-level pieces:

- `PrefixCache` stores exact serialized prompt identities, conversation committed heads, optional prompt heads, and global roots in [runtime/src/cache/prefix_cache.cpp](/home/khkramer/src/nemotron-inference/runtime/src/cache/prefix_cache.cpp).
- `RunGreedyConversationTurn()` looks up a conversation-scoped cache entry first, restores request state, runs only the uncached tail when possible, then publishes the committed head in [runtime/src/api/single_token_forward_model.cpp](/home/khkramer/src/nemotron-inference/runtime/src/api/single_token_forward_model.cpp).
- `SnapshotRequestState()` and `RestoreRequestState()` serialize and rehydrate both attention KV and Mamba recurrent state in [runtime/src/api/state_snapshot.cpp](/home/khkramer/src/nemotron-inference/runtime/src/api/state_snapshot.cpp).
- `RequestExecutionContext` owns the live request-local KV pages and releases them on reset in [runtime/src/backend/request_context.cpp](/home/khkramer/src/nemotron-inference/runtime/src/backend/request_context.cpp).
- The v1 design document already says the cache should be exact-prefix based, keep committed heads and global roots, and include full Mamba state in [docs/v1_cache_architecture.md](/home/khkramer/src/nemotron-inference/docs/v1_cache_architecture.md).

This is functionally past the prototype stage. The remaining work is about making the ownership model, eviction behavior, and cache API polished instead of merely correct.

## vLLM Reference Shape

vLLM’s prefix cache is block-based and hash-indexed. The key lessons from [third_party/vllm/docs/design/prefix_caching.md](/home/khkramer/src/nemotron-inference/third_party/vllm/docs/design/prefix_caching.md) are:

- cache full blocks, not partial fragments
- use an explicit shared pool with clear allocation and eviction semantics
- keep cache hits cheap to discover and cheap to retain

For hybrid models, vLLM’s [third_party/vllm/docs/design/hybrid_kv_cache_manager.md](/home/khkramer/src/nemotron-inference/third_party/vllm/docs/design/hybrid_kv_cache_manager.md) is also important because it shows that Mamba prefix support is still a weak area in vLLM. That makes our Mamba-aware prefix cache a justified product divergence, but it does not justify sloppy ownership or slow cache internals.

## Alignment Summary

### Aligned

- Exact identity matching on tokenized prompts is the right correctness rule for this project.
- Conversation committed heads are the right fast path for append-only multi-turn chat.
- Global shared roots are the right reusable boundary for long system prompts and boilerplate.
- Full Mamba state reuse is intentional and necessary for correctness of the custom cache.
- The request-context snapshot/restore round-trip is already a good basis for regression testing.

### Intentional Divergences

- We cache full request state, not only attention KV. This is required for the custom prefix-caching goal.
- We use conversation-aware cache nodes, not just anonymous block hashes. This fits the workload and the branch goal.
- We allow boundary logits to be cached with a node. That is a convenience feature for exact-hit decode, not a vLLM requirement.

### Accidental Divergences to Close

- `prompt_head` is extra cache surface area that v1 docs explicitly defer. It is currently present in code and tests, but it is not part of the core committed-head design in [docs/v1_cache_architecture.md](/home/khkramer/src/nemotron-inference/docs/v1_cache_architecture.md). This should be fenced off, justified, or removed from the hot path.
- `ReusableStateArena` allocates each reusable snapshot with `cudaMalloc` / `cudaFree` and copies the snapshot payload into separate device storage in [runtime/src/cache/reusable_state.cpp](/home/khkramer/src/nemotron-inference/runtime/src/cache/reusable_state.cpp). That is correct, but it is not polished and it is a likely startup / fragmentation cost.
- Prefix-cache eviction is implemented as a full scan over nodes by `last_access_tick` in [runtime/src/cache/prefix_cache.cpp](/home/khkramer/src/nemotron-inference/runtime/src/cache/prefix_cache.cpp). That works at small scale but is not the kind of cache discipline we want to keep long term.
- Global-root lookup is also a linear scan over all nodes rather than a proper exact-prefix index. That is acceptable for a tiny cache, but it is not a polished implementation.
- The cache and the request-local page allocator are split into separate ownership systems. `RequestExecutionContext` owns live KV pages through `PagedKvCacheArena`, while the cache owns copied snapshots through `ReusableStateArena`. That split is understandable, but it adds memory movement and makes the lifecycle story more complex than it needs to be.
- Restore requires a caller-supplied `token_count` and trusts that it matches the descriptor. That is usable, but the snapshot format should make the restored prefix length harder to misuse.

## Adversarial Review

### 1. Prefix reuse semantics

Current code:

- lookup checks conversation committed head first, then optional prompt head, then global roots
- prefix matching is exact on serialized token identity
- conversation heads are refreshed on successful turn completion

Review:

- This is the right logical model for the branch.
- The accidental risk is the extra prompt-head path, which complicates reasoning about what a cache hit means.
- The exact-token identity contract should remain, but the cache index should become more direct if node count grows.

### 2. Reusable snapshot ownership

Current code:

- snapshot state is deep-copied into `ReusableStateArena`
- `ReusableStateArena` retains and releases descriptors explicitly
- cache nodes hold reusable-state descriptors, not live request pointers

Review:

- The ownership model is safe.
- The memory story is still not polished because it is copy-heavy and allocator-heavy.
- If we want this to feel like a finished subsystem rather than a correctness scaffold, this is the biggest area to tighten.

### 3. Attention KV ownership

Current code:

- request-local KV pages are owned by `RequestExecutionContext`
- page allocation and release are handled by `PagedKvCacheArena`
- snapshotting copies the live prefix pages into separate cache-owned storage

Review:

- The request-local ownership is clean.
- The cache does not currently share page ownership with live requests, so it cannot behave like a block pool.
- That divergence is not required by the product goal; it is an implementation choice that should be re-evaluated.

### 4. Mamba state reuse

Current code:

- Mamba conv state and SSM state are both captured and restored
- tests already verify round-trip equality for both tensors
- the v1 design explicitly requires full Mamba recurrent state

Review:

- This is the right intentional divergence from vLLM.
- The remaining question is not whether to keep it, but whether the reusable-state transport and allocation model should be simplified.

### 5. Page allocation and lifecycle

Current code:

- `PagedKvCacheArena` allocates per-layer page handles from a simple free list
- `RequestExecutionContext::ResetForNewRequest()` returns all page IDs to the arena
- page handles are not shared with the reusable cache

Review:

- The lifecycle is correct.
- The implementation is still too split between request-local paging and cache-owned snapshots.
- This is a likely source of accidental divergence from a polished vLLM-like ownership model.

## Concrete Implementation Tasks

1. Decide whether `prompt_head` stays in v1.
- If it stays, document it as an explicit non-hot-path feature and add tests that prove the committed-head path remains the default.
- If it does not stay, remove it from the main cache API and production lookup path.

2. Replace the cache’s linear eviction and lookup discipline with a more explicit index.
- Keep the exact-prefix semantics.
- Add a fast prefix index for global roots and any other repeated cache nodes.
- Keep eviction byte-based, but avoid whole-cache scans in the hot path.

3. Revisit reusable-state storage.
- Decide whether `ReusableStateArena` should remain a deep-copy snapshot store or become a pooled arena with preallocated slabs.
- Prefer the pooled design if the goal is a polished runtime rather than just correctness.
- Measure allocation latency, release latency, and fragmentation before choosing a final shape.

4. Reconcile page ownership with snapshot ownership.
- Either keep the current copy-on-snapshot model and document it clearly, or move toward shared page ownership with refcounts.
- Do not leave the system in an ambiguous middle state.

5. Harden restore semantics.
- Record prefix length in the snapshot descriptor if it is not already explicit enough for safe restore.
- Make restore reject mismatched length / layout cases with clearer diagnostics.

6. Keep Mamba state reuse, but simplify surrounding machinery.
- This part is intentional.
- The cleanup target is the transport and ownership model, not the feature itself.

## Missing Evidence

- No benchmark yet measures snapshot publish / restore overhead versus cold prefill on this branch.
- No benchmark yet measures allocator cost from repeated `cudaMalloc` / `cudaFree` in `ReusableStateArena`.
- No stress test yet covers a large number of conversation heads and global roots to validate the linear scan behavior.
- No concurrency test yet exercises competing updates to the same conversation ID.
- No evidence yet shows whether `prompt_head` is actually worth keeping beyond test coverage.
- No measurement yet compares the current cache ownership model against a pooled or shared-page alternative.

## Ordering Advice

1. Decide the fate of `prompt_head` first. It is a small surface area, but it affects the semantics of the whole cache.
2. Tighten the index / eviction structure next. That is the clearest accidental divergence from a polished cache implementation.
3. Benchmark and then refactor reusable-state allocation. This is the biggest likely startup and fragmentation risk.
4. Reconcile page ownership after the allocator decision, because the right answer depends on whether the cache stays copy-based or becomes pooled.
5. Only after the ownership model is settled should we spend time on additional cache heuristics or promotion policies.

## Success Criteria

- committed-head lookup remains the default multi-turn fast path
- Mamba state is restored correctly and intentionally, not as a side effect
- prefix-cache ownership is easy to explain without hand-waving
- allocator behavior is predictable and not dominated by per-snapshot CUDA allocations
- accidental divergence is reduced to the minimum needed to support custom prefix caching
