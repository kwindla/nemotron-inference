# Plan: Mamba Alignment and Gap Closure

Project directory: `./proj-2026-04-03-1816`

**Superseded status (2026-04-04):** Background analysis only.
Follow `proj-2026-04-04-0133/PLAN.md` for active execution sequencing and
verification requirements. This file is not the authoritative source for the
current BF16 multi-token and prefix-reuse alignment work.

## Goal

Close all accidental divergences in the Mamba path so this branch has one polished, production-quality Mamba implementation.

The current branch already has real Mamba functionality. The work now is not to invent a new Mamba design. The work is to make sure we are:

1. aligned with vLLM where vLLM is the right reference
2. intentionally divergent only where the project goal requires it
3. free of leftover debug, fallback, or shape-guessing code that does not belong in the final runtime

## Current State

The branch currently implements Mamba as a dedicated runtime slice in `runtime/src/backend/mamba_layer.cpp`, with:

- a multi-token prefill path using `RunMambaConvPrefill`, `RunMambaSsdPrefill`, and `RunMambaGatedGroupNorm`
- a single-token decode path that can use `RunFusedMambaDecode` when enabled
- optional host-side comparison scaffolding for fused decode validation
- per-request recurrent storage in `RequestExecutionContext` for both convolution state and SSM state
- exact-prefix snapshot/restore support through `runtime/src/api/state_snapshot.cpp` and `runtime/src/cache/prefix_cache.cpp`

The request context currently allocates one flat Mamba convolution buffer and one flat Mamba recurrent buffer per request in `runtime/src/backend/request_context.cpp`, and the prefix cache snapshots both buffers together as part of a reusable node.

## vLLM Reference Shape

vLLM’s Nemotron-H path in `third_party/vllm/vllm/model_executor/models/nemotron_h.py` uses `MambaMixer2` with Mamba-specific state shape and copy helpers from `third_party/vllm/vllm/model_executor/layers/mamba/mamba_utils.py`.

The relevant reference behaviors are:

- Mamba state is modeled as two reusable pieces: convolution state and temporal/SSM state
- prefix caching is state-aware, not just KV-aware
- chunked prefill and decode are handled through one Mamba mixer contract
- decode uses selective state update rather than a separate ad hoc request-local model
- the model class advertises `SupportsMambaPrefixCaching`

The important nuance is that vLLM’s hybrid Mamba prefix-caching support is still documented as work in progress in `third_party/vllm/docs/design/hybrid_kv_cache_manager.md`, so not every vLLM ideal is already a production baseline. We still use it as the reference shape for what polished should look like.

## Alignment Assessment

### Aligned

- The branch treats Mamba as a first-class model family instead of a hidden special case.
- The branch stores both conv state and SSM state, which is the right high-level reusable state for Nemotron Mamba.
- The branch already supports restoring cached Mamba state together with attention KV state when a conversation prefix is reused.
- The branch already has oracle coverage for Mamba layer behavior and state dumps in `testing/oracle/` and `testing/api/`.

### Partially Aligned

- Prefill and decode both exist, but they are split into different execution shapes in our runtime rather than one unified mixer contract.
- The fused decode path is optional and guarded by environment variables, which is appropriate for validation but not yet a polished final-state design.
- The current Mamba kernels are correctness-first and explicit, not yet the kind of single polished backend abstraction vLLM exposes.

### Divergent by Design

- The prefix cache uses exact-prefix reusable nodes with full Mamba state at selected boundaries, rather than vLLM’s block-hash KV-block cache model.
- The runtime keeps a custom request-local state model because the project goal is long-context multi-turn reuse, not generic vLLM architecture parity.

Those divergences are acceptable if they stay intentional. They should not be allowed to leak into unrelated fallback behavior or kernel shape drift.

## Accidental Gaps To Close

1. The decode path still carries compare/debug scaffolding in the hot implementation. That should be fenced off hard enough that the production path is obviously the production path.
2. The prefill path is functional, but it is still a straightforward sequential kernel pipeline rather than a polished chunk-aware Mamba execution contract.
3. The request-context state layout is correct for this branch, but the offset arithmetic is spread across the runner and kernels. That is workable, but it is a maintainability gap compared with a cleaner state contract.
4. The branch does not yet have strong enough tests for Mamba state restore across cached conversation boundaries and multi-token continuation after a restored prefix.
5. We do not yet have a direct performance story for Mamba alone that isolates prefill, decode, and restored-prefix behavior on the target 5090 setup.

## Non-Gaps

These should not be “fixed” just because they differ from vLLM:

- exact-prefix whole-state reuse in the prefix cache
- request-level reusable snapshots instead of dense Mamba block checkpointing
- keeping Mamba state coupled to the conversation boundary that actually matters for append-only chat

That is the right direction for this project.

## Concrete Tasks

1. Make the production Mamba path explicit.
   - Keep fused decode, prefill kernels, and debug comparison in separate clearly named paths.
   - Ensure the hot path does not depend on compare-mode logic or hidden side effects.

2. Tighten the prefill/decode contract.
   - Verify that `RunMambaConvPrefill` and `RunMambaSsdPrefill` preserve the correct recurrent state across continuation.
   - Check whether any Mamba-specific state handoff needs an explicit runner-level helper, rather than relying on repeated offset arithmetic.

3. Add missing parity tests.
   - Restore a cached Mamba prefix and continue generation.
   - Compare cold vs restored execution for the same prompt boundary.
   - Cover token-count > 1 prefill after a restored prefix.
   - Include at least one oracle that exercises both conv state and SSM state round-tripping.

4. Benchmark the Mamba slice in isolation.
   - Measure cold prefill, restored-prefix prefill, and single-token decode.
   - Compare fused decode against the fallback path using the same oracle fixture.
   - Record whether the current kernel split is still the right final split or just the current validation split.

5. Remove or quarantine accidental fallback leakage.
   - Keep any host-side reference comparisons behind explicit debug gates.
   - If a fallback remains for correctness, document exactly why it is still needed.

## Missing Evidence

- no direct comparison of our Mamba prefill/decode behavior against vLLM’s chunked Mamba behavior on the same 5090 hardware
- no benchmark showing where fused decode wins or loses after prefix restore
- no stress test covering repeated conversation-head commits with Mamba state reuse
- no evidence yet that the current kernel split is the final polished shape rather than a transitional implementation

## Sequencing

1. Lock correctness first.
   - cached-prefix restore
   - cold vs warm parity
   - decode equivalence

2. Lock the execution shape second.
   - make the hot path obvious
   - minimize or remove debug compare path leakage
   - decide whether the current prefill/decode split is final

3. Lock performance third.
   - benchmark the isolated Mamba slice
   - keep only the paths that earn their complexity budget

4. Only after that, consider whether any additional Mamba-specific backend consolidation is worth the complexity.
