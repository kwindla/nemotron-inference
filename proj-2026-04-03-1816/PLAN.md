# Plan: Adversarial vLLM Alignment and Gap Closure for Nano on RTX 5090

Project directory: `./proj-2026-04-03-1816`

**Superseded status (2026-04-04):** This document is now background analysis only.
The authoritative execution plan is `proj-2026-04-04-0133/PLAN.md`.
This file does not reflect the current `7e -> 7g -> 8 -> 9 -> 10` sequencing,
the latest `57/57` verification status, or the current BF16 multi-token and
local vLLM-on-RTX-5090 alignment findings.

## Program Judgment

Closing accidental divergences is now the critical priority.

The branch is far enough along that the main risk is no longer “missing core runtime pieces.” The main risk is carrying forward correctness scaffolding, fallback-heavy execution, and control-plane shortcuts that were useful during bring-up but do not belong in a polished implementation.

The guiding rule after adversarial review is:

1. keep the intentional divergence: exact-prefix whole-state reuse for KV + Mamba state
2. close the accidental divergences everywhere else unless there is measured evidence to keep them
3. align to vLLM at the level of component contracts, prepared-weight ownership, backend selection, and verification discipline
4. do **not** cargo-cult vLLM mechanisms that are unnecessary for this branch, especially its generic scheduler and block-hash cache model

## Scope

This project directory applies only to the `Nemotron Nano on RTX 5090` branch.

It does **not** claim readiness for:

- Nemotron 3 Super
- DGX Spark
- branch-independent backend decisions

## Completed Review Work

- [x] Created the top-level alignment project
- [x] Spawned sub-agents across all component bullets
- [x] Collected one sub-plan per component
- [x] Adversarially reviewed the sub-plans against the current code
- [x] Rewrote this top-level plan around only the gaps that survived review

## Sub-Plans

- `RUNTIME_ENGINE_PLAN.md`
- `PREFIX_CACHE_PLAN.md`
- `ATTENTION_PLAN.md`
- `MAMBA_PLAN.md`
- `MOE_SURFACE_PLAN.md`
- `MOE_ROUTING_PLAN.md`
- `NVFP4_WEIGHTS_PLAN.md`
- `LINEAR_QUANT_PLAN.md`
- `VERIFICATION_PLAN.md`

## Consolidated Review Verdicts

### 1. Runtime Engine

Keep:

- single-process C++ runtime
- exact-prefix conversation-turn control flow
- manifest-backed bootstrap

Close:

- monolithic bootstrap assembly
- implicit scheduling policy embedded in `SingleTokenForwardModel`
- heuristic layer-role inference in `ModelSchedule`
- broad `RequestExecutionContext` ownership that mixes persistent state and scratch

Review correction:

- We should adopt vLLM’s explicit request/engine boundaries, but we should **not** blindly copy vLLM’s queueing and engine architecture. This branch needs clearer contracts, not a forced scheduler transplant.

### 2. Prefix Cache and Reusable State

Keep:

- exact-token prefix identity
- committed-head reuse
- global roots
- full Mamba state reuse

Close:

- ambiguous `prompt_head` role in v1
- per-snapshot `cudaMalloc` / `cudaFree` churn in `ReusableStateArena`
- linear lookup and eviction structure
- restore semantics that rely too much on caller discipline

Review correction:

- Shared page ownership and block-pool semantics are not yet obviously the right replacement for snapshot copies. We should benchmark a pooled copy-based arena before committing to refcounted shared-page reuse.

### 3. Attention / Paged KV

Keep:

- paged KV layout
- exact-prefix state restore model
- simpler local allocator if it remains explicitly scoped
- Nano-specific decode specialization only if it keeps winning

Close:

- ad hoc backend selection
- per-call cuDNN plan construction
- env-gated backend policy leaking into hot paths
- weakly specified allocator/cache ABI

Review correction:

- The local allocator is a justified simplification for this branch. The problem is not “it is not vLLM’s block pool.” The problem is that its invariants and backend policy are under-specified.

### 4. Mamba

Keep:

- Mamba as a first-class layer family
- full conv + SSM state capture in the prefix cache
- separate prefill and decode kernels if measurement justifies them

Close:

- debug/compare scaffolding leaking into the hot implementation
- weak restore/continuation regression coverage
- under-specified runner/state contract around recurrent buffers

Review correction:

- We do not need a unified “vLLM-looking” mixer abstraction for aesthetic reasons alone. The real goal is one explicit production path, one explicit debug/fallback path, and stronger state-handling tests.

### 5. MoE Execution Surface

Keep:

- one MoE layer abstraction with backend selection underneath
- `UnifiedFusedBackend` as the primary target
- slower fallback only for unsupported or nonresident cases

Close:

- redundant single-token backend fragmentation
- duplicated `PrepareWeights()` logic
- host-routed fallback remaining too prominent in the code
- support predicates that encode historical boundaries instead of capability

Review correction:

- `FusedDecodeBackend` is the first backend that should have to re-earn its existence. It should not survive by inertia if unified fused or decode cuBLASLt already cover the same regime better.

### 6. MoE Routing Contract

Keep:

- `topk_ids` + `topk_weights` as the canonical contract
- grouped-top-k semantics
- expert-major compaction only as an adapter

Close:

- host reconstruction in prefill and fallback paths
- duplicated host/device routing semantics
- unclear tie-breaking / ordering guarantees

Review correction:

- The routing contract should be frozen in docs and tests before more backend cleanup. Otherwise later refactors will keep smuggling expert-major routing back into the API boundary.

### 7. NVFP4 Prepared Weights

Keep:

- load-time prepared weights
- monolithic expert residency
- backend-native scale swizzling

Close:

- on-demand upload in the intended fast path
- overlapping prepared-weight ownership types without clear boundaries
- unproven equivalence to vLLM prep/swap logic
- host fallback leakage in the GEMM bridge

Review correction:

- Removing on-demand upload from the fast path depends on making residency/prepared-weight support explicit at backend selection time. We should not remove the fallback before that contract is clean.

### 8. Non-MoE Linear / Quantization

Keep:

- load-time preparation for genuinely dense fp32 paths
- fallback/reference rails while native quantized paths are being tightened

Close:

- `ScaledFp8LinearOp` using host fp32 dequantization as the default execution model
- lack of a first-class quantization-method abstraction
- env/fallback policy carrying too much semantic weight

Review correction:

- This is the biggest remaining non-MoE accidental divergence. The branch does not have a polished story for generic FP8 linear yet; it has a correctness bridge.

### 9. Verification / Parity / Benchmarking

Keep:

- token-id oracle flow
- slice-level oracle tests
- exact-token vLLM parity path
- separate internal and external benchmark harnesses

Close:

- too many overlapping entry points
- compact oracle payloads that do not always localize drift well
- incomplete internal MoE-window benchmarking
- weak benchmark provenance guarantees

Review correction:

- Verification has to become the gatekeeper before we delete or demote execution paths. Otherwise we will simplify the runtime faster than we can prove it remains correct.

## Cross-Component Truths

These conclusions survived review across multiple sub-plans:

- The branch’s main intentional divergence is the cache model, not a license for unrelated divergence elsewhere.
- Host-side reconstruction and per-call runtime work are the most common accidental gaps across components.
- Backend selection exists in several places, but it is still too often encoded through env vars, fallback order, or incidental state instead of explicit capability and policy objects.
- Prepared-weight ownership is converging in MoE, but not yet fully clean; non-MoE quantized linear is further behind.
- Verification is broad, but not yet canonical. We have many good tests and scripts, but not a small set of trusted gates.

## Closure Sequence

### Phase 1: Freeze Contracts and Gates

Before more refactors, lock down the component contracts that later cleanup depends on.

- Canonical verification gates:
  - local correctness/oracle replay
  - exact-token vLLM parity
  - internal performance
  - external TTFT
- Canonical MoE routing contract:
  - `topk_ids`
  - `topk_weights`
  - ordering / tie semantics
  - adapter boundary for expert-major compaction
- Canonical prepared-weight contract:
  - what is prepared at load time
  - what is allowed as fallback only
  - what fast-path support requires
- Canonical attention backend policy:
  - explicit selection
  - explicit plan caching
  - no env var as long-term production policy
- Canonical request/cache boundary:
  - request lifecycle
  - persistent state vs scratch
  - exact-prefix cache semantics

### Phase 2: Collapse Hot-Path Fragmentation

After the contracts are explicit, simplify the live execution surface.

- MoE:
  - make unified fused the first-class path
  - demote/remove redundant decode-only backends if measurement supports it
  - keep host routing only as fallback
- Attention:
  - add a real backend selector
  - cache cuDNN plans
  - quarantine debug compare logic and env policy leakage
- Mamba:
  - make the production path explicit
  - fence debug paths away from the hot implementation
- NVFP4:
  - make fast-path support depend on prepared/resident weights
  - stop synthesizing prepared views on demand in the intended fast path
- Linear/quant:
  - stop treating fp32 dequantized weights as the default FP8 execution model

### Phase 3: Polish Ownership and Runtime Boundaries

Once the hot paths are simpler, clean up the component boundaries around them.

- split runtime bootstrap into validation, planning, and assembly phases
- replace heuristic layer-role inference with explicit loader/manifest roles
- split request-owned persistent state from execution scratch
- decide the fate of `prompt_head`
- replace `ReusableStateArena` allocation churn with a pooled design if measurement justifies it
- tighten page allocator and snapshot ABI invariants

### Phase 4: Re-measure, Prune, and Freeze

After the runtime is structurally cleaner, remove what no longer earns its cost.

- benchmark surviving MoE backends by token regime
- decide whether Nano-specific attention decode specialization stays
- decide whether any remaining Mamba split is justified
- delete dead or fallback-only-by-default paths
- pin the surviving intentional divergences in docs

## Immediate Worklist

This is the current recommended order of attack.

1. Canonicalize verification gates and benchmark provenance.
2. Freeze the MoE routing contract in docs and tests.
3. Make prepared-weight support explicit enough that fast paths can reject nonresident / nonprepared states cleanly.
4. Collapse MoE backend fragmentation around `UnifiedFusedBackend`.
5. Add explicit attention backend policy and cuDNN plan caching.
6. Replace the default FP8 bridge with a real native quantized-linear plan.
7. Extract request lifecycle and bootstrap boundaries after the execution surfaces above are stable.
8. Revisit cache allocator and page/snapshot ownership only after real publish/restore measurements exist.

## Explicit Non-Goals

Do **not** spend the next milestone on these unless a dependency forces it:

- copying vLLM’s full engine/scheduler architecture into this branch
- replacing exact-prefix whole-state reuse with block-hash KV caching
- inventing new one-off kernels before the accidental contract gaps are closed
- polishing fallback paths that should instead be demoted or deleted

## Success Criteria

This alignment project is complete when all of the following are true:

- the only major intentional divergence from vLLM is the exact-prefix whole-state cache model
- hot-path execution does not depend on host reconstruction, just-in-time preparation, or env-gated policy
- each major component has one obvious production path and clearly fenced fallback/debug paths
- verification consists of a small, trusted set of canonical gates with strong provenance
- the remaining code shape reads like a finished runtime, not a bring-up runtime that happens to pass tests
