# Startup Residency Policy

## Purpose

This note records the current startup-residency policy for Nemotron on GB10.

It is intentionally separate from the main implementation plan because the question is operational:

- which weights and metadata should be resident in the hot path at process startup
- which families should be staged lazily or selectively
- which decisions are provisional until GB10 measurements are complete

Current service priorities remain:

1. low TTFT
2. decode speed

## Status

This is a provisional policy.

It is informed by:

- the current runtime architecture
- the checkpoint inventory from preflight
- the first uploaded-weight dense benchmark harness

It is not final because:

- the local benchmark stack is currently CUDA `13.0`, not the intended pinned `13.2+` stack
- only the uploaded FP32 dense path is benchmarked so far
- NVFP4 and BF16 dense execution paths are not yet validated on GB10

## Policy Shape

Residency decisions should be made by class, not by one global rule.

The main split is:

- always-hot weights that every request or decode step touches
- high-value shared weights that are used often but are not on every token path
- routed or long-tail weights whose aggregate size makes universal startup upload too expensive

## Provisional Residency Decisions

### 1. Keep Always-Hot Shared Roots Resident

These should be startup-resident by default:

- token embeddings
- shared system-prompt and cache metadata needed for fast prefix reuse
- page-table and reusable-node metadata once paged attention lands

Reason:

- they are used broadly across requests
- they directly affect TTFT
- their reuse is not workload-specific in the same way as routed experts

### 2. Keep Core Dense Families Eligible For Startup Residency

These dense families should be treated as first-class startup-residency candidates:

- attention projections
- Mamba projection weights
- shared expert projections

Reason:

- they sit on the always-executed backbone path
- they are touched even at small concurrency
- repeated host-to-device copies would directly damage TTFT and decode

Operationally:

- the runtime should retain an explicit uploaded-weight path for these families
- the final per-family decision should be made after rerunning the dense benchmark suite on the pinned GB10 stack

### 3. Do Not Assume Universal Startup Residency For Routed Experts

Routed expert weights should not be assumed startup-resident by default.

Reason:

- the aggregate expert set is too large
- only a subset is touched per token
- the right strategy is likely selective residency, a hot pool, or another staged policy rather than universal eager upload

This includes:

- MoE routed experts in the backbone
- expert-heavy MTP paths

### 4. Treat Scale Buffers As Separate Residency Objects

Scale buffers and related auxiliary metadata should be planned separately from packed weights.

Reason:

- NVFP4 contract validation is still open
- auxiliary placement may materially affect operator support and bandwidth
- we should not bury scale residency inside a generic packed-weight assumption

## What The Current Dense Benchmark Already Tells Us

The current local dense smoke artifact is:

- [artifacts/benchmarks/gb10_dense_gemm_smoke.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_dense_gemm_smoke.json)

The current local full-default artifact target is:

- [artifacts/benchmarks/gb10_dense_gemm_default_local_cuda13000.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_dense_gemm_default_local_cuda13000.json)
- [artifacts/benchmarks/gb10_dense_gemm_default_local_cuda13000.env.txt](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/gb10_dense_gemm_default_local_cuda13000.env.txt)

The current early signal is already useful:

- uploaded dense weights are the correct measurement basis for hot-path dense execution
- workspace size can materially affect at least some core dense cases on GB10

Examples from the current local full-default run:

- `4096 x 4096` attention core projection at `m=256` improved from about `0.910 ms` at `0` workspace to about `0.609 ms` at `4194304` bytes
- `18560 x 4096` Mamba input projection at `m=256` improved from about `6.683 ms` at `0` workspace to about `3.713 ms` at `16777216` bytes
- some smaller MTP expert shapes showed little or no benefit from extra workspace on this stack

That is enough to justify:

- keeping explicit uploaded-weight runtime objects
- measuring residency policy by family instead of reverting to per-call upload

## Policy Gate Before Freezing

Do not freeze the final startup-residency policy until all of these are true:

- the dense suite has been rerun on the pinned GB10 stack, ideally CUDA `13.2+`
- BF16 and NVFP4 dense paths have at least one representative measured implementation
- NVFP4 auxiliary scale placement is validated
- the model-level memory budget is checked against the chosen resident families

## Immediate Next Use

This note should be used to guide the next implementation steps:

1. rerun the full dense suite on the pinned stack
2. use those results to decide which dense families become startup-resident in the real mixed-precision runtime
3. carry the same class-based residency logic into paged attention and expert-path implementation
