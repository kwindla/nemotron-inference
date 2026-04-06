# Plan: Single-Path TTFT Reduction After Correctness Lock

Project directory: `./proj-2026-04-05-1704`

## Status

This plan supersedes the earlier routed-MoE optimization plan in
`proj-2026-04-05-0445`. Since that plan was written:

- the grouped CUTLASS MoE experiment was fully reverted
- the SM120 runtime was collapsed back to a single production forward path
- correctness was revalidated on the active Nano manifest
- fresh TTFT profiling was captured for short, medium, and long prefill/tail
  regimes

This plan now tracks the current, correct `SM120` baseline on commit
`b431e21`. The original frozen reference point remains the saved April 5
artifacts, but implementation work should use the latest committed TTFT and
correctness state on this branch. The optimization job remains the same:
reduce TTFT without reintroducing fallbacks, backend ladders, env-gated
alternate paths, or size-based split dispatch.

This plan is intentionally written as the implementation source of truth. It
should be possible to execute the work from this file alone without having to
recover decisions from chat history.

External code review update:

- On this machine, the relevant vLLM MoE prior art is the FlashInfer
  CUTLASS-style NVFP4 path, not its runtime backend oracle.
- On this machine, the relevant TRT-LLM MoE prior art is `CutlassFusedMoE`,
  not `TRTLLMGenFusedMoE`, because TRTLLM-Gen explicitly does not support
  `SM120`.

## Goal

Reduce TTFT for the real Nano serving path on local RTX 5090 hardware across
all three prompt/tail regimes that matter:

- short: `4` tokens
- medium: `128` tokens
- long: `4096` tokens

The required benchmark set is:

- cold prefill: `prefix4`, `prefix128`, `prefix4096`
- cached prefill with short tail: `tail4`
- cached prefill with medium tail: `tail128`
- cached prefill with long tail: `tail4096`

The production target remains:

- Nemotron 3 Nano NVFP4
- consumer Blackwell `SM120`
- `build-sm120-relwithdebinfo`
- one serving implementation path per operation

Explicit exclusions:

- do not use `SM75` builds for this work
- do not use generic `build/` results as performance evidence
- do not keep alternate implementations in-tree for A/B comparison; use git for
  that

## Non-Negotiable Contract

### 1. One implementation path per operation

For the active `SM120` forward pass, each operation class gets exactly one
production implementation:

- embedding / final norm / `lm_head`
- attention
- Mamba
- MoE routing and MoE execution
- projection GEMMs

This does **not** mean every operation must be hand-written CUDA today. It
means there is only one active implementation in the runtime for each
operation on the target stack.

### 2. No fallback code in the production path

Do not add or restore:

- backend registries
- runtime backend selection ladders
- env-var gates for alternate kernels
- size-based split dispatch
- host-routing adapters
- "temporary" comparison paths left in tree

If an operation is replaced, the old implementation must be deleted in the
same change or remain only on a separate git branch.

The rule for comparison work is simple:

- compare alternatives in git history or on separate branches
- do not land both alternatives in the same runtime tree

### 3. Prior art is reference material, not a second runtime

Use vLLM and TRT-LLM for:

- algorithmic structure
- memory/layout ideas
- kernel tiling ideas
- contract comparison

Do **not** mirror their internal "multiple backend" runtime structure in this
repo.

### 4. Every hot path is judged on stalls, transfers, and launch count

For every material optimization step, evaluate the hot path against all three
of these criteria, not just GPU math time:

- avoid host-visible pipeline stalls
- avoid HtoD and DtoH transfers in the runtime hot path
- fuse operations or otherwise reduce kernel-launch count where practical

Treat these as implementation rules:

- any hot-path `cudaMemcpy` / `cudaMemcpyAsync` must be either removed or
  explicitly justified by the API contract
- any explicit `cudaStreamSynchronize` / `cudaDeviceSynchronize` must be either
  removed or explicitly justified by the API contract
- any long serial kernel chain must be treated as suspicious until we explain
  why the boundaries must remain

Correctness and the single-path rule still dominate. Do not satisfy these
criteria by adding alternate runtime implementations.

## Current Baseline

Reference artifacts:

- `artifacts/profiles/ttft_prefill_20260405/cold_prefix_4_128_4096.stdout.txt`
- `artifacts/profiles/ttft_prefill_20260405/cached_tail4_prefix_4_128_4096.stdout.txt`
- `artifacts/profiles/ttft_prefill_20260405/cached_tail128_prefix_4_128_4096.stdout.txt`
- `artifacts/profiles/ttft_prefill_20260405/cached_tail4096_prefix_4_128_4096.stdout.txt`
- `artifacts/profiles/ttft_prefill_20260405/prefill_trace_cases.stdout.txt`
- `artifacts/profiles/ttft_prefill_20260405/nsys_cold_prefill_prefix128.nsys-rep`
- `artifacts/profiles/ttft_prefill_20260405/nsys_cached_committed_head_prefix4096_tail4096.nsys-rep`

The required benchmark matrix for every material optimization step is the full
cross-product below:

| Prefix | Tail |
|--------|------|
| `4` | cold / `4` / `128` / `4096` |
| `128` | cold / `4` / `128` / `4096` |
| `4096` | cold / `4` / `128` / `4096` |

### Cold TTFT medians

| Case | Median |
|------|--------|
| `cold_prefill_prefix4` | `59.707 ms` |
| `cold_prefill_prefix128` | `233.231 ms` |
| `cold_prefill_prefix4096` | `1340.429 ms` |

### Current committed working baseline on `b431e21`

Focused post-optimization reruns on the current branch:

| Case | Median |
|------|--------|
| `cold_prefill_prefix4` | `40.415 ms` |
| `cold_prefill_prefix128` | `142.156 ms` |
| `cold_prefill_prefix4096` | `1191.797 ms` |

Representative cached committed-head hot-prefix reruns:

| Case | Hot TTFT |
|------|----------|
| `prefix4_tail4` | `35.281 ms` |
| `prefix128_tail4` | `40.655 ms` |
| `prefix4096_tail4` | `83.871 ms` |

### Hot-prefix TTFT medians

Representative cached committed-head results:

| Case | Hot TTFT | Tail prefill | Restore |
|------|----------|--------------|---------|
| `prefix4_tail4` | `52.876 ms` | `37.634 ms` | `0.341 ms` |
| `prefix128_tail4` | `59.509 ms` | `43.515 ms` | `0.343 ms` |
| `prefix4096_tail4` | `104.236 ms` | `67.433 ms` | `0.621 ms` |
| `prefix4_tail128` | `228.755 ms` | `212.854 ms` | `0.346 ms` |
| `prefix128_tail128` | `236.729 ms` | `219.923 ms` | `0.352 ms` |
| `prefix4096_tail128` | `288.061 ms` | `250.513 ms` | `0.626 ms` |
| `prefix4_tail4096` | `1335.909 ms` | `1299.224 ms` | `0.567 ms` |
| `prefix128_tail4096` | `1350.947 ms` | `1313.465 ms` | `0.569 ms` |
| `prefix4096_tail4096` | `1684.394 ms` | `1626.069 ms` | `0.846 ms` |

### Root-cause profile

Per-layer prefill tracing shows:

- `cold_prefill_prefix128`: expert prefill dominates at about `193.7 ms` of
  `216.1 ms`; attention is only about `2.0 ms`
- `cold_prefill_prefix4096`: Mamba dominates at about `732.9 ms` of
  `1301.7 ms`; expert prefill is about `396.2 ms`; attention is about
  `172.6 ms`
- `cached_committed_head_prefix4096_tail4096`: Mamba stays about `732.7 ms`,
  expert prefill stays about `396.1 ms`, and attention grows to about
  `491.6 ms` because the total active sequence is now `8192`

`nsys` adds two important details:

- `cold_prefill_prefix128` is still spending substantial time in NVFP4 GEMM,
  runtime packing/scaling, `cudaMemcpy`, and `cudaStreamSynchronize`
- `cached_committed_head_prefix4096_tail4096` is dominated by long-running
  Mamba and attention kernels rather than cache restore overhead

More concrete `nsys` findings that should drive the work order:

- `cold_prefill_prefix128`: the expert-path quantization/packing stack is a
  first-class target, not a side note:
  - `PackAndSwizzleRowMajorFp32ToNvfp4Kernel`: about `62.3 ms`
  - `ComputeGlobalMaxAbsKernel`: about `23.1 ms`
  - `MultiplyTensorScalesKernel`: about `22.1 ms`
  - `WriteTensorScaleKernel`: about `17.9 ms`
  - total quantization/packing/scaling overhead: about `125 ms`, or about
    `24.5%` of GPU kernel time in the profiled run
- `cold_prefill_prefix128`: `cudaMemcpy` is the largest API-time bucket at
  about `53.3%`, so the plan must explicitly separate one-time/model-load
  copies from per-layer hot-path traffic before optimizing blindly
- the large HtoD bucket is misleading if treated as a hot-path headline by
  itself: much of that traffic is model-load or setup cost that appears in both
  short and long profiles; the hot-path runtime burden is more clearly launch
  count, allocation churn, dynamic scale recomputation, and synchronization
- `cold_prefill_prefix128`: DtoH traffic is small in absolute time, but the
  remaining routed-MoE `expert_offsets` copy is still architecturally
  unacceptable because it keeps host control in the middle of expert execution
- `cold_prefill_prefix128`: the current projection GEMM decision is highly
  load-bearing because the active NVFP4 GEMM kernel is still about `49.1%` of
  GPU kernel time in the representative `nsys` run
- `cold_prefill_prefix128`: expert prefill is not "mostly GPU math". The wall
  time still includes substantial CPU/API overhead from kernel launches,
  `cudaMalloc` / `cudaFree`, and host-visible synchronization. Those must be
  treated as first-class bottlenecks alongside kernel time.
- `cached_committed_head_prefix4096_tail4096`: `cudaStreamSynchronize` is
  about `76.8%` of API time, which mostly means the host is waiting on long
  kernels, but Steps 3 and 4 should still explicitly audit for unnecessary
  mid-layer sync points

Scaling reference:

- The `token_count=8192` trace is useful reference data, even though it is not
  a separate benchmark case. Mamba and expert prefill both scale close to
  linearly versus `4096`, while attention grows more sharply with total
  context. That means the current evidence does **not** suggest pathological
  superlinear scaling in Mamba or expert prefill.

Restore time is negligible in every measured regime. First-token decode is not
the main problem either. TTFT is still a prefill problem.

## Current Operation Contracts

This is the active production contract that optimization work must preserve
unless an operation is replaced end-to-end.

| Operation | Current implementation contract | Notes |
|-----------|---------------------------------|-------|
| embedding / final norm / `lm_head` | current native forward model path | keep single path |
| attention | native kernels in `attention_native_kernels.cu` | no alternate backend ladder |
| Mamba | native prefill/decode kernels | one path only |
| MoE routing / execution | current native routing + expert execution path | no split dispatch |
| projection GEMMs | current `UploadedLinearOp` / `ScaledFp8LinearOp` / `nvfp4_gemm_runner` stack | model-bound at load time, not a runtime fallback ladder |

Important clarification: the grouped CUTLASS MoE experiment is gone, but
CUTLASS-backed / cuBLASLt-backed GEMM kernels still exist inside the active
single projection path. That is visible in:

- `runtime/src/backend/linear_op.cpp`
- `runtime/src/backend/nvfp4_gemm_runner.cpp`

So before changing projection math again, decide explicitly whether:

1. the current library-backed GEMM path remains the long-term production path,
   or
2. it will be replaced completely by a native fused kernel path

Do **not** allow a mixed transition state to land in the main branch.

This architectural decision is a hard blocker for implementation order. Do not
start another projection-path rewrite until Step 1 records the answer in
writing.

External alignment note:

- vLLM's NVFP4 MoE path explicitly runs a backend-selection oracle with
  shape-specific fallbacks.
- TRT-LLM's Torch MoE stack also exposes multiple backend families and
  fallback/selection behavior.

Those systems are useful references for support matrices and kernel contracts,
but our main branch must not copy their multi-backend runtime structure.

## Cross-Cutting Audit Targets

This section translates the stall/transfer/fusion rules into concrete hot-path
targets in the current tree.

### 1. NVFP4 pack/scaling

Current concerns inside `runtime/src/backend/device_nvfp4_matrix.cu`:

- fixed-scale packing still performs a hot-path HtoD tensor-scale copy
- dynamic packing still runs a serial chain of reduction, tensor-scale write,
  scale-buffer clear, pack, and later tensor-scale multiply before GEMM
- the pack path is still optimized primarily around the current cuBLASLt NVFP4
  contract, not around minimum launches

Immediate questions:

- can fixed tensor scales remain device-resident end-to-end
- can the tensor-scale multiply path be removed or folded when host/device
  scale ownership is already known
- can `matmul_block_scales` zeroing be removed or folded into a kernel that
  already walks the same rows/blocks
- can `WriteTensorScaleKernel<<<1, 1>>>` be removed entirely by folding tensor
  scale derivation into a kernel that is already launched for the same pack
  operation
- can the pack/scaling launch chain be shortened without creating a second
  projection path

### 2. MoE prefill control flow

Current concerns inside `runtime/src/backend/fused_moe_prefill.cu`:

- per-run DtoH copy of `expert_offsets`
- host-side loop over active experts
- per-expert plan construction and serial launch structure
- hot-path `cudaMalloc` / `cudaFree` churn caused by shape-variant resource and
  plan setup
- repeated per-expert dynamic quantization / packing on the routed path

This is not acceptable as the long-term shape of the hot path, even if the
math kernels remain the same. The goal is to reduce stalls and transfers
without restoring split-dispatch or any alternate MoE backend.

### 3. MoE launch chain around GEMMs

Current concerns inside `runtime/src/backend/fused_moe_prefill.cu` and
`runtime/src/backend/expert_layer.cpp`:

- routed path is still `gather -> pack -> GEMM -> activation -> pack -> GEMM -> scatter`
- shared path is still `pack -> GEMM -> activation -> pack -> GEMM -> residual add`
- decode follows the same general shape

Some boundaries may be required, but every one of them should now be assumed
expensive until proven otherwise.

The key architectural suspicion is no longer "is gather expensive" but:

- are we quantizing too late and too often because the routed path is built
  around per-expert host-controlled execution rather than a single device-side
  execution contract
- can the routed path move to one-pass activation quantization before dispatch,
  with grouped device-side execution metadata, without restoring the rejected
  grouped-CUTLASS split-dispatch design

### 4. Token selection and logits ownership

Current concerns inside `runtime/src/api/single_token_forward_model.cpp`:

- prompt logits are still copied to host for argmax
- decode-step logits are still copied to host for argmax
- top-level "ready on return" syncs are still part of the API contract

These are not just small glue costs. They are part of the TTFT contract and
should be reviewed explicitly as part of the optimization series.

### 5. Mamba boundary costs

Current concerns inside `runtime/src/backend/mamba_layer.cpp`:

- fused norm and scan are surrounded by explicit dtype / projection boundaries
- Mamba remains one of the largest long-prefill costs
- supporting setup and launch structure still need to be separated from SSD
  kernel cost in profiling

Expectation setting:

- Mamba likely has a lower optimization ceiling than MoE or long-context
  attention because the SSD scan is fundamentally sequential across tokens
- unless profiling proves otherwise, assume the likely gain is meaningful but
  bounded, not a multi-x rewrite opportunity

### 6. Attention metadata and bridge costs

Current concerns inside `runtime/src/backend/attention_layer.cpp`:

- request metadata is still uploaded from host each run
- the BF16 bridge path still contains casts plus a sync
- the current multi-token attention kernel still iterates over visible KV
  tokens one position at a time rather than using a real block-tiled KV
  traversal

Even if these costs are smaller than MoE or Mamba today, they still belong to
the same stall/transfer/fusion audit framework.

## Narrowing Decision

Do **not** try to close every remaining architectural gap in one sweep.

That would mix at least five distinct jobs:

- MoE prefill contract replacement
- token-selection / logits-contract replacement
- Mamba long-prefill optimization
- attention long-tail optimization
- possible broader projection-path replacement

That scope is too wide. It would make TTFT attribution muddy, increase the
chance of another "almost works but destabilizes correctness" detour, and
pull us back toward generic abstractions instead of target-specific execution.

The correct scope for the next implementation phase is narrower:

1. replace the host-visible greedy token-selection contract with a device-side
   token-selection contract
2. replace the generic MoE prefill contract with a Nano / `SM120`-specific,
   device-resident execution contract
3. only after those land and are re-profiled, decide whether Mamba or
   attention is the next hotspot

This is still aggressive. It closes two real architectural gaps. It is simply
not trying to close every remaining one at the same time.

## Target-Specific Assumptions

This phase should exploit the real, fixed target instead of preserving
cross-model flexibility.

Hard assumptions we are allowed to bake into code:

- model: Nemotron 3 Nano 30B A3B NVFP4
- GPU: single consumer Blackwell RTX 5090, `SM120`
- hidden size: `2688`
- vocab size: `131072`
- routed experts: `128`
- experts per token: `6`
- routed expert intermediate size: `1856`
- shared expert intermediate size: `3712`
- attention heads / KV heads: `32 / 2`
- total layers: `52`
- build target: `build-sm120-relwithdebinfo`
- serving mode in scope for token selection: greedy decode

Design consequences:

- prefer target-specific kernel code and workspace layouts over generic helper
  layers
- precompute and store any expert pointer tables, weight metadata, and
  scale-layout metadata at model-load time if it removes runtime work
- choose kernel launch geometry, tiling, and data layout for this exact model
  and this exact GPU family
- do not preserve extension points for other models, other quantization
  schemes, or other GPU families inside the hot path

Out of scope for this phase:

- generalized sampling backends
- beam search or speculative decode redesign
- multi-model runtime generalization
- multi-GPU MoE / expert-parallel runtime contracts
- replacing every standalone projection GEMM in the runtime

## Implementation Order

The next phase should be executed in this order:

1. device-side token selection and explicit logits ownership
2. Nano-specific MoE prefill contract replacement
3. full re-profile of `4` / `128` / `4096` and only then choose between Mamba
   and attention as the next hotspot

Why token selection first even though MoE is the larger TTFT block:

- it is smaller and more isolated
- it removes a clear hot DtoH cost from every prompt/decode path
- it forces the forward API to stop assuming host-visible logits by default
- it simplifies the boundary-logits story before the larger MoE refactor lands

Why MoE is still the main performance target of this phase:

- it is still the dominant `128`-class prefill cost
- its remaining host loop and DtoH metadata dependency are not acceptable as a
  production architecture
- the grouped CUTLASS attempt already proved that we should not wait for a
  "perfect prewritten kernel" to fix the contract problem for us

Long-tail caveat:

- for `cached_committed_head_prefix4096_tail4096`, long-context attention may
  now have the highest single-step upside if a true tiled rewrite replaces the
  current per-token KV traversal
- that does not change the business priority of removing the greedy-logits DtoH
  contract and the MoE host-control contract first; it only means the
  post-Step-3 reprofile must make the next hotspot decision explicit

## Steps

- [x] **0. Freeze the current stable baseline**
  Save the current TTFT matrix, prefill trace, and representative `nsys`
  captures as the reference point for all follow-on work.
  Key artifacts:
  `artifacts/profiles/ttft_prefill_20260405/`

- [x] **1. Lock the current operation boundaries before replacing contracts**
  The active operation boundaries and current projection-path decision are
  recorded in `STEP1_OPERATION_BOUNDARY_AUDIT.md`.

  The important consequence for this phase is:

  - standalone projection GEMMs remain on the current single cuBLASLt /
    CUTLASS-backed path unless this plan is intentionally superseded
  - MoE execution is still allowed to become a dedicated Nano / `SM120`
    operation with target-specific kernels and target-specific metadata if that
    is the cleanest way to remove host control

- [ ] **2. Replace default host-visible logits with device-side token selection**
  The end state is:

  - default greedy prefill / decode paths do **not** copy full logits to host
  - the device computes the next token id
  - host copies out only the selected token id and other minimal control data
  - full logits are copied out only when explicitly requested by the caller

  Concrete design requirements:

  - add a dedicated device-side argmax path specialized for vocab size
    `131072`
  - support the two real cases only:
    - last-row argmax for prefill logits
    - row-0 argmax for single-token decode logits
  - change the greedy forward contract so that "next token id is ready" is the
    default result, not "host copy of logits is ready"
  - make boundary-logits ownership explicit:
    - default cache path stores boundary token id
    - boundary logits are optional debug / oracle data, not mandatory serving
      state
  - make any full-logits copy opt-in at the API boundary rather than buried in
    `RunGreedyDecode`, `RunGreedyConversationTurn`, or `ContinueGreedyDecode`

  Inspiration / alignment:

  - TRT-LLM already carries dedicated Blackwell argmax kernels and device-side
    sampling primitives
  - our implementation should copy the contract idea, not their full runtime
    sampler stack

  Explicit non-goals:

  - do not build a generic sampling subsystem in this phase
  - do not preserve full-logits copies as the default behavior "for future
    flexibility"
  - do not add a second decode path just for tests or benchmarks

  Acceptance criteria for Step 2:

  - no full-logits DtoH copy on the default greedy path
  - exact correctness retained on the Nano oracle path
  - prefix-cache exact-hit flow works without requiring host-visible boundary
    logits by default
  - TTFT is not worse in any of the `4` / `128` / `4096` cases and should
    improve most obviously in short / medium prompt regimes

- [ ] **3. Replace generic MoE prefill parameters with a Nano / `SM120` fused-op contract**
  The end state is:

  - no runtime DtoH copy of routing metadata
  - no host-side loop over active experts
  - no per-expert runtime plan-build / launch-control logic on the CPU
  - a fixed host launch schedule whose structure does not depend on active
    expert count
  - all routed-MoE execution metadata lives in device-resident workspace owned
    by the layer or request context

  This step should stop treating MoE prefill as "generic tensors + generic
  scratch + host orchestration". Replace that with a target-specific operation
  boundary for exactly Nano on `SM120`.

  Concrete contract changes:

  - replace `FusedMoePrefillParams` with an opaque Nano-specific execution
    contract that does not expose host arrays of expert descriptors / weight
    views in the hot path
  - prebuild and retain any device-side expert pointer tables, scale tables,
    and other invariant metadata at model-load time
  - move all runtime-routed metadata into device-resident workspace:
    - sorted token ids
    - top-k weights
    - expert offsets / counts
    - active expert table
    - any packed-activation metadata needed by the final execution kernel(s)
  - explicitly target zero hot-path `cudaMalloc` / `cudaFree` after warmup for
    the landed MoE path
  - replace per-expert activation quantization as the organizing principle of
    the routed path; the target pattern is one-pass activation quantization
    before device-side dispatch, not repeated host-driven `PackInto` per expert
  - make the runtime API "one MoE op in, one output tensor out", not "host
    loop over experts with scratch tensors exposed"

  Execution strategy decision:

  - because the exact grouped CUTLASS MoE kernel did not meet the bar on this
    branch, the default strategy is now a hand-crafted Nano / `SM120`
    execution design
  - the relevant reference architecture is still FlashInfer / TRT-LLM style:
    device-side routing metadata, one-pass quantization before grouped expert
    execution, and a single fused operator boundary
  - prewritten library kernels may remain only where they fit inside the one
    fused production path and do not reintroduce host control, runtime split
    dispatch, or a second MoE backend

  This means the plan is allowed to:

  - write target-specific kernels for routing prepare / finalize
  - write target-specific kernels that fuse gather / activation / weighted
    scatter / reduction
  - hard-code Nano shape assumptions and Blackwell-friendly launch geometry

  This plan is **not** allowed to:

  - restore grouped CUTLASS split-dispatch
  - reintroduce per-expert host launch control
  - keep a generic MoE path and a Nano-specific fast path side by side
  - leave "temporary" adapters in the landed tree once the contract is
    replaced

  Intermediate staging rule:

  - branch-local refactors are allowed if needed to get there
  - each landed commit must still expose only one production MoE path in the
    runtime tree

  Acceptance criteria for Step 3:

  - no hot-path DtoH / HtoD routing metadata traffic
  - no host loop whose iteration count depends on active experts
  - zero hot-path `cudaMalloc` / `cudaFree` after warmup
  - fixed launch schedule independent of active-expert count
  - exact correctness retained on the Nano oracle path
  - material TTFT gain in `prefix128` and cached `tail4` / `tail128`
  - no meaningful regression in `prefix4` or `prefix4096`

- [ ] **4. Re-profile the full matrix and decide the next hotspot**
  After Steps 2 and 3 land:

  - rerun the full TTFT matrix
  - rerun prefill tracing
  - rerun representative `nsys`
  - update this project plan with the new dominant hotspot

  Only then choose the next implementation target:

  - Mamba long-prefill work if it is still the dominant cold `4096` cost
  - attention long-tail work if resumed `4096 + 4096` is now clearly larger

  Expectations for that decision:

  - if attention remains next, treat it as an architectural rewrite, not a
    tuning pass, because the current kernel still lacks real block-level KV
    tiling
  - if Mamba remains next, assume a more bounded improvement ceiling unless new
    profiling evidence says the current support code, not the SSD scan itself,
    is the dominant cost

  Do **not** assume today that both will be part of the same implementation
  phase.

- [ ] **5. Treat re-profile + correctness as a stage gate after every landed optimization**
  This is not a trailing cleanup step. It is a gate after Steps 2 and 3 and
  again after Step 4 decides the next hotspot.

  After each substantial change:
  - rerun correctness
  - rerun the `4` / `128` / `4096` TTFT buckets
  - rerun the representative trace / `nsys` capture if the hotspot moved
  - rerun the stall/transfer/fusion audit for the code that changed

  Any change that removes code must be tested immediately before the next
  cleanup step begins.

- [ ] **6. Treat contract cleanup as a gate in every landed series**
  This is also not a trailing step. It applies to every landed optimization.

  When an optimization lands, remove dead code and stale assumptions in the
  same series:
  - old scratch paths
  - dead helper kernels
  - obsolete comments and docs
  - stale benchmark knobs that no longer reflect the runtime contract

  The end state should be faster **and** cleaner than the current baseline.

## Adversarial Review Of This Plan

### Why not close every architectural gap now

Because that would be too broad and would likely repeat the exact failure mode
we already saw with grouped CUTLASS MoE: too much architectural movement at
once, unclear attribution, and correctness risk rising faster than performance
confidence.

### Why not jump straight to a full custom projection rewrite

Because the current evidence still says the hottest remaining problems are
contract and orchestration problems around MoE and greedy decode, not just raw
GEMM throughput. A projection rewrite now would be too large and would blur
whether the real win came from math, launch structure, or transfer removal.

### Why token selection first

This is the most contained contract change that removes a known DtoH tax from
every path. If we cannot make logits ownership explicit here, then every later
optimization is still working around a host-visible API that we already know is
too generic.

### Why the plan still narrows scope instead of "fusing everything"

Because "fuse everything" is not a plan. The correct interpretation of the
single-target constraint is:

- be specific
- hard-code shapes and hardware assumptions where that helps
- remove genericity where it is costing us
- but still change one dominant contract at a time so we can prove each win and
  keep correctness locked

### Main risks to watch

- hidden test or cache dependencies on full host-visible logits
- MoE refactors that accidentally keep the old host-loop control path alive in
  another form
- broad "helper" abstractions that sneak genericity back into the new contract
- overfitting token-selection changes to benchmarks while breaking oracle /
  interactive usage
- trying to make the MoE replacement portable before it is fast and correct on
  Nano / `SM120`

### Final decision after review

Keep the next phase narrow in scope but extremely specific in implementation:

- fully close the greedy token-selection / logits-ownership gap
- fully close the generic MoE prefill / host-control gap
- then re-profile before deciding on Mamba or attention

## Measurement Discipline

Always use:

- `build-sm120-relwithdebinfo`
- `artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json`
- the checked-in TTFT bench
- `nsys` for whole-system attribution
- `ncu` only after `nsys` identifies a kernel that is actually worth deep
  analysis
- `uv` for any helper parser or analysis script

Primary commands:

```bash
export NEMOTRON_FORWARD_MANIFEST=artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json

./build-sm120-relwithdebinfo/benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench \
  --warmup 1 \
  --iterations 5 \
  --tail-token-count 4

./build-sm120-relwithdebinfo/benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench \
  --warmup 1 \
  --iterations 5 \
  --tail-token-count 128

./build-sm120-relwithdebinfo/benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench \
  --warmup 1 \
  --iterations 5 \
  --tail-token-count 4096
```

When reporting results, always break them out by the full regime matrix:

- cold `prefix4`, `prefix128`, `prefix4096`
- cached `tail4` for `prefix4`, `prefix128`, `prefix4096`
- cached `tail128` for `prefix4`, `prefix128`, `prefix4096`
- cached `tail4096` for `prefix4`, `prefix128`, `prefix4096`

Tracing / profiling commands:

```bash
NEMOTRON_FORWARD_PREFILL_TRACE=1 \
./build-sm120-relwithdebinfo/benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench \
  --warmup 0 \
  --iterations 5 \
  --tail-token-count 4096 \
  --case cold_prefill_prefix4096 \
  --case cached_committed_head_prefix4096_tail4096

/usr/local/cuda-13.0/bin/nsys profile --force-overwrite true \
  --output artifacts/profiles/ttft_prefill_20260405/nsys_cold_prefill_prefix128 \
  ./build-sm120-relwithdebinfo/benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench \
  --warmup 0 \
  --iterations 5 \
  --tail-token-count 128 \
  --case cold_prefill_prefix128

/usr/local/cuda-13.0/bin/nsys profile --force-overwrite true \
  --output artifacts/profiles/ttft_prefill_20260405/nsys_cached_committed_head_prefix4096_tail4096 \
  ./build-sm120-relwithdebinfo/benchmarks/nano_prefix_cache_ttft/nano_prefix_cache_ttft_bench \
  --warmup 0 \
  --iterations 5 \
  --tail-token-count 4096 \
  --case cached_committed_head_prefix4096_tail4096
```

If a helper parser is needed for result summarization, run it with `uv`.

## Validation Discipline

At minimum, after each material change:

- `ctest --test-dir build-sm120-relwithdebinfo --output-on-failure`
- `proj-2026-04-03-0318/verify_correctness.sh --build-dir build-sm120-relwithdebinfo --manifest artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json`
- the TTFT benchmark set above

Focused checks that should be included when relevant:

- `nano_16_token_correctness_test`
- `full_forward_manifest_smoke_test`
- `single_token_decode_oracle_test`
- `nvfp4_vllm_golden_test`

If a change touches projection GEMMs, MoE math, or NVFP4 quantization flow,
`nvfp4_vllm_golden_test` is mandatory before the full TTFT rerun.

If the change is localized, run the relevant focused tests first, then the full
SM120 regression set before considering the change complete.

## Exit Criteria

This plan is complete only when all of the following are true:

- TTFT is materially better in the `4`, `128`, and `4096` regimes
- the gains survive cached-tail cases, not just cold-prefill microbenches
- correctness remains exact on the Nano oracle path
- the default greedy path does not perform full-logits DtoH copies
- boundary logits are copied only when explicitly requested
- the MoE prefill path no longer performs runtime DtoH routing-metadata copies
- the MoE prefill path no longer contains a host loop whose trip count depends
  on active experts
- the active `SM120` forward pass still has one implementation path per
  operation
- no new fallback code or backend-selection scaffolding has been introduced
- the implementation path that lands is explicitly aligned with the relevant
  `SM120` prior art contracts from vLLM and TRT-LLM, without importing their
  runtime backend-selection structure
