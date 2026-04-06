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

This plan starts from the current, correct `SM120` baseline on commit
`e2196e5`. The optimization job now is to reduce TTFT without reintroducing
fallbacks, backend ladders, env-gated alternate paths, or size-based split
dispatch.

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
- `cold_prefill_prefix128`: the current projection GEMM decision is highly
  load-bearing because the active NVFP4 GEMM kernel is still about `49.1%` of
  GPU kernel time in the representative `nsys` run
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

## Optimization Order

The profile gives a clear order. Work should proceed in this sequence.

- short and medium TTFT first: expert prefill
- long cold TTFT second: Mamba prefill
- long resumed TTFT third: attention over long total context

That order matches the measured bottlenecks and avoids optimizing the wrong
operation first.

## Steps

- [x] **0. Freeze the current baseline**
  Save the current TTFT matrix, prefill trace, and representative `nsys`
  captures as the reference point for all follow-on work.
  Key artifacts:
  `artifacts/profiles/ttft_prefill_20260405/`

- [ ] **1. Re-audit the operation contracts and lock the next replacement boundary**
  Before more kernel work, do one explicit audit of the active operation
  surfaces and record the exact replacement boundary for each one:
  - attention
  - Mamba
  - MoE
  - projection GEMMs

  This step must answer one unresolved architectural question:

  - Are projection GEMMs staying on the current cuBLASLt / CUTLASS-backed
    runtime path as the permanent single implementation?
  - Or are they being fully replaced later by a native path?

  This step must also record the external alignment conclusion:

  - vLLM reference for Nano NVFP4 / `SM120`: FlashInfer CUTLASS-style MoE
  - TRT-LLM reference for Nano NVFP4 / `SM120`: CutlassFusedMoE
  - not TRTLLM-Gen, which is not an `SM120` implementation target

  Step 1 deliverable:

  - a short written record in this project directory that states, for each
    operation, what the single production implementation boundary is
  - an explicit yes/no decision on whether the current projection GEMM stack is
    the permanent implementation boundary

  Do not start another partial GEMM rewrite until that answer is explicit.

- [ ] **2. Optimize expert prefill for `128`-class TTFT without adding a second MoE path**
  The target here is `cold_prefill_prefix128` and the `tail4` / `tail128`
  cached cases. The current evidence says the main opportunities are:
  - runtime activation packing/scaling cost
  - avoidable HtoD traffic
  - workspace / allocation churn
  - synchronization around the current expert execution sequence

  Step 2 should be treated as three explicit sub-targets, in this order:

  1. quantify and reduce the expert-path quantization/packing/scaling stack
     because it is already about `125 ms` in the profiled `prefix128` run
  2. attribute `cudaMemcpy` time into one-time/setup traffic versus per-layer
     hot-path copies, then remove the hot-path portion
  3. only then decide whether the remaining expert-path bottleneck is GEMM,
     dispatch/finalize, or synchronization

  Required diagnostics before changing kernels:

  - determine whether the dominant `cudaMemcpy` calls are one-time/model-load,
    benchmark harness overhead, or true per-layer hot-path copies
  - identify whether any `cudaStreamSynchronize` calls fence inside expert
    execution rather than only at benchmark boundaries
  - confirm whether the current quantization/packing work is duplicated across
    experts, layers, or per-window processing

  Allowed work:
  - reduce or eliminate redundant per-layer packing/scaling work
  - move scratch/workspace ownership out of the hot path
  - restructure the existing MoE execution flow to lower launch count

  Disallowed work:
  - split dispatch by token count or expert size
  - host fallback helpers
  - "temporary" side paths for comparison

  If the final answer is a grouped expert kernel, it must replace the current
  MoE execution path outright and the old path must be deleted in the same
  series.

  Explicit non-goal for Step 2:

  - do not resume the earlier grouped CUTLASS split-dispatch MoE experiment
    from `proj-2026-04-05-0445`

- [ ] **3. Optimize Mamba prefill for `4096`-class cold TTFT**
  The target here is `cold_prefill_prefix4096`. The current trace says Mamba is
  the largest cold-prefill cost by far.

  Focus areas:
  - `MambaSsdPrefillFixedKernel`
  - conv-prefill support kernels
  - fused norm/quant boundaries around Mamba input/output
  - state-update path and chunk metadata construction
  - launch shape, tiling, and memory traffic
  - any remaining synchronization or workspace overhead in the Mamba layer

  Required diagnostics before changing kernels:

  - verify whether the main cost is inside SSD prefill proper or in supporting
    conv/norm/state-update setup around it
  - use the `8192` trace as a scaling reference to distinguish linear work from
    pathologically growing work

  Keep one implementation path for Mamba. Do not introduce a special
  "long-prefill-only" Mamba backend.

- [ ] **4. Optimize attention for long resumed tails**
  The target here is `cached_committed_head_prefix4096_tail4096`. After the
  cache hit, attention becomes much larger because the total active sequence is
  `8192`.

  Focus areas:
  - `PagedAttentionNanoMultiTokenKernel`
  - long-sequence tiling and memory access
  - minimizing wasted work when only the tail is new
  - auditing whether any host-visible sync points serialize work inside the
    layer, rather than only fencing at the benchmark boundary

  Required diagnostics before changing kernels:

  - confirm whether the dominant cost is expected long-kernel execution or
    avoidable serialization between launches
  - verify whether any cache-restore or tail-shape bookkeeping is causing extra
    work inside the attention layer

  Keep the current native attention path as the only production path. Use
  vLLM / TRT-LLM only as algorithm references, not as runtime templates with
  multiple backend choices.

- [ ] **5. Treat re-profile + correctness as a stage gate after every optimization step**
  This is not a trailing cleanup step. It is a gate after Steps 2, 3, and 4.

  After each substantial change:
  - rerun correctness
  - rerun the `4` / `128` / `4096` TTFT buckets
  - rerun the representative trace / `nsys` capture if the hotspot moved

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
- the active `SM120` forward pass still has one implementation path per
  operation
- no new fallback code or backend-selection scaffolding has been introduced
- the implementation path that lands is explicitly aligned with the relevant
  `SM120` prior art contracts from vLLM and TRT-LLM, without importing their
  runtime backend-selection structure
