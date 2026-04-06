# Step 3 Recovery Plan

Project directory: `./proj-2026-04-05-1704`

## Goal

Recover from the current custom-MoE-prefill regression, validate that the new
device-side contract is still the right architectural direction, and then push
past the best pre-refactor TTFT numbers.

## Current Priority: Prefill Optimization

The contract work is now good enough to treat as fixed for this phase:

- one active runtime code path
- device-only runtime contract
- no hot-path DtoH
- correctness restored on the committed-head reuse path

That means the priority is no longer "prove the contract." The priority is:

1. optimize prefill first
2. recover the cold TTFT baseline
3. beat the baseline with a stronger math core

Everything else is secondary until cold prefill is back under control.

## Specialization Assumptions

This plan is intentionally specialized for one concrete deployment target:

- model: official Nemotron 3 Nano NVFP4 checkpoint
- GPU: GeForce RTX 5090 (`SM120`, Blackwell)
- routed expert shape:
  - hidden size `2688`
  - routed intermediate size `1856`
  - shared intermediate size `3712`
  - routed experts `128`
  - routed experts selected per token `6`

That means:

- the first objective is not a generic MoE kernel family
- we should tune tile sizes, staging strategy, launch shape, and scratch layout
  for these exact dimensions
- we should optimize for RTX 5090 execution characteristics directly
- we should prefer a single specialized fast path over a more abstract slower
  implementation, as long as we do not create a second runtime path
- one-time load-time specialization or repacking is acceptable if it improves
  the active path; per-request repacking is not

Generality matters only where it is required to preserve correctness across the
benchmarked prompt lengths, cache states, and serving scenarios for this same
model and GPU target.

## Prefill Optimization Plan

This is the active plan for the next phase of work.

### Success Criteria

Primary success criteria:

- beat the old cold-prefill baseline on the plan-aligned benchmark surface
- keep hot-prefix TTFT at or below the old baseline
- preserve one runtime path and the current device-only contract

Required correctness gates:

- `fused_moe_prefill_test`
- `expert_routing_device_test`
- `multi_turn_prefix_reuse_test`
- focused interactive-forward sanity checks

Required benchmark gates before claiming success:

- plan-aligned TTFT sweep from `PLAN.md`
- throughput bench
- full `ctest`

### Guiding Principles

1. Do not add a second implementation path.
2. Do not add host-side control or DtoH in the hot path.
3. Specialize aggressively for the deployed Nemotron Nano NVFP4 on RTX 5090
   shape when that improves the active runtime path.
4. Use TRT-LLM as the required structural reference for operation boundaries,
   work decomposition, and staging.
5. Do not spend more time polishing scalar dot-product kernels once the data
   says the architecture is wrong.
6. Replace one operation boundary at a time and benchmark after each change.
7. Keep the runtime contract stable while changing the math core underneath it.

### Required TRT-LLM Comparison Process

Before implementing or changing any major prefill operation, compare our code
against the closest TRT-LLM operation and record the comparison in this project
file.

Required comparison targets:

- routing / renormalize
- token permutation / expert-major layout
- grouped first projection
- activation / requantization
- grouped second projection
- finalize / unpermute

Questions to answer for each operation:

- what work decomposition does TRT-LLM use
- where does TRT-LLM create expert-major or tile-major parallelism
- where does TRT-LLM preserve locality between stages
- what parts of TRT-LLM are useful structure for us to copy directly
- what parts should be specialized more aggressively for `SM120` and the exact
  Nano dimensions

## Assumption Audit

This section records the key assumptions behind the current plan so they stay
explicit and reviewable.

### 1. Shape regime matters, and it changes with prefix length

For Nano:

- `prefix4`: `24` routed selections total, average `0.1875` tokens per expert
- `prefix128`: `768` routed selections total, average `6` tokens per expert
- `prefix4096`: `24,576` routed selections total, average `192` tokens per
  expert

These are materially different execution regimes:

- `prefix4` is effectively a sparse tiny-batch regime
- `prefix128` is an awkward small-batch grouped regime
- `prefix4096` is a real grouped-GEMM regime

Plan consequence:

- we still want one runtime code path
- we do not want three separate math stacks
- therefore the chosen math core has to tolerate under-filled small-`M` cases
  while still scaling well at `prefix128` and `prefix4096`

Design-center assumption:

- optimize primarily for `prefix128` and `prefix4096`
- keep `prefix4` correct and accept that it is the least hardware-friendly
  point on the benchmark surface
- if small-`M` handling is needed, prefer padding or tile under-fill inside the
  same grouped kernel family rather than dispatching to a second algorithmic
  path

### 2. Launch count was a structural problem, but the current routed baseline already fixed most of it

The earlier grouped implementation really did suffer from huge routed launch
fan-out. That was a valid concern.

But the retained single-launch grouped baseline has already collapsed the
routed path from per-expert launches to one routed-up and one routed-down
launch per expert layer:

- routed grouped expert matvec launches in current `nsys`:
  `230` over `5` measured iterations = `46` per iteration = `23` layers x `2`
  routed stages

The shared path still contributes another `46` dense expert matvec launches per
iteration, so the total expert-math launch count is still not minimal, but the
main routed launch explosion is no longer the dominant issue.

Plan consequence:

- further launch-count reduction is welcome
- but the current cold-prefill gap is now mostly a math-core efficiency
  problem, not a `5,888`-launch structural problem

### 3. Tensor-pipe use must become a hard success criterion

The current `ncu` profile shows effectively `0%` tensor-pipe activity in the
dominant routed expert math kernel.

Plan consequence:

- any replacement routed-up or routed-down design that still shows effectively
  zero tensor-pipe activity should be treated as suspect
- "clear microbench win" is not enough by itself; the win should come from a
  better execution mechanism, not another scalar dequant-plus-dot-product loop

Operational gate:

- first routed-up replacement must move tensor-pipe activity above zero
- later iterations should aim to raise it materially, not merely make it
  nonzero

### 4. cuBLASLt is not the active implementation target for this path

This is an explicit architectural assumption, not an oversight.

We are not currently planning to use cuBLASLt as the shipping prefill math core
for this MoE path.

Reasons:

- we intentionally moved this path away from a library-controlled expert
  execution model to keep the runtime contract fully under our control
- we want one device-driven runtime path, not a path that depends on opaque
  library algorithm selection and host-orchestrated grouped GEMM setup
- the awkward `M` distribution for `prefix4` and `prefix128` would still force
  us to make padding and scheduling decisions; cuBLASLt does not remove that
  problem, it only moves the math into a library call
- TRT-LLM is the architectural reference, but the goal is to reproduce its
  strong operation shape with a much narrower custom implementation specialized
  to this model and GPU

Plan consequence:

- TRT-LLM remains the reference for operation structure
- cuBLASLt can still be used as a benchmark or sanity yardstick if needed
- but it is not the assumed end-state implementation for the active prefill
  path

### 5. Shared experts are not a late afterthought

The current single-launch profile still shows the shared expert dense path at
about `23%` of GPU kernel time on cold `prefix128`.

Plan consequence:

- routed expert math is still the first priority because it is larger
- but shared expert execution is large enough that we should evaluate it early
  once the routed-up replacement shape is clear
- shared should not remain parked at the end of the plan as a vague cleanup
  task

## Routed-Up Operation Comparison

This is the first required operation-level comparison between the current code
and TRT-LLM.

### Current runtime path

Current routed-up in `runtime/src/backend/fused_moe_prefill.cu`:

- input contract is already reduced to grouped routed rows:
  `selection_count x hidden_size`
- launch shape is one block per `(expert, output_row)`
- grid size is `n_experts * routed_intermediate_size`
- each block decodes FP4 weights on the fly and computes scalar dot products
  over the full hidden dimension
- each block handles up to `kGroupedTokenTile = 8` routed rows at a time
- output layout is row-major `selection_count x routed_intermediate_size`

Implications:

- work decomposition is output-row-major, not expert-tile-major
- `M` only changes how many inner-loop row tiles each block processes
- tensor-pipe usage is effectively zero
- the kernel remains scalar and control-heavy even when `M` becomes large

### TRT-LLM reference path

Relevant files:

- `third_party/TensorRT-LLM/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.cu`
- `third_party/TensorRT-LLM/cpp/tensorrt_llm/kernels/cuteDslKernels/moeUtils.h`

TRT-LLM routed-up structure:

- routing builds permutation and launch metadata:
  - `permutedIdxToTokenIdx`
  - `numNonExitingCtas`
  - `ctaIdxXyToBatchIdx`
  - `ctaIdxXyToMnLimit`
- `PermuteGemm1::Runner::run()` executes a grouped batched GEMM over the
  routed tokens and experts
- work is organized around expert-major or tile-major grouped GEMM launch
  metadata rather than one block per output row
- the grouped GEMM runner is configured with `tileTokensDim`, `routeAct`, and
  transpose/epilogue behavior so the routed-up stage stays in a GEMM-friendly
  layout for the next stage

### Structural mismatch

The key mismatch is not only "we use a different kernel." It is:

- TRT-LLM decomposes routed-up as grouped GEMM over expert-major tiles
- our current kernel decomposes routed-up as scalar row-wise matvec over
  `n_experts * output_rows`

That means TRT-LLM creates parallelism along the tile axes that matter for
tensor-core math, while our current kernel creates a huge number of scalar
blocks that each do their own FP4 decode and reduction work.

This comparison reinforces the current architectural target:

- keep the runtime contract
- replace the routed-up math core
- move the work decomposition toward expert-major grouped tiles

## Routed-Up Microbench Baseline

The first Phase-1 harness is now implemented at:

- `benchmarks/nano_moe_prefill/nano_routed_up_bench.cpp`

It benchmarks the current active routed-up math core directly through
`RunGroupedNvfp4ExpertMatVec()` using the exact Nano routed-up dimensions:

- hidden size `2688`
- routed intermediate size `1856`
- routed experts `128`
- top-k `6`

Current measured results with `--warmup 1 --iterations 5`:

- `prefix4`
  - selection count `24`
  - active experts `24`
  - hot mean `0.520 ms`
  - effective routed-up throughput `0.461 TFLOP/s`
- `prefix128`
  - selection count `768`
  - active experts `128`
  - hot mean `10.282 ms`
  - effective routed-up throughput `0.745 TFLOP/s`
- `prefix4096`
  - selection count `24,576`
  - active experts `128`
  - hot mean `320.900 ms`
  - effective routed-up throughput `0.764 TFLOP/s`

Immediate interpretation:

- the kernel does not materially improve its effective compute rate from
  `prefix128` to `prefix4096`
- that is strong evidence that the ceiling is the scalar math core itself, not
  only small-`M` inefficiency
- the routed-up kernel alone already reproduces the same ~`0.75 TFLOP/s`
  ceiling implied by the end-to-end profile

This is the current baseline to beat for the routed-up replacement.

### Benchmark Contract

Fast iteration benchmarks:

- `cold_prefill_prefix128`
- `cached_committed_head_prefix128_tail4`

Official checkpoint benchmarks:

- `cold_prefill_prefix4`
- `cold_prefill_prefix128`
- `cold_prefill_prefix4096`
- `cached_committed_head_prefix4_tail4`
- `cached_committed_head_prefix128_tail4`
- `cached_committed_head_prefix4096_tail4`

Interpretation rules:

- cold prefill is the primary optimization target
- cached hot-prefix TTFT must not regress while recovering cold prefill
- throughput is a validation gate, not the tuning loop metric

### Phase 0. Freeze the Current Baseline

Objective:

- treat the current grouped single-launch path as the baseline to beat

Work:

- keep the current `nsys` and `ncu` artifacts as the control profile
- record the retained benchmark numbers in this project file
- use the current device-only grouped path as the reference implementation for
  correctness during the rewrite

Stop condition:

- do not change benchmark surface or contract assumptions during this phase

### Phase 1. Build Operation-Level Tuning Harnesses

Objective:

- make each prefill operation measurable on its own

Work:

- add a microbench for routed row gather / permute
- add a microbench for routed-up
- add a microbench for activation + requantization
- add a microbench for routed-down
- add a microbench for finalize / unpermute / weighted reduction
- keep all timed regions device-only
- size the first version of every harness for the exact Nano dimensions before
  considering any broader parameterization

Reference requirement:

- for each harness, point to the closest TRT-LLM operation and note the current
  structural mismatch before writing new kernel code

Deliverable:

- one small benchmark target or harness per operation

Stop condition:

- do not start new kernel work until the touched operation has a measurable
  microbench

### Phase 2. Replace Routed-Up with a Real Grouped Math Core

Objective:

- eliminate the scalar routed-up matvec as the dominant cold-prefill hotspot

Target shape:

- expert-major or tile-major work decomposition
- many tokens per launch, not one output row per block
- explicit staging of input tiles and weight tiles
- enough regularity to support tensor-like or tensor-core-friendly math

Reference shape:

- TRT-LLM `route -> permute -> grouped gemm1`

Work:

- compare our current routed-up path against TRT-LLM `PermuteGemm1` first
- design the routed-up scratch layout around grouped tiles rather than scalar
  output rows
- choose tile sizes and staging specifically for `hidden=2688`,
  `routed=1856`, `top_k=6`, and RTX 5090 execution behavior
- replace `Nvfp4GroupedExpertMatVecRowsKernel` for routed-up first
- benchmark the routed-up microbench before wiring it into end-to-end prefill
- run fast TTFT benchmarks after integration

Success gate:

- routed-up microbench shows a clear win
- routed-up profile shows nonzero tensor-pipe activity
- cold `prefix128` improves materially

### Phase 3. Replace Activation + Requantization with a Tiled Pipeline Stage

Objective:

- keep the grouped routed-up output in a layout that feeds routed-down without
  falling back to a scalar cleanup phase

Target shape:

- vectorized activation
- vectorized requantization
- minimal extra global-memory traffic
- no thread-0 scalar passes

Reference shape:

- TRT-LLM activation/quant stage between grouped `gemm1` and grouped `gemm2`

Work:

- compare our current activation stage against TRT-LLM activation/quant staging
  first
- define the intermediate layout needed by both routed-up and routed-down
- fuse where the code stays understandable
- keep the intermediate layout specialized for the routed-up / routed-down tile
  shape we actually deploy
- benchmark this stage independently before full integration

Important note:

- this phase does not assume activation/requant must remain a permanent
  standalone stage
- if the chosen grouped math core makes this boundary simpler or partially
  fusable, take the simpler design

Success gate:

- activation/quant is no longer an architectural blocker between grouped
  routed-up and grouped routed-down

### Phase 4. Replace Routed-Down and Finalize

Objective:

- complete the routed path with a grouped routed-down stage and an explicit
  finalize step

Target shape:

- grouped routed-down math over expert-major tiles
- explicit finalize / unpermute / weighted reduction back to token order
- no scalar in-kernel accumulation as the primary combine strategy

Reference shape:

- TRT-LLM `grouped gemm2 -> finalize`

Work:

- compare our current routed-down and weighted reduction against TRT-LLM
  `Gemm2 -> finalize` first
- replace the current routed-down scalar grouped matvec
- make finalize its own explicit operation boundary
- keep weighted reduction semantics identical to the current contract

Success gate:

- routed path becomes fully grouped end to end
- cold `prefix128` moves substantially closer to baseline

### Phase 5. Rework the Shared Expert Path

Objective:

- decide whether the shared path should reuse the grouped core or stay as a
  smaller specialized path

Work:

- measure the shared path separately after the routed path is upgraded
- compare against TRT-LLM shared-expert separation before changing this path
- if it is still a meaningful cold-prefill cost, replace its scalar matvec
  stages too
- otherwise keep it simple and leave it alone

Decision rule:

- optimize shared only after routed no longer dominates
- if the new grouped math core is already natural for the shared dense path,
  pull shared forward rather than waiting for a separate late phase

### Phase 6. Recover and Beat the Baseline

Objective:

- move from "less bad" to "better than before"

Work:

- rerun the full plan-aligned TTFT sweep
- rerun throughput
- rerun full `ctest`
- rerun interactive conversation sanity checks
- compare directly against the old recorded baseline artifacts

Finish line:

- cold prefill better than the pre-refactor baseline
- cached hot-prefix at least maintained
- no correctness regressions

This phase is not "rewrite everything again". It is:

1. profile the current custom kernel stack
2. compare each kernel operation against stronger reference implementations
3. optimize one operation at a time with micro tests and benchmark checkpoints
4. stop only when we are better than the pre-refactor baseline or we have
   proven that a specific operation boundary must be redesigned

## Current Status

Checkpoint commit:

- `9c93ed2` `Checkpoint custom MoE prefill refactor`

Current known regression from the native TTFT run in
`artifacts/benchmarks/nano_prefix_cache_ttft_20260405T_native.stdout.txt`:

- `cold_prefill_prefix256`: `2234.009 ms`
- `cold_prefill_prefix1024`: `7781.018 ms`
- `cold_prefill_prefix4096`: `28173.273 ms`
- `cached_committed_head_prefix256_tail32` hot-prefix: `1151.702 ms`
- `cached_committed_head_prefix1024_tail32` hot-prefix: `1159.389 ms`

Recent pre-refactor baselines:

- `proj-2026-04-05-0445/step7_runs/20260405T/nano_prefix_cache_ttft_initial.txt`
  - `cold_prefill_prefix256`: `293.446 ms`
  - `cold_prefill_prefix1024`: `439.427 ms`
  - `cold_prefill_prefix4096`: `1385.281 ms`
  - `cached_committed_head_prefix256_tail32` hot-prefix: `162.210 ms`
  - `cached_committed_head_prefix1024_tail32` hot-prefix: `178.329 ms`
- `proj-2026-04-05-0445/step9_runs/20260405T/post_attention_ttft.txt`
  - `cold_prefill_prefix1024`: `440.312 ms`
  - `cold_prefill_prefix4096`: `1369.694 ms`

Current slowdown versus the fresher baselines:

- `cold_prefill_prefix256`: `7.61x`
- `cold_prefill_prefix1024`: `17.67x`
- `cold_prefill_prefix4096`: `20.57x`
- `cached_committed_head_prefix256_tail32` hot-prefix: `7.10x`
- `cached_committed_head_prefix1024_tail32` hot-prefix: `6.50x`

Post-correctness update on the current working tree:

- committed-head reuse is green again after fixing the custom MoE prefill
  semantic mismatch so the prefill path returns `routed + shared`, matching the
  decode path
- the focused checks now pass again:
  - `fused_moe_prefill_test`
  - `multi_turn_prefix_reuse_test`
- the remaining standalone diagnostic
  `testing/api/split_prefill_localization_test.cpp` still reports a
  `full vs prefix-only prefix attention KV` mismatch for the isolated
  2-token prefix snapshot case, but that mismatch does not reproduce in the
  committed-head reuse path

Plan-aligned rerun on the current working tree from
`artifacts/benchmarks/ttft_20260406_tail4_prefix4_128_4096.stdout.txt`:

- cold medians:
  - `cold_prefill_prefix4`: `1133.700 ms` vs `40.415 ms` baseline (`28.05x`)
  - `cold_prefill_prefix128`: `1153.988 ms` vs `142.156 ms` baseline (`8.12x`)
  - `cold_prefill_prefix4096`: `28235.126 ms` vs `1191.797 ms` baseline (`23.69x`)
- committed-head hot-prefix medians:
  - `prefix4_tail4`: `1145.679 ms` vs `35.281 ms` baseline (`32.47x`)
  - `prefix128_tail4`: `1143.402 ms` vs `40.655 ms` baseline (`28.12x`)
  - `prefix4096_tail4`: `1190.753 ms` vs `83.871 ms` baseline (`14.20x`)
- short-tail behavior is especially informative: `prefix4_tail4` and
  `prefix128_tail4` are both pinned near `1.14 s`, which suggests a large
  roughly fixed prefill cost rather than a tail-length-specific problem
- the long-prefix cached case still gets a real cache win (`24.684x` vs cold),
  but the hot-prefix tail prefill itself is still far above baseline at
  `1155.880 ms`

Re-profile update on the clean checkpoint `22b0463`:

- `nsys` traces were captured for:
  - `artifacts/profiles/ttft_prefill_20260406/nsys_cold_prefill_prefix128.nsys-rep`
  - `artifacts/profiles/ttft_prefill_20260406/nsys_cached_committed_head_prefix128_tail4.nsys-rep`
- the cold `prefix128` trace shows the custom MoE prefill kernel still
  dominating the end-to-end wall time:
  - `FusedMoePrefillKernel`: `5602.587 ms` total over `115` launches
    (`48.718 ms` average), which is about `1120 ms` per measured benchmark
    iteration and therefore nearly the whole cold TTFT
  - the next largest GPU bucket is `MambaSsdPrefillFixedKernel<128>` at only
    `45.257 ms` total
- the cached `prefix128_tail4` trace tells the same story:
  - `FusedMoePrefillKernel`: `16790.389 ms` total over `345` launches
    (`48.668 ms` average)
  - the hot-prefix tail prefill itself was `1126.400 ms`, again pointing at the
    expert path rather than restore or decode
- `ncu` on the first `FusedMoePrefillKernel` launch from cold `prefix128`
  (`artifacts/profiles/ttft_prefill_20260406/ncu_cold_prefill_prefix128_fused_moe.csv`)
  shows a serialized kernel rather than a bandwidth-limited kernel:
  - block size `256`, grid size `128`
  - `54` registers per thread
  - `26880` bytes shared memory per block
  - occupancy limited to `3` blocks/SM by shared memory and `4` by registers
  - DRAM throughput only `2.18%` of peak
  - scheduler eligibility only about `0.09` warps per cycle active
- that profile is consistent with the code structure in
  `runtime/src/backend/fused_moe_prefill.cu`, where input quantization and both
  activation-plus-requantization stages are still serialized on `tid == 0`
- immediate next optimization step: keep the current contract and kernel shape,
  but parallelize the thread-0 quantize / `Relu2` / requantize sections before
  attempting a larger grouped-math redesign

First Priority-1 recovery step on top of `22b0463`:

- changed `fused_decode::QuantizeDequantizeNvfp4Row` to a block-parallel row
  implementation instead of a single-thread walk
- changed `FusedMoePrefillKernel` so the initial selected-expert copy and both
  `Relu2` stages are block-parallel instead of `tid == 0`
- correctness stayed green on:
  - `fused_moe_prefill_test`
  - `multi_turn_prefix_reuse_test`
- focused TTFT rerun on `prefix128` / `tail4` improved, but only modestly:
  - `cold_prefill_prefix128`: `1153.988 ms` -> `1114.050 ms` (`1.036x`, `3.46%`)
  - `cached_committed_head_prefix128_tail4`: `1143.402 ms` -> `1099.152 ms`
    (`1.040x`, `3.87%`)
- conclusion: the thread-0 sections were real, but they were not the dominant
  source of the regression. The next step should stop polishing around these
  scalar loops and move to the larger structural gap versus vLLM / TRT-LLM:
  grouped routed-expert math with token-by-expert scheduling and weight reuse

Priority-2 grouped routed-expert rewrite on top of `cdd890d`:

- replaced the one-block-per-token routed path with a grouped routed-expert
  prefill implementation in `runtime/src/backend/fused_moe_prefill.cu`
  built around `DeviceExpertRouting` plus the existing request/workspace
  scratch tensors
- the current grouped path keeps a single active implementation path and now
  does:
  - device routing into sorted token/expert order
  - grouped gather over routed rows
  - grouped routed-up and routed-down custom NVFP4 matvec launches
  - ordered top-k reduction using `selection_to_sorted`
  - separate shared-expert custom path using the same scratch discipline
- correctness stayed green on:
  - `fused_moe_prefill_test`
  - `expert_routing_device_test`
  - `multi_turn_prefix_reuse_test`
- focused TTFT rerun recorded in
  `artifacts/benchmarks/ttft_20260406_grouped_prefix128_tail4.stdout.txt`
  shows a large recovery on the plan-aligned `prefix128` / `tail4` case:
  - `cold_prefill_prefix128`: `1114.050 ms` -> `719.458 ms`
    (`1.548x`, `35.42%` faster)
  - `cached_committed_head_prefix128_tail4`: `1099.152 ms` -> `65.347 ms`
    (`16.82x`, `94.05%` faster)
- relative to the pre-refactor plan baseline:
  - `cold_prefill_prefix128`: still `5.06x` slower than `142.156 ms`
  - `cached_committed_head_prefix128_tail4`: now only `1.61x` slower than
    `40.655 ms`
- interpretation:
  - the grouped rewrite fixed most of the short-tail hot-prefix disaster
  - cold prefill is still far too slow, so the remaining bottleneck is likely
    inside the routed expert math itself rather than cache restore, snapshot,
    or the earlier thread-0 scalar sections
  - because the kernel shape has changed materially, the next step should be a
    fresh `nsys` / `ncu` pass on this grouped version before more surgery

Grouped-path reprofiling update:

- fresh cold `prefix128` `nsys` trace recorded at:
  - `artifacts/profiles/ttft_prefill_20260406_grouped/nsys_cold_prefill_prefix128.nsys-rep`
  - `artifacts/profiles/ttft_prefill_20260406_grouped/nsys_cold_prefill_prefix128.stats_cuda_gpu_kern_sum.csv`
- unlike the old one-block-per-token kernel, the grouped implementation is now
  dominated almost entirely by the custom NVFP4 matvec stages:
  - routed grouped matvec
    `Nvfp4MatVecRowsKernel<(bool)1>`: `2605.270 ms` total over `29440`
    launches, `88.494 us` average, `74.0%` of GPU time
  - shared contiguous matvec
    `Nvfp4MatVecRowsKernel<(bool)0>`: `774.534 ms` total over `230`
    launches, `3.368 ms` average, `22.0%` of GPU time
  - together they account for about `96%` of GPU kernel time on cold
    `prefix128`
- that means the current bottleneck is no longer quantize / activation glue.
  It is the custom NVFP4 math core itself.

Grouped routed-matvec `ncu` update:

- focused `ncu` report for the first routed grouped matvec launch is recorded
  at:
  - `artifacts/profiles/ttft_prefill_20260406_grouped/ncu_cold_prefill_prefix128_matvec_exact.csv`
- key metrics for
  `Nvfp4MatVecRowsKernel<(bool)1>(...)`:
  - block size `256`, grid size `1856`
  - `56` registers per thread
  - `17408` bytes shared memory per block
  - occupancy limited to `4` blocks/SM by registers and `5` by shared memory
  - `sm__warps_active.avg.per_cycle_active = 30.12`, but
    `smsp__warps_eligible.avg.per_cycle_active = 0.15`
  - `sm__issue_active.avg.pct_of_peak_sustained_elapsed = 11.80%`
  - `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed = 0.40%`
  - `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed = 15.14%`
  - tensor pipe activity is `0%`
- interpretation:
  - this kernel is not bandwidth-bound
  - it is also not using tensor cores
  - the main problem is scheduler starvation in a scalar/control-heavy matvec
    shape with poor warp eligibility

Failed micro-optimization note:

- a follow-up experiment replaced the per-tile 256-thread shared-memory
  reduction in `Nvfp4MatVecRowsKernel` with warp-level reduction plus a
  cross-warp finalize
- correctness stayed green, but TTFT regressed on
  `artifacts/benchmarks/ttft_20260406_grouped_warpreduce_prefix128_tail4.stdout.txt`
  compared with the retained grouped baseline in
  `artifacts/benchmarks/ttft_20260406_grouped_prefix128_tail4.stdout.txt`
  - `cold_prefill_prefix128`: `719.458 ms` -> `832.126 ms`
  - `cached_committed_head_prefix128_tail4`: `65.347 ms` -> `67.580 ms`
- that change was reverted
- conclusion: more reduction micro-polish on the current scalar matvec is not
  the right next bet

Updated next step:

- keep the grouped-path baseline from
  `artifacts/benchmarks/ttft_20260406_grouped_prefix128_tail4.stdout.txt`
- stop polishing around reduction details
- redesign the custom NVFP4 expert math around a more parallel launch shape:
  - fewer per-expert launches
  - better cross-token reuse inside the routed-up / routed-down kernels
  - a tiled/grouped math core that can materially increase warp eligibility
    and instruction issue rate

Single-launch grouped-expert follow-up:

- implemented a device-resident routed weight-view table so the grouped routed
  matvec pass can launch once per routed-up or routed-down stage instead of
  once per expert
- this does not add any runtime DtoH transfer:
  - the routed weight views are already known on the host at layer
    construction time
  - they are uploaded once to device memory during setup
  - the active runtime path uses only the device routing buffers, device
    scratch buffers, and the device weight-view table
- follow-up cleanup removed the per-call host-view dependency from
  `RunFusedMoePrefill`; the runtime prefill contract is device-only again
- focused TTFT rerun recorded at
  `artifacts/benchmarks/ttft_20260406_grouped_singlelaunch_prefix128_tail4.stdout.txt`
  improved further over the retained grouped baseline:
  - `cold_prefill_prefix128`: `719.458 ms` -> `681.460 ms`
    (`1.056x`, `5.28%` faster)
  - `cached_committed_head_prefix128_tail4`: `65.347 ms` -> `52.054 ms`
    (`1.255x`, `20.34%` faster)
- relative to the pre-refactor plan baseline:
  - `cold_prefill_prefix128`: now `4.79x` slower than `142.156 ms`
  - `cached_committed_head_prefix128_tail4`: now only `1.28x` slower than
    `40.655 ms`
- interpretation:
  - per-expert launch fan-out was a real part of the remaining regression
  - cached tail prefill is getting close to baseline
  - cold prefill is still dominated by the scalar custom NVFP4 math kernel, so
    the next step should target the math core itself rather than launch count

Reference-scope note:

- vLLM comparison in this plan uses the CUDA FP4 fused-MoE stack plus the
  Nemotron-H shared-expert wiring, which is the closest in-tree match to the
  Nano routed-plus-shared shape.
- TRT-LLM comparison on this platform should be treated as CUTLASS-path
  guidance. Its `TRTLLMGenFusedMoE` min-latency backend explicitly excludes
  `SM120`, so it is useful for operation structure but not as a direct
  same-platform execution path for the local RTX 5090 target.

## Working Hypothesis

The new contract still looks directionally right:

- no host `expert_offsets` readback
- no host loop over active experts
- no cuBLASLt plan churn on the hot path
- no runtime requirement to pack full-token activations for a library call

The likely failure is implementation quality inside the new kernel, not the
top-level contract. The benchmark counters still show the same layer run counts
as the old path, which argues against a scheduler explosion or a duplicated
high-level execution path.

The immediate suspicion is that the current custom kernel is spending most of
its time in scalarized per-token work:

- one block per token
- serial expert loop inside the block
- thread-0-only quantization and activation passes
- no grouped/token-batched expert GEMM
- no overlap between routed and shared expert work

## Benchmark Contract

Primary checkpoint cases:

- `cold_prefill_prefix1024`
- `cold_prefill_prefix4096`
- `cached_committed_head_prefix1024_tail32`
- `cached_committed_head_prefix4096_tail32`

Fast iteration cases:

- `cold_prefill_prefix256`
- `cached_committed_head_prefix256_tail32`
- a dedicated fused-MoE microbench with representative Nano dimensions

Success gates:

1. recover to at least the pre-refactor baseline
2. beat the baseline on `cold_prefill_prefix1024` and
   `cached_committed_head_prefix1024_tail32`
3. keep correctness green on focused dummy-data and manifest-smoke coverage

Stop conditions:

- If a profiling pass shows the regression is dominated by a single operation,
  optimize that operation before touching anything else.
- If two successive optimization steps each improve TTFT by less than `10%`,
  and the profile still shows the same dominating scalar hotspot, stop and
  redesign that operation boundary instead of polishing around it.
- If the current one-block-per-token design cannot get within `20%` of the old
  baseline after fixing clear serialization problems, treat that as evidence
  that the grouping/scheduling shape must change.

## Operation Catalog

This section is the backbone of the work. Every optimization step must map to
one of these operations.

### 1. Routing Contract Ingress

Our code now assumes routing is already done before fused expert execution:

- inputs: `selected_indices`, `selected_weights`, `input`, `normalized`
- no router logits inside the kernel

Current implementation:

- our kernel only copies `top_k` expert ids and weights into block-shared state
- no internal routing or token regrouping

Reference implementations:

- vLLM keeps routing separate but uses fused/grouped top-k helpers when
  available:
  - `vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py`
- TRT-LLM supports both "routing outside the fused op" and an integrated
  min-latency form:
  - `tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py`
  - `tensorrt_llm/_torch/modules/fused_moe/fused_moe_trtllm_gen.py`

Assessment:

- this part is probably not the main regression source
- keep the current contract stable unless profiling proves otherwise

### 2. Token-to-Expert Work Decomposition

Current implementation:

- one CUDA block per token
- serial loop over `top_k` routed experts inside the block
- shared expert runs in the same block after routed experts finish

Reference implementations:

- vLLM converts `topk_ids` into expert-grouped token order using
  `moe_align_block_size()`:
  - `vllm/model_executor/layers/fused_moe/moe_align_block_size.py`
- vLLM then shuffles rows and executes grouped expert matmuls over all routed
  tokens for an expert set:
  - `vllm/model_executor/layers/fused_moe/cutlass_moe.py`
- TRT-LLM follows the same high-level shape:
  - routing
  - scatter / token grouping
  - grouped `gemm1`
  - activation
  - grouped `gemm2`
  - finalize route

Assessment:

- this is the most likely architectural performance gap
- our current kernel gives up all cross-token reuse
- if profiling shows routed up/down projections dominate, we should expect this
  operation to become the main redesign target

### 3. Input Quantization / Activation Formatting

Current implementation:

- thread `0` runs `QuantizeDequantizeNvfp4Row(normalized_row, quantized_input)`
- the entire block waits at a barrier

Reference implementations:

- vLLM quantizes expert inputs in dedicated kernels over grouped routed rows:
  - `scaled_fp4_experts_quant`
  - `silu_and_mul_scaled_fp4_experts_quant`
  - invoked from `cutlass_moe.py`
- TRT-LLM describes this as dynamic quantization before or inside the fused MoE
  operator depending on backend and mode:
  - `fused_moe_cutlass.py`
  - `fused_moe_trtllm_gen.py`

Assessment:

- our contract to avoid host-side pack/scaling is fine
- the current implementation is almost certainly too serialized
- first optimization candidate inside the existing kernel shape: parallelize
  row quantization across the block and measure how much that alone recovers

### 4. Routed Up Projection

Current implementation:

- each thread computes different rows of the routed expert intermediate
- dot products are scalar row-major loops via `Nvfp4RowMajorDot`
- no cross-token batching
- no tensor-core grouped matmul

Reference implementations:

- vLLM uses grouped FP4/CUTLASS MoE matmuls over shuffled tokens:
  - `vllm/model_executor/layers/fused_moe/cutlass_moe.py`
- TRT-LLM CUTLASS path follows the same grouped `gemm1` structure:
  - `tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py`

Assessment:

- this is a prime suspect for the catastrophic slowdown
- if ncu shows low tensor utilization or massive instruction count in this
  operation, the fix is not micro-polish; it is a real custom grouped matmul
  step

### 5. Routed Activation

Current implementation:

- thread `0` loops over the entire routed intermediate buffer
- applies `Relu2`
- then re-quantizes the same buffer in place

Reference implementations:

- vLLM either fuses activation with quantization or runs vectorized activation
  before quantization:
  - `cutlass_moe.py`
- TRT-LLM documents activation as a distinct step inside the fused path, not a
  thread-0 scalar loop

Assessment:

- this is another obvious serialization point
- optimize here before changing less important code

### 6. Routed Down Projection + Weighted Reduction

Current implementation:

- each routed expert contribution is projected back to hidden size
- routing weight is applied per column
- accumulation is done directly into `output_row`

Reference implementations:

- vLLM runs grouped second-stage expert GEMM, then reshuffles rows back and
  performs weighted reduction:
  - `cutlass_moe.py`
- TRT-LLM uses grouped `gemm2` followed by finalize-route

Assessment:

- direct accumulation is a reasonable contract
- but the current per-token/per-expert scalar projection is a likely hotspot
- if this dominates, direct accumulation should stay but the math kernel must
  change

### 7. Shared Expert Execution

Current implementation:

- shared up/down work is embedded into the same per-token block after routed
  experts complete
- it reuses the same scratch and the same one-token execution shape

Reference implementations:

- vLLM keeps shared experts logically separate:
  - `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`
- TRT-LLM `SharedMoE` also computes routed and shared outputs as separate
  components and then combines them:
  - `tensorrt_llm/layers/moe.py`

Assessment:

- the current fused treatment may be correct functionally but is not aligned
  with the strongest reference shapes
- we should profile routed-only and shared-only cost separately
- likely optimization direction: split shared expert execution back out so it
  can be tuned and optionally overlapped independently

### 8. Final Output Writeback

Current implementation:

- initialize `output_row = input_row`
- add routed weighted contributions
- add shared contribution
- optional debug outputs for routed/shared contributions

Reference implementations:

- both vLLM and TRT-LLM finalize after the routed path and then combine shared
  expert output outside the grouped expert math core

Assessment:

- this is not the first place to optimize
- keep it simple unless profiling shows unexpected memory traffic

## Profiling Strategy

### Phase A. Controlled Benchmark Confirmation

Before new profiles, rerun only:

- `cold_prefill_prefix256`
- `cold_prefill_prefix1024`
- `cached_committed_head_prefix256_tail32`
- `cached_committed_head_prefix1024_tail32`

Requirements:

- isolate single-case runs
- no concurrent GPU jobs
- record exact command lines and output artifacts in this project directory

Purpose:

- verify the regression is stable and reproducible
- reduce turnaround time versus the full sweep

### Phase B. End-to-End `nsys`

Profile:

- `cold_prefill_prefix1024`
- `cached_committed_head_prefix1024_tail32`

Questions to answer:

- what fraction of TTFT is now in fused MoE prefill
- is the cost dominated by routed path, shared path, or both
- are there still hidden syncs or allocator churn around the new kernel
- how does the kernel-time split compare to
  `proj-2026-04-05-0445/post_attention_root_cause.md`

Output:

- one short root-cause note with before/after kernel-time tables

### Phase C. `ncu` on the New Kernel

Use a reduced, representative case with Nano dimensions and one or a few tokens.

Questions to answer:

- occupancy
- registers per thread
- shared-memory pressure
- local-memory spills
- tensor core usage or lack of it
- memory throughput
- dominant warp stall reasons
- instruction mix in `Nvfp4RowMajorDot` and the thread-0 quantize/activation
  sections

Purpose:

- decide whether the next step is a local parallelization fix or a math-kernel
  redesign

### Phase D. Add Micro Instrumentation

Create narrow timing hooks or micro tests around these operations:

- input row quantize/dequantize
- routed up projection
- routed activation + re-quantize
- routed down projection + weighted accumulate
- shared up/down projection

Rules:

- dummy data only
- no host copies in the timed region
- keep one micro test per operation so improvements are attributable

## Optimization Order

Optimize in descending order of likely impact and lowest redesign cost.

### Priority 1. Remove obvious serialization in the existing kernel

Candidates:

- parallelize input quantization across the block
- parallelize `Relu2`
- parallelize intermediate re-quantization
- reduce unnecessary barriers

Checkpoint after each change:

- run focused micro test
- run `fused_moe_prefill_test`
- rerun `cold_prefill_prefix256`
- rerun `cached_committed_head_prefix256_tail32`

### Priority 2. Decide whether one-block-per-token is salvageable

If Priority 1 does not recover a large fraction of the regression:

- prototype a custom grouped routed-up operation
- keep the current external contract
- move from per-token scheduling to token-grouped-by-expert scheduling

This is the likely pivot point. The references do not preserve the current
one-token scheduling shape for the expert GEMMs.

### Priority 3. Split shared experts from routed experts

If routed and shared cost are materially different:

- move shared expert execution into a separate custom path
- keep the final combine explicit
- tune routed and shared paths independently

### Priority 4. Replace scalar dot-product math with custom grouped math

If `Nvfp4RowMajorDot` dominates:

- write a real custom grouped expert matmul path
- retain the Step 3 contract
- add a dedicated microbench for:
  - grouped routed up
  - grouped routed down
  - shared up/down

## Validation Strategy

After every optimization step:

1. micro test for the touched operation
2. `fused_moe_prefill_test`
3. `expert_layer_fastpath_test`
4. `full_forward_manifest_smoke_test`
5. targeted TTFT rerun on the fast cases

Before claiming recovery:

1. rerun the full TTFT sweep
2. rerun throughput bench
3. rerun full `ctest`
4. sanity-check the interactive conversation utility

## Point-by-Point Review

1. The current regression is too large for blind tuning. Profiling first is
   mandatory.
2. The run-count parity with earlier artifacts suggests the contract is not
   fundamentally broken at the scheduler level.
3. The current kernel has multiple thread-0-only sections. Those are immediate
   targets and easy to validate with micro tests.
4. The biggest structural gap versus vLLM and TRT-LLM is token grouping for
   expert GEMMs. We should expect that to matter.
5. Shared experts should not stay fused into the same scheduling shape unless
   profiling proves the coupling is helping.
6. The stop conditions are explicit. If the current schedule shape cannot get
   close to baseline, we should stop polishing and replace the offending
   operation with a better custom kernel.
7. Baseline recovery is not the end state. The finish line for this phase is
   "better than the old baseline", not merely "less bad than today".

## First-Principles Roofline Snapshot

This section is the current reality check for the custom MoE prefill path on
the Nano 30B configuration and the local RTX 5090 target.

### Nano expert dimensions

From `runtime/src/api/single_token_forward_model.cpp`:

- hidden size: `2688`
- routed expert intermediate size: `1856`
- shared expert intermediate size: `3712`
- routed experts: `128`
- routed experts selected per token: `6`
- expert layers per forward pass on the current benchmark surface: `23`

### Expert math per token

Per expert layer, the prefill path computes:

- routed up + down: `2 * hidden * routed * top_k`
- shared up + down: `2 * hidden * shared`

For the Nano configuration, that is:

- `79,822,848` FMAs per token per expert layer
- `159,645,696` FLOPs per token per expert layer

For `prefix128` across `23` expert layers, that is:

- `234,998,464,512` FMAs
- `469,996,929,024` FLOPs

### Expert weight traffic lower bound

If the grouped kernel shape is working well, the routed path should stream each
active expert weight once per layer and reuse it across the routed tokens
assigned to that expert. For a `prefix128` batch on this model, that usually
means essentially all `128` routed experts are active.

With packed FP4 weights plus one FP8 block scale per `16` weights, the storage
cost is approximately `0.5625` bytes per weight.

That implies:

- routed + shared weights per expert layer: about `695.83 MiB`
- routed + shared weights across all `23` expert layers: about `15.63 GiB`

Using the RTX 5090 Blackwell memory-bandwidth figure of `1.792 TB/s`, the
ideal expert-weight streaming lower bound for `prefix128` is only about
`9.36 ms`.

Source:

- NVIDIA RTX Blackwell architecture appendix listing `1792 GB/sec` memory
  bandwidth for GeForce RTX 5090:
  https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf

### Compute lower bound

Using the same NVIDIA architecture appendix, the RTX 5090 peak tensor rates
are vastly above the current effective throughput:

- peak BF16 tensor throughput with FP32 accumulate: `209.5 TFLOP/s`
- peak INT8 tensor throughput: `1676 TOPS`

Even if we pessimistically compare our expert path against the BF16 tensor
figure, the raw compute lower bound for the `prefix128` expert work is about
`2.24 ms`. So the physical floor for this path is not set by arithmetic
capacity. It is set by moving the expert weights efficiently and keeping the
machine busy while doing so.

### What the current profile says

From the retained grouped single-launch run:

- `cold_prefill_prefix128 = 681.460 ms`
- `cached_committed_head_prefix128_tail4 hot-prefix = 52.054 ms`

From `artifacts/profiles/ttft_prefill_20260406_singlelaunch/`:

- total GPU kernel time per measured cold `prefix128` iteration: about
  `674.392 ms`
- grouped routed expert matvec:
  `Nvfp4GroupedExpertMatVecRowsKernel = 72.9%`
- shared expert matvec:
  `Nvfp4MatVecRowsKernel<(bool)0> = 23.0%`
- total expert matvec time per measured cold `prefix128` iteration: about
  `646.571 ms`
- all other GPU kernels combined: about `27.821 ms`

That implies:

- current effective expert throughput: about `0.73 TFLOP/s`
- current effective expert-weight streaming rate: about `24.2 GiB/s`

Relative to hardware capability, both numbers are tiny. Relative even to the
old `142.156 ms` baseline, the current cold path is still far from the limit.
The old baseline itself only implied roughly `110 GiB/s` of effective
expert-weight streaming, so it was not especially close to hardware roofline
either.

### Conclusion from first principles

The current `4.79x` cold regression is not because the old code was already
optimal. It is because the current grouped path still uses a scalar dot-product
math core that leaves most of the GPU idle.

The `ncu` data already says the same thing from the scheduler side:

- tensor pipe activity is effectively `0%`
- DRAM throughput is only `0.40%` of peak
- issue rate is low
- warp eligibility is poor

So the path forward is not "make this scalar matvec a bit nicer." The path
forward is to replace the math core with a launch shape that creates enough
parallel work, enough reuse, and enough regularity for the hardware to run near
its real limits.

## Current Focus: Math-Core Recovery Strategy

The earlier priority steps already removed the obvious serialization and
collapsed per-expert launch fan-out. The remaining work is now centered on the
math core itself.

### Operation target shape

Our current `RunFusedMoePrefill()` path still looks like this:

1. route and sort token selections
2. gather routed rows into `selection_count x hidden`
3. quantize/dequantize routed rows
4. grouped routed-up scalar matvec
5. activation + requantization
6. grouped routed-down scalar matvec
7. weighted reduction back to token order
8. separate shared up/down scalar matvec path
9. accumulate shared output into final token output

The TRT-LLM reference shape is materially different:

1. routing / renormalize
2. permute tokens into expert-major tiles
3. grouped `gemm1`
4. activation / quantization
5. grouped `gemm2`
6. finalize / unpermute / combine

That shape appears in:

- `third_party/TensorRT-LLM/cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.cu`
- `third_party/TensorRT-LLM/cpp/tensorrt_llm/kernels/cuteDslKernels/moeUtils.h`

The next phase should move our implementation toward that structure while
preserving a single device-only runtime contract.

### Architectural objective

For the active prefill path, the core kernel stack should eventually satisfy
all of these:

- one runtime code path
- no DtoH on the hot path
- token-to-expert regrouping fully on device
- expert-major or tile-major work decomposition
- grouped routed-up and routed-down math
- enough work per launch to keep warps eligible
- enough regularity to make tensor-core or tensor-like batched math viable
- explicit finalize/unpermute step instead of implicit scalar accumulation in
  the matvec kernel

### Priorities from here

1. Treat the current scalar grouped matvec as the baseline to beat, not the
   structure to preserve.
2. Build a dedicated operation-by-operation microbench for the expert math
   core:
   - route/permute only
   - grouped routed-up only
   - activation/quant only
   - grouped routed-down only
   - finalize/unpermute only
3. Replace routed-up first, because it is the largest cross-token reuse
   opportunity and the cleanest place to prove a new tiled math core.
4. Then replace routed-down with the same scheduling discipline.
5. After the routed path is stable and fast, re-evaluate whether the shared
   path should use the same grouped math core or a simpler specialized path.
6. Do not spend more time on reduction micro-polish unless a new profile shows
   that a replacement math core has already shifted the bottleneck elsewhere.

### Stop conditions for the current scalar kernel family

The current scalar grouped-matvec family should be considered finished unless
it can beat these bars:

- materially higher warp eligibility
- materially higher issue rate
- materially higher effective expert-weight streaming rate
- clear progress toward the old `142.156 ms` cold `prefix128` baseline

If a change does not move those metrics, it is not the right kind of change.
