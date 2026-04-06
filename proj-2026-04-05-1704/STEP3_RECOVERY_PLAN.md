# Step 3 Recovery Plan

Project directory: `./proj-2026-04-05-1704`

## Goal

Recover from the current custom-MoE-prefill regression, validate that the new
device-side contract is still the right architectural direction, and then push
past the best pre-refactor TTFT numbers.

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
