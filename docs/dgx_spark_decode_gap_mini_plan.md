# DGX Spark Decode Gap Mini-Plan

IMPORTANT: implement this plan continuously without stopping unless a hard blocker is encountered that cannot be overcome within the current environment.

## Current Context

The runtime is now functionally correct and the oracle-backed forward path is green, but the real 16-token append-only decode path is still far from DGX Spark community / NVIDIA-recipe throughput.

Current baseline artifact:

- [single_token_decode_20260331T090018Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260331T090018Z_cuda132.json)

Current measured shape:

- `environment_build_ms ≈ 2039`
- `model_build_ms ≈ 10558`
- first `4` decode tokens: about `5015 ms/token`
- last `8` decode tokens: about `999 ms/token`
- last `4` decode tokens: about `937 ms/token`
- generated output tokens remain stable on the current short-chat fixture

So the remaining gap to `~14 tok/s` is not a TTFT problem anymore. It is a steady-state serving-path problem.

## Reconciled Findings

### Local profile / codebase findings

- The local throughput profile attributes the largest measured budget to MoE, then Mamba, then attention.
- The measured 16-token run shows a large early-token cliff plus a poor steady-state tail, which means persistent decode behavior matters more than 1-token microbench numbers.
- The grouped routed-expert pass helped, but not nearly enough. Launch-count cleanup alone is not the answer.
- Dense / scaled-FP8 fallback incidence is still a likely major gap, and the current runtime still spends too much time in correctness-first surfaces.
- Attention still rebuilds too much decode-local state per token.

### Official / external framework findings

The current published DGX Spark guidance for Nemotron 3 Super NVFP4 points to:

- vLLM:
  - FlashInfer MoE latency backend
  - CUDA graph capture at small interactive batch sizes
  - FlashInfer attention warmup / autotune
  - MTP enabled for the large throughput jump
  - Mamba SSM cache kept at `float32` for released Nemotron 3 Super checkpoints
- TensorRT-LLM:
  - CUDA graph config enabled by default on the recommended path
  - `CUTLASS` MoE backend on DGX Spark
  - model transforms such as `multi_stream_moe`, `insert_cached_ssm_attention`, `fuse_nvfp4_moe`, and `fuse_fp8_moe`
  - speculative decoding / MTP as part of the high-throughput path

The important implication is that the production stacks are not relying on one trick. They combine:

1. persistent / captured decode execution
2. fused or backend-specialized MoE
3. warmed persistent attention / SSM machinery
4. speculative decoding for the final large multiplier

The local follow-up to that analysis matters too:

- the current grouped cuBLASLt pointer-array NVFP4 path is not landing on this CUDA 13.2 / GB10 stack
- after lookup-table and aligned-plan fixes, the grouped routed-expert path now reaches the actual pointer-array NVFP4 grouped call and fails there on first use in every expert layer
- the latest grouped debug probe shows the failure is at heuristic selection, not later execution:
  - `cublas_status = 7`
  - `heuristic_results = 0`
  - real routed shape: `batch_count = 22, m = 1, n = 2688, k = 1024`
- that means grouped cuBLASLt dispatch is at best an intermediate step here, not the end-state MoE backend
- the practical production target remains a fused/backend-specialized MoE path closer to the FlashInfer / TRT-LLM recipes described in [reference_framework_analysis.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/docs/reference_framework_analysis.md)

## Strategic Alignment

The current local custom fused routed-MoE path is now an interim backend, not the final serving target.

The primary next throughput path is:

1. integrate FlashInfer CUTLASS fused NVFP4 MoE as described in [flashinfer_moe_integration_plan.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/docs/flashinfer_moe_integration_plan.md)
2. keep the current custom fused routed-MoE path as the correctness/fallback backend
3. add CUDA graph capture only after the FlashInfer serving path is device-only and graph-safe

The most useful upstream implementation detail is now explicit:

- the production-aligned target on Spark is the FlashInfer `cutlass_fused_moe` path, not the TRT split runner
- the CUTLASS path accepts precomputed top-k ids and weights and can consume the original packed NVFP4 expert tensors plus raw block scales
- this matters because it matches the monolithic-kernel architecture described in [reference_framework_analysis.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/docs/reference_framework_analysis.md) without forcing a second shuffled-major-K resident weight copy
- the earlier repo-local TRT split experiment was still useful because it proved the plugin seam, but it also proved the wrong end-state on Spark:
  - it instantiated and ran on real layers
  - then it drove `model_build_ms` to roughly `87s`
  - and it later died with `NVRM ... NV_ERR_NO_MEMORY` / `Xid MMU Fault` because the backend duplicated routed expert residency in a second layout

That changes the interpretation of the current repair work:

- routed lookup repair is still the blocker for graph capture on the current custom backend
- but the more important strategic point is that we should stop treating the current custom backend as the final MoE design
- the current custom backend is still useful because it keeps the runtime correct, gives us an exact 16-token gate, and provides a fallback while FlashInfer integration lands

## Adjusted Four-Win Focus

### Win 1: Eliminate serving-path reference fallback incidence

Why first:

- If dense BF16 / scaled-FP8 ops are still falling back to correctness-first reference surfaces, CUDA graphs and further fusion will not close the gap.
- We need direct measurement of how often the runtime is still missing native execution.

Required outcomes:

- add explicit serving-path counters for dense / scaled-FP8 native success vs. fallback
- make the 16-token benchmark emit those counters
- reduce fallback incidence materially on the hot decode path before moving graphs to the front of the queue
- keep unsafe dense-native experiments scoped and reject any path that changes the 16-token output stream even if it lowers fallback counts

### Win 2: Make attention decode persistent and graph-safe

Why second:

- The current attention decode path still rebuilds plan state and page-table layout that production stacks keep persistent.
- Official guidance and framework code both point to persistent attention state and CUDA graph compatibility.

Required outcomes:

- cache decode-time cuDNN paged-attention plan state at the layer level
- remove host-side batch-plan / page-table rebuilds from the batch-1 decode path
- keep decode aux/state on device and make the batch-1 decode shape reusable across tokens

### Win 3: Replace the interim routed-expert backend with FlashInfer CUTLASS fused MoE

Why third:

- MoE is still the largest measured family.
- The grouped cuBLASLt NVFP4 route is dead on this stack.
- The TRT split FlashInfer route is also now demoted: it instantiated, but its shuffled-weight residency model is the wrong memory trade on Spark.
- The current custom fused backend is materially better than the old path, but it is still an interim local backend, not the community-validated production route.
- FlashInfer CUTLASS fused MoE is the first path that is plausibly aligned with the `~14 tok/s` recipe stack on DGX Spark.

Required outcomes:

- build or import the FlashInfer SM121 CUTLASS fused-MoE backend described in [flashinfer_moe_integration_plan.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/docs/flashinfer_moe_integration_plan.md)
- feed that backend the original packed NVFP4 routed expert tensors plus raw block-scale tensors, avoiding the rejected TRT-style second shuffled resident layout
- make the default serving routed-expert path call the fused FlashInfer CUTLASS kernel
- keep the current custom fused routed-MoE backend as the fallback/oracle path until FlashInfer is fully accepted on the 16-token gate

### Win 4: Add steady-state decode capture / speculation scaffolding

Why fourth:

- The official stacks rely on CUDA graphs and MTP, but they do so on top of the fused production MoE backend, not on top of the current interim local backend.
- We should only capture the steady-state decode once the FlashInfer routed-expert path is active and no longer crossing the host boundary.

Required outcomes:

- add a batch-1 steady-state decode capture path on top of the FlashInfer-backed serving route
- measure real 16-token tail improvement
- stage the runtime so MTP / speculative decode can be layered on top next rather than treated as a separate redesign

## Validation Rule

After each substantive implementation step:

1. run the 16-token decode benchmark
2. compare generated output tokens against the known-correct baseline
3. reject or revert changes that break token correctness
4. record the new artifact and update progress docs

The correctness bar for this mini-plan is output-token agreement on the 16-token test, not just microbench speed.

## Memory Safety Rule

DGX Spark is a UMA platform and earlier runs on this host have already triggered real OOM storms severe enough to kill user-session services and `tailscaled`. So decode / startup benchmarking now has to follow a hard operating rule:

1. run only one heavy model-build or decode benchmark at a time
2. capture host plus CUDA memory snapshots before environment build, after environment build, after model build, and after decode
3. treat `host_budget_bytes = MemAvailable + SwapFree` as the primary safety signal, with `cudaMemGetInfo()` only advisory
4. keep a warning threshold of at least `20 GiB` host budget for Spark experiments
5. use `NEMOTRON_BENCH_ABORT_ON_LOW_HOST_BUDGET=1` for unattended or long-running decode studies so unsafe runs fail fast instead of destabilizing the box

The benchmark path now emits these snapshots in JSON and on `stderr`, so no new decode optimization pass should proceed without checking the memory snapshots alongside the token outputs.

## Immediate Execution Order

1. Treat [flashinfer_moe_integration_plan.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/docs/flashinfer_moe_integration_plan.md) as the primary MoE throughput plan.
2. Land the FlashInfer CUTLASS dependency/build/bootstrap path for SM121 and wire a runtime MoE backend abstraction around it.
3. Replace the old shuffled-layout seam with the raw packed NVFP4 + raw scale seam that the CUTLASS backend can consume without duplicate routed expert residency.
4. Validate every step on the 16-token output stream, keeping the current custom fused routed-MoE path as fallback.
5. Only after FlashInfer CUTLASS is the accepted serving path, add CUDA graph capture on the steady-state decode tail.
6. Keep memory telemetry and low-budget guard rails enabled for every heavy run.

## Current Status

Completed so far:

- runtime decode counters and JSON reporting are implemented
- persistent batch-1 attention decode plan reuse is implemented
- device-side attention page-table assembly is implemented
- request-local token-ID device storage plus device-ID embedding lookup are implemented
- serving-path token-ID and decode-attention control uploads now use async/default-stream copies
- the default routed MoE decode path now uses the custom fused packed-NVFP4 kernels end to end
- full expert-selection metadata downloads are gone from the default 16-token decode path
- decode benchmark graph-readiness summaries are emitted in JSON and stdout
- there is now a routed MoE backend abstraction plus plugin-style FlashInfer seam in the runtime
- the runtime now has a CUTLASS-aligned raw FlashInfer weight seam for routed experts:
  - original packed NVFP4 weight bytes
  - raw linear block scales
  - tensor-scale / dequant-scale metadata
- the build now detects the local FlashInfer SM120 CUTLASS generated source cache
- the repo also has deterministic routed-layout prep utilities for the older TRT path, but those are now diagnostic/reference utilities rather than the primary serving target
- the in-tree FlashInfer plugin is still the older TRT split adapter and now explicitly rejects the new raw CUTLASS seam
- the plugin/runtime seam now has an explicit backend-kind hook so a future CUTLASS plugin can switch the runtime to `cutlass_raw` automatically
- there is still no accepted local FlashInfer CUTLASS serving backend in the runtime, so the remaining concrete integration step is adapter build around the CUTLASS runner, not the rejected TRT split runner

Latest FlashInfer plugin findings:

- forced FlashInfer/TensorRT split backend one-token build:
  - instantiated on real routed expert layers
  - drove `model_build_ms` to about `87320`
  - left only about `3.3 GiB` CUDA free after model build
  - then died with `NVRM ... NV_ERR_NO_MEMORY` and later `Xid MMU Fault`

Interpretation:

- the plugin seam itself is real
- the TRT split FlashInfer backend is not the right serving backend for Spark
- the next blocker is no longer uncertainty about the seam; it is landing the CUTLASS fused backend without the rejected duplicate shuffled-weight residency model

Current best validated 16-token artifact:

- [single_token_decode_20260331T_graph_readiness_cleanup_16tok.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260331T_graph_readiness_cleanup_16tok.json)

Measured shape on that artifact:

- full 16-token sequence: `~15897.4 ms`
- first `4` tokens: `~1936.5 ms/token`
- last `8` tokens: `~614.9 ms/token`
- last `4` tokens: `~608.6 ms/token`
- output tokens: `[5130 x16]`

Important runtime counters on that artifact:

- `dense_native_success = 992`
- `dense_reference_fallbacks = 1824`
- `scaled_fp8_native_success = 0`
- `scaled_fp8_reference_fallbacks = 2176`
- `attention_decode_plan_creates = 8`
- `attention_decode_plan_hits = 120`
- `attention_decode_device_page_table_copies = 128`
- `expert_selection_metadata_downloads = 0`
- `routed_lookup_repair_downloads = 395`
- `routed_lookup_repair_experts = 3130`
- `grouped_routed_expert_fastpath_uses = 640`
- `hot_graph_safe_steps = 0`
- `hot_max_graph_safe_streak = 0`
- `hot_tail_graph_safe_streak = 0`
- `hot_first_graph_safe_tail_token_index = -1`

That changed the working read:

- attention persistence was a real serving-path win
- fused routed-MoE dead-weight cleanup was another real serving-path win
- dense/scaled-FP8 fallback incidence is still far too high
- scaled-FP8 native execution is still not landing on the current serving path
- the next large blocker is now routed lookup repair, not another local attention or Mamba cleanup
- strategically, the next major backend step is FlashInfer fused MoE, not more incremental tuning of the interim local backend

Latest dense-family counter artifact:

- [single_token_decode_20260331T105916Z_dense_family_counters.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260331T105916Z_dense_family_counters.json)

What it showed on an output-correct 16-token run:

- `dense_plan_build_failures = 112`
- `dense_plan_build_failures_attention = 16`
- `dense_plan_build_failures_expert = 79`
- `dense_plan_build_failures_other = 17`
- `dense_reference_fallbacks = 1824`
- `dense_reference_fallbacks_attention = 288`
- `dense_reference_fallbacks_expert = 1264`
- `dense_reference_fallbacks_other = 272`

Interpretation:

- the dense fallback cluster is primarily MoE control, not attention
- eliminating attention dense fallbacks alone is not enough to move steady-state decode materially
- the next large dense/native correctness problem is in expert-control surfaces

Rejected experiment:

- [single_token_decode_20260331T_runtime_stats_scaled_fp8_mamba_fastpath.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260331T_runtime_stats_scaled_fp8_mamba_fastpath.json)

Why rejected:

- output tokens changed to `[72773 x16]`
- full 16-token sequence regressed to `~34350.2 ms`
- therefore the dequantized-dense scaled-FP8 serving attempt is not an acceptable default path

Latest graph-prep cleanup result:

- [single_token_decode_20260331T_device_token_ids_async_control.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260331T_device_token_ids_async_control.json)

Latest FlashInfer backend seam probe:

- [single_token_decode_20260331T_flashinfer_backend_probe_16tok.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260331T_flashinfer_backend_probe_16tok.json)

What it showed:

- exact generated token stream stayed `[5130 x16]`
- `flashinfer_routed_expert_uses = 0`
- `grouped_routed_expert_fastpath_uses = 640`
- interpretation: forcing `NEMOTRON_ROUTED_MOE_BACKEND=flashinfer` currently falls back cleanly to the accepted custom fused backend because there is still no compatible FlashInfer plugin/library installed locally

Takeaway:

- output tokens stayed correct: `[5130 x16]`
- the token-ID reuse plus async control-copy cleanup did not materially improve tail latency on its own
- it stays because it removes graph blockers, not because it was a standalone throughput win

Latest MoE counter result:

- [single_token_decode_20260331T_moe_runtime_stats.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260331T_moe_runtime_stats.json)

What it showed on the same 16-token output-correct run:

- `expert_selection_metadata_downloads = 640`
- `routed_expert_materializations = 6238`

Interpretation:

- the grouped MoE path is still crossing the host boundary constantly
- lazy routed-expert materialization is still a dominant early-horizon decode event
- that makes the next implementation target unambiguous: fix MoE control/materialization before expecting CUDA graph capture to move the tail materially

Latest graph-readiness result:

- [single_token_decode_20260331T_graph_readiness_cleanup_16tok.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260331T_graph_readiness_cleanup_16tok.json)

What it showed:

- the fused routed-MoE cleanup materially improved decode even though the repair curve did not change
- there is still no graph-safe decode step in the 16-token horizon
- so immediate post-first-token CUDA graph capture is still premature on the current default path
- that means the correct ordering is:
  1. FlashInfer fused MoE integration
  2. then CUDA graph capture

Latest dense-native experiment results:

- all dense families on device-plan surface:
  - [single_token_decode_20260331T105510Z_dense_device_plan_surface_lifetime_fix.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260331T105510Z_dense_device_plan_surface_lifetime_fix.json)
  - `dense_reference_fallbacks = 0`
  - output token changed to `72773`
- expert-only device-plan surface:
  - [single_token_decode_20260331T110042Z_dense_device_plan_surface_expert_only.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260331T110042Z_dense_device_plan_surface_expert_only.json)
  - `dense_reference_fallbacks_expert = 0`
  - output token changed to `72773`
- attention-only device-plan surface:
  - [single_token_decode_20260331T110131Z_dense_device_plan_surface_attention_only.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260331T110131Z_dense_device_plan_surface_attention_only.json)
  - `dense_reference_fallbacks_attention = 0`
  - output token stayed `5130`
  - no material 16-token throughput win

Current read:

- the dense device-plan surface bug was not just descriptor lifetime
- expert-control dense-native execution is still numerically unsafe as a default serving path
- attention-native dense execution is safe but not where the large remaining decode gap lives
- the next effective focus remains MoE control/backend work plus scaled-FP8/Mamba fallback reduction, not another attention-only pass

Latest grouped-MoE backend follow-up:

- reordered device-side grouped lookup probe:
  - [single_token_decode_20260331T_device_grouped_lookup_reuse_reordered.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260331T_device_grouped_lookup_reuse_reordered.json)
  - output tokens stayed exact: `[5130 x16]`
  - `hot_mean_ms ≈ 25694.1`
- grouped NVFP4 global-disable follow-up:
  - [single_token_decode_20260331T_grouped_nvfp4_global_disable.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260331T_grouped_nvfp4_global_disable.json)
  - output tokens stayed exact: `[5130 x16]`
  - `hot_mean_ms ≈ 27679.9`
  - `expert_selection_metadata_downloads = 640`
  - `routed_expert_materializations = 6238`
  - `grouped_routed_expert_fastpath_uses = 0`
  - `grouped_routed_expert_matmul_fallbacks = 1`
  - `grouped_routed_expert_prereq_fallbacks = 568`

Interpretation:

- the device-side lookup reuse work is now in place and keeps grouped-path misses cheap
- the new process-wide grouped NVFP4 disable prevents repeated failed grouped matmul probes after the first proof that the backend is unavailable
- but the decisive result did not change:
  - grouped cuBLASLt NVFP4 is still not a viable routed MoE backend on this stack for the real `batch_count = 22, m = 1, n = 2688, k = 1024` decode shape
- therefore the mini-plan should stay pointed at:
  1. fused / backend-specialized MoE kernel integration
  2. CUDA graph capture after the MoE path is graph-safe

Latest fused-routed-MoE default-path result:

- [single_token_decode_20260331T185338Z_fused_moe_custom_routed_retry_16tok.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260331T185338Z_fused_moe_custom_routed_retry_16tok.json)

What changed:

- the routed decode path now uses custom packed-NVFP4 fused kernels for:
  - routed `up_proj`
  - routed `relu^2` pack
  - weighted routed `down_proj`
- on a cold layer, selected-expert lookup misses now lazily prepare just the required routed NVFP4 lookup entries and retry the fused path in the same token instead of falling back to the old per-expert host execution loop

What it showed:

- output tokens stayed exact: `[5130 x16]`
- full 16-token sequence improved to `~26920.3 ms`
- `grouped_routed_expert_fastpath_uses = 640`
- `grouped_routed_expert_fastpath_fallbacks = 0`
- `expert_selection_metadata_downloads = 395`
- `routed_expert_materializations = 0`

Interpretation:

- the default serving path now really is running the fused routed-expert decode path end to end
- the remaining MoE host/control cost is the lazy selected-expert lookup repair on cold misses, not the old host-dispatched per-expert compute loop
- that makes CUDA graph work plausible, but only if graph capture is paired with a bounded/hybrid expert residency policy

Rejected eager graph-prep experiment:

- [single_token_decode_20260331T185639Z_fused_moe_custom_routed_eager_16tok.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260331T185639Z_fused_moe_custom_routed_eager_16tok.json)

What it showed:

- output tokens still stayed exact: `[5130 x16]`
- `expert_selection_metadata_downloads = 0`
- `grouped_routed_expert_fastpath_uses = 640`
- but `model_build_ms ≈ 201865.4`
- and `hot_mean_ms ≈ 59754.5`
- by the end of decode, host budget dropped to about `15.1 GiB` and CUDA free memory to about `1.2 GiB`

Interpretation:

- eager all-expert routed lookup preparation is graph-friendly in the narrow sense, but completely wrong for Spark memory/performance balance
- the graph-capture path should not use full eager routed-expert preparation
- the next implementation target should be bounded/hybrid expert residency on top of the working lazy fused routed-expert path

Rejected bounded-prefetch follow-up:

- [single_token_decode_20260331T193926Z_fused_moe_prefetch32_16tok.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260331T193926Z_fused_moe_prefetch32_16tok.json)

What it tried:

- `NEMOTRON_ROUTED_LOOKUP_PREFETCH_TOPN=32`
- on the first cold miss in each MoE layer, prebuild a bounded hot set of `32` corrected-score routed experts for that layer

What it showed:

- output tokens still stayed exact: `[5130 x16]`
- `expert_selection_metadata_downloads` only moved from `395` to `389`
- `routed_expert_prefetch_layers = 40`
- `routed_expert_prefetch_experts = 1280`
- but `model_build_ms ≈ 23593.6`
- and `hot_mean_ms ≈ 49204.1`

Interpretation:

- a one-shot bounded prefetch driven by the first router scores is not a good enough hybrid policy
- it pays a large startup and decode cost while barely shrinking the remaining MoE host-control surface
- the next graph-capture preparation step needs a smarter residency policy than:
  - full eager all-expert preparation
  - or first-miss top-`N` prefetch

Latest graph-prep follow-up:

- accepted artifact:
  - [single_token_decode_20260331T200103Z_step_runtime_stats_16tok.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260331T200103Z_step_runtime_stats_16tok.json)
  - exact output tokens: `[5130 x16]`
  - `hot_mean_ms ≈ 25557.2`
  - `expert_selection_metadata_downloads = 0`
  - `routed_lookup_repair_downloads = 395`
  - `routed_lookup_repair_experts = 3130`
  - `routed_expert_materializations = 0`
- what changed:
  - request-local expert aux scratch now removes the per-token auxiliary MoE tensor allocations on the default decode path
  - grouped fused routed-MoE cold repair now copies back only the missing expert IDs rather than the full selected index/weight payload
  - the decode bench now writes per-token routed lookup repair deltas to JSON
- the new per-token repair profile matters:
  - repair downloads per token:
    - `[40, 32, 33, 32, 31, 28, 31, 23, 17, 19, 14, 28, 21, 19, 18, 9]`
  - repair experts per token:
    - `[880, 565, 526, 398, 254, 83, 143, 55, 21, 30, 17, 70, 30, 23, 23, 12]`
- interpretation:
  - the default fused routed-MoE path is now free of full selection-metadata downloads
  - routed lookup repair cost does decay materially over the 16-token horizon
  - but it is still non-zero even on token `16`
  - so the next graph-capture step should not be “capture right after token 1”
  - it should be:
    1. smarter hybrid expert residency that reduces repair downloads further
    2. then capture on a post-warm shape once the repair counter actually reaches zero

Rejected static bias hotset experiment:

- [single_token_decode_20260331T200428Z_bias_top8_16tok.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260331T200428Z_bias_top8_16tok.json)
- exact output tokens: `[5130 x16]`
- `model_build_ms ≈ 12672.9`
- `hot_mean_ms ≈ 25745.0`
- `routed_lookup_repair_downloads` stayed `395`
- `routed_lookup_repair_experts` only moved `3130 -> 2990`

Interpretation:

- a small static bias hotset is not a good enough proxy for actual routed-expert demand on this trace
- it increases startup cost while barely shrinking the remaining routed repair surface
- so the next hybrid residency pass should be more dynamic / per-layer than a simple static bias top-`N`
