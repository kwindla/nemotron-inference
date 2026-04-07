# GPU Path Rollout Plan

IMPORTANT: implement this full plan continuously without stopping unless a hard blocker is encountered that cannot be overcome within the current environment.

This note narrows the next implementation stage after the initial forward-pass milestone:

- remove host roundtrips from the runtime hot path
- remove CPU fallbacks from the runtime hot path
- preserve correctness by keeping oracle-checked rollout gates at every replacement step

This is not a generic optimization note. It is tied to the current runtime implementation and its known hot-path fallback surfaces.

## Goal

Move the runtime from:

- correctness-first composed forward execution with several host-assisted operators

to:

- fully GPU-resident single-request forward execution for the serving path

while preserving the current oracle coverage and decode correctness.

## Important Scope Boundary

“Replace every host roundtrip and CPU fallback” applies to the serving hot path.

It does not mean:

- removing debug-only dumps used by oracle tests
- removing offline Python oracle tooling
- removing benchmark-only comparison modes
- removing trace-only host copies in [single_token_forward_model.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/api/single_token_forward_model.cpp) that are only exercised when trace capture is explicitly enabled

So the target is:

- no `CopyToHost` / `CopyFromHost` or CPU fallback on the default runtime forward path
- debug/test-only host copies may remain behind explicit non-serving code paths

## Immediate Follow-Up Tasks

The next three concrete follow-ups after the first GPU rollout pass are:

1. Make expert routing scratch request-local.
   Why:
   - shared mutable selection scratch inside [expert_layer.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/expert_layer.cpp) is not safe once two requests can hit the same layer concurrently.
   Success criteria:
   - the default serving path uses request-local or per-call selection buffers
   - no shared mutable routing scratch remains on `ExpertLayerSlice::Impl`
   - standalone exact-input/oracle tests stay green

2. Remove the remaining attention page-ID host copy.
   Why:
   - even after moving the large-tensor attention staging to device kernels, the default path still rebuilt and uploaded layer page IDs from host on every call.
   Success criteria:
   - request-local attention page IDs live on device in [request_context.h](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/include/nemotron/request_context.h)
   - [attention_layer.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/attention_layer.cpp) consumes those device-resident page IDs directly
   - the only remaining host uploads in attention are the still-open cuDNN auxiliary control buffers

3. Re-measure manifest-backed startup after pooled expert upload.
   Why:
   - pooled expert upload removed the biggest obvious `cudaMalloc` explosion, so we need to know whether startup materially improved before changing anything else.
   Success criteria:
   - rerun the warmed decode benchmark on the real manifest
   - capture environment build timing separately from model construction timing
   - record whether the startup bottleneck moved or remained dominated by model materialization

Status after this pass:

- Task 1 is implemented: the serving path now uses request-local expert-selection scratch via [request_context.h](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/include/nemotron/request_context.h) and [expert_layer.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/expert_layer.cpp).
- Task 2 is implemented: request-local device page-ID buffers now feed [attention_layer.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/attention_layer.cpp).
- Task 3 was re-measured repeatedly, and the startup picture is now much clearer:
  - initial build-only baseline: `model_build_ms = 406161.696`
  - after device-side BF16/FP8-to-FP32 materialization: `model_build_ms = 133095.916`
  - after conservative concurrent layer construction: `model_build_ms = 118329.436`
  - environment bootstrap stayed roughly flat at about `2.1s`
  - the dominant unresolved startup blocker remains model materialization, not runtime-environment bootstrap

The next three TTFT-focused follow-ups after that startup pass were:

1. Measure warmed second-token decode on the real manifest path.
2. Promote model cache to the next active sub-project in planning.
3. Implement the first deterministic cache format, writer, and `SingleTokenForwardModel::CreateFromCache(...)`.

Status after implementing those three follow-ups:

- warm second-token measurement is now in hand:
  - environment bootstrap about `2.1s`
  - model build about `14.7s`
  - first decode about `5.6s`
  - second decode about `0.73s`
- model-cache planning is now active in [model_cache_plan.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/docs/model_cache_plan.md)
- the first deterministic model-cache writer/loader path is implemented, but the initial execution-ready payload is not a win on DGX Spark:
  - cache write payload: about `102.1 GB`
  - cache-backed load: about `110.4s`
  - cache-backed first decode: about `1.09s`
  - cache-backed second decode: about `0.82s`
  - conclusion: keep the cache subsystem, but redesign the payload format before treating it as the next serving-path speed win

## Active Throughput Focus

The next throughput/caching work should be treated as one connected bundle, not four isolated ideas:

1. Native packed scaled-FP8 / BF16 linear execution.
   Why:
   - the current correctness-first FP32 execution surfaces are still too expensive for both throughput and cache size
   - compact model cache design depends on being able to keep dense/scaled-FP8 tensors packed on device
   Success criteria:
   - scaled-FP8 paths stop dequantizing weights into permanent FP32 execution storage
   - BF16 dense paths execute natively on GPU where checkpoint format allows
   - existing operator and forward oracles remain green

2. Device-side expert dispatch and merge with fewer sync points.
   Why:
   - current expert helper kernels still issue many launches and explicit synchronizations around routing, scatter, activation, and merge
   - this is both a throughput bottleneck and a blocker for cleaner cached serving
   Success criteria:
   - default serving path keeps expert selection, routing, and merge on device
   - launch count and D→H control traffic are reduced materially
   - expert oracle coverage stays green

3. Fused or reduced-launch Mamba path.
   Why:
   - the current Mamba slice still chains separate conv, SSM update, grouped norm, and projection stages
   - Mamba dominates most of the non-attention stack, so launch overhead compounds heavily
   Success criteria:
   - conv + SSM + gating/norm pipeline uses materially fewer launches than the current path
   - Mamba runtime-input oracle coverage stays green

4. Fused or reduced-launch attention path.
   Why:
   - attention is only eight layers, but it is still the cleanest place to reduce launch count around RMSNorm, QKV projection, SDPA plumbing, and output projection
   - it is also a natural consumer of native packed BF16 execution once that path exists
  Success criteria:
  - attention serving path removes as many standalone staging kernels as possible
  - attention oracle coverage stays green

Status after the first pass through this bundle:

- native packed BF16 / FP8 linear execution is in the serving path
- the aligned uncached decode path is now around:
  - `environment_build_ms ≈ 2.1s`
  - `model_build_ms ≈ 9.6-9.9s`
  - first decode `≈ 5.0s`
  - warmed decode `≈ 727-729 ms/token`
- decode-only Mamba fusion is implemented and oracle-checked, but it only improved warmed decode from about `728.556 ms` to about `727.544 ms`
- conclusion: Mamba inner-kernel fusion helped a little, but it is not the main remaining steady-state decode bottleneck

That changes the active implementation order for throughput work:

1. grouped / indirect routed-expert dispatch
2. deeper expert merge / launch-count reduction around routed NVFP4 execution
3. only then more Mamba-local fusion work
4. attention hot-path tightening after the MoE path moves

Immediate next throughput milestone:

- stop returning to the host between `SelectTopExpertsFp32(...)` and the routed expert GEMMs
- use `cublasLt` grouped batch mode with device pointer arrays on this stack if it proves stable for the selected NVFP4 shapes
- otherwise build a device-driven routed-expert gather/dispatch surface that still collapses the current per-expert host control loop materially

Current recommendation about impact:

- compared to a fully optimized memory-bound roofline, fusion alone is probably only a `10-20%` batch-1 gain
- compared to the runtime as it exists today, native packed linear execution plus the fusion bundle can be a several-x gain because the baseline still includes correctness-first reference kernels and frequent synchronizations
- that is why the implementation order should stay:
  1. native packed linear execution
  2. expert dispatch/merge
  3. Mamba fusion
  4. attention fusion

Latest DGX Spark decode-gap follow-up:

- the attention-persistence/runtime-stats pass was a real 16-token win:
  - best validated artifact: [single_token_decode_20260331T_runtime_stats_experimental_disabled.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/single_token_decode_20260331T_runtime_stats_experimental_disabled.json)
  - full sequence `~25591.7 ms`
  - last `8` tokens `~698.7 ms/token`
  - last `4` tokens `~667.7 ms/token`
- the decode counters from that run show the main remaining hot-path problem explicitly:
  - `dense_reference_fallbacks = 1824`
  - `scaled_fp8_reference_fallbacks = 2176`
  - `scaled_fp8_native_success = 0`
- a more aggressive scaled-FP8 serving experiment was rejected:
  - artifact: [single_token_decode_20260331T_runtime_stats_scaled_fp8_mamba_fastpath.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/single_token_decode_20260331T_runtime_stats_scaled_fp8_mamba_fastpath.json)
  - output tokens changed to `[72773 x16]`
  - sequence time regressed to `~34350.2 ms`
- request-local device token-ID storage plus async/default-stream control uploads are now in the serving path and preserve exact 16-token output correctness, but they did not materially improve the tail on their own:
  - latest graph-prep artifact: [single_token_decode_20260331T_device_token_ids_async_control.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/single_token_decode_20260331T_device_token_ids_async_control.json)
  - last `8` tokens `~709.1 ms/token`
  - conclusion: keep them because they remove graph blockers, not because they are the next decode multiplier

So the current throughput order is even narrower now:

1. remove grouped-MoE host selection/control from the default decode path
2. then attempt real steady-state CUDA graph capture on the 16-token append-only decode path
3. only after that, revisit deeper scaled-FP8 fallback elimination or larger fused-kernel swaps

The latest MoE runtime counters now justify that order directly:

- [single_token_decode_20260331T_moe_runtime_stats.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/single_token_decode_20260331T_moe_runtime_stats.json)
- token correctness stayed exact: `[5130 x16]`
- `expert_selection_metadata_downloads = 640`
- `routed_expert_materializations = 6238`

So the next major serving-path throughput work should not be framed as “more generic graph prep.” It should be framed as:

1. MoE host/control elimination
2. MoE first-touch/materialization smoothing or hybrid residency
3. then steady-state CUDA graph capture

## Latest Construction Timing Result

The decode benchmark now emits build-time phase and per-layer timing from [single_token_forward_model.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/api/single_token_forward_model.cpp), and [single_token_decode_bench.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/benchmarks/decode_bench/single_token_decode_bench.cpp) now writes that build report into its JSON artifact.

Current measured build-only progression on the real manifest:

- baseline artifact: `artifacts/benchmarks/single_token_decode_model_build_only_20260330T234103Z.json`
  - `environment_build_ms = 2114.590`
  - `model_build_ms = 406161.696`
  - `layer_loop_total_ms = 384613.620`
- device-conversion artifact: `artifacts/benchmarks/single_token_decode_model_build_only_20260331T003246Z.json`
  - `environment_build_ms = 2081.808`
  - `model_build_ms = 133095.916`
  - `layer_loop_total_ms = 130485.300`
- concurrent-build artifact: `artifacts/benchmarks/single_token_decode_model_build_only_20260331T004129Z.json`
  - `environment_build_ms = 2055.868`
  - `model_build_ms = 118329.436`
  - `layer_loop_total_ms = 114931.048`
- current best detailed artifact: `artifacts/benchmarks/single_token_decode_model_build_only_with_details_v3.json`
  - `environment_build_ms = 2187.065`
  - `model_build_ms = 90566.993`
  - `layer_loop_total_ms = 89596.374`
  - top accumulated detail buckets at that checkpoint:
    - `nvfp4_pool_upload`: about `73355 ms`
    - `norm_gate_fc2_uploads`: about `55613 ms`
    - `in_proj_create`: about `31503 ms`
    - `nvfp4_pool_stage`: about `29460 ms`
- later follow-up constructor experiments were measured and rejected because they lost to the `~90.6s` baseline:
  - pooled expert dense control uploads:
    - `artifacts/benchmarks/single_token_decode_model_build_only_20260331T022130Z_control_pool.json`
    - `model_build_ms = 94845.481`
  - async NVFP4 pooled expert upload:
    - `artifacts/benchmarks/single_token_decode_model_build_only_20260331T022740Z_async_nvfp4.json`
    - `model_build_ms = 95342.537`
  - pooled Mamba norm/conv/state tensor uploads:
    - `artifacts/benchmarks/single_token_decode_model_build_only_20260331T023344Z_mamba_pool.json`
    - `model_build_ms = 112701.443`
  - dynamic layer work-queue scheduling:
    - measured during a later follow-up run
    - regressed wall time to about `107.1s`
    - rejected; the strided `4`-worker scheduler remains the active path
  - eager artifact loading:
    - front-loaded about `74.1s` into environment build and then failed before forward-model construction completed
    - rejected as a startup path on this stack
  - `mmap` prefetch hints:
    - `artifacts/benchmarks/single_token_decode_model_build_only_20260331T_mmap_compare.json`
      - `environment_build_ms = 2095.646`
      - `model_build_ms = 96696.711`
    - `artifacts/benchmarks/single_token_decode_model_build_only_20260331T_mmap_prefetch_compare.json`
      - `environment_build_ms = 2083.194`
      - `model_build_ms = 99109.744`
    - rejected; plain `mmap` remained better
- current best startup result after lazy aligned routed-expert NVFP4 materialization:
  - `artifacts/benchmarks/single_token_decode_model_build_only_20260331T_lazy_routed_experts_aligned.json`
  - `environment_build_ms = 2114.968`
  - `model_build_ms = 15114.569`
- current first real manifest-backed first-token decode:
  - `artifacts/benchmarks/single_token_decode_first_token_20260331T_lazy_routed_experts_aligned.json`
  - `environment_build_ms = 2057.393`
  - `model_build_ms = 13955.328`
  - `hot_0_ms = 11870.280`
  - `predicted_token_id = 5130`
  - rough end-to-end first token from cold start: about `27.9s`

Breakdown by layer family at the latest checkpoint:

- because the latest run overlaps layer creation, per-layer totals are no longer additive; use them as hotspot indicators, not as a sum that should match wall time
- expert slices still dominate the worst observed layer builds:
  - layer `70`: about `9962 ms`
  - layer `79`: about `9039 ms`
  - layer `65`: about `8909 ms`
  - layer `72`: about `8142 ms`
  - layer `67`: about `7564 ms`
- Mamba slices are still the next-largest repeated cost:
  - several layers remain in the `1.0-4.2s` range even after the device-conversion work
- attention slice creation is still comparatively cheap on average, though a few layers now show higher wall time because they overlap with other build activity

Other one-time uploads are also non-trivial:

- embedding upload fell from about `11412 ms` to about `1167 ms`
- lm_head upload fell from about `9941 ms` to about `2022 ms`

Current interpretation:

- the startup problem is now clearly construction/materialization, not runtime-environment bootstrap
- expert slice creation is still the dominant layer-level cost
- Mamba slice creation is still the next-largest repeated cost
- attention construction is comparatively cheap
- the biggest measured win so far came from moving BF16 / FP8 weight materialization onto the device
- a smaller but real second win came from overlapping layer creation with conservative parallelism
- raising the default build worker count from `2` to `4` improved wall time again, and the best later detailed run reached about `90.6s`
- capping the inner expert-staging thread fan-out was tried and rejected; it pushed the wall time back toward `112s` and did not reduce the slowest-layer spikes enough to justify keeping it
- pinned expert host staging did not materially improve total wall time and should not be treated as the main lever
- three more local constructor experiments were also rejected after direct measurement:
  - pooled expert dense control uploads slightly improved one local timing bucket but lost wall time overall
  - async NVFP4 pooled expert upload did not beat the blocking pooled copy on this stack
  - pooled Mamba norm/conv/state tensor uploads regressed wall time sharply despite reducing allocation count
- loader-policy and scheduling follow-ups were also rejected after direct measurement:
  - dynamic work-queue scheduling increased contention and lost to the simpler strided scheduler
  - `readall` eager loading was too expensive up front and did not reach a successful model build
  - `mmap` prefetch hints did not beat plain `mmap`
- the newest successful startup win came from changing routed-expert NVFP4 upload strategy, not from more generic loader/scheduler tuning:
  - eager routed-expert upload was removed from model construction
  - lazy routed-expert NVFP4 materialization initially exposed a real alignment issue in the cuBLASLt planner
  - the aligned view-backed NVFP4 staging fix preserved expert-layer oracle correctness and reduced build-only startup to about `15.1s`

The latest narrowed profile attempt used `nsys` on model construction only. CUDA trace data was not captured on this stack, but the `osrt_sum` report still showed time dominated by driver-facing waits (`poll` and `ioctl`). A fallback `perf` run was blocked by `perf_event_paranoid=4`. So the next profitable work item is to optimize expert and Mamba construction directly, not to invest more in generic profiling setup first.

The current build loop now supports:

- `NEMOTRON_FORWARD_BUILD_DEBUG=1` for streaming phase/layer progress
- `NEMOTRON_FORWARD_BUILD_THREADS=<n>` to override the default concurrent layer-construction worker count, which is now `4`

## Current Hot-Path Fallback Inventory

### 1. Dense fallback surface

Current file:

- [linear_op.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/linear_op.cpp)

Current issue:

- dense BF16-backed checkpoint descriptors can still fall back to:
  - host activation download
  - host weight decode
  - CPU matmul
  - host output upload

Why it matters:

- this path sits under attention and MoE dense projections
- it already caused measurable decode drift before the higher-accuracy accumulation fixes

### 2. Scaled-FP8 linear surface

Current file:

- [scaled_fp8_linear.cu](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/scaled_fp8_linear.cu)

Current issue:

- the operator still downloads activations to host
- performs correctness-first quantize/dequant + matmul on host
- uploads output back to device
- only uses the device path opportunistically today, behind the `NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH` gate and narrow shape assumptions

Why it matters:

- this is on the Mamba hot path
- it previously contributed to decode drift before the host accumulation fix

### 3. Attention host-assisted staging

Current file:

- [attention_layer.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/attention_layer.cpp)

Current issue:

- Q/K/V tensors and paged-cache buffers still go through host-assisted BF16 staging in the slice implementation
- output staging also uses host roundtrips before re-entering the projection path

Why it matters:

- attention is already on the selected GB10 backend path
- this is the cleanest next place to make the full serving path GPU-resident

### 4. Mamba scan and state-update host path

Current file:

- [mamba_layer.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/mamba_layer.cpp)

Current issue:

- the recurrent scan, conv update, and state updates still use host copies in the composed slice
- multi-row prefill replay is still correctness-first and sequential on the host side

Why it matters:

- Mamba is on the critical path for both TTFT and decode
- this is the largest remaining non-GPU serving surface

### 5. Expert-layer host-assisted routed/shared execution

Current file:

- [expert_layer.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/expert_layer.cpp)

Current issue:

- router/top-k, latent staging, routed NVFP4 execution plumbing, shared path handling, and merge flow still use host-visible intermediate buffers in the current correctness-first slice

Why it matters:

- the decode investigation already showed that tiny upstream differences can be amplified materially through routed NVFP4 expert paths
- if this path stays partly host-based, it will remain both a latency problem and a correctness-drift risk

## Rollout Strategy

Do not replace everything at once.

Use this order:

1. dense + scaled-FP8 linear surfaces
2. attention slice
3. Mamba slice
4. expert slice
5. full composed forward path cleanup

Reason:

- the first step is one bundle because dense linear and scaled-FP8 linear are the same class of fix:
  - remove host matmul fallback
  - enable the serving-path device execution unconditionally
  - keep only reference/debug CPU behavior
- attention is already backend-selected and oracle-gated
- Mamba and expert cleanup should happen only after their underlying linear surfaces are truly GPU-resident

## Per-Stage Plan

### Stage 1. Eliminate dense and scaled-FP8 CPU fallbacks together

Files:

- [linear_op.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/linear_op.cpp)
- [dense_gemm_runner.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/dense_gemm_runner.cpp)
- [scaled_fp8_linear.cu](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/scaled_fp8_linear.cu)

Plan:

- make dense BF16-backed checkpoint descriptors execute on GPU by default
- keep descriptor decode on startup or weight upload, not per-call
- remove hot-path host activation download and host output upload from dense execution
- remove the `NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH` dependency from the default serving path
- if the current direct `cublasLt` fastpath rejects a shape, fall back to a device-side reference path, not to CPU
- keep the current FP8 semantic contract:
  - apply checkpoint `input_scale`
  - quantize to `E4M3`
  - use the quantized activation semantics required by the checkpoint
- add the small device-side FP8 activation quantization round-trip kernel needed for that contract
- move the quantize/dequant and matmul execution fully onto device
- if the current direct device fastpath rejects a shape, fall back to a device-side dequant + dense GEMM path rather than a CPU matmul
- use the host path only as a reference/testing implementation

Correctness gate:

- existing dense tests remain green
- [scaled_fp8_linear_compare_test.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/testing/backend/scaled_fp8_linear_compare_test.cpp) stays green
- affected attention / Mamba / expert exact-input replays stay green at the layers already used in decode localization
- decode sweep must not regress

Exit criteria:

- no serving-path call through [linear_op.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/linear_op.cpp) uses CPU fallback in normal execution
- no activation `CopyToHost` / output `CopyFromHost` on the default scaled-FP8 serving path

### Stage 2. Remove host-assisted attention staging

Files:

- [attention_layer.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/attention_layer.cpp)

Plan:

- keep Q/K/V projections on device
- keep paged KV updates on device
- keep page-table and sequence-length inputs device-resident for the runtime path
- feed cuDNN FE directly from device tensors without host BF16 staging
- add simple device transpose / scatter kernels rather than trying to solve this with a broad tensor-runtime abstraction
- keep only debug/oracle dumps on the host side

Correctness gate:

- [attention_layer_oracle_test.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/testing/backend/attention_layer_oracle_test.cpp)
- runtime-input attention oracle replays
- prefill prefix oracle

Exit criteria:

- no host-assisted BF16 staging in the default attention slice path

### Stage 3. Replace Mamba host scan/state math with GPU kernels

Files:

- [mamba_layer.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/mamba_layer.cpp)

Plan:

- implement a fully device-resident FP32 Mamba scan/update baseline first
- keep the current request-local conv and SSM state layout
- reuse the already-proven update patterns from the GB10 Mamba cache benchmark where practical, instead of inventing a separate math path
- preserve exact-input Mamba oracle parity before attempting FP16+SR runtime substitution
- only after FP32 GPU parity is stable, add the optimized state-format path

Correctness gate:

- [mamba_layer_oracle_test.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/testing/backend/mamba_layer_oracle_test.cpp)
- exact-input runtime replay fixtures for layers `9`, `13`, `15`, `18`, `24`, `27`, `29`, `31`
- decode sweep must not move the first-failing layer earlier again

Exit criteria:

- no host state downloads/uploads in the default Mamba serving path

### Stage 4. Replace expert host-assisted execution with GPU dispatch

Files:

- [expert_layer.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/expert_layer.cpp)

Plan:

- move router logits and top-k dispatch to GPU
- use device-resident latent and merge buffers
- replace `RunNvfp4LinearHost(...)` with the validated device-side NVFP4 runner path
- keep routed/shared expert execution on validated GPU paths
- add small device kernels for expert selection bookkeeping and weighted merge rather than leaving those pieces on the host
- add the explicit new-kernel work this stage requires:
  - device-side top-k / expert selection
  - device-side scatter from tokens to per-expert worklists
  - device-side `relu2`
  - device-side weighted gather-and-merge
- treat this as the stage most likely to need net-new kernel code, not just wiring existing operators
- remove host-visible intermediate assembly from the default path
- keep the current oracle trace hooks available only behind test/debug controls

Correctness gate:

- [expert_layer_oracle_test.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/testing/backend/expert_layer_oracle_test.cpp)
- runtime-input exact-input expert replays at layers `5`, `8`, `14`, `19`, `26`, `28`, `30`
- final functional decode oracle remains green

Exit criteria:

- no host roundtrip in the default expert slice path

### Stage 5. Full composed-path cleanup

Files:

- [single_token_forward_model.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/api/single_token_forward_model.cpp)
- request-context owned buffers and slices

Plan:

- verify the composed single-request prefill and decode paths do not accidentally trigger any remaining host fallback
- add runtime assertions or debug counters for fallback activation in dev builds
- keep lazy first-use weight materialization separate from the fallback discussion:
  - lazy materialization during the first real forward call is acceptable startup amortization
  - it is not a serving hot-path fallback once the weight object is resident
- leave explicit reference/debug paths available only behind test-only or debug-only call sites

Correctness gate:

- [full_forward_manifest_smoke_test.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/testing/api/full_forward_manifest_smoke_test.cpp)
- [prefill_prefix_oracle_test.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/testing/api/prefill_prefix_oracle_test.cpp)
- [single_token_decode_oracle_test.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/testing/api/single_token_decode_oracle_test.cpp)

Exit criteria:

- the default composed forward path is GPU-resident
- fallback surfaces exist only in explicit reference/debug code paths

## Correctness Method

For each stage:

1. keep the old path available temporarily behind a reference/debug switch
2. add or reuse an exact-input oracle test at the slice boundary
3. switch the default serving path only after exact-input parity is green
4. rerun:
   - the targeted slice oracle
   - the relevant runtime-input replay tests
   - prefill oracle if the slice affects prefill
   - full functional decode oracle

Do not trade correctness visibility for speed.

## Performance Method

Once a GPU replacement is correct:

- benchmark the operator in isolation if a harness already exists
- then rerun the composed forward path tests
- only after both pass should the old host path be demoted to reference-only

This avoids “optimizing” a path that still moves the decode boundary backward.

Wall-time expectation for this sub-plan:

- the current correctness-first all-layer single-token decode path is much too slow for a GPU-resident runtime target
- after full GPU migration, a single-token decode through all `88` layers on DGX Spark should be on the order of seconds, not minutes
- the working target for this sub-plan is:
  - single-token decode wall time under `30` seconds on DGX Spark

This is not the final serving target, but it is the threshold that tells us the migration actually removed the dominant host-path bottlenecks.

## Definition Of Done

This sub-plan is complete when all of the following are true:

- the default runtime forward path for single-request prefill and one-token decode does not use CPU fallbacks
- the default runtime forward path does not perform host roundtrips except for explicit debug/test dumps
- all current oracle tests remain green
- the full suite remains green under `ctest`
- the final functional decode oracle remains green on the real manifest
- single-token decode wall time is under `30` seconds on DGX Spark

## Immediate Next Step

Start with Stage 1 and continue directly into Stage 2 if no hard blocker appears.

The generic dense fallback in [linear_op.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/linear_op.cpp) is the best first target because:

- it sits underneath both attention and expert slices
- it already proved capable of moving the decode boundary when fixed for accuracy
- removing it reduces both latency and another class of chained drift

The existing notes in [forward_pass_gpu_migration.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/docs/forward_pass_gpu_migration.md) are consistent with this plan and sharpen the same initial move:

- dense + scaled-FP8 first
- then attention scatter/layout cleanup
- then Mamba state math
- then expert NVFP4 dispatch/merge cleanup

## Latest Rollout Status

Current implementation status:

- Stage 1 is implemented:
  - dense BF16 and scaled-FP8 default execution now have native packed `cublasLt` paths
  - FP32 reference execution remains only as a guarded fallback surface
  - current files:
    - [linear_op.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/linear_op.cpp)
    - [dense_gemm_runner.h](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/include/nemotron/dense_gemm_runner.h)
    - [dense_gemm_runner.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/dense_gemm_runner.cpp)
    - [dense_reference_gemm.cu](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/dense_reference_gemm.cu)
    - [scaled_fp8_linear.cu](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/scaled_fp8_linear.cu)
    - [storage_conversion.cu](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/storage_conversion.cu)
    - [device_tensor.h](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/include/nemotron/device_tensor.h)
    - [device_tensor.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/device_tensor.cpp)
- Stage 2 is implemented:
  - attention layout/scatter/staging are device-side
  - K and V cache scatter now use one fused launch
  - serving-path helper sync barriers have been removed from the layout and cuDNN execute surfaces
  - current files:
    - [attention_layout.h](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/include/nemotron/attention_layout.h)
    - [attention_layout.cu](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/attention_layout.cu)
    - [attention_layer.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/attention_layer.cpp)
    - [cudnn_paged_attention.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/cudnn_paged_attention.cpp)
- Stage 3 is implemented:
  - Mamba scan/update/state math are device-side
  - serving-path helper sync barriers are removed, so the current path is reduced-launch even though it is not yet a single fused kernel
  - current files:
    - [mamba_ops.h](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/include/nemotron/mamba_ops.h)
    - [mamba_ops.cu](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/mamba_ops.cu)
    - [mamba_layer.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/mamba_layer.cpp)
- Stage 4 is largely implemented:
  - routed/shared large-tensor expert execution is on GPU
  - serving-path helper sync barriers are removed from the current dispatch/merge helpers
  - current files:
    - [expert_ops.h](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/include/nemotron/expert_ops.h)
    - [expert_ops.cu](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/expert_ops.cu)
    - [expert_layer.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/backend/expert_layer.cpp)
  - grouped routed-expert dispatch is now implemented for the single-token path:
    - grouped pointer-array NVFP4 `up_proj`
    - device-side `relu2 + pack`
    - grouped pointer-array NVFP4 `down_proj`
    - device-side weighted merge
  - remaining caveat:
    - the runtime still needs host awareness of first-touch selected experts because lazy routed-expert materialization remains the startup win that keeps model build near `10s`
    - so the current grouped path still preserves a host-assisted fallback when a request selects an expert whose aligned device view has not yet been materialized
- Stage 5 is partially implemented:
  - full correctness suite is green again at `59/59`
  - a warmed manifest-backed decode harness now exists:
    - [single_token_decode_bench.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/benchmarks/decode_bench/single_token_decode_bench.cpp)
    - [run_bench.sh](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/benchmarks/decode_bench/run_bench.sh)
  - current measurement status:
    - manifest-backed environment bootstrap is only about `2.1s`
    - current best uncached model construction on the packed-linear path is about `9.6s`
    - current first real manifest-backed first-token decode on that path is about `5.1s`
    - current warmed decode is now about `0.71s/token`
    - the rough `<30 s` end-to-end first-token target is now comfortably satisfied on the current benchmark path
  - the remaining default-path host interaction (expert indices/weights D→H copy) is now pre-allocated instead of per-call `cudaMalloc`/`cudaFree`, with grow-on-demand for batch sizes beyond the pre-allocated capacity

## Current Evaluation

The current rollout result is asymmetric:

- startup and cold first-token TTFT improved materially
- warmed decode throughput did not move much yet

Latest aligned measurement:

- [single_token_decode_20260331T_throughput_pass_uncached_threads4.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/single_token_decode_20260331T_throughput_pass_uncached_threads4.json)
- `environment_build_ms ≈ 2103`
- `model_build_ms ≈ 9603`
- `warmup_0_ms ≈ 5064`
- `hot_mean_ms ≈ 733`
- top-1 token still `5130`

So the packed-linear work is useful and correct, but it is not the final throughput unlock by itself. The next throughput passes should prioritize:

1. hybrid expert residency / cache strategy so more routed experts are already materialized before decode
2. lower-overhead routed-expert compute backend beyond the current grouped `cublasLt` pointer-array path
3. batching / MTP work, because further micro-launch cleanup is no longer a large multiplier

## Latest Grouped Routed-Expert Result

The routed-expert dispatch rewrite landed and stayed correct:

- routed expert selection now feeds grouped NVFP4 pointer-array matmuls instead of a host loop over one expert GEMM at a time
- routed expert activation squaring / repacking is device-side
- routed expert weighted merge is device-side
- device lookup tables are now maintained for routed expert packed / scale pointers and tensor scales

Latest aligned grouped-expert measurement:

- [single_token_decode_20260331T_grouped_routed_experts_device_lookup_threads4.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/single_token_decode_20260331T_grouped_routed_experts_device_lookup_threads4.json)
- `environment_build_ms ≈ 2099`
- `model_build_ms ≈ 8769`
- `warmup_mean_ms ≈ 6508`
- `hot_mean_ms ≈ 710.6`

Interpretation:

- grouped routed-expert dispatch is a real win, but only a modest one
- the runtime moved from about `728.6` to about `710.6 ms/token`, which is meaningful but far smaller than a launch-count-only model predicted
- the runtime is now clearly at the point where the next throughput multiplier will not come from more host-loop cleanup
- the remaining work should shift toward:
  - expert residency / cache strategy
  - a lower-overhead routed-expert compute backend
  - batching / MTP

## Latest Long-Generation Result

The runtime now has a real persistent decode-step path rather than only repeated single-token fresh-request timing:

- [single_token_forward_model.h](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/include/nemotron/single_token_forward_model.h)
- [single_token_forward_model.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/runtime/src/api/single_token_forward_model.cpp)
- [single_token_decode_bench.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/benchmarks/decode_bench/single_token_decode_bench.cpp)

Latest append-only decode measurement:

- [single_token_decode_20260331T090018Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/platforms/super-spark/artifacts/benchmarks/single_token_decode_20260331T090018Z_cuda132.json)
- `environment_build_ms ≈ 2039`
- `model_build_ms ≈ 10558`
- `generate_tokens = 16`
- full sequence `≈ 36321 ms`

Per-step interpretation:

- first `4` tokens average `≈ 5015 ms/token`
- last `8` tokens average `≈ 999 ms/token`
- last `4` tokens average `≈ 937 ms/token`

This matters because it changes the steady-state read:

- the earlier `~710 ms/token` grouped-routed microbench is still useful as a short-run indicator
- but the first realistic append-only decode study settles closer to `~0.94-1.00 s/token`
- so the next throughput pass should target the early-token hot path and the dense fallback surfaces that still show up repeatedly during long decode

Updated throughput priorities:

1. reduce serving-path dense fallback incidence
2. reduce first-touch expert costs on early decode tokens
3. keep expert residency / cache work active
4. only after that, treat batching / MTP as the next multiplier

## Model Construction Wall-Time Analysis

The 4+ minute model construction is dominated by expert weight uploads:

- `40` MoE layers × `512` experts × `2` projections = `40,960` `UploadedLinearOp::Create` calls
- each `DeviceNvfp4Weight::Upload` does `4` `cudaMalloc` + `4` `cudaMemcpy` (packed data, block scales, matmul scales, tensor scale)
- total: `163,840` `cudaMalloc` calls and `163,840` small H→D copies for expert weights alone
- plus `40,960` `SwizzleRowMajorNvfp4ScalesForExecution` CPU calls

Estimated cost breakdown:

- `cudaMalloc` overhead at ~10µs each: ~1.6s minimum, likely much worse with memory fragmentation at `112.7 GB` total
- raw H→D bandwidth for `112.7 GB` at `273 GB/s` peak: ~0.4s
- individual copies average ~2.75 MB, far too small to saturate bandwidth
- scale swizzle CPU work serializes with GPU

**Fix implemented**: pooled expert weight allocation in `ExpertLayerSlice::Create()`. All NVFP4 expert weights for one layer are staged into a single host buffer, uploaded with one `cudaMalloc` + one `cudaMemcpy`, and each expert gets a non-owning `DeviceNvfp4Weight::CreateView`. Sub-buffers are 256-byte aligned (required by cublasLt NVFP4). This reduces `cudaMalloc` calls from `163,840` to `40` and `cudaMemcpy` calls from `163,840` to `40`.

Current best read:

- the serving path is now GPU-resident enough to preserve correctness across the full `59/59` regression suite
- the remaining closure items are:
  - eliminate the tiny expert-selection metadata host copy, or explicitly redesign the expert dispatch interface so that copy is no longer needed
  - reduce manifest-backed model construction/materialization cost enough to obtain and then satisfy the warmed decode target
