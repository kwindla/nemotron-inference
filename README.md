# Nemotron Runtime

This directory is the clean workspace for the C++/CUDA runtime implementation.

Scope boundaries:

- `runtime/` and `kernels/` are reserved for the serving runtime and should stay C++/CUDA.
- `tools/` contains offline support utilities such as checkpoint inspection.
- `testing/` contains oracle inputs and other non-runtime validation assets.
- `benchmarks/` contains harnesses that exercise the runtime or external baselines.
- `artifacts/preflight/` stores generated reports from pre-implementation testing.

## Preflight

Run the pre-implementation gate from this directory:

```bash
./scripts/run_preflight.sh
```

Outputs land in `artifacts/preflight/`.

Implementation progress is tracked in [PROGRESS.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/PROGRESS.md).
The v1 cache design note is in [docs/v1_cache_architecture.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/docs/v1_cache_architecture.md).
The focused sub-plan for the first oracle-checked end-to-end forward path is in [docs/initial_forward_pass_plan.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/docs/initial_forward_pass_plan.md).
The follow-on sub-plan for removing host roundtrips and CPU fallbacks from the runtime hot path is in [docs/gpu_path_rollout_plan.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/docs/gpu_path_rollout_plan.md).
The current DGX Spark decode-gap note is in [docs/dgx_spark_decode_gap_mini_plan.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/docs/dgx_spark_decode_gap_mini_plan.md).
The primary MoE backend integration note is in [docs/flashinfer_moe_integration_plan.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/docs/flashinfer_moe_integration_plan.md).
The active production-aligned MoE target is now FlashInfer CUTLASS fused NVFP4 MoE; the earlier TRT split FlashInfer path is retained only as a rejected reference experiment and interim seam history.
The GB10 performance-planning note is in [docs/gb10_performance_plan.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/docs/gb10_performance_plan.md).
The GB10 performance history log is in [docs/gb10_performance_progress.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/docs/gb10_performance_progress.md).
The short GB10 operator policy note is in [docs/gb10_execution_policy.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/docs/gb10_execution_policy.md).
The provisional startup-residency note is in [docs/startup_residency_policy.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/docs/startup_residency_policy.md).

## Build

Configure and build the current runtime libraries, tests, and probes:

```bash
cmake -S . -B build
cmake --build build -j
```

Run the current C++ tests:

```bash
ctest --test-dir build --output-on-failure
```

For lightweight local development, that command is fine. For real manifest-backed or parity validation on DGX Spark / GB10, use the guarded runbook in
`../proj-2026-04-04-nvfp4-activation-quant-alignment/TEST-RUNBOOK.md`
instead of blindly launching the whole suite in parallel.

Current vLLM parity baseline note:

- the checked-in oracle path at [tools/oracle/generate_vllm_trace.sh](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/tools/oracle/generate_vllm_trace.sh) is pinned to `nemotron-local/dgx-spark-vllm:0.17.1-b31e9326a-fi065`
- that pinned image is the active parity oracle today; it is not automatically the latest upstream `vllm` `main`
- if you want parity against current upstream `main`, update the image pin intentionally and regenerate the vLLM trace artifacts before treating comparisons as authoritative

Current Nemotron parity status:

- manifest-backed [prompt_matched_parity_test.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/api/prompt_matched_parity_test.cpp) is now green against the pinned vLLM oracle on the direct `Create(...)` path
- `CreateFromCache(...)` is not yet qualified for that same Nemotron parity path
- the current cache-backed qualification attempt fails during model construction:
  - first at the `SingleTokenForwardModel::CreateFromCache(...)` memory-budget guard on DGX Spark UMA
  - and, with that guard bypassed, in cache-backed NVFP4 view construction under [model_cache.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/runtime/src/api/model_cache.cpp)
- treat cache-backed Nemotron forward execution as in progress until that path is fixed and revalidated

If `/usr/local/cuda/compat` or `/usr/local/cuda-13.2/compat` exists, `ctest` now prepends it automatically for the runtime test binaries. The benchmark wrapper scripts below do the same. Direct binary launches still need an equivalent `LD_LIBRARY_PATH` if the compat stack is required.

Run the first GB10 dense microbenchmark harness:

```bash
./build/benchmarks/gb10_dense_gemm/gb10_dense_gemm_bench --list-cases
./build/benchmarks/gb10_dense_gemm/gb10_dense_gemm_bench --case attention_core_proj --rows 1,64 --workspace-bytes 0,4194304
./benchmarks/gb10_dense_gemm/run_default_bench.sh
./build/benchmarks/gb10_nvfp4_gemm/gb10_nvfp4_gemm_bench --list-cases
./build/benchmarks/gb10_nvfp4_gemm/gb10_nvfp4_gemm_bench --case attention_core_proj --rows 64 --workspace-bytes 0,4194304
./build/benchmarks/gb10_nvfp4_gemm/gb10_nvfp4_gemm_bench --case attention_core_proj --rows 64 --workspace-bytes 0,4194304 --activation-staging host
./benchmarks/gb10_nvfp4_gemm/run_default_bench.sh
./benchmarks/gb10_nvfp4_contract/run_default_bench.sh
./build/benchmarks/gb10_paged_attention/gb10_paged_attention_bench --list-cases
./build/benchmarks/gb10_paged_attention/gb10_paged_attention_bench --case decode_b1_kv64k --warmup 1 --iterations 3
./benchmarks/gb10_paged_attention/run_default_bench.sh
./build/benchmarks/gb10_mamba_cache/gb10_mamba_cache_bench --concurrency 1,4,8 --operations clone,update,mamba_update,fixture_update --sr-seed 5639999111463849804
./build/benchmarks/gb10_mamba_cache/gb10_mamba_cache_bench --operations fixture_chain --concurrency 1,8 --fixture-chain-steps 8,32,128 --warmup 1 --iterations 3
./build/benchmarks/gb10_mamba_cache/gb10_mamba_cache_bench --operations fixture_trace --concurrency 1,8 --fixture-trace-steps 128,320,544,704 --warmup 1 --iterations 3
./benchmarks/gb10_mamba_cache/run_default_bench.sh
./benchmarks/gb10_mamba_trace_compare/run_compare.sh
./benchmarks/gb10_mamba_trace_family/run_summary.sh
./benchmarks/gb10_mamba_trace_family/run_guardrails.sh
./benchmarks/gb10_mamba_trace_operands/run_analysis.sh
./benchmarks/gb10_mamba_trace_operands/run_analysis.sh --target-fixture mamba_layer0_target_chat_trace_markdown_flat
./benchmarks/gb10_mamba_trace_dt_hotspots/run_analysis.sh
./benchmarks/gb10_mamba_trace_dt_hotspots/run_analysis.sh --target-fixture mamba_layer0_target_chat_trace_markdown_flat
./benchmarks/gb10_mamba_trace_token_hotspots/run_analysis.sh
./benchmarks/gb10_mamba_trace_token_hotspots/run_analysis.sh --fixture-name mamba_layer0_target_chat_trace_markdown_flat
./benchmarks/gb10_mamba_trace_think_boundaries/run_analysis.sh
./benchmarks/gb10_mamba_trace_dt_buckets/run_analysis.sh
./benchmarks/gb10_mamba_trace_dt_scale_sweep/run_sweep.sh
./benchmarks/gb10_mamba_dt_threshold_sweep/run_sweep.sh
./benchmarks/gb10_stochastic_rounding_probe/run_probe.sh
./build/benchmarks/decode_bench/single_token_decode_bench --manifest ./artifacts/manifests/forward_runtime_manifest_unverified.json --iterations 1
./build/benchmarks/decode_bench/single_token_decode_bench --manifest ./artifacts/manifests/forward_runtime_manifest_unverified.json --warmup 0 --iterations 0
NEMOTRON_FORWARD_BUILD_DEBUG=1 ./build/benchmarks/decode_bench/single_token_decode_bench --manifest ./artifacts/manifests/forward_runtime_manifest_unverified.json --warmup 0 --iterations 0
NEMOTRON_FORWARD_BUILD_THREADS=1 ./build/benchmarks/decode_bench/single_token_decode_bench --manifest ./artifacts/manifests/forward_runtime_manifest_unverified.json --warmup 0 --iterations 0
./benchmarks/decode_bench/run_bench.sh ./artifacts/manifests/forward_runtime_manifest_unverified.json
./build/benchmarks/gb10_loader_modes/gb10_loader_modes_bench --manifest /path/to/manifest.json --repeats 3
./benchmarks/gb10_loader_modes/run_bench.sh /path/to/manifest.json 3
./benchmarks/gb10_gemm_compare/run_compare.sh
./tools/oracle/generate_nvfp4_operator_fixture.sh
OPERATOR_KIND=down_proj ./tools/oracle/generate_nvfp4_operator_fixture.sh
OPERATOR_KIND=shared_down_proj ./tools/oracle/generate_nvfp4_operator_fixture.sh
./tools/oracle/generate_attention_layer_fixture.sh
./tools/oracle/generate_mamba_update_fixture.sh
./tools/oracle/generate_mamba_layer_fixture.sh
./tools/oracle/generate_expert_layer_fixture.sh
./tools/oracle/generate_single_token_forward_fixture.sh
MODE=prefix_prefill PROMPT_NAME=short_chat PROMPT_TOKEN_COUNT=4 CAPTURE_LAYERS=0,1,7 STOP_LAYER=7 ./tools/oracle/generate_single_token_forward_fixture.sh
./tools/oracle/generate_mamba_target_trace_fixture.sh
PROFILE=/workspace/nemotron-runtime/testing/oracle/mamba_target_chat_profile_structured.json OUTPUT_DIR=/workspace/nemotron-runtime/testing/oracle/mamba_layer0_target_chat_trace_structured ./tools/oracle/generate_mamba_target_trace_fixture.sh
PROFILE=/workspace/nemotron-runtime/testing/oracle/mamba_target_chat_profile_toolish.json OUTPUT_DIR=/workspace/nemotron-runtime/testing/oracle/mamba_layer0_target_chat_trace_toolish ./tools/oracle/generate_mamba_target_trace_fixture.sh
PROFILE=/workspace/nemotron-runtime/testing/oracle/mamba_target_chat_profile_markdownish.json OUTPUT_DIR=/workspace/nemotron-runtime/testing/oracle/mamba_layer0_target_chat_trace_markdownish ./tools/oracle/generate_mamba_target_trace_fixture.sh
PROFILE=/workspace/nemotron-runtime/testing/oracle/mamba_target_chat_profile_markdown_flat.json OUTPUT_DIR=/workspace/nemotron-runtime/testing/oracle/mamba_layer0_target_chat_trace_markdown_flat ./tools/oracle/generate_mamba_target_trace_fixture.sh
python3 ./tools/oracle/make_mamba_trace_phase_dt_variant.py --source-root ./testing/oracle/mamba_layer0_target_chat_trace_markdown_headerless --output-root ./artifacts/benchmarks/generated_fixtures/mamba_phase_user_turn_1_tail_prefill_dt_1p25 --phase-name user_turn_1_tail_prefill --dt-scale 1.25
```

## Runtime Control

Current runtime config support includes:

- `NEMOTRON_PREFIX_CACHE=0` to force the uncached path and disable prefix-cache lookup/publication
- `NEMOTRON_FORWARD_DEBUG=1` to emit stage/layer progress from the correctness-first composed forward path
- `NEMOTRON_FORWARD_BUILD_DEBUG=1` to emit build-time phase and per-layer timing from `SingleTokenForwardModel::Create(...)`
- `NEMOTRON_FORWARD_BUILD_THREADS=<n>` to override the conservative default layer-construction concurrency used during model build timing / startup
- `NEMOTRON_BENCH_SAFETY_HEADROOM_GIB=<n>` to override the decode bench bootstrap safety-headroom budget
- `NEMOTRON_BENCH_WARNING_HOST_BUDGET_GIB=<n>` to set the decode bench low-memory warning threshold based on `MemAvailable + SwapFree`
- `NEMOTRON_BENCH_ABORT_ON_LOW_HOST_BUDGET=1` to make the decode bench fail fast when the host budget falls below that threshold
- `NEMOTRON_DISABLE_DENSE_DEVICE_PLAN_SURFACE=1` to disable the dense-native device-plan surface (enabled by default)
- `NEMOTRON_EXPERIMENTAL_DENSE_DEVICE_PLAN_SURFACE_FAMILY=attention|expert|other|all` to scope that experiment to one dense family during 16-token decode validation
- `NEMOTRON_EXPERIMENTAL_DENSE_DEVICE_PLAN_SURFACE_TENSORS=<comma-separated substrings>` to scope the dense-native experiment to specific tensor-name classes such as `gate.weight`, `fc1_latent_proj.weight`, or `fc2_latent_proj.weight`
- `NEMOTRON_GROUPED_NVFP4_DEBUG=1` to print grouped routed-expert cuBLASLt failure stage / status on the decode path
- `NEMOTRON_ROUTED_MOE_BACKEND=auto|custom|flashinfer` to select the routed MoE serving backend (`auto` uses FlashInfer if a compatible plugin is present, otherwise the accepted custom fused backend)
- `NEMOTRON_ROUTED_MOE_BACKEND_STRICT=1` to fail instead of falling back when `flashinfer` is requested but unavailable
- `NEMOTRON_FLASHINFER_MOE_LIBRARY=/path/to/libnemotron_flashinfer_moe.so` to override the routed FlashInfer plugin lookup path
- `NEMOTRON_FLASHINFER_WEIGHT_SURFACE=legacy_trt_prepared|cutlass_raw` to choose which routed expert-weight contract the FlashInfer seam prepares (`legacy_trt_prepared` remains the default until the CUTLASS plugin lands; `cutlass_raw` is the new CUTLASS-aligned raw packed NVFP4 seam)
- `NEMOTRON_EAGER_ROUTED_NVFP4_LOOKUPS=1` to eagerly prepare all routed-expert NVFP4 lookup tables at model-build time for graph-capture experiments; this is currently experimental and not a recommended Spark default
- `NEMOTRON_ROUTED_LOOKUP_PREFETCH_TOPN=<n>` to try a bounded routed-expert hot-set prefetch on the first cold MoE miss; this is also experimental and currently not a recommended Spark default

DGX Spark operating rule for heavy decode / startup benches:

- run one heavy model-build or decode benchmark at a time
- run one heavy manifest-backed test at a time as well
- inspect the decode-bench `memory_snapshots` JSON field before trusting a result
- on Spark, treat `host_budget_bytes = MemAvailable + SwapFree` as the main safety signal and `cudaMemGetInfo()` as advisory only
- prefer `NEMOTRON_BENCH_ABORT_ON_LOW_HOST_BUDGET=1` with a warning threshold of at least `20 GiB` for unattended or long-running runs
- for the current test procedure and expected failures, see `../proj-2026-04-04-nvfp4-activation-quant-alignment/TEST-RUNBOOK.md`

Current startup timing direction on the real manifest-backed build-only path:

- baseline build-only artifact: about `406.2s` model build
- after device-side BF16 / FP8 materialization: about `133.1s`
- after conservative concurrent layer construction: about `118.3s`
- later detailed hotspot baseline with `4` build workers: about `90.6s`
- current best build-only result with lazy aligned routed-expert NVFP4 materialization: about `15.1s`
- current first real manifest-backed first-token decode:
  - environment bootstrap about `2.1s`
  - model build about `14.0s`
  - first token about `11.9s`
  - end-to-end about `27.9s`
- three follow-up constructor experiments were measured and rejected because they regressed wall time:
  - pooled expert dense control uploads
  - async NVFP4 pooled expert upload
  - pooled Mamba norm/conv/state tensor uploads
- later startup-policy follow-ups also lost:
  - plain `mmap` still beat `mmap + prefetch`
  - `readall` pushed environment build to about `74.1s` and then failed before model creation completed

Current loader/bootstrap support includes:

- verified `manifest.json` decode and file-range/checksum validation
- manifest-backed runtime bootstrap via `RuntimeEnvironment::BuildFromManifestFile(...)`
- runtime-owned artifact bytes through `ArtifactLoader`, with explicit `mmap` and eager-read load modes
- optional benchmark-only `mmap` prefetch hints so startup-policy experiments can be measured without changing the default runtime path
- optional `/proc/meminfo`-based host-memory clamp during runtime bootstrap for GB10 memory-budget planning
- typed `TensorCatalog` descriptors for manifest-backed startup, including explicit `block_scales` and `tensor_scale` auxiliary-role resolution
- aligned `WeightArenaPlan` placement descriptors for future read-only weight upload
- materialized read-only `WeightArena` storage with copied tensor bytes
- arena-resident `KernelCatalog` descriptors for later operator setup
- manifest-backed `ModelSchedule` metadata that now enumerates ordered layers, global tensors, and coarse per-layer family flags from tensor names
- operator-specific `GemmCatalog` descriptors plus `GemmLaunchPlan` / heuristic-cache support for the first GEMM execution path
- backend-facing `PreparedGemmExecution` setup plus a service-owned GEMM heuristic cache in `RuntimeEnvironment`
- operator-specific `EmbeddingCatalog` descriptors plus manifest-backed runtime-environment ownership for embedding weights
- backend-specific `cublasLt` GEMM plan generation with alignment, transform, operand-order, validated contract, and scale-mode validation
- the first real executable forward primitive: dense row-major FP32 GEMM via `cublasLtMatmul`, validated against a CPU reference path
- the dense row-major upload path now accepts BF16-backed checkpoint descriptors and converts them into the current FP32 execution surface, which removes the most immediate dense-format blocker for registry-backed layer construction
- request-owned FP32 device tensors for activation/output staging, plus a device-tensor execution path for the first dense GEMM primitive
- a narrow `RequestExecutionContext` that now owns hidden/residual/scratch device buffers, request-local KV pages, FP32 Mamba recurrent-state storage, and FP32 Mamba conv-state storage for the first single-request forward path
- the first CUDA-backed primitive composition ops for real layer slices: FP32 residual add and FP32 RMSNorm, each validated against CPU references
- a correctness-first scaled-FP8 linear helper that reproduces checkpoint FP8 input quantization and dequantized-weight execution for the first Mamba path, with a CPU fallback when the local dense `cublasLt` path is shape-fragile
- a first real layer-0 Mamba slice that composes outer RMSNorm, scaled-FP8 `in_proj`, request-local conv and SSM state updates, grouped gated RMSNorm, scaled-FP8 `out_proj`, and residual merge, and now also supports correctness-first multi-row prefill replay by scanning token rows sequentially over the recurrent state
- a first real layer-1 expert slice that composes outer RMSNorm, dense router logits, correctness-first top-k dispatch, scaled-FP8 latent/shared-up projections, routed/shared NVFP4 expert matmuls under the validated host pack/dequant contract, dense latent projection, and residual merge, and now also supports multi-row prefill replay
- a registry-backed composed forward scaffold under `runtime/include/nemotron/single_token_forward_model.h` and `runtime/src/api/single_token_forward_model.cpp`, plus a tested `SingleTokenForwardPlan` builder that centralizes ordered layer dispatch, request-local KV sizing, per-layer Mamba state offsets, and a first `RunPrefill(...)` entry point for token matrices
- a manifest-backed single-request full-forward path that now executes the real `88`-layer checkpoint end to end in correctness-first mode through `full_forward_manifest_smoke_test`, using no-copy manifest bootstrap, lazy per-layer weight materialization, and CPU fallbacks where the current dense BF16 / FP8 execution paths are still shape-fragile
- the serving-path GPU migration is now mostly implemented:
  - dense + scaled-FP8 default execution stays on GPU
  - attention layout/staging stays on GPU
  - Mamba scan/state math stays on GPU
  - expert routed/shared large-tensor math stays on GPU
  - trace/debug downloads still exist behind explicit trace paths
  - one tiny expert-selection metadata copy still remains in the default path to drive the current host-dispatched expert launch interface
- the warmed decode harness under [benchmarks/decode_bench](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/benchmarks/decode_bench) now shows:
  - manifest bootstrap about `2.1s`
  - build-only model construction about `15.1s` with lazy aligned routed-expert NVFP4 materialization
  - first real single-token decode about `11.9s`
  - on the same manifest path, a warmed second token about `0.73s`
  - the rollout's rough `<30s` end-to-end first-token target is now met on the current benchmark path
- the bench now also supports real append-only multi-token generation on one request context through `RunDecodeStep(...)`, and the first longer run shows:
  - [single_token_decode_20260331T090018Z_cuda132.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260331T090018Z_cuda132.json)
  - `environment_build_ms ≈ 2.0s`
  - `model_build_ms ≈ 10.6s`
  - `16` generated decode steps from a single synthetic seed token
  - full sequence about `36.3s`
  - first `4` tokens average about `5.0s/token`
  - last `8` tokens average about `1.0s/token`
  - last `4` tokens average about `0.94s/token`
  - this longer run is the better current steady-state proxy than the earlier one-token microbench
- the routed-expert decode path now has a working custom packed-NVFP4 fused fast path on the default lazy-resident serving path:
  - [single_token_decode_20260331T185338Z_fused_moe_custom_routed_retry_16tok.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260331T185338Z_fused_moe_custom_routed_retry_16tok.json)
  - output tokens stayed exact: `[5130 x16]`
  - full 16-token sequence about `26.9s`
  - `grouped_routed_expert_fastpath_uses = 640`
  - `expert_selection_metadata_downloads = 395`
- an eager all-expert routed-lookup graph-prep mode was measured and rejected as a Spark default:
  - [single_token_decode_20260331T185639Z_fused_moe_custom_routed_eager_16tok.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260331T185639Z_fused_moe_custom_routed_eager_16tok.json)
  - output tokens stayed exact, and decode had zero host MoE metadata downloads
  - but model build rose to about `201.9s`, hot decode to about `59.8s`, and memory pressure became unacceptable
  - graph-capture work therefore needs a bounded/hybrid expert residency policy, not eager preparation of every routed expert
- a first bounded-prefetch variant was also measured and rejected as a Spark default:
  - [single_token_decode_20260331T193926Z_fused_moe_prefetch32_16tok.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260331T193926Z_fused_moe_prefetch32_16tok.json)
  - output tokens stayed exact
  - but `expert_selection_metadata_downloads` only moved from `395` to `389`
  - while model build rose to about `23.6s` and hot decode to about `49.2s`
  - so the next graph-prep step needs a smarter hybrid residency policy than simple first-miss top-`N` prefetch
- the first deterministic model-cache path is now implemented under [model_cache.h](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/runtime/include/nemotron/model_cache.h) and [model_cache.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/runtime/src/api/model_cache.cpp), with [single_token_forward_model.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/runtime/src/api/single_token_forward_model.cpp) now supporting `CreateFromCache(...)`
- current model-cache status is mixed:
  - functional cache-backed decode works and still predicts token `5130`
  - but the current execution-ready cache payload is about `102.1 GB`, and cache-backed startup is about `113.7s`
  - so the cache subsystem is now real, but the payload format needs redesign before it becomes the next serving-path TTFT win
- device-resident FP32 dense-weight upload for the first dense GEMM path, plus an uploaded-weight execution path that avoids per-call weight copies
- device-resident NVFP4 packed-weight upload that preserves logical `block_scales` / `tensor_scale` buffers and now also prepares separate padded-swizzled execution-layout scale buffers for `cuBLASLt`
- the first real runtime-side NVFP4 GEMM executor for the validated row-major `cuBLASLt` contract, now pointing `A/B_SCALE_POINTER` at padded-swizzled execution scales and folding FP32 per-tensor scales into `alpha`
- a host-side FP32-to-NVFP4 activation-packing utility for runtime-side mixed-precision benchmarking and future activation staging work
- a device-side FP32-to-NVFP4 activation-packing path with dynamic tensor-scale derivation, plus a direct `device fp32 activations -> packed NVFP4 -> cuBLASLt` execution slice for the mixed-precision runtime path
- optional fixed activation tensor-scale support for the host-side and device-side NVFP4 packers, so the runtime can reproduce checkpoint-provided `input_scale` values when needed
- device-resident FP32 embedding-table upload and lookup, now also accepting BF16-backed checkpoint embedding descriptors, plus a minimal `token ids -> embeddings -> dense projection` fragment validated against a CPU reference
- a standalone GB10 dense-GEMM microbenchmark harness with checkpoint-derived default shape families, workspace sweeps, and JSON result capture
- a standalone GB10 runtime-side NVFP4 GEMM benchmark harness with matching case names for direct comparison against the FP32 dense baseline, explicit `activation_pack_ms` reporting, and a default device-staging mode with host-staging retained for comparison
- a standalone GB10 NVFP4 contract-validation harness with JSON artifact capture for cuBLASLt operand-order and transform experiments
- a standalone GB10 paged-attention benchmark harness on top of the BF16 cuDNN FE backend, with decode/prefill benchmark cases and JSON artifact capture
- a standalone GB10 Mamba cache-format benchmark harness covering FP32, FP16, FP16 plus deterministic Philox-style stochastic rounding, exploratory grouped-INT8, a recurrence-like `mamba_update` path, a one-step `fixture_update` oracle path, a multi-step `fixture_chain` oracle path with cross-seed drift statistics, and a target-shaped `fixture_trace` path that replays a long shared-root plus append-only multi-turn chat profile with automatic phase-end checkpoints
- a current oracle-trace family with narrative, structured, tool-style, and markdown-heavy serialization patterns, plus summary and guardrail reports over the latest family artifacts
- a standalone GB10 stochastic-rounding PTX probe that checks whether `sm_121` accepts a direct `cvt.rs.f16.f32` path under the pinned CUDA 13.2 toolchain
- a standalone GB10 loader-mode benchmark harness for `mmap` vs eager-read startup measurement
- initial paged-attention runtime scaffolding with explicit KV bytes-per-token/page derivation, fixed-page KV allocation, and padded `INT32` page-table planning
- a first executable BF16 cuDNN FE paged-attention backend path with CPU-reference-validated decode and causal-prefill coverage on real device execution
- a first real attention-layer slice that composes RMSNorm, uploaded linear ops, cuDNN FE paged attention, output projection, and residual merge
- a checkpoint-derived layer-7 attention oracle under `tools/oracle/` and `testing/oracle/` that now gates final output plus request-local paged KV contents for that slice
- a checkpoint-derived layer-0 Mamba decode-block oracle under `tools/oracle/` and `testing/oracle/` that now gates final output plus updated request-local conv and SSM state for the first Mamba slice
- a checkpoint-derived layer-1 expert oracle under `tools/oracle/` and `testing/oracle/` that now gates router logits, selected experts and weights, routed latent output, shared expert output, mixer output, and final layer output for the first MoE slice
- a first full-model single-token oracle generator under `tools/oracle/` that can walk the complete checkpoint contract offline, though the current pure-Python MoE path is still too slow for routine generation across all `88` layers
- a first practical composed prefix-prefill oracle gate under [testing/api/prefill_prefix_oracle_test.cpp](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/testing/api/prefill_prefix_oracle_test.cpp), now using a `4`-token short-chat fixture stopped at layer `7`
- checkpoint-derived NVFP4 oracle-fixture tooling under `tools/oracle/`, plus a metadata-driven fixture-backed C++ numerical oracle test that now gates real routed-expert `up_proj` / `down_proj` and shared-expert `down_proj` weights with fixed activation tensor scales from the local Nemotron checkpoint
- layer-derived Mamba decode-update oracle-fixture tooling under `tools/oracle/`, plus a CUDA fixture test that now gates the layer-0 decode-step state update and output against dumped checkpoint-derived gold tensors
- target-shaped Mamba trace-fixture tooling under `tools/oracle/`, plus a GB10 benchmark path that now covers the earlier synthetic serving-profile trace and multiple oracle-derived tokenized multi-turn proxy profiles, with comparison tooling for phase-by-phase drift analysis and family-summary reporting
- standalone expert-layer oracle coverage for both:
  - layer `1` with `shared_down_family = nvfp4`
  - layer `3` with `shared_down_family = scaled_fp8`

Current composed-forward reality:

- the narrowed `4`-token prefill oracle through layer `7` is now green after fixing the oracle-side TF32 artifact
- the first real remaining composed-forward blocker is now the all-layer single-token decode oracle
- the decode oracle now has:
  - stop-layer override support
  - one-process stop-layer sweep support
  - captured-layer-only comparisons for truncated localization runs
- the forward model now owns its uploaded embedding table, final norm, LM head, and prepared attention / Mamba / expert slices, so repeated runs do not recreate those static objects inside `RunSingleToken(...)`
- the first failing full-decode layer is now localized to layer `19`
- layer `19` itself is not the obvious standalone culprit:
  - the isolated layer-19 expert oracle passes on the real layer-18 decode input
  - the full-decode layer-19 output also matches the standalone layer-19 output on that same dumped runtime layer-18 input
  - the remaining decode bug surface is therefore upstream of layer `19`, not inside the layer-19 expert slice
- the newer state-aware decode sweep shows why that layer-19 failure is misleading as a root cause:
  - standalone exact-input Mamba decode slices are clean at layers `13` and `18`
  - per-layer oracle Mamba state snapshots are now captured in the full single-token fixture
  - Mamba state drift shows up materially earlier than the first hidden-output gate failure, with the strongest current SSM-state hotspot at layer `9` and the strongest conv-state hotspot at layer `15`
  - the current decode problem therefore looks like earlier cumulative state drift that later becomes visible as a layer-19 hidden-output failure
- the latest hotspot follow-up makes that more specific:
  - exact-input Mamba runtime-input replays are now clean at layers `9`, `13`, `15`, and `18`
  - in this single-token path those Mamba layers begin from zero state, so the hotspot drift is being fed in through hidden-input mismatch from earlier layers rather than from bad per-layer recurrent carryover
  - the next useful boundary to inspect is therefore the non-Mamba producer immediately before those hotspots
- the next upstream producer checks are now clean too:
  - exact-input expert runtime-input replays are green at layers `8` and `14`
  - that pushes the likely decode source earlier than those MoE producers
  - layer `7` attention is now the clearest next boundary to inspect
- the latest decode-localization round moved one layer earlier again:
  - layer `7` attention is clean on exact dumped runtime input
  - layer `6` Mamba is clean on exact dumped runtime input
  - the earlier layer-5 expert hotspot is now understood much more precisely:
    - FP8 activation round-trip matches the oracle exactly
    - FP8 weight dequant matches the oracle exactly
    - the remaining `fc1_latent` delta came from the host accumulation surface in the correctness-first `scaled_fp8_linear` path
  - after tightening that host accumulation:
    - the layer-5 exact-input expert replay now passes
    - the rebuilt all-layer decode sweep now stays green through `stop=23`
    - the first hidden-output gate failure moved from layer `19` to layer `24`
  - the current decode blocker is therefore no longer the old layer-19 boundary; it is the new `24-31` region
- the latest follow-up narrowed that region again:
  - exact-input replays are now green through layers `24`, `25`, `26`, `27`, `28`, `29`, `30`, and `31`
  - so the current `24-31` decode failure band is not yet pointing at another obvious broken local slice
  - after the two accumulation fixes, the remaining blocker in that band currently looks more like chained amplification of smaller upstream differences than a new standalone layer bug
- the first end-to-end functional decode check is now also in hand:
  - final decode top-1 token agrees exactly with the oracle
  - top-5 overlap is `5/5`
  - top-10 overlap is `10/10`
  - top-20 overlap is `17/20`
  - top-50 overlap is `45/50`
  - so the remaining all-layer decode blocker is now partly about how strict the hidden-state/logit localization gate should be, not just about raw runtime breakage
- that gate split is now implemented:
  - stop-layer decode sweeps still use strict hidden-state localization
  - the full all-layer decode oracle now gates on functional decode behavior and a final-logit `rel_l2` tripwire
  - historical note: that oracle passed in an earlier 60-test tree, but the current tree has moved on
- verification status after this round:
  - the old `60/60` statement is stale
  - the latest guarded CUDA-host run executed `65` registered tests with `59` passes and `6` failures
  - see `../proj-2026-04-04-nvfp4-activation-quant-alignment/TEST-RUNBOOK.md` for the current procedure and failure list

Latest throughput reality on DGX Spark:

- the packed BF16 / FP8 linear pass plus reduced-sync helper work is now in the runtime default path
- correctness is not currently green end to end; use the dedicated runbook for the latest pass/fail state
- current aligned uncached decode measurement:
  - bootstrap `≈ 2.1s`
  - model build `≈ 9.6s`
  - first token `≈ 5.1s`
  - warmed decode `≈ 0.73s/token`
- so TTFT improved materially, but steady-state token throughput is still bottlenecked by the remaining expert-dispatch and deeper fusion work rather than by startup alone
- latest control-path cleanup kept correctness green and moved warmed decode only slightly:
  - updated aligned run: warmed decode `≈ 0.729s/token`
  - that confirms the next meaningful throughput work is deeper MoE / Mamba execution restructuring rather than more small alloc/copy cleanup
- the first decode-only fused Mamba inner pass is now also in the runtime default path:
  - aligned fused artifact: [single_token_decode_20260331T_mamba_decode_fused_threads4.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260331T_mamba_decode_fused_threads4.json)
  - bootstrap `≈ 2.08s`
  - model build `≈ 9.86s`
  - first token `≈ 5.05s`
  - warmed decode `≈ 0.728s/token`
  - interpretation: Mamba inner-kernel fusion is correct, but it only moved steady-state decode by about `1 ms/token`
- grouped / indirect routed-expert dispatch is now in the single-token serving path:
  - grouped artifact: [single_token_decode_20260331T_grouped_routed_experts_device_lookup_threads4.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260331T_grouped_routed_experts_device_lookup_threads4.json)
  - bootstrap `≈ 2.10s`
  - model build `≈ 8.77s`
  - warmed decode `≈ 0.711s/token`
  - predicted token still `5130`
- interpretation:
  - grouped routed-expert dispatch is correct and measurably better than the previous `~0.729s/token` path
  - but it is not the final throughput unlock
  - the next big throughput wins are now more likely to come from expert residency/cache strategy, a lower-overhead routed-expert backend, and batching/MTP than from more small dispatch cleanup
- latest fused-MoE graph-prep follow-up:
  - accepted artifact: [single_token_decode_20260331T200103Z_step_runtime_stats_16tok.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260331T200103Z_step_runtime_stats_16tok.json)
  - exact output tokens stayed `[5130 x16]`
  - `hot_mean_ms ≈ 25557.2`
  - full selection-metadata downloads are now gone on the default fused routed-MoE path:
    - `expert_selection_metadata_downloads = 0`
  - the remaining MoE graph blocker is now the smaller routed lookup repair surface:
    - `routed_lookup_repair_downloads = 395`
    - `routed_lookup_repair_experts = 3130`
  - those repairs decay materially across the sequence, but do not hit zero by token `16`
  - so the next graph-capture step is not immediate post-first-token capture; it is a smarter hybrid expert-residency policy first
- latest accepted fused-MoE cleanup result:
  - artifact: [single_token_decode_20260331T_graph_readiness_cleanup_16tok.json](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/artifacts/benchmarks/single_token_decode_20260331T_graph_readiness_cleanup_16tok.json)
  - exact output tokens stayed `[5130 x16]`
  - full `16`-token sequence improved to `≈ 15897.4 ms`
  - first `4` tokens `≈ 1936.5 ms/token`
  - last `8` tokens `≈ 614.9 ms/token`
  - last `4` tokens `≈ 608.6 ms/token`
  - `expert_selection_metadata_downloads = 0`
  - `routed_lookup_repair_downloads = 395`
  - `routed_lookup_repair_experts = 3130`
  - the improvement came from removing fused-MoE dead weight, not from suppressing lookup repair
  - the new graph-readiness counters confirm there is still no capture-safe tail in the `16`-token horizon:
    - `hot_graph_safe_steps = 0`
    - `hot_max_graph_safe_streak = 0`
    - `hot_tail_graph_safe_streak = 0`
    - `hot_first_graph_safe_tail_token_index = -1`
  - so the next accepted throughput step is still smarter routed-lookup-repair suppression before steady-state CUDA graph capture
