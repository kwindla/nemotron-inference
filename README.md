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
The GB10 performance-planning note is in [docs/gb10_performance_plan.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/docs/gb10_performance_plan.md).
The GB10 performance history log is in [docs/gb10_performance_progress.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/docs/gb10_performance_progress.md).
The Blackwell inference progress log is in [docs/blackwell_inference_progress.md](docs/blackwell_inference_progress.md).
The short GB10 operator policy note is in [docs/gb10_execution_policy.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/docs/gb10_execution_policy.md).
The provisional startup-residency note is in [docs/startup_residency_policy.md](/home/khkramer/src/nemotron-march-2026/nemotron-runtime/docs/startup_residency_policy.md).
The local vLLM runbook for Nemotron Nano NVFP4 on RTX 5090 is in [docs/vllm_rtx5090.md](docs/vllm_rtx5090.md).

## Build

Configure and build the current runtime libraries, tests, and probes:

```bash
cmake -S . -B build
cmake --build build -j
```

The default local build target is now GeForce RTX 5090 / Blackwell consumer (`CMAKE_CUDA_ARCHITECTURES=120`).
Use `-DCMAKE_CUDA_ARCHITECTURES=121` explicitly for GB10 / DGX Spark builds.

Generate a local Nano manifest for RTX 5090:

```bash
./tools/manifest/generate_forward_manifest_rtx5090_nano.sh
```

That wrapper expects the Nano checkpoint under
`artifacts/checkpoints/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` by default.

For the working local vLLM setup and benchmark path on RTX 5090, see
[docs/vllm_rtx5090.md](docs/vllm_rtx5090.md).

Generate the existing GB10-oriented manifest explicitly:

```bash
./tools/manifest/generate_forward_manifest.sh
```

Run the current C++ tests:

```bash
ctest --test-dir build --output-on-failure
```

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

Current loader/bootstrap support includes:

- verified `manifest.json` decode and file-range/checksum validation
- manifest-backed runtime bootstrap via `RuntimeEnvironment::BuildFromManifestFile(...)`
- runtime-owned artifact bytes through `ArtifactLoader`, with explicit `mmap` and eager-read load modes
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
  - with the real manifest wired in, that all-layer decode oracle now passes
- verification status after this round:
  - total tests in tree: `59`
  - the full suite is now green at `59/59` with `NEMOTRON_FORWARD_MANIFEST` set
