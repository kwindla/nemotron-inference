# Plan: Reference-Translated Decode Roofline Loop

Project directory: `./proj-2026-04-01-1534`

## Goal

Get correct steady-state single-token decode as close as practical to roofline on the RTX 5090 for the Nano 30B-A3B NVFP4 model.

Working target: `~20 ms/token`.

Current measured baseline from the saved 2026-04-01 benchmark artifact:

- hot steady-state mean: `22859.297232 ms/token`
- steady-state generated tokens/sec: `0.043746`
- artifact: `artifacts/benchmarks/nano_fused_decode_16tok_20260401T161556Z_cuda130.json`

This plan is only about steady-state decode throughput, but no throughput checkpoint counts unless the affected path is still correct.

## Execution Rules

- Correctness is a hard gate. Do not accept a faster path that fails either:
  - the fixed 16-token Nano prompt checks in `nano_16_token_correctness_test`
  - the real-model split-prefill smoke in `full_forward_manifest_smoke_test`
- For every hot-path optimization, inspect the fastest proven reference design first and translate it into repo-owned code that we compile ourselves. Do not adopt an external runtime as a dependency.
- If we intentionally diverge from the fastest known reference design, record why the divergence is required for repo control, prefix-cache work, startup-time work, or local build/runtime constraints.
- Measure after every major checkpoint. Do not postpone artifact capture and Nsight profiling to the end.
- Keep checkpoint scopes independent:
  - step 1 fixes linear/MoE fastpath correctness, not expert residency policy
  - step 3 changes attention metadata ownership and synchronization behavior, not attention math
  - step 5 changes attention math and dispatch, not metadata lifetime
- Do not wire a decode-only optimization in as the generic multi-token default path without an explicit dispatcher.

## Current Reality

- The benchmark and diagnostic surfaces already exist:
  - `nano_fused_decode_bench --mode=steady-state`
  - `nano_fused_decode_bench --mode=profile-ready`
  - `--strict-linear`
  - per-tensor `LinearOpTrace`
  - expert staging counters
  - prompt-boundary route matrix and layer probes in `nano_16_token_correctness_test`
- The largest likely win is still the linear device fastpath, but it is not yet a valid benchmark path:
  - with `NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH=1`, the real-model split-prefill check diverges
  - the first observed split-prefill divergence is already localized to expert layer `13`
- The current fused direct-MoE path stages every routed expert pair on every decode step, not just the selected experts.
- The current attention fallback path pays avoidable host-side overhead:
  - per-call auxiliary-buffer allocation/upload
  - helper-level `cudaDeviceSynchronize()` fences in the release hot path
- The next optimization after linear fastpath cannot be chosen honestly from intuition alone. It must be chosen from post-fastpath artifacts and Nsight traces.

## Reference Anchors

Translate from these upstream designs, not from generic intuition:

- `vllm/vllm/_custom_ops.py`
  - `scaled_fp4_quant`: small-`M` TensorRT-LLM backend uses a distinct NVFP4 scale-factor layout (`8x4`) when `m <= 32`
  - `scaled_fp4_experts_quant` / `silu_and_mul_scaled_fp4_experts_quant`: MoE activations are packed by token-to-expert layout, not by uploading expert weights per token
- `vllm/vllm/model_executor/layers/quantization/utils/flashinfer_utils.py`
  - `align_fp4_moe_weights_for_fi`
  - `convert_moe_weights_to_flashinfer_trtllm_block_layout`
- `vllm/vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py`
  - `prepare_static_weights_for_trtllm_fp4_moe`
  - gated `[w1, w3] -> [w3, w1]` reorder
  - offline per-expert permutation and block-scale interleave
- `vllm/vllm/v1/attention/backends/utils.py`
  - `split_decodes_and_prefills`
  - chunked-prefill metadata generation
- `vllm/vllm/v1/attention/backends/flashinfer.py`
  - persistent workspace/wrapper planning
  - explicit decode-vs-prefill execution split
- `vllm/vllm/v1/attention/ops/chunked_prefill_paged_decode.py`
  - decode-oriented paged attention path separate from context attention
- `TensorRT-LLM/tensorrt_llm/_torch/attention_backend/trtllm_gen.py`
  - separate context and generation support checks
  - persistent workspace allocation
  - paged-KV generation constraints
- `TensorRT-LLM/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunnerParams.h`
  - distinct kernel types for context, generation, and speculative generation

## Shared Iteration Inputs

Use these inputs for every checkpoint unless a step says otherwise:

- Manifest:
  - `artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json`
- Core build targets:
  - `cmake --build build-phase1-tests --target full_forward_manifest_smoke_test nano_16_token_correctness_test -j4`
  - `cmake --build build-benchmarks --target nano_fused_decode_bench -j4`
- Core correctness runs:
  - `env NEMOTRON_FORWARD_MANIFEST=$PWD/artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json NEMOTRON_FORWARD_BUILD_MODEL=1 ctest --test-dir build-phase1-tests -R '^full_forward_manifest_smoke_test$' --output-on-failure`
  - `env NEMOTRON_FORWARD_MANIFEST=$PWD/artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json ./build-phase1-tests/testing/nano_16_token_correctness_test`
- Core benchmark runs:
  - `env NEMOTRON_BUILD_DIR=build-benchmarks NEMOTRON_FORWARD_MANIFEST=$PWD/artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json bash benchmarks/nano_fused_decode/run_default_bench.sh --mode=steady-state --decode-tokens 16`
  - `env NEMOTRON_BUILD_DIR=build-benchmarks NEMOTRON_FORWARD_MANIFEST=$PWD/artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json bash benchmarks/nano_fused_decode/run_nsight_capture.sh -- --strict-linear`

## Steps

- [ ] **1. Make the linear fastpath correct on the real benchmark path**
  Goal:
  - turn `NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH=1` from a diagnostic mode into a correct hot path
  Scope:
  - only linear/MoE fastpath correctness and backend-native layout translation
  - do not add expert residency policy here
  Reference design to translate:
  - `vllm/vllm/_custom_ops.py`: small-`M` NVFP4 activation packing chooses TRT-LLM-style `8x4` scale-factor layout for decode-like shapes
  - `vllm/vllm/model_executor/layers/quantization/utils/flashinfer_utils.py`: pad intermediate dims to backend alignment requirements
  - `vllm/vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py`: precompute per-expert weight permutations and block-scale interleave once with the weights, not during decode
  Implementation checklist:
  - use `full_forward_manifest_smoke_test` split-prefill plus `nano_16_token_correctness_test` route matrix/layer probes to localize the expert-layer-13 divergence to the exact sub-op and tensor
  - audit whether the current decode activation quantization path is using the wrong NVFP4 scale-factor layout for `M=1` or other small-`M` cases
  - translate any required expert weight padding, gated-projection reorder, permutation, or block-scale interleave into create-time repacking
  - keep or extend diagnostics so failures report:
    - tensor name
    - `M/N/K`
    - backend path
    - activation scale-factor layout
    - plan-build versus execute failure
  - if a fix changes backend-native tensor layout, do it at `Create()` or load time, not per token
  Verification commands:
  - `env NEMOTRON_FORWARD_MANIFEST=$PWD/artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json NEMOTRON_FORWARD_BUILD_MODEL=1 NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH=1 NEMOTRON_FORWARD_DEBUG=1 ctest --test-dir build-phase1-tests -R '^full_forward_manifest_smoke_test$' --output-on-failure`
  - `env NEMOTRON_FORWARD_MANIFEST=$PWD/artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH=1 NEMOTRON_FORWARD_LINEAR_TRACE=1 NEMOTRON_NANO_16_TRACE_DIVERGENCE=1 NEMOTRON_NANO_16_TRACE_EMBEDDING=1 NEMOTRON_NANO_16_STRICT_LINEAR=1 ./build-phase1-tests/testing/nano_16_token_correctness_test`
  - `env NEMOTRON_BUILD_DIR=build-benchmarks NEMOTRON_FORWARD_MANIFEST=$PWD/artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH=1 NEMOTRON_FORWARD_LINEAR_TRACE=1 bash benchmarks/nano_fused_decode/run_default_bench.sh --mode=steady-state --decode-tokens 16 --strict-linear`
  Accept when:
  - the real-model split-prefill smoke passes with `NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH=1`
  - the fixed 16-token Nano prompt parity check passes on the benchmark path
  - `nano_fused_decode_bench --strict-linear` reports zero reference fallbacks
  - a new saved benchmark artifact establishes the post-fastpath baseline
  Key files:
  - `runtime/src/backend/linear_op.cpp`
  - `runtime/src/backend/scaled_fp8_linear.cu`
  - `runtime/src/backend/expert_layer.cpp`
  - `runtime/src/loader/gemm_execution.cpp`
  - `runtime/src/loader/cublaslt_gemm_plan.cpp`
  - `testing/api/nano_16_token_correctness_test.cpp`
  - `testing/api/full_forward_manifest_smoke_test.cpp`
  - `benchmarks/nano_fused_decode/nano_fused_decode_bench.cpp`

- [ ] **2. Re-baseline with real operator evidence before choosing the next bottleneck**
  Goal:
  - replace guesswork with measured post-fastpath evidence
  Scope:
  - pure measurement and artifact capture
  Implementation checklist:
  - run `nano_fused_decode_bench --mode=steady-state --strict-linear`
  - run `nano_fused_decode_bench --mode=profile-ready --strict-linear` under Nsight
  - save:
    - benchmark JSON artifact
    - stdout log
    - linear trace summary
    - expert staging counter summary
    - Nsight Systems capture
  - compare against the pre-fastpath baseline with `compare_artifacts.py`
  - write a short progress note naming the top remaining bottlenecks by evidence
  Verification commands:
  - `env NEMOTRON_BUILD_DIR=build-benchmarks NEMOTRON_FORWARD_MANIFEST=$PWD/artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH=1 NEMOTRON_FORWARD_LINEAR_TRACE=1 bash benchmarks/nano_fused_decode/run_default_bench.sh --mode=steady-state --decode-tokens 16 --strict-linear`
  - `env NEMOTRON_BUILD_DIR=build-benchmarks NEMOTRON_FORWARD_MANIFEST=$PWD/artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH=1 NEMOTRON_FORWARD_LINEAR_TRACE=1 bash benchmarks/nano_fused_decode/run_nsight_capture.sh -- --strict-linear`
  - `python3 benchmarks/nano_fused_decode/compare_artifacts.py <baseline.json> <current.json>`
  Accept when:
  - a saved artifact and profile exist for the corrected fastpath path
  - the next bottleneck ranking is justified by counters and trace data, not by stale assumptions
  Key files:
  - `benchmarks/nano_fused_decode/nano_fused_decode_bench.cpp`
  - `benchmarks/nano_fused_decode/run_default_bench.sh`
  - `benchmarks/nano_fused_decode/run_nsight_capture.sh`
  - `benchmarks/nano_fused_decode/compare_artifacts.py`
  - `docs/blackwell_inference_progress.md`

- [ ] **3. Remove unconditional hot-path synchronization and transient attention metadata churn**
  Goal:
  - eliminate obvious host-side overhead that is already visible in the current code without changing attention math
  Scope:
  - attention metadata lifetime, workspace lifetime, and helper synchronization only
  - do not replace the attention math kernel here
  Reference design to translate:
  - `vllm/vllm/v1/attention/backends/flashinfer.py`: persistent workspace buffers and wrapper `plan()` reuse
  - `TensorRT-LLM/tensorrt_llm/_torch/attention_backend/trtllm_gen.py`: aligned persistent workspace manager instead of per-call scratch allocation
  Implementation checklist:
  - move `seq_len_q`, `seq_len_kv`, `query_starts`, `page_table_k`, and `page_table_v` into persistent scratch owned by `AttentionLayerSlice::Impl` or request-context-owned attention scratch
  - size those buffers for the max supported batch/page configuration and update them in place with async H2D copies only for touched values
  - remove unconditional helper-level `cudaDeviceSynchronize()` from:
    - query layout conversion
    - KV scatter
    - fallback launch helpers
  - audit fused Mamba and fused MoE wrappers for required versus accidental per-token host synchronization
  - keep debug compare and timing fences behind explicit debug/compare gates only
  Verification commands:
  - `env NEMOTRON_FORWARD_MANIFEST=$PWD/artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json NEMOTRON_FORWARD_COMPARE_DEVICE_ATTENTION=1 ./build-phase1-tests/testing/nano_16_token_correctness_test`
  - `env NEMOTRON_BUILD_DIR=build-benchmarks NEMOTRON_FORWARD_MANIFEST=$PWD/artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json bash benchmarks/nano_fused_decode/run_default_bench.sh --mode=steady-state --decode-tokens 16`
  - `env NEMOTRON_BUILD_DIR=build-benchmarks NEMOTRON_FORWARD_MANIFEST=$PWD/artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json bash benchmarks/nano_fused_decode/run_nsight_capture.sh`
  Accept when:
  - the release decode hot path no longer performs helper-level device-wide syncs per token
  - correctness is preserved
  - benchmark artifacts or Nsight traces show reduced host-side gaps before any new attention kernel lands
  Key files:
  - `runtime/src/backend/attention_layer.cpp`
  - `runtime/src/backend/attention_device_fallback.cu`
  - `runtime/src/backend/fused_mamba_decode.cu`
  - `runtime/src/backend/fused_moe_decode.cu`

- [ ] **4. Translate the routed-expert residency design that step 2 proves we need**
  Goal:
  - stop re-uploading routed expert weights on every decode step
  Scope:
  - routed-expert residency policy only
  - reuse the backend-native packed layout from step 1
  Reference design to translate:
  - `vllm/vllm/model_executor/layers/fused_moe/layer.py`: expert weights are resident model state, not per-token uploads
  - `vllm/vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py`: static FP4 expert preparation happens once with the weights
  - `vllm/vllm/_custom_ops.py`: runtime moves token-expert activations and offsets, not weights
  Decision rule:
  - first evaluate full permanent residency for routed experts on the single-GPU 5090 path
  - if full permanent residency fits alongside the benchmark working set, prefer it over any cache
  - if it does not fit, implement a single globally budgeted residency manager
  Implementation checklist:
  - quantify post-step-1 packed routed-expert memory footprint and available headroom for:
    - all routed expert weights
    - embedding/lm_head/final norm
    - attention KV cache
    - Mamba state
    - benchmark scratch/workspaces
  - compute the routed-expert footprint from runtime facts, not guesswork:
    - use `model.plan().expert_layer_count`
    - use the actual packed weight object sizes after step 1 repacking
    - for rough planning, one Nano routed expert up+down pair is about `4.76 MiB` before scale tensors, so `128` experts is about `609 MiB` per expert layer before scale tensors
  - if full residency fits, materialize routed experts at `Create()` into backend-native device objects and prebuild any device views needed by the fused direct-MoE path
  - if full residency does not fit, implement a global residency manager keyed by:
    - layer index
    - expert id
    - projection kind
    - packed-layout version
  - never auto-size independently per layer from free VRAM
  - add counters:
    - cache/residency hits
    - misses
    - evictions
    - bytes resident
    - bytes uploaded
  Verification commands:
  - `env NEMOTRON_BUILD_DIR=build-benchmarks NEMOTRON_FORWARD_MANIFEST=$PWD/artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH=1 bash benchmarks/nano_fused_decode/run_default_bench.sh --mode=steady-state --decode-tokens 16`
  - compare `expert_staging_counters.total_bytes_uploaded` and `staging_elapsed_us` before and after
  Accept when:
  - uploaded expert bytes per token collapse on the benchmark path
  - memory use is bounded by an explicit residency decision and budget
  - correctness is preserved
  - the saved artifact shows a measurable decode gain
  Key files:
  - `runtime/src/backend/expert_layer.cpp`
  - `runtime/include/nemotron/expert_layer.h`
  - `runtime/include/nemotron/expert_staging_counters.h`
  - `runtime/src/backend/expert_staging_counters.cpp`

- [ ] **5. Translate a production decode attention path without regressing multi-token behavior**
  Goal:
  - replace the current decode hot-path attention fallback with a production generation kernel and explicit dispatcher
  Scope:
  - attention math and dispatch only
  - metadata lifetime and sync cleanup should already be done in step 3
  Reference design to translate:
  - `vllm/vllm/v1/attention/backends/utils.py`: explicit decode-vs-prefill split
  - `vllm/vllm/v1/attention/ops/chunked_prefill_paged_decode.py`: decode path distinct from context attention
  - `TensorRT-LLM/tensorrt_llm/_torch/attention_backend/trtllm_gen.py`: separate context and generation support rules
  - `TensorRT-LLM/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunnerParams.h`: distinct kernel types for context, generation, and speculative generation
  Local constraints to preserve:
  - `tokens_per_page == 16` already matches the TensorRT-LLM generation kernel family’s supported page sizes
  - local decode head shape is `head_dim == 128`, GQA ratio `32 / 2 = 16`
  Implementation checklist:
  - add an explicit attention dispatcher:
    - decode kernel for `token_count == 1`
    - separate fallback or prefill path for `token_count > 1`
    - separate debug/compare path
  - keep the existing scalar/device fallback as the forced fallback path behind an env gate
  - implement decode math with:
    - paged KV iteration
    - online softmax
    - cooperative reduction across a warp or better
    - no shared-memory footprint that scales linearly with sequence length
  - do not route multi-token prefill through the decode kernel
  - preserve the ability to compare device results against the host/reference path behind `NEMOTRON_FORWARD_COMPARE_DEVICE_ATTENTION`
  Verification commands:
  - `env NEMOTRON_FORWARD_MANIFEST=$PWD/artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json NEMOTRON_FORWARD_COMPARE_DEVICE_ATTENTION=1 ./build-phase1-tests/testing/nano_16_token_correctness_test`
  - `env NEMOTRON_BUILD_DIR=build-benchmarks NEMOTRON_FORWARD_MANIFEST=$PWD/artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json bash benchmarks/nano_fused_decode/run_default_bench.sh --mode=steady-state --decode-tokens 16`
  - `env NEMOTRON_BUILD_DIR=build-benchmarks NEMOTRON_FORWARD_MANIFEST=$PWD/artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json bash benchmarks/nano_fused_decode/run_nsight_capture.sh`
  Accept when:
  - the non-cuDNN decode path is materially faster
  - multi-token paths still have a valid dispatcher
  - correctness is preserved
  - the saved artifact and profile show attention is no longer the dominant stall
  Key files:
  - `runtime/src/backend/attention_device_fallback.cu`
  - `runtime/include/nemotron/attention_device_fallback.h`
  - `runtime/src/backend/attention_layer.cpp`

- [ ] **6. Repeat the measure -> choose -> translate loop until the next bottleneck is no longer obvious**
  Goal:
  - turn the remaining work into an evidence-driven loop instead of a one-shot guess
  Scope:
  - post-checkpoint measurement and reprioritization
  Implementation checklist:
  - after each major throughput checkpoint, save:
    - a benchmark artifact
    - an Nsight capture
    - the linear trace summary
    - the expert staging summary
  - compare against the previous artifact
  - choose only one next bottleneck at a time
  - each checkpoint note must say:
    - what changed
    - what was verified
    - what risk remains
    - which reference design was translated
    - why any intentional deviation was necessary
  Accept when:
  - artifacts are saved under `artifacts/benchmarks/`
  - profiles are saved under `artifacts/profiles/`
  - the next bottleneck is chosen from evidence rather than from stale plan text
  Key files:
  - `benchmarks/nano_fused_decode/run_default_bench.sh`
  - `benchmarks/nano_fused_decode/run_nsight_capture.sh`
  - `benchmarks/nano_fused_decode/compare_artifacts.py`
  - `docs/blackwell_inference_progress.md`

## Progress

| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Linear fastpath correctness + layout translation | pending | — | correctness gate before any roofline claim |
| 2 | Re-baseline with counters and Nsight | pending | — | choose next bottleneck from evidence |
| 3 | Attention metadata/workspace ownership + sync cleanup | pending | — | no new attention math in this step |
| 4 | Routed-expert residency translation | pending | — | prefer full residency if it fits; else global cache |
| 5 | Production decode attention translation | pending | — | explicit decode-vs-prefill dispatcher required |
| 6 | Repeat evidence-driven roofline loop | pending | — | save artifacts and justify each next move |

## Progress Log

- 2026-04-01: Rewrote this plan after a second adversarial review against the local codebase plus the public vLLM and TensorRT-LLM reference sources.
- 2026-04-01: Main corrections:
  - step 1 now names the exact NVFP4 and MoE layout/preparation references to translate
  - step 3 now cleanly separates attention plumbing from attention math
  - step 4 now has an explicit decision rule: full routed-expert residency first if it fits, otherwise a single globally budgeted residency manager
  - step 5 now mirrors the reference split between context/prefill and generation kernels instead of proposing one generic replacement kernel
  - every step now has enough scope, reference, implementation, and verification detail to iterate independently
