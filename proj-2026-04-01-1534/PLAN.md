# Plan: Reference-Translated Decode Roofline Loop

Project directory: `./proj-2026-04-01-1534`

## Goal

Get correct steady-state single-token decode as close as practical to roofline on the RTX 5090 for the Nano 30B-A3B NVFP4 model.

Working target: `~20 ms/token`.

Current measured state (2026-04-02, post monolithic expert refactor):

- hot steady-state mean: `1805.328 ms/token`
- steady-state generated tokens/sec: `0.554`
- expert staging bytes uploaded per run: `0` (all 23 layers monolithic-resident)
- linear reference fallbacks: `0` (all cuBLASLt fastpath)
- artifact: `artifacts/benchmarks/nano_fused_decode_16tok_steady_state_20260402T031714Z_cuda130.json`

Original baseline: `22859 ms/token` → current `1805 ms/token` = **12.7x speedup so far**.

This plan is only about steady-state decode throughput, but no throughput checkpoint counts unless the affected path is still correct.

## Execution Rules

- Correctness is a hard gate. Do not accept a faster path that fails either:
  - the fixed 16-token Nano prompt checks in `nano_16_token_correctness_test`
  - the real-model split-prefill smoke in `full_forward_manifest_smoke_test`
- For every hot-path optimization, inspect the fastest proven reference design first and translate it into repo-owned code that we compile ourselves. Do not adopt an external runtime as a dependency.
- If we intentionally diverge from the fastest known reference design, record why the divergence is required for repo control, prefix-cache work, startup-time work, or local build/runtime constraints.
- Measure after every major checkpoint. Do not postpone artifact capture and Nsight profiling to the end.
- Do not wire a decode-only optimization in as the generic multi-token default path without an explicit dispatcher.
- Always check `nvidia-smi` for stale processes and free VRAM before running tests or benchmarks.

## Current Reality (updated 2026-04-02)

What's landed and working:
- Linear device fastpath (cuBLASLt) enabled with 128x4 activation scale layout — zero reference fallbacks
- Monolithic expert tensor residency — all 23 expert layers fully GPU-resident via contiguous 3-buffer allocations (6 cudaMalloc per layer vs 1,024 before), zero per-token expert weight uploads
- VRAM-aware allocation — weights loaded first, KV cache and Mamba state sized to measured remaining VRAM
- Persistent attention metadata buffers — no per-call cudaMalloc for page tables/seq lengths
- Removed 6 unconditional cudaDeviceSynchronize calls from attention/Mamba/MoE hot paths
- Device-side argmax — 4-byte token ID copy instead of vocab_size×4 logits download

What we know about the remaining 1,805 ms/token gap (from Codex code review):
- **Serialized execution is the dominant bottleneck.** Every RmsNorm, ResidualAdd, dense GEMM, NVFP4 GEMM, NVFP4 activation pack, embedding lookup, and argmax does an unconditional `cudaDeviceSynchronize`. That's hundreds of host-device round-trips per token.
- **~281 temporary `cudaMalloc/cudaFree` per token.** Each allocation also calls `cudaGetDeviceCount()`. This is pure overhead.
- **Expert fast path allocates 4 tensors it never uses** before reaching the monolithic return.
- **Attention per-call BF16 buffer allocation** still happens even with persistent metadata.
- The scalar attention fallback (1 thread per query) exists but is likely second-order at short sequence lengths (only 6 attention layers).
- The fused Mamba and MoE kernels are single-CTA (grid=1) which limits GPU utilization but is correct for single-token decode.

What we still need to measure:
- Exact wall-time breakdown via Nsight Systems: how much is in sync calls vs kernel compute vs allocation overhead
- Whether removing syncs reveals kernel compute as the next bottleneck or whether there are more structural issues

## Reference Anchors

Translate from these upstream designs, not from generic intuition:

- `vllm/vllm/v1/attention/backends/utils.py`: explicit decode-vs-prefill split
- `vllm/vllm/v1/attention/ops/chunked_prefill_paged_decode.py`: decode-oriented paged attention path
- `TensorRT-LLM/tensorrt_llm/_torch/attention_backend/trtllm_gen.py`: separate context and generation support
- `TensorRT-LLM/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunnerParams.h`: distinct kernel types for context, generation, speculative generation

## Shared Iteration Inputs

- Manifest: `artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json`
- All runs require: `NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH=1 NEMOTRON_FORWARD_FUSED_MAMBA_DECODE=1 NEMOTRON_FORWARD_FUSED_MOE_DECODE=1`
- Pre-test: always run `nvidia-smi` to check for stale processes and verify free VRAM
- Core smoke test: `env NEMOTRON_FORWARD_MANIFEST=... NEMOTRON_FORWARD_BUILD_MODEL=1 NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH=1 NEMOTRON_FORWARD_FUSED_MAMBA_DECODE=1 NEMOTRON_FORWARD_FUSED_MOE_DECODE=1 ./build-phase1-tests/testing/full_forward_manifest_smoke_test`
- Core benchmark: `env NEMOTRON_BUILD_DIR=build-benchmarks NEMOTRON_FORWARD_MANIFEST=... NEMOTRON_FORWARD_LINEAR_DEVICE_FASTPATH=1 NEMOTRON_FORWARD_FUSED_MAMBA_DECODE=1 NEMOTRON_FORWARD_FUSED_MOE_DECODE=1 bash benchmarks/nano_fused_decode/run_default_bench.sh --mode=steady-state --decode-tokens 16`

## Steps

- [x] **1. Make the linear fastpath correct on the real benchmark path**
  Done. cuBLASLt NVFP4 fastpath with 128x4 activation layout. Zero reference fallbacks. Smoke PASS.

- [x] **2. Re-baseline with real operator evidence**
  Done. Expert staging was 92% of runtime (588 GB uploaded per run). Drove the expert residency work.

- [x] **3. Remove unconditional hot-path synchronization and transient attention metadata churn**
  Done. Persistent attention buffers, removed 6 cudaDeviceSynchronize calls. Smoke PASS.

- [x] **4. Routed-expert residency — monolithic tensors matching vLLM memory layout**
  Done. All 23 expert layers monolithic-resident (15.4 GB in 6 cudaMalloc per layer). Zero per-token uploads. VRAM-aware KV cache sizing. Decode at 1,805 ms/token.

- [ ] **5. Profile and identify the next bottleneck**
  Goal:
  - decompose the remaining 1,805 ms/token into operator-level costs, with specific focus on host-side overhead
  Scope:
  - pure measurement — no code changes
  Implementation:
  - run `--mode=profile-ready` under Nsight Systems to capture a clean single-decode-step trace
  - measure and report separately:
    - total CUDA API time (`cudaDeviceSynchronize`, `cudaMalloc`, `cudaFree`, `cudaMemcpy`)
    - host idle gaps between kernel launches
    - top-10 kernel launches by duration
    - number of `cudaDeviceSynchronize` calls per token
    - number of `cudaMalloc/cudaFree` calls per token
  - save the Nsight capture under `artifacts/profiles/`
  - write a progress note with the measured breakdown and the evidence-based next target
  Accept when:
  - an Nsight capture exists for the monolithic-resident fastpath decode
  - the sync/alloc overhead is quantified separately from kernel compute time
  - the next bottleneck is identified from trace data
  Key files: `benchmarks/nano_fused_decode/run_nsight_capture.sh`

- [ ] **6a. Remove explicit cudaDeviceSynchronize from hot-path ops + fix dead allocations**
  Goal:
  - remove unconditional `cudaDeviceSynchronize()` from all hot-path operators
  - move dead expert allocations after the fused-path early return
  - remove per-alloc `cudaGetDeviceCount()` (check once at startup)
  Scope:
  - sync removal only — no allocation pattern changes yet
  - IMPORTANT: `cudaFree` is also an implicit sync barrier, so this step alone may show muted gains. The full benefit requires step 6b (allocation reuse) to also land.
  Safety:
  - all current ops run on the default CUDA stream; stream ordering guarantees correctness without explicit syncs between device ops
  - keep syncs that precede host reads (D2H copies are blocking anyway via `cudaMemcpyDeviceToHost`)
  - error localization gets harder — async kernel faults surface at the next blocking call, not at the failing op
  Confirmed sync sites to remove:
  - `primitive_ops.cu:89,107` — RmsNorm, ResidualAdd
  - `dense_gemm_runner.cpp:325` — dense GEMM
  - `nvfp4_gemm_runner.cpp:397` — NVFP4 GEMM
  - `device_nvfp4_matrix.cu:424,432,449,470` — NVFP4 activation pack (4 syncs per pack)
  - `embedding_table.cu:183,190,218` — embedding lookup
  - `device_argmax.cu:90` — argmax
  - `scaled_fp8_linear.cu:213,296` — scaled FP8 quantize + GEMM (used by Mamba projections)
  Also fix:
  - `device_tensor.cpp:50,125` — remove per-alloc `cudaGetDeviceCount()`
  - `expert_layer.cpp:1442-1445` — move 4 dead tensor allocs after fused early return
  Key files: `primitive_ops.cu`, `dense_gemm_runner.cpp`, `nvfp4_gemm_runner.cpp`, `device_nvfp4_matrix.cu`, `embedding_table.cu`, `device_argmax.cu`, `scaled_fp8_linear.cu`, `device_tensor.cpp`, `expert_layer.cpp`

- [ ] **6b. Replace per-token temporary allocations with request-context scratch buffers**
  Goal:
  - eliminate ~281 `cudaMalloc/cudaFree` per token by reusing pre-allocated buffers
  Scope:
  - allocation reuse only — use request-context-owned scratch buffers instead of per-op temporaries
  - request-context scratch (not model-global) to avoid races with multiple concurrent requests
  Key allocation sites to convert:
  - `single_token_forward_model.cpp:1295-1297` — model-level hidden/residual/scratch tensors per RunTokens call
  - `mamba_layer.cpp:492-495` — per-call Mamba scratch
  - `expert_layer.cpp:1440-1445` — per-call expert scratch (for non-fused fallback)
  - `attention_layer.cpp:557-563,664-667` — per-call attention FP32 scratch + BF16 buffers
  - `device_nvfp4_matrix.cu:409,436` — temp buffers in NVFP4 activation pack
  - `scaled_fp8_linear.cu` — quantized activation temp buffer
  Approach:
  - for `token_count == 1` decode, all shapes are deterministic from layer config — allocate at request-context creation
  - add a scratch buffer pool to `RequestExecutionContext` sized for the max-shape layer in the model
  - operators take a scratch pointer + size instead of allocating internally
  Key files: `request_context.cpp`, `single_token_forward_model.cpp`, `mamba_layer.cpp`, `expert_layer.cpp`, `attention_layer.cpp`, `device_nvfp4_matrix.cu`, `scaled_fp8_linear.cu`

- [ ] **6c. Pre-allocate attention BF16 buffers and cuDNN plan/workspace**
  Goal:
  - eliminate per-call attention buffer allocations and cuDNN plan rebuilds
  Scope:
  - attention-layer-specific allocation reuse
  Key sites:
  - `attention_layer.cpp:664-667` — BF16 query, KV scatter, output buffers allocated per call
  - `attention_layer.cpp:748` — cuDNN plan built per call (when cuDNN is available)
  - `cudnn_paged_attention.cpp:133,296,363` — cuDNN plan creation, workspace alloc, execute sync
  Approach:
  - move BF16 buffers to AttentionLayerSlice::Impl, sized for max token count
  - cache cuDNN plan/workspace in Impl (rebuild only when batch plan shape changes)
  Key files: `attention_layer.cpp`, `attention_device_fallback.cu`, `cudnn_paged_attention.cpp`

- [ ] **7. Re-measure and decide: production attention or next structural fix**
  Goal:
  - after sync/alloc cleanup, re-profile to see the new bottleneck ranking
  Scope:
  - measurement + decision
  Implementation:
  - run benchmark and Nsight capture post-step-6
  - if attention is now the dominant cost → proceed to production attention kernel
  - if kernel compute is now dominant → focus on kernel optimization (multi-CTA fused kernels, better occupancy)
  - if something else dominates → address that first
  Accept when:
  - post-cleanup artifact and profile saved
  - next target chosen from evidence

- [ ] **8. Production decode attention (if step 7 says so)**
  Goal:
  - replace the scalar attention fallback with a production decode kernel
  Precondition:
  - only execute if step 7 confirms attention is a significant remaining cost
  Reference design to translate:
  - `vllm/vllm/v1/attention/ops/chunked_prefill_paged_decode.py`
  - `TensorRT-LLM/cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunnerParams.h`
  Local constraints:
  - `tokens_per_page == 16`, `head_dim == 128`, GQA ratio `32 / 2 = 16`
  Implementation:
  - explicit decode-vs-prefill dispatcher
  - production decode: paged KV, online softmax, cooperative warp reduction
  - keep scalar fallback behind env gate
  - preserve device-vs-host compare hook
  Accept when:
  - decode attention materially faster, correctness preserved
  Key files: `runtime/src/backend/attention_device_fallback.cu`, `runtime/include/nemotron/attention_device_fallback.h`, `runtime/src/backend/attention_layer.cpp`

- [ ] **9. Repeat the measure → choose → translate loop**
  Goal:
  - continue evidence-driven optimization until 20ms target or architectural limit
  Implementation:
  - save artifact + profile after each checkpoint
  - compare with `compare_artifacts.py`
  - choose one next bottleneck from evidence
  Accept when:
  - artifacts saved, next bottleneck chosen from evidence

## Progress

| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Linear fastpath correctness | done | b31feb4 | cuBLASLt NVFP4 128x4; zero fallbacks |
| 2 | Re-baseline with evidence | done | 8394930 | expert staging = 92% of runtime |
| 3 | Attention sync cleanup | done | 6fb664b | persistent buffers, removed 6 syncs |
| 4 | Monolithic expert residency | done | 5a624e1 | 23/23 resident, 0 uploads, 1805 ms/token |
| 5 | Profile next bottleneck | pending | — | focus on sync/alloc overhead, not just kernels |
| 6a | Remove explicit syncs + fix dead allocs | pending | — | muted gains alone; needs 6b |
| 6b | Request-context scratch buffers | pending | — | ~281 mallocs/token → 0 |
| 6c | Attention BF16 + cuDNN plan reuse | pending | — | attention-specific alloc cleanup |
| 7 | Re-measure and decide | pending | — | attention or kernel optimization next? |
| 8 | Production decode attention | pending | — | contingent on step 7 evidence |
| 9 | Evidence-driven loop | pending | — | |
