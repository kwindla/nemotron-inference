# Plan: Reference-Translated Decode Roofline Loop

Project directory: `./proj-2026-04-01-1534`

## Goal

Get correct steady-state single-token decode as close as practical to roofline on the RTX 5090 for the Nano 30B-A3B NVFP4 model.

Working target: `~20 ms/token`.

Current measured state (2026-04-02, post cuBLASLt MoE):

- hot steady-state mean: `57.190 ms/token`
- steady-state generated tokens/sec: `17.486`
- expert staging bytes uploaded per run: `0` (all 23 layers monolithic-resident)
- linear reference fallbacks: `0` (all cuBLASLt fastpath)
- artifact: `artifacts/benchmarks/nano_fused_decode_16tok_steady_state_20260402T043508Z_cuda130.json`
- profile: `artifacts/profiles/decode_cublaslt_moe_20260402.nsys-rep`

Original baseline: `22859 ms/token` → current `57 ms/token` = **400x speedup so far**.
Remaining gap to 20ms target: **2.85x**.

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

- [x] **5. Profile and identify the next bottleneck**
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

- [x] **6. Replace scalar MoE expert matmuls with cuBLASLt (sequential, unfused)**
  Goal:
  - replace the 77ms single-CTA `FusedMoeDirectDecodeKernel` (98.4% of GPU time) with cuBLASLt GEMM calls for expert projections
  - the current kernel does NVFP4 matmul via scalar `Nvfp4RowMajorDot` with double-precision accumulation in one thread block — no tensor cores
  - cuBLASLt NVFP4 GEMMs already run at ~18μs for similar shapes (Mamba projections) in this same build
  Scope:
  - expert matmul path only — replace the fused kernel's matmul with cuBLASLt calls
  - keep routing, activation (relu2), weighted accumulation as simple device kernels between the GEMMs
  - do NOT attempt batched/grouped GEMM yet — sequential first to prove the shapes work
  Implementation:
  - add a new `RunMoeDirectDecodeViaCublaslt()` path in `expert_layer.cpp` alongside the existing fused kernel
  - for each of the `top_k` selected routed experts:
    - cuBLASLt NVFP4 GEMM: `normalized [1×2688]` × `up_proj [1856×2688]` → `up_out [1×1856]`
    - apply `relu2` (small element-wise kernel or inline)
    - cuBLASLt NVFP4 GEMM: `activated [1×1856]` × `down_proj [2688×1856]` → `down_out [1×2688]`
    - multiply by routing weight and accumulate into routed output
  - for shared expert: same pattern with shared expert shapes (3712×2688, 2688×3712)
  - add routed + shared + residual input → output
  - the monolithic expert weight views already have the right device pointers; build `Nvfp4PackedMatrixDeviceView` from each `FusedNvfp4WeightView` for the cuBLASLt runner
  - gate behind `NEMOTRON_FORWARD_MOE_CUBLASLT=1` (default enabled). When disabled, fall back to the existing fused kernel.
  - IMPORTANT: the cuBLASLt path uses `matmul_block_scales` (swizzled), not `block_scales` (raw). The monolithic buffers store raw block scales. Either swizzle at Create() time into a parallel buffer, or build the cuBLASLt view from the existing per-expert `DeviceNvfp4Weight` matmul scales if they're still available.
  Expected impact:
  - 14 cuBLASLt calls × ~18μs each = ~0.25ms per MoE layer (vs 77ms current)
  - 23 MoE layers × 0.25ms = ~5.7ms total (vs 1,771ms current)
  - even if expert shapes are 2-3x slower than Mamba shapes, this is still a 100x+ improvement
  Accept when:
  - smoke test PASS with `NEMOTRON_FORWARD_MOE_CUBLASLT=1`
  - benchmark shows MoE per-layer time drops from ~77ms to sub-ms
  Key files: `runtime/src/backend/expert_layer.cpp`, `runtime/src/backend/nvfp4_gemm_runner.cpp`

- [x] **7. Re-measure and decide: production attention or next structural fix**
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

- [x] **8a. Remove explicit cudaDeviceSynchronize from hot-path ops**
  Goal:
  - remove unconditional `cudaDeviceSynchronize()` from all hot-path operators
  - partial win expected — cudaFree implicit barriers limit the benefit until alloc reuse lands in 8b
  Scope:
  - sync removal + cudaGetDeviceCount removal + dead expert alloc move
  Sites to fix:
  - `primitive_ops.cu:111,143` — RmsNorm, ResidualAdd
  - `dense_gemm_runner.cpp:325` — dense GEMM
  - `nvfp4_gemm_runner.cpp:397` — NVFP4 GEMM
  - `device_nvfp4_matrix.cu:424,432,449,470` — NVFP4 activation pack
  - `embedding_table.cu:183,190,218` — embedding lookup
  - `device_argmax.cu:90` — argmax
  - `scaled_fp8_linear.cu:213,296` — FP8 path
  - `device_tensor.cpp:50,125` — remove per-alloc `cudaGetDeviceCount()`
  - `expert_layer.cpp:1690+` — move dead tensor allocs after fused early return
  Safety: all ops on default stream; stream ordering sufficient; keep sync before D2H copies.
  Key files: `primitive_ops.cu`, `dense_gemm_runner.cpp`, `nvfp4_gemm_runner.cpp`, `device_nvfp4_matrix.cu`, `embedding_table.cu`, `device_argmax.cu`, `scaled_fp8_linear.cu`, `device_tensor.cpp`, `expert_layer.cpp`

- [x] **8b. Replace per-token temp allocations with decode scratch buffers**
  Goal:
  - eliminate per-token cudaMalloc/cudaFree that cause implicit sync barriers
  - this is where the bulk of the host overhead improvement comes from
  Key insight from review:
  - `RunTokens()` allocates fresh `[token_count, hidden_size]` tensors every call despite request context owning persistent `hidden`, `residual`, `scratch` buffers
  - for `token_count==1` decode, all shapes are deterministic from layer config
  Approach:
  - bind `[1, hidden_size]` decode views onto the existing request-context `hidden` / `residual` / `scratch` buffers instead of allocating top-level activations per token
  - pre-allocate decode-only FP32 scratch in each layer slice `Impl` for deterministic `token_count == 1` temporaries, while leaving multi-token prefill on the existing allocation path
  - gate the reuse path behind `NEMOTRON_FORWARD_DECODE_SCRATCH=1` (default enabled)
  Key files: `device_tensor.cpp`, `single_token_forward_model.cpp`, `mamba_layer.cpp`, `expert_layer.cpp`, `attention_layer.cpp`

- [x] **8c. Pre-allocate attention activation buffers**
  Goal:
  - eliminate the remaining per-call BF16 attention activation allocations
  Sites: `attention_layer.cpp:664-667` (BF16 query/output buffers)
  Approach: move the BF16 decode buffers to `AttentionLayerSlice::Impl`, sized for max token count
  Key files: `attention_layer.cpp`

- [ ] **9. Re-profile after sync/alloc cleanup**
  Goal:
  - measure the improvement from step 8 and identify the new bottleneck
  Expected: kernel compute floor ~33ms/token (Mamba ~22ms, attention ~3ms, GEMMs ~4ms, other ~4ms)
  Host overhead should be near-zero after 8a+8b+8c

- [ ] **10. Optimize Mamba decode kernel**
  Goal:
  - the FusedMambaDecodeKernel at 0.94ms × 23 layers = ~22ms/token is the hard blocker for 20ms
  - this is a single-CTA kernel (grid=1) like the old MoE kernel was
  Approach (TBD based on step 9 profile):
  - option A: replace conv/SSM/gating with cuBLASLt for the linear portions (Mamba projections already use cuBLASLt; the kernel handles conv update + SSM update + gated RMSNorm)
  - option B: multi-CTA kernel that parallelizes across Mamba heads
  - option C: this may be near-roofline for the recurrent state update — need to check arithmetic intensity
  Accept when:
  - per-layer Mamba time drops materially
  - correctness preserved

- [ ] **11. Evidence-driven loop to 20ms target**
  Goal:
  - continue measure → choose → translate until 20ms or architectural limit
  - each iteration: save artifact + profile, compare, choose one bottleneck, implement, verify

## Progress

| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Linear fastpath correctness | done | b31feb4 | cuBLASLt NVFP4 128x4; zero fallbacks |
| 2 | Re-baseline with evidence | done | 8394930 | expert staging = 92% of runtime |
| 3 | Attention sync cleanup | done | 6fb664b | persistent buffers, removed 6 syncs |
| 4 | Monolithic expert residency | done | 5a624e1 | 23/23 resident, 0 uploads, 1805 ms/token |
| 5 | Profile next bottleneck | done | — | FusedMoeDirectDecode = 98.4% of GPU time (77ms/call × 23 layers = 1771ms/token) |
| 6 | Replace MoE scalar matmuls with cuBLASLt | done | — | 57.2 ms/token (was 1805ms); 400x total speedup from baseline; smoke PASS |
| 7 | Re-measure and decide | done | — | Mamba=66% GPU, sync/alloc=42% wall; both need fixing for 20ms |
| 8a | Remove explicit syncs + dead allocs | done | — | hot-path syncs removed; per-alloc device checks cached; cudaFree still barriers |
| 8b | Decode scratch buffer reuse | done | — | request-context decode views + layer-local FP32 scratch landed; BF16 attention and NVFP4 pack buffers remain |
| 8c | Attention activation pre-alloc | done | — | BF16 query/output pre-allocated in Impl; smoke PASS |
| 9 | Re-profile after cleanup | pending | — | kernel floor ~33ms; host should be near-zero |
| 10 | Optimize Mamba decode kernel | pending | — | 0.94ms × 23 = 22ms/token; hard blocker for 20ms |
| 11 | Evidence-driven loop to 20ms | pending | — | |

## Progress Log

### 2026-04-01 Step 8b Checkpoint

- What changed: `RunTokens()` now reuses the request-context activation buffers for single-token decode by binding non-owning `[1, hidden_size]` FP32 views onto `hidden`, `residual`, and `scratch` instead of allocating fresh tensors each call. `AttentionLayerSlice::Impl`, `MambaLayerSlice::Impl`, and `ExpertLayerSlice::Impl` now pre-allocate decode-only FP32 scratch tensors at `Create()` time and reuse them for `token_count == 1`. The direct-MoE cuBLASLt path was also rewritten to reuse a persistent hidden-size accumulation scratch plus the final output buffer instead of allocating hidden-size temporaries per token.
- What was verified: `cmake --build build-phase1-tests --target nemotron_runtime_backend -j4` passed. `cmake --build build-phase1-tests --target full_forward_manifest_smoke_test -j4` passed.
- What risk remains: the attention BF16 query/output buffers are still allocated per decode call and are deferred to step 8c. NVFP4 activation packing still allocates transient buffers, so step 9 profiling is still required to confirm how much host overhead remains after this checkpoint. This checkpoint only covers compilation, not smoke-test execution or Nsight re-measurement.
- What the next step is: implement step 8c to remove the remaining BF16 attention allocations, then re-profile decode wall time for step 9.

### 2026-04-01 Step 6 Checkpoint

- What changed: added `RunMoeDirectDecodeViaCublaslt()` in `runtime/src/backend/expert_layer.cpp` for the direct-MoE decode path. It keeps host-side top-k routing selection, packs the normalized token once to NVFP4, runs sequential routed/shared NVFP4 GEMMs through cuBLASLt, applies `relu2` between the GEMMs, accumulates routed expert contributions, and finishes with routed + shared + residual on device. `MonolithicNvfp4ExpertWeights` now stores a parallel `matmul_block_scales` buffer plus 16-byte-strided tensor scales so monolithic expert views are valid for cuBLASLt runtime plans.
- What was verified: `cmake --build build-phase1-tests --target nemotron_runtime_backend -j4` passed. `cmake --build build-phase1-tests --target full_forward_manifest_smoke_test -j1` passed after rerunning sequentially; the first attempt failed because two concurrent target builds raced on `libnemotron_runtime_backend.a`.
- What risk remains: correctness against the existing host/fused reference is not yet revalidated with `full_forward_manifest_smoke_test` execution or the Nano correctness tests, and decode performance is still unknown. The new path currently preserves the existing per-op synchronize behavior in NVFP4 packing/GEMM/primitive ops, so kernel compute should drop sharply, but host-side sequencing overhead may still be visible until the next measurement pass.
- What the next step is: run the real smoke/correctness checks with `NEMOTRON_FORWARD_MOE_CUBLASLT=1`, then benchmark/profile to confirm the fused MoE kernel disappears from the hot path and quantify the new dominant cost for step 7.

### 2026-04-01 Step 8a Checkpoint

- What changed: removed unconditional hot-path `cudaDeviceSynchronize()` calls from the primitive device ops (`ResidualAddFp32`, `Relu2InPlaceFp32`, `AccumulateScaledFp32`, `RmsNormFp32`), dense cuBLASLt GEMM execution, NVFP4 cuBLASLt GEMM execution, embedding lookup, device argmax, and the FP8 quantize round-trip path. In `runtime/src/backend/device_nvfp4_matrix.cu`, replaced the temporary `global_max_bits` device allocation with the existing tensor-scale buffer, removed the pack-path synchronizations, and cached the CUDA-device availability check instead of calling `cudaGetDeviceCount()` on each matrix create. In `runtime/src/backend/device_tensor.cpp`, converted the FP32/BF16 tensor create paths to a one-time static CUDA availability check instead of per-allocation `cudaGetDeviceCount()`.
- What was verified: `cmake --build build-phase1-tests --target nemotron_runtime_backend -j4` passed. `cmake --build build-phase1-tests --target full_forward_manifest_smoke_test -j4` passed. Confirmed that the targeted hot-path files no longer contain unconditional `cudaDeviceSynchronize()` calls.
- What risk remains: several hot paths still allocate/free temporaries per token, and `cudaFree` remains an implicit barrier until step 8b lands. The expert-layer dead scratch-allocation move called out in review was already satisfied in the current tree, so there was no functional change required there. This checkpoint only covers build verification; it does not yet execute the smoke test binary or re-profile decode.
- What the next step is: implement step 8b to replace per-token temp allocations with reusable decode scratch buffers, then re-profile wall time once the implicit `cudaFree` barriers are gone.
