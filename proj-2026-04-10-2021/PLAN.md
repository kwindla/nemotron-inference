# Plan: Native FP4 MMA for shared expert GEMMs

Project directory: `./proj-2026-04-10-2021`

## Context
The current fused prefill implementation has two MoE boundaries that materialize BF16 or simulate FP4 in ways that the target Blackwell inference paths do not cleanly mirror:

- Shared expert path: `cudaMemcpy(normalized)` → `QuantizeDequantizeRows` → `LaunchContiguousMatVec(shared_up)` → `Relu2` → `QuantizeDequantizeRows` → `LaunchContiguousMatVec(shared_down)`
- Routed path: `gemm1_output_bf16` → `LaunchRoutedBf16Relu2Pack` → packed FC2 input

The shared path is still the immediate Phase 1 target because it runs on every expert layer for every token and is the strongest current suspect for the short-prompt degeneration. But that is a prioritization decision, not proof that the routed BF16 boundary is harmless. The plan must preserve attribution: fix the shared path first, re-measure, then only expand to routed BF16 removal if the residual drift is still material.

The behavioral parity baseline on RTX 5090 is the repo's local vendored `vllm` server configuration, which launches `third_party/vllm` with `--moe-backend flashinfer_cutlass`. That is what our parity scripts have been using as the default anchor. Vendored vLLM's `trtllm_nvfp4_moe` backend is still gated to capability family `100`, and vendored TensorRT-LLM DenseGEMM is still SM100/103 + SwiGLU only, so neither should be treated as the exact structural oracle for this shared ReLU2 path on SM120. Exact oracle status belongs to the local reference/runtime comparisons; vLLM and TRT-LLM are behavioral references.

For implementation structure, the closer precedent is the existing local direct-decode path that already separates routed and shared packed activations and runs dense NVFP4 GEMMs for each side. That path is still limited to a single token, so it is a precedent for pack/plan separation and dense/shared decomposition, not proof that the prefill kernel's multi-row CTA mapping or partial-row behavior is correct. Upstream dense Blackwell examples should be treated the same way: as related dense-kernel references, not as evidence that the shared path should be cloned mechanically from routed P5.

## Rules

- The shared expert kernel must use native FP4 block-scaled MMA (SM120 `mma.nvfp4`), not decode-to-BF16 WMMA.
- Shared activations must be packed via `PackIntoPerRow` with `Nvfp4ScaleLayout::kSwizzled128x4`, not the fake `QuantizeDequantizeRows` round-trip.
- `kSwizzled128x4` must be forced intentionally, not inherited from the generic activation-layout resolver, because small-row inputs would otherwise fall back to `kSwizzled8x4` and block the P5-style SFB path.
- Reuse existing payload buffers unless duplication is required by a measured optimization. Today that means:
  - `normalized_pack` is the default shared FC1 input pack.
  - `fused_prefill_shared_up_pack` is the default shared FC2 input pack.
- If clearer naming is needed, add aliases or rename the parameter fields. Do not allocate new pack buffers just to mirror the routed path.
- In the routed path, weight-side A/SFA TMA descriptors are owned by `MonolithicNvfp4ExpertWeights`, with `FusedNvfp4WeightView` only borrowing pointers. The shared path should follow that ownership principle by extending its own long-lived weight container, which today is `DeviceNvfp4Weight` for `shared_up` and `shared_down`.
- Activation-side B/SFB descriptors must be owned by a long-lived packed-activation container or contiguous CTA-plan cache, not rebuilt ad hoc per launch. The shared path needs a contiguous analogue of the routed `DeviceNvfp4Matrix` + launch-plan descriptor cache.
- The new shared kernel should enter the existing planning, tracing, and benchmarking architecture as its own kernel family/backend kind. It should not be an untracked one-off launcher path.
- The kernel must support arbitrary token counts (1–512+), including partial row tiles.
- Phase 1 must not intentionally change routed execution behavior beyond any necessary cleanup of shared-path plumbing.
- Validation categories:
  - Local high-signal reference: per-layer diagnostic against the local direct-decode/reference path. This is a strong structural comparison, not a mathematically exact oracle.
  - Behavioral parity oracle on RTX 5090: constrained greedy comparison against local vendored `vllm` via `tools/oracle/compare_chat_runtimes.py`.
  - Secondary external reference: local vendored `trtllm`, useful for gross regression detection but not an exact implementation oracle for this path.
  - Claude coherence scoring is a smoke test only, not acceptance proof.
  - Primary performance gate: end-to-end prefill latency.
  - Secondary diagnostic metrics: steady-state hot kernel time for the shared GEMMs and cold descriptor-build/setup overhead. These metrics exist to explain end-to-end results, not replace them.

## Steps

- [x] **1. Reuse existing shared activation packs and make the contract explicit**
  Update `FusedMoePrefillParams` and the prefill workspace wiring so the shared path explicitly consumes the existing packed buffers instead of introducing duplicate payloads. `normalized_pack` should be the default shared FC1 pack. `fused_prefill_shared_up_pack` should be the default shared FC2 pack. If the parameter surface is confusing, add clearly named aliases such as `shared_fc1_pack` and `shared_fc2_pack`, but keep storage ownership in the existing workspace objects.
  Key files: `runtime/include/nemotron/fused_moe_prefill.h`, `runtime/src/backend/expert_layer.cpp`, `runtime/src/backend/request_context.cpp`

- [x] **2. Extend shared weight views to retain cached TMA descriptors**
  Use the routed ownership pattern as the precedent: routed A/SFA descriptors are owned by `MonolithicNvfp4ExpertWeights`, and views only borrow pointers. For the shared path, extend `DeviceNvfp4Weight` so the long-lived shared weight objects own cached P5-style A/SFA descriptors, then expose borrowed pointers through `FusedNvfp4WeightView`. Immutable weight descriptors should be built once during weight preparation, not reconstructed every launch.
  Key files: `runtime/include/nemotron/nvfp4_weight.h`, `runtime/src/backend/nvfp4_weight.cpp`, `runtime/include/nemotron/fused_moe_decode.h`, `runtime/src/backend/expert_layer.cpp`, `runtime/src/backend/monolithic_expert_weights.cu`

- [x] **3. Add a reusable contiguous shared-NVFP4 planning and descriptor-cache layer**
  Introduce a shared-kernel planning surface that fits the existing runtime architecture instead of bypassing it. At minimum this means:
  - a distinct kernel family/backend kind for native contiguous SM120 NVFP4 shared GEMMs
  - a contiguous CTA/profile selection model keyed by token-count bucket, tile shape, and relevant weight dimensions
  - long-lived activation-side B/SFB descriptor caching for `DeviceNvfp4Matrix` packs or a sibling shared-plan cache object
  - traceability and benchmarkability under the same umbrella as the existing GEMM plan/trace infrastructure
  The goal is to make future tile-shape tuning and profile expansion straightforward instead of hard-coding a single launcher path.
  Key files: `runtime/include/nemotron/gemm_catalog.h`, `runtime/include/nemotron/gemm_execution.h`, `runtime/include/nemotron/gemm_planner.h`, `runtime/src/backend/fused_moe_prefill.cu`

- [ ] **4. Write a contiguous FP4 MMA shared GEMM kernel**
  Add `LaunchContiguousFp4MatVec` to replace `LaunchContiguousMatVec` for shared experts. Inputs: packed `DeviceNvfp4Matrix` activations plus `FusedNvfp4WeightView`; output: FP32. The kernel should reuse the local routed P5 FP4 MMA conventions where they actually apply:
  - native SM120 block-scaled MMA atoms
  - weight-side cached A/SFA TMA descriptors
  - activation-side packed B/SFB loads with `kSwizzled128x4`
  - row-tile × output-tile traversal without routing indirection
  Do not assume the routed kernel is a drop-in template. This is a contiguous shared kernel, not "routed MoE minus routing." Do not infer the final tile shape from profile enum names alone: the local routed P5 profile is named `kP5_128x128x64_SwapTrue`, while other nearby code paths use `kMacroTileK = 128`, so the shared kernel's tile shape must be derived explicitly from the chosen CUTE atom/tiling constraints and then justified by measurement on real hardware. The implementation should consume the planning/cache layer from Step 3 rather than inventing a parallel ad hoc dispatch path.
  Key files: `runtime/src/backend/fused_moe_prefill.cu`

- [ ] **5. Replace the shared prefill pipeline with FP4 packing + native FP4 MMA**
  In `RunFusedMoePrefill`, replace the current shared path:
  - Before: `cudaMemcpy(normalized→scratch)` → `QuantizeDequantizeRows` → `LaunchContiguousMatVec(shared_up)` → `Relu2` → `QuantizeDequantizeRows` → `LaunchContiguousMatVec(shared_down)` → `AccumulateSharedOutput`
  - After: `PackIntoPerRow(normalized→shared_fc1_pack)` → `LaunchContiguousFp4MatVec(shared_fc1_pack, shared_up)` → `Relu2` → `PackIntoPerRow(relu2_output→shared_fc2_pack)` → `LaunchContiguousFp4MatVec(shared_fc2_pack, shared_down)` → `AccumulateSharedOutput`
  `shared_up_scratch` remains the FP32 FC1 output / activation workspace unless later profiling justifies further fusion. `routed_gather_scratch` remains the FP32 destination for shared-down output accumulation.
  Key files: `runtime/src/backend/fused_moe_prefill.cu`, `runtime/src/backend/expert_layer.cpp`

- [ ] **6. Validate correctness across multi-row shapes and decide whether routed BF16 removal is required**
  Rebuild and run `fused_moe_prefill_test` and the existing runtime test suite. Add or update tests that exercise multi-row shared-prefill buckets, especially small-row cases that would otherwise default to `kSwizzled8x4`, and partial-row tile boundaries that the single-token direct-decode path does not cover. Run the per-layer diagnostic (`NEMOTRON_DEBUG_COMPARE_PREFILL_VS_LEGACY=1`) on the short-prompt repro and compare the error curve against the pre-fix baseline. Run the prompt-length sweep with native direct MoE prefill enabled. Run constrained greedy parity against local vendored `vllm`, and use local vendored `trtllm` as an additional behavioral reference. Acceptance is: materially reduced early-layer error and elimination or clear mitigation of the observed degeneration. Claude coherence is advisory only.
  Key files: `tools/oracle/compare_chat_runtimes.py`, `tools/oracle/prompt_length_sweep.py`, `testing/backend/fused_moe_prefill_test.cpp`, `testing/backend/expert_layer_fastpath_test.cpp`

- [ ] **7. Benchmark the shared path with end-to-end prefill latency as the primary gate**
  Add or adapt benchmark coverage so the new shared-kernel family can be evaluated with the same artifact/reporting conventions used elsewhere in the repo. The primary acceptance metric is end-to-end prefill latency. Also record secondary diagnostic metrics for:
  - hot shared-kernel time after descriptor/cache warmup
  - cold descriptor-build/setup cost
  - token-count bucket sensitivity
  The benchmark plan should make it possible to compare future tile/profile changes without rewriting measurement code.
  Key files: `tools/benchmark_analysis/compare_gemm_benchmarks.py`, benchmark artifacts/scripts for the shared path

- [ ] **8. Conditional Phase 2: remove the routed BF16 FC1→activation boundary if needed**
  If Phase 1 still leaves material residual drift, expand the work to remove `gemm1_output_bf16` / `LaunchRoutedBf16Relu2Pack` from the routed path and keep the FC1→activation→FC2 boundary in a more upstream-like packed/fused representation. Do not start this step until Phase 1 has been measured, because otherwise we lose attribution.
  Key files: `runtime/src/backend/fused_moe_prefill.cu`, routed MoE packing/epilogue helpers

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Reuse existing shared activation packs | done | 6c3de5b | shared_fc1_pack/shared_fc2_pack aliases wired through params + validation |
| 2 | Extend shared weight TMA descriptor caching | done | 3e9cf1c | DeviceNvfp4Weight owns descriptors; view borrows pointers |
| 3 | Add shared planning/cache layer | done | — | kSm120ContiguousSharedNvfp4 family + cached B/SFB descriptors + token-bucket profiles |
| 4 | Write contiguous FP4 MMA shared GEMM kernel | pending | — | Consume Step 3 dispatch/cache plumbing |
| 5 | Replace shared prefill pipeline | pending | — | |
| 6 | Validate correctness across multi-row shapes | pending | — | Include small-row and partial-tile coverage |
| 7 | Benchmark end-to-end prefill latency | pending | — | Primary gate is end-to-end prefill latency |
| 8 | Conditional routed BF16 cleanup | pending | — | Only if Phase 1 still leaves material drift |
