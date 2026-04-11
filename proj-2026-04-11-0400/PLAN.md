# Plan: Fused in-epilogue FP4 packing for routed FC1 GEMM

Project directory: `./proj-2026-04-11-0400`

## Context
The routed prefill FC1→FC2 path materializes a BF16 intermediate: P5 grouped GEMM → `gemm1_output_bf16` (2 B/elem write) → `LaunchRoutedBf16Relu2Pack` (2 B/elem read, 0.5 B/elem write, plus scale writes). This is the main avoidable FC1→FC2 boundary cost, and it inserts a BF16 truncation boundary between GEMM1 and FP4 packing. The opportunity is still real: keep FP32 accumulators live until the activation/quantization boundary, apply the exact existing activation semantics, compute scales cooperatively, and write the packed routed FC2 input directly.

Important review correction: upstream precedent is weaker than originally stated. Vendored vLLM on RTX 5090 / SM120 supports Nemotron `RELU2_NO_MUL`, but its CUTLASS FP4 path fuses only the SiLU gated case; the ReLU2 path is activation followed by a separate FP4 quantization step. Vendored TRT-LLM has a fused `relu2+nvfp4` transform in some paths, but the relevant SM120 routed MoE backend is CUTLASS, not TRTLLMGen. We should treat external code as a directional reference, not as proof that a routed SM120 P5 epilogue-fusion design is already validated elsewhere.

## Reference implementations

**Current BF16 chain** (`fused_moe_prefill.cu:10704-10751`):
1. `LaunchPlannedPackedInputMatVecBf16` (line 10710) → BF16 to `gemm1_output_bf16`
2. `LaunchRoutedBf16Relu2Pack` (line 10728) → BF16→FP4 with ReLU2 to `gemm1_output` (DeviceNvfp4Matrix)

**P5 kernel epilogue** (`StoreTracedP5CFragmentsTranspose`, line 3831):
- Iterates over 32 per-thread accumulator elements: 4 regs × 4 m_fragments × 2 n_fragments
- Physical coordinate map from `FillPhysicalCoordMapCopyViewLimited` via `partition_C(identity_tensor)`
- Currently: `__float2bfloat16(accum * alpha)` → global store at `output[input_row * output_rows_per_expert + col]`
- The epilogue is called once after the K-tile mainloop completes, when all mainloop shared memory (A/B tiles, scale buffers) is free for reuse

**Block-scale FP4 packing** (`RoutedBf16Relu2PackKernel`, line 2277):
- Per 16-element block: max-abs → `ClampNvfp4Scale(max / 6.0)` → `EncodeFp8Scale` → FP8 block scale
- Quantize: `EncodeFp4(value / block_scale)` → nibble pair packing
- Current grouped BF16 path writes absolute per-block dequant scales to `output_dequant_scales` and does **not** compute per-expert tensor scales in this kernel
- Legacy `PackDeviceRowMajorBf16ToNvfp4PerExpert` uses a different contract: expert tensor scale × relative FP8 block scale, while also optionally materializing absolute dequant scales

**Existing in-epilogue cooperative scale pattern** (P13/P15 kernels, line 6165):
- Already use shared memory for cooperative scale computation in their epilogues
- Establishes the pattern: mainloop smem → `__syncthreads()` → reuse for epilogue cooperative work

**P5 kernel shared memory** (`fused_moe_prefill.cu:5595-5616`):
- `smem_swizzled_a_storage`: FP4 A tile × 2 pipeline stages
- `smem_swizzled_b_storage`: FP4 B tile × 2 pipeline stages
- `a_scale_smem_storage`: scale A × 2 stages
- `b_scale_smem_storage`: scale B
- Total: 64KB+ (SM120 has 228KB per SM)
- These are currently separate `__shared__` declarations, not a union. Reusing this storage for a 128×128 FP32 staging tile is plausible only if we prove the compiler/runtime resource picture and the aliasing strategy. Do **not** assume "free" reuse without a probe.
- Probe result (`proj-2026-04-11-0400/smem_probe.md`):
  - current separate declarations: `81956 B` static shared, `1` CTA/SM
  - naive current + 128×128 FP32 staging tile: compile failure (`148484 B` > `101376 B` block opt-in limit)
  - aliased/union-style storage: `83968 B` static shared, `1` CTA/SM

## Review findings

1. `Relu2` here means squared ReLU, not clamp-to-2. The exact current grouped path semantics are `Relu2(alpha * accum)`, not `alpha * Relu2(accum)`.
2. Tensor-scale normalization uses `6.0f * 448.0f`, not `6.0f * 240.0f`.
3. The first implementation must preserve the existing routed FC2 contract instead of inventing a new private one.
   Decision: the fused path must produce a fully valid `DeviceNvfp4Matrix` and keep `activation_output_scale` as the existing 2D absolute dequant-scale tensor (`[padded_rows, intermediate/16]`) for routed FC2 consumption.
4. `DeviceNvfp4Matrix` validity for this path means populating `packed_data`, `block_scales_data`, `matmul_block_scales_data`, setting `tensor_scale_data = 1.0f`, and filling `per_row_tensor_scales = 1.0f` so generic readers reconstruct the same absolute block scales.
5. CUTLASS/CUTE already imply that SM120 16-element scale-vector ownership is quad/warp-local for the narrow-precision MMA shape we use. A diagnostic dump is still useful, but the default design target should be warp/quad-local scale reduction, not an assumed inter-warp reduction.
6. The shared-memory probe is complete. The result is:
   - naive extra 64KB staging does not fit and is not implementable as an additional shared allocation
   - aliased/union-style storage does fit, but it is not literally free: the probe raised static shared from `81956 B` to `83968 B`
   - therefore, a full-tile FP32 stage is at best a correctness/reference technique; the production target should remain warp/quad-local reduction and direct pack
7. Because the routed BF16 path is the only validated grouped FC1 path today, no BF16 workspace removal is allowed until the direct path is validated and the direct contract is stable.

## Current state

- P5 kernel template: `Nvfp4LaunchPlannedPackedInputGroupedFp4UnifiedSwapTrueKernel` (line 5565) — already templated on `OutputType`, but all routed FC1 instantiations use `__nv_bfloat16`
- Epilogue: `StoreTracedP5CFragmentsTranspose` (line 3831) — `if constexpr (is_same_v<OutputType, __nv_bfloat16>)` branching, 32 elements per thread, coordinate map from CUTE partition_C
- `kTracedP5MFragments = 4`, `kTracedP5NFragments = 2`, `kTracedP5CCopyCoordCapacity = 32` (lines 405-407)
- P5 consumer warps: `kFp4ConsumerWarps = size(TracedP5TiledMma) / 32` (line 5588) = 8 warps = 256 threads
- MMA tile: 128×128×128 (line 81)
- FP4 block width: 16 elements (fused_decode_common.cuh:12)
- FC1 launcher: `LaunchPlannedPackedInputMatVecBf16` (line 10068) — dispatches by `SelectRoutedGemm1Profile`
- Orchestrator: `RunFusedMoePrefill` (line 10551) — `use_grouped_gemm1_output` gates BF16 path
- Output layout (transposed): `output[input_row * output_rows_per_expert + output_col]` where input_row = token index, output_col = weight output dimension

## Rules

### Kernel correctness
- The fused epilogue must preserve the current grouped BF16→pack routed FC2 contract in phase 1.
  - `activation_output_scale` remains the 2D absolute dequant-scale tensor consumed as `input_dq_scales` by routed FC2.
  - `DeviceNvfp4Matrix` must still be structurally valid: `packed_data`, `block_scales_data`, `matmul_block_scales_data`, and `tensor_scale_data` must all be coherent.
- `per_row_tensor_scales` must be initialized to `1.0f` across the matrix so the direct output is also coherent for generic per-row-scale readers.
- Block-scale computation for the phase-1 direct path: `dequant_scale = ClampNvfp4Scale(block_max_abs / 6.0f)`, encoded as FP8 E4M3 with `tensor_scale_data = 1.0f`.
- ReLU2 semantics: squared ReLU applied **after** alpha scaling, i.e. `Relu2(alpha * accum)`.
- The fused kernel outputs directly to `DeviceNvfp4Matrix`, and FC2 must be able to consume the result through the existing `input_dq_scales` path with no FC2 contract changes.
- The kernel must support arbitrary token counts (1–512+), including partial row tiles. Partial-tile masking must be correct in both the smem staging phase and the FP4 output phase.
- Output row mapping: expert routing via `cta_row_starts`, `cta_valid_rows`, `cta_idx_xy_to_batch_idx` from DeviceMoeLaunchPlan — identical to current P5 kernel.

### Resource constraints
- The shared-memory proof is now in hand:
  - adding a separate 128×128 FP32 staging tile is impossible on this kernel shape (`148484 B` static shared; compile failure against the `101376 B` per-block opt-in limit on RTX 5090)
  - an aliased/union-style layout fits (`83968 B` static shared), but it still increases static shared and leaves the kernel at `1` CTA/SM
  - the preferred production design is therefore smaller warp/quad-local staging and direct pack; full-tile FP32 staging is allowed only as a reference/fallback implementation justified by correctness needs
- Reuse existing payload buffers unless duplication is required by a measured optimization. The fused path should write to the existing `gemm1_output` / `fc2_grouped_pack` DeviceNvfp4Matrix, not allocate a parallel output buffer.

### Safety and sequencing
- Do not disable the existing BF16 grouped FC1 path (`use_grouped_gemm1_output`) until the FP4-direct path is fully validated and passing the complete test suite plus prompt-length sweep. The BF16 path must remain as a working fallback throughout development. This is the lesson from proj-2026-04-10-2021 step 8, where removing the BF16 buffer before having a validated replacement disabled the only working grouped FC1 path.
- Phase 1 shared-path changes (proj-2026-04-10-2021 steps 1-5) must not be affected.
- The fused kernel should enter the existing tracing and benchmarking architecture. Use `AppendLinearOpTraceEntry` or equivalent so the new kernel family is visible in `NEMOTRON_FORWARD_LINEAR_TRACE` output and can be profiled with the same artifact conventions as existing kernels.

### Validation categories
- **Local high-signal reference**: per-layer diagnostic via `NEMOTRON_DEBUG_COMPARE_PREFILL_VS_LEGACY=1`, comparing the prefill path against the single-token decode path. This is the most sensitive detector of per-layer drift — it caught the step 8 regression immediately.
- **Direct-output oracle**: compare the new fused output against the old `P5 -> BF16 -> pack` output on the same routed tile, before FC2. Compare `packed_data`, `block_scales_data`, `matmul_block_scales_data`, and `activation_output_scale`.
- **Behavioral parity oracle on RTX 5090**: constrained greedy comparison against local vendored `vllm` via `tools/oracle/compare_chat_runtimes.py`.
- **Secondary external reference**: local vendored `trtllm`, useful for gross regression detection but not an exact implementation oracle for this path.
- **Claude coherence scoring**: smoke test only, not acceptance proof.
- **Primary performance gate**: end-to-end prefill latency.
- **Secondary diagnostic metrics**: hot kernel time for the fused FC1 epilogue vs old P5+BF16Relu2Pack pair, cold descriptor-build/setup overhead, and an explicit TTFT/TPS split (short-prompt first-token latency vs long-prompt throughput). These metrics exist to explain end-to-end results, not replace them.

## Acceptance Matrix

The first validating implementation must cover all of:

- Token counts: `1, 2, 8, 24, 64, 128, 256, 512`
- At least one launch where the final CTA has `valid_rows < 128`
- At least one case where the routed output rows are not an exact multiple of `128`
- At least one case where the final FP4 block in a row is present but the row is padded/inactive
- Both `top_k = 1` and the Nemotron routed default `top_k`
- At least one multi-expert launch where different experts have different active row counts

Acceptance checks:

- direct-output oracle passes
- `NEMOTRON_DEBUG_COMPARE_PREFILL_VS_LEGACY=1` stays within the existing validated envelope or improves
- routed FC2 output matches the legacy grouped path
- one generic `DeviceNvfp4Matrix` reader path reconstructs the same activations from the direct output without using `input_dq_scales`

## Steps

- [x] **1. Lock the exact routed contract and activation math**
  Write down the exact phase-1 direct-path contract in repo-local terms before implementation:
  1. activation is squared ReLU, applied as `Relu2(alpha * accum)`
  2. `activation_output_scale` remains the current absolute dequant-scale tensor (`[padded_rows, intermediate/16]`)
  3. `DeviceNvfp4Matrix` remains fully valid and non-private
  4. `tensor_scale_data` is initialized consistently with the selected contract (phase 1 target: `1.0f`)
  5. `per_row_tensor_scales` are initialized to `1.0f` so generic readers see the same absolute block scales
  Save the decision and rationale in `proj-2026-04-11-0400/contract_notes.md`.
  Key files: `runtime/src/backend/fused_moe_prefill.cu`

- [x] **2. Probe P5 partition_C ownership with CUTLASS/CUTE as the baseline assumption**
  Add a diagnostic device function `DumpP5PartitionCLayout` (gated behind `NEMOTRON_P5_PARTITION_DEBUG`) to the existing P5 kernel that, for blockIdx=(0,0), prints `thread_id, warp_id, physical_idx, m_coord, n_coord` for all 32 per-thread accumulator elements. The goal is not to rediscover the whole mapping from scratch; it is to verify that our concrete local mapping matches the CUTLASS/CUTE expectation that 16-element scale vectors can be reduced within a warp/quad on SM120. Save the analysis as `proj-2026-04-11-0400/partition_c_analysis.md`.
  Key files: `runtime/src/backend/fused_moe_prefill.cu`

- [x] **3. Prove the shared-memory story before touching the epilogue**
  Probe tooling lives in `proj-2026-04-11-0400/p5_smem_probe.cu` and `proj-2026-04-11-0400/run_p5_smem_probe.sh`, with results captured in `proj-2026-04-11-0400/smem_probe.md`.
  Outcome:
  1. current P5 shared declarations fit at `81956 B` static shared
  2. adding a separate 128×128 FP32 staging tile does **not** fit (`148484 B`; compile failure)
  3. an aliased/union-style storage model fits at `83968 B`, but it is not zero-cost and still leaves the kernel at `1` CTA/SM
  This invalidates the original "no increase in per-CTA smem allocation" assumption and shifts the production target toward warp/quad-local reduction.
  Key files: `proj-2026-04-11-0400/p5_smem_probe.cu`, `proj-2026-04-11-0400/run_p5_smem_probe.sh`, `proj-2026-04-11-0400/smem_probe.md`

- [x] **4. Prototype a correctness-first fused pack kernel**
  Based on the partition_C and smem probes, write a standalone test kernel `TestStagedFp4PackKernel` in a new test file. The kernel takes a 128×128 FP32 input tile (simulating accumulators after alpha scaling), applies squared ReLU, computes per-16-element scales, writes packed output plus both scale layouts, and materializes the absolute dequant-scale tensor expected by routed FC2. Validate against a CPU/reference implementation and against `PackDeviceRowMajorBf16ToNvfp4PerExpert` semantics for the selected contract. Also test partial tiles (`valid_rows < 128`).
  Because the smem probe ruled out a naive extra 64KB staging tile, this kernel is a correctness/reference tool. If it uses full-tile staging, it must do so through the proven aliased-storage model rather than through additional shared declarations.
  Add a direct-output compare harness that runs:
  1. old path: `P5 -> BF16 -> LaunchRoutedBf16Relu2Pack`
  2. new path: staged/fused direct pack
  and compares `packed_data`, `block_scales_data`, `matmul_block_scales_data`, and `activation_output_scale` before FC2.
  Key files: new `testing/backend/staged_fp4_pack_test.cu` or `testing/backend/staged_fp4_pack_test.cpp`, `runtime/src/backend/fused_moe_prefill.cu`

- [x] **5. Implement the fused in-epilogue FP4 packing for P5**
  Replace `StoreTracedP5CFragmentsTranspose` with a routed-only fused epilogue variant for the P5 grouped FC1 kernel. Phase-1 requirements:
  1. each consumer thread computes `activated = Relu2(alpha * accum)`
  2. cooperative scale reduction follows the verified SM120 ownership pattern
  3. the epilogue writes `packed_data`, `block_scales_data`, and `matmul_block_scales_data`
  4. the launcher or epilogue materializes the current routed `activation_output_scale` absolute dequant tensor
  5. `tensor_scale_data` is set consistently with the preserved routed contract
  Implementation preference after the smem probe:
  - primary path: warp/quad-local reduction and direct pack
  - optional reference/fallback path: aliased full-tile FP32 staging, only if needed to simplify correctness bring-up
  Gate the new epilogue behind a template parameter or constexpr branch so the BF16 epilogue remains available for fallback.
  Key files: `runtime/src/backend/fused_moe_prefill.cu`

- [x] **6. Add an FP4-direct FC1 launcher variant and wire it into `RunFusedMoePrefill`**
  Add `LaunchPlannedPackedInputMatVecFp4Direct` — a host-side launcher that instantiates the P5 kernel with the fused FP4 epilogue. It must:
  1. clear the full `DeviceNvfp4Matrix` payload coherently
  2. clear or initialize any auxiliary scale buffers required by the preserved routed contract
  3. launch the P5 kernel with the direct epilogue path enabled
  4. keep the existing BF16 grouped FC1 path as the fallback for non-P5 profiles and for validation
  In `RunFusedMoePrefill`, gate the direct path on a new `use_fp4_direct_fc1` condition, but do **not** change the routed FC2 consumer contract in phase 1.
  Key files: `runtime/src/backend/fused_moe_prefill.cu`, `runtime/include/nemotron/fused_moe_prefill.h`, `runtime/src/backend/expert_layer.cpp`

- [x] **7. Validate, then remove BF16 workspace only if the contract is stable**
  First run full validation: `ctest -j1`, `nano_24_token_prefill_regression_test` with `NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1` and `NEMOTRON_DEBUG_COMPARE_PREFILL_VS_LEGACY=1`, prompt-length sweep (`tools/oracle/prompt_length_sweep.py --runtimes native --skip-claude-eval`), and vLLM parity check (`tools/oracle/compare_chat_runtimes.py --runtimes native,vllm`). Compare per-layer diffs against the existing Phase 1 baseline. Only after the direct path is stable should we remove `fused_prefill_gemm1_output_bf16` and any now-dead scale scratch from `MoePrefillWorkspace`.
  Include one non-routed/generic consumer validation:
  - decode the produced `DeviceNvfp4Matrix` through a generic reader path that uses `block_scales_data` / `matmul_block_scales_data` and `per_row_tensor_scales`, without `input_dq_scales`
  - compare the reconstructed activations against the legacy grouped path
  Key files: `runtime/include/nemotron/request_context.h`, `runtime/src/backend/request_context.cpp`, `runtime/src/backend/expert_layer.cpp`

- [ ] **8. Benchmark and compare**
  Run `benchmarks/nano_shared_prefill/run_default_bench.sh` and compare against the Phase 1 baseline artifact using `tools/benchmark_analysis/compare_shared_prefill_benchmarks.py`. Record:
  1. end-to-end prefill latency
  2. short-prompt TTFT
  3. long-prompt TPS
  4. Nsight kernel timings for fused P5 vs old P5 + BF16Relu2Pack
  5. cold descriptor-build/setup overhead
  The fused path should reduce the FC1→FC2 boundary cost without regressing TTFT via setup overhead.
  Key files: `benchmarks/nano_shared_prefill/run_default_bench.sh`, `tools/oracle/compare_chat_runtimes.py`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Lock routed contract + activation math | completed | — | `contract_notes.md` freezes activation, scales, and matrix invariants |
| 2 | Probe partition_C layout | done | 0baaf39 | 16-elem blocks are warp-local (8 lanes × 2 cols); only 16/32 physical slots populated; not quad-local |
| 3 | Prove shared-memory story | completed | — | Current: `81956 B`; naive +64KB stage: compile failure; aliased union: `83968 B` |
| 4 | Prototype staged FP4 pack | done | 6f0105f | GPU vs CPU byte-exact; FP32 vs BF16 legacy byte-exact; 69/72 pass |
| 5 | Implement fused epilogue | done | d7e9bff | Warp-local path, shfl_xor max-abs + shfl_down nibble pack; fixed reg loop OOB; 69/72 |
| 6 | Add FP4-direct launcher + wire it in | done | 802e05d | Wired, env-gated, BF16 default stable 69/72; direct path has correctness blocker (step 7 diagnosis) |
| 7 | Validate, then remove BF16 workspace if safe | done | — | Smem staging fix + zero-fill; BF16 workspace NOT removed (P5 not naturally selected for nano test); 69/72 |
| 8 | Benchmark and compare | pending | — | TTFT + TPS + kernel timing + vLLM parity |
