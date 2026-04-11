# Plan: Warp-local direct FP4 pack epilogue (approach B)

Project directory: `./proj-2026-04-11-1554`

## Captured baseline

- Summary: `proj-2026-04-11-1554/baseline_reference/README.md`
- Forced-P5 compare reference: `proj-2026-04-11-1554/baseline_reference/forced_p5_compare_relwithdebinfo_forward_debug.log`
  - P5 direct path confirmed active: `routed_gemm1 dispatch_rows=8 active_selection_count=138 profile=p5_128x128x64_swap_true mode=fp4_direct`
  - Current compare summary: `packed_data=4150`, `block_scales_data=8305`, `matmul_block_scales_data=8284`, `activation_output_scale=9154`
- Long-prompt integration guardrail benchmarks:
  - `proj-2026-04-11-1554/baseline_reference/manual_bench/prompt_0384.json`
    - `cold_prefill_ms=885.247876`, `hot_prefill_mean_ms=783.608886`, `cold_first_token_ms=929.095388`, `hot_first_token_mean_ms=825.789506`
  - `proj-2026-04-11-1554/baseline_reference/manual_bench/prompt_0512.json`
    - `cold_prefill_ms=1164.751821`, `hot_prefill_mean_ms=1059.598123`, `cold_first_token_ms=1221.956369`, `hot_first_token_mean_ms=1115.096786`
- Important limitation: the default long-prompt benchmark is not a direct P5-epilogue measurement. `proj-2026-04-11-1554/baseline_reference/staged_direct_bench_0384_forward_debug.log` shows `routed_gemm1 dispatch_rows=23 ... profile=legacy`, so `384/512` are TTFT/TPS guardrails, not direct evidence for the P5 direct-FP4 epilogue.

## Context
The smem-staged FP4 direct epilogue (proj-2026-04-11-0400 step 7) is correct but shows no performance gain over the BF16 boundary path. The warp-local approach eliminates the staged shared-memory tile, the zero-fill, and the CTA barriers by using only register-to-register warp shuffles, but the original implementation (step 5) failed because it used 3D `accum_tensor(reg, m_frag, n_frag)` indexing while the lookup tables were derived from the linear `part_c(i)` iteration order. This plan derives the correct linear mapping via a runtime probe, builds constexpr lookup tables, implements a register-only warp-shuffle epilogue, validates it against a low-level oracle, and then deletes the smem-staged production path instead of leaving a permanent fallback.

## Reference implementations

**Working smem-staged path** (`fused_moe_prefill.cu:678-841`):
- `StageUnifiedRoutedFp4DirectActivated` (line 678): scatters `Relu2(bf16(row_alpha*accum))` to smem using `part_c(logical)` linear iteration, where `row_alpha = per_row_tensor_scales[input_row] * weight_tensor_scale` when per-row scales are present — proven correct
- `PackUnifiedRoutedFp4DirectStaged` (line 743): cooperatively packs 16-wide blocks from smem — proven correct
- Key insight: `accum_tensor(logical)` and `part_c(logical)` with the same 1D index are guaranteed consistent

**Step 2 partition_C analysis** (`proj-2026-04-11-0400/partition_c_analysis.md`):
- 256 consumer threads, 16 valid elements per thread
- Warp decomposition: `warp_id = 4*g_n + g_m`, `lane_id = 4*q + r`
- Each 16-element FP4 block is owned by one warp: 8 lanes at stride 4, each with 2 m-values
- The analysis used `FillPhysicalCoordMapCopyViewLimited` (nested-loop iteration), which may differ from `part_c(i)` (linear 1D iteration)

**BF16 epilogue** (`fused_moe_prefill.cu:899-993`): The P5 branch of `StoreUnifiedRoutedFp4Output` — useful as a math/activation reference, but it still uses `FillPhysicalCoordMapCopyViewLimited` plus 3D `accum_tensor(reg, m_fragment, n_fragment)` indexing and is not the mapping baseline for the warp-local design

**RoutedBf16Relu2PackKernel** (`fused_moe_prefill.cu:2288-2424`): Reference for FP4 packing math (scales, nibble encoding)

## Current state

- `StageUnifiedRoutedFp4DirectActivated` at line 678: linear `part_c(i)` iteration, scatter to smem
- `PackUnifiedRoutedFp4DirectStaged` at line 743: cooperative smem → FP4 pack
- `StoreUnifiedRoutedFp4DirectPack` at line 843: wrapper calling both
- Kernel-level smem: `fp4_direct_stage_storage[32*128]` at line 6319 (16KB)
- Kernel epilogue site: lines 7724-7768 (zero-fill → stage → sync → pack)
- FC1 boundary compare harness: `NEMOTRON_DEBUG_COMPARE_FP4_DIRECT` at line 2164
- P5 direct gating: `use_fp4_direct_fc1` at line 11647
- Constants: `kTracedP5DirectStageRows=32`, `kTracedP5DirectStageCols=128`, `kTracedP5DirectBlocksPerRow=8` at lines 167-170
- `kTracedP5CCopyCoordCapacity=32` (but only 16 valid per step 2) at line 405
- `fused_decode::kNvfp4BlockWidth=16`, `kNvfp4Fp4MaxFinite=6.0f` in `fused_decode_common.cuh`

## Rules

### Kernel correctness
- The warp-local path must produce output identical to the smem-staged path (which matches the BF16 reference modulo GEMM non-determinism at near-zero blocks).
- Iterate accumulator and partition_C with the same 1D linear index: `accum_tensor(i)` and `part_c(i)`.
- The zero-block scale convention must match the BF16 reference: `block_scale = 1.0f` when all activated values are zero.
- BF16 truncation before Relu2: `Relu2(__bfloat162float(__float2bfloat16(scaled)))` to match the BF16 path.

### Mapping invariants
- Treat the current P5 ownership pattern as a contract that must be checked explicitly before relying on warp-local shuffle tables.
- Encode compile-time invariants where possible (`cute::size(part_c) == 16`, `fused_decode::kNvfp4BlockWidth == 16`, one CTA thread group per 16-wide output block). For properties that depend on the live `partition_C` mapping, add debug-only verification in the probe and low-level oracle.
- The warp-local design assumes the observed step-2 ownership pattern: one warp per 16-column FP4 block, 8 participating lanes per block, stride-4 lane ownership, and token-row-local `row_alpha`. If the probe does not confirm those assumptions under `part_c(i)` linear iteration, stop and regenerate the tables rather than patch around it.

### Resource constraints
- The warp-local path must NOT use shared memory for staging. The 16KB `fp4_direct_stage_storage` should be removed (or reduced to 1 element when not in use).
- Do not increase per-CTA shared memory allocation beyond what the mainloop uses.
- GPU memory is tight (RTX 5090, 32GB). No new workspace allocations.

### Safety and sequencing
- Keep the smem-staged path only as a development oracle while bringing up the warp-local path. Do not leave a permanent production fallback once the warp-local path is validated.
- Keep the BF16 grouped FC1 path as the default (unchanged).
- Keep the `NEMOTRON_DEBUG_COMPARE_FP4_DIRECT` compare harness for verification.
- Do not modify the BF16 epilogue, the FC2 consumer, or any non-P5 code path.

### Validation categories
- **FC1 boundary compare**: `NEMOTRON_DEBUG_COMPARE_FP4_DIRECT=1` under forced P5 dispatch. Use the explicit forced-P5 command below. Target: profile log confirms `P5`, and compare summary reports zero mismatches (or only near-zero noise from BF16 truncation / GEMM non-determinism).
- **Token oracle**: `nano_24_token_prefill_regression_test` with `NEMOTRON_ENABLE_FP4_DIRECT_FC1=1` (natural profile selection). Must PASS.
- **Per-layer diagnostic**: `NEMOTRON_DEBUG_COMPARE_PREFILL_VS_LEGACY=1` envelope must not regress.
- **Low-level bitwise oracle**: extend `testing/backend/staged_fp4_pack_test.cpp` with a tiny kernel seam that feeds the warp-local packer deterministic accumulator data plus deliberately non-uniform per-row tensor scales, then bitwise-compares `packed_data`, `block_scales_data`, `matmul_block_scales_data`, `activation_output_scale`, `tensor_scale`, and `per_row_tensor_scales` against the staged/reference packer.
- **Benchmark**: compare `warp-local` against recorded `smem-staged direct` numbers first, and against the BF16 baseline second. Hot prefill is the primary metric; cold setup and TTFT are secondary but must not regress materially. For the existing `384/512` long-prompt benchmark, treat results as end-to-end guardrails only because the routed MoE window still dispatches `23` rows and selects `legacy` rather than `P5`.
- **Test baseline**: `ctest -j1` must remain 69/72.

### Test protocol
After any code change:
```
cmake --build build --parallel $(nproc)
```
After runtime/backend code changes:
```
ctest --test-dir build --output-on-failure -j1
```
After kernel changes affecting MoE prefill:
```
NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1 build/testing/nano_24_token_prefill_regression_test
```
Forced-P5 code-path validation command:
```
NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1 \
NEMOTRON_ENABLE_FP4_DIRECT_FC1=1 \
NEMOTRON_DEBUG_USE_SELECTED_TOKEN_TILE_FOR_DISPATCH=1 \
NEMOTRON_ROUTED_PROFILE_DEBUG=1 \
NEMOTRON_DEBUG_COMPARE_FP4_DIRECT=1 \
build/testing/nano_24_token_prefill_regression_test
```
Use this command to prove the warp-local P5 path is actually running. This is a code-path diagnostic, not the natural-dispatch PASS oracle.

## Steps

- [x] **1. Lock the probe, invariants, and test seam before kernel work**
  Add a diagnostic (gated behind `NEMOTRON_P5_LINEAR_PARTITION_DEBUG`) that dumps `logical_idx, m_coord, n_coord` for `i=0..cute::size(part_c)-1` using `part_c(i)` with 1D linear indexing for all 32 lanes of at least one representative warp in each `n` half. Compare against the step-2 dump which used `FillPhysicalCoordMapCopyViewLimited` (nested-loop iteration). If they match, the existing tables from step 2 can be reused directly. If they differ, derive new tables from the linear ordering. Save the mapping as `proj-2026-04-11-1554/linear_partition_c_mapping.md`.
  In the same step, extend `testing/backend/staged_fp4_pack_test.cpp` with the low-level test seam that will call the new warp-local helper directly on deterministic accumulator data. Land the test harness first so the implementation has a bitwise oracle from day 1.
  Also add the invariants we expect to hold, using `static_assert` where possible and debug-only checks otherwise: 16 logical coords, one warp per 16-wide FP4 block, 8 participating lanes, stride-4 lane ownership, and token-row-local `row_alpha`.
  Key files: `runtime/src/backend/fused_moe_prefill.cu`, `testing/backend/staged_fp4_pack_test.cpp`

- [x] **2. Build constexpr lookup tables and implement warp-local direct pack**
  Based on the probe results, build constexpr arrays indexed by linear index `i` (0..15):
  - `kLinearToBlock[16]`: 0 (m < 64) or 1 (m >= 64)
  - `kLinearToTokenGroup[16]`: 0..3 (which of the 4 token rows this element belongs to)
  - `kLinearToMHalf[16]`: 0 (low m within block, position q) or 1 (high m, position 8+q)
  Replace `StageUnifiedRoutedFp4DirectActivated` + `PackUnifiedRoutedFp4DirectStaged` with a new `StoreUnifiedRoutedFp4WarpLocalPack` that:
  1. Iterates `i=0..cute::size(part_c)-1`, reads `accum_tensor(i)`, classifies via lookup tables
  2. Preserves the current direct-path math exactly: for each logical element compute `row_alpha = per_row_tensor_scales[input_row] * weight_tensor_scale` when per-row scales are present, otherwise use scalar `alpha`, then compute `Relu2(bf16(row_alpha * accum))`
  3. For each of 8 FP4 blocks (2 m-blocks × 4 token groups): max-abs reduction via `__shfl_xor_sync(4/8/16)`
  4. FP4 encode per element, nibble pair assembly via `__shfl_down_sync(4)`
  5. Scale + packed byte writes by designated lanes (q=0 for scales, even-q for packed bytes)
  Add a temporary `P5EpilogueMode::kFp4WarpLocal` value while bringing the code up, but the end state is to make the warp-local implementation the only production direct-FP4 P5 epilogue and delete the smem-staged path rather than keep a permanent selector.
  Key files: `runtime/src/backend/fused_moe_prefill.cu`

- [x] **3. Wire the warp-local path and validate**
  Update `LaunchPlannedPackedInputMatVecFp4Direct` and the kernel instantiation to use the warp-local direct epilogue. Use the new low-level bitwise oracle as the primary bring-up check, then use the explicit forced-P5 runtime command to prove the live P5 code path. Once the low-level oracle and forced-P5 runtime checks pass, remove the 16KB `fp4_direct_stage_storage` allocation and delete the staged direct production path. Run validation:
  1. `ctest -j1` → 69/72
  2. low-level warp-local pack test with non-uniform per-row tensor scales → bitwise match to staged/reference outputs
  3. `nano_24_token_prefill_regression_test` with `NEMOTRON_ENABLE_FP4_DIRECT_FC1=1` → PASS
  4. forced-P5 code-path validation command → log shows `profile=P5` and FC1 boundary compare reports zero or near-zero mismatches
  5. Per-layer diagnostic → within envelope
  Key files: `runtime/src/backend/fused_moe_prefill.cu`, `testing/backend/staged_fp4_pack_test.cpp`

- [ ] **4. Benchmark at 384/512 tokens and compare**
  The current baseline is already captured at `proj-2026-04-11-1554/baseline_reference/manual_bench/`. After wiring the warp-local path, rerun the long-prompt benchmark at `384/512` and compare first against those recorded numbers and second against the BF16 baseline. Report hot prefill as the primary metric, with cold setup and TTFT as guardrails.
  Important: these long-prompt runs are integration guardrails, not direct P5-epilogue measurements, because the routed MoE window still dispatches `23` rows and selects `legacy`. Do not treat a flat `384/512` result as proof that the warp-local P5 epilogue failed to help.
  Key files: `benchmarks/nano_shared_prefill/run_default_bench.sh`, `tools/benchmark_analysis/compare_shared_prefill_benchmarks.py`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Lock probe + invariants + test seam | done | 7cf6f27 | static_assert on part_c size==16; ValidateP5WarpLocalInvariants confirms ownership pattern; test seam passes with non-uniform per-row scales |
| 2 | Build tables + warp-local implementation | done | ded3015 | constexpr tables + StoreUnifiedRoutedFp4WarpLocalPack + kFp4WarpLocal dispatch; compile-clean, not wired yet |
| 3 | Wire and validate | done | — | Wired kFp4WarpLocal→kFp4Direct, removed staged path + 16KB smem, fixed shuffle mask (subgroup_mask), test seam passes, ctest 69/72, MoE PASS, forced-P5 compare still divergent (pre-existing) |
| 4 | Benchmark at 384/512 | pending | — | |
