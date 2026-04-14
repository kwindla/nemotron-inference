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

## Native correction

- The target contract is now the true native direct path: `Relu2(row_alpha * accum)` with no BF16 truncation before activation.
- `NEMOTRON_DEBUG_COMPARE_FP4_DIRECT` remains useful only as a diagnostic against the old BF16-boundary path. It is no longer the primary correctness oracle and zero mismatches are not expected.
- The low-level oracle now exercises the live P5 packer through `RunP5NativeDirectPackOracleForTesting`, which materializes the real `TracedP5AccumProfileLayout` and calls the production `StoreUnifiedRoutedFp4DirectPack` helper.
- That live-layout oracle proved the previous lookup table was wrong. The actual linear `part_c(i)` order is recorded in `proj-2026-04-11-1554/linear_partition_c_mapping.md`, and the production tables were retabled to match it.
- Current verification after the native-path correction:
  - `staged_fp4_pack_test` PASS
  - `nano_24_token_prefill_regression_test` PASS under natural dispatch with native direct FC1 enabled
  - `ctest --test-dir build --output-on-failure -j1` remains `69/72`, with the same three failures: `nvfp4_weight_test`, `expert_layer_oracle_test`, and `expert_layer8_oracle_test`

## Post-native validation

- Sequential validation artifacts are recorded under `proj-2026-04-11-1554/post_native_benchmarks/README.md`.
- This section is now historical only. It captured the state before removing Nano's `23`-token routed-MoE prefill clamp and before making the supported P5 direct FC1 path unconditional.
- For TTFT, forcing `--moe-prefill-window-tokens 4096` materially reduces prefill-driven latency while leaving first-token decode roughly unchanged, which is the current best evidence that the native direct path is helping once the runtime is allowed to use it.

## Post-default-direct validation

- Sequential validation artifacts are recorded under `proj-2026-04-11-1554/post_default_direct_benchmarks/README.md`.
- Nano no longer hard-codes `moe_prefill_window_tokens = 23`; the default runtime now resolves routed-MoE prefill windowing from workspace capacity (`4096` tokens on this RTX 5090 setup).
- The supported packed-input/grouped-output P5 FC1 path no longer depends on `NEMOTRON_ENABLE_FP4_DIRECT_FC1`; the native direct implementation is now unconditional for that regime.
- Saved routed-profile proof: `proj-2026-04-11-1554/post_default_direct_benchmarks/diagnostics/prompt_0384_routed_profile.stderr.txt`
  - Natural long-prompt throughput now uses `routed_gemm1 dispatch_rows=384 active_selection_count=2304 profile=p5_128x128x64_swap_true mode=fp4_direct`
- Shared-prefill throughput is now a direct measurement of the path we care about:
  - `384`: `hot_prefill_mean_ms 783.608886 -> 188.403850` (`-75.96%`), `hot_first_token_mean_ms 825.789506 -> 229.611812` (`-72.19%`)
  - `512`: `hot_prefill_mean_ms 1059.598123 -> 227.061275` (`-78.57%`), `hot_first_token_mean_ms 1115.096786 -> 282.020398` (`-74.71%`)
- Default TTFT now matches the old forced-`4096` run within measurement noise, which is the expected outcome after removing the `23`-token clamp:
  - `cold_prefill_prefix256`: `166.604 -> 167.024 ms` (`+0.25%`)
  - `cold_prefill_prefix1024`: `494.662 -> 494.160 ms` (`-0.10%`)
  - `cold_prefill_prefix4096`: `2186.216 -> 2186.433 ms` (`+0.01%`)
  - `cached_committed_head_prefix4096_tail32 hot-prefix TTFT`: `138.974 -> 139.445 ms` (`+0.34%`)
- Default TTFT versus the old default-window regime shows the real end-to-end win from using the wide routed-MoE prefill window in production:
  - `cold_prefill_prefix256`: `666.521 -> 167.024 ms` (`-74.94%`)
  - `cold_prefill_prefix1024`: `2684.803 -> 494.160 ms` (`-81.59%`)
  - `cold_prefill_prefix4096`: `12179.834 -> 2186.433 ms` (`-82.05%`)
  - `cached_committed_head_prefix256_tail32 hot-prefix TTFT`: `122.426 -> 97.807 ms` (`-20.11%`)
  - `cached_committed_head_prefix1024_tail32 hot-prefix TTFT`: `136.342 -> 106.269 ms` (`-22.06%`)
  - `cached_committed_head_prefix4096_tail32 hot-prefix TTFT`: `183.702 -> 139.445 ms` (`-24.09%`)

## Reference implementations

**Working smem-staged path** (`fused_moe_prefill/nvfp4_bridge.cuh::StoreUnifiedRoutedFp4DirectPack` and siblings):
- Historical correctness reference: a staged oracle that scatters `Relu2(row_alpha * accum)` using `part_c(logical)` linear iteration, where `row_alpha = per_row_tensor_scales[input_row] * weight_tensor_scale` when per-row scales are present
- `PackUnifiedRoutedFp4DirectStaged`: cooperatively packs 16-wide blocks from smem — proven correct
- Key insight: `accum_tensor(logical)` and `part_c(logical)` with the same 1D index are guaranteed consistent

**Step 2 partition_C analysis** (`proj-2026-04-11-0400/partition_c_analysis.md`):
- 256 consumer threads, 16 valid elements per thread
- Warp decomposition: `warp_id = 4*g_n + g_m`, `lane_id = 4*q + r`
- Each 16-element FP4 block is owned by one warp: 8 lanes at stride 4, each with 2 m-values
- The analysis used `FillPhysicalCoordMapCopyViewLimited` (nested-loop iteration), which may differ from `part_c(i)` (linear 1D iteration)

**BF16 epilogue** (`fused_moe_prefill/nvfp4_bridge.cuh::StoreUnifiedRoutedFp4Output`): The P5 branch — useful as a math/activation reference, but it still uses `FillPhysicalCoordMapCopyViewLimited` plus 3D `accum_tensor(reg, m_fragment, n_fragment)` indexing and is not the mapping baseline for the warp-local design

**RoutedBf16Relu2PackKernel** (`fused_moe_prefill/trt_helpers_pre_nano_epilogue.cuh::RoutedBf16Relu2PackKernel`): Reference for FP4 packing math (scales, nibble encoding)

## Current state

- `StageUnifiedRoutedFp4DirectActivated` at line 678: linear `part_c(i)` iteration, scatter to smem
- `PackUnifiedRoutedFp4DirectStaged` at line 743: cooperative smem → FP4 pack
- `StoreUnifiedRoutedFp4DirectPack` at line 843: wrapper calling both
- Kernel-level smem: `fp4_direct_stage_storage[32*128]` at line 6319 (16KB)
- Kernel epilogue site: lines 7724-7768 (zero-fill → stage → sync → pack)
- FC1 boundary compare harness: `NEMOTRON_DEBUG_COMPARE_FP4_DIRECT` at line 2164
- P5 direct selection: native direct is unconditional for the supported packed-input/grouped-output `P5` regime
- Constants: `kTracedP5DirectStageRows=32`, `kTracedP5DirectStageCols=128`, `kTracedP5DirectBlocksPerRow=8` at lines 167-170
- `kTracedP5CCopyCoordCapacity=32` (but only 16 valid per step 2) at line 405
- `fused_decode::kNvfp4BlockWidth=16`, `kNvfp4Fp4MaxFinite=6.0f` in `fused_decode_common.cuh`

## Rules

### Kernel correctness
- The warp-local path must produce output identical to the staged/reference native oracle.
- Iterate accumulator and partition_C with the same 1D linear index: `accum_tensor(i)` and `part_c(i)`.
- The zero-block scale convention remains the current direct-pack contract: `block_scale = 1.0f` when all activated values are zero.
- No BF16 truncation before Relu2: `Relu2(row_alpha * accum)`.

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
- Keep native direct as the only production FC1 implementation for the supported packed-input/grouped-output `P5` regime. Non-`P5`/legacy routed paths remain until the broader row-count launch-matrix rewrite lands.
- Keep the `NEMOTRON_DEBUG_COMPARE_FP4_DIRECT` compare harness for verification.
- Do not modify the BF16 epilogue, the FC2 consumer, or any non-P5 code path.

### Validation categories
- **FC1 boundary compare**: `NEMOTRON_DEBUG_COMPARE_FP4_DIRECT=1` under forced P5 dispatch. Use the explicit forced-P5 command below. Target: profile log confirms `P5`, and the compare output is treated as a numerics-drift diagnostic against the old BF16-boundary path, not as a zero-mismatch gate.
- **Token oracle**: `nano_24_token_prefill_regression_test` under natural profile selection. Must PASS.
- **Per-layer diagnostic**: `NEMOTRON_DEBUG_COMPARE_PREFILL_VS_LEGACY=1` envelope must not regress.
- **Low-level bitwise oracle**: `testing/backend/staged_fp4_pack_test.cpp` must drive the live P5 direct packer through the real `TracedP5AccumProfileLayout`, with deliberately non-uniform per-row tensor scales, and bitwise-compare `packed_data`, `block_scales_data`, `matmul_block_scales_data`, `activation_output_scale`, `tensor_scale`, and `per_row_tensor_scales` against the staged/reference native oracle.
- **Benchmark**: compare `warp-local` against recorded `smem-staged direct` numbers first, and against the BF16 baseline second. Hot prefill is the primary metric; cold setup and TTFT are secondary but must not regress materially. After removing Nano's `23`-token clamp, `384/512` long-prompt runs are now direct measurements of the native `P5` routed-MoE path rather than legacy-only guardrails.
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
  2. Uses the native direct-path math: for each logical element compute `row_alpha = per_row_tensor_scales[input_row] * weight_tensor_scale` when per-row scales are present, otherwise use scalar `alpha`, then compute `Relu2(row_alpha * accum)`
  3. For each of 8 FP4 blocks (2 m-blocks × 4 token groups): max-abs reduction via `__shfl_xor_sync(4/8/16)`
  4. FP4 encode per element, nibble pair assembly via `__shfl_down_sync(4)`
  5. Scale + packed byte writes by designated lanes (q=0 for scales, even-q for packed bytes)
  Add a temporary `P5EpilogueMode::kFp4WarpLocal` value while bringing the code up, but the end state is to make the warp-local implementation the only production direct-FP4 P5 epilogue and delete the smem-staged path rather than keep a permanent selector.
  Key files: `runtime/src/backend/fused_moe_prefill.cu`

- [x] **3. Wire the warp-local path and validate**
  Update `LaunchPlannedPackedInputMatVecFp4Direct` and the kernel instantiation to use the warp-local direct epilogue. Use the new low-level bitwise oracle as the primary bring-up check, then use the explicit forced-P5 runtime command to prove the live P5 code path. Once the low-level oracle and forced-P5 runtime checks pass, remove the 16KB `fp4_direct_stage_storage` allocation and delete the staged direct production path. Run validation:
  1. `ctest -j1` → 69/72
  2. low-level warp-local pack test with non-uniform per-row tensor scales → bitwise match to staged/reference outputs
  3. `nano_24_token_prefill_regression_test` → PASS
  4. forced-P5 code-path validation command → log shows `profile=P5`; treat FC1 boundary compare as a diagnostic against the old BF16 path, not a zero-mismatch gate
  5. Per-layer diagnostic → within envelope
  Key files: `runtime/src/backend/fused_moe_prefill.cu`, `testing/backend/staged_fp4_pack_test.cpp`

- [x] **4. Benchmark at 384/512 tokens and compare**
  The current baseline is already captured at `proj-2026-04-11-1554/baseline_reference/manual_bench/`. After wiring the warp-local path, rerun the long-prompt benchmark at `384/512` and compare first against those recorded numbers and second against the BF16 baseline. Report hot prefill as the primary metric, with cold setup and TTFT as guardrails.
  After removing Nano's `23`-token clamp, these long-prompt runs become direct `P5` routed-MoE measurements and should be used as the main throughput evidence for the new path.
  Key files: `benchmarks/nano_shared_prefill/run_default_bench.sh`, `tools/benchmark_analysis/compare_shared_prefill_benchmarks.py`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Lock probe + invariants + test seam | done | 7cf6f27 | static_assert on part_c size==16; ValidateP5WarpLocalInvariants confirms ownership pattern; test seam passes with non-uniform per-row scales |
| 2 | Build tables + warp-local implementation | done | ded3015 | constexpr tables + StoreUnifiedRoutedFp4WarpLocalPack + kFp4WarpLocal dispatch; compile-clean, not wired yet |
| 3 | Wire and validate | done | 36d071d | Wired kFp4WarpLocal→kFp4Direct, removed staged path + 16KB smem, fixed shuffle mask (subgroup_mask), test seam passes, ctest 69/72, MoE PASS, forced-P5 compare still divergent (pre-existing) |
| 4 | Benchmark at 384/512 | done | e4762f7 | Flat: 384 hot_prefill 783.6→784.2ms (+0.08%), 512 hot_prefill 1059.6→1068.1ms (+0.80%). Integration guardrails only (legacy profile, not P5). |
| 5 | Native-path correction + live-layout oracle | done | worktree | Removed BF16 truncation from live P5 direct pack, added `RunP5NativeDirectPackOracleForTesting`, corrected the lookup tables to the actual `part_c(i)` order, `staged_fp4_pack_test` PASS, `nano_24_token_prefill_regression_test` PASS, `ctest -j1` still 69/72 |
