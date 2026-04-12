# Plan: Native FP4 routed FC1 P1 kernel via TRT-LLM reference, from scratch (v5)

Project directory: `./proj-2026-04-12-1022`

## Project context

From-scratch C++/CUDA inference engine for Nemotron-3 Nano (30B) hybrid Mamba-attention-MoE model on RTX 5090. Target end-state for this project family: single production routed FC1 path, fully native FP4, env-gate removed, all dead kernels deleted.

Current primary performance baseline (single-request warmed uncached prefill, `NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1`):
- `256`: warm uncached prefill `~145 ms`
- `1024`: warm uncached prefill `~474 ms`
- `4096`: warm uncached prefill `~2171 ms`

These numbers reflect the cumulative effect of preceding projects:
- **proj-2026-04-10-2021**: Native FP4 block-scaled MMA for shared expert GEMMs
- **proj-2026-04-10-2330**: Per-row FP4 tensor scales for batched MoE prefill
- **proj-2026-04-11-0400**: Fused in-epilogue FP4 packing for routed FC1
- **proj-2026-04-11-1554**: Warp-local direct FP4 pack epilogue; removed 23-token routed-MoE clamp; enabled 4096-token MoE prefill window
- **proj-2026-04-11-2015**: Routed MoE FC1 profile unification (kP5/kP13 covered, kP1 bucket remains on BF16-WMMA fallback)

kP1 is currently served by the **BF16-WMMA fallback** `Nvfp4LaunchPlannedPackedInputGroupedKernelSwapFalseFp4Direct<128, 64>` (defined at `runtime/src/backend/fused_moe_prefill.cu:6710`, launched from the kP1 dispatch path at line 13137, commit `c36e020`). kP1 is the dominant FC1 profile for 4096-token single-request prefill (per `proj-2026-04-11-2015/diagnostics/prompt_4096_profile_mix.stderr.txt`: kP1 covers 100% of logged FC1 dispatches and 100% of logged active_selection_count at 4096 tokens). Without a native kP1 path, the `NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL` env gate cannot be removed because the BF16-WMMA boundary is still required for one production-reachable case.

Starting point for this plan:
- Current branch: `unified-routed-fp4`
- Current branch head: `7c514ab` (CLAUDE.md probe-then-decide methodology commit)
- Prior plan: plan v4 of this same project directory, archived as `PLAN_v4_abandoned.md`. v1-v4 all failed to converge on a working kP1 kernel over ~1.5 days of debugging.

## Goal

Ship a bitwise-correct native FP4 routed FC1 kernel for the kP1 bucket, replacing the BF16-WMMA fallback, with the kP1 kernel's correctness anchored to an instrumented TRT-LLM reference harness that produces ground-truth output values bit-for-bit.

Correctness first, performance deferred. Once kP1 is correct end-to-end (vLLM parity, prompt sweep, per-layer diagnostic all green), a follow-on plan handles TMA enablement and roofline optimization.

Short version:
- pick TRT-LLM as the authoritative kernel-level reference
- stand up an instrumented TRT-LLM reference harness that captures the highest-fidelity source-level reference surface it exposes
- delete all kP1-specific broken code and tests before writing any new code
- rebuild the kP1 kernel path from scratch with new names, modeled on TRT-LLM's architecture
- validate each layer against that reference surface, bitwise where the contracts match and with a documented tolerance only where TRT-LLM exposes a dequantized or fused surface
- wire to production dispatch only after bitwise correctness and full end-to-end validation
- delete the BF16-WMMA fallback only after the new path is in production

Expanded version:
- TRT-LLM is the primary external kernel-architecture reference because it is vendored at source under `third_party/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/`, pure C++/CUDA, NVIDIA-authored, and patchable for instrumentation. The `.venv-trtllm` FlashInfer wheel is **not** the authoritative kernel oracle for this plan: the local `flashinfer.fused_moe.trtllm_fp4_block_scale_*` entrypoints route through `get_trtllm_moe_sm100_module()`, so they may be useful as smoke checks but not as the source-level SM120 reference harness.
- vLLM remains the **end-to-end** behavioral oracle via `tools/oracle/compare_chat_runtimes.py` (unchanged from prior plans). That is the acceptance gate in step 6, not the kernel-level reference.
- Every previous kP1 debugging attempt assumed the existing broken code was close enough to correct to be worth fixing. After 1.5 days the probe stack is deep and we still don't have a story for why the v3-probe "one-line AccumLayout fix" produced bit-for-bit identical wrong outputs. The accumulated confusion is the bug — not a specific layout mismatch. Delete-before-rewrite breaks the loop.

## Scope

In scope:
- The kP1 bucket of routed MoE FC1 (16-row and 24-row token chunks, dominant at 4096 tokens)
- The correctness-level reference harness against TRT-LLM
- Deletion of broken kP1-specific surfaces (kernels, Traits, tests, bridge helpers, dormant branches)
- Wiring the new kP1 kernel into production dispatch
- End-to-end validation (vLLM parity, prompt sweep, per-layer diagnostic)

Out of scope for this plan (deferred to a follow-on roofline plan):
- TMA enablement for kP1 (the unified kernel's `use_p5_tma_*` constexpr gates are kP5-only today)
- Roofline analysis and perf optimization
- kP1 scale layout optimization (the production kP1 input pack's `kSwizzled128x4` layout must be respected; performance optimization of the swizzle handling is out of scope)
- Any changes to kP5, kP12, kP13, or kP15 paths
- Changes to other routed FC1 buckets
- Decode-path changes

## Current state

### What works and must stay live

- **BF16-WMMA fallback kernel**: `Nvfp4LaunchPlannedPackedInputGroupedKernelSwapFalseFp4Direct<128, 64>` at `runtime/src/backend/fused_moe_prefill.cu:6710` (launched from the kP1 dispatch path at line 13137). This is the only kP1 path that produces correct Nemotron Nano tokens today. Stays live through step 6; deleted in step 7 after the new kernel passes full end-to-end validation.
- **kP1 dispatch entry**: `LaunchPlannedPackedInputMatVecRelu2Pack` at `runtime/src/backend/fused_moe_prefill.cu:13066`. The actual dispatch call at line 13137 routes to the BF16-WMMA fallback during bring-up, then switches to the new kernel in step 6.
- **Unified swap-true kernel**: `Nvfp4LaunchPlannedPackedInputGroupedFp4UnifiedSwapTrueKernel` at line 7863. Used by kP5 and kP13 production dispatches. This plan does **not** instantiate the existing unified kernel for kP1 unless step 4 first adds a dedicated kP1-specialized FP4-direct epilogue and reintroduces a sound `UnifiedRoutedFp4Traits<kP1>` surface. The default implementation path in this plan is a standalone `NanoP1Nvfp4MoeGemmKernel`. The existing unified kernel otherwise stays untouched except for the deletion of the dormant kP1 scale-load branch at line 8744.
- **All kP5, kP12, kP13, kP15 infrastructure**: CollectiveMainloop / Traits / kernels / tests / production dispatch. Untouched by this plan.
- **The vLLM behavioral oracle harness** at `tools/oracle/compare_chat_runtimes.py` and the full ctest suite minus the three known pre-existing failures listed in PLAN_RULES.md.

### Operative dimensions

- `routed_expert_intermediate_size = 1856` (logical) at `runtime/src/api/single_token_forward_model.cpp:781`
- `DefaultRoutedExpertIntermediateSizePadded(1856)` rounds to `kRoutedExpertExecutionAlignment = 128` → **execution width = 1920** (15 tiles of 128)
- Runtime allocates routed-up weights at 1920 per `runtime/src/backend/expert_layer.cpp:2302`, prefill output pack at 1920 per `runtime/src/backend/request_context.cpp:327`
- All new kernel contract code, oracle gates, and roofline math must use **N=1920**, not 1856. Test cases that used 1856 during earlier bring-up are converted to 1920.

### What to delete in step 3, and why (exhaustive list)

All file paths are in `runtime/src/backend/fused_moe_prefill.cu` unless noted. Deletions are sequenced in sub-commits 3a-3e to keep the tree buildable at each step.

**Group A — Bespoke kP1 kernels**

- `Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapFalseK64ScaleSmemP1Direct` (~line 7444)
- `Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapFalseK64ScaleSmemP1` (~line 7235)

**Why delete:** These were the first two native kP1 attempts. Neither has ever produced correct output in any test. Both are built on the bespoke `nvfp4_bridge::Gemm` + `CFragment64` / `AFragment64` / `BFragment64` surface with hand-coded SM80_16x8 fragment layouts that probe-verification has shown disagree with the canonical PTX spec (the `compare_manual_atom_contract` test artifact in `p1_fragment_debug_oracle_test` is a direct symptom of this disagreement). Neither is reachable from production dispatch — kP1 production currently dispatches to the BF16-WMMA fallback. Keeping them in the tree makes future debugging harder because git-grep for `P1` keeps surfacing these as candidate-reference-implementations that they are not.

**Group B — `TracedP1*` Traits and constants**

- `TracedP1MmaTileShape`, `TracedP1ClusterShape`, `TracedP1Epilogue`, `TracedP1StageCountAutoCarveout`, `TracedP1CollectiveMainloop`, `TracedP1TiledMma`, `TracedP1SmemLayoutA/B/SFA/SFB`, `TracedP1SmemCopyAtomA/B/SFA/SFB` (lines 223-264)
- `kTracedP1ScaleSmemCosizeA/B`, `kTracedP1ScaleFragmentCosizeA/B`, `TracedP1AccumProfileLayout`, `kTracedP1AccumProfileCosize` (lines 265-283)
- The stale layout comment block at lines 265-273
- `UnifiedRoutedFp4Traits<UnifiedRoutedFp4Profile::kP1>` specialization (lines 598-631)

**Why delete:** `TracedP1AccumProfileLayout` is built from `partition_fragment_C(TracedP1TiledMma, make_shape(Int<128>, Int<32>))` — a `(128, 32)` shape that does not match the TiledMma's `(128, 128, 64)` CTA tile. Compile-time `ShowInt<N>` probes confirmed the resulting partition contract violation: `tCrB.size<1>=8` but `accum.size<2>=2`. This invariant is the direct source of the "natural P1 oracle produces wrong values" failure, and the v3-probe attempt to change the shape to `(128, 128)` produced bit-for-bit identical wrong outputs — ruling out any one-line fix. The stale comment block at lines 265-273 documents `accum_profile.layout` as `((_2,_2),_2,(_2,_4))` (a shape that would have modes (4, 2, 8), cosize 64, matching the `(128, 128)` partition) which **contradicts** the actual probed cosize of 16, meaning the comment is either lying about the code state or predates a CollectiveBuilder change that silently broke it. Either way, the Traits is not reusable: any correct kP1 implementation needs different AccumLayout shape arguments, different (or same) TiledMma, and a test harness that cannot accidentally inherit the broken Traits by naming collision. Starting from a clean slate with new names is the only way to prevent the same confusion cycle.

**Group C — Dormant kP1 branch in the unified kernel**

- The `if constexpr (Profile == UnifiedRoutedFp4Profile::kP1)` branch at `runtime/src/backend/fused_moe_prefill.cu:8744` that reads `input_matmul_block_scales` with row-major offsets (`source_row * blocks_per_row + block_base + scale_index`)

**Why delete:** Production routed-FC1 input packs are built with `Nvfp4ScaleLayout::kSwizzled128x4` (`runtime/src/backend/request_context.cpp:62` returns `kSwizzled128x4`; see also `request_context.cpp:301` and `expert_layer.cpp:934`). The `Profile == kP1` branch at line 8744 bypasses layout-aware loading and interprets the scale buffer as row-major — an assumption that does not match production. The branch has never been exercised in production because the kP1 dispatch routes to the BF16-WMMA fallback, not the unified swap-true kernel. It is dead code with a wrong assumption. A correct kP1 dispatch through the unified kernel must use the layout-aware `LoadExecutionScaleByte` path (the `else` branch at line 8750) — which means the current kP1 branch has to go regardless of which direction step 4's new kernel takes.

**Group D — kP1 test oracles and their infrastructure**

- `testing/backend/p1_natural_fp4_mma_oracle_test.cpp` + its CMake target
- `testing/backend/p1_fragment_debug_oracle_test.cpp` + its CMake target
- `testing/backend/p1_native_fp4_mma_oracle_test.cpp` + its CMake target
- Device kernels: `P1NaturalFp4MmaOracleKernel`, `P1FragmentDebugOracleKernel`, `P1NativeFp4MmaOracleKernel`
- Host hooks: `RunP1NaturalFp4MmaOracleForTesting`, `RunP1FragmentDebugOracleForTesting`, `RunP1NativeFp4MmaOracleForTesting`
- Trace structs and support: `P1NaturalFp4MmaTrace`, `P1FragmentDebugTrace`, `P1NativeFp4MmaTrace` in `runtime/include/nemotron/fused_moe_prefill.h`, plus all `kP1NaturalFp4MmaTrace*`, `kP1FragmentDebug*`, `kP1NativeFp4MmaTrace*` constants, plus `CopyP1NaturalFp4MmaTrace` / `ResetP1NaturalFp4MmaTrace` / equivalents for the other two oracles, plus the `__managed__` globals (`g_p1_natural_fp4_mma_trace`, `g_p1_fragment_debug_trace`, `g_p1_native_fp4_mma_trace`)

**Why delete:** Each of the three test oracles has produced at least one independently-confirmed test artifact during this bring-up:

- `p1_natural_fp4_mma_oracle_test` uses a flat `accum_tensor(i)` walk paired with `part_c(i)` walk (`fused_moe_prefill.cu:14062`, pre-reset state). Probe-verified: the two flat walks visit the same per-thread storage in different orders because `partition_fragment_C` and `partition_C` produce equivalent-but-not-identically-ordered layouts. Every value this test reports as "actual" is potentially shuffled relative to the coord it is compared against. Even if we fixed the walk, the test depends on the broken `TracedP1AccumProfileLayout` and would still report wrong values for partition-contract reasons.
- `p1_fragment_debug_oracle_test` uses hand-coded SM80_16x8 layouts in `compare_manual_atom_contract` (`build_manual_a_word` / `build_manual_b_word`) and `compare_c_atom_manual_store_bucket` (the `{row_group, row_group+2, row_group+1, row_group+3}` token_rows formula). Both disagree with the canonical PTX spec. The third C-side check, `compare_c_atom_bucket`, stages `kPlannedWmmaTileM = 16` rows of input into smem but `partition_C` walks 32 rows, leaving rows 16-31 reading uninitialized smem. All three C-side failures are test artifacts compounded by the underlying broken TracedP1 Traits.
- `p1_native_fp4_mma_oracle_test` tests the bespoke `LoadFragmentA_RowMajor16x64TracedScale*TiledP1` + `nvfp4_bridge::Gemm` bridge — which is the exact surface we are deleting in Group F. Even if its hand-coded layout tables were right, it would only validate a kernel surface that is going away.

All three oracles compounded false leads during debugging. None has produced a trustworthy kernel-level oracle for kP1. The replacement in steps 2, 4, and 5 is a bitwise comparison against the TRT-LLM reference harness, which is authoritative.

**Group E — Bespoke FP4 MMA bridge helpers**

- `nvfp4_bridge::Gemm` at ~line 5587 (and the `CFragment64` / `AFragment64` / `BFragment64` structs at ~line 478) — **but only if grep confirms no remaining callers in kP13/kP15/kP12 production paths after Groups A-D are deleted**
- `LoadFragmentA_RowMajor16x64TracedScale*TiledP1`, `LoadFragmentB_ColMajor64x8TracedScale*TiledP1`
- `Sm120BlockScaledFp4Mma` PTX wrapper — only if grep confirms no remaining callers
- `kTracedP1ScaleFragmentCosizeA/B` literals at lines 266-267 (callers go away when Groups A and D are deleted)

**Why delete:** The bespoke `Gemm` bridge was built when the CUTLASS `cute::gemm` path did not cleanly support block-scaled FP4 MMA. It wraps the raw PTX `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` instruction with hand-coded A/B/C fragment layouts. The fragment layouts are opaque integer constants (`kTracedP1ScaleFragmentCosizeA = 8`, `kTracedP1ScaleFragmentCosizeB = 32`) whose derivation is not documented. Because the bridge is opaque, probing what it actually does against the canonical PTX spec is difficult. The correct path forward uses CUTLASS `mma_atom.call` or `cute::gemm` on properly-partitioned fragments (the path TRT-LLM uses, and the path kP5/kP13 production already uses). Once Groups A, B, and D are deleted, the bespoke bridge surface may still be used by kP13/kP15 in some dark corners — the grep is required before deletion. If any caller remains, leave those specific helpers intact and document why; delete the rest.

**Group F — Dormant generic-direct-stage path**

- `nvfp4_bridge::StageUnifiedRoutedFp4DirectPrepack<Profile>` at ~line 1118
- `nvfp4_bridge::PackUnifiedRoutedFp4DirectStage<Profile>` at ~line 1156
- The `kUseGenericDirectStage` constexpr branch and its smem allocation inside the unified swap-true kernel at lines 7773-7811
- `testing/backend/p13_generic_direct_stage_oracle_test.cpp` and its CMake target
- `RunP13GenericDirectStageOracleForTesting` host hook and `P13GenericDirectStageOracleKernel` device kernel

**Why delete:** The generic-direct-stage path was an alternative FP4-direct epilogue that used the same flat `accum_tensor(i)` walk pattern the natural P1 oracle uses. It is not exercised by production dispatch. Production uses `StoreUnifiedRoutedFp4DirectPack` for kP5 FP4-direct; the separate flat-walk dense store in `StoreUnifiedRoutedFp4Output` at line 1083-1121 is a non-direct output path and is **not** what Group F deletes. Its test oracle (`p13_generic_direct_stage_oracle_test.cpp`) was written by me in a prior plan iteration to probe this dormant code, and the test was the source of hours of debugging on something that turns out to be dead code. Keeping it around encourages future confusion about "is this the path we should be using?" Deleting it in this step reduces the number of FP4 direct-pack surfaces in the tree from three (kP5 specialized, generic dormant, kP1 dormant-broken) to one (kP5 specialized), which is a cleaner baseline for step 4's new kP1 kernel to model on.

**What to keep untouched (explicit)**

- The kP5 specialized direct packer (`StoreUnifiedRoutedFp4DirectPack` at line 760)
- The positive P5 direct-pack oracle surface: `RunP5NativeDirectPackOracleForTesting` and `testing/backend/staged_fp4_pack_test.cpp`
- The kP5 `StoreTracedP13CFragmentsRowMajor` helper at line 5124 (reference for runtime coord derivation via `FillPhysicalCoordMapCopyViewLimited`)
- The `UnifiedRoutedFp4Traits<kP5>`, `<kP12>`, `<kP13>`, `<kP15>` specializations
- The `UnifiedRoutedFp4Profile::kP1` enum value itself (even without a Traits spec, the enum constant may still be referenced by the dispatch selector in `SelectRoutedGemm1Profile`). Step 3 removes only the Traits specialization; step 4 re-adds it with new underlying types.
- All non-P1 tests in the ctest suite

After step 3, a grep for `P1` in `runtime/src/backend/fused_moe_prefill.cu` should show only: (a) the BF16-WMMA fallback kernel, (b) the kP1 dispatch entry / launch site in `LaunchPlannedPackedInputMatVecRelu2Pack`, (c) the `UnifiedRoutedFp4Profile::kP1` enum constant wherever it is referenced in the profile selector / launch planner, and (d) any explanatory comments.

### Branch

`unified-routed-fp4`. Plan v4 archived at `proj-2026-04-12-1022/PLAN_v4_abandoned.md`. Historical plan iterations (v1, v2, v3) were never committed separately — they exist only in the git history of PLAN.md edits and in the `proj-2026-04-12-1022/` directory.

Latest commits on the branch:
- `7c514ab` Document probe-then-decide methodology in CLAUDE.md
- `45bad97` Trace P1 atom contract mismatch
- `8ccaeaf` Bifurcate P1 fragment scale staging from load path
- `68a1d2f` Add P1 fragment debug oracle
- `640b1d4` Add natural P1 FP4 MMA oracle sentinel
- `113826d` Add P1 native FP4 MMA oracle test
- `c36e020` Fix P1 FP4 direct path via BF16-boundary WMMA kernel

## Design intent

Plan v1-v4 assumed the existing broken kP1 surface was close enough to correct that probe-then-fix iteration would converge. It did not, over ~1.5 days. The root failure mode was not any single wrong hypothesis — it was that each probe's result had to be interpreted against a tree full of stale/broken/entangled reference points, and every "fix" had to thread through the same confusion.

v5 makes three structural bets:

1. **Delete-before-rewrite.** The broken code cannot remain in the tree while new code is being written. Name collisions, muscle-memory references, and git-grep false positives will repeatedly drag the new implementation back toward the old shape. Step 3 is a big aggressive delete pass with sub-commits for safety.

2. **External reference, not self-constructed oracle.** Every test oracle we built for kP1 in v1-v4 turned out to have an independent bug (hand-coded SM80_16x8 layouts, flat walks, under-staging, wrong token_row formulas). Self-constructing a kernel-level oracle is harder than we thought. TRT-LLM is NVIDIA-authored, vendored at source, and provides a known-good implementation of the same contract. Step 2 stands up a reference harness that captures TRT-LLM's exact output tensors for a synthetic problem; steps 5 and 6 validate the new kP1 kernel bitwise against that harness.

3. **Correctness first, performance deferred.** v1-v4 tried to combine correctness and performance in the same plan. v5 ships bitwise correctness via a cooperative-load path in steps 4-7, and explicitly defers all performance work (TMA enablement, roofline optimization, scale layout optimization) to a follow-on plan that starts from a working kernel.

## Rules

### Safety and sequencing (from PLAN_RULES.md)

- Keep the BF16-WMMA fallback kernel live through steps 1-6. It is the only kP1 production path that produces correct Nemotron tokens today. It may not be deleted until step 7, after the new kernel passes full end-to-end validation.
- Reuse existing payload buffers unless duplication is required by a measured optimization.
- New kernels and launcher paths enter the existing planning, tracing, and benchmarking architecture (`AppendLinearOpTraceEntry`, `GemmKernelFamily`, `GemmHeuristicCache`). No untracked one-off launcher paths.
- Kernels must support arbitrary token counts (1-512+), including partial row tiles.

### Validation categories (from PLAN_RULES.md)

- **Per-layer diagnostic** (`NEMOTRON_DEBUG_COMPARE_PREFILL_VS_LEGACY=1`): per-layer drift detector
- **Behavioral parity oracle on RTX 5090**: constrained greedy comparison against local vendored vLLM via `tools/oracle/compare_chat_runtimes.py`. Final acceptance gate.
- **Secondary external reference**: local vendored TRT-LLM. In v5, TRT-LLM is promoted from "gross regression detection only" to the **kernel-level correctness oracle** for the new kP1 kernel.
- **Claude coherence scoring**: smoke test only
- **Primary performance gate**: end-to-end prefill latency (deferred to the follow-on roofline plan)
- **Secondary diagnostic metrics**: hot kernel time and cold descriptor-build/setup overhead (also deferred)

### Resource constraints (from PLAN_RULES.md)

- Per-CTA shared memory: do not increase beyond what the existing kP5 unified kernel uses
- GPU memory: RTX 5090, 32GB, fully consumed by the model. Workspace allocations must be justified

### Test protocol (from PLAN_RULES.md)

Run sequentially (`-j1`). Current build tree has **77 tests** (corrected from the stale `68/71` in earlier plan versions).

**Known pre-existing failures** (do not count as regressions):
- `nvfp4_weight_test` — model weight loading
- `expert_layer_oracle_test` — model weight loading
- `expert_layer8_oracle_test` — model weight loading

After any runtime/backend change:
```bash
cmake --build build-sm120-relwithdebinfo --parallel $(nproc)
ctest --test-dir build-sm120-relwithdebinfo --output-on-failure -j1
```
Baseline expected: **74/77 pass** (77 minus the 3 pre-existing failures).

After step 3 (deletions) completes the ctest total drops by however many tests were deleted (natural, fragment-debug, native, p13-generic). Step 3's expected pass count is recomputed then; the rule is "no new failures beyond pre-existing."

After kernel changes affecting MoE prefill, also run:
```bash
NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1 build-sm120-relwithdebinfo/testing/nano_24_token_prefill_regression_test
```
Must produce exact token-oracle matches for prompt23 and prompt24.

After changes that could affect inference quality, also run:
```bash
NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1 uv run tools/oracle/prompt_length_sweep.py --runtimes native --skip-claude-eval
```
Expected: `ALL CHECKS PASSED`.

### Correctness

- Do not reintroduce BF16 truncation before `Relu²`. The new kP1 kernel must keep the accumulator in FP32 through activation and quantize directly to FP4 — matching the native-direct contract established in proj-2026-04-11-0400.
- The new kP1 kernel must preserve the FC2-facing `DeviceNvfp4Matrix` contract. Any divergence from kP5's output layout must be explicitly justified and revalidated.
- Do not change the canonical token-major `topk_ids` / `topk_weights` routing contract or move shipping-path routing reconstruction back onto the host.

### Plan-specific rules

- **TRT-LLM is the external kernel reference, not an unqualified bitwise oracle for our packed output contract.** Step 2 must first document the highest-fidelity surface TRT-LLM can expose on a source-level SM120 path. Use bitwise comparison only when the compared surface is actually the same. If TRT-LLM only exposes dequantized or fused outputs, use it to validate mainloop and activation numerics, and use a local direct-pack contract reference patterned on `RunP5NativeDirectPackOracleForTesting` for packed-bytes and scale-buffer validation.
- **Delete-before-rewrite.** Step 3 deletes all broken kP1 code BEFORE any new code is written. This is the only way to prevent accidental reuse of confusing types and names.
- **New names for new code.** The new kP1 kernel, Traits, and tests get names prefixed with `Nano` (e.g. `NanoP1Nvfp4MoeGemm*`). This avoids git-grep false positives against any residual references to deleted types, and avoids any chance of template-instantiation collisions during the transition.
- **Bitwise correctness before performance.** Steps 1-7 are correctness-only. Performance work (TMA enablement, roofline, scale layout optimization) is deferred to a follow-on plan.
- **Probe-then-decide.** Per `CLAUDE.md`, compile-time probes first, then synthetic-data oracle tests, then fragment-level dump oracles, then end-to-end. This plan applies probe-then-decide to reading TRT-LLM in step 1 and to the new kernel's sub-commits in step 4.
- **Operative N width is 1920, not 1856.** All kernel contract code, oracle gate criteria, and benchmarks must use 1920. Earlier plan iterations that used 1856 must be updated.
- **Roofline work is deferred.** This plan explicitly does not specify a roofline denominator, numerator, or bandwidth target. The follow-on roofline plan will use the 1792 GB/s RTX 5090 GDDR7 peak from the NVIDIA Blackwell architecture PDF and per-CTA traffic accounting.

## Success criteria

- The TRT-LLM reference harness in step 2 produces reproducible reference artifacts for the highest-fidelity surface TRT-LLM exposes for a minimal synthetic NVFP4 MoE GEMM problem, with the exact observable contract and any unavoidable tolerance documented explicitly.
- Step 3 leaves the tree buildable and the existing ctest suite green (minus the three pre-existing failures and minus deleted P1-specific tests).
- The new kP1 kernel produced in step 4 passes its isolated oracle test bitwise against TRT-LLM on the shared observable surface, and its packed output channels pass bitwise against a local direct-pack contract reference patterned on the existing P5 direct-pack oracle.
- The new kP1 kernel passes the same two-part oracle at a realistic Nano bucket (`h=2688`, `i=1920`, `n_experts=128`, 16-row and 24-row token chunks) in step 5.
- Step 6 end-to-end validation is fully green:
  - `cmake --build` clean
  - `ctest -j1` with no new failures
  - `nano_24_token_prefill_regression_test` with `NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1` produces exact token-oracle matches for prompt23 and prompt24
  - `prompt_length_sweep.py --runtimes native --skip-claude-eval` reports `ALL CHECKS PASSED`
  - `tools/oracle/compare_chat_runtimes.py` against vLLM is bitwise-identical for a defined eval slice
  - Per-layer diagnostic reports no per-layer drift with `NEMOTRON_DEBUG_COMPARE_PREFILL_VS_LEGACY=1`
- Step 7 deletes the BF16-WMMA fallback and step 8 opens a follow-on roofline plan. At that point, the kP1 path has one native-FP4 implementation with no BF16 boundary.

## Steps

- [ ] **1. Read TRT-LLM NVFP4 MoE GEMM end-to-end; produce architecture summary**
  Read the NVFP4 MoE GEMM path in `third_party/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/` from its top-level dispatcher down through the kernel template, mainloop, epilogue, and scale handling. Walk the code — do not skim.

  Specific facts to document in `proj-2026-04-12-1022/trtllm_architecture.md`:
  - **Top-level entry point**: which function is the entry for an NVFP4 grouped GEMM with FP4 direct output
  - **Input layout and scale format**: how A (activations) and B (weights) are expected, what scale format (row-major? swizzled? per-block FP8?), how per-row tensor scales and output scales are threaded
  - **CollectiveBuilder instantiation**: TensorOp, MmaTileShape, ClusterShape, ElementSFCompute, stages — the exact template argument list
  - **TiledMma construction**: built via CollectiveBuilder or hand-assembled from `SM120_MXF4NVF4_SS_m16n8k64_SB` atoms? What is the atom layout shape?
  - **Mainloop structure**: how TRT-LLM iterates K, M, N; how it partitions `partition_fragment_C` (at CTA tile or at a sub-tile); how many MMA atom invocations per CTA tile; whether `cute::gemm` or `mma_atom.call` is used
  - **Epilogue structure**: how the accumulator is consumed, how `Relu²` is applied, how per-block FP8 scale is computed, how FP4 nibble packing is done, how output is laid out (packed bytes + scale blocks)
  - **Output shapes**: packed FP4 tensor shape, per-block scale tensor shape, activation-output-scale tensor shape
  - **Existing TRT-LLM tests**: which tests in TRT-LLM's own test suite exercise this NVFP4 MoE GEMM path; we can use them as the first reference for valid inputs/outputs
  - **Key file paths and line numbers** for each of the above, so step 2 can find them quickly

  No code changes in this step. Output is a Markdown document that completely characterizes the TRT-LLM NVFP4 MoE kernel contract.

  Gate: `trtllm_architecture.md` exists and is detailed enough that step 4 can implement against it without re-reading TRT-LLM.

  Key files: `third_party/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/` (read-only), `proj-2026-04-12-1022/trtllm_architecture.md` (new).

- [ ] **2. Stand up an instrumented TRT-LLM reference harness capturing two reference surfaces**
  Build a standalone test harness that runs TRT-LLM's SM120 NVFP4 MoE GEMM on a synthetic problem and captures **two** concrete reference surfaces identified by the step 1 architecture read. These become the external oracles for steps 4-6.

  **Surfaces to capture** (both required):

  1. **BF16 intermediate output of `MoeGemmRunner::moeGemm(...)`**, captured immediately after the kernel returns and **before** `doActivationKernel` at `third_party/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:2063`. This is the bitwise primary kernel-level oracle for step 4b's mainloop. TRT-LLM's MoE GEMM kernel's own epilogue is `LinearCombination<ElementD, float, void, float>` (per-expert `alpha` only — no activation, no quantization, see step 1 architecture doc §6), so the BF16 intermediate is the highest-fidelity surface TRT-LLM exposes for the `A*B` core.
  2. **TRT-LLM's `maybePrintSm120P1CompileProbe` dump** at `third_party/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl:309`, triggered via the `TLLM_FUSED_MOE_PRINT_COMPILE_PROBE_P1=1` env var. This dumps TRT-LLM's own `SmemLayoutAtomSFA`, `SmemLayoutSFA`, `SmemCopyAtomSFA`, `LayoutSFA_TV`, and `tCrC_profile` values for the SM120 P1 CollectiveMainloop instantiation (SwapAB=false, `CTA_M=128, CTA_N=128, CTA_K=64 FP4 elements`, cluster `1×1×1`). This is a no-kernel-run Traits-level oracle for step 4a: we compare the step 4a `ShowInt<N>` probe output against this captured text bit-for-bit.

  We do **not** try to capture a packed-FP4 reference surface from TRT-LLM. TRT-LLM does not have one at the GEMM-kernel boundary — its FP4 pack is in a separate post-GEMM `doActivationKernel`. Validation of our fused direct-pack epilogue (step 4c) anchors against a **local** direct-pack contract reference patterned on `RunP5NativeDirectPackOracleForTesting` at `runtime/src/backend/fused_moe_prefill.cu:13515` and `testing/backend/staged_fp4_pack_test.cpp:1001-1050`, not against TRT-LLM.

  **Build-path investigation first** — surface 2 (compile probe) is cheap; surface 1 (BF16 reference data) is where the build-path choice matters. Do **not** use the `.venv-trtllm` FlashInfer wheel: its `trtllm_fp4_block_scale_moe` routes through `get_trtllm_moe_sm100_module()` and is Hopper-only. Options for surface 1 from cheapest to most expensive:

  - **(a)** Include the TRT-LLM headers directly in a small C++/CUDA test binary built under our existing `testing/` CMake. `#include "third_party/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_gemm_kernels.h"` and explicitly instantiate `MoeGemmRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16>`. Compile the specific instantiations we need with nvcc against our existing CUTLASS headers; let our CMake handle the build. No TRT-LLM-side build infrastructure. Cheapest if TRT-LLM's headers compile cleanly as-is against our CUTLASS version.
  - **(b)** Build the vendored `third_party/TensorRT-LLM/` via its own CMake and link against the produced library from our test binary. More robust but requires understanding TRT-LLM's build system.
  - **(c)** Write a Python script using `torch` that does FP32 GEMM + per-block FP4 quantize, use it as the BF16 reference via a BF16-rounding step, and drop surface 1 as a TRT-LLM-captured artifact. Only if both (a) and (b) fail. Documents what we could NOT verify.

  Pick the cheapest viable option for surface 1. Surface 2 (compile probe) is independent and can be captured via whichever build path exposes `moe_gemm_tma_ws_launcher.inl` instantiation — normally option (a).

  **Surface 2 capture detail.** The compile probe runs at kernel-launcher instantiation time, not at kernel runtime. Any build that instantiates `tma_warp_specialized_generic_moe_gemm_kernelLauncher<cutlass::arch::Sm120, __nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16, ..., CTA_Shape=(128,128,64_bytes), Cluster=(1,1,1), ..., SwapAB=false>` with the env var set will emit the P1 layout text to stderr or the diagnostic stream. The env var controls which profile prints (`P1`, `P5`, `P15`). Capture the P1 output verbatim to `proj-2026-04-12-1022/trtllm_reference/golden/sm120_p1_compile_probe.txt`.

  **Surface 1 harness scope**:
  - Minimal synthetic problem: 1 expert, token count 16, small hidden dim 128, small intermediate dim 128, FP32 → NVFP4 quantize via TRT-LLM's own quantizer (so the FP4 input bits and per-block FP8 scales match what `MoeGemmRunner::moeGemm` expects)
  - Run `moeGemm(...)` with `OutputType=__nv_bfloat16`, bias=null, per-expert alpha=1.0
  - Copy the BF16 output to host, save to `proj-2026-04-12-1022/trtllm_reference/golden/sm120_p1_bf16_reference.bin` as raw little-endian BF16 bytes plus a JSON sidecar describing dtype/shape/(expert, token, intermediate) axis order
  - Also save the exact input tensors we fed in: `input_fp32.bin`, `input_fp4_packed.bin`, `input_fp4_block_scales.bin`, `weight_fp32.bin`, `weight_fp4_packed.bin`, `weight_fp4_block_scales.bin`, `per_row_alpha.bin`, plus a `problem.json` describing dims and seeds
  - **Host-side cross-check**: compute the same GEMM in fp32 from the original fp32 tensors (before quantization); the fp32 result should agree with the TRT-LLM BF16 output within the documented BF16 rounding envelope. Record the observed max_abs_diff and ulp-diff distribution in `NOTES.md`.
  - Scale up to a realistic Nano bucket in step 5, not here.

  **Output of this step**:
  - `proj-2026-04-12-1022/trtllm_reference/` directory with the harness source + build hookup + README for running it
  - `proj-2026-04-12-1022/trtllm_reference/golden/sm120_p1_bf16_reference.bin` (surface 1, kernel-runtime capture) + the input tensor binaries + `problem.json`
  - `proj-2026-04-12-1022/trtllm_reference/golden/sm120_p1_compile_probe.txt` (surface 2, compile-time capture)
  - `proj-2026-04-12-1022/trtllm_reference/NOTES.md` documenting: which build-path option was used for surface 1; any TRT-LLM include-path adjustments required; the observed BF16 vs fp32 tolerance envelope; exact instructions for reproducing both captures; anything unexpected

  Gate: both surface 1 (`sm120_p1_bf16_reference.bin` + input binaries) and surface 2 (`sm120_p1_compile_probe.txt`) exist, the BF16 output agrees with the fp32 host reference within a documented tolerance, and the reproducible capture instructions in `NOTES.md` work on a fresh checkout.

  Key files: `proj-2026-04-12-1022/trtllm_reference/` (new), plus whatever build/CMake plumbing is needed.

- [ ] **3. Delete all kP1-specific broken code (5 sub-commits 3a-3e)**
  Hard delete pass. See "What to delete in step 3, and why" in the Current State section for the exhaustive list and the per-item justification. Commit order is chosen to keep the tree buildable at every step.

  **3a — Delete the P1 test oracles**
  - `testing/backend/p1_natural_fp4_mma_oracle_test.cpp` + CMake target
  - `testing/backend/p1_fragment_debug_oracle_test.cpp` + CMake target
  - `testing/backend/p1_native_fp4_mma_oracle_test.cpp` + CMake target
  - `testing/backend/p13_generic_direct_stage_oracle_test.cpp` + CMake target (Group F)
  - Run `ctest -j1` — expected: 73 tests remain; 70 pass and the same 3 pre-existing failures remain.

  **3b — Delete the P1 device kernel hooks and trace infrastructure**
  - `P1NaturalFp4MmaOracleKernel`, `P1FragmentDebugOracleKernel`, `P1NativeFp4MmaOracleKernel` in `fused_moe_prefill.cu`
  - `P13GenericDirectStageOracleKernel` in `fused_moe_prefill.cu`
  - Host hooks: `RunP1NaturalFp4MmaOracleForTesting`, `RunP1FragmentDebugOracleForTesting`, `RunP1NativeFp4MmaOracleForTesting`, `RunP13GenericDirectStageOracleForTesting`
  - Trace struct definitions and constants in `runtime/include/nemotron/fused_moe_prefill.h`: `P1NaturalFp4MmaTrace`, `P1FragmentDebugTrace`, `P1NativeFp4MmaTrace`, all `kP1NaturalFp4MmaTrace*`, `kP1FragmentDebug*`, `kP1NativeFp4MmaTrace*` constants
  - `__managed__` trace globals: `g_p1_natural_fp4_mma_trace`, `g_p1_fragment_debug_trace`, `g_p1_native_fp4_mma_trace`, and their `Copy*` / `Reset*` helpers
  - `StageUnifiedRoutedFp4DirectPrepack<Profile>` and `PackUnifiedRoutedFp4DirectStage<Profile>` (Group F)
  - The `kUseGenericDirectStage` constexpr branch and its smem allocation at lines 7773-7811 (Group F)
  - Run `cmake --build` + `ctest -j1` — no new failures.

  **3c — Delete the bespoke kP1 kernels and the dormant kP1 branch in the unified kernel**
  - `Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapFalseK64ScaleSmemP1Direct` (Group A)
  - `Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapFalseK64ScaleSmemP1` (Group A)
  - The `if constexpr (Profile == UnifiedRoutedFp4Profile::kP1)` branch at line 8744 in the unified kernel (Group C)
  - Diagnostic globals tied to the bespoke P1 path (verify via grep before deletion)
  - Run `cmake --build` + `ctest -j1` — no new failures. The BF16-WMMA fallback at line 6710 still dispatches for production kP1.

  **3d — Delete `TracedP1*` Traits and constants**
  - Everything in Group B: `TracedP1MmaTileShape`, `TracedP1ClusterShape`, `TracedP1Epilogue`, `TracedP1StageCountAutoCarveout`, `TracedP1CollectiveMainloop`, `TracedP1TiledMma`, `TracedP1SmemLayout*`, `TracedP1SmemCopyAtom*`, `TracedP1AccumProfileLayout`, `kTracedP1AccumProfileCosize`, `kTracedP1ScaleSmemCosizeA/B`, `kTracedP1ScaleFragmentCosizeA/B`, and the stale comment block at lines 265-273
  - `UnifiedRoutedFp4Traits<UnifiedRoutedFp4Profile::kP1>` specialization (leave the `UnifiedRoutedFp4Profile::kP1` enum constant itself intact — only the Traits spec goes)
  - Note: between step 3d and step 4a, any code that instantiates `UnifiedRoutedFp4Traits<kP1>` will fail to compile. Verify via grep that no production path does so. The BF16-WMMA fallback at line 6710 does not use Traits<kP1>.
  - Run `cmake --build` + `ctest -j1` — no new failures.

  **3e — Delete bespoke FP4 MMA bridge helpers (Group E), conditional on grep**
  - Run `grep -rn 'nvfp4_bridge::Gemm\|CFragment64\|AFragment64\|BFragment64\|LoadFragmentA_RowMajor16x64\|LoadFragmentB_ColMajor64x8\|Sm120BlockScaledFp4Mma' runtime testing benchmarks` and count callers for each symbol.
  - For symbols with no remaining callers after sub-commits 3a-3d: delete.
  - For symbols still used by kP13/kP15/kP12 production paths: leave intact and document in a short comment.
  - Run `cmake --build` + `ctest -j1` — no new failures.

  Source-size target: ~1500-2500 lines deleted from `fused_moe_prefill.cu` across sub-commits 3a-3e.

  Gate: after 3e, `cmake --build` clean, `ctest -j1` passes all remaining tests, BF16-WMMA fallback still dispatches for production kP1 from `LaunchPlannedPackedInputMatVecRelu2Pack`, and a git-grep for `P1` in `runtime/src/backend/fused_moe_prefill.cu` shows only the BF16-WMMA fallback kernel, the dispatch site, the `UnifiedRoutedFp4Profile::kP1` enum constant, and explanatory comments.

  Key files: `runtime/src/backend/fused_moe_prefill.cu`, `runtime/include/nemotron/fused_moe_prefill.h`, `testing/backend/p1_*.cpp` (deleted), `testing/backend/p13_generic_direct_stage_oracle_test.cpp` (deleted), `testing/CMakeLists.txt`.

- [ ] **4. Implement the new kP1 kernel from scratch, modeled on TRT-LLM (3 sub-commits 4a-4c)**
  Write a fresh kernel path in `runtime/src/backend/fused_moe_prefill.cu` that matches TRT-LLM's architecture as documented in step 1.

  **New names** (no accidental reuse of deleted types):
  - `NanoP1Nvfp4MoeGemmCollectiveMainloop`
  - `NanoP1Nvfp4MoeGemmTiledMma`
  - `NanoP1Nvfp4MoeGemmSmemLayoutA` / `SmemLayoutB` / `SmemLayoutSFA` / `SmemLayoutSFB`
  - `NanoP1Nvfp4MoeGemmAccumLayout`
  - `UnifiedRoutedFp4Traits<UnifiedRoutedFp4Profile::kP1>` specialization (same name as the deleted one, but referencing the new `NanoP1*` types) only if shared helper reuse genuinely requires it
  - Kernel template: a new standalone `NanoP1Nvfp4MoeGemmKernel`. Reusing the existing unified swap-true kernel is out of scope for this plan unless a later sub-commit explicitly adds a kP1-specialized FP4-direct epilogue branch and documents that architecture change.

  **Sub-commits**:

  - **4a — Traits + TiledMma + smem layouts**
    Instantiate the CollectiveBuilder with TRT-LLM's exact template argument list. Define all the new `NanoP1*` types and, only if helpful for shared helper reuse, the `UnifiedRoutedFp4Traits<kP1>` specialization that references them. No kernel yet — compile-only. Add compile-time `ShowInt<N>` probes (ifdef-guarded) for the per-thread partition mode sizes of A/B/C and remove them after the first successful build to confirm the layout matches TRT-LLM's documented per-thread mode sizes. Gate: `cmake --build` clean, probe values match TRT-LLM's documented per-thread mode sizes.

  - **4b — Mainloop (FP32 dense accumulator output, bypass the direct-pack epilogue)**
    Implement the new kernel's mainloop: smem A/B staging, `fp4_shift` handling, MMA loop, accumulator write-back into smem as FP32 dense. Bypass the direct-pack epilogue for this sub-commit to keep the MMA core isolated. New test `testing/backend/nano_p1_mainloop_oracle_test.cpp` that runs the mainloop on the step 2 synthetic problem and compares against the highest-fidelity TRT-LLM surface step 2 can actually expose. If step 2 can dump a pre-epilogue accumulator or post-mainloop activation tile, compare bitwise to that. If step 2 only exposes a dequantized final surface, compare mainloop plus a temporary dense activation store against that surface and document why accumulator-level capture is unavailable. Gate: mainloop test passes against the step 2 reference surface with the exact comparison mode documented in step 2.

  - **4c — Epilogue (Relu², per-block max-abs, FP8 scale encode, FP4 nibble pack)**
    Implement the direct-pack epilogue on top of 4b's mainloop. **Anchor the direct-pack contract to the existing positive P5 direct-pack surface**: mirror the structure of `RunP5NativeDirectPackOracleForTesting` / `testing/backend/staged_fp4_pack_test.cpp`, but specialized for the new `NanoP1*` kernel. **Derive the lane→coord mapping at runtime** from `partition_C(make_identity_tensor(...))` + `FillPhysicalCoordMapCopyViewLimited` (pattern from `StoreTracedP13CFragmentsRowMajor` at line 5124). Do NOT hand-code SM80_16x8 warp/lane tables — the deleted v1-v4 P1 oracles all produced test artifacts because of hand-coded tables, and the new implementation must not repeat that mistake. New test `testing/backend/nano_p1_direct_pack_oracle_test.cpp` that runs the full kernel (mainloop + epilogue), compares all 4 packed output channels (packed bytes, block scales, matmul block scales, activation output scales) bitwise against a local host direct-pack reference, and separately dequantizes the result to compare against the TRT-LLM reference surface from step 2 when available. Gate: oracle test passes bitwise on all 4 packed output channels, and the dequantized output agrees with TRT-LLM on the shared observable surface.

  Each sub-commit must build clean and pass its own test. The BF16-WMMA fallback remains the production dispatch through all of step 4.

  Key files: `runtime/src/backend/fused_moe_prefill.cu`, `runtime/include/nemotron/fused_moe_prefill.h`, `testing/backend/nano_p1_*.cpp` (new), `testing/CMakeLists.txt`.

- [ ] **5. Scale the bitwise oracle to a realistic Nano bucket**
  Run the new kP1 kernel and the TRT-LLM reference harness on a production-realistic bucket:
  - `h = 2688` (Nano hidden dimension)
  - `i = 1920` (Nano padded intermediate width — NOT 1856)
  - `n_experts = 128` (Nano expert count)
  - 16-row and 24-row token chunks (representative of kP1's dominant prefill buckets)
  - Synthetic weights and inputs seeded identically across both runners

  Compare using the same two-part oracle as step 4c:
  - packed output channels (packed bytes, block scales, matmul block scales, activation output scales) bitwise against the local host direct-pack reference
  - dequantized output against the TRT-LLM reference surface from step 2 on the shared observable contract

  Gate: all 4 packed output channels match bitwise at both bucket sizes, and the dequantized output agrees with TRT-LLM on the shared observable surface. If TRT-LLM only exposes a tolerance-based surface, document the tolerance and any residual quantitatively (including bounded `max_abs_diff`).

  Extend `proj-2026-04-12-1022/trtllm_reference/` with the larger synthetic problem and the new kernel's output comparison.

  Key files: `proj-2026-04-12-1022/trtllm_reference/` (extend), `testing/backend/nano_p1_direct_pack_oracle_test.cpp` (extend with the larger test case).

- [ ] **6. Wire the new kP1 kernel into production dispatch and run full validation**
  Replace the BF16-WMMA fallback dispatch at `runtime/src/backend/fused_moe_prefill.cu:13137` with a call to the new kernel. The dispatch entry (`LaunchPlannedPackedInputMatVecRelu2Pack` at line 13066) remains the same — only the kernel launch argument changes.

  Verify via `NEMOTRON_ROUTED_PROFILE_DEBUG=1` that kP1 hits the new kernel path in a real forward pass at the 4096-token bucket.

  Run the full validation hierarchy:
  - `cmake --build build-sm120-relwithdebinfo --parallel $(nproc)` clean
  - `ctest --test-dir build-sm120-relwithdebinfo --output-on-failure -j1` — no new failures beyond pre-existing
  - `NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1 build-sm120-relwithdebinfo/testing/nano_24_token_prefill_regression_test` — exact token-oracle matches for prompt23 and prompt24
  - `NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1 uv run tools/oracle/prompt_length_sweep.py --runtimes native --skip-claude-eval` — `ALL CHECKS PASSED`
  - vLLM behavioral parity oracle at `tools/oracle/compare_chat_runtimes.py` against the vendored vLLM — binding correctness gate
  - Per-layer diagnostic with `NEMOTRON_DEBUG_COMPARE_PREFILL_VS_LEGACY=1` — no per-layer drift

  Save all artifact logs under `proj-2026-04-12-1022/post_implementation_validation/`.

  Gate: all validation green. The BF16-WMMA fallback remains in the source code (not yet deleted) for reversion safety until step 7.

  Key files: `runtime/src/backend/fused_moe_prefill.cu:13137` (dispatch site), `proj-2026-04-12-1022/post_implementation_validation/` (new, artifacts).

- [ ] **7. Delete the BF16-WMMA fallback kernel**
  Only after step 6's validation is fully clean. Delete:
  - `Nvfp4LaunchPlannedPackedInputGroupedKernelSwapFalseFp4Direct<128, 64>` at line 6710
  - Any WMMA bridge helpers exclusively used by it (verify via grep first)

  Re-run the full validation hierarchy from step 6. Gate: all green, production still works, no regressions.

  Key files: `runtime/src/backend/fused_moe_prefill.cu`.

- [ ] **8. Document the new kP1 architecture; open the follow-on roofline plan**
  Write `proj-2026-04-12-1022/architecture.md` (final version, post-implementation) documenting:
  - The final kernel construction (new `NanoP1Nvfp4MoeGemm*` types, their CollectiveBuilder arguments, their per-thread partition layout)
  - The exact divergences from TRT-LLM (if any) and the rationale for each
  - The v1-v4 hypotheses that turned out wrong and why, as a caveat list for future maintainers (a short post-mortem)
  - Performance state: cooperative-load path is correct but unoptimized. TMA enablement, scale layout optimization, and roofline analysis are deferred to the follow-on plan.

  Open the follow-on roofline plan via `/plan-init` using the architecture document as input. The follow-on plan handles:
  - Relaxing the `Profile == kP5`-only constexpr gates at lines 8012-8039 to allow `Profile == kP1` for TMA / pipeline paths
  - Resolving the kP1 dispatch's interaction with the production `kSwizzled128x4` input pack format
  - Roofline measurement (denominator: 1792 GB/s RTX 5090 GDDR7 peak; numerator: per-CTA traffic, not per-unique-expert)
  - Benchmark iteration with targeted perf sub-commits

  Gate: `architecture.md` exists, follow-on plan directory exists with its own `PLAN.md`, and this plan's `PLAN.md` is marked complete in the progress table.

  Key files: `proj-2026-04-12-1022/architecture.md` (new), `proj-YYYY-MM-DD-HHMM-p1-roofline/PLAN.md` (new, follow-on).

## Progress

| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Read TRT-LLM NVFP4 MoE GEMM; produce `trtllm_architecture.md` | pending | — | Read-only; no code changes |
| 2 | Stand up instrumented TRT-LLM reference harness | pending | — | Captures BF16 intermediate (kernel runtime) + P1 compile probe (compile time) |
| 3 | Delete all kP1-specific broken code (5 sub-commits 3a-3e) | pending | — | BF16-WMMA fallback stays live |
| 4 | Implement new kP1 kernel from scratch (3 sub-commits 4a-4c) | pending | — | New `NanoP1*` names |
| 5 | Scale bitwise oracle to realistic Nano bucket | pending | — | h=2688, i=1920, n_experts=128 |
| 6 | Wire new kP1 to production dispatch; full validation | pending | — | BF16-WMMA fallback kept as safety net |
| 7 | Delete BF16-WMMA fallback | pending | — | Only after step 6 green |
| 8 | Document architecture; open follow-on roofline plan | pending | — | Performance work deferred |
