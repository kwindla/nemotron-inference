# Plan: Native FP4 MMA kernel for routed FC1 P1 swap-false direct path (v4)

Project directory: `./proj-2026-04-12-1022`

## Context

The routed FC1 kP1 direct FP4 path is currently served by a BF16-WMMA fallback (`Nvfp4LaunchPlannedPackedInputGroupedKernelSwapFalseFp4Direct<128, 64>`, commit `c36e020`). Plan v1-v3 of this project assumed the kP1 native FP4 MMA core was already correct and only needed a dispatch wiring fix. Probe-then-decide diagnostics in the v3 drafting pass showed this assumption was **wrong**:

1. **kP1's AccumLayout is built at shape `(128, 32)` while the CTA tile is `(128, 128, 64)`.** At `runtime/src/backend/fused_moe_prefill.cu:278-282`, `TracedP1AccumProfileLayout` uses `partition_fragment_C(TiledMma, make_shape(Int<128>, Int<32>))`. kP13 (which is production-verified for similar routed BF16 paths) uses `partition_fragment_C(TiledMma, make_shape(Int<128>, Int<128>))` at `runtime/src/backend/fused_moe_prefill.cu:710-714` — matching its CTA tile exactly. This is the only structural difference between kP1 and kP13's Traits setup.
2. **The CUTE partition contract is violated for kP1.** Compile-time `ShowInt<N>` probes of the natural P1 oracle (`runtime/src/backend/fused_moe_prefill.cu:13852+`) measured:
   - `tCrA` modes: `(V=32, M=2, K=1)` — matches expected per-thread A partition
   - `tCrB` modes: `(V=16, N=8, K=1)` — **N iter = 8**, reflecting the full CTA N tile
   - `accum_tensor` modes: `(V=4, M=2, N=2)` — **N mode = 2**, reflecting the (128, 32) partition
   - `part_c` modes: `(V=4, M=2, N=2)` — matches accum
   - `kAccumProfileCosize = 16`, `kMFragments=2`, `kNFragments=8`, `kRegsPerFragment = 16/(2*8) = 1`
   `cute::gemm`'s static_assert at `third_party/.../cute/algorithm/gemm.hpp:282` requires `size<1>(B) == size<2>(C)`. For kP1: `8 != 2`. The existing hand-coded MMA loop compiles only because it uses `mma_atom.call` directly and bypasses the static_assert.
3. **The comparison probe confirmed kP13 passes the contract**: `tCrA=(32,2,1)`, `tCrB=(16,8,1)`, `accum_tensor=(4,2,8)`. `size<1>(B)=8 == size<2>(C)=8`. ✓ kP13's CTA tile and AccumLayout shape both match at `(128, 128)`.
4. **A naive one-line fix (change kP1's partition_fragment_C shape to `(128, 128)`) is insufficient.** Plan v3-probe tried this: `TracedP1AccumProfileLayout` built at `(128, 128)` compiled cleanly and made accum per-thread size 64, but the oracle test produced **bit-for-bit identical wrong output values** as before the change. This rules out the simplest interpretation. The actual kP1 bug has at least two layers: an AccumLayout shape mismatch AND something else (possibly nested mode coord semantics in partition_fragment_C vs partition_C, possibly A/B smem staging, possibly the `fp4_shift_A/B` ordering) that only reveals itself after the first layer is fixed.
5. **Production kP13 dispatches through the same unified kernel that this plan wants kP1 to use**: `Nvfp4LaunchPlannedPackedInputGroupedFp4UnifiedSwapTrueKernel<kP13, float>` at `runtime/src/backend/fused_moe_prefill.cu:12490`. Production kP13 uses `make_tensor` + `mma_atom.call` with `Traits<kP13>::AccumLayout` — the same CUTE-style pattern the broken kP1 natural oracle uses. The difference that makes kP13 work and kP1 not: kP13's `AccumLayout` is built from shape `(128, 128)` matching its CTA tile, so the partition contract holds. Production kP13 also passes swizzled input scales (`Nvfp4ScaleLayout::kSwizzled128x4` at line 12516) and reads them via layout-aware `LoadExecutionScaleByte` in the `else` branch at line 8750.

The intended outcome is a production kP1 native FP4 path that runs on RTX 5090 SM120 with bitwise-correct output, lets the project delete the BF16 fallback, and produces a documented per-bucket memory bandwidth number vs the RTX 5090 GDDR7 peak. This plan is **probe-driven**: each step up to the fix commits only to a diagnostic question, not to a specific fix.

## Reference implementations

External (vendored under `third_party/`):

- **`third_party/vllm/vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py`** — vLLM's FlashInfer CUTLASS backend, the path vLLM actually dispatches Nemotron Nano to per the backend selection chain at `third_party/vllm/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py:140-146` (TRTLLM rejected at `experts/trtllm_nvfp4_moe.py:113`, CUTEDSL only supports SILU per `experts/flashinfer_cutedsl_moe.py:71`). **Primary** upstream architectural reference for the NVFP4 output contract (packed FP4 + per-block FP8 scales + RELU2_NO_MUL).
- **`third_party/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_gemm_kernels.h:105`** — TRT-LLM's NVFP4 outer-dim scale alignment requirement. Secondary reference only.

Internal (the primary architectural model):

- **`runtime/src/backend/fused_moe_prefill.cu:697-731`** — `UnifiedRoutedFp4Traits<kP13>`. **Reference Traits setup.** Note: `using AccumLayout = decltype(partition_fragment_C(TiledMma, make_shape(Int<128>, Int<128>)).layout())` — the shape matches the full CTA tile, not a sub-tile. Probe-verified to produce `tCrB.size<1>=8 == accum.size<2>=8` (contract holds).
- **`runtime/src/backend/fused_moe_prefill.cu:598-631`** — `UnifiedRoutedFp4Traits<kP1>`. Same CollectiveMainloop and TiledMma construction as kP13 (both use `TracedP5TensorOp`, `(128,128,64)` MmaTileShape, same layouts), except the AccumLayout passes `(128, 32)` instead of `(128, 128)`. That single difference produces the partition contract violation.
- **`runtime/src/backend/fused_moe_prefill.cu:7737`** — `Nvfp4LaunchPlannedPackedInputGroupedFp4UnifiedSwapTrueKernel` template. The kP1 dispatch target (post-fix). Existing kP1 input-scale branch at line 8744 has been in place since the unified kernel was written but has never been exercised by production.
- **`runtime/src/backend/fused_moe_prefill.cu:760`** — `StoreUnifiedRoutedFp4DirectPack`, the kP5-specialized direct packer. **Structural model only** for the new `StoreUnifiedRoutedFp4DirectPackP1`. The kP5 packer hand-codes warp/lane/atom tables specific to kP5's `(M=128, N=128)` AccumLayout shape. Hand-coded SM80_16x8 tables have already produced test artifacts in this bring-up — **the new kP1 packer must derive coord mapping at runtime from `partition_C(make_identity_tensor)` + `FillPhysicalCoordMapCopyViewLimited`**, following the pattern at `runtime/src/backend/fused_moe_prefill.cu:5124`.
- **`runtime/src/backend/fused_moe_prefill.cu:13515`** — `RunP5NativeDirectPackOracleForTesting`, the **primary positive-oracle template** for the new kP1 direct packer's isolated test. Used by `testing/backend/staged_fp4_pack_test.cpp:1028`.
- **`runtime/src/backend/fused_moe_prefill.cu:8346-8476`** — production kP5 MMA loop. Uses `M_tiles = cute::size<1>(tCrA)`, `N_tiles = cute::size<1>(tCrB)` (the same pattern the broken natural P1 oracle uses). Works for kP5 because kP5's `tCrB.size<1>` matches its accum mode-2.
- **`testing/backend/staged_fp4_pack_test.cpp`** — host-only reference and positive oracle pattern. Already exercises the live kP5 direct packer. Template for the new kP1 direct-pack oracle.

## Current state

### Probe-verified per-thread layouts (2026-04-12 session)

**kP1 (broken):**
- `TracedP1MmaTileShape = (128, 128, 64)` at line 223, but `cute::tile_shape(TracedP1TiledMma)` reports `(128, 32, 64)` per the stale comment at line 265-273. This needs re-verification — the stale comment may predate a CollectiveBuilder change.
- `TracedP1AccumProfileLayout` at line 278-282: built from shape `(128, 32)`, cosize=16, per-thread modes `(V=4, M=2, N=2)`.
- `tCrA` per-thread: `(V=32, M=2, K=1)` — 2 M iterations per thread
- `tCrB` per-thread: `(V=16, N=8, K=1)` — **8 N iterations**, mismatches accum's N=2
- `part_c` per-thread from `partition_C(identity(128, 32))`: `(V=4, M=2, N=2)` matching accum

**kP13 (works in production):**
- `TracedP13MmaTileShape = (128, 128, 64)` at line 284 — same as kP1
- `TracedP13CollectiveMainloop` at line 303-316 — structurally identical to kP1's at line 242-255 except for `StageCountAutoCarveout` (different epilogue tag)
- `UnifiedRoutedFp4Traits<kP13>::AccumLayout` at line 710-714: built from shape **`(128, 128)`** not `(128, 32)`. Cosize=64, per-thread modes `(V=4, M=2, N=8)`.
- `tCrA`/`tCrB` probed to be structurally identical to kP1's (same TiledMma)
- Contract holds: `tCrB.size<1>=8 == accum.size<2>=8`

**Probe v3-experiment (one-line fix):** changing kP1's AccumLayout shape to `(128, 128)` compiled but the natural oracle still produced identical bit-for-bit wrong outputs. This conclusively rules out "just change the shape argument" as a sufficient fix.

### Production code path (to be replaced in step 8, deleted in step 10)

- `runtime/src/backend/fused_moe_prefill.cu:6587` — `Nvfp4LaunchPlannedPackedInputGroupedKernelSwapFalseFp4Direct<128, 64>` (BF16-WMMA fallback from `c36e020`)
- `runtime/src/backend/fused_moe_prefill.cu:12947` — `LaunchPlannedPackedInputMatVecRelu2Pack` (kP1 dispatch entry)
- `runtime/src/backend/fused_moe_prefill.cu:12984` — the actual `LaunchProgrammaticKernel(..., Nvfp4LaunchPlannedPackedInputGroupedKernelSwapFalseFp4Direct<128, 64>, ...)` call

### Bespoke broken kernels (to be deleted in step 10)

- `runtime/src/backend/fused_moe_prefill.cu:7444` — `Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapFalseK64ScaleSmemP1Direct`
- `runtime/src/backend/fused_moe_prefill.cu:7235` — `Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapFalseK64ScaleSmemP1`

### Dormant unverified code (to be deleted in step 10)

- `runtime/src/backend/fused_moe_prefill.cu:1118` — `StageUnifiedRoutedFp4DirectPrepack<Profile>`
- `runtime/src/backend/fused_moe_prefill.cu:1156` — `PackUnifiedRoutedFp4DirectStage<Profile>`
- `runtime/src/backend/fused_moe_prefill.cu:7773-7811` — `kUseGenericDirectStage` constexpr branch

### Known-failing tests (kept as live probes through step 4)

- `testing/backend/p1_natural_fp4_mma_oracle_test.cpp:402` — divergence sentinel; fails bitwise at multiple (0, *) coords with specific values (e.g. actual=84 vs expected=0 at (0, 21)). Flipped to positive equality in step 4.
- `testing/backend/p1_fragment_debug_oracle_test.cpp:924` — fails first at `c_atom_post_mma` with `actual=134.625 vs expected=257.031` at lane=0 reg=0 coord=(0,0). Uses bespoke `LoadFragmentA/B` + `nvfp4_bridge::Gemm` bridge (scheduled for deletion in step 10). **Keep as a secondary live probe** — may or may not clear when step 4 fixes the natural oracle (it exercises a different code path). If it still fails after step 4, leave it in place until step 10a deletes the bespoke bridge entirely.

### Operative dimensions

- `routed_expert_intermediate_size = 1856` (logical) at `runtime/src/api/single_token_forward_model.cpp:781`
- `DefaultRoutedExpertIntermediateSizePadded(1856)` rounds to `kRoutedExpertExecutionAlignment = 128` → **execution width = 1920** (15 tiles of 128)
- Runtime allocates routed-up weights at 1920 per `runtime/src/backend/expert_layer.cpp:2302`, prefill output pack at 1920 per `runtime/src/backend/request_context.cpp:327`
- All kernel contract, oracle gates, and roofline math use **N=1920**, not 1856

### Branch

`unified-routed-fp4`. Latest commits:
- `7c514ab` Document probe-then-decide methodology in CLAUDE.md
- `45bad97` Trace P1 atom contract mismatch
- `8ccaeaf` Bifurcate P1 fragment scale staging from load path
- `68a1d2f` Add P1 fragment debug oracle
- `640b1d4` Add natural P1 FP4 MMA oracle sentinel
- `113826d` Add P1 native FP4 MMA oracle test
- `c36e020` Fix P1 FP4 direct path via BF16-boundary WMMA kernel

## Rules

### Safety and sequencing (from PLAN_RULES.md)

- Do not disable a working code path until its replacement is fully validated and passing the complete test suite plus prompt-length sweep. Keep the BF16-WMMA fallback live throughout development.
- Reuse existing payload buffers unless duplication is required by a measured optimization.
- New kernels and launcher paths enter the existing planning, tracing, and benchmarking architecture (`AppendLinearOpTraceEntry`, `GemmKernelFamily`, `GemmHeuristicCache`).
- Kernels must support arbitrary token counts (1-512+), including partial row tiles.

### Validation categories (from PLAN_RULES.md)

- **Per-layer diagnostic** (`NEMOTRON_DEBUG_COMPARE_PREFILL_VS_LEGACY=1`): per-layer drift detector
- **Behavioral parity oracle on RTX 5090**: constrained greedy comparison against local vendored vLLM via `tools/oracle/compare_chat_runtimes.py`. Ground truth.
- **Secondary external reference**: local vendored TRT-LLM, gross regression detection only
- **Claude coherence scoring**: smoke test only
- **Primary performance gate**: end-to-end prefill latency
- **Secondary diagnostic metrics**: hot kernel time and cold descriptor-build/setup overhead

### Resource constraints (from PLAN_RULES.md)

- Do not increase per-CTA shared memory allocation beyond what the existing kernel uses
- RTX 5090, 32GB, fully consumed by the model. Workspace allocations must be justified

### Test protocol (from PLAN_RULES.md)

Run sequentially (`-j1`). Current build tree has **77 tests** (not 68/71 — updated from v3).

**Known pre-existing failures** (do not count as regressions):
- `nvfp4_weight_test`
- `expert_layer_oracle_test`
- `expert_layer8_oracle_test`

After runtime/backend changes:
```bash
cmake --build build-sm120-relwithdebinfo --parallel $(nproc)
ctest --test-dir build-sm120-relwithdebinfo --output-on-failure -j1
```
Expected: **74/77 pass** (77 total minus the 3 pre-existing failures above).

After kernel changes affecting MoE prefill:
```bash
NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1 build-sm120-relwithdebinfo/testing/nano_24_token_prefill_regression_test
```
Must produce exact token-oracle matches for prompt23 and prompt24.

After changes that could affect inference quality:
```bash
NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1 uv run tools/oracle/prompt_length_sweep.py --runtimes native --skip-claude-eval
```
Expected: `ALL CHECKS PASSED`.

### Plan-specific rules

- **Probe-then-decide.** Per `CLAUDE.md`. Compile-time `ShowInt<N>` probes first, then synthetic-data oracle tests, then fragment-level dump oracles, then end-to-end. Never write a kernel fix before the cheapest probe that can answer the open question has been run.
- **Verify the test reference before trusting a bisection result.** When an oracle fails, first check whether the test reference is wrong before assuming the kernel is wrong. This bring-up has already produced multiple test artifacts (SFA hand-coded layout, A_pre_shift hand-coded layout, natural P1 oracle's flat `accum_tensor(i)` walk, the kP1 native oracle's `compare_manual_atom_contract` hand-coded SM80_16x8 layout). Assume it can happen again.
- **Bitwise correctness for kernel-level tests.** All MMA-level correctness gates use bitwise comparison against an FP32 reference computed from the same dequantized inputs. Tolerance gates are reserved for end-to-end integration tests against production data.
- **Keep working sentinels live until their replacement lands.** The fragment-debug oracle and the natural P1 oracle are both divergence sentinels today. They stay in the tree as live probes until a positive kP1 oracle passes bitwise. Only after that do they get deleted.
- **No new TiledMma instantiations unless a probe proves it's necessary.** Plan v3's "option A" (rebuild `TracedP1TiledMma`) was ruled out by the kP13 comparison probe — kP13 uses the SAME CollectiveMainloop/TiledMma as kP1 and works. The fix lives in the Traits or the epilogue, not the TiledMma.
- **Operative N width is 1920, not 1856.** All kernel contract code, oracle gate criteria, and roofline math must use 1920.
- **Roofline denominator: 1792 GB/s** — the NVIDIA Blackwell architecture PDF's RTX 5090 GDDR7 peak. Report achieved bandwidth as `achieved / 1792`. Do not use 1300 GB/s effective as the primary denominator; it's a local observation, not a spec.
- **Roofline numerator counts per-CTA traffic, not per-unique-expert.** Each CTA binds `(expert_index, output_row_base)`; the same expert weights can be touched by multiple CTAs. Count what the grid actually loads, not the deduped set.

## Steps

- [ ] **1. Probe: re-verify kP1 and kP13 tile_shape and layout structure from current source**
  The stale comment at `runtime/src/backend/fused_moe_prefill.cu:265-273` predates an unknown number of CollectiveBuilder changes. Before committing to any fix, re-run compile-time `ShowInt<N>` probes on the live code for:

  - `cute::tile_shape(TracedP1TiledMma{})` — M, N, K values
  - `cute::tile_shape(TracedP13TiledMma{})` — M, N, K values
  - `TracedP1AccumProfileLayout` — rank, all mode sizes, cosize
  - `UnifiedRoutedFp4Traits<kP13>::AccumLayout` — rank, all mode sizes, cosize
  - For each, also probe the underlying flat mode structure. Specifically: **does `cute::size<2>(AccumLayout)` return the nested product or the outer rank?** For a layout like `((_2,_2),_2,(_2,_4))`, the answer matters for the paired `(v, m, n)` indexing semantics.
  - Probe `tCrA` / `tCrB` for both kP1 and kP13 inside their respective test kernels.
  - Probe `part_c = thread_mma.partition_C(identity((128, 128)))` for both profiles. Compare its mode structure to the corresponding AccumLayout.

  Document findings in `proj-2026-04-12-1022/probes.md`. Revert all probe code before committing. Output is a facts-only file that step 2 consults when picking a fix direction.

  Gate: `probes.md` exists and contains definitive mode-size values for both profiles.

  Key files: `runtime/src/backend/fused_moe_prefill.cu` (probe-only, reverted), `proj-2026-04-12-1022/probes.md` (new).

- [ ] **2. Diagnose: why does the one-line fix fail?**
  Plan v3-probe changed kP1's AccumLayout from `(128, 32)` to `(128, 128)` and the test still produced bit-for-bit identical wrong outputs. That's the key diagnostic puzzle. This step's job is to explain **why**.

  **Leading hypothesis** (to test first):

  **(H1) The kP1 TiledMma's native tile is `(128, 32, 64)` — partition at `(128, 128)` creates a 4-way sub-tile replication that the natural oracle's MMA loop doesn't service.** If `cute::tile_shape(TracedP1TiledMma)` returns `(128, 32, 64)` (as suggested by the stale comment at line 267), then `partition_fragment_C(TiledMma, (128, 128))` partitions at 4× the native tile — producing 64 per-thread slots = 4 groups of 16, one per sub-tile. For kP13 production to be correct with this layout, its mainloop must run the TiledMma **4 times per CTA** (once per `(128, 32)` N sub-tile, sliding across the (128, 128) CTA tile). The broken natural P1 oracle only runs the TiledMma **once**, leaving 3 of the 4 sub-tile slot groups at zero and reading the same 16 slots (sub-tile 0) as before the AccumLayout change. That would exactly explain identical bitwise output.

  Test by probing `cute::tile_shape(TracedP1TiledMma{})` (fresh — the stale comment is not authoritative for the current code) AND by inspecting the kP13 production MMA loop structure at `runtime/src/backend/fused_moe_prefill.cu:8346-8476`. Count how many TiledMma invocations kP13 issues per CTA tile. If kP13 loops over N sub-tiles inside the K-block loop, H1 is confirmed.

  **Fallback hypotheses** (test only if H1 is ruled out):

  - **(H2)** The AccumLayout change didn't reach the kernel (trait instantiation caching). Add a compile-time probe for `Traits<kP1>::kAccumProfileCosize` inside the natural oracle kernel — expect 64 after the change.
  - **(H3)** `partition_fragment_C(TiledMma, shape)` and `partition_C(identity(shape))` produce different nested mode structures. Paired `(v, m, n)` indexing may land at different elements even when total size matches. Probe both separately with `ShowInt<N>` on each mode size.
  - **(H4)** Something in A/B smem staging, `fp4_shift_A/B`, or the atom's C register packing is wrong independent of AccumLayout. Highest cost — pursue last.

  Document the confirmed hypothesis in `proj-2026-04-12-1022/probes.md`. Gate: the root cause is identified with a specific probe that reproduces the finding.

  Key files: `runtime/src/backend/fused_moe_prefill.cu` (probes, reverted), `proj-2026-04-12-1022/probes.md` (updated).

- [ ] **3. Pick a fix direction based on step 2's findings and write an architecture note**
  Based on step 2's root cause, commit to one of:

  - **(A) Mirror kP13 fully**: change kP1's Traits AccumLayout shape to `(128, 128)` AND rewrite the natural P1 oracle kernel's mainloop to run multiple TiledMma invocations per CTA (one per `(128, 32)` N sub-tile across the full `(128, 128)` CTA N dimension). This is the expected fix if step 2 confirms H1 — the TiledMma's native tile is `(128, 32, 64)` and production kP13 iterates the sub-tiles inside its K-block loop. The epilogue then walks all 64 per-thread accum slots, derives coords via `partition_C(identity((128, 128)))` + `FillPhysicalCoordMapCopyViewLimited`, and filters writes by `token_row < kTokenRows=32`. Mirrors the kP13 code path at `runtime/src/backend/fused_moe_prefill.cu:1040-1082` for the epilogue and at `runtime/src/backend/fused_moe_prefill.cu:8346-8476` for the mainloop sub-tile iteration.
  - **(B) Narrower AccumLayout fix**: if step 2 rules out H1 and identifies a different root cause (e.g. nested mode coord semantics H3 or mainloop staging H4), apply that narrower fix. This branch is only viable if step 2 produces a probe that explains the v3-experiment's identical-bitwise-output finding without needing the sub-tile iteration change.
  - **(C) Rebuild `TracedP1CollectiveMainloop` with a different `MmaTileShape`** (e.g. `(128, 32, 64)` CTA tile). Highest cost; cascades through smem layouts, swizzle atoms, stage counts. Only choose this if step 2 rules out both (A) and (B). Note: kP1 and kP13 currently share the same CollectiveBuilder template arguments except `StageCountAutoCarveout`, so this is unlikely to be needed.

  Write `proj-2026-04-12-1022/architecture.md` committing to the chosen direction. Include:

  - The specific probe result (from step 2) that motivates the choice
  - Operand orientation (swap-true mirror kP5: A=weight, B=input)
  - Accumulator storage pattern
  - MMA loop pattern
  - Direct-pack epilogue structure (runtime coord derivation via `partition_C(make_identity_tensor)` + `FillPhysicalCoordMapCopyViewLimited`, **not** hand-coded SM80_16x8 tables)
  - TMA descriptor reuse vs new plumbing. Note: `use_p5_tma_a/b/sfa/sfb/pipeline` constexpr gates at lines 8012-8039 are currently `Profile == kP5`-only. For kP1 to benefit from TMA loads in the unified kernel, these gates must be relaxed to also permit `kP1`. Flag this as a prerequisite.
  - kP1 input scale layout asymmetry: the `Profile == kP1` branch at line 8744 reads `input_matmul_block_scales` as row-major, while production routed-FC1 packs are built with `kSwizzled128x4` scale layout (`runtime/src/backend/request_context.cpp:62` returns `kSwizzled128x4`). The kP1 branch has never been exercised in production. Either:
    - (i) Change the kP1 branch at line 8744 to use `LoadExecutionScaleByte` (layout-aware), OR
    - (ii) Have the kP1 dispatch pass a row-major scale side buffer (which `block_scales_data()` provides but may conflict with TMA/swizzled pack pipeline assumptions)
    
    Step 8's dispatch wiring decides; architecture note calls out the asymmetry.
  - Operative N width = 1920 for production dispatch
  - Primary upstream reference: FlashInfer CUTLASS (vLLM's actual Nano path)

  Send the note to Codex for review (`/cx-delegate --background --fresh`). **Codex review is a hard gate** before step 4 starts — v1 and v2 of this plan both required rewrites after Codex review.

  Key files: `proj-2026-04-12-1022/architecture.md` (new).

- [ ] **4. Apply the chosen fix; verify natural P1 oracle passes bitwise**
  Apply the specific fix decided in step 3. Convert the natural P1 oracle's assertion from `expect_divergence` to `expect`:
  ```cpp
  passed = expect(equal, "natural traced P1 MMA oracle must match the exact FP32 reference bitwise");
  ```
  Also delete the now-misleading sentinel comment.

  Gate: `build-sm120-relwithdebinfo/testing/p1_natural_fp4_mma_oracle_test` passes bitwise. If the fix also makes the fragment-debug oracle's `c_atom_post_mma` bucket pass, note that in the commit message (it confirms the root cause is shared between the two oracles). If the fragment-debug still fails, it uses a different code path (bespoke `nvfp4_bridge::Gemm` bridge) and is a separate issue — keep it as a live probe.

  Run `ctest -j1`: expected 74/77 with no new failures beyond the three pre-existing ones.

  Key files: `runtime/src/backend/fused_moe_prefill.cu`, `testing/backend/p1_natural_fp4_mma_oracle_test.cpp`.

- [ ] **5. Add `StoreUnifiedRoutedFp4DirectPackP1` direct packer + isolated oracle**
  New device function in `runtime/src/backend/fused_moe_prefill.cu`, sibling to `StoreUnifiedRoutedFp4DirectPack` at line 760.

  **Body structure:**

  - Reconstruct the per-thread view: `auto thread_mma = TracedP1TiledMma{}.get_thread_slice(thread_idx); auto accum_tensor = make_tensor(reinterpret_cast<CRegister*>(accum_storage), Traits<kP1>::AccumLayout{}); auto coord_tensor = thread_mma.partition_C(make_identity_tensor(shape_matching_accum));`
  - Walk via whatever indexing was proven correct in step 4 (paired `(v, m, n)` OR flat `(i)` — step 4's fix picks)
  - **Derive the lane→coord mapping at runtime** from `FillPhysicalCoordMapCopyViewLimited(coord_tensor, ..., row_coords, col_coords)`. Do NOT hand-code SM80_16x8 warp/lane tables. Reason: plan v3 identified three test artifacts in this bring-up caused by hand-coded SM80_16x8 layouts disagreeing with the canonical PTX spec.
  - Per-block reduction: compute per-block max-abs via lane shuffles, derive the lane mask at runtime from the coord map (not hardcoded)
  - FP8 scale encode + FP4 nibble pack: `block_scale = ClampNvfp4Scale(block_max_abs / kNvfp4Fp4MaxFinite)`, `EncodeFp8Scale`, `EncodeFp4(activated / block_scale)`, pack 2 nibbles/byte
  - One warp-elected lane per (token_row, output_col_base) block writes the final scale and packed bytes

  **Isolated oracle test**: build `RunP1NativeDirectPackOracleForTesting` in `runtime/src/backend/fused_moe_prefill.cu` and a declaration in `runtime/include/nemotron/fused_moe_prefill.h`. **Model it on `RunP5NativeDirectPackOracleForTesting` at `runtime/src/backend/fused_moe_prefill.cu:13515` and `testing/backend/staged_fp4_pack_test.cpp:1001-1050`** — the live positive kP5 direct-pack oracle. Seed the accumulator from a live `partition_C` coord map (same pattern), run the packer, copy FP4 outputs back to host, compute a host-side reference (tensor scales → Relu² → per-block max-abs → FP8 scale → FP4 quantize), and **bitwise** compare all 4 output channels (`packed_data`, `block_scales`, `matmul_block_scales`, `activation_output_scale`).

  Gate: the new oracle passes bitwise on all 4 output channels at synthetic inputs chosen to be exactly representable in FP4.

  Key files: `runtime/src/backend/fused_moe_prefill.cu` (new packer + new test hook), `runtime/include/nemotron/fused_moe_prefill.h` (hook declaration), `testing/backend/p1_native_direct_pack_oracle_test.cpp` (new, modeled on staged_fp4_pack_test.cpp), `testing/CMakeLists.txt` (register).

- [ ] **6. Resolve kP1 input scale layout asymmetry**
  The `Profile == kP1` branch at `runtime/src/backend/fused_moe_prefill.cu:8744` reads `input_matmul_block_scales` with row-major offsets:
  ```cpp
  const std::size_t scale_offset =
      source_row * blocks_per_row +
      block_base + static_cast<std::size_t>(scale_index);
  scale_bytes[scale_index] = input_matmul_block_scales[scale_offset];
  ```
  But production routed-FC1 packs are built with `kSwizzled128x4` (`runtime/src/backend/request_context.cpp:62`). This branch has never been exercised in production.

  Pick one:

  - **(i) Layout-aware loading**: change line 8744 to call `LoadExecutionScaleByte(input_matmul_block_scales, source_row, block_base + scale_index, padded_blocks_per_row, input_scale_layout)` (mirroring the `else` branch at line 8750). This is symmetric with how kP5/kP13 handle the swizzled pack.
  - **(ii) Pass a row-major side buffer**: at the dispatch site (step 8), pass `input_pack.block_scales_data()` (the raw row-major input scales, before swizzling) plus `Nvfp4ScaleLayout::kRowMajor`. The kernel branch stays row-major.

  Prefer **(i)** unless the live kP1 test path has proven the branch's row-major-ness is load-bearing for some performance reason — probe first. Choose (ii) only if (i) regresses a measurable benchmark.

  Gate: the resolved path is verified by the natural P1 oracle (step 4 still passes after the change) and by running `ctest -j1` to catch any tests that exercise the kP1 branch.

  Key files: `runtime/src/backend/fused_moe_prefill.cu` (line 8744 branch + possibly `proj-2026-04-12-1022/architecture.md` to record the decision).

- [ ] **7. Extend TMA constexpr gates to support kP1 (correctness-path first)**
  Lines `runtime/src/backend/fused_moe_prefill.cu:8012-8039` hard-gate `use_p5_tma_a/b/sfa/sfb/pipeline` on `Profile == kP5`. As-is, dispatching kP1 through the unified kernel in step 8 falls through to the cooperative gmem load path. That's a valid **correctness** path but not a roofline path.

  **This step only extends the gates enough to compile-check and keep kP1 on the cooperative path by default.** Relax the constexpr branches to `Profile == kP5 || Profile == kP1` so the template instantiates for `Profile=kP1`, but the dispatch site in step 8 passes `nullptr` for the TMA descriptor arguments so all `use_p5_tma_*` predicates evaluate to `false` at runtime. kP1 correctness bring-up runs on cooperative gmem loads.

  Actual TMA enablement for kP1 (producing `use_p5_tma_a = true`) is deferred to step 11 (perf step). That requires verifying the kP5 swap-true TMA descriptor stride/alignment assumptions hold for kP1's weight and input packs, and may require new descriptor infra depending on how production kP1 input packs differ from kP5's.

  Gate: `ctest -j1` passes with no new failures. The unified kernel compiles when instantiated with `Profile=kP1`. At runtime, kP1 dispatch still uses cooperative loads.

  Key files: `runtime/src/backend/fused_moe_prefill.cu` (lines 8012-8039 and any dependent constexpr uses).

- [ ] **8. Wire kP1 dispatch through the unified swap-true kernel**
  Two-part change in `runtime/src/backend/fused_moe_prefill.cu`:

  **Part 8a — extend the specialized-direct-pack dispatch inside `Nvfp4LaunchPlannedPackedInputGroupedFp4UnifiedSwapTrueKernel`** (template at line 7737). The constexpr at line 7770 currently routes `kP5 + kFp4Direct` only. Replace with:
  ```cpp
  constexpr bool kUseSpecializedFp4DirectOutput =
      (Profile == kP5 || Profile == kP1) && P5Mode == kFp4Direct;
  constexpr bool kUseGenericDirectStage =
      UseFp4DirectOutput && !kUseSpecializedFp4DirectOutput;
  ```
  Inside the epilogue dispatch (around line 9015), profile-template the specialized branch:
  ```cpp
  if constexpr (kUseSpecializedFp4DirectOutput) {
    if constexpr (Profile == kP5) {
      nvfp4_bridge::StoreUnifiedRoutedFp4DirectPack(/* kP5 args */);
    } else {  // Profile == kP1
      nvfp4_bridge::StoreUnifiedRoutedFp4DirectPackP1(/* kP1 args */);
    }
  } else if constexpr (kUseGenericDirectStage) { /* dormant, deleted in step 10 */ }
  else { nvfp4_bridge::StoreUnifiedRoutedFp4Output<Profile, P5Mode>(/* args */); }
  ```

  **Part 8b — replace the dispatch site at line 12984.** Replace `LaunchProgrammaticKernel(..., Nvfp4LaunchPlannedPackedInputGroupedKernelSwapFalseFp4Direct<128, 64>, ...)` with:
  ```cpp
  Nvfp4LaunchPlannedPackedInputGroupedFp4UnifiedSwapTrueKernel<
      nvfp4_bridge::UnifiedRoutedFp4Profile::kP1,
      __nv_bfloat16,
      nvfp4_bridge::P5EpilogueMode::kFp4Direct,
      /*UseFp4DirectOutput=*/true>
  ```
  Parameter list mirrors the kP5+kFp4Direct dispatch at line 12526, with kP1-specific differences from step 6 (input scales) and step 7 (TMA descriptors → nullptr so kP1 falls through to cooperative).
  - `output_rows_per_expert` = 1920 (already upstream; no change)
  - `fp4_cols` = 1920 (output pack's column count)
  - Per-expert weight TMA descriptors: passed via `weights` array (unchanged vs kP5)

  **Gates**: run the full test protocol after Part 8b.
  - `cmake --build` clean
  - `ctest -j1`: expected 74/77
  - `nano_24_token_prefill_regression_test` with `NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1`: exact token match for prompt23 and prompt24
  - `prompt_length_sweep.py --runtimes native --skip-claude-eval`: `ALL CHECKS PASSED`
  - Host fp64 oracle reports `direct_prepack` mismatches at or below the BF16-WMMA fallback's baseline (`≤ 83 mismatches at ≤ 0.0145 max_abs_diff`) at the realistic bucket: h=2688, **i=1920**, n_experts=128, 16-row token chunk.

  Key files: `runtime/src/backend/fused_moe_prefill.cu` (lines 7737, 7770, 9015, 12984).

- [ ] **9. Full test protocol + regression validation**
  After step 8 lands, run the complete validation hierarchy:

  - `cmake --build build-sm120-relwithdebinfo --parallel $(nproc)` (must build clean)
  - `ctest --test-dir build-sm120-relwithdebinfo --output-on-failure -j1` (74/77 expected)
  - `NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1 build-sm120-relwithdebinfo/testing/nano_24_token_prefill_regression_test`
  - `NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1 uv run tools/oracle/prompt_length_sweep.py --runtimes native --skip-claude-eval`
  - vLLM behavioral parity oracle at `tools/oracle/compare_chat_runtimes.py` against the local vendored vLLM (binding correctness gate)
  - Per-layer diagnostic with `NEMOTRON_DEBUG_COMPARE_PREFILL_VS_LEGACY=1`

  Save artifact logs under `proj-2026-04-12-1022/post_implementation_validation/`.

  Gate: all green. Any red blocks step 10. The BF16-WMMA fallback stays in production code until step 10 confirms.

  Key files: artifacts only.

- [ ] **10. Delete dead code**
  After step 9 confirms the new path, delete the obsolete code in **deletion-order-safe** sub-commits. Each sub-commit must build clean and pass `ctest -j1` before the next starts.

  **Sub-commit 10a — Delete callers (kernels and tests that use the bespoke bridge helpers):**
  - `Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapFalseK64ScaleSmemP1Direct` at line 7444
  - `Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapFalseK64ScaleSmemP1` at line 7235
  - `Nvfp4LaunchPlannedPackedInputGroupedKernelSwapFalseFp4Direct` at line 6587 (BF16-WMMA fallback)
  - Diagnostic-trace globals and helpers tied to the bespoke P1 path (verify via grep first)
  - `testing/backend/p1_fragment_debug_oracle_test.cpp` and its CMake target (only if step 4's fix also made its `c_atom_post_mma` bucket pass; otherwise keep it pending a separate investigation)
  - `testing/backend/p1_native_fp4_mma_oracle_test.cpp` (the existing bespoke native P1 oracle that uses `LoadFragmentA/B` + `nvfp4_bridge::Gemm`)

  **Sub-commit 10b — Delete the dormant generic-direct-stage path:**
  - `nvfp4_bridge::StageUnifiedRoutedFp4DirectPrepack` at line 1118
  - `nvfp4_bridge::PackUnifiedRoutedFp4DirectStage` at line 1156
  - `kUseGenericDirectStage` constexpr branch and its smem allocation (lines 7773-7811)
  - `testing/backend/p13_generic_direct_stage_oracle_test.cpp` and its CMake target
  - `RunP13GenericDirectStageOracleForTesting` host hook

  **Sub-commit 10c — Delete the bespoke bridge surface (only after 10a, 10b clean):**
  - Verify via `grep -rn 'nvfp4_bridge::Gemm\|CFragment64\|AFragment64\|BFragment64\|LoadFragmentA_RowMajor16x64\|LoadFragmentB_ColMajor64x8' runtime testing benchmarks` that no caller remains
  - Delete `nvfp4_bridge::Gemm` at line 5587 and `CFragment64/AFragment64/BFragment64` structs at line 478
  - Delete `LoadFragmentA/B…TracedScale*Tiled*` helpers (line 4724, 4839, 4884, P13 variants)
  - Delete `Sm120BlockScaledFp4Mma` PTX wrapper if no other caller
  - Delete `kTracedP1ScaleFragmentCosizeA/B` literals at lines 266-267

  **Sub-commit 10d — TMA infrastructure decision:**
  - The swap-false TMA descriptors plumbed in commits `3b08244` / `9c3c6ec` (per-expert weight-as-B, per-CTA input-as-A) are not used by this plan. Keep them as deferred infrastructure. Add a one-line comment at each definition site: "Unused by unified kP1 dispatch (see proj-2026-04-12-1022/architecture.md). Reserved for a future swap-false kernel."

  Re-run the full test protocol after each sub-commit. Source size should shrink by ~800-1200 lines after 10a-10c.

  Key files: `runtime/src/backend/fused_moe_prefill.cu`, `runtime/include/nemotron/fused_moe_prefill.h`, `testing/backend/*.cpp` (multiple), `testing/CMakeLists.txt`, `runtime/src/backend/routed_p5_tma_descriptor.{cuh,cu}`.

- [ ] **11. Roofline benchmark + perf analysis**
  Run the fresh-prefix cold sweep with the new kernel:
  ```bash
  NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1 \
    build-sm120-relwithdebinfo/benchmarks/nano_fused_decode/nano_fused_decode_bench \
    --manifest artifacts/manifests/forward_runtime_manifest_nano_rtx5090_unverified.json \
    --mode phased --prompt-tokens 256 --decode-tokens 1 --warmup 1 --iterations 3
  # repeat for 1024 and 4096
  ```
  Plus the routed-profile-mix diagnostic with `NEMOTRON_ROUTED_PROFILE_DEBUG=1` to confirm kP1 hits the new path. Compute kP1's share of routed-FC1 time at the 16-row and 24-row buckets.

  **Roofline math:**
  - **Denominator**: 1792 GB/s peak (RTX 5090 GDDR7 per NVIDIA Blackwell architecture PDF). This is the primary denominator. Locally we have historically used ~1300 GB/s as an "empirical effective" number for the delivered-to-peak ratio; if reporting that in passing, label it explicitly as a local observation and not a spec.
  - **Numerator**: total bytes touched per FC1 forward pass, computed **per CTA** (not per unique expert). Each CTA binds `(expert_index, output_row_base)`; the same expert weights can be touched by multiple CTAs. Count the actual grid's per-CTA loads:
    - Per-CTA: weight tile (M_cta × K) in FP4 packed bytes + per-block FP8 scales + input tile (N_cta × K) in FP4 packed bytes + input per-block FP8 scales
    - Times (grid_m × grid_n) CTAs per layer
    - Times number of active experts in the layer (`≤ token_count × top_k`, deduped to unique experts actually dispatched)
  - **Weight width**: use the padded execution width **1920** per expert, not 1856

  **Baseline comparison**: measure kP1 cooperative-path bandwidth first (since step 7 kept it on cooperative). Then, as an optional sub-step, enable TMA for kP1 by making the dispatch site pass non-null TMA descriptor pointers (verifying kP5 descriptor stride/alignment compatibility with kP1 packs) and re-measure. Document both numbers side-by-side.

  **Target and iteration**: compare the kP1 cooperative measurement to the **production kP5 cooperative-path bandwidth at the same bucket size** (measure kP5 by running the same bench with the kP5-dominant route — requires building a short probe manifest). Target: kP1 within 10% of kP5 at matching bucket. If below target, profile via `nsight-compute` to identify the bottleneck (TMA stage count, smem bank conflicts, accumulator register pressure, warp scheduling), iterate one optimization per sub-commit, re-measure. Stop when the target is met OR a specific bottleneck is documented for deferral with a new follow-on plan.

  Document in `proj-2026-04-12-1022/post_implementation/roofline.md`: achieved kP1 bandwidth, % of 1792 peak, kP1 vs kP5 comparison at matching bucket, per-bucket share of FC1 wall time, cooperative vs TMA numbers (if both measured).

  Key files: `benchmarks/nano_fused_decode/nano_fused_decode_bench.cpp` (run only), `proj-2026-04-12-1022/post_implementation/roofline.md` (new).

## Progress

| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Re-probe kP1 / kP13 tile_shape and AccumLayout modes | pending | — | Produces `probes.md` |
| 2 | Diagnose why one-line (128,128) fix failed | pending | — | Identifies real root cause |
| 3 | Pick fix direction; write architecture note | pending | — | Codex review hard gate |
| 4 | Apply fix; verify natural P1 oracle passes bitwise | pending | — | |
| 5 | Add `StoreUnifiedRoutedFp4DirectPackP1` + isolated oracle | pending | — | Modeled on `RunP5NativeDirectPackOracleForTesting` |
| 6 | Resolve kP1 input scale layout asymmetry | pending | — | |
| 7 | Extend TMA constexpr gates (correctness path only) | pending | — | kP1 on cooperative loads by default |
| 8 | Wire kP1 dispatch through unified swap-true kernel | pending | — | |
| 9 | Full test protocol + regression validation | pending | — | |
| 10 | Delete dead code | pending | — | 4 deletion-order sub-commits 10a-10d |
| 11 | Roofline benchmark + perf analysis | pending | — | Per-CTA math, 1792 GB/s peak |
