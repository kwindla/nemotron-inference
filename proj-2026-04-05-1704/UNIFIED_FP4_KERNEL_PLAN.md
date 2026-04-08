# Unified FP4 Routed Kernel

## End Goal

Replace the hand-rolled routed FP4 kernel variants that currently back the
dominant traced profiles `P5`, `P12`, `P13`, and `P15` across routed `FC1`
and `FC2` with a single parameterized kernel template that uses the proven
swizzled-smem + CUTE-copy + atom-level-zip architecture from `P15`.

After this rewrite:

- One kernel template handles all targeted traced profiles in this rewrite
- The tile shape, `swap_ab`, and macro-tile-K are template parameters
- A per-profile traits layer binds each instantiation to the exact
  `CollectiveBuilder`, scale layouts, and dispatch profile id
- The CUTLASS `CollectiveBuilder` produces `SmemLayoutA/B`, `TiledMma`,
  `SmemCopyAtomA/B` for each profile automatically
- No probe-generated coordinate tables, no manual fragment families, no
  per-element nibble extraction
- The same kernel body runs P5 (`128x128x64 swap_ab=true`),
  P12 (`128x128x128 swap_ab=true`), P13 (`128x128x64 swap_ab=true`),
  and P15 (`256x128x64 swap_ab=true`)

Scope note:
- This rewrite explicitly targets `P5` on routed `FC1` and `P12/P13/P15` on
  routed `FC2`
- the remaining traced `FC1` swap_ab=false profiles (`P0/P1/P4/P7`) stay out
  of scope for this first unified landing unless they are pulled in later

## Why a Single Rewrite

The four existing hand-rolled kernels share the same structural flaw: they
load packed FP4 data into flat row-major smem, extract individual nibbles
through coordinate-table lookups, pack them into `uint32_t` registers, and
call the MMA atom through manual fragment families.  This approach:

1. **Is slower than the BF16 fallback it replaced.**  The P5 FP4
   scale-smem path regressed prefix4 from 50 ms to 63 ms (+27%) and
   prefix128 from 120 ms to 231 ms (+92%).  The bottleneck is the
   per-element nibble extraction loop (8 dependent byte loads + shifts
   per register), which is instruction-bound while the BF16 WMMA path
   is bandwidth-bound on smem.

2. **Has a correctness bug in P12.**  The `128x128x128` k=128 macro
   tile with the Nano FC2 dimension `k=1856` produces 14 full k-tiles
   and 1 partial 32-wide tail tile.  The hand-rolled P12 kernel
   corrupts the partial tile, producing `token_id=2147483647` at
   prefix128 with the production `window=4096` contract.

3. **Required 10+ probe-patch cycles for P15.**  P15's `256x128x64`
   tile has different V-counts between operand and scale fragments,
   which made the hand-rolled approach structurally incompatible with
   `cute::gemm`.  Only switching to swizzled smem + CUTE copy atoms +
   atom-level zip resolved it.

4. **Cannot be fixed incrementally.**  Debugging P12's partial-tile
   handling, optimizing P5's nibble extraction, and maintaining four
   independent coordinate-table families are all wasted work if the
   end state is the swizzled-smem architecture anyway.

The P15 kernel already proves the target architecture works.  Generalizing
it to cover all profiles is less work than fixing the remaining bugs in
the old architecture, and it eliminates the entire class of problems at
once.

## Execution Rules

These rules are mandatory for the rewrite:

1. No new FP4 routed attempt may preserve flat row-packed operand staging in
   the hot path.  If an attempt still depends on `a_packed` / `b_packed`
   hand-packed nibble extraction, reject it before coding.
2. No new probe-generated coord tables or fragment-family shims for
   `P5/P12/P13` unless a specific CUTLASS type fact is still unknown.  The
   goal of this rewrite is to delete that architecture, not extend it.
3. Every correctness or TTFT claim must be checked under the production
   `window=4096` contract.  Small-prefix-only smoke runs are not sufficient.
4. Old kernels are deleted only after the unified path passes the focused and
   matrix gates listed below.

## Latest Landing Status

Current stable routed state:
- `P5`, `P12`, and `P13` are now live on the unified swizzled FP4 kernel
- `P15` remains on the already-proven exact FP4 kernel
- the live `P5` trace hook remains available through `NEMOTRON_P5_SCALE_DEBUG`

What resolved `P5`:
- built the standalone
  [p5_swizzled_pipeline_test.cu](/home/khkramer/src/nemotron-inference/testing/backend/p5_swizzled_pipeline_test.cu)
  instead of continuing the old probe-patch loop inside the fused kernel
- proved the swizzled-smem + CUTE copy + zipped atom-MMA path in isolation for
  the real `P5` token tile (`128x32x64`), `dispatch_rows=4/5`, and Nano-like
  `total_k=1344`
- corrected the activation-side execution-scale source to
  `matmul_block_scales_data()`
- switched default routed `P5` dispatch to the unified kernel only after:
  - `fused_moe_prefill_test`
  - `multi_turn_prefix_reuse_test`
  - the new TTFT smoke cases
  all passed on the unified path

Current focused validation:
- targeted `ctest` pass:
  - `fused_moe_prefill_test`
  - `p5_swizzled_pipeline_test`
  - `multi_turn_prefix_reuse_test`
  - `nano_prefix_cache_ttft_prefix128_tail4_smoke`
  - `nano_prefix_cache_ttft_prefix256_tail128_committed_smoke`
  - `nano_prefix_cache_ttft_prefix256_tail128_global_root_smoke`
  - result: `6/6` passed

Current measured TTFT smoke results on the live default unified `P5` path:
- `prefix128 / tail4`
  - `cold_prefill_prefix128 = 161.579 ms`
  - `cached_committed_head_prefix128_tail4 hot-prefix = 63.801 ms`
  - `cached_global_root_prefix128_tail4 hot-prefix = 63.917 ms`
- `prefix256 / tail128`
  - `cached_committed_head_prefix256_tail128 hot-prefix = 165.190 ms`
  - `cached_global_root_prefix256_tail128 hot-prefix = 165.499 ms`

Measured benefit versus the previous default exact `P5` path on the same smoke
cases:
- `prefix128 / tail4`
  - cold prefill: `256.344 -> 161.579 ms`
  - committed-head hot-prefix: `73.566 -> 63.801 ms`
  - global-root hot-prefix: `73.975 -> 63.917 ms`
- `prefix256 / tail128`
  - committed-head hot-prefix: `267.605 -> 165.190 ms`
  - global-root hot-prefix: `263.181 -> 165.499 ms`

What this now proves:
- the old exact `P5` kernel was a material part of the current TTFT
  regression, and the unified swizzled path is the correct replacement
- the remaining cold-TTFT gap is no longer just a `P5` problem, because even
  the improved unified `P5` path is still well above the older
  `~120 ms prefix128` checkpoint range
- the next optimization work should therefore treat unified `P5` as landed and
  move on to the remaining routed FP4 kernels and broader prefill overhead

## What TRT-LLM Does

The live TRT-LLM NemotronH routed path is the CUTLASS fused-MoE custom op,
selected through
`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl`.
From our live tactic traces and compile probes:

- the live NVFP4 path builds `CollectiveMainloop` through
  `cutlass::gemm::collective::CollectiveBuilder` in
  `moe_gemm_tma_ws_launcher.inl`
- the tactic selector picks `(TileM, TileN, TileK, swap_ab)` per row count
- compile probes on that live path directly exposed
  `SmemLayoutA/B/SFA/SFB`, `SmemCopyAtomA/B/SFA/SFB`, `TiledMma`, and
  `partition_S(as_position_independent_swizzle_tensor(...))`

For mainloop structure, the closest readable blockscaled reference is the
analogous SM120 TRT/CUTLASS file
`fp8_blockscale_gemm/sm120_blockwise_gemm/sm120_fp8_moe_gemm_1d1d.cuh`
alongside `sm120_utils.cuh`.  Those files are not the literal live NVFP4 MoE
kernel, but they do show the same builder-driven blockscaled pipeline shape:

```text
partition_S(as_position_independent_swizzle_tensor(sA/sB/sSFA/sSFB))
-> copy(smem -> reg)
-> fp4/fp8 shift or fragment transform
-> blockscaled MMA with zipped operand+scale tensors
```

Mechanistically, the routed TRT family uses one builder/collective pattern
across tactics even though tile shapes and epilogue variants differ.

- TRT-like collective pattern, stated at the mechanism level:
  ```text
  TMA loads into swizzled SmemLayoutA/B/SFA/SFB
  -> partition_S(as_position_independent_swizzle_tensor(...))
  -> copy(smem -> reg fragments)
  -> fp4/fp8 shift or fragment transform
  -> blockscaled MMA on operand+scale fragments
  ```
- Confirmed fact: the blockscaled collective applies the operand transform
  step before MMA.  In the analogous SM120 blockscaled CUTLASS reference this
  is `fp4_shift_A/B` for FP4-style operand views and
  `transform_fragment_for_qmma` for the FP8 blockscaled helper path.
- Multi-stage pipelining overlaps TMA loads with MMA compute in TRT
- Epilogue/finalize behavior varies by tactic; the live traced gemm2 path often
  has finalize fusion enabled (`epilogue_fusion=1`)

## Technical Requirements

### Template Parameters

```cpp
template <int TileM, int TileN, int TileK, bool SwapAB, typename OutputType>
```

Instantiated for the traced tactic family:
- `<128, 128, 64, true, float>` — P5 and P13
- `<128, 128, 128, true, float>` — P12
- `<256, 128, 64, true, float>` — P15

### CUTLASS Type Machinery

Each instantiation uses the CUTLASS `CollectiveBuilder` to produce:

```cpp
using CollectiveMainloop = cutlass::gemm::collective::CollectiveBuilder<
    Sm120, OpClassBlockScaledTensorOp,
    nv_float4_t<float_e2m1_t>, LayoutA*, 32,
    nv_float4_t<float_e2m1_t>, LayoutB*, 32,
    float,
    Shape<Int<TileM>, Int<TileN>, Int<TileK>>,
    Shape<_1, _1, _1>,
    StageCountAutoCarveout, KernelScheduleAuto>::CollectiveOp;
```

Extract from the collective:
- `TiledMma`, `SmemLayoutA`, `SmemLayoutB`
- `SmemCopyAtomA`, `SmemCopyAtomB`

Scale layouts should not be pulled ad hoc from scattered `TracedP*...`
aliases.  Instead, define one local unified profile key plus one per-profile
traits layer:

```cpp
enum class UnifiedRoutedFp4Profile { P5, P12, P13, P15 };

template <UnifiedRoutedFp4Profile Profile>
struct RoutedFp4ProfileTraits;
```

Each specialization owns:
- `TileM`, `TileN`, `TileK`, `SwapAB`
- the exact `CollectiveMainloop`
- `SmemLayoutA/B`
- `SmemLayoutSFA/SFB`
- `SmemCopyAtomA/B/SFA/SFB`
- profile-specific dispatch name / debug string
- which runtime selectors bind to it (`RoutedGemm1Profile` and/or
  `RoutedGemm2Profile`)

This keeps the kernel body unified while letting each traced TRT tactic bind
to its own validated CUTLASS types.

For scale copy atoms specifically, prefer the live `CollectiveMainloop`
aliases (`CollectiveMainloop::SmemCopyAtomSFA/SFB`) when they are available,
because that is what our current TRT launcher probes and runtime aliases use.
The analogous helper reference in `sm120_utils.cuh` reduces to
`Copy_Atom<AutoVectorizingCopy, ElementSFLoad>`, so these should agree in
practice, but the launcher/collective aliases are the direct source of truth
for this rewrite.

### Shared Memory

Single-stage allocation sized by `cute::size(take<0,2>(SmemLayoutA))` and
`cute::size(take<0,2>(SmemLayoutB))`.  This is the pattern proven in P15:
full `SmemLayoutA` type for the tensor view but only stage-0 storage
allocated.

Scale smem uses the profile traits `SmemLayoutSFA/SFB` types and their fill
path, but the production helper must be generalized.  Do not keep
`StoreTracedP5ScaleWordK64` as the unified kernel entry point.  Replace it
with one helper such as `StoreExecutionScaleWords<Traits>(...)` that writes
execution-scale bytes into swizzled `SmemLayoutSFA/SFB` for any `TileK` and
traced profile.

No flat `a_packed`/`b_packed` arrays.  Data goes directly from global
memory into swizzled smem via byte-level scatter:

```cpp
auto stage0 = SmemLayoutA{}(_, _, Int<0>{});
auto* sw = reinterpret_cast<uint8_t*>(smem_swizzled_a);
for (int i = tid; i < total_bytes; i += blockDim.x) {
    int row = i / row_bytes;
    int col_byte = i % row_bytes;
    auto elem_offset = stage0(row, col_byte * 2);
    sw[int(elem_offset) / 2] = global_src[row * stride + col_byte];
}
```

The direct-to-swizzled staging path must explicitly zero-fill masked tail
regions.  Do not rely on "natural" zeros from the source contract.  The exact
`P12` fix already proved that the partial `K` tail requires explicit masked
writes for:
- packed operand bytes beyond `input_cols`
- execution-scale bytes/words beyond `input_cols`

### MMA Consumer

The pattern from the working P15 kernel:

```cpp
// Copy from swizzled smem to registers
copy(s2r_copy_A, tCsA(_, _, _, Int<0>{}), tCrA_cv);
copy(s2r_copy_B, tCsB(_, _, _, Int<0>{}), tCrB_cv);

// FP4 shift
fp4_shift_A(MMAOp{}, tCrA_cv(_, _, k));
fp4_shift_B(MMAOp{}, tCrB_cv(_, _, k));

// Atom-level zipped GEMM
for k, n, m:
    mma_atom.call(c_atom,
        make_zip_tensor(a_atom, sfa_atom),
        make_zip_tensor(b_atom, sfb_atom),
        c_atom);
```

This operand shift is a confirmed requirement, not an open question.  The
analogous SM120 blockscaled collective applies the transform immediately after
smem->reg copy and before MMA.

The atom-level dispatch is required for FP4 because the BLOCKSCALED MMA
atom's SFA has `size=64` while the A fragment's V-count is 32.
In our local runtime/TU, the currently proven path is atom-level dispatch with
`mma_atom.call(make_zip_tensor(...))`; our earlier tiled-level `cute::gemm`
attempts on the hand-built FP4 path did not preserve the right register-tensor
contract.  TRT/CUTLASS proper can express the full blockscaled path through its
collective mainloop types, so this is a constraint of the current native
rewrite, not a universal CUTLASS limitation.

The `SFAtomLayout` is:
```cpp
Layout<Shape<Shape<_16, _4>>, Stride<Stride<_0, _1>>>
```
(64 logical elements, 4 physical ue4m3 bytes, broadcast groups of 16.)
This is the same for all FP4 profiles — it comes from the MMA atom's
`SFALayout`, not from the tile shape.

### Scale Loading

The production contract should follow the now-working exact `P15` path:
- execution scales are staged into swizzled `SmemLayoutSFA/SFB`
- `copy(...)` moves them into register fragments `tCrSFA/tCrSFB`
- the atom-level zipped MMA consumes those register fragments directly

The older flat row-major `uint32` row-scale lookup is still useful as a debug
reference and cross-check, and the `partition_C(identity_tensor)` base-row
mapping has already been proven bounds-safe for `P15`.  But it should not be
the primary production data path for the unified kernel, because TRT/CUTLASS
itself stages scale data through `SmemLayoutSFA/SFB -> copy -> reg fragment`.

### K-tile Loop

```cpp
for (size_t k_base = 0; k_base < input_cols; k_base += TileK) {
    // Stage operands and scales for this k-tile into swizzled smem
    // Scatter from global to swizzled smem (one pass, no flat staging)
    __syncthreads();
    // Copy smem → registers, shift, zip, MMA
    __syncthreads();
}
```

The partial tail tile (when `input_cols` is not a multiple of `TileK`) is
handled by the data staging with explicit masking and zero-fill.  No profile
may assume the tail is safe without guarded staging.

### Epilogue / Store

The `partition_C(identity_tensor)` store path from P15.  Per-element
scalar store with boundary checking.  This is correct but not optimized.
The TRT-LLM vectorized epilogue (smem-staged, 128-bit writes) is a
later optimization.

The first unified landing is correctness-first.  Keep the scalar
`partition_C(identity_tensor)` store path until the unified kernel passes all
correctness and TTFT gates.  Only then consider TRT-style vectorized epilogue
work.

### What Gets Deleted

- `LoadFragmentA_RowMajor16x64Tiled` and all per-profile variants
- `LoadFragmentB_ColMajor64x8Tiled` and all per-profile variants
- `LoadTracedP15AFragmentsRowMajor16x64`,
  `LoadTracedP15BFragmentsColMajor64x8`
- `StoreTracedP15CFragmentsRowMajor`
- `AFragment64`, `BFragment64`, `CFragment64` types
- All `kTraced*ACopyCoordCapacity`, `kTraced*BCopyCoordCapacity`,
  `kTraced*CCopyCoordCapacity` constants
- `FillPhysicalCoordMapCopyViewLimited`
- `p15_probe_generated.h`, `p15_smem_partition_generated.h`
  (probe-generated coordinate tables)
- The current separate routed FP4 kernel entry points and their
  profile-specific helper stacks, including:
  `Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapTrueK64ScaleSmem`,
  `Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapTrueK64ScaleSmemP15`,
  `Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapTrueK128P12`
  (and any P1/P7 variants if they exist)

### What Gets Kept

- The `CollectiveBuilder`-derived type facts, carried through the new
  per-profile traits layer
- The validated traced `SmemLayoutSFA/SFB` contracts, surfaced via the
  traits layer
- A generic execution-scale staging helper, not the existing
  `StoreTracedP5ScaleWordK64` name and contract
- The dispatch tables in `SelectRoutedGemm1Profile`,
  `SelectRoutedGemm2Profile`, and `LaunchPlannedPackedInputMatVec`
- The standalone `p15_swizzled_pipeline_test.cu` (generalized to cover
  all profiles)

## Execution Strategy

1. Build the per-profile traits layer first.  The unified kernel body should
   consume `Traits`, not raw scattered aliases.

2. Write the unified kernel using the working swizzled-smem `P15` body as the
   starting point.  Parameterize by `Traits` and `OutputType`.

3. Instantiate for `P5 (128,128,64,true)` first.  This is the
   highest-traffic profile and the one with the known performance
   regression.

4. Validate the unified `P5` instantiation:
   - `fused_moe_prefill_test`
   - `multi_turn_prefix_reuse_test`
   - `nano_prefix_cache_ttft_prefix128_tail4_smoke`
   - prefix4 / prefix128 TTFT under the `window=4096` contract
     (must recover toward the pre-`72e8e40` baseline)

5. Instantiate for `P12 (128,128,128,true)`.  Validate:
   - `fused_moe_prefill_test`
   - `multi_turn_prefix_reuse_test`
   - `nano_prefix_cache_ttft_prefix128_tail4_smoke`
   - `prefix128` no longer crashes under the full matrix contract

6. Instantiate for `P13 (128,128,64,true)`.  This shares the `P5` tile
   shape; the main difference is dispatch regime and FC2 integration.  Validate
   both `tail128` smoke tests after landing it.

7. Once `P5`, `P12`, `P13`, and `P15` all run through the unified path and
   the regression gates pass, delete the old hand-rolled kernel variants and
   all supporting infrastructure.

8. Full TTFT sweep:
   - `prefix4`, `prefix128`, `prefix4096` with `tail4`
   - production `window=4096` contract preserved
   - resumed `tail128` smoke cases still passing

## Validation Bar

Focused regression gates before deleting any old FP4 kernel:
- `nano_prefix_cache_ttft_prefix128_tail4_smoke`
- `nano_prefix_cache_ttft_prefix256_tail128_committed_smoke`
- `nano_prefix_cache_ttft_prefix256_tail128_global_root_smoke`

These smoke tests must keep the live `window=4096` contract.  They exist
because small-prefix-only TTFT runs already hid real routed failures.

After each profile instantiation:
- `p15_swizzled_pipeline_test` (generalized)
- `fused_moe_prefill_test`
- `multi_turn_prefix_reuse_test`
- `compute-sanitizer --tool memcheck ./testing/fused_moe_prefill_test`

After all profiles:
- Full TTFT matrix with `--moe-prefill-window-tokens 4096`
- prefix4 cold TTFT ≤ 55 ms (recover baseline)
- prefix128 cold TTFT ≤ 130 ms (recover baseline, no crash)
- prefix4096 cold TTFT ≤ 2600 ms (no regression)
- Decode throughput ≥ 75 tok/s (no regression)

## Risk

The main risk is that the atom-level dispatch (manual triple loop instead
of `cute::gemm`) is inherently slower than TRT-LLM's `TiledMma`-level
dispatch due to worse register reuse and lack of serpentine iteration.
If the unified kernel is correct but slower than the BF16 fallback, the
next step would be either:

- Optimizing the atom-level dispatch (serpentine iteration, register
  blocking)
- Or finding a way to make `cute::gemm` work with the FP4 V-count
  mismatch (the fundamental blocker we hit during P15 development)

Both are tractable follow-ups, not blockers for the rewrite.
