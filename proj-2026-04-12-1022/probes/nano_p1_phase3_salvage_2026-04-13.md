# NanoP1 Phase 3 — salvage notes from the 8.5 h oscillating debug session

Date: 2026-04-13
Test: `nano_p1_mainloop_oracle_test` Phase 3 (Nano bucket K=2688) —
`testing/backend/nano_p1_mainloop_oracle_test.cpp:973` `RunPhase3NanoBucketK2688()`
Kernel under test: `ComputeNanoP1AccumTile` →
`runtime/src/backend/fused_moe_prefill.cu:12218`
Reference: `proj-2026-04-12-1022/trtllm_reference/golden_nano_k2688/bf16_gemm1_tactic1.bin`
(BF16 dump of flashinfer's `gen_cutlass_fused_moe_sm120_module` P1 tactic
gemm1 output for the Nano K=2688 bucket; the directory is named
`trtllm_reference/` for historical reasons, but the actual producer is
flashinfer's JIT-built module, not the TRT-LLM binary. See §3 of "Status
of open questions after the 2026-04-13 probe runs" below for the full
attribution chain.).

## Why this file exists

A `/loop`-driven debug session ran 16 compactions in 8.5 h on this Phase 3
mismatch with zero commits. Each compaction re-derived the problem
characterization from scratch and proposed a different root-cause theory; each
"fix" oscillated the mismatch count but did not converge. The trajectory was
hypothesis drift, not progress, and the cheapest probe path from CLAUDE.md
("Probe-then-decide" §1) was never actually taken — every iteration ran the
full Phase 3 integration test with runtime DIAG `printf` bisection, which is
why context kept blowing out every ~30 min.

This file freezes the durable empirical findings and the ruled-out hypotheses
so the next pass starts from the right place instead of re-deriving.

## Mismatch oscillation history

| Wall time (2026-04-12 → 13) | Mismatches | % of tile | Hypothesis active at the time |
| --- | --- | --- | --- |
| 16:36 | 230,087 | 93.6 % | Wrong `partition_fragment_A`/`_B` convention + swapped data placement |
| 19:50 | 243,611 | 99.1 % | "Fix" to the convention made it worse |
| 21:55 | ~230,000 | ~93.6 % | `do_shuffle=True` row-permutation patch — "barely changed the count" |
| 00:47 | 230,087 | 93.6 % | New theory: `TracedP5PermTileN` causes `NanoP1AccumLayout` M/N role inversion |

Two distinct attractor counts (230,087 ≈ 93.6 %, 243,611 ≈ 99.1 %) and zero
movement away from them across four "fix" attempts. The shape of the
mismatch — see "Pattern" below — is also unchanged across all four states,
which is the strongest evidence that none of the candidate fixes touched the
actual variable.

## Empirical findings worth preserving

These are the things the runtime bisection actually pinned down. Do not throw
them away on the next pass — re-derive only if a probe contradicts them.

1. **NanoP1TiledMma `(M,N)` thread mapping** — captured empirically by writing
   per-lane sentinels into `accumulator_scratch` from inside the mainloop and
   reading the on-device coord. The mapping is consistent with the cooperative
   SM120 block-scaled atom geometry recorded in
   `proj-2026-04-12-1022/probes/step_4a_probe_values.txt` (`AtomLayoutMNK =
   (_4,_2,_1)`, `ValLayoutMNK = Tile<128, ((_8,_2,_2), stride (1,16,8)),
   64>`, 32 threads/atom, 256 threads total, `tCrC_profile = (4,2,8)`).

2. **`SM80_16x8_Row` C-layout is column-major (M-fast).** Confirmed against the
   TRTLLM reference. This kills the "C is row-major and we're transposing on
   store" theory.

3. **Mismatch pattern is structural, not noise.** The 8×8 dump from
   `RunPhase3NanoBucketK2688` (lines 1053–1065 of the test) shows
   surviving matches concentrated on an axis-aligned sub-grid pattern,
   not random scatter. **What the prior session read as the exact
   pattern:** `M % 8 ∈ {0, 2}` **and** `N % 8 ∈ {0, 2}` (25% per axis,
   ≈6.25% overall), which the prior session interpreted as the footprint
   of an intra-atom thread→(M, N) role mismatch. **Caveat added 2026-04-13
   re-audit:** the prior session was demonstrably wrong about causal
   reasoning 16 times in a row, so their *reading* of the pattern also
   deserves fresh verification against a larger diagnostic window
   (32×32 + `M % 32` / `N % 32` histogram). A 2/32 ≈ 6.25% rate is also
   consistent with the fixed points of `srcToDstBlk32RowMap`
   (`r ∈ {0, 31}` per 32-block), which would instead point at candidate
   G1 in "Gaps in the hypothesis list" below. Re-verifying which reading
   is correct is the first step of the recommended next probe.

4. **Convergence at K=2688 is a real anchor.**
   `proj-2026-04-12-1022/trtllm_reference/NOTES.md` §5 shows that all legal
   SM120 FP4 MoE tactics converge bitwise on this bucket, so the reference
   `bf16_gemm1_tactic1.bin` is the right ground truth: matching it proves
   mathematical consistency with a legal SM120 FP4 MoE kernel at this Nano
   scale, even if it does not by itself prove "same `CollectiveBuilder`
   instantiation as TRT-LLM" (that sharper claim still belongs to step 4a's
   compile-time probe).

## Hypotheses already ruled out (do not retest without new evidence)

- **`UseF8f6f4 = false`.** The atom in use is the `f8f6f4` block-scaled
  variant; the alternative was attempted and produced the *same* mismatch
  count, ruling out the dispatch as the cause.
- **`CLayout` row-major.** SM80_16x8_Row's CLayout is column-major (M-fast)
  here; flipping it does not change the mismatch pattern.
- **`do_shuffle=True` row-permutation *math*.** The prior session verified
  that the inverse-shuffle formula at `fused_moe_prefill.cu:12331-12338`
  (`(r%4)*8 + r/4`) is the bitwise correct inverse of flashinfer's
  `srcToDstBlk32RowMap`, and that the staged smem bytes equal what the
  host would stage for the same logical input. **Caveat:** "right inverse"
  only rules out the math, not the *direction*. Whether the kernel should
  be applying the inverse *at all* (vs reading shuffled data natively like
  flashinfer's kernel may do internally) is an open question and is now
  listed as candidate #1 under "Gaps in the hypothesis list" below.
- **Scale-loading *sizes*.** `tCrSFA` / `tCrSFB` were measured at step 4a
  (`(64, 2, 2)` / `(64, 8, 2)`) and match the reverted reference exactly.
  **Caveat:** this proves size/shape match, not per-K-group assignment.
  The custom `NanoP1PartitionScaleA`/`NanoP1PartitionScaleB` helpers at
  `fused_moe_prefill.cu:12284-12285` parallel the CUTLASS
  `sm120_blockscaled_mma_tma.hpp` logic but were never verified against it
  at the per-K-group level. Listed as candidate #3 under "Gaps in the
  hypothesis list" below.
- **(2026-04-13 probe) Layouts moved by the prior session's edits.** The
  compile-time layout probe was run twice with identical probe code: once
  against the working tree (with all 57 lines of prior-session edits in
  place) and once against the clean HEAD baseline of
  `runtime/src/backend/fused_moe_prefill.cu`. Both stderrs are byte-for-byte
  identical (md5 `52f31c2dd0473dafa6777300dc73b5ec`). All 8 probed layouts —
  `tCsA`, `tCsB`, `tCrA`, `tCrB`, `tCrA_cv`, `tCrB_cv`, `partC`, `accum` —
  are bit-identical between baseline and current. None of the prior
  session's "fixes" (A↔B operand swap in the staging loops, three
  `CoordGet0`↔`CoordGet1` swaps in the store paths, the `do_shuffle` row
  permutation, the `kTileM`↔`kTileN` loop-bound shuffle) moves any CUTE
  partitioning. **The bug is purely behavioral — what data lands in which
  smem and how coords are unpacked — not in layouts.**
- **(2026-04-13 probe) `partC` vs `NanoP1AccumLayout` stride mismatch.**
  Decoded `partC =
  ((2,2), 2, (2,4)) : ((N:1, M:8), M:64, (N:8, N:32))`,
  where `N:k` / `M:k` denote ScaledBasis strides along basis-1 (N) /
  basis-0 (M). Same shape as `NanoP1AccumLayout`, and crucially mode 1 of
  `partC` is pure-M (stride 64) and mode 2 is pure-N (strides 8 and 32) —
  so `partition_C` and `partition_fragment_C` walk the same MMA iteration
  axes in the same order. The `StoreNanoP1CFragmentsRowMajor` triple-nested
  loop pairs the right accumulator slot with the right `(M, N)` coord at
  every `(reg, n_fragment, m_fragment)` index. The misleading `n_fragment`
  / `m_fragment` variable names cancel cleanly. The original 00:47
  "M/N role inversion" hypothesis is **definitively dead** on this code path.

## Status of open questions after the 2026-04-13 probe runs

After the compile-time layout probe on 2026-04-13 and the follow-up source
reads into flashinfer and TRT-LLM, the prior session's four-item "what we
still need to know" list was narrowed to two remaining items plus one
settled item (operand-role intent). **Important (2026-04-13 re-audit):**
that two-item narrowing was an artifact of the prior session's
convergence trajectory, not an exhaustive enumeration. Six additional
candidate classes (G1..G6) that are consistent with Phase 1 passing and
Phase 3 failing are captured in a new section **"Gaps in the hypothesis
list"** after the settled subsection; treat them as co-equal candidates
with the two listed below.

### Open: `fp4_shift_A` / `fp4_shift_B` orientation under `TracedP5PermTileN`

The shifts at `fused_moe_prefill.cu:12428-12429` (current line numbers in
the prior-session-edited tree) target the natural K-sequential FP4 nibble
order. With `NanoP1ValLayoutMNK = Tile<128, TracedP5PermTileN, 64>`
introducing a non-trivial N permutation, the per-K-block shift on the B
side may be reading the wrong nibble pair. Cheapest test: hold every other
variable constant and toggle one of `fp4_shift_A` / `fp4_shift_B`
independently while watching the mismatch count.

### Open: source-side staging vs the smem swizzle the MMA copy expects

The hand-rolled byte loops at `fused_moe_prefill.cu:12329-12347` (A) and
`12372-12386` (B) write into the swizzled smem indexed by `stage0_A(row,
byte_index*2)` / `stage0_B(row, byte_index*2)`. The CUTE layouts are
identical between the baseline and the current edited tree (proved
2026-04-13), so the staging loops correctness depends entirely on whether
the byte stream they emit matches what the s2r copy *expects* given those
layouts. The `tCsA` mode-0 stride `(1, 8192)` and `tCsB` mode-1 stride
`4096` in `nano_p1_phase3_layouts_2026-04-13.md` describe what the s2r
copy sees. Validate via the "verify the test reference before trusting a
bisection" pattern (CLAUDE.md): write a known sentinel into the staged
smem, dump the per-lane `tCrA_cv` / `tCrB_cv` register state after
`cute::copy`, and compare to a host model. The dump-oracle template at
`testing/backend/p13_generic_direct_stage_oracle_test.cpp` is the right
starting point.

### Settled: operand-role intent — DEFINITIVELY SETTLED (2026-04-13)

*The investigation took a detour through two misleading signals before
landing on the correct answer; the full trail is preserved here so nothing
has to be re-derived.*

**Phase 3 reference is produced by flashinfer, not by
`libtensorrt_llm.so`.** `proj-2026-04-12-1022/trtllm_reference/capture_bf16_gemm1.py:249`
calls `gen_cutlass_fused_moe_sm120_module(use_fast_build=False).build_and_load()`,
which is a flashinfer JIT-built module. The `NEMOTRON_TRTLLM_DUMP_GEMM1`
sentinel is patched into flashinfer's vendored
`fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh` (confirmed at
`capture_bf16_gemm1.py:133`), not into any TRT-LLM header. Nothing in the
Phase 3 pipeline touches `libtensorrt_llm.so` — that binary only matters
for the separate pytorch/trtllm backend runtime path, which is not used to
capture the reference.

**Flashinfer ships only the public variant.** The vendored source tree at
`.venv-trtllm/lib/python3.12/site-packages/flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/`
(the path contains "nv_internal" but the sub-directory is the *public*
`cutlass_kernels`, not `internal_cutlass_kernels`) is bit-for-bit
identical to the TRT-LLM cutlass_kernels source on the operand-layout and
SwapAB-binding definitions:

- `include/moe_gemm_kernels.h:79-80`:
  `LayoutA = RowMajor`, `LayoutB = ColumnMajor`.
- `moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl:480-483`:
  ```cpp
  using SwappedMainloopElementA =
      std::conditional_t<SwapAB, MainloopElementWeight, MainloopElementAct>;
  using SwappedMainloopElementB =
      std::conditional_t<SwapAB, MainloopElementAct, MainloopElementWeight>;
  ```

No `internal_cutlass_kernels` directory, no precompiled archive. The
`USING_INTERNAL_CUTLASS` macro is never defined in flashinfer's JIT build
path.

**P1 tactic selection in Python** (`capture_bf16_gemm1.py:35-50`):

```python
EXPECTED_GEMM1_TACTICS = [
    ("CtaShape128x128x128B_Cluster1x1x1", False),
    ("CtaShape128x128x64B_Cluster1x1x1",  False),  # <-- P1
    ...
]
P1_GEMM1_TACTIC_ID = 1
```

Tactic 1 is the `CtaShape128x128x64B` + `SwapAB=false` entry. The capture
script passes `profile_ids=[tactic_id, gemm2_tactic_id]` explicitly,
bypassing AutoTuner, so the P1 reference is deterministic and uses
`SwapAB=false`.

**Conclusion:** with flashinfer's vendored public-variant kernel and
`SwapAB=false`, the dispatch resolves to `SwappedMainloopElementA =
MainloopElementAct = activations` and `SwappedMainloopElementB =
MainloopElementWeight = weights`. **The reference
`bf16_gemm1_tactic1.bin` was produced by a kernel where activations are in
the A slot and weights are in the B slot.** The current state of
`ComputeNanoP1AccumTile` (`input_packed` → A smem, `weight_packed` → B
smem) matches this. **The baseline (HEAD) was wrong; the prior session's
A↔B swap was in the right direction.** The `do_shuffle` row-permutation
inverse on the A path is also correctly placed — the shuffle is applied to
activation tokens during `nvfp4_quantize`, so its inverse belongs on the
activation (A) source.

#### Detours (for the record)

- **Detour 1: NOTES.md was cited via the TRT-LLM source path**, so the
  first investigation walked the TRT-LLM cutlass_kernels tree at
  `third_party/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/`.
  That source is structurally identical to the flashinfer-vendored copy
  and reached the right A/B binding, but the citation path was in a
  location that does not actually run on Phase 3. The investigation was
  correct by accident.

- **Detour 2: `nm -D --defined-only libtensorrt_llm.so | c++filt`** showed
  8 hits for `tensorrt_llm::_v1::MoeGemmRunner<__nv_fp4_e2m1, __nv_fp4_e2m1,
  __nv_bfloat16, __nv_bfloat16>::*` and 0 hits for
  `tensorrt_llm::_v1::kernels::cutlass_kernels::MoeGemmRunner<...>`. The
  non-nested namespace matches the `internal_cutlass_kernels` variant.
  This triggered a (temporary) retraction of the operand-role conclusion,
  because the internal variant's header at
  `third_party/TensorRT-LLM/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/include/moe_gemm_kernels.h:84-94`
  has `LayoutA = TransposeLayoutTag<RowMajor> = ColumnMajor` with the
  comment "Layout for A and B is transposed and then swapped in the
  implementation. This uses B^T · A^T = (A · B)^T to get a better layout
  for the GEMM", which is the *opposite* convention from the public
  variant. The retraction was wrong, because **`libtensorrt_llm.so` is not
  what produces the Phase 3 reference dump**: flashinfer's JIT module is.
  The internal_cutlass_kernels variant exists in the TRT-LLM tree and runs
  under the pytorch/trtllm backend, but is unreachable from this project's
  reference-capture path.

#### If someone later wants to verify all of this the direct way

- Disassemble the flashinfer-vendored source built by the JIT: flashinfer
  writes compiled CUs under `~/.cache/flashinfer/` (or equivalent). The
  relevant symbol mangled name for the SM120 P1 FP4 kernel should be
  findable there, not in `libtensorrt_llm.so`.
- Or run an oracle A/B: build `ComputeNanoP1AccumTile` with `input_packed →
  A smem / weight_packed → B smem` vs the inverse, and compare Phase 3
  mismatch counts against `bf16_gemm1_tactic1.bin`. The source analysis
  above already says which orientation to keep, so this should not be
  necessary unless a subsequent bug is suspected to have hidden a sign
  flip somewhere.

## Gaps in the hypothesis list (2026-04-13 re-audit)

After walking through the "two remaining open items" (`fp4_shift`
orientation, source-side staging vs smem swizzle) and noticing that the
list was an artifact of the prior session's convergence trajectory rather
than an exhaustive enumeration, here are additional bug classes that are
consistent with Phase 1 (uniform 0x22 fill) passing and Phase 3 (real
Nemotron weights/activations at K=2688) failing, and which were **not**
ruled out by any of the 2026-04-13 probes. The list is deliberately
broader than the prior "two open" framing.

Treat these as co-equal candidates with `fp4_shift` and staging-vs-swizzle
until each is independently probed.

### G1. Reference-vs-test M-indexing order mismatch

The runtime kernel applies an inverse `srcToDstBlk32RowMap` at
`fused_moe_prefill.cu:12331-12338` when staging `input_packed` into A
smem. Its output is therefore in **logical token order**. The reference
dump `bf16_gemm1_tactic1.bin` was captured from flashinfer running on
`input_fp4_permuted.bin` (shuffled data written to disk by
`capture_bf16_gemm1.py:229-234`). **Whether flashinfer's kernel applies
its own inverse shuffle before the gemm, or computes on the shuffled
data natively and produces output in shuffled-M order, is not verified.**
If it's the latter, every runtime position `r` is being compared against
reference position `srcToDst[r]`, and matches occur only at the
permutation's fixed points within each 32-row block. The fixed points of
`(r%4)*8 + r/4` are `r ∈ {0, 31}` per 32-block — a 2/32 ≈ 6.25% per-axis
match rate, which multiplied across M×N gives 6.25% × 100% = 6.25%
overall (close to the observed 6.4%). Phase 1 is invariant to this
because uniform data is permutation-invariant.

Cheapest falsification: compare runtime output row 0 of one M-tile
against reference rows {0, 8, 16, 24} (the first few destinations of
`srcToDst` applied to logical 0..3). If runtime row 0 matches reference
row 0, the indexing is aligned. If it matches reference row 8 / 16 / 24,
there is a single-direction inverse-shuffle mismatch. If it matches none,
the bug is elsewhere.

### G2. Nibble high/low ordering within packed FP4 byte

FP4 packs two elements per uint8. The runtime's staging loop at
`fused_moe_prefill.cu:12340-12347` writes the whole byte via
`swizzled_a_bytes[stage0_A(row, byte_index*2) / 2] = value`, which picks
a specific convention for which element index (`2*byte_index` or
`2*byte_index+1`) lives in the low nibble vs the high nibble. If this
convention disagrees with what `flashinfer.nvfp4_quantize` produces, the
kernel reads every element with its K-neighbor's value. Phase 1's 0x22
fill is **exactly invariant** to nibble high/low swap (both nibbles are
2), so Phase 1 cannot detect this. The related `fp4_shift_A` /
`fp4_shift_B` functions also care about nibble order.

Cheapest falsification: trace the on-disk byte layout for one known
element (print the raw byte of `input_fp4[0][0]` vs the source tensor
`hs[0][0]`, `hs[0][1]` from `inputs.pt`). A one-line test script.

### G3. `NanoP1PartitionScaleA` / `NanoP1PartitionScaleB` per-K-group assignment

These are hand-rolled helpers at `fused_moe_prefill.cu:12284-12285` that
parallel the CUTLASS `sm120_blockscaled_mma_tma.hpp` scale-fragment
partition logic *without* instantiating any forbidden runtime
`Collective*` surface. Step 4a's probe measured only their *sizes*
(`(64, 2, 2)` for SFA, `(64, 8, 2)` for SFB) against the reverted
reference; it did not verify that each thread's `tCrSFA[k_group]` /
`tCrSFB[k_group]` entry corresponds to the same K-group that the CUTLASS
reference would pair it with. If the per-K-group assignment is
permutation-shifted, scales are applied to the wrong K ranges and every
Phase 3 output is wrong. Phase 1 cannot detect this because uniform-data
output is invariant to which scale gets applied to which K-group (they
are all the same unit scale).

Cheapest falsification: add a sentinel-style check where each K-group
has a unique scale value (e.g., `scale[k] = k + 1`) and both A and B are
uniform non-zero; the expected output is then a known function of the
scale pattern.

### G4. `g1_alphas` application timing

The runtime multiplies by `alpha = g1_alphas[0]` in the store path
(`fused_moe_prefill.cu:5293, 12596, 12634`). Flashinfer's SM120 P1 kernel
may fold `alpha` into the block-scaled scale factors instead, applying
it pre-GEMM. If alpha is applied in the wrong place, the runtime output
differs from the reference by a constant factor — but only for
non-uniform inputs. Phase 1 with `g1_alphas = 1.0` (the capture default
per `capture_bf16_gemm1.py:187`) would not detect this.

Cheapest falsification: set runtime alpha to 1.0 regardless of input and
check whether the reference `g1_alphas[0]` is numerically close to 1.0.
If yes, this isn't the bug. If no, look at where flashinfer applies it.

### G5. `MakePackedUnitScaleWord()` sentinel value

Trailing scale padding (for K ranges outside `available_blocks`) is
filled with `unit_scale_byte = LoadScaleByte(unit_scale_word, 0)` at
`fused_moe_prefill.cu:12352`. **Likely moot for Phase 3**: Nano K=2688 is
exactly 21 × 128, so there is no K-tail padding in this test, and this
code path should not fire. But the same sentinel is used in the
"row_valid=false" fallback (line 12406 etc.) which *does* fire if any
thread overshoots `kTileM` or `kTileN`. For 128 tokens and kTileM=128,
`valid_rows == 128` always and there is no overshoot. So this candidate
is probably inert, but confirming that requires reading the exact loop
bounds.

### G6. The pattern observation itself

Finding #3 of the "Empirical findings" section above says "M % 8 ∈ {0,
2} AND N % 8 ∈ {0, 2}" — a 25%-per-axis / 6.25%-overall match rate. I
was trusting this as a clean empirical observation, but the prior
session's causal reasoning was demonstrably wrong 16 times in a row, so
their *reading* of the pattern also deserves fresh verification. If the
pattern is actually `M % 32 ∈ {0, 31}` (i.e., the `srcToDst` fixed-point
footprint), the hypothesis space shifts dramatically toward G1. If it's
something else entirely, that's a new signal.

Cheapest falsification: extend the Phase 3 test's 8×8 debug dump to
16×16 or 32×32, plus a per-axis histogram of match positions over
`M % 32` and `N % 32`. One test file edit + one Phase 3 run. This is
also a pre-requisite for most of the other candidates above: without
knowing the actual pattern we are hunting, we can't tell what a "large
count change" means in the cheapest-falsification paths.

### Summary table

| ID  | Candidate                                         | Est. cost   | Info if fired      | Info if not       |
| --- | ------------------------------------------------- | ----------- | ------------------ | ----------------- |
| G1  | Reference/runtime M-indexing order mismatch       | ~2 min      | **Definitive**     | Narrows strongly  |
| G2  | Nibble high/low ordering                          | ~5 min      | Binary             | Binary            |
| G3  | Per-K-group scale assignment                      | ~30 min     | Localized          | Localized         |
| G4  | `g1_alphas` application timing                    | ~5 min      | Constant factor    | Binary            |
| G5  | `unit_scale_byte` sentinel                        | ~2 min      | Probably inert     | Confirms inert    |
| G6  | Pattern re-verification                           | ~10 min     | **Foundational**   | **Foundational**  |

**G6 and G1 together are cheapest and highest-value**, and they can be
done in a single Phase 3 run: extend the test's diagnostic dump, run
once, look at the pattern AND compare runtime row 0 against reference
rows {0, 8, 16, 24}. ~15 minutes total.

## Status of probe work and next step

**2026-04-13 probes: all done.** The compile-time `NanoP1ShowType<...>`
probe block was inserted into `fused_moe_prefill.cu` and built three
times:

1. Current working-tree state (57 lines of prior-session edits in place)
   — 7 layouts: `tCsA`, `tCsB`, `tCrA`, `tCrB`, `tCrA_cv`, `tCrB_cv`,
   `NanoP1AccumLayout`.
2. Same state, probe extended with `partition_C(make_identity_tensor(
   (128, 128)))` — 8 layouts (added `partC`).
3. Clean HEAD baseline (prior-session edits `git stash`-ed away) with the
   same 8-layout probe.

The two 8-layout stderrs (2 and 3) are **byte-for-byte identical**
(md5 `52f31c2dd0473dafa6777300dc73b5ec`), which proved that the prior
session's edits do not move any CUTE layout. Decoded layouts and the full
baseline-vs-current comparison live in `nano_p1_phase3_layouts_2026-04-13.md`.
The raw archive of run 2 is at
`nano_p1_phase3_probe_2026-04-13.stderr.txt` (59 lines, 8 errors).

The probe block has been fully reverted from
`runtime/src/backend/fused_moe_prefill.cu`. After revert, `git diff --stat`
reports `57 +/22 -` — the same pre-existing working-tree delta the file
had before this investigation started, so no accidental residue was left
behind. `make nemotron_runtime_backend` succeeds cleanly.

**Next investigative step:** **do NOT** pick up one of the two
"fp4_shift" / "staging-vs-swizzle" candidates in isolation. The
2026-04-13 re-audit (see "Gaps in the hypothesis list" above) identified
six additional candidate classes that are equally plausible and mostly
cheaper to falsify. The right next step is the combined **G6 + G1 probe**:

1. Extend the Phase 3 debug dump in
   `testing/backend/nano_p1_mainloop_oracle_test.cpp:1052-1077` from 8×8
   to 32×32, plus a per-axis histogram of match positions over `M % 32`
   and `N % 32`.
2. Add a cross-reference check: runtime output row 0 (first M of first
   tile) vs reference rows {0, 8, 16, 24} — the first four destinations
   of the `srcToDstBlk32RowMap` permutation applied to logical rows
   {0, 1, 2, 3}.
3. Run Phase 3 **once** (not in a loop), capture the output, and read
   both diagnostics.

This single Phase 3 run answers:

- **G6:** what the actual mismatch pattern is (re-verifies or falsifies
  finding #3's "M % 8 ∈ {0, 2}" claim).
- **G1:** whether there is a `srcToDstBlk32RowMap` direction mismatch
  between the runtime's inverse-shuffle and flashinfer's expectation.

Depending on what the run shows, the next step branches:

- If G6 shows pattern = `M % 32 ∈ {0, 31}` and G1 shows row-0 = ref-row-0
  → the prior session's pattern reading was wrong and this is almost
  certainly G1.
- If G6 confirms pattern = `M % 8 ∈ {0, 2}` and G1 shows row-0 matches
  ref-row-0 → G1 is not the bug; fall back to the prior-session's two
  candidates (`fp4_shift`, staging-vs-swizzle) or drop into G2/G3/G4.
- If G6 shows an entirely different pattern → re-read the summary table
  and pick whichever candidate the new pattern most directly suggests.

Do **not** run Phase 3 in a `/loop` with DIAG printf bisection. That is
the exact pattern the prior session used for 8.5 hours without
converging. The G6 + G1 probe is a **single** diagnostic-enrichment run,
not a bisection. If the first run doesn't localize, go read source code
(or do one of the one-line single-variable toggles like G2, G4) — don't
re-run Phase 3 repeatedly with different kernel edits.

## Pointers

- Plan in flight: `proj-2026-04-12-1022/PLAN.md` (step 5 part 1)
- Step 4a probe values (sizes only): `proj-2026-04-12-1022/probes/step_4a_probe_values.txt`
- Reference dump: `proj-2026-04-12-1022/trtllm_reference/golden_nano_k2688/bf16_gemm1_tactic1.bin`
- TRT-LLM convergence note: `proj-2026-04-12-1022/trtllm_reference/NOTES.md` §5
- Methodology: `CLAUDE.md` "Probe-then-decide" §1 (compile-time layout probes)
