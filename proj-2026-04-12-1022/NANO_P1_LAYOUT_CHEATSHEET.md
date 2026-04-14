# NanoP1 Layout Cheat Sheet

Empirically-verified layout facts for `NanoP1TiledMma` and `ComputeNanoP1AccumTile`, salvaged from Claude Code session `86e37198` (2026-04-13, 8.5h / 16 compactions / 0 commits attempting to fix `nano_p1_mainloop_oracle_test` Phase 3). These are the ground-truth findings — not the hypotheses that flip-flopped during the session.

**Source of truth:** DIAG printf output from the live kernel (threads 0, 1, 2) and compile-time `ShowInt<>` probes on CUTE type instantiations. Every claim below is traceable to either a DIAG line in the session transcript (`~/.claude/projects/-home-khkramer-src-nemotron-inference/86e37198-c2d2-460f-8096-61c13808afb6.jsonl`) or a compile-time probe output captured during the session.

**Scope:** facts only. Hypotheses that were tried and didn't work are deliberately omitted. Use this as the baseline for the next debugging round; do not re-derive what's already here.

## Type bundle

```cpp
// runtime/src/backend/fused_moe_prefill.cu (pre-cleanup line numbers; valid until Step 0 of proj-2026-04-13-1936)
using NanoP1MmaOp         = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<...>;
using NanoP1MmaTileShape  = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<128>>;
using NanoP1AtomLayoutMNK = cute::Layout<cute::Shape<cute::_4, cute::_2, cute::_1>>;
using NanoP1ValLayoutMNK  = cute::Tile<cute::Int<128>, TracedP5PermTileN, cute::Int<64>>;
using NanoP1TiledMma      = cute::TiledMMA<NanoP1MmaAtom, NanoP1AtomLayoutMNK, NanoP1ValLayoutMNK>;
```

- 256 threads per CTA (32 atom threads × 4×2×1 atom layout).
- `cute::size<2>(tCrA) = kMmaKBlocks = 2` (128 K / 64 K-per-atom).
- Per-thread accumulator shape = `(4, 2, 8)` = 64 elements. Dimensions are labeled `(reg, nf, mf)` in code — **but see the role-inversion note below; the labels are inverted relative to the coordinates they actually control.**

## `TracedP5PermTileN`

```cpp
using TracedP5PermTileN =
    cute::Layout<cute::Shape<cute::_8, cute::_2, cute::_2>,
                 cute::Stride<cute::_1, cute::_16, cute::_8>>;
```

Flat-index mapping for `i ∈ [0..31]`:

| Input range      | Output       |
|------------------|--------------|
| `i ∈ [0..7]`     | `i` (identity)    |
| `i ∈ [8..15]`    | `i + 8`      |
| `i ∈ [16..23]`   | `i − 8`      |
| `i ∈ [24..31]`   | `i` (identity)    |

Net effect: the pair `[8..15] ↔ [16..23]` is swapped inside each 32-wide N block; `[0..7]` and `[24..31]` are untouched.

## `SM80_16x8_Row` C-layout — empirically column-major

```cpp
// cutlass include/cute/atom/mma_traits_sm80.hpp
using SM80_16x8_Row = cute::Layout<
    cute::Shape<cute::Shape<cute::_4, cute::_8>, cute::Shape<cute::_2, cute::_2>>,
    cute::Stride<cute::Stride<cute::_32, cute::_1>, cute::Stride<cute::_16, cute::_8>>>;
```

DIAG-confirmed decoding (**column-major, M-fast**):

- flat index = `M + 16 * N`
- `M = flat % 16`, `N = flat / 16`

Thread 0 covers `(M=0, N=0), (M=0, N=1), (M=8, N=0), (M=8, N=1)` — two M values × two N values per thread, with M advancing faster than N along the flat index. This is the opposite of the initial row-major reading (see "Ruled out" below).

## Thread → (M, N) mapping for `NanoP1TiledMma`

Derived from DIAG printf covering threads 0, 1, 2 across all `(reg, nf, mf)` slots, then generalized via the 4×2×1 atom layout.

```cpp
// For thread t in [0, 256):
int atom_m  = (t % 128) / 32;   // {0, 1, 2, 3}
int atom_n  =  t / 128;         // {0, 1}
int t_local =  t % 32;          // {0..31}

// For each accumulator slot (reg ∈ [0..4), nf ∈ [0..2), mf ∈ [0..8)):
int M = atom_m * 16 + (t_local / 4) + (reg / 2) * 8 + nf * 64;
int N = (t_local % 4) * 2 + (reg % 2) + N_mf_table[mf];
```

### `N_mf_table`

For **`atom_n = 0`** (threads 0–127), DIAG-verified:

```
N_mf_table[mf=0..7] = { 0, 8, 32, 40, 64, 72, 96, 104 }
```

Pattern: pairs of `{base, base+8}` at base offsets `{0, 32, 64, 96}`.

For **`atom_n = 1`** (threads 128–255), predicted but **not** DIAG-verified in the killed session:

```
N_mf_table[mf=0..7] = { 16, 24, 48, 56, 80, 88, 112, 120 }
```

The prediction adds 16 to each base offset (atom_n contributes a 16-wide N shift per the atom layout). Verify with a second DIAG pass before trusting it.

## M/N role inversion (labels ≠ coordinates)

The `AccumLayout` dimensions are named `(reg, nf, mf)` in code, with `nf` meaning "N fragment" and `mf` meaning "M fragment". **These names are inverted relative to the coordinates they actually control.** DIAG proves:

- `nf` (labeled N) controls **M** — the `nf * 64` term in the M formula.
- `mf` (labeled M) controls **N** — the `N_mf_table[mf]` term in the N formula.

Cause: `TracedP5PermTileN` in `NanoP1ValLayoutMNK` permutes the N axis, and combined with the MMA atom's native ordering it ends up swapping the role of the two fragment dimensions. The names were chosen before this interaction was understood; they're misleading but renaming would be pure churn.

**When reading `ComputeNanoP1AccumTile`:** the `nf` loop iterator actually walks M, and the `mf` loop iterator actually walks N. Any diagnostic that prints "M=nf, N=mf" is lying — use the formulas above.

## Phase 3 mismatch signature

On `nano_p1_mainloop_oracle_test` Phase 3 (Nano K=2688 bucket: M=128, K=2688, N=1920), the kernel output matches `bf16_gemm1_tactic1.bin` **only** at positions where:

```
M % 8 ∈ {0, 2}   AND   N % 8 ∈ {0, 2}
```

That's a 6.25% match rate (4/64): 230,087 mismatches out of 245,760 elements, 93.6% miss. This specific pattern is the **fingerprint** of the bug — any correct fix should collapse this signature cleanly. A fix that just shifts it to a different pattern (e.g., `M % 4 ∈ {0}` or `N % 16 ∈ {0..7}`) is suspect and should be investigated for whether it's actually solving the root cause.

## Ruled out (probe-verified, do not re-investigate)

1. **`sm120_rr_smem_copy_selector_A/B` with `UseF8f6f4=true`.** Suspected that flipping the flag from `false` to `true` would change the fragment layout and fix the mismatch. Ruled out by source inspection: both `SM75_U32x4_LDSM_N` (returned when `UseF8f6f4=false`) and `SM100_SU4_DU8x16_x4_LDSM_N` (returned when `UseF8f6f4=true`) have identical `SrcLayout` and `DstLayout`. The flag doesn't change the per-thread byte mapping for this atom.

2. **`SM80_16x8_Row` interpreted as row-major (M-slow).** DIAG output for threads 0–2 directly contradicts the row-major interpretation. Thread 0's outputs at `(nf=0, mf=0)` across `reg=0..3` land on M values that advance faster than N — M-fast, not M-slow. See "empirically column-major" above.

## Still open for the next session

- **Exact `smem_pos → accumulator_slot` permutation** for the `s2r_copy_A` / `s2r_copy_B` copies after `partition_S` in `ComputeNanoP1AccumTile`. The kernel's smem fill loops assume one mapping; the `LDSM.N` s2r copy atom and the `Layout_K_SW64_Atom` swizzle together produce a different one. **Cheapest falsifier:** compile-time `ShowInt<>` probes on `tCsA` and `tCsB` after `partition_S` — extract the exact layouts, cross-reference against the thread→(M,N) mapping above, and the mismatch should pinpoint which axis (M-side, N-side, or both) needs a different smem load pattern. This is the move the killed session identified as next-needed and then didn't make.

- **Whether the M-side and N-side mismatch signatures share a mechanism.** `M % 8 ∈ {0, 2}` and `N % 8 ∈ {0, 2}` both hit 2-out-of-8 — same pattern on both axes. Could be one bug or two. A targeted DIAG covering threads from multiple `atom_m` groups (not just 0–2 in `atom_m = 0`) should disambiguate.

## Where the DIAG infrastructure lives

`StoreNanoP1CFragmentsRowMajor` in `runtime/src/backend/fused_moe_prefill.cu` (pre-cleanup, around line 5146). Printf guarded by `kDiagThread0 || kDiagThread1 || kDiagThread2`. Output format:

```
DIAG tid=%d reg=%d nf=%d mf=%d M=%d N=%d raw=%.2f val=%.6f
```

Post-split (after Step 1 of `proj-2026-04-13-1936`), this will live in `fused_moe_prefill/nano_p1_epilogue.cuh`. The printf is the only diagnostic infrastructure the killed session actually built and verified; preserve it until the bug is found.
