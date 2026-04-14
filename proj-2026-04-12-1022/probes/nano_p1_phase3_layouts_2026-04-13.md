# NanoP1 Phase 3 — compile-time layout probe values (2026-04-13)

Decoded from the `NanoP1ShowType<...>` incomplete-type diagnostics emitted by
nvcc when fused_moe_prefill.cu was compiled with `-DNANO_P1_PROBE_LAYOUTS=1`.
Raw stderr archived alongside this file as
`nano_p1_phase3_probe_2026-04-13.stderr.txt`. The probe block has been
reverted; this file is the durable record.

For each probed object the format is `Layout = (shape) : (stride)` in CUTE
notation, with shape and stride aligned mode-by-mode. The
`std::conditional_t<true, X, ...>` SFINAE wrappers in the raw diagnostic are
unwrapped to their first arm (the actual chosen type).

## NanoP1AccumLayout — `partition_fragment_C(NanoP1TiledMma, (128,128)).layout()`

```
((2,2), 2, (2,4)) : ((1,2), 4, (8,16))
```

| Mode | Sub-mode | Shape | Stride | Notes |
| --- | --- | --- | --- | --- |
| 0 | 0 | 2 | 1 | per-thread atom-val component |
| 0 | 1 | 2 | 2 | per-thread atom-val component |
| 1 | — | 2 | 4 | iteration count along **CUTE mode-1** |
| 2 | 0 | 2 | 8 | iteration count piece along **CUTE mode-2** |
| 2 | 1 | 4 | 16 | iteration count piece along **CUTE mode-2** |

- Total `cosize = 64`, matches `static_assert(kNanoP1AccumCoordCount == 64)`.
- `size<0>` = 4, `size<1>` = 2, `size<2>` = 8.

## tCrA — `partition_fragment_A(NanoP1TiledMma, sA_stage0).layout()`

```
((8,2,2), 2, 2) : ((1,8,16), 64, 32)
```

`(Val, M_iter, K_iter) = (32, 2, 2)`.

## tCrB — `partition_fragment_B(NanoP1TiledMma, sB_stage0).layout()`

```
((8,2), (2,4), 2) : ((1,8), (32,64), 16)
```

`(Val, N_iter, K_iter) = (16, (2,4)=8, 2)`.

The N-iteration mode of `tCrB` is decomposed as a `(2,4)` tuple, exactly
matching the inner structure of `NanoP1AccumLayout` mode-2 `(2,4)`.

## tCrA_cv — `s2r_thr_A.retile_D(tCrA).layout()`

```
((32,2), 1, 2) : ((1,64), 0, 32)
```

The 32 in the inner shape is the per-thread elements consumed by the s2r copy
atom in one issue.

## tCrB_cv — `s2r_thr_B.retile_D(tCrB).layout()`

```
(((16,2),1), 4, 2) : (((1,32),0), 64, 16)
```

## tCsA — `s2r_thr_A.partition_S(sA_full).layout()` (sA_full = full 4-stage)

```
((32,2), 1, 2, (1,4)) : ((1,8192), 0, 64, (0,16384))
```

- Mode 0 `(32,2):(1,8192)` — per-issue copy values plus M-strip jump 8192
- Mode 1 `1:0` — degenerate (no extra inner M iterations beyond what the copy atom owns at this level)
- Mode 2 `2:64` — K iterations × stride 64 (matches `kTileK / 2` byte stride)
- Mode 3 `(1,4):(0,16384)` — pipeline stages × stage stride 16384 (= 65536 / 4)

## tCsB — `s2r_thr_B.partition_S(sB_full).layout()` (sB_full = full 4-stage)

```
((32,1), 4, 2, (1,4)) : ((1,0), 4096, 64, (0,16384))
```

- Mode 0 `(32,1):(1,0)`
- Mode 1 `4:4096` — note the **4** here is a non-trivial iteration count along
  **mode 1** of `tCsB`, in contrast to `tCsA` which has `1:0` at mode 1.
- Mode 2 `2:64` — K iterations
- Mode 3 `(1,4):(0,16384)` — pipeline stages

## Raw diagnostic

See `nano_p1_phase3_probe_2026-04-13.stderr.txt` (59 lines, 8 errors —
one per `ShowType<...>` declaration; this is the post-partC rerun, which
superseded the initial 7-layout archive).

## What this proves about earlier hypotheses

1. **The layouts are all fully static and well-defined.** The `partition_*` /
   `retile_D` calls all produce concrete `cute::Layout<>` types — no runtime
   stride, no integer holes. So whatever the bug is, it is not "the layouts
   are different at runtime than at compile time".

2. **`NanoP1AccumLayout` mode 2 `(2,4)` matches `tCrB` N-iteration mode
   `(2,4)`.** The accumulator's mode-2 sub-tuple is structurally identical
   to `tCrB`'s N-iteration sub-tuple. By the standard CUTE
   `partition_fragment_C` contract, that makes mode-2 the **N** axis of the
   accumulator and mode-1 (`size = 2`) the **M** axis. The source code at
   `runtime/src/backend/fused_moe_prefill.cu:480-481` names them in the
   reverse order:
   ```cpp
   constexpr int kNanoP1NFragments = cute::size<1>(NanoP1AccumLayout{}); // = 2
   constexpr int kNanoP1MFragments = cute::size<2>(NanoP1AccumLayout{}); // = 8
   ```
   The variable names "M" and "N" are swapped relative to CUTE's convention.

3. **However, the swap is *internally consistent* in the store loop.** In
   `StoreNanoP1CFragmentsRowMajor`
   (`runtime/src/backend/fused_moe_prefill.cu:5240`) the loop order is
   `for reg / for n_fragment / for m_fragment` and indexes both `part_c` and
   `accum_tensor` as `(reg, n_fragment, m_fragment)`. Both tensors share the
   same shape and the same MMA partitioning convention, so reading
   `accum_tensor` and resolving `part_c` at the same `(reg, n_fragment,
   m_fragment)` index is consistent. The misleading variable names by
   themselves are not enough to corrupt the writeback — they are an
   ergonomics issue, not a data issue.

   **This contradicts the 00:47 "M/N role inversion" hypothesis as stated.**
   The 00:47 hypothesis cannot be the root cause on this code path alone —
   the swap exists but cancels out within the store loop.

4. **Open question: where does the actual 230,087-mismatch divergence come
   from?** The probe has not yet answered this. Candidates that the probe
   data still leaves on the table, in the order they should be cheapest to
   falsify next:
   - **`partition_C` vs `partition_fragment_C` stride mismatch.** The store
     loop uses `partition_C(dense_c)` for the coordinate map but
     `accum_tensor` is built on `partition_fragment_C`-derived
     `NanoP1AccumLayout`. The shape match is asserted at lines 5261-5263 but
     the *iteration order* the gemm uses to fill the accumulator may not
     follow the same `(reg, m_iter, n_iter)` traversal as the store loop's
     manual loops. Probe by also dumping `partition_C(dense_c).layout()` and
     comparing its strides mode-by-mode against `NanoP1AccumLayout`.
   - **`fp4_shift_A` / `fp4_shift_B` orientation.** The shifts at lines
     12428-12429 assume natural K-sequential FP4 with row-permuted M. With
     `NanoP1ValLayoutMNK = Tile<128, TracedP5PermTileN, 64>` introducing a
     non-trivial N permutation, the B shift may be using the wrong
     orientation for this MMA atom. Check by toggling `fp4_shift_B` off /
     swapping its operand convention while everything else is held constant.
   - **Source-side staging vs. CUTE-expected smem layout.** The hand-rolled
     row loops at 12329-12347 (A) and 12372-12386 (B) write into
     `swizzled_a_bytes` / `swizzled_b_bytes` indexed by `stage0_A(row,
     byte_index*2)` and `stage0_B(row, byte_index*2)`. That's the **logical
     M-major / N-major view** of the smem layout. `tCsA`'s stride pattern
     above (mode 0 = `(1, 8192)`) shows the MMA copy's view of the same
     smem; the host-side staging would be wrong if the MMA's mode-0 stride
     8192 doesn't correspond to the same M-stripe boundary the staging loop
     is walking. Probe by writing a known sentinel pattern into
     `swizzled_a_bytes`, dumping the `tCrA_cv` register state for a
     specific lane after `cute::copy`, and comparing against a host model
     of "what a thread should see if the staging is correct".

## Follow-up: baseline vs. current diff and `partC`

After the initial probe ran on the in-flight (working-tree) state of
`fused_moe_prefill.cu`, the same probe block — extended with a new
`partition_C(make_identity_tensor((128,128)))` probe — was re-run twice:

1. Once against the working-tree state (with the prior session's unstaged
   edits in place).
2. Once against the clean HEAD baseline of `runtime/src/backend/fused_moe_prefill.cu`
   (achieved by `git stash`-ing the prior edits before the probe build).

Both stderrs (8 layouts each, including the new partC) are **byte-for-byte
identical** (md5 `52f31c2dd0473dafa6777300dc73b5ec`). That answers two
questions at once:

- **None of the prior session's edits move a single CUTE layout.** The A↔B
  operand-role swap in the staging loops, the three `CoordGet0`↔`CoordGet1`
  swaps in the store paths, the `do_shuffle` row-permutation patch, and the
  `kTileM`/`kTileN` loop-bound shuffle are all *behavioral* changes — they
  alter what data lands in which smem buffer and which coord component is
  read as a row vs column, but they do not move any layout. The CUTE machinery
  sees the exact same partitioning either way. The bug is not in layouts.

- **`partition_C(identity (128,128))` and `NanoP1AccumLayout` are
  structurally consistent.**

### partC layout (`partition_C(make_identity_tensor((128,128)))`)

Decoded from the `ScaledBasis<C<k>, b>` types in the diagnostic:

```
shape  =  ((2,2),       2,        (2,4))
stride =  ((N:1, M:8),  M:64,     (N:8, N:32))
```

(`N:k` = stride k along basis-1, the N axis of the dense (M,N) tensor; `M:k`
= stride k along basis-0, the M axis. Cosize spans M=0..127 and N=0..127.)

### Comparing with `NanoP1AccumLayout`

```
NanoP1AccumLayout shape   = ((2,2), 2, (2,4))      [identical]
NanoP1AccumLayout stride  = ((1, 2), 4, (8, 16))   [pure register slots]
```

- **Shapes match mode-by-mode**, as the existing static asserts at
  `fused_moe_prefill.cu:5261-5263` already require.
- **partC mode 0 sub-shape `(2,2)`** has `stride = (N:1, M:8)`. So within a
  single thread's 4-register accumulator value group, sub-position `(0)`
  walks N by 1 while sub-position `(1)` walks M by 8. The accumulator's
  matching mode-0 stride is `(1, 2)` — adjacent register slots — meaning
  reg slot 0 holds value at coord `(M+0, N+0)`, reg slot 1 at `(M+0, N+1)`,
  reg slot 2 at `(M+8, N+0)`, reg slot 3 at `(M+8, N+1)`. This is the
  standard `SM80_16x8_F32F16F16F32_TN` per-thread accumulator quadruple.
- **partC mode 1** is a pure `M:64` step: iterating mode 1 of `part_c`
  walks across M (token rows) in steps of 64 rows. Both `part_c` and
  `accum_tensor` have `size<1> = 2` → two M-iterations per CTA tile.
- **partC mode 2** sub-shape `(2,4)` has `stride = (N:8, N:32)`: iterating
  mode 2 walks across N (output columns) in 8 steps that span N=0..120.
  Both tensors have `size<2> = 8`.

**This rules out hypothesis #1 from the original layouts file.** `partC` and
`NanoP1AccumLayout` walk the same MMA iteration axes in the same order. The
`StoreNanoP1CFragmentsRowMajor` triple-nested loop reads the right
accumulator slot for the coord it resolves at every `(reg, n_fragment,
m_fragment)` index. The misleading `n_fragment` / `m_fragment` variable
names cancel out cleanly; mode 1 is M-iter and mode 2 is N-iter for both
tensors, no bug there.

### Implications for the remaining hypotheses

The probe data has now eliminated:

- (originally ruled out) `UseF8f6f4 = false`, `CLayout` row-major
- (now also ruled out) **layouts changed by prior session's edits** — they
  didn't
- (now also ruled out) **`partC` vs `accum_tensor` stride mismatch**
- (later, via flashinfer source read) **operand-role intent** —
  definitively settled; the current state (activations → A smem, weights →
  B smem) matches the public-variant flashinfer-vendored kernel that
  produces the Phase 3 reference dump. See `nano_p1_phase3_salvage_2026-04-13.md`
  → "Status of open questions after the 2026-04-13 probe runs" → "Settled:
  operand-role intent" for the full chain of evidence and the two detours
  (TRT-LLM source path, `libtensorrt_llm.so` `nm` inspection) that did not
  change the conclusion.

What's still on the table:

- **`fp4_shift_A` / `fp4_shift_B` orientation** under the non-trivial
  `TracedP5PermTileN` N-permutation in `NanoP1ValLayoutMNK`. The shifts at
  `fused_moe_prefill.cu:12428-12429` (current line numbers may differ due to
  the prior session's edits) target the natural K-sequential FP4 nibble
  order. With the N-permutation introducing extra structure on the B side,
  the per-K-block shift may be reading the wrong nibble pair.

- **Source-side staging vs the smem swizzle the MMA copy expects.** The
  hand-rolled row loops at 12329-12347 (A) and 12372-12386 (B) — line
  numbers in the prior-session-edited tree — write into the swizzled smem
  bytes through `stage0_A(row, byte_index*2)` and `stage0_B(row,
  byte_index*2)`. The CUTE layouts (`tCsA` stride mode-0 `(1, 8192)`,
  `tCsB` stride mode-1 `4096`) describe the MMA copy's view of smem; the
  staging loop is correct only if the byte stream it emits matches what
  the s2r copy expects given those strides. Validate via the "verify the
  test reference before trusting a bisection" pattern from CLAUDE.md:
  write a known sentinel into the staged smem, dump per-lane `tCrA_cv` /
  `tCrB_cv` register state after `cute::copy`, and compare against a host
  model. The dump-oracle template at
  `testing/backend/p13_generic_direct_stage_oracle_test.cpp` is the right
  starting point.

## Methodology check

The cost of getting these eight layouts (the initial seven + `partC`
added on the follow-up run): **three nvcc compiles (~30 s each) + ~20
minutes of work**. The cost of *not* having them: 8.5 hours of /loop
oscillation between 230 k and 243 k mismatches. CLAUDE.md "Probe-then-decide"
§1 is the right tool; the previous session simply did not use it.
