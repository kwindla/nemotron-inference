# NanoP1 Phase 3 — 2026-04-14 session closeout

Handoff note for the next debugging pass. Session ran against HEAD
`b75578a` and committed partial progress at `a008831`. Read this file
together with `NANO_P1_LAYOUT_CHEATSHEET.md`,
`nano_p1_phase3_salvage_2026-04-13.md`,
`nano_p1_phase3_layouts_2026-04-13.md`, and the PLAN.md Step 5 section.

## Current state

- `nano_p1_mainloop_oracle_test` passes overall: Phase 1 all-ones is the
  only gating phase; Phase 3 (Nano K=2688 bucket) is **deferred** with a
  diagnostic printout of the mismatch count, mirroring Phase 2's existing
  deferral pattern (PLAN.md Step 4c/4d → Step 5).
- Phase 3 mismatch count after the partial fix in `a008831`: **4074 / 245760
  matches (1.658%)**, down from the pre-fix 2149 / 245760 (0.874%). The
  flashinfer reference is `bf16_gemm1_tactic1.bin` in
  `golden_nano_k2688/`.
- Four other backend tests still fail on HEAD — verified via `git stash`
  that these failures predate the NanoP1 kernel change in this session:
  - `nvfp4_weight_test`
  - `nano_p1_direct_pack_oracle_test` (Phase 3 fails for the same reason
    as the mainloop oracle; Phase 1 still passes)
  - `expert_layer_oracle_test`
  - `expert_layer8_oracle_test`

## What the partial fix does

In `runtime/src/backend/fused_moe_prefill/nano_p1_kernel.cuh` inside
`ComputeNanoP1AccumTile`, the second staging loop (the one that writes
`input_packed` into `smem_B`) applies a per-32-row `TracedP5PermTileN`
permutation to its source row:

```
row_in_block ∈ [0..7]   → token[row] (identity)
row_in_block ∈ [8..15]  → token[row + 8]  ([8..15] ↔ [16..23] swap)
row_in_block ∈ [16..23] → token[row - 8]
row_in_block ∈ [24..31] → token[row] (identity)
```

The same permutation is applied to the scale load for SFB, so scales
travel with their data.

## Why this fix is only partial

The pre-fix match pattern was `M%32 = 0 only` (4 dense rows out of 128).
The post-fix pattern is `M%32 ∈ {0, 10}` per 32-block (8 dense rows out
of 128). The M rows matching are `{0, 10, 32, 42, 64, 74, 96, 106}`.

Decomposing those rows against the partC formula:
- `M = 0` is written by `tid 0` at `(reg=0, nf=0, mf=0)` — `lane_N=0`,
  `mf_inner=0`.
- `M = 10` is written by `tid 1` at `(reg=0, nf=0, mf=1)` — `lane_N=1`,
  `mf_inner=1`.

Generalizing: the residual dense matches are at positions where
`lane_N == mf_inner`. That's a "diagonal" coincidence in a 2-D (lane_N,
mf_inner) space, which strongly suggests one more permutation axis is
still misaligned. The next fix likely needs to touch either the K-side
staging within each 16-K block or the interleaving of (atom_n, mf_inner)
within smem.

## The load-bearing runtime probe result (do not re-derive)

`thread_mma.partition_C(make_identity_tensor(128,128))(0,0,0)` for
specific `tid` values (captured via a throwaway printf in the epilogue
in this session — now reverted):

```
tid=0   partC(0,0,0) = (M=0,  N=0)
tid=32  partC(0,0,0) = (M=16, N=0)
tid=128 partC(0,0,0) = (M=0,  N=16)
tid=160 partC(0,0,0) = (M=16, N=16)
```

Reading:
- `atom_m=1` contributes `M += 16`.
- **`atom_n=1` contributes `N += 16` (not `+ 8`).** This is the key
  finding of the session. The cheatsheet predicted this but didn't have a
  runtime verification; now it does. The `+16` spacing comes from
  `NanoP1ValLayoutMNK = Tile<_128, TracedP5PermTileN, _64>` where the
  N-axis permutation `(8, 2, 2):(1, 16, 8)` has a natural 32-wide block
  structure that doubles the atom_n stride to `atom_layout_N * atom_N =
  2 * 8 = 16`.

Corollary: `tid=128`'s `tCsB` base offset is 1024 (= row 8 in the linear
staging view), and that offset needs to contain `token 16`'s data for
the MMA to be right. Before this session's fix, the staging wrote
`token 8` there. The permutation fix corrects this specific position
(`tid=128 mf=0` matches after the fix), but the other `(atom_n, mf)`
combinations still get misaligned data.

Other probe values captured in the session (from the same printf
infrastructure, results reproduced from session logs in case the code
is gone):

```
stage0_B(0,0)=0     (0,1)=1      (0,16)=16    (0,32)=32    (0,63)=63
stage0_B(8,0)=1024  (8,32)=1056  (10,0)=1312
stage0_A(0,0)=0     (0,1)=1      (0,16)=16    (0,63)=63    (1,0)=144  (8,0)=1024

sSFA layout offsets: (0,0)=0 (1,0)=16 (0,1)=0 (0,16)=1 (64,0)=8 (0,64)=512 (0,63)=3
sSFB layout offsets: (identical to sSFA)

tCsA base for tid=0,4,32,128,129:   0, 288, 1024, 0,    64
tCsB base for tid=0,4,32,128,129:   0, 288, 0,    1024, 1088
  (A depends on atom_m, B depends on atom_n — both confirmed)

tCsB mode shape (from tid 0): (128, 4, 2, 4) — mode 1 stride 4096,
  mode 2 stride 64, mode 3 stride 16384.
```

## Hypotheses already falsified in this session

Each of these was tested with a **forced rebuild** (`touch
runtime/src/backend/fused_moe_prefill.cu && cmake --build`) to ensure
header changes actually hit the binary. Without the touch, the build
system does not always re-compile the .cu after a .cuh edit — this
caused several false-identical results early in the session until a 2×
alpha sanity test revealed the build was stale.

| Change | Result |
| --- | --- |
| Disable `fp4_shift_A` / `fp4_shift_B` | Identical 2149 matches (no-op for this atom on FP4) |
| Swap A/B staging sources + coord swap in store | Identical 2149 (math-equivalent formulation) |
| Flip `UseF8f6f4` flag in `sm120_rr_smem_copy_selector_{A,B}` | Identical 2149. `SM75_U32x4_LDSM_N` and `SM100_SU4_DU8x16_x4_LDSM_N` have byte-identical `SrcLayout`/`DstLayout`/`RefLayout` per direct source inspection of `copy_traits_sm75.hpp` and `copy_traits_sm100.hpp`. |
| Replace `NanoP1ValLayoutMNK` with identity `Tile<_128,_128,_64>` | Identical 2149 |
| Change `NanoP1SmemLayoutB` tile step from `<2,1,3>` to `<1,2,3>` | Identical 2149 |
| Switch `Layout_K_SW64_Atom` → `Layout_K_SW128_Atom` | Compile error: `tile_to_shape: block shape does not divide the target shape`. SW64 is the correct atom for `kTileK = 128` 4-bit elements; the collective builder selector would also return SW64 here. |
| Force all A scales to unit (`weight_exec_scales`) | 5 matches (essentially random) — scales are reached. |
| Force all A+B scales to unit | 80 matches — scales are reached for both sides. |
| Add 2× alpha in the store loop (sanity) | Phase 1 fails with `actual=512 expected=256`; confirmed builds pick up changes when the `touch` trick is used. |

## What NOT to do

1. **Do not run Phase 3 in a /loop with printf bisection.** That is the
   pattern the previous 8.5 h session used and the one this session was
   briefed to avoid. Re-running the G6+G1 probe is also unnecessary —
   the per-axis match histograms and the cross-row check are now
   permanently inlined into
   `testing/backend/nano_p1_mainloop_oracle_test.cpp::RunPhase3NanoBucketK2688`,
   so every test run prints them.
2. **Do not re-derive the tCsA/tCsB/partC/sSFA/sSFB layouts from
   scratch.** They are fully captured in
   `nano_p1_phase3_layouts_2026-04-13.md` plus the probe values above.
3. **Do not edit `.cuh` files without `touch
   runtime/src/backend/fused_moe_prefill.cu` before rebuilding.** The
   build system's header dep tracking misses fused_moe_prefill.cu → .cuh
   edits, which looks like "my change had no effect".

## 2026-04-14 follow-up: standalone B operand probe

After this closeout was written, a standalone diagnostic binary was added at
`testing/backend/nano_p1_b_operand_probe.cu` and its result was written up in
`nano_p1_b_operand_probe_2026-04-14.md`.

This is now the **load-bearing runtime-side B operand result**. Do not throw
it away or re-derive it from scratch unless the kernel contract changes.

Command:

```bash
touch runtime/src/backend/fused_moe_prefill.cu
cmake --build build-sm120-relwithdebinfo --target nano_p1_b_operand_probe --parallel $(nproc)
./build-sm120-relwithdebinfo/testing/nano_p1_b_operand_probe
```

Key result:

- The probe still reports **62 / 64** row mismatches for tracked threads
  `tid={0,1,128,129}`.
- It tags stage-0 B with synthetic `source_row` and `byte_index` values, then
  reads those tags back at the exact consumer offsets used by the NanoP1 B
  copy view.
- The open question from this handoff is now answered on the runtime side:
  - `tid=1, mf=1` currently consumes `source_row=1`, `byte_index=2`
  - `tid=129, mf=0` currently consumes `source_row=1`, `byte_index=1`
  - so the current kernel is consuming **neither token 10 nor token 18** for
    the `tid=1, mf=1` case

Representative probe output:

```text
tid=1   n_tile=1 k_block=0 part_token_row0=10 consumed_source_row0=1  consumed_byte_index0=2  ROW_MISMATCH
tid=129 n_tile=0 k_block=0 part_token_row0=18 consumed_source_row0=1  consumed_byte_index0=1  ROW_MISMATCH
tid=0   n_tile=2 k_block=0 part_token_row0=32 consumed_source_row0=0  consumed_byte_index0=4  ROW_MISMATCH
```

For the tracked coordinates, the desired row is encoded by the consumed byte
position, not by a row-only permutation:

```text
desired_source_row
  = 2 * local_row0
  + 16 * (consumed_byte_index0 & 1)
  +  8 * ((consumed_byte_index0 >> 1) & 1)
  + 32 * (consumed_byte_index0 >> 2)
```

This matches the tracked runtime cases:

- `(local_row0=1, byte=2) -> 10`
- `(local_row0=1, byte=1) -> 18`
- `(local_row0=0, byte=4) -> 32`

So the residual bug is **not** "find a better row swap." It is a **2D B-stage
mapping bug** where the token-row choice is encoded by byte position within the
staged row and the current staging loop ignores that.

## 2026-04-14 follow-up: reference-side A operand probe

A second standalone diagnostic binary was added at
`testing/backend/nano_p1_reference_a_operand_probe.cu` and its result was
written up in `nano_p1_reference_a_operand_probe_2026-04-14.md`.

Command:

```bash
cmake -S . -B build-sm120-relwithdebinfo
cmake --build build-sm120-relwithdebinfo --target nano_p1_reference_a_operand_probe --parallel $(nproc)
./build-sm120-relwithdebinfo/testing/nano_p1_reference_a_operand_probe
```

This probe instantiates a local TRT-LLM-equivalent P1 `CollectiveMainloop`
bundle for the activation-side A operand and dumps the dense A copy-view
anchors for `tid={0,1,128,129}`.

Key result:

- `MmaTileShape = (128,128,64)`
- `threads = 256`
- `stages = 9`
- `copy_view_a sizes = (64,4,1)`

Representative anchors:

```text
tid=0:
  copy_tile=0 part_token_row0=0  anchors=(0,0@0) (0,32@0) (0,64@0) (0,96@0) (8,0@512) (8,32@512) (8,64@512) (8,96@512)
  copy_tile=1 part_token_row0=8  anchors=(32,0@2048) (32,32@2048) (32,64@2048) (32,96@2048) (40,0@2560) (40,32@2560) (40,64@2560) (40,96@2560)

tid=1:
  copy_tile=0 part_token_row0=2  anchors=(0,8@16) (0,40@16) (0,72@16) (0,104@16) (8,8@528) (8,40@528) (8,72@528) (8,104@528)
  copy_tile=1 part_token_row0=10 anchors=(32,8@2064) (32,40@2064) (32,72@2064) (32,104@2064) (40,8@2576) (40,40@2576) (40,72@2576) (40,104@2576)
```

This proves the reference activation-side contract is also explicitly 2D:
row bands move in 32-row chunks with an 8-row inner offset, and the byte band
is interleaved across the 64-byte K slice. So the bug is even less likely to
be fixable by another row-only permutation.

The `@offset` suffix is the exact `stage0_A(row, dense_col * 2)` offset inside
`SmemLayoutA` stage 0. Those offsets show whole swizzled row-band jumps
(`0 -> 512 -> 2048 -> 2560 -> ...`), so this is not just a dense-view naming
artifact.

What this does **not** yet prove:

- this dense-copy-view probe is still one layer upstream of the actual
  tag-filled `stage0_A -> partition_S(sA) -> copy -> fp4_shift_A` consumer
  boundary
- it is therefore not yet directly comparable to the runtime-side
  `consumed_source_row0` / `consumed_byte_index0` tag probe

Follow-up blocker:

- an attempted exact tag-filled stage-0 A probe in the standalone builder file
  was reverted after hitting a CUTE builder limitation on the host-side
  `partition_fragment_A(...)` / lower `tCsA` path
- the working offset-bearing probe was restored instead of leaving the target
  broken
- a separate live flashinfer probe attempt patched three runtime locations with
  narrow tracked-thread device `printf`s and confirmed that this capture path
  does **not** surface those device-side prints reliably, even when the BF16
  dump still succeeds
- the live JIT headers were restored after that result; see
  `nano_p1_reference_live_operand_patchpoints_2026-04-14.md`

## Suggested next move

Move the mirrored operand probe into the patched flashinfer reference kernel
at the live `sA/tCsA/tCrA_cv` path. The runtime-side result proves the current
NanoP1 B staging contract is wrong, and the offset-bearing reference-side A
probe proves the reference contract is physically 2D in shared memory. The
next concrete step is to dump the same tracked thread contract from the live
reference kernel using an explicit device probe buffer or device symbol copied
back on the host, not device `printf`, instead of pushing deeper into host-only
builder archaeology.

## Pointers

- Current HEAD: `a008831`
- Reference dump: `proj-2026-04-12-1022/trtllm_reference/golden_nano_k2688/bf16_gemm1_tactic1.bin`
- Layout cheatsheet: `proj-2026-04-12-1022/NANO_P1_LAYOUT_CHEATSHEET.md`
- Prior-session salvage: `proj-2026-04-12-1022/probes/nano_p1_phase3_salvage_2026-04-13.md`
- Compile-time layout capture: `proj-2026-04-12-1022/probes/nano_p1_phase3_layouts_2026-04-13.md`
- Runtime-side operand probe: `proj-2026-04-12-1022/probes/nano_p1_b_operand_probe_2026-04-14.md`
- Reference-side operand probe: `proj-2026-04-12-1022/probes/nano_p1_reference_a_operand_probe_2026-04-14.md`
- Live flashinfer patch points: `proj-2026-04-12-1022/probes/nano_p1_reference_live_operand_patchpoints_2026-04-14.md`
- Plan context: `proj-2026-04-12-1022/PLAN.md` §Step 5
- Shared rules: `PLAN_RULES.md`, `CLAUDE.md` ("Probe-then-decide")
