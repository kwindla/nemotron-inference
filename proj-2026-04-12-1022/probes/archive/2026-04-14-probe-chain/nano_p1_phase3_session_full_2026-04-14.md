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
- the earlier mixed-input candidate path was also corrected after source
  inspection:
  - `moe_gemm_tma_ws_mixed_input_launcher.inl` requires `swap_ab == true`
  - the Nano oracle tactic uses `swap_ab == false`
  - the real live tactic path is CUTLASS SM120 blockscaled in
    `cutlass/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp`

## Suggested next move

Move the mirrored operand probe into the patched flashinfer reference kernel
at the live `tCsB -> tCrB_copy_view -> fp4_shift_B` path in
`cutlass/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp`.

The runtime-side result proves the current NanoP1 B staging contract is wrong,
and the standalone reference-side probes prove the reference operand contract
is physically 2D in shared memory. The next concrete step is to dump the same
tracked-thread B-side contract from the live reference kernel using an explicit
device probe buffer or device symbol copied back on the host, not device
`printf`, instead of pushing deeper into host-only builder archaeology or back
into the wrong mixed-input header family.

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

## 2026-04-14 follow-up: live tactic-1 reference B operand probe

The live flashinfer/CUTLASS tactic-1 B-side probe is now working through an
explicit probe buffer carried in `TmaWarpSpecializedGroupedGemmInput`
workspace, not through the grouped kernel workspace.

New harness controls:

- `NEMOTRON_TRTLLM_OPERAND_PROBE_TIDS=tid0,tid1,...`
- `NEMOTRON_TRTLLM_ONLY_GEMM1_TACTIC=<id>`

The first successful tactic-1-specific run was:

```bash
NEMOTRON_TRTLLM_ONLY_GEMM1_TACTIC=1 \
NEMOTRON_TRTLLM_OPERAND_PROBE_TIDS=32,48,64,80 \
  /home/khkramer/src/nemotron-inference/vllm-env-cu128/bin/python \
  proj-2026-04-12-1022/trtllm_reference/capture_bf16_gemm1.py \
  --num-tokens 128 --hidden-size 2688 --inter-size 1920 \
  --num-experts 1 --top-k 1 --seed 12648430 \
  --golden-dir proj-2026-04-12-1022/trtllm_reference/golden_probe_tmp \
  --metadata-path proj-2026-04-12-1022/trtllm_reference/golden_probe_tmp/bf16_gemm1_metadata.json \
  --input-save-dir proj-2026-04-12-1022/trtllm_reference/golden_probe_tmp \
  --flashinfer-src-root /home/khkramer/src/nemotron-inference/vllm-env-cu128/lib/python3.12/site-packages/flashinfer/data/csrc \
  --venv-root /home/khkramer/src/nemotron-inference/vllm-env-cu128
```

Observed facts:

- The real P1 reference kernel is indeed tactic 1:
  `CtaShape128x128x64B_Cluster1x1x1`.
- The live tactic-1 consumer tids are not `0/1/128/129`; the discovery run
  first surfaced `tid=32..95`.
- For tracked tactic-1 tids `32,48,64,80`:
  - `k_blocks={0,1}`
  - `n_tiles={0..7}`
  - `reg_pre == reg_post`
- The consumer surface is explicitly 2D:
  - `tid=32/64`: `local0.col = stage0_offset = {0,4,8,12,16,20,24,28}`
  - `tid=48/80`: `local0.col = stage0_offset = {1,5,9,13,17,21,25,29}`
  - `part_c0.col = {0,8,32,40,64,72,96,104}`
  - the row bands differ by 16: `16 -> 32`, `20 -> 36`

This is the first trustworthy live reference B contract. The details are in:

- `proj-2026-04-12-1022/probes/nano_p1_reference_b_operand_probe_2026-04-14.md`

One harness caveat remains: an all-tactic targeted sweep still hit
`cudaErrorIllegalInstruction` after tactic 0, so the tactic-1-only capture
path above is the reliable way to collect live P1 operand data right now.

## 2026-04-14 follow-up: Option-A normalized diff and `fp4_shift_B` check

Claude's suggested cheap checks are now done.

### `fp4_shift_B` is not a probe artifact

The live patched probe points in the CUTLASS tactic-1 path are at distinct
program points:

- `RecordBOperandPre(...)` immediately after the B copy into
  `tCrB_copy_view`
- `RecordBOperandPost(...)` immediately after `fp4_shift_B(...)`

And the selected generic `fp4_shift_B(MMA_Op const&, Tensor&&)` body for this
live tactic-1 path is empty in
`cute/atom/mma_traits_sm120.hpp`. The non-identity overloads there are for
`16x8x32` specializations, not the live `16x8x64` blockscaled tactic-1 kernel.

So the observed `reg_pre == reg_post` result is real and structural for this
reference path.

### Runtime and reference match on the normalized consumer surface

Runtime probe rerun:

```bash
NEMOTRON_NANO_P1_B_PROBE_TIDS=32,48,64,80 \
  ./build-sm120-relwithdebinfo/testing/nano_p1_b_operand_probe \
  > /tmp/nano_p1_b_operand_probe_runtime_32_48_64_80.txt
```

Reference probe file:

- `proj-2026-04-12-1022/trtllm_reference/golden_probe_tmp/operand_probe_tactic1.txt`

Normalization:

- runtime `part_output_col0` maps to reference `part_c0.row`
- runtime `part_token_row0` maps to reference `part_c0.col`

Mechanical parser diff result:

```text
shared_entries 64
coord_mismatches 0
```

This is the first direct proof that the Nano runtime and the live flashinfer
reference agree on the tracked tactic-1 B consumer coordinates:

- `part_c0`
- `local0/local1`
- `stage0_offsets`
- `n_tile`
- `k_block`

So the remaining B-side bug is not another `tCsB`/`part_c` layout mismatch.
It is a payload-staging bug: Nano is writing the wrong token bytes into the
same stage-0 shared-memory positions that the reference consumes.

### What this changes

The next diagnostic/fix work should no longer spend time on another broad B
coordinate probe. The right next move is narrower:

1. either add a Nano-side SFB scale-tag probe so data and scale staging can be
   rewritten together, or
2. rewrite the Nano B staging scatter from a row-only permutation to a
   byte-position-based mapping, then immediately rerun the Phase 3 BF16 oracle.

The all-tactic `cudaErrorIllegalInstruction` remains a separate probe transport
problem. Do not use the full sweep as a truth source; the tactic-1-only harness
is the trustworthy path.

## 2026-04-14 follow-up: local builder-derived K64 probe is structurally different from live tactic 1

The standalone local builder-derived K64 B probe now builds and compares
mechanically against the live tactic-1 flashinfer probe.

Files:

- local probe: `testing/backend/nano_p1_reference_b_operand_probe.cu`
- diff helper: `proj-2026-04-12-1022/trtllm_reference/compare_b_operand_probe.py`

Commands:

```bash
cmake --build build-sm120-relwithdebinfo --target nano_p1_reference_b_operand_probe --parallel 4
NEMOTRON_NANO_P1_REFERENCE_B_PROBE_TIDS=32,48,64,80 \
NEMOTRON_NANO_P1_REFERENCE_B_PROBE_STAGE_SLOT_LOW_BYTE=1 \
  ./build-sm120-relwithdebinfo/testing/nano_p1_reference_b_operand_probe \
  > /tmp/nano_p1_reference_b_probe_stage_slot_lowbyte.txt

python3 proj-2026-04-12-1022/trtllm_reference/compare_b_operand_probe.py \
  --live proj-2026-04-12-1022/trtllm_reference/golden_probe_tmp/operand_probe_tactic1.txt \
  --runtime /tmp/nano_p1_reference_b_probe_stage_slot_lowbyte.txt
```

Mechanical diff result:

```text
live_entries=64 runtime_entries=32 shared_entries=32
missing_in_runtime=32 missing_in_live=0
coord_mismatches=0/32
k_count_mismatches=32/32
reg_mismatches=32/32
```

Interpretation:

- the shared `k_block=0` consumer coordinates still line up exactly
- the local builder-derived probe still only exposes `k_block=0`
- for every shared entry:
  - live `k_counts=(2,2,2)`
  - local `k_counts=(1,1,1)`

This is stronger than the earlier "coords right, regs wrong" result. The live
TRT tactic-1 B contract differs from the closest local `UnifiedRoutedFp4Traits<kP13>`
reconstruction at the tensor-structure level (`tCrB`, `tCrB_copy_view`,
`tCsB_coords`), not just at the payload-staging level.

The exact compact shape delta is now captured:

```text
live tactic-1:
  tCrB   = (16,8,2)
  tCrBcv = (32,4,2)
  tCsB   = (32,4,2,4)

local kP13 proxy:
  tCrB   = (16,8,1)
  tCrBcv = (32,4,1)
  tCsB   = (32,4,1,9)
```

Interpretation:

- the first two dimensions match exactly, so the tracked per-thread lane surface
  is not the problem
- the live path has a genuine extra B `k_block` axis and a 4-stage pipeline
- the local `kP13` proxy collapses that to one B `k_block` and a 9-stage
  pipeline

So the local `kP13` builder is now demoted from "candidate oracle" to "lane
surface proxy". Do not keep trying to derive the live tactic-1 payload contract
from it.

This remains true even when the local probe is rebuilt against the same
`vllm-env-cu128` CUTLASS tree used by the live oracle harness. A side build with

```bash
cmake -S . -B build-sm120-vllmref \
  -DNEMOTRON_LOCAL_CUTLASS_INCLUDE_DIR=/home/khkramer/src/nemotron-inference/vllm-env-cu128/lib/python3.12/site-packages/flashinfer/data/cutlass/include \
  -DNEMOTRON_TEST_LOCAL_CUTLASS_INCLUDE_DIR=/home/khkramer/src/nemotron-inference/vllm-env-cu128/lib/python3.12/site-packages/flashinfer/data/cutlass/include
```

still produced:

```text
tCrB   = (16,8,1)
tCrBcv = (32,4,1)
tCsB   = (32,4,1,9)
```

So the local/live mismatch is not a simple "wrong site-packages tree" problem.
It persists even under the live oracle's CUTLASS snapshot.

The local proxy dispatch type is now explicitly captured from the standalone
probe:

```text
cutlass::gemm::MainloopSm120ArrayTmaWarpSpecializedBlockScaled<
  9, 3, cluster(1,1,1),
  KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120<3>>
```

That matters because it rules out another common dead-end:

- the local `kP13` proxy is **already** in the same broad SM120 array-TMA
  blockscaled family as the live tactic-1 probe patchpoint
- the local/live split is therefore not "array-TMA vs non-array-TMA" and not
  "wrong CUTLASS tree"

What remains unexplained is narrower and more useful:

- local proxy resolves to `PipelineStages=9` and one visible B `k_block`
- live tactic 1 exposes `tCsB(...,4)` and `k_counts=(2,2,2)`

So the next useful work is to explain that stage/depth mismatch inside the live
array-TMA blockscaled path, not to keep swapping local CUTLASS roots or local
builder families.

Implication for the next step:

- do **not** rewrite NanoP1 B staging again from this local K64 builder probe
- continue on the live-reference side and dump a compact structural signature of
  the selected tactic-1 B operand tensors until the `k_counts=(2,2,2)` vs
  `(1,1,1)` split is explained

## 2026-04-14 follow-up: validated physical-offset decode, but not yet a live fix

The standalone runtime B probe now has a stronger result than the earlier
row/byte formula, and one failed live-kernel experiment is now ruled out.

### Probe-side result

The standalone `nano_p1_b_operand_probe` had a missing barrier between:

- reading back the source-row tags, and
- overwriting stage-0 B with byte-index tags

After inserting that `__syncthreads()`, the `use_position_map=1` validation
became trustworthy.

Tracked tactic-1 runtime families:

- `32,36,40,44,48,52,56,60`
- `33,37,41,45,129,133,137,141`
- `64,68,72,76,80,84,88,92`
- `145,149,153,157,33,37,41,45`

all now report:

```text
total_row_mismatches=0/128
```

when the probe stages source-row tags using a decode based on the physical
`stage0_B(row, byte_index * 2) & 0x1ff` offset:

```text
if band_offset < 32:
  source_row =
      ((band_offset >> 2) & 1) << 3 |
      ((band_offset >> 3) & 1) << 5 |
      ((band_offset >> 4) & 1) << 6

if 128 <= band_offset < 160:
  source_row =
      0x2 |
      ((band_offset >> 1) & 1) << 4 |
      ((band_offset >> 2) & 1) << 3 |
      ((band_offset >> 3) & 1) << 5 |
      (((~band_offset) >> 4) & 1) << 6
```

Interpretation:

- for the tracked tactic-1 consumer surface, the desired token row is a stable
  function of the **physical stage-0 offset**
- this is stronger than the older row-only and `(local_row, byte_index)` stories
- this is now the load-bearing runtime-side B probe fact

### Failed live promotion

I tried promoting that helper directly into the live NanoP1 B staging path,
including paired SFB scale staging by 4-byte groups.

That experiment was **wrong** and was reverted.

Observed failure:

```text
Phase 1 synthetic all-ones mismatch:
  actual   = 64
  expected = 256
```

So this helper is not yet the full live-kernel staging contract. It is
currently validated only on the tracked tactic-1 consumer subset.

Current tree state after revert:

- Phase 1 all-ones: PASS
- Phase 3 Nano bucket K=2688: back to the known partial-fix baseline
  `4074 / 245760`

### What this means for the next session

Do **not** try to land the physical-offset helper directly into the kernel yet.

The next concrete step is to derive the **full staging-surface ownership** for
`stage0_B(row, byte_index * 2)`:

1. enumerate which physical offsets are actually populated across the full live
   staging surface, not just the tracked consumer subset
2. confirm which offsets feed the passing Phase 1 all-ones path
3. only then extend the physical-offset rule into a live staging rewrite

The probe is now strong enough to guide that work, but it is not yet a safe
drop-in kernel fix.

## 2026-04-14 follow-up: real SFB fragment probe and corrected live-promotion result

I extended the standalone `nano_p1_b_operand_probe` so the B-side scale path is
measured from the actual `tCrSFB` fragment after `cute::copy(...)`, not from a
single anchor cell in shared memory.

Mechanically:

- fill `sSFB_stage0` with logical-row tags
- `cute::copy(...)` into `tCrSFB_cv`
- pack `tCrSFB(_, n_tile, k_block)` into one 32-bit word
- repeat with logical-column tags

This yields a direct consumer contract for Nano P1 SFB.

### SFB consumer contract

For the validated standalone B data position-map run
(`NEMOTRON_NANO_P1_B_PROBE_USE_POSITION_MAP=1`):

```text
consumed_scale_row_byte
  = (consumed_source_row0 & ~7) | (scale_local_row0 & 7)

consumed_scale_col_bytes(k_block = 0)
  = [0, 16, 32, 48]

consumed_scale_col_bytes(k_block = 1)
  = [64, 80, 96, 112]
```

Equivalently, the logical scale-byte indices are:

```text
k_block = 0 -> [0, 1, 2, 3]
k_block = 1 -> [4, 5, 6, 7]
```

So the SFB K-side split is now explicit, and the row side is a grouped
8-row-band contract rather than a direct token-row identity.

### Correction: the full B data live patch is still not Phase-1-safe

An earlier working assumption was:

- full B data position-map promotion preserves Phase 1
- therefore SFB is the next live-kernel blocker

That assumption is wrong on the current tree.

I reran the live-kernel experiment in two forms:

1. full B data position-map only, with the old row-only SFB staging left intact
2. full B data position-map plus a simple identity-row SFB experiment

Both immediately fail Phase 1 synthetic all-ones:

```text
Phase 1 synthetic all-ones FAIL total_mismatches=32768
```

Both experiments were reverted. The tree is back at:

```text
Phase 1 synthetic all-ones PASS
Phase 3 total matches=4074/245760
```

This is a load-bearing correction for the next session:

- the standalone B data helper is probe-valid
- the standalone SFB fragment contract is probe-valid
- but the live kernel still has an unresolved gap **before** we can say "B data
  is fixed and only SFB remains"

### Updated next step

Do not treat SFB as the only missing live fix yet.

The next concrete move should be to explain why the stage-offset B decode is
valid on the standalone consumer surface but still breaks Phase 1 when applied
to the live staging loop. That likely means the live loop populates additional
positions or depends on a staging invariant the current standalone probe does
not cover.

## 2026-04-14 follow-up: why the live B helper fails immediately

I added a full-surface counter to `nano_p1_b_operand_probe` over the entire
`stage0_B(row, byte_index * 2)` write domain.

Measured result:

```text
rows      = 128
bytes/row = 64
total     = 8192 stage-0 write positions

full_stage0_surface invalid_source_rows=6144
first_invalid=(row=1 byte=32 offset=32)
full_stage0_surface band_counts=512,512,512,512,512,512,512,512,
                               512,512,512,512,512,512,512,512
```

This is the missing explanation for the failed live promotions:

- the current `NanoP1SourceRowForBStageOffset(...)` helper only covers
  `2048 / 8192` write positions
- `6144 / 8192` positions fall into uncovered `band_offset` windows and resolve
  to `-1`
- the earlier standalone "row_mismatches=0" result only sampled a narrow
  consumer subset that happened to live in the covered windows

So the live failure is no longer mysterious. The helper is not a near-complete
kernel fix; it is only a quarter-surface decode.

### Concrete implication

The next B probe extension must enumerate the **full B consumer fragment
surface**, not just the first two `tCsB` anchor coordinates per
`(tid, n_tile, k_block)`.

That is now the critical-path task. Without it, every live promotion of the
current helper will necessarily drop bytes on uncovered stage-0 bands.

## 2026-04-14 follow-up: full consumed B surface is clean in the standalone probe

I extended the standalone `nano_p1_b_operand_probe` sweep to the full
`tid=0..255` space in 8-thread batches, still using
`NEMOTRON_NANO_P1_B_PROBE_USE_POSITION_MAP=1`, and checked the full per-entry
consumed B copy-view fragment for every `(tid, n_tile, k_block)`.

Aggregate result over all `4096` entries:

```text
copy_view_row_mismatches = 0
row_bad_count            = 0
missing_count            = 0
```

So the stage-offset B decode is not just correct for the earlier sampled
tactic-1 families. It covers the entire **consumed** B copy-view surface in the
standalone runtime model.

This means the earlier "quarter-surface decode explains the live failure"
story is incomplete. The full write-surface invalid-count result is still real,
but the consumed surface we can currently observe is already clean.

## 2026-04-14 follow-up: coarse byte + SFB invariants are also globally clean

Over the same `4096` entries, the current probe output satisfies:

- `consumed_byte_index0 == local_col0 / 2`
- `consumed_scale_row_word0 == repeat((consumed_source_row0 & ~7) | (scale_local_row0 & 7))`
- `consumed_scale_col_word0 == [0,16,32,48]` for `k_block=0`,
  `[64,80,96,112]` for `k_block=1`

Aggregate result:

```text
byte_bad       = 0
scale_row_bad  = 0
scale_col_bad  = 0
```

So the remaining standalone B-side uncertainty is no longer at the coarse
`(source_row, byte_index)` or first-word SFB level.

## 2026-04-14 follow-up: the copied B view is sub-byte

I added a check against the actual copied `tCrB_cv` payload after filling
stage-0 B with byte-index tags.

Result:

```text
copy_view_byte_bad_count = 4096
```

This is not a new kernel failure. The destination view elements are
`cute::subbyte_reference` FP4 values, not raw packed bytes, so the naive
"expect byte index `col / 2`" invariant is wrong at that layer.

The supporting coarse register snapshots remain stable:

```text
k_block=0: reg0=0x03020100 reg1=0x13121110
k_block=1: reg0=0x23222120 reg1=0x33323130
reg_pre == reg_post
```

So the standalone probe has pushed the remaining B-side ambiguity down to the
sub-byte / nibble contract.

## Updated next step

The highest-value next move is no longer "find more B rows."

It is one of:

1. reconstruct the exact `(source_row, byte_index)` tuple for each packed B
   register byte and compare that against the live flashinfer tactic-1
   `reg_pre` dump using the saved real `input_fp4_permuted.bin`, or
2. run the live flashinfer tactic-1 probe on a synthetic activation pattern
   whose FP4 nibble codes are known in advance, so `reg_pre` becomes directly
   decodable at the nibble level

## 2026-04-14 follow-up: live/runtime B probes agree on coordinates and disagree on regs

I added a mechanical diff helper at:

- `proj-2026-04-12-1022/trtllm_reference/compare_b_operand_probe.py`

and used it to compare the live tactic-1 B probe against the standalone Nano
runtime B probe on two synthetic source modes:

1. `stage_slot_bit=0`
2. `stage_slot_low_byte=1`

Both comparisons give the same top-line result:

```text
shared_entries   = 64
coord_mismatches = 0 / 64
reg_mismatches   = 64 / 64
```

So the tracked live tactic-1 kernel and the standalone Nano runtime probe
agree on the whole first-level consumer surface:

- `part_c0`
- `local0/local1`
- `stage0_offsets`
- `n_tile`
- `k_block`

but they still disagree on **every** pre-shift B register word.

Representative `stage_slot_bit=0` mismatch:

```text
live:    tid=32 n_tile=0 k_block=0 reg_pre=(0xc0303060,0x1cafa45f)
runtime: tid=32 n_tile=0 k_block=0 reg_pre=(0x18081000,0xffffffff)
```

Representative `stage_slot_low_byte=1` mismatch:

```text
live:    tid=32 n_tile=0 k_block=0 reg_pre=(0x86a4d2c0,0xfe866d6f)
runtime: tid=32 n_tile=0 k_block=0 reg_pre=(0x03020100,0x13121110)
```

The low-byte run matters because it kills the "maybe bit-0 was a weird probe
plane" escape hatch. The runtime-side pre-shift words collapse to fixed
`0x03020100/0x13121110` and `0x23222120/0x33323130` patterns while the live
reference varies across `n_tile` and `k_block`.

### Updated fault domain

The remaining B-side fault is **not** the stage-0 B coordinate map.

It sits downstream of shared-memory address selection and upstream of MMA:

- `make_tiled_copy_B(...)`
- `retile_D(tCrB)`
- `partition_fragment_B(...)`
- or a materially different `TiledMma` / `SmemCopyAtomB` contract

Another B-row scatter rewrite is not the right next move.

### Updated next step

Build a local builder-derived K64 B probe using the existing bridge types in
`runtime/src/backend/fused_moe_prefill/nvfp4_bridge.cuh`, specifically the
`UnifiedRoutedFp4Traits<kP13>` / `TracedP13TiledMma` /
`TracedP13SmemCopyAtomB` path, and compare that local K64 contract against the
live tactic-1 probe.

If that local K64 probe matches the live reference, then the NanoP1 custom
`NanoP1TiledMma` / `NanoP1SmemCopyAtomB` path is the wrong contract and should
be rewritten around the builder-derived one. If it does not match, then the
remaining gap is in the exact tactic-1 kernel family rather than just the
current NanoP1 custom path.

## 2026-04-14 follow-up: the local K64 builder probe does not match the live tactic-1 contract either

I implemented the local builder-derived K64 B probe at:

- `testing/backend/nano_p1_reference_b_operand_probe.cu`

The important implementation detail is that it only built cleanly once it used
the same include stack as `runtime/src/backend/fused_moe_prefill.cu`:

- `common_helpers.cuh`
- `nvfp4_cute.cuh`
- `nvfp4_bridge.cuh`

Trying to include `nvfp4_bridge.cuh` standalone was a dead end and should not
be repeated.

### Command

```bash
NEMOTRON_NANO_P1_REFERENCE_B_PROBE_TIDS=32,48,64,80 \
NEMOTRON_NANO_P1_REFERENCE_B_PROBE_STAGE_SLOT_LOW_BYTE=1 \
  ./build-sm120-relwithdebinfo/testing/nano_p1_reference_b_operand_probe \
  > /tmp/nano_p1_reference_b_probe_stage_slot_lowbyte.txt

python3 proj-2026-04-12-1022/trtllm_reference/compare_b_operand_probe.py \
  --live proj-2026-04-12-1022/trtllm_reference/golden_probe_stage_slot_lowbyte_rawbytes/operand_probe_tactic1.txt \
  --runtime /tmp/nano_p1_reference_b_probe_stage_slot_lowbyte.txt
```

### Result

```text
live_entries=64 runtime_entries=32 shared_entries=32
missing_in_runtime=32 missing_in_live=0
coord_mismatches=0/32
reg_mismatches=32/32
```

Representative mismatch:

```text
live:  tid=32 n_tile=0 k_block=0 reg_pre=(0x86a4d2c0,0xfe866d6f)
local: tid=32 n_tile=0 k_block=0 reg_pre=(0x03020100,0x13121110)
```

### What this settles

- The local `UnifiedRoutedFp4Traits<kP13>` / `TracedP13TiledMma` /
  `TracedP13SmemCopyAtomB` reconstruction reproduces the same synthetic
  register pattern as our existing Nano runtime-side B probe.
- It still does **not** reproduce the live TRT tactic-1 register contract.
- The live-only entries are exactly the `k_block=1` half of the live tactic-1
  surface, while the local reconstruction only exposes `k_block=0`.

So the remaining B-side gap is not just "our custom NanoP1 path differs from
the builder-derived K64 path." The builder-derived local K64 path also differs
from the live tactic-1 kernel.

### Updated next step

The next probe needs to stay on the live reference side and dump more of the
structural `tCrB_copy_view` contract, not more stage-0 scatter guesses. The
highest-value fields are:

1. live `size<2>(tCrB_copy_view)` / `K_BLOCK_MAX`
2. a structural signature of the selected `SmemCopyAtomB` / `TiledMma`
3. enough layout metadata to explain why live tactic 1 exposes `k_block={0,1}`
   while the local `kP13` reconstruction only surfaces `k_block=0`

## 2026-04-14 follow-up: explicit live dispatch tuple now confirms the local proxy mismatch

I patched the live flashinfer/TRT operand-probe transport so the captured file
now includes the actual dispatch constants from
`CollectiveMainloop::DispatchPolicy`.

### Live tactic-1 rerun

After rerunning the tactic-1-only capture, the probe header now says:

```text
NEMOTRON_B_OPERAND_PROBE ... dispatch=(4,3) ...
```

and each entry line repeats `dispatch=(4,3)`.

This turns the earlier structural inference into an explicit fact:

- live tactic 1 is running `dispatch=(4,3)`

### Local proxy rerun

I updated the standalone local B probe to emit the same machine-readable field.
Its output now says:

```text
nano_p1_reference_b_operand_probe: dispatch=(9,3) dispatch_policy=...
  n_tile=0 k_block=0 dispatch=(9,3) ...
```

So the local proxy is explicitly:

- local `TracedP13CollectiveMainloop`: `dispatch=(9,3)`

### Updated mechanical diff

Using the updated `compare_b_operand_probe.py` against the fresh live tactic-1
dump and the refreshed local probe:

```text
live_entries=64 runtime_entries=32 shared_entries=32
missing_in_runtime=32 missing_in_live=0
coord_mismatches=0/32
dispatch_mismatches=32/32
k_count_mismatches=32/32
reg_mismatches=32/32
```

Representative mismatch:

```text
live:    tid=32 n_tile=0 k_block=0 dispatch=(4,3) k_counts=(2,2,2)
runtime: tid=32 n_tile=0 k_block=0 dispatch=(9,3) k_counts=(1,1,1)
```

### Updated interpretation

This is the cleanest statement of the current state so far:

- the live tactic-1 kernel and the local `kP13` proxy share the same consumer
  coordinate surface (`coord_mismatches=0/32`)
- they do **not** share the same dispatch stage depth
- they do **not** share the same visible B `k_block` count
- they do **not** share the same register payload

So the local `kP13` / builder-derived proxy is only a lane-surface orientation
tool. It is not a faithful live tactic-1 oracle, and it should not be used as
the basis for another Nano kernel rewrite.

### Updated next move

Stay on the live-reference path. The best next probe is to capture one more
compact structural signature from the live array-TMA blockscaled mainloop that
explains why it resolves to `dispatch=(4,3)` and `k_counts=(2,2,2)` while the
local proxy resolves to `dispatch=(9,3)` and `k_counts=(1,1,1)`.

## 2026-04-14 follow-up: shared-storage bytes show the live/local split is not just a 1 KB carveout difference

I extended the live probe header and the local standalone probe to print:

- `dispatch=(stages,scheduler_stages)`
- `shared_storage_bytes=(epilogue,mainloop)`

Fresh header values:

```text
live:  dispatch=(4,3) shared_storage_bytes=(13312,74752)
local: dispatch=(9,3) shared_storage_bytes=(14336,83968)
```

The builder source confirms that `StageCountAutoCarveout` is driven by:

- SMEM capacity
- `sizeof(CollectiveEpilogue::SharedStorage)` as the carveout
- per-stage storage for A/B/SFA/SFB plus the pipeline

That means the new numbers are informative:

- epilogue carveout delta is only `1024` bytes
- stage-count delta is `5`
- mainloop shared-storage delta is `9216` bytes

Per-stage rough averages:

```text
live:  74752 / 4 ~= 18688 bytes per stage
local: 83968 / 9 ~=  9329 bytes per stage
```

So the live/local split is not well explained by "the epilogue carveout changed
a little and StageCountAutoCarveout picked a nearby different answer." The live
kernel has a materially different **per-stage storage contract**. That is
consistent with the earlier structural findings:

- live: `dispatch=(4,3)`, `k_counts=(2,2,2)`, `tCsB(...,4)`
- local: `dispatch=(9,3)`, `k_counts=(1,1,1)`, `tCsB(...,9)`

### Updated next move

The next useful probe should target the source of that per-stage storage
difference in the live array-TMA blockscaled path, not another Nano kernel
rewrite and not another row/byte staging hypothesis. The most likely candidates
are the live B-scale / SFB contract and the exact blockscaled array-TMA
mainloop specialization selected by tactic 1.

## 2026-04-14 follow-up: live/local SFB structure also diverges

I extended the existing B operand probe with compact scale-side structure:

- `sf_counts=(tCrSFB_k_blocks,tCrSFB_copy_view_k_blocks,tCsSFB_coord_k_blocks)`
- `sf_shapes=tCrSFB(...) tCrSFBcv(...) tCsSFB(...)`

### Live tactic-1

Fresh live tactic-1 entries now report:

```text
dispatch=(4,3)
k_counts=(2,2,2)
sf_counts=(2,2,2)
shapes=tCrB(16,8,2) tCrBcv(32,4,2) tCsB(32,4,2,4)
sf_shapes=tCrSFB(64,8,2) tCrSFBcv(128,4,2) tCsSFB(128,4,2,4)
```

### Local `kP13` proxy

Fresh local probe entries report:

```text
dispatch=(9,3)
k_counts=(1,1,1)
sf_counts=(64,64,1)
shapes=tCrB(16,8,1) tCrBcv(32,4,1) tCsB(32,4,1,9)
sf_shapes=tCrSFB(64,8,64) tCrSFBcv(128,4,64) tCsSFB(128,4,1,9)
```

### Updated diff

The refreshed mechanical diff now says:

```text
coord_mismatches=0/32
dispatch_mismatches=32/32
k_count_mismatches=32/32
sf_count_mismatches=32/32
reg_mismatches=32/32
```

### Updated interpretation

This is a stronger version of the same conclusion:

- live and local still agree on the consumer coordinate surface
- but they disagree on dense-B structure
- and they also disagree on the B-scale / SFB structure

So the local `kP13` proxy is not just the wrong dense-B oracle. It is also the
wrong scale oracle. That means another Nano kernel rewrite based on it would be
guessing on both operands of the zipped B-side input.

### Updated next move

Stay on the live-reference path. The next high-value probe should extract one
more structural signature from the live array-TMA blockscaled specialization
that explains the compact live `sf_counts=(2,2,2)` / `tCsSFB(...,2,4)` contract.
The strongest candidate is the exact live scale-layout / `SmemCopyAtomSFB`
specialization rather than any further local-proxy archaeology.

## 2026-04-14 follow-up: live/local `SmemLayoutSFB` stage shape is now explicit

I finished the pending probe schema patch and reran both sides. The probe dump
now carries:

```text
sf_layout=(stage_elems,total_elems,stage_shape_dim0,stage_shape_dim1,layout_tv_dim0,layout_tv_dim1)
```

This required patching both live blockscaled probe call sites
(`sm120_blockscaled_mma_array_tma.hpp` and `sm120_blockscaled_mma_tma.hpp`),
plus both printers, because the non-array-TMA path still compiles in the
flashinfer JIT build even when tactic 1 itself uses the array-TMA path.

### New measured values

Live tactic-1:

```text
dispatch=(4,3)
k_counts=(2,2,2)
sf_counts=(2,2,2)
sf_layout=(1024,4096,128,128,256,128)
```

Local `kP13` proxy:

```text
dispatch=(9,3)
k_counts=(1,1,1)
sf_counts=(64,64,1)
sf_layout=(512,4608,128,64,256,128)
```

Updated diff from `compare_b_operand_probe.py`:

```text
coord_mismatches=0/32
dispatch_mismatches=32/32
k_count_mismatches=32/32
sf_count_mismatches=32/32
sf_layout_mismatches=32/32
reg_mismatches=32/32
```

### Why this matters

This is the first explicit proof that the live/local scale-side divergence is
not just count metadata or stage depth:

- `layout_sfb_tv_dim0/1` still match exactly: `(256,128)`
- but the per-stage `SmemLayoutSFB` shape differs:
  - live: `(128,128)`
  - local: `(128,64)`
- and the per-stage SFB storage footprint differs in lockstep:
  - live: `1024`
  - local: `512`

So the live tactic-1 kernel is carrying a genuinely different scale-stage
layout, specifically a doubled K-side stage shape, while preserving the same
TV layout. That means the local builder-derived `kP13` proxy is wrong at an
even deeper level than previously stated:

- wrong dispatch stage depth
- wrong dense-B `k_block` contract
- wrong SFB `k_block` contract
- wrong `SmemLayoutSFB` stage shape

Another Nano kernel rewrite from that proxy would still be guessing.

### Updated next move

Stay on the live-reference path, but narrow it further. The next probe should
identify **why** the live tactic-1 blockscaled path resolves to

- `SmemLayoutSFB` stage shape `(128,128)`
- `sf_counts=(2,2,2)`
- `dispatch=(4,3)`

while the local `kP13` proxy resolves to

- `SmemLayoutSFB` stage shape `(128,64)`
- `sf_counts=(64,64,1)`
- `dispatch=(9,3)`

The most likely remaining sources are the selected live `SmemLayoutAtomSFB`
and/or `SmemCopyAtomSFB` specialization, not stage-row scattering in Nano P1.

### Source-derived negative result

I also checked the generic CUTLASS SM120 dense blockscaled builder path in
`cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl`.

For dense `nv_float4_t` inputs, source says:

- `SfVectorSize = 16`
- `Blk_MN = 128`
- `Blk_SF = 4`
- `SmemLayoutAtomSFB` is computed directly from `TileShape_MNK`,
  `SFVectorSize`, and `MMA_NSF`

That generic builder math predicts the local proxy's scale-stage shape:

```text
sf_layout stage shape = (128,64)
```

which matches the standalone local probe:

```text
sf_layout=(512,4608,128,64,256,128)
```

It does **not** predict the live tactic-1 result:

```text
sf_layout=(1024,4096,128,128,256,128)
```

So this is now a load-bearing negative result:

- the live tactic-1 kernel is not simply the generic SM120 dense blockscaled
  builder contract we can reconstruct locally
- the next probe should target the TRT/flashinfer specialization or layout
  override that yields the doubled live `SmemLayoutSFB` stage shape

### 2026-04-14 follow-up: `sf_atom` makes the scale-side split explicit

I extended both the live tactic-1 B probe and the standalone local reference-B
probe to emit:

```text
sf_atom=(SFVecSize, SmemLayoutAtomSFB dim0, SmemLayoutAtomSFB dim1)
```

Fresh results:

- live tactic 1:

  ```text
  sf_atom=(16,128,128)
  sf_layout=(1024,4096,128,128,256,128)
  ```

- local `kP13` proxy:

  ```text
  sf_atom=(16,128,64)
  sf_layout=(512,4608,128,64,256,128)
  ```

The updated mechanical diff now reports:

```text
coord_mismatches=0/32
dispatch_mismatches=32/32
k_count_mismatches=32/32
sf_count_mismatches=32/32
sf_atom_mismatches=32/32
sf_layout_mismatches=32/32
reg_mismatches=32/32
```

This matters because it removes the last ambiguity in the earlier scale-side
result:

- `SFVecSize` still matches: `16`
- `layout_sfb_tv` still matches: `(256,128)`
- but the live/local split is already present in `SmemLayoutAtomSFB` itself:
  - live atom: `(128,128)`
  - local atom: `(128,64)`

So the next live-reference step should target the selected live
`SmemLayoutAtomSFB` / `SmemCopyAtomSFB` specialization directly. Another Nano
kernel rewrite would still be guessing from the wrong atom contract.

### 2026-04-14 correction: TRT's `64B` label means byte-K, so the old local `kP13` proxy was off by 2x in K

I traced the SM120 tile dispatch in
`.../moe_gemm/moe_gemm_template_dispatch_tma_ws.h`.

The `SHAPE_CASE` macro does:

```cpp
constexpr int KtileBytes =
    (K * 8) / cutlass::sizeof_bits<T>::value;
using TileShape = Shape<_M, _N, Int<KtileBytes>>;
```

For FP4, `sizeof_bits<T> == 4`, so:

- `CtaShape128x128x64B` -> `TileK = 128`
- `CtaShape128x128x128B` -> `TileK = 256`

That resolves the earlier contradiction around the live probe's
`sf_atom=(16,128,128)`: the live tactic-1 kernel is structurally a `TileK=128`
kernel, and the old standalone local `kP13` proxy
`TracedP13MmaTileShape=(128,128,64)` was never equivalent.

I repointed `testing/backend/nano_p1_reference_b_operand_probe.cu` to the
existing local `kP12` / `TracedP5CollectiveMainloop` path
(`TracedP5MmaTileShape=(128,128,128)`) and reran the diff against the live
tactic-1 flashinfer capture.

Updated mechanical diff:

```text
live_entries=64 runtime_entries=64 shared_entries=64
missing_in_runtime=0 missing_in_live=0
coord_mismatches=0/64
dispatch_mismatches=0/64
k_count_mismatches=0/64
sf_count_mismatches=64/64
sf_atom_mismatches=0/64
sf_layout_mismatches=0/64
reg_mismatches=64/64
```

What matters in that result:

- the corrected local proxy now matches the live tactic-1 kernel on the
  load-bearing structural contract:
  - `dispatch=(4,3)`
  - `k_counts=(2,2,2)`
  - `sf_atom=(16,128,128)`
  - `sf_layout=(1024,4096,128,128,256,128)`
  - `tCsSFB(128,4,2,4)`
  - full tracked consumer coordinates
- the remaining `sf_count_mismatches=64/64` are a probe-schema mismatch:
  live reports logical `k_block` counts `(2,2,2)`, while the corrected local
  probe's `tCrSFB` / `tCrSFBcv` expose flattened fragment extent `128`; the
  load-bearing `tCsSFB(...,2,4)` already matches

So the big structural problem is now gone. After correcting the K-width
mistake, the remaining live/local divergence is in `reg_pre` payload only.

That changes the next move:

- stop spending time on stage-count / atom-layout archaeology
- use the corrected local `kP12` proxy plus the Nano runtime probe to localize
  the packed-B register payload mismatch directly

### 2026-04-14 follow-up: live `w1_fp4` K=128 captures invalidate the old `+64` theory

I added a direct `w1_fp4` override path to
`proj-2026-04-12-1022/trtllm_reference/capture_bf16_gemm1.py` so the live
flashinfer tactic-1 path can run with raw B payload bytes set to either:

- `row_index`
- `byte_index`

The first pass at the default `hidden_size=256` made it look like the live
kernel wanted a `+64` byte offset. That was wrong. With `TileK=128`, the
`hidden_size=256` run can observe the second `k_base` iteration.

The apples-to-apples captures are the K=128 runs:

- `proj-2026-04-12-1022/trtllm_reference/golden_probe_w1_row_index_k128/`
- `proj-2026-04-12-1022/trtllm_reference/golden_probe_w1_byte_index_k128/`

Those runs use `--hidden-size 128`, so tactic 1 only sees one K tile.

### 2026-04-14 follow-up: exact live/runtime B mismatch at K=128

I added
`proj-2026-04-12-1022/trtllm_reference/compare_w1_runtime_b_probe.py`
to compare:

- live tactic-1 `operand_probe_tactic1.txt`
- runtime `nano_p1_b_operand_probe`

on the common tracked key:

- `(tid, n_tile, k_block)`
- `part_c0`
- `local0`
- `stage0_offset0`
- `reg_pre`

Current command:

```bash
python3 proj-2026-04-12-1022/trtllm_reference/compare_w1_runtime_b_probe.py \
  --live proj-2026-04-12-1022/trtllm_reference/golden_probe_w1_byte_index_k128/operand_probe_tactic1.txt \
  --runtime /tmp/nano_p1_b_operand_probe_runtime_32_48_64_80_posmap_current.txt \
  --show-mismatches
```

Current result:

```text
live_entries=64 runtime_entries=64 shared_entries=64
missing_in_runtime=0 missing_in_live=0
coord_mismatches=0/64
reg_mismatches=32/64
even_family_reg_mismatches=0
odd_family_reg_mismatches=32
odd_family_swap_candidates=32
reg_mismatches_after_odd_family_kblock_swap=0/64
```

This is the best current localization.

It means:

- the live and runtime B consumer coordinate surface matches exactly
- the remaining tracked B payload bug is not row placement
- the remaining tracked B payload bug is not a global `+64` byte shift
- the remaining tracked B payload bug is exactly an odd-family `k_block` swap

More concretely:

- even family (`tid=32,64`, `local_col0={0,4,8,...}`, `stage0_offset0={0,4,8,...}`):
  live and runtime `reg_pre` match
- odd family (`tid=48,80`, `local_col0={1,5,9,...}`, `stage0_offset0={1,5,9,...}`):
  runtime's `reg_pre` windows are swapped between `k_block=0` and `k_block=1`

Representative mismatch:

```text
tid=48 n_tile=0 local_col0=1 stage0_offset0=1
runtime k_block=0 reg_pre=(0x23222120,0x33323130)
live    k_block=0 reg_pre=(0x03020100,0x13121110)
runtime k_block=1 reg_pre=(0x03020100,0x13121110)
live    k_block=1 reg_pre=(0x23222120,0x33323130)
```

So a virtual odd-family `k_block` swap explains the entire tracked live/runtime
B payload gap.

### 2026-04-14 follow-up: raw stage-byte fill is not yet proven guilty

I also extended `testing/backend/nano_p1_b_operand_probe.cu` to emit a compact
copy-view stage-byte-slot signature for each tracked entry.

That did not surface a new split yet. For the tracked first-anchor positions:

- `tid=32, n_tile=0`: `copy_view_stage_byte_slots=[0,...]`
- `tid=48, n_tile=0`: `copy_view_stage_byte_slots=[0,...]`
- `tid=32, n_tile=1`: `copy_view_stage_byte_slots=[2,...]`
- `tid=48, n_tile=1`: `copy_view_stage_byte_slots=[2,...]`

The signature still collapses to a single repeated byte slot per tracked entry,
so it does not yet prove that the divergence is in the coarse staged-byte fill.

That pushes the next useful probe boundary downstream, closer to:

- the B copy-view / retile path
- or the mapping from copied B values into the final `reg_pre` payload

### 2026-04-14 baseline check after reverting the bad `^32` promotion

After reverting the attempted live-kernel byte remap, the real oracle is back at
the known state:

- Phase 1: `PASS`
- Phase 3: `241686` mismatches, i.e. `4074 / 245760` matches

So the tree is back at the pre-experiment baseline.

### 2026-04-14 follow-up: runtime `tCrB_cv` shows the odd-family swap before recast

I extended `testing/backend/nano_p1_b_operand_probe.cu` again so the standalone
runtime probe now dumps a compact raw signature from the copied `tCrB_cv`
values before `recast<BRegister>(tCrB(...))`.

Command:

```bash
NEMOTRON_NANO_P1_B_PROBE_TIDS=32,48,64,80 \
NEMOTRON_NANO_P1_B_INPUT_FILE=/tmp/golden_probe_w1_byte_index_k128_w1_fp4_first128.bin \
NEMOTRON_NANO_P1_B_INPUT_ROWS=128 \
NEMOTRON_NANO_P1_B_PACKED_ROW_BYTES=64 \
  ./build-sm120-relwithdebinfo/testing/nano_p1_b_operand_probe \
  > /tmp/nano_p1_b_operand_probe_runtime_w1_byte_rowperm_rawsig.txt
```

Load-bearing sampled result:

```text
even family: local_col0=0
  k_block=0 raw=[0,0,1,0,2,0,3,0] reg_pre=(0x03020100,0x13121110)
  k_block=1 raw=[0,2,1,2,2,2,3,2] reg_pre=(0x23222120,0x33323130)

odd family: local_col0=1
  k_block=0 raw=[0,2,1,2,2,2,3,2] reg_pre=(0x23222120,0x33323130)
  k_block=1 raw=[0,0,1,0,2,0,3,0] reg_pre=(0x03020100,0x13121110)
```

This answers the recast question:

1. The odd-family `k_block` inversion is already present in `tCrB_cv`.
2. `recast<BRegister>(tCrB(...))` is not introducing the first divergence.
3. The next useful probe stays upstream of recast:
   - widen the raw `tCrB_cv` dump to the exact elements that feed `reg_pre`, or
   - instrument the B copy-view / staging path directly for the odd family.

Caveat:

- The current raw signature only covers the first 8 physical `tCrB_cv`
  elements. That is enough to prove the low-byte odd-family inversion above,
  but it does not yet explain every later `n_tile`: some higher-`n_tile`
  windows collapse to zeros while `reg_pre` is still non-zero.

### 2026-04-14 follow-up: `copy_view_raw` was only a physical surface

I widened the runtime raw probe to 32 copied `tCrB_cv` elements and also dumped
the logical fragment `tCrB(_, n_tile, k_block)` as `frag_raw_consumed_bytes`.

Command:

```bash
NEMOTRON_NANO_P1_B_PROBE_TIDS=32,48,64,80 \
NEMOTRON_NANO_P1_B_INPUT_FILE=/tmp/golden_probe_w1_byte_index_k128_w1_fp4_first128.bin \
NEMOTRON_NANO_P1_B_INPUT_ROWS=128 \
NEMOTRON_NANO_P1_B_PACKED_ROW_BYTES=64 \
  ./build-sm120-relwithdebinfo/testing/nano_p1_b_operand_probe \
  > /tmp/nano_p1_b_operand_probe_runtime_w1_byte_fragraw.txt
```

The result split the two surfaces cleanly:

- for `n_tile=0..3`, `copy_view_raw` and `frag_raw` agree
- for `n_tile=4..7`, they do not

Representative even-family case:

```text
n_tile=4 k_block=0
  copy_view_raw = [0,2,1,2,2,2,3,2,...]
  frag_raw      = [0,0,1,0,2,0,3,0,...]

n_tile=4 k_block=1
  copy_view_raw = [0,0,0,0,0,0,0,0,...]
  frag_raw      = [0,2,1,2,2,2,3,2,...]
```

So `copy_view_raw` is only a physical retiled surface. It is not a stable
logical comparison surface for upper-half `n_tile`s.

The logical fragment `frag_raw` is stable and load-bearing:

- even family:
  - `k_block=0 -> [0,0,1,0,2,0,3,0,0,1,1,1,2,1,3,1]`
  - `k_block=1 -> [0,2,1,2,2,2,3,2,0,3,1,3,2,3,3,3]`
- odd family:
  - those two fragment patterns are swapped across `k_block`

Mechanical summary:

```text
frag_swap_rule_matches = 64 / 64
```

That sharpens the boundary:

1. the odd-family `k_block` inversion is already present in the logical
   fragment `tCrB`
2. `recast<BRegister>(tCrB(...))` is not the first bad boundary
3. future runtime-side comparisons should use `frag_raw` / `reg_pre`, not
   `copy_view_raw`, when `n_tile >= 4`

### 2026-04-14 follow-up: payload-independent stage-slot map shows the same swap

I also ran the standalone runtime probe with `stage_slot_low_byte=1`:

```bash
NEMOTRON_NANO_P1_B_PROBE_TIDS=32,48,64,80 \
NEMOTRON_NANO_P1_B_PROBE_STAGE_SLOT_LOW_BYTE=1 \
  ./build-sm120-relwithdebinfo/testing/nano_p1_b_operand_probe \
  > /tmp/nano_p1_b_operand_probe_stage_slot_frag.txt
```

In this mode, `source_row_reg0_pre/source_row_reg1_pre` become a direct
stage-slot low-byte map into the logical fragment.

Observed rule:

- even family:
  - `k_block=0 -> (0x03020100, 0x13121110)`
  - `k_block=1 -> (0x23222120, 0x33323130)`
- odd family:
  - `k_block=0 -> (0x23222120, 0x33323130)`
  - `k_block=1 -> (0x03020100, 0x13121110)`

So the odd-family swap is not a property of the real saved `w1_fp4` payload. It
already exists in the stage-slot -> logical fragment mapping itself.

That moves the root-cause boundary upstream again:

- the remaining bug is in how odd-family logical fragments map stage slots /
  K-blocks
- it is not in the final recast, and not in the particular saved payload bytes

### 2026-04-14 follow-up: swapping odd-family B `k_block`s at MMA time does not fix Phase 3

I tried one tightly-scoped live-kernel experiment in
`runtime/src/backend/fused_moe_prefill/nano_p1_kernel.cuh`:

- derive an `odd_b_kblock_family` bit from the kernel's own `tCsB` coords
- for that family only, feed `cute::gemm(...)` from the opposite
  `tCrB(_, _, k_block)` / `tCrSFB(_, _, k_block)` slice

This kept the edit strictly at the fragment-to-MMA boundary, with staging left
unchanged.

Result after forcing a rebuild and rerunning `nano_p1_mainloop_oracle_test`:

```text
Phase 1 synthetic all-ones PASS
Phase 3 total matches = 4071 / 245760
```

That is slightly *worse* than the known baseline `4074 / 245760`, so the
tracked odd-family virtual swap is not a safe live fix. I reverted that kernel
experiment immediately.

Interpretation:

- the tracked odd-family `k_block` inversion is real
- but swapping B at MMA time is not sufficient to improve end-to-end Phase 3
- therefore the tracked live/runtime reg surface is not, by itself, the full
  causal fix surface

### 2026-04-14 follow-up: alternate B copy selector is not the cause either

I then added a probe-only alternate B copy path in
`testing/backend/nano_p1_b_operand_probe.cu` using:

```c++
sm120_rr_smem_copy_selector_B<
    NanoP1ElementAct,
    NanoP1ElementWeight,
    true>()
```

and captured both the current `byte_tag_reg*_pre` and alternate
`byte_tag_reg*_pre_alt` from the same staged shared-memory contents.

I checked:

1. payload-independent `stage_slot_low_byte=1`
2. the saved real-input `w1_fp4` byte-index payload

Result:

```text
stage_slot_low_byte: byte_tag_reg_pre_alt == byte_tag_reg_pre for all sampled entries
w1_fp4 payload:      alt_diff = 0 / 64
```

So the odd-family inversion is **not** explained by the current
`sm120_rr_smem_copy_selector_B<..., false>()` vs
`sm120_rr_smem_copy_selector_B<..., true>()` choice. At the probe boundary, the
two selector specializations are behaviorally identical.

Updated implication:

- live-kernel odd-family `k_block` swap at MMA time: not a fix
- alternate B copy selector: no effect
- the next useful work is narrower than both of those:
  inspect or probe the mapping from stage slots into logical `tCrB`
  positions beyond the currently sampled live surface, rather than trying
  another kernel rewrite
