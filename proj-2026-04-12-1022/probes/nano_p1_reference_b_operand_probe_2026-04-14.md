# NanoP1 Live Reference B Operand Probe (2026-04-14)

This note records the first successful live flashinfer/CUTLASS tactic-1 B-side
operand probe.

## What Changed

- The probe transport was moved out of the grouped GEMM kernel workspace and
  into `TmaWarpSpecializedGroupedGemmInput`'s own persistent workspace.
- The live SM120 blockscaled array-TMA mainloop now receives that probe pointer
  explicitly through `CollectiveMainloop::Arguments`.
- The flashinfer capture harness now accepts:
  - `NEMOTRON_TRTLLM_OPERAND_PROBE_TIDS=tid0,tid1,...`
  - `NEMOTRON_TRTLLM_ONLY_GEMM1_TACTIC=<id>`

Relevant patched files:

- `flashinfer/.../cutlass/include/cutlass/gemm/collective/nemotron_operand_probe.hpp`
- `flashinfer/.../cutlass/include/cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp`
- `flashinfer/.../cutlass_backend/cutlass_fused_moe_kernels.cuh`
- `flashinfer/.../moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl`
- `flashinfer/.../include/moe_gemm_kernels.h`
- `flashinfer/.../moe_gemm_tma_warp_specialized_input.cu`
- `proj-2026-04-12-1022/trtllm_reference/capture_bf16_gemm1.py`

## Commands

Discovery run to find the real tactic-1 consumer tids:

```bash
NEMOTRON_TRTLLM_OPERAND_PROBE_ANY_THREAD=1 \
  /home/khkramer/src/nemotron-inference/vllm-env-cu128/bin/python \
  proj-2026-04-12-1022/trtllm_reference/capture_bf16_gemm1.py \
  --num-tokens 128 \
  --hidden-size 2688 \
  --inter-size 1920 \
  --num-experts 1 \
  --top-k 1 \
  --seed 12648430 \
  --golden-dir proj-2026-04-12-1022/trtllm_reference/golden_probe_tmp \
  --metadata-path proj-2026-04-12-1022/trtllm_reference/golden_probe_tmp/bf16_gemm1_metadata.json \
  --input-save-dir proj-2026-04-12-1022/trtllm_reference/golden_probe_tmp \
  --flashinfer-src-root /home/khkramer/src/nemotron-inference/vllm-env-cu128/lib/python3.12/site-packages/flashinfer/data/csrc \
  --venv-root /home/khkramer/src/nemotron-inference/vllm-env-cu128
```

Targeted tactic-1 run:

```bash
NEMOTRON_TRTLLM_ONLY_GEMM1_TACTIC=1 \
NEMOTRON_TRTLLM_OPERAND_PROBE_TIDS=32,48,64,80 \
  /home/khkramer/src/nemotron-inference/vllm-env-cu128/bin/python \
  proj-2026-04-12-1022/trtllm_reference/capture_bf16_gemm1.py \
  --num-tokens 128 \
  --hidden-size 2688 \
  --inter-size 1920 \
  --num-experts 1 \
  --top-k 1 \
  --seed 12648430 \
  --golden-dir proj-2026-04-12-1022/trtllm_reference/golden_probe_tmp \
  --metadata-path proj-2026-04-12-1022/trtllm_reference/golden_probe_tmp/bf16_gemm1_metadata.json \
  --input-save-dir proj-2026-04-12-1022/trtllm_reference/golden_probe_tmp \
  --flashinfer-src-root /home/khkramer/src/nemotron-inference/vllm-env-cu128/lib/python3.12/site-packages/flashinfer/data/csrc \
  --venv-root /home/khkramer/src/nemotron-inference/vllm-env-cu128
```

## Key Output

The discovery run proved that the live tactic-1 B consumers are not on
`tid={0,1,128,129}`. The first live entries came from `tid=32..95`.

The targeted tactic-1 run succeeded and produced:

- `tracked_tids=32,48,64,80`
- `k_blocks={0,1}`
- `n_tiles={0..7}`
- `reg_pre == reg_post` for every captured entry

Representative lines:

```text
entry=0  tid=32 k_block=0 n_tile=0 part_c0=(16,0)   local0=(0,0)  stage0_offsets=(0,0)
entry=2  tid=32 k_block=0 n_tile=1 part_c0=(16,8)   local0=(0,4)  stage0_offsets=(4,4)
entry=4  tid=32 k_block=0 n_tile=2 part_c0=(16,32)  local0=(0,8)  stage0_offsets=(8,8)
entry=16 tid=48 k_block=0 n_tile=0 part_c0=(20,0)   local0=(0,1)  stage0_offsets=(1,1)
entry=32 tid=64 k_block=0 n_tile=0 part_c0=(32,0)   local0=(0,0)  stage0_offsets=(0,0)
entry=48 tid=80 k_block=0 n_tile=0 part_c0=(36,0)   local0=(0,1)  stage0_offsets=(1,1)
```

Compactly, for tactic 1:

- `tid=32` and `tid=64` are identical consumers except for the 16-row band shift
  in `part_c0.row`:
  - `part_c0.row = 16` vs `32`
  - `part_c0.col = {0,8,32,40,64,72,96,104}`
  - `local0.col = stage0_offset = {0,4,8,12,16,20,24,28}`
- `tid=48` and `tid=80` follow the same pattern at the odd byte lane:
  - `part_c0.row = 20` vs `36`
  - `local0.col = stage0_offset = {1,5,9,13,17,21,25,29}`

## What This Proves

1. The real live P1 reference kernel is indeed tactic 1:
   `CtaShape128x128x64B_Cluster1x1x1`.

2. The live tactic-1 B contract is explicitly 2D:
   - row bands are separated by 16 rows in `part_c`
   - byte lanes advance by `+4` across `n_tile`
   - odd/even byte lanes are split across different consumer tids

3. The tracked tactic-1 B-side `fp4_shift_B` path is effectively a no-op for
   these captured registers:

   ```text
   reg_pre == reg_post
   ```

4. The live consumer tids that matter for tactic 1 are not the Nano runtime
   probe's original `0/1/128/129` set. Any live reference-vs-runtime comparison
   has to normalize by consumed coordinates, not assume tid identity.

## 2026-04-14 Follow-Up: `fp4_shift_B` Verification

The `reg_pre == reg_post` result is now source-verified, not just probe-observed.

Two cheap falsification checks both passed:

1. The probe samples are taken at distinct program points in the live patched
   CUTLASS path:
   - `RecordBOperandPre(...)` immediately after the `copy(... tCrB_copy_view ...)`
   - `RecordBOperandPost(...)` immediately after `fp4_shift_B(...)`
   in
   `flashinfer/data/cutlass/include/cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp`
2. The selected generic `fp4_shift_B(MMA_Op const&, Tensor&&)` body in
   `flashinfer/data/cutlass/include/cute/atom/mma_traits_sm120.hpp`
   is empty for this path. The non-identity overloads in that file are for
   `16x8x32` specializations, not this tactic-1 `16x8x64` blockscaled path.

So for the live tactic-1 reference kernel, `reg_pre == reg_post` is a
structural property of the selected `fp4_shift_B` specialization, not a probe
artifact.

## 2026-04-14 Follow-Up: Normalized Runtime Diff

The Nano runtime probe was rerun with the same tracked tids:

```bash
NEMOTRON_NANO_P1_B_PROBE_TIDS=32,48,64,80 \
  ./build-sm120-relwithdebinfo/testing/nano_p1_b_operand_probe \
  > /tmp/nano_p1_b_operand_probe_runtime_32_48_64_80.txt
```

After normalizing runtime
`(part_output_col0, part_token_row0)` to reference `part_c0(row,col)`, a
mechanical diff over all shared entries reported:

```text
shared_entries 64
coord_mismatches 0
```

That means the runtime and reference agree on the entire tracked consumer
surface:

- `part_c0`
- `local0/local1`
- `stage0_offsets`
- `n_tile`
- `k_block`

The divergence is therefore downstream of coordinate selection and upstream of
MMA: the runtime is staging the wrong payload into the positions that both
kernels agree should be consumed.

## What This Does Not Yet Prove

- This live reference probe still does not directly label
  `(consumed_source_row, consumed_byte_index)` the way
  `testing/backend/nano_p1_b_operand_probe.cu` does.
- A naive reverse lookup against `input_fp4_permuted.bin` is the wrong search
  space for this B-side probe because the B operand here is FC1 weight data,
  not the activation input dump.
- The full eight-tactic targeted sweep is still unstable with this probe path;
  an all-tactic run hit `cudaErrorIllegalInstruction` after tactic 0. The
  tactic-1-only harness avoids that noise and is the trustworthy path for P1.

## Immediate Next Step

Use this tactic-1 contract as the reference side of the comparison and either:

1. mirror the same `(part_c0, local0, stage0_offset, n_tile, k_block)` surface
   out of the Nano runtime probe, or
2. tag the live reference B input with synthetic row/byte markers so the probe
   yields `consumed_source_row` directly instead of raw packed words.

The key point is that the reference side is no longer blind. We now have a
repeatable live tactic-1 probe and we know exactly which consumer threads to
target.

That normalization step is now complete. The next useful probe is no longer
another B coordinate probe. It is either:

1. a Nano-side SFB scale-tag probe, or
2. a direct rewrite of the Nano B staging scatter using the byte-position-based
   source-row map, followed immediately by Phase 3 BF16 revalidation.

## 2026-04-14 follow-up: raw-byte live diff says the coordinates are right and the regs are wrong

I compared the live tactic-1 B probe against the standalone Nano runtime probe
with a small parser/diff helper:

- `proj-2026-04-12-1022/trtllm_reference/compare_b_operand_probe.py`

Two synthetic source modes were used:

1. `stage_slot_bit=0`
2. `stage_slot_low_byte=1`

In both cases:

```text
shared_entries   = 64
coord_mismatches = 0 / 64
reg_mismatches   = 64 / 64
```

So the tracked live tactic-1 kernel and the standalone Nano runtime agree on:

- `part_c0`
- `local0/local1`
- `stage0_offsets`
- `n_tile`
- `k_block`

but they disagree on **every** pre-shift `reg_pre`.

Representative `stage_slot_bit=0` mismatch:

```text
live:    reg_pre=(0xc0303060,0x1cafa45f)
runtime: reg_pre=(0x18081000,0xffffffff)
```

Representative `stage_slot_low_byte=1` mismatch:

```text
live:    reg_pre=(0x86a4d2c0,0xfe866d6f)
runtime: reg_pre=(0x03020100,0x13121110)
```

The low-byte run matters because it rules out a one-bit probe artifact. The
runtime-side pre-shift words collapse to fixed `0x03020100/0x13121110` and
`0x23222120/0x33323130` patterns, while the live reference varies across
`n_tile` and `k_block`.

## Updated conclusion

The remaining B-side problem is **not** stage-0 B coordinate selection.

It is now localized to the register-copy / fragment-assembly path:

- `make_tiled_copy_B(...)`
- `retile_D(tCrB)`
- `partition_fragment_B(...)`
- or a materially different `TiledMma` / `SmemCopyAtomB` contract

So the next useful local probe is a builder-derived K64 B probe using
`UnifiedRoutedFp4Traits<kP13>` / `TracedP13TiledMma` /
`TracedP13SmemCopyAtomB` from
`runtime/src/backend/fused_moe_prefill/nvfp4_bridge.cuh`.

## 2026-04-14 follow-up: the builder-derived local K64 probe matches our runtime, not the live TRT tactic-1 contract

I implemented that local probe at:

- `testing/backend/nano_p1_reference_b_operand_probe.cu`

It now builds cleanly by mirroring the runtime TU include stack
(`common_helpers.cuh` -> `nvfp4_cute.cuh` -> `nvfp4_bridge.cuh`) instead of
trying to include `nvfp4_bridge.cuh` standalone.

### Commands

Build:

```bash
cmake --build build-sm120-relwithdebinfo --target nano_p1_reference_b_operand_probe --parallel 4
```

Run on the same synthetic live-comparison pattern:

```bash
NEMOTRON_NANO_P1_REFERENCE_B_PROBE_TIDS=32,48,64,80 \
NEMOTRON_NANO_P1_REFERENCE_B_PROBE_STAGE_SLOT_LOW_BYTE=1 \
  ./build-sm120-relwithdebinfo/testing/nano_p1_reference_b_operand_probe \
  > /tmp/nano_p1_reference_b_probe_stage_slot_lowbyte.txt
```

Compare:

```bash
python3 proj-2026-04-12-1022/trtllm_reference/compare_b_operand_probe.py \
  --live proj-2026-04-12-1022/trtllm_reference/golden_probe_stage_slot_lowbyte_rawbytes/operand_probe_tactic1.txt \
  --runtime /tmp/nano_p1_reference_b_probe_stage_slot_lowbyte.txt
```

### Result

Top-line result:

```text
live_entries=64 runtime_entries=32 shared_entries=32
missing_in_runtime=32 missing_in_live=0
coord_mismatches=0/32
k_count_mismatches=32/32
reg_mismatches=32/32
```

Representative mismatch:

```text
live:  tid=32 n_tile=0 k_block=0 reg_pre=(0x86a4d2c0,0xfe866d6f)
local: tid=32 n_tile=0 k_block=0 reg_pre=(0x03020100,0x13121110)
```

The local K64 builder-derived probe only exposes `k_block=0` and its pre-shift
register words are the same fixed synthetic patterns already seen in the
standalone Nano runtime probe. In other words:

- the local builder-derived K64 probe matches our current runtime-side B
  contract, not the live TRT tactic-1 contract
- the missing live-only entries are exactly the `k_block=1` half of the live
  tactic-1 surface
- on the shared `k_block=0` half, the consumer coordinates still match exactly,
  but the structural counts do not:
  - live `k_counts=(2,2,2)`
  - local `k_counts=(1,1,1)`

### Follow-up shape dump

Both probes now print compact tensor shapes alongside `k_counts`.

For the live tactic-1 probe:

```text
shapes=tCrB(16,8,2) tCrBcv(32,4,2) tCsB(32,4,2,4)
```

For the local builder-derived `kP13` probe:

```text
shapes=tCrB(16,8,1) tCrBcv(32,4,1) tCsB(32,4,1,9)
```

What matches:

- `tCrB` leading dimensions: `16 x 8`
- `tCrB_copy_view` leading dimensions: `32 x 4`
- `tCsB_coords` leading dimensions: `32 x 4`

What diverges:

- live has a real extra B `k_block` axis (`... ,2`) at all three layers
- live has 4 pipeline stages on `tCsB_coords`
- local `kP13` has only one B `k_block`
- local `kP13` has 9 pipeline stages on `tCsB_coords`

This is the cleanest structural explanation so far:

- the local `kP13` proxy preserves the same tracked per-thread lane surface as
  the live tactic-1 kernel
- but it is not the same mainloop family, because the B operand tensors have a
  different rank/depth contract before payload values are even considered

### Follow-up: rebuilding the local proxy against `vllm-env-cu128` does not change the result

I also configured a separate repo build that points both runtime and tests at
the same CUTLASS tree used by the live oracle harness:

```bash
cmake -S . -B build-sm120-vllmref \
  -DNEMOTRON_LOCAL_CUTLASS_INCLUDE_DIR=/home/khkramer/src/nemotron-inference/vllm-env-cu128/lib/python3.12/site-packages/flashinfer/data/cutlass/include \
  -DNEMOTRON_TEST_LOCAL_CUTLASS_INCLUDE_DIR=/home/khkramer/src/nemotron-inference/vllm-env-cu128/lib/python3.12/site-packages/flashinfer/data/cutlass/include

cmake --build build-sm120-vllmref --target nano_p1_reference_b_operand_probe --parallel 4
NEMOTRON_NANO_P1_REFERENCE_B_PROBE_TIDS=32,48,64,80 \
NEMOTRON_NANO_P1_REFERENCE_B_PROBE_STAGE_SLOT_LOW_BYTE=1 \
  ./build-sm120-vllmref/testing/nano_p1_reference_b_operand_probe
```

The output is unchanged:

```text
k_counts=(1,1,1)
shapes=tCrB(16,8,1) tCrBcv(32,4,1) tCsB(32,4,1,9)
```

So the local/live structural gap is **not** explained by the repo build using
`.venv-trtllm` while the live oracle uses `vllm-env-cu128`. The local proxy is
structurally different even when built against the same CUTLASS snapshot as the
live oracle.

### Follow-up: exact local proxy dispatch type

The standalone local probe now prints the instantiated dispatch type:

```text
dispatch_policy =
  cutlass::gemm::MainloopSm120ArrayTmaWarpSpecializedBlockScaled<
    9, 3, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1>>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120<3>>
```

So the local `kP13` proxy is not a plain-TMA or wrong-family path. It is
already the same broad **SM120 array-TMA blockscaled** family as the live
reference probe patchpoint. The remaining structural split is narrower:

- local proxy: array-TMA blockscaled, `PipelineStages=9`, scheduler pipeline `3`
- live tactic 1: array-TMA blockscaled consumer surface with `tCsB(...,4)` and
  `k_counts=(2,2,2)`

I also checked the CUTLASS builder source in both envs:

- `.venv-trtllm/.../sm120_blockscaled_mma_builder.inl`
- `vllm-env-cu128/.../sm120_blockscaled_mma_builder.inl`

Those files are byte-identical, and both define:

```text
SchedulerPipelineStageCount = 3
DispatchPolicy = MainloopSm120ArrayTmaWarpSpecializedBlockScaled<PipelineStages, 3, ...>
```

So the next question is no longer "wrong CUTLASS tree?" or "wrong mainloop
family?" It is:

- why the standalone local `TracedP13CollectiveMainloop` resolves to
  `PipelineStages=9` and only one visible B `k_block`, while the live tactic-1
  kernel exposes a 4-stage `tCsB` surface and two B `k_block`s

### Updated conclusion

This is stronger than the earlier "coords right, regs wrong" result.

It means the gap is not just in NanoP1's custom row/byte staging or in a small
bridge helper layered on top of the right CUTLASS builder. Even the closest
local `UnifiedRoutedFp4Traits<kP13>` / `TracedP13TiledMma` reconstruction still
does not reproduce the live TRT tactic-1 B register contract, and the mismatch
is now visible at the tensor-structure level before looking at payload values.
The proxy is useful for lane-surface orientation only. It is not a faithful
stand-in for the live tactic-1 B mainloop.

So the next probe should move back to the live reference path and capture more
of the **structural** contract at the `tCrB_copy_view` boundary, for example:

- `size<2>(tCrB_copy_view)` / the live `K_BLOCK_MAX`
- the selected `SmemCopyAtomB` / `TiledMma` family or an equivalent structural
  signature
- any layout metadata needed to explain why the live tactic-1 path has
  `k_block={0,1}` while the local `kP13` reconstruction only surfaces
  `k_block=0`

## 2026-04-14 follow-up: live dispatch constants are now explicit in the probe dump

I patched the live flashinfer/TRT operand probe transport so the dumped file now
carries the actual dispatch tuple from
`CollectiveMainloop::DispatchPolicy::{Stages,SchedulerPipelineStageCount}`.

### Live tactic-1 rerun

Command:

```bash
NEMOTRON_TRTLLM_ONLY_GEMM1_TACTIC=1 \
NEMOTRON_TRTLLM_OPERAND_PROBE_TIDS=32,48,64,80 \
NEMOTRON_TRTLLM_OPERAND_PROBE_STAGE_SLOT_LOW_BYTE=1 \
  /home/khkramer/src/nemotron-inference/vllm-env-cu128/bin/python \
  proj-2026-04-12-1022/trtllm_reference/capture_bf16_gemm1.py \
    --num-tokens 128 \
    --hidden-size 2688 \
    --inter-size 1920 \
    --num-experts 1 \
    --top-k 1 \
    --seed 12648430 \
    --golden-dir proj-2026-04-12-1022/trtllm_reference/golden_probe_tmp \
    --metadata-path proj-2026-04-12-1022/trtllm_reference/golden_probe_tmp/bf16_gemm1_metadata.json \
    --input-save-dir proj-2026-04-12-1022/trtllm_reference/golden_probe_tmp \
    --flashinfer-src-root /home/khkramer/src/nemotron-inference/vllm-env-cu128/lib/python3.12/site-packages/flashinfer/data/csrc \
    --venv-root /home/khkramer/src/nemotron-inference/vllm-env-cu128
```

Probe header now starts with:

```text
NEMOTRON_B_OPERAND_PROBE ... dispatch=(4,3) ...
entry=0 ... dispatch=(4,3) k_counts=(2,2,2) shapes=tCrB(16,8,2) tCrBcv(32,4,2) tCsB(32,4,2,4) ...
```

So the earlier live `tCsB(...,4)` inference is now confirmed directly by the
live dispatch policy:

- live tactic 1: `dispatch=(4,3)`

### Local proxy rerun with matching machine-readable dispatch

I also updated the standalone local probe to emit the dispatch tuple in the same
format:

```text
nano_p1_reference_b_operand_probe: dispatch=(9,3) dispatch_policy=...
  n_tile=0 k_block=0 dispatch=(9,3) k_counts=(1,1,1) ...
```

### Mechanical diff after the dispatch update

Command:

```bash
NEMOTRON_NANO_P1_REFERENCE_B_PROBE_TIDS=32,48,64,80 \
NEMOTRON_NANO_P1_REFERENCE_B_PROBE_STAGE_SLOT_LOW_BYTE=1 \
  ./build-sm120-relwithdebinfo/testing/nano_p1_reference_b_operand_probe \
  > /tmp/nano_p1_reference_b_probe_stage_slot_lowbyte.txt

python3 proj-2026-04-12-1022/trtllm_reference/compare_b_operand_probe.py \
  --live proj-2026-04-12-1022/trtllm_reference/golden_probe_tmp/operand_probe_tactic1.txt \
  --runtime /tmp/nano_p1_reference_b_probe_stage_slot_lowbyte.txt \
  --show-limit 4
```

Result:

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

### What this settles

This is stronger than the earlier `tCsB(...,4)` / `k_counts` argument:

- the live tactic-1 kernel is explicitly running `dispatch=(4,3)`
- the local `TracedP13CollectiveMainloop` proxy is explicitly running
  `dispatch=(9,3)`
- the two still agree on the consumer coordinate surface
  (`coord_mismatches=0/32`)
- but they diverge on stage depth, visible B `k_block` count, and payload

So the local proxy is not just a wrong-payload stand-in. It is a different
stage-depth contract that happens to share the same consumer lane surface. That
makes it unsafe as a source of live-kernel fixes.

## 2026-04-14 follow-up: shared-storage bytes weaken the "just epilogue carveout" theory

I extended the same probe transport one step further so the live dump header now
prints:

- `dispatch=(...)`
- `shared_storage_bytes=(epilogue,mainloop)`

The fresh headers are:

```text
live:  dispatch=(4,3) shared_storage_bytes=(13312,74752)
local: dispatch=(9,3) shared_storage_bytes=(14336,83968)
```

The local numbers come from:

```text
nano_p1_reference_b_operand_probe: shared_storage_bytes=(14336,83968)
```

and the live numbers from the updated tactic-1 probe header:

```text
NEMOTRON_B_OPERAND_PROBE ... dispatch=(4,3) shared_storage_bytes=(13312,74752) ...
```

I also checked the CUTLASS builder formula in
`sm100_blockscaled_umma_builder.inl` / `sm120_blockscaled_mma_builder.inl`.
`StageCountAutoCarveout` resolves pipeline depth from:

- SMEM capacity
- `carveout_bytes = sizeof(CollectiveEpilogue::SharedStorage)`
- per-stage storage for A/B/SFA/SFB plus the mainloop pipeline

That matters because the new numbers show:

- epilogue carveout only differs by `1024` bytes (`14336 - 13312`)
- but the chosen stage depth differs by `5` (`9` vs `4`)
- and the total mainloop shared-storage footprint differs by `9216` bytes

Per-stage rough averages:

- live:  `74752 / 4  ~= 18688` bytes per stage
- local: `83968 / 9 ~=  9329` bytes per stage

That is too large a gap to write off as "the epilogue carveout is slightly
different." The live kernel is carrying a materially different **per-stage
storage contract**, which matches the other structural facts:

- live exposes `k_counts=(2,2,2)` and `tCsB(...,4)`
- local proxy exposes `k_counts=(1,1,1)` and `tCsB(...,9)`

So the next question is not "why did `StageCountAutoCarveout` round down a bit
differently?" It is "what live blockscaled array-TMA mainloop contract doubles
the effective B-side stage payload relative to the local proxy?"

## 2026-04-14 follow-up: the live/local split extends into the B-scale / SFB contract

I extended the same probe record with compact SFB structure:

- `sf_counts=(tCrSFB_k_blocks,tCrSFB_copy_view_k_blocks,tCsSFB_coord_k_blocks)`
- `sf_shapes=tCrSFB(... ) tCrSFBcv(... ) tCsSFB(... )`

### Live tactic-1 SFB surface

The updated live tactic-1 probe now reports:

```text
sf_counts=(2,2,2)
sf_shapes=tCrSFB(64,8,2) tCrSFBcv(128,4,2) tCsSFB(128,4,2,4)
```

Representative entry:

```text
entry=0 tid=32 ... dispatch=(4,3)
  k_counts=(2,2,2) shapes=tCrB(16,8,2) tCrBcv(32,4,2) tCsB(32,4,2,4)
  sf_counts=(2,2,2) sf_shapes=tCrSFB(64,8,2) tCrSFBcv(128,4,2) tCsSFB(128,4,2,4)
```

### Local `kP13` proxy SFB surface

The standalone local probe now reports:

```text
sf_counts=(64,64,1)
sf_shapes=tCrSFB(64,8,64) tCrSFBcv(128,4,64) tCsSFB(128,4,1,9)
```

Representative entry:

```text
n_tile=0 k_block=0 dispatch=(9,3)
  k_counts=(1,1,1) shapes=tCrB(16,8,1) tCrBcv(32,4,1) tCsB(32,4,1,9)
  sf_counts=(64,64,1) sf_shapes=tCrSFB(64,8,64) tCrSFBcv(128,4,64) tCsSFB(128,4,1,9)
```

### Updated diff

The refreshed `compare_b_operand_probe.py` report is now:

```text
coord_mismatches=0/32
dispatch_mismatches=32/32
k_count_mismatches=32/32
sf_count_mismatches=32/32
reg_mismatches=32/32
```

### What this settles

This is stronger than the earlier dense-B-only conclusion.

The live/local divergence is not just:

- dense B `k_block` count
- dense B payload
- dispatch stage depth

It also extends to the scale fragment contract itself:

- live SFB surface is compact and 2-way in `k_block`
- local proxy SFB surface is drastically wider and only exposes one source-side
  `tCsSFB` `k_block`

So the local `kP13` proxy is not merely a wrong dense-B reconstruction. It is
also a wrong B-scale / SFB reconstruction. That makes it even less suitable as
the basis for a Nano kernel rewrite.

## 2026-04-14 follow-up: live/local SFB stage layout diverges even when TV dims match

I extended both the live flashinfer probe dump and the standalone local probe
dump so `sf_layout` now carries:

```text
sf_layout=(
  sfb_stage_elems,
  sfb_total_elems,
  sfb_stage_shape_dim0,
  sfb_stage_shape_dim1,
  layout_sfb_tv_dim0,
  layout_sfb_tv_dim1)
```

The local proxy still builds and runs via:

```bash
cmake --build build-sm120-relwithdebinfo --target nano_p1_reference_b_operand_probe --parallel 4
NEMOTRON_NANO_P1_REFERENCE_B_PROBE_TIDS=32,48,64,80 \
NEMOTRON_NANO_P1_REFERENCE_B_PROBE_STAGE_SLOT_LOW_BYTE=1 \
  ./build-sm120-relwithdebinfo/testing/nano_p1_reference_b_operand_probe \
  > /tmp/nano_p1_reference_b_probe_stage_slot_lowbyte.txt
```

The live tactic-1 capture was rerun successfully after patching both the
array-TMA and non-array-TMA blockscaled probe call sites:

```bash
NEMOTRON_TRTLLM_ONLY_GEMM1_TACTIC=1 \
NEMOTRON_TRTLLM_OPERAND_PROBE_TIDS=32,48,64,80 \
NEMOTRON_TRTLLM_OPERAND_PROBE_STAGE_SLOT_LOW_BYTE=1 \
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

### New `sf_layout` result

Live tactic-1:

```text
sf_layout=(1024,4096,128,128,256,128)
```

Local `kP13` proxy:

```text
sf_layout=(512,4608,128,64,256,128)
```

Representative entries:

```text
live:  tid=32 n_tile=0 k_block=0 ... sf_layout=(1024,4096,128,128,256,128)
local: tid=32 n_tile=0 k_block=0 ... sf_layout=(512,4608,128,64,256,128)
```

Updated mechanical diff:

```text
coord_mismatches=0/32
dispatch_mismatches=32/32
k_count_mismatches=32/32
sf_count_mismatches=32/32
sf_layout_mismatches=32/32
reg_mismatches=32/32
```

### What this tightens

This is the cleanest scale-side localization so far:

- the live/local `layout_sfb_tv` dims still match exactly: `(256,128)`
- the live/local `SmemLayoutSFB` stage shape does **not** match:
  - live stage shape: `(128,128)`
  - local stage shape: `(128,64)`
- the live per-stage SFB footprint doubles the local one:
  - live `sfb_stage_elems=1024`
  - local `sfb_stage_elems=512`

So the remaining divergence is no longer just "different stage count" or
"different copy-view counts." The live tactic-1 kernel is carrying a different
`SmemLayoutSFB` stage contract, specifically a doubled K-side stage shape,
while preserving the same SFB TV layout.

That further demotes the local `kP13` proxy: it is not just the wrong consumer
surface shape, and not just the wrong `k_block` count. It is the wrong scale
stage layout.

### Source check: the generic SM120 dense blockscaled builder predicts the local result, not the live one

I followed the scale-layout construction in
`cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl`.

Relevant facts from source:

- dense `nv_float4_t<T>` blockscaled GEMM uses `SfVectorSize = 16`
- `Blk_MN = 128`, `Blk_SF = 4`
- `SmemLayoutAtomSFB` is built directly from:
  - `TileShape_MNK`
  - `SFVectorSize`
  - `MMA_NSF`

Those constants are enough to explain the local proxy's stage shape:

```text
generic dense builder -> sf_layout stage shape = (128,64)
```

which is exactly what the standalone local `kP13` probe reports:

```text
sf_layout=(512,4608,128,64,256,128)
```

The live tactic-1 probe still reports:

```text
sf_layout=(1024,4096,128,128,256,128)
```

So the live tactic-1 kernel is **not** behaving like the plain generic SM120
dense blockscaled builder contract we can reconstruct locally. That is a useful
negative result: the next live-reference investigation should focus on whatever
TRT/flashinfer specialization or layout override yields the doubled
`SmemLayoutSFB` stage shape, not on re-deriving the generic builder again.

## 2026-04-14 follow-up: `sf_atom` proves the split starts at the scale-layout atom

I extended both the live tactic-1 probe and the standalone local probe to emit
the scale-layout atom tuple:

```text
sf_atom=(SFVecSize, SmemLayoutAtomSFB dim0, SmemLayoutAtomSFB dim1)
```

Fresh runs:

- local probe (`stage_slot_low_byte=1`):

  ```text
  sf_atom=(16,128,64)
  sf_layout=(512,4608,128,64,256,128)
  ```

- live tactic-1 flashinfer capture:

  ```text
  sf_atom=(16,128,128)
  sf_layout=(1024,4096,128,128,256,128)
  ```

Updated mechanical diff from
`proj-2026-04-12-1022/trtllm_reference/compare_b_operand_probe.py`:

```text
live_entries=64 runtime_entries=32 shared_entries=32
missing_in_runtime=32 missing_in_live=0
coord_mismatches=0/32
dispatch_mismatches=32/32
k_count_mismatches=32/32
sf_count_mismatches=32/32
sf_atom_mismatches=32/32
sf_layout_mismatches=32/32
reg_mismatches=32/32
```

Representative mismatch:

```text
live:    tid=32 n_tile=0 k_block=0 dispatch=(4,3) k_counts=(2,2,2) sf_counts=(2,2,2)
         sf_atom=(16,128,128) sf_layout=(1024,4096,128,128,256,128)
runtime: tid=32 n_tile=0 k_block=0 dispatch=(9,3) k_counts=(1,1,1) sf_counts=(64,64,1)
         sf_atom=(16,128,64)  sf_layout=(512,4608,128,64,256,128)
```

### What this tightens

This removes the remaining ambiguity in the earlier `sf_layout` result.

- `SFVecSize` still matches: `16`
- `layout_sfb_tv` still matches: `(256,128)`
- but the live/local split is already present in `SmemLayoutAtomSFB` itself:
  - live atom: `(128,128)`
  - local atom: `(128,64)`

So the live tactic-1 kernel is not just carrying a different derived
`SmemLayoutSFB` stage shape. It is selecting a different **scale-layout atom
contract**.

That pushes the next investigation one level lower and narrows it further:

- inspect the live tactic-1 `SmemLayoutAtomSFB` specialization directly
- inspect the live tactic-1 `SmemCopyAtomSFB` specialization directly
- stop spending time on local generic-builder reconstructions, because the
  live/local split is already upstream of the derived stage-layout math

## 2026-04-14 correction: `64B` is a byte K label, so the old `kP13` local proxy was wrong

I traced TRT's SM120 tile-shape dispatch in:

- `.../moe_gemm/moe_gemm_template_dispatch_tma_ws.h`

The key source is the `SHAPE_CASE` macro:

```cpp
constexpr int KtileBytes =
    (K * 8) / cutlass::sizeof_bits<
        typename kernels::cutlass_kernels::TllmToCutlassTypeAdapter<T>::type>::value;
using TileShape = Shape<_M, _N, Int<KtileBytes>>;
```

For FP4 (`sizeof_bits<T> == 4`), that means:

- `CtaShape128x128x64B` -> `TileK = 128`
- `CtaShape128x128x128B` -> `TileK = 256`

So the earlier standalone local `kP13` proxy
`TracedP13MmaTileShape = (128,128,64)` was never a tactic-1-equivalent proxy.
It was off by 2x in K from the start.

I then repointed `testing/backend/nano_p1_reference_b_operand_probe.cu` to the
existing local `kP12` / `TracedP5CollectiveMainloop` path
(`TracedP5MmaTileShape = (128,128,128)`) and rebuilt against the same
`vllm-env-cu128` CUTLASS snapshot as the live oracle.

Corrected local probe header:

```text
dispatch=(4,3)
shared_storage_bytes=(14336,74752)
```

Representative corrected local entries:

```text
n_tile=0 k_block=0 dispatch=(4,3) k_counts=(2,2,2)
sf_shapes=tCrSFB(64,8,128) tCrSFBcv(128,4,128) tCsSFB(128,4,2,4)
sf_atom=(16,128,128) sf_layout=(1024,4096,128,128,256,128)

n_tile=0 k_block=1 dispatch=(4,3) k_counts=(2,2,2)
sf_shapes=tCrSFB(64,8,128) tCrSFBcv(128,4,128) tCsSFB(128,4,2,4)
sf_atom=(16,128,128) sf_layout=(1024,4096,128,128,256,128)
```

Updated mechanical diff versus the live tactic-1 flashinfer capture:

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

### What changed in the diagnosis

This collapses the earlier live/local structural gap.

The corrected local proxy now matches the live tactic-1 kernel on the
load-bearing structural contract:

- `dispatch=(4,3)`
- `k_counts=(2,2,2)`
- `sf_atom=(16,128,128)`
- `sf_layout=(1024,4096,128,128,256,128)`
- `tCsSFB(128,4,2,4)`
- the full tracked consumer coordinate surface

The remaining `sf_count_mismatches=64/64` are a probe-schema mismatch, not a
new structural split:

- live reports logical scale `k_block` counts as `(2,2,2)`
- the corrected local probe's `tCrSFB` / `tCrSFBcv` report flattened fragment
  extent `128`
- but the load-bearing coord tensor count `tCsSFB(...,2,4)` already matches

So after correcting the K-width mistake, the remaining live/local divergence is
no longer stage depth, tile K, or scale-layout atom selection. It is the
payload in `reg_pre`.

That is the real narrowing we wanted:

- structural contract now matches
- remaining mismatch is packed-byte / nibble payload assembly

The next useful step is no longer more live structural archaeology. It is to
use the corrected local `kP12` proxy and the Nano runtime probe to localize the
packed-B register payload mismatch directly.

## 2026-04-14 follow-up: live `w1_fp4` K=128 captures collapse the B mismatch to an odd-family `k_block` swap

I added a small harness override in
`proj-2026-04-12-1022/trtllm_reference/capture_bf16_gemm1.py` so the live
flashinfer tactic-1 path can run with raw `w1_fp4` bytes set to either:

- `row_index`
- `byte_index`

That matters because the older synthetic stage-slot bit overwrites were not a
raw-byte oracle. They were being crushed by FP4 semantics.

I first ran those overrides at the default `hidden_size=256`. That produced an
apparent `+64` byte shift on the live side, but that turned out to be a false
lead: with `TileK=128`, `hidden_size=256` lets the live kernel observe the
second `k_base` iteration.

The apples-to-apples captures are:

- `proj-2026-04-12-1022/trtllm_reference/golden_probe_w1_row_index_k128/`
- `proj-2026-04-12-1022/trtllm_reference/golden_probe_w1_byte_index_k128/`

Those runs use `--hidden-size 128`, so tactic 1 only sees one K tile.

I added
`proj-2026-04-12-1022/trtllm_reference/compare_w1_runtime_b_probe.py`
to compare the live `operand_probe_tactic1.txt` output against the local
`nano_p1_b_operand_probe` output on the common key:

- `(tid, n_tile, k_block)`
- `part_c0`
- `local0`
- `stage0_offset0`
- `reg_pre`

Command:

```bash
python3 proj-2026-04-12-1022/trtllm_reference/compare_w1_runtime_b_probe.py \
  --live proj-2026-04-12-1022/trtllm_reference/golden_probe_w1_byte_index_k128/operand_probe_tactic1.txt \
  --runtime /tmp/nano_p1_b_operand_probe_runtime_32_48_64_80_posmap_current.txt \
  --show-mismatches
```

Result:

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

What that settles:

- the live and runtime B consumer coordinate surface now matches exactly
- the remaining mismatch is not row placement
- the remaining mismatch is not a global `+64` byte shift
- the remaining mismatch is exactly this:
  - even family (`tid=32,64`, `local_col0={0,4,8,...}`, `stage0_offset0={0,4,8,...}`): live and runtime match
  - odd family (`tid=48,80`, `local_col0={1,5,9,...}`, `stage0_offset0={1,5,9,...}`): runtime's `reg_pre` windows are swapped between `k_block=0` and `k_block=1`

Representative mismatch:

```text
tid=48 n_tile=0 local_col0=1 stage0_offset0=1
runtime k_block=0 reg_pre=(0x23222120,0x33323130)
live    k_block=0 reg_pre=(0x03020100,0x13121110)
runtime k_block=1 reg_pre=(0x03020100,0x13121110)
live    k_block=1 reg_pre=(0x23222120,0x33323130)
```

So a virtual odd-family `k_block` swap explains the entire remaining tracked
live/runtime B payload gap.

## 2026-04-14 follow-up: first stage-slot signature does not implicate the raw B fill yet

I also extended `testing/backend/nano_p1_b_operand_probe.cu` to emit a compact
copy-view slot signature for each tracked entry.

That did not produce a new split yet. For the tracked first anchor positions:

- `tid=32, n_tile=0`: `copy_view_stage_byte_slots=[0,...]`
- `tid=48, n_tile=0`: `copy_view_stage_byte_slots=[0,...]`
- `tid=32, n_tile=1`: `copy_view_stage_byte_slots=[2,...]`
- `tid=48, n_tile=1`: `copy_view_stage_byte_slots=[2,...]`

The signature still collapses to a single repeated byte slot per tracked entry,
so it does not yet prove that the divergence is in the raw stage-byte fill.

That pushes the next useful probe boundary downstream of the coarse staged-byte
surface and closer to:

- the B copy-view / retile path
- or the mapping from copied B values into the final `reg_pre` payload
