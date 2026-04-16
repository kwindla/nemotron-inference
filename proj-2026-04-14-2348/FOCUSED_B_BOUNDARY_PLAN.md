# Focused Attempt: NanoP1 B-Boundary Debug

This is a narrower replacement for the broad checkpoint-ladder rewrite in
`proj-2026-04-14-2348/PLAN.md`.

The goal is not to redesign the whole kernel. The goal is to explain the first
wrong consumed B operand byte on the live FlashInfer tactic-1 path, then make
the smallest production fix that removes that discrepancy.

## Why this attempt

Current local evidence says the bug is narrower than "the whole NanoP1 mainloop
is wrong":

- `nano_p1_mainloop_oracle_test` currently fails Phase 3 at **241686**
  bitwise mismatches and **4074** bitwise matches vs flashinfer.
- `testing/backend/nano_p1_b_operand_probe.cu` with the real
  `input_fp4_permuted.bin` and `use_position_map=1` reaches
  `total_row_mismatches=0/128`.
- The same probe with `row_only_permute` stays at `total_row_mismatches=62/128`.
- So B-side **row mapping is carrying real signal already**.
- What is still wrong is the **byte / nibble / family mapping within a B row**,
  or the immediate `copy -> fp4_shift_B` interpretation after staging.

This attempt treats the live FlashInfer tactic-1 B-operand path as the oracle
and focuses only on the B boundary:

`smem_B stage-0 bytes -> tCrB_copy_view -> fp4_shift_B -> first consumed regs`

## Non-goals

Do not spend time on these during this attempt:

- full-kernel rewrites
- accumulator dumps
- whole-smem checkpoint ladders
- deleting `NanoP1PermuteBSourceRow` or `NanoP1SourceRowForBStageOffset`
- vLLM parity as a primary correctness oracle
- full `ctest` on every probe-only edit

If a step does not directly improve understanding of the B boundary, skip it.

## Frozen reference

Use these exact local artifacts as the pinned reference:

- FlashInfer source tree:
  `vllm-env-cu128/lib/python3.12/site-packages/flashinfer/data/csrc`
- Nano bucket inputs:
  `proj-2026-04-12-1022/trtllm_reference/golden_nano_k2688/`
- Exact shape from `bf16_gemm1_metadata.json`:
  - `num_tokens=128`
  - `hidden_size=2688`
  - `inter_size=1920`
  - `num_experts=1`
  - `top_k=1`
  - `seed=12648430`
- FlashInfer target tactic:
  - `gemm1 tactic_id=1`
  - `CtaShape128x128x64B_Cluster1x1x1`
  - `swap_ab=false`

Do not switch trees or regenerate with a different env during this attempt.

## Files in play

- Production kernel:
  `runtime/src/backend/fused_moe_prefill/nano_p1_kernel.cuh`
- Current B helpers:
  `runtime/src/backend/fused_moe_prefill/nvfp4_bridge.cuh`
- Runtime probe:
  `testing/backend/nano_p1_b_operand_probe.cu`
- Live FlashInfer capture harness:
  `proj-2026-04-12-1022/trtllm_reference/capture_bf16_gemm1.py`
- Existing runtime/live compare helper:
  `proj-2026-04-12-1022/trtllm_reference/compare_w1_runtime_b_probe.py`

Do not use `testing/backend/nano_p1_reference_b_operand_probe.cu` as a primary
oracle for this attempt. The live FlashInfer tactic-1 path is stronger.

## Success criteria

This attempt is successful if it does all of the following:

1. Localizes the first wrong B-byte family to one concrete rule difference.
2. Produces a probe-only change that improves runtime/live agreement without
   regressing `total_row_mismatches=0/128`.
3. Ports exactly that change into production and moves Phase 3 materially below
   `241686` bitwise mismatches.

If the production mismatch count does not move after a probe-proven fix, stop
and reassess instead of stacking more edits.

## Baseline commands

### 1. Reconfirm current Phase 3 baseline

```bash
build-sm120-relwithdebinfo/testing/nano_p1_mainloop_oracle_test
```

Expected current outcome:

- Phase 1 PASS
- Phase 2 DEFERRED
- Phase 3 primary gate FAIL
- `mismatches=241686`

### 2. Reconfirm current runtime B probe baseline

Save the runtime probe output to a file. This file becomes the baseline for all
later comparisons in this attempt.

```bash
mkdir -p proj-2026-04-14-2348/archive/focused_b_attempt

NEMOTRON_NANO_P1_B_INPUT_FILE=proj-2026-04-12-1022/trtllm_reference/golden_nano_k2688/input_fp4_permuted.bin \
NEMOTRON_NANO_P1_B_PROBE_USE_INPUT_FILE=1 \
NEMOTRON_NANO_P1_B_PROBE_USE_POSITION_MAP=1 \
NEMOTRON_NANO_P1_B_INPUT_ROWS=128 \
NEMOTRON_NANO_P1_B_PACKED_ROW_BYTES=1344 \
build-sm120-relwithdebinfo/testing/nano_p1_b_operand_probe \
  | tee proj-2026-04-14-2348/archive/focused_b_attempt/runtime_probe_baseline.txt
```

Expected current outcome:

- `total_row_mismatches=0/128`
- `tid=0` `copy_view_byte_mismatches` sum stays at `462`

Also capture the row-only comparison once so future regressions are obvious:

```bash
NEMOTRON_NANO_P1_B_INPUT_FILE=proj-2026-04-12-1022/trtllm_reference/golden_nano_k2688/input_fp4_permuted.bin \
NEMOTRON_NANO_P1_B_PROBE_USE_INPUT_FILE=1 \
NEMOTRON_NANO_P1_B_INPUT_ROWS=128 \
NEMOTRON_NANO_P1_B_PACKED_ROW_BYTES=1344 \
build-sm120-relwithdebinfo/testing/nano_p1_b_operand_probe \
  | tee proj-2026-04-14-2348/archive/focused_b_attempt/runtime_probe_row_only.txt
```

Expected current outcome:

- `total_row_mismatches=62/128`

## Main attempt

### Step A: Capture the live FlashInfer tactic-1 B probe on the exact Nano bucket

Run the existing capture harness on the pinned shape, tactic 1 only, with the
operand probe enabled and tracked tids aligned to the runtime probe.

```bash
mkdir -p proj-2026-04-14-2348/archive/focused_b_attempt/live_capture

NEMOTRON_HARNESS_M=128 \
NEMOTRON_HARNESS_K=2688 \
NEMOTRON_HARNESS_N=1920 \
NEMOTRON_HARNESS_E=1 \
NEMOTRON_HARNESS_TOPK=1 \
NEMOTRON_HARNESS_SEED=12648430 \
NEMOTRON_TRTLLM_ONLY_GEMM1_TACTIC=1 \
NEMOTRON_TRTLLM_OPERAND_PROBE_TIDS=0,1,128,129 \
bash proj-2026-04-12-1022/trtllm_reference/run_capture.sh \
  --golden-dir proj-2026-04-14-2348/archive/focused_b_attempt/live_capture
```

This should produce:

- `bf16_gemm1_tactic1.bin`
- `operand_probe_tactic1.txt`

Sanity checks:

- the metadata should still point at `vllm-env-cu128`
- the operand probe file should report `tracked_tids=0,1,128,129`
- the BF16 dump should still be byte-identical to the pinned tactic-1 dump

### Step B: Compare live FlashInfer vs runtime on the same tracked entries

Use the existing compare helper first, even though it is incomplete. It already
compares `(tid, n_tile, k_block)`-keyed coordinates plus `reg_pre`.

```bash
python3 proj-2026-04-12-1022/trtllm_reference/compare_w1_runtime_b_probe.py \
  --live proj-2026-04-14-2348/archive/focused_b_attempt/live_capture/operand_probe_tactic1.txt \
  --runtime proj-2026-04-14-2348/archive/focused_b_attempt/runtime_probe_baseline.txt \
  --show-mismatches \
  | tee proj-2026-04-14-2348/archive/focused_b_attempt/live_vs_runtime_compare.txt
```

Interpretation:

- If coordinates differ, do **not** touch production. Fix the probe contract or
  coordinate interpretation first.
- If coordinates match but `reg_pre` differs, the bug is between staged bytes
  and the first consumed B registers.
- If `reg_pre` matches but Phase 3 still fails badly, shift attention to
  `fp4_shift_B`, scale staging, or later mainloop state.

### Step C: Re-run synthetic family probes before changing production

Before guessing at a real-data transform, use synthetic patterns to explain
which stage-slot families each thread actually consumes.

Run these on both the live FlashInfer path and the runtime probe:

- `stage_slot_low_byte`
- `stage_slot_bit=0`
- `stage_slot_bit=1`
- `stage_slot_bit=2`
- `stage_slot_bit=3`
- `stage_slot_bit=4`
- `stage_slot_bit=5`
- `stage_slot_bit=6`
- `stage_slot_bit=7`

For FlashInfer, use:

- `NEMOTRON_TRTLLM_OPERAND_PROBE_STAGE_SLOT_LOW_BYTE=1`
- `NEMOTRON_TRTLLM_OPERAND_PROBE_STAGE_SLOT_BIT=<n>`

For runtime, use the matching probe envs already supported by
`testing/backend/nano_p1_b_operand_probe.cu`.

The point of this step is not to pass a test. The point is to answer:

- do runtime and live agree on the consumed stage-slot family?
- is the disagreement per-`k_block`?
- is it isolated to odd/even `local_col0` families?
- is it consistent with nibble swap, byte-pair swap, or adjacent-family swap?

Do not proceed to production edits until the mismatch can be described in one
sentence.

### Step D: Add probe-only candidate transforms

If Step C identifies a small family of plausible transforms, add them as
**probe-only** env-gated options in `testing/backend/nano_p1_b_operand_probe.cu`.

Allowed candidate transforms:

- swap high / low nibble within the selected packed byte
- swap adjacent source-byte pairs inside a 32-bit chunk
- swap `k_block 0 <-> 1` for one family only
- apply a family-specific byte-index remap based on `local_col0`
- apply a stage-slot-parity remap if and only if Step C proves that parity rule

Not allowed:

- editing the production kernel yet
- adding a giant generated lookup table
- changing row mapping while `total_row_mismatches` is already `0/128`

Each candidate transform is judged by four numbers:

1. `total_row_mismatches`
2. total `copy_view_byte_mismatches`
3. live/runtime `reg_pre` mismatches
4. Phase 3 BF16 mismatches after the production port, if it gets that far

If a transform does not improve the first three, delete it and move on.

### Step E: Only then port the winning rule into production

Once one probe-only rule clearly improves runtime/live B-boundary agreement,
port exactly that rule into:

- `runtime/src/backend/fused_moe_prefill/nano_p1_kernel.cuh`

Do not delete helpers in this step. Keep the row-mapping helper live until
Phase 3 is actually green.

After any production `.cuh` edit:

```bash
touch runtime/src/backend/fused_moe_prefill.cu
cmake --build build-sm120-relwithdebinfo --parallel $(nproc)
```

Then run:

```bash
build-sm120-relwithdebinfo/testing/nano_p1_mainloop_oracle_test
```

Success for this step is not necessarily "green". Success is a **material**
drop below `241686` mismatches that is consistent with the probe improvement.

## Escalation path

Escalate to a different approach only if one of these happens:

1. Live/runtime coordinates cannot be made to agree.
2. Synthetic family probes do not isolate any stable rule difference.
3. Probe-only candidate transforms never improve `reg_pre` agreement.
4. A probe-proven B-boundary fix does not move Phase 3 at all.

If any of those happen, stop the B-boundary attempt and switch to a substitution
harness:

- hardwire the live FlashInfer `smem_B` or `tCrB` payload for one CTA
- replace one ingredient at a time
- do not continue guessing at source-byte formulas

## Deliverables for this attempt

At the end of the attempt, record:

- the baseline runtime probe output
- the live FlashInfer operand probe output
- the runtime/live compare output
- a one-sentence description of the first wrong B-byte rule
- either:
  - the minimal production fix that moved Phase 3, or
  - the explicit stop condition that forced escalation
