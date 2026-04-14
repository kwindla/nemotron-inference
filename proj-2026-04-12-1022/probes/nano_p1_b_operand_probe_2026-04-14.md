# NanoP1 B Operand Probe (2026-04-14)

This note records the first standalone NanoP1 B-side operand probe added in
this session.

## What Was Added

- Diagnostic binary:
  `testing/backend/nano_p1_b_operand_probe.cu`
- Build target:
  `nano_p1_b_operand_probe`

The probe instantiates the live NanoP1 `TiledMma`, `SmemLayoutB`, and
`SmemCopyAtomB` contract and dumps, for tracked threads `tid={0,1,128,129}`:

- the `part_c` token row seen by the thread at `part_c(_, 0, n_tile)(0)`
- the `tCsB` stage-0 logical coordinate consumed by the copy view
- the current source-side B permutation result
  `NanoP1PermuteBSourceRow(local_row)`
- the stage-0 logical offset from `NanoP1SmemLayoutB{}(_, _, 0)`
- the actual source-row tag observed at that stage-0 offset after staging
- the actual byte-index tag observed at that stage-0 offset after staging
- a byte-tag register snapshot before/after `cute::fp4_shift_B`

## Command

```bash
touch runtime/src/backend/fused_moe_prefill.cu
cmake --build build-sm120-relwithdebinfo --target nano_p1_b_operand_probe --parallel $(nproc)
./build-sm120-relwithdebinfo/testing/nano_p1_b_operand_probe
```

## Key Output

The probe reports:

- `tid=0`: `14 / 16` row mismatches
- `tid=1`: `16 / 16` row mismatches
- `tid=128`: `16 / 16` row mismatches
- `tid=129`: `16 / 16` row mismatches
- total: `62 / 64` mismatches

Representative lines:

```text
tid=1   n_tile=1 k_block=0 part_token_row0=10 consumed_source_row0=1  consumed_byte_index0=2  ROW_MISMATCH
tid=129 n_tile=0 k_block=0 part_token_row0=18 consumed_source_row0=1  consumed_byte_index0=1  ROW_MISMATCH
tid=0   n_tile=2 k_block=0 part_token_row0=32 consumed_source_row0=0  consumed_byte_index0=4  ROW_MISMATCH
```

## What This Proves

1. The current B-side fix in
   `runtime/src/backend/fused_moe_prefill/nano_p1_kernel.cuh`
   is only a **row-only permutation**:
   `source_row = NanoP1PermuteBSourceRow(row)`.

2. That is not the contract the NanoP1 copy-view is consuming.
   For the tracked threads, the B consumer coordinates depend on both:
   - the logical row-like component (`local_row0`)
   - the per-atom-call / per-fragment position (`n_tile`, visible here via
     `local_col0` / `consumed_byte_index0`)

3. So the remaining NanoP1 Phase 3 bug is not "pick a better row swap".
   It is a **2D B staging mismatch** keyed at least by `(row, mf)` and
   likely by the full `(row, col)` consumer position.

4. The specific open question from the 2026-04-14 handoff is now answered
   for the tracked consumer coordinate:
   - `tid=1, mf=1` is currently consuming `source_row=1`, not token `10`
     and not token `18`.
   - `tid=129, mf=0` is currently consuming `source_row=1`, not token `18`.

5. For the tracked coordinates, the desired row can be read directly from the
   probe output as a function of the consumed byte index. The first consumed
   coordinate follows:

   ```text
   desired_source_row
     = 2 * local_row0
     + 16 * (consumed_byte_index0 & 1)
     +  8 * ((consumed_byte_index0 >> 1) & 1)
     + 32 * (consumed_byte_index0 >> 2)
   ```

   This matches all tracked cases:
   - `(local_row0=1, byte=2)  -> 10`
   - `(local_row0=1, byte=1)  -> 18`
   - `(local_row0=0, byte=4)  -> 32`

   So the B staging bug is not merely "wrong token row." It is "the token-row
   choice is encoded by byte position within the staged row, and the current
   code ignores that."

## What This Does *Not* Yet Prove

- It does **not** yet identify the exact byte/nibble ordering inside
  every element of `tCrB_cv`.
- It does **not** yet compare against the flashinfer reference kernel's
  internal B fragment state.
- The per-atom byte-tag register snapshot is still only a secondary hint. The
  trusted signal is the direct `(consumed_source_row0, consumed_byte_index0)`
  tag read at the consumer offsets.

## Immediate Next Step

Mirror the same tagged-operand probe against the flashinfer reference types.
The cleanest path is likely an env-gated helper in the patched flashinfer
TMA-warp-specialized launcher that instantiates the exact `CollectiveMainloop`
types and runs the same synthetic B-stage tag check.

If the reference emits a different `(source_row, byte_index)` contract than
the one above, we can rewrite the NanoP1 B staging loop to match it. If it
matches, the next bug is downstream in the B copy-view / shift / atom-assembly
path rather than the initial staging scatter.
