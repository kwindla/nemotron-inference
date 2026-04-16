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

## 2026-04-14 Follow-Up: Normalized Tactic-1 Diff

That live reference probe now exists, and the first normalized comparison is
decisive.

Runtime probe rerun:

```bash
NEMOTRON_NANO_P1_B_PROBE_TIDS=32,48,64,80 \
  ./build-sm120-relwithdebinfo/testing/nano_p1_b_operand_probe \
  > /tmp/nano_p1_b_operand_probe_runtime_32_48_64_80.txt
```

Reference probe file:

- `proj-2026-04-12-1022/trtllm_reference/golden_probe_tmp/operand_probe_tactic1.txt`

Normalization rule:

- runtime `part_output_col0` == reference `part_c0.row`
- runtime `part_token_row0` == reference `part_c0.col`

A mechanical parser diff over the shared `64` `(tid, n_tile, k_block)` entries
reported:

```text
shared_entries 64
coord_mismatches 0
```

So for the tracked live tactic-1 tids, the Nano runtime and the flashinfer
reference agree exactly on:

- consumed `part_c0`
- consumed `local0/local1`
- consumed `stage0_offset0/stage0_offset1`

What still disagrees is the payload written to those positions. For example,
runtime `tid=32` matches the reference coordinate surface
`part_c0=(16,{0,8,32,40,...})`, `local0.col={0,4,8,12,...}`,
`stage0_offset={0,4,8,12,...}`, but the staged row tags still decode as
`0,0,4,6,8,10,12,14` instead of the desired token rows
`0,8,32,40,64,72,96,104`.

This closes the main ambiguity from the earlier sessions:

1. The remaining B-side bug is not a mismatch in the consumer coordinate
   contract.
2. The remaining B-side bug is a staging-scatter payload bug:
   the Nano kernel is writing the wrong token bytes into the right stage-0
   shared-memory positions.
3. Any actual fix now needs to focus on the B staging scatter itself,
   and likely the paired SFB scale staging, not on another `tCsB`/`part_c`
   reinterpretation.

## 2026-04-14 Follow-Up: Validated Physical-Offset Decode

The standalone probe now has two additional hard results.

### Probe transport fix

The first `use_position_map=1` validation was invalid because the probe was
overwriting the shared-memory source-row tags with byte-index tags before all
tracked threads had finished reading `consumed_source_row*`.

Fix:

- add `__syncthreads()` between:
  - the source-row-tag readback block, and
  - the byte-index-tag fill block

Without that barrier, the source-row readout can spuriously show the later
byte-index payload (`2,4,6,...`) instead of the source-row tags.

### Validated tactic-1 physical-offset map

After the barrier fix, the tracked tactic-1 runtime families all validate
cleanly against the exact physical-offset decode:

```bash
NEMOTRON_NANO_P1_B_PROBE_USE_POSITION_MAP=1 \
NEMOTRON_NANO_P1_B_PROBE_TIDS=32,36,40,44,48,52,56,60 \
  ./build-sm120-relwithdebinfo/testing/nano_p1_b_operand_probe

NEMOTRON_NANO_P1_B_PROBE_USE_POSITION_MAP=1 \
NEMOTRON_NANO_P1_B_PROBE_TIDS=33,37,41,45,129,133,137,141 \
  ./build-sm120-relwithdebinfo/testing/nano_p1_b_operand_probe

NEMOTRON_NANO_P1_B_PROBE_USE_POSITION_MAP=1 \
NEMOTRON_NANO_P1_B_PROBE_TIDS=64,68,72,76,80,84,88,92 \
  ./build-sm120-relwithdebinfo/testing/nano_p1_b_operand_probe

NEMOTRON_NANO_P1_B_PROBE_USE_POSITION_MAP=1 \
NEMOTRON_NANO_P1_B_PROBE_TIDS=145,149,153,157,33,37,41,45 \
  ./build-sm120-relwithdebinfo/testing/nano_p1_b_operand_probe
```

Each run reported:

```text
total_row_mismatches=0/128
```

So for the tracked tactic-1 consumer surface, `desired_source_row` is a stable
function of the **physical** `stage0_B(row, byte_index * 2) & 0x1ff` offset,
not of the logical staging row.

The validated decode is:

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

This matches all currently sampled tactic-1 families and is the new
load-bearing runtime-side B probe result.

### Important limitation

Promoting this decode directly into the live NanoP1 kernel is **not** yet safe.

An attempted live-kernel rewrite that used this helper for B data and SFB scale
staging regressed Phase 1 all-ones from:

```text
expected = 256
actual   = 64
```

and was reverted immediately. That means this helper is currently validated only
for the tracked **consumer subset**, not for the entire staging surface used by
the live kernel.

## 2026-04-14 Follow-Up: Real-Input `tCrB_cv` Raw Signature

The standalone runtime probe was extended again to dump a compact raw signature
from the copied `tCrB_cv` values before the fragment is viewed as final
`BRegister` words.

Command:

```bash
NEMOTRON_NANO_P1_B_PROBE_TIDS=32,48,64,80 \
NEMOTRON_NANO_P1_B_INPUT_FILE=/tmp/golden_probe_w1_byte_index_k128_w1_fp4_first128.bin \
NEMOTRON_NANO_P1_B_INPUT_ROWS=128 \
NEMOTRON_NANO_P1_B_PACKED_ROW_BYTES=64 \
  ./build-sm120-relwithdebinfo/testing/nano_p1_b_operand_probe \
  > /tmp/nano_p1_b_operand_probe_runtime_w1_byte_rowperm_rawsig.txt
```

Key sampled lines:

```text
tid=32 local_col0=0 k_block=0 raw=[0,0,1,0,2,0,3,0] reg_pre=(0x03020100,0x13121110)
tid=32 local_col0=0 k_block=1 raw=[0,2,1,2,2,2,3,2] reg_pre=(0x23222120,0x33323130)

tid=48 local_col0=1 k_block=0 raw=[0,2,1,2,2,2,3,2] reg_pre=(0x23222120,0x33323130)
tid=48 local_col0=1 k_block=1 raw=[0,0,1,0,2,0,3,0] reg_pre=(0x03020100,0x13121110)
```

This is the important boundary answer:

1. The odd-lane-family `k_block` inversion is already visible in the copied
   `tCrB_cv` values.
2. The `recast<BRegister>(tCrB(...))` boundary is **not** introducing that
   swap. The copied sub-byte payload already carries it.
3. So the remaining bug surface moved upstream again:
   - B staging payload assembly into shared memory, and/or
   - the B copy-view / retile contract that reads those stage-0 bytes

Caveat:

- The current raw signature only dumps the first 8 physical `tCrB_cv` elements.
  That is enough to prove the low-byte odd-family inversion above, but it does
  not yet span the full copied payload for every `n_tile`. Some later `n_tile`
  entries collapse to zeros in this window even while `reg_pre` is non-zero.
- So the next deeper probe should widen this raw window or target the exact
  `tCrB_cv` element positions that feed `reg_pre`.

## 2026-04-14 Follow-Up: Fragment-Logical Raw View Is Stable Across All `n_tile`

The standalone runtime probe was extended twice more:

1. widen the physical `tCrB_cv` raw signature from 8 to 32 elements
2. dump the logical fragment view `tCrB(_, n_tile, k_block)` itself as
   `frag_raw_consumed_bytes`

Command:

```bash
NEMOTRON_NANO_P1_B_PROBE_TIDS=32,48,64,80 \
NEMOTRON_NANO_P1_B_INPUT_FILE=/tmp/golden_probe_w1_byte_index_k128_w1_fp4_first128.bin \
NEMOTRON_NANO_P1_B_INPUT_ROWS=128 \
NEMOTRON_NANO_P1_B_PACKED_ROW_BYTES=64 \
  ./build-sm120-relwithdebinfo/testing/nano_p1_b_operand_probe \
  > /tmp/nano_p1_b_operand_probe_runtime_w1_byte_fragraw.txt
```

This split the two surfaces cleanly:

### 1. Physical `tCrB_cv` order is not stable across all `n_tile`

For `n_tile=0..3`, `copy_view_raw_consumed_bytes` and `frag_raw_consumed_bytes`
agree.

For `n_tile=4..7`, they do not. Example:

```text
tid=32 n_tile=4 k_block=0
  copy_view_raw = [0,2,1,2,2,2,3,2,...]
  frag_raw      = [0,0,1,0,2,0,3,0,...]

tid=32 n_tile=4 k_block=1
  copy_view_raw = [0,0,0,0,0,0,0,0,...]
  frag_raw      = [0,2,1,2,2,2,3,2,...]
```

So `copy_view_raw` is a physical retiled surface, not a stable logical
comparison surface for upper-half `n_tile`s.

### 2. Logical fragment `frag_raw` is stable and load-bearing

Across all tracked `64 / 64` entries, `frag_raw` follows one exact rule:

- even family (`local_col0` even):
  - `k_block=0 -> [0,0,1,0,2,0,3,0,0,1,1,1,2,1,3,1]`
  - `k_block=1 -> [0,2,1,2,2,2,3,2,0,3,1,3,2,3,3,3]`
- odd family (`local_col0` odd):
  - the two fragment patterns are swapped across `k_block`

Mechanical summary:

```text
frag_swap_rule_matches = 64 / 64
```

This is stronger than the earlier recast result:

1. The odd-family `k_block` inversion is already present in the logical
   fragment `tCrB`, not just in the final `reg_pre` words.
2. The upper-half ambiguity was only in physical `tCrB_cv` order.
3. Future comparisons should treat `frag_raw` / `reg_pre` as the trustworthy
   logical surfaces, not `copy_view_raw`, when `n_tile >= 4`.

## 2026-04-14 Follow-Up: Payload-Independent Stage-Slot Map Shows the Same Swap

Using the current probe with `stage_slot_low_byte=1`:

```bash
NEMOTRON_NANO_P1_B_PROBE_TIDS=32,48,64,80 \
NEMOTRON_NANO_P1_B_PROBE_STAGE_SLOT_LOW_BYTE=1 \
  ./build-sm120-relwithdebinfo/testing/nano_p1_b_operand_probe \
  > /tmp/nano_p1_b_operand_probe_stage_slot_frag.txt
```

the first-phase fragment words `source_row_reg*_pre` become a direct low-byte
map of the stage-slot positions feeding the logical fragment.

Observed rule:

- even family:
  - `k_block=0 -> (0x03020100, 0x13121110)`
  - `k_block=1 -> (0x23222120, 0x33323130)`
- odd family:
  - `k_block=0 -> (0x23222120, 0x33323130)`
  - `k_block=1 -> (0x03020100, 0x13121110)`

So the odd-family swap is not a property of the real saved `w1_fp4` bytes. It
already exists in the payload-independent stage-slot map into the logical
fragment.

That moves the root cause boundary one step earlier again:

- the bug is in how the odd-family fragment maps stage slots / K-blocks, not in
  the final register recast and not in the particular saved payload values.

So the next step is not "land this helper." The next step is:

1. derive the full staging-surface ownership / offset coverage, and
2. extend the physical-offset rule until it preserves Phase 1 before trying
   another live-kernel promotion.

## 2026-04-14 follow-up: SFB probe switched to the real fragment path

The standalone probe now measures B-side scale consumption from the actual
`tCrSFB` fragment after `cute::copy(...)`, not by sampling a single anchor cell
in `sSFB_stage0`.

Implementation change:

- fill `sSFB_stage0` with logical-row tags, run `cute::copy(...)`, then pack the
  resulting `tCrSFB(_, n_tile, k_block)` fragment into one 32-bit word
- repeat with logical-column tags

This follows the same pattern used in the P5 scale-fragment tests and gives a
real consumer contract for the scale path.

### Validated SFB consumer contract

With `NEMOTRON_NANO_P1_B_PROBE_USE_POSITION_MAP=1`, the tracked tactic-1 lane
families now show:

- `consumed_scale_row_word0` = one repeated row byte
- `consumed_scale_col_word0` = four packed logical column bytes

Observed contract:

```text
consumed_scale_row_byte
  = (consumed_source_row0 & ~7) | (scale_local_row0 & 7)

consumed_scale_col_bytes(k_block = 0)
  = [0, 16, 32, 48]

consumed_scale_col_bytes(k_block = 1)
  = [64, 80, 96, 112]
```

Interpreting the packed column tags by dividing by 16:

```text
k_block = 0 -> logical scale-byte indices [0, 1, 2, 3]
k_block = 1 -> logical scale-byte indices [4, 5, 6, 7]
```

So the scale K-side contract is now clear: the two `k_block`s consume the low
and high 4-byte halves of the 8-byte Nano scale row.

### Important correction to the live-kernel story

Earlier notes treated the full B data position-map promotion as if it had
preserved Phase 1 and only exposed SFB as the next blocker.

That is **not** trustworthy.

I reran the live-kernel promotion on the current tree in two forms:

1. full B data position-map only, keeping the old row-only SFB path
2. full B data position-map plus a simple identity-row SFB experiment

Both fail immediately on Phase 1 synthetic all-ones:

```text
Phase 1 synthetic all-ones FAIL total_mismatches=32768
```

and were reverted.

So the corrected state is:

- the standalone B data probe is real
- the standalone SFB fragment probe is real
- but the full B data position-map helper is **still not a safe live-kernel
  drop-in**, even before a final SFB rewrite

Current tree after revert:

```text
Phase 1 synthetic all-ones PASS
Phase 3 total matches = 4074 / 245760
```

That means the next live-kernel step is not "apply the validated B helper and
then fix SFB." The next step is to explain why the helper is probe-valid but
still violates the Phase 1 kernel path.

### Root cause of the live mismatch: the helper only covers one quarter of stage-0 B

I extended the standalone probe with a full-surface counter over the **entire**
`stage0_B(row, byte_index * 2)` write domain:

```text
rows      = 128
bytes/row = 64
total     = 8192 stage-0 write positions
```

For the current `NanoP1SourceRowForStageOffset(...)` helper, the probe reports:

```text
full_stage0_surface invalid_source_rows=6144
first_invalid=(row=1 byte=32 offset=32)
full_stage0_surface band_counts=512,512,512,512,512,512,512,512,
                               512,512,512,512,512,512,512,512
```

Interpretation:

- the full stage-0 B write surface is evenly spread across all 16
  `32`-offset bands in `band_offset = stage_offset & 0x1ff`
- the current helper only returns valid rows for `2048 / 8192` positions
- `6144 / 8192` positions fall into uncovered bands and resolve to `-1`

So the standalone row-match success was never a proof that the helper covered
the live kernel. It only proved that the **sampled consumer subset** happened
to live inside the four covered bands.

This explains the live Phase 1 failure directly. The helper is not "almost
complete"; it is a quarter-surface decode.

### Updated next step

The next probe extension must enumerate the **full B consumer fragment surface**,
not just the first two anchor coordinates per `(tid, n_tile, k_block)`.

Until that happens, any live-kernel promotion of the current helper is
guaranteed to drop bytes on uncovered stage-0 bands.

## 2026-04-14 Follow-Up: Full consumed B surface is actually covered

The previous quarter-surface conclusion was too pessimistic.

I extended the standalone probe from a few tracked tactic-1 tids to the full
`tid=0..255` consumer surface in 8-thread batches, still using
`NEMOTRON_NANO_P1_B_PROBE_USE_POSITION_MAP=1`, and checked the full per-entry
consumed B copy-view fragment for every `(tid, n_tile, k_block)`.

Aggregate result over all `4096` entries:

```text
copy_view_row_mismatches = 0
row_bad_count            = 0
missing_count            = 0
```

So the current stage-offset decode is not merely valid for the earlier sampled
families. It covers the full **consumed** B copy-view surface.

This supersedes the earlier implication that the live failure is explained by
"unsampled consumer families" alone.

## 2026-04-14 Follow-Up: byte-index and first-word SFB formulas also hold globally

Using the current probe output over the same `4096` entries, I checked:

- `consumed_byte_index0 == local_col0 / 2`
- `consumed_scale_row_word0 == repeat((consumed_source_row0 & ~7) | (scale_local_row0 & 7))`
- `consumed_scale_col_word0 == [0,16,32,48]` for `k_block=0`,
  `[64,80,96,112]` for `k_block=1`

Aggregate result:

```text
entries        = 4096
byte_bad       = 0
scale_row_bad  = 0
scale_col_bad  = 0
```

So at the current probe granularity, coarse B staging and the first observed
SFB fragment word are both globally consistent.

## 2026-04-14 Follow-Up: the copied B view is sub-byte, not byte-addressable

I then added a check against the actual copied `tCrB_cv` payload after filling
stage-0 B with byte-index tags.

Result:

```text
copy_view_row_bad_count  = 0
copy_view_byte_bad_count = 4096
```

This does **not** mean the copy is wrong. The destination view elements are
`cute::subbyte_reference` FP4 values, not raw packed bytes, so a naive
"expect `col / 2` as a byte" comparison is the wrong invariant.

The important supporting observation is that the raw packed register words are
stable and structured under the byte-tag fill:

```text
k_block=0: reg0=0x03020100 reg1=0x13121110
k_block=1: reg0=0x23222120 reg1=0x33323130
reg_pre == reg_post
```

That says:

1. the byte-level packed register assembly is coherent in the standalone probe
2. `fp4_shift_B` is still a no-op here
3. the remaining open question is now the **sub-byte / nibble contract**, not
   the coarse `(source_row, byte_index)` contract

## Updated implication

The current standalone runtime probe has now eliminated these as primary
unknowns:

- full consumed B source-row selection
- anchor byte-index selection
- first observed SFB word selection
- packed B register byte ordering at a coarse level

The remaining work should focus on one of:

1. reconstructing the exact `(source_row, byte_index)` tuple for each packed
   register byte and comparing that against the live flashinfer tactic-1
   `reg_pre` dump using the real `input_fp4_permuted.bin`, or
2. generating a live reference run with known FP4 nibble patterns so the
   tactic-1 `reg_pre` dump becomes directly decodable at the nibble level

## 2026-04-14 Follow-Up: alternate B copy selector is a no-op in the standalone probe

I added a second probe-only B copy atom:

```c++
sm120_rr_smem_copy_selector_B<
    NanoP1ElementAct,
    NanoP1ElementWeight,
    true>()
```

and captured both the current `byte_tag_reg*_pre` and the alternate
`byte_tag_reg*_pre_alt` from the same staged shared-memory contents.

I checked two cases:

1. payload-independent `stage_slot_low_byte=1`
2. the saved real-input `w1_fp4` byte-index payload

Commands:

```bash
NEMOTRON_NANO_P1_B_PROBE_TIDS=32,48,64,80 \
NEMOTRON_NANO_P1_B_PROBE_STAGE_SLOT_LOW_BYTE=1 \
  ./build-sm120-relwithdebinfo/testing/nano_p1_b_operand_probe \
  > /tmp/nano_p1_b_operand_probe_stage_slot_alt_copy.txt

NEMOTRON_NANO_P1_B_PROBE_TIDS=32,48,64,80 \
NEMOTRON_NANO_P1_B_INPUT_FILE=/tmp/golden_probe_w1_byte_index_k128_w1_fp4_first128.bin \
NEMOTRON_NANO_P1_B_INPUT_ROWS=128 \
NEMOTRON_NANO_P1_B_PACKED_ROW_BYTES=64 \
  ./build-sm120-relwithdebinfo/testing/nano_p1_b_operand_probe \
  > /tmp/nano_p1_b_operand_probe_runtime_w1_alt_copy.txt
```

Result:

```text
stage_slot_low_byte: byte_tag_reg_pre_alt == byte_tag_reg_pre for all sampled entries
w1_fp4 payload:      alt_diff = 0 / 64
```

Representative odd-family sampled entry:

```text
local_col0=1 k_block=0
  default = (0x23222120,0x33323130)
  alt     = (0x23222120,0x33323130)
```

So the current odd-family `k_block` inversion is **not** caused by choosing the
wrong `sm120_rr_smem_copy_selector_B<..., false/true>` specialization. The two
copy-atom choices are behaviorally identical at this probe boundary.
