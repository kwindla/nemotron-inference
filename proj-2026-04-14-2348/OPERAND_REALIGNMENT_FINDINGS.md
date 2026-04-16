# Operand Realignment Findings (2026-04-15)

This note records what the focused B-boundary attempt actually settled.

## What was done

1. Extended `testing/backend/nano_p1_b_operand_probe.cu` to emit a packed-byte
   `copy_sig` surface that matches the live FlashInfer probe format:
   `stage0_offset:packed_byte`.
2. Rebuilt `nano_p1_b_operand_probe`.
3. Re-ran the real-input `position_map` probe on the saved Nano activation dump:

```bash
NEMOTRON_NANO_P1_B_PROBE_TIDS=32,48,64,80 \
NEMOTRON_NANO_P1_B_PROBE_SOURCE_ROW_MODE=position_map \
NEMOTRON_NANO_P1_B_INPUT_FILE=proj-2026-04-14-2348/archive/focused_b_attempt/live_capture_consumer_tids/input_fp4_permuted.bin \
NEMOTRON_NANO_P1_B_PROBE_USE_INPUT_FILE=1 \
NEMOTRON_NANO_P1_B_INPUT_ROWS=128 \
NEMOTRON_NANO_P1_B_PACKED_ROW_BYTES=1344 \
NEMOTRON_NANO_P1_B_INPUT_STAGE_MODE=packed_bytes \
build-sm120-relwithdebinfo/testing/nano_p1_b_operand_probe \
  > proj-2026-04-14-2348/archive/focused_b_attempt/probe_sweep/position_map_packed_bytes_real_input.txt
```

## Immediate result

The runtime packed-byte `copy_sig` does **not** match the live tactic-1
FlashInfer `copy_sig` on the same tracked entries.

Representative mismatch:

- Live:
  - `n_tile=0 k_block=0 stage0_offsets=(0,0) copy_sig=0:142,...`
- Runtime:
  - `n_tile=0 k_block=0 stage0_offset0=0 copy_sig=0:174,...`

So the staged packed bytes already disagree before any later `reg_pre` /
`fp4_shift_B` interpretation.

## More important result

The focused plan was built around the wrong operand pairing.

Local source says:

- FlashInfer / TRT-LLM live tactic-1 probe:
  - `partition_fragment_A(sInput(...))`
  - `partition_fragment_B(sfc1_weight(...))`
  - file:
    `vllm-env-cu128/lib/python3.12/site-packages/flashinfer/data/csrc/nv_internal/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm/kernel/fused_moe_kernel_routine.cuh`
- Our Nano kernel:
  - `partition_fragment_A(sA_stage0)` where `sA_stage0` is filled from `weight_packed`
  - `partition_fragment_B(sB_stage0)` where `sB_stage0` is filled from `input_packed`
  - file:
    `runtime/src/backend/fused_moe_prefill/nano_p1_kernel.cuh`

That means:

- live FlashInfer `B` probe is on the **weight** operand
- our Nano runtime `B` probe is on the **activation** operand

So comparing live `B` to runtime `B` is not a trustworthy correctness oracle.

## What this invalidates

These steps from the focused attempt are no longer sufficient as written:

- treating live tactic-1 `B` as the direct oracle for `nano_p1_b_operand_probe`
- inferring a production fix from live/runtime `B` parity on the current probe
- interpreting the real-input activation-side `copy_sig` mismatch as a direct
  FlashInfer-vs-Nano bug localization

## What is still true

- The runtime activation-side probe changes were useful.
- The added runtime `copy_sig` output is worth keeping.
- The activation-side runtime probe still shows a structured packed-byte
  mismatch on its own surface.

## Correct next move

Realign the oracle before touching production:

1. Either compare live FlashInfer `B` against a **Nano weight-side / A-side**
   probe.
2. Or compare Nano runtime `B` against a **reference activation-side / A-side**
   probe.

Do not keep iterating on the old live-`B` vs runtime-`B` comparison.

## Activation-side live probe follow-up

I implemented option 2 in the live FlashInfer tree under
`vllm-env-cu128/.../flashinfer/data/...`:

- added `operand_kind` selection to the existing operand-probe state
- added `NEMOTRON_TRTLLM_OPERAND_PROBE_OPERAND=activation`
- extended the SM120 blockscaled mainloop probes so the live kernel can dump
  `A=input` instead of only `B=weight`

The Nano-bucket tactic-1 capture now succeeds with:

```bash
NEMOTRON_TRTLLM_ONLY_GEMM1_TACTIC=1 \
NEMOTRON_TRTLLM_OPERAND_PROBE_TIDS=32,48,64,80 \
NEMOTRON_TRTLLM_OPERAND_PROBE_OPERAND=activation \
  /home/khkramer/src/nemotron-inference/vllm-env-cu128/bin/python \
  proj-2026-04-12-1022/trtllm_reference/capture_bf16_gemm1.py \
  --num-tokens 128 \
  --hidden-size 2688 \
  --inter-size 1920 \
  --num-experts 1 \
  --top-k 1 \
  --seed 12648430 \
  --golden-dir proj-2026-04-14-2348/archive/focused_b_attempt/live_capture_activation_probe \
  --metadata-path proj-2026-04-14-2348/archive/focused_b_attempt/live_capture_activation_probe/bf16_gemm1_metadata.json \
  --input-save-dir proj-2026-04-14-2348/archive/focused_b_attempt/live_capture_activation_probe \
  --flashinfer-src-root /home/khkramer/src/nemotron-inference/vllm-env-cu128/lib/python3.12/site-packages/flashinfer/data/csrc \
  --venv-root /home/khkramer/src/nemotron-inference/vllm-env-cu128
```

The resulting header confirms the live dump is now activation-side:

- `operand_probe_tactic1.txt`:
  - `operand_kind=1 operand=activation`

## What that changed

The new live activation probe is real, but the naive keyed comparison is still
not directly valid.

Using `proj-2026-04-14-2348/compare_activation_probe_alignment.py` on:

- live:
  `archive/focused_b_attempt/live_capture_activation_probe/operand_probe_tactic1.txt`
- runtime:
  `archive/focused_b_attempt/probe_sweep/position_map_packed_bytes_real_input.txt`

produces:

- `shared=64`
- `part_c0_mismatches=62`
- `stage0_offsets_mismatches=64`
- `copy_sig_mismatches=64`
- `part_c0_swapped_matches=64`

So the old mistake is fixed, but another one became visible:

- live FlashInfer activation uses `partition_fragment_A`
- Nano activation still uses `partition_fragment_B`
- same `(tid, n_tile, k_block)` on the two sides does **not** denote the same
  fragment orientation
- the output anchor coordinates are systematically swapped before we even ask
  whether the consumed packed bytes agree

That means the next comparison has to be normalized by **logical activation
coordinates**, not raw `(tid, n_tile, k_block)` fragment coordinates.

## Tactic-5 result

The next useful correction was to stop comparing Nano activation against live
tactic-1 `A=input` and instead compare Nano activation against **live tactic-5
`B` with `swap_ab=true`**.

That comparison is the first one that is structurally aligned:

- live:
  `archive/focused_b_attempt/live_capture_tactic5_bside/operand_probe_tactic5.txt`
- runtime:
  `archive/focused_b_attempt/probe_sweep/position_map_packed_bytes_real_input.txt`

On the shared 64 `(tid, n_tile, k_block)` entries:

- `part` matches `64/64`
- `local0` matches `64/64`
- `local1` matches `64/64`
- `stage0_offsets` match `64/64`
- raw byte-index expectations match `64/64`

What still fails is the staged byte payload itself:

- `copy_sig` mismatches `64/64`

So the remaining disagreement is no longer an orientation mismatch. It is on
the actual bytes staged into the aligned B-side surface.

## What simple fixes did not explain

I swept the remaining low-complexity staging assumptions against the live
tactic-5 `copy_sig` using the frozen Nano-bucket activation dump:

- source-row modes:
  - `row_only_permute`
  - `position_map`
  - `position_map_odd_plus4`
- odd-slot byte tweak:
  - `NEMOTRON_NANO_P1_B_PROBE_ODD_STAGE_OFFSET_XOR32=0`
  - `NEMOTRON_NANO_P1_B_PROBE_ODD_STAGE_OFFSET_XOR32=1`

All 6 combinations still mismatch `64/64` on `copy_sig`.

I also re-ran the runtime probe with the more production-like input staging
modes:

- `input_stage_mode=packed_bytes`
- `input_stage_mode=unpack_nibbles`
- `input_stage_mode=unpack_nibbles_swap_halves`

Results:

- `packed_bytes`: `copy_sig` mismatches `64/64`
- `unpack_nibbles`: `copy_sig` mismatches `64/64`
- `unpack_nibbles_swap_halves`: `copy_sig` mismatches `64/64`

So the remaining bug is **not** fixed by:

- toggling between the existing source-row decode variants
- the old odd-slot `xor32` guess
- switching between raw packed staging and the obvious nibble-unpack variants

## Synthetic low-byte result

I also ran a fresh live tactic-5 B-side capture with:

- `NEMOTRON_TRTLLM_OPERAND_PROBE_STAGE_SLOT_LOW_BYTE=1`
- `NEMOTRON_TRTLLM_OPERAND_PROBE_OVERWRITE_ALL_THREADS=1`

The output is in:

- `archive/focused_b_attempt/live_capture_tactic5_lowbyte/operand_probe_tactic5.txt`

This was useful for one reason: it showed that the live probe’s synthetic
overwrite path does **not** behave like the Nano probe’s raw-byte tag path.

Examples from the live tactic-5 low-byte capture:

- stage slot `0` produced `copy_sig=0:16`
- stage slot `2` produced `copy_sig=4:84`
- stage slot `4` produced `copy_sig=8:145`

That means the live synthetic overwrite is going through the operand element
representation in a way that the Nano probe’s raw byte-tag writes do not
mirror. So the old synthetic-vs-synthetic comparisons are weaker than they
looked.

## Remaining uncertainty

There is still one unresolved oracle gap: the live tactic-5 captures are not
guaranteed to come from the same logical CTA tile as the single-CTA Nano probe.

Evidence:

- the live tactic-5 real-input capture recorded `block=(0,13,0)`
- the live tactic-5 synthetic low-byte capture recorded `block=(0,1,0)`

The local `(tid, n_tile, k_block)` structure can still match across runs while
the underlying global tile origin differs.

I attempted to patch the live probe to dump the current CTA/work-tile
coordinates directly, but that patch hit the wrong SM120 hook site and did not
compile cleanly, so it was backed out.

## Current best next move

The shortest remaining path is:

1. Extend the live tactic-5 probe at the **actual** SM120 kernel call site that
   has access to `cta_coord_mnkl` / work-tile info.
2. Re-capture tactic-5 real-input with those global tile coordinates recorded.
3. Re-run the Nano probe on the matching logical tile origin instead of always
   assuming `row_start=0`.
4. Only after that revisit source-row reconstruction or byte-index transforms.

## Update: real GEMM1 input dump plus live overwrite on the true buffer

I then added a dump of the actual `gemm1_input` buffer used immediately before
GEMM1 and re-ran the tactic-5 capture.

### The old Python-side activation dump was not the live surface

For the tactic-5 real-input capture:

- `archive/focused_b_attempt/live_capture_tactic5_bside_real_input/input_fp4_permuted.bin`
- `archive/focused_b_attempt/live_capture_tactic5_bside_real_input/gemm1_input_tactic5.bin`

are both `172032` bytes, but they differ in `160601` byte positions. The first
difference is at offset `1344`.

So the old Python-side `input_fp4_permuted.bin` was not the actual live tactic-5
GEMM1 input surface.

### Probe semantics correction for tactic 5

The tactic-5 launcher is `swap_ab=true`, so the semantic operand mapping is:

- `A = weight`
- `B = activation`

The current SM120 probe code still interprets
`NEMOTRON_TRTLLM_OPERAND_PROBE_OPERAND=activation` as "record `A`", not
"record the semantic activation operand". So for tactic 5, the semantically
correct live activation probe is currently the run with:

```bash
NEMOTRON_TRTLLM_OPERAND_PROBE_OPERAND=weight
```

because that leaves the probe on `B`, and for `swap_ab=true` `B` is the
activation operand.

### Live overwrite of the real GEMM1 input buffer

I added `NEMOTRON_TRTLLM_GEMM1_INPUT_PATTERN` so the capture can overwrite the
true `gemm1_input` buffer immediately before GEMM1. The supported modes are:

- `row_index`
- `byte_index_low8`
- `linear_index_low8`
- `row_xor_byte_low8`

This is the first synthetic path that definitely writes the exact device buffer
consumed by the live tactic-5 kernel.

### What the new synthetic runs proved

Using the semantically correct tactic-5 live `B` probe
(`OPERAND_PROBE_OPERAND=weight`):

1. `byte_index_low8`
   - live vs Nano `copy_sig` matches exactly with `position_map`
   - live vs Nano `copy_sig` also matches exactly with `row_only_permute`
   - this rules out a simple byte-within-row staging bug

2. `row_index`
   - `row_only_permute`: `copy_sig_mismatches=0/64`
   - `position_map`: `copy_sig_mismatches=56/64`
   - `position_map_odd_plus4`: `copy_sig_mismatches=56/64`
   - this disproves the current `position_map` source-row decoder against the
     live tactic-5 oracle

3. `row_xor_byte_low8`
   - `row_only_permute`: `copy_sig_mismatches=0/64`
   - `position_map`: `copy_sig_mismatches=56/64`
   - `position_map_odd_plus4`: `copy_sig_mismatches=56/64`
   - this shows the live-vs-runtime agreement under `row_only_permute`
     survives a combined row+byte synthetic, not just the two independent axes

### Current contradiction after the synthetic proof

Despite those synthetic matches, the real-input tactic-5 comparison still does
not match even when the runtime probe reads the dumped live `gemm1_input` and
uses `row_only_permute`:

- live: `archive/focused_b_attempt/live_capture_tactic5_bside_real_input/operand_probe_tactic5.txt`
- runtime: `archive/focused_b_attempt/probe_sweep_tactic5_real_row_only/runtime.txt`

That compare still reports:

- `copy_sig_mismatches=64/64`

So the current state is:

- `position_map` is definitely wrong
- `row_only_permute` matches every current synthetic overwrite on the true live
  buffer
- but the real dumped `gemm1_input` bytes still do not explain the live real
  probe bytes

This now looks less like a Nano row/byte decoder bug and more like a remaining
real-surface inconsistency:

- either the dumped real `gemm1_input_tactic5.bin` is still not the exact bytes
  reaching the live tactic-5 `B` probe in the real run
- or the real live `B` probe is observing an additional real-data-dependent
  transform that the synthetic byte overwrites are not exposing

At this point, the shortest next experiment is not another Nano row-mode tweak.
It is to instrument the live tactic-5 real path further so we can explain why
the real `B` probe bytes disagree with the dumped real `gemm1_input`, while the
synthetic overwrites agree exactly.

## 2026-04-15 follow-up: Nano A-side probe now builds, but direct live-weight vs Nano-weight is still not a valid keyed oracle

I added a standalone runtime weight/A probe at:

- `testing/backend/nano_p1_a_operand_probe.cu`

and wired it into:

- `testing/CMakeLists.txt`
- `proj-2026-04-14-2348/compare_activation_probe_alignment.py`

The probe does the same minimal thing the live activation probe does:

- stage raw packed weight bytes into `smem_A` using `stage0_A(row, byte_index * 2)`
- call `partition_fragment_A(sA_stage0)`
- record `part_c0`, local coords, `stage0_offsets`, `copy_sig`, and `reg_pre`
  for tids `32,48,64,80`

### A-vs-A synthetic sanity check passed

Against the stable block-0 live activation captures:

- `archive/focused_b_attempt/live_capture_tactic5_a_block000_row_index/operand_probe_tactic5.txt`
- `archive/focused_b_attempt/live_capture_tactic5_a_block000_byte_index/operand_probe_tactic5.txt`

the runtime A probe matches on all 64 shared entries for:

- `local0`
- `local1`
- `stage0_offsets`
- `copy_sig_count`
- `copy_sig`

The only mismatch is the same swapped `part_c0` convention:

- `part_c0_mismatches=62`
- `part_c0_swapped_matches=64`

So the runtime A probe is structurally sound for `A`-surface comparisons.

### Direct real-input compare against live weight/B is still not trustworthy

Using the real Nano weight bytes:

- runtime file:
  `proj-2026-04-12-1022/trtllm_reference/golden_nano_k2688/inputs_w1_fp4.bin`
- live file:
  `archive/focused_b_attempt/live_capture_tactic5_block000_run1/operand_probe_tactic5.txt`

the direct keyed compare reports:

- `shared=64`
- `local0_mismatches=62`
- `local1_mismatches=62`
- `stage0_offsets_mismatches=62`
- `copy_sig_mismatches=64`
- `part_c0_swapped_matches=64`

That is not evidence of a production weight bug by itself. It is evidence that
`live weight(B)` and `runtime weight(A)` still do not share a directly keyed
fragment surface.

### Synthetic weight tuples make the normalization failure explicit

To remove real-input ambiguity, I compared paired synthetic weight captures:

- live row tags:
  `archive/focused_b_attempt/live_capture_tactic5_bside_row_index/operand_probe_tactic5.txt`
- live byte tags:
  `archive/focused_b_attempt/live_capture_tactic5_bside_byte_index_low8/operand_probe_tactic5.txt`
- runtime row tags:
  `archive/focused_b_attempt/runtime_a_probe/row_index.txt`
- runtime byte tags:
  `archive/focused_b_attempt/runtime_a_probe/byte_index.txt`

and decoded the logical `(source_row, byte_index)` tuple from the first
`copy_sig` byte in:

- `proj-2026-04-14-2348/analyze_weight_operand_synthetics.py`

Saved report:

- `archive/focused_b_attempt/runtime_a_probe/weight_synthetic_tuple_analysis.txt`

Result:

- `shared_entries=64`
- `logical_tuple_mismatches=60`

Representative samples:

- `tid=32 n_tile=0 k_block=0 part=(16,0)`:
  live tuple `(0,0)`, runtime tuple `(0,1)`
- `tid=32 n_tile=1 k_block=0 part=(16,8)`:
  live tuple `(0,2)`, runtime tuple `(0,1)`
- `tid=32 n_tile=2 k_block=0 part=(16,32)`:
  live tuple `(0,4)`, runtime tuple `(0,1)`

For the tracked entries, the live weight/B probe keeps `source_row=0` but walks
byte indices `0,2,4,...` with `n_tile`, while the runtime weight/A probe stays
fixed at byte index `1` or `2` per tid.

That settles the current state:

- the new runtime A probe itself is not obviously broken, because it matches the
  live `A` surface exactly under synthetic activation captures
- the keyed `live weight(B)` vs `runtime weight(A)` compare is still not a valid
  correctness oracle
- before any production weight-path patch, we need one more normalization step:
  either a semantic live weight oracle that emits common logical source coords,
  or a local B-shaped weight proxy that can be compared directly to the live B
  surface and then bridged back to Nano A

## Live tactic-1 weight semantic probe: packed-byte coordinate fixed, row still unresolved

I then continued on the live tactic-1 `B=weight` probe to make the live weight
surface emit semantic source tags directly.

Changed files in the installed FlashInfer tree:

- `vllm-env-cu128/lib/python3.12/site-packages/flashinfer/data/cutlass/include/cutlass/gemm/collective/nemotron_operand_probe.hpp`
- `vllm-env-cu128/lib/python3.12/site-packages/flashinfer/data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh`

### What was wrong in the first semantic patch

The first version populated `semantic_copy_view`, but it was obviously wrong on
the tactic-1 synthetic runs:

- `row_index`: `archive/semantic_weight_probe_tactic1_row/operand_probe_tactic1.txt`
- `byte_index`: `archive/semantic_weight_probe_tactic1_byte/operand_probe_tactic1.txt`

Symptoms:

- semantic byte stayed pinned at `127`
- later stage slots fell back to `-1`
- the synthetic byte payload in `byte_index` had no effect on the semantic byte
  field

I added a compact producer-side debug log to the live probe:

- `archive/semantic_weight_probe_tactic1_byte_debug/operand_probe_tactic1.txt`

That exposed the real bug. For the producer `dst` coords, the varying stage
column was not in the second flattened component. It was in the third:

- `dst=(0,0,0,0,0,0)`
- `dst=(0,0,1,0,0,0)`
- `dst=(0,0,2,0,0,0)`

So the producer was keying the semantic stage map with the wrong destination
coordinate and collapsing many writes into slot `0`.

### What I fixed

I made three corrections to the live tactic-1 semantic stage map:

1. Use the third flattened producer destination component as the stage-space
   column when computing the stage slot.
2. Keep the first producer write per stage slot instead of the last one, so the
   stage map tracks the low packed byte that the consumer probe is actually
   reporting.
3. Convert the producer source byte component to packed-byte units by dividing
   by `2`, so the semantic byte index matches the consumer probe’s
   `raw_copy_view` byte index rather than half-byte coordinates.

Verification runs:

- `byte_index` after dst-slot fix:
  `archive/semantic_weight_probe_tactic1_byte_fixed/operand_probe_tactic1.txt`
- `byte_index` after first-writer policy:
  `archive/semantic_weight_probe_tactic1_byte_fixed2/operand_probe_tactic1.txt`
- `byte_index` after packed-byte normalization:
  `archive/semantic_weight_probe_tactic1_byte_fixed3/operand_probe_tactic1.txt`
- `row_index` on the same final build:
  `archive/semantic_weight_probe_tactic1_row_fixed3/operand_probe_tactic1.txt`

### Current result

The live tactic-1 semantic probe now emits a correct packed-byte coordinate for
the tracked weight/B surface.

Examples from `byte_fixed3`:

- `stage0_offset=0`:
  `raw_copy_view ... byte_index=0`
  `semantic_copy_view logical=(0,0,0)`
- `stage0_offset=4`:
  `raw_copy_view ... byte_index=2`
  `semantic_copy_view logical=(0,2,0)`
- `stage0_offset=16`:
  `raw_copy_view ... byte_index=8`
  `semantic_copy_view logical=(0,8,0)`

So the semantic byte index now agrees with the consumer-side packed-byte index.

What is still missing:

- On `row_index`, the semantic row is still pinned at `0`:
  `archive/semantic_weight_probe_tactic1_row_fixed3/operand_probe_tactic1.txt`
- The raw stage bytes clearly move with row synthetic tags
  (for example `reg_pre=0x08080808` at `stage0_offset=4`,
  `reg_pre=0x40404040` at `stage0_offset=16`), so the live probe is not yet
  emitting the correct semantic row coordinate.

### Updated conclusion

The live weight semantic probe is now partially normalized:

- packed-byte coordinate: usable on tactic-1 weight/B
- source row coordinate: still unresolved

That is real progress, because the failure is now isolated to row extraction
rather than the whole semantic mapping scheme. The next clean step is to extend
the producer debug log beyond the first-row slice so we can see which producer
source component begins to vary when later weight rows are loaded, and then port
that row extraction rule into the semantic stage map. After that, the same
pattern needs to be mirrored to the live `A` path, because the current tactic-5
semantic weight surface is `A`, not `B`.

## Tactic-5 live A semantic follow-up

I finished the activation-side semantic row patch in the live FlashInfer tree:

- `vllm-env-cu128/.../nemotron_operand_probe.hpp`

The live tactic-5 block-0 synthetic captures now emit usable semantic copy-view
coordinates on the `A` path:

- row synthetic:
  `archive/tactic5_a_semantic_row_index_block000_v2/operand_probe_tactic5.txt`
- byte synthetic:
  `archive/tactic5_a_semantic_byte_index_block000_v1/operand_probe_tactic5.txt`

Representative entries:

- `entry=0`, `stage0_offset=2`:
  `semantic_copy_view logical0=(16,2561,20)`
- `entry=48`, `stage0_offset=4`:
  `semantic_copy_view logical0=(32,2562,20)`

This fixed the copy-view normalization problem. Re-running:

- `proj-2026-04-14-2348/analyze_weight_operand_synthetics.py`

on the new live tactic-5 `A` captures vs:

- `archive/focused_b_attempt/runtime_a_probe/row_index.txt`
- `archive/focused_b_attempt/runtime_a_probe/byte_index.txt`

now gives:

- `shared_entries=64`
- `logical_tuple_mismatches=0`

So the live tactic-5 `A` surface and the local Nano A probe now agree on the
synthetic logical `(row_tag, byte_tag)` tuple for the first `copy_sig` byte.

## Copy-view was not deep enough

That success did **not** extend to the consumed register payload.

I added:

- `proj-2026-04-14-2348/analyze_weight_reg_pre_synthetics.py`

and saved the report to:

- `archive/focused_b_attempt/runtime_a_probe/reg_pre_tuple_analysis.txt`

This decodes `(row_tag, byte_tag)` tuples from the synthetic `reg_pre` bytes
instead of from `copy_sig`.

Result on the same stabilized tactic-5 block-0 synthetic captures:

- `shared_entries=64`
- `entry_mismatches=40`
- `lane_mismatches=320`

The pattern is not a small lane-order issue:

- for many entries starting at `tid=32, n_tile=2`, all 8 decoded lanes differ
- live `reg_pre` tuples become effectively arbitrary payload bytes
- runtime `reg_pre` tuples stay structured or collapse to zero, depending on
  `n_tile` / `k_block`

So the local Nano A probe and the live tactic-5 `A` probe agree at the
copy-view boundary, but they still disagree at the actual consumed `reg_pre`
boundary.

## Real-input consequence

I reran the live tactic-5 real-input capture on the current versioned probe
with a fixed CTA:

- `archive/focused_b_attempt/live_capture_tactic5_activation_real_input_block000_v2/operand_probe_tactic5.txt`

and compared it against a local runtime A probe driven from the extracted
packed weight bytes:

- live packed weights extracted from
  `archive/focused_b_attempt/live_capture_tactic5_activation_real_input_block000_v2/inputs.pt`
- runtime probe output:
  `archive/focused_b_attempt/runtime_a_probe/real_input_from_live_tactic5.txt`

The direct comparison gives:

- `shared=64`
- `part_matches=64`
- `local0_matches=64`
- `local1_matches=64`
- `stage0_matches=64`
- `sig_matches=0`
- `reg_pre_matches=16`

The fixed-CTA live real-input probe also exposes another inconsistency:

- `entry=0` reports `semantic_copy_view logical0=(16,2561,20)`
- that implies source row `16`, packed-byte index `1`
- but the observed live `copy_sig` byte is `26`
- the saved packed weight tensor has `w1_fp4[0,16,1] = 28`

So the current semantic copy-view tag still does not identify the byte feeding
the consumed register path on real payloads.

## Updated conclusion

The major blocker has shifted again:

- copy-view / stage-slot normalization for tactic-5 live `A` is now good enough
- consumed-register (`reg_pre`) normalization is not

The next valid debugging step is to instrument semantic source coordinates at
the `reg_pre` boundary itself, rather than inferring them from `copy_sig` or
`semantic_copy_view`. Until that exists, real-input conclusions drawn from the
current copy-view semantic map are still unsafe.

## Weight-scale isolation sweep

I added Phase 3 execution-scale override controls to:

- `testing/backend/nano_p1_direct_pack_oracle_test.cpp`
- `testing/backend/nano_p1_mainloop_oracle_test.cpp`

The direct-pack oracle now accepts:

- `NEMOTRON_NANO_P1_DIRECT_PACK_INPUT_SCALE_MODE`
- `NEMOTRON_NANO_P1_DIRECT_PACK_WEIGHT_SCALE_MODE`
- `NEMOTRON_NANO_P1_DIRECT_PACK_INPUT_SCALE_REGION`
- `NEMOTRON_NANO_P1_DIRECT_PACK_WEIGHT_SCALE_REGION`

with regions:

- `all`
- `lo4`
- `hi4`
- `slot0..slot7`

These overrides operate on **logical execution-scale block coordinates**, not
 raw bytes, so they are valid for the captured `kSwizzled128x4` tensors.

### Direct-pack result

Reconfirmed baseline:

- captured/captured:
  - `packed_bytes mismatches = 95,929`
  - `block_scales mismatches = 13,759`
  - `matmul_block_scales mismatches = 13,759`
  - `activation_output_scales mismatches = 0`

Replacing **all weight execution scales** with unit scales still makes Phase 3
pass bitwise end-to-end in the direct-pack oracle.

But partial overrides do **not** isolate the bug to one byte position:

- `unit/lo4`: packed `95,763`, block scales `13,704`
- `unit/hi4`: packed `96,155`, block scales `13,766`

Per-slot sweep:

- `slot0`: packed `95,886`, block scales `13,735`
- `slot1`: packed `95,813`, block scales `13,752`
- `slot2`: packed `96,124`, block scales `13,796`
- `slot3`: packed `95,856`, block scales `13,696`
- `slot4`: packed `96,072`, block scales `13,839`
- `slot5`: packed `95,792`, block scales `13,748`
- `slot6`: packed `96,238`, block scales `13,799`
- `slot7`: packed `95,841`, block scales `13,782`

Interpretation:

- the remaining bug is definitely on the **weight-scale path**
- but it is **not** one isolated `block % 8` slot
- `lo4` helps slightly and `hi4` hurts slightly, so the first 64 K-columns are
  somewhat more implicated than the second 64, but only weakly
- the failure still looks like a **broad SFA placement/consumption error**, not
  a single poisoned scale byte

### Mainloop result

The mainloop oracle now accepts the analogous env vars:

- `NEMOTRON_NANO_P1_MAINLOOP_INPUT_SCALE_MODE`
- `NEMOTRON_NANO_P1_MAINLOOP_WEIGHT_SCALE_MODE`
- `NEMOTRON_NANO_P1_MAINLOOP_INPUT_SCALE_REGION`
- `NEMOTRON_NANO_P1_MAINLOOP_WEIGHT_SCALE_REGION`

Important caveat:

- these overrides intentionally change the math
- so `runtime kernel vs flashinfer tactic1` is no longer a correctness oracle
  under override modes

What *is* still useful is the first-step debug payload.

Observed on the current probe:

- baseline captured/captured:
  - primary gate `7,883 / 245,760` bitwise
  - `probe fragment_word[4] = 0x78727576`
  - `probe first_k_step matches_expected_64 = 0 / 16,384`
- weight=`unit/all`:
  - `probe fragment_word[4] = 0x38383838`
  - first-step values shrink dramatically in magnitude
- weight=`unit/lo4`:
  - `probe fragment_word[4] = 0x38383838`
  - first-step values match the `unit/all` shape
- weight=`unit/hi4`:
  - `probe fragment_word[4] = 0x78727576`
  - first-step values stay at the captured/captured scale

Interpretation:

- the first recorded `rSFA(0)` debug word is driven by the **low four** scale
  slots of the 128-wide K tile
- the currently printed first-step probe is therefore only seeing the low-half
  scale effect directly
- that still does **not** explain the full bug, because the direct-pack slot
  sweep says the error is broader than one half or one slot

## Updated next step

The clean next move is no longer another output-level override sweep.

The next valid probe is a **scale-register boundary probe**:

- dump more than one `rSFA` word, not just `fragment_word[4]`
- attach semantic `(source_row, block_index)` tags to the SFA consumer path
- compare that against a live FlashInfer scale probe or a semantically-tagged
  local synthetic

At this point the weight-byte path is much better understood than the
weight-scale path. The scale path now needs the same semantic-boundary treatment
the byte path already received.

## 2026-04-15: live scale probe correction

The first live weight-scale comparison turned out to be using the wrong side of
the SM120 copy boundary.

### What changed

- I patched the live FlashInfer operand probe to dump `scale_reg_post` in
  addition to `scale_reg_pre`.
- The new field is recorded **after**
  `copy(tCsSFA_stage/tCsSFB_stage, tCrSF*_copy_view(_,_,k_block))`.
- This matters because the old `scale_reg_pre` field was captured before that
  copy and therefore contained stale register state for many `n_tile` lanes.

Patched files:

- `vllm-env-cu128/.../nemotron_operand_probe.hpp`
- `vllm-env-cu128/.../cutlass_fused_moe_kernels.cuh`
- `vllm-env-cu128/.../sm120_blockscaled_mma_tma.hpp`
- `vllm-env-cu128/.../sm120_blockscaled_mma_array_tma.hpp`

Fresh capture:

- `archive/live_scale_probe_tactic5_weight_block000_v2/operand_probe_tactic5.txt`

### Consequence

The earlier statement “live-vs-Nano mismatch at the consumed weight-scale
register boundary” was too strong, because it relied on `scale_reg_pre`.

The corrected interpretation is:

- `scale_reg_pre` is not a trustworthy oracle for the current `k_block`
- `scale_reg_post` is the correct current register surface

### Current live post facts

Using `scale_reg_post` on the stable tactic-5 block-0 capture:

- `tid=0,2 n_tile=0 k_block=0` -> `0x74757174`
- `tid=0,2 n_tile=0 k_block=1` -> `0x72757372`
- `tid=0,2 n_tile=1 k_block=0` -> `0x7274766e`
- `tid=0,2 n_tile=1 k_block=1` -> `0x6e717376`
- `tid=0,2 n_tile=2 k_block=0` -> `0x706d7276`
- `tid=0,2 n_tile=2 k_block=1` -> `0x00000000`
- `tid=0,2 n_tile=3 k_block=0` -> `0x71747174`
- `tid=0,2 n_tile=3 k_block=1` -> `0x00000000`
- `tid=0,2 n_tile>=4` -> `0x00000000`

This already removes the previous contradiction where the live capture was
showing nonzero small words that did not exist anywhere in `inputs_w1_sf.bin`.

### Local alignment result

The standalone Nano scale probe now matches the first live post group exactly
when pointed at the late-K scale window instead of `block_base=0`.

With:

- `NEMOTRON_NANO_P1_A_SCALE_BLOCK_BASE=160`
- `NEMOTRON_NANO_P1_A_SCALE_OUTPUT_COL_BASE=0`

the Nano probe reproduces:

- `n_tile=0 k_block=0` -> `0x74757174`
- `n_tile=0 k_block=1` -> `0x72757372`

With:

- `NEMOTRON_NANO_P1_A_SCALE_BLOCK_BASE=160`
- `NEMOTRON_NANO_P1_A_SCALE_OUTPUT_COL_BASE=64`

the Nano probe reproduces:

- `n_tile=0 k_block=0` -> `0x7274766e`
- `n_tile=0 k_block=1` -> `0x6e717376`

Those are exactly the live `n_tile=1` post words.

So the corrected scale picture is:

- the live current scale window is in the **late-K region**, not at `block_base=0`
- at least part of the remaining structure is explainable by **row-group
  selection** (`row=0` vs `row=64`)
- the old “broad arbitrary SFA corruption” diagnosis was overstated

### Candidate report

I added:

- `proj-2026-04-14-2348/analyze_live_weight_scale_post.py`

and saved its first report to:

- `archive/live_scale_probe_tactic5_weight_block000_v2/post_scale_candidate_report.txt`

High-signal candidates from that report:

- `n_tile=0 k_block=0 post=0x74757174` has an exact hit at `(row=0, block_base=160)`
- `n_tile=0 k_block=1 post=0x72757372` has an exact hit at `(row=0, block_base=164)`
- `n_tile=1 k_block=0 post=0x7274766e` has an exact hit at `(row=64, block_base=160)`
- `n_tile=1 k_block=1 post=0x6e717376` has an exact hit at `(row=64, block_base=164)`
- `n_tile=2 k_block=0 post=0x706d7276` has an exact hit at `(row=0, block_base=156)`
- `n_tile=3 k_block=0 post=0x71747174` has an exact hit at `(row=64, block_base=156)`

### Updated interpretation

The scale path is still implicated by the direct-pack `unit` override result,
but the boundary-level story is now narrower:

- a large part of the earlier live-vs-local mismatch was just a **bad oracle**
  (`pre` instead of `post`)
- another large part was a **bad global-K assumption** (`block_base=0` instead
  of the late-K window actually present at the current probe point)

The next useful move is to make the Nano scale probe flexible enough to model
the same `(row_group, block_base, active_k_block)` pattern across `n_tile`
groups, then compare that against Phase 3 behavior before touching production.

## 2026-04-15 runtime mainloop A-scale probe

I added a production-path A-scale probe to:

- `runtime/src/backend/fused_moe_prefill/nano_p1_kernel.cuh`
- `testing/backend/nano_p1_mainloop_oracle_test.cpp`

The new probe records, for the real Nano kernel loop:

- `tid`
- `k_step`
- `k_base`
- `block_base`
- `n_tile`
- `k_block`
- `part_token_row0`
- `part_output_col0`
- post-copy `rSFA` words (`reg_post`)

The baseline run was saved to:

- `archive/runtime_mainloop_a_scale_probe_baseline.txt`

and the late-K extract to:

- `archive/runtime_mainloop_a_scale_probe_late_k_extract.txt`

### Result

The real runtime kernel reaches the same late-K weight-scale window as the
live FlashInfer probe, and it reproduces the key post-copy register words
exactly.

At:

- `k_step=20`
- `k_base=2560`
- `block_base=160`

the runtime mainloop probe reports:

- `tid=0,2 n_tile=0 k_block=0` -> `0x74757174`
- `tid=0,2 n_tile=0 k_block=1` -> `0x72757372`
- `tid=0,2 n_tile=1 k_block=0` -> `0x7274766e`
- `tid=0,2 n_tile=1 k_block=1` -> `0x6e717376`
- `tid=0,2 n_tile=2 k_block=0` -> `0x72757372`
- `tid=0,2 n_tile=2 k_block=1` -> `0x00000000`
- `tid=0,2 n_tile=3 k_block=0` -> `0x6e717376`
- `tid=0,2 n_tile=3 k_block=1` -> `0x00000000`
- `tid=0,2 n_tile>=4` -> `0x00000000`

Those first four words are exactly the live tactic-5 `scale_reg_post` values
for the active `n_tile=0,1` slice.

The neighboring runtime step at:

- `k_step=19`
- `k_base=2432`
- `block_base=152`

also produces the late-K words:

- `0x706d7276`
- `0x71747174`

which are the same values that showed up in the live capture's smaller
late-window group.

### Updated interpretation

This materially changes the diagnosis again:

- the real production Nano kernel's weight-scale register boundary is not
  obviously wrong
- the earlier scale-path suspicion was mostly an oracle problem
- the remaining Phase 3 failure is more likely on the weight FP4 byte path
  than on the weight-scale path

The best next move is now to return to the weight preload rule in
`nano_p1_kernel.cuh`, using the live/runtime weight-byte evidence, rather than
spending more time on SFA first.

## `nano_p1_layout_compare`: broad bundle mismatch mostly ruled out

I added and built:

- `testing/backend/nano_p1_layout_compare.cu`

and saved the run output to:

- `archive/post_failed_trial_revert/nano_p1_layout_compare.txt`

### Result

The builder-derived traced P5 contract and the hand-assembled Nano P1 contract
match on the weight-byte path:

- `same_tiled_mma=1`
- `same_smem_layout_a=1`
- `same_smem_layout_b=1`
- `same_copy_atom_a=1`
- `same_copy_atom_b=1`
- `same_accum_layout=1`
- `same_part_c_layout=1`
- `same_tCrA_layout=1`
- `same_tCrB_layout=1`
- `same_tCsA_layout=1`
- `same_tCsB_layout=1`
- `same_tCrA_cv_layout=1`
- `same_tCrB_cv_layout=1`

The only structural difference reported by this compare probe is the scale
shared-memory layout type:

- `same_smem_layout_sfa=0`
- `same_smem_layout_sfb=0`
- but `same_copy_atom_sfa=1`
- and `same_copy_atom_sfb=1`

The scale layouts still agree on total/stage sizes:

- traced: `sfa_cosize=4096 sfb_cosize=4096 sfa_stage=1024 sfb_stage=1024`
- nano: `sfa_cosize=4096 sfb_cosize=4096 sfa_stage=1024 sfb_stage=1024`

The printed layout strings differ:

- traced SFA/SFB:
  `(((_32,_4),_1),((_16,_4),_1,_2),_4):(((_16,_4),_512),((_0,_1),_4,_512),_1024)`
- nano SFA/SFB:
  `(((_32,_4),_1),((_16,_4),_2),(_1,_4)):(((_16,_4),_0),((_0,_1),_512),(_0,_1024))`

### Updated interpretation

This rules out the broad theory that the manual Nano P1 MMA bundle is simply
the wrong CUTLASS/CUTE contract for the weight-byte path. The A/B operand and
accumulator-side contract matches exactly in this probe.

The remaining structural mismatch is limited to the scale shared-memory layout
type. That is still worth keeping in mind, but it is no longer the best lead,
because the live/runtime late-K scale-register probe already matched exactly on
the real kernel path.

So the best current read is:

- broad `TiledMma` / A/B copy-view mismatch is not the bug
- scale layout type mismatch exists, but has not yet shown up at the
  post-copy live/runtime boundary
- the strongest remaining suspect is still the weight FP4 byte preload /
  fragment-consumption path in `nano_p1_kernel.cuh`

## Production A-operand probe: real kernel still misses live weight registers

I added a production-path A-byte probe to:

- `runtime/src/backend/fused_moe_prefill/nano_p1_kernel.cuh`
- `testing/backend/nano_p1_mainloop_oracle_test.cpp`

and saved the baseline run to:

- `archive/post_failed_trial_revert/mainloop_with_a_operand_probe.txt`

I also saved the direct overlap compare against the live tactic-5 fixed-CTA
weight capture to:

- `archive/post_failed_trial_revert/production_a_operand_vs_live_weight_overlap.txt`

### Result

The good news is that this removed another probe ambiguity.

The real kernel only emitted 16 `probe a_operand` entries for the tracked
weight-byte surface:

- tids: `32,48,64,80`
- `n_tile=0,1`
- `k_block=0,1`

It did **not** emit the old standalone-probe `n_tile=2,3` families. That
means the earlier standalone A-probe evidence about:

- `n_tile=2,3` zero-vs-nonzero mismatches

was not a trustworthy production-kernel signal and should be discarded.

But the stronger result is that the overlapping real-kernel entries still miss
the live tactic-5 weight probe completely.

On the 16 shared keys:

- `(tid, n_tile, k_block)` overlap count = `16`
- `part_output_col0` / `part_token_row0` match on all 16
- exact `reg_post` matches = `0`
- exact `reg_post` mismatches = `16`

So the real production kernel is still wrong on the weight-byte register
surface, even after removing the standalone-probe artifact.

### Updated interpretation

This is the clearest current state:

- the broad CUTLASS/CUTE A/B bundle mismatch theory is ruled out
- the standalone A probe over-modeled extra tiles, so its `n_tile=2,3`
  mismatch family is not production evidence
- the real kernel still fails the live weight-register oracle on every
  overlapping `n_tile=0,1` entry

That means the bug remains on the real weight FP4 byte path before or at the
`fp4_shift_A` / consumed-register boundary. The next debugging move should stay
on that surface, but it should use the production A-operand probe as the
runtime-side reference, not the older standalone A-probe tile count.

## Activation byte synthetic: producer map fixed, consumer decode still wrong

I continued the tactic-5 activation-side work on the widened valid live
synthetic capture:

- live:
  `archive/current_tactic5_activation_byteindex_tidscan_v7/operand_probe_tactic5.txt`
- runtime baseline:
  `archive/current_tactic5_activation_byteindex_tidscan_v7/nano_b_identity.txt`

The live probe already had the producer-side information needed to explain the
remaining activation mismatch:

- `producer_debug` for stage-0 slot `0` comes from `src=(0,0,0)`
- slot `256` comes from `src=(4,0,0)`
- slot `258` comes from `src=(4,4,0)`

That implies the correct synthetic source tag on this surface is not plain
`byte_index`, but:

- `8 * source_row + byte_index`

I added a probe-only switch to `testing/backend/nano_p1_b_operand_probe.cu`:

- `NEMOTRON_NANO_P1_B_SYNTHETIC_LINEAR_ROW8=1`

and saved the output to:

- `archive/current_tactic5_activation_byteindex_tidscan_v7/probe_variants/nano_b_identity_linearrow8_byteindex_tidscan_v7.txt`

### Result

This settled one layer cleanly.

Against the widened live tactic-5 activation byte synthetic:

- geometry still matches `128/128`
- `copy_sig` now matches `128/128`
- `reg_pre` still matches only `12/128`

So the stage-slot / source-byte mapping is now explained exactly, but the
consumer-side register payload is still wrong.

This is the strongest activation result so far:

- producer/source mapping before the consumer boundary is no longer the lead
  suspect
- the remaining activation bug is between `stage0_B` and the consumed `tCrB`
  register payload

### What I ruled out next

I also generated a synthetic input file whose packed bytes are exactly:

- `value = 8 * row + byte_index`

saved at:

- `archive/current_tactic5_activation_byteindex_tidscan_v7/probe_variants/nano_b_linearrow8_input_128x1344.bin`

Then I re-ran the local probe with the two obvious packed-to-smem staging
variants:

- `input_stage_mode=unpack_nibbles`
- `input_stage_mode=unpack_nibbles_swap_halves`

saved at:

- `archive/current_tactic5_activation_byteindex_tidscan_v7/probe_variants/nano_b_identity_linearrow8_input_unpack.txt`
- `archive/current_tactic5_activation_byteindex_tidscan_v7/probe_variants/nano_b_identity_linearrow8_input_unpackswap.txt`

Those did **not** fix the register mismatch:

- `unpack_nibbles`: `reg_pre` matches `8/128`
- `unpack_nibbles_swap_halves`: `reg_pre` matches `0/128`

So the remaining activation consumer mismatch is **not** explained by:

- the old wrong stage-slot/source mapping
- a simple packed-byte vs nibble-unpack staging choice
- a simple half-swap of the unpacked nibbles

### Updated interpretation

At this point the activation-side search space is much smaller.

What is fixed:

- the widened tactic-5 activation producer/source map
- the exact `copy_sig` stage-slot contents on the valid live synthetic

What is still wrong:

- the register payload seen after the local B-side consumer copy

So the next valid move is to probe an alternate **consumer** contract, not
another source-row/source-byte staging guess. The most plausible next target is
the exact live `tCrB` consumer path itself:

- either a traced P5/FlashInfer-aligned B consumer using the real builder types
- or a minimal local experiment that swaps only the consumer view/layout while
  keeping the now-correct stage-slot source mapping fixed

## Correction: synthetic reg-pre was the wrong lead

I revisited that conclusion against the local repo-managed traced P5 types and
the live probe fields, and the “consumer contract mismatch” diagnosis was too
aggressive.

Running:

- `./build/testing/nano_p1_layout_compare`

and saving the output to:

- `archive/nano_p1_layout_compare_2026-04-15.txt`

shows that traced P5 and Nano P1 already agree on the B-side consumer contract
that matters here:

- `same_copy_atom_b=1`
- `same_tCrB_layout=1`
- `same_tCsB_layout=1`
- `same_tCrB_cv_layout=1`

So the repo-managed traced P5 consumer and the local Nano consumer are not
using different B copy atoms or fragment layouts.

The live probe output also shows why the synthetic `reg_pre` comparison was not
portable. On:

- `archive/current_tactic5_activation_byteindex_tidscan_v7/operand_probe_tactic5.txt`

the live probe reports three different layers:

- `copy_sig`: the literal staged shared-memory byte
- `raw_copy_view_components`: a deeper per-component sequence inside the copy
  view
- `reg_pre`: the packed register payload after those component IDs are
  materialized

Example, `tid=60, n_tile=3, k_block=0`:

- `copy_sig=525:38`
- `raw_copy_view_components=(4,13,32,...),(4,13,33,...),...,(4,13,63,...)`
- `reg_pre=(0x03020100,0x13121110)`

So `reg_pre` is not a direct byte-for-byte echo of the staged shared-memory
byte tag. Our probe-only `8 * row + byte_index` synthetic was enough to prove
the stage-slot producer mapping, but it was not enough to make `reg_pre` a
trustworthy cross-implementation oracle.

Updated correction:

- the synthetic activation `copy_sig` result is still valid
- the synthetic activation `reg_pre` mismatch is **not** good evidence of a
  Nano-only B consumer bug

## Real-input activation mismatch is still real

What *is* still trustworthy is the real-input stage-byte mismatch.

The compare in:

- `archive/late_k_production_a_probe_v1/activation_vs_runtime_b_compare_live_input_override.txt`

uses the same live artifact as:

- `archive/current_tactic5_activation_block000_v7/operand_probe_tactic5.txt`
- `archive/current_tactic5_activation_block000_v7/gemm1_input_tactic5.bin`

I added:

- `proj-2026-04-14-2348/analyze_activation_real_input_stage_bytes.py`

and saved its report to:

- `archive/current_tactic5_activation_block000_v7/real_input_stage_byte_report.txt`

That report checks whether the live `copy_sig` bytes match the dumped
`gemm1_input_tactic5.bin` bytes under simple hypotheses. On the 12 saved sample
entries:

- direct same-coordinate byte match: `0 / 12`
- same-row nearby-byte match within `[-8,+8]`: `0 / 12`
- simple byte transforms (`swap_nibbles`, `dup_lo`, `dup_hi`, `xor_ff`,
  `xor_55`, `xor_aa`): all `0 / 12`
- adjacent-byte nibble-pair recombinations: all `0 / 12`

Representative samples:

- `key=(32,0,0)`: file byte `174`, live stage byte `209`
- `key=(32,1,0)`: file byte `28`, live stage byte `158`
- `key=(32,2,0)`: file byte `98`, live stage byte `145`
- `key=(32,3,0)`: file byte `218`, live stage byte `65`

So the remaining activation mismatch is still on the real path, but the right
interpretation is narrower:

- we have **not** proved a B consumer/layout bug
- we **have** proved that the current real-input source-byte interpretation is
  wrong or incomplete

The next valid step is at the producer boundary:

- extend the live producer probe so it records the actual producer-side source
  value for the tracked stage byte slots, not just semantic coords
- then compare those producer-side source values against
  `gemm1_input_tactic5.bin` and the runtime `copy_sig`

Until that producer-value probe exists, treating the real-input mismatch as a
consumer bug would still be guesswork.

## 2026-04-15 late-K activation realignment

The apparent live-probe contradiction in
`archive/current_tactic5_activation_block000_v14/operand_probe_tactic5.txt`
turned out to be a probe-timing issue, not a raw-SMEM contradiction:

- `stage_byte_snapshot` is captured only once, on the **first** time the lead
  tracked thread hits `RecordOperandPreImpl`
- each `entry=(tid,n_tile,k_block)` record is overwritten and therefore ends up
  showing the **last** time that entry was seen

So the snapshot was showing an early stage-0 read, while the final entries were
showing the late-K stage-0 contents. The live per-entry `copy_sig` data is still
usable; the global snapshot was the misleading surface.

With that corrected, I reran the local standalone B probe on the exact live
activation dump:

- input: `archive/current_tactic5_activation_block000_v15/gemm1_input_tactic5.bin`
- row rule: `identity`
- packed-row bytes: `1344`
- packed-byte offset: `1280`

Artifacts:

- live: `archive/current_tactic5_activation_block000_v15/operand_probe_tactic5.txt`
- local: `archive/current_tactic5_activation_block000_v15/runtime_probe_identity_offset1280.txt`

This resolves the activation producer/stage-byte question:

- local `copy_sig` vs live `copy_sig`: `64 / 64` matches

So the activation producer/staging path is not the blocker at this late-K
surface.

What remains is narrower:

- live `reg_pre` vs local `byte_tag_reg*_pre`: `32 / 64` exact matches
- `tid=32` and `tid=64`: `16 / 16` matches each
- `tid=48` and `tid=80`: `0 / 16` matches each

For the mismatching odd-lane class, the pattern is exact:

- live `(tid=48 or 80, n_tile=X, k_block=0)` equals local `k_block=1`
- live `(tid=48 or 80, n_tile=X, k_block=1)` equals local `k_block=0`

Examples:

- `(48,0,0)`: live `0x49563915,0xb5bdcfb1`; local `0x3e9ae3ac,0x2afec911`
- `(48,0,1)`: live `0x3e9ae3ac,0x2afec911`; local `0x49563915,0xb5bdcfb1`
- `(80,1,0)`: live `0xb9ece7f0,0x93e91141`; local `0xcb427b56,0x3edbd159`
- `(80,1,1)`: live `0xcb427b56,0x3edbd159`; local `0xb9ece7f0,0x93e91141`

I also patched the standalone probe to copy B per `k_block` instead of bulk
copying the whole `tCrB_cv` tensor:

- `archive/current_tactic5_activation_block000_v15/runtime_probe_identity_offset1280_perkblock.txt`

That did **not** change the result:

- stage bytes still match `64 / 64`
- odd-lane `k_block` swap in `reg_pre` still remains

Current interpretation:

- activation staging into `smem_B` is cleared at this late-K live surface
- the remaining activation discrepancy is an odd-lane `k_block` ordering issue
  at the B fragment/register boundary
- because the per-`k_block` probe copy did not fix it, this is probably not
  just the bulk-copy call site in the standalone probe

The next valid move is to determine whether this `k_block` swap is:

1. only a probe-labeling / fragment-indexing difference, or
2. a real production mismatch between B and its paired scale / MMA consumption

The clean way to answer that is a production-kernel late-K B-fragment probe on
the same `(tid,n_tile,k_block)` surface, rather than another blind kernel edit.

## 2026-04-15 production overwrite check

I added one more runtime B-fragment snapshot to the production Phase 3 probe:

- `reg_after_own_b_copy`: captured immediately after the thread copies its own
  `k_block`
- `reg_after_all_b_copies`: captured after the full B copy loop
- `reg_post`: captured at the existing post-copy probe point

Artifacts:

- production log:
  `archive/late_k_production_b_probe_v6_overwrite/mainloop_live_input.log`
- live reference:
  `archive/current_tactic5_activation_block000_v15/operand_probe_tactic5.txt`

This falsified the current overwrite hypothesis:

- tracked late-K production entries: `64`
- `reg_after_own_b_copy != reg_after_all_b_copies`: `0`
- `reg_after_all_b_copies != reg_post`: `0`

So the production kernel is **not** clobbering tracked B fragments after their
own copy, and the existing late-K activation mismatch is not explained by
cross-`k_block` overwrite in the B copy loop.

The live-vs-production late-K compare on the same keyed surface is now:

- geometry fields (`part`, `local0`, `local1`, `stage0_offsets`): `64 / 64`
  matches
- direct `reg` matches: `16 / 64`
- `k_block`-swapped `reg` matches: `16 / 64`
- remaining `reg` entries with neither direct nor swapped match: `32 / 64`

The structure is exact:

- `tid=32,64`: even `n_tile` direct, odd `n_tile` neither direct nor swapped
- `tid=48,80`: even `n_tile` swapped, odd `n_tile` neither direct nor swapped

So the remaining production mismatch is the **odd-`n_tile` half** of the late-K
activation consumer surface, not a global odd-lane bug and not a copy-loop
overwrite.

One more concrete source-level inconsistency showed up while tracing that path:

- `NanoP1PartitionScaleB(...)` in `runtime/src/backend/fused_moe_prefill/nvfp4_bridge.cuh`
  builds `thr_vnk` with `cute::get<2>(thr_vmnk)`
- the generic `PartitionScaleB(...)` in the same file uses `cute::get<1>(thr_vmnk)`

I tried to validate that discrepancy by adding a second, probe-only generic
`PartitionScaleB` fragment to the production kernel, but that inflated the
debug build enough that `RunNanoP1KernelForTesting` stopped launching cleanly.
I reverted that experiment.

So the next clean move is:

1. keep the production overwrite probe in place
2. move the generic-vs-Nano `PartitionScaleB` comparison into an isolated
   reference probe, not the production kernel
3. use that isolated probe to test whether the scale partition discrepancy
   explains the odd-`n_tile` activation mismatch

## 2026-04-16 isolated `PartitionScaleB` result

I patched `testing/backend/nano_p1_reference_b_operand_probe.cu` to compare the
generic `PartitionScaleB(...)` mapping against `NanoP1PartitionScaleB(...)`
without materializing both scale fragments through a second `cute::copy`. The
first direct-copy attempt faulted with an illegal SMEM access, so the final
reference harness uses the same physical-copy-view coordinate extraction that
`trt_helpers_pre_nano_epilogue.cuh` uses:

- build target:
  `build-sm120-relwithdebinfo/testing/nano_p1_reference_b_operand_probe`
- log:
  `archive/reference_b_scale_partition_compare_v1/reference_probe.log`
- comparison summary:
  `archive/reference_b_scale_partition_compare_v1/compare_summary.json`

Run configuration:

- tracked tids: `32,48,64,80`
- scale file:
  `proj-2026-04-12-1022/trtllm_reference/golden_nano_k2688/input_sf_permuted.bin`
- hidden size: `2688`
- late-K scale block base: `160`

Results:

- isolated generic vs Nano packed scale payloads: `64 / 64` identical
- isolated generic vs live `scale_reg_post`: `0 / 64`
- isolated Nano vs live `scale_reg_post`: `0 / 64`

So the source-level `thr_vnk` inconsistency between the generic and Nano B-scale
partition helpers does **not** produce a visible payload difference in this
isolated late-K reference harness. That makes it a weak lead for the current
odd-`n_tile` activation mismatch.

Important caveat: this isolated reference harness also does **not** reproduce
the live activation `scale_reg_post` bytes at all, so it should not be promoted
to an oracle for absolute scale correctness. What it does tell us is narrower:

- `generic PartitionScaleB` vs `NanoP1PartitionScaleB` is not the source of the
  current late-K odd-`n_tile` split, at least not in a way that survives into
  the packed scale register payload under this local harness

That shifts the highest-value next step back to the activation **B fragment**
consumer surface, not the B-scale partition helper. The remaining live-vs-prod
structure is still:

- even `n_tile`: partly direct / partly `k_block`-swapped
- odd `n_tile`: neither direct nor swapped for the affected tracked lanes

So the next best move is to normalize or instrument the odd-`n_tile` B-fragment
consumer contract directly, rather than spending another cycle on `PartitionScaleB`.

## 2026-04-16 late-K B consumer dead ends cleared

Two more narrow hypotheses are now cleared on the same late-K activation
surface:

1. alternate B copy atom
2. SM120 FP4 B-lane shift

### Alternate B copy atom

The standalone late-K activation probe in
`archive/current_tactic5_activation_block000_v15/runtime_probe_identity_offset1280_perkblock.txt`
already records both:

- default B payload: `byte_tag_reg{0,1}_pre`
- alternate B payload: `byte_tag_reg{0,1}_pre_alt`

I parsed those against the live tactic-5 activation probe in
`archive/current_tactic5_activation_block000_v15/operand_probe_tactic5.txt`.

Results:

- parsed runtime entries: `64`
- `byte_tag_reg*_pre_alt == byte_tag_reg*_pre`: `64 / 64`
- default direct matches vs live: `32 / 64`
- alternate direct matches vs live: `32 / 64`
- default `k_block`-swapped matches vs live: `32 / 64`
- alternate `k_block`-swapped matches vs live: `32 / 64`

So the alternate Nano B copy atom is not changing the payload at all on this
surface, and it does not explain the odd-`n_tile` mismatch.

### `ApplySm120Fp4ShiftB`

`ApplySm120Fp4ShiftB(...)` in
`runtime/src/backend/fused_moe_prefill/trt_helpers_pre_nano_epilogue.cuh`
is just:

- `b0 <<= 2`
- `b1 <<= 2`

I applied that transform offline to the runtime late-K `byte_tag_reg*_pre`
values and compared the shifted result to live `reg_pre`.

Results:

- raw direct matches: `32 / 64`
- shifted direct matches: `0 / 64`
- raw `k_block`-swapped matches: `32 / 64`
- shifted `k_block`-swapped matches: `0 / 64`

So the remaining mismatch is not “missing `ApplySm120Fp4ShiftB`”.

That leaves the open issue where it already looked strongest:

- a real odd-`n_tile` discrepancy in the activation B fragment/register
  consumer contract, not a scale-partition issue, not an alternate copy-atom
  issue, and not the SM120 B shift helper

## 2026-04-16: activation B row fix was real; remaining B mismatch is oracle labeling

I reran the late-K production activation probe after the production
`source_row = row_start + row` patch in
`runtime/src/backend/fused_moe_prefill/nano_p1_kernel.cuh`.

Current production log:

- `archive/post_row_identity_patch/mainloop_live_input.log`

Reference standalone late-K probe:

- `archive/current_tactic5_activation_block000_v15/runtime_probe_identity_offset1280_perkblock.txt`

Live late-K activation probe:

- `archive/current_tactic5_activation_block000_v15/operand_probe_tactic5.txt`

Results:

- current production vs standalone `reg_post`: `64 / 64`
- current production vs standalone `frag_raw_packed`: `64 / 64`
- current production vs live `reg_pre` direct: `32 / 64`
- current production vs live `reg_pre` with `k_block` relabeled on `tid=48,80`: `64 / 64`

So the activation row-identity patch fixed a real production bug. The
remaining activation B register mismatch is not a data mismatch anymore; it is
explained by a lane-class-specific `k_block` labeling difference in the live
oracle for `tid=48,80`.

## 2026-04-16: traced P5 and Nano are identical on synthetic B and synthetic SFB

I extended `testing/backend/nano_p1_layout_compare.cu` to compare both `B`
register fragments and `SFB` scale fragments under the same synthetic stage-0
payloads.

Saved output:

- `archive/layout_compare_sfb_v1/nano_p1_layout_compare.txt`

Results:

- `runtime_b_fragment_compare_summary direct_pairs=32 swapped_pairs=0 other_pairs=0`
- `runtime_b_scale_fragment_compare_summary direct_pairs=32 swapped_pairs=0 other_pairs=0`

So there is no Nano-only consumer bug on the local synthetic surface for either
activation payload bytes or activation scales. Traced P5 and Nano produce the
same fragments.

## 2026-04-16: old activation-scale reference probe was wrong; actual traced SFB still does not match live

I patched `testing/backend/nano_p1_reference_b_operand_probe.cu` to dump the
actual traced `partition_fragment_SFB(...)` words in addition to the older
coordinate-reconstruction path.

Saved output:

- `archive/reference_b_scale_partition_compare_v2/reference_probe.log`
- `archive/reference_b_scale_partition_compare_v2/compare_summary.json`

Results against live `scale_reg_post` from
`archive/current_tactic5_activation_block000_v15/operand_probe_tactic5.txt`:

- shared entries: `64`
- old coordinate reconstruction (`scale_reg_generic`) matches: `0 / 64`
- actual traced `partition_fragment_SFB` matches: `4 / 64`
- actual Nano `NanoP1PartitionScaleB` fragment matches: `4 / 64`
- actual traced and actual Nano match each other: `64 / 64`

So the old coordinate-based activation-scale probe really was wrong, but fixing
it did **not** make the live mismatch disappear.

One more important comparison:

- production late-K `scale_reg_post` vs actual traced reference: `16 / 64`

That means the shared local reference harness is still not equivalent to the
production late-K activation-scale surface, and the direct live-vs-local
activation-scale comparison remains unstable.

Current interpretation:

- activation payload bytes are largely cleared
- activation scales are still unresolved
- but the remaining activation-scale evidence is now blocked more by oracle
  alignment / surface equivalence than by an identified Nano-only bug
