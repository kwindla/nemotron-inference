# NanoP1 Reference A Operand Probe (2026-04-14)

This note records the first standalone **reference-side** operand probe added
after the runtime-side `nano_p1_b_operand_probe`.

The goal of this probe is narrower than the runtime-side tag probe:

- instantiate the TRT-LLM-equivalent P1 `CollectiveMainloop` bundle locally
- inspect the dense activation-side copy-view contract for tracked threads
- avoid more guesswork about what the reference kernel thinks A-consumer
  coordinates should be

## What Was Added

- Diagnostic binary:
  `testing/backend/nano_p1_reference_a_operand_probe.cu`
- Build target:
  `nano_p1_reference_a_operand_probe`

The probe instantiates a local SM120 block-scaled `CollectiveMainloop` with:

- `MmaTileShape = (128, 128, 64)`
- `ClusterShape = (1, 1, 1)`
- `LayoutA = RowMajor`
- `LayoutB = ColumnMajor`
- `KernelScheduleAuto`
- `SwapAB = false` semantics encoded by `MainloopElementA = act`,
  `MainloopElementB = weight`

It then builds the dense A-side copy-view using:

- `thread_mma.partition_B(dense_a)`
- `make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma)`
- `smem_thr_copy_a.retile_D(part_a)`

and dumps anchor coordinates plus their corresponding stage-0 `SmemLayoutA`
offsets for `tid={0,1,128,129}`.

## Command

```bash
cmake -S . -B build-sm120-relwithdebinfo
cmake --build build-sm120-relwithdebinfo --target nano_p1_reference_a_operand_probe --parallel $(nproc)
./build-sm120-relwithdebinfo/testing/nano_p1_reference_a_operand_probe
```

## Key Output

The probe reports:

- `MmaTileShape=(128,128,64)`
- `threads=256`
- `stages=9`
- `copy_view_a sizes=(64,4,1)`
- `part_c sizes=(4,2,8)`

Representative output:

```text
tid=0:
  copy_tile=0 part_token_row0=0  anchors=(0,0@0) (0,32@0) (0,64@0) (0,96@0) (8,0@512) (8,32@512) (8,64@512) (8,96@512)
  copy_tile=1 part_token_row0=8  anchors=(32,0@2048) (32,32@2048) (32,64@2048) (32,96@2048) (40,0@2560) (40,32@2560) (40,64@2560) (40,96@2560)

tid=1:
  copy_tile=0 part_token_row0=2  anchors=(0,8@16) (0,40@16) (0,72@16) (0,104@16) (8,8@528) (8,40@528) (8,72@528) (8,104@528)
  copy_tile=1 part_token_row0=10 anchors=(32,8@2064) (32,40@2064) (32,72@2064) (32,104@2064) (40,8@2576) (40,40@2576) (40,72@2576) (40,104@2576)

tid=128:
  copy_tile=0 part_token_row0=16 anchors=(16,0@1024) (16,32@1024) (16,64@1024) (16,96@1024) (24,0@1536) (24,32@1536) (24,64@1536) (24,96@1536)
  copy_tile=1 part_token_row0=24 anchors=(48,0@3072) (48,32@3072) (48,64@3072) (48,96@3072) (56,0@3584) (56,32@3584) (56,64@3584) (56,96@3584)
```

The `@offset` suffix is the exact `stage0_A(row, dense_col * 2)` offset inside
`SmemLayoutA` stage 0.

## What This Proves

1. The reference activation-side copy-view is **not row-only**.
   Even before shared-memory tag probing, the dense contract already varies
   across:
   - `copy_tile`
   - intra-tile anchor group
   - byte band within the 64-byte K slice

2. The reference-side A contract naturally groups rows in:
   - 32-row macro bands (`0/32/64/96` or `16/48/80/112`)
   - with an 8-row inner offset (`0/8`, `32/40`, `16/24`, etc.)

3. The stage-0 offsets confirm that those bands are physically far apart in
   shared memory, not just a pretty dense-view artifact:
   - `copy_tile=0 -> row-band base offsets 0 / 512 / 1024 / 1536`
   - `copy_tile=1 -> row-band base offsets 2048 / 2560 / 3072 / 3584`
   - so the reference contract is stepping by whole swizzled row bands, not
     by a trivial row-only permutation

4. So the remaining NanoP1 bug is even less likely to be a simple
   row-permutation than the runtime-side B probe already suggested.
   The reference-side contract is explicitly 2D in `(row band, byte band)`.

## What This Does *Not* Yet Prove

- It does **not** yet give the final stage-0 shared-memory consumer offsets.
  This probe is still at the dense-copy-view layer; it computes exact
  `SmemLayoutA` offsets for those anchors, but it does not yet tag stage-0 A
  and read back the consumed tags from a live `partition_S(...)->copy->fp4_shift`
  path.
- It does **not** yet map one-to-one onto the runtime-side
  `consumed_source_row0` values from `nano_p1_b_operand_probe`.
  The two probes are looking at adjacent but not identical layers.

## Follow-up Blocker

An attempted next step in this same file was to deepen the probe into an exact
tag-filled `stage0_A -> partition_S(sA) -> copy -> fp4_shift_A` path, mirroring
`nano_p1_b_operand_probe`.

That attempt was reverted after it hit a CUTE builder limitation:

- `partition_fragment_A(...)` on the host-side identity / logical staging path
  does not instantiate cleanly for this builder configuration
- assuming the dense-copy-view's `copy_tile=4` shape also held for the lower
  `tCsA` source view was incorrect
- the target was restored to the last working offset-bearing probe instead of
  leaving it broken

## Immediate Next Step

Do **not** keep pushing the host-only builder probe further blind. The next
useful step is to add the mirrored operand probe in the patched flashinfer
reference kernel itself, where the true `sA/tCsA/tCrA_cv` live path already
exists and the exact thread contract can be dumped without reconstructing it
through a host-side identity tensor.
