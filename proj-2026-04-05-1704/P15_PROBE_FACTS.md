# P15 Probe Facts

Everything recovered from offline builder probes about the CUTLASS SM120
block-scaled FP4 tactic `P15` (`256x128x64`, `swap_ab=true`, `cluster=1x1x1`).

Where relevant, P13 (`128x128x64`, `swap_ab=true`) is shown side-by-side as the
closest working reference.

All data is for **thread 0** of the `TiledMma` unless noted otherwise.

---

## 0. Additional 2026-04-08 Probes

### Scale Register Fragment Probe

Source: `artifacts/benchmarks/trt_p15_scale_register_fragment_dump_20260408.log`

This probe dumps the **stage-0 traced P15 scale register fragment contract** as
instantiated from the CUTLASS collective.

```
Stages = 6
tCrSFA_raw.layout    = (_64,(_2,_2),(_16,_4)):(_0,(_64,_128),(_1,_16))
tCrSFB_raw.layout    = (_64,(_2,_4),(_16,_4)):(_0,(_256,_64),(_1,_16))
tCrSFA_raw.shape     = (_64,_4,_64)
tCrSFB_raw.shape     = (_64,_8,_64)
tCrSFA_raw.cosize    = _256
tCrSFB_raw.cosize    = _512
tCrSFA_raw_k0.cosize = _193
tCrSFB_raw_k0.cosize = _449
```

This is important because it shows the traced CUTLASS scale fragments are
**not compact atom-scale packs**. They are large, sparse, broadcast-heavy
register views. That means the native `P15` kernel should not try to recreate a
full `tCrSFA/tCrSFB` equivalent just to feed the MMA atom. For the live atom
path, a separate flat `uint32` row-scale buffer is the simpler and safer
production contract.

### Atom-to-Scale Row Bounds Probe

Source: `artifacts/benchmarks/trt_p15_scale_row_bounds_dump_20260408.log`

This probe evaluates the candidate scale-row mapping based on
`thread_mma.partition_C(identity_tensor)` and the base coordinate `c_atom(0)`.

Results:

```
a_base_bounds_ok = 1
b_base_bounds_ok = 1
a_band_uniform   = 1
b_band_uniform   = 1
min_a_row = 0
max_a_row = 255
min_b_row = 0
max_b_row = 127
first_bad_thread = -1
```

So for every live `P15` atom tile:
- the candidate base rows are in-bounds
- each atom stays within one M scale band (`128`)
- each atom stays within one N scale band (`128`)

That makes `c_atom(0)` a safe base-coordinate source for flat row-scale lookup.

---

## 1. Tile Shape and Pipeline

| Property | P13 | P15 |
|---|---|---|
| **tile_mnk (atom)** | `(128, 32, 64)` | `(128, 32, 64)` |
| **Profile tile (full CTA)** | `128 x 128` | `256 x 128` |
| **Stages** | 9 | 6 |
| **size(TiledMma)** | 256 | 256 |
| **cluster** | `1x1x1` | `1x1x1` |

The atom tile is identical. P15 doubles the M dimension of the full CTA tile
from 128 to 256, which drives all the downstream contract differences.

Sources:
- `artifacts/benchmarks/trt_p13_runtime_layout_dump_20260407.log` line 1
- `artifacts/benchmarks/trt_p15_runtime_layout_dump_20260407.log` line 1

---

## 2. Shared-Memory Sizes

| Property | P13 | P15 |
|---|---|---|
| **cosize(SmemLayoutSFA)** | 4608 | 6144 |
| **cosize(SmemLayoutSFB)** | 4608 | 3072 |

P15 grows SFA smem by 33% and shrinks SFB smem by 33%. The total SF smem
budget is 9216 for P13 vs 9216 for P15 (same total, different split).

Sources:
- P13: `trt_p13_runtime_layout_dump_20260407.log` line 4
- P15: `trt_p15_runtime_layout_dump_20260407.log` line 4

---

## 3. Scale Factor Register Fragments

| Property | P13 | P15 |
|---|---|---|
| **size(tCrSFA)** | 128 | 256 |
| **cosize(tCrSFA)** | 8 | 16 |
| **size(tCrSFB)** | 512 | 512 |
| **cosize(tCrSFB)** | 32 | 32 |

P15 doubles the SFA register fragment. SFB is unchanged.

### SFA Copy View Layout

**P13:**
```
tCrSFA_copy_view: ((_1,(_16,_8)),_1,_1):((_0,(_0,_1)),_0,_0)
```

**P15:**
```
tCrSFA_copy_view: ((_1,(_16,_8)),_2,_1):((_0,(_0,_1)),_8,_0)
```

The `_2` in P15's second mode (vs `_1` in P13) is the extra M-tile split.
That `_2` with stride `_8` means P15 holds two independent SFA slices per
k-block, one for each 128-row half of the 256-row M tile.

### SFB Copy View Layout (same for both)

```
tCrSFB_copy_view: ((_1,(_16,_4,_2)),_4,_1):((_0,(_0,_1,_16)),_4,_0)
```

### Smem-Side Scale Layouts

**P15 tCsSFA:**
```
((_1,((_16,_4),_2)),_2,_1,_6):((_0,((_0,_1),_8)),_512,_0,_1024)
```

**P13 tCsSFA:**
```
((_1,((_16,_4),_2)),_1,_1,_9):((_0,((_0,_1),_8)),_0,_0,_512)
```

Key difference: P15 has `_2` in the MMA_M mode with stride `_512`, P13 has `_1`.

Sources:
- `trt_p15_runtime_layout_dump_20260407.log` lines 4-9
- `trt_p15_copy_to_reg_dump_20260407.log` lines 5-8
- `trt_p13_runtime_layout_dump_20260407.log` lines 4-9

---

## 4. Scale-as-C Views (for Accumulator Rescaling)

Source: `trt_p15_scale_as_c_dump_20260407.log`

### Global Parameters

```
Stages = 6
ScaleGranularityM = 128
ScaleGranularityN = 128
ScaleMsPerTile = 2
ScaleNsPerTile = 1
```

P13 would have `ScaleMsPerTile = 1` (128-row tile / 128 granularity = 1).
P15 has `ScaleMsPerTile = 2` (256-row tile / 128 granularity = 2).

This is the key rescaling difference: P15 has **two independent scale-A
values per accumulator column**, one for rows 0..127 and one for rows 128..255.

### Smem Scale Tensors (C-partitioned)

```
sScaleAViewAsC: ((_128,_2),_128,_6):((_0,_1),_0,_2)
sScaleBViewAsC: (_256,(_128,_1),_6):(_0,(_0,_1),_1)
```

### Thread-Partitioned Smem Views

```
tCsScaleAViewAsC: ((_2,_2),(_2,_2),_8,_6):((_0,_0),(_0,_1),_0,_2)
tCsScaleBViewAsC: ((_2,_2),_4,_8,_6):((_0,_0),_0,_0,_1)
```

### Thread-Local Register Views

```
tCrScaleAViewAsC: ((_2,_2),(_2,_2),_8):((_0,_0),(_0,_1),_0)
tCrScaleBViewAsC: ((_2,_2),_4,_8):((_0,_0),_0,_0)
size(tCrScaleAViewAsC) = 128
size(tCrScaleBViewAsC) = 128
```

### Scale A Coordinate Map (thread 0, stage 0)

```
tCcScaleAViewAsC_stage0: ((_2,_2),(_2,_2),(_2,_4)):((_1@1,_8@0@0),(_64@0@0,_1@1@0),(_8@1,_32@1))
```

The `(_2,_2)` in the MMA_M position means the scale view covers **four M-bands**:
rows {0, 64} x {scale_idx 0, 1}. Sample coords (first column only):

```
MMA_M=(0,0): row=0     scale_idx=0
MMA_M=(0,1): row=64    scale_idx=0
MMA_M=(1,0): row=0     scale_idx=1
MMA_M=(1,1): row=64    scale_idx=1
```

### Scale B Coordinate Map (thread 0, stage 0)

```
tCcScaleBViewAsC_stage0: ((_2,_2),_4,(_2,_4)):((_1@0@1,_8@0),_64@0,(_8@0@1,_32@0@1))
```

Four M-bands: rows {0, 64, 128, 192}. Sample coords:

```
MMA_M=0: row=0
MMA_M=1: row=64
MMA_M=2: row=128
MMA_M=3: row=192
```

### Rescale Semantics

In the CUTLASS mainloop (`sm120_mma_array_tma_blockwise_scaling.hpp`
lines 740-770), the `ScaleMsPerTile > 1 && ScaleNsPerTile == 1` branch applies:

```cpp
ElementSF scale_b = tCrScaleBViewAsC.data()[0];
for (int i = 0; i < size(tCrScaleAViewAsC); i++) {
    tCrScaleAViewAsC.data()[i] *= scale_b;
}
// Then:
for (int i = 0; i < size(accum); ++i) {
    accum(i) += tmp_accum(i) * tCrScaleAViewAsC(i);
    tmp_accum(i) = 0;
}
```

Each accumulator element is rescaled by `scaleA[m_band] * scaleB`.

---

## 5. MMA Atom Partition Shapes (partA / partB / partC)

These are the per-atom shapes. Identical for P13 and P15 because the atom is
the same `128x32x64` tile.

### partA

```
layout: ((_8,_2,_2),_2,_1):((_1@1,_8@0,_32@1),_64@0,_0)
size = 64   cosize = 2920
```

Thread 0 coordinates (row, col):
```
Atom elem  MMA_M=0    MMA_M=1
 0          (0,0)      (64,0)
 1          (0,1)      (64,1)
 ...
 7          (0,7)      (64,7)
 8          (8,0)      (72,0)
 ...
15          (8,7)      (72,7)
16          (0,32)     (64,32)
...
31          (8,39)     (72,39)
```

Two row bands: {0,8} and {64,72}. Two K groups: {0..7} and {32..39}.

### partB

```
layout: ((_8,_2),_2,_1):((_1@1,_32@1),_8@0,_0)
size = 32   cosize = 360
```

Thread 0 coordinates:
```
Atom elem  MMA_N=0    MMA_N=1
 0          (0,0)      (8,0)
 1          (0,1)      (8,1)
...
 7          (0,7)      (8,7)
 8          (0,32)     (8,32)
...
15          (0,39)     (8,39)
```

Two row bands: {0} and {8}. Two K groups: {0..7} and {32..39}.

### partC

```
layout: ((_2,_2),_2,_2):((_1@1,_8@0),_64@0,_8@1)
size = 16   cosize = 730
```

Thread 0 coordinates (row, col):
```
Atom (MMA_M, MMA_N):
 (0,0): (0,0)    (0,1): (0,8)
 (1,0): (64,0)   (1,1): (64,8)

MMA_Atom values:
 0: (0,0)   1: (0,1)   2: (8,0)   3: (8,1)
```

Flat order:
```
 0  (0,0)     1  (0,1)     2  (8,0)     3  (8,1)
 4  (64,0)    5  (64,1)    6  (72,0)    7  (72,1)
 8  (0,8)     9  (0,9)    10  (8,8)    11  (8,9)
12  (64,8)   13  (64,9)   14  (72,8)   15  (72,9)
```

Sources:
- `trt_p15_runtime_layout_dump_20260407.log` lines 3, 10-12
- `trt_p15_copy_view_dump_20260407.log` lines 3-5, 104-218
- `trt_p15_fragment_contract_dump_20260407.log` lines 10-288

---

## 6. Register Fragment Layouts (tCrA / tCrB / tCrC)

### Atom-Tile Fragments (from fragment_contract probe)

Identical for P13 and P15 at the atom level:

```
tCrA: ((_8,_2,_2),_4,_1):((_1,_8,_16),_32,_0)
      size=128  cosize=128

tCrB: ((_8,_2),(_2,_4),_1):((_1,_8),(_16,_32),_0)
      size=128  cosize=128

tCrC: ((_2,_2),_2,_2):((_1,_2),_4,_8)
      size=16   cosize=16
```

Per-instruction register vectors:
```
rA: ((_1,_2,_2),_4,_1):((_1,_1,_2),_4,_0)   size=16
rB: ((_1,_2),(_2,_4),_1):((_1,_1),(_2,_4),_0)   size=16
rC: ((_2,_2),_2,_2):((_1,_2),_4,_8)   size=16
```

Sources:
- `trt_p15_fragment_contract_dump_20260407.log` lines 1-9
- `trt_p15_copy_to_reg_dump_20260407.log` lines 9-12

### Full-Profile Fragments (P15 only)

This is where P15 diverges. `partition_fragment_C(tiled_mma, (256, 128))`
produces:

```
tCrA: ((_8,_2,_2),_4,_1):((_1,_8,_16),_32,_0)
      size=128  cosize=128
      size<0>=32  size<1>=4  size<2>=1

tCrB: ((_8,_2),(_2,_4),_1):((_1,_8),(_16,_32),_0)
      size=128  cosize=128
      size<0>=16  size<1>=8  size<2>=1

tCrC: ((_2,_2),_4,(_2,_4)):((_1,_2),_4,(_16,_32))
      size=128  cosize=128
      size<0>=4  size<1>=4  size<2>=8
```

**tCrA and tCrB do not change** between atom and full profile. They describe
one k-block's worth of operand data regardless of how many M/N tiles the CTA
covers.

**tCrC changes fundamentally**:
- Atom: `((_2,_2),_2,_2)` = 16 values, covering `128x32` of output
- Full profile: `((_2,_2),_4,(_2,_4))` = 128 values, covering `256x128` of output
- The `_4` in mode 1 (was `_2`): 4 N-tiles of 32 columns each = 128 columns
- The `(_2,_4)` in mode 2 (was `_2`): 2 M-halves x 4 N-groups = 8 slices

Source: `trt_p15_full_profile_fragment_dump_20260407.log`

---

## 7. Copy View Layouts (Smem -> Register Retile)

Source: `trt_p15_copy_to_reg_dump_20260407.log`

### A Operand

```
tCrA_copy_view: ((_32,_2),_2,_1):((_1,_32),_64,_0)
size = 128   cosize = 128
```

Per-instruction retiled view:
```
rA_from_copy_view: ((_4,_2),_2,_1):((_1,_4),_8,_0)
size = 16   cosize = 16
```

### B Operand

```
tCrB_copy_view: ((_32,_1),_4,_1):((_1,_0),_32,_0)
size = 128   cosize = 128
```

Per-instruction retiled view:
```
rB_from_copy_view: ((_4,_1),_4,_1):((_1,_0),_4,_0)
size = 16   cosize = 16
```

### SFA Scale

```
tCrSFA_copy_view: ((_1,(_16,_8)),_2,_1):((_0,(_0,_1)),_8,_0)
size = 256   cosize = 16
```

Per-instruction retiled view:
```
rSFA_from_copy_view: ((_1,(_16,_2)),_2,_1):((_0,(_0,_1)),_2,_0)
size = 64   cosize = 4
```

### SFB Scale

```
tCrSFB_copy_view: ((_1,(_16,_4,_2)),_4,_1):((_0,(_0,_1,_16)),_4,_0)
size = 512   cosize = 32
```

Per-instruction retiled view:
```
rSFB_from_copy_view: ((_1,(_16,_1,_2)),_4,_1):((_0,(_0,_1,_4)),_1,_0)
size = 128   cosize = 8
```

---

## 8. Full-Profile Dense Operand Coordinates

Source: `trt_p15_true_dense_operand_coords_dump_20260407.log`

### A Operand (weights, 256 rows x 64 cols)

```
copy_view_a_dense: (((_8,_2,_2),_2),_2,_1):(((_1@1,_8@0,_32@1),_64@0),_128@0,_0)
size = 128   cosize = 8040
```

Thread 0 flat coordinate map (row, col):

```
 0-7:     row=0,   col=0..7
 8-15:    row=8,   col=0..7
16-23:    row=0,   col=32..39
24-31:    row=8,   col=32..39
32-39:    row=64,  col=0..7
40-47:    row=72,  col=0..7
48-55:    row=64,  col=32..39
56-63:    row=72,  col=32..39
64-71:    row=128, col=0..7
72-79:    row=136, col=0..7
80-87:    row=128, col=32..39
88-95:    row=136, col=32..39
96-103:   row=192, col=0..7
104-111:  row=200, col=0..7
112-119:  row=192, col=32..39
120-127:  row=200, col=32..39
```

**Four row bands**: {0,8}, {64,72}, {128,136}, {192,200}.
Two K groups: {0..7}, {32..39}.

### B Operand (activations, 128 rows x 64 cols)

```
copy_view_b_dense: (((_8,_2,_2),_1),_4,_1):(((_1@1,_32@1,_8@0),_0),_32@0,_0)
size = 128   cosize = 4200
```

Thread 0 flat coordinate map:

```
  0-7:    row=0,   col=0..7
  8-15:   row=0,   col=32..39
 16-23:   row=8,   col=0..7
 24-31:   row=8,   col=32..39
 32-39:   row=32,  col=0..7
 40-47:   row=32,  col=32..39
 48-55:   row=40,  col=0..7
 56-63:   row=40,  col=32..39
 64-71:   row=64,  col=0..7
 72-79:   row=64,  col=32..39
 80-87:   row=72,  col=0..7
 88-95:   row=72,  col=32..39
 96-103:  row=96,  col=0..7
104-111:  row=96,  col=32..39
112-119:  row=104, col=0..7
120-127:  row=104, col=32..39
```

**Eight row bands**: {0}, {8}, {32}, {40}, {64}, {72}, {96}, {104}.
Two K groups: {0..7}, {32..39}.

---

## 9. Full-Profile Dense Store Coordinates (Accumulator -> Output)

Source: `trt_p15_true_dense_store_coords_dump_20260407.log`

### Accumulator C (256 rows x 128 cols)

```
part_c_dense: ((_2,_2),_4,(_2,_4)):((_1@1,_8@0),_64@0,(_8@1,_32@1))
size = 128   cosize = 21306
```

Register storage layout:
```
accum_profile: ((_2,_2),_4,(_2,_4)):((_1,_2),_4,(_16,_32))
size = 128   cosize = 128
```

Thread 0 flat coordinate map (row, col) -- all 128 values:

```
Slice 0 (flat 0-15):
  0  (0,0)      1  (0,1)      2  (8,0)      3  (8,1)
  4  (64,0)     5  (64,1)     6  (72,0)     7  (72,1)
  8  (128,0)    9  (128,1)   10  (136,0)   11  (136,1)
 12  (192,0)   13  (192,1)   14  (200,0)   15  (200,1)

Slice 1 (flat 16-31):
 16  (0,8)     17  (0,9)     18  (8,8)     19  (8,9)
 20  (64,8)    21  (64,9)    22  (72,8)    23  (72,9)
 24  (128,8)   25  (128,9)   26  (136,8)   27  (136,9)
 28  (192,8)   29  (192,9)   30  (200,8)   31  (200,9)

Slice 2 (flat 32-47):
 32  (0,32)    33  (0,33)    34  (8,32)    35  (8,33)
 36  (64,32)   37  (64,33)   38  (72,32)   39  (72,33)
 40  (128,32)  41  (128,33)  42  (136,32)  43  (136,33)
 44  (192,32)  45  (192,33)  46  (200,32)  47  (200,33)

Slice 3 (flat 48-63):
 48  (0,40)    49  (0,41)    50  (8,40)    51  (8,41)
 52  (64,40)   53  (64,41)   54  (72,40)   55  (72,41)
 56  (128,40)  57  (128,41)  58  (136,40)  59  (136,41)
 60  (192,40)  61  (192,41)  62  (200,40)  63  (200,41)

Slice 4 (flat 64-79):
 64  (0,64)    65  (0,65)    66  (8,64)    67  (8,65)
 68  (64,64)   69  (64,65)   70  (72,64)   71  (72,65)
 72  (128,64)  73  (128,65)  74  (136,64)  75  (136,65)
 76  (192,64)  77  (192,65)  78  (200,64)  79  (200,65)

Slice 5 (flat 80-95):
 80  (0,72)    81  (0,73)    82  (8,72)    83  (8,73)
 84  (64,72)   85  (64,73)   86  (72,72)   87  (72,73)
 88  (128,72)  89  (128,73)  90  (136,72)  91  (136,73)
 92  (192,72)  93  (192,73)  94  (200,72)  95  (200,73)

Slice 6 (flat 96-111):
 96  (0,96)    97  (0,97)    98  (8,96)    99  (8,97)
100  (64,96)  101  (64,97)  102  (72,96)  103  (72,97)
104  (128,96) 105  (128,97) 106  (136,96) 107  (136,97)
108  (192,96) 109  (192,97) 110  (200,96) 111  (200,97)

Slice 7 (flat 112-127):
112  (0,104)  113  (0,105)  114  (8,104)  115  (8,105)
116  (64,104) 117  (64,105) 118  (72,104) 119  (72,105)
120  (128,104) 121 (128,105) 122 (136,104) 123 (136,105)
124  (192,104) 125 (192,105) 126 (200,104) 127 (200,105)
```

### Structure of the 128-Value Accumulator

**8 slices** of 16 values each. Each slice covers a different N-column pair.

Within each 16-value slice, the pattern is:

```
 0  (row+0,   col+0)
 1  (row+0,   col+1)
 2  (row+8,   col+0)
 3  (row+8,   col+1)
 4  (row+64,  col+0)
 5  (row+64,  col+1)
 6  (row+72,  col+0)
 7  (row+72,  col+1)
 8  (row+128, col+0)
 9  (row+128, col+1)
10  (row+136, col+0)
11  (row+136, col+1)
12  (row+192, col+0)
13  (row+192, col+1)
14  (row+200, col+0)
15  (row+200, col+1)
```

**Row bands per slice**: {0, 8, 64, 72, 128, 136, 192, 200} = 8 rows.
**Column pairs per slice**: {col, col+1} = 2 columns.
**Slice column starts**: {0, 8, 32, 40, 64, 72, 96, 104}.

For comparison, P13's atom-level accumulator (16 values) only covers:
```
{0, 8, 64, 72} x {col, col+1, col+8, col+9}
```

---

## 10. Atom-Level Copy View Coordinates

Source: `trt_p15_copy_view_dump_20260407.log`

### copy_view_a (64 elements per atom)

```
layout: (((_8,_2,_2),_2),_1,_1):(((_1@1,_8@0,_32@1),_64@0),_0,_0)
```

Row structure (two MMA_M subtiles):
```
MMA_M=0: rows {0, 8}   x K {0..7, 32..39}   = 32 elements
MMA_M=1: rows {64, 72} x K {0..7, 32..39}   = 32 elements
```

### copy_view_b (32 elements per atom)

```
layout: (((_8,_2,_2),_1),_1,_1):(((_1@1,_32@1,_8@0),_0),_0,_0)
```

Row structure (two MMA_N subtiles):
```
MMA_N=0: row {0} x K {0..7, 32..39}   = 16 elements
MMA_N=1: row {8} x K {0..7, 32..39}   = 16 elements
```

---

## 11. tCcA / tCcB / tCcC Coordinate Layouts

Source: `trt_p15_fragment_contract_dump_20260407.log`

These are the **smem-space** (coordinate) views that map logical fragment
indices to tile (row, col) positions.

### tCcA

```
layout: ((_8,_2,_2),_4,_1):((_1@0@1,_1@1@0,_32@0@1),_8@1@0,_0)
size = 128   cosize = 1040
```

Covers rows at M-offsets: 0, 1, ..., 7 (packed per half-warp) across 4 K-slices
at K-offsets: 0, 8, 16, 24.

### tCcB

```
layout: ((_8,_2),(_2,_4),_1):((_1@0@1,_32@0@1),(_1@1@0,_4@1@0),_0)
size = 128   cosize = 560
```

Covers N-column offsets: {0,1}, {4,5}, {8,9}, {12,13} across 8 groups of 16 elements.

### tCcC

```
layout: ((_2,_2),_2,_2):((_1@1,_8@0),_64@0,_8@1)
size = 16   cosize = 730
```

(Same as P13. This is the atom-level view, not the full profile.)

---

## 12. Probe Artifact Index

| Probe | Artifact | Key Facts Recovered |
|---|---|---|
| Runtime layout | `trt_p15_runtime_layout_dump_20260407.log` | Stages, TiledMma size, tile_mnk, partA/B/C sizes, SF smem/reg sizes, SF copy view layouts, all partA/B/C coords |
| Copy views | `trt_p15_copy_view_dump_20260407.log` | copy_view_a/b layouts and per-element coords, partA/B/C coords |
| Accum order | `trt_p15_accum_order_dump_20260407.log` | tCcC/tCrC layouts, 16-element flat destination order |
| Scale-as-C | `trt_p15_scale_as_c_dump_20260407.log` | ScaleGranularity, ScaleMsPerTile, smem/reg scale view layouts, all scale-as-C coords |
| Fragment contract | `trt_p15_fragment_contract_dump_20260407.log` | tCrA/tCrB/tCrC reg layouts, rA/rB/rC per-instruction, tCcA/tCcB/tCcC smem coord layouts, all coords |
| Copy-to-reg | `trt_p15_copy_to_reg_dump_20260407.log` | Copy view and retiled-to-register layouts for A/B/SFA/SFB |
| Dense operand coords | `trt_p15_true_dense_operand_coords_dump_20260407.log` | Full-profile copy_view_a_dense (128 elem), copy_view_b_dense (128 elem), all coords |
| Dense store coords | `trt_p15_true_dense_store_coords_dump_20260407.log` | Full-profile part_c_dense (128 elem), accum_profile layout, all 128 output coords |
| Full-profile fragments | `trt_p15_full_profile_fragment_dump_20260407.log` | tCrA/tCrB/tCrC with full (256,128) tile, size decomposition |
| Staged-smem source coords | `trt_p15_smem_partition_dump_latest.log` | Full-thread `tCsA/tCsB` stage-0 source coords after swizzle partitioning |
| P13 runtime layout | `trt_p13_runtime_layout_dump_20260407.log` | P13 comparison data (stages, SF sizes, layouts) |
| P13 accum order | `trt_p13_accum_order_dump_20260407.log` | P13 tCcC/tCrC and accum_profile |

Probe sources: `artifacts/tmp/trt_p15_*.cu`

---

## 13. Staged-Smem Source Contract

Source: `trt_p15_smem_partition_dump_latest.log`

This is the exact `partition_S(as_position_independent_swizzle_tensor(sA/sB))`
stage-0 source contract for traced `P15`.

### Probe Meta

```
threads = 256
tcsa_stage0_size = 128
tcsb_stage0_size = 128
```

### Exact pattern

For thread `0`, the first `TCSA` entries are:

```
((0,0),(0,0),(0,0))
((0,0),(1,0),(0,0))
...
((0,24),(31,0),(0,0))
```

For thread `0`, the first `TCSB` entries are:

```
((0,0),(0,0),(0,0))
((0,0),(1,0),(0,0))
...
((0,12),(31,0),(0,0))
```

### Compression result

The all-thread stage-0 source contract is highly regular:

- `TCSA`:
  - `warp_mod4` shifts only the second leaf's second component by `+2`
  - `upper_half` contributes no change
- `TCSB`:
  - `upper_half` shifts only the second leaf's second component by `+2`
  - `warp_mod4` contributes no change

This is now emitted as compact generated helpers in:

- [generate_p15_smem_partition_tables.py](/home/khkramer/src/nemotron-inference/proj-2026-04-05-1704/generate_p15_smem_partition_tables.py)
- [p15_smem_partition_generated.h](/home/khkramer/src/nemotron-inference/runtime/include/nemotron/p15_smem_partition_generated.h)
