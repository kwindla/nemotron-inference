# P15 (256x128x64 swap_ab=true) Debug Summary & Approach Notes

## What P15 Is

P15 is the tactic profile for FC2 (routed-down gemm2) with tile shape `256x128x64`,
`swap_ab=true`, `cluster=1x1x1`. TRT-LLM selects it for short-input FC2 at
`num_rows=2..7` and `num_rows=16`. It has the largest M-tile (256) of any traced
profile, which means its accumulator, operand, and scale-factor contracts are
structurally different from all the working profiles (P5, P12, P13).

## Current State

**P15 is on the BF16 grouped fallback.** The dormant exact FP4 kernel exists in
`fused_moe_prefill.cu` but is permanently disabled (`#if ... && 0`).

Working profiles and what they share:
- **P5** (`128x128x64 swap_ab=true`): first exact FP4 path landed; scale-smem bridge works
- **P13** (`128x128x64 swap_ab=true`): reused P5's bridge pattern directly; works
- **P12** (`128x128x128 swap_ab=true`): reused P5/P13 bridge as two k64 subtiles; works
- **P15** (`256x128x64 swap_ab=true`): **cannot reuse any of the above patterns**

## Timeline of P15 Attempts (All Failed)

### Attempt 1: Reuse P13 exact-scale-smem kernel shape
- Assumed P15 could share P13's fragment family
- **Failed immediately**: `max_abs_diff = 15.4386`
- Restored to BF16 fallback

### Attempt 2: Rewire to exact traced TiledMma + SmemLayoutSFA/SFB
- Used builder-probe-recovered exact scale layouts
- Still failed with same routed diff
- **Root cause**: not just scale layout; full operand/store fragment contract differs

### Attempt 3: Dedicated 2x2 A/B/C fragment family + traced operand order
- Switched to `A=weights, B=activations` per traced order
- Used two 128-row subtiles for 256 output tile
- **Failed**: `max_abs_diff = 15.5936`, committed-head argmax mismatch
- Restored to BF16 fallback

### Attempt 4: Scale-copy flattening (rank-4 copy-view tensor handling)
- Updated `FillPhysicalCoordMapCopyViewLimited` to flatten rank-4 tensors
- **Failed**: identical error pattern, rank-4 flattening alone not the issue

### Attempt 5: Traced accumulator/store order
- Recovered exact `tCcC/tCrC` flat accumulator destination order from probe
- Rewired `StoreTracedP15CFragmentsRowMajor` to match
- **Failed**: same routed diff

### Attempt 6: CUTLASS-style C-view rescale fold
- Used probed `ScaleMsPerTile=2, ScaleNsPerTile=1` contract
- Applied neutral instruction scales + rescale in C-view
- **Failed**: `max_abs_diff = 15.5936`

### Attempt 7: Fragment-contract replacement
- Replaced hand-rolled fragment family with CUTE-aligned thread-fragment path
- Tried using traced `tCrA/tCrB/tCrC` and `rA/rB/rC` contracts
- Discovered traced P15 is NOT naturally modeled as the manual fragment family

### Attempt 8: Full-profile accumulator (256x128 partition_fragment_C)
- Discovered P15 needs `partition_fragment_C(tiled_mma, (256,128))` = 8 slices of 16 values
- Old `CFragment64[2][2]` model too small
- **Compile failure**: `cute::gemm(...)` static assertions fail
  - `size<1>(A) == size<1>(C)` failed
  - `size<1>(B) == size<2>(C)` failed

### Attempt 9: Manual MMA path with traced scale-smem loaders
- Disabled local-CUTE `cute::gemm(...)` branch
- Used manual MMA with traced scale-smem
- **First time P15 compiled and built cleanly**
- **Runtime failure**: `max_abs_diff = 15.5936`
- **Memory safety failure**: `compute-sanitizer` found OOB global read (1 byte past 256-byte allocation)

### Attempt 10: Fix 128-row B staging (was only staging 32-row atom)
- Corrected to `b_packed[128]` with `kProfileTokenRows=128`
- Removes one structural bug but does not activate P15
- **Next compile attempt**: fails because local bridge tensor type is plain
  `subbyte_iterator<uint4_t>`, not the richer zipped tensor contract CUTLASS expects

## Why P15 Is Different

### Fragment Sizes (from builder probes)

| Property | P5/P13 (128x128x64) | P15 (256x128x64) |
|---|---|---|
| Stages | 9 | 6 |
| SmemLayoutSFA cosize | 4608 | 6144 (+33%) |
| SmemLayoutSFB cosize | 4608 | 3072 (-33%) |
| tCrSFA size | 128 | 256 (doubled) |
| tCrSFB size | 512 | 512 (same) |
| tCrSFA cosize | 8 | 16 (doubled) |
| Accumulator profile | `((_2,_2),_2,_2)` 16 values | `((_2,_2),_4,(_2,_4))` 128 values |
| Accum slices | 1 atom tile | 8 atom-sized slices |
| A copy rows | 0/8 | 0/64/128/192 (4 row bands) |
| B copy rows | 0/8 | 0/32/64/96 (4 row bands) |

### Key structural differences
1. **Accumulator is 8x larger per thread**: 128 values vs 16
2. **A operand spans 4 row bands** (0/64/128/192), not 2
3. **B operand spans 4 row bands** (0/32/64/96), not 2
4. **Scale factor A is doubled** in register capacity
5. **Fewer pipeline stages** (6 vs 9) despite larger tile

### Exact Traced Layouts

**tCrA** (register fragment A):
```
Layout: ((_8,_2,_2),_4,_1):((_1,_8,_16),_32,_0)
size=128, cosize=128
```

**tCrB** (register fragment B):
```
Layout: ((_8,_2),(_2,_4),_1):((_1,_8),(_16,_32),_0)
size=128, cosize=128
```

**tCrC** (accumulator, full profile):
```
Layout: ((_2,_2),_4,(_2,_4)):((_1,_2),_4,(_16,_32))
size=128, cosize=128
size<0>=4, size<1>=4, size<2>=8
```

**Copy views (smem -> register retile)**:
```
tCrA_copy_view: ((_32,_2),_2,_1):((_1,_32),_64,_0), size=128
tCrB_copy_view: ((_32,_1),_4,_1):((_1,_0),_32,_0), size=128
tCrSFA_copy_view: ((_1,(_16,_8)),_2,_1):((_0,(_0,_1)),_8,_0), size=256
tCrSFB_copy_view: ((_1,(_16,_4,_2)),_4,_1):((_0,(_0,_1,_16)),_4,_0), size=512
```

**Scale-as-C view** (for rescaling accumulator):
```
ScaleMsPerTile = 2, ScaleNsPerTile = 1
tCsScaleAViewAsC: ((_2,_2),(_2,_2),_8,_6):((_0,_0),(_0,_1),_0,_2)
tCsScaleBViewAsC: ((_2,_2),_4,_8,_6):((_0,_0),_0,_0,_1)
tCrScaleAViewAsC: ((_2,_2),(_2,_2),_8):((_0,_0),(_0,_1),_0)
tCrScaleBViewAsC: ((_2,_2),_4,_8):((_0,_0),_0,_0)
```

**Flat accumulator destination order** (per thread):
```
 0..3  -> (  0,0) (  0,1) (  8,0) (  8,1)
 4..7  -> ( 64,0) ( 64,1) ( 72,0) ( 72,1)
 8..11 -> (  0,8) (  0,9) (  8,8) (  8,9)
12..15 -> ( 64,8) ( 64,9) ( 72,8) ( 72,9)
```

## Probe Artifacts Reference

All live in `artifacts/benchmarks/` or `artifacts/tmp/`:

| Probe | File | What it recovers |
|---|---|---|
| Runtime layout | `trt_p15_runtime_layout_dump_20260407.log` | Stages, TiledMma size, tile_mnk, partA/B/C sizes |
| Copy views | `trt_p15_copy_view_dump_20260407.log` | copy_view_a/b element coords |
| Accum order | `trt_p15_accum_order_dump_20260407.log` | tCcC/tCrC layouts, flat destination order |
| Scale-as-C | `trt_p15_scale_as_c_dump_20260407.log` | ScaleGranularity, ScaleMsPerTile, scale view layouts |
| Fragment contract | `trt_p15_fragment_contract_dump_20260407.log` | tCrA/tCrB/tCrC/rA/rB/rC |
| Copy-to-reg | `trt_p15_copy_to_reg_dump_20260407.log` | Copy view to register retile layouts |
| Dense operand coords | `trt_p15_true_dense_operand_coords_dump_20260407.log` | Full-profile A/B row spans |
| Dense store coords | `trt_p15_true_dense_store_coords_dump_20260407.log` | Full-profile C output mapping |
| Full-profile fragments | `trt_p15_full_profile_fragment_dump_20260407.log` | tCrA/tCrB/tCrC with full (256,128) tile |

Probe sources in `artifacts/tmp/trt_p15_*.cu`.

## How TRT-LLM Actually Does This

### The CUTLASS SM120 Collective Mainloop

Source: `cutlass/gemm/collective/sm120_mma_array_tma_blockwise_scaling.hpp`

Key patterns the native P15 must replicate:

**1. Fragment creation from smem (not hand-rolled)**
```cpp
TiledMma tiled_mma;
auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
Tensor tCrA = thread_mma.partition_fragment_A(sA(_,_,Int<0>{}));   // (MMA,MMA_M,MMA_K)
Tensor tCrB = thread_mma.partition_fragment_B(sB(_,_,Int<0>{}));   // (MMA,MMA_N,MMA_K)
```

**2. smem-to-register retiling via copy atoms (not manual coord maps)**
```cpp
auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(thread_idx);
Tensor tCsA = smem_thr_copy_A.partition_S(sA);                    // source in smem
Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);           // dest = register fragment
copy(smem_tiled_copy_A, tCsA_stage(_,_,k_block), tCrA_copy_view(_,_,k_block));
```

**3. FP4 shift before MMA**
```cpp
fp4_shift_A(MMAOp{}, tCrA_copy_view(_,_,k_block));
fp4_shift_B(MMAOp{}, tCrB_copy_view(_,_,k_block));
```

**4. Scale loading via C-partitioned views**
```cpp
Tensor tCsScaleAViewAsC = thread_mma.partition_C(sScaleAViewAsC);
copy(tCsScaleAViewAsC(_, _, _, read_stage), tCrScaleAViewAsC);
// Pre-multiply: scaleA[i] *= scaleB (when ScaleMsPerTile > 1, ScaleNsPerTile == 1)
```

**5. Main GEMM into tmp_accum, then rescale into real accum**
```cpp
cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tmp_accum);
// ... after all k_blocks in scale group:
for (int i = 0; i < size(accum); ++i) {
    accum(i) += tmp_accum(i) * tCrScaleAViewAsC(i);
    tmp_accum(i) = 0;
}
```

**6. Pipeline: producer fills smem via TMA, consumer reads via copy atoms**
- AB stages (e.g. 6 for P15, 9 for P13)
- SF stages
- NamedBarrier sync between last k_block and next tile
- Phase flips for double-buffering

### TRT-LLM's MoE-Specific Kernel (sm120_fp8_moe_gemm_1d1d.cuh)

This is the actual kernel TRT-LLM runs for MoE. Different from the CUTLASS
collective because it zips scales into operands:

```cpp
cute::gemm(mma,
    make_zip_tensor(tCrA, tCrSFA_stage),     // A + scale_A zipped
    make_zip_tensor(tCrB, tCrSFB_stage),     // B + scale_B zipped
    accum);
```

Scale fragments are created via custom helpers:
```cpp
auto tCrSFA = KT::partition_fragment_SFA(sSFA(_, _, Int<0>{}), thr_mma);
auto tCrSFA_frg = KT::transform_fragment_for_qmma(tCrSFA);
// transform_fragment_for_qmma recasts int32 -> float_ue8m0_t with layout (_32, num_mn, _4, _4)
```

Epilogue: convert float accum -> BF16, copy to smem, then smem -> gmem with
boundary predication.

## Tracing & Instrumentation Guide

### Builder Probes (Offline, No GPU Needed for Type Recovery)

The most productive debugging technique so far. Build a standalone `.cu` that
instantiates the exact CUTLASS `CollectiveMainloop` type for P15's parameters
and prints the CUTE layout strings.

Template:
```cpp
#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/builders/sm120_common.inl>
// ... build the CollectiveMainloop type with P15 parameters ...
// Print: size/cosize/layout of tCrA, tCrB, tCrC, copy views, scale views
```

Existing probes to copy from: `artifacts/tmp/trt_p15_*.cu`

### Key Diagnostic Prints

When activating the dormant P15 path, add these at the kernel entry:
```cpp
if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("P15 active: output_tile=%d k_tile=%d\n", kOutputTile, kMacroTileK);
    printf("P15 a_packed ptr=%p b_packed ptr=%p\n", a_packed, b_packed);
    printf("P15 sfa ptr=%p sfb ptr=%p\n", sfa_smem, sfb_smem);
}
```

### compute-sanitizer (Memory Safety)

Essential before any correctness debugging:
```bash
compute-sanitizer --tool memcheck ./testing/fused_moe_prefill_test
```
The last P15 activation found an OOB read (1 byte past 256-byte allocation) in
thread (15,0,0), block (0,1,0). Fix memory safety before chasing numeric diffs.

### Correctness Gates

Always run in this order:
1. `compute-sanitizer --tool memcheck ./testing/fused_moe_prefill_test`
2. `./testing/fused_moe_prefill_test`
3. `./testing/multi_turn_prefix_reuse_test`

If any gate fails, restore P15 to BF16 fallback before continuing.

### Error Signature Meaning

The consistent `max_abs_diff = 15.5936` (or 15.4386) with `actual=0 expected=15.x`
means P15 is producing zeros (or near-zeros) where it should produce real values.
This is not a small numerical drift -- it's a structural contract mismatch where
the accumulator/store path is writing to wrong locations or the operand load is
reading wrong data.

### nsys Profiling

```bash
nsys profile --stats=true -o profile_p15 \
    ./benchmarks/nano_moe_prefill/nano_prefix_cache_ttft_bench \
    --prefix-length 4 --tail-token-count 4
```

### ncu Kernel Profiling

```bash
ncu --target-processes all \
    --kernel-name "Nvfp4LaunchPlannedPackedInputGroupedFp4KernelSwapTrueK64ScaleSmemP15" \
    --launch-count 1 \
    --set full \
    -o ncu_p15 \
    ./testing/fused_moe_prefill_test
```

## Diagnosis: Why Every Attempt Failed

The core issue is that **P15 requires the full CUTLASS retile/copy/gemm pipeline
and cannot be faithfully reproduced with hand-rolled fragment families.**

Every failed attempt shared the same fundamental mistake: trying to manually
reconstruct the CUTE fragment contract piece by piece, fixing one layout at a
time, when the contract is an interconnected system where:

1. `partition_fragment_A/B` creates register layouts
2. `make_tiled_copy_A/B` + `retile_D` creates the smem-to-register copy views
3. `fp4_shift_A/B` adjusts subbyte encoding
4. `cute::gemm(tiled_mma, tCrA, tCrB, tmp_accum)` consumes exactly those fragment shapes
5. Scale views are partitioned via `partition_C` (C-space, not A/B-space)
6. Rescaling operates element-wise on the same accumulator tensor

When any one of these is wrong, the error propagates through all downstream
stages. The probes recovered the exact layouts, but the manual kernel code
cannot enforce the implicit consistency contracts that CUTE enforces
automatically.

## Suggested New Approach

### Option A: Use CUTLASS Directly (Recommended)

Stop trying to hand-roll P15. Instead:

1. **Build a thin CUTLASS kernel wrapper** that instantiates the real
   `CollectiveMainloop` for P15's tile shape and calls its `mma()` method
2. Use the existing local bridge infrastructure for the parts that work
   (launch plan, activation packing, result scatter)
3. The wrapper only needs to:
   - Set up smem tensors with the correct layouts
   - Call the collective's `load()` (producer) and `mma()` (consumer) methods
   - Handle the epilogue store
4. This bypasses ALL the fragment/retile/copy mismatches because CUTLASS
   handles them internally

**Why this hasn't worked before**: the full CUTLASS header stack (copy_atom.hpp,
prefetch.hpp, cuda_host_adapter.hpp) doesn't drop in cleanly. But the collective
mainloop header DOES compile with C++20. The problem was the *broader* header
dependencies, not the core mainloop.

**Concrete next step**: isolate exactly which headers are needed for just the
`mma()` consumer path (not the TMA producer). The consumer only needs:
- `sm120_mma_array_tma_blockwise_scaling.hpp` (for `mma()`)
- MMA traits for SM120
- `cute/algorithm/gemm.hpp`
- `cute/tensor.hpp`

The TMA producer can remain in the existing bridge code since it's just
global-to-smem copies.

### Option B: Generate Fragment Code From Probes

If CUTLASS headers remain too heavy:

1. Write a code generator that takes the probe output (exact tCrA/tCrB/tCrC
   layouts, copy view layouts, scale view layouts) and emits correct C++ for:
   - Fragment declarations with exact CUTE layout types
   - Load functions using the exact copy-view-to-register retile
   - Store functions using the exact accumulator-to-output retile
2. The generated code would be verified by a standalone test that compares its
   output against CUTLASS's own `cute::gemm()` on random data
3. Once verified, plug the generated load/gemm/store into the dormant P15 kernel

This avoids importing CUTLASS headers but produces code that is provably
equivalent to what CUTLASS would generate.

### Option C: Profile-Guided Tactic Substitution

Since P15 only covers `num_rows=2..7` and `num_rows=16`:

1. Measure whether the BF16 fallback at these row counts is actually a
   meaningful performance bottleneck vs the working FP4 profiles at higher row
   counts
2. If the BF16 fallback is "good enough" for these small-M cases, defer P15
   and focus optimization effort on the profiles that handle the bulk of
   prefix128/prefix4096 work
3. If P15 does matter, consider whether the dispatch table can be adjusted so
   these row counts land on P13 (which works) with appropriate padding

## Key TRT-LLM Source References

| File | Path | What to read |
|---|---|---|
| SM120 utils/builder | `third_party/TensorRT-LLM/cpp/.../sm120_blockwise_gemm/sm120_utils.cuh` | MMA atom, smem layouts, scale partition helpers, transform_fragment_for_qmma |
| SM120 MoE GEMM | `third_party/TensorRT-LLM/cpp/.../sm120_blockwise_gemm/sm120_fp8_moe_gemm_1d1d.cuh` | make_zip_tensor pattern, scheduler, epilogue |
| CUTLASS collective mainloop | `artifacts/trtllm_source_build7/cpp-build/_deps/cutlass-src/include/cutlass/gemm/collective/sm120_mma_array_tma_blockwise_scaling.hpp` | The canonical `mma()` consumer: copy_kblock, gemm_kblock, rescale lambdas |
| FP4 GEMM template | `third_party/TensorRT-LLM/cpp/.../fp4_gemm/nvfp4_nvfp4_gemm_template_sm120.h` | CollectiveBuilder usage for FP4 |
| Fused MoE kernel | `third_party/TensorRT-LLM/cpp/.../cutlass_extensions/.../fused_moe_kernel.cuh` | Problem visitor, residue handling |
| CUTE DSL reference | `third_party/TensorRT-LLM/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/blockscaled_contiguous_grouped_gemm.py` | Python-level reference for the same pattern |

## The Real Bottleneck Hierarchy

Before spending more time on P15, consider the overall performance picture:

| Metric | Native | TRT-LLM | Gap |
|---|---|---|---|
| prefix4 cold | 54.016 ms | 35.922 ms | 1.50x |
| prefix128 cold | 107.215 ms (best) | 38.407 ms | 2.79x |
| prefix4096 cold | ~1633 ms (best) | 66.789 ms | 24.5x |

P15 handles `num_rows=2..7` and `num_rows=16`. At prefix128, the average tokens
per expert is 6, so P15 IS relevant for that design-center case. But the main
gap is the grouped math core itself (scalar WMMA vs TRT's block-scaled FP4 TMA
mainloop), not just P15's fragment contract.

**Priority recommendation**: Fix P15 via Option A or Option C, then focus the
bulk of effort on the mainloop alignment (TMA transport, warp specialization,
multi-stage pipelining) that would close the 2.79x gap at prefix128.
