# P15 Divergences from TRT-LLM Reference

Line-by-line comparison of our P15 kernel (commit `57b4ef4`) against the
probed TRT-LLM MoE kernel (`sm120_fp8_moe_gemm_1d1d.cuh` lines 530-700)
and probed layout/fragment data in `P15_PROBE_FACTS.md`.

All line numbers reference `runtime/src/backend/fused_moe_prefill.cu` at
commit `57b4ef4` unless otherwise noted.

---

## Divergence 1: `fp4_shift` May Be Incorrect for Zipped-Scale Path

**Lines 5420-5421:**
```cpp
cute::fp4_shift_A(P15MMAOp{}, tCrA_cv(cute::_, cute::_, k));
cute::fp4_shift_B(P15MMAOp{}, tCrB_cv(cute::_, cute::_, k));
```

**Issue:** TRT-LLM's MoE kernel (`sm120_fp8_moe_gemm_1d1d.cuh` lines
604-611) does NOT call `fp4_shift_A/B` before its zipped `cute::gemm`.
The CUTLASS collective mainloop (`sm120_mma_array_tma_blockwise_scaling.hpp`
line 714) DOES call it, but that code uses the separate C-view rescale
pattern, not the zipped-scale pattern.

The `fp4_shift_A` function (defined in `mma_traits_sm120.hpp` lines
243-248) left-shifts the uint32 register by 2 bits when the element type
is `float_e2m1_t`.  This adjusts the FP4 nibble encoding for the MMA
instruction.

**The question:** Does the SM120 BLOCKSCALED MMA instruction expect
pre-shifted FP4 data when scales are zipped into the operands?  If yes,
the shift is needed.  If the instruction handles the encoding internally
when block scales are provided, the shift is double-applying.

**Evidence from our tests:** The standalone `p15_swizzled_pipeline_test.cu`
with FP4(1.0) inputs and identity scales produced exactly 64.0 per element
(= 1.0 * 1.0 * 64 K-elements).  This is the correct result WITHOUT any
shift-induced factor, which suggests either (a) the shift is correct and
the instruction expects it, or (b) the shift and instruction cancel out
for value 1.0 specifically.  Need a test with a value where the shift would
visibly change the result (e.g., FP4 nibble 0x3 = 1.5).

**Action:** Add a test case to `p15_swizzled_pipeline_test.cu` that uses
FP4(1.5) with and without `fp4_shift` and compares against the
mathematically expected result.  This definitively answers whether the
shift is needed.

---

## Divergence 2: Identity Scales Instead of Real Per-Row Scales

**Lines 5442-5443:**
```cpp
auto sfa_reg = std::uint32_t{0x38383838u};
auto sfb_reg = std::uint32_t{0x38383838u};
```

**Issue:** Hardcoded identity ue4m3 scales (1.0).  Real scale data is
loaded into `a_scale_smem`/`b_scale_smem` via `StoreTracedP5ScaleWordK64`
(lines 5309, 5323), but the MMA consumer ignores it.

**Probed reference data that should be used:**
- `SmemLayoutSFA cosize = 6144`, `SmemLayoutSFB cosize = 3072`
  (P15_PROBE_FACTS.md section 2)
- `tCrSFA size = 256, cosize = 16` per thread (section 3)
- `ScaleMsPerTile = 2` -- two A-scale groups for the 256-row M-tile
  (section 4)
- `ScaleNsPerTile = 1` -- one B-scale group for the 128-row N-tile
  (section 4)
- The atom-level `rSFA` has `size=64, cosize=4`
  (from `trt_p15_fragment_contract_dump_20260407.log` line 13-14)

**Action:** For each atom at tile position `(m, n)`, load the uint32
scale word for the M-row that atom handles (from `a_scale_smem` via the
swizzled scale tensor) and the N-row (from `b_scale_smem`).  The
`partition_C(identity_tensor)` approach gives the atom's base row.  The
scale smem is indexed through `TracedP15SmemLayoutSFA` which has the
probed layout.

---

## Divergence 3: `SFAtomLayout` Not Validated Against Probed `rSFA`

**Lines 5425-5427:**
```cpp
using SFAtomLayout = cute::Layout<
    cute::Shape<cute::Shape<cute::_16, cute::_4>>,
    cute::Stride<cute::Stride<cute::_0, cute::_1>>>;
```

**Issue:** This layout was derived from first principles (64 logical
elements, 4 physical bytes, broadcast groups of 16).  But the probed
`rSFA` fragment from `trt_p15_fragment_contract_dump_20260407.log` line 13
has:
```
rSFA.layout: ((_16,_1),(_2,_2),_1):((_0,_1),(_1,_2),_0)
rSFA.size=_64 cosize=_4
```

The probed layout is rank-3 `((_16,_1), (_2,_2), _1)` while our
`SFAtomLayout` is rank-1 `((_16,_4))`.  Both have `size=64, cosize=4`.
After `filter_zeros` and `recast<uint32_t>`, both should yield 1 register.
But the intermediate `recast<RegTypeSFA>` step operates on the layout
before `filter_zeros`, and the byte ordering within the 4-byte scale word
depends on the stride pattern.

The probed layout's inner strides `((_0,_1),(_1,_2),_0)` mean:
- First 16 elements: stride 0 (broadcast from byte 0)
- Then mode `(_2,_2)` with strides `(_1,_2)`: bytes at offsets 0,1,2,3

Our layout strides `((_0,_1))` mean:
- Groups of 16: stride 0 (broadcast)
- 4 groups: stride 1 (sequential bytes)

These encode the same 4 bytes in the same order: bytes 0,1,2,3.  So the
layouts are functionally equivalent despite different rank structure.  But
this equivalence should be stated explicitly in the code as a verified
fact, referencing the probe artifact.

**Action:** Add a comment at line 5425 referencing
`trt_p15_fragment_contract_dump_20260407.log` and stating the equivalence.
Optionally add a `static_assert` that
`size(SFAtomLayout{}) == 64 && cosize(SFAtomLayout{}) == 4`.

---

## Divergence 4: Double-Staging (Flat -> Swizzled)

**Lines 5218-5237 (allocation) and 5340-5365 (scatter):**
```cpp
__shared__ std::uint8_t a_packed[kOutputTile][64 / 2];           // 8 KB
__shared__ alignas(1024) cute::array_aligned<...> smem_swizzled_a_storage;  // ~8 KB
// ...
sw_a[static_cast<int>(elem_offset) / 2] = a_packed[row][col_byte];
```

**Issue:** Data is loaded from global memory into flat `a_packed` (line
5302), then scattered from flat to swizzled `smem_swizzled_a` (line 5352).
This uses ~16 KB extra smem for A and ~8 KB for B (total ~24 KB), plus an
extra barrier.

TRT-LLM loads directly from global into swizzled smem via TMA
(`sm120_fp8_moe_gemm_1d1d.cuh` lines 446-474), which writes directly to
`shared_storage.tensors.load.smem_A` in swizzled layout.  No intermediate
buffer.

**Action:** Replace the flat load + scatter with a direct scatter from
global to swizzled smem.  For each thread, compute the swizzled byte
offset for its assigned `(row, col_byte)` pair and write the global byte
directly there.  This eliminates `a_packed`/`b_packed` entirely and saves
~24 KB of smem.

The swizzled offset computation is:
`SmemLayoutA{}(row, col_byte * 2, Int<0>{})` gives the element offset;
divide by 2 for the byte offset.  This is the same computation already at
line 5351, just sourcing from global memory instead of flat smem.

---

## Divergence 5: Scalar Epilogue Store

**Lines 5472-5489:**
```cpp
auto part_c = thread_mma.partition_C(dense_c);
for (int i = 0; i < size(part_c); ++i) {
  output[...] = accum_tensor(i) * output_alpha;
}
```

**Issue:** Per-element scalar store with individual global memory writes.
TRT-LLM (`sm120_fp8_moe_gemm_1d1d.cuh` lines 648-700) uses a
three-stage epilogue:
1. Convert float->BF16 via `NumericArrayConverter<ElementD, ElementAccum, 2>`
   (vectorized pair conversion)
2. Register -> smem via `make_tiled_copy_C` + `SmemCopyAtomR2S`
   (vectorized smem write)
3. Smem -> register via `TiledCopyS2G` (vectorized smem read)
4. Register -> global via `GmemCopyAtomR2G` (vectorized 128-bit global
   write with boundary predication)

The probed P15 store coordinates (P15_PROBE_FACTS.md section 9) confirm
128 accumulator values per thread, stored to 8 distinct row bands with
2-column pairs each.  This access pattern is suitable for vectorized
writes.

**Action:** This is a performance optimization.  The current scalar store
is correct.  The plan should include this as a later step after scales are
working, referencing the TRT-LLM epilogue pattern.

---

## Divergence 6: Manual Atom-Level GEMM Dispatch

**Lines 5433-5451:**
```cpp
for (int k = 0; k < K_blocks; ++k) {
  for (int n = 0; n < N_tiles; ++n) {
    for (int m = 0; m < M_tiles; ++m) {
      mma_atom.call(c_atom, a_zipped, b_zipped, c_atom);
    }
  }
}
```

**Issue:** We iterate `k -> n -> m`.  TRT-LLM's `cute::gemm` (via
`gemm.hpp` dispatch [4], lines 348-385) uses serpentine iteration: for
64-bit A and 32-bit B types, it keeps A in the outer loop and serpentines
B.  For 32-bit A and 64-bit B, the reverse.  This improves register reuse.

For our FP4 types: `sizeof(ValTypeA) * size<0>(A) = 4 bits * 32 = 16 bytes`
and `sizeof(ValTypeB) * size<0>(B) = 4 bits * 16 = 8 bytes`.  Neither
matches the 8+4 byte check exactly.  `cute::gemm` dispatch [4] would fall
through to the generic serpentine at lines 374-385: col-major serpentine
with `ms = (n & 1) ? M-1-m : m`.

**Action:** Change the loop to `n -> m(serpentine)` to match `cute::gemm`'s
behavior.  This is a minor performance optimization, not a correctness
issue.
