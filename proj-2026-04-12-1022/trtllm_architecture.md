# TRT-LLM NVFP4 MoE GEMM — Architecture Reference for `proj-2026-04-12-1022`

All file paths are relative to
`/home/khkramer/src/nemotron-inference/third_party/TensorRT-LLM`
unless otherwise noted. Line numbers are the lines in the files as of the
vendored copy in this repo.

**Scope note:** This document covers only `cpp/tensorrt_llm/kernels/cutlass_kernels/`.
Anything that points outward (to `cutlass_extensions/`, plugins, Python glue, CUTLASS
itself) is noted at the reference point but not deep-dived.

---

## 0. TL;DR summary (read first)

1.  **TRT-LLM has real, compiled SM120 support for NVFP4 MoE GEMM.** It lives in
    `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl`
    and is gated on `cutlass::arch::Sm120` (plus `Sm121`). It is **not** just the SM100
    path in disguise: the arch selection flips the `TensorOp` tag, the per-element wrappers
    (`nv_float4_t` vs. `cute::tuple`), the `KernelSchedule`, and the `EpilogueSchedule`.
2.  **SM120 intentionally hands the mainloop to CUTLASS's
    `KernelScheduleAuto`.** TRT-LLM never names an MMA atom (`SM120_MXF4NVF4_SS_m16n8k64_SB`
    or otherwise) anywhere in the MoE / FP4 GEMM paths; it lets the CUTLASS
    `CollectiveBuilder` pick the atom from `OpClassBlockScaledTensorOp` +
    `cutlass::nv_float4_t<ElementAct>` + `MmaTileShape` + `ClusterShape<1,1,1>`. See
    `moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl:777–778` and
    `fp4_gemm/nvfp4_nvfp4_gemm_template_sm120.h:109–114`. Step 2's bring-up will
    therefore delegate atom selection to CUTLASS rather than assembling atoms by hand.
3.  **TRT-LLM's NVFP4 MoE FC1 does NOT emit packed FP4 bytes from the GEMM kernel
    itself.** The GEMM kernel writes a BF16 (or FP16) intermediate and a separate,
    post-GEMM CUDA kernel (`doActivationKernel`,
    `moe_gemm/moe_kernels.cu:2063`) applies `Relu / Relu² / Gelu / Silu / Swiglu` AND
    performs per-16-element max-abs → FP8 E4M3 block scale → FP4 packing into the next
    GEMM's input contract. The GEMM kernel's own epilogue is just
    `LinearCombination<ElementD, float, void, float>` (per-expert `alpha` only, no `beta`,
    no activation, no quantization).
4.  **Consequence for our Step 2 oracle surface.** A bitwise comparison against TRT-LLM
    at the *packed-FP4-out-of-FC1* level is not possible with stock TRT-LLM, because
    TRT-LLM does not have such a surface. TRT-LLM's comparable bitwise surface is
    the BF16 intermediate `gemm1_output` immediately after `moe_gemm_runner_.moeGemm(...)`
    and before `doActivationKernel`. Our Step 2 harness should expose that BF16
    intermediate as the primary oracle and keep the `doActivationKernel` quantization as a
    separate reference (it can be reused verbatim against our kernel's own dense BF16
    staging if we route our kernel through an unfused path for the oracle run).
5.  **The in-tree TRT-LLM C++ test for NVFP4 MoE GEMM is tolerance-based, not bitwise**,
    uses `Relu` (not `Relu²`), and only checks the final post-FC2 output
    (`cpp/tests/unit_tests/kernels/mixtureOfExpertsTest.cu:1811–1869`,
    `compareFinal` at `:1652–1699`, tolerance `0.05` for NVFP4 at `:214`). It is not
    a bitwise reference; we will need to construct our own.

---

## 1. Top-level entry point and dispatch chain

### 1.1 Two separate entry points, both in `cutlass_kernels/`

TRT-LLM exposes NVFP4 block-scaled GEMM through two distinct entry points:

1.  **Grouped / MoE GEMM** (the one we want for FC1):
    - Class: `kernels::cutlass_kernels::MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>`
      at `cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_gemm_kernels.h:262`.
    - Explicit NVFP4 instantiations at
      `moe_gemm/moe_gemm_kernels_fp4_fp4.cu:24–29` (for `__nv_fp4_e2m1` × `__nv_fp4_e2m1`
      with `OutputType = half` or `__nv_bfloat16`).
    - Public methods: `moeGemm(...)` and `moeGemmBiasAct(...)`, declared at
      `include/moe_gemm_kernels.h:297–301`. For NVFP4, the caller uses
      `moeGemm(...)` (no bias+act fusion; see §6 for why).

2.  **Plain single-batch NVFP4 GEMM** (used for non-MoE FP4 ops):
    - Class: `CutlassFp4GemmRunner<T, FP4GemmType::W4A4_NVFP4_NVFP4>` at
      `include/fp4_gemm.h:72`.
    - SM120 launch macro at
      `fp4_gemm/nvfp4_nvfp4_gemm_template_sm120.h:63–258`.
    - SM100 launch macro at
      `fp4_gemm/nvfp4_nvfp4_gemm_template_sm100.h:133–326`.
    - Arch dispatch in `fp4_gemm/fp4_gemm_template.h:343–384`; SM120 branch at `:378–381`
      calls `dispatchNVFP4xNVFP4GemmCTAShapeSm120<T>`.

For the FC1 kernel work, the **MoE GEMM path** is the authoritative reference. The plain
FP4 GEMM path is still useful as a cross-reference for how a single NVFP4 SM120 GEMM is
configured — it uses `cutlass::arch::Sm120` with a `nv_float4_t<float_e2m1_t>` element
pair type and `OpClassBlockScaledTensorOp`, which is structurally identical to what the
MoE path instantiates.

### 1.2 MoE GEMM dispatch chain (SM120 NVFP4)

From `MoeGemmRunner::moeGemm(...)` all the way down to the kernel instantiation, in order:

1.  `moe_gemm/moe_gemm_template_dispatch.h` — the top-level dispatcher (not fully read
    in this pass; the SM120-relevant branch is handed to the TMA WS dispatcher below).
2.  `moe_gemm/moe_gemm_template_dispatch_tma_ws.h:391`:
    `dispatchMoeGemmSelectTileShapeTmaWarpSpecialized<T, WeightType, OutputType, EpilogueTag, FUSION>(...)`.
    - SM120 branch at `:486–500`. It gates on
      `isValidSM120MOESpecialisation<T, WeightType, EpilogueTag, FUSION>()` and expands
      through `SHAPE_CASE(120, ...)` at `:493–496`.
3.  `moe_gemm/moe_gemm_template_dispatch_tma_ws.h:349`:
    `dispatchMoeGemmSelectClusterShapeTmaWarpSpecialized<cutlass::arch::Sm120, ...>`.
    - SM120 only supports `1×1×1` cluster (see `are_tile_shapes_supported_sm120` at
      `:278–293`).
4.  `moe_gemm/moe_gemm_template_dispatch_tma_ws.h:125`:
    `dispatchMoeGemmFinalDispatchTmaWarpSpecialized<cutlass::arch::Sm120, ...>`.
    - SM120 branch at `:199–213`. Key: `EpilogueSchedule = void` (it is hardcoded in
      the launcher), `dynamic_cga = false`, picks `swap_ab` true/false based on
      `hopper_input.swap_ab`.
5.  `moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl`:
    `tma_warp_specialized_generic_moe_gemm_kernelLauncher<cutlass::arch::Sm120, T, WeightType, OutputType, ...>(...)`.
    - This is the `INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM` macro body
      (`:533–1060`). This is where the CollectiveBuilder calls live.

### 1.3 TMA WS valid tile shapes for SM120 NVFP4

`moe_gemm/moe_gemm_template_dispatch_tma_ws.h:278–293`:
```cpp
template <typename CtaShape, typename ClusterShape, typename DataType>
constexpr bool are_tile_shapes_supported_sm120()
{
    if (cluster != 1x1x1) return false;
    TileM = size<0>(CtaShape);
    TileN = size<1>(CtaShape);
    TileK = size<2>(CtaShape);
    return (TileM==128 && TileN==128 && TileK==128)
        || (TileM==128 && TileN==128 && TileK==256)
        || (TileM==128 && TileN==256 && TileK==128)
        || (TileM==256 && TileN==128 && TileK==128);
}
```

The SHAPE_CASE list at `:493–496`:
```
SHAPE_CASE(120, 128, 128, 64)   // TileK_bytes=64 → TileK_elem=128 FP4
SHAPE_CASE(120, 128, 128, 128)  // TileK_bytes=128 → TileK_elem=256 FP4
SHAPE_CASE(120, 128, 256, 64)
SHAPE_CASE(120, 256, 128, 64)
```

**Important:** The last macro argument `K` (`64`, `128`) is in **bytes**, not elements.
The dispatcher converts to elements at `moe_gemm_template_dispatch_tma_ws.h:400–402`:
```cpp
constexpr int KtileBytes = (K * 8) / cutlass::sizeof_bits<...T...>::value;
using KTileDim = Int<KtileBytes>;
using TileShape = Shape<_M, _N, KTileDim>;
```
For FP4 (4 bits per element), `K=64` bytes → TileK elements = 128.

### 1.4 Tile shapes that actually hit a P1/P5/P15 FC1/FC2 codegen compile probe

`moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl:810–829` contains three hardcoded
`if constexpr` blocks that match the tile/cluster combinations TRT-LLM's internal
"compile probes" target for SM120 NVFP4 MoE:

-   **P5 (FC1)**: SwapAB=true, `CTA_M=128`, `CTA_N=128`, `CTA_K=128`, cluster `1×1×1`
    (see `maybePrintSm120P5CompileProbe<CollectiveMainloop>()` at `:812`).
-   **P1 (FC1)**: SwapAB=false, `CTA_M=128`, `CTA_N=128`, `CTA_K=64`, cluster `1×1×1`
    (see `maybePrintSm120P1CompileProbe<CollectiveMainloop>()` at `:820`).
-   **P15 (FC2, FINALIZE fusion)**: SwapAB=true, `CTA_M=256`, `CTA_N=128`, `CTA_K=64`,
    cluster `1×1×1` (see `maybePrintSm120P15CompileProbe<CollectiveMainloop>()` at `:824`).

These are the exact sets of TRT-LLM's internal compile-probe names (`P1`, `P5`, `P15`)
that show up repeatedly in our own plans. They all go through
`INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM` with `IsSM120 && IsFP4 && !IsMXFPX` and print
layout diagnostics when `TLLM_FUSED_MOE_PRINT_COMPILE_PROBE*` is set. This is load-bearing
for Step 2: when we instantiate TRT-LLM against our FC1 shapes, we can set
`TLLM_FUSED_MOE_PRINT_COMPILE_PROBE_P1=1` (or `..._P5`) to get the `SmemLayoutAtomSFA`,
`SmemLayoutSFA`, `SmemCopyAtomSFA`, `LayoutSFA_TV`, and `tCrC_profile` layouts dumped
directly from the same `CollectiveMainloop` / `TiledMma` that TRT-LLM itself builds.

---

## 2. Architecture gating

### 2.1 There are three distinct arch families in this dispatcher

`moe_gemm/moe_gemm_template_dispatch_tma_ws.h:129–164`:
- SM80 (Ampere): fallback, not TMA WS.
- SM90 (Hopper): `COMPILE_HOPPER_TMA_GROUPED_GEMMS`, requires
  `isValidHopperMOESpecialisation<T, WeightType, EpilogueTag>()`.
  **For FP4, Hopper is explicitly excluded** —
  `moe_gemm/moe_tma_warp_specialized_traits.h:76–78` requires
  `!cutlass::platform::is_same<T, __nv_fp4_e2m1>::value`.
- SM100 / SM103 (Blackwell datacenter): `COMPILE_BLACKWELL_TMA_GROUPED_GEMMS` /
  `COMPILE_BLACKWELL_SM103_TMA_GROUPED_GEMMS`.
- SM120 / SM121 (Blackwell client): `COMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS`
  (gate at `moe_gemm_template_dispatch_tma_ws.h:160–164`). This is the RTX 5090 path.

### 2.2 SM120 is not inherited from SM100 — it branches

`moe_gemm/moe_gemm_template_dispatch_tma_ws.h:183–212`:
```cpp
if constexpr (Arch::kMinComputeCapability >= 100 && Arch::kMinComputeCapability < 120)
{
    // SM100/SM103 path: dynamic CGA, TMA or NoSmem epilogue schedule choice,
    // fixed MMA atoms (KernelPtrArrayTmaWarpSpecialized{1,2}SmNvf4Sm100)
    ...
}
else if constexpr (Arch::kMinComputeCapability >= 120 || ...)
{
    using EpilogueSchedule = void; // hardcoded in the launcher
    constexpr bool dynamic_cga = false;
    // Direct launcher call with dynamic_cga=false
    ...
}
```

Inside the launcher itself
(`moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl`):

- `KernelScheduleSM120 = cutlass::gemm::collective::KernelScheduleAuto` (line 777).
- `KernelScheduleSM100` = explicit SM100 NVFP4 schedule
  (`KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100` or `...2Sm...`) at lines 760–765.
- `EpilogueScheduleSM120 = cutlass::epilogue::TmaWarpSpecialized` (line 699).
- `EpilogueScheduleSM10x` = SM100-specific schedules at lines 694–698.

**Surprising but important:** Under SM120, TRT-LLM wraps element types
differently than under SM100.
`moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl:647–652`:
```cpp
using ElementActBlockScaled = std::conditional_t<IsSM120,
    std::conditional_t<IsMXFPX, cutlass::mx_float8_t<ElementAct>, cutlass::nv_float4_t<ElementAct>>,
    cute::tuple<ElementAct, ElementSF>>;
using ElementWeightBlockScaled = std::conditional_t<IsSM120,
    std::conditional_t<IsMXFPX, cutlass::mx_float4_t<ElementWeight>, cutlass::nv_float4_t<ElementWeight>>,
    cute::tuple<ElementWeight, ElementSF>>;
```

So SM120 passes `cutlass::nv_float4_t<cutlass::float_e2m1_t>` as the A/B element type
to the CollectiveBuilder, while SM100 passes a raw `cute::tuple<ElementA, ElementSF>`.
The `nv_float4_t<...>` wrapper is a block-scaled element pair known to CUTLASS's
SM120-specific atoms. This is the primary compile-time signal that routes inside
`CollectiveBuilder` to a different CollectiveOp than SM100. **We must replicate this
wrapping exactly** if we want to match TRT-LLM's SM120 instantiation.

### 2.3 Explicit SM120 validity check

`moe_gemm/moe_tma_warp_specialized_traits.h:37–47`:
```cpp
template <...>
constexpr bool isValidSM120MOESpecialisation()
{
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    return ((is_same_v<T, __nv_fp4_e2m1> && is_same_v<T, WeightType>)
               || (is_same_v<T, __nv_fp8_e4m3> && is_same_v<WeightType, __nv_fp4_e2m1>))
        && is_same_v<EpilogueTag, EpilogueOpDefault>;
#else
    return false;
#endif
}
```

So SM120 NVFP4 MoE is compiled **only** when `CUTLASS_ARCH_MMA_SM120_SUPPORTED` is set,
and only for:
- NVFP4 × NVFP4 (`__nv_fp4_e2m1` acts AND weights), or
- FP8×MXFP4 (`__nv_fp8_e4m3` acts × `__nv_fp4_e2m1` weights, MXFPX scaling).

Our target (NVFP4 × NVFP4) is the first case.

---

## 3. Input layout, scale format, and per-expert/per-token pointers

### 3.1 A and B matrix layouts

`moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl:793–796`:
```cpp
using LayoutA = TmaWarpSpecializedGroupedGemmInput::LayoutA;   // row-major
using LayoutB = TmaWarpSpecializedGroupedGemmInput::LayoutB;   // column-major
```

Both are **pointer layouts** (`LayoutA*`, `LayoutB*`) because this is a grouped GEMM
(one pointer per expert).

`include/moe_gemm_kernels.h:82–97`:
```cpp
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using StrideA = remove_pointer_t<TagToStrideA_t<LayoutA*>>;
using StrideB = remove_pointer_t<TagToStrideB_t<LayoutB*>>;
```

These are the logical layouts **before** any SwapAB transformation. When
`hopper_input.swap_ab` is true (common for small-M MoE), TRT-LLM swaps A and B at the
kernel-argument level. The launcher does this at `moe_gemm_tma_ws_launcher.inl:789–800`:
```cpp
using SwappedMainloopElementA = conditional_t<SwapAB, MainloopElementWeight, MainloopElementAct>;
using SwappedMainloopElementB = conditional_t<SwapAB, MainloopElementAct, MainloopElementWeight>;
constexpr auto SwappedAlignmentA = SwapAB ? AlignmentWeight : AlignmentAct;
constexpr auto SwappedAlignmentB = SwapAB ? AlignmentAct : AlignmentWeight;
```

Plus the pointers are swapped at `:892–902`. This means the kernel that actually runs
treats the weights as A (row-major FP4 weights) and the activations as B (column-major
FP4 activations) for FC1 when SwapAB=true. **The on-disk layouts do not change** —
only the GEMM-kernel-internal assignment of which tensor is A vs. B.

### 3.2 SwapAB logic for NVFP4 MoE FC1

For FC1, the logical problem is `(tokens, hidden) @ (hidden, inter_size) = (tokens, inter)`,
with tokens small (one MoE token-per-expert batch) and inter_size large.
With SwapAB=true the physical role is inverted to
`(inter_size, hidden) @ (hidden, tokens) = (inter_size, tokens)`.
TRT-LLM's FC1 tile config for SM120 is `CTA_M=128, CTA_N=128, CTA_K=64 bytes = 128 FP4`
with SwapAB=true in the P5 codepath and SwapAB=false in the P1 codepath (see §1.4).

### 3.3 Alignments

`moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl:655–662`:
```cpp
constexpr static int AlignmentAct = 128 / sizeof_bits<ElementAct>::value;  // FP4 → 32
constexpr static int AlignmentWeight = IsWFP4AFP8 ? 128
    : (128 / sizeof_bits<ElementWeight>::value);                             // FP4 → 32
```

`include/moe_gemm_kernels.h:99–120` documents the SF alignment constraints:
```cpp
constexpr static int NVFP4BlockScaleVectorSize = 16;
constexpr static int MXFPXBlockScaleVectorSize = 32;
using NVFP4BlockScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<NVFP4BlockScaleVectorSize>;

constexpr static int MinNDimAlignmentNVFP4 = cute::size<0>(NVFP4BlockScaledConfig::SfAtom{});  // 128
constexpr static int MinKDimAlignmentNVFP4 = cute::size<1>(NVFP4BlockScaledConfig::SfAtom{});  // 64
```

Comment at `:105–108` explicitly says: for NVFP4 the outer dim (M or N) must be a
multiple of 128, and the inner dim (K) must be a multiple of 64. TRT-LLM's `alignToSfDim`
helper (`include/moe_gemm_kernels.h:122–126`) pads up.

### 3.4 Scale factor format: per-16-element FP8 E4M3, SWIZZLED 128x4 tile

`include/moe_gemm_kernels.h:99–103`:
```cpp
constexpr static int NVFP4BlockScaleVectorSize = 16;
using NVFP4BlockScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<NVFP4BlockScaleVectorSize>;
```

The `SfAtom` from that config has shape `(128, 64)` for NVFP4 (outer × inner), based on
the `MinNDimAlignmentNVFP4 = 128` and `MinKDimAlignmentNVFP4 = 64` computed at lines
110–111 and 119–120.

The SF layout in memory is **SWIZZLED**, not linear. Canonical definition at
`cpp/tensorrt_llm/kernels/quantization.h:24–39`:
```cpp
enum class QuantizationSFLayout
{
    // Block scale factors are stored in swizzled layout for cutlass FP4 kernel. Scale factor
    // blocks are organized in 512-byte blocks in global memory, with each block having 128x4 FP8 values.
    // The SF matrix dimensions are therefore padded - rows to the nearest multiple of 128 and columns to
    // the nearest multiple of 4.
    //
    // The scale factor block rows map to data block rows in an interleaved pattern:
    // For a scale factor row 'i', it maps to data block row: (i % 4) * 32 + (i / 4)
    // Column 'j' in the scale factor block corresponds to scaling the j-th block in the data tensor.
    //
    // Please refer to https://nvbugs/4165523 for more details about the swizzled layout.
    SWIZZLED,
    LINEAR
};
```

The swizzle formula for a `(mIdx, kIdx)` SF-vector index is at
`cpp/tensorrt_llm/kernels/quantization.cuh:673–711`:
```cpp
// SF layout [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
// --> index  [mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]
int32_t innerKIdx = (kIdx % 4);
int32_t innerMIdx = (mIdx % (32 * 4)) / 32;          // M tile layout [32,4] is column-major
int32_t outerMIdx = (mIdx % 32);
int32_t kTileIdx  = (kIdx / 4);
int32_t numKTiles = (numColVecs + 3) / 4;
int32_t mTileIdx  = mIdx / (32 * 4);
int32_t numMTiles = (numRows.value_or(0) + 127) / 128;
```

Row/column padding happens in
`cpp/tensorrt_llm/kernels/quantization.h:52–57`:
```cpp
inline int64_t computeSwizzledLayoutSFSize(int totalRow, int totalColumn) {
    int paddedRow = PadUpFn(totalRow, 128);
    int paddedColumn = PadUpFn(totalColumn, 4);
    return static_cast<int64_t>(paddedRow) * paddedColumn;
}
```

Each FP8 E4M3 scale factor covers 16 FP4 data elements in the K direction. Scale element
type at the runtime interface is `uint8_t` alias
(`include/moe_gemm_kernels.h:190` `using ElementSF = uint8_t;`), interpreted as
`cutlass::float_ue4m3_t` inside the kernel
(`moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl:645`:
`using ElementSF = std::conditional_t<IsMXFPX, cutlass::float_ue8m0_t, cutlass::float_ue4m3_t>`).

### 3.5 Per-expert and per-token scale pointers

`moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl:880–938` shows how the mainloop
arguments are constructed. For `IsBlockScaled && !SwapAB`:
```cpp
MainloopArguments{
    reinterpret_cast<ElementAct const**>(tma_ws_input.ptr_act),
    reinterpret_cast<StrideA*>(tma_ws_input.stride_act),
    reinterpret_cast<ElementWeight const**>(tma_ws_input.ptr_weight),
    reinterpret_cast<StrideB*>(tma_ws_input.stride_weight),
    reinterpret_cast<ElementSF const**>(tma_ws_input.fpX_block_scaling_factors_act),
    <stride of SFA layout>,
    reinterpret_cast<ElementSF const**>(tma_ws_input.fpX_block_scaling_factors_weight),
    <stride of SFB layout>,
};
```

Each of the five `***`-pointer fields is an array of length `num_experts_per_node`.

**Per-expert per-token activation layout:** in a MoE GEMM, each expert sees the subset of
tokens routed to it. TRT-LLM maintains `expert_first_token_offset[0..num_experts]` to
describe this (`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:1284–1286`).
The per-expert activation pointer for expert `e` is offset by
`num_tokens_before_expert[e] * hidden_size` into the permuted activation buffer. Weight
pointers are simply `expert * n * k * sizeof(fp4)/2`.

The per-expert scaling factor offset is computed via `getOffsetActivationSF` at
`moe_gemm/moe_kernels.cu:983–987`, which for NVFP4 returns
`num_tokens_before_expert(padded) * (num_cols / 16)` (the per-expert SF base plus the
swizzled stride comes out of the NVFP4BlockScaledConfig tile-atom layout).

### 3.6 Per-row tensor scales (alpha per expert)

The per-expert `alpha` (the scalar that scales the accumulator before writing to D) is
threaded through as `alpha_scale_ptr_array`, a per-expert array of `float*`.
- Storage/workspace: `moe_kernels.cu:2767, 2819–2820, 2914–2915`.
- Per-expert population: `moe_kernels.cu:1313–1314`.
- Plumbed into the CollectiveEpilogue arguments at
  `moe_gemm_tma_ws_launcher.inl:983–986`:
  ```cpp
  return construct_if_true<(IsSimpleAlphaBeta && !IsFinalizeFusion), EpilogueScalars>(
      tma_ws_input.alpha_scale_ptr_array);
  ```
- **The activation kernel applies its own per-expert `fc2_act_global_scale`** on top of
  that later, during quantization to FC2's FP4 input — these are **different** scalars,
  both needed. See §7.4.

### 3.7 Per-token activation scale ("fp4 flat")

This is the output from the previous stage's activation-quant pass (or the
pre-permute input quantization for FC1). It is just a `uint8_t*` buffer laid out
in the same SWIZZLED 128x4 tile format as the weights' SF. TRT-LLM calls it
`fc1_fp4_act_scale_` (input to FC1) and `fc2_fp4_act_scale_` (output from FC1's
post-activation, input to FC2). It is **allocated as one big flat workspace buffer**,
indexed per-expert via `getOffsetActivationSF`. See
`moe_kernels.cu:1331`, `:4092–4094` and `:4103–4110` for the buffer alias between
FC1 and FC2 in MXFP4 mode.

---

## 4. `CollectiveBuilder` template argument lists

### 4.1 MoE TMA WS launcher (SM120 NVFP4 case)

Full argument list from
`moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl:797–803`:

```cpp
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,                          // cutlass::arch::Sm120
    TensorOp,                         // OpClassBlockScaledTensorOp (FP4 path)
    SwappedMainloopElementA,          // For SM120: cutlass::nv_float4_t<cutlass::float_e2m1_t>
    LayoutA*,                         // cutlass::layout::RowMajor*    (grouped-GEMM ptr layout)
    SwappedAlignmentA,                // 32 (128 / sizeof_bits<fp4> = 128/4)
    SwappedMainloopElementB,          // Same nv_float4_t<e2m1_t> for NVFP4 × NVFP4
    LayoutB*,                         // cutlass::layout::ColumnMajor*
    SwappedAlignmentB,                // 32
    ElementAccumulator,               // float
    MmaTileShape,                     // Shape<Int<CTA_M>, Int<CTA_N>, Int<CTA_K_ELEM>>
    ClusterShape,                     // Shape<_1,_1,_1> (SM120 only)
    StageCountAutoCarveout,           // sizeof(CollectiveEpilogue::SharedStorage)
    KernelSchedule                    // For SM120: KernelScheduleAuto
>::CollectiveOp;
```

Key things to note:

-   `TensorOp` at line 782–783:
    ```cpp
    using TensorOp = std::conditional_t<IsBlackwell && IsBlockScaled,
        cutlass::arch::OpClassBlockScaledTensorOp, cutlass::arch::OpClassTensorOp>;
    ```
    For NVFP4 this is always `OpClassBlockScaledTensorOp`.
-   `MmaTileShape` at line 559–560:
    ```cpp
    using MmaTileShape = Shape<Int<CTA_M*(Is2SM?2:1)>, Int<CTA_N>, Int<CTA_K*(IsSM103?3:1)>>;
    ```
    For SM120 this is just `Shape<Int<CTA_M>, Int<CTA_N>, Int<CTA_K>>`.
-   `StageCountAutoCarveout` at line 753–754: pipeline stages are computed at compile
    time as the remainder of shared memory after the epilogue's SharedStorage.
-   `KernelSchedule` at line 777–780:
    ```cpp
    using KernelScheduleSM120 = cutlass::gemm::collective::KernelScheduleAuto;
    using KernelScheduleBW    = conditional_t<IsSM120, KernelScheduleSM120, KernelScheduleSM10x>;
    using KernelSchedule      = conditional_t<IsBlackwell, KernelScheduleBW, KernelScheduleSM90>;
    ```
    SM120 leaves kernel schedule selection to CUTLASS `KernelScheduleAuto`.

### 4.2 Epilogue builder (SM120 default NO-fusion case)

`moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl:724–731`:
```cpp
using CollectiveEpilogueDefault = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,                          // cutlass::arch::Sm120
    EpilogueTensorOp,                 // OpClassBlockScaledTensorOp (NVFP4 FC1 writes BF16 D, but OpClass is BlockScaled)
    MmaTileShape,                     // same as mainloop
    ClusterShape,                     // Shape<_1,_1,_1>
    EpilogueSubTile,                  // EpilogueTileAuto for SM120 (SM100 special-cases fp4+CTA_N=256)
    ElementAccumulator,               // float
    ElementAccumulator,               // float  (compute element)
    EpilogueElementC,                 // ElementCSafe=ElementD for SM120 (no separate C)
    LayoutC*, AlignmentC,
    ElementD,                         // __nv_bfloat16 for NVFP4 FC1
    LayoutD*, AlignmentD,
    EpilogueSchedule                  // For SM120: cutlass::epilogue::TmaWarpSpecialized
>::CollectiveOp;
```

`EpilogueTensorOp` at `:703–704`:
```cpp
using EpilogueTensorOp = conditional_t<IsBlackwell && IsBlockScaled,
    OpClassBlockScaledTensorOp, OpClassTensorOp>;
```

So even the epilogue uses `OpClassBlockScaledTensorOp` for NVFP4. But importantly, the
`ElementD` is **BF16 or F16**, not FP4. Line 631–633:
```cpp
using ElementD = typename TllmToCutlassTypeAdapter<
    TmaWarpSpecializedGroupedGemmInput::OutputTypeAdaptor_t<OutputType>>::type;
```
`OutputTypeAdaptor_t<__nv_bfloat16>` is `__nv_bfloat16`
(`include/moe_gemm_kernels.h:137–138`), so `ElementD = cutlass::bfloat16_t`.

### 4.3 Plain FP4 GEMM SM120 builder (single-batch cross-reference)

`fp4_gemm/nvfp4_nvfp4_gemm_template_sm120.h:109–114`:
```cpp
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<Arch,
    cutlass::arch::OpClassBlockScaledTensorOp,
    ElementPairA, LayoutA, AlignmentA,
    ElementPairB, LayoutB, AlignmentB,
    ElementAccumulator,
    CTAShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<sizeof(CollectiveEpilogue::SharedStorage)>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative   // <<< explicit in plain-FP4 path
>::CollectiveOp;
```
Note: this non-MoE SM120 path uses the explicit `KernelTmaWarpSpecializedCooperative`
schedule rather than `KernelScheduleAuto`. The two are expected to resolve to the same
MMA atom but TRT-LLM's plain-FP4 path is more conservative about naming it.

`ElementPairA/B` is `cutlass::nv_float4_t<cutlass::float_e2m1_t>` (line 98–99). This is
the same wrapper the MoE path uses for SM120 (§2.2).

---

## 5. TiledMma construction

### 5.1 TiledMma is extracted from `CollectiveMainloop::TiledMma`

**TRT-LLM never hand-assembles the MMA atom for NVFP4 SM120.** It always extracts
`TiledMma` as `CollectiveMainloop::TiledMma` (implicitly via
`GemmUniversalAdapter`). The compile-probe functions in
`moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl:135–396` dump
`typename CollectiveMainloop::TiledMma` as-is. There is no
`SM120_MXF4NVF4_SS_m16n8k64_SB` or similar atom name anywhere in the TRT-LLM source tree
for the MoE or FP4 GEMM paths — it is chosen inside CUTLASS by the SM120
BlockScaled builder for the given `nv_float4_t<float_e2m1_t>` element pair + `MmaTileShape`
+ `KernelScheduleAuto` combination.

**Implication:** For Step 2, we replicate TRT-LLM's configuration at the CollectiveBuilder
level — same `ArchTag = Sm120`, same `OpClassBlockScaledTensorOp`, same
`nv_float4_t<float_e2m1_t>` element types, same alignments (32 elements), same
`MmaTileShape`, same `ClusterShape<1,1,1>`, same `KernelScheduleAuto` — and let CUTLASS
pick the atom. That atom **must** be the same one TRT-LLM compiles to, because the input
template arguments are identical and there is only one SM120 BlockScaled NVFP4 atom in
CUTLASS 3.7.

### 5.2 Partition API used

The compile probes at `moe_gemm_tma_ws_launcher.inl:160–208, 240–306, 338–394` show
what TRT-LLM touches. Key ops:

```cpp
TiledMma tiled_mma;
auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

// A/B register fragments via the thread's MMA slice:
auto tCrA = thread_mma.partition_fragment_A(sA(_, _, Int<0>{}));
auto tCrB = thread_mma.partition_fragment_B(sB(_, _, Int<0>{}));

// Scale-factor register fragments via mainloop helper:
auto tCrSFA = collective_mainloop.partition_fragment_SFA(sSFA(_, _, Int<0>{}), thread_mma);
auto tCrSFB = collective_mainloop.partition_fragment_SFB(sSFB(_, _, Int<0>{}), thread_mma);

// C fragment (accumulator):
auto accum_profile = thread_mma.partition_fragment_C(Shape<_128, _128>{});
// -- OR from the tiled_mma directly --
auto tCrC_profile  = partition_fragment_C(tiled_mma, take<0,2>(blk_shape_profile));
```

Key: the **SF register fragments come from `CollectiveMainloop::partition_fragment_SFA/B`**,
which is a CollectiveMainloop-level helper that already knows the
`SmemLayoutAtomSFA/B` and the `layoutSFA/B_TV` (thread-value) layouts. It is not
something we hand-assemble.

TRT-LLM also builds `smem_tiled_copy_SFA` from
`make_tiled_copy_impl(SmemCopyAtomSFA, layoutSFA_TV(tiled_mma), tile_shape_mnk.select(M,K))`
— again, all of these types come out of the CollectiveMainloop, not the atom directly.

### 5.3 What types the probes dump (= what Step 2 should dump)

Looking at `maybePrintSm120P1CompileProbe` (`:308–396`) as the FC1 probe, the set of
types the probe captures is the **ground truth list** for Step 2's layout oracle:

-   `CollectiveMainloop::DispatchPolicy::Stages` (pipeline stages)
-   `CollectiveMainloop::ThreadCount`
-   `CollectiveMainloop::SmemLayoutAtomSFA` / `...SFB`
-   `CollectiveMainloop::SmemLayoutSFA`   / `...SFB`
-   `CollectiveMainloop::SmemCopyAtomSFA` / `...SFB`
-   `CollectiveMainloop::SmemLayoutA`     / `...LayoutB`
-   `collective_mainloop.get_layoutSFA_TV(tiled_mma)` / `...SFB_TV`
-   `thread_mma.partition_fragment_B(sA)` layout (this is `tCrA` — note TRT-LLM's
    swapped convention: `tCrA = partition_fragment_B(sA)` at line 343 is because
    SwapAB is *already* baked into which tensor is called A vs B on the collective's
    own convention; the P1 probe corresponds to the `!SwapAB` case)
-   `thread_mma.partition_fragment_C(Shape<_128,_128>{})` layout

Step 2's compile-time probes will extract and print these exact fields so the resulting
output is a drop-in reference for our own kernel's layout validation.

---

## 6. Mainloop structure

### 6.1 TRT-LLM does not write the mainloop body — CUTLASS does

TRT-LLM instantiates `GemmUniversalAdapter<GemmKernel>` where
`GemmKernel = GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, void, void>`
at `moe_gemm_tma_ws_launcher.inl:805–808`. Every mainloop iteration (K, M, N), pipeline
fill, scale-factor partition, MMA-atom invocation, and fragment management happens inside
CUTLASS's own `CollectiveMainloopSm120` (or whatever the builder picks). TRT-LLM never
writes `cute::gemm(tiled_mma, tCrA, tCrB, accum)` or `mma_atom.call(...)` itself for the
FP4 paths.

**This is the single most important structural fact about this reference**:
TRT-LLM's value is not in showing us how to write a custom mainloop — it's in
showing us the exact CollectiveBuilder template arguments that produce a correct
SM120 NVFP4 MoE mainloop.

### 6.2 What we can still probe about the mainloop

-   Pipeline stages: `CollectiveMainloop::DispatchPolicy::Stages` (dumped by the probe).
-   Thread count per CTA: `CollectiveMainloop::ThreadCount` (dumped).
-   SMEM layout of A, B, SFA, SFB: same.
-   TV layouts of SF: `collective_mainloop.get_layoutSFA_TV(tiled_mma)`.

These are the compile-time facts that pin down how TRT-LLM's mainloop will consume data.
A layout probe CU that constructs a `CollectiveMainloop` identically and prints these
values is the authoritative Step 2 artifact.

### 6.3 SwapAB and K-tile partition

`moe_gemm_template_dispatch_tma_ws.h:399–402` shows that tile-K shape in the SHAPE_CASE
macro is in bytes, not elements. For FP4: `K_bytes = 64 → K_elem = 128`,
`K_bytes = 128 → K_elem = 256`. Each mainloop K step consumes `K_elem` FP4 values, which
corresponds to `K_elem / 16 = 8` or `16` NVFP4 scale-factor vectors per CTA tile K step.

At the MMA-atom level, each `mma.sync.aligned.mxf4nvf4.m16n8k64.scale_vec::4X` consumes
`K_mma = 64` FP4 elements per atom in the K direction plus 4 scale-factor vectors (one per
`K/16` sub-block). So `K_elem = 128` tile needs 2 atom K-steps per CTA-tile K-step,
and `K_elem = 256` tile needs 4.

(These atom-level counts are inferred from the PTX ISA documentation, not from TRT-LLM
source — TRT-LLM never writes the atom call.)

### 6.4 `fp4_shift_A / fp4_shift_B` — where is it?

None of the TRT-LLM FP4 paths reference `cute::fp4_shift_A` or `cute::fp4_shift_B`.
That operation is applied inside CUTLASS's own SM120 NVFP4 mainloop before the MMA atom
is called. TRT-LLM is oblivious to it.

---

## 7. Epilogue structure

This is where the picture **diverges from our `DeviceNvfp4Matrix` contract**, and it is
the most important finding for Step 2.

### 7.1 The GEMM epilogue writes BF16 (or F16), not packed FP4

`moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl:683–684`:
```cpp
using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementAccumulator, ElementC, ElementAccumulator>;
```

That is the *entire* fusion operation. The epilogue:
1.  Takes the FP32 accumulator.
2.  Multiplies by per-expert `alpha_scale_ptr_array[expert]` (float).
3.  Converts to `ElementD = __nv_bfloat16` (or `half` for the F16 instantiation).
4.  Stores through the grouped-GEMM pointer `ptr_D[expert]` with row-major stride.

**No activation function.** No FP4 packing. No block-scale computation. No per-token max.

The assertion at `moe_gemm_tma_ws_launcher.inl:679–681` is explicit:
```cpp
static_assert(is_same<EpilogueTag, EpilogueOpDefault>::value,
    "TMA Warp Specialized Grouped GEMM specialisation doesn't support fused activation");
```

### 7.2 `EpilogueFusion::FINALIZE` — FC2 path, not FC1

`moe_gemm_tma_ws_launcher.inl:733–748` shows an alternative epilogue built from a
`ScaledAccPerRowBiasPerColScaleScatter` (or the SwapAB variant
`ScaledAccPerColBiasPerRowScaleScatter`). This is the **FC2** fused-finalize path: it
scales the accumulator, applies router scales, scatters into the final output
buffer. It is only used for gemm2 in TRT-LLM. Gemm1 (FC1) always uses the plain
`LinearCombination` epilogue and `EpilogueFusion::NONE`.

The compile-probe for P15 (`maybePrintSm120P15CompileProbe` at `:211–306`) targets this
FINALIZE-fusion epilogue specifically for FC2. This is not relevant to our FC1 kernel.

### 7.3 `Relu² / doActivationKernel` is a separate post-GEMM CUDA kernel

After the FC1 GEMM writes the BF16 intermediate, the caller runs
`doActivation<T, UnfusedGemmOutputType>(...)` at
`moe_gemm/moe_kernels.cu:3160–3164` (for FP4) or `:3001–3003` (for the DeepSeek path,
same shape). The implementation of `doActivationKernel` is at
`moe_gemm/moe_kernels.cu:2063–2336`.

**Key facts about `doActivationKernel` for NVFP4 FC1:**

1.  **It takes `UnfusedGemmOutputType const* gemm_result` = `__nv_bfloat16 const*`**
    and outputs `T* output = __nv_fp4_e2m1*` along with a separate
    `TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_act_sf_flat` pointer for
    the block scales (`moe_kernels.cu:2063–2068`).
2.  **Activation choices (`moe_kernels.cu:2375–2409`):**
    `Identity`, `Gelu`, `Relu`, `Silu`, `Swiglu`, `Geglu`, `SwigluBias`, `Relu2`.
    **`Relu²` is present** (`cutlass::epilogue::thread::Relu2` at `:2406`). Our FC1 uses
    this branch.
3.  **Activation is applied post-alpha-scale** (alpha is already baked into the BF16
    GEMM output via the GEMM epilogue's `LinearCombination`).
4.  **A separate per-expert global scale `fc2_act_global_scale[expert]`** is applied during
    quantization (`moe_kernels.cu:2172`, `:2244`). This scale is distinct from `alpha` —
    see §7.4.
5.  **Gate/linear handling (gated activation):**
    `fc1_value = loadVec(gemm_result[elem + gated_off])` and
    `linear_value = loadVec(gemm_result[elem])` are both loaded
    (`moe_kernels.cu:2204–2222`). For a non-gated activation like `Relu²`, `IsGated` is
    false and only `fc1_value` is loaded, then `fn(fc1_value)` is called.
6.  **FP4 packing and block-scale writing**
    (`moe_kernels.cu:2240–2258`, via
    `quantizePackedFPXValue<__nv_bfloat16, __nv_fp4_e2m1, ComputeElem, 16>(...)`):
    ```cpp
    auto res = quantizePackedFPXValue<GemmOutputType, T, ComputeElem, VecSize>(
        post_act_val, global_scale_val, num_tokens_before_expert, expert, token,
        elem_index, inter_size, fc2_act_sf_flat,
        FpXBlockScalingType::NVFP4);
    output_vec[elem_index] = res; // res is a uint32_t = 8 FP4 values = 4 bytes
    ```
7.  **No cross-warp reduction for the per-block max-abs.** The max-abs is reduced across
    only **2 threads** (`__shfl_xor_sync(.., .., 1)`) for `VecSize=16, CVT_ELTS_PER_THREAD=8`,
    because 16/8 = 2 threads cooperatively hold the 16 values for one scale block
    (`cpp/tensorrt_llm/kernels/quantization.cuh:440–446`).

### 7.4 Block-scale computation details

`cpp/tensorrt_llm/kernels/quantization.cuh:427–510` (`cvt_warp_fp16_to_fp4`):
1.  Each thread holds 8 elements of the 16-element block (`CVT_ELTS_PER_THREAD = 8`,
    line 278).
2.  Thread computes the local max-abs across its 8 values (lines 431–438).
3.  Pair of threads shuffle-reduces to get the full 16-value max (line 442).
4.  `vecMax` is the max |x|.
5.  `SFValue = SFScaleVal * (vecMax / 6.0f)` (line 468) where `SFScaleVal` is
    `global_scale_val` = `fc2_act_global_scale[expert]`. So the per-block FP8 E4M3 stored
    scale factor is `E4M3(SFScaleVal * vecMax / 6)`.
6.  The per-block scale written to memory is `tmp.__x` where `tmp = __nv_fp8_e4m3(SFValue)`
    (lines 470–471).
7.  `outputScale = 1 / (SFValue_narrow / SFScaleVal)` where `SFValue_narrow = float(E4M3(SFValue))`,
    i.e. the output scale undoes both the quantized SF and the global scale
    (lines 474–475):
    ```cpp
    outputScale = reciprocal(SFValue * reciprocal(SFScaleVal))
    ```
8.  Each element is converted via `x * outputScale` and packed 8 per `uint32_t` via
    `fp32_vec_to_e2m1` (lines 485–506).

So the **block-scale recipe** is: `stored_scale = E4M3(global_scale * vecMax / 6)`, and the
quantized e2m1 is computed against `outputScale = 1 / E4M3^{-1}(stored_scale / global_scale)`.
This is **not** the textbook "block_scale = vecMax / 6; xq = x / block_scale" recipe —
the extra global_scale factor is baked in so that dequant downstream can recover
`E4M3(stored_scale / global_scale)` as the effective per-block scale with no further
global-scale application. This is load-bearing for any bitwise comparison.

### 7.5 SF gmem destination: swizzled per-expert tile with N-dim and K-dim padding

`cpp/tensorrt_llm/kernels/quantization.cuh:713–756` (`cvt_quant_get_sf_out_offset`):
```cpp
if (layout == QuantizationSFLayout::SWIZZLED) {
    int32_t kIdx = colVecIdx / CVT_NUM_THREADS_PER_SF;   // one SF per 16 elements
    int32_t mIdx = rowIdx;                                // per-token row
    auto SFOffset = get_sf_out_offset_128x4(batchIdx, mIdx, kIdx, numRows, numColVecs);
    return SFout + SFOffset;
}
```

So every block scale is written to a swizzled address in the flat `fc2_act_sf_flat` buffer,
offset per-expert by `getOffsetActivationSF(expert, num_tokens_before_expert, num_cols, NVFP4)`
(`moe_kernels.cu:1013–1014`). The padding rules:

-   Rows (M / token) padded up to `MinNDimAlignmentNVFP4 = 128` per expert
    (`moe_kernels.cu:2279`).
-   Cols (K / inter) padded up to `MinKDimAlignmentNVFP4 = 64` per expert
    (`moe_kernels.cu:2097`).

`doActivationKernel` also **writes zeros into the padded SF entries** at the N-dim end
(`:2272–2334`) and the K-dim end (`:2252–2258`) so the next GEMM won't see NaN during
padded loads.

### 7.6 Direct "register → packed bytes → gmem" path, no smem staging

Looking at `moe_kernels.cu:2183–2190, 2240–2258`:
```cpp
using OutputElem = std::conditional_t<IsNVFP4, uint32_t, ...>;
auto* output_vec = reinterpret_cast<OutputElem*>(output + output_offset);
...
output_vec[elem_index] = res;  // res is uint32_t = 8 packed FP4 bytes (4 bytes total)
```

`doActivationKernel` goes **straight from per-lane register** (after the `cvt` warp
reduction for block scale) **to gmem** via `STG.32`. There is no smem staging between the
pack and the global store. The "direct pack" path exists in TRT-LLM, but it is in the
**post-GEMM activation/quant kernel**, not the GEMM epilogue.

---

## 8. Output contract — comparison with our `DeviceNvfp4Matrix`

### 8.1 What TRT-LLM's FC1 GEMM actually writes (outputs at the GEMM-kernel boundary)

At the exit of `moe_gemm_runner_.moeGemm(...)` for FC1:
| Tensor | Shape | Type | Layout | Notes |
|---|---|---|---|---|
| `gemm1_output` (alias: `glu_inter_result_` or `fc1_result_`) | `[expanded_num_rows, inter_size_gated]` (per-expert stride) | `__nv_bfloat16` | row-major, pointer-per-expert | Already multiplied by per-expert alpha. Padding in the N dim is CUTLASS-handled. No activation, no quantization. |
| `alpha_scale_ptr_array[expert]` | `float[1]` per expert | `float const*` | — | Consumed by the GEMM epilogue during writeback, not emitted. |

There are **no** output FP4 bytes, **no** FP8 block scales, **no** per-token or per-expert
output tensor scales at this surface. Those only come into existence after the subsequent
`doActivationKernel` launch.

### 8.2 What TRT-LLM's FC1 → activation → FC2-input pipeline produces

At the exit of `doActivationKernel` (immediately before FC2 consumes it):
| Tensor | Shape | Type | Layout | Source of shape |
|---|---|---|---|---|
| `fc1_result_` = FC2 input activations | `[expanded_num_rows, inter_size]` (per-expert stride) | `__nv_fp4_e2m1` (packed, 2 per byte) | row-major, per-expert pointer; K-dim padded up to `MinKDimAlignmentNVFP4 = 64` | `moe_kernels.cu:2151, 2183–2190` |
| `fc2_act_sf_flat` = FC2 input block scales | swizzled `[num_experts, 128×4 tiles]` with per-expert N-pad of 128 | `uint8_t` (interpreted as `float_ue4m3_t`) | SWIZZLED 128×4 tile; per-expert offset via `getOffsetActivationSF` | `moe_kernels.cu:2243–2249, quantization.cuh:673–711` |
| `fc2_act_global_scale[expert]` | `float[num_experts]` | `float const*` | — | input constant to the activation kernel; not output |

Still no per-row tensor scale, no per-expert output tensor scale, no bias in the NVFP4
case (TRT-LLM forces `mUseBias = false` for NVFP4 — see
`mixtureOfExpertsTest.cu:1814–1818`).

### 8.3 Divergence from our `DeviceNvfp4Matrix` contract

Our `DeviceNvfp4Matrix` is (per the callsite docstrings in this repo):
```
packed_data         // packed FP4 bytes
block_scales        // per-16-element FP8 E4M3 scales for THIS matrix's data
matmul_block_scales // FP8 E4M3 scales laid out for the NEXT GEMM's input side
activation_output_scale  // per-token float scalar for dequant
per_row_tensor_scales    // per-row (per-token) float scalar
```

TRT-LLM's output for the "same abstract surface" (post FC1 + activation, ready to feed FC2):
```
fc1_result_ (packed FP4 bytes)
fc2_act_sf_flat (per-16 FP8 scales, SWIZZLED 128x4, NEXT-GEMM layout — this is both
                 `block_scales` and `matmul_block_scales`, same buffer)
alpha_scale_ptr_array[expert] (from GEMM1, applied pre-activation, not emitted; it's
                               logically a "per-expert accumulator scale", not a per-token
                               or per-row scale)
fc2_act_global_scale[expert]   (input to the activation kernel, not emitted; logically
                                the "next GEMM's alpha" pre-recipe, baked into the stored
                                block scale, not stored separately)
```

**Concrete divergences:**

1.  **No per-token `activation_output_scale`.** TRT-LLM uses a *per-expert* float, not a
    per-token float. If our contract stores a per-token scale, it cannot be produced by
    TRT-LLM's activation kernel as-is.
2.  **No per-row `per_row_tensor_scales`.** TRT-LLM's SF is per 16-element block, not per
    row. Per-row information is represented implicitly via the row-major tiling of the
    SWIZZLED 128x4 layout plus the per-expert `fc2_act_global_scale`.
3.  **`block_scales` and `matmul_block_scales` are the same buffer.** TRT-LLM does not
    distinguish "this GEMM's output quantization scales" from "next GEMM's input
    quantization scales" — they are both `fc2_act_sf_flat` in the same SWIZZLED 128x4
    format. The next GEMM consumes this buffer directly as its `ptr_SFA` or `ptr_SFB`.
4.  **The block-scale recipe bakes in the global_scale.** Per §7.4, the stored
    per-16-element scale is `E4M3(global_scale * vecMax / 6)`, not `E4M3(vecMax / 6)`.
    If our kernel uses the textbook `vecMax / 6` recipe and applies `global_scale` later,
    we will not match bitwise.

### 8.4 Implication for Step 2 oracle surface

There are three candidate oracle surfaces, in order from cheapest/worst-match to
most-expensive/best-match:

**(A) Bitwise compare against BF16 `gemm1_output`.** TRT-LLM's FC1 GEMM output, exactly
as it leaves the CUTLASS epilogue. This is BF16 with per-expert alpha already applied.
This is the **cleanest** surface to compare against because:
- It is produced by a single well-defined CUTLASS kernel we can reproduce at the
  CollectiveBuilder level.
- It does not involve any of our `DeviceNvfp4Matrix` contract questions.
- It is what the P1/P5 compile probes already inspect (`accum_profile`).
- Bitwise equality is achievable because both kernels use identical MMA atoms and
  identical accumulator/cast paths, assuming we match all CollectiveBuilder args.

**(B) Match TRT-LLM's `fc1_result_` + `fc2_act_sf_flat`** after `doActivationKernel`.
This requires our kernel to either (a) emit a dense BF16 intermediate, then
invoke `doActivationKernel` verbatim, or (b) emit packed FP4 using the same pack
recipe as TRT-LLM. Option (a) loses the benefit of our "direct pack" optimization;
option (b) requires us to match TRT-LLM's block-scale recipe bit-for-bit (which bakes in
`global_scale`, see §7.4).

**(C) Match a dequantized reference computed from TRT-LLM's outputs.** Load TRT-LLM's
`fc1_result_` and `fc2_act_sf_flat`, dequantize using TRT-LLM's own dequant recipe
(or our `nvfp4_bridge` dequant), and compare against our kernel's dequantized
intermediate. This is `relative_err < ε` territory, not bitwise — usable only as a
sanity gate, not as a kernel-correctness oracle.

**Recommendation for Step 2:** use surface **(A)**. The plan's step 2 harness should:
1.  Invoke TRT-LLM's `moeGemm(...)` directly at the FC1 entry for our chosen tile
    config and our chosen input data.
2.  Copy out the BF16 `gemm1_output` buffer.
3.  Compare our kernel's own BF16 intermediate (before activation) against it bitwise.
4.  Separately, reuse `doActivationKernel` verbatim to produce the FC2-input contract
    (packed FP4 + SWIZZLED FP8 scales), and run our own activation-and-pack kernel
    against the same BF16 intermediate, then bitwise-compare the packed output and
    the swizzled SF buffer.

This cleanly separates two orthogonal concerns — MMA correctness and
activation/pack/scale correctness — and gives each concern its own bitwise oracle.

---

## 9. TRT-LLM in-tree tests

### 9.1 `cpp/tests/unit_tests/kernels/mixtureOfExpertsTest.cu`

This is the only MoE NVFP4 C++ test in the tree
(`find cpp/tests -iname "*moe*"` returned nothing else relevant for our path).

Key facts:
- `Types` typelist at `:1764–1789` includes `WeightParams<SafeFP4, SafeFP4, __nv_bfloat16,
  SafeFP8, SafeFP8>` — NVFP4 activations × NVFP4 weights, BF16 output, FP8 input/weight
  global scales.
- The test forces NVFP4 with `mActType == Relu`
  (`:1820–1828` — skips all other activations).
- For NVFP4, bias is forced off (`:1814–1818`).
- Main test entry is `BasicPermuteTest` at `:1811`.
- Tolerance at `:214`:
  ```cpp
  : NVFP4                                         ? 0.05
  ```
- Comparison is **only at the final post-FC2 output**, not at FC1 intermediate surfaces
  (`compareFinal` at `:1652–1699`). It uses `ASSERT_NEAR(..., getTolerance(sum))`, not
  bitwise.
- The test sweeps `k ∈ {1, 2, 3}` and `num_tokens ∈ [...]` via `PermuteSweepNumTokens*`
  (`:1886–1896`).

**Verdict:** this test is **not** a bitwise kernel-level oracle. It will catch gross
errors (wrong activation, wrong quant recipe) but cannot validate our kernel at the FP4
packing bit level. Step 2 should use this test only to (a) set up a known-good TRT-LLM
runtime harness and (b) confirm our invocation of TRT-LLM produces the same tolerance-level
`mFinalOutput` as TRT-LLM's own test. The bitwise comparison happens at the intermediate
BF16 surface we extract ourselves.

---

## 10. Things not found / not investigated

-   **No standalone C++ test for `CutlassFp4GemmRunner` (plain FP4 GEMM) on SM120.**
    The plain path is instantiated but there is no in-tree test that exercises it
    at the C++ level.
-   **Generic `fp8_blockscale_gemm/sm120_blockwise_gemm/*`** — these exist
    (`find ... -iname "*sm120*"` finds them) but are for DeepSeek FP8 block-scale GEMM,
    not NVFP4. Not relevant.
-   **`CollectiveMainloopSm120Nvf4` / the actual CUTLASS implementation.** This lives
    in CUTLASS itself (outside TRT-LLM). Not investigated per the scope rules.
-   **Python kernel generator (`cutlass_kernels/python/generate_kernels.py`).**
    Referenced by a comment at `moe_gemm_tma_ws_launcher.h:27` ("Keep in sync with the
    signature generated by generate_kernels.py") but not deep-dived because our own
    build is not using the generated scaffolding.
-   **`moe_gemm_template_dispatch.h` (the non-TMA-WS dispatcher).** Only touched briefly;
    the SM120 NVFP4 path always goes through the TMA WS dispatcher.
-   **`cutlass_extensions/*`** references (e.g. `gemm/kernel/moe_cutlass_kernel.h`,
    `epilogue_helpers.h`, `compute_occupancy.h`). Referenced at
    `moe_gemm_template_dispatch_tma_ws.h:43–46`, not deep-dived; these are used by the
    SM80/SM90 non-TMA-WS paths primarily.

---

## 11. Key line-number index for quick lookup

Everything above references these files. Quick index:

| Area | File | Lines |
|---|---|---|
| MoE FP4×FP4 runner instantiation | `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp4_fp4.cu` | 24–29 |
| MoE public interface + alignment constraints | `cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_gemm_kernels.h` | 99–120, 242–415 |
| SM120 MoE tile shape validity | `moe_gemm/moe_gemm_template_dispatch_tma_ws.h` | 278–293 |
| SM120 MoE SHAPE_CASE dispatch | same | 486–500 |
| SM120 final dispatcher (into launcher) | same | 199–212 |
| MoE TMA WS launcher macro body | `moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl` | 533–1060 |
| SM120 P5/P1/P15 compile probes | same | 810–829, 135–396 |
| CollectiveMainloop builder call | same | 797–803 |
| CollectiveEpilogue default builder call | same | 724–731 |
| CollectiveEpilogue finalize builder call | same | 733–748 |
| `EpilogueOp = LinearCombination` | same | 683–684 |
| SM120 KernelScheduleAuto selection | same | 777–780 |
| SM120 `nv_float4_t` element wrapping | same | 647–652 |
| SwapAB argument construction (block-scaled) | same | 886–938 |
| SM120 validity check | `moe_gemm/moe_tma_warp_specialized_traits.h` | 37–47 |
| Plain FP4 SM120 CollectiveBuilder | `fp4_gemm/nvfp4_nvfp4_gemm_template_sm120.h` | 83–140 |
| Plain FP4 SM120 GemmUniversal+adapter | same | 137–140 |
| Plain FP4 SM120 `Sm12xOnly` wrapper (arch check trap) | same | 116–136 |
| `doActivationKernel` (post-GEMM activation + FP4 pack) | `moe_gemm/moe_kernels.cu` | 2063–2336 |
| `doActivation` dispatcher + `Relu2` | same | 2375–2409, 2414–2440 |
| `quantizePackedFPXValue` | same | 994–1041 |
| `writeSF` (padded-SF writes) | same | 1044–1089 |
| Per-expert alpha setup | same | 1311–1315, 2914–2915 |
| FC1 call site (TMA WS path) | same | 3117–3176 |
| `setupTmaWarpSpecializedInputs` (pointer plumbing) | same | 4013–4122 |
| SwizzledLayout 128x4 offset helper | `cpp/tensorrt_llm/kernels/quantization.cuh` | 673–711 |
| `cvt_quant_get_sf_out_offset` | same | 713–756 |
| `cvt_warp_fp16_to_fp4` (block-scale recipe) | same | 427–510 |
| QuantizationSFLayout docstring | `cpp/tensorrt_llm/kernels/quantization.h` | 24–39 |
| `computeSwizzledLayoutSFSize` | same | 52–57 |
| MoE NVFP4 test (tolerance only) | `cpp/tests/unit_tests/kernels/mixtureOfExpertsTest.cu` | 1811–1869 |
| Tolerance table (NVFP4 = 0.05) | same | 206–215 |
| NVFP4-specific test skip (non-Relu) | same | 1820–1828 |
| `compareFinal` (post-FC2 final comparison) | same | 1652–1699 |
