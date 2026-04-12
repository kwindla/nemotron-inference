# TRT-LLM NVFP4 SM120 P1 reference — quick distillation for step 4a

This file is a short step-4a-facing distillation of `proj-2026-04-12-1022/trtllm_architecture.md`. It exists so that step 4a can be implemented without re-reading `moe_gemm_tma_ws_launcher.inl` top to bottom. Every claim below has a `trtllm_architecture.md` section reference and a `third_party/TensorRT-LLM/cpp/...` file:line citation. If a claim here and the architecture doc disagree, the architecture doc wins.

TRT-LLM is `1.3.0rc11`, editable-installed at `/home/khkramer/src/nemotron-inference/third_party/TensorRT-LLM`. `libtensorrt_llm.so` at `third_party/TensorRT-LLM/tensorrt_llm/libs/libtensorrt_llm.so` contains embedded SM120 SASS (verified via `cuobjdump --list-elf`).

---

## 0. Target configuration for the new `NanoP1Nvfp4MoeGemm` kernel

| Axis | Value | Source |
|---|---|---|
| Arch | `cutlass::arch::Sm120` | `moe_gemm/moe_gemm_template_dispatch_tma_ws.h:486–500` (SHAPE_CASE(120, ...)) |
| Cluster | `cute::Shape<cute::_1, cute::_1, cute::_1>` | `moe_gemm/moe_gemm_template_dispatch_tma_ws.h:278–293` (cluster != 1x1x1 → false on SM120) |
| CTA M | `Int<128>` | `moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl:818` (P1 constexpr branch: `CTA_M_ == 128`) |
| CTA N | `Int<128>` | same, `CTA_N_ == 128` |
| CTA K (**elements**) | `Int<128>` (= 64 bytes × 8 bits / 4 bits-per-elem) | `:818` says `CTA_K_ == 64` (bytes); `moe_gemm_template_dispatch_tma_ws.h:399–402` converts bytes → elements for FP4 |
| SwapAB | `false` (this is the defining P1 flag vs P5) | `:818` (`!SwapAB`) |
| Activation element | `cutlass::nv_float4_t<cutlass::float_e2m1_t>` | §2.2, §4.1 of architecture doc; `:785–788` (`MainloopElementAct = nv_float4_t<ElementAct>` when `IsBlackwell && IsBlockScaled`) |
| Weight element | same (`cutlass::nv_float4_t<cutlass::float_e2m1_t>`) | `:787–788` |
| Output element (D) | `cutlass::bfloat16_t` (for Nemotron `OutputType=__nv_bfloat16`) | §4.2; `:631–633` |
| Compute / accumulator | `float` | `:801` (ElementAccumulator = float) |
| Kernel schedule | `cutlass::gemm::collective::KernelScheduleAuto` | §2.4, §4.1; `:777–778` |
| Epilogue schedule | `cutlass::epilogue::TmaWarpSpecialized` | §4.2; `:699` (`EpilogueScheduleSM120 = TmaWarpSpecialized`) |
| Epilogue tensor op | `cutlass::arch::OpClassBlockScaledTensorOp` | §4.2; `:703–704` |
| Epilogue subtile | `cutlass::epilogue::collective::EpilogueTileAuto` | `:710–712` (SM100 has a special case; SM120 uses auto) |
| Epilogue op | `cutlass::epilogue::fusion::LinearCombination<ElementD, float, ElementC, float>` | `:683–684` |
| Finalize fusion | **false** for this plan | we target `EpilogueTag = EpilogueOpDefault`, not the finalize variant at `:734–748` |

### Alignments

From `:612–677`:
- `AlignmentBitsAct = 128` and `AlignmentAct = 128 / sizeof_bits<float_e2m1_t>::value = 128 / 4 = 32` elements.
- `AlignmentBitsWeight = 128` and `AlignmentWeight = 32` elements.
- `AlignmentC = 128 / sizeof_bits<ElementC>::value` (128 / 16 for bf16 → 8 elements). `AlignmentD` same (8 elements).

For SwapAB=false (P1):
- `SwappedAlignmentA = AlignmentAct = 32`
- `SwappedAlignmentB = AlignmentWeight = 32`

### Layouts (grouped GEMM pointer variants)

From `:793–796`:
- `LayoutA = TmaWarpSpecializedGroupedGemmInput::LayoutA` — `cutlass::layout::RowMajor` at `moe_gemm_kernels.h:~` (grouped-gemm pointer layout, one pointer per expert)
- `LayoutB = TmaWarpSpecializedGroupedGemmInput::LayoutB` — `cutlass::layout::ColumnMajor`
- `LayoutC = TmaWarpSpecializedGroupedGemmInput::LayoutC` (for SwapAB=false per `:714–715`) — likely `RowMajor`
- `LayoutD = TmaWarpSpecializedGroupedGemmInput::LayoutD` (for SwapAB=false per `:718–719`) — likely `RowMajor`

All Layouts are passed to the builder as **pointer types** (`LayoutA*`, `LayoutB*`, `LayoutC*`, `LayoutD*`) because this is the pointer-array grouped-GEMM convention.

---

## 1. Mainloop CollectiveBuilder argument list (literal copy)

From `moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl:797–803` (macro body):

```cpp
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,                    // cutlass::arch::Sm120
    TensorOp,                   // cutlass::arch::OpClassBlockScaledTensorOp
    SwappedMainloopElementA,    // cutlass::nv_float4_t<cutlass::float_e2m1_t>
    LayoutA*,                   // TmaWarpSpecializedGroupedGemmInput::LayoutA*  (RowMajor*)
    SwappedAlignmentA,          // 32 elements
    SwappedMainloopElementB,    // cutlass::nv_float4_t<cutlass::float_e2m1_t>
    LayoutB*,                   // TmaWarpSpecializedGroupedGemmInput::LayoutB*  (ColumnMajor*)
    SwappedAlignmentB,          // 32 elements
    ElementAccumulator,         // float
    MmaTileShape,               // cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<128>>
    ClusterShape,               // cute::Shape<cute::_1, cute::_1, cute::_1>
    StageCountAutoCarveout,     // cutlass::gemm::collective::StageCountAutoCarveout<sizeof(CollectiveEpilogue::SharedStorage)>
    KernelSchedule              // cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```

**Note on the type-alias chain for Element A/B**: `MainloopElementAct = cutlass::nv_float4_t<ElementAct>` at `:785–786` when `IsBlackwell && IsBlockScaled`, where `ElementAct = cutlass::float_e2m1_t` for NVFP4 (`moe_gemm_kernels_fp4_fp4.cu:27` instantiates `MoeGemmRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16>` and `TllmToCutlassTypeAdapter<__nv_fp4_e2m1>` maps to `cutlass::float_e2m1_t`). For SwapAB=false: `SwappedMainloopElementA = MainloopElementAct` (`:789`).

`StageCountAutoCarveout` is computed as `StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>` at `:753–754`. This means **the epilogue type must be defined before the mainloop type** in the Traits header, or at least be a non-recursive forward reference.

---

## 2. Epilogue CollectiveBuilder argument list (default no-fusion variant)

From `moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl:724–731`:

```cpp
using CollectiveEpilogueDefault = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,                    // cutlass::arch::Sm120
    EpilogueTensorOp,           // cutlass::arch::OpClassBlockScaledTensorOp
    MmaTileShape,               // cute::Shape<_128, _128, _128>
    ClusterShape,               // cute::Shape<_1, _1, _1>
    EpilogueSubTile,            // cutlass::epilogue::collective::EpilogueTileAuto
    ElementAccumulator,         // float (the D combine accumulator)
    ElementAccumulator,         // float (the Compute element)
    EpilogueElementC,           // = ElementCSafe = ElementD for SM120 (there is no separate C in non-finalize case)
    LayoutC*,                   // TmaWarpSpecializedGroupedGemmInput::LayoutC*  (SwapAB=false: RowMajor*)
    AlignmentC,                 // 128 / sizeof_bits<ElementC>::value  (8 for bf16)
    ElementD,                   // cutlass::bfloat16_t
    LayoutD*,                   // TmaWarpSpecializedGroupedGemmInput::LayoutD*  (SwapAB=false: RowMajor*)
    AlignmentD,                 // 128 / sizeof_bits<ElementD>::value  (8 for bf16)
    EpilogueSchedule            // cutlass::epilogue::TmaWarpSpecialized  (SM120)
>::CollectiveOp;
```

The epilogue op that the builder uses internally is `cutlass::epilogue::fusion::LinearCombination<ElementD, float, ElementC, float>` (`:683–684`). This computes `alpha * accum + beta * C` where `beta = 0` because `ElementC = void` in practice (no source tensor). **No activation, no quantization, no FP4 packing** — the epilogue only writes BF16 dense output.

**Critical architectural fact** (repeated from the architecture doc's §7 and TL;DR): TRT-LLM's MoE GEMM kernel epilogue writes BF16 dense, NOT packed FP4. The FP4 packing + `Relu²` lives in a separate post-GEMM CUDA kernel `doActivationKernel` at `moe_kernels.cu:2063`, which our plan does **not** mirror — our fused direct-pack epilogue is a Nemotron-specific optimization that combines both stages into one kernel, validated against the existing kP5 direct-pack surface (`RunP5NativeDirectPackOracleForTesting`), not against TRT-LLM.

---

## 3. TiledMma extraction

From `:810–811` (P5 probe) and `:818–821` (P1 probe) and `:822–824` (P15 probe): the probes always extract

```cpp
using TiledMma = typename CollectiveMainloop::TiledMma;
TiledMma tiled_mma;
auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
```

TRT-LLM **never** names an MMA atom directly. There is no `SM120_MXF4NVF4_SS_m16n8k64_SB` or similar name in the TRT-LLM tree (verified by `grep`). The atom is whatever CUTLASS's SM120 BlockScaled builder picks for `nv_float4_t<float_e2m1_t>` inputs + `MmaTileShape<_128,_128,_128>` + `KernelScheduleAuto`. The implication for step 4a: **do not hand-assemble the atom**; use `NanoP1Nvfp4MoeGemmCollectiveMainloop::TiledMma`.

### Partition API (from the P1 probe at `:338–346`)

```cpp
auto tCrA = thread_mma.partition_fragment_B(sA);   // Note: the "B" name here is CUTE's internal
auto tCrB = thread_mma.partition_fragment_A(sB);   //   convention; with SwapAB=false,
                                                    //   our "A=activation" is collective's "B" operand
auto tCrSFA = collective_mainloop.partition_fragment_SFA(sSFA(_, _, Int<0>{}), thread_mma);
auto tCrSFB = collective_mainloop.partition_fragment_SFB(sSFB(_, _, Int<0>{}), thread_mma);
auto accum_profile = thread_mma.partition_fragment_C(Shape<_128, _128>{});
```

Note the **partition_fragment_B-for-A / partition_fragment_A-for-B inversion**: at the CollectiveMainloop level, once `SwapAB=false` is resolved, the "A" operand inside the CollectiveMainloop is actually the weight (swapped) and "B" is the activation. This is because at `:789–790`:
```cpp
using SwappedMainloopElementA = conditional_t<SwapAB, MainloopElementWeight, MainloopElementAct>;
using SwappedMainloopElementB = conditional_t<SwapAB, MainloopElementAct, MainloopElementWeight>;
```
For SwapAB=false: `SwappedMainloopElementA = MainloopElementAct` (activation) and `SwappedMainloopElementB = MainloopElementWeight` (weight). But in the P1 probe, the comment block at `:626–628` of the architecture doc says:

> TRT-LLM's swapped convention: `tCrA = partition_fragment_B(sA)` at line 343 is because SwapAB is *already* baked into which tensor is called A vs B on the collective's own convention; the P1 probe corresponds to the `!SwapAB` case

Translation: when we write our own kernel, we should use **the same inversion pattern as the P1 probe** — don't second-guess it.

---

## 4. Step 4a checklist

Step 4a's job is to create a new `UnifiedRoutedFp4Traits<UnifiedRoutedFp4Profile::kP1>` specialization that produces a bitwise-identical `CollectiveMainloop` and `TiledMma` to what TRT-LLM's launcher macro would produce for the P1 tile shape with Nemotron-relevant parameters (`ElementAct = float_e2m1_t`, `ElementWeight = float_e2m1_t`, `OutputType = __nv_bfloat16`, SwapAB=false, CTA=128x128x128 elements, cluster=1x1x1, KernelScheduleAuto).

### Required type aliases

| New `NanoP1*` alias | Must be equal to |
|---|---|
| `NanoP1Nvfp4MoeGemmArchTag` | `cutlass::arch::Sm120` |
| `NanoP1Nvfp4MoeGemmTensorOp` | `cutlass::arch::OpClassBlockScaledTensorOp` |
| `NanoP1Nvfp4MoeGemmElementAct` | `cutlass::nv_float4_t<cutlass::float_e2m1_t>` |
| `NanoP1Nvfp4MoeGemmElementWeight` | `cutlass::nv_float4_t<cutlass::float_e2m1_t>` |
| `NanoP1Nvfp4MoeGemmElementAccumulator` | `float` |
| `NanoP1Nvfp4MoeGemmElementD` | `cutlass::bfloat16_t` |
| `NanoP1Nvfp4MoeGemmLayoutA` | `cutlass::layout::RowMajor` (pointer-array) |
| `NanoP1Nvfp4MoeGemmLayoutB` | `cutlass::layout::ColumnMajor` (pointer-array) |
| `NanoP1Nvfp4MoeGemmAlignmentA` | `32` (elements) |
| `NanoP1Nvfp4MoeGemmAlignmentB` | `32` (elements) |
| `NanoP1Nvfp4MoeGemmMmaTileShape` | `cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<128>>` |
| `NanoP1Nvfp4MoeGemmClusterShape` | `cute::Shape<cute::_1, cute::_1, cute::_1>` |
| `NanoP1Nvfp4MoeGemmKernelSchedule` | `cutlass::gemm::collective::KernelScheduleAuto` |
| `NanoP1Nvfp4MoeGemmEpilogueSchedule` | `cutlass::epilogue::TmaWarpSpecialized` |
| `NanoP1Nvfp4MoeGemmEpilogueSubTile` | `cutlass::epilogue::collective::EpilogueTileAuto` |
| `NanoP1Nvfp4MoeGemmAlignmentC` | `8` (128 / 16 bits) |
| `NanoP1Nvfp4MoeGemmAlignmentD` | `8` |
| `NanoP1Nvfp4MoeGemmCollectiveEpilogue` | from `cutlass::epilogue::collective::CollectiveBuilder` with args from §2 above |
| `NanoP1Nvfp4MoeGemmStageCountAutoCarveout` | `cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename NanoP1Nvfp4MoeGemmCollectiveEpilogue::SharedStorage))>` |
| `NanoP1Nvfp4MoeGemmCollectiveMainloop` | from `cutlass::gemm::collective::CollectiveBuilder` with args from §1 above |
| `NanoP1Nvfp4MoeGemmTiledMma` | `typename NanoP1Nvfp4MoeGemmCollectiveMainloop::TiledMma` |
| `NanoP1Nvfp4MoeGemmSmemLayoutA` | `typename NanoP1Nvfp4MoeGemmCollectiveMainloop::SmemLayoutA` |
| `NanoP1Nvfp4MoeGemmSmemLayoutB` | `typename NanoP1Nvfp4MoeGemmCollectiveMainloop::SmemLayoutB` |
| `NanoP1Nvfp4MoeGemmSmemLayoutSFA` | `typename NanoP1Nvfp4MoeGemmCollectiveMainloop::SmemLayoutSFA` |
| `NanoP1Nvfp4MoeGemmSmemLayoutSFB` | `typename NanoP1Nvfp4MoeGemmCollectiveMainloop::SmemLayoutSFB` |
| `NanoP1Nvfp4MoeGemmSmemCopyAtomSFA` | `typename NanoP1Nvfp4MoeGemmCollectiveMainloop::SmemCopyAtomSFA` |
| `NanoP1Nvfp4MoeGemmSmemCopyAtomSFB` | `typename NanoP1Nvfp4MoeGemmCollectiveMainloop::SmemCopyAtomSFB` |

### Step 4a compile-time `ShowInt<N>` probe checklist

After instantiating the aliases above, step 4a should add temporary `ShowInt<N>` probes for the following values (dump them during compilation, then revert the probes before committing 4a):

1. `cute::size<0>(NanoP1Nvfp4MoeGemmTiledMma::AtomThrID{})` — threads per atom (expected: 32 for SM120 m16n8 atom)
2. `cute::cosize_v<NanoP1Nvfp4MoeGemmSmemLayoutA{}>` (after resolving stage axis)
3. `cute::cosize_v<NanoP1Nvfp4MoeGemmSmemLayoutB{}>`
4. `cute::cosize_v<NanoP1Nvfp4MoeGemmSmemLayoutSFA{}>`
5. `cute::cosize_v<NanoP1Nvfp4MoeGemmSmemLayoutSFB{}>`
6. `NanoP1Nvfp4MoeGemmCollectiveMainloop::DispatchPolicy::Stages` (pipeline stage count)
7. `NanoP1Nvfp4MoeGemmCollectiveMainloop::ThreadCount`
8. For a representative thread id (e.g. 0): each of `cute::size<i>` on the output of
   - `thread_mma.partition_fragment_A(sA(_, _, Int<0>{}))` — gives `tCrA` shape
   - `thread_mma.partition_fragment_B(sB(_, _, Int<0>{}))` — gives `tCrB` shape
   - `collective_mainloop.partition_fragment_SFA(sSFA(_, _, Int<0>{}), thread_mma)` — `tCrSFA` shape
   - `collective_mainloop.partition_fragment_SFB(sSFB(_, _, Int<0>{}), thread_mma)` — `tCrSFB` shape
   - `thread_mma.partition_fragment_C(Shape<_128, _128>{})` — accumulator shape

Expected values: see `trtllm_architecture.md` §5.3 for the set of layouts TRT-LLM's P1 probe function dumps (`SmemLayoutAtomSFA/B`, `SmemLayoutSFA/B`, `LayoutSFA_TV`, `tCrA_k0`, `tCrB_k0`, `tCsSFA`, `tCsSFB`, `tCrSFA_copy_view`, `tCrSFB_copy_view`, `tCrC_profile`).

For a value-level sanity check against TRT-LLM specifically (not required; `ShowInt<N>` probe self-consistency is the primary gate), step 2b (optional follow-up) would build a minimal C++ harness that links against `libtensorrt_llm.so` and calls TRT-LLM's own `maybePrintSm120P1CompileProbe<CollectiveMainloop>()` to emit the exact strings TRT-LLM would dump, for side-by-side comparison. That is deferred.

---

## 5. Deferred — step 2b runtime harness (if ever needed)

A runtime harness that captures:
- **Surface 1**: BF16 output of `MoeGemmRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16>::moeGemm(...)` immediately after the kernel returns (before `doActivationKernel`), as a bitwise reference for step 4b's mainloop validation
- **Surface 2**: live emission of `maybePrintSm120P1CompileProbe` text for side-by-side comparison against step 4a's `ShowInt<N>` probe output

is deferred. Current state:

- `libtensorrt_llm.so` exists with embedded SM120 SASS and exports the `MoeGemmRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16>::moeGemm` symbol (verified via `nm -D`). Linking against it works in principle but requires transitive torch dependencies (`libtorch_cpu.so`, `libc10.so`, `libtorch_cuda.so`, `libcublas.so.13`) that are **not** part of our current CMake target chain.
- The Python torch-op entry point `torch.classes.trtllm.FusedMoeRunner.run_moe` is the correct Python binding (from `cpp/tensorrt_llm/thop/moeOp.cpp:1274–1281`) — it routes through `CutlassMoeFCRunner → MoeGemmRunner::moeGemm → tma_warp_specialized_generic_moe_gemm_kernelLauncher`, which is where the P1 compile probe fires. But no existing TRT-LLM test exercises it with NVFP4 inputs; reproducing the input contract would require reverse-engineering `moeOp.cpp`.
- The P1 compile probe branch in the launcher does NOT short-circuit after printing (unlike P5 and P15). So capturing the P1 probe requires the kernel to actually launch with valid workspace + stream + tensors, not just an abort-after-print.

Re-open step 2b if:
- step 4b's mainloop test fails against a host fp32 reference and we can't explain the discrepancy, or
- step 4a's compile-time probes disagree with the architecture doc's documented types in a way that can't be explained from the CollectiveBuilder argument list alone.

Otherwise, the architecture-level reference in this file plus the host fp32 reference for step 4b are sufficient for correctness-only bring-up.
