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

From `moe_gemm_kernels.h:82-89` (authoritative source, not speculation):
- `LayoutA = cutlass::layout::RowMajor` (grouped-gemm pointer layout, one pointer per expert)
- `LayoutB = cutlass::layout::ColumnMajor`
- `LayoutC = cutlass::layout::RowMajor` (for SwapAB=false; transposed to `LayoutC_T = ColumnMajor` when SwapAB=true)
- `LayoutD = cutlass::layout::RowMajor` (for SwapAB=false; transposed to `LayoutD_T = ColumnMajor` when SwapAB=true)

The header comment at `moe_gemm_kernels.h:80-81` confirms: "These are always the layout of A & B matrices, activations and weights will be assigned to either A or B based on swap_ab". SwapAB=false (our P1) binds activations → A (RowMajor) and weights → B (ColumnMajor).

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

## 4. Step 4a reference — TRT-LLM CollectiveBuilder alias table (REFERENCE ONLY — do not copy into runtime/)

**Kernel Provenance constraint** (`PLAN_RULES.md § Kernel Provenance`): the production NanoP1 kernel lives on the prefill hot path and must be written from scratch using only the CUTE atom/layout allow-list plus inline PTX for the SM120 block-scaled MMA. **`cutlass::gemm::collective::CollectiveBuilder` is forbidden in `runtime/`.** The alias table below is the TRT-LLM reference — it documents what the TRT-LLM launcher would produce for the P1 tile — but it is NOT a template for our runtime code. Step 4a's job is to define equivalent hand-written types using `cute::MMA_Atom<cute::SM120_16x8x64_TN_VS<...>>`, `cute::TiledMMA`, `cute::Copy_Atom`, `cute::Layout`, swizzle functors, and `Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA/SFB`, reproducing the same shapes, stage counts, and per-thread fragment layouts that the reference would have produced.

The old (reverted) step 4a commit `9be1580` instantiated `CollectiveBuilder` directly in `runtime/src/backend/fused_moe_prefill.cu` and was reverted in `8ea9241` because that violated the policy. Every future reference to "`NanoP1*` type bundle" in this file should be read as "our hand-written equivalent of what the table below would have produced".

Per PLAN.md v6 step 3d, **`UnifiedRoutedFp4Traits<UnifiedRoutedFp4Profile::kP1>` remains deleted** and is NOT reintroduced by step 4. The `UnifiedRoutedFp4Profile::kP1` enum constant stays intact.

### TRT-LLM reference types (for our hand-written kernel to mirror, not to instantiate)

The names below use the old `NanoP1Nvfp4MoeGemm*` prefix because they were copy-pasted from the reverted design. Read every row as "TRT-LLM's CollectiveBuilder would emit type X for this slot; our hand-written runtime code must produce an equivalent CUTE structure under name Y per `PLAN.md` step 4a".

### Reference type aliases (TRT-LLM would emit these — our runtime code mirrors the shapes, not the names)

| TRT-LLM reference alias | Would be equal to (CollectiveBuilder output) |
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

After instantiating OUR OWN hand-written types (`NanoP1MmaAtom`, `NanoP1TiledMma`, `NanoP1SmemLayoutA/B/SFA/SFB`, etc. per `PLAN.md` step 4a), add temporary `ShowInt<N>` probes for the following values (dump during compilation, then remove the probes before committing 4a). These verify that OUR hand-written CUTE types produce the shapes we designed — the values below are the TRT-LLM reference the design spec came from.

| Probe | Source (our hand-written type) | Reference (what TRT-LLM would emit) |
|---|---|---|
| threads per atom | `cute::size(typename NanoP1TiledMma::AtomThrID{})` | `32` (SM120 m16n8 atom) |
| smem A cosize | `cute::cosize_v<NanoP1SmemLayoutA{}>` | see `step_4a_probe_values.txt` from reverted commit `9be1580` for the reference values we'd compute against |
| smem B cosize | `cute::cosize_v<NanoP1SmemLayoutB{}>` | same |
| smem SFA cosize | `cute::cosize_v<NanoP1SmemLayoutSFA{}>` | same |
| smem SFB cosize | `cute::cosize_v<NanoP1SmemLayoutSFB{}>` | same |
| pipeline stages | `NanoP1PipelineStages` (we pick this, not a `DispatchPolicy::Stages`) | `4` |
| thread count | `NanoP1ThreadsPerCta` (we pick this) | `256` |
| tCrA / tCrB shapes | our partition on `NanoP1SmemLayoutA/B` slices | see `trtllm_architecture.md` §5.3 |
| tCrSFA / tCrSFB shapes | our partition on `NanoP1SmemLayoutSFA/B` | same |
| tCrC profile | `cute::partition_fragment_C(NanoP1TiledMma{}, Shape<_128, _128>{})` | same |

The reverted step 4a commit `9be1580` captured the TRT-LLM-equivalent probe values by running the CollectiveBuilder; those values are still the design target, just not the runtime implementation. Re-capture them via the same probe pattern against OUR hand-written types to confirm we reproduced the intended shapes.

For a value-level sanity check against TRT-LLM specifically (not required; `ShowInt<N>` probe self-consistency is the primary gate), step 2b (optional follow-up) would build a minimal C++ harness that links against `libtensorrt_llm.so` and calls TRT-LLM's own `maybePrintSm120P1CompileProbe<CollectiveMainloop>()` to emit the exact strings TRT-LLM would dump, for side-by-side comparison. That is deferred.

---

## 5. Step 2b runtime harness — chosen path

Plan v6 step 2 was reopened in a narrower form: capture TRT-LLM's BF16 `gemm1_output` immediately after `MoeGemmRunner::moeGemm(...)` returns for one synthetic SM120 P1 problem, so that step 4b's mainloop test can compare bitwise against a real external BF16 boundary rather than only against a host fp32→bf16 reference.

### Path not taken

- **Direct link against `libtensorrt_llm.so`** — rejected. The exports we need (`CutlassMoeFCRunner::runMoe`, `MoeGemmRunner::moeGemm`) pull in `libtorch_cpu.so`, `libc10.so`, `libtorch_cuda.so`, and `libcublas.so.13` as transitive dependencies, none of which are currently part of Nemotron's CMake target chain. Adding them would turn step 2 into a multi-day build-system project.
- **Hand-rolled CUTLASS harness** — rejected. `moeGemm` consumes a populated `GroupedGemmInput` + `TmaWarpSpecializedGroupedGemmInput` whose strides/SF pointers are normally computed by `CutlassMoeFCRunner::computeStridesTmaWarpSpecialized`. Reimplementing that logic from scratch would dwarf the step-4b kernel itself.
- **vLLM `FlashInferExperts` path** — rejected for step 2b-as-source-of-truth. It routes through the same flashinfer binding we're using, so it would only add an extra wrapper layer between Nemotron and the real kernel.
- **`trtllm_fp4_block_scale_*` flashinfer top-level entry** — rejected. That one maps to `gen_trtllm_gen_fused_moe_sm100_module`, which is the SM100 trtllm-gen path, NOT the SM120 CUTLASS path. Plan v6 step 2 explicitly warns against it.

### Path taken

Drive flashinfer's vendored TRT-LLM MoE GEMM sources directly through the low-level binding `fused_moe_runner.run_moe(...)`. The Python wrapper `flashinfer.fused_moe.core.cutlass_fused_moe(...)` is intentionally NOT used — it routes through the flashinfer AutoTuner, which leaves tactic-cache state in the process that contaminates reproducibility. The harness grabs the raw tvm-ffi JIT module by calling the JIT spec directly:

```python
from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm120_module
raw_jit_module = gen_cutlass_fused_moe_sm120_module(use_fast_build=False).build_and_load()
fused_moe_runner = raw_jit_module.init(bf16, torch.int64, bf16, False, False, False, False)
# ... later, once per GEMM1 tactic:
fused_moe_runner.run_moe(..., [tactic_id, gemm2_tactic_id], ...)
```

**Not** `flashinfer.fused_moe.core.get_cutlass_fused_moe_module(backend="120").init(...)` — that helper at `core.py:333-715` returns a `SimpleNamespace(cutlass_fused_moe=...)` that only exposes the high-level wrapper, not the tvm-ffi module with `.init(...)`. Getting the raw module requires either calling the JIT spec directly (what the harness does) or constructing a `MoERunner(...)` instance and pulling `moe_runner.fused_moe_runner` off it. The JIT-spec path is cleaner because it bypasses `MoERunner` altogether.

The JIT sources live under

    <venv>/lib/python3.12/site-packages/flashinfer/data/csrc/
      nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/...   # TRT-LLM kernel .cu/.cuh
      fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh         # moe_kernels.cu equivalent
      fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu  # Python binding

On SM120 the backend is `gen_cutlass_fused_moe_sm120_module`, which compiles the same `moe_gemm_tma_ws_launcher.inl` kernel instantiations we target in step 4b. The warm build is cached under `~/.cache/flashinfer/0.6.6/120a/cached_ops/fused_moe_120/` — the ninja file there has absolute source paths baked in, so `run_capture.sh` must patch whichever venv that cache references (auto-detected via `awk` on `build.ninja`). In the current environment that's `vllm-env-cu128/.../flashinfer/data/csrc`, not `.venv-trtllm/.../flashinfer/data/csrc`.

### The BF16 boundary contract

At the call site in `cutlass_fused_moe_kernels.cuh::CutlassMoeFCRunner::gemm1(...)`, the TMA warp-specialized NVFP4 branch pulls the following pointers from scope:

- `T` = `__nv_fp4_e2m1` (NVFP4 weights/acts)
- `WeightType` = `__nv_fp4_e2m1`
- `OutputType` = `__nv_bfloat16`
- `UnfusedGemmOutputType` = alias of `OutputType` = `__nv_bfloat16`
- `has_different_gemm_output_type = true` (T ≠ OutputType) → `has_intermediate = true`
- `gemm_output = intermediate_result` (the `glu_inter_result_` / `fc1_result_` workspace chunk passed in from the outer `runMoe`)
- `fc1_out_size` = `is_gated_activation ? inter_size * 2 : inter_size`; for Nemotron ReLU² this is `inter_size`
- Post-GEMM, `gemm_output` holds `expanded_num_rows * fc1_out_size` BF16 elements, row-major, stride `fc1_out_size`

The BF16 boundary is **post-alpha-scaled, post-BF16 epilogue cast, pre-activation, pre-pack**. This is the `alpha * (FP4 A · FP4 B) + beta*C` output of `cutlass::epilogue::fusion::LinearCombination<bf16, float, ElementC, float>` with `beta=0, ElementC=void`. No `Relu²`, no block-scale recompute, no FP4 pack. Any subsequent transform happens inside `doActivation`, which this hook intentionally precedes.

### What the patch does

`patches/flashinfer_bf16_gemm1_dump.patch` adds exactly two hunks to `cutlass_fused_moe_kernels.cuh`:

1. Five `#include`s (`<cstdint>`, `<cstdio>`, `<cstdlib>`, `<cstring>`, `<vector>`) at the top of the file.
2. A 42-line env-gated dump immediately after the `sync_check_cuda_error(stream)` that follows `gemm_runner.moeGemm(universal_input, tma_ws_input)` in the TMA WS branch of `gemm1()`. When `NEMOTRON_TRTLLM_DUMP_GEMM1=<path>` is set, the hook `cudaMemcpyAsync`s `expanded_num_rows * fc1_out_size * sizeof(UnfusedGemmOutputType)` bytes from `gemm_output` to host, and writes a 64-byte header plus the raw BF16 body to that path. See `README.md` for the byte-level header layout.

The hook is strictly a device→host copy + `std::fwrite`; it does not mutate the kernel state, does not allocate on the CUDA stream, and is compiled as a runtime branch that only fires when the env var is set. The patch is applied by `run_capture.sh`, which also reverts it on exit via an EXIT trap — the flashinfer installed tree is left unchanged after each capture run.

### Tactic selection: enumerate, do not rely on fallback

The SM120 NVFP4 grouped-GEMM base tactic list from `cutlass_heuristic.cpp:601-616` is four entries:

| base_id | `CutlassTileConfigSM120` |
|---|---|
| 0 | `CtaShape128x128x128B` |
| 1 | `CtaShape128x128x64B` |
| 2 | `CtaShape128x256x64B` |
| 3 | `CtaShape256x128x64B` |

That base list is then duplicated with `swap_ab = true` by `MoeGemmRunner::getTmaWarpSpecializedConfigs` at `moe_gemm_template_dispatch.h:657-663`, giving **8** GEMM1 tactics total. `mAllProfiles` in the binding is populated from this 8-entry list for GEMM1, then extended with GEMM2 tactics. See "Observed runtime" below for the full `tactic_id → (tile, swap_ab)` table that step 4b reads.

The previous harness approach ("pass `-1`, take `mAllProfiles.front()`") would have captured `tactic_id = 0` = `CtaShape128x128x128B` (SwapAB=false), which is NOT the tile shape plan v6 calls P1 (plan-v6 P1 = `CtaShape128x128x64B` SwapAB=false = `tactic_id = 1`). The current harness instead:

1. Bypasses `cutlass_fused_moe(...)` (and therefore the AutoTuner) entirely — it builds the raw JIT module with `gen_cutlass_fused_moe_sm120_module(...).build_and_load()`, calls `raw_jit_module.init(...)` to construct the low-level `fused_moe_runner`, and then issues `fused_moe_runner.run_moe(..., [tactic_id, gemm2_tactic_id], ...)` directly with explicit profile IDs. See "Path taken" above for the exact call sequence.
2. Resets `AutoTuner.get().profiling_cache` before the first call as paranoia against any other in-process tuning state leaking in.
3. Loops over every GEMM1 tactic id `0..get_gemm1_tactic_count()-1`, captures one BF16 dump per tactic to `bf16_gemm1_tactic${tactic_id}.bin`, and records `(tactic_id → expected tile label, expected swap_ab)` plus the `is_plan_v6_p1` flag in `bf16_gemm1_metadata.json`.
4. Verifies each dump file exists, is non-trivially sized, and has `mtime ≥ t0_of_this_run` (guards against an old file false-passing the existence check).

Step 4b's oracle picks the tactic whose `is_plan_v6_p1 == true` (currently `tactic_id == 1`, `CtaShape128x128x64B_Cluster1x1x1`, SwapAB=false), reads the corresponding dump, and compares bitwise against the NanoP1 kernel output. If `get_gemm1_tactic_count()` differs from the expected **8**, the harness emits a WARN and the metadata surfaces `gemm1_tactic_count` so step 4b can re-verify against `cutlass_heuristic.cpp` + `moe_gemm_template_dispatch.h` before trusting the label.

### Observed runtime: `get_gemm1_tactic_count() == 8`

In the current flashinfer build (`0.6.6`, `vllm-env-cu128`) `get_gemm1_tactic_count()` returns **8**. The reason is in `moe_gemm_template_dispatch.h:657`:

```cpp
auto swap_ab_configs = tma_ws_configs;
std::transform(swap_ab_configs.begin(), swap_ab_configs.end(),
               std::back_inserter(tma_ws_configs),
               [](auto& config) { config.swap_ab = true; return config; });
```

`getConfigs` duplicates every base tile shape with `swap_ab = true` appended, so the effective list is:

| tactic_id | label                              | swap_ab |
|---|---|---|
| 0 | `CtaShape128x128x128B_Cluster1x1x1` | false  |
| 1 | `CtaShape128x128x64B_Cluster1x1x1`  | false ← **plan v6 P1** |
| 2 | `CtaShape128x256x64B_Cluster1x1x1`  | false  |
| 3 | `CtaShape256x128x64B_Cluster1x1x1`  | false  |
| 4 | `CtaShape128x128x128B_Cluster1x1x1` | true   |
| 5 | `CtaShape128x128x64B_Cluster1x1x1`  | true   |
| 6 | `CtaShape128x256x64B_Cluster1x1x1`  | true   |
| 7 | `CtaShape256x128x64B_Cluster1x1x1`  | true   |

The harness captures one dump per tactic and labels both fields in `bf16_gemm1_metadata.json`. Plan v6 P1 (`SwapAB=false`, `CtaShape128x128x64B`) remains tactic_id 1.

### Observed convergence across tactics (expected and benign)

On the default synthetic problem (M=128, K=256, N=256, seed=0xC0FFEE), all 8 tactic dumps are **bitwise identical**. For the current capture:

- full-file md5 (64 B header + 65536 B body): `d220d1c7aaa6d1b19190c5b6f02b2d1d`
- body-only md5 (65536 B BF16 payload, no header): `fd46efaa59d45cd081eeb14d64d62bf1`

Both hashes are stable across tactic 0..7. Use whichever representation your diff tool makes easy; the body-only hash is invariant to header-format changes, the full-file hash is what `md5sum golden/*.bin` reports.

This was initially alarming but is expected:

1. An instrumented build (temporary `fprintf` on `config.tile_config_sm120 / config.swap_ab`) confirmed the runtime dispatch is actually selecting different tactics per call — tactic 1 hits `tile_config_sm120=128128064, swap_ab=0`, tactic 5 hits the same tile with `swap_ab=1`, etc. The underlying CUTLASS kernels are in fact different template instantiations (the primary `DispatchToTmaWSFunction<>` template at `moe_gemm_tma_ws_launcher.inl:95` is empty; an un-instantiated tile would fail to compile rather than silently falling back).

2. The observed convergence is a property of the math + low-precision output:
   - FP4 values are bounded in `[-6, 6]`, so each product `A·B` is bounded in `[-36, 36]`
   - For `K=256` (and tested up to `K=2048`), fp32 partial sums stay well below fp32's ~7-digit precision range
   - The fp32 reduction error across different tile boundaries is ≲ `1e-3`
   - BF16 has 8 mantissa bits → LSB of a value near 1.0 is ~`3.9e-3`
   - fp32 rounding error `<` BF16 LSB → every legal reduction order rounds to the **same** BF16 bit pattern

For step 4b, this means the NanoP1 kernel can be compared against any of `tactic0`..`tactic3` (SwapAB=false variants) and still validate correctness for small synthetic problems. The canonical comparison target is `bf16_gemm1_tactic1.bin` — the plan v6 P1 label — and tactics 4..7 are captured as SwapAB=true diagnostic checks (NanoP1 is SwapAB=false, so tactics 4..7 should only be used for curiosity).

The convergence may break at larger K where partial sums approach fp32 precision limits, or at larger M×N where tile boundary effects dominate. If step 4b sees a mismatch against tactic1 but finds another tactic that matches, re-verify the instrumentation trace from this run (saved as a diagnostic — if not, re-instrument via the same temporary `fprintf` pattern).

### Files

- `README.md` — operator-facing overview and run instructions
- `capture_bf16_gemm1.py` — per-tactic harness: builds synthetic inputs, enumerates `get_gemm1_tactic_count()`, calls `fused_moe_runner.run_moe(..., [tactic_id, gemm2_tactic_id], ...)` for each tactic, verifies dump mtime, writes metadata
- `run_capture.sh` — auto-detect venv from warm ninja cache → clean stale `golden/` artifacts → apply patch → run harness → revert patch (always, via trap)
- `patches/flashinfer_bf16_gemm1_dump.patch` — the ~47-line unified diff; applies cleanly to any flashinfer data/csrc tree the warm cache points at
- `golden/` — run output: per-tactic `bf16_gemm1_tactic${id}.bin` dumps, `bf16_gemm1_metadata.json`, `inputs.pt`, `final_moe_output.pt`

### Gate for step 4b

Once `run_capture.sh` has produced `golden/bf16_gemm1_tactic1.bin` (= the plan-v6 P1 tactic: `CtaShape128x128x64B_Cluster1x1x1`) plus the full set of other tactic dumps, `bf16_gemm1_metadata.json`, and `inputs.pt`, step 4b has:
- The BF16 reference tensor to compare bitwise against (tactic 1 specifically; the others are captured for diagnostic reference)
- The exact synthetic inputs to feed the new NanoP1 kernel (so step 4b reproduces the problem deterministically)
- A metadata record naming the tactic labels, shapes, seed, source tree, and venv so any mismatch can be traced

Step 4b is blocked until this capture exists.
