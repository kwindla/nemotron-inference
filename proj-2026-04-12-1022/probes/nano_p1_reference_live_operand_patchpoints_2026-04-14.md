# NanoP1 Reference Live Operand Patch Points (2026-04-14)

This note records the exact flashinfer/JIT source locations for the **next**
reference-side operand probe. The goal is to stop losing time rediscovering
where the live TRT-LLM-equivalent operand path actually lives.

## Warm JIT Source Root

Auto-detected from:

```text
~/.cache/flashinfer/0.6.6/120a/cached_ops/fused_moe_120/build.ninja
```

Resolved source root:

```text
/home/khkramer/src/nemotron-inference/vllm-env-cu128/lib/python3.12/site-packages/flashinfer/data/csrc
```

This is the tree `run_capture.sh` will patch, not `.venv-trtllm/...`.

## Existing BF16 Dump Hook

The current external BF16 oracle patch is still the right top-level entry:

```text
fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh
```

Relevant call site:

- `cutlass_fused_moe_kernels.cuh:3064`
  `gemm_runner.moeGemm(universal_input, tma_ws_input);`
- `cutlass_fused_moe_kernels.cuh:3066`
  `sync_check_cuda_error(stream);`

The existing `flashinfer_bf16_gemm1_dump.patch` already hooks immediately after
that `sync_check_cuda_error(stream)`.

## Live Operand Path Candidates

The actual shared-memory to register operand path is deeper in the TRT-LLM
cutlass extensions tree under `nv_internal/tensorrt_llm/...`.

### Candidate 1: Interleaved Mixed-Input Path

Most promising location:

```text
nv_internal/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm/collective/sm90_mma_interleaved_tma_gmma_rs_warpspecialized_mixed_input.hpp
```

Relevant lines from the current install:

- `:907` `Tensor tCsA = mma_thread_slice.partition_A(sA);`
- `:915` `Tensor tCsA_remapped = tCsA.compose(interleave_remapping);`
- `:930` `auto smem_tiled_copy_A = make_tiled_copy_A(InternalSmemCopyAtomA{}, tiled_mma);`
- `:934` `Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA_load);`
- `:992` first `copy_A_and_extra_info(...)`
- `:1001` first `transform_A_kblock(...)`
- `:1007` first `cute::gemm(...)`

This is the best place to insert a mirrored reference operand probe because it
already has the live:

- `sA`
- `tCsA` / `tCsA_remapped`
- `tCrA_copy_view`
- per-`k_block` copy and transform sequence

If we need the exact thread contract for tracked lanes, instrument here.

### Candidate 2: Array Mixed-Input Path

Secondary location:

```text
nv_internal/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_.hpp
```

Relevant lines:

- `:909` `Tensor tCsA = mma_thread_slice.partition_A(sA);`
- `:931` `auto smem_tiled_copy_A = make_tiled_copy_A(SwappedSmemCopyAtomA{}, tiled_mma);`
- `:934` `Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA_load);`
- `:980` first `Utils::copy_tensors_MK(...)`
- `:988` first `Utils::convert_A_kblock(...)`
- `:1002` first `cute::gemm(...)`

If the installed P1 instantiation is resolving through the array path instead
of the interleaved path, instrument here instead.

## Concrete Probe Shape

The mirrored live reference probe should follow the runtime-side
`nano_p1_b_operand_probe` pattern as closely as possible:

- gate on tracked threads `tid={0,1,128,129}`
- capture one small `(copy_tile or equivalent, k_block, read_stage)` surface
- dump:
  - `tCsA` or `tCsA_remapped` source coordinates
  - stage-0 smem offsets
  - first `tCrA_copy_view` words before and after the A transform step
  - the logical row tag and byte tag consumed at those offsets

Do **not** start with a broad printf flood. Add a narrow env-gated probe right
next to the existing BF16 dump workflow so one patched capture run produces:

- the current BF16 boundary dump
- the tracked live operand contract

## Attempt 1: `printf` Is Not A Reliable Channel Here

A first live probe attempt patched three candidate runtime locations with very
narrow tracked-thread `printf` probes:

- `sm90_mma_interleaved_tma_gmma_rs_warpspecialized_mixed_input.hpp`
- `sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_.hpp`
- `fused_moe_kernel_routine.cuh`

The probe:

- compiled successfully after one C++17 fix (`remove_cvref_t` -> compatible
  traits)
- did **not** perturb the existing BF16 dump path
- produced **zero** runtime probe lines during a successful
  `run_capture.sh --golden-dir .../golden_probe_tmp` run

So as of 2026-04-14:

- device-side `printf` is **not** a trustworthy telemetry channel for this
  flashinfer/JIT path in our current capture workflow
- continuing to chase `printf` insertion points is not a good use of time

The live JIT headers were restored after this result.

## Why This Matters

The standalone builder probe is now good enough to prove the reference contract
is physically 2D in `SmemLayoutA`, but it is not the final consumer oracle.
The next trustworthy answer has to come from the live flashinfer kernel path
above, not from more host-only reconstruction.

Given the failed `printf` attempt, the next live probe should use an explicit
device probe buffer or device symbol copied back on the host after
`gemm_runner.moeGemm(...)`, not console logging from inside the kernel.
