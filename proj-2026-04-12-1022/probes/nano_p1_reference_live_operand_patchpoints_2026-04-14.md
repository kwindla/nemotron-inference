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

## Corrected Live Operand Path

The earlier mixed-input patch-point guess was wrong for the Nano `tactic_id=1`
reference path.

Why it was wrong:

- `moe_gemm_tma_ws_mixed_input_launcher.inl` hard-checks
  `hopper_inputs.swap_ab == true`
- the plan-v6 P1 tactic we are using for the Nano oracle is still
  `swap_ab = false`
- in the non-min-latency fused-MoE path, flashinfer quantizes BF16 activations
  to NVFP4 first, then launches an SM120 FP4xFP4 blockscaled GEMM

So the live TRT-LLM-equivalent mainloop is not the Hopper mixed-input path.
It is the CUTLASS SM120 blockscaled collective instantiated through:

```text
cutlass/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl
```

which resolves to:

```text
cutlass/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp
```

### Actual Patch Point: SM120 Blockscaled B Operand Path

Relevant lines from the current install:

- `:721` `auto thread_mma = tiled_mma.get_thread_slice(thread_idx);`
- `:743` `auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);`
- `:745` `Tensor tCsB = smem_thr_copy_B.partition_S(as_position_independent_swizzle_tensor(sB));`
- `:747` `Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);`
- `:795` `int read_stage = smem_pipe_read.index();`
- `:797` `auto tCsB_stage = tCsB(_,_,_,read_stage);`
- `:804` `copy(smem_tiled_copy_B, tCsB_stage(_,_,k_block), tCrB_copy_view(_,_,k_block));`
- `:809` `fp4_shift_B(MMAOp{}, tCrB_copy_view(_,_,k_block));`

This is the right live reference operand boundary for the tracked B-side
consumer probe because it already has:

- the real `thread_idx`
- the real `SmemLayoutB`
- the real `tCsB` copy-source tensor
- the real `tCrB_copy_view`
- the real per-stage / per-`k_block` `copy -> fp4_shift_B -> gemm` sequence

Any new live operand probe should start here, not in the mixed-input TRT-LLM
extension headers.

## Concrete Probe Shape

The mirrored live reference probe should follow the runtime-side
`nano_p1_b_operand_probe` pattern as closely as possible:

- gate on tracked threads `tid={0,1,128,129}`
- gate on `blockIdx == (0,0,0)` to avoid grouped-GEMM noise
- capture one small `(n_tile, k_block, read_stage)` surface from the live
  `tCsB -> tCrB_copy_view -> fp4_shift_B` path
- dump:
  - `partC` anchor coordinates for the tracked `n_tile`
  - `tCsB` logical source coordinates from an identity tensor
  - stage-0 smem offsets
  - first `tCrB` register words before and after `fp4_shift_B`

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
- the previous mixed-input candidate locations should be treated as a closed
  false lead for the Nano `tactic_id=1` path

The live JIT headers were restored after this result.

## Why This Matters

The standalone mixed-input builder probe is still useful as a structural hint,
but it is **not** the definitive live tactic-1 contract. The next trustworthy
answer has to come from the live flashinfer kernel path above, not from more
host-only reconstruction around the wrong collective family.

Given the failed `printf` attempt, the next live probe should use an explicit
device probe buffer or device symbol copied back on the host after
`gemm_runner.moeGemm(...)`, not console logging from inside the kernel.
