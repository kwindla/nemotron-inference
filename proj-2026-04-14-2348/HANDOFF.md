# NanoP1 Debug Handoff

This directory is the current pickup point for the `NanoP1` routed-FC1 kernel
debugging work.

## Current Diagnosis

- The production activation `B` row permutation bug was real and has been fixed
  in `runtime/src/backend/fused_moe_prefill/nano_p1_kernel.cuh`.
- After that fix, the late-K activation `B` register mismatch against live
  FlashInfer is explained by a lane-class-specific `k_block` labeling
  difference for `tid=48,80`, not by a remaining payload-byte mismatch.
- Traced P5 and Nano local consumers now match exactly on synthetic `B` and
  synthetic `SFB` surfaces.
- The remaining unresolved surface is activation `SFB` against live. The old
  coordinate-reconstruction probe was wrong; the corrected traced `SFB`
  fragment still does not match live except on a small subset.

The running log for this is:

- `proj-2026-04-14-2348/OPERAND_REALIGNMENT_FINDINGS.md`

The focused debugging plan that led here is:

- `proj-2026-04-14-2348/FOCUSED_B_BOUNDARY_PLAN.md`

## What Is In This Commit

- Production/runtime changes:
  - `runtime/src/backend/fused_moe_prefill/nano_p1_kernel.cuh`
  - `runtime/src/backend/fused_moe_prefill/nvfp4_bridge.cuh`
- Debug/test executables:
  - `testing/backend/nano_p1_b_operand_probe.cu`
  - `testing/backend/nano_p1_a_operand_probe.cu`
  - `testing/backend/nano_p1_a_scale_probe.cu`
  - `testing/backend/nano_p1_layout_compare.cu`
  - `testing/backend/nano_p1_reference_b_operand_probe.cu`
  - `testing/backend/nano_p1_mainloop_oracle_test.cpp`
  - `testing/backend/nano_p1_direct_pack_oracle_test.cpp`
  - `testing/CMakeLists.txt`
- Analysis scripts:
  - `proj-2026-04-14-2348/*.py`
  - `proj-2026-04-12-1022/trtllm_reference/compare_b_operand_probe.py`
  - `proj-2026-04-12-1022/trtllm_reference/compare_w1_runtime_b_probe.py`
- Live FlashInfer capture machinery:
  - `proj-2026-04-12-1022/trtllm_reference/capture_bf16_gemm1.py`
  - `proj-2026-04-12-1022/trtllm_reference/run_capture.sh`
  - `proj-2026-04-12-1022/trtllm_reference/patches/flashinfer_bf16_gemm1_dump.patch`

No generated `archive/` artifacts are required from this machine. They can be
regenerated from the committed code and notes.

## Prerequisites

- CUDA-capable machine with the same repo checkout.
- `flashinfer==0.6.6` in both:
  - `.venv-trtllm`
  - `vllm-env-cu128`
- A warmed FlashInfer fused-MoE JIT cache under:
  - `~/.cache/flashinfer/0.6.6/120a/cached_ops/fused_moe_120/`

## Build Targets

Typical rebuild:

```bash
cmake -S . -B build
cmake --build build --target \
  nano_p1_mainloop_oracle_test \
  nano_p1_direct_pack_oracle_test \
  nano_p1_b_operand_probe \
  nano_p1_a_operand_probe \
  nano_p1_a_scale_probe \
  nano_p1_layout_compare \
  nano_p1_reference_b_operand_probe -j8
```

## Live Capture

`run_capture.sh` now applies the repo-managed FlashInfer patch set from the
FlashInfer `data/` root, not from the machine-local `csrc/` root.

Basic usage:

```bash
bash proj-2026-04-12-1022/trtllm_reference/run_capture.sh \
  --golden-dir proj-2026-04-12-1022/trtllm_reference/golden_probe_tmp
```

The script auto-detects the warmed JIT source tree, applies
`flashinfer_bf16_gemm1_dump.patch`, runs the harness, then reverts the patch.

## Most Important Local Comparisons

- Live vs runtime activation `B`:
  - `proj-2026-04-14-2348/compare_activation_probe_alignment.py`
- Live vs runtime weight-side synthetics:
  - `proj-2026-04-14-2348/analyze_weight_operand_synthetics.py`
  - `proj-2026-04-14-2348/analyze_weight_reg_pre_synthetics.py`
  - `proj-2026-04-14-2348/analyze_live_weight_scale_post.py`
- Focused sweep helpers:
  - `proj-2026-04-14-2348/run_focused_b_probe_sweep.py`
  - `proj-2026-04-14-2348/run_focused_b_stage_slot_compare.py`

## Best Next Step

Do not make another production patch first.

The highest-signal next move is to stabilize the activation `SFB` oracle:

1. Capture a live late-K activation-scale surface with semantic source tags, or
   extend the existing live probe so `scale_reg_post` can be tied back to the
   actual source row/block coordinates.
2. Compare that live semantic surface to:
   - production late-K `scale_reg_post` from `nano_p1_mainloop_oracle_test`
   - traced local `partition_fragment_SFB` from
     `nano_p1_reference_b_operand_probe`
3. Only patch production after that surface is aligned.
