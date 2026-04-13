#!/usr/bin/env python3
"""Step 2b harness: capture BF16 gemm1_output from flashinfer's SM120 NVFP4
CUTLASS MoE GEMM kernel — once per GEMM1 tactic, under explicit profile IDs.

Bypasses flashinfer.fused_moe.core.cutlass_fused_moe (and therefore its
AutoTuner / tactic-cache state). Talks directly to the underlying
fused_moe_runner.run_moe binding with explicit [tactic_id, 0] profile IDs, so
tactic selection is deterministic and independent of AutoTuner state.

For each GEMM1 tactic it records one BF16 dump. Step 4b picks whichever
tactic matches the P1 tile shape (per CUTLASS heuristic order: on SM120
FP4 grouped-GEMM this is tactic_id 1 → CtaShape128x128x64B).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import List, Tuple

import torch


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0

# From cutlass_heuristic.cpp:601 (get_candidate_configs_sm120 FP4 grouped
# GEMM). `MoeGemmRunner::getTmaWarpSpecializedConfigs` then duplicates each
# base tile with swap_ab=true (see moe_gemm_template_dispatch.h:657), so
# the Python-side mAllProfiles index 0..3 is swap_ab=false, index 4..7 is
# the same four tile shapes with swap_ab=true.
EXPECTED_GEMM1_TACTICS = [
    ("CtaShape128x128x128B_Cluster1x1x1", False),
    ("CtaShape128x128x64B_Cluster1x1x1",  False),
    ("CtaShape128x256x64B_Cluster1x1x1",  False),
    ("CtaShape256x128x64B_Cluster1x1x1",  False),
    ("CtaShape128x128x128B_Cluster1x1x1", True),
    ("CtaShape128x128x64B_Cluster1x1x1",  True),
    ("CtaShape128x256x64B_Cluster1x1x1",  True),
    ("CtaShape256x128x64B_Cluster1x1x1",  True),
]
# The P1 profile that plan v6 references is `CtaShape128x128x64B` with
# SwapAB=false (per moe_gemm_tma_ws_launcher.inl:818). In the enumerated
# list above, that is tactic_id 1.
P1_GEMM1_TACTIC_ID = 1
P1_GEMM1_TACTIC_LABEL = EXPECTED_GEMM1_TACTICS[P1_GEMM1_TACTIC_ID][0]
P1_GEMM1_SWAP_AB = EXPECTED_GEMM1_TACTICS[P1_GEMM1_TACTIC_ID][1]


def quant_one_expert(w_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from flashinfer.fp4_quantization import nvfp4_quantize, SfLayout

    w_amax = w_e.abs().float().max().clamp_min(1e-4)
    w_gs = torch.tensor(
        FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w_amax.item(),
        device=w_e.device,
        dtype=torch.float32,
    )
    w_fp4, w_sf = nvfp4_quantize(
        w_e.contiguous(),
        w_gs,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=True,
    )
    return w_fp4, w_sf, w_gs


def write_tensor_raw(tensor: torch.Tensor, path: pathlib.Path) -> None:
    t = tensor.detach().cpu().contiguous()
    byte_view = t.view(torch.uint8) if t.dtype == torch.bfloat16 else t
    try:
        raw = byte_view.numpy().tobytes()
    except TypeError:
        raw = bytes(byte_view.untyped_storage())
    path.write_bytes(raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--inter-size", type=int, default=256)
    parser.add_argument("--num-experts", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0xC0FFEE)
    parser.add_argument(
        "--golden-dir",
        type=pathlib.Path,
        required=True,
        help="Directory where per-tactic dump files are written.",
    )
    parser.add_argument(
        "--metadata-path",
        type=pathlib.Path,
        required=True,
        help="Destination file for the JSON metadata record.",
    )
    parser.add_argument(
        "--input-save-dir",
        type=pathlib.Path,
        required=True,
        help="Directory where raw + quantized input tensors are saved.",
    )
    parser.add_argument(
        "--flashinfer-src-root",
        type=pathlib.Path,
        required=True,
        help="Flashinfer data/csrc source root that the warm JIT cache is wired to.",
    )
    parser.add_argument(
        "--venv-root",
        type=pathlib.Path,
        required=True,
        help="Venv root that owns the flashinfer install being driven.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.stderr.write("CUDA device required for step 2b capture.\n")
        return 2

    # Fail fast if the patch hook is not compiled into the .so we are about
    # to import: a missing env var check in the kernel means the dump never
    # fires, regardless of what we do on the Python side. The check is simple:
    # the patched source contains the sentinel "NEMOTRON_TRTLLM_DUMP_GEMM1".
    patched_cuh = args.flashinfer_src_root / "fused_moe" / "cutlass_backend" / "cutlass_fused_moe_kernels.cuh"
    if not patched_cuh.exists():
        sys.stderr.write(f"patch target not found: {patched_cuh}\n")
        return 2
    if "NEMOTRON_TRTLLM_DUMP_GEMM1" not in patched_cuh.read_text():
        sys.stderr.write(
            f"ERROR: {patched_cuh} does NOT contain the dump hook sentinel.\n"
            "       Patch is not applied — did run_capture.sh revert prematurely?\n"
        )
        return 3

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    num_tokens = args.num_tokens
    hidden = args.hidden_size
    inter = args.inter_size
    num_experts = args.num_experts
    top_k = args.top_k
    expanded_num_rows = num_tokens * top_k

    try:
        from flashinfer.fused_moe.core import ActivationType
        from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm120_module
        from flashinfer.autotuner import AutoTuner
    except Exception as exc:
        sys.stderr.write(f"flashinfer import failed: {exc}\n")
        raise

    # Paranoia: ensure AutoTuner is in its no-tuning state. We are not going
    # to use it for tactic selection (we pass explicit profile_ids below),
    # but resetting its in-memory cache avoids any surprises if another
    # module in the same process has warmed the tuner for this op.
    autotuner = AutoTuner.get()
    autotuner.is_tuning_mode = False
    if hasattr(autotuner, "profiling_cache") and isinstance(autotuner.profiling_cache, dict):
        autotuner.profiling_cache.clear()

    hs = (torch.randn(num_tokens, hidden, device=device, dtype=dtype) / 10.0).contiguous()
    w1_16 = (torch.randn(num_experts, inter, hidden, device=device, dtype=dtype) / 15.0).contiguous()
    w2_16 = (torch.randn(num_experts, hidden, inter, device=device, dtype=dtype) / 15.0).contiguous()

    w1_fp4_l, w1_sf_l, w1_gs_l = [], [], []
    w2_fp4_l, w2_sf_l, w2_gs_l = [], [], []
    for e in range(num_experts):
        f, s, g = quant_one_expert(w1_16[e])
        w1_fp4_l.append(f); w1_sf_l.append(s); w1_gs_l.append(g)
        f, s, g = quant_one_expert(w2_16[e])
        w2_fp4_l.append(f); w2_sf_l.append(s); w2_gs_l.append(g)

    w1_fp4 = torch.stack(w1_fp4_l)
    w2_fp4 = torch.stack(w2_fp4_l)
    w1_sf = torch.stack(w1_sf_l)
    w2_sf = torch.stack(w2_sf_l)
    w1_gs = torch.stack(w1_gs_l)
    w2_gs = torch.stack(w2_gs_l)

    a1_gscale = torch.ones(num_experts, device=device, dtype=torch.float32)
    a2_gscale = torch.ones(num_experts, device=device, dtype=torch.float32)
    g1_alphas = (1.0 / w1_gs).to(torch.float32)
    g2_alphas = (1.0 / w2_gs).to(torch.float32)

    quant_scales = [
        a1_gscale,
        w1_sf.view(torch.int32),
        g1_alphas,
        a2_gscale,
        w2_sf.view(torch.int32),
        g2_alphas,
    ]

    token_selected_experts = torch.zeros(num_tokens, top_k, device=device, dtype=torch.int32)
    token_final_scales = torch.ones(num_tokens, top_k, device=device, dtype=torch.float32)

    args.input_save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "hs": hs.cpu(),
            "w1_16": w1_16.cpu(),
            "w2_16": w2_16.cpu(),
            "w1_fp4": w1_fp4.cpu(),
            "w2_fp4": w2_fp4.cpu(),
            "w1_sf": w1_sf.cpu(),
            "w2_sf": w2_sf.cpu(),
            "w1_gs": w1_gs.cpu(),
            "w2_gs": w2_gs.cpu(),
            "a1_gscale": a1_gscale.cpu(),
            "a2_gscale": a2_gscale.cpu(),
            "g1_alphas": g1_alphas.cpu(),
            "g2_alphas": g2_alphas.cpu(),
            "token_selected_experts": token_selected_experts.cpu(),
            "token_final_scales": token_final_scales.cpu(),
        },
        str(args.input_save_dir / "inputs.pt"),
    )

    if num_experts == 1 and top_k == 1:
        from flashinfer.fp4_quantization import nvfp4_quantize, SfLayout

        input_fp4_permuted, input_sf_permuted = nvfp4_quantize(
            hs.contiguous(),
            a1_gscale[0],
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=True,
        )
        write_tensor_raw(
            input_fp4_permuted,
            args.input_save_dir / "input_fp4_permuted.bin",
        )
        write_tensor_raw(
            input_sf_permuted,
            args.input_save_dir / "input_sf_permuted.bin",
        )

    # Resolve the low-level fused_moe_runner directly so we bypass the tuner.
    # Note: flashinfer's top-level helper `get_cutlass_fused_moe_module` returns
    # a `SimpleNamespace(cutlass_fused_moe=...)` that only exposes the high-level
    # wrapper (see core.py:715). To reach the underlying tvm-ffi module with
    # `.init(...)`, call the JIT spec directly.
    raw_jit_module = gen_cutlass_fused_moe_sm120_module(use_fast_build=False).build_and_load()
    fused_moe_runner = raw_jit_module.init(
        dtype,          # activation_dtype
        torch.int64,    # weight_dtype (FP4 viewed as long)
        dtype,          # output_dtype
        False,          # use_deepseek_fp8_block_scale
        False,          # use_w4_group_scaling
        False,          # use_mxfp8_act_scaling
        False,          # use_packed_weights
    )
    gemm1_tactic_count = int(fused_moe_runner.get_gemm1_tactic_count())
    gemm2_tactic_count = int(fused_moe_runner.get_gemm2_tactic_count())
    print(f"[harness] gemm1_tactic_count={gemm1_tactic_count} "
          f"gemm2_tactic_count={gemm2_tactic_count}")

    if gemm1_tactic_count != len(EXPECTED_GEMM1_TACTICS):
        sys.stderr.write(
            f"WARN: gemm1_tactic_count={gemm1_tactic_count} does not match the "
            f"{len(EXPECTED_GEMM1_TACTICS)} tactics expected from "
            "cutlass_heuristic.cpp:601 (base tiles) + "
            "moe_gemm_template_dispatch.h:657 (SwapAB duplication). "
            "The tactic-id -> tile-shape labels in metadata may be out of "
            "date; verify against the source before trusting step 4b's "
            "tactic selection.\n"
        )

    # Pick a valid GEMM2 tactic once; GEMM2 output never hits our dump hook.
    gemm2_tactic_id = 0 if gemm2_tactic_count > 0 else -1

    args.golden_dir.mkdir(parents=True, exist_ok=True)

    captured: List[dict] = []
    overall_t0 = time.monotonic()
    for tactic_id in range(gemm1_tactic_count):
        dump_path = args.golden_dir / f"bf16_gemm1_tactic{tactic_id}.bin"
        dump_path.unlink(missing_ok=True)  # ensure no stale bytes survive

        os.environ["NEMOTRON_TRTLLM_DUMP_GEMM1"] = str(dump_path)

        # Pre-run wall clock so we can verify the dump actually fires this run.
        t0 = time.time()
        out = torch.empty(num_tokens, hidden, device=device, dtype=dtype)
        fused_moe_runner.run_moe(
            out,
            hs,
            token_selected_experts,
            token_final_scales,
            w1_fp4.view(torch.long),
            None,                         # fc1_expert_biases
            w2_fp4.view(torch.long),
            None,                         # fc2_expert_biases
            quant_scales,
            None,                         # input_sf
            None, None, None,             # swiglu alpha/beta/limit
            1, 0,                         # tp_size, tp_rank
            1, 0,                         # ep_size, ep_rank
            1, 0,                         # cluster_size, cluster_rank
            False,                        # enable_alltoall
            False,                        # min_latency_mode
            [tactic_id, gemm2_tactic_id], # profile_ids — explicit
            False,                        # enable_pdl
            int(ActivationType.Relu2),
        )
        torch.cuda.synchronize()

        if not dump_path.exists():
            sys.stderr.write(
                f"ERROR: tactic {tactic_id} finished but no dump file at {dump_path}\n"
                "       Did the patch actually get compiled into the .so?\n"
            )
            return 4
        stat = dump_path.stat()
        if stat.st_mtime < t0 - 0.5:  # tolerate small clock skew
            sys.stderr.write(
                f"ERROR: dump file {dump_path} has stale mtime ({stat.st_mtime}) "
                f"compared to pre-run timestamp ({t0}). Stale write, not this run.\n"
            )
            return 5
        if stat.st_size <= 64:
            sys.stderr.write(
                f"ERROR: dump file {dump_path} is too small ({stat.st_size} bytes) "
                "— expected 64 byte header plus BF16 body.\n"
            )
            return 6

        if tactic_id < len(EXPECTED_GEMM1_TACTICS):
            label, expected_swap_ab = EXPECTED_GEMM1_TACTICS[tactic_id]
        else:
            label, expected_swap_ab = "unknown", None
        captured.append({
            "tactic_id": tactic_id,
            "expected_tile_label": label,
            "expected_swap_ab": expected_swap_ab,
            "dump_path": str(dump_path),
            "dump_size": stat.st_size,
            "dump_mtime": stat.st_mtime,
            "is_plan_v6_p1": tactic_id == P1_GEMM1_TACTIC_ID,
        })
        swap_tag = "swap_ab=true" if expected_swap_ab else "swap_ab=false"
        print(f"[harness] tactic {tactic_id} ({label}, {swap_tag}): {stat.st_size} bytes -> {dump_path.name}")

    # Save the final MoE output from the last tactic run purely for diagnostic
    # reference; the GEMM1 BF16 boundary, not the post-FC2 output, is the step
    # 4b oracle.
    torch.save({"moe_output": out.cpu()}, str(args.input_save_dir / "final_moe_output.pt"))

    expected_body_bytes = expanded_num_rows * inter * 2
    metadata = {
        "seed": args.seed,
        "shape": {
            "num_tokens": num_tokens,
            "hidden_size": hidden,
            "inter_size": inter,
            "num_experts": num_experts,
            "top_k": top_k,
            "expanded_num_rows": expanded_num_rows,
            "fc1_out_size": inter,
            "is_gated": False,
            "activation": "Relu2",
            "dtype": "bfloat16",
            "quant_dtype": "nvfp4",
        },
        "bf16_header_bytes": 64,
        "bf16_body_bytes_expected": expected_body_bytes,
        "captured_tactics": captured,
        "gemm1_tactic_count": gemm1_tactic_count,
        "gemm2_tactic_count": gemm2_tactic_count,
        "gemm2_tactic_used": gemm2_tactic_id,
        "p1_gemm1_tactic_id": P1_GEMM1_TACTIC_ID,
        "p1_gemm1_tactic_label": P1_GEMM1_TACTIC_LABEL,
        "p1_gemm1_swap_ab": P1_GEMM1_SWAP_AB,
        "flashinfer_src_root": str(args.flashinfer_src_root),
        "venv_root": str(args.venv_root),
        "runner_class": "flashinfer.jit.fused_moe.gen_cutlass_fused_moe_sm120_module().build_and_load().init(...)",
        "invocation": "fused_moe_runner.run_moe(..., [tactic_id, gemm2_tactic_id], ...) (AutoTuner bypassed)",
        "permuted_activation_dump": {
            "input_fp4_permuted": str((args.input_save_dir / "input_fp4_permuted.bin").resolve())
            if num_experts == 1 and top_k == 1 else None,
            "input_sf_permuted": str((args.input_save_dir / "input_sf_permuted.bin").resolve())
            if num_experts == 1 and top_k == 1 else None,
        },
        "elapsed_sec": round(time.monotonic() - overall_t0, 3),
    }
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[harness] metadata: {args.metadata_path}")
    print(f"[harness] P1 tactic: tactic_id={P1_GEMM1_TACTIC_ID} label={P1_GEMM1_TACTIC_LABEL}")
    print(f"[harness] P1 dump:   {args.golden_dir / f'bf16_gemm1_tactic{P1_GEMM1_TACTIC_ID}.bin'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
