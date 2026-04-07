#!/usr/bin/env python3
"""Replay pinned vLLM _chunk_cumsum_fwd on saved Stage-1 inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from vllm.model_executor.layers.mamba.ops.ssd_chunk_state import _chunk_cumsum_fwd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay vLLM Triton chunk cumsum on saved dt_pre/A/dt_bias inputs."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--dt-dtype",
        choices=("fp32", "bf16"),
        default="fp32",
        help="Replay _chunk_cumsum_fwd using dt_pre loaded as fp32 or cast to bf16 first.",
    )
    return parser.parse_args()


def load_fp32(path: Path) -> torch.Tensor:
    data = torch.frombuffer(path.read_bytes(), dtype=torch.float32)
    return data.clone()


def write_fp32(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tensor.detach().contiguous().cpu().float().numpy().tobytes())
    print(f"dump: {path} ({tensor.numel()} -> fp32, shape={list(tensor.shape)})")


def mad(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    diff_sq = float(torch.sum((a - b) ** 2).item())
    ref_sq = float(torch.sum(b ** 2).item())
    return (diff_sq / ref_sq) ** 0.5 if ref_sq > 0.0 else (0.0 if diff_sq == 0.0 else float("inf"))


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")

    k_h = 128
    dt_pre = load_fp32(input_dir / "dt_pre_fp32.bin")
    A = load_fp32(input_dir / "A_fp32.bin")
    dt_bias = load_fp32(input_dir / "dt_bias_fp32.bin")
    if dt_pre.numel() % k_h != 0:
        raise ValueError("dt_pre dump size is not divisible by 128 heads")
    k_t = dt_pre.numel() // k_h
    dt_pre = dt_pre.reshape(k_t, k_h).cuda()
    if args.dt_dtype == "bf16":
        dt_pre = dt_pre.to(torch.bfloat16)
    A = A.reshape(k_h).cuda()
    dt_bias = dt_bias.reshape(k_h).cuda()

    boundaries = [0]
    while boundaries[-1] < k_t:
        boundaries.append(min(boundaries[-1] + args.chunk_size, k_t))
    cu_chunk_seqlens = torch.tensor(boundaries, dtype=torch.int32, device="cuda")

    dA_cumsum, dt_chunk = _chunk_cumsum_fwd(
        dt_pre,
        A,
        args.chunk_size,
        cu_chunk_seqlens,
        dt_bias=dt_bias,
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
    )

    write_fp32(output_dir / "dt_chunk_fp32.bin", dt_chunk)
    write_fp32(output_dir / "dA_cumsum_fp32.bin", dA_cumsum)

    ref_dt_path = input_dir / "dt_chunk_fp32.bin"
    ref_dA_path = input_dir / "dA_cumsum_fp32.bin"
    if ref_dt_path.exists() and ref_dA_path.exists():
      ref_dt = load_fp32(ref_dt_path).reshape_as(dt_chunk.cpu())
      ref_dA = load_fp32(ref_dA_path).reshape_as(dA_cumsum.cpu())
      print("reference comparison:")
      print(f"  dt_chunk: mad={mad(dt_chunk.cpu(), ref_dt)} rl2={rel_l2(dt_chunk.cpu(), ref_dt)}")
      print(f"  dA_cumsum: mad={mad(dA_cumsum.cpu(), ref_dA)} rl2={rel_l2(dA_cumsum.cpu(), ref_dA)}")

    metadata = {
        "chunk_size": args.chunk_size,
        "dt_dtype": args.dt_dtype,
        "dt_softplus": True,
        "dt_limit": [0.0, "inf"],
        "token_count": int(k_t),
        "num_heads": k_h,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
