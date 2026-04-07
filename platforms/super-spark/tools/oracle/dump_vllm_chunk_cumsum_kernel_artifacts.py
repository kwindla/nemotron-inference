#!/usr/bin/env python3
"""Dump the exact compiled Triton artifacts for pinned vLLM _chunk_cumsum_fwd."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from vllm.model_executor.layers.mamba.ops.ssd_chunk_state import (
    _chunk_cumsum_fwd,
    _chunk_cumsum_fwd_kernel,
)
from vllm.triton_utils import triton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay pinned vLLM _chunk_cumsum_fwd on saved inputs, then dump the "
            "compiled Triton artifacts for the exact best autotuned specialization."
        )
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument(
        "--dt-dtype",
        choices=("fp32", "bf16"),
        default="fp32",
        help="Load dt_pre as fp32 or cast it to bf16 before replay.",
    )
    return parser.parse_args()


def load_fp32(path: Path) -> torch.Tensor:
    data = torch.frombuffer(path.read_bytes(), dtype=torch.float32)
    return data.clone()


def serialize_json(value: Any) -> Any:
    if hasattr(value, "_asdict"):
        return serialize_json(value._asdict())
    if isinstance(value, dict):
        return {str(k): serialize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__") and not isinstance(value, torch.Tensor):
        return serialize_json(vars(value))
    return value


def write_artifact(path: Path, value: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding="utf-8")
    print(f"dump: {path}")


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
    torch.cuda.synchronize()

    best_config = getattr(_chunk_cumsum_fwd_kernel, "best_config", None)
    if best_config is None:
        raise RuntimeError("Autotuner did not record best_config after replay")

    nchunks = cu_chunk_seqlens.numel() - 1
    grid = (
        int(nchunks),
        int(triton.cdiv(k_h, best_config.kwargs["BLOCK_SIZE_H"])),
        1,
    )
    compiled = _chunk_cumsum_fwd_kernel.fn.warmup(
        dt_ptr=dt_pre,
        A_ptr=A,
        dt_bias_ptr=dt_bias,
        dt_out_ptr=dt_chunk,
        dA_cumsum_ptr=dA_cumsum,
        cu_chunk_seqlens_ptr=cu_chunk_seqlens,
        seqlen=k_t,
        nheads=k_h,
        chunk_size=args.chunk_size,
        dt_min=0.0,
        dt_max=float("inf"),
        stride_dt_seqlen=dt_pre.stride(0),
        stride_dt_head=dt_pre.stride(1),
        stride_A_head=A.stride(0),
        stride_dt_bias_head=dt_bias.stride(0),
        stride_dt_out_head=dt_chunk.stride(0),
        stride_dt_out_chunk=dt_chunk.stride(1),
        stride_dt_out_csize=dt_chunk.stride(2),
        stride_dA_cs_head=dA_cumsum.stride(0),
        stride_dA_cs_chunk=dA_cumsum.stride(1),
        stride_dA_cs_csize=dA_cumsum.stride(2),
        DT_SOFTPLUS=True,
        HAS_DT_BIAS=True,
        BLOCK_SIZE_CHUNK=triton.next_power_of_2(args.chunk_size),
        grid=grid,
        **best_config.all_kwargs(),
    )
    if compiled is None:
        raise RuntimeError("Failed to compile _chunk_cumsum_fwd_kernel specialization")

    metadata = {
        "chunk_size": args.chunk_size,
        "dt_dtype": args.dt_dtype,
        "token_count": int(k_t),
        "num_heads": k_h,
        "grid": list(grid),
        "best_config": serialize_json(best_config.__dict__),
        "compiled_metadata": serialize_json(compiled.metadata),
        "metadata_group": serialize_json(compiled.metadata_group),
    }
    write_artifact(
        output_dir / "metadata.json",
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    )

    suffix_map = {
        "ttir": "ttir.mlir",
        "ttgir": "ttgir.mlir",
        "llir": "llir.ll",
        "ptx": "ptx.ptx",
        "cubin": "cubin.bin",
    }
    for key, artifact in compiled.asm.items():
        file_name = suffix_map.get(key, f"{key}.txt")
        write_artifact(output_dir / file_name, artifact)


if __name__ == "__main__":
    main()
