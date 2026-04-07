#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump a layer-derived Mamba decode-update oracle fixture.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer-index", type=int, default=0)
    return parser.parse_args()


def build_weight_map(index_path: Path) -> dict[str, str]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return payload["weight_map"]


def load_named_tensor(model_dir: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    shard = weight_map[name]
    with safe_open(str(model_dir / shard), framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def deterministic_pattern(shape: tuple[int, ...], scale: float, offset: int) -> torch.Tensor:
    total = 1
    for dim in shape:
        total *= dim
    values = torch.arange(offset, offset + total, dtype=torch.float32)
    values = ((values % 257) - 128.0) * scale
    return values.view(*shape)


def write_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.write_bytes(tensor.contiguous().to(torch.float32).numpy().tobytes())


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    weight_map = build_weight_map(model_dir / "model.safetensors.index.json")

    num_heads = int(config["mamba_num_heads"])
    head_dim = int(config["mamba_head_dim"])
    state_size = int(config["ssm_state_size"])
    n_groups = int(config["n_groups"])
    assert num_heads % n_groups == 0

    prefix = f"backbone.layers.{args.layer_index}.mixer"
    A_log = load_named_tensor(model_dir, weight_map, prefix + ".A_log").to(torch.float32)
    D = load_named_tensor(model_dir, weight_map, prefix + ".D").to(torch.float32)
    dt_bias = load_named_tensor(model_dir, weight_map, prefix + ".dt_bias").to(torch.float32)

    batch_size = 1
    ssm_state = deterministic_pattern((batch_size, num_heads, head_dim, state_size), 1.0e-3, 0)
    hidden = deterministic_pattern((batch_size, num_heads, head_dim), 3.0e-3, 17)
    dt_pre = deterministic_pattern((batch_size, num_heads), 1.0e-2, 31)
    B_grouped = deterministic_pattern((batch_size, n_groups, state_size), 2.0e-3, 47)
    C_grouped = deterministic_pattern((batch_size, n_groups, state_size), 2.5e-3, 79)

    B = B_grouped[..., None, :].expand(batch_size, n_groups, num_heads // n_groups, state_size).reshape(
        batch_size, num_heads, state_size
    )
    C = C_grouped[..., None, :].expand(batch_size, n_groups, num_heads // n_groups, state_size).reshape(
        batch_size, num_heads, state_size
    )

    dt = torch.nn.functional.softplus(dt_pre[..., None].expand(batch_size, num_heads, head_dim) + dt_bias[None, :, None])
    A = -torch.exp(A_log).view(1, num_heads, 1, 1).expand(batch_size, num_heads, head_dim, state_size)
    B_expanded = B[:, :, None, :].expand(batch_size, num_heads, head_dim, state_size)
    C_expanded = C
    D_expanded = D.view(1, num_heads, 1).expand(batch_size, num_heads, head_dim)

    dA = torch.exp(dt[..., None] * A)
    dB = dt[..., None] * B_expanded
    dBx = dB * hidden[..., None]
    next_state = ssm_state * dA + dBx
    y = torch.matmul(next_state.view(batch_size * num_heads, head_dim, state_size), C_expanded.view(batch_size * num_heads, state_size, 1))
    y = y.view(batch_size, num_heads, head_dim)
    output = y + hidden * D_expanded

    write_tensor(output_dir / "ssm_state_fp32.bin", ssm_state)
    write_tensor(output_dir / "hidden_fp32.bin", hidden)
    write_tensor(output_dir / "dt_fp32.bin", dt)
    write_tensor(output_dir / "A_fp32.bin", A)
    write_tensor(output_dir / "B_fp32.bin", B)
    write_tensor(output_dir / "C_fp32.bin", C)
    write_tensor(output_dir / "D_fp32.bin", D_expanded)
    write_tensor(output_dir / "expected_next_state_fp32.bin", next_state)
    write_tensor(output_dir / "expected_output_fp32.bin", output)

    metadata = {
        "fixture_kind": "mamba_decode_update",
        "layer_index": args.layer_index,
        "batch_size": batch_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "state_size": state_size,
        "n_groups": n_groups,
        "source_tensors": {
            "A_log": prefix + ".A_log",
            "D": prefix + ".D",
            "dt_bias": prefix + ".dt_bias",
        },
        "notes": [
            "This fixture uses real layer tensors A_log, D, and dt_bias from the checkpoint.",
            "The decode-step B, C, hidden, dt_pre, and initial ssm_state inputs are deterministic synthetic tensors with real model shapes.",
            "expected_output_fp32 is the pre-norm Mamba decode output y + hidden * D for one decode step.",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
