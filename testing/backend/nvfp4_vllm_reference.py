#!/usr/bin/env python3

import argparse
import sys
import types
from pathlib import Path

import torch


SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT / "third_party" / "vllm"))

from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (  # noqa: E402
    NvFp4MoeBackend,
    prepare_nvfp4_moe_layer_for_fi_or_cutlass,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (  # noqa: E402
    swizzle_blockscale,
)


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _uint8_tensor(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    raw = torch.frombuffer(bytearray(_read_bytes(path)), dtype=torch.uint8)
    return raw.reshape(shape).to("cuda")


def _fp8_tensor(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    return _uint8_tensor(path, shape).view(torch.float8_e4m3fn)


def _fp32_tensor(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    raw = torch.frombuffer(bytearray(_read_bytes(path)), dtype=torch.float32)
    return raw.reshape(shape).to("cuda")


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()


def run_swizzle(args: argparse.Namespace) -> int:
    input_shape = (args.rows, args.cols // 16)
    scale = _fp8_tensor(Path(args.input), input_shape)
    swizzled = swizzle_blockscale(scale)
    _write_bytes(Path(args.output), _tensor_bytes(swizzled))
    return 0


def run_prepare_moe(args: argparse.Namespace) -> int:
    experts = args.num_experts
    up_rows = args.up_rows
    up_cols = args.up_cols
    down_rows = args.down_rows
    down_cols = args.down_cols

    w13 = _uint8_tensor(Path(args.up_packed), (experts, up_rows, up_cols // 2))
    w13_scale = _fp8_tensor(Path(args.up_scales), (experts, up_rows, up_cols // 16))
    w13_global_scale = _fp32_tensor(Path(args.up_tensor_scales), (experts,))
    w2 = _uint8_tensor(Path(args.down_packed), (experts, down_rows, down_cols // 2))
    w2_scale = _fp8_tensor(Path(args.down_scales), (experts, down_rows, down_cols // 16))
    w2_global_scale = _fp32_tensor(Path(args.down_tensor_scales), (experts,))

    layer = types.SimpleNamespace(
        activation=types.SimpleNamespace(is_gated=False),
        moe_config=types.SimpleNamespace(intermediate_size_per_partition=up_rows),
    )

    (
        ref_w13,
        ref_w13_scale,
        ref_w13_scale_2,
        _ref_a13_scale,
        ref_w2,
        ref_w2_scale,
        ref_w2_scale_2,
        _ref_a2_scale,
    ) = prepare_nvfp4_moe_layer_for_fi_or_cutlass(
        backend=NvFp4MoeBackend.VLLM_CUTLASS,
        layer=layer,
        w13=w13,
        w13_scale=w13_scale,
        w13_scale_2=(1.0 / w13_global_scale),
        a13_scale=torch.ones((experts, 1), dtype=torch.float32, device="cuda"),
        w2=w2,
        w2_scale=w2_scale,
        w2_scale_2=(1.0 / w2_global_scale),
        a2_scale=torch.ones((experts, 1), dtype=torch.float32, device="cuda"),
        is_act_and_mul=False,
    )

    output_dir = Path(args.output_dir)
    _write_bytes(output_dir / "ref_w13_packed.bin", _tensor_bytes(ref_w13))
    _write_bytes(output_dir / "ref_w13_scales.bin", _tensor_bytes(ref_w13_scale))
    _write_bytes(
        output_dir / "ref_w13_tensor_scales.bin",
        (1.0 / ref_w13_scale_2).contiguous().cpu().numpy().tobytes(),
    )
    _write_bytes(output_dir / "ref_w2_packed.bin", _tensor_bytes(ref_w2))
    _write_bytes(output_dir / "ref_w2_scales.bin", _tensor_bytes(ref_w2_scale))
    _write_bytes(
        output_dir / "ref_w2_tensor_scales.bin",
        (1.0 / ref_w2_scale_2).contiguous().cpu().numpy().tobytes(),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    swizzle = subparsers.add_parser("swizzle")
    swizzle.add_argument("--rows", type=int, required=True)
    swizzle.add_argument("--cols", type=int, required=True)
    swizzle.add_argument("--input", required=True)
    swizzle.add_argument("--output", required=True)
    swizzle.set_defaults(func=run_swizzle)

    prepare = subparsers.add_parser("prepare_moe")
    prepare.add_argument("--num-experts", type=int, required=True)
    prepare.add_argument("--up-rows", type=int, required=True)
    prepare.add_argument("--up-cols", type=int, required=True)
    prepare.add_argument("--down-rows", type=int, required=True)
    prepare.add_argument("--down-cols", type=int, required=True)
    prepare.add_argument("--up-packed", required=True)
    prepare.add_argument("--up-scales", required=True)
    prepare.add_argument("--up-tensor-scales", required=True)
    prepare.add_argument("--down-packed", required=True)
    prepare.add_argument("--down-scales", required=True)
    prepare.add_argument("--down-tensor-scales", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.set_defaults(func=run_prepare_moe)
    return parser


def main() -> int:
    if not torch.cuda.is_available():
        print("nvfp4_vllm_reference: CUDA is required", file=sys.stderr)
        return 1
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
