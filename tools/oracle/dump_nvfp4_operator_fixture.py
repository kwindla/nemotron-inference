#!/usr/bin/env python3

import argparse
import importlib
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers.activations import ACT2FN
from transformers import AutoTokenizer


FP4_POSITIVE_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_DECODE_TABLE = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]
FP4_MAX_FINITE = 6.0
MIN_SCALE = 1.0 / 1024.0
CONTAINER_IMAGE = "nemotron-local/dgx-spark-vllm:0.17.1-b31e9326a-fi065"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump a checkpoint-derived NVFP4 operator oracle fixture.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--prompt-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--layer-index", type=int, default=1)
    parser.add_argument("--expert-index", type=int, default=0)
    parser.add_argument(
        "--operator-kind",
        choices=["up_proj", "down_proj", "shared_down_proj"],
        default="up_proj",
    )
    return parser.parse_args()


def write_mamba_stub(root: Path) -> None:
    package_root = root / "mamba_ssm" / "ops" / "triton"
    package_root.mkdir(parents=True, exist_ok=True)
    for init_path in [
        root / "mamba_ssm" / "__init__.py",
        root / "mamba_ssm" / "ops" / "__init__.py",
        root / "mamba_ssm" / "ops" / "triton" / "__init__.py",
    ]:
        init_path.write_text("", encoding="utf-8")

    (package_root / "layernorm_gated.py").write_text(
        "import torch\n"
        "import torch.nn.functional as F\n"
        "\n"
        "def rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5, group_size=None, norm_before_gate=False):\n"
        "    y = x.to(torch.float32)\n"
        "    if group_size is None or group_size <= 0:\n"
        "        variance = y.pow(2).mean(dim=-1, keepdim=True)\n"
        "        y = y * torch.rsqrt(variance + eps)\n"
        "    else:\n"
        "        last = y.shape[-1]\n"
        "        if last % group_size != 0:\n"
        "            raise ValueError(f'group_size={group_size} must divide hidden dimension {last}')\n"
        "        grouped = y.view(*y.shape[:-1], last // group_size, group_size)\n"
        "        variance = grouped.pow(2).mean(dim=-1, keepdim=True)\n"
        "        grouped = grouped * torch.rsqrt(variance + eps)\n"
        "        y = grouped.view_as(y)\n"
        "    y = y * weight.to(torch.float32)\n"
        "    if bias is not None:\n"
        "        y = y + bias.to(torch.float32)\n"
        "    if z is not None:\n"
        "        y = y * F.silu(z.to(torch.float32))\n"
        "    return y.to(x.dtype)\n",
        encoding="utf-8",
    )
    (package_root / "selective_state_update.py").write_text(
        "def selective_state_update(*args, **kwargs):\n"
        "    raise RuntimeError('selective_state_update stub should not execute in oracle generation')\n",
        encoding="utf-8",
    )
    (package_root / "ssd_combined.py").write_text(
        "def mamba_chunk_scan_combined(*args, **kwargs):\n"
        "    raise RuntimeError('mamba_chunk_scan_combined stub should not execute in oracle generation')\n"
        "\n"
        "def mamba_split_conv1d_scan_combined(*args, **kwargs):\n"
        "    raise RuntimeError('mamba_split_conv1d_scan_combined stub should not execute in oracle generation')\n",
        encoding="utf-8",
    )


def load_local_modules(model_dir: Path) -> tuple[Any, Any]:
    with tempfile.TemporaryDirectory(prefix="nemotron_mamba_stub_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        write_mamba_stub(tmpdir)
        package_root = tmpdir / "oracle_model"
        package_root.symlink_to(model_dir, target_is_directory=True)
        sys.path.insert(0, str(tmpdir))
        try:
            config_module = importlib.import_module("oracle_model.configuration_nemotron_h")
            modeling_module = importlib.import_module("oracle_model.modeling_nemotron_h")
        finally:
            sys.path.pop(0)
    return config_module, modeling_module


def load_prompt_case(path: Path, prompt_name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for case in payload["cases"]:
        if case["name"] == prompt_name:
            return case
    raise KeyError(f"Prompt named {prompt_name!r} was not found in {path}")


def build_weight_map(index_path: Path) -> dict[str, str]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return payload["weight_map"]


def load_named_tensor(model_dir: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    shard = weight_map[name]
    with safe_open(str(model_dir / shard), framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def fp8_raw_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def clamp_scale(value: float) -> float:
    if not math.isfinite(value) or value < MIN_SCALE:
        return MIN_SCALE
    return value


def quantize_input_fp8(x: torch.Tensor, input_scale: float) -> torch.Tensor:
    scale = clamp_scale(float(input_scale))
    quantized = (x.to(torch.float32) / scale).to(torch.float8_e4m3fn)
    return quantized.to(torch.float32) * scale


class ScaledFp8Linear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, weight_scale: float, input_scale: float, bias: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("weight", weight.to(torch.float32) * float(weight_scale))
        self.register_buffer("bias", None if bias is None else bias.to(torch.float32))
        self.input_scale = float(input_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(quantize_input_fp8(x, self.input_scale), self.weight, self.bias)


def encode_fp4_e2m1(value: float) -> int:
    sign_bit = 0
    candidate = float(value)
    if math.copysign(1.0, candidate) < 0.0:
        sign_bit = 0x8
        candidate = -candidate
    best_index = 0
    best_error = float("inf")
    for index, representable in enumerate(FP4_POSITIVE_VALUES):
        error = abs(candidate - representable)
        if error < best_error:
            best_index = index
            best_error = error
    return sign_bit | best_index


def decode_fp4_e2m1(code: int) -> float:
    return FP4_DECODE_TABLE[code & 0x0F]


def pack_fp32_to_nvfp4_fixed(values: torch.Tensor, tensor_scale: float) -> dict[str, Any]:
    if values.dim() != 2 or values.shape[1] % 16 != 0:
        raise ValueError("values must be rank-2 with a column count divisible by 16")
    fixed_tensor_scale = clamp_scale(float(tensor_scale))
    rows, cols = values.shape
    packed = bytearray((rows * cols + 1) // 2)
    block_scales = bytearray(rows * (cols // 16))
    values_f32 = values.to(torch.float32)
    packed_index = 0
    scale_index = 0
    for row in range(rows):
        for block in range(cols // 16):
            start = block * 16
            chunk = values_f32[row, start : start + 16]
            block_max = float(chunk.abs().max().item())
            block_scale = 1.0
            if block_max > 0.0:
                block_scale = clamp_scale(block_max / (FP4_MAX_FINITE * fixed_tensor_scale))
            block_scale_tensor = torch.tensor([block_scale], dtype=torch.float32).to(torch.float8_e4m3fn)
            block_scales[scale_index] = int(block_scale_tensor.view(torch.uint8).item())
            scale_index += 1
            full_scale = fixed_tensor_scale * block_scale
            for offset in range(0, 16, 2):
                lhs = float(chunk[offset].item()) / full_scale
                rhs = float(chunk[offset + 1].item()) / full_scale
                packed[packed_index] = encode_fp4_e2m1(lhs) | (encode_fp4_e2m1(rhs) << 4)
                packed_index += 1
    return {
        "packed": bytes(packed),
        "block_scales": bytes(block_scales),
        "tensor_scale": fixed_tensor_scale,
    }


def dequantize_nvfp4_matrix(
    packed: bytes,
    block_scales: bytes,
    tensor_scale: float,
    rows: int,
    cols: int,
) -> torch.Tensor:
    if cols % 16 != 0:
        raise ValueError("cols must be divisible by 16")
    scales_tensor = torch.tensor(list(block_scales), dtype=torch.uint8).view(rows, cols // 16)
    block_scale_values = scales_tensor.view(torch.float8_e4m3fn).to(torch.float32)
    packed_tensor = torch.tensor(list(packed), dtype=torch.uint8)
    output = torch.zeros((rows, cols), dtype=torch.float32)
    packed_index = 0
    for row in range(rows):
        for block in range(cols // 16):
            block_scale = float(block_scale_values[row, block].item()) * float(tensor_scale)
            start = block * 16
            for offset in range(0, 16, 2):
                byte = int(packed_tensor[packed_index].item())
                output[row, start + offset] = decode_fp4_e2m1(byte & 0x0F) * block_scale
                output[row, start + offset + 1] = decode_fp4_e2m1((byte >> 4) & 0x0F) * block_scale
                packed_index += 1
    return output


def load_nvfp4_linear_metadata(model_dir: Path, weight_map: dict[str, str], prefix: str) -> dict[str, Any]:
    weight = load_named_tensor(model_dir, weight_map, prefix + ".weight")
    weight_scales = load_named_tensor(model_dir, weight_map, prefix + ".weight_scale")
    return {
        "prefix": prefix,
        "weight": weight,
        "input_scale": float(load_named_tensor(model_dir, weight_map, prefix + ".input_scale").item()),
        "weight_scales": weight_scales,
        "weight_scale_2": float(load_named_tensor(model_dir, weight_map, prefix + ".weight_scale_2").item()),
        "input_cols": int(weight_scales.shape[1] * 16),
    }


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    prompts_path = Path(args.prompts)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_module, modeling_module = load_local_modules(model_dir)
    NemotronHConfig = config_module.NemotronHConfig
    NemotronHBlock = modeling_module.NemotronHBlock
    NemotronHRMSNorm = modeling_module.NemotronHRMSNorm

    config = NemotronHConfig(**json.loads((model_dir / "config.json").read_text(encoding="utf-8")))
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=True,
    )
    prompt_case = load_prompt_case(prompts_path, args.prompt_name)
    token_ids = tokenizer.apply_chat_template(
        prompt_case["messages"],
        tokenize=True,
        add_generation_prompt=prompt_case.get("add_generation_prompt", True),
    )
    input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)

    weight_map = build_weight_map(model_dir / "model.safetensors.index.json")

    embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size)
    embeddings.weight.data.copy_(load_named_tensor(model_dir, weight_map, "backbone.embeddings.weight").to(torch.float32))

    layer0 = NemotronHBlock(config, layer_idx=0).eval()
    layer0.norm.weight.data.copy_(load_named_tensor(model_dir, weight_map, "backbone.layers.0.norm.weight").to(torch.float32))
    layer0.mixer.A_log.data.copy_(load_named_tensor(model_dir, weight_map, "backbone.layers.0.mixer.A_log").to(torch.float32))
    layer0.mixer.D.data.copy_(load_named_tensor(model_dir, weight_map, "backbone.layers.0.mixer.D").to(torch.float32))
    layer0.mixer.dt_bias.data.copy_(load_named_tensor(model_dir, weight_map, "backbone.layers.0.mixer.dt_bias").to(torch.float32))
    layer0.mixer.conv1d.weight.data.copy_(
        load_named_tensor(model_dir, weight_map, "backbone.layers.0.mixer.conv1d.weight").to(torch.float32)
    )
    layer0.mixer.conv1d.bias.data.copy_(
        load_named_tensor(model_dir, weight_map, "backbone.layers.0.mixer.conv1d.bias").to(torch.float32)
    )
    layer0.mixer.norm.weight.data.copy_(
        load_named_tensor(model_dir, weight_map, "backbone.layers.0.mixer.norm.weight").to(torch.float32)
    )
    layer0.mixer.in_proj = ScaledFp8Linear(
        load_named_tensor(model_dir, weight_map, "backbone.layers.0.mixer.in_proj.weight"),
        float(load_named_tensor(model_dir, weight_map, "backbone.layers.0.mixer.in_proj.weight_scale").item()),
        float(load_named_tensor(model_dir, weight_map, "backbone.layers.0.mixer.in_proj.input_scale").item()),
    )
    layer0.mixer.out_proj = ScaledFp8Linear(
        load_named_tensor(model_dir, weight_map, "backbone.layers.0.mixer.out_proj.weight"),
        float(load_named_tensor(model_dir, weight_map, "backbone.layers.0.mixer.out_proj.weight_scale").item()),
        float(load_named_tensor(model_dir, weight_map, "backbone.layers.0.mixer.out_proj.input_scale").item()),
    )

    layer1_norm = NemotronHRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon).eval()
    layer1_norm.weight.data.copy_(load_named_tensor(model_dir, weight_map, "backbone.layers.1.norm.weight").to(torch.float32))

    fc1_latent = ScaledFp8Linear(
        load_named_tensor(model_dir, weight_map, "backbone.layers.1.mixer.fc1_latent_proj.weight"),
        float(load_named_tensor(model_dir, weight_map, "backbone.layers.1.mixer.fc1_latent_proj.weight_scale").item()),
        float(load_named_tensor(model_dir, weight_map, "backbone.layers.1.mixer.fc1_latent_proj.input_scale").item()),
    ).eval()

    expert_base_prefix = f"backbone.layers.{args.layer_index}.mixer.experts.{args.expert_index}"
    shared_base_prefix = f"backbone.layers.{args.layer_index}.mixer.shared_experts"
    up_proj = load_nvfp4_linear_metadata(model_dir, weight_map, expert_base_prefix + ".up_proj")
    down_proj = load_nvfp4_linear_metadata(model_dir, weight_map, expert_base_prefix + ".down_proj")
    shared_down_proj = load_nvfp4_linear_metadata(model_dir, weight_map, shared_base_prefix + ".down_proj")
    shared_up_proj = ScaledFp8Linear(
        load_named_tensor(model_dir, weight_map, shared_base_prefix + ".up_proj.weight"),
        float(load_named_tensor(model_dir, weight_map, shared_base_prefix + ".up_proj.weight_scale").item()),
        float(load_named_tensor(model_dir, weight_map, shared_base_prefix + ".up_proj.input_scale").item()),
    ).eval()

    with torch.no_grad():
        hidden_states = embeddings(input_ids).to(torch.float32)
        hidden_states = layer0(hidden_states)
        hidden_states = layer1_norm(hidden_states)
        latent_states = fc1_latent(hidden_states).reshape(-1, config.moe_latent_size).to(torch.float32)
        shared_input_states = hidden_states.reshape(-1, config.hidden_size).to(torch.float32)

    source_states = latent_states if args.operator_kind in ["up_proj", "down_proj"] else shared_input_states

    row_count = min(args.rows, source_states.shape[0])
    if row_count <= 0:
        raise ValueError("prompt did not produce any activations")
    source_activations = source_states[:row_count].contiguous()

    if args.operator_kind == "up_proj":
        target_operator = up_proj
        activations = source_activations
        activation_path = "latent_states"
    elif args.operator_kind == "down_proj":
        up_activation_pack = pack_fp32_to_nvfp4_fixed(source_activations, up_proj["input_scale"])
        up_activation_dequant = dequantize_nvfp4_matrix(
            up_activation_pack["packed"],
            up_activation_pack["block_scales"],
            up_activation_pack["tensor_scale"],
            row_count,
            source_activations.shape[1],
        )
        up_weight_dequant = dequantize_nvfp4_matrix(
            up_proj["weight"].contiguous().numpy().tobytes(),
            fp8_raw_bytes(up_proj["weight_scales"]),
            up_proj["weight_scale_2"],
            up_proj["weight"].shape[0],
            up_proj["input_cols"],
        )
        activations = ACT2FN[config.mlp_hidden_act](up_activation_dequant @ up_weight_dequant.transpose(0, 1)).to(
            torch.float32
        )
        target_operator = down_proj
        activation_path = f"{up_proj['prefix']} -> {config.mlp_hidden_act}"
    else:
        activations = ACT2FN[config.mlp_hidden_act](shared_up_proj(source_activations).to(torch.float32))
        target_operator = shared_down_proj
        activation_path = f"{shared_base_prefix}.up_proj -> {config.mlp_hidden_act}"

    activation_pack = pack_fp32_to_nvfp4_fixed(activations, target_operator["input_scale"])
    activation_dequant = dequantize_nvfp4_matrix(
        activation_pack["packed"],
        activation_pack["block_scales"],
        activation_pack["tensor_scale"],
        row_count,
        activations.shape[1],
    )

    weight_packed_bytes = target_operator["weight"].contiguous().numpy().tobytes()
    weight_block_scale_bytes = fp8_raw_bytes(target_operator["weight_scales"])
    weight_dequant = dequantize_nvfp4_matrix(
        weight_packed_bytes,
        weight_block_scale_bytes,
        target_operator["weight_scale_2"],
        target_operator["weight"].shape[0],
        target_operator["input_cols"],
    )

    expected_output = activation_dequant @ weight_dequant.transpose(0, 1)

    (output_dir / "activations_fp32.bin").write_bytes(activations.contiguous().numpy().astype("float32").tobytes())
    (output_dir / "activation_tensor_scale.bin").write_bytes(
        torch.tensor([activation_pack["tensor_scale"]], dtype=torch.float32).numpy().tobytes()
    )
    (output_dir / "weight_packed.bin").write_bytes(weight_packed_bytes)
    (output_dir / "weight_block_scales.bin").write_bytes(weight_block_scale_bytes)
    (output_dir / "weight_tensor_scale.bin").write_bytes(
        torch.tensor([target_operator["weight_scale_2"]], dtype=torch.float32).numpy().tobytes()
    )
    (output_dir / "expected_output_fp32.bin").write_bytes(
        expected_output.contiguous().numpy().astype("float32").tobytes()
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "fixture_kind": "nvfp4_operator_oracle_v1",
                "generator": "dump_nvfp4_operator_fixture.py",
                "container_image": CONTAINER_IMAGE,
                "model_dir": str(model_dir),
                "prompt_name": args.prompt_name,
                "operator_kind": args.operator_kind,
                "target_operator": target_operator["prefix"],
                "activation_path": activation_path,
                "layer0_path": "checkpoint-derived partial CPU reconstruction with stubbed mamba_ssm RMSNorm and dequantized FP8 linears",
                "rows": row_count,
                "input_cols": int(activations.shape[1]),
                "output_rows": int(target_operator["weight"].shape[0]),
                "token_count": len(token_ids),
                "tokens": token_ids,
                "activation_tensor_scale": activation_pack["tensor_scale"],
                "weight_tensor_scale": target_operator["weight_scale_2"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
