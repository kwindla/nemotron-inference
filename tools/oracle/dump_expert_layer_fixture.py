#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open


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
FP4_POSITIVE_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_MAX_FINITE = 6.0
FP8_E4M3_MAX_FINITE = 448.0
MIN_SCALE = 1.0 / 1024.0
FP4_DECODE_TABLE_TENSOR = torch.tensor(FP4_DECODE_TABLE, dtype=torch.float32)
FP4_POSITIVE_VALUES_TENSOR = torch.tensor(FP4_POSITIVE_VALUES, dtype=torch.float32)


def configure_torch_precision() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump a checkpoint-derived expert-layer oracle fixture.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer-index", type=int, default=1)
    parser.add_argument("--input-hidden-bin", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def build_weight_map(index_path: Path) -> dict[str, str]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return payload["weight_map"]


def load_named_tensor(
    model_dir: Path,
    weight_map: dict[str, str],
    name: str,
    device: torch.device | None = None,
) -> torch.Tensor:
    shard = weight_map[name]
    with safe_open(str(model_dir / shard), framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(name)
    if device is not None:
        return tensor.to(device=device)
    return tensor


def deterministic_pattern(shape: tuple[int, ...], scale: float, offset: int) -> torch.Tensor:
    total = 1
    for dim in shape:
        total *= dim
    values = torch.arange(offset, offset + total, dtype=torch.float32)
    values = ((values % 257) - 128.0) * scale
    return values.view(*shape)


def load_float32_matrix(path: Path, cols: int) -> torch.Tensor:
    data = path.read_bytes()
    if len(data) % 4 != 0:
        raise ValueError(f"{path} is not a float32 buffer")
    values = torch.frombuffer(bytearray(data), dtype=torch.float32).clone()
    if values.numel() % cols != 0:
        raise ValueError(f"{path} does not divide evenly into rows of width {cols}")
    return values.view(-1, cols)


def write_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.write_bytes(tensor.contiguous().to(torch.float32).cpu().numpy().tobytes())


def write_raw_uint8(path: Path, tensor: torch.Tensor) -> None:
    path.write_bytes(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())


def clamp_scale(value: float) -> float:
    if not math.isfinite(value) or value < MIN_SCALE:
        return MIN_SCALE
    return value


def rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    return hidden_states * torch.rsqrt(variance + epsilon) * weight.to(torch.float32)


def scaled_fp8_linear(
    activations: torch.Tensor,
    weight_quantized: torch.Tensor,
    weight_scale: float,
    input_scale: float,
) -> torch.Tensor:
    quantized_input = (activations.to(torch.float32) / input_scale).to(torch.float8_e4m3fn)
    round_tripped_input = quantized_input.to(torch.float32) * input_scale
    dequantized_weight = weight_quantized.to(torch.float32) * weight_scale
    return torch.nn.functional.linear(round_tripped_input, dequantized_weight)


def scaled_fp8_roundtrip_input(activations: torch.Tensor, input_scale: float) -> torch.Tensor:
    quantized_input = (activations.to(torch.float32) / input_scale).to(torch.float8_e4m3fn)
    return quantized_input.to(torch.float32) * input_scale


def scaled_fp8_dequantized_weight(weight_quantized: torch.Tensor, weight_scale: float) -> torch.Tensor:
    return weight_quantized.to(torch.float32) * weight_scale


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


def _as_uint8_tensor(data: bytes | torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data.to(dtype=torch.uint8)
        return tensor.to(device=device) if device is not None else tensor
    return torch.tensor(list(data), dtype=torch.uint8, device=device)


def _as_scalar_tensor(value: float | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.tensor(float(value), dtype=torch.float32, device=device)


def pack_fp32_to_nvfp4_dynamic(values: torch.Tensor) -> dict[str, object]:
    if values.dim() != 2 or values.shape[1] % 16 != 0:
        raise ValueError("values must be rank-2 with a column count divisible by 16")
    rows, cols = values.shape
    device = values.device
    blocks = cols // 16
    values_f32 = values.to(torch.float32)
    global_max_abs = float(values_f32.abs().max().item())
    if global_max_abs > FP4_MAX_FINITE * FP8_E4M3_MAX_FINITE:
        tensor_scale = clamp_scale(global_max_abs / (FP4_MAX_FINITE * FP8_E4M3_MAX_FINITE))
    else:
        tensor_scale = 1.0

    chunks = values_f32.view(rows, blocks, 16)
    block_max = chunks.abs().amax(dim=-1)
    block_scale = torch.where(
        block_max > 0.0,
        block_max / (FP4_MAX_FINITE * tensor_scale),
        torch.ones_like(block_max),
    )
    block_scale = torch.clamp(block_scale, min=MIN_SCALE)
    full_scale = tensor_scale * block_scale.unsqueeze(-1)
    normalized = chunks / full_scale

    distances = torch.abs(
        normalized.abs().unsqueeze(-1) -
        FP4_POSITIVE_VALUES_TENSOR.to(device=device).view(1, 1, 1, -1)
    )
    encoded_magnitude = distances.argmin(dim=-1).to(torch.uint8)
    sign_bit = (normalized < 0).to(torch.uint8) * 0x8
    encoded = encoded_magnitude | sign_bit
    packed = (encoded[..., 0::2] | (encoded[..., 1::2] << 4)).contiguous().view(-1)
    block_scales = (
        block_scale.to(torch.float8_e4m3fn)
        .contiguous()
        .view(torch.uint8)
        .view(-1)
    )
    return {
        "packed": packed,
        "block_scales": block_scales,
        "tensor_scale": torch.tensor(float(tensor_scale), dtype=torch.float32, device=device),
        "rows": rows,
        "cols": cols,
    }


def dequantize_nvfp4_matrix(
    packed: bytes | torch.Tensor,
    block_scales: bytes | torch.Tensor,
    tensor_scale: float | torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    if cols % 16 != 0:
        raise ValueError("cols must be divisible by 16")
    blocks = cols // 16
    packed_tensor = _as_uint8_tensor(packed)
    block_scale_tensor = _as_uint8_tensor(block_scales, device=packed_tensor.device)
    decode_table = FP4_DECODE_TABLE_TENSOR.to(device=packed_tensor.device)
    tensor_scale_tensor = _as_scalar_tensor(tensor_scale, packed_tensor.device)
    packed_tensor = packed_tensor.view(rows, blocks, 8)
    decoded_low = decode_table[(packed_tensor & 0x0F).to(torch.long)]
    decoded_high = decode_table[((packed_tensor >> 4) & 0x0F).to(torch.long)]
    decoded = torch.stack((decoded_low, decoded_high), dim=-1).view(rows, blocks, 16)
    block_scale_values = block_scale_tensor.view(rows, blocks).view(torch.float8_e4m3fn).to(torch.float32)
    return (decoded * (block_scale_values * tensor_scale_tensor).unsqueeze(-1)).view(rows, cols)


def load_nvfp4_linear_metadata(model_dir: Path, weight_map: dict[str, str], prefix: str) -> dict[str, object]:
    weight = load_named_tensor(model_dir, weight_map, prefix + ".weight")
    weight_scales = load_named_tensor(model_dir, weight_map, prefix + ".weight_scale")
    tensor_scale_name = prefix + ".weight_scale_2"
    if tensor_scale_name not in weight_map:
        tensor_scale_name = prefix + ".input_scale"
    return {
        "family": "nvfp4",
        "prefix": prefix,
        "weight": weight.contiguous().view(torch.uint8).flatten().cpu(),
        "weight_scales": weight_scales.contiguous().view(torch.uint8).flatten().cpu(),
        "weight_scale_2": float(load_named_tensor(model_dir, weight_map, tensor_scale_name).item()),
        "input_scale": float(load_named_tensor(model_dir, weight_map, prefix + ".input_scale").item()),
        "input_cols": int(weight_scales.shape[1] * 16),
        "rows": int(weight.shape[0]),
        "cols": int(weight_scales.shape[1] * 16),
        "tensor_scale_name": tensor_scale_name,
    }


def load_dense_or_scaled_fp8_linear_metadata(
    model_dir: Path,
    weight_map: dict[str, str],
    prefix: str,
    device: torch.device | None = None,
) -> dict[str, object]:
    weight = load_named_tensor(model_dir, weight_map, prefix + ".weight", device=device)
    weight_scale_name = prefix + ".weight_scale"
    input_scale_name = prefix + ".input_scale"
    if weight_scale_name in weight_map and input_scale_name in weight_map:
        return {
            "family": "scaled_fp8",
            "prefix": prefix,
            "weight": weight,
            "weight_scale": float(load_named_tensor(model_dir, weight_map, weight_scale_name, device=device).item()),
            "input_scale": float(load_named_tensor(model_dir, weight_map, input_scale_name, device=device).item()),
        }
    return {
        "family": "dense",
        "prefix": prefix,
        "weight": weight.to(torch.float32),
    }


def load_shared_down_metadata(
    model_dir: Path,
    weight_map: dict[str, str],
    prefix: str,
    device: torch.device | None = None,
) -> dict[str, object]:
    if prefix + ".weight_scale_2" in weight_map:
        return load_nvfp4_linear_metadata(model_dir, weight_map, prefix)
    return load_dense_or_scaled_fp8_linear_metadata(model_dir, weight_map, prefix, device=device)


def run_dense_or_scaled_fp8_linear(activations: torch.Tensor, linear_metadata: dict[str, object]) -> torch.Tensor:
    if linear_metadata["family"] == "scaled_fp8":
        return scaled_fp8_linear(
            activations,
            linear_metadata["weight"],
            linear_metadata["weight_scale"],
            linear_metadata["input_scale"],
        ).to(torch.float32)
    if linear_metadata["family"] == "dense":
        return F.linear(activations.to(torch.float32), linear_metadata["weight"].to(torch.float32))
    raise ValueError(f"unsupported linear family {linear_metadata['family']!r}")


def compute_router_selections(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.sigmoid(router_logits.to(torch.float32))
    scores_for_choice = scores + correction_bias.to(torch.float32)
    experts_per_group = scores_for_choice.shape[-1] // n_group
    group_scores = (
        scores_for_choice.view(-1, n_group, experts_per_group)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, n_group, experts_per_group)
        .reshape(-1, scores_for_choice.shape[-1])
    )
    masked_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
    topk_indices = torch.topk(masked_scores, k=top_k, dim=-1, sorted=False)[1]
    topk_weights = scores.gather(1, topk_indices)
    if norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights / denominator
    topk_weights = topk_weights * routed_scaling_factor
    return topk_indices.to(torch.int32), topk_weights.to(torch.float32)


def nvfp4_linear(activations: torch.Tensor, linear_metadata: dict[str, object]) -> torch.Tensor:
    packed_activation = pack_fp32_to_nvfp4_dynamic(activations.to(torch.float32))
    activation_dequant = dequantize_nvfp4_matrix(
        packed_activation["packed"],
        packed_activation["block_scales"],
        packed_activation["tensor_scale"],
        packed_activation["rows"],
        packed_activation["cols"],
    )
    device_key = str(activations.device)
    device_cache = linear_metadata.setdefault("dequantized_weight_by_device", {})
    weight_dequant = device_cache.get(device_key)
    if weight_dequant is None:
        host_weight = linear_metadata.get("dequantized_weight_host")
        if host_weight is None:
            host_weight = dequantize_nvfp4_matrix(
                linear_metadata["weight"],
                linear_metadata["weight_scales"],
                float(linear_metadata["weight_scale_2"]),
                int(linear_metadata["rows"]),
                int(linear_metadata["cols"]),
            )
            linear_metadata["dequantized_weight_host"] = host_weight
        weight_dequant = host_weight.to(device=activations.device)
        device_cache[device_key] = weight_dequant
    return activation_dequant.to(device=activations.device) @ weight_dequant.transpose(0, 1)


def relu2(x: torch.Tensor) -> torch.Tensor:
    return torch.square(torch.relu(x))


def main() -> int:
    configure_torch_precision()
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_name = args.device
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA oracle device, but torch.cuda.is_available() is false")

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    weight_map = build_weight_map(model_dir / "model.safetensors.index.json")
    prefix = f"backbone.layers.{args.layer_index}"
    mixer_prefix = prefix + ".mixer"

    hidden_size = int(config["hidden_size"])
    moe_latent_size = int(config["moe_latent_size"])
    moe_intermediate_size = int(config["moe_intermediate_size"])
    shared_intermediate_size = int(config["moe_shared_expert_intermediate_size"])
    n_routed_experts = int(config["n_routed_experts"])
    top_k = int(config["num_experts_per_tok"])
    n_group = int(config["n_group"])
    topk_group = int(config["topk_group"])
    routed_scaling_factor = float(config["routed_scaling_factor"])
    norm_topk_prob = bool(config["norm_topk_prob"])
    layer_norm_eps = float(config["layer_norm_epsilon"])

    if args.input_hidden_bin is not None:
        input_hidden = load_float32_matrix(Path(args.input_hidden_bin), hidden_size).to(device=device, dtype=torch.float32)
    else:
        input_hidden = deterministic_pattern((1, hidden_size), 2.0e-3, 211).to(device=device)
    norm_weight = load_named_tensor(model_dir, weight_map, prefix + ".norm.weight", device=device).to(torch.float32)
    gate_weight = load_named_tensor(model_dir, weight_map, mixer_prefix + ".gate.weight", device=device).to(torch.float32)
    gate_score_correction_bias = load_named_tensor(
        model_dir, weight_map, mixer_prefix + ".gate.e_score_correction_bias", device=device
    ).to(torch.float32)

    fc1_latent = load_dense_or_scaled_fp8_linear_metadata(
        model_dir, weight_map, mixer_prefix + ".fc1_latent_proj", device=device
    )
    fc2_latent_weight = load_named_tensor(model_dir, weight_map, mixer_prefix + ".fc2_latent_proj.weight", device=device).to(torch.float32)

    shared_up = load_dense_or_scaled_fp8_linear_metadata(
        model_dir, weight_map, mixer_prefix + ".shared_experts.up_proj", device=device
    )
    shared_down = load_shared_down_metadata(model_dir, weight_map, mixer_prefix + ".shared_experts.down_proj", device=device)

    norm_output = rms_norm(input_hidden, norm_weight, layer_norm_eps)
    router_logits = F.linear(norm_output.to(torch.float32), gate_weight.to(torch.float32))
    selected_indices, selected_weights = compute_router_selections(
        router_logits,
        gate_score_correction_bias,
        top_k,
        n_group,
        topk_group,
        routed_scaling_factor,
        norm_topk_prob,
    )

    latent_states = run_dense_or_scaled_fp8_linear(norm_output, fc1_latent).to(torch.float32)

    routed_output = torch.zeros((input_hidden.shape[0], moe_latent_size), dtype=torch.float32, device=device)
    selected_expert_ids_per_row: list[list[int]] = []
    selected_weights_rows: list[torch.Tensor] = []
    routed_activated_hidden_per_row: list[list[torch.Tensor]] = []
    routed_expert_outputs_per_row: list[list[torch.Tensor]] = []
    routed_weighted_contributions_per_row: list[list[torch.Tensor]] = []
    cached_experts: dict[int, tuple[dict[str, object], dict[str, object]]] = {}
    for token_index in range(input_hidden.shape[0]):
        selected_expert_ids = [int(value.item()) for value in selected_indices[token_index]]
        selected_expert_ids_per_row.append(selected_expert_ids)
        selected_weights_rows.append(selected_weights[token_index].detach().clone())
        row_activated_hidden: list[torch.Tensor] = []
        row_expert_outputs: list[torch.Tensor] = []
        row_contributions: list[torch.Tensor] = []
        latent_row = latent_states[token_index : token_index + 1]
        routed_row = torch.zeros((1, moe_latent_size), dtype=torch.float32, device=device)
        for slot, expert_index in enumerate(selected_expert_ids):
            if expert_index not in cached_experts:
                expert_prefix = mixer_prefix + f".experts.{expert_index}"
                cached_experts[expert_index] = (
                    load_nvfp4_linear_metadata(model_dir, weight_map, expert_prefix + ".up_proj"),
                    load_nvfp4_linear_metadata(model_dir, weight_map, expert_prefix + ".down_proj"),
                )
            up_proj, down_proj = cached_experts[expert_index]
            expert_hidden = nvfp4_linear(latent_row, up_proj)
            expert_hidden = relu2(expert_hidden)
            expert_output = nvfp4_linear(expert_hidden, down_proj)
            row_activated_hidden.append(expert_hidden.detach().clone())
            row_expert_outputs.append(expert_output.detach().clone())
            weighted_contribution = expert_output * float(selected_weights[token_index, slot].item())
            row_contributions.append(weighted_contribution.detach().clone())
            routed_row = routed_row + weighted_contribution
        routed_output[token_index : token_index + 1] = routed_row
        routed_activated_hidden_per_row.append(row_activated_hidden)
        routed_expert_outputs_per_row.append(row_expert_outputs)
        routed_weighted_contributions_per_row.append(row_contributions)

    unique_selected_experts = sorted({expert for row in selected_expert_ids_per_row for expert in row})
    for expert_index in unique_selected_experts:
        up_proj, down_proj = cached_experts[expert_index]
        expert_dir = output_dir / f"expert_{expert_index:03d}"
        expert_dir.mkdir(parents=True, exist_ok=True)
        (expert_dir / "up_proj_weight_packed.bin").write_bytes(up_proj["weight"].contiguous().numpy().tobytes())
        (expert_dir / "up_proj_weight_block_scales.bin").write_bytes(
            up_proj["weight_scales"].contiguous().view(torch.uint8).numpy().tobytes()
        )
        write_tensor(
            expert_dir / "up_proj_weight_tensor_scale_fp32.bin",
            torch.tensor([up_proj["weight_scale_2"]], dtype=torch.float32),
        )
        (expert_dir / "down_proj_weight_packed.bin").write_bytes(down_proj["weight"].contiguous().numpy().tobytes())
        (expert_dir / "down_proj_weight_block_scales.bin").write_bytes(
            down_proj["weight_scales"].contiguous().view(torch.uint8).numpy().tobytes()
        )
        write_tensor(
            expert_dir / "down_proj_weight_tensor_scale_fp32.bin",
            torch.tensor([down_proj["weight_scale_2"]], dtype=torch.float32),
        )

    routed_projected = F.linear(routed_output.to(torch.float32), fc2_latent_weight.to(torch.float32))
    shared_up_output = run_dense_or_scaled_fp8_linear(norm_output, shared_up).to(torch.float32)
    if shared_down["family"] == "nvfp4":
        shared_output = nvfp4_linear(relu2(shared_up_output), shared_down)
    else:
        shared_output = run_dense_or_scaled_fp8_linear(relu2(shared_up_output), shared_down).to(torch.float32)
    mixer_output = routed_projected + shared_output
    final_output = input_hidden.to(torch.float32) + mixer_output

    write_tensor(output_dir / "input_hidden_fp32.bin", input_hidden)
    write_tensor(output_dir / "expected_norm_output_fp32.bin", norm_output)
    write_tensor(output_dir / "norm_weight_fp32.bin", norm_weight)
    write_tensor(output_dir / "gate_weight_fp32.bin", gate_weight)
    write_tensor(output_dir / "gate_score_correction_bias_fp32.bin", gate_score_correction_bias)
    if fc1_latent["family"] == "scaled_fp8":
        write_raw_uint8(output_dir / "fc1_latent_weight_fp8.bin", fc1_latent["weight"].to(torch.float8_e4m3fn))
        write_tensor(
            output_dir / "fc1_latent_weight_scale_fp32.bin",
            torch.tensor([fc1_latent["weight_scale"]], dtype=torch.float32),
        )
        write_tensor(
            output_dir / "fc1_latent_input_scale_fp32.bin",
            torch.tensor([fc1_latent["input_scale"]], dtype=torch.float32),
        )
        write_tensor(
            output_dir / "expected_fc1_latent_quantized_input_fp32.bin",
            scaled_fp8_roundtrip_input(norm_output, fc1_latent["input_scale"]),
        )
        write_tensor(
            output_dir / "expected_fc1_latent_weight_dequant_fp32.bin",
            scaled_fp8_dequantized_weight(fc1_latent["weight"], fc1_latent["weight_scale"]),
        )
    else:
        write_tensor(output_dir / "fc1_latent_weight_fp32.bin", fc1_latent["weight"].to(torch.float32))
    write_tensor(output_dir / "fc2_latent_weight_fp32.bin", fc2_latent_weight)
    if shared_up["family"] == "scaled_fp8":
        write_raw_uint8(output_dir / "shared_up_weight_fp8.bin", shared_up["weight"].to(torch.float8_e4m3fn))
        write_tensor(
            output_dir / "shared_up_weight_scale_fp32.bin",
            torch.tensor([shared_up["weight_scale"]], dtype=torch.float32),
        )
        write_tensor(
            output_dir / "shared_up_input_scale_fp32.bin",
            torch.tensor([shared_up["input_scale"]], dtype=torch.float32),
        )
    else:
        write_tensor(output_dir / "shared_up_weight_fp32.bin", shared_up["weight"].to(torch.float32))
    if shared_down["family"] == "nvfp4":
        (output_dir / "shared_down_weight_packed.bin").write_bytes(shared_down["weight"].contiguous().numpy().tobytes())
        (output_dir / "shared_down_weight_block_scales.bin").write_bytes(
            shared_down["weight_scales"].contiguous().view(torch.uint8).numpy().tobytes()
        )
        write_tensor(
            output_dir / "shared_down_weight_tensor_scale_fp32.bin",
            torch.tensor([shared_down["weight_scale_2"]], dtype=torch.float32),
        )
    elif shared_down["family"] == "scaled_fp8":
        write_raw_uint8(output_dir / "shared_down_weight_fp8.bin", shared_down["weight"].to(torch.float8_e4m3fn))
        write_tensor(
            output_dir / "shared_down_weight_scale_fp32.bin",
            torch.tensor([shared_down["weight_scale"]], dtype=torch.float32),
        )
        write_tensor(
            output_dir / "shared_down_input_scale_fp32.bin",
            torch.tensor([shared_down["input_scale"]], dtype=torch.float32),
        )
    else:
        write_tensor(output_dir / "shared_down_weight_fp32.bin", shared_down["weight"].to(torch.float32))

    write_tensor(output_dir / "expected_router_logits_fp32.bin", router_logits)
    write_tensor(output_dir / "expected_fc1_latent_output_fp32.bin", latent_states)
    write_tensor(output_dir / "expected_routed_latent_output_fp32.bin", routed_output)
    write_tensor(output_dir / "expected_shared_output_fp32.bin", shared_output)
    write_tensor(output_dir / "expected_mixer_output_fp32.bin", mixer_output)
    write_tensor(output_dir / "expected_final_output_fp32.bin", final_output)
    (output_dir / "expected_selected_expert_indices_u32.bin").write_bytes(
        torch.tensor(selected_expert_ids_per_row, dtype=torch.int32).numpy().tobytes()
    )
    write_tensor(
        output_dir / "expected_selected_expert_weights_fp32.bin",
        torch.stack(selected_weights_rows, dim=0).to(torch.float32),
    )
    for token_index, row_contributions in enumerate(routed_weighted_contributions_per_row):
        row_hidden = routed_activated_hidden_per_row[token_index]
        row_outputs = routed_expert_outputs_per_row[token_index]
        for slot, contribution in enumerate(row_contributions):
            write_tensor(
                output_dir / f"expected_routed_activated_hidden_row{token_index:02d}_slot{slot:02d}_fp32.bin",
                row_hidden[slot],
            )
            write_tensor(
                output_dir / f"expected_routed_expert_output_row{token_index:02d}_slot{slot:02d}_fp32.bin",
                row_outputs[slot],
            )
            write_tensor(
                output_dir / f"expected_routed_contribution_row{token_index:02d}_slot{slot:02d}_fp32.bin",
                contribution,
            )

    metadata = {
        "fixture_kind": "expert_layer_oracle_v1",
        "layer_index": args.layer_index,
        "input_rows": int(input_hidden.shape[0]),
        "hidden_size": hidden_size,
        "moe_latent_size": moe_latent_size,
        "routed_expert_intermediate_size": moe_intermediate_size,
        "shared_expert_intermediate_size": shared_intermediate_size,
        "n_routed_experts": n_routed_experts,
        "top_k": top_k,
        "n_group": n_group,
        "topk_group": topk_group,
        "routed_scaling_factor": routed_scaling_factor,
        "norm_topk_prob": norm_topk_prob,
        "rms_epsilon": layer_norm_eps,
        "fc1_latent_family": fc1_latent["family"],
        "shared_up_family": shared_up["family"],
        "shared_down_family": shared_down["family"],
        "oracle_device": device.type,
        "selected_expert_indices": selected_expert_ids_per_row,
        "source_tensors": {
            "norm_weight": prefix + ".norm.weight",
            "gate_weight": mixer_prefix + ".gate.weight",
            "gate_score_correction_bias": mixer_prefix + ".gate.e_score_correction_bias",
            "fc1_latent_weight": mixer_prefix + ".fc1_latent_proj.weight",
            "fc2_latent_weight": mixer_prefix + ".fc2_latent_proj.weight",
            "shared_up_weight": mixer_prefix + ".shared_experts.up_proj.weight",
            "shared_down_weight": mixer_prefix + ".shared_experts.down_proj.weight",
        },
        "notes": [
            "The input hidden state is deterministic unless --input-hidden-bin is provided.",
            "Router, latent projections, shared expert, and selected routed experts all use real checkpoint weights.",
            "The routed and shared expert paths mirror the current correctness-first runtime contract for the family actually present in the checkpoint.",
            "expected_final_output_fp32 includes the outer residual add.",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
