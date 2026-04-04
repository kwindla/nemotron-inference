#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

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
FP4_POSITIVE_VALUES_TENSOR = torch.tensor(FP4_POSITIVE_VALUES, dtype=torch.float32)
FP4_DECODE_TABLE_TENSOR = torch.tensor(FP4_DECODE_TABLE, dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump a checkpoint-derived Mamba decode-layer oracle fixture.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--input-hidden-bin", default=None)
    parser.add_argument("--initial-conv-state-bin", default=None)
    parser.add_argument("--initial-ssm-state-bin", default=None)
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


def load_float32_matrix(path: Path, cols: int) -> torch.Tensor:
    data = path.read_bytes()
    if len(data) % 4 != 0:
        raise ValueError(f"{path} is not a float32 buffer")
    values = torch.frombuffer(bytearray(data), dtype=torch.float32).clone()
    if values.numel() % cols != 0:
        raise ValueError(f"{path} does not divide evenly into rows of width {cols}")
    return values.view(-1, cols)


def load_float32_tensor(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    data = path.read_bytes()
    if len(data) % 4 != 0:
        raise ValueError(f"{path} is not a float32 buffer")
    values = torch.frombuffer(bytearray(data), dtype=torch.float32).clone()
    expected = 1
    for dim in shape:
        expected *= dim
    if values.numel() != expected:
        raise ValueError(f"{path} expected {expected} float32 values, found {values.numel()}")
    return values.view(*shape)


def write_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.write_bytes(tensor.contiguous().to(torch.float32).numpy().tobytes())


def write_raw_uint8(path: Path, tensor: torch.Tensor) -> None:
    path.write_bytes(tensor.contiguous().view(torch.uint8).numpy().tobytes())


def load_scale_scalar(model_dir: Path, weight_map: dict[str, str], name: str) -> float:
    tensor = load_named_tensor(model_dir, weight_map, name).to(torch.float32)
    return float(tensor.max().item())


def clamp_scale(value: float) -> float:
    if not math.isfinite(value) or value < MIN_SCALE:
        return MIN_SCALE
    return value


def bf16_roundtrip(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.bfloat16).to(torch.float32)


def fused_add_rms_norm_bf16(
    hidden_input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined = hidden_input.to(torch.float32) + residual.to(torch.float32)
    variance = combined.pow(2).mean(dim=-1, keepdim=True)
    updated_residual = bf16_roundtrip(combined)
    normalized = bf16_roundtrip(
        combined * torch.rsqrt(variance + epsilon) * weight.to(torch.float32)
    )
    return updated_residual, normalized


def rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    return hidden_states * torch.rsqrt(variance + epsilon) * weight.to(torch.float32)


def grouped_rms_norm_gated(
    hidden_states: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    n_groups: int,
) -> torch.Tensor:
    hidden_states = hidden_states.to(torch.float32)
    gate = gate.to(torch.float32)
    weight = weight.to(torch.float32)
    gated = hidden_states * F.silu(gate)
    group_size = hidden_states.shape[-1] // n_groups
    grouped = gated.view(gated.shape[0], n_groups, group_size)
    variance = grouped.pow(2).mean(dim=-1, keepdim=True)
    normalized = grouped * torch.rsqrt(variance + epsilon)
    return normalized.view_as(gated) * weight


def scaled_fp8_linear(
    activations: torch.Tensor,
    weight_quantized: torch.Tensor,
    weight_scale: float,
    input_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    quantized_input = (activations.to(torch.float32) / input_scale).to(torch.float8_e4m3fn)
    round_tripped_input = quantized_input.to(torch.float32) * input_scale
    dequantized_weight = weight_quantized.to(torch.float32) * weight_scale
    output = torch.nn.functional.linear(round_tripped_input, dequantized_weight)
    return round_tripped_input, output


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
        "weight_scale_2": load_scale_scalar(model_dir, weight_map, tensor_scale_name),
        "input_scale": load_scale_scalar(model_dir, weight_map, prefix + ".input_scale"),
        "rows": int(weight.shape[0]),
        "cols": int(weight_scales.shape[1] * 16),
    }


def load_dense_or_scaled_fp8_linear_metadata(
    model_dir: Path,
    weight_map: dict[str, str],
    prefix: str,
) -> dict[str, object]:
    if prefix + ".weight_scale_2" in weight_map:
        return load_nvfp4_linear_metadata(model_dir, weight_map, prefix)
    weight = load_named_tensor(model_dir, weight_map, prefix + ".weight")
    weight_scale_name = prefix + ".weight_scale"
    input_scale_name = prefix + ".input_scale"
    if weight_scale_name in weight_map and input_scale_name in weight_map:
        return {
            "family": "scaled_fp8",
            "prefix": prefix,
            "weight": weight,
            "weight_scale": load_scale_scalar(model_dir, weight_map, weight_scale_name),
            "input_scale": load_scale_scalar(model_dir, weight_map, input_scale_name),
        }
    return {
        "family": "dense",
        "prefix": prefix,
        "weight": weight.to(torch.float32),
    }


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
    return torch.nn.functional.linear(
        activation_dequant.to(device=activations.device),
        weight_dequant,
    )


def run_dense_or_scaled_fp8_linear(activations: torch.Tensor, linear_metadata: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    if linear_metadata["family"] == "nvfp4":
        output = nvfp4_linear(activations, linear_metadata).to(torch.float32)
        return activations.to(torch.float32), output
    if linear_metadata["family"] == "scaled_fp8":
        return scaled_fp8_linear(
            activations,
            linear_metadata["weight"],
            linear_metadata["weight_scale"],
            linear_metadata["input_scale"],
        )
    if linear_metadata["family"] == "dense":
        activations_f32 = activations.to(torch.float32)
        return activations_f32, F.linear(activations_f32, linear_metadata["weight"].to(torch.float32))
    raise ValueError(f"unsupported linear family {linear_metadata['family']!r}")


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    weight_map = build_weight_map(model_dir / "model.safetensors.index.json")

    hidden_size = int(config["hidden_size"])
    num_heads = int(config["mamba_num_heads"])
    head_dim = int(config["mamba_head_dim"])
    state_size = int(config["ssm_state_size"])
    n_groups = int(config["n_groups"])
    conv_kernel = int(config["conv_kernel"])
    layer_norm_eps = float(config["layer_norm_epsilon"])
    time_step_min = float(config["time_step_min"])
    intermediate_size = num_heads * head_dim
    conv_dim = intermediate_size + 2 * n_groups * state_size

    prefix = f"backbone.layers.{args.layer_index}"
    mixer_prefix = prefix + ".mixer"
    block_norm_name = prefix + ".norm.weight"
    mixer_norm_name = mixer_prefix + ".norm.weight"
    in_proj_weight_name = mixer_prefix + ".in_proj.weight"
    conv_weight_name = mixer_prefix + ".conv1d.weight"
    conv_bias_name = mixer_prefix + ".conv1d.bias"
    A_log_name = mixer_prefix + ".A_log"
    D_name = mixer_prefix + ".D"
    dt_bias_name = mixer_prefix + ".dt_bias"
    out_proj_weight_name = mixer_prefix + ".out_proj.weight"

    block_norm_weight = load_named_tensor(model_dir, weight_map, block_norm_name).to(torch.float32)
    mixer_norm_weight = load_named_tensor(model_dir, weight_map, mixer_norm_name).to(torch.float32)
    in_proj = load_dense_or_scaled_fp8_linear_metadata(model_dir, weight_map, mixer_prefix + ".in_proj")
    conv_weight = load_named_tensor(model_dir, weight_map, conv_weight_name).to(torch.float32)
    conv_bias = load_named_tensor(model_dir, weight_map, conv_bias_name).to(torch.float32)
    A_log = load_named_tensor(model_dir, weight_map, A_log_name).to(torch.float32)
    D = load_named_tensor(model_dir, weight_map, D_name).to(torch.float32)
    dt_bias = load_named_tensor(model_dir, weight_map, dt_bias_name).to(torch.float32)
    out_proj = load_dense_or_scaled_fp8_linear_metadata(model_dir, weight_map, mixer_prefix + ".out_proj")

    if args.input_hidden_bin is not None:
        input_hidden = load_float32_matrix(Path(args.input_hidden_bin), hidden_size)
    else:
        input_hidden = deterministic_pattern((1, hidden_size), 2.0e-3, 11)
    if args.initial_conv_state_bin is not None:
        initial_conv_state = load_float32_tensor(
            Path(args.initial_conv_state_bin),
            (1, conv_dim, conv_kernel),
        )
    else:
        initial_conv_state = deterministic_pattern((1, conv_dim, conv_kernel), 1.0e-3, 37)
    if args.initial_ssm_state_bin is not None:
        initial_ssm_state = load_float32_tensor(
            Path(args.initial_ssm_state_bin),
            (1, num_heads, head_dim, state_size),
        )
    else:
        initial_ssm_state = deterministic_pattern((1, num_heads, head_dim, state_size), 5.0e-4, 73)

    input_hidden_bf16 = bf16_roundtrip(input_hidden)
    residual = torch.zeros_like(input_hidden_bf16)
    _, norm_output = fused_add_rms_norm_bf16(
        input_hidden_bf16,
        residual,
        block_norm_weight,
        layer_norm_eps,
    )
    quantized_norm_input, in_proj_output = run_dense_or_scaled_fp8_linear(
        norm_output,
        in_proj,
    )

    gate = in_proj_output[:, :intermediate_size]
    hidden_states_B_C = in_proj_output[:, intermediate_size : intermediate_size + conv_dim]
    dt_pre = in_proj_output[:, intermediate_size + conv_dim :]

    updated_conv_state = initial_conv_state.clone()
    updated_conv_state = torch.roll(updated_conv_state, shifts=-1, dims=-1)
    updated_conv_state[:, :, -1] = hidden_states_B_C

    conv_output = (updated_conv_state * conv_weight[:, 0, :][None, :, :]).sum(dim=-1)
    conv_output = conv_output + conv_bias[None, :]
    conv_output = F.silu(conv_output)

    hidden_after_conv = conv_output[:, :intermediate_size]
    B_grouped = conv_output[:, intermediate_size : intermediate_size + (n_groups * state_size)]
    C_grouped = conv_output[:, intermediate_size + (n_groups * state_size) :]

    dt = F.softplus(dt_pre[:, :, None] + dt_bias[None, :, None]).expand(-1, -1, head_dim)
    dt = torch.clamp(dt, min=time_step_min)
    A = -torch.exp(A_log).view(1, num_heads, 1, 1)
    D_expanded = D.view(1, num_heads, 1)

    B = B_grouped.view(1, n_groups, state_size)
    B = B[:, :, None, :].expand(1, n_groups, num_heads // n_groups, state_size).reshape(1, num_heads, state_size)
    C = C_grouped.view(1, n_groups, state_size)
    C = C[:, :, None, :].expand(1, n_groups, num_heads // n_groups, state_size).reshape(1, num_heads, state_size)
    hidden_ssm = hidden_after_conv.view(1, num_heads, head_dim)

    dA = torch.exp(dt[:, :, :, None] * A)
    dB = dt[:, :, :, None] * B[:, :, None, :]
    dBx = dB * hidden_ssm[:, :, :, None]
    next_ssm_state = initial_ssm_state * dA + dBx

    y = torch.matmul(
        next_ssm_state.view(num_heads, head_dim, state_size),
        C.view(num_heads, state_size, 1),
    ).view(1, num_heads, head_dim)
    y = y + hidden_ssm * D_expanded
    y_flat = y.view(1, intermediate_size)

    scan_output = grouped_rms_norm_gated(
        y_flat,
        gate,
        mixer_norm_weight,
        layer_norm_eps,
        n_groups,
    )
    quantized_scan_output, projected_output = run_dense_or_scaled_fp8_linear(
        scan_output,
        out_proj,
    )
    projected_output_bf16 = bf16_roundtrip(projected_output)
    final_output = input_hidden + projected_output_bf16

    write_tensor(output_dir / "input_hidden_fp32.bin", input_hidden)
    write_tensor(output_dir / "initial_conv_state_fp32.bin", initial_conv_state)
    write_tensor(output_dir / "initial_ssm_state_fp32.bin", initial_ssm_state)
    write_tensor(output_dir / "input_norm_weight_fp32.bin", block_norm_weight)
    write_tensor(output_dir / "mixer_norm_weight_fp32.bin", mixer_norm_weight)
    if in_proj["family"] == "nvfp4":
        write_raw_uint8(output_dir / "in_proj_weight_packed.bin", in_proj["weight"])
        write_raw_uint8(output_dir / "in_proj_weight_block_scales.bin", in_proj["weight_scales"])
        write_tensor(output_dir / "in_proj_weight_tensor_scale_fp32.bin", torch.tensor([in_proj["weight_scale_2"]], dtype=torch.float32))
    elif in_proj["family"] == "scaled_fp8":
        write_raw_uint8(output_dir / "in_proj_weight_fp8.bin", in_proj["weight"].to(torch.float8_e4m3fn))
        write_tensor(output_dir / "in_proj_weight_scale_fp32.bin", torch.tensor([in_proj["weight_scale"]], dtype=torch.float32))
        write_tensor(output_dir / "in_proj_input_scale_fp32.bin", torch.tensor([in_proj["input_scale"]], dtype=torch.float32))
    else:
        write_tensor(output_dir / "in_proj_weight_fp32.bin", in_proj["weight"].to(torch.float32))
    write_tensor(output_dir / "conv1d_weight_fp32.bin", conv_weight)
    write_tensor(output_dir / "conv1d_bias_fp32.bin", conv_bias)
    write_tensor(output_dir / "A_log_fp32.bin", A_log)
    write_tensor(output_dir / "D_fp32.bin", D)
    write_tensor(output_dir / "dt_bias_fp32.bin", dt_bias)
    if out_proj["family"] == "nvfp4":
        write_raw_uint8(output_dir / "out_proj_weight_packed.bin", out_proj["weight"])
        write_raw_uint8(output_dir / "out_proj_weight_block_scales.bin", out_proj["weight_scales"])
        write_tensor(output_dir / "out_proj_weight_tensor_scale_fp32.bin", torch.tensor([out_proj["weight_scale_2"]], dtype=torch.float32))
    elif out_proj["family"] == "scaled_fp8":
        write_raw_uint8(output_dir / "out_proj_weight_fp8.bin", out_proj["weight"].to(torch.float8_e4m3fn))
        write_tensor(output_dir / "out_proj_weight_scale_fp32.bin", torch.tensor([out_proj["weight_scale"]], dtype=torch.float32))
        write_tensor(output_dir / "out_proj_input_scale_fp32.bin", torch.tensor([out_proj["input_scale"]], dtype=torch.float32))
    else:
        write_tensor(output_dir / "out_proj_weight_fp32.bin", out_proj["weight"].to(torch.float32))
    write_tensor(output_dir / "expected_norm_output_fp32.bin", norm_output)
    write_tensor(output_dir / "expected_quantized_norm_input_fp32.bin", quantized_norm_input)
    write_tensor(output_dir / "expected_in_proj_output_fp32.bin", in_proj_output)
    write_tensor(output_dir / "expected_updated_conv_state_fp32.bin", updated_conv_state)
    write_tensor(output_dir / "expected_scan_output_fp32.bin", scan_output)
    write_tensor(output_dir / "expected_quantized_scan_output_fp32.bin", quantized_scan_output)
    write_tensor(output_dir / "expected_next_ssm_state_fp32.bin", next_ssm_state)
    write_tensor(output_dir / "expected_projected_output_fp32.bin", projected_output)
    write_tensor(output_dir / "expected_final_output_fp32.bin", final_output)

    metadata = {
        "fixture_kind": "mamba_layer_decode_oracle_v1",
        "layer_index": args.layer_index,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "state_size": state_size,
        "n_groups": n_groups,
        "conv_kernel_size": conv_kernel,
        "input_rms_epsilon": layer_norm_eps,
        "mixer_rms_epsilon": layer_norm_eps,
        "time_step_min": time_step_min,
        "source_tensors": {
            "input_norm_weight": block_norm_name,
            "mixer_norm_weight": mixer_norm_name,
            "in_proj_weight": in_proj_weight_name,
            "conv1d_weight": conv_weight_name,
            "conv1d_bias": conv_bias_name,
            "A_log": A_log_name,
            "D": D_name,
            "dt_bias": dt_bias_name,
            "out_proj_weight": out_proj_weight_name,
        },
        "notes": [
            "The input hidden state and initial conv/SSM state are deterministic synthetic tensors unless explicit input/state binaries are provided.",
            "The fixture uses real layer-0 Mamba weights from the checkpoint.",
            "Projection fixtures mirror the runtime descriptor contract: scaled-FP8 uses E4M3 input round-trip, NVFP4 uses the fp32 activation path with dynamic activation packing.",
            "expected_final_output_fp32 includes the fp32 wrapper residual add after the BF16 inner-layer delta cast.",
        ],
    }
    metadata["in_proj_family"] = in_proj["family"]
    metadata["out_proj_family"] = out_proj["family"]
    if in_proj["family"] == "scaled_fp8":
        metadata["source_tensors"]["in_proj_weight_scale"] = mixer_prefix + ".in_proj.weight_scale"
        metadata["source_tensors"]["in_proj_input_scale"] = mixer_prefix + ".in_proj.input_scale"
    elif in_proj["family"] == "nvfp4":
        metadata["source_tensors"]["in_proj_weight_scale"] = mixer_prefix + ".in_proj.weight_scale"
        metadata["source_tensors"]["in_proj_weight_scale_2"] = mixer_prefix + ".in_proj.weight_scale_2"
    if out_proj["family"] == "scaled_fp8":
        metadata["source_tensors"]["out_proj_weight_scale"] = mixer_prefix + ".out_proj.weight_scale"
        metadata["source_tensors"]["out_proj_input_scale"] = mixer_prefix + ".out_proj.input_scale"
    elif out_proj["family"] == "nvfp4":
        metadata["source_tensors"]["out_proj_weight_scale"] = mixer_prefix + ".out_proj.weight_scale"
        metadata["source_tensors"]["out_proj_weight_scale_2"] = mixer_prefix + ".out_proj.weight_scale_2"
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
