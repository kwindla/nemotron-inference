#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer


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
FP4_LEVELS_TENSOR = torch.tensor(FP4_POSITIVE_VALUES, dtype=torch.float32)
FP4_DECODE_TENSOR = torch.tensor(FP4_DECODE_TABLE, dtype=torch.float32)


def configure_torch_precision() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump a checkpoint-derived single-token full-model oracle fixture.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--prompt-name", required=True)
    parser.add_argument("--mode", choices=("single_token", "prefix_prefill"), default="single_token")
    parser.add_argument("--token-index", type=int, default=0)
    parser.add_argument("--prompt-token-count", type=int, default=0)
    parser.add_argument("--capture-layers", default="0,1,7")
    parser.add_argument("--stop-layer", type=int, default=-1)
    parser.add_argument("--output-dir", required=True)
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


def load_prompt_case(path: Path, prompt_name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for case in payload["cases"]:
        if case["name"] == prompt_name:
            return case
    raise KeyError(f"prompt named {prompt_name!r} was not found in {path}")


def write_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.write_bytes(tensor.contiguous().to(torch.float32).cpu().numpy().tobytes())


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
) -> torch.Tensor:
    quantized_input = (activations.to(torch.float32) / input_scale).to(torch.float8_e4m3fn)
    round_tripped_input = quantized_input.to(torch.float32) * input_scale
    dequantized_weight = weight_quantized.to(torch.float32) * weight_scale
    return torch.nn.functional.linear(round_tripped_input, dequantized_weight)


def clamp_scale(value: float) -> float:
    if not math.isfinite(value) or value < MIN_SCALE:
        return MIN_SCALE
    return value


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
    rows, cols = values.shape
    assert cols % 16 == 0
    device = values.device
    values_f32 = values.to(torch.float32)
    blocks = values_f32.view(rows, cols // 16, 16)
    one = torch.tensor(1.0, dtype=torch.float32, device=device)
    min_scale = torch.tensor(MIN_SCALE, dtype=torch.float32, device=device)
    fp4_levels = FP4_LEVELS_TENSOR.to(device=device)

    global_max_abs = values_f32.abs().amax()
    tensor_scale = torch.where(
        global_max_abs > (FP4_MAX_FINITE * FP8_E4M3_MAX_FINITE),
        global_max_abs / (FP4_MAX_FINITE * FP8_E4M3_MAX_FINITE),
        one,
    )
    tensor_scale = torch.clamp(tensor_scale, min=min_scale)

    block_max = blocks.abs().amax(dim=-1)
    block_scale = torch.where(
        block_max > 0,
        block_max / (FP4_MAX_FINITE * tensor_scale),
        one,
    )
    block_scale = torch.clamp(block_scale, min=min_scale)
    block_scales = block_scale.to(torch.float8_e4m3fn).view(torch.uint8).contiguous().view(-1)

    full_scale = tensor_scale * block_scale.unsqueeze(-1)
    normalized = blocks / full_scale
    sign_bits = (normalized < 0).to(torch.uint8) * 0x08
    positive_index = torch.argmin(
        (normalized.abs().unsqueeze(-1) - fp4_levels.view(1, 1, 1, -1)).abs(),
        dim=-1,
    ).to(torch.uint8)
    fp4_codes = sign_bits | positive_index
    fp4_pairs = fp4_codes.view(rows, cols // 16, 8, 2)
    packed = (fp4_pairs[..., 0] | (fp4_pairs[..., 1] << 4)).contiguous().view(-1)

    return {
        "packed": packed,
        "block_scales": block_scales,
        "tensor_scale": tensor_scale,
        "rows": rows,
        "cols": cols,
    }


def dequantize_nvfp4_matrix(
    packed: torch.Tensor,
    block_scales: torch.Tensor,
    tensor_scale: float,
    rows: int,
    cols: int,
) -> torch.Tensor:
    assert cols % 16 == 0
    device = packed.device if isinstance(packed, torch.Tensor) else None
    packed_tensor = _as_uint8_tensor(packed, device=device)
    block_scale_tensor = _as_uint8_tensor(block_scales, device=packed_tensor.device)
    decode_table = FP4_DECODE_TENSOR.to(device=packed_tensor.device)
    tensor_scale_tensor = _as_scalar_tensor(tensor_scale, packed_tensor.device)

    packed_bytes = packed_tensor.view(rows, cols // 16, 8)
    low = torch.bitwise_and(packed_bytes, 0x0F)
    high = torch.bitwise_right_shift(packed_bytes, 4)
    decoded_pairs = torch.stack((decode_table[low.long()], decode_table[high.long()]), dim=-1)
    decoded = decoded_pairs.view(rows, cols // 16, 16)

    block_scale_values = block_scale_tensor.view(rows, cols // 16).view(torch.float8_e4m3fn).to(torch.float32)
    scales = block_scale_values * tensor_scale_tensor
    return (decoded * scales.unsqueeze(-1)).view(rows, cols)


def load_nvfp4_linear_metadata(model_dir: Path, weight_map: dict[str, str], prefix: str) -> dict[str, object]:
    weight = load_named_tensor(model_dir, weight_map, prefix + ".weight")
    weight_scales = load_named_tensor(model_dir, weight_map, prefix + ".weight_scale")
    tensor_scale_name = prefix + ".weight_scale_2"
    if tensor_scale_name not in weight_map:
        tensor_scale_name = prefix + ".input_scale"
    weight_scale_2 = float(load_named_tensor(model_dir, weight_map, tensor_scale_name).item())
    return {
        "family": "nvfp4",
        "prefix": prefix,
        "weight": weight.contiguous().view(torch.uint8).flatten().cpu(),
        "weight_scales": weight_scales.contiguous().view(torch.uint8).flatten().cpu(),
        "weight_scale_2": weight_scale_2,
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
            "weight": weight,
            "weight_scale": float(load_named_tensor(model_dir, weight_map, weight_scale_name, device=device).item()),
            "input_scale": float(load_named_tensor(model_dir, weight_map, input_scale_name, device=device).item()),
        }
    return {
        "family": "dense",
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


def nvfp4_linear(activations: torch.Tensor, linear_metadata: dict[str, object]) -> torch.Tensor:
    packed_activation = pack_fp32_to_nvfp4_dynamic(activations.to(torch.float32))
    dequantized_activation = dequantize_nvfp4_matrix(
        packed_activation["packed"],
        packed_activation["block_scales"],
        float(packed_activation["tensor_scale"]),
        packed_activation["rows"],
        packed_activation["cols"],
    )
    device_key = str(activations.device)
    device_cache = linear_metadata.setdefault("dequantized_weight_by_device", {})
    dequantized_weight = device_cache.get(device_key)
    if dequantized_weight is None:
        host_weight = linear_metadata.get("dequantized_weight_host")
        if host_weight is None:
            host_weight = dequantize_nvfp4_matrix(
                linear_metadata["weight"],
                linear_metadata["weight_scales"],
                float(linear_metadata["weight_scale_2"]),
                linear_metadata["rows"],
                linear_metadata["cols"],
            )
            linear_metadata["dequantized_weight_host"] = host_weight
        dequantized_weight = host_weight.to(device=activations.device)
        device_cache[device_key] = dequantized_weight
    return torch.nn.functional.linear(dequantized_activation.to(device=activations.device), dequantized_weight)


def relu2(x: torch.Tensor) -> torch.Tensor:
    return torch.square(torch.relu(x))


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
    group_size = scores.shape[-1] // n_group
    grouped_scores = scores_for_choice.view(scores.shape[0], n_group, group_size)
    top2 = torch.topk(grouped_scores, k=min(2, group_size), dim=-1, sorted=True)[0]
    group_scores = top2.sum(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    expert_mask = group_mask.repeat_interleave(group_size, dim=-1)
    masked_scores = torch.where(expert_mask > 0, scores_for_choice, torch.zeros_like(scores_for_choice))
    topk_scores, topk_indices = torch.topk(masked_scores, k=top_k, dim=-1, sorted=False)
    selected_weights = torch.gather(scores, 1, topk_indices)
    if norm_topk_prob:
        selected_weights = selected_weights / (selected_weights.sum(dim=-1, keepdim=True) + 1.0e-20)
    selected_weights = selected_weights * routed_scaling_factor
    return topk_indices.to(torch.int32), selected_weights.to(torch.float32)


def run_attention_block(
    hidden_states: torch.Tensor,
    model_dir: Path,
    weight_map: dict[str, str],
    config: dict[str, Any],
    layer_index: int,
    device: torch.device,
) -> torch.Tensor:
    prefix = f"backbone.layers.{layer_index}"
    norm_weight = load_named_tensor(model_dir, weight_map, prefix + ".norm.weight", device=device).to(torch.float32)
    q_weight = load_named_tensor(model_dir, weight_map, prefix + ".mixer.q_proj.weight", device=device).to(torch.float32)
    k_weight = load_named_tensor(model_dir, weight_map, prefix + ".mixer.k_proj.weight", device=device).to(torch.float32)
    v_weight = load_named_tensor(model_dir, weight_map, prefix + ".mixer.v_proj.weight", device=device).to(torch.float32)
    o_weight = load_named_tensor(model_dir, weight_map, prefix + ".mixer.o_proj.weight", device=device).to(torch.float32)

    hidden_size = int(config["hidden_size"])
    num_heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    epsilon = float(config["layer_norm_epsilon"])

    norm_output = rms_norm(hidden_states, norm_weight, epsilon)
    q = F.linear(norm_output, q_weight).to(torch.float32)
    k = F.linear(norm_output, k_weight).to(torch.float32)
    v = F.linear(norm_output, v_weight).to(torch.float32)

    token_count = hidden_states.shape[0]
    query_states = q.view(1, token_count, num_heads, head_dim).transpose(1, 2).contiguous()
    key_states = k.view(1, token_count, kv_heads, head_dim).transpose(1, 2).contiguous()
    value_states = v.view(1, token_count, kv_heads, head_dim).transpose(1, 2).contiguous()

    expanded_key = key_states[:, :, None, :, :].expand(1, kv_heads, num_heads // kv_heads, token_count, head_dim)
    expanded_key = expanded_key.reshape(1, num_heads, token_count, head_dim)
    expanded_value = value_states[:, :, None, :, :].expand(1, kv_heads, num_heads // kv_heads, token_count, head_dim)
    expanded_value = expanded_value.reshape(1, num_heads, token_count, head_dim)

    attn_weights = torch.matmul(
        query_states.to(torch.float32),
        expanded_key.transpose(2, 3).to(torch.float32),
    ) * (head_dim ** -0.5)
    causal_mask = torch.triu(
        torch.ones((token_count, token_count), dtype=torch.bool, device=attn_weights.device),
        diagonal=1,
    )
    attn_weights = attn_weights.masked_fill(causal_mask.view(1, 1, token_count, token_count), float("-inf"))
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)
    attention_output = torch.matmul(attn_weights.to(torch.bfloat16), expanded_value.to(torch.bfloat16))
    attention_output = attention_output.transpose(1, 2).contiguous().view(1, token_count, hidden_size).to(torch.float32)
    projected_output = F.linear(attention_output, o_weight).to(torch.float32)
    return hidden_states.to(torch.float32) + projected_output.view(token_count, hidden_size)


def run_mamba_block(
    hidden_states: torch.Tensor,
    model_dir: Path,
    weight_map: dict[str, str],
    config: dict[str, Any],
    layer_index: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prefix = f"backbone.layers.{layer_index}"
    mixer_prefix = prefix + ".mixer"

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

    input_norm_weight = load_named_tensor(model_dir, weight_map, prefix + ".norm.weight", device=device).to(torch.float32)
    mixer_norm_weight = load_named_tensor(model_dir, weight_map, mixer_prefix + ".norm.weight", device=device).to(torch.float32)
    in_proj_weight = load_named_tensor(model_dir, weight_map, mixer_prefix + ".in_proj.weight", device=device)
    in_proj_scaled_fp8 = (
        (mixer_prefix + ".in_proj.weight_scale") in weight_map and
        (mixer_prefix + ".in_proj.input_scale") in weight_map
    )
    if in_proj_scaled_fp8:
        in_proj_weight_scale = float(load_named_tensor(model_dir, weight_map, mixer_prefix + ".in_proj.weight_scale", device=device).item())
        in_proj_input_scale = float(load_named_tensor(model_dir, weight_map, mixer_prefix + ".in_proj.input_scale", device=device).item())
    else:
        in_proj_weight_scale = 0.0
        in_proj_input_scale = 0.0
    conv_weight = load_named_tensor(model_dir, weight_map, mixer_prefix + ".conv1d.weight", device=device).to(torch.float32)
    conv_bias = load_named_tensor(model_dir, weight_map, mixer_prefix + ".conv1d.bias", device=device).to(torch.float32)
    A_log = load_named_tensor(model_dir, weight_map, mixer_prefix + ".A_log", device=device).to(torch.float32)
    D = load_named_tensor(model_dir, weight_map, mixer_prefix + ".D", device=device).to(torch.float32)
    dt_bias = load_named_tensor(model_dir, weight_map, mixer_prefix + ".dt_bias", device=device).to(torch.float32)
    out_proj_weight = load_named_tensor(model_dir, weight_map, mixer_prefix + ".out_proj.weight", device=device)
    out_proj_scaled_fp8 = (
        (mixer_prefix + ".out_proj.weight_scale") in weight_map and
        (mixer_prefix + ".out_proj.input_scale") in weight_map
    )
    if out_proj_scaled_fp8:
        out_proj_weight_scale = float(load_named_tensor(model_dir, weight_map, mixer_prefix + ".out_proj.weight_scale", device=device).item())
        out_proj_input_scale = float(load_named_tensor(model_dir, weight_map, mixer_prefix + ".out_proj.input_scale", device=device).item())
    else:
        out_proj_weight_scale = 0.0
        out_proj_input_scale = 0.0

    token_count = hidden_states.shape[0]
    conv_state = torch.zeros((1, conv_dim, conv_kernel), dtype=torch.float32, device=device)
    ssm_state = torch.zeros((1, num_heads, head_dim, state_size), dtype=torch.float32, device=device)

    norm_output = rms_norm(hidden_states, input_norm_weight, layer_norm_eps)
    if in_proj_scaled_fp8:
        in_proj_output = scaled_fp8_linear(norm_output, in_proj_weight, in_proj_weight_scale, in_proj_input_scale)
    else:
        in_proj_output = F.linear(norm_output.to(torch.float32), in_proj_weight.to(torch.float32))

    gate = in_proj_output[:, :intermediate_size]
    hidden_states_B_C = in_proj_output[:, intermediate_size : intermediate_size + conv_dim]
    dt_pre = in_proj_output[:, intermediate_size + conv_dim :]
    scan_rows: list[torch.Tensor] = []

    for token_index in range(token_count):
        token_hidden_states_B_C = hidden_states_B_C[token_index : token_index + 1, :]
        token_gate = gate[token_index : token_index + 1, :]
        token_dt_pre = dt_pre[token_index : token_index + 1, :]

        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state[:, :, -1] = token_hidden_states_B_C.view(1, conv_dim)

        conv_output = (conv_state * conv_weight[:, 0, :][None, :, :]).sum(dim=-1)
        conv_output = F.silu(conv_output + conv_bias[None, :])

        hidden_after_conv = conv_output[:, :intermediate_size]
        B_grouped = conv_output[:, intermediate_size : intermediate_size + n_groups * state_size]
        C_grouped = conv_output[:, intermediate_size + n_groups * state_size :]

        dt = F.softplus(token_dt_pre.view(1, num_heads, 1) + dt_bias[None, :, None]).expand(-1, -1, head_dim)
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
        ssm_state = ssm_state * dA + dBx

        y = torch.matmul(
            ssm_state.view(num_heads, head_dim, state_size),
            C.view(num_heads, state_size, 1),
        ).view(1, num_heads, head_dim)
        y = y + hidden_ssm * D_expanded
        y_flat = y.view(1, intermediate_size)

        scan_rows.append(
            grouped_rms_norm_gated(
                y_flat,
                token_gate,
                mixer_norm_weight,
                layer_norm_eps,
                n_groups,
            )
        )

    scan_output = torch.cat(scan_rows, dim=0)
    if out_proj_scaled_fp8:
        projected_output = scaled_fp8_linear(
            scan_output,
            out_proj_weight,
            out_proj_weight_scale,
            out_proj_input_scale,
        )
    else:
        projected_output = F.linear(scan_output.to(torch.float32), out_proj_weight.to(torch.float32))
    return (
        hidden_states.to(torch.float32) + projected_output,
        conv_state.to(torch.float32),
        ssm_state.to(torch.float32),
    )


def run_expert_block(
    hidden_states: torch.Tensor,
    model_dir: Path,
    weight_map: dict[str, str],
    config: dict[str, Any],
    layer_index: int,
    device: torch.device,
) -> torch.Tensor:
    prefix = f"backbone.layers.{layer_index}"
    mixer_prefix = prefix + ".mixer"

    moe_latent_size = int(config["moe_latent_size"])
    top_k = int(config["num_experts_per_tok"])
    n_group = int(config["n_group"])
    topk_group = int(config["topk_group"])
    routed_scaling_factor = float(config["routed_scaling_factor"])
    norm_topk_prob = bool(config["norm_topk_prob"])
    layer_norm_eps = float(config["layer_norm_epsilon"])

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
    shared_down = load_shared_down_metadata(
        model_dir, weight_map, mixer_prefix + ".shared_experts.down_proj", device=device
    )

    norm_output = rms_norm(hidden_states, norm_weight, layer_norm_eps)
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

    token_count = hidden_states.shape[0]
    routed_output = torch.zeros((token_count, moe_latent_size), dtype=torch.float32, device=device)
    expert_linear_cache: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
    for token_index in range(token_count):
        token_latent = latent_states[token_index : token_index + 1, :]
        for slot, expert_index in enumerate(selected_indices[token_index].tolist()):
            expert_prefix = mixer_prefix + f".experts.{expert_index}"
            expert_entry = expert_linear_cache.get(expert_prefix)
            if expert_entry is None:
                expert_entry = (
                    load_nvfp4_linear_metadata(model_dir, weight_map, expert_prefix + ".up_proj"),
                    load_nvfp4_linear_metadata(model_dir, weight_map, expert_prefix + ".down_proj"),
                )
                expert_linear_cache[expert_prefix] = expert_entry
            up_proj, down_proj = expert_entry
            expert_hidden = nvfp4_linear(token_latent, up_proj)
            expert_hidden = relu2(expert_hidden)
            expert_output = nvfp4_linear(expert_hidden, down_proj)
            routed_output[token_index : token_index + 1, :] = routed_output[token_index : token_index + 1, :] + (
                expert_output * float(selected_weights[token_index, slot].item())
            )

    routed_projected = F.linear(routed_output.to(torch.float32), fc2_latent_weight.to(torch.float32))
    shared_up_output = run_dense_or_scaled_fp8_linear(norm_output, shared_up).to(torch.float32)
    shared_up_activated = relu2(shared_up_output)
    if shared_down["family"] == "nvfp4":
        shared_output = nvfp4_linear(shared_up_activated, shared_down)
    else:
        shared_output = run_dense_or_scaled_fp8_linear(shared_up_activated, shared_down).to(torch.float32)
    mixer_output = routed_projected + shared_output
    return hidden_states.to(torch.float32) + mixer_output


def main() -> int:
    configure_torch_precision()
    args = parse_args()
    model_dir = Path(args.model_dir)
    prompts_path = Path(args.prompts)
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
    prompt_case = load_prompt_case(prompts_path, args.prompt_name)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=True,
    )
    prompt_token_ids = tokenizer.apply_chat_template(
        prompt_case["messages"],
        tokenize=True,
        add_generation_prompt=prompt_case.get("add_generation_prompt", True),
    )
    if not prompt_token_ids:
        raise ValueError("prompt produced zero tokens")
    selected_token_id = None
    if args.mode == "single_token":
        if args.token_index < 0 or args.token_index >= len(prompt_token_ids):
            raise IndexError(f"token index {args.token_index} is out of range for {len(prompt_token_ids)} prompt tokens")
        selected_token_id = int(prompt_token_ids[args.token_index])
        runtime_token_ids = [selected_token_id]
    else:
        prompt_token_count = args.prompt_token_count if args.prompt_token_count > 0 else len(prompt_token_ids)
        prompt_token_count = min(prompt_token_count, len(prompt_token_ids))
        if prompt_token_count <= 0:
            raise ValueError("prefix_prefill mode requires at least one prompt token")
        runtime_token_ids = [int(token) for token in prompt_token_ids[:prompt_token_count]]
    capture_layers = sorted({int(value) for value in args.capture_layers.split(",") if value.strip()})
    stop_layer = args.stop_layer if args.stop_layer >= 0 else int(config["num_hidden_layers"]) - 1

    embeddings = load_named_tensor(model_dir, weight_map, "backbone.embeddings.weight", device=device).to(torch.float32)
    hidden_states = embeddings[runtime_token_ids, :].to(torch.float32)
    write_tensor(output_dir / "expected_embedding_output_fp32.bin", hidden_states)

    captured_outputs: dict[int, torch.Tensor] = {}
    layer_kinds: list[str] = []
    num_layers = int(config["num_hidden_layers"])
    for layer_index in range(num_layers):
        if layer_index > stop_layer:
            break
        block_type = config["layers_block_type"][layer_index]
        layer_kinds.append(block_type)
        final_mamba_conv_state = None
        final_mamba_ssm_state = None
        if block_type == "attention":
            hidden_states = run_attention_block(hidden_states, model_dir, weight_map, config, layer_index, device)
        elif block_type == "mamba":
            hidden_states, final_mamba_conv_state, final_mamba_ssm_state = run_mamba_block(
                hidden_states,
                model_dir,
                weight_map,
                config,
                layer_index,
                device,
            )
        elif block_type == "moe":
            hidden_states = run_expert_block(hidden_states, model_dir, weight_map, config, layer_index, device)
        else:
            raise ValueError(f"unsupported block type {block_type!r} at layer {layer_index}")

        if layer_index in capture_layers:
            captured_outputs[layer_index] = hidden_states.detach().clone()
            write_tensor(output_dir / f"expected_layer_{layer_index:03d}_output_fp32.bin", hidden_states)
            if final_mamba_conv_state is not None and final_mamba_ssm_state is not None:
                write_tensor(
                    output_dir / f"expected_layer_{layer_index:03d}_mamba_conv_state_fp32.bin",
                    final_mamba_conv_state.reshape(-1),
                )
                write_tensor(
                    output_dir / f"expected_layer_{layer_index:03d}_mamba_ssm_state_fp32.bin",
                    final_mamba_ssm_state.reshape(-1),
                )

    final_hidden = hidden_states.to(torch.float32)
    write_tensor(output_dir / "expected_final_hidden_fp32.bin", final_hidden)

    final_norm_name = None
    for candidate in ("backbone.norm_f.weight", "norm_f.weight", "backbone.final_norm.weight", "final_norm.weight"):
        if candidate in weight_map:
            final_norm_name = candidate
            break
    if final_norm_name is not None:
        final_norm_weight = load_named_tensor(model_dir, weight_map, final_norm_name, device=device).to(torch.float32)
        final_hidden_normed = rms_norm(final_hidden, final_norm_weight, float(config["layer_norm_epsilon"]))
    else:
        final_hidden_normed = final_hidden
    write_tensor(output_dir / "expected_final_hidden_normed_fp32.bin", final_hidden_normed)

    logits = None
    if stop_layer >= num_layers - 1:
        lm_head_weight = load_named_tensor(model_dir, weight_map, "lm_head.weight", device=device).to(torch.float32)
        logits = F.linear(final_hidden_normed, lm_head_weight).to(torch.float32)
        write_tensor(output_dir / "expected_logits_fp32.bin", logits)

    metadata = {
        "fixture_kind": "single_token_full_model_oracle_v1" if args.mode == "single_token" else "prefix_prefill_prefix_oracle_v1",
        "mode": args.mode,
        "prompt_name": args.prompt_name,
        "prompt_token_count": len(prompt_token_ids),
        "runtime_token_count": len(runtime_token_ids),
        "selected_token_index": args.token_index if args.mode == "single_token" else None,
        "selected_token_id": selected_token_id,
        "runtime_token_ids": runtime_token_ids,
        "stop_layer": stop_layer,
        "hidden_size": int(config["hidden_size"]),
        "vocab_size": int(config["vocab_size"]),
        "num_hidden_layers": num_layers,
        "capture_layers": capture_layers,
        "layer_kinds": layer_kinds,
        "oracle_device": str(device),
        "final_norm_tensor": final_norm_name,
        "source_tensors": {
            "embeddings": "backbone.embeddings.weight",
            "lm_head": "lm_head.weight",
        },
        "notes": [
            "single_token mode runs one token from scratch with zeroed cache/state.",
            "prefix_prefill mode runs a short prompt prefix from scratch and can stop after an early capture layer to keep fixture generation practical.",
            "Attention, Mamba, and MoE blocks mirror the current correctness-first runtime contracts already used by the layer-level oracle fixtures.",
            "Captured per-layer outputs are intended for the composed registry-backed forward-path validation.",
            "Captured Mamba layers also include final conv/SSM state snapshots for decode-boundary localization.",
        ],
    }
    (output_dir / "prompt_token_ids.json").write_text(json.dumps(prompt_token_ids, indent=2) + "\n", encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
