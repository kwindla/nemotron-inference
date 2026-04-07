#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


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
    in_proj_weight_scale_name = mixer_prefix + ".in_proj.weight_scale"
    in_proj_input_scale_name = mixer_prefix + ".in_proj.input_scale"
    conv_weight_name = mixer_prefix + ".conv1d.weight"
    conv_bias_name = mixer_prefix + ".conv1d.bias"
    A_log_name = mixer_prefix + ".A_log"
    D_name = mixer_prefix + ".D"
    dt_bias_name = mixer_prefix + ".dt_bias"
    out_proj_weight_name = mixer_prefix + ".out_proj.weight"
    out_proj_weight_scale_name = mixer_prefix + ".out_proj.weight_scale"
    out_proj_input_scale_name = mixer_prefix + ".out_proj.input_scale"

    block_norm_weight = load_named_tensor(model_dir, weight_map, block_norm_name).to(torch.float32)
    mixer_norm_weight = load_named_tensor(model_dir, weight_map, mixer_norm_name).to(torch.float32)
    in_proj_weight = load_named_tensor(model_dir, weight_map, in_proj_weight_name)
    in_proj_scaled_fp8 = (
        in_proj_weight_scale_name in weight_map and
        in_proj_input_scale_name in weight_map
    )
    if in_proj_scaled_fp8:
        in_proj_weight_scale = float(load_named_tensor(model_dir, weight_map, in_proj_weight_scale_name).item())
        in_proj_input_scale = float(load_named_tensor(model_dir, weight_map, in_proj_input_scale_name).item())
    else:
        in_proj_weight_scale = 0.0
        in_proj_input_scale = 0.0
    conv_weight = load_named_tensor(model_dir, weight_map, conv_weight_name).to(torch.float32)
    conv_bias = load_named_tensor(model_dir, weight_map, conv_bias_name).to(torch.float32)
    A_log = load_named_tensor(model_dir, weight_map, A_log_name).to(torch.float32)
    D = load_named_tensor(model_dir, weight_map, D_name).to(torch.float32)
    dt_bias = load_named_tensor(model_dir, weight_map, dt_bias_name).to(torch.float32)
    out_proj_weight = load_named_tensor(model_dir, weight_map, out_proj_weight_name)
    out_proj_scaled_fp8 = (
        out_proj_weight_scale_name in weight_map and
        out_proj_input_scale_name in weight_map
    )
    if out_proj_scaled_fp8:
        out_proj_weight_scale = float(load_named_tensor(model_dir, weight_map, out_proj_weight_scale_name).item())
        out_proj_input_scale = float(load_named_tensor(model_dir, weight_map, out_proj_input_scale_name).item())
    else:
        out_proj_weight_scale = 0.0
        out_proj_input_scale = 0.0

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

    norm_output = rms_norm(input_hidden, block_norm_weight, layer_norm_eps)
    if in_proj_scaled_fp8:
        quantized_norm_input, in_proj_output = scaled_fp8_linear(
            norm_output,
            in_proj_weight,
            in_proj_weight_scale,
            in_proj_input_scale,
        )
    else:
        quantized_norm_input = norm_output.to(torch.float32)
        in_proj_output = F.linear(
            norm_output.to(torch.float32),
            in_proj_weight.to(torch.float32),
        )

    batch_size = input_hidden.shape[0]
    gate = in_proj_output[:, :intermediate_size]
    hidden_states_B_C = in_proj_output[:, intermediate_size : intermediate_size + conv_dim]
    dt_pre = in_proj_output[:, intermediate_size + conv_dim :]

    updated_conv_state = initial_conv_state.clone()
    next_ssm_state = initial_ssm_state.clone()
    scan_output_rows: list[torch.Tensor] = []
    quantized_scan_output_rows: list[torch.Tensor] = []
    projected_output_rows: list[torch.Tensor] = []

    A = -torch.exp(A_log).view(1, num_heads, 1, 1)
    D_expanded = D.view(1, num_heads, 1)

    for row in range(batch_size):
        row_hidden_states_B_C = hidden_states_B_C[row : row + 1]
        row_dt_pre = dt_pre[row : row + 1]
        row_gate = gate[row : row + 1]

        updated_conv_state = torch.roll(updated_conv_state, shifts=-1, dims=-1)
        updated_conv_state[:, :, -1] = row_hidden_states_B_C

        conv_output = (updated_conv_state * conv_weight[:, 0, :][None, :, :]).sum(dim=-1)
        conv_output = conv_output + conv_bias[None, :]
        conv_output = F.silu(conv_output)

        hidden_after_conv = conv_output[:, :intermediate_size]
        B_grouped = conv_output[:, intermediate_size : intermediate_size + (n_groups * state_size)]
        C_grouped = conv_output[:, intermediate_size + (n_groups * state_size) :]

        dt = F.softplus(row_dt_pre[:, :, None] + dt_bias[None, :, None]).expand(-1, -1, head_dim)
        dt = torch.clamp(dt, min=time_step_min)

        B = B_grouped.view(1, n_groups, state_size)
        B = B[:, :, None, :].expand(1, n_groups, num_heads // n_groups, state_size).reshape(1, num_heads, state_size)
        C = C_grouped.view(1, n_groups, state_size)
        C = C[:, :, None, :].expand(1, n_groups, num_heads // n_groups, state_size).reshape(1, num_heads, state_size)
        hidden_ssm = hidden_after_conv.view(1, num_heads, head_dim)

        dA = torch.exp(dt[:, :, :, None] * A)
        dB = dt[:, :, :, None] * B[:, :, None, :]
        dBx = dB * hidden_ssm[:, :, :, None]
        next_ssm_state = next_ssm_state * dA + dBx

        y = torch.matmul(
            next_ssm_state.view(num_heads, head_dim, state_size),
            C.view(num_heads, state_size, 1),
        ).view(1, num_heads, head_dim)
        y = y + hidden_ssm * D_expanded
        y_flat = y.view(1, intermediate_size)

        scan_output_row = grouped_rms_norm_gated(
            y_flat,
            row_gate,
            mixer_norm_weight,
            layer_norm_eps,
            n_groups,
        )
        if out_proj_scaled_fp8:
            quantized_scan_output_row, projected_output_row = scaled_fp8_linear(
                scan_output_row,
                out_proj_weight,
                out_proj_weight_scale,
                out_proj_input_scale,
            )
        else:
            quantized_scan_output_row = scan_output_row.to(torch.float32)
            projected_output_row = F.linear(
                scan_output_row.to(torch.float32),
                out_proj_weight.to(torch.float32),
            )

        scan_output_rows.append(scan_output_row)
        quantized_scan_output_rows.append(quantized_scan_output_row)
        projected_output_rows.append(projected_output_row)

    scan_output = torch.cat(scan_output_rows, dim=0)
    quantized_scan_output = torch.cat(quantized_scan_output_rows, dim=0)
    projected_output = torch.cat(projected_output_rows, dim=0)
    final_output = input_hidden + projected_output

    write_tensor(output_dir / "input_hidden_fp32.bin", input_hidden)
    write_tensor(output_dir / "initial_conv_state_fp32.bin", initial_conv_state)
    write_tensor(output_dir / "initial_ssm_state_fp32.bin", initial_ssm_state)
    write_tensor(output_dir / "input_norm_weight_fp32.bin", block_norm_weight)
    write_tensor(output_dir / "mixer_norm_weight_fp32.bin", mixer_norm_weight)
    if in_proj_scaled_fp8:
        write_raw_uint8(output_dir / "in_proj_weight_fp8.bin", in_proj_weight.to(torch.float8_e4m3fn))
        write_tensor(output_dir / "in_proj_weight_scale_fp32.bin", torch.tensor([in_proj_weight_scale], dtype=torch.float32))
        write_tensor(output_dir / "in_proj_input_scale_fp32.bin", torch.tensor([in_proj_input_scale], dtype=torch.float32))
    else:
        write_tensor(output_dir / "in_proj_weight_fp32.bin", in_proj_weight.to(torch.float32))
    write_tensor(output_dir / "conv1d_weight_fp32.bin", conv_weight)
    write_tensor(output_dir / "conv1d_bias_fp32.bin", conv_bias)
    write_tensor(output_dir / "A_log_fp32.bin", A_log)
    write_tensor(output_dir / "D_fp32.bin", D)
    write_tensor(output_dir / "dt_bias_fp32.bin", dt_bias)
    if out_proj_scaled_fp8:
        write_raw_uint8(output_dir / "out_proj_weight_fp8.bin", out_proj_weight.to(torch.float8_e4m3fn))
        write_tensor(output_dir / "out_proj_weight_scale_fp32.bin", torch.tensor([out_proj_weight_scale], dtype=torch.float32))
        write_tensor(output_dir / "out_proj_input_scale_fp32.bin", torch.tensor([out_proj_input_scale], dtype=torch.float32))
    else:
        write_tensor(output_dir / "out_proj_weight_fp32.bin", out_proj_weight.to(torch.float32))
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
        "input_rows": batch_size,
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
            "Scaled FP8 in_proj and out_proj expectations mirror the current correctness-first runtime contract: input round-trip through E4M3 using input_scale, then dequantized-weight linear math.",
            "expected_final_output_fp32 includes the outer block residual add.",
        ],
    }
    if in_proj_scaled_fp8:
        metadata["source_tensors"]["in_proj_weight_scale"] = in_proj_weight_scale_name
        metadata["source_tensors"]["in_proj_input_scale"] = in_proj_input_scale_name
    if out_proj_scaled_fp8:
        metadata["source_tensors"]["out_proj_weight_scale"] = out_proj_weight_scale_name
        metadata["source_tensors"]["out_proj_input_scale"] = out_proj_input_scale_name
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
