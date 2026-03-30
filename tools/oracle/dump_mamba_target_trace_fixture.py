#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer


MIN_INPUT_SCALE = 1.0 / 1024.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump a target-shaped Mamba multi-turn trace fixture driven by real tokenization and layer-0 weights."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--layer-index", type=int, default=0)
    return parser.parse_args()


def build_weight_map(index_path: Path) -> dict[str, str]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return payload["weight_map"]


def load_named_tensor(model_dir: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    shard = weight_map[name]
    with safe_open(str(model_dir / shard), framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def write_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.write_bytes(tensor.contiguous().to(torch.float32).numpy().tobytes())


def clamp_scale(value: float) -> float:
    if not torch.isfinite(torch.tensor(value)) or value < MIN_INPUT_SCALE:
        return MIN_INPUT_SCALE
    return float(value)


def quantize_input_fp8(x: torch.Tensor, input_scale: float) -> torch.Tensor:
    scale = clamp_scale(float(input_scale))
    quantized = (x.to(torch.float32) / scale).to(torch.float8_e4m3fn)
    return quantized.to(torch.float32) * scale


def scaled_fp8_linear(x: torch.Tensor, weight: torch.Tensor, weight_scale: float, input_scale: float) -> torch.Tensor:
    weight_f32 = weight.to(torch.float32) * float(weight_scale)
    return F.linear(quantize_input_fp8(x, input_scale), weight_f32)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    y = x.to(torch.float32)
    variance = y.pow(2).mean(dim=-1, keepdim=True)
    return y * torch.rsqrt(variance + eps) * weight.to(torch.float32)


def expand_text(spec: dict[str, Any]) -> str:
    content = str(spec["content"])
    repeat = int(spec.get("repeat", 1))
    joiner = str(spec.get("joiner", "\n"))
    pieces = []
    for index in range(repeat):
        pieces.append(content.replace("{index}", str(index + 1)))
    return joiner.join(pieces)


def materialize_message(spec: dict[str, Any]) -> dict[str, str]:
    return {
        "role": str(spec["role"]),
        "content": expand_text(spec),
    }


def tokenize_messages(tokenizer: AutoTokenizer, messages: list[dict[str, str]], add_generation_prompt: bool) -> list[int]:
    return list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    )


def common_prefix_length(lhs: list[int], rhs: list[int]) -> int:
    length = min(len(lhs), len(rhs))
    index = 0
    while index < length and lhs[index] == rhs[index]:
        index += 1
    return index


def split_projection(projected_states: torch.Tensor, intermediate_size: int, conv_dim: int, num_heads: int, n_groups: int, state_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    d_to_remove = 2 * intermediate_size + 2 * n_groups * state_size + num_heads
    d_mlp = (projected_states.shape[-1] - d_to_remove) // 2
    split_sizes = [d_mlp, d_mlp, intermediate_size, conv_dim, num_heads]
    _, _, _, hidden_states_b_c, dt = torch.split(projected_states, split_sizes, dim=-1)
    return hidden_states_b_c, dt


def expand_grouped_state(flat: torch.Tensor, num_heads: int, n_groups: int, state_size: int) -> torch.Tensor:
    grouped = flat.view(n_groups, state_size)
    group_expand = num_heads // n_groups
    return grouped[:, None, :].expand(n_groups, group_expand, state_size).reshape(num_heads, state_size)


def update_state(
    state: torch.Tensor,
    hidden: torch.Tensor,
    dt: torch.Tensor,
    B: torch.Tensor,
    A: torch.Tensor,
) -> torch.Tensor:
    dA = torch.exp(dt[..., None] * A)
    dB = dt[..., None] * B[:, None, :]
    dBx = dB * hidden[..., None]
    return state * dA + dBx


def build_trace_tokens(
    tokenizer: AutoTokenizer,
    profile: dict[str, Any],
) -> tuple[list[int], list[int], list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
    shared_system_message = materialize_message(profile["shared_system_root"])
    committed_history = [materialize_message(spec) for spec in profile["committed_history"]]
    phase_specs = profile["trace_phases"]

    shared_root_messages = [shared_system_message]
    committed_messages = [shared_system_message, *committed_history]
    committed_tokens = tokenize_messages(tokenizer, committed_messages, add_generation_prompt=False)

    messages_context = committed_messages.copy()
    previous_tokens = committed_tokens
    trace_phases: list[dict[str, Any]] = []
    trace_token_ids: list[int] = []
    for spec in phase_specs:
        message = materialize_message(spec)
        phase_kind = str(spec["kind"])
        with_message = [*messages_context, message]
        full_tokens = tokenize_messages(
            tokenizer,
            with_message,
            add_generation_prompt=(phase_kind == "user_tail"),
        )
        prefix_length = common_prefix_length(previous_tokens, full_tokens)
        phase_tokens = full_tokens[prefix_length:]
        if not phase_tokens:
            raise RuntimeError(f"trace phase {spec['name']!r} did not produce any tokens")
        trace_phases.append(
            {
                "name": str(spec["name"]),
                "kind": phase_kind,
                "length": len(phase_tokens),
                "source_repeat": int(spec.get("repeat", 1)),
                "prefix_overlap_tokens": prefix_length,
                "replaced_generation_prompt_tokens": len(previous_tokens) - prefix_length,
                "tokens": phase_tokens,
            }
        )
        trace_token_ids.extend(phase_tokens)
        messages_context = with_message
        previous_tokens = full_tokens if phase_kind == "user_tail" else tokenize_messages(
            tokenizer, messages_context, add_generation_prompt=False
        )
    return (
        tokenize_messages(tokenizer, shared_root_messages, add_generation_prompt=False),
        committed_tokens,
        trace_phases,
        shared_root_messages,
        committed_messages,
    )


def build_trace_phase_metadata(trace_phases: list[dict[str, Any]], trace_token_text: list[str]) -> list[dict[str, Any]]:
    phase_metadata: list[dict[str, Any]] = []
    phase_offset = 0
    for phase in trace_phases:
        start_step = phase_offset
        end_step = start_step + int(phase["length"])
        phase_metadata.append(
            {
                "name": phase["name"],
                "kind": phase["kind"],
                "length": phase["length"],
                "start_step": start_step,
                "end_step": end_step,
                "source_repeat": phase["source_repeat"],
                "prefix_overlap_tokens": phase["prefix_overlap_tokens"],
                "replaced_generation_prompt_tokens": phase["replaced_generation_prompt_tokens"],
                "preview_tokens": trace_token_text[start_step : min(end_step, start_step + 12)],
            }
        )
        phase_offset = end_step
    return phase_metadata


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    profile_path = Path(args.profile)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    weight_map = build_weight_map(model_dir / "model.safetensors.index.json")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=True,
    )

    hidden_size = int(config["hidden_size"])
    num_heads = int(config["mamba_num_heads"])
    head_dim = int(config["mamba_head_dim"])
    state_size = int(config["ssm_state_size"])
    n_groups = int(config["n_groups"])
    conv_kernel = int(config["conv_kernel"])
    intermediate_size = num_heads * head_dim
    conv_dim = intermediate_size + 2 * n_groups * state_size
    batch_size = 1

    prefix = f"backbone.layers.{args.layer_index}.mixer"
    layer_norm_eps = float(config["layer_norm_epsilon"])

    embeddings_weight = load_named_tensor(model_dir, weight_map, "backbone.embeddings.weight").to(torch.float32)
    layer0_norm_weight = load_named_tensor(model_dir, weight_map, f"backbone.layers.{args.layer_index}.norm.weight").to(torch.float32)
    in_proj_weight = load_named_tensor(model_dir, weight_map, prefix + ".in_proj.weight")
    in_proj_weight_scale = float(load_named_tensor(model_dir, weight_map, prefix + ".in_proj.weight_scale").item())
    in_proj_input_scale = float(load_named_tensor(model_dir, weight_map, prefix + ".in_proj.input_scale").item())
    conv1d_weight = load_named_tensor(model_dir, weight_map, prefix + ".conv1d.weight").to(torch.float32)
    conv1d_bias = load_named_tensor(model_dir, weight_map, prefix + ".conv1d.bias").to(torch.float32)
    dt_bias = load_named_tensor(model_dir, weight_map, prefix + ".dt_bias").to(torch.float32)
    A_log = load_named_tensor(model_dir, weight_map, prefix + ".A_log").to(torch.float32)
    D = load_named_tensor(model_dir, weight_map, prefix + ".D").to(torch.float32)

    (
        shared_root_tokens,
        committed_tokens,
        trace_phases,
        shared_root_messages,
        committed_messages,
    ) = build_trace_tokens(tokenizer, profile)

    trace_token_ids = [token for phase in trace_phases for token in phase["tokens"]]
    if not trace_token_ids:
        raise RuntimeError("trace profile did not produce any trace tokens")
    trace_token_text = [
        tokenizer.decode([token_id], clean_up_tokenization_spaces=False) for token_id in trace_token_ids
    ]

    prefix_input_ids = torch.tensor(committed_tokens, dtype=torch.long)
    trace_input_ids = torch.tensor(trace_token_ids, dtype=torch.long)

    prefix_inputs = rms_norm(embeddings_weight[prefix_input_ids], layer0_norm_weight, layer_norm_eps)
    trace_inputs = rms_norm(embeddings_weight[trace_input_ids], layer0_norm_weight, layer_norm_eps)

    prefix_projected = scaled_fp8_linear(prefix_inputs, in_proj_weight, in_proj_weight_scale, in_proj_input_scale)
    trace_projected = scaled_fp8_linear(trace_inputs, in_proj_weight, in_proj_weight_scale, in_proj_input_scale)

    prefix_conv_inputs, prefix_dt_raw = split_projection(
        prefix_projected,
        intermediate_size,
        conv_dim,
        num_heads,
        n_groups,
        state_size,
    )
    trace_conv_inputs, trace_dt_raw = split_projection(
        trace_projected,
        intermediate_size,
        conv_dim,
        num_heads,
        n_groups,
        state_size,
    )

    prefix_conv_outputs = F.silu(
        F.conv1d(
            prefix_conv_inputs.transpose(0, 1).unsqueeze(0),
            conv1d_weight,
            conv1d_bias,
            padding=conv_kernel - 1,
            groups=conv_dim,
        ).transpose(1, 2)[:, : prefix_conv_inputs.shape[0], :]
    ).squeeze(0)
    prefix_hidden, prefix_B_grouped, prefix_C_grouped = torch.split(
        prefix_conv_outputs,
        [intermediate_size, n_groups * state_size, n_groups * state_size],
        dim=-1,
    )

    A = -torch.exp(A_log).view(num_heads, 1, 1).expand(num_heads, head_dim, state_size).to(torch.float32)
    D_expanded = D.view(num_heads, 1).expand(num_heads, head_dim).to(torch.float32)

    seed_prefix_steps = min(int(profile.get("seed_prefix_replay_steps", 128)), prefix_hidden.shape[0])
    initial_state = torch.zeros((batch_size, num_heads, head_dim, state_size), dtype=torch.float32)
    for index in range(prefix_hidden.shape[0] - seed_prefix_steps, prefix_hidden.shape[0]):
        hidden_step = prefix_hidden[index].view(num_heads, head_dim)
        dt_step = torch.clamp(
            F.softplus(prefix_dt_raw[index][:, None].expand(num_heads, head_dim) + dt_bias[:, None]),
            min=float(config["time_step_min"]),
        )
        B_step = expand_grouped_state(prefix_B_grouped[index], num_heads, n_groups, state_size)
        initial_state[0] = update_state(initial_state[0], hidden_step, dt_step, B_step, A)

    conv_state = torch.zeros((conv_dim, conv_kernel), dtype=torch.float32)
    prefix_seed = prefix_conv_inputs[-conv_kernel:]
    conv_state[:, -prefix_seed.shape[0] :] = prefix_seed.transpose(0, 1)

    hidden_trace = torch.zeros((len(trace_token_ids), intermediate_size), dtype=torch.float32)
    dt_trace = torch.zeros((len(trace_token_ids), intermediate_size), dtype=torch.float32)
    B_trace = torch.zeros((len(trace_token_ids), num_heads * state_size), dtype=torch.float32)
    C_trace = torch.zeros((len(trace_token_ids), num_heads * state_size), dtype=torch.float32)

    for step in range(len(trace_token_ids)):
        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state[:, -1] = trace_conv_inputs[step]
        conv_output = torch.sum(conv_state * conv1d_weight[:, 0, :], dim=-1)
        conv_output = F.silu(conv_output + conv1d_bias)
        hidden_step, B_grouped_step, C_grouped_step = torch.split(
            conv_output,
            [intermediate_size, n_groups * state_size, n_groups * state_size],
            dim=-1,
        )
        dt_step = torch.clamp(
            F.softplus(trace_dt_raw[step][:, None].expand(num_heads, head_dim) + dt_bias[:, None]),
            min=float(config["time_step_min"]),
        )
        hidden_trace[step] = hidden_step
        dt_trace[step] = dt_step.reshape(-1)
        B_trace[step] = expand_grouped_state(B_grouped_step, num_heads, n_groups, state_size).reshape(-1)
        C_trace[step] = expand_grouped_state(C_grouped_step, num_heads, n_groups, state_size).reshape(-1)

    write_tensor(output_dir / "initial_state_fp32.bin", initial_state)
    write_tensor(output_dir / "A_fp32.bin", A.view(batch_size, num_heads, head_dim, state_size))
    write_tensor(output_dir / "D_fp32.bin", D_expanded.view(batch_size, num_heads, head_dim))
    write_tensor(output_dir / "hidden_trace_fp32.bin", hidden_trace.view(len(trace_token_ids), batch_size, num_heads, head_dim))
    write_tensor(output_dir / "dt_trace_fp32.bin", dt_trace.view(len(trace_token_ids), batch_size, num_heads, head_dim))
    write_tensor(output_dir / "B_trace_fp32.bin", B_trace.view(len(trace_token_ids), batch_size, num_heads, state_size))
    write_tensor(output_dir / "C_trace_fp32.bin", C_trace.view(len(trace_token_ids), batch_size, num_heads, state_size))
    (output_dir / "trace_tokens.json").write_text(
        json.dumps(
            {
                "trace_step_count": len(trace_token_ids),
                "tokens": [
                    {
                        "step": index,
                        "token_id": int(token_id),
                        "text": trace_token_text[index],
                    }
                    for index, token_id in enumerate(trace_token_ids)
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    metadata = {
        "fixture_kind": "mamba_target_chat_trace_v2",
        "generator": "dump_mamba_target_trace_fixture.py",
        "profile_path": str(profile_path),
        "layer_index": args.layer_index,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "state_size": state_size,
        "n_groups": n_groups,
        "shared_system_root_tokens": int(profile["target_shared_system_root_tokens"]),
        "committed_head_tokens": int(profile["target_committed_head_tokens"]),
        "actual_shared_system_root_tokens": len(shared_root_tokens),
        "actual_committed_head_tokens": len(committed_tokens),
        "seed_prefix_replay_steps": seed_prefix_steps,
        "trace_step_count": len(trace_token_ids),
        "trace_phases": build_trace_phase_metadata(trace_phases, trace_token_text),
        "source_tensors": {
            "embeddings": "backbone.embeddings.weight",
            "layer_norm": f"backbone.layers.{args.layer_index}.norm.weight",
            "in_proj": prefix + ".in_proj.weight",
            "conv1d_weight": prefix + ".conv1d.weight",
            "conv1d_bias": prefix + ".conv1d.bias",
            "A_log": prefix + ".A_log",
            "D": prefix + ".D",
            "dt_bias": prefix + ".dt_bias",
        },
        "messages": {
            "shared_system_root": shared_root_messages,
            "committed_history": committed_messages[1:],
        },
        "notes": [
            "This target-shaped trace now uses real tokenizer output, real embedding vectors, layer-0 RMSNorm, the checkpoint FP8 in_proj, and the real depthwise conv1d weights.",
            "The initial SSM state is no longer fixed synthetic state; it is seeded by replaying the last committed-prefix steps through the real projected prefix operands.",
            "The target shared-root and committed-head token counts remain service-profile targets, while actual tokenized counts are recorded separately in metadata.",
            "trace_tokens.json records the per-step token ids and decoded token pieces so operand hotspots can be mapped back to serializer patterns.",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
