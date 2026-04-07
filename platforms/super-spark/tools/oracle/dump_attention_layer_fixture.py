#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump a checkpoint-derived attention-layer oracle fixture.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--prompt-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer-index", type=int, default=7)
    parser.add_argument("--tokens-per-page", type=int, default=16)
    parser.add_argument("--input-hidden-bin", default=None)
    return parser.parse_args()


def build_weight_map(index_path: Path) -> dict[str, str]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return payload["weight_map"]


def load_named_tensor(model_dir: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    shard = weight_map[name]
    with safe_open(str(model_dir / shard), framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def load_prompt_case(path: Path, prompt_name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for case in payload["cases"]:
        if case["name"] == prompt_name:
            return case
    raise KeyError(f"prompt named {prompt_name!r} was not found in {path}")


def deterministic_hidden(token_ids: list[int], hidden_size: int) -> torch.Tensor:
    rows = []
    for token_index, token_id in enumerate(token_ids):
        values = torch.arange(hidden_size, dtype=torch.float32)
        row = ((values * 17.0 + float(token_id) * 13.0 + float(token_index) * 19.0) % 257.0) - 128.0
        row = row * 2.5e-3
        rows.append(row)
    return torch.stack(rows, dim=0).unsqueeze(0)


def load_float32_matrix(path: Path, cols: int) -> torch.Tensor:
    values = torch.frombuffer(path.read_bytes(), dtype=torch.float32)
    if values.numel() % cols != 0:
        raise ValueError(f"{path} does not contain a whole number of rows for hidden_size={cols}")
    return values.view(-1, cols).unsqueeze(0).clone()


def rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
    return hidden_states * torch.rsqrt(variance + epsilon) * weight.to(torch.float32)


def write_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.write_bytes(tensor.contiguous().to(torch.float32).numpy().tobytes())


def write_bf16_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.write_bytes(tensor.contiguous().to(torch.bfloat16).numpy().tobytes())


def scatter_kv_to_pages(
    states: torch.Tensor,
    tokens_per_page: int,
    total_pages: int,
) -> torch.Tensor:
    batch_size, kv_heads, token_count, head_dim = states.shape
    assert batch_size == 1
    cache = torch.zeros((total_pages, kv_heads, tokens_per_page, head_dim), dtype=torch.float32)
    for token in range(token_count):
        page_id = token // tokens_per_page
        page_offset = token % tokens_per_page
        cache[page_id, :, page_offset, :] = states[0, :, token, :].to(torch.float32)
    return cache


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    prompts_path = Path(args.prompts)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
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
    if not token_ids:
        raise ValueError("prompt produced zero tokens")

    weight_map = build_weight_map(model_dir / "model.safetensors.index.json")
    prefix = f"backbone.layers.{args.layer_index}"
    norm_name = prefix + ".norm.weight"
    q_name = prefix + ".mixer.q_proj.weight"
    k_name = prefix + ".mixer.k_proj.weight"
    v_name = prefix + ".mixer.v_proj.weight"
    o_name = prefix + ".mixer.o_proj.weight"

    hidden_size = int(config["hidden_size"])
    num_heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    layer_norm_eps = float(config["layer_norm_epsilon"])
    if args.input_hidden_bin is not None:
        hidden_states = load_float32_matrix(Path(args.input_hidden_bin), hidden_size)
    else:
        hidden_states = deterministic_hidden(token_ids, hidden_size)
    token_count = hidden_states.shape[1]
    total_pages = (token_count + args.tokens_per_page - 1) // args.tokens_per_page
    norm_weight = load_named_tensor(model_dir, weight_map, norm_name).to(torch.float32)
    q_weight = load_named_tensor(model_dir, weight_map, q_name).to(torch.float32)
    k_weight = load_named_tensor(model_dir, weight_map, k_name).to(torch.float32)
    v_weight = load_named_tensor(model_dir, weight_map, v_name).to(torch.float32)
    o_weight = load_named_tensor(model_dir, weight_map, o_name).to(torch.float32)

    norm_output = rms_norm(hidden_states, norm_weight, layer_norm_eps)
    q_fp32 = torch.nn.functional.linear(norm_output, q_weight).to(torch.float32)
    k_fp32 = torch.nn.functional.linear(norm_output, k_weight).to(torch.float32)
    v_fp32 = torch.nn.functional.linear(norm_output, v_weight).to(torch.float32)

    query_states = q_fp32.view(1, token_count, num_heads, head_dim).transpose(1, 2).contiguous()
    key_states = k_fp32.view(1, token_count, kv_heads, head_dim).transpose(1, 2).contiguous()
    value_states = v_fp32.view(1, token_count, kv_heads, head_dim).transpose(1, 2).contiguous()

    query_bf16 = query_states.to(torch.bfloat16)
    key_bf16 = key_states.to(torch.bfloat16)
    value_bf16 = value_states.to(torch.bfloat16)

    expanded_key = key_bf16[:, :, None, :, :].expand(1, kv_heads, num_heads // kv_heads, token_count, head_dim)
    expanded_key = expanded_key.reshape(1, num_heads, token_count, head_dim)
    expanded_value = value_bf16[:, :, None, :, :].expand(1, kv_heads, num_heads // kv_heads, token_count, head_dim)
    expanded_value = expanded_value.reshape(1, num_heads, token_count, head_dim)

    attn_weights = torch.matmul(
        query_bf16.to(torch.float32),
        expanded_key.transpose(2, 3).to(torch.float32),
    ) * (head_dim ** -0.5)
    causal_mask = torch.triu(
        torch.full((token_count, token_count), float("-inf"), dtype=torch.float32),
        diagonal=1,
    )
    attn_weights = attn_weights + causal_mask.view(1, 1, token_count, token_count)
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.bfloat16)
    attention_output = torch.matmul(attn_weights, expanded_value.to(torch.bfloat16))
    attention_output = attention_output.transpose(1, 2).contiguous().view(1, token_count, hidden_size)

    projected_output = torch.nn.functional.linear(attention_output.to(torch.float32), o_weight).to(torch.float32)
    final_output = hidden_states.to(torch.float32) + projected_output

    expected_key_cache = scatter_kv_to_pages(key_bf16.to(torch.float32), args.tokens_per_page, total_pages)
    expected_value_cache = scatter_kv_to_pages(value_bf16.to(torch.float32), args.tokens_per_page, total_pages)

    write_tensor(output_dir / "input_hidden_fp32.bin", hidden_states)
    write_tensor(output_dir / "norm_weight_fp32.bin", norm_weight)
    write_tensor(output_dir / "q_proj_weight_fp32.bin", q_weight)
    write_tensor(output_dir / "k_proj_weight_fp32.bin", k_weight)
    write_tensor(output_dir / "v_proj_weight_fp32.bin", v_weight)
    write_tensor(output_dir / "o_proj_weight_fp32.bin", o_weight)
    write_tensor(output_dir / "expected_norm_fp32.bin", norm_output)
    write_tensor(output_dir / "expected_q_fp32.bin", q_fp32)
    write_tensor(output_dir / "expected_k_fp32.bin", k_fp32)
    write_tensor(output_dir / "expected_v_fp32.bin", v_fp32)
    write_tensor(output_dir / "expected_attention_output_fp32.bin", attention_output)
    write_tensor(output_dir / "expected_projected_output_fp32.bin", projected_output)
    write_tensor(output_dir / "expected_final_output_fp32.bin", final_output)
    write_tensor(output_dir / "expected_key_cache_fp32.bin", expected_key_cache)
    write_tensor(output_dir / "expected_value_cache_fp32.bin", expected_value_cache)
    (output_dir / "token_ids.json").write_text(json.dumps(token_ids, indent=2) + "\n", encoding="utf-8")

    metadata = {
        "fixture_kind": "attention_layer_oracle_v1",
        "prompt_name": args.prompt_name,
        "layer_index": args.layer_index,
        "token_count": token_count,
        "tokens_per_page": args.tokens_per_page,
        "total_pages": total_pages,
        "hidden_size": hidden_size,
        "query_head_count": num_heads,
        "kv_head_count": kv_heads,
        "head_dim": head_dim,
        "rms_epsilon": layer_norm_eps,
        "source_tensors": {
            "norm_weight": norm_name,
            "q_proj": q_name,
            "k_proj": k_name,
            "v_proj": v_name,
            "o_proj": o_name,
        },
        "notes": [
            "The hidden-state input is deterministic and prompt-shaped; token count comes from the serialized prompt.",
            "If --input-hidden-bin is provided, the hidden-state input and token count come from that dumped runtime boundary instead.",
            "The fixture uses real layer-7 attention weights from the checkpoint.",
            "The expected attention path matches the current eager causal-attention contract without rotary position handling.",
            "Key and value cache tensors are dumped in page-major layout after BF16 staging to match the current runtime slice contract.",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
