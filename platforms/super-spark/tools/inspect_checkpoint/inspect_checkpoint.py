#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi
from safetensors import safe_open


CHAR_TO_BLOCK = {
    "M": "mamba",
    "E": "moe",
    "*": "attention",
    "-": "mlp",
}

BLOCK_TO_CHAR = {value: key for key, value in CHAR_TO_BLOCK.items()}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def classify_tensor(name: str) -> str:
    if name == "backbone.embeddings.weight":
        return "embedding"
    if ".self_attn." in name:
        return "attention"
    if ".mixer.in_proj" in name or ".mixer.out_proj" in name:
        return "mamba_projection"
    if ".mixer.conv1d." in name or name.endswith(".A_log") or name.endswith(".D") or name.endswith(".dt_bias"):
        return "mamba_state"
    if ".mixer.experts." in name:
        return "routed_expert"
    if ".mixer.shared_experts." in name:
        return "shared_expert"
    if ".mixer.gate." in name:
        return "router"
    if name.endswith(".norm.weight") or ".norm." in name:
        return "norm"
    if name.startswith("mtp."):
        return "mtp"
    if name.startswith("lm_head"):
        return "lm_head"
    return "other"


def layer_pattern_from_config(config: dict) -> tuple[str, str]:
    if "hybrid_override_pattern" in config:
        pattern = config["hybrid_override_pattern"]
    else:
        pattern = "".join(BLOCK_TO_CHAR[layer] for layer in config["layers_block_type"])

    if "mtp_hybrid_override_pattern" in config:
        mtp_pattern = config["mtp_hybrid_override_pattern"]
    else:
        mtp_pattern = "".join(BLOCK_TO_CHAR[layer] for layer in config.get("mtp_layers_block_type", []))

    return pattern, mtp_pattern


def inspect_shard(shard_path: Path) -> tuple[Counter, Counter, Counter, list[dict]]:
    dtype_counts = Counter()
    op_counts = Counter()
    suffix_counts = Counter()
    samples: list[dict] = []
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        for index, key in enumerate(handle.keys()):
            view = handle.get_slice(key)
            dtype = str(view.get_dtype())
            shape = list(view.get_shape())
            dtype_counts[dtype] += 1
            op_counts[classify_tensor(key)] += 1
            suffix = key.rsplit(".", 1)[-1]
            suffix_counts[suffix] += 1
            if index < 24:
                samples.append({"name": key, "dtype": dtype, "shape": shape})
    return dtype_counts, op_counts, suffix_counts, samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the Nemotron checkpoint and summarize tensor structure.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    config = read_json(model_dir / "config.json")
    tokenizer_config = read_json(model_dir / "tokenizer_config.json")
    safetensors_index = read_json(model_dir / "model.safetensors.index.json")
    quant_config = read_json(model_dir / "hf_quant_config.json")

    api = HfApi()
    info = api.model_info(args.repo_id, files_metadata=True)

    pattern, mtp_pattern = layer_pattern_from_config(config)
    layer_counts = Counter(CHAR_TO_BLOCK[ch] for ch in pattern)
    mtp_counts = Counter(CHAR_TO_BLOCK[ch] for ch in mtp_pattern)

    shard_tensor_counts: dict[str, int] = {}
    total_dtype_counts = Counter()
    total_op_counts = Counter()
    total_suffix_counts = Counter()
    shard_samples: dict[str, list[dict]] = {}

    shard_names = sorted({Path(name).name for name in safetensors_index["weight_map"].values()})
    for shard_name in shard_names:
        shard_path = model_dir / shard_name
        dtype_counts, op_counts, suffix_counts, samples = inspect_shard(shard_path)
        shard_tensor_counts[shard_name] = sum(dtype_counts.values())
        total_dtype_counts.update(dtype_counts)
        total_op_counts.update(op_counts)
        total_suffix_counts.update(suffix_counts)
        shard_samples[shard_name] = samples

    report = {
        "repo_id": info.id,
        "repo_sha": info.sha,
        "gated": getattr(info, "gated", None),
        "model_dir": str(model_dir),
        "file_count": len(info.siblings or []),
        "config_summary": {
            "model_type": config["model_type"],
            "architectures": config["architectures"],
            "num_hidden_layers": len(pattern),
            "layers_block_type_counts": dict(layer_counts),
            "attention_layer_indices": [idx for idx, ch in enumerate(pattern) if ch == "*"],
            "num_attention_heads": config["num_attention_heads"],
            "num_key_value_heads": config["num_key_value_heads"],
            "head_dim": config["head_dim"],
            "hidden_size": config["hidden_size"],
            "mamba_num_heads": config["mamba_num_heads"],
            "mamba_head_dim": config["mamba_head_dim"],
            "n_groups": config["n_groups"],
            "ssm_state_size": config["ssm_state_size"],
            "conv_kernel": config["conv_kernel"],
            "num_experts_per_tok": config["num_experts_per_tok"],
            "n_routed_experts": config["n_routed_experts"],
            "n_shared_experts": config["n_shared_experts"],
            "moe_intermediate_size": config["moe_intermediate_size"],
            "moe_latent_size": config["moe_latent_size"],
            "mamba_ssm_cache_dtype": config["mamba_ssm_cache_dtype"],
            "max_position_embeddings": config["max_position_embeddings"],
            "num_nextn_predict_layers": config["num_nextn_predict_layers"],
            "mtp_layers_block_type_counts": dict(mtp_counts),
        },
        "tokenizer_summary": {
            "tokenizer_class": tokenizer_config.get("tokenizer_class"),
            "chat_template_in_tokenizer_config": bool(tokenizer_config.get("chat_template")),
            "bos_token": tokenizer_config.get("bos_token"),
            "eos_token": tokenizer_config.get("eos_token"),
            "pad_token": tokenizer_config.get("pad_token"),
            "added_tokens_decoder_count": len(tokenizer_config.get("added_tokens_decoder", {})),
            "chat_template_file_present": (model_dir / "chat_template.jinja").exists(),
        },
        "quantization_summary": {
            "quant_method": config["quantization_config"].get("quant_method"),
            "quant_algo": config["quantization_config"].get("quant_algo"),
            "kv_cache_scheme": config["quantization_config"].get("kv_cache_scheme"),
            "config_groups": len(config["quantization_config"].get("config_groups", {})),
            "quantized_layers": len(config["quantization_config"].get("quantized_layers", [])),
            "hf_quant_config_keys": sorted(quant_config.keys()),
        },
        "tensor_inventory": {
            "tensor_count": len(safetensors_index["weight_map"]),
            "shard_count": len(shard_names),
            "shard_tensor_counts": shard_tensor_counts,
            "dtype_counts": dict(total_dtype_counts),
            "op_class_counts": dict(total_op_counts),
            "suffix_counts_top_20": dict(total_suffix_counts.most_common(20)),
            "sample_tensors_by_shard": shard_samples,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
