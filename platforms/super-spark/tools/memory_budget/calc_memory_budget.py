#!/usr/bin/env python3

import argparse
import json
from math import floor
from pathlib import Path


def bytes_to_mib(value: int) -> float:
    return value / (1024.0 * 1024.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive reusable-state memory budgets from checkpoint metadata.")
    parser.add_argument("--checkpoint-report", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    report = json.loads(Path(args.checkpoint_report).read_text(encoding="utf-8"))
    cfg = report["config_summary"]
    kv_bits = report["quantization_summary"]["kv_cache_scheme"]["num_bits"]

    attention_layers = cfg["layers_block_type_counts"]["attention"]
    mamba_layers = cfg["layers_block_type_counts"]["mamba"]
    kv_bytes_per_scalar = kv_bits // 8
    kv_bytes_per_token = (
        attention_layers
        * cfg["num_key_value_heads"]
        * cfg["head_dim"]
        * 2
        * kv_bytes_per_scalar
    )

    mamba_inner = cfg["mamba_num_heads"] * cfg["mamba_head_dim"]
    conv_elems_per_layer = (mamba_inner + 2 * cfg["n_groups"] * cfg["ssm_state_size"]) * cfg["conv_kernel"]
    ssm_elems_per_layer = cfg["mamba_num_heads"] * cfg["mamba_head_dim"] * cfg["ssm_state_size"]
    recurrent_elems_per_layer = conv_elems_per_layer + ssm_elems_per_layer

    snapshot_bytes_fp32 = mamba_layers * recurrent_elems_per_layer * 4
    snapshot_bytes_fp16 = mamba_layers * recurrent_elems_per_layer * 2

    budgets_gib = [8, 16, 32]
    block_sizes = [32, 64, 128]
    scenarios = []
    for budget_gib in budgets_gib:
        budget_bytes = budget_gib * 1024 * 1024 * 1024
        for snapshot_dtype, snapshot_bytes in (("fp32", snapshot_bytes_fp32), ("fp16", snapshot_bytes_fp16)):
            max_blocks = floor(budget_bytes / snapshot_bytes)
            for block_size in block_sizes:
                scenarios.append(
                    {
                        "cache_budget_gib": budget_gib,
                        "snapshot_dtype": snapshot_dtype,
                        "block_size": block_size,
                        "snapshot_bytes": snapshot_bytes,
                        "max_blocks": max_blocks,
                        "max_prefix_tokens": max_blocks * block_size,
                        "amortized_snapshot_bytes_per_token": snapshot_bytes / block_size,
                        "amortized_total_reuse_bytes_per_token": kv_bytes_per_token + (snapshot_bytes / block_size),
                    }
                )

    summary = {
        "kv_bytes_per_token": kv_bytes_per_token,
        "kv_mib_per_1k_tokens": bytes_to_mib(kv_bytes_per_token * 1000),
        "mamba_conv_elems_per_layer": conv_elems_per_layer,
        "mamba_ssm_elems_per_layer": ssm_elems_per_layer,
        "mamba_recurrent_elems_per_layer": recurrent_elems_per_layer,
        "mamba_snapshot_bytes": {
            "fp32": snapshot_bytes_fp32,
            "fp16": snapshot_bytes_fp16,
        },
        "mamba_snapshot_mib": {
            "fp32": bytes_to_mib(snapshot_bytes_fp32),
            "fp16": bytes_to_mib(snapshot_bytes_fp16),
        },
        "scenarios": scenarios,
        "key_findings": [
            "The shipped Python cache object stores both conv_states and ssm_states for Mamba layers.",
            "Reusable Mamba snapshot cost is independent of block size; only the amortization per token changes.",
            "At FP32, one full Mamba snapshot across 40 Mamba layers is larger than 166 MiB.",
            "Smaller reusable blocks improve prefix-match granularity but materially worsen Mamba snapshot bytes per token.",
        ],
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Memory Budget Preflight",
        "",
        f"- Attention KV bytes per token: `{kv_bytes_per_token}` ({bytes_to_mib(kv_bytes_per_token):.4f} MiB/token)",
        f"- Attention KV per 1k tokens: `{summary['kv_mib_per_1k_tokens']:.2f} MiB`",
        f"- Mamba conv elems per layer: `{conv_elems_per_layer}`",
        f"- Mamba SSM elems per layer: `{ssm_elems_per_layer}`",
        f"- Mamba recurrent elems per layer: `{recurrent_elems_per_layer}`",
        f"- Full reusable Mamba snapshot FP32: `{snapshot_bytes_fp32}` bytes ({bytes_to_mib(snapshot_bytes_fp32):.2f} MiB)",
        f"- Full reusable Mamba snapshot FP16: `{snapshot_bytes_fp16}` bytes ({bytes_to_mib(snapshot_bytes_fp16):.2f} MiB)",
        "",
        "## Scenario Table",
        "",
        "| Cache GiB | Snapshot dtype | Block | Max blocks | Max prefix tokens | Snapshot bytes/token | Total reuse bytes/token |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for scenario in scenarios:
        lines.append(
            "| "
            f"{scenario['cache_budget_gib']} | "
            f"{scenario['snapshot_dtype']} | "
            f"{scenario['block_size']} | "
            f"{scenario['max_blocks']} | "
            f"{scenario['max_prefix_tokens']} | "
            f"{scenario['amortized_snapshot_bytes_per_token']:.1f} | "
            f"{scenario['amortized_total_reuse_bytes_per_token']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This report treats a reusable Mamba prefix state as the pair `(conv_states + ssm_states)` because that is what the shipped Python cache object materializes.",
            "- KV reuse remains cheap at roughly 4 KiB/token; the dominant memory pressure is the recurrent snapshot budget.",
            "- This makes block admission policy and cache budget enforcement mandatory before runtime implementation.",
        ]
    )
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
