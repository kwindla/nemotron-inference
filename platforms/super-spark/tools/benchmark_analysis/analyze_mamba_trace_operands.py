#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Analyze operand distributions across oracle-derived GB10 Mamba trace fixtures."
    )
    parser.add_argument(
        "--artifacts-dir",
        default=str(root / "artifacts" / "benchmarks"),
    )
    parser.add_argument(
        "--oracle-dir",
        default=str(root / "testing" / "oracle"),
    )
    parser.add_argument(
        "--target-fixture",
        default="mamba_layer0_target_chat_trace_markdownish",
        help="Fixture to compare against the oracle-derived family median.",
    )
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def load_benchmark_payload(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if "results" not in payload or "environment" not in payload:
        raise ValueError(f"{path} did not look like a benchmark artifact")
    return payload


def discover_latest_oracle_artifacts(artifacts_dir: Path) -> dict[str, Path]:
    by_fixture: dict[str, Path] = {}
    for path in sorted(artifacts_dir.glob("gb10_mamba_cache_*_cuda132.json")):
        try:
            payload = load_benchmark_payload(path)
        except Exception:
            continue
        rows = [
            row
            for row in payload["results"]
            if row.get("operation") == "fixture_trace" and row.get("trace_phase_name") and row.get("fixture_name")
        ]
        if not rows:
            continue
        fixture_names = sorted({str(row["fixture_name"]) for row in rows})
        if len(fixture_names) != 1:
            continue
        phase_end_steps = payload["environment"].get("fixture_trace_phase_end_steps") or []
        if not phase_end_steps or max(int(step) for step in phase_end_steps) <= 704:
            continue
        fixture_name = fixture_names[0]
        current = by_fixture.get(fixture_name)
        if current is None or path.name > current.name:
            by_fixture[fixture_name] = path
    if not by_fixture:
        raise FileNotFoundError("no oracle-derived fixture_trace artifacts found")
    return by_fixture


def load_tensor(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    return data.reshape(shape)


def abs_stats(values: np.ndarray) -> dict[str, float]:
    abs_values = np.abs(values.astype(np.float64, copy=False))
    return {
        "abs_mean": float(abs_values.mean()),
        "abs_p99": float(np.percentile(abs_values, 99.0)),
        "abs_max": float(abs_values.max()),
        "rms": float(np.sqrt(np.mean(np.square(values.astype(np.float64, copy=False))))),
    }


def load_fixture_operands(root: Path) -> dict[str, Any]:
    metadata = load_json(root / "metadata.json")
    trace_step_count = int(metadata["trace_step_count"])
    batch_size = int(metadata["batch_size"])
    num_heads = int(metadata["num_heads"])
    head_dim = int(metadata["head_dim"])
    state_size = int(metadata["state_size"])

    return {
        "metadata": metadata,
        "hidden": load_tensor(
            root / "hidden_trace_fp32.bin", (trace_step_count, batch_size, num_heads, head_dim)
        ),
        "dt": load_tensor(
            root / "dt_trace_fp32.bin", (trace_step_count, batch_size, num_heads, head_dim)
        ),
        "B": load_tensor(
            root / "B_trace_fp32.bin", (trace_step_count, batch_size, num_heads, state_size)
        ),
        "C": load_tensor(
            root / "C_trace_fp32.bin", (trace_step_count, batch_size, num_heads, state_size)
        ),
    }


def phase_rows_by_key(payload: dict[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in payload["results"]:
        if row.get("operation") != "fixture_trace":
            continue
        phase_name = row.get("trace_phase_name")
        if not phase_name:
            continue
        key = (str(row.get("format")), int(row.get("active_requests")), str(phase_name))
        by_key[key] = row
    return by_key


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# GB10 Mamba Trace Operand Analysis")
    lines.append("")
    lines.append(f"Target fixture: `{report['target_fixture']}`")
    lines.append("")
    lines.append("## Profiles")
    lines.append("")
    lines.append("| Fixture | Artifact | Shared Root Tokens | Committed Head Tokens | Trace Steps |")
    lines.append("|---|---|---:|---:|---:|")
    for item in report["profiles"]:
        lines.append(
            f"| `{item['fixture_name']}` | `{item['artifact']}` | `{item['actual_shared_system_root_tokens']}` | "
            f"`{item['actual_committed_head_tokens']}` | `{item['trace_step_count']}` |"
        )

    lines.append("")
    lines.append("## Weakest Drift Rows")
    lines.append("")
    lines.append(
        "| Requests | Phase | Fixture | `fp16/fp16_sr` | `fp16` Mean | `fp16_sr` Mean | `fp16_sr` Stddev | `dt` Mean | `dt` P99 | `hidden` RMS | `C` P99 |"
    )
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in report["weakest_rows"]:
        lines.append(
            f"| `{row['active_requests']}` | `{row['trace_phase_name']}` | `{row['fixture_name']}` | "
            f"`{row['fp16_over_fp16_sr_ratio']}` | `{row['fp16_mean_output_abs_diff']}` | "
            f"`{row['fp16_sr_mean_output_abs_diff']}` | `{row['fp16_sr_stddev_mean_output_abs_diff']}` | "
            f"`{row['dt_abs_mean']}` | `{row['dt_abs_p99']}` | `{row['hidden_rms']}` | `{row['C_abs_p99']}` |"
        )

    lines.append("")
    lines.append("## Target Fixture Vs Family Median")
    lines.append("")
    lines.append(
        "| Phase | Target `fp16/fp16_sr` | Family Median `fp16/fp16_sr` | Target `dt` Mean | Family Median `dt` Mean | Target `dt` P99 | Family Median `dt` P99 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in report["target_vs_family"]:
        lines.append(
            f"| `{row['trace_phase_name']}` | `{row['target_ratio']}` | `{row['family_ratio_median']}` | "
            f"`{row['target_dt_abs_mean']}` | `{row['family_dt_abs_mean_median']}` | "
            f"`{row['target_dt_abs_p99']}` | `{row['family_dt_abs_p99_median']}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    oracle_dir = Path(args.oracle_dir)

    latest = discover_latest_oracle_artifacts(artifacts_dir)
    profiles: list[dict[str, Any]] = []
    weakest_rows: list[dict[str, Any]] = []
    phase_grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}

    for fixture_name, artifact_path in sorted(latest.items()):
        fixture_root = oracle_dir / fixture_name
        fixture = load_fixture_operands(fixture_root)
        metadata = fixture["metadata"]
        benchmark = load_benchmark_payload(artifact_path)
        benchmark_rows = phase_rows_by_key(benchmark)

        profiles.append(
            {
                "fixture_name": fixture_name,
                "artifact": artifact_path.name,
                "actual_shared_system_root_tokens": int(metadata["actual_shared_system_root_tokens"]),
                "actual_committed_head_tokens": int(metadata["actual_committed_head_tokens"]),
                "trace_step_count": int(metadata["trace_step_count"]),
            }
        )

        offset = 0
        for phase in metadata["trace_phases"]:
            phase_name = str(phase["name"])
            phase_kind = str(phase["kind"])
            length = int(phase["length"])
            phase_slice = slice(offset, offset + length)
            offset += length

            stats = {
                "hidden": abs_stats(fixture["hidden"][phase_slice]),
                "dt": abs_stats(fixture["dt"][phase_slice]),
                "B": abs_stats(fixture["B"][phase_slice]),
                "C": abs_stats(fixture["C"][phase_slice]),
            }

            for active_requests in (1, 8):
                fp16 = benchmark_rows.get(("fp16", active_requests, phase_name))
                fp16_sr = benchmark_rows.get(("fp16_sr", active_requests, phase_name))
                if fp16 is None or fp16_sr is None:
                    continue
                ratio = float(fp16["mean_output_abs_diff"]) / float(fp16_sr["mean_output_abs_diff"])
                row = {
                    "fixture_name": fixture_name,
                    "trace_phase_name": phase_name,
                    "trace_phase_kind": phase_kind,
                    "active_requests": active_requests,
                    "fp16_over_fp16_sr_ratio": round(ratio, 6),
                    "fp16_mean_output_abs_diff": round(float(fp16["mean_output_abs_diff"]), 9),
                    "fp16_sr_mean_output_abs_diff": round(float(fp16_sr["mean_output_abs_diff"]), 9),
                    "fp16_sr_stddev_mean_output_abs_diff": round(
                        float(fp16_sr.get("stddev_mean_output_abs_diff") or 0.0), 9
                    ),
                    "hidden_abs_mean": round(stats["hidden"]["abs_mean"], 9),
                    "hidden_abs_p99": round(stats["hidden"]["abs_p99"], 9),
                    "hidden_abs_max": round(stats["hidden"]["abs_max"], 9),
                    "hidden_rms": round(stats["hidden"]["rms"], 9),
                    "dt_abs_mean": round(stats["dt"]["abs_mean"], 9),
                    "dt_abs_p99": round(stats["dt"]["abs_p99"], 9),
                    "dt_abs_max": round(stats["dt"]["abs_max"], 9),
                    "dt_rms": round(stats["dt"]["rms"], 9),
                    "B_abs_mean": round(stats["B"]["abs_mean"], 9),
                    "B_abs_p99": round(stats["B"]["abs_p99"], 9),
                    "C_abs_mean": round(stats["C"]["abs_mean"], 9),
                    "C_abs_p99": round(stats["C"]["abs_p99"], 9),
                }
                weakest_rows.append(row)
                phase_grouped.setdefault((phase_name, active_requests), []).append(row)

    weakest_rows.sort(key=lambda row: (row["fp16_over_fp16_sr_ratio"], row["active_requests"], row["trace_phase_name"], row["fixture_name"]))

    target_vs_family: list[dict[str, Any]] = []
    for (phase_name, active_requests), rows in sorted(phase_grouped.items()):
        target = next((row for row in rows if row["fixture_name"] == args.target_fixture), None)
        if target is None:
            continue
        target_vs_family.append(
            {
                "trace_phase_name": phase_name,
                "active_requests": active_requests,
                "target_ratio": target["fp16_over_fp16_sr_ratio"],
                "family_ratio_median": round(float(np.median([row["fp16_over_fp16_sr_ratio"] for row in rows])), 6),
                "target_dt_abs_mean": target["dt_abs_mean"],
                "family_dt_abs_mean_median": round(float(np.median([row["dt_abs_mean"] for row in rows])), 9),
                "target_dt_abs_p99": target["dt_abs_p99"],
                "family_dt_abs_p99_median": round(float(np.median([row["dt_abs_p99"] for row in rows])), 9),
            }
        )

    report = {
        "target_fixture": args.target_fixture,
        "profiles": profiles,
        "weakest_rows": weakest_rows[:16],
        "target_vs_family": target_vs_family,
    }

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        output_path = Path(args.markdown_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps({"profile_count": len(profiles), "weakest_ratio": weakest_rows[0]["fp16_over_fp16_sr_ratio"]}, indent=2))


if __name__ == "__main__":
    main()
