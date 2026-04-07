#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Analyze per-step dt hotspots for an oracle-derived GB10 Mamba trace phase."
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
        "--phase-name",
        default="user_turn_1_tail_prefill",
    )
    parser.add_argument(
        "--head-steps",
        type=int,
        default=64,
        help="How many early phase steps to treat as the head window.",
    )
    parser.add_argument(
        "--target-fixture",
        default="mamba_layer0_target_chat_trace_markdownish",
        help="Fixture to treat as the target profile for hotspot reporting.",
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
    return np.fromfile(path, dtype=np.float32).reshape(shape)


def phase_rows_by_key(payload: dict[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    result: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in payload["results"]:
        if row.get("operation") != "fixture_trace":
            continue
        phase_name = row.get("trace_phase_name")
        if not phase_name:
            continue
        result[(str(row.get("format")), int(row.get("active_requests")), str(phase_name))] = row
    return result


def per_step_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    values64 = values.astype(np.float64, copy=False)
    abs_values = np.abs(values64)
    return {
        "abs_mean": abs_values.mean(axis=tuple(range(1, values.ndim))),
        "abs_p99": np.percentile(abs_values, 99.0, axis=tuple(range(1, values.ndim))),
        "abs_max": abs_values.max(axis=tuple(range(1, values.ndim))),
        "rms": np.sqrt(np.mean(np.square(values64), axis=tuple(range(1, values.ndim)))),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# GB10 Mamba dt Hotspot Analysis")
    lines.append("")
    lines.append(f"Phase: `{report['phase_name']}`")
    lines.append("")
    lines.append(f"Target fixture: `{report['target_fixture']}`")
    lines.append("")
    lines.append("## Profile Summary")
    lines.append("")
    lines.append(
        "| Fixture | Artifact | Steps | `fp16/fp16_sr` @1 | `fp16/fp16_sr` @8 | Phase `dt` Mean | Phase `dt` P99 | Head `dt` Mean | Tail `dt` Mean | Top-8 In Head |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in report["profiles"]:
        lines.append(
            f"| `{item['fixture_name']}` | `{item['artifact']}` | `{item['phase_length']}` | "
            f"`{item['ratio_requests_1']}` | `{item['ratio_requests_8']}` | "
            f"`{item['phase_dt_abs_mean']}` | `{item['phase_dt_abs_p99']}` | "
            f"`{item['head_dt_abs_mean']}` | `{item['tail_dt_abs_mean']}` | "
            f"`{item['top8_steps_in_head']}` |"
        )

    lines.append("")
    lines.append("## Target Fixture Vs Family Median")
    lines.append("")
    lines.append(
        "| Metric | Target Fixture | Family Median |"
    )
    lines.append("|---|---:|---:|")
    for key, value in report["target_vs_family"].items():
        if key == "fixture_name":
            continue
        lines.append(f"| `{key}` | `{value['target']}` | `{value['family_median']}` |")

    lines.append("")
    lines.append("## Target Fixture Top dt Steps")
    lines.append("")
    lines.append(
        "| Phase Step | Phase Progress | `dt` Mean | `dt` P99 | `dt` Max | `hidden` RMS | `C` P99 |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in report["target_top_steps"]:
        lines.append(
            f"| `{row['phase_step_index']}` | `{row['phase_progress']}` | `{row['dt_abs_mean']}` | "
            f"`{row['dt_abs_p99']}` | `{row['dt_abs_max']}` | `{row['hidden_rms']}` | `{row['C_abs_p99']}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    oracle_dir = Path(args.oracle_dir)
    latest = discover_latest_oracle_artifacts(artifacts_dir)

    profile_rows: list[dict[str, Any]] = []
    target_top_steps: list[dict[str, Any]] = []

    for fixture_name, artifact_path in sorted(latest.items()):
        fixture_root = oracle_dir / fixture_name
        metadata = load_json(fixture_root / "metadata.json")
        trace_step_count = int(metadata["trace_step_count"])
        batch_size = int(metadata["batch_size"])
        num_heads = int(metadata["num_heads"])
        head_dim = int(metadata["head_dim"])
        state_size = int(metadata["state_size"])

        hidden = load_tensor(fixture_root / "hidden_trace_fp32.bin", (trace_step_count, batch_size, num_heads, head_dim))
        dt = load_tensor(fixture_root / "dt_trace_fp32.bin", (trace_step_count, batch_size, num_heads, head_dim))
        C = load_tensor(fixture_root / "C_trace_fp32.bin", (trace_step_count, batch_size, num_heads, state_size))

        offset = 0
        target_phase = None
        for phase in metadata["trace_phases"]:
            length = int(phase["length"])
            if str(phase["name"]) == args.phase_name:
                target_phase = {
                    "phase": phase,
                    "slice": slice(offset, offset + length),
                    "length": length,
                }
                break
            offset += length
        if target_phase is None:
            continue

        benchmark = load_benchmark_payload(artifact_path)
        rows = phase_rows_by_key(benchmark)
        fp16_1 = rows[("fp16", 1, args.phase_name)]
        fp16_sr_1 = rows[("fp16_sr", 1, args.phase_name)]
        fp16_8 = rows[("fp16", 8, args.phase_name)]
        fp16_sr_8 = rows[("fp16_sr", 8, args.phase_name)]

        phase_hidden = hidden[target_phase["slice"]]
        phase_dt = dt[target_phase["slice"]]
        phase_C = C[target_phase["slice"]]
        step_hidden = per_step_stats(phase_hidden)
        step_dt = per_step_stats(phase_dt)
        step_C = per_step_stats(phase_C)

        phase_len = target_phase["length"]
        head_len = min(args.head_steps, phase_len)
        head_slice = slice(0, head_len)
        tail_slice = slice(head_len, phase_len)

        phase_row = {
            "fixture_name": fixture_name,
            "artifact": artifact_path.name,
            "phase_length": phase_len,
            "ratio_requests_1": round(float(fp16_1["mean_output_abs_diff"]) / float(fp16_sr_1["mean_output_abs_diff"]), 6),
            "ratio_requests_8": round(float(fp16_8["mean_output_abs_diff"]) / float(fp16_sr_8["mean_output_abs_diff"]), 6),
            "phase_dt_abs_mean": round(float(step_dt["abs_mean"].mean()), 9),
            "phase_dt_abs_p99": round(float(np.percentile(step_dt["abs_p99"], 99.0)), 9),
            "head_dt_abs_mean": round(float(step_dt["abs_mean"][head_slice].mean()), 9),
            "tail_dt_abs_mean": round(float(step_dt["abs_mean"][tail_slice].mean()) if phase_len > head_len else float(step_dt["abs_mean"][head_slice].mean()), 9),
            "head_dt_abs_p99": round(float(np.percentile(step_dt["abs_p99"][head_slice], 99.0)), 9),
            "tail_dt_abs_p99": round(float(np.percentile(step_dt["abs_p99"][tail_slice], 99.0)) if phase_len > head_len else float(np.percentile(step_dt["abs_p99"][head_slice], 99.0)), 9),
            "top8_steps_in_head": 0,
        }

        top_indices = np.argsort(step_dt["abs_mean"])[::-1][: min(8, phase_len)]
        phase_row["top8_steps_in_head"] = int(sum(1 for idx in top_indices if int(idx) < head_len))
        profile_rows.append(phase_row)

        if fixture_name == args.target_fixture:
            for idx in top_indices:
                target_top_steps.append(
                    {
                        "phase_step_index": int(idx),
                        "phase_progress": round((int(idx) + 1) / phase_len, 6),
                        "dt_abs_mean": round(float(step_dt["abs_mean"][idx]), 9),
                        "dt_abs_p99": round(float(step_dt["abs_p99"][idx]), 9),
                        "dt_abs_max": round(float(step_dt["abs_max"][idx]), 9),
                        "hidden_rms": round(float(step_hidden["rms"][idx]), 9),
                        "C_abs_p99": round(float(step_C["abs_p99"][idx]), 9),
                    }
                )

    if not profile_rows:
        raise RuntimeError(f"phase {args.phase_name!r} not found in the oracle-derived fixture family")

    target = next(row for row in profile_rows if row["fixture_name"] == args.target_fixture)
    target_vs_family = {}
    for key in (
        "ratio_requests_1",
        "ratio_requests_8",
        "phase_dt_abs_mean",
        "phase_dt_abs_p99",
        "head_dt_abs_mean",
        "tail_dt_abs_mean",
        "head_dt_abs_p99",
        "tail_dt_abs_p99",
        "top8_steps_in_head",
    ):
        target_vs_family[key] = {
            "target": target[key],
            "family_median": round(float(np.median([row[key] for row in profile_rows])), 9 if isinstance(target[key], float) else 6),
        }

    report = {
        "phase_name": args.phase_name,
        "head_steps": args.head_steps,
        "target_fixture": args.target_fixture,
        "profiles": sorted(profile_rows, key=lambda row: row["ratio_requests_1"]),
        "target_vs_family": target_vs_family,
        "target_top_steps": target_top_steps,
    }

    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        out = Path(args.markdown_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps({"profile_count": len(profile_rows), "weakest_ratio_requests_1": report["profiles"][0]["ratio_requests_1"]}, indent=2))


if __name__ == "__main__":
    main()
