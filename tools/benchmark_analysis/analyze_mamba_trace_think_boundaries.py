#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Analyze <think> / </think> boundary tokens across oracle-derived Mamba trace fixtures."
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
        help="User-tail phase whose trailing <think> boundary should be analyzed.",
    )
    parser.add_argument(
        "--window-radius",
        type=int,
        default=2,
        help="How many steps before and after the <think> step to include.",
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


def escape_token(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# GB10 Mamba Think-Boundary Analysis")
    lines.append("")
    lines.append(f"Phase: `{report['phase_name']}`")
    lines.append("")
    lines.append("## Boundary Summary")
    lines.append("")
    lines.append(
        "| Fixture | Artifact | `fp16/fp16_sr` @1 | `fp16/fp16_sr` @8 | `<think>` Step | `<think>` `dt` Mean | `<think>` Global Rank | `</think>` Step | `</think>` `dt` Mean | `</think>` Global Rank |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in report["summary_rows"]:
        lines.append(
            f"| `{item['fixture_name']}` | `{item['artifact']}` | `{item['ratio_requests_1']}` | "
            f"`{item['ratio_requests_8']}` | `{item['think_step']}` | `{item['think_dt_abs_mean']}` | "
            f"`{item['think_global_rank']}` | `{item['end_think_step']}` | `{item['end_think_dt_abs_mean']}` | "
            f"`{item['end_think_global_rank']}` |"
        )

    lines.append("")
    lines.append("## First-User-Tail Boundary Windows")
    lines.append("")
    lines.append(
        "| Fixture | Offset | Step | Token | `dt` Mean | Global Rank | Phase Progress |"
    )
    lines.append("|---|---:|---:|---|---:|---:|---:|")
    for row in report["boundary_windows"]:
        lines.append(
            f"| `{row['fixture_name']}` | `{row['offset']}` | `{row['step']}` | `{row['token_text']}` | "
            f"`{row['dt_abs_mean']}` | `{row['global_rank']}` | `{row['phase_progress']}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    oracle_dir = Path(args.oracle_dir)
    latest = discover_latest_oracle_artifacts(artifacts_dir)

    summary_rows: list[dict[str, Any]] = []
    boundary_windows: list[dict[str, Any]] = []

    for fixture_name, artifact_path in sorted(latest.items()):
        fixture_root = oracle_dir / fixture_name
        metadata = load_json(fixture_root / "metadata.json")
        trace_tokens = load_json(fixture_root / "trace_tokens.json")["tokens"]
        trace_step_count = int(metadata["trace_step_count"])
        batch_size = int(metadata["batch_size"])
        num_heads = int(metadata["num_heads"])
        head_dim = int(metadata["head_dim"])

        dt = load_tensor(fixture_root / "dt_trace_fp32.bin", (trace_step_count, batch_size, num_heads, head_dim))
        step_dt_abs_mean = np.abs(dt.astype(np.float64, copy=False)).mean(axis=(1, 2, 3))

        target_phase = next(
            (phase for phase in metadata["trace_phases"] if str(phase["name"]) == args.phase_name),
            None,
        )
        if target_phase is None:
            continue
        start_step = int(target_phase["start_step"])
        end_step = int(target_phase["end_step"])
        phase_len = end_step - start_step

        think_step = next(
            (
                int(token["step"])
                for token in trace_tokens
                if start_step <= int(token["step"]) < end_step and str(token["text"]) == "<think>"
            ),
            None,
        )
        end_think_step = next(
            (
                int(token["step"])
                for token in trace_tokens
                if end_step <= int(token["step"]) < trace_step_count and str(token["text"]) == "</think>"
            ),
            None,
        )
        if think_step is None or end_think_step is None:
            continue

        benchmark = load_benchmark_payload(artifact_path)
        rows = phase_rows_by_key(benchmark)
        fp16_1 = rows[("fp16", 1, args.phase_name)]
        fp16_sr_1 = rows[("fp16_sr", 1, args.phase_name)]
        fp16_8 = rows[("fp16", 8, args.phase_name)]
        fp16_sr_8 = rows[("fp16_sr", 8, args.phase_name)]

        summary_rows.append(
            {
                "fixture_name": fixture_name,
                "artifact": artifact_path.name,
                "ratio_requests_1": round(float(fp16_1["mean_output_abs_diff"]) / float(fp16_sr_1["mean_output_abs_diff"]), 6),
                "ratio_requests_8": round(float(fp16_8["mean_output_abs_diff"]) / float(fp16_sr_8["mean_output_abs_diff"]), 6),
                "think_step": think_step,
                "think_dt_abs_mean": round(float(step_dt_abs_mean[think_step]), 9),
                "think_global_rank": int(np.sum(step_dt_abs_mean > step_dt_abs_mean[think_step])) + 1,
                "end_think_step": end_think_step,
                "end_think_dt_abs_mean": round(float(step_dt_abs_mean[end_think_step]), 9),
                "end_think_global_rank": int(np.sum(step_dt_abs_mean > step_dt_abs_mean[end_think_step])) + 1,
            }
        )

        for offset in range(-args.window_radius, args.window_radius + 3):
            step = think_step + offset
            if step < start_step or step >= min(trace_step_count, think_step + args.window_radius + 3):
                continue
            token = trace_tokens[step]
            boundary_windows.append(
                {
                    "fixture_name": fixture_name,
                    "offset": offset,
                    "step": step,
                    "token_text": escape_token(str(token["text"])),
                    "dt_abs_mean": round(float(step_dt_abs_mean[step]), 9),
                    "global_rank": int(np.sum(step_dt_abs_mean > step_dt_abs_mean[step])) + 1,
                    "phase_progress": round((step - start_step + 1) / phase_len, 6),
                }
            )

    if not summary_rows:
        raise RuntimeError("no fixtures contained the requested think boundary")

    report = {
        "phase_name": args.phase_name,
        "window_radius": args.window_radius,
        "summary_rows": summary_rows,
        "boundary_windows": boundary_windows,
    }

    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        out = Path(args.markdown_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(report), encoding="utf-8")

    worst = min(summary_rows, key=lambda row: row["ratio_requests_1"])
    print(json.dumps({"profile_count": len(summary_rows), "weakest_ratio_requests_1": worst["ratio_requests_1"]}, indent=2))


if __name__ == "__main__":
    main()
