#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a first-user-tail dt-scale sweep over generated Mamba trace fixtures."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input in the form scale=/abs/path/to/artifact.json|/abs/path/to/fixture_root",
    )
    parser.add_argument("--phase-name", default="user_turn_1_tail_prefill")
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    return parser.parse_args()


def parse_input(spec: str) -> tuple[float, Path, Path]:
    if "=" not in spec or "|" not in spec:
        raise ValueError(f"invalid input spec: {spec}")
    scale_text, remainder = spec.split("=", 1)
    artifact_text, fixture_text = remainder.split("|", 1)
    return float(scale_text), Path(artifact_text), Path(fixture_text)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    x_std = float(x_arr.std())
    y_std = float(y_arr.std())
    if x_std == 0.0 or y_std == 0.0:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# GB10 Mamba dt Scale Sweep")
    lines.append("")
    lines.append(f"Phase: `{report['phase_name']}`")
    lines.append("")
    lines.append("## Sweep Summary")
    lines.append("")
    lines.append(
        "| dt Scale | Fixture | Steps | Phase `dt` Mean | `fp16/fp16_sr` @1 | `fp16/fp16_sr` @8 | `fp16` Mean @1 | `fp16_sr` Mean @1 |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in report["rows"]:
        lines.append(
            f"| `{row['dt_scale']}` | `{row['fixture_name']}` | `{row['phase_length']}` | "
            f"`{row['phase_dt_abs_mean']}` | `{row['ratio_requests_1']}` | `{row['ratio_requests_8']}` | "
            f"`{row['fp16_mean_output_abs_diff_1']}` | `{row['fp16_sr_mean_output_abs_diff_1']}` |"
        )

    lines.append("")
    lines.append("## Correlation Snapshot")
    lines.append("")
    lines.append("| Relationship | Pearson r |")
    lines.append("|---|---:|")
    for row in report["correlations"]:
        lines.append(f"| `{row['name']}` | `{row['pearson_r']}` |")

    lines.append("")
    lines.append("## Monotonic Read")
    lines.append("")
    lines.append(f"- `ratio@1` nonincreasing with dt scale: `{report['ratio_1_nonincreasing']}`")
    lines.append(f"- `ratio@8` nonincreasing with dt scale: `{report['ratio_8_nonincreasing']}`")
    lines.append(f"- phase `dt` mean nondecreasing with dt scale: `{report['dt_mean_nondecreasing']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []

    for spec in args.input:
        dt_scale, artifact_path, fixture_root = parse_input(spec)
        artifact = load_json(artifact_path)
        metadata = load_json(fixture_root / "metadata.json")
        phase = next((item for item in metadata["trace_phases"] if item["name"] == args.phase_name), None)
        if phase is None:
            raise ValueError(f"phase {args.phase_name!r} not found in {fixture_root}")
        phase_length = int(phase["length"])

        hidden_count = int(metadata["batch_size"]) * int(metadata["num_heads"]) * int(metadata["head_dim"])
        dt = np.fromfile(fixture_root / "dt_trace_fp32.bin", dtype=np.float32).reshape(phase_length, hidden_count)
        phase_dt_abs_mean = float(np.abs(dt.astype(np.float64, copy=False)).mean())

        benchmark_rows = [
            row for row in artifact["results"]
            if row.get("operation") == "fixture_trace"
            and row.get("fixture_steps") == phase_length
            and row.get("trace_phase_name") == args.phase_name
        ]
        by_key = {(str(row["format"]), int(row["active_requests"])): row for row in benchmark_rows}
        fp16_1 = by_key[("fp16", 1)]
        fp16_sr_1 = by_key[("fp16_sr", 1)]
        fp16_8 = by_key[("fp16", 8)]
        fp16_sr_8 = by_key[("fp16_sr", 8)]

        rows.append(
            {
                "dt_scale": dt_scale,
                "fixture_name": str(fp16_1["fixture_name"]),
                "phase_length": phase_length,
                "phase_dt_abs_mean": round(phase_dt_abs_mean, 9),
                "ratio_requests_1": round(float(fp16_1["mean_output_abs_diff"]) / float(fp16_sr_1["mean_output_abs_diff"]), 6),
                "ratio_requests_8": round(float(fp16_8["mean_output_abs_diff"]) / float(fp16_sr_8["mean_output_abs_diff"]), 6),
                "fp16_mean_output_abs_diff_1": round(float(fp16_1["mean_output_abs_diff"]), 9),
                "fp16_sr_mean_output_abs_diff_1": round(float(fp16_sr_1["mean_output_abs_diff"]), 9),
                "fp16_mean_output_abs_diff_8": round(float(fp16_8["mean_output_abs_diff"]), 9),
                "fp16_sr_mean_output_abs_diff_8": round(float(fp16_sr_8["mean_output_abs_diff"]), 9),
                "artifact": str(artifact_path),
                "fixture_root": str(fixture_root),
            }
        )

    rows.sort(key=lambda row: row["dt_scale"])
    scales = [float(row["dt_scale"]) for row in rows]
    dt_means = [float(row["phase_dt_abs_mean"]) for row in rows]
    ratio_1 = [float(row["ratio_requests_1"]) for row in rows]
    ratio_8 = [float(row["ratio_requests_8"]) for row in rows]

    report = {
        "phase_name": args.phase_name,
        "rows": rows,
        "correlations": [
            {"name": "dt_scale vs phase_dt_abs_mean", "pearson_r": round_or_none(pearson(scales, dt_means))},
            {"name": "dt_scale vs ratio@1", "pearson_r": round_or_none(pearson(scales, ratio_1))},
            {"name": "dt_scale vs ratio@8", "pearson_r": round_or_none(pearson(scales, ratio_8))},
            {"name": "phase_dt_abs_mean vs ratio@1", "pearson_r": round_or_none(pearson(dt_means, ratio_1))},
            {"name": "phase_dt_abs_mean vs ratio@8", "pearson_r": round_or_none(pearson(dt_means, ratio_8))},
        ],
        "ratio_1_nonincreasing": all(ratio_1[i] >= ratio_1[i + 1] for i in range(len(ratio_1) - 1)),
        "ratio_8_nonincreasing": all(ratio_8[i] >= ratio_8[i + 1] for i in range(len(ratio_8) - 1)),
        "dt_mean_nondecreasing": all(dt_means[i] <= dt_means[i + 1] for i in range(len(dt_means) - 1)),
    }

    if args.json_output:
        Path(args.json_output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    if args.markdown_output:
        Path(args.markdown_output).write_text(markdown, encoding="utf-8")
    if not args.json_output and not args.markdown_output:
        print(markdown)


if __name__ == "__main__":
    main()
