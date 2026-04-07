#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Bucket first-user-tail dt magnitudes across the oracle-derived GB10 Mamba trace family."
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
        "--bucket-percentiles",
        default="50,75,90",
        help="Comma-separated family percentiles used to split per-step dt abs-mean into buckets.",
    )
    parser.add_argument(
        "--target-fixture",
        default="",
        help="Fixture to highlight against the family median. Defaults to the headerless control if present, otherwise markdownish.",
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


def parse_percentiles(text: str) -> list[float]:
    values: list[float] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if value <= 0.0 or value >= 100.0:
            raise ValueError("bucket percentiles must be between 0 and 100")
        values.append(value)
    values = sorted(set(values))
    if not values:
        raise ValueError("at least one bucket percentile is required")
    return values


def choose_target_fixture(latest: dict[str, Path], requested: str) -> str:
    if requested:
        return requested
    if "mamba_layer0_target_chat_trace_markdown_headerless" in latest:
        return "mamba_layer0_target_chat_trace_markdown_headerless"
    if "mamba_layer0_target_chat_trace_markdownish" in latest:
        return "mamba_layer0_target_chat_trace_markdownish"
    return sorted(latest.keys())[0]


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
    lines.append("# GB10 Mamba dt Bucket Analysis")
    lines.append("")
    lines.append(f"Phase: `{report['phase_name']}`")
    lines.append("")
    lines.append(f"Target fixture: `{report['target_fixture']}`")
    lines.append("")
    lines.append("## Family Bucket Thresholds")
    lines.append("")
    lines.append("| Bucket | Definition |")
    lines.append("|---|---|")
    for bucket in report["bucket_thresholds"]:
        lines.append(f"| `{bucket['label']}` | `{bucket['definition']}` |")

    lines.append("")
    lines.append("## Phase Summary")
    lines.append("")
    lines.append(
        "| Fixture | Artifact | Steps | `fp16/fp16_sr` @1 | `fp16/fp16_sr` @8 | Phase `dt` Mean | `>p75` Share | `>p90` Share |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in report["profiles"]:
        lines.append(
            f"| `{row['fixture_name']}` | `{row['artifact']}` | `{row['phase_length']}` | "
            f"`{row['ratio_requests_1']}` | `{row['ratio_requests_8']}` | "
            f"`{row['phase_dt_abs_mean']}` | `{row['share_above_p75']}` | `{row['share_above_p90']}` |"
        )

    lines.append("")
    lines.append("## Per-Bucket Step Share")
    lines.append("")
    header = "| Fixture | " + " | ".join(f"`{bucket['label']}`" for bucket in report["bucket_thresholds"]) + " |"
    divider = "|---|" + "|".join(["---:"] * len(report["bucket_thresholds"])) + "|"
    lines.append(header)
    lines.append(divider)
    for row in report["profiles"]:
        shares = " | ".join(f"`{value}`" for value in row["bucket_shares"])
        lines.append(f"| `{row['fixture_name']}` | {shares} |")

    lines.append("")
    lines.append("## Target Fixture Vs Family Median")
    lines.append("")
    lines.append("| Metric | Target | Family Median |")
    lines.append("|---|---:|---:|")
    for key, value in report["target_vs_family"].items():
        lines.append(f"| `{key}` | `{value['target']}` | `{value['family_median']}` |")

    lines.append("")
    lines.append("## Correlation Snapshot")
    lines.append("")
    lines.append("| Relationship | Pearson r |")
    lines.append("|---|---:|")
    for row in report["correlations"]:
        lines.append(f"| `{row['name']}` | `{row['pearson_r']}` |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    oracle_dir = Path(args.oracle_dir)
    latest = discover_latest_oracle_artifacts(artifacts_dir)
    target_fixture = choose_target_fixture(latest, args.target_fixture)
    percentiles = parse_percentiles(args.bucket_percentiles)

    fixture_rows: list[dict[str, Any]] = []
    all_step_dt_means: list[np.ndarray] = []

    for fixture_name, artifact_path in sorted(latest.items()):
        fixture_root = oracle_dir / fixture_name
        metadata = load_json(fixture_root / "metadata.json")
        trace_step_count = int(metadata["trace_step_count"])
        batch_size = int(metadata["batch_size"])
        num_heads = int(metadata["num_heads"])
        head_dim = int(metadata["head_dim"])

        dt = load_tensor(fixture_root / "dt_trace_fp32.bin", (trace_step_count, batch_size, num_heads, head_dim))

        offset = 0
        target_phase = None
        for phase in metadata["trace_phases"]:
            length = int(phase["length"])
            if str(phase["name"]) == args.phase_name:
                target_phase = {
                    "length": length,
                    "slice": slice(offset, offset + length),
                }
                break
            offset += length
        if target_phase is None:
            continue

        phase_dt = dt[target_phase["slice"]]
        step_dt_mean = np.abs(phase_dt.astype(np.float64, copy=False)).mean(axis=(1, 2, 3))
        all_step_dt_means.append(step_dt_mean)

        benchmark = load_benchmark_payload(artifact_path)
        rows = phase_rows_by_key(benchmark)
        fp16_1 = rows[("fp16", 1, args.phase_name)]
        fp16_sr_1 = rows[("fp16_sr", 1, args.phase_name)]
        fp16_8 = rows[("fp16", 8, args.phase_name)]
        fp16_sr_8 = rows[("fp16_sr", 8, args.phase_name)]

        fixture_rows.append(
            {
                "fixture_name": fixture_name,
                "artifact": artifact_path.name,
                "phase_length": int(target_phase["length"]),
                "step_dt_mean": step_dt_mean,
                "ratio_requests_1": float(fp16_1["mean_output_abs_diff"]) / float(fp16_sr_1["mean_output_abs_diff"]),
                "ratio_requests_8": float(fp16_8["mean_output_abs_diff"]) / float(fp16_sr_8["mean_output_abs_diff"]),
                "phase_dt_abs_mean": float(step_dt_mean.mean()),
            }
        )

    if not fixture_rows:
        raise RuntimeError(f"phase {args.phase_name!r} not found in the oracle-derived fixture family")

    family_values = np.concatenate(all_step_dt_means)
    threshold_values = [float(np.percentile(family_values, percentile)) for percentile in percentiles]
    threshold_labels: list[dict[str, str]] = []
    previous = None
    for percentile, threshold in zip(percentiles, threshold_values):
        if previous is None:
            label = f"<=p{int(percentile)}"
            definition = f"`dt abs mean <= {threshold:.9f}`"
        else:
            label = f"p{int(previous)}-p{int(percentile)}"
            definition = f"`{previous:.0f}th < dt abs mean <= {threshold:.9f}`"
        threshold_labels.append({"label": label, "definition": definition})
        previous = percentile
    threshold_labels.append(
        {
            "label": f">p{int(percentiles[-1])}",
            "definition": f"`dt abs mean > {threshold_values[-1]:.9f}`",
        }
    )

    threshold_for_p75 = threshold_values[percentiles.index(75.0)] if 75.0 in percentiles else None
    threshold_for_p90 = threshold_values[percentiles.index(90.0)] if 90.0 in percentiles else threshold_values[-1]

    for row in fixture_rows:
        values = row["step_dt_mean"]
        indices = np.searchsorted(threshold_values, values, side="right")
        bucket_counts = np.bincount(indices, minlength=len(threshold_labels))
        row["bucket_shares"] = [round(float(count / len(values)), 6) for count in bucket_counts.tolist()]
        row["share_above_p75"] = round(float((values > threshold_for_p75).mean()), 6) if threshold_for_p75 is not None else None
        row["share_above_p90"] = round(float((values > threshold_for_p90).mean()), 6)
        row["ratio_requests_1"] = round(row["ratio_requests_1"], 6)
        row["ratio_requests_8"] = round(row["ratio_requests_8"], 6)
        row["phase_dt_abs_mean"] = round(row["phase_dt_abs_mean"], 9)
        del row["step_dt_mean"]

    fixture_rows.sort(key=lambda row: (row["ratio_requests_1"], row["ratio_requests_8"], row["fixture_name"]))

    target_row = next((row for row in fixture_rows if row["fixture_name"] == target_fixture), None)
    if target_row is None:
        raise RuntimeError(f"target fixture {target_fixture!r} not present in the discovered family")

    def median_for(key: str) -> float:
        return float(np.median([float(row[key]) for row in fixture_rows]))

    target_vs_family = {
        "ratio_requests_1": {"target": target_row["ratio_requests_1"], "family_median": round_or_none(median_for("ratio_requests_1"))},
        "ratio_requests_8": {"target": target_row["ratio_requests_8"], "family_median": round_or_none(median_for("ratio_requests_8"))},
        "phase_dt_abs_mean": {"target": target_row["phase_dt_abs_mean"], "family_median": round_or_none(median_for("phase_dt_abs_mean"), 9)},
        "share_above_p75": {"target": target_row["share_above_p75"], "family_median": round_or_none(median_for("share_above_p75"))},
        "share_above_p90": {"target": target_row["share_above_p90"], "family_median": round_or_none(median_for("share_above_p90"))},
    }

    correlations = []
    share_above_p90 = [float(row["share_above_p90"]) for row in fixture_rows]
    share_above_p75 = [float(row["share_above_p75"]) for row in fixture_rows if row["share_above_p75"] is not None]
    ratio_requests_1 = [float(row["ratio_requests_1"]) for row in fixture_rows]
    ratio_requests_8 = [float(row["ratio_requests_8"]) for row in fixture_rows]
    phase_dt_means = [float(row["phase_dt_abs_mean"]) for row in fixture_rows]
    correlations.append(
        {
            "name": "ratio@1 vs phase_dt_abs_mean",
            "pearson_r": round_or_none(pearson(ratio_requests_1, phase_dt_means)),
        }
    )
    correlations.append(
        {
            "name": "ratio@1 vs share_above_p90",
            "pearson_r": round_or_none(pearson(ratio_requests_1, share_above_p90)),
        }
    )
    if share_above_p75:
        correlations.append(
            {
                "name": "ratio@1 vs share_above_p75",
                "pearson_r": round_or_none(pearson(ratio_requests_1, share_above_p75)),
            }
        )
    correlations.append(
        {
            "name": "ratio@8 vs phase_dt_abs_mean",
            "pearson_r": round_or_none(pearson(ratio_requests_8, phase_dt_means)),
        }
    )
    correlations.append(
        {
            "name": "ratio@8 vs share_above_p90",
            "pearson_r": round_or_none(pearson(ratio_requests_8, share_above_p90)),
        }
    )

    report = {
        "phase_name": args.phase_name,
        "target_fixture": target_fixture,
        "bucket_percentiles": percentiles,
        "bucket_thresholds": threshold_labels,
        "profiles": fixture_rows,
        "target_vs_family": target_vs_family,
        "correlations": correlations,
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
