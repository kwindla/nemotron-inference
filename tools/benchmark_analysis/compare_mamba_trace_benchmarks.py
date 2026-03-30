#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    artifacts_dir = root / "artifacts" / "benchmarks"
    parser = argparse.ArgumentParser(
        description="Compare two GB10 Mamba fixture_trace benchmark artifacts phase by phase."
    )
    parser.add_argument("--artifacts-dir", default=str(artifacts_dir))
    parser.add_argument("--baseline-json")
    parser.add_argument("--candidate-json")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    if "results" not in payload or not isinstance(payload["results"], list):
        raise ValueError(f"{path} did not contain a benchmark results list")
    if "environment" not in payload or not isinstance(payload["environment"], dict):
        raise ValueError(f"{path} did not contain an environment object")
    return payload


def round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


def basename(path: Path) -> str:
    return path.name


def artifact_kind(payload: dict[str, Any]) -> str:
    phase_steps = payload["environment"].get("fixture_trace_phase_end_steps") or []
    if phase_steps and max(int(step) for step in phase_steps) > 704:
        return "longer_trace"
    if phase_steps:
        return "shorter_trace"
    return "unknown"


def has_phase_rows(payload: dict[str, Any]) -> bool:
    for result in payload["results"]:
        if result.get("operation") == "fixture_trace" and result.get("trace_phase_name"):
            return True
    return False


def find_preferred_artifact(artifacts_dir: Path, kind: str) -> Path:
    matches = sorted(artifacts_dir.glob("gb10_mamba_cache_*_cuda132.json"))
    ranked: list[tuple[str, Path]] = []
    for path in matches:
        try:
            payload = load_json(path)
        except Exception:
            continue
        if not has_phase_rows(payload):
            continue
        if artifact_kind(payload) != kind:
            continue
        ranked.append((path.name, path))
    if not ranked:
        raise FileNotFoundError(
            f"no phase-labeled GB10 Mamba trace artifacts of kind {kind!r} under {artifacts_dir}"
        )
    ranked.sort()
    return ranked[-1][1]


def comparison_key(result: dict[str, Any]) -> tuple[Any, ...]:
    return (
        result.get("format"),
        int(result.get("active_requests")),
        result.get("trace_phase_name"),
        result.get("trace_phase_kind"),
    )


def build_phase_rows(payload: dict[str, Any]) -> dict[tuple[Any, ...], dict[str, Any]]:
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for result in payload["results"]:
        if result.get("operation") != "fixture_trace":
            continue
        if not result.get("trace_phase_name"):
            continue
        rows[comparison_key(result)] = result
    return rows


def build_sr_advantage_map(payload: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    grouped: dict[tuple[int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for result in payload["results"]:
        if result.get("operation") != "fixture_trace":
            continue
        phase_name = result.get("trace_phase_name")
        if not phase_name:
            continue
        result_format = result.get("format")
        if result_format not in {"fp16", "fp16_sr"}:
            continue
        key = (int(result.get("active_requests")), phase_name)
        grouped[key][result_format] = result

    advantage: dict[tuple[int, str], dict[str, Any]] = {}
    for key, entry in grouped.items():
        fp16 = entry.get("fp16")
        fp16_sr = entry.get("fp16_sr")
        if fp16 is None or fp16_sr is None:
            continue
        ratio = safe_div(
            float(fp16.get("mean_output_abs_diff")),
            float(fp16_sr.get("mean_output_abs_diff")),
        )
        advantage[key] = {
            "active_requests": key[0],
            "trace_phase_name": key[1],
            "trace_phase_kind": fp16.get("trace_phase_kind"),
            "fp16_mean_output_abs_diff": round(float(fp16.get("mean_output_abs_diff")), 9),
            "fp16_sr_mean_output_abs_diff": round(float(fp16_sr.get("mean_output_abs_diff")), 9),
            "fp16_over_fp16_sr_mean_ratio": round_or_none(ratio, 6),
        }
    return advantage


def format_key(key: tuple[Any, ...]) -> str:
    result_format, active_requests, trace_phase_name, trace_phase_kind = key
    return (
        f"{result_format}:requests={active_requests}:phase={trace_phase_name}:kind={trace_phase_kind}"
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# GB10 Mamba Trace Comparison")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Baseline artifact: `{report['baseline_artifact']}`")
    lines.append(f"- Candidate artifact: `{report['candidate_artifact']}`")
    lines.append(f"- Baseline label: `{report['baseline_label']}`")
    lines.append(f"- Candidate label: `{report['candidate_label']}`")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    baseline_env = report["baseline_environment"]
    candidate_env = report["candidate_environment"]
    lines.append(
        f"- Baseline phase-end steps: `{baseline_env.get('fixture_trace_phase_end_steps')}`"
    )
    lines.append(
        f"- Candidate phase-end steps: `{candidate_env.get('fixture_trace_phase_end_steps')}`"
    )
    lines.append(
        f"- Baseline trace steps: `{baseline_env.get('fixture_trace_steps')}`"
    )
    lines.append(
        f"- Candidate trace steps: `{candidate_env.get('fixture_trace_steps')}`"
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    summary = report["summary"]
    lines.append(f"- Matched phase rows: `{summary['matched_rows']}`")
    lines.append(f"- Baseline-only rows: `{summary['baseline_only_count']}`")
    lines.append(f"- Candidate-only rows: `{summary['candidate_only_count']}`")
    for result_format, metrics in summary["by_format"].items():
        lines.append(
            f"- `{result_format}` candidate mean larger on `{metrics['candidate_mean_gt_baseline_count']}` / `{metrics['matched_rows']}` matched rows"
        )
        lines.append(
            f"- `{result_format}` candidate max larger on `{metrics['candidate_max_gt_baseline_count']}` / `{metrics['matched_rows']}` matched rows"
        )
    lines.append("")
    lines.append("## SR Mean-Drift Advantage")
    lines.append("")
    lines.append(
        "| Requests | Phase | Kind | Baseline `fp16/fp16_sr` | Candidate `fp16/fp16_sr` | Candidate vs Baseline |"
    )
    lines.append("|---:|---|---|---:|---:|---:|")
    for item in report["sr_advantage"]:
        lines.append(
            f"| `{item['active_requests']}` | `{item['trace_phase_name']}` | `{item['trace_phase_kind']}` | "
            f"`{item['baseline_fp16_over_fp16_sr_mean_ratio']}` | "
            f"`{item['candidate_fp16_over_fp16_sr_mean_ratio']}` | "
            f"`{item['candidate_over_baseline_ratio']}` |"
        )
    lines.append("")
    lines.append("## Detailed Phase Comparison")
    lines.append("")
    lines.append(
        "| Format | Requests | Phase | Baseline Steps | Candidate Steps | Baseline Mean | Candidate Mean | Mean Ratio | Baseline Max | Candidate Max | Max Ratio | Baseline Stddev Mean | Candidate Stddev Mean | Baseline Hot ms | Candidate Hot ms |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in report["comparisons"]:
        lines.append(
            f"| `{item['format']}` | `{item['active_requests']}` | `{item['trace_phase_name']}` | "
            f"`{item['baseline_steps']}` | `{item['candidate_steps']}` | "
            f"`{item['baseline_mean_output_abs_diff']}` | `{item['candidate_mean_output_abs_diff']}` | "
            f"`{item['candidate_over_baseline_mean_ratio']}` | "
            f"`{item['baseline_max_output_abs_diff']}` | `{item['candidate_max_output_abs_diff']}` | "
            f"`{item['candidate_over_baseline_max_ratio']}` | "
            f"`{item['baseline_stddev_mean_output_abs_diff']}` | `{item['candidate_stddev_mean_output_abs_diff']}` | "
            f"`{item['baseline_hot_mean_ms']}` | `{item['candidate_hot_mean_ms']}` |"
        )
    if report["baseline_only_rows"]:
        lines.append("")
        lines.append("## Baseline-Only Rows")
        lines.append("")
        for item in report["baseline_only_rows"]:
            lines.append(f"- `{item}`")
    if report["candidate_only_rows"]:
        lines.append("")
        lines.append("## Candidate-Only Rows")
        lines.append("")
        for item in report["candidate_only_rows"]:
            lines.append(f"- `{item}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    baseline_path = (
        Path(args.baseline_json)
        if args.baseline_json
        else find_preferred_artifact(artifacts_dir, "shorter_trace")
    )
    candidate_path = (
        Path(args.candidate_json)
        if args.candidate_json
        else find_preferred_artifact(artifacts_dir, "longer_trace")
    )

    baseline_payload = load_json(baseline_path)
    candidate_payload = load_json(candidate_path)

    baseline_rows = build_phase_rows(baseline_payload)
    candidate_rows = build_phase_rows(candidate_payload)

    matched_keys = sorted(set(baseline_rows) & set(candidate_rows))
    baseline_only_keys = sorted(set(baseline_rows) - set(candidate_rows))
    candidate_only_keys = sorted(set(candidate_rows) - set(baseline_rows))

    comparisons: list[dict[str, Any]] = []
    by_format: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "matched_rows": 0,
            "candidate_mean_gt_baseline_count": 0,
            "candidate_max_gt_baseline_count": 0,
        }
    )

    for key in matched_keys:
        baseline_row = baseline_rows[key]
        candidate_row = candidate_rows[key]

        baseline_mean = float(baseline_row.get("mean_output_abs_diff"))
        candidate_mean = float(candidate_row.get("mean_output_abs_diff"))
        baseline_max = float(baseline_row.get("max_output_abs_diff"))
        candidate_max = float(candidate_row.get("max_output_abs_diff"))
        baseline_stddev_mean = float(baseline_row.get("stddev_mean_output_abs_diff", 0.0))
        candidate_stddev_mean = float(candidate_row.get("stddev_mean_output_abs_diff", 0.0))

        result_format = str(baseline_row.get("format"))
        by_format[result_format]["matched_rows"] += 1
        if candidate_mean > baseline_mean:
            by_format[result_format]["candidate_mean_gt_baseline_count"] += 1
        if candidate_max > baseline_max:
            by_format[result_format]["candidate_max_gt_baseline_count"] += 1

        comparisons.append(
            {
                "format": result_format,
                "active_requests": int(baseline_row.get("active_requests")),
                "trace_phase_name": baseline_row.get("trace_phase_name"),
                "trace_phase_kind": baseline_row.get("trace_phase_kind"),
                "baseline_steps": int(baseline_row.get("fixture_steps")),
                "candidate_steps": int(candidate_row.get("fixture_steps")),
                "baseline_mean_output_abs_diff": round(baseline_mean, 9),
                "candidate_mean_output_abs_diff": round(candidate_mean, 9),
                "candidate_over_baseline_mean_ratio": round_or_none(
                    safe_div(candidate_mean, baseline_mean), 6
                ),
                "baseline_max_output_abs_diff": round(baseline_max, 9),
                "candidate_max_output_abs_diff": round(candidate_max, 9),
                "candidate_over_baseline_max_ratio": round_or_none(
                    safe_div(candidate_max, baseline_max), 6
                ),
                "baseline_stddev_mean_output_abs_diff": round(baseline_stddev_mean, 9),
                "candidate_stddev_mean_output_abs_diff": round(candidate_stddev_mean, 9),
                "baseline_hot_mean_ms": round(float(baseline_row.get("hot_mean_ms")), 6),
                "candidate_hot_mean_ms": round(float(candidate_row.get("hot_mean_ms")), 6),
            }
        )

    baseline_sr_advantage = build_sr_advantage_map(baseline_payload)
    candidate_sr_advantage = build_sr_advantage_map(candidate_payload)
    matched_sr_keys = sorted(set(baseline_sr_advantage) & set(candidate_sr_advantage))
    sr_advantage_rows: list[dict[str, Any]] = []
    for key in matched_sr_keys:
        baseline_item = baseline_sr_advantage[key]
        candidate_item = candidate_sr_advantage[key]
        sr_advantage_rows.append(
            {
                "active_requests": baseline_item["active_requests"],
                "trace_phase_name": baseline_item["trace_phase_name"],
                "trace_phase_kind": baseline_item["trace_phase_kind"],
                "baseline_fp16_over_fp16_sr_mean_ratio": baseline_item[
                    "fp16_over_fp16_sr_mean_ratio"
                ],
                "candidate_fp16_over_fp16_sr_mean_ratio": candidate_item[
                    "fp16_over_fp16_sr_mean_ratio"
                ],
                "candidate_over_baseline_ratio": round_or_none(
                    safe_div(
                        candidate_item["fp16_over_fp16_sr_mean_ratio"],
                        baseline_item["fp16_over_fp16_sr_mean_ratio"],
                    ),
                    6,
                ),
            }
        )

    report = {
        "baseline_artifact": basename(baseline_path),
        "candidate_artifact": basename(candidate_path),
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "baseline_environment": baseline_payload["environment"],
        "candidate_environment": candidate_payload["environment"],
        "summary": {
            "matched_rows": len(matched_keys),
            "baseline_only_count": len(baseline_only_keys),
            "candidate_only_count": len(candidate_only_keys),
            "by_format": by_format,
        },
        "sr_advantage": sr_advantage_rows,
        "comparisons": comparisons,
        "baseline_only_rows": [format_key(key) for key in baseline_only_keys],
        "candidate_only_rows": [format_key(key) for key in candidate_only_keys],
    }

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        output_path = Path(args.markdown_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
