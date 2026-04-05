#!/usr/bin/env python3
"""Compare two nano_fused_decode benchmark JSON artifacts."""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple


_MISSING = object()


@dataclass(frozen=True)
class MetricSpec:
    name: str
    candidate_paths: Tuple[Tuple[str, ...], ...]
    regression_mode: Optional[str] = None


METRICS: Tuple[MetricSpec, ...] = (
    MetricSpec(
        name="hot_steady_state_mean_ms",
        candidate_paths=(
            ("hot_steady_state_mean_ms",),
            ("benchmark", "hot_steady_state_mean_ms"),
        ),
        regression_mode="increase",
    ),
    MetricSpec(
        name="cold_prefill_ms",
        candidate_paths=(
            ("cold_prefill_ms",),
            ("benchmark", "cold_prefill_ms"),
        ),
        regression_mode="increase",
    ),
    MetricSpec(
        name="steady_state_prefill_ms",
        candidate_paths=(
            ("steady_state_prefill_ms",),
            ("benchmark", "steady_state_prefill_ms"),
        ),
        regression_mode="increase",
    ),
    MetricSpec(
        name="steady_state_generated_tokens_per_second",
        candidate_paths=(
            ("steady_state_generated_tokens_per_second",),
            ("benchmark", "steady_state_generated_tokens_per_second"),
        ),
    ),
    MetricSpec(
        name="full_decode_tokens_per_second",
        candidate_paths=(
            ("full_decode_tokens_per_second",),
            ("benchmark", "full_decode_tokens_per_second"),
        ),
    ),
    MetricSpec(
        name="expert_staging_counters.total_bytes_uploaded",
        candidate_paths=(
            ("expert_staging_counters", "total_bytes_uploaded"),
            ("benchmark", "expert_staging_counters", "total_bytes_uploaded"),
            ("total_bytes_uploaded",),
        ),
    ),
    MetricSpec(
        name="expert_staging_counters.staging_elapsed_us",
        candidate_paths=(
            ("expert_staging_counters", "staging_elapsed_us"),
            ("benchmark", "expert_staging_counters", "staging_elapsed_us"),
            ("staging_elapsed_us",),
        ),
    ),
)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two nano_fused_decode benchmark JSON artifacts."
    )
    parser.add_argument("baseline", help="Path to the baseline JSON artifact")
    parser.add_argument("current", help="Path to the current JSON artifact")
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Percentage regression threshold for timing metrics (default: 10)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit comparison as JSON instead of a table",
    )
    return parser.parse_args(argv)


def load_artifact(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise SystemExit(f"compare_artifacts.py: file not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"compare_artifacts.py: failed to parse JSON {path}: {exc}")


def lookup_path(data: Any, path: Sequence[str]) -> Any:
    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return _MISSING
        current = current[key]
    return current


def coerce_number(value: Any) -> Any:
    if isinstance(value, bool):
        return _MISSING
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not math.isfinite(numeric):
            return _MISSING
        if isinstance(value, int):
            return int(value)
        return numeric
    return _MISSING


def extract_metric(data: Any, spec: MetricSpec) -> Any:
    for path in spec.candidate_paths:
        value = lookup_path(data, path)
        if value is _MISSING:
            continue
        numeric = coerce_number(value)
        if numeric is not _MISSING:
            return numeric
    return _MISSING


def compute_delta(baseline: Any, current: Any) -> Optional[float]:
    if baseline is _MISSING or current is _MISSING:
        return None
    if isinstance(baseline, int) and isinstance(current, int):
        return current - baseline
    return float(current) - float(baseline)


def compute_percent_change(baseline: Any, current: Any) -> Optional[float]:
    if baseline is _MISSING or current is _MISSING:
        return None
    baseline_value = float(baseline)
    current_value = float(current)
    if baseline_value == 0.0:
        if current_value == 0.0:
            return 0.0
        return None
    return ((current_value - baseline_value) / baseline_value) * 100.0


def format_number(value: Any) -> str:
    if value is _MISSING or value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)

    numeric = float(value)
    if numeric == 0.0:
        return "0.00"
    if abs(numeric) >= 1000.0:
        return f"{numeric:.2f}"
    if abs(numeric) >= 1.0:
        return f"{numeric:.2f}"
    return f"{numeric:.6f}"


def format_percent(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.1f}%"


def is_regression(spec: MetricSpec, percent_change: Optional[float], threshold: float) -> bool:
    if spec.regression_mode is None or percent_change is None:
        return False
    if spec.regression_mode == "increase":
        return percent_change > threshold
    if spec.regression_mode == "decrease":
        return percent_change < (-threshold)
    return False


def build_comparisons(
    baseline_data: Any,
    current_data: Any,
    threshold: float,
) -> List[dict]:
    rows: List[dict] = []
    for spec in METRICS:
        baseline_value = extract_metric(baseline_data, spec)
        current_value = extract_metric(current_data, spec)
        delta = compute_delta(baseline_value, current_value)
        percent_change = compute_percent_change(baseline_value, current_value)
        rows.append(
            {
                "metric": spec.name,
                "baseline": None if baseline_value is _MISSING else baseline_value,
                "current": None if current_value is _MISSING else current_value,
                "delta": delta,
                "percent_change": percent_change,
                "regression_checked": spec.regression_mode is not None,
                "regressed": is_regression(spec, percent_change, threshold),
            }
        )
    return rows


def print_table(rows: Sequence[dict]) -> None:
    rendered_rows = []
    for row in rows:
        rendered_rows.append(
            (
                row["metric"],
                format_number(row["baseline"]),
                format_number(row["current"]),
                format_number(row["delta"]),
                format_percent(row["percent_change"]),
            )
        )

    metric_width = max(len("metric"), max(len(row[0]) for row in rendered_rows))
    baseline_width = max(len("baseline"), max(len(row[1]) for row in rendered_rows))
    current_width = max(len("current"), max(len(row[2]) for row in rendered_rows))
    delta_width = max(len("delta"), max(len(row[3]) for row in rendered_rows))
    change_width = max(len("change"), max(len(row[4]) for row in rendered_rows))

    print(
        f"{'metric':<{metric_width}}  "
        f"{'baseline':>{baseline_width}}  "
        f"{'current':>{current_width}}  "
        f"{'delta':>{delta_width}}  "
        f"{'change':>{change_width}}"
    )
    for metric, baseline, current, delta, change in rendered_rows:
        print(
            f"{metric:<{metric_width}}  "
            f"{baseline:>{baseline_width}}  "
            f"{current:>{current_width}}  "
            f"{delta:>{delta_width}}  "
            f"{change:>{change_width}}"
        )


def emit_json(
    baseline_path: Path,
    current_path: Path,
    threshold: float,
    rows: Sequence[dict],
) -> None:
    regressions = [row for row in rows if row["regressed"]]
    payload = {
        "baseline_path": str(baseline_path),
        "current_path": str(current_path),
        "threshold_percent": threshold,
        "regression_detected": bool(regressions),
        "regressions": regressions,
        "comparisons": list(rows),
    }
    json.dump(payload, sys.stdout, indent=2, sort_keys=False)
    sys.stdout.write("\n")


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    if args.threshold < 0:
        raise SystemExit("compare_artifacts.py: --threshold must be non-negative")

    baseline_path = Path(args.baseline)
    current_path = Path(args.current)

    baseline_data = load_artifact(baseline_path)
    current_data = load_artifact(current_path)
    rows = build_comparisons(baseline_data, current_data, args.threshold)

    if args.json_output:
        emit_json(baseline_path, current_path, args.threshold, rows)
    else:
        print_table(rows)

    return 1 if any(row["regressed"] for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
