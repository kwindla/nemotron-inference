#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    artifacts_dir = root / "artifacts" / "benchmarks"
    parser = argparse.ArgumentParser(
        description="Compare two shared-prefill benchmark artifacts."
    )
    parser.add_argument("--artifacts-dir", default=str(artifacts_dir))
    parser.add_argument("--baseline-json")
    parser.add_argument("--current-json")
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    if "results" not in payload or not isinstance(payload["results"], list):
        raise ValueError(f"{path} did not contain a benchmark results list")
    return payload


def find_latest_artifacts(artifacts_dir: Path) -> tuple[Path, Path]:
    matches = sorted(artifacts_dir.glob("nano_shared_prefill_*.json"))
    valid_matches: list[Path] = []
    for path in matches:
        try:
            load_json(path)
        except Exception:
            continue
        valid_matches.append(path)
    if len(valid_matches) < 2:
        raise FileNotFoundError(
            f"need at least two valid nano_shared_prefill_*.json artifacts under {artifacts_dir}"
        )
    return valid_matches[-2], valid_matches[-1]


def safe_ratio(baseline: float | None, current: float | None) -> float | None:
    if baseline is None or current is None or current <= 0.0:
        return None
    return baseline / current


def round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def key_for_result(result: dict[str, Any]) -> int:
    return int(result["prompt_token_count"])


def bucket_label(result: dict[str, Any]) -> str | None:
    token_buckets = result.get("shared_profile_debug", {}).get("token_buckets", [])
    if not token_buckets:
        return None
    labels = []
    for bucket in token_buckets:
        upper = bucket.get("bucket_upper")
        upper_text = "inf" if upper is None else str(upper)
        labels.append(
            f"{bucket.get('tensor')}:{bucket.get('bucket_lower')}-{upper_text}:{bucket.get('profile')}"
        )
    return ",".join(labels)


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Shared Prefill Benchmark Comparison")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Baseline artifact: `{report['baseline_artifact']}`")
    lines.append(f"- Current artifact: `{report['current_artifact']}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    summary = report["summary"]
    lines.append(f"- Matched prompt lengths: `{summary['matched_lengths']}`")
    lines.append(f"- Current faster on hot prefill: `{summary['hot_prefill_win_count']}` / `{summary['matched_lengths']}`")
    lines.append(f"- Current faster on cold prefill: `{summary['cold_prefill_win_count']}` / `{summary['matched_lengths']}`")
    lines.append(
        f"- Current faster on hot shared-kernel time: `{summary['shared_kernel_win_count']}` / `{summary['shared_kernel_comparisons']}`"
    )
    lines.append(
        f"- Best hot-prefill speedup: `{summary['best_hot_prefill_length']}` tokens = `{summary['best_hot_prefill_speedup']}`"
    )
    lines.append("")
    lines.append("## By Prompt Length")
    lines.append("")
    lines.append(
        "| Prompt Tokens | Baseline Cold ms | Current Cold ms | Cold Speedup | Baseline Hot ms | Current Hot ms | Hot Speedup | Baseline Shared ms | Current Shared ms | Shared Speedup | Baseline Bucket | Current Bucket |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for item in report["comparisons"]:
        lines.append(
            f"| `{item['prompt_token_count']}` | "
            f"`{item['baseline_cold_prefill_ms']}` | `{item['current_cold_prefill_ms']}` | "
            f"`{item['cold_prefill_speedup']}` | `{item['baseline_hot_prefill_mean_ms']}` | "
            f"`{item['current_hot_prefill_mean_ms']}` | `{item['hot_prefill_speedup']}` | "
            f"`{item['baseline_hot_shared_kernel_ms']}` | `{item['current_hot_shared_kernel_ms']}` | "
            f"`{item['hot_shared_kernel_speedup']}` | "
            f"`{item['baseline_bucket'] or 'n/a'}` | `{item['current_bucket'] or 'n/a'}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    if args.baseline_json and args.current_json:
        baseline_path = Path(args.baseline_json)
        current_path = Path(args.current_json)
    else:
        baseline_path, current_path = find_latest_artifacts(artifacts_dir)

    baseline_payload = load_json(baseline_path)
    current_payload = load_json(current_path)

    baseline_results = {key_for_result(result): result for result in baseline_payload["results"]}
    current_results = {key_for_result(result): result for result in current_payload["results"]}
    matched_lengths = sorted(set(baseline_results) & set(current_results))

    comparisons: list[dict[str, Any]] = []
    hot_prefill_win_count = 0
    cold_prefill_win_count = 0
    shared_kernel_win_count = 0
    shared_kernel_comparisons = 0

    for length in matched_lengths:
        baseline = baseline_results[length]
        current = current_results[length]
        baseline_cold_prefill_ms = float(baseline["cold_prefill_ms"])
        current_cold_prefill_ms = float(current["cold_prefill_ms"])
        baseline_hot_prefill_mean_ms = float(baseline["hot_prefill_mean_ms"])
        current_hot_prefill_mean_ms = float(current["hot_prefill_mean_ms"])
        baseline_hot_shared_kernel_ms = baseline.get("hot_shared_kernel_ms")
        current_hot_shared_kernel_ms = current.get("hot_shared_kernel_ms")

        cold_speedup = safe_ratio(baseline_cold_prefill_ms, current_cold_prefill_ms)
        hot_speedup = safe_ratio(baseline_hot_prefill_mean_ms, current_hot_prefill_mean_ms)
        shared_speedup = safe_ratio(
            None if baseline_hot_shared_kernel_ms is None else float(baseline_hot_shared_kernel_ms),
            None if current_hot_shared_kernel_ms is None else float(current_hot_shared_kernel_ms),
        )

        if current_cold_prefill_ms < baseline_cold_prefill_ms:
            cold_prefill_win_count += 1
        if current_hot_prefill_mean_ms < baseline_hot_prefill_mean_ms:
            hot_prefill_win_count += 1
        if shared_speedup is not None:
            shared_kernel_comparisons += 1
            if float(current_hot_shared_kernel_ms) < float(baseline_hot_shared_kernel_ms):
                shared_kernel_win_count += 1

        comparisons.append(
            {
                "prompt_token_count": length,
                "baseline_cold_prefill_ms": round(baseline_cold_prefill_ms, 6),
                "current_cold_prefill_ms": round(current_cold_prefill_ms, 6),
                "cold_prefill_speedup": round_or_none(cold_speedup),
                "baseline_hot_prefill_mean_ms": round(baseline_hot_prefill_mean_ms, 6),
                "current_hot_prefill_mean_ms": round(current_hot_prefill_mean_ms, 6),
                "hot_prefill_speedup": round_or_none(hot_speedup),
                "baseline_hot_shared_kernel_ms": round_or_none(
                    None if baseline_hot_shared_kernel_ms is None else float(baseline_hot_shared_kernel_ms)
                ),
                "current_hot_shared_kernel_ms": round_or_none(
                    None if current_hot_shared_kernel_ms is None else float(current_hot_shared_kernel_ms)
                ),
                "hot_shared_kernel_speedup": round_or_none(shared_speedup),
                "baseline_bucket": bucket_label(baseline),
                "current_bucket": bucket_label(current),
            }
        )

    best_hot_prefill = max(
        comparisons,
        key=lambda item: -1.0 if item["hot_prefill_speedup"] is None else item["hot_prefill_speedup"],
        default=None,
    )

    report = {
        "baseline_artifact": baseline_path.name,
        "current_artifact": current_path.name,
        "summary": {
            "matched_lengths": len(matched_lengths),
            "hot_prefill_win_count": hot_prefill_win_count,
            "cold_prefill_win_count": cold_prefill_win_count,
            "shared_kernel_win_count": shared_kernel_win_count,
            "shared_kernel_comparisons": shared_kernel_comparisons,
            "best_hot_prefill_length": None if best_hot_prefill is None else best_hot_prefill["prompt_token_count"],
            "best_hot_prefill_speedup": None if best_hot_prefill is None else best_hot_prefill["hot_prefill_speedup"],
        },
        "comparisons": comparisons,
    }

    if args.json_output:
        Path(args.json_output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.markdown_output:
        Path(args.markdown_output).write_text(render_markdown(report), encoding="utf-8")
    if not args.json_output and not args.markdown_output:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
