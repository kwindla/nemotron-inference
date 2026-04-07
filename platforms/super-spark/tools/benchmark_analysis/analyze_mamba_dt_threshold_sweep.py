#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any


FIRST_PHASE = "user_turn_1_tail_prefill"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a Mamba dt-threshold sweep over the oracle-derived trace family."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input in the form threshold=/abs/path/to/artifact.json",
    )
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def parse_input(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"invalid input spec: {spec}")
    label, path = spec.split("=", 1)
    return label, Path(path)


def phase_ratio_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    fixture_name = None
    for row in payload["results"]:
        if row.get("operation") != "fixture_trace":
            continue
        phase_name = row.get("trace_phase_name")
        if not phase_name:
            continue
        fixture_name = str(row.get("fixture_name"))
        key = (str(row.get("format")), int(row.get("active_requests")), str(phase_name))
        by_key[key] = row
    if fixture_name is None:
        return rows
    phase_names = sorted({key[2] for key in by_key})
    for phase_name in phase_names:
        for active_requests in (1, 8):
            fp16 = by_key.get(("fp16", active_requests, phase_name))
            fp16_sr = by_key.get(("fp16_sr", active_requests, phase_name))
            if fp16 is None or fp16_sr is None:
                continue
            rows.append(
                {
                    "fixture_name": fixture_name,
                    "phase_name": phase_name,
                    "active_requests": active_requests,
                    "ratio": float(fp16["mean_output_abs_diff"]) / float(fp16_sr["mean_output_abs_diff"]),
                    "fp16_mean": float(fp16["mean_output_abs_diff"]),
                    "fp16_sr_mean": float(fp16_sr["mean_output_abs_diff"]),
                }
            )
    return rows


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# GB10 Mamba dt-Threshold Sweep")
    lines.append("")
    lines.append(f"- Threshold labels: `{report['threshold_labels']}`")
    lines.append(f"- Fixture count: `{report['fixture_count']}`")
    lines.append("")
    lines.append("## Threshold Summary")
    lines.append("")
    lines.append(
        "| Threshold | First-Phase Min @1 | First-Phase Min @8 | First-Phase Wins @1 | First-Phase Wins @8 | Worst Later Δ @1 | Worst Later Δ @8 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in report["threshold_summary"]:
        lines.append(
            f"| `{row['threshold']}` | `{row['first_phase_min_ratio_1']}` | `{row['first_phase_min_ratio_8']}` | "
            f"`{row['first_phase_win_count_1']}` | `{row['first_phase_win_count_8']}` | "
            f"`{row['worst_later_delta_1']}` | `{row['worst_later_delta_8']}` |"
        )

    lines.append("")
    lines.append("## Weak-Profile First-Phase Detail")
    lines.append("")
    lines.append(
        "| Threshold | Fixture | Ratio @1 | Ratio @8 | `fp16` Mean @1 | `fp16_sr` Mean @1 |"
    )
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in report["first_phase_detail"]:
        lines.append(
            f"| `{row['threshold']}` | `{row['fixture_name']}` | `{row['ratio_1']}` | `{row['ratio_8']}` | "
            f"`{row['fp16_mean_1']}` | `{row['fp16_sr_mean_1']}` |"
        )

    lines.append("")
    lines.append("## Worst Later-Phase Regressions")
    lines.append("")
    lines.append("| Threshold | Requests | Fixture | Phase | Candidate Ratio | Baseline Ratio | Delta |")
    lines.append("|---|---:|---|---|---:|---:|---:|")
    for row in report["worst_later_regressions"]:
        lines.append(
            f"| `{row['threshold']}` | `{row['active_requests']}` | `{row['fixture_name']}` | `{row['phase_name']}` | "
            f"`{row['candidate_ratio']}` | `{row['baseline_ratio']}` | `{row['delta']}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    payloads: dict[str, list[dict[str, Any]]] = {}
    for spec in args.input:
        label, path = parse_input(spec)
        payloads.setdefault(label, []).append(load_json(path))

    if "inf" not in payloads:
        raise ValueError("threshold label 'inf' must be present as the ungated baseline")

    structured: dict[str, dict[tuple[str, str, int], dict[str, float]]] = {}
    fixture_names: set[str] = set()
    for threshold, payload_list in payloads.items():
        rows_by_key: dict[tuple[str, str, int], dict[str, float]] = {}
        for payload in payload_list:
            for row in phase_ratio_rows(payload):
                key = (row["fixture_name"], row["phase_name"], row["active_requests"])
                rows_by_key[key] = row
                fixture_names.add(row["fixture_name"])
        structured[threshold] = rows_by_key

    threshold_labels = sorted(
        structured.keys(),
        key=lambda item: float("inf") if item == "inf" else float(item),
    )

    weak_fixtures = {
        "mamba_layer0_target_chat_trace_markdown_flat",
        "mamba_layer0_target_chat_trace_markdownish",
        "mamba_layer0_target_chat_trace_markdown_headerless",
    }

    threshold_summary = []
    first_phase_detail = []
    worst_later_regressions = []
    baseline = structured["inf"]

    for threshold in threshold_labels:
        rows = structured[threshold]
        first_phase_rows_1 = [row for key, row in rows.items() if key[1] == FIRST_PHASE and key[2] == 1]
        first_phase_rows_8 = [row for key, row in rows.items() if key[1] == FIRST_PHASE and key[2] == 8]
        first_phase_min_ratio_1 = min((row["ratio"] for row in first_phase_rows_1), default=float("nan"))
        first_phase_min_ratio_8 = min((row["ratio"] for row in first_phase_rows_8), default=float("nan"))
        first_phase_win_count_1 = sum(1 for row in first_phase_rows_1 if row["ratio"] > 1.0)
        first_phase_win_count_8 = sum(1 for row in first_phase_rows_8 if row["ratio"] > 1.0)

        worst_regression: dict[int, tuple[tuple[str, str, int], float]] = {}
        for key, candidate in rows.items():
            fixture_name, phase_name, active_requests = key
            if phase_name == FIRST_PHASE:
                continue
            base = baseline.get(key)
            if base is None:
                continue
            delta = candidate["ratio"] - base["ratio"]
            current = worst_regression.get(active_requests)
            if current is None or delta < current[1]:
                worst_regression[active_requests] = (key, delta)

        threshold_summary.append(
            {
                "threshold": threshold,
                "first_phase_min_ratio_1": round(first_phase_min_ratio_1, 6),
                "first_phase_min_ratio_8": round(first_phase_min_ratio_8, 6),
                "first_phase_win_count_1": first_phase_win_count_1,
                "first_phase_win_count_8": first_phase_win_count_8,
                "worst_later_delta_1": round(worst_regression.get(1, ((None, None, None), 0.0))[1], 6),
                "worst_later_delta_8": round(worst_regression.get(8, ((None, None, None), 0.0))[1], 6),
            }
        )

        for active_requests in (1, 8):
            item = worst_regression.get(active_requests)
            if item is not None:
                key, delta = item
                candidate = rows[key]
                base = baseline[key]
                worst_later_regressions.append(
                    {
                        "threshold": threshold,
                        "active_requests": active_requests,
                        "fixture_name": key[0],
                        "phase_name": key[1],
                        "candidate_ratio": round(candidate["ratio"], 6),
                        "baseline_ratio": round(base["ratio"], 6),
                        "delta": round(delta, 6),
                    }
                )

        for fixture_name in sorted(weak_fixtures):
            row_1 = rows.get((fixture_name, FIRST_PHASE, 1))
            row_8 = rows.get((fixture_name, FIRST_PHASE, 8))
            if row_1 is None or row_8 is None:
                continue
            first_phase_detail.append(
                {
                    "threshold": threshold,
                    "fixture_name": fixture_name,
                    "ratio_1": round(row_1["ratio"], 6),
                    "ratio_8": round(row_8["ratio"], 6),
                    "fp16_mean_1": round(row_1["fp16_mean"], 9),
                    "fp16_sr_mean_1": round(row_1["fp16_sr_mean"], 9),
                }
            )

    report = {
        "threshold_labels": threshold_labels,
        "fixture_count": len(fixture_names),
        "threshold_summary": threshold_summary,
        "first_phase_detail": first_phase_detail,
        "worst_later_regressions": worst_later_regressions,
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
