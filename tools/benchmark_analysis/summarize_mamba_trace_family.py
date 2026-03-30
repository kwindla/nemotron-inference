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
        description="Summarize a family of GB10 oracle-derived Mamba fixture_trace artifacts."
    )
    parser.add_argument("--artifacts-dir", default=str(artifacts_dir))
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Artifact input in the form label=/abs/path/to/artifact.json. If omitted, the latest oracle-derived fixture_trace artifact is used for each discovered fixture_name.",
    )
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    if "results" not in payload or not isinstance(payload["results"], list):
        raise ValueError(f"{path} did not contain benchmark results")
    if "environment" not in payload or not isinstance(payload["environment"], dict):
        raise ValueError(f"{path} did not contain benchmark environment metadata")
    return payload


def find_trace_fixture_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        result
        for result in payload["results"]
        if result.get("operation") == "fixture_trace" and result.get("trace_phase_name")
    ]


def find_fixture_name(payload: dict[str, Any]) -> str | None:
    rows = find_trace_fixture_rows(payload)
    names = sorted({str(row.get("fixture_name")) for row in rows if row.get("fixture_name")})
    if len(names) == 1:
        return names[0]
    return None


def is_oracle_fixture_trace(payload: dict[str, Any]) -> bool:
    rows = find_trace_fixture_rows(payload)
    if not rows:
        return False
    phase_steps = payload["environment"].get("fixture_trace_phase_end_steps") or []
    if not phase_steps:
        return False
    return max(int(step) for step in phase_steps) > 704


def discover_artifacts(artifacts_dir: Path) -> list[tuple[str, Path]]:
    by_fixture: dict[str, tuple[str, Path]] = {}
    for path in sorted(artifacts_dir.glob("gb10_mamba_cache_*_cuda132.json")):
        try:
            payload = load_json(path)
        except Exception:
            continue
        if not is_oracle_fixture_trace(payload):
            continue
        fixture_name = find_fixture_name(payload)
        if fixture_name is None:
            continue
        current = by_fixture.get(fixture_name)
        if current is None or path.name > current[0]:
            by_fixture[fixture_name] = (path.name, path)
    return sorted((fixture_name, value[1]) for fixture_name, value in by_fixture.items())


def parse_artifact_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected label=/path/to/artifact.json, got {value!r}")
    label, path_str = value.split("=", 1)
    if not label:
        raise ValueError(f"missing label in {value!r}")
    return label, Path(path_str)


def load_fixture_metadata(root: Path, fixture_name: str) -> dict[str, Any] | None:
    metadata_path = root / "testing" / "oracle" / fixture_name / "metadata.json"
    if not metadata_path.exists():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    payload["_metadata_path"] = str(metadata_path)
    return payload


def metric_key(result: dict[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(result.get("format")),
        int(result.get("active_requests")),
        str(result.get("trace_phase_name")),
        str(result.get("trace_phase_kind")),
    )


def round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# GB10 Oracle-Derived Mamba Trace Family Summary")
    lines.append("")
    lines.append("## Profiles")
    lines.append("")
    lines.append(
        "| Label | Artifact | Fixture | Shared Root Tokens | Committed Head Tokens | Trace Steps | Phase-End Steps |"
    )
    lines.append("|---|---|---|---:|---:|---:|---|")
    for item in report["profiles"]:
        lines.append(
            f"| `{item['label']}` | `{item['artifact']}` | `{item['fixture_name']}` | "
            f"`{item['actual_shared_system_root_tokens']}` | `{item['actual_committed_head_tokens']}` | "
            f"`{item['trace_step_count']}` | `{item['phase_end_steps']}` |"
        )
    lines.append("")
    lines.append("## FP16_SR Mean-Drift Advantage Range")
    lines.append("")
    lines.append(
        "| Requests | Phase | Kind | Min `fp16/fp16_sr` | Min Label | Max `fp16/fp16_sr` | Max Label |"
    )
    lines.append("|---:|---|---|---:|---|---:|---|")
    for item in report["sr_advantage_ranges"]:
        lines.append(
            f"| `{item['active_requests']}` | `{item['trace_phase_name']}` | `{item['trace_phase_kind']}` | "
            f"`{item['min_ratio']}` | `{item['min_label']}` | "
            f"`{item['max_ratio']}` | `{item['max_label']}` |"
        )
    lines.append("")
    lines.append("## Phase Metric Ranges")
    lines.append("")
    lines.append(
        "| Format | Requests | Phase | Mean Min | Mean Min Label | Mean Max | Mean Max Label | Max-Spike Min | Max-Spike Min Label | Max-Spike Max | Max-Spike Max Label |"
    )
    lines.append("|---|---:|---|---:|---|---:|---|---:|---|---:|---|")
    for item in report["phase_metric_ranges"]:
        lines.append(
            f"| `{item['format']}` | `{item['active_requests']}` | `{item['trace_phase_name']}` | "
            f"`{item['min_mean_output_abs_diff']}` | `{item['min_mean_label']}` | "
            f"`{item['max_mean_output_abs_diff']}` | `{item['max_mean_label']}` | "
            f"`{item['min_max_output_abs_diff']}` | `{item['min_max_label']}` | "
            f"`{item['max_max_output_abs_diff']}` | `{item['max_max_label']}` |"
        )
    lines.append("")
    lines.append("## Observed Guardrail Anchors")
    lines.append("")
    lines.append(
        "| Requests | Phase | Kind | Weakest `fp16/fp16_sr` | Weakest Label | `fp16_sr` Mean Ceiling | Mean Ceiling Label | `fp16_sr` Max-Spike Ceiling | Max-Spike Label | `fp16_sr` Mean-Stddev Ceiling | Mean-Stddev Label |"
    )
    lines.append("|---:|---|---|---:|---|---:|---|---:|---|---:|---|")
    for item in report["guardrail_anchors"]:
        lines.append(
            f"| `{item['active_requests']}` | `{item['trace_phase_name']}` | `{item['trace_phase_kind']}` | "
            f"`{item['weakest_fp16_over_fp16_sr_ratio']}` | `{item['weakest_ratio_label']}` | "
            f"`{item['fp16_sr_mean_output_abs_diff_ceiling']}` | `{item['mean_ceiling_label']}` | "
            f"`{item['fp16_sr_max_output_abs_diff_ceiling']}` | `{item['max_ceiling_label']}` | "
            f"`{item['fp16_sr_stddev_mean_output_abs_diff_ceiling']}` | `{item['stddev_ceiling_label']}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    artifacts_dir = Path(args.artifacts_dir)

    artifact_inputs = (
        [parse_artifact_arg(value) for value in args.artifact]
        if args.artifact
        else discover_artifacts(artifacts_dir)
    )
    if not artifact_inputs:
        raise FileNotFoundError("no oracle-derived Mamba fixture_trace artifacts found")

    profiles: list[dict[str, Any]] = []
    phase_metric_values: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    sr_advantage_values: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)

    for label, artifact_path in artifact_inputs:
        payload = load_json(artifact_path)
        fixture_name = find_fixture_name(payload)
        if fixture_name is None:
            raise ValueError(f"{artifact_path} did not contain a single oracle trace fixture name")
        metadata = load_fixture_metadata(root, fixture_name) or {}
        profiles.append(
            {
                "label": label,
                "artifact": artifact_path.name,
                "fixture_name": fixture_name,
                "actual_shared_system_root_tokens": metadata.get("actual_shared_system_root_tokens"),
                "actual_committed_head_tokens": metadata.get("actual_committed_head_tokens"),
                "trace_step_count": metadata.get("trace_step_count"),
                "phase_end_steps": payload["environment"].get("fixture_trace_phase_end_steps"),
            }
        )

        rows = find_trace_fixture_rows(payload)
        by_sr_key: dict[tuple[int, str, str], dict[str, float]] = defaultdict(dict)
        for row in rows:
            key = metric_key(row)
            phase_metric_values[key].append(
                {
                    "label": label,
                    "mean_output_abs_diff": float(row.get("mean_output_abs_diff")),
                    "max_output_abs_diff": float(row.get("max_output_abs_diff")),
                    "stddev_mean_output_abs_diff": float(row.get("stddev_mean_output_abs_diff") or 0.0),
                }
            )
            if row.get("format") in {"fp16", "fp16_sr"}:
                sr_key = (
                    int(row.get("active_requests")),
                    str(row.get("trace_phase_name")),
                    str(row.get("trace_phase_kind")),
                )
                by_sr_key[sr_key][str(row.get("format"))] = float(row.get("mean_output_abs_diff"))

        for sr_key, entry in by_sr_key.items():
            if "fp16" not in entry or "fp16_sr" not in entry:
                continue
            sr_advantage_values[sr_key].append(
                {
                    "label": label,
                    "ratio": safe_div(entry["fp16"], entry["fp16_sr"]),
                }
            )

    phase_metric_ranges: list[dict[str, Any]] = []
    for key, values in sorted(phase_metric_values.items()):
        min_mean = min(values, key=lambda item: item["mean_output_abs_diff"])
        max_mean = max(values, key=lambda item: item["mean_output_abs_diff"])
        min_max = min(values, key=lambda item: item["max_output_abs_diff"])
        max_max = max(values, key=lambda item: item["max_output_abs_diff"])
        phase_metric_ranges.append(
            {
                "format": key[0],
                "active_requests": key[1],
                "trace_phase_name": key[2],
                "trace_phase_kind": key[3],
                "min_mean_output_abs_diff": round(min_mean["mean_output_abs_diff"], 9),
                "min_mean_label": min_mean["label"],
                "max_mean_output_abs_diff": round(max_mean["mean_output_abs_diff"], 9),
                "max_mean_label": max_mean["label"],
                "min_max_output_abs_diff": round(min_max["max_output_abs_diff"], 9),
                "min_max_label": min_max["label"],
                "max_max_output_abs_diff": round(max_max["max_output_abs_diff"], 9),
                "max_max_label": max_max["label"],
            }
        )

    sr_advantage_ranges: list[dict[str, Any]] = []
    for key, values in sorted(sr_advantage_values.items()):
        min_ratio = min(values, key=lambda item: item["ratio"] or float("inf"))
        max_ratio = max(values, key=lambda item: item["ratio"] or float("-inf"))
        sr_advantage_ranges.append(
            {
                "active_requests": key[0],
                "trace_phase_name": key[1],
                "trace_phase_kind": key[2],
                "min_ratio": round_or_none(min_ratio["ratio"], 6),
                "min_label": min_ratio["label"],
                "max_ratio": round_or_none(max_ratio["ratio"], 6),
                "max_label": max_ratio["label"],
            }
        )

    guardrail_anchors: list[dict[str, Any]] = []
    for key, values in sorted(sr_advantage_values.items()):
        active_requests, trace_phase_name, trace_phase_kind = key
        weakest_ratio = min(values, key=lambda item: item["ratio"] or float("inf"))
        fp16_sr_rows = phase_metric_values.get(("fp16_sr", active_requests, trace_phase_name, trace_phase_kind), [])
        if not fp16_sr_rows:
            continue
        mean_ceiling = max(fp16_sr_rows, key=lambda item: item["mean_output_abs_diff"])
        max_ceiling = max(fp16_sr_rows, key=lambda item: item["max_output_abs_diff"])
        stddev_ceiling = max(
            fp16_sr_rows,
            key=lambda item: float(item.get("stddev_mean_output_abs_diff") or 0.0),
        )
        guardrail_anchors.append(
            {
                "active_requests": active_requests,
                "trace_phase_name": trace_phase_name,
                "trace_phase_kind": trace_phase_kind,
                "weakest_fp16_over_fp16_sr_ratio": round_or_none(weakest_ratio["ratio"], 6),
                "weakest_ratio_label": weakest_ratio["label"],
                "fp16_sr_mean_output_abs_diff_ceiling": round_or_none(mean_ceiling["mean_output_abs_diff"], 9),
                "mean_ceiling_label": mean_ceiling["label"],
                "fp16_sr_max_output_abs_diff_ceiling": round_or_none(max_ceiling["max_output_abs_diff"], 9),
                "max_ceiling_label": max_ceiling["label"],
                "fp16_sr_stddev_mean_output_abs_diff_ceiling": round_or_none(
                    float(stddev_ceiling.get("stddev_mean_output_abs_diff") or 0.0), 9
                ),
                "stddev_ceiling_label": stddev_ceiling["label"],
            }
        )

    report = {
        "profiles": profiles,
        "sr_advantage_ranges": sr_advantage_ranges,
        "phase_metric_ranges": phase_metric_ranges,
        "guardrail_anchors": guardrail_anchors,
    }

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        output_path = Path(args.markdown_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps({"profile_count": len(profiles)}, indent=2))


if __name__ == "__main__":
    main()
