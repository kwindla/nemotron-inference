#!/usr/bin/env python3

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    artifacts_dir = root / "artifacts" / "benchmarks"
    parser = argparse.ArgumentParser(
        description="Compare GB10 FP32 dense GEMM and NVFP4 GEMM benchmark artifacts."
    )
    parser.add_argument("--artifacts-dir", default=str(artifacts_dir))
    parser.add_argument("--dense-json")
    parser.add_argument("--nvfp4-json")
    parser.add_argument("--activation-staging", default="device", choices=["device", "host"])
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


def find_preferred_artifact(artifacts_dir: Path, pattern: str) -> Path:
    matches = sorted(artifacts_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no artifacts matched {pattern!r} under {artifacts_dir}")
    ranked: list[tuple[int, str, Path]] = []
    for path in matches:
        try:
            results_count = len(load_json(path)["results"])
        except Exception:
            continue
        ranked.append((results_count, path.name, path))
    if not ranked:
        raise ValueError(f"no valid benchmark JSON artifacts matched {pattern!r} under {artifacts_dir}")
    ranked.sort()
    return ranked[-1][2]


def key_for_result(result: dict[str, Any]) -> tuple[Any, ...]:
    return (
        result["case_name"],
        int(result["m"]),
        int(result["n"]),
        int(result["k"]),
        int(result["workspace_bytes"]),
    )


def safe_div(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


def round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def basename(path: Path) -> str:
    return path.name


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# GB10 Dense vs NVFP4 GEMM Comparison")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Dense artifact: `{report['dense_artifact']}`")
    lines.append(f"- NVFP4 artifact: `{report['nvfp4_artifact']}`")
    lines.append(f"- Activation staging: `{report['activation_staging']}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    summary = report["summary"]
    lines.append(f"- Matched comparisons: `{summary['matched_comparisons']}`")
    lines.append(f"- Dense-only keys: `{summary['dense_only_count']}`")
    lines.append(f"- NVFP4-only keys: `{summary['nvfp4_only_count']}`")
    lines.append(
        f"- NVFP4 faster on pure GEMM hot latency: `{summary['nvfp4_compute_win_count']}` / `{summary['matched_comparisons']}`"
    )
    lines.append(
        f"- NVFP4 faster on runtime-facing hot latency: `{summary['nvfp4_service_win_count']}` / `{summary['matched_comparisons']}`"
    )
    lines.append(
        f"- Best compute hot speedup: `{summary['best_compute_speedup_label']}` = `{summary['best_compute_speedup']}`"
    )
    lines.append(
        f"- Best runtime-facing hot speedup: `{summary['best_service_speedup_label']}` = `{summary['best_service_speedup']}`"
    )
    lines.append("")
    lines.append("## Best By Case And Batch")
    lines.append("")
    lines.append(
        "| Case | m | Best Workspace | Dense Hot ms | NVFP4 Hot ms | NVFP4 Runtime Hot ms | Compute Speedup | Runtime Speedup |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for item in report["best_by_case_and_m"]:
        lines.append(
            f"| `{item['case_name']}` | `{item['m']}` | `{item['workspace_bytes']}` | "
            f"`{item['dense_hot_mean_ms']}` | `{item['nvfp4_hot_mean_ms']}` | "
            f"`{item['nvfp4_runtime_hot_ms']}` | `{item['compute_speedup_vs_dense']}` | "
            f"`{item['runtime_speedup_vs_dense']}` |"
        )
    lines.append("")
    lines.append("## Detailed Comparisons")
    lines.append("")
    lines.append(
        "| Case | m | n | k | Workspace | Dense Hot ms | Dense Cold ms | NVFP4 Hot ms | NVFP4 Runtime Hot ms | NVFP4 Cold Runtime ms | Compute Speedup | Runtime Speedup | Pack ms |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in report["comparisons"]:
        lines.append(
            f"| `{item['case_name']}` | `{item['m']}` | `{item['n']}` | `{item['k']}` | `{item['workspace_bytes']}` | "
            f"`{item['dense_hot_mean_ms']}` | `{item['dense_cold_first_call_ms']}` | "
            f"`{item['nvfp4_hot_mean_ms']}` | `{item['nvfp4_runtime_hot_ms']}` | "
            f"`{item['nvfp4_runtime_cold_ms']}` | `{item['compute_speedup_vs_dense']}` | "
            f"`{item['runtime_speedup_vs_dense']}` | `{item['activation_pack_ms']}` |"
        )
    if report["dense_only_keys"]:
        lines.append("")
        lines.append("## Dense-Only Cases")
        lines.append("")
        for item in report["dense_only_keys"]:
            lines.append(f"- `{item}`")
    if report["nvfp4_only_keys"]:
        lines.append("")
        lines.append("## NVFP4-Only Cases")
        lines.append("")
        for item in report["nvfp4_only_keys"]:
            lines.append(f"- `{item}`")
    return "\n".join(lines) + "\n"


def format_key(key: tuple[Any, ...]) -> str:
    case_name, m, n, k, workspace_bytes = key
    return f"{case_name}:m={m}:n={n}:k={k}:workspace={workspace_bytes}"


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    artifacts_dir = Path(args.artifacts_dir)
    dense_path = Path(args.dense_json) if args.dense_json else find_preferred_artifact(
        artifacts_dir, "gb10_dense_gemm_default_*_cuda132.json"
    )
    nvfp4_path = Path(args.nvfp4_json) if args.nvfp4_json else find_preferred_artifact(
        artifacts_dir, "gb10_nvfp4_gemm_default_*_cuda132.json"
    )

    dense_payload = load_json(dense_path)
    nvfp4_payload = load_json(nvfp4_path)

    dense_results = {key_for_result(result): result for result in dense_payload["results"]}
    nvfp4_results = {
        key_for_result(result): result
        for result in nvfp4_payload["results"]
        if result.get("activation_staging") == args.activation_staging
    }

    matched_keys = sorted(set(dense_results) & set(nvfp4_results))
    dense_only_keys = sorted(set(dense_results) - set(nvfp4_results))
    nvfp4_only_keys = sorted(set(nvfp4_results) - set(dense_results))

    comparisons: list[dict[str, Any]] = []
    grouped_best: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    nvfp4_compute_win_count = 0
    nvfp4_service_win_count = 0

    for key in matched_keys:
        dense_result = dense_results[key]
        nvfp4_result = nvfp4_results[key]

        dense_hot_mean_ms = float(dense_result["hot_mean_ms"])
        dense_cold_first_call_ms = float(dense_result["cold_first_call_ms"])
        nvfp4_hot_mean_ms = float(nvfp4_result["hot_mean_ms"])
        activation_pack_ms = float(nvfp4_result["activation_pack_ms"])
        activation_upload_ms = float(nvfp4_result["activation_upload_ms"])
        nvfp4_runtime_hot_ms = nvfp4_hot_mean_ms + activation_pack_ms + activation_upload_ms
        nvfp4_runtime_cold_ms = float(nvfp4_result["cold_first_call_ms"]) + activation_pack_ms + activation_upload_ms

        compute_speedup = safe_div(dense_hot_mean_ms, nvfp4_hot_mean_ms)
        runtime_speedup = safe_div(dense_hot_mean_ms, nvfp4_runtime_hot_ms)
        cold_runtime_speedup = safe_div(dense_cold_first_call_ms, nvfp4_runtime_cold_ms)

        if nvfp4_hot_mean_ms < dense_hot_mean_ms:
            nvfp4_compute_win_count += 1
        if nvfp4_runtime_hot_ms < dense_hot_mean_ms:
            nvfp4_service_win_count += 1

        comparison = {
            "case_name": dense_result["case_name"],
            "source_tensor": dense_result["source_tensor"],
            "op_class": dense_result["op_class"],
            "m": int(dense_result["m"]),
            "n": int(dense_result["n"]),
            "k": int(dense_result["k"]),
            "workspace_bytes": int(dense_result["workspace_bytes"]),
            "dense_hot_mean_ms": round(dense_hot_mean_ms, 6),
            "dense_cold_first_call_ms": round(dense_cold_first_call_ms, 6),
            "dense_hot_tflops": round(float(dense_result["hot_tflops"]), 6),
            "nvfp4_hot_mean_ms": round(nvfp4_hot_mean_ms, 6),
            "nvfp4_runtime_hot_ms": round(nvfp4_runtime_hot_ms, 6),
            "nvfp4_runtime_cold_ms": round(nvfp4_runtime_cold_ms, 6),
            "nvfp4_hot_tflops": round(float(nvfp4_result["hot_tflops"]), 6),
            "activation_pack_ms": round(activation_pack_ms, 6),
            "activation_upload_ms": round(activation_upload_ms, 6),
            "compute_speedup_vs_dense": round_or_none(compute_speedup),
            "runtime_speedup_vs_dense": round_or_none(runtime_speedup),
            "cold_runtime_speedup_vs_dense": round_or_none(cold_runtime_speedup),
        }
        comparisons.append(comparison)
        grouped_best[(comparison["case_name"], comparison["m"])].append(comparison)

    comparisons.sort(key=lambda item: (item["case_name"], item["m"], item["workspace_bytes"]))

    best_by_case_and_m: list[dict[str, Any]] = []
    for (_, _), entries in sorted(grouped_best.items()):
        best = max(
            entries,
            key=lambda item: (-1.0 if item["runtime_speedup_vs_dense"] is None else item["runtime_speedup_vs_dense"]),
        )
        best_by_case_and_m.append(best)

    best_compute = max(
        comparisons,
        key=lambda item: (-1.0 if item["compute_speedup_vs_dense"] is None else item["compute_speedup_vs_dense"]),
        default=None,
    )
    best_service = max(
        comparisons,
        key=lambda item: (-1.0 if item["runtime_speedup_vs_dense"] is None else item["runtime_speedup_vs_dense"]),
        default=None,
    )

    report = {
        "dense_artifact": basename(dense_path),
        "nvfp4_artifact": basename(nvfp4_path),
        "activation_staging": args.activation_staging,
        "device": {
            "dense": dense_payload.get("device", {}),
            "nvfp4": nvfp4_payload.get("device", {}),
        },
        "summary": {
            "matched_comparisons": len(comparisons),
            "dense_only_count": len(dense_only_keys),
            "nvfp4_only_count": len(nvfp4_only_keys),
            "nvfp4_compute_win_count": nvfp4_compute_win_count,
            "nvfp4_service_win_count": nvfp4_service_win_count,
            "best_compute_speedup": None if best_compute is None else best_compute["compute_speedup_vs_dense"],
            "best_compute_speedup_label": None
            if best_compute is None
            else format_key(
                (
                    best_compute["case_name"],
                    best_compute["m"],
                    best_compute["n"],
                    best_compute["k"],
                    best_compute["workspace_bytes"],
                )
            ),
            "best_service_speedup": None if best_service is None else best_service["runtime_speedup_vs_dense"],
            "best_service_speedup_label": None
            if best_service is None
            else format_key(
                (
                    best_service["case_name"],
                    best_service["m"],
                    best_service["n"],
                    best_service["k"],
                    best_service["workspace_bytes"],
                )
            ),
        },
        "comparisons": comparisons,
        "best_by_case_and_m": best_by_case_and_m,
        "dense_only_keys": [format_key(key) for key in dense_only_keys],
        "nvfp4_only_keys": [format_key(key) for key in nvfp4_only_keys],
    }

    json_output = Path(args.json_output) if args.json_output else None
    markdown_output = Path(args.markdown_output) if args.markdown_output else None

    if json_output is not None:
      json_output.parent.mkdir(parents=True, exist_ok=True)
      json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown_output is not None:
      markdown_output.parent.mkdir(parents=True, exist_ok=True)
      markdown_output.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    if json_output is not None:
        print(f"wrote {json_output}")
    if markdown_output is not None:
        print(f"wrote {markdown_output}")


if __name__ == "__main__":
    main()
