#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_BUILD_DIR = "build-sm120-relwithdebinfo"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "artifacts"
    / "manifests"
    / "forward_runtime_manifest_nano_rtx5090_unverified.json"
)
DEFAULT_JSON_OUTPUT = (
    REPO_ROOT / "artifacts" / "benchmarks" / "nano_shared_prefill_latest.json"
)
DEFAULT_LENGTHS = [4, 8, 16, 24, 32, 64, 128, 256]
SHARED_KERNEL_NAME_SUBSTRING = "Nvfp4ContiguousSharedFp4P5Kernel"
SHARED_PROFILE_PATTERN = re.compile(
    r"shared_contiguous profile "
    r"tensor=(?P<tensor>\S+) "
    r"rows=(?P<rows>\d+) "
    r"profile=(?P<profile>\S+) "
    r"tile=(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+) "
    r"bucket=(?P<bucket_lower>\d+)-(?P<bucket_upper>\S+) "
    r"cta=(?P<cta_m>\d+)x(?P<cta_n>\d+) "
    r"backend=(?P<backend>\S+) "
    r"algorithm=(?P<algorithm>\d+) "
    r"cached=(?P<cached>[01])"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run prompt-length shared-prefill benchmarks with native direct MoE prefill enabled "
            "and optionally capture hot shared-kernel time through Nsight Systems."
        )
    )
    parser.add_argument("--build-dir", default=DEFAULT_BUILD_DIR)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
    )
    parser.add_argument(
        "--lengths",
        default=",".join(str(length) for length in DEFAULT_LENGTHS),
        help="Comma-separated prompt token counts.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup iterations for phased measurements.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Hot iterations for phased measurements.",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=1,
        help="Decode tokens used by phased runs. Prefill is the primary metric, so keep this small.",
    )
    parser.add_argument(
        "--profile-warmup",
        type=int,
        default=1,
        help="Warmup iterations before the Nsight profile-ready capture.",
    )
    parser.add_argument(
        "--capture-shared-kernel-time",
        action="store_true",
        help="Capture hot shared-kernel time with nsys profile + cudaProfilerStart/Stop.",
    )
    parser.add_argument(
        "--skip-shared-profile-debug",
        action="store_true",
        help="Disable NEMOTRON_SHARED_PROFILE_DEBUG parsing.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directory for raw per-length benchmark artifacts. Defaults next to --json-output.",
    )
    parser.add_argument(
        "--native-env",
        action="append",
        default=[],
        help="Extra KEY=VALUE env overrides for the benchmark process. Repeatable.",
    )
    return parser.parse_args()


def parse_lengths(text: str) -> list[int]:
    lengths: list[int] = []
    for chunk in text.split(","):
        stripped = chunk.strip()
        if not stripped:
            continue
        value = int(stripped)
        if value <= 0:
            raise ValueError(f"prompt length must be positive, got {value}")
        lengths.append(value)
    if not lengths:
        raise ValueError("at least one prompt length is required")
    return lengths


def parse_env_overrides(items: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid env override {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"invalid env override {item!r}")
        overrides[key] = value
    return overrides


def ensure_binary(build_dir: str) -> Path:
    binary = REPO_ROOT / build_dir / "benchmarks" / "nano_fused_decode" / "nano_fused_decode_bench"
    if not binary.exists():
        raise FileNotFoundError(f"benchmark binary not found: {binary}")
    return binary


def run_command(
    cmd: list[str],
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(cmd)}\n"
            f"stdout: {stdout_path}\nstderr: {stderr_path}"
        )
    return completed


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def parse_shared_profile_events(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for match in SHARED_PROFILE_PATTERN.finditer(text):
        bucket_upper_text = match.group("bucket_upper")
        bucket_upper: int | None
        if bucket_upper_text == "inf":
            bucket_upper = None
        else:
            bucket_upper = int(bucket_upper_text)
        events.append(
            {
                "tensor": match.group("tensor"),
                "rows": int(match.group("rows")),
                "profile": match.group("profile"),
                "tile_m": int(match.group("tile_m")),
                "tile_n": int(match.group("tile_n")),
                "tile_k": int(match.group("tile_k")),
                "bucket_lower": int(match.group("bucket_lower")),
                "bucket_upper": bucket_upper,
                "cta_m": int(match.group("cta_m")),
                "cta_n": int(match.group("cta_n")),
                "backend": match.group("backend"),
                "algorithm": int(match.group("algorithm")),
                "cached": match.group("cached") == "1",
            }
        )
    return events


def summarize_shared_profile_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[Any, ...], int] = defaultdict(int)
    for event in events:
        key = (
            event["tensor"],
            event["rows"],
            event["profile"],
            event["tile_m"],
            event["tile_n"],
            event["tile_k"],
            event["bucket_lower"],
            event["bucket_upper"],
            event["cta_m"],
            event["cta_n"],
            event["backend"],
            event["algorithm"],
            event["cached"],
        )
        by_key[key] += 1

    summaries: list[dict[str, Any]] = []
    for key, count in sorted(by_key.items()):
        (
            tensor,
            rows,
            profile,
            tile_m,
            tile_n,
            tile_k,
            bucket_lower,
            bucket_upper,
            cta_m,
            cta_n,
            backend,
            algorithm,
            cached,
        ) = key
        summaries.append(
            {
                "tensor": tensor,
                "rows": rows,
                "profile": profile,
                "tile_m": tile_m,
                "tile_n": tile_n,
                "tile_k": tile_k,
                "bucket_lower": bucket_lower,
                "bucket_upper": bucket_upper,
                "cta_m": cta_m,
                "cta_n": cta_n,
                "backend": backend,
                "algorithm": algorithm,
                "cached": cached,
                "count": count,
            }
        )

    by_tensor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        by_tensor[summary["tensor"]].append(summary)

    token_buckets = []
    seen_buckets: set[tuple[Any, ...]] = set()
    for summary in summaries:
        bucket_key = (
            summary["tensor"],
            summary["bucket_lower"],
            summary["bucket_upper"],
            summary["profile"],
        )
        if bucket_key in seen_buckets:
            continue
        seen_buckets.add(bucket_key)
        token_buckets.append(
            {
                "tensor": summary["tensor"],
                "bucket_lower": summary["bucket_lower"],
                "bucket_upper": summary["bucket_upper"],
                "profile": summary["profile"],
            }
        )

    return {
        "event_count": len(events),
        "summaries": summaries,
        "by_tensor": dict(by_tensor),
        "token_buckets": token_buckets,
    }


def parse_nsys_kernel_sum_report(text: str, kernel_name_substring: str) -> dict[str, Any]:
    lines = text.splitlines()
    header_index = None
    for index, line in enumerate(lines):
        if line.startswith("Time (%),Total Time (ns),Instances,Avg (ns),"):
            header_index = index
            break
    if header_index is None:
        raise ValueError("failed to find cuda_gpu_kern_sum CSV header in nsys output")

    reader = csv.DictReader(lines[header_index:])
    total_time_ns = 0
    total_instances = 0
    matched_rows: list[dict[str, Any]] = []
    for row in reader:
        name = row.get("Name", "")
        if kernel_name_substring not in name:
            continue
        total_ns = int(float(row["Total Time (ns)"]))
        instances = int(float(row["Instances"]))
        total_time_ns += total_ns
        total_instances += instances
        matched_rows.append(
            {
                "name": name,
                "total_time_ns": total_ns,
                "instances": instances,
                "avg_ns": int(float(row["Avg (ns)"])),
            }
        )
    return {
        "kernel_name_substring": kernel_name_substring,
        "total_time_ns": total_time_ns,
        "total_time_ms": round(total_time_ns / 1.0e6, 6),
        "instances": total_instances,
        "matched_rows": matched_rows,
    }


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def run_nsys_capture(
    *,
    length: int,
    binary: Path,
    manifest: Path,
    env: dict[str, str],
    raw_dir: Path,
    profile_warmup: int,
) -> dict[str, Any]:
    if shutil.which("nsys") is None:
        return {
            "available": False,
            "reason": "nsys not found in PATH",
        }

    prefix = raw_dir / f"prompt_{length:04d}_profile"
    json_output = raw_dir / f"prompt_{length:04d}_profile.json"
    stdout_path = raw_dir / f"prompt_{length:04d}_profile.stdout.txt"
    stderr_path = raw_dir / f"prompt_{length:04d}_profile.stderr.txt"
    rep_path = prefix.with_suffix(".nsys-rep")

    cmd = [
        "nsys",
        "profile",
        "--force-overwrite",
        "true",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--sample=none",
        "--cpuctxsw=none",
        "--trace=cuda",
        "--output",
        str(prefix),
        str(binary),
        "--manifest",
        str(manifest),
        "--mode=profile-ready",
        "--prompt-tokens",
        str(length),
        "--warmup",
        str(profile_warmup),
        "--json-output",
        str(json_output),
    ]
    completed = run_command(cmd, env, stdout_path, stderr_path)
    profile_json = load_json(json_output)
    shared_profile_events = parse_shared_profile_events(
        completed.stdout + "\n" + completed.stderr
    )
    stats_cmd = [
        "nsys",
        "stats",
        "--report",
        "cuda_gpu_kern_sum",
        "--format",
        "csv",
        "--output",
        "-",
        str(rep_path),
    ]
    stats_completed = subprocess.run(
        stats_cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    stats_stdout_path = raw_dir / f"prompt_{length:04d}_profile.cuda_gpu_kern_sum.csv"
    stats_stderr_path = raw_dir / f"prompt_{length:04d}_profile.cuda_gpu_kern_sum.stderr.txt"
    stats_stdout_path.write_text(stats_completed.stdout, encoding="utf-8")
    stats_stderr_path.write_text(stats_completed.stderr, encoding="utf-8")
    if stats_completed.returncode != 0:
        raise RuntimeError(
            f"nsys stats failed ({stats_completed.returncode}) for prompt length {length}: {rep_path}"
        )
    kernel_summary = parse_nsys_kernel_sum_report(
        stats_completed.stdout, SHARED_KERNEL_NAME_SUBSTRING
    )
    return {
        "available": True,
        "profile_ready_prefill_ms": profile_json["benchmark"]["steady_state_prefill_ms"],
        "shared_kernel": kernel_summary,
        "shared_profile_debug": summarize_shared_profile_events(shared_profile_events),
        "raw_artifacts": {
            "rep": relative_to_repo(rep_path),
            "json": relative_to_repo(json_output),
            "stdout": relative_to_repo(stdout_path),
            "stderr": relative_to_repo(stderr_path),
            "kernel_sum_csv": relative_to_repo(stats_stdout_path),
            "kernel_sum_stderr": relative_to_repo(stats_stderr_path),
        },
    }


def main() -> int:
    args = parse_args()
    lengths = parse_lengths(args.lengths)
    extra_env = parse_env_overrides(args.native_env)
    binary = ensure_binary(args.build_dir)
    manifest = args.manifest.resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    raw_dir = args.raw_dir
    if raw_dir is None:
        raw_dir = args.json_output.with_suffix("")
        raw_dir = raw_dir.parent / f"{raw_dir.name}_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)

    base_env = dict(os.environ)
    base_env["NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL"] = "1"
    if not args.skip_shared_profile_debug:
        base_env["NEMOTRON_SHARED_PROFILE_DEBUG"] = "1"
    base_env.update(extra_env)

    results: list[dict[str, Any]] = []
    warnings: list[str] = []
    first_environment: dict[str, Any] | None = None

    for length in lengths:
        phased_json = raw_dir / f"prompt_{length:04d}_phased.json"
        phased_stdout = raw_dir / f"prompt_{length:04d}_phased.stdout.txt"
        phased_stderr = raw_dir / f"prompt_{length:04d}_phased.stderr.txt"
        phased_cmd = [
            str(binary),
            "--manifest",
            str(manifest),
            "--mode=phased",
            "--prompt-tokens",
            str(length),
            "--decode-tokens",
            str(args.decode_tokens),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--json-output",
            str(phased_json),
        ]

        print(f"benchmark prompt_tokens={length} phased...", flush=True)
        phased_completed = run_command(phased_cmd, base_env, phased_stdout, phased_stderr)
        phased_payload = load_json(phased_json)
        if first_environment is None:
            first_environment = phased_payload.get("environment", {})

        benchmark = phased_payload["benchmark"]
        shared_profile_events = parse_shared_profile_events(
            phased_completed.stdout + "\n" + phased_completed.stderr
        )
        shared_profile_summary = summarize_shared_profile_events(shared_profile_events)
        result: dict[str, Any] = {
            "prompt_token_count": int(benchmark["prompt_token_count"]),
            "decode_token_count": int(benchmark["decode_token_count"]),
            "cold_prefill_ms": round(float(benchmark["cold_prefill_ms"]), 6),
            "hot_prefill_mean_ms": round(float(benchmark["hot_prefill_mean_ms"]), 6),
            "cold_setup_overhead_ms_estimate": round(
                max(
                    float(benchmark["cold_prefill_ms"]) -
                    float(benchmark["hot_prefill_mean_ms"]),
                    0.0,
                ),
                6,
            ),
            "hot_first_token_mean_ms": round(float(benchmark["hot_first_token_mean_ms"]), 6),
            "linear_op_counters": phased_payload.get("linear_op_counters", {}),
            "expert_staging_counters": phased_payload.get("expert_staging_counters", {}),
            "shared_profile_debug": shared_profile_summary,
            "raw_artifacts": {
                "phased_json": relative_to_repo(phased_json),
                "phased_stdout": relative_to_repo(phased_stdout),
                "phased_stderr": relative_to_repo(phased_stderr),
            },
        }

        if args.capture_shared_kernel_time:
            print(f"benchmark prompt_tokens={length} nsys profile...", flush=True)
            nsys_summary = run_nsys_capture(
                length=length,
                binary=binary,
                manifest=manifest,
                env=base_env,
                raw_dir=raw_dir,
                profile_warmup=args.profile_warmup,
            )
            result["nsys"] = nsys_summary
            if nsys_summary.get("available"):
                shared_kernel = nsys_summary["shared_kernel"]
                result["hot_shared_kernel_ms"] = shared_kernel["total_time_ms"]
                result["hot_shared_kernel_instances"] = shared_kernel["instances"]
                profile_ready_prefill_ms = float(nsys_summary["profile_ready_prefill_ms"])
                result["profile_ready_prefill_ms"] = round(profile_ready_prefill_ms, 6)
                result["hot_shared_kernel_fraction_of_prefill"] = round(
                    shared_kernel["total_time_ms"] / profile_ready_prefill_ms,
                    6,
                ) if profile_ready_prefill_ms > 0.0 else None
            else:
                warnings.append(str(nsys_summary.get("reason", "nsys unavailable")))
                result["hot_shared_kernel_ms"] = None
                result["hot_shared_kernel_instances"] = 0
                result["profile_ready_prefill_ms"] = None
                result["hot_shared_kernel_fraction_of_prefill"] = None
        else:
            result["nsys"] = {
                "available": False,
                "reason": "capture disabled",
            }
            result["hot_shared_kernel_ms"] = None
            result["hot_shared_kernel_instances"] = 0
            result["profile_ready_prefill_ms"] = None
            result["hot_shared_kernel_fraction_of_prefill"] = None

        results.append(result)

    payload = {
        "benchmark": "nano_shared_prefill_benchmark",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": first_environment or {},
        "options": {
            "build_dir": args.build_dir,
            "manifest_path": str(manifest),
            "lengths": lengths,
            "warmup_iterations": args.warmup,
            "hot_iterations": args.iterations,
            "decode_token_count": args.decode_tokens,
            "profile_warmup_iterations": args.profile_warmup,
            "capture_shared_kernel_time": args.capture_shared_kernel_time,
            "shared_profile_debug": not args.skip_shared_profile_debug,
            "native_env": {
                "NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL": "1",
                **({"NEMOTRON_SHARED_PROFILE_DEBUG": "1"} if not args.skip_shared_profile_debug else {}),
                **extra_env,
            },
            "raw_dir": str(raw_dir.resolve()),
        },
        "warnings": warnings,
        "results": results,
    }
    args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"run_shared_prefill_benchmark.py: {exc}", file=sys.stderr)
        raise
