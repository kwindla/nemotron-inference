#!/usr/bin/env python3
"""Offline vLLM baseline benchmark for Nemotron Nano NVFP4 on RTX 5090.

This script intentionally uses vLLM's offline ``LLM`` entrypoint and then
drives ``llm.llm_engine.add_request()/step()`` directly so TTFT is measured
from request submission to the first emitted token, rather than total
``generate()`` completion time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any

import numpy as np


SCRIPT_PATH = pathlib.Path(__file__).resolve()
PROJECT_DIR = SCRIPT_PATH.parent
REPO_ROOT = PROJECT_DIR.parent
LOCAL_VLLM_ROOT = REPO_ROOT / "third_party" / "vllm"

if str(LOCAL_VLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_VLLM_ROOT))

from vllm import LLM, SamplingParams  # type: ignore[import-not-found]
from vllm.outputs import RequestOutput  # type: ignore[import-not-found]
from vllm.sampling_params import RequestOutputKind  # type: ignore[import-not-found]


DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
DEFAULT_TOKEN_COUNTS = (1, 32, 64, 128, 256)
ROUTING_TOKEN_COUNTS = {32, 64, 128}
PROFILE_MATCH_KEYWORDS = (
    "fused_moe",
    "flashinfer",
    "cutlass_fused_moe",
    "moe_align_block_size",
    "moe_kernel_quantize_input",
    "topk_weight_and_reduce",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark vLLM offline TTFT for NVIDIA Nemotron Nano NVFP4 with "
            "flashinfer_cutlass."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--token-counts",
        nargs="+",
        type=int,
        default=list(DEFAULT_TOKEN_COUNTS),
        help="Prompt token counts to benchmark. Token count 1 is treated as decode.",
    )
    parser.add_argument(
        "--decode-prefix-tokens",
        type=int,
        default=256,
        help=(
            "Fixed prompt length used to seed a full prefix-cache hit for the "
            "decode case."
        ),
    )
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Override vLLM max_num_batched_tokens. Default is max(prompt len, decode prefix len).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=PROJECT_DIR / "vllm_baseline_results.json",
    )
    parser.add_argument(
        "--skip-moe-profile",
        action="store_true",
        help="Skip the optional vLLM torch-profiler pass for MoE operator timing.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to vLLM if the local environment requires it.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Force eager execution. Default keeps vLLM's normal hybrid mode.",
    )
    return parser.parse_args()


def git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(LOCAL_VLLM_ROOT), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def duration_token_to_ms(token: str | None) -> float | None:
    if not token:
        return None
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)(ns|us|ms|s)", token.strip())
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "ns":
        return value / 1_000_000.0
    if unit == "us":
        return value / 1_000.0
    if unit == "ms":
        return value
    if unit == "s":
        return value * 1_000.0
    return None


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * q
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "count": float(len(values)),
        "min_ms": min(values),
        "max_ms": max(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
    }


def build_prompt_token_ids(
    *,
    vocab_size: int,
    special_ids: set[int],
    token_count: int,
    salt: int,
) -> list[int]:
    # Different first token per salt prevents accidental prefix-cache hits.
    ids: list[int] = []
    state = (salt * 1_103_515_245 + 12_345) % max(vocab_size, 2)
    while len(ids) < token_count:
        state = (state * 48271 + 1) % max(vocab_size, 2)
        candidate = int(state)
        if candidate in special_ids:
            continue
        ids.append(candidate)
    return ids


def make_sampling_params() -> SamplingParams:
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        min_tokens=1,
        ignore_eos=True,
        detokenize=False,
        output_kind=RequestOutputKind.DELTA,
    )


def collect_worker_moe_metadata(llm: LLM) -> list[dict[str, Any]]:
    def _collect(model: Any) -> list[dict[str, Any]]:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE

        layers: list[dict[str, Any]] = []
        for name, module in model.named_modules():
            if not isinstance(module, FusedMoE):
                continue
            quant_method = getattr(module, "quant_method", None)
            moe_kernel = getattr(quant_method, "moe_kernel", None)
            fused_experts = getattr(moe_kernel, "fused_experts", None)
            nvfp4_backend = getattr(quant_method, "nvfp4_backend", None)
            layers.append(
                {
                    "module_name": name,
                    "layer_name": getattr(module, "layer_name", None),
                    "layer_id": int(module.layer_id),
                    "requested_moe_backend": getattr(module.moe_config, "moe_backend", None),
                    "quant_method": (
                        quant_method.__class__.__name__ if quant_method is not None else None
                    ),
                    "selected_nvfp4_backend": (
                        getattr(nvfp4_backend, "value", str(nvfp4_backend))
                        if nvfp4_backend is not None
                        else None
                    ),
                    "experts_cls": (
                        fused_experts.__class__.__name__ if fused_experts is not None else None
                    ),
                    "global_num_experts": int(module.global_num_experts),
                    "top_k": int(module.top_k),
                    "hidden_size": int(module.moe_config.hidden_dim_unpadded),
                }
            )
        return sorted(layers, key=lambda item: item["layer_id"])

    results = llm.apply_model(_collect)
    return results[0] if results else []


def run_request(
    llm: LLM,
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
    request_label: str,
) -> dict[str, Any]:
    request_id = f"{request_label}-{uuid.uuid4().hex}"
    first_token_time_ns: int | None = None
    final_output: RequestOutput | None = None
    start_ns = time.perf_counter_ns()
    llm.llm_engine.add_request(
        request_id,
        {"prompt_token_ids": prompt_token_ids},
        sampling_params,
        arrival_time=time.time(),
    )

    while llm.llm_engine.has_unfinished_requests():
        step_outputs = llm.llm_engine.step()
        step_end_ns = time.perf_counter_ns()
        for output in step_outputs:
            if not isinstance(output, RequestOutput) or output.request_id != request_id:
                continue
            if first_token_time_ns is None and any(
                completion.token_ids for completion in output.outputs
            ):
                first_token_time_ns = step_end_ns
            if output.finished:
                final_output = output
        if final_output is not None and first_token_time_ns is not None:
            break

    if final_output is None:
        raise RuntimeError(f"Request {request_id} finished without a RequestOutput.")
    if first_token_time_ns is None:
        raise RuntimeError(f"Request {request_id} never emitted a first token.")

    completion = final_output.outputs[0]
    return {
        "request_id": request_id,
        "ttft_ms": (first_token_time_ns - start_ns) / 1_000_000.0,
        "generated_token_ids": list(int(token_id) for token_id in completion.token_ids),
        "routed_experts": completion.routed_experts,
        "num_cached_tokens": final_output.num_cached_tokens,
    }


def build_routing_histogram(
    routed_experts: np.ndarray | None,
    *,
    moe_layer_ids: list[int],
    num_experts: int,
    top_k: int,
) -> dict[str, Any] | None:
    if routed_experts is None:
        return None

    histograms: dict[str, Any] = {}
    for layer_id in moe_layer_ids:
        layer_experts = routed_experts[:, layer_id, :]
        flat = layer_experts.reshape(-1)
        counts = np.bincount(flat, minlength=num_experts)
        counts = counts[:num_experts]
        active = int(np.count_nonzero(counts))
        histograms[str(layer_id)] = {
            "selection_count": int(counts.sum()),
            "active_experts": active,
            "bucket_counts": {
                "M=1": int(np.count_nonzero(counts == 1)),
                "M=2": int(np.count_nonzero(counts == 2)),
                "M=3": int(np.count_nonzero(counts == 3)),
                "M=4+": int(np.count_nonzero(counts >= 4)),
            },
            "per_expert_token_counts": counts.astype(int).tolist(),
        }

    return {
        "prompt_token_count": int(routed_experts.shape[0]),
        "top_k": top_k,
        "per_layer": histograms,
    }


def parse_profiler_rows(table_path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in table_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("-") or stripped.startswith("Self CPU"):
            continue
        columns = re.split(r"\s{2,}", stripped)
        if len(columns) < 7 or columns[0] == "Name":
            continue
        rows.append(
            {
                "name": columns[0],
                "self_cpu_percent": columns[1] if len(columns) > 1 else None,
                "self_cpu": columns[2] if len(columns) > 2 else None,
                "cpu_total_percent": columns[3] if len(columns) > 3 else None,
                "cpu_total": columns[4] if len(columns) > 4 else None,
                "cpu_time_avg": columns[5] if len(columns) > 5 else None,
                "self_cuda": columns[6] if len(columns) > 6 else None,
                "self_cuda_percent": columns[7] if len(columns) > 7 else None,
                "cuda_total": columns[8] if len(columns) > 8 else None,
                "cuda_time_avg": columns[9] if len(columns) > 9 else None,
                "calls": columns[10] if len(columns) > 10 else None,
                "raw": stripped,
            }
        )
    return rows


def profile_moe_case(
    *,
    llm: LLM,
    sampling_params: SamplingParams,
    prompt_token_ids: list[int],
    profile_root: pathlib.Path,
    request_label: str,
) -> dict[str, Any] | None:
    if profile_root.exists():
        for old_file in profile_root.glob("profiler_out_*.txt"):
            old_file.unlink()

    try:
        llm.start_profile(f"{request_label}_profile")
        run_request(llm, prompt_token_ids, sampling_params, f"{request_label}_profile")
        llm.stop_profile()
    except Exception as exc:  # pragma: no cover - best effort path
        try:
            llm.stop_profile()
        except Exception:
            pass
        return {
            "available": False,
            "error": str(exc),
        }

    matched_rows: list[dict[str, Any]] = []
    for table_path in sorted(profile_root.glob("profiler_out_*.txt")):
        for row in parse_profiler_rows(table_path):
            name = str(row["name"]).lower()
            if any(keyword in name for keyword in PROFILE_MATCH_KEYWORDS):
                row["source_file"] = table_path.name
                row["self_cuda_ms"] = duration_token_to_ms(row.get("self_cuda"))
                row["cuda_total_ms"] = duration_token_to_ms(row.get("cuda_total"))
                matched_rows.append(row)

    if not matched_rows:
        return {
            "available": False,
            "reason": "No MoE or FlashInfer rows were found in the torch-profiler summary.",
        }

    self_cuda_ms_total = sum(
        row["self_cuda_ms"] for row in matched_rows if row["self_cuda_ms"] is not None
    )
    return {
        "available": True,
        "self_cuda_ms_total": self_cuda_ms_total,
        "matched_rows": matched_rows,
    }


def benchmark_prefill_case(
    *,
    llm: LLM,
    sampling_params: SamplingParams,
    token_count: int,
    warmup_iters: int,
    iters: int,
    seed: int,
    vocab_size: int,
    special_ids: set[int],
    moe_layer_ids: list[int],
    num_experts: int,
    top_k: int,
    profile_root: pathlib.Path | None,
) -> dict[str, Any]:
    llm.reset_prefix_cache()

    for warmup_idx in range(warmup_iters):
        prompt = build_prompt_token_ids(
            vocab_size=vocab_size,
            special_ids=special_ids,
            token_count=token_count,
            salt=seed * 10_000 + token_count * 100 + warmup_idx,
        )
        run_request(llm, prompt, sampling_params, f"warmup_prefill_{token_count}_{warmup_idx}")

    iterations: list[dict[str, Any]] = []
    routing_histogram: dict[str, Any] | None = None
    for iter_idx in range(iters):
        prompt = build_prompt_token_ids(
            vocab_size=vocab_size,
            special_ids=special_ids,
            token_count=token_count,
            salt=seed * 100_000 + token_count * 1_000 + iter_idx,
        )
        result = run_request(llm, prompt, sampling_params, f"prefill_{token_count}_{iter_idx}")
        iterations.append(
            {
                "iteration": iter_idx,
                "ttft_ms": result["ttft_ms"],
                "num_cached_tokens": result["num_cached_tokens"],
                "generated_token_ids": result["generated_token_ids"],
            }
        )
        if token_count in ROUTING_TOKEN_COUNTS and routing_histogram is None:
            routing_histogram = build_routing_histogram(
                result["routed_experts"],
                moe_layer_ids=moe_layer_ids,
                num_experts=num_experts,
                top_k=top_k,
            )

    profile = None
    if profile_root is not None:
        profile_prompt = build_prompt_token_ids(
            vocab_size=vocab_size,
            special_ids=special_ids,
            token_count=token_count,
            salt=seed * 1_000_000 + token_count,
        )
        profile = profile_moe_case(
            llm=llm,
            sampling_params=sampling_params,
            prompt_token_ids=profile_prompt,
            profile_root=profile_root,
            request_label=f"prefill_{token_count}",
        )

    ttft_values = [iteration["ttft_ms"] for iteration in iterations]
    cached_values = [float(iteration["num_cached_tokens"] or 0) for iteration in iterations]
    return {
        "label": f"prefill_{token_count}",
        "mode": "prefill",
        "benchmark_token_count": token_count,
        "prompt_token_count": token_count,
        "generated_token_count": 1,
        "ttft_summary_ms": summarize(ttft_values),
        "iterations": iterations,
        "num_cached_tokens_summary": summarize(cached_values),
        "routing_histogram": routing_histogram,
        "moe_profile": profile,
        "notes": [
            "Prefix caching remains enabled globally but per-iteration prompt IDs are salted to avoid cache hits.",
            "TTFT is wall-clock from add_request() to the first emitted output token.",
        ],
    }


def seed_prefix_cache(
    *,
    llm: LLM,
    sampling_params: SamplingParams,
    prompt_token_ids: list[int],
) -> None:
    llm.reset_prefix_cache()
    run_request(llm, prompt_token_ids, sampling_params, "decode_seed")


def benchmark_decode_case(
    *,
    llm: LLM,
    sampling_params: SamplingParams,
    decode_prefix_tokens: int,
    warmup_iters: int,
    iters: int,
    seed: int,
    vocab_size: int,
    special_ids: set[int],
    profile_root: pathlib.Path | None,
) -> dict[str, Any]:
    decode_prompt = build_prompt_token_ids(
        vocab_size=vocab_size,
        special_ids=special_ids,
        token_count=decode_prefix_tokens,
        salt=seed * 13 + 1,
    )

    seed_prefix_cache(
        llm=llm,
        sampling_params=sampling_params,
        prompt_token_ids=decode_prompt,
    )

    for warmup_idx in range(warmup_iters):
        run_request(llm, decode_prompt, sampling_params, f"warmup_decode_{warmup_idx}")

    iterations: list[dict[str, Any]] = []
    for iter_idx in range(iters):
        result = run_request(llm, decode_prompt, sampling_params, f"decode_{iter_idx}")
        iterations.append(
            {
                "iteration": iter_idx,
                "ttft_ms": result["ttft_ms"],
                "num_cached_tokens": result["num_cached_tokens"],
                "generated_token_ids": result["generated_token_ids"],
            }
        )

    profile = None
    if profile_root is not None:
        profile = profile_moe_case(
            llm=llm,
            sampling_params=sampling_params,
            prompt_token_ids=decode_prompt,
            profile_root=profile_root,
            request_label="decode",
        )

    ttft_values = [iteration["ttft_ms"] for iteration in iterations]
    cached_values = [float(iteration["num_cached_tokens"] or 0) for iteration in iterations]
    return {
        "label": "decode_1",
        "mode": "decode",
        "benchmark_token_count": 1,
        "decode_prefix_prompt_token_count": decode_prefix_tokens,
        "generated_token_count": 1,
        "ttft_summary_ms": summarize(ttft_values),
        "iterations": iterations,
        "num_cached_tokens_summary": summarize(cached_values),
        "routing_histogram": None,
        "moe_profile": profile,
        "notes": [
            "Decode is measured as a repeated full-prefix-cache hit on a fixed prompt, then first-token generation.",
            "The cached-token summary should be inspected to confirm the request stayed on the cache-hit path.",
        ],
    }


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    llm: LLM | None = None

    requested_token_counts = tuple(int(value) for value in args.token_counts)
    if args.iters < 1:
        raise ValueError("--iters must be at least 1.")
    if args.warmup_iters < 0:
        raise ValueError("--warmup-iters must be non-negative.")
    if args.decode_prefix_tokens < 1:
        raise ValueError("--decode-prefix-tokens must be at least 1.")
    if any(token_count < 1 for token_count in requested_token_counts):
        raise ValueError("All token counts must be at least 1.")
    if 1 not in requested_token_counts:
        raise ValueError("Token count 1 must be included for the decode measurement.")

    max_prompt_tokens = max(max(requested_token_counts), args.decode_prefix_tokens)
    max_num_batched_tokens = args.max_num_batched_tokens or max_prompt_tokens

    profiler_root: pathlib.Path | None = None
    profiler_tmp_dir: tempfile.TemporaryDirectory[str] | None = None
    profiler_config: dict[str, Any] | None = None
    if not args.skip_moe_profile:
        profiler_tmp_dir = tempfile.TemporaryDirectory(
            dir=str(PROJECT_DIR),
            prefix=".tmp_vllm_profiler_",
        )
        profiler_root = pathlib.Path(profiler_tmp_dir.name)
        profiler_config = {
            "profiler": "torch",
            "torch_profiler_dir": str(profiler_root),
            "torch_profiler_with_stack": False,
            "torch_profiler_with_flops": False,
            "torch_profiler_with_memory": False,
            "torch_profiler_use_gzip": False,
            "torch_profiler_dump_cuda_time_total": True,
            "ignore_frontend": True,
        }

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=1,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_prompt_tokens + 8,
        enable_prefix_caching=True,
        enable_return_routed_experts=True,
        enforce_eager=args.enforce_eager,
        profiler_config=profiler_config,
        moe_backend="flashinfer_cutlass",
    )

    try:
        tokenizer = llm.get_tokenizer()
        vocab_size = int(getattr(tokenizer, "vocab_size"))
        special_ids = set(int(value) for value in getattr(tokenizer, "all_special_ids", []) or [])

        worker_moe_layers = collect_worker_moe_metadata(llm)
        if not worker_moe_layers:
            raise RuntimeError("Failed to discover any FusedMoE layers in the loaded model.")

        moe_layer_ids = [int(layer["layer_id"]) for layer in worker_moe_layers]
        selected_backends = sorted(
            {
                layer["selected_nvfp4_backend"]
                for layer in worker_moe_layers
                if layer["selected_nvfp4_backend"] is not None
            }
        )

        num_experts = int(worker_moe_layers[0]["global_num_experts"])
        top_k = int(worker_moe_layers[0]["top_k"])

        sampling_params = make_sampling_params()

        cases: list[dict[str, Any]] = []
        cases.append(
            benchmark_decode_case(
                llm=llm,
                sampling_params=sampling_params,
                decode_prefix_tokens=args.decode_prefix_tokens,
                warmup_iters=args.warmup_iters,
                iters=args.iters,
                seed=args.seed,
                vocab_size=vocab_size,
                special_ids=special_ids,
                profile_root=profiler_root,
            )
        )

        for token_count in requested_token_counts:
            if token_count == 1:
                continue
            cases.append(
                benchmark_prefill_case(
                    llm=llm,
                    sampling_params=sampling_params,
                    token_count=token_count,
                    warmup_iters=args.warmup_iters,
                    iters=args.iters,
                    seed=args.seed,
                    vocab_size=vocab_size,
                    special_ids=special_ids,
                    moe_layer_ids=moe_layer_ids,
                    num_experts=num_experts,
                    top_k=top_k,
                    profile_root=profiler_root,
                )
            )

        result = {
            "benchmark_name": "vllm_nemotron_nano_ttft",
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "repo_root": str(REPO_ROOT),
            "project_dir": str(PROJECT_DIR),
            "model": {
                "id": args.model,
                "hidden_size": int(worker_moe_layers[0]["hidden_size"]),
                "routed_experts": num_experts,
                "top_k": top_k,
            },
            "vllm": {
                "local_path": str(LOCAL_VLLM_ROOT),
                "git_tag": git_output("describe", "--tags", "--always"),
                "git_commit": git_output("rev-parse", "HEAD"),
                "requested_moe_backend": "flashinfer_cutlass",
                "selected_nvfp4_backends": selected_backends,
                "worker_moe_layers": worker_moe_layers,
            },
            "engine_config": {
                "tensor_parallel_size": args.tensor_parallel_size,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_num_batched_tokens": max_num_batched_tokens,
                "max_model_len": max_prompt_tokens + 8,
                "enable_prefix_caching": True,
                "enable_return_routed_experts": True,
                "enforce_eager": args.enforce_eager,
            },
            "environment": {
                "VLLM_USE_FLASHINFER_MOE_FP4": os.environ.get("VLLM_USE_FLASHINFER_MOE_FP4"),
                "VLLM_FLASHINFER_MOE_BACKEND": os.environ.get("VLLM_FLASHINFER_MOE_BACKEND"),
                "VLLM_NVFP4_GEMM_BACKEND": os.environ.get("VLLM_NVFP4_GEMM_BACKEND"),
                "VLLM_MOE_PADDING": os.environ.get("VLLM_MOE_PADDING"),
            },
            "cases": cases,
            "notes": [
                "Prefill cases use unique prompt IDs per iteration to keep prefix caching from changing TTFT.",
                "Decode uses a seeded prefix-cache hit on a fixed prompt rather than a 1-token uncached prompt.",
                "MoE timing is best-effort from vLLM torch-profiler summaries and may be null if the trace does not expose MoE rows.",
            ],
        }

        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return 0
    finally:
        # Keep the benchmark output, but do not leave behind temporary profiler dumps.
        if profiler_tmp_dir is not None:
            profiler_tmp_dir.cleanup()
        if llm is not None:
            try:
                llm.reset_prefix_cache()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
