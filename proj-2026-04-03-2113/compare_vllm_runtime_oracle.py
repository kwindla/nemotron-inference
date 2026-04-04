#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

from vllm import LLM, SamplingParams

from runtime_oracle_support import (
    DEFAULT_MODEL,
    DEFAULT_PROMPTS_FILE,
    REPO_ROOT,
    default_oracle_output_path,
    find_latest_oracle,
    resolve_prompt_token_ids,
    run_runtime_oracle,
)


LOCAL_VLLM_ROOT = REPO_ROOT / "third_party" / "vllm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a runtime oracle token sequence against local vLLM on the same "
            "prompt token IDs. The oracle can be loaded from disk or generated on "
            "demand from token IDs, a named prompt, or raw text."
        )
    )
    parser.add_argument(
        "--oracle",
        type=pathlib.Path,
        help=(
            "Path to a saved runtime oracle JSON file. Defaults to the latest oracle "
            "when no prompt source is provided."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--moe-backend", default="flashinfer_cutlass")
    parser.add_argument("--decode-token-count", type=int, default=16)
    parser.add_argument("--runtime-oracle-output", type=pathlib.Path)
    parser.add_argument("--artifact-output", type=pathlib.Path)
    parser.add_argument("--runtime-build-dir")
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument(
        "--include-deep-regression-payload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include the optional deep regression payload when generating a runtime "
            "oracle on demand."
        ),
    )
    parser.add_argument("--prompts-file", type=pathlib.Path, default=DEFAULT_PROMPTS_FILE)
    parser.add_argument("--prompt-token-ids")
    parser.add_argument("--prompt-token-ids-file", type=pathlib.Path)
    parser.add_argument("--prompt-name")
    parser.add_argument("--prompt-text")
    parser.add_argument("--prompt-file", type=pathlib.Path)
    return parser.parse_args()


def decode_ids(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def should_generate_runtime_oracle(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in (
            args.prompt_token_ids,
            args.prompt_token_ids_file,
            args.prompt_name,
            args.prompt_text,
            args.prompt_file,
        )
    )


def git_output(cwd: pathlib.Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(cwd), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


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

    try:
        results = llm.apply_model(_collect)
    except Exception:
        return []
    return results[0] if results else []


def collect_vllm_environment() -> dict[str, str]:
    keys = (
        "PYTHONPATH",
        "PYTORCH_CUDA_ALLOC_CONF",
        "VLLM_ALLOW_INSECURE_SERIALIZATION",
        "VLLM_FLASHINFER_MOE_BACKEND",
        "VLLM_MOE_PADDING",
        "VLLM_NVFP4_GEMM_BACKEND",
        "VLLM_USE_FLASHINFER_MOE_FP4",
    )
    result: dict[str, str] = {}
    for key in keys:
        value = os.environ.get(key)
        if value:
            result[key] = value
    return result


def default_artifact_output_path(oracle_path: pathlib.Path) -> pathlib.Path:
    return oracle_path.with_name(f"{oracle_path.stem}.vllm_parity.json")


def first_mismatch_index(lhs: list[int], rhs: list[int]) -> int | None:
    if lhs == rhs:
        return None
    return next(
        (
            index
            for index, (left, right) in enumerate(zip(lhs, rhs))
            if left != right
        ),
        min(len(lhs), len(rhs)),
    )


def main() -> int:
    args = parse_args()
    if args.decode_token_count <= 0:
        raise ValueError("--decode-token-count must be positive")

    generate_runtime_oracle = should_generate_runtime_oracle(args)
    if args.oracle is not None and generate_runtime_oracle:
        raise ValueError("--oracle cannot be combined with an explicit prompt source")

    prompt_source: str | None = None
    oracle_path = args.oracle
    if generate_runtime_oracle:
        prompt_token_ids, prompt_source = resolve_prompt_token_ids(
            model=args.model,
            prompt_token_ids=args.prompt_token_ids,
            prompt_token_ids_file=args.prompt_token_ids_file,
            prompt_name=args.prompt_name,
            prompt_text=args.prompt_text,
            prompt_file=args.prompt_file,
            prompts_file=args.prompts_file,
        )
        oracle_path = args.runtime_oracle_output or default_oracle_output_path(
            len(prompt_token_ids)
        )
        run_runtime_oracle(
            prompt_token_ids=prompt_token_ids,
            output_path=oracle_path,
            decode_token_count=args.decode_token_count,
            build_dir=args.runtime_build_dir,
            manifest=args.manifest,
            include_deep_regression_payload=args.include_deep_regression_payload,
        )
        print(f"generated_runtime_oracle={oracle_path}")
        print(f"prompt_source={prompt_source}")
    elif oracle_path is None:
        oracle_path = find_latest_oracle()

    payload = json.loads(oracle_path.read_text(encoding="utf-8"))
    prompt_source = prompt_source or payload.get("prompt_source") or f"oracle:{oracle_path}"
    prompt_token_ids = [int(value) for value in payload["prompt_token_ids"]]
    runtime_token_ids = [int(value) for value in payload["generated_token_ids"]]
    decode_len = len(runtime_token_ids)
    min_context = len(prompt_token_ids) + decode_len
    max_model_len = max(args.max_model_len or 0, min_context)
    max_num_batched_tokens = max(args.max_num_batched_tokens or 0, min_context)

    llm = LLM(
        args.model,
        trust_remote_code=True,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        moe_backend=args.moe_backend,
    )
    tokenizer = llm.get_tokenizer()
    worker_moe_layers = collect_worker_moe_metadata(llm)
    selected_backends = sorted(
        {
            layer["selected_nvfp4_backend"]
            for layer in worker_moe_layers
            if layer.get("selected_nvfp4_backend") is not None
        }
    )

    sampling_params = SamplingParams(
        max_tokens=decode_len,
        min_tokens=decode_len,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
    )
    output = llm.generate([{"prompt_token_ids": prompt_token_ids}], sampling_params)[0]
    vllm_token_ids = [int(value) for value in output.outputs[0].token_ids]

    prompt_text = decode_ids(tokenizer, prompt_token_ids)
    runtime_text = decode_ids(tokenizer, runtime_token_ids)
    vllm_text = decode_ids(tokenizer, vllm_token_ids)
    mismatch_index = first_mismatch_index(runtime_token_ids, vllm_token_ids)
    exact_match = mismatch_index is None
    mismatch_runtime_token = (
        runtime_token_ids[mismatch_index]
        if mismatch_index is not None and mismatch_index < len(runtime_token_ids)
        else None
    )
    mismatch_vllm_token = (
        vllm_token_ids[mismatch_index]
        if mismatch_index is not None and mismatch_index < len(vllm_token_ids)
        else None
    )

    artifact_path = args.artifact_output or default_artifact_output_path(oracle_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "gate": "exact_token_vllm_parity",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "exact_match" if exact_match else "token_mismatch",
        "exact_token_match": exact_match,
        "mismatch_index": mismatch_index,
        "mismatch_runtime_token_id": mismatch_runtime_token,
        "mismatch_vllm_token_id": mismatch_vllm_token,
        "oracle_path": str(oracle_path),
        "prompt_source": prompt_source,
        "prompt_token_ids": prompt_token_ids,
        "decode_token_count": decode_len,
        "prompt_text": prompt_text,
        "runtime": {
            "model_id": payload.get("model_id"),
            "manifest_path": payload.get("manifest_path"),
            "build_dir": payload.get("build_dir"),
            "git_revision": payload.get("git_revision"),
            "backend_flags": payload.get("backend_flags", {}),
            "route": payload.get("route"),
            "route_description": payload.get("route_description"),
            "boundary_token_id": payload.get("boundary_token_id"),
            "generated_token_ids": runtime_token_ids,
            "generated_text": runtime_text,
        },
        "vllm": {
            "model_id": args.model,
            "local_source_tree": str(LOCAL_VLLM_ROOT),
            "git_tag": git_output(LOCAL_VLLM_ROOT, "describe", "--tags", "--always"),
            "git_revision": git_output(LOCAL_VLLM_ROOT, "rev-parse", "HEAD"),
            "moe_backend_requested": args.moe_backend,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "trust_remote_code": True,
            "enforce_eager": True,
            "environment": collect_vllm_environment(),
            "selected_nvfp4_backends": selected_backends,
            "worker_moe_layers": worker_moe_layers,
            "generated_token_ids": vllm_token_ids,
            "generated_text": vllm_text,
        },
    }
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"oracle_path={oracle_path}")
    print(f"artifact_path={artifact_path}")
    print(f"prompt_token_ids={prompt_token_ids}")
    print(f"prompt_text={prompt_text!r}")
    print(f"runtime_generated_token_ids={runtime_token_ids}")
    print(f"runtime_generated_text={runtime_text!r}")
    print(f"vllm_generated_token_ids={vllm_token_ids}")
    print(f"vllm_generated_text={vllm_text!r}")

    if exact_match:
        print("token_match=exact")
        return 0

    print(f"token_match=diverged first_mismatch_index={mismatch_index}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
