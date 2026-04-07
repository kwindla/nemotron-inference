#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pathlib
import random
import time
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "artifacts" / "checkpoints" / "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"


def load_tokenizer(tokenizer_name: str):
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name, fix_mistral_regex=True, **kwargs)
    except TypeError:
        return AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)


def build_prompt_token_ids(tokenizer: Any, token_count: int, seed: int) -> list[int]:
    special_ids = {
        token_id
        for token_id in [
            getattr(tokenizer, "bos_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "pad_token_id", None),
        ]
        if isinstance(token_id, int) and token_id >= 0
    }
    rng = random.Random(seed + token_count)
    vocab_size = int(tokenizer.vocab_size)
    candidates = [token_id for token_id in range(32, vocab_size) if token_id not in special_ids]
    if not candidates:
        raise RuntimeError("no usable tokenizer ids available for prompt construction")
    return [candidates[rng.randrange(len(candidates))] for _ in range(token_count)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe vLLM prefix caching on identical prompt token IDs.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--token-count", type=int, action="append", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--result-path", required=True)
    return parser.parse_args()


def run_probe(
    *,
    model: str,
    prompt_token_ids: list[int],
    repetitions: int,
    enable_prefix_caching: bool,
    gpu_memory_utilization: float,
    max_tokens: int,
) -> list[dict[str, Any]]:
    llm_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "enforce_eager": True,
        "gpu_memory_utilization": gpu_memory_utilization,
        "moe_backend": "flashinfer_cutlass",
        "max_model_len": max(len(prompt_token_ids) + max_tokens + 64, 512),
    }
    if enable_prefix_caching:
        llm_kwargs["enable_prefix_caching"] = True
        llm_kwargs["block_size"] = 16
        llm_kwargs["mamba_block_size"] = 16
    else:
        llm_kwargs["enable_prefix_caching"] = False

    llm = LLM(model, **llm_kwargs)
    try:
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            min_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            ignore_eos=True,
        )
        results: list[dict[str, Any]] = []
        for repetition in range(repetitions):
            start = time.perf_counter()
            output = llm.generate([{"prompt_token_ids": prompt_token_ids}], sampling_params)[0]
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            results.append(
                {
                    "repetition": repetition,
                    "elapsed_ms": elapsed_ms,
                    "num_cached_tokens": int(output.num_cached_tokens or 0),
                    "generated_token_ids": [int(token_id) for token_id in output.outputs[0].token_ids],
                    "generated_text": output.outputs[0].text,
                }
            )
        return results
    finally:
        del llm


def main() -> int:
    args = parse_args()
    tokenizer = load_tokenizer(args.model)
    result_path = pathlib.Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    for token_count in args.token_count:
        prompt_token_ids = build_prompt_token_ids(tokenizer, token_count, args.seed)
        for enable_prefix_caching in (False, True):
            repetitions = run_probe(
                model=args.model,
                prompt_token_ids=prompt_token_ids,
                repetitions=args.repetitions,
                enable_prefix_caching=enable_prefix_caching,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_tokens=args.max_tokens,
            )
            cases.append(
                {
                    "token_count": token_count,
                    "enable_prefix_caching": enable_prefix_caching,
                    "prompt_token_ids": prompt_token_ids,
                    "repetitions": repetitions,
                }
            )

    payload = {
        "model": args.model,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "cases": cases,
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
