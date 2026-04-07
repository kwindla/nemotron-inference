#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pathlib
import time

from vllm import LLM, SamplingParams
from vllm.config.profiler import ProfilerConfig

from prefill_profile_common import DEFAULT_MODEL, build_prompt_token_ids, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile one warmed vLLM prefill+decode1 request with exact prompt token IDs.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--tokenizer")
    parser.add_argument("--token-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=4608)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--result-path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer_name = args.tokenizer or args.model
    tokenizer = load_tokenizer(tokenizer_name)
    prompt_token_ids = build_prompt_token_ids(tokenizer, args.token_count, args.seed)

    llm = LLM(
        args.model,
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        moe_backend="flashinfer_cutlass",
        enable_prefix_caching=False,
        max_model_len=args.max_model_len,
        profiler_config=ProfilerConfig(profiler="cuda"),
    )
    try:
        sampling_params = SamplingParams(
            max_tokens=1,
            min_tokens=1,
            temperature=0.0,
            top_p=1.0,
            ignore_eos=True,
        )
        inputs = [{"prompt_token_ids": prompt_token_ids}]
        for _ in range(args.warmup_iters):
            llm.generate(inputs, sampling_params)

        llm.start_profile()
        start = time.perf_counter()
        output = llm.generate(inputs, sampling_params)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        llm.stop_profile()

        payload = {
            "model": args.model,
            "token_count": args.token_count,
            "elapsed_ms": elapsed_ms,
            "num_cached_tokens": int(output.num_cached_tokens or 0),
            "generated_token_ids": [int(token_id) for token_id in output.outputs[0].token_ids],
        }
        if args.result_path:
            result_path = pathlib.Path(args.result_path)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0
    finally:
        del llm


if __name__ == "__main__":
    raise SystemExit(main())
