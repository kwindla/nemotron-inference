#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pathlib
import time

import torch
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig, SamplingParams

from prefill_profile_common import DEFAULT_MODEL, build_prompt_token_ids, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile one warmed TRT-LLM PyTorch prefill+decode1 request with exact prompt token IDs.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--tokenizer")
    parser.add_argument("--token-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-num-tokens", type=int, default=4608)
    parser.add_argument("--max-seq-len", type=int, default=4608)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--result-path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer_name = args.tokenizer or args.model
    tokenizer = load_tokenizer(tokenizer_name)
    prompt_token_ids = build_prompt_token_ids(tokenizer, args.token_count, args.seed)
    sampling_params = SamplingParams(
        max_tokens=1,
        min_tokens=1,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
    )

    with LLM(
        args.model,
        tokenizer=tokenizer_name,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_batch_size=1,
        max_num_tokens=args.max_num_tokens,
        max_seq_len=args.max_seq_len,
        enable_chunked_prefill=True,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=True,
            free_gpu_memory_fraction=0.8,
            dtype="fp8",
            mamba_ssm_cache_dtype="float32",
        ),
        moe_config=MoeConfig(backend="CUTLASS", max_num_tokens=args.max_num_tokens),
    ) as llm:
        for _ in range(args.warmup_iters):
            llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)

        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        start = time.perf_counter()
        output = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        torch.cuda.cudart().cudaProfilerStop()

        payload = {
            "model": args.model,
            "token_count": args.token_count,
            "elapsed_ms": elapsed_ms,
            "generated_token_ids": [int(token_id) for token_id in output.outputs[0].token_ids],
        }
        if args.result_path:
            result_path = pathlib.Path(args.result_path)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
