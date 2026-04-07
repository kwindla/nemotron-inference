#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "transformers>=4.51",
# ]
# ///

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any


SCRIPT_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
DEFAULT_PROMPT = "Tell me a complicated and hilarious joke in the form of a story."


def load_tokenizer(tokenizer_name: str):
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name, fix_mistral_regex=True, **kwargs)
    except TypeError:
        return AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)


def wait_for_server(base_url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    models_url = f"{base_url}/v1/models"
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(models_url, timeout=2.0) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(1.0)
    raise RuntimeError(f"Server did not become ready at {models_url}: {last_error}")


def make_prompt(base_prompt: str, run_index: int, vary_prompt: bool) -> str:
    if not vary_prompt:
        return base_prompt
    return base_prompt + f"\n\nBenchmark run marker: {run_index + 1}."


def collect_stream_response(
    *,
    base_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_at = time.perf_counter()
    first_token_at: float | None = None
    pieces: list[str] = []
    chunk_count = 0
    with urllib.request.urlopen(request, timeout=600.0) as response:
        for raw_line in response:
            if not raw_line:
                continue
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            choices = event.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if isinstance(content, str) and content:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                pieces.append(content)
                chunk_count += 1
                continue
            if isinstance(content, list):
                text = "".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and isinstance(item.get("text"), str)
                )
                if text:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    pieces.append(text)
                    chunk_count += 1
                    continue
            reasoning = delta.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                pieces.append(reasoning)
                chunk_count += 1
    ended_at = time.perf_counter()
    ttft_ms = None if first_token_at is None else (first_token_at - started_at) * 1000.0
    total_ms = (ended_at - started_at) * 1000.0
    return {
        "text": "".join(pieces),
        "chunk_count": chunk_count,
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a streaming single-request story benchmark against an OpenAI chat server.")
    parser.add_argument("--server-script")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tokenizer", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--vary-prompt", action="store_true")
    parser.add_argument("--server-timeout-s", type=float, default=900.0)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--port-env", default="")
    parser.add_argument("--port", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer)
    result_path = pathlib.Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    server_proc: subprocess.Popen[bytes] | None = None
    if args.server_script:
        env = dict(os.environ)
        if args.port_env and args.port:
            env[args.port_env] = str(args.port)
        server_proc = subprocess.Popen(
            [args.server_script],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            env=env,
        )
        wait_for_server(args.base_url, args.server_timeout_s)

    try:
        for warmup_index in range(args.warmup_runs):
            prompt = make_prompt(args.prompt, warmup_index, args.vary_prompt)
            collect_stream_response(
                base_url=args.base_url,
                model_name=args.model_name,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

        runs: list[dict[str, Any]] = []
        for run_index in range(args.runs):
            prompt = make_prompt(args.prompt, args.warmup_runs + run_index, args.vary_prompt)
            stream_result = collect_stream_response(
                base_url=args.base_url,
                model_name=args.model_name,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            output_token_count = len(tokenizer.encode(stream_result["text"], add_special_tokens=False))
            ttft_ms = float(stream_result["ttft_ms"]) if stream_result["ttft_ms"] is not None else None
            total_ms = float(stream_result["total_ms"])
            post_first_ms = None if ttft_ms is None else max(total_ms - ttft_ms, 0.0)
            total_toks_s = 0.0 if total_ms <= 0.0 else (output_token_count * 1000.0) / total_ms
            post_first_toks_s = 0.0
            if post_first_ms and post_first_ms > 0.0 and output_token_count > 1:
                post_first_toks_s = ((output_token_count - 1) * 1000.0) / post_first_ms
            runs.append(
                {
                    "run_index": run_index,
                    "prompt": prompt,
                    "generated_text": stream_result["text"],
                    "generated_token_count": output_token_count,
                    "chunk_count": int(stream_result["chunk_count"]),
                    "ttft_ms": ttft_ms,
                    "total_ms": total_ms,
                    "total_generated_tokens_per_second": total_toks_s,
                    "post_first_generated_tokens_per_second": post_first_toks_s,
                }
            )

        summary = {
            "ttft_ms_mean": sum(run["ttft_ms"] for run in runs if run["ttft_ms"] is not None) / len(runs),
            "ttft_ms_min": min(run["ttft_ms"] for run in runs if run["ttft_ms"] is not None),
            "ttft_ms_max": max(run["ttft_ms"] for run in runs if run["ttft_ms"] is not None),
            "generated_token_count_mean": sum(run["generated_token_count"] for run in runs) / len(runs),
            "total_generated_tokens_per_second_mean": sum(
                run["total_generated_tokens_per_second"] for run in runs
            ) / len(runs),
            "post_first_generated_tokens_per_second_mean": sum(
                run["post_first_generated_tokens_per_second"] for run in runs
            ) / len(runs),
        }
        payload = {
            "base_url": args.base_url,
            "model_name": args.model_name,
            "tokenizer": args.tokenizer,
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "warmup_runs": args.warmup_runs,
            "runs": runs,
            "summary": summary,
        }
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    finally:
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
