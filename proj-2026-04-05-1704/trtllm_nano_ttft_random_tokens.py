#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SERVER = REPO_ROOT / "proj-2026-04-05-1704" / "run_trtllm_nano_serve.sh"
DEFAULT_BENCH = REPO_ROOT / "third_party" / "TensorRT-LLM" / "tensorrt_llm" / "serve" / "scripts" / "benchmark_serving.py"
DEFAULT_VENV = REPO_ROOT / ".venv-trtllm"
DEFAULT_CHECKPOINT = REPO_ROOT / "artifacts" / "checkpoints" / "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"


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


def run_case(args: argparse.Namespace, token_count: int) -> None:
    result_path = pathlib.Path(args.result_dir) / f"trtllm_serve_ttft_prefix{token_count}_out1_{args.tag}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(pathlib.Path(args.python_bin)),
        str(pathlib.Path(args.bench_script)),
        "--backend",
        "openai-chat",
        "--base-url",
        args.base_url,
        "--model",
        args.model_name,
        "--tokenizer",
        args.tokenizer,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(token_count),
        "--random-output-len",
        "1",
        "--random-ids",
        "--tokenize-on-client",
        "--num-prompts",
        str(args.num_prompts),
        "--seed",
        str(args.seed),
        "--trust-remote-code",
        "--disable-tqdm",
        "--save-result",
        "--result-filename",
        result_path.name,
        "--result-dir",
        str(result_path.parent),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRT-LLM Nano TTFT serving benchmarks on fixed random token counts.")
    parser.add_argument("--token-count", type=int, action="append", required=True)
    parser.add_argument("--num-prompts", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--base-url", default="http://127.0.0.1:8012")
    parser.add_argument("--model-name", default="nemotron-nano-trtllm")
    parser.add_argument("--tokenizer", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--server-timeout-s", type=float, default=900.0)
    parser.add_argument("--tag", default="20260406")
    parser.add_argument("--result-dir", default=str(REPO_ROOT / "artifacts" / "benchmarks"))
    parser.add_argument("--server-script", default=str(DEFAULT_SERVER))
    parser.add_argument("--bench-script", default=str(DEFAULT_BENCH))
    parser.add_argument("--python-bin", default=str(DEFAULT_VENV / "bin" / "python"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server_proc = subprocess.Popen(
        [str(pathlib.Path(args.server_script))],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        env={
            **os.environ,
            "TRTLLM_PORT": str(args.port),
        },
    )
    try:
        wait_for_server(args.base_url, args.server_timeout_s)
        for token_count in args.token_count:
            run_case(args, token_count)
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
