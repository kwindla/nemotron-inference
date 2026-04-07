#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "jinja2>=3.1",
#   "transformers>=4.51",
# ]
# ///

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time
from collections.abc import Mapping
from typing import Any


SCRIPT_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_BUILD_DIR = "build-sm120-relwithdebinfo"
DEFAULT_MANIFEST = REPO_ROOT / "artifacts" / "manifests" / "forward_runtime_manifest_nano_rtx5090_unverified.json"
DEFAULT_PROMPT = "Tell me a complicated and hilarious joke in the form of a story."


def load_manifest(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_runtime_value(manifest: dict[str, Any], key: str) -> Any:
    runtime = manifest.get("runtime")
    if isinstance(runtime, dict) and key in runtime:
        return runtime[key]
    return manifest.get(key)


def resolve_server_binary(build_dir: str) -> pathlib.Path:
    return (REPO_ROOT / build_dir / "tools" / "interactive_forward" / "nemotron_interactive_forward_server").resolve()


def load_tokenizer(tokenizer_name: str):
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name, fix_mistral_regex=True, **kwargs)
    except TypeError:
        return AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)


def normalize_token_ids(token_ids: Any) -> list[int]:
    if isinstance(token_ids, Mapping):
        token_ids = token_ids.get("input_ids", [])
    elif hasattr(token_ids, "input_ids"):
        token_ids = getattr(token_ids, "input_ids")
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def render_chat_token_ids(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> list[int]:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    return normalize_token_ids(token_ids)


def build_turn_prompt_token_ids(
    tokenizer: Any,
    history_messages: list[dict[str, str]],
    exact_history_token_ids: list[int],
    user_text: str,
) -> list[int]:
    pending_messages = history_messages + [{"role": "user", "content": user_text}]
    if not exact_history_token_ids:
        return render_chat_token_ids(tokenizer, pending_messages, add_generation_prompt=True)
    rendered_history_token_ids = render_chat_token_ids(tokenizer, history_messages, add_generation_prompt=False)
    rendered_turn_token_ids = render_chat_token_ids(tokenizer, pending_messages, add_generation_prompt=True)
    history_length = len(rendered_history_token_ids)
    if rendered_turn_token_ids[:history_length] != rendered_history_token_ids:
        raise RuntimeError("chat template did not preserve the completed history prefix")
    return exact_history_token_ids + rendered_turn_token_ids[history_length:]


def resolve_eos_token_ids(tokenizer: Any) -> list[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return []
    if isinstance(eos_token_id, int):
        return [eos_token_id]
    return [int(token_id) for token_id in eos_token_id]


def send_command(process: subprocess.Popen[str], command: str) -> dict[str, Any]:
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(command + "\n")
    process.stdin.flush()
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("native server exited unexpectedly")
    payload = json.loads(line)
    if not payload.get("ok", False):
        raise RuntimeError(payload.get("error", "native server returned an error"))
    return payload


def make_prompt(base_prompt: str, run_index: int, vary_prompt: bool) -> str:
    if not vary_prompt:
        return base_prompt
    return base_prompt + f"\n\nBenchmark run marker: {run_index + 1}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-request long-response benchmark against the native interactive forward server.")
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--build-dir", default=DEFAULT_BUILD_DIR)
    parser.add_argument("--tokenizer")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--target-context-tokens", type=int, default=8192)
    parser.add_argument("--prefix-cache", choices=("on", "off"), default="off")
    parser.add_argument("--tenant-namespace", default="native-story-bench")
    parser.add_argument("--serializer-revision", default="native-story-bench-v1")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--vary-prompt", action="store_true")
    parser.add_argument("--result-path", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    tokenizer_name = args.tokenizer or manifest_runtime_value(manifest, "model_id")
    if not isinstance(tokenizer_name, str) or not tokenizer_name:
        raise RuntimeError("manifest does not provide a usable tokenizer name")
    tokenizer = load_tokenizer(tokenizer_name)
    eos_token_ids = resolve_eos_token_ids(tokenizer)
    server_binary = resolve_server_binary(args.build_dir)

    cmd = [
        str(server_binary),
        "--manifest",
        str(args.manifest.resolve()),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--target-context-tokens",
        str(args.target_context_tokens),
        "--prefix-cache",
        args.prefix_cache,
        "--tenant-namespace",
        args.tenant_namespace,
        "--serializer-revision",
        args.serializer_revision,
    ]
    result_path = pathlib.Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        assert process.stdout is not None
        ready_line = process.stdout.readline()
        if not ready_line:
            raise RuntimeError("native server exited during startup")
        ready_payload = json.loads(ready_line)
        if not ready_payload.get("ok", False):
            raise RuntimeError(ready_payload.get("error", "native server failed to start"))

        for warmup_index in range(args.warmup_runs):
            prompt = make_prompt(args.prompt, warmup_index, args.vary_prompt)
            messages: list[dict[str, str]] = []
            committed_token_ids: list[int] = []
            prompt_token_ids = build_turn_prompt_token_ids(tokenizer, messages, committed_token_ids, prompt)
            send_command(
                process,
                "TURN\t{}\t{}\t{}".format(
                    args.max_new_tokens,
                    ",".join(str(token_id) for token_id in eos_token_ids),
                    ",".join(str(token_id) for token_id in prompt_token_ids),
                ),
            )
            send_command(process, "RESET")

        runs: list[dict[str, Any]] = []
        for run_index in range(args.runs):
            prompt = make_prompt(args.prompt, args.warmup_runs + run_index, args.vary_prompt)
            messages: list[dict[str, str]] = []
            committed_token_ids: list[int] = []
            tokenize_start = time.perf_counter()
            prompt_token_ids = build_turn_prompt_token_ids(tokenizer, messages, committed_token_ids, prompt)
            tokenize_ms = (time.perf_counter() - tokenize_start) * 1000.0
            response = send_command(
                process,
                "TURN\t{}\t{}\t{}".format(
                    args.max_new_tokens,
                    ",".join(str(token_id) for token_id in eos_token_ids),
                    ",".join(str(token_id) for token_id in prompt_token_ids),
                ),
            )
            generated_text = tokenizer.decode(
                response["generated_token_ids"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            ttft_ms = (
                tokenize_ms
                + float(response["cache_lookup_ms"])
                + float(response["cache_restore_ms"])
                + float(response["prefill_ms"])
                + float(response["prompt_logits_copy_ms"])
                + float(response["first_token_select_ms"])
            )
            total_ms = tokenize_ms + float(response["total_ms"])
            generated_token_count = int(response["generated_token_count"])
            post_first_toks_s = 0.0
            decode_total_ms = float(response["decode_total_ms"])
            if generated_token_count > 1 and decode_total_ms > 0.0:
                post_first_toks_s = ((generated_token_count - 1) * 1000.0) / decode_total_ms
            total_toks_s = 0.0 if total_ms <= 0.0 else (generated_token_count * 1000.0) / total_ms
            runs.append(
                {
                    "run_index": run_index,
                    "prompt": prompt,
                    "prompt_token_count": int(response["prompt_token_count"]),
                    "generated_token_count": generated_token_count,
                    "generated_text": generated_text,
                    "ttft_ms": ttft_ms,
                    "prefill_ms": float(response["prefill_ms"]),
                    "decode_total_ms": decode_total_ms,
                    "total_ms": total_ms,
                    "total_generated_tokens_per_second": total_toks_s,
                    "post_first_generated_tokens_per_second": post_first_toks_s,
                    "raw_response": response,
                }
            )
            send_command(process, "RESET")

        summary = {
            "ttft_ms_mean": sum(run["ttft_ms"] for run in runs) / len(runs),
            "ttft_ms_min": min(run["ttft_ms"] for run in runs),
            "ttft_ms_max": max(run["ttft_ms"] for run in runs),
            "generated_token_count_mean": sum(run["generated_token_count"] for run in runs) / len(runs),
            "total_generated_tokens_per_second_mean": sum(
                run["total_generated_tokens_per_second"] for run in runs
            ) / len(runs),
            "post_first_generated_tokens_per_second_mean": sum(
                run["post_first_generated_tokens_per_second"] for run in runs
            ) / len(runs),
        }
        payload = {
            "manifest": str(args.manifest.resolve()),
            "tokenizer": tokenizer_name,
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "warmup_runs": args.warmup_runs,
            "runs": runs,
            "summary": summary,
        }
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        send_command(process, "EXIT")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
