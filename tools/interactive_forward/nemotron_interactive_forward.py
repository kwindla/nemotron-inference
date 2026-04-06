#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "jinja2>=3.1",
#   "sentencepiece>=0.2",
#   "transformers>=4.51",
# ]
# ///

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_BUILD_DIR = "build"
DEFAULT_MANIFEST = (
    REPO_ROOT / "artifacts" / "manifests" / "forward_runtime_manifest_unverified.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive chat-style forward-pass tester for the direct Nemotron runtime path."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the runtime manifest JSON.",
    )
    parser.add_argument(
        "--build-dir",
        default=DEFAULT_BUILD_DIR,
        help="Build directory that contains nemotron_interactive_forward_server.",
    )
    parser.add_argument(
        "--server-binary",
        type=Path,
        help="Explicit path to nemotron_interactive_forward_server.",
    )
    parser.add_argument(
        "--tokenizer",
        help="Tokenizer name or path. Defaults to the manifest model_id.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per turn.",
    )
    parser.add_argument(
        "--target-context-tokens",
        type=int,
        default=8192,
        help="Request-context token capacity for the helper.",
    )
    parser.add_argument(
        "--graph-bytes",
        type=int,
        default=4 * 1024 * 1024 * 1024,
        help="Graph memory budget passed into RuntimeBootstrapOptions.",
    )
    parser.add_argument(
        "--system",
        default="",
        help="Optional system prompt inserted at the start of the chat history.",
    )
    parser.add_argument(
        "--eos-token-id",
        type=int,
        action="append",
        default=[],
        help="Override EOS token ids. Repeat to pass multiple ids.",
    )
    parser.add_argument(
        "--once",
        help="Run a single user turn and exit.",
    )
    parser.add_argument(
        "--show-token-ids",
        action="store_true",
        help="Print generated token ids for each turn.",
    )
    parser.add_argument(
        "--show-step-stats",
        action="store_true",
        help="Print per-step decode latencies.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def manifest_runtime_value(manifest: dict[str, Any], key: str) -> Any:
    runtime = manifest.get("runtime")
    if isinstance(runtime, dict) and key in runtime:
        return runtime[key]
    return manifest.get(key)


def resolve_server_binary(args: argparse.Namespace) -> Path:
    if args.server_binary is not None:
        return args.server_binary.resolve()
    return (
        REPO_ROOT
        / args.build_dir
        / "tools"
        / "interactive_forward"
        / "nemotron_interactive_forward_server"
    ).resolve()


def load_tokenizer(tokenizer_name: str):
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name, fix_mistral_regex=True, **kwargs)
    except TypeError:
        return AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)


def normalize_token_ids(token_ids: Any) -> list[int]:
    if hasattr(token_ids, "input_ids"):
        token_ids = getattr(token_ids, "input_ids")
    if isinstance(token_ids, dict):
        token_ids = token_ids.get("input_ids", [])
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def render_chat_token_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    if getattr(tokenizer, "chat_template", None) is None:
        raise RuntimeError(
            "tokenizer does not expose a chat template; pass a chat-capable tokenizer"
        )
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    return normalize_token_ids(token_ids)


def resolve_eos_token_ids(args: argparse.Namespace, tokenizer: Any) -> list[int]:
    if args.eos_token_id:
        return list(args.eos_token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return []
    if isinstance(eos_token_id, int):
        return [eos_token_id]
    return [int(token_id) for token_id in eos_token_id]


def send_command(process: subprocess.Popen[str], command: str) -> dict[str, Any]:
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("interactive helper process does not have connected pipes")
    process.stdin.write(command + "\n")
    process.stdin.flush()
    line = process.stdout.readline()
    if line == "":
        raise RuntimeError("interactive helper exited unexpectedly")
    payload = json.loads(line)
    if not payload.get("ok", False):
        raise RuntimeError(payload.get("error", "interactive helper returned an error"))
    return payload


def format_ms(value: float) -> str:
    return f"{value:.3f} ms"


def format_tokens_per_second(value: float) -> str:
    return f"{value:.3f} tok/s"


def print_help() -> None:
    print("Commands:")
    print("  /help          Show this help")
    print("  /reset         Clear chat history")
    print("  /quit          Exit")
    print("  /cache ...     Unsupported in this helper")


def print_turn_summary(
    response: dict[str, Any],
    tokenize_ms: float,
    detokenize_ms: float,
    generated_text: str,
    args: argparse.Namespace,
) -> None:
    print()
    print(f"assistant> {generated_text if generated_text else '<empty>'}")
    print(
        "timing: "
        f"tokenize={format_ms(tokenize_ms)} "
        f"prefill={format_ms(float(response['prefill_ms']))} "
        f"last_row_copy={format_ms(float(response['prompt_logits_copy_ms']))} "
        f"select={format_ms(float(response['first_token_select_ms']))} "
        f"decode={format_ms(float(response['decode_total_ms']))} "
        f"detokenize={format_ms(detokenize_ms)} "
        f"total={format_ms(float(response['total_ms']))}"
    )
    print(
        "tokens: "
        f"prompt={response['prompt_token_count']} "
        f"generated={response['generated_token_count']} "
        f"sequence_after={response['sequence_length_after']}/{response['max_context_tokens']}"
    )
    if response["generated_token_count"]:
        print(
            "throughput: "
            f"decode={format_tokens_per_second(float(response['decode_tokens_per_second']))} "
            f"end_to_end={format_tokens_per_second(float(response['total_generated_tokens_per_second']))}"
        )
    if response.get("hit_capacity_limit"):
        print("status: hit context capacity limit before max_new_tokens")
    if response.get("hit_eos"):
        print("status: hit EOS")
    if args.show_token_ids:
        print(f"generated_token_ids: {response['generated_token_ids']}")
    if args.show_step_stats and response["decode_step_ms"]:
        print(f"decode_step_ms: {response['decode_step_ms']}")
    runtime_stats = response.get("runtime_stats", {})
    print(
        "runtime: "
        f"dense_native={runtime_stats.get('dense_native_success', 0)} "
        f"dense_fallback={runtime_stats.get('dense_reference_fallbacks', 0)} "
        f"scaled_fp8_native={runtime_stats.get('scaled_fp8_native_success', 0)} "
        f"scaled_fp8_fallback={runtime_stats.get('scaled_fp8_reference_fallbacks', 0)} "
        f"attn_decode_plan_hits={runtime_stats.get('attention_decode_plan_hits', 0)} "
        f"forward_graph_replays={runtime_stats.get('forward_graph_replays', 0)}"
    )
    print()


class InteractiveSession:
    def __init__(
        self,
        process: subprocess.Popen[str],
        tokenizer: Any,
        eos_token_ids: list[int],
        args: argparse.Namespace,
        ready_payload: dict[str, Any],
    ) -> None:
        self.process = process
        self.tokenizer = tokenizer
        self.eos_token_ids = eos_token_ids
        self.args = args
        self.ready_payload = ready_payload
        self.messages: list[dict[str, str]] = []
        if args.system:
            self.messages.append({"role": "system", "content": args.system})

    def reset(self) -> None:
        self.messages = []
        if self.args.system:
            self.messages.append({"role": "system", "content": self.args.system})
        send_command(self.process, "RESET")

    def run_turn(self, user_text: str) -> None:
        pending_messages = self.messages + [{"role": "user", "content": user_text}]
        tokenize_start = time.perf_counter()
        prompt_token_ids = render_chat_token_ids(
            self.tokenizer,
            pending_messages,
            add_generation_prompt=True,
        )
        tokenize_end = time.perf_counter()
        if len(prompt_token_ids) > int(self.ready_payload["target_context_tokens"]):
            raise RuntimeError(
                "rendered prompt exceeds target context tokens: "
                f"{len(prompt_token_ids)} > {self.ready_payload['target_context_tokens']}"
            )
        eos_csv = ",".join(str(token_id) for token_id in self.eos_token_ids)
        prompt_csv = ",".join(str(token_id) for token_id in prompt_token_ids)
        response = send_command(
            self.process,
            f"TURN\t{self.args.max_new_tokens}\t{eos_csv}\t{prompt_csv}",
        )
        detokenize_start = time.perf_counter()
        generated_token_ids = [int(token_id) for token_id in response["generated_token_ids"]]
        generated_text = self.tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=False,
        )
        detokenize_end = time.perf_counter()
        self.messages = pending_messages + [{"role": "assistant", "content": generated_text}]
        print_turn_summary(
            response,
            tokenize_ms=(tokenize_end - tokenize_start) * 1000.0,
            detokenize_ms=(detokenize_end - detokenize_start) * 1000.0,
            generated_text=generated_text,
            args=self.args,
        )

    def repl(self) -> None:
        print("Type /help for commands.")
        while True:
            try:
                user_text = input("user> ").strip()
            except EOFError:
                print()
                break
            if not user_text:
                continue
            if user_text == "/help":
                print_help()
                continue
            if user_text == "/quit":
                break
            if user_text == "/reset":
                self.reset()
                print("conversation reset")
                continue
            if user_text.startswith("/cache"):
                print("prefix-cache reuse is not implemented in this helper")
                continue
            self.run_turn(user_text)

    def close(self) -> None:
        try:
            send_command(self.process, "EXIT")
        except Exception:
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    tokenizer_name = args.tokenizer or manifest_runtime_value(manifest, "model_id")
    if not tokenizer_name:
        raise RuntimeError("manifest does not include runtime.model_id and --tokenizer was not set")
    server_binary = resolve_server_binary(args)
    if not server_binary.exists():
        raise RuntimeError(f"server binary not found: {server_binary}")

    tokenizer = load_tokenizer(str(tokenizer_name))
    eos_token_ids = resolve_eos_token_ids(args, tokenizer)

    process = subprocess.Popen(
        [
            str(server_binary),
            "--manifest",
            str(args.manifest.resolve()),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--target-context-tokens",
            str(args.target_context_tokens),
            "--graph-bytes",
            str(args.graph_bytes),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
    )

    if process.stdout is None:
        raise RuntimeError("interactive helper stdout pipe was not created")
    ready_line = process.stdout.readline()
    if ready_line == "":
        raise RuntimeError("interactive helper exited before signaling readiness")
    ready_payload = json.loads(ready_line)
    if not ready_payload.get("ok", False):
        raise RuntimeError(ready_payload.get("error", "interactive helper failed during startup"))

    print(
        "helper: "
        f"model_id={ready_payload['model_id']} "
        f"target_context_tokens={ready_payload['target_context_tokens']} "
        f"graph_bytes={ready_payload['graph_bytes']} "
        f"env_build={format_ms(float(ready_payload['environment_build_ms']))} "
        f"model_build={format_ms(float(ready_payload['model_build_ms']))}"
    )
    print("helper: direct Create(...) path only; prefix-cache reuse is unsupported")

    session = InteractiveSession(process, tokenizer, eos_token_ids, args, ready_payload)
    try:
        if args.once is not None:
            session.run_turn(args.once)
        else:
            session.repl()
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
