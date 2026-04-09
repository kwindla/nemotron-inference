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
import subprocess
import sys
import time
from collections.abc import Mapping
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive chat-style forward-pass tester for the native SM120 runtime path."
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
        help="Build directory that contains the native helper.",
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
        default=4096,
        help="Maximum generated tokens per turn.",
    )
    parser.add_argument(
        "--target-context-tokens",
        type=int,
        default=8192,
        help="Request-context token capacity for the native helper.",
    )
    parser.add_argument(
        "--prefix-cache",
        choices=("on", "off"),
        default="on",
        help="Enable or disable the runtime prefix cache.",
    )
    parser.add_argument(
        "--system",
        default="",
        help="Optional system prompt inserted at the start of the chat history.",
    )
    parser.add_argument(
        "--serializer-revision",
        default="interactive-chat-v1",
        help="Serializer revision string used for prefix-cache identity.",
    )
    parser.add_argument(
        "--tenant-namespace",
        default="interactive-forward",
        help="Tenant namespace used for prefix-cache identity.",
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
        help="Print the full per-step decode latency list.",
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
    if isinstance(token_ids, Mapping):
        token_ids = token_ids.get("input_ids", [])
    elif hasattr(token_ids, "input_ids"):
        token_ids = getattr(token_ids, "input_ids")
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


def extract_visible_assistant_text(raw_text: str) -> str:
    if not raw_text:
        return raw_text
    if "</think>" in raw_text:
        visible = raw_text.rsplit("</think>", maxsplit=1)[-1].strip()
        if visible:
            return visible
    return raw_text.strip()


def build_turn_prompt_token_ids(
    tokenizer: Any,
    history_messages: list[dict[str, str]],
    exact_history_token_ids: list[int],
    user_text: str,
) -> list[int]:
    pending_messages = history_messages + [{"role": "user", "content": user_text}]
    if not exact_history_token_ids:
        return render_chat_token_ids(
            tokenizer,
            pending_messages,
            add_generation_prompt=True,
        )

    rendered_history_token_ids = render_chat_token_ids(
        tokenizer,
        history_messages,
        add_generation_prompt=False,
    )
    rendered_turn_token_ids = render_chat_token_ids(
        tokenizer,
        pending_messages,
        add_generation_prompt=True,
    )
    history_length = len(rendered_history_token_ids)
    if len(rendered_turn_token_ids) < history_length or rendered_turn_token_ids[:history_length] != rendered_history_token_ids:
        raise RuntimeError("chat template did not preserve the completed history prefix")
    if exact_history_token_ids != rendered_history_token_ids:
        return rendered_turn_token_ids
    return exact_history_token_ids + rendered_turn_token_ids[history_length:]


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
    print("  /reset         Clear chat history and runtime prefix cache")
    print("  /cache on      Enable runtime prefix cache")
    print("  /cache off     Disable runtime prefix cache")
    print("  /quit          Exit")


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
        "timing:"
        f" tokenize={format_ms(tokenize_ms)}"
        f" cache_lookup={format_ms(float(response['cache_lookup_ms']))}"
        f" cache_restore={format_ms(float(response['cache_restore_ms']))}"
        f" prefill={format_ms(float(response['prefill_ms']))}"
        f" logits_copy={format_ms(float(response['prompt_logits_copy_ms']))}"
        f" first_select={format_ms(float(response['first_token_select_ms']))}"
        f" decode_total={format_ms(float(response['decode_total_ms']))}"
        f" detokenize={format_ms(detokenize_ms)}"
        f" total={format_ms(float(response['total_ms']) + tokenize_ms + detokenize_ms)}"
    )
    print(
        "tokens:"
        f" prompt={response['prompt_token_count']}"
        f" matched_prefix={response['matched_prefix_tokens']}"
        f" prefill={response['prefill_token_count']}"
        f" generated={response['generated_token_count']}"
        f" sequence_after={response['sequence_length_after']}/{response['max_context_tokens']}"
    )
    print(
        "cache:"
        f" enabled={'yes' if response['prefix_cache_enabled'] else 'no'}"
        f" mode={response['cache_mode']}"
        f" source={response['cache_lookup_source']}"
        f" nodes={response['prefix_cache']['node_count']}"
        f" bytes={response['prefix_cache']['current_bytes']}"
    )
    print(
        "decode:"
        f" first_step={format_ms(float(response['first_decode_step_ms']))}"
        f" mean={format_ms(float(response['decode_mean_ms']))}"
        f" min={format_ms(float(response['decode_min_ms']))}"
        f" max={format_ms(float(response['decode_max_ms']))}"
        f" decode_rate={format_tokens_per_second(float(response['decode_tokens_per_second']))}"
        f" end_to_end_rate={format_tokens_per_second(float(response['total_generated_tokens_per_second']))}"
    )
    print(
        "ops:"
        f" attention_native_runs={response['attention']['native_multi_token_runs']}"
        f" attention_row_replay_runs={response['attention']['row_replay_runs']}"
        f" mamba_native_runs={response['mamba']['native_multi_token_runs']}"
        f" mamba_row_replay_runs={response['mamba']['row_replay_runs']}"
        f" expert_native_runs={response['expert']['native_multi_token_runs']}"
        f" expert_row_replay_runs={response['expert']['row_replay_runs']}"
    )
    print(
        "kernel:"
        f" dense_exec={response['linear']['dense_fastpath_execute']}"
        f" nvfp4_exec={response['linear']['nvfp4_fastpath_execute']}"
        f" fp8_exec={response['linear']['scaled_fp8_fastpath_execute']}"
        f" staged_bytes={response['expert_staging']['total_bytes_uploaded']}"
        f" staging_calls={response['expert_staging']['total_staging_calls']}"
    )
    if args.show_token_ids:
        print(f"generated_token_ids: {response['generated_token_ids']}")
    if args.show_step_stats:
        step_stats = ", ".join(f"{value:.3f}" for value in response["decode_step_ms"])
        print(f"decode_step_ms: {step_stats}")
    print()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        print(f"interactive_forward: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    server_binary = resolve_server_binary(args)
    if not server_binary.exists():
        print(
            "interactive_forward: native helper not found: "
            f"{server_binary}\nBuild it first, for example:\n"
            f"  cmake --build {args.build_dir} --target nemotron_interactive_forward_server -j",
            file=sys.stderr,
        )
        return 1

    manifest = load_manifest(manifest_path)
    tokenizer_name = args.tokenizer or manifest_runtime_value(manifest, "model_id")
    if not isinstance(tokenizer_name, str) or not tokenizer_name:
        print(
            "interactive_forward: manifest does not provide a usable model_id for tokenizer loading",
            file=sys.stderr,
        )
        return 1
    tokenizer = load_tokenizer(tokenizer_name)
    eos_token_ids = resolve_eos_token_ids(args, tokenizer)

    server_cmd = [
        str(server_binary),
        "--manifest",
        str(manifest_path),
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

    process = subprocess.Popen(
        server_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    try:
        assert process.stdout is not None
        ready_line = process.stdout.readline()
        if ready_line == "":
            raise RuntimeError("interactive helper exited during startup")
        ready_payload = json.loads(ready_line)
        if not ready_payload.get("ok", False):
            raise RuntimeError(ready_payload.get("error", "interactive helper failed to start"))

        print(
            "interactive_forward:"
            f" model_id={ready_payload['model_id']}"
            f" prefix_cache={'on' if ready_payload['prefix_cache_enabled'] else 'off'}"
            f" conversation_id={ready_payload['conversation_id']}"
        )
        if args.once is None:
            print_help()

        messages: list[dict[str, str]] = []
        committed_token_ids: list[int] = []
        if args.system:
            messages.append({"role": "system", "content": args.system})

        def run_turn(user_text: str) -> None:
            nonlocal messages
            nonlocal committed_token_ids

            pending_messages = messages + [{"role": "user", "content": user_text}]

            tokenize_start = time.perf_counter()
            prompt_token_ids = build_turn_prompt_token_ids(
                tokenizer,
                messages,
                committed_token_ids,
                user_text,
            )
            tokenize_ms = (time.perf_counter() - tokenize_start) * 1000.0

            command = "TURN\t{}\t{}\t{}".format(
                args.max_new_tokens,
                ",".join(str(token_id) for token_id in eos_token_ids),
                ",".join(str(token_id) for token_id in prompt_token_ids),
            )
            response = send_command(process, command)

            detokenize_start = time.perf_counter()
            raw_generated_text = tokenizer.decode(
                response["generated_token_ids"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            detokenize_ms = (time.perf_counter() - detokenize_start) * 1000.0
            generated_text = extract_visible_assistant_text(raw_generated_text)

            messages = pending_messages + [{"role": "assistant", "content": generated_text}]
            committed_token_ids = render_chat_token_ids(
                tokenizer,
                messages,
                add_generation_prompt=False,
            )
            print_turn_summary(response, tokenize_ms, detokenize_ms, generated_text, args)

        if args.once is not None:
            run_turn(args.once)
            send_command(process, "EXIT")
            return 0

        while True:
            try:
                user_text = input("user> ").strip()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
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
                send_command(process, "RESET")
                messages = []
                committed_token_ids = []
                if args.system:
                    messages.append({"role": "system", "content": args.system})
                print("interactive_forward: history and prefix cache reset")
                continue
            if user_text.startswith("/cache "):
                _, _, value = user_text.partition(" ")
                value = value.strip().lower()
                if value not in {"on", "off"}:
                    print("interactive_forward: usage: /cache on|off")
                    continue
                send_command(process, f"SET_PREFIX_CACHE\t{value}")
                print(f"interactive_forward: prefix cache {value}")
                continue

            run_turn(user_text)

        try:
            send_command(process, "EXIT")
        except Exception:
            pass
        return 0
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
