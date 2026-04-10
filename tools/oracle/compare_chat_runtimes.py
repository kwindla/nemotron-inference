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
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
import urllib.parse
from dataclasses import asdict, dataclass, field
from typing import Any


SCRIPT_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_BUILD_DIR = "build-sm120-relwithdebinfo"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "artifacts"
    / "manifests"
    / "forward_runtime_manifest_nano_rtx5090_unverified.json"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "artifacts" / "checkpoints" / "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
)
DEFAULT_VLLM_SERVER = REPO_ROOT / "proj-2026-04-05-1704" / "run_vllm_nano_serve_local.sh"
DEFAULT_TRTLLM_SERVER = REPO_ROOT / "proj-2026-04-05-1704" / "run_trtllm_nano_serve.sh"
DEFAULT_RESULT_PATH = (
    REPO_ROOT / "artifacts" / "benchmarks" / "chat_runtime_parity_20260410.json"
)


@dataclass
class TurnSpec:
    user: str
    max_tokens: int = 64


@dataclass
class CaseSpec:
    name: str
    turns: list[TurnSpec]
    system: str = ""


@dataclass
class TurnResult:
    prompt_messages: list[dict[str, str]]
    raw_text: str
    visible_text: str
    normalized_visible_text: str
    visible_token_ids: list[int]
    normalized_visible_token_ids: list[int]
    generated_token_ids: list[int] | None = None
    prompt_token_ids: list[int] | None = None
    cache_mode: str | None = None


@dataclass
class RuntimeCaseResult:
    runtime: str
    case_name: str
    turns: list[TurnResult] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare greedy chat outputs across native runtime, vLLM, and TRT-LLM."
    )
    parser.add_argument(
        "--result-path",
        type=pathlib.Path,
        default=DEFAULT_RESULT_PATH,
        help="Where to write the JSON comparison artifact.",
    )
    parser.add_argument(
        "--tokenizer",
        default=str(DEFAULT_CHECKPOINT),
        help="Tokenizer path or model id used to render the chat template.",
    )
    parser.add_argument(
        "--runtimes",
        default="native,vllm,trtllm",
        help="Comma-separated runtimes to run.",
    )
    parser.add_argument(
        "--shared-history-from",
        default="vllm",
        help="Runtime whose visible assistant outputs are used as shared history for later turns.",
    )
    parser.add_argument(
        "--native-build-dir",
        default=DEFAULT_BUILD_DIR,
        help="Build directory containing nemotron_interactive_forward_server.",
    )
    parser.add_argument(
        "--native-manifest",
        type=pathlib.Path,
        default=DEFAULT_MANIFEST,
        help="Manifest passed to the native helper.",
    )
    parser.add_argument(
        "--native-prefix-cache",
        choices=("on", "off"),
        default="off",
        help="Enable or disable native runtime prefix cache during parity runs.",
    )
    parser.add_argument(
        "--native-target-context-tokens",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--native-server-binary",
        type=pathlib.Path,
        help="Explicit path to nemotron_interactive_forward_server.",
    )
    parser.add_argument(
        "--native-env",
        action="append",
        default=[],
        help="Extra KEY=VALUE environment overrides for the native helper. Repeatable.",
    )
    parser.add_argument(
        "--vllm-server-script",
        type=pathlib.Path,
        default=DEFAULT_VLLM_SERVER,
    )
    parser.add_argument(
        "--vllm-base-url",
        default="http://127.0.0.1:8011",
    )
    parser.add_argument(
        "--vllm-model-name",
        default="nemotron-nano-vllm",
    )
    parser.add_argument(
        "--trtllm-server-script",
        type=pathlib.Path,
        default=DEFAULT_TRTLLM_SERVER,
    )
    parser.add_argument(
        "--trtllm-base-url",
        default="http://127.0.0.1:8012",
    )
    parser.add_argument(
        "--trtllm-model-name",
        default="nemotron-nano-trtllm",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--server-timeout-s",
        type=float,
        default=900.0,
    )
    return parser.parse_args()


def load_tokenizer(tokenizer_name: str):
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name, fix_mistral_regex=True, **kwargs)
    except TypeError:
        return AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)


def normalize_token_ids(token_ids: Any) -> list[int]:
    if hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids
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
    if exact_history_token_ids != rendered_history_token_ids:
        return rendered_turn_token_ids
    return exact_history_token_ids + rendered_turn_token_ids[history_length:]


def extract_visible_assistant_text(raw_text: str) -> str:
    if not raw_text:
        return raw_text
    if "</think>" in raw_text:
        visible = raw_text.rsplit("</think>", maxsplit=1)[-1].strip()
        if visible:
            return visible
    return raw_text.strip()


def normalize_visible_text_for_oracle(text: str) -> str:
    normalized = text
    for token in (
        "<|im_end|>",
        "<|im_start|>",
        "<｜end▁of▁sentence｜>",
        "<｜Assistant｜>",
        "<｜User｜>",
    ):
        normalized = normalized.replace(token, "")
    return normalized.strip()


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
    raise RuntimeError(f"server did not become ready at {models_url}: {last_error}")


def openai_chat_completion(
    *,
    base_url: str,
    model_name: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
    choices = payload.get("choices", [])
    if not choices:
        raise RuntimeError(f"no choices in response: {payload}")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    raise RuntimeError(f"unexpected response content: {content!r}")


class NativeRuntime:
    def __init__(self, args: argparse.Namespace, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._process: subprocess.Popen[str] | None = None
        if args.native_server_binary is not None:
            self._binary = args.native_server_binary.resolve()
        else:
            self._binary = (
                REPO_ROOT
                / args.native_build_dir
                / "tools"
                / "interactive_forward"
                / "nemotron_interactive_forward_server"
            ).resolve()
        self._manifest = args.native_manifest.resolve()
        self._prefix_cache = args.native_prefix_cache
        self._target_context_tokens = args.native_target_context_tokens
        self._extra_env = dict(parse_env_overrides(args.native_env))

    def start(self) -> None:
        cmd = [
            str(self._binary),
            "--manifest",
            str(self._manifest),
            "--prefix-cache",
            self._prefix_cache,
            "--target-context-tokens",
            str(self._target_context_tokens),
        ]
        env = dict(os.environ)
        env.update(self._extra_env)
        self._process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        ready = self._read_line()
        if not ready.get("ok", False) or ready.get("status") != "ready":
            raise RuntimeError(f"unexpected native helper startup response: {ready}")

    def stop(self) -> None:
        if self._process is None:
            return
        if self._process.stdin is not None:
            try:
                self._process.stdin.write("QUIT\n")
                self._process.stdin.flush()
            except OSError:
                pass
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
        self._process = None

    def reset(self) -> None:
        self._send_line("RESET")

    def run_turn(
        self,
        history_messages: list[dict[str, str]],
        exact_history_token_ids: list[int],
        *,
        user_text: str,
        max_tokens: int,
    ) -> TurnResult:
        prompt_token_ids = build_turn_prompt_token_ids(
            self._tokenizer,
            history_messages,
            exact_history_token_ids,
            user_text,
        )
        eos_ids = getattr(self._tokenizer, "eos_token_id", None)
        if eos_ids is None:
            eos_csv = ""
        elif isinstance(eos_ids, int):
            eos_csv = str(eos_ids)
        else:
            eos_csv = ",".join(str(int(token_id)) for token_id in eos_ids)
        prompt_csv = ",".join(str(token_id) for token_id in prompt_token_ids)
        payload = self._send_line(f"TURN\t{max_tokens}\t{eos_csv}\t{prompt_csv}")
        raw_text = self._tokenizer.decode(payload["generated_token_ids"], skip_special_tokens=False)
        visible_text = extract_visible_assistant_text(raw_text)
        normalized_visible_text = normalize_visible_text_for_oracle(visible_text)
        return TurnResult(
            prompt_messages=history_messages + [{"role": "user", "content": user_text}],
            raw_text=raw_text,
            visible_text=visible_text,
            normalized_visible_text=normalized_visible_text,
            visible_token_ids=self._tokenizer.encode(visible_text, add_special_tokens=False),
            normalized_visible_token_ids=self._tokenizer.encode(
                normalized_visible_text,
                add_special_tokens=False,
            ),
            generated_token_ids=[int(value) for value in payload["generated_token_ids"]],
            prompt_token_ids=[int(value) for value in prompt_token_ids],
            cache_mode=payload.get("cache_mode"),
        )

    def _send_line(self, line: str) -> dict[str, Any]:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("native helper is not running")
        self._process.stdin.write(line + "\n")
        self._process.stdin.flush()
        return self._read_line()

    def _read_line(self) -> dict[str, Any]:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("native helper is not running")
        response = self._process.stdout.readline()
        if response == "":
            stderr_text = ""
            if self._process.stderr is not None:
                stderr_text = self._process.stderr.read()
            raise RuntimeError(f"native helper exited unexpectedly: {stderr_text}")
        payload = json.loads(response)
        if not payload.get("ok", False):
            raise RuntimeError(payload.get("error", "native helper returned an error"))
        return payload


class OpenAIRuntime:
    def __init__(
        self,
        *,
        server_script: pathlib.Path,
        base_url: str,
        model_name: str,
        timeout_s: float,
        port_env: str,
        port: str,
    ) -> None:
        self._server_script = server_script.resolve()
        self._base_url = base_url
        self._model_name = model_name
        self._timeout_s = timeout_s
        self._port_env = port_env
        self._port = port
        self._process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        env = dict(os.environ)
        env[self._port_env] = self._port
        self._process = subprocess.Popen(
            [str(self._server_script)],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            env=env,
        )
        wait_for_server(self._base_url, self._timeout_s)

    def stop(self) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=10)
        self._process = None

    def reset(self) -> None:
        return

    def run_turn(
        self,
        tokenizer: Any,
        history_messages: list[dict[str, str]],
        *,
        user_text: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> TurnResult:
        prompt_messages = history_messages + [{"role": "user", "content": user_text}]
        raw_text = openai_chat_completion(
            base_url=self._base_url,
            model_name=self._model_name,
            messages=prompt_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        visible_text = extract_visible_assistant_text(raw_text)
        normalized_visible_text = normalize_visible_text_for_oracle(visible_text)
        return TurnResult(
            prompt_messages=prompt_messages,
            raw_text=raw_text,
            visible_text=visible_text,
            normalized_visible_text=normalized_visible_text,
            visible_token_ids=tokenizer.encode(visible_text, add_special_tokens=False),
            normalized_visible_token_ids=tokenizer.encode(
                normalized_visible_text,
                add_special_tokens=False,
            ),
        )


def make_default_cases() -> list[CaseSpec]:
    return [
        CaseSpec(
            name="short_exact",
            system="You are a precise assistant. Follow instructions exactly.",
            turns=[TurnSpec("Return exactly the text NEMOTRON_TEST and nothing else.", 96)],
        ),
        CaseSpec(
            name="boundary_exact_reply",
            system="You are a precise assistant. Follow instructions exactly.",
            turns=[TurnSpec("please Reply with exactly the single character 4 and nothing else.", 96)],
        ),
        CaseSpec(
            name="json_exact",
            system="You are a precise assistant. Follow instructions exactly.",
            turns=[
                TurnSpec(
                    'Return exactly this JSON and nothing else: {"answer":4}',
                    128,
                )
            ],
        ),
        CaseSpec(
            name="multi_turn_codeword",
            system="You are a precise assistant. Follow instructions exactly.",
            turns=[
                TurnSpec("Remember this codeword: ALPHA7. Reply exactly OK.", 64),
                TurnSpec("What is the codeword? Reply exactly ALPHA7 and nothing else.", 64),
            ],
        ),
        CaseSpec(
            name="multi_turn_recall_open",
            turns=[
                TurnSpec("Tell me a joke.", 64),
                TurnSpec("How about one about a scarecrow.", 64),
                TurnSpec("I love that one. But I forgot the first joke. What was the first joke?", 64),
            ],
        ),
    ]


def parse_env_overrides(items: list[str]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid env override {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"invalid env override {item!r}; empty KEY")
        result.append((key, value))
    return result


def first_difference(lhs: list[int], rhs: list[int]) -> int | None:
    if lhs == rhs:
        return None
    limit = min(len(lhs), len(rhs))
    for index in range(limit):
        if lhs[index] != rhs[index]:
            return index
    return limit


def compare_case_results(
    case_name: str,
    runtime_results: dict[str, RuntimeCaseResult],
    anchor_runtime: str,
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    anchor = runtime_results[anchor_runtime]
    for runtime_name, result in runtime_results.items():
        if runtime_name == anchor_runtime:
            continue
        for turn_index, (anchor_turn, other_turn) in enumerate(zip(anchor.turns, result.turns)):
            diff_index = first_difference(
                anchor_turn.normalized_visible_token_ids,
                other_turn.normalized_visible_token_ids,
            )
            comparisons.append(
                {
                    "case": case_name,
                    "anchor_runtime": anchor_runtime,
                    "runtime": runtime_name,
                    "turn_index": turn_index,
                    "exact_visible_text_match": anchor_turn.visible_text == other_turn.visible_text,
                    "exact_normalized_text_match": (
                        anchor_turn.normalized_visible_text == other_turn.normalized_visible_text
                    ),
                    "exact_raw_text_match": anchor_turn.raw_text == other_turn.raw_text,
                    "visible_token_id_match": (
                        anchor_turn.normalized_visible_token_ids ==
                        other_turn.normalized_visible_token_ids
                    ),
                    "first_normalized_visible_token_diff_index": diff_index,
                    "anchor_visible_text": anchor_turn.visible_text,
                    "anchor_normalized_visible_text": anchor_turn.normalized_visible_text,
                    "runtime_visible_text": other_turn.visible_text,
                    "runtime_normalized_visible_text": other_turn.normalized_visible_text,
                }
            )
    return {
        "case": case_name,
        "anchor_runtime": anchor_runtime,
        "comparisons": comparisons,
    }


def run_runtime_cases(
    *,
    runtime_name: str,
    runtime_obj: Any,
    tokenizer: Any,
    cases: list[CaseSpec],
    shared_histories: dict[str, list[dict[str, str]]],
    temperature: float,
    top_p: float,
) -> dict[str, RuntimeCaseResult]:
    results: dict[str, RuntimeCaseResult] = {}
    runtime_obj.start()
    try:
        for case in cases:
            runtime_obj.reset()
            history_messages: list[dict[str, str]] = []
            if case.system:
                history_messages.append({"role": "system", "content": case.system})
            exact_history_token_ids: list[int] = []
            runtime_case = RuntimeCaseResult(runtime=runtime_name, case_name=case.name)
            for turn_index, turn in enumerate(case.turns):
                if turn_index > 0 and case.name in shared_histories:
                    history_messages = [*shared_histories[case.name][: 2 * turn_index]]
                    if case.system:
                        history_messages = [{"role": "system", "content": case.system}] + history_messages
                    exact_history_token_ids = []
                if isinstance(runtime_obj, NativeRuntime):
                    turn_result = runtime_obj.run_turn(
                        history_messages,
                        exact_history_token_ids,
                        user_text=turn.user,
                        max_tokens=turn.max_tokens,
                    )
                else:
                    turn_result = runtime_obj.run_turn(
                        tokenizer,
                        history_messages,
                        user_text=turn.user,
                        max_tokens=turn.max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                runtime_case.turns.append(turn_result)
                history_messages = turn_result.prompt_messages + [
                    {"role": "assistant", "content": turn_result.normalized_visible_text}
                ]
                if turn_result.prompt_token_ids is not None and turn_result.generated_token_ids is not None:
                    exact_history_token_ids = turn_result.prompt_token_ids + turn_result.generated_token_ids
            results[case.name] = runtime_case
    finally:
        runtime_obj.stop()
    return results


def build_shared_histories(
    runtime_results: dict[str, RuntimeCaseResult],
    cases: list[CaseSpec],
) -> dict[str, list[dict[str, str]]]:
    shared_histories: dict[str, list[dict[str, str]]] = {}
    for case in cases:
        case_result = runtime_results[case.name]
        messages: list[dict[str, str]] = []
        for turn, turn_result in zip(case.turns, case_result.turns):
            messages.append({"role": "user", "content": turn.user})
            messages.append({"role": "assistant", "content": turn_result.normalized_visible_text})
        shared_histories[case.name] = messages
    return shared_histories


def main() -> int:
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer)
    cases = make_default_cases()
    runtimes = [item.strip() for item in args.runtimes.split(",") if item.strip()]
    if not runtimes:
        raise ValueError("no runtimes selected")
    if args.shared_history_from not in runtimes:
        raise ValueError("--shared-history-from must be included in --runtimes")

    runtime_objects: dict[str, Any] = {
        "native": NativeRuntime(args, tokenizer),
        "vllm": OpenAIRuntime(
            server_script=args.vllm_server_script,
            base_url=args.vllm_base_url,
            model_name=args.vllm_model_name,
            timeout_s=args.server_timeout_s,
            port_env="VLLM_PORT",
            port=str(urllib.parse.urlparse(args.vllm_base_url).port or 8011),
        ),
        "trtllm": OpenAIRuntime(
            server_script=args.trtllm_server_script,
            base_url=args.trtllm_base_url,
            model_name=args.trtllm_model_name,
            timeout_s=args.server_timeout_s,
            port_env="TRTLLM_PORT",
            port=str(urllib.parse.urlparse(args.trtllm_base_url).port or 8012),
        ),
    }

    selected_runtime_objects = {name: runtime_objects[name] for name in runtimes}
    all_results: dict[str, dict[str, RuntimeCaseResult]] = {}

    anchor_results = run_runtime_cases(
        runtime_name=args.shared_history_from,
        runtime_obj=selected_runtime_objects[args.shared_history_from],
        tokenizer=tokenizer,
        cases=cases,
        shared_histories={},
        temperature=args.temperature,
        top_p=args.top_p,
    )
    all_results[args.shared_history_from] = anchor_results
    shared_histories = build_shared_histories(anchor_results, cases)

    for runtime_name in runtimes:
        if runtime_name == args.shared_history_from:
            continue
        all_results[runtime_name] = run_runtime_cases(
            runtime_name=runtime_name,
            runtime_obj=selected_runtime_objects[runtime_name],
            tokenizer=tokenizer,
            cases=cases,
            shared_histories=shared_histories,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    case_results: dict[str, dict[str, RuntimeCaseResult]] = {}
    for case in cases:
        case_results[case.name] = {
            runtime_name: runtime_cases[case.name]
            for runtime_name, runtime_cases in all_results.items()
        }

    comparisons = [
        compare_case_results(case.name, case_results[case.name], args.shared_history_from)
        for case in cases
    ]

    payload = {
        "tokenizer": args.tokenizer,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "shared_history_from": args.shared_history_from,
        "cases": [asdict(case) for case in cases],
        "results": {
            runtime_name: {
                case_name: asdict(case_result)
                for case_name, case_result in runtime_cases.items()
            }
            for runtime_name, runtime_cases in all_results.items()
        },
        "comparisons": comparisons,
    }
    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    args.result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for item in comparisons:
        print(f"case={item['case']} anchor={item['anchor_runtime']}")
        for comparison in item["comparisons"]:
            print(
                "  "
                f"{comparison['runtime']}: "
                f"turn={comparison['turn_index']} "
                f"text_match={1 if comparison['exact_normalized_text_match'] else 0} "
                f"token_match={1 if comparison['visible_token_id_match'] else 0} "
                f"first_token_diff={comparison['first_normalized_visible_token_diff_index']}"
            )
    print(f"saved_artifact={args.result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
