#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "anthropic>=0.40",
#   "jinja2>=3.1",
#   "transformers>=4.51",
# ]
# ///
"""
Prompt-length sweep: run chat-templated prompts of varying user-content
lengths through native, vLLM, and TRT-LLM, then judge coherence and
cross-runtime similarity using the Claude API.

Usage:

  # Native-only (no external servers needed):
  uv run tools/oracle/prompt_length_sweep.py --runtimes native

  # Full tri-runtime comparison:
  uv run tools/oracle/prompt_length_sweep.py --runtimes native,vllm,trtllm

  # With the fused MoE prefill path enabled:
  uv run tools/oracle/prompt_length_sweep.py --runtimes native \
      --native-env NEMOTRON_UNSAFE_ENABLE_NATIVE_DIRECT_MOE_PREFILL=1
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import textwrap
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
    REPO_ROOT / "artifacts" / "benchmarks" / "prompt_length_sweep_latest.json"
)

TARGET_USER_CONTENT_TOKEN_LENGTHS = [
    1, 2, 3, 4, 7, 8, 9, 16, 22, 23, 24, 33, 64, 117, 128, 261,
]

PLAUSIBLE_ENGLISH_SEED = (
    "The quick brown fox jumps over the lazy dog near the riverbank where tall "
    "oak trees provide shade during warm summer afternoons and children play "
    "games while their parents watch from wooden benches painted green with "
    "small brass plaques dedicating them to beloved community members who "
    "contributed greatly to the neighborhood park restoration project that "
    "transformed an abandoned lot into a beautiful garden filled with native "
    "wildflowers attracting butterflies and hummingbirds throughout the growing "
    "season from early spring until the first frost of autumn when leaves turn "
    "brilliant shades of orange and crimson before gently falling to the ground "
    "creating a colorful carpet that crunches underfoot as visitors stroll along "
    "winding gravel paths connecting various themed sections including a "
    "Japanese meditation garden with carefully raked sand patterns and a small "
    "koi pond surrounded by bamboo fencing and ornamental grasses swaying in "
    "the gentle breeze that carries the sweet fragrance of blooming jasmine"
)

SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and concisely."


def load_tokenizer(tokenizer_name: str):
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    try:
        return AutoTokenizer.from_pretrained(
            tokenizer_name, fix_mistral_regex=True, **kwargs
        )
    except TypeError:
        return AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)


def generate_prompts_at_lengths(
    tokenizer: Any,
    target_lengths: list[int],
) -> list[dict[str, Any]]:
    """Generate plausible English prompts whose user content tokenizes to
    exactly the requested number of tokens."""
    seed_tokens = tokenizer.encode(PLAUSIBLE_ENGLISH_SEED, add_special_tokens=False)
    prompts: list[dict[str, Any]] = []
    for target_len in target_lengths:
        if target_len <= len(seed_tokens):
            content_tokens = seed_tokens[:target_len]
        else:
            content_tokens = []
            while len(content_tokens) < target_len:
                content_tokens.extend(seed_tokens)
            content_tokens = content_tokens[:target_len]
        user_content = tokenizer.decode(content_tokens, skip_special_tokens=True)
        actual_len = len(tokenizer.encode(user_content, add_special_tokens=False))
        if actual_len != target_len:
            # Re-encode/decode can shift length by 1-2 tokens due to whitespace.
            # Binary search for the right truncation.
            for delta in range(-3, 4):
                candidate_tokens = content_tokens[: target_len + delta]
                if not candidate_tokens:
                    continue
                candidate = tokenizer.decode(candidate_tokens, skip_special_tokens=True)
                if len(tokenizer.encode(candidate, add_special_tokens=False)) == target_len:
                    user_content = candidate
                    actual_len = target_len
                    break
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        full_token_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        if hasattr(full_token_ids, "input_ids"):
            full_token_ids = full_token_ids.input_ids
        if full_token_ids and isinstance(full_token_ids[0], list):
            full_token_ids = full_token_ids[0]
        prompts.append(
            {
                "target_user_content_tokens": target_len,
                "actual_user_content_tokens": actual_len,
                "total_prompt_tokens": len(full_token_ids),
                "user_content": user_content,
                "messages": messages,
                "prompt_token_ids": [int(t) for t in full_token_ids],
            }
        )
    return prompts


# ---------------------------------------------------------------------------
# Runtime helpers (adapted from compare_chat_runtimes.py)
# ---------------------------------------------------------------------------


def extract_visible_assistant_text(raw_text: str) -> str:
    if not raw_text:
        return raw_text
    if "</think>" in raw_text:
        visible = raw_text.rsplit("</think>", maxsplit=1)[-1].strip()
        if visible:
            return visible
    return raw_text.strip()


def normalize_visible_text(text: str) -> str:
    for token in (
        "<|im_end|>",
        "<|im_start|>",
        "<\u00ef\u00bd\u009cend\u2581of\u2581sentence\u00ef\u00bd\u009c>",
    ):
        text = text.replace(token, "")
    return text.strip()


def wait_for_server(base_url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    models_url = f"{base_url}/v1/models"
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(models_url, timeout=2.0) as response:
                if response.status == 200:
                    return
        except Exception as exc:
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
) -> str:
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
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
        body = json.loads(response.read().decode("utf-8"))
    choices = body.get("choices", [])
    if not choices:
        raise RuntimeError(f"no choices in response: {body}")
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
    raise RuntimeError(f"unexpected content: {content!r}")


def parse_env_overrides(items: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid env override {item!r}")
        key, value = item.split("=", 1)
        result[key.strip()] = value
    return result


@dataclass
class PromptResult:
    runtime: str
    target_user_content_tokens: int
    total_prompt_tokens: int
    user_content: str
    raw_text: str
    visible_text: str
    normalized_visible_text: str
    generated_token_ids: list[int] | None = None
    unique_token_count: int = 0
    unique_token_ratio: float = 0.0
    error: str | None = None


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
        self._target_context_tokens = args.native_target_context_tokens
        self._extra_env = parse_env_overrides(args.native_env)

    def start(self) -> None:
        cmd = [
            str(self._binary),
            "--manifest",
            str(self._manifest),
            "--prefix-cache",
            "off",
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
            raise RuntimeError(f"native startup failed: {ready}")

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

    def run_prompt(
        self,
        prompt_token_ids: list[int],
        max_tokens: int,
    ) -> dict[str, Any]:
        self._send_reset()
        eos_ids = getattr(self._tokenizer, "eos_token_id", None)
        if eos_ids is None:
            eos_csv = ""
        elif isinstance(eos_ids, int):
            eos_csv = str(eos_ids)
        else:
            eos_csv = ",".join(str(int(t)) for t in eos_ids)
        prompt_csv = ",".join(str(t) for t in prompt_token_ids)
        return self._send_line(f"TURN\t{max_tokens}\t{eos_csv}\t{prompt_csv}")

    def _send_reset(self) -> None:
        self._send_line("RESET")

    def _send_line(self, line: str) -> dict[str, Any]:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("native helper not running")
        self._process.stdin.write(line + "\n")
        self._process.stdin.flush()
        return self._read_line()

    def _read_line(self) -> dict[str, Any]:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("native helper not running")
        response = self._process.stdout.readline()
        if response == "":
            stderr_text = ""
            if self._process.stderr is not None:
                stderr_text = self._process.stderr.read()
            raise RuntimeError(f"native helper exited: {stderr_text[:500]}")
        payload = json.loads(response)
        if not payload.get("ok", False):
            raise RuntimeError(payload.get("error", "native error"))
        return payload


# ---------------------------------------------------------------------------
# Claude evaluation
# ---------------------------------------------------------------------------


def load_anthropic_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def _strip_markdown_json(text: str) -> str:
    """Strip ```json ... ``` wrapper that some models add."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n", 1)
        if len(lines) > 1:
            stripped = lines[1]
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
    return stripped


def claude_judge_coherence(
    client: Any,
    user_content: str,
    assistant_output: str,
    runtime_name: str,
) -> dict[str, Any]:
    prompt = textwrap.dedent(f"""\
        You are evaluating the output of an LLM inference runtime.

        The user sent this message:
        <user_message>{user_content}</user_message>

        The runtime ({runtime_name}) produced this response:
        <response>{assistant_output}</response>

        Evaluate:
        1. Is the response coherent English (not repetitive garbage, not degenerate)?
        2. Is it a plausible response to the user message?

        Reply with exactly this JSON and nothing else:
        {{"coherent": true/false, "plausible": true/false, "explanation": "brief reason"}}
    """)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    text = _strip_markdown_json(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"coherent": None, "plausible": None, "explanation": f"parse error: {text[:200]}"}


def claude_judge_similarity(
    client: Any,
    user_content: str,
    outputs: dict[str, str],
    anchor_runtime: str,
) -> dict[str, Any]:
    anchor_text = outputs.get(anchor_runtime, "")
    comparisons: dict[str, Any] = {}
    for runtime_name, text in outputs.items():
        if runtime_name == anchor_runtime:
            continue
        prompt = textwrap.dedent(f"""\
            You are comparing outputs from two LLM inference runtimes.

            User message:
            <user_message>{user_content}</user_message>

            Reference output ({anchor_runtime}):
            <reference>{anchor_text[:1000]}</reference>

            Test output ({runtime_name}):
            <test>{text[:1000]}</test>

            Are these outputs semantically similar (same meaning/intent, even if
            wording differs)? Minor differences in style, emoji, or phrasing are OK.

            Reply with exactly this JSON and nothing else:
            {{"similar": true/false, "explanation": "brief reason"}}
        """)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text_resp = _strip_markdown_json(response.content[0].text.strip())
        try:
            comparisons[runtime_name] = json.loads(text_resp)
        except json.JSONDecodeError:
            comparisons[runtime_name] = {
                "similar": None,
                "explanation": f"parse error: {text_resp[:200]}",
            }
    return comparisons


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt-length sweep with LLM-judged coherence evaluation."
    )
    parser.add_argument(
        "--result-path",
        type=pathlib.Path,
        default=DEFAULT_RESULT_PATH,
    )
    parser.add_argument(
        "--tokenizer",
        default=str(DEFAULT_CHECKPOINT),
    )
    parser.add_argument(
        "--runtimes",
        default="native",
        help="Comma-separated: native,vllm,trtllm",
    )
    parser.add_argument(
        "--anchor-runtime",
        default=None,
        help="Runtime to use as reference for similarity. Defaults to vllm if available.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens to generate (large enough to not cut off thinking).",
    )
    parser.add_argument(
        "--native-build-dir",
        default=DEFAULT_BUILD_DIR,
    )
    parser.add_argument(
        "--native-manifest",
        type=pathlib.Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--native-target-context-tokens",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--native-server-binary",
        type=pathlib.Path,
        default=None,
    )
    parser.add_argument(
        "--native-env",
        action="append",
        default=[],
        help="Extra KEY=VALUE env overrides for native. Repeatable.",
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
        "--server-timeout-s",
        type=float,
        default=900.0,
    )
    parser.add_argument(
        "--skip-claude-eval",
        action="store_true",
        help="Skip Claude API coherence/similarity evaluation.",
    )
    parser.add_argument(
        "--lengths",
        default=None,
        help="Comma-separated custom lengths (overrides default sweep).",
    )
    return parser.parse_args()


def run_native_sweep(
    args: argparse.Namespace,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    max_tokens: int,
) -> list[PromptResult]:
    native = NativeRuntime(args, tokenizer)
    results: list[PromptResult] = []
    native.start()
    try:
        for prompt in prompts:
            target_len = prompt["target_user_content_tokens"]
            print(f"  native: user_content_tokens={target_len} ...", end="", flush=True)
            try:
                payload = native.run_prompt(prompt["prompt_token_ids"], max_tokens)
                gen_ids = [int(t) for t in payload["generated_token_ids"]]
                raw_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
                visible = extract_visible_assistant_text(raw_text)
                normalized = normalize_visible_text(visible)
                unique = len(set(gen_ids))
                ratio = unique / len(gen_ids) if gen_ids else 0.0
                result = PromptResult(
                    runtime="native",
                    target_user_content_tokens=target_len,
                    total_prompt_tokens=prompt["total_prompt_tokens"],
                    user_content=prompt["user_content"],
                    raw_text=raw_text,
                    visible_text=visible,
                    normalized_visible_text=normalized,
                    generated_token_ids=gen_ids,
                    unique_token_count=unique,
                    unique_token_ratio=ratio,
                )
                status = "OK" if ratio > 0.1 else "LOW-DIVERSITY"
                print(f" {status} unique={unique}/{len(gen_ids)}")
            except Exception as exc:
                print(f" ERROR: {exc}")
                result = PromptResult(
                    runtime="native",
                    target_user_content_tokens=target_len,
                    total_prompt_tokens=prompt["total_prompt_tokens"],
                    user_content=prompt["user_content"],
                    raw_text="",
                    visible_text="",
                    normalized_visible_text="",
                    error=str(exc),
                )
            results.append(result)
    finally:
        native.stop()
    return results


def run_openai_sweep(
    *,
    runtime_name: str,
    base_url: str,
    model_name: str,
    server_script: pathlib.Path,
    timeout_s: float,
    port_env: str,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    max_tokens: int,
) -> list[PromptResult]:
    port = str(urllib.parse.urlparse(base_url).port or 8011)
    env = dict(os.environ)
    env[port_env] = port
    process = subprocess.Popen(
        [str(server_script.resolve())],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        env=env,
    )
    results: list[PromptResult] = []
    try:
        wait_for_server(base_url, timeout_s)
        for prompt in prompts:
            target_len = prompt["target_user_content_tokens"]
            print(f"  {runtime_name}: user_content_tokens={target_len} ...", end="", flush=True)
            try:
                raw_text = openai_chat_completion(
                    base_url=base_url,
                    model_name=model_name,
                    messages=prompt["messages"],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                visible = extract_visible_assistant_text(raw_text)
                normalized = normalize_visible_text(visible)
                gen_ids = tokenizer.encode(raw_text, add_special_tokens=False)
                unique = len(set(gen_ids))
                ratio = unique / len(gen_ids) if gen_ids else 0.0
                result = PromptResult(
                    runtime=runtime_name,
                    target_user_content_tokens=target_len,
                    total_prompt_tokens=prompt["total_prompt_tokens"],
                    user_content=prompt["user_content"],
                    raw_text=raw_text,
                    visible_text=visible,
                    normalized_visible_text=normalized,
                    generated_token_ids=gen_ids,
                    unique_token_count=unique,
                    unique_token_ratio=ratio,
                )
                print(f" OK unique={unique}/{len(gen_ids)}")
            except Exception as exc:
                print(f" ERROR: {exc}")
                result = PromptResult(
                    runtime=runtime_name,
                    target_user_content_tokens=target_len,
                    total_prompt_tokens=prompt["total_prompt_tokens"],
                    user_content=prompt["user_content"],
                    raw_text="",
                    visible_text="",
                    normalized_visible_text="",
                    error=str(exc),
                )
            results.append(result)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    return results


def main() -> int:
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer)
    runtimes = [r.strip() for r in args.runtimes.split(",") if r.strip()]

    if args.lengths:
        lengths = [int(x.strip()) for x in args.lengths.split(",")]
    else:
        lengths = TARGET_USER_CONTENT_TOKEN_LENGTHS

    print(f"Generating prompts for {len(lengths)} target lengths: {lengths}")
    prompts = generate_prompts_at_lengths(tokenizer, lengths)
    for p in prompts:
        print(
            f"  target={p['target_user_content_tokens']:>3d}  "
            f"actual={p['actual_user_content_tokens']:>3d}  "
            f"total={p['total_prompt_tokens']:>3d}  "
            f"content={p['user_content'][:60]!r}..."
        )

    all_results: dict[str, list[PromptResult]] = {}

    for runtime_name in runtimes:
        print(f"\nRunning {runtime_name}...")
        if runtime_name == "native":
            all_results["native"] = run_native_sweep(
                args, tokenizer, prompts, args.max_tokens
            )
        elif runtime_name == "vllm":
            all_results["vllm"] = run_openai_sweep(
                runtime_name="vllm",
                base_url=args.vllm_base_url,
                model_name=args.vllm_model_name,
                server_script=args.vllm_server_script,
                timeout_s=args.server_timeout_s,
                port_env="VLLM_PORT",
                tokenizer=tokenizer,
                prompts=prompts,
                max_tokens=args.max_tokens,
            )
        elif runtime_name == "trtllm":
            all_results["trtllm"] = run_openai_sweep(
                runtime_name="trtllm",
                base_url=args.trtllm_base_url,
                model_name=args.trtllm_model_name,
                server_script=args.trtllm_server_script,
                timeout_s=args.server_timeout_s,
                port_env="TRTLLM_PORT",
                tokenizer=tokenizer,
                prompts=prompts,
                max_tokens=args.max_tokens,
            )

    # Claude evaluation
    claude_evaluations: list[dict[str, Any]] = []
    api_key = load_anthropic_key()
    if not args.skip_claude_eval and api_key:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        anchor = args.anchor_runtime or ("vllm" if "vllm" in all_results else runtimes[0])

        print("\nRunning Claude coherence evaluation...")
        for runtime_name, results in all_results.items():
            for result in results:
                if result.error or not result.normalized_visible_text:
                    continue
                print(
                    f"  judging {runtime_name} len={result.target_user_content_tokens}...",
                    end="",
                    flush=True,
                )
                coherence = claude_judge_coherence(
                    client,
                    result.user_content,
                    result.normalized_visible_text[:1000],
                    runtime_name,
                )
                status = "COHERENT" if coherence.get("coherent") else "INCOHERENT"
                print(f" {status}")
                claude_evaluations.append(
                    {
                        "type": "coherence",
                        "runtime": runtime_name,
                        "target_user_content_tokens": result.target_user_content_tokens,
                        **coherence,
                    }
                )

        if len(all_results) > 1 and anchor in all_results:
            print("\nRunning Claude similarity evaluation...")
            for i, prompt in enumerate(prompts):
                target_len = prompt["target_user_content_tokens"]
                outputs: dict[str, str] = {}
                for runtime_name, results in all_results.items():
                    if i < len(results) and not results[i].error:
                        outputs[runtime_name] = results[i].normalized_visible_text
                if len(outputs) > 1 and anchor in outputs:
                    print(f"  comparing len={target_len}...", end="", flush=True)
                    similarity = claude_judge_similarity(
                        client,
                        prompt["user_content"],
                        outputs,
                        anchor,
                    )
                    for rt, sim in similarity.items():
                        status = "SIMILAR" if sim.get("similar") else "DIVERGENT"
                        print(f" {rt}={status}", end="")
                    print()
                    claude_evaluations.append(
                        {
                            "type": "similarity",
                            "target_user_content_tokens": target_len,
                            "anchor": anchor,
                            "comparisons": similarity,
                        }
                    )
    elif not api_key:
        print("\nSkipping Claude evaluation (no ANTHROPIC_API_KEY)")

    # Save artifact
    artifact = {
        "lengths": lengths,
        "runtimes": runtimes,
        "prompts": prompts,
        "results": {
            runtime_name: [asdict(r) for r in results]
            for runtime_name, results in all_results.items()
        },
        "claude_evaluations": claude_evaluations,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    args.result_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {args.result_path}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"{'len':>4s}  {'total':>5s}  "
        + "  ".join(f"{rt:>12s}" for rt in runtimes)
    )
    print("-" * (12 + 14 * len(runtimes)))

    failures = 0
    for i, prompt in enumerate(prompts):
        target_len = prompt["target_user_content_tokens"]
        total = prompt["total_prompt_tokens"]
        cols = []
        for runtime_name in runtimes:
            if runtime_name in all_results and i < len(all_results[runtime_name]):
                r = all_results[runtime_name][i]
                if r.error:
                    cols.append("ERROR")
                    failures += 1
                elif r.unique_token_ratio < 0.1:
                    cols.append(f"DEGENERATE")
                    failures += 1
                else:
                    cols.append(f"{r.unique_token_count}/{len(r.generated_token_ids or [])}")
            else:
                cols.append("--")
        print(
            f"{target_len:>4d}  {total:>5d}  "
            + "  ".join(f"{c:>12s}" for c in cols)
        )

    # Claude summary
    coherence_fails = [
        e for e in claude_evaluations
        if e.get("type") == "coherence" and not e.get("coherent")
    ]
    similarity_fails = [
        e for e in claude_evaluations
        if e.get("type") == "similarity"
        and any(
            not v.get("similar")
            for v in e.get("comparisons", {}).values()
        )
    ]
    if coherence_fails:
        print(f"\nClaude flagged {len(coherence_fails)} INCOHERENT outputs:")
        for e in coherence_fails:
            print(f"  {e['runtime']} len={e['target_user_content_tokens']}: {e.get('explanation', '')}")
        failures += len(coherence_fails)
    if similarity_fails:
        print(f"\nClaude flagged {len(similarity_fails)} DIVERGENT outputs:")
        for e in similarity_fails:
            for rt, sim in e.get("comparisons", {}).items():
                if not sim.get("similar"):
                    print(f"  {rt} len={e['target_user_content_tokens']}: {sim.get('explanation', '')}")
        failures += len(similarity_fails)

    if failures == 0:
        print("\nALL CHECKS PASSED")
    else:
        print(f"\n{failures} FAILURE(S)")

    return 1 if failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
