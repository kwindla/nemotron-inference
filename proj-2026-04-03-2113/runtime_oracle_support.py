#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import pathlib
import subprocess
from datetime import datetime, timezone
from typing import Any


SCRIPT_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
DEFAULT_MANIFEST = (
    REPO_ROOT / "artifacts" / "manifests" / "forward_runtime_manifest_nano_rtx5090_unverified.json"
)
DEFAULT_PROMPTS_FILE = REPO_ROOT / "testing" / "oracle" / "prompts.json"
ORACLE_GLOB = "nano_*_token_oracle_*.json"
BUILD_DIR_CANDIDATES = ("build-sm120-relwithdebinfo",)


def git_output(cwd: pathlib.Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(cwd), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def parse_token_ids_text(text: str) -> list[int]:
    normalized = text
    for ch in "[],\n\r\t":
        normalized = normalized.replace(ch, " ")
    token_ids = [int(chunk) for chunk in normalized.split()]
    if not token_ids:
        raise ValueError("prompt token IDs are empty")
    if any(token_id < 0 for token_id in token_ids):
        raise ValueError("prompt token IDs must be non-negative")
    return token_ids


def find_latest_oracle() -> pathlib.Path:
    oracle_dir = REPO_ROOT / "artifacts" / "oracles"
    candidates = sorted(oracle_dir.glob(ORACLE_GLOB))
    if not candidates:
        raise FileNotFoundError(
            f"No oracle files matching {ORACLE_GLOB!r} were found in {oracle_dir}"
        )
    return candidates[-1]


def default_oracle_output_path(prompt_token_count: int) -> pathlib.Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        REPO_ROOT
        / "artifacts"
        / "oracles"
        / f"nano_{prompt_token_count}_token_oracle_{timestamp}.json"
    )


def find_runtime_oracle_binary(build_dir: str | None = None) -> pathlib.Path:
    if build_dir is not None:
        candidate = REPO_ROOT / build_dir
        if candidate.is_file():
            return candidate
        binary = candidate / "testing" / "nano_save_prompt_oracle"
        if binary.is_file():
            return binary
        raise FileNotFoundError(f"Runtime oracle binary not found under {candidate}")

    for candidate in BUILD_DIR_CANDIDATES:
        binary = REPO_ROOT / candidate / "testing" / "nano_save_prompt_oracle"
        if binary.is_file():
            return binary
    raise FileNotFoundError(
        "Failed to locate nano_save_prompt_oracle; set --build-dir explicitly"
    )


def resolve_runtime_oracle_build_dir(
    binary: pathlib.Path, build_dir: str | None = None
) -> pathlib.Path | None:
    if build_dir is not None:
        return (REPO_ROOT / build_dir).resolve() if not pathlib.Path(build_dir).is_absolute() else pathlib.Path(build_dir).resolve()

    parent = binary.resolve().parent
    if parent.name != "testing":
        return None
    return parent.parent.resolve()


def load_named_prompt_case(
    prompt_name: str, prompts_file: pathlib.Path = DEFAULT_PROMPTS_FILE
) -> dict[str, Any]:
    payload = json.loads(prompts_file.read_text(encoding="utf-8"))
    for case in payload.get("cases", []):
        if case.get("name") == prompt_name:
            return case
    raise KeyError(f"Prompt {prompt_name!r} not found in {prompts_file}")


def _load_tokenizer(model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )


def resolve_prompt_token_ids(
    *,
    model: str = DEFAULT_MODEL,
    prompt_token_ids: str | None = None,
    prompt_token_ids_file: pathlib.Path | None = None,
    prompt_name: str | None = None,
    prompt_text: str | None = None,
    prompt_file: pathlib.Path | None = None,
    prompts_file: pathlib.Path = DEFAULT_PROMPTS_FILE,
) -> tuple[list[int], str]:
    provided_sources = [
        prompt_token_ids is not None,
        prompt_token_ids_file is not None,
        prompt_name is not None,
        prompt_text is not None,
        prompt_file is not None,
    ]
    if sum(provided_sources) != 1:
        raise ValueError(
            "Provide exactly one of --prompt-token-ids, --prompt-token-ids-file, "
            "--prompt-name, --prompt-text, or --prompt-file"
        )

    if prompt_token_ids is not None:
        return parse_token_ids_text(prompt_token_ids), "prompt_token_ids"

    if prompt_token_ids_file is not None:
        text = prompt_token_ids_file.read_text(encoding="utf-8")
        return parse_token_ids_text(text), str(prompt_token_ids_file)

    tokenizer = _load_tokenizer(model)

    if prompt_name is not None:
        case = load_named_prompt_case(prompt_name, prompts_file=prompts_file)
        token_ids = tokenizer.apply_chat_template(
            case["messages"],
            tokenize=True,
            add_generation_prompt=case.get("add_generation_prompt", True),
        )
        return [int(value) for value in token_ids], f"prompt_name:{prompt_name}"

    if prompt_text is not None:
        token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        return [int(value) for value in token_ids], "prompt_text"

    assert prompt_file is not None
    text = prompt_file.read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return [int(value) for value in token_ids], str(prompt_file)


def resolve_manifest_path(manifest: pathlib.Path | None = None) -> pathlib.Path:
    if manifest is not None:
        return manifest.resolve()
    manifest_env = os.environ.get("NEMOTRON_FORWARD_MANIFEST")
    if manifest_env:
        return pathlib.Path(manifest_env).resolve()
    return DEFAULT_MANIFEST.resolve()


def run_runtime_oracle(
    *,
    prompt_token_ids: list[int],
    output_path: pathlib.Path,
    decode_token_count: int,
    build_dir: str | None = None,
    manifest: pathlib.Path | None = None,
    include_deep_regression_payload: bool = False,
) -> pathlib.Path:
    binary = find_runtime_oracle_binary(build_dir)
    resolved_build_dir = resolve_runtime_oracle_build_dir(binary, build_dir)
    manifest_path = resolve_manifest_path(manifest)
    env = os.environ.copy()
    env["NEMOTRON_FORWARD_MANIFEST"] = str(manifest_path)
    env["NEMOTRON_REPO_ROOT"] = str(REPO_ROOT)
    git_revision = git_output(REPO_ROOT, "rev-parse", "HEAD")
    if git_revision:
        env["NEMOTRON_GIT_REVISION"] = git_revision
    if resolved_build_dir is not None:
        env["NEMOTRON_ORACLE_BUILD_DIR"] = str(resolved_build_dir)

    command = [
        str(binary),
        "--prompt-token-ids",
        ",".join(str(token_id) for token_id in prompt_token_ids),
        "--decode-token-count",
        str(decode_token_count),
        "--output",
        str(output_path),
    ]
    if include_deep_regression_payload:
        command.append("--include-deep-regression-payload")
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)
    return output_path
