#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from runtime_oracle_support import (
    DEFAULT_MODEL,
    DEFAULT_PROMPTS_FILE,
    default_oracle_output_path,
    resolve_prompt_token_ids,
    run_runtime_oracle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a runtime oracle JSON for the Nano NVFP4 checkpoint using "
            "either explicit token IDs, a named chat prompt, or raw prompt text."
        )
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--decode-token-count", type=int, default=16)
    parser.add_argument(
        "--include-deep-regression-payload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Include the optional deep regression payload with full prompt-boundary "
            "logits in the oracle JSON."
        ),
    )
    parser.add_argument("--build-dir")
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompts-file", type=pathlib.Path, default=DEFAULT_PROMPTS_FILE)
    parser.add_argument("--prompt-token-ids")
    parser.add_argument("--prompt-token-ids-file", type=pathlib.Path)
    parser.add_argument("--prompt-name")
    parser.add_argument("--prompt-text")
    parser.add_argument("--prompt-file", type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.decode_token_count <= 0:
        raise ValueError("--decode-token-count must be positive")

    prompt_token_ids, prompt_source = resolve_prompt_token_ids(
        model=args.model,
        prompt_token_ids=args.prompt_token_ids,
        prompt_token_ids_file=args.prompt_token_ids_file,
        prompt_name=args.prompt_name,
        prompt_text=args.prompt_text,
        prompt_file=args.prompt_file,
        prompts_file=args.prompts_file,
    )
    output_path = args.output or default_oracle_output_path(len(prompt_token_ids))
    run_runtime_oracle(
        prompt_token_ids=prompt_token_ids,
        output_path=output_path,
        decode_token_count=args.decode_token_count,
        build_dir=args.build_dir,
        manifest=args.manifest,
        include_deep_regression_payload=args.include_deep_regression_payload,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    print(f"oracle_path={output_path}")
    print(f"prompt_source={prompt_source}")
    print(f"prompt_token_count={len(prompt_token_ids)}")
    print(f"prompt_token_ids={prompt_token_ids}")
    print(f"manifest_path={payload.get('manifest_path')}")
    print(f"build_dir={payload.get('build_dir')}")
    print(f"git_revision={payload.get('git_revision')}")
    print(
        "backend_flags="
        + json.dumps(payload.get("backend_flags", {}), sort_keys=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
