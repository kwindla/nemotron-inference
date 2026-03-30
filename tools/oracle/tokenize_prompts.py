#!/usr/bin/env python3

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

from transformers import AutoTokenizer


def shared_prefix_length(a: list[int], b: list[int]) -> int:
    count = 0
    for lhs, rhs in zip(a, b):
        if lhs != rhs:
            break
        count += 1
    return count


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tokenizer oracle vectors for fixed chat prompts.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))

    cases = []
    ids_by_name: dict[str, list[int]] = {}
    for prompt in prompts["cases"]:
        rendered = tokenizer.apply_chat_template(
            prompt["messages"],
            tokenize=False,
            add_generation_prompt=prompt.get("add_generation_prompt", True),
        )
        token_ids = tokenizer.apply_chat_template(
            prompt["messages"],
            tokenize=True,
            add_generation_prompt=prompt.get("add_generation_prompt", True),
        )
        ids_by_name[prompt["name"]] = token_ids
        cases.append(
            {
                "name": prompt["name"],
                "token_count": len(token_ids),
                "rendered_sha256": sha256_text(rendered),
                "rendered_preview": rendered[:240],
                "first_64_token_ids": token_ids[:64],
            }
        )

    shared_prefixes = []
    for left, right in combinations(cases, 2):
        left_ids = ids_by_name[left["name"]]
        right_ids = ids_by_name[right["name"]]
        shared_prefixes.append(
            {
                "left": left["name"],
                "right": right["name"],
                "shared_prefix_tokens": shared_prefix_length(left_ids, right_ids),
            }
        )

    report = {
        "tokenizer_class": tokenizer.__class__.__name__,
        "chat_template_present": bool(getattr(tokenizer, "chat_template", None)),
        "cases": cases,
        "shared_prefixes": shared_prefixes,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
