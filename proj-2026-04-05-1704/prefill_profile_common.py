from __future__ import annotations

import pathlib
import random
from typing import Any


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "artifacts" / "checkpoints" / "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"


def load_tokenizer(tokenizer_name: str):
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name, fix_mistral_regex=True, **kwargs)
    except TypeError:
        return AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)


def build_prompt_token_ids(tokenizer: Any, token_count: int, seed: int) -> list[int]:
    special_ids = {
        token_id
        for token_id in [
            getattr(tokenizer, "bos_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "pad_token_id", None),
        ]
        if isinstance(token_id, int) and token_id >= 0
    }
    rng = random.Random(seed + token_count)
    vocab_size = int(tokenizer.vocab_size)
    candidates = [token_id for token_id in range(32, vocab_size) if token_id not in special_ids]
    if not candidates:
        raise RuntimeError("no usable tokenizer ids available for prompt construction")
    return [candidates[rng.randrange(len(candidates))] for _ in range(token_count)]
