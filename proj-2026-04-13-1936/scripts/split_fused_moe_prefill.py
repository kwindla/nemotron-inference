#!/usr/bin/env python3
"""Split fused_moe_prefill.cu into an umbrella + private .cuh fragments.

Pure text-move: concatenating the fragments in #include order inside the
umbrella scaffolding reproduces the original file byte-for-byte. No
header comments are added to fragments. No extra blank lines are
introduced. The reconstruction invariant is:

    git show HEAD:runtime/src/backend/fused_moe_prefill.cu ==
    (umbrella with every #include "fused_moe_prefill/<name>" inlined)

This is checked by proj-2026-04-13-1936/scripts/verify_split.py after
the split runs.

Fragment layout (1-based inclusive line ranges in the post-Step-0 source):
  - common_helpers.cuh                     55-102
  - nvfp4_cute.cuh                        103-116
  - nvfp4_bridge.cuh                      117-1911
  - trt_helpers_pre_nano_epilogue.cuh    1912-5013
  - nano_p1_epilogue.cuh                 5014-5243
  - trt_helpers_post_nano_epilogue.cuh   5244-11783
  - nano_p1_kernel.cuh                  11784-12416
  - public_exports.cuh                  12418-13234

Umbrella scaffolding reused verbatim from the source:
  lines 1-54   (includes, namespace nemotron {, fwd decl, namespace { open)
  line 12417   (}  // namespace — anon ns close)
  line 13235   (}  // namespace nemotron — outer ns close)
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILE = REPO_ROOT / "runtime/src/backend/fused_moe_prefill.cu"
FRAGMENTS_DIR = REPO_ROOT / "runtime/src/backend/fused_moe_prefill"

FRAGMENTS_INSIDE_ANON = [
    ("common_helpers.cuh", 55, 102),
    ("nvfp4_cute.cuh", 103, 116),
    ("nvfp4_bridge.cuh", 117, 1911),
    ("trt_helpers_pre_nano_epilogue.cuh", 1912, 5013),
    ("nano_p1_epilogue.cuh", 5014, 5243),
    ("trt_helpers_post_nano_epilogue.cuh", 5244, 11783),
    ("nano_p1_kernel.cuh", 11784, 12416),
]
FRAGMENT_OUTSIDE_ANON = ("public_exports.cuh", 12418, 13234)
ALL_FRAGMENTS = FRAGMENTS_INSIDE_ANON + [FRAGMENT_OUTSIDE_ANON]

UMBRELLA_PROLOGUE_END = 54       # lines 1..54 verbatim
ANON_CLOSE_LINE = 12417          # }  // namespace
OUTER_CLOSE_LINE = 13235         # }  // namespace nemotron


def main() -> int:
    lines = SOURCE_FILE.read_text().splitlines(keepends=True)
    total = len(lines)
    if total < OUTER_CLOSE_LINE:
        print(f"ERROR: source has {total} lines, expected at least {OUTER_CLOSE_LINE}", file=sys.stderr)
        return 2

    anchors = {
        44: "namespace nemotron {",
        54: "namespace {",
        ANON_CLOSE_LINE: "}  // namespace",
        OUTER_CLOSE_LINE: "}  // namespace nemotron",
    }
    for ln, expected in anchors.items():
        actual = lines[ln - 1].rstrip("\n").strip()
        if actual != expected:
            print(f"ERROR: anchor line {ln} mismatch: expected '{expected}', got '{actual}'", file=sys.stderr)
            return 2

    # Verify contiguity of fragment ranges (with the anon-ns close between
    # the last inside-anon fragment and the outside-anon one).
    expected = UMBRELLA_PROLOGUE_END + 1
    for name, start, end in FRAGMENTS_INSIDE_ANON:
        if start != expected:
            print(f"ERROR: {name} starts at {start}, expected {expected}", file=sys.stderr)
            return 2
        expected = end + 1
    if expected != ANON_CLOSE_LINE:
        print(
            f"ERROR: last inside-anon fragment ends at {expected - 1}, "
            f"expected {ANON_CLOSE_LINE - 1} (line before anon ns close)",
            file=sys.stderr,
        )
        return 2
    _, post_start, post_end = FRAGMENT_OUTSIDE_ANON
    if post_start != ANON_CLOSE_LINE + 1:
        print(
            f"ERROR: public_exports starts at {post_start}, expected {ANON_CLOSE_LINE + 1}",
            file=sys.stderr,
        )
        return 2
    if post_end != OUTER_CLOSE_LINE - 1:
        print(
            f"ERROR: public_exports ends at {post_end}, expected {OUTER_CLOSE_LINE - 1}",
            file=sys.stderr,
        )
        return 2

    if FRAGMENTS_DIR.exists() and any(FRAGMENTS_DIR.iterdir()):
        print(f"ERROR: {FRAGMENTS_DIR} already exists and is non-empty; refusing to overwrite", file=sys.stderr)
        return 2
    FRAGMENTS_DIR.mkdir(exist_ok=True)

    for name, start, end in ALL_FRAGMENTS:
        fragment = "".join(lines[start - 1 : end])
        (FRAGMENTS_DIR / name).write_text(fragment)
        print(f"  wrote {FRAGMENTS_DIR.relative_to(REPO_ROOT) / name} ({end - start + 1} lines)")

    # Build umbrella: lines 1..54 verbatim, then the #includes and ns closes.
    umbrella_parts: list[str] = ["".join(lines[:UMBRELLA_PROLOGUE_END])]
    for name, _, _ in FRAGMENTS_INSIDE_ANON:
        umbrella_parts.append(f'#include "fused_moe_prefill/{name}"\n')
    umbrella_parts.append("}  // namespace\n")
    umbrella_parts.append(f'#include "fused_moe_prefill/{FRAGMENT_OUTSIDE_ANON[0]}"\n')
    umbrella_parts.append("}  // namespace nemotron\n")

    umbrella = "".join(umbrella_parts)
    SOURCE_FILE.write_text(umbrella)
    print(f"  wrote {SOURCE_FILE.relative_to(REPO_ROOT)} ({len(umbrella.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
