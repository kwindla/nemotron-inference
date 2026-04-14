#!/usr/bin/env python3
"""Strip preprocessor gates from a CUDA source file.

Handles:
  - NEMOTRON_P5_PARTITION_DEBUG / NEMOTRON_P5_LINEAR_PARTITION_DEBUG:
    delete the entire block including body (never-defined dead code).
  - NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE: keep the #if branch, drop the #else
    branch (if present), drop the scaffolding (#if/#else/#endif lines).
  - __CUDA_ARCH__: keep the #if branch (SM120 always satisfies the
    numeric comparisons used in this repo), drop fallbacks.

Nesting is handled via a depth-counting parser; inner gates are processed
recursively so that e.g. a LOCAL_CUTE block containing P5_PARTITION_DEBUG
blocks inlines the LOCAL_CUTE body while the inner debug blocks get
deleted.

Assumes #elif is not used (verified for this repo's .cu files before the
script ran). Errors out if it encounters #elif.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

IF_RE = re.compile(r"^\s*#\s*if\b|^\s*#\s*ifdef\b|^\s*#\s*ifndef\b")
ENDIF_RE = re.compile(r"^\s*#\s*endif\b")
ELSE_RE = re.compile(r"^\s*#\s*else\b")
ELIF_RE = re.compile(r"^\s*#\s*elif\b")
IF_MACRO_RE = re.compile(r"^\s*#\s*if\s+(.*)$")

DROP_ENTIRE_MACROS = (
    "NEMOTRON_P5_PARTITION_DEBUG",
    "NEMOTRON_P5_LINEAR_PARTITION_DEBUG",
)
KEEP_IF_BRANCH_MACROS = (
    "NEMOTRON_RUNTIME_HAVE_LOCAL_CUTE",
    "__CUDA_ARCH__",
)


def find_matching_endif(lines: list[str], start: int, end: int) -> int:
    """Given start pointing at an #if line, return the index of its matching #endif in [start+1, end)."""
    depth = 1
    for i in range(start + 1, end):
        line = lines[i]
        if ELIF_RE.match(line):
            raise ValueError(
                f"#elif at line {i + 1} is not supported; plan assumed no #elif in these files."
            )
        if IF_RE.match(line):
            depth += 1
        elif ENDIF_RE.match(line):
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"No matching #endif for #if at line {start + 1}")


def find_top_level_else(lines: list[str], start: int, endif_i: int) -> int | None:
    """Find a #else at depth 1 between start+1 and endif_i. Returns None if none exists."""
    depth = 1
    for i in range(start + 1, endif_i):
        line = lines[i]
        if IF_RE.match(line):
            depth += 1
        elif ENDIF_RE.match(line):
            depth -= 1
        elif ELSE_RE.match(line) and depth == 1:
            return i
    return None


def classify(cond: str) -> str:
    """Classify a #if condition string. Returns 'drop', 'keep_if', or 'untouched'."""
    for macro in DROP_ENTIRE_MACROS:
        if macro in cond:
            return "drop"
    for macro in KEEP_IF_BRANCH_MACROS:
        if macro in cond:
            return "keep_if"
    return "untouched"


def process(lines: list[str], start: int, end: int) -> list[str]:
    """Process lines[start:end] and return the transformed output."""
    result: list[str] = []
    i = start
    while i < end:
        line = lines[i]
        m_if = IF_MACRO_RE.match(line)
        if m_if:
            cond = m_if.group(1).strip()
            action = classify(cond)
            if action == "untouched":
                # Leave intact (should not happen for this plan; assert for safety).
                raise ValueError(
                    f"Unrecognized #if gate at line {i + 1}: {line.rstrip()}. "
                    f"This script only handles {DROP_ENTIRE_MACROS} and {KEEP_IF_BRANCH_MACROS}."
                )
            endif_i = find_matching_endif(lines, i, end)
            else_i = find_top_level_else(lines, i, endif_i)
            if action == "drop":
                # Drop entire block including body.
                i = endif_i + 1
                continue
            else:
                # keep_if: keep the #if branch body, drop the #else branch (if any),
                # drop the scaffolding lines.
                inner_end = else_i if else_i is not None else endif_i
                inner = process(lines, i + 1, inner_end)
                result.extend(inner)
                i = endif_i + 1
                continue
        # Non-gate line: copy through (but also skip stray #endif/#else/#ifdef/#ifndef
        # that aren't covered by the if-branch logic above — in this repo they shouldn't
        # appear at the top level because all gates start with #if).
        if IF_RE.match(line):
            # #ifdef / #ifndef outside our scope — we don't expect any in the target .cu files.
            raise ValueError(
                f"Unhandled #ifdef/#ifndef at line {i + 1}: {line.rstrip()}"
            )
        result.append(line)
        i += 1
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="file to process in place")
    ap.add_argument("--dry-run", action="store_true", help="print summary without writing")
    args = ap.parse_args()

    raw = args.path.read_text()
    lines = raw.splitlines(keepends=True)

    before_count = sum(
        1 for ln in lines if IF_MACRO_RE.match(ln) and classify(IF_MACRO_RE.match(ln).group(1).strip()) != "untouched"
    )

    try:
        out = process(lines, 0, len(lines))
    except ValueError as e:
        print(f"ERROR in {args.path}: {e}", file=sys.stderr)
        return 2

    after_raw = "".join(out)
    after_lines = after_raw.splitlines(keepends=True)
    after_count = sum(
        1
        for ln in after_lines
        if IF_MACRO_RE.match(ln) and classify(IF_MACRO_RE.match(ln).group(1).strip()) != "untouched"
    )

    delta_lines = len(lines) - len(out)
    print(
        f"{args.path}: in={len(lines)} out={len(out)} removed={delta_lines} "
        f"gates_before={before_count} gates_after={after_count}"
    )

    if args.dry_run:
        return 0
    args.path.write_text(after_raw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
