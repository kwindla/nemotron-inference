#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Map GB10 Mamba dt hotspot steps back to decoded oracle trace tokens."
    )
    parser.add_argument(
        "--dt-hotspots",
        default="",
        help="Optional path to a gb10_mamba_trace_dt_hotspots artifact. Defaults to the latest one.",
    )
    parser.add_argument(
        "--oracle-dir",
        default=str(root / "testing" / "oracle"),
    )
    parser.add_argument(
        "--fixture-name",
        default="mamba_layer0_target_chat_trace_markdownish",
    )
    parser.add_argument("--json-output")
    parser.add_argument("--markdown-output")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def latest_dt_hotspot_artifact(root: Path) -> Path:
    matches = sorted((root / "artifacts" / "benchmarks").glob("gb10_mamba_trace_dt_hotspots_*.json"))
    if not matches:
        raise FileNotFoundError("no gb10_mamba_trace_dt_hotspots artifacts found")
    return matches[-1]


def escape_token(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# GB10 Mamba Token Hotspot Mapping")
    lines.append("")
    lines.append(f"Fixture: `{report['fixture_name']}`")
    lines.append("")
    lines.append(f"Source hotspot artifact: `{report['dt_hotspots_artifact']}`")
    lines.append("")
    lines.append("## Repeated Hotspot Tokens")
    lines.append("")
    lines.append("| Token Text | Count | Step Examples |")
    lines.append("|---|---:|---|")
    for item in report["repeated_tokens"]:
        lines.append(
            f"| `{item['token_text']}` | `{item['count']}` | `{item['step_examples']}` |"
        )
    lines.append("")
    lines.append("## Hotspot Steps")
    lines.append("")
    lines.append(
        "| Step | Token ID | Token Text | Prev | Next | Phase Progress | `dt` Mean | `dt` P99 | `hidden` RMS |"
    )
    lines.append("|---:|---:|---|---|---|---:|---:|---:|---:|")
    for item in report["hotspot_steps"]:
        lines.append(
            f"| `{item['step']}` | `{item['token_id']}` | `{item['token_text']}` | "
            f"`{item['prev_token_text']}` | `{item['next_token_text']}` | "
            f"`{item['phase_progress']}` | `{item['dt_abs_mean']}` | `{item['dt_abs_p99']}` | `{item['hidden_rms']}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    args = parse_args()
    dt_hotspots_path = Path(args.dt_hotspots) if args.dt_hotspots else latest_dt_hotspot_artifact(root)
    dt_hotspots = load_json(dt_hotspots_path)
    fixture_name = args.fixture_name

    hotspot_rows = dt_hotspots.get("target_top_steps", dt_hotspots.get("markdownish_top_steps", []))
    fixture_root = Path(args.oracle_dir) / fixture_name
    trace_tokens = load_json(fixture_root / "trace_tokens.json")["tokens"]

    hotspot_steps: list[dict[str, Any]] = []
    token_counter: Counter[str] = Counter()
    token_examples: dict[str, list[int]] = {}
    for row in hotspot_rows:
        step = int(row["phase_step_index"])
        token = trace_tokens[step]
        prev_text = escape_token(trace_tokens[step - 1]["text"]) if step > 0 else "<BOF>"
        next_text = escape_token(trace_tokens[step + 1]["text"]) if step + 1 < len(trace_tokens) else "<EOF>"
        token_text = escape_token(str(token["text"]))
        hotspot_steps.append(
            {
                "step": step,
                "token_id": int(token["token_id"]),
                "token_text": token_text,
                "prev_token_text": prev_text,
                "next_token_text": next_text,
                "phase_progress": row["phase_progress"],
                "dt_abs_mean": row["dt_abs_mean"],
                "dt_abs_p99": row["dt_abs_p99"],
                "hidden_rms": row["hidden_rms"],
            }
        )
        token_counter[token_text] += 1
        token_examples.setdefault(token_text, []).append(step)

    repeated_tokens = [
        {
            "token_text": token_text,
            "count": count,
            "step_examples": token_examples[token_text][:4],
        }
        for token_text, count in token_counter.most_common()
    ]

    report = {
        "fixture_name": fixture_name,
        "dt_hotspots_artifact": dt_hotspots_path.name,
        "repeated_tokens": repeated_tokens,
        "hotspot_steps": hotspot_steps,
    }

    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        out = Path(args.markdown_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps({"hotspot_step_count": len(hotspot_steps), "top_repeated_token": repeated_tokens[0]["token_text"] if repeated_tokens else ""}, indent=2))


if __name__ == "__main__":
    main()
