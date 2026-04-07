#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Create a phase-only Mamba trace fixture variant with scaled dt values."
    )
    parser.add_argument(
        "--source-root",
        default=str(root / "testing" / "oracle" / "mamba_layer0_target_chat_trace_markdown_headerless"),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--phase-name", default="user_turn_1_tail_prefill")
    parser.add_argument("--dt-scale", type=float, default=1.0)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)

    metadata = load_json(source_root / "metadata.json")
    phase = next((item for item in metadata["trace_phases"] if item["name"] == args.phase_name), None)
    if phase is None:
      raise ValueError(f"phase {args.phase_name!r} not found in {source_root}")
    if int(phase["start_step"]) != 0:
      raise ValueError("phase-only fixture generation currently supports only phases that start at step 0")

    trace_step_count = int(metadata["trace_step_count"])
    batch_size = int(metadata["batch_size"])
    num_heads = int(metadata["num_heads"])
    head_dim = int(metadata["head_dim"])
    state_size = int(metadata["state_size"])
    phase_length = int(phase["length"])
    hidden_count = batch_size * num_heads * head_dim
    bc_count = batch_size * num_heads * state_size

    output_root.mkdir(parents=True, exist_ok=True)

    def copy_binary(name: str) -> None:
        shutil.copy2(source_root / name, output_root / name)

    copy_binary("A_fp32.bin")
    copy_binary("D_fp32.bin")
    copy_binary("initial_state_fp32.bin")

    hidden = np.fromfile(source_root / "hidden_trace_fp32.bin", dtype=np.float32).reshape(trace_step_count, hidden_count)
    dt = np.fromfile(source_root / "dt_trace_fp32.bin", dtype=np.float32).reshape(trace_step_count, hidden_count)
    b_tensor = np.fromfile(source_root / "B_trace_fp32.bin", dtype=np.float32).reshape(trace_step_count, bc_count)
    c_tensor = np.fromfile(source_root / "C_trace_fp32.bin", dtype=np.float32).reshape(trace_step_count, bc_count)

    phase_slice = slice(0, phase_length)
    hidden[phase_slice].astype(np.float32, copy=False).tofile(output_root / "hidden_trace_fp32.bin")
    (dt[phase_slice] * np.float32(args.dt_scale)).astype(np.float32, copy=False).tofile(output_root / "dt_trace_fp32.bin")
    b_tensor[phase_slice].astype(np.float32, copy=False).tofile(output_root / "B_trace_fp32.bin")
    c_tensor[phase_slice].astype(np.float32, copy=False).tofile(output_root / "C_trace_fp32.bin")

    trace_tokens = load_json(source_root / "trace_tokens.json")
    trace_tokens["trace_step_count"] = phase_length
    trace_tokens["token_ids"] = list(trace_tokens.get("token_ids", [])[:phase_length])
    trace_tokens["token_text"] = list(trace_tokens.get("token_text", [])[:phase_length])
    trace_tokens["source_fixture_name"] = source_root.name
    trace_tokens["source_phase_name"] = args.phase_name
    trace_tokens["dt_scale"] = args.dt_scale
    (output_root / "trace_tokens.json").write_text(json.dumps(trace_tokens, indent=2) + "\n", encoding="utf-8")

    phase_metadata = dict(phase)
    phase_metadata["length"] = phase_length
    phase_metadata["start_step"] = 0
    phase_metadata["end_step"] = phase_length

    metadata["trace_step_count"] = phase_length
    metadata["trace_phases"] = [phase_metadata]
    metadata["fixture_kind"] = "mamba_target_chat_trace_phase_dt_variant_v1"
    metadata["source_fixture_name"] = source_root.name
    metadata["source_phase_name"] = args.phase_name
    metadata["dt_scale"] = args.dt_scale
    metadata["notes"] = list(metadata.get("notes", [])) + [
        "This fixture is a generated phase-only variant for dt-scale sensitivity analysis.",
        f"The trace is truncated to phase {args.phase_name!r} and dt_trace_fp32 is scaled by {args.dt_scale}.",
    ]
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
