#!/usr/bin/env python3
"""Capture vLLM Mamba2 chunked-scan intermediates for layer 0.

Monkey-patches _mamba_chunk_scan_combined_fwd to dump the 5-stage
intermediate buffers for the first Mamba layer call during prefill.
"""

from __future__ import annotations

import argparse
import json
import os
import types
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

import torch
import vllm
from einops import rearrange
from vllm import LLM, SamplingParams
from vllm.model_executor.layers.mamba.ops.ssd_bmm import _bmm_chunk_fwd
from vllm.model_executor.layers.mamba.ops.ssd_chunk_scan import _chunk_scan_fwd
from vllm.model_executor.layers.mamba.ops.ssd_chunk_state import (
    _chunk_cumsum_fwd,
    _chunk_state_fwd,
)
from vllm.model_executor.layers.mamba.ops.ssd_state_passing import _state_passing_fwd


def configure_torch_precision() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump vLLM Mamba2 chunked-scan intermediates for parity comparison."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompts-fixture", required=True)
    parser.add_argument("--target-layer", type=int, default=0,
                        help="Which Mamba layer call to capture (0 = first)")
    parser.add_argument("--image-tag", default="unknown")
    return parser.parse_args()


def write_tensor(path: Path, tensor: torch.Tensor) -> None:
    data = tensor.detach().contiguous().cpu().float().numpy().tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(f"dump: {path} ({tensor.numel()} -> fp32, shape={list(tensor.shape)})")


def load_prompt_token_ids(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(t, int) for t in payload):
        raise TypeError(f"{path} must contain a JSON array of integer token IDs")
    return payload


class ChunkedScanCapture:
    """Intercepts calls to _mamba_chunk_scan_combined_fwd and dumps intermediates."""

    def __init__(self, output_dir: Path, target_call: int):
        self.output_dir = output_dir
        self.target_call = target_call
        self.call_count = 0
        self.captured = False

    def patched_fwd(
        self,
        original_fwd,
        x, dt, A, B, C, chunk_size, out,
        D=None, z=None, dt_bias=None,
        initial_states=None, return_intermediate_states=False,
        seq_idx=None, cu_seqlens=None, cu_chunk_seqlens=None,
        last_chunk_indices=None, dt_softplus=False,
        dt_limit=(0.0, float("inf")), state_dtype=None,
    ):
        current_call = self.call_count
        self.call_count += 1

        if current_call != self.target_call or self.captured:
            # Not the target layer — run normally
            return original_fwd(
                x, dt, A, B, C, chunk_size, out,
                D=D, z=z, dt_bias=dt_bias,
                initial_states=initial_states,
                return_intermediate_states=return_intermediate_states,
                seq_idx=seq_idx, cu_seqlens=cu_seqlens,
                cu_chunk_seqlens=cu_chunk_seqlens,
                last_chunk_indices=last_chunk_indices,
                dt_softplus=dt_softplus, dt_limit=dt_limit,
                state_dtype=state_dtype,
            )

        self.captured = True
        dr = self.output_dir
        print(f"\n=== Capturing chunked-scan intermediates for call {current_call} ===")
        print(f"x.shape={list(x.shape)}, dt.shape={list(dt.shape)}, B.shape={list(B.shape)}")
        print(f"chunk_size={chunk_size}, dt_softplus={dt_softplus}, dt_limit={dt_limit}")

        # Dump inputs
        write_tensor(dr / "x_fp32.bin", x)
        write_tensor(dr / "dt_pre_fp32.bin", dt)
        write_tensor(dr / "A_fp32.bin", A)
        write_tensor(dr / "B_fp32.bin", B)
        write_tensor(dr / "C_fp32.bin", C)
        if D is not None:
            write_tensor(dr / "D_fp32.bin", D)
        if dt_bias is not None:
            write_tensor(dr / "dt_bias_fp32.bin", dt_bias)
        if initial_states is not None:
            write_tensor(dr / "initial_states_fp32.bin", initial_states)
        if z is not None:
            write_tensor(dr / "z_fp32.bin", z)

        # Ensure contiguity (same as original function)
        from vllm.model_executor.layers.mamba.ops.ssd_combined import is_int_pow_2
        assert is_int_pow_2(chunk_size)
        seqlen, nheads, headdim = x.shape
        _, ngroups, dstate = B.shape
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if x.stride(-1) != 1 and x.stride(0) != 1:
            x = x.contiguous()
        if z is not None and z.stride(-1) != 1 and z.stride(0) != 1:
            z = z.contiguous()
        if D is not None and D.stride(-1) != 1:
            D = D.contiguous()
        assert cu_seqlens is not None

        # Stage 1: chunk cumsum
        dA_cumsum, dt_out = _chunk_cumsum_fwd(
            dt, A, chunk_size, cu_chunk_seqlens,
            dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit,
        )
        write_tensor(dr / "dA_cumsum_fp32.bin", dA_cumsum)
        write_tensor(dr / "dt_chunk_fp32.bin", dt_out)

        # Stage 2: chunk state
        states = _chunk_state_fwd(
            B, x, dt_out, dA_cumsum, cu_chunk_seqlens, states_in_fp32=True
        )
        write_tensor(dr / "chunk_delta_fp32.bin", states)

        # Stage 3: state passing
        states_flat = _state_passing_fwd(
            rearrange(states, "... p n -> ... (p n)"),
            dA_cumsum,
            cu_chunk_seqlens,
            initial_states=rearrange(initial_states, "... p n -> ... (p n)")
            if initial_states is not None else None,
            seq_idx=seq_idx,
            out_dtype=state_dtype if state_dtype is not None else C.dtype,
        )
        states = rearrange(states_flat, "... (p n) -> ... p n", n=dstate)
        write_tensor(dr / "boundary_state_fp32.bin", states)

        # Stage 4: BMM chunk
        CB = _bmm_chunk_fwd(C, B, chunk_size, cu_chunk_seqlens, output_dtype=torch.float32)
        write_tensor(dr / "cb_chunk_fp32.bin", CB)

        # Stage 5: chunk scan
        _chunk_scan_fwd(
            CB, x, dt_out, dA_cumsum, C, states, cu_chunk_seqlens,
            out, seq_idx, D=D, z=z, initial_states=initial_states,
        )
        write_tensor(dr / "y_output_fp32.bin", out)

        # Final SSM state
        if last_chunk_indices is not None:
            final_states = states[last_chunk_indices]
        else:
            final_states = states
        write_tensor(dr / "ssm_state_out_fp32.bin", final_states)

        print(f"=== Capture complete ===\n")

        if return_intermediate_states:
            return states
        else:
            return final_states


def main() -> None:
    args = parse_args()
    configure_torch_precision()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir)
    prompts_fixture = Path(args.prompts_fixture)
    prompt_token_ids = load_prompt_token_ids(prompts_fixture)
    assert len(prompt_token_ids) == 40, f"expected 40 tokens, got {len(prompt_token_ids)}"

    # Write prompt tokens for verification
    (output_dir / "prompt_token_ids.json").write_text(
        json.dumps(prompt_token_ids, indent=2) + "\n", encoding="utf-8"
    )

    sampling_params = SamplingParams(
        temperature=0.0, top_k=1, top_p=1.0, max_tokens=1,
        ignore_eos=True, detokenize=False,
    )

    llm = LLM(
        model=str(model_dir),
        tokenizer=str(model_dir),
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=48,
        max_num_batched_tokens=48,
        max_num_seqs=1,
        enable_chunked_prefill=False,
    )

    # Set up the capture
    capture = ChunkedScanCapture(output_dir, target_call=args.target_layer)

    # Monkey-patch the combined scan function
    import vllm.model_executor.layers.mamba.ops.ssd_combined as ssd_mod
    original_fwd = ssd_mod._mamba_chunk_scan_combined_fwd

    def patched(*args, **kwargs):
        return capture.patched_fwd(original_fwd, *args, **kwargs)

    ssd_mod._mamba_chunk_scan_combined_fwd = patched

    # Also patch the module-level reference used by MambaMixer2
    # (it may import the function directly)
    try:
        import vllm.model_executor.layers.mamba.mamba_mixer2 as mixer_mod
        if hasattr(mixer_mod, '_mamba_chunk_scan_combined_fwd'):
            mixer_mod._mamba_chunk_scan_combined_fwd = patched
    except ImportError:
        pass

    try:
        outputs = llm.generate(
            [{"prompt_token_ids": prompt_token_ids}],
            sampling_params,
            use_tqdm=False,
        )
        print(f"Generated {len(outputs)} outputs")
        if outputs and outputs[0].outputs:
            token_ids = list(outputs[0].outputs[0].token_ids)
            print(f"First generated token: {token_ids[0] if token_ids else 'none'}")
    except Exception as e:
        print(f"Generation failed (expected if decode crashes): {e}")

    if capture.captured:
        print(f"\nSuccessfully captured intermediates for Mamba layer {args.target_layer}")
        print(f"Output directory: {output_dir}")
    else:
        print(f"\nWARNING: Did not capture layer {args.target_layer}")
        print(f"Total chunked-scan calls seen: {capture.call_count}")

    # Write metadata
    metadata = {
        "target_layer": args.target_layer,
        "captured": capture.captured,
        "total_calls": capture.call_count,
        "prompt_token_count": len(prompt_token_ids),
        "image_tag": args.image_tag,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
