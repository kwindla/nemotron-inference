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


def tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "is_contiguous": bool(tensor.is_contiguous()),
    }


def load_prompt_token_ids(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(t, int) for t in payload):
        raise TypeError(f"{path} must contain a JSON array of integer token IDs")
    return payload


def build_in_proj_metadata(llm: LLM, target_layer: int) -> dict[str, Any]:
    def resolve_layer_root(model: Any) -> Any | None:
        seen: set[int] = set()
        queue = [model]
        while queue:
            current = queue.pop(0)
            if current is None:
                continue
            current_id = id(current)
            if current_id in seen:
                continue
            seen.add(current_id)
            if hasattr(current, "backbone") or hasattr(current, "layers"):
                return current
            for attr in ("model", "module", "inner_model"):
                if hasattr(current, attr):
                    queue.append(getattr(current, attr))
        return None

    runtime_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    root_model = resolve_layer_root(runtime_model)
    if root_model is None:
        return {
            "error": "runtime model has no layer container",
            "runtime_model_type": type(runtime_model).__name__,
        }
    layer_container = getattr(root_model, "backbone", root_model)
    layer = layer_container.layers[target_layer]
    mixer = layer.mixer
    in_proj = mixer.in_proj
    quant_method = getattr(in_proj, "quant_method", None)
    fp8_linear = getattr(quant_method, "fp8_linear", None)
    quant_config = getattr(quant_method, "quant_config", None)
    return {
        "runtime_model_type": type(runtime_model).__name__,
        "root_model_type": type(root_model).__name__,
        "layer_container_type": type(layer_container).__name__,
        "layer_type": type(layer).__name__,
        "mixer_type": type(mixer).__name__,
        "in_proj_type": type(in_proj).__name__,
        "quant_method_type": type(quant_method).__name__ if quant_method is not None else None,
        "fp8_linear_type": type(fp8_linear).__name__ if fp8_linear is not None else None,
        "fp8_output_padding": (
            fp8_linear.get_output_padding() if fp8_linear is not None else None
        ),
        "weight_dtype": str(getattr(in_proj, "weight").dtype),
        "weight_shape": list(getattr(in_proj, "weight").shape),
        "weight_stride": list(getattr(in_proj, "weight").stride()),
        "weight_is_contiguous": bool(getattr(in_proj, "weight").is_contiguous()),
        "weight_scale_dtype": str(getattr(in_proj, "weight_scale").dtype),
        "weight_scale_shape": list(getattr(in_proj, "weight_scale").shape),
        "input_scale_dtype": str(getattr(in_proj, "input_scale").dtype),
        "input_scale_shape": list(getattr(in_proj, "input_scale").shape),
        "quant_config_type": type(quant_config).__name__ if quant_config is not None else None,
    }


class ChunkedScanCapture:
    """Intercepts calls to _mamba_chunk_scan_combined_fwd and dumps intermediates."""

    def __init__(self, output_dir: Path, target_call: int):
        self.output_dir = output_dir
        self.target_call = target_call
        self.call_count = 0
        self.conv_call_count = 0
        self.forward_call_count = 0
        self.captured = False
        self.captured_norm = False
        self.captured_preconv = False
        self.captured_postconv = False
        self.capture_postconv_next = False

    def patched_mixer_forward(
        self,
        original_mixer_forward,
        mixer,
        hidden_states,
        mup_vector,
    ):
        current_call = self.forward_call_count
        self.forward_call_count += 1

        if current_call == self.target_call and not self.captured_norm:
            self.captured_norm = True
            dr = self.output_dir
            write_tensor(dr / "norm_output_fp32.bin", hidden_states)
            if mup_vector is not None:
                write_tensor(dr / "mup_vector_fp32.bin", mup_vector)

        return original_mixer_forward(
            mixer,
            hidden_states,
            mup_vector=mup_vector,
        )

    def patched_conv_ssm_forward(
        self,
        original_conv_ssm_forward,
        mixer,
        projected_states,
        output,
    ):
        current_call = self.conv_call_count
        self.conv_call_count += 1

        if current_call == self.target_call and not self.captured_preconv:
            self.captured_preconv = True
            hidden_states_B_C, dt = torch.split(
                projected_states[..., mixer.tped_intermediate_size :],
                [mixer.tped_conv_size, mixer.tped_dt_size],
                dim=-1,
            )
            dr = self.output_dir
            write_tensor(dr / "projected_states_fp32.bin", projected_states)
            write_tensor(dr / "hidden_states_B_C_pre_conv_fp32.bin", hidden_states_B_C)
            write_tensor(dr / "dt_pre_conv_fp32.bin", dt)
            write_tensor(dr / "conv1d_weight_fp32.bin", mixer.conv_weights)
            write_tensor(dr / "conv1d_bias_fp32.bin", mixer.conv1d.bias)

        should_capture_postconv = (
            current_call == self.target_call and not self.captured_postconv
        )
        if should_capture_postconv:
            self.capture_postconv_next = True
        try:
            return original_conv_ssm_forward(mixer, projected_states, output)
        finally:
            if should_capture_postconv:
                self.capture_postconv_next = False

    def patched_causal_conv1d_fn(self, original_causal_conv1d_fn, *args, **kwargs):
        out = original_causal_conv1d_fn(*args, **kwargs)
        if self.capture_postconv_next and not self.captured_postconv:
            self.captured_postconv = True
            # causal_conv1d_fn returns channel-major [dim, tokens]; dump token-major
            # to match the rest of the captured Mamba surfaces.
            write_tensor(self.output_dir / "hidden_states_B_C_post_conv_fp32.bin", out.transpose(0, 1))
        return out

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

        metadata = {
            "call_index": current_call,
            "chunk_size": int(chunk_size),
            "dt_softplus": bool(dt_softplus),
            "dt_limit": [float(dt_limit[0]), str(dt_limit[1])],
            "x": tensor_metadata(x),
            "dt": tensor_metadata(dt),
            "A": tensor_metadata(A),
            "B": tensor_metadata(B),
            "C": tensor_metadata(C),
            "out": tensor_metadata(out),
        }
        if D is not None:
            metadata["D"] = tensor_metadata(D)
        if dt_bias is not None:
            metadata["dt_bias"] = tensor_metadata(dt_bias)
        if initial_states is not None:
            metadata["initial_states"] = tensor_metadata(initial_states)
        if z is not None:
            metadata["z"] = tensor_metadata(z)
        (dr / "capture_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"dump: {dr / 'capture_metadata.json'}")

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
    prompt_token_count = len(prompt_token_ids)
    if prompt_token_count == 0:
        raise ValueError("prompt fixture must contain at least one token")

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
        max_model_len=max(48, prompt_token_count + 8),
        max_num_batched_tokens=max(48, prompt_token_count + 8),
        max_num_seqs=1,
        enable_chunked_prefill=False,
    )
    in_proj_metadata = build_in_proj_metadata(llm, args.target_layer)
    (output_dir / "in_proj_metadata.json").write_text(
        json.dumps(in_proj_metadata, indent=2) + "\n",
        encoding="utf-8",
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
        import vllm.model_executor.layers.mamba.ops.causal_conv1d as causal_conv_mod
        if hasattr(mixer_mod, '_mamba_chunk_scan_combined_fwd'):
            mixer_mod._mamba_chunk_scan_combined_fwd = patched
        if hasattr(mixer_mod, "MambaMixer2") and hasattr(mixer_mod.MambaMixer2, "forward"):
            original_mixer_forward = mixer_mod.MambaMixer2.forward

            def patched_mixer_forward(self, hidden_states, mup_vector=None):
                return capture.patched_mixer_forward(
                    original_mixer_forward,
                    self,
                    hidden_states,
                    mup_vector,
                )

            mixer_mod.MambaMixer2.forward = patched_mixer_forward
        if hasattr(mixer_mod, "MambaMixer2") and hasattr(mixer_mod.MambaMixer2, "conv_ssm_forward"):
            original_conv_ssm_forward = mixer_mod.MambaMixer2.conv_ssm_forward

            def patched_conv_ssm_forward(self, projected_states, output):
                return capture.patched_conv_ssm_forward(
                    original_conv_ssm_forward,
                    self,
                    projected_states,
                    output,
                )

            mixer_mod.MambaMixer2.conv_ssm_forward = patched_conv_ssm_forward
        if hasattr(causal_conv_mod, "causal_conv1d_fn"):
            original_causal_conv1d_fn = causal_conv_mod.causal_conv1d_fn

            def patched_causal_conv1d_fn(*args, **kwargs):
                return capture.patched_causal_conv1d_fn(
                    original_causal_conv1d_fn,
                    *args,
                    **kwargs,
                )

            causal_conv_mod.causal_conv1d_fn = patched_causal_conv1d_fn
            if hasattr(mixer_mod, "causal_conv1d_fn"):
                mixer_mod.causal_conv1d_fn = patched_causal_conv1d_fn
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
        "captured_norm": capture.captured_norm,
        "captured_preconv": capture.captured_preconv,
        "captured_postconv": capture.captured_postconv,
        "total_calls": capture.call_count,
        "prompt_token_count": len(prompt_token_ids),
        "image_tag": args.image_tag,
        "in_proj_metadata": in_proj_metadata,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
