#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import traceback
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

import torch
import vllm
from vllm import LLM, SamplingParams


def configure_torch_precision() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump authoritative vLLM prefill/decode traces for the NemotronH runtime parity fixture."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompts-fixture", required=True)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--decode-capture-layers", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(payload, sort_keys=True) + "\n")


def write_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.write_bytes(tensor.contiguous().cpu().float().numpy().tobytes())


def to_cpu_fp32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().cpu().float()


def extract_hidden_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected tensor output, got {type(output)!r}")
    return output


def extract_layer_boundary_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        if len(output) < 2:
            raise TypeError(
                f"Expected decoder-layer tuple output to have at least 2 items, got {len(output)}"
            )
        hidden_states = output[0]
        residual = output[1]
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(f"Expected hidden_states tensor, got {type(hidden_states)!r}")
        if residual is None:
            return hidden_states
        if not isinstance(residual, torch.Tensor):
            raise TypeError(f"Expected residual tensor or None, got {type(residual)!r}")
        return hidden_states + residual
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected tensor layer output, got {type(output)!r}")
    return output


def load_prompt_token_ids(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(token, int) for token in payload):
        raise TypeError(f"{path} must contain a JSON array of integer token IDs")
    return payload


def detect_vllm_commit() -> str:
    match = re.search(r"\+g([0-9a-f]{7,40})", getattr(vllm, "__version__", ""))
    if match:
        return match.group(1)
    return "b31e9326a"


@dataclass
class TraceCapture:
    output_dir: Path
    num_layers: int
    total_decode_steps: int
    decode_capture_layers: bool
    prefill_written: bool = False
    next_decode_forward_step: int = 1
    active_stage: str | None = None
    active_step: int | None = None
    pending_logit_stage: str | None = None
    pending_logit_step: int | None = None
    current_embedding: torch.Tensor | None = None
    current_layers: dict[int, torch.Tensor] = field(default_factory=dict)
    current_final_hidden: torch.Tensor | None = None
    current_final_hidden_normed: torch.Tensor | None = None
    prefill_argmax_token_id: int | None = None
    decode_argmax_token_ids: dict[int, int] = field(default_factory=dict)
    step_sources: dict[int, str] = field(default_factory=dict)
    step_logit_shapes: dict[int, list[int]] = field(default_factory=dict)
    step_layer_counts: dict[int, int] = field(default_factory=dict)
    forward_log_path: Path | None = None
    forward_call_index: int = 0

    def reset_current_stage(self) -> None:
        self.current_embedding = None
        self.current_layers = {}
        self.current_final_hidden = None
        self.current_final_hidden_normed = None

    def record_forward_call(
        self,
        *,
        phase: str,
        num_tokens: int,
        stage: str | None,
        step: int | None,
        positions_shape: list[int] | None = None,
        output_type: str | None = None,
        error: str | None = None,
    ) -> None:
        if self.forward_log_path is None:
            return
        self.forward_call_index += 1
        append_jsonl(
            self.forward_log_path,
            {
                "index": self.forward_call_index,
                "phase": phase,
                "num_tokens": num_tokens,
                "stage": stage,
                "step": step,
                "positions_shape": positions_shape,
                "output_type": output_type,
                "error": error,
            },
        )

    def begin_stage(self, stage: str | None, step: int | None) -> None:
        self.active_stage = stage
        self.active_step = step
        self.pending_logit_stage = None
        self.pending_logit_step = None
        self.reset_current_stage()

    def finish_stage(self) -> None:
        self.active_stage = None
        self.active_step = None
        self.pending_logit_stage = None
        self.pending_logit_step = None
        self.reset_current_stage()

    def capture_embedding(self, tensor: torch.Tensor) -> None:
        if self.active_stage == "prefill":
            self.current_embedding = to_cpu_fp32(tensor)

    def capture_layer(self, layer_idx: int, tensor: torch.Tensor) -> None:
        if self.active_stage == "prefill":
            self.current_layers[layer_idx] = to_cpu_fp32(tensor)
            return
        if self.active_stage == "decode" and self.decode_capture_layers:
            self.current_layers[layer_idx] = to_cpu_fp32(tensor)

    def capture_final_norm(self, inputs: tuple[Any, ...], output: Any) -> None:
        if self.active_stage != "prefill":
            return
        if isinstance(output, tuple):
            normed_tensor = output[0]
        else:
            normed_tensor = output
        if not isinstance(normed_tensor, torch.Tensor):
            raise TypeError("Unexpected final norm hook payload")
        if not inputs:
            raise TypeError("Final norm hook did not receive hidden-state inputs")
        final_hidden_tensor = inputs[0]
        if not isinstance(final_hidden_tensor, torch.Tensor):
            raise TypeError("Unexpected final norm hidden-state input payload")
        if len(inputs) > 1 and inputs[1] is not None:
            residual_tensor = inputs[1]
            if not isinstance(residual_tensor, torch.Tensor):
                raise TypeError("Unexpected final norm residual input payload")
            final_hidden_tensor = final_hidden_tensor + residual_tensor
        self.current_final_hidden = to_cpu_fp32(final_hidden_tensor)
        self.current_final_hidden_normed = to_cpu_fp32(normed_tensor)

    def _step_dir(self, step: int) -> Path:
        return self.output_dir / f"decode_step_{step:03d}"

    def _write_step_payload(
        self,
        *,
        step: int,
        logits_cpu: torch.Tensor,
        source: str,
        layer_tensors: dict[int, torch.Tensor] | None,
        argmax_token_id: int,
    ) -> None:
        step_dir = self._step_dir(step)
        step_dir.mkdir(parents=True, exist_ok=True)
        write_tensor(step_dir / "vllm_decode_logits_fp32.bin", logits_cpu)

        layer_count = 0
        if layer_tensors is not None:
            if len(layer_tensors) != self.num_layers:
                raise RuntimeError(
                    f"Expected {self.num_layers} layer tensors for step {step}, found {len(layer_tensors)}"
                )
            for layer_idx in range(self.num_layers):
                write_tensor(
                    step_dir / f"vllm_layer_{layer_idx:03d}_output_fp32.bin",
                    layer_tensors[layer_idx],
                )
            layer_count = self.num_layers

        self.decode_argmax_token_ids[step] = argmax_token_id
        self.step_sources[step] = source
        self.step_logit_shapes[step] = list(logits_cpu.shape)
        self.step_layer_counts[step] = layer_count

    def capture_logits(self, logits: torch.Tensor) -> None:
        if self.pending_logit_stage is None:
            return
        logits_cpu = to_cpu_fp32(logits)
        if logits_cpu.ndim != 2 or logits_cpu.shape[0] != 1:
            raise RuntimeError(
                f"Expected [1, vocab] logits for captured step, got {tuple(logits_cpu.shape)}"
            )
        argmax_token_id = int(torch.argmax(logits_cpu[0]).item())

        if self.pending_logit_stage == "prefill":
            if self.current_embedding is None:
                raise RuntimeError("Prefill embedding output was not captured")
            if len(self.current_layers) != self.num_layers:
                raise RuntimeError(
                    f"Expected {self.num_layers} prefill layer outputs, found {len(self.current_layers)}"
                )
            if self.current_final_hidden is None or self.current_final_hidden_normed is None:
                raise RuntimeError("Final prefill hidden states were not captured")

            write_tensor(self.output_dir / "vllm_embedding_output_fp32.bin", self.current_embedding)
            for layer_idx in range(self.num_layers):
                write_tensor(
                    self.output_dir / f"vllm_layer_{layer_idx:03d}_output_fp32.bin",
                    self.current_layers[layer_idx],
                )
            write_tensor(self.output_dir / "vllm_final_hidden_fp32.bin", self.current_final_hidden)
            write_tensor(
                self.output_dir / "vllm_final_hidden_normed_fp32.bin",
                self.current_final_hidden_normed,
            )
            write_tensor(self.output_dir / "vllm_prefill_logits_fp32.bin", logits_cpu)

            self.prefill_argmax_token_id = argmax_token_id
            step0_layers = None
            if self.decode_capture_layers:
                step0_layers = {
                    layer_idx: self.current_layers[layer_idx][-1:].contiguous()
                    for layer_idx in range(self.num_layers)
                }
            # The first generated token is selected from the prompt terminal
            # logits. Expose it as decode_step_000 so the decode-step numbering
            # stays aligned with max_tokens=decode_steps.
            self._write_step_payload(
                step=0,
                logits_cpu=logits_cpu,
                source="prefill_terminal_logits",
                layer_tensors=step0_layers,
                argmax_token_id=argmax_token_id,
            )
            self.prefill_written = True
        elif self.pending_logit_stage == "decode":
            step = self.pending_logit_step
            if step is None:
                raise RuntimeError("Decode step index was not set")
            step_layers = self.current_layers if self.decode_capture_layers else None
            self._write_step_payload(
                step=step,
                logits_cpu=logits_cpu,
                source="decode_forward",
                layer_tensors=step_layers,
                argmax_token_id=argmax_token_id,
            )
        else:
            raise RuntimeError(f"Unexpected pending stage: {self.pending_logit_stage}")

        self.finish_stage()


def install_trace_hooks(
    runtime_model: Any,
    trace: TraceCapture,
    prompt_token_count: int,
):
    handles: list[Any] = []

    def embedding_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        trace.capture_embedding(extract_hidden_tensor(output))

    def layer_hook(layer_idx: int):
        def _hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            trace.capture_layer(layer_idx, extract_layer_boundary_tensor(output))

        return _hook

    def final_norm_hook(module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        del module
        trace.capture_final_norm(inputs, output)

    handles.append(runtime_model.model.embed_tokens.register_forward_hook(embedding_hook))
    for layer_idx in range(trace.num_layers):
        handles.append(runtime_model.model.layers[layer_idx].register_forward_hook(layer_hook(layer_idx)))
    handles.append(runtime_model.model.norm_f.register_forward_hook(final_norm_hook))

    original_inner_forward = runtime_model.model.forward
    original_compute_logits = runtime_model.compute_logits

    def wrapped_inner_forward(
        inner_self: Any,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> Any:
        del inner_self
        if input_ids is not None:
            num_tokens = int(input_ids.numel())
        elif inputs_embeds is not None:
            num_tokens = int(inputs_embeds.shape[0])
        else:
            num_tokens = 0

        stage = None
        step = None
        if num_tokens == prompt_token_count and not trace.prefill_written:
            stage = "prefill"
        elif num_tokens == 1 and trace.next_decode_forward_step < trace.total_decode_steps:
            stage = "decode"
            step = trace.next_decode_forward_step
            trace.next_decode_forward_step += 1

        trace.begin_stage(stage, step)
        trace.record_forward_call(
            phase="begin",
            num_tokens=num_tokens,
            stage=stage,
            step=step,
            positions_shape=list(positions.shape),
        )
        try:
            result = original_inner_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )
        except Exception as exc:
            trace.record_forward_call(
                phase="exception",
                num_tokens=num_tokens,
                stage=stage,
                step=step,
                positions_shape=list(positions.shape),
                error=repr(exc),
            )
            trace.finish_stage()
            raise
        if stage is not None:
            trace.pending_logit_stage = stage
            trace.pending_logit_step = step
        else:
            trace.finish_stage()
        trace.record_forward_call(
            phase="end",
            num_tokens=num_tokens,
            stage=stage,
            step=step,
            positions_shape=list(positions.shape),
            output_type=type(result).__name__,
        )
        return result

    def wrapped_compute_logits(model_self: Any, hidden_states: torch.Tensor) -> torch.Tensor | None:
        del model_self
        logits = original_compute_logits(hidden_states)
        if logits is not None:
            trace.capture_logits(logits)
        return logits

    runtime_model.model.forward = types.MethodType(wrapped_inner_forward, runtime_model.model)
    runtime_model.compute_logits = types.MethodType(wrapped_compute_logits, runtime_model)

    def cleanup() -> None:
        runtime_model.model.forward = original_inner_forward
        runtime_model.compute_logits = original_compute_logits
        for handle in reversed(handles):
            handle.remove()

    return cleanup


def build_llm(model_dir: str, prompt_token_count: int, decode_steps: int) -> LLM:
    return LLM(
        model=model_dir,
        tokenizer=model_dir,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=prompt_token_count + decode_steps,
        max_num_batched_tokens=prompt_token_count + decode_steps,
        max_num_seqs=1,
        enable_chunked_prefill=False,
    )


def write_decode_summaries(
    output_dir: Path,
    generated_token_ids: list[int],
    trace: TraceCapture,
) -> None:
    for step in range(trace.total_decode_steps):
        if step not in trace.decode_argmax_token_ids:
            raise RuntimeError(f"Missing captured logits for decode step {step}")
        token_id = int(generated_token_ids[step])
        argmax_token_id = trace.decode_argmax_token_ids[step]
        if token_id != argmax_token_id:
            raise RuntimeError(
                f"Greedy decode mismatch at step {step}: output token {token_id}, logits argmax {argmax_token_id}"
            )
        summary = [
            f"decode_step: {step:03d}",
            f"source: {trace.step_sources.get(step, 'unknown')}",
            f"generated_token_id: {token_id}",
            f"argmax_token_id: {argmax_token_id}",
            f"logits_shape: {trace.step_logit_shapes.get(step)}",
            f"decode_capture_layers: {trace.decode_capture_layers}",
            f"captured_layer_count: {trace.step_layer_counts.get(step, 0)}",
        ]
        (output_dir / f"decode_step_{step:03d}" / "summary.txt").write_text(
            "\n".join(summary) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    configure_torch_precision()

    if args.decode_steps < 1:
        raise ValueError("--decode-steps must be at least 1")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"model directory does not exist: {model_dir}")

    prompts_fixture = Path(args.prompts_fixture)
    if not prompts_fixture.exists():
        raise FileNotFoundError(f"prompts fixture does not exist: {prompts_fixture}")

    prompt_token_ids = load_prompt_token_ids(prompts_fixture)
    if len(prompt_token_ids) != 40:
        raise ValueError(f"expected 40 prompt tokens, found {len(prompt_token_ids)}")
    write_json(output_dir / "prompt_token_ids.json", prompt_token_ids)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        max_tokens=args.decode_steps,
        ignore_eos=True,
        detokenize=False,
    )

    llm: LLM | None = None
    cleanup_hooks = None
    try:
        llm = build_llm(str(model_dir), len(prompt_token_ids), args.decode_steps)

        # The pinned image defaults to multiprocess mode; with
        # VLLM_ENABLE_V1_MULTIPROCESSING=0 this resolves to a direct in-process
        # worker/model path that we can instrument safely.
        runtime_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        num_layers = len(runtime_model.model.layers)
        if num_layers != 88:
            raise ValueError(f"expected 88 layers, found {num_layers}")

        trace = TraceCapture(
            output_dir=output_dir,
            num_layers=num_layers,
            total_decode_steps=args.decode_steps,
            decode_capture_layers=args.decode_capture_layers,
            forward_log_path=output_dir / "forward_calls.jsonl",
        )
        cleanup_hooks = install_trace_hooks(runtime_model, trace, len(prompt_token_ids))

        outputs = llm.generate(
            [{"prompt_token_ids": prompt_token_ids}],
            sampling_params,
            use_tqdm=False,
        )
        if len(outputs) != 1:
            raise RuntimeError(f"expected exactly one request output, found {len(outputs)}")

        request_output = outputs[0]
        if request_output.prompt_token_ids is not None:
            actual_prompt_token_ids = list(request_output.prompt_token_ids)
            if actual_prompt_token_ids != prompt_token_ids:
                raise RuntimeError("vLLM returned prompt_token_ids that do not match the fixture exactly")

        if len(request_output.outputs) != 1:
            raise RuntimeError(f"expected exactly one completion output, found {len(request_output.outputs)}")

        generated_token_ids = list(request_output.outputs[0].token_ids)
        if len(generated_token_ids) != args.decode_steps:
            raise RuntimeError(
                f"expected {args.decode_steps} generated token IDs, found {len(generated_token_ids)}"
            )
        if trace.prefill_argmax_token_id is None:
            raise RuntimeError("prefill logits were not captured")
        if int(generated_token_ids[0]) != trace.prefill_argmax_token_id:
            raise RuntimeError(
                "first generated token does not match the prefill logits argmax"
            )

        write_json(output_dir / "generated_token_ids.json", generated_token_ids)
        write_decode_summaries(output_dir, generated_token_ids, trace)

        tokenizer = llm.get_tokenizer()
        tokenizer_name = getattr(tokenizer, "name_or_path", None) or str(model_dir)
        metadata = {
            "vllm_commit": detect_vllm_commit(),
            "image_tag": args.image_tag,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "model_dir": str(model_dir),
            "tokenizer": str(tokenizer_name),
            "decode_mode": "greedy",
            "sampling_config": {
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "max_tokens": args.decode_steps,
                "ignore_eos": True,
            },
            "hidden_size": int(runtime_model.config.hidden_size),
            "vocab_size": int(runtime_model.config.vocab_size),
            "num_layers": num_layers,
            "prompt_token_count": len(prompt_token_ids),
            "decode_step_count": args.decode_steps,
        }
        write_json(output_dir / "metadata.json", metadata)
    except Exception as exc:
        failure = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(output_dir / "failure.json", failure)
        raise
    finally:
        if cleanup_hooks is not None:
            cleanup_hooks()


if __name__ == "__main__":
    main()
