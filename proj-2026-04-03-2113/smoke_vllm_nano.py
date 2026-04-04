#!/usr/bin/env python3

from __future__ import annotations

from typing import Any


def collect_worker_moe_metadata(llm: Any) -> list[dict[str, Any]]:
    def _collect(model: Any) -> list[dict[str, Any]]:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE

        layers: list[dict[str, Any]] = []
        for name, module in model.named_modules():
            if not isinstance(module, FusedMoE):
                continue
            quant_method = getattr(module, "quant_method", None)
            nvfp4_backend = getattr(quant_method, "nvfp4_backend", None)
            layers.append(
                {
                    "module_name": name,
                    "layer_id": int(module.layer_id),
                    "requested_moe_backend": getattr(module.moe_config, "moe_backend", None),
                    "selected_nvfp4_backend": (
                        getattr(nvfp4_backend, "value", str(nvfp4_backend))
                        if nvfp4_backend is not None
                        else None
                    ),
                }
            )
        return sorted(layers, key=lambda item: item["layer_id"])

    results = llm.apply_model(_collect)
    return results[0] if results else []


def main() -> None:
    from vllm import LLM, SamplingParams

    print("creating LLM...")
    llm = LLM(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
        trust_remote_code=True,
        max_model_len=256,
        max_num_batched_tokens=256,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        moe_backend="flashinfer_cutlass",
    )

    layers = collect_worker_moe_metadata(llm)
    if layers:
        print("first_moe_layer:", layers[0])

    print("running generate...")
    out = llm.generate(
        ["Hello"],
        SamplingParams(max_tokens=8, temperature=0.0, top_p=1.0),
    )
    print("OUTPUT_START")
    print(repr(out[0].outputs[0].text))
    print("OUTPUT_END")


if __name__ == "__main__":
    main()
