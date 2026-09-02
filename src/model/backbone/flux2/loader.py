from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors

from ..contract import BackboneComponents
from ..protocol import BLOCK_PROTOCOL_FLUX2
from .flux2_video_expert import Flux2VideoExpert
from .imports import ensure_flux2_importable
from .text_encoder import Flux2Qwen3TextEncoder


def load_flux2_components(
    *,
    model_path: str,
    ae_model_path: str,
    variant: str,
    flux2_src_path: str | None,
    qwen3_model_spec: str,
    qwen3_output_layers: list[int] | tuple[int, ...],
    tokenizer_max_len: int,
    device: str | torch.device,
    torch_dtype: torch.dtype,
    load_text_encoder: bool,
    attention_backend: str,
    skip_dit_load_from_pretrain: bool,
) -> BackboneComponents:
    ensure_flux2_importable(flux2_src_path)
    video_expert = Flux2VideoExpert.from_pretrained(
        model_path=model_path,
        variant=variant,
        flux2_src_path=flux2_src_path,
        device=str(device),
        torch_dtype=torch_dtype,
        attention_backend=attention_backend,
        skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
    )

    from flux2.autoencoder import AutoEncoder, AutoEncoderParams

    with torch.device("meta"):
        vae = AutoEncoder(AutoEncoderParams()).to(torch_dtype)
    ae_state = load_safetensors(str(ae_model_path), device=str(device))
    vae.load_state_dict(ae_state, strict=True, assign=True)
    vae = vae.eval().requires_grad_(False).to(device=device, dtype=torch_dtype)

    text_encoder = None
    tokenizer = None
    text_dim = 7680 if "4b" in str(variant).lower() else 12288
    if load_text_encoder:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(qwen3_model_spec)
        qwen = AutoModelForCausalLM.from_pretrained(
            qwen3_model_spec,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).eval().requires_grad_(False).to(device)
        text_encoder = Flux2Qwen3TextEncoder(
            qwen,
            tokenizer,
            output_layers=qwen3_output_layers,
            max_length=tokenizer_max_len,
        )
        text_dim = int(text_encoder.dim)

    return BackboneComponents(
        name="flux2",
        dit=video_expert,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=None,
        text_dim=text_dim,
        block_protocol=BLOCK_PROTOCOL_FLUX2,
        dit_path="SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else str(Path(model_path)),
        vae_path=str(Path(ae_model_path)),
        text_encoder_path=qwen3_model_spec if load_text_encoder else None,
        tokenizer_path=qwen3_model_spec if load_text_encoder else None,
        metadata={"variant": variant, "qwen3_output_layers": list(qwen3_output_layers)},
    )
