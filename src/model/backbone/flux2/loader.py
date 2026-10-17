from __future__ import annotations

from pathlib import Path

import torch

from ..contract import BackboneComponents, TextEncoderComponents, TextEncoding
from ..protocol import BLOCK_PROTOCOL_STAGED
from .action_initialization import initialize_from_flux as initialize_action
from .flux2_video_dit import Flux2VideoDiT
from .flux2_video_text_encoder import Flux2Qwen3TextEncoder
from .flux2_video_vae import Flux2VideoVAE


def load_flux2_components(
    *,
    model_path: str,
    ae_model_path: str,
    variant: str,
    qwen3_model_spec: str,
    qwen3_output_layers: list[int] | tuple[int, ...],
    tokenizer_max_len: int,
    device: str | torch.device,
    torch_dtype: torch.dtype,
    load_text_encoder: bool,
    attention_backend: str,
    skip_dit_load_from_pretrain: bool,
    use_gradient_checkpointing: bool = False,
) -> BackboneComponents:
    dit = Flux2VideoDiT.from_pretrained(
        model_path=model_path,
        variant=variant,
        device=str(device),
        torch_dtype=torch_dtype,
        attention_backend=attention_backend,
        skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
    )

    dit.use_gradient_checkpointing = bool(use_gradient_checkpointing)

    vae = Flux2VideoVAE.from_pretrained(
        ae_model_path,
        device=device,
        torch_dtype=torch_dtype,
    )

    text_encoder = None
    text_dim = 7680 if "4b" in str(variant).lower() else 12288
    if load_text_encoder:
        text_encoder = Flux2Qwen3TextEncoder.from_pretrained(
            qwen3_model_spec,
            torch_dtype=torch_dtype,
            output_layers=qwen3_output_layers,
            max_length=tokenizer_max_len,
        ).to(device)
        text_dim = int(text_encoder.dim)

    return BackboneComponents(
        name="flux2",
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=None,
        text_dim=text_dim,
        block_protocol=BLOCK_PROTOCOL_STAGED,
        dit_path=(
            "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else str(Path(model_path))
        ),
        vae_path=str(Path(ae_model_path)),
        text_encoder_path=qwen3_model_spec if load_text_encoder else None,
        tokenizer_path=qwen3_model_spec if load_text_encoder else None,
        metadata={"variant": variant, "qwen3_output_layers": list(qwen3_output_layers)},
    )


def _text_model(cfg):
    return str(
        cfg.get(
            "qwen3_model_spec",
            (
                "Qwen/Qwen3-4B"
                if "4b" in cfg.get("variant", "klein-base-4b").lower()
                else "Qwen/Qwen3-8B"
            ),
        )
    )


def load_components(cfg, *, device, torch_dtype, skip_dit_load_from_pretrain=False):
    return load_flux2_components(
        model_path=cfg["model_path"],
        ae_model_path=cfg["ae_model_path"],
        variant=str(cfg.get("variant", "klein-base-4b")),
        qwen3_model_spec=_text_model(cfg),
        qwen3_output_layers=tuple(cfg.get("qwen3_output_layers", (9, 18, 27))),
        tokenizer_max_len=int(cfg.get("tokenizer_max_len", 512)),
        device=device,
        torch_dtype=torch_dtype,
        load_text_encoder=bool(cfg.get("load_text_encoder", False)),
        attention_backend=str(cfg.get("attention_backend", "sdpa")),
        skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
        use_gradient_checkpointing=bool(cfg.get("use_gradient_checkpointing", False)),
    )


def load_dit(cfg, *, device, torch_dtype, skip_dit_load_from_pretrain=False):
    model = Flux2VideoDiT.from_pretrained(
        model_path=cfg["model_path"],
        variant=str(cfg.get("variant", "klein-base-4b")),
        device=str(device),
        torch_dtype=torch_dtype,
        attention_backend=str(cfg.get("attention_backend", "sdpa")),
        skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
    )
    model.use_gradient_checkpointing = bool(
        cfg.get("use_gradient_checkpointing", False)
    )
    return model


def load_text_encoder(cfg, *, device, torch_dtype, context_len):
    path = _text_model(cfg)
    model = (
        Flux2Qwen3TextEncoder.from_pretrained(
            path,
            torch_dtype=torch_dtype,
            output_layers=tuple(cfg.get("qwen3_output_layers", (9, 18, 27))),
            max_length=context_len,
        )
        .to(device)
        .eval()
    )

    def encode(prompts):
        context, mask = model(prompts)
        return TextEncoding(context, mask.bool(), mask)

    return TextEncoderComponents(model, encode, 1, path, path)
