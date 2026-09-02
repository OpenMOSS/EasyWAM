"""Backbone-neutral component loading for EasyWAM models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from .contract import BackboneComponents
from .protocol import BLOCK_PROTOCOL_MAIN


def normalize_backbone_config(config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config, Mapping):
        raise TypeError(f"backbone config must be dict-like, got {type(config)}")
    result = dict(config)
    name = str(result.get("name", "")).strip().lower()
    if name not in {"wan22", "cosmos25", "flux2"}:
        raise ValueError(f"Unsupported backbone name: {name!r}")
    result["name"] = name
    return result


def load_easywam_backbone(
    config: Mapping[str, Any] | DictConfig,
    *,
    device: str | torch.device,
    torch_dtype: torch.dtype,
    skip_dit_load_from_pretrain: bool = False,
):
    cfg = normalize_backbone_config(config)
    name = cfg["name"]
    if name == "wan22":
        from .wan22.loader import load_wan22_ti2v_5b_components

        loaded = load_wan22_ti2v_5b_components(
            device=str(device),
            torch_dtype=torch_dtype,
            model_id=cfg["model_id"],
            tokenizer_model_id=cfg["tokenizer_model_id"],
            tokenizer_max_len=int(cfg.get("tokenizer_max_len", 128)),
            dit_config=dict(cfg["dit_config"]),
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=bool(cfg.get("load_text_encoder", False)),
        )
        return BackboneComponents(
            name=name,
            dit=loaded.dit,
            vae=loaded.vae,
            text_encoder=loaded.text_encoder,
            tokenizer=loaded.tokenizer,
            text_dim=int(cfg["text_dim"]),
            block_protocol=BLOCK_PROTOCOL_MAIN,
            dit_path=loaded.dit_path,
            vae_path=loaded.vae_path,
            text_encoder_path=loaded.text_encoder_path,
            tokenizer_path=loaded.tokenizer_path,
        )

    if name == "flux2":
        from .flux2.loader import load_flux2_components

        variant = str(cfg.get("variant", "klein-base-4b"))
        default_qwen = "Qwen/Qwen3-4B" if "4b" in variant.lower() else "Qwen/Qwen3-8B"
        return load_flux2_components(
            model_path=cfg["model_path"],
            ae_model_path=cfg["ae_model_path"],
            variant=variant,
            flux2_src_path=cfg.get("flux2_src_path"),
            qwen3_model_spec=str(cfg.get("qwen3_model_spec", default_qwen)),
            qwen3_output_layers=tuple(cfg.get("qwen3_output_layers", (9, 18, 27))),
            tokenizer_max_len=int(cfg.get("tokenizer_max_len", 512)),
            device=device,
            torch_dtype=torch_dtype,
            load_text_encoder=bool(cfg.get("load_text_encoder", False)),
            attention_backend=str(cfg.get("attention_backend", "sdpa")),
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
        )

    from .cosmos25.loader import load_cosmos25_components

    loaded = load_cosmos25_components(
        model_id=cfg["model_id"],
        reason_model_id=cfg["reason_model_id"],
        device=device,
        torch_dtype=torch_dtype,
        load_text_encoder=bool(cfg.get("load_text_encoder", False)),
        tokenizer_max_len=int(cfg.get("tokenizer_max_len", 128)),
        attention_backend=str(cfg.get("attention_backend", "sdpa")),
        use_gradient_checkpointing=bool(cfg.get("use_gradient_checkpointing", False)),
        video_attention_mask_mode=str(
            cfg.get("video_attention_mask_mode", "bidirectional")
        ),
        conditional_frame_timestep=float(
            cfg.get("video_scheduler", {}).get("conditional_frame_timestep", 0.0001)
        ),
        skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
    )
    return BackboneComponents(
        name=name,
        dit=loaded.dit,
        vae=loaded.vae,
        text_encoder=loaded.text_encoder,
        tokenizer=loaded.tokenizer,
        text_dim=int(cfg["text_dim"]),
        block_protocol=BLOCK_PROTOCOL_MAIN,
        dit_path=loaded.dit_path,
        vae_path=loaded.vae_path,
        text_encoder_path=loaded.text_encoder_path,
        tokenizer_path=loaded.tokenizer_path,
    )
