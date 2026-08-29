"""Backbone-neutral component loading for EasyWAM models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf


def normalize_backbone_config(config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config, Mapping):
        raise TypeError(f"backbone config must be dict-like, got {type(config)}")
    result = dict(config)
    name = str(result.get("name", "")).strip().lower()
    if name not in {"wan22", "cosmos25"}:
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

        return load_wan22_ti2v_5b_components(
            device=str(device),
            torch_dtype=torch_dtype,
            model_id=cfg["model_id"],
            tokenizer_model_id=cfg["tokenizer_model_id"],
            tokenizer_max_len=int(cfg.get("tokenizer_max_len", 128)),
            dit_config=dict(cfg["dit_config"]),
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=bool(cfg.get("load_text_encoder", False)),
        )

    from .cosmos25.loader import load_cosmos25_components

    return load_cosmos25_components(
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
        skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
    )
