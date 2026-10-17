from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .registry import BACKBONE_MODULES, get_backbone_backend


def normalize_backbone_config(config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config, Mapping):
        raise TypeError(f"backbone config must be dict-like, got {type(config)}")
    result = dict(config)
    name = str(result["name"]).strip().lower()
    if name not in BACKBONE_MODULES:
        raise ValueError(f"Unsupported backbone name: {name!r}")
    result["name"] = name
    return result


def load_easywam_backbone(config, **kwargs):
    cfg = normalize_backbone_config(config)
    return get_backbone_backend(cfg["name"]).load_components(cfg, **kwargs)


def load_backbone_dit(config, **kwargs):
    cfg = normalize_backbone_config(config)
    return get_backbone_backend(cfg["name"]).load_dit(cfg, **kwargs)


def load_backbone_text_encoder(config, **kwargs):
    cfg = normalize_backbone_config(config)
    return get_backbone_backend(cfg["name"]).load_text_encoder(cfg, **kwargs)


def initialize_action_from_backbone(config, action, visual, *, alpha=True):
    cfg = normalize_backbone_config(config)
    return get_backbone_backend(cfg["name"]).initialize_action(
        action, visual, alpha=alpha
    )
