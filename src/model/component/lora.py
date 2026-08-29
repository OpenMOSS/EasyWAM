from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora.layer import LoraLayer

from utils.logging_config import get_logger


logger = get_logger(__name__)

LORA_ADAPTER_NAME = "default"
LORA_CHECKPOINT_FORMAT = "trainable_lora"


def _normalize_lora_config(config: Any) -> LoraConfig:
    if isinstance(config, LoraConfig):
        return config
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    if isinstance(config, Mapping):
        config = dict(config)
        config.pop("_target_", None)
        return LoraConfig(**config)
    raise TypeError(
        f"`lora` must be a peft.LoraConfig or dict-like config, got {type(config)}"
    )


def iter_lora_layers(module: nn.Module):
    for name, child in module.named_modules():
        if isinstance(child, LoraLayer):
            yield name, child


def has_lora(module: nn.Module) -> bool:
    return any(True for _ in iter_lora_layers(module))


def inject_video_dit_lora(video_dit: nn.Module, config: Any) -> LoraConfig:
    if has_lora(video_dit):
        raise ValueError("Video DiT already contains LoRA layers.")

    lora_config = _normalize_lora_config(config)
    target_modules = lora_config.target_modules
    if target_modules is None or isinstance(target_modules, str):
        raise ValueError(
            "Video DiT LoRA requires an explicit list of `target_modules`."
        )
    try:
        target_modules = list(target_modules)
    except TypeError as error:
        raise ValueError(
            "Video DiT LoRA requires an explicit list of `target_modules`."
        ) from error
    if not target_modules:
        raise ValueError("Video DiT LoRA `target_modules` cannot be empty.")

    inject_adapter_in_model(
        lora_config,
        video_dit,
        adapter_name=LORA_ADAPTER_NAME,
    )
    injected = list(iter_lora_layers(video_dit))
    invalid = [name for name, _ in injected if not name.startswith("blocks.")]
    if invalid:
        raise RuntimeError(
            "LoRA was injected outside `video_dit.blocks`: "
            f"{invalid[:8]}"
        )

    blocks = getattr(video_dit, "blocks", None)
    if blocks is None:
        raise AttributeError("Video DiT must expose a `blocks` module list.")
    expected_count = len(blocks) * len(target_modules)
    if len(injected) != expected_count:
        raise RuntimeError(
            "Unexpected number of Video DiT LoRA layers: "
            f"expected={expected_count}, actual={len(injected)}, "
            f"targets={target_modules}"
        )

    video_dit._easywam_lora_config = lora_config
    logger.info(
        "Injected Video DiT LoRA: layers=%d rank=%d alpha=%d dropout=%.4f targets=%s",
        len(injected),
        int(lora_config.r),
        int(lora_config.lora_alpha),
        float(lora_config.lora_dropout),
        target_modules,
    )
    return lora_config


def set_video_dit_lora_trainable(video_dit: nn.Module) -> None:
    layers = list(iter_lora_layers(video_dit))
    if not layers:
        raise ValueError("Cannot enable LoRA training: Video DiT has no LoRA layers.")

    video_dit.requires_grad_(False)
    for _, layer in layers:
        layer.set_adapter(LORA_ADAPTER_NAME)
        layer.enable_adapters(enabled=True)


def is_lora_checkpoint(payload: Mapping[str, Any]) -> bool:
    return payload.get("format") == LORA_CHECKPOINT_FORMAT


def _find_lora_roots(model: nn.Module) -> dict[str, nn.Module]:
    roots = {
        name: module
        for name, module in model.named_modules()
        if name and hasattr(module, "_easywam_lora_config")
    }
    if not roots:
        raise ValueError("Cannot build LoRA checkpoint: model has no LoRA root.")
    return roots


def lora_model_checkpoint_state(model: nn.Module) -> dict[str, Any]:
    lora_roots = list(_find_lora_roots(model))
    trainable_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not trainable_parameters:
        raise RuntimeError("Cannot save LoRA checkpoint: model has no trainable parameters.")
    full_state = model.state_dict()
    missing = set(trainable_parameters).difference(full_state)
    if missing:
        raise RuntimeError(
            "Trainable parameters are missing from model state_dict: "
            f"{sorted(missing)[:8]}"
        )
    trainable_state = {
        name: tensor
        for name, tensor in full_state.items()
        if name in trainable_parameters
    }
    numel = sum(parameter.numel() for parameter in trainable_parameters.values())
    source_gib = sum(
        parameter.numel() * parameter.element_size()
        for parameter in trainable_parameters.values()
    ) / (1024**3)
    logger.info(
        "Prepared filtered trainable state_dict: tensors=%d params=%.6fB size=%.3fGiB",
        len(trainable_state),
        numel / 1e9,
        source_gib,
    )

    return {
        "lora_roots": lora_roots,
        "trainable_state": trainable_state,
    }


def build_lora_checkpoint_payload(
    model: nn.Module,
    *,
    step: int | None,
) -> dict[str, Any]:
    payload = {
        "format": LORA_CHECKPOINT_FORMAT,
        "step": step,
        "torch_dtype": str(model.torch_dtype),
        "backbone_name": getattr(model, "backbone_name", "wan22"),
        **lora_model_checkpoint_state(model),
    }
    logger.info("Finished building trainable LoRA checkpoint payload")
    return payload


def load_lora_model_checkpoint_state(
    model: nn.Module,
    payload: Mapping[str, Any],
    *,
    merge_after_load: bool,
) -> None:
    trainable_state = payload.get("trainable_state")
    lora_roots = payload.get("lora_roots")
    if not isinstance(trainable_state, Mapping) or not isinstance(
        lora_roots, (list, tuple)
    ):
        raise ValueError(
            "LoRA checkpoint must contain `trainable_state` and `lora_roots`."
        )

    for name in lora_roots:
        lora_root = model.get_submodule(name)
        if not has_lora(lora_root):
            raise RuntimeError(
                "LoRA checkpoint requires a model created with the matching "
                f"LoRA task config; adapter is missing at `{name}`."
            )

    incompatible = model.load_state_dict(trainable_state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected keys in trainable LoRA checkpoint: "
            f"{incompatible.unexpected_keys[:8]}"
        )

    if merge_after_load:
        for name in lora_roots:
            merge_and_unload_video_dit_lora(model.get_submodule(name))


def _replace_submodule(root: nn.Module, name: str, replacement: nn.Module) -> None:
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = root.get_submodule(parent_name)
    else:
        parent = root
        child_name = name
    setattr(parent, child_name, replacement)


@torch.no_grad()
def merge_and_unload_video_dit_lora(video_dit: nn.Module) -> None:
    layers = list(iter_lora_layers(video_dit))
    if not layers:
        raise ValueError("Cannot merge Video DiT LoRA: no LoRA layers found.")

    for name, layer in layers:
        layer.merge(
            safe_merge=True,
            adapter_names=[LORA_ADAPTER_NAME],
        )
        _replace_submodule(video_dit, name, layer.get_base_layer())

    if hasattr(video_dit, "peft_config"):
        delattr(video_dit, "peft_config")
    if hasattr(video_dit, "_easywam_lora_config"):
        delattr(video_dit, "_easywam_lora_config")
    if has_lora(video_dit):
        raise RuntimeError("Video DiT still contains PEFT layers after merge.")


def _reset_lora_delta(module: nn.Module) -> None:
    for _, layer in iter_lora_layers(module):
        if LORA_ADAPTER_NAME not in layer.lora_B:
            raise RuntimeError(
                f"LoRA layer is missing adapter `{LORA_ADAPTER_NAME}`."
            )
        lora_b = layer.lora_B[LORA_ADAPTER_NAME]
        nn.init.zeros_(lora_b.weight)


def load_standard_state_dict(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    strict: bool = False,
):
    """Load a normal Linear state dict into either a base or PEFT-injected module."""
    layers = dict(iter_lora_layers(module))
    if not layers:
        return module.load_state_dict(state_dict, strict=strict)

    translated = dict(state_dict)
    for name in layers:
        weight_key = f"{name}.weight"
        if weight_key in translated:
            translated[f"{name}.base_layer.weight"] = translated.pop(weight_key)
        bias_key = f"{name}.bias"
        if bias_key in translated:
            translated[f"{name}.base_layer.bias"] = translated.pop(bias_key)

    incompatible = module.load_state_dict(translated, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing_non_lora = [
        key
        for key in incompatible.missing_keys
        if ".lora_" not in key
    ]
    if unexpected or missing_non_lora:
        raise RuntimeError(
            "Failed to load standard checkpoint into LoRA model: "
            f"missing_non_lora={missing_non_lora[:8]}, unexpected={unexpected[:8]}"
        )
    _reset_lora_delta(module)
    return incompatible
