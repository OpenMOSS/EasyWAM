from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch

from utils.logging_config import get_logger


logger = get_logger(__name__)


def configure_model_execution(
    model: torch.nn.Module,
    *,
    vae_micro_batch_size: int | None = 1,
    inference_cross_kv_reuse: bool = True,
) -> torch.nn.Module:
    """Apply ordinary runtime settings shared by training and evaluation."""
    if vae_micro_batch_size is not None:
        if isinstance(vae_micro_batch_size, bool) or int(vae_micro_batch_size) <= 0:
            raise ValueError("vae_micro_batch_size must be a positive integer or null.")
        vae_micro_batch_size = int(vae_micro_batch_size)
    vae = getattr(model, "vae", None)
    if vae is not None:
        setter = getattr(vae, "set_micro_batch_size", None)
        if callable(setter):
            setter(vae_micro_batch_size)
        else:
            vae.micro_batch_size = vae_micro_batch_size
    try:
        model.inference_cross_kv_reuse = bool(inference_cross_kv_reuse)
    except AttributeError:
        # Some lightweight test/proxy objects intentionally do not allow attributes.
        pass
    logger.info(
        "Model execution settings: vae_micro_batch_size=%s inference_cross_kv_reuse=%s",
        "full" if vae_micro_batch_size is None else vae_micro_batch_size,
        bool(inference_cross_kv_reuse),
    )
    return model


def configure_inference_compile(
    model: torch.nn.Module,
    enabled: bool = False,
    *,
    mode: str | None = None,
    backend: str | None = None,
    fullgraph: bool | None = None,
    dynamic: bool | None = None,
    options: Mapping[str, Any] | None = None,
) -> torch.nn.Module:
    """Compile the model-specific tensor-heavy inference functions on demand."""
    if not enabled:
        return model

    mode = _normalize_optional_string(mode, "mode")
    backend = _normalize_optional_string(backend, "backend")
    if fullgraph is not None and not isinstance(fullgraph, bool):
        raise TypeError(f"`fullgraph` must be bool or None, got {type(fullgraph)}.")
    if dynamic is not None and not isinstance(dynamic, bool):
        raise TypeError(f"`dynamic` must be bool or None, got {type(dynamic)}.")
    compile_options = dict(options or {})
    compile_config = (
        mode,
        backend,
        fullgraph,
        dynamic,
        tuple(sorted(compile_options.items())),
    )
    if bool(getattr(model, "_inference_compile_enabled", False)):
        previous_config = getattr(model, "_inference_compile_config", None)
        if previous_config != compile_config:
            raise RuntimeError(
                "Inference targets are already compiled with different torch.compile settings."
            )
        return model

    target_names: Iterable[str] = getattr(model, "inference_compile_targets", ())
    target_names = tuple(target_names)
    if not target_names:
        raise ValueError(
            f"{type(model).__name__} does not declare any inference compile targets."
        )

    compile_kwargs: dict[str, Any] = {}
    if mode is not None:
        compile_kwargs["mode"] = mode
    if backend is not None:
        compile_kwargs["backend"] = backend
    if fullgraph is not None:
        compile_kwargs["fullgraph"] = fullgraph
    if dynamic is not None:
        compile_kwargs["dynamic"] = dynamic
    if compile_options:
        compile_kwargs["options"] = compile_options

    for name in target_names:
        target = getattr(model, name, None)
        if not callable(target):
            raise AttributeError(
                f"Inference compile target `{name}` is not callable on "
                f"{type(model).__name__}."
            )
        setattr(model, name, torch.compile(target, **compile_kwargs))

    model._inference_compile_enabled = True
    model._inference_compile_config = compile_config
    logger.info(
        "Enabled torch.compile for %s inference targets: %s (mode=%s backend=%s "
        "fullgraph=%s dynamic=%s options=%s)",
        type(model).__name__,
        ", ".join(target_names),
        mode or "default",
        backend or "inductor",
        fullgraph if fullgraph is not None else False,
        dynamic if dynamic is not None else "auto",
        compile_options,
    )
    return model


def configure_inference_compile_from_config(
    model: torch.nn.Module,
    config: Mapping[str, Any],
) -> torch.nn.Module:
    """Configure inference compilation from an EVALUATION/inference config."""
    return configure_inference_compile(
        model,
        enabled=bool(config.get("torch_compile", False)),
        mode=config.get("torch_compile_mode"),
        backend=config.get("torch_compile_backend"),
        fullgraph=config.get("torch_compile_fullgraph"),
        dynamic=config.get("torch_compile_dynamic"),
        options=config.get("torch_compile_options"),
    )


def _normalize_optional_string(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"`{name}` must not be empty.")
    return normalized
