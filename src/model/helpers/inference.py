from __future__ import annotations

from collections.abc import Iterable

import torch

from utils.logging_config import get_logger


logger = get_logger(__name__)


def configure_inference_compile(
    model: torch.nn.Module,
    enabled: bool = False,
) -> torch.nn.Module:
    """Compile the model-specific tensor-heavy inference functions on demand."""
    if not enabled:
        return model
    if bool(getattr(model, "_inference_compile_enabled", False)):
        return model

    target_names: Iterable[str] = getattr(model, "inference_compile_targets", ())
    target_names = tuple(target_names)
    if not target_names:
        raise ValueError(
            f"{type(model).__name__} does not declare any inference compile targets."
        )

    for name in target_names:
        target = getattr(model, name, None)
        if not callable(target):
            raise AttributeError(
                f"Inference compile target `{name}` is not callable on "
                f"{type(model).__name__}."
            )
        setattr(model, name, torch.compile(target))

    model._inference_compile_enabled = True
    logger.info(
        "Enabled torch.compile for %s inference targets: %s",
        type(model).__name__,
        ", ".join(target_names),
    )
    return model
