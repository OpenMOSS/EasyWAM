import torch

from ...component.action_dit import standard_initialization_payload


def _missing_value(key, target):
    if (
        key == "text_embedding.2.weight"
        and target.ndim == 2
        and target.shape[0] == target.shape[1]
    ):
        return torch.eye(target.shape[0], dtype=target.dtype)
    if key.endswith(".weight") and "norm3" in key:
        return torch.ones_like(target)
    return torch.zeros_like(target)


def initialize_action(action, visual, *, alpha=True):
    if action.preset != "standard":
        raise ValueError("Wan visual initialization requires ActionDiT preset=standard")
    return standard_initialization_payload(
        action,
        visual,
        source_key=lambda key: key,
        missing_value=_missing_value,
        source_backbone="wan22",
        mapping_version=1,
        alpha=alpha,
    )
