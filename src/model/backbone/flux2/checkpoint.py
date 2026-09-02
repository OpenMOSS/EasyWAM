from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def convert_imagewam_flux2_checkpoint_payload(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert an ImageWAM FLUX.2 checkpoint to the EasyWAM checkpoint contract."""
    if "mot" not in payload or not isinstance(payload["mot"], Mapping):
        raise ValueError("ImageWAM FLUX.2 checkpoint must contain a mapping at `mot`.")
    if "state_encoder" in payload and "proprio_encoder" in payload:
        raise ValueError(
            "Checkpoint contains both `state_encoder` and legacy `proprio_encoder`; "
            "refusing an ambiguous state-adapter migration."
        )

    converted = dict(payload)
    renamed_top_level: dict[str, str] = {}
    if "proprio_encoder" in converted:
        converted["state_encoder"] = converted.pop("proprio_encoder")
        renamed_top_level["proprio_encoder"] = "state_encoder"
    converted.setdefault("backbone_name", "flux2")
    converted.setdefault("model_variant", "mot")

    mot_state = dict(converted["mot"])
    if mot_state and all(key.startswith("module.") for key in mot_state):
        mot_state = {key[len("module.") :]: value for key, value in mot_state.items()}
        mot_prefix_removed = "module."
    elif any(key.startswith("module.") for key in mot_state):
        raise ValueError("Mixed prefixed/unprefixed ImageWAM MoT keys are ambiguous.")
    else:
        mot_prefix_removed = None
    converted["mot"] = mot_state
    converted["checkpoint_source"] = "imagewam_flux2"

    report = {
        "source_format": "imagewam_flux2",
        "renamed_top_level": renamed_top_level,
        "mot_prefix_removed": mot_prefix_removed,
        "source_mot_tensors": len(mot_state),
    }
    return converted, report


def audit_mot_state_dict(
    target: torch.nn.Module,
    source_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Return exact key/shape coverage before mutating the target module."""
    target_state = target.state_dict()
    source_keys = set(source_state)
    target_keys = set(target_state)
    common = source_keys & target_keys
    shape_mismatches = {
        key: {
            "source": tuple(source_state[key].shape),
            "target": tuple(target_state[key].shape),
        }
        for key in sorted(common)
        if tuple(source_state[key].shape) != tuple(target_state[key].shape)
    }
    matched = common - set(shape_mismatches)
    matched_numel = sum(int(target_state[key].numel()) for key in matched)
    target_numel = sum(int(value.numel()) for value in target_state.values())
    return {
        "source_tensors": len(source_keys),
        "target_tensors": len(target_keys),
        "matched_tensors": len(matched),
        "missing_keys": sorted(target_keys - source_keys),
        "unexpected_keys": sorted(source_keys - target_keys),
        "shape_mismatches": shape_mismatches,
        "matched_numel": matched_numel,
        "target_numel": target_numel,
        "numel_coverage": 1.0 if target_numel == 0 else matched_numel / target_numel,
    }


def require_exact_imagewam_coverage(report: Mapping[str, Any]) -> None:
    if (
        report["missing_keys"]
        or report["unexpected_keys"]
        or report["shape_mismatches"]
        or float(report["numel_coverage"]) != 1.0
    ):
        raise RuntimeError(
            "ImageWAM→EasyWAM FLUX.2 checkpoint is not exact: "
            f"missing={len(report['missing_keys'])}, "
            f"unexpected={len(report['unexpected_keys'])}, "
            f"shape_mismatches={len(report['shape_mismatches'])}, "
            f"numel_coverage={report['numel_coverage']:.8f}."
        )
