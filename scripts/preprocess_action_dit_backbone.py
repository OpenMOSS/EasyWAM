"""Prepare ActionDiT initialization using the selected backbone's weight mapping."""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from model.backbone import (
    initialize_action_from_backbone,
    load_backbone_dit,
    normalize_backbone_config,
)
from model.backbone.registry import BACKBONE_MODULES
from model.component.action_dit import ActionDiT


def _parse_dtype(name):
    types = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return types[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def _parse_bool(value):
    if str(value).lower() in {"1", "true", "yes", "y"}:
        return True
    if str(value).lower() in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _load_model_config(path, backbone_name=None):
    def read_recipe(recipe, stack=()):
        resolved = recipe.resolve()
        if resolved in stack:
            raise ValueError(f"Recursive model recipe inheritance: {recipe}")
        current = OmegaConf.load(recipe)
        defaults = current.pop("defaults", [])
        merged = OmegaConf.create({})
        for entry in defaults:
            if entry == "_self_":
                merged = OmegaConf.merge(merged, current)
            elif isinstance(entry, str):
                merged = OmegaConf.merge(
                    merged,
                    read_recipe(recipe.parent / (entry + ".yaml"), (*stack, resolved)),
                )
            else:
                for group, name in entry.items():
                    if name is not None:
                        merged = OmegaConf.merge(
                            merged,
                            {
                                group: OmegaConf.load(
                                    recipe.parent / group / (name + ".yaml")
                                )
                            },
                        )
        if "_self_" not in defaults:
            merged = OmegaConf.merge(merged, current)
        return merged

    cfg = read_recipe(Path(path))
    if "backbone" not in cfg or "action_dit_config" not in cfg:
        raise ValueError(
            "Model recipe must select a backbone and define action_dit_config."
        )
    root = OmegaConf.create(
        {
            "model": cfg,
            "data": {
                "train": {
                    "processor": {
                        "action_output_dim": 7,
                        "proprio_output_dim": 7,
                    }
                }
            },
        }
    )
    backbone = normalize_backbone_config(root.model.backbone)
    if backbone_name is not None and backbone_name != backbone["name"]:
        raise ValueError(
            "--backbone must match the backbone selected by --model-config"
        )
    action = OmegaConf.to_container(root.model.action_dit_config, resolve=True)
    return backbone, action


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ActionDiT weights from the model recipe's visual backbone."
    )
    parser.add_argument(
        "--model-config", required=True, help="Path to the model YAML recipe."
    )
    parser.add_argument(
        "--output", required=True, help="Output initialization .pt path."
    )
    parser.add_argument(
        "--backbone",
        choices=tuple(BACKBONE_MODULES),
        help="Optional check of the recipe's backbone identity.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", default="float32", choices=("float32", "float16", "bfloat16")
    )
    parser.add_argument(
        "--apply-alpha-scaling",
        default="true",
        help="Scale resized input dimensions by sqrt(source_dim / target_dim); default: true.",
    )
    args = parser.parse_args()
    torch.manual_seed(42)
    backbone, action_cfg = _load_model_config(Path(args.model_config), args.backbone)
    dtype = _parse_dtype(args.dtype)
    alpha = _parse_bool(args.apply_alpha_scaling)
    visual = load_backbone_dit(backbone, device=args.device, torch_dtype=dtype)
    action = ActionDiT(**action_cfg).to(device=args.device, dtype=dtype)
    payload = initialize_action_from_backbone(backbone, action, visual, alpha=alpha)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    report = payload.get("initialization", {})
    counts = {
        key: len(report[key]) if isinstance(report[key], list) else report[key]
        for key in ("copied", "resized", "initialized")
        if key in report
    }
    print(
        f"Saved ActionDiT initialization to {output}: backbone={backbone['name']}, alpha_scaling={alpha}, counts={counts}"
    )


if __name__ == "__main__":
    main()
