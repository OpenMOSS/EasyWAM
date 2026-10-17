"""Require a complete trained policy before starting a benchmark server.

Keep the shared model loaders' legacy compatibility outside this evaluation
boundary. Validate on CPU before loading or merging any checkpoint weights.
"""
from collections.abc import Mapping

import torch

from model.component.lora import has_lora, iter_lora_layers, set_video_dit_lora_trainable


def _check_state(expected, actual, label):
    if not isinstance(actual, Mapping):
        raise ValueError(f"Evaluation checkpoint requires complete {label} weights")
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatched = sorted(key for key in set(expected) & set(actual)
                        if not isinstance(actual[key], torch.Tensor)
                        or actual[key].shape != expected[key].shape)
    if missing or unexpected or mismatched:
        raise ValueError(f"Invalid evaluation checkpoint {label}: missing={missing[:5]}, "
                         f"unexpected={unexpected[:5]}, shape_mismatches={mismatched[:5]}")


def _standard_state(module):
    """Match full checkpoints even when the selected model has LoRA wrappers."""
    layers = [name for name, _ in iter_lora_layers(module)]
    expected = {}
    for key, tensor in module.state_dict().items():
        if any(key.startswith(name + ".lora_") for name in layers):
            continue
        for name in layers:
            prefix = name + ".base_layer."
            if key.startswith(prefix):
                key = name + "." + key[len(prefix):]
                break
        expected[key] = tensor
    return expected


def _lora_state(model):
    # Same trainable coverage as Trainer's DiT-only policy: all non-video DiT
    # parameters, active video adapters, and the state encoder. VAE/text/frozen
    # video-base weights come from the pretrained model, not the adapter file.
    video = model.video_dit
    set_video_dit_lora_trainable(video)
    trained = {id(parameter) for parameter in model.dit.parameters()}
    trained.difference_update(id(parameter) for parameter in video.parameters())
    trained.update(id(parameter) for parameter in video.parameters() if parameter.requires_grad)
    state_encoder = getattr(model, "state_encoder", None)
    if state_encoder is not None:
        trained.update(id(parameter) for parameter in state_encoder.parameters())
    return {key: parameter for key, parameter in model.named_parameters() if id(parameter) in trained}


def validate_evaluation_checkpoint(model, payload):
    if not isinstance(payload, Mapping):
        raise ValueError("Evaluation checkpoint must be an EasyWAM policy mapping")
    if payload.get("format") == "trainable_lora":
        roots = {name for name, module in model.named_modules()
                 if name and hasattr(module, "_easywam_lora_config")}
        saved_roots = payload.get("lora_roots")
        if (not has_lora(model.video_dit) or not isinstance(saved_roots, (list, tuple))
                or not all(isinstance(name, str) for name in saved_roots)
                or len(saved_roots) != len(set(saved_roots)) or set(saved_roots) != roots):
            raise ValueError("Evaluation checkpoint requires matching complete LoRA roots")
        _check_state(_lora_state(model), payload.get("trainable_state"), "trainable_state")
    elif hasattr(model, "mot"):
        if "proprio_encoder" in payload:
            if model.backbone_name != "flux2":
                raise ValueError("Legacy proprio_encoder is only supported for FLUX.2")
            from model.backbone.flux2.checkpoint import convert_imagewam_flux2_checkpoint_payload
            payload, _ = convert_imagewam_flux2_checkpoint_payload(payload)
        _check_state(_standard_state(model.mot), payload.get("mot"), "mot")
        if getattr(model, "state_encoder", None) is not None:
            _check_state(model.state_encoder.state_dict(), payload.get("state_encoder"), "state_encoder")
    else:
        # Hidden/Unified package the video, action and state modules in `dit`.
        _check_state(_standard_state(model.dit), payload.get("dit"), "dit")


def load_evaluation_checkpoint(model, path):
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    validate_evaluation_checkpoint(model, payload)
    del payload
    model.load_checkpoint(str(path), merge_lora=True)
