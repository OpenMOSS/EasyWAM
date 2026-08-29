import hashlib
from pathlib import Path
from typing import Any

import torch


TEXT_EMBEDDING_CACHE_VERSION = 2
DEFAULT_TEXT_ENCODER_ID = "wan22ti2v5b"


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def text_embedding_cache_filename(
    prompt_or_hash: str,
    context_len: int,
    encoder_id: str,
    *,
    is_hash: bool = False,
) -> str:
    hashed = prompt_or_hash if is_hash else prompt_hash(prompt_or_hash)
    return f"{hashed}.text_len{int(context_len)}.{encoder_id}.pt"


def build_text_embedding_payload(
    *,
    context: torch.Tensor,
    mask: torch.Tensor,
    context_len: int,
    encoder_id: str,
    prompt_digest: str,
) -> dict[str, Any]:
    payload = {
        "format_version": TEXT_EMBEDDING_CACHE_VERSION,
        "encoder_id": str(encoder_id),
        "context_len": int(context_len),
        "prompt_hash": str(prompt_digest),
        "context": context.to(device="cpu", dtype=torch.bfloat16).contiguous(),
        "mask": mask.to(device="cpu", dtype=torch.bool).contiguous(),
    }
    validate_text_embedding_payload(
        payload,
        expected_context_len=context_len,
        expected_encoder_id=encoder_id,
        expected_prompt_hash=prompt_digest,
    )
    return payload


def validate_text_embedding_payload(
    payload: Any,
    *,
    expected_context_len: int,
    expected_encoder_id: str,
    expected_prompt_hash: str | None = None,
) -> None:
    if not isinstance(payload, dict):
        raise TypeError(
            f"Text embedding cache must contain a dict, got {type(payload).__name__}."
        )
    required = {"context", "mask"}
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"Text embedding cache is missing fields: {sorted(missing)}.")
    if "format_version" in payload and int(payload["format_version"]) != TEXT_EMBEDDING_CACHE_VERSION:
        raise ValueError(
            "Text embedding cache format version mismatch: "
            f"expected {TEXT_EMBEDDING_CACHE_VERSION}, got {payload['format_version']}."
        )
    if "encoder_id" in payload and str(payload["encoder_id"]) != str(expected_encoder_id):
        raise ValueError(
            "Text embedding cache encoder mismatch: "
            f"expected {expected_encoder_id}, got {payload['encoder_id']}."
        )
    if "context_len" in payload and int(payload["context_len"]) != int(expected_context_len):
        raise ValueError(
            "Text embedding cache context length mismatch: "
            f"expected {expected_context_len}, got {payload['context_len']}."
        )
    if (
        expected_prompt_hash is not None
        and "prompt_hash" in payload
        and str(payload["prompt_hash"]) != str(expected_prompt_hash)
    ):
        raise ValueError(
            "Text embedding cache prompt hash mismatch: "
            f"expected {expected_prompt_hash}, got {payload['prompt_hash']}."
        )

    context = payload["context"]
    mask = payload["mask"]
    if not isinstance(context, torch.Tensor) or context.ndim != 2:
        raise ValueError(
            f"`context` must be a [L,D] tensor, got {type(context)} "
            f"with shape={getattr(context, 'shape', None)}."
        )
    if not isinstance(mask, torch.Tensor) or mask.ndim != 1:
        raise ValueError(
            f"`mask` must be a [L] tensor, got {type(mask)} "
            f"with shape={getattr(mask, 'shape', None)}."
        )
    if context.shape[0] != expected_context_len or mask.shape[0] != expected_context_len:
        raise ValueError(
            "Text embedding cache sequence length mismatch: "
            f"context={context.shape[0]}, mask={mask.shape[0]}, "
            f"expected={expected_context_len}."
        )
    if context.dtype != torch.bfloat16:
        raise TypeError(f"`context` must use bfloat16, got {context.dtype}.")
    if mask.dtype != torch.bool:
        raise TypeError(f"`mask` must use bool, got {mask.dtype}.")


def load_text_embedding_cache(
    cache_path: str | Path,
    context_len: int,
    encoder_id: str,
    prompt_digest: str | None = None,
) -> dict[str, Any]:
    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing text embedding cache: {path}. "
            "Run scripts/precompute_text_embeds.py for the selected task."
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_text_embedding_payload(
        payload,
        expected_context_len=context_len,
        expected_encoder_id=encoder_id,
        expected_prompt_hash=prompt_digest,
    )
    return payload
