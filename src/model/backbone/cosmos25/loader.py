"""Strict local checkpoint loading for Cosmos-Predict2.5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional

import torch

from .cosmos_video_dit import Cosmos25DiTConfig, Cosmos25VideoDiT
from .cosmos_video_text_encoder import Cosmos25TextEncoder
from utils.logging_config import get_logger


logger = get_logger(__name__)


POST_TRAINED_RELATIVE_PATH = Path(
    "base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"
)
ORIGINAL_REASON_CONTEXT_DIM = 28 * 3584


@dataclass
class Cosmos25Components:
    dit: Cosmos25VideoDiT
    vae: torch.nn.Module
    text_encoder: Optional[Cosmos25TextEncoder]
    tokenizer: None
    dit_path: str
    vae_path: str
    text_encoder_path: Optional[str]
    tokenizer_path: Optional[str]


def resolve_cosmos25_dit_path(model_id: str | Path) -> Path:
    root = Path(model_id)
    if root.is_file():
        return root
    candidate = root / POST_TRAINED_RELATIVE_PATH
    if not candidate.is_file():
        raise FileNotFoundError(f"Cosmos DiT checkpoint not found: {candidate}")
    return candidate


def _read_torch_checkpoint(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, mmap=True, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location, weights_only=True)


def _clean_dit_state_dict(raw: dict) -> dict[str, torch.Tensor]:
    state = {}
    ignored = {
        "accum_video_sample_counter",
        "accum_image_sample_counter",
        "accum_iteration",
    }
    for key, value in raw.items():
        key = key.removeprefix("net.")
        if key.endswith("._extra_state") or key.startswith("accum_") or key in ignored:
            continue
        if isinstance(value, torch.Tensor):
            state[key] = value
    return state


def _collapse_reason_projection(
    state: dict[str, torch.Tensor],
    target_context_dim: int,
) -> None:
    """Fold the 28-layer Reason projection onto one 3584-wide hidden state."""
    key = "crossattn_proj.0.weight"
    weight = state.get(key)
    if weight is None or weight.shape[1] == target_context_dim:
        return
    if weight.shape[1] != ORIGINAL_REASON_CONTEXT_DIM or target_context_dim != 3584:
        raise RuntimeError(
            f"Cannot adapt {key} from shape {tuple(weight.shape)} to input width "
            f"{target_context_dim}."
        )
    state[key] = weight.reshape(weight.shape[0], 28, target_context_dim).sum(dim=1)


def load_cosmos25_dit(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    torch_dtype: torch.dtype = torch.bfloat16,
    attention_backend: str = "sdpa",
    use_gradient_checkpointing: bool = False,
    video_attention_mask_mode: str = "bidirectional",
) -> Cosmos25VideoDiT:
    path = Path(checkpoint_path)
    raw = _read_torch_checkpoint(path)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a state dict in {path}, got {type(raw)}.")
    state = _clean_dit_state_dict(raw)
    config = Cosmos25DiTConfig(
        attention_backend=attention_backend,
        use_gradient_checkpointing=use_gradient_checkpointing,
        video_attention_mask_mode=video_attention_mask_mode,
    )
    _collapse_reason_projection(state, config.reason_context_dim)
    with torch.device("meta"):
        model = Cosmos25VideoDiT(config)
    model.load_state_dict(state, strict=True, assign=True)
    model.to(device=device, dtype=torch_dtype)
    return model


def load_cosmos25_components(
    model_id: str | Path,
    reason_model_id: str | Path,
    device: str | torch.device = "cpu",
    torch_dtype: torch.dtype = torch.bfloat16,
    load_text_encoder: bool = True,
    tokenizer_max_len: int = 128,
    attention_backend: str = "sdpa",
    use_gradient_checkpointing: bool = False,
    video_attention_mask_mode: str = "bidirectional",
    skip_dit_load_from_pretrain: bool = False,
) -> Cosmos25Components:
    from .cosmos_video_vae import CosmosVideoVAE

    logger.info("Loading Cosmos-Predict2.5-2B components...")
    start = time.time()
    root = Path(model_id)
    dit_path = resolve_cosmos25_dit_path(root)
    vae_path = root / "tokenizer.pth"
    if not vae_path.is_file():
        raise FileNotFoundError(f"Cosmos video tokenizer checkpoint not found: {vae_path}")
    if skip_dit_load_from_pretrain:
        logger.info(
            "Skipping pretrained video DiT load (`skip_dit_load_from_pretrain=True`); "
            "initializing video expert randomly and expecting checkpoint override."
        )
        dit = Cosmos25VideoDiT(
            Cosmos25DiTConfig(
                attention_backend=attention_backend,
                use_gradient_checkpointing=use_gradient_checkpointing,
                video_attention_mask_mode=video_attention_mask_mode,
            )
        ).to(device=device, dtype=torch_dtype)
        resolved_dit_path = "SKIPPED_PRETRAIN"
    else:
        logger.info("Loading pretrained Cosmos video DiT from %s", dit_path)
        dit = load_cosmos25_dit(
            dit_path, device=device, torch_dtype=torch_dtype,
            attention_backend=attention_backend,
            use_gradient_checkpointing=use_gradient_checkpointing,
            video_attention_mask_mode=video_attention_mask_mode,
        )
        resolved_dit_path = str(dit_path)
        logger.info("Finished loading pretrained Cosmos video DiT.")
    text_encoder = None
    reason_path = None
    if load_text_encoder:
        reason_path = str(Path(reason_model_id))
        logger.info("Loading Cosmos-Reason1 text encoder/tokenizer from %s", reason_path)
        text_encoder = Cosmos25TextEncoder.from_pretrained(
            reason_path, torch_dtype=torch_dtype, max_length=tokenizer_max_len
        ).to(device)
        logger.info("Finished loading Cosmos-Reason1 text encoder/tokenizer.")
    else:
        logger.info(
            "Skipping pretrained text encoder/tokenizer load (`load_text_encoder=False`); "
            "training must provide cached `context/context_mask`."
        )
    logger.info("Loading Cosmos video VAE/tokenizer from %s", vae_path)
    vae = CosmosVideoVAE.from_pretrained(
        vae_path,
        device=device,
        torch_dtype=torch_dtype,
    )
    logger.info("Finished loading Cosmos video VAE/tokenizer.")
    logger.info(
        "Finished loading Cosmos-Predict2.5-2B components in %.2f seconds.",
        time.time() - start,
    )
    return Cosmos25Components(
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=None,
        dit_path=resolved_dit_path,
        vae_path=str(vae_path),
        text_encoder_path=reason_path,
        tokenizer_path=reason_path if load_text_encoder else None,
    )
