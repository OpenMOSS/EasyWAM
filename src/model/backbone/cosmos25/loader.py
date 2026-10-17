from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional

import torch

from utils.logging_config import get_logger
from ..contract import BackboneComponents, TextEncoderComponents, TextEncoding
from .action_initialization import initialize_action
from .cosmos_video_dit import Cosmos25DiTConfig, Cosmos25VideoDiT
from .cosmos_video_text_encoder import Cosmos25TextEncoder


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
    return torch.load(path, map_location=map_location, mmap=True, weights_only=True)


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


def load_cosmos25_text_projection(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Module:
    state = _clean_dit_state_dict(_read_torch_checkpoint(Path(checkpoint_path)))
    weight = state["crossattn_proj.0.weight"]
    bias = state["crossattn_proj.0.bias"]
    if tuple(weight.shape) != (1024, ORIGINAL_REASON_CONTEXT_DIM):
        raise RuntimeError(
            "Unexpected Cosmos text projection shape: " f"{tuple(weight.shape)}."
        )
    with torch.device("meta"):
        linear = torch.nn.Linear(ORIGINAL_REASON_CONTEXT_DIM, 1024, bias=True)
    linear.load_state_dict({"weight": weight, "bias": bias}, strict=True, assign=True)
    projection = torch.nn.Sequential(linear, torch.nn.GELU())
    return projection.eval().requires_grad_(False).to(device=device, dtype=torch_dtype)


def load_cosmos25_dit(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    torch_dtype: torch.dtype = torch.bfloat16,
    attention_backend: str = "sdpa",
    use_gradient_checkpointing: bool = False,
    video_attention_mask_mode: str = "bidirectional",
    conditional_frame_timestep: float = 0.0001,
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
        conditional_frame_timestep=float(conditional_frame_timestep),
    )
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
    conditional_frame_timestep: float = 0.0001,
    skip_dit_load_from_pretrain: bool = False,
) -> Cosmos25Components:
    from .cosmos_video_vae import CosmosVideoVAE

    logger.info("Loading Cosmos-Predict2.5-2B components...")
    start = time.time()
    root = Path(model_id)
    dit_path = resolve_cosmos25_dit_path(root)
    vae_path = root / "tokenizer.pth"
    if not vae_path.is_file():
        raise FileNotFoundError(
            f"Cosmos video tokenizer checkpoint not found: {vae_path}"
        )
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
                conditional_frame_timestep=float(conditional_frame_timestep),
            )
        ).to(device=device, dtype=torch_dtype)
        resolved_dit_path = "SKIPPED_PRETRAIN"
    else:
        logger.info("Loading pretrained Cosmos video DiT from %s", dit_path)
        dit = load_cosmos25_dit(
            dit_path,
            device=device,
            torch_dtype=torch_dtype,
            attention_backend=attention_backend,
            use_gradient_checkpointing=use_gradient_checkpointing,
            video_attention_mask_mode=video_attention_mask_mode,
            conditional_frame_timestep=float(conditional_frame_timestep),
        )
        resolved_dit_path = str(dit_path)
        logger.info("Finished loading pretrained Cosmos video DiT.")
    text_encoder = None
    reason_path = None
    if load_text_encoder:
        reason_path = str(Path(reason_model_id))
        logger.info(
            "Loading Cosmos-Reason1 text encoder/tokenizer from %s", reason_path
        )
        text_encoder = Cosmos25TextEncoder.from_pretrained(
            reason_path, torch_dtype=torch_dtype, max_length=tokenizer_max_len
        ).to(device)
        text_encoder.set_projector(dit.project_text_context)
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


def _dit_options(cfg):
    return dict(
        attention_backend=str(cfg.get("attention_backend", "sdpa")),
        use_gradient_checkpointing=bool(cfg.get("use_gradient_checkpointing", False)),
        video_attention_mask_mode=str(
            cfg.get("video_attention_mask_mode", "bidirectional")
        ),
        conditional_frame_timestep=float(
            cfg.get("video_scheduler", {}).get("conditional_frame_timestep", 0.0001)
        ),
    )


def load_components(cfg, *, device, torch_dtype, skip_dit_load_from_pretrain=False):
    loaded = load_cosmos25_components(
        model_id=cfg["model_id"],
        reason_model_id=cfg["reason_model_id"],
        device=device,
        torch_dtype=torch_dtype,
        load_text_encoder=bool(cfg.get("load_text_encoder", False)),
        tokenizer_max_len=int(cfg.get("tokenizer_max_len", 128)),
        skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
        **_dit_options(cfg),
    )
    return BackboneComponents(
        name="cosmos25", text_dim=int(cfg["text_dim"]), **vars(loaded)
    )


def load_dit(cfg, *, device, torch_dtype, skip_dit_load_from_pretrain=False):
    if skip_dit_load_from_pretrain:
        return Cosmos25VideoDiT(Cosmos25DiTConfig(**_dit_options(cfg))).to(
            device=device, dtype=torch_dtype
        )
    return load_cosmos25_dit(
        resolve_cosmos25_dit_path(cfg["model_id"]),
        device=device,
        torch_dtype=torch_dtype,
        **_dit_options(cfg),
    )


def load_text_encoder(cfg, *, device, torch_dtype, context_len):
    model = (
        Cosmos25TextEncoder.from_pretrained(
            cfg["reason_model_id"],
            torch_dtype=torch_dtype,
            max_length=context_len,
        )
        .to(device)
        .eval()
    )
    model.set_projector(
        load_cosmos25_text_projection(
            resolve_cosmos25_dit_path(cfg["model_id"]),
            device=device,
            torch_dtype=torch_dtype,
        )
    )

    def encode(prompts):
        context, mask, token_mask = model(prompts, return_token_mask=True)
        return TextEncoding(context, mask.bool(), token_mask)

    return TextEncoderComponents(
        model, encode, 1, str(cfg["reason_model_id"]), str(cfg["reason_model_id"])
    )
