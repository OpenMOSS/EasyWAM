"""Native PyTorch Cosmos-Predict2.5 2B diffusion transformer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ...component.attention import (
    AttentionSegment,
    StructuredAttentionMask,
    build_structured_attention_mask,
    normalize_attention_backend,
    run_attention,
)
from utils.logging_config import get_logger


logger = get_logger(__name__)


class CosmosTimestepEmbedding(nn.Module):
    def __init__(self, hidden_size: int, adaln_lora_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

    def forward(self, sample: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        modulation = self.linear_2(self.activation(self.linear_1(sample)))
        return sample, modulation


class CosmosPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, patch_size: tuple[int, int, int]):
        super().__init__()
        self.patch_size = patch_size
        patch_volume = math.prod(patch_size)
        self.proj = nn.Sequential(nn.Identity(), nn.Linear(in_channels * patch_volume, hidden_size, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pt, ph, pw = self.patch_size
        b, c, t, h, w = x.shape
        if t % pt or h % ph or w % pw:
            raise ValueError(f"Input {(t, h, w)} must be divisible by patch size {self.patch_size}.")
        x = x.reshape(b, c, t // pt, pt, h // ph, ph, w // pw, pw)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(b, t // pt, h // ph, w // pw, -1)
        return self.proj(x)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class CosmosVideoRope3D(nn.Module):
    """Split-half 3D rotary embedding used by the 2B checkpoint."""

    def __init__(self, head_dim: int = 128, rope_scale: tuple[float, float, float] = (1.0, 3.0, 3.0)):
        super().__init__()
        if head_dim != 128:
            raise ValueError("Cosmos-Predict2.5-2B uses a 128-dimensional attention head.")
        self.head_dim = head_dim
        dim_t, dim_h, dim_w = 44, 42, 42
        self.theta_t = 10000.0 * rope_scale[0] ** (dim_t / (dim_t - 2))
        self.theta_h = 10000.0 * rope_scale[1] ** (dim_h / (dim_h - 2))
        self.theta_w = 10000.0 * rope_scale[2] ** (dim_w / (dim_w - 2))
        # The checkpoint stores these ranges as non-learned compatibility buffers.
        self.register_buffer("seq", torch.arange(head_dim, dtype=torch.float32), persistent=True)
        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2, dtype=torch.float32) / dim_h,
            persistent=True,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2, dtype=torch.float32) / dim_t,
            persistent=True,
        )

    def _axis_angles(
        self,
        length: int,
        dimension_range: torch.Tensor,
        theta: float,
        device: torch.device,
    ) -> torch.Tensor:
        positions = self.seq[:length].to(device=device, dtype=torch.float32)
        frequencies = theta ** (-dimension_range.to(device=device, dtype=torch.float32))
        return positions[:, None] * frequencies[None, :]

    def forward(self, t: int, h: int, w: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        # 44 temporal + 42 height + 42 width channels = a 128-D head.
        angles_t = self._axis_angles(t, self.dim_temporal_range, self.theta_t, device)[:, None, None, :].expand(t, h, w, -1)
        angles_h = self._axis_angles(h, self.dim_spatial_range, self.theta_h, device)[None, :, None, :].expand(t, h, w, -1)
        angles_w = self._axis_angles(w, self.dim_spatial_range, self.theta_w, device)[None, None, :, :].expand(t, h, w, -1)
        half_angles = torch.cat((angles_t, angles_h, angles_w), dim=-1).reshape(t * h * w, 64)
        angles = torch.cat((half_angles, half_angles), dim=-1)
        return angles.cos(), angles.sin()


class CosmosAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: Optional[int], num_heads: int, backend: str):
        super().__init__()
        self.is_selfattn = context_dim is None
        context_dim = query_dim if context_dim is None else context_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.backend = normalize_attention_backend(backend)
        self.q_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.output_proj = nn.Linear(query_dim, query_dim, bias=False)

    def _norm_heads(self, x: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        b, s, _ = x.shape
        return norm(x.reshape(b, s, self.num_heads, self.head_dim)).reshape(b, s, -1)

    def apply_rope(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        b, s, _ = x.shape
        x_heads = x.reshape(b, s, self.num_heads, self.head_dim)
        cos, sin = rope
        cos = cos.to(device=x.device, dtype=x.dtype)[None, :, None, :]
        sin = sin.to(device=x.device, dtype=x.dtype)[None, :, None, :]
        rotated = x_heads * cos + _rotate_half(x_heads) * sin
        return rotated.reshape(b, s, -1)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        rope: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor | StructuredAttentionMask] = None,
    ) -> torch.Tensor:
        source = x if context is None else context
        q = self._norm_heads(self.q_proj(x), self.q_norm)
        k = self._norm_heads(self.k_proj(source), self.k_norm)
        v = self.v_proj(source)
        if self.is_selfattn and rope is not None:
            q = self.apply_rope(q, rope)
            k = self.apply_rope(k, rope)
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]
        result = run_attention(q, k, v, self.num_heads, attention_mask=attention_mask, backend=self.backend)
        return self.output_proj(result)


class CosmosFeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.activation = nn.GELU()
        self.layer2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.activation(self.layer1(x)))


def _adaln_layer(hidden_size: int, rank: int) -> nn.Sequential:
    return nn.Sequential(
        nn.SiLU(),
        nn.Linear(hidden_size, rank, bias=False),
        nn.Linear(rank, 3 * hidden_size, bias=False),
    )


class CosmosTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, context_dim: int, num_heads: int, intermediate_size: int, rank: int, backend: str):
        super().__init__()
        self.layer_norm_self_attn = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = CosmosAttention(hidden_size, None, num_heads, backend)
        self.layer_norm_cross_attn = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CosmosAttention(hidden_size, context_dim, num_heads, backend)
        self.layer_norm_mlp = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = CosmosFeedForward(hidden_size, intermediate_size)
        self.adaln_modulation_self_attn = _adaln_layer(hidden_size, rank)
        self.adaln_modulation_cross_attn = _adaln_layer(hidden_size, rank)
        self.adaln_modulation_mlp = _adaln_layer(hidden_size, rank)

    @staticmethod
    def _modulate(
        x: torch.Tensor,
        norm: nn.Module,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return norm(x) * (1 + scale) + shift

    @staticmethod
    def _prepare_token_layout(
        x: torch.Tensor,
        modulation: torch.Tensor,
        modulation_indices: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
        """Align frame-level modulation with flattened or heterogeneous tokens."""
        if modulation_indices is None:
            steps = modulation.shape[1]
            if not steps or x.shape[1] % steps:
                raise ValueError(
                    "Token count must be divisible by compact timestep count: "
                    f"tokens={x.shape[1]}, timesteps={steps}."
                )
            tokens_per_step = x.shape[1] // steps
            laid_out = x.reshape(x.shape[0], steps, tokens_per_step, x.shape[-1])
            return laid_out, lambda value: value[:, :, None, :]

        indices = modulation_indices.to(device=x.device, dtype=torch.long)
        if indices.ndim != 1 or indices.numel() != x.shape[1]:
            raise ValueError(
                "modulation_indices must be [S] and aligned with tokens, got "
                f"{tuple(indices.shape)} for S={x.shape[1]}."
            )
        if indices.numel() and (indices.min() < 0 or indices.max() >= modulation.shape[1]):
            raise ValueError("modulation_indices contains an out-of-range timestep index.")
        return x, lambda value: value.index_select(1, indices)

    def prepare_mixed_attention(
        self,
        x: torch.Tensor,
        timestep_embedding: torch.Tensor,
        adaln_lora: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        modulation_indices: Optional[torch.Tensor] = None,
    ):
        if timestep_embedding.shape[:2] != adaln_lora.shape[:2]:
            raise ValueError("Cosmos timestep and AdaLN-LoRA tensors must share [B,T].")
        x_layout, align_modulation = self._prepare_token_layout(
            x, timestep_embedding, modulation_indices
        )
        self_mod = (
            self.adaln_modulation_self_attn(timestep_embedding) + adaln_lora
        ).chunk(3, dim=-1)
        normalized = self._modulate(
            x_layout,
            self.layer_norm_self_attn,
            align_modulation(self_mod[0]),
            align_modulation(self_mod[1]),
        )
        normalized = normalized.reshape(x.shape)
        attn = self.self_attn
        q = attn._norm_heads(attn.q_proj(normalized), attn.q_norm)
        k = attn._norm_heads(attn.k_proj(normalized), attn.k_norm)
        v = attn.v_proj(normalized)
        q = attn.apply_rope(q, rope)
        k = attn.apply_rope(k, rope)
        return q, k, v, {
            "x": x_layout,
            "flat_shape": x.shape,
            "align_modulation": align_modulation,
            "self_mod": self_mod,
            "t": timestep_embedding,
            "adaln": adaln_lora,
        }

    def finish_mixed_attention(
        self,
        mixed_attention: torch.Tensor,
        state: dict,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = state["x"]
        flat_shape = state["flat_shape"]
        align_modulation = state["align_modulation"]
        self_mod = state["self_mod"]
        mixed_attention = self.self_attn.output_proj(mixed_attention).reshape(x.shape)
        x = torch.addcmul(
            x,
            align_modulation(self_mod[2]).to(x.dtype),
            mixed_attention.to(x.dtype),
        )
        t = state["t"]
        adaln = state["adaln"]
        cross_mod = (self.adaln_modulation_cross_attn(t) + adaln).chunk(3, dim=-1)
        normalized = self._modulate(
            x,
            self.layer_norm_cross_attn,
            align_modulation(cross_mod[0]),
            align_modulation(cross_mod[1]),
        )
        attended = self.cross_attn(
            normalized.reshape(flat_shape),
            context=context,
            attention_mask=context_mask,
        ).reshape(x.shape)
        x = torch.addcmul(
            x, align_modulation(cross_mod[2]).to(x.dtype), attended.to(x.dtype)
        )
        mlp_mod = (self.adaln_modulation_mlp(t) + adaln).chunk(3, dim=-1)
        normalized = self._modulate(
            x,
            self.layer_norm_mlp,
            align_modulation(mlp_mod[0]),
            align_modulation(mlp_mod[1]),
        )
        x = torch.addcmul(
            x,
            align_modulation(mlp_mod[2]).to(x.dtype),
            self.mlp(normalized).to(x.dtype),
        )
        return x.reshape(flat_shape)

    def forward_tokens(
        self,
        x: torch.Tensor,
        timestep_embedding: torch.Tensor,
        context: torch.Tensor,
        adaln_lora: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        context_mask: Optional[torch.Tensor],
        self_attn_mask: Optional[torch.Tensor | StructuredAttentionMask] = None,
        modulation_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, k, v, state = self.prepare_mixed_attention(
            x, timestep_embedding, adaln_lora, rope, modulation_indices
        )
        attended = run_attention(
            q, k, v, self.self_attn.num_heads,
            attention_mask=self_attn_mask,
            backend=self.self_attn.backend,
        )
        return self.finish_mixed_attention(attended, state, context, context_mask)

    def forward(
        self,
        x: torch.Tensor,
        timestep_embedding: torch.Tensor,
        context: torch.Tensor,
        adaln_lora: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, t, h, w, d = x.shape
        tokens = self.forward_tokens(
            x.reshape(b, t * h * w, d), timestep_embedding, context, adaln_lora,
            rope, context_mask,
        )
        return tokens.reshape(b, t, h, w, d)


class CosmosFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, patch_size: tuple[int, int, int], rank: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels * math.prod(patch_size), bias=False)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, rank, bias=False), nn.Linear(rank, 2 * hidden_size, bias=False)
        )

    def condition_tokens(
        self,
        x: torch.Tensor,
        embedding: torch.Tensor,
        adaln_lora: torch.Tensor,
    ) -> torch.Tensor:
        """Apply native Cosmos output normalization/modulation without projection."""
        shift, scale = (self.adaln_modulation(embedding) + adaln_lora[..., : 2 * x.shape[-1]]).chunk(2, -1)
        for _ in range(x.ndim - embedding.ndim):
            shift = shift.unsqueeze(2)
            scale = scale.unsqueeze(2)
        return self.layer_norm(x) * (1 + scale) + shift

    def forward(self, x: torch.Tensor, embedding: torch.Tensor, adaln_lora: torch.Tensor) -> torch.Tensor:
        return self.linear(self.condition_tokens(x, embedding, adaln_lora))


@dataclass(frozen=True)
class Cosmos25DiTConfig:
    in_channels: int = 18
    out_channels: int = 16
    hidden_size: int = 2048
    context_dim: int = 1024
    reason_context_dim: int = 28 * 3584
    num_layers: int = 28
    num_heads: int = 16
    intermediate_size: int = 8192
    adaln_lora_dim: int = 256
    patch_size: tuple[int, int, int] = (1, 2, 2)
    attention_backend: str = "sdpa"
    use_gradient_checkpointing: bool = False
    video_attention_mask_mode: str = "bidirectional"
    conditional_frame_timestep: float = 0.0001


class Cosmos25VideoDiT(nn.Module):
    """Cosmos-Predict2.5-2B Image2World DiT with native checkpoint names."""

    def __init__(self, config: Cosmos25DiTConfig = Cosmos25DiTConfig()):
        super().__init__()
        self.config = config
        self.x_embedder = CosmosPatchEmbed(config.in_channels, config.hidden_size, config.patch_size)
        self.pos_embedder = CosmosVideoRope3D(config.hidden_size // config.num_heads)
        self.t_embedder = nn.Sequential(nn.Identity(), CosmosTimestepEmbedding(config.hidden_size, config.adaln_lora_dim))
        self.blocks = nn.ModuleList([
            CosmosTransformerBlock(
                config.hidden_size, config.context_dim, config.num_heads, config.intermediate_size,
                config.adaln_lora_dim, config.attention_backend,
            ) for _ in range(config.num_layers)
        ])
        self.final_layer = CosmosFinalLayer(config.hidden_size, config.out_channels, config.patch_size, config.adaln_lora_dim)
        self.t_embedding_norm = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.crossattn_proj = nn.Sequential(
            nn.Linear(config.reason_context_dim, config.context_dim, bias=True),
            nn.GELU(),
        )
        # EasyWAM backbone protocol metadata.
        self.backbone_name = "cosmos25"
        self.hidden_dim = config.hidden_size
        self.freq_dim = config.hidden_size
        self.text_dim = config.context_dim
        self.num_heads = config.num_heads
        self.attn_head_dim = config.hidden_size // config.num_heads
        self.patch_size = config.patch_size
        self.attention_backend = normalize_attention_backend(config.attention_backend)
        self.use_gradient_checkpointing = bool(config.use_gradient_checkpointing)
        if self.use_gradient_checkpointing:
            logger.info(
                "Using gradient checkpointing for DiT blocks. "
                "This will save memory but use more computation."
            )
        self.video_attention_mask_mode = str(config.video_attention_mask_mode)
        if self.video_attention_mask_mode not in {
            "bidirectional", "first_frame_causal", "per_frame_causal"
        }:
            raise ValueError(
                "video_attention_mask_mode must be 'bidirectional', "
                "'first_frame_causal', or 'per_frame_causal'."
            )
        self.fuse_vae_embedding_in_latents = True

    def project_text_context(self, context: torch.Tensor) -> torch.Tensor:
        if context.shape[-1] == self.config.context_dim:
            return context
        if context.shape[-1] != self.config.reason_context_dim:
            raise ValueError(
                "Cosmos context must be raw Reason1 features or preprojected features; "
                f"got width {context.shape[-1]}."
            )
        return self.crossattn_proj(context.to(dtype=self.crossattn_proj[0].weight.dtype))

    @staticmethod
    def _timestep_features(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        if timesteps.ndim == 1:
            timesteps = timesteps[:, None]
        half = dim // 2
        exponent = -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
        phases = timesteps.float().reshape(-1, 1) * exponent.exp()[None]
        embedding = torch.cat((phases.cos(), phases.sin()), dim=-1)
        return embedding.reshape(timesteps.shape[0], timesteps.shape[1], dim)

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        pt, ph, pw = self.config.patch_size
        b, t, h, w, _ = x.shape
        x = x.reshape(b, t, h, w, ph, pw, pt, self.config.out_channels)
        return x.permute(0, 7, 1, 6, 2, 4, 3, 5).reshape(b, self.config.out_channels, t * pt, h * ph, w * pw)

    def build_structured_video_attention_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> StructuredAttentionMask:
        mode = self.video_attention_mask_mode
        segments = []
        if mode == "bidirectional":
            segments.append(AttentionSegment(0, video_seq_len, ((0, video_seq_len),)))
        elif mode == "first_frame_causal":
            first = min(video_tokens_per_frame, video_seq_len)
            segments.append(AttentionSegment(0, first, ((0, first),)))
            if first < video_seq_len:
                segments.append(AttentionSegment(first, video_seq_len, ((0, video_seq_len),)))
        elif mode == "per_frame_causal":
            if video_seq_len % video_tokens_per_frame:
                raise ValueError("video_seq_len must be divisible by video_tokens_per_frame")
            for start in range(0, video_seq_len, video_tokens_per_frame):
                end = start + video_tokens_per_frame
                segments.append(AttentionSegment(start, end, ((0, end),)))
        else:
            raise ValueError(f"Unsupported video attention mask mode: {mode}")
        return build_structured_attention_mask(video_seq_len, video_seq_len, segments, device)

    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = True,
        **_: object,
    ) -> dict:
        if x.ndim != 5 or x.shape[1] != self.config.in_channels - 2:
            raise ValueError(f"x must be [B,16,T,H,W], got {tuple(x.shape)}")
        b, _, frames, height, width = x.shape
        if timestep.ndim == 1:
            timestep = timestep[:, None].expand(b, frames).clone()
        elif timestep.shape != (b, frames):
            raise ValueError(f"timestep must be [B] or [B,T], got {tuple(timestep.shape)}")
        condition_mask = torch.zeros((b, 1, frames, height, width), device=x.device, dtype=x.dtype)
        if fuse_vae_embedding_in_latents:
            condition_mask[:, :, :1] = 1
            timestep[:, 0] = self.config.conditional_frame_timestep
        padding_mask = torch.zeros_like(condition_mask)
        hidden_5d = self.x_embedder(torch.cat((x, condition_mask, padding_mask), dim=1))
        _, tf, ph, pw, dim = hidden_5d.shape
        tokens = hidden_5d.reshape(b, tf * ph * pw, dim)
        context = self.project_text_context(context).to(dtype=tokens.dtype)
        # Native Cosmos cross-attention is mask-free, which also preserves FA4.
        del context_mask
        context_mask = None
        features = self._timestep_features(timestep, self.config.hidden_size).to(tokens.dtype)
        frame_t, frame_adaln = self.t_embedder[1](features)
        frame_t = self.t_embedding_norm(frame_t)
        rope = self.pos_embedder(tf, ph, pw, tokens.device)
        return {
            "tokens": tokens,
            "freqs": rope,
            "t": frame_t,
            "t_mod": {"embedding": frame_t, "adaln_lora": frame_adaln},
            "context": context,
            "context_mask": context_mask,
            "meta": {"grid_size": (tf, ph, pw), "tokens_per_frame": ph * pw, "batch_size": b,
                     "frame_adaln": frame_adaln},
        }

    def forward_block(
        self,
        layer_index: int,
        tokens: torch.Tensor,
        pre_state: dict,
        self_attn_mask: Optional[torch.Tensor | StructuredAttentionMask] = None,
    ) -> torch.Tensor:
        mod = pre_state["t_mod"]
        return self.blocks[layer_index].forward_tokens(
            tokens, mod["embedding"], pre_state["context"], mod["adaln_lora"],
            pre_state["freqs"], pre_state["context_mask"], self_attn_mask,
            mod.get("token_to_timestep"),
        )

    def post_dit(self, tokens: torch.Tensor, pre_state: dict) -> torch.Tensor:
        tf, ph, pw = pre_state["meta"]["grid_size"]
        hidden = tokens.reshape(tokens.shape[0], tf, ph, pw, tokens.shape[-1])
        output = self.final_layer(
            hidden.to(pre_state["context"].dtype),
            pre_state["t"],
            pre_state["meta"]["frame_adaln"],
        )
        return self._unpatchify(output)

    def _aux_rope(self, length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        temporal = self.pos_embedder._axis_angles(
            length,
            self.pos_embedder.dim_temporal_range,
            self.pos_embedder.theta_t,
            device,
        )
        half = torch.cat((temporal, temporal.new_zeros((length, 42))), dim=-1)
        angles = torch.cat((half, half), dim=-1)
        return angles.cos(), angles.sin()

    def pre_unified_dit(
        self,
        x: torch.Tensor,
        timestep_video: torch.Tensor,
        action_tokens: torch.Tensor,
        timestep_action: torch.Tensor,
        state_tokens: torch.Tensor,
        timestep_state: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> dict:
        video = self.pre_dit(x, timestep_video, context, context_mask, True)
        video_len = video["tokens"].shape[1]
        parts = [video["tokens"], action_tokens, state_tokens]
        tokens = torch.cat(parts, dim=1)
        embeddings = [video["t_mod"]["embedding"]]
        adaln_parts = [video["t_mod"]["adaln_lora"]]
        video_steps = video["t_mod"]["embedding"].shape[1]
        modulation_indices = [
            torch.arange(video_steps, device=tokens.device).repeat_interleave(
                video["meta"]["tokens_per_frame"]
            )
        ]
        rope_cos, rope_sin = video["freqs"]
        cos_parts, sin_parts = [rope_cos], [rope_sin]
        for aux, timestep in ((action_tokens, timestep_action), (state_tokens, timestep_state)):
            features = self._timestep_features(timestep[:, None], self.config.hidden_size).to(tokens.dtype)
            emb, adaln = self.t_embedder[1](features)
            emb = self.t_embedding_norm(emb)
            embeddings.append(emb)
            adaln_parts.append(adaln)
            modulation_indices.append(
                torch.full(
                    (aux.shape[1],),
                    sum(part.shape[1] for part in embeddings[:-1]),
                    device=tokens.device,
                    dtype=torch.long,
                )
            )
            cos, sin = self._aux_rope(aux.shape[1], tokens.device)
            cos_parts.append(cos)
            sin_parts.append(sin)
        projected_context = video["context"]
        if video["context_mask"] is None:
            non_state_len = tokens.shape[1] - state_tokens.shape[1]
            segments = [AttentionSegment(0, non_state_len, ((0, context.shape[1]),))]
            if state_tokens.shape[1]:
                segments.append(AttentionSegment(non_state_len, tokens.shape[1], ()))
            joint_context_mask = (
                None
                if not state_tokens.shape[1]
                else build_structured_attention_mask(
                    tokens.shape[1], context.shape[1], segments, tokens.device
                )
            )
        else:
            joint_context_mask = context_mask.to(torch.bool)[:, None].expand(-1, tokens.shape[1], -1).clone()
            if state_tokens.shape[1]:
                joint_context_mask[:, -state_tokens.shape[1]:] = False
        video["tokens"] = tokens
        video["t_mod"] = {
            "embedding": torch.cat(embeddings, dim=1),
            "adaln_lora": torch.cat(adaln_parts, dim=1),
            "token_to_timestep": torch.cat(modulation_indices),
        }
        video["freqs"] = (torch.cat(cos_parts), torch.cat(sin_parts))
        video["context"] = projected_context
        video["context_mask"] = joint_context_mask
        video["meta"].update(
            {
                "video_len": video_len,
                "action_len": action_tokens.shape[1],
                "state_len": state_tokens.shape[1],
                "action_modulation_index": video_steps,
            }
        )
        return video

    def post_unified_dit(self, tokens: torch.Tensor, pre_state: dict) -> dict[str, torch.Tensor]:
        video_len = int(pre_state["meta"]["video_len"])
        action_len = int(pre_state["meta"]["action_len"])
        action_index = int(pre_state["meta"]["action_modulation_index"])
        action_tokens = tokens[:, video_len:video_len + action_len]
        embedding = pre_state["t_mod"]["embedding"][:, action_index:action_index + 1]
        adaln_lora = pre_state["t_mod"]["adaln_lora"][:, action_index:action_index + 1]
        embedding = embedding.expand(-1, action_len, -1)
        adaln_lora = adaln_lora.expand(-1, action_len, -1)
        return {
            "video": self.post_dit(tokens[:, :video_len], pre_state),
            "action_tokens": self.final_layer.condition_tokens(
                action_tokens, embedding, adaln_lora
            ),
        }

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        condition_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if condition_mask is not None or padding_mask is not None:
            # Standalone callers can still provide explicit masks through the native path.
            b, _, t, h, w = x.shape
            if padding_mask is None:
                padding_mask = torch.zeros((b, 1, t, h, w), device=x.device, dtype=x.dtype)
            elif padding_mask.ndim == 3:
                padding_mask = padding_mask[:, None, None].expand(-1, -1, t, -1, -1)
            elif padding_mask.ndim == 4 and padding_mask.shape[1] == 1:
                padding_mask = padding_mask[:, :, None].expand(-1, -1, t, -1, -1)
            elif padding_mask.ndim != 5:
                raise ValueError(
                    "padding_mask must be [B,H,W], [B,1,H,W], or [B,1,T,H,W]."
                )
            if padding_mask.shape != (b, 1, t, h, w):
                raise ValueError(
                    f"padding_mask must resolve to {(b, 1, t, h, w)}, got "
                    f"{tuple(padding_mask.shape)}."
                )
            if condition_mask is None:
                condition_mask = torch.zeros_like(padding_mask)
            if condition_mask.shape != (b, 1, t, h, w):
                raise ValueError(
                    f"condition_mask must be {(b, 1, t, h, w)}, got "
                    f"{tuple(condition_mask.shape)}."
                )
            hidden = self.x_embedder(torch.cat((x, condition_mask, padding_mask.to(x.dtype)), dim=1))
            if timestep.ndim == 1:
                timestep = timestep[:, None].expand(b, t)
            projected = self.crossattn_proj(context.to(dtype=hidden.dtype))
            features = self._timestep_features(timestep, self.config.hidden_size).to(hidden.dtype)
            embedding, adaln = self.t_embedder[1](features)
            embedding = self.t_embedding_norm(embedding)
            rope = self.pos_embedder(hidden.shape[1], hidden.shape[2], hidden.shape[3], hidden.device)
            for block in self.blocks:
                if self.use_gradient_checkpointing and torch.is_grad_enabled():
                    def _block_forward(value, current_block=block):
                        return current_block(
                            value, embedding, projected, adaln, rope, None
                        )

                    hidden = checkpoint(
                        _block_forward, hidden, use_reentrant=False
                    )
                else:
                    hidden = block(hidden, embedding, projected, adaln, rope, None)
            return self._unpatchify(self.final_layer(hidden.to(projected.dtype), embedding, adaln))
        pre = self.pre_dit(x, timestep, context, context_mask, False)
        tokens = pre["tokens"]
        mask = None
        if self.video_attention_mask_mode != "bidirectional":
            mask = self.build_structured_video_attention_mask(
                tokens.shape[1], pre["meta"]["tokens_per_frame"], tokens.device
            )
        for index in range(len(self.blocks)):
            if self.use_gradient_checkpointing and torch.is_grad_enabled():
                def _block_forward(value, layer_index=index):
                    return self.forward_block(layer_index, value, pre, mask)

                tokens = checkpoint(
                    _block_forward, tokens, use_reentrant=False
                )
            else:
                tokens = self.forward_block(index, tokens, pre, mask)
        return self.post_dit(tokens, pre)
