from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..backbone.flux2.imports import ensure_flux2_importable
from ..backbone.protocol import BLOCK_PROTOCOL_FLUX2
from utils.logging_config import get_logger

logger = get_logger(__name__)


class SlimFlux2SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, attn_head_dim: int):
        super().__init__()
        ensure_flux2_importable()
        from flux2.model import QKNorm

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.attn_dim = self.num_heads * self.attn_head_dim
        self.qkv = nn.Linear(self.hidden_dim, 3 * self.attn_dim, bias=False)
        self.norm = QKNorm(self.attn_head_dim)
        self.proj = nn.Linear(self.attn_dim, self.hidden_dim, bias=False)


class SlimFlux2DoubleBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, attn_head_dim: int, mlp_ratio: float):
        super().__init__()
        ensure_flux2_importable()
        from flux2.model import SiLUActivation

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.attn_dim = self.num_heads * self.attn_head_dim
        mlp_hidden_dim = int(round(self.hidden_dim * float(mlp_ratio)))
        self.img_norm1 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.img_attn = SlimFlux2SelfAttention(self.hidden_dim, self.num_heads, self.attn_head_dim)
        self.img_norm2 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, mlp_hidden_dim * 2, bias=False),
            SiLUActivation(),
            nn.Linear(mlp_hidden_dim, self.hidden_dim, bias=False),
        )

    def prepare_qkv(self, x: torch.Tensor, pe: torch.Tensor, modulation):
        from einops import rearrange
        from flux2.model import apply_rope

        mod1, mod2 = modulation
        shift1, scale1, gate1 = mod1
        shift2, scale2, gate2 = mod2
        qkv = self.img_attn.qkv((1 + scale1) * self.img_norm1(x) + shift1)
        q, k, v = rearrange(qkv, "b l (k h d) -> k b h l d", k=3, h=self.num_heads)
        q, k = self.img_attn.norm(q, k, v)
        q, k = apply_rope(q, k, pe)
        flatten = lambda value: value.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.attn_dim)
        return {
            "q": flatten(q), "k": flatten(k), "v": flatten(v), "residual_x": x,
            "shift2": shift2, "scale2": scale2, "gate1": gate1, "gate2": gate2,
        }

    def apply_post(self, attention: torch.Tensor, state: dict[str, torch.Tensor]) -> torch.Tensor:
        x = state["residual_x"] + state["gate1"] * self.img_attn.proj(attention)
        return x + state["gate2"] * self.img_mlp(
            (1 + state["scale2"]) * self.img_norm2(x) + state["shift2"]
        )


class SlimFlux2SingleBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, attn_head_dim: int, mlp_ratio: float):
        super().__init__()
        ensure_flux2_importable()
        from flux2.model import QKNorm, SiLUActivation

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.attn_dim = self.num_heads * self.attn_head_dim
        self.mlp_hidden_dim = int(round(self.hidden_dim * float(mlp_ratio)))
        self.linear1 = nn.Linear(self.hidden_dim, 3 * self.attn_dim + 2 * self.mlp_hidden_dim, bias=False)
        self.linear2 = nn.Linear(self.attn_dim + self.mlp_hidden_dim, self.hidden_dim, bias=False)
        self.norm = QKNorm(self.attn_head_dim)
        self.pre_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp_act = SiLUActivation()

    def prepare_qkv(self, x: torch.Tensor, pe: torch.Tensor, modulation):
        from einops import rearrange
        from flux2.model import apply_rope

        shift, scale, gate = modulation
        projected = self.linear1((1 + scale) * self.pre_norm(x) + shift)
        qkv, mlp = torch.split(projected, [3 * self.attn_dim, 2 * self.mlp_hidden_dim], dim=-1)
        q, k, v = rearrange(qkv, "b l (k h d) -> k b h l d", k=3, h=self.num_heads)
        q, k = self.norm(q, k, v)
        q, k = apply_rope(q, k, pe)
        flatten = lambda value: value.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.attn_dim)
        return {"q": flatten(q), "k": flatten(k), "v": flatten(v), "mlp": mlp, "gate": gate, "residual_x": x}

    def apply_post(self, attention: torch.Tensor, state: dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.linear2(torch.cat((attention, self.mlp_act(state["mlp"])), dim=2))
        return state["residual_x"] + state["gate"] * output


class Flux2ActionHead(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, action_dim, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        )

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        return self.linear((1 + scale[:, None]) * self.norm_final(x) + shift[:, None])


class ActionDiTFlux2(nn.Module):
    """A slim action stream following FLUX.2's double/single stage topology."""

    block_protocol = BLOCK_PROTOCOL_FLUX2

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 1024,
        num_heads: int = 24,
        attn_head_dim: int = 128,
        num_layers_double: int = 5,
        num_layers_single: int = 20,
        mlp_ratio: float = 4.0,
        max_action_horizon: int = 64,
        attention_backend: str = "sdpa",
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        ensure_flux2_importable()
        from flux2.model import MLPEmbedder, Modulation

        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_kv_heads = self.num_heads
        self.attn_head_dim = int(attn_head_dim)
        self.double_layers = int(num_layers_double)
        self.single_layers = int(num_layers_single)
        self.max_action_horizon = int(max_action_horizon)
        self.attention_backend = str(attention_backend)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.action_encoder = nn.Linear(self.action_dim, self.hidden_dim)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_dim, disable_bias=True)
        self.double_stream_modulation_img = Modulation(self.hidden_dim, double=True, disable_bias=True)
        self.single_stream_modulation = Modulation(self.hidden_dim, double=False, disable_bias=True)
        self.double_blocks = nn.ModuleList([
            SlimFlux2DoubleBlock(self.hidden_dim, self.num_heads, self.attn_head_dim, mlp_ratio)
            for _ in range(self.double_layers)
        ])
        self.single_blocks = nn.ModuleList([
            SlimFlux2SingleBlock(self.hidden_dim, self.num_heads, self.attn_head_dim, mlp_ratio)
            for _ in range(self.single_layers)
        ])
        self.head = Flux2ActionHead(self.hidden_dim, self.action_dim)

    @property
    def blocks(self):
        return list(self.double_blocks) + list(self.single_blocks)

    @classmethod
    def from_pretrained(cls, action_dit_config: dict[str, Any], action_dit_pretrained_path=None,
                        skip_dit_load_from_pretrain=False, device="cuda", torch_dtype=torch.bfloat16):
        model = cls(**dict(action_dit_config)).to(device=device, dtype=torch_dtype)
        if skip_dit_load_from_pretrain or not action_dit_pretrained_path:
            return model
        payload = torch.load(action_dit_pretrained_path, map_location="cpu", weights_only=True)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            logger.warning("FLUX.2 action checkpoint mismatch: missing=%d unexpected=%d", len(missing), len(unexpected))
        return model

    @staticmethod
    def build_action_ids(batch_size: int, seq_len: int, *, device, dtype) -> torch.Tensor:
        ids = torch.zeros(batch_size, seq_len, 4, device=device, dtype=dtype)
        ids[..., 0] = 2.0
        ids[..., 1] = torch.arange(seq_len, device=device, dtype=dtype)[None]
        return ids

    def pre_dit(self, action_tokens: torch.Tensor, timestep: torch.Tensor, **_: Any) -> dict[str, Any]:
        batch_size, seq_len, action_dim = action_tokens.shape
        if action_dim != self.action_dim or seq_len > self.max_action_horizon:
            raise ValueError(f"Invalid FLUX.2 action shape {tuple(action_tokens.shape)}")
        from flux2.model import timestep_embedding

        tokens = self.action_encoder(action_tokens)
        vec = self.time_in(timestep_embedding(timestep, 256)).to(dtype=tokens.dtype)
        return {
            "tokens": tokens,
            "ids": self.build_action_ids(batch_size, seq_len, device=tokens.device, dtype=tokens.dtype),
            "t_mod": {
                "vec": vec,
                "double_img": self.double_stream_modulation_img(vec),
                "single": self.single_stream_modulation(vec)[0],
            },
            "context": None, "context_mask": None,
            "meta": {"batch_size": batch_size, "seq_len": seq_len},
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: dict[str, Any]) -> torch.Tensor:
        return self.head(tokens, pre_state["t_mod"]["vec"])
