import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..backbone.protocol import BLOCK_PROTOCOL_STANDARD
from .dit import EmbedND, MLPEmbedder, Modulation, QKNorm, SiLUActivation, apply_rope

from model.helpers.rope import apply_prepared_rope, prepare_rope, rope_apply
from utils.logging_config import get_logger

from ..helpers.gradient import gradient_checkpoint_forward
from .attention import (
    KeyPaddingMask,
    StructuredAttentionMask,
    prepare_key_padding_mask,
    require_attention_backend,
    run_attention,
)

logger = get_logger(__name__)

STATE_POSITIONS = ("context", "sequence")


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(
                dim // 2
            ),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


class ActionSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        attention_backend: str = "sdpa",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = nn.RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = nn.RMSNorm(self.attn_hidden_dim, eps=eps)
        self.attention_backend = require_attention_backend(attention_backend)

    def project_qkv(self, x):
        if hasattr(self, "qkv"):
            q, k, v = self.qkv(x).chunk(3, dim=-1)
        else:
            q, k, v = self.q(x), self.k(x), self.v(x)
        return self.norm_q(q), self.norm_k(k), v

    def forward(
        self,
        x,
        freqs,
        self_attn_mask: Optional[torch.Tensor | StructuredAttentionMask] = None,
    ):
        q, k, v = self.project_qkv(x)
        rope = prepare_rope(freqs, q.device, q.dtype)
        q = apply_prepared_rope(q, rope, self.num_heads)
        k = apply_prepared_rope(k, rope, self.num_heads)
        x = run_attention(
            q=q,
            k=k,
            v=v,
            num_heads=self.num_heads,
            attention_mask=self_attn_mask,
            backend=self.attention_backend,
        )
        return self.o(x)


class ActionCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        attention_backend: str = "sdpa",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = nn.RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = nn.RMSNorm(self.attn_hidden_dim, eps=eps)
        self.attention_backend = require_attention_backend(attention_backend)

    def project_kv(self, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.norm_k(self.k(ctx)), self.v(ctx)

    def forward_with_projected_kv(
        self,
        x: torch.Tensor,
        projected_kv: tuple[torch.Tensor, torch.Tensor],
        ctx_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k, v = projected_kv
        x = run_attention(
            q=q,
            k=k,
            v=v,
            num_heads=self.num_heads,
            attention_mask=ctx_mask,
            backend=self.attention_backend,
        )
        return self.o(x)

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        ctx_mask: Optional[torch.Tensor] = None,
        projected_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        if projected_kv is None:
            projected_kv = self.project_kv(ctx)
        return self.forward_with_projected_kv(x, projected_kv, ctx_mask)


class ActionGateModule(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class ActionDiTBlock(nn.Module):
    uses_mixed_attention_api = False

    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        attention_backend: str = "sdpa",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = ActionSelfAttention(
            hidden_dim,
            attn_head_dim,
            num_heads,
            eps,
            attention_backend=attention_backend,
        )
        self.cross_attn = ActionCrossAttention(
            hidden_dim,
            attn_head_dim,
            num_heads,
            eps,
            attention_backend=attention_backend,
        )
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.gate = ActionGateModule()

    def forward(
        self,
        x,
        context,
        t_mod,
        freqs,
        context_mask=None,
        self_attn_mask: Optional[torch.Tensor | StructuredAttentionMask] = None,
        context_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).unbind(dim=2)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(
            x, gate_msa, self.self_attn(input_x, freqs, self_attn_mask=self_attn_mask)
        )
        if self.cross_attn is not None:
            x = x + self.cross_attn(
                self.norm3(x), context, ctx_mask=context_mask, projected_kv=context_kv
            )
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


def normalize_state_position(value: str) -> str:
    position = str(value).strip().lower()
    if position not in STATE_POSITIONS:
        raise ValueError(
            f"Unsupported state_position: {value!r}. Expected one of {STATE_POSITIONS}."
        )
    return position


def validate_checkpoint_state_position(payload: dict, current_position: str) -> None:
    checkpoint_position = payload.get("state_position")
    if checkpoint_position is None:
        logger.warning(
            "Checkpoint has no state_position metadata; using configured value %r.",
            current_position,
        )
        return
    checkpoint_position = normalize_state_position(checkpoint_position)
    if checkpoint_position != current_position:
        raise ValueError(
            f"Checkpoint state_position {checkpoint_position!r} does not match model "
            f"state_position {current_position!r}."
        )


class ActionEncoder(nn.Module):
    """Two-layer action value encoder; timestep and position are applied outside."""

    def __init__(self, action_dim: int, hidden_dim: int, projector_hidden_dim: int):
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.projector_hidden_dim = int(projector_hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(action_dim, projector_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(projector_hidden_dim, hidden_dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 3:
            raise ValueError(
                f"`actions` must be 3D [B, T, D], got {tuple(actions.shape)}"
            )
        if actions.shape[2] != self.action_dim:
            raise ValueError(
                f"`actions` last dim must be {self.action_dim}, got {actions.shape[2]}"
            )
        return self.net(actions)


class StateEncoder(nn.Module):

    def __init__(self, state_dim: int, hidden_dim: int, projector_hidden_dim: int):
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.projector_hidden_dim = int(projector_hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim, projector_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(projector_hidden_dim, hidden_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"`state` last dim must be {self.state_dim}, got {state.shape[-1]}"
            )
        return self.net(state)


class ActionDecoder(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, projector_hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, projector_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(projector_hidden_dim, action_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


class StagedSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, attn_head_dim: int):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.attn_dim = self.num_heads * self.attn_head_dim
        self.qkv = nn.Linear(self.hidden_dim, 3 * self.attn_dim, bias=False)
        self.norm = QKNorm(self.attn_head_dim)
        self.proj = nn.Linear(self.attn_dim, self.hidden_dim, bias=False)

class SequentialActionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attn_head_dim: int,
        mlp_ratio: float,
        options,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.attn_dim = self.num_heads * self.attn_head_dim
        mlp_hidden_dim = int(round(self.hidden_dim * float(mlp_ratio)))
        self.img_norm1 = nn.LayerNorm(
            self.hidden_dim, elementwise_affine=False, eps=1e-6
        )
        self.img_attn = StagedSelfAttention(
            self.hidden_dim, self.num_heads, self.attn_head_dim
        )
        configure_attention(self.img_attn, options, 1e-6)
        self.img_norm2 = nn.LayerNorm(
            self.hidden_dim, elementwise_affine=False, eps=1e-6
        )
        self.img_mlp = make_ffn(
            self.hidden_dim,
            mlp_hidden_dim,
            options["ffn_activation"],
            False,
        )

    def prepare_qkv(self, x: torch.Tensor, pe: torch.Tensor, modulation):

        mod1, mod2 = modulation
        shift1, scale1, gate1 = mod1
        shift2, scale2, gate2 = mod2
        q, k, v = staged_qkv(self.img_attn, (1 + scale1) * self.img_norm1(x) + shift1)
        q, k = apply_rope(q, k, pe)
        flatten = lambda value: value.transpose(1, 2).reshape(
            x.shape[0], x.shape[1], self.attn_dim
        )
        return {
            "q": flatten(q),
            "k": flatten(k),
            "v": flatten(v),
            "residual_x": x,
            "shift2": shift2,
            "scale2": scale2,
            "gate1": gate1,
            "gate2": gate2,
        }

    def apply_post(
        self, attention: torch.Tensor, state: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        x = state["residual_x"] + state["gate1"] * self.img_attn.proj(attention)
        return x + state["gate2"] * self.img_mlp(
            (1 + state["scale2"]) * self.img_norm2(x) + state["shift2"]
        )

class ParallelActionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attn_head_dim: int,
        mlp_ratio: float,
        options,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.attn_dim = self.num_heads * self.attn_head_dim
        self.mlp_hidden_dim = int(round(self.hidden_dim * float(mlp_ratio)))
        self.options = options
        self.mlp_factor = 2 if options["ffn_activation"] == "swiglu" else 1
        self.linear1 = nn.Linear(
            self.hidden_dim,
            3 * self.attn_dim + self.mlp_factor * self.mlp_hidden_dim,
            bias=options["qkv_bias"],
        )
        if options["qkv_projection"] == "separate":
            del self.linear1
            self.q = nn.Linear(self.hidden_dim, self.attn_dim, bias=options["qkv_bias"])
            self.k = nn.Linear(self.hidden_dim, self.attn_dim, bias=options["qkv_bias"])
            self.v = nn.Linear(self.hidden_dim, self.attn_dim, bias=options["qkv_bias"])
            self.mlp_in = nn.Linear(
                self.hidden_dim,
                self.mlp_factor * self.mlp_hidden_dim,
                bias=options["qkv_bias"],
            )
        if options["qk_norm"] == "projection":
            self.norm_q = nn.RMSNorm(self.attn_dim, eps=1e-6)
            self.norm_k = nn.RMSNorm(self.attn_dim, eps=1e-6)
        self.linear2 = nn.Linear(
            self.attn_dim + self.mlp_hidden_dim, self.hidden_dim, bias=False
        )
        self.norm = QKNorm(self.attn_head_dim)
        if options["qk_norm"] != "head":
            del self.norm
        self.pre_norm = nn.LayerNorm(
            self.hidden_dim, elementwise_affine=False, eps=1e-6
        )
        self.mlp_act = (
            SiLUActivation()
            if options["ffn_activation"] == "swiglu"
            else nn.GELU(approximate="tanh")
        )

    def prepare_qkv(self, x: torch.Tensor, pe: torch.Tensor, modulation):

        shift, scale, gate = modulation
        value = (1 + scale) * self.pre_norm(x) + shift
        if self.options["qkv_projection"] == "fused":
            projected = self.linear1(value)
            qkv, mlp = torch.split(
                projected,
                [3 * self.attn_dim, self.mlp_factor * self.mlp_hidden_dim],
                dim=-1,
            )
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            q, k, v, mlp = (
                self.q(value),
                self.k(value),
                self.v(value),
                self.mlp_in(value),
            )
        if self.options["qk_norm"] == "projection":
            q, k = self.norm_q(q), self.norm_k(k)
        split = lambda value: value.unflatten(
            -1, (self.num_heads, self.attn_head_dim)
        ).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        if self.options["qk_norm"] == "head":
            q, k = self.norm(q, k, v)
        q, k = apply_rope(q, k, pe)
        flatten = lambda value: value.transpose(1, 2).reshape(
            x.shape[0], x.shape[1], self.attn_dim
        )
        return {
            "q": flatten(q),
            "k": flatten(k),
            "v": flatten(v),
            "mlp": mlp,
            "gate": gate,
            "residual_x": x,
        }

    def apply_post(
        self, attention: torch.Tensor, state: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        output = self.linear2(torch.cat((attention, self.mlp_act(state["mlp"])), dim=2))
        return state["residual_x"] + state["gate"] * output

class TimeModulatedActionHead(nn.Module):
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

class GatedSiLU(nn.Module):
    def forward(self, x):
        left, right = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(left) * right

def make_ffn(hidden, width, activation, bias):
    return nn.Sequential(
        nn.Linear(hidden, width * (2 if activation == "swiglu" else 1), bias=bias),
        GatedSiLU() if activation == "swiglu" else nn.GELU(approximate="tanh"),
        nn.Linear(width, hidden, bias=bias),
    )

class HeadNorm(nn.Module):
    def __init__(self, heads, width, eps):
        super().__init__()
        self.heads, self.width = heads, width
        self.norm = nn.RMSNorm(width, eps=eps)

    def forward(self, value):
        return self.norm(value.unflatten(-1, (self.heads, self.width))).flatten(-2)

def configure_attention(attn, options, eps, standard=False):
    hidden, heads, width = attn.hidden_dim, attn.num_heads, attn.attn_head_dim
    dimension = heads * width
    if standard:
        if options["qkv_projection"] == "fused":
            del attn.q, attn.k, attn.v
            attn.qkv = nn.Linear(hidden, 3 * dimension, bias=options["qkv_bias"])
        elif not options["qkv_bias"]:
            for key in ("q", "k", "v"):
                setattr(attn, key, nn.Linear(hidden, dimension, bias=False))
        if options["qk_norm"] != "projection":
            for key in ("norm_q", "norm_k"):
                setattr(
                    attn,
                    key,
                    (
                        HeadNorm(heads, width, eps)
                        if options["qk_norm"] == "head"
                        else nn.Identity()
                    ),
                )
        return
    # Staged QK tensors are [B,H,L,D]. Projection normalization is applied
    # before splitting heads, so both normalization conventions are explicit.
    attn.projection = options["qkv_projection"]
    attn.norm_kind = options["qk_norm"]
    if attn.projection == "separate":
        del attn.qkv
        attn.q = nn.Linear(hidden, dimension, bias=options["qkv_bias"])
        attn.k = nn.Linear(hidden, dimension, bias=options["qkv_bias"])
        attn.v = nn.Linear(hidden, dimension, bias=options["qkv_bias"])
    elif options["qkv_bias"]:
        attn.qkv = nn.Linear(hidden, dimension * 3, bias=True)
    if attn.norm_kind != "head":
        del attn.norm
    if attn.norm_kind == "projection":
        attn.norm_q = nn.RMSNorm(dimension, eps=eps)
        attn.norm_k = nn.RMSNorm(dimension, eps=eps)

def staged_qkv(attn, value):
    if attn.projection == "fused":
        q, k, v = attn.qkv(value).chunk(3, dim=-1)
    else:
        q, k, v = attn.q(value), attn.k(value), attn.v(value)
    if attn.norm_kind == "projection":
        q, k = attn.norm_q(q), attn.norm_k(k)
    split = lambda x: x.unflatten(-1, (attn.num_heads, attn.attn_head_dim)).transpose(
        1, 2
    )
    q, k, v = split(q), split(k), split(v)
    if attn.norm_kind == "head":
        q, k = attn.norm(q, k, v)
    return q, k, v


class ActionDiT(nn.Module):
    block_protocol = BLOCK_PROTOCOL_STANDARD
    ACTION_BACKBONE_SKIP_PREFIXES = ("action_encoder.", "action_decoder.", "head.")
    ACTION_BACKBONE_META_KEYS = (
        "hidden_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
        "eps",
    )

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        ffn_dim: int | None = None,
        text_dim: int = 0,
        freq_dim: int = 256,
        eps: float = 1e-6,
        num_heads: int = 24,
        attn_head_dim: int = 128,
        num_layers: int | None = None,
        projector_hidden_dim: int = 64,
        use_gradient_checkpointing: bool = False,
        attention_backend: str = "sdpa",
        preset: str = "standard",
        cross_attention: bool | None = None,
        qkv_projection: str | None = None,
        qkv_bias: bool | None = None,
        qk_norm: str | None = None,
        ffn_activation: str | None = None,
        encoder_type: str | None = None,
        decoder_type: str | None = None,
        position_encoding: str | None = None,
        time_embedding_type: str | None = None,
        num_layers_double: int = 5,
        num_layers_single: int = 20,
        mlp_ratio: float = 4.0,
        max_action_horizon: int = 1024,
    ):
        super().__init__()
        if preset not in {"standard", "staged"}:
            raise ValueError(f"Unknown ActionDiT preset: {preset!r}")
        defaults = (
            {
                "cross_attention": True,
                "qkv_projection": "separate",
                "qkv_bias": True,
                "qk_norm": "projection",
                "ffn_activation": "gelu",
                "encoder_type": "mlp",
                "decoder_type": "mlp",
                "position_encoding": "rope_1d",
                "time_embedding_type": "standard",
            }
            if preset == "standard"
            else {
                "cross_attention": False,
                "qkv_projection": "fused",
                "qkv_bias": False,
                "qk_norm": "head",
                "ffn_activation": "swiglu",
                "encoder_type": "linear",
                "decoder_type": "time_modulated",
                "position_encoding": "rope_nd",
                "time_embedding_type": "scaled",
            }
        )
        options = dict(
            cross_attention=cross_attention,
            qkv_projection=qkv_projection,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            ffn_activation=ffn_activation,
            encoder_type=encoder_type,
            decoder_type=decoder_type,
            position_encoding=position_encoding,
            time_embedding_type=time_embedding_type,
        )
        options = {
            key: defaults[key] if value is None else value
            for key, value in options.items()
        }
        choices = {
            "qkv_projection": {"fused", "separate"},
            "qk_norm": {"projection", "head", "none"},
            "ffn_activation": {"gelu", "swiglu"},
            "encoder_type": {"linear", "mlp"},
            "decoder_type": {"linear", "mlp", "time_modulated"},
            "position_encoding": {"rope_1d", "rope_nd"},
            "time_embedding_type": {"standard", "scaled"},
        }
        for key, allowed in choices.items():
            if options[key] not in allowed:
                raise ValueError(
                    f"Invalid ActionDiT {key}: {options[key]!r}; expected {sorted(allowed)}"
                )
        if not isinstance(options["cross_attention"], bool) or not isinstance(
            options["qkv_bias"], bool
        ):
            raise ValueError("cross_attention and qkv_bias must be booleans")
        if preset == "standard" and options["position_encoding"] != "rope_1d":
            raise ValueError(
                "standard layers require rope_1d; use staged layers for rope_nd"
            )
        if (
            preset == "staged"
            and num_layers is not None
            and num_layers != num_layers_double + num_layers_single
        ):
            raise ValueError(
                "staged num_layers must equal num_layers_double + num_layers_single"
            )
        num_layers = 30 if num_layers is None else num_layers
        if min(hidden_dim, action_dim, freq_dim, num_layers) <= 0 or freq_dim % 2:
            raise ValueError("Dimensions/depth must be positive; freq_dim must be even")
        if preset == "staged" and freq_dim != 256:
            raise ValueError(
                "The staged timestep projection currently requires freq_dim=256"
            )
        if preset == "staged" and eps != 1e-6:
            raise ValueError("The staged normalization currently requires eps=1e-6")
        ffn_dim = int(ffn_dim if ffn_dim is not None else hidden_dim * mlp_ratio)
        if ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        self.resolved_config = dict(
            preset=preset,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            ffn_dim=ffn_dim,
            text_dim=text_dim,
            freq_dim=freq_dim,
            eps=eps,
            num_heads=num_heads,
            attn_head_dim=attn_head_dim,
            num_layers=(
                num_layers
                if preset == "standard"
                else num_layers_double + num_layers_single
            ),
            projector_hidden_dim=projector_hidden_dim,
            num_layers_double=num_layers_double,
            num_layers_single=num_layers_single,
            max_action_horizon=max_action_horizon,
            **options,
        )
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.projector_hidden_dim = int(projector_hidden_dim)
        self.max_action_horizon = max_action_horizon
        self.preset = preset
        self.cross_attention = options["cross_attention"]
        self.options = options
        if preset == "staged":
            self._init_staged(
                hidden_dim,
                action_dim,
                text_dim,
                num_heads,
                attn_head_dim,
                num_layers_double,
                num_layers_single,
                ffn_dim / hidden_dim,
                max_action_horizon,
                projector_hidden_dim,
                attention_backend,
                use_gradient_checkpointing,
            )
            return
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.projector_hidden_dim = int(projector_hidden_dim)
        self.attention_backend = require_attention_backend(attention_backend)

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(
                f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}"
            )

        self.action_encoder = ActionEncoder(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            projector_hidden_dim=self.projector_hidden_dim,
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6)
        )
        self.blocks = nn.ModuleList(
            [
                ActionDiTBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                    attention_backend=self.attention_backend,
                )
                for _ in range(num_layers)
            ]
        )
        self.action_decoder = ActionDecoder(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            projector_hidden_dim=self.projector_hidden_dim,
        )
        self.freqs = precompute_freqs_cis(attn_head_dim, end=max_action_horizon)

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self._configure_standard(options, eps, projector_hidden_dim)

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        return {
            key
            for key in keys
            if not any(
                key.startswith(prefix) for prefix in cls.ACTION_BACKBONE_SKIP_PREFIXES
            )
        }

    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ActionDiT":
        if action_dit_config is None:
            raise ValueError(
                "`action_dit_config` is required for ActionDiT.from_pretrained()."
            )
        if action_dit_config.get("preset") == "staged":
            model = cls(**action_dit_config).to(device=device, dtype=torch_dtype)
            if action_dit_pretrained_path and not skip_dit_load_from_pretrain:
                payload = torch.load(
                    action_dit_pretrained_path, map_location="cpu", weights_only=True
                )
                signature = lambda config: {
                    k: v for k, v in config.items() if k != "action_dim"
                }
                if not isinstance(payload.get("action_config"), dict) or signature(
                    payload["action_config"]
                ) != signature(model.resolved_config):
                    raise ValueError(
                        "Preprocessed ActionDiT configuration does not match requested structure"
                    )
                state = model.state_dict()
                backbone = payload.get("backbone_state_dict", {})
                if set(backbone) != cls.backbone_key_set(state):
                    raise ValueError(
                        "Preprocessed ActionDiT must contain every configured backbone tensor"
                    )
                state.update(backbone)
                model.load_state_dict(state, strict=True)
            return model
        if skip_dit_load_from_pretrain:
            logger.info(
                "Skipping ActionDiT pretrained load (`skip_dit_load_from_pretrain=True`); "
                "initializing action expert randomly and expecting checkpoint override."
            )
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        if not action_dit_pretrained_path:
            logger.info(
                "No `action_dit_pretrained_path` provided, initializing ActionDiT with random weights."
            )
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        from pathlib import Path

        p = Path(action_dit_pretrained_path)
        action_dit_pretrained_path = str(p)
        if not os.path.isfile(action_dit_pretrained_path):
            raise FileNotFoundError(
                f"`action_dit_pretrained_path` does not exist: {action_dit_pretrained_path}"
            )

        action_cfg = dict(action_dit_config)
        action_expert = cls(**action_cfg).to(device=device, dtype=torch_dtype)
        action_state = action_expert.state_dict()
        expected_backbone_keys = cls.backbone_key_set(action_state.keys())

        payload = torch.load(action_dit_pretrained_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid action backbone payload type from {action_dit_pretrained_path}: {type(payload)}"
            )

        policy = payload.get("policy", {})
        if policy:
            logger.info(f"ActionDiT backbone payload policy: {policy}")

        meta = payload.get("meta")
        expected_meta = {
            "hidden_dim": int(action_cfg["hidden_dim"]),
            "ffn_dim": int(action_cfg["ffn_dim"]),
            "num_layers": int(action_cfg["num_layers"]),
            "num_heads": int(action_cfg["num_heads"]),
            "attn_head_dim": int(action_cfg["attn_head_dim"]),
            "text_dim": int(action_cfg["text_dim"]),
            "freq_dim": int(action_cfg["freq_dim"]),
            "eps": float(action_cfg["eps"]),
        }
        for key in cls.ACTION_BACKBONE_META_KEYS:
            if key not in meta:
                raise ValueError(
                    f"`meta.{key}` missing in {action_dit_pretrained_path}"
                )
            expected_value = expected_meta[key]
            got_value = meta[key]
            if key == "eps":
                if abs(float(got_value) - float(expected_value)) > 1e-12:
                    raise ValueError(
                        f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                        f"expected {expected_value}, got {got_value}"
                    )
            elif int(got_value) != int(expected_value):
                raise ValueError(
                    f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                    f"expected {expected_value}, got {got_value}"
                )

        backbone_state_dict = payload.get("backbone_state_dict")
        if not isinstance(backbone_state_dict, dict):
            raise ValueError(
                f"`backbone_state_dict` must be a dict in {action_dit_pretrained_path}, "
                f"got {type(backbone_state_dict)}"
            )

        provided_keys = set(backbone_state_dict.keys())
        missing_keys = sorted(expected_backbone_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - expected_backbone_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "Action backbone key mismatch in preprocessed payload. "
                f"missing={missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}, "
                f"unexpected={unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
            )

        merged_state = dict(action_state)
        for key in expected_backbone_keys:
            value = backbone_state_dict[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"`backbone_state_dict[{key}]` must be torch.Tensor in {action_dit_pretrained_path}, "
                    f"got {type(value)}"
                )
            target = merged_state[key]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for `{key}` in {action_dit_pretrained_path}: "
                    f"expected {tuple(target.shape)}, got {tuple(value.shape)}"
                )
            merged_state[key] = value.to(device=target.device, dtype=target.dtype)

        action_expert.load_state_dict(merged_state, strict=True)
        logger.info(
            "Loaded ActionDiT backbone from %s (keys=%d; random_kept_prefixes=%s).",
            action_dit_pretrained_path,
            len(expected_backbone_keys),
            list(cls.ACTION_BACKBONE_SKIP_PREFIXES),
        )
        return action_expert.to(device=device, dtype=torch_dtype)

    def project_context(self, context: torch.Tensor) -> torch.Tensor:
        if not self.cross_attention:
            raise ValueError("This ActionDiT has cross_attention disabled")
        return self.text_embedding(context)

    def build_cross_attention_kv_cache(
        self, projected_context: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        if not self.cross_attention:
            raise ValueError(
                "Cross-attention cache requested with cross_attention disabled"
            )
        return tuple(
            block.cross_attn.project_kv(projected_context) for block in self.blocks
        )

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor | None = None,
        context_mask: Optional[torch.Tensor] = None,
        context_is_projected: bool = False,
        cross_kv_cache: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None,
        prepare_rotary_factors: bool = True,
    ) -> Dict[str, Any]:
        if self.preset == "staged":
            return self._staged_pre_dit(
                action_tokens,
                timestep,
                context,
                context_mask,
                context_is_projected,
                cross_kv_cache,
            )
        if action_tokens.ndim != 3:
            raise ValueError(
                f"`action_tokens` must be 3D [B, T, action_dim], got shape {tuple(action_tokens.shape)}"
            )
        if action_tokens.shape[2] != self.action_dim:
            raise ValueError(
                f"`action_tokens` last dim must be {self.action_dim}, got {action_tokens.shape[2]}"
            )
        if timestep.ndim != 1:
            raise ValueError(
                f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}"
            )
        if context is None and not self.cross_attention:
            context = action_tokens.new_zeros(action_tokens.shape[0], 0, self.text_dim)
        if context is None or context.ndim != 3:
            raise ValueError(
                f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}"
            )

        batch_size = action_tokens.shape[0]
        if context.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch between action tokens and text context: {batch_size} vs {context.shape[0]}"
            )
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError(
                    "During training, action timestep length must match batch_size."
                )
            timestep = timestep.expand(batch_size)

        if context_mask is not None:
            if context_mask.ndim != 2:
                raise ValueError(
                    f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}"
                )
            if (
                context_mask.shape[0] != batch_size
                or context_mask.shape[1] != context.shape[1]
            ):
                raise ValueError(
                    f"`context_mask` shape must match `context` shape [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        seq_len = action_tokens.shape[1]
        if seq_len > self.freqs.shape[0]:
            raise ValueError(
                f"Action token length {seq_len} exceeds RoPE cache {self.freqs.shape[0]}."
            )

        t = self.time_embedding(self._time_features(timestep, self.freq_dim))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim)).unsqueeze(1)

        tokens = self.action_encoder(action_tokens)
        context_emb = (
            (context if context_is_projected else self.project_context(context))
            if self.cross_attention
            else None
        )
        context_attn_mask = prepare_key_padding_mask(context_mask)
        freqs = self.freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device)

        pre_state = {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "cross_kv_cache": cross_kv_cache,
            "meta": {
                "batch_size": batch_size,
                "seq_len": seq_len,
            },
        }
        if prepare_rotary_factors:
            pre_state["rope"] = prepare_rope(freqs, tokens.device, tokens.dtype)
        return pre_state

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        if self.preset == "staged":
            return self._staged_post_dit(tokens, pre_state)
        action_len = int(pre_state.get("meta", {}).get("action_len", tokens.shape[1]))
        value = tokens[:, :action_len]
        return (
            self.head(value, pre_state["t"])
            if self.options["decoder_type"] == "time_modulated"
            else self.action_decoder(value)
        )

    def append_state_tokens(
        self,
        pre_state: Dict[str, Any],
        state_tokens: torch.Tensor,
    ) -> Dict[str, Any]:
        if self.preset == "staged":
            return self._staged_append_state_tokens(pre_state, state_tokens)
        action_tokens = pre_state["tokens"]
        if state_tokens.ndim != 3 or state_tokens.shape[0] != action_tokens.shape[0]:
            raise ValueError(
                "State tokens must be [B,S,D] with the same batch size as action tokens."
            )
        if state_tokens.shape[2] != self.hidden_dim:
            raise ValueError(
                f"State token width must be {self.hidden_dim}, got {state_tokens.shape[2]}."
            )
        state_len = int(state_tokens.shape[1])
        action_len = int(action_tokens.shape[1])
        total_len = state_len + action_len
        if total_len > self.freqs.shape[0]:
            raise ValueError(
                f"State/action token length {total_len} exceeds RoPE cache {self.freqs.shape[0]}."
            )

        zero_timestep = pre_state["t"].new_zeros(action_tokens.shape[0])
        state_t = self.time_embedding(self._time_features(zero_timestep, self.freq_dim))
        state_t_mod = self.time_projection(state_t).unflatten(1, (6, self.hidden_dim))
        action_t_mod = pre_state["t_mod"]
        pre_state["tokens"] = torch.cat([action_tokens, state_tokens], dim=1)
        pre_state["freqs"] = (
            self.freqs[:total_len].view(total_len, 1, -1).to(action_tokens.device)
        )
        if "rope" in pre_state:
            pre_state["rope"] = prepare_rope(
                pre_state["freqs"], action_tokens.device, action_tokens.dtype
            )
        pre_state["t_mod"] = torch.cat(
            [
                action_t_mod.expand(-1, action_len, -1, -1),
                state_t_mod[:, None].expand(-1, state_len, -1, -1),
            ],
            dim=1,
        )
        pre_state["meta"].update(
            {"state_len": state_len, "action_len": action_len, "seq_len": total_len}
        )
        return pre_state

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor | None = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pre_state = self.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        return self.forward_prepared(pre_state)

    @staticmethod
    def build_action_ids(
        batch_size: int, seq_len: int, *, device, dtype
    ) -> torch.Tensor:
        ids = torch.zeros(batch_size, seq_len, 4, device=device, dtype=dtype)
        ids[..., 0] = 2.0
        ids[..., 1] = torch.arange(seq_len, device=device, dtype=dtype)[None]
        return ids

    def _staged_pre_dit(
        self,
        action_tokens,
        timestep,
        context=None,
        context_mask=None,
        context_is_projected=False,
        cross_kv_cache=None,
    ) -> dict[str, Any]:
        if action_tokens.ndim != 3:
            raise ValueError("Action tokens must be [B,T,D]")
        batch_size, seq_len, action_dim = action_tokens.shape
        if timestep.shape != (batch_size,):
            raise ValueError("Action timestep must contain one value per batch element")
        if self.cross_attention and (
            context is None or context.ndim != 3 or context.shape[0] != batch_size
        ):
            raise ValueError("cross_attention requires a matching [B,L,D] context")
        if context_mask is not None and (
            context is None or context_mask.shape != context.shape[:2]
        ):
            raise ValueError("Context mask must match context [B,L]")
        if action_dim != self.action_dim or seq_len > self.max_action_horizon:
            raise ValueError(
                f"Invalid staged action shape {tuple(action_tokens.shape)}"
            )
        tokens = self.action_encoder(action_tokens)
        vec = self.time_in(self._time_features(timestep, 256)).to(dtype=tokens.dtype)
        return {
            "tokens": tokens,
            "ids": self.build_action_ids(
                batch_size, seq_len, device=tokens.device, dtype=tokens.dtype
            ),
            "t_mod": {
                "vec": vec,
                "double_img": self.double_stream_modulation_img(vec),
                "single": self.single_stream_modulation(vec)[0],
            },
            "context": (
                (context if context_is_projected else self.project_context(context))
                if self.cross_attention
                else None
            ),
            "context_mask": prepare_key_padding_mask(context_mask),
            "cross_kv_cache": cross_kv_cache,
            "meta": {"batch_size": batch_size, "seq_len": seq_len},
        }

    def _staged_post_dit(
        self, tokens: torch.Tensor, pre_state: dict[str, Any]
    ) -> torch.Tensor:
        action_len = int(pre_state.get("meta", {}).get("action_len", tokens.shape[1]))
        value = tokens[:, :action_len]
        return (
            self.head(value, pre_state["t_mod"]["vec"])
            if self.options["decoder_type"] == "time_modulated"
            else self.action_decoder(value)
        )

    @staticmethod
    def _concat_modulation(
        left: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        right: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        action_len: int,
        state_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(
            torch.cat(
                [action.expand(-1, action_len, -1), state.expand(-1, state_len, -1)],
                dim=1,
            )
            for action, state in zip(left, right)
        )

    def _staged_append_state_tokens(
        self,
        pre_state: dict[str, Any],
        state_tokens: torch.Tensor,
    ) -> dict[str, Any]:
        action_tokens = pre_state["tokens"]
        if state_tokens.ndim != 3 or state_tokens.shape[0] != action_tokens.shape[0]:
            raise ValueError(
                "State tokens must be [B,S,D] with the same batch size as action tokens."
            )
        if state_tokens.shape[2] != self.hidden_dim:
            raise ValueError(
                f"State token width must be {self.hidden_dim}, got {state_tokens.shape[2]}."
            )
        state_len = int(state_tokens.shape[1])
        action_len = int(action_tokens.shape[1])
        total_len = state_len + action_len

        zero_timestep = pre_state["t_mod"]["vec"].new_zeros(action_tokens.shape[0])
        zero_vec = self.time_in(self._time_features(zero_timestep, 256)).to(
            dtype=action_tokens.dtype
        )
        state_mod = {
            "double_img": self.double_stream_modulation_img(zero_vec),
            "single": self.single_stream_modulation(zero_vec)[0],
        }
        action_mod = pre_state["t_mod"]
        pre_state["tokens"] = torch.cat([action_tokens, state_tokens], dim=1)
        pre_state["ids"] = self.build_action_ids(
            action_tokens.shape[0],
            total_len,
            device=action_tokens.device,
            dtype=pre_state["ids"].dtype,
        )
        pre_state["t_mod"] = {
            "vec": action_mod["vec"],
            "double_img": tuple(
                self._concat_modulation(action, state, action_len, state_len)
                for action, state in zip(
                    action_mod["double_img"], state_mod["double_img"]
                )
            ),
            "single": self._concat_modulation(
                action_mod["single"], state_mod["single"], action_len, state_len
            ),
        }
        pre_state["meta"].update(
            {"state_len": state_len, "action_len": action_len, "seq_len": total_len}
        )
        return pre_state

    def _time_features(self, timestep, width):
        if self.options["time_embedding_type"] == "standard":
            return sinusoidal_embedding_1d(width, timestep)
        import math

        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(width // 2, device=timestep.device, dtype=torch.float32)
            / (width // 2)
        )
        phase = (timestep * 1000.0).float()[:, None] * frequencies[None]
        return torch.cat([phase.cos(), phase.sin()], -1).to(timestep.dtype)

    def _configure_standard(self, options, eps, projector_hidden_dim):

        for block in self.blocks:
            configure_attention(block.self_attn, options, eps, standard=True)
            if options["ffn_activation"] != "gelu":
                block.ffn = make_ffn(
                    self.hidden_dim, self.ffn_dim, options["ffn_activation"], True
                )
            if not self.cross_attention:
                block.cross_attn = None
        if not self.cross_attention:
            self.text_embedding = None
        if options["encoder_type"] == "linear":
            self.action_encoder = nn.Linear(self.action_dim, self.hidden_dim)
        if options["decoder_type"] == "linear":
            self.action_decoder = nn.Linear(self.hidden_dim, self.action_dim)
        elif options["decoder_type"] == "time_modulated":
            self.head = TimeModulatedActionHead(self.hidden_dim, self.action_dim)
            self.action_decoder = None

    def _init_staged(
        self,
        hidden_dim,
        action_dim,
        text_dim,
        num_heads,
        attn_head_dim,
        double_layers,
        single_layers,
        mlp_ratio,
        horizon,
        projector_hidden_dim,
        attention_backend,
        checkpointing,
    ):
        from ..backbone.protocol import BLOCK_PROTOCOL_STAGED

        if min(hidden_dim, action_dim, num_heads, attn_head_dim, horizon) <= 0:
            raise ValueError("ActionDiT dimensions and horizon must be positive")
        if double_layers < 0 or single_layers < 0 or double_layers + single_layers == 0:
            raise ValueError(
                "Staged ActionDiT needs at least one nonnegative stage depth"
            )
        if attn_head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        self.block_protocol = BLOCK_PROTOCOL_STAGED
        self.hidden_dim, self.action_dim, self.text_dim = (
            hidden_dim,
            action_dim,
            text_dim,
        )
        self.num_heads, self.num_kv_heads, self.attn_head_dim = (
            num_heads,
            num_heads,
            attn_head_dim,
        )
        self.double_layers, self.single_layers = double_layers, single_layers
        self.max_action_horizon = horizon
        self.attention_backend = require_attention_backend(attention_backend)
        self.use_gradient_checkpointing = checkpointing
        self.action_encoder = (
            nn.Linear(action_dim, hidden_dim)
            if self.options["encoder_type"] == "linear"
            else ActionEncoder(action_dim, hidden_dim, projector_hidden_dim)
        )
        self.time_in = MLPEmbedder(256, hidden_dim, disable_bias=True)
        self.double_stream_modulation_img = Modulation(
            hidden_dim, double=True, disable_bias=True
        )
        self.single_stream_modulation = Modulation(
            hidden_dim, double=False, disable_bias=True
        )
        self.double_blocks = nn.ModuleList(
            [
                SequentialActionBlock(
                    hidden_dim,
                    num_heads,
                    attn_head_dim,
                    mlp_ratio,
                    options=self.options,
                )
                for _ in range(double_layers)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                ParallelActionBlock(
                    hidden_dim,
                    num_heads,
                    attn_head_dim,
                    mlp_ratio,
                    options=self.options,
                )
                for _ in range(single_layers)
            ]
        )
        # A single registration avoids duplicate tensor paths in checkpoints.
        self.blocks = tuple(self.double_blocks) + tuple(self.single_blocks)
        if self.options["decoder_type"] == "time_modulated":
            self.head = TimeModulatedActionHead(hidden_dim, action_dim)
        else:
            self.action_decoder = (
                nn.Linear(hidden_dim, action_dim)
                if self.options["decoder_type"] == "linear"
                else ActionDecoder(hidden_dim, action_dim, projector_hidden_dim)
            )
        if self.cross_attention:
            if text_dim <= 0:
                raise ValueError("cross_attention requires positive text_dim")
            self.text_embedding = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for block in self.blocks:
                block.cross_attn = ActionCrossAttention(
                    hidden_dim,
                    attn_head_dim,
                    num_heads,
                    attention_backend=attention_backend,
                )
                # New condition paths initially preserve the pretrained residual function.
                nn.init.zeros_(block.cross_attn.o.weight)
                nn.init.zeros_(block.cross_attn.o.bias)

    def prepare_position(self, pre, pe_embedder):
        if self.options["position_encoding"] == "rope_nd":
            return pe_embedder(pre["ids"])

        embed = EmbedND(self.attn_head_dim, 10000, [self.attn_head_dim]).to(
            pre["tokens"].device
        )
        ids = torch.arange(
            pre["tokens"].shape[1],
            device=pre["tokens"].device,
            dtype=pre["tokens"].dtype,
        )
        return embed(ids[None, :, None].expand(pre["tokens"].shape[0], -1, -1))

    def forward_block(self, index, tokens, pre, attention_mask=None, pe_embedder=None):
        if self.preset != "staged":
            cache = pre.get("cross_kv_cache")
            return self.blocks[index](
                tokens,
                pre["context"],
                pre["t_mod"],
                pre.get("rope", pre["freqs"]),
                context_mask=pre["context_mask"],
                self_attn_mask=attention_mask,
                context_kv=None if cache is None else cache[index],
            )
        if pe_embedder is None:
            dims = [self.attn_head_dim // 4] * 4
            if any(d % 2 for d in dims) or sum(dims) != self.attn_head_dim:
                raise ValueError("Default rope_nd needs head dimension divisible by 8")
            pe_embedder = EmbedND(self.attn_head_dim, 2000, dims).to(tokens.device)
        block = self.blocks[index]
        pe = self.prepare_position(pre, pe_embedder)
        mod = (
            pre["t_mod"]["double_img"]
            if index < self.double_layers
            else pre["t_mod"]["single"]
        )
        state = block.prepare_qkv(tokens, pe, mod)
        mixed = run_attention(
            state["q"],
            state["k"],
            state["v"],
            num_heads=self.num_heads,
            attention_mask=attention_mask,
            backend=self.attention_backend,
        )
        output = block.apply_post(mixed, state)
        if self.cross_attention:
            cache = pre.get("cross_kv_cache")
            output = output + block.cross_attn(
                output,
                pre["context"],
                ctx_mask=pre["context_mask"],
                projected_kv=None if cache is None else cache[index],
            )
        return output

    def forward_prepared(self, pre, pe_embedder=None):
        tokens = pre["tokens"]
        for index in range(len(self.blocks)):
            tokens = gradient_checkpoint_forward(
                lambda value, i=index: self.forward_block(
                    i, value, pre, pe_embedder=pe_embedder
                ),
                self.use_gradient_checkpointing and torch.is_grad_enabled(),
                tokens,
            )
        return self.post_dit(tokens, pre)


def resize_weight(value, shape, alpha=True):
    result = _resize_tensor_to_shape(value.float(), shape)
    if alpha and value.ndim >= 2 and value.shape[-1] != shape[-1]:
        result = result * (value.shape[-1] / shape[-1]) ** 0.5
    return result.to(value.dtype)

def _interpolate_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[-1] == new_size:
        return tensor
    flat = tensor.reshape(-1, 1, tensor.shape[-1]).to(torch.float32)
    flat = F.interpolate(flat, size=new_size, mode="linear", align_corners=True)
    return flat.reshape(*tensor.shape[:-1], new_size)

def _resize_tensor_to_shape(
    src: torch.Tensor, target_shape: tuple[int, ...]
) -> torch.Tensor:
    if tuple(src.shape) == tuple(target_shape):
        return src

    out = src.to(torch.float32)
    while out.ndim < len(target_shape):
        out = out.unsqueeze(0)
    while out.ndim > len(target_shape):
        if out.shape[0] != 1:
            raise ValueError(
                f"Cannot reduce tensor rank for resize: src shape={tuple(src.shape)}, target={target_shape}"
            )
        out = out.squeeze(0)

    for dim, new_size in enumerate(target_shape):
        current_size = out.shape[dim]
        if current_size == new_size:
            continue
        # Interpolate one dimension at a time and restore the original axis order.
        perm = [i for i in range(out.ndim) if i != dim] + [dim]
        inv_perm = [0] * out.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        out_perm = out.permute(*perm).contiguous()
        prefix_shape = out_perm.shape[:-1]
        out_perm = _interpolate_last_dim(out_perm, new_size)
        out_perm = out_perm.reshape(*prefix_shape, new_size)
        out = out_perm.permute(*inv_perm).contiguous()

    return out.to(dtype=src.dtype)

def standard_initialization_payload(
    action,
    visual,
    *,
    source_key,
    missing_value,
    source_backbone,
    mapping_version,
    alpha=True,
    repeat_head_norm=False,
):
    source = visual.state_dict()
    output = {}
    counts = dict(copied=0, resized=0, initialized=0)
    for key, target in sorted(action.state_dict().items()):
        if key.startswith(action.ACTION_BACKBONE_SKIP_PREFIXES):
            continue
        mapped = source_key(key)
        value = source.get(mapped) if mapped is not None else None
        if value is None:
            result = missing_value(key, target)
            category = "initialized"
        elif (
            repeat_head_norm
            and (".norm_q.weight" in key or ".norm_k.weight" in key)
            and value.ndim == target.ndim == 1
            and target.numel() % value.numel() == 0
        ):
            result = value.repeat(target.numel() // value.numel())
            category = "resized"
        elif tuple(value.shape) == tuple(target.shape):
            result = value
            category = "copied"
        else:
            result = _resize_tensor_to_shape(value, tuple(target.shape))
            if alpha and value.ndim >= 2 and value.shape[-1] != target.shape[-1]:
                result = (
                    result.float()
                    * (float(value.shape[-1]) / float(target.shape[-1])) ** 0.5
                )
            category = "resized"
        output[key] = result.detach().to(dtype=target.dtype, device="cpu").contiguous()
        counts[category] += 1
    return {
        "policy": {
            "skip_prefixes": list(action.ACTION_BACKBONE_SKIP_PREFIXES),
            "alpha_scaling": bool(alpha),
            "interpolation": "sequential_1d_linear_align_corners_true",
            "source_backbone": source_backbone,
            "mapping_version": mapping_version,
        },
        "backbone_state_dict": output,
        "meta": {
            key: action.resolved_config[key]
            for key in (
                "hidden_dim",
                "ffn_dim",
                "num_layers",
                "num_heads",
                "attn_head_dim",
                "text_dim",
                "freq_dim",
                "eps",
            )
        },
        "initialization": counts,
    }


def policy_action_config(model):
    action = getattr(model, "action_expert", None)
    if action is None:
        action = getattr(model, "action_dit", None)
    return None if action is None else action.resolved_config


def validate_policy_action_config(model, payload):
    if "action_config" in payload and payload["action_config"] != policy_action_config(
        model
    ):
        raise ValueError(
            "Checkpoint ActionDiT configuration does not match the configured policy"
        )
