from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange, repeat
from safetensors.torch import load_file as load_safetensors
from torch import Tensor
from torch.nn import functional as F

from ..protocol import BLOCK_PROTOCOL_STAGED
from ...component.dit import (
    EmbedND,
    MLPEmbedder,
    Modulation,
    QKNorm,
    SiLUActivation,
    apply_rope,
    timestep_embedding,
)


@dataclass
class Flux2Params:
    in_channels: int = 128
    context_in_dim: int = 15360
    hidden_size: int = 6144
    num_heads: int = 48
    depth: int = 8
    depth_single_blocks: int = 48
    axes_dim: list[int] = field(default_factory=lambda: [32, 32, 32, 32])
    theta: int = 2000
    mlp_ratio: float = 3.0
    use_guidance_embed: bool = True


@dataclass
class Klein9BParams:
    in_channels: int = 128
    context_in_dim: int = 12288
    hidden_size: int = 4096
    num_heads: int = 32
    depth: int = 8
    depth_single_blocks: int = 24
    axes_dim: list[int] = field(default_factory=lambda: [32, 32, 32, 32])
    theta: int = 2000
    mlp_ratio: float = 3.0
    use_guidance_embed: bool = False


@dataclass
class Klein4BParams:
    in_channels: int = 128
    context_in_dim: int = 7680
    hidden_size: int = 3072
    num_heads: int = 24
    depth: int = 5
    depth_single_blocks: int = 20
    axes_dim: list[int] = field(default_factory=lambda: [32, 32, 32, 32])
    theta: int = 2000
    mlp_ratio: float = 3.0
    use_guidance_embed: bool = False


class Flux2(nn.Module):
    def __init__(self, params: Flux2Params):
        super().__init__()

        self.in_channels = params.in_channels
        self.out_channels = params.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)
        self.time_in = MLPEmbedder(
            in_dim=256, hidden_dim=self.hidden_size, disable_bias=True
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size, bias=False)

        self.use_guidance_embed = params.use_guidance_embed
        if self.use_guidance_embed:
            self.guidance_in = MLPEmbedder(
                in_dim=256, hidden_dim=self.hidden_size, disable_bias=True
            )

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.double_stream_modulation_img = Modulation(
            self.hidden_size,
            double=True,
            disable_bias=True,
        )
        self.double_stream_modulation_txt = Modulation(
            self.hidden_size,
            double=True,
            disable_bias=True,
        )
        self.single_stream_modulation = Modulation(
            self.hidden_size, double=False, disable_bias=True
        )

        self.final_layer = LastLayer(
            self.hidden_size,
            self.out_channels,
        )

    def forward(
        self,
        x: Tensor,
        x_ids: Tensor,
        timesteps: Tensor,
        ctx: Tensor,
        ctx_ids: Tensor,
        guidance: Tensor | None,
    ):
        num_txt_tokens = ctx.shape[1]

        timestep_emb = timestep_embedding(timesteps, 256)
        vec = self.time_in(timestep_emb)
        if self.use_guidance_embed:
            guidance_emb = timestep_embedding(guidance, 256)
            vec = vec + self.guidance_in(guidance_emb)

        double_block_mod_img = self.double_stream_modulation_img(vec)
        double_block_mod_txt = self.double_stream_modulation_txt(vec)
        single_block_mod, _ = self.single_stream_modulation(vec)

        img = self.img_in(x)
        txt = self.txt_in(ctx)

        pe_x = self.pe_embedder(x_ids)
        pe_ctx = self.pe_embedder(ctx_ids)

        for block in self.double_blocks:
            img, txt, _ = block.forward_kv_extract(
                img,
                txt,
                pe_x,
                pe_ctx,
                double_block_mod_img,
                double_block_mod_txt,
                num_ref_tokens=0,
            )

        img = torch.cat((txt, img), dim=1)
        pe = torch.cat((pe_ctx, pe_x), dim=2)

        for block in self.single_blocks:
            img, _ = block.forward_kv_extract(
                img,
                pe,
                single_block_mod,
                num_txt_tokens,
                num_ref_tokens=0,
            )

        img = img[:, num_txt_tokens:, ...]

        img = self.final_layer(img, vec)
        return img

    def forward_kv_extract(
        self,
        x: Tensor,
        x_ids: Tensor,
        timesteps: Tensor,
        ctx: Tensor,
        ctx_ids: Tensor,
        guidance: Tensor | None,
        x_seq_concat: Tensor,
        x_seq_concat_ids: Tensor,
        ref_fixed_timestep: float = 0.0,
    ) -> tuple[Tensor, dict]:
        """
        First denoising step with reference tokens. Runs full forward pass and
        extracts KV cache for reference tokens to reuse on subsequent steps.

        Input x layout becomes [ref, img] after concatenation.
        Returns (prediction, kv_cache).
        """
        num_txt_tokens = ctx.shape[1]
        num_ref_tokens = x_seq_concat.shape[1]

        x = torch.cat([x_seq_concat, x], dim=1)
        x_ids = torch.cat([x_seq_concat_ids, x_ids], dim=1)

        # Timestep embeddings
        timestep_emb = timestep_embedding(timesteps, 256)
        vec = self.time_in(timestep_emb)
        ref_vec = self.time_in(
            timestep_embedding(torch.full_like(timesteps, ref_fixed_timestep), 256)
        )

        if self.use_guidance_embed:
            guidance_emb = timestep_embedding(guidance, 256)
            vec = vec + self.guidance_in(guidance_emb)
            ref_vec = ref_vec + self.guidance_in(guidance_emb)

        # Modulations
        double_block_mod_img = self.double_stream_modulation_img(vec)
        double_block_mod_txt = self.double_stream_modulation_txt(vec)
        single_block_mod, _ = self.single_stream_modulation(vec)

        ref_double_mod = self.double_stream_modulation_img(ref_vec)
        ref_single_mod, _ = self.single_stream_modulation(ref_vec)

        img = self.img_in(x)
        txt = self.txt_in(ctx)

        pe_x = self.pe_embedder(x_ids)
        pe_ctx = self.pe_embedder(ctx_ids)

        # Blend double block modulations: [ref_mod, img_mod]
        L_img = img.shape[1]
        double_block_mod_img = _blend_double_mods(
            double_block_mod_img, ref_double_mod, num_ref_tokens, L_img
        )

        double_block_cache = []
        for block in self.double_blocks:
            img, txt, cache = block.forward_kv_extract(
                img,
                txt,
                pe_x,
                pe_ctx,
                double_block_mod_img,
                double_block_mod_txt,
                num_ref_tokens,
            )
            double_block_cache.append(cache)

        img = torch.cat((txt, img), dim=1)
        pe = torch.cat((pe_ctx, pe_x), dim=2)

        # Blend single block modulations: [txt_mod, ref_mod, img_mod]
        L = img.shape[1]
        single_block_mod = _blend_single_mods(
            single_block_mod, ref_single_mod, num_txt_tokens, num_ref_tokens, L
        )

        single_block_cache = []
        for block in self.single_blocks:
            img, cache = block.forward_kv_extract(
                img,
                pe,
                single_block_mod,
                num_txt_tokens,
                num_ref_tokens,
            )
            single_block_cache.append(cache)

        # Strip txt + ref tokens
        img = img[:, num_txt_tokens + num_ref_tokens :, ...]
        img = self.final_layer(img, vec)

        kv_cache = {
            "double_blocks": double_block_cache,
            "single_blocks": single_block_cache,
            "num_ref_tokens": num_ref_tokens,
        }
        return img, kv_cache

    def forward_kv_cached(
        self,
        x: Tensor,
        x_ids: Tensor,
        timesteps: Tensor,
        ctx: Tensor,
        ctx_ids: Tensor,
        guidance: Tensor | None,
        kv_cache: dict,
    ) -> Tensor:
        """
        Subsequent denoising steps using cached KV for reference tokens.
        Input x has layout [img] only (no ref tokens).
        """
        num_txt_tokens = ctx.shape[1]

        timestep_emb = timestep_embedding(timesteps, 256)
        vec = self.time_in(timestep_emb)

        if self.use_guidance_embed:
            guidance_emb = timestep_embedding(guidance, 256)
            vec = vec + self.guidance_in(guidance_emb)

        double_block_mod_img = self.double_stream_modulation_img(vec)
        double_block_mod_txt = self.double_stream_modulation_txt(vec)
        single_block_mod, _ = self.single_stream_modulation(vec)

        img = self.img_in(x)
        txt = self.txt_in(ctx)

        pe_x = self.pe_embedder(x_ids)
        pe_ctx = self.pe_embedder(ctx_ids)

        for i, block in enumerate(self.double_blocks):
            img, txt = block.forward_kv_cached(
                img,
                txt,
                pe_x,
                pe_ctx,
                double_block_mod_img,
                double_block_mod_txt,
                kv_cache["double_blocks"][i],
            )

        img = torch.cat((txt, img), dim=1)
        pe = torch.cat((pe_ctx, pe_x), dim=2)

        for i, block in enumerate(self.single_blocks):
            img = block.forward_kv_cached(
                img,
                pe,
                single_block_mod,
                num_txt_tokens,
                kv_cache["single_blocks"][i],
            )

        # Strip txt tokens only (no ref tokens in sequence)
        img = img[:, num_txt_tokens:, ...]
        img = self.final_layer(img, vec)
        return img


def _blend_mod_triple(img_m: tuple, ref_m: tuple, num_ref: int, seq_len: int) -> tuple:
    """Blend a (shift, scale, gate) triple: first num_ref positions get ref_m, rest get img_m."""
    blended = []
    for im, rm in zip(img_m, ref_m):
        if im.ndim == 2:
            im = im[:, None, :]
            rm = rm[:, None, :]
        B = im.shape[0]
        blended.append(
            torch.cat(
                [rm.expand(B, num_ref, -1), im.expand(B, seq_len, -1)[:, num_ref:, :]],
                dim=1,
            )
        )
    return tuple(blended)


def _blend_double_mods(img_mod, ref_mod, num_ref: int, seq_len: int):
    """Blend double block modulations (mod1, mod2) for [ref, img] layout."""
    img_mod1, img_mod2 = img_mod
    ref_mod1, ref_mod2 = ref_mod
    return (
        _blend_mod_triple(img_mod1, ref_mod1, num_ref, seq_len),
        _blend_mod_triple(img_mod2, ref_mod2, num_ref, seq_len),
    )


def _blend_single_mods(single_mod, ref_mod, num_txt: int, num_ref: int, seq_len: int):
    """Blend single block modulations for [txt, ref, img] layout."""
    blended = []
    for im, rm in zip(single_mod, ref_mod):
        if im.ndim == 2:
            im = im[:, None, :]
            rm = rm[:, None, :]
        B = im.shape[0]
        im_expanded = im.expand(B, seq_len, -1)
        rm_expanded = rm.expand(B, num_ref, -1)
        blended.append(
            torch.cat(
                [
                    im_expanded[:, :num_txt, :],
                    rm_expanded,
                    im_expanded[:, num_txt + num_ref :, :],
                ],
                dim=1,
            )
        )
    return tuple(blended)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)

        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim, bias=False)


class LastLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        )

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        mod = self.adaLN_modulation(vec)
        shift, scale = mod.chunk(2, dim=-1)
        if shift.ndim == 2:
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        x = (1 + scale) * self.norm_final(x) + shift
        x = self.linear(x)
        return x


class SingleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = head_dim**-0.5
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_mult_factor = 2

        self.linear1 = nn.Linear(
            hidden_size,
            hidden_size * 3 + self.mlp_hidden_dim * self.mlp_mult_factor,
            bias=False,
        )

        self.linear2 = nn.Linear(
            hidden_size + self.mlp_hidden_dim, hidden_size, bias=False
        )

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = SiLUActivation()

    def _qkv(self, x: Tensor, mod: tuple[Tensor, Tensor, Tensor]):
        mod_shift, mod_scale, mod_gate = mod
        x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift

        qkv, mlp = torch.split(
            self.linear1(x_mod),
            [3 * self.hidden_size, self.mlp_hidden_dim * self.mlp_mult_factor],
            dim=-1,
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        return q, k, v, mlp, mod_gate

    def _out(self, x: Tensor, attn: Tensor, mlp: Tensor, mod_gate: Tensor) -> Tensor:
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod_gate * output

    def forward_kv_extract(
        self,
        x: Tensor,
        pe: Tensor,
        mod: tuple[Tensor, Tensor, Tensor],
        num_txt_tokens: int,
        num_ref_tokens: int,
    ) -> tuple[Tensor, dict]:
        """Forward with causal attention. Extracts and returns ref KV cache."""
        q, k, v, mlp, mod_gate = self._qkv(x, mod)
        q, k = apply_rope(q, k, pe)

        ref_start = num_txt_tokens
        ref_end = num_txt_tokens + num_ref_tokens
        cache = {
            "k_ref": k[:, :, ref_start:ref_end, :].clone(),
            "v_ref": v[:, :, ref_start:ref_end, :].clone(),
        }

        attn = causal_attn_fn(q, k, v, num_txt_tokens, num_ref_tokens)
        return self._out(x, attn, mlp, mod_gate), cache

    def forward_kv_cached(
        self,
        x: Tensor,
        pe: Tensor,
        mod: tuple[Tensor, Tensor, Tensor],
        num_txt_tokens: int,
        kv_cache: dict,
    ) -> Tensor:
        """Forward using cached ref KV. Input x has layout [txt, img] (no ref)."""
        q, k, v, mlp, mod_gate = self._qkv(x, mod)
        q, k = apply_rope(q, k, pe)
        num_ref_tokens = kv_cache["k_ref"].shape[2]
        attn = causal_attn_fn(q, k, v, num_txt_tokens, num_ref_tokens, kv_cache)
        return self._out(x, attn, mlp, mod_gate)


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
    ):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        assert (
            hidden_size % num_heads == 0
        ), f"{hidden_size=} must be divisible by {num_heads=}"

        self.hidden_size = hidden_size
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_mult_factor = 2

        self.img_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
        )

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim * self.mlp_mult_factor, bias=False),
            SiLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        )

        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
        )

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(
                hidden_size,
                mlp_hidden_dim * self.mlp_mult_factor,
                bias=False,
            ),
            SiLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        )

    def _prepare_qkv(self, img, txt, pe, pe_ctx, mod_img, mod_txt):
        img_mod1, img_mod2 = mod_img
        txt_mod1, txt_mod2 = mod_txt

        img_mod1_shift, img_mod1_scale, img_mod1_gate = img_mod1
        img_mod2_shift, img_mod2_scale, img_mod2_gate = img_mod2
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate = txt_mod1
        txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = txt_mod2

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1_scale) * img_modulated + img_mod1_shift

        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1_scale) * txt_modulated + txt_mod1_shift

        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        num_txt_tokens = txt_q.shape[2]
        pe_full = torch.cat((pe_ctx, pe), dim=2)

        mods = (
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        )

        return q, k, v, pe_full, num_txt_tokens, mods

    def _apply_residuals(self, img, txt, img_attn, txt_attn, mods):
        (
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = mods

        img = img + img_mod1_gate * self.img_attn.proj(img_attn)
        img = img + img_mod2_gate * self.img_mlp(
            (1 + img_mod2_scale) * (self.img_norm2(img)) + img_mod2_shift
        )

        txt = txt + txt_mod1_gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2_gate * self.txt_mlp(
            (1 + txt_mod2_scale) * (self.txt_norm2(txt)) + txt_mod2_shift
        )
        return img, txt

    def forward_kv_extract(
        self,
        img: Tensor,
        txt: Tensor,
        pe: Tensor,
        pe_ctx: Tensor,
        mod_img: tuple[Tensor, Tensor],
        mod_txt: tuple[Tensor, Tensor],
        num_ref_tokens: int,
    ) -> tuple[Tensor, Tensor, dict]:
        """Forward with causal attention. img has layout [ref, img]. Extracts ref KV cache."""
        q, k, v, pe_full, num_txt_tokens, mods = self._prepare_qkv(
            img, txt, pe, pe_ctx, mod_img, mod_txt
        )
        q, k = apply_rope(q, k, pe_full)

        ref_start = num_txt_tokens
        ref_end = num_txt_tokens + num_ref_tokens
        cache = {
            "k_ref": k[:, :, ref_start:ref_end, :].clone(),
            "v_ref": v[:, :, ref_start:ref_end, :].clone(),
        }

        attn = causal_attn_fn(q, k, v, num_txt_tokens, num_ref_tokens)
        txt_attn, img_attn = attn[:, :num_txt_tokens], attn[:, num_txt_tokens:]
        img, txt = self._apply_residuals(img, txt, img_attn, txt_attn, mods)
        return img, txt, cache

    def forward_kv_cached(
        self,
        img: Tensor,
        txt: Tensor,
        pe: Tensor,
        pe_ctx: Tensor,
        mod_img: tuple[Tensor, Tensor],
        mod_txt: tuple[Tensor, Tensor],
        kv_cache: dict,
    ) -> tuple[Tensor, Tensor]:
        """Forward using cached ref KV. img has layout [img] only (no ref)."""
        q, k, v, pe_full, num_txt_tokens, mods = self._prepare_qkv(
            img, txt, pe, pe_ctx, mod_img, mod_txt
        )
        q, k = apply_rope(q, k, pe_full)

        num_ref_tokens = kv_cache["k_ref"].shape[2]
        attn = causal_attn_fn(q, k, v, num_txt_tokens, num_ref_tokens, kv_cache)
        txt_attn, img_attn = attn[:, :num_txt_tokens], attn[:, num_txt_tokens:]
        return self._apply_residuals(img, txt, img_attn, txt_attn, mods)


def causal_attn_fn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    num_txt_tokens: int,
    num_ref_tokens: int,
    kv_cache: dict | None = None,
) -> Tensor:
    """
    Causal attention where reference tokens only attend to themselves.

    Without cache: layout is [txt, ref, img]. txt+img attend to all, ref self-attends.
    With cache: layout is [txt, img]. Cached ref K/V injected into attention.
    """
    if kv_cache is not None:
        k_ref = kv_cache["k_ref"]
        v_ref = kv_cache["v_ref"]

        q_txt = q[:, :, :num_txt_tokens, :]
        q_img = q[:, :, num_txt_tokens:, :]
        k_txt = k[:, :, :num_txt_tokens, :]
        v_txt = v[:, :, :num_txt_tokens, :]
        k_img = k[:, :, num_txt_tokens:, :]
        v_img = v[:, :, num_txt_tokens:, :]

        q_txt_img = torch.cat([q_txt, q_img], dim=2)
        k_all = torch.cat([k_txt, k_ref, k_img], dim=2)
        v_all = torch.cat([v_txt, v_ref, v_img], dim=2)
        out = F.scaled_dot_product_attention(q_txt_img, k_all, v_all, is_causal=False)

    else:
        ref_start = num_txt_tokens
        ref_end = num_txt_tokens + num_ref_tokens

        q_txt = q[:, :, :ref_start, :]
        q_ref = q[:, :, ref_start:ref_end, :]
        q_img = q[:, :, ref_end:, :]
        k_txt = k[:, :, :ref_start, :]
        v_txt = v[:, :, :ref_start, :]
        k_ref = k[:, :, ref_start:ref_end, :]
        v_ref = v[:, :, ref_start:ref_end, :]
        k_img = k[:, :, ref_end:, :]
        v_img = v[:, :, ref_end:, :]

        # txt+img attend to all keys
        q_txt_img = torch.cat([q_txt, q_img], dim=2)
        k_all = torch.cat([k_txt, k_ref, k_img], dim=2)
        v_all = torch.cat([v_txt, v_ref, v_img], dim=2)
        attn_txt_img = F.scaled_dot_product_attention(
            q_txt_img, k_all, v_all, is_causal=False
        )
        attn_txt = attn_txt_img[:, :, :ref_start, :]
        attn_img = attn_txt_img[:, :, ref_start:, :]

        # ref only attends to itself
        attn_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=False)

        out = torch.cat([attn_txt, attn_ref, attn_img], dim=2)

    return rearrange(out, "b h n d -> b n (h d)")


class Flux2VideoDiT(nn.Module):

    scheduler_family = "continuous_flow"

    backbone_name = "flux2"
    block_protocol = BLOCK_PROTOCOL_STAGED
    video_attention_mask_mode = "flux2_reference_causal"
    fuse_vae_embedding_in_latents = False
    visual_target = "endpoint"
    evaluation_metric_frames = (-1,)
    patch_size = (1, 1, 1)
    freq_dim = 256

    def __init__(
        self, transformer: nn.Module, variant: str, attention_backend: str = "sdpa"
    ):
        super().__init__()
        self.transformer = transformer
        self.variant = str(variant)
        self.hidden_dim = int(transformer.hidden_size)
        self.context_dim = self.hidden_dim
        self.num_heads = int(transformer.num_heads)
        self.num_kv_heads = self.num_heads
        self.attn_head_dim = self.hidden_dim // self.num_heads
        self.double_layers = len(self.double_blocks)
        self.single_layers = len(self.single_blocks)
        self.attention_backend = str(attention_backend)
        self.use_gradient_checkpointing = bool(
            getattr(transformer, "gradient_checkpointing", False)
        )

    lora_block_prefixes = ("transformer.double_blocks.", "transformer.single_blocks.")

    @property
    def double_blocks(self):
        return self.transformer.double_blocks

    @property
    def single_blocks(self):
        return self.transformer.single_blocks

    @property
    def blocks(self):
        return list(self.double_blocks) + list(self.single_blocks)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        variant: str = "klein-base-4b",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        attention_backend: str = "sdpa",
        skip_dit_load_from_pretrain: bool = False,
    ) -> "Flux2VideoDiT":

        key = str(variant).lower().replace("_", "-")
        if key in {"klein-base-4b", "flux.2-klein-base-4b", "4b", "base-4b"}:
            params = Klein4BParams()
        elif key in {"klein-base-9b", "flux.2-klein-base-9b", "9b", "base-9b"}:
            params = Klein9BParams()
        else:
            raise ValueError(f"Unsupported FLUX.2 Klein variant: {variant!r}")

        if skip_dit_load_from_pretrain:
            transformer = Flux2(params).to(device=device, dtype=torch_dtype)
        else:
            with torch.device("meta"):
                transformer = Flux2(params).to(torch_dtype)
            state_dict = load_safetensors(str(model_path), device=str(device))
            transformer.load_state_dict(state_dict, strict=True, assign=True)
            transformer = transformer.to(device=device, dtype=torch_dtype)
        return cls(transformer, variant=key, attention_backend=attention_backend)

    @staticmethod
    def build_txt_ids(batch_size: int, seq_len: int, *, device, dtype) -> torch.Tensor:
        ids = torch.zeros(batch_size, seq_len, 4, device=device, dtype=dtype)
        ids[..., 3] = torch.arange(seq_len, device=device, dtype=dtype)[None]
        return ids

    @staticmethod
    def build_img_ids(
        batch_size: int,
        token_height: int,
        token_width: int,
        *,
        time_value: float,
        device,
        dtype,
    ) -> torch.Tensor:
        ids = torch.zeros(token_height, token_width, 4, device=device, dtype=dtype)
        ids[..., 0] = float(time_value)
        ids[..., 1] = torch.arange(token_height, device=device, dtype=dtype)[:, None]
        ids[..., 2] = torch.arange(token_width, device=device, dtype=dtype)[None]
        return repeat(ids, "h w c -> b (h w) c", b=batch_size)

    @staticmethod
    def pack_latents(latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 4:
            raise ValueError(f"`latents` must be [B,C,H,W], got {tuple(latents.shape)}")
        return rearrange(latents, "b c h w -> b (h w) c")

    @staticmethod
    def unpack_latents(
        tokens: torch.Tensor, latent_height: int, latent_width: int
    ) -> torch.Tensor:
        return rearrange(
            tokens, "b (h w) c -> b c h w", h=latent_height, w=latent_width
        )

    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        ref_image_hidden_states: torch.Tensor | None = None,
        target_img_ids: torch.Tensor | None = None,
        ref_img_ids: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if x.ndim != 3 or context.ndim != 3:
            raise ValueError("FLUX.2 x/context must be [B,N,C]/[B,L,D].")
        batch_size, target_len = int(x.shape[0]), int(x.shape[1])
        cond_len = (
            0
            if ref_image_hidden_states is None
            else int(ref_image_hidden_states.shape[1])
        )
        if target_img_ids is None:
            raise ValueError("`target_img_ids` is required for FLUX.2.")
        if ref_image_hidden_states is not None and ref_img_ids is None:
            raise ValueError("`ref_img_ids` is required with reference image tokens.")

        transformer = self.transformer
        img_tokens = (
            x
            if ref_image_hidden_states is None
            else torch.cat([ref_image_hidden_states, x], 1)
        )
        img_ids = (
            target_img_ids
            if ref_img_ids is None
            else torch.cat([ref_img_ids, target_img_ids], 1)
        )

        vec = transformer.time_in(timestep_embedding(timestep, 256))
        zero_vec = transformer.time_in(
            timestep_embedding(torch.zeros_like(timestep), 256)
        )
        txt = transformer.txt_in(context)
        img = transformer.img_in(img_tokens)
        txt_ids = self.build_txt_ids(
            batch_size, txt.shape[1], device=img.device, dtype=img_ids.dtype
        )
        txt_pe = transformer.pe_embedder(txt_ids)
        img_pe = transformer.pe_embedder(img_ids)
        text_mask = (
            torch.ones(batch_size, txt.shape[1], device=img.device, dtype=torch.bool)
            if context_mask is None
            else context_mask.to(device=img.device, dtype=torch.bool)
        )

        def join_mod(clean, noisy, clean_len, noisy_len):
            return tuple(
                torch.cat([c.expand(-1, clean_len, -1), n.expand(-1, noisy_len, -1)], 1)
                for c, n in zip(clean, noisy)
            )

        double_img = tuple(
            join_mod(c, n, cond_len, target_len)
            for c, n in zip(
                transformer.double_stream_modulation_img(zero_vec),
                transformer.double_stream_modulation_img(vec),
            )
        )
        single = join_mod(
            transformer.single_stream_modulation(zero_vec)[0],
            transformer.single_stream_modulation(vec)[0],
            txt.shape[1] + cond_len,
            target_len,
        )
        return {
            "tokens": {"txt": txt, "img": img},
            "freqs": {"txt": txt_pe, "img": img_pe},
            "t_mod": {
                "vec": vec,
                "double_img": double_img,
                "double_txt": transformer.double_stream_modulation_txt(zero_vec),
                "single": single,
            },
            "context": None,
            "context_mask": text_mask,
            "txt_len": int(txt.shape[1]),
            "target_len": target_len,
            "cond_len": cond_len,
            "text_mask": text_mask,
            "meta": {
                "batch_size": batch_size,
                "txt_len": int(txt.shape[1]),
                "target_len": target_len,
                "cond_len": cond_len,
            },
        }

    def post_dit(
        self, tokens: dict[str, torch.Tensor], pre_state: dict[str, Any]
    ) -> torch.Tensor:
        img = tokens["img"]
        start = int(pre_state["cond_len"])
        target = img[:, start : start + int(pre_state["target_len"])]
        return self.transformer.final_layer(target, pre_state["t_mod"]["vec"])

    def prepare_attention(self, index, tokens, pre):
        """Prepare flattened Q/K/V without imposing an architecture's mask."""

        flat = lambda value: value.transpose(1, 2).flatten(2)
        txt, img = tokens["txt"], tokens["img"]
        if index < self.double_layers:
            block = self.double_blocks[index]
            q, k, v, pe, text_len, residual = block._prepare_qkv(
                img,
                txt,
                pre["freqs"]["img"],
                pre["freqs"]["txt"],
                pre["t_mod"]["double_img"],
                pre["t_mod"]["double_txt"],
            )
            q, k = apply_rope(q, k, pe)
            state = (txt, img, text_len, residual)
        else:
            block = self.single_blocks[index - self.double_layers]
            value = torch.cat([txt, img], dim=1)
            q, k, v, mlp, gate = block._qkv(value, pre["t_mod"]["single"])
            q, k = apply_rope(
                q, k, torch.cat([pre["freqs"]["txt"], pre["freqs"]["img"]], dim=2)
            )
            state = (value, txt.shape[1], mlp, gate)
        return flat(q), flat(k), flat(v), state

    def finish_attention(self, index, attention, state):
        if index < self.double_layers:
            txt, img, count, residual = state
            img, txt = self.double_blocks[index]._apply_residuals(
                img, txt, attention[:, count:], attention[:, :count], residual
            )
        else:
            value, count, mlp, gate = state
            value = self.single_blocks[index - self.double_layers]._out(
                value, attention, mlp, gate
            )
            txt, img = value[:, :count], value[:, count:]
        return {"txt": txt, "img": img}

    def forward_block(self, index, tokens, pre, attention_mask=None):
        from ...component.attention import run_attention

        q, k, v, state = self.prepare_attention(index, tokens, pre)
        attention = run_attention(
            q,
            k,
            v,
            num_heads=self.num_heads,
            attention_mask=attention_mask,
            backend=self.attention_backend,
        )
        return self.finish_attention(index, attention, state)

    def visual_features(self, tokens):
        return tokens["img"]

    def pre_unified_dit(
        self, pre, action_tokens, state_tokens, timestep_action, state_position
    ):
        """Put action/state values through the same image-side transformer weights."""

        batch, action_len = action_tokens.shape[:2]
        state_len = state_tokens.shape[1]
        image_len = pre["tokens"]["img"].shape[1]
        from ...component.action_dit import ActionDiT

        ids = ActionDiT.build_action_ids(
            batch,
            action_len + state_len,
            device=action_tokens.device,
            dtype=action_tokens.dtype,
        )
        ids[:, action_len:, 0] = 3
        if state_position == "context":
            # State is a clean condition, placed before visual tokens in the text stream.
            pre["tokens"]["txt"] = torch.cat([pre["tokens"]["txt"], state_tokens], 1)
            state_ids = ids[:, action_len:]
            pre["freqs"]["txt"] = torch.cat(
                [pre["freqs"]["txt"], self.transformer.pe_embedder(state_ids)], 2
            )
            pre["text_mask"] = torch.cat(
                [
                    pre["text_mask"],
                    torch.ones(batch, state_len, device=ids.device, dtype=torch.bool),
                ],
                1,
            )
            pre["txt_len"] += state_len
            extra, ids = action_tokens, ids[:, :action_len]
            state_seq_len = 0
        else:
            extra = torch.cat([action_tokens, state_tokens], 1)
            state_seq_len = state_len
        pre["tokens"]["img"] = torch.cat([pre["tokens"]["img"], extra], 1)
        pre["freqs"]["img"] = torch.cat(
            [pre["freqs"]["img"], self.transformer.pe_embedder(ids)], 2
        )
        vec = self.transformer.time_in(timestep_embedding(timestep_action, 256))
        zero = self.transformer.time_in(
            timestep_embedding(torch.zeros_like(timestep_action), 256)
        )

        def append(base, noisy, clean):
            return tuple(
                torch.cat(
                    [b, n.expand(-1, action_len, -1), c.expand(-1, state_seq_len, -1)],
                    1,
                )
                for b, n, c in zip(base, noisy, clean)
            )

        pre["t_mod"]["double_img"] = tuple(
            append(base, n, c)
            for base, n, c in zip(
                pre["t_mod"]["double_img"],
                self.transformer.double_stream_modulation_img(vec),
                self.transformer.double_stream_modulation_img(zero),
            )
        )
        pre["t_mod"]["single"] = append(
            pre["t_mod"]["single"],
            self.transformer.single_stream_modulation(vec)[0],
            self.transformer.single_stream_modulation(zero)[0],
        )
        if state_position == "context":
            # Single-stage text condition gained state tokens after preparation.
            count = pre["txt_len"] - state_len
            pre["t_mod"]["single"] = tuple(
                torch.cat([m[:, :count], c.expand(-1, state_len, -1), m[:, count:]], 1)
                for m, c in zip(
                    pre["t_mod"]["single"],
                    self.transformer.single_stream_modulation(zero)[0],
                )
            )
        pre["action_slice"] = slice(image_len, image_len + action_len)
        return pre

    def post_unified_dit(self, tokens, pre):
        return {
            "video": self.post_dit(tokens, pre),
            "action_tokens": tokens["img"][:, pre["action_slice"]],
        }
