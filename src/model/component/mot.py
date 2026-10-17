from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..helpers.gradient import gradient_checkpoint_forward
from .attention import (
    StructuredAttentionMask,
    normalize_attention_backend,
    run_attention,
)
from ..backbone.protocol import (
    BLOCK_PROTOCOL_STAGED,
    BLOCK_PROTOCOL_STANDARD,
    SUPPORTED_BLOCK_PROTOCOLS,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)


def _modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x * (1 + scale) + shift


def _rope_apply(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    b, s, _ = x.shape
    x = x.view(b, s, num_heads, -1)
    x_complex = torch.view_as_complex(x.float().reshape(b, s, num_heads, -1, 2))
    freqs = freqs.to(device=x.device)
    freqs = freqs.squeeze(1)
    out = torch.view_as_real(x_complex * freqs[None, :, None]).flatten(3)
    return out.to(x.dtype).reshape(b, s, -1)


class MoT(nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
    ):
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if "video" not in mixtures or "action" not in mixtures:
            raise ValueError(
                "`mixtures` must include both 'video' and 'action' experts."
            )

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())

        first_expert = self.mixtures[self.expert_order[0]]
        self.block_protocol = str(first_expert.block_protocol)
        if self.block_protocol not in SUPPORTED_BLOCK_PROTOCOLS:
            raise ValueError(f"Unsupported MoT block protocol: {self.block_protocol!r}")
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim
        self.attention_backend = normalize_attention_backend(
            first_expert.attention_backend
        )

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            expert_protocol = str(expert.block_protocol)
            if expert_protocol != self.block_protocol:
                raise ValueError(
                    f"All experts must use the same block protocol; got {self.block_protocol!r} "
                    f"and {expert_protocol!r} for {name!r}."
                )
            if len(expert.blocks) != self.num_layers:
                raise ValueError(
                    f"All experts must have same number of layers; got {self.num_layers} and {len(expert.blocks)}"
                )
            if expert.num_heads != self.num_heads:
                raise ValueError(
                    f"All experts must have same num_heads; got {self.num_heads} and {expert.num_heads}"
                )
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(
                    "All experts must have same attn_head_dim; "
                    f"got {self.attn_head_dim} and {expert.attn_head_dim}"
                )
            if (
                normalize_attention_backend(expert.attention_backend)
                != self.attention_backend
            ):
                raise ValueError(
                    "All experts must use the same attention_backend; "
                    f"got {self.attention_backend} and {expert.attention_backend}"
                )

        self.double_layers = 0
        self.single_layers = 0
        if self.block_protocol == BLOCK_PROTOCOL_STAGED:
            self.double_layers = int(first_expert.double_layers)
            self.single_layers = int(first_expert.single_layers)
            for name in self.expert_order[1:]:
                expert = self.mixtures[name]
                if int(expert.double_layers) != self.double_layers:
                    raise ValueError(
                        "All staged experts must have the same double-stage depth."
                    )
                if int(expert.single_layers) != self.single_layers:
                    raise ValueError(
                        "All staged experts must have the same single-stage depth."
                    )

        logger.info(
            "Initialized MoT with experts=%s protocol=%s num_layers=%d",
            self.expert_order,
            self.block_protocol,
            self.num_layers,
        )
        total_params = 0
        for name in self.expert_order:
            expert = self.mixtures[name]
            expert_params = sum(p.numel() for p in expert.parameters())
            total_params += expert_params
            logger.info(f"  Expert '{name}': num_params={expert_params / 1e9:.6f} B")
        logger.info(f"  MoT experts total: num_params={total_params / 1e9:.6f} B")

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            base_mod + t_mod
        ).unbind(dim=2)
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor | StructuredAttentionMask,
    ) -> torch.Tensor:
        return run_attention(
            q=q_cat,
            k=k_cat,
            v=v_cat,
            num_heads=self.num_heads,
            attention_mask=attention_mask.to(device=q_cat.device),
            backend=self.attention_backend,
        )

    @staticmethod
    def _context_for_layer(
        context_payload: Optional[dict], layer_index: int
    ) -> Optional[dict]:
        if context_payload is None:
            return None
        kv_cache = context_payload.get("kv_cache")
        if kv_cache is None:
            return context_payload
        if len(kv_cache) <= layer_index:
            raise ValueError(
                "Cross-attention KV cache does not cover every transformer layer."
            )
        return {
            "context": context_payload.get("context"),
            "mask": context_payload.get("mask"),
            "kv": kv_cache[layer_index],
        }

    @staticmethod
    def _apply_expert_post_block(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))

        if context_payload is not None:
            context = context_payload.get("context")
            if context is not None and block.cross_attn is not None:
                context_mask = context_payload.get("mask")
                context_kv = context_payload.get("kv")
                x = x + block.cross_attn(
                    block.norm3(x),
                    context,
                    ctx_mask=context_mask,
                    projected_kv=context_kv,
                )

        mlp_input = _modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return x

    def _build_expert_attention_io(
        self,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if block.uses_mixed_attention_api:
            q, k, v, state = block.prepare_mixed_attention(
                x,
                t_mod["embedding"],
                t_mod["adaln_lora"],
                freqs,
                t_mod.get("token_to_timestep"),
            )
            return q, k, v, state, None, None, None, None

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self._split_modulation(block, t_mod)
        )
        attn_input = _modulate(block.norm1(x), shift_msa, scale_msa)

        if hasattr(block.self_attn, "project_qkv"):
            q, k, v = block.self_attn.project_qkv(attn_input)
        else:
            q = block.self_attn.norm_q(block.self_attn.q(attn_input))
            k = block.self_attn.norm_k(block.self_attn.k(attn_input))
            v = block.self_attn.v(attn_input)

        q = _rope_apply(q, freqs, block.num_heads)
        k = _rope_apply(k, freqs, block.num_heads)

        return (
            q,
            k,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    def _apply_post_block(
        self,
        block,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        if block.uses_mixed_attention_api:
            context = (
                None if context_payload is None else context_payload.get("context")
            )
            context_mask = (
                None if context_payload is None else context_payload.get("mask")
            )
            context_kv = None if context_payload is None else context_payload.get("kv")
            if context_kv is None:
                return block.finish_mixed_attention(
                    mixed_slice, residual_x, context, context_mask
                )
            return block.finish_mixed_attention(
                mixed_slice, residual_x, context, context_mask, context_kv=context_kv
            )

        def _post_fn(
            _mixed_slice: torch.Tensor,
            _x: torch.Tensor,
            _gate_msa: torch.Tensor,
            _shift_mlp: torch.Tensor,
            _scale_mlp: torch.Tensor,
            _gate_mlp: torch.Tensor,
            _block=block,
            _context_payload=context_payload,
        ) -> torch.Tensor:
            return self._apply_expert_post_block(
                block=_block,
                residual_x=_x,
                mixed_attn_out=_mixed_slice,
                gate_msa=_gate_msa,
                shift_mlp=_shift_mlp,
                scale_mlp=_scale_mlp,
                gate_mlp=_gate_mlp,
                context_payload=_context_payload,
            )

        return _post_fn(
            mixed_slice,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor | StructuredAttentionMask,
    ) -> list[dict[str, torch.Tensor]]:
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `prefill_video_cache`.")
        if video_attention_mask.ndim != 2:
            raise ValueError(
                f"`video_attention_mask` must be 2D [S,S], got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_attention_mask.shape[1]:
            raise ValueError(
                f"`video_attention_mask` must be square, got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_tokens.shape[1]:
            raise ValueError(
                "`video_attention_mask` seq length mismatch: "
                f"mask={video_attention_mask.shape[0]} vs tokens={video_tokens.shape[1]}"
            )

        expert = self.mixtures["video"]
        x = video_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = self._build_expert_attention_io(
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k,
                v_cat=v,
                attention_mask=video_attention_mask,
            )
            x = self._apply_post_block(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                mixed_slice=mixed,
                context_payload=self._context_for_layer(
                    video_context_payload, layer_idx
                ),
            )
            kv_cache.append({"k": k, "v": v})
        return kv_cache

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor | StructuredAttentionMask,
        video_seq_len: int,
    ) -> torch.Tensor:
        if "action" not in self.mixtures:
            raise ValueError(
                "MoT requires `action` expert for `forward_action_with_video_cache`."
            )
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(
                f"`attention_mask` must be 2D [S,S], got shape {tuple(attention_mask.shape)}"
            )
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(
                f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}"
            )

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs expected_total={total_seq_len}"
            )
        if isinstance(attention_mask, StructuredAttentionMask):
            action_attention_mask = attention_mask.slice(video_seq_len, total_seq_len)
        else:
            action_attention_mask = attention_mask[
                video_seq_len:total_seq_len, :total_seq_len
            ]

        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = self._build_expert_attention_io(
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            layer_cache = video_kv_cache[layer_idx]
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` must contain `k` and `v`."
                )

            k_video = layer_cache["k"]
            v_video = layer_cache["v"]
            if k_video.shape[1] != video_seq_len or v_video.shape[1] != video_seq_len:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` seq len mismatch, expected {video_seq_len}."
                )

            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)
            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_attention_mask,
            )
            x = self._apply_post_block(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                mixed_slice=mixed,
                context_payload=self._context_for_layer(
                    action_context_payload, layer_idx
                ),
            )
        return x

    def forward(
        self,
        embeds_all: Dict[str, object],
        attention_mask: (
            torch.Tensor | StructuredAttentionMask | dict[str, torch.Tensor]
        ),
        freqs_all: Dict[str, object],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, object],
    ):
        missing = [k for k in self.expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f"Missing expert tokens for {missing}")
        missing = [k for k in self.expert_order if k not in freqs_all]
        if missing:
            raise ValueError(f"Missing expert freqs for {missing}")
        missing = [k for k in self.expert_order if k not in t_mod_all]
        if missing:
            raise ValueError(f"Missing expert t_mod for {missing}")

        if self.block_protocol == BLOCK_PROTOCOL_STAGED:
            raise ValueError(
                "Staged experts require forward_prepared with explicit token layouts"
            )

        if attention_mask.ndim != 2:
            raise ValueError(
                f"`attention_mask` must be 2D [S, S], got shape {tuple(attention_mask.shape)}"
            )
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(
                f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}"
            )

        tokens_all = {k: v for k, v in embeds_all.items()}
        use_gradient_checkpointing = any(
            expert.use_gradient_checkpointing for expert in self.mixtures.values()
        )

        for layer_idx in range(self.num_layers):

            def _layer_forward(*layer_tokens, layer_index=layer_idx):
                current = dict(zip(self.expert_order, layer_tokens))
                q_chunks, k_chunks, v_chunks, seq_lens = [], [], [], []
                cached = {}
                for name in self.expert_order:
                    block = self.mixtures[name].blocks[layer_index]
                    x = current[name]
                    values = self._build_expert_attention_io(
                        block=block,
                        x=x,
                        freqs=freqs_all[name],
                        t_mod=t_mod_all[name],
                    )
                    q, k, v, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                        values
                    )
                    q_chunks.append(q)
                    k_chunks.append(k)
                    v_chunks.append(v)
                    seq_lens.append(x.shape[1])
                    cached[name] = (
                        block,
                        residual_x,
                        gate_msa,
                        shift_mlp,
                        scale_mlp,
                        gate_mlp,
                    )

                q_cat = torch.cat(q_chunks, dim=1)
                k_cat = torch.cat(k_chunks, dim=1)
                v_cat = torch.cat(v_chunks, dim=1)
                if attention_mask.shape[0] != q_cat.shape[1]:
                    raise ValueError(
                        "Attention mask seq length mismatch: "
                        f"mask={attention_mask.shape[0]} vs tokens={q_cat.shape[1]}"
                    )
                mixed = self._mixed_attention(q_cat, k_cat, v_cat, attention_mask)
                outputs = []
                start = 0
                for name, seq_len in zip(self.expert_order, seq_lens):
                    end = start + seq_len
                    block, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                        cached[name]
                    )
                    outputs.append(
                        self._apply_post_block(
                            block=block,
                            residual_x=residual_x,
                            gate_msa=gate_msa,
                            shift_mlp=shift_mlp,
                            scale_mlp=scale_mlp,
                            gate_mlp=gate_mlp,
                            mixed_slice=mixed[:, start:end, :],
                            context_payload=self._context_for_layer(
                                context_all.get(name), layer_index
                            ),
                        )
                    )
                    start = end
                return tuple(outputs)

            inputs = tuple(tokens_all[name] for name in self.expert_order)
            if use_gradient_checkpointing and torch.is_grad_enabled():
                outputs = torch.utils.checkpoint.checkpoint(
                    _layer_forward, *inputs, use_reentrant=False
                )
            else:
                outputs = _layer_forward(*inputs)
            tokens_all.update(zip(self.expert_order, outputs))

        return tokens_all

    def forward_prepared(self, pres, action_pre, attention_mask):
        expert, action = self.mixtures["video"], self.mixtures["action"]
        action_tokens = action_pre["tokens"]
        mask = attention_mask
        tokens = [p["tokens"] for p in pres]
        action_pe = action.prepare_position(action_pre, expert.transformer.pe_embedder)
        for index in range(len(expert.blocks)):

            def layer(scene_tokens, values, i=index):
                streams = [
                    expert.prepare_attention(i, t, p)
                    for t, p in zip(scene_tokens, pres)
                ]
                block = action.blocks[i]
                mod = (
                    action_pre["t_mod"]["double_img"]
                    if i < action.double_layers
                    else action_pre["t_mod"]["single"]
                )
                act = block.prepare_qkv(values, action_pe, mod)
                q = torch.cat([s[0] for s in streams] + [act["q"]], 1)
                k = torch.cat([s[1] for s in streams] + [act["k"]], 1)
                v = torch.cat([s[2] for s in streams] + [act["v"]], 1)
                mixed = run_attention(
                    q,
                    k,
                    v,
                    num_heads=expert.num_heads,
                    attention_mask=mask,
                    backend=expert.attention_backend,
                )
                lengths = [s[0].shape[1] for s in streams] + [values.shape[1]]
                pieces = mixed.split(lengths, 1)
                updated = [
                    expert.finish_attention(i, out, stream[3])
                    for out, stream in zip(pieces, streams)
                ]
                action_out = block.apply_post(pieces[-1], act)
                if action.cross_attention:
                    action_out = action_out + block.cross_attn(
                        action_out,
                        action_pre["context"],
                        ctx_mask=action_pre["context_mask"],
                    )
                return updated, action_out

            tokens, action_tokens = gradient_checkpoint_forward(
                layer,
                (expert.use_gradient_checkpointing or action.use_gradient_checkpointing)
                and torch.is_grad_enabled(),
                tokens,
                action_tokens,
            )
        return expert.post_dit(tokens[0], pres[0]), action.post_dit(
            action_tokens, action_pre
        )
