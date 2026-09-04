from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .attention import StructuredAttentionMask, normalize_attention_backend, run_attention
from ..backbone.protocol import BLOCK_PROTOCOL_FLUX2, BLOCK_PROTOCOL_MAIN, SUPPORTED_BLOCK_PROTOCOLS
from utils.logging_config import get_logger

logger = get_logger(__name__)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def _rope_apply(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    b, s, _ = x.shape
    x = x.view(b, s, num_heads, -1)
    x_complex = torch.view_as_complex(x.float().reshape(b, s, num_heads, -1, 2))
    freqs = freqs.to(device=x.device)
    if freqs.ndim == 3:
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
            raise ValueError("`mixtures` must include both 'video' and 'action' experts.")

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())

        first_expert = self.mixtures[self.expert_order[0]]
        self.block_protocol = str(getattr(first_expert, "block_protocol", BLOCK_PROTOCOL_MAIN))
        if self.block_protocol not in SUPPORTED_BLOCK_PROTOCOLS:
            raise ValueError(f"Unsupported MoT block protocol: {self.block_protocol!r}")
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim
        self.attention_backend = normalize_attention_backend(first_expert.attention_backend)

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            expert_protocol = str(getattr(expert, "block_protocol", BLOCK_PROTOCOL_MAIN))
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
            if normalize_attention_backend(expert.attention_backend) != self.attention_backend:
                raise ValueError(
                    "All experts must use the same attention_backend; "
                    f"got {self.attention_backend} and {expert.attention_backend}"
                )

        self.double_layers = 0
        self.single_layers = 0
        if self.block_protocol == BLOCK_PROTOCOL_FLUX2:
            self.double_layers = int(getattr(first_expert, "double_layers"))
            self.single_layers = int(getattr(first_expert, "single_layers"))
            for name in self.expert_order[1:]:
                expert = self.mixtures[name]
                if int(getattr(expert, "double_layers")) != self.double_layers:
                    raise ValueError("All FLUX.2 experts must have the same double-stage depth.")
                if int(getattr(expert, "single_layers")) != self.single_layers:
                    raise ValueError("All FLUX.2 experts must have the same single-stage depth.")
        
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
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
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
            raise ValueError("Cross-attention KV cache does not cover every transformer layer.")
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
            if context is not None:
                context_mask = context_payload.get("mask")
                context_kv = context_payload.get("kv")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
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
        """Build per-expert attention tensors and post-block states.

        Args:
            block: Transformer block for current layer (`expert.blocks[layer_idx]`).
            x: Current expert tokens, shape [B, S, D].
            freqs: RoPE frequencies aligned with token sequence, shape [S, 1, rope_dim].
            t_mod: Time modulation tensor for this expert/layer.

        Returns:
            q: Query after q-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            k: Key after k-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            v: Value after v-proj, shape [B, S, H*Dh].
            residual_x: Original input `x` for residual path in post block.
            gate_msa: Gating tensor for self-attention residual branch.
            shift_mlp: Shift tensor for MLP modulation.
            scale_mlp: Scale tensor for MLP modulation.
            gate_mlp: Gating tensor for MLP residual branch.
        """
        if hasattr(block, "prepare_mixed_attention"):
            q, k, v, state = block.prepare_mixed_attention(
                x,
                t_mod["embedding"],
                t_mod["adaln_lora"],
                freqs,
                t_mod.get("token_to_timestep"),
            )
            return q, k, v, state, None, None, None, None

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        attn_input = _modulate(block.norm1(x), shift_msa, scale_msa)

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
        """Apply post-attention computations.

        Args:
            block: Transformer block for current layer.
            residual_x: Residual input tokens before attention update, shape [B, S, D].
            gate_msa: Gating tensor used after mixed self-attention.
            shift_mlp: Shift tensor for MLP input modulation.
            scale_mlp: Scale tensor for MLP input modulation.
            gate_mlp: Gating tensor used after MLP.
            mixed_slice: Mixed-attention output for this expert, shape [B, S, H*Dh].
            context_payload: Optional dict for cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, S, L] or [B, 1, S, L]

        Returns:
            Updated expert tokens after self-attn residual, optional cross-attn, and MLP.
        """
        if hasattr(block, "finish_mixed_attention"):
            context = None if context_payload is None else context_payload.get("context")
            context_mask = None if context_payload is None else context_payload.get("mask")
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
        """Prefill video branch once and cache per-layer K/V for action denoising.

        Args:
            video_tokens: Video tokens before layer 0, shape [B, Sv, D].
            video_freqs: Video RoPE frequencies, shape [Sv, 1, rope_dim].
            video_t_mod: Video time modulation tensor.
            video_context_payload: Optional dict for video cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sv, L] or [B, 1, Sv, L]
            video_attention_mask: Video self-attention mask, shape [Sv, Sv].

        Returns:
            Layer-wise cache list with length `num_layers`.
            Each entry contains:
                - `k`: video key tensor [B, Sv, H*Dh]
                - `v`: video value tensor [B, Sv, H*Dh]
        """
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
            # Build video Q/K/V from current layer input tokens.
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
            # Video prefill uses only video self-attention mask.
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k,
                v_cat=v,
                attention_mask=video_attention_mask,
            )
            # Update video tokens for the next layer and persist current layer K/V.
            x = self._apply_post_block(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                mixed_slice=mixed,
                context_payload=self._context_for_layer(video_context_payload, layer_idx),
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
        """Run action branch with cached video K/V instead of recomputing video tokens.

        Args:
            action_tokens: Action tokens before layer 0, shape [B, Sa, D].
            action_freqs: Action RoPE frequencies, shape [Sa, 1, rope_dim].
            action_t_mod: Action time modulation tensor.
            action_context_payload: Optional dict for action cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sa, L] or [B, 1, Sa, L]
            video_kv_cache: Layer-wise cached video K/V from `prefill_video_cache`.
            attention_mask: Joint [video+action] mask, shape [Sv+Sa, Sv+Sa].
            video_seq_len: Video token count `Sv` in the joint sequence prefix.

        Returns:
            Updated action tokens after all layers, shape [B, Sa, D].
        """
        if "action" not in self.mixtures:
            raise ValueError("MoT requires `action` expert for `forward_action_with_video_cache`.")
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S,S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs expected_total={total_seq_len}"
            )
        # Use the action query rows from the joint [video+action] mask.
        if isinstance(attention_mask, StructuredAttentionMask):
            action_attention_mask = attention_mask.slice(video_seq_len, total_seq_len)
        else:
            action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]

        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Action query/key/value are still step-dependent and must be recomputed each step.
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

            # Mixed attention: action queries attend to cached video K/V plus current action K/V.
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
                context_payload=self._context_for_layer(action_context_payload, layer_idx),
            )
        return x

    def forward(
        self,
        embeds_all: Dict[str, object],
        attention_mask: torch.Tensor | StructuredAttentionMask | dict[str, torch.Tensor],
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

        if self.block_protocol == BLOCK_PROTOCOL_FLUX2:
            return self._forward_flux2(
                embeds_all=embeds_all,
                attention_mask=attention_mask,
                freqs_all=freqs_all,
                context_all=context_all,
                t_mod_all=t_mod_all,
            )

        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S, S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        tokens_all = {k: v for k, v in embeds_all.items()}
        use_gradient_checkpointing = any(
            bool(getattr(expert, "use_gradient_checkpointing", False))
            for expert in self.mixtures.values()
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
                    q, k, v, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = values
                    q_chunks.append(q)
                    k_chunks.append(k)
                    v_chunks.append(v)
                    seq_lens.append(x.shape[1])
                    cached[name] = (block, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)

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
                    block, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = cached[name]
                    outputs.append(self._apply_post_block(
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
                    ))
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

    @staticmethod
    def _flux2_flatten_heads(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.transpose(1, 2).reshape(
            tensor.shape[0], tensor.shape[2], tensor.shape[1] * tensor.shape[3]
        )

    def _flux2_video_single_io(self, block, x: torch.Tensor, pe: torch.Tensor, modulation) -> dict:
        from flux2.model import apply_rope

        q, k, v, mlp, gate = block._qkv(x, modulation)
        q, k = apply_rope(q, k, pe)
        return {
            "q": self._flux2_flatten_heads(q),
            "k": self._flux2_flatten_heads(k),
            "v": self._flux2_flatten_heads(v),
            "mlp": mlp,
            "gate": gate,
            "residual_x": x,
        }

    def _forward_flux2(
        self,
        embeds_all: Dict[str, object],
        attention_mask: object,
        freqs_all: Dict[str, object],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, object],
    ):
        if not isinstance(attention_mask, dict):
            raise ValueError("FLUX.2 expects attention_mask={'double_joint', 'single'}.")
        if set(("double_joint", "single")) - set(attention_mask):
            raise ValueError("FLUX.2 attention mask needs double_joint and single entries.")
        video_expert = self.mixtures["video"]
        action_expert = self.mixtures["action"]
        video_state = embeds_all["video"]
        if not isinstance(video_state, dict):
            raise ValueError("FLUX.2 video tokens must contain txt/img streams.")
        txt, img = video_state["txt"], video_state["img"]
        action = embeds_all["action"]
        if not isinstance(action, torch.Tensor):
            raise ValueError("FLUX.2 action tokens must be a tensor.")

        video_freqs = freqs_all["video"]
        txt_pe, img_pe = video_freqs["txt"], video_freqs["img"]
        action_payload = context_all.get("action") or {}
        action_ids = action_payload.get("ids")
        if action_ids is None:
            raise ValueError("FLUX.2 action context payload must provide position ids.")
        action_pe = video_expert.transformer.pe_embedder(
            action_ids.to(device=img.device, dtype=img.dtype)
        )
        video_mod = t_mod_all["video"]
        action_mod = t_mod_all["action"]

        from flux2.model import apply_rope

        for layer_idx in range(self.double_layers):
            video_block = video_expert.double_blocks[layer_idx]
            action_block = action_expert.double_blocks[layer_idx]
            q, k, v, pe, num_txt, residuals = video_block._prepare_qkv(
                img,
                txt,
                img_pe,
                txt_pe,
                video_mod["double_img"],
                video_mod["double_txt"],
            )
            q, k = apply_rope(q, k, pe)
            action_state = action_block.prepare_qkv(action, action_pe, action_mod["double_img"])
            mixed = self._mixed_attention(
                torch.cat([self._flux2_flatten_heads(q), action_state["q"]], dim=1),
                torch.cat([self._flux2_flatten_heads(k), action_state["k"]], dim=1),
                torch.cat([self._flux2_flatten_heads(v), action_state["v"]], dim=1),
                attention_mask["double_joint"],
            )
            video_attention, action_attention = torch.split(
                mixed, [txt.shape[1] + img.shape[1], action.shape[1]], dim=1
            )
            txt_attention, img_attention = torch.split(
                video_attention, [num_txt, img.shape[1]], dim=1
            )
            img, txt = video_block._apply_residuals(
                img, txt, img_attention, txt_attention, residuals
            )
            action = action_block.apply_post(action_attention, action_state)

        video_stream = torch.cat([txt, img], dim=1)
        stream_pe = torch.cat([txt_pe, img_pe], dim=2)
        for layer_idx in range(self.single_layers):
            video_block = video_expert.single_blocks[layer_idx]
            action_block = action_expert.single_blocks[layer_idx]
            video_state = self._flux2_video_single_io(
                video_block, video_stream, stream_pe, video_mod["single"]
            )
            action_state = action_block.prepare_qkv(action, action_pe, action_mod["single"])
            mixed = self._mixed_attention(
                torch.cat([video_state["q"], action_state["q"]], dim=1),
                torch.cat([video_state["k"], action_state["k"]], dim=1),
                torch.cat([video_state["v"], action_state["v"]], dim=1),
                attention_mask["single"],
            )
            video_attention, action_attention = torch.split(
                mixed, [video_stream.shape[1], action.shape[1]], dim=1
            )
            video_stream = video_block._out(
                video_state["residual_x"],
                video_attention,
                video_state["mlp"],
                video_state["gate"],
            )
            action = action_block.apply_post(action_attention, action_state)

        txt_len = int(txt.shape[1])
        return {
            "video": {"txt": video_stream[:, :txt_len], "img": video_stream[:, txt_len:]},
            "action": action,
        }

    @torch.no_grad()
    def prefill_flux2_video_cache(
        self,
        video_tokens: dict[str, torch.Tensor],
        video_freqs: dict[str, torch.Tensor],
        video_t_mod: dict[str, object],
        attention_mask: dict[str, torch.Tensor],
    ) -> dict[str, object]:
        """Run the fixed FLUX.2 text/image prefix once and retain layer K/V."""
        if self.block_protocol != BLOCK_PROTOCOL_FLUX2:
            raise ValueError("`prefill_flux2_video_cache` requires block_protocol='flux2'.")
        video_expert = self.mixtures["video"]
        txt = video_tokens["txt"]
        img = video_tokens["img"]
        txt_pe = video_freqs["txt"]
        img_pe = video_freqs["img"]

        from flux2.model import apply_rope

        double_cache = []
        for layer_idx in range(self.double_layers):
            block = video_expert.double_blocks[layer_idx]
            q, k, v, pe_full, num_txt_tokens, mods = block._prepare_qkv(
                img,
                txt,
                img_pe,
                txt_pe,
                video_t_mod["double_img"],
                video_t_mod["double_txt"],
            )
            q, k = apply_rope(q, k, pe_full)
            flat_q = self._flux2_flatten_heads(q)
            flat_k = self._flux2_flatten_heads(k)
            flat_v = self._flux2_flatten_heads(v)
            double_cache.append({"k": flat_k, "v": flat_v})
            mixed = self._mixed_attention(
                flat_q, flat_k, flat_v, attention_mask["double_joint"]
            )
            txt_attention, img_attention = torch.split(
                mixed, [num_txt_tokens, img.shape[1]], dim=1
            )
            img, txt = block._apply_residuals(
                img, txt, img_attention, txt_attention, mods
            )

        video_stream = torch.cat([txt, img], dim=1)
        stream_pe = torch.cat([txt_pe, img_pe], dim=2)
        single_cache = []
        for layer_idx in range(self.single_layers):
            block = video_expert.single_blocks[layer_idx]
            state = self._flux2_video_single_io(
                block, video_stream, stream_pe, video_t_mod["single"]
            )
            single_cache.append({"k": state["k"], "v": state["v"]})
            mixed = self._mixed_attention(
                state["q"], state["k"], state["v"], attention_mask["single"]
            )
            video_stream = block._out(
                state["residual_x"], mixed, state["mlp"], state["gate"]
            )

        return {
            "double": double_cache,
            "single": single_cache,
            "txt_len": int(txt.shape[1]),
            "img_len": int(img.shape[1]),
        }

    def forward_flux2_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_ids: torch.Tensor,
        action_t_mod: dict[str, object],
        video_kv_cache: dict[str, object],
        attention_mask: dict[str, torch.Tensor],
        video_seq_len: int,
    ) -> torch.Tensor:
        """Denoise FLUX.2 action tokens against a cached text/image prefix."""
        if self.block_protocol != BLOCK_PROTOCOL_FLUX2:
            raise ValueError(
                "`forward_flux2_action_with_video_cache` requires block_protocol='flux2'."
            )
        video_expert = self.mixtures["video"]
        action_expert = self.mixtures["action"]
        action = action_tokens
        action_pe = video_expert.transformer.pe_embedder(
            action_ids.to(device=action.device, dtype=action.dtype)
        )
        action_seq_len = int(action.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len

        def _action_rows(mask: torch.Tensor) -> torch.Tensor:
            if mask.ndim == 2:
                return mask[video_seq_len:total_seq_len, :total_seq_len]
            if mask.ndim == 3:
                return mask[:, video_seq_len:total_seq_len, :total_seq_len]
            if mask.ndim == 4:
                return mask[:, :, video_seq_len:total_seq_len, :total_seq_len]
            raise ValueError(f"Unsupported FLUX.2 attention mask rank: {mask.ndim}")

        double_mask = _action_rows(attention_mask["double_joint"])
        for layer_idx, cache in enumerate(video_kv_cache["double"]):
            block = action_expert.double_blocks[layer_idx]
            state = block.prepare_qkv(action, action_pe, action_t_mod["double_img"])
            k_cat = torch.cat([cache["k"].to(dtype=state["k"].dtype), state["k"]], dim=1)
            v_cat = torch.cat([cache["v"].to(dtype=state["v"].dtype), state["v"]], dim=1)
            mixed = self._mixed_attention(state["q"], k_cat, v_cat, double_mask)
            action = block.apply_post(mixed, state)

        single_mask = _action_rows(attention_mask["single"])
        for layer_idx, cache in enumerate(video_kv_cache["single"]):
            block = action_expert.single_blocks[layer_idx]
            state = block.prepare_qkv(action, action_pe, action_t_mod["single"])
            k_cat = torch.cat([cache["k"].to(dtype=state["k"].dtype), state["k"]], dim=1)
            v_cat = torch.cat([cache["v"].to(dtype=state["v"].dtype), state["v"]], dim=1)
            mixed = self._mixed_attention(state["q"], k_cat, v_cat, single_mask)
            action = block.apply_post(mixed, state)
        return action
