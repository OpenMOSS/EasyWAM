from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.helpers.static_cache import cache_static_method
from utils.logging_config import get_logger

from .component.action_dit import (
    ActionDecoder,
    ActionEncoder,
    StateEncoder,
    normalize_state_position,
    validate_checkpoint_state_position,
)
from .component.attention import (
    AttentionSegment,
    StructuredAttentionMask,
    build_structured_attention_mask,
    elide_fully_valid_attention_mask,
    prepare_key_padding_mask,
)
from .backbone.wan22 import load_wan22_ti2v_5b_components
from .schedulers.scheduler_continuous import ContinuousFlowMatchScheduler
from .schedulers.scheduler_flow_unipc import FlowUniPCScheduler
from .helpers.batching import (
    ActionInferenceResult,
    JointInferenceResult,
    ActionBatchResult,
    JointBatchResult,
    decode_video_batch,
    encode_image_batch,
    randn_per_sample,
    batch_single_tensor,
    batch_single_state,
    validate_batch_inputs,
)

logger = get_logger(__name__)


class EasyWAMUnified(nn.Module):

    inference_compile_targets = ("_forward_dit",)

    def __init__(
        self,
        video_dit,
        vae,
        action_dim: int,
        state_dim: int,
        state_position: str = "context",
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        projector_hidden_dim: int = 64,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        video_scheduler_config: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.inference_cross_kv_reuse = False
        self.backbone_name = video_dit.backbone_name
        if not hasattr(video_dit, "pre_unified_dit") or not hasattr(
            video_dit, "post_unified_dit"
        ):
            raise TypeError(
                f"Backbone {type(video_dit).__name__} does not implement the Unified staged API."
            )
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.state_position = normalize_state_position(state_position)
        self.projector_hidden_dim = int(projector_hidden_dim)

        self.hidden_dim = int(video_dit.hidden_dim)
        self.context_dim = int(video_dit.context_dim)
        self.freq_dim = int(video_dit.freq_dim)
        self.num_heads = int(video_dit.num_heads)
        self.attn_head_dim = int(video_dit.attn_head_dim)
        self.patch_size = tuple(video_dit.patch_size)

        self.dit = nn.ModuleDict(
            {
                "video_dit": video_dit,
                "state_encoder": StateEncoder(
                    state_dim=self.state_dim,
                    hidden_dim=(
                        self.context_dim
                        if self.state_position == "context"
                        else self.hidden_dim
                    ),
                    projector_hidden_dim=self.projector_hidden_dim,
                ),
                "action_encoder": ActionEncoder(
                    action_dim=self.action_dim,
                    hidden_dim=self.hidden_dim,
                    projector_hidden_dim=self.projector_hidden_dim,
                ),
                "action_decoder": ActionDecoder(
                    hidden_dim=self.hidden_dim,
                    action_dim=self.action_dim,
                    projector_hidden_dim=self.projector_hidden_dim,
                ),
            }
        )
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError(
                    "`text_dim` is required when `text_encoder` is not loaded."
                )
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.infer_shift = float(video_infer_shift)

        scheduler_cfg = dict(video_scheduler_config or {})
        if self.backbone_name == "wan22":
            self.scheduler = ContinuousFlowMatchScheduler(
                num_train_timesteps=video_num_train_timesteps,
                shift=video_train_shift,
            )
        elif self.backbone_name == "cosmos25":
            time_distribution = str(
                scheduler_cfg.get("time_distribution", "logitnormal")
            )
            training_weight = str(scheduler_cfg.get("training_weight", "uniform"))
            use_karras_sigmas = bool(scheduler_cfg.get("use_karras_sigmas", True))
            self.scheduler = FlowUniPCScheduler(
                video_num_train_timesteps,
                video_train_shift,
                use_karras_sigmas=use_karras_sigmas,
                time_distribution=time_distribution,
                training_weight_method=training_weight,
            )
        else:
            raise ValueError(
                "EasyWAM-Unified supports scheduler families for 'wan22' and "
                f"'cosmos25', got backbone {self.backbone_name!r}."
            )
        # Video and action share a scheduler and timestep.
        self.train_scheduler = self.scheduler
        self.infer_scheduler = self.scheduler
        self.to(device=self.device, dtype=self.torch_dtype)

    @property
    def video_dit(self):
        return self.dit["video_dit"]

    @property
    def blocks(self):
        return self.video_dit.blocks

    @staticmethod
    def _build_single_chunk_mask(
        clean_video_len: int,
        future_video_len: int,
        action_len: int,
        device: torch.device,
        state_len: int = 0,
        video_attention_mask_mode: str = "first_frame_causal",
    ) -> StructuredAttentionMask:
        video_len = clean_video_len + future_video_len
        state_start = video_len + action_len
        total = state_start + state_len
        if video_attention_mask_mode == "bidirectional":
            segments = [
                AttentionSegment(0, total, ((0, total),)),
            ]
        elif video_attention_mask_mode == "first_frame_causal":
            clean_key_ranges = [(0, clean_video_len)]
            if state_len:
                clean_key_ranges.append((state_start, total))
            segments = [
                AttentionSegment(0, clean_video_len, tuple(clean_key_ranges)),
            ]
        elif video_attention_mask_mode == "per_frame_causal":
            if clean_video_len <= 0 or video_len % clean_video_len:
                raise ValueError(
                    "Unified video token length must be divisible by tokens per frame "
                    "in per_frame_causal mode."
                )
            segments = []
            for frame_start in range(0, video_len, clean_video_len):
                frame_end = frame_start + clean_video_len
                key_ranges = [(0, frame_end)]
                if state_len:
                    key_ranges.append((state_start, total))
                segments.append(
                    AttentionSegment(frame_start, frame_end, tuple(key_ranges))
                )
        else:
            raise ValueError(
                "EasyWAM-Unified supports `video_attention_mask_mode` values "
                "'first_frame_causal', 'per_frame_causal', and 'bidirectional', "
                f"got {video_attention_mask_mode!r}."
            )
        if video_attention_mask_mode == "first_frame_causal" and clean_video_len < state_start:
            segments.append(
                AttentionSegment(clean_video_len, state_start, ((0, total),))
            )
        if video_attention_mask_mode == "per_frame_causal" and video_len < state_start:
            segments.append(AttentionSegment(video_len, state_start, ((0, total),)))
        if video_attention_mask_mode != "bidirectional" and state_len:
            segments.append(AttentionSegment(state_start, total, ((0, total),)))
        return build_structured_attention_mask(
            query_len=total,
            key_len=total,
            segments=segments,
            device=device,
        )

    @cache_static_method()
    def _cached_single_chunk_mask(self, **kwargs):
        return self._build_single_chunk_mask(**kwargs)

    def _forward_dit(
        self,
        x: torch.Tensor,
        timestep_video: torch.Tensor,
        action: torch.Tensor,
        timestep_action: torch.Tensor,
        state: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        fuse_vae_embedding_in_latents: bool = True,
        projected_context: Optional[torch.Tensor] = None,
        cross_kv_cache: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None,
        prepared_state_tokens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del fuse_vae_embedding_in_latents
        if x.ndim != 5:
            raise ValueError(f"`x` must be 5D [B,C,T,H,W], got {tuple(x.shape)}")
        if action.ndim != 3 or action.shape[2] != self.action_dim:
            raise ValueError(
                f"`action` must be [B,T,{self.action_dim}], got {tuple(action.shape)}"
            )
        if state.ndim != 3 or state.shape[2] != self.state_dim:
            raise ValueError(f"`state` must be [B,S,{self.state_dim}], got {tuple(state.shape)}")
        if context.ndim != 3:
            raise ValueError(f"`context` must be [B,L,D], got {tuple(context.shape)}")

        batch_size = x.shape[0]
        if timestep_video.shape != (batch_size,) or timestep_action.shape != (batch_size,):
            raise ValueError("Video/action timestep batch size must match input batch size.")
        if context_mask is not None:
            context_mask = context_mask.to(device=x.device, dtype=torch.bool)

        action_tokens = self.dit["action_encoder"](action)
        state_tokens = prepared_state_tokens
        if state_tokens is None:
            state_tokens = self.dit["state_encoder"](state.to(device=x.device, dtype=x.dtype))
        pre = self.video_dit.pre_unified_dit(
            x=x,
            timestep_video=timestep_video.to(device=x.device, dtype=x.dtype),
            action_tokens=action_tokens,
            timestep_action=timestep_action.to(device=x.device, dtype=x.dtype),
            state_tokens=state_tokens,
            timestep_state=torch.zeros_like(timestep_action, device=x.device, dtype=x.dtype),
            state_position=self.state_position,
            context=context if projected_context is None else projected_context,
            context_mask=context_mask,
            context_is_projected=projected_context is not None,
            cross_kv_cache=cross_kv_cache,
        )
        tokens = pre["tokens"]
        video_len = int(pre["meta"]["video_len"])
        tokens_per_frame = int(pre["meta"]["tokens_per_frame"])
        attention_mask = self._cached_single_chunk_mask(
            clean_video_len=tokens_per_frame,
            future_video_len=video_len - tokens_per_frame,
            action_len=action_tokens.shape[1],
            state_len=(state_tokens.shape[1] if self.state_position == "sequence" else 0),
            device=tokens.device,
            video_attention_mask_mode=self.video_dit.video_attention_mask_mode,
        )

        for layer_index in range(len(self.video_dit.blocks)):
            if self.video_dit.use_gradient_checkpointing and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                def _block_forward(value, index=layer_index):
                    return self.video_dit.forward_block(
                        index, value, pre, attention_mask
                    )

                tokens = checkpoint(
                    _block_forward, tokens, use_reentrant=False
                )
            else:
                tokens = self.video_dit.forward_block(
                    layer_index, tokens, pre, attention_mask
                )
        outputs = self.video_dit.post_unified_dit(tokens, pre)
        action_out = self.dit["action_decoder"](outputs["action_tokens"])
        return outputs["video"], action_out

    @classmethod
    def from_backbone_pretrained(
        cls,
        backbone: dict[str, Any],
        *,
        action_dim: int,
        state_dim: int,
        state_position: str = "context",
        projector_hidden_dim: int = 64,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        video_scheduler_config: Optional[dict[str, Any]] = None,
    ) -> "EasyWAMUnified":
        from .backbone import load_easywam_backbone, normalize_backbone_config

        cfg = normalize_backbone_config(backbone)
        components = load_easywam_backbone(
            cfg,
            device=device,
            torch_dtype=torch_dtype,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
        )
        if video_scheduler_config is None:
            video_scheduler_config = dict(cfg.get("video_scheduler", {}))
        model = cls(
            video_dit=components.dit,
            vae=components.vae,
            action_dim=action_dim,
            state_dim=state_dim,
            state_position=state_position,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(cfg["text_dim"]),
            projector_hidden_dim=projector_hidden_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            video_scheduler_config=video_scheduler_config,
        )
        model.backbone_name = cfg["name"]
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
        }
        return model

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "./checkpoints/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "./checkpoints/Wan2.2-TI2V-5B/google/umt5-xxl",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dim: int | None = None,
        state_dim: int | None = None,
        state_position: str = "context",
        projector_hidden_dim: int = 64,
        skip_dit_load_from_pretrain: bool = False,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        video_scheduler_config: Optional[dict[str, Any]] = None,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for EasyWAM-Unified.")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for EasyWAM-Unified.")
        if action_dim is None or state_dim is None:
            raise ValueError("`action_dim` and `state_dim` are required for EasyWAM-Unified.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )
        model = cls(
            video_dit=components.dit,
            vae=components.vae,
            action_dim=int(action_dim),
            state_dim=int(state_dim),
            state_position=state_position,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            projector_hidden_dim=int(projector_hidden_dim),
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            video_scheduler_config=video_scheduler_config,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        if self.tokenizer is None:
            context, mask = self.text_encoder(prompt)
            return context.to(device=self.device), mask.to(device=self.device, dtype=torch.bool)
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        return prompt_emb.to(device=self.device), mask

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor):
        return self.vae.encode(video_tensor, device=self.device)

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor):
        return encode_image_batch(self.vae, input_image, self.device)

    def _decode_latents(self, latents):
        return decode_video_batch(self.vae, latents, self.device)

    def build_inputs(self, sample):
        video = sample["video"]
        context = sample.get("context")
        context_mask = sample.get("context_mask")
        action = sample.get("action")
        state = sample.get("proprio")
        if context is None or context_mask is None:
            raise ValueError("EasyWAM-Unified training requires `context` and `context_mask`.")
        if action is None:
            raise ValueError("EasyWAM-Unified training requires `action`.")
        if state is None:
            raise ValueError("EasyWAM-Unified training requires `proprio` as state.")
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"`video` must be [B,3,T,H,W], got {tuple(video.shape)}")
        if action.ndim != 3:
            raise ValueError(f"`action` must be [B,T,D], got {tuple(action.shape)}")
        if state.ndim != 3:
            raise ValueError(f"`proprio` must be [B,T,D], got {tuple(state.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"Video spatial dims must be multiples of 16, got H={height}, W={width}")
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError("EasyWAM-Unified requires at least 2 video frames.")
        if action.shape[0] != batch_size:
            raise ValueError("Action batch size must match video batch size.")
        if state.shape[0] != batch_size:
            raise ValueError("State batch size must match video batch size.")

        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video)
        first_frame_latents = input_latents[:, :, 0:1]

        context_mask = elide_fully_valid_attention_mask(context_mask)
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if context_mask is not None:
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        state = state[:, 0:1].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        image_is_pad = sample.get("image_is_pad")
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "action": action,
            "state": state,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if latent_tail_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={latent_tail_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )
        valid = (~latent_tail_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def training_loss(self, sample):
        inputs = self.build_inputs(sample)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]

        clean_first = inputs["first_frame_latents"]
        future_latents = input_latents[:, :, 1:]
        noise_video = torch.randn_like(future_latents)
        timestep_video = self.scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        noisy_future = self.scheduler.add_noise(future_latents, noise_video, timestep_video)
        target_video = self.scheduler.training_target(future_latents, noise_video, timestep_video)
        noisy_video = torch.cat([clean_first, noisy_future], dim=2)

        action = inputs["action"]
        noise_action = torch.randn_like(action)
        timestep_action = timestep_video.to(device=self.device, dtype=action.dtype)
        noisy_action = self.scheduler.add_noise(
            action, noise_action, timestep_action
        )
        target_action = self.scheduler.training_target(
            action, noise_action, timestep_action
        )

        pred_video, pred_action = self._forward_dit(
            x=noisy_video,
            timestep_video=timestep_video,
            action=noisy_action,
            timestep_action=timestep_action,
            state=inputs["state"],
            context=inputs["context"],
            context_mask=inputs["context_mask"],
        )
        pred_video = pred_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=inputs["image_is_pad"],
        )
        video_weight = self.scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if inputs["action_is_pad"] is not None:
            valid = (~inputs["action_is_pad"]).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)
        action_weight = self.scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_action * loss_action + self.loss_lambda_video * loss_video
        loss_dict = {
            "loss_video": loss_video.detach(),
            "loss_action": loss_action.detach(),
            "weighted_loss_action": loss_action.detach() * self.loss_lambda_action,
            "weighted_loss_video": loss_video.detach() * self.loss_lambda_video,
        }
        return loss_total, loss_dict

    def _prepare_context(
        self,
        prompt: Optional[Union[str, Sequence[str]]],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prompt is not None:
            return self.encode_prompt(prompt)
        return (
            context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
        )

    @torch.inference_mode()
    def infer_joint_batch(
        self,
        prompt: Optional[Sequence[str]],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[Union[int, Sequence[Optional[int]]]] = None,
        rand_device: str = "cpu",
        decode_video: bool = True,
        **kwargs,
    ) -> JointBatchResult:
        validate_batch_inputs(input_image, prompt, context, context_mask)
        del action, negative_prompt, text_cfg_scale, kwargs
        self.eval()
        batch_size = int(input_image.shape[0])
        if proprio is None:
            raise ValueError("EasyWAM-Unified inference requires `proprio` as state.")
        if proprio.ndim == 2:
            proprio = proprio.unsqueeze(1)
        if proprio.ndim != 3 or proprio.shape[0] != batch_size or proprio.shape[1] == 0 or proprio.shape[-1] != self.state_dim:
            raise ValueError(f"`proprio` must be [B,D] or [B,T,D], got {tuple(proprio.shape)}")

        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(f"`input_image` must be resized to multiples of 16, got HxW=({height},{width})")
        if checked_t != num_video_frames:
            raise ValueError(f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}")

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        latents_video = randn_per_sample(
            (self.vae.model.z_dim, latent_t, latent_h, latent_w), seeds=seed,
            batch_size=batch_size, rand_device=rand_device,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = randn_per_sample(
            (action_horizon, self.action_dim), seeds=seed,
            batch_size=batch_size, rand_device=rand_device,
        ).to(device=self.device, dtype=self.torch_dtype)

        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image.to(device=self.device, dtype=self.torch_dtype),
        )
        latents_video[:, :, 0:1] = first_frame_latents
        state = proprio[:, 0:1].to(device=self.device, dtype=self.torch_dtype)
        context, context_mask = self._prepare_context(prompt, context, context_mask)
        projected_context = self.video_dit.project_context(context)
        cross_kv_cache = None
        if self.inference_cross_kv_reuse:
            cross_kv_cache = self.video_dit.build_cross_attention_kv_cache(
                projected_context
            )

        context_mask = prepare_key_padding_mask(context_mask)
        prepared_state_tokens = self.dit["state_encoder"](state)
        infer_timesteps, infer_deltas = self.scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=self.infer_shift if sigma_shift is None else sigma_shift,
        )
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep_video = step_t.expand(batch_size).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = timestep_video.to(
                dtype=latents_action.dtype, device=self.device
            )
            pred_video, pred_action = self._forward_dit(
                x=latents_video,
                timestep_video=timestep_video,
                action=latents_action,
                timestep_action=timestep_action,
                state=state,
                context=context,
                context_mask=context_mask,
                projected_context=projected_context,
                cross_kv_cache=cross_kv_cache,
                prepared_state_tokens=prepared_state_tokens,
            )
            pred_video[:, :, 0:1] = 0
            if isinstance(self.scheduler, FlowUniPCScheduler):
                latents_video = self.scheduler.step(
                    pred_video, step_delta, latents_video, stream_id="video"
                )
                latents_action = self.scheduler.step(
                    pred_action, step_delta, latents_action, stream_id="action"
                )
            else:
                latents_video = self.scheduler.step(pred_video, step_delta, latents_video)
                latents_action = self.scheduler.step(pred_action, step_delta, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents

        result = {
            "action": latents_action.detach().to(device="cpu", dtype=torch.float32),
        }
        if decode_video:
            result["video"] = self._decode_latents(latents_video)
        return result

    @torch.inference_mode()
    def infer_action_batch(
        self,
        prompt: Optional[Sequence[str]],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_video_frames: int = 5,
        **kwargs,
    ) -> ActionBatchResult:
        kwargs.pop("decode_video", None)
        out = self.infer_joint_batch(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            decode_video=False,
            **kwargs,
        )
        return {"action": out["action"]}

    @torch.inference_mode()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        decode_video: bool = True,
        **kwargs,
    ) -> JointInferenceResult:
        input_image = batch_single_tensor(input_image, 3, "input_image")
        proprio = batch_single_state(proprio, history=True)
        context = batch_single_tensor(context, 2, "context")
        context_mask = batch_single_tensor(context_mask, 1, "context_mask")
        action = batch_single_tensor(action, 2, "action")
        result = self.infer_joint_batch(
            prompt=[prompt] if prompt is not None else None,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            decode_video=decode_video,
            **kwargs,
        )
        result["action"] = result["action"][0]
        if "video" in result:
            result["video"] = result["video"][0]
        return result

    @torch.inference_mode()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_video_frames: int = 5,
        **kwargs,
    ) -> ActionInferenceResult:
        input_image = batch_single_tensor(input_image, 3, "input_image")
        proprio = batch_single_state(proprio, history=True)
        context = batch_single_tensor(context, 2, "context")
        context_mask = batch_single_tensor(context_mask, 1, "context_mask")
        result = self.infer_action_batch(
            prompt=[prompt] if prompt is not None else None,
            input_image=input_image,
            action_horizon=action_horizon,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_video_frames=num_video_frames,
            **kwargs,
        )
        return {"action": result["action"][0]}

    @torch.inference_mode()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ):
        del action_cfg_scale
        if action_horizon is None:
            raise ValueError("`action_horizon` is required for EasyWAM-Unified inference.")
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        from .component.lora import (
            build_lora_checkpoint_payload,
            has_lora,
        )

        if has_lora(self.video_dit):
            payload = build_lora_checkpoint_payload(self, step=step)
        else:
            payload = {
                "dit": self.dit.state_dict(),
                "step": step,
                "torch_dtype": str(self.torch_dtype),
                "backbone_name": self.backbone_name,
            }
        payload["state_position"] = self.state_position
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None, merge_lora: bool = False):
        from .component.lora import (
            is_lora_checkpoint,
            load_lora_model_checkpoint_state,
            load_standard_state_dict,
        )

        payload = torch.load(path, map_location="cpu")
        validate_checkpoint_state_position(payload, self.state_position)
        checkpoint_backbone = payload.get("backbone_name")
        if checkpoint_backbone is not None and checkpoint_backbone != self.backbone_name:
            raise ValueError(
                f"Checkpoint backbone {checkpoint_backbone!r} does not match model backbone "
                f"{self.backbone_name!r}."
            )
        if is_lora_checkpoint(payload):
            load_lora_model_checkpoint_state(
                self,
                payload,
                merge_after_load=bool(merge_lora),
            )
        elif "dit" in payload:
            load_standard_state_dict(self.dit, payload["dit"], strict=False)
        else:
            self.load_state_dict(payload, strict=False)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, sample):
        return self.training_loss(sample)
