from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.helpers.static_cache import cache_static_method
from utils.logging_config import get_logger

from .backbone.protocol import BLOCK_PROTOCOL_STAGED
from .component.action_dit import (
    ActionDiT,
    StateEncoder,
    normalize_state_position,
    validate_checkpoint_state_position,
    policy_action_config,
    validate_policy_action_config,
)
from .component.attention import (
    AttentionSegment,
    StructuredAttentionMask,
    prepare_key_padding_mask,
    build_structured_attention_mask,
    elide_fully_valid_attention_mask,
)
from .component.mot import MoT
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
from .helpers import endpoint
from .schedulers.scheduler_continuous import ContinuousFlowMatchScheduler

logger = get_logger(__name__)


class EasyWAMMoT(torch.nn.Module):

    _forward_endpoint = endpoint.predict
    _forward_endpoint_cached_action = endpoint._cached_action

    model_variant = "mot"

    inference_compile_targets = (
        "_predict_video_noise",
        "_predict_action_noise_with_cache",
        "_predict_joint_noise",
    )

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        state_dim: Optional[int] = None,
        state_position: str = "context",
        projector_hidden_dim: int = 64,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
    ):
        super().__init__()
        self.inference_cross_kv_reuse = False
        self.video_expert = video_expert
        self.backbone_name = video_expert.backbone_name
        if getattr(video_expert, "visual_target", None) == "endpoint":
            self.inference_compile_targets = (
                "_forward_endpoint",
                "_forward_endpoint_cached_action",
            )
        self.action_expert = action_expert
        self.mot = mot
        self.dit = self.mot

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
        self.state_dim = None if state_dim is None else int(state_dim)
        self.state_position = normalize_state_position(state_position)
        self.projector_hidden_dim = int(projector_hidden_dim)
        if self.state_dim is not None:
            state_output_dim = (
                self.text_dim
                if self.state_position == "context"
                else int(self.action_expert.hidden_dim)
            )
            self.state_encoder = StateEncoder(
                state_dim=self.state_dim,
                hidden_dim=state_output_dim,
                projector_hidden_dim=self.projector_hidden_dim,
            ).to(torch_dtype)
        else:
            self.state_encoder = None

        self.train_video_scheduler = ContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = ContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = ContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = ContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )

        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)

        self.to(device=self.device, dtype=self.torch_dtype)

    @property
    def video_dit(self):
        return self.video_expert

    @classmethod
    def from_backbone_pretrained(
        cls,
        backbone: dict[str, Any],
        *,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        state_dim: Optional[int] = None,
        state_position: str = "context",
        projector_hidden_dim: int = 64,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        **model_init_kwargs: Any,
    ):
        from .backbone import load_easywam_backbone, normalize_backbone_config

        cfg = normalize_backbone_config(backbone)
        components = load_easywam_backbone(
            cfg,
            device=device,
            torch_dtype=torch_dtype,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
        )
        video_expert = components.dit
        action_expert_cls = ActionDiT
        if components.block_protocol == BLOCK_PROTOCOL_STAGED:
            if action_dit_config.get("preset") != "staged":
                raise ValueError(
                    "Staged mixed attention requires ActionDiT preset=staged"
                )
            action_dit_config = dict(action_dit_config)
            for key, expected in dict(
                num_heads=video_expert.num_heads,
                attn_head_dim=video_expert.attn_head_dim,
                num_layers_double=video_expert.double_layers,
                num_layers_single=video_expert.single_layers,
                attention_backend=video_expert.attention_backend,
            ).items():
                if key in action_dit_config and action_dit_config[key] != expected:
                    raise ValueError(
                        f"MoT ActionDiT {key} must match backbone: {expected}"
                    )
                action_dit_config.setdefault(key, expected)
        action_expert = action_expert_cls.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT num_heads must match the video expert.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT attn_head_dim must match the video expert.")
        if len(action_expert.blocks) != len(video_expert.blocks):
            raise ValueError("ActionDiT num_layers must match the video expert.")
        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=MoT({"video": video_expert, "action": action_expert}),
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(components.text_dim),
            state_dim=state_dim,
            state_position=state_position,
            projector_hidden_dim=projector_hidden_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            **model_init_kwargs,
        )
        model.backbone_name = cfg["name"]
        from .schedulers.factory import build_video_schedulers

        model.train_video_scheduler, model.infer_video_scheduler = (
            build_video_schedulers(
                family=components.dit.scheduler_family,
                config=cfg.get("video_scheduler"),
                num_train_timesteps=video_num_train_timesteps,
                train_shift=video_train_shift,
                infer_shift=video_infer_shift,
            )
        )
        model.train_scheduler = model.train_video_scheduler
        model.infer_scheduler = model.infer_video_scheduler
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": action_dit_pretrained_path,
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
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
            return context.to(device=self.device), mask.to(
                device=self.device, dtype=torch.bool
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        return prompt_emb.to(device=self.device), mask

    def _append_state_to_context(
        self,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if (
            self.state_position != "context"
            or self.state_encoder is None
            or state is None
        ):
            return context, context_mask
        if state.ndim != 2:
            raise ValueError(
                f"`state` must be 2D [B, D], got shape {tuple(state.shape)}"
            )
        if self.state_dim is None or state.shape[1] != self.state_dim:
            raise ValueError(
                f"`state` last dim must be {self.state_dim}, got {state.shape[1]}"
            )
        state_token = self.state_encoder(
            state.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype)
        if context_mask is None:
            return torch.cat([context, state_token], dim=1), None
        state_mask = torch.ones(
            (context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device
        )
        return (
            torch.cat([context, state_token], dim=1),
            torch.cat([context_mask, state_mask], dim=1),
        )

    def _append_state_to_action_sequence(
        self,
        action_pre: dict[str, Any],
        state: Optional[torch.Tensor],
    ) -> dict[str, Any]:
        if self.state_position != "sequence" or self.state_encoder is None:
            return action_pre
        if state is None:
            return action_pre
        if state.ndim != 2:
            raise ValueError(
                f"`state` must be 2D [B,D], got shape {tuple(state.shape)}"
            )
        if self.state_dim is None or state.shape[1] != self.state_dim:
            raise ValueError(
                f"`state` last dim must be {self.state_dim}, got {state.shape[1]}"
            )
        state_tokens = self.state_encoder(
            state.to(device=self.device, dtype=action_pre["tokens"].dtype).unsqueeze(1)
        ).to(dtype=action_pre["tokens"].dtype)
        return self.action_expert.append_state_tokens(action_pre, state_tokens)

    def _state_sequence_length(self, state: Optional[torch.Tensor]) -> int:
        return int(
            self.state_position == "sequence"
            and self.state_encoder is not None
            and state is not None
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor):
        if endpoint.enabled(self):
            return endpoint.encode_video(self, video_tensor)
        z = self.vae.encode(video_tensor, device=self.device)
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor):
        return encode_image_batch(self.vae, input_image, self.device)

    def _decode_latents(self, latents):
        if endpoint.enabled(self):
            return endpoint.decode_latents(self, latents)
        return decode_video_batch(self.vae, latents, self.device)

    @staticmethod
    def _scheduler_timestep_to_unit(timestep: torch.Tensor, scheduler) -> torch.Tensor:
        return timestep / float(scheduler.num_train_timesteps)

    def build_inputs(self, sample):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "EasyWAM-MoT training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(
                f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}"
            )
        if video.shape[1] != 3:
            raise ValueError(
                f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}"
            )

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(
                f"Video T must be > 1 for action-conditioned training, got T={num_frames}"
            )

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for EasyWAM-MoT training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(
                f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}"
            )
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if (
                action_is_pad.shape[0] != batch_size
                or action_is_pad.shape[1] != action_horizon
            ):
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if (
                image_is_pad.shape[0] != batch_size
                or image_is_pad.shape[1] != num_frames
            ):
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )

        input_video = video.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        input_latents = self._encode_video_latents(input_video)

        first_frame_latents = None
        fuse_flag = False
        if self.video_expert.fuse_vae_embedding_in_latents:
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context_mask = elide_fully_valid_attention_mask(context_mask)
        context = context.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        if context_mask is not None:
            context_mask = context_mask.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        state = None
        if self.state_encoder is not None:
            if proprio is None:
                raise ValueError(
                    "`sample['proprio']` is required when `state_dim` is enabled."
                )
            if proprio.ndim != 3:
                raise ValueError(
                    f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}"
                )
            if proprio.shape[2] != self.state_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.state_dim}, got {proprio.shape[2]}"
                )
            state = proprio[:, 0, :]
            context, context_mask = self._append_state_to_context(
                context=context,
                context_mask=context_mask,
                state=state.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )

        return {
            "context": context,
            "context_mask": context_mask,
            "state": state,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    @torch.no_grad()
    @cache_static_method("video_expert.video_attention_mask_mode")
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> StructuredAttentionMask:
        total_seq_len = video_seq_len + action_seq_len
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mode = self.video_expert.video_attention_mask_mode
        segments = []

        if mode == "bidirectional":
            segments.append(AttentionSegment(0, video_seq_len, ((0, video_seq_len),)))
        elif mode == "first_frame_causal":
            segments.append(
                AttentionSegment(0, first_frame_tokens, ((0, first_frame_tokens),))
            )
            if first_frame_tokens < video_seq_len:
                segments.append(
                    AttentionSegment(
                        first_frame_tokens, video_seq_len, ((0, video_seq_len),)
                    )
                )
        elif mode == "per_frame_causal":
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError(
                    "`video_seq_len` must be divisible by `video_tokens_per_frame` in per_frame_causal mode."
                )
            for frame_start in range(0, video_seq_len, video_tokens_per_frame):
                frame_end = frame_start + video_tokens_per_frame
                segments.append(
                    AttentionSegment(frame_start, frame_end, ((0, frame_end),))
                )
        else:
            raise ValueError(f"Unsupported video attention mask mode: {mode}")

        if action_seq_len > 0:
            if first_frame_tokens == video_seq_len:
                action_key_ranges = ((0, total_seq_len),)
            else:
                action_key_ranges = (
                    (0, first_frame_tokens),
                    (video_seq_len, total_seq_len),
                )
            segments.append(
                AttentionSegment(video_seq_len, total_seq_len, action_key_ranges)
            )
        return build_structured_attention_mask(
            query_len=total_seq_len,
            key_len=total_seq_len,
            segments=segments,
            device=device,
        )

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(
            pred_video.float(), target_video.float(), reduction="none"
        ).mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(
                f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}."
            )
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(
            image_is_pad.shape[0], -1, temporal_factor
        ).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(
            device=video_loss_token.device, dtype=video_loss_token.dtype
        )
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def training_loss(self, sample):
        if endpoint.enabled(self):
            return endpoint.training_loss(self, sample)
        inputs = self.build_inputs(sample)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(
            input_latents, noise_video, timestep_video
        )
        target_video = self.train_video_scheduler.training_target(
            input_latents, noise_video, timestep_video
        )

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(
            action, noise_action, timestep_action
        )
        target_action = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action
        )

        video_pre = self.video_expert.pre_dit(
            prepare_rotary_factors=False,
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        action_pre = self.action_expert.pre_dit(
            prepare_rotary_factors=False,
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_pre = self._append_state_to_action_sequence(
            action_pre, inputs.get("state")
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(
                device=action_loss_token.device, dtype=action_loss_token.dtype
            )
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = (
            self.loss_lambda_action * loss_action + self.loss_lambda_video * loss_video
        )
        loss_dict = {
            "loss_video": loss_video.detach(),
            "loss_action": loss_action.detach(),
            "weighted_loss_action": loss_action.detach() * self.loss_lambda_action,
            "weighted_loss_video": loss_video.detach() * self.loss_lambda_video,
        }
        return loss_total, loss_dict

    @torch.no_grad()
    def _prepare_inference_cross_attention(
        self,
        context: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]],
        Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]],
    ]:
        video_context = self.video_expert.project_context(context)
        action_context = (
            self.action_expert.project_context(context)
            if self.action_expert.cross_attention
            else None
        )
        if not self.inference_cross_kv_reuse:
            return video_context, action_context, None, None
        return (
            video_context,
            action_context,
            self.video_expert.build_cross_attention_kv_cache(video_context),
            (
                self.action_expert.build_cross_attention_kv_cache(action_context)
                if self.action_expert.cross_attention
                else None
            ),
        )

    @torch.no_grad()
    def _predict_video_noise(
        self,
        latents_video: torch.Tensor,
        timestep_video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        projected_context: Optional[torch.Tensor] = None,
        cross_kv_cache: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None,
    ) -> torch.Tensor:
        video_pre = self.video_expert.pre_dit(
            prepare_rotary_factors=False,
            x=latents_video,
            timestep=timestep_video,
            context=context if projected_context is None else projected_context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            context_is_projected=projected_context is not None,
            cross_kv_cache=cross_kv_cache,
        )
        video_tokens = video_pre["tokens"]
        attention_mask = None
        if self.video_expert.video_attention_mask_mode != "bidirectional":
            attention_mask = self.video_expert.build_structured_video_attention_mask(
                video_seq_len=video_tokens.shape[1],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_tokens.device,
            )
        for layer_index in range(len(self.video_expert.blocks)):
            video_tokens = self.video_expert.forward_block(
                layer_index,
                video_tokens,
                video_pre,
                attention_mask,
            )
        return self.video_expert.post_dit(video_tokens, video_pre)

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
        video_context: Optional[torch.Tensor] = None,
        action_context: Optional[torch.Tensor] = None,
        video_cross_kv_cache: Optional[
            tuple[tuple[torch.Tensor, torch.Tensor], ...]
        ] = None,
        action_cross_kv_cache: Optional[
            tuple[tuple[torch.Tensor, torch.Tensor], ...]
        ] = None,
        state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            prepare_rotary_factors=False,
            x=latents_video,
            timestep=timestep_video,
            context=context if video_context is None else video_context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            context_is_projected=video_context is not None,
            cross_kv_cache=video_cross_kv_cache,
        )
        action_pre = self.action_expert.pre_dit(
            prepare_rotary_factors=False,
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context if action_context is None else action_context,
            context_mask=context_mask,
            context_is_projected=action_context is not None,
            cross_kv_cache=action_cross_kv_cache,
        )
        action_pre = self._append_state_to_action_sequence(action_pre, state)

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )

        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                    "kv_cache": video_pre["cross_kv_cache"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                    "kv_cache": action_pre["cross_kv_cache"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(
            timestep_action, dtype=first_frame_latents.dtype, device=self.device
        )
        video_pre = self.video_expert.pre_dit(
            prepare_rotary_factors=False,
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            prepare_rotary_factors=False,
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_pre = self._append_state_to_action_sequence(action_pre, state)

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor | StructuredAttentionMask,
        video_seq_len: int,
        action_context: Optional[torch.Tensor] = None,
        action_cross_kv_cache: Optional[
            tuple[tuple[torch.Tensor, torch.Tensor], ...]
        ] = None,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            prepare_rotary_factors=False,
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context if action_context is None else action_context,
            context_mask=context_mask,
            context_is_projected=action_context is not None,
            cross_kv_cache=action_cross_kv_cache,
        )
        action_pre = self._append_state_to_action_sequence(action_pre, state)
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
                "kv_cache": action_pre["cross_kv_cache"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    def _action_only_inference_step_count(
        self,
        num_inference_steps: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        sigma_shift: Optional[float],
    ) -> int:
        video_timesteps, _ = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=device,
            dtype=dtype,
            shift_override=sigma_shift,
        )
        return len(video_timesteps)

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
        test_action_with_infer_action: bool = False,
        decode_video: bool = True,
    ) -> JointBatchResult:
        if endpoint.enabled(self):
            return endpoint.infer(
                self,
                action_only=False,
                **{k: v for k, v in locals().items() if k != "self"},
            )
        validate_batch_inputs(input_image, prompt, context, context_mask)
        self.eval()
        if test_action_with_infer_action:
            raise ValueError(
                "Batch inference does not support `test_action_with_infer_action`."
            )

        batch_size, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(
            height, width, num_video_frames
        )
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if (
                action.ndim != 3
                or action.shape[0] != batch_size
                or action.shape[1] != action_horizon
            ):
                raise ValueError(
                    f"`action` must have shape [B,T,D], got {tuple(action.shape)} with B={batch_size} and T={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.state_dim is None:
                raise ValueError(
                    "`proprio` was provided but `state_dim=None` so `state_encoder` is disabled."
                )
            if proprio.ndim != 2 or proprio.shape[0] != batch_size:
                raise ValueError(
                    f"`proprio` must be [B,D], got shape {tuple(proprio.shape)}"
                )
            if proprio.shape[1] != self.state_dim:
                raise ValueError(
                    f"`proprio` last dim must be {self.state_dim}, got {proprio.shape[1]}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        latents_video = randn_per_sample(
            (self.vae.model.z_dim, latent_t, latent_h, latent_w),
            seeds=seed,
            batch_size=batch_size,
            rand_device=rand_device,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = randn_per_sample(
            (action_horizon, self.action_expert.action_dim),
            seeds=seed,
            batch_size=batch_size,
            rand_device=rand_device,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image
        )
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(self.video_expert.fuse_vae_embedding_in_latents)

        if prompt is not None:
            context, context_mask = self.encode_prompt(prompt)
        else:
            context = context.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
            context_mask = context_mask.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        if proprio is not None:
            context, context_mask = self._append_state_to_context(
                context=context,
                context_mask=context_mask,
                state=proprio,
            )
        context_mask = prepare_key_padding_mask(context_mask)
        video_context, action_context, video_cross_kv_cache, action_cross_kv_cache = (
            self._prepare_inference_cross_attention(context)
        )

        infer_timesteps_video, infer_deltas_video = (
            self.infer_video_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_video.dtype,
                shift_override=sigma_shift,
            )
        )
        infer_timesteps_action, infer_deltas_action = (
            self.infer_action_scheduler.build_inference_schedule(
                # Align action steps with the effective visual inference schedule.
                num_inference_steps=len(infer_timesteps_video),
                device=self.device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.expand(batch_size).to(
                dtype=latents_video.dtype, device=self.device
            )
            timestep_action = step_t_action.expand(batch_size).to(
                dtype=latents_action.dtype, device=self.device
            )

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
                video_context=video_context,
                action_context=action_context,
                video_cross_kv_cache=video_cross_kv_cache,
                action_cross_kv_cache=action_cross_kv_cache,
                state=proprio,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(
                pred_video, step_delta_video, latents_video
            )
            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta_action, latents_action
            )
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        result = {
            "action": latents_action.detach().to(device="cpu", dtype=torch.float32)
        }
        if decode_video:
            result["video"] = self._decode_latents(latents_video)
        return result

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
        test_action_with_infer_action: bool = False,
        decode_video: bool = True,
    ) -> JointInferenceResult:
        input_image = batch_single_tensor(input_image, 3, "input_image")
        proprio = batch_single_state(proprio, history=False)
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
            test_action_with_infer_action=False,
            decode_video=decode_video,
        )
        result["action"] = result["action"][0]
        if "video" in result:
            result["video"] = result["video"][0]
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
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[Union[int, Sequence[Optional[int]]]] = None,
        rand_device: str = "cpu",
        num_video_frames: int = 5,
    ) -> ActionBatchResult:
        if endpoint.enabled(self):
            return endpoint.infer(
                self,
                action_only=True,
                **{k: v for k, v in locals().items() if k != "self"},
            )
        validate_batch_inputs(input_image, prompt, context, context_mask)
        self.eval()
        attention_mode = self.video_expert.video_attention_mask_mode

        if attention_mode == "bidirectional":
            joint_out = self.infer_joint_batch(
                prompt=prompt,
                input_image=input_image,
                num_video_frames=num_video_frames,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                negative_prompt=negative_prompt,
                text_cfg_scale=text_cfg_scale,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                test_action_with_infer_action=False,
                decode_video=False,
            )
            return {"action": joint_out["action"]}
        if attention_mode != "first_frame_causal":
            raise ValueError(
                "`infer_action` supports `video_attention_mask_mode` values "
                "'first_frame_causal' and 'bidirectional', "
                f"got {attention_mode!r}."
            )

        batch_size, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.state_dim is None:
                raise ValueError(
                    "`proprio` was provided but `state_dim=None` so `state_encoder` is disabled."
                )
            if proprio.ndim != 2 or proprio.shape[0] != batch_size:
                raise ValueError(
                    f"`proprio` must be [B,D], got shape {tuple(proprio.shape)}"
                )
            if proprio.shape[1] != self.state_dim:
                raise ValueError(
                    f"`proprio` last dim must be {self.state_dim}, got {proprio.shape[1]}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latents_action = randn_per_sample(
            (action_horizon, self.action_expert.action_dim),
            seeds=seed,
            batch_size=batch_size,
            rand_device=rand_device,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image
        )
        fuse_flag = bool(self.video_expert.fuse_vae_embedding_in_latents)

        if prompt is not None:
            context, context_mask = self.encode_prompt(prompt)
        else:
            context = context.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
            context_mask = context_mask.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        if proprio is not None:
            context, context_mask = self._append_state_to_context(
                context=context,
                context_mask=context_mask,
                state=proprio,
            )
        context_mask = prepare_key_padding_mask(context_mask)
        video_context, action_context, _, action_cross_kv_cache = (
            self._prepare_inference_cross_attention(context)
        )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            prepare_rotary_factors=False,
            x=first_frame_latents,
            timestep=timestep_video,
            context=video_context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
            context_is_projected=True,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=(
                latents_action.shape[1] + self._state_sequence_length(proprio)
            ),
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask.slice(
                0, video_seq_len, 0, video_seq_len
            ),
        )

        action_inference_steps = self._action_only_inference_step_count(
            num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            sigma_shift=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = (
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=action_inference_steps,
                device=self.device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
        )
        for step_t_action, step_delta_action in zip(
            infer_timesteps_action, infer_deltas_action
        ):
            timestep_action = step_t_action.expand(batch_size).to(
                dtype=latents_action.dtype, device=self.device
            )

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                action_context=action_context,
                action_cross_kv_cache=action_cross_kv_cache,
                state=proprio,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta_action, latents_action
            )

        return {
            "action": latents_action.detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.inference_mode()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        num_video_frames: int = 5,
    ) -> ActionInferenceResult:
        input_image = batch_single_tensor(input_image, 3, "input_image")
        proprio = batch_single_state(proprio, history=False)
        context = batch_single_tensor(context, 2, "context")
        context_mask = batch_single_tensor(context_mask, 1, "context_mask")
        result = self.infer_action_batch(
            prompt=[prompt] if prompt is not None else None,
            input_image=input_image,
            action_horizon=action_horizon,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            num_video_frames=num_video_frames,
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
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ):
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

        is_lora_model = has_lora(self.video_dit)
        if is_lora_model:
            payload = build_lora_checkpoint_payload(self, step=step)
        else:
            payload = {
                "mot": self.mot.state_dict(),
                "step": step,
                "torch_dtype": str(self.torch_dtype),
                "backbone_name": self.backbone_name,
                "model_variant": self.model_variant,
            }
        if self.state_encoder is not None and not is_lora_model:
            payload["state_encoder"] = self.state_encoder.state_dict()
        payload["state_position"] = self.state_position
        payload["action_config"] = policy_action_config(self)
        if endpoint.enabled(self):
            payload["structure"] = endpoint.structure(self)
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None, merge_lora: bool = False):
        payload = torch.load(path, map_location="cpu", mmap=True)
        validate_checkpoint_state_position(payload, self.state_position)
        validate_policy_action_config(self, payload)
        if endpoint.enabled(self):
            endpoint.validate_structure(self, payload)
        checkpoint_backbone = payload.get("backbone_name")
        if (
            checkpoint_backbone is not None
            and checkpoint_backbone != self.backbone_name
        ):
            raise ValueError(
                f"Checkpoint backbone {checkpoint_backbone!r} does not match model backbone "
                f"{self.backbone_name!r}."
            )
        checkpoint_variant = payload.get("model_variant")
        if checkpoint_variant is not None and checkpoint_variant != self.model_variant:
            raise ValueError(
                f"Checkpoint model variant {checkpoint_variant!r} does not match current "
                f"model variant {self.model_variant!r}."
            )
        is_lora_payload = payload.get("format") == "trainable_lora"
        if is_lora_payload:
            from .component.lora import load_lora_model_checkpoint_state

            load_lora_model_checkpoint_state(
                self,
                payload,
                merge_after_load=bool(merge_lora),
            )
        elif "mot" in payload:
            if any(
                hasattr(module, "_easywam_lora_config") for module in self.mot.modules()
            ):
                from .component.lora import load_standard_state_dict

                load_standard_state_dict(
                    self.mot,
                    payload["mot"],
                    strict=False,
                )
            else:
                self.mot.load_state_dict(
                    payload["mot"],
                    strict=False,
                )
        elif "dit" in payload:
            logger.warning(
                "Loading an older `dit` checkpoint into the video expert only."
            )
            if hasattr(self.video_expert, "_easywam_lora_config"):
                from .component.lora import load_standard_state_dict

                load_standard_state_dict(
                    self.video_expert, payload["dit"], strict=False
                )
            else:
                self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if not is_lora_payload and self.state_encoder is not None:
            if "state_encoder" in payload:
                self.state_encoder.load_state_dict(
                    payload["state_encoder"], strict=True
                )
            else:
                logger.warning(
                    "Checkpoint has no `state_encoder` weights; keeping current `state_encoder` params."
                )
        elif "state_encoder" in payload:
            logger.warning(
                "Checkpoint contains `state_encoder` weights but current model has `state_dim=None`; ignoring."
            )

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])

        return payload

    def forward(self, sample):
        return self.training_loss(sample)
