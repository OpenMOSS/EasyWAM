from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from utils.logging_config import get_logger

from .component.attention import AttentionSegment, StructuredAttentionMask, build_structured_attention_mask
from .easywam_mot_joint import EasyWAMMoTJoint

logger = get_logger(__name__)


def _concat_sequence_values(left: Any, right: Any, dim: int) -> Any:
    """Concatenate tensor-like backbone metadata along its token/frame axis."""
    if left is None and right is None:
        return None
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.ndim <= dim or right.ndim <= dim:
            raise ValueError(
                f"Cannot concatenate values along dim={dim}: "
                f"left={tuple(left.shape)}, right={tuple(right.shape)}"
            )
        return torch.cat([left, right], dim=dim)
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise ValueError(
                "Backbone metadata dictionaries must have matching keys, "
                f"got {sorted(left)} and {sorted(right)}."
            )
        return {key: _concat_sequence_values(left[key], right[key], dim) for key in left}
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        if len(left) != len(right):
            raise ValueError("Backbone metadata sequences must have matching lengths.")
        values = [_concat_sequence_values(a, b, dim) for a, b in zip(left, right)]
        return type(left)(values)
    raise TypeError(
        "Cannot concatenate incompatible backbone metadata values: "
        f"{type(left).__name__} and {type(right).__name__}."
    )


class EasyWAMMoTIDM(EasyWAMMoTJoint):
    """EasyWAM MoT-IDM model with teacher-forced conditional video."""

    model_variant = "mot_idm"

    def __init__(self, *args, video_cond_noise_prob: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        video_cond_noise_prob = float(video_cond_noise_prob)
        if not 0.0 <= video_cond_noise_prob <= 1.0:
            raise ValueError(
                "`video_cond_noise_prob` must be between 0 and 1, "
                f"got {video_cond_noise_prob}."
            )
        self.video_cond_noise_prob = video_cond_noise_prob

    @torch.no_grad()
    def _build_teacher_forcing_attention_mask(
        self,
        noisy_video_seq_len: int,
        cond_video_seq_len: int,
        action_seq_len: int,
        noisy_video_tokens_per_frame: int,
        cond_video_tokens_per_frame: int,
        device: torch.device,
    ) -> StructuredAttentionMask:
        if noisy_video_tokens_per_frame != cond_video_tokens_per_frame:
            raise ValueError(
                "Teacher-forcing requires identical `tokens_per_frame` for noisy and cond video branches, "
                f"got {noisy_video_tokens_per_frame} and {cond_video_tokens_per_frame}."
            )

        noisy_end = noisy_video_seq_len
        cond_end = noisy_end + cond_video_seq_len
        total_seq_len = cond_end + action_seq_len
        mode = self.video_expert.video_attention_mask_mode
        segments: list[AttentionSegment] = []

        def append_video_segments(offset: int, length: int) -> None:
            first_frame_tokens = min(noisy_video_tokens_per_frame, length)
            if mode == "bidirectional":
                segments.append(AttentionSegment(offset, offset + length, ((offset, offset + length),)))
            elif mode == "first_frame_causal":
                segments.append(
                    AttentionSegment(
                        offset,
                        offset + first_frame_tokens,
                        ((offset, offset + first_frame_tokens),),
                    )
                )
                if first_frame_tokens < length:
                    segments.append(
                        AttentionSegment(
                            offset + first_frame_tokens,
                            offset + length,
                            ((offset, offset + length),),
                        )
                    )
            elif mode == "per_frame_causal":
                if length % noisy_video_tokens_per_frame != 0:
                    raise ValueError(
                        "Video sequence length must be divisible by tokens_per_frame "
                        "in per_frame_causal mode."
                    )
                for frame_start in range(0, length, noisy_video_tokens_per_frame):
                    frame_end = frame_start + noisy_video_tokens_per_frame
                    segments.append(
                        AttentionSegment(
                            offset + frame_start,
                            offset + frame_end,
                            ((offset, offset + frame_end),),
                        )
                    )
            else:
                raise ValueError(f"Unsupported video attention mask mode: {mode}")

        append_video_segments(0, noisy_video_seq_len)
        append_video_segments(noisy_end, cond_video_seq_len)
        if action_seq_len > 0:
            segments.append(
                AttentionSegment(
                    cond_end,
                    total_seq_len,
                    ((noisy_end, cond_end), (cond_end, total_seq_len)),
                )
            )
        return build_structured_attention_mask(
            query_len=total_seq_len,
            key_len=total_seq_len,
            segments=segments,
            device=device,
        )

    def training_loss(self, sample):
        inputs = self.build_inputs(sample)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        fuse_flag = inputs["fuse_vae_embedding_in_latents"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents_noisy = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
        if inputs["first_frame_latents"] is not None:
            latents_noisy[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        cond_noise_mask = torch.rand((batch_size,), device=self.device) < float(self.video_cond_noise_prob)
        timestep_video_cond = torch.zeros_like(timestep_video, dtype=input_latents.dtype, device=self.device)
        latents_cond = input_latents
        if bool(cond_noise_mask.any()):
            timestep_video_cond_sampled = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=input_latents.dtype,
            )
            timestep_video_cond = torch.where(
                cond_noise_mask, timestep_video_cond_sampled, timestep_video_cond
            )
            noise_video_cond = torch.randn_like(input_latents)
            latents_cond_noisy = self.train_video_scheduler.add_noise(
                input_latents, noise_video_cond, timestep_video_cond_sampled
            )
            cond_noise_selector = cond_noise_mask.view(batch_size, 1, 1, 1, 1)
            latents_cond = torch.where(cond_noise_selector, latents_cond_noisy, input_latents)
        if inputs["first_frame_latents"] is not None:
            latents_cond = latents_cond.clone()
            latents_cond[:, :, 0:1] = inputs["first_frame_latents"]

        video_pre_noisy = self.video_expert.pre_dit(
            x=latents_noisy,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_pre_cond = self.video_expert.pre_dit(
            x=latents_cond,
            timestep=timestep_video_cond,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        noisy_video_seq_len = int(video_pre_noisy["tokens"].shape[1])
        cond_video_seq_len = int(video_pre_cond["tokens"].shape[1])
        noisy_tokens_per_frame = int(video_pre_noisy["meta"]["tokens_per_frame"])
        cond_tokens_per_frame = int(video_pre_cond["meta"]["tokens_per_frame"])

        merged_video_tokens = torch.cat(
            [video_pre_noisy["tokens"], video_pre_cond["tokens"]], dim=1
        )
        merged_video_freqs = _concat_sequence_values(
            video_pre_noisy["freqs"], video_pre_cond["freqs"], dim=0
        )
        merged_video_t_mod = _concat_sequence_values(
            video_pre_noisy["t_mod"], video_pre_cond["t_mod"], dim=1
        )
        merged_video_context_mask = _concat_sequence_values(
            video_pre_noisy["context_mask"], video_pre_cond["context_mask"], dim=1
        ) if video_pre_noisy["context_mask"] is not None else None
        if merged_video_context_mask is None and video_pre_cond["context_mask"] is not None:
            raise ValueError("Noisy and conditional video context masks must be both present or both absent.")

        attention_mask = self._build_teacher_forcing_attention_mask(
            noisy_video_seq_len=noisy_video_seq_len,
            cond_video_seq_len=cond_video_seq_len,
            action_seq_len=action_pre["tokens"].shape[1],
            noisy_video_tokens_per_frame=noisy_tokens_per_frame,
            cond_video_tokens_per_frame=cond_tokens_per_frame,
            device=merged_video_tokens.device,
        )
        tokens_out = self.mot(
            embeds_all={"video": merged_video_tokens, "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": merged_video_freqs, "action": action_pre["freqs"]},
            context_all={
                "video": {
                    "context": video_pre_noisy["context"],
                    "mask": merged_video_context_mask,
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={"video": merged_video_t_mod, "action": action_pre["t_mod"]},
        )

        pred_video = self.video_expert.post_dit(
            tokens_out["video"][:, :noisy_video_seq_len], video_pre_noisy
        )
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

        loss_total = self.loss_lambda_action * loss_action + self.loss_lambda_video * loss_video
        loss_dict = {
            "loss_video": loss_video.detach(),
            "loss_action": loss_action.detach(),
            "weighted_loss_action": loss_action.detach() * self.loss_lambda_action,
            "weighted_loss_video": loss_video.detach() * self.loss_lambda_video,
        }
        return loss_total, loss_dict

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
    ) -> dict[str, Any]:
        del action, negative_prompt, text_cfg_scale, test_action_with_infer_action
        self.eval()

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(
            height, width, num_video_frames
        )
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                "`input_image` must be resized before infer, "
                f"expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}")

        if proprio is not None:
            if self.state_dim is None:
                raise ValueError("`proprio` was provided but `state_dim=None` so state encoding is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim != 2 or proprio.shape[0] != 1:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.state_dim:
                raise ValueError(f"`proprio` last dim must be {self.state_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got "
                    f"{tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_state_to_context(
                context=context,
                context_mask=context_mask,
                state=proprio,
            )
        video_context, action_context, video_cross_kv_cache, action_cross_kv_cache = (
            self._prepare_inference_cross_attention(context)
        )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video in zip(infer_timesteps_video, infer_deltas_video):
            timestep_video = step_t_video.unsqueeze(0).to(
                dtype=latents_video.dtype, device=self.device
            )
            pred_video = self._predict_video_noise(
                latents_video=latents_video,
                timestep_video=timestep_video,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                projected_context=video_context,
                cross_kv_cache=video_cross_kv_cache,
            )
            latents_video = self.infer_video_scheduler.step(
                pred_video, step_delta_video, latents_video
            )
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        timestep_video_cond = torch.zeros(
            (latents_video.shape[0],), dtype=latents_video.dtype, device=self.device
        )
        video_pre_cond = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video_cond,
            context=video_context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
            context_is_projected=True,
            cross_kv_cache=video_cross_kv_cache,
        )
        video_seq_len = int(video_pre_cond["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre_cond["meta"]["tokens_per_frame"]),
            device=video_pre_cond["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre_cond["tokens"],
            video_freqs=video_pre_cond["freqs"],
            video_t_mod=video_pre_cond["t_mod"],
            video_context_payload={
                "context": video_pre_cond["context"],
                "mask": video_pre_cond["context_mask"],
                "kv_cache": video_pre_cond["cross_kv_cache"],
            },
            video_attention_mask=attention_mask.slice(0, video_seq_len, 0, video_seq_len),
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            # Cosmos' Karras schedule can contain one more point than the requested step count.
            num_inference_steps=len(infer_timesteps_video),
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(
                dtype=latents_action.dtype, device=self.device
            )
            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                action_context=action_context,
                action_cross_kv_cache=action_cross_kv_cache,
            )
            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta_action, latents_action
            )

        result = {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
        if decode_video:
            result["video"] = self._decode_latents(latents_video)
        return result

    @torch.inference_mode()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        out = self.infer_joint(
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
        return {"action": out["action"]}
