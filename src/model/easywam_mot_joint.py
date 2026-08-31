from __future__ import annotations

from typing import Any, Optional

import torch

from utils.logging_config import get_logger

from .component.attention import AttentionSegment, StructuredAttentionMask, build_structured_attention_mask
from .easywam_mot import EasyWAMMoT

logger = get_logger(__name__)


class EasyWAMMoTJoint(EasyWAMMoT):
    """EasyWAM MoT-Joint model with joint video/action denoising."""

    model_variant = "mot_joint"

    @torch.no_grad()
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
        segments: list[AttentionSegment] = []

        if mode == "bidirectional":
            segments.append(AttentionSegment(0, video_seq_len, ((0, video_seq_len),)))
        elif mode == "first_frame_causal":
            segments.append(AttentionSegment(0, first_frame_tokens, ((0, first_frame_tokens),)))
            if first_frame_tokens < video_seq_len:
                segments.append(
                    AttentionSegment(first_frame_tokens, video_seq_len, ((0, video_seq_len),))
                )
        elif mode == "per_frame_causal":
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError(
                    "`video_seq_len` must be divisible by `video_tokens_per_frame` "
                    "in per_frame_causal mode."
                )
            for frame_start in range(0, video_seq_len, video_tokens_per_frame):
                frame_end = frame_start + video_tokens_per_frame
                segments.append(AttentionSegment(frame_start, frame_end, ((0, frame_end),)))
        else:
            raise ValueError(f"Unsupported video attention mask mode: {mode}")

        if action_seq_len > 0:
            segments.append(
                AttentionSegment(
                    video_seq_len,
                    total_seq_len,
                    ((0, video_seq_len), (video_seq_len, total_seq_len)),
                )
            )

        return build_structured_attention_mask(
            query_len=total_seq_len,
            key_len=total_seq_len,
            segments=segments,
            device=device,
        )

    @torch.no_grad()
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
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        if test_action_with_infer_action:
            logger.warning(
                "EasyWAMMoTJoint.infer_joint always uses joint video/action denoising; "
                "ignoring test_action_with_infer_action=True."
            )
        return super().infer_joint(
            prompt=prompt,
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
        )

    @torch.no_grad()
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
            action=None,
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
        )
        return {"action": out["action"]}
