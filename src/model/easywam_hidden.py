from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from utils.logging_config import get_logger

from .component.action_dit import ActionDiT, StateEncoder
from .helpers.gradient import gradient_checkpoint_forward
from .backbone.wan22.loader import load_wan22_ti2v_5b_components
from .schedulers.scheduler_continuous import ContinuousFlowMatchScheduler


logger = get_logger(__name__)


class EasyWAMHidden(nn.Module):
    """EasyWAM-Hidden model with cached video-hidden action conditioning."""

    inference_compile_targets = ("forward_video", "forward_action")

    def __init__(
        self,
        video_dit,
        action_dit: ActionDiT,
        vae,
        state_dim: int,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        projector_hidden_dim: int = 64,
        video_hidden_layer: int = 17,
        detach_video_hidden: bool = True,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
    ):
        super().__init__()
        self.backbone_name = getattr(video_dit, "backbone_name", "wan22")
        self.dit = nn.ModuleDict(
            {
                "video_dit": video_dit,
                "action_dit": action_dit,
                "video_context_projector": nn.Linear(
                    int(video_dit.hidden_dim),
                    int(action_dit.text_dim),
                ),
                "state_encoder": StateEncoder(
                    state_dim=int(state_dim),
                    hidden_dim=int(action_dit.hidden_dim),
                    projector_hidden_dim=int(projector_hidden_dim),
                ),
            }
        )
        self.action_dim = int(action_dit.action_dim)
        self.state_dim = int(state_dim)
        self.video_hidden_layer = int(video_hidden_layer)
        self.detach_video_hidden = bool(detach_video_hidden)
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        if not 0 <= self.video_hidden_layer < len(video_dit.blocks):
            raise ValueError(
                "`video_hidden_layer` must be a valid 0-based Video DiT block index, "
                f"got {self.video_hidden_layer} for {len(video_dit.blocks)} blocks."
            )
        if int(video_dit.patch_size[0]) != 1:
            raise ValueError(
                "EasyWAM-Hidden video padding alignment requires temporal patch_size=1, "
                f"got {video_dit.patch_size[0]}."
            )

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
        self.to(device=self.device, dtype=self.torch_dtype)

    @property
    def video_dit(self):
        return self.dit["video_dit"]

    @property
    def action_dit(self):
        return self.dit["action_dit"]

    @staticmethod
    def _build_video_token_mask(
        image_is_pad: Optional[torch.Tensor],
        *,
        latent_frames: int,
        tokens_per_frame: int,
        temporal_downsample_factor: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if image_is_pad is None:
            return None
        if image_is_pad.ndim != 2:
            raise ValueError(
                f"`image_is_pad` must be [B,T], got {tuple(image_is_pad.shape)}"
            )
        temporal_factor = int(temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(
                "`temporal_downsample_factor` must be positive, "
                f"got {temporal_downsample_factor}."
            )
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent frames: "
                f"frames={image_is_pad.shape[1]}, factor={temporal_factor}."
            )

        image_is_pad = image_is_pad.to(device=device, dtype=torch.bool)
        if not bool(image_is_pad.any().item()):
            return None
        tail_is_pad = image_is_pad[:, 1:].view(
            image_is_pad.shape[0], -1, temporal_factor
        ).all(dim=2)
        latent_is_pad = torch.cat([image_is_pad[:, :1], tail_is_pad], dim=1)
        if latent_is_pad.shape[1] != latent_frames:
            raise ValueError(
                "Video padding/latent length mismatch: "
                f"mask={latent_is_pad.shape[1]}, latent_frames={latent_frames}."
            )
        return (~latent_is_pad).repeat_interleave(tokens_per_frame, dim=1)

    def forward_video(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
        *,
        fuse_vae_embedding_in_latents: bool,
        image_is_pad: Optional[torch.Tensor] = None,
        temporal_downsample_factor: int = 1,
        return_prediction: bool = True,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        video_pre = self.video_dit.pre_dit(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        tokens = video_pre["tokens"]
        self_attn_mask = (
            self.video_dit.build_structured_video_attention_mask(
                video_seq_len=tokens.shape[1],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=tokens.device,
            )
            if self.video_dit.video_attention_mask_mode != "bidirectional"
            else None
        )
        video_token_mask = self._build_video_token_mask(
            image_is_pad,
            latent_frames=int(video_pre["meta"]["grid_size"][0]),
            tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            temporal_downsample_factor=temporal_downsample_factor,
            device=tokens.device,
        )

        captured_hidden = None
        for layer_index, block in enumerate(self.video_dit.blocks):
            if self.video_dit.use_gradient_checkpointing:
                def _block_forward(value, index=layer_index):
                    return self.video_dit.forward_block(
                        index, value, video_pre, self_attn_mask
                    )

                tokens = gradient_checkpoint_forward(
                    _block_forward,
                    self.video_dit.use_gradient_checkpointing,
                    tokens,
                )
            else:
                tokens = self.video_dit.forward_block(
                    layer_index, tokens, video_pre, self_attn_mask
                )
            if layer_index == self.video_hidden_layer:
                captured_hidden = tokens.detach() if self.detach_video_hidden else tokens
                if not return_prediction:
                    break

        if captured_hidden is None:
            raise RuntimeError(
                f"Video hidden was not captured at block {self.video_hidden_layer}."
            )
        prediction = (
            self.video_dit.post_dit(tokens, video_pre) if return_prediction else None
        )
        return prediction, captured_hidden, video_token_mask

    def forward_action(
        self,
        action: torch.Tensor,
        timestep: torch.Tensor,
        state: torch.Tensor,
        video_hidden: torch.Tensor,
        video_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if state.ndim == 2:
            state = state.unsqueeze(1)
        if state.ndim != 3 or state.shape[-1] != self.state_dim:
            raise ValueError(
                f"`state` must be [B,S,{self.state_dim}], got {tuple(state.shape)}"
            )

        video_context = self.dit["video_context_projector"](video_hidden)
        action_pre = self.action_dit.pre_dit(
            action_tokens=action,
            timestep=timestep,
            context=video_context,
            context_mask=video_token_mask,
        )
        state_tokens = self.dit["state_encoder"](
            state.to(device=action.device, dtype=action.dtype)
        )
        state_len = state_tokens.shape[1]
        tokens = torch.cat([state_tokens, action_pre["tokens"]], dim=1)
        if tokens.shape[1] > self.action_dit.freqs.shape[0]:
            raise ValueError(
                f"State/action token length {tokens.shape[1]} exceeds ActionDiT RoPE cache "
                f"{self.action_dit.freqs.shape[0]}."
            )
        freqs = self.action_dit.freqs[: tokens.shape[1]].view(
            tokens.shape[1], 1, -1
        ).to(tokens.device)

        context_attn_mask = None
        if video_token_mask is not None:
            context_attn_mask = video_token_mask.unsqueeze(1).expand(
                -1, tokens.shape[1], -1
            )

        for block in self.action_dit.blocks:
            if self.action_dit.use_gradient_checkpointing:
                tokens = gradient_checkpoint_forward(
                    block,
                    self.action_dit.use_gradient_checkpointing,
                    tokens,
                    action_pre["context"],
                    action_pre["t_mod"],
                    freqs,
                    context_mask=context_attn_mask,
                )
            else:
                tokens = block(
                    tokens,
                    action_pre["context"],
                    action_pre["t_mod"],
                    freqs,
                    context_mask=context_attn_mask,
                )
        return self.action_dit.action_decoder(tokens[:, state_len:])

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
        image_is_pad: Optional[torch.Tensor] = None,
        temporal_downsample_factor: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_video, video_hidden, video_token_mask = self.forward_video(
            x=x,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            image_is_pad=image_is_pad,
            temporal_downsample_factor=temporal_downsample_factor,
            return_prediction=True,
        )
        pred_action = self.forward_action(
            action=action,
            timestep=timestep_action,
            state=state,
            video_hidden=video_hidden,
            video_token_mask=video_token_mask,
        )
        if pred_video is None:
            raise RuntimeError("Video prediction is required during joint forward.")
        return pred_video, pred_action


    @classmethod
    def from_backbone_pretrained(
        cls,
        backbone: dict[str, Any],
        *,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None,
        action_dim: int,
        state_dim: int,
        projector_hidden_dim: int = 64,
        video_hidden_layer: int = 17,
        detach_video_hidden: bool = True,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
    ) -> "EasyWAMHidden":
        from .backbone.loader import load_easywam_backbone, normalize_backbone_config

        cfg = normalize_backbone_config(backbone)
        action_cfg = dict(action_dit_config)
        action_cfg["action_dim"] = int(action_dim)
        components = load_easywam_backbone(
            cfg,
            device=device,
            torch_dtype=torch_dtype,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
        )
        action_dit = ActionDiT.from_pretrained(
            action_dit_config=action_cfg,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        model = cls(
            video_dit=components.dit,
            action_dit=action_dit,
            vae=components.vae,
            state_dim=state_dim,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(cfg["text_dim"]),
            projector_hidden_dim=projector_hidden_dim,
            video_hidden_layer=video_hidden_layer,
            detach_video_hidden=detach_video_hidden,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
        )
        model.backbone_name = cfg["name"]
        if cfg["name"] == "cosmos25":
            from .schedulers.scheduler_flow_unipc import FlowUniPCScheduler
            scheduler_cfg = dict(cfg.get("video_scheduler", {}))
            model.train_video_scheduler = FlowUniPCScheduler(
                video_num_train_timesteps,
                float(scheduler_cfg.get("train_shift", video_train_shift)),
                use_karras_sigmas=False,
                time_distribution=str(scheduler_cfg.get("time_distribution", "logitnormal")),
                training_weight_method=str(scheduler_cfg.get("training_weight", "uniform")),
            )
            model.infer_video_scheduler = FlowUniPCScheduler(
                video_num_train_timesteps,
                float(scheduler_cfg.get("infer_shift", video_infer_shift)),
                use_karras_sigmas=bool(scheduler_cfg.get("use_karras_sigmas", True)),
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
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        action_dim: int | None = None,
        state_dim: int | None = None,
        projector_hidden_dim: int = 64,
        video_hidden_layer: int = 17,
        detach_video_hidden: bool = True,
        skip_dit_load_from_pretrain: bool = False,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
    ) -> "EasyWAMHidden":
        if video_dit_config is None or action_dit_config is None:
            raise ValueError(
                "`video_dit_config` and `action_dit_config` are required for EasyWAM-Hidden."
            )
        if action_dim is None or state_dim is None:
            raise ValueError("`action_dim` and `state_dim` are required for EasyWAM-Hidden.")
        action_dit_config = dict(action_dit_config)
        configured_action_dim = int(action_dit_config.get("action_dim", action_dim))
        if configured_action_dim != int(action_dim):
            raise ValueError(
                "`action_dit_config.action_dim` must match `action_dim`: "
                f"{configured_action_dim} != {action_dim}."
            )
        action_dit_config["action_dim"] = int(action_dim)

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
        action_dit = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        model = cls(
            video_dit=components.dit,
            action_dit=action_dit,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            state_dim=int(state_dim),
            projector_hidden_dim=int(projector_hidden_dim),
            video_hidden_layer=int(video_hidden_layer),
            detach_video_hidden=bool(detach_video_hidden),
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN"
                if skip_dit_load_from_pretrain
                else action_dit_pretrained_path
            ),
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
        ids, mask = self.tokenizer(
            prompt, return_mask=True, add_special_tokens=True
        )
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    @torch.no_grad()
    def _encode_video_latents(
        self,
        video_tensor,
    ):
        return self.vae.encode(video_tensor, device=self.device)

    @torch.no_grad()
    def _encode_input_image_latents_tensor(
        self,
        input_image: torch.Tensor,
    ):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if (
            input_image.ndim != 4
            or input_image.shape[0] != 1
            or input_image.shape[1] != 3
        ):
            raise ValueError(
                "`input_image` must have shape [1,3,H,W] or [3,H,W], "
                f"got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode(
            [image],
            device=self.device,
        )
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(
        self,
        latents,
    ):
        video_tensor = self.vae.decode(latents, device=self.device)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample):
        video = sample["video"]
        context = sample.get("context")
        context_mask = sample.get("context_mask")
        action = sample.get("action")
        state = sample.get("proprio")
        if context is None or context_mask is None:
            raise ValueError(
                "EasyWAM-Hidden training requires `context` and `context_mask`."
            )
        if action is None:
            raise ValueError("EasyWAM-Hidden training requires `action`.")
        if state is None:
            raise ValueError("EasyWAM-Hidden training requires `proprio` as state.")
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(
                f"`video` must be [B,3,T,H,W], got {tuple(video.shape)}"
            )
        if action.ndim != 3:
            raise ValueError(
                f"`action` must be [B,T,D], got {tuple(action.shape)}"
            )
        if state.ndim != 3:
            raise ValueError(
                f"`proprio` must be [B,T,D], got {tuple(state.shape)}"
            )

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                "Video spatial dims must be multiples of 16, "
                f"got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError("EasyWAM-Hidden requires at least 2 video frames.")
        if action.shape[0] != batch_size:
            raise ValueError("Action batch size must match video batch size.")
        if state.shape[0] != batch_size:
            raise ValueError("State batch size must match video batch size.")

        input_video = video.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        input_latents = self._encode_video_latents(input_video)
        first_frame_latents = input_latents[:, :, 0:1]

        context = context.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        context_mask = context_mask.to(
            device=self.device, dtype=torch.bool, non_blocking=True
        )
        action = action.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        state = state[:, 0:1].to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )

        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        image_is_pad = sample.get("image_is_pad")
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )

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
        video_loss_token = F.mse_loss(
            pred_video.float(), target_video.float(), reduction="none"
        ).mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(
            image_is_pad.shape[0], -1, temporal_factor
        ).all(dim=2)
        if latent_tail_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={latent_tail_is_pad.shape[1]}, "
                f"loss steps={video_loss_token.shape[1]}."
            )
        valid = (~latent_tail_is_pad).to(
            device=video_loss_token.device, dtype=video_loss_token.dtype
        )
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def _prepare_context(
        self,
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError(
                "`prompt` and `context/context_mask` are mutually exclusive."
            )
        if not use_prompt and not use_context:
            raise ValueError(
                "Either `prompt` or both `context/context_mask` must be provided."
            )
        if use_prompt:
            return self.encode_prompt(prompt)
        if context is None or context_mask is None:
            raise ValueError(
                "`context` and `context_mask` must be both provided together."
            )
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        return (
            context.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            ),
            context_mask.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            ),
        )

    def training_loss(self, sample):
        inputs = self.build_inputs(sample)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]

        future_latents = input_latents[:, :, 1:]
        noise_video = torch.randn_like(future_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        noisy_future = self.train_video_scheduler.add_noise(
            future_latents, noise_video, timestep_video
        )
        target_video = self.train_video_scheduler.training_target(
            future_latents, noise_video, timestep_video
        )
        noisy_video = torch.cat([inputs["first_frame_latents"], noisy_future], dim=2)

        action = inputs["action"]
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

        pred_video, pred_action = self._forward_dit(
            x=noisy_video,
            timestep_video=timestep_video,
            action=noisy_action,
            timestep_action=timestep_action,
            state=inputs["state"],
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            fuse_vae_embedding_in_latents=bool(
                getattr(self.video_dit, "fuse_vae_embedding_in_latents", False)
            ),
            image_is_pad=inputs["image_is_pad"],
            temporal_downsample_factor=int(self.vae.temporal_downsample_factor),
        )
        pred_video = pred_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=inputs["image_is_pad"],
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)
        if inputs["action_is_pad"] is not None:
            valid = (~inputs["action_is_pad"]).to(
                device=action_loss_token.device, dtype=action_loss_token.dtype
            )
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid.sum(
                dim=1
            ).clamp(min=1.0)
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = loss_action + self.loss_lambda_video * loss_video
        return loss_total, {
            "loss_video": loss_video.detach(),
            "loss_action": loss_action.detach(),
            "weighted_loss_action": loss_action.detach(),
            "weighted_loss_video": loss_video.detach() * self.loss_lambda_video,
        }

    def _prepare_inference_inputs(
        self,
        *,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        proprio: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        seed: Optional[int],
        rand_device: str,
    ) -> dict[str, torch.Tensor]:
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[:2] != (1, 3):
            raise ValueError(
                "`input_image` must be [1,3,H,W] or [3,H,W], "
                f"got {tuple(input_image.shape)}"
            )
        if proprio is None:
            raise ValueError("EasyWAM-Hidden inference requires `proprio` as state.")
        if proprio.ndim == 1:
            proprio = proprio.view(1, 1, -1)
        elif proprio.ndim == 2:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim != 3 or proprio.shape[-1] != self.state_dim:
            raise ValueError(
                f"`proprio` must end in state_dim={self.state_dim}, got {tuple(proprio.shape)}"
            )

        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(
            height, width, num_video_frames
        )
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized to multiples of 16, got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor
        video_generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        action_generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image.to(device=self.device, dtype=self.torch_dtype),
        )
        latents_video[:, :, 0:1] = first_frame_latents
        context, context_mask = self._prepare_context(prompt, context, context_mask)
        return {
            "latents_video": latents_video,
            "latents_action": latents_action,
            "first_frame_latents": first_frame_latents,
            "state": proprio[:, 0:1].to(device=self.device, dtype=self.torch_dtype),
            "context": context,
            "context_mask": context_mask,
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
        num_video_frames: int = 5,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        **kwargs,
    ) -> dict[str, Any]:
        del negative_prompt, text_cfg_scale, kwargs
        self.eval()
        inputs = self._prepare_inference_inputs(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            seed=seed,
            rand_device=rand_device,
        )
        video_timesteps, _ = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=inputs["latents_video"].dtype,
            shift_override=sigma_shift,
        )
        _, video_hidden, video_token_mask = self.forward_video(
            x=inputs["latents_video"],
            timestep=video_timesteps[0].unsqueeze(0),
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            fuse_vae_embedding_in_latents=bool(
                getattr(self.video_dit, "fuse_vae_embedding_in_latents", False)
            ),
            return_prediction=False,
        )

        action_timesteps, action_deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=len(video_timesteps),
            device=self.device,
            dtype=inputs["latents_action"].dtype,
            shift_override=sigma_shift,
        )
        latents_action = inputs["latents_action"]
        for step_t, step_delta in zip(action_timesteps, action_deltas):
            pred_action = self.forward_action(
                action=latents_action,
                timestep=step_t.unsqueeze(0),
                state=inputs["state"],
                video_hidden=video_hidden,
                video_token_mask=video_token_mask,
            )
            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta, latents_action
            )
        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        }

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
        **kwargs,
    ) -> dict[str, Any]:
        del action, negative_prompt, text_cfg_scale, kwargs
        self.eval()
        inputs = self._prepare_inference_inputs(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            seed=seed,
            rand_device=rand_device,
        )
        video_timesteps, video_deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=inputs["latents_video"].dtype,
            shift_override=sigma_shift,
        )
        action_timesteps, action_deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=len(video_timesteps),
            device=self.device,
            dtype=inputs["latents_action"].dtype,
            shift_override=sigma_shift,
        )

        latents_video = inputs["latents_video"]
        latents_action = inputs["latents_action"]
        cached_hidden = None
        cached_mask = None
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            video_timesteps, video_deltas, action_timesteps, action_deltas
        ):
            pred_video, current_hidden, current_mask = self.forward_video(
                x=latents_video,
                timestep=step_t_video.unsqueeze(0),
                context=inputs["context"],
                context_mask=inputs["context_mask"],
                fuse_vae_embedding_in_latents=bool(
                    getattr(self.video_dit, "fuse_vae_embedding_in_latents", False)
                ),
                return_prediction=True,
            )
            if cached_hidden is None:
                cached_hidden = current_hidden
                cached_mask = current_mask
            pred_action = self.forward_action(
                action=latents_action,
                timestep=step_t_action.unsqueeze(0),
                state=inputs["state"],
                video_hidden=cached_hidden,
                video_token_mask=cached_mask,
            )
            if pred_video is None:
                raise RuntimeError("Joint inference requires a video prediction.")
            pred_video[:, :, 0:1] = 0
            latents_video = self.infer_video_scheduler.step(
                pred_video, step_delta_video, latents_video
            )
            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta_action, latents_action
            )
            latents_video[:, :, 0:1] = inputs["first_frame_latents"]

        return {
            "video": self._decode_latents(latents_video),
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

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
    ) -> dict[str, Any]:
        del action_cfg_scale
        if action_horizon is None:
            raise ValueError("`action_horizon` is required for EasyWAM-Hidden inference.")
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
                "backbone_name": getattr(self, "backbone_name", "wan22"),
            }
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
        checkpoint_backbone = payload.get("backbone_name")
        if checkpoint_backbone is not None and checkpoint_backbone != getattr(self, "backbone_name", "wan22"):
            raise ValueError(
                f"Checkpoint backbone {checkpoint_backbone!r} does not match model backbone "
                f"{getattr(self, 'backbone_name', 'wan22')!r}."
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
