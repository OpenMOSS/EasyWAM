from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from utils.logging_config import get_logger

from .component.action_dit import ActionDiT, StateEncoder
from .component.attention import AttentionSegment, StructuredAttentionMask, build_structured_attention_mask
from .component.mot import MoT
from .schedulers.scheduler_continuous import ContinuousFlowMatchScheduler

logger = get_logger(__name__)


class EasyWAMMoT(torch.nn.Module):
    """MoT world model with video/action experts."""

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
        self.video_expert = video_expert
        self.backbone_name = getattr(video_expert, "backbone_name", "wan22")
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.state_dim = None if state_dim is None else int(state_dim)
        self.projector_hidden_dim = int(projector_hidden_dim)
        if self.state_dim is not None:
            if getattr(video_expert, "block_protocol", "main") == "flux2":
                # ImageWAM FLUX.2 checkpoints used a single linear proprio token.
                self.state_encoder = nn.Linear(self.state_dim, self.text_dim).to(torch_dtype)
            else:
                self.state_encoder = StateEncoder(
                    state_dim=self.state_dim,
                    hidden_dim=self.text_dim,
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
        # Optional aliases for consistency with Wan22Core naming.
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
        from .backbone.loader import load_easywam_backbone, normalize_backbone_config

        cfg = normalize_backbone_config(backbone)
        components = load_easywam_backbone(
            cfg,
            device=device,
            torch_dtype=torch_dtype,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
        )
        video_expert = components.dit
        action_expert_cls = ActionDiT
        if components.block_protocol == "flux2":
            from .component.action_dit_flux2 import ActionDiTFlux2

            action_expert_cls = ActionDiTFlux2
            action_dit_config = dict(action_dit_config)
            action_dit_config.update(
                num_heads=int(video_expert.num_heads),
                attn_head_dim=int(video_expert.attn_head_dim),
                num_layers_double=int(video_expert.double_layers),
                num_layers_single=int(video_expert.single_layers),
                attention_backend=str(video_expert.attention_backend),
            )
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
        state_dim: Optional[int] = None,
        projector_hidden_dim: int = 64,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
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
        from .backbone.wan22.loader import load_wan22_ti2v_5b_components

        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for EasyWAM-MoT.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for EasyWAM-MoT.")

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

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            state_dim=state_dim,
            projector_hidden_dim=int(projector_hidden_dim),
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
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
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
        # Frozen encoders are feature extractors and must not inherit train mode
        # when the DiT/action experts enter training.
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
            return context.to(device=self.device), mask.to(device=self.device, dtype=torch.bool)
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_state_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.state_encoder is None or state is None:
            return context, context_mask
        if state.ndim != 2:
            raise ValueError(f"`state` must be 2D [B, D], got shape {tuple(state.shape)}")
        if self.state_dim is None or state.shape[1] != self.state_dim:
            raise ValueError(
                f"`state` last dim must be {self.state_dim}, got {state.shape[1]}"
            )
        state_token = self.state_encoder(
            state.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        state_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, state_token], dim=1),
            torch.cat([context_mask, state_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor):
        z = self.vae.encode(video_tensor, device=self.device)
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents):
        video_tensor = self.vae.decode(latents, device=self.device)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    @staticmethod
    def _scheduler_timestep_to_unit(timestep: torch.Tensor, scheduler) -> torch.Tensor:
        return timestep / float(scheduler.num_train_timesteps)

    @torch.no_grad()
    def _encode_flux2_image_tokens(
        self, image: torch.Tensor, *, time_value: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .backbone.flux2.flux2_video_expert import Flux2VideoExpert

        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"FLUX.2 image must be [B,3,H,W], got {tuple(image.shape)}")
        if image.shape[-2] % 16 or image.shape[-1] % 16:
            raise ValueError("FLUX.2 image height/width must be multiples of 16.")
        latent = self.vae.encode(
            image.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        ).to(dtype=self.torch_dtype)
        tokens = Flux2VideoExpert.pack_latents(latent)
        ids = Flux2VideoExpert.build_img_ids(
            int(latent.shape[0]),
            int(latent.shape[-2]),
            int(latent.shape[-1]),
            time_value=time_value,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return tokens, ids

    @torch.no_grad()
    def _encode_flux2_text(self, sample) -> tuple[torch.Tensor, torch.Tensor]:
        context = sample.get("context", sample.get("text_hidden_states"))
        mask = sample.get("context_mask", sample.get("text_attention_mask"))
        if context is not None:
            if mask is None:
                raise ValueError("FLUX.2 cached context requires context_mask/text_attention_mask.")
            return (
                context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
                mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
            )
        prompt = sample.get("instruction", sample.get("prompt", sample.get("task")))
        if prompt is None:
            raise ValueError("FLUX.2 requires cached context or instruction/prompt/task.")
        if self.text_encoder is None:
            raise ValueError("Online Qwen3 encoding requires backbone.load_text_encoder=true.")
        context, mask = self.text_encoder(prompt)
        return (
            context.to(device=self.device, dtype=self.torch_dtype),
            mask.to(device=self.device, dtype=torch.bool),
        )

    def build_inputs_flux2(self, sample):
        video = sample.get("video")
        target = sample.get("target_latent")
        target_ids = sample.get("target_img_ids")
        if target is None or target_ids is None:
            target_image = sample.get("next_frame", sample.get("target_image"))
            if target_image is None and video is not None:
                if video.ndim != 5:
                    raise ValueError("FLUX.2 video must be [B,3,T,H,W].")
                target_image = video[:, :, -1]
            if target_image is None:
                raise ValueError("FLUX.2 requires a target image or precomputed target tokens.")
            target, target_ids = self._encode_flux2_image_tokens(target_image, time_value=0.0)
        else:
            target = target.to(device=self.device, dtype=self.torch_dtype)
            target_ids = target_ids.to(device=self.device, dtype=self.torch_dtype)

        reference = sample.get("ref_image_latents")
        reference_ids = sample.get("ref_img_ids")
        if reference is None or reference_ids is None:
            reference_image = sample.get("current_frame", sample.get("input_image"))
            if reference_image is None and video is not None:
                reference_image = video[:, :, 0]
            if reference_image is None:
                raise ValueError("FLUX.2 requires a reference image or precomputed reference tokens.")
            reference, reference_ids = self._encode_flux2_image_tokens(reference_image, time_value=10.0)
        else:
            reference = reference.to(device=self.device, dtype=self.torch_dtype)
            reference_ids = reference_ids.to(device=self.device, dtype=self.torch_dtype)

        context, context_mask = self._encode_flux2_text(sample)
        proprio = sample.get("proprio")
        if self.state_encoder is not None:
            if proprio is None or proprio.ndim != 3:
                raise ValueError("FLUX.2 with state_dim requires proprio [B,T,D].")
            context, context_mask = self._append_state_to_context(
                context, context_mask, proprio[:, 0].to(device=self.device, dtype=self.torch_dtype)
            )
        action = sample["action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool)
        action_dim_is_pad = sample.get("action_dim_is_pad")
        if action_dim_is_pad is not None:
            action_dim_is_pad = action_dim_is_pad.to(device=self.device, dtype=torch.bool)
        return {
            "target_latent": target,
            "target_img_ids": target_ids,
            "ref_image_latents": reference,
            "ref_img_ids": reference_ids,
            "context": context,
            "context_mask": context_mask,
            "action": action,
            "action_is_pad": action_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
        }

    @torch.no_grad()
    def _build_mot_attention_mask_flux2(
        self,
        *,
        batch_size: int,
        txt_len: int,
        target_len: int,
        cond_len: int,
        action_len: int,
        device: torch.device,
        text_attention_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        ref_start = txt_len
        target_start = txt_len + cond_len
        action_start = target_start + target_len
        total = action_start + action_len
        mask = torch.zeros(batch_size, total, total, dtype=torch.bool, device=device)
        mask[:, :target_start, :target_start] = True
        mask[:, target_start:action_start, :action_start] = True
        mask[:, action_start:, :target_start] = True
        mask[:, action_start:, action_start:] = True
        if text_attention_mask is not None:
            if tuple(text_attention_mask.shape) != (batch_size, txt_len):
                raise ValueError("FLUX.2 text attention mask shape mismatch.")
            mask[:, :, :ref_start] &= text_attention_mask[:, None, :].to(dtype=torch.bool)
        return {"double_joint": mask, "single": mask.clone()}

    def build_inputs(self, sample):
        if self.mot.block_protocol == "flux2":
            return self.build_inputs_flux2(sample)
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "EasyWAM-MoT training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for EasyWAM-MoT training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
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
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
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
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video)

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.state_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `state_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.state_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.state_dim}, got {proprio.shape[2]}"
                )
            state = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_state_to_context(
                context=context,
                context_mask=context_mask,
                state=state.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

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
        segments = []

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
                    "`video_seq_len` must be divisible by `video_tokens_per_frame` in per_frame_causal mode."
                )
            for frame_start in range(0, video_seq_len, video_tokens_per_frame):
                frame_end = frame_start + video_tokens_per_frame
                segments.append(AttentionSegment(frame_start, frame_end, ((0, frame_end),)))
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
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def training_loss(self, sample):
        if self.mot.block_protocol == "flux2":
            return self._training_loss_flux2(sample)
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
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
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

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2) # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
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

    def _training_loss_flux2(self, sample):
        inputs = self.build_inputs_flux2(sample)
        clean_image = inputs["target_latent"]
        action = inputs["action"]
        batch_size = int(clean_image.shape[0])

        noise_image = torch.randn_like(clean_image)
        timestep_image = self.train_video_scheduler.sample_training_t(
            batch_size, self.device, clean_image.dtype
        )
        noisy_image = self.train_video_scheduler.add_noise(clean_image, noise_image, timestep_image)
        target_image = self.train_video_scheduler.training_target(
            clean_image, noise_image, timestep_image
        )

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size, self.device, action.dtype
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action
        )

        video_pre = self.video_expert.pre_dit(
            x=noisy_image,
            timestep=self._scheduler_timestep_to_unit(
                timestep_image, self.train_video_scheduler
            ),
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            ref_image_hidden_states=inputs["ref_image_latents"],
            target_img_ids=inputs["target_img_ids"],
            ref_img_ids=inputs["ref_img_ids"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=self._scheduler_timestep_to_unit(
                timestep_action, self.train_action_scheduler
            ),
        )
        attention_mask = self._build_mot_attention_mask_flux2(
            batch_size=batch_size,
            txt_len=int(video_pre["txt_len"]),
            target_len=int(video_pre["target_len"]),
            cond_len=int(video_pre["cond_len"]),
            action_len=int(action_pre["tokens"].shape[1]),
            device=noisy_image.device,
            text_attention_mask=video_pre["text_mask"],
        )
        tokens = self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": None},
            context_all={"video": None, "action": {"ids": action_pre["ids"]}},
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        pred_image = self.video_expert.post_dit(tokens["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens["action"], action_pre)

        image_per_sample = F.mse_loss(
            pred_image.float(), target_image.float(), reduction="none"
        ).flatten(1).mean(1)
        image_weight = self.train_video_scheduler.training_weight(timestep_image).to(
            device=image_per_sample.device, dtype=image_per_sample.dtype
        )
        loss_video = (image_per_sample * image_weight).mean()

        action_error = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        )
        dim_is_pad = inputs.get("action_dim_is_pad")
        if dim_is_pad is not None:
            if dim_is_pad.ndim == 1:
                dim_is_pad = dim_is_pad.unsqueeze(0)
            valid_dim = (~dim_is_pad).to(action_error.dtype)[:, None, :]
            action_error = (action_error * valid_dim).sum(2) / valid_dim.sum(2).clamp(min=1)
        else:
            action_error = action_error.mean(2)
        action_is_pad = inputs.get("action_is_pad")
        if action_is_pad is not None:
            valid_step = (~action_is_pad).to(action_error.dtype)
            action_per_sample = (action_error * valid_step).sum(1) / valid_step.sum(1).clamp(min=1)
        else:
            action_per_sample = action_error.mean(1)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            device=action_per_sample.device, dtype=action_per_sample.dtype
        )
        loss_action = (action_per_sample * action_weight).mean()
        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return loss_total, {
            "loss_video": loss_video.detach(),
            "loss_action": loss_action.detach(),
            "weighted_loss_video": loss_video.detach() * self.loss_lambda_video,
            "weighted_loss_action": loss_action.detach() * self.loss_lambda_action,
        }

    @torch.no_grad()
    def _predict_video_noise(
        self,
        latents_video: torch.Tensor,
        timestep_video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        """Run standalone video denoising through the shared staged backbone API."""
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

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
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

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
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
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
        if getattr(self, "backbone_name", None) != "cosmos25":
            return int(num_inference_steps)
        video_timesteps, _ = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=device,
            dtype=dtype,
            shift_override=sigma_shift,
        )
        return len(video_timesteps)

    @torch.inference_mode()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
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
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                num_video_frames=num_video_frames,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                proprio=proprio.clone() if proprio is not None else None,
            )["action"]
        
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.state_dim is None:
                raise ValueError("`proprio` was provided but `state_dim=None` so `state_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.state_dim:
                raise ValueError(f"`proprio` last dim must be {self.state_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
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
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_state_to_context(
                context=context,
                context_mask=context_mask,
                state=proprio,
            )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            # Match Cosmos' N+1 Karras schedule; Wan schedules remain length N.
            num_inference_steps=len(infer_timesteps_video),
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        result = {"action": action_out}
        if decode_video:
            result["video"] = self._decode_latents(latents_video)
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
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        num_video_frames: int = 5,
    ) -> dict[str, Any]:
        self.eval()
        attention_mode = str(
            getattr(self.video_expert, "video_attention_mask_mode", "")
        )
        if attention_mode == "flux2_reference_causal":
            return self._infer_action_flux2(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
            )
        if attention_mode == "bidirectional":
            joint_out = self.infer_joint(
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

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.state_dim is None:
                raise ValueError("`proprio` was provided but `state_dim=None` so `state_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.state_dim:
                raise ValueError(f"`proprio` last dim must be {self.state_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image)
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
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_state_to_context(
                context=context,
                context_mask=context_mask,
                state=proprio,
            )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
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
            video_attention_mask=attention_mask.slice(0, video_seq_len, 0, video_seq_len),
        )

        action_inference_steps = self._action_only_inference_step_count(
            num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            sigma_shift=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=action_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.inference_mode()
    def _infer_action_flux2(
        self,
        *,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        rand_device: str,
    ) -> dict[str, Any]:
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        height, width = int(input_image.shape[-2]), int(input_image.shape[-1])
        if height % 16 or width % 16:
            raise ValueError(
                f"FLUX.2 image spatial dims must be multiples of 16, got HxW=({height},{width})"
            )

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
                raise ValueError("`context` and `context_mask` must be provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got "
                    f"{tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool)

        if self.state_encoder is not None:
            if proprio is None:
                raise ValueError("FLUX.2 action inference requires proprio when state_dim is enabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 2 or proprio.shape != (1, self.state_dim):
                raise ValueError(
                    f"`proprio` must be [1,{self.state_dim}], got {tuple(proprio.shape)}"
                )
            context, context_mask = self._append_state_to_context(
                context, context_mask, proprio.to(device=self.device, dtype=self.torch_dtype)
            )
        elif proprio is not None:
            raise ValueError("`proprio` was provided but state_encoder is disabled.")

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        ref_tokens, ref_img_ids = self._encode_flux2_image_tokens(
            input_image, time_value=10.0
        )
        batch_size = int(ref_tokens.shape[0])
        empty_target = ref_tokens.new_zeros(batch_size, 0, ref_tokens.shape[-1])
        empty_target_ids = ref_img_ids.new_zeros(batch_size, 0, ref_img_ids.shape[-1])

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (batch_size, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )

        video_pre = self.video_expert.pre_dit(
            x=empty_target,
            timestep=torch.zeros(batch_size, dtype=ref_tokens.dtype, device=self.device),
            context=context,
            context_mask=context_mask,
            ref_image_hidden_states=ref_tokens,
            target_img_ids=empty_target_ids,
            ref_img_ids=ref_img_ids,
        )
        prefix_mask = self._build_mot_attention_mask_flux2(
            batch_size=batch_size,
            txt_len=int(video_pre["txt_len"]),
            target_len=0,
            cond_len=int(video_pre["cond_len"]),
            action_len=0,
            device=latents_action.device,
            text_attention_mask=video_pre["text_mask"],
        )
        video_kv_cache = self.mot.prefill_flux2_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            attention_mask=prefix_mask,
        )
        full_mask = self._build_mot_attention_mask_flux2(
            batch_size=batch_size,
            txt_len=int(video_pre["txt_len"]),
            target_len=0,
            cond_len=int(video_pre["cond_len"]),
            action_len=int(latents_action.shape[1]),
            device=latents_action.device,
            text_attention_mask=video_pre["text_mask"],
        )
        prefix_len = int(video_pre["txt_len"]) + int(video_pre["cond_len"])
        for step_t, step_delta in zip(timesteps, deltas):
            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=self._scheduler_timestep_to_unit(
                    step_t.expand(batch_size).to(dtype=latents_action.dtype),
                    self.infer_action_scheduler,
                ),
            )
            action_tokens = self.mot.forward_flux2_action_with_video_cache(
                action_tokens=action_pre["tokens"],
                action_ids=action_pre["ids"],
                action_t_mod=action_pre["t_mod"],
                video_kv_cache=video_kv_cache,
                attention_mask=full_mask,
                video_seq_len=prefix_len,
            )
            pred_action = self.action_expert.post_dit(action_tokens, action_pre)
            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta, latents_action
            )

        return {"action": latents_action[0].detach().to(device="cpu", dtype=torch.float32)}

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
                "backbone_name": getattr(self, "backbone_name", "wan22"),
                "model_variant": self.model_variant,
            }
        if self.state_encoder is not None and not is_lora_model:
            payload["state_encoder"] = self.state_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None, merge_lora: bool = False):
        payload = torch.load(path, map_location="cpu", mmap=True)
        imagewam_conversion = None
        imagewam_load_report = None
        if "proprio_encoder" in payload:
            from .backbone.flux2.checkpoint import (
                audit_mot_state_dict,
                convert_imagewam_flux2_checkpoint_payload,
                require_exact_imagewam_coverage,
            )

            if self.mot.block_protocol != "flux2":
                raise ValueError("ImageWAM `proprio_encoder` migration is only valid for FLUX.2.")
            payload, imagewam_conversion = convert_imagewam_flux2_checkpoint_payload(payload)
            imagewam_load_report = audit_mot_state_dict(self.mot, payload["mot"])
            require_exact_imagewam_coverage(imagewam_load_report)
        checkpoint_backbone = payload.get("backbone_name")
        if checkpoint_backbone is not None and checkpoint_backbone != getattr(self, "backbone_name", "wan22"):
            raise ValueError(
                f"Checkpoint backbone {checkpoint_backbone!r} does not match model backbone "
                f"{getattr(self, 'backbone_name', 'wan22')!r}."
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
                hasattr(module, "_easywam_lora_config")
                for module in self.mot.modules()
            ):
                from .component.lora import load_standard_state_dict

                load_standard_state_dict(
                    self.mot,
                    payload["mot"],
                    strict=imagewam_load_report is not None,
                )
            else:
                self.mot.load_state_dict(
                    payload["mot"],
                    strict=imagewam_load_report is not None,
                )
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
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
                self.state_encoder.load_state_dict(payload["state_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `state_encoder` weights; keeping current `state_encoder` params.")
        elif "state_encoder" in payload:
            logger.warning("Checkpoint contains `state_encoder` weights but current model has `state_dim=None`; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if imagewam_load_report is not None:
            self._last_checkpoint_load_report = {
                **imagewam_load_report,
                "conversion": imagewam_conversion,
                "checkpoint_step": payload.get("step"),
            }
            logger.info(
                "Loaded ImageWAM FLUX.2 checkpoint exactly: tensors=%d coverage=%.8f step=%s",
                imagewam_load_report["matched_tensors"],
                imagewam_load_report["numel_coverage"],
                payload.get("step"),
            )
        return payload

    def forward(self, sample):
        return self.training_loss(sample)
