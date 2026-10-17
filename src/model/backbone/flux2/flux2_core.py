from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ...helpers.endpoint import decode_latents, encode_video
from ...helpers.gradient import gradient_checkpoint_forward
from ...schedulers.factory import build_video_schedulers
from ..loader import normalize_backbone_config
from .flux2_video_dit import Flux2VideoDiT
from .loader import load_components


class Flux2Core(nn.Module):
    """Standalone conditional endpoint training and inference for the visual DiT."""

    def __init__(
        self,
        dit: Flux2VideoDiT,
        vae: nn.Module,
        text_encoder: nn.Module | None = None,
        tokenizer=None,
        *,
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        train_shift: float = 5.0,
        infer_shift: float = 5.0,
        num_train_timesteps: int = 1000,
        video_scheduler_config: dict | None = None,
    ):
        super().__init__()
        self.dit = dit
        self.vae = vae.eval().requires_grad_(False)
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if self.text_encoder is not None:
            self.text_encoder.eval().requires_grad_(False)
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.text_dim = int(dit.transformer.txt_in.in_features)
        self.train_scheduler, self.infer_scheduler = build_video_schedulers(
            family=dit.scheduler_family,
            config=video_scheduler_config,
            num_train_timesteps=num_train_timesteps,
            train_shift=train_shift,
            infer_shift=infer_shift,
        )
        self.to(device=self.device, dtype=torch_dtype)

    @property
    def video_dit(self):
        return self.dit

    @classmethod
    def from_flux2_pretrained(
        cls,
        backbone: Mapping,
        *,
        device: str | torch.device = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "Flux2Core":
        cfg = normalize_backbone_config(backbone)
        if cfg["name"] != "flux2":
            raise ValueError("Flux2Core requires a flux2 backbone configuration.")
        components = load_components(cfg, device=device, torch_dtype=torch_dtype)
        scheduler = dict(cfg.get("video_scheduler", {}))
        model = cls(
            dit=components.dit,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            device=device,
            torch_dtype=torch_dtype,
            train_shift=float(scheduler.get("train_shift", 5.0)),
            infer_shift=float(scheduler.get("infer_shift", 5.0)),
            num_train_timesteps=int(scheduler.get("num_train_timesteps", 1000)),
            video_scheduler_config=scheduler,
        )
        model.model_paths = {
            "dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
        }
        return model

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        return self

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        parameter = next(self.dit.parameters())
        self.device = parameter.device
        self.torch_dtype = parameter.dtype
        return result

    @torch.no_grad()
    def encode_prompt(self, prompt: str | Sequence[str]):
        if self.text_encoder is None:
            raise RuntimeError(
                "The text encoder was not loaded; supply cached context and context_mask."
            )
        context, mask = self.text_encoder(prompt)
        return context.to(self.device, self.torch_dtype), mask.to(
            self.device, torch.bool
        )

    def _resolve_context(self, prompt=None, context=None, context_mask=None):
        if context is None:
            if prompt is None:
                raise ValueError("Provide prompt or cached context and context_mask.")
            context, context_mask = self.encode_prompt(prompt)
        if (
            not isinstance(context, torch.Tensor)
            or context.ndim != 3
            or context.shape[-1] != self.text_dim
        ):
            raise ValueError(f"context must be [B,L,{self.text_dim}].")
        if context_mask is None:
            context_mask = torch.ones(
                context.shape[:2], device=context.device, dtype=torch.bool
            )
        if context_mask.shape != context.shape[:2]:
            raise ValueError(
                "context_mask must match context's batch and sequence dimensions."
            )
        return context.to(self.device, self.torch_dtype), context_mask.to(
            self.device, torch.bool
        )

    @staticmethod
    def validate_video_shape(num_frames: int, height: int, width: int):
        if num_frames < 2:
            raise ValueError("Endpoint training needs at least two video frames.")
        if height % 16 or width % 16:
            raise ValueError("Image height and width must be divisible by 16.")

    def build_inputs(self, sample: dict):
        video = sample.get("video")
        if (
            not isinstance(video, torch.Tensor)
            or video.ndim != 5
            or video.shape[1] != 3
        ):
            raise ValueError("sample['video'] must be [B,3,T,H,W].")
        self.validate_video_shape(video.shape[2], *video.shape[-2:])
        context, mask = self._resolve_context(
            sample.get("prompt"), sample.get("context"), sample.get("context_mask")
        )
        if context.shape[0] != video.shape[0]:
            raise ValueError("Text and video batch sizes must match.")
        with torch.no_grad():
            latents = encode_video(self, video)
        valid = torch.ones(video.shape[0], device=self.device, dtype=torch.bool)
        padding = sample.get("image_is_pad")
        if padding is not None:
            if padding.shape != (video.shape[0], video.shape[2]):
                raise ValueError("image_is_pad must be [B,T] matching video.")
            valid = ~padding[:, -1].to(self.device, torch.bool)
        return dict(
            input_latents=latents,
            context=context,
            context_mask=mask,
            target_valid=valid,
        )

    def _model_fn(
        self,
        target,
        reference,
        timestep,
        context,
        context_mask,
        grid,
        target_valid=None,
    ):
        batch = target.shape[0]
        target_ids = self.dit.build_img_ids(
            batch, *grid, time_value=0, device=target.device, dtype=target.dtype
        )
        reference_ids = self.dit.build_img_ids(
            batch, *grid, time_value=10, device=target.device, dtype=target.dtype
        )
        pre = self.dit.pre_dit(
            x=target,
            timestep=timestep / float(self.train_scheduler.num_train_timesteps),
            context=context,
            context_mask=context_mask,
            ref_image_hidden_states=reference,
            target_img_ids=target_ids,
            ref_img_ids=reference_ids,
        )
        prefix = pre["txt_len"] + pre["cond_len"]
        total = prefix + pre["target_len"]
        attention_mask = torch.ones(
            batch, total, total, device=target.device, dtype=torch.bool
        )
        attention_mask[:, :prefix, prefix:] = False
        attention_mask[:, :, : pre["txt_len"]] &= pre["text_mask"][:, None, :]
        if target_valid is not None:
            attention_mask[:, :, prefix:] &= target_valid[:, None, None]
        tokens = pre["tokens"]
        for index in range(len(self.dit.blocks)):
            tokens = gradient_checkpoint_forward(
                lambda values, i=index: self.dit.forward_block(
                    i, values, pre, attention_mask
                ),
                self.dit.use_gradient_checkpointing and torch.is_grad_enabled(),
                tokens,
            )
        return self.dit.post_dit(tokens, pre)

    def training_loss(self, sample: dict):
        inputs = self.build_inputs(sample)
        latents = inputs["input_latents"]
        reference = latents[:, :, 0].flatten(2).transpose(1, 2)
        target = latents[:, :, 1].flatten(2).transpose(1, 2)
        timestep = self.train_scheduler.sample_training_t(
            target.shape[0], self.device, target.dtype
        )
        noise = torch.randn_like(target)
        noisy = self.train_scheduler.add_noise(target, noise, timestep)
        prediction = self._model_fn(
            noisy,
            reference,
            timestep,
            inputs["context"],
            inputs["context_mask"],
            latents.shape[-2:],
            inputs["target_valid"],
        )
        error = (
            (prediction.float() - (noise - target).float()).square().flatten(1).mean(1)
        )
        loss = (
            error
            * inputs["target_valid"]
            * self.train_scheduler.training_weight(timestep)
        ).mean()
        return loss, {"loss_video": loss.detach()}

    @staticmethod
    def _image_tensor(image):
        if isinstance(image, Image.Image):
            array = np.array(image.convert("RGB"), dtype=np.float32)
            image = torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1.0
        if not isinstance(image, torch.Tensor):
            raise TypeError("input_image must be a tensor or PIL.Image.")
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[:2] != (1, 3):
            raise ValueError("input_image must be [3,H,W] or [1,3,H,W].")
        return image

    @torch.inference_mode()
    def infer(
        self,
        input_image,
        num_frames: int = 2,
        prompt: str | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        negative_prompt: str | None = None,
        negative_context: torch.Tensor | None = None,
        negative_context_mask: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: float | None = None,
        seed: int | None = None,
        rand_device: str = "cpu",
    ):
        if num_frames != 2:
            raise ValueError(
                "FLUX endpoint inference produces exactly two frames; use num_frames=2."
            )
        if guidance_scale != 1.0 or any(
            value is not None
            for value in (negative_prompt, negative_context, negative_context_mask)
        ):
            raise ValueError(
                "Endpoint inference requires guidance_scale=1 and no negative conditions."
            )
        self.eval()
        image = self._image_tensor(input_image)
        self.validate_video_shape(num_frames, *image.shape[-2:])
        context, mask = self._resolve_context(prompt, context, context_mask)
        if context.shape[0] != 1:
            raise ValueError("infer accepts one image and one text context.")
        latent = encode_video(self, image.unsqueeze(2))
        reference = latent[:, :, 0].flatten(2).transpose(1, 2)
        generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        target = torch.randn(
            reference.shape,
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(self.device, self.torch_dtype)
        timesteps, deltas = self.infer_scheduler.build_inference_schedule(
            num_inference_steps,
            self.device,
            target.dtype,
            shift_override=sigma_shift,
        )
        for timestep, delta in zip(timesteps, deltas):
            prediction = self._model_fn(
                target, reference, timestep.expand(1), context, mask, latent.shape[-2:]
            )
            target = self.infer_scheduler.step(
                prediction, delta, target, stream_id="video"
            )
        future = target.transpose(1, 2).reshape_as(latent[:, :, 0])
        output = torch.stack([latent[:, :, 0], future], dim=2)
        return {"video": decode_latents(self, output)[0]}

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "dit": self.dit.state_dict(),
            "step": step,
            "backbone_name": "flux2",
            "variant": self.dit.variant,
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, Path(path))

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if (
            payload.get("backbone_name") != "flux2"
            or payload.get("variant") != self.dit.variant
        ):
            raise ValueError("Checkpoint backbone or variant does not match Flux2Core.")
        self.dit.load_state_dict(payload["dit"], strict=True)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, sample: dict):
        return self.training_loss(sample)
