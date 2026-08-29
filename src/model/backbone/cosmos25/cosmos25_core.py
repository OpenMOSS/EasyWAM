"""Standalone Cosmos-Predict2.5 Image2World core."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...schedulers.scheduler_flow_unipc import FlowUniPCScheduler
from .cosmos_video_dit import Cosmos25VideoDiT
from .loader import load_cosmos25_components


class Cosmos25Core(torch.nn.Module):
    def __init__(
        self,
        dit: Cosmos25VideoDiT,
        vae,
        text_encoder=None,
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
        train_shift: float = 5.0,
        infer_shift: float = 5.0,
        num_train_timesteps: int = 1000,
    ):
        super().__init__()
        self.dit = dit
        self.vae = vae
        self.text_encoder = text_encoder
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.train_scheduler = FlowUniPCScheduler(
            num_train_timesteps, train_shift, use_karras_sigmas=False
        )
        self.infer_scheduler = FlowUniPCScheduler(
            num_train_timesteps, infer_shift, use_karras_sigmas=True
        )
        self.to(device=self.device)

    @classmethod
    def from_cosmos25_pretrained(
        cls,
        model_id: str | Path = "./checkpoints/Cosmos-Predict2.5-2B",
        reason_model_id: str | Path = "./checkpoints/Cosmos-Reason1-7B",
        device: str | torch.device = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        load_text_encoder: bool = True,
        tokenizer_max_len: int = 128,
        attention_backend: str = "sdpa",
        use_gradient_checkpointing: bool = False,
        train_shift: float = 5.0,
        infer_shift: float = 5.0,
        num_train_timesteps: int = 1000,
    ) -> "Cosmos25Core":
        components = load_cosmos25_components(
            model_id=model_id,
            reason_model_id=reason_model_id,
            device=device,
            torch_dtype=torch_dtype,
            load_text_encoder=load_text_encoder,
            tokenizer_max_len=tokenizer_max_len,
            attention_backend=attention_backend,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        core = cls(
            dit=components.dit,
            vae=components.vae,
            text_encoder=components.text_encoder,
            device=device,
            torch_dtype=torch_dtype,
            train_shift=train_shift,
            infer_shift=infer_shift,
            num_train_timesteps=num_train_timesteps,
        )
        core.model_paths = {
            "dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
        }
        return core

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        if args:
            if isinstance(args[0], torch.dtype):
                dtype = args[0]
            else:
                device = args[0]
            if len(args) > 1 and isinstance(args[1], torch.dtype):
                dtype = args[1]
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.torch_dtype = dtype
        return result

    @staticmethod
    def validate_video_shape(num_frames: int, height: int, width: int) -> None:
        if num_frames % 4 != 1:
            raise ValueError(f"num_frames must satisfy T % 4 == 1, got {num_frames}.")
        if height % 16 or width % 16:
            raise ValueError(f"height and width must be divisible by 16, got {(height, width)}.")

    def encode_prompt(self, prompt: str | Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_encoder is None:
            raise RuntimeError(
                "The text encoder was not loaded. Pass cached `context` and `context_mask`, "
                "or construct with load_text_encoder=True."
            )
        context, mask = self.text_encoder(prompt)
        return context.to(self.device), mask.to(self.device, dtype=torch.bool)

    def _resolve_context(
        self,
        prompt=None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (prompt is None) == (context is None):
            raise ValueError("Provide exactly one of `prompt` or cached `context`.")
        if context is None:
            return self.encode_prompt(prompt)
        if context.ndim != 3 or context.shape[-1] != self.dit.config.reason_context_dim:
            raise ValueError(
                f"context must be [B,L,{self.dit.config.reason_context_dim}], got {tuple(context.shape)}"
            )
        if context_mask is None:
            context_mask = torch.ones(context.shape[:2], dtype=torch.bool, device=context.device)
        if context_mask.shape != context.shape[:2]:
            raise ValueError("context_mask shape must equal context.shape[:2].")
        return context.to(self.device), context_mask.to(self.device, dtype=torch.bool)

    @staticmethod
    def _image_tensor(image: torch.Tensor | Image.Image) -> torch.Tensor:
        if isinstance(image, Image.Image):
            array = np.asarray(image.convert("RGB"), dtype=np.float32)
            image = torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1.0
        if not isinstance(image, torch.Tensor):
            raise TypeError("input_image must be a torch.Tensor or PIL.Image.")
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError(f"input_image must be [3,H,W] or [1,3,H,W], got {tuple(image.shape)}")
        return image

    def build_inputs(self, sample: dict) -> dict[str, torch.Tensor]:
        video = sample.get("video")
        if not isinstance(video, torch.Tensor) or video.ndim != 5 or video.shape[1] != 3:
            raise ValueError("sample['video'] must be a [B,3,T,H,W] tensor.")
        self.validate_video_shape(video.shape[2], video.shape[3], video.shape[4])
        context, context_mask = self._resolve_context(
            prompt=sample.get("prompt"), context=sample.get("context"), context_mask=sample.get("context_mask")
        )
        if context.shape[0] != video.shape[0]:
            raise ValueError("Text/video batch sizes do not match.")
        latents = self.vae.encode(video.to(self.device, dtype=self.torch_dtype))
        return {"input_latents": latents, "context": context, "context_mask": context_mask}

    def _model_fn(self, latents, timestep, context, context_mask=None, condition_mask=None):
        return self.dit(
            latents,
            timestep,
            context,
            context_mask=context_mask,
            condition_mask=condition_mask,
        )

    def training_loss(self, sample: dict):
        inputs = self.build_inputs(sample)
        clean = inputs["input_latents"]
        batch, _, latent_t, _, _ = clean.shape
        noise = torch.randn_like(clean)
        future_timestep = self.train_scheduler.sample_training_t(batch, self.device, clean.dtype)
        noisy = self.train_scheduler.add_noise(clean, noise, future_timestep)
        noisy[:, :, :1] = clean[:, :, :1]
        timestep = future_timestep[:, None].expand(batch, latent_t).clone()
        timestep[:, 0] = 0.1
        condition_mask = torch.zeros((batch, 1, latent_t, clean.shape[3], clean.shape[4]), device=clean.device, dtype=clean.dtype)
        condition_mask[:, :, 0] = 1
        prediction = self._model_fn(
            noisy, timestep, inputs["context"], inputs["context_mask"], condition_mask
        )
        target = noise - clean
        loss = F.mse_loss(prediction[:, :, 1:].float(), target[:, :, 1:].float())
        return loss, {"loss_video": loss.detach()}

    @torch.no_grad()
    def infer(
        self,
        input_image: torch.Tensor | Image.Image,
        num_frames: int,
        prompt: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        negative_context: Optional[torch.Tensor] = None,
        negative_context_mask: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.0,
        num_inference_steps: int = 35,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, list[Image.Image]]:
        self.eval()
        image = self._image_tensor(input_image)
        height, width = image.shape[-2:]
        self.validate_video_shape(num_frames, height, width)
        positive, positive_mask = self._resolve_context(prompt, context, context_mask)
        if guidance_scale != 1.0:
            if negative_context is not None:
                negative, negative_mask = self._resolve_context(None, negative_context, negative_context_mask)
            else:
                negative, negative_mask = self._resolve_context("" if negative_prompt is None else negative_prompt)
        else:
            negative = negative_mask = None
        condition = self.vae.encode(image.to(self.device, dtype=self.torch_dtype).unsqueeze(2))
        latent_t = (num_frames - 1) // 4 + 1
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents = torch.randn(
            (1, 16, latent_t, height // 8, width // 8),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(self.device, dtype=self.torch_dtype)
        latents[:, :, :1] = condition
        condition_mask = torch.zeros((1, 1, latent_t, height // 8, width // 8), device=self.device, dtype=latents.dtype)
        condition_mask[:, :, 0] = 1
        timesteps = self.infer_scheduler.set_timesteps(num_inference_steps, self.device, sigma_shift)
        for scalar_t in timesteps:
            model_t = scalar_t.to(latents.dtype).expand(1, latent_t).clone()
            model_t[:, 0] = 0.1
            positive_velocity = self._model_fn(latents, model_t, positive, positive_mask, condition_mask)
            velocity = positive_velocity
            if negative is not None:
                negative_velocity = self._model_fn(latents, model_t, negative, negative_mask, condition_mask)
                velocity = negative_velocity + guidance_scale * (positive_velocity - negative_velocity)
            velocity[:, :, :1] = 0
            latents = self.infer_scheduler.step(velocity, scalar_t, latents)
            latents[:, :, :1] = condition
        decoded = self.vae.decode(latents)[0].detach().float().clamp(-1, 1)
        decoded = ((decoded + 1) * 127.5).round().to(torch.uint8).cpu()
        frames = [Image.fromarray(decoded[:, index].permute(1, 2, 0).numpy()) for index in range(decoded.shape[1])]
        return {"video": frames}

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {"dit": self.dit.state_dict(), "step": step}
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.dit.load_state_dict(payload["dit"], strict=True)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, sample: dict):
        return self.training_loss(sample)
