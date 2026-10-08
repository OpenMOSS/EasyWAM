import os
from pathlib import Path
from typing import Optional
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from .text_embedding_cache import (
    DEFAULT_TEXT_ENCODER_ID,
    load_text_embedding_cache,
    prompt_hash,
    text_embedding_cache_filename,
)
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from .prompts import DEFAULT_PROMPT
from .resources import reuse_during_construction, shared_resource
from utils.logging_config import get_logger
from utils import misc, pytorch_utils
from accelerate import PartialState
logger = get_logger(__name__)


class RobotVideoDataset(torch.utils.data.Dataset):
    @reuse_during_construction
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        text_encoder_id=DEFAULT_TEXT_ENCODER_ID,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal",
        override_instruction: Optional[str] = None,
        mixed_precision: Optional[str] = None,
    ):
        if num_frames <= 1:
            raise ValueError(f"`num_frames` must be greater than 1, got {num_frames}.")
        if action_video_freq_ratio <= 0:
            raise ValueError(
                "`action_video_freq_ratio` must be greater than 0, "
                f"got {action_video_freq_ratio}."
            )
        if (num_frames - 1) % action_video_freq_ratio != 0:
            raise ValueError(
                "num_frames-1 must be divisible by action_video_freq_ratio, "
                f"got {num_frames - 1} and {action_video_freq_ratio}"
            )
        if ((num_frames - 1) // action_video_freq_ratio) % 4 != 0:
            raise ValueError(
                "video transitions must be divisible by 4 for tokenization, "
                f"got {(num_frames - 1) // action_video_freq_ratio}"
            )
        self.video_sample_indices = tuple(range(0, num_frames, action_video_freq_ratio))

        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            image_obs_indices=self.video_sample_indices,
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = (
            None
            if text_embedding_cache_dir is None
            else Path(text_embedding_cache_dir).expanduser()
        )
        self.text_encoder_id = str(text_encoder_id)
        self.text_embedding_dtype = (
            None if mixed_precision is None else pytorch_utils.mixed_precision_to_dtype(mixed_precision)
        )
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    stats_output = os.path.join(work_dir, "dataset_stats.json")
                    shared_resource(("saved_stats", stats_output, str(pretrained_norm_stats)), lambda: save_dataset_stats_to_json(dataset_stats, stats_output))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    stats_output = os.path.join(work_dir, "dataset_stats.json")
                    shared_resource(("saved_stats", stats_output, str(pretrained_norm_stats)), lambda: save_dataset_stats_to_json(dataset_stats, stats_output))

            processor.include_gt_action = False
            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)
        
    def __len__(self):
        return len(self.lerobot_dataset)

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset.__getitem__(sample_idx)

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = any(bool(pad.any().item()) for pad in (action_is_pad, image_is_pad, proprio_is_pad))

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))
        
        image_is_pad = sample["image_is_pad"]

        images = sample["pixel_values"]
        cameras = self.lerobot_dataset.processor.camera_views(images)
        expected_video_frames = len(self.video_sample_indices)
        if any(camera.shape[0] != expected_video_frames for camera in cameras):
            raise ValueError(f"Sparse video frame count mismatch: expected {expected_video_frames}.")
        if image_is_pad.shape[0] != expected_video_frames:
            raise ValueError(f"Sparse image padding length mismatch: expected {expected_video_frames}, got {image_is_pad.shape[0]}.")
        if self.concat_multi_camera == "robotwin":
            if len(cameras) != 3:
                raise ValueError(f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {len(cameras)}")
            sizes = [(256, 320), (128, 160), (128, 160)]
            cameras = [
                camera if tuple(camera.shape[-2:]) == size else transforms_F.resize(
                    camera, list(size),
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )
                for camera, size in zip(cameras, sizes)
            ]
            video = cameras[0].new_empty((expected_video_frames, 3, 384, 320))
            video[:, :, :256] = cameras[0]
            video[:, :, 256:, :160] = cameras[1]
            video[:, :, 256:, 160:] = cameras[2]
        elif len(cameras) > 1:
            if self.concat_multi_camera not in {"horizontal", "vertical"}:
                raise ValueError(f"Invalid concat_multi_camera: {self.concat_multi_camera}")
            video = torch.cat(cameras, dim=-1 if self.concat_multi_camera == "horizontal" else -2)
        else:
            video = cameras[0]

        if tuple(video.shape[-2:]) != tuple(self.video_size):
            video = self.resize_transform(video)
            video = self.crop_transform(video)
        video = self.normalize_transform(video)

        video = video.permute(1, 0, 2, 3)

        action = sample["action"]
        proprio = sample["proprio"][:-1, :]
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = sample["instruction"]
        
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        
        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        if "action_dim_is_pad" in sample:
            data["action_dim_is_pad"] = sample["action_dim_is_pad"]
        if "proprio_dim_is_pad" in sample:
            data["proprio_dim_is_pad"] = sample["proprio_dim_is_pad"]
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("`text_embedding_cache_dir` is not set.")
        hashed = prompt_hash(prompt)
        dtype = self.text_embedding_dtype
        if self.text_encoder_id == "qwen3_flux2":
            cache_path = self.text_embedding_cache_dir / text_embedding_cache_filename(
                hashed, self.context_len, self.text_encoder_id, is_hash=True, dtype=dtype
            )
            if not cache_path.is_file():
                raise FileNotFoundError(
                    f"Missing FLUX.2 Qwen3 text cache: {cache_path}. "
                    "Run scripts/precompute_text_embeds.py for the selected task."
                )
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            context = payload["text_hidden_states"]
            context_mask = payload["text_attention_mask"].bool()
            if context.ndim != 2 or context_mask.ndim != 1:
                raise ValueError(
                    "FLUX.2 Qwen3 cache must contain [L,D] hidden states and a [L] mask."
                )
            if context.shape[0] != self.context_len or context_mask.shape[0] != self.context_len:
                raise ValueError(
                    f"FLUX.2 Qwen3 cache length must be {self.context_len}, got "
                    f"{context.shape[0]}/{context_mask.shape[0]}."
                )
            if context.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
                raise TypeError(f"Unsupported text embedding dtype: {context.dtype}.")
            if dtype is not None and context.dtype != dtype:
                raise TypeError(f"Text embedding dtype mismatch: expected {dtype}, got {context.dtype}.")
            return context, context_mask
        cache_path = self.text_embedding_cache_dir / text_embedding_cache_filename(
            hashed,
            self.context_len,
            self.text_encoder_id,
            is_hash=True,
            dtype=dtype,
        )
        payload = load_text_embedding_cache(
            cache_path,
            self.context_len,
            self.text_encoder_id,
            hashed,
            expected_dtype=dtype,
        )
        context = payload["context"]
        context_mask = payload["mask"].bool()
        context[~context_mask] = 0
        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
