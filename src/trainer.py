import logging
import json
import inspect
import os
import re
from contextlib import nullcontext
from importlib.metadata import version as package_version
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.fs import ensure_dir
from utils.logging_config import get_logger, setup_logging
from utils.pytorch_utils import set_global_seed
from utils.samplers import ResumableEpochSampler
from utils.video_io import save_mp4
from utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


class DataLoaderWorkerInit:
    def __init__(self, base_worker_init_fn, worker_threads: int):
        self.base_worker_init_fn = base_worker_init_fn
        self.worker_threads = int(worker_threads)

    def __call__(self, worker_id: int):
        if self.base_worker_init_fn is not None:
            self.base_worker_init_fn(worker_id)
        if self.worker_threads > 0:
            torch.set_num_threads(self.worker_threads)


def _count_parameters(module, trainable_only: bool = False) -> int:
    return sum(
        p.numel()
        for p in module.parameters()
        if not trainable_only or p.requires_grad
    )


def _forward_training_model(model, sample):
    output = model(sample)
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError(
            "Training model forward must return a `(loss, metrics)` tuple, "
            f"got {type(output).__name__}."
        )
    loss, metrics = output
    if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
        raise TypeError(
            "Training model forward must return a scalar tensor as loss, "
            f"got {type(loss).__name__} with shape={getattr(loss, 'shape', None)}."
        )
    if not isinstance(metrics, dict):
        raise TypeError(
            "Training model forward must return a dict as metrics, "
            f"got {type(metrics).__name__}."
        )
    return loss, metrics


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig, accelerator: Accelerator):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.warmup_ratio = float(cfg.warmup_ratio)
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(
                f"`warmup_ratio` must be in [0, 1), got {self.warmup_ratio}."
            )
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.dataloader_prefetch_factor = int(cfg.get("dataloader_prefetch_factor", 4))
        self.dataloader_persistent_workers = bool(cfg.get("dataloader_persistent_workers", True))
        self.dataloader_pin_memory = bool(cfg.get("dataloader_pin_memory", torch.cuda.is_available()))
        self.dataloader_worker_threads = int(cfg.get("dataloader_worker_threads", 1))
        max_steps = cfg.max_steps
        if max_steps is None:
            raise ValueError("`max_steps` must be set explicitly; epoch-based training is not used.")
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError(f"`max_steps` must be > 0, got {self.max_steps}.")
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.eval_save_video = bool(cfg.get("eval_save_video", False))
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        self.accelerator = accelerator
        self.is_deepspeed = self.accelerator.distributed_type == DistributedType.DEEPSPEED

        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        deepspeed_config = deepspeed_plugin.deepspeed_config if deepspeed_plugin is not None else {}
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            deepspeed_config.get("zero_optimization", {}).get("stage", "disabled"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        logger.info(
            "DataLoader settings: batch_size=%d num_workers=%d pin_memory=%s persistent_workers=%s prefetch_factor=%d worker_threads=%d",
            self.batch_size,
            self.num_workers,
            self.dataloader_pin_memory,
            self.dataloader_persistent_workers if self.num_workers > 0 else False,
            max(self.dataloader_prefetch_factor, 1) if self.num_workers > 0 else 0,
            self.dataloader_worker_threads,
        )
        logger.info(
            "Training eval settings: eval_every=%d inference_steps=%d save_video=%s",
            self.eval_every,
            self.eval_num_inference_steps,
            self.eval_save_video,
        )
        base_worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        worker_init_fn = self._build_worker_init_fn(base_worker_init_fn)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional state encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        total_params = _count_parameters(self.model)
        trainable_params_count = _count_parameters(self.model, trainable_only=True)
        if self.accelerator.is_main_process:
            logger.info(
                "Model parameters: total=%.6f B trainable=%.6f B frozen=%.6f B trainable_ratio=%.4f%%",
                total_params / 1e9,
                trainable_params_count / 1e9,
                (total_params - trainable_params_count) / 1e9,
                100.0 * trainable_params_count / max(total_params, 1),
            )
        trainable_params = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        if not trainable_params:
            raise RuntimeError("Model has no trainable parameters.")
        optimizer_kwargs = {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
            "betas": (0.9, 0.95),
        }
        if torch.cuda.is_available():
            optimizer_kwargs["fused"] = True
        self.optimizer = torch.optim.AdamW(trainable_params, **optimizer_kwargs)
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self.max_steps
        warmup_steps = int(total_train_steps * self.warmup_ratio)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")
        self._interval_data_time = 0.0
        self._interval_forward_time = 0.0
        self._interval_backward_time = 0.0
        self._interval_steps = 0
        self._interval_loss_sums = None
        self._interval_loss_counts = None

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self._log_prepared_runtime()
        if not self.is_deepspeed:
            self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _log_prepared_runtime(self):
        if not self.is_deepspeed:
            logger.info(
                "Prepared runtime: model=%s optimizer=%s scheduler=%s",
                type(self.model).__name__,
                type(self.optimizer).__name__,
                type(self.scheduler).__name__,
            )
            return

        plugin = self.accelerator.state.deepspeed_plugin
        if plugin is None:
            raise RuntimeError("Accelerator reports DeepSpeed mode without a DeepSpeed plugin.")
        ds_config = plugin.deepspeed_config
        zero_config = ds_config.get("zero_optimization", {})
        gradient_clipping = float(ds_config.get("gradient_clipping", 0.0))
        if gradient_clipping != self.max_grad_norm:
            raise ValueError(
                "DeepSpeed `gradient_clipping` must match trainer `max_grad_norm`: "
                f"{gradient_clipping} != {self.max_grad_norm}."
            )
        engine = getattr(self.accelerator.deepspeed_engine_wrapped, "engine", None)
        if engine is None:
            raise RuntimeError("DeepSpeed engine was not initialized by accelerator.prepare().")
        if engine.lr_scheduler is None:
            raise RuntimeError("DeepSpeed engine did not take ownership of the LR scheduler.")
        engine_optimizer = engine.optimizer
        logger.info(
            "Prepared DeepSpeed runtime: deepspeed=%s engine=%s zero_stage=%s bf16=%s gradient_clipping=%.4f "
            "overlap_comm=%s reduce_bucket_size=%s allgather_bucket_size=%s "
            "optimizer=%s engine_scheduler=%s trainer_scheduler=%s",
            package_version("deepspeed"),
            type(engine).__name__,
            zero_config.get("stage", "unknown"),
            engine.bfloat16_enabled(),
            gradient_clipping,
            getattr(engine_optimizer, "overlap_comm", "unknown"),
            getattr(engine_optimizer, "reduce_bucket_size", "unknown"),
            getattr(engine_optimizer, "allgather_bucket_size", "unknown"),
            type(engine_optimizer).__name__,
            type(engine.lr_scheduler).__name__,
            type(self.scheduler).__name__,
        )

    def _deepspeed_grad_norm(self, loss: torch.Tensor) -> torch.Tensor:
        engine = self.accelerator.deepspeed_engine_wrapped.engine
        grad_norm = engine.get_global_grad_norm()
        if grad_norm is None:
            return torch.tensor(float("nan"), device=loss.device, dtype=torch.float32)
        return torch.as_tensor(grad_norm, device=loss.device, dtype=torch.float32)

    def _rank0_gpu_peak_memory_gb(self) -> dict[str, float] | None:
        if not self.accelerator.is_main_process:
            return None
        device = torch.device(self.accelerator.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            return None
        return {
            "allocated": torch.cuda.max_memory_allocated(device) / (1024**3),
            "reserved": torch.cuda.max_memory_reserved(device) / (1024**3),
        }

    def _reset_rank0_gpu_peak_memory(self) -> None:
        if not self.accelerator.is_main_process:
            return
        device = torch.device(self.accelerator.device)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "sampler": self.train_sampler,
            "num_workers": self.num_workers,
            "pin_memory": self.dataloader_pin_memory,
            "worker_init_fn": worker_init_fn,
        }
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = self.dataloader_persistent_workers
            loader_kwargs["prefetch_factor"] = max(self.dataloader_prefetch_factor, 1)
        return DataLoader(dataset, **loader_kwargs)

    def _build_worker_init_fn(self, base_worker_init_fn):
        return DataLoaderWorkerInit(base_worker_init_fn, self.dataloader_worker_threads)

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        if self.is_deepspeed:
            from deepspeed.runtime.lr_schedules import WarmupCosineLR, WarmupLR

            deepspeed_warmup_steps = max(warmup_steps, 2)
            if scheduler_type == "cosine":
                return WarmupCosineLR(
                    self.optimizer,
                    total_num_steps=total_train_steps,
                    warmup_min_ratio=0.0,
                    warmup_num_steps=deepspeed_warmup_steps,
                    warmup_type="linear",
                    cos_min_ratio=0.01,
                )
            if scheduler_type == "constant":
                return WarmupLR(
                    self.optimizer,
                    warmup_min_lr=0.0,
                    warmup_max_lr=self.learning_rate,
                    warmup_num_steps=deepspeed_warmup_steps,
                    warmup_type="linear",
                )
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        from model.wan22.lora import has_lora, set_video_dit_lora_trainable

        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        video_dit = getattr(model, "video_dit", None)
        if video_dit is not None and has_lora(video_dit):
            set_video_dit_lora_trainable(video_dit)
        state_encoder = getattr(model, "state_encoder", None)
        if state_encoder is not None:
            state_encoder.train()
            state_encoder.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        video_path = self._save_eval_video(
            pred_video_tensor,
            vae_video_tensor,
            gt_video_tensor,
        )

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_eval_video(
        self,
        pred_video_tensor: torch.Tensor,
        vae_video_tensor: torch.Tensor,
        gt_video_tensor: torch.Tensor,
    ) -> str | None:
        if not self.eval_save_video:
            return None

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (
                stitched_video_tensor[:, t]
                .permute(1, 2, 0)
                .clamp(0.0, 1.0)
                .numpy()
                * 255.0
            ).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)
        return video_path

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        temporary_path = f"{ckpt_path}.tmp"
        started_at = time.perf_counter()
        logger.info("[ckpt] saving weights: %s", ckpt_path)
        model.save_checkpoint(
            temporary_path,
            optimizer=None,
            step=self.global_step,
        )
        os.replace(temporary_path, ckpt_path)
        size_gib = os.path.getsize(ckpt_path) / (1024**3)
        logger.info(
            "[ckpt] saved weights: path=%s size=%.3fGiB elapsed=%.2fs",
            ckpt_path,
            size_gib,
            time.perf_counter() - started_at,
        )
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        state_started_at = time.perf_counter()
        if self.accelerator.is_main_process:
            logger.info("[ckpt] saving distributed state: %s", state_path)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
            logger.info(
                "[ckpt] saved distributed state: path=%s elapsed=%.2fs",
                state_path,
                time.perf_counter() - state_started_at,
            )
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def _accumulate_train_interval(
        self,
        loss: torch.Tensor,
        loss_dict: dict,
        data_time: float,
        forward_time: float,
        backward_time: float,
    ) -> None:
        metric_values = []
        metric_counts = []
        metric_keys = (
            "loss",
            "loss_action",
            "loss_video",
            "weighted_loss_action",
            "weighted_loss_video",
        )
        for key in metric_keys:
            value = loss if key == "loss" else loss_dict.get(key)
            if value is None:
                metric_values.append(torch.zeros((), device=loss.device, dtype=torch.float32))
                metric_counts.append(torch.zeros((), device=loss.device, dtype=torch.float32))
                continue
            value = torch.as_tensor(value, device=loss.device).detach()
            if value.ndim != 0:
                raise ValueError(f"Training metric `{key}` must be scalar, got shape={tuple(value.shape)}.")
            metric_values.append(value.to(dtype=torch.float32))
            metric_counts.append(torch.ones((), device=loss.device, dtype=torch.float32))

        values = torch.stack(metric_values)
        counts = torch.stack(metric_counts)
        if self._interval_loss_sums is None:
            self._interval_loss_sums = torch.zeros_like(values)
            self._interval_loss_counts = torch.zeros_like(counts)
        self._interval_loss_sums.add_(values)
        self._interval_loss_counts.add_(counts)
        self._interval_data_time += float(data_time)
        self._interval_forward_time += float(forward_time)
        self._interval_backward_time += float(backward_time)
        self._interval_steps += 1

    def _reduce_train_interval(self) -> tuple[dict[str, float], dict[str, float]]:
        if self._interval_loss_sums is None or self._interval_loss_counts is None:
            raise RuntimeError("Cannot reduce an empty training log interval.")

        timing = torch.tensor(
            [
                self._interval_data_time,
                self._interval_forward_time,
                self._interval_backward_time,
                float(self._interval_steps),
            ],
            device=self._interval_loss_sums.device,
            dtype=torch.float32,
        )
        local_stats = torch.cat(
            [self._interval_loss_sums, self._interval_loss_counts, timing]
        ).reshape(1, -1)
        gathered = self.accelerator.gather(local_stats).reshape(-1, local_stats.shape[1])
        global_stats = gathered.sum(dim=0)

        num_loss_metrics = 5
        loss_sums = global_stats[:num_loss_metrics]
        loss_counts = global_stats[num_loss_metrics:2 * num_loss_metrics]
        names = (
            "loss_avg",
            "loss_action_avg",
            "loss_video_avg",
            "weighted_loss_action_avg",
            "weighted_loss_video_avg",
        )
        loss_averages = {
            name: float((value_sum / count.clamp_min(1.0)).item())
            for name, value_sum, count in zip(names, loss_sums, loss_counts)
        }
        timing_offset = 2 * num_loss_metrics
        interval_steps = global_stats[timing_offset + 3].clamp_min(1.0)
        timing_names = (
            "data_time",
            "forward_time",
            "backward_time",
        )
        timing_averages = {
            name: float(
                (global_stats[timing_offset + index] / interval_steps).item()
            )
            for index, name in enumerate(timing_names)
        }

        self._interval_loss_sums.zero_()
        self._interval_loss_counts.zero_()
        self._interval_data_time = 0.0
        self._interval_forward_time = 0.0
        self._interval_backward_time = 0.0
        self._interval_steps = 0
        return loss_averages, timing_averages

    def train(self):
        self._set_dit_only_train_mode()

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        progress_bar = tqdm(
            total=self.max_steps,
            initial=self.global_step,
            disable=not self.accelerator.is_local_main_process,
            desc="Training",
            dynamic_ncols=True,
        )

        while self.global_step < self.max_steps:
            if self._interval_steps == 0:
                self._reset_rank0_gpu_peak_memory()
            try:
                data_start = time.perf_counter()
                sample = next(data_iter)
                data_time = time.perf_counter() - data_start
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                forward_context = nullcontext() if self.is_deepspeed else self.accelerator.autocast()
                forward_start = time.perf_counter()
                with forward_context:
                    loss, loss_dict = _forward_training_model(self.model, sample)
                forward_time = time.perf_counter() - forward_start

                backward_start = time.perf_counter()
                self.accelerator.backward(loss)
                backward_time = time.perf_counter() - backward_start

                if self.accelerator.sync_gradients:
                    if self.is_deepspeed:
                        grad_norm = self._deepspeed_grad_norm(loss)
                    else:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        if not self.accelerator.optimizer_step_was_skipped:
                            self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)

            self._accumulate_train_interval(
                loss,
                loss_dict,
                data_time,
                forward_time,
                backward_time,
            )

            if self.accelerator.sync_gradients:
                self.global_step += 1
                current_lr = float(self.optimizer.param_groups[0]["lr"])
                if self.accelerator.is_local_main_process:
                    progress_bar.set_postfix(
                        {
                            "data_time": f"{data_time:.4f}",
                            "forward_time": f"{forward_time:.4f}",
                            "backward_time": f"{backward_time:.4f}",
                            "loss": f"{loss.detach().float().item():.4f}",
                            "grad_norm": f"{torch.as_tensor(grad_norm).detach().float().item():.4f}",
                        },
                        refresh=False,
                    )
                progress_bar.update(1)
                should_log = self.log_every > 0 and (
                    self.global_step % self.log_every == 0 or self.global_step == self.max_steps
                )

                if should_log:
                    loss_averages, timing_averages = self._reduce_train_interval()
                    grad_norm_tensor = torch.as_tensor(
                        grad_norm, device=loss.device, dtype=torch.float32
                    ).reshape(1)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())
                if should_log and self.accelerator.is_main_process:
                    description = (
                        "[train] epoch=%d step=%d/%d loss_avg=%.6f "
                        "loss_action_avg=%.6f loss_video_avg=%.6f "
                        "weighted_loss_action_avg=%.6f weighted_loss_video_avg=%.6f lr=%.4e "
                        "data_time=%.4fs forward_time=%.4fs backward_time=%.4fs"
                    ) % (
                        self.epoch,
                        self.global_step,
                        self.max_steps,
                        loss_averages["loss_avg"],
                        loss_averages["loss_action_avg"],
                        loss_averages["loss_video_avg"],
                        loss_averages["weighted_loss_action_avg"],
                        loss_averages["weighted_loss_video_avg"],
                        current_lr,
                        timing_averages["data_time"],
                        timing_averages["forward_time"],
                        timing_averages["backward_time"],
                    )
                    gpu_memory = self._rank0_gpu_peak_memory_gb()
                    if gpu_memory is not None:
                        description += (
                            " gpu_peak_allocated=%.3fGB gpu_peak_reserved=%.3fGB"
                            % (gpu_memory["allocated"], gpu_memory["reserved"])
                        )
                    logger.info(description)

                    wandb_payload = {
                        "train/loss_avg": loss_averages["loss_avg"],
                        "train/loss_action_avg": loss_averages["loss_action_avg"],
                        "train/loss_video_avg": loss_averages["loss_video_avg"],
                        "train/weighted_loss_action_avg": loss_averages["weighted_loss_action_avg"],
                        "train/weighted_loss_video_avg": loss_averages["weighted_loss_video_avg"],
                        "train/grad_norm": global_grad_norm,
                        "train/lr": current_lr,
                        "performance/data_time": timing_averages["data_time"],
                        "performance/forward_time": timing_averages["forward_time"],
                        "performance/backward_time": timing_averages["backward_time"],
                    }
                    if gpu_memory is not None:
                        wandb_payload.update(
                            {
                                "gpu/peak_allocated_gb": gpu_memory["allocated"],
                                "gpu/peak_reserved_gb": gpu_memory["reserved"],
                            }
                        )
                    self._wandb_log(wandb_payload)

                if (
                    self.eval_every > 0
                    and self.val_dataset is not None
                    and self.global_step % self.eval_every == 0
                ):
                    metrics = self.evaluate()
                    self.accelerator.wait_for_everyone()
                    if metrics is not None and self.accelerator.is_main_process:
                        description = "[eval] step=%d val_loss=%.6f infer_psnr=%.4f infer_ssim=%.4f" % (
                            self.global_step,
                            metrics["val_loss"],
                            metrics["psnr_rd"],
                            metrics["ssim_rd"],
                        )
                        if "action_l2" in metrics:
                            description += " action_l2=%.4f" % metrics["action_l2"]
                        if "action_l1" in metrics:
                            description += " action_l1=%.4f" % metrics["action_l1"]
                        logger.info(description)
                        eval_payload = {
                            "eval/val_loss": float(metrics["val_loss"]),
                            "eval/psnr_rg": float(metrics["psnr_rg"]),
                            "eval/ssim_rg": float(metrics["ssim_rg"]),
                            "eval/psnr_rd": float(metrics["psnr_rd"]),
                            "eval/ssim_rd": float(metrics["ssim_rd"]),
                            "eval/psnr_dg": float(metrics["psnr_dg"]),
                            "eval/ssim_dg": float(metrics["ssim_dg"]),
                        }
                        if "action_l2" in metrics:
                            eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                        if "action_l1" in metrics:
                            eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                        self._wandb_log(eval_payload)

                if self.save_every > 0 and self.global_step % self.save_every == 0:
                    ckpt_info = self.save_checkpoint()
                    if self.accelerator.is_main_process:
                        logger.info(
                            "[ckpt] step=%d weights=%s state=%s",
                            self.global_step,
                            ckpt_info["weights_path"],
                            ckpt_info["state_path"],
                        )

                if self.global_step >= self.max_steps:
                    progress_bar.close()
                    ckpt_info = self.save_checkpoint()
                    if self.accelerator.is_main_process:
                        logger.info(
                            "[done] max_steps reached step=%d weights=%s state=%s",
                            self.global_step,
                            ckpt_info["weights_path"],
                            ckpt_info["state_path"],
                        )
                    return

        progress_bar.close()
        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        
