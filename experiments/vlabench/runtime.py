"""EasyWAM-side model runtime using the existing processor and inference batcher."""
from __future__ import annotations

import threading
from collections import deque
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate

from data.lerobot.prompts import DEFAULT_PROMPT
from data.lerobot.utils.normalizer import load_dataset_stats_from_json
from experiments.batched_inference import DynamicInferenceBatcher
from experiments.vlabench.checkpoint import load_evaluation_checkpoint
from model.helpers.inference import configure_inference_compile_from_config, configure_model_execution
from utils.pytorch_utils import mixed_precision_to_dtype, set_global_seed
from experiments.vlabench.contract import image_tensor


class PolicySession:
    def __init__(self, predict_chunk, replan_steps: int, *, seed=42):
        if replan_steps <= 0:
            raise ValueError("replan_steps must be positive")
        self.predict_chunk = predict_chunk
        self.replan_steps = int(replan_steps)
        self._lock = threading.Lock()
        self.seed = seed
        self._actions = deque()
        self._instruction = None
        self._chunk_index = 0

    def reset(self, seed=None):
        with self._lock:
            self._actions.clear()
            self._instruction = None
            self._chunk_index = 0
            if seed is not None:
                self.seed = int(seed)

    def predict(self, payload):
        with self._lock:
            instruction = payload["instruction"]
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("Expected a nonempty instruction")
            if self._instruction != instruction:
                self._actions.clear()
                self._instruction = instruction
            if not self._actions:
                seed = None if self.seed is None else self.seed + self._chunk_index
                actions = np.asarray(self.predict_chunk(payload, seed), dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] != 10 or not len(actions) or not np.isfinite(actions).all():
                    raise ValueError("Model must return finite [horizon,10] actions")
                self._actions.extend(actions[:self.replan_steps].copy())
                self._chunk_index += 1
            return self._actions.popleft()

    def close(self):
        self.reset()


class VLABenchRuntime:
    def __init__(self, cfg):
        evaluation = cfg.EVALUATION
        checkpoint = Path(str(cfg.ckpt)).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        stats_path = evaluation.dataset_stats_path
        if stats_path is None:
            stats_path = next((p / "dataset_stats.json" for p in checkpoint.parents
                               if (p / "dataset_stats.json").is_file()), None)
        if stats_path is None or not Path(str(stats_path)).is_file():
            raise FileNotFoundError("Set EVALUATION.dataset_stats_path to the training statistics")
        set_global_seed(int(cfg.seed))
        self.model = instantiate(cfg.model, device=str(evaluation.device),
                                 model_dtype=mixed_precision_to_dtype(str(cfg.mixed_precision)))
        load_evaluation_checkpoint(self.model, checkpoint)
        self.model.eval()
        configure_model_execution(self.model, vae_micro_batch_size=cfg.vae_micro_batch_size,
                                  inference_cross_kv_reuse=cfg.inference_cross_kv_reuse)
        configure_inference_compile_from_config(self.model, evaluation)
        self.processor = instantiate(cfg.data.train.processor)
        self.processor.eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))
        self.horizon = int(evaluation.action_horizon or (cfg.data.train.num_frames - 1))
        if self.horizon <= 0 or int(evaluation.replan_steps) <= 0 or int(evaluation.num_inference_steps) <= 0:
            raise ValueError("Action horizon, replan steps and inference steps must be positive")
        self.num_video_frames = 1 + (int(cfg.data.train.num_frames)-1)//int(cfg.data.train.action_video_freq_ratio)
        self.cfg = cfg
        self.batcher = DynamicInferenceBatcher(
            self.model, max_batch_size=int(cfg.MULTIRUN.inference_batch_size),
            wait_ms=float(cfg.MULTIRUN.inference_batch_wait_ms),
            prompt_cache_size=int(cfg.MULTIRUN.prompt_cache_size))

    @torch.inference_mode()
    def predict_chunk(self, payload, seed):
        state = torch.as_tensor(np.asarray(payload["state"], dtype=np.float32)).reshape(1, 1, 10)
        if not torch.isfinite(state).all():
            raise ValueError("Nonfinite proprioception")
        state = self.processor.normalizer.forward({"state": {"default": state}})["state"]["default"]
        ev = self.cfg.EVALUATION
        kwargs = dict(action_horizon=self.horizon, num_inference_steps=int(ev.num_inference_steps),
                      text_cfg_scale=float(ev.text_cfg_scale), negative_prompt=str(ev.negative_prompt),
                      sigma_shift=None if ev.sigma_shift is None else float(ev.sigma_shift),
                      rand_device=str(ev.rand_device), num_video_frames=self.num_video_frames)
        result = self.batcher.submit("action", prompt=DEFAULT_PROMPT.format(task=payload["instruction"]),
                                     input_image=image_tensor(payload["cameras"]),
                                     proprio=state[0, 0], seed=seed, **kwargs)
        action = result["action"].detach().cpu().float().unsqueeze(0)
        # Use the same normalizer as training, without postprocess's window slicing.
        denormalized = self.processor.normalizer.normalizers["action"]["default"].backward(action)
        return denormalized[0].numpy()

    def new_session(self):
        return PolicySession(self.predict_chunk, min(int(self.cfg.EVALUATION.replan_steps), self.horizon),
                             seed=int(self.cfg.seed))

    def close(self):
        self.batcher.close()
