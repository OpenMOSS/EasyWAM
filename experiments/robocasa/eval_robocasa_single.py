"""Single-task and reusable-runtime evaluation for RoboCasa365."""

from __future__ import annotations

import inspect
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.lerobot.processors.wam_processor import WAMProcessor  # noqa: E402
from data.lerobot.prompts import DEFAULT_PROMPT  # noqa: E402
from data.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from experiments.prompt_context_cache import PromptContextCache  # noqa: E402
from experiments.robocasa.env_process import CAMERA_KEYS, RoboCasaEnvProcess  # noqa: E402
from model.helpers.inference import (  # noqa: E402
    configure_inference_compile_from_config,
    configure_model_execution,
)
from utils.pytorch_utils import set_global_seed  # noqa: E402

for name, resolver in (
    ("eval", eval),
    ("max", lambda x: max(x)),
    ("split", lambda s, idx: s.split("/")[int(idx)]),
):
    if not OmegaConf.has_resolver(name):
        OmegaConf.register_new_resolver(name, resolver)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

STATE_KEYS = (
    "state.base_position",
    "state.base_rotation",
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
)


@dataclass
class RoboCasaEvalRuntime:
    model: torch.nn.Module
    processor: WAMProcessor
    action_horizon: int
    input_w: int
    input_h: int
    model_device: str
    prompt_cache: PromptContextCache
    batcher: Any = None


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def extract_robocasa_state(observation: dict[str, Any]) -> np.ndarray:
    """Flatten state in exactly the order used by RoboCasa365 LeRobot data."""
    state = np.concatenate([np.asarray(observation[key]) for key in STATE_KEYS])
    if state.shape != (16,):
        raise ValueError(f"Expected 16-D RoboCasa state, got shape {state.shape}.")
    return state.astype(np.float32, copy=False)


def concatenate_robocasa_images(observation: dict[str, Any]) -> np.ndarray:
    """Concatenate left, right, and wrist RGB cameras in training order."""
    images = [np.asarray(observation[key], dtype=np.uint8) for key in CAMERA_KEYS]
    if any(image.shape != (256, 256, 3) for image in images):
        raise ValueError(
            "RoboCasa evaluation expects three 256x256 RGB observations; got "
            f"{[image.shape for image in images]}."
        )
    return np.concatenate(images, axis=1)


def robocasa_action_dict(action: np.ndarray) -> dict[str, np.ndarray]:
    """Map the RoboCasa365 12-D dataset action to the Gym action dictionary."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (12,):
        raise ValueError(f"Expected a 12-D RoboCasa action, got shape {action.shape}.")
    action = np.clip(action, -1.0, 1.0)
    # Do not use robocasa.utils.env_utils.convert_action: its flat order differs
    # from the official RoboCasa365 LeRobot dataset used for training here.
    return {
        "action.base_motion": action[0:4].copy(),
        "action.control_mode": action[4:5].copy(),
        "action.end_effector_position": action[5:8].copy(),
        "action.end_effector_rotation": action[8:11].copy(),
        "action.gripper_close": action[11:12].copy(),
    }


def _normalize_mixed_precision(value: str) -> str:
    key = str(value).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(f"Unsupported mixed_precision: {value}.")
    return key


def _model_dtype(value: str) -> torch.dtype:
    return {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        _normalize_mixed_precision(value)
    ]


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    candidates: list[Path] = []
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    if explicit:
        candidates.append(Path(os.path.expandvars(os.path.expanduser(str(explicit)))))
    ckpt = Path(os.path.expandvars(os.path.expanduser(str(cfg.ckpt))))
    candidates.extend(parent / "dataset_stats.json" for parent in list(ckpt.parents)[:4])
    for candidate in candidates:
        if candidate.resolve().is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate dataset_stats.json beside the checkpoint. Pass "
        "EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )


def _normalize_state(state: np.ndarray, processor: WAMProcessor) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError("RoboCasa evaluation expects one merged state field.")
    key = state_meta[0]["key"]
    batch = {"state": {key: torch.as_tensor(state).float().unsqueeze(0)}}
    batch = processor.action_state_transform(batch)
    batch = processor.normalizer.forward(batch)
    return batch["state"][key]


def _denormalize_action(action: torch.Tensor, processor: WAMProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action [B,T,D], got {tuple(action.shape)}.")
    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError("RoboCasa evaluation expects one merged action field.")
    key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][key]
    return normalizer.backward(action.float().cpu()).numpy()


def observation_to_model_input(
    observation: dict[str, Any], runtime: RoboCasaEvalRuntime
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    rgb = concatenate_robocasa_images(observation)
    if (rgb.shape[0], rgb.shape[1]) != (runtime.input_h, runtime.input_w):
        raise ValueError(
            f"Concatenated image is {rgb.shape[:2]}, but config expects "
            f"{(runtime.input_h, runtime.input_w)}."
        )
    image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float()
    image = image * (2.0 / 255.0) - 1.0
    proprio = _normalize_state(extract_robocasa_state(observation), runtime.processor)
    return image, proprio, rgb


def _predict_action_chunk(
    observation: dict[str, Any],
    instruction: str,
    runtime: RoboCasaEvalRuntime,
    cfg: DictConfig,
    timing: dict[str, float],
) -> np.ndarray:
    image, proprio, _ = observation_to_model_input(observation, runtime)
    prompt = DEFAULT_PROMPT.format(task=instruction)
    configured_steps = cfg.EVALUATION.get("num_inference_steps")
    kwargs = {
        "input_image": image,
        "action_horizon": runtime.action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": int(
            cfg.eval_num_inference_steps if configured_steps is None else configured_steps
        ),
        "proprio": proprio,
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.sigma_shift)
        ),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
    }
    if getattr(runtime.model, "_eval_supports_num_video_frames", False):
        kwargs["num_video_frames"] = (
            (int(cfg.data.train.num_frames) - 1)
            // int(cfg.data.train.action_video_freq_ratio)
            + 1
        )

    started = time.perf_counter()
    if runtime.batcher is not None:
        prediction = runtime.batcher.submit("action", prompt=prompt, **kwargs)
        timing["prompt_encode_seconds"] += float(
            prediction.get("prompt_encode_seconds", 0.0)
        )
    else:
        context, context_mask = runtime.prompt_cache.get(prompt)
        kwargs.update(
            input_image=image.to(runtime.model_device, dtype=runtime.model.torch_dtype),
            proprio=proprio.to(runtime.model_device, dtype=runtime.model.torch_dtype),
            prompt=None,
            context=context,
            context_mask=context_mask,
        )
        with torch.inference_mode():
            prediction = runtime.model.infer_action(**kwargs)
    timing["inference_seconds"] += time.perf_counter() - started

    action = _denormalize_action(prediction["action"], runtime.processor)[0]
    if action.shape[-1] != 12:
        raise ValueError(f"Model returned action dimension {action.shape[-1]}, expected 12.")
    return action


def _video_mode(cfg: DictConfig) -> str:
    mode = str(cfg.EVALUATION.get("video_mode", "none")).lower()
    if mode not in {"none", "failures", "all"}:
        raise ValueError("EVALUATION.video_mode must be one of: none, failures, all.")
    return mode


def _save_video(
    path: Path, frames: list[np.ndarray], instruction: str, *, fps: int
) -> None:
    if not frames:
        return
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps)
    path.with_suffix(".txt").write_text(instruction + "\n", encoding="utf-8")


def run_single_episode(
    env: RoboCasaEnvProcess,
    runtime: RoboCasaEvalRuntime,
    cfg: DictConfig,
    episode_index: int,
    timing: dict[str, float],
) -> tuple[bool, str, list[np.ndarray]]:
    started = time.perf_counter()
    observation, _ = env.reset()
    timing["simulation_seconds"] += time.perf_counter() - started
    instruction = str(observation["annotation.human.task_description"])
    pending_actions: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    collect_video = _video_mode(cfg) != "none"
    if collect_video:
        frames.append(concatenate_robocasa_images(observation))

    replan_steps = int(cfg.EVALUATION.get("replan_steps", 16))
    if replan_steps <= 0:
        raise ValueError(f"EVALUATION.replan_steps must be positive, got {replan_steps}.")
    progress = tqdm(
        total=env.horizon,
        desc=f"{cfg.EVALUATION.task_name} episode {episode_index + 1}",
        disable=not bool(cfg.EVALUATION.get("progress", False)),
    )
    success = False
    try:
        for _ in range(env.horizon):
            if not pending_actions:
                chunk = _predict_action_chunk(observation, instruction, runtime, cfg, timing)
                pending_actions.extend(chunk[:replan_steps])
                if not pending_actions:
                    raise ValueError("Model returned an empty action chunk.")
            action = pending_actions.pop(0)
            need_observation = not pending_actions
            started = time.perf_counter()
            result = env.step(
                robocasa_action_dict(action),
                return_observation=need_observation,
                return_render=collect_video,
            )
            timing["simulation_seconds"] += time.perf_counter() - started
            progress.update(1)
            if collect_video and result["frame"] is not None:
                frames.append(result["frame"])
            success = bool(result["success"])
            if success or result["terminated"] or result["truncated"]:
                break
            if need_observation:
                observation = result["observation"]
    finally:
        progress.close()
    return success, instruction, frames


def evaluate_task_with_runtime(
    cfg: DictConfig,
    runtime: RoboCasaEvalRuntime,
    *,
    result_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.time()
    task_set = str(cfg.EVALUATION.task_set)
    task_name = str(cfg.EVALUATION.task_name)
    split = str(cfg.EVALUATION.split)
    num_trials = int(cfg.EVALUATION.num_trials)
    results: dict[str, Any] = {
        "task_set": task_set,
        "task_name": task_name,
        "split": split,
        "successes": 0,
        "total_episodes": num_trials,
        "success_episodes": [],
        "failure_episodes": [],
        "task_description": None,
        "gpu_id": cfg.gpu_id,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
    }
    if result_metadata:
        results.update(result_metadata)

    timing = {
        "environment_initialize_seconds": 0.0,
        "prompt_encode_seconds": 0.0,
        "inference_seconds": 0.0,
        "simulation_seconds": 0.0,
    }
    env_started = time.perf_counter()
    env = RoboCasaEnvProcess(
        task_name,
        split,
        None if cfg.get("seed") is None else int(cfg.seed),
        robocasa_root=str(cfg.EVALUATION.get("robocasa_root") or "") or None,
    )
    timing["environment_initialize_seconds"] = time.perf_counter() - env_started
    results["horizon"] = env.horizon
    video_mode = _video_mode(cfg)
    video_dir = Path(str(cfg.EVALUATION.output_dir)) / task_set / task_name / "videos"
    try:
        for episode_index in range(num_trials):
            episode_started = time.perf_counter()
            success, instruction, frames = run_single_episode(
                env, runtime, cfg, episode_index, timing
            )
            if results["task_description"] is None:
                results["task_description"] = instruction
            key = "success_episodes" if success else "failure_episodes"
            results[key].append(episode_index)
            results["successes"] += int(success)
            if video_mode == "all" or (video_mode == "failures" and not success):
                suffix = "success" if success else "failure"
                _save_video(
                    video_dir / f"episode_{episode_index:03d}_{suffix}.mp4",
                    frames,
                    instruction,
                    fps=int(cfg.EVALUATION.get("video_fps", 20)),
                )
            logging.info(
                "[%s/%s] episode %d/%d success=%s cumulative=%d duration=%.2fs",
                task_set,
                task_name,
                episode_index + 1,
                num_trials,
                success,
                results["successes"],
                time.perf_counter() - episode_started,
            )
    finally:
        env.close()

    results["duration"] = time.time() - started
    if bool(cfg.EVALUATION.get("timing_enabled", False)):
        results["timing"] = timing
    return results


def build_eval_runtime(cfg: DictConfig) -> RoboCasaEvalRuntime:
    partial_state = PartialState()
    partial_state.config = cfg
    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _video_mode(cfg)

    model_device = str(
        cfg.EVALUATION.get("device")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = instantiate(
        cfg.model,
        model_dtype=_model_dtype(cfg.get("mixed_precision", "bf16")),
        device=model_device,
    )
    model = configure_model_execution(
        model,
        vae_micro_batch_size=cfg.get("vae_micro_batch_size", 1),
        inference_cross_kv_reuse=cfg.get("inference_cross_kv_reuse", True),
    )
    model.load_checkpoint(str(cfg.ckpt), merge_lora=True)
    model = model.to(model_device).eval()
    model = configure_inference_compile_from_config(model, cfg.EVALUATION)
    model._eval_supports_num_video_frames = (
        "num_video_frames" in inspect.signature(model.infer_action_batch).parameters
    )

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: WAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    configured_horizon = cfg.EVALUATION.get("action_horizon")
    action_horizon = (
        int(cfg.data.train.num_frames) - 1
        if configured_horizon is None
        else int(configured_horizon)
    )
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}.")
    video_size = cfg.data.train.video_size
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H,W], got {video_size}.")
    return RoboCasaEvalRuntime(
        model=model,
        processor=processor,
        action_horizon=action_horizon,
        input_h=int(video_size[0]),
        input_w=int(video_size[1]),
        model_device=model_device,
        prompt_cache=PromptContextCache(model),
    )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, cls=NumpyEncoder)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="benchmark/sim_robocasa.yaml",
)
def main(cfg: DictConfig) -> None:
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    if not cfg.EVALUATION.get("task_name"):
        raise ValueError("EVALUATION.task_name is required for single-task evaluation.")
    runtime = build_eval_runtime(cfg)
    result = evaluate_task_with_runtime(cfg, runtime)
    destination = (
        Path(str(cfg.EVALUATION.output_dir))
        / str(cfg.EVALUATION.task_set)
        / str(cfg.EVALUATION.task_name)
        / "result.json"
    )
    write_json_atomic(destination, result)
    print(
        f"{cfg.EVALUATION.task_name}: "
        f"{result['successes']}/{result['total_episodes']} successes"
    )


if __name__ == "__main__":
    main()
