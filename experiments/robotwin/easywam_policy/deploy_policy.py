import logging
import os
import sys
import time
import inspect
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.lerobot.processors.wam_processor import WAMProcessor
from data.lerobot.robot_video_dataset import DEFAULT_PROMPT
from data.lerobot.utils.normalizer import load_dataset_stats_from_json
from experiments.robotwin.batched_inference import DynamicInferenceBatcher
from model.helpers.inference import configure_inference_compile, configure_model_execution

logger = logging.getLogger(__name__)


@dataclass
class _PolicySession:
    pending_actions: deque[np.ndarray] = field(default_factory=deque)
    observation: Optional[Dict[str, Any]] = None
    batch_observations: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    episode_count: int = 0
    step_count: int = 0
    timing: Dict[str, float] = field(
        default_factory=lambda: {"prompt_encode_s": 0.0, "infer_s": 0.0, "sim_s": 0.0}
    )


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "benchmark/sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _observation_instruction(observation: Dict[str, Any]) -> str:
    value = observation.get("instruction", observation.get("instructions", ""))
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value)


def pack_xpolicylab_joint_state(observation: Dict[str, Any]) -> np.ndarray:
    """Pack XPolicyLab dual-arm joint state in the joint-only dataset order."""
    state = observation.get("state")
    if not isinstance(state, dict):
        raise KeyError("XPolicyLab observation is missing the 'state' mapping.")
    keys = (
        "left_arm_joint_state",
        "left_ee_joint_state",
        "right_arm_joint_state",
        "right_ee_joint_state",
    )
    missing = [key for key in keys if key not in state]
    if missing:
        raise KeyError(f"XPolicyLab observation is missing state keys: {missing}")
    vector = np.concatenate(
        [np.asarray(state[key], dtype=np.float32).reshape(-1) for key in keys]
    )
    if vector.shape != (14,):
        raise ValueError(f"Expected a 14-D joint state, got shape {vector.shape}.")
    return vector


def action_vector_to_xpolicylab(action: np.ndarray) -> Dict[str, Any]:
    """Convert EasyWAM's 14-D action to XPolicyLab joint fields."""
    vector = np.asarray(action, dtype=np.float32).reshape(-1)
    if vector.shape != (14,):
        raise ValueError(f"Expected a 14-D RoboTwin action, got shape {vector.shape}.")
    return {
        "left_arm_joint_state": vector[:6],
        "left_ee_joint_state": vector[6:7],
        "right_arm_joint_state": vector[7:13],
        "right_ee_joint_state": vector[13:14],
        "action_type": "joint",
    }


class WorldActionRobotWinPolicy:
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        timing_enabled: bool,
        torch_compile: bool,
        torch_compile_mode: Optional[str],
        torch_compile_backend: Optional[str],
        torch_compile_fullgraph: bool,
        torch_compile_dynamic: bool,
        torch_compile_options: Dict[str, Any],
        num_video_frames: int,
        vae_micro_batch_size: int | None = 1,
        inference_cross_kv_reuse: bool = True,
        inference_batch_size: int = 4,
        inference_batch_wait_ms: float = 10,
        prompt_cache_size: int = 40,
    ) -> None:
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = True

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model = configure_model_execution(
            self.model,
            vae_micro_batch_size=vae_micro_batch_size,
            inference_cross_kv_reuse=inference_cross_kv_reuse,
        )
        self.model.load_checkpoint(checkpoint_path, merge_lora=True)
        self.model = self.model.to(device).eval()
        self.model = configure_inference_compile(
            self.model,
            enabled=torch_compile,
            mode=torch_compile_mode,
            backend=torch_compile_backend,
            fullgraph=torch_compile_fullgraph,
            dynamic=torch_compile_dynamic,
            options=torch_compile_options,
        )

        self.processor: WAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)
        self.batcher = DynamicInferenceBatcher(
            self.model,
            max_batch_size=inference_batch_size,
            wait_ms=inference_batch_wait_ms,
            prompt_cache_size=prompt_cache_size,
        )
        self._batch_executor = ThreadPoolExecutor(max_workers=max(1, inference_batch_size))
        self._supports_num_video_frames = (
            "num_video_frames" in inspect.signature(self.model.infer_action_batch).parameters
        )

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.timing_enabled = bool(timing_enabled)
        self._num_video_frames = int(num_video_frames)

        self._sessions: Dict[str, _PolicySession] = {"local": _PolicySession()}
        self._sessions_lock = threading.Lock()

        logger.info(
            "Initialized WorldActionRobotWinPolicy | ckpt=%s | stats=%s | horizon=%d | replan=%d",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {
            "state": {
                state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            }
        }
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        vision = observation.get("vision")
        if not isinstance(vision, dict):
            raise KeyError("XPolicyLab observation is missing the 'vision' mapping.")

        def camera_rgb(name: str) -> np.ndarray:
            camera = vision.get(name)
            if not isinstance(camera, dict) or "color" not in camera:
                raise KeyError(f"XPolicyLab observation is missing vision.{name}.color.")
            return np.asarray(camera["color"], dtype=np.uint8)

        head = _resize_rgb(camera_rgb("cam_head"), (320, 256))
        left = _resize_rgb(camera_rgb("cam_left_wrist"), (160, 128))
        right = _resize_rgb(camera_rgb("cam_right_wrist"), (160, 128))
        bottom = np.concatenate([left, right], axis=1)
        image = np.concatenate([head, bottom], axis=0)  # [384, 320, 3]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def _infer_action_chunk(
        self, observation: Dict[str, Any], instruction: str, session: _PolicySession
    ) -> np.ndarray:
        image_tensor = self._build_robotwin_image_tensor(observation)
        state_vector = pack_xpolicylab_joint_state(observation)
        proprio = self._normalize_state(state_vector)

        prompt = DEFAULT_PROMPT.format(task=instruction)
        infer_kwargs = {
            "action_horizon": self.action_horizon,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "rand_device": self.rand_device,
        }
        if self._supports_num_video_frames:
            infer_kwargs["num_video_frames"] = int(self._num_video_frames)
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        pred = self.batcher.submit(
            "action", prompt=prompt, input_image=image_tensor,
            proprio=proprio, seed=self.seed, **infer_kwargs,
        )
        if self.timing_enabled:
            prompt_seconds = float(pred.get("prompt_encode_seconds", 0.0))
            session.timing["prompt_encode_s"] += prompt_seconds
            session.timing["infer_s"] += max(0.0, time.perf_counter() - infer_t0 - prompt_seconds)

        action_tensor = pred["action"]  # [T, D]
        action_chunk = self._denormalize_action(action_tensor)[0]  # [T, D]
        return action_chunk

    def _fill_action_queue(
        self, observation: Dict[str, Any], instruction: str, session: _PolicySession
    ) -> None:
        action_chunk = self._infer_action_chunk(observation, instruction, session)
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        for i in range(n_exec):
            session.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))

    def should_request_observation(self) -> bool:
        return not self._sessions["local"].pending_actions

    def remote_step(self, payload: Dict[str, Any]) -> np.ndarray:
        return self._remote_step(payload, self._sessions["local"])

    def _remote_step(self, payload: Dict[str, Any], session: _PolicySession) -> np.ndarray:
        observation = payload.get("observation")
        instruction = str(payload.get("instruction", ""))
        if not session.pending_actions:
            if observation is None:
                raise ValueError("Observation is required at a remote replan step.")
            self._fill_action_queue(observation, instruction, session)
        if not session.pending_actions:
            raise RuntimeError("The policy generated no actions.")
        session.step_count += 1
        return session.pending_actions.popleft()

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        session = self._sessions["local"]
        if not session.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(EasyWAM replan step)."
                )
            instruction = task_env.get_instruction()
            self._fill_action_queue(observation, instruction, session)

        if not session.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = session.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            session.timing["sim_s"] += time.perf_counter() - sim_t0
        session.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._sessions["local"].timing = {"prompt_encode_s": 0.0, "infer_s": 0.0, "sim_s": 0.0}

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            key: float(value) for key, value in self._sessions["local"].timing.items()
        }

    def reset(self) -> None:
        session = self._sessions["local"]
        session.pending_actions.clear()
        session.observation = None
        session.batch_observations.clear()
        session.episode_count += 1
        session.step_count = 0
        self.reset_timing_rollout()

    def invoke(self, session_id: str, command: str, payload: Any = None) -> Any:
        with self._sessions_lock:
            session = self._sessions.setdefault(session_id, _PolicySession())
        if command == "update_obs":
            if not isinstance(payload, dict):
                raise TypeError("update_obs expects one XPolicyLab observation mapping.")
            session.observation = payload
            return None
        if command == "update_obs_batch":
            if not isinstance(payload, list):
                raise TypeError("update_obs_batch expects a list of observations.")
            observations = {}
            for observation in payload:
                if not isinstance(observation, dict) or "env_idx" not in observation:
                    raise TypeError("Each batch observation must contain env_idx.")
                index = int(observation["env_idx"])
                if index in observations:
                    raise ValueError(f"Duplicate env_idx in batch: {index}")
                observations[index] = observation
            session.batch_observations = observations
            return None
        if command == "get_action":
            if session.observation is None:
                raise ValueError("get_action requires update_obs to be called first.")
            instruction = _observation_instruction(session.observation)
            action_chunk = self._infer_action_chunk(
                session.observation, instruction, session
            )
            n_exec = min(self.replan_steps, action_chunk.shape[0])
            return [
                action_vector_to_xpolicylab(action_chunk[index])
                for index in range(n_exec)
            ]
        if command == "get_action_batch":
            if not isinstance(payload, list):
                raise TypeError("get_action_batch expects a list of environment indices.")
            indices = [int(index) for index in payload]
            if len(set(indices)) != len(indices):
                raise ValueError("get_action_batch contains duplicate environment indices.")
            missing = [index for index in indices if index not in session.batch_observations]
            if missing:
                raise ValueError(f"Missing observations for environment indices: {missing}")

            def infer(index: int) -> list[Dict[str, Any]]:
                observation = session.batch_observations[index]
                instruction = _observation_instruction(observation)
                chunk = self._infer_action_chunk(observation, instruction, session)
                return [
                    action_vector_to_xpolicylab(chunk[step])
                    for step in range(min(self.replan_steps, chunk.shape[0]))
                ]

            return list(self._batch_executor.map(infer, indices))
        if command == "reset":
            session.pending_actions.clear()
            session.observation = None
            session.batch_observations.clear()
            session.episode_count += 1
            session.step_count = 0
            session.timing = {"prompt_encode_s": 0.0, "infer_s": 0.0, "sim_s": 0.0}
            return None
        if command == "trial_end":
            session.pending_actions.clear()
            session.observation = None
            session.batch_observations.clear()
            return None
        if command == "get_timing_rollout":
            return {key: float(value) for key, value in session.timing.items()}
        raise AttributeError(f"No policy command named {command!r}")

    def close_session(self, session_id: str) -> None:
        with self._sessions_lock:
            self._sessions.pop(session_id, None)

    def close(self) -> None:
        self._batch_executor.shutdown(wait=True)
        self.batcher.close()


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")

    mixed_precision = str(
        usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16")
    )
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = (
            eval_horizon
            if eval_horizon is not None
            else int(cfg.data.train.num_frames) - 1
        )
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(
            cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps)
        )

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(
        usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0))
    )
    negative_prompt = str(
        usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", ""))
    )
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu")))
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )
    torch_compile = _parse_bool(
        usr_args.get("torch_compile", cfg.EVALUATION.get("torch_compile", False))
    )
    torch_compile_mode = str(
        usr_args.get(
            "torch_compile_mode",
            cfg.EVALUATION.get("torch_compile_mode", "reduce-overhead"),
        )
    )
    torch_compile_backend = str(
        usr_args.get(
            "torch_compile_backend",
            cfg.EVALUATION.get("torch_compile_backend", "inductor"),
        )
    )
    torch_compile_fullgraph = _parse_bool(
        usr_args.get(
            "torch_compile_fullgraph",
            cfg.EVALUATION.get("torch_compile_fullgraph", False),
        )
    )
    torch_compile_dynamic = _parse_bool(
        usr_args.get(
            "torch_compile_dynamic",
            cfg.EVALUATION.get("torch_compile_dynamic", False),
        )
    )
    torch_compile_options = dict(cfg.EVALUATION.get("torch_compile_options", {}))

    policy = WorldActionRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        timing_enabled=timing_enabled,
        torch_compile=torch_compile,
        torch_compile_mode=torch_compile_mode,
        torch_compile_backend=torch_compile_backend,
        torch_compile_fullgraph=torch_compile_fullgraph,
        torch_compile_dynamic=torch_compile_dynamic,
        torch_compile_options=torch_compile_options,
        num_video_frames=(int(cfg.data.train.num_frames) - 1)
        // int(cfg.data.train.action_video_freq_ratio)
        + 1,
        vae_micro_batch_size=cfg.get("vae_micro_batch_size", 1),
        inference_cross_kv_reuse=cfg.get("inference_cross_kv_reuse", True),
        inference_batch_size=int(
            usr_args.get("inference_batch_size", cfg.MULTIRUN.inference_batch_size)
        ),
        inference_batch_wait_ms=float(
            usr_args.get(
                "inference_batch_wait_ms", cfg.MULTIRUN.inference_batch_wait_ms
            )
        ),
        prompt_cache_size=int(
            usr_args.get("prompt_cache_size", cfg.MULTIRUN.prompt_cache_size)
        ),
    )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
