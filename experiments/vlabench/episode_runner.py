"""Single-episode execution aligned with VLABench's MIT-licensed Evaluator.

Upstream: OpenMOSS/VLABench, VLABench/evaluation/evaluator/base.py.
Keep max_substeps=1, tolerance=.01 and termination semantics identical. This
local boundary adds try/finally cleanup and independent metric error reporting.
"""
from __future__ import annotations

import math
import random

import numpy as np

from experiments.vlabench.contract import matrix_to_euler_xyz, quaternion_wxyz_to_matrix


def euler_to_quaternion(euler):
    x, y, z = np.asarray(euler, dtype=np.float64) / 2
    cx, cy, cz, sx, sy, sz = np.cos(x), np.cos(y), np.cos(z), np.sin(x), np.sin(y), np.sin(z)
    return (cx*cy*cz+sx*sy*sz, sx*cy*cz-cx*sy*sz,
            cx*sy*cz+sx*cy*sz, cx*cy*sz-sx*sy*cz)


def task_horizon(task_name):
    import json
    import os
    from VLABench.configs import name2config

    with open(os.path.join(os.environ["VLABENCH_ROOT"], "configs/task_config.json")) as handle:
        configurations = json.load(handle)
    group = next((key for key, tasks in name2config.items() if task_name in tasks), None)
    return int(configurations.get(group, {}).get("evaluation", {}).get("max_episode_length", 200))


def run_episode(policy, task_name, episode_config, *, seed=42, horizon=None,
                env_factory=None, capture_frame=None):
    if env_factory is None:
        # Import only in the simulator process, after MUJOCO_GL has been set.
        from VLABench.envs import load_env
        import VLABench.tasks  # noqa: F401
        import VLABench.robots  # noqa: F401
        env_factory = load_env
    horizon = task_horizon(task_name) if horizon is None else int(horizon)
    if horizon <= 0:
        raise ValueError("Episode horizon must be positive")
    np.random.seed(seed)
    random.seed(seed)
    policy.reset(seed=seed)
    env = env_factory(task_name, episode_config=episode_config, random_init=False,
                      eval=False, run_mode="eval")
    try:
        env.reset()
        success, last_action, consumed = False, None, 0
        for index in range(horizon):
            observation = env.get_observation(require_pcd=False)
            observation["instruction"] = env.task.get_instruction()
            observation["robot_frame"] = env.get_robot_frame_position()
            if last_action is None:
                ee = observation["ee_state"]
                last_action = np.concatenate((ee[:3], matrix_to_euler_xyz(quaternion_wxyz_to_matrix(ee[3:7]))))
            observation["last_action"] = last_action
            if capture_frame is not None:
                capture_frame(observation["rgb"])
            position, euler, gripper = policy.predict(observation, unnorm_key="primitive", max_episode_length=horizon)
            last_action = np.concatenate((position, euler))
            _, joints = env.robot.get_qpos_from_ee_pos(physics=env.physics, pos=position,
                                                      quat=euler_to_quaternion(euler))
            action = np.concatenate((joints, gripper))
            timestep = env.step(action)  # official max_substeps=1
            success = bool(timestep.last())
            consumed = index if success else index + 1  # upstream consumed_step convention
            if success:
                break
        result = {"success": success, "consumed_step": consumed, "metric_errors": {}}
        for key, method in (("intention_score", lambda: env.get_intention_score(threshold=.1)),
                            ("progress_score", env.get_task_progress)):
            try:
                value = float(method())
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Metric outside [0,1]: {value}")
                result[key] = value
            except Exception as error:
                result[key] = None
                result["metric_errors"][key] = f"{type(error).__name__}: {error}"
        return result
    finally:
        env.close()
