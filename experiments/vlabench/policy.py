"""VLABench-side policy, independent of torch and of the EasyWAM model env."""
from __future__ import annotations

from experiments.vlabench.transport import PolicyClient
from experiments.vlabench.contract import action_to_target, observation_cameras, observation_to_state


class EasyWAMVLABenchPolicy:
    name = "easywam"
    control_mode = "ee"

    def __init__(self, url, *, timeout=300, gripper_threshold=.5, gripper_width=.04):
        self.client = PolicyClient(url, timeout=timeout)
        self.gripper_threshold = gripper_threshold
        self.gripper_width = gripper_width

    def reset(self, seed=None):
        self.client.reset(seed=seed)

    def predict(self, observation, **_kwargs):
        instruction = observation.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("VLABench episode has no instruction")
        action = self.client.predict({
            "cameras": observation_cameras(observation),
            "state": observation_to_state(observation),
            "instruction": instruction,
        })
        return action_to_target(action, observation["robot_frame"],
                                gripper_threshold=self.gripper_threshold,
                                gripper_width=self.gripper_width)

    def close(self):
        self.client.close()
