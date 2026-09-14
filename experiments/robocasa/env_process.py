"""Spawn-isolated RoboCasa Gym environment.

RoboCasa and robosuite are imported only in the child process. This keeps normal
EasyWAM imports and unit tests independent of the simulator installation.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import traceback
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np


CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)


class RoboCasaEnvProcessError(RuntimeError):
    """Raised when an isolated RoboCasa environment cannot serve a command."""


def _create_env(
    task_name: str,
    split: str,
    seed: int | None,
    robocasa_root: str | None,
):
    if robocasa_root:
        root = str(Path(robocasa_root).expanduser().resolve())
        if root not in sys.path:
            sys.path.insert(0, root)

    import gymnasium as gym
    import robocasa  # noqa: F401 - import registers the Gym environments
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    env = gym.make(
        f"robocasa/{task_name}",
        split=split,
        seed=seed,
        camera_widths=256,
        camera_heights=256,
        enable_render=True,
    )
    return env, int(get_task_horizon(task_name))


def _render_frame(observation: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(observation[key], dtype=np.uint8) for key in CAMERA_KEYS], axis=1
    )


def _send_error(connection: Connection, command: str, error: BaseException) -> None:
    connection.send(
        (
            "error",
            {
                "command": command,
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
    )


def _environment_main(
    connection: Connection,
    task_name: str,
    split: str,
    seed: int | None,
    robocasa_root: str | None,
) -> None:
    env = None
    try:
        env, horizon = _create_env(task_name, split, seed, robocasa_root)
        connection.send(("ok", {"horizon": horizon}))
        while True:
            command, payload = connection.recv()
            try:
                if command == "reset":
                    observation, info = env.reset()
                    result = (observation, info)
                elif command == "step":
                    action = payload["action"]
                    return_observation = bool(payload.get("return_observation", True))
                    return_render = bool(payload.get("return_render", False))
                    observation, reward, terminated, truncated, info = env.step(action)
                    result = {
                        "observation": observation if return_observation else None,
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "success": bool(info.get("success", False)),
                        "frame": _render_frame(observation) if return_render else None,
                    }
                elif command == "close":
                    env.close()
                    env = None
                    connection.send(("ok", None))
                    break
                else:
                    raise ValueError(f"Unsupported environment command: {command}")
                connection.send(("ok", result))
            except BaseException as error:
                _send_error(connection, command, error)
                break
    except BaseException as error:
        try:
            _send_error(connection, "initialize", error)
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException:
                pass
        connection.close()


class RoboCasaEnvProcess:
    """Synchronous proxy whose MuJoCo state lives in a spawned child process."""

    def __init__(
        self,
        task_name: str,
        split: str,
        seed: int | None,
        *,
        robocasa_root: str | None = None,
    ) -> None:
        context = mp.get_context("spawn")
        parent_connection, child_connection = context.Pipe()
        self._connection = parent_connection
        self._task_label = f"{task_name}:{split}"
        self._closed = False
        self._process = context.Process(
            target=_environment_main,
            args=(child_connection, task_name, split, seed, robocasa_root),
            name=f"robocasa-env-{task_name}",
            daemon=True,
        )
        self._process.start()
        child_connection.close()
        try:
            metadata = self._receive("initialize")
            self.horizon = int(metadata["horizon"])
        except BaseException:
            self._closed = True
            self._connection.close()
            self._process.join(timeout=1)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5)
            raise

    def _process_failure(self, command: str) -> RoboCasaEnvProcessError:
        self._process.join(timeout=1)
        exit_code = self._process.exitcode
        if exit_code is None:
            detail = "closed its IPC channel"
        elif exit_code < 0:
            detail = f"exited from signal {-exit_code} (return code {exit_code})"
        else:
            detail = f"exited with return code {exit_code}"
        return RoboCasaEnvProcessError(
            f"RoboCasa environment for {self._task_label} {detail} while handling {command}."
        )

    def _receive(self, command: str):
        try:
            status, payload = self._connection.recv()
        except (EOFError, BrokenPipeError, OSError) as error:
            raise self._process_failure(command) from error
        if status == "ok":
            return payload
        raise RoboCasaEnvProcessError(
            f"RoboCasa environment for {self._task_label} failed during {payload['command']}: "
            f"{payload['error_type']}: {payload['message']}\n{payload['traceback']}"
        )

    def _request(self, command: str, payload=None):
        if self._closed:
            raise RoboCasaEnvProcessError(
                f"RoboCasa environment for {self._task_label} is already closed."
            )
        try:
            self._connection.send((command, payload))
        except (BrokenPipeError, EOFError, OSError) as error:
            raise self._process_failure(command) from error
        return self._receive(command)

    def reset(self):
        return self._request("reset")

    def step(
        self,
        action: dict[str, np.ndarray],
        *,
        return_observation: bool = True,
        return_render: bool = False,
    ) -> dict[str, Any]:
        return self._request(
            "step",
            {
                "action": action,
                "return_observation": return_observation,
                "return_render": return_render,
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._process.is_alive():
                self._request("close")
        finally:
            self._closed = True
            self._connection.close()
            self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5)
