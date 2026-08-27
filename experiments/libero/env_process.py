"""Spawn-isolated MuJoCo environment used by LIBERO evaluation workers."""

from __future__ import annotations

import multiprocessing as mp
import traceback
from multiprocessing.connection import Connection
from typing import Any


class LiberoEnvProcessError(RuntimeError):
    """Raised when the isolated MuJoCo environment cannot serve a command."""


def _install_render_context_guard() -> None:
    """Rebind the EGL context before every render after long model inference gaps."""
    from robosuite.utils.binding_utils import MjRenderContext

    if getattr(MjRenderContext.render, "_libero_context_guard", False):
        return
    original_render = MjRenderContext.render

    def render_with_current_context(self, *args, **kwargs):
        self.gl_ctx.make_current()
        return original_render(self, *args, **kwargs)

    render_with_current_context._libero_context_guard = True
    MjRenderContext.render = render_with_current_context


def _create_env(env_args: dict[str, Any], seed: int | None):
    _install_render_context_guard()
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env


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
    env_args: dict[str, Any],
    seed: int | None,
) -> None:
    env = None
    try:
        env = _create_env(env_args, seed)
        connection.send(("ok", None))
        while True:
            command, payload = connection.recv()
            try:
                if command == "reset":
                    result = env.reset()
                elif command == "set_init_state":
                    result = env.set_init_state(payload)
                elif command == "step":
                    result = env.step(payload)
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


class LiberoEnvProcess:
    """Small synchronous proxy whose MuJoCo and EGL state lives in a child."""

    def __init__(
        self,
        env_args: dict[str, Any],
        seed: int | None,
        *,
        task_label: str,
    ) -> None:
        context = mp.get_context("spawn")
        parent_connection, child_connection = context.Pipe()
        self._connection = parent_connection
        self._task_label = task_label
        self._closed = False
        self._process = context.Process(
            target=_environment_main,
            args=(child_connection, env_args, seed),
            name=f"libero-env-{task_label}",
            daemon=True,
        )
        self._process.start()
        child_connection.close()
        try:
            self._receive("initialize")
        except BaseException:
            self._closed = True
            self._connection.close()
            self._process.join(timeout=1)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5)
            raise

    def _process_failure(self, command: str) -> LiberoEnvProcessError:
        self._process.join(timeout=1)
        exit_code = self._process.exitcode
        if exit_code is None:
            detail = "closed its IPC channel"
        elif exit_code < 0:
            detail = f"exited from signal {-exit_code} (return code {exit_code})"
        else:
            detail = f"exited with return code {exit_code}"
        return LiberoEnvProcessError(
            f"LIBERO environment for {self._task_label} {detail} while handling {command}."
        )

    def _receive(self, command: str):
        try:
            status, payload = self._connection.recv()
        except (EOFError, BrokenPipeError, OSError) as error:
            raise self._process_failure(command) from error
        if status == "ok":
            return payload
        raise LiberoEnvProcessError(
            f"LIBERO environment for {self._task_label} failed during {payload['command']}: "
            f"{payload['error_type']}: {payload['message']}\n{payload['traceback']}"
        )

    def _request(self, command: str, payload=None):
        if self._closed:
            raise LiberoEnvProcessError(
                f"LIBERO environment for {self._task_label} is already closed."
            )
        try:
            self._connection.send((command, payload))
        except (BrokenPipeError, EOFError, OSError) as error:
            raise self._process_failure(command) from error
        return self._receive(command)

    def reset(self):
        return self._request("reset")

    def set_init_state(self, initial_state):
        return self._request("set_init_state", initial_state)

    def step(self, action):
        return self._request("step", action)

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
