"""MuJoCo rendering backend selection shared by LIBERO managers."""

from __future__ import annotations

from typing import MutableMapping


def select_mujoco_render_backend(concurrent_envs: int) -> str:
    """Use EGL for one environment and OSMesa for concurrent environments."""
    if concurrent_envs <= 0:
        raise ValueError("concurrent_envs must be positive.")
    return "egl" if concurrent_envs == 1 else "osmesa"


def configure_mujoco_worker_env(
    env: MutableMapping[str, str],
    concurrent_envs: int,
) -> str:
    """Force a consistent MuJoCo and PyOpenGL backend in a worker environment."""
    backend = select_mujoco_render_backend(concurrent_envs)
    env["MUJOCO_GL"] = backend
    env["PYOPENGL_PLATFORM"] = backend
    return backend
