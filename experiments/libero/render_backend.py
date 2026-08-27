"""MuJoCo rendering backend selection shared by LIBERO managers."""

from __future__ import annotations

from typing import MutableMapping


def select_mujoco_render_backend(max_tasks_per_gpu: int) -> str:
    """Use EGL for one worker per GPU and OSMesa for concurrent workers."""
    if max_tasks_per_gpu <= 0:
        raise ValueError("max_tasks_per_gpu must be positive.")
    return "egl" if max_tasks_per_gpu == 1 else "osmesa"


def configure_mujoco_worker_env(
    env: MutableMapping[str, str],
    max_tasks_per_gpu: int,
) -> str:
    """Force a consistent MuJoCo and PyOpenGL backend in a worker environment."""
    backend = select_mujoco_render_backend(max_tasks_per_gpu)
    env["MUJOCO_GL"] = backend
    env["PYOPENGL_PLATFORM"] = backend
    return backend
