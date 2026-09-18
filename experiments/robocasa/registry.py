"""Small, testable adapter around RoboCasa's official task registry."""

from __future__ import annotations

import sys
import importlib.metadata
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from packaging.version import InvalidVersion, Version


OFFICIAL_TASK_SET_SIZES = {
    "atomic_seen": 18,
    "composite_seen": 16,
    "composite_unseen": 16,
}


def validate_task_sets(
    registry: Mapping[str, Sequence[str]], requested: Iterable[str]
) -> list[tuple[str, str]]:
    jobs: list[tuple[str, str]] = []
    seen_tasks: set[str] = set()
    for task_set in requested:
        if task_set not in registry:
            raise KeyError(
                f"RoboCasa task set {task_set!r} is unavailable; "
                f"available sets: {sorted(registry)}"
            )
        tasks = [str(task) for task in registry[task_set]]
        expected = OFFICIAL_TASK_SET_SIZES.get(task_set)
        if expected is not None and len(tasks) != expected:
            raise RuntimeError(
                f"RoboCasa {task_set} contains {len(tasks)} tasks; expected {expected}. "
                "Update the official RoboCasa checkout before evaluating."
            )
        for task in tasks:
            if task in seen_tasks:
                raise RuntimeError(f"Task {task!r} appears in multiple requested task sets.")
            seen_tasks.add(task)
            jobs.append((task_set, task))
    return jobs


def load_official_jobs(
    requested: Iterable[str], robocasa_root: str | Path | None
) -> tuple[list[tuple[str, str]], object]:
    try:
        robosuite_version = importlib.metadata.version("robosuite")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "RoboCasa evaluation requires robosuite from its current master branch. "
            "Install robosuite before loading the task registry."
        ) from error
    try:
        if Version(robosuite_version) < Version("1.5.0"):
            raise RuntimeError(
                f"Installed robosuite {robosuite_version} is too old for RoboCasa365 "
                "(it lacks PandaOmron). Install the robosuite master branch "
                "as described in the RoboCasa evaluation guide."
            )
    except InvalidVersion as error:
        raise RuntimeError(f"Could not parse robosuite version {robosuite_version!r}.") from error
    if robocasa_root:
        root = str(Path(robocasa_root).expanduser().resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
    try:
        from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
        from robocasa.utils.dataset_registry_utils import get_task_horizon
    except Exception as error:
        raise RuntimeError(
            "Could not import the official RoboCasa task registry: "
            f"{type(error).__name__}: {error}. Check the RoboCasa and robosuite "
            "installation, or set EVALUATION.robocasa_root to its checkout."
        ) from error
    try:
        installed_version = importlib.metadata.version("robocasa")
        if Version(installed_version) < Version("1.0.1"):
            raise RuntimeError(
                f"RoboCasa {installed_version} is too old; v1.0.1 or newer is "
                "required for the official 1.5x task horizons."
            )
    except importlib.metadata.PackageNotFoundError:
        # A source checkout is validated through its registry and task horizons.
        pass
    except InvalidVersion as error:
        raise RuntimeError("Could not parse the installed RoboCasa version.") from error
    jobs = validate_task_sets(TASK_SET_REGISTRY, requested)
    for _, task_name in jobs:
        horizon = int(get_task_horizon(task_name))
        if horizon <= 0:
            raise RuntimeError(f"Invalid horizon {horizon} for RoboCasa task {task_name}.")
    return jobs, get_task_horizon
