"""Validation helpers for resumable RoboTwin evaluation results."""

from __future__ import annotations

from pathlib import Path


def parse_success_rate(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            pass
    if not values:
        raise ValueError(f"No success rate found in {path}")
    value = values[-1]
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Success rate in {path} is outside [0, 1]: {value}")
    return value


def valid_phase_result(output_dir: Path, task: str, phase: str) -> bool:
    try:
        parse_success_rate(output_dir / task / f"_result_{phase}.txt")
        return True
    except (FileNotFoundError, OSError, ValueError):
        return False


def task_is_complete(output_dir: Path, task: str) -> bool:
    return all(valid_phase_result(output_dir, task, phase) for phase in ("clean", "random"))
