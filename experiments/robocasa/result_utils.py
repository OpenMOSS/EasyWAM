"""Result paths and strict resume validation for RoboCasa evaluation."""

from __future__ import annotations

import json
from pathlib import Path


def task_result_path(output_dir: Path, task_set: str, task_name: str) -> Path:
    if not task_set or "/" in task_set or not task_name or "/" in task_name:
        raise ValueError(f"Unsafe RoboCasa task identity: {task_set!r}/{task_name!r}.")
    return output_dir / task_set / task_name / "result.json"


def valid_result_path(
    output_dir: Path,
    task_set: str,
    task_name: str,
    split: str,
    expected_episodes: int | None,
) -> Path | None:
    path = task_result_path(output_dir, task_set, task_name)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        total = int(payload.get("total_episodes"))
        successes = int(payload.get("successes"))
        success_episodes = list(payload.get("success_episodes", []))
        failure_episodes = list(payload.get("failure_episodes", []))
        episode_ids = success_episodes + failure_episodes
        valid = (
            str(payload.get("task_set")) == task_set
            and str(payload.get("task_name")) == task_name
            and str(payload.get("split")) == split
            and total > 0
            and (expected_episodes is None or total == expected_episodes)
            and successes == len(success_episodes)
            and len(episode_ids) == total
            and sorted(int(value) for value in episode_ids) == list(range(total))
            and 0 <= successes <= total
        )
        return path if valid else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
