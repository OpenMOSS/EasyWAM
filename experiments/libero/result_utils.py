"""Result validation shared by standard LIBERO manager and workers."""

from __future__ import annotations

import json
from pathlib import Path


def valid_result_path(
    output_dir: Path,
    suite_name: str,
    task_id: int,
    expected_episodes: int | None,
) -> Path | None:
    suite_dir = output_dir / suite_name
    if not suite_dir.is_dir():
        return None
    for path in sorted(suite_dir.glob(f"gpu*_task{task_id}_results.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                str(payload.get("task_suite")) == suite_name
                and int(payload.get("task_id")) == task_id
                and int(payload.get("total_episodes")) > 0
                and (
                    expected_episodes is None
                    or int(payload.get("total_episodes")) == expected_episodes
                )
                and 0
                <= int(payload.get("successes"))
                <= int(payload.get("total_episodes"))
            ):
                return path
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return None
