"""Result validation for the published RoboDojo simulation protocol."""

from __future__ import annotations

import json
from pathlib import Path


def expected_episodes(
    task_name: str, override: int | None = None, dimension: str | None = None
) -> int:
    native = 25 if dimension == "generalization" or task_name.endswith("_random") else 50
    if override is not None:
        if override <= 0:
            raise ValueError("eval_num_episodes must be positive.")
        return min(override, native)
    return native


def result_parent(root: Path, task_name: str, env_cfg_type: str, seed: int, tag: str) -> Path:
    return (
        root / "eval_result" / "RoboDojo" / task_name / "easywam_policy"
        / env_cfg_type / f"{seed}_ckpt_name={tag},action_type=joint"
    )


def valid_result(path: Path, expected: int) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        details = payload["details"]
        if not isinstance(details, dict) or len(details) < expected:
            return False
        if int(payload["eval_time"]) < expected:
            return False
        entries = sorted((int(key), item) for key, item in details.items())
        for _, item in entries[:expected]:
            if not isinstance(item, dict) or not isinstance(item.get("success"), bool):
                return False
            score = float(item["score"])
            if not 0.0 <= score <= 1.0:
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def latest_valid_result(parent: Path, expected: int) -> Path | None:
    if not parent.is_dir():
        return None
    candidates = sorted(
        (directory / "_result.json" for directory in parent.iterdir() if (directory / "_result.json").is_file()),
        reverse=True,
    )
    return candidates[0] if candidates and valid_result(candidates[0], expected) else None


def summarize_jobs(jobs: list[dict], root: Path, env_cfg_type: str, tag: str) -> list[dict]:
    rows = []
    for job in jobs:
        task, seed, count = job["task_name"], job["seed"], job["episodes"]
        result = latest_valid_result(result_parent(root, task, env_cfg_type, seed, tag), count)
        if result is None:
            rows.append({**job, "complete": False, "result_path": None})
            continue
        payload = json.loads(result.read_text(encoding="utf-8"))
        details = [item for _, item in sorted(
            (int(key), item) for key, item in payload["details"].items()
        )[:count]]
        rows.append({
            **job,
            "complete": True,
            "success_rate": 100.0 * sum(bool(item["success"]) for item in details) / count,
            "score": 100.0 * sum(float(item["score"]) for item in details) / count,
            "result_path": str(result),
        })
    return rows
