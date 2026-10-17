"""Read official episode inventories without importing the simulator."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from experiments.vlabench.contract import FIXED_TRACKS

UPSTREAM_REVISION = "cf588fe60c0c7282174fe979f5913170cfe69017"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_digest(path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def validate_root(root) -> Path:
    root = Path(root).expanduser().resolve()
    if not (root / "VLABench/configs/evaluation/tracks").is_dir():
        raise FileNotFoundError(f"Not a VLABench checkout: {root}")
    return root


def upstream_identity(root) -> dict:
    root = validate_root(root)
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    # Hash execution sources too: local edits must invalidate resumed results.
    sources = {}
    for path in sorted((root / "VLABench").rglob("*")):
        if path.is_file() and path.suffix in {".py", ".json"}:
            sources[str(path.relative_to(root))] = file_digest(path)
    return {"commit": result.stdout.strip() if result.returncode == 0 else None,
            "source_digest": digest(sources)}


def build_jobs(root, tracks, tasks=None, *, num_trials=10, seed=42, custom_track_file=None):
    root = validate_root(root)
    if type(num_trials) is not int or num_trials <= 0:
        raise ValueError("num_trials must be a positive integer")
    tracks = list(tracks)
    if not tracks or len(tracks) != len(set(tracks)):
        raise ValueError("tracks must be nonempty and distinct")
    selected = None if tasks is None else set(tasks)
    if selected is not None and (not selected or len(selected) != len(tasks)):
        raise ValueError("tasks must be nonempty and distinct when supplied")
    inventories = {}
    for track in tracks:
        if track == "track_5_cross_task":
            if not custom_track_file:
                raise ValueError("Track 5 requires EVALUATION.custom_track_file")
            path = Path(custom_track_file).expanduser().resolve()
        elif track in FIXED_TRACKS:
            path = root / "VLABench/configs/evaluation/tracks" / f"{track}.json"
        else:
            raise ValueError(f"Unsupported VLABench track: {track}")
        inventory = json.loads(path.read_text())
        if not isinstance(inventory, dict) or not inventory:
            raise ValueError(f"Empty or invalid track inventory: {path}")
        inventories[track] = inventory
    available = {task for inventory in inventories.values() for task in inventory}
    if selected is not None and selected - available:
        raise ValueError(f"Tasks absent from selected tracks: {sorted(selected-available)}")
    jobs = []
    for track, inventory in inventories.items():
        for task, episodes in inventory.items():
            if selected is not None and task not in selected:
                continue
            if not isinstance(task, str) or not task or "/" in task or "\\" in task or task in {".", ".."}:
                raise ValueError(f"Invalid task identity: {task!r}")
            if not isinstance(episodes, list):
                raise ValueError(f"Invalid episode list: {track}/{task}")
            # Same precondition as the official Evaluator: a uniform requested
            # episode count must be available for every selected task.
            if len(episodes) < num_trials:
                raise ValueError(f"{track}/{task}: requested {num_trials} episodes, but only "
                                 f"{len(episodes)} official configurations are available. "
                                 f"Set EVALUATION.num_trials<={len(episodes)} or explicitly "
                                 "select tasks with sufficient configurations.")
            for index in range(num_trials):
                config = episodes[index]
                if not isinstance(config, dict):
                    raise ValueError(f"Invalid episode config: {track}/{task}/{index}")
                jobs.append({"track": track, "task_name": task, "episode_id": index,
                             "seed": int(seed) + index, "episode_config": config,
                             "episode_digest": digest(config)})
    if not jobs:
        raise ValueError("No selected VLABench jobs")
    return jobs
