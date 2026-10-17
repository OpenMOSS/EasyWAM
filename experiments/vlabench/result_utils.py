"""Atomic per-episode records and coverage-aware VLABench summaries."""
from __future__ import annotations

import csv
import json
import math
import os
import uuid
from collections import defaultdict
from pathlib import Path

METRICS = ("success_rate", "progress_score", "intention_score")


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def result_path(output, job):
    track, task = job["track"], job["task_name"]
    if any(not isinstance(x, str) or not x or "/" in x or "\\" in x or x in {".", ".."} for x in (track, task)):
        raise ValueError("Unsafe result identity")
    return Path(output) / track / task / f"episode_{int(job['episode_id']):04d}.json"


def read_result(output, job, fingerprint):
    try:
        value = json.loads(result_path(output, job).read_text())
        if (value["fingerprint"] != fingerprint or
                any(value[key] != job[key] for key in ("track", "task_name", "episode_id", "episode_digest", "seed"))):
            return None
        if value.get("status") in {"error", "missing_config"}:
            return value
        if value.get("status") != "completed" or type(value.get("success")) is not bool:
            return None
        for metric in ("progress_score", "intention_score"):
            score = value.get(metric)
            if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))
                                      or not math.isfinite(score) or not 0 <= score <= 1):
                return None
        return value
    except (OSError, ValueError, TypeError, KeyError):
        return None


def is_complete(output, job, fingerprint):
    result = read_result(output, job, fingerprint)
    # Metric failures remain resumable, rather than silently passing coverage.
    return bool(result and result["status"] == "completed" and
                all(result.get(key) is not None for key in METRICS[1:]))


def _aggregate(records):
    completed = [row for row in records if row is not None and row.get("status") == "completed"]
    summary = {"expected_episodes": len(records), "completed_episodes": len(completed),
               "error_episodes": sum(row is not None and row.get("status") == "error" for row in records),
               "missing_episodes": sum(row is None or row.get("status") == "missing_config" for row in records),
               "missing_official_configs": sum(row is not None and row.get("status") == "missing_config" for row in records)}
    coverage = {}
    for metric in METRICS:
        values = [float(row["success"]) if metric == "success_rate" else row.get(metric) for row in completed]
        valid = [float(value) for value in values if value is not None]
        coverage[metric] = len(valid)
        # Official score only when every requested episode has this metric.
        summary[metric] = sum(valid)/len(valid) if len(valid) == len(records) and records else None
        summary[f"partial_{metric}"] = sum(valid)/len(valid) if valid else None
    summary["metric_coverage"] = coverage
    summary["complete"] = bool(records) and all(n == len(records) for n in coverage.values())
    return summary


def summarize(output, jobs, fingerprint):
    grouped = defaultdict(list)
    for job in jobs:
        grouped[(job["track"], job["task_name"])].append(read_result(output, job, fingerprint))
    tasks = [{"track": track, "task_name": task, **_aggregate(records)}
             for (track, task), records in sorted(grouped.items())]
    tracks = []
    for track in sorted({row["track"] for row in tasks}):
        rows = [row for row in tasks if row["track"] == track]
        entry = {"track": track, "expected_tasks": len(rows),
                 "completed_tasks": sum(row["complete"] for row in rows),
                 "expected_episodes": sum(row["expected_episodes"] for row in rows),
                 "completed_episodes": sum(row["completed_episodes"] for row in rows),
                 "error_episodes": sum(row["error_episodes"] for row in rows),
                 "missing_episodes": sum(row["missing_episodes"] for row in rows),
                 "missing_official_configs": sum(row["missing_official_configs"] for row in rows),
                 "complete": all(row["complete"] for row in rows)}
        for metric in METRICS:
            values = [row[metric] for row in rows]
            entry[metric] = sum(values)/len(values) if all(x is not None for x in values) else None
        tracks.append(entry)
    overall = {"complete": bool(tracks) and all(row["complete"] for row in tracks)}
    for key in ("expected_episodes", "completed_episodes", "error_episodes", "missing_episodes", "missing_official_configs"):
        overall[key] = sum(row[key] for row in tracks)
    for metric in METRICS:
        values = [row[metric] for row in tracks]
        overall[metric] = sum(values)/len(values) if values and all(x is not None for x in values) else None
    result = {"fingerprint": fingerprint, "score_scale": "0..1", "tasks": tasks,
              "tracks": tracks, "overall": overall}
    write_json_atomic(Path(output) / "summary.json", result)
    Path(output).mkdir(parents=True, exist_ok=True)
    for filename, rows in (("task_results.csv", tasks), ("summary.csv", tracks)):
        if rows:
            with (Path(output)/filename).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return result
