"""Create task, task-set, and weighted overall RoboCasa summaries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from experiments.robocasa.result_utils import valid_result_path


def summarize_results(
    output_dir: str | Path,
    jobs: Iterable[tuple[str, str]] | None = None,
    *,
    split: str = "pretrain",
    expected_episodes: int | None = None,
) -> dict:
    output_dir = Path(output_dir).expanduser().resolve()
    if jobs is None:
        jobs = [
            (path.parents[1].name, path.parent.name)
            for path in sorted(output_dir.glob("*/*/result.json"))
        ]
    jobs = list(jobs)
    suite_accumulator = defaultdict(
        lambda: {"tasks": 0, "successes": 0, "episodes": 0, "duration": 0.0}
    )
    task_rows = []
    missing = []
    for task_set, task_name in jobs:
        path = valid_result_path(
            output_dir, task_set, task_name, split, expected_episodes
        )
        if path is None:
            missing.append({"task_set": task_set, "task_name": task_name})
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        successes = int(payload["successes"])
        episodes = int(payload["total_episodes"])
        duration = float(payload.get("duration", 0.0))
        suite = suite_accumulator[task_set]
        suite["tasks"] += 1
        suite["successes"] += successes
        suite["episodes"] += episodes
        suite["duration"] += duration
        task_rows.append(
            {
                "task_set": task_set,
                "task_name": task_name,
                "task_description": payload.get("task_description") or "",
                "successes": successes,
                "episodes": episodes,
                "success_rate": successes / episodes,
                "duration_seconds": duration,
            }
        )

    suite_stats = {}
    total_successes = 0
    total_episodes = 0
    total_duration = 0.0
    for task_set in dict.fromkeys(task_set for task_set, _ in jobs):
        stats = suite_accumulator[task_set]
        episodes = int(stats["episodes"])
        suite_stats[task_set] = {
            "completed_tasks": int(stats["tasks"]),
            "successes": int(stats["successes"]),
            "episodes": episodes,
            "success_rate": stats["successes"] / episodes if episodes else None,
            "duration_seconds": float(stats["duration"]),
        }
        total_successes += int(stats["successes"])
        total_episodes += episodes
        total_duration += float(stats["duration"])

    summary = {
        "split": split,
        "expected_tasks": len(jobs),
        "completed_tasks": len(task_rows),
        "missing_tasks": missing,
        "task_set_stats": suite_stats,
        "overall": {
            # Weight by all episodes/tasks, matching the 50-task leaderboard mean
            # when every task has the configured number of trials.
            "successes": total_successes,
            "episodes": total_episodes,
            "success_rate": total_successes / total_episodes if total_episodes else None,
            "duration_seconds": total_duration,
        },
        "task_results": task_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_set",
                "completed_tasks",
                "successes",
                "episodes",
                "success_rate",
                "duration_seconds",
            ],
        )
        writer.writeheader()
        for task_set, stats in suite_stats.items():
            writer.writerow({"task_set": task_set, **stats})
        writer.writerow(
            {
                "task_set": "overall",
                "completed_tasks": len(task_rows),
                **summary["overall"],
            }
        )
    with (output_dir / "task_success_rates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "task_set",
            "task_name",
            "task_description",
            "successes",
            "episodes",
            "success_rate",
            "duration_seconds",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(task_rows)

    rate = summary["overall"]["success_rate"]
    print(
        f"RoboCasa: {len(task_rows)}/{len(jobs)} tasks complete; "
        f"overall success rate={rate * 100:.2f}%"
        if rate is not None
        else f"RoboCasa: 0/{len(jobs)} tasks complete."
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--split", default="pretrain")
    parser.add_argument("--expected-episodes", type=int)
    args = parser.parse_args()
    summarize_results(
        args.output_dir, split=args.split, expected_episodes=args.expected_episodes
    )


if __name__ == "__main__":
    main()
