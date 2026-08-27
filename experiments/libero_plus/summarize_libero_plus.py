"""Coverage-aware multidimensional result summary for LIBERO-Plus."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero_plus.libero_plus_utils import (  # noqa: E402
    TaskSpec,
    error_path,
    is_valid_result,
    read_task_jsonl,
    result_path,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _difficulty_label(value: int | None) -> str:
    return "unclassified" if value is None else str(value)


def _aggregate(
    group_type: str,
    tasks: Iterable[TaskSpec],
    results: dict[tuple[str, int], dict[str, Any]],
    errors: set[tuple[str, int]],
    key_fn: Callable[[TaskSpec], tuple[str, ...]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[TaskSpec]] = defaultdict(list)
    for task in tasks:
        grouped[key_fn(task)].append(task)

    rows: list[dict[str, Any]] = []
    for group_key in sorted(grouped):
        selected = grouped[group_key]
        completed_results = [
            results[(task.suite, task.task_id)]
            for task in selected
            if (task.suite, task.task_id) in results
        ]
        completed = len(completed_results)
        successes = sum(int(result["successes"]) for result in completed_results)
        total_episodes = sum(int(result["total_episodes"]) for result in completed_results)
        infrastructure_errors = sum(
            (task.suite, task.task_id) in errors for task in selected
        )
        row: dict[str, Any] = {
            "group_type": group_type,
            "group": " / ".join(group_key),
            "suite": "",
            "category": "",
            "difficulty": "",
            "selected_tasks": len(selected),
            "completed_tasks": completed,
            "missing_tasks": len(selected) - completed,
            "infrastructure_errors": infrastructure_errors,
            "coverage_percent": completed / len(selected) * 100.0,
            "successes": successes,
            "total_episodes": total_episodes,
            "success_rate_percent": (
                successes / total_episodes * 100.0 if total_episodes else None
            ),
            "total_duration_seconds": sum(
                float(result.get("duration", 0.0)) for result in completed_results
            ),
        }
        if group_type == "suite":
            row["suite"] = group_key[0]
        elif group_type == "category":
            row["category"] = group_key[0]
        elif group_type == "difficulty":
            row["difficulty"] = group_key[0]
        elif group_type == "category_difficulty":
            row["category"], row["difficulty"] = group_key
        rows.append(row)
    return rows


def summarize_results(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    task_manifest_path = output_dir / "tasks.jsonl"
    if not task_manifest_path.is_file():
        raise FileNotFoundError(f"LIBERO-Plus task list is missing: {task_manifest_path}.")

    tasks = read_task_jsonl(task_manifest_path)
    results: dict[tuple[str, int], dict[str, Any]] = {}
    errors: set[tuple[str, int]] = set()
    task_rows: list[dict[str, Any]] = []

    for task in tasks:
        key = (task.suite, task.task_id)
        task_result_path = result_path(output_dir, task)
        task_error_path = error_path(output_dir, task)
        result = None
        if is_valid_result(task_result_path, task):
            result = _read_json(task_result_path)
            results[key] = result
            status = "completed"
        elif task_error_path.is_file():
            errors.add(key)
            status = "infrastructure_error"
        else:
            status = "missing"

        task_rows.append(
            {
                **task.to_dict(),
                "difficulty_level": _difficulty_label(task.difficulty_level),
                "status": status,
                "successes": "" if result is None else int(result["successes"]),
                "total_episodes": "" if result is None else int(result["total_episodes"]),
                "success_rate_percent": (
                    ""
                    if result is None
                    else float(result["successes"]) / int(result["total_episodes"]) * 100.0
                ),
                "duration_seconds": "" if result is None else float(result.get("duration", 0.0)),
                "result_path": "" if result is None else str(task_result_path),
                "error_path": str(task_error_path) if key in errors else "",
            }
        )

    summary_rows: list[dict[str, Any]] = []
    summary_rows.extend(_aggregate("overall", tasks, results, errors, lambda _: ("overall",)))
    summary_rows.extend(_aggregate("suite", tasks, results, errors, lambda task: (task.suite,)))
    summary_rows.extend(
        _aggregate("category", tasks, results, errors, lambda task: (task.category,))
    )
    summary_rows.extend(
        _aggregate(
            "difficulty",
            tasks,
            results,
            errors,
            lambda task: (_difficulty_label(task.difficulty_level),),
        )
    )
    summary_rows.extend(
        _aggregate(
            "category_difficulty",
            tasks,
            results,
            errors,
            lambda task: (task.category, _difficulty_label(task.difficulty_level)),
        )
    )

    summary_payload = {
        "run_id": output_dir.name,
        "selected_tasks": len(tasks),
        "completed_tasks": len(results),
        "missing_tasks": len(tasks) - len(results),
        "infrastructure_errors": len(errors),
        "groups": summary_rows,
    }
    _write_json_atomic(output_dir / "summary.json", summary_payload)

    summary_columns = [
        "group_type",
        "group",
        "suite",
        "category",
        "difficulty",
        "selected_tasks",
        "completed_tasks",
        "missing_tasks",
        "infrastructure_errors",
        "coverage_percent",
        "successes",
        "total_episodes",
        "success_rate_percent",
        "total_duration_seconds",
    ]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_columns)
        writer.writeheader()
        writer.writerows(summary_rows)

    task_columns = list(task_rows[0]) if task_rows else []
    with (output_dir / "task_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=task_columns)
        if task_columns:
            writer.writeheader()
            writer.writerows(task_rows)

    overall = summary_rows[0]
    rate = overall["success_rate_percent"]
    rate_text = "N/A" if rate is None else f"{rate:.2f}%"
    print("\n=== LIBERO-Plus Summary ===")
    print(
        f"Coverage: {overall['completed_tasks']}/{overall['selected_tasks']} "
        f"({overall['coverage_percent']:.2f}%)"
    )
    print(f"Success rate over completed episodes: {rate_text}")
    print(f"Infrastructure errors: {overall['infrastructure_errors']}")
    print(f"Summary JSON: {output_dir / 'summary.json'}")
    print(f"Summary CSV: {output_dir / 'summary.csv'}")
    print(f"Task results CSV: {output_dir / 'task_results.csv'}")
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    summarize_results(args.output_dir)


if __name__ == "__main__":
    main()
