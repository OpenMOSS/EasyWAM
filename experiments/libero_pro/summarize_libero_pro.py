"""Coverage-aware official-matrix result summary for LIBERO-Pro."""

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

from experiments.libero_pro.libero_pro_utils import (  # noqa: E402
    BASE_SUITES,
    PERTURBATION_SUFFIXES,
    TaskSpec,
    error_path,
    is_valid_result,
    read_task_jsonl,
    result_path,
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
        successes = sum(int(result["successes"]) for result in completed_results)
        episodes = sum(int(result["total_episodes"]) for result in completed_results)
        completed = len(completed_results)
        rate = successes / episodes if episodes else None
        row: dict[str, Any] = {
            "group_type": group_type,
            "group": " / ".join(group_key),
            "base_suite": "",
            "perturbation": "",
            "selected_tasks": len(selected),
            "completed_tasks": completed,
            "missing_tasks": len(selected) - completed,
            "infrastructure_errors": sum(
                (task.suite, task.task_id) in errors for task in selected
            ),
            "coverage_percent": completed / len(selected) * 100.0,
            "successes": successes,
            "total_episodes": episodes,
            "success_rate": rate,
            "success_rate_percent": None if rate is None else rate * 100.0,
            "total_duration_seconds": sum(
                float(result.get("duration", 0.0)) for result in completed_results
            ),
        }
        if group_type == "suite":
            row["base_suite"] = group_key[0]
        elif group_type == "perturbation":
            row["perturbation"] = group_key[0]
        elif group_type == "suite_perturbation":
            row["base_suite"], row["perturbation"] = group_key
        rows.append(row)
    return rows


def summarize_results(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    manifest_path = output_dir / "tasks.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"LIBERO-Pro task manifest is missing: {manifest_path}")
    tasks = read_task_jsonl(manifest_path)
    results: dict[tuple[str, int], dict[str, Any]] = {}
    errors: set[tuple[str, int]] = set()
    task_rows: list[dict[str, Any]] = []
    for task in tasks:
        key = (task.suite, task.task_id)
        result_file = result_path(output_dir, task)
        error_file = error_path(output_dir, task)
        result = None
        if is_valid_result(result_file, task):
            result = _read_json(result_file)
            results[key] = result
            status = "completed"
        elif error_file.is_file():
            errors.add(key)
            status = "infrastructure_error"
        else:
            status = "missing"
        rate = (
            None
            if result is None
            else int(result["successes"]) / int(result["total_episodes"])
        )
        task_rows.append(
            {
                **task.to_dict(),
                "status": status,
                "successes": "" if result is None else int(result["successes"]),
                "total_episodes": (
                    "" if result is None else int(result["total_episodes"])
                ),
                "success_rate": "" if rate is None else rate,
                "success_rate_percent": "" if rate is None else rate * 100.0,
                "duration_seconds": (
                    "" if result is None else float(result.get("duration", 0.0))
                ),
                "task_description": (
                    "" if result is None else result.get("task_description", "")
                ),
                "result_path": "" if result is None else str(result_file),
                "error_path": str(error_file) if key in errors else "",
            }
        )

    summary_rows: list[dict[str, Any]] = []
    summary_rows.extend(
        _aggregate("overall", tasks, results, errors, lambda _: ("overall",))
    )
    summary_rows.extend(
        _aggregate("suite", tasks, results, errors, lambda task: (task.base_suite,))
    )
    summary_rows.extend(
        _aggregate(
            "perturbation",
            tasks,
            results,
            errors,
            lambda task: (task.perturbation,),
        )
    )
    matrix_rows = _aggregate(
        "suite_perturbation",
        tasks,
        results,
        errors,
        lambda task: (task.base_suite, task.perturbation),
    )
    summary_rows.extend(matrix_rows)

    cells = {
        (row["base_suite"], row["perturbation"]): row for row in matrix_rows
    }
    leaderboard_rows: list[dict[str, Any]] = []
    complete_cell_rates: list[float] = []
    selected_bases = [
        suite for suite in BASE_SUITES if any(t.base_suite == suite for t in tasks)
    ]
    selected_perturbations = [
        value
        for value in PERTURBATION_SUFFIXES
        if any(t.perturbation == value for t in tasks)
    ]
    for base_suite in selected_bases:
        row: dict[str, Any] = {"base_suite": base_suite}
        suite_rates: list[float] = []
        for perturbation in selected_perturbations:
            cell = cells.get((base_suite, perturbation))
            complete = bool(cell) and cell["missing_tasks"] == 0
            rate = None if not complete else cell["success_rate"]
            row[perturbation] = "" if rate is None else rate
            if rate is not None:
                suite_rates.append(float(rate))
                complete_cell_rates.append(float(rate))
        row["average"] = "" if not suite_rates else sum(suite_rates) / len(suite_rates)
        leaderboard_rows.append(row)
    leaderboard_average = (
        None
        if not complete_cell_rates
        else sum(complete_cell_rates) / len(complete_cell_rates)
    )
    leaderboard_complete = all(
        cells.get((base_suite, perturbation), {}).get("missing_tasks") == 0
        for base_suite in selected_bases
        for perturbation in selected_perturbations
    )

    summary_payload = {
        "run_id": output_dir.name,
        "selected_tasks": len(tasks),
        "completed_tasks": len(results),
        "missing_tasks": len(tasks) - len(results),
        "infrastructure_errors": len(errors),
        "leaderboard_complete": leaderboard_complete,
        "leaderboard_average": leaderboard_average,
        "groups": summary_rows,
    }
    _write_json_atomic(output_dir / "summary.json", summary_payload)

    summary_columns = list(summary_rows[0]) if summary_rows else []
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns)
        if summary_columns:
            writer.writeheader()
            writer.writerows(summary_rows)
    task_columns = list(task_rows[0]) if task_rows else []
    with (output_dir / "task_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=task_columns)
        if task_columns:
            writer.writeheader()
            writer.writerows(task_rows)
    leaderboard_columns = ["base_suite", *selected_perturbations, "average"]
    with (output_dir / "leaderboard.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=leaderboard_columns)
        writer.writeheader()
        writer.writerows(leaderboard_rows)

    overall = summary_rows[0] if summary_rows else None
    print("\n=== LIBERO-Pro Summary ===")
    if overall:
        print(
            f"Coverage: {overall['completed_tasks']}/{overall['selected_tasks']} "
            f"({overall['coverage_percent']:.2f}%)"
        )
        rate = overall["success_rate_percent"]
        rate_text = "N/A" if rate is None else f"{rate:.2f}%"
        print(f"Success rate over completed episodes: {rate_text}")
        print(f"Infrastructure errors: {overall['infrastructure_errors']}")
    average_text = "N/A" if leaderboard_average is None else f"{leaderboard_average:.4f}"
    print(f"Leaderboard macro average: {average_text} (complete={leaderboard_complete})")
    print(f"Summary JSON: {output_dir / 'summary.json'}")
    print(f"Leaderboard CSV: {output_dir / 'leaderboard.csv'}")
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    summarize_results(args.output_dir)


if __name__ == "__main__":
    main()
