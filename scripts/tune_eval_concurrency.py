#!/usr/bin/env python3
"""Benchmark evaluation concurrency settings on the current machine.

The script launches a real benchmark manager for every requested combination.
Arguments after ``--`` are forwarded to that manager. Keep the workload fixed
and representative (same tasks, episodes, checkpoint, and GPU selection) so
wall-clock times remain comparable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANAGERS = {
    "libero": PROJECT_ROOT / "experiments/libero/run_libero_manager.py",
    "libero_plus": PROJECT_ROOT
    / "experiments/libero_plus/run_libero_plus_manager.py",
    "robocasa": PROJECT_ROOT / "experiments/robocasa/run_robocasa_manager.py",
    "robotwin": PROJECT_ROOT / "experiments/robotwin/run_robotwin_manager.py",
}
CONTROLLED_OVERRIDES = {
    "EVALUATION.output_dir",
    "MULTIRUN.env_num_per_worker",
    "MULTIRUN.inference_batch_size",
    "MULTIRUN.inference_batch_wait_ms",
}
OOM_PATTERN = re.compile(
    r"out of memory|cuda error: out of memory|cublas_status_alloc_failed",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Candidate:
    env_num_per_worker: int
    inference_batch_size: int
    inference_batch_wait_ms: float

    @property
    def key(self) -> str:
        wait = str(self.inference_batch_wait_ms).replace(".", "p")
        return (
            f"env{self.env_num_per_worker}_batch{self.inference_batch_size}"
            f"_wait{wait}ms"
        )


@dataclass
class RunResult:
    candidate: Candidate
    repeat: int
    status: str
    return_code: int | None
    wall_seconds: float
    output_dir: str
    log_path: str
    peak_gpu_memory_mib: dict[str, int]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.update(asdict(self.candidate))
        del payload["candidate"]
        return payload


def _csv_values(raw: str, cast, label: str) -> list:
    try:
        values = [cast(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid {label}: {raw!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{label} must not be empty.")
    return values


def _parse_gpu_ids(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    values = _csv_values(raw, int, "monitor GPU IDs")
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("Monitor GPU IDs must be non-negative.")
    return set(values)


def _query_gpu_memory(selected: set[int] | None) -> dict[str, int]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    result = {}
    for line in completed.stdout.splitlines():
        try:
            raw_index, raw_memory = line.split(",", 1)
            index = int(raw_index.strip())
            memory = int(raw_memory.strip())
        except ValueError:
            continue
        if selected is None or index in selected:
            result[str(index)] = memory
    return result


def _monitor_memory(
    stop: threading.Event,
    selected: set[int] | None,
    peak: dict[str, int],
    interval_seconds: float,
) -> None:
    while not stop.is_set():
        for gpu_id, memory in _query_gpu_memory(selected).items():
            peak[gpu_id] = max(peak.get(gpu_id, 0), memory)
        stop.wait(interval_seconds)


def _override_key(argument: str) -> str | None:
    if "=" not in argument:
        return None
    return argument.split("=", 1)[0].lstrip("+~")


def _base_manager_args(arguments: Sequence[str]) -> list[str]:
    values = list(arguments)
    if values and values[0] == "--":
        values = values[1:]
    return [value for value in values if _override_key(value) not in CONTROLLED_OVERRIDES]


def _terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _run_candidate(
    *,
    manager: Path,
    manager_args: Sequence[str],
    candidate: Candidate,
    repeat: int,
    output_root: Path,
    timeout_seconds: float | None,
    monitor_gpu_ids: set[int] | None,
    monitor_interval_seconds: float,
    dry_run: bool,
) -> RunResult:
    run_dir = output_root / candidate.key / f"repeat_{repeat:02d}"
    eval_output = run_dir / "evaluation"
    log_path = run_dir / "manager.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(manager),
        *manager_args,
        f"MULTIRUN.env_num_per_worker={candidate.env_num_per_worker}",
        f"MULTIRUN.inference_batch_size={candidate.inference_batch_size}",
        f"MULTIRUN.inference_batch_wait_ms={candidate.inference_batch_wait_ms}",
        f"EVALUATION.output_dir={eval_output}",
    ]
    (run_dir / "command.json").write_text(
        json.dumps(command, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[{candidate.key} repeat={repeat}] {' '.join(command)}", flush=True)
    if dry_run:
        return RunResult(
            candidate, repeat, "dry_run", None, 0.0,
            str(eval_output), str(log_path), {},
        )

    peak_memory: dict[str, int] = {}
    stop_monitor = threading.Event()
    monitor = threading.Thread(
        target=_monitor_memory,
        args=(
            stop_monitor,
            monitor_gpu_ids,
            peak_memory,
            monitor_interval_seconds,
        ),
        daemon=True,
    )
    started = time.perf_counter()
    status = "failed"
    return_code = None
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        monitor.start()
        try:
            return_code = process.wait(timeout=timeout_seconds)
            status = "ok" if return_code == 0 else "failed"
        except subprocess.TimeoutExpired:
            status = "timeout"
            _terminate_process_group(process)
            return_code = process.returncode
        except KeyboardInterrupt:
            _terminate_process_group(process)
            raise
        finally:
            stop_monitor.set()
            monitor.join(timeout=max(1.0, monitor_interval_seconds * 2))
    wall_seconds = time.perf_counter() - started
    if status == "failed":
        try:
            if OOM_PATTERN.search(log_path.read_text(encoding="utf-8", errors="replace")):
                status = "oom"
        except OSError:
            pass
    print(
        f"  -> {status}, {wall_seconds:.2f}s, peak GPU memory={peak_memory}",
        flush=True,
    )
    return RunResult(
        candidate, repeat, status, return_code, wall_seconds,
        str(eval_output), str(log_path), peak_memory,
    )


def _aggregate(results: Sequence[RunResult], tolerance: float) -> tuple[list[dict], dict | None]:
    grouped: dict[Candidate, list[RunResult]] = defaultdict(list)
    for result in results:
        grouped[result.candidate].append(result)
    rows = []
    for candidate, runs in grouped.items():
        successful = [run for run in runs if run.status == "ok"]
        peak_by_gpu: dict[str, int] = {}
        for run in runs:
            for gpu_id, memory in run.peak_gpu_memory_mib.items():
                peak_by_gpu[gpu_id] = max(peak_by_gpu.get(gpu_id, 0), memory)
        rows.append(
            {
                **asdict(candidate),
                "successful_repeats": len(successful),
                "total_repeats": len(runs),
                "median_wall_seconds": (
                    statistics.median(run.wall_seconds for run in successful)
                    if successful
                    else None
                ),
                "peak_gpu_memory_mib": peak_by_gpu,
                "statuses": [run.status for run in runs],
            }
        )
    valid = [row for row in rows if row["successful_repeats"] == row["total_repeats"]]
    if not valid:
        return rows, None
    fastest = min(float(row["median_wall_seconds"]) for row in valid)
    near_fastest = [
        row
        for row in valid
        if float(row["median_wall_seconds"]) <= fastest * (1.0 + tolerance)
    ]
    recommendation = min(
        near_fastest,
        key=lambda row: (
            int(row["env_num_per_worker"]),
            int(row["inference_batch_size"]),
            float(row["inference_batch_wait_ms"]),
            float(row["median_wall_seconds"]),
        ),
    )
    return rows, recommendation


def _write_reports(
    output_root: Path,
    results: Sequence[RunResult],
    aggregate: Sequence[dict],
    recommendation: dict | None,
) -> None:
    payload = {
        "runs": [result.to_dict() for result in results],
        "aggregate": list(aggregate),
        "recommendation": recommendation,
    }
    (output_root / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "env_num_per_worker",
        "inference_batch_size",
        "inference_batch_wait_ms",
        "successful_repeats",
        "total_repeats",
        "median_wall_seconds",
        "peak_gpu_memory_mib",
        "statuses",
    ]
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in aggregate:
            serialized = dict(row)
            serialized["peak_gpu_memory_mib"] = json.dumps(
                serialized["peak_gpu_memory_mib"], sort_keys=True
            )
            serialized["statuses"] = ",".join(serialized["statuses"])
            writer.writerow(serialized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep real evaluation concurrency settings and recommend a stable configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark", choices=sorted(MANAGERS), required=True)
    parser.add_argument("--env-counts", default="4,8")
    parser.add_argument("--batch-sizes", default="2,4,8")
    parser.add_argument("--wait-ms", default="0,10")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--monitor-gpu-ids",
        help="Comma-separated physical GPU IDs to include in memory sampling; default samples all.",
    )
    parser.add_argument("--monitor-interval-seconds", type=float, default=0.5)
    parser.add_argument(
        "--throughput-tolerance",
        type=float,
        default=0.03,
        help="Prefer a smaller configuration when it is within this fraction of the fastest.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("manager_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive.")
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive.")
    if args.monitor_interval_seconds <= 0:
        parser.error("--monitor-interval-seconds must be positive.")
    if args.throughput_tolerance < 0:
        parser.error("--throughput-tolerance must be non-negative.")
    return args


def main() -> None:
    args = parse_args()
    env_counts = _csv_values(args.env_counts, int, "environment counts")
    batch_sizes = _csv_values(args.batch_sizes, int, "batch sizes")
    waits = _csv_values(args.wait_ms, float, "wait times")
    if any(value <= 0 for value in env_counts + batch_sizes):
        raise SystemExit("Environment counts and batch sizes must be positive.")
    if any(value < 0 for value in waits):
        raise SystemExit("Wait times must be non-negative.")
    monitor_gpu_ids = _parse_gpu_ids(args.monitor_gpu_ids)
    manager_args = _base_manager_args(args.manager_args)
    if not manager_args:
        raise SystemExit(
            "Pass the benchmark's task/checkpoint and fixed workload overrides after `--`."
        )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = (
        args.output_root
        if args.output_root is not None
        else PROJECT_ROOT
        / "evaluate_results"
        / "concurrency_tuning"
        / args.benchmark
        / timestamp
    ).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    candidates = [
        Candidate(env_count, batch_size, wait_ms)
        for env_count in env_counts
        for batch_size in batch_sizes
        for wait_ms in waits
        if batch_size <= env_count
    ]
    if not candidates:
        raise SystemExit("No valid candidates: every batch size exceeds every environment count.")
    print(f"Running {len(candidates)} candidates x {args.repeats} repeats", flush=True)
    results = []
    for candidate in candidates:
        for repeat in range(1, args.repeats + 1):
            result = _run_candidate(
                manager=MANAGERS[args.benchmark],
                manager_args=manager_args,
                candidate=candidate,
                repeat=repeat,
                output_root=output_root,
                timeout_seconds=args.timeout_seconds,
                monitor_gpu_ids=monitor_gpu_ids,
                monitor_interval_seconds=args.monitor_interval_seconds,
                dry_run=args.dry_run,
            )
            results.append(result)
            aggregate, recommendation = _aggregate(
                results, args.throughput_tolerance
            )
            _write_reports(output_root, results, aggregate, recommendation)

    aggregate, recommendation = _aggregate(results, args.throughput_tolerance)
    print(f"Reports: {output_root / 'summary.csv'}", flush=True)
    if recommendation is None:
        print("No configuration completed every repeat successfully.", flush=True)
        raise SystemExit(1 if not args.dry_run else 0)
    print("Recommended overrides:", flush=True)
    print(
        "  "
        f"MULTIRUN.env_num_per_worker={recommendation['env_num_per_worker']} "
        f"MULTIRUN.inference_batch_size={recommendation['inference_batch_size']} "
        f"MULTIRUN.inference_batch_wait_ms={recommendation['inference_batch_wait_ms']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
