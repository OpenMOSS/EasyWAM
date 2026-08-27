"""Long-lived worker for the standard LIBERO benchmark."""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, open_dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.libero.eval_libero_single import (  # noqa: E402
    benchmark,
    build_eval_runtime,
    evaluate_task_with_runtime,
    write_json_atomic,
)
from utils.pytorch_utils import set_global_seed  # noqa: E402
from experiments.libero.result_utils import valid_result_path  # noqa: E402


def _read_tasks(path: Path) -> list[tuple[str, int]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [(row[0].strip(), int(row[1])) for row in csv.reader(f) if row]
    if any(not suite for suite, _ in rows):
        raise ValueError(f"Invalid empty suite name in {path}.")
    return rows


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def main(cfg: DictConfig) -> None:
    task_file_value = cfg.WORKER.get("task_file")
    if not task_file_value:
        raise ValueError("WORKER.task_file is required.")
    tasks = _read_tasks(Path(str(task_file_value)).expanduser().resolve())
    if not tasks:
        return

    output_dir = Path(str(cfg.EVALUATION.output_dir)).expanduser().resolve()
    runtime = build_eval_runtime(cfg)
    benchmark_dict = benchmark.get_benchmark_dict()
    suites = {
        name: benchmark_dict[name]() for name in dict.fromkeys(name for name, _ in tasks)
    }
    logging.info("Loaded model once for %d standard LIBERO tasks", len(tasks))

    worker_index = int(cfg.WORKER.get("worker_index", cfg.gpu_id))
    for position, (suite_name, task_id) in enumerate(tasks, start=1):
        destination = output_dir / suite_name / f"gpu{worker_index}_task{task_id}_results.json"
        existing = valid_result_path(
            output_dir, suite_name, task_id, int(cfg.EVALUATION.num_trials)
        )
        if existing is not None:
            logging.info(
                "[%d/%d] Skip completed %s:%d (%s)",
                position,
                len(tasks),
                suite_name,
                task_id,
                existing,
            )
            continue
        if cfg.get("seed") is not None:
            set_global_seed(int(cfg.seed), get_worker_init_fn=False)
        with open_dict(cfg):
            cfg.EVALUATION.task_suite_name = suite_name
            cfg.EVALUATION.task_id = task_id
        result = evaluate_task_with_runtime(cfg, runtime, task_suite=suites[suite_name])
        write_started = time.perf_counter()
        write_json_atomic(destination, result)
        write_seconds = time.perf_counter() - write_started
        logging.info("[%d/%d] Completed %s:%d", position, len(tasks), suite_name, task_id)
        if bool(cfg.EVALUATION.get("timing_enabled", False)):
            logging.info("Result write time for %s:%d: %.6fs", suite_name, task_id, write_seconds)


if __name__ == "__main__":
    main()
