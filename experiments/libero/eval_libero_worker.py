"""Long-lived worker for the standard LIBERO benchmark."""

from __future__ import annotations

import csv
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

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
from experiments.libero.result_utils import valid_result_path  # noqa: E402
from experiments.batched_inference import DynamicInferenceBatcher  # noqa: E402
from experiments.task_dispatch import FileTaskDispatcher  # noqa: E402


def _read_tasks(path: Path) -> list[tuple[str, int]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [(row[0].strip(), int(row[1])) for row in csv.reader(f) if row]
    if any(not suite for suite, _ in rows):
        raise ValueError(f"Invalid empty suite name in {path}.")
    return rows


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="benchmark/sim_libero.yaml",
)
def main(cfg: DictConfig) -> None:
    task_file_value = cfg.WORKER.get("task_file")
    if not task_file_value:
        raise ValueError("WORKER.task_file is required.")
    task_path = Path(str(task_file_value)).expanduser().resolve()
    tasks = _read_tasks(task_path)
    if not tasks:
        return
    cursor_value = cfg.WORKER.get("task_cursor")
    if not cursor_value:
        raise ValueError("WORKER.task_cursor is required.")
    dispatcher = FileTaskDispatcher(task_path, Path(str(cursor_value)).expanduser().resolve())

    output_dir = Path(str(cfg.EVALUATION.output_dir)).expanduser().resolve()
    runtime = build_eval_runtime(cfg)
    actor_count = int(cfg.MULTIRUN.env_num_per_worker)
    runtime.batcher = DynamicInferenceBatcher(
        runtime.model,
        max_batch_size=int(cfg.MULTIRUN.inference_batch_size),
        wait_ms=float(cfg.MULTIRUN.inference_batch_wait_ms),
        prompt_cache_size=int(cfg.MULTIRUN.prompt_cache_size),
    )
    worker_index = int(cfg.WORKER.get("worker_index", cfg.gpu_id))

    def actor_main(actor_index: int) -> None:
        benchmark_dict = benchmark.get_benchmark_dict()
        suites = {}
        while True:
            claimed = dispatcher.claim_with_index()
            if claimed is None:
                return
            task_index, raw = claimed
            suite_name, task_id_text = next(csv.reader([raw]))
            task_id = int(task_id_text)
            destination = output_dir / suite_name / f"gpu{worker_index}_task{task_id}_results.json"
            if valid_result_path(output_dir, suite_name, task_id, int(cfg.EVALUATION.num_trials)) is not None:
                continue
            logging.info(
                "[Task %d/%d] Actor %d started %s:%d",
                task_index + 1, len(tasks), actor_index, suite_name, task_id,
            )
            local_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
            with open_dict(local_cfg):
                local_cfg.EVALUATION.task_suite_name = suite_name
                local_cfg.EVALUATION.task_id = task_id
            if suite_name not in suites:
                suites[suite_name] = benchmark_dict[suite_name]()
            result = evaluate_task_with_runtime(local_cfg, runtime, task_suite=suites[suite_name])
            write_json_atomic(destination, result)
            logging.info(
                "[Task %d/%d] Actor %d completed %s:%d: successes=%d/%d duration=%.2fs",
                task_index + 1, len(tasks), actor_index, suite_name, task_id,
                result["successes"], result["total_episodes"], result["duration"],
            )

    try:
        with ThreadPoolExecutor(max_workers=actor_count, thread_name_prefix="rollout") as executor:
            futures = [executor.submit(actor_main, index) for index in range(actor_count)]
            for future in futures:
                future.result()
    finally:
        runtime.batcher.close()


if __name__ == "__main__":
    main()
