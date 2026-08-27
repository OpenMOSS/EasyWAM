"""Long-lived single-GPU worker for LIBERO-Plus evaluation."""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback
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
from experiments.libero_plus.libero_plus_utils import (  # noqa: E402
    error_path,
    instantiate_suite,
    is_valid_result,
    read_task_jsonl,
    result_path,
)
from utils.pytorch_utils import set_global_seed  # noqa: E402


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero_plus.yaml")
def main(cfg: DictConfig) -> None:
    if int(cfg.EVALUATION.num_trials) != 1:
        raise ValueError("LIBERO-Plus requires EVALUATION.num_trials=1.")
    task_file_value = cfg.WORKER.get("task_file")
    if not task_file_value:
        raise ValueError("WORKER.task_file is required.")

    task_file = Path(str(task_file_value)).expanduser().resolve()
    output_dir = Path(str(cfg.EVALUATION.output_dir)).expanduser().resolve()
    tasks = read_task_jsonl(task_file)
    if not tasks:
        logging.info("Worker shard is empty: %s", task_file)
        return

    runtime = build_eval_runtime(cfg)
    # LIBERO-Plus init-state files are trusted local NumPy pickles. Set this
    # only after the model checkpoint has loaded; upstream calls torch.load
    # without an explicit weights_only argument when tasks are evaluated.
    os.environ.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    suites = {
        suite_name: instantiate_suite(benchmark, suite_name)
        for suite_name in dict.fromkeys(task.suite for task in tasks)
    }
    logging.info("Loaded model once for %d LIBERO-Plus tasks", len(tasks))

    for position, task in enumerate(tasks, start=1):
        destination = result_path(output_dir, task)
        if is_valid_result(destination, task):
            logging.info("[%d/%d] Skip completed %s:%d", position, len(tasks), task.suite, task.task_id)
            continue

        task_error_path = error_path(output_dir, task)
        task_error_path.unlink(missing_ok=True)
        task_start = time.time()
        try:
            if cfg.get("seed") is not None:
                # Persistent workers must reproduce the old one-process-per-task RNG semantics.
                set_global_seed(int(cfg.seed), get_worker_init_fn=False)
            with open_dict(cfg):
                cfg.EVALUATION.task_suite_name = task.suite
                cfg.EVALUATION.task_id = task.task_id

            task_suite = suites[task.suite]
            actual_name = str(task_suite.get_task(task.task_id).name)
            if actual_name != task.task_name:
                raise RuntimeError(
                    f"Task manifest mismatch for {task.suite}:{task.task_id}: "
                    f"expected {task.task_name!r}, installed benchmark has {actual_name!r}."
                )

            results = evaluate_task_with_runtime(
                cfg,
                runtime,
                task_suite=task_suite,
                result_metadata={
                    "task_name": task.task_name,
                    "category": task.category,
                    "category_label": task.category_label,
                    "difficulty_level": task.difficulty_level,
                    "classification_id": task.classification_id,
                },
            )
            write_started = time.perf_counter()
            write_json_atomic(destination, results)
            write_seconds = time.perf_counter() - write_started
            logging.info(
                "[%d/%d] Completed %s:%d success=%d duration=%.2fs",
                position,
                len(tasks),
                task.suite,
                task.task_id,
                results["successes"],
                results["duration"],
            )
            if bool(cfg.EVALUATION.get("timing_enabled", False)):
                logging.info(
                    "Result write time for %s:%d: %.6fs",
                    task.suite,
                    task.task_id,
                    write_seconds,
                )
        except BaseException as exc:
            error_payload = {
                **task.to_dict(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "duration": time.time() - task_start,
            }
            write_json_atomic(task_error_path, error_payload)
            logging.exception("Infrastructure failure in %s:%d", task.suite, task.task_id)
            raise


if __name__ == "__main__":
    main()
