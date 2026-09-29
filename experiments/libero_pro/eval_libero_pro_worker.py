"""Long-lived single-GPU worker for LIBERO-Pro evaluation."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.batched_inference import DynamicInferenceBatcher  # noqa: E402
from experiments.libero_pro.libero_pro_utils import (  # noqa: E402
    TaskSpec,
    activate_runtime,
    error_path,
    instantiate_suite,
    is_valid_result,
    read_task_jsonl,
    resolve_path,
    result_path,
)
from experiments.task_dispatch import FileTaskDispatcher  # noqa: E402


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="benchmark/sim_libero_pro.yaml",
)
def main(cfg: DictConfig) -> None:
    task_file_value = cfg.WORKER.get("task_file")
    cursor_value = cfg.WORKER.get("task_cursor")
    config_value = cfg.EVALUATION.get("libero_pro_config_path")
    if not task_file_value or not cursor_value or not config_value:
        raise ValueError(
            "WORKER.task_file, WORKER.task_cursor, and "
            "EVALUATION.libero_pro_config_path are required."
        )
    activate_runtime(cfg.EVALUATION.libero_pro_root, config_value)

    # Import only after the requested LIBERO-Pro checkout has been activated.
    from experiments.libero.eval_libero_single import (
        benchmark,
        build_eval_runtime,
        evaluate_task_with_runtime,
        write_json_atomic,
    )

    task_file = resolve_path(str(task_file_value))
    tasks = read_task_jsonl(task_file)
    if not tasks:
        logging.info("Worker task list is empty: %s", task_file)
        return
    output_dir = resolve_path(str(cfg.EVALUATION.output_dir))
    dispatcher = FileTaskDispatcher(task_file, resolve_path(str(cursor_value)))
    runtime = build_eval_runtime(cfg)
    # Official init states are trusted PyTorch pickles. Change this only after
    # loading the model checkpoint so checkpoint deserialization stays strict.
    os.environ.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    actor_count = int(cfg.MULTIRUN.env_num_per_worker)
    runtime.batcher = DynamicInferenceBatcher(
        runtime.model,
        max_batch_size=int(cfg.MULTIRUN.inference_batch_size),
        wait_ms=float(cfg.MULTIRUN.inference_batch_wait_ms),
        prompt_cache_size=int(cfg.MULTIRUN.prompt_cache_size),
    )

    def actor_main(actor_index: int) -> None:
        suites = {}
        while True:
            claimed = dispatcher.claim_with_index()
            if claimed is None:
                return
            task_index, raw = claimed
            task = TaskSpec.from_dict(json.loads(raw))
            destination = result_path(output_dir, task)
            if is_valid_result(destination, task):
                continue
            task_error_path = error_path(output_dir, task)
            task_error_path.unlink(missing_ok=True)
            task_start = time.time()
            try:
                logging.info(
                    "[Task %d/%d] Actor %d started %s:%d (%s)",
                    task_index + 1,
                    len(tasks),
                    actor_index,
                    task.suite,
                    task.task_id,
                    task.task_name,
                )
                local_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
                with open_dict(local_cfg):
                    local_cfg.EVALUATION.task_suite_name = task.suite
                    local_cfg.EVALUATION.task_id = task.task_id
                    local_cfg.EVALUATION.num_trials = task.num_trials
                    local_cfg.EVALUATION.max_steps = task.max_steps
                if task.suite not in suites:
                    suites[task.suite] = instantiate_suite(benchmark, task.suite)
                task_suite = suites[task.suite]
                actual_name = str(task_suite.get_task(task.task_id).name)
                if actual_name != task.task_name:
                    raise RuntimeError(
                        f"Task manifest mismatch for {task.suite}:{task.task_id}: "
                        f"expected {task.task_name!r}, upstream has {actual_name!r}."
                    )
                results = evaluate_task_with_runtime(
                    local_cfg,
                    runtime,
                    task_suite=task_suite,
                    result_metadata={
                        "base_suite": task.base_suite,
                        "perturbation": task.perturbation,
                        "official_max_steps": task.max_steps,
                        "protocol": task.protocol,
                        "source_commit": task.source_commit,
                    },
                )
                write_json_atomic(destination, results)
                logging.info(
                    "[Task %d/%d] Actor %d completed %s:%d: successes=%d/%d duration=%.2fs",
                    task_index + 1,
                    len(tasks),
                    actor_index,
                    task.suite,
                    task.task_id,
                    results["successes"],
                    results["total_episodes"],
                    results["duration"],
                )
            except BaseException as exc:
                write_json_atomic(
                    task_error_path,
                    {
                        **task.to_dict(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "duration": time.time() - task_start,
                    },
                )
                raise

    try:
        with ThreadPoolExecutor(
            max_workers=actor_count, thread_name_prefix="rollout"
        ) as executor:
            futures = [executor.submit(actor_main, index) for index in range(actor_count)]
            for future in futures:
                future.result()
    finally:
        runtime.batcher.close()


if __name__ == "__main__":
    main()
