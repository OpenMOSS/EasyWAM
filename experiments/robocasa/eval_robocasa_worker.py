"""Long-lived model worker for dynamically batched RoboCasa rollouts."""

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

from experiments.batched_inference import DynamicInferenceBatcher  # noqa: E402
from experiments.robocasa.eval_robocasa_single import (  # noqa: E402
    build_eval_runtime,
    evaluate_task_with_runtime,
    write_json_atomic,
)
from experiments.robocasa.result_utils import (  # noqa: E402
    task_result_path,
    valid_result_path,
)
from experiments.task_dispatch import FileTaskDispatcher  # noqa: E402


def read_tasks(path: Path) -> list[tuple[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [(row[0].strip(), row[1].strip()) for row in csv.reader(handle) if row]
    if any(not task_set or not task_name for task_set, task_name in rows):
        raise ValueError(f"Invalid RoboCasa task row in {path}.")
    return rows


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="benchmark/sim_robocasa.yaml",
)
def main(cfg: DictConfig) -> None:
    task_file_value = cfg.WORKER.get("task_file")
    cursor_value = cfg.WORKER.get("task_cursor")
    if not task_file_value or not cursor_value:
        raise ValueError("WORKER.task_file and WORKER.task_cursor are required.")
    task_path = Path(str(task_file_value)).expanduser().resolve()
    tasks = read_tasks(task_path)
    if not tasks:
        return
    dispatcher = FileTaskDispatcher(
        task_path, Path(str(cursor_value)).expanduser().resolve()
    )
    output_dir = Path(str(cfg.EVALUATION.output_dir)).expanduser().resolve()
    runtime = build_eval_runtime(cfg)
    runtime.batcher = DynamicInferenceBatcher(
        runtime.model,
        max_batch_size=int(cfg.MULTIRUN.inference_batch_size),
        wait_ms=float(cfg.MULTIRUN.inference_batch_wait_ms),
        prompt_cache_size=int(cfg.MULTIRUN.prompt_cache_size),
    )
    actor_count = int(cfg.MULTIRUN.env_num_per_worker)
    worker_index = int(cfg.WORKER.get("worker_index", cfg.gpu_id))

    def actor_main(actor_index: int) -> None:
        while True:
            claimed = dispatcher.claim_with_index()
            if claimed is None:
                return
            task_index, raw = claimed
            task_set, task_name = next(csv.reader([raw]))
            if valid_result_path(
                output_dir,
                task_set,
                task_name,
                str(cfg.EVALUATION.split),
                int(cfg.EVALUATION.num_trials),
            ):
                continue
            logging.info(
                "[Task %d/%d] actor=%d task=%s/%s",
                task_index + 1,
                len(tasks),
                actor_index,
                task_set,
                task_name,
            )
            local_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
            with open_dict(local_cfg):
                local_cfg.EVALUATION.task_set = task_set
                local_cfg.EVALUATION.task_name = task_name
            result = evaluate_task_with_runtime(
                local_cfg,
                runtime,
                result_metadata={"worker_index": worker_index, "actor_index": actor_index},
            )
            write_json_atomic(
                task_result_path(output_dir, task_set, task_name), result
            )
            logging.info(
                "Completed %s/%s: %d/%d in %.2fs",
                task_set,
                task_name,
                result["successes"],
                result["total_episodes"],
                result["duration"],
            )

    try:
        with ThreadPoolExecutor(
            max_workers=actor_count, thread_name_prefix="robocasa-rollout"
        ) as executor:
            futures = [executor.submit(actor_main, index) for index in range(actor_count)]
            for future in futures:
                future.result()
    finally:
        runtime.batcher.close()


if __name__ == "__main__":
    main()
