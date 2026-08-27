"""Multi-GPU manager for the full LIBERO-Plus robustness benchmark."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER_ENTRY = PROJECT_ROOT / "experiments" / "libero_plus" / "eval_libero_plus_worker.py"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero_plus.libero_plus_utils import (  # noqa: E402
    TaskSpec,
    is_valid_result,
    load_libero_plus_catalog,
    result_path,
    select_tasks,
    write_jsonl,
)
from experiments.libero.render_backend import (  # noqa: E402
    configure_mujoco_worker_env,
    select_mujoco_render_backend,
)
from experiments.libero.run_libero_manager import build_worker_slots  # noqa: E402


def _resolve_path(value: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(value))).resolve()


def _string_override(key: str, value: Any) -> str:
    return f"{key}={json.dumps(str(value))}"


def _optional_list(value) -> Optional[list[Any]]:
    if value is None:
        return None
    return list(value)


def _resolve_task_choice() -> str:
    choice = HydraConfig.get().runtime.choices.get("task")
    if choice is None or not str(choice).strip():
        raise ValueError("Hydra task choice is empty; pass task=libero_easywam_mot or another LIBERO task.")
    return str(choice)


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    blocked = {
        "task",
        "ckpt",
        "gpu_id",
        "EVALUATION.output_dir",
        "EVALUATION.dataset_stats_path",
        "EVALUATION.task_suite_name",
        "EVALUATION.task_id",
    }
    return (
        key in blocked
        or key.startswith("MULTIRUN.")
        or key.startswith("WORKER.")
        or key.startswith("hydra.")
    )


def _collect_worker_overrides() -> list[str]:
    return [
        value
        for value in HydraConfig.get().overrides.task
        if not _is_blocked_override(value)
    ]


def _resolve_dataset_stats_path(cfg: DictConfig) -> Optional[Path]:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    if explicit is not None:
        return _resolve_path(str(explicit))
    if cfg.ckpt is None:
        return None
    checkpoint = _resolve_path(str(cfg.ckpt))
    for parent in list(checkpoint.parents)[:4]:
        candidate = parent / "dataset_stats.json"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _pending_tasks(output_dir: Path, tasks: list[TaskSpec]) -> list[TaskSpec]:
    completed = [task for task in tasks if is_valid_result(result_path(output_dir, task), task)]
    completed_keys = {(task.suite, task.task_id) for task in completed}
    return [task for task in tasks if (task.suite, task.task_id) not in completed_keys]


def _terminate_processes(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.time() + 10
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                process.kill()
    for process in processes:
        if process.poll() is None:
            process.wait()


def _build_shards(tasks: list[TaskSpec], worker_count: int) -> list[list[TaskSpec]]:
    if worker_count <= 0:
        raise ValueError(f"worker_count must be positive, got {worker_count}.")
    shards: list[list[TaskSpec]] = [[] for _ in range(worker_count)]
    for index, task in enumerate(tasks):
        shards[index % worker_count].append(task)
    return shards


def _summarize(output_dir: Path) -> None:
    from experiments.libero_plus.summarize_libero_plus import summarize_results

    summarize_results(output_dir)


def _run_workers(
    cfg: DictConfig,
    *,
    output_dir: Path,
    task_choice: str,
    checkpoint: Path,
    dataset_stats: Path,
    pending: list[TaskSpec],
) -> None:
    num_gpus = int(cfg.MULTIRUN.get("num_gpus", 1))
    max_tasks_per_gpu = int(cfg.MULTIRUN.get("max_tasks_per_gpu", 1))
    slots = build_worker_slots(num_gpus, max_tasks_per_gpu)
    render_backend = select_mujoco_render_backend(max_tasks_per_gpu)
    print(
        f"MuJoCo rendering backend: {render_backend} "
        f"(num_gpus={num_gpus}, max_tasks_per_gpu={max_tasks_per_gpu})"
    )
    worker_count = min(len(slots), len(pending))
    shards = _build_shards(pending, worker_count)

    worker_dir = output_dir / "workers"
    log_dir = output_dir / "logs"
    worker_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    extra_overrides = _collect_worker_overrides()
    processes: list[subprocess.Popen] = []
    log_handles = []
    try:
        for worker_index, ((gpu_id, slot), shard) in enumerate(zip(slots, shards)):
            shard_path = worker_dir / f"worker_{worker_index:03d}.jsonl"
            write_jsonl(shard_path, (task.to_dict() for task in shard))
            log_path = log_dir / f"worker_{worker_index:03d}_gpu_{gpu_id}_slot_{slot}.log"
            log_handle = log_path.open("a", encoding="utf-8")
            log_handles.append(log_handle)
            command = [
                sys.executable,
                str(WORKER_ENTRY),
                f"task={task_choice}",
                _string_override("ckpt", checkpoint),
                _string_override("EVALUATION.output_dir", output_dir),
                _string_override("EVALUATION.dataset_stats_path", dataset_stats),
                f"gpu_id={gpu_id}",
                _string_override("WORKER.task_file", shard_path),
                *extra_overrides,
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            configure_mujoco_worker_env(env, max_tasks_per_gpu)
            env.setdefault("PYTHONFAULTHANDLER", "1")
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
            print(
                f"Started worker {worker_index} on GPU {gpu_id} slot {slot}: "
                f"{len(shard)} tasks, log={log_path}"
            )

        while processes:
            failed = next(
                (process for process in processes if process.poll() not in (None, 0)),
                None,
            )
            if failed is not None:
                return_code = int(failed.returncode)
                _terminate_processes(processes)
                raise RuntimeError(
                    f"LIBERO-Plus worker failed with return code {return_code}. "
                    f"Other workers were terminated; inspect {log_dir} and {output_dir / 'errors'}."
                )
            if all(process.poll() == 0 for process in processes):
                break
            time.sleep(2)
    except BaseException:
        _terminate_processes(processes)
        raise
    finally:
        for log_handle in log_handles:
            log_handle.close()


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero_plus.yaml")
def main(cfg: DictConfig) -> None:
    if int(cfg.EVALUATION.num_trials) != 1:
        raise ValueError("Official LIBERO-Plus evaluation requires EVALUATION.num_trials=1.")
    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    suite_names = [str(value) for value in cfg.MULTIRUN.task_suite_names]
    catalog = load_libero_plus_catalog(suite_names)
    tasks = select_tasks(
        catalog.tasks_by_suite,
        categories=_optional_list(cfg.MULTIRUN.get("categories")),
        difficulty_levels=_optional_list(cfg.MULTIRUN.get("difficulty_levels")),
        task_ids=_optional_list(cfg.MULTIRUN.get("task_ids")),
    )
    task_choice = _resolve_task_choice()
    checkpoint = None if cfg.ckpt is None else _resolve_path(str(cfg.ckpt))
    dataset_stats = _resolve_dataset_stats_path(cfg)

    if not bool(cfg.MULTIRUN.get("create_only", False)):
        if checkpoint is None or not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        if dataset_stats is None or not dataset_stats.is_file():
            raise FileNotFoundError(
                "dataset_stats.json was not found. Pass EVALUATION.dataset_stats_path explicitly."
            )

    write_jsonl(output_dir / "tasks.jsonl", (task.to_dict() for task in tasks))
    OmegaConf.save(config=cfg, f=str(output_dir / "manager_config.yaml"))

    print(f"LIBERO-Plus classification: {catalog.classification_path}")
    print(f"Selected tasks: {len(tasks)}")
    print(f"Output directory: {output_dir}")
    if bool(cfg.MULTIRUN.get("create_only", False)):
        print("create_only=true: validated installation and wrote task manifests.")
        return

    pending = _pending_tasks(output_dir, tasks)
    print(f"Completed tasks: {len(tasks) - len(pending)}; pending tasks: {len(pending)}")
    if not pending:
        _summarize(output_dir)
        return

    try:
        _run_workers(
            cfg,
            output_dir=output_dir,
            task_choice=task_choice,
            checkpoint=checkpoint,
            dataset_stats=dataset_stats,
            pending=pending,
        )
    except BaseException:
        _summarize(output_dir)
        raise

    _summarize(output_dir)
    remaining = _pending_tasks(output_dir, tasks)
    if remaining:
        raise RuntimeError(
            f"Workers exited successfully but {len(remaining)} task results are missing. "
            f"Resume from {output_dir} after inspecting worker logs."
        )


if __name__ == "__main__":
    main()
