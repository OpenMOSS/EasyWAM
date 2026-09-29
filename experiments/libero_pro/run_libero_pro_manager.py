"""Multi-GPU manager for the official LIBERO-Pro five-dimension benchmark."""

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
WORKER_ENTRY = PROJECT_ROOT / "experiments" / "libero_pro" / "eval_libero_pro_worker.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.render_backend import configure_mujoco_worker_env  # noqa: E402
from experiments.libero_pro.libero_pro_utils import (  # noqa: E402
    TaskSpec,
    is_valid_result,
    load_catalog,
    prepare_runtime_config,
    resolve_path,
    result_path,
    select_tasks,
    write_jsonl,
)
from experiments.task_dispatch import (  # noqa: E402
    build_worker_slots,
    count_workers_by_gpu,
    resolve_gpu_ids,
)


def _string_override(key: str, value: Any) -> str:
    return f"{key}={json.dumps(str(value))}"


def _optional_list(value) -> Optional[list[Any]]:
    return None if value is None else list(value)


def _resolve_task_choice() -> str:
    choice = HydraConfig.get().runtime.choices.get("task")
    if choice is None or not str(choice).strip():
        raise ValueError(
            "Hydra task choice is empty; pass task=libero_easywam_mot_wan22 "
            "or another LIBERO task."
        )
    return str(choice)


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    blocked = {
        "task",
        "ckpt",
        "gpu_id",
        "EVALUATION.output_dir",
        "EVALUATION.dataset_stats_path",
        "EVALUATION.libero_pro_root",
        "EVALUATION.libero_pro_data_root",
        "EVALUATION.libero_pro_config_path",
        "EVALUATION.task_suite_name",
        "EVALUATION.task_id",
        "EVALUATION.max_steps",
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
        return resolve_path(str(explicit))
    if cfg.ckpt is None:
        return None
    checkpoint = resolve_path(str(cfg.ckpt))
    for parent in list(checkpoint.parents)[:4]:
        candidate = parent / "dataset_stats.json"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _pending_tasks(output_dir: Path, tasks: list[TaskSpec]) -> list[TaskSpec]:
    return [task for task in tasks if not is_valid_result(result_path(output_dir, task), task)]


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


def _summarize(output_dir: Path) -> None:
    from experiments.libero_pro.summarize_libero_pro import summarize_results

    summarize_results(output_dir)


def _run_workers(
    cfg: DictConfig,
    *,
    output_dir: Path,
    source_root: Path,
    data_root: Path,
    runtime_config: Path,
    task_choice: str,
    checkpoint: Path,
    dataset_stats: Path,
    pending: list[TaskSpec],
) -> None:
    num_gpus = int(cfg.MULTIRUN.get("num_gpus", 1))
    gpu_ids = resolve_gpu_ids(num_gpus=num_gpus, gpu_ids=cfg.MULTIRUN.get("gpu_ids"))
    workers_per_gpu = int(cfg.MULTIRUN.workers_per_gpu)
    env_num_per_worker = int(cfg.MULTIRUN.env_num_per_worker)
    batch_size = int(cfg.MULTIRUN.inference_batch_size)
    if env_num_per_worker <= 0:
        raise ValueError("env_num_per_worker must be positive.")
    if batch_size <= 0 or batch_size > env_num_per_worker:
        raise ValueError(
            "inference_batch_size must be positive and cannot exceed env_num_per_worker."
        )
    slots = build_worker_slots(
        num_gpus=num_gpus,
        gpu_ids=gpu_ids,
        workers_per_gpu=workers_per_gpu,
        pending_jobs=len(pending),
    )
    active_workers = count_workers_by_gpu(slots)
    print(
        f"Model workers: {len(slots)} (gpu_ids={gpu_ids}, "
        f"workers_per_gpu={workers_per_gpu}, env_num_per_worker={env_num_per_worker})"
    )

    worker_dir = output_dir / "workers"
    log_dir = output_dir / "logs"
    cache_dir = output_dir / "runtime" / "cache"
    worker_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    shared_task_path = worker_dir / "pending_tasks.jsonl"
    write_jsonl(shared_task_path, (task.to_dict() for task in pending))
    cursor_path = worker_dir / "task_cursor.txt"
    cursor_path.write_text("0", encoding="utf-8")
    extra_overrides = _collect_worker_overrides()
    processes: list[subprocess.Popen] = []
    log_handles = []
    try:
        for slot in slots:
            worker_index = slot.worker_index
            gpu_id = slot.gpu_id
            log_path = log_dir / f"worker_{worker_index:03d}_gpu_{gpu_id}.log"
            log_handle = log_path.open("a", encoding="utf-8")
            log_handles.append(log_handle)
            command = [
                sys.executable,
                str(WORKER_ENTRY),
                f"task={task_choice}",
                _string_override("ckpt", checkpoint),
                _string_override("EVALUATION.output_dir", output_dir),
                _string_override("EVALUATION.dataset_stats_path", dataset_stats),
                _string_override("EVALUATION.libero_pro_root", source_root),
                _string_override("EVALUATION.libero_pro_data_root", data_root),
                _string_override("EVALUATION.libero_pro_config_path", runtime_config),
                f"gpu_id={gpu_id}",
                _string_override("WORKER.task_file", shared_task_path),
                _string_override("WORKER.task_cursor", cursor_path),
                f"WORKER.worker_index={worker_index}",
                f"MULTIRUN.workers_per_gpu={workers_per_gpu}",
                f"MULTIRUN.env_num_per_worker={env_num_per_worker}",
                f"MULTIRUN.inference_batch_size={batch_size}",
                f"MULTIRUN.inference_batch_wait_ms={float(cfg.MULTIRUN.inference_batch_wait_ms)}",
                f"MULTIRUN.prompt_cache_size={int(cfg.MULTIRUN.prompt_cache_size)}",
                *extra_overrides,
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            current_pythonpath = env.get("PYTHONPATH")
            env["PYTHONPATH"] = (
                str(source_root)
                if not current_pythonpath
                else os.pathsep.join((str(source_root), current_pythonpath))
            )
            env["LIBERO_CONFIG_PATH"] = str(runtime_config.parent)
            worker_cache_dir = cache_dir / f"worker_{worker_index:03d}"
            numba_cache_dir = worker_cache_dir / "numba"
            matplotlib_config_dir = worker_cache_dir / "matplotlib"
            numba_cache_dir.mkdir(parents=True, exist_ok=True)
            matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
            env["NUMBA_CACHE_DIR"] = str(numba_cache_dir)
            env["MPLCONFIGDIR"] = str(matplotlib_config_dir)
            concurrent_envs = active_workers[gpu_id] * env_num_per_worker
            render_backend = configure_mujoco_worker_env(env, concurrent_envs)
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
                f"Started model worker {worker_index} on GPU {gpu_id}: "
                f"gpu_worker={slot.gpu_worker_index}, envs={env_num_per_worker}, "
                f"render={render_backend}, log={log_path}"
            )

        while processes:
            failed_index = next(
                (
                    index
                    for index, process in enumerate(processes)
                    if process.poll() not in (None, 0)
                ),
                None,
            )
            if failed_index is not None:
                return_code = int(processes[failed_index].returncode)
                slot = slots[failed_index]
                _terminate_processes(processes)
                raise RuntimeError(
                    f"LIBERO-Pro worker {slot.worker_index} on GPU {slot.gpu_id} "
                    f"failed with return code {return_code}; inspect {log_dir} and "
                    f"{output_dir / 'errors'}."
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


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="benchmark/sim_libero_pro.yaml",
)
def main(cfg: DictConfig) -> None:
    output_dir = resolve_path(str(cfg.EVALUATION.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    source_root = resolve_path(str(cfg.EVALUATION.libero_pro_root))
    data_root = resolve_path(str(cfg.EVALUATION.libero_pro_data_root))
    runtime_config = prepare_runtime_config(
        source_root,
        data_root,
        output_dir / "runtime" / "libero_config",
    )
    catalog = load_catalog(
        source_root=source_root,
        data_root=data_root,
        config_path=runtime_config,
        base_suites=[str(value) for value in cfg.MULTIRUN.task_suite_names],
        perturbations=[str(value) for value in cfg.MULTIRUN.perturbations],
        num_trials=int(cfg.EVALUATION.num_trials),
    )
    tasks = select_tasks(
        catalog.tasks_by_suite,
        task_ids=_optional_list(cfg.MULTIRUN.get("task_ids")),
    )
    checkpoint = None if cfg.ckpt is None else resolve_path(str(cfg.ckpt))
    dataset_stats = _resolve_dataset_stats_path(cfg)
    task_choice = _resolve_task_choice()

    if not bool(cfg.MULTIRUN.get("create_only", False)):
        if checkpoint is None or not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        if dataset_stats is None or not dataset_stats.is_file():
            raise FileNotFoundError(
                "dataset_stats.json was not found. Pass EVALUATION.dataset_stats_path explicitly."
            )

    write_jsonl(output_dir / "tasks.jsonl", (task.to_dict() for task in tasks))
    OmegaConf.save(config=cfg, f=str(output_dir / "manager_config.yaml"))
    (output_dir / "upstream.json").write_text(
        json.dumps(
            {
                "source_root": str(catalog.source_root),
                "data_root": str(catalog.data_root),
                "git_commit": catalog.source_commit,
                "benchmark_module": str(Path(catalog.benchmark.__file__).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"LIBERO-Pro source: {source_root} ({catalog.source_commit or 'unknown commit'})")
    print(f"LIBERO-Pro data: {data_root}")
    print(f"Selected tasks: {len(tasks)} ({len(tasks) * int(cfg.EVALUATION.num_trials)} episodes)")
    print(f"Output directory: {output_dir}")
    if bool(cfg.MULTIRUN.get("create_only", False)):
        print("create_only=true: validated source/data and wrote task manifests.")
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
            source_root=source_root,
            data_root=data_root,
            runtime_config=runtime_config,
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
