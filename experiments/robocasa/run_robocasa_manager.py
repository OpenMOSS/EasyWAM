"""Multi-GPU manager for the official RoboCasa365 50-task benchmark."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER_ENTRY = PROJECT_ROOT / "experiments" / "robocasa" / "eval_robocasa_worker.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.render_backend import (  # noqa: E402
    configure_mujoco_worker_env,
    select_mujoco_render_backend,
)
from experiments.robocasa.registry import load_official_jobs  # noqa: E402
from experiments.robocasa.result_utils import valid_result_path  # noqa: E402


def create_task_file(path: Path, jobs: list[tuple[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(jobs)
    print(f"Task list created: {path} ({len(jobs)} tasks)")
    return path


def _is_blocked_override(raw: str) -> bool:
    key = raw.split("=", 1)[0].lstrip("+~")
    return key in {
        "task",
        "ckpt",
        "gpu_id",
        "EVALUATION.output_dir",
        "EVALUATION.task_set",
        "EVALUATION.task_name",
    } or key.startswith(("MULTIRUN.", "WORKER.", "hydra."))


def _terminate(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def _format_return_code(return_code: int) -> str:
    if return_code < 0:
        try:
            import signal

            return f"{return_code} ({signal.Signals(-return_code).name})"
        except (ValueError, OSError):
            pass
    return str(return_code)


def _write_environment_metadata(path: Path, robocasa_root: Path) -> None:
    try:
        version = importlib.metadata.version("robocasa")
    except importlib.metadata.PackageNotFoundError:
        version = None
    commit = None
    if (robocasa_root / ".git").exists():
        completed = subprocess.run(
            ["git", "-C", str(robocasa_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            commit = completed.stdout.strip()
    path.write_text(
        json.dumps(
            {
                "robocasa_root": str(robocasa_root),
                "robocasa_version": version,
                "robocasa_git_commit": commit,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def run_evaluation(
    cfg: DictConfig,
    jobs: list[tuple[str, str]],
    task_choice: str,
    output_dir: Path,
) -> None:
    split = str(cfg.EVALUATION.split)
    expected_episodes = int(cfg.EVALUATION.num_trials)
    pending = [
        f"{task_set},{task_name}"
        for task_set, task_name in jobs
        if valid_result_path(
            output_dir, task_set, task_name, split, expected_episodes
        )
        is None
    ]
    print(f"Completed tasks: {len(jobs) - len(pending)}; pending tasks: {len(pending)}")
    if not pending:
        from experiments.robocasa.summarize_results import summarize_results

        summarize_results(
            output_dir, jobs, split=split, expected_episodes=expected_episodes
        )
        return

    num_gpus = int(cfg.MULTIRUN.num_gpus)
    envs_per_gpu = int(cfg.MULTIRUN.env_num_per_gpu)
    batch_size = int(cfg.MULTIRUN.inference_batch_size)
    if num_gpus <= 0 or envs_per_gpu <= 0:
        raise ValueError("MULTIRUN.num_gpus and env_num_per_gpu must be positive.")
    if batch_size <= 0:
        raise ValueError("MULTIRUN.inference_batch_size must be positive.")
    if batch_size > envs_per_gpu:
        raise ValueError("inference_batch_size cannot exceed env_num_per_gpu.")
    render_backend = select_mujoco_render_backend(envs_per_gpu)
    print(
        f"MuJoCo rendering backend: {render_backend} "
        f"(num_gpus={num_gpus}, envs_per_gpu={envs_per_gpu})"
    )
    worker_count = min(len(pending), num_gpus)
    worker_dir = output_dir / "workers"
    log_dir = output_dir / "logs"
    worker_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    shared_task_path = worker_dir / "pending_tasks.csv"
    shared_task_path.write_text("\n".join(pending) + "\n", encoding="utf-8")
    cursor_path = worker_dir / "task_cursor.txt"
    cursor_path.write_text("0", encoding="utf-8")
    extra = [
        value
        for value in HydraConfig.get().overrides.task
        if not _is_blocked_override(value)
    ]

    processes: list[subprocess.Popen] = []
    handles = []
    try:
        for worker_index in range(worker_count):
            gpu_id = worker_index
            handle = (log_dir / f"worker_{worker_index:03d}_gpu_{gpu_id}.log").open(
                "a", encoding="utf-8"
            )
            handles.append(handle)
            command = [
                sys.executable,
                str(WORKER_ENTRY),
                f"task={task_choice}",
                f"ckpt={cfg.ckpt}",
                f"gpu_id={gpu_id}",
                f"EVALUATION.output_dir={output_dir}",
                f"WORKER.task_file={shared_task_path}",
                f"WORKER.task_cursor={cursor_path}",
                f"WORKER.worker_index={worker_index}",
                f"MULTIRUN.env_num_per_gpu={envs_per_gpu}",
                f"MULTIRUN.inference_batch_size={batch_size}",
                f"MULTIRUN.inference_batch_wait_ms={float(cfg.MULTIRUN.inference_batch_wait_ms)}",
                f"MULTIRUN.prompt_cache_size={int(cfg.MULTIRUN.prompt_cache_size)}",
                *extra,
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            configure_mujoco_worker_env(env, envs_per_gpu)
            env.setdefault("PYTHONFAULTHANDLER", "1")
            env.setdefault("PYTHONUNBUFFERED", "1")
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            )
            print(f"Started model worker {worker_index}: gpu={gpu_id}, envs={envs_per_gpu}")
        while not all(process.poll() == 0 for process in processes):
            failed = next(
                (
                    index
                    for index, process in enumerate(processes)
                    if process.poll() not in (None, 0)
                ),
                None,
            )
            if failed is not None:
                code = int(processes[failed].returncode)
                _terminate(processes)
                raise RuntimeError(
                    f"RoboCasa worker {failed} failed with return code "
                    f"{_format_return_code(code)}; inspect {log_dir}."
                )
            time.sleep(2)
    finally:
        _terminate(processes)
        for handle in handles:
            handle.close()

    from experiments.robocasa.summarize_results import summarize_results

    summarize_results(
        output_dir, jobs, split=split, expected_episodes=expected_episodes
    )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robocasa.yaml")
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None and not bool(cfg.MULTIRUN.get("create_only", False)):
        raise ValueError("ckpt must not be None.")
    output_dir = Path(
        os.path.expandvars(os.path.expanduser(str(cfg.EVALUATION.output_dir)))
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    robocasa_root = Path(str(cfg.EVALUATION.robocasa_root)).expanduser().resolve()
    task_sets = [str(value) for value in cfg.MULTIRUN.task_sets]
    jobs, _ = load_official_jobs(task_sets, robocasa_root)
    requested_task = cfg.EVALUATION.get("task_name")
    if requested_task:
        jobs = [job for job in jobs if job[1] == str(requested_task)]
        if not jobs:
            raise ValueError(
                f"EVALUATION.task_name={requested_task!r} is not in {task_sets}."
            )
    task_file_value = cfg.MULTIRUN.get("task_file")
    task_file = (
        Path(str(task_file_value)).expanduser().resolve()
        if task_file_value
        else output_dir / "tasks.csv"
    )
    create_task_file(task_file, jobs)
    OmegaConf.save(cfg, output_dir / "manager_config.yaml")
    _write_environment_metadata(
        output_dir / "robocasa_environment.json", robocasa_root
    )
    if bool(cfg.MULTIRUN.get("create_only", False)):
        return
    task_choice = HydraConfig.get().runtime.choices.get("task")
    if not task_choice:
        raise ValueError("Hydra task choice is empty.")
    run_evaluation(cfg, jobs, str(task_choice), output_dir)


if __name__ == "__main__":
    main()
