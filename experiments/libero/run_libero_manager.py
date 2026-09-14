"""Multi-GPU manager using persistent standard-LIBERO model workers."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER_ENTRY = PROJECT_ROOT / "experiments" / "libero" / "eval_libero_worker.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.render_backend import (  # noqa: E402
    configure_mujoco_worker_env,
)
from experiments.libero.result_utils import valid_result_path  # noqa: E402
from experiments.task_dispatch import (  # noqa: E402
    build_worker_slots,
    count_workers_by_gpu,
    resolve_gpu_ids,
)


def create_task_file(output_file: Path, task_suite_names: list[str]) -> Path:
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output_file.open("w", encoding="utf-8") as f:
        for suite_name in task_suite_names:
            suite = benchmark_dict[suite_name]()
            for task_id in range(int(suite.n_tasks)):
                f.write(f"{suite_name},{task_id}\n")
                total += 1
    print(f"Task list created: {output_file} ({total} tasks)")
    return output_file


def _read_task_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _is_blocked_override(raw: str) -> bool:
    key = raw.split("=", 1)[0].lstrip("+~")
    return key in {
        "task",
        "ckpt",
        "gpu_id",
        "EVALUATION.output_dir",
        "EVALUATION.task_suite_name",
        "EVALUATION.task_id",
    } or key.startswith(
        ("MULTIRUN.", "WORKER.", "hydra.")
    )


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


def run_evaluation(cfg: DictConfig, task_file: Path, task_choice: str, output_dir: Path) -> None:
    all_tasks = _read_task_lines(task_file)
    expected_episodes = int(cfg.EVALUATION.num_trials)
    tasks = []
    for task in all_tasks:
        suite_name, raw_task_id = task.split(",", 1)
        if valid_result_path(
            output_dir, suite_name, int(raw_task_id), expected_episodes
        ) is None:
            tasks.append(task)
    print(f"Completed tasks: {len(all_tasks) - len(tasks)}; pending tasks: {len(tasks)}")
    if not tasks:
        from experiments.libero.summarize_results import summarize_results

        summarize_results(str(output_dir))
        return
    num_gpus = int(cfg.MULTIRUN.num_gpus)
    gpu_ids = resolve_gpu_ids(
        num_gpus=num_gpus,
        gpu_ids=cfg.MULTIRUN.get("gpu_ids"),
    )
    workers_per_gpu = int(cfg.MULTIRUN.workers_per_gpu)
    env_num_per_worker = int(cfg.MULTIRUN.env_num_per_worker)
    batch_size = int(cfg.MULTIRUN.inference_batch_size)
    if env_num_per_worker <= 0:
        raise ValueError("env_num_per_worker must be positive.")
    if batch_size <= 0 or batch_size > env_num_per_worker:
        raise ValueError(
            "inference_batch_size must be positive and cannot exceed "
            "env_num_per_worker."
        )
    slots = build_worker_slots(
        num_gpus=num_gpus,
        gpu_ids=gpu_ids,
        workers_per_gpu=workers_per_gpu,
        pending_jobs=len(tasks),
    )
    active_workers = count_workers_by_gpu(slots)
    print(
        f"Model workers: {len(slots)} "
        f"(gpu_ids={gpu_ids}, workers_per_gpu={workers_per_gpu}, "
        f"env_num_per_worker={env_num_per_worker})"
    )

    worker_dir = output_dir / "workers"
    log_dir = output_dir / "logs"
    worker_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    extra = [value for value in HydraConfig.get().overrides.task if not _is_blocked_override(value)]
    processes: list[subprocess.Popen] = []
    handles = []
    try:
        shared_task_path = worker_dir / "pending_tasks.txt"
        shared_task_path.write_text("\n".join(tasks) + "\n", encoding="utf-8")
        cursor_path = worker_dir / "task_cursor.txt"
        cursor_path.write_text("0", encoding="utf-8")
        for slot in slots:
            worker_index = slot.worker_index
            gpu_id = slot.gpu_id
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
                f"MULTIRUN.workers_per_gpu={workers_per_gpu}",
                f"MULTIRUN.env_num_per_worker={env_num_per_worker}",
                f"MULTIRUN.inference_batch_size={batch_size}",
                f"MULTIRUN.inference_batch_wait_ms={float(cfg.MULTIRUN.inference_batch_wait_ms)}",
                f"MULTIRUN.prompt_cache_size={int(cfg.MULTIRUN.prompt_cache_size)}",
                *extra,
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
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
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            )
            print(
                f"Started model worker {worker_index}: gpu={gpu_id}, "
                f"gpu_worker={slot.gpu_worker_index}, "
                f"envs={env_num_per_worker}, render={render_backend}"
            )
        while not all(process.poll() == 0 for process in processes):
            failed_index = next(
                (
                    index
                    for index, process in enumerate(processes)
                    if process.poll() not in (None, 0)
                ),
                None,
            )
            if failed_index is not None:
                code = int(processes[failed_index].returncode)
                slot = slots[failed_index]
                _terminate(processes)
                raise RuntimeError(
                    f"LIBERO worker {slot.worker_index} on GPU {slot.gpu_id} "
                    f"(gpu_worker={slot.gpu_worker_index}) failed "
                    f"with return code {_format_return_code(code)}; inspect {log_dir}."
                )
            time.sleep(2)
    finally:
        _terminate(processes)
        for handle in handles:
            handle.close()

    from experiments.libero.summarize_results import summarize_results

    summarize_results(str(output_dir))


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="benchmark/sim_libero.yaml",
)
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("ckpt must not be None.")
    output_dir = Path(
        os.path.expanduser(os.path.expandvars(str(cfg.EVALUATION.output_dir)))
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    task_file_cfg = cfg.MULTIRUN.get("task_file")
    task_file = (
        Path(str(task_file_cfg)).expanduser().resolve()
        if task_file_cfg
        else output_dir / "tasks.txt"
    )
    create_task_file(task_file, [str(value) for value in cfg.MULTIRUN.task_suite_names])
    OmegaConf.save(cfg, output_dir / "manager_config.yaml")
    if bool(cfg.MULTIRUN.get("create_only", False)):
        return
    task_choice = HydraConfig.get().runtime.choices.get("task")
    if not task_choice:
        raise ValueError("Hydra task choice is empty.")
    run_evaluation(cfg, task_file, str(task_choice), output_dir)


if __name__ == "__main__":
    main()
