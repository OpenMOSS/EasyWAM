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
    select_mujoco_render_backend,
)
from experiments.libero.result_utils import valid_result_path  # noqa: E402


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


def build_worker_slots(num_gpus: int, max_tasks_per_gpu: int) -> list[tuple[int, int]]:
    if num_gpus <= 0 or max_tasks_per_gpu <= 0:
        raise ValueError("num_gpus and max_tasks_per_gpu must both be positive.")
    return [(gpu_id, slot) for gpu_id in range(num_gpus) for slot in range(max_tasks_per_gpu)]


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
    max_tasks_per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    slots = build_worker_slots(num_gpus, max_tasks_per_gpu)
    render_backend = select_mujoco_render_backend(max_tasks_per_gpu)
    print(
        f"MuJoCo rendering backend: {render_backend} "
        f"(num_gpus={num_gpus}, max_tasks_per_gpu={max_tasks_per_gpu})"
    )
    worker_count = min(len(tasks), len(slots))
    shards = [[] for _ in range(worker_count)]
    for index, task in enumerate(tasks):
        shards[index % worker_count].append(task)

    worker_dir = output_dir / "workers"
    log_dir = output_dir / "logs"
    worker_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    extra = [value for value in HydraConfig.get().overrides.task if not _is_blocked_override(value)]
    processes: list[subprocess.Popen] = []
    handles = []
    try:
        for worker_index, ((gpu_id, slot), shard) in enumerate(zip(slots, shards)):
            shard_path = worker_dir / f"worker_{worker_index:03d}.txt"
            shard_path.write_text("\n".join(shard) + "\n", encoding="utf-8")
            handle = (log_dir / f"worker_{worker_index:03d}_gpu_{gpu_id}_slot_{slot}.log").open(
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
                f"WORKER.task_file={shard_path}",
                f"WORKER.worker_index={worker_index}",
                *extra,
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            configure_mujoco_worker_env(env, max_tasks_per_gpu)
            env.setdefault("PYTHONFAULTHANDLER", "1")
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
            processes.append(subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT))
            print(f"Started worker {worker_index}: gpu={gpu_id} slot={slot}, tasks={len(shard)}")
        while not all(process.poll() == 0 for process in processes):
            failed_index = next(
                (index for index, process in enumerate(processes) if process.poll() not in (None, 0)),
                None,
            )
            if failed_index is not None:
                code = int(processes[failed_index].returncode)
                gpu_id, slot = slots[failed_index]
                _terminate(processes)
                raise RuntimeError(
                    f"LIBERO worker {failed_index} on GPU {gpu_id} slot {slot} failed "
                    f"with return code {_format_return_code(code)}; inspect {log_dir}."
                )
            time.sleep(2)
    finally:
        _terminate(processes)
        for handle in handles:
            handle.close()

    from experiments.libero.summarize_results import summarize_results

    summarize_results(str(output_dir))


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("ckpt must not be None.")
    output_dir = Path(os.path.expanduser(os.path.expandvars(str(cfg.EVALUATION.output_dir)))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    task_file_cfg = cfg.MULTIRUN.get("task_file")
    task_file = Path(str(task_file_cfg)).expanduser().resolve() if task_file_cfg else output_dir / "tasks.txt"
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
