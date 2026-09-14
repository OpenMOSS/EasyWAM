"""RoboTwin manager using persistent model-server workers."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robotwin.result_utils import (  # noqa: E402
    parse_success_rate,
    task_is_complete,
)
from experiments.robotwin.upstream import validate_robotwin_root  # noqa: E402
from experiments.task_dispatch import build_worker_slots, resolve_gpu_ids  # noqa: E402

WORKER_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "eval_robotwin_worker.py"


def _resolve_path(value: str, base: Path = PROJECT_ROOT) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(value)))
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _resolve_ckpt_tag(checkpoint: Path) -> str:
    parts = checkpoint.parts
    if "runs" in parts:
        index = parts.index("runs")
        if index + 2 < len(parts):
            return f"{parts[index + 1]}_{parts[index + 2]}"
    return checkpoint.stem


def _load_all_tasks(robotwin_root: Path) -> list[str]:
    task_file = robotwin_root / "env_cfg" / "task_config" / "_eval_step_limit.yml"
    payload = yaml.safe_load(task_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Invalid task map: {task_file}")
    return list(dict.fromkeys(str(key) for key in payload))


def _is_blocked_override(raw: str) -> bool:
    key = raw.split("=", 1)[0].lstrip("+~")
    return key in {
        "ckpt",
        "gpu_id",
        "EVALUATION.output_dir",
        "EVALUATION.task_name",
        "EVALUATION.task_config",
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


def _write_summary(output_dir: Path, tasks: list[str]) -> None:
    rows = []
    for task in tasks:
        clean = parse_success_rate(output_dir / task / "_result_clean.txt")
        random = parse_success_rate(output_dir / task / "_result_random.txt")
        rows.append((task, clean, random))
    clean_mean = sum(row[1] for row in rows) / len(rows)
    random_mean = sum(row[2] for row in rows) / len(rows)
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_name", "clean_success_rate", "random_success_rate"])
        writer.writerows(rows)
        writer.writerow(["__overall__", clean_mean, random_mean])
    payload = {
        "per_task": [
            {"task_name": task, "clean_success_rate": clean, "random_success_rate": random}
            for task, clean, random in rows
        ],
        "overall": {
            "clean_mean_success_rate": clean_mean,
            "random_mean_success_rate": random_mean,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="benchmark/sim_robotwin.yaml",
)
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("ckpt must not be None.")
    checkpoint = _resolve_path(str(cfg.ckpt))
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    robotwin_root = validate_robotwin_root(
        _resolve_path(str(cfg.EVALUATION.robotwin_root))
    )
    configured_task = cfg.EVALUATION.task_name
    tasks = (
        _load_all_tasks(robotwin_root)
        if configured_task is None or not str(configured_task).strip()
        else [str(configured_task)]
    )
    raw_output = _resolve_path(str(cfg.EVALUATION.output_dir))
    output_dir = (
        PROJECT_ROOT
        / "evaluate_results"
        / "robotwin"
        / _resolve_ckpt_tag(checkpoint)
        / raw_output.name
    )
    pending_tasks = [task for task in tasks if not task_is_complete(output_dir, task)]
    print(
        f"Completed tasks: {len(tasks) - len(pending_tasks)}; "
        f"pending tasks: {len(pending_tasks)}"
    )
    if not pending_tasks:
        _write_summary(output_dir, tasks)
        print(f"RoboTwin evaluation already complete: {output_dir}")
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
        pending_jobs=len(pending_tasks),
    )
    print(
        f"Model workers: {len(slots)} "
        f"(gpu_ids={gpu_ids}, workers_per_gpu={workers_per_gpu}, "
        f"env_num_per_worker={env_num_per_worker})"
    )

    worker_dir = output_dir / "workers"
    log_dir = output_dir / "logs"
    worker_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "manager_config.yaml")
    task_choice = HydraConfig.get().runtime.choices.get("task")
    extra = [
        value
        for value in HydraConfig.get().overrides.task
        if not _is_blocked_override(value)
    ]
    processes: list[subprocess.Popen] = []
    handles = []
    try:
        jobs = [{"task_name": task} for task in pending_tasks]
        task_path = worker_dir / "pending_jobs.jsonl"
        task_path.write_text(
            "".join(json.dumps(job) + "\n" for job in jobs), encoding="utf-8"
        )
        cursor_path = worker_dir / "task_cursor.txt"
        cursor_path.write_text("0", encoding="utf-8")
        for slot in slots:
            worker_index = slot.worker_index
            gpu_id = slot.gpu_id
            log_path = log_dir / f"worker_{worker_index:03d}_gpu_{gpu_id}.log"
            handle = log_path.open("a", encoding="utf-8")
            handles.append(handle)
            command = [
                sys.executable,
                str(WORKER_ENTRY),
                f"task={task_choice}",
                f"ckpt={checkpoint}",
                f"gpu_id={gpu_id}",
                f"WORKER.task_file={task_path}",
                f"WORKER.task_cursor={cursor_path}",
                f"EVALUATION.output_dir={output_dir}",
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
                f"Started model worker {worker_index}: "
                f"gpu={gpu_id}, gpu_worker={slot.gpu_worker_index}, "
                f"envs={env_num_per_worker}"
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
                code = processes[failed_index].returncode
                slot = slots[failed_index]
                _terminate(processes)
                raise RuntimeError(
                    f"RoboTwin worker {slot.worker_index} on GPU {slot.gpu_id} "
                    f"(gpu_worker={slot.gpu_worker_index}) failed with return "
                    f"code {code}; "
                    f"inspect {log_dir}."
                )
            time.sleep(2)
    finally:
        _terminate(processes)
        for handle in handles:
            handle.close()
    _write_summary(output_dir, tasks)
    print(f"RoboTwin evaluation complete: {output_dir}")


if __name__ == "__main__":
    main()
