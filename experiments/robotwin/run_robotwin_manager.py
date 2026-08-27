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

from experiments.robotwin.result_utils import parse_success_rate, task_is_complete  # noqa: E402

WORKER_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "eval_robotwin_worker.py"
EVAL_STEP_LIMIT_FILE = PROJECT_ROOT / "third_party" / "RoboTwin" / "task_config" / "_eval_step_limit.yml"


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


def _load_all_tasks() -> list[str]:
    payload = yaml.safe_load(EVAL_STEP_LIMIT_FILE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Invalid task map: {EVAL_STEP_LIMIT_FILE}")
    return list(dict.fromkeys(str(key) for key in payload))


def _build_worker_slots(num_gpus: int, max_tasks_per_gpu: int) -> list[tuple[int, int]]:
    if num_gpus <= 0 or max_tasks_per_gpu <= 0:
        raise ValueError("num_gpus and max_tasks_per_gpu must both be positive.")
    return [(gpu, slot) for gpu in range(num_gpus) for slot in range(max_tasks_per_gpu)]


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


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("ckpt must not be None.")
    checkpoint = _resolve_path(str(cfg.ckpt))
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    configured_task = cfg.EVALUATION.task_name
    tasks = _load_all_tasks() if configured_task is None or not str(configured_task).strip() else [str(configured_task)]
    slots = _build_worker_slots(int(cfg.MULTIRUN.num_gpus), int(cfg.MULTIRUN.max_tasks_per_gpu))

    raw_output = _resolve_path(str(cfg.EVALUATION.output_dir))
    output_dir = PROJECT_ROOT / "evaluate_results" / "robotwin" / _resolve_ckpt_tag(checkpoint) / raw_output.name
    pending_tasks = [task for task in tasks if not task_is_complete(output_dir, task)]
    print(f"Completed tasks: {len(tasks) - len(pending_tasks)}; pending tasks: {len(pending_tasks)}")
    if not pending_tasks:
        _write_summary(output_dir, tasks)
        print(f"RoboTwin evaluation already complete: {output_dir}")
        return

    worker_count = min(len(pending_tasks), len(slots))
    shards = [[] for _ in range(worker_count)]
    for index, task in enumerate(pending_tasks):
        shards[index % worker_count].append(task)

    worker_dir = output_dir / "workers"
    log_dir = output_dir / "logs"
    worker_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "manager_config.yaml")
    task_choice = HydraConfig.get().runtime.choices.get("task")
    extra = [value for value in HydraConfig.get().overrides.task if not _is_blocked_override(value)]
    processes: list[subprocess.Popen] = []
    handles = []
    try:
        for worker_index, ((gpu_id, slot), shard) in enumerate(zip(slots, shards)):
            shard_path = worker_dir / f"worker_{worker_index:03d}.json"
            shard_path.write_text(json.dumps(shard), encoding="utf-8")
            log_path = log_dir / f"worker_{worker_index:03d}_gpu_{gpu_id}_slot_{slot}.log"
            handle = log_path.open("a", encoding="utf-8")
            handles.append(handle)
            command = [
                sys.executable, str(WORKER_ENTRY), f"task={task_choice}", f"ckpt={checkpoint}",
                f"gpu_id={gpu_id}", f"WORKER.task_file={shard_path}",
                f"EVALUATION.output_dir={output_dir}",
                f"WORKER.worker_index={worker_index}", *extra,
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env.setdefault("PYTHONFAULTHANDLER", "1")
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
            processes.append(subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT))
            print(f"Started worker {worker_index}: gpu={gpu_id} slot={slot}, tasks={len(shard)}")
        while not all(process.poll() == 0 for process in processes):
            failed = next((p for p in processes if p.poll() not in (None, 0)), None)
            if failed is not None:
                code = failed.returncode
                _terminate(processes)
                raise RuntimeError(f"RoboTwin worker failed with return code {code}; inspect {log_dir}.")
            time.sleep(2)
    finally:
        _terminate(processes)
        for handle in handles:
            handle.close()
    _write_summary(output_dir, tasks)
    print(f"RoboTwin evaluation complete: {output_dir}")


if __name__ == "__main__":
    main()
