"""Multi-GPU, dynamically batched EasyWAM evaluation for RoboDojo."""

from __future__ import annotations

import csv
import json
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robodojo.result_utils import (  # noqa: E402
    expected_episodes,
    latest_valid_result,
    result_parent,
    summarize_jobs,
)
from experiments.robodojo.upstream import (  # noqa: E402
    build_client_command,
    install_policy_adapter,
    load_inventory,
    validate_robodojo_root,
)
from experiments.task_dispatch import (  # noqa: E402
    FileTaskDispatcher,
    build_worker_slots,
    resolve_gpu_ids,
)

SERVER_ENTRY = PROJECT_ROOT / "experiments/robodojo/policy_server.py"
POLICY_CONFIG = PROJECT_ROOT / "experiments/robodojo/easywam_policy/deploy_policy.yml"
SIM_CONFIG = PROJECT_ROOT / "configs/benchmark/sim_robodojo.yaml"


def _resolve_path(path_str: str, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(path_str)))
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _resolve_dataset_stats_path(cfg: DictConfig, checkpoint: Path) -> Path:
    value = cfg.EVALUATION.dataset_stats_path
    explicit = None
    if value is not None and str(value).strip().lower() not in {"", "none", "null"}:
        explicit = _resolve_path(str(value))
    candidates = ([explicit] if explicit is not None else []) + [
        (parent / "dataset_stats.json").resolve()
        for parent in list(checkpoint.parents)[:4]
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. Pass "
        "EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(str(value))


def _model_overrides(cfg: DictConfig, checkpoint: Path, dataset_stats: Path) -> list[str]:
    values = {
        "ckpt_setting": str(checkpoint),
        "seed": cfg.seed,
        "sim_cfg_path": str(SIM_CONFIG.resolve()),
        "sim_task": HydraConfig.get().runtime.choices.get("task"),
        "mixed_precision": cfg.mixed_precision,
        "device": cfg.EVALUATION.device,
        "dataset_stats_path": str(dataset_stats),
        "action_horizon": cfg.EVALUATION.action_horizon,
        "replan_steps": cfg.EVALUATION.replan_steps,
        "num_inference_steps": cfg.EVALUATION.num_inference_steps,
        "sigma_shift": cfg.EVALUATION.sigma_shift,
        "text_cfg_scale": cfg.EVALUATION.text_cfg_scale,
        "negative_prompt": cfg.EVALUATION.negative_prompt,
        "rand_device": cfg.EVALUATION.rand_device,
        "timing_enabled": cfg.EVALUATION.timing_enabled,
        "torch_compile": cfg.EVALUATION.torch_compile,
        "torch_compile_mode": cfg.EVALUATION.torch_compile_mode,
        "torch_compile_backend": cfg.EVALUATION.torch_compile_backend,
        "torch_compile_fullgraph": cfg.EVALUATION.torch_compile_fullgraph,
        "torch_compile_dynamic": cfg.EVALUATION.torch_compile_dynamic,
        "inference_batch_size": cfg.MULTIRUN.inference_batch_size,
        "inference_batch_wait_ms": cfg.MULTIRUN.inference_batch_wait_ms,
        "prompt_cache_size": cfg.MULTIRUN.prompt_cache_size,
    }
    overrides = []
    for key, value in values.items():
        if value is not None:
            overrides.extend([f"--{key}", _format_override_value(value)])
    return overrides


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(process: subprocess.Popen, port: int, timeout: float = 600) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Policy server exited with status {process.returncode}.")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"Timed out waiting for policy server on port {port}.")


def _checkpoint_tag(checkpoint: Path) -> str:
    parts = checkpoint.parts
    if "runs" in parts:
        index = parts.index("runs")
        if index + 2 < len(parts):
            return f"{parts[index + 1]}_{parts[index + 2]}"
    return checkpoint.stem


def _select_jobs(records: list[dict], task_name: str | None, seeds: list[int], episode_override: int | None) -> list[dict]:
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty list of distinct non-negative integers.")
    names = {str(record["name"]) for record in records}
    if task_name and task_name not in names:
        raise ValueError(f"Unknown RoboDojo task: {task_name}")
    selected = records
    if task_name:
        selected = [
            record for record in records
            if record["name"] == task_name
            or (record["name"] == f"{task_name}_random" and task_name + "_random" in names)
        ]
    jobs = []
    for record in selected:
        name = str(record["name"])
        dimension = str(record["dimension"])
        for seed in seeds:
            jobs.append({
                "task_name": name,
                "dimension": dimension,
                "seed": int(seed),
                "episodes": expected_episodes(name, episode_override, dimension),
            })
    return jobs


def create_task_file(path: Path, jobs: list[dict]) -> Path:
    """Write all selected task/seed jobs; resume filtering happens separately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(job) + "\n" for job in jobs), encoding="utf-8"
    )
    print(f"Task list created: {path} ({len(jobs)} jobs)")
    return path


def _write_summary(output_dir: Path, jobs: list[dict], root: Path, env_cfg_type: str, tag: str) -> list[dict]:
    rows = summarize_jobs(jobs, root, env_cfg_type, tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_task: dict[tuple[str, int], dict[str, dict]] = {}
    for row in rows:
        base = row["task_name"].removesuffix("_random")
        by_task.setdefault((base, row["seed"]), {})[
            "random" if row["task_name"].endswith("_random") else "standard"
        ] = row
    per_task = []
    for (task_name, seed), phases in sorted(by_task.items()):
        if "standard" not in phases and "random" in phases:
            entry = phases["random"]
            per_task.append({
                "task_name": entry["task_name"], "dimension": entry["dimension"],
                "seed": seed, "complete": entry["complete"],
                **({"success_rate": entry["success_rate"], "score": entry["score"]}
                   if entry["complete"] else {}),
            })
            continue
        required = ("standard", "random") if phases.get("standard", {}).get("dimension") == "generalization" else ("standard",)
        if not all(phases.get(name, {}).get("complete") for name in required):
            per_task.append({"task_name": task_name, "seed": seed, "complete": False})
            continue
        entries = [phases[name] for name in required]
        count = sum(entry["episodes"] for entry in entries)
        per_task.append({
            "task_name": task_name, "dimension": entries[0]["dimension"], "seed": seed,
            "complete": True,
            "success_rate": sum(entry["success_rate"] * entry["episodes"] for entry in entries) / count,
            "score": sum(entry["score"] * entry["episodes"] for entry in entries) / count,
        })
    dimension_scores = {}
    for dimension in sorted({row["dimension"] for row in per_task if row["complete"]}):
        values = [row for row in per_task if row["complete"] and row["dimension"] == dimension]
        by_seed_values: dict[int, list[dict]] = {}
        for row in values:
            by_seed_values.setdefault(row["seed"], []).append(row)
        dimension_scores[dimension] = {
            metric: sum(
                sum(row[metric] for row in seed_rows) / len(seed_rows)
                for seed_rows in by_seed_values.values()
            ) / len(by_seed_values)
            for metric in ("success_rate", "score")
        } | {
            "completed_cells": len(values),
        }
    complete_cells = [row for row in per_task if row["complete"]]
    seed_dimensions: dict[int, dict[str, list[dict]]] = {}
    for row in complete_cells:
        seed_dimensions.setdefault(row["seed"], {}).setdefault(row["dimension"], []).append(row)
    seed_overall = []
    for dimension_rows in seed_dimensions.values():
        seed_overall.append({
            metric: sum(
                sum(row[metric] for row in values) / len(values)
                for values in dimension_rows.values()
            ) / len(dimension_rows)
            for metric in ("success_rate", "score")
        })
    summary = {
        "jobs": rows,
        "per_task_seed": per_task,
        "dimensions": dimension_scores,
        "overall": {
            "completed_cells": len(complete_cells),
            "expected_cells": len(per_task),
            "complete": len(complete_cells) == len(per_task),
            "success_rate": (
                sum(row["success_rate"] for row in seed_overall) / len(seed_overall)
                if seed_overall else None
            ),
            "score": (
                sum(row["score"] for row in seed_overall) / len(seed_overall)
                if seed_overall else None
            ),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("task_name", "dimension", "seed", "episodes", "complete", "success_rate", "score", "result_path"))
        for row in rows:
            writer.writerow((
                row["task_name"], row["dimension"], row["seed"], row["episodes"],
                row["complete"], row.get("success_rate", ""), row.get("score", ""),
                row.get("result_path") or "",
            ))
    return rows


def _terminate(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _run_client_with_retries(
    command: list[str], root: Path, env: dict[str, str], log_path: Path,
    server: subprocess.Popen, stop: threading.Event,
    clients: list[subprocess.Popen], clients_lock: threading.Lock,
) -> int:
    """Keep RoboDojo's run ID stable across recoverable simulator restarts."""
    code = -1
    with log_path.open("w", encoding="utf-8") as log:
        for attempt in range(10):
            if stop.is_set() or server.poll() is not None:
                break
            process = subprocess.Popen(
                command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            with clients_lock:
                clients.append(process)
            try:
                while process.poll() is None:
                    if server.poll() is not None or stop.is_set():
                        process.terminate()
                    time.sleep(2)
                code = int(process.returncode)
            finally:
                with clients_lock:
                    clients.remove(process)
            if code not in (99, 134, 139) or stop.is_set() or server.poll() is not None:
                break
            print(f"RoboDojo simulator restart {attempt + 1}/10 (exit={code})", flush=True)
            time.sleep(5)
    return code


def _sim_launch(
    *, root: Path, task_name: str, seed: int, episodes: int, env_cfg_type: str,
    checkpoint_tag: str, physical_gpu: int, port: int, sim_env: str | None,
    sim_num_envs: int | None, run_id: str,
) -> tuple[list[str], dict[str, str]]:
    command = build_client_command(
        root=root, task_name=task_name, host="127.0.0.1", port=port,
        env_cfg_type=env_cfg_type, checkpoint_tag=checkpoint_tag,
        seed=seed, env_gpu=0, sim_env=sim_env, sim_num_envs=sim_num_envs,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(root), str(root / "XPolicyLab"), env.get("PYTHONPATH", ""))
    )
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    env["EVAL_NUM"] = str(episodes)
    env["ROBODOJO_RUN_ID"] = run_id
    env["PYTHONUNBUFFERED"] = "1"
    return command, env


def _write_official_summary(root: Path) -> None:
    summary_env = os.environ.copy()
    summary_env["ROBODOJO_EVAL_ROOT"] = str(root / "eval_result/RoboDojo")
    subprocess.run(
        [sys.executable, str(root / "scripts/internal/summarize_result.py")],
        cwd=root, env=summary_env, check=True,
    )
    print(f"Official summary: {root / 'eval_result/RoboDojo/_summary.md'}")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="benchmark/sim_robodojo.yaml")
def main(cfg: DictConfig) -> None:
    if str(cfg.EVALUATION.action_type) != "joint":
        raise ValueError("RoboDojo EasyWAM evaluation requires action_type=joint.")
    root = validate_robodojo_root(_resolve_path(str(cfg.EVALUATION.robodojo_root), base=PROJECT_ROOT))
    records = load_inventory(root)
    if cfg.EVALUATION.task_name is None:
        canonical = [record for record in records if record.get("variant") == "standard"]
        random_variants = [record for record in records if record.get("variant") == "random"]
        if len(canonical) != 42 or len(random_variants) != 12:
            raise ValueError(
                "Expected the official RoboDojo inventory of 42 tasks and "
                f"12 random variants; found {len(canonical)} and {len(random_variants)}."
            )
    override = cfg.EVALUATION.eval_num_episodes
    override = None if override is None else int(override)
    jobs = _select_jobs(records, cfg.EVALUATION.task_name, list(cfg.EVALUATION.seeds), override)
    if not jobs:
        raise ValueError("No RoboDojo jobs selected.")
    create_only = bool(cfg.MULTIRUN.get("create_only", False))
    checkpoint = None
    stats = None
    if not create_only:
        if cfg.ckpt is None:
            raise ValueError("ckpt must be set unless MULTIRUN.create_only=true.")
        checkpoint = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        stats = _resolve_dataset_stats_path(cfg, checkpoint)
    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "manager_config.yaml")
    task_file_value = cfg.MULTIRUN.get("task_file")
    selected_task_file = (
        _resolve_path(str(task_file_value), base=PROJECT_ROOT)
        if task_file_value else output_dir / "tasks.jsonl"
    )
    create_task_file(selected_task_file, jobs)
    if create_only:
        return
    assert checkpoint is not None and stats is not None
    install_policy_adapter(root)
    tag = _checkpoint_tag(checkpoint)
    pending = [
        job for job in jobs
        if latest_valid_result(
            result_parent(root, job["task_name"], str(cfg.EVALUATION.env_cfg_type), job["seed"], tag),
            job["episodes"],
        ) is None
    ]
    print(f"RoboDojo: {len(jobs) - len(pending)} complete; {len(pending)} pending", flush=True)
    if not pending:
        _write_summary(output_dir, jobs, root, str(cfg.EVALUATION.env_cfg_type), tag)
        if override is None and cfg.EVALUATION.task_name is None and list(cfg.EVALUATION.seeds) == [0, 1, 2]:
            _write_official_summary(root)
        return

    multirun = cfg.MULTIRUN
    num_gpus = int(multirun.num_gpus)
    candidate_ids = multirun.get("gpu_ids")
    policy_ids = resolve_gpu_ids(
        num_gpus=int(multirun.policy_num_gpus),
        gpu_ids=multirun.get("policy_gpu_ids") or candidate_ids,
    )
    env_ids = resolve_gpu_ids(
        num_gpus=int(multirun.env_num_gpus),
        gpu_ids=multirun.get("env_gpu_ids") or candidate_ids,
    )
    if num_gpus <= 0:
        raise ValueError("MULTIRUN.num_gpus must be positive.")
    env_slots = build_worker_slots(
        num_gpus=len(env_ids), gpu_ids=env_ids,
        workers_per_gpu=int(multirun.sim_workers_per_gpu), pending_jobs=len(pending),
    )
    policy_slots = build_worker_slots(
        num_gpus=len(policy_ids), gpu_ids=policy_ids,
        workers_per_gpu=int(multirun.workers_per_gpu), pending_jobs=len(env_slots),
    )
    if not policy_slots or not env_slots:
        raise ValueError("At least one policy worker and simulator worker are required.")
    if int(multirun.inference_batch_size) <= 0:
        raise ValueError("inference_batch_size must be positive.")
    log_dir = output_dir / "logs"
    worker_dir = output_dir / "workers"
    log_dir.mkdir(parents=True, exist_ok=True)
    worker_dir.mkdir(parents=True, exist_ok=True)
    task_file = worker_dir / "pending_jobs.jsonl"
    create_task_file(task_file, pending)
    cursor_file = worker_dir / "task_cursor.txt"
    cursor_file.write_text("0", encoding="utf-8")
    dispatcher = FileTaskDispatcher(task_file, cursor_file)
    claim_lock = threading.Lock()
    model_overrides = _model_overrides(cfg, checkpoint, stats)
    servers: list[subprocess.Popen] = []
    clients: list[subprocess.Popen] = []
    handles = []
    clients_lock = threading.Lock()
    stop = threading.Event()
    ports: list[int] = []
    try:
        for slot in policy_slots:
            port = _free_port()
            ports.append(port)
            log = (log_dir / f"policy_{slot.worker_index:03d}_gpu_{slot.gpu_id}.log").open("a", encoding="utf-8")
            handles.append(log)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(slot.gpu_id)
            env["PYTHONUNBUFFERED"] = "1"
            command = [
                sys.executable, "-u", str(SERVER_ENTRY), "--robodojo-root", str(root),
                "--config", str(POLICY_CONFIG), "--port", str(port),
                "--overrides", *model_overrides,
            ]
            server = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            servers.append(server)
            _wait_for_server(server, port)
            print(f"Policy worker {slot.worker_index}: GPU {slot.gpu_id}, port {port}", flush=True)

        def sim_worker(slot_index: int) -> None:
            slot = env_slots[slot_index]
            policy_index = slot_index % len(ports)
            port = ports[policy_index]
            server = servers[policy_index]
            while not stop.is_set():
                with claim_lock:
                    claimed = dispatcher.claim_with_index()
                if claimed is None:
                    return
                job_index, raw = claimed
                job = json.loads(raw)
                task_name = str(job["task_name"])
                seed = int(job["seed"])
                log_path = log_dir / f"sim_{slot_index:03d}_{task_name}_seed{seed}.log"
                command, env = _sim_launch(
                    root=root, task_name=task_name, seed=seed, episodes=int(job["episodes"]),
                    env_cfg_type=str(cfg.EVALUATION.env_cfg_type), checkpoint_tag=tag,
                    physical_gpu=slot.gpu_id, port=port,
                    sim_env=None if cfg.EVALUATION.sim_env is None else str(cfg.EVALUATION.sim_env),
                    sim_num_envs=cfg.EVALUATION.sim_num_envs,
                    run_id=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{os.getpid()}_{job_index}",
                )
                print(f"[{job_index + 1}/{len(pending)}] GPU {slot.gpu_id}: {task_name} seed={seed}", flush=True)
                code = _run_client_with_retries(
                    command, root, env, log_path, server, stop, clients, clients_lock
                )
                if code != 0:
                    stop.set()
                    raise RuntimeError(f"RoboDojo client failed ({code}); inspect {log_path}")
                parent = result_parent(root, task_name, str(cfg.EVALUATION.env_cfg_type), seed, tag)
                if latest_valid_result(parent, int(job["episodes"])) is None:
                    stop.set()
                    raise RuntimeError(f"RoboDojo client exited without a complete result; inspect {log_path}")

        with ThreadPoolExecutor(max_workers=len(env_slots)) as executor:
            futures = [executor.submit(sim_worker, index) for index in range(len(env_slots))]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    stop.set()
                    with clients_lock:
                        _terminate(list(clients))
                    raise
    finally:
        stop.set()
        with clients_lock:
            _terminate(list(clients))
        _terminate(servers)
        for handle in handles:
            handle.close()
        rows = _write_summary(output_dir, jobs, root, str(cfg.EVALUATION.env_cfg_type), tag)
        print(f"RoboDojo complete jobs: {sum(row['complete'] for row in rows)}/{len(rows)}", flush=True)
    if all(row["complete"] for row in rows) and override is None and cfg.EVALUATION.task_name is None and list(cfg.EVALUATION.seeds) == [0, 1, 2]:
        _write_official_summary(root)
    print(f"EasyWAM summary: {output_dir}")


if __name__ == "__main__":
    main()
