"""Persistent RoboTwin model-server worker."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robotwin.eval_robotwin_single import (
    POLICY_NAME,
    _append_override,
    _ensure_policy_symlink,
    _resolve_dataset_stats_path,
    _resolve_path,
)
from experiments.robotwin.result_utils import valid_phase_result
from experiments.task_dispatch import FileTaskDispatcher


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(process: subprocess.Popen, port: int, timeout: float = 600) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Model server exited with return code {process.returncode}.")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"Timed out waiting for model server on port {port}.")


def _common_overrides(cfg: DictConfig, checkpoint: Path, dataset_stats: Path) -> list[str]:
    overrides: list[str] = []
    sim_task = HydraConfig.get().runtime.choices.get("task")
    values = {
        "ckpt_setting": str(checkpoint),
        "seed": cfg.seed,
        "policy_name": cfg.EVALUATION.policy_name,
        "sim_cfg_path": str((PROJECT_ROOT / "configs" / "sim_robotwin.yaml").resolve()),
        "sim_task": sim_task,
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
        "inference_batch_size": cfg.MULTIRUN.inference_batch_size,
        "inference_batch_wait_ms": cfg.MULTIRUN.inference_batch_wait_ms,
    }
    for key, value in values.items():
        _append_override(overrides, key, value)
    return overrides


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig) -> None:
    task_file = cfg.WORKER.get("task_file")
    if not task_file:
        raise ValueError("WORKER.task_file is required.")
    task_path = Path(str(task_file)).expanduser().resolve()
    if not task_path.read_text(encoding="utf-8").strip():
        return
    cursor = cfg.WORKER.get("task_cursor")
    if not cursor:
        raise ValueError("WORKER.task_cursor is required.")
    dispatcher = FileTaskDispatcher(task_path, Path(str(cursor)).expanduser().resolve())

    checkpoint = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    policy_source = (PROJECT_ROOT / "experiments" / "robotwin" / POLICY_NAME).resolve()
    _ensure_policy_symlink(robotwin_root, policy_source)
    dataset_stats = _resolve_dataset_stats_path(cfg, checkpoint)
    output_dir = Path(str(cfg.EVALUATION.output_dir)).expanduser().resolve()
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    worker_index = int(cfg.WORKER.worker_index)
    port = _free_port()
    policy_config = f"policy/{POLICY_NAME}/deploy_policy.yml"
    common = _common_overrides(cfg, checkpoint, dataset_stats)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    server_log_path = log_dir / f"worker_{worker_index:03d}_server.log"
    with server_log_path.open("a", encoding="utf-8") as server_log:
        server_cmd = [
            sys.executable, "-u", "script/policy_model_server.py",
            "--port", str(port), "--config", policy_config, "--overrides", *common,
        ]
        server = subprocess.Popen(
            server_cmd, cwd=robotwin_root, env=env,
            stdout=server_log, stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_server(server, port)
            actor_count = int(cfg.MULTIRUN.env_num_per_gpu)
            print(f"Worker {worker_index} loaded one model for {actor_count} actors", flush=True)

            def actor_main(actor_index: int) -> None:
                while True:
                    claimed = dispatcher.claim_with_index()
                    if claimed is None:
                        return
                    task_index, raw = claimed
                    job = json.loads(raw)
                    task_name = str(job["task_name"])
                    print(
                        f"[Task {task_index + 1}/{len(dispatcher.tasks)}] "
                        f"Actor {actor_index} started {task_name}",
                        flush=True,
                    )
                    for phase, task_config in (("clean", "demo_clean"), ("random", "demo_randomized")):
                        if valid_phase_result(output_dir, task_name, phase):
                            print(
                                f"[Task {task_index + 1}/{len(dispatcher.tasks)}] "
                                f"Actor {actor_index} skipped completed {task_name} {phase}",
                                flush=True,
                            )
                            continue
                        print(
                            f"[Task {task_index + 1}/{len(dispatcher.tasks)}] "
                            f"Actor {actor_index} started {task_name} {phase}",
                            flush=True,
                        )
                        client_overrides = list(common)
                        for key, value in {
                            "task_name": task_name,
                            "task_config": task_config,
                            "instruction_type": cfg.EVALUATION.instruction_type,
                            "eval_num_episodes": cfg.EVALUATION.eval_num_episodes,
                            "eval_output_dir": str(output_dir / task_name),
                            "skip_get_obs_within_replan": cfg.EVALUATION.skip_get_obs_within_replan,
                        }.items():
                            _append_override(client_overrides, key, value)
                        client_cmd = [
                            sys.executable, "-u", "script/eval_policy_client.py",
                            "--port", str(port), "--config", policy_config,
                            "--overrides", *client_overrides,
                        ]
                        client_log_path = log_dir / f"worker_{worker_index:03d}_{task_name}_{phase}.log"
                        with client_log_path.open("w", encoding="utf-8") as client_log:
                            subprocess.run(
                                client_cmd, cwd=robotwin_root, env=env,
                                stdout=client_log, stderr=subprocess.STDOUT, check=True,
                            )
                        print(
                            f"[Task {task_index + 1}/{len(dispatcher.tasks)}] "
                            f"Actor {actor_index} completed {task_name} {phase}",
                            flush=True,
                        )

            with ThreadPoolExecutor(max_workers=actor_count, thread_name_prefix="rollout") as executor:
                futures = [executor.submit(actor_main, index) for index in range(actor_count)]
                for future in futures:
                    future.result()
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    main()
