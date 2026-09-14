"""Persistent EasyWAM worker for an external RoboTwin checkout."""

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
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robotwin.eval_robotwin_single import (  # noqa: E402
    _model_overrides,
    _resolve_dataset_stats_path,
    _resolve_path,
)
from experiments.robotwin.result_utils import valid_phase_result  # noqa: E402
from experiments.robotwin.upstream import (  # noqa: E402
    build_eval_command,
    normalize_phase_result,
    optional_text,
    validate_robotwin_root,
)
from experiments.task_dispatch import FileTaskDispatcher  # noqa: E402

SERVER_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "policy_server.py"
POLICY_CONFIG = (
    PROJECT_ROOT / "experiments" / "robotwin" / "easywam_policy" / "deploy_policy.yml"
)


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


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="benchmark/sim_robotwin.yaml",
)
def main(cfg: DictConfig) -> None:
    task_file = cfg.WORKER.get("task_file")
    cursor_file = cfg.WORKER.get("task_cursor")
    if not task_file or not cursor_file:
        raise ValueError("WORKER.task_file and WORKER.task_cursor are required.")
    task_path = Path(str(task_file)).expanduser().resolve()
    if not task_path.read_text(encoding="utf-8").strip():
        return
    dispatcher = FileTaskDispatcher(
        task_path, Path(str(cursor_file)).expanduser().resolve()
    )

    checkpoint = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    robotwin_root = validate_robotwin_root(
        _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    )
    dataset_stats = _resolve_dataset_stats_path(cfg, checkpoint)
    output_dir = Path(str(cfg.EVALUATION.output_dir)).expanduser().resolve()
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    worker_index = int(cfg.WORKER.worker_index)
    port = _free_port()
    common = _model_overrides(cfg, checkpoint, dataset_stats)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env["PYTHONUNBUFFERED"] = "1"

    server_log_path = log_dir / f"worker_{worker_index:03d}_server.log"
    with server_log_path.open("a", encoding="utf-8") as server_log:
        server_cmd = [
            sys.executable,
            "-u",
            str(SERVER_ENTRY),
            "--robotwin-root",
            str(robotwin_root),
            "--config",
            str(POLICY_CONFIG),
            "--port",
            str(port),
            "--overrides",
            *common,
        ]
        server = subprocess.Popen(
            server_cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_server(server, port)
            actor_count = int(cfg.MULTIRUN.env_num_per_worker)
            print(
                f"Worker {worker_index} loaded one model for {actor_count} actors",
                flush=True,
            )

            def actor_main(actor_index: int) -> None:
                while True:
                    claimed = dispatcher.claim_with_index()
                    if claimed is None:
                        return
                    task_index, raw = claimed
                    task_name = str(json.loads(raw)["task_name"])
                    print(
                        f"[Task {task_index + 1}/{len(dispatcher.tasks)}] "
                        f"Actor {actor_index} started {task_name}",
                        flush=True,
                    )
                    for phase, task_config in (
                        ("clean", "demo_clean"),
                        ("random", "demo_randomized"),
                    ):
                        if valid_phase_result(output_dir, task_name, phase):
                            print(
                                f"[Task {task_index + 1}/{len(dispatcher.tasks)}] "
                                f"Actor {actor_index} skipped completed {task_name} {phase}",
                                flush=True,
                            )
                            continue

                        phase_root = output_dir / task_name / "upstream" / phase
                        phase_root.mkdir(parents=True, exist_ok=True)
                        previous = {
                            path.resolve() for path in phase_root.rglob("_result.txt")
                        }
                        client_cmd = build_eval_command(
                            robotwin_root=robotwin_root,
                            task_name=task_name,
                            task_config=task_config,
                            policy_name=str(cfg.EVALUATION.policy_name),
                            checkpoint_tag=checkpoint.stem,
                            host="127.0.0.1",
                            port=port,
                            env_cfg_type=str(cfg.EVALUATION.env_cfg_type),
                            action_type=str(cfg.EVALUATION.action_type),
                            seed=int(cfg.seed),
                            episodes=int(cfg.EVALUATION.eval_num_episodes),
                            instruction_type=optional_text(
                                cfg.EVALUATION.instruction_type
                            ),
                        )
                        client_log_path = (
                            log_dir
                            / f"worker_{worker_index:03d}_{task_name}_{phase}.log"
                        )
                        with client_log_path.open("w", encoding="utf-8") as client_log:
                            subprocess.run(
                                client_cmd,
                                cwd=phase_root,
                                env=env,
                                stdout=client_log,
                                stderr=subprocess.STDOUT,
                                check=True,
                            )
                        normalize_phase_result(
                            phase_root,
                            previous_results=previous,
                            canonical_result=(
                                output_dir / task_name / f"_result_{phase}.txt"
                            ),
                        )
                        print(
                            f"[Task {task_index + 1}/{len(dispatcher.tasks)}] "
                            f"Actor {actor_index} completed {task_name} {phase}",
                            flush=True,
                        )

            with ThreadPoolExecutor(
                max_workers=actor_count, thread_name_prefix="rollout"
            ) as executor:
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
