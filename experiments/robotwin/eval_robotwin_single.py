"""Single-task EasyWAM evaluation against an external RoboTwin checkout."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
POLICY_NAME = "easywam_policy"
SERVER_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "policy_server.py"
POLICY_CONFIG = (
    PROJECT_ROOT / "experiments" / "robotwin" / POLICY_NAME / "deploy_policy.yml"
)


def _resolve_path(path_str: str, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _resolve_optional_path(path_value: Any, *, base: Path) -> Path | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return _resolve_path(text, base=base)


def _resolve_dataset_stats_path(cfg: DictConfig, ckpt_path: Path) -> Path:
    explicit = _resolve_optional_path(
        cfg.EVALUATION.dataset_stats_path, base=PROJECT_ROOT
    )
    candidates = ([explicit] if explicit is not None else []) + [
        (parent / "dataset_stats.json").resolve()
        for parent in list(ckpt_path.parents)[:4]
    ]
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. Pass "
        "EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        index = parts.index("runs")
        if index + 2 < len(parts):
            return f"{parts[index + 1]}_{parts[index + 2]}"
    return ckpt_path.stem


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(str(value))


def _append_override(
    overrides: list[str], key: str, value: Any, *, skip_none: bool = True
) -> None:
    if skip_none and value is None:
        return
    overrides.extend([f"--{key}", _format_override_value(value)])


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


def _model_overrides(cfg: DictConfig, checkpoint: Path, dataset_stats: Path) -> list[str]:
    values = {
        "ckpt_setting": str(checkpoint),
        "seed": cfg.seed,
        "sim_cfg_path": str(
            (PROJECT_ROOT / "configs" / "benchmark" / "sim_robotwin.yaml").resolve()
        ),
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
    overrides: list[str] = []
    for key, value in values.items():
        _append_override(overrides, key, value)
    return overrides


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="benchmark/sim_robotwin.yaml",
)
def main(cfg: DictConfig) -> None:
    from experiments.robotwin.upstream import (
        build_eval_command,
        normalize_phase_result,
        optional_text,
        validate_robotwin_root,
    )

    if cfg.ckpt is None or cfg.EVALUATION.task_name is None:
        raise ValueError("ckpt and EVALUATION.task_name are required.")
    checkpoint = _resolve_path(str(cfg.ckpt))
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    robotwin_root = validate_robotwin_root(
        _resolve_path(str(cfg.EVALUATION.robotwin_root))
    )
    dataset_stats = _resolve_dataset_stats_path(cfg, checkpoint)
    output_hint = _resolve_path(str(cfg.EVALUATION.output_dir))
    run_output = (
        PROJECT_ROOT
        / "evaluate_results"
        / "robotwin"
        / _resolve_ckpt_tag(checkpoint)
        / output_hint.name
    )
    task_name = str(cfg.EVALUATION.task_name)
    task_config = str(cfg.EVALUATION.task_config)
    phase = "random" if task_config == "demo_randomized" else "clean"
    phase_root = run_output / task_name / "upstream" / phase
    phase_root.mkdir(parents=True, exist_ok=True)
    previous = {path.resolve() for path in phase_root.rglob("_result.txt")}
    log_dir = run_output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    env.setdefault("PYTHONUTF8", "1")
    env["PYTHONUNBUFFERED"] = "1"

    server_log_path = log_dir / f"server_{task_name}_{phase}.log"
    with server_log_path.open("w", encoding="utf-8") as server_log:
        server = subprocess.Popen(
            [
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
                *_model_overrides(cfg, checkpoint, dataset_stats),
            ],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_server(server, port)
            command = build_eval_command(
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
                instruction_type=optional_text(cfg.EVALUATION.instruction_type),
            )
            client_log_path = log_dir / f"eval_{task_name}_{phase}.log"
            with client_log_path.open("w", encoding="utf-8") as client_log:
                subprocess.run(
                    command,
                    cwd=phase_root,
                    env=env,
                    stdout=client_log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()

    result = normalize_phase_result(
        phase_root,
        previous_results=previous,
        canonical_result=run_output / task_name / f"_result_{phase}.txt",
    )
    OmegaConf.save(cfg, run_output / f"eval_config_{task_name}_{phase}.yaml")
    print(f"RoboTwin evaluation complete: {result}")


if __name__ == "__main__":
    main()
