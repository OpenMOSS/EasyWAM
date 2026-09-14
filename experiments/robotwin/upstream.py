"""Helpers for invoking an external, current RoboTwin checkout."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

ROBOTWIN_EVAL_SCRIPT = Path("scripts/eval_policy_xpolicylab.py")
ROBOTWIN_TASK_CONFIG_ROOT = Path("env_cfg/task_config")


def validate_robotwin_root(robotwin_root: Path) -> Path:
    root = robotwin_root.expanduser().resolve()
    required = (
        ROBOTWIN_EVAL_SCRIPT,
        ROBOTWIN_TASK_CONFIG_ROOT / "_eval_step_limit.yml",
        ROBOTWIN_TASK_CONFIG_ROOT / "demo_clean.yml",
        ROBOTWIN_TASK_CONFIG_ROOT / "demo_randomized.yml",
        Path("XPolicyLab/client_server/ws/model_server.py"),
    )
    missing = [str(path) for path in required if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Invalid RoboTwin checkout at {root}; missing: {', '.join(missing)}. "
            "Clone it with --recurse-submodules."
        )
    return root


def optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in {"", "none", "null"} else text


def build_eval_command(
    *,
    robotwin_root: Path,
    task_name: str,
    task_config: str,
    policy_name: str,
    checkpoint_tag: str,
    host: str,
    port: int,
    env_cfg_type: str,
    action_type: str,
    seed: int,
    episodes: int,
    instruction_type: str | None,
) -> list[str]:
    if action_type != "joint":
        raise ValueError("EasyWAM RoboTwin evaluation supports action_type=joint.")
    command = [
        sys.executable,
        "-u",
        str(robotwin_root / ROBOTWIN_EVAL_SCRIPT),
        "--bench_name",
        "RoboTwin",
        "--task_name",
        task_name,
        "--env_cfg_type",
        env_cfg_type,
        "--policy_name",
        policy_name,
        "--host",
        host,
        "--port",
        str(port),
        "--protocol",
        "ws",
        "--root_dir",
        str(robotwin_root),
        "--device_id",
        "0",
        "--additional_info",
        f"ckpt_name={checkpoint_tag},action_type={action_type}",
        "--seed",
        str(seed),
        "--task_config",
        task_config,
        "--test_num",
        str(episodes),
    ]
    if instruction_type is not None:
        command.extend(("--instruction_type", instruction_type))
    return command


def normalize_phase_result(
    phase_root: Path,
    *,
    previous_results: set[Path],
    canonical_result: Path,
) -> Path:
    results = {path.resolve() for path in phase_root.rglob("_result.txt")}
    created = sorted(results - previous_results, key=lambda path: path.stat().st_mtime_ns)
    if len(created) != 1:
        raise RuntimeError(
            f"Expected exactly one new RoboTwin result under {phase_root}, found {len(created)}."
        )
    canonical_result.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(created[0], canonical_result)
    return created[0]
