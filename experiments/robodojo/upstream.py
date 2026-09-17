"""Helpers for the official RoboDojo simulator checkout."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_SOURCE = PROJECT_ROOT / "experiments" / "robodojo" / "easywam_policy"


def validate_robodojo_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    required = (
        "scripts/eval_policy.sh",
        "scripts/internal/task_inventory.py",
        "scripts/internal/summarize_result.py",
        "src/eval_client/main.py",
        "env_cfg/arx_x5.yml",
        "XPolicyLab/client_server/ws/model_server.py",
    )
    missing = [item for item in required if not (root / item).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Invalid RoboDojo checkout at {root}; missing: {', '.join(missing)}"
        )
    return root


def install_policy_adapter(root: Path) -> Path:
    """Install only our small simulator-side adapter; never overwrite foreign files."""
    destination = root / "XPolicyLab" / "policy" / "easywam_policy"
    for name in ("deploy.py", "deploy.yml"):
        source = POLICY_SOURCE / name
        target = destination / name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise FileExistsError(f"Refusing to overwrite a different policy adapter: {target}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("deploy.py", "deploy.yml"):
        target = destination / name
        if not target.exists():
            shutil.copy2(POLICY_SOURCE / name, target)
    return destination


def load_inventory(root: Path) -> list[dict]:
    result = subprocess.run(
        [sys.executable, str(root / "scripts/internal/task_inventory.py"), "--format", "json", "--check"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    records = payload.get("tasks")
    if not isinstance(records, list) or not records:
        raise ValueError("RoboDojo task inventory is empty or malformed.")
    return [record for record in records if record.get("runnable")]


def build_client_command(
    *, root: Path, task_name: str, host: str, port: int, env_cfg_type: str,
    checkpoint_tag: str, seed: int, env_gpu: int, sim_env: str | None,
    sim_num_envs: int | None = None,
) -> list[str]:
    if sim_num_envs is None:
        eval_cfg = yaml.safe_load((root / "env_cfg" / f"{env_cfg_type}.yml").read_text(encoding="utf-8"))
        sim_cfg_name = eval_cfg["config"]["sim"]
        sim_cfg = yaml.safe_load((root / "env_cfg" / "sim" / f"{sim_cfg_name}.yml").read_text(encoding="utf-8"))
        sim_num_envs = int(sim_cfg.get("scene", {}).get("num_envs", 1))
    if sim_num_envs <= 0:
        raise ValueError("sim_num_envs must be positive.")
    args = [
        "python", "-u", str(root / "src/eval_client/main.py"),
        "--task_name", task_name, "--env_cfg_type", env_cfg_type,
        "--num_envs", str(sim_num_envs), "--enable_cameras",
        "--kit_args", " --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera",
        "--device_id", str(env_gpu),
        "--policy_name", "easywam_policy", "--host", host,
        "--port", str(port), "--protocol", "ws",
        "--policy_server_url", f"ws://{host}:{port}",
        "--additional_info", f"ckpt_name={checkpoint_tag},action_type=joint",
        "--seed", str(seed), "--headless",
    ]
    return (["conda", "run", "--no-capture-output", "-n", sim_env] + args) if sim_env else args
