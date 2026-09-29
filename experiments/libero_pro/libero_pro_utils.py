"""Metadata, upstream isolation, validation, and persistence for LIBERO-Pro."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml


BASE_SUITES = ("libero_goal", "libero_spatial", "libero_10", "libero_object")
PERTURBATION_SUFFIXES = {
    "object": "object",
    "position": "swap",
    "semantic": "lan",
    "task": "task",
    "environment": "env",
}
OFFICIAL_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
TASKS_PER_SUITE = 10
PROTOCOL_ID = "libero-pro-official-five-dimension-v1"


@dataclass(frozen=True)
class TaskSpec:
    base_suite: str
    suite: str
    perturbation: str
    task_id: int
    task_name: str
    num_trials: int
    max_steps: int
    protocol: str
    source_commit: Optional[str]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskSpec":
        return cls(
            base_suite=str(value["base_suite"]),
            suite=str(value["suite"]),
            perturbation=str(value["perturbation"]),
            task_id=int(value["task_id"]),
            task_name=str(value["task_name"]),
            num_trials=int(value["num_trials"]),
            max_steps=int(value["max_steps"]),
            protocol=str(value["protocol"]),
            source_commit=(
                None
                if value.get("source_commit") is None
                else str(value["source_commit"])
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LiberoProCatalog:
    benchmark: Any
    tasks_by_suite: dict[str, list[TaskSpec]]
    source_root: Path
    data_root: Path
    source_commit: Optional[str]
    runtime_config: Path


def resolve_path(value: str | Path) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(str(value)))).resolve()


def registered_suite_name(base_suite: str, perturbation: str) -> str:
    if base_suite not in BASE_SUITES:
        raise ValueError(f"Unsupported LIBERO-Pro base suite: {base_suite!r}.")
    try:
        suffix = PERTURBATION_SUFFIXES[perturbation]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported LIBERO-Pro perturbation: {perturbation!r}. "
            f"Expected one of {list(PERTURBATION_SUFFIXES)}."
        ) from exc
    return f"{base_suite}_{suffix}"


def _validate_source_root(source_root: Path) -> None:
    required = [
        source_root / "libero" / "libero" / "__init__.py",
        source_root / "libero" / "libero" / "benchmark" / "__init__.py",
        source_root / "libero" / "libero" / "assets",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            f"LIBERO-Pro source root is incomplete: {source_root}. Missing: {missing}"
        )


def _git_commit(source_root: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def prepare_runtime_config(
    source_root: str | Path,
    data_root: str | Path,
    config_dir: str | Path,
) -> Path:
    """Create an isolated LIBERO path config without modifying the upstream tree."""
    source_root = resolve_path(source_root)
    data_root = resolve_path(data_root)
    config_dir = resolve_path(config_dir)
    _validate_source_root(source_root)

    bddl_root = data_root / "bddl_files"
    init_root = data_root / "init_files"
    missing = [str(path) for path in (bddl_root, init_root) if not path.is_dir()]
    if missing:
        raise RuntimeError(
            "LIBERO-Pro evaluation data is incomplete. Download the official "
            f"zhouxueyang/LIBERO-Pro dataset under {data_root}. Missing: {missing}"
        )

    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    payload = {
        "benchmark_root": str(source_root / "libero" / "libero"),
        "bddl_files": str(bddl_root),
        "init_states": str(init_root),
        "assets": str(source_root / "libero" / "libero" / "assets"),
    }
    temporary = config_path.with_suffix(f".tmp.{os.getpid()}.yaml")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, config_path)
    finally:
        temporary.unlink(missing_ok=True)
    return config_path


def activate_runtime(source_root: str | Path, config_path: str | Path) -> None:
    """Prioritize the requested upstream checkout and its isolated path config."""
    source_root = resolve_path(source_root)
    config_path = resolve_path(config_path)
    _validate_source_root(source_root)
    if not config_path.is_file():
        raise RuntimeError(f"LIBERO-Pro runtime config is missing: {config_path}")
    source_text = str(source_root)
    if source_text in sys.path:
        sys.path.remove(source_text)
    sys.path.insert(0, source_text)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_path.parent)


def load_benchmark(source_root: str | Path, config_path: str | Path):
    activate_runtime(source_root, config_path)
    benchmark = importlib.import_module("libero.libero.benchmark")
    expected_root = resolve_path(source_root)
    actual_path = Path(benchmark.__file__).resolve()
    if not actual_path.is_relative_to(expected_root):
        raise RuntimeError(
            "Imported the wrong 'libero' package. "
            f"Expected a module under {expected_root}, got {actual_path}. "
            "Run LIBERO-Pro through its dedicated manager in a fresh Python process."
        )
    return benchmark


def instantiate_suite(benchmark: Any, suite_name: str):
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise RuntimeError(
            f"LIBERO-Pro suite {suite_name!r} is not registered by {benchmark.__file__}."
        )
    with contextlib.redirect_stdout(io.StringIO()):
        return benchmark_dict[suite_name]()


def load_catalog(
    *,
    source_root: str | Path,
    data_root: str | Path,
    config_path: str | Path,
    base_suites: Iterable[str],
    perturbations: Iterable[str],
    num_trials: int,
) -> LiberoProCatalog:
    source_root = resolve_path(source_root)
    data_root = resolve_path(data_root)
    config_path = resolve_path(config_path)
    selected_bases = list(dict.fromkeys(str(value) for value in base_suites))
    selected_perturbations = list(
        dict.fromkeys(str(value).strip().lower() for value in perturbations)
    )
    if not selected_bases:
        raise ValueError("At least one LIBERO-Pro base suite must be selected.")
    if not selected_perturbations:
        raise ValueError("At least one LIBERO-Pro perturbation must be selected.")
    unknown_bases = sorted(set(selected_bases) - set(BASE_SUITES))
    unknown_perturbations = sorted(
        set(selected_perturbations) - set(PERTURBATION_SUFFIXES)
    )
    if unknown_bases:
        raise ValueError(f"Unsupported LIBERO-Pro base suites: {unknown_bases}.")
    if unknown_perturbations:
        raise ValueError(
            f"Unsupported LIBERO-Pro perturbations: {unknown_perturbations}."
        )
    if num_trials <= 0:
        raise ValueError("EVALUATION.num_trials must be positive.")

    benchmark = load_benchmark(source_root, config_path)
    source_commit = _git_commit(source_root)
    tasks_by_suite: dict[str, list[TaskSpec]] = {}
    bddl_root = data_root / "bddl_files"
    init_root = data_root / "init_files"
    for base_suite in selected_bases:
        for perturbation in selected_perturbations:
            suite_name = registered_suite_name(base_suite, perturbation)
            suite = instantiate_suite(benchmark, suite_name)
            if int(suite.n_tasks) != TASKS_PER_SUITE:
                raise RuntimeError(
                    f"Suite {suite_name} contains {suite.n_tasks} tasks; expected "
                    f"{TASKS_PER_SUITE}."
                )
            suite_specs: list[TaskSpec] = []
            for task_id in range(TASKS_PER_SUITE):
                task = suite.get_task(task_id)
                bddl_path = bddl_root / task.problem_folder / task.bddl_file
                init_path = init_root / task.problem_folder / task.init_states_file
                missing = [
                    str(path) for path in (bddl_path, init_path) if not path.is_file()
                ]
                if missing:
                    raise RuntimeError(
                        f"LIBERO-Pro files are missing for {suite_name}:{task_id} "
                        f"({task.name}): {missing}"
                    )
                suite_specs.append(
                    TaskSpec(
                        base_suite=base_suite,
                        suite=suite_name,
                        perturbation=perturbation,
                        task_id=task_id,
                        task_name=str(task.name),
                        num_trials=int(num_trials),
                        max_steps=OFFICIAL_MAX_STEPS[base_suite],
                        protocol=PROTOCOL_ID,
                        source_commit=source_commit,
                    )
                )
            tasks_by_suite[suite_name] = suite_specs

    return LiberoProCatalog(
        benchmark=benchmark,
        tasks_by_suite=tasks_by_suite,
        source_root=source_root,
        data_root=data_root,
        source_commit=source_commit,
        runtime_config=config_path,
    )


def select_tasks(
    tasks_by_suite: dict[str, list[TaskSpec]],
    *,
    task_ids: Optional[Iterable[int]] = None,
) -> list[TaskSpec]:
    task_id_filter = None if task_ids is None else {int(value) for value in task_ids}
    if task_id_filter is not None:
        invalid = sorted(value for value in task_id_filter if value not in range(TASKS_PER_SUITE))
        if invalid:
            raise ValueError(
                f"LIBERO-Pro task IDs must be in [0, {TASKS_PER_SUITE - 1}]: {invalid}"
            )
    selected = [
        task
        for suite_tasks in tasks_by_suite.values()
        for task in suite_tasks
        if task_id_filter is None or task.task_id in task_id_filter
    ]
    if not selected:
        raise ValueError("LIBERO-Pro task filters selected zero tasks.")
    return selected


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_task_jsonl(path: Path) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                tasks.append(TaskSpec.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid task record at {path}:{line_number}") from exc
    return tasks


def result_path(output_dir: Path, task: TaskSpec) -> Path:
    return (
        output_dir
        / "results"
        / task.base_suite
        / task.perturbation
        / f"task_{task.task_id:02d}.json"
    )


def error_path(output_dir: Path, task: TaskSpec) -> Path:
    return (
        output_dir
        / "errors"
        / task.base_suite
        / task.perturbation
        / f"task_{task.task_id:02d}.json"
    )


def is_valid_result(path: Path, task: TaskSpec) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        successes = int(result.get("successes"))
        return (
            result.get("task_suite") == task.suite
            and result.get("base_suite") == task.base_suite
            and result.get("perturbation") == task.perturbation
            and int(result.get("task_id")) == task.task_id
            and result.get("task_name") == task.task_name
            and result.get("protocol") == task.protocol
            and result.get("source_commit") == task.source_commit
            and int(result.get("total_episodes")) == task.num_trials
            and 0 <= successes <= task.num_trials
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
