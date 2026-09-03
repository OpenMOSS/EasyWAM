"""Shared metadata and persistence helpers for LIBERO-Plus evaluation."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from experiments.libero.task_language import resolve_bddl_source_path


LIBERO_PLUS_SUITE_COUNTS = {
    "libero_spatial": 2402,
    "libero_object": 2518,
    "libero_goal": 2591,
    "libero_10": 2519,
}

CATEGORY_LABELS = {
    "layout": "Objects Layout",
    "camera": "Camera Viewpoints",
    "robot": "Robot Initial States",
    "language": "Language Instructions",
    "light": "Light Conditions",
    "background": "Background Textures",
    "noise": "Sensor Noise",
}

_CATEGORY_SLUGS = {label.casefold(): slug for slug, label in CATEGORY_LABELS.items()}


@dataclass(frozen=True)
class TaskSpec:
    suite: str
    task_id: int
    task_name: str
    category: str
    category_label: str
    difficulty_level: Optional[int]
    classification_id: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskSpec":
        return cls(
            suite=str(value["suite"]),
            task_id=int(value["task_id"]),
            task_name=str(value["task_name"]),
            category=str(value["category"]),
            category_label=str(value["category_label"]),
            difficulty_level=(
                None
                if value.get("difficulty_level") is None
                else int(value["difficulty_level"])
            ),
            classification_id=int(value["classification_id"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LiberoPlusCatalog:
    benchmark: Any
    get_libero_path: Any
    classification_path: Path
    tasks_by_suite: dict[str, list[TaskSpec]]


def _libero_config_file() -> Path:
    config_root = Path(
        os.path.expanduser(os.environ.get("LIBERO_CONFIG_PATH", "~/.libero"))
    )
    return config_root / "config.yaml"


def _load_libero_modules() -> tuple[Any, Any]:
    config_file = _libero_config_file()
    if not config_file.is_file():
        raise RuntimeError(
            f"LIBERO config is missing: {config_file}. Install LIBERO-Plus and "
            "initialize its path configuration before running evaluation."
        )
    try:
        libero_module = importlib.import_module("libero.libero")
        benchmark_module = importlib.import_module("libero.libero.benchmark")
    except Exception as exc:
        raise RuntimeError(
            "Failed to import LIBERO-Plus from the `libero` package. The vanilla "
            "LIBERO and LIBERO-Plus packages cannot be installed together."
        ) from exc
    return benchmark_module, libero_module.get_libero_path


def instantiate_suite(benchmark: Any, suite_name: str):
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise ValueError(
            f"Unknown LIBERO-Plus suite {suite_name!r}. "
            f"Available suites: {sorted(benchmark_dict)}"
        )
    suite_class = benchmark_dict[suite_name]
    # LIBERO-Plus has shipped two compatible benchmark APIs. Newer revisions
    # expose every perturbation from the default constructor. The paper
    # revision exposes one category per constructor through `category_value`.
    # Normalize both to one all-category suite so task manifests stay stable.
    with contextlib.redirect_stdout(io.StringIO()):
        suite = suite_class()
    expected_count = LIBERO_PLUS_SUITE_COUNTS.get(suite_name)
    if expected_count is None or int(suite.n_tasks) == expected_count:
        return suite

    category_suites = []
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for label in CATEGORY_LABELS.values():
                category_suites.append(suite_class(category_value=label))
    except TypeError:
        # Let the caller's authoritative count check report the incompatible
        # installation when this is neither supported API.
        return suite

    combined_tasks = []
    seen_names = set()
    for category_suite in category_suites:
        for task in category_suite.tasks:
            task_name = str(task.name)
            if task_name in seen_names:
                raise RuntimeError(
                    f"Duplicate LIBERO-Plus task {task_name!r} while combining categories "
                    f"for {suite_name}."
                )
            seen_names.add(task_name)
            combined_tasks.append(task)
    if len(combined_tasks) != expected_count:
        return suite
    # The paper revision accidentally shuffles even task_order_index=0 in
    # get_ids_by_category(), so two fresh processes can assign different task
    # names to the same integer index. Reconstruct the canonical order from the
    # classification IDs used by that constructor.
    classification_path = Path(benchmark.__file__).resolve().parent / "task_classification.json"
    if classification_path.is_file():
        with classification_path.open("r", encoding="utf-8") as f:
            classification = json.load(f)
        raw_metadata = classification.get(suite_name, [])
        classification_ids = {
            str(item["name"]): int(item["id"])
            for item in raw_metadata
        }
        missing_names = sorted(seen_names - set(classification_ids))
        if missing_names:
            raise RuntimeError(
                f"LIBERO-Plus classification is missing combined tasks for {suite_name}: "
                f"{missing_names[:8]}"
            )
        combined_tasks.sort(key=lambda task: classification_ids[str(task.name)])
    else:
        # Dependency-light test doubles do not necessarily expose the metadata
        # file; name order still makes repeated instantiation deterministic.
        combined_tasks.sort(key=lambda task: str(task.name))
    suite.tasks = combined_tasks
    suite.n_tasks = len(combined_tasks)
    return suite


def _resolve_bddl_source_path(task_bddl_path: Path) -> Path:
    """Resolve a LIBERO-Plus virtual task filename to its source BDDL file."""
    return resolve_bddl_source_path(task_bddl_path)


def _validate_libero_paths(get_libero_path: Any) -> None:
    config_file = _libero_config_file()
    with config_file.open("r", encoding="utf-8") as f:
        path_config = yaml.safe_load(f) or {}
    required_keys = {"benchmark_root", "bddl_files", "init_states", "assets"}
    missing_keys = sorted(required_keys - set(path_config))
    if missing_keys:
        raise RuntimeError(
            f"LIBERO config {config_file} is missing keys: {missing_keys}."
        )

    resolved_paths = {key: Path(get_libero_path(key)).expanduser() for key in required_keys}
    missing_paths = [f"{key}={path}" for key, path in resolved_paths.items() if not path.exists()]
    if missing_paths:
        raise RuntimeError("LIBERO-Plus paths do not exist: " + ", ".join(missing_paths))

    assets_root = resolved_paths["assets"]
    extended_assets = [
        "new_objects",
        "scenes",
        "textures",
        "turbosquid_objects",
    ]
    missing_assets = [name for name in extended_assets if not (assets_root / name).exists()]
    if missing_assets:
        raise RuntimeError(
            f"LIBERO-Plus extended assets are incomplete under {assets_root}: "
            f"missing {missing_assets}."
        )


def load_libero_plus_catalog(
    suite_names: Iterable[str],
    *,
    validate_paths: bool = True,
) -> LiberoPlusCatalog:
    requested_suites = list(dict.fromkeys(str(name) for name in suite_names))
    unsupported = sorted(set(requested_suites) - set(LIBERO_PLUS_SUITE_COUNTS))
    if unsupported:
        raise ValueError(
            f"Unsupported LIBERO-Plus suites: {unsupported}. The classified benchmark "
            f"contains only {list(LIBERO_PLUS_SUITE_COUNTS)}."
        )
    if not requested_suites:
        raise ValueError("At least one LIBERO-Plus task suite must be selected.")

    benchmark, get_libero_path = _load_libero_modules()
    if validate_paths:
        _validate_libero_paths(get_libero_path)

    classification_path = Path(benchmark.__file__).resolve().parent / "task_classification.json"
    if not classification_path.is_file():
        raise RuntimeError(
            f"task_classification.json was not found next to the installed benchmark: "
            f"{classification_path}. This does not appear to be LIBERO-Plus."
        )
    with classification_path.open("r", encoding="utf-8") as f:
        classification = json.load(f)

    tasks_by_suite: dict[str, list[TaskSpec]] = {}
    bddl_root = Path(get_libero_path("bddl_files")).expanduser()
    for suite_name in requested_suites:
        expected_count = LIBERO_PLUS_SUITE_COUNTS[suite_name]
        suite = instantiate_suite(benchmark, suite_name)
        if int(suite.n_tasks) != expected_count:
            raise RuntimeError(
                f"Suite {suite_name} contains {suite.n_tasks} tasks; expected "
                f"{expected_count} from LIBERO-Plus. The installed package may be vanilla LIBERO."
            )

        raw_metadata = classification.get(suite_name)
        if not isinstance(raw_metadata, list) or len(raw_metadata) != expected_count:
            raise RuntimeError(
                f"Classification metadata for {suite_name} has "
                f"{0 if raw_metadata is None else len(raw_metadata)} entries; "
                f"expected {expected_count}."
            )
        metadata_by_name = {str(item["name"]): item for item in raw_metadata}
        if len(metadata_by_name) != len(raw_metadata):
            raise RuntimeError(f"Duplicate task names found in {suite_name} classification metadata.")

        suite_specs: list[TaskSpec] = []
        for task_id in range(expected_count):
            task = suite.get_task(task_id)
            task_name = str(task.name)
            metadata = metadata_by_name.get(task_name)
            if metadata is None:
                raise RuntimeError(
                    f"Task {suite_name}:{task_id} ({task_name}) has no classification metadata."
                )
            category_label = str(metadata["category"])
            category = _CATEGORY_SLUGS.get(category_label.casefold())
            if category is None:
                raise RuntimeError(
                    f"Unknown LIBERO-Plus category {category_label!r} for {suite_name}:{task_id}."
                )
            difficulty = metadata.get("difficulty_level")
            if difficulty is not None:
                difficulty = int(difficulty)
                if difficulty not in range(1, 6):
                    raise RuntimeError(
                        f"Invalid difficulty {difficulty} for {suite_name}:{task_id}."
                    )
            bddl_path = bddl_root / task.problem_folder / task.bddl_file
            bddl_source_path = _resolve_bddl_source_path(bddl_path)
            if validate_paths and not bddl_source_path.is_file():
                raise RuntimeError(
                    f"BDDL source file is missing for {suite_name}:{task_id}: "
                    f"task={bddl_path}, source={bddl_source_path}"
                )
            suite_specs.append(
                TaskSpec(
                    suite=suite_name,
                    task_id=task_id,
                    task_name=task_name,
                    category=category,
                    category_label=category_label,
                    difficulty_level=difficulty,
                    classification_id=int(metadata["id"]),
                )
            )
        tasks_by_suite[suite_name] = suite_specs

    return LiberoPlusCatalog(
        benchmark=benchmark,
        get_libero_path=get_libero_path,
        classification_path=classification_path,
        tasks_by_suite=tasks_by_suite,
    )


def select_tasks(
    tasks_by_suite: dict[str, list[TaskSpec]],
    *,
    categories: Optional[Iterable[str]] = None,
    difficulty_levels: Optional[Iterable[int]] = None,
    task_ids: Optional[Iterable[int]] = None,
) -> list[TaskSpec]:
    category_filter = None
    if categories is not None:
        category_filter = {str(value).strip().lower() for value in categories}
        unknown = sorted(category_filter - set(CATEGORY_LABELS))
        if unknown:
            raise ValueError(
                f"Unknown category slugs: {unknown}. Expected values from {list(CATEGORY_LABELS)}."
            )

    difficulty_filter = None
    if difficulty_levels is not None:
        difficulty_filter = {int(value) for value in difficulty_levels}
        invalid = sorted(difficulty_filter - set(range(1, 6)))
        if invalid:
            raise ValueError(f"Invalid difficulty levels: {invalid}. Expected values 1 through 5.")

    task_id_filter = None if task_ids is None else {int(value) for value in task_ids}
    if task_id_filter is not None and any(value < 0 for value in task_id_filter):
        raise ValueError("Task IDs must be zero-based non-negative integers.")

    selected: list[TaskSpec] = []
    for suite_name, suite_tasks in tasks_by_suite.items():
        if task_id_filter is not None:
            out_of_range = sorted(value for value in task_id_filter if value >= len(suite_tasks))
            if out_of_range:
                raise ValueError(
                    f"Task IDs out of range for {suite_name} ({len(suite_tasks)} tasks): "
                    f"{out_of_range}"
                )
        for task in suite_tasks:
            if category_filter is not None and task.category not in category_filter:
                continue
            if difficulty_filter is not None and task.difficulty_level not in difficulty_filter:
                continue
            if task_id_filter is not None and task.task_id not in task_id_filter:
                continue
            selected.append(task)
    if not selected:
        raise ValueError("LIBERO-Plus task filters selected zero tasks.")
    return selected


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_task_jsonl(path: Path) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                tasks.append(TaskSpec.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid task record at {path}:{line_number}") from exc
    return tasks


def result_path(output_dir: Path, task: TaskSpec) -> Path:
    return output_dir / "results" / task.suite / f"task_{task.task_id:04d}.json"


def error_path(output_dir: Path, task: TaskSpec) -> Path:
    return output_dir / "errors" / task.suite / f"task_{task.task_id:04d}.json"


def is_valid_result(path: Path, task: TaskSpec) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            result = json.load(f)
        return (
            result.get("task_suite") == task.suite
            and int(result.get("task_id")) == task.task_id
            and result.get("task_name") == task.task_name
            and int(result.get("total_episodes")) == 1
            and 0 <= int(result.get("successes")) <= 1
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
