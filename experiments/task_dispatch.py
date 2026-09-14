from __future__ import annotations

import fcntl
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class WorkerSlot:
    worker_index: int
    gpu_id: int
    gpu_worker_index: int


def resolve_gpu_ids(
    *, num_gpus: int, gpu_ids: Sequence[int] | None = None
) -> list[int]:
    """Select ``num_gpus`` physical devices from an optional ordered candidate pool."""
    if num_gpus <= 0:
        raise ValueError("num_gpus must be positive.")
    if gpu_ids is None:
        return list(range(num_gpus))

    if any(
        isinstance(gpu_id, bool) or not isinstance(gpu_id, Integral)
        for gpu_id in gpu_ids
    ):
        raise ValueError("gpu_ids must contain integer device IDs.")
    resolved = [int(gpu_id) for gpu_id in gpu_ids]
    if not resolved:
        raise ValueError("gpu_ids must not be empty when configured.")
    if any(gpu_id < 0 for gpu_id in resolved):
        raise ValueError("gpu_ids must contain only non-negative device IDs.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("gpu_ids must not contain duplicate device IDs.")
    if len(resolved) < num_gpus:
        raise ValueError(
            f"gpu_ids provides {len(resolved)} candidates, fewer than "
            f"num_gpus={num_gpus}."
        )
    return resolved[:num_gpus]


def build_worker_slots(
    *,
    num_gpus: int,
    workers_per_gpu: int,
    pending_jobs: int,
    gpu_ids: Sequence[int] | None = None,
) -> list[WorkerSlot]:
    """Assign model workers across GPUs in round-robin order."""
    resolved_gpu_ids = resolve_gpu_ids(num_gpus=num_gpus, gpu_ids=gpu_ids)
    if workers_per_gpu <= 0:
        raise ValueError("workers_per_gpu must be positive.")
    if pending_jobs < 0:
        raise ValueError("pending_jobs must not be negative.")
    gpu_count = len(resolved_gpu_ids)
    worker_count = min(pending_jobs, gpu_count * workers_per_gpu)
    return [
        WorkerSlot(
            worker_index=index,
            gpu_id=resolved_gpu_ids[index % gpu_count],
            gpu_worker_index=index // gpu_count,
        )
        for index in range(worker_count)
    ]


def count_workers_by_gpu(slots: list[WorkerSlot]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for slot in slots:
        counts[slot.gpu_id] = counts.get(slot.gpu_id, 0) + 1
    return counts


class FileTaskDispatcher:
    def __init__(self, task_file: Path, cursor_file: Path) -> None:
        self.tasks = [
            line
            for line in task_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.cursor_file = cursor_file
        self.cursor_file.parent.mkdir(parents=True, exist_ok=True)
        self.cursor_file.touch(exist_ok=True)

    def claim(self) -> str | None:
        claimed = self.claim_with_index()
        return None if claimed is None else claimed[1]

    def claim_with_index(self) -> tuple[int, str] | None:
        with self.cursor_file.open("r+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            raw = handle.read().strip()
            index = int(raw) if raw else 0
            if index >= len(self.tasks):
                return None
            handle.seek(0)
            handle.truncate()
            handle.write(str(index + 1))
            handle.flush()
            return index, self.tasks[index]
