from __future__ import annotations

import fcntl
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkerSlot:
    worker_index: int
    gpu_id: int
    gpu_worker_index: int


def build_worker_slots(
    *, num_gpus: int, workers_per_gpu: int, pending_jobs: int
) -> list[WorkerSlot]:
    """Assign model workers across GPUs in round-robin order."""
    if num_gpus <= 0:
        raise ValueError("num_gpus must be positive.")
    if workers_per_gpu <= 0:
        raise ValueError("workers_per_gpu must be positive.")
    if pending_jobs < 0:
        raise ValueError("pending_jobs must not be negative.")
    worker_count = min(pending_jobs, num_gpus * workers_per_gpu)
    return [
        WorkerSlot(
            worker_index=index,
            gpu_id=index % num_gpus,
            gpu_worker_index=index // num_gpus,
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
