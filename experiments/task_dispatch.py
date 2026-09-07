from __future__ import annotations

import fcntl
from pathlib import Path


class FileTaskDispatcher:
    def __init__(self, task_file: Path, cursor_file: Path) -> None:
        self.tasks = [line for line in task_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.cursor_file = cursor_file
        self.cursor_file.parent.mkdir(parents=True, exist_ok=True)
        self.cursor_file.touch(exist_ok=True)

    def claim(self) -> str | None:
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
            return self.tasks[index]
