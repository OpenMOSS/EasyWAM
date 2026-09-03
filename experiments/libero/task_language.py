"""Task-language helpers shared by LIBERO and LIBERO-Plus evaluation."""

from __future__ import annotations

import re
from pathlib import Path


_NUMBER_PATTERN = r"-?\d+(?:\.\d+)?"
_LIBERO_PLUS_METADATA_SUFFIX_PATTERNS = (
    re.compile(
        rf"\s+view(?:\s+{_NUMBER_PATTERN}){{5}}"
        rf"\s+initstate\s+\d+(?:\s+noise\s+\d+)?$",
        re.IGNORECASE,
    ),
    re.compile(r"\s+(?:table|tb|light|add)\s+\d+$", re.IGNORECASE),
    re.compile(r"\s+level\d+\s+sample\d+$", re.IGNORECASE),
)


def resolve_bddl_source_path(task_bddl_path: Path) -> Path:
    """Resolve a virtual LIBERO-Plus task path to the BDDL that defines it."""
    if task_bddl_path.is_file():
        return task_bddl_path

    filename = task_bddl_path.name
    if "_view_" in filename and "_initstate_" in filename:
        source_filename = filename.split("_view_", 1)[0] + ".bddl"
        return task_bddl_path.with_name(source_filename)
    return task_bddl_path


def strip_libero_plus_metadata(task_language: str) -> str:
    """Remove only LIBERO-Plus perturbation metadata appended to task language."""
    for pattern in _LIBERO_PLUS_METADATA_SUFFIX_PATTERNS:
        cleaned_language, substitutions = pattern.subn("", task_language)
        if substitutions:
            return cleaned_language
    return task_language
