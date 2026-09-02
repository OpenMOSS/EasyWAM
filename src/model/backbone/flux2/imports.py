from __future__ import annotations

import sys
from pathlib import Path


def ensure_flux2_importable(source_path: str | None = None) -> None:
    """Expose the official FLUX.2 source tree without vendoring it into EasyWAM."""
    if source_path:
        source_root = Path(source_path).expanduser().resolve()
        # Official checkouts use a src/flux2 package layout, while installed or
        # exported source trees may expose flux2 directly at their root.
        import_roots = (source_root / "src", source_root)
        for candidate in reversed(import_roots):
            resolved = str(candidate)
            if candidate.is_dir() and resolved not in sys.path:
                sys.path.insert(0, resolved)
    try:
        import flux2  # noqa: F401
    except ImportError as exc:
        hint = f" from {source_path!r}" if source_path else ""
        raise ImportError(
            "The official FLUX.2 Python package is required for backbone='flux2'"
            f"{hint}. Set backbone.flux2_src_path to its source root."
        ) from exc
