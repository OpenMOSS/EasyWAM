"""Assemble the GitHub Pages entry point from small, editable HTML sections."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
TEMPLATE = SOURCE / "pages" / "index.html"
OUTPUT = ROOT / "index.html"
INCLUDE = re.compile(r"<!-- include:(components/[a-z0-9/_-]+\.html) -->")


def build() -> str:
    template = TEMPLATE.read_text(encoding="utf-8")

    def replace(match: re.Match[str]) -> str:
        component = SOURCE / match.group(1)
        return component.read_text(encoding="utf-8").rstrip("\n")

    result = INCLUDE.sub(replace, template)
    if "<!-- include:" in result:
        raise ValueError("Unresolved component include in page template")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if index.html needs rebuilding")
    args = parser.parse_args()
    output = build()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != output:
            parser.exit(1, "index.html is out of date; run python scripts/build.py\n")
        print("index.html is up to date")
        return
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as file:
        file.write(output)
    print("Built index.html")


if __name__ == "__main__":
    main()
