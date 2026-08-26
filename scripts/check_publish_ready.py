"""Pre-push check for scaffolding that should not ship.

During development a codebase accumulates notes-to-self, implementation hints,
and unfinished stubs. They are useful while building and embarrassing in a
public repository, and they are easy to miss by eye across a dozen files.

This is deliberately NOT a pytest test. It is expected to fail while work is in
progress; failing unit test runs for that would be noise. Run it before pushing
anything you want a stranger to read.

    python scripts/check_publish_ready.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Directories that are not part of the published source.
SKIP_DIRS = {".git", ".venv", "__pycache__", "data", "reports", ".pytest_cache"}

# (regex, why it matters). Case-insensitive.
PATTERNS: list[tuple[str, str]] = [
    (r"\bSTEVEN\b", "personal note to self"),
    (r"THIS MODULE IS YOURS", "development handoff note"),
    (r"\bHint:", "implementation hint left in a docstring"),
    (r"raise NotImplementedError", "unimplemented stub"),
    (r"\bTODO\b", "unfinished work marker"),
    (r"\bFIXME\b", "known-broken marker"),
    (r"\bXXX\b", "scratch marker"),
    (r"^\s*(print|breakpoint)\(.*debug", "leftover debug statement"),
]

# Files allowed to contain the markers. This script names them by necessity.
ALLOWLIST = {"check_publish_ready.py"}


def iter_source_files():
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix not in {".py", ".md", ".yaml", ".yml", ".sql"}:
            continue
        if path.name in ALLOWLIST:
            continue
        yield path


def main() -> int:
    findings: list[tuple[Path, int, str, str]] = []

    for path in iter_source_files():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue

        for number, line in enumerate(lines, start=1):
            for pattern, reason in PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append((path, number, line.strip(), reason))
                    break

    print()
    print("statcast-advance :: publish readiness")
    print("=" * 62)

    if not findings:
        print("\nNo scaffolding found. Safe to push.\n")
        return 0

    by_file: dict[Path, list] = {}
    for path, number, text, reason in findings:
        by_file.setdefault(path, []).append((number, text, reason))

    print(f"\n{len(findings)} item(s) to clean up across {len(by_file)} file(s):\n")

    for path, items in by_file.items():
        print(f"  {path.relative_to(ROOT)}")
        for number, text, reason in items:
            snippet = text if len(text) <= 60 else text[:57] + "..."
            print(f"    line {number:<5} {reason}")
            print(f"              {snippet}")
        print()

    print("Unimplemented stubs are expected mid-phase. The ones worth acting on")
    print("are notes-to-self and docstring hints, which stop being useful the")
    print("moment the function they describe exists.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
