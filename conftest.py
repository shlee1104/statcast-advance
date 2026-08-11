"""Pytest configuration.

Puts the project root on sys.path so tests can `from src import clean`
without the package needing to be pip-installed.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
