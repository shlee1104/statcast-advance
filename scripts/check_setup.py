"""Phase 0 verification.

Confirms the environment is ready before any real code gets written:
Python version, required packages, Baseball Savant reachability, and
that the config file parses.

Run from the project root:
    python scripts/check_setup.py
"""

from __future__ import annotations

import sys
import time
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PASS = "  [ok]  "
FAIL = "  [XX]  "
WARN = "  [--]  "

failures: list[str] = []


def check_python() -> None:
    major, minor = sys.version_info[:2]
    version = f"{major}.{minor}.{sys.version_info[2]}"
    if (major, minor) < (3, 10):
        print(f"{FAIL}Python {version} — need 3.10 or newer")
        failures.append(
            "Python is too old. Install 3.11 or 3.12 (see README), then rebuild the venv."
        )
    elif (major, minor) >= (3, 14):
        print(f"{WARN}Python {version} — newer than tested; pybaseball may not have wheels yet")
    else:
        print(f"{PASS}Python {version}")

    # Make sure we are inside a virtual environment, not the system Python.
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        print(f"{PASS}Running inside a virtual environment")
    else:
        print(f"{WARN}Not in a virtual environment — activate .venv first")


def check_imports() -> None:
    packages = [
        ("pandas", "pandas"),
        ("duckdb", "duckdb"),
        ("requests", "requests"),
        ("yaml", "pyyaml"),
        ("plotly", "plotly"),
        ("jinja2", "jinja2"),
        ("pybaseball", "pybaseball"),
        ("pytest", "pytest"),
    ]
    for module, pip_name in packages:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "")
            print(f"{PASS}{pip_name} {version}".rstrip())
        except ImportError:
            print(f"{FAIL}{pip_name} not installed")
            failures.append(f"Missing package: {pip_name}. Run: pip install -r requirements.txt")


def check_config() -> None:
    path = ROOT / "config.yaml"
    if not path.exists():
        print(f"{FAIL}config.yaml not found")
        failures.append("config.yaml is missing from the project root.")
        return
    try:
        import yaml

        with path.open() as handle:
            cfg = yaml.safe_load(handle)
        season = cfg["data"]["default_season"]
        gates = len(cfg["flags"]["min_n"])
        print(f"{PASS}config.yaml parses (default season {season}, {gates} sample gates)")
    except Exception as exc:  # noqa: BLE001 - surface anything to the user
        print(f"{FAIL}config.yaml failed to parse: {exc}")
        failures.append("config.yaml is not valid YAML.")


def check_savant() -> None:
    """Pull a single day of pitch data straight from the Savant CSV endpoint.

    This is the same endpoint the real fetch client will use, so if this
    works, Phase 1 is unblocked.
    """
    try:
        import pandas as pd
        import requests
    except ImportError:
        print(f"{WARN}Skipping Savant check — install packages first")
        return

    url = "https://baseballsavant.mlb.com/statcast_search/csv"
    params = {
        "all": "true",
        "hfGT": "R|",
        "game_date_gt": "2025-04-15",
        "game_date_lt": "2025-04-15",
        "type": "details",
    }

    try:
        start = time.time()
        response = requests.get(
            url,
            params=params,
            timeout=120,
            headers={"User-Agent": "statcast-advance/0.1 (personal project)"},
        )
        response.raise_for_status()
        frame = pd.read_csv(StringIO(response.text), low_memory=False)
        elapsed = time.time() - start
        if frame.empty:
            print(f"{FAIL}Savant returned an empty result")
            failures.append("Savant responded but returned no rows. Try a different date.")
            return
        print(f"{PASS}Savant reachable — {len(frame):,} pitches, {len(frame.columns)} columns, {elapsed:.1f}s")

        # Confirm the columns the project depends on are actually present.
        needed = [
            "pitch_type",
            "release_speed",
            "balls",
            "strikes",
            "stand",
            "p_throws",
            "game_pk",
            "at_bat_number",
            "pitch_number",
            "description",
            "estimated_woba_using_speedangle",
        ]
        missing = [c for c in needed if c not in frame.columns]
        if missing:
            print(f"{FAIL}Missing expected columns: {', '.join(missing)}")
            failures.append("Savant schema differs from what the plan assumes.")
        else:
            print(f"{PASS}All {len(needed)} required columns present")
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL}Could not reach Savant: {type(exc).__name__}: {exc}")
        failures.append("Baseball Savant is unreachable. Check your internet connection.")


def check_id_lookup() -> None:
    try:
        from pybaseball import playerid_lookup
    except ImportError:
        print(f"{WARN}Skipping ID lookup check — pybaseball not installed")
        return
    try:
        result = playerid_lookup("skubal", "tarik")
        if len(result) == 0:
            print(f"{FAIL}Player ID lookup returned nothing")
            failures.append("pybaseball ID lookup is not working.")
        else:
            mlbam = int(result.iloc[0]["key_mlbam"])
            print(f"{PASS}Player ID lookup works (Tarik Skubal = {mlbam})")
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL}Player ID lookup failed: {type(exc).__name__}: {exc}")
        failures.append("pybaseball ID lookup failed. It downloads a register on first use.")


def main() -> int:
    print()
    print("statcast-advance :: Phase 0 setup check")
    print("=" * 55)

    print("\nEnvironment")
    check_python()

    print("\nPackages")
    check_imports()

    print("\nConfiguration")
    check_config()

    print("\nData source (this may take up to a minute)")
    check_savant()
    check_id_lookup()

    print("\n" + "=" * 55)
    if failures:
        print(f"{len(failures)} problem(s) to fix:\n")
        for i, problem in enumerate(failures, 1):
            print(f"  {i}. {problem}")
        print()
        return 1

    print("Everything checks out. Phase 0 complete — ready for Phase 1.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
