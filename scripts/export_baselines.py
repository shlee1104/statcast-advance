"""Dump the league reference tables to CSV.

The baselines live in the DuckDB cache, which means anything that cannot open
that file cannot see them. This writes the three reference tables out as plain
CSV so they can be read anywhere — for inspection, for diffing between
rebuilds, or for working on the report layer without a database to hand.

Reads only. Nothing here touches the cache.

Usage:
    python scripts/export_baselines.py
    python scripts/export_baselines.py --season 2025 --throws L
    python scripts/export_baselines.py --out-dir data/exports
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import baselines, config, store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int,
                        default=config.get("data.default_season", 2025))
    parser.add_argument("--throws", default="R", choices=["R", "L"],
                        help="Pitcher handedness the baselines are computed for")
    parser.add_argument("--out-dir", default="data/exports")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = store.connect()
    try:
        summary = baselines.sample_summary(conn, args.season)
        if summary.empty or not summary.iloc[0]["pitches"]:
            print("No league sample in the cache yet.")
            print("Run scripts/build_baselines.py first.")
            return 1

        row = summary.iloc[0]
        print(f"League sample: {int(row['pitches']):,} pitches "
              f"from {int(row['dates'])} dates")

        tables = {
            "pitch_outcomes": baselines.pitch_outcomes(conn, args.season, args.throws),
            "putaway_rates": baselines.putaway_rates(conn, args.season, args.throws),
            "count_mix": baselines.count_mix(conn, args.season, args.throws),
        }
    finally:
        conn.close()

    for name, table in tables.items():
        path = out_dir / f"league_{name}_{args.season}_{args.throws}.csv"
        table.to_csv(path, index=False)
        print(f"  {path.relative_to(ROOT)}  ({len(table)} rows, "
              f"{len(table.columns)} columns)")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
