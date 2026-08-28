"""Build the league reference sample and print what it contains.

Fetches a stratified sample of dates across a season and stores them in the
local DuckDB cache. Roughly seven minutes on a first run, given the rate limit
between requests. Interrupting is safe — already-fetched dates are skipped, so
rerunning resumes where it stopped.

Usage:
    python scripts/build_baselines.py
    python scripts/build_baselines.py --season 2025 --n-dates 30
    python scripts/build_baselines.py --show-only     # skip fetching
"""

from __future__ import annotations

import argparse
import logging
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
    parser.add_argument("--n-dates", type=int, default=None)
    parser.add_argument("--show-only", action="store_true",
                        help="Report on the existing sample without fetching")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    conn = store.connect()

    print(f"\nstatcast-advance :: league baselines, {args.season}")
    print("=" * 62)

    if not args.show_only:
        dates = baselines.sample_dates(args.season, args.n_dates)
        print(f"\nStratified sample: {len(dates)} dates from "
              f"{dates[0]} to {dates[-1]}\n")
        added = baselines.build(conn, args.season, args.n_dates)
        print(f"\nAdded {added:,} pitches")

    summary = baselines.sample_summary(conn, args.season)
    if summary.empty or summary.iloc[0]["pitches"] == 0:
        print("\nNo league data stored yet. Run without --show-only first.\n")
        conn.close()
        return 1

    row = summary.iloc[0]
    print(f"\nSample: {int(row['pitches']):,} pitches, "
          f"{int(row['pitchers']):,} pitchers, "
          f"{int(row['dates'])} dates "
          f"({row['first_date']} to {row['last_date']})")

    for hand, label in (("R", "RIGHT-HANDED"), ("L", "LEFT-HANDED")):
        print(f"\n\n{label} PITCHERS")
        print("=" * 62)

        outcomes = baselines.pitch_outcomes(conn, args.season, hand)
        if outcomes.empty:
            print("  not enough data")
            continue

        print("\nPitch type rates")
        print(outcomes.to_string(index=False))

        print("\nTwo-strike putaway")
        print(baselines.putaway_rates(conn, args.season, hand).to_string(index=False))

        print("\nMost fastball-heavy counts (share of the top pitch)")
        mix = baselines.count_mix(conn, args.season, hand)
        top = mix[mix["rank_in_count"] == 1].sort_values("league_share",
                                                         ascending=False)
        print(top[["count", "pitch_type", "league_share", "count_total"]]
              .head(6).to_string(index=False))

    conn.close()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
