"""End-to-end check: fetch -> clean -> store -> read back.

The unit tests prove each piece works in isolation against invented data.
This proves the pieces work together against real data, which is a different
question and the one that actually breaks.

It also demonstrates the cache: run it twice and the second run should skip
the network entirely.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --pitcher "Zack Wheeler" --season 2025
    python scripts/smoke_test.py --force-fetch     # ignore the cache
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import clean, config, fetch, store  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

problems: list[str] = []


def check(condition: bool, description: str, detail: str = "") -> None:
    """Assert-with-a-report-card, so one failure does not hide the rest."""
    if condition:
        print(f"  [ok]  {description}")
    else:
        print(f"  [XX]  {description}" + (f" -- {detail}" if detail else ""))
        problems.append(description)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pitcher", default="Tarik Skubal")
    parser.add_argument("--season", type=int, default=config.get("data.default_season", 2025))
    parser.add_argument("--force-fetch", action="store_true", help="Bypass the cache")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")

    print("\nstatcast-advance :: end-to-end smoke test")
    print("=" * 60)

    # ---------------------------------------------------------------- resolve
    print(f"\n1. Resolving '{args.pitcher}'")
    try:
        player = fetch.resolve_player(args.pitcher)
    except fetch.PlayerNotFound as exc:
        print(f"  [XX]  {exc}\n")
        return 1
    print(f"  [ok]  {player.full_name}, MLBAM {player.mlbam_id}")

    conn = store.connect()

    # ------------------------------------------------------------------ fetch
    print(f"\n2. Getting {args.season} data")
    cached = store.is_cached(conn, player.mlbam_id, args.season) and not args.force_fetch

    if cached:
        print("  [ok]  Cache hit - skipping the network")
        stored = store.load_pitcher_season(conn, player.mlbam_id, args.season)
        raw = None
    else:
        fixture = FIXTURE_DIR / f"{player.last_name.lower()}_{args.season}_raw.csv.gz"
        if fixture.exists() and not args.force_fetch:
            print(f"  [ok]  Reading saved fixture ({fixture.name})")
            raw = pd.read_csv(fixture, low_memory=False)
        else:
            start = time.time()
            raw = fetch.fetch_player_season(player.mlbam_id, args.season)
            print(f"  [ok]  Fetched {len(raw):,} pitches in {time.time() - start:.1f}s")

        if raw.empty:
            print(f"  [XX]  No pitches found for {player.full_name} in {args.season}\n")
            return 1

        # -------------------------------------------------------------- clean
        print("\n3. Cleaning")
        cleaned = clean.prepare_for_storage(raw)

        print(f"  [ok]  {len(raw):,} raw -> {len(cleaned):,} cleaned "
              f"({len(raw) - len(cleaned):,} removed)")

        from src.store import PITCH_SCHEMA
        check(
            list(cleaned.columns) == list(PITCH_SCHEMA.keys()),
            "Columns match the storage schema",
            f"got {len(cleaned.columns)}, expected {len(PITCH_SCHEMA)}",
        )
        check(
            cleaned[["game_pk", "at_bat_number", "pitch_number"]].notna().all().all(),
            "No nulls in the primary key",
        )
        check(
            not cleaned.duplicated(subset=["game_pk", "at_bat_number", "pitch_number"]).any(),
            "Primary key is unique",
        )

        # -------------------------------------------------------------- store
        print("\n4. Storing")
        written = store.save_pitches(conn, cleaned, player.mlbam_id, args.season)
        print(f"  [ok]  Wrote {written:,} rows to DuckDB")

        stored = store.load_pitcher_season(conn, player.mlbam_id, args.season)
        check(len(stored) == written, "Round trip preserved every row",
              f"wrote {written}, read back {len(stored)}")

    # ------------------------------------------------------- validate content
    print("\n5. Validating what came back")

    check(len(stored) > 0, "Cache returned rows")
    if len(stored) == 0:
        print()
        return 1

    check(
        stored["balls"].between(0, 3).all() and stored["strikes"].between(0, 2).all(),
        "All counts are legal",
    )
    check(
        stored["pitch_type"].notna().all(),
        "No null pitch types survived cleaning",
    )
    check(
        not stored["pitch_type"].isin(["FT", "FA", "KC", "CS", "SV", "FO"]).any(),
        "Deprecated pitch codes were consolidated",
    )
    check(
        not stored["game_type"].isin(["S", "E", "A"]).any(),
        "Spring training, exhibition, and All-Star games excluded",
    )
    check(
        stored["release_speed"].between(50, 110).all(),
        "Velocities are physically plausible",
        f"range {stored['release_speed'].min():.1f}-{stored['release_speed'].max():.1f}",
    )

    # --------------------------------------------------- prove SQL works
    print("\n6. Sample query - pitch mix by count")
    print("   (this is the shape Phase 2 builds on)")

    mix = conn.execute(
        """
        SELECT
            balls || '-' || strikes            AS count,
            COUNT(*)                            AS pitches,
            ROUND(100.0 * SUM(CASE WHEN pitch_type IN ('FF','SI','FC')
                                   THEN 1 ELSE 0 END) / COUNT(*), 1) AS fastball_pct
        FROM pitches
        WHERE pitcher = ? AND game_year = ?
        GROUP BY balls, strikes
        ORDER BY balls, strikes
        """,
        [player.mlbam_id, args.season],
    ).df()

    print()
    print("     count   pitches   fastball%")
    for row in mix.itertuples():
        print(f"     {row.count:>5}   {row.pitches:>7}   {row.fastball_pct:>8.1f}")

    check(len(mix) > 0, "SQL aggregation returned results")
    check(len(mix) <= 12, "No impossible counts in the data", f"got {len(mix)} distinct counts")

    # ------------------------------------------------------------- summary
    print("\n" + "=" * 60)
    print(f"{player.full_name}, {args.season}: {len(stored):,} pitches cached")

    arsenal = (
        stored.groupby("pitch_type").size().sort_values(ascending=False)
        / len(stored) * 100
    )
    print("Arsenal: " + ", ".join(f"{p} {v:.0f}%" for p, v in arsenal.head(6).items()))

    # Close before measuring: DuckDB buffers writes, so the file on disk is
    # still near-empty until the connection is closed and flushed.
    conn.close()

    db_path = config.resolve_path("data.db_path")
    if db_path.exists():
        print(f"Database: {db_path.stat().st_size / 1_048_576:.1f} MB")

    if problems:
        print(f"\n{len(problems)} check(s) failed:")
        for item in problems:
            print(f"  - {item}")
        print()
        return 1

    print("\nAll checks passed. Phase 1 complete - ready for Phase 2.\n")
    print("Run it again to confirm the cache skips the network.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
