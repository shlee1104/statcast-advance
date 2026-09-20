"""Single-command report generation.

    python -m src.cli --pitcher "Yoshinobu Yamamoto" --season 2025

Player name in, coach-ready report out. Everything between those two points —
resolving the name to an MLBAM id, fetching or reading the cache, cleaning,
computing every metric, running the flags, rendering — happens without further
input, because a tool that needs five commands in the right order is a tool
that gets used once.

League baselines are used when they exist and skipped when they do not. A
report without them is thinner but still correct, and refusing to produce one
would make the first run of this project fail.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src import baselines, clean, config, report, store
from src.metrics import events

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "tests" / "fixtures"


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Run the cleaning pipeline and attach the derived outcome columns.

    Uses the individual cleaning steps rather than prepare_for_storage(),
    which reindexes to the database schema and drops the derived count columns
    the metrics need.
    """
    for step in (
        clean.consolidate_pitch_types,
        clean.drop_non_competitive,
        clean.filter_game_types,
        clean.add_count_state,
    ):
        frame = step(frame)
    return events.add_event_flags(frame)


def load_from_fixture(stem: str, season: int) -> pd.DataFrame:
    """Read a committed fixture, so the CLI works offline."""
    path = FIXTURE_DIR / f"{stem}_{season}_raw.csv.gz"
    if not path.exists():
        available = sorted(p.name for p in FIXTURE_DIR.glob("*_raw.csv.gz"))
        raise SystemExit(
            f"No fixture at {path.name}.\n"
            f"Available: {', '.join(available) or 'none'}"
        )
    return pd.read_csv(path, low_memory=False)


def load_from_savant(pitcher: str, season: int, refresh: bool = False) -> pd.DataFrame:
    """Resolve the name, then read the cache or fetch and cache.

    Cached rows come back through the schema, which means they have already
    been coerced to the stored column set. A fresh fetch returns Savant's raw
    frame. Both go through the same cleaning pipeline afterwards, so the two
    paths converge before any metric sees them.
    """
    from src import fetch

    player = fetch.resolve_player(pitcher)
    print(f"  Resolved to MLBAM {player.mlbam_id} ({player.full_name})")

    conn = store.connect()
    try:
        # A completed season never expires, which is correct — the data cannot
        # change. But it also means that widening PITCH_SCHEMA leaves cached
        # rows with NULLs in the new columns forever, since nothing will ever
        # decide to re-fetch them. --refresh is the escape hatch for that.
        if not refresh and store.is_cached(conn, player.mlbam_id, season):
            frame = store.load_pitcher_season(conn, player.mlbam_id, season)
            if len(frame):
                print(f"  Cache hit: {len(frame):,} pitches")
                return frame
            # A logged fetch with no rows means the cache is lying; fall
            # through and re-fetch rather than reporting on an empty season.
            log.warning("Cache reported a hit but returned no rows; refetching")

        print("  Fetching from Baseball Savant (5-20s)...")
        frame = fetch.fetch_player_season(player.mlbam_id, season)
        if len(frame):
            store.save_pitches(conn, frame, player.mlbam_id, season)
        return frame
    finally:
        conn.close()


def load_league(
    season: int,
    hand: str,
) -> dict[str, pd.DataFrame | None]:
    """Read league baselines from the cache, or return Nones if absent."""
    blank = {"league_outcomes": None, "league_mix": None, "league_putaway": None}

    try:
        conn = store.connect()
    except Exception as exc:  # cache missing or unreadable
        log.warning("No league cache available (%s)", exc)
        return blank

    try:
        summary = baselines.sample_summary(conn, season)
        if summary.empty or not summary.iloc[0]["pitches"]:
            print("  No league sample built — run scripts/build_baselines.py "
                  "for deviation columns")
            return blank

        row = summary.iloc[0]
        print(f"  League sample: {int(row['pitches']):,} pitches "
              f"from {int(row['dates'])} dates ({hand}HP)")
        return {
            "league_outcomes": baselines.pitch_outcomes(conn, season, hand),
            "league_mix": baselines.count_mix(conn, season, hand),
            "league_putaway": baselines.putaway_rates(conn, season, hand),
        }
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pitcher", required=True,
                        help='Full name, e.g. "Yoshinobu Yamamoto"')
    parser.add_argument("--season", type=int,
                        default=config.get("data.default_season", 2025))
    parser.add_argument("--fixture", metavar="STEM",
                        help="Read a committed fixture instead of fetching "
                             "(e.g. --fixture yamamoto). Works offline.")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-fetch even if cached. Needed after new "
                             "columns are added to the schema, since a "
                             "completed season never expires on its own.")
    parser.add_argument("--no-league", action="store_true",
                        help="Skip league baselines even if they exist")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path (default: reports/<name>_<season>.html)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s  %(message)s",
    )

    print(f"\n{args.pitcher} — {args.season}")

    if args.fixture:
        raw = load_from_fixture(args.fixture, args.season)
        print(f"  Fixture: {len(raw):,} raw rows")
    else:
        raw = load_from_savant(args.pitcher, args.season, refresh=args.refresh)
        print(f"  Fetched: {len(raw):,} raw rows")

    frame = prepare(raw)
    if len(frame) == 0:
        raise SystemExit("No competitive pitches left after cleaning.")

    hand = str(frame["p_throws"].mode().iloc[0])
    print(f"  Cleaned: {len(frame):,} pitches, throws {hand}")

    league = (
        {"league_outcomes": None, "league_mix": None, "league_putaway": None}
        if args.no_league
        else load_league(args.season, hand)
    )

    path = report.write(frame, args.pitcher, args.season,
                        out_path=args.out, **league)

    size_kb = path.stat().st_size / 1024
    try:
        shown = path.relative_to(Path.cwd())
    except ValueError:
        shown = path
    print(f"\n  Wrote {shown}  ({size_kb:,.0f} KB)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
