"""Save one real pitcher-season as a test fixture.

Everything up to now has been tested against small hand-built frames. Those
catch logic errors but say nothing about whether real Savant data has the
shape the code assumes.

This script pulls one pitcher-season and saves the raw, untouched response to
tests/fixtures/. From then on the whole project can be developed and tested
offline against genuine data — which matters because Phase 2 is a dozen metric
functions, and none of them should require a live API call to test.

Usage:
    python scripts/fetch_fixture.py
    python scripts/fetch_fixture.py --pitcher "Zack Wheeler" --season 2025
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, fetch  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pitcher",
        default="Yoshinobu Yamamoto",
        help="Pitcher name. Default is a durable starter with a full season "
             "and a five-pitch mix, which exercises more code paths than a "
             "reliever would.",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=config.get("data.default_season", 2025),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    print(f"\nResolving '{args.pitcher}'...")
    try:
        player = fetch.resolve_player(args.pitcher)
    except fetch.PlayerNotFound as exc:
        print(f"\nERROR: {exc}\n")
        return 1

    print(f"  -> {player.full_name}, MLBAM {player.mlbam_id}")

    print(f"\nFetching {args.season} season from Savant (this takes 5-20s)...")
    try:
        frame = fetch.fetch_player_season(player.mlbam_id, args.season)
    except fetch.SavantError as exc:
        print(f"\nERROR: {exc}\n")
        return 1

    if frame.empty:
        print(f"\nERROR: no pitches found for {player.full_name} in {args.season}.")
        print("Check that he actually pitched that season.\n")
        return 1

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    slug = player.last_name.lower().replace(" ", "_")
    out_path = FIXTURE_DIR / f"{slug}_{args.season}_raw.csv.gz"

    # Gzip keeps the repo small; pandas reads it back transparently.
    frame.to_csv(out_path, index=False, compression="gzip")

    size_mb = out_path.stat().st_size / 1_048_576

    print(f"\nSaved {out_path.relative_to(FIXTURE_DIR.parent.parent)}")
    print(f"  {len(frame):,} pitches")
    print(f"  {len(frame.columns)} columns")
    print(f"  {size_mb:.2f} MB on disk")

    # A quick look at what actually came back, so problems surface here
    # rather than three modules downstream.
    print("\nRaw pitch_type values (before consolidation):")
    counts = frame["pitch_type"].value_counts(dropna=False)
    for code, n in counts.items():
        print(f"  {str(code):>6}  {n:>5}")

    print("\nGame types present:")
    for code, n in frame["game_type"].value_counts(dropna=False).items():
        print(f"  {str(code):>6}  {n:>5}")

    missing_pct = frame.isna().mean()
    concerning = missing_pct[missing_pct > 0.5].sort_values(ascending=False)
    if len(concerning) > 0:
        print(f"\n{len(concerning)} columns are more than 50% null "
              f"(usually fine - most are batted-ball fields):")
        for name, pct in concerning.head(8).items():
            print(f"  {name:<38} {pct:.0%}")

    print("\nFixture saved. Commit it so tests run offline.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
