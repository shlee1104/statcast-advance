"""Run the metrics modules against a real pitcher and print the results.

An interactive workbench for the metrics layer, until the report layer exists.
Reads a saved fixture, runs the cleaning pipeline, computes every metric, and
joins each one to league baselines for the same pitcher handedness so the
numbers read as findings rather than facts.

Falls back to bare metrics if no league sample has been built yet; run
scripts/build_baselines.py first for the comparison columns.

Usage:
    python scripts/explore.py
    python scripts/explore.py --pitcher skubal
    python scripts/explore.py --no-league
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import baselines, clean, store  # noqa: E402
from src.metrics import (  # noqa: E402
    arsenal, counts, events, location, sequencing, splits,
)

FIXTURE_DIR = ROOT / "tests" / "fixtures"


def load(pitcher: str, season: int) -> pd.DataFrame:
    """Read a fixture and run it through the full cleaning pipeline.

    Uses the individual cleaning steps rather than prepare_for_storage(),
    because that function reindexes to the database schema and drops the
    derived count columns the metrics need.
    """
    path = FIXTURE_DIR / f"{pitcher}_{season}_raw.csv.gz"
    if not path.exists():
        available = sorted(p.name for p in FIXTURE_DIR.glob("*_raw.csv.gz"))
        raise SystemExit(
            f"No fixture at {path.name}.\n"
            f"Available: {', '.join(available) or 'none'}\n"
            f"Create one with: python scripts/fetch_fixture.py --pitcher \"Name\""
        )

    frame = pd.read_csv(path, low_memory=False)
    frame = clean.consolidate_pitch_types(frame)
    frame = clean.drop_non_competitive(frame)
    frame = clean.filter_game_types(frame)
    frame = clean.add_count_state(frame)
    frame = events.add_event_flags(frame)
    return frame


def show(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pitcher", default="yamamoto", help="fixture name stem")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--min-n", type=int, default=25, help="setup_pairs sample gate")
    parser.add_argument("--no-league", action="store_true",
                        help="Skip league comparisons")
    args = parser.parse_args()

    pd.set_option("display.width", 150)
    pd.set_option("display.max_columns", 24)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    frame = load(args.pitcher, args.season)
    hand = frame["p_throws"].mode().iloc[0]
    print(f"\n{args.pitcher.title()} {args.season}: {len(frame):,} pitches, "
          f"throws {hand}")

    # Load league reference data for this handedness, if it has been built.
    league_outcomes = league_putaway = league_mix = None
    if not args.no_league:
        conn = store.connect()
        summary = baselines.sample_summary(conn, args.season)
        if not summary.empty and summary.iloc[0]["pitches"]:
            row = summary.iloc[0]
            print(f"League sample: {int(row['pitches']):,} pitches from "
                  f"{int(row['dates'])} dates, {hand}HP baselines")
            league_outcomes = baselines.pitch_outcomes(conn, args.season, hand)
            league_putaway = baselines.putaway_rates(conn, args.season, hand)
            league_mix = baselines.count_mix(conn, args.season, hand)
        else:
            print("No league sample yet — run scripts/build_baselines.py")
        conn.close()

    # ---------------------------------------------------------------- arsenal
    mix = counts.pitch_mix(frame)
    show("ARSENAL vs LEAGUE" if league_outcomes is not None else "ARSENAL")
    if league_outcomes is not None:
        table = baselines.compare_arsenal(mix, league_outcomes)
        print(table[["pitch_type", "usage", "league_usage", "usage_delta",
                     "usage_ratio"]].to_string(index=False))
    else:
        for pitch, share in mix.items():
            print(f"  {pitch:<4} {share:6.1%}")

    print(f"\n  arsenal (>=1% usage): {', '.join(counts.primary_arsenal(frame))}")

    show("PITCH CHARACTERISTICS" + (" vs LEAGUE" if league_outcomes is not None else ""))
    profile = arsenal.profile(frame)
    if league_outcomes is not None:
        table = baselines.compare_profile(profile, league_outcomes)
        print(table[["pitch_type", "n", "usage", "velo", "velo_delta",
                     "whiff_rate", "whiff_delta", "xwoba", "xwoba_delta"]]
              .to_string(index=False))
        print("\n  xwoba_delta is signed so positive is bad for the pitcher:")
        print("  hitters are doing more damage than against the league's version.")
    else:
        print(profile[
            ["n", "velo", "velo_max", "spin", "h_break", "v_break",
             "zone_rate", "whiff_rate", "chase_rate", "xwoba"]
        ].to_string())

    show("RELEASE CONSISTENCY (tipping check)")
    print(arsenal.release_consistency(frame)[
        ["n", "rel_x", "rel_z", "rel_x_sd", "rel_z_sd",
         "arm_angle", "dist_from_ref"]
    ].to_string())
    print("\n  dist_from_ref is inches from the most-used pitch's release point.")
    print("  An offering released well off that slot can be identified early.")

    # -------------------------------------------------------------- location
    show("LOCATION PROFILE")
    print(location.location_profile(frame).to_string())
    print("\n  arm_side is inches toward the pitcher's arm side; height is a")
    print("  fraction of the batter's own zone, 0.0 bottom and 1.0 top.")
    print("  The two sd columns are the command read.")

    show("QUADRANT SHARES")
    print(location.quadrant_shares(frame).to_string())

    show("LOCATION LEAKS")
    leaks = location.location_leaks(frame)
    if len(leaks) == 0:
        print("  nothing clears the concentration threshold")
    else:
        print(leaks.to_string(index=False))
        print("\n  excess is share above the 0.25 a uniform pitcher would show.")

    # ---------------------------------------------------------------- splits
    show("PLATOON GAPS")
    print(splits.platoon_gaps(frame).to_string(index=False))

    show("FATIGUE (fastball velocity by pitch count)")
    print(splits.fatigue(frame, bucket_size=15).to_string(index=False))

    show("TIMES THROUGH THE ORDER")
    tto = splits.times_through_order(frame)
    if len(tto) == 0:
        print("  n_thruorder_pitcher not present in this frame")
    else:
        print(tto.to_string(index=False))
        decomposition = splits.decompose_tto(frame)
        print(f"\n  velocity change 1st -> 3rd : {decomposition['velo_delta']:+.2f} mph")
        print(f"  xwOBA change 1st -> 3rd    : {decomposition['xwoba_delta']:+.3f}")
        print(f"  attribution                : {decomposition['attribution']}")
        print(f"  {decomposition['note']}")

    # -------------------------------------------------------- predictability
    pred = counts.predictability(frame, min_pitches=20)
    show("PREDICTABILITY BY COUNT" + (" vs LEAGUE" if league_mix is not None else ""))
    if league_mix is not None:
        table = baselines.compare_predictability(pred, league_mix)
        print(table[["count", "n", "top_pitch", "top_share", "league_top_share",
                     "share_delta", "share_z", "predictability"]]
              .to_string(index=False))
        print("\n  share_z is standard errors from the league rate; |z| > 2 is")
        print("  worth reporting, and a large delta on a small n will not clear it.")
    else:
        print(pred.to_string())

    # ------------------------------------------------------------ first pitch
    show("FIRST PITCH")
    first = counts.first_pitch_tendencies(frame)
    print(f"  n = {first['n']}, strike rate = {first['strike_rate']:.1%}")
    print(f"  primary = {first['primary_pitch']} at {first['primary_share']:.1%}")

    # ---------------------------------------------------------------- putaway
    putaway = sequencing.putaway(frame, min_pitches=10)
    show("TWO-STRIKE PUTAWAY" + (" vs LEAGUE" if league_putaway is not None else ""))
    if league_putaway is not None:
        table = baselines.compare_putaway(putaway, league_putaway)
        print(table[["pitch_type", "n", "usage", "league_usage", "usage_ratio",
                     "putaway_rate", "league_putaway", "putaway_delta",
                     "putaway_z"]].to_string(index=False))
    else:
        print(putaway.to_string())

    # ----------------------------------------------------------- setup pairs
    show(f"SETUP PAIRS (all counts, min_n={args.min_n})")
    setups = sequencing.setup_pairs(frame, min_n=args.min_n)
    cols = ["setup_pitch", "setup_band", "next_pitch", "n",
            "p_next", "baseline_p", "freq_lift", "effect_lift", "score"]
    print(setups[cols].head(10).to_string(index=False))

    for state in ("ahead", "even", "behind"):
        show(f"SETUP PAIRS ({state} in the count)")
        subset = sequencing.setup_pairs(frame, min_n=max(10, args.min_n // 2),
                                        count_state=state)
        if len(subset) == 0:
            print("  nothing clears the sample gate")
            continue
        print(subset[cols].head(5).to_string(index=False))

    show("TRANSITION MATRIX")
    print(sequencing.transition_matrix(frame, min_transitions=20).to_string())

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
