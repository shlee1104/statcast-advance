"""Run the metrics modules against a real pitcher and print the results.

An interactive workbench for the metrics layer, until the report layer in
Phase 3 exists. Reads a saved fixture, runs the cleaning pipeline, and prints
each metric so the numbers can be sanity-checked against Baseball Savant by
eye.

Usage:
    python scripts/explore.py
    python scripts/explore.py --pitcher skubal
    python scripts/explore.py --min-n 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import clean  # noqa: E402
from src.metrics import counts, events, sequencing  # noqa: E402

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
    args = parser.parse_args()

    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    frame = load(args.pitcher, args.season)
    print(f"\n{args.pitcher.title()} {args.season}: {len(frame):,} pitches")

    show("ARSENAL")
    mix = counts.pitch_mix(frame)
    for pitch, share in mix.items():
        print(f"  {pitch:<4} {share:6.1%}")

    show("PREDICTABILITY BY COUNT")
    pred = counts.predictability(frame, min_pitches=20)
    print(pred.to_string())

    show("FIRST PITCH")
    first = counts.first_pitch_tendencies(frame)
    print(f"  n = {first['n']}, strike rate = {first['strike_rate']:.1%}")
    print(f"  primary = {first['primary_pitch']} at {first['primary_share']:.1%}")

    show("TWO-STRIKE PUTAWAY")
    print(sequencing.putaway(frame, min_pitches=10).to_string())

    show(f"SETUP PAIRS (all counts, min_n={args.min_n})")
    setups = sequencing.setup_pairs(frame, min_n=args.min_n)
    cols = ["setup_pitch", "setup_band", "next_pitch", "n",
            "p_next", "baseline_p", "freq_lift", "effect_lift", "score"]
    print(setups[cols].head(12).to_string(index=False))

    for state in ("ahead", "even", "behind"):
        show(f"SETUP PAIRS ({state} in the count)")
        subset = sequencing.setup_pairs(frame, min_n=max(10, args.min_n // 2),
                                        count_state=state)
        if len(subset) == 0:
            print("  nothing clears the sample gate")
            continue
        print(subset[cols].head(6).to_string(index=False))

    show("TRANSITION MATRIX")
    print(sequencing.transition_matrix(frame, min_transitions=20).to_string())

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
