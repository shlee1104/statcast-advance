"""Check whether the hitting plan holds up on games it never saw.

For each pitcher on the calibration list, the plan is built from one part of
his season and every call is checked against the rest (see src/validation.py
for what is measured and why). Writes to reports/holdout/:

  holdout_calls.csv    one row per pitcher per rule: the call each part made,
                       and the number behind it
  holdout_summary.csv  one row per rule: how often the advice held, how often
                       it flipped, and how much better the pitcher's own
                       number predicts than the average pitcher's

and prints the summary.

Usage:
    python scripts/holdout.py                          # first half vs second half
    python scripts/holdout.py --split postseason       # regular season vs October
    python scripts/holdout.py --season 2026
    python scripts/holdout.py --names "Blake Snell" "Logan Webb"
    python scripts/holdout.py --offline                # committed fixtures only
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import cli, config, validation  # noqa: E402

OUT_DIR = ROOT / "reports" / "holdout"


def _batch():
    """The calibration script's loaders, so both runs read pitchers the same way."""
    spec = importlib.util.spec_from_file_location(
        "batch_reports", ROOT / "scripts" / "batch_reports.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fmt(v):
    if isinstance(v, float):
        return "" if v != v else round(v, 3)
    return v


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _fmt(r.get(k, "")) for k in keys})


def print_summary(summary: list[dict], split: str, pitchers: int) -> None:
    what = ("first half → second half" if split == "halves"
            else "regular season → postseason")
    print(f"\nDoes the plan hold up?  {what}, {pitchers} pitchers")
    print("-" * 78)
    def pct(v):
        return f"{v:.0%}" if v == v else "—"

    print(f"  {'Rule':<34}{'advice':>7}{'held':>7}{'weaker':>8}{'flipped':>9}{'skill':>8}")
    for s in summary:
        skill = f"{s['skill']:+.2f}" if s["skill"] == s["skill"] else "—"
        print(f"  {s['label']:<34}{s['advice_given']:>7}{pct(s['held_pct']):>7}"
              f"{pct(s['weakened_pct']):>8}{pct(s['flipped_pct']):>9}{skill:>8}")
    two = next((s for s in summary if s["rule"] == "two_strikes"), None)
    if two and two["flipped_broad_pct"] == two["flipped_broad_pct"]:
        print(f"\n  Two strikes, counting \"expect the X\" → \"look fastball\" as a flip: "
              f"{pct(two['flipped_broad_pct'])} flipped")
    print()
    print("  advice   times the first part told the hitter something")
    print("  held     the second part gave the same advice")
    print("  weaker   the second part said less — \"no case\", \"he mixes\", \"look fastball\"")
    print("  flipped  the second part pointed the other way — the failure that matters")
    print("  skill    above 0: his own number beats the average pitcher's at")
    print("           predicting what he did next; 0 means knowing him adds nothing")


def main(argv: list[str] | None = None) -> int:
    batch = _batch()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, default=config.get("data.default_season", 2026))
    parser.add_argument("--split", choices=["halves", "postseason"], default="halves")
    parser.add_argument("--list", type=Path, default=batch.DEFAULT_LIST)
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    splitter = (validation.split_halves if args.split == "halves"
                else validation.split_postseason)

    calls: list[dict] = []
    ran = 0
    for entry in batch.read_list(args.list, args.names):
        try:
            raw, resolved, _ = batch.load_pitcher(entry, args.season, args.offline, False)
            frame = cli.prepare(raw)
            first, second = splitter(frame)
            if len(second) == 0:
                print(f"skip  {entry['name']}: no {'postseason' if args.split == 'postseason' else 'second-half'} pitches")
                continue
            rows = validation.compare(first, second)
            for r in rows:
                calls.append({"pitcher": resolved, "pitches_first": len(first),
                              "pitches_second": len(second), **r})
            ran += 1
            print(f"ok    {resolved}: {len(first):,} / {len(second):,} pitches, "
                  f"{len(rows)} calls checked")
        except Exception as exc:  # one bad pitcher must not stop the run
            print(f"FAILED  {entry['name']}: {type(exc).__name__}: {exc}")

    summary = validation.summarise(calls)
    write_csv(args.out / f"holdout_calls_{args.split}.csv", calls)
    write_csv(args.out / f"holdout_summary_{args.split}.csv", summary)
    print_summary(summary, args.split, ran)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
