"""Check whether each kind of finding repeats between halves of a season.

For each pitcher, everything the report would find in the first half of his
regular season is re-measured in the second half, and every number in the
arsenal, location and release tables is computed in both (see
src/replication.py for what is measured and why). Writes to
reports/replication/<season>/:

  findings.csv      one row per first-half finding: the number in each half,
                    whether it held and whether it kept its direction
  per_pitch.csv     one row per pitch thrown 100+ times in both halves, every
                    table column in each half
  per_pitcher.csv   numbers that exist once per pitcher or once per count
  card.csv          every slice of the dugout card's pies and every cell of
                    its zone maps, by batter side, in each half
  tunnel.csv        every pair of pitch types thrown back to back often enough
                    in both halves: its separations and late movement in each
  setups.csv        every row of the first half's setup table, as the evidence
                    file prints it, with the same row in the second half
  full_season.csv   one row per pitcher: findings by type, comparisons run,
                    and how many survive correction

and prints the summary. This is the check behind what the report keeps and
what it cut, so a section that is added should be added here first.

Usage:
    python scripts/replication.py --season 2026
    python scripts/replication.py --season 2025 --names "Blake Snell" "Logan Webb"
    python scripts/replication.py --season 2026 --cached   # local cache only, no network
    python scripts/replication.py --offline                # committed fixtures only
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

from src import cli, config, replication, store, validation  # noqa: E402
from src.metrics import tunneling  # noqa: E402

OUT_DIR = ROOT / "reports" / "replication"


def _batch():
    """The calibration script's loaders, so both runs read pitchers the same way."""
    spec = importlib.util.spec_from_file_location(
        "batch_reports", ROOT / "scripts" / "batch_reports.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fmt(v):
    if isinstance(v, float):
        return "" if v != v else round(v, 4)
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


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def cached_pitchers(season: int, names: list[str] | None, batch) -> list[tuple[str, int]]:
    """(name, MLBAM id) for every pitcher-season in the local cache.

    For `--cached`: nothing is looked up or fetched, so a run cannot be
    changed by what the network returns that day, and it works on a machine
    that is offline. `names` keeps only pitchers whose name contains every
    word of one of them, accents ignored.
    """
    conn = store.connect()
    try:
        summary = store.cache_summary(conn)
    finally:
        conn.close()
    summary = summary[(summary["season"] == season)
                      & (summary["player_type"] == "pitcher")
                      & (summary["row_count"] > 0)]

    wanted = [set(batch.slug(n).split("_")) for n in names or []]
    out = []
    for row in summary.itertuples():
        last, _, given = str(row.player_name).partition(", ")
        name = f"{given} {last}".strip()
        words = set(batch.slug(name).split("_"))
        if wanted and not any(w <= words for w in wanted):
            continue
        out.append((name, int(row.mlbam_id)))
    return sorted(out)


def load_cached(mlbam_id: int, season: int) -> pd.DataFrame:
    conn = store.connect()
    try:
        return store.load_pitcher_season(conn, mlbam_id, season)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def _pct(v: float) -> str:
    return f"{v:.0%}" if v == v else "—"


def _span(lo: float, hi: float) -> str:
    return f"({lo:.0%}–{hi:.0%})" if lo == lo and hi == hi else ""


def print_summary(findings: list[dict], pitch: list[dict], pitcher: list[dict],
                  season_totals: dict, season: int, pitchers: int,
                  card: list[dict] | None = None,
                  tunnel: list[dict] | None = None,
                  setups: dict | None = None) -> None:
    print(f"\nDo the findings repeat?  first half → second half, {season}, "
          f"{pitchers} pitchers")
    print("-" * 96)

    print("\n  Found in the first half, re-measured in the second")
    print(f"  {'Finding':<30}{'found':>6}{'pitchers':>9}{'held':>6}{'95% CI':>12}"
          f"{'by pitcher':>12}"
          f"{'same way':>10}{'95% CI':>12}{'median 1st':>12}{'2nd':>7}")
    for s in findings:
        print(f"  {s['label']:<30}{s['found']:>6}{s['pitchers']:>9}"
              f"{_pct(s['held_pct']):>6}{_span(s['held_lo'], s['held_hi']):>12}"
              f"{_span(s['held_pitcher_lo'], s['held_pitcher_hi']) or '—':>12}"
              f"{_pct(s['direction_pct']):>10}"
              f"{_span(s['direction_lo'], s['direction_hi']):>12}"
              f"{s['median_first']:>12.2f}{s['median_second']:>7.2f}")
    if not findings:
        print("  nothing cleared a rule's bar in a first half")

    print("\n  The same number in each half, correlated across pitcher-pitches")
    for s in pitch:
        print(f"  {s['label']:<40}r = {s['r']:+.2f}   ({s['pairs']} pairs)")
    if not pitch:
        print(f"  fewer than {replication.MIN_PAIRS} pitches thrown "
              f"{replication.MIN_PITCHES_PER_HALF}+ times in both halves")

    print("\n  The same number in each half, correlated across pitchers")
    for s in pitcher:
        print(f"  {s['label']:<44}r = {s['r']:+.2f}   ({s['pairs']} pairs)")
    if not pitcher:
        print(f"  fewer than {replication.MIN_PAIRS} pitchers to correlate")

    if card:
        print("\n  The dugout card's pictures, the same number in each half")
        print(f"  {'':<52}{'r':>6}{'median move':>13}{'90% within':>12}"
              f"{'under 5 pts':>13}{'numbers':>9}{'same top':>10}{'of':>6}")
        for s in card:
            print(f"  {s['label']:<52}{s['r']:>+6.2f}"
                  f"{s['median_change'] * 100:>9.1f} pts"
                  f"{s['p90_change'] * 100:>8.1f} pts"
                  f"{_pct(s['within_five']):>13}{s['pairs']:>9}"
                  f"{_pct(s['same_top']):>10}{s['pictures']:>6}")
        print(f"  from {max(s['pitchers'] for s in card)} pitchers with "
              f"{replication.MIN_HALF_PITCHES}+ pitches in each half; \"same top\" is")
        print("  how often a pie's largest slice, or a map's busiest zone, is the")
        print("  same one in both halves")

    if tunnel:
        print("\n  Tunneling, the same pair of pitch types in each half")
        print(f"  {'':<46}{'r':>6}{'median move':>13}{'90% within':>12}"
              f"{'pairs':>7}{'r within pair type':>20}{'of':>6}")
        for s in tunnel:
            unit = {"flight_ms": "ms", "tunnel_d": "  "}.get(s["metric"], "in")
            print(f"  {s['label']:<46}{s['r']:>+6.2f}"
                  f"{s['median_change']:>10.1f} {unit}"
                  f"{s['p90_change']:>9.1f} {unit}{s['pairs']:>7}"
                  f"{s['r_within']:>+20.2f}{s['within_n']:>6}")
        print(f"  from {max(s['pitchers'] for s in tunnel)} pitchers; a pair is "
              f"compared when it occurred {tunneling.min_pairs()}+ times to one")
        print("  side in each half. \"r within pair type\" is the correlation left "
              "after each")
        print("  pair type's average across pitchers is removed: whether the "
              "number says")
        print("  something about this pitcher, not about fastballs and curveballs")

    if setups:
        print("\n  The setup table in the evidence file, the same row in each half")
        for m in setups["metrics"]:
            print(f"  {m['label']:<46}r = {m['r']:+.2f}   ({m['pairs']} rows)")
        print(f"  {setups['rows']} printed rows from {setups['pitchers']} pitchers came back "
              f"in the second half; {_pct(setups['clear_share'])} of them")
        print(f"  stood three standard errors clear in the first. Still above his usual "
              f"rate: {_pct(setups['same_way_clear'])}")
        print(f"  of those, {_pct(setups['same_way_rest'])} of the rest")

    if season_totals.get("pitchers"):
        t = season_totals
        print(f"\n  Full-season reports, {t['pitchers']} pitchers")
        for f in t["by_flag"]:
            print(f"  {f['flag']:<24}fires for {f['pitchers']:>2} of {t['pitchers']}"
                  f", {f['findings']:>3} findings")
        print(f"  {t['comparisons']:,} run-value comparisons "
              f"(median {t['comparisons_median']:.0f} per report); "
              f"{t['raw_p05']} clear an uncorrected p < .05 "
              f"({_pct(t['raw_p05_pct'])}); {t['bh_survivors']} survive correction")
        print(f"  {t['findings']} findings in all, {t['resolved']} with a run "
              f"consequence that survived")

    print()
    print("  held      the second half cleared the same bar the rule sets")
    print("  same way  it at least still pointed the same direction")
    print("  95% CI    Wilson intervals; a second half too thin to measure a")
    print("            finding in produces no row, not a failure")
    print("  by pitcher  the same interval from resampling pitchers, since one")
    print("            pitcher can give several findings; — under six pitchers")
    print("  r         on a deliberately varied list of pitchers, so higher")
    print("            than it would be across a random sample of the league")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    batch = _batch()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, default=config.get("data.default_season", 2026))
    parser.add_argument("--list", type=Path, default=batch.DEFAULT_LIST)
    parser.add_argument("--names", nargs="+")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--offline", action="store_true",
                        help="Use committed fixtures only")
    source.add_argument("--cached", action="store_true",
                        help="Use only pitcher-seasons already in the local "
                             "cache; never touches the network")
    parser.add_argument("--no-league", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    out = args.out or OUT_DIR / str(args.season)
    out.mkdir(parents=True, exist_ok=True)

    if args.cached:
        work = [(name, lambda i=mlbam: load_cached(i, args.season))
                for name, mlbam in cached_pitchers(args.season, args.names, batch)]
        if not work:
            print(f"Nothing in the cache for {args.season}"
                  f"{' matching ' + ', '.join(args.names) if args.names else ''}.")
    else:
        work = [(entry["name"],
                 lambda e=entry: batch.load_pitcher(e, args.season, args.offline, False)[0])
                for entry in batch.read_list(args.list, args.names)]

    findings: list[dict] = []
    per_pitch: list[dict] = []
    per_pitcher: list[dict] = []
    card: list[dict] = []
    tunnel: list[dict] = []
    setups: list[dict] = []
    full_season: list[dict] = []
    ran = 0
    for name, load in work:
        try:
            frame = cli.prepare(load())
            if len(frame) == 0:
                print(f"skip  {name}: no competitive pitches")
                continue
            hand = str(frame["p_throws"].mode().iloc[0])
            league = batch.load_league(
                args.season, hand, args.offline, not args.no_league
            )["league_outcomes"]
            first, second = validation.split_halves(frame)

            full_season.append({"pitcher": name,
                                **replication.full_season_row(frame)})
            for rows, table in (
                (replication.finding_rows(first, second), findings),
                (replication.per_pitch_rows(first, second, league), per_pitch),
                (replication.per_pitcher_rows(first, second), per_pitcher),
                (replication.card_rows(first, second), card),
                (replication.tunnel_rows(first, second), tunnel),
                (replication.setup_table_rows(first, second), setups),
            ):
                table.extend({"pitcher": name, **r} for r in rows)
            ran += 1
            print(f"ok    {name}: {len(first):,} / {len(second):,} pitches")
        except Exception as exc:  # one bad pitcher must not stop the run
            print(f"FAILED  {name}: {type(exc).__name__}: {exc}")

    write_csv(out / "findings.csv", findings)
    write_csv(out / "per_pitch.csv", per_pitch)
    write_csv(out / "per_pitcher.csv", per_pitcher)
    write_csv(out / "card.csv", card)
    write_csv(out / "tunnel.csv", tunnel)
    write_csv(out / "setups.csv", setups)
    write_csv(out / "full_season.csv", full_season)
    print_summary(
        replication.summarise_findings(findings),
        replication.pitch_correlations(per_pitch),
        replication.pitcher_correlations(per_pitcher),
        replication.summarise_full_season(full_season),
        args.season, ran,
        card=replication.card_summary(card),
        tunnel=replication.tunnel_summary(tunnel),
        setups=replication.setup_table_summary(setups),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
