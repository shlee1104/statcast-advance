"""Check what the report says, and whether it is right, on a few starts.

For each starter, half of his regular-season games are held out, and reports
are built from 3, 5 and 10 of the other games and then from all of them (see
src/sample_size.py for the design and why). Everything a report of each size
would print is scored against the held-out games. Writes to
reports/sample_size/<seasons>/:

  printed.csv   one row per pitcher, shuffle and size: how many lines, graph
                rows, coloured follow-ups and tunneling pairs the report has
  scored.csv    one row per claim a report made: the number in the sample and
                in the held-out games, whether it held, whether it pointed
                the wrong way. Rows with starts = "by date" are the control:
                his first half in date order, scored on his second
  shares.csv    one row per follow-up share the sequence graph would draw: the
                share in the sample and in the held-out games

and prints the summary. Seasons are pooled, and intervals resample pitchers,
so a pitcher with two seasons counts as one arm.

Usage:
    python scripts/sample_size.py --season 2025 2026 --cached
    python scripts/sample_size.py --season 2026 --names "Logan Webb" --cached
    python scripts/sample_size.py --season 2025 --offline --draws 3
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import cli, config, sample_size  # noqa: E402

OUT_DIR = ROOT / "reports" / "sample_size"


def _script(name: str):
    """Another script's loaders, so every check reads pitchers the same way."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def _head(size) -> str:
    return {sample_size.HALF: "half season",
            sample_size.BY_DATE: "half, by date"}.get(size, f"{size} starts")


def _pct(v: float) -> str:
    return f"{v:.0%}" if v == v else "—"


def print_summary(printed: list[dict], scored: list[dict], shares: list[dict],
                  seasons: list[int], pitchers: int, pitcher_seasons: int,
                  draws: int) -> None:
    print(f"\nWhat does the report say on a few starts, and is it right?  "
          f"{' + '.join(str(s) for s in seasons)}")
    print(f"{pitchers} pitchers, {pitcher_seasons} seasons, {draws} shuffles "
          f"each; scored on the half of his games held out")
    print("-" * 100)
    if not printed:
        print("  no starter seasons to run")
        return

    # The by-date half is a control with one sample a season; it gets its own
    # table, beside the shuffled half it is a control for.
    sizes = [s for s in dict.fromkeys(r["starts"] for r in printed)
             if s != sample_size.BY_DATE]
    by = {(r["metric"], r["starts"]): r for r in printed}

    print("\n  What one report prints, on average")
    print(f"  {'':<38}" + "".join(f"{_head(s):>15}" for s in sizes))
    for metric, label in sample_size.PRINTED_LABELS.items():
        digits = 0 if metric == "pitches" else 1
        print(f"  {label:<38}" + "".join(
            f"{by[(metric, s)]['mean']:>15,.{digits}f}" for s in sizes))

    print("\n  Share of reports with at least one")
    for metric in ("graph_coloured", "count_cells", "tunnel_read"):
        print(f"  {sample_size.PRINTED_LABELS[metric]:<38}" + "".join(
            f"{_pct(by[(metric, s)]['any']):>15}" for s in sizes))

    cells = {(r["kind"], r["starts"]): r for r in scored}
    print("\n  Is it right?  claims per report, share that held, share that "
          "pointed the wrong way")
    print(f"  {'':<34}" + "".join(f"{_head(s):>17}" for s in sizes))
    for kind, label in sample_size.KIND_LABELS.items():
        line = f"  {label:<34}"
        for s in sizes:
            c = cells.get((kind, s))
            if c is None or not c["claims"]:
                line += f"{'—':>17}"
                continue
            # A rate from one or two pitchers is those pitchers, not the rule.
            rates = (f"{_pct(c['held_pct']):>5}{_pct(c['wrong_pct']):>5}"
                     if c["pitchers"] >= 3 else f"{'—':>5}{'—':>5}")
            line += f"{c['per_report']:>7.2f}{rates}"
        print(line)

    print("\n  95% interval on the share that held, resampling pitchers")
    for kind, label in sample_size.KIND_LABELS.items():
        line = f"  {label:<34}"
        for s in sizes:
            c = cells.get((kind, s))
            span = (f"{c['held_lo']:.0%}–{c['held_hi']:.0%}"
                    if c and c["held_lo"] == c["held_lo"] else "—")
            line += f"{span:>17}"
        print(line)

    print("\n  How big the tendency looked in the sample, and in the held-out "
          "games (median)")
    for kind in ("count_lift", "follow_up_more", "follow_up_less", "sequencing"):
        line = f"  {sample_size.KIND_LABELS[kind]:<34}"
        for s in sizes:
            c = cells.get((kind, s))
            pair = (f"{c['median_first']:.2f} → {c['median_second']:.2f}"
                    if c and c["claims"] and c["pitchers"] >= 3 else "—")
            line += f"{pair:>17}"
        print(line)

    if any(k[1] == sample_size.BY_DATE for k in cells):
        print("\n  The control: a half season of shuffled games, and his first "
              "half scored on his second")
        print(f"  {'':<34}{'shuffled':>12}{'':>12}{'by date':>12}")
        print(f"  {'':<34}{'held':>12}{'wrong way':>12}{'held':>12}"
              f"{'wrong way':>12}{'claims':>9}")
        for kind, label in sample_size.KIND_LABELS.items():
            a = cells.get((kind, sample_size.HALF))
            b = cells.get((kind, sample_size.BY_DATE))
            if not a or not b or not a["claims"] or not b["claims"]:
                continue
            print(f"  {label:<34}{_pct(a['held_pct']):>12}{_pct(a['wrong_pct']):>12}"
                  f"{_pct(b['held_pct']):>12}{_pct(b['wrong_pct']):>12}"
                  f"{b['claims']:>9}")

    if shares:
        print("\n  A follow-up share on the sequence graph against the same "
              "share in the held-out games")
        print(f"  {'':<16}{'shares':>8}{'median gap':>13}{'90% within':>13}"
              f"{'under 5 pts':>13}{'under 10 pts':>14}")
        for r in shares:
            print(f"  {_head(r['starts']):<16}{r['cells']:>8,}"
                  f"{r['median_gap'] * 100:>9.1f} pts{r['p90_gap'] * 100:>9.1f} pts"
                  f"{_pct(r['within_five']):>13}{_pct(r['within_ten']):>14}")

    print()
    print("  held       the held-out games clear the bar the line was written at")
    print("  wrong way  a plan line that named a different pitch; a tendency that")
    print("             came back on the other side of his usual rate; a tunneling")
    print("             word whose pair landed nearer the opposite word's bar")
    print("  —          no claims, or claims from fewer than three pitchers")
    print("  These are major-league starters cut down to a college-sized sample,")
    print("  not college pitchers: this is the sample-size question and no other.")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    rep = _script("replication")
    batch = rep._batch()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, nargs="+",
                        default=[config.get("data.default_season", 2026)])
    parser.add_argument("--list", type=Path, default=batch.DEFAULT_LIST)
    parser.add_argument("--names", nargs="+")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--offline", action="store_true",
                        help="Use committed fixtures only")
    source.add_argument("--cached", action="store_true",
                        help="Use only pitcher-seasons already in the local "
                             "cache; never touches the network")
    parser.add_argument("--draws", type=int,
                        default=int(config.get("sample_size.draws", 20)))
    parser.add_argument("--seed", type=int,
                        default=int(config.get("sample_size.seed", 17)))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    seasons = sorted(set(args.season))
    out = args.out or OUT_DIR / "_".join(str(s) for s in seasons)
    out.mkdir(parents=True, exist_ok=True)

    work = []
    for season in seasons:
        if args.cached:
            work += [(name, season, lambda i=mlbam, s=season: rep.load_cached(i, s))
                     for name, mlbam in rep.cached_pitchers(season, args.names, batch)]
        else:
            work += [(entry["name"], season,
                      lambda e=entry, s=season: batch.load_pitcher(
                          e, s, args.offline, False)[0])
                     for entry in batch.read_list(args.list, args.names)]

    tables: dict = {"printed": [], "scored": [], "shares": []}
    ran: list[str] = []
    for name, season, load in work:
        try:
            frame = cli.prepare(load())
            if not sample_size.is_starter_season(frame):
                print(f"skip  {name} {season}: not a full season of starts")
                continue
            result = sample_size.run_pitcher(frame, name, season,
                                             draws=args.draws, seed=args.seed)
            for key, rows in result.items():
                tables[key].extend(rows)
            ran.append(name)
            print(f"ok    {name} {season}: {len(result['scored']):,} claims scored")
        except FileNotFoundError as exc:  # offline, and no fixture for him
            print(f"skip  {name} {season}: {exc}")
        except Exception as exc:  # one bad pitcher must not stop the run
            print(f"FAILED  {name} {season}: {type(exc).__name__}: {exc}")

    for key, rows in tables.items():
        rep.write_csv(out / f"{key}.csv", rows)
    print_summary(
        sample_size.summarise_printed(tables["printed"]),
        sample_size.summarise_scored(tables["scored"], tables["printed"]),
        sample_size.summarise_shares(tables["shares"]),
        seasons, len(set(ran)), len(ran), args.draws,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
