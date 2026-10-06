"""Run the report on a list of pitchers and show where the cutoffs land.

Every threshold in the hitting plan was set against two pitchers. This script
runs the whole pipeline across a deliberately varied list — starters and
relievers, both hands, sinkerballers, splitter pitchers, two-pitch closers,
soft-tossers, injury-shortened seasons — and writes three things to
reports/calibration/:

  <pitcher>.html       each report, for reading
  summary.csv          one row per pitcher: the numbers behind every plan rule,
                       including the rules that stayed silent
  plan_lines.csv       every sentence the plan wrote, one per row

and prints, for each cutoff in config.yaml's `plan` section, how the measured
values are spread across pitchers and what share of them the current cutoff
fires on. That is the view that tells you a cutoff is wrong: a rule that fires
for every pitcher says nothing, and one that never fires is dead weight.

A pitcher that fails — an ambiguous name, a season with no data — is recorded
with its error and the run carries on. Downloads are rate-limited by the fetch
client, so a full first run takes several minutes; after that the cache makes
it fast.

Usage:
    python scripts/batch_reports.py
    python scripts/batch_reports.py --season 2026
    python scripts/batch_reports.py --names "Blake Snell" "Logan Webb"
    python scripts/batch_reports.py --offline        # committed fixtures only
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import charts, cli, config, flags, gameplan, report  # noqa: E402
from src.metrics import counts, events, location, splits  # noqa: E402

DEFAULT_LIST = ROOT / "scripts" / "calibration_pitchers.csv"
OUT_DIR = ROOT / "reports" / "calibration"
FIXTURE_DIR = ROOT / "tests" / "fixtures"
EXPORT_DIR = ROOT / "data" / "exports"


def slug(name: str) -> str:
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", plain.lower()).strip("_")


def read_list(path: Path, names: list[str] | None) -> list[dict]:
    if names:
        return [{"name": n, "group": "ad hoc", "why": ""} for n in names]
    with path.open() as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_pitcher(entry: dict, season: int, offline: bool, refresh: bool) -> tuple:
    """Return (raw frame, resolved name, mlbam id or None)."""
    if offline:
        stem = slug(entry["name"]).split("_")[-1]
        path = FIXTURE_DIR / f"{stem}_{season}_raw.csv.gz"
        if not path.exists():
            raise FileNotFoundError(f"no fixture {path.name} (offline mode)")
        return pd.read_csv(path, low_memory=False), entry["name"], None

    from src import fetch

    player = fetch.resolve_player(entry["name"])
    frame = cli.load_by_id(player.mlbam_id, season, refresh=refresh)
    return frame, player.full_name, player.mlbam_id


_league_cache: dict = {}


def load_league(season: int, hand: str, offline: bool, use_league: bool) -> dict:
    """League tables for one handedness, loaded once per run.

    Offline, falls back to the CSVs written by scripts/export_baselines.py.
    """
    blank = {"league_outcomes": None, "league_mix": None, "league_putaway": None}
    if not use_league:
        return blank
    key = (season, hand, offline)
    if key in _league_cache:
        return _league_cache[key]

    if offline:
        files = {
            "league_outcomes": EXPORT_DIR / f"league_pitch_outcomes_{season}_{hand}.csv",
            "league_mix": EXPORT_DIR / f"league_count_mix_{season}_{hand}.csv",
            "league_putaway": EXPORT_DIR / f"league_putaway_rates_{season}_{hand}.csv",
        }
        league = {k: (pd.read_csv(v) if v.exists() else None) for k, v in files.items()}
    else:
        league = cli.load_league(season, hand)

    _league_cache[key] = league
    return league


# ---------------------------------------------------------------------------
# What to record per pitcher
# ---------------------------------------------------------------------------


def _r(v, digits: int = 3):
    if v is None:
        return ""
    try:
        if v != v:
            return ""
    except TypeError:
        return v
    return round(float(v), digits) if isinstance(v, float) else v


def side_numbers(frame: pd.DataFrame, usage: pd.DataFrame, side: str) -> dict:
    """The measurement behind each per-side plan rule, whether or not it fired."""
    out = {f"{side}_pitches": int((frame["stand"] == side).sum())}

    fp = gameplan.first_pitch_numbers(frame, usage, side)
    out[f"{side}_fp_called_on_takes"] = _r(fp["called_on_takes"]) if fp else ""
    out[f"{side}_fp_takes"] = fp["takes"] if fp else ""

    hc = gameplan.hitters_count_numbers(usage, side)
    if hc:
        hard_cut = gameplan._cut("sit_hard_share", 0.60)
        pitch_cut = gameplan._cut("sit_pitch_share", 0.40)
        rule = ("sit hard" if hc["hard_share"] >= hard_cut
                else f"sit {hc['top']}" if hc["top_share"] >= pitch_cut
                else "mixes")
        out.update({
            f"{side}_ahead_hard_share": _r(hc["hard_share"]),
            f"{side}_ahead_top": hc["top"],
            f"{side}_ahead_top_share": _r(hc["top_share"]),
            f"{side}_ahead_rule": rule,
        })
    else:
        out.update({f"{side}_ahead_hard_share": "", f"{side}_ahead_top": "",
                    f"{side}_ahead_top_share": "", f"{side}_ahead_rule": "gated"})

    ts = gameplan.two_strike_numbers(frame, usage, side)
    out.update({
        f"{side}_two_top": ts["top"] if ts else "",
        f"{side}_two_top_share": _r(ts["top_share"]) if ts else "",
        f"{side}_two_whiff": _r(ts["whiff"]) if ts else "",
        f"{side}_two_swings": ts["swings"] if ts else "",
        f"{side}_two_below": _r(ts["below"]) if ts else "",
    })

    rare = gameplan.rare_pitches(frame, side)
    out[f"{side}_rare"] = (
        "; ".join(f"{p} {s:.1%}" for p, s in rare["rare"]) if rare else "gated"
    )
    out[f"{side}_min_share"] = _r(rare["min_share"]) if rare else ""
    return out


def movement_overlaps(payload: dict) -> int:
    """How many movement-plot labels overlap a bubble or another label."""
    bubbles = payload.get("movement", {}).get("bubbles", [])
    boxes = []
    for b in bubbles:
        width = max(len(b["name"]) * 7.0, 78.0)
        boxes.append((b["label_x"] - width / 2, b["label_y"] - 12,
                      b["label_x"] + width / 2, b["label_y"] + 16))
    hits = 0
    for i, box in enumerate(boxes):
        for j, other in enumerate(bubbles):
            if i == j:
                continue
            circle = (other["cx"] - other["r"], other["cy"] - other["r"],
                      other["cx"] + other["r"], other["cy"] + other["r"])
            hits += charts._overlaps(box, circle)
        for k in range(i + 1, len(boxes)):
            hits += charts._overlaps(box, boxes[k])
    return hits


def summarise(entry: dict, frame: pd.DataFrame, payload: dict) -> tuple[dict, list[dict]]:
    """One summary row and the plan lines for a pitcher that ran."""
    usage = counts.situational_usage(frame)
    line = events.season_line(frame)
    mix = counts.pitch_mix(counts.restrict_to_arsenal(frame))

    row = {
        "pitches": line["pitches"],
        "games": line["games"],
        "batters": line["batters"],
        "hand": payload["meta"]["hand"],
        "arsenal": " ".join(f"{p}:{s:.0%}" for p, s in mix.items()),
        "arsenal_size": len(mix),
    }
    for side in ("L", "R"):
        row.update(side_numbers(frame, usage, side))

    lifts = counts.count_lifts(frame, min_n=int(config.get("flags.min_n.predictable_count", 20)))
    lifts = lifts[lifts["count"] != "3-0"] if len(lifts) else lifts
    if len(lifts):
        best = lifts.sort_values("lift", ascending=False).iloc[0]
        row.update({"count_tell_pitch": best["pitch_type"],
                    "count_tell_count": best["count"],
                    "count_tell_lift": _r(best["lift"])})
    else:
        row.update({"count_tell_pitch": "", "count_tell_count": "", "count_tell_lift": ""})

    tells = location.location_tells(frame, by="v_band",
                                    min_n=int(config.get("flags.min_n.location_tell", 40)))
    if len(tells):
        top = tells.sort_values("score", ascending=False).iloc[0]
        row.update({"location_band": top["band"], "location_pitch": top["pitch_type"],
                    "location_lift": _r(top["lift"]), "location_share": _r(top["p_pitch"])})
    else:
        row.update({"location_band": "", "location_pitch": "",
                    "location_lift": "", "location_share": ""})

    form = splits.recent_form(frame)
    shifts = form["shifts"]
    row["recent_notable"] = int(shifts["notable"].sum()) if len(shifts) else 0
    row["recent_velo_delta"] = _r(form["velo_delta"], 2)

    grid = payload.get("count_grid", {}).get("pitches", [])
    row["grid_gated_cells"] = sum(
        1 for p in grid for c in p["cells"].values() if not c["reliable"]
    )
    row["movement_label_overlaps"] = movement_overlaps(payload)
    row["findings"] = payload.get("finding_count", 0)
    row["findings_resolved"] = payload.get("resolved_count", 0)
    row["headline_count"] = payload.get("count_grid", {}).get("headline") or ""
    row["headline_platoon"] = payload.get("platoon_slope", {}).get("headline") or ""
    row["headline_movement"] = payload.get("movement", {}).get("headline") or ""

    plan = payload["card"]["plan"]
    for side in ("L", "R"):
        row[f"{side}_plan_lines"] = len(plan[side])
    row["both_plan_lines"] = len(plan["both"])

    lines = [
        {"pitcher": entry["name"], "side": side, "label": k["label"],
         "text": k["text"], "n": k["n"]}
        for side in ("L", "R", "both") for k in plan[side]
    ]
    return row, lines


# ---------------------------------------------------------------------------
# Threshold distributions
# ---------------------------------------------------------------------------


def _values(rows: list[dict], key_suffix: str) -> list[float]:
    vals = []
    for r in rows:
        for side in ("L", "R"):
            v = r.get(f"{side}_{key_suffix}", "")
            if v != "" and v is not None:
                vals.append(float(v))
    return vals


def _quantiles(vals: list[float]) -> str:
    if not vals:
        return "no data"
    s = pd.Series(vals)
    q = s.quantile([0, .25, .5, .75, 1]).tolist()
    return "min {:.2f} · p25 {:.2f} · median {:.2f} · p75 {:.2f} · max {:.2f}".format(*q)


def thresholds(rows: list[dict]) -> list[dict]:
    """For each plan cutoff: the spread of measured values and how often it fires."""
    ok = [r for r in rows if r.get("status") == "ok"]
    out = []

    def add(name, cutoff, vals, fires, note):
        n = len(vals)
        out.append({
            "cutoff": name, "value": cutoff, "pitcher_sides": n,
            "fires": fires, "fires_pct": round(fires / n, 3) if n else "",
            "spread": _quantiles(vals), "note": note,
        })

    called = _values(ok, "fp_called_on_takes")
    hi, lo = gameplan._cut("first_pitch_swing_above", .55), gameplan._cut("first_pitch_take_below", .45)
    add("first_pitch_swing_above", hi, called, sum(v >= hi for v in called),
        "share told 'be ready to swing'")
    add("first_pitch_take_below", lo, called, sum(v <= lo for v in called),
        "share told 'make him throw one'")

    hard = _values(ok, "ahead_hard_share")
    cut = gameplan._cut("sit_hard_share", .60)
    add("sit_hard_share", cut, hard, sum(v >= cut for v in hard), "share told 'sit hard'")

    rules = [r.get(f"{s}_ahead_rule", "") for r in ok for s in ("L", "R")]
    sit_pitch = [r for r in rules if r.startswith("sit ") and r != "sit hard"]
    top_shares = [float(r.get(f"{s}_ahead_top_share")) for r in ok for s in ("L", "R")
                  if r.get(f"{s}_ahead_rule", "").startswith("sit ")
                  and r.get(f"{s}_ahead_rule") != "sit hard"]
    add("sit_pitch_share", gameplan._cut("sit_pitch_share", .40), top_shares, len(sit_pitch),
        "'sit <pitch>' lines; spread is the share each one fired on")

    below = _values(ok, "two_below")
    cut = gameplan._cut("lay_off_low_share", .50)
    add("lay_off_low_share", cut, below, sum(v >= cut for v in below), "share told 'lay off it low'")

    least = _values(ok, "min_share")
    cut = gameplan._cut("rare_share", .03)
    add("rare_share", cut, least, sum(v < cut for v in least),
        "pitcher-sides with a 'won't see' line; spread is each side's least-used pitch")

    lifts = [float(r["count_tell_lift"]) for r in ok if r.get("count_tell_lift") not in ("", None)]
    cut = gameplan._cut("count_tell_lift", 2.0)
    add("count_tell_lift", cut, lifts, sum(v >= cut for v in lifts),
        "pitchers with a count tell; spread is each pitcher's best lift")

    loc = [(float(r["location_lift"]), float(r["location_share"])) for r in ok
           if r.get("location_lift") not in ("", None)]
    lift_cut = float(config.get("flags.thresholds.location_tell_lift", 1.35))
    share_cut = float(config.get("flags.thresholds.location_tell_share", 0.45))
    add("location_tell_lift + share", f"{lift_cut} / {share_cut}", [l for l, _ in loc],
        sum(l >= lift_cut and s >= share_cut for l, s in loc),
        "pitchers with a 'read the height' line; spread is best lift")
    return out


def print_thresholds(rows: list[dict]) -> None:
    print("\nWhere each cutoff lands across the calibration list")
    print("-" * 52)
    for t in rows:
        pct = f"{t['fires_pct']:.0%}" if t["fires_pct"] != "" else "—"
        print(f"\n  {t['cutoff']}  (currently {t['value']})")
        print(f"    fires on {t['fires']} of {t['pitcher_sides']}  ({pct}) — {t['note']}")
        print(f"    {t['spread']}")
    print("\n  A cutoff firing on almost every pitcher says nothing about any of them;")
    print("  one that never fires is dead weight. Neither is automatically wrong —")
    print("  read the plan lines before moving a number.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, default=config.get("data.default_season", 2025))
    parser.add_argument("--list", type=Path, default=DEFAULT_LIST)
    parser.add_argument("--names", nargs="+", help="Run these names instead of the list")
    parser.add_argument("--offline", action="store_true",
                        help="Use committed fixtures and exported league CSVs only")
    parser.add_argument("--no-league", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch even if cached")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    entries = read_list(args.list, args.names)
    if args.offline:
        # Offline is for testing the pipeline itself, so only the pitchers with
        # a committed fixture are meaningful; the rest are skipped, not failed.
        have = [e for e in entries
                if (FIXTURE_DIR / f"{slug(e['name']).split('_')[-1]}_{args.season}_raw.csv.gz").exists()]
        skipped = len(entries) - len(have)
        entries = have
        if skipped:
            print(f"Offline: skipping {skipped} pitchers with no committed fixture.")
    args.out.mkdir(parents=True, exist_ok=True)
    summary, all_lines = [], []

    print(f"\nCalibration run: {len(entries)} pitchers, {args.season}"
          f"{' (offline)' if args.offline else ''}\n")

    for i, entry in enumerate(entries, start=1):
        base = {"name": entry["name"], "group": entry.get("group", ""),
                "why": entry.get("why", "")}
        print(f"[{i:2d}/{len(entries)}] {entry['name']:<22s}", end=" ", flush=True)
        try:
            raw, resolved, mlbam = load_pitcher(entry, args.season, args.offline, args.refresh)
            frame = cli.prepare(raw)
            if len(frame) == 0:
                raise ValueError("no competitive pitches after cleaning")
            hand = str(frame["p_throws"].mode().iloc[0])
            league = load_league(args.season, hand, args.offline, not args.no_league)
            payload = report.build_payload(frame, resolved, args.season, **league)
            out_path = args.out / f"{slug(entry['name'])}_{args.season}.html"
            out_path.write_text(report.render(payload), encoding="utf-8")

            row, lines = summarise(entry, frame, payload)
            summary.append({**base, "status": "ok", "resolved": resolved,
                            "mlbam": mlbam or "", **row, "report": out_path.name})
            all_lines.extend(lines)
            print(f"ok  {resolved} ({hand}HP) · {row['pitches']:,} pitches · "
                  f"{row['L_plan_lines']}+{row['R_plan_lines']}+{row['both_plan_lines']} plan lines")
        except Exception as exc:  # one bad pitcher must not stop the run
            summary.append({**base, "status": "error",
                            "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED  {type(exc).__name__}: {exc}")
            if not isinstance(exc, (FileNotFoundError, ValueError, LookupError)):
                traceback.print_exc(limit=2)

    write_csv(args.out / "summary.csv", summary)
    write_csv(args.out / "plan_lines.csv", all_lines)
    cut_rows = thresholds(summary)
    write_csv(args.out / "thresholds.csv", cut_rows)

    ok = sum(1 for r in summary if r["status"] == "ok")
    try:
        shown = args.out.relative_to(ROOT)
    except ValueError:
        shown = args.out
    print(f"\n{ok} of {len(summary)} ran. Written to {shown}/: "
          f"summary.csv, plan_lines.csv, thresholds.csv and one report per pitcher.")
    overlaps = [r["name"] for r in summary if r.get("movement_label_overlaps")]
    if overlaps:
        print(f"Movement-plot labels overlap for: {', '.join(overlaps)}")
    print_thresholds(cut_rows)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
