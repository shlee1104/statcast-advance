"""Assemble every metric into one payload and render the report.

The report is two self-contained HTML files. The dugout card is one printed
page and is the file a coach is sent: static, no script, nothing to load. The
detail file beside it is the evidence for the card, for whoever wants to check
a line. No server, no external data fetch, no build step — either file is
opened and it works, on a phone in a parking lot if that is where the reader
is. That constraint is why the detail's data is embedded as JSON rather than
loaded, and why its only external dependency is a charting library from a CDN
with a graceful fallback to tables.

They were one file, card first. The card said "one page" and the file was
twelve, so a reader who wanted what a coach gets had to stop scrolling at the
right place. Now the default file is the page, and the rest is one link away.

Two layers live here:

  build_payload()  runs the metrics modules and flattens everything into a
                   JSON-serializable dict. All the baseball happens here.
  render()         hands that dict to a Jinja2 template, once per file. No
                   computation.

Keeping them apart means the payload can be inspected, diffed between runs, or
fed to something else entirely without touching presentation — and the
template cannot quietly introduce a number that no metric produced.

The detail file is in the order a reader would want it while checking
whether to believe a line on the card.

Two things the page used to print are no longer built here. The ranked
findings list (`flags.evaluate`) repeated, as sentences with a "strength"
score, patterns the card and the tables already show; the score's weights
were a judgment call, and the run cost it existed to attach survived
correction for 1 finding in 287. The per-count entropy "predictability"
table fed no rule. Both still exist in the metrics layer — `src/flags.py`
and `counts.predictability` — and `scripts/replication.py` still measures
them; the report does not call them.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src import baselines, charts, config, gameplan
from src.metrics import (
    arsenal, counts, events, location, seasons, sequencing, splits, tunneling,
)

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = ROOT / "templates"
TEMPLATE_NAME = "report.html.j2"


def _clean(value: Any) -> Any:
    """Make a value JSON-safe.

    NaN and infinity are not valid JSON. json.dumps emits them anyway as bare
    NaN/Infinity tokens, which parse as a syntax error in the browser and take
    the whole page down — so every missing number becomes null here instead.
    """
    if value is None:
        return None
    if isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if hasattr(value, "item"):  # numpy scalar
        return _clean(value.item())
    if pd.isna(value):
        return None
    return str(value)


def _records(frame: pd.DataFrame, index_name: str | None = None) -> list[dict]:
    """Turn a DataFrame into a list of JSON-safe dicts."""
    if len(frame) == 0:
        return []
    working = frame.reset_index() if index_name else frame
    if index_name and "index" in working.columns:
        working = working.rename(columns={"index": index_name})
    return [
        {str(key): _clean(val) for key, val in row.items()}
        for row in working.to_dict(orient="records")
    ]


def _display_names(frame: pd.DataFrame) -> dict[str, str]:
    """Map pitch codes to readable names, for arsenal pitches only.

    Restricted to the arsenal for the same reason every other section is: a
    lookup built from raw labels advertises a sweeper the arsenal table has
    never heard of.
    """
    if "pitch_display" not in frame.columns:
        return {}
    subset = counts.restrict_to_arsenal(frame)
    pairs = subset[["pitch_type", "pitch_display"]].drop_duplicates()
    return {row.pitch_type: row.pitch_display for row in pairs.itertuples()}


# Feed columns the metrics read that are not required for a report to render.
# Their absence removes a section rather than raising, so it has to be said
# out loud on the page instead of looking like an empty result.
OPTIONAL_COLUMNS: dict[str, str] = {
    "n_thruorder_pitcher": "times through the order",
    "arm_angle": "release arm angle",
}


def _data_gaps(frame: pd.DataFrame) -> list[dict]:
    """Report metric inputs missing from this frame."""
    return [
        {"column": col, "affects": what}
        for col, what in OPTIONAL_COLUMNS.items()
        if col not in frame.columns
    ]


def build_payload(
    frame: pd.DataFrame,
    pitcher: str,
    season: int | str,
    league_outcomes: pd.DataFrame | None = None,
    league_mix: pd.DataFrame | None = None,
    league_putaway: pd.DataFrame | None = None,
) -> dict:
    """Run every metric and flatten the results into one dict.

    `frame` must already have passed through the cleaning pipeline and
    `events.add_event_flags()`. League tables are optional; every section that
    needs one degrades to the un-compared version rather than disappearing, so
    a report generated before `build_baselines.py` has run is thinner but not
    broken.

    `league_mix` is accepted so a caller can pass the three league tables
    together, and is not read: the two sections that used it, the findings
    list and the count-predictability table, are no longer in the report.
    """
    if len(frame) == 0:
        raise ValueError("Cannot build a report from an empty frame.")

    # Combined seasons: name what changed, then drop pitches he has shelved
    # from everything except the season line. The season line keeps every
    # pitch because removing a pitch type removes the plate appearances it
    # ended, which would bend his K% and BB%.
    full_frame = frame
    changes = seasons.season_changes(frame)
    frame = seasons.drop_retired_pitches(frame, changes)

    hand = str(frame["p_throws"].mode().iloc[0])
    has_league = league_outcomes is not None

    arsenal_rows = _build_arsenal(frame, league_outcomes)

    putaway = sequencing.putaway(frame, min_pitches=10)
    if has_league and league_putaway is not None and len(putaway):
        putaway_rows = _records(baselines.compare_putaway(putaway, league_putaway))
    else:
        putaway_rows = _records(putaway, index_name="pitch_type")

    fatigue = splits.fatigue(frame, bucket_size=15)

    # Read once: the card's sequence line and the sequence graph are the same
    # follow-ups, and must come from the same table.
    sequences = _sequence_table(frame)

    return {
        "meta": {
            "pitcher": pitcher,
            "season": season,
            "seasons": seasons.season_counts(full_frame),
            "excluded_pitches": int(len(full_frame) - len(frame)),
            "hand": hand,
            "pitches": int(len(full_frame)),
            "games": int(full_frame["game_pk"].nunique()),
            # Same count as the season line, so the page shows one number.
            # Grouping by at-bat ran a few higher, because plate appearances
            # that end on a pickoff or caught stealing have no result row.
            "batters_faced": int(events.season_line(full_frame)["batters"]),
            "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "has_league": has_league,
            "league_note": (
                f"Compared against {hand}HP league baselines"
                if has_league
                else "No league baselines loaded — deviations unavailable"
            ),
        },
        "data_gaps": _data_gaps(frame),
        "card": _build_card(frame, changes=changes, season_frame=full_frame,
                            sequences=sequences),
        "arsenal": arsenal_rows,
        "display_names": _display_names(frame),
        "count_grid": _build_count_grid(frame),
        "platoon_slope": _build_platoon_slope(frame),
        "movement": _build_movement(frame, league_outcomes),
        "putaway": putaway_rows,
        "sequence_graph": _build_sequence_graph(frame, sequences),
        "tunneling": _build_tunneling(frame, sequences, hand),
        "setups": _records(setup_table(frame)),
        "setup_min_n": SETUP_MIN_N,
        "location_profile": _records(
            location.location_profile(frame), index_name="pitch_type"
        ),
        "location_tells": _records(_without_runs(
            location.location_tells(frame).head(12))),
        "band_information": {
            band: {k: _clean(v) for k, v in location.band_information(frame, by=band).items()}
            for band in ("v_band", "h_band", "quadrant")
        },
        "zone_slices": _build_zone_slices(frame),
        "release": _records(arsenal.release_consistency(frame), index_name="pitch_type"),
        "fatigue": _build_fatigue(fatigue, splits.velocity_pitch(frame)),
        "tto": _records(splits.times_through_order(frame)),
        "gates": {
            "note": (
                "Every claim states its sample size. Thresholds and minimum "
                "samples are defined in config.yaml, not in code."
            ),
        },
    }


# The run-value columns the metric tables carry. flags.py reads them to
# correct across every comparison it runs; the page does not show them, so
# they are not sent to it.
RUN_COLUMNS: list[str] = [
    "rv_when", "rv_base", "rv_delta", "runs_cost", "runs_lo", "runs_hi",
    "significant", "rv_p",
]


# The evidence file's setup table: the strongest pairs that occurred this
# often, and the columns it prints.
SETUP_ROWS = 12
SETUP_MIN_N = 25
SETUP_COLUMNS: list[str] = [
    "setup_pitch", "setup_band", "next_pitch", "n", "p_next", "baseline_p",
    "freq_lift",
]


def setup_table(frame: pd.DataFrame) -> pd.DataFrame:
    """The setup pairs the evidence file prints, and only what it prints.

    `sequencing.setup_pairs` still measures what happened after each setup —
    the follow-up's whiff rate against its usual, the xwOBA on contact — and a
    `score` it ranks by. None of it is printed. Between halves of a season, on
    the calibration pitchers, the whiff change came back at r of -0.03 and
    0.07 (2025, 2026) and the xwOBA at 0.27 and 0.32, where the share and the
    lift that are printed came back at 0.74 to 0.86 (`scripts/replication.py`
    prints all four). The score is a sort key, not a number a reader can use.
    """
    table = sequencing.setup_pairs(frame, min_n=SETUP_MIN_N).head(SETUP_ROWS)
    return table[[c for c in SETUP_COLUMNS if c in table.columns]]


def _without_runs(table: pd.DataFrame) -> pd.DataFrame:
    """A setup or location table without its uncorrected run estimates.

    Each row's run cost is one of about a hundred comparisons, and on the
    calibration pitchers 1 finding in 287 had a cost that survived correction.
    The page prints no run figure anywhere for that reason.
    """
    return table.drop(columns=[c for c in RUN_COLUMNS if c in table.columns])


def _build_arsenal(
    frame: pd.DataFrame,
    league_outcomes: pd.DataFrame | None,
) -> list[dict]:
    """Each pitch against the league's version of it, and by batter side.

    One table for what the card's arsenal table does not carry. It replaced
    three pieces that each held one thing shown nowhere else and repeated the
    rest: a second arsenal table (the league comparisons and zone rate, next
    to movement columns already on the card), a platoon table (whiff rate by
    batter side, next to usage by side already on the card and in the slope
    chart), and a bar chart of usage against the league's.

    The by-side whiff rates are blank where he threw too few of the pitch to
    that side for a rate to mean anything (see splits.platoon_gaps), and the
    by-side columns are left off altogether when he barely faced one side.
    """
    profile = arsenal.profile(frame)
    if league_outcomes is not None:
        rows = _records(baselines.compare_profile(profile, league_outcomes))
    else:
        rows = _records(profile, index_name="pitch_type")

    sides = splits.platoon_gaps(frame)
    if len(sides):
        by_pitch = sides.set_index("pitch_type")
        for row in rows:
            pitch = row.get("pitch_type")
            for column in ("whiff_L", "n_L", "whiff_R", "n_R"):
                row[column] = (
                    _clean(by_pitch.at[pitch, column])
                    if pitch in by_pitch.index else None
                )
    return rows


def _build_count_grid(frame: pd.DataFrame) -> dict:
    """Balls-by-strikes grids, one per pitch, for the count section.

    Replaces a 100% stacked bar of pitch mix by count. That chart had three
    problems a reviewer identified, all structural. It laid twelve counts out
    as a flat row of labels, throwing away the fact that a count is a
    coordinate — 2-1 and 1-2 are neighbours — so a pitch clustering in one
    corner of the lattice could not be seen clustering. Only its bottom series
    could be read accurately, since every other segment started at a
    different height. And it plotted raw share, so most of what it showed was
    his season mix repeated twelve times.

    Each cell here carries the lift over his own rate (the colour), plus the
    share and the pitch count (the text), so colour is never the only channel.
    `log2_lift` is supplied so the template can place ratios on a symmetric
    scale without computing logarithms: 2x and 0.5x sit the same distance from
    centre, where on a linear scale "half as often" would be squashed.
    """
    grid = counts.count_grid(frame)
    if len(grid) == 0:
        return {"pitches": [], "excluded": [], "headline": None, "min_count": 0}

    min_count = int(config.get("report.count_grid_min_count", 30))
    working = counts.restrict_to_arsenal(frame)
    overall = counts.pitch_mix(working)
    shown = list(dict.fromkeys(grid["pitch_type"]))

    pitches = []
    for pitch in shown:
        rows = grid[grid["pitch_type"] == pitch].set_index("count")
        cells = {}
        for count in counts.ALL_COUNTS:
            r = rows.loc[count]
            cells[count] = {
                "n": int(r["n"]),
                "count_n": int(r["count_n"]),
                "share": _clean(r["share"]),
                "lift": _clean(r["lift"]),
                "log2_lift": _clean(r["log2_lift"]),
                "reliable": bool(r["reliable"]),
                "clears": bool(r["clears"]),
            }
        pitches.append({
            "pitch": pitch,
            "own_rate": _clean(float(rows["own_rate"].iloc[0])),
            "cells": cells,
        })

    excluded = [
        {"pitch": p, "share": _clean(float(overall[p]))}
        for p in overall.index if p not in shown
    ]

    return {
        "pitches": pitches,
        "excluded": excluded,
        "headline": gameplan.count_headline(frame),
        "min_count": min_count,
        "min_z": float(config.get("flags.thresholds.frequency_min_z", 3.0)),
    }


def _build_platoon_slope(frame: pd.DataFrame) -> dict:
    """Usage against lefties and righties, as a two-point slope chart.

    Replaces grouped bars, which made the reader subtract one bar from another
    to see a split. A line from one side to the other turns the gap into a
    slope you see at once, and a pitch that one side never sees becomes a line
    that starts at the floor.

    Lines are drawn solid whatever the pitch count on one side. The reviewer's
    version dotted the sinker and slider for being under 25 pitches to lefties,
    but usage is a share of every pitch thrown to that side — Yamamoto's 9
    sinkers to lefties are 9 of about 1,600, a precisely measured 0.6%. The
    small number is the finding, not a doubt about it; dotting it would say the
    opposite. The same principle governs the count grid, and it is why the
    arsenal table blanks a thin side's whiff rate while the card keeps its usage.
    """
    gaps = splits.platoon_gaps(frame, min_rate_pitches=0)
    if len(gaps) == 0:
        return {"pitches": [], "headline": None}

    rows = [
        {"pitch_type": r.pitch_type,
         "usage_L": _clean(r.usage_L) or 0.0,
         "usage_R": _clean(r.usage_R) or 0.0}
        for r in gaps.itertuples()
    ]
    layout = charts.slope_layout(rows)
    side_n = frame["stand"].value_counts()
    return {
        **layout,
        "headline": gameplan.platoon_headline(frame),
        "side_n": {"L": int(side_n.get("L", 0)), "R": int(side_n.get("R", 0))},
    }


def _sequence_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Every pitch-to-next-pitch follow-up by batter side, standouts marked."""
    table = sequencing.transitions(frame)
    return table.assign(standout=sequencing.transition_standouts(table))


def _build_sequence_graph(
    frame: pd.DataFrame,
    table: pd.DataFrame | None = None,
) -> dict:
    """What follows each pitch, to each batter side, as two-column graphs.

    One panel per side, because he does not throw the same pitches to both
    and a pro advance report splits everything a hitter reads by side. Each
    panel lists his pitches twice — just thrown, then next — and draws a line
    for each follow-up, coloured when it departs from his usual by the rule in
    `sequencing.transition_standouts`.

    Pitches are listed in the same order in both panels, most used first, so a
    pitch sits in the same place on each and the two can be compared at a
    glance. A pitch he never throws to a side is left off that side's panel.
    """
    row_gate = int(config.get("flags.min_n.sequence_follow_ups", 40))
    if table is None:
        table = _sequence_table(frame)
    order = list(counts.pitch_mix(counts.restrict_to_arsenal(frame)).index)

    panels = []
    for side, label in (("L", "To lefties"), ("R", "To righties")):
        rows = table[table["stand"] == side]
        seen = set(rows["pitch_type"]) | set(rows["next_pitch"])
        total = int(rows["total"].iloc[0]) if len(rows) else 0
        panel = {"side": side, "label": label, "total": total,
                 "enough": total >= row_gate}
        if panel["enough"]:
            panel.update(charts.sequence_layout(
                [p for p in order if p in seen], _records(rows), row_gate))
        panels.append(panel)

    return {
        "panels": panels,
        "headline": gameplan.sequence_headline(table),
        "row_gate": row_gate,
        "more_lift": float(config.get("flags.thresholds.sequence_more_lift", 1.25)),
        "less_lift": float(config.get("flags.thresholds.sequence_less_lift", 0.80)),
        "colors": {"more": charts.SEQUENCE_MORE, "less": charts.SEQUENCE_LESS,
                   "usual": charts.SEQUENCE_USUAL},
    }


TUNNEL_ROWS_PER_SIDE = 8


def _build_tunneling(frame: pd.DataFrame, sequences: pd.DataFrame, hand: str) -> dict:
    """The follow-ups the sequence graph draws, read for what a hitter can see.

    One row per pair of pitch types thrown back to back often enough to one
    side, the most common first and at most TUNNEL_ROWS_PER_SIDE a side: how
    often the follow-up comes (from the same transition table the graph
    uses), then where it is relative to the pitch before it — out of the
    hand, at the decision point, and in how it moves after that. Offsets, not
    mean distances: see `tunneling.pair_table` for why.

    `available` is false when the frame has no trajectory, which is what a
    pitcher-season cached before those columns were stored looks like. The
    page then says how to fill it in, where an empty table would read as "he
    has no pairs".

    It is in the evidence file and not on the card. The offset at the
    decision point comes back in the other half of a season at r of about
    0.65, where the card's own numbers are 0.73 and up.
    """
    base = {
        "available": tunneling.has_trajectory(frame),
        "tunnel_ft": tunneling.tunnel_point_ft(),
        "min_pairs": tunneling.min_pairs(),
        "same_bar": float(config.get("tunneling.same_window_d", 0.75)),
        "apart_bar": float(config.get("tunneling.apart_d", 1.25)),
        "rows": [], "scatter": {}, "headline": None,
    }
    if not base["available"]:
        return base

    table = tunneling.pair_table(frame)
    pairs = table[table["pitch_type"] != table["next_pitch"]]
    shares = sequences.set_index(["stand", "pitch_type", "next_pitch"])["share"] \
        if len(sequences) else pd.Series(dtype=float)

    rows = []
    for side in ("L", "R"):
        for r in pairs[pairs["stand"] == side].head(TUNNEL_ROWS_PER_SIDE).itertuples():
            key = (r.stand, r.pitch_type, r.next_pitch)
            rows.append({
                "stand": r.stand, "pitch_type": r.pitch_type, "next_pitch": r.next_pitch,
                "n": int(r.n),
                "share": _clean(shares.get(key, float("nan"))),
                "release_offset": _clean(r.release_offset),
                "tunnel_offset": _clean(r.tunnel_offset),
                "tunnel_where": gameplan.where(r.tunnel_dx, r.tunnel_dz, r.stand, hand),
                "tunnel_d": _clean(r.tunnel_d),
                "reading": r.reading,
                "late_offset": _clean(r.late_offset),
                "late_where": gameplan.where(r.late_dx, r.late_dz, r.stand, hand),
                "flight_ms": _clean(r.flight_ms),
            })
    # Both sides are always named: a pitcher whose pairs all came to one side
    # has no scatter for the other, and the page has to be able to say so.
    scatter = {"L": None, "R": None}
    scatter.update({side: _clean(float(group["spread"].iloc[0]))
                    for side, group in table.groupby("stand")})
    return {**base, "rows": rows, "scatter": scatter,
            "headline": gameplan.tunnel_headline(table, hand)}


def _build_movement(frame: pd.DataFrame, league_outcomes: pd.DataFrame | None) -> dict:
    """Ride against arm-side run, per pitch, with the league's version behind it.

    The single most standard chart in pitching analysis, and it was missing:
    the numbers existed only as two columns of the arsenal table, where nobody
    sees that a splitter sits in empty space between the fastballs and the
    breaking balls.

    League markers appear only when the league table carries movement columns
    (`avg_ivb`, `avg_arm_run`), which older exports do not. A report built
    without them draws his pitches alone rather than failing.
    """
    table = arsenal.scout_arsenal(frame)
    if len(table) == 0:
        return {"bubbles": [], "league": [], "headline": None}
    velo = arsenal.profile(frame)["velo"]
    names = _display_names(frame)

    rows = [
        {"pitch_type": p, "name": names.get(p, p),
         "arm_run": float(r.arm_run), "ivb": float(r.ivb),
         "usage": float(r.usage), "whiff_rate": _clean(r.whiff_rate),
         "velo": _clean(float(velo.get(p, float("nan"))))}
        for p, r in table.iterrows()
        if r.arm_run == r.arm_run and r.ivb == r.ivb
    ]

    league = []
    if league_outcomes is not None and {"avg_ivb", "avg_arm_run"} <= set(league_outcomes.columns):
        for r in league_outcomes.itertuples():
            if r.pitch_type in table.index:
                league.append({"pitch_type": r.pitch_type,
                               "arm_run": _clean(r.avg_arm_run),
                               "ivb": _clean(r.avg_ivb)})

    return {
        **charts.movement_layout(rows, league),
        "headline": gameplan.movement_headline(frame),
        "has_league": bool(league),
    }


def _build_zone_slices(frame: pd.DataFrame) -> dict:
    """Zone counts per pitch, count state and handedness, plus the lift base.

    Emits raw COUNTS rather than shares, and the per-pitch baseline separately,
    so the page can combine slices — "ahead, both hands", "all counts, vs
    lefties" — by summing counts and recomputing. Pre-computing every
    combination would trade a much larger payload for arithmetic a browser does
    instantly, and shares cannot be averaged across slices of different sizes
    without weighting, which is a thing that gets silently gotten wrong.
    """
    table = location.zone_slices(frame)
    if len(table) == 0:
        return {"zones": location.ALL_ZONES, "states": location.COUNT_STATES,
                "pitches": [], "cells": [], "base": {}, "min_slice": 25}

    # Only cells with pitches in them; zeros are implied and are most of the
    # table, so dropping them roughly halves the payload.
    cells = [
        {"p": r.pitch_type, "s": r.state, "h": r.stand,
         "z": int(r.zone), "n": int(r.n)}
        for r in table.itertuples() if r.n
    ]

    base = {}
    for pitch, group in table.groupby("pitch_type"):
        first = group.drop_duplicates("zone").set_index("zone")["base_share"]
        base[pitch] = {str(z): _clean(first.get(z, 0.0))
                       for z in location.ALL_ZONES}

    order = (
        table.groupby("pitch_type")["n"].sum().sort_values(ascending=False).index
    )

    return {
        "zones": location.ALL_ZONES,
        "inner": location.INNER_ZONES,
        "outer": location.OUTER_ZONES,
        "states": location.COUNT_STATES,
        "pitches": list(order),
        "cells": cells,
        "base": base,
        # Below this many pitches in a slice, the grid is noise however it is
        # coloured, so the renderer says so rather than drawing it confidently.
        "min_slice": 25,
    }


def _build_fatigue(fatigue: pd.DataFrame, pitch: str | None = None) -> dict:
    """Velocity-by-pitch-count series, with each bucket marked trustworthy or not.

    Two series go out. `velo` is the plain average of each bucket. `paired` is
    his opening velocity plus the average WITHIN-outing change to that bucket,
    and it is the one the chart draws: a late bucket holds only the starts
    that got that deep, which are the starts that were going well, so the
    plain average of the tail suffers from survivorship and not just small
    samples. Yamamoto's last bucket is a handful of pitches across 4 outings
    whose plain average is 0.4 mph ABOVE his opening velocity, which reads as
    recovery and is selection; within those same outings he is down 0.2.

    Buckets below `flags.min_n.fatigue_outings` separate outings are still
    marked unreliable rather than dropped: removing them would make the curve
    look cleaner than the evidence, and a reader would not know the line had
    been trimmed. The chart greys them; the number stays available.

    `pitch` is the one fastball the series is read from.
    """
    min_outings = int(config.get("flags.min_n.fatigue_outings", 10))

    if len(fatigue) == 0:
        return {"buckets": [], "velo": [], "paired": [], "delta": [],
                "outings": [], "pairs": [], "n": [], "reliable": [],
                "min_outings": min_outings, "pitch": pitch}

    opening = float(fatigue["velo"].iloc[0])
    return {
        "buckets": list(fatigue["bucket"]),
        "velo": [_clean(v) for v in fatigue["velo"]],
        "paired": [_clean(opening + d) for d in fatigue["velo_delta"]],
        "delta": [_clean(v) for v in fatigue["velo_delta"]],
        "outings": [_clean(v) for v in fatigue["outings"]],
        "pairs": [_clean(v) for v in fatigue["pairs"]],
        "n": [_clean(v) for v in fatigue["n"]],
        "reliable": [bool(v >= min_outings) for v in fatigue["pairs"]],
        "min_outings": min_outings,
        "pitch": pitch,
    }


CARD_SIDES: list[tuple[str, str]] = [("L", "vs LHH"), ("R", "vs RHH")]


def _build_card(
    frame: pd.DataFrame,
    last_n: int = 5,
    changes: dict | None = None,
    season_frame: pd.DataFrame | None = None,
    sequences: pd.DataFrame | None = None,
) -> dict:
    """The dugout card: page one of the report, the part a coach takes out.

    Front offices describe paring "a phone book's amount of information" down
    to a few pages, with one-sentence notes for use mid-game. This is that
    page, and it is one printed page. In reading order: the season line, the
    arsenal as a scout writes it, his last outings, his mix in five
    situations as pies and where each pitch goes as zone maps, both split by
    batter side, and the keys — the lines of the hitting plan a hitter can
    act on.

    What earns a place is what repeats. Usage, velocity, movement and
    location are nearly the same pitcher in both halves of a season (r of
    0.92 and up, see scripts/replication.py), whiff and chase rates mostly
    (0.73 to 0.82), and the keys are the lines the holdout check scores as
    advice. A pitch's xwOBA (0.6) and anything in runs are not on the card.

    The payload still carries everything the page used to print: `plan` is
    the whole hitting plan, `key_lines` the lines of it chosen for the card (see
    `gameplan.card_keys`), `tendencies` the top pitches by situation that the
    pies replaced. Nothing was removed to make the page shorter; the template
    prints less.

    It replaced a strip of headline numbers whose most prominent cell was
    "most predictable count: 3-0, four-seam 82%". That is 32 pitches, a count
    in which every pitcher throws a fastball, and one of the few patterns whose
    run consequence actually resolves — in his favour. It was the least useful
    true fact on the page and it was in the most prominent position.
    """
    usage = counts.situational_usage(frame)
    tendencies = []
    for situation in counts.SITUATIONS:
        row = {"situation": situation,
               "label": counts.SITUATION_LABELS[situation]}
        for side in ("L", "R"):
            cell = usage[(usage["situation"] == situation) & (usage["stand"] == side)]
            cell = cell.sort_values("share", ascending=False)
            row[side] = {
                "n": int(cell["situation_n"].iloc[0]) if len(cell) else 0,
                "reliable": bool(cell["reliable"].iloc[0]) if len(cell) else False,
                "top": [
                    {"pitch": r.pitch_type, "share": _clean(r.share)}
                    for r in cell.head(3).itertuples()
                ],
            }
        tendencies.append(row)

    form = splits.recent_form(frame, last_n=last_n)
    starts = _records(form["starts"])
    for start in starts:
        day = pd.Timestamp(start["date"])
        start["day"] = f"{day.strftime('%b')} {day.day}"

    scout = arsenal.scout_arsenal(frame)
    order = list(scout.index)
    plan = gameplan.build(frame, last_n=last_n, changes=changes)
    dated = frame if season_frame is None else season_frame

    return {
        "season": {k: _clean(v) for k, v in events.season_line(dated).items()},
        "through": _last_date(dated),
        "arsenal": _records(scout, index_name="pitch_type"),
        "order": order,
        "colors": {p: charts.color(p) for p in order},
        "short_names": {p: gameplan.short(p) for p in order},
        "rate_min": int(config.get("card.min_rate_sample", 30)),
        "rare_share": float(config.get("plan.rare_share", 0.03)),
        "tendencies": tendencies,
        "pies": _build_card_pies(usage, order),
        "zones": _build_card_zones(frame, order),
        "recent": {
            "last_n": form["last_n"],
            "dates": form["dates"],
            "fastball": form["fastball"],
            "velo_recent": _clean(form["velo_recent"]),
            "velo_earlier": _clean(form["velo_earlier"]),
            "velo_delta": _clean(form["velo_delta"]),
            "earlier_label": form["earlier_label"],
            "n_recent": form["n_recent"],
            "starts": starts,
            "shifts": _records(form["shifts"]),
        },
        "plan": plan,
        # Not "keys": in the template `card.keys` is the dict's own method.
        "key_lines": gameplan.card_keys(plan, sequences),
        "coach_names": gameplan.COACH_NAMES,
    }


def _last_date(frame: pd.DataFrame) -> str | None:
    """The date of his last outing in the sample, as "Sep 24".

    No year: it is printed next to the season, and the last outing is in it.
    """
    if "game_date" not in frame.columns or len(frame) == 0:
        return None
    last = pd.to_datetime(frame["game_date"]).max()
    if pd.isna(last):
        return None
    return f"{last.strftime('%b')} {last.day}"


def _build_card_pies(usage: pd.DataFrame, order: list[str]) -> list[dict]:
    """His mix in each of five situations, to each batter side, as pies.

    The tendency chart every advance report carries, drawn the way they draw
    it. It took the place of a table of the top three pitches per situation:
    the same shares, but every pitch rather than three, and a hitter sees
    that two strikes is mostly one colour before he reads a number.

    One row of pies per side. Pitches go round each pie in the same order,
    most used first (see `charts.pie_layout`). A situation with fewer pitches
    than `situational_usage` calls reliable is still drawn, marked so the
    page can fade it; one with no pitches at all gets an empty pie.
    """
    sides = []
    for side, label in CARD_SIDES:
        pies = []
        for situation in counts.SITUATIONS:
            cell = usage[(usage["situation"] == situation) & (usage["stand"] == side)]
            by_pitch = cell.set_index("pitch_type") if len(cell) else cell
            rows = [
                {"pitch": p, "short": gameplan.short(p),
                 "share": float(by_pitch.at[p, "share"]),
                 "n": int(by_pitch.at[p, "n"])}
                for p in order if len(cell) and p in by_pitch.index
            ]
            pies.append({
                "situation": situation,
                "label": counts.SITUATION_LABELS[situation],
                "n": int(cell["situation_n"].iloc[0]) if len(cell) else 0,
                "reliable": bool(cell["reliable"].iloc[0]) if len(cell) else False,
                **charts.pie_layout(rows),
            })
        sides.append({"side": side, "label": label, "pies": pies})
    return sides


def _build_card_zones(frame: pd.DataFrame, order: list[str]) -> dict:
    """Where each pitch goes, to each batter side, on Savant's thirteen zones.

    One small map per pitch per side, all counts together: the count-by-count
    slices are on the Location page, where there is room to say how thin each
    one is. A map is drawn only for a pitch thrown `card.zone_map_min_pitches`
    times to that side; below that it carries the pitch count instead, so the
    same pitch keeps the same column in both rows.

    Each cell is the share of that pitch, to that side, finishing there, with
    its fill from `charts.zone_cell`. A cell no pitch reached has no number.
    """
    min_pitches = int(config.get("card.zone_map_min_pitches", 40))
    table = location.zone_slices(frame)
    if len(table):
        tallies = table.groupby(["stand", "pitch_type", "zone"])["n"].sum()
    else:
        tallies = pd.Series(dtype=float)

    def cell(pitch: str, n: int, total: int, zone: int) -> dict:
        share = n / total if total else 0.0
        # A share that rounds to nothing prints as nothing, not as "0".
        return {"zone": zone, "n": n, "share": _clean(share),
                "pct": int(round(share * 100)) or None,
                **charts.zone_cell(pitch, share)}

    sides = []
    for side, label in CARD_SIDES:
        maps = []
        for pitch in order:
            by_zone = {
                zone: int(tallies.get((side, pitch, zone), 0))
                for zone in location.ALL_ZONES
            }
            total = sum(by_zone.values())
            entry = {"pitch": pitch, "short": gameplan.short(pitch),
                     "n": total, "enough": total >= min_pitches}
            if entry["enough"]:
                entry["inner"] = [cell(pitch, by_zone[z], total, z)
                                  for z in location.INNER_ZONES]
                entry["outer"] = [cell(pitch, by_zone[z], total, z)
                                  for z in location.OUTER_ZONES]
            maps.append(entry)
        sides.append({"side": side, "label": label, "maps": maps})
    return {"sides": sides, "min_pitches": min_pitches}


PAGES: tuple[str, ...] = ("card", "detail")


def detail_path(card_path: Path) -> Path:
    """Where the evidence file goes: beside the card, as `<name>_detail.html`."""
    return card_path.with_name(f"{card_path.stem}_detail{card_path.suffix}")


def render(
    payload: dict,
    page: str,
    links: dict[str, str] | None = None,
    template_dir: Path | None = None,
) -> str:
    """Render one of the report's two files from the payload.

    `page` is "card", the one-page dugout card, or "detail", the evidence
    behind it. `links` maps each page to its file name so the two can point
    at each other; without it neither carries a link, which is what a caller
    rendering one page on its own wants.

    Autoescaping is on. Findings are rendered from computed metrics rather
    than user input, but a pitch name arriving from an upstream feed still has
    no business being able to inject markup into a file that gets emailed
    around.
    """
    from jinja2 import Environment, FileSystemLoader

    if page not in PAGES:
        raise ValueError(f"page must be one of {PAGES}, not {page!r}")

    env = Environment(
        loader=FileSystemLoader(str(template_dir or TEMPLATE_DIR)),
        # autoescape=True rather than select_autoescape(["html", "xml"]).
        # select_autoescape decides by file extension, and this template is
        # report.html.j2 — the extension is "j2", so the HTML rule never
        # matched and escaping was silently off. Nothing looked wrong: the
        # page rendered, the tests passed, and a pitch name containing markup
        # would have gone straight into the document. This renders HTML and
        # only HTML, so the unconditional form is both safer and honest about
        # that.
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(TEMPLATE_NAME)

    # The payload goes into the detail file as one JSON blob the page parses
    # on load. Passing it through Jinja as nested structures would mean
    # escaping decisions in a hundred places instead of one. The card draws
    # everything in the template and carries no blob.
    #
    # allow_nan=False makes a stray NaN fail loudly here rather than silently
    # producing a page that will not parse in the browser. The slash escape
    # stops any string containing "</script>" from closing the tag early —
    # pitch names will never do that, but the payload is machine-generated
    # from an upstream feed and one day something will.
    blob = json.dumps(payload, allow_nan=False).replace("</", "<\\/")

    return template.render(payload=payload, payload_json=blob, page=page, links=links)


def write_pages(payload: dict, card_path: Path) -> tuple[Path, Path]:
    """Render both files from one payload and write them side by side.

    Returns (card path, detail path). The detail's name is derived from the
    card's (see `detail_path`), and each file links to the other by name, so
    the pair can be moved or emailed together and the links still work.
    """
    detail = detail_path(card_path)
    links = {"card": card_path.name, "detail": detail.name}
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(render(payload, "card", links), encoding="utf-8")
    detail.write_text(render(payload, "detail", links), encoding="utf-8")
    return card_path, detail


def write(
    frame: pd.DataFrame,
    pitcher: str,
    season: int | str,
    out_path: Path | None = None,
    **league: pd.DataFrame | None,
) -> Path:
    """Build, render, and write the report's two files.

    Returns the card's path, which is `out_path` when one is given. The
    evidence is written beside it, at `detail_path()` of that.
    """
    payload = build_payload(frame, pitcher, season, **league)

    if out_path is None:
        out_dir = ROOT / str(config.get("report.output_dir", "reports"))
        stem = pitcher.lower().replace(" ", "_")
        tag = str(season).replace("–", "-").replace(", ", "_")
        out_path = out_dir / f"{stem}_{tag}.html"

    card, _ = write_pages(payload, out_path)
    return card
