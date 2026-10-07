"""Assemble every metric into one payload and render the report.

The report is a single self-contained HTML file. No server, no external data
fetch, no build step — a coach gets sent one file, opens it, and it works, on a
phone in a parking lot if that is where he is. That constraint is why all data
is embedded as JSON rather than loaded, and why the only external dependency
is a charting library from a CDN with a graceful fallback to tables.

Two layers live here:

  build_payload()  runs the metrics modules and flattens everything into a
                   JSON-serializable dict. All the baseball happens here.
  render()         hands that dict to a Jinja2 template. No computation.

Keeping them apart means the payload can be inspected, diffed between runs, or
fed to something else entirely without touching presentation — and the
template cannot quietly introduce a number that no metric produced.

Takeaways lead the page. Everything below them is the evidence for them, in
the order a reader would want it while checking whether to believe a claim.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src import baselines, charts, config, flags, gameplan
from src.metrics import arsenal, counts, events, location, seasons, sequencing, splits

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = ROOT / "templates"
TEMPLATE_NAME = "report.html.j2"

# The zone grid is defined alongside the data that uses it, not here — the
# report renders what location.py describes.
INNER_ZONES = location.INNER_ZONES
OUTER_ZONES = location.OUTER_ZONES


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


# What each finding type is called on the page. The codes are for the
# program; a coach reading "location_tell" next to a finding learns nothing.
FLAG_LABELS: dict[str, str] = {
    "predictable_count": "Count tendency",
    "handedness_gap": "Lefty/righty split",
    "hittable_pitch": "Hittable pitch",
    "fatigue": "Velocity late in starts",
    "times_through_order": "Third time through",
    "sequencing_tell": "Sequencing",
    "location_tell": "Height tell",
    "first_pitch": "First pitch",
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

    profile = arsenal.profile(frame)
    if has_league:
        arsenal_rows = _records(baselines.compare_profile(profile, league_outcomes))
    else:
        arsenal_rows = _records(profile, index_name="pitch_type")

    predictability = counts.predictability(frame, min_pitches=20)
    if has_league and league_mix is not None and len(predictability):
        predictability_rows = _records(
            baselines.compare_predictability(predictability, league_mix)
        )
    else:
        predictability_rows = _records(predictability, index_name="count")

    putaway = sequencing.putaway(frame, min_pitches=10)
    if has_league and league_putaway is not None and len(putaway):
        putaway_rows = _records(baselines.compare_putaway(putaway, league_putaway))
    else:
        putaway_rows = _records(putaway, index_name="pitch_type")

    zone = location.zone_table(frame)
    fatigue = splits.fatigue(frame, bucket_size=15)
    first_pitch = counts.first_pitch_tendencies(frame)

    takeaways = flags.takeaways(frame, league_outcomes, league_mix)
    all_findings = flags.evaluate(frame, league_outcomes, league_mix)

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
        "card": _build_card(frame, changes=changes, season_frame=full_frame),
        "takeaways": _records(takeaways),
        "findings": _records(all_findings),
        "finding_count": int(len(all_findings)),
        "resolved_count": int(all_findings["resolved"].sum()) if len(all_findings) else 0,
        "comparisons": flags.comparison_count(frame, league_mix),
        "arsenal": arsenal_rows,
        "display_names": _display_names(frame),
        "flag_labels": FLAG_LABELS,
        "count_grid": _build_count_grid(frame),
        "platoon_slope": _build_platoon_slope(frame),
        "movement": _build_movement(frame, league_outcomes),
        "predictability": predictability_rows,
        "count_lifts": _records(counts.count_lifts(frame, min_n=20)),
        "putaway": putaway_rows,
        "setups": _records(sequencing.setup_pairs(frame, min_n=25).head(12)),
        "location_profile": _records(
            location.location_profile(frame), index_name="pitch_type"
        ),
        "location_tells": _records(location.location_tells(frame).head(12)),
        "quadrants": _records(location.quadrant_shares(frame), index_name="pitch_type"),
        "band_information": {
            band: {k: _clean(v) for k, v in location.band_information(frame, by=band).items()}
            for band in ("v_band", "h_band", "quadrant")
        },
        "zone": {
            "inner": INNER_ZONES,
            "outer": OUTER_ZONES,
            "rows": _records(zone, index_name="pitch_type"),
        },
        "zone_slices": _build_zone_slices(frame),
        "release": _records(arsenal.release_consistency(frame), index_name="pitch_type"),
        "platoon": _records(splits.platoon_gaps(frame)),
        "fatigue": _build_fatigue(fatigue),
        "tto": _records(splits.times_through_order(frame)),
        "tto_decomposition": {
            k: _clean(v) for k, v in splits.decompose_tto(frame).items()
        },
        "first_pitch": {
            "n": _clean(first_pitch["n"]),
            "strike_rate": _clean(first_pitch["strike_rate"]),
            "primary_pitch": _clean(first_pitch["primary_pitch"]),
            "primary_share": _clean(first_pitch["primary_share"]),
        },
        "gates": {
            "note": (
                "Every claim states its sample size. Thresholds and minimum "
                "samples are defined in config.yaml, not in code."
            ),
            "max_takeaways": int(config.get("report.max_takeaways", 5)),
        },
    }


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
    opposite. The same principle governs the count grid and the platoon table.
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


def _build_fatigue(fatigue: pd.DataFrame) -> dict:
    """Velocity-by-pitch-count series, with each bucket marked trustworthy or not.

    A late bucket is built from the starts that got that deep, which are the
    starts that were going well — so the tail of this curve suffers from
    survivorship, not just small samples. Yamamoto's last bucket is 12 pitches
    across 4 outings and shows velocity rising 0.76 mph, which reads as
    recovery and is actually selection.

    Buckets below `flags.min_n.fatigue_outings` separate outings are marked
    unreliable rather than dropped: removing them would make the curve look
    cleaner than the evidence, and a reader would not know the line had been
    trimmed. The chart greys them; the number stays available.
    """
    min_outings = int(config.get("flags.min_n.fatigue_outings", 10))

    if len(fatigue) == 0:
        return {"buckets": [], "velo": [], "delta": [], "outings": [], "n": [],
                "reliable": [], "min_outings": min_outings}

    return {
        "buckets": list(fatigue["bucket"]),
        "velo": [_clean(v) for v in fatigue["velo"]],
        "delta": [_clean(v) for v in fatigue["velo_delta"]],
        "outings": [_clean(v) for v in fatigue["outings"]],
        "n": [_clean(v) for v in fatigue["n"]],
        "reliable": [bool(v >= min_outings) for v in fatigue["outings"]],
        "min_outings": min_outings,
    }


def _build_card(
    frame: pd.DataFrame,
    last_n: int = 5,
    changes: dict | None = None,
    season_frame: pd.DataFrame | None = None,
) -> dict:
    """The dugout card: page one of the report, the part a coach takes out.

    Front offices describe paring "a phone book's amount of information" down
    to a few pages, with one-sentence notes for use mid-game. This is that
    page. It carries five things, in the order a hitter reads them: the season
    line, the hitting plan split by side, the arsenal as a scout writes it,
    the count tendencies split by side, and what he has done lately.

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

    return {
        "season": {k: _clean(v) for k, v in events.season_line(
            frame if season_frame is None else season_frame).items()},
        "arsenal": _records(arsenal.scout_arsenal(frame), index_name="pitch_type"),
        "tendencies": tendencies,
        "recent": {
            "last_n": form["last_n"],
            "dates": form["dates"],
            "fastball": form["fastball"],
            "velo_recent": _clean(form["velo_recent"]),
            "velo_earlier": _clean(form["velo_earlier"]),
            "velo_delta": _clean(form["velo_delta"]),
            "earlier_label": form["earlier_label"],
            "n_recent": form["n_recent"],
            "starts": _records(form["starts"]),
            "shifts": _records(form["shifts"]),
        },
        "plan": gameplan.build(frame, last_n=last_n, changes=changes),
        "coach_names": gameplan.COACH_NAMES,
    }


def render(payload: dict, template_dir: Path | None = None) -> str:
    """Render the payload into HTML.

    Autoescaping is on. Findings are rendered from computed metrics rather
    than user input, but a pitch name arriving from an upstream feed still has
    no business being able to inject markup into a file that gets emailed
    around.
    """
    from jinja2 import Environment, FileSystemLoader

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

    # The payload goes in as one JSON blob the page parses on load. Passing it
    # through Jinja as nested structures would mean escaping decisions in a
    # hundred places instead of one.
    #
    # allow_nan=False makes a stray NaN fail loudly here rather than silently
    # producing a page that will not parse in the browser. The slash escape
    # stops any string containing "</script>" from closing the tag early —
    # pitch names will never do that, but the payload is machine-generated
    # from an upstream feed and one day something will.
    blob = json.dumps(payload, allow_nan=False).replace("</", "<\\/")

    return template.render(payload=payload, payload_json=blob)


def write(
    frame: pd.DataFrame,
    pitcher: str,
    season: int | str,
    out_path: Path | None = None,
    **league: pd.DataFrame | None,
) -> Path:
    """Build, render, and write the report. Returns the path written."""
    payload = build_payload(frame, pitcher, season, **league)
    html = render(payload)

    if out_path is None:
        out_dir = ROOT / str(config.get("report.output_dir", "reports"))
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = pitcher.lower().replace(" ", "_")
        tag = str(season).replace("–", "-").replace(", ", "_")
        out_path = out_dir / f"{stem}_{tag}.html"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
