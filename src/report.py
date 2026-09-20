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

from src import baselines, config, flags
from src.metrics import arsenal, counts, events, location, sequencing, splits

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
    season: int,
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

    hand = str(frame["p_throws"].mode().iloc[0])
    has_league = league_outcomes is not None

    profile = arsenal.profile(frame)
    if has_league:
        arsenal_rows = _records(baselines.compare_profile(profile, league_outcomes))
    else:
        arsenal_rows = _records(profile, index_name="pitch_type")

    mix_table = counts.mix_by_count(frame, min_pitches=5)
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
            "hand": hand,
            "pitches": int(len(frame)),
            "games": int(frame["game_pk"].nunique()),
            "batters_faced": int(frame.groupby(["game_pk", "at_bat_number"]).ngroups),
            "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "has_league": has_league,
            "league_note": (
                f"Compared against {hand}HP league baselines"
                if has_league
                else "No league baselines loaded — deviations unavailable"
            ),
        },
        "data_gaps": _data_gaps(frame),
        "kpis": _build_kpis(frame, profile, first_pitch, predictability),
        "takeaways": _records(takeaways),
        "findings": _records(all_findings),
        "finding_count": int(len(all_findings)),
        "resolved_count": int(all_findings["resolved"].sum()) if len(all_findings) else 0,
        "comparisons": flags.comparison_count(frame, league_mix),
        "arsenal": arsenal_rows,
        "display_names": _display_names(frame),
        "count_mix": {
            "counts": list(mix_table.index),
            "pitches": list(mix_table.columns),
            "series": {
                str(pitch): [_clean(v) for v in mix_table[pitch].tolist()]
                for pitch in mix_table.columns
            },
        },
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


def _build_kpis(
    frame: pd.DataFrame,
    profile: pd.DataFrame,
    first_pitch: dict,
    predictability: pd.DataFrame,
) -> list[dict]:
    """The four headline numbers, chosen to orient rather than to impress.

    Arsenal size and overall whiff rate say what kind of pitcher this is;
    first-pitch strike rate is the number every pitching coach already knows
    by heart; the most predictable count is the one a hitter can use.
    """
    whiff = events.whiff_rate(frame)
    top_count = None
    if len(predictability):
        best = predictability["predictability"].idxmax()
        top_count = {
            "count": str(best),
            "pitch": str(predictability.loc[best, "top_pitch"]),
            "share": _clean(predictability.loc[best, "top_share"]),
        }

    kpis = [
        {
            "label": "Pitches",
            "value": f"{len(frame):,}",
            "note": f"{frame['game_pk'].nunique()} games",
        },
        {
            "label": "Arsenal",
            "value": str(len(profile)),
            "note": ", ".join(profile.index) if len(profile) else "—",
        },
        {
            "label": "Whiff rate",
            "value": f"{whiff:.1%}" if not math.isnan(whiff) else "—",
            "note": "per swing",
        },
        {
            "label": "First-pitch strikes",
            "value": (
                f"{first_pitch['strike_rate']:.1%}"
                if first_pitch["n"] else "—"
            ),
            "note": f"n={first_pitch['n']}",
        },
    ]

    if top_count:
        kpis.append({
            "label": "Most predictable count",
            "value": top_count["count"],
            "note": f"{top_count['pitch']} {top_count['share']:.0%}",
        })

    return kpis


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
    season: int,
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
        out_path = out_dir / f"{stem}_{season}.html"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
