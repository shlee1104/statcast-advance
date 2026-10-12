"""Does each kind of finding come back in games it was not found in?

`validation.py` asks that question of the hitting plan's lines. This module
asks it of everything else the report shows, because a section earns its space
the same way a plan line does: by still being true in the games it was not
built from. Each pitcher's regular season is split into a first and second
half by game (see `validation.split_halves`), and three things are measured.

  findings     Every finding the rules make from the first half, at the
               rules' own bars, re-measured in the second half. Two readings
               are recorded: `held`, the second half clears the same bar on
               its own; and `direction`, it at least still points the same
               way. A finding that holds its direction but not its bar is a
               real tendency the bar happens to sit on top of; one that loses
               both was noise.

  per pitch    Each number in the arsenal, location and release tables, for
               every pitch he threw at least 100 times in both halves, as a
               first-half and second-half pair. Correlated across all
               pitcher-pitches, this says which columns describe the pitch
               and which describe the half.

  per pitcher  Numbers that exist once per pitcher or once per count: the
               velocity change in his worst pitch-count bucket, the
               third-trip changes, count predictability, and how much a
               location band gives away.

A fourth table describes the full season rather than the split: how many
findings of each type the report makes, how many run-value comparisons it
runs, and how many of those survive correction.

Every bar here is read from the same config keys the rules read, and every
first-half finding is selected by the same conditions as the rule that prints
it, so what is tested is what the report would have said. Where the report
prints one finding per pitch and several cells qualified, every qualifying
cell is tested: the question is whether the pattern repeats, not whether the
sentence does.

A second half too thin to measure a finding in is not counted as a failure,
for the same reason as in the holdout check: "too few pitches to say" is not
a wrong call. Those cases produce no row.

A rule that fails this check comes out of the report, and then out of this
module. Two went that way: the verdict on the third trip through the order,
which came back in 1 of 7 cases, and the hittable-pitch finding, which fired
for 2 of 30 pitchers. The run that measured them is reproducible at commit
432abb0, where both rules still exist.

What this cannot tell you: a correlation across a deliberately varied list of
pitchers is higher than it would be across a random sample of the league,
because the list was built to span very different pitchers.
"""

from __future__ import annotations

import pandas as pd

from src import config, flags, gameplan
from src.metrics import arsenal, counts, location, sequencing, splits
from src.validation import wilson_interval

# A half needs this many pitches before anything is measured in it.
MIN_HALF_PITCHES = 200

# A pitch type needs this many pitches in EACH half for its per-pitch numbers
# to be compared, and an xwOBA needs this many plate appearances ending on
# the pitch in each half, since only those carry one.
MIN_PITCHES_PER_HALF = 100
MIN_XWOBA_PAS = 30

# A velocity drop found in the first half is re-measured in the same bucket
# of the second half only if at least this many second-half outings reach it.
MIN_SECOND_HALF_OUTINGS = 5

# Correlations over fewer pairs than this are not reported.
MIN_PAIRS = 6

FINDING_LABELS: dict[str, str] = {
    "count_lift": "Count tendency",
    "sequencing": "Sequencing",
    "location": "Height tell",
    "handedness": "Lefty/righty split",
    "fatigue": "Velocity late in starts",
    "first_pitch": "Predictable first pitch",
}

# Per-pitch columns, in the order they are reported: what he throws, how it
# moves, where it goes, and only then what happened to it.
PITCH_COLUMNS: dict[str, str] = {
    "usage": "Usage",
    "velo": "Velocity",
    "h_break": "Horizontal break",
    "v_break": "Vertical break",
    "loc_height": "Location: height",
    "loc_arm_side": "Location: arm side",
    "release_dist": "Release: distance from fastball slot",
    "zone_rate": "Zone rate",
    "whiff_rate": "Whiff rate",
    "chase_rate": "Chase rate",
    "loc_height_sd": "Command spread: height",
    "loc_arm_side_sd": "Command spread: arm side",
    "loc_top_share": "Top quadrant share",
    "xwoba": "xwOBA",
    "xwoba_vs_lg": "xwOBA vs league",
    "run_value": "Run value per pitch",
}

PITCHER_METRICS: dict[str, str] = {
    "worst_bucket_velo_delta": "Velocity change, worst pitch-count bucket",
    "tto_velo_delta": "Third trip: velocity change",
    "tto_xwoba_delta": "Third trip: xwOBA change",
    "first_pitch_primary_share": "First pitch: share of his usual opener",
    "count_predictability": "Count predictability (per count)",
    "info_gain_v_band": "What a pitch's height gives away",
    "info_gain_h_band": "What a pitch's side gives away",
}

FLAG_TYPES: tuple[str, ...] = (
    "predictable_count", "handedness_gap", "fatigue",
    "sequencing_tell", "location_tell", "first_pitch",
)


def _row(kind: str, key: str, first: float, second: float, n_second: int,
         held: bool, direction: bool) -> dict:
    return {
        "kind": kind, "key": key,
        "first": float(first), "second": float(second),
        "n_second": int(n_second),
        "held": bool(held), "direction": bool(direction),
    }


def _min_z() -> float:
    return float(config.get("flags.thresholds.frequency_min_z", 3.0))


# ---------------------------------------------------------------------------
# Findings: found in the first half, re-measured in the second
# ---------------------------------------------------------------------------


def _count_lift_rows(first: pd.DataFrame, second: pd.DataFrame) -> list[dict]:
    """Count-and-pitch cells at the count rule's bar: is the lift still there?"""
    min_n = int(config.get("flags.min_n.predictable_count", 20))
    min_lift = float(config.get("flags.thresholds.predictable_count_own_lift", 1.50))

    found = counts.count_lifts(first, min_n=min_n)
    if len(found) == 0:
        return []
    found = found[(found["lift"] >= min_lift) & (found["lift_z"] >= _min_z())]

    working = counts.restrict_to_arsenal(second)
    own = counts.pitch_mix(working)
    rows = []
    for r in found.itertuples():
        cell = working[working["count"] == r.count]
        if len(cell) < min_n or r.pitch_type not in own:
            continue
        lift = float((cell["pitch_type"] == r.pitch_type).mean() / own[r.pitch_type])
        rows.append(_row("count_lift", f"{r.count} {r.pitch_type}", r.lift, lift,
                         len(cell), lift >= min_lift, lift > 1.0))
    return rows


def _sequencing_rows(first: pd.DataFrame, second: pd.DataFrame) -> list[dict]:
    """Setup pairs at the sequencing rule's bar: does the follow-up still jump?"""
    min_n = int(config.get("flags.min_n.sequencing_tell", 25))
    min_prob = float(config.get("flags.thresholds.sequencing_tell_prob", 0.65))
    min_lift = float(config.get("flags.thresholds.sequencing_tell_lift", 1.50))

    found = sequencing.setup_pairs(first, min_n=min_n)
    if len(found) == 0:
        return []
    found = found[((found["p_next"] >= min_prob) | (found["freq_lift"] >= min_lift))
                  & (found["lift_z"] >= _min_z())]

    pairs = sequencing._pair_pitches(second)
    if len(pairs) == 0:
        return []
    pairs = pairs.assign(band=pairs["zone"].map(sequencing._zone_band))
    baseline = pairs["next_pitch_type"].value_counts(normalize=True)

    rows = []
    for r in found.itertuples():
        after = pairs[(pairs["pitch_type"] == r.setup_pitch)
                      & (pairs["band"] == r.setup_band)]
        if len(after) < min_n or r.next_pitch not in baseline:
            continue
        # A follow-up he never threw after this setup in the second half is a
        # lift of zero, and it is counted: that is the finding failing.
        lift = float((after["next_pitch_type"] == r.next_pitch).mean()
                     / baseline[r.next_pitch])
        rows.append(_row("sequencing",
                         f"{r.setup_pitch} {r.setup_band} > {r.next_pitch}",
                         r.freq_lift, lift, len(after),
                         lift >= min_lift, lift > 1.0))
    return rows


def _location_rows(first: pd.DataFrame, second: pd.DataFrame) -> list[dict]:
    """Height bands at the location rule's bar: does the band still say the pitch?"""
    min_n = int(config.get("flags.min_n.location_tell", 40))
    min_lift = float(config.get("flags.thresholds.location_tell_lift", 1.35))
    min_share = float(config.get("flags.thresholds.location_tell_share", 0.45))

    found = location.location_tells(first, by="v_band", min_n=min_n)
    if len(found) == 0:
        return []
    fastball_up = ((found["band"] == "UP")
                   & (found["pitch_type"].map(gameplan.PITCH_FAMILY) == "hard"))
    found = found[(found["lift"] >= min_lift) & (found["p_pitch"] >= min_share)
                  & (found["lift_z"] >= _min_z()) & ~fastball_up]

    again = location.location_tells(second, by="v_band", min_n=1)
    rows = []
    for r in found.itertuples():
        match = again[(again["band"] == r.band) & (again["pitch_type"] == r.pitch_type)]
        if len(match) == 0 or int(match.iloc[0]["band_n"]) < min_n:
            continue
        lift = float(match.iloc[0]["lift"])
        share = float(match.iloc[0]["p_pitch"])
        rows.append(_row("location", f"{r.band} {r.pitch_type}", r.lift, lift,
                         int(match.iloc[0]["band_n"]),
                         lift >= min_lift and share >= min_share, lift > 1.0))
    return rows


def _handedness_rows(first: pd.DataFrame, second: pd.DataFrame) -> list[dict]:
    """Usage gaps between batter sides at the handedness rule's bar."""
    min_n = int(config.get("flags.min_n.handedness_gap", 40))
    min_gap = float(config.get("flags.thresholds.handedness_usage_gap", 0.20))

    found, again = splits.platoon_gaps(first), splits.platoon_gaps(second)
    if len(found) == 0 or len(again) == 0:
        return []
    again = again.set_index("pitch_type")

    rows = []
    for r in found.itertuples():
        thrown = max(0 if pd.isna(r.n_L) else r.n_L, 0 if pd.isna(r.n_R) else r.n_R)
        if abs(r.gap) < min_gap or thrown < min_n or r.pitch_type not in again.index:
            continue
        gap = float(again.loc[r.pitch_type, "gap"])
        same_side = (gap > 0) == (r.gap > 0) and gap != 0
        faced = int(min(again.loc[r.pitch_type, "faced_L"],
                        again.loc[r.pitch_type, "faced_R"]))
        rows.append(_row("handedness", r.pitch_type, r.gap, gap, faced,
                         same_side and abs(gap) >= min_gap,
                         same_side and abs(gap) >= min_gap / 2))
    return rows


def _worst_bucket(first: pd.DataFrame, second: pd.DataFrame) -> dict | None:
    """His worst trusted pitch-count bucket in the first half, and the same
    bucket in the second: the pair the velocity rule and its correlation read.
    """
    min_outings = int(config.get("flags.min_n.fatigue_outings", 10))
    early, late = splits.fatigue(first), splits.fatigue(second)
    if len(early) == 0 or len(late) == 0:
        return None
    trusted = early[early["pairs"] >= min_outings]
    if len(trusted) == 0:
        return None
    worst = trusted.loc[trusted["velo_delta"].idxmin()]
    match = late[late["bucket"] == worst["bucket"]]
    if len(match) == 0 or int(match.iloc[0]["pairs"]) < MIN_SECOND_HALF_OUTINGS:
        return None
    return {
        "bucket": str(worst["bucket"]),
        "first": float(worst["velo_delta"]),
        "first_se": float(worst["delta_se"]),
        "second": float(match.iloc[0]["velo_delta"]),
        "pairs_second": int(match.iloc[0]["pairs"]),
    }


def _fatigue_rows(first: pd.DataFrame, second: pd.DataFrame) -> list[dict]:
    """A velocity drop at the fatigue rule's bar: is that bucket still slower?"""
    min_drop = float(config.get("flags.thresholds.fatigue_velo_drop_mph", 1.0))
    bucket = _worst_bucket(first, second)
    if bucket is None:
        return []
    drop, se = -bucket["first"], bucket["first_se"]
    if not (drop >= min_drop and (se != se or drop >= 2 * se)):
        return []
    return [_row("fatigue", bucket["bucket"], bucket["first"], bucket["second"],
                 bucket["pairs_second"],
                 bucket["second"] <= -min_drop,
                 bucket["second"] <= -min_drop / 2)]


def _first_pitch_rows(first: pd.DataFrame, second: pd.DataFrame) -> list[dict]:
    """A predictable opener at the first-pitch rule's bar: still his opener?"""
    min_n = int(config.get("flags.min_n.first_pitch", 30))
    min_share = float(config.get("flags.thresholds.first_pitch_share", 0.70))
    early = counts.first_pitch_tendencies(first)
    late = counts.first_pitch_tendencies(second)
    if early["n"] < min_n or late["n"] < min_n or early["primary_share"] < min_share:
        return []
    share = float(late["mix"].get(early["primary_pitch"], 0.0))
    return [_row("first_pitch", str(early["primary_pitch"]),
                 early["primary_share"], share, late["n"],
                 share >= min_share, share >= min_share - 0.10)]


def finding_rows(first: pd.DataFrame, second: pd.DataFrame) -> list[dict]:
    """Every first-half finding, with what the second half says about it.

    One dict per finding: `kind`, `key` (which count, pitch or bucket),
    `first` and `second` (the number the rule reads, in each half),
    `n_second` (the second-half sample it was re-measured on), `held` and
    `direction`. Returns nothing when either half is under
    MIN_HALF_PITCHES.
    """
    if len(first) < MIN_HALF_PITCHES or len(second) < MIN_HALF_PITCHES:
        return []
    rows: list[dict] = []
    for measure in (_count_lift_rows, _sequencing_rows, _location_rows,
                    _handedness_rows, _fatigue_rows, _first_pitch_rows):
        rows.extend(measure(first, second))
    return rows


# ---------------------------------------------------------------------------
# Descriptive numbers: the same quantity in each half
# ---------------------------------------------------------------------------


def per_pitch_rows(
    first: pd.DataFrame,
    second: pd.DataFrame,
    league_outcomes: pd.DataFrame | None = None,
) -> list[dict]:
    """Each arsenal, location and release number for a pitch, in each half.

    One dict per pitch type thrown at least MIN_PITCHES_PER_HALF times in both
    halves, with every column in PITCH_COLUMNS as `<column>_1` and
    `<column>_2`. `xwoba_n1` / `xwoba_n2` are the plate appearances behind
    each xwOBA, which `pitch_correlations` gates on.

    `xwoba_vs_lg` subtracts the same league figure from both halves, so its
    two columns correlate exactly as raw xwOBA does within one pitch type; it
    differs across pitch types, which is where a correlation over all
    pitcher-pitches gets most of its spread. It is the fairer test of the
    "against the league" column, and it needs `league_outcomes`.
    """
    if len(first) < MIN_HALF_PITCHES or len(second) < MIN_HALF_PITCHES:
        return []
    profiles = arsenal.profile(first), arsenal.profile(second)
    releases = arsenal.release_consistency(first), arsenal.release_consistency(second)
    locations = location.location_profile(first), location.location_profile(second)
    league = (league_outcomes.set_index("pitch_type")["avg_xwoba"]
              if league_outcomes is not None else None)

    rows = []
    for pitch in profiles[0].index.intersection(profiles[1].index):
        n1, n2 = (int(p.loc[pitch, "n"]) for p in profiles)
        if n1 < MIN_PITCHES_PER_HALF or n2 < MIN_PITCHES_PER_HALF:
            continue
        row: dict = {"pitch": pitch, "n1": n1, "n2": n2}
        for column in ("usage", "velo", "whiff_rate", "chase_rate", "zone_rate",
                       "xwoba", "run_value", "h_break", "v_break"):
            for half, profile in enumerate(profiles, start=1):
                row[f"{column}_{half}"] = float(profile.loc[pitch, column])
        for half, profile in enumerate(profiles, start=1):
            row[f"xwoba_n{half}"] = int(profile.loc[pitch, "xwoba_n"])
        if all(pitch in r.index for r in releases):
            for half, release in enumerate(releases, start=1):
                row[f"release_dist_{half}"] = float(release.loc[pitch, "dist_from_ref"])
        if all(pitch in p.index for p in locations):
            for column in ("height", "arm_side", "height_sd", "arm_side_sd", "top_share"):
                for half, profile in enumerate(locations, start=1):
                    row[f"loc_{column}_{half}"] = float(profile.loc[pitch, column])
        if league is not None and pitch in league.index:
            for half in (1, 2):
                row[f"xwoba_vs_lg_{half}"] = row[f"xwoba_{half}"] - float(league[pitch])
        rows.append(row)
    return rows


def _third_trip(frame: pd.DataFrame) -> dict | None:
    """What changes from the first trip through the order to the third.

    Velocity is the within-outing change the by-trip table prints; xwOBA is
    the difference between the two trips' plain figures. None unless both
    trips clear the table's outings gate.
    """
    trips = splits.times_through_order(frame)
    if len(trips) == 0:
        return None
    trips = trips.set_index("tto")
    if 1 not in trips.index or 3 not in trips.index:
        return None
    first, third = trips.loc[1], trips.loc[3]
    if not (bool(first["reliable"]) and bool(third["reliable"])):
        return None
    return {
        "tto_velo_delta": float(third["velo_delta"]),
        "tto_xwoba_delta": float(third["xwoba"] - first["xwoba"]),
    }


def per_pitcher_rows(first: pd.DataFrame, second: pd.DataFrame) -> list[dict]:
    """Numbers that exist once per pitcher, or once per count, in each half.

    One dict per number: `metric` (a key of PITCHER_METRICS), `key` (the
    count, for the per-count metric), `first` and `second`.
    """
    if len(first) < MIN_HALF_PITCHES or len(second) < MIN_HALF_PITCHES:
        return []
    rows: list[dict] = []

    def add(metric: str, a: float, b: float, key: str = "") -> None:
        rows.append({"metric": metric, "key": key,
                     "first": float(a), "second": float(b)})

    bucket = _worst_bucket(first, second)
    if bucket is not None:
        add("worst_bucket_velo_delta", bucket["first"], bucket["second"])

    early, late = _third_trip(first), _third_trip(second)
    if early is not None and late is not None:
        for metric in ("tto_velo_delta", "tto_xwoba_delta"):
            if early[metric] == early[metric] and late[metric] == late[metric]:
                add(metric, early[metric], late[metric])

    min_n = int(config.get("flags.min_n.first_pitch", 30))
    opens = counts.first_pitch_tendencies(first), counts.first_pitch_tendencies(second)
    if opens[0]["n"] >= min_n and opens[1]["n"] >= min_n:
        add("first_pitch_primary_share", opens[0]["primary_share"],
            float(opens[1]["mix"].get(opens[0]["primary_pitch"], 0.0)))

    scores = counts.predictability(first), counts.predictability(second)
    for count in scores[0].index.intersection(scores[1].index):
        add("count_predictability", scores[0].loc[count, "predictability"],
            scores[1].loc[count, "predictability"], key=str(count))

    for band in ("v_band", "h_band"):
        info = (location.band_information(first, by=band),
                location.band_information(second, by=band))
        if info[0]["n"] and info[1]["n"]:
            add(f"info_gain_{band}", info[0]["info_gain_pct"], info[1]["info_gain_pct"])
    return rows


# ---------------------------------------------------------------------------
# The full season: what the report itself would print
# ---------------------------------------------------------------------------


def full_season_row(frame: pd.DataFrame) -> dict:
    """How many findings the report makes, and how many comparisons it runs.

    `comparisons` is every run-value comparison behind the report,
    `raw_p05` how many of them clear an uncorrected p < .05, `bh_survivors`
    how many survive Benjamini-Hochberg, and `resolved` how many of the
    findings carry a run consequence that survived. One `n_<flag>` column per
    finding type.
    """
    found = flags.evaluate(frame)
    family = flags.comparison_p_values(frame)
    p_values = [p for values in family.values() for p in values if p == p]
    row = {
        "pitches": int(len(frame)),
        "games": int(frame["game_pk"].nunique()),
        "comparisons": len(p_values),
        "raw_p05": int(sum(p < 0.05 for p in p_values)),
        "bh_survivors": int(sum(flags.benjamini_hochberg(p_values))),
        "findings": int(len(found)),
        "resolved": int(found["resolved"].sum()) if len(found) else 0,
    }
    for flag in FLAG_TYPES:
        row[f"n_{flag}"] = int((found["flag"] == flag).sum()) if len(found) else 0
    return row


# ---------------------------------------------------------------------------
# Summarising across pitchers
# ---------------------------------------------------------------------------


def _correlation(pairs: pd.DataFrame) -> float:
    pairs = pairs.dropna()
    if len(pairs) < MIN_PAIRS:
        return float("nan")
    return float(pairs.iloc[:, 0].corr(pairs.iloc[:, 1]))


def summarise_findings(rows: list[dict]) -> list[dict]:
    """One line per finding type: how often it came back.

    `held` and `direction` each carry a 95% Wilson interval, for the same
    reason the holdout rates do: 1 of 5 is a rate somewhere between 4% and
    62%, not a rate of 20%.
    """
    table = pd.DataFrame(rows)
    out: list[dict] = []
    if table.empty:
        return out
    for kind, label in FINDING_LABELS.items():
        sub = table[table["kind"] == kind]
        if sub.empty:
            continue
        n = len(sub)
        held, same_way = int(sub["held"].sum()), int(sub["direction"].sum())
        held_lo, held_hi = wilson_interval(held, n)
        dir_lo, dir_hi = wilson_interval(same_way, n)
        out.append({
            "kind": kind, "label": label, "found": n,
            "pitchers": (int(sub["pitcher"].nunique())
                         if "pitcher" in sub.columns else float("nan")),
            "held": held, "held_pct": held / n,
            "held_lo": held_lo, "held_hi": held_hi,
            "direction": same_way, "direction_pct": same_way / n,
            "direction_lo": dir_lo, "direction_hi": dir_hi,
            "median_first": float(sub["first"].median()),
            "median_second": float(sub["second"].median()),
        })
    return out


def pitch_correlations(rows: list[dict]) -> list[dict]:
    """Split-half correlation of each per-pitch number, across pitcher-pitches.

    The xwOBA columns use only pitches with MIN_XWOBA_PAS plate appearances
    ending on them in each half. Columns with fewer than MIN_PAIRS pairs are
    left out rather than reported off a handful.
    """
    table = pd.DataFrame(rows)
    out: list[dict] = []
    if table.empty:
        return out
    for column, label in PITCH_COLUMNS.items():
        a, b = f"{column}_1", f"{column}_2"
        if a not in table.columns:
            continue
        sub = table
        if column.startswith("xwoba"):
            sub = table[(table["xwoba_n1"] >= MIN_XWOBA_PAS)
                        & (table["xwoba_n2"] >= MIN_XWOBA_PAS)]
        pairs = sub[[a, b]].dropna()
        r = _correlation(pairs)
        if r == r:
            out.append({"column": column, "label": label, "r": r, "pairs": len(pairs)})
    return out


def pitcher_correlations(rows: list[dict]) -> list[dict]:
    """Split-half correlation of each per-pitcher number."""
    table = pd.DataFrame(rows)
    out: list[dict] = []
    if table.empty:
        return out
    for metric, label in PITCHER_METRICS.items():
        pairs = table[table["metric"] == metric][["first", "second"]].dropna()
        r = _correlation(pairs)
        if r == r:
            out.append({"metric": metric, "label": label, "r": r,
                        "pairs": len(pairs),
                        "mean_abs_first": float(pairs["first"].abs().mean())})
    return out


def summarise_full_season(rows: list[dict]) -> dict:
    """Totals across the pitchers' full-season reports.

    `by_flag` gives, for each finding type, the number of pitchers it fired
    for and the number of findings it produced.
    """
    table = pd.DataFrame(rows)
    if table.empty:
        return {"pitchers": 0, "by_flag": []}
    by_flag = [
        {"flag": flag,
         "pitchers": int((table[f"n_{flag}"] > 0).sum()),
         "findings": int(table[f"n_{flag}"].sum())}
        for flag in FLAG_TYPES if f"n_{flag}" in table.columns
    ]
    comparisons = int(table["comparisons"].sum())
    return {
        "pitchers": int(len(table)),
        "by_flag": by_flag,
        "comparisons": comparisons,
        "comparisons_median": float(table["comparisons"].median()),
        "raw_p05": int(table["raw_p05"].sum()),
        "raw_p05_pct": (float(table["raw_p05"].sum() / comparisons)
                        if comparisons else float("nan")),
        "bh_survivors": int(table["bh_survivors"].sum()),
        "findings": int(table["findings"].sum()),
        "resolved": int(table["resolved"].sum()),
    }
