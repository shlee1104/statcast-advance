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

  tunneling    Each pair of pitch types he throws back to back to one batter
               side: how far apart the two are at release, at the decision
               point and at the plate, and how differently they move late,
               in each half. Reported twice over: across all pairs, and
               again after each pair type's average across pitchers is taken
               out, because a fastball and a curveball are far apart for
               everyone and that is not a fact about this pitcher.

  card         What the one-page dugout card draws: every slice of its
               situation pies and every cell of its zone maps, by batter
               side, in each half. Reported as how far a number moves between
               halves, in points, as well as a correlation, because a hitter
               reads the number.

Another table describes the full season rather than the split: how many
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
in 5 of 58 pitcher-seasons. The run that measured them is reproducible at
commit 432abb0, where both rules still exist.

What this cannot tell you: a correlation across a deliberately varied list of
pitchers is higher than it would be across a random sample of the league,
because the list was built to span very different pitchers.
"""

from __future__ import annotations

import pandas as pd

from src import config, flags, gameplan, report
from src.metrics import arsenal, counts, location, sequencing, splits, tunneling
from src.validation import pitcher_interval, wilson_interval

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
    "follow_up_more": "Follow-up: more than usual",
    "follow_up_less": "Follow-up: less than usual",
    "location": "Height tell",
    "handedness": "Lefty/righty split",
    "fatigue": "Velocity late in starts",
    "first_pitch": "Predictable first pitch",
    "tunnel_same": "Tunneling: same window",
    "tunnel_apart": "Tunneling: apart early",
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

# What the dugout card draws that no other table here covers: the per-pitch
# columns above are season-wide, and the card splits by batter side.
CARD_METRICS: dict[str, str] = {
    "pie_share": "Card pie: a pitch's share of a situation, by side",
    "zone_share": "Card zone map: a pitch's share of a zone, by side",
}

# What the tunneling table holds for a pair of pitch types. Inches, except
# d' (a ratio) and the flight time.
TUNNEL_METRICS: dict[str, str] = {
    # Mean distances between the two pitches, as published metrics define them.
    "release": "Mean distance at release",
    "tunnel": "Mean distance at the decision point",
    "tunnel_vs_floor": "  the same, above his same-pitch floor",
    "late": "Mean distance in late movement",
    "plate": "Mean distance at the plate",
    # Offsets: the length of the average difference, scatter cancelled.
    "release_offset": "Offset at release",
    "tunnel_offset": "Offset at the decision point",
    "tunnel_d": "  the same, over his scatter (d')",
    "late_offset": "Offset in late movement",
    "plate_offset": "Offset at the plate",
    "flight_ms": "Difference in flight time (ms)",
}

# The evidence file's setup table, column by column. The last two were
# printed until this check showed they do not repeat; they are measured here
# so that stays rerunnable.
SETUP_METRICS: dict[str, str] = {
    "p_next": "Chance next",
    "freq_lift": "How much more likely",
    "effect_lift": "Whiff change (no longer printed)",
    "xwoba_when": "xwOBA after (no longer printed)",
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


def _follow_up_rows(first: pd.DataFrame, second: pd.DataFrame) -> list[dict]:
    """The follow-ups the sequence graph would colour: still off his usual?

    Read through `sequencing.transition_standouts`, the rule the graph itself
    uses. Re-measured on the same batter side, where the pitch needs the
    rule's own number of follow-ups in the second half for its row to be
    read at all.
    """
    row_gate = int(config.get("flags.min_n.sequence_follow_ups", 40))
    more_lift = float(config.get("flags.thresholds.sequence_more_lift", 1.25))
    less_lift = float(config.get("flags.thresholds.sequence_less_lift", 0.80))

    found = sequencing.transitions(first)
    if len(found) == 0:
        return []
    found = found.assign(standout=sequencing.transition_standouts(found))
    found = found[found["standout"] != ""]
    again = sequencing.transitions(second)
    if len(found) == 0 or len(again) == 0:
        return []
    again = again.set_index(["stand", "pitch_type", "next_pitch"])

    rows = []
    for r in found.itertuples():
        key = (r.stand, r.pitch_type, r.next_pitch)
        if key not in again.index:
            continue
        cell = again.loc[key]
        if int(cell["row_n"]) < row_gate or not cell["lift"] == cell["lift"]:
            continue
        lift = float(cell["lift"])
        label = f"{r.pitch_type} > {r.next_pitch} to {r.stand}HH"
        if r.standout == "more":
            rows.append(_row("follow_up_more", label, r.lift, lift, int(cell["row_n"]),
                             lift >= more_lift, lift > 1.0))
        else:
            rows.append(_row("follow_up_less", label, r.lift, lift, int(cell["row_n"]),
                             lift <= less_lift, lift < 1.0))
    return rows


def _tunnel_reading_rows(first: pd.DataFrame, second: pd.DataFrame) -> list[dict]:
    """Pairs the tunneling table would put a word on: the same word again?

    Read through `tunneling.reading`, the rule the table itself uses, on the
    pair's offset at the decision point over his scatter (d'). Re-measured
    where the same pair to the same side clears the table's gate in the
    second half. It held if the second half earns the same word; it kept its
    direction if it stayed on the same side of the midpoint between the two
    bars.
    """
    found = tunneling.pair_table(first)
    again = tunneling.pair_table(second)
    if len(found) == 0 or len(again) == 0:
        return []
    found = found[(found["reading"] != "") & (found["pitch_type"] != found["next_pitch"])]
    again = again.set_index(["stand", "pitch_type", "next_pitch"])
    same_bar = float(config.get("tunneling.same_window_d", 0.75))
    apart_bar = float(config.get("tunneling.apart_d", 1.25))
    middle = (same_bar + apart_bar) / 2.0

    rows = []
    for r in found.itertuples():
        key = (r.stand, r.pitch_type, r.next_pitch)
        if key not in again.index:
            continue
        cell = again.loc[key]
        d = float(cell["tunnel_d"])
        if d != d:
            continue
        label = f"{r.pitch_type} > {r.next_pitch} to {r.stand}HH"
        if r.reading == tunneling.SAME_WINDOW:
            rows.append(_row("tunnel_same", label, r.tunnel_d, d, int(cell["n"]),
                             d <= same_bar, d < middle))
        else:
            rows.append(_row("tunnel_apart", label, r.tunnel_d, d, int(cell["n"]),
                             d >= apart_bar, d > middle))
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
    for measure in (_count_lift_rows, _sequencing_rows, _follow_up_rows,
                    _tunnel_reading_rows, _location_rows, _handedness_rows,
                    _fatigue_rows, _first_pitch_rows):
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
# The dugout card: its pies and its zone maps
# ---------------------------------------------------------------------------


def card_rows(first: pd.DataFrame, second: pd.DataFrame) -> list[dict]:
    """Every number the card's pies and zone maps print, in each half.

    One dict per number: `metric` (a key of CARD_METRICS), `key`, `first` and
    `second` (shares), and `n_first` / `n_second`, the pitches each share is
    a share of.

    A pie is compared only where the card would draw it unfaded in both
    halves, and a zone map only where the card would draw it at all in both
    (`card.zone_map_min_pitches`): those are the pictures a reader is asked
    to trust, so those are the ones tested. A pitch in one half's mix and not
    the other's is compared at a share of zero, not left out — a slice that
    disappears is the largest change a pie can make.
    """
    if len(first) < MIN_HALF_PITCHES or len(second) < MIN_HALF_PITCHES:
        return []
    rows: list[dict] = []

    usage = counts.situational_usage(first), counts.situational_usage(second)
    for side in ("L", "R"):
        for situation in counts.SITUATIONS:
            cells = [u[(u["stand"] == side) & (u["situation"] == situation)]
                     for u in usage]
            if any(len(c) == 0 or not bool(c["reliable"].iloc[0]) for c in cells):
                continue
            shares = [c.set_index("pitch_type")["share"] for c in cells]
            totals = [int(c["situation_n"].iloc[0]) for c in cells]
            for pitch in sorted(set(shares[0].index) | set(shares[1].index)):
                rows.append({
                    "metric": "pie_share", "key": f"{side}/{situation}/{pitch}",
                    "first": float(shares[0].get(pitch, 0.0)),
                    "second": float(shares[1].get(pitch, 0.0)),
                    "n_first": totals[0], "n_second": totals[1],
                })

    min_pitches = int(config.get("card.zone_map_min_pitches", 40))
    slices = location.zone_slices(first), location.zone_slices(second)
    if any(len(t) == 0 for t in slices):
        return rows
    tallies = [t.groupby(["stand", "pitch_type", "zone"])["n"].sum() for t in slices]
    totals = [t.groupby(level=["stand", "pitch_type"]).sum() for t in tallies]
    for side, pitch in totals[0].index.intersection(totals[1].index):
        n1, n2 = (int(t[(side, pitch)]) for t in totals)
        if n1 < min_pitches or n2 < min_pitches:
            continue
        for zone in location.ALL_ZONES:
            rows.append({
                "metric": "zone_share", "key": f"{side}/{pitch}/{zone}",
                "first": float(tallies[0].get((side, pitch, zone), 0)) / n1,
                "second": float(tallies[1].get((side, pitch, zone), 0)) / n2,
                "n_first": n1, "n_second": n2,
            })
    return rows


def card_summary(rows: list[dict]) -> list[dict]:
    """How far each kind of card number moves from one half to the other.

    Per metric: the split-half correlation, the median and 90th-percentile
    absolute change in share, and the share of numbers that moved less than
    five points. The correlation alone would flatter both: a pitch he throws
    40% of the time and one he throws 4% are far apart in either half, which
    is most of what it measures. The change in points is what a reader of the
    number is exposed to.

    `same_top` is the reading a hitter actually takes from the picture: of
    the `pictures` compared (pies, or zone maps), the share whose largest
    slice or busiest zone was the same one in both halves.
    """
    table = pd.DataFrame(rows)
    out: list[dict] = []
    if table.empty:
        return out
    if "pitcher" not in table.columns:
        table["pitcher"] = ""
    # A key is side/situation/pitch or side/pitch/zone: all but the last part
    # names the picture the number belongs to.
    table["picture"] = table["pitcher"] + "/" + table["key"].str.rsplit("/", n=1).str[0]
    for metric, label in CARD_METRICS.items():
        sub = table[table["metric"] == metric].dropna(subset=["first", "second"])
        r = _correlation(sub[["first", "second"]])
        if r != r:
            continue
        moved = (sub["second"] - sub["first"]).abs()
        same = [
            group.loc[group["first"].idxmax(), "key"]
            == group.loc[group["second"].idxmax(), "key"]
            for _, group in sub.groupby("picture")
        ]
        out.append({
            "metric": metric, "label": label, "r": r, "pairs": len(sub),
            "median_change": float(moved.median()),
            "p90_change": float(moved.quantile(0.90)),
            "within_five": float((moved < 0.05).mean()),
            "pictures": len(same),
            "same_top": sum(same) / len(same),
            "pitchers": int(sub["pitcher"].nunique()),
        })
    return out


# ---------------------------------------------------------------------------
# Tunneling: pairs of pitch types, at the decision point and after it
# ---------------------------------------------------------------------------


def tunnel_rows(first: pd.DataFrame, second: pd.DataFrame) -> list[dict]:
    """Every row of the tunneling table that both halves would print.

    One dict per (side, pitch, follow-up) that clears the table's own gate,
    `tunneling.min_pairs`, in each half: `key` ("L/FF/FS"), `pair` ("FF/FS",
    the same for every pitcher and side), `n_first` and `n_second`, and each
    of TUNNEL_METRICS as `<metric>_1` and `<metric>_2`. The same pitch thrown
    twice is left out: it is the floor the others are read against, not a
    pair anyone is told about.

    Empty when either half has no trajectory, which is what a cache row
    fetched before the fit columns were stored looks like.
    """
    if len(first) < MIN_HALF_PITCHES or len(second) < MIN_HALF_PITCHES:
        return []
    tables = tunneling.pair_table(first), tunneling.pair_table(second)
    if any(len(t) == 0 for t in tables):
        return []

    keys = ["stand", "pitch_type", "next_pitch"]
    both = tables[0].merge(tables[1], on=keys, suffixes=("_1", "_2"))
    both = both[both["pitch_type"] != both["next_pitch"]]
    rows = []
    for r in both.to_dict("records"):
        row = {"key": f"{r['stand']}/{r['pitch_type']}/{r['next_pitch']}",
               "pair": f"{r['pitch_type']}/{r['next_pitch']}",
               "n_first": int(r["n_1"]), "n_second": int(r["n_2"])}
        for metric in TUNNEL_METRICS:
            row[f"{metric}_1"] = float(r[f"{metric}_1"])
            row[f"{metric}_2"] = float(r[f"{metric}_2"])
        rows.append(row)
    return rows


def tunnel_summary(rows: list[dict]) -> list[dict]:
    """Whether a pair's tunneling numbers are the pitcher's or the half's.

    Per metric: `r`, the split-half correlation across every pitcher's pairs;
    the median and 90th-percentile absolute change between halves; and
    `r_within`, the same correlation after subtracting, from both halves,
    that pair type's average over every pitcher who throws it (`within_n`
    rows, from pair types at least three pitcher-sides share).

    The first correlation is easy to pass. A four-seam and a curveball are a
    foot apart late for every pitcher alive, so a table of pairs correlates
    with itself across halves however little it says about anyone. The second
    asks the question a scouting report needs answered: among pitchers who all
    throw four-seam then slider, is this one's pair tighter than the others'
    in both halves?
    """
    table = pd.DataFrame(rows)
    out: list[dict] = []
    if table.empty:
        return out
    shared = table.groupby("pair")["pair"].transform("size") >= 3
    for metric, label in TUNNEL_METRICS.items():
        a, b = f"{metric}_1", f"{metric}_2"
        pairs = table[[a, b]].dropna()
        r = _correlation(pairs.rename(columns={a: "first", b: "second"}))
        if r != r:
            continue
        moved = (pairs[b] - pairs[a]).abs()

        inside = table[shared].dropna(subset=[a, b])
        centre = inside.groupby("pair")[[a, b]].transform("mean").mean(axis=1)
        resid = pd.DataFrame({"first": inside[a] - centre, "second": inside[b] - centre})
        out.append({
            "metric": metric, "label": label, "r": r, "pairs": len(pairs),
            "median_change": float(moved.median()),
            "p90_change": float(moved.quantile(0.90)),
            "r_within": _correlation(resid), "within_n": len(resid),
            "pitchers": int(table["pitcher"].nunique()) if "pitcher" in table else 0,
        })
    return out


# ---------------------------------------------------------------------------
# The count grid, as the evidence file colours it
# ---------------------------------------------------------------------------

# A cell is "coloured" when its lift is past the legend's 1.5x or 0.67x ticks.
GRID_STRONG_LIFT = 1.5


def count_grid_rows(first: pd.DataFrame, second: pd.DataFrame) -> list[dict]:
    """Every cell of the count grid both halves can read.

    One dict per (pitch, count) where the count clears the grid's own gate in
    each half: `key` ("SL 0-2"), `share_1`/`share_2`, `lift_1`/`lift_2`, and
    the first half's `lift_z` and `clears`, which is what the page outlines.
    """
    if len(first) < MIN_HALF_PITCHES or len(second) < MIN_HALF_PITCHES:
        return []
    grids = counts.count_grid(first), counts.count_grid(second)
    if any(len(g) == 0 for g in grids):
        return []
    both = grids[0].merge(grids[1], on=["pitch_type", "count"], suffixes=("_1", "_2"))
    both = both[both["reliable_1"] & both["reliable_2"]]
    return [{"key": f"{r['pitch_type']} {r['count']}",
             "share_1": float(r["share_1"]), "share_2": float(r["share_2"]),
             "lift_1": float(r["lift_1"]), "lift_2": float(r["lift_2"]),
             "lift_z": float(r["lift_z_1"]), "clears": bool(r["clears_1"])}
            for r in both.to_dict("records")]


def count_grid_summary(rows: list[dict]) -> dict:
    """How far a grid cell moves, and what its outline is worth.

    `share_r`, `median_change`, `p90_change`: the share in each half, as the
    card's pies are read. Then, of the cells coloured past 1.5x or 0.67x in
    the first half, split by whether the page outlined them: how many came
    back on the same side of his usual rate (`same_side`) and how many past
    the same tick again (`held`).
    """
    table = pd.DataFrame(rows)
    if table.empty:
        return {}
    moved = (table["share_2"] - table["share_1"]).abs()
    lo = 1.0 / GRID_STRONG_LIFT
    strong = table[(table["lift_1"] >= GRID_STRONG_LIFT) | (table["lift_1"] <= lo)]
    up = strong["lift_1"] > 1.0
    same_side = (strong["lift_2"] > 1.0) == up
    held = (up & (strong["lift_2"] >= GRID_STRONG_LIFT)) | (~up & (strong["lift_2"] <= lo))

    def part(mask: pd.Series) -> dict:
        n = int(mask.sum())
        return {"cells": n,
                "same_side": float(same_side[mask].mean()) if n else float("nan"),
                "held": float(held[mask].mean()) if n else float("nan")}

    return {
        "cells": int(len(table)),
        "pitchers": int(table["pitcher"].nunique()) if "pitcher" in table else 0,
        "share_r": _correlation(table[["share_1", "share_2"]]),
        "median_change": float(moved.median()),
        "p90_change": float(moved.quantile(0.90)),
        "outlined": part(strong["clears"]),
        "not_outlined": part(~strong["clears"]),
    }


# ---------------------------------------------------------------------------
# The setup table, as the evidence file prints it
# ---------------------------------------------------------------------------


def setup_table_rows(first: pd.DataFrame, second: pd.DataFrame) -> list[dict]:
    """Every row of the first half's setup table, with the same row in the second.

    The table is read as the page prints it: the strongest pairs that occurred
    `report.SETUP_MIN_N` times, at most `report.SETUP_ROWS`, with no bar on how
    far a pair stands from chance. A row is compared where the same setup,
    height and follow-up clears the same gate in the second half. Each of
    SETUP_METRICS is recorded as `<metric>_1` and `<metric>_2`, with the first
    half's `lift_z` and whether it cleared three standard errors.
    """
    if len(first) < MIN_HALF_PITCHES or len(second) < MIN_HALF_PITCHES:
        return []
    printed = sequencing.setup_pairs(first, min_n=report.SETUP_MIN_N).head(report.SETUP_ROWS)
    again = sequencing.setup_pairs(second, min_n=report.SETUP_MIN_N)
    if len(printed) == 0 or len(again) == 0:
        return []
    keys = ["setup_pitch", "setup_band", "next_pitch"]
    both = printed.merge(again, on=keys, suffixes=("_1", "_2"))
    rows = []
    for r in both.to_dict("records"):
        row = {"key": f"{r['setup_pitch']} {r['setup_band']} > {r['next_pitch']}",
               "n_first": int(r["n_1"]), "n_second": int(r["n_2"]),
               "lift_z": float(r["lift_z_1"]),
               "clears": bool(r["lift_z_1"] >= _min_z())}
        for metric in SETUP_METRICS:
            for half in ("1", "2"):
                value = float(r[f"{metric}_{half}"])
                # A follow-up never swung at in one half has no whiff ratio.
                row[f"{metric}_{half}"] = value if abs(value) != float("inf") else float("nan")
        rows.append(row)
    return rows


def setup_table_summary(rows: list[dict]) -> dict:
    """Which of the setup table's columns describe the pitcher, and which the half.

    `metrics`: per column, the split-half correlation over every printed row
    re-measured in the second half. `clear_share`: how many printed rows
    stood three standard errors clear of no lift in the first half.
    `same_way_clear` and `same_way_rest`: how often a row's lift was still
    above 1 in the second half, for those rows and for the rest.
    """
    table = pd.DataFrame(rows)
    if table.empty:
        return {}
    metrics = []
    for metric, label in SETUP_METRICS.items():
        pairs = table[[f"{metric}_1", f"{metric}_2"]].rename(
            columns={f"{metric}_1": "first", f"{metric}_2": "second"})
        metrics.append({"metric": metric, "label": label,
                        "r": _correlation(pairs), "pairs": int(len(pairs.dropna()))})
    still = table["freq_lift_2"] > 1.0
    clear = table["clears"]
    return {
        "metrics": metrics,
        "rows": int(len(table)),
        "pitchers": int(table["pitcher"].nunique()) if "pitcher" in table else 0,
        "clear_share": float(clear.mean()),
        "same_way_clear": float(still[clear].mean()) if clear.any() else float("nan"),
        "same_way_rest": float(still[~clear].mean()) if (~clear).any() else float("nan"),
    }


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

    Wilson counts every finding as independent, and a pitcher can contribute
    several. `held_pitcher_lo` and `held_pitcher_hi` are the same interval
    from resampling pitchers (see `validation.pitcher_interval`), printed
    beside it so the two can be compared: where they agree the clustering
    does not matter, and where the second is much wider the findings came
    from a few pitchers who mostly agreed with themselves.
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
        by_lo, by_hi = pitcher_interval(sub, "held", min_pitchers=MIN_PAIRS)
        out.append({
            "kind": kind, "label": label, "found": n,
            "pitchers": (int(sub["pitcher"].nunique())
                         if "pitcher" in sub.columns else float("nan")),
            "held": held, "held_pct": held / n,
            "held_lo": held_lo, "held_hi": held_hi,
            "held_pitcher_lo": by_lo, "held_pitcher_hi": by_hi,
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
