"""Situational splits: handedness, fatigue, and times through the order.

The times-through-order table is descriptive on purpose. This module used to
turn it into a verdict — "third time through: hit harder", "velocity drops,
results hold" — and the verdict did not repeat. Built on half a season and
checked on the other, across 30 pitchers in 2025 and 28 in 2026, it came back
in 1 of 7 cases; the third-trip change in hitters' xwOBA correlated between
halves at 0.06 and 0.29, while the change in fastball velocity correlated at
0.86 and 0.76 (scripts/replication.py). So the table keeps the velocity
change, measured within outing, and prints the outcome columns without
drawing a conclusion from them.

Inputs are expected to have passed through `clean.py` and
`events.add_event_flags()`.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.metrics import counts, events

# A plate appearance is uniquely identified by game and at-bat number.
PA_KEYS = ["game_pk", "at_bat_number"]

# The fastball family. Velocity trends are read from ONE of these — the one he
# throws most, see velocity_pitch() — never from a blend of them.
FASTBALL_TYPES: frozenset[str] = frozenset({"FF", "SI", "FC"})


def min_outings() -> int:
    """Separate outings required before a within-outing number is trustworthy."""
    return int(config.get("flags.min_n.fatigue_outings", 10))


def mark_reliable(
    table: pd.DataFrame,
    outings_column: str = "outings",
    minimum: int | None = None,
) -> pd.DataFrame:
    """Add a boolean `reliable` column based on how many outings contributed.

    Anything measured late in an outing suffers from survivorship, not merely
    small samples: the starts that reach pitch 105, or a fourth trip through
    the order, are the starts that were going well enough to continue. A late
    bucket showing velocity *rising* is that selection, not recovery.

    This lives here rather than in the fatigue chart that first needed it,
    because the problem is a class and not an instance. The same artifact
    appears in the times-through-order table, where a fourth trip drawn from a
    handful of starts shows the best velocity of the game.

    Rows are marked, never dropped. Removing them would make a curve look
    better supported than the evidence, and a reader would have no way to know
    the tail had been trimmed.
    """
    if minimum is None:
        minimum = min_outings()

    result = table.copy()
    if len(result) == 0:
        result["reliable"] = pd.Series(dtype=bool)
        return result

    result["reliable"] = result[outings_column].fillna(0) >= minimum
    return result


def platoon(frame: pd.DataFrame, min_pitches: int | None = None) -> pd.DataFrame:
    """Pitch usage and results split by batter handedness.

    Returns a DataFrame indexed by (stand, pitch_type) with usage computed
    within each handedness, plus whiff rate, zone rate, and xwOBA.

    Usage is normalized within the split rather than across the whole sample,
    so the numbers answer "what does he throw to lefties" instead of "what
    share of his total pitches were sliders to lefties" — which is the same
    row-versus-column error that makes a count table useless.
    """
    columns = ["n", "usage", "velo", "zone_rate", "swing_rate", "whiff_rate", "xwoba"]

    if len(frame) == 0:
        return pd.DataFrame(columns=columns)

    keep = counts.primary_arsenal(frame, min_share=None, min_pitches=min_pitches)
    subset = frame[frame["pitch_type"].isin(keep)] if keep else frame.iloc[0:0]
    if len(subset) == 0:
        return pd.DataFrame(columns=columns)

    records = []
    for stand, side in subset.groupby("stand"):
        side_total = len(side)
        for pitch_type, group in side.groupby("pitch_type"):
            swings = int(group["is_swing"].eq(True).sum())
            whiffs = int(group["is_whiff"].eq(True).sum())
            records.append({
                "stand": stand,
                "pitch_type": pitch_type,
                "n": len(group),
                "usage": len(group) / side_total,
                "velo": _mean(group, "release_speed"),
                "zone_rate": float(group["is_in_zone"].eq(True).mean()),
                "swing_rate": swings / len(group),
                "whiff_rate": whiffs / swings if swings else float("nan"),
                "xwoba": _mean(group, "estimated_woba_using_speedangle"),
            })

    table = pd.DataFrame(records).set_index(["stand", "pitch_type"])
    return table.sort_values(["stand", "usage"], ascending=[True, False])[columns]


def platoon_gaps(
    frame: pd.DataFrame,
    min_pitches: int | None = None,
    min_side_total: int | None = None,
    min_rate_pitches: int | None = None,
) -> pd.DataFrame:
    """Where a pitcher's approach differs most between lefties and righties.

    Returns one row per pitch type with usage against each side and the gap
    between them, sorted by absolute gap.

    A large gap on an effective pitch is the finding: a weapon one side of the
    plate rarely sees is a weapon that side has not learned to handle, and it
    also means the other side is facing a narrower arsenal than the raw
    repertoire suggests.

    Two different quantities in this table carry two different sample sizes,
    and conflating them is easy:

      The usage gap is a rate over every pitch thrown to that side. A pitcher
      who threw 9 sinkers to left-handers across 1,600 pitches has a usage of
      0.6% that is measured very precisely — "he does not show lefties this
      pitch" is exactly the finding, and gating the row on that 9 would delete
      it. What must be adequate is the DENOMINATOR: pitches thrown to each
      side, governed by `min_side_total`.

      The whiff rates are rates over swings at that pitch type on that side, so
      the same 9 sinkers give a whiff rate that is nearly meaningless. Those
      cells are blanked below `min_rate_pitches` rather than the row being
      dropped, so the trustworthy half of the row survives.

    `n_L` / `n_R` are pitches of that type thrown to each side; `faced_L` /
    `faced_R` are all pitches thrown to each side, the denominators of the two
    usage rates.

    Both default to `flags.min_n.handedness_gap`. Gating here rather than in the
    rules engine keeps the exploratory table and the flagged finding in
    agreement about what counts as enough evidence.
    """
    if min_side_total is None:
        min_side_total = int(config.get("flags.min_n.handedness_gap", 40))
    if min_rate_pitches is None:
        min_rate_pitches = int(config.get("flags.min_n.handedness_gap", 40))

    columns = ["pitch_type", "usage_L", "usage_R", "gap",
               "whiff_L", "whiff_R", "n_L", "n_R", "faced_L", "faced_R"]

    split = platoon(frame, min_pitches)
    if len(split) == 0:
        return pd.DataFrame(columns=columns)

    # A side the pitcher barely faced cannot support a usage rate at all, so
    # the comparison is refused rather than reported against a stub.
    faced = split.groupby("stand")["n"].sum()
    if any(faced.get(side, 0) < min_side_total for side in ("L", "R")):
        return pd.DataFrame(columns=columns)

    wide = split.reset_index().pivot(
        index="pitch_type", columns="stand",
        values=["usage", "whiff_rate", "n"],
    )
    wide.columns = [f"{metric}_{side}" for metric, side in wide.columns]

    for column in ("usage_L", "usage_R", "whiff_rate_L", "whiff_rate_R", "n_L", "n_R"):
        if column not in wide.columns:
            wide[column] = float("nan")

    wide = wide.rename(columns={"whiff_rate_L": "whiff_L", "whiff_rate_R": "whiff_R"})
    wide["gap"] = wide["usage_L"].fillna(0) - wide["usage_R"].fillna(0)

    # The denominators travel with the row, so anything quoting a usage rate
    # can quote what it is a share of.
    wide["faced_L"] = int(faced.get("L", 0))
    wide["faced_R"] = int(faced.get("R", 0))

    # Blank the effectiveness cells that rest on too few pitches, leaving the
    # usage gap — which does not — intact.
    for side in ("L", "R"):
        thin = wide[f"n_{side}"].fillna(0) < min_rate_pitches
        wide.loc[thin, f"whiff_{side}"] = float("nan")

    result = wide.reset_index()[columns]
    return result.sort_values("gap", ascending=False, key=abs).reset_index(drop=True)


def velocity_pitch(frame: pd.DataFrame) -> str | None:
    """The one pitch a within-outing velocity trend is read from.

    His most-used pitch in the fastball family: four-seam, sinker or cutter.
    One pitch, not the family averaged together, because the three sit at
    different speeds and he does not throw them in the same proportion all
    game. Yamamoto's cutter is 91 mph against a 95.5 mph four-seam, and it
    grows from 18% of his fastballs in the first 15 pitches to 24% by pitch
    90. Averaged together that reads as a 1.0 mph "velocity drop", of which a
    quarter is him choosing the slower pitch more often: the four-seam alone
    is down 0.8, and 0.65 once each outing is compared with itself.

    A cutter counts here, unlike in primary_fastball(), so a cutter-first
    pitcher still gets a velocity trend — from the pitch he actually leans on.
    """
    if "pitch_type" not in frame.columns:
        return None
    hard = frame[frame["pitch_type"].isin(FASTBALL_TYPES)]["pitch_type"]
    if len(hard) == 0:
        return None
    return str(hard.value_counts().index[0])


def paired_change(
    frame: pd.DataFrame,
    group: str,
    start,
    end,
    value: str = "release_speed",
) -> tuple[float, float, int]:
    """Mean within-outing change in `value` from `group` == start to end.

    Returns (mean change, its standard error, outings compared).

    Each outing is compared with ITSELF: the change is taken inside every game
    that has both ends, then averaged across games. Comparing two pooled
    averages instead — every late pitch against every early pitch — mixes in
    which games got that deep, and the starts that reach pitch 100 or a third
    trip are the ones where he had his best stuff. That selection is why a
    late bucket could show velocity RISING; pairing removes it rather than
    just flagging the bucket as thin.

    The standard error is across outings, which are the independent units
    here. Pitches within one outing share that day's arm.
    """
    blank = (float("nan"), float("nan"), 0)
    if len(frame) == 0 or value not in frame.columns:
        return blank
    working = frame[frame[group].isin([start, end])]
    values = pd.to_numeric(working[value], errors="coerce")
    per_game = (
        values.groupby([working["game_pk"], working[group]]).mean().unstack()
    )
    if start not in per_game.columns or end not in per_game.columns:
        return blank
    change = (per_game[end] - per_game[start]).dropna()
    if len(change) == 0:
        return blank
    se = float(change.std(ddof=1) / len(change) ** 0.5) if len(change) > 1 else float("nan")
    return float(change.mean()), se, int(len(change))


def add_appearance_pitch_number(frame: pd.DataFrame) -> pd.DataFrame:
    """Number each pitch by its position within that day's outing.

    Savant does not ship a running pitch count, so it is derived: within a
    game, order by plate appearance and pitch number, then count. This is what
    makes a fatigue curve possible.
    """
    result = frame.sort_values(["game_pk"] + PA_KEYS[1:] + ["pitch_number"]).copy()
    result["appearance_pitch"] = result.groupby("game_pk").cumcount() + 1
    return result


def fatigue(
    frame: pd.DataFrame,
    bucket_size: int = 15,
    fastballs_only: bool = True,
) -> pd.DataFrame:
    """Fastball velocity as a function of pitch count within the outing.

    Returns one row per bucket of `bucket_size` pitches:

      bucket        str, the pitch-count range
      n             int, pitches in the bucket
      velo          float, mean velocity of every pitch in the bucket
      velo_delta    float, mean within-outing change from the first bucket
      outings       int, how many distinct games contribute to the bucket
      pairs         int, how many of those also have a first-bucket reading,
                    which is what `velo_delta` is averaged over
      delta_se      float, standard error of `velo_delta` across outings

    Restricted by default to his most-used fastball (see velocity_pitch). A
    curveball in a velocity series makes it track pitch selection rather than
    fatigue, and so does a cutter mixed in with a four-seam.

    `velo_delta` is paired: each outing's bucket is compared with that same
    outing's first bucket, then averaged (see paired_change). `velo` is the
    plain average of the bucket, for drawing the curve, so `velo` minus the
    first row's `velo` will not equal `velo_delta` exactly — the difference is
    the selection effect the pairing removes.

    `pairs` is the number that governs interpretation. A 2 mph drop in the
    90-105 bucket means little if only two starts ever got that far.
    """
    columns = ["bucket", "n", "velo", "velo_delta", "outings", "pairs", "delta_se"]

    if len(frame) == 0:
        return pd.DataFrame(columns=columns)

    working = add_appearance_pitch_number(frame)
    if fastballs_only:
        working = working[working["pitch_type"] == velocity_pitch(frame)]
    if len(working) == 0:
        return pd.DataFrame(columns=columns)

    working = working.copy()
    working["bucket_index"] = (working["appearance_pitch"] - 1) // bucket_size
    first_bucket = working["bucket_index"].min()

    records = []
    for bucket_index, group in working.groupby("bucket_index"):
        low = int(bucket_index) * bucket_size + 1
        if bucket_index == first_bucket:
            delta, se, pairs = 0.0, 0.0, int(group["game_pk"].nunique())
        else:
            delta, se, pairs = paired_change(
                working, "bucket_index", first_bucket, bucket_index
            )
        records.append({
            "bucket": f"{low}-{low + bucket_size - 1}",
            "n": len(group),
            "velo": _mean(group, "release_speed"),
            "velo_delta": delta,
            "outings": int(group["game_pk"].nunique()),
            "pairs": pairs,
            "delta_se": se,
        })

    return pd.DataFrame(records)[columns]


def times_through_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Velocity, whiff rate, and xwOBA by trip through the lineup.

    Requires the `n_thruorder_pitcher` column, which Savant provides on the raw
    feed. Returns an empty frame if it is absent, since the alternative is
    reconstructing lineup turns from batter identity, which is error-prone in
    the presence of pinch hitters.

    `velo` is his most-used fastball only (see velocity_pitch), and `xwoba` is
    per plate appearance ending in that trip (see events.season_line). Both
    are plain per-trip averages, and a later trip holds only the outings that
    lasted that long.

    `velo_delta` is the one column that corrects for that: the change in that
    fastball from the first trip, taken inside each outing that reached both
    trips and then averaged (see paired_change), with `velo_pairs` the number
    of outings it rests on. It is blank for the first trip. Read the velocity
    trend from this column, not from the difference between two `velo` cells.

    No column here says why anything changes. The third trip is also roughly
    the 70th to 100th pitch, so trip number and pitch count move together and
    a by-trip table cannot separate fatigue from familiarity.
    """
    columns = ["tto", "n", "velo", "velo_delta", "velo_pairs", "whiff_rate",
               "xwoba", "batters", "outings", "reliable"]

    if len(frame) == 0 or "n_thruorder_pitcher" not in frame.columns:
        return pd.DataFrame(columns=columns)

    working = frame[frame["n_thruorder_pitcher"].notna()]
    if len(working) == 0:
        return pd.DataFrame(columns=columns)

    fastball = velocity_pitch(working)
    all_fastballs = working[working["pitch_type"] == fastball]

    records = []
    for tto, group in working.groupby("n_thruorder_pitcher"):
        fastballs = group[group["pitch_type"] == fastball]
        swings = int(group["is_swing"].eq(True).sum())
        whiffs = int(group["is_whiff"].eq(True).sum())
        if int(tto) == 1:
            velo_delta, velo_pairs = float("nan"), 0
        else:
            velo_delta, _, velo_pairs = paired_change(
                all_fastballs, "n_thruorder_pitcher", 1, tto
            )
        records.append({
            "tto": int(tto),
            "n": len(group),
            "velo": _mean(fastballs, "release_speed"),
            "velo_delta": velo_delta,
            "velo_pairs": velo_pairs,
            "whiff_rate": whiffs / swings if swings else float("nan"),
            "xwoba": _mean(group, "estimated_woba_using_speedangle"),
            "batters": int(group.groupby(PA_KEYS).ngroups),
            # A fourth trip only happens in the starts that were going well
            # enough to get there, which is why its velocity can read higher
            # than the first. Same survivorship as a late fatigue bucket, so
            # the same gate applies.
            "outings": int(group["game_pk"].nunique()),
        })

    table = pd.DataFrame(records).sort_values("tto")
    return mark_reliable(table)[columns]


def primary_fastball(frame: pd.DataFrame) -> str | None:
    """His main fastball: the more-used of four-seam and sinker.

    Velocity trends are read off one pitch, not a blend. Folding a cutter into
    a fastball average makes the average move whenever his cutter usage moves,
    which reads as a velocity change and is actually a pitch-mix change.
    """
    candidates = frame[frame["pitch_type"].isin(["FF", "SI"])]["pitch_type"]
    if len(candidates) == 0:
        return None
    return str(candidates.value_counts().index[0])


# Kept under its old name for the callers that already use it.
_two_proportion_z = events.two_proportion_z


def recent_form(
    frame: pd.DataFrame,
    last_n: int = 5,
    min_delta: float = 0.05,
    min_z: float = 2.5,
) -> dict:
    """What he has been doing in his last few starts, against before that.

    Returns a dict:
      last_n        int, starts in the recent window
      dates         list[str], their dates, most recent first
      starts        DataFrame, one row per recent start: date, pitches,
                    fb_velo, whiff_rate, strikeouts, walks
      shifts        DataFrame, per pitch: recent_share, earlier_share, delta,
                    z, notable
      fastball      str, the pitch the velocity numbers are read from
      velo_recent, velo_earlier, velo_delta   float, mph
      n_recent, n_earlier                     int, pitches in each window

    A season average is the wrong default for an advance report, which is
    written for the next game. Scouts watch the last few weeks because hitters
    and pitchers both drift — a pitcher who added a cutter in August looks, in
    a full-season table, like a pitcher who throws a few cutters.

    The comparison is recent against EARLIER, not against the full season,
    because the full season contains the recent starts and would mute any
    change by the share of the season they make up.

    A shift is marked `notable` only if it moves at least `min_delta` and
    clears |z| >= `min_z`. The bar is higher than the usual 2 because every
    pitch in the arsenal is tested at once, and five or six comparisons at 2
    will produce a false "change" regularly.
    """
    blank = {
        "last_n": 0, "dates": [], "starts": pd.DataFrame(),
        "shifts": pd.DataFrame(), "fastball": None,
        "velo_recent": float("nan"), "velo_earlier": float("nan"),
        "velo_delta": float("nan"), "n_recent": 0, "n_earlier": 0,
        "earlier_label": "the rest of the sample",
    }
    if len(frame) == 0 or "game_date" not in frame.columns:
        return blank

    working = counts.restrict_to_arsenal(frame).copy()
    if len(working) == 0:
        return blank

    working["game_date"] = pd.to_datetime(working["game_date"])
    games = (
        working.groupby("game_pk")["game_date"].min()
        .sort_values(ascending=False)
    )
    recent_games = list(games.index[:last_n])
    if not recent_games:
        return blank

    recent = working[working["game_pk"].isin(recent_games)]
    earlier = working[~working["game_pk"].isin(recent_games)]
    # A pitcher with no four-seam or sinker is read off his cutter. Without
    # this the table printed no velocity at all for a cutter-and-slider
    # closer, under a heading that read "None velo".
    fastball = primary_fastball(working) or velocity_pitch(working)

    # With seasons combined, "lately" is measured against the rest of the
    # current season, not against both. Otherwise it repeats what the "Since
    # last season" line already says — Snell's report gave "changeup down to
    # 17%" twice, once per line, for what was one season-to-season change.
    earlier_label = "the rest of the sample"
    if "game_year" in working.columns and working["game_year"].nunique() > 1:
        latest = working["game_year"].max()
        this_season = earlier[earlier["game_year"] == latest]
        if len(this_season):
            earlier = this_season
            earlier_label = f"the rest of {int(latest)}"

    start_rows = []
    for game in recent_games:
        g = recent[recent["game_pk"] == game]
        swings = int(g["is_swing"].eq(True).sum())
        whiffs = int(g["is_whiff"].eq(True).sum())
        pa_end = g["events"].dropna() if "events" in g else pd.Series(dtype=object)
        start_rows.append({
            "date": games[game].strftime("%Y-%m-%d"),
            "pitches": len(g),
            "fb_velo": _mean(g[g["pitch_type"] == fastball], "release_speed"),
            "whiff_rate": whiffs / swings if swings else float("nan"),
            "strikeouts": int(pa_end.isin(["strikeout", "strikeout_double_play"]).sum()),
            "walks": int(pa_end.isin(["walk", "intent_walk"]).sum()),
        })

    shift_rows = []
    n_recent, n_earlier = len(recent), len(earlier)
    for pitch in working["pitch_type"].value_counts().index:
        x1 = int((recent["pitch_type"] == pitch).sum())
        x2 = int((earlier["pitch_type"] == pitch).sum())
        r_share = x1 / n_recent if n_recent else float("nan")
        e_share = x2 / n_earlier if n_earlier else float("nan")
        delta = r_share - e_share
        z = _two_proportion_z(x1, n_recent, x2, n_earlier)
        shift_rows.append({
            "pitch_type": pitch,
            "recent_share": r_share,
            "earlier_share": e_share,
            "delta": delta,
            "z": z,
            "notable": bool(
                z == z and abs(delta) >= min_delta and abs(z) >= min_z
            ),
        })

    velo_recent = _mean(recent[recent["pitch_type"] == fastball], "release_speed")
    velo_earlier = _mean(earlier[earlier["pitch_type"] == fastball], "release_speed")
    # The difference of the two numbers as printed, so the page never shows
    # "96.3 against 95.5 (+0.7)".
    velo_delta = round(round(velo_recent, 1) - round(velo_earlier, 1), 1)

    return {
        "last_n": len(recent_games),
        "dates": [games[g].strftime("%Y-%m-%d") for g in recent_games],
        "starts": pd.DataFrame(start_rows),
        "shifts": pd.DataFrame(shift_rows),
        "fastball": fastball,
        "velo_recent": velo_recent,
        "velo_earlier": velo_earlier,
        "velo_delta": velo_delta,
        "n_recent": n_recent,
        "n_earlier": n_earlier,
        "earlier_label": earlier_label,
    }


def _mean(group: pd.DataFrame, column: str) -> float:
    if column not in group.columns or len(group) == 0:
        return float("nan")
    return float(pd.to_numeric(group[column], errors="coerce").mean())
