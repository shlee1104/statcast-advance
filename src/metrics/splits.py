"""Situational splits: handedness, fatigue, and times through the order.

The times-through-order section is the one that repays attention. That
performance decays on later trips is well known and not worth reporting on its
own. What matters is *why*, because the two causes call for opposite advice:

  - velocity declines           -> fatigue      -> he is beatable late, extend
                                                   at-bats and force pitches
  - velocity holds, results decay -> familiarity -> his pattern is readable on
                                                   second look, sit on the
                                                   sequence he showed earlier

Both look identical in an xwOBA-by-trip table. Separating them is the whole
point of decompose_tto().

Inputs are expected to have passed through `clean.py` and
`events.add_event_flags()`.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.metrics import counts

# A plate appearance is uniquely identified by game and at-bat number.
PA_KEYS = ["game_pk", "at_bat_number"]

# Pitch groups treated as fastballs when measuring velocity decay. Mixing a
# curveball into an average velocity makes the whole series meaningless.
FASTBALL_TYPES: frozenset[str] = frozenset({"FF", "SI", "FC"})


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

    Both default to `flags.min_n.handedness_gap`. Gating here rather than in the
    rules engine keeps the exploratory table and the flagged finding in
    agreement about what counts as enough evidence.
    """
    if min_side_total is None:
        min_side_total = int(config.get("flags.min_n.handedness_gap", 40))
    if min_rate_pitches is None:
        min_rate_pitches = int(config.get("flags.min_n.handedness_gap", 40))

    columns = ["pitch_type", "usage_L", "usage_R", "gap",
               "whiff_L", "whiff_R", "n_L", "n_R"]

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

    # Blank the effectiveness cells that rest on too few pitches, leaving the
    # usage gap — which does not — intact.
    for side in ("L", "R"):
        thin = wide[f"n_{side}"].fillna(0) < min_rate_pitches
        wide.loc[thin, f"whiff_{side}"] = float("nan")

    result = wide.reset_index()[columns]
    return result.sort_values("gap", ascending=False, key=abs).reset_index(drop=True)


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
      velo          float, mean velocity
      velo_delta    float, change from the first bucket
      outings       int, how many distinct games contribute

    Restricted to fastballs by default. Averaging a curveball into a velocity
    series produces a curve that tracks pitch selection rather than fatigue,
    and selection genuinely does change late in outings.

    `outings` is the number that governs interpretation. A 2 mph drop in the
    90-105 bucket means little if only two starts ever got that far.
    """
    columns = ["bucket", "n", "velo", "velo_delta", "outings"]

    if len(frame) == 0:
        return pd.DataFrame(columns=columns)

    working = add_appearance_pitch_number(frame)
    if fastballs_only:
        working = working[working["pitch_type"].isin(FASTBALL_TYPES)]
    if len(working) == 0:
        return pd.DataFrame(columns=columns)

    working = working.copy()
    working["bucket_index"] = (working["appearance_pitch"] - 1) // bucket_size

    records = []
    for bucket_index, group in working.groupby("bucket_index"):
        low = int(bucket_index) * bucket_size + 1
        records.append({
            "bucket": f"{low}-{low + bucket_size - 1}",
            "n": len(group),
            "velo": _mean(group, "release_speed"),
            "outings": int(group["game_pk"].nunique()),
        })

    table = pd.DataFrame(records)
    baseline = table.iloc[0]["velo"]
    table["velo_delta"] = table["velo"] - baseline
    return table[columns]


def times_through_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Velocity, whiff rate, and contact quality by trip through the lineup.

    Requires the `n_thruorder_pitcher` column, which Savant provides on the raw
    feed. Returns an empty frame if it is absent, since the alternative is
    reconstructing lineup turns from batter identity, which is error-prone in
    the presence of pinch hitters.
    """
    columns = ["tto", "n", "velo", "whiff_rate", "xwoba", "batters"]

    if len(frame) == 0 or "n_thruorder_pitcher" not in frame.columns:
        return pd.DataFrame(columns=columns)

    working = frame[frame["n_thruorder_pitcher"].notna()]
    if len(working) == 0:
        return pd.DataFrame(columns=columns)

    records = []
    for tto, group in working.groupby("n_thruorder_pitcher"):
        fastballs = group[group["pitch_type"].isin(FASTBALL_TYPES)]
        swings = int(group["is_swing"].eq(True).sum())
        whiffs = int(group["is_whiff"].eq(True).sum())
        records.append({
            "tto": int(tto),
            "n": len(group),
            "velo": _mean(fastballs, "release_speed"),
            "whiff_rate": whiffs / swings if swings else float("nan"),
            "xwoba": _mean(group, "estimated_woba_using_speedangle"),
            "batters": int(group.groupby(PA_KEYS).ngroups),
        })

    return pd.DataFrame(records).sort_values("tto")[columns]


def decompose_tto(frame: pd.DataFrame) -> dict:
    """Attribute times-through-order decay to fatigue or to familiarity.

    Compares the third trip against the first and reports both changes:

      velo_delta      float, mph change in fastball velocity
      xwoba_delta     float, change in expected wOBA
      whiff_delta     float, change in whiff rate
      attribution     str, "fatigue" | "familiarity" | "both" | "none"
      note            str, one-line reading of the split

    The attribution threshold on velocity is 0.7 mph, chosen because
    measurement noise on an outing-level average is well under that, while a
    genuine fatigue effect in the literature is typically larger.

    A pitcher whose velocity holds while results decay is being *solved*, not
    tiring, and the scouting advice inverts accordingly.
    """
    blank = {
        "velo_delta": float("nan"),
        "xwoba_delta": float("nan"),
        "whiff_delta": float("nan"),
        "attribution": "unknown",
        "note": "insufficient data",
    }

    table = times_through_order(frame)
    if len(table) == 0:
        return blank

    indexed = table.set_index("tto")
    if 1 not in indexed.index or 3 not in indexed.index:
        return blank

    first, third = indexed.loc[1], indexed.loc[3]

    velo_delta = float(third["velo"] - first["velo"])
    xwoba_delta = float(third["xwoba"] - first["xwoba"])
    whiff_delta = float(third["whiff_rate"] - first["whiff_rate"])

    tiring = velo_delta <= -0.7
    worse = xwoba_delta >= 0.020

    if tiring and worse:
        attribution = "both"
        note = ("Velocity drops and results decay together — he is tiring, and "
                "the third time through is the time to attack.")
    elif tiring:
        attribution = "fatigue"
        note = ("Velocity drops without a clear results decline yet — the stuff "
                "is fading before the outcomes have caught up.")
    elif worse:
        attribution = "familiarity"
        note = ("Velocity holds while results decay — hitters are solving the "
                "pattern rather than out-lasting the arm. Sit on the sequence "
                "he showed earlier.")
    else:
        attribution = "none"
        note = "No meaningful third-time-through decline."

    return {
        "velo_delta": velo_delta,
        "xwoba_delta": xwoba_delta,
        "whiff_delta": whiff_delta,
        "attribution": attribution,
        "note": note,
    }


def _mean(group: pd.DataFrame, column: str) -> float:
    if column not in group.columns or len(group) == 0:
        return float("nan")
    return float(pd.to_numeric(group[column], errors="coerce").mean())
