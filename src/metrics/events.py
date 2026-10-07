"""Derived pitch outcomes: swings, whiffs, chases, called strikes.

Savant gives you a `description` string per pitch and a `zone` number. Almost
every rate statistic in the report is built by counting those two fields in
some combination, so the definitions live here once rather than being
re-derived (and quietly re-defined) in five different modules.

These definitions are opinionated and worth understanding, because different
public sources compute whiff rate differently and the numbers will not match
if you assume otherwise.
"""

from __future__ import annotations

import math

import pandas as pd

# Every description that involves the batter offering at the pitch.
# Bunts count: a missed bunt is a whiff, and a bunt attempt is a swing
# decision. Excluding them would inflate contact rates for pitchers who
# face a lot of bunt attempts.
SWING_DESCRIPTIONS: frozenset[str] = frozenset({
    "foul",
    "foul_bunt",
    "foul_tip",
    "bunt_foul_tip",
    "hit_into_play",
    "missed_bunt",
    "swinging_strike",
    "swinging_strike_blocked",
})

# Swings that made no contact at all.
#
# Note foul_tip is NOT a whiff. A foul tip is caught by the catcher and is
# scored a strike, but the batter did touch the ball, so counting it as a miss
# would overstate a pitcher's swing-and-miss ability.
WHIFF_DESCRIPTIONS: frozenset[str] = frozenset({
    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt",
})

# Taken pitches ruled strikes.
CALLED_STRIKE_DESCRIPTIONS: frozenset[str] = frozenset({"called_strike"})

# Savant divides the plate into zones 1-9 (inside the strike zone) and
# 11-14 (the four quadrants outside it). Zone 10 does not exist.
IN_ZONE_VALUES: frozenset[int] = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9})


def add_event_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the boolean outcome columns every metric module depends on.

    Adds:
      is_swing         batter offered at the pitch
      is_whiff         batter offered and missed entirely
      is_called_strike batter took it and it was called a strike
      is_in_zone       pitch crossed inside the strike zone
      is_chase         batter swung at a pitch outside the zone
      is_contact       batter offered and made contact (fair or foul)

    Rate statistics are then just means over these columns, optionally
    grouped. Whiff rate is `is_whiff.sum() / is_swing.sum()`, not
    `is_whiff.mean()` — the denominator is swings, not pitches. That
    distinction is the single most common way these numbers get miscomputed.

    Does not mutate the input frame.
    """
    result = frame.copy()

    description = result["description"]

    result["is_swing"] = description.isin(SWING_DESCRIPTIONS)
    result["is_whiff"] = description.isin(WHIFF_DESCRIPTIONS)
    result["is_called_strike"] = description.isin(CALLED_STRIKE_DESCRIPTIONS)
    result["is_contact"] = result["is_swing"] & ~result["is_whiff"]

    # A null zone means Statcast could not locate the pitch. Treating that as
    # "outside the zone" would silently inflate chase rate, so it stays false
    # for both in-zone and chase.
    result["is_in_zone"] = result["zone"].isin(IN_ZONE_VALUES)
    result["is_chase"] = result["is_swing"] & result["zone"].notna() & ~result["is_in_zone"]

    return result


def swing_rate(frame: pd.DataFrame) -> float:
    """Share of pitches the batter offered at."""
    return _safe_mean(frame, "is_swing")


def whiff_rate(frame: pd.DataFrame) -> float:
    """Share of SWINGS that missed. Denominator is swings, not pitches."""
    swings = frame["is_swing"].sum()
    if swings == 0:
        return float("nan")
    return float(frame["is_whiff"].sum() / swings)


def chase_rate(frame: pd.DataFrame) -> float:
    """Share of OUT-OF-ZONE pitches the batter offered at."""
    out_of_zone = (~frame["is_in_zone"] & frame["zone"].notna()).sum()
    if out_of_zone == 0:
        return float("nan")
    return float(frame["is_chase"].sum() / out_of_zone)


def zone_rate(frame: pd.DataFrame) -> float:
    """Share of pitches thrown inside the strike zone."""
    return _safe_mean(frame, "is_in_zone")


def called_strike_rate(frame: pd.DataFrame) -> float:
    """Share of TAKEN pitches called strikes."""
    takes = (~frame["is_swing"]).sum()
    if takes == 0:
        return float("nan")
    return float(frame["is_called_strike"].sum() / takes)


STRIKEOUT_EVENTS: frozenset[str] = frozenset({"strikeout", "strikeout_double_play"})
WALK_EVENTS: frozenset[str] = frozenset({"walk", "intent_walk"})


def season_line(frame: pd.DataFrame) -> dict:
    """The short stat line at the top of a scouting report.

    Returns a dict of: pitches, games, batters (plate appearances with a
    recorded outcome), k_rate, bb_rate, k_minus_bb, whiff_rate, chase_rate,
    zone_rate, first_strike_rate, xwoba.

    Deliberately small. People who build these reports for a living are
    consistent that ERA and batting average against tell a hitter nothing he
    can use in the box, and should not take up space. Strikeout and walk rate
    stay because they say how much he lives in and around the zone; whiff,
    chase and zone rate say how he gets there.

    Rates use plate appearances with an outcome as the denominator. A handful
    of plate appearances end on a play that is not a pitch — a pickoff, a
    caught stealing — and have no outcome row. The report header uses this
    same count, so the page never shows two different batter totals.
    """
    blank = {k: float("nan") for k in (
        "k_rate", "bb_rate", "k_minus_bb", "whiff_rate", "chase_rate",
        "zone_rate", "first_strike_rate", "xwoba")}
    blank.update({"pitches": len(frame), "games": 0, "batters": 0})
    if len(frame) == 0:
        return blank

    outcomes = frame["events"].dropna() if "events" in frame else pd.Series(dtype=object)
    outcomes = outcomes[outcomes.astype(str) != ""]
    batters = len(outcomes)

    k_rate = float(outcomes.isin(STRIKEOUT_EVENTS).mean()) if batters else float("nan")
    bb_rate = float(outcomes.isin(WALK_EVENTS).mean()) if batters else float("nan")

    first = frame[(frame["balls"] == 0) & (frame["strikes"] == 0)]
    first_strike = (
        float(first["type"].isin(["S", "X"]).mean()) if len(first) else float("nan")
    )

    return {
        "pitches": int(len(frame)),
        "games": int(frame["game_pk"].nunique()),
        "batters": int(batters),
        "k_rate": k_rate,
        "bb_rate": bb_rate,
        "k_minus_bb": k_rate - bb_rate,
        "whiff_rate": whiff_rate(frame),
        "chase_rate": chase_rate(frame),
        "zone_rate": zone_rate(frame),
        "first_strike_rate": first_strike,
        "xwoba": _column_mean(frame, XWOBA_COLUMN),
    }


def _safe_mean(frame: pd.DataFrame, column: str) -> float:
    """Mean of a boolean column, returning NaN rather than dividing by zero."""
    if len(frame) == 0:
        return float("nan")
    return float(frame[column].mean())


# ---------------------------------------------------------------------------
# Outcome coupling
# ---------------------------------------------------------------------------
#
# A tell and a weakness are different things, and a report that only measures
# frequency cannot tell them apart. Yamamoto's splitter is the case in point:
# hitters know it is coming with two strikes and still post a .191 xwOBA
# against it. That pattern is predictable and worthless to a hitter, and a
# report that flags it without saying so is handing a coach a plan that does
# not work.
#
# So every tell carries what happens when the pattern holds, against what
# normally happens, in runs.
#
# `delta_run_exp` is Savant's change in run expectancy on the pitch. The sign
# convention is not documented consistently in public sources, so it was
# established from the data: home runs average +1.43 and strikeouts −0.22 in
# this fixture, so POSITIVE FAVOURS THE BATTER. A positive delta therefore
# means the pattern is costing the pitcher, which is the direction a reader
# expects a "bad for him" number to point.

RUN_VALUE_COLUMN = "delta_run_exp"
XWOBA_COLUMN = "estimated_woba_using_speedangle"


def outcome_delta(subset: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    """What happens when a pattern holds, against what normally happens.

    Returns:
      rv_when     float, mean run value on the pattern's pitches
      rv_base     float, mean run value on the baseline pitches
      rv_delta    float, when minus base; positive means it costs the pitcher
      runs_cost   float, rv_delta * n, total runs over the season
      xwoba_when  float, contact quality when the pattern holds
      xwoba_base  float, contact quality normally
      xwoba_delta float, when minus base; positive is worse for the pitcher

    Run value is the primary measure and xwOBA the secondary, because xwOBA is
    only defined on batted balls while run value exists on every pitch — a tell
    whose consequence is extra called strikes is invisible to xwOBA and plain
    in run value.

    `runs_cost` is the closest thing this project has to a common currency.
    Frequency lifts, velocity drops and share deltas cannot be ranked against
    each other; runs can.
    """
    blank = {
        "rv_when": float("nan"), "rv_base": float("nan"),
        "rv_delta": float("nan"), "runs_cost": float("nan"),
        "runs_lo": float("nan"), "runs_hi": float("nan"),
        "rv_se": float("nan"), "rv_p": float("nan"), "significant": False,
        "xwoba_when": float("nan"), "xwoba_base": float("nan"),
        "xwoba_delta": float("nan"),
    }
    if len(subset) == 0 or len(baseline) == 0:
        return blank

    rv_when = _column_mean(subset, RUN_VALUE_COLUMN)
    rv_base = _column_mean(baseline, RUN_VALUE_COLUMN)
    xw_when = _column_mean(subset, XWOBA_COLUMN)
    xw_base = _column_mean(baseline, XWOBA_COLUMN)

    rv_delta = rv_when - rv_base
    n = len(subset)

    # Run value per pitch has a standard deviation around 0.19, because most
    # pitches move run expectancy barely at all and a home run moves it by
    # 1.4. The differences being measured here are 0.01 to 0.02. That ratio
    # means a season of one pitcher cannot resolve them, and the interval says
    # so: a "3.4 runs" estimate on 393 pitches carries a 95% interval of
    # roughly [-4, +11].
    #
    # This is reported rather than hidden because the alternative is a report
    # that states a run cost to one decimal place and implies a precision that
    # does not exist. A frequency pattern on 393 pitches is solid; its run
    # consequence, from the same 393 pitches, is not.
    se_delta = _welch_se(subset, baseline, RUN_VALUE_COLUMN)
    runs_cost = rv_delta * n
    margin = 1.96 * se_delta * n if se_delta == se_delta else float("nan")

    return {
        "rv_when": rv_when,
        "rv_base": rv_base,
        "rv_delta": rv_delta,
        "rv_se": se_delta,
        "runs_cost": runs_cost,
        "runs_lo": runs_cost - margin,
        "runs_hi": runs_cost + margin,
        # An UNCORRECTED p-value for this one comparison. It is deliberately
        # not a verdict: a report runs well over a hundred of these, so the
        # decision about which survive belongs to whatever can see the whole
        # family at once. flags.evaluate() applies Benjamini-Hochberg across
        # all of them.
        "rv_p": _two_sided_p(rv_delta, se_delta),
        # Uncorrected. Kept for diagnostics only — read `resolved` on a
        # finding instead.
        "significant": bool(
            margin == margin and abs(rv_delta) > 1.96 * se_delta
        ),
        "xwoba_when": xw_when,
        "xwoba_base": xw_base,
        "xwoba_delta": xw_when - xw_base,
    }


def _two_sided_p(delta: float, se: float) -> float:
    """Two-sided normal p-value for a difference in means.

    Normal rather than t because every comparison here rests on dozens to
    hundreds of pitches, where the difference between the two is far smaller
    than the thing being measured.
    """
    if not (delta == delta and se == se) or se <= 0:
        return float("nan")
    z = abs(delta / se)
    return float(math.erfc(z / math.sqrt(2)))


def _welch_se(a: pd.DataFrame, b: pd.DataFrame, column: str) -> float:
    """Standard error of the difference in means, unequal variances."""
    if column not in a.columns or column not in b.columns:
        return float("nan")
    x = pd.to_numeric(a[column], errors="coerce").dropna()
    y = pd.to_numeric(b[column], errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    return float((x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y)) ** 0.5)


def _column_mean(frame: pd.DataFrame, column: str) -> float:
    """Mean of a numeric column, tolerating absence and all-null."""
    if column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")
