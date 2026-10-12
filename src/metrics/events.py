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

# Values of `events` that are not a completed plate appearance. Savant writes
# one on the last pitch before a runner is thrown out to end an inning
# ("truncated_pa": the batter comes up again next inning), and on a few plays
# that happen between pitches. Counting them as batters faced puts a plate
# appearance in the K% and BB% denominators that nobody finished — one or two a
# season, so the rates move in the fourth decimal, but "batters faced" is a
# number a reader will check against Savant's.
NON_PA_EVENTS: frozenset[str] = frozenset({
    "truncated_pa",
    "caught_stealing_2b", "caught_stealing_3b", "caught_stealing_home",
    "pickoff_1b", "pickoff_2b", "pickoff_3b",
    "pickoff_caught_stealing_2b", "pickoff_caught_stealing_3b",
    "pickoff_caught_stealing_home",
    "stolen_base_2b", "stolen_base_3b", "stolen_base_home",
    "wild_pitch", "passed_ball", "balk", "other_out", "other_advance",
    "runner_double_play", "game_advisory", "ejection",
})


def plate_appearance_outcomes(frame: pd.DataFrame) -> pd.Series:
    """The `events` value of every completed plate appearance in the frame."""
    if "events" not in frame:
        return pd.Series(dtype=object)
    outcomes = frame["events"].dropna()
    outcomes = outcomes[outcomes.astype(str) != ""]
    return outcomes[~outcomes.isin(NON_PA_EVENTS)]


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

    Rates use completed plate appearances as the denominator. A handful end on
    a play that is not the batter's — a runner thrown out for the third out —
    and Savant marks those rows `truncated_pa`; they are left out (see
    NON_PA_EVENTS). The report header uses this same count, so the page never
    shows two different batter totals.

    `xwoba` is the mean of Savant's `estimated_woba_using_speedangle`, which
    in current feeds is filled on every plate-appearance-ending pitch:
    batted balls get the expected value from exit speed and launch angle,
    strikeouts get 0 and walks their wOBA weight. So this is xwOBA per plate
    appearance, the number Savant's leaderboards show — not contact quality
    alone, and it should not be compared with an xwOBA-on-contact figure.
    """
    blank = {k: float("nan") for k in (
        "k_rate", "bb_rate", "k_minus_bb", "whiff_rate", "chase_rate",
        "zone_rate", "first_strike_rate", "xwoba")}
    blank.update({"pitches": len(frame), "games": 0, "batters": 0})
    if len(frame) == 0:
        return blank

    outcomes = plate_appearance_outcomes(frame)
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
# hitters know it is coming with two strikes and still post a .163 xwOBA
# against it there, below the .203 on his other two-strike pitches. That
# pattern is predictable and worthless to a hitter, and a report that flags it
# without saying so is handing a coach a plan that does not work.
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
    only defined on the pitch that ends a plate appearance while run value
    exists on every pitch — a tell whose consequence is extra called strikes
    mid-count is invisible to xwOBA and plain in run value.

    `subset` is normally a slice of `baseline` — his splitters in 1-2 against
    all of his splitters — and the standard error accounts for that overlap
    (see `_delta_se`).

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
    # so: an estimate of a few runs on a few hundred pitches carries a 95%
    # interval several times its own size.
    #
    # This is reported rather than hidden because the alternative is a report
    # that states a run cost to one decimal place and implies a precision that
    # does not exist. A frequency pattern on 393 pitches is solid; its run
    # consequence, from the same 393 pitches, is not.
    se_delta = _delta_se(subset, baseline, RUN_VALUE_COLUMN)
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


def two_proportion_z(x1: int, n1: int, x2: int, n2: int) -> float:
    """z for the difference between two observed proportions, pooled SE.

    x1 of n1 against x2 of n2, where the two groups do not overlap. NaN when
    either group is empty or the pooled rate is 0 or 1, where there is nothing
    to test.
    """
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    pooled = (x1 + x2) / (n1 + n2)
    if pooled <= 0.0 or pooled >= 1.0:
        return float("nan")
    se = (pooled * (1 - pooled) * (1 / n1 + 1 / n2)) ** 0.5
    return float((x1 / n1 - x2 / n2) / se)


def lift_z(n: int, group_n: int, pitch_total: int, total: int) -> float:
    """How many standard errors a usage lift stands clear of no lift at all.

    `n` of the `group_n` pitches in some situation (a count, a location band,
    the pitches after a setup) were a given pitch type, against `pitch_total`
    of all `total` pitches. The test is the situation against everything
    OUTSIDE it — the situation's own pitches are part of his overall rate, so
    testing against the overall rate would be testing a number partly against
    itself.

    This exists because a minimum count is not a noise filter. "At least 20
    of that pitch, at 1.5 times his usual rate" admits 20 where 13.3 were
    expected, which is about 1.8 standard errors — and a report looks at forty
    such cells per pitcher. The count gate keeps out rates built on a handful
    of pitches; this keeps out lifts that chance produces by itself.
    """
    return two_proportion_z(n, group_n, pitch_total - n, total - group_n)


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


def _delta_se(subset: pd.DataFrame, baseline: pd.DataFrame, column: str) -> float:
    """Standard error of mean(subset) − mean(baseline).

    Every caller compares a slice with the whole it was cut from: his
    splitters in 1-2 against all of his splitters. Those are not independent
    samples — the slice is inside the baseline — and the two-sample (Welch)
    formula, which assumes they are, gets the error wrong in the conservative
    direction. Writing f for the slice's share of the baseline,

        mean(slice) − mean(all) = (1 − f) · (mean(slice) − mean(rest))

    and the slice and the rest ARE independent, so

        SE = (1 − f) · sqrt(s²_slice / n_slice + s²_rest / n_rest)

    Welch on (slice, all) instead gives roughly sqrt((1 + f) / (1 − f)) times
    that: 9% too wide when the slice is a tenth of the baseline, 36% too wide
    at three tenths. Under a permutation null on Yamamoto's four-seam the
    nominal 5% test rejected 4.3%, 0.9% and 0.1% of the time at f = 0.1, 0.3
    and 0.5; this form rejects 6.4%, 5.3% and 5.2%. (The small excess at
    f = 0.1 is the normal approximation meeting a skewed run-value
    distribution on ~110 pitches, not the overlap.)

    Falls back to Welch when `subset` is not contained in `baseline`, which is
    the right formula for two separate samples.
    """
    if column not in subset.columns or column not in baseline.columns:
        return float("nan")
    x = pd.to_numeric(subset[column], errors="coerce").dropna()
    everything = pd.to_numeric(baseline[column], errors="coerce").dropna()

    nested = (
        baseline.index.is_unique
        and subset.index.is_unique
        and bool(subset.index.isin(baseline.index).all())
    )
    if not nested:
        return _welch_se(x, everything)

    rest = everything[~everything.index.isin(subset.index)]
    if len(rest) == 0:
        # The slice is the whole baseline: the difference is zero by
        # construction and there is nothing to test.
        return float("nan")
    share = len(x) / (len(x) + len(rest))
    return (1.0 - share) * _welch_se(x, rest)


def _welch_se(x: pd.Series, y: pd.Series) -> float:
    """Standard error of the difference in means of two INDEPENDENT samples."""
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    return float((x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y)) ** 0.5)


def _column_mean(frame: pd.DataFrame, column: str) -> float:
    """Mean of a numeric column, tolerating absence and all-null."""
    if column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")
