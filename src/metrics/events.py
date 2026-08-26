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


def _safe_mean(frame: pd.DataFrame, column: str) -> float:
    """Mean of a boolean column, returning NaN rather than dividing by zero."""
    if len(frame) == 0:
        return float("nan")
    return float(frame[column].mean())
