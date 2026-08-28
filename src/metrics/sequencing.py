"""Pitch sequencing: what follows what.

Pitches are not thrown independently. A slider after a fastball up is a
different pitch, competitively, than the same slider thrown cold — the hitter
has been given a reason to expect something else. Usage rates alone cannot see
this, because they treat every pitch as an isolated draw from a distribution.

This module models the dependence: transition probabilities between
consecutive pitches, the same transitions conditioned on what the previous
pitch did, two-strike putaway behavior, and setup-pitch detection.

All pairing happens strictly within a plate appearance. The final pitch of one
PA never sets up the first pitch of the next.

Inputs are expected to have passed through `clean.py` and
`events.add_event_flags()`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A plate appearance is uniquely identified by game and at-bat number.
PA_KEYS = ["game_pk", "at_bat_number"]

# Descriptions counted as a foul ball for outcome conditioning.
FOUL_DESCRIPTIONS: frozenset[str] = frozenset(
    {"foul", "foul_tip", "foul_bunt", "bunt_foul_tip"}
)

# Terminal events that end a plate appearance in a strikeout.
STRIKEOUT_EVENTS: frozenset[str] = frozenset({"strikeout", "strikeout_double_play"})

# Savant divides the plate into a 3x3 grid (zones 1-9) plus four outside
# quadrants (11-14). Collapsing to vertical bands is what makes a setup pitch
# identifiable: "fastball" is not a setup, "fastball up" is.
ZONE_BANDS: dict[str, frozenset[int]] = {
    "UP": frozenset({1, 2, 3, 11, 12}),
    "MID": frozenset({4, 5, 6}),
    "DOWN": frozenset({7, 8, 9, 13, 14}),
}


# ---------------------------------------------------------------------------
# Shared pairing logic
# ---------------------------------------------------------------------------


def _pair_pitches(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach each pitch's successor within the same plate appearance.

    Returns one row per (pitch, next pitch) pair, with the successor's
    attributes prefixed `next_`. Rows whose pitch ends a plate appearance are
    dropped, since they have no successor.

    The plate-appearance grouping is what enforces the boundary: `shift(-1)`
    inside a groupby yields NaN at the end of every group, so the final pitch
    of each PA can never be paired with the first pitch of the next one.
    """
    columns = ["pitch_type", "description", "zone", "is_swing", "is_whiff"]

    ordered = frame.sort_values(PA_KEYS + ["pitch_number"]).copy()

    if len(ordered) == 0:
        for column in columns:
            ordered[f"next_{column}"] = pd.Series(dtype=object)
        return ordered

    grouped = ordered.groupby(PA_KEYS, sort=False)
    for column in columns:
        if column in ordered.columns:
            ordered[f"next_{column}"] = grouped[column].shift(-1)

    return ordered[ordered["next_pitch_type"].notna()].copy()


def _row_normalize(
    counts: pd.DataFrame,
    min_transitions: int,
) -> pd.DataFrame:
    """Turn a contingency table into row-wise conditional probabilities."""
    if counts.empty:
        return pd.DataFrame()

    counts = counts[counts.sum(axis=1) >= min_transitions]
    if counts.empty:
        return pd.DataFrame()

    return counts.div(counts.sum(axis=1), axis=0).fillna(0.0)


def _zone_band(zone: object) -> str:
    """Collapse a Savant zone number into UP / MID / DOWN."""
    if pd.isna(zone):
        return "UNKNOWN"
    value = int(zone)
    for band, members in ZONE_BANDS.items():
        if value in members:
            return band
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Transition matrices
# ---------------------------------------------------------------------------


def transition_matrix(frame: pd.DataFrame, min_transitions: int = 1) -> pd.DataFrame:
    """P(next pitch type | current pitch type), within a plate appearance.

    Returns a DataFrame where the index is the current pitch, the columns are
    the next pitch, and each ROW sums to 1. Cell (FF, SL) reads: "given he
    just threw a four-seam, how often does a slider come next".

    Pairs are formed only within the same plate appearance, in `pitch_number`
    order rather than row order. The final pitch of a PA has no successor and
    contributes nothing. Rows whose total transition count falls below
    `min_transitions` are excluded, and missing combinations are 0.0, not NaN.

    The plate-appearance boundary is the critical constraint. A starter faces
    roughly 25 batters a game, so pairing across boundaries injects one
    fabricated transition per PA — a few percent of the sample, every one of
    them a spurious "he follows X with Y" claim. The resulting matrix looks
    entirely reasonable, which is what makes the error dangerous.
    """
    pairs = _pair_pitches(frame)
    if len(pairs) == 0:
        return pd.DataFrame()

    counts = pd.crosstab(pairs["pitch_type"], pairs["next_pitch_type"])
    return _row_normalize(counts, min_transitions)


def transition_after_outcome(
    frame: pd.DataFrame,
    outcome: str,
    min_transitions: int = 1,
) -> pd.DataFrame:
    """Transition matrix conditioned on what the previous pitch did.

    Same output shape as transition_matrix(), but only counts pairs where the
    FIRST pitch of the pair had the given outcome.

    `outcome` is one of:
      "whiff"   the batter swung and missed
      "take"    the batter did not swing
      "foul"    the batter fouled it off
      "contact" the batter made contact (fair or foul)

    Derived from the boolean columns added by `events.add_event_flags()`,
    except "foul", which reads the raw `description` column.

    Conditioning captures intent, which is what makes this the more actionable
    view. "After a swinging strike on the slider, he goes back to it 61% of the
    time" is a pattern a hitter can sit on; the unconditional matrix averages
    that together with every other situation and washes it out.

    Caveat worth carrying into interpretation: the previous pitch's outcome
    also determines the resulting count, so a raw comparison against the
    unconditional matrix conflates sequencing intent with ordinary count logic.
    Comparing within a fixed count isolates the former.
    """
    selectors = {
        "whiff": lambda f: f["is_whiff"].eq(True),
        "take": lambda f: ~f["is_swing"].eq(True),
        "foul": lambda f: f["description"].isin(FOUL_DESCRIPTIONS),
        "contact": lambda f: (
            f["is_swing"].eq(True)
            & ~f["is_whiff"].eq(True)
        ),
    }

    if outcome not in selectors:
        raise ValueError(
            f"Unknown outcome {outcome!r}. Expected one of: "
            f"{', '.join(sorted(selectors))}."
        )

    pairs = _pair_pitches(frame)
    if len(pairs) == 0:
        return pd.DataFrame()

    selected = pairs[selectors[outcome](pairs)]
    if len(selected) == 0:
        return pd.DataFrame()

    counts = pd.crosstab(selected["pitch_type"], selected["next_pitch_type"])
    return _row_normalize(counts, min_transitions)


# ---------------------------------------------------------------------------
# Two-strike approach
# ---------------------------------------------------------------------------


def putaway(frame: pd.DataFrame, min_pitches: int = 10) -> pd.DataFrame:
    """Two-strike approach: what he goes to, and how often it finishes the job.

    Considers only pitches thrown with two strikes. Returns a DataFrame indexed
    by pitch_type, sorted by usage descending, with columns:

      n              int, two-strike pitches of this type
      usage          float, share of all two-strike pitches
      strikeouts     int, how many ended the PA in a strikeout
      putaway_rate   float, strikeouts / n
      whiff_rate     float, whiffs / swings on this pitch in two-strike counts

    Pitch types with fewer than `min_pitches` two-strike appearances are
    excluded; a putaway rate over four pitches is not worth printing.

    Usage and putaway_rate answer different questions and frequently disagree.
    A pitcher may go to his fastball most often with two strikes while his
    slider finishes a far higher share of the plate appearances it appears in.
    That gap is itself the exploitable finding, so both are reported.
    """
    columns = ["n", "usage", "strikeouts", "putaway_rate", "whiff_rate"]

    if len(frame) == 0:
        return pd.DataFrame(columns=columns)

    if "is_two_strike" in frame.columns:
        two_strike = frame[frame["is_two_strike"].eq(True)]
    else:
        two_strike = frame[frame["strikes"] == 2]

    if len(two_strike) == 0:
        return pd.DataFrame(columns=columns)

    total = len(two_strike)
    records = []

    for pitch_type, group in two_strike.groupby("pitch_type"):
        n = len(group)
        if n < min_pitches:
            continue

        strikeouts = int(group["events"].isin(STRIKEOUT_EVENTS).sum())
        swings = int(group["is_swing"].eq(True).sum())
        whiffs = int(group["is_whiff"].eq(True).sum())

        records.append({
            "pitch_type": pitch_type,
            "n": n,
            "usage": n / total,
            "strikeouts": strikeouts,
            "putaway_rate": strikeouts / n,
            "whiff_rate": whiffs / swings if swings else float("nan"),
        })

    if not records:
        return pd.DataFrame(columns=columns)

    table = pd.DataFrame(records).set_index("pitch_type")
    return table.sort_values("usage", ascending=False)[columns]


# ---------------------------------------------------------------------------
# Setup pitches
# ---------------------------------------------------------------------------


def setup_pairs(
    frame: pd.DataFrame,
    min_n: int = 25,
    count_state: str | None = None,
) -> pd.DataFrame:
    """Find pitch pairs where the first pitch sets up the second.

    A setup pitch is not merely the pitch that happens to precede another. It
    is one that makes the next pitch both more likely and more effective, and
    it is defined by location as well as type — "fastball" is not a setup,
    "fastball up" is.

    Two independent quantities are reported, and they can disagree:

      freq_lift    P(next | setup) / P(next).  Above 1 means the setup makes
                   that follow-up more likely than the pitcher's baseline.
                   This is the number a hitter can act on.

      effect_lift  whiff rate on the follow-up after this setup, divided by
                   its whiff rate overall. Above 1 means the sequence is
                   actually working, not merely habitual.

    High freq_lift with effect_lift near or below 1 is the most exploitable
    finding available: a predictable sequence that is not buying the pitcher
    anything.

    `count_state` optionally restricts the analysis to "ahead" (strikes >
    balls), "behind" (balls > strikes), or "even". Because the previous pitch's
    outcome determines the resulting count, an unrestricted lift conflates
    sequencing intent with ordinary count logic; holding the count fixed
    isolates the sequencing signal.

    Returns one row per (setup_pitch, setup_band, next_pitch) combination with
    at least `min_n` occurrences, sorted by `score` descending, where
    score = (freq_lift - 1) * n. Scoring by excess frequency times sample size
    keeps a dramatic lift on a thin sample from outranking a moderate one that
    can be trusted.
    """
    columns = [
        "setup_pitch", "setup_band", "next_pitch", "n",
        "p_next", "baseline_p", "freq_lift",
        "whiff_after", "whiff_baseline", "effect_lift", "score",
    ]

    pairs = _pair_pitches(frame)
    if len(pairs) == 0:
        return pd.DataFrame(columns=columns)

    if count_state is not None:
        states = {
            "ahead": pairs["strikes"] > pairs["balls"],
            "behind": pairs["balls"] > pairs["strikes"],
            "even": pairs["balls"] == pairs["strikes"],
        }
        if count_state not in states:
            raise ValueError(
                f"Unknown count_state {count_state!r}. "
                f"Expected one of: {', '.join(sorted(states))}."
            )
        pairs = pairs[states[count_state]]
        if len(pairs) == 0:
            return pd.DataFrame(columns=columns)

    pairs = pairs.copy()
    pairs["setup_band"] = pairs["zone"].map(_zone_band)

    # groupby().shift() widens boolean columns to object dtype, where .mean()
    # raises. Comparing to True restores a genuine bool column without relying
    # on pandas' deprecated silent downcasting in fillna.
    pairs["next_is_swing"] = pairs["next_is_swing"].eq(True)
    pairs["next_is_whiff"] = pairs["next_is_whiff"].eq(True)

    # Baselines are computed over the same filtered population the lifts are
    # measured against, so a count restriction narrows both sides equally.
    baseline_p = pairs["next_pitch_type"].value_counts(normalize=True)

    swings = pairs[pairs["next_is_swing"]]
    whiff_baseline = (
        swings.groupby("next_pitch_type")["next_is_whiff"].mean()
        if len(swings) else pd.Series(dtype=float)
    )

    records = []
    for (setup_pitch, band), group in pairs.groupby(["pitch_type", "setup_band"]):
        if band == "UNKNOWN":
            continue

        follow_counts = group["next_pitch_type"].value_counts()
        group_total = len(group)

        for next_pitch, n in follow_counts.items():
            if n < min_n:
                continue

            p_next = n / group_total
            base = float(baseline_p.get(next_pitch, float("nan")))
            freq_lift = p_next / base if base else float("nan")

            follow = group[group["next_pitch_type"] == next_pitch]
            follow_swings = follow[follow["next_is_swing"]]
            whiff_after = (
                float(follow_swings["next_is_whiff"].mean())
                if len(follow_swings) else float("nan")
            )
            base_whiff = float(whiff_baseline.get(next_pitch, float("nan")))
            effect_lift = (
                whiff_after / base_whiff
                if base_whiff and np.isfinite(base_whiff) and np.isfinite(whiff_after)
                else float("nan")
            )

            records.append({
                "setup_pitch": setup_pitch,
                "setup_band": band,
                "next_pitch": next_pitch,
                "n": int(n),
                "p_next": p_next,
                "baseline_p": base,
                "freq_lift": freq_lift,
                "whiff_after": whiff_after,
                "whiff_baseline": base_whiff,
                "effect_lift": effect_lift,
                "score": (freq_lift - 1) * n if np.isfinite(freq_lift) else float("nan"),
            })

    if not records:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(records)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)[columns]
    )
