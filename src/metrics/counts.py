"""Count-state tendencies.

The count is the single strongest predictor of what a pitcher will throw. A
hitter who knows the mix in 1-2 versus 3-1 holds more usable information than
one who knows the season-long arsenal, because a season-long mix averages over
situations that demand entirely different pitches.

This module quantifies that: usage by count, how a pitcher opens a plate
appearance, and a normalized-entropy score for how predictable each count is.

Inputs are expected to have passed through `clean.py` and
`events.add_event_flags()`, so `count`, `is_two_strike`, `is_swing`, and
`is_whiff` are already present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The twelve legal counts, in the order a scouting report should display them.
ALL_COUNTS: list[str] = [
    "0-0", "0-1", "0-2",
    "1-0", "1-1", "1-2",
    "2-0", "2-1", "2-2",
    "3-0", "3-1", "3-2",
]


def pitch_mix(frame: pd.DataFrame) -> pd.Series:
    
    return frame["pitch_type"].value_counts(normalize=True)

def mix_by_count(frame: pd.DataFrame, min_pitches: int = 1) -> pd.DataFrame:
    """Pitch-type usage broken out by count.

    Returns a DataFrame with counts as the index and pitch types as columns,
    where each ROW sums to 1. Cell (count, pitch) is "of all pitches thrown in
    this count, what share were this pitch type."

    Rows appear in ALL_COUNTS order and only for counts present in the data.
    Counts with fewer than `min_pitches` total are excluded rather than shown
    with an unstable percentage. Missing combinations are 0.0, not NaN.

    Row-normalized rather than column-normalized because the scouting question
    is "given it is 1-2, what is he throwing?", not "of all his sliders, when
    does he throw them?". The transposed version produces a table that looks
    reasonable and answers a question nobody asked.
    """

    per_count = frame["count"].value_counts()
    keep = per_count[per_count >= min_pitches].index
    subset = frame[frame["count"].isin(keep)]

    table = pd.crosstab(subset["count"], subset["pitch_type"], normalize="index")

    order = [c for c in ALL_COUNTS if c in table.index]
    table = table.reindex(order)

    return table.fillna(0.0)


def first_pitch_tendencies(frame: pd.DataFrame) -> dict:
    """Summarize how the pitcher opens a plate appearance.

    Return a dict with:
      n                  int, pitches thrown in 0-0 counts
      mix                Series, pitch-type shares in 0-0
      strike_rate        float, share of 0-0 pitches that were strikes
      primary_pitch      str, most-used 0-0 pitch (None if no data)
      primary_share      float, that pitch's share (nan if no data)

    A "strike" here means the pitch was either called a strike, swung at, or
    put in play — i.e. anything that is not a ball or hit-by-pitch. Use the
    `type` column, which Savant sets to "S", "B", or "X". Count "S" and "X"
    as strikes.

    First-pitch strike rate is one of the few numbers every pitching coach
    already knows by heart, so it is worth getting exactly right.
    """
    first = frame[frame["is_first_pitch"]]
    if len(first) == 0:
        return {
            "n": 0,
            "mix": pd.Series(dtype=float),
            "strike_rate": float("nan"),
            "primary_pitch": None,
            "primary_share": float("nan"),
            }

    mix = pitch_mix(first)
    strike_rate = first["type"].isin(["S", "X"]).mean()

    return {
        "n": len(first),
        "mix": mix,
        "strike_rate": float(strike_rate),
        "primary_pitch": mix.index[0],
        "primary_share": float(mix.iloc[0]),
    }


def predictability(frame: pd.DataFrame, min_pitches: int = 20) -> pd.DataFrame:
    """Score how predictable the pitcher is in each count.

    Measures the normalized Shannon entropy of the pitch-mix distribution in
    each count, inverted so that higher means more predictable.

    For a distribution p over k pitch types:

        H = -sum(p_i * log2(p_i))          for p_i > 0
        predictability = 1 - H / log2(k)

    H is maximized at log2(k) when every pitch is equally likely, so the
    normalized score falls in [0, 1]. A score of 1.0 means exactly one pitch is
    thrown in that count; 0.0 means the mix is perfectly uniform.

    k is the size of the pitcher's overall arsenal, not the number of pitch
    types observed in the individual count. A count containing only fastballs
    would otherwise give k=1 and log2(1)=0. The full arsenal is also the more
    meaningful reference: the question is how much of the repertoire is live in
    a given count, not how varied that particular handful of pitches was.

    Returns a DataFrame indexed by count in ALL_COUNTS order, with columns:
      n               int, pitches in that count
      entropy         float, Shannon entropy in bits
      predictability  float in [0, 1]
      top_pitch       str, most-used pitch in that count
      top_share       float, that pitch's share

    Counts with fewer than `min_pitches` are excluded. Entropy computed over a
    handful of pitches is noise wearing a number.

    A single-pitch arsenal is handled separately, since log2(1) = 0 would make
    the normalization undefined; such a pitcher scores 1.0 in every count by
    definition.
    """
    columns = ["n", "entropy", "predictability", "top_pitch", "top_share"]

    if len(frame) == 0:
        return pd.DataFrame(columns=columns)

    k = frame["pitch_type"].nunique()
    max_entropy = np.log2(k) if k > 1 else None

    per_count = frame["count"].value_counts()
    mix = mix_by_count(frame, min_pitches=min_pitches)

    rows = []
    for count_label, shares in mix.iterrows():
        nonzero = shares[shares > 0]
        entropy = -(nonzero * np.log2(nonzero)).sum()
        pred = 1.0 if max_entropy is None else 1 - entropy / max_entropy


        rows.append({
        "n": int(per_count[count_label]),
        "entropy": float(entropy),
        "predictability": float(pred),
        "top_pitch": shares.idxmax(),
        "top_share": float(shares.max()),
        })

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows, index=mix.index)[columns]


