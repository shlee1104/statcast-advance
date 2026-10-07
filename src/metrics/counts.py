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

from src import config
from src.metrics import events

# The twelve legal counts, in the order a scouting report should display them.
ALL_COUNTS: list[str] = [
    "0-0", "0-1", "0-2",
    "1-0", "1-1", "1-2",
    "2-0", "2-1", "2-2",
    "3-0", "3-1", "3-2",
]


def primary_arsenal(
    frame: pd.DataFrame,
    min_share: float | None = None,
    min_pitches: int | None = None,
) -> list[str]:
    """Pitch types the pitcher throws often enough to count as real offerings.

    Statcast misclassifies a small number of pitches every season. A pitcher
    credited with two sweepers across 3,000 pitches does not have a sweeper,
    and treating that stray label as a genuine offering distorts anything
    computed over arsenal size — most importantly the entropy ceiling in
    predictability(), where one phantom pitch type raises log2(k) and depresses
    every score in the table.

    The primary gate is a usage SHARE rather than a raw count, because arsenal
    membership is a rate question. An absolute threshold that is sensible
    against a starter's 3,000 pitches would discard genuine offerings from a
    reliever's 900. `min_pitches` is a secondary floor, guarding only against
    samples small enough that a 1% share can be a single pitch.

    Defaults come from `report.min_arsenal_share` and
    `report.min_arsenal_pitches` in config.yaml.
    """
    if min_share is None:
        min_share = float(config.get("report.min_arsenal_share", 0.01))
    if min_pitches is None:
        min_pitches = int(config.get("report.min_arsenal_pitches", 5))

    if len(frame) == 0:
        return []

    tallies = frame["pitch_type"].value_counts()
    shares = tallies / tallies.sum()
    qualifying = tallies[(shares >= min_share) & (tallies >= min_pitches)]
    return list(qualifying.index)


def pitch_mix(frame: pd.DataFrame) -> pd.Series:
    """Overall pitch-type usage as proportions summing to 1.

    Index is pitch_type, values are shares, sorted descending so the pitcher's
    primary offering comes first. Returns an empty Series for an empty frame.
    """
    return frame["pitch_type"].value_counts(normalize=True)


def restrict_to_arsenal(
    frame: pd.DataFrame,
    min_share: float | None = None,
    min_pitches: int | None = None,
) -> pd.DataFrame:
    """Drop pitches outside the pitcher's real arsenal.

    Every section of a report must answer to the same definition of what this
    pitcher throws. Yamamoto's 2025 season contains exactly one pitch labelled
    a sweeper: `primary_arsenal` correctly excluded it from the arsenal table
    and the predictability scores, while `mix_by_count` — which applied no gate
    at all — carried it into the count chart as a seventh column. One pitcher,
    two answers to "how many pitches does he throw", depending on which part of
    the page you read. That is worse than including or excluding it, because
    neither number is wrong on its own and the disagreement is invisible.
    """
    keep = primary_arsenal(frame, min_share=min_share, min_pitches=min_pitches)
    if not keep:
        return frame.iloc[0:0]
    return frame[frame["pitch_type"].isin(keep)]


def mix_by_count(
    frame: pd.DataFrame,
    min_pitches: int = 1,
    arsenal_only: bool = True,
) -> pd.DataFrame:
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

    `arsenal_only` applies the same share gate the rest of the module uses, so
    a single misclassified pitch cannot appear here as a column while being
    absent from every other section. Pass False only to inspect raw labels.
    """
    if arsenal_only:
        frame = restrict_to_arsenal(frame)
        if len(frame) == 0:
            return pd.DataFrame()

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


# The situations a scouting report's tendency table is built from, in the
# order it prints them. "Ahead" and "behind" are from the PITCHER's side, as
# scouting reports conventionally write them; "behind" is a hitter's count.
SITUATIONS: list[str] = [
    "first_pitch", "pitcher_ahead", "even", "pitcher_behind", "two_strikes",
]

SITUATION_LABELS: dict[str, str] = {
    "first_pitch": "First pitch",
    # Named the way the plan names them. The plan's "Hitter's count" line sat
    # directly above a row labelled "He's ahead" — the same word meaning the
    # opposite count — so both now use the hitter's/pitcher's count terms.
    "pitcher_ahead": "Pitcher's count",
    "even": "Even",
    "pitcher_behind": "Hitter's count",
    "two_strikes": "Two strikes",
}


def _situation_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks for each situation.

    "Even" excludes 0-0, because the first pitch has its own row and a hitter
    approaches it differently from 1-1 or 2-2. "Two strikes" overlaps the
    others on purpose: 1-2 is both "ahead" and "two strikes", and a report
    shows the two-strike row because it is the put-away question, which is
    asked separately from the count-leverage one.
    """
    balls, strikes = frame["balls"], frame["strikes"]
    first = (balls == 0) & (strikes == 0)
    return {
        "first_pitch": first,
        "pitcher_ahead": strikes > balls,
        "even": (balls == strikes) & ~first,
        "pitcher_behind": balls > strikes,
        "two_strikes": strikes == 2,
    }


def situational_usage(
    frame: pd.DataFrame,
    min_situation: int = 30,
) -> pd.DataFrame:
    """Pitch usage by situation, against each batter hand and overall.

    Returns one row per (situation, stand, pitch_type), where `stand` is "L",
    "R" or "All":

      situation     str, one of SITUATIONS
      stand         str
      pitch_type    str
      n             int, pitches of this type in this situation and side
      situation_n   int, all pitches in this situation and side
      share         float, n / situation_n
      reliable      bool, whether situation_n clears `min_situation`

    This is the tendency chart every advance report carries, and it is split
    by batter hand because pitchers often run entirely different plans to each
    side. An overall first-pitch mix averages a lefty plan with a righty plan
    and describes neither.

    Restricted to the arsenal so the rows agree with every other section of
    the report about which pitches exist.
    """
    columns = ["situation", "stand", "pitch_type", "n", "situation_n",
               "share", "reliable"]

    working = restrict_to_arsenal(frame)
    if len(working) == 0:
        return pd.DataFrame(columns=columns)

    sides = {"All": working}
    for side in ("L", "R"):
        sides[side] = working[working["stand"] == side]

    records = []
    for stand, side_frame in sides.items():
        if len(side_frame) == 0:
            continue
        masks = _situation_masks(side_frame)
        for situation in SITUATIONS:
            subset = side_frame[masks[situation]]
            situation_n = len(subset)
            if situation_n == 0:
                continue
            for pitch_type, n in subset["pitch_type"].value_counts().items():
                records.append({
                    "situation": situation,
                    "stand": stand,
                    "pitch_type": pitch_type,
                    "n": int(n),
                    "situation_n": situation_n,
                    "share": n / situation_n,
                    "reliable": situation_n >= min_situation,
                })

    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records)[columns]


# Counts where the hitter is ahead, and two-strike counts. 3-0 is left out of
# the hitter's set: nearly every pitcher throws a fastball there, so it says
# something about the count rather than about him.
HITTERS_COUNTS: list[str] = ["1-0", "2-0", "2-1", "3-1"]
TWO_STRIKE_COUNTS: list[str] = ["0-2", "1-2", "2-2", "3-2"]


def count_grid(
    frame: pd.DataFrame,
    min_count: int | None = None,
    min_usage: float = 0.05,
) -> pd.DataFrame:
    """Every pitch in every count, laid out for a balls-by-strikes grid.

    Returns one row per (pitch_type, count) for every pitch with at least
    `min_usage` of his total and every one of the twelve counts, including
    counts where he never threw that pitch:

      pitch_type, count, balls, strikes
      n          int, pitches of this type in this count
      count_n    int, all pitches in this count
      share      float, n / count_n
      own_rate   float, his overall rate for this pitch
      lift       float, share / own_rate
      log2_lift  float, log2(lift), floored at -5 for a lift of zero
      reliable   bool, whether count_n clears `min_count`

    A count is a coordinate, not a category. 2-1 and 1-2 are neighbours on a
    balls-by-strikes lattice, and laying the counts out that way is what lets a
    pitch that clusters in one corner be seen clustering there.

    The reliability gate is on the COUNT'S total, not on how many of this
    pitch were thrown in it, and that is a deliberate departure from the
    reviewer's version of this chart. Gating on the pitch blanked every cell
    where he rarely throws it — Yamamoto's cutter in 0-2 is 5 of 225 pitches,
    in 1-2 it is 6 of 330 — and those near-empty cells are the finding: he
    abandons the cutter with two strikes. A share's precision is governed by
    its denominator, so 5 of 225 is a well-measured 2%, not a missing value.
    The same principle governs the platoon table.

    Pitches below `min_usage` get no grid at all. A pitch thrown 3% of the time
    appears a handful of times per count, and its lift swings by a factor of two
    on a single pitch.
    """
    columns = ["pitch_type", "count", "balls", "strikes", "n", "count_n",
               "share", "own_rate", "lift", "log2_lift", "reliable"]

    if min_count is None:
        min_count = int(config.get("report.count_grid_min_count", 30))

    working = restrict_to_arsenal(frame)
    if len(working) == 0:
        return pd.DataFrame(columns=columns)

    overall = pitch_mix(working)
    pitches = [p for p in overall.index if overall[p] >= min_usage]
    per_count = working["count"].value_counts()
    tallies = working.groupby(["count", "pitch_type"]).size()

    records = []
    for pitch in pitches:
        own = float(overall[pitch])
        for count in ALL_COUNTS:
            count_n = int(per_count.get(count, 0))
            n = int(tallies.get((count, pitch), 0))
            share = n / count_n if count_n else float("nan")
            lift = share / own if (own and share == share) else float("nan")
            if lift != lift:
                log2_lift = float("nan")
            elif lift > 0:
                log2_lift = max(float(np.log2(lift)), -5.0)
            else:
                log2_lift = -5.0
            balls, strikes = (int(x) for x in count.split("-"))
            records.append({
                "pitch_type": pitch,
                "count": count,
                "balls": balls,
                "strikes": strikes,
                "n": n,
                "count_n": count_n,
                "share": share,
                "own_rate": own,
                "lift": lift,
                "log2_lift": log2_lift,
                "reliable": count_n >= min_count,
            })

    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records)[columns]


def count_region_lifts(frame: pd.DataFrame, min_usage: float = 0.05) -> pd.DataFrame:
    """Each pitch's lift pooled over hitter's counts and over two-strike counts.

    Returns one row per pitch: own_rate, hitter_lift, hitter_n, two_strike_lift,
    two_strike_n. Pooled by summing pitches across the counts in each region
    before dividing, so a big count weighs more than a small one — averaging
    the per-count lifts would let 3-1's 83 pitches count as much as 1-0's 307.
    """
    columns = ["own_rate", "hitter_lift", "hitter_n",
               "two_strike_lift", "two_strike_n"]

    working = restrict_to_arsenal(frame)
    if len(working) == 0:
        return pd.DataFrame(columns=columns)

    overall = pitch_mix(working)
    hitters = working[working["count"].isin(HITTERS_COUNTS)]
    two = working[working["count"].isin(TWO_STRIKE_COUNTS)]

    records = []
    for pitch in [p for p in overall.index if overall[p] >= min_usage]:
        own = float(overall[pitch])
        h_share = (hitters["pitch_type"] == pitch).mean() if len(hitters) else float("nan")
        t_share = (two["pitch_type"] == pitch).mean() if len(two) else float("nan")
        records.append({
            "pitch_type": pitch,
            "own_rate": own,
            "hitter_lift": h_share / own if own else float("nan"),
            "hitter_n": len(hitters),
            "two_strike_lift": t_share / own if own else float("nan"),
            "two_strike_n": len(two),
        })

    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records).set_index("pitch_type")[columns]


def count_lifts(
    frame: pd.DataFrame,
    min_n: int = 20,
    min_count_pitches: int = 20,
) -> pd.DataFrame:
    """How much each count changes the odds of each pitch, against his own mix.

    Returns one row per (count, pitch_type) where the pitch was thrown at least
    `min_n` times in that count, sorted by `score` descending:

      count        str
      pitch_type   str
      n            int, pitches of this type in this count
      count_n      int, all pitches in this count
      share        float, P(pitch | count)
      own_rate     float, his overall rate for that pitch
      lift         float, share / own_rate
      score        float, (lift - 1) * n

    This exists because predictability() reports only the most-used pitch in
    each count, and the most useful pattern is frequently not the most-used
    pitch. Yamamoto goes to his cutter 27% of the time in 2-1 — 2.4 times his
    own overall rate, and a genuinely exploitable tendency — while the
    four-seam is still the plurality there, so a top-pitch-only table never
    mentions it.

    Scoring matches sequencing.setup_pairs and location_tells: excess lift
    times sample size, so a dramatic lift on a thin count cannot outrank a
    moderate one that can be trusted.
    """
    columns = ["count", "pitch_type", "n", "count_n", "share",
               "own_rate", "lift", "score",
               "rv_when", "rv_base", "rv_delta", "runs_cost",
               "runs_lo", "runs_hi", "significant", "rv_p",
               "xwoba_when", "xwoba_delta"]

    if len(frame) == 0:
        return pd.DataFrame(columns=columns)

    working = restrict_to_arsenal(frame)
    if len(working) == 0:
        return pd.DataFrame(columns=columns)

    overall = pitch_mix(working)
    per_count = working["count"].value_counts()

    records = []
    for count_label, group in working.groupby("count"):
        count_n = len(group)
        if count_n < min_count_pitches:
            continue

        for pitch_type, n in group["pitch_type"].value_counts().items():
            if n < min_n:
                continue

            share = n / count_n
            own_rate = float(overall.get(pitch_type, float("nan")))
            lift = share / own_rate if own_rate else float("nan")

            # Outcomes when he goes to this pitch in this count, against what
            # the pitch does for him generally. A count where he becomes
            # predictable with a pitch that still works is a tendency; one
            # where the predictable pitch gets hit is the finding.
            in_count = group[group["pitch_type"] == pitch_type]
            all_of_pitch = working[working["pitch_type"] == pitch_type]
            outcome = events.outcome_delta(in_count, all_of_pitch)

            records.append({
                "count": count_label,
                "pitch_type": pitch_type,
                "n": int(n),
                "count_n": count_n,
                "share": share,
                "own_rate": own_rate,
                "lift": lift,
                "score": (lift - 1.0) * n,
                "rv_when": outcome["rv_when"],
                "rv_base": outcome["rv_base"],
                "rv_delta": outcome["rv_delta"],
                "runs_cost": outcome["runs_cost"],
                "runs_lo": outcome["runs_lo"],
                "runs_hi": outcome["runs_hi"],
                "significant": outcome["significant"],
                "rv_p": outcome["rv_p"],
                "xwoba_when": outcome["xwoba_when"],
                "xwoba_delta": outcome["xwoba_delta"],
            })

    if not records:
        return pd.DataFrame(columns=columns)

    table = pd.DataFrame(records)
    order = [c for c in ALL_COUNTS if c in set(table["count"])]
    table["_order"] = table["count"].map({c: i for i, c in enumerate(order)})
    return (
        table.sort_values("score", ascending=False)
        .drop(columns="_order")
        .reset_index(drop=True)[columns]
    )


def predictability(
    frame: pd.DataFrame,
    min_pitches: int = 20,
    min_arsenal_share: float | None = None,
) -> pd.DataFrame:
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
      own_rate        float, his overall rate for that pitch, all counts
      own_lift        float, top_share / own_rate
      top_pitch_n     int, pitches of that type thrown in that count

    `own_lift` is the column that makes this table answer a hitter's question.
    A share against the league rate tells you what kind of pitcher this is: a
    splitter specialist throws it far above league in all twelve counts, which
    is one fact reported twelve times. A share against his OWN overall rate
    tells you what he is about to throw — whether this count changes his mind.
    Those are different questions and the second is the one a hitter is asking
    in the box.

    Yamamoto is the case that proves it. His splitter beats league by 5-7x in
    every two-strike count and buries the genuinely useful pattern: a cutter at
    2.4x his own rate in 2-1 and 3-1, which he then almost abandons with two
    strikes. Against league the cutter never stood out, because his cutter
    usage overall is only 1.5x league.

    Counts with fewer than `min_pitches` are excluded. Entropy computed over a
    handful of pitches is noise wearing a number.

    A single-pitch arsenal is handled separately, since log2(1) = 0 would make
    the normalization undefined; such a pitcher scores 1.0 in every count by
    definition.

    Arsenal membership is gated by usage share (see primary_arsenal) rather
    than counting every distinct label that appears. A single misclassified
    pitch would otherwise add a whole category to k, raise the entropy ceiling,
    and quietly depress every score in the table.
    """
    columns = ["n", "entropy", "predictability", "top_pitch", "top_share",
               "own_rate", "own_lift", "top_pitch_n"]

    if len(frame) == 0:
        return pd.DataFrame(columns=columns)

    arsenal = primary_arsenal(frame, min_share=min_arsenal_share)
    if not arsenal:
        return pd.DataFrame(columns=columns)

    # Restrict to real offerings before measuring anything. Gating only the
    # ceiling while leaving stray pitches in the distribution would let the
    # observed entropy exceed log2(k) and drive predictability below zero.
    frame = frame[frame["pitch_type"].isin(arsenal)]

    k = len(arsenal)
    max_entropy = np.log2(k) if k > 1 else None

    per_count = frame["count"].value_counts()
    mix = mix_by_count(frame, min_pitches=min_pitches)

    # His own overall rate for each pitch, measured over the same restricted
    # population the shares are, so the ratio compares like with like.
    overall = pitch_mix(frame)

    rows = []
    for count_label, shares in mix.iterrows():
        nonzero = shares[shares > 0]
        entropy = -(nonzero * np.log2(nonzero)).sum()
        pred = 1.0 if max_entropy is None else 1 - entropy / max_entropy

        top_pitch = shares.idxmax()
        top_share = float(shares.max())
        own_rate = float(overall.get(top_pitch, float("nan")))
        in_count = int(per_count[count_label])

        rows.append({
            "n": in_count,
            "entropy": float(entropy),
            "predictability": float(pred),
            "top_pitch": top_pitch,
            "top_share": top_share,
            "own_rate": own_rate,
            "own_lift": top_share / own_rate if own_rate else float("nan"),
            "top_pitch_n": int(round(top_share * in_count)),
        })

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows, index=mix.index)[columns]


