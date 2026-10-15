"""How many starts does the report need before it says something, and is it right?

Every check so far splits a major-league season in half, which gives each side
about 1,400 pitches from a starter. That happens to be a full college season
from a weekend starter (15 starts of about 95 pitches), and it is far more than
anyone has on him in March. This module asks what the report does with less:
three starts, five, ten.

For one pitcher's regular season:

  1. His games are shuffled and cut in two. One side is held out and never
     changes. It is the answer key: what he did in games the report did not
     see.
  2. From the other side, the first 3, 5 and 10 games in the shuffled order are
     taken as training samples, each inside the next, and then the whole side.
  3. A report is built from each sample by the functions the real report
     uses, at the bars in config.yaml, and everything it would print is scored
     against the held-out games.

That is repeated for a number of shuffles per pitcher, and two things are
counted at each size:

  printed   how much a report of that size says at all: lines on the card,
            rows of the sequence graph it can draw, follow-ups it colours,
            pitch pairs it puts a tunneling word on
  scored    of what it said, how often the held-out games clear the same bar
            (`held`), and how often they point the other way (`wrong`)

Two of the kinds scored are not lines the report prints. The count tendency
and the setup by pitch and height are rules of the findings engine
(`flags.py`), which the report still computes and no longer lists. They are
scored here because they are the best-measured answer to one question: what
does a three-standard-error bar let through on a small sample? The count
tendency makes ten times the claims the sequence graph does.

The bars are counts, not shares — 40 follow-ups to a side before a row of the
sequence graph is read, 20 of a follow-up before it is coloured, three standard
errors — so the hope was that a small sample would say less and be wrong no
more often. Half of that is what happens. A small report says much less, and
what it says almost never points the wrong way. But what clears three standard
errors on three starts is a mix of real tendencies and ordinary ones on a good
run, and the held-out games give back much of their size, so a coloured cell
clears its bar again far less often than one found on a half season. The
numbers are in the README.

Whole games are drawn, not pitches, because pitch sequencing lives inside a
plate appearance and a scout's sample arrives in games. They are drawn at
random, not in date order, so that the only thing changing between sizes is
the size. A real report is built from his first games and read against his
later ones, and he changes in between, so one more sample is scored as a
control: the first half of his season by date against the second, as
`validation.split_halves` cuts it. The gap between that and the shuffled
half is what a season's drift costs.

What this cannot tell you. The pitchers are major-league starters cut down to
a college-sized sample. They are not college pitchers. A college starter may
well be easier to read — fewer pitches, less command, a catcher calling a
simpler game — and nothing here measures that. This answers the sample-size
question and no other: with this few pitches, does the report still hold?

And the draws are not independent. Twenty shuffles of one pitcher are the same
arm twenty times, so every interval here resamples pitchers, not rows, and a
rate that rests on fewer than MIN_PITCHERS pitchers is not reported at all.

The report acts on what this found: under `report.standouts_min_pitches` it
stops marking follow-ups and counts as standing out (see `report._sample`).
This check scores the rules as if that bar were not there, since it is the
evidence the bar was set from.

Seasons can be pooled, and the two cached ones are not alike: every cutoff was
set on 2025. Pooling is for the shape, how a rate climbs with the sample. For
the level, run the season no cutoff has seen on its own.
"""

from __future__ import annotations

import math
import zlib

import numpy as np
import pandas as pd

from src import config, gameplan, replication, report, validation
from src.metrics import counts, sequencing, tunneling
from src.validation import pitcher_interval

# The label for the largest training sample: every game on the training side.
HALF = "half"

# The control: his first half by date, scored on his second. One per season.
BY_DATE = "by date"

# The card's "Lately" line compares his last outings with the ones before. On
# games drawn at random there is no "lately", so the line is left out of the
# count of what a report prints.
UNORDERED_LABELS: frozenset[str] = frozenset({"Lately"})

# A rate is reported when at least this many pitchers made the claim. Under
# that it describes those pitchers, not the rule, and a bootstrap over three
# or four of them has no percentiles worth printing.
MIN_PITCHERS = 6

KIND_LABELS: dict[str, str] = {
    **validation.RULE_LABELS,
    "follow_up_more": "Sequence graph: more than usual",
    "follow_up_less": "Sequence graph: less than usual",
    "tunnel_same": "Tunneling: same window",
    "tunnel_apart": "Tunneling: apart early",
    # Rules the report computes and does not print; see the module docstring.
    "count_lift": "Rule: count tendency",
    "sequencing": "Rule: setup by pitch and height",
}

PRINTED_LABELS: dict[str, str] = {
    "pitches": "Pitches in the sample",
    "card_lines": "Lines on the card",
    "graph_rows": "Sequence graph: rows it can read",
    "graph_coloured": "Sequence graph: follow-ups coloured",
    "sequence_lines": "Card: a Sequence line",
    "tunnel_pairs": "Tunneling: pairs in the table",
    "tunnel_read": "Tunneling: pairs given a word",
    "count_tendencies": "Rule: count tendencies found",
}


def starts() -> tuple[int, ...]:
    """The sample sizes tried, in games."""
    return tuple(int(n) for n in config.get("sample_size.starts", [3, 5, 10]))


def is_starter_season(frame: pd.DataFrame) -> bool:
    """A season with enough starts to hold half out and still draw from the rest.

    The largest fixed size has to fit on the training side, so the season needs
    twice that many games, and they have to be starts: a reliever's ten games
    are 150 pitches, which is a different sample from ten starts.
    """
    regular = validation._regular(frame)
    if len(regular) == 0:
        return False
    per_game = regular.groupby("game_pk").size()
    return (len(per_game) >= int(config.get("sample_size.min_games", 20))
            and float(per_game.median())
            >= float(config.get("sample_size.min_pitches_per_game", 60)))


def rng_for(name: str, season: int, seed: int) -> np.random.Generator:
    """A generator that depends only on the pitcher, the season and the seed.

    So adding a pitcher to the list, or running one pitcher alone, does not
    change anyone else's draws.
    """
    return np.random.default_rng([seed, int(season), zlib.crc32(name.encode())])


def draw(frame: pd.DataFrame, rng: np.random.Generator,
         sizes: tuple[int, ...] | None = None) -> tuple[dict, pd.DataFrame]:
    """One shuffle: ({size: training sample}, the games held out).

    The training samples are nested — the three-start sample is inside the
    five-start one — and all are scored against the same held-out games, so a
    difference between two sizes is a difference in what was added.
    """
    sizes = starts() if sizes is None else sizes
    regular = validation._regular(frame)
    order = rng.permutation(np.sort(regular["game_pk"].unique()))
    cut = math.ceil(len(order) / 2)
    training, held_out = order[:cut], order[cut:]

    samples: dict = {n: regular[regular["game_pk"].isin(training[:n])]
                     for n in sizes if n < len(training)}
    samples[HALF] = regular[regular["game_pk"].isin(training)]
    return samples, regular[regular["game_pk"].isin(held_out)]


# ---------------------------------------------------------------------------
# What a report of this size prints
# ---------------------------------------------------------------------------


def _sequence_table(frame: pd.DataFrame) -> pd.DataFrame:
    table = sequencing.transitions(frame)
    return table.assign(standout=sequencing.transition_standouts(table))


def printed(sample: pd.DataFrame) -> dict:
    """How much a report built from `sample` says, as counts.

    Read through the functions the report itself calls, so a count here is a
    count of things on the page.
    """
    table = _sequence_table(sample)
    keys = gameplan.card_keys(gameplan.build(sample), table)
    lines = [k for side in ("L", "R", "both") for k in keys[side]
             if k["label"] not in UNORDERED_LABELS]

    row_gate = int(config.get("flags.min_n.sequence_follow_ups", 40))
    rows = table.drop_duplicates(["stand", "pitch_type"])

    cells = counts.count_lifts(
        sample, min_n=int(config.get("flags.min_n.predictable_count", 20)))
    if len(cells):
        cells = cells[
            (cells["lift"] >= float(config.get(
                "flags.thresholds.predictable_count_own_lift", 1.50)))
            & (cells["lift_z"] >= replication._min_z())]

    # The table prints the most common pairs to each side, up to its limit.
    pairs = tunneling.pair_table(sample)
    if len(pairs):
        pairs = (pairs[pairs["pitch_type"] != pairs["next_pitch"]]
                 .groupby("stand").head(report.TUNNEL_ROWS_PER_SIDE))

    return {
        "pitches": int(len(sample)),
        "card_lines": len(lines),
        "graph_rows": int((rows["row_n"] >= row_gate).sum()),
        "graph_coloured": int((table["standout"] != "").sum()),
        "sequence_lines": sum(k["label"] == "Sequence" for k in lines),
        "tunnel_pairs": int(len(pairs)),
        "tunnel_read": int((pairs["reading"] != "").sum()) if len(pairs) else 0,
        "count_tendencies": int(len(cells)),
    }


# ---------------------------------------------------------------------------
# Whether what it printed was right
# ---------------------------------------------------------------------------


def scored(sample: pd.DataFrame, held_out: pd.DataFrame) -> list[dict]:
    """Everything the sample's report claims, checked against the held-out games.

    One dict per claim: `kind`, `key`, `first` and `second` (the number behind
    it in the sample and in the held-out games), `held` and `wrong`.

    The plan's lines come from `validation.compare` and the rest from the
    replication check's own functions, so "held" means here what it means in
    those: the held-out games clear the bar the line was written at. `wrong`
    is the failure that costs a hitter: a plan line whose held-out call points
    to a different pitch, or a tendency that came back on the other side of
    his usual rate. A claim the held-out games are too thin to measure
    produces no row, as in both of those checks.
    """
    rows: list[dict] = []
    if len(sample) == 0 or len(held_out) == 0:
        return rows

    for r in validation.compare(sample, held_out):
        if not validation._advice_given(r):
            continue
        rows.append({
            "kind": r["rule"], "key": f"{r['side']} {r['call_first']}",
            "first": r["value_first"], "second": r["value_second"],
            "held": bool(r["same_call"]),
            "wrong": bool(r["flipped"] and not r["same_call"]),
        })

    for measure in (replication._count_lift_rows, replication._follow_up_rows,
                    replication._sequencing_rows, replication._tunnel_reading_rows):
        for r in measure(sample, held_out):
            rows.append({
                "kind": r["kind"], "key": r["key"],
                "first": r["first"], "second": r["second"],
                "held": bool(r["held"]), "wrong": not r["direction"],
            })
    return rows


def follow_up_shares(sample: pd.DataFrame, held_out: pd.DataFrame) -> list[dict]:
    """Each follow-up share the sequence graph would draw, in both parts.

    One row per pitch-to-next-pitch cell, by batter side, where the pitch has
    the graph's own number of follow-ups in the sample and in the held-out
    games. This is the number a hitter reads off the graph — "after a slider,
    another slider 64% of the time" — whether or not it was coloured.
    """
    row_gate = int(config.get("flags.min_n.sequence_follow_ups", 40))
    first, second = sequencing.transitions(sample), sequencing.transitions(held_out)
    if len(first) == 0 or len(second) == 0:
        return []
    keys = ["stand", "pitch_type", "next_pitch"]
    both = first.merge(second, on=keys, suffixes=("_first", "_second"))
    both = both[(both["row_n_first"] >= row_gate) & (both["row_n_second"] >= row_gate)]
    return [{"stand": r.stand, "pitch_type": r.pitch_type, "next_pitch": r.next_pitch,
             "row_n": int(r.row_n_first), "first": float(r.share_first),
             "second": float(r.share_second)}
            for r in both.itertuples()]


# ---------------------------------------------------------------------------
# One pitcher
# ---------------------------------------------------------------------------


def run_pitcher(frame: pd.DataFrame, name: str, season: int,
                draws: int | None = None, seed: int | None = None) -> dict:
    """Every draw at every size for one pitcher-season.

    Returns {"printed": [...], "scored": [...], "shares": [...]}, each row
    carrying `pitcher`, `season`, `draw` and `starts`. After the draws comes
    one more sample per season, `starts` = BY_DATE: his first half in date
    order, scored on his second.
    """
    draws = int(config.get("sample_size.draws", 20)) if draws is None else draws
    seed = int(config.get("sample_size.seed", 17)) if seed is None else seed
    out: dict = {"printed": [], "scored": [], "shares": []}
    rng = rng_for(name, season, seed)

    def record(size, number: int, sample: pd.DataFrame, held_out: pd.DataFrame):
        tag = {"pitcher": name, "season": season, "draw": number, "starts": size}
        out["printed"].append({**tag, "games": int(sample["game_pk"].nunique()),
                               "held_out_pitches": int(len(held_out)),
                               **printed(sample)})
        out["scored"].extend({**tag, **r} for r in scored(sample, held_out))
        out["shares"].extend({**tag, **r}
                             for r in follow_up_shares(sample, held_out))

    for number in range(draws):
        samples, held_out = draw(frame, rng)
        for size, sample in samples.items():
            record(size, number, sample, held_out)

    record(BY_DATE, 0, *validation.split_halves(frame))
    return out


# ---------------------------------------------------------------------------
# Summarising across pitchers
# ---------------------------------------------------------------------------


def _sizes(table: pd.DataFrame) -> list:
    """The sizes in a table, smallest first, the two half seasons last."""
    present = set(table["starts"])
    named = [s for s in (HALF, BY_DATE) if s in present]
    return sorted((s for s in present if s not in named), key=int) + named


def summarise_printed(rows: list[dict]) -> list[dict]:
    """One line per size and count: the average report, and how many say anything.

    `mean` is per report. `any` is the share of reports where the count is not
    zero, which for a small sample is the more honest number: an average of
    0.3 coloured follow-ups is seven reports in ten with none.
    """
    table = pd.DataFrame(rows)
    out: list[dict] = []
    if table.empty:
        return out
    for size in _sizes(table):
        sub = table[table["starts"] == size]
        for metric, label in PRINTED_LABELS.items():
            lo, hi = pitcher_interval(sub, metric)
            out.append({
                "starts": size, "metric": metric, "label": label,
                "reports": int(len(sub)), "pitchers": int(sub["pitcher"].nunique()),
                "mean": float(sub[metric].mean()),
                "mean_lo": lo, "mean_hi": hi,
                "median": float(sub[metric].median()),
                "any": float((sub[metric] > 0).mean()),
            })
    return out


def summarise_scored(rows: list[dict], reports: list[dict]) -> list[dict]:
    """One line per kind of claim and size: how many, and how often right.

    `per_report` divides by every report of that size, including the ones
    that made no such claim, so it falls as the sample does. `held_pct` and
    `wrong_pct` divide by the claims made, and are left empty, with their
    intervals and medians, when fewer than MIN_PITCHERS pitchers made any:
    `claims` and `pitchers` still say how few there were.
    """
    table = pd.DataFrame(rows)
    out: list[dict] = []
    if table.empty:
        return out
    made = pd.DataFrame(reports).groupby("starts").size()
    for kind, label in KIND_LABELS.items():
        of_kind = table[table["kind"] == kind]
        for size in _sizes(table):
            sub = of_kind[of_kind["starts"] == size]
            n = int(len(sub))
            pitchers = int(sub["pitcher"].nunique()) if n else 0
            enough = pitchers >= MIN_PITCHERS
            held_lo, held_hi = pitcher_interval(sub, "held", min_pitchers=MIN_PITCHERS)
            wrong_lo, wrong_hi = pitcher_interval(sub, "wrong", min_pitchers=MIN_PITCHERS)
            out.append({
                "kind": kind, "label": label, "starts": size,
                "claims": n,
                "pitchers": pitchers,
                "per_report": n / int(made.get(size, 0)) if made.get(size, 0) else float("nan"),
                "held_pct": float(sub["held"].mean()) if enough else float("nan"),
                "held_lo": held_lo, "held_hi": held_hi,
                "wrong_pct": float(sub["wrong"].mean()) if enough else float("nan"),
                "wrong_lo": wrong_lo, "wrong_hi": wrong_hi,
                "median_first": float(sub["first"].median()) if enough else float("nan"),
                "median_second": float(sub["second"].median()) if enough else float("nan"),
            })
    return out


def summarise_shares(rows: list[dict]) -> list[dict]:
    """One line per size: how far a follow-up share sits from the held-out one.

    In points of share, because that is how the graph is read. `r` is across
    every cell, so most of it is his pitch mix; the distances are the answer.
    """
    table = pd.DataFrame(rows)
    out: list[dict] = []
    if table.empty:
        return out
    table = table.assign(gap=(table["first"] - table["second"]).abs())
    for size in _sizes(table):
        sub = table[table["starts"] == size]
        lo, hi = pitcher_interval(sub, "gap")
        out.append({
            "starts": size, "cells": int(len(sub)),
            "pitchers": int(sub["pitcher"].nunique()),
            "median_gap": float(sub["gap"].median()),
            "mean_gap": float(sub["gap"].mean()),
            "mean_gap_lo": lo, "mean_gap_hi": hi,
            "p90_gap": float(sub["gap"].quantile(0.9)),
            "within_five": float((sub["gap"] <= 0.05).mean()),
            "within_ten": float((sub["gap"] <= 0.10).mean()),
            "r": replication._correlation(sub[["first", "second"]]),
        })
    return out
