"""Combining seasons: what changed between them.

A pitcher who missed half a season leaves too few pitches for count- and
side-level tendencies to settle. Pooling his previous season fixes the sample
and creates a new problem: pitchers change between seasons. They add a pitch,
shelve one, lose a tick, or reshuffle how often they throw what. Pooled
numbers average over all of that without saying so.

So when seasons are combined, the report pools them for tendencies — usage
rates are what need the sample — and separately compares the latest season
against the earlier ones to name what moved. A pitch he has stopped throwing
is left out of the report entirely, because a plan that tells a hitter to look
for a pitch that no longer exists is worse than a plan built on less data.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.metrics import splits


def _cut(key: str, default: float) -> float:
    return float(config.get(f"seasons.{key}", default))


def season_counts(frame: pd.DataFrame) -> list[dict]:
    """Pitches and games per season, oldest first."""
    if "game_year" not in frame.columns or len(frame) == 0:
        return []
    out = []
    for year, group in frame.groupby("game_year"):
        out.append({
            "season": int(year),
            "pitches": int(len(group)),
            "games": int(group["game_pk"].nunique()),
        })
    return out


def season_label(seasons: list[int]) -> str:
    """"2025–2026" for a run of years, "2024, 2026" for a gap, "2026" for one."""
    years = sorted(set(int(s) for s in seasons))
    if len(years) == 1:
        return str(years[0])
    if years == list(range(years[0], years[-1] + 1)):
        return f"{years[0]}–{years[-1]}"
    return ", ".join(str(y) for y in years)


def season_changes(frame: pd.DataFrame) -> dict | None:
    """Compare the latest season with the earlier ones.

    Returns None for a single season. Otherwise:
      latest, earlier     the latest year and the list of earlier years
      n_latest, n_earlier pitches on each side of the comparison
      new                 [(pitch, latest_share)] — barely thrown before, real now
      dropped             [(pitch, earlier_share)] — real before, barely thrown now
      shifts              [(pitch, earlier_share, latest_share)] — usage moved
                          by a lot and by more than noise
      velo                [(pitch, earlier_mph, latest_mph)] — speed moved
    """
    if "game_year" not in frame.columns:
        return None
    years = sorted(int(y) for y in frame["game_year"].dropna().unique())
    if len(years) < 2:
        return None

    latest = years[-1]
    is_latest = frame["game_year"] == latest
    now, before = frame[is_latest], frame[~is_latest]
    n_now, n_before = len(now), len(before)
    if n_now == 0 or n_before == 0:
        return None

    absent = _cut("absent_below", 0.01)
    present = _cut("present_at", 0.03)
    min_shift = _cut("usage_shift", 0.08)
    min_z = _cut("usage_shift_z", 2.5)
    min_velo = _cut("velo_change_mph", 1.0)
    min_pitches = int(_cut("min_pitches", 50))

    count_now = now["pitch_type"].value_counts()
    count_before = before["pitch_type"].value_counts()
    pitches = sorted(set(count_now.index) | set(count_before.index),
                     key=lambda p: -(count_now.get(p, 0) + count_before.get(p, 0)))

    new, dropped, shifts, velo = [], [], [], []
    for p in pitches:
        x_now, x_before = int(count_now.get(p, 0)), int(count_before.get(p, 0))
        s_now, s_before = x_now / n_now, x_before / n_before

        if s_before < absent and s_now >= present:
            new.append((p, s_now))
            continue
        if s_now < absent and s_before >= present:
            dropped.append((p, s_before))
            continue
        if s_now < present and s_before < present:
            continue

        z = splits._two_proportion_z(x_now, n_now, x_before, n_before)
        if abs(s_now - s_before) >= min_shift and z == z and abs(z) >= min_z:
            shifts.append((p, s_before, s_now))

        if x_now >= min_pitches and x_before >= min_pitches:
            v_now = now.loc[now["pitch_type"] == p, "release_speed"].mean()
            v_before = before.loc[before["pitch_type"] == p, "release_speed"].mean()
            if v_now == v_now and v_before == v_before and abs(v_now - v_before) >= min_velo:
                velo.append((p, float(v_before), float(v_now)))

    return {
        "latest": latest,
        "earlier": years[:-1],
        "n_latest": n_now,
        "n_earlier": n_before,
        "new": new,
        "dropped": dropped,
        "shifts": shifts,
        "velo": velo,
    }


def drop_retired_pitches(frame: pd.DataFrame, changes: dict | None) -> pd.DataFrame:
    """The frame without pitches he has stopped throwing."""
    if not changes or not changes["dropped"]:
        return frame
    gone = [p for p, _ in changes["dropped"]]
    return frame[~frame["pitch_type"].isin(gone)]
