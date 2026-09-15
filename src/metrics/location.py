"""Where each pitch is thrown, in coordinates that survive comparison.

Raw `plate_x` and `plate_z` cannot be compared across pitchers or across
batters. Two normalizations fix that, and both are necessary:

Horizontally, Savant measures from the catcher's point of view, so the same
physical location carries opposite signs depending on who is throwing and who
is hitting. "Down and away" is a different half of the plate against a lefty
than against a righty, and averaging the two together produces a pitcher who
appears to live over the middle while actually living on one edge against each
side. Coordinates here are therefore re-expressed twice: once relative to the
pitcher's arm side, which is the frame a pitching coach describes location in,
and once relative to the batter, which is the frame that determines whether a
pitch is in or away.

Vertically, the strike zone is a function of the hitter's height. A pitch at
2.9 feet is at the letters on one batter and above the head of another, so
height is expressed as a fraction of that batter's own zone rather than in
feet.

The sign convention is not assumed. It was established from hit-by-pitch
locations, which are by definition as far inside as a pitch can get, for both
a right-handed and a left-handed pitcher: those pitches sit at negative
`plate_x` against right-handed batters and positive `plate_x` against
left-handed ones, in both cases.

Inputs are expected to have passed through `clean.py` and
`events.add_event_flags()`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.metrics import counts

FEET_TO_INCHES = 12.0

# Half the width of home plate, in feet. Splitting that half-width into thirds
# puts the band edges at +/- 0.236 ft, which is where the inner third of the
# plate ends.
PLATE_HALF_WIDTH_FEET = 17.0 / 2.0 / FEET_TO_INCHES
BAND_EDGE_FEET = PLATE_HALF_WIDTH_FEET / 3.0

# Vertical band edges, as a fraction of the batter's own strike zone. 0.0 is
# the bottom of the zone and 1.0 the top, so values outside [0, 1] are balls.
LOW_BAND_EDGE = 1.0 / 3.0
HIGH_BAND_EDGE = 2.0 / 3.0

# Used when a coordinate is missing. Savant fails to locate a small number of
# pitches every season, and they must not be silently binned as "middle".
UNKNOWN = "UNKNOWN"

QUADRANTS: list[str] = ["UP-ARM", "UP-GLOVE", "DOWN-ARM", "DOWN-GLOVE"]


def add_location_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add normalized location columns.

    Adds:
      x_arm      float, feet toward the pitcher's arm side (positive = arm side)
      x_in       float, feet toward the batter (positive = inside)
      z_norm     float, height as a fraction of that batter's strike zone,
                 where 0.0 is the bottom of the zone and 1.0 the top
      h_band     str, ARM / MIDDLE / GLOVE, by thirds of the plate
      v_band     str, UP / MIDDLE / DOWN, by thirds of the zone
      quadrant   str, one of QUADRANTS, splitting at the middle of each axis

    `h_band` and `v_band` divide into thirds and are what a heat map renders.
    `quadrant` splits each axis in two instead, because a four-way split is
    what makes "40% of his sliders go to one quadrant" a meaningful sentence —
    against a uniform 25%, that is a real concentration, whereas one cell of a
    nine-cell grid has a uniform share of 11% and clears any fixed threshold
    far too easily.

    A pitch missing plate coordinates, strike-zone bounds, or a batter stand is
    labelled UNKNOWN on every band rather than being assigned a location. Does
    not mutate the input frame.
    """
    result = frame.copy()

    plate_x = pd.to_numeric(result.get("plate_x"), errors="coerce")
    plate_z = pd.to_numeric(result.get("plate_z"), errors="coerce")
    sz_top = pd.to_numeric(result.get("sz_top"), errors="coerce")
    sz_bot = pd.to_numeric(result.get("sz_bot"), errors="coerce")

    throws = result.get("p_throws", pd.Series(index=result.index, dtype=object))
    stand = result.get("stand", pd.Series(index=result.index, dtype=object))

    # A right-hander's arm side is the right-handed batter's box, which sits at
    # negative plate_x; a left-hander's arm side is the opposite. Flipping by
    # p_throws puts every pitcher in the same frame, so "arm side" means the
    # same thing in a table covering both.
    arm_sign = np.where(throws.eq("R"), -1.0, np.where(throws.eq("L"), 1.0, np.nan))
    result["x_arm"] = plate_x * arm_sign

    # Inside is toward the batter, which is the mirror image: negative plate_x
    # is inside to a righty, positive is inside to a lefty.
    in_sign = np.where(stand.eq("R"), -1.0, np.where(stand.eq("L"), 1.0, np.nan))
    result["x_in"] = plate_x * in_sign

    # A zero-height zone would divide by zero. It should never happen, but a
    # corrupt row should produce a missing value rather than an infinity that
    # propagates into every average downstream.
    zone_height = sz_top - sz_bot
    zone_height = zone_height.where(zone_height > 0)
    result["z_norm"] = (plate_z - sz_bot) / zone_height

    result["h_band"] = _band(
        result["x_arm"], [(-BAND_EDGE_FEET, "GLOVE"), (BAND_EDGE_FEET, "MIDDLE")], "ARM"
    )
    result["v_band"] = _band(
        result["z_norm"], [(LOW_BAND_EDGE, "DOWN"), (HIGH_BAND_EDGE, "MIDDLE")], "UP"
    )

    vertical = np.where(
        result["z_norm"].isna(), UNKNOWN,
        np.where(result["z_norm"] >= 0.5, "UP", "DOWN"),
    )
    horizontal = np.where(
        result["x_arm"].isna(), UNKNOWN,
        np.where(result["x_arm"] >= 0.0, "ARM", "GLOVE"),
    )
    result["quadrant"] = np.where(
        (vertical == UNKNOWN) | (horizontal == UNKNOWN),
        UNKNOWN,
        pd.Series(vertical, index=result.index).str.cat(
            pd.Series(horizontal, index=result.index), sep="-"
        ),
    )

    return result


def location_profile(
    frame: pd.DataFrame,
    min_pitches: int | None = None,
) -> pd.DataFrame:
    """Where each pitch type lives, and how tightly it is commanded.

    Returns a DataFrame indexed by pitch_type, sorted by usage descending:

      n              int, pitches with a usable location
      arm_side       float, mean horizontal location in inches, positive toward
                     the pitcher's arm side
      arm_side_sd    float, its standard deviation
      height         float, mean height as a fraction of the batter's zone
      height_sd      float, its standard deviation
      in_zone        float, share thrown inside the strike zone
      top_quadrant   str, the quadrant he goes to most with this pitch
      top_share      float, that quadrant's share of the pitch type

    The two standard deviations are the command read, and they are reported
    separately from the means because they answer a different question. A mean
    says where he intends the pitch to go; the spread says whether it gets
    there. A pitch with a tight spread in a predictable spot is attackable in a
    way that the same mean with a wide spread is not.
    """
    columns = [
        "n", "arm_side", "arm_side_sd", "height", "height_sd",
        "in_zone", "top_quadrant", "top_share",
    ]

    subset = _located(frame, min_pitches)
    if subset is None:
        return pd.DataFrame(columns=columns)

    records = []
    for pitch_type, group in subset.groupby("pitch_type"):
        shares = group["quadrant"].value_counts(normalize=True)
        records.append({
            "pitch_type": pitch_type,
            "n": len(group),
            "arm_side": float(group["x_arm"].mean()) * FEET_TO_INCHES,
            "arm_side_sd": float(group["x_arm"].std()) * FEET_TO_INCHES,
            "height": float(group["z_norm"].mean()),
            "height_sd": float(group["z_norm"].std()),
            "in_zone": float(group["is_in_zone"].eq(True).mean()),
            "top_quadrant": shares.index[0],
            "top_share": float(shares.iloc[0]),
        })

    table = pd.DataFrame(records).set_index("pitch_type")
    return table.sort_values("n", ascending=False)[columns]


def quadrant_shares(
    frame: pd.DataFrame,
    min_pitches: int | None = None,
) -> pd.DataFrame:
    """Share of each pitch type going to each quadrant.

    Rows are pitch types, columns are the four quadrants, and each ROW sums to
    1: "of his curveballs, what share went down and to the glove side". Every
    quadrant appears as a column even when unused, so the table has a stable
    shape the report can render without checking which columns exist.
    """
    subset = _located(frame, min_pitches)
    if subset is None:
        return pd.DataFrame(columns=QUADRANTS)

    table = pd.crosstab(subset["pitch_type"], subset["quadrant"], normalize="index")
    table = table.reindex(columns=QUADRANTS, fill_value=0.0).fillna(0.0)

    order = subset["pitch_type"].value_counts().index
    return table.reindex([p for p in order if p in table.index])


def location_leaks(
    frame: pd.DataFrame,
    min_pitches: int | None = None,
    min_share: float | None = None,
) -> pd.DataFrame:
    """Pitch types concentrated in one quadrant past the reporting threshold.

    Returns one row per qualifying pitch type, sorted by share descending:

      pitch_type   str
      n            int, pitches of this type with a usable location
      quadrant     str, the quadrant he concentrates in
      share        float, that quadrant's share
      excess       float, share minus the 0.25 a uniform pitcher would show
      in_zone      float, share of the pitch type thrown in the strike zone

    This is the input to the `location_leak` flag. Defaults come from
    `flags.thresholds.location_leak_share` and `flags.min_n.location_leak`.

    Concentration is not automatically a weakness — a pitcher who lives down
    and away with his slider is executing a plan, not leaking information. It
    becomes a weakness when the concentration is high enough that a hitter can
    commit to a region before recognizing the pitch, which is why the threshold
    is configurable and why the sample gate is enforced here rather than left
    to the caller.
    """
    columns = ["pitch_type", "n", "quadrant", "share", "excess", "in_zone"]

    if min_share is None:
        min_share = float(
            config.get("flags.thresholds.location_leak_share", 0.40)
        )
    if min_pitches is None:
        min_pitches = int(config.get("flags.min_n.location_leak", 40))

    profile = location_profile(frame, min_pitches=None)
    if len(profile) == 0:
        return pd.DataFrame(columns=columns)

    qualifying = profile[
        (profile["n"] >= min_pitches) & (profile["top_share"] >= min_share)
    ]
    if len(qualifying) == 0:
        return pd.DataFrame(columns=columns)

    uniform_share = 1.0 / len(QUADRANTS)
    result = (
        qualifying.reset_index()
        .rename(columns={"top_quadrant": "quadrant", "top_share": "share"})
    )
    result["excess"] = result["share"] - uniform_share

    return result.sort_values("share", ascending=False).reset_index(drop=True)[columns]


def zone_table(frame: pd.DataFrame, min_pitches: int | None = None) -> pd.DataFrame:
    """Pitch-type usage across Savant's thirteen zones, for the heat map.

    Rows are pitch types, columns are zones 1-9 (the strike zone, read left to
    right and top to bottom from the catcher's view) followed by 11-14 (the
    four regions outside it). Each row sums to 1.

    This uses Savant's own zone numbering rather than the normalized
    coordinates above, because the report renders a fixed grid and the zone
    field is already the grid. The normalized coordinates are what the
    comparisons and flags are built from.
    """
    zones = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]

    if len(frame) == 0 or "zone" not in frame.columns:
        return pd.DataFrame(columns=zones)

    keep = counts.primary_arsenal(frame, min_pitches=min_pitches)
    subset = frame[frame["pitch_type"].isin(keep)] if keep else frame.iloc[0:0]
    subset = subset[subset["zone"].notna()]
    if len(subset) == 0:
        return pd.DataFrame(columns=zones)

    table = pd.crosstab(
        subset["pitch_type"],
        subset["zone"].astype(int),
        normalize="index",
    )
    table = table.reindex(columns=zones, fill_value=0.0).fillna(0.0)

    order = subset["pitch_type"].value_counts().index
    return table.reindex([p for p in order if p in table.index])


def _band(values: pd.Series, edges: list[tuple[float, str]], above: str) -> pd.Series:
    """Bin a numeric column into named bands, leaving nulls as UNKNOWN.

    `edges` is a list of (upper_bound, label) pairs applied in order; anything
    at or above the last bound gets `above`.
    """
    result = pd.Series(above, index=values.index, dtype=object)
    for bound, label in reversed(edges):
        result = result.mask(values < bound, label)
    return result.mask(values.isna(), UNKNOWN)


def _located(frame: pd.DataFrame, min_pitches: int | None) -> pd.DataFrame | None:
    """Arsenal pitches carrying a usable location, or None if there are none."""
    if len(frame) == 0:
        return None

    keep = counts.primary_arsenal(frame, min_pitches=min_pitches)
    if not keep:
        return None

    working = add_location_frame(frame[frame["pitch_type"].isin(keep)])
    working = working[working["quadrant"] != UNKNOWN]

    return working if len(working) else None
