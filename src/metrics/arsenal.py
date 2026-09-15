"""Pitch-type profiles: what each offering is, and where it comes from.

Two things live here. The first is the arsenal table every scouting report
opens with — velocity, movement, and results by pitch type. The second is
release consistency, which asks a different question: does the pitcher give the
pitch away before it leaves his hand?

Release consistency is the cheap half of tunneling. Full tunnel metrics need
trajectory reconstruction; a release point that shifts by pitch type is visible
in three columns and is the more immediately actionable finding, because a
hitter who can identify the pitch out of the hand does not need to read spin.

Inputs are expected to have passed through `clean.py` and
`events.add_event_flags()`.
"""

from __future__ import annotations

import pandas as pd

from src.metrics import counts

# Statcast reports movement in feet. Scouting reports use inches.
FEET_TO_INCHES = 12.0


def profile(frame: pd.DataFrame, min_pitches: int | None = None) -> pd.DataFrame:
    """Per-pitch-type physical characteristics and results.

    Returns a DataFrame indexed by pitch_type, sorted by usage descending:

      n            int, pitches thrown
      usage        float, share of all pitches
      velo         float, average release speed (mph)
      velo_max     float, 95th percentile release speed
      spin         float, average spin rate (rpm)
      h_break      float, horizontal movement (inches, catcher's view)
      v_break      float, vertical movement (inches, positive is "rise")
      extension    float, release extension toward the plate (feet)
      zone_rate    float, share thrown inside the strike zone
      swing_rate   float, share the batter offered at
      whiff_rate   float, whiffs per SWING (not per pitch)
      chase_rate   float, swings per pitch OUTSIDE the zone
      xwoba        float, expected wOBA on contact
      run_value    float, average change in run expectancy

    Only pitch types clearing the arsenal gate are included, so a handful of
    misclassified pitches do not appear as a phantom sixth offering.

    velo_max is the 95th percentile rather than the maximum, because a single
    mis-tracked reading would otherwise define the top of a pitcher's range.
    """
    columns = [
        "n", "usage", "velo", "velo_max", "spin", "h_break", "v_break",
        "extension", "zone_rate", "swing_rate", "whiff_rate", "chase_rate",
        "xwoba", "run_value",
    ]

    if len(frame) == 0:
        return pd.DataFrame(columns=columns)

    keep = counts.primary_arsenal(frame, min_share=None, min_pitches=min_pitches)
    if not keep:
        return pd.DataFrame(columns=columns)

    subset = frame[frame["pitch_type"].isin(keep)]
    total = len(subset)

    records = []
    for pitch_type, group in subset.groupby("pitch_type"):
        swings = int(group["is_swing"].eq(True).sum())
        whiffs = int(group["is_whiff"].eq(True).sum())

        out_of_zone = group[group["zone"].notna() & ~group["is_in_zone"].eq(True)]
        chases = int(out_of_zone["is_swing"].eq(True).sum())

        records.append({
            "pitch_type": pitch_type,
            "n": len(group),
            "usage": len(group) / total,
            "velo": _mean(group, "release_speed"),
            "velo_max": _quantile(group, "release_speed", 0.95),
            "spin": _mean(group, "release_spin_rate"),
            "h_break": _mean(group, "pfx_x") * FEET_TO_INCHES,
            "v_break": _mean(group, "pfx_z") * FEET_TO_INCHES,
            "extension": _mean(group, "release_extension"),
            "zone_rate": float(group["is_in_zone"].eq(True).mean()),
            "swing_rate": swings / len(group),
            "whiff_rate": whiffs / swings if swings else float("nan"),
            "chase_rate": chases / len(out_of_zone) if len(out_of_zone) else float("nan"),
            "xwoba": _mean(group, "estimated_woba_using_speedangle"),
            "run_value": _mean(group, "delta_run_exp"),
        })

    table = pd.DataFrame(records).set_index("pitch_type")
    return table.sort_values("usage", ascending=False)[columns]


def release_consistency(
    frame: pd.DataFrame,
    min_pitches: int | None = None,
) -> pd.DataFrame:
    """How tightly each pitch type is released, and where relative to the rest.

    Returns a DataFrame indexed by pitch_type, sorted by usage descending:

      n              int, pitches thrown
      rel_x          float, mean horizontal release point (inches)
      rel_z          float, mean vertical release point (inches)
      rel_x_sd       float, standard deviation of horizontal release
      rel_z_sd       float, standard deviation of vertical release
      arm_angle      float, mean arm slot (degrees), when available
      arm_angle_sd   float, its standard deviation
      dx_from_ref    float, horizontal offset from the reference pitch (inches)
      dz_from_ref    float, vertical offset from the reference pitch (inches)
      dist_from_ref  float, straight-line offset from the reference (inches)

    The reference is the pitcher's most-used pitch, since that is the look a
    hitter calibrates to. `dist_from_ref` is the tipping signal: an offering
    released an inch or more away from the fastball slot is identifiable before
    the ball has travelled far enough for movement to matter.

    Standard deviations matter separately. A pitch with a consistent but
    distinct slot is a tell; a pitch with an erratic slot is a command problem.
    """
    columns = [
        "n", "rel_x", "rel_z", "rel_x_sd", "rel_z_sd",
        "arm_angle", "arm_angle_sd", "dx_from_ref", "dz_from_ref", "dist_from_ref",
    ]

    if len(frame) == 0:
        return pd.DataFrame(columns=columns)

    keep = counts.primary_arsenal(frame, min_share=None, min_pitches=min_pitches)
    if not keep:
        return pd.DataFrame(columns=columns)

    subset = frame[frame["pitch_type"].isin(keep)]
    has_arm_angle = "arm_angle" in subset.columns

    records = []
    for pitch_type, group in subset.groupby("pitch_type"):
        records.append({
            "pitch_type": pitch_type,
            "n": len(group),
            "rel_x": _mean(group, "release_pos_x") * FEET_TO_INCHES,
            "rel_z": _mean(group, "release_pos_z") * FEET_TO_INCHES,
            "rel_x_sd": _std(group, "release_pos_x") * FEET_TO_INCHES,
            "rel_z_sd": _std(group, "release_pos_z") * FEET_TO_INCHES,
            "arm_angle": _mean(group, "arm_angle") if has_arm_angle else float("nan"),
            "arm_angle_sd": _std(group, "arm_angle") if has_arm_angle else float("nan"),
        })

    table = pd.DataFrame(records).set_index("pitch_type")
    table = table.sort_values("n", ascending=False)

    reference = table.index[0]
    ref_x = table.loc[reference, "rel_x"]
    ref_z = table.loc[reference, "rel_z"]

    table["dx_from_ref"] = table["rel_x"] - ref_x
    table["dz_from_ref"] = table["rel_z"] - ref_z
    table["dist_from_ref"] = (
        table["dx_from_ref"] ** 2 + table["dz_from_ref"] ** 2
    ) ** 0.5

    return table[columns]


def _mean(group: pd.DataFrame, column: str) -> float:
    if column not in group.columns:
        return float("nan")
    return float(pd.to_numeric(group[column], errors="coerce").mean())


def _std(group: pd.DataFrame, column: str) -> float:
    if column not in group.columns:
        return float("nan")
    return float(pd.to_numeric(group[column], errors="coerce").std())


def _quantile(group: pd.DataFrame, column: str, q: float) -> float:
    if column not in group.columns:
        return float("nan")
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    return float(values.quantile(q)) if len(values) else float("nan")
