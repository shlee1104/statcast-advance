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
      arm_run      float, the same movement toward his ARM side, positive,
                   as scout_arsenal reports it. This is the one a report
                   should print: `h_break` flips sign with pitcher hand, and
                   the page once showed a right-hander's four-seam as +7.0 in
                   one table and -7.0 in the next.
      v_break      float, vertical movement (inches, positive is "rise")
      extension    float, release extension toward the plate (feet)
      zone_rate    float, share thrown inside the strike zone
      swing_rate   float, share the batter offered at
      whiff_rate   float, whiffs per SWING (not per pitch)
      chase_rate   float, swings per pitch OUTSIDE the zone
      xwoba        float, xwOBA per plate appearance ending on this pitch
                   (strikeouts count as zero; see events.season_line)
      xwoba_n      int, plate appearances that ended on this pitch
      xwoba_se     float, standard error of `xwoba`
      run_value    float, average change in run expectancy

    Only pitch types clearing the arsenal gate are included, so a handful of
    misclassified pitches do not appear as a phantom sixth offering.

    `xwoba_n` is much smaller than `n`: only the last pitch of a plate
    appearance carries an xwOBA, about one pitch in four. A pitch thrown 200
    times has an xwOBA resting on roughly 50 plate appearances, and its
    standard error says so.

    velo_max is the 95th percentile rather than the maximum, because a single
    mis-tracked reading would otherwise define the top of a pitcher's range.
    """
    columns = [
        "n", "usage", "velo", "velo_max", "spin", "h_break", "arm_run", "v_break",
        "extension", "zone_rate", "swing_rate", "whiff_rate", "chase_rate",
        "xwoba", "xwoba_n", "xwoba_se", "run_value",
    ]

    if len(frame) == 0:
        return pd.DataFrame(columns=columns)

    keep = counts.primary_arsenal(frame, min_share=None, min_pitches=min_pitches)
    if not keep:
        return pd.DataFrame(columns=columns)

    subset = frame[frame["pitch_type"].isin(keep)]
    total = len(subset)
    arm_sign = _arm_sign(subset)

    records = []
    for pitch_type, group in subset.groupby("pitch_type"):
        swings = int(group["is_swing"].eq(True).sum())
        whiffs = int(group["is_whiff"].eq(True).sum())
        xwoba = _values(group, "estimated_woba_using_speedangle")

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
            "arm_run": _mean(group, "pfx_x") * FEET_TO_INCHES * arm_sign,
            "v_break": _mean(group, "pfx_z") * FEET_TO_INCHES,
            "extension": _mean(group, "release_extension"),
            "zone_rate": float(group["is_in_zone"].eq(True).mean()),
            "swing_rate": swings / len(group),
            "whiff_rate": whiffs / swings if swings else float("nan"),
            "chase_rate": chases / len(out_of_zone) if len(out_of_zone) else float("nan"),
            "xwoba": float(xwoba.mean()) if len(xwoba) else float("nan"),
            "xwoba_n": int(len(xwoba)),
            "xwoba_se": (float(xwoba.std(ddof=1) / len(xwoba) ** 0.5)
                         if len(xwoba) > 1 else float("nan")),
            "run_value": _mean(group, "delta_run_exp"),
        })

    table = pd.DataFrame(records).set_index("pitch_type")
    return table.sort_values("usage", ascending=False)[columns]


def scout_arsenal(frame: pd.DataFrame, min_pitches: int | None = None) -> pd.DataFrame:
    """The arsenal table in the form a scouting report prints it.

    Returns a DataFrame indexed by pitch_type, sorted by usage descending:

      n            int, pitches thrown
      usage        float, share of all pitches
      usage_L      float, share of pitches to left-handed hitters
      usage_R      float, share of pitches to right-handed hitters
      sits_lo      float, 10th percentile velocity
      sits_hi      float, 90th percentile velocity
      touches      float, 99th percentile velocity
      ivb          float, induced vertical break, inches (positive is "ride")
      arm_run      float, horizontal break toward his arm side, inches
      spin         float, average spin rate (rpm)
      whiff_rate   float, whiffs per swing
      chase_rate   float, swings per pitch outside the zone
      xwoba        float, xwOBA per plate appearance ending on this pitch
      swings       int, swings the whiff rate is a share of
      chances      int, pitches outside the zone the chase rate is a share of

    Three choices here follow how scouts actually read a pitch, not how the
    data arrives.

    Velocity is a range — "sits 94-97, touches 98" — because an average hides
    the thing a hitter needs, which is how hard the hardest one will be. The
    range is the 10th to 90th percentile, and "touches" is the 99th rather than
    the maximum, so one mis-tracked reading cannot set the top of his range.

    Usage is split by batter hand inside the same table, because that is the
    first thing a hitter checks: a pitcher who never throws his slider to
    lefties is a four-pitch pitcher to a lefty, and the overall usage column
    hides that.

    Horizontal break is expressed toward the pitcher's ARM side, positive,
    matching `location.py`. Raw `pfx_x` is from the catcher's view and flips
    sign with pitcher hand, so the same sinker would read negative for a
    right-hander and positive for a left-hander.
    """
    columns = [
        "n", "usage", "usage_L", "usage_R", "sits_lo", "sits_hi", "touches",
        "ivb", "arm_run", "spin", "whiff_rate", "chase_rate", "xwoba",
        "swings", "chances",
    ]

    if len(frame) == 0:
        return pd.DataFrame(columns=columns)

    keep = counts.primary_arsenal(frame, min_share=None, min_pitches=min_pitches)
    if not keep:
        return pd.DataFrame(columns=columns)

    subset = frame[frame["pitch_type"].isin(keep)]
    total = len(subset)
    side_totals = subset["stand"].value_counts() if "stand" in subset else pd.Series()

    arm_sign = _arm_sign(subset)

    records = []
    for pitch_type, group in subset.groupby("pitch_type"):
        velo = pd.to_numeric(group["release_speed"], errors="coerce").dropna()
        swings = int(group["is_swing"].eq(True).sum())
        whiffs = int(group["is_whiff"].eq(True).sum())
        out_of_zone = group[group["zone"].notna() & ~group["is_in_zone"].eq(True)]
        chases = int(out_of_zone["is_swing"].eq(True).sum())

        def side_usage(side: str) -> float:
            side_n = int(side_totals.get(side, 0))
            if not side_n:
                return float("nan")
            return float((group["stand"] == side).sum() / side_n)

        records.append({
            "pitch_type": pitch_type,
            "n": len(group),
            "usage": len(group) / total,
            "usage_L": side_usage("L"),
            "usage_R": side_usage("R"),
            "sits_lo": float(velo.quantile(0.10)) if len(velo) else float("nan"),
            "sits_hi": float(velo.quantile(0.90)) if len(velo) else float("nan"),
            "touches": float(velo.quantile(0.99)) if len(velo) else float("nan"),
            "ivb": _mean(group, "pfx_z") * FEET_TO_INCHES,
            "arm_run": _mean(group, "pfx_x") * FEET_TO_INCHES * arm_sign,
            "spin": _mean(group, "release_spin_rate"),
            "whiff_rate": whiffs / swings if swings else float("nan"),
            "chase_rate": chases / len(out_of_zone) if len(out_of_zone) else float("nan"),
            "xwoba": _mean(group, "estimated_woba_using_speedangle"),
            "swings": swings,
            "chances": len(out_of_zone),
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


def _arm_sign(frame: pd.DataFrame) -> float:
    """Multiplier that turns catcher's-view `pfx_x` into arm-side run.

    A right-hander's arm side is negative pfx_x from the catcher's view; a
    left-hander's is positive. One place, so every table agrees.
    """
    if "p_throws" not in frame or not frame["p_throws"].notna().any():
        return -1.0
    return -1.0 if str(frame["p_throws"].mode().iloc[0]) == "R" else 1.0


def _values(group: pd.DataFrame, column: str) -> pd.Series:
    if column not in group.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(group[column], errors="coerce").dropna()


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
