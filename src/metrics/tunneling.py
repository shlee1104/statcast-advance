"""Pitch tunneling: where two pitches are when the hitter has to decide.

A hitter commits to a swing well before the ball arrives. Two pitches that are
still close together at that moment cannot be told apart when it matters, and
whatever separates them afterwards is deception. This module measures both:
how far apart a pitch and the one that followed it were at the decision
point, and how far apart they finished.

**The trajectory.** Savant publishes a constant-acceleration fit for every
pitch: velocity (`vx0`, `vy0`, `vz0`) and acceleration (`ax`, `ay`, `az`) at
the plane 50 feet from the back of the plate, where y runs from the plate
toward the mound. The position on that plane is not published, so it is worked
back from the release point, which is at y = `release_pos_y`, about 54 feet
and so *behind* the plane, at a negative time. From there the ball's position
at any distance is closed-form.

The check that this is right is in the data: run forward to the front of the
plate, the fit has to land on Savant's own `plate_x` and `plate_z`. On the two
committed fixtures, 6,400 pitches, it does to within 0.07 of an inch — the
rounding of the published values. Treating `release_pos_x` and `release_pos_z`
as positions on the 50-foot plane, the other reading of those columns, misses
by a median of two inches.

**The decision point** is a convention, and it is Baseball Prospectus's: 23.8
feet from the plate, where an average fastball is about 175 ms from contact.
It is a distance, and a hitter's deadline is a time, so it falls at a
different moment for each pitch: 172 ms before the plate for Yamamoto's
four-seam and 215 ms for his curveball. Comparing both pitches at one instant
instead ranks pairs almost identically (Spearman 0.96 over 51 pair types on
the fixtures) and puts curveball pairs 3 to 4 inches further apart, so the
standard distance is kept and understates how readable a slow curveball is.
`tunneling.tunnel_point_ft` in config.yaml moves it.

**What is measured for a pair**, all in the plane the hitter looks into (x
across, z up), in inches:

  release    how far apart the two left his hand
  tunnel     how far apart they were at the decision point
  plate      how far apart they crossed the front of the plate
  late       how differently they moved after the decision point: for each
             pitch, where it finished against where it was headed in a
             straight line from that point, and then the distance between
             those two for the pair. Two pitches can be far apart at the
             plate because they were already aimed at different spots, which
             a hitter can see; this is the part he cannot.

and the difference in flight time, because two pitches on one path are still
separable if one arrives 80 ms after the other.

**Two ways to average a pair of pitch types.** The published way is the mean
of those distances, and it needs care: two consecutive fastballs are 8 inches
apart at the decision point, because he does not throw to one spot, so a
mean distance is mostly that scatter. The mean distance from a four-seam to
the splitter after it is 9.2 inches for Yamamoto against lefties, barely
above the 8.1 for two four-seams.

The other way is the offset: average the *difference* first, across and up
separately, and then take its length. Scatter cancels and what is left is
where the follow-up sits relative to the pitch before it. For the same pair
that is 4.4 inches, nearly all of it down. The scatter then gets its own
number, and the offset divided by it says whether a hitter could tell the two
apart from where the ball is: `tunnel_d`, a discriminability index. At 1.0,
position alone at the decision point picks the right pitch about 69% of the
time; at 0 it is a coin flip.

What this is not: a claim that a pair which tunnels well gets hitters out. It
describes what the hitter is shown. Arm speed, grip and what the hitter
remembers from earlier at-bats are not in pitch tracking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.metrics import arsenal, counts, sequencing

FEET_TO_INCHES = 12.0

# The plane Savant's velocity and acceleration terms are reported on, and the
# front edge of home plate, where `plate_x` and `plate_z` are measured. Both
# in feet from the back point of the plate.
REFERENCE_Y_FT = 50.0
PLATE_FRONT_Y_FT = 17.0 / 12.0

# What the reconstruction reads. `release_pos_y` and the six fit terms are the
# columns the cache did not store before this module existed.
TRAJECTORY_COLUMNS: list[str] = [
    "release_pos_x", "release_pos_y", "release_pos_z",
    "vx0", "vy0", "vz0", "ax", "ay", "az",
]

FLIGHT_COLUMNS: list[str] = [
    "release_x", "release_z", "tunnel_x", "tunnel_z", "plate_x_fit",
    "plate_z_fit", "late_x", "late_z", "flight_time", "decision_time",
]


def tunnel_point_ft() -> float:
    """How far from the plate the decision point is taken to be."""
    return float(config.get("tunneling.tunnel_point_ft", 23.8))


def has_trajectory(frame: pd.DataFrame) -> bool:
    """Whether the frame carries the fit, with at least one pitch that has it.

    A cache row written before the columns were added has them as nulls, so
    their presence alone is not enough.
    """
    if not set(TRAJECTORY_COLUMNS) <= set(frame.columns):
        return False
    return bool(frame[TRAJECTORY_COLUMNS].notna().all(axis=1).any())


def _fit(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[TRAJECTORY_COLUMNS].apply(pd.to_numeric, errors="coerce")


def time_at(frame: pd.DataFrame, y_ft: float | pd.Series) -> pd.Series:
    """When each pitch crosses the plane `y_ft` feet from the plate, in seconds.

    Measured from the moment it crosses the 50-foot plane, so a plane behind
    that one — the release point — is at a negative time. Solves
    y(t) = 50 + vy0·t + ½·ay·t² for the root that runs forward toward the
    plate. `vy0` is negative (toward the plate) and `ay` positive (drag).
    """
    fit = _fit(frame)
    distance = REFERENCE_Y_FT - y_ft
    root = np.sqrt(fit["vy0"] ** 2 - 2.0 * fit["ay"] * distance)
    return (-fit["vy0"] - root) / fit["ay"]


def position_at(frame: pd.DataFrame, y_ft: float) -> pd.DataFrame:
    """Where each pitch is as it crosses the plane `y_ft` feet from the plate.

    Returns x and z in feet, in Savant's frame (x positive toward the
    catcher's right), and `t`, seconds since release. A pitch missing any
    term of its fit gets nulls.
    """
    fit = _fit(frame)
    t_release = time_at(frame, fit["release_pos_y"])
    t = time_at(frame, y_ft)

    # The position on the 50-foot plane, worked back from the release point:
    # release is at a negative time, so this steps forward from it.
    x0 = fit["release_pos_x"] - fit["vx0"] * t_release - 0.5 * fit["ax"] * t_release ** 2
    z0 = fit["release_pos_z"] - fit["vz0"] * t_release - 0.5 * fit["az"] * t_release ** 2

    return pd.DataFrame({
        "x": x0 + fit["vx0"] * t + 0.5 * fit["ax"] * t ** 2,
        "z": z0 + fit["vz0"] * t + 0.5 * fit["az"] * t ** 2,
        "t": t - t_release,
    }, index=frame.index)


def flight(frame: pd.DataFrame, tunnel_ft: float | None = None) -> pd.DataFrame:
    """Everything a pair comparison reads, for each pitch.

    Indexed like `frame`, with FLIGHT_COLUMNS: positions in feet at release,
    at the decision point and at the front of the plate; `late_x` / `late_z`,
    how far the pitch finished from where it was headed in a straight line at
    the decision point; `flight_time`, release to plate, and `decision_time`,
    decision point to plate, both in seconds.

    Under constant acceleration the late movement is ½·a·Δt² with Δt the
    decision time: everything the pitch does after the decision point that
    its direction at that point did not already show. Gravity is part of it.
    A slower pitch is in the air longer after the point and drops further,
    and the hitter does not see that coming either.
    """
    if tunnel_ft is None:
        tunnel_ft = tunnel_point_ft()
    if not set(TRAJECTORY_COLUMNS) <= set(frame.columns):
        return pd.DataFrame(columns=FLIGHT_COLUMNS, index=frame.index, dtype=float)

    fit = _fit(frame)
    tunnel = position_at(frame, tunnel_ft)
    plate = position_at(frame, PLATE_FRONT_Y_FT)
    decision_time = plate["t"] - tunnel["t"]

    return pd.DataFrame({
        "release_x": fit["release_pos_x"],
        "release_z": fit["release_pos_z"],
        "tunnel_x": tunnel["x"],
        "tunnel_z": tunnel["z"],
        "plate_x_fit": plate["x"],
        "plate_z_fit": plate["z"],
        "late_x": 0.5 * fit["ax"] * decision_time ** 2,
        "late_z": 0.5 * fit["az"] * decision_time ** 2,
        "flight_time": plate["t"],
        "decision_time": decision_time,
    }, index=frame.index)[FLIGHT_COLUMNS]


# Where the signed differences are taken: the follow-up minus the pitch before
# it, in inches, `dx` toward his arm side and `dz` up.
PAIR_POINTS: dict[str, tuple[str, str]] = {
    "release": ("release_x", "release_z"),
    "tunnel": ("tunnel_x", "tunnel_z"),
    "plate": ("plate_x_fit", "plate_z_fit"),
    "late": ("late_x", "late_z"),
}

PAIR_COLUMNS: list[str] = [
    "stand", "pitch_type", "next_pitch", "release_diff", "tunnel_diff",
    "plate_diff", "late_diff", "flight_diff_ms",
] + [f"{point}_{axis}" for point in PAIR_POINTS for axis in ("dx", "dz")]


def _apart(pairs: pd.DataFrame, x: str, z: str) -> pd.Series:
    """Inches between a pitch and its follow-up in the (x, z) plane."""
    return np.hypot(pairs[f"next_{x}"] - pairs[x],
                    pairs[f"next_{z}"] - pairs[z]) * FEET_TO_INCHES


def pairs(frame: pd.DataFrame, tunnel_ft: float | None = None) -> pd.DataFrame:
    """Each pitch and the one that followed it in the same plate appearance.

    One row per pair, with PAIR_COLUMNS: the batter side, the two pitch
    types, the four separations in inches (see the module docstring),
    `flight_diff_ms`, how much longer the follow-up was in the air, and for
    each of the four points the signed difference behind the separation:
    `<point>_dx` toward his arm side and `<point>_dz` up, follow-up minus the
    pitch before it.

    Pitches are paired before strays are dropped, as in
    `sequencing.transitions`: removing a mislabelled pitch first would pair
    the pitches either side of it, which were not consecutive. A pair is kept
    when both pitches are in his arsenal and both have a trajectory.
    """
    if len(frame) == 0 or not has_trajectory(frame):
        return pd.DataFrame(columns=PAIR_COLUMNS)

    kept = counts.primary_arsenal(frame)
    working = frame.join(flight(frame, tunnel_ft))
    paired = sequencing._pair_pitches(working, extra=FLIGHT_COLUMNS)
    paired = paired[paired["pitch_type"].isin(kept)
                    & paired["next_pitch_type"].isin(kept)]
    paired = paired.dropna(subset=FLIGHT_COLUMNS + [f"next_{c}" for c in FLIGHT_COLUMNS])

    out = {
        "stand": paired["stand"],
        "pitch_type": paired["pitch_type"],
        "next_pitch": paired["next_pitch_type"],
        "flight_diff_ms": (paired["next_flight_time"] - paired["flight_time"]) * 1000.0,
    }
    # Savant's x runs toward the catcher's right; a report says arm side.
    arm_side = arsenal._arm_sign(frame)
    for point, (x, z) in PAIR_POINTS.items():
        out[f"{point}_diff"] = _apart(paired, x, z)
        out[f"{point}_dx"] = (paired[f"next_{x}"] - paired[x]) * FEET_TO_INCHES * arm_side
        out[f"{point}_dz"] = (paired[f"next_{z}"] - paired[z]) * FEET_TO_INCHES
    return pd.DataFrame(out)[PAIR_COLUMNS].reset_index(drop=True)


TABLE_COLUMNS: list[str] = [
    "stand", "pitch_type", "next_pitch", "n", "release", "tunnel", "plate",
    "late", "flight_ms", "floor", "floor_n", "tunnel_vs_floor", "tunnel_z",
    "release_offset", "tunnel_offset", "tunnel_dx", "tunnel_dz",
    "late_offset", "late_dx", "late_dz", "plate_offset", "plate_dx", "plate_dz",
    "spread", "tunnel_d", "reading",
]

# What a pair's d' is called. Between the two bars it is called nothing.
SAME_WINDOW, APART = "same window", "apart early"


def min_pairs() -> int:
    """How many times a pair has to occur to one side before it is averaged."""
    return int(config.get("tunneling.min_pairs", 30))


def reading(d: float) -> str:
    """A pair's offset at the decision point, over his scatter, in words.

    "same window" at or under `tunneling.same_window_d`, "apart early" at or
    over `tunneling.apart_d`, and an empty string between them or when there
    is no number. Three words would claim a precision the number does not
    have: between halves of a season it moves by a median of 0.2.
    """
    if d != d:
        return ""
    if d <= float(config.get("tunneling.same_window_d", 0.75)):
        return SAME_WINDOW
    if d >= float(config.get("tunneling.apart_d", 1.25)):
        return APART
    return ""


def _offset(group: pd.DataFrame, point: str) -> float:
    """Length of the average difference: scatter cancels, the offset is left."""
    return float(np.hypot(group[f"{point}_dx"].mean(), group[f"{point}_dz"].mean()))


def pair_table(
    frame: pd.DataFrame,
    min_n: int | None = None,
    tunnel_ft: float | None = None,
) -> pd.DataFrame:
    """Each pair of pitch types he throws back to back, to each batter side.

    One row per (side, pitch, follow-up) that occurred at least `min_n`
    times, most frequent first within a side, with TABLE_COLUMNS.

    The mean distances, as published tunneling metrics are defined:

      n                pairs averaged
      release, tunnel, plate, late
                       mean separation in inches at each point (see `pairs`)
      flight_ms        how much longer the follow-up is in the air, on average
      floor, floor_n   the mean tunnel separation of the SAME pitch thrown
                       twice in a row to this side, over every pitch type, and
                       how many such pairs that is
      tunnel_vs_floor  tunnel minus floor, in inches
      tunnel_z         that gap in standard errors (Welch)

    The offsets, with the scatter taken out (see the module docstring):

      release_offset, tunnel_offset, late_offset, plate_offset
                       length in inches of the average difference at each point
      tunnel_dx, tunnel_dz, late_dx, late_dz, plate_dx, plate_dz
                       that average difference itself, toward his arm side
                       and up
      spread           how far one of his pitches lands from where that pitch
                       type usually is at the decision point, along one axis,
                       in inches: the scatter the offset has to be read against
      tunnel_d         tunnel_offset / spread
      reading          `reading(tunnel_d)`: "same window", "apart early" or ""

    The floor is what makes a mean distance readable. Two consecutive
    fastballs are about eight inches apart at the decision point, because he
    does not throw to one spot. So eight inches is what "the same pitch" looks
    like to a hitter, and a different pitch that comes through nine inches
    from the last one is, at that moment, inside the spread of the pitch he
    just saw. Without the floor every pair looks far apart.

    `plate` divided by `tunnel` is not reported as a ratio. For the same
    pitch thrown twice it is about 1.7, which is geometry: two straight lines
    from one release point are 1.7 times further apart after 53 feet than
    after 30. Most of any pair's ratio is that number.

    `spread` comes from the same-pitch pairs too. The difference between two
    pitches of one type has twice the variance of one pitch, so the spread of
    one is the spread of those differences over root two, pooled across both
    axes and every pitch type to that side.

    That step assumes two consecutive pitches of a type are no more alike
    than two taken at random, which is nearly true and not exactly. Measured
    directly, as each pitch's distance from its type's average, the scatter
    across the 58 cached pitcher-seasons (a one-off run, not a script in the
    repo) came out a median 6% larger (the
    middle 80% of pitcher-sides: 1% to 13%): back-to-back pitches of a type
    land a little nearer each other. So `tunnel_d` reads about 6% high, and a
    d' of 1.0 is nearer 68% than 69%. The test suite holds the two within 15%
    of each other on the fixtures. It is left as it is because the bars it is
    read against are a judgment call a good deal coarser than 6%.

    The offsets carry a bias of their own. The length of an average of noisy
    differences is never zero even when the two pitches share a centre, so a
    small offset is overstated: by a median 0.2 inch on those seasons, which
    would change the word on 14 of 567 pairs. Half-to-half, the same number
    moves 1.3 inches.
    """
    if min_n is None:
        min_n = min_pairs()
    each = pairs(frame, tunnel_ft)
    if len(each) == 0:
        return pd.DataFrame(columns=TABLE_COLUMNS)

    rows = []
    for side, side_pairs in each.groupby("stand"):
        same_pairs = side_pairs[side_pairs["pitch_type"] == side_pairs["next_pitch"]]
        same = same_pairs["tunnel_diff"]
        floor = float(same.mean()) if len(same) else float("nan")
        spread = float(np.sqrt(
            (same_pairs["tunnel_dx"].var(ddof=1) + same_pairs["tunnel_dz"].var(ddof=1)) / 4.0
        )) if len(same_pairs) > 1 else float("nan")

        for (pitch, follow), group in side_pairs.groupby(["pitch_type", "next_pitch"]):
            if len(group) < min_n:
                continue
            tunnel = float(group["tunnel_diff"].mean())
            # Welch: the pair's tunnel separations against the same-pitch
            # ones, without assuming the two spread alike.
            se = float(np.sqrt(group["tunnel_diff"].var(ddof=1) / len(group)
                               + same.var(ddof=1) / len(same))) if len(same) > 1 else float("nan")
            tunnel_offset = _offset(group, "tunnel")
            rows.append({
                "stand": side, "pitch_type": pitch, "next_pitch": follow,
                "n": int(len(group)),
                "release": float(group["release_diff"].mean()),
                "tunnel": tunnel,
                "plate": float(group["plate_diff"].mean()),
                "late": float(group["late_diff"].mean()),
                "flight_ms": float(group["flight_diff_ms"].mean()),
                "floor": floor, "floor_n": int(len(same)),
                "tunnel_vs_floor": tunnel - floor,
                "tunnel_z": (tunnel - floor) / se if se == se and se > 0 else float("nan"),
                "release_offset": _offset(group, "release"),
                "tunnel_offset": tunnel_offset,
                "tunnel_dx": float(group["tunnel_dx"].mean()),
                "tunnel_dz": float(group["tunnel_dz"].mean()),
                "late_offset": _offset(group, "late"),
                "late_dx": float(group["late_dx"].mean()),
                "late_dz": float(group["late_dz"].mean()),
                "plate_offset": _offset(group, "plate"),
                "plate_dx": float(group["plate_dx"].mean()),
                "plate_dz": float(group["plate_dz"].mean()),
                "spread": spread,
                "tunnel_d": tunnel_offset / spread if spread == spread and spread > 0 else float("nan"),
            })
            rows[-1]["reading"] = reading(rows[-1]["tunnel_d"])
    if not rows:
        return pd.DataFrame(columns=TABLE_COLUMNS)
    table = pd.DataFrame(rows)[TABLE_COLUMNS]
    return table.sort_values(["stand", "n"], ascending=[True, False]).reset_index(drop=True)
