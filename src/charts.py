"""Geometry for the report's hand-drawn SVG charts.

The report draws its key charts as inline SVG from the template, not through a
charting library, for the same reason the dugout card is server-rendered: the
page has to print, and it has to work with scripts blocked. That leaves layout
— where each point, line and label goes — to be computed somewhere, and it is
computed here, so the template only draws and the arithmetic can be tested.

Nothing here decides what a number means. Every value arrives already computed
by the metrics layer; this module only turns values into coordinates and keeps
labels from landing on top of each other, which is the one thing a chart drawn
by hand gets wrong most often.
"""

from __future__ import annotations

import math

# One colour per pitch type, used everywhere a pitch is drawn as a series, so a
# reader who learns that the splitter is orange in one chart can rely on it in
# the next. Chosen to stay distinguishable for the common colour-vision
# deficiencies: no pitch relies on a red/green contrast with another.
PITCH_COLORS: dict[str, str] = {
    "FF": "#2f6fb2",  # four-seam: blue
    "SI": "#7a6aa8",  # sinker: purple
    "FC": "#8a3b3f",  # cutter: brick
    "FS": "#c8703a",  # splitter: orange
    "CH": "#3f8f5a",  # changeup: green
    "SC": "#3f8f5a",
    "FO": "#c8703a",
    "CU": "#2a8a8f",  # curveball: teal
    "KC": "#2a8a8f",
    "SL": "#8a7355",  # slider: brown
    "ST": "#b05a8a",  # sweeper: magenta
    "SV": "#b05a8a",
}
DEFAULT_COLOR = "#6b7280"


def color(pitch_type: str) -> str:
    return PITCH_COLORS.get(pitch_type, DEFAULT_COLOR)


def spread_labels(
    positions: list[float],
    min_gap: float,
    lo: float,
    hi: float,
) -> list[float]:
    """Nudge label positions apart so none sit closer than `min_gap`.

    Keeps each label as close to its true position as possible while
    guaranteeing the gap, and keeps the whole set inside [lo, hi]. Returns the
    adjusted positions in the input order.

    A slope chart puts a label at each end of each line, and two pitches thrown
    at similar rates — a sinker at 1% and a slider at 0% — would otherwise print
    one label on top of the other. The lines still end at the true values; only
    the text moves.
    """
    if not positions:
        return []

    order = sorted(range(len(positions)), key=lambda i: positions[i])
    placed = [positions[i] for i in order]

    # Push down the stack, then up from the bottom edge, then re-clamp at the
    # top. Two passes are enough for the handful of labels a chart carries.
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + min_gap)
    if placed[-1] > hi:
        placed[-1] = hi
        for i in range(len(placed) - 2, -1, -1):
            placed[i] = min(placed[i], placed[i + 1] - min_gap)
    if placed[0] < lo:
        placed[0] = lo
        for i in range(1, len(placed)):
            placed[i] = max(placed[i], placed[i - 1] + min_gap)

    result = [0.0] * len(positions)
    for slot, index in enumerate(order):
        result[index] = placed[slot]
    return result


# ---------------------------------------------------------------------------
# Platoon slope chart
# ---------------------------------------------------------------------------

SLOPE = {"width": 560, "height": 360, "top": 24, "bottom": 40,
         "x_left": 175, "x_right": 385}


def slope_layout(rows: list[dict]) -> dict:
    """Coordinates for a two-point slope chart of usage vs lefties and righties.

    `rows` carry pitch_type, usage_L, usage_R. Returns the canvas size, the
    y-axis ticks, and per pitch the two end points and the two label positions.

    The y-axis starts at zero and tops out at the next 10% above the largest
    share, with a floor of 40%, so that two pitchers with ordinary mixes are
    drawn on the same scale and a steep line means the same thing on both.
    """
    s = SLOPE
    top, bottom = s["top"], s["height"] - s["bottom"]
    peak = max([r["usage_L"] or 0 for r in rows] + [r["usage_R"] or 0 for r in rows] + [0])
    y_max = max(0.40, math.ceil(peak * 10 + 1e-9) / 10)

    def y(v: float) -> float:
        return bottom - (v / y_max) * (bottom - top)

    left = spread_labels([y(r["usage_L"] or 0) for r in rows], 16, top, bottom)
    right = spread_labels([y(r["usage_R"] or 0) for r in rows], 16, top, bottom)

    pitches = []
    for row, ly, ry in zip(rows, left, right):
        lv, rv = row["usage_L"] or 0, row["usage_R"] or 0
        gap = lv - rv
        # A pitch one side barely sees is drawn strong even when the absolute
        # gap is small: 0% to 5% is a 5-point gap and also the whole story for
        # that pitch. Matches the rule the chart's headline uses, so the line
        # the title talks about is never the faded one.
        unseen = min(lv, rv) < 0.03 <= max(lv, rv)
        pitches.append({
            "pitch": row["pitch_type"],
            "color": color(row["pitch_type"]),
            "usage_L": row["usage_L"],
            "usage_R": row["usage_R"],
            "y_L": round(y(row["usage_L"] or 0), 1),
            "y_R": round(y(row["usage_R"] or 0), 1),
            "label_y_L": round(ly, 1),
            "label_y_R": round(ry, 1),
            # Lines that barely move are drawn quieter, so the eye goes to the
            # pitches whose usage actually depends on the hitter's side.
            "strong": abs(gap) >= 0.05 or unseen,
        })

    ticks = [
        {"value": v / 10, "y": round(y(v / 10), 1)}
        for v in range(0, int(round(y_max * 10)) + 1)
    ]
    return {**s, "plot_top": top, "plot_bottom": bottom,
            "y_max": y_max, "ticks": ticks, "pitches": pitches}


# ---------------------------------------------------------------------------
# Movement plot
# ---------------------------------------------------------------------------

MOVE = {"width": 560, "height": 470, "left": 52, "right": 16, "top": 16,
        "bottom": 46, "x_min": -22.0, "x_max": 22.0, "y_min": -22.0, "y_max": 24.0}

# Whiff-rate colour ramp: pale to deep orange, one hue, so it reads as "more"
# rather than as a category, and it survives greyscale printing as light/dark.
WHIFF_LO, WHIFF_HI = 0.10, 0.45
_PALE, _DEEP = (247, 232, 214), (166, 64, 28)


def whiff_color(rate: float | None) -> str:
    if rate is None or rate != rate:
        return "#d8dde3"
    t = min(max((rate - WHIFF_LO) / (WHIFF_HI - WHIFF_LO), 0.0), 1.0)
    rgb = [round(a + (b - a) * t) for a, b in zip(_PALE, _DEEP)]
    return "rgb({},{},{})".format(*rgb)


def _overlaps(a: tuple, b: tuple) -> bool:
    """Axis-aligned rectangles as (x0, y0, x1, y1)."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def movement_layout(rows: list[dict], league: list[dict] | None = None) -> dict:
    """Coordinates for a movement plot: arm-side run across, ride up.

    `rows` carry pitch_type, arm_run, ivb, usage, whiff_rate, velo, name.
    `league` optionally carries pitch_type, arm_run, ivb for the league-average
    version of each pitch, drawn as a hollow marker behind his.

    The axes are fixed rather than fitted to the pitcher, so his splitter and
    another pitcher's splitter land in comparable places and the plot can be
    read against memory of other reports.

    Labels go below each bubble when there is room, and otherwise try above,
    right, then left, checking against every bubble and every label already
    placed. Bubbles are placed largest first so the most-used pitches get first
    choice of position.
    """
    m = MOVE
    x0, x1 = m["left"], m["width"] - m["right"]
    y0, y1 = m["top"], m["height"] - m["bottom"]

    def px(v: float) -> float:
        v = min(max(v, m["x_min"]), m["x_max"])
        return x0 + (v - m["x_min"]) / (m["x_max"] - m["x_min"]) * (x1 - x0)

    def py(v: float) -> float:
        v = min(max(v, m["y_min"]), m["y_max"])
        return y1 - (v - m["y_min"]) / (m["y_max"] - m["y_min"]) * (y1 - y0)

    bubbles = []
    for r in rows:
        bubbles.append({
            "pitch": r["pitch_type"],
            "name": r["name"],
            "cx": round(px(r["arm_run"]), 1),
            "cy": round(py(r["ivb"]), 1),
            # Area, not radius, proportional to usage, so a pitch thrown twice
            # as often does not look four times as big.
            "r": round(5 + 26 * math.sqrt(max(r["usage"] or 0, 0)), 1),
            "fill": whiff_color(r.get("whiff_rate")),
            "stroke": color(r["pitch_type"]),
            "usage": r["usage"],
            "velo": r.get("velo"),
            "whiff_rate": r.get("whiff_rate"),
        })

    obstacles = [(b["cx"] - b["r"], b["cy"] - b["r"], b["cx"] + b["r"], b["cy"] + b["r"])
                 for b in bubbles]
    placed: list[tuple] = []

    for b in sorted(bubbles, key=lambda b: -(b["usage"] or 0)):
        width = max(len(b["name"]) * 7.0, 78.0)
        height = 28.0
        candidates = {
            "below": (b["cx"] - width / 2, b["cy"] + b["r"] + 4),
            "above": (b["cx"] - width / 2, b["cy"] - b["r"] - 4 - height),
            "right": (b["cx"] + b["r"] + 6, b["cy"] - height / 2),
            "left": (b["cx"] - b["r"] - 6 - width, b["cy"] - height / 2),
        }
        own = (b["cx"] - b["r"], b["cy"] - b["r"], b["cx"] + b["r"], b["cy"] + b["r"])
        chosen = "below"
        for where, (lx, ly) in candidates.items():
            rect = (lx, ly, lx + width, ly + height)
            inside = lx >= x0 and lx + width <= x1 and ly >= y0 and ly + height <= y1
            clear = not any(_overlaps(rect, o) for o in obstacles if o != own) and \
                not any(_overlaps(rect, p) for p in placed)
            if inside and clear:
                chosen = where
                break
        lx, ly = candidates[chosen]
        placed.append((lx, ly, lx + width, ly + height))
        b["label_x"] = round(lx + width / 2, 1)
        b["label_y"] = round(ly + 12, 1)
        b["label_pos"] = chosen

    hollow = []
    for lg in league or []:
        match = next((b for b in bubbles if b["pitch"] == lg["pitch_type"]), None)
        if match is None or lg.get("arm_run") is None or lg.get("ivb") is None:
            continue
        hollow.append({
            "pitch": lg["pitch_type"],
            "cx": round(px(lg["arm_run"]), 1),
            "cy": round(py(lg["ivb"]), 1),
            "to_x": match["cx"],
            "to_y": match["cy"],
            "stroke": color(lg["pitch_type"]),
        })

    x_ticks = [{"value": v, "x": round(px(v), 1)} for v in range(-20, 21, 5)]
    y_ticks = [{"value": v, "y": round(py(v), 1)} for v in range(-20, 21, 5)]
    return {**m, "plot": {"x0": x0, "x1": x1, "y0": y0, "y1": y1},
            "zero_x": round(px(0), 1), "zero_y": round(py(0), 1),
            "x_ticks": x_ticks, "y_ticks": y_ticks,
            "bubbles": bubbles, "league": hollow,
            "whiff_lo": WHIFF_LO, "whiff_hi": WHIFF_HI,
            "whiff_ramp": [whiff_color(WHIFF_LO), whiff_color((WHIFF_LO + WHIFF_HI) / 2),
                           whiff_color(WHIFF_HI)]}
