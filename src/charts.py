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


# ---------------------------------------------------------------------------
# Sequence graph
# ---------------------------------------------------------------------------

SEQUENCE = {"width": 520, "row": 46, "top": 46, "bottom": 20,
            "x_left": 158, "x_right": 362, "header_y": 16}

# Rows are spaced further apart when there are few of them. A two-pitch
# reliever can have every one of his four lines coloured, and four labels
# need more room than two rows 46 apart leave.
_SEQUENCE_ROW_BY_PITCHES = {1: 78, 2: 78, 3: 66, 4: 54}

# More than usual and less than usual, in the same two hues as the count grid
# and the zone maps: orange more, blue less. A line's colour says which way it
# departs from his usual, never which pitch it is, so the pitch colours stay
# on the dots at either end.
SEQUENCE_MORE, SEQUENCE_LESS, SEQUENCE_USUAL = "#bf5e27", "#2768a8", "#c3cad3"

# A follow-up rarer than this is not drawn when it is within his usual: thirty
# hairlines say nothing a reader can use and bury the ones that do.
SEQUENCE_MIN_SHARE = 0.08

# Where along its line a label may sit, in order of preference. Nearer the
# pitch he just threw first: lines leave a pitch in different directions, so
# labels there are apart from each other, and the label is read next to the
# pitch it is about.
_LABEL_STOPS = (0.30, 0.50, 0.70, 0.40, 0.60, 0.22, 0.78)


def _node_radius(share: float) -> float:
    """Area, not radius, proportional to the share, as on the movement plot."""
    return round(3.5 + 12.0 * math.sqrt(max(share or 0.0, 0.0)), 1)


def _crosses(line: dict, rect: tuple) -> bool:
    """Whether a line passes through a rectangle, checked at points along it."""
    for i in range(25):
        t = i / 24
        x = line["x1"] + (line["x2"] - line["x1"]) * t
        y = line["y1"] + (line["y2"] - line["y1"]) * t
        if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
            return True
    return False


def sequence_layout(pitches: list[str], rows: list[dict], row_gate: int) -> dict:
    """Coordinates for one batter side of the sequence graph.

    Two columns of the same pitches: the pitch he just threw on the left, the
    pitch that followed on the right, so reading left to right is reading in
    time. `pitches` is the order to list them in. `rows` are that side's rows
    of `sequencing.transitions()`, each with a `standout` of "more", "less" or
    "" (see `sequencing.transition_standouts`). A pitch with fewer than
    `row_gate` follow-ups keeps its dot but sends out no lines.

    Returns the canvas size, the dots in each column, and the lines: the ones
    within his usual first, the coloured ones after so they are drawn on top.
    Each coloured line carries a label position.

    Line thickness is the share of follow-ups, on one scale for every line in
    every panel, so a thick line means the same thing wherever it is.

    A label sits just above or just below its line, at whichever of seven
    points along it lands on the fewest labels already placed and the fewest
    other coloured lines. Strongest departures are placed first, so they get
    first choice.
    """
    s = SEQUENCE
    row = _SEQUENCE_ROW_BY_PITCHES.get(len(pitches), s["row"])
    y_of = {p: s["top"] + i * row for i, p in enumerate(pitches)}
    height = s["top"] + max(len(pitches) - 1, 0) * row + s["bottom"]

    by_source: dict[str, dict] = {}
    by_target: dict[str, dict] = {}
    for r in rows:
        by_source.setdefault(r["pitch_type"], r)
        by_target.setdefault(r["next_pitch"], r)
    total = rows[0]["total"] if rows else 0

    left, right = [], []
    for p in pitches:
        source, target = by_source.get(p), by_target.get(p)
        row_n = int(source["row_n"]) if source else 0
        left.append({
            "pitch": p, "y": y_of[p], "color": color(p), "row_n": row_n,
            "read": row_n >= row_gate,
            "r": _node_radius(row_n / total if total else 0.0),
        })
        usual = float(target["usual"]) if target else 0.0
        right.append({
            "pitch": p, "y": y_of[p], "color": color(p), "usual": usual,
            "r": _node_radius(usual),
        })
    radius_left = {n["pitch"]: n["r"] for n in left}
    radius_right = {n["pitch"]: n["r"] for n in right}
    read = {n["pitch"] for n in left if n["read"]}

    usual_lines, standouts = [], []
    for r in rows:
        a, b = r["pitch_type"], r["next_pitch"]
        if a not in read or a not in y_of or b not in y_of:
            continue
        kind = r.get("standout") or ""
        if not kind and (r["share"] < SEQUENCE_MIN_SHARE or r["n"] <= 0):
            continue
        width = 0.8 + 12.0 * r["share"]
        line = {
            "from": a, "to": b, "kind": kind,
            "x1": round(s["x_left"] + radius_left[a] + 2, 1), "y1": y_of[a],
            # A coloured line stops short of the dot to leave room for its
            # arrowhead.
            "x2": round(s["x_right"] - radius_right[b] - (9 if kind else 2), 1),
            "y2": y_of[b],
            "width": round(max(width, 2.2) if kind else width, 1),
            "stroke": {"more": SEQUENCE_MORE, "less": SEQUENCE_LESS}.get(kind, SEQUENCE_USUAL),
            "n": int(r["n"]), "row_n": int(r["row_n"]),
            "share": float(r["share"]), "usual": float(r["usual"]),
            "lift": float(r["lift"]) if r["lift"] == r["lift"] else None,
        }
        (standouts if kind else usual_lines).append(line)

    placed: list[tuple] = []
    lane = (s["x_left"] + 14, s["x_right"] - 14)
    for line in sorted(standouts, key=lambda l: -abs(math.log(max(l["lift"] or 1e-3, 1e-3)))):
        text = f"{line['share']:.0%} · usual {line['usual']:.0%}"
        box_w, box_h = 6.1 * len(text) + 8, 15.0
        others = [o for o in standouts if o is not line]
        best = None
        spots = [(t, -9) for t in _LABEL_STOPS] + [(t, 12) for t in _LABEL_STOPS]
        for rank, (t, offset) in enumerate(spots):
            cx = line["x1"] + (line["x2"] - line["x1"]) * t
            cy = line["y1"] + (line["y2"] - line["y1"]) * t + offset
            cx = min(max(cx, lane[0] + box_w / 2), lane[1] - box_w / 2)
            rect = (cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2)
            cost = (100 * sum(_overlaps(rect, p) for p in placed)
                    + 10 * sum(_crosses(o, rect) for o in others)
                    + rank)
            if best is None or cost < best[0]:
                best = (cost, cx, cy, rect)
        _, cx, cy, rect = best
        placed.append(rect)
        line["label"] = text
        line["label_x"] = round(cx, 1)
        line["label_y"] = round(cy + 4, 1)

    return {
        "width": s["width"], "height": height,
        "x_left": s["x_left"], "x_right": s["x_right"], "header_y": s["header_y"],
        "left": left, "right": right,
        "lines": usual_lines + standouts,
        "standouts": len(standouts),
        "label_boxes": placed,
    }


# ---------------------------------------------------------------------------
# Dugout card: situation pies and zone maps
# ---------------------------------------------------------------------------

PIE = {"size": 120, "r": 56}

# A slice thinner than this cannot hold its own label. It is still drawn, in
# its pitch colour, and the arsenal table beside it is the colour key.
PIE_LABEL_MIN_SHARE = 0.10

# A slice thinner than this gets a smaller label, set nearer the rim where the
# slice is widest. At one radius, the labels of two neighbouring 13% slices
# ran into each other.
PIE_SMALL_SHARE = 0.16

# The share of a pitch going to one zone at which a zone-map cell reaches the
# pitch's full colour. Fixed, never fitted to the map, so the same share is the
# same depth of colour in every map on the card.
ZONE_FULL_SHARE = 0.30

# Past this depth of colour the number in a cell is printed in white.
ZONE_DARK_TEXT_FROM = 0.60


def pie_layout(rows: list[dict]) -> dict:
    """Slices for one pie, clockwise from twelve o'clock in the order given.

    `rows` are dicts with a `pitch` and a `share`; any other keys are passed
    through to the slice. Shares are expected to sum to one. A pitch with no
    share gets no slice.

    Pitches go round in the order given, which the card keeps the same in
    every pie — most used first — so a pitch starts from the same place in
    each and a reader compares the pies by where the colours fall. Sorting
    each pie by its own shares would move the colours around between them.

    A slice holding the whole pie is returned with `full` set and no path: an
    arc from a point back to itself draws nothing.
    """
    size, r = PIE["size"], PIE["r"]
    cx = cy = size / 2
    slices = []
    angle = -math.pi / 2
    for row in rows:
        share = float(row.get("share") or 0.0)
        if share <= 0:
            continue
        start, end = angle, angle + 2 * math.pi * share
        mid = (start + end) / 2
        full = share >= 0.999
        small = share < PIE_SMALL_SHARE
        label_r = 0.0 if full else r * (0.72 if small else 0.60)
        piece = {
            **row,
            "color": color(row["pitch"]),
            "full": full,
            "d": None if full else (
                f"M{cx:.1f},{cy:.1f} "
                f"L{cx + r * math.cos(start):.1f},{cy + r * math.sin(start):.1f} "
                f"A{r},{r} 0 {1 if share > 0.5 else 0} 1 "
                f"{cx + r * math.cos(end):.1f},{cy + r * math.sin(end):.1f} Z"
            ),
            "labelled": share >= PIE_LABEL_MIN_SHARE,
            "small": small,
            "label_x": round(cx + label_r * math.cos(mid), 1),
            "label_y": round(cy + label_r * math.sin(mid), 1),
        }
        slices.append(piece)
        angle = end
    return {"size": size, "cx": cx, "cy": cy, "r": r, "slices": slices}


def tint(hex_color: str, strength: float) -> str:
    """A colour mixed toward white: 0 is white, 1 is the colour itself."""
    t = min(max(strength, 0.0), 1.0)
    channels = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    return "rgb({},{},{})".format(*(round(255 + (c - 255) * t) for c in channels))


def zone_cell(pitch_type: str, share: float) -> dict:
    """How one cell of a card zone map is drawn: its fill and its text colour.

    The fill is the pitch's own colour, deeper the more of that pitch goes to
    the cell. So each map is in the colour its pitch has in the pies and the
    arsenal table, and the depth means the same share in every map.
    """
    strength = min(max(share, 0.0) / ZONE_FULL_SHARE, 1.0)
    return {
        "bg": tint(color(pitch_type), strength),
        "dark": strength >= ZONE_DARK_TEXT_FROM,
    }
