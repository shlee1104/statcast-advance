"""Do the plan's tendencies hold up on games the plan never saw?

A scouting report is a prediction. It says what a pitcher did, and the reader
acts as if he will keep doing it. This module tests that the only honest way:
build the plan from one part of a pitcher's season and check every call
against the rest.

Two splits are offered:

  halves      his regular-season games in date order, first half against
              second half. The report's real job — "what has he been doing"
              predicting "what will he do next" — in its plainest form.
  postseason  regular season against postseason. The use case that prompted
              this, and the small-sample one: a few October outings cannot
              settle much, and the output says how few there were.

For each plan rule, three things are measured across pitchers:

  same call     the second part gives the same advice as the first
  flipped       the second part gives contradicting advice — "take" became
                "swing", "sit slider" became "sit four-seam". This is the
                failure that matters: a plan that is merely less sure later is
                fine, one that points the hitter the wrong way is not.
  skill         how much better his own first-part number predicts his
                second-part number than the average pitcher's does. Above zero
                means the pitcher-specific tendency carries information a
                generic "most pitchers do this" would not. This is the number
                that says whether the report is worth reading at all.

Every rule is read through the same functions the report uses, so what is
tested is exactly what a hitter would have been told.
"""

from __future__ import annotations

import math

import pandas as pd

from src import config, gameplan
from src.metrics import counts, location

HARD = frozenset(p for p, fam in gameplan.PITCH_FAMILY.items() if fam == "hard")

# A pitch the plan said a side "won't see" still counts as rarely seen below
# this share. The plan's own cutoff is 3%; a pitch at 4% the next month is
# still one a hitter can ignore.
WONT_SEE_HOLDS_BELOW = 0.05


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def _regular(frame: pd.DataFrame) -> pd.DataFrame:
    if "game_type" not in frame.columns:
        return frame
    return frame[frame["game_type"].astype(str) == "R"]


def split_halves(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First half of his regular-season games against the second, by date.

    Split by games, not by calendar date: an injured pitcher's season has a
    hole in it, and a calendar midpoint could leave one half nearly empty.
    """
    regular = _regular(frame)
    dates = (regular.groupby("game_pk")["game_date"].min()
             .astype(str).sort_values())
    if len(dates) < 2:
        return regular.iloc[0:0], regular.iloc[0:0]
    cut = math.ceil(len(dates) / 2)
    first_games = set(dates.index[:cut])
    in_first = regular["game_pk"].isin(first_games)
    return regular[in_first], regular[~in_first]


def split_postseason(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regular season against postseason."""
    if "game_type" not in frame.columns:
        return frame, frame.iloc[0:0]
    regular = frame["game_type"].astype(str) == "R"
    return frame[regular], frame[~regular]


# ---------------------------------------------------------------------------
# Reading the plan's calls from one part of a season
# ---------------------------------------------------------------------------


def _share_in(rows: pd.DataFrame, pitch: str) -> float:
    hit = rows[rows["pitch_type"] == pitch]
    return float(hit["share"].iloc[0]) if len(hit) else 0.0


def side_calls(frame: pd.DataFrame, side: str) -> dict:
    """Every per-side plan call, with the number behind it, or None if gated."""
    usage = counts.situational_usage(frame)
    out: dict = {}

    fp = gameplan.first_pitch_numbers(frame, usage, side)
    rule = gameplan.first_pitch_rule(fp) if fp else None
    out["first_pitch"] = (
        {"call": rule, "value": fp["called_on_takes"], "n": fp["takes"]}
        if rule else None
    )

    hc = gameplan.hitters_count_numbers(usage, side)
    if hc:
        kind = gameplan.hitters_count_rule(hc)
        call = {"hard": "sit hard", "pitch": f"sit {hc['top']}", "mixes": "mixes"}[kind]
        out["hitters_count"] = {"call": call, "value": hc["hard_share"],
                                "n": hc["n"], "rows": hc["rows"]}
    else:
        out["hitters_count"] = None

    ts = gameplan.two_strike_numbers(frame, usage, side)
    if ts:
        kind = gameplan.two_strike_rule(ts)
        call = f"expect {ts['top']}" if kind == "expect" else kind
        out["two_strikes"] = {"call": call, "top": ts["top"], "value": ts["top_share"],
                              "below": ts["below"], "n": ts["n"], "rows": ts["rows"]}
    else:
        out["two_strikes"] = None

    out["rare"] = gameplan.rare_pitches(frame, side)
    out["side_n"] = int((frame["stand"] == side).sum())
    return out


def _target(call: str | None, look_fastball_is_a_call: bool = False) -> frozenset | None:
    """The pitches a pitch-type call tells a hitter to look for; None otherwise.

    "Look fastball and adjust" is the default two-strike approach, not a call
    on a pitch, so it cannot contradict one: "expect the slider" becoming
    "look fastball and adjust to the slider" is weaker advice, not opposite
    advice. The first version of this check counted it as a flip, which put
    the two-strike flip rate at 12% when the cases where the named pitch
    actually changed to a different pitch were 2%. The broad reading is kept,
    behind the flag, so both numbers can be reported.
    """
    if call == "sit hard" or (call == "look_fastball" and look_fastball_is_a_call):
        return HARD
    if call and call.startswith(("sit ", "expect ")):
        return frozenset({call.split(" ", 1)[1]})
    return None


def _advice_given(row: pd.Series) -> bool:
    """Did the first part tell the hitter something, rather than "no case"?"""
    rule, call = row["rule"], row["call_first"]
    if rule == "first_pitch":
        return call in ("take", "swing")
    if rule in ("hitters_count", "two_strikes"):
        return call != "mixes"
    if rule == "lay_off_low":
        return call == "lay off low"
    return True


def _flipped(a: str | None, b: str | None, broad: bool = False) -> bool:
    """True when both parts give advice and the advice points different ways."""
    if {a, b} == {"take", "swing"}:
        return True
    ta, tb = _target(a, broad), _target(b, broad)
    return ta is not None and tb is not None and not (ta & tb)


# ---------------------------------------------------------------------------
# Comparing two parts
# ---------------------------------------------------------------------------


def _row(rule, side, call_a, call_b, value_a, value_b, n_a, n_b, same=None, flipped=None):
    return {
        "rule": rule, "side": side,
        "call_first": call_a, "call_second": call_b,
        "same_call": bool(call_a == call_b) if same is None else bool(same),
        "flipped": _flipped(call_a, call_b) if flipped is None else bool(flipped),
        "flipped_broad": (_flipped(call_a, call_b, broad=True) if flipped is None
                          else bool(flipped)),
        "value_first": value_a, "value_second": value_b,
        "n_first": n_a, "n_second": n_b,
    }


def compare(first: pd.DataFrame, second: pd.DataFrame) -> list[dict]:
    """Every plan call made from `first`, checked against `second`.

    A rule appears only when both parts cleared its sample gate. Gated rules
    are not counted as failures — "too few pitches to say" is not a wrong
    call — but the script reports how many were gated, so a thin second part
    is visible rather than silently flattering.
    """
    rows: list[dict] = []
    if len(first) == 0 or len(second) == 0:
        return rows

    for side in ("L", "R"):
        a, b = side_calls(first, side), side_calls(second, side)

        if a["first_pitch"] and b["first_pitch"]:
            fa, fb = a["first_pitch"], b["first_pitch"]
            rows.append(_row("first_pitch", side, fa["call"], fb["call"],
                             fa["value"], fb["value"], fa["n"], fb["n"]))

        if a["hitters_count"] and b["hitters_count"]:
            ha, hb = a["hitters_count"], b["hitters_count"]
            rows.append(_row("hitters_count", side, ha["call"], hb["call"],
                             ha["value"], hb["value"], ha["n"], hb["n"]))

        if a["two_strikes"] and b["two_strikes"]:
            ta, tb = a["two_strikes"], b["two_strikes"]
            # The value tracked is the share of the pitch HE WAS SAID TO
            # FAVOUR, in the second part — not whatever led the second part.
            rows.append(_row("two_strikes", side, ta["call"], tb["call"],
                             ta["value"], _share_in(tb["rows"], ta["top"]),
                             ta["n"], tb["n"]))

            low_cut = gameplan._cut("lay_off_low_share", 0.50)
            if ta["below"] == ta["below"]:
                below_b, located = gameplan._finishes_low(second, side, ta["top"])
                if located >= 20:
                    call_a = "lay off low" if ta["below"] >= low_cut else "no low call"
                    call_b = "lay off low" if below_b >= low_cut else "no low call"
                    rows.append(_row("lay_off_low", side, call_a, call_b,
                                     ta["below"], below_b, ta["n"], located,
                                     flipped=False))

        if a["rare"] and a["rare"]["rare"] and b["side_n"] >= 100:
            side_b = second[second["stand"] == side]["pitch_type"].value_counts(normalize=True)
            for pitch, share_a in a["rare"]["rare"]:
                share_b = float(side_b.get(pitch, 0.0))
                rows.append(_row(
                    "wont_see", side, f"won't see {pitch}",
                    f"won't see {pitch}" if share_b < WONT_SEE_HOLDS_BELOW
                    else f"sees {pitch} {share_b:.0%}",
                    share_a, share_b, a["rare"]["n"], b["side_n"],
                    same=share_b < WONT_SEE_HOLDS_BELOW,
                    flipped=share_b >= 0.10,
                ))

    rows.extend(_count_tell_rows(first, second))
    rows.extend(_height_rows(first, second))
    return rows


def _count_tell_rows(first: pd.DataFrame, second: pd.DataFrame) -> list[dict]:
    """The count tell from the first part: is the pitch still elevated there?"""
    strong = gameplan.count_tells(first)
    if len(strong) == 0:
        return []
    pitch = strong.sort_values("score", ascending=False).iloc[0]["pitch_type"]
    told = strong[strong["pitch_type"] == pitch]
    in_counts = second[second["count"].isin(set(told["count"]))]
    if len(in_counts) < int(config.get("flags.min_n.predictable_count", 20)):
        return []

    own_b = float((second["pitch_type"] == pitch).mean())
    share_b = float((in_counts["pitch_type"] == pitch).mean())
    lift_b = share_b / own_b if own_b else float("nan")
    lift_a = float(told["lift"].max())
    holds = lift_b == lift_b and lift_b >= 1.5
    label = f"{pitch} in {'/'.join(sorted(told['count']))}"
    return [_row("count_tell", "both", label,
                 label if holds else f"{pitch} not elevated ({lift_b:.1f}x)",
                 lift_a, lift_b, int(told["n"].sum()), len(in_counts),
                 same=holds, flipped=lift_b == lift_b and lift_b < 1.0)]


def _height_rows(first: pd.DataFrame, second: pd.DataFrame) -> list[dict]:
    """The height line from the first part: does that height still say that pitch?"""
    strong = gameplan.height_tells(first)
    if len(strong) == 0:
        return []
    top = strong.sort_values("score", ascending=False).iloc[0]
    tells_b = location.location_tells(second, by="v_band", min_n=1)
    if len(tells_b) == 0:
        return []
    match = tells_b[(tells_b["band"] == top.band) & (tells_b["pitch_type"] == top.pitch_type)]
    n_band = int(match["n"].iloc[0]) if len(match) else 0
    if n_band < int(config.get("flags.min_n.location_tell", 40)):
        return []

    share_b = float(match["p_pitch"].iloc[0])
    lift_b = float(match["lift"].iloc[0])
    min_share = float(config.get("flags.thresholds.location_tell_share", 0.45))
    holds = share_b >= min_share
    label = f"{top.band.lower()} = {top.pitch_type}"
    return [_row("height", "both", label,
                 label if holds else f"{top.band.lower()} = {top.pitch_type} {share_b:.0%}",
                 float(top.p_pitch), share_b, int(top.n), n_band,
                 same=holds, flipped=lift_b < 1.0)]


# ---------------------------------------------------------------------------
# Summarising across pitchers
# ---------------------------------------------------------------------------

# Skill needs a number that varies across pitchers. A "won't see" pitch is
# near 0% for everyone, and the count and height lines track a different
# pitch per pitcher, so an average-pitcher baseline means nothing for them.
SKILL_RULES = frozenset({"first_pitch", "hitters_count", "two_strikes", "lay_off_low"})

RULE_LABELS: dict[str, str] = {
    "first_pitch": "First pitch: take / swing",
    "hitters_count": "Hitter's count: what to sit on",
    "two_strikes": "Two strikes: the put-away pitch",
    "lay_off_low": "Two strikes: lay off it low",
    "wont_see": "Won't see",
    "count_tell": "Count tell",
    "height": "Read the height",
}


def summarise(rows: list[dict]) -> list[dict]:
    """One line per rule: how often the call held, flipped, and the skill score.

    Skill compares two predictions of each second-part number: the pitcher's
    own first-part number, and the average first-part number across every
    pitcher in the run. Skill = 1 − (own error ÷ average-pitcher error), so 0
    means knowing the pitcher adds nothing and 1 means it predicts perfectly.
    The average uses first-part numbers only, so nothing from the second part
    leaks into the baseline.
    """
    table = pd.DataFrame(rows)
    out = []
    if table.empty:
        return out
    for rule, label in RULE_LABELS.items():
        sub = table[table["rule"] == rule]
        if sub.empty:
            continue
        # "Neutral" and "mixes" are calls too, but the question that matters
        # for a hitter is whether ADVICE held, so hold rates count only the
        # cases where the first part told him something.
        advised = sub[sub.apply(_advice_given, axis=1)]
        values = sub.dropna(subset=["value_first", "value_second"])
        skill, own_err, avg_err = float("nan"), float("nan"), float("nan")
        if rule in SKILL_RULES and len(values) >= 3:
            own_err = float((values["value_second"] - values["value_first"]).abs().mean())
            avg_err = float((values["value_second"] - values["value_first"].mean()).abs().mean())
            skill = 1 - own_err / avg_err if avg_err else float("nan")
        # Every piece of advice lands in exactly one of three places: the
        # second part said the same thing (held), said something that points
        # the other way (flipped), or said less — "no case", "he mixes",
        # "look fastball" — (weakened). Weakened advice costs a hitter
        # nothing he would not have had without the report; flipped advice
        # sends him the wrong way.
        held = advised["same_call"]
        flipped = advised["flipped"] & ~held
        weakened = ~held & ~flipped

        def pct(mask) -> float:
            return float(mask.mean()) if len(advised) else float("nan")

        out.append({
            "rule": rule,
            "label": label,
            "checked": int(len(sub)),
            "advice_given": int(len(advised)),
            "advice_held": int(held.sum()),
            "held_pct": pct(held),
            "weakened": int(weakened.sum()),
            "weakened_pct": pct(weakened),
            "flipped": int(flipped.sum()),
            "flipped_pct": pct(flipped),
            "flipped_broad_pct": (pct(advised["flipped_broad"] & ~held)
                                  if "flipped_broad" in advised else float("nan")),
            "same_call_all_pct": float(sub["same_call"].mean()),
            "own_error": own_err,
            "average_pitcher_error": avg_err,
            "skill": skill,
        })
    return out
