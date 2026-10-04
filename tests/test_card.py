"""Tests for the dugout card: scout arsenal, situational usage, recent form,
season line, and the hitting plan built from them.

The hitting plan is the part of the report a player acts on directly, so the
tests here are less about arithmetic than about whether each sentence is
justified by the number behind it. The first-pitch rule shipped once with the
right arithmetic on the wrong number, and a test below exists to stop that
recurring.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src import gameplan
from src.metrics import arsenal, counts, events, splits


def make_frame(specs: list[dict]) -> pd.DataFrame:
    """Build a frame carrying every column the card touches.

    Defaults describe a right-hander's four-seam over the middle to a
    right-handed hitter, taken for a called strike, in a regular-season game.
    """
    rows = []
    at_bat = 0
    for spec in specs:
        for _ in range(spec["n"]):
            at_bat += 1
            balls = spec.get("balls", 0)
            strikes = spec.get("strikes", 0)
            description = spec.get("description", "called_strike")
            swing = spec.get("is_swing", False)
            rows.append({
                "game_pk": spec.get("game_pk", 1),
                "game_date": spec.get("game_date", "2025-06-01"),
                "game_type": spec.get("game_type", "R"),
                "at_bat_number": at_bat,
                "pitch_number": 1,
                "balls": balls,
                "strikes": strikes,
                "count": f"{balls}-{strikes}",
                "is_first_pitch": balls == 0 and strikes == 0,
                "is_two_strike": strikes == 2,
                "is_ahead": strikes > balls,
                "is_behind": balls > strikes,
                "pitch_type": spec["pitch_type"],
                "p_throws": spec.get("p_throws", "R"),
                "stand": spec.get("stand", "R"),
                "release_speed": spec.get("velo", 95.0),
                "release_spin_rate": 2200.0,
                "pfx_x": spec.get("pfx_x", -0.5),
                "pfx_z": spec.get("pfx_z", 1.3),
                "plate_x": 0.0,
                "plate_z": spec.get("plate_z", 2.5),
                "sz_top": 3.5,
                "sz_bot": 1.5,
                "zone": spec.get("zone", 5),
                "type": spec.get("type", "S"),
                "description": description,
                "events": spec.get("events", None),
                "estimated_woba_using_speedangle": spec.get("xwoba", None),
                "delta_run_exp": 0.0,
                "is_swing": swing,
                "is_whiff": spec.get("is_whiff", False),
                "is_called_strike": spec.get(
                    "is_called_strike", description == "called_strike"),
                "is_in_zone": spec.get("zone", 5) in range(1, 10),
                "is_chase": False,
                "is_contact": False,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scout arsenal
# ---------------------------------------------------------------------------


class TestScoutArsenal:
    def test_usage_is_split_by_batter_hand(self):
        """A pitcher who never shows lefties his slider is a smaller arsenal to
        a lefty, and the overall column hides that."""
        frame = make_frame([
            {"pitch_type": "FF", "n": 100, "stand": "L"},
            {"pitch_type": "FF", "n": 60, "stand": "R"},
            {"pitch_type": "SL", "n": 40, "stand": "R"},
        ])
        table = arsenal.scout_arsenal(frame)
        assert table.loc["SL", "usage_L"] == pytest.approx(0.0)
        assert table.loc["SL", "usage_R"] == pytest.approx(0.40)
        assert table.loc["FF", "usage_L"] == pytest.approx(1.0)

    def test_velocity_is_a_range_not_an_average(self):
        frame = pd.concat([
            make_frame([{"pitch_type": "FF", "n": 1, "velo": v}])
            for v in range(90, 100)
        ]).reset_index(drop=True)
        frame = pd.concat([frame] * 10).reset_index(drop=True)
        table = arsenal.scout_arsenal(frame)
        assert table.loc["FF", "sits_lo"] < table.loc["FF", "sits_hi"]
        assert table.loc["FF", "touches"] >= table.loc["FF", "sits_hi"]
        assert table.loc["FF", "touches"] <= 99.0

    def test_arm_side_is_positive_for_both_hands(self):
        """Raw pfx_x flips sign with pitcher hand. The same arm-side run must
        read positive for a right-hander and a left-hander alike."""
        righty = make_frame([{"pitch_type": "SI", "n": 50, "p_throws": "R", "pfx_x": -1.0}])
        lefty = make_frame([{"pitch_type": "SI", "n": 50, "p_throws": "L", "pfx_x": 1.0}])
        assert arsenal.scout_arsenal(righty).loc["SI", "arm_run"] == pytest.approx(12.0)
        assert arsenal.scout_arsenal(lefty).loc["SI", "arm_run"] == pytest.approx(12.0)

    def test_movement_is_in_inches(self):
        frame = make_frame([{"pitch_type": "FF", "n": 50, "pfx_z": 1.5}])
        assert arsenal.scout_arsenal(frame).loc["FF", "ivb"] == pytest.approx(18.0)

    def test_empty_frame(self):
        frame = make_frame([{"pitch_type": "FF", "n": 1}]).iloc[0:0]
        assert len(arsenal.scout_arsenal(frame)) == 0


# ---------------------------------------------------------------------------
# Situational usage
# ---------------------------------------------------------------------------


class TestSituationalUsage:
    def build(self):
        return make_frame([
            {"pitch_type": "FF", "n": 40, "balls": 0, "strikes": 0},
            {"pitch_type": "SL", "n": 10, "balls": 0, "strikes": 0},
            {"pitch_type": "FS", "n": 30, "balls": 1, "strikes": 2},
            {"pitch_type": "FF", "n": 20, "balls": 1, "strikes": 2},
            {"pitch_type": "FF", "n": 30, "balls": 2, "strikes": 0},
            {"pitch_type": "SL", "n": 10, "balls": 1, "strikes": 1},
        ])

    def test_shares_sum_to_one_per_situation_and_side(self):
        table = counts.situational_usage(self.build())
        totals = table.groupby(["situation", "stand"])["share"].sum()
        for value in totals:
            assert value == pytest.approx(1.0)

    def test_even_excludes_the_first_pitch(self):
        """0-0 has its own row; a hitter approaches it differently from 1-1."""
        table = counts.situational_usage(self.build())
        even = table[(table["situation"] == "even") & (table["stand"] == "All")]
        assert even["situation_n"].iloc[0] == 10

    def test_two_strikes_overlaps_ahead_on_purpose(self):
        table = counts.situational_usage(self.build())
        ahead = table[(table["situation"] == "pitcher_ahead") & (table["stand"] == "All")]
        two = table[(table["situation"] == "two_strikes") & (table["stand"] == "All")]
        assert ahead["situation_n"].iloc[0] == 50
        assert two["situation_n"].iloc[0] == 50

    def test_behind_is_a_hitters_count(self):
        table = counts.situational_usage(self.build())
        behind = table[(table["situation"] == "pitcher_behind") & (table["stand"] == "All")]
        assert behind.iloc[0]["pitch_type"] == "FF"
        assert behind["situation_n"].iloc[0] == 30

    def test_split_by_side_and_overall(self):
        table = counts.situational_usage(self.build())
        assert set(table["stand"]) == {"All", "R"}

    def test_thin_situations_are_marked(self):
        table = counts.situational_usage(self.build(), min_situation=30)
        even = table[(table["situation"] == "even")]
        assert not even["reliable"].any()


# ---------------------------------------------------------------------------
# Recent form
# ---------------------------------------------------------------------------


class TestRecentForm:
    def build(self, recent_ff: int = 10, recent_fs: int = 90):
        earlier = [
            {"pitch_type": "FF", "n": 60, "game_pk": g, "game_date": f"2025-05-{g:02d}",
             "velo": 95.0}
            for g in range(1, 11)
        ] + [
            {"pitch_type": "FS", "n": 40, "game_pk": g, "game_date": f"2025-05-{g:02d}"}
            for g in range(1, 11)
        ]
        recent = [
            {"pitch_type": "FF", "n": recent_ff, "game_pk": 100 + g,
             "game_date": f"2025-09-{g:02d}", "velo": 96.5}
            for g in range(1, 6)
        ] + [
            {"pitch_type": "FS", "n": recent_fs, "game_pk": 100 + g,
             "game_date": f"2025-09-{g:02d}"}
            for g in range(1, 6)
        ]
        return make_frame(earlier + recent)

    def test_picks_the_most_recent_outings(self):
        form = splits.recent_form(self.build(), last_n=5)
        assert form["last_n"] == 5
        assert all(d.startswith("2025-09") for d in form["dates"])
        assert form["dates"][0] == "2025-09-05"

    def test_compares_against_earlier_not_the_full_season(self):
        """The full season contains the recent starts and would mute any
        change by the share of the season they make up."""
        form = splits.recent_form(self.build(), last_n=5)
        ff = form["shifts"].set_index("pitch_type").loc["FF"]
        assert ff["earlier_share"] == pytest.approx(0.60)
        assert ff["recent_share"] == pytest.approx(0.10)

    def test_a_large_shift_is_notable(self):
        form = splits.recent_form(self.build(), last_n=5)
        assert form["shifts"].set_index("pitch_type").loc["FF", "notable"]

    def test_no_shift_is_not_notable(self):
        form = splits.recent_form(self.build(recent_ff=60, recent_fs=40), last_n=5)
        assert not form["shifts"]["notable"].any()

    def test_velocity_is_read_from_one_fastball(self):
        form = splits.recent_form(self.build(), last_n=5)
        assert form["fastball"] == "FF"
        assert form["velo_delta"] == pytest.approx(1.5)

    def test_primary_fastball_prefers_the_more_used(self):
        frame = make_frame([
            {"pitch_type": "SI", "n": 80}, {"pitch_type": "FF", "n": 20},
            {"pitch_type": "FC", "n": 100},
        ])
        assert splits.primary_fastball(frame) == "SI"

    def test_empty_frame(self):
        frame = make_frame([{"pitch_type": "FF", "n": 1}]).iloc[0:0]
        assert splits.recent_form(frame)["last_n"] == 0


# ---------------------------------------------------------------------------
# Season line
# ---------------------------------------------------------------------------


class TestSeasonLine:
    def test_rates_use_plate_appearances_with_an_outcome(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 30, "events": "strikeout"},
            {"pitch_type": "FF", "n": 10, "events": "walk"},
            {"pitch_type": "FF", "n": 60, "events": "field_out"},
            {"pitch_type": "FF", "n": 50},  # mid-at-bat pitches, no outcome
        ])
        line = events.season_line(frame)
        assert line["batters"] == 100
        assert line["k_rate"] == pytest.approx(0.30)
        assert line["bb_rate"] == pytest.approx(0.10)
        assert line["k_minus_bb"] == pytest.approx(0.20)

    def test_empty_frame(self):
        frame = make_frame([{"pitch_type": "FF", "n": 1}]).iloc[0:0]
        assert events.season_line(frame)["batters"] == 0


# ---------------------------------------------------------------------------
# Hitting plan
# ---------------------------------------------------------------------------


class TestFirstPitchKey:
    """The rule that once gave backwards advice."""

    def test_uses_called_strikes_on_takes_not_strikes_or_swings(self):
        """60% of first pitches are swung at (inflating the usual strike rate
        to over 60%), but the TAKEN ones are called strikes only 25% of the
        time. Taking is the right advice; the inflated number says the
        opposite."""
        frame = make_frame([
            {"pitch_type": "FF", "n": 60, "is_swing": True, "type": "S",
             "description": "foul"},
            {"pitch_type": "FF", "n": 10, "is_swing": False, "type": "S",
             "description": "called_strike"},
            {"pitch_type": "FF", "n": 30, "is_swing": False, "type": "B",
             "description": "ball"},
        ])
        usage = counts.situational_usage(frame)
        key = gameplan.first_pitch_key(frame, usage, "R")
        assert key is not None
        assert "25%" in key["text"]
        assert "make him throw one" in key["text"]
        assert "down 0-1" not in key["text"]

    def test_recommends_swinging_when_takes_are_usually_strikes(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 70, "is_swing": False,
             "description": "called_strike"},
            {"pitch_type": "FF", "n": 30, "is_swing": False, "type": "B",
             "description": "ball"},
        ])
        usage = counts.situational_usage(frame)
        key = gameplan.first_pitch_key(frame, usage, "R")
        assert "down 0-1" in key["text"]

    def test_silent_below_the_sample_gate(self):
        frame = make_frame([{"pitch_type": "FF", "n": 10}])
        usage = counts.situational_usage(frame)
        assert gameplan.first_pitch_key(frame, usage, "R") is None


class TestPlanKeys:
    def test_sit_hard_when_fastballs_dominate_hitters_counts(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 40, "balls": 2, "strikes": 0},
            {"pitch_type": "SI", "n": 30, "balls": 2, "strikes": 0},
            {"pitch_type": "CU", "n": 20, "balls": 2, "strikes": 0},
        ])
        key = gameplan.hitters_count_key(counts.situational_usage(frame), "R")
        assert key["text"].startswith("Sit hard")
        assert "78%" in key["text"]

    def test_does_not_list_a_pitch_he_never_throws_there(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 90, "balls": 2, "strikes": 0},
            {"pitch_type": "CU", "n": 10, "balls": 2, "strikes": 0},
            {"pitch_type": "SI", "n": 100, "balls": 0, "strikes": 2},
        ])
        key = gameplan.hitters_count_key(counts.situational_usage(frame), "R")
        assert "sinker" not in key["text"]

    def test_wont_show_names_the_pitches_and_shrinks_the_arsenal(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 100, "stand": "L"},
            {"pitch_type": "CU", "n": 60, "stand": "L"},
            {"pitch_type": "SL", "n": 1, "stand": "L"},
            {"pitch_type": "SL", "n": 80, "stand": "R"},
        ])
        key = gameplan.wont_show_key(frame, "L")
        assert "slider" in key["text"]
        assert "two-pitch" in key["text"]

    def test_two_strike_key_says_where_it_finishes(self):
        frame = make_frame([
            {"pitch_type": "FS", "n": 60, "strikes": 2, "plate_z": 1.0,
             "is_swing": True, "is_whiff": True, "description": "swinging_strike"},
            {"pitch_type": "FF", "n": 20, "strikes": 2},
        ])
        key = gameplan.two_strike_key(frame, counts.situational_usage(frame), "R")
        assert "splitter" in key["text"]
        assert "lay off it low" in key["text"]

    def test_count_tell_ignores_three_oh(self):
        """Every pitcher throws a fastball 3-0. A lift there describes the
        count, not the pitcher."""
        frame = make_frame([
            {"pitch_type": "FF", "n": 40, "balls": 3, "strikes": 0},
            {"pitch_type": "FF", "n": 100, "balls": 0, "strikes": 0},
            {"pitch_type": "CU", "n": 200, "balls": 0, "strikes": 0},
        ])
        assert gameplan.count_tell_key(frame) is None


class TestPlanAssembly:
    def build(self):
        specs = []
        for side in ("L", "R"):
            specs += [
                {"pitch_type": "FF", "n": 60, "stand": side, "balls": 0, "strikes": 0},
                {"pitch_type": "CU", "n": 20, "stand": side, "balls": 0, "strikes": 0},
                {"pitch_type": "FF", "n": 60, "stand": side, "balls": 2, "strikes": 0},
                {"pitch_type": "FS", "n": 60, "stand": side, "balls": 1, "strikes": 2,
                 "plate_z": 1.0},
            ]
        return make_frame(specs)

    def test_has_both_sides_and_a_stated_basis(self):
        plan = gameplan.build(self.build())
        assert plan["L"] and plan["R"]
        assert "how often" in plan["basis"]

    def test_every_key_carries_its_sample_size(self):
        plan = gameplan.build(self.build())
        for key in plan["L"] + plan["R"] + plan["both"]:
            assert key["n"] > 0

    def test_written_for_the_hitter(self):
        """No analyst vocabulary in the sentences a player reads."""
        plan = gameplan.build(self.build())
        text = " ".join(k["text"] for k in plan["L"] + plan["R"] + plan["both"])
        for jargon in ("lift", "xwOBA", "z =", "entropy", "severity"):
            assert jargon not in text

    def test_empty_frame(self):
        frame = make_frame([{"pitch_type": "FF", "n": 1}]).iloc[0:0]
        plan = gameplan.build(frame)
        assert plan["L"] == [] and plan["R"] == []
