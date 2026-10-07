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

    def test_no_sit_line_on_a_pitch_he_throws_less_than_half_the_time(self):
        """Snell's report said "sit four-seam — 42%". A plurality is not a plan."""
        frame = make_frame([
            {"pitch_type": "FF", "n": 45, "balls": 2, "strikes": 0},
            {"pitch_type": "SL", "n": 30, "balls": 2, "strikes": 0},
            {"pitch_type": "CH", "n": 25, "balls": 2, "strikes": 0},
        ])
        key = gameplan.hitters_count_key(counts.situational_usage(frame), "R")
        assert key["text"].startswith("He still mixes")

    def test_league_typical_fastball_use_is_not_sit_hard(self):
        """League pitchers throw a fastball about 63% of the time in hitter's
        counts. Doing the same is not a tendency worth a line."""
        frame = make_frame([
            {"pitch_type": "FF", "n": 64, "balls": 2, "strikes": 0},
            {"pitch_type": "SL", "n": 36, "balls": 2, "strikes": 0},
        ])
        key = gameplan.hitters_count_key(counts.situational_usage(frame), "R")
        assert not key["text"].startswith("Sit hard")
        assert key["text"].startswith("Sit four-seam")

    def test_two_strikes_says_he_mixes_when_no_pitch_owns_the_count(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 28, "strikes": 2},
            {"pitch_type": "ST", "n": 26, "strikes": 2},
            {"pitch_type": "SI", "n": 24, "strikes": 2},
            {"pitch_type": "CH", "n": 22, "strikes": 2},
        ])
        key = gameplan.two_strike_key(frame, counts.situational_usage(frame), "R")
        assert key["text"].startswith("No single put-away pitch")
        assert "Expect" not in key["text"]

    def test_near_tie_with_fastball_says_look_fastball(self):
        """Snell to lefties: four-seam 35%, curveball 31%, slider 28%. No pitch
        to sit on, so the plan gives the standard approach and names what to
        adjust to."""
        frame = make_frame([
            {"pitch_type": "FF", "n": 35, "strikes": 2},
            {"pitch_type": "CU", "n": 31, "strikes": 2, "plate_z": 1.0},
            {"pitch_type": "SL", "n": 28, "strikes": 2},
            {"pitch_type": "CH", "n": 6, "strikes": 2},
        ])
        key = gameplan.two_strike_key(frame, counts.situational_usage(frame), "R")
        assert "Expect" not in key["text"]
        assert "Look fastball and adjust to the curveball and the slider." in key["text"]
        assert "of his curveballs finish below the zone — lay off it low" in key["text"]

    def test_clear_leader_still_gets_expect(self):
        frame = make_frame([
            {"pitch_type": "SL", "n": 55, "strikes": 2},
            {"pitch_type": "FF", "n": 30, "strikes": 2},
            {"pitch_type": "CH", "n": 15, "strikes": 2},
        ])
        key = gameplan.two_strike_key(frame, counts.situational_usage(frame), "R")
        assert key["text"].startswith("Expect the slider")

    def test_no_look_fastball_advice_without_fastballs(self):
        frame = make_frame([
            {"pitch_type": "SL", "n": 36, "strikes": 2},
            {"pitch_type": "CH", "n": 34, "strikes": 2},
            {"pitch_type": "CU", "n": 20, "strikes": 2},
            {"pitch_type": "FF", "n": 10, "strikes": 2},
        ])
        key = gameplan.two_strike_key(frame, counts.situational_usage(frame), "R")
        assert "Look fastball" not in key["text"]
        assert "Mostly off-speed" in key["text"]

    def test_count_tell_needs_the_pitch_to_be_worth_looking_for(self):
        """Doubling a rare curveball to 5% is a real lift and useless advice."""
        frame = make_frame([
            {"pitch_type": "FF", "n": 380, "balls": 0, "strikes": 0},
            {"pitch_type": "CU", "n": 20, "balls": 0, "strikes": 0},
            {"pitch_type": "FF", "n": 600, "balls": 1, "strikes": 1},
        ])
        lifts = counts.count_lifts(frame, min_n=20)
        cell = lifts[(lifts["pitch_type"] == "CU") & (lifts["count"] == "0-0")]
        assert float(cell["lift"].iloc[0]) >= 2.0
        assert gameplan.count_tell_key(frame) is None

    def test_count_tell_fires_when_lift_and_share_both_clear(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 60, "balls": 0, "strikes": 2},
            {"pitch_type": "FS", "n": 40, "balls": 0, "strikes": 2},
            {"pitch_type": "FF", "n": 600, "balls": 1, "strikes": 1},
            {"pitch_type": "FS", "n": 20, "balls": 1, "strikes": 1},
        ])
        key = gameplan.count_tell_key(frame)
        assert key is not None
        assert "splitter" in key["text"] and "0-2" in key["text"]

    def test_count_tell_ignores_three_oh(self):
        """Every pitcher throws a fastball 3-0. A lift there describes the
        count, not the pitcher."""
        frame = make_frame([
            {"pitch_type": "FF", "n": 40, "balls": 3, "strikes": 0},
            {"pitch_type": "FF", "n": 100, "balls": 0, "strikes": 0},
            {"pitch_type": "CU", "n": 200, "balls": 0, "strikes": 0},
        ])
        assert gameplan.count_tell_key(frame) is None


class TestHeightLine:
    """Sixteen of twenty-two calibration pitchers said "if it's up, it's the
    four-seam". That is true of nearly everyone, so it is left out."""

    def tells(self, monkeypatch, rows):
        table = pd.DataFrame(rows)
        monkeypatch.setattr(gameplan.location, "location_tells",
                            lambda frame, by, min_n: table)
        return make_frame([{"pitch_type": "FF", "n": 10}])

    def test_fastball_up_is_not_a_plan_line(self, monkeypatch):
        frame = self.tells(monkeypatch, [
            {"band": "UP", "pitch_type": "FF", "lift": 1.9, "p_pitch": 0.74,
             "score": 5.0, "n": 600},
        ])
        assert gameplan.location_key(frame) is None

    def test_a_sinker_up_is_also_a_fastball(self, monkeypatch):
        frame = self.tells(monkeypatch, [
            {"band": "UP", "pitch_type": "SI", "lift": 1.5, "p_pitch": 0.70,
             "score": 5.0, "n": 300},
        ])
        assert gameplan.location_key(frame) is None

    def test_offspeed_down_still_reads(self, monkeypatch):
        frame = self.tells(monkeypatch, [
            {"band": "UP", "pitch_type": "FF", "lift": 1.9, "p_pitch": 0.74,
             "score": 9.0, "n": 600},
            {"band": "DOWN", "pitch_type": "CH", "lift": 1.7, "p_pitch": 0.54,
             "score": 4.0, "n": 500},
        ])
        key = gameplan.location_key(frame)
        assert key["text"] == "If it's down, it's probably the changeup (54% of pitches there)."


class TestCountGrid:
    def build(self):
        """Cutter in hitter's counts, splitter with two strikes, four-seam
        everywhere — the Yamamoto shape in miniature."""
        return make_frame([
            {"pitch_type": "FF", "n": 60, "balls": 0, "strikes": 0},
            {"pitch_type": "FS", "n": 10, "balls": 0, "strikes": 0},
            {"pitch_type": "FC", "n": 10, "balls": 0, "strikes": 0},
            {"pitch_type": "FF", "n": 30, "balls": 2, "strikes": 1},
            {"pitch_type": "FC", "n": 30, "balls": 2, "strikes": 1},
            {"pitch_type": "FF", "n": 30, "balls": 1, "strikes": 2},
            {"pitch_type": "FS", "n": 38, "balls": 1, "strikes": 2},
            {"pitch_type": "FC", "n": 2, "balls": 1, "strikes": 2},
            {"pitch_type": "FF", "n": 5, "balls": 3, "strikes": 0},
        ])

    def test_every_pitch_gets_all_twelve_counts(self):
        grid = counts.count_grid(self.build(), min_count=30)
        per_pitch = grid.groupby("pitch_type").size()
        assert set(per_pitch) == {12}

    def test_lift_is_against_his_own_rate(self):
        grid = counts.count_grid(self.build(), min_count=30)
        cell = grid[(grid.pitch_type == "FC") & (grid["count"] == "2-1")].iloc[0]
        own = 42 / 215
        assert cell["share"] == pytest.approx(0.5)
        assert cell["lift"] == pytest.approx(0.5 / own)

    def test_a_rare_pitch_in_a_well_sampled_count_stays_visible(self):
        """Gated on the COUNT, not the pitch: 2 cutters out of 70 pitches in
        1-2 is a measured 3%, and "he almost never throws it here" is the
        point. Gating on the pitch would blank exactly this cell."""
        grid = counts.count_grid(self.build(), min_count=30)
        cell = grid[(grid.pitch_type == "FC") & (grid["count"] == "1-2")].iloc[0]
        assert cell["n"] == 2
        assert bool(cell["reliable"])
        assert cell["lift"] < 0.25

    def test_a_thin_count_is_marked(self):
        grid = counts.count_grid(self.build(), min_count=30)
        cell = grid[(grid.pitch_type == "FF") & (grid["count"] == "3-0")].iloc[0]
        assert not bool(cell["reliable"])

    def test_log2_is_symmetric_and_floored(self):
        grid = counts.count_grid(self.build(), min_count=30)
        zero = grid[(grid.pitch_type == "FS") & (grid["count"] == "2-1")].iloc[0]
        assert zero["lift"] == 0
        assert zero["log2_lift"] == -5.0

    def test_rare_pitches_get_no_grid(self):
        frame = pd.concat([self.build(), make_frame([{"pitch_type": "SL", "n": 6}])])
        grid = counts.count_grid(frame.reset_index(drop=True), min_count=30, min_usage=0.05)
        assert "SL" not in set(grid["pitch_type"])


class TestCountHeadline:
    def test_names_both_pitches_and_their_separation(self):
        """The chart title should state the finding, not name the metric."""
        frame = make_frame([
            {"pitch_type": "FF", "n": 100, "balls": 0, "strikes": 0},
            {"pitch_type": "FC", "n": 60, "balls": 2, "strikes": 1},
            {"pitch_type": "FF", "n": 40, "balls": 2, "strikes": 1},
            {"pitch_type": "FS", "n": 60, "balls": 1, "strikes": 2},
            {"pitch_type": "FF", "n": 40, "balls": 1, "strikes": 2},
            {"pitch_type": "FS", "n": 10, "balls": 0, "strikes": 0},
            {"pitch_type": "FC", "n": 10, "balls": 0, "strikes": 0},
        ])
        text = gameplan.count_headline(frame)
        assert "cutter" in text and "hitter's counts" in text
        assert "splitter" in text and "two strikes" in text
        assert "each rarely shows up" in text

    def test_two_strike_pitch_alone_is_still_a_sentence(self):
        """Six reports titled a chart "The slider with two strikes." """
        frame = make_frame([
            {"pitch_type": "FF", "n": 100, "balls": 0, "strikes": 0},
            {"pitch_type": "SL", "n": 20, "balls": 0, "strikes": 0},
            {"pitch_type": "FF", "n": 50, "balls": 2, "strikes": 1},
            {"pitch_type": "SL", "n": 10, "balls": 2, "strikes": 1},
            {"pitch_type": "SL", "n": 60, "balls": 1, "strikes": 2},
            {"pitch_type": "FF", "n": 40, "balls": 1, "strikes": 2},
        ])
        assert gameplan.count_headline(frame) == "The slider comes out with two strikes."

    def test_does_not_crash_when_a_region_is_empty(self):
        """A pitcher with no two-strike pitches in the sample once took the
        whole report down here."""
        frame = make_frame([
            {"pitch_type": "FF", "n": 50, "balls": 0, "strikes": 0},
            {"pitch_type": "SL", "n": 50, "balls": 0, "strikes": 0},
        ])
        assert gameplan.count_headline(frame) is None

    def test_silent_when_nothing_moves(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 50, "balls": b, "strikes": s}
            for b in range(3) for s in range(3)
        ] + [
            {"pitch_type": "SL", "n": 50, "balls": b, "strikes": s}
            for b in range(3) for s in range(3)
        ])
        assert gameplan.count_headline(frame) is None


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
