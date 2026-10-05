"""Tests for src/charts.py and the chart headlines in src/gameplan.py.

Chart geometry is tested for the two things a hand-drawn chart gets wrong:
labels landing on each other, and a visual emphasis that disagrees with what
the chart's own title says.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import charts, gameplan
from tests.test_card import make_frame


class TestSpreadLabels:
    def test_keeps_the_minimum_gap(self):
        result = charts.spread_labels([100, 101, 102], min_gap=16, lo=0, hi=400)
        ordered = sorted(result)
        assert all(b - a >= 16 - 1e-9 for a, b in zip(ordered, ordered[1:]))

    def test_leaves_well_spaced_labels_alone(self):
        assert charts.spread_labels([50, 150, 250], 16, 0, 400) == [50, 150, 250]

    def test_stays_inside_the_bounds(self):
        result = charts.spread_labels([395, 398, 399], 16, 0, 400)
        assert max(result) <= 400
        assert min(result) >= 0

    def test_preserves_input_order(self):
        """Labels must stay attached to their own line."""
        result = charts.spread_labels([300, 100, 101], 16, 0, 400)
        assert result[0] == 300
        assert result[1] < result[2]

    def test_empty(self):
        assert charts.spread_labels([], 16, 0, 400) == []


class TestSlopeLayout:
    def rows(self):
        return [
            {"pitch_type": "FF", "usage_L": 0.37, "usage_R": 0.31},
            {"pitch_type": "SI", "usage_L": 0.006, "usage_R": 0.143},
            {"pitch_type": "SL", "usage_L": 0.004, "usage_R": 0.05},
            {"pitch_type": "FC", "usage_L": 0.107, "usage_R": 0.118},
        ]

    def test_two_near_zero_labels_do_not_collide(self):
        """Sinker 1% and slider 0% would print on top of each other."""
        layout = charts.slope_layout(self.rows())
        ys = sorted(p["label_y_L"] for p in layout["pitches"])
        assert all(b - a >= 16 - 1e-9 for a, b in zip(ys, ys[1:]))

    def test_axis_starts_at_zero_with_a_shared_floor(self):
        layout = charts.slope_layout(self.rows())
        assert layout["ticks"][0]["value"] == 0
        assert layout["y_max"] == pytest.approx(0.40)

    def test_a_pitch_one_side_never_sees_is_drawn_strong(self):
        """0% to 5% is a small gap and the whole story; it must not be the faded
        line when the chart's headline is about it."""
        layout = charts.slope_layout(self.rows())
        slider = next(p for p in layout["pitches"] if p["pitch"] == "SL")
        cutter = next(p for p in layout["pitches"] if p["pitch"] == "FC")
        assert slider["strong"]
        assert not cutter["strong"]

    def test_every_pitch_has_a_colour(self):
        layout = charts.slope_layout(self.rows())
        assert all(p["color"].startswith("#") for p in layout["pitches"])


class TestMovementLayout:
    def rows(self):
        """Skubal-like: changeup and sinker close together on the arm side."""
        return [
            {"pitch_type": "CH", "name": "Changeup", "arm_run": 15.0, "ivb": 6.4,
             "usage": 0.31, "whiff_rate": 0.47, "velo": 88.1},
            {"pitch_type": "FF", "name": "Four-Seam", "arm_run": 3.4, "ivb": 17.0,
             "usage": 0.30, "whiff_rate": 0.25, "velo": 97.7},
            {"pitch_type": "SI", "name": "Sinker", "arm_run": 13.1, "ivb": 13.6,
             "usage": 0.23, "whiff_rate": 0.17, "velo": 97.4},
        ]

    def label_box(self, b):
        return (b["label_x"] - 40, b["label_y"] - 12, b["label_x"] + 40, b["label_y"] + 16)

    def test_no_label_overlaps_another_bubble(self):
        layout = charts.movement_layout(self.rows())
        for b in layout["bubbles"]:
            box = self.label_box(b)
            for other in layout["bubbles"]:
                if other is b:
                    continue
                circle = (other["cx"] - other["r"], other["cy"] - other["r"],
                          other["cx"] + other["r"], other["cy"] + other["r"])
                assert not charts._overlaps(box, circle), (b["pitch"], other["pitch"])

    def test_bubble_area_tracks_usage(self):
        """Radius grows with the square root, so twice the usage is twice the
        area, not four times."""
        layout = charts.movement_layout(self.rows())
        ch = next(b for b in layout["bubbles"] if b["pitch"] == "CH")
        si = next(b for b in layout["bubbles"] if b["pitch"] == "SI")
        assert ch["r"] > si["r"]
        assert (ch["r"] - 5) / (si["r"] - 5) == pytest.approx((0.31 / 0.23) ** 0.5, rel=1e-2)

    def test_arm_side_plots_to_the_right(self):
        layout = charts.movement_layout(self.rows())
        ch = next(b for b in layout["bubbles"] if b["pitch"] == "CH")
        assert ch["cx"] > layout["zero_x"]

    def test_league_markers_only_for_pitches_he_throws(self):
        league = [{"pitch_type": "CH", "arm_run": 13.0, "ivb": 8.0},
                  {"pitch_type": "CU", "arm_run": -8.0, "ivb": -10.0}]
        layout = charts.movement_layout(self.rows(), league)
        assert [h["pitch"] for h in layout["league"]] == ["CH"]

    def test_whiff_colour_saturates_at_the_ends(self):
        assert charts.whiff_color(0.0) == charts.whiff_color(charts.WHIFF_LO)
        assert charts.whiff_color(0.9) == charts.whiff_color(charts.WHIFF_HI)


class TestChartHeadlines:
    def test_platoon_names_a_pitch_one_side_never_sees(self):
        # Small gaps on the shared pitches (75/70, 25/22.5), so the never-see
        # clause leads rather than a big gap.
        frame = make_frame([
            {"pitch_type": "FF", "n": 150, "stand": "L"},
            {"pitch_type": "CU", "n": 50, "stand": "L"},
            {"pitch_type": "FF", "n": 140, "stand": "R"},
            {"pitch_type": "CU", "n": 45, "stand": "R"},
            {"pitch_type": "SI", "n": 15, "stand": "R"},
        ])
        text = gameplan.platoon_headline(frame)
        assert text.startswith("Lefties almost never see the sinker")

    def test_platoon_leads_with_a_big_gap(self):
        frame = make_frame([
            {"pitch_type": "SI", "n": 90, "stand": "L"},
            {"pitch_type": "FF", "n": 110, "stand": "L"},
            {"pitch_type": "SI", "n": 30, "stand": "R"},
            {"pitch_type": "FF", "n": 170, "stand": "R"},
        ])
        text = gameplan.platoon_headline(frame)
        assert text.startswith("Lefties get the sinker 45%")

    def test_platoon_says_so_when_nothing_changes(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 100, "stand": s} for s in ("L", "R")
        ] + [
            {"pitch_type": "SL", "n": 100, "stand": s} for s in ("L", "R")
        ])
        assert "barely changes" in gameplan.platoon_headline(frame)

    def test_movement_compares_best_whiff_pitch_to_the_fastball(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 200, "velo": 95.0, "pfx_z": 1.33,
             "is_swing": True, "description": "foul"},
            {"pitch_type": "FS", "n": 100, "velo": 91.0, "pfx_z": 0.15,
             "is_swing": True, "is_whiff": True, "description": "swinging_strike"},
        ])
        text = gameplan.movement_headline(frame)
        assert "splitter" in text and "four-seam" in text
        assert "drops 14 inches more" in text
        assert "only 4 mph slower" in text

    def test_movement_makes_no_tunneling_claim(self):
        """Whether two pitches look alike out of the hand is not measured
        anywhere in this project, so the headline must not say it."""
        frame = make_frame([
            {"pitch_type": "FF", "n": 200, "velo": 95.0, "pfx_z": 1.33},
            {"pitch_type": "FS", "n": 100, "velo": 91.0, "pfx_z": 0.15,
             "is_swing": True, "is_whiff": True, "description": "swinging_strike"},
        ])
        text = gameplan.movement_headline(frame).lower()
        assert "tunnel" not in text and "out of the hand" not in text and "looks like" not in text
