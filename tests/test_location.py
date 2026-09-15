"""Tests for src/metrics/location.py.

The sign conventions are the whole risk in this module — a flipped axis
produces a table that looks entirely reasonable and describes the opposite
half of the plate. Every normalization is therefore tested against both
pitcher handednesses and both batter handednesses, with values chosen so the
expected answer can be read off by hand.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.metrics import location


def make_pitches(specs: list[dict]) -> pd.DataFrame:
    """Build a frame from specification dicts, one group per spec.

    Defaults put the pitch dead center of a two-foot strike zone: sz_bot 1.5,
    sz_top 3.5, so plate_z 2.5 normalizes to exactly 0.5.
    """
    rows = []
    at_bat = 0
    for spec in specs:
        for _ in range(spec["n"]):
            at_bat += 1
            rows.append({
                "game_pk": 1,
                "at_bat_number": at_bat,
                "pitch_number": 1,
                "pitch_type": spec["pitch_type"],
                "p_throws": spec.get("p_throws", "R"),
                "stand": spec.get("stand", "R"),
                "plate_x": spec.get("plate_x", 0.0),
                "plate_z": spec.get("plate_z", 2.5),
                "sz_top": spec.get("sz_top", 3.5),
                "sz_bot": spec.get("sz_bot", 1.5),
                "zone": spec.get("zone", 5),
                "is_in_zone": spec.get("zone", 5) in range(1, 10),
            })
    return pd.DataFrame(rows)


class TestHorizontalNormalization:
    def test_right_hander_arm_side_is_negative_plate_x(self):
        """Established from hit-by-pitch locations: a righty's arm side is the
        right-handed batter's box, which sits at negative plate_x."""
        frame = make_pitches([
            {"pitch_type": "FF", "n": 10, "p_throws": "R", "plate_x": -0.5},
        ])
        result = location.add_location_frame(frame)
        assert result["x_arm"].iloc[0] == pytest.approx(0.5)
        assert result["h_band"].iloc[0] == "ARM"

    def test_left_hander_arm_side_is_mirrored(self):
        """The same physical location is the opposite side of the rubber for a
        lefty, so the sign must flip with p_throws."""
        frame = make_pitches([
            {"pitch_type": "FF", "n": 10, "p_throws": "L", "plate_x": -0.5},
        ])
        result = location.add_location_frame(frame)
        assert result["x_arm"].iloc[0] == pytest.approx(-0.5)
        assert result["h_band"].iloc[0] == "GLOVE"

    def test_inside_is_relative_to_the_batter_not_the_pitcher(self):
        """A pitch at one plate_x is inside to a righty and away from a lefty,
        regardless of who threw it."""
        frame = make_pitches([
            {"pitch_type": "FF", "n": 5, "stand": "R", "plate_x": -0.5},
            {"pitch_type": "SL", "n": 5, "stand": "L", "plate_x": -0.5},
        ])
        result = location.add_location_frame(frame)
        righty = result[result["stand"] == "R"].iloc[0]
        lefty = result[result["stand"] == "L"].iloc[0]
        assert righty["x_in"] == pytest.approx(0.5)
        assert lefty["x_in"] == pytest.approx(-0.5)

    def test_middle_of_the_plate_is_its_own_band(self):
        frame = make_pitches([{"pitch_type": "FF", "n": 10, "plate_x": 0.0}])
        result = location.add_location_frame(frame)
        assert result["h_band"].iloc[0] == "MIDDLE"


class TestVerticalNormalization:
    def test_height_is_a_fraction_of_the_batters_own_zone(self):
        """Two batters of different heights, one pitch height each, both at the
        vertical middle of their own zone."""
        frame = make_pitches([
            {"pitch_type": "FF", "n": 5, "plate_z": 2.5, "sz_bot": 1.5, "sz_top": 3.5},
            {"pitch_type": "SL", "n": 5, "plate_z": 2.5, "sz_bot": 1.0, "sz_top": 4.0},
        ])
        result = location.add_location_frame(frame)
        assert result["z_norm"].iloc[0] == pytest.approx(0.5)
        assert result["z_norm"].iloc[-1] == pytest.approx(0.5)

    def test_zero_is_the_bottom_of_the_zone_and_one_the_top(self):
        frame = make_pitches([
            {"pitch_type": "FF", "n": 5, "plate_z": 1.5},
            {"pitch_type": "SL", "n": 5, "plate_z": 3.5},
        ])
        result = location.add_location_frame(frame)
        assert result["z_norm"].iloc[0] == pytest.approx(0.0)
        assert result["z_norm"].iloc[-1] == pytest.approx(1.0)

    def test_bands_split_the_zone_into_thirds(self):
        frame = make_pitches([
            {"pitch_type": "FF", "n": 5, "plate_z": 3.3},   # 0.90 -> UP
            {"pitch_type": "SL", "n": 5, "plate_z": 2.5},   # 0.50 -> MIDDLE
            {"pitch_type": "CU", "n": 5, "plate_z": 1.7},   # 0.10 -> DOWN
        ])
        result = location.add_location_frame(frame)
        bands = result.groupby("pitch_type")["v_band"].first()
        assert bands["FF"] == "UP"
        assert bands["SL"] == "MIDDLE"
        assert bands["CU"] == "DOWN"

    def test_a_degenerate_zone_yields_no_height(self):
        """sz_top equal to sz_bot would divide by zero; it must produce a
        missing value rather than an infinity that poisons every average."""
        frame = make_pitches([
            {"pitch_type": "FF", "n": 10, "sz_bot": 2.0, "sz_top": 2.0},
        ])
        result = location.add_location_frame(frame)
        assert math.isnan(result["z_norm"].iloc[0])
        assert result["v_band"].iloc[0] == location.UNKNOWN


class TestQuadrants:
    def test_assigns_the_expected_corner(self):
        frame = make_pitches([
            {"pitch_type": "FF", "n": 5, "p_throws": "R",
             "plate_x": -0.6, "plate_z": 3.2},
        ])
        result = location.add_location_frame(frame)
        assert result["quadrant"].iloc[0] == "UP-ARM"

    def test_missing_coordinates_are_not_binned_as_middle(self):
        """Savant fails to locate a few pitches a season. Assigning them a
        quadrant would quietly bias every share in the table."""
        frame = make_pitches([{"pitch_type": "FF", "n": 10}])
        frame.loc[0, "plate_x"] = float("nan")
        result = location.add_location_frame(frame)
        assert result["quadrant"].iloc[0] == location.UNKNOWN

    def test_shares_sum_to_one_per_pitch_type(self):
        frame = make_pitches([
            {"pitch_type": "FF", "n": 30, "plate_x": -0.6, "plate_z": 3.2},
            {"pitch_type": "SL", "n": 30, "plate_x": 0.6, "plate_z": 1.7},
        ])
        table = location.quadrant_shares(frame)
        for _, row in table.iterrows():
            assert row.sum() == pytest.approx(1.0)

    def test_all_four_quadrants_are_columns_even_when_unused(self):
        """The report renders a fixed grid, so the table's shape must not
        depend on where this particular pitcher happened to throw."""
        frame = make_pitches([
            {"pitch_type": "FF", "n": 20, "plate_x": -0.6, "plate_z": 3.2},
        ])
        table = location.quadrant_shares(frame)
        assert list(table.columns) == location.QUADRANTS
        assert table.loc["FF", "UP-ARM"] == pytest.approx(1.0)
        assert table.loc["FF", "DOWN-GLOVE"] == pytest.approx(0.0)


class TestLocationProfile:
    def test_reports_spread_separately_from_the_mean(self):
        """Two pitches with the same average location, one commanded and one
        not, must be distinguishable."""
        tight = make_pitches([
            {"pitch_type": "FF", "n": 10, "plate_x": -0.4},
            {"pitch_type": "FF", "n": 10, "plate_x": -0.6},
        ])
        loose = make_pitches([
            {"pitch_type": "SL", "n": 10, "plate_x": 0.5},
            {"pitch_type": "SL", "n": 10, "plate_x": -1.5},
        ])
        frame = pd.concat([tight, loose]).reset_index(drop=True)
        table = location.location_profile(frame)
        assert table.loc["FF", "arm_side"] == pytest.approx(6.0)
        assert table.loc["SL", "arm_side"] == pytest.approx(6.0)
        assert table.loc["SL", "arm_side_sd"] > table.loc["FF", "arm_side_sd"]

    def test_top_quadrant_is_the_most_used(self):
        frame = make_pitches([
            {"pitch_type": "FS", "n": 30, "plate_x": -0.6, "plate_z": 1.7},
            {"pitch_type": "FS", "n": 10, "plate_x": 0.6, "plate_z": 3.2},
        ])
        table = location.location_profile(frame)
        assert table.loc["FS", "top_quadrant"] == "DOWN-ARM"
        assert table.loc["FS", "top_share"] == pytest.approx(0.75)

    def test_empty_frame(self):
        frame = make_pitches([{"pitch_type": "FF", "n": 1}]).iloc[0:0]
        assert len(location.location_profile(frame)) == 0


class TestLocationLeaks:
    def test_fires_above_the_threshold(self):
        frame = make_pitches([
            {"pitch_type": "SL", "n": 80, "plate_x": 0.6, "plate_z": 1.7},
            {"pitch_type": "SL", "n": 20, "plate_x": -0.6, "plate_z": 3.2},
        ])
        leaks = location.location_leaks(frame, min_pitches=40, min_share=0.40)
        assert len(leaks) == 1
        assert leaks.iloc[0]["quadrant"] == "DOWN-GLOVE"
        assert leaks.iloc[0]["share"] == pytest.approx(0.80)

    def test_excess_is_measured_against_a_uniform_quarter(self):
        frame = make_pitches([
            {"pitch_type": "SL", "n": 50, "plate_x": 0.6, "plate_z": 1.7},
            {"pitch_type": "SL", "n": 50, "plate_x": -0.6, "plate_z": 3.2},
        ])
        leaks = location.location_leaks(frame, min_pitches=40, min_share=0.40)
        assert leaks.iloc[0]["excess"] == pytest.approx(0.25)

    def test_stays_silent_below_the_threshold(self):
        """An evenly distributed pitch is not leaking anything."""
        frame = make_pitches([
            {"pitch_type": "FF", "n": 25, "plate_x": 0.6, "plate_z": 1.7},
            {"pitch_type": "FF", "n": 25, "plate_x": 0.6, "plate_z": 3.2},
            {"pitch_type": "FF", "n": 25, "plate_x": -0.6, "plate_z": 1.7},
            {"pitch_type": "FF", "n": 25, "plate_x": -0.6, "plate_z": 3.2},
        ])
        assert len(location.location_leaks(frame, min_pitches=40, min_share=0.40)) == 0

    def test_sample_gate_suppresses_a_thin_pitch(self):
        """A pitch thrown 10 times lands in one quadrant easily, and means
        nothing by doing so."""
        frame = make_pitches([
            {"pitch_type": "FF", "n": 200, "plate_x": 0.0, "plate_z": 2.5},
            {"pitch_type": "SL", "n": 10, "plate_x": 0.6, "plate_z": 1.7},
        ])
        leaks = location.location_leaks(frame, min_pitches=40, min_share=0.40)
        assert "SL" not in list(leaks["pitch_type"])

    def test_empty_frame(self):
        frame = make_pitches([{"pitch_type": "FF", "n": 1}]).iloc[0:0]
        assert len(location.location_leaks(frame)) == 0


class TestZoneTable:
    def test_columns_are_the_thirteen_savant_zones(self):
        frame = make_pitches([{"pitch_type": "FF", "n": 20, "zone": 5}])
        table = location.zone_table(frame)
        assert list(table.columns) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]

    def test_rows_sum_to_one(self):
        frame = make_pitches([
            {"pitch_type": "FF", "n": 20, "zone": 5},
            {"pitch_type": "FF", "n": 10, "zone": 13},
            {"pitch_type": "SL", "n": 30, "zone": 14},
        ])
        table = location.zone_table(frame)
        for _, row in table.iterrows():
            assert row.sum() == pytest.approx(1.0)

    def test_shares_are_within_pitch_type(self):
        frame = make_pitches([
            {"pitch_type": "FF", "n": 30, "zone": 5},
            {"pitch_type": "FF", "n": 10, "zone": 11},
            {"pitch_type": "SL", "n": 100, "zone": 14},
        ])
        table = location.zone_table(frame)
        assert table.loc["FF", 5] == pytest.approx(0.75)
        assert table.loc["SL", 14] == pytest.approx(1.0)

    def test_empty_frame(self):
        frame = make_pitches([{"pitch_type": "FF", "n": 1}]).iloc[0:0]
        assert len(location.zone_table(frame)) == 0
