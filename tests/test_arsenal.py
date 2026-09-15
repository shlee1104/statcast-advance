"""Tests for src/metrics/arsenal.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.metrics import arsenal


def make_pitches(specs: list[dict]) -> pd.DataFrame:
    """Build a frame from (pitch_type, n, **overrides) specifications."""
    rows = []
    at_bat = 0
    for spec in specs:
        for _ in range(spec["n"]):
            at_bat += 1
            row = {
                "game_pk": 1,
                "at_bat_number": at_bat,
                "pitch_number": 1,
                "pitch_type": spec["pitch_type"],
                "release_speed": spec.get("velo", 95.0),
                "release_spin_rate": spec.get("spin", 2300.0),
                "release_pos_x": spec.get("rel_x", 1.5),
                "release_pos_z": spec.get("rel_z", 6.0),
                "release_extension": spec.get("extension", 6.5),
                "arm_angle": spec.get("arm_angle", 45.0),
                "pfx_x": spec.get("pfx_x", 1.0),
                "pfx_z": spec.get("pfx_z", 1.0),
                "zone": spec.get("zone", 5),
                "description": spec.get("description", "called_strike"),
                "estimated_woba_using_speedangle": spec.get("xwoba", 0.300),
                "delta_run_exp": spec.get("run_value", 0.0),
                "is_swing": spec.get("is_swing", False),
                "is_whiff": spec.get("is_whiff", False),
                "is_in_zone": spec.get("zone", 5) in range(1, 10),
            }
            rows.append(row)
    return pd.DataFrame(rows)


class TestProfile:
    def test_usage_sums_to_one(self):
        frame = make_pitches([
            {"pitch_type": "FF", "n": 60},
            {"pitch_type": "SL", "n": 40},
        ])
        table = arsenal.profile(frame)
        assert table["usage"].sum() == pytest.approx(1.0)

    def test_whiff_denominator_is_swings_not_pitches(self):
        """20 swings out of 100 pitches, 10 of them missed -> 50%, not 10%."""
        frame = make_pitches([
            {"pitch_type": "FF", "n": 10, "is_swing": True, "is_whiff": True},
            {"pitch_type": "FF", "n": 10, "is_swing": True, "is_whiff": False},
            {"pitch_type": "FF", "n": 80, "is_swing": False, "is_whiff": False},
        ])
        table = arsenal.profile(frame)
        assert table.loc["FF", "whiff_rate"] == pytest.approx(0.50)

    def test_movement_converted_to_inches(self):
        frame = make_pitches([{"pitch_type": "FF", "n": 50, "pfx_z": 1.5}])
        table = arsenal.profile(frame)
        assert table.loc["FF", "v_break"] == pytest.approx(18.0)

    def test_velo_max_is_a_percentile_not_the_maximum(self):
        """One mis-tracked 105 mph reading must not define the top of the range."""
        frame = make_pitches([
            {"pitch_type": "FF", "n": 99, "velo": 95.0},
            {"pitch_type": "FF", "n": 1, "velo": 105.0},
        ])
        table = arsenal.profile(frame)
        assert table.loc["FF", "velo_max"] < 100.0

    def test_chase_denominator_is_out_of_zone_pitches(self):
        """4 of 8 out-of-zone pitches swung at -> 50%, regardless of in-zone."""
        frame = make_pitches([
            {"pitch_type": "FF", "n": 4, "zone": 13, "is_swing": True},
            {"pitch_type": "FF", "n": 4, "zone": 13, "is_swing": False},
            {"pitch_type": "FF", "n": 42, "zone": 5, "is_swing": True},
        ])
        table = arsenal.profile(frame)
        assert table.loc["FF", "chase_rate"] == pytest.approx(0.50)

    def test_excludes_pitch_types_below_the_arsenal_gate(self):
        frame = make_pitches([
            {"pitch_type": "FF", "n": 500},
            {"pitch_type": "SL", "n": 500},
            {"pitch_type": "ST", "n": 6},
        ])
        table = arsenal.profile(frame)
        assert "ST" not in table.index

    def test_sorted_by_usage_descending(self):
        frame = make_pitches([
            {"pitch_type": "SL", "n": 20},
            {"pitch_type": "FF", "n": 60},
            {"pitch_type": "CH", "n": 40},
        ])
        table = arsenal.profile(frame)
        assert list(table.index) == ["FF", "CH", "SL"]

    def test_empty_frame(self):
        frame = make_pitches([{"pitch_type": "FF", "n": 1}]).iloc[0:0]
        assert len(arsenal.profile(frame)) == 0


class TestReleaseConsistency:
    def test_reference_pitch_has_zero_offset(self):
        frame = make_pitches([
            {"pitch_type": "FF", "n": 60, "rel_z": 6.0},
            {"pitch_type": "SL", "n": 40, "rel_z": 5.8},
        ])
        table = arsenal.release_consistency(frame)
        assert table.loc["FF", "dist_from_ref"] == pytest.approx(0.0)

    def test_detects_a_lower_release_slot(self):
        """A slider released 0.2 ft below the fastball is 2.4 inches off."""
        frame = make_pitches([
            {"pitch_type": "FF", "n": 60, "rel_z": 6.0},
            {"pitch_type": "SL", "n": 40, "rel_z": 5.8},
        ])
        table = arsenal.release_consistency(frame)
        assert table.loc["SL", "dz_from_ref"] == pytest.approx(-2.4, abs=1e-6)
        assert table.loc["SL", "dist_from_ref"] == pytest.approx(2.4, abs=1e-6)

    def test_consistent_slot_has_near_zero_deviation(self):
        frame = make_pitches([{"pitch_type": "FF", "n": 50, "rel_x": 1.5}])
        table = arsenal.release_consistency(frame)
        assert table.loc["FF", "rel_x_sd"] == pytest.approx(0.0, abs=1e-9)

    def test_erratic_slot_shows_larger_deviation(self):
        frame = pd.concat([
            make_pitches([{"pitch_type": "SL", "n": 25, "rel_x": 1.0}]),
            make_pitches([{"pitch_type": "SL", "n": 25, "rel_x": 2.0}]),
            make_pitches([{"pitch_type": "FF", "n": 50, "rel_x": 1.5}]),
        ])
        table = arsenal.release_consistency(frame)
        assert table.loc["SL", "rel_x_sd"] > table.loc["FF", "rel_x_sd"]

    def test_reference_is_the_most_used_pitch(self):
        frame = make_pitches([
            {"pitch_type": "SL", "n": 70, "rel_z": 5.5},
            {"pitch_type": "FF", "n": 30, "rel_z": 6.0},
        ])
        table = arsenal.release_consistency(frame)
        assert table.index[0] == "SL"
        assert table.loc["SL", "dist_from_ref"] == pytest.approx(0.0)

    def test_empty_frame(self):
        frame = make_pitches([{"pitch_type": "FF", "n": 1}]).iloc[0:0]
        assert len(arsenal.release_consistency(frame)) == 0
