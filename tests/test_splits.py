"""Tests for src/metrics/splits.py."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.metrics import splits


def make_pitches(specs: list[dict]) -> pd.DataFrame:
    """Build a frame from specification dicts, one group per spec."""
    rows = []
    at_bat = 0
    for spec in specs:
        for i in range(spec["n"]):
            at_bat += 1
            rows.append({
                "game_pk": spec.get("game_pk", 1),
                "at_bat_number": at_bat,
                "pitch_number": 1,
                "pitch_type": spec["pitch_type"],
                "stand": spec.get("stand", "R"),
                "release_speed": spec.get("velo", 95.0),
                "zone": spec.get("zone", 5),
                "estimated_woba_using_speedangle": spec.get("xwoba", 0.300),
                "n_thruorder_pitcher": spec.get("tto", 1),
                "is_swing": spec.get("is_swing", False),
                "is_whiff": spec.get("is_whiff", False),
                "is_in_zone": spec.get("zone", 5) in range(1, 10),
            })
    return pd.DataFrame(rows)


class TestPlatoon:
    def test_usage_normalized_within_each_side(self):
        """Usage answers 'what does he throw to lefties', not 'what share of
        all his pitches were sliders to lefties'."""
        frame = make_pitches([
            {"pitch_type": "FF", "n": 30, "stand": "L"},
            {"pitch_type": "SL", "n": 10, "stand": "L"},
            {"pitch_type": "FF", "n": 100, "stand": "R"},
            {"pitch_type": "SL", "n": 100, "stand": "R"},
        ])
        table = splits.platoon(frame)
        assert table.loc[("L", "FF"), "usage"] == pytest.approx(0.75)
        assert table.loc[("R", "FF"), "usage"] == pytest.approx(0.50)

    def test_each_side_sums_to_one(self):
        frame = make_pitches([
            {"pitch_type": "FF", "n": 30, "stand": "L"},
            {"pitch_type": "SL", "n": 10, "stand": "L"},
            {"pitch_type": "FF", "n": 60, "stand": "R"},
        ])
        table = splits.platoon(frame).reset_index()
        for _, side in table.groupby("stand"):
            assert side["usage"].sum() == pytest.approx(1.0)

    def test_empty_frame(self):
        frame = make_pitches([{"pitch_type": "FF", "n": 1}]).iloc[0:0]
        assert len(splits.platoon(frame)) == 0


class TestPlatoonGaps:
    def test_identifies_a_pitch_withheld_from_one_side(self):
        """A slider thrown 40% to righties and 5% to lefties is a weapon
        lefties have not had to learn."""
        frame = make_pitches([
            {"pitch_type": "FF", "n": 95, "stand": "L"},
            {"pitch_type": "SL", "n": 5, "stand": "L"},
            {"pitch_type": "FF", "n": 60, "stand": "R"},
            {"pitch_type": "SL", "n": 40, "stand": "R"},
        ])
        table = splits.platoon_gaps(frame)
        slider = table[table["pitch_type"] == "SL"].iloc[0]
        assert slider["gap"] == pytest.approx(0.05 - 0.40, abs=1e-6)

    def test_sorted_by_absolute_gap(self):
        frame = make_pitches([
            {"pitch_type": "FF", "n": 95, "stand": "L"},
            {"pitch_type": "SL", "n": 5, "stand": "L"},
            {"pitch_type": "FF", "n": 60, "stand": "R"},
            {"pitch_type": "SL", "n": 40, "stand": "R"},
        ])
        table = splits.platoon_gaps(frame)
        gaps = [abs(g) for g in table["gap"]]
        assert gaps == sorted(gaps, reverse=True)

    def test_a_withheld_pitch_keeps_its_gap(self):
        """The usage rate's denominator is every pitch to that side, not the
        handful of that pitch type, so a pitch shown 5 times out of 200 has a
        precisely measured 2.5% usage. Gating on the 5 would delete the
        finding."""
        frame = make_pitches([
            {"pitch_type": "FF", "n": 195, "stand": "L"},
            {"pitch_type": "SL", "n": 5, "stand": "L"},
            {"pitch_type": "FF", "n": 120, "stand": "R"},
            {"pitch_type": "SL", "n": 80, "stand": "R"},
        ])
        table = splits.platoon_gaps(frame)
        slider = table[table["pitch_type"] == "SL"].iloc[0]
        assert slider["n_L"] == 5
        assert slider["gap"] == pytest.approx(0.025 - 0.40, abs=1e-6)

    def test_thin_side_loses_its_whiff_rate_but_not_its_row(self):
        """A whiff rate over 5 pitches is noise; the usage gap beside it is
        not. Blank the cell, keep the row."""
        frame = make_pitches([
            {"pitch_type": "FF", "n": 195, "stand": "L"},
            {"pitch_type": "SL", "n": 5, "stand": "L", "is_swing": True,
             "is_whiff": True},
            {"pitch_type": "FF", "n": 120, "stand": "R"},
            {"pitch_type": "SL", "n": 80, "stand": "R", "is_swing": True,
             "is_whiff": True},
        ])
        table = splits.platoon_gaps(frame)
        slider = table[table["pitch_type"] == "SL"].iloc[0]
        assert math.isnan(slider["whiff_L"])
        assert slider["whiff_R"] == pytest.approx(1.0)
        assert not math.isnan(slider["gap"])

    def test_refuses_the_comparison_when_a_side_is_barely_faced(self):
        """Against 8 left-handers all season there is no left-handed usage
        rate to report, whatever the arithmetic says."""
        frame = make_pitches([
            {"pitch_type": "FF", "n": 8, "stand": "L"},
            {"pitch_type": "FF", "n": 300, "stand": "R"},
            {"pitch_type": "SL", "n": 200, "stand": "R"},
        ])
        assert len(splits.platoon_gaps(frame)) == 0

    def test_rate_gate_is_adjustable(self):
        frame = make_pitches([
            {"pitch_type": "FF", "n": 195, "stand": "L"},
            {"pitch_type": "SL", "n": 5, "stand": "L", "is_swing": True,
             "is_whiff": True},
            {"pitch_type": "FF", "n": 120, "stand": "R"},
            {"pitch_type": "SL", "n": 80, "stand": "R"},
        ])
        table = splits.platoon_gaps(frame, min_rate_pitches=1)
        slider = table[table["pitch_type"] == "SL"].iloc[0]
        assert slider["whiff_L"] == pytest.approx(1.0)


class TestMissingColumnIsNotThinData:
    """A missing feed column and a thin sample both produce no result, and
    they call for opposite responses. Saying "insufficient data" for both hid
    a caching bug behind a plausible message across 831 batters faced."""

    def test_says_so_when_the_column_is_absent(self):
        frame = make_pitches([{"pitch_type": "FF", "n": 400, "tto": 1}])
        frame = frame.drop(columns=["n_thruorder_pitcher"])
        result = splits.decompose_tto(frame)
        assert result["attribution"] == "unknown"
        assert "n_thruorder_pitcher" in result["note"]
        assert "re-fetch" in result["note"].lower()

    def test_still_says_insufficient_when_the_data_is_merely_thin(self):
        frame = make_pitches([{"pitch_type": "FF", "n": 20, "tto": 1}])
        result = splits.decompose_tto(frame)
        assert result["note"] == "insufficient data"


class TestFatigue:
    def test_buckets_by_pitch_count_within_the_outing(self):
        frame = make_pitches([{"pitch_type": "FF", "n": 45}])
        table = splits.fatigue(frame, bucket_size=15)
        assert list(table["bucket"]) == ["1-15", "16-30", "31-45"]

    def test_velo_delta_is_relative_to_the_first_bucket(self):
        frame = pd.concat([
            make_pitches([{"pitch_type": "FF", "n": 15, "velo": 96.0}]),
            make_pitches([{"pitch_type": "FF", "n": 15, "velo": 94.0}]),
        ]).reset_index(drop=True)
        frame["at_bat_number"] = range(1, len(frame) + 1)
        table = splits.fatigue(frame, bucket_size=15)
        assert table.iloc[0]["velo_delta"] == pytest.approx(0.0)
        assert table.iloc[1]["velo_delta"] == pytest.approx(-2.0)

    def test_excludes_offspeed_by_default(self):
        """Averaging a curveball into a velocity series tracks pitch selection,
        not fatigue."""
        frame = make_pitches([
            {"pitch_type": "FF", "n": 10, "velo": 96.0},
            {"pitch_type": "CU", "n": 10, "velo": 78.0},
        ])
        table = splits.fatigue(frame, bucket_size=100)
        assert table.iloc[0]["velo"] == pytest.approx(96.0)

    def test_counts_distinct_outings(self):
        frame = pd.concat([
            make_pitches([{"pitch_type": "FF", "n": 10, "game_pk": 1}]),
            make_pitches([{"pitch_type": "FF", "n": 10, "game_pk": 2}]),
        ])
        table = splits.fatigue(frame, bucket_size=100)
        assert table.iloc[0]["outings"] == 2

    def test_empty_frame(self):
        frame = make_pitches([{"pitch_type": "FF", "n": 1}]).iloc[0:0]
        assert len(splits.fatigue(frame)) == 0


class TestTimesThroughOrder:
    def test_groups_by_trip(self):
        frame = make_pitches([
            {"pitch_type": "FF", "n": 30, "tto": 1},
            {"pitch_type": "FF", "n": 30, "tto": 2},
            {"pitch_type": "FF", "n": 30, "tto": 3},
        ])
        table = splits.times_through_order(frame)
        assert list(table["tto"]) == [1, 2, 3]

    def test_missing_column_returns_empty(self):
        frame = make_pitches([{"pitch_type": "FF", "n": 10}])
        frame = frame.drop(columns=["n_thruorder_pitcher"])
        assert len(splits.times_through_order(frame)) == 0


class TestDecomposeTTO:
    def test_detects_fatigue_when_velocity_drops(self):
        frame = make_pitches([
            {"pitch_type": "FF", "n": 40, "tto": 1, "velo": 96.0, "xwoba": 0.300},
            {"pitch_type": "FF", "n": 40, "tto": 3, "velo": 94.0, "xwoba": 0.340},
        ])
        result = splits.decompose_tto(frame)
        assert result["velo_delta"] == pytest.approx(-2.0)
        assert result["attribution"] == "both"

    def test_reports_results_decay_without_claiming_a_cause(self):
        """Velocity steady while results worsen used to be reported as
        "familiarity" — hitters solving him. This design cannot support that:
        the third trip is also the 70th-to-100th pitch, so trip number and
        pitch count are nearly collinear and a marginal comparison cannot
        separate them. The output describes what moved, not why.
        """
        frame = make_pitches([
            {"pitch_type": "FF", "n": 40, "tto": 1, "velo": 96.0, "xwoba": 0.280},
            {"pitch_type": "FF", "n": 40, "tto": 3, "velo": 95.9, "xwoba": 0.360},
        ])
        result = splits.decompose_tto(frame)
        assert result["attribution"] == "results_decay"
        assert "solving" not in result["note"]
        assert "familiarity" not in result["note"].lower()

    def test_never_prints_minus_zero(self):
        """Nine reports read "velocity holds (-0.0 mph)"."""
        frame = make_pitches([
            {"pitch_type": "FF", "n": 40, "tto": 1, "velo": 96.02, "xwoba": 0.280},
            {"pitch_type": "FF", "n": 40, "tto": 3, "velo": 96.0, "xwoba": 0.360},
        ])
        note = splits.decompose_tto(frame)["note"]
        assert "-0.0" not in note
        assert "velocity holds (+0.0 mph)" in note

    def test_does_not_say_holds_about_a_near_one_mph_drop(self):
        """Wheeler: "velocity holds (-0.9 mph)" contradicted itself."""
        frame = make_pitches([
            {"pitch_type": "FF", "n": 40, "tto": 1, "velo": 96.0, "xwoba": 0.280},
            {"pitch_type": "FF", "n": 40, "tto": 3, "velo": 95.1, "xwoba": 0.360},
        ])
        note = splits.decompose_tto(frame)["note"]
        assert "holds" not in note
        assert "velocity dips slightly (-0.9 mph)" in note

    def test_has_a_plain_headline_not_a_code(self):
        frame = make_pitches([
            {"pitch_type": "FF", "n": 40, "tto": 1, "velo": 96.0, "xwoba": 0.280},
            {"pitch_type": "FF", "n": 40, "tto": 3, "velo": 95.9, "xwoba": 0.360},
        ])
        assert splits.decompose_tto(frame)["headline"] == "Third time through: hit harder"

    def test_carries_the_collinearity_caveat(self):
        frame = make_pitches([
            {"pitch_type": "FF", "n": 40, "tto": 1, "velo": 96.0, "xwoba": 0.280},
            {"pitch_type": "FF", "n": 40, "tto": 3, "velo": 95.9, "xwoba": 0.360},
        ])
        caveat = splits.decompose_tto(frame)["caveat"]
        assert "pitch count" in caveat
        assert "cannot say which" in caveat

    def test_reports_neither_when_nothing_moves(self):
        frame = make_pitches([
            {"pitch_type": "FF", "n": 40, "tto": 1, "velo": 96.0, "xwoba": 0.300},
            {"pitch_type": "FF", "n": 40, "tto": 3, "velo": 96.0, "xwoba": 0.300},
        ])
        assert splits.decompose_tto(frame)["attribution"] == "neither"

    def test_a_noise_sized_xwoba_change_is_not_decay(self):
        """The old version printed an xwOBA change of +0.004 alongside a
        velocity claim as if it were corroboration. It is noise."""
        frame = make_pitches([
            {"pitch_type": "FF", "n": 40, "tto": 1, "velo": 96.0, "xwoba": 0.300},
            {"pitch_type": "FF", "n": 40, "tto": 3, "velo": 96.0, "xwoba": 0.304},
        ])
        assert splits.decompose_tto(frame)["attribution"] == "neither"

    def test_missing_third_trip_returns_unknown(self):
        frame = make_pitches([
            {"pitch_type": "FF", "n": 40, "tto": 1, "velo": 96.0},
        ])
        result = splits.decompose_tto(frame)
        assert result["attribution"] == "unknown"
        assert math.isnan(result["velo_delta"])
