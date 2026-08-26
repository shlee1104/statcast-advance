"""Tests for src/metrics/counts.py.

Hand-constructed frames with small, countable values. Entropy expectations are
worked out longhand in comments so the arithmetic can be checked independently
of the implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.metrics import counts


def make_pitches(specs: list[tuple[str, str, int]]) -> pd.DataFrame:
    """Build a frame from (count, pitch_type, repetitions) tuples."""
    rows = []
    at_bat = 0
    for count_str, pitch_type, n in specs:
        balls, strikes = count_str.split("-")
        for _ in range(n):
            at_bat += 1
            rows.append({
                "game_pk": 1,
                "at_bat_number": at_bat,
                "pitch_number": 1,
                "balls": int(balls),
                "strikes": int(strikes),
                "count": count_str,
                "pitch_type": pitch_type,
                "type": "S",
                "description": "called_strike",
                "is_two_strike": int(strikes) == 2,
                "is_first_pitch": count_str == "0-0",
            })
    return pd.DataFrame(rows)


class TestPitchMix:
    def test_shares_sum_to_one(self):
        frame = make_pitches([("0-0", "FF", 60), ("0-0", "SL", 40)])
        mix = counts.pitch_mix(frame)
        assert mix.sum() == pytest.approx(1.0)

    def test_computes_correct_shares(self):
        frame = make_pitches([("0-0", "FF", 75), ("0-0", "CH", 25)])
        mix = counts.pitch_mix(frame)
        assert mix["FF"] == pytest.approx(0.75)
        assert mix["CH"] == pytest.approx(0.25)

    def test_sorted_descending(self):
        frame = make_pitches([("0-0", "SL", 10), ("0-0", "FF", 50), ("0-0", "CH", 40)])
        mix = counts.pitch_mix(frame)
        assert list(mix.index) == ["FF", "CH", "SL"]

    def test_empty_frame(self):
        frame = make_pitches([("0-0", "FF", 1)]).iloc[0:0]
        assert len(counts.pitch_mix(frame)) == 0


class TestMixByCount:
    def test_rows_sum_to_one(self):
        frame = make_pitches([
            ("0-0", "FF", 30), ("0-0", "SL", 10),
            ("1-2", "SL", 20), ("1-2", "FF", 5),
        ])
        table = counts.mix_by_count(frame)
        for _, row in table.iterrows():
            assert row.sum() == pytest.approx(1.0)

    def test_correct_cell_values(self):
        frame = make_pitches([
            ("0-0", "FF", 30), ("0-0", "SL", 10),   # 75% / 25%
            ("3-1", "FF", 18), ("3-1", "SL", 2),    # 90% / 10%
        ])
        table = counts.mix_by_count(frame)
        assert table.loc["0-0", "FF"] == pytest.approx(0.75)
        assert table.loc["3-1", "FF"] == pytest.approx(0.90)

    def test_rows_in_canonical_order(self):
        frame = make_pitches([
            ("3-2", "FF", 10), ("0-0", "FF", 10), ("1-1", "FF", 10),
        ])
        table = counts.mix_by_count(frame)
        assert list(table.index) == ["0-0", "1-1", "3-2"]

    def test_missing_combinations_are_zero_not_nan(self):
        frame = make_pitches([("0-0", "FF", 10), ("1-1", "SL", 10)])
        table = counts.mix_by_count(frame)
        assert table.loc["0-0", "SL"] == 0.0
        assert not table.isna().any().any()

    def test_min_pitches_gate_excludes_thin_counts(self):
        frame = make_pitches([("0-0", "FF", 50), ("3-0", "FF", 3)])
        table = counts.mix_by_count(frame, min_pitches=20)
        assert "0-0" in table.index
        assert "3-0" not in table.index


class TestFirstPitchTendencies:
    def test_counts_only_first_pitches(self):
        frame = make_pitches([("0-0", "FF", 40), ("1-1", "SL", 100)])
        result = counts.first_pitch_tendencies(frame)
        assert result["n"] == 40

    def test_identifies_primary_pitch(self):
        frame = make_pitches([("0-0", "FF", 30), ("0-0", "CH", 10)])
        result = counts.first_pitch_tendencies(frame)
        assert result["primary_pitch"] == "FF"
        assert result["primary_share"] == pytest.approx(0.75)

    def test_strike_rate_counts_S_and_X(self):
        frame = make_pitches([("0-0", "FF", 10)])
        # 6 strikes, 2 in play (also strikes), 2 balls -> 80%
        frame.loc[0:5, "type"] = "S"
        frame.loc[6:7, "type"] = "X"
        frame.loc[8:9, "type"] = "B"
        result = counts.first_pitch_tendencies(frame)
        assert result["strike_rate"] == pytest.approx(0.80)

    def test_empty_frame_returns_zero_n(self):
        frame = make_pitches([("1-1", "FF", 10)])
        result = counts.first_pitch_tendencies(frame)
        assert result["n"] == 0
        assert result["primary_pitch"] is None


class TestPredictability:
    def test_single_pitch_in_count_is_fully_predictable(self):
        """All fastballs in 3-0 -> entropy 0 -> predictability 1."""
        frame = make_pitches([
            ("3-0", "FF", 25),
            ("0-0", "FF", 20), ("0-0", "SL", 20),  # arsenal is {FF, SL}, k=2
        ])
        table = counts.predictability(frame, min_pitches=20)
        assert table.loc["3-0", "entropy"] == pytest.approx(0.0)
        assert table.loc["3-0", "predictability"] == pytest.approx(1.0)

    def test_uniform_mix_is_maximally_unpredictable(self):
        """50/50 over a 2-pitch arsenal -> H = 1 bit = log2(2) -> pred = 0."""
        frame = make_pitches([("0-0", "FF", 25), ("0-0", "SL", 25)])
        table = counts.predictability(frame, min_pitches=20)
        assert table.loc["0-0", "entropy"] == pytest.approx(1.0)
        assert table.loc["0-0", "predictability"] == pytest.approx(0.0)

    def test_known_entropy_value(self):
        """75/25 split: H = -(0.75*log2 0.75 + 0.25*log2 0.25) = 0.8113 bits.

        Arsenal is {FF, SL} so k=2, log2(2)=1, predictability = 1 - 0.8113.
        """
        frame = make_pitches([("0-0", "FF", 75), ("0-0", "SL", 25)])
        table = counts.predictability(frame, min_pitches=20)
        expected_h = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
        assert table.loc["0-0", "entropy"] == pytest.approx(expected_h, abs=1e-6)
        assert table.loc["0-0", "predictability"] == pytest.approx(1 - expected_h, abs=1e-6)

    def test_k_comes_from_full_arsenal_not_the_count(self):
        """The critical detail.

        In 3-0 he throws only FF. His overall arsenal is 4 pitches, so
        k=4 and log2(4)=2. Entropy in 3-0 is 0, so predictability is
        1 - 0/2 = 1.0. If k were taken from the count itself, k=1 and
        log2(1)=0 would be a division by zero.
        """
        frame = make_pitches([
            ("3-0", "FF", 30),
            ("0-0", "FF", 25), ("0-0", "SL", 25),
            ("1-1", "CH", 25), ("1-1", "CU", 25),
        ])
        table = counts.predictability(frame, min_pitches=20)
        assert table.loc["3-0", "predictability"] == pytest.approx(1.0)
        # 0-0 is a 50/50 over k=4: H=1, max=2, pred = 1 - 0.5 = 0.5
        assert table.loc["0-0", "predictability"] == pytest.approx(0.5)

    def test_reports_top_pitch_and_share(self):
        frame = make_pitches([("1-2", "SL", 30), ("1-2", "FF", 10)])
        table = counts.predictability(frame, min_pitches=20)
        assert table.loc["1-2", "top_pitch"] == "SL"
        assert table.loc["1-2", "top_share"] == pytest.approx(0.75)

    def test_sample_gate_excludes_thin_counts(self):
        frame = make_pitches([("0-0", "FF", 50), ("3-0", "FF", 5)])
        table = counts.predictability(frame, min_pitches=20)
        assert "3-0" not in table.index

    def test_rows_in_canonical_order(self):
        frame = make_pitches([
            ("3-2", "FF", 25), ("0-0", "FF", 25), ("2-1", "FF", 25),
        ])
        table = counts.predictability(frame, min_pitches=20)
        assert list(table.index) == ["0-0", "2-1", "3-2"]

    def test_single_pitch_arsenal_does_not_divide_by_zero(self):
        """A pitcher who throws exactly one pitch type. log2(1) = 0."""
        frame = make_pitches([("0-0", "FF", 50)])
        table = counts.predictability(frame, min_pitches=20)
        assert table.loc["0-0", "predictability"] == pytest.approx(1.0)
        assert np.isfinite(table.loc["0-0", "predictability"])

    def test_empty_frame_returns_empty_with_columns(self):
        frame = make_pitches([("0-0", "FF", 1)]).iloc[0:0]
        table = counts.predictability(frame)
        assert len(table) == 0
        for column in ["n", "entropy", "predictability", "top_pitch", "top_share"]:
            assert column in table.columns
