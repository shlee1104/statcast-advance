"""Tests for src/metrics/sequencing.py.

Frames are built from explicit plate appearances so expected transition counts
can be worked out by hand.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.metrics import sequencing


def make_at_bats(
    at_bats: list[list[str]],
    descriptions: list[list[str]] | None = None,
    events: list[str] | None = None,
    strikes: list[list[int]] | None = None,
) -> pd.DataFrame:
    """Build a frame from a list of plate appearances.

    Each plate appearance is a list of pitch types in the order thrown.
    """
    rows = []
    for pa_index, pitches in enumerate(at_bats, start=1):
        for pitch_index, pitch_type in enumerate(pitches, start=1):
            description = (
                descriptions[pa_index - 1][pitch_index - 1]
                if descriptions else "called_strike"
            )
            strike_count = (
                strikes[pa_index - 1][pitch_index - 1]
                if strikes else 0
            )
            is_last = pitch_index == len(pitches)
            rows.append({
                "game_pk": 1,
                "at_bat_number": pa_index,
                "pitch_number": pitch_index,
                "pitch_type": pitch_type,
                "description": description,
                "events": (events[pa_index - 1] if events and is_last else None),
                "strikes": strike_count,
                "is_two_strike": strike_count == 2,
                "is_swing": description in {
                    "swinging_strike", "foul", "hit_into_play", "foul_tip",
                },
                "is_whiff": description == "swinging_strike",
                "is_contact": description in {"foul", "hit_into_play", "foul_tip"},
            })
    return pd.DataFrame(rows)


class TestTransitionMatrix:
    def test_simple_transitions(self):
        """FF is followed by SL twice and CH once -> 2/3 and 1/3."""
        frame = make_at_bats([["FF", "SL"], ["FF", "SL"], ["FF", "CH"]])
        matrix = sequencing.transition_matrix(frame)
        assert matrix.loc["FF", "SL"] == pytest.approx(2 / 3)
        assert matrix.loc["FF", "CH"] == pytest.approx(1 / 3)

    def test_rows_sum_to_one(self):
        frame = make_at_bats([["FF", "SL", "CH"], ["SL", "FF"], ["CH", "CH"]])
        matrix = sequencing.transition_matrix(frame)
        for _, row in matrix.iterrows():
            assert row.sum() == pytest.approx(1.0)

    def test_does_not_pair_across_plate_appearances(self):
        """Two single-pitch PAs contain zero transitions, not one.

        If this fails, the code is treating the end of one PA as setting up
        the start of the next, which fabricates data.
        """
        frame = make_at_bats([["FF"], ["SL"]])
        matrix = sequencing.transition_matrix(frame)
        assert matrix.empty or matrix.sum().sum() == 0

    def test_last_pitch_of_pa_has_no_successor(self):
        """One 3-pitch PA yields exactly 2 transitions, not 3."""
        frame = make_at_bats([["FF", "SL", "CH"]])
        matrix = sequencing.transition_matrix(frame)
        # FF->SL and SL->CH. CH is terminal.
        assert matrix.loc["FF", "SL"] == pytest.approx(1.0)
        assert matrix.loc["SL", "CH"] == pytest.approx(1.0)
        assert "CH" not in matrix.index

    def test_respects_pitch_number_order_not_row_order(self):
        """Shuffled rows must produce the same answer."""
        frame = make_at_bats([["FF", "SL", "CH"]])
        shuffled = frame.iloc[::-1].reset_index(drop=True)
        matrix = sequencing.transition_matrix(shuffled)
        assert matrix.loc["FF", "SL"] == pytest.approx(1.0)

    def test_back_to_back_same_pitch(self):
        frame = make_at_bats([["SL", "SL"], ["SL", "SL"], ["SL", "FF"]])
        matrix = sequencing.transition_matrix(frame)
        assert matrix.loc["SL", "SL"] == pytest.approx(2 / 3)

    def test_missing_combinations_are_zero(self):
        frame = make_at_bats([["FF", "SL"]])
        matrix = sequencing.transition_matrix(frame)
        assert not matrix.isna().any().any()

    def test_min_transitions_gate(self):
        frame = make_at_bats([
            ["FF", "SL"], ["FF", "SL"], ["FF", "SL"],
            ["CU", "CH"],
        ])
        matrix = sequencing.transition_matrix(frame, min_transitions=3)
        assert "FF" in matrix.index
        assert "CU" not in matrix.index

    def test_empty_frame(self):
        frame = make_at_bats([["FF", "SL"]]).iloc[0:0]
        matrix = sequencing.transition_matrix(frame)
        assert len(matrix) == 0


class TestTransitionAfterOutcome:
    def test_conditions_on_whiff(self):
        """Only the PA where the first pitch was whiffed should count."""
        frame = make_at_bats(
            [["SL", "SL"], ["SL", "FF"]],
            descriptions=[
                ["swinging_strike", "called_strike"],
                ["called_strike", "called_strike"],
            ],
        )
        matrix = sequencing.transition_after_outcome(frame, "whiff")
        assert matrix.loc["SL", "SL"] == pytest.approx(1.0)
        assert matrix.sum().sum() == pytest.approx(1.0)

    def test_conditions_on_take(self):
        frame = make_at_bats(
            [["FF", "SL"], ["FF", "CH"]],
            descriptions=[
                ["ball", "called_strike"],
                ["swinging_strike", "called_strike"],
            ],
        )
        matrix = sequencing.transition_after_outcome(frame, "take")
        assert matrix.loc["FF", "SL"] == pytest.approx(1.0)
        assert "CH" not in matrix.columns or matrix.loc["FF", "CH"] == 0.0

    def test_conditions_on_foul(self):
        frame = make_at_bats(
            [["FF", "SL"], ["FF", "CH"]],
            descriptions=[
                ["foul", "called_strike"],
                ["ball", "called_strike"],
            ],
        )
        matrix = sequencing.transition_after_outcome(frame, "foul")
        assert matrix.loc["FF", "SL"] == pytest.approx(1.0)

    def test_unknown_outcome_raises(self):
        frame = make_at_bats([["FF", "SL"]])
        with pytest.raises((ValueError, KeyError)):
            sequencing.transition_after_outcome(frame, "bunt_attempt")


class TestPutaway:
    def test_only_counts_two_strike_pitches(self):
        frame = make_at_bats(
            [["FF", "SL"]] * 10,
            strikes=[[0, 2]] * 10,
            events=["strikeout"] * 10,
        )
        table = sequencing.putaway(frame, min_pitches=1)
        assert "SL" in table.index
        assert "FF" not in table.index
        assert table.loc["SL", "n"] == 10

    def test_putaway_rate(self):
        """6 of 10 two-strike sliders end in a strikeout -> 0.60."""
        frame = make_at_bats(
            [["SL"]] * 10,
            strikes=[[2]] * 10,
            events=["strikeout"] * 6 + ["field_out"] * 4,
        )
        table = sequencing.putaway(frame, min_pitches=1)
        assert table.loc["SL", "strikeouts"] == 6
        assert table.loc["SL", "putaway_rate"] == pytest.approx(0.60)

    def test_usage_shares_sum_to_one(self):
        frame = make_at_bats(
            [["SL"]] * 30 + [["FF"]] * 10,
            strikes=[[2]] * 40,
            events=["field_out"] * 40,
        )
        table = sequencing.putaway(frame, min_pitches=1)
        assert table["usage"].sum() == pytest.approx(1.0)
        assert table.loc["SL", "usage"] == pytest.approx(0.75)

    def test_sorted_by_usage_descending(self):
        frame = make_at_bats(
            [["SL"]] * 30 + [["FF"]] * 10 + [["CH"]] * 20,
            strikes=[[2]] * 60,
            events=["field_out"] * 60,
        )
        table = sequencing.putaway(frame, min_pitches=1)
        assert list(table.index) == ["SL", "CH", "FF"]

    def test_min_pitches_gate(self):
        frame = make_at_bats(
            [["SL"]] * 30 + [["CU"]] * 3,
            strikes=[[2]] * 33,
            events=["field_out"] * 33,
        )
        table = sequencing.putaway(frame, min_pitches=10)
        assert "SL" in table.index
        assert "CU" not in table.index

    def test_counts_strikeout_double_play(self):
        frame = make_at_bats(
            [["SL"]] * 10,
            strikes=[[2]] * 10,
            events=["strikeout_double_play"] * 4 + ["field_out"] * 6,
        )
        table = sequencing.putaway(frame, min_pitches=1)
        assert table.loc["SL", "strikeouts"] == 4

    def test_empty_frame(self):
        frame = make_at_bats([["SL"]], strikes=[[2]]).iloc[0:0]
        table = sequencing.putaway(frame)
        assert len(table) == 0
