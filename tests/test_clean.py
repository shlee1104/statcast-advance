"""Tests for src/clean.py.

These define the contract. Implement clean.py until they all pass:

    pytest tests/test_clean.py -v

They use small hand-built frames rather than real API data, so they run
offline in under a second and every expected value can be verified by eye.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import clean


def make_frame(rows: list[dict]) -> pd.DataFrame:
    """Build a test frame with sensible defaults for unspecified columns."""
    defaults = {
        "game_pk": 1,
        "game_date": "2025-05-01",
        "game_type": "R",
        "game_year": 2025,
        "at_bat_number": 1,
        "pitch_number": 1,
        "pitcher": 669373,
        "batter": 1,
        "pitch_type": "FF",
        "description": "ball",
        "balls": 0,
        "strikes": 0,
        "release_speed": 95.0,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


class TestConsolidatePitchTypes:
    def test_deprecated_two_seam_becomes_sinker(self):
        result = clean.consolidate_pitch_types(make_frame([{"pitch_type": "FT"}]))
        assert result["pitch_type"].iloc[0] == "SI"

    def test_generic_fastball_becomes_four_seam(self):
        result = clean.consolidate_pitch_types(make_frame([{"pitch_type": "FA"}]))
        assert result["pitch_type"].iloc[0] == "FF"

    def test_curve_variants_fold_together(self):
        result = clean.consolidate_pitch_types(
            make_frame([{"pitch_type": "KC"}, {"pitch_type": "CS"}, {"pitch_type": "CU"}])
        )
        assert list(result["pitch_type"]) == ["CU", "CU", "CU"]

    def test_sweeper_stays_separate_from_slider(self):
        """The sweeper is a distinct pitch. Folding it into SL loses real info."""
        result = clean.consolidate_pitch_types(
            make_frame([{"pitch_type": "ST"}, {"pitch_type": "SL"}, {"pitch_type": "SV"}])
        )
        assert list(result["pitch_type"]) == ["ST", "SL", "SL"]

    def test_unmapped_codes_pass_through(self):
        result = clean.consolidate_pitch_types(
            make_frame([{"pitch_type": "CH"}, {"pitch_type": "FC"}])
        )
        assert list(result["pitch_type"]) == ["CH", "FC"]

    def test_adds_display_names(self):
        result = clean.consolidate_pitch_types(
            make_frame([{"pitch_type": "FT"}, {"pitch_type": "ST"}])
        )
        assert list(result["pitch_display"]) == ["Sinker", "Sweeper"]

    def test_does_not_mutate_input(self):
        original = make_frame([{"pitch_type": "FT"}])
        clean.consolidate_pitch_types(original)
        assert original["pitch_type"].iloc[0] == "FT"


class TestDropNonCompetitive:
    def test_drops_pitchouts_and_intentional_balls(self):
        result = clean.drop_non_competitive(
            make_frame([
                {"pitch_type": "FF"},
                {"pitch_type": "PO"},
                {"pitch_type": "IN"},
                {"pitch_type": "SL"},
            ])
        )
        assert list(result["pitch_type"]) == ["FF", "SL"]

    def test_drops_by_description_even_when_pitch_type_looks_normal(self):
        """Savant sometimes labels an intentional ball as an ordinary fastball."""
        result = clean.drop_non_competitive(
            make_frame([
                {"pitch_type": "FF", "description": "ball"},
                {"pitch_type": "FF", "description": "intent_ball"},
                {"pitch_type": "FF", "description": "pitchout"},
            ])
        )
        assert len(result) == 1
        assert result["description"].iloc[0] == "ball"

    def test_drops_null_and_empty_pitch_types(self):
        result = clean.drop_non_competitive(
            make_frame([{"pitch_type": "FF"}, {"pitch_type": None}, {"pitch_type": ""}])
        )
        assert len(result) == 1

    def test_index_is_reset(self):
        result = clean.drop_non_competitive(
            make_frame([{"pitch_type": "PO"}, {"pitch_type": "FF"}])
        )
        assert list(result.index) == [0]


class TestFilterGameTypes:
    def test_keeps_regular_season(self):
        result = clean.filter_game_types(make_frame([{"game_type": "R"}]))
        assert len(result) == 1

    def test_drops_spring_training(self):
        """Spring training pitches are experiments, not scouting evidence."""
        result = clean.filter_game_types(
            make_frame([{"game_type": "R"}, {"game_type": "S"}, {"game_type": "E"}])
        )
        assert list(result["game_type"]) == ["R"]

    def test_keeps_postseason(self):
        result = clean.filter_game_types(
            make_frame([
                {"game_type": "F"},
                {"game_type": "D"},
                {"game_type": "L"},
                {"game_type": "W"},
            ])
        )
        assert len(result) == 4

    def test_explicit_game_types_override_config(self):
        result = clean.filter_game_types(
            make_frame([{"game_type": "R"}, {"game_type": "S"}]),
            game_types=["S"],
        )
        assert list(result["game_type"]) == ["S"]


class TestAddCountState:
    def test_builds_count_string(self):
        result = clean.add_count_state(
            make_frame([{"balls": 0, "strikes": 0}, {"balls": 3, "strikes": 2}])
        )
        assert list(result["count"]) == ["0-0", "3-2"]

    def test_ahead_behind_flags(self):
        result = clean.add_count_state(
            make_frame([
                {"balls": 0, "strikes": 2},  # ahead
                {"balls": 3, "strikes": 0},  # behind
                {"balls": 1, "strikes": 1},  # even
            ])
        )
        assert list(result["is_ahead"]) == [True, False, False]
        assert list(result["is_behind"]) == [False, True, False]

    def test_two_strike_and_first_pitch_flags(self):
        result = clean.add_count_state(
            make_frame([
                {"balls": 0, "strikes": 0},
                {"balls": 1, "strikes": 2},
            ])
        )
        assert list(result["is_first_pitch"]) == [True, False]
        assert list(result["is_two_strike"]) == [False, True]

    def test_drops_impossible_counts(self):
        """A '4-2 count' in the report would be an obvious data-quality miss."""
        result = clean.add_count_state(
            make_frame([
                {"balls": 0, "strikes": 0},
                {"balls": 4, "strikes": 2},
                {"balls": 1, "strikes": 3},
                {"balls": -1, "strikes": 0},
            ])
        )
        assert len(result) == 1
        assert result["count"].iloc[0] == "0-0"

    def test_all_twelve_counts_survive(self):
        rows = [{"balls": b, "strikes": s} for b in range(4) for s in range(3)]
        result = clean.add_count_state(make_frame(rows))
        assert len(result) == 12
        assert result["count"].nunique() == 12


class TestPrepareForStorage:
    def test_returns_exactly_the_storage_schema(self):
        from src.store import PITCH_SCHEMA

        result = clean.prepare_for_storage(make_frame([{"pitch_type": "FF"}]))
        assert list(result.columns) == list(PITCH_SCHEMA.keys())

    def test_runs_the_full_pipeline(self):
        result = clean.prepare_for_storage(
            make_frame([
                {"pitch_type": "FT", "game_type": "R", "balls": 0, "strikes": 0},
                {"pitch_type": "PO", "game_type": "R", "balls": 1, "strikes": 0},
                {"pitch_type": "FF", "game_type": "S", "balls": 2, "strikes": 0},
                {"pitch_type": "SL", "game_type": "R", "balls": 5, "strikes": 0},
            ])
        )
        # Only the first row survives: FT consolidates to SI, the pitchout is
        # dropped, spring training is dropped, the 5-ball count is dropped.
        assert len(result) == 1
        assert result["pitch_type"].iloc[0] == "SI"

    def test_missing_columns_are_created_as_null(self):
        """Savant occasionally omits columns; storage must still succeed."""
        frame = make_frame([{"pitch_type": "FF"}])
        result = clean.prepare_for_storage(frame)
        assert "release_spin_rate" in result.columns
        assert result["release_spin_rate"].isna().all()

    def test_game_date_is_datetime(self):
        result = clean.prepare_for_storage(make_frame([{"pitch_type": "FF"}]))
        assert pd.api.types.is_datetime64_any_dtype(result["game_date"])


@pytest.mark.parametrize(
    "func",
    [
        clean.consolidate_pitch_types,
        clean.drop_non_competitive,
        clean.filter_game_types,
        clean.add_count_state,
    ],
)
def test_handles_empty_frame(func):
    """Every step must survive a player with no matching pitches."""
    empty = make_frame([{"pitch_type": "FF"}]).iloc[0:0]
    result = func(empty)
    assert len(result) == 0
