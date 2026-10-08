"""Tests for the holdout check: does the plan hold up on games it never saw?"""

from __future__ import annotations

import pandas as pd
import pytest

from src import validation
from tests.test_card import make_frame


def season(games: list[list[dict]], game_type: str = "R", start: int = 1) -> pd.DataFrame:
    """One frame per game, stacked, with distinct ids and increasing dates."""
    frames = []
    for i, specs in enumerate(games):
        g = start + i
        frame = make_frame([{**s, "game_pk": g, "game_date": f"2025-{4 + g // 28:02d}-{g % 28 + 1:02d}",
                             "game_type": game_type} for s in specs])
        frame["at_bat_number"] += 1000 * g
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class TestSplits:
    def test_halves_split_by_games_in_date_order(self):
        frame = season([[{"pitch_type": "FF", "n": 10}]] * 6)
        first, second = validation.split_halves(frame)
        assert first["game_pk"].nunique() == 3 and second["game_pk"].nunique() == 3
        assert first["game_date"].max() < second["game_date"].min()

    def test_halves_leave_out_postseason(self):
        frame = pd.concat([
            season([[{"pitch_type": "FF", "n": 10}]] * 4),
            season([[{"pitch_type": "SL", "n": 10}]], game_type="D", start=50),
        ], ignore_index=True)
        first, second = validation.split_halves(frame)
        assert "SL" not in set(first["pitch_type"]) | set(second["pitch_type"])

    def test_postseason_split(self):
        frame = pd.concat([
            season([[{"pitch_type": "FF", "n": 10}]] * 4),
            season([[{"pitch_type": "SL", "n": 10}]], game_type="W", start=50),
        ], ignore_index=True)
        regular, october = validation.split_postseason(frame)
        assert set(october["pitch_type"]) == {"SL"}
        assert len(regular) == 40


class TestFlipped:
    def test_take_and_swing_are_opposites(self):
        assert validation._flipped("take", "swing")

    def test_sit_hard_and_sit_four_seam_agree(self):
        assert not validation._flipped("sit hard", "sit FF")

    def test_sit_slider_then_sit_four_seam_is_a_flip(self):
        assert validation._flipped("sit SL", "sit FF")

    def test_look_fastball_does_not_contradict_a_pitch_call(self):
        """"Expect the slider" then "look fastball, adjust to the slider" is
        weaker advice, not opposite advice — but the broad reading, which
        counts it, is still available so both numbers can be reported."""
        assert not validation._flipped("expect SL", "look_fastball")
        assert validation._flipped("expect SL", "look_fastball", broad=True)
        assert validation._flipped("expect SL", "expect FF")

    def test_losing_confidence_is_not_a_flip(self):
        """"Sit hard" becoming "he mixes" is less advice, not wrong advice."""
        assert not validation._flipped("sit hard", "mixes")
        assert not validation._flipped("take", "neutral")


def two_halves(first_specs: list[dict], second_specs: list[dict]) -> tuple:
    frame = pd.concat([
        season([first_specs] * 3),
        season([second_specs] * 3, start=10),
    ], ignore_index=True)
    return validation.split_halves(frame)


class TestCompare:
    def test_a_steady_pitcher_holds_every_call(self):
        specs = [
            {"pitch_type": "FF", "n": 30, "balls": 2, "strikes": 0},
            {"pitch_type": "SL", "n": 6, "balls": 2, "strikes": 0},
            {"pitch_type": "SL", "n": 25, "strikes": 2, "plate_z": 1.0},
            {"pitch_type": "FF", "n": 8, "strikes": 2},
        ]
        rows = validation.compare(*two_halves(specs, specs))
        by_rule = {r["rule"]: r for r in rows if r["side"] == "R"}
        assert by_rule["hitters_count"]["call_first"] == "sit hard"
        assert by_rule["hitters_count"]["same_call"]
        assert by_rule["two_strikes"]["call_first"] == "expect SL"
        assert by_rule["two_strikes"]["same_call"]
        assert by_rule["lay_off_low"]["same_call"]

    def test_a_changed_pitcher_is_caught(self):
        before = [{"pitch_type": "SL", "n": 30, "balls": 2, "strikes": 0},
                  {"pitch_type": "FF", "n": 6, "balls": 2, "strikes": 0}]
        after = [{"pitch_type": "FF", "n": 30, "balls": 2, "strikes": 0},
                 {"pitch_type": "SL", "n": 6, "balls": 2, "strikes": 0}]
        rows = validation.compare(*two_halves(before, after))
        row = next(r for r in rows if r["rule"] == "hitters_count" and r["side"] == "R")
        assert row["call_first"] == "sit SL"
        assert row["call_second"] == "sit hard"
        assert row["flipped"]

    def test_tracks_the_named_pitch_not_the_new_leader(self):
        """Two-strike stability is the share of the pitch he WAS said to
        favour, measured in the second half."""
        before = [{"pitch_type": "SL", "n": 30, "strikes": 2},
                  {"pitch_type": "FF", "n": 10, "strikes": 2}]
        after = [{"pitch_type": "SL", "n": 10, "strikes": 2},
                 {"pitch_type": "FF", "n": 30, "strikes": 2}]
        rows = validation.compare(*two_halves(before, after))
        row = next(r for r in rows if r["rule"] == "two_strikes" and r["side"] == "R")
        assert row["value_first"] == pytest.approx(0.75)
        assert row["value_second"] == pytest.approx(0.25)

    def test_nothing_to_compare_without_a_second_part(self):
        frame = season([[{"pitch_type": "FF", "n": 50}]])
        assert validation.compare(frame, frame.iloc[0:0]) == []


class TestSummarise:
    def rows(self):
        def r(rule, a, b, va, vb, flipped=False):
            return {"rule": rule, "side": "R", "call_first": a, "call_second": b,
                    "same_call": a == b, "flipped": flipped,
                    "value_first": va, "value_second": vb, "n_first": 100, "n_second": 100}
        return [
            r("hitters_count", "sit hard", "sit hard", 0.80, 0.78),
            r("hitters_count", "sit SL", "sit FF", 0.40, 0.45, flipped=True),
            r("hitters_count", "mixes", "mixes", 0.55, 0.57),
            r("hitters_count", "sit hard", "mixes", 0.72, 0.66),
        ]

    def test_hold_and_flip_rates_count_only_advice(self):
        (s,) = validation.summarise(self.rows())
        assert s["checked"] == 4
        assert s["advice_given"] == 3          # "mixes" first is not advice
        assert s["advice_held"] == 1
        assert s["flipped"] == 1

    def test_every_piece_of_advice_is_held_weakened_or_flipped(self):
        (s,) = validation.summarise(self.rows())
        assert s["advice_held"] + s["weakened"] + s["flipped"] == s["advice_given"]
        assert s["weakened"] == 1             # "sit hard" became "mixes"

    def test_skill_is_positive_when_his_own_number_predicts_better(self):
        (s,) = validation.summarise(self.rows())
        assert s["own_error"] < s["average_pitcher_error"]
        assert s["skill"] > 0

    def test_no_skill_score_for_wont_see(self):
        rows = [{"rule": "wont_see", "side": "L", "call_first": "won't see SL",
                 "call_second": "won't see SL", "same_call": True, "flipped": False,
                 "value_first": 0.01, "value_second": v, "n_first": 300, "n_second": 300}
                for v in (0.0, 0.02, 0.01)]
        (s,) = validation.summarise(rows)
        assert s["skill"] != s["skill"]  # NaN


class TestScript:
    def test_runs_offline_on_a_fixture(self):
        import csv
        import importlib.util
        import tempfile
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location("holdout", root / "scripts" / "holdout.py")
        holdout = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(holdout)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            code = holdout.main(["--offline", "--season", "2025",
                                 "--names", "Tarik Skubal", "--out", str(out)])
            assert code == 0
            with (out / "holdout_summary_halves.csv").open() as handle:
                rules = {row["rule"] for row in csv.DictReader(handle)}
            assert {"first_pitch", "two_strikes", "wont_see"} <= rules
