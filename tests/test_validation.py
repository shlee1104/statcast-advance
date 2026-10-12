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

    def test_lay_off_low_records_the_sample_it_was_measured_on(self):
        """The share that finishes low is a share of the named pitch's located
        pitches (75 a half here), not of every two-strike pitch (99)."""
        specs = [
            {"pitch_type": "SL", "n": 25, "strikes": 2, "plate_z": 1.0},
            {"pitch_type": "FF", "n": 8, "strikes": 2},
        ]
        rows = validation.compare(*two_halves(specs, specs))
        row = next(r for r in rows if r["rule"] == "lay_off_low" and r["side"] == "R")
        assert row["n_first"] == 75 and row["n_second"] == 75

    def test_no_lay_off_low_check_when_the_first_part_was_too_thin_to_say(self):
        thin = [
            {"pitch_type": "SL", "n": 5, "strikes": 2, "plate_z": 1.0},
            {"pitch_type": "FF", "n": 6, "strikes": 2},
        ]
        thick = [
            {"pitch_type": "SL", "n": 25, "strikes": 2, "plate_z": 1.0},
            {"pitch_type": "FF", "n": 8, "strikes": 2},
        ]
        rows = validation.compare(*two_halves(thin, thick))
        assert any(r["rule"] == "two_strikes" for r in rows)
        assert not any(r["rule"] == "lay_off_low" for r in rows)

    def test_wont_see_records_both_readings_of_held(self):
        """A pitch at 4% in the second half is still one a hitter can ignore
        (held), but the plan's own 3% cutoff would not have written the line
        (not held strictly). Both are kept so neither is mistaken for the
        other."""
        before = [{"pitch_type": "FF", "n": 60}, {"pitch_type": "SL", "n": 38},
                  {"pitch_type": "CU", "n": 2}]
        after = [{"pitch_type": "FF", "n": 60}, {"pitch_type": "SL", "n": 36},
                 {"pitch_type": "CU", "n": 4}]
        rows = validation.compare(*two_halves(before, after))
        row = next(r for r in rows if r["rule"] == "wont_see" and r["side"] == "R")
        assert row["value_second"] == pytest.approx(0.04)
        assert row["same_call"] and not row["same_call_strict"]


class TestWilsonInterval:
    def test_zero_of_eleven_is_not_a_flip_rate_of_zero(self):
        """The table once read "flipped 0%" on eleven pieces of advice."""
        lo, hi = validation.wilson_interval(0, 11)
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(0.2588, abs=1e-3)

    def test_seven_of_seven_is_not_certain(self):
        lo, hi = validation.wilson_interval(7, 7)
        assert lo == pytest.approx(0.6457, abs=1e-3)
        assert hi == pytest.approx(1.0)

    def test_narrows_with_sample(self):
        small = validation.wilson_interval(8, 11)
        large = validation.wilson_interval(80, 110)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_no_observations(self):
        lo, hi = validation.wilson_interval(0, 0)
        assert lo != lo and hi != hi


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

    def test_rates_carry_intervals(self):
        (s,) = validation.summarise(self.rows())
        assert s["held_lo"] < s["held_pct"] < s["held_hi"]
        assert s["flipped_lo"] < s["flipped_pct"] < s["flipped_hi"]

    def test_skill_interval_resamples_pitchers_not_rows(self):
        """Two rows from one pitcher are one arm on the same nights. With the
        pitcher named, the interval treats them as one unit and is wider."""
        import numpy as np
        rng = np.random.default_rng(3)
        rows, labelled = [], []
        for i in range(12):
            base = 0.3 + 0.04 * i
            for side in ("L", "R"):
                row = {"rule": "hitters_count", "side": side, "call_first": "sit hard",
                       "call_second": "sit hard", "same_call": True, "flipped": False,
                       "value_first": base, "value_second": base + rng.normal(0, 0.03),
                       "n_first": 100, "n_second": 100}
                rows.append(row)
                labelled.append({**row, "pitcher": f"P{i}"})
        (by_row,) = validation.summarise(rows, n_boot=400)
        (by_pitcher,) = validation.summarise(labelled, n_boot=400)
        assert by_pitcher["skill"] == pytest.approx(by_row["skill"])
        assert by_pitcher["skill_lo"] < by_pitcher["skill"] < by_pitcher["skill_hi"]
        assert by_pitcher["pitchers"] == 12
        assert ((by_pitcher["skill_hi"] - by_pitcher["skill_lo"])
                > (by_row["skill_hi"] - by_row["skill_lo"]))

    def test_strict_held_is_reported_beside_held(self):
        rows = [{"rule": "wont_see", "side": "L", "call_first": "won't see SL",
                 "call_second": "won't see SL", "same_call": True,
                 "same_call_strict": strict, "flipped": False,
                 "value_first": 0.01, "value_second": v, "n_first": 300, "n_second": 300}
                for v, strict in ((0.0, True), (0.04, False), (0.01, True))]
        (s,) = validation.summarise(rows)
        assert s["advice_held"] == 3 and s["held_strict"] == 2

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
