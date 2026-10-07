"""Tests for the calibration run: the plan's measurement functions and
scripts/batch_reports.py.

The point of the calibration run is to see the numbers behind every plan rule
across many pitchers — including the pitchers where a rule stayed silent,
because those are the cases that show whether a cutoff is in the right place.
So the measurement functions are tested for returning their numbers whatever
the rule then decides to say.
"""

from __future__ import annotations

import csv
import importlib.util
import tempfile
from pathlib import Path

import pytest

from src import gameplan
from src.metrics import counts
from tests.test_card import make_frame

ROOT = Path(__file__).resolve().parent.parent


def load_batch():
    spec = importlib.util.spec_from_file_location(
        "batch_reports", ROOT / "scripts" / "batch_reports.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestMeasurementsAreReturnedWhenRulesStaySilent:
    def test_hitters_count_numbers_when_he_mixes(self):
        """No "sit" line fires here, and the numbers must still come back."""
        frame = make_frame([
            {"pitch_type": "FF", "n": 35, "balls": 2, "strikes": 0},
            {"pitch_type": "SL", "n": 35, "balls": 2, "strikes": 0},
            {"pitch_type": "CH", "n": 30, "balls": 2, "strikes": 0},
        ])
        usage = counts.situational_usage(frame)
        m = gameplan.hitters_count_numbers(usage, "R")
        assert m["hard_share"] == pytest.approx(0.35)
        assert m["top_share"] == pytest.approx(0.35)
        key = gameplan.hitters_count_key(usage, "R")
        assert key["text"].startswith("He still mixes")

    def test_rare_pitches_reports_the_least_used_share_even_with_nothing_rare(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 80, "stand": "L"},
            {"pitch_type": "SL", "n": 20, "stand": "L"},
        ])
        m = gameplan.rare_pitches(frame, "L")
        assert m["rare"] == []
        assert m["min_share"] == pytest.approx(0.20)

    def test_first_pitch_numbers_carry_the_take_rate(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 20, "is_swing": False, "description": "called_strike"},
            {"pitch_type": "FF", "n": 20, "is_swing": False, "type": "B", "description": "ball"},
        ])
        m = gameplan.first_pitch_numbers(frame, counts.situational_usage(frame), "R")
        assert m["takes"] == 40
        assert m["called_on_takes"] == pytest.approx(0.5)

    def test_measurements_respect_their_gates(self):
        frame = make_frame([{"pitch_type": "FF", "n": 5}])
        usage = counts.situational_usage(frame)
        assert gameplan.first_pitch_numbers(frame, usage, "R") is None
        assert gameplan.rare_pitches(frame, "R") is None


class TestCutoffsComeFromConfig:
    def test_every_plan_cutoff_is_in_config(self):
        """The calibration run can only move a cutoff that lives in config."""
        from src import config
        for key in ("first_pitch_swing_above", "first_pitch_take_below",
                    "sit_hard_share", "sit_pitch_share", "lay_off_low_share",
                    "min_two_strike_swings", "rare_share", "count_tell_lift",
                    "two_strike_min_share", "count_tell_min_share",
                    "two_strike_min_lead", "two_strike_look_hard_min"):
            assert config.get(f"plan.{key}") is not None, key


class TestBatchHelpers:
    def test_slug_strips_accents_and_punctuation(self):
        batch = load_batch()
        assert batch.slug("Cristopher Sánchez") == "cristopher_sanchez"
        assert batch.slug("Edwin Díaz") == "edwin_diaz"

    def test_read_list_uses_names_when_given(self):
        batch = load_batch()
        rows = batch.read_list(Path("unused.csv"), ["Blake Snell"])
        assert rows == [{"name": "Blake Snell", "group": "ad hoc", "why": ""}]

    def test_the_calibration_list_is_well_formed(self):
        batch = load_batch()
        rows = batch.read_list(batch.DEFAULT_LIST, None)
        assert len(rows) >= 25
        assert all(r["name"] and r["group"] and r["why"] for r in rows)
        assert len({r["name"] for r in rows}) == len(rows)

    def test_movement_overlap_check_finds_a_collision(self):
        batch = load_batch()
        payload = {"movement": {"bubbles": [
            {"name": "Sinker", "cx": 100, "cy": 100, "r": 15, "label_x": 100, "label_y": 140},
            {"name": "Changeup", "cx": 100, "cy": 145, "r": 15, "label_x": 100, "label_y": 300},
        ]}}
        assert batch.movement_overlaps(payload) >= 1

    def test_movement_overlap_check_passes_a_clear_layout(self):
        batch = load_batch()
        payload = {"movement": {"bubbles": [
            {"name": "Sinker", "cx": 100, "cy": 100, "r": 15, "label_x": 100, "label_y": 135},
            {"name": "Changeup", "cx": 400, "cy": 300, "r": 15, "label_x": 400, "label_y": 335},
        ]}}
        assert batch.movement_overlaps(payload) == 0

    def test_where_names_the_file_and_line_an_error_came_from(self):
        batch = load_batch()
        try:
            int(float("nan"))
        except ValueError as exc:
            location = batch.where(exc)
        assert location.startswith("test_batch.py:")
        assert "test_where_names_the_file_and_line" in location

    def test_thresholds_count_how_often_a_cutoff_fires(self):
        batch = load_batch()
        rows = [
            {"status": "ok", "L_ahead_hard_share": 0.80, "R_ahead_hard_share": 0.50,
             "L_two_rule": "expect", "R_two_rule": "mixes", "count_tell_fires": 1},
            {"status": "ok", "L_ahead_hard_share": 0.72, "R_ahead_hard_share": 0.62,
             "L_two_rule": "expect", "R_two_rule": "expect", "count_tell_fires": 0},
            {"status": "error"},
        ]
        result = {t["cutoff"]: t for t in batch.thresholds(rows)}
        hard = result["sit_hard_share"]
        assert hard["pitcher_sides"] == 4
        assert hard["fires"] == 2
        assert result["two_strike_min_share + lead"]["fires"] == 3
        assert result["count_tell_lift + share"]["fires"] == 1

    def test_summary_rules_match_the_plan_sentences(self):
        """The batch run counts what the report says, through the same rule
        functions, rather than re-deriving the cutoffs."""
        batch = load_batch()
        frame = make_frame([
            {"pitch_type": "FF", "n": 45, "balls": 2, "strikes": 0},
            {"pitch_type": "SL", "n": 55, "balls": 2, "strikes": 0},
            {"pitch_type": "FF", "n": 28, "strikes": 2},
            {"pitch_type": "ST", "n": 26, "strikes": 2},
            {"pitch_type": "SI", "n": 24, "strikes": 2},
            {"pitch_type": "CH", "n": 22, "strikes": 2},
        ])
        usage = counts.situational_usage(frame)
        out = batch.side_numbers(frame, usage, "R")
        assert out["R_ahead_rule"] == "sit SL"
        assert gameplan.hitters_count_key(usage, "R")["text"].startswith("Sit slider")
        assert out["R_two_rule"] == "look_fastball"


class TestOfflineRun:
    def test_runs_end_to_end_on_a_fixture(self):
        """One real pitcher through the whole batch: report, summary, plan
        lines and threshold table all written, and a failure would not stop it."""
        batch = load_batch()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            code = batch.main(["--offline", "--no-league", "--season", "2025",
                               "--names", "Tarik Skubal",
                               "--out", str(out)])
            assert code == 0
            assert (out / "tarik_skubal_2025.html").exists()
            with (out / "summary.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            assert rows[0]["status"] == "ok"
            assert rows[0]["hand"] == "L"
            assert rows[0]["L_fp_called_on_takes"] != ""
            assert (out / "plan_lines.csv").exists()
            assert (out / "thresholds.csv").exists()
