"""Tests for the outcome comparison in src/metrics/events.py.

Every run-value figure in the report comes through `outcome_delta`, and each
one compares a slice of his pitches with the whole it was cut from. The
standard error has to know that.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.metrics import events


def run_values(values) -> pd.DataFrame:
    return pd.DataFrame({events.RUN_VALUE_COLUMN: list(values)})


class TestDeltaStandardError:
    def test_a_slice_is_compared_with_the_rest_not_with_itself(self):
        """mean(slice) - mean(all) = (1 - f) * (mean(slice) - mean(rest)), so
        the standard error is (1 - f) times the two-sample one on slice and
        rest."""
        rng = np.random.default_rng(0)
        everything = run_values(rng.normal(0, 0.2, 400))
        piece, rest = everything.iloc[:120], everything.iloc[120:]
        x, y = piece[events.RUN_VALUE_COLUMN], rest[events.RUN_VALUE_COLUMN]
        expected = (1 - 120 / 400) * math.sqrt(x.var() / 120 + y.var() / 280)
        assert events._delta_se(piece, everything, events.RUN_VALUE_COLUMN) == pytest.approx(expected)

    def test_is_narrower_than_treating_the_two_as_independent(self):
        rng = np.random.default_rng(1)
        everything = run_values(rng.normal(0, 0.2, 400))
        piece = everything.iloc[:120]
        naive = events._welch_se(piece[events.RUN_VALUE_COLUMN],
                                 everything[events.RUN_VALUE_COLUMN])
        nested = events._delta_se(piece, everything, events.RUN_VALUE_COLUMN)
        # sqrt((1 + f) / (1 - f)) at f = 0.3 is 1.36
        assert naive / nested == pytest.approx(1.36, abs=0.08)

    def test_two_separate_samples_use_the_two_sample_formula(self):
        a = run_values([0.1, 0.2, 0.3, 0.4])
        b = run_values([0.0, 0.1, 0.0, 0.1, 0.0])
        b.index = range(100, 105)       # no rows in common with `a`
        x, y = a[events.RUN_VALUE_COLUMN], b[events.RUN_VALUE_COLUMN]
        expected = math.sqrt(x.var() / 4 + y.var() / 5)
        assert events._delta_se(a, b, events.RUN_VALUE_COLUMN) == pytest.approx(expected)

    def test_a_slice_that_is_the_whole_baseline_has_nothing_to_test(self):
        everything = run_values([0.1, -0.1, 0.2, 0.0])
        assert math.isnan(events._delta_se(everything, everything, events.RUN_VALUE_COLUMN))

    def test_a_true_null_is_rejected_about_five_percent_of_the_time(self):
        """The check that found the problem. With the two-sample formula a
        slice making up 30% of its baseline was rejected under 1% of the time
        at a nominal 5%, so real effects were being reported as unresolved."""
        rng = np.random.default_rng(7)
        values = rng.normal(0, 0.2, 500)
        rejected, runs = 0, 1000
        for _ in range(runs):
            everything = run_values(rng.permutation(values))
            result = events.outcome_delta(everything.iloc[:150], everything)
            rejected += result["rv_p"] < 0.05
        assert 0.03 < rejected / runs < 0.075

    def test_interval_and_p_value_agree(self):
        rng = np.random.default_rng(2)
        everything = run_values(rng.normal(0, 0.2, 300))
        result = events.outcome_delta(everything.iloc[:90], everything)
        excludes_zero = result["runs_lo"] > 0 or result["runs_hi"] < 0
        assert excludes_zero == (result["rv_p"] < 0.05)


class TestTwoProportionZ:
    def test_known_value(self):
        # 30/150 against 102/850: pooled 0.132, se 0.02998
        assert events.two_proportion_z(30, 150, 102, 850) == pytest.approx(2.668, abs=1e-2)

    def test_no_difference_is_zero(self):
        assert events.two_proportion_z(10, 100, 20, 200) == pytest.approx(0.0)

    def test_untestable_cases_are_nan(self):
        assert math.isnan(events.two_proportion_z(0, 0, 5, 10))
        assert math.isnan(events.two_proportion_z(10, 10, 20, 20))

    def test_lift_z_compares_the_cell_with_everything_outside_it(self):
        assert events.lift_z(30, 150, 132, 1000) == pytest.approx(
            events.two_proportion_z(30, 150, 102, 850))


class TestPlateAppearanceOutcomes:
    def test_leaves_out_plays_that_are_not_a_finished_plate_appearance(self):
        frame = pd.DataFrame({"events": ["strikeout", None, "truncated_pa", "walk",
                                          "", "pickoff_1b", "field_out"]})
        assert list(events.plate_appearance_outcomes(frame)) == [
            "strikeout", "walk", "field_out"]

    def test_tolerates_a_frame_without_the_column(self):
        assert len(events.plate_appearance_outcomes(pd.DataFrame({"x": [1]}))) == 0
