"""Tests for src/baselines.py.

Covers the sampling logic only, which is pure and runs offline. The SQL
aggregations require a populated database and are exercised by
scripts/build_baselines.py against real data.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from src import baselines


class TestSampleDates:
    def test_returns_requested_count(self):
        dates = baselines.sample_dates(2025, n_dates=30)
        assert len(dates) == 30

    def test_deterministic_for_a_given_seed(self):
        """Baselines must not drift between rebuilds for no reason."""
        first = baselines.sample_dates(2025, n_dates=20, seed=17)
        second = baselines.sample_dates(2025, n_dates=20, seed=17)
        assert first == second

    def test_different_seeds_give_different_samples(self):
        first = baselines.sample_dates(2025, n_dates=20, seed=1)
        second = baselines.sample_dates(2025, n_dates=20, seed=2)
        assert first != second

    def test_dates_fall_inside_the_season_window(self):
        dates = baselines.sample_dates(2025, n_dates=30)
        parsed = [dt.date.fromisoformat(d) for d in dates]
        assert min(parsed) >= dt.date(2025, 3, 20)
        assert max(parsed) <= dt.date(2025, 10, 1)

    def test_returns_chronological_order(self):
        dates = baselines.sample_dates(2025, n_dates=25)
        assert dates == sorted(dates)

    def test_stratification_spreads_across_the_season(self):
        """Every month of the season should be represented.

        This is the property that distinguishes stratified sampling from
        drawing 30 dates at random, where clustering is likely and a month can
        go missing entirely.
        """
        dates = baselines.sample_dates(2025, n_dates=30)
        months = {dt.date.fromisoformat(d).month for d in dates}
        assert {4, 5, 6, 7, 8, 9}.issubset(months)

    def test_no_interval_contributes_twice(self):
        """One date per interval means no duplicates."""
        dates = baselines.sample_dates(2025, n_dates=30)
        assert len(set(dates)) == len(dates)

    def test_rejects_impossible_request(self):
        with pytest.raises(ValueError):
            baselines.sample_dates(2025, n_dates=5000)

    def test_rejects_zero_dates(self):
        with pytest.raises(ValueError):
            baselines.sample_dates(2025, n_dates=0)

    def test_works_for_other_seasons(self):
        dates = baselines.sample_dates(2024, n_dates=10)
        assert all(d.startswith("2024") for d in dates)


class TestCompareCountMix:
    def test_computes_delta_against_league(self):
        pitcher = pd.DataFrame(
            {"FF": [0.60, 0.30], "SL": [0.40, 0.70]},
            index=pd.Index(["0-0", "1-2"], name="count"),
        )
        league = pd.DataFrame([
            {"count": "0-0", "pitch_type": "FF", "league_share": 0.50},
            {"count": "0-0", "pitch_type": "SL", "league_share": 0.50},
            {"count": "1-2", "pitch_type": "FF", "league_share": 0.35},
            {"count": "1-2", "pitch_type": "SL", "league_share": 0.65},
        ])

        result = baselines.compare_count_mix(pitcher, league)
        row = result[(result["count"] == "0-0") & (result["pitch_type"] == "FF")].iloc[0]
        assert row["delta"] == pytest.approx(0.10)

    def test_sorted_by_absolute_delta(self):
        """The largest deviation leads, whichever direction it runs."""
        pitcher = pd.DataFrame(
            {"FF": [0.55, 0.10], "SL": [0.45, 0.90]},
            index=pd.Index(["0-0", "1-2"], name="count"),
        )
        league = pd.DataFrame([
            {"count": "0-0", "pitch_type": "FF", "league_share": 0.50},
            {"count": "0-0", "pitch_type": "SL", "league_share": 0.50},
            {"count": "1-2", "pitch_type": "FF", "league_share": 0.35},
            {"count": "1-2", "pitch_type": "SL", "league_share": 0.65},
        ])

        result = baselines.compare_count_mix(pitcher, league)
        assert abs(result.iloc[0]["delta"]) == pytest.approx(0.25)

    def test_missing_league_entry_yields_nan_not_a_crash(self):
        pitcher = pd.DataFrame(
            {"KN": [1.0]},
            index=pd.Index(["0-0"], name="count"),
        )
        league = pd.DataFrame([
            {"count": "0-0", "pitch_type": "FF", "league_share": 0.50},
        ])
        result = baselines.compare_count_mix(pitcher, league)
        assert result["league_share"].isna().all()
