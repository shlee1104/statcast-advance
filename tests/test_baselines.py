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


class TestProportionZ:
    def test_zero_when_matching_league(self):
        assert baselines.proportion_z(0.50, 100, 0.50) == pytest.approx(0.0)

    def test_larger_sample_yields_larger_z_for_the_same_gap(self):
        """The whole point: a 16-point gap on 39 pitches is not the same
        finding as a 16-point gap on 900."""
        small = baselines.proportion_z(0.82, 39, 0.66)
        large = baselines.proportion_z(0.82, 900, 0.66)
        assert large > small
        assert small == pytest.approx(2.13, abs=0.05)

    def test_sign_follows_direction(self):
        assert baselines.proportion_z(0.80, 100, 0.60) > 0
        assert baselines.proportion_z(0.40, 100, 0.60) < 0

    def test_degenerate_inputs_return_nan(self):
        import math
        assert math.isnan(baselines.proportion_z(0.5, 0, 0.5))
        assert math.isnan(baselines.proportion_z(0.5, 100, 0.0))
        assert math.isnan(baselines.proportion_z(0.5, 100, 1.0))


class TestCompareArsenal:
    def test_ratio_surfaces_unusual_usage(self):
        """A pitch thrown at six times the league rate is the defining fact
        about a pitcher, and a raw percentage never says so."""
        pitcher = pd.Series({"FS": 0.259, "FF": 0.340}, name="usage")
        league = pd.DataFrame([
            {"pitch_type": "FS", "league_usage": 0.046, "avg_velo": 86.3,
             "whiff_rate": 0.324, "avg_xwoba": 0.281},
            {"pitch_type": "FF", "league_usage": 0.316, "avg_velo": 94.9,
             "whiff_rate": 0.180, "avg_xwoba": 0.353},
        ])
        result = baselines.compare_arsenal(pitcher, league)
        splitter = result[result["pitch_type"] == "FS"].iloc[0]
        assert splitter["usage_ratio"] == pytest.approx(5.63, abs=0.05)


class TestCompareProfile:
    def test_xwoba_delta_is_signed_so_positive_is_bad_for_the_pitcher(self):
        """A pitch hitters do more damage against than the league's version is
        the clearest 'sit on this' finding a report can make."""
        pitcher = pd.DataFrame(
            [{"pitch_type": "SL", "usage": 0.22, "velo": 84.1,
              "whiff_rate": 0.240, "xwoba": 0.401}]
        ).set_index("pitch_type")
        league = pd.DataFrame([
            {"pitch_type": "SL", "league_usage": 0.180, "avg_velo": 84.8,
             "whiff_rate": 0.352, "avg_xwoba": 0.281},
        ])
        result = baselines.compare_profile(pitcher, league)
        row = result.iloc[0]
        assert row["xwoba_delta"] == pytest.approx(0.120, abs=1e-6)
        assert row["whiff_delta"] == pytest.approx(-0.112, abs=1e-6)
        assert row["velo_delta"] == pytest.approx(-0.7, abs=1e-6)


class TestComparePutaway:
    def test_joins_and_differences_against_league(self):
        pitcher = pd.DataFrame(
            {"n": [404], "usage": [0.399], "strikeouts": [97],
             "putaway_rate": [0.240], "whiff_rate": [0.352]},
            index=pd.Index(["FS"], name="pitch_type"),
        )
        league = pd.DataFrame([
            {"pitch_type": "FS", "league_usage": 0.068,
             "putaway_rate": 0.200, "whiff_rate": 0.295},
        ])
        result = baselines.compare_putaway(pitcher, league)
        row = result.iloc[0]
        assert row["putaway_delta"] == pytest.approx(0.040, abs=1e-6)
        assert row["usage_ratio"] == pytest.approx(5.868, abs=0.01)


class TestComparePredictability:
    def test_attaches_league_rate_for_the_top_pitch(self):
        """The 3-0 case: 82% four-seams against a league rate of 65.8%."""
        pitcher = pd.DataFrame(
            {"n": [39], "entropy": [0.958], "predictability": [0.659],
             "top_pitch": ["FF"], "top_share": [0.821]},
            index=pd.Index(["3-0"], name="count"),
        )
        league = pd.DataFrame([
            {"count": "3-0", "pitch_type": "FF", "league_share": 0.658},
        ])
        result = baselines.compare_predictability(pitcher, league)
        row = result.iloc[0]
        assert row["league_top_share"] == pytest.approx(0.658)
        assert row["share_delta"] == pytest.approx(0.163, abs=1e-6)
        assert row["share_z"] == pytest.approx(2.15, abs=0.1)


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
