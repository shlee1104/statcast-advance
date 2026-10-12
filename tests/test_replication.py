"""Tests for src/replication.py and scripts/replication.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src import replication
from src.metrics import splits


def make_frame(specs: list[dict]) -> pd.DataFrame:
    """Build a frame carrying every column the rules and tables touch.

    Each spec is a group of identical plate appearances. A spec with `then`
    is a two-pitch plate appearance — the pitch, then its follow-up — which is
    what the sequencing checks need; without it, each pitch is its own.
    Defaults describe a four-seam over the middle to a right-handed hitter.
    """
    rows = []
    at_bat = 0
    for spec in specs:
        for _ in range(spec["n"]):
            at_bat += 1
            pitches = [spec] + ([spec["then"]] if "then" in spec else [])
            for number, pitch in enumerate(pitches, start=1):
                balls = pitch.get("balls", 0)
                strikes = pitch.get("strikes", 0)
                zone = pitch.get("zone", 5)
                rows.append({
                    "game_pk": spec.get("game_pk", 1),
                    "game_date": "2025-06-01",
                    "game_type": "R",
                    "at_bat_number": at_bat,
                    "pitch_number": number,
                    "balls": balls,
                    "strikes": strikes,
                    "count": f"{balls}-{strikes}",
                    "is_first_pitch": balls == 0 and strikes == 0,
                    "is_two_strike": strikes == 2,
                    "is_ahead": strikes > balls,
                    "is_behind": balls > strikes,
                    "pitch_type": pitch["pitch_type"],
                    "pitch_display": pitch["pitch_type"],
                    "p_throws": "R",
                    "stand": pitch.get("stand", spec.get("stand", "R")),
                    "release_speed": pitch.get("velo", 95.0),
                    "release_spin_rate": 2200.0,
                    "release_extension": 6.5,
                    "release_pos_x": -1.9,
                    "release_pos_z": 5.4,
                    "arm_angle": 42.0,
                    "pfx_x": -0.5,
                    "pfx_z": 1.3,
                    "plate_x": 0.0,
                    "plate_z": pitch.get("plate_z", 2.5),
                    "sz_top": 3.5,
                    "sz_bot": 1.5,
                    "zone": zone,
                    "description": "called_strike",
                    "events": None,
                    "type": "S",
                    "estimated_woba_using_speedangle": pitch.get("xwoba", 0.300),
                    # Alternating, so run-value comparisons have a variance
                    # to test against; constant run values give no p-value.
                    "delta_run_exp": 0.05 if at_bat % 2 else -0.05,
                    "n_thruorder_pitcher": 1,
                    "is_swing": False,
                    "is_whiff": False,
                    "is_called_strike": True,
                    "is_in_zone": zone in range(1, 10),
                    "is_chase": False,
                    "is_contact": False,
                })
    return pd.DataFrame(rows)


def of_kind(rows: list[dict], kind: str, key: str | None = None) -> list[dict]:
    return [r for r in rows
            if r["kind"] == kind and (key is None or r["key"] == key)]


def slider_in_0_2(sliders: int, of: int = 100) -> pd.DataFrame:
    """400 pitches, a quarter of them sliders, with `sliders` of the `of`
    pitches thrown in 0-2 being sliders."""
    return make_frame([
        {"pitch_type": "SL", "n": sliders, "strikes": 2},
        {"pitch_type": "FF", "n": of - sliders, "strikes": 2},
        {"pitch_type": "SL", "n": 100 - sliders},
        {"pitch_type": "FF", "n": 300 - (of - sliders)},
    ])


class TestCountLifts:
    def test_a_lift_that_is_still_there_holds(self):
        """Sliders are 25% overall and 60% in 0-2, in both halves."""
        rows = replication.finding_rows(slider_in_0_2(60), slider_in_0_2(60))
        (row,) = of_kind(rows, "count_lift", "0-2 SL")
        assert row["first"] == pytest.approx(2.4)
        assert row["second"] == pytest.approx(2.4)
        assert row["n_second"] == 100
        assert row["held"] and row["direction"]

    def test_a_weaker_lift_keeps_its_direction_without_holding(self):
        """35% in 0-2 against 25% overall is 1.4x: still more than usual,
        under the rule's 1.5x."""
        rows = replication.finding_rows(slider_in_0_2(60), slider_in_0_2(35))
        (row,) = of_kind(rows, "count_lift", "0-2 SL")
        assert row["second"] == pytest.approx(1.4)
        assert not row["held"]
        assert row["direction"]

    def test_a_lift_that_vanishes_loses_both(self):
        rows = replication.finding_rows(slider_in_0_2(60), slider_in_0_2(25))
        (row,) = of_kind(rows, "count_lift", "0-2 SL")
        assert row["second"] == pytest.approx(1.0)
        assert not row["held"] and not row["direction"]

    def test_a_second_half_too_thin_to_say_is_not_a_failure(self):
        """Ten pitches in 0-2 cannot confirm or refute anything, so the
        finding produces no row rather than a miss."""
        rows = replication.finding_rows(slider_in_0_2(60), slider_in_0_2(6, of=10))
        assert of_kind(rows, "count_lift", "0-2 SL") == []

    def test_nothing_is_measured_in_a_thin_half(self):
        thin = make_frame([{"pitch_type": "FF", "n": 150}])
        assert replication.finding_rows(slider_in_0_2(60), thin) == []
        assert replication.per_pitch_rows(slider_in_0_2(60), thin) == []
        assert replication.per_pitcher_rows(slider_in_0_2(60), thin) == []


class TestSequencing:
    def half(self, curve_after_high_fastball: int) -> pd.DataFrame:
        """240 two-pitch plate appearances; 100 follow-ups are curveballs,
        `curve_after_high_fastball` of them after one of 80 high four-seams."""
        after = curve_after_high_fastball
        return make_frame([
            {"pitch_type": "FF", "zone": 2, "n": after, "then": {"pitch_type": "CU"}},
            {"pitch_type": "FF", "zone": 2, "n": 80 - after, "then": {"pitch_type": "FF"}},
            {"pitch_type": "SL", "zone": 5, "n": 100 - after, "then": {"pitch_type": "CU"}},
            {"pitch_type": "SL", "zone": 5, "n": 60 + after, "then": {"pitch_type": "FF"}},
        ])

    def test_a_setup_that_still_works_holds(self):
        rows = replication.finding_rows(self.half(60), self.half(60))
        (row,) = of_kind(rows, "sequencing", "FF UP > CU")
        assert row["first"] == pytest.approx(0.75 / (100 / 240))
        assert row["held"] and row["direction"]
        assert row["n_second"] == 80

    def test_a_follow_up_that_never_comes_is_a_miss_not_a_gap(self):
        """He set up the curveball with a high four-seam in the first half and
        never did in the second. That is the finding failing, and it has to
        be counted: a lift of zero, not a missing row."""
        rows = replication.finding_rows(self.half(60), self.half(0))
        (row,) = of_kind(rows, "sequencing", "FF UP > CU")
        assert row["second"] == 0.0
        assert not row["held"] and not row["direction"]


class TestHeightTells:
    def half(self, sliders_low: int) -> pd.DataFrame:
        return make_frame([
            {"pitch_type": "SL", "n": sliders_low, "plate_z": 1.7},
            {"pitch_type": "FF", "n": 150 - sliders_low, "plate_z": 1.7},
            {"pitch_type": "FF", "n": sliders_low, "plate_z": 3.3},
            {"pitch_type": "SL", "n": 150 - sliders_low, "plate_z": 3.3},
        ])

    def test_a_band_that_still_says_the_pitch_holds(self):
        rows = replication.finding_rows(self.half(120), self.half(120))
        (row,) = of_kind(rows, "location", "DOWN SL")
        assert row["first"] == pytest.approx(1.6)
        assert row["held"] and row["direction"]

    def test_fastball_up_is_not_tested_because_it_is_not_a_finding(self):
        rows = replication.finding_rows(self.half(120), self.half(120))
        assert of_kind(rows, "location", "UP FF") == []

    def test_holding_needs_the_share_as_well_as_the_lift(self):
        """75 of 150 low pitches are sliders: 50% against 50% overall."""
        rows = replication.finding_rows(self.half(120), self.half(75))
        (row,) = of_kind(rows, "location", "DOWN SL")
        assert not row["held"]


class TestHandednessGaps:
    def half(self, sliders_to_lefties: int, sliders_to_righties: int) -> pd.DataFrame:
        return make_frame([
            {"pitch_type": "SL", "n": sliders_to_lefties, "stand": "L"},
            {"pitch_type": "FF", "n": 200 - sliders_to_lefties, "stand": "L"},
            {"pitch_type": "SL", "n": sliders_to_righties, "stand": "R"},
            {"pitch_type": "FF", "n": 200 - sliders_to_righties, "stand": "R"},
        ])

    def test_a_gap_that_is_still_there_holds(self):
        rows = replication.finding_rows(self.half(50, 120), self.half(50, 120))
        (row,) = of_kind(rows, "handedness", "SL")
        assert row["first"] == pytest.approx(-0.35)
        assert row["n_second"] == 200
        assert row["held"] and row["direction"]

    def test_a_gap_that_narrows_keeps_its_direction(self):
        """25% to lefties and 40% to righties: 15 points, the same way."""
        rows = replication.finding_rows(self.half(50, 120), self.half(50, 80))
        (row,) = of_kind(rows, "handedness", "SL")
        assert not row["held"]
        assert row["direction"]

    def test_a_gap_that_reverses_is_not_holding_at_any_size(self):
        rows = replication.finding_rows(self.half(50, 120), self.half(120, 50))
        (row,) = of_kind(rows, "handedness", "SL")
        assert row["second"] == pytest.approx(0.35)
        assert not row["held"] and not row["direction"]


def bucket(name: str, delta: float, pairs: int, se: float = 0.2) -> dict:
    return {"bucket": name, "n": pairs * 15, "velo": 95.0 + delta,
            "velo_delta": delta, "outings": pairs, "pairs": pairs, "delta_se": se}


class TestVelocityDrop:
    """The fatigue table needs a season of real outings to build, so these
    hand the rule its table and test what is done with it."""

    def run(self, monkeypatch, early: list[dict], late: list[dict]):
        first, second = slider_in_0_2(60), slider_in_0_2(60)
        tables = {id(first): pd.DataFrame(early), id(second): pd.DataFrame(late)}
        monkeypatch.setattr(splits, "fatigue", lambda frame: tables[id(frame)])
        return (replication.finding_rows(first, second),
                replication.per_pitcher_rows(first, second))

    def test_the_same_bucket_is_re_measured(self, monkeypatch):
        rows, _ = self.run(
            monkeypatch,
            [bucket("1-15", 0.0, 14), bucket("76-90", -1.5, 12)],
            [bucket("1-15", 0.0, 13), bucket("76-90", -1.2, 8)],
        )
        (row,) = of_kind(rows, "fatigue")
        assert row["key"] == "76-90"
        assert (row["first"], row["second"]) == (-1.5, -1.2)
        assert row["n_second"] == 8
        assert row["held"]

    def test_half_the_drop_is_the_same_direction_but_not_the_same_finding(self, monkeypatch):
        rows, _ = self.run(
            monkeypatch,
            [bucket("76-90", -1.5, 12)], [bucket("76-90", -0.6, 8)],
        )
        (row,) = of_kind(rows, "fatigue")
        assert not row["held"]
        assert row["direction"]

    def test_a_drop_inside_its_own_noise_is_not_a_finding_to_begin_with(self, monkeypatch):
        """The rule requires the drop to be two standard errors clear. The
        pair still goes to the correlation, which needs no bar."""
        rows, per_pitcher = self.run(
            monkeypatch,
            [bucket("76-90", -1.5, 12, se=1.0)], [bucket("76-90", -1.2, 8)],
        )
        assert of_kind(rows, "fatigue") == []
        (pair,) = [r for r in per_pitcher if r["metric"] == "worst_bucket_velo_delta"]
        assert (pair["first"], pair["second"]) == (-1.5, -1.2)

    def test_too_few_second_half_outings_reach_the_bucket(self, monkeypatch):
        rows, per_pitcher = self.run(
            monkeypatch,
            [bucket("91-105", -1.5, 12)], [bucket("91-105", -2.0, 3)],
        )
        assert of_kind(rows, "fatigue") == []
        assert [r for r in per_pitcher if r["metric"] == "worst_bucket_velo_delta"] == []


class TestThirdTripVerdict:
    def verdict(self, attribution: str, xwoba_delta: float) -> dict:
        return {"reliable": True, "attribution": attribution,
                "xwoba_delta": xwoba_delta, "velo_delta": -0.2}

    def run(self, monkeypatch, early: dict, late: dict) -> list[dict]:
        first, second = slider_in_0_2(60), slider_in_0_2(60)
        verdicts = {id(first): early, id(second): late}
        monkeypatch.setattr(splits, "decompose_tto", lambda frame: verdicts[id(frame)])
        monkeypatch.setattr(splits, "times_through_order", lambda frame: pd.DataFrame(
            [{"tto": 1, "n": 300}, {"tto": 3, "n": 200}]))
        return of_kind(replication.finding_rows(first, second), "tto")

    def test_holding_means_reaching_the_same_verdict(self, monkeypatch):
        (row,) = self.run(monkeypatch, self.verdict("results_decay", 0.12),
                          self.verdict("results_decay", 0.10))
        assert row["held"] and row["direction"]

    def test_no_verdict_the_second_time_is_a_miss(self, monkeypatch):
        (row,) = self.run(monkeypatch, self.verdict("results_decay", 0.12),
                          self.verdict("neither", 0.01))
        assert not row["held"] and not row["direction"]

    def test_no_verdict_the_first_time_is_nothing_to_test(self, monkeypatch):
        assert self.run(monkeypatch, self.verdict("neither", 0.0),
                        self.verdict("results_decay", 0.10)) == []


class TestPerPitchRows:
    def test_only_pitches_thrown_enough_in_both_halves(self):
        first = make_frame([{"pitch_type": "FF", "n": 150},
                            {"pitch_type": "SL", "n": 60}])
        second = make_frame([{"pitch_type": "FF", "n": 150},
                             {"pitch_type": "SL", "n": 200}])
        rows = replication.per_pitch_rows(first, second)
        assert [r["pitch"] for r in rows] == ["FF"]

    def test_each_number_is_recorded_for_each_half(self):
        first = make_frame([{"pitch_type": "FF", "n": 150, "velo": 96.0},
                            {"pitch_type": "SL", "n": 150, "velo": 86.0}])
        second = make_frame([{"pitch_type": "FF", "n": 100, "velo": 95.0},
                             {"pitch_type": "SL", "n": 300, "velo": 85.0}])
        rows = {r["pitch"]: r for r in replication.per_pitch_rows(first, second)}
        assert (rows["FF"]["velo_1"], rows["FF"]["velo_2"]) == (96.0, 95.0)
        assert rows["SL"]["usage_1"] == pytest.approx(0.50)
        assert rows["SL"]["usage_2"] == pytest.approx(0.75)

    def test_league_figure_is_the_same_in_both_halves(self):
        first = make_frame([{"pitch_type": "FF", "n": 200, "xwoba": 0.350}])
        second = make_frame([{"pitch_type": "FF", "n": 200, "xwoba": 0.250}])
        league = pd.DataFrame([{"pitch_type": "FF", "avg_xwoba": 0.300}])
        (row,) = replication.per_pitch_rows(first, second, league)
        assert row["xwoba_vs_lg_1"] == pytest.approx(0.050)
        assert row["xwoba_vs_lg_2"] == pytest.approx(-0.050)


class TestSummariseFindings:
    def rows(self) -> list[dict]:
        def r(pitcher, kind, held, direction, first=2.0, second=1.6):
            return {"pitcher": pitcher, "kind": kind, "key": "x", "first": first,
                    "second": second, "n_second": 50, "held": held,
                    "direction": direction}
        return [
            r("A", "count_lift", True, True),
            r("A", "count_lift", True, True),
            r("B", "count_lift", False, True),
            r("B", "count_lift", False, False),
            r("A", "handedness", True, True),
        ]

    def test_one_line_per_finding_type_found(self):
        summary = replication.summarise_findings(self.rows())
        assert [s["kind"] for s in summary] == ["count_lift", "handedness"]

    def test_held_and_direction_are_counted_separately(self):
        counts, _ = replication.summarise_findings(self.rows())
        assert (counts["found"], counts["pitchers"]) == (4, 2)
        assert counts["held"] == 2 and counts["held_pct"] == pytest.approx(0.5)
        assert counts["direction"] == 3

    def test_rates_carry_intervals(self):
        """One finding that held is not a 100% hold rate."""
        _, gap = replication.summarise_findings(self.rows())
        assert gap["held_pct"] == 1.0
        assert gap["held_lo"] < 0.25
        assert gap["held_hi"] == pytest.approx(1.0)

    def test_no_findings(self):
        assert replication.summarise_findings([]) == []


class TestCorrelations:
    def pitch_rows(self, n: int) -> list[dict]:
        return [{"pitch": "FF", "usage_1": 0.1 * i, "usage_2": 0.1 * i + 0.01,
                 "xwoba_1": 0.30 + 0.01 * i, "xwoba_2": 0.30 - 0.01 * i,
                 "xwoba_n1": 40, "xwoba_n2": 40}
                for i in range(n)]

    def test_a_number_that_carries_over_correlates(self):
        by_column = {s["column"]: s for s in
                     replication.pitch_correlations(self.pitch_rows(8))}
        assert by_column["usage"]["r"] == pytest.approx(1.0)
        assert by_column["usage"]["pairs"] == 8
        assert by_column["xwoba"]["r"] == pytest.approx(-1.0)

    def test_xwoba_needs_enough_plate_appearances_in_each_half(self):
        """Only the pitch that ends a plate appearance has an xwOBA, so a
        pitch thrown 100 times can rest on a dozen of them."""
        rows = self.pitch_rows(8)
        for row in rows[:4]:
            row["xwoba_n2"] = 12
        by_column = {s["column"]: s for s in replication.pitch_correlations(rows)}
        assert by_column["usage"]["pairs"] == 8
        assert "xwoba" not in by_column          # four pairs left: too few

    def test_too_few_pairs_is_no_correlation_not_a_bad_one(self):
        assert replication.pitch_correlations(self.pitch_rows(5)) == []

    def test_per_pitcher_numbers_are_correlated_by_metric(self):
        rows = [{"metric": "tto_velo_delta", "key": "", "first": -0.1 * i,
                 "second": -0.1 * i} for i in range(7)]
        rows += [{"metric": "tto_xwoba_delta", "key": "", "first": 0.01 * i,
                  "second": 0.02} for i in range(3)]
        (velo,) = replication.pitcher_correlations(rows)
        assert velo["metric"] == "tto_velo_delta"
        assert velo["r"] == pytest.approx(1.0)


class TestFullSeason:
    def test_counts_findings_by_type_and_the_comparisons_behind_them(self):
        row = replication.full_season_row(slider_in_0_2(60))
        assert row["pitches"] == 400
        assert row["n_predictable_count"] == 1
        assert row["findings"] == sum(row[f"n_{f}"] for f in replication.FLAG_TYPES)
        assert row["comparisons"] > 0
        assert row["bh_survivors"] <= row["raw_p05"] <= row["comparisons"]
        assert row["resolved"] <= row["findings"]

    def test_totals_across_pitchers(self):
        rows = [replication.full_season_row(slider_in_0_2(60)) for _ in range(3)]
        totals = replication.summarise_full_season(rows)
        assert totals["pitchers"] == 3
        counted = {f["flag"]: f for f in totals["by_flag"]}
        assert counted["predictable_count"] == {
            "flag": "predictable_count", "pitchers": 3, "findings": 3}
        assert totals["findings"] == 3 * rows[0]["findings"]

    def test_no_pitchers(self):
        assert replication.summarise_full_season([])["pitchers"] == 0


def load_script():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "replication_script", root / "scripts" / "replication.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestScript:
    def test_runs_offline_on_a_fixture(self):
        import csv
        import tempfile
        from pathlib import Path

        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            code = script.main(["--offline", "--season", "2025",
                                "--names", "Tarik Skubal", "--out", str(out)])
            assert code == 0
            with (out / "full_season.csv").open() as handle:
                (season,) = list(csv.DictReader(handle))
            assert season["pitcher"] == "Tarik Skubal"
            assert int(season["comparisons"]) > 0
            with (out / "findings.csv").open() as handle:
                findings = list(csv.DictReader(handle))
            assert findings
            assert {"pitcher", "kind", "key", "first", "second", "n_second",
                    "held", "direction"} <= set(findings[0])
            assert (out / "per_pitch.csv").exists()
            assert (out / "per_pitcher.csv").exists()

    def test_cached_run_matches_names_without_a_lookup(self, monkeypatch):
        """--cached reads who is in the cache from the cache. Savant stores
        "Last, First", and a name typed without its accent still matches."""
        script = load_script()

        class Closed:
            def close(self):
                pass

        cache = pd.DataFrame([
            {"mlbam_id": 1, "season": 2026, "player_type": "pitcher",
             "row_count": 2500, "player_name": "Yamamoto, Yoshinobu"},
            {"mlbam_id": 2, "season": 2026, "player_type": "pitcher",
             "row_count": 900, "player_name": "Díaz, Edwin"},
            {"mlbam_id": 3, "season": 2026, "player_type": "pitcher",
             "row_count": 0, "player_name": "Clase, Emmanuel"},
            {"mlbam_id": 4, "season": 2025, "player_type": "pitcher",
             "row_count": 3000, "player_name": "Skubal, Tarik"},
        ])
        monkeypatch.setattr(script.store, "connect", lambda: Closed())
        monkeypatch.setattr(script.store, "cache_summary", lambda conn: cache)
        batch = script._batch()

        assert script.cached_pitchers(2026, None, batch) == [
            ("Edwin Díaz", 2), ("Yoshinobu Yamamoto", 1)]
        assert script.cached_pitchers(2026, ["Edwin Diaz"], batch) == [("Edwin Díaz", 2)]
        assert script.cached_pitchers(2026, ["yamamoto"], batch) == [
            ("Yoshinobu Yamamoto", 1)]
        assert script.cached_pitchers(2026, ["Tarik Skubal"], batch) == []
