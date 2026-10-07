"""Tests for combining seasons.

Pooling two seasons buys sample for a pitcher who missed time, at the cost of
blurring one who changed. These check that the change is named, that a pitch
he has shelved is kept out of the plan, and that the season line still counts
every plate appearance.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import cli, gameplan, report
from src.metrics import seasons
from tests.test_card import make_frame


def two_seasons(before: list[dict], after: list[dict]) -> pd.DataFrame:
    """A 2025 frame and a 2026 frame, stacked, with distinct games."""
    old = make_frame([{**s, "game_pk": s.get("game_pk", 1)} for s in before])
    new = make_frame([{**s, "game_pk": s.get("game_pk", 2),
                       "game_date": "2026-06-01"} for s in after])
    old["game_year"], new["game_year"] = 2025, 2026
    new["at_bat_number"] += len(old)
    return pd.concat([old, new], ignore_index=True)


class TestSeasonChanges:
    def test_single_season_has_nothing_to_compare(self):
        frame = make_frame([{"pitch_type": "FF", "n": 50}])
        frame["game_year"] = 2026
        assert seasons.season_changes(frame) is None

    def test_names_a_new_pitch_and_a_dropped_one(self):
        frame = two_seasons(
            [{"pitch_type": "FF", "n": 500}, {"pitch_type": "FC", "n": 100}],
            [{"pitch_type": "FF", "n": 500}, {"pitch_type": "FS", "n": 100}],
        )
        c = seasons.season_changes(frame)
        assert c["latest"] == 2026 and c["earlier"] == [2025]
        assert [p for p, _ in c["new"]] == ["FS"]
        assert [p for p, _ in c["dropped"]] == ["FC"]

    def test_usage_shift_must_be_large_and_beyond_noise(self):
        frame = two_seasons(
            [{"pitch_type": "FF", "n": 600}, {"pitch_type": "SL", "n": 400}],
            [{"pitch_type": "FF", "n": 450}, {"pitch_type": "SL", "n": 550}],
        )
        c = seasons.season_changes(frame)
        assert {p for p, *_ in c["shifts"]} == {"FF", "SL"}

    def test_small_wobble_is_not_a_shift(self):
        frame = two_seasons(
            [{"pitch_type": "FF", "n": 600}, {"pitch_type": "SL", "n": 400}],
            [{"pitch_type": "FF", "n": 580}, {"pitch_type": "SL", "n": 420}],
        )
        assert seasons.season_changes(frame)["shifts"] == []

    def test_velocity_change_is_reported(self):
        frame = two_seasons(
            [{"pitch_type": "FF", "n": 200, "velo": 96.0}],
            [{"pitch_type": "FF", "n": 200, "velo": 94.5}],
        )
        (pitch, before, now), = seasons.season_changes(frame)["velo"]
        assert pitch == "FF"
        assert before == pytest.approx(96.0) and now == pytest.approx(94.5)

    def test_drop_retired_pitches_removes_only_the_shelved_pitch(self):
        frame = two_seasons(
            [{"pitch_type": "FF", "n": 500}, {"pitch_type": "FC", "n": 100}],
            [{"pitch_type": "FF", "n": 500}],
        )
        kept = seasons.drop_retired_pitches(frame, seasons.season_changes(frame))
        assert set(kept["pitch_type"]) == {"FF"}
        assert len(kept) == 1000


class TestLabelsAndCounts:
    def test_consecutive_seasons_read_as_a_range(self):
        assert seasons.season_label([2026, 2025]) == "2025–2026"

    def test_a_gap_is_listed(self):
        assert seasons.season_label([2024, 2026]) == "2024, 2026"

    def test_one_season_is_just_the_year(self):
        assert seasons.season_label([2026]) == "2026"

    def test_counts_per_season(self):
        frame = two_seasons([{"pitch_type": "FF", "n": 30}],
                            [{"pitch_type": "FF", "n": 70}])
        assert seasons.season_counts(frame) == [
            {"season": 2025, "pitches": 30, "games": 1},
            {"season": 2026, "pitches": 70, "games": 1},
        ]


class TestSinceLastSeasonLine:
    def test_names_every_kind_of_change(self):
        changes = {"latest": 2026, "earlier": [2025], "n_latest": 900, "n_earlier": 1500,
                   "new": [("FS", 0.12)], "dropped": [("FC", 0.08)],
                   "shifts": [("SL", 0.18, 0.30)], "velo": [("FF", 96.1, 94.7)]}
        key = gameplan.season_change_key(changes)
        assert key["label"] == "Since last season"
        assert "new splitter (12%)" in key["text"]
        assert "dropped the cutter" in key["text"] and "left out" in key["text"]
        assert "slider up to 30% from 18%" in key["text"]
        assert "four-seam down 1.4 mph" in key["text"]
        assert key["n"] == 900

    def test_says_so_when_nothing_changed(self):
        changes = {"latest": 2026, "earlier": [2025], "n_latest": 900, "n_earlier": 1500,
                   "new": [], "dropped": [], "shifts": [], "velo": []}
        assert "looks like 2025" in gameplan.season_change_key(changes)["text"]

    def test_absent_for_one_season(self):
        assert gameplan.season_change_key(None) is None


class TestCombinedReport:
    def build(self):
        return two_seasons(
            [{"pitch_type": "FF", "n": 400, "stand": "L"},
             {"pitch_type": "FF", "n": 400, "stand": "R"},
             {"pitch_type": "FC", "n": 150, "stand": "R",
              "events": "strikeout", "description": "swinging_strike",
              "type": "S", "is_swing": True, "is_whiff": True}],
            [{"pitch_type": "FF", "n": 300, "stand": "L"},
             {"pitch_type": "FF", "n": 300, "stand": "R"},
             {"pitch_type": "SL", "n": 150, "stand": "R"}],
        )

    def test_shelved_pitch_stays_out_of_the_plan_and_arsenal(self):
        payload = report.build_payload(self.build(), "Test Pitcher", "2025–2026")
        assert payload["meta"]["excluded_pitches"] == 150
        assert [s["season"] for s in payload["meta"]["seasons"]] == [2025, 2026]
        assert all(r["pitch_type"] != "FC" for r in payload["card"]["arsenal"])
        both = payload["card"]["plan"]["both"]
        assert both[0]["label"] == "Since last season"
        assert "dropped the cutter" in both[0]["text"]

    def test_season_line_still_counts_every_plate_appearance(self):
        """Removing the cutter's rows would remove the strikeouts it ended."""
        payload = report.build_payload(self.build(), "Test Pitcher", "2025–2026")
        assert payload["meta"]["pitches"] == 1700
        assert payload["card"]["season"]["k_rate"] > 0

    def test_renders(self):
        payload = report.build_payload(self.build(), "Test Pitcher", "2025–2026")
        html = report.render(payload)
        assert "Seasons combined" in html


class TestCli:
    def test_stack_tags_each_frame_with_its_season(self):
        a = make_frame([{"pitch_type": "FF", "n": 5}])
        b = make_frame([{"pitch_type": "FF", "n": 7}])
        stacked = cli.stack_seasons([a, b], [2025, 2026])
        assert stacked.groupby("game_year").size().to_dict() == {2025: 5, 2026: 7}

    def test_stack_skips_an_empty_season(self):
        a = make_frame([{"pitch_type": "FF", "n": 5}])
        stacked = cli.stack_seasons([a, pd.DataFrame()], [2025, 2026])
        assert len(stacked) == 5

    def test_thin_single_season_suggests_adding_the_last_one(self):
        text = cli.thin_sample_suggestion(900, [2026])
        assert "--seasons 2025 2026" in text

    def test_no_hint_for_a_full_season_or_a_combined_run(self):
        assert cli.thin_sample_suggestion(3000, [2026]) is None
        assert cli.thin_sample_suggestion(900, [2025, 2026]) is None


class TestLatelyWithCombinedSeasons:
    def test_lately_is_measured_against_the_current_season(self):
        """Snell's card said "changeup down to 17%" in both the "Since last
        season" and "Lately" lines, because "Lately" was measured against both
        seasons. It is now measured against the rest of the current one."""
        from src.metrics import splits
        specs_2025 = [{"pitch_type": "CH", "n": 40, "game_pk": g,
                       "game_date": f"2025-06-{g:02d}"} for g in range(1, 11)]
        specs_2026 = [{"pitch_type": "FF", "n": 40, "game_pk": 100 + g,
                       "game_date": f"2026-06-{g:02d}"} for g in range(1, 11)]
        old = make_frame(specs_2025)
        new = make_frame(specs_2026)
        old["game_year"], new["game_year"] = 2025, 2026
        form = splits.recent_form(pd.concat([old, new], ignore_index=True), last_n=5)
        assert form["earlier_label"] == "the rest of 2026"
        shifts = form["shifts"].set_index("pitch_type")
        assert shifts.loc["CH", "earlier_share"] == 0.0

    def test_velocity_change_matches_the_printed_numbers(self):
        from src.metrics import splits
        specs = ([{"pitch_type": "FF", "n": 20, "velo": 95.46, "game_pk": g,
                   "game_date": f"2026-05-{g:02d}"} for g in range(1, 6)]
                 + [{"pitch_type": "FF", "n": 20, "velo": 96.34, "game_pk": 10 + g,
                     "game_date": f"2026-07-{g:02d}"} for g in range(1, 6)])
        form = splits.recent_form(make_frame(specs), last_n=5)
        assert round(form["velo_recent"], 1) == 96.3
        assert round(form["velo_earlier"], 1) == 95.5
        assert form["velo_delta"] == pytest.approx(0.8)
