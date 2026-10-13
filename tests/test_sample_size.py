"""Tests for src/sample_size.py and scripts/sample_size.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import sample_size
from tests.test_replication import make_frame


def season(games: int, specs: list[dict] | None = None, game_type: str = "R",
           start: int = 1) -> pd.DataFrame:
    """`games` identical games, with distinct ids and increasing dates."""
    specs = specs or [{"pitch_type": "FF", "n": 70}]
    frames = []
    for g in range(start, start + games):
        frame = make_frame([{**s, "game_pk": g} for s in specs])
        frame["game_date"] = f"2025-{4 + g // 28:02d}-{g % 28 + 1:02d}"
        frame["game_type"] = game_type
        frame["at_bat_number"] += 1000 * g
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def doubles_up(slider_after_slider: int = 14, slider_after_fastball: int = 4) -> list[dict]:
    """One start of 40 two-pitch plate appearances to righties: 20 begin with a
    slider and 20 with a four-seam."""
    return [
        {"pitch_type": "SL", "n": slider_after_slider, "then": {"pitch_type": "SL"}},
        {"pitch_type": "SL", "n": 20 - slider_after_slider, "then": {"pitch_type": "FF"}},
        {"pitch_type": "FF", "n": slider_after_fastball, "then": {"pitch_type": "SL"}},
        {"pitch_type": "FF", "n": 20 - slider_after_fastball, "then": {"pitch_type": "FF"}},
    ]


def games_in(frame: pd.DataFrame) -> set:
    return set(frame["game_pk"])


# ---------------------------------------------------------------------------
# Which seasons are used
# ---------------------------------------------------------------------------


class TestStarterSeason:
    def test_twenty_starts_is_enough(self):
        assert sample_size.is_starter_season(season(20))

    def test_nineteen_is_not(self):
        """Half of 19 held out leaves ten on the training side, and the
        ten-start sample would then be the whole of it."""
        assert not sample_size.is_starter_season(season(19))

    def test_a_reliever_is_left_out_however_many_games(self):
        relief = season(60, [{"pitch_type": "FF", "n": 15}])
        assert not sample_size.is_starter_season(relief)

    def test_postseason_starts_do_not_count(self):
        frame = pd.concat([season(15), season(10, game_type="D", start=40)])
        assert not sample_size.is_starter_season(frame)

    def test_an_empty_season(self):
        assert not sample_size.is_starter_season(season(20).iloc[0:0])


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


class TestDraw:
    def draw(self, games: int = 24, seed: int = 1, sizes=(3, 5, 10)):
        return sample_size.draw(season(games), np.random.default_rng(seed), sizes)

    def test_no_game_is_on_both_sides(self):
        samples, held_out = self.draw()
        assert not games_in(samples[sample_size.HALF]) & games_in(held_out)

    def test_every_game_is_on_one_side(self):
        samples, held_out = self.draw()
        assert (games_in(samples[sample_size.HALF]) | games_in(held_out)
                == set(range(1, 25)))

    def test_an_odd_game_goes_to_the_training_side(self):
        samples, held_out = self.draw(games=25)
        assert len(games_in(samples[sample_size.HALF])) == 13
        assert len(games_in(held_out)) == 12

    def test_a_sample_has_the_number_of_starts_it_is_named_for(self):
        samples, _ = self.draw()
        assert [len(games_in(samples[n])) for n in (3, 5, 10)] == [3, 5, 10]

    def test_each_sample_is_inside_the_next(self):
        """So the difference between five starts and three is two starts, not
        a different five."""
        samples, _ = self.draw()
        assert (games_in(samples[3]) < games_in(samples[5]) < games_in(samples[10])
                < games_in(samples[sample_size.HALF]))

    def test_whole_games_are_drawn(self):
        """Sequencing is read inside a plate appearance, so a sample of loose
        pitches would have none."""
        samples, _ = self.draw()
        assert len(samples[3]) == 3 * 70

    def test_a_size_that_does_not_fit_is_left_out(self):
        samples, _ = self.draw(games=20, sizes=(3, 10, 12))
        assert set(samples) == {3, sample_size.HALF}

    def test_the_same_generator_state_gives_the_same_draw(self):
        (a, held_a), (b, held_b) = self.draw(seed=7), self.draw(seed=7)
        assert games_in(a[3]) == games_in(b[3])
        assert games_in(held_a) == games_in(held_b)

    def test_shuffles_differ(self):
        rng = np.random.default_rng(3)
        held = [frozenset(games_in(sample_size.draw(season(24), rng)[1]))
                for _ in range(5)]
        assert len(set(held)) > 1

    def test_postseason_games_are_never_drawn(self):
        frame = pd.concat([season(24), season(4, game_type="D", start=50)])
        samples, held_out = sample_size.draw(frame, np.random.default_rng(1))
        assert max(games_in(samples[sample_size.HALF]) | games_in(held_out)) == 24


class TestGenerator:
    def first(self, *args) -> float:
        return float(sample_size.rng_for(*args).random())

    def test_it_is_the_same_every_run(self):
        assert self.first("Logan Webb", 2026, 17) == self.first("Logan Webb", 2026, 17)

    def test_each_pitcher_season_and_seed_has_its_own(self):
        """A pitcher's draws must not depend on who ran before him."""
        base = self.first("Logan Webb", 2026, 17)
        assert base != self.first("Max Fried", 2026, 17)
        assert base != self.first("Logan Webb", 2025, 17)
        assert base != self.first("Logan Webb", 2026, 18)


# ---------------------------------------------------------------------------
# What a report prints
# ---------------------------------------------------------------------------


class TestPrinted:
    def test_a_habit_shows_up_on_ten_starts(self):
        """Ten starts: 200 follow-ups to each pitch, another slider after 140
        of the sliders and 40 of the four-seams."""
        out = sample_size.printed(season(10, doubles_up()))
        assert out["pitches"] == 10 * 80
        assert out["graph_rows"] == 2
        # Slider after slider is more than usual; slider after four-seam and
        # four-seam after slider are the shortfalls that go with it.
        assert out["graph_coloured"] >= 2
        assert out["sequence_lines"] == 1

    def test_one_start_cannot_read_a_row(self):
        """Twenty follow-ups to each pitch, under the graph's forty."""
        out = sample_size.printed(season(1, doubles_up()))
        assert out["graph_rows"] == 0
        assert out["graph_coloured"] == 0
        assert out["sequence_lines"] == 0

    def test_a_milder_habit_needs_more_starts(self):
        """Another slider 60% of the time after a slider and 30% after a
        four-seam. On two starts that is 24 of 40 against 12 of 40, which is
        2.7 standard errors: the rows are read and nothing is coloured. On
        ten it is the same shares at six standard errors."""
        milder = doubles_up(12, 6)
        two = sample_size.printed(season(2, milder))
        assert two["graph_rows"] == 2
        assert two["graph_coloured"] == 0
        assert sample_size.printed(season(10, milder))["graph_coloured"] >= 2

    def test_the_lately_line_is_not_counted(self):
        """It compares his last outings with the ones before, and a shuffled
        sample has no order."""
        early = [{"pitch_type": "FF", "n": 60}, {"pitch_type": "SL", "n": 10}]
        late = [{"pitch_type": "FF", "n": 20}, {"pitch_type": "SL", "n": 50}]
        frame = pd.concat([season(10, early), season(5, late, start=11)])
        from src import gameplan
        labels = [k["label"] for k in gameplan.build(frame)["both"]]
        assert "Lately" in labels
        keys = gameplan.card_keys(gameplan.build(frame))
        counted = sum(len(v) for v in keys.values()) - 1
        assert sample_size.printed(frame)["card_lines"] == counted

    def test_no_trajectory_means_no_tunneling_and_no_error(self):
        out = sample_size.printed(season(10, doubles_up()))
        assert out["tunnel_pairs"] == 0 and out["tunnel_read"] == 0


# ---------------------------------------------------------------------------
# Whether it was right
# ---------------------------------------------------------------------------


def of_kind(rows: list[dict], kind: str) -> list[dict]:
    return [r for r in rows if r["kind"] == kind]


class TestScored:
    def test_a_habit_that_continues_held(self):
        rows = sample_size.scored(season(10, doubles_up()), season(10, doubles_up()))
        (row,) = [r for r in of_kind(rows, "follow_up_more")
                  if r["key"] == "SL > SL to RHH"]
        assert row["held"] and not row["wrong"]
        assert row["first"] == pytest.approx(0.70 / 0.45)

    def test_a_habit_that_shrinks_did_not_hold_and_is_not_wrong(self):
        """50% after a slider against 45% after anything: still more than
        usual, under the bar. This is what a small sample mostly does."""
        rows = sample_size.scored(season(10, doubles_up()),
                                  season(10, doubles_up(10, 8)))
        (row,) = [r for r in of_kind(rows, "follow_up_more")
                  if r["key"] == "SL > SL to RHH"]
        assert not row["held"] and not row["wrong"]

    def test_a_habit_that_reverses_is_wrong(self):
        rows = sample_size.scored(season(10, doubles_up()),
                                  season(10, doubles_up(4, 14)))
        (row,) = [r for r in of_kind(rows, "follow_up_more")
                  if r["key"] == "SL > SL to RHH"]
        assert not row["held"] and row["wrong"]

    def test_plan_lines_are_scored_too(self):
        behind = [{"pitch_type": "FF", "n": 30, "balls": 2},
                  {"pitch_type": "SL", "n": 40}]
        rows = sample_size.scored(season(10, behind), season(10, behind))
        (row,) = of_kind(rows, "hitters_count")
        assert row["key"] == "R sit hard"
        assert row["held"] and not row["wrong"]

    def test_he_mixes_is_not_a_claim(self):
        """The holdout check counts only advice, and so does this."""
        mixes = [{"pitch_type": "FF", "n": 15, "balls": 2},
                 {"pitch_type": "SL", "n": 15, "balls": 2},
                 {"pitch_type": "CH", "n": 40}]
        rows = sample_size.scored(season(10, mixes), season(10, mixes))
        assert of_kind(rows, "hitters_count") == []

    def test_a_plan_line_that_names_another_pitch_is_wrong(self):
        hard = [{"pitch_type": "FF", "n": 30, "balls": 2},
                {"pitch_type": "SL", "n": 40}]
        soft = [{"pitch_type": "SL", "n": 30, "balls": 2},
                {"pitch_type": "FF", "n": 40}]
        rows = sample_size.scored(season(10, hard), season(10, soft))
        (row,) = of_kind(rows, "hitters_count")
        assert not row["held"] and row["wrong"]

    def test_nothing_held_out_scores_nothing(self):
        frame = season(10, doubles_up())
        assert sample_size.scored(frame, frame.iloc[0:0]) == []
        assert sample_size.scored(frame.iloc[0:0], frame) == []


class TestFollowUpShares:
    def test_the_share_in_each_part(self):
        rows = sample_size.follow_up_shares(season(3, doubles_up()),
                                            season(10, doubles_up(10, 8)))
        (row,) = [r for r in rows
                  if (r["pitch_type"], r["next_pitch"]) == ("SL", "SL")]
        assert row["stand"] == "R"
        assert row["row_n"] == 60
        assert row["first"] == pytest.approx(0.70)
        assert row["second"] == pytest.approx(0.50)

    def test_every_follow_up_of_a_readable_row_is_there(self):
        """Coloured or not: the graph draws them all."""
        rows = sample_size.follow_up_shares(season(3, doubles_up()),
                                            season(10, doubles_up()))
        assert len(rows) == 4

    def test_a_row_the_sample_cannot_read_is_left_out(self):
        rows = sample_size.follow_up_shares(season(1, doubles_up()),
                                            season(10, doubles_up()))
        assert rows == []

    def test_a_row_the_held_out_games_cannot_read_is_left_out(self):
        rows = sample_size.follow_up_shares(season(10, doubles_up()),
                                            season(1, doubles_up()))
        assert rows == []


# ---------------------------------------------------------------------------
# One pitcher
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def result() -> dict:
    return sample_size.run_pitcher(season(24, doubles_up()), "Test Pitcher",
                                   2025, draws=2, seed=17)


class TestRunPitcher:
    def test_every_draw_at_every_size_and_the_control(self, result):
        sizes = [(r["draw"], r["starts"]) for r in result["printed"]]
        assert sizes == [(0, 3), (0, 5), (0, 10), (0, sample_size.HALF),
                         (1, 3), (1, 5), (1, 10), (1, sample_size.HALF),
                         (0, sample_size.BY_DATE)]

    def test_each_row_says_whose_it_is(self, result):
        for table in result.values():
            assert {(r["pitcher"], r["season"]) for r in table} == {("Test Pitcher", 2025)}

    def test_the_control_is_the_first_half_by_date(self, result):
        control = result["printed"][-1]
        assert control["games"] == 12
        assert control["held_out_pitches"] == 12 * 80

    def test_the_held_out_side_is_the_same_for_every_size_in_a_draw(self, result):
        held = {r["draw"]: set() for r in result["printed"]
                if r["starts"] != sample_size.BY_DATE}
        for r in result["printed"]:
            if r["starts"] != sample_size.BY_DATE:
                held[r["draw"]].add(r["held_out_pitches"])
        assert all(len(v) == 1 for v in held.values())

    def test_a_steady_habit_is_found_once_the_sample_is_big_enough(self, result):
        by = {(r["draw"], r["starts"]): r for r in result["printed"]}
        assert by[(0, 10)]["graph_coloured"] >= 2
        found = [r for r in result["scored"]
                 if r["kind"] == "follow_up_more" and r["starts"] == 10]
        assert found and all(r["held"] for r in found)

    def test_it_is_reproducible(self, result):
        again = sample_size.run_pitcher(season(24, doubles_up()), "Test Pitcher",
                                        2025, draws=2, seed=17)
        assert again == result


# ---------------------------------------------------------------------------
# Summarising
# ---------------------------------------------------------------------------


def claims(pitcher: str, held: list[bool], starts=3, kind="follow_up_more",
           wrong: list[bool] | None = None) -> list[dict]:
    wrong = wrong or [False] * len(held)
    return [{"pitcher": pitcher, "season": 2025, "draw": 0, "starts": starts,
             "kind": kind, "key": "SL > SL to RHH", "first": 1.6, "second": 1.2,
             "held": h, "wrong": w} for h, w in zip(held, wrong)]


def reports(n: int, starts=3, pitcher: str = "A", **counts) -> list[dict]:
    blank = dict.fromkeys(sample_size.PRINTED_LABELS, 0)
    return [{"pitcher": pitcher, "season": 2025, "draw": i, "starts": starts,
             **blank, **counts} for i in range(n)]


class TestPitcherInterval:
    def test_two_pitchers_is_too_few(self):
        table = pd.DataFrame(claims("A", [True]) + claims("B", [False]))
        lo, hi = sample_size.pitcher_interval(table, "held")
        assert lo != lo and hi != hi

    def test_pitchers_who_agree_leave_no_width(self):
        table = pd.DataFrame(claims("A", [True] * 5) + claims("B", [True] * 5)
                             + claims("C", [True] * 5))
        assert sample_size.pitcher_interval(table, "held") == (1.0, 1.0)

    def test_it_resamples_pitchers_not_rows(self):
        """Forty claims from one arm that all held and one each from two that
        did not: by rows that is 95% and a tight interval, but leave that one
        pitcher out of a resample and the rate is zero."""
        table = pd.DataFrame(claims("A", [True] * 40) + claims("B", [False])
                             + claims("C", [False]))
        lo, hi = sample_size.pitcher_interval(table, "held")
        assert lo == 0.0
        assert hi > 0.95

    def test_an_empty_table(self):
        lo, hi = sample_size.pitcher_interval(pd.DataFrame(), "held")
        assert lo != lo and hi != hi


class TestSummarisePrinted:
    def rows(self) -> list[dict]:
        return (reports(2, starts=sample_size.BY_DATE)
                + reports(4, starts=sample_size.HALF, graph_coloured=2)
                + reports(3, starts=10, graph_coloured=1)
                + reports(1, starts=3, graph_coloured=3) + reports(3, starts=3))

    def line(self, starts, metric="graph_coloured") -> dict:
        (row,) = [r for r in sample_size.summarise_printed(self.rows())
                  if r["starts"] == starts and r["metric"] == metric]
        return row

    def test_sizes_run_smallest_first_and_the_control_last(self):
        sizes = list(dict.fromkeys(
            r["starts"] for r in sample_size.summarise_printed(self.rows())))
        assert sizes == [3, 10, sample_size.HALF, sample_size.BY_DATE]

    def test_the_mean_is_per_report(self):
        assert self.line(3)["mean"] == pytest.approx(0.75)
        assert self.line(3)["reports"] == 4

    def test_any_is_the_share_of_reports_that_say_something(self):
        """An average of 0.75 here is one report in four with any at all."""
        assert self.line(3)["any"] == pytest.approx(0.25)
        assert self.line(10)["any"] == pytest.approx(1.0)

    def test_nothing_in(self):
        assert sample_size.summarise_printed([]) == []


class TestSummariseScored:
    def test_per_report_counts_the_reports_that_claimed_nothing(self):
        out = sample_size.summarise_scored(claims("A", [True, False]),
                                           reports(8))
        (row,) = [r for r in out if r["kind"] == "follow_up_more"]
        assert row["claims"] == 2
        assert row["per_report"] == pytest.approx(0.25)

    def test_held_and_wrong_are_shares_of_the_claims(self):
        rows = claims("A", [True, False, False, False],
                      wrong=[False, False, False, True])
        (row,) = [r for r in sample_size.summarise_scored(rows, reports(4))
                  if r["kind"] == "follow_up_more"]
        assert row["held_pct"] == pytest.approx(0.25)
        assert row["wrong_pct"] == pytest.approx(0.25)
        assert row["median_first"] == pytest.approx(1.6)
        assert row["median_second"] == pytest.approx(1.2)

    def test_a_kind_nobody_claimed_is_a_zero_not_a_gap(self):
        out = sample_size.summarise_scored(claims("A", [True]), reports(4))
        (row,) = [r for r in out if r["kind"] == "tunnel_same"]
        assert row["claims"] == 0
        assert row["per_report"] == 0
        assert row["held_pct"] != row["held_pct"]

    def test_each_size_is_its_own_line(self):
        rows = claims("A", [True]) + claims("A", [False, False], starts=10)
        out = [r for r in sample_size.summarise_scored(
                   rows, reports(4) + reports(4, starts=10))
               if r["kind"] == "follow_up_more"]
        assert [(r["starts"], r["held_pct"]) for r in out] == [(3, 1.0), (10, 0.0)]

    def test_nothing_in(self):
        assert sample_size.summarise_scored([], reports(4)) == []


class TestSummariseShares:
    def rows(self, gaps: list[float], starts=3) -> list[dict]:
        return [{"pitcher": "A", "season": 2025, "draw": 0, "starts": starts,
                 "stand": "R", "pitch_type": "SL", "next_pitch": "SL",
                 "row_n": 60, "first": 0.40 + gap, "second": 0.40}
                for gap in gaps]

    def test_gaps_are_in_share_and_either_direction_counts(self):
        (row,) = sample_size.summarise_shares(self.rows([0.02, -0.04, 0.12]))
        assert row["cells"] == 3
        assert row["median_gap"] == pytest.approx(0.04)
        assert row["within_five"] == pytest.approx(2 / 3)
        assert row["within_ten"] == pytest.approx(2 / 3)

    def test_nothing_in(self):
        assert sample_size.summarise_shares([]) == []


# ---------------------------------------------------------------------------
# The script
# ---------------------------------------------------------------------------


class TestScript:
    def test_runs_offline_on_a_fixture(self, capsys):
        import csv
        import importlib.util
        import tempfile
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            "sample_size_script", root / "scripts" / "sample_size.py")
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            code = script.main(["--offline", "--season", "2025", "--draws", "1",
                                "--names", "Tarik Skubal", "--out", str(out)])
            assert code == 0
            with (out / "printed.csv").open() as handle:
                printed = list(csv.DictReader(handle))
            with (out / "scored.csv").open() as handle:
                kinds = {row["kind"] for row in csv.DictReader(handle)}
            assert (out / "shares.csv").exists()

        assert [r["starts"] for r in printed] == ["3", "5", "10", "half", "by date"]
        assert int(printed[0]["pitches"]) < int(printed[2]["pitches"])
        assert {"two_strikes", "wont_see"} <= kinds
        said = capsys.readouterr().out
        assert "1 pitchers, 1 seasons, 1 shuffles" in said
        assert "not college pitchers" in said
