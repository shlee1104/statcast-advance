"""Tests for src/metrics/sequencing.py.

Frames are built from explicit plate appearances so expected transition counts
can be worked out by hand.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.metrics import sequencing


def make_at_bats(
    at_bats: list[list[str]],
    descriptions: list[list[str]] | None = None,
    events: list[str] | None = None,
    strikes: list[list[int]] | None = None,
) -> pd.DataFrame:
    """Build a frame from a list of plate appearances.

    Each plate appearance is a list of pitch types in the order thrown.
    """
    rows = []
    for pa_index, pitches in enumerate(at_bats, start=1):
        for pitch_index, pitch_type in enumerate(pitches, start=1):
            description = (
                descriptions[pa_index - 1][pitch_index - 1]
                if descriptions else "called_strike"
            )
            strike_count = (
                strikes[pa_index - 1][pitch_index - 1]
                if strikes else 0
            )
            is_last = pitch_index == len(pitches)
            rows.append({
                "game_pk": 1,
                "at_bat_number": pa_index,
                "pitch_number": pitch_index,
                "pitch_type": pitch_type,
                "description": description,
                "events": (events[pa_index - 1] if events and is_last else None),
                "strikes": strike_count,
                "is_two_strike": strike_count == 2,
                "is_swing": description in {
                    "swinging_strike", "foul", "hit_into_play", "foul_tip",
                },
                "is_whiff": description == "swinging_strike",
                "is_contact": description in {"foul", "hit_into_play", "foul_tip"},
            })
    return pd.DataFrame(rows)


class TestTransitionMatrix:
    def test_simple_transitions(self):
        """FF is followed by SL twice and CH once -> 2/3 and 1/3."""
        frame = make_at_bats([["FF", "SL"], ["FF", "SL"], ["FF", "CH"]])
        matrix = sequencing.transition_matrix(frame)
        assert matrix.loc["FF", "SL"] == pytest.approx(2 / 3)
        assert matrix.loc["FF", "CH"] == pytest.approx(1 / 3)

    def test_rows_sum_to_one(self):
        frame = make_at_bats([["FF", "SL", "CH"], ["SL", "FF"], ["CH", "CH"]])
        matrix = sequencing.transition_matrix(frame)
        for _, row in matrix.iterrows():
            assert row.sum() == pytest.approx(1.0)

    def test_does_not_pair_across_plate_appearances(self):
        """Two single-pitch PAs contain zero transitions, not one.

        If this fails, the code is treating the end of one PA as setting up
        the start of the next, which fabricates data.
        """
        frame = make_at_bats([["FF"], ["SL"]])
        matrix = sequencing.transition_matrix(frame)
        assert matrix.empty or matrix.sum().sum() == 0

    def test_last_pitch_of_pa_has_no_successor(self):
        """One 3-pitch PA yields exactly 2 transitions, not 3."""
        frame = make_at_bats([["FF", "SL", "CH"]])
        matrix = sequencing.transition_matrix(frame)
        # FF->SL and SL->CH. CH is terminal.
        assert matrix.loc["FF", "SL"] == pytest.approx(1.0)
        assert matrix.loc["SL", "CH"] == pytest.approx(1.0)
        assert "CH" not in matrix.index

    def test_respects_pitch_number_order_not_row_order(self):
        """Shuffled rows must produce the same answer."""
        frame = make_at_bats([["FF", "SL", "CH"]])
        shuffled = frame.iloc[::-1].reset_index(drop=True)
        matrix = sequencing.transition_matrix(shuffled)
        assert matrix.loc["FF", "SL"] == pytest.approx(1.0)

    def test_back_to_back_same_pitch(self):
        frame = make_at_bats([["SL", "SL"], ["SL", "SL"], ["SL", "FF"]])
        matrix = sequencing.transition_matrix(frame)
        assert matrix.loc["SL", "SL"] == pytest.approx(2 / 3)

    def test_missing_combinations_are_zero(self):
        frame = make_at_bats([["FF", "SL"]])
        matrix = sequencing.transition_matrix(frame)
        assert not matrix.isna().any().any()

    def test_min_transitions_gate(self):
        frame = make_at_bats([
            ["FF", "SL"], ["FF", "SL"], ["FF", "SL"],
            ["CU", "CH"],
        ])
        matrix = sequencing.transition_matrix(frame, min_transitions=3)
        assert "FF" in matrix.index
        assert "CU" not in matrix.index

    def test_empty_frame(self):
        frame = make_at_bats([["FF", "SL"]]).iloc[0:0]
        matrix = sequencing.transition_matrix(frame)
        assert len(matrix) == 0


class TestTransitionAfterOutcome:
    def test_conditions_on_whiff(self):
        """Only the PA where the first pitch was whiffed should count."""
        frame = make_at_bats(
            [["SL", "SL"], ["SL", "FF"]],
            descriptions=[
                ["swinging_strike", "called_strike"],
                ["called_strike", "called_strike"],
            ],
        )
        matrix = sequencing.transition_after_outcome(frame, "whiff")
        assert matrix.loc["SL", "SL"] == pytest.approx(1.0)
        assert matrix.sum().sum() == pytest.approx(1.0)

    def test_conditions_on_take(self):
        frame = make_at_bats(
            [["FF", "SL"], ["FF", "CH"]],
            descriptions=[
                ["ball", "called_strike"],
                ["swinging_strike", "called_strike"],
            ],
        )
        matrix = sequencing.transition_after_outcome(frame, "take")
        assert matrix.loc["FF", "SL"] == pytest.approx(1.0)
        assert "CH" not in matrix.columns or matrix.loc["FF", "CH"] == 0.0

    def test_conditions_on_foul(self):
        frame = make_at_bats(
            [["FF", "SL"], ["FF", "CH"]],
            descriptions=[
                ["foul", "called_strike"],
                ["ball", "called_strike"],
            ],
        )
        matrix = sequencing.transition_after_outcome(frame, "foul")
        assert matrix.loc["FF", "SL"] == pytest.approx(1.0)

    def test_unknown_outcome_raises(self):
        frame = make_at_bats([["FF", "SL"]])
        with pytest.raises((ValueError, KeyError)):
            sequencing.transition_after_outcome(frame, "bunt_attempt")


class TestPutaway:
    def test_only_counts_two_strike_pitches(self):
        frame = make_at_bats(
            [["FF", "SL"]] * 10,
            strikes=[[0, 2]] * 10,
            events=["strikeout"] * 10,
        )
        table = sequencing.putaway(frame, min_pitches=1)
        assert "SL" in table.index
        assert "FF" not in table.index
        assert table.loc["SL", "n"] == 10

    def test_putaway_rate(self):
        """6 of 10 two-strike sliders end in a strikeout -> 0.60."""
        frame = make_at_bats(
            [["SL"]] * 10,
            strikes=[[2]] * 10,
            events=["strikeout"] * 6 + ["field_out"] * 4,
        )
        table = sequencing.putaway(frame, min_pitches=1)
        assert table.loc["SL", "strikeouts"] == 6
        assert table.loc["SL", "putaway_rate"] == pytest.approx(0.60)

    def test_usage_shares_sum_to_one(self):
        frame = make_at_bats(
            [["SL"]] * 30 + [["FF"]] * 10,
            strikes=[[2]] * 40,
            events=["field_out"] * 40,
        )
        table = sequencing.putaway(frame, min_pitches=1)
        assert table["usage"].sum() == pytest.approx(1.0)
        assert table.loc["SL", "usage"] == pytest.approx(0.75)

    def test_sorted_by_usage_descending(self):
        frame = make_at_bats(
            [["SL"]] * 30 + [["FF"]] * 10 + [["CH"]] * 20,
            strikes=[[2]] * 60,
            events=["field_out"] * 60,
        )
        table = sequencing.putaway(frame, min_pitches=1)
        assert list(table.index) == ["SL", "CH", "FF"]

    def test_min_pitches_gate(self):
        frame = make_at_bats(
            [["SL"]] * 30 + [["CU"]] * 3,
            strikes=[[2]] * 33,
            events=["field_out"] * 33,
        )
        table = sequencing.putaway(frame, min_pitches=10)
        assert "SL" in table.index
        assert "CU" not in table.index

    def test_counts_strikeout_double_play(self):
        frame = make_at_bats(
            [["SL"]] * 10,
            strikes=[[2]] * 10,
            events=["strikeout_double_play"] * 4 + ["field_out"] * 6,
        )
        table = sequencing.putaway(frame, min_pitches=1)
        assert table.loc["SL", "strikeouts"] == 4

    def test_empty_frame(self):
        frame = make_at_bats([["SL"]], strikes=[[2]]).iloc[0:0]
        table = sequencing.putaway(frame)
        assert len(table) == 0


SWING_DESCRIPTIONS = {"swinging_strike", "foul", "hit_into_play", "foul_tip"}


def make_pairs(
    specs: list[tuple],
    balls: int = 0,
    strikes: int = 0,
) -> pd.DataFrame:
    """Build two-pitch plate appearances from explicit setup -> follow specs.

    Each spec is (setup_pitch, setup_zone, next_pitch, repetitions,
    next_description).
    """
    rows = []
    pa = 0
    for setup_pitch, setup_zone, next_pitch, n, next_description in specs:
        for _ in range(n):
            pa += 1
            rows.append({
                "game_pk": 1, "at_bat_number": pa, "pitch_number": 1,
                "pitch_type": setup_pitch, "zone": setup_zone,
                "description": "called_strike", "events": None,
                "balls": balls, "strikes": strikes,
                "is_swing": False, "is_whiff": False,
            })
            rows.append({
                "game_pk": 1, "at_bat_number": pa, "pitch_number": 2,
                "pitch_type": next_pitch, "zone": 5,
                "description": next_description, "events": None,
                "balls": balls, "strikes": strikes,
                "is_swing": next_description in SWING_DESCRIPTIONS,
                "is_whiff": next_description == "swinging_strike",
            })
    return pd.DataFrame(rows)


class TestSetupPairs:
    def test_frequency_lift(self):
        """FF up always precedes CU; CU is 40% of follow-ups overall.

        p_next = 1.0, baseline = 0.4, so freq_lift = 2.5.
        """
        frame = make_pairs([
            ("FF", 2, "CU", 40, "called_strike"),   # zone 2 -> UP
            ("FF", 8, "FF", 60, "called_strike"),   # zone 8 -> DOWN
        ])
        table = sequencing.setup_pairs(frame, min_n=25)
        row = table[
            (table["setup_pitch"] == "FF")
            & (table["setup_band"] == "UP")
            & (table["next_pitch"] == "CU")
        ].iloc[0]
        assert row["n"] == 40
        assert row["p_next"] == pytest.approx(1.0)
        assert row["baseline_p"] == pytest.approx(0.4)
        assert row["freq_lift"] == pytest.approx(2.5)

    def test_zone_banding(self):
        """Zones 1-3 and 11-12 are UP; 7-9 and 13-14 are DOWN; 4-6 are MID."""
        frame = make_pairs([
            ("FF", 1, "CU", 30, "called_strike"),
            ("FF", 5, "CU", 30, "called_strike"),
            ("FF", 13, "CU", 30, "called_strike"),
        ])
        table = sequencing.setup_pairs(frame, min_n=25)
        assert set(table["setup_band"]) == {"UP", "MID", "DOWN"}

    def test_effect_lift(self):
        """Splitter whiffs 100% after FF up, 0% after SL. Baseline is 50%."""
        frame = make_pairs([
            ("FF", 2, "FS", 40, "swinging_strike"),
            ("SL", 5, "FS", 40, "foul"),
        ])
        table = sequencing.setup_pairs(frame, min_n=25)

        after_ff = table[table["setup_pitch"] == "FF"].iloc[0]
        after_sl = table[table["setup_pitch"] == "SL"].iloc[0]

        assert after_ff["whiff_after"] == pytest.approx(1.0)
        assert after_ff["whiff_baseline"] == pytest.approx(0.5)
        assert after_ff["effect_lift"] == pytest.approx(2.0)
        assert after_sl["effect_lift"] == pytest.approx(0.0)

    def test_min_n_gate(self):
        frame = make_pairs([
            ("FF", 2, "CU", 40, "called_strike"),
            ("SL", 2, "CH", 10, "called_strike"),
        ])
        table = sequencing.setup_pairs(frame, min_n=25)
        assert "CH" not in set(table["next_pitch"])
        assert "CU" in set(table["next_pitch"])

    def test_sorted_by_score_descending(self):
        frame = make_pairs([
            ("FF", 2, "CU", 40, "called_strike"),
            ("FF", 8, "FF", 60, "called_strike"),
        ])
        table = sequencing.setup_pairs(frame, min_n=25)
        assert list(table["score"]) == sorted(table["score"], reverse=True)

    def test_unknown_zone_excluded(self):
        """A pitch Statcast could not locate cannot be attributed to a band."""
        frame = make_pairs([("FF", None, "CU", 40, "called_strike")])
        table = sequencing.setup_pairs(frame, min_n=25)
        assert len(table) == 0

    def test_count_state_filter(self):
        frame = make_pairs(
            [("FF", 2, "CU", 40, "called_strike")], balls=0, strikes=2
        )
        assert len(sequencing.setup_pairs(frame, min_n=25, count_state="ahead")) == 1
        assert len(sequencing.setup_pairs(frame, min_n=25, count_state="behind")) == 0

    def test_count_state_is_the_count_the_follow_up_is_thrown_in(self):
        """The pitcher picks the next pitch in the count the last one left.

        A first-pitch four-seam taken for a strike makes it 0-1: the setup was
        thrown even, the curveball that follows is thrown ahead. Filtering on
        the setup's count called that pair "even" and so held nothing fixed.
        """
        frame = make_pairs([("FF", 2, "CU", 40, "called_strike")])
        frame.loc[frame["pitch_number"] == 2, "strikes"] = 1
        assert len(sequencing.setup_pairs(frame, min_n=25, count_state="ahead")) == 1
        assert len(sequencing.setup_pairs(frame, min_n=25, count_state="even")) == 0

    def test_unknown_count_state_raises(self):
        frame = make_pairs([("FF", 2, "CU", 40, "called_strike")])
        with pytest.raises(ValueError):
            sequencing.setup_pairs(frame, min_n=25, count_state="tied")

    def test_empty_frame_returns_columns(self):
        frame = make_pairs([("FF", 2, "CU", 2, "called_strike")]).iloc[0:0]
        table = sequencing.setup_pairs(frame)
        assert len(table) == 0
        for column in ["setup_pitch", "setup_band", "next_pitch", "freq_lift"]:
            assert column in table.columns


def sided(groups: list[tuple[str, int, list[str]]]) -> pd.DataFrame:
    """Plate appearances by batter side: (side, how many, pitches in order)."""
    rows = []
    at_bat = 0
    for stand, times, pitches in groups:
        for _ in range(times):
            at_bat += 1
            for number, pitch_type in enumerate(pitches, start=1):
                rows.append({
                    "game_pk": 1,
                    "at_bat_number": at_bat,
                    "pitch_number": number,
                    "pitch_type": pitch_type,
                    "stand": stand,
                    "description": "called_strike",
                    "zone": 5,
                    "balls": 0,
                    "strikes": min(number - 1, 2),
                    "is_swing": False,
                    "is_whiff": False,
                })
    return pd.DataFrame(rows)


def cell(table: pd.DataFrame, stand: str, pitch: str, next_pitch: str) -> pd.Series:
    match = table[(table["stand"] == stand) & (table["pitch_type"] == pitch)
                  & (table["next_pitch"] == next_pitch)]
    assert len(match) == 1
    return match.iloc[0]


class TestTransitions:
    def frame(self) -> pd.DataFrame:
        """Six follow-ups to righties: FF>SL three times, FF>FF once, SL>FF twice."""
        return sided([("R", 3, ["FF", "SL"]), ("R", 1, ["FF", "FF"]),
                      ("R", 2, ["SL", "FF"])])

    def test_carries_the_counts_a_share_is_made_of(self):
        row = cell(sequencing.transitions(self.frame()), "R", "FF", "SL")
        assert (row["n"], row["row_n"], row["next_n"], row["total"]) == (3, 4, 3, 6)
        assert row["share"] == pytest.approx(0.75)

    def test_lift_is_against_how_often_the_pitch_follows_anything(self):
        """Sliders are 3 of the 6 follow-ups, so 50% is his usual and 75%
        after a four-seam is 1.5 times it. Two would be expected at his
        usual rate."""
        row = cell(sequencing.transitions(self.frame()), "R", "FF", "SL")
        assert row["usual"] == pytest.approx(0.50)
        assert row["lift"] == pytest.approx(1.5)
        assert row["expected_n"] == pytest.approx(2.0)

    def test_a_follow_up_that_never_came_is_a_row_not_a_gap(self):
        """He never doubled up on the slider. That is the finding, and a
        missing row could not carry it."""
        row = cell(sequencing.transitions(self.frame()), "R", "SL", "SL")
        assert row["n"] == 0
        assert row["share"] == 0.0
        assert row["expected_n"] == pytest.approx(1.0)

    def test_each_row_of_shares_sums_to_one(self):
        table = sequencing.transitions(self.frame())
        for _, row in table.groupby(["stand", "pitch_type"]):
            assert row["share"].sum() == pytest.approx(1.0)

    def test_each_batter_side_is_its_own_table(self):
        """He does not throw the same pitches to both sides, so "his usual"
        to a lefty is measured on lefties."""
        frame = sided([("R", 8, ["FF", "SL"]), ("L", 6, ["FF", "CH"]),
                       ("L", 2, ["FF", "FF"])])
        table = sequencing.transitions(frame)
        assert cell(table, "R", "FF", "SL")["share"] == pytest.approx(1.0)
        assert cell(table, "L", "FF", "CH")["share"] == pytest.approx(0.75)
        assert cell(table, "L", "FF", "CH")["total"] == 8
        assert set(table[table["stand"] == "L"]["next_pitch"]) == {"CH", "FF"}

    def test_usual_is_among_follow_ups_not_all_pitches(self):
        """The first pitch of a plate appearance follows nothing. Twenty
        one-pitch plate appearances of curveballs do not make the curveball a
        common follow-up."""
        frame = pd.concat([self.frame(),
                           sided([("R", 20, ["CU"])]).assign(game_pk=2)])
        table = sequencing.transitions(frame)
        assert cell(table, "R", "FF", "SL")["usual"] == pytest.approx(0.50)
        assert "CU" not in set(table["next_pitch"])

    def test_does_not_pair_across_plate_appearances(self):
        frame = sided([("R", 1, ["FF"]), ("R", 1, ["SL"])])
        assert len(sequencing.transitions(frame)) == 0

    def test_a_stray_pitch_does_not_join_its_neighbours(self):
        """One knuckleball in four hundred pitches is a mislabel, not part of
        his arsenal. Dropping it before pairing would close the gap and
        invent a four-seam-to-slider follow-up that never happened."""
        frame = pd.concat([
            sided([("R", 100, ["FF", "FF"]), ("R", 100, ["SL", "SL"])]),
            sided([("R", 1, ["FF", "KN", "SL"])]).assign(game_pk=2),
        ])
        table = sequencing.transitions(frame)
        assert "KN" not in set(table["pitch_type"]) | set(table["next_pitch"])
        assert cell(table, "R", "FF", "SL")["n"] == 0

    def test_empty_frame(self):
        frame = sided([("R", 1, ["FF"])]).iloc[0:0]
        table = sequencing.transitions(frame)
        assert len(table) == 0
        assert list(table.columns) == sequencing.TRANSITION_COLUMNS


class TestTransitionStandouts:
    """Which follow-ups the sequence graph colours."""

    def label(self, **values) -> str:
        row = {"n": 30, "row_n": 100, "lift": 1.5, "lift_z": 4.0, "expected_n": 20.0}
        row.update(values)
        return sequencing.transition_standouts(pd.DataFrame([row])).iloc[0]

    def test_more_often_than_usual_and_clear_of_chance(self):
        assert self.label() == "more"

    def test_a_lift_inside_chance_is_not_coloured(self):
        """Thirty where twenty were expected, on a hundred: 1.5 times usual
        and only two standard errors from nothing."""
        assert self.label(lift_z=2.0) == ""

    def test_a_small_lift_is_not_coloured_however_sure(self):
        assert self.label(lift=1.1, lift_z=6.0) == ""

    def test_more_needs_enough_occurrences(self):
        assert self.label(n=12) == ""

    def test_a_pitch_with_too_few_follow_ups_is_not_read(self):
        assert self.label(row_n=30) == ""
        assert self.label(row_n=30, n=4, lift=0.4, lift_z=-3.5, expected_n=25.0) == ""

    def test_less_often_than_usual(self):
        assert self.label(n=8, lift=0.4, lift_z=-3.5, expected_n=20.0) == "less"

    def test_never_is_only_news_if_there_would_normally_be_some(self):
        """No sweepers after a changeup, where eight would be expected, is
        not a shortfall anyone could act on."""
        assert self.label(n=0, lift=0.0, lift_z=-3.2, expected_n=8.0) == ""

    def test_no_table_no_labels(self):
        empty = pd.DataFrame(columns=sequencing.TRANSITION_COLUMNS)
        assert len(sequencing.transition_standouts(empty)) == 0

    def test_doubling_up_is_found_from_pitches(self):
        """After a slider the next pitch is a slider 70 times in 100; after a
        four-seam, 20 in 100."""
        frame = sided([("R", 70, ["SL", "SL"]), ("R", 30, ["SL", "FF"]),
                       ("R", 20, ["FF", "SL"]), ("R", 80, ["FF", "FF"])])
        table = sequencing.transitions(frame)
        table["standout"] = sequencing.transition_standouts(table)
        assert cell(table, "R", "SL", "SL")["standout"] == "more"
        assert cell(table, "R", "FF", "SL")["standout"] == "less"
        assert cell(table, "R", "SL", "SL")["lift"] == pytest.approx(0.70 / 0.45)
