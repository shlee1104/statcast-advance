"""Tests for src/flags.py.

Two things are being protected here. The first is that no rule fires below its
gate, because a flag that reports a tendency off nine pitches is the specific
failure this project exists to avoid. The second is that severity ranks a
trustworthy moderate finding above a dramatic thin one — the ordering bug that
the first version of the severity formula had, where effect size was scored
without reference to sample size at all.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src import flags


def make_frame(specs: list[dict]) -> pd.DataFrame:
    """Build a frame carrying every column the rules touch.

    Defaults describe an unremarkable four-seam over the middle, taken for a
    called strike, so a spec only has to state what makes its group different.
    """
    rows = []
    at_bat = 0
    for spec in specs:
        for i in range(spec["n"]):
            at_bat += 1
            balls = spec.get("balls", 0)
            strikes = spec.get("strikes", 0)
            rows.append({
                "game_pk": spec.get("game_pk", 1),
                "at_bat_number": at_bat,
                "pitch_number": spec.get("pitch_number", 1),
                "balls": balls,
                "strikes": strikes,
                "count": f"{balls}-{strikes}",
                "is_first_pitch": balls == 0 and strikes == 0,
                "is_two_strike": strikes == 2,
                "is_ahead": strikes > balls,
                "is_behind": balls > strikes,
                "pitch_type": spec["pitch_type"],
                "pitch_display": spec["pitch_type"],
                "p_throws": spec.get("p_throws", "R"),
                "stand": spec.get("stand", "R"),
                "release_speed": spec.get("velo", 95.0),
                "release_spin_rate": 2200.0,
                "release_extension": 6.5,
                "release_pos_x": -1.9,
                "release_pos_z": 5.4,
                "pfx_x": -0.5,
                "pfx_z": 1.3,
                "plate_x": spec.get("plate_x", 0.0),
                "plate_z": spec.get("plate_z", 2.5),
                "sz_top": 3.5,
                "sz_bot": 1.5,
                "zone": spec.get("zone", 5),
                "description": spec.get("description", "called_strike"),
                "events": spec.get("events", None),
                "type": spec.get("type", "S"),
                "estimated_woba_using_speedangle": spec.get("xwoba", 0.300),
                "delta_run_exp": 0.0,
                "n_thruorder_pitcher": spec.get("tto", 1),
                "is_swing": spec.get("is_swing", False),
                "is_whiff": spec.get("is_whiff", False),
                "is_called_strike": spec.get("description", "called_strike")
                == "called_strike",
                "is_in_zone": spec.get("zone", 5) in range(1, 10),
                "is_chase": False,
                "is_contact": False,
            })
    return pd.DataFrame(rows)


class TestThresholdSeverity:
    def test_at_threshold_with_minimum_sample_scores_the_weight(self):
        assert flags._threshold_severity(1.0, 1.0, 2.0, 40, 40) == pytest.approx(2.0)

    def test_rises_with_effect_size(self):
        small = flags._threshold_severity(1.2, 1.0, 2.0, 40, 40)
        large = flags._threshold_severity(2.4, 1.0, 2.0, 40, 40)
        assert large > small

    def test_rises_with_sample_size(self):
        thin = flags._threshold_severity(1.5, 1.0, 2.0, 40, 40)
        thick = flags._threshold_severity(1.5, 1.0, 2.0, 360, 40)
        assert thick == pytest.approx(thin * 3.0)

    def test_a_trusted_moderate_finding_outranks_a_thin_dramatic_one(self):
        """The ordering bug this formula exists to fix: 2.2x over 27 pitches
        must not outrank 1.8x over 393."""
        thin_dramatic = flags._threshold_severity(2.2, 1.35, 2.0, 27, 25)
        broad_moderate = flags._threshold_severity(1.8, 1.35, 2.0, 393, 40)
        assert broad_moderate > thin_dramatic

    def test_never_negative_for_a_finding_that_fired(self):
        """A familiarity finding — velocity holding steady — used to score
        below every other finding precisely because it was the clean case."""
        assert flags._threshold_severity(0.022, 0.020, 2.0, 800, 150) > 0

    def test_guards_against_a_zero_threshold(self):
        assert math.isnan(flags._threshold_severity(1.0, 0.0, 2.0, 40, 40))

    def test_guards_against_an_empty_sample(self):
        assert math.isnan(flags._threshold_severity(1.0, 1.0, 2.0, 0, 40))


class TestHandednessGaps:
    def test_fires_on_a_pitch_withheld_from_one_side(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 180, "stand": "L"},
            {"pitch_type": "SL", "n": 20, "stand": "L"},
            {"pitch_type": "FF", "n": 100, "stand": "R"},
            {"pitch_type": "SL", "n": 100, "stand": "R"},
        ])
        found = flags.handedness_gaps(frame)
        assert found
        assert all(f["flag"] == "handedness_gap" for f in found)
        assert any("SL" in f["claim"] for f in found)

    def test_mirror_image_gaps_are_expected_and_collapsed(self):
        """Usage gaps across an arsenal sum to zero, so every gap has a mirror
        — with two pitches, "more fastballs to lefties" and "fewer sliders to
        lefties" are one fact stated twice. The rule reports both, and
        takeaways() keeps only the strongest, which is where the redundancy is
        supposed to be removed."""
        frame = make_frame([
            {"pitch_type": "FF", "n": 180, "stand": "L"},
            {"pitch_type": "SL", "n": 20, "stand": "L"},
            {"pitch_type": "FF", "n": 100, "stand": "R"},
            {"pitch_type": "SL", "n": 100, "stand": "R"},
        ])
        assert len(flags.handedness_gaps(frame)) == 2

        collapsed = flags.takeaways(frame)
        assert (collapsed["flag"] == "handedness_gap").sum() == 1

    def test_claim_carries_its_sample_size(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 180, "stand": "L"},
            {"pitch_type": "SL", "n": 20, "stand": "L"},
            {"pitch_type": "FF", "n": 100, "stand": "R"},
            {"pitch_type": "SL", "n": 100, "stand": "R"},
        ])
        assert "n=" in flags.handedness_gaps(frame)[0]["claim"]

    def test_silent_when_the_gap_is_small(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 100, "stand": "L"},
            {"pitch_type": "SL", "n": 100, "stand": "L"},
            {"pitch_type": "FF", "n": 105, "stand": "R"},
            {"pitch_type": "SL", "n": 95, "stand": "R"},
        ])
        assert flags.handedness_gaps(frame) == []

    def test_needs_no_league_data(self):
        """The comparison is the pitcher against himself, which is what lets
        this rule work before any baseline exists."""
        frame = make_frame([
            {"pitch_type": "FF", "n": 180, "stand": "L"},
            {"pitch_type": "SL", "n": 20, "stand": "L"},
            {"pitch_type": "FF", "n": 100, "stand": "R"},
            {"pitch_type": "SL", "n": 100, "stand": "R"},
        ])
        assert len(flags.evaluate(frame, None, None)) > 0


class TestOwnRateBaseline:
    """The count flag compares him to himself, not to the league.

    Against league rates a splitter specialist shows 5-7x in every count he
    uses the pitch in, producing one true fact — "he throws a splitter" —
    reported five times with the count changed, while a pitch he only elevates
    in two specific counts never crosses any threshold because his overall
    usage of it is close to league.
    """

    def test_fires_without_league_data(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 200},
            {"pitch_type": "SL", "n": 100},
            {"pitch_type": "SL", "n": 60, "balls": 2, "strikes": 1},
            {"pitch_type": "FF", "n": 20, "balls": 2, "strikes": 1},
        ])
        assert flags.predictable_counts(frame, None)

    def test_deduplicates_by_pitch_not_by_count(self):
        """A pitch elevated in several related counts is one tendency, and
        reporting it once per count crowds out every other pitch."""
        frame = make_frame([
            {"pitch_type": "FF", "n": 300},
            {"pitch_type": "SL", "n": 100},
            {"pitch_type": "SL", "n": 60, "balls": 0, "strikes": 2},
            {"pitch_type": "FF", "n": 20, "balls": 0, "strikes": 2},
            {"pitch_type": "SL", "n": 60, "balls": 1, "strikes": 2},
            {"pitch_type": "FF", "n": 20, "balls": 1, "strikes": 2},
        ])
        found = flags.predictable_counts(frame, None)
        pitches = [f["claim"].split(" the ")[1].split(" ")[0] for f in found]
        assert len(pitches) == len(set(pitches))

    def test_claim_states_his_own_rate(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 200},
            {"pitch_type": "SL", "n": 100},
            {"pitch_type": "SL", "n": 60, "balls": 2, "strikes": 1},
            {"pitch_type": "FF", "n": 20, "balls": 2, "strikes": 1},
        ])
        claim = flags.predictable_counts(frame, None)[0]["claim"]
        assert "his own" in claim
        assert "n=" in claim


class TestRunConsequence:
    """A tell and a weakness are different things.

    Yamamoto's splitter is predictable with two strikes and hitters still
    cannot touch it. Flagging that as exploitable hands a coach a plan that
    does not work, so every tell reports what the pattern costs him.
    """

    def test_a_costly_pattern_is_exploitable_when_resolved(self):
        exploitable, phrase = flags._consequence(
            3.4, 1.0, significant=True, runs_lo=1.2, runs_hi=5.6
        )
        assert exploitable
        assert "costs him" in phrase
        assert "95% CI" in phrase

    def test_a_pattern_that_works_for_him_is_not(self):
        exploitable, phrase = flags._consequence(
            -2.4, 1.0, significant=True, runs_lo=-4.1, runs_hi=-0.7
        )
        assert not exploitable
        assert "working" in phrase
        assert "rather than a weakness" in phrase

    def test_no_consequence_either_way(self):
        exploitable, phrase = flags._consequence(
            0.2, 1.0, significant=True, runs_lo=0.05, runs_hi=0.35
        )
        assert not exploitable
        assert "no run consequence" in phrase

    def test_an_unresolved_estimate_does_not_claim_a_cost(self):
        """The case that is almost always true. Run value per pitch has an SD
        near 0.19 against differences of 0.01-0.02, so one season of one
        pitcher cannot resolve them, and a report that prints "+3.4 runs"
        without the interval implies a precision that does not exist."""
        exploitable, phrase = flags._consequence(
            3.4, 1.0, significant=False, runs_lo=-3.9, runs_hi=10.6
        )
        assert exploitable
        assert "unresolved" in phrase
        assert "does not exclude zero" in phrase
        assert "costs him" not in phrase

    def test_unresolved_findings_are_not_demoted(self):
        """An unresolved run consequence is not evidence the pattern is
        harmless, so it must not halve the severity the way a demonstrated
        positive-for-him pattern does."""
        exploitable, _ = flags._consequence(
            0.0, 1.0, significant=False, runs_lo=-5.0, runs_hi=5.0
        )
        assert exploitable

    def test_missing_run_data_does_not_suppress_the_finding(self):
        """Run value is absent from some frames. That must not silently
        delete every tell."""
        exploitable, phrase = flags._consequence(float("nan"), 1.0)
        assert exploitable
        assert phrase == ""

    def test_findings_carry_a_runs_column(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 120, "plate_z": 3.3},
            {"pitch_type": "SL", "n": 30, "plate_z": 3.3},
            {"pitch_type": "SL", "n": 120, "plate_z": 1.7},
            {"pitch_type": "FF", "n": 30, "plate_z": 1.7},
        ])
        table = flags.evaluate(frame)
        assert "runs_cost" in table.columns


class TestBenjaminiHochberg:
    """The correction that decides whether a run estimate is reportable.

    A report runs well over a hundred comparisons before picking five things
    to say. At 120 comparisons, six clear an uncorrected p < .05 by chance
    alone, so "9 significant" is barely distinguishable from coin flips.
    """

    def test_nothing_survives_pure_noise(self):
        assert not any(flags.benjamini_hochberg([0.6] * 20))

    def test_a_clear_signal_survives(self):
        result = flags.benjamini_hochberg([1e-6] + [0.6] * 19)
        assert result[0]
        assert sum(result) == 1

    def test_is_less_conservative_than_bonferroni(self):
        """Several moderate p-values together are evidence in a way that none
        of them is alone, which is the point of controlling the false
        discovery rate rather than the family-wise error rate."""
        p_values = [0.001, 0.002, 0.003, 0.004] + [0.7] * 16
        survivors = sum(flags.benjamini_hochberg(p_values))
        bonferroni = sum(p <= 0.05 / len(p_values) for p in p_values)
        assert survivors >= bonferroni
        assert survivors >= 3

    def test_missing_p_values_never_survive(self):
        result = flags.benjamini_hochberg([float("nan"), 1e-9, None])
        assert result[0] is False
        assert result[2] is False
        assert result[1] is True

    def test_empty_input(self):
        assert flags.benjamini_hochberg([]) == []

    def test_all_missing(self):
        assert flags.benjamini_hochberg([float("nan")] * 3) == [False] * 3

    def test_respects_the_q_level(self):
        p_values = [0.02] + [0.7] * 9
        assert not any(flags.benjamini_hochberg(p_values, q=0.001))
        assert any(flags.benjamini_hochberg(p_values, q=0.5))


class TestComparisonCount:
    def test_reports_a_denominator_per_family(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 200, "plate_z": 3.3},
            {"pitch_type": "SL", "n": 200, "plate_z": 1.7},
        ])
        result = flags.comparison_count(frame)
        assert set(result["families"]) == {"count", "location", "sequencing"}
        assert result["total"] == sum(result["families"].values())

    def test_states_how_many_are_expected_by_chance(self):
        """Eight findings means something different out of 15 comparisons than
        out of 400, and a reader cannot tell without the denominator."""
        frame = make_frame([
            {"pitch_type": "FF", "n": 200, "plate_z": 3.3},
            {"pitch_type": "SL", "n": 200, "plate_z": 1.7},
        ])
        result = flags.comparison_count(frame)
        assert result["expected_by_chance"] == pytest.approx(
            0.05 * result["total"], abs=0.05
        )


class TestCorrectionIsAppliedInTheEngine:
    def test_findings_carry_a_resolved_flag(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 120, "plate_z": 3.3},
            {"pitch_type": "SL", "n": 30, "plate_z": 3.3},
            {"pitch_type": "SL", "n": 120, "plate_z": 1.7},
            {"pitch_type": "FF", "n": 30, "plate_z": 1.7},
        ])
        table = flags.evaluate(frame)
        assert "resolved" in table.columns
        assert table["resolved"].dtype == bool

    def test_unresolved_findings_say_so_in_the_claim(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 120, "plate_z": 3.3},
            {"pitch_type": "SL", "n": 30, "plate_z": 3.3},
            {"pitch_type": "SL", "n": 120, "plate_z": 1.7},
            {"pitch_type": "FF", "n": 30, "plate_z": 1.7},
        ])
        table = flags.evaluate(frame)
        unresolved = table[~table["resolved"] & table["runs_cost"].notna()]
        for claim in unresolved["claim"]:
            assert "unresolved" in claim


class TestLocationTellFlag:
    def test_fires_on_a_band_that_points_to_one_pitch(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 120, "plate_z": 3.3},
            {"pitch_type": "SL", "n": 30, "plate_z": 3.3},
            {"pitch_type": "SL", "n": 120, "plate_z": 1.7},
            {"pitch_type": "FF", "n": 30, "plate_z": 1.7},
        ])
        found = flags.location_tells(frame)
        assert any(f["flag"] == "location_tell" for f in found)

    def test_share_floor_rejects_a_real_lift_onto_a_minority(self):
        """A band lifting a pitch from 26% to 37% is a genuine 1.42x that no
        hitter can commit to, so it must not fire."""
        frame = make_frame([
            {"pitch_type": "FF", "n": 74, "plate_z": 1.7},
            {"pitch_type": "SL", "n": 42, "plate_z": 1.7},
            {"pitch_type": "CU", "n": 42, "plate_z": 1.7},
            {"pitch_type": "FC", "n": 42, "plate_z": 1.7},
            {"pitch_type": "FF", "n": 50, "plate_z": 3.3},
            {"pitch_type": "SL", "n": 50, "plate_z": 3.3},
            {"pitch_type": "CU", "n": 50, "plate_z": 3.3},
            {"pitch_type": "FC", "n": 50, "plate_z": 3.3},
        ])
        for finding in flags.location_tells(frame):
            assert finding["observed"] >= 0.45


class TestFirstPitch:
    def test_fires_on_a_predictable_opener(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 80, "balls": 0, "strikes": 0},
            {"pitch_type": "SL", "n": 20, "balls": 0, "strikes": 0},
        ])
        found = flags.first_pitch_tendency(frame)
        assert any("opens with" in f["claim"] for f in found)

    def test_fires_on_a_low_strike_rate(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 40, "balls": 0, "strikes": 0, "type": "S"},
            {"pitch_type": "SL", "n": 60, "balls": 0, "strikes": 0, "type": "B",
             "description": "ball"},
        ])
        found = flags.first_pitch_tendency(frame)
        assert any("strike rate" in f["claim"] for f in found)

    def test_silent_below_the_sample_gate(self):
        frame = make_frame([{"pitch_type": "FF", "n": 10, "balls": 0, "strikes": 0}])
        assert flags.first_pitch_tendency(frame) == []


class TestLeagueRulesAreOptional:
    def test_predictable_counts_skipped_without_league_data(self):
        frame = make_frame([{"pitch_type": "FF", "n": 100, "balls": 3, "strikes": 0}])
        assert flags.predictable_counts(frame, None) == []

    def test_hittable_pitches_skipped_without_league_data(self):
        frame = make_frame([{"pitch_type": "FF", "n": 100}])
        assert flags.hittable_pitches(frame, None) == []


class TestEvaluate:
    def test_returns_the_expected_columns(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 180, "stand": "L"},
            {"pitch_type": "SL", "n": 20, "stand": "L"},
            {"pitch_type": "FF", "n": 100, "stand": "R"},
            {"pitch_type": "SL", "n": 100, "stand": "R"},
        ])
        table = flags.evaluate(frame)
        assert list(table.columns) == flags.FINDING_COLUMNS

    def test_sorted_by_severity_descending(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 180, "stand": "L", "plate_z": 3.3},
            {"pitch_type": "SL", "n": 20, "stand": "L", "plate_z": 1.7},
            {"pitch_type": "FF", "n": 100, "stand": "R", "plate_z": 3.3},
            {"pitch_type": "SL", "n": 100, "stand": "R", "plate_z": 1.7},
        ])
        severities = list(flags.evaluate(frame)["severity"])
        assert severities == sorted(severities, reverse=True)

    def test_every_claim_states_its_sample_size(self):
        """The rule that makes the report safe to act on."""
        frame = make_frame([
            {"pitch_type": "FF", "n": 180, "stand": "L", "plate_z": 3.3},
            {"pitch_type": "SL", "n": 20, "stand": "L", "plate_z": 1.7},
            {"pitch_type": "FF", "n": 100, "stand": "R", "plate_z": 3.3},
            {"pitch_type": "SL", "n": 100, "stand": "R", "plate_z": 1.7},
        ])
        table = flags.evaluate(frame)
        assert len(table) > 0
        for claim in table["claim"]:
            assert "n=" in claim

    def test_every_finding_records_its_severity_basis(self):
        """A weighted judgment must never be mistakable for a z-score."""
        frame = make_frame([
            {"pitch_type": "FF", "n": 180, "stand": "L"},
            {"pitch_type": "SL", "n": 20, "stand": "L"},
            {"pitch_type": "FF", "n": 100, "stand": "R"},
            {"pitch_type": "SL", "n": 100, "stand": "R"},
        ])
        bases = set(flags.evaluate(frame)["severity_basis"])
        assert bases <= {flags.BASIS_Z, flags.BASIS_THRESHOLD}
        assert bases

    def test_empty_frame(self):
        frame = make_frame([{"pitch_type": "FF", "n": 1}]).iloc[0:0]
        table = flags.evaluate(frame)
        assert len(table) == 0
        assert list(table.columns) == flags.FINDING_COLUMNS

    def test_unremarkable_pitcher_produces_nothing(self):
        """A pitcher with an even mix, even location, and no platoon split has
        no weaknesses to report, and the engine must be willing to say so."""
        frame = make_frame([
            {"pitch_type": "FF", "n": 50, "stand": "L", "plate_z": 3.3},
            {"pitch_type": "SL", "n": 50, "stand": "L", "plate_z": 1.7},
            {"pitch_type": "FF", "n": 50, "stand": "R", "plate_z": 3.3},
            {"pitch_type": "SL", "n": 50, "stand": "R", "plate_z": 1.7},
            {"pitch_type": "FF", "n": 50, "stand": "L", "plate_z": 1.7},
            {"pitch_type": "SL", "n": 50, "stand": "L", "plate_z": 3.3},
            {"pitch_type": "FF", "n": 50, "stand": "R", "plate_z": 1.7},
            {"pitch_type": "SL", "n": 50, "stand": "R", "plate_z": 3.3},
        ])
        assert len(flags.evaluate(frame)) == 0


class TestTakeaways:
    def test_one_finding_per_flag_type(self):
        """Five variations of the same tell tell a coach one thing."""
        frame = make_frame([
            {"pitch_type": "FF", "n": 180, "stand": "L", "plate_z": 3.3},
            {"pitch_type": "SL", "n": 20, "stand": "L", "plate_z": 1.7},
            {"pitch_type": "FF", "n": 100, "stand": "R", "plate_z": 3.3},
            {"pitch_type": "SL", "n": 100, "stand": "R", "plate_z": 1.7},
        ])
        table = flags.takeaways(frame)
        assert len(table["flag"]) == len(set(table["flag"]))

    def test_keeps_the_highest_severity_finding_within_a_flag(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 180, "stand": "L", "plate_z": 3.3},
            {"pitch_type": "SL", "n": 20, "stand": "L", "plate_z": 1.7},
            {"pitch_type": "FF", "n": 100, "stand": "R", "plate_z": 3.3},
            {"pitch_type": "SL", "n": 100, "stand": "R", "plate_z": 1.7},
        ])
        full = flags.evaluate(frame)
        best = flags.takeaways(frame)
        for flag_name in best["flag"]:
            expected = full[full["flag"] == flag_name]["severity"].max()
            actual = best[best["flag"] == flag_name].iloc[0]["severity"]
            assert actual == pytest.approx(expected)

    def test_respects_the_limit(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 180, "stand": "L", "plate_z": 3.3},
            {"pitch_type": "SL", "n": 20, "stand": "L", "plate_z": 1.7},
            {"pitch_type": "FF", "n": 100, "stand": "R", "plate_z": 3.3},
            {"pitch_type": "SL", "n": 100, "stand": "R", "plate_z": 1.7},
        ])
        assert len(flags.takeaways(frame, limit=1)) <= 1

    def test_empty_frame(self):
        frame = make_frame([{"pitch_type": "FF", "n": 1}]).iloc[0:0]
        assert len(flags.takeaways(frame)) == 0
