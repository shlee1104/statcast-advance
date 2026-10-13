"""Tests for src/metrics/tunneling.py.

The physics has one check that matters more than any synthetic case: run to
the plate, the reconstructed trajectory has to land where Savant says the
pitch crossed it. Those two numbers are published independently of the fit
terms, so agreeing with them is evidence the columns are being read the right
way round — which plane the release point is on, which root of the quadratic
runs forward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import cli, store
from src.metrics import sequencing, tunneling

PLATE = tunneling.PLATE_FRONT_Y_FT


def pitch(
    x0: float = -1.5, z0: float = 5.5,
    vx0: float = 4.0, vy0: float = -135.0, vz0: float = -4.0,
    ax: float = -8.0, ay: float = 28.0, az: float = -16.0,
    release_y: float = 54.5,
) -> dict:
    """A pitch described on the 50-foot plane, as Savant's columns report it.

    Works out the release point and plate location the same pitch would be
    published with, by running the constant-acceleration path back to
    `release_y` and forward to the front of the plate.
    """
    def when(y: float) -> float:
        return (-vy0 - np.sqrt(vy0 ** 2 - 2 * ay * (50.0 - y))) / ay

    def where(t: float) -> tuple[float, float]:
        return x0 + vx0 * t + 0.5 * ax * t ** 2, z0 + vz0 * t + 0.5 * az * t ** 2

    t_release, t_plate = when(release_y), when(PLATE)
    release_x, release_z = where(t_release)
    plate_x, plate_z = where(t_plate)
    return {
        "release_pos_x": release_x, "release_pos_y": release_y, "release_pos_z": release_z,
        "vx0": vx0, "vy0": vy0, "vz0": vz0, "ax": ax, "ay": ay, "az": az,
        "plate_x": plate_x, "plate_z": plate_z,
        "_x0": x0, "_z0": z0, "_t_release": t_release, "_t_plate": t_plate,
    }


def one(**kwargs) -> pd.DataFrame:
    return pd.DataFrame([pitch(**kwargs)])


def appearances(*groups: tuple[int, list[dict]], stand: str = "R") -> pd.DataFrame:
    """Plate appearances: (how many, [pitch, pitch, ...]) with each pitch a
    dict of `pitch_type` and any `pitch()` arguments."""
    rows, at_bat = [], 0
    for n, sequence in groups:
        for _ in range(n):
            at_bat += 1
            for number, spec in enumerate(sequence, start=1):
                spec = dict(spec)
                kind = spec.pop("pitch_type")
                rows.append({"game_pk": 1, "at_bat_number": at_bat, "pitch_number": number,
                             "stand": spec.pop("stand", stand), "pitch_type": kind,
                             **pitch(**spec)})
    return pd.DataFrame(rows)


FASTBALL = {"pitch_type": "FF"}
# Same release, same direction out of the hand, more drop: a splitter.
SPLITTER = {"pitch_type": "FS", "vy0": -128.0, "az": -30.0}


class TestTrajectory:
    def test_runs_forward_to_the_published_plate_location(self):
        frame = one()
        at_plate = tunneling.position_at(frame, PLATE)
        assert at_plate["x"].iloc[0] == pytest.approx(frame["plate_x"].iloc[0], abs=1e-9)
        assert at_plate["z"].iloc[0] == pytest.approx(frame["plate_z"].iloc[0], abs=1e-9)

    def test_the_release_point_is_behind_the_reference_plane(self):
        """It is at y = release_pos_y, about 54 feet, so its time is negative.
        Reading the release columns as positions on the 50-foot plane misses
        the plate by a median of two inches on real pitches."""
        frame = one()
        assert tunneling.time_at(frame, frame["release_pos_y"]).iloc[0] < 0
        at_release = tunneling.position_at(frame, 54.5)
        assert at_release["x"].iloc[0] == pytest.approx(frame["release_pos_x"].iloc[0])
        assert at_release["z"].iloc[0] == pytest.approx(frame["release_pos_z"].iloc[0])
        assert at_release["t"].iloc[0] == pytest.approx(0.0, abs=1e-12)

    def test_recovers_the_position_on_the_reference_plane(self):
        frame = one(x0=-1.2, z0=5.1)
        at_fifty = tunneling.position_at(frame, 50.0)
        assert at_fifty["x"].iloc[0] == pytest.approx(-1.2)
        assert at_fifty["z"].iloc[0] == pytest.approx(5.1)

    def test_time_runs_forward_toward_the_plate(self):
        frame = one()
        times = [tunneling.position_at(frame, y)["t"].iloc[0] for y in (50.0, 23.8, PLATE)]
        assert times == sorted(times)
        assert 0.35 < times[-1] < 0.50     # a 92 mph pitch is in the air about 0.4 s

    def test_a_slower_pitch_reaches_the_decision_point_with_more_time_left(self):
        """The decision point is a distance and a hitter's deadline is a time."""
        fast = tunneling.flight(one(vy0=-140.0))["decision_time"].iloc[0]
        slow = tunneling.flight(one(vy0=-112.0))["decision_time"].iloc[0]
        assert slow > fast

    def test_a_pitch_missing_part_of_its_fit_gets_nulls(self):
        frame = one()
        frame.loc[0, "ay"] = np.nan
        assert tunneling.position_at(frame, PLATE).isna().all(axis=None)

    @pytest.mark.parametrize("stem", ["yamamoto", "skubal"])
    def test_lands_on_savants_plate_location_for_every_real_pitch(self, stem):
        """The self-check. plate_x and plate_z are published to a hundredth of
        a foot, so the fit can only agree to about that: 0.07 inch."""
        frame = cli.prepare(cli.load_from_fixture(stem, 2025))
        fit = tunneling.flight(frame)
        miss = np.hypot(fit["plate_x_fit"] - frame["plate_x"],
                        fit["plate_z_fit"] - frame["plate_z"]) * 12
        assert len(miss) > 3000 and miss.notna().all()
        assert miss.max() < 0.1


class TestFlight:
    def test_late_movement_is_what_the_direction_at_the_decision_point_hid(self):
        """Half the acceleration times the time left, squared."""
        frame = one(ax=-10.0, az=-20.0)
        row = tunneling.flight(frame).iloc[0]
        left = row["decision_time"]
        assert row["late_x"] == pytest.approx(0.5 * -10.0 * left ** 2)
        assert row["late_z"] == pytest.approx(0.5 * -20.0 * left ** 2)

    def test_a_pitch_on_a_straight_line_has_no_late_movement(self):
        row = tunneling.flight(one(ax=0.0, az=0.0)).iloc[0]
        assert row["late_x"] == 0.0 and row["late_z"] == 0.0

    def test_where_it_finishes_is_where_it_was_headed_plus_the_late_movement(self):
        frame = one()
        fit = tunneling.flight(frame).iloc[0]
        # Its velocity at the decision point, from two nearby positions.
        here = tunneling.position_at(frame, 23.8).iloc[0]
        near = tunneling.position_at(frame, 23.8 - 1e-4).iloc[0]
        step = near["t"] - here["t"]
        headed_x = here["x"] + (near["x"] - here["x"]) / step * fit["decision_time"]
        assert fit["plate_x_fit"] == pytest.approx(headed_x + fit["late_x"], abs=1e-4)

    def test_decision_point_is_configurable(self):
        frame = one()
        near = tunneling.flight(frame, tunnel_ft=15.0)["decision_time"].iloc[0]
        far = tunneling.flight(frame, tunnel_ft=30.0)["decision_time"].iloc[0]
        assert far > near

    def test_a_frame_without_the_fit_gives_an_empty_flight(self):
        frame = pd.DataFrame({"pitch_type": ["FF"], "plate_x": [0.0]})
        assert tunneling.flight(frame).isna().all(axis=None)
        assert not tunneling.has_trajectory(frame)

    def test_columns_that_are_all_null_are_not_a_trajectory(self):
        """What a cache row written before the columns existed looks like."""
        frame = one()
        frame[tunneling.TRAJECTORY_COLUMNS] = np.nan
        assert not tunneling.has_trajectory(frame)


class TestPairs:
    def test_a_pitch_is_compared_with_the_one_that_followed_it(self):
        frame = appearances((20, [FASTBALL, SPLITTER]))
        pairs = tunneling.pairs(frame)
        assert len(pairs) == 20
        assert set(pairs["pitch_type"]) == {"FF"} and set(pairs["next_pitch"]) == {"FS"}

    def test_same_release_and_direction_then_more_drop(self):
        """The splitter leaves on the fastball's line and falls off it."""
        row = tunneling.pairs(appearances((20, [FASTBALL, SPLITTER]))).iloc[0]
        assert row["release_diff"] < 1.0
        assert row["tunnel_diff"] < row["plate_diff"]
        assert row["late_diff"] > 2.0
        assert row["flight_diff_ms"] > 0          # the splitter is slower

    def test_the_same_pitch_twice_is_no_distance_apart(self):
        row = tunneling.pairs(appearances((20, [FASTBALL, FASTBALL]))).iloc[0]
        for column in ("release_diff", "tunnel_diff", "plate_diff", "late_diff",
                       "flight_diff_ms"):
            assert row[column] == pytest.approx(0.0, abs=1e-9)

    def test_separations_are_in_inches(self):
        """Two pitches identical but for starting a foot apart stay a foot apart."""
        other = {"pitch_type": "FF", "x0": -0.5}
        row = tunneling.pairs(appearances((20, [FASTBALL, other]))).iloc[0]
        assert row["tunnel_diff"] == pytest.approx(12.0)
        assert row["plate_diff"] == pytest.approx(12.0)

    def test_pairs_do_not_cross_plate_appearances(self):
        frame = appearances((20, [FASTBALL]), (20, [SPLITTER]))
        assert len(tunneling.pairs(frame)) == 0

    def test_a_stray_pitch_does_not_join_its_neighbours(self):
        """Dropping a mislabelled pitch before pairing would pair the pitches
        either side of it, which were never consecutive."""
        stray = {"pitch_type": "KN"}
        frame = appearances((200, [FASTBALL, SPLITTER]), (1, [FASTBALL, stray, SPLITTER]))
        pairs = tunneling.pairs(frame)
        assert len(pairs) == 200
        assert "KN" not in set(pairs["pitch_type"]) | set(pairs["next_pitch"])

    def test_no_trajectory_no_pairs(self):
        frame = appearances((20, [FASTBALL, SPLITTER]))
        frame[tunneling.TRAJECTORY_COLUMNS] = np.nan
        assert len(tunneling.pairs(frame)) == 0
        assert list(tunneling.pairs(frame).columns) == tunneling.PAIR_COLUMNS

    def test_pairing_carries_the_follow_ups_flight(self):
        frame = appearances((5, [FASTBALL, SPLITTER]))
        paired = sequencing._pair_pitches(frame, extra=["vy0"])
        assert set(paired["next_vy0"]) == {-128.0}


class TestPairTable:
    def scattered(self) -> pd.DataFrame:
        """Fastballs thrown to two spots, so two in a row are sometimes apart;
        and a splitter that follows the fastball's line."""
        up = {"pitch_type": "FF", "z0": 5.9}
        return appearances(
            (20, [FASTBALL, FASTBALL]), (20, [FASTBALL, up]), (20, [up, FASTBALL]),
            (40, [FASTBALL, SPLITTER]),
        )

    def test_one_row_per_pair_thrown_often_enough(self):
        table = tunneling.pair_table(self.scattered(), min_n=30)
        assert [(r.pitch_type, r.next_pitch, r.n) for r in table.itertuples()] == [
            ("FF", "FF", 60), ("FF", "FS", 40)]

    def test_the_floor_is_the_same_pitch_thrown_twice(self):
        """What "the same pitch" looks like at the decision point, because he
        does not throw to one spot."""
        table = tunneling.pair_table(self.scattered(), min_n=30)
        same = table[table["next_pitch"] == "FF"].iloc[0]
        assert same["floor"] == pytest.approx(same["tunnel"])
        assert same["floor"] > 0 and same["floor_n"] == 60
        assert same["tunnel_vs_floor"] == pytest.approx(0.0)

    def test_a_pair_is_read_against_the_floor(self):
        """A cutter that flies exactly as the fastball before it is closer to
        that fastball than another fastball usually is; a curveball that
        starts a foot away is further."""
        cutter = {"pitch_type": "FC"}
        curve = {"pitch_type": "CU", "x0": -0.5}
        frame = pd.concat([
            self.scattered(),
            appearances((40, [FASTBALL, cutter])).assign(game_pk=2),
            appearances((40, [FASTBALL, curve])).assign(game_pk=3),
        ], ignore_index=True)
        table = tunneling.pair_table(frame, min_n=30).set_index("next_pitch")

        for row in table.itertuples():
            assert row.tunnel_vs_floor == pytest.approx(row.tunnel - row.floor)
        assert table.loc["FC", "tunnel_vs_floor"] < 0 and table.loc["FC", "tunnel_z"] < 0
        assert table.loc["CU", "tunnel_vs_floor"] > 0 and table.loc["CU", "tunnel_z"] > 0

    def test_each_side_has_its_own_floor(self):
        left = self.scattered().assign(stand="L")
        right = appearances((60, [FASTBALL, {"pitch_type": "FF", "x0": -0.5}]))
        right["at_bat_number"] += 1000
        table = tunneling.pair_table(pd.concat([left, right], ignore_index=True), min_n=30)
        floors = table.groupby("stand")["floor"].first()
        assert floors["R"] == pytest.approx(12.0)
        assert floors["L"] < floors["R"]

    def test_the_offset_is_the_average_difference_with_the_scatter_gone(self):
        """A splitter that leaves three inches below whichever fastball came
        before it. The fastballs go to two spots, so the mean distance mixes
        that scatter in; the offset is the three inches."""
        low = {"pitch_type": "FS", "z0": 5.25}
        low_after_up = {"pitch_type": "FS", "z0": 5.65}
        up = {"pitch_type": "FF", "z0": 5.9}
        frame = pd.concat([
            self.scattered(),
            appearances((20, [FASTBALL, low])).assign(game_pk=2),
            appearances((20, [up, low_after_up])).assign(game_pk=3),
        ], ignore_index=True)
        frame = frame[~((frame["game_pk"] == 1) & (frame["pitch_type"] == "FS"))]
        row = tunneling.pair_table(frame, min_n=30).set_index("next_pitch").loc["FS"]
        assert row["tunnel_offset"] == pytest.approx(3.0, abs=0.05)
        assert row["tunnel_dz"] == pytest.approx(-3.0, abs=0.05)
        assert abs(row["tunnel_dx"]) < 0.05

    def test_two_of_the_same_pitch_have_no_offset_but_do_have_a_floor(self):
        """Up then down as often as down then up: far apart pitch to pitch,
        and on average nowhere."""
        table = tunneling.pair_table(self.scattered(), min_n=30)
        same = table[table["next_pitch"] == "FF"].iloc[0]
        assert same["tunnel"] > 2.0
        assert same["tunnel_offset"] == pytest.approx(0.0, abs=1e-9)
        assert same["tunnel_d"] == pytest.approx(0.0, abs=1e-9)

    def test_discriminability_is_the_offset_over_his_scatter(self):
        table = tunneling.pair_table(self.scattered(), min_n=30)
        split = table[table["next_pitch"] == "FS"].iloc[0]
        assert split["spread"] > 0
        assert split["tunnel_d"] == pytest.approx(split["tunnel_offset"] / split["spread"])

    def test_across_is_signed_toward_his_arm_side(self):
        """Savant's x runs to the catcher's right. A right-hander's arm side
        is the other way; a left-hander's is that way."""
        glove = {"pitch_type": "FC", "x0": -1.0}        # half a foot to the catcher's right
        frame = appearances((40, [FASTBALL, glove]))
        righty = tunneling.pairs(frame.assign(p_throws="R"))
        lefty = tunneling.pairs(frame.assign(p_throws="L"))
        assert righty["tunnel_dx"].iloc[0] == pytest.approx(-6.0)
        assert lefty["tunnel_dx"].iloc[0] == pytest.approx(6.0)

    def test_empty_without_a_trajectory(self):
        frame = self.scattered()
        frame[tunneling.TRAJECTORY_COLUMNS] = np.nan
        table = tunneling.pair_table(frame)
        assert len(table) == 0 and list(table.columns) == tunneling.TABLE_COLUMNS

    def test_runs_on_a_real_season(self):
        frame = cli.prepare(cli.load_from_fixture("yamamoto", 2025))
        table = tunneling.pair_table(frame)
        assert len(table) > 10 and (table["n"] >= tunneling.min_pairs()).all()
        # Two of the same pitch are several inches apart at the decision point.
        assert 5 < table["floor"].min() < table["floor"].max() < 12


class TestReading:
    def test_three_way_and_silent_in_the_middle(self):
        """Three words would claim a precision the number does not have."""
        assert tunneling.reading(0.4) == tunneling.SAME_WINDOW
        assert tunneling.reading(0.75) == tunneling.SAME_WINDOW
        assert tunneling.reading(1.0) == ""
        assert tunneling.reading(1.25) == tunneling.APART
        assert tunneling.reading(2.6) == tunneling.APART

    def test_no_number_no_word(self):
        assert tunneling.reading(float("nan")) == ""

    def test_the_table_carries_the_word(self):
        cutter = {"pitch_type": "FC"}                    # flies as the fastball does
        curve = {"pitch_type": "CU", "x0": -0.5}         # a foot away throughout
        up = {"pitch_type": "FF", "z0": 5.9}
        frame = pd.concat([
            appearances((30, [FASTBALL, up]), (30, [up, FASTBALL])),
            appearances((40, [FASTBALL, cutter])).assign(game_pk=2),
            appearances((40, [FASTBALL, curve])).assign(game_pk=3),
        ], ignore_index=True)
        table = tunneling.pair_table(frame, min_n=30).set_index("next_pitch")
        assert table.loc["FC", "reading"] == tunneling.SAME_WINDOW
        assert table.loc["CU", "reading"] == tunneling.APART


class TestCacheCarriesTheFit:
    def test_the_schema_stores_every_column_the_reconstruction_reads(self):
        assert set(tunneling.TRAJECTORY_COLUMNS) <= set(store.PITCH_SCHEMA)

    def test_an_older_cache_gains_the_columns_as_nulls(self, tmp_path, monkeypatch):
        """A cache built before the columns existed is widened in place, and
        reads back with nulls there: no trajectory, so no pairs, not an error."""
        path = tmp_path / "cache.duckdb"
        old = {k: v for k, v in store.PITCH_SCHEMA.items()
               if k not in ("release_pos_y", "vx0", "vy0", "vz0", "ax", "ay", "az")}
        monkeypatch.setattr(store, "PITCH_SCHEMA", old)
        store.connect(path).close()
        monkeypatch.undo()

        conn = store.connect(path)
        try:
            columns = {row[0] for row in conn.execute("DESCRIBE pitches").fetchall()}
        finally:
            conn.close()
        assert set(tunneling.TRAJECTORY_COLUMNS) <= columns
