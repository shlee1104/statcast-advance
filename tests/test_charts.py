"""Tests for src/charts.py and the chart headlines in src/gameplan.py.

Chart geometry is tested for the two things a hand-drawn chart gets wrong:
labels landing on each other, and a visual emphasis that disagrees with what
the chart's own title says.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import charts, gameplan
from tests.test_card import make_frame


class TestSpreadLabels:
    def test_keeps_the_minimum_gap(self):
        result = charts.spread_labels([100, 101, 102], min_gap=16, lo=0, hi=400)
        ordered = sorted(result)
        assert all(b - a >= 16 - 1e-9 for a, b in zip(ordered, ordered[1:]))

    def test_leaves_well_spaced_labels_alone(self):
        assert charts.spread_labels([50, 150, 250], 16, 0, 400) == [50, 150, 250]

    def test_stays_inside_the_bounds(self):
        result = charts.spread_labels([395, 398, 399], 16, 0, 400)
        assert max(result) <= 400
        assert min(result) >= 0

    def test_preserves_input_order(self):
        """Labels must stay attached to their own line."""
        result = charts.spread_labels([300, 100, 101], 16, 0, 400)
        assert result[0] == 300
        assert result[1] < result[2]

    def test_empty(self):
        assert charts.spread_labels([], 16, 0, 400) == []


class TestSlopeLayout:
    def rows(self):
        return [
            {"pitch_type": "FF", "usage_L": 0.37, "usage_R": 0.31},
            {"pitch_type": "SI", "usage_L": 0.006, "usage_R": 0.143},
            {"pitch_type": "SL", "usage_L": 0.004, "usage_R": 0.05},
            {"pitch_type": "FC", "usage_L": 0.107, "usage_R": 0.118},
        ]

    def test_two_near_zero_labels_do_not_collide(self):
        """Sinker 1% and slider 0% would print on top of each other."""
        layout = charts.slope_layout(self.rows())
        ys = sorted(p["label_y_L"] for p in layout["pitches"])
        assert all(b - a >= 16 - 1e-9 for a, b in zip(ys, ys[1:]))

    def test_axis_starts_at_zero_with_a_shared_floor(self):
        layout = charts.slope_layout(self.rows())
        assert layout["ticks"][0]["value"] == 0
        assert layout["y_max"] == pytest.approx(0.40)

    def test_a_pitch_one_side_never_sees_is_drawn_strong(self):
        """0% to 5% is a small gap and the whole story; it must not be the faded
        line when the chart's headline is about it."""
        layout = charts.slope_layout(self.rows())
        slider = next(p for p in layout["pitches"] if p["pitch"] == "SL")
        cutter = next(p for p in layout["pitches"] if p["pitch"] == "FC")
        assert slider["strong"]
        assert not cutter["strong"]

    def test_every_pitch_has_a_colour(self):
        layout = charts.slope_layout(self.rows())
        assert all(p["color"].startswith("#") for p in layout["pitches"])


class TestMovementLayout:
    def rows(self):
        """Skubal-like: changeup and sinker close together on the arm side."""
        return [
            {"pitch_type": "CH", "name": "Changeup", "arm_run": 15.0, "ivb": 6.4,
             "usage": 0.31, "whiff_rate": 0.47, "velo": 88.1},
            {"pitch_type": "FF", "name": "Four-Seam", "arm_run": 3.4, "ivb": 17.0,
             "usage": 0.30, "whiff_rate": 0.25, "velo": 97.7},
            {"pitch_type": "SI", "name": "Sinker", "arm_run": 13.1, "ivb": 13.6,
             "usage": 0.23, "whiff_rate": 0.17, "velo": 97.4},
        ]

    def label_box(self, b):
        return (b["label_x"] - 40, b["label_y"] - 12, b["label_x"] + 40, b["label_y"] + 16)

    def test_no_label_overlaps_another_bubble(self):
        layout = charts.movement_layout(self.rows())
        for b in layout["bubbles"]:
            box = self.label_box(b)
            for other in layout["bubbles"]:
                if other is b:
                    continue
                circle = (other["cx"] - other["r"], other["cy"] - other["r"],
                          other["cx"] + other["r"], other["cy"] + other["r"])
                assert not charts._overlaps(box, circle), (b["pitch"], other["pitch"])

    def test_bubble_area_tracks_usage(self):
        """Radius grows with the square root, so twice the usage is twice the
        area, not four times."""
        layout = charts.movement_layout(self.rows())
        ch = next(b for b in layout["bubbles"] if b["pitch"] == "CH")
        si = next(b for b in layout["bubbles"] if b["pitch"] == "SI")
        assert ch["r"] > si["r"]
        assert (ch["r"] - 5) / (si["r"] - 5) == pytest.approx((0.31 / 0.23) ** 0.5, rel=1e-2)

    def test_arm_side_plots_to_the_right(self):
        layout = charts.movement_layout(self.rows())
        ch = next(b for b in layout["bubbles"] if b["pitch"] == "CH")
        assert ch["cx"] > layout["zero_x"]

    def test_league_markers_only_for_pitches_he_throws(self):
        league = [{"pitch_type": "CH", "arm_run": 13.0, "ivb": 8.0},
                  {"pitch_type": "CU", "arm_run": -8.0, "ivb": -10.0}]
        layout = charts.movement_layout(self.rows(), league)
        assert [h["pitch"] for h in layout["league"]] == ["CH"]

    def test_whiff_colour_saturates_at_the_ends(self):
        assert charts.whiff_color(0.0) == charts.whiff_color(charts.WHIFF_LO)
        assert charts.whiff_color(0.9) == charts.whiff_color(charts.WHIFF_HI)


class TestChartHeadlines:
    def test_platoon_names_a_pitch_one_side_never_sees(self):
        # Small gaps on the shared pitches (75/70, 25/22.5), so the never-see
        # clause leads rather than a big gap.
        frame = make_frame([
            {"pitch_type": "FF", "n": 150, "stand": "L"},
            {"pitch_type": "CU", "n": 50, "stand": "L"},
            {"pitch_type": "FF", "n": 140, "stand": "R"},
            {"pitch_type": "CU", "n": 45, "stand": "R"},
            {"pitch_type": "SI", "n": 15, "stand": "R"},
        ])
        text = gameplan.platoon_headline(frame)
        assert text.startswith("Lefties almost never see the sinker")

    def test_platoon_leads_with_a_big_gap(self):
        frame = make_frame([
            {"pitch_type": "SI", "n": 90, "stand": "L"},
            {"pitch_type": "FF", "n": 110, "stand": "L"},
            {"pitch_type": "SI", "n": 30, "stand": "R"},
            {"pitch_type": "FF", "n": 170, "stand": "R"},
        ])
        text = gameplan.platoon_headline(frame)
        assert text.startswith("Lefties get the sinker 45%")

    def test_platoon_says_so_when_nothing_changes(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 100, "stand": s} for s in ("L", "R")
        ] + [
            {"pitch_type": "SL", "n": 100, "stand": s} for s in ("L", "R")
        ])
        assert "barely changes" in gameplan.platoon_headline(frame)

    def test_movement_compares_best_whiff_pitch_to_the_fastball(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 200, "velo": 95.0, "pfx_z": 1.33,
             "is_swing": True, "description": "foul"},
            {"pitch_type": "FS", "n": 100, "velo": 91.0, "pfx_z": 0.15,
             "is_swing": True, "is_whiff": True, "description": "swinging_strike"},
        ])
        text = gameplan.movement_headline(frame)
        assert "splitter" in text and "four-seam" in text
        assert "drops 14 inches more" in text
        assert "only 4 mph slower" in text

    def test_movement_makes_no_tunneling_claim(self):
        """Whether two pitches look alike out of the hand is not measured
        anywhere in this project, so the headline must not say it."""
        frame = make_frame([
            {"pitch_type": "FF", "n": 200, "velo": 95.0, "pfx_z": 1.33},
            {"pitch_type": "FS", "n": 100, "velo": 91.0, "pfx_z": 0.15,
             "is_swing": True, "is_whiff": True, "description": "swinging_strike"},
        ])
        text = gameplan.movement_headline(frame).lower()
        assert "tunnel" not in text and "out of the hand" not in text and "looks like" not in text


def transition(pitch: str, next_pitch: str, n: int, row_n: int, usual: float,
               standout: str = "", total: int = 400) -> dict:
    share = n / row_n
    return {"stand": "R", "pitch_type": pitch, "next_pitch": next_pitch, "n": n,
            "row_n": row_n, "next_n": round(usual * total), "total": total,
            "share": share, "usual": usual, "lift": share / usual,
            "standout": standout}


class TestSequenceLayout:
    PITCHES = ["FF", "SL", "CU"]

    def rows(self) -> list[dict]:
        """He doubles up on the slider and goes away from it after a four-seam."""
        return [
            transition("FF", "FF", 120, 200, 0.45),
            transition("FF", "SL", 40, 200, 0.40, "less"),
            transition("FF", "CU", 40, 200, 0.15),
            transition("SL", "FF", 45, 150, 0.45),
            transition("SL", "SL", 96, 150, 0.40, "more"),
            transition("SL", "CU", 9, 150, 0.15),
            transition("CU", "FF", 15, 50, 0.45),
            transition("CU", "SL", 24, 50, 0.40),
            transition("CU", "CU", 11, 50, 0.15),
        ]

    def layout(self, rows=None, gate: int = 40) -> dict:
        return charts.sequence_layout(self.PITCHES, self.rows() if rows is None else rows, gate)

    def test_the_same_pitch_sits_at_the_same_height_in_both_columns(self):
        """A flat line is the same pitch twice, so doubling up reads as flat."""
        layout = self.layout()
        assert [n["pitch"] for n in layout["left"]] == self.PITCHES
        assert [n["y"] for n in layout["left"]] == [n["y"] for n in layout["right"]]
        doubled = next(l for l in layout["lines"] if (l["from"], l["to"]) == ("SL", "SL"))
        assert doubled["y1"] == doubled["y2"]

    def test_time_runs_left_to_right(self):
        for line in self.layout()["lines"]:
            assert line["x1"] < line["x2"]

    def test_thickness_is_the_share_of_follow_ups(self):
        lines = {(l["from"], l["to"]): l for l in self.layout()["lines"]}
        assert lines[("FF", "FF")]["width"] > lines[("FF", "CU")]["width"]
        assert lines[("SL", "SL")]["width"] > lines[("FF", "SL")]["width"]

    def test_coloured_lines_are_drawn_last_and_labelled(self):
        """Drawn last so they sit on top of the grey ones; labelled with the
        share and his usual, which is the comparison the colour is about."""
        lines = self.layout()["lines"]
        kinds = [l["kind"] for l in lines]
        assert kinds == sorted(kinds)                      # "" before "less"/"more"
        coloured = [l for l in lines if l["kind"]]
        assert {l["label"] for l in coloured} == {"64% · usual 40%", "20% · usual 40%"}
        assert all("label" not in l for l in lines if not l["kind"])
        assert self.layout()["standouts"] == 2

    def test_more_and_less_are_different_colours(self):
        by_kind = {l["kind"]: l["stroke"] for l in self.layout()["lines"]}
        assert by_kind["more"] == charts.SEQUENCE_MORE
        assert by_kind["less"] == charts.SEQUENCE_LESS
        assert by_kind[""] == charts.SEQUENCE_USUAL

    def test_a_rare_follow_up_within_his_usual_is_not_drawn(self):
        """Nine curveballs in 150 follow-ups is 6%: a hairline that says
        nothing and sits on top of the lines that do."""
        pairs = {(l["from"], l["to"]) for l in self.layout()["lines"]}
        assert ("SL", "CU") not in pairs
        assert ("FF", "CU") in pairs

    def test_a_rare_follow_up_is_drawn_when_it_is_the_finding(self):
        rows = self.rows()
        rows[5] = transition("SL", "CU", 3, 150, 0.15, "less")
        line = next(l for l in self.layout(rows)["lines"] if (l["from"], l["to"]) == ("SL", "CU"))
        assert line["kind"] == "less"
        assert line["width"] >= 2

    def test_a_pitch_with_too_few_follow_ups_sends_out_no_lines(self):
        layout = self.layout(gate=60)
        curve = next(n for n in layout["left"] if n["pitch"] == "CU")
        assert curve["row_n"] == 50 and not curve["read"]
        assert all(l["from"] != "CU" for l in layout["lines"])
        # It can still be what comes next.
        assert any(l["to"] == "CU" for l in layout["lines"])

    def test_labels_do_not_land_on_each_other(self):
        """A two-pitch reliever with all four lines coloured is the tightest
        case the graph has to draw."""
        rows = [
            transition("FF", "FF", 134, 200, 0.53, "more"),
            transition("FF", "SL", 66, 200, 0.47, "less"),
            transition("SL", "FF", 60, 180, 0.53, "less"),
            transition("SL", "SL", 120, 180, 0.47, "more"),
        ]
        layout = charts.sequence_layout(["FF", "SL"], rows, 40)
        boxes = layout["label_boxes"]
        assert len(boxes) == 4
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                assert not charts._overlaps(a, b)

    def test_few_pitches_get_taller_rows(self):
        two = charts.sequence_layout(["FF", "SL"], [], 40)
        six = charts.sequence_layout(["FF", "SL", "CU", "FS", "FC", "SI"], [], 40)
        assert two["left"][1]["y"] - two["left"][0]["y"] > six["left"][1]["y"] - six["left"][0]["y"]

    def test_no_rows_still_gives_a_canvas(self):
        layout = charts.sequence_layout(self.PITCHES, [], 40)
        assert layout["lines"] == [] and layout["standouts"] == 0
        assert layout["height"] > 0


class TestSequenceHeadline:
    def table(self, *rows) -> pd.DataFrame:
        frame = pd.DataFrame(list(rows))
        frame["lift_z"] = [r.get("lift_z", 4.0) for r in rows]
        return frame

    def test_says_so_when_nothing_stands_out(self):
        table = self.table(transition("FF", "SL", 80, 200, 0.40))
        text = gameplan.sequence_headline(table)
        assert text.startswith("No pitch-to-pitch sequence stands out")
        assert gameplan.sequence_headline(table.iloc[0:0]) == text

    def test_doubling_up(self):
        text = gameplan.sequence_headline(self.table(
            transition("SL", "SL", 96, 150, 0.40, "more")))
        assert text == ("To righties he doubles up on the slider: another one follows "
                        "64% of the time, against his usual 40% (96 of 150).")

    def test_one_pitch_setting_up_another(self):
        text = gameplan.sequence_headline(self.table(
            transition("CU", "SI", 42, 104, 0.20, "more")))
        assert text == ("To righties, the curveball sets up the sinker: it follows "
                        "40% of the time, against his usual 20% (42 of 104).")

    def test_rarely_the_same_pitch_twice(self):
        text = gameplan.sequence_headline(self.table(
            {**transition("FC", "FC", 17, 202, 0.16, "less"), "stand": "L"}))
        assert text.startswith("To lefties he rarely throws the cutter twice in a row: 8%")
        assert "(17 of 202)" in text

    def test_going_away_from_a_pitch(self):
        text = gameplan.sequence_headline(self.table(
            transition("SI", "FF", 59, 232, 0.37, "less")))
        assert text.startswith("To righties he goes away from the four-seam after the sinker: 25%")

    def test_leads_with_the_departure_furthest_from_chance(self):
        text = gameplan.sequence_headline(self.table(
            {**transition("SL", "SL", 96, 150, 0.40, "more"), "lift_z": 3.5},
            {**transition("FF", "SL", 40, 200, 0.40, "less"), "lift_z": -6.0},
        ))
        assert "goes away from the slider after the four-seam" in text

    def test_carries_its_sample_size(self):
        text = gameplan.sequence_headline(self.table(
            transition("SL", "SL", 96, 150, 0.40, "more")))
        assert "96 of 150" in text



class TestPieLayout:
    def rows(self) -> list[dict]:
        return [
            {"pitch": "FF", "share": 0.55, "n": 55},
            {"pitch": "SL", "share": 0.33, "n": 33},
            {"pitch": "CU", "share": 0.12, "n": 12},
        ]

    def test_goes_round_clockwise_from_twelve_in_the_order_given(self):
        """The card gives every pie the same order, so a pitch starts from
        the same place in each and the pies compare by eye."""
        pie = charts.pie_layout(self.rows())
        assert [s["pitch"] for s in pie["slices"]] == ["FF", "SL", "CU"]
        assert pie["slices"][0]["d"].startswith("M60.0,60.0 L60.0,4.0 ")
        assert pie["slices"][-1]["d"].endswith(" 60.0,4.0 Z")

    def test_a_majority_slice_takes_the_long_way_round(self):
        pie = charts.pie_layout(self.rows())
        assert " 0 1 1 " in pie["slices"][0]["d"]
        assert " 0 0 1 " in pie["slices"][1]["d"]

    def test_a_pitch_not_thrown_gets_no_slice(self):
        pie = charts.pie_layout(self.rows() + [{"pitch": "CH", "share": 0.0}])
        assert "CH" not in [s["pitch"] for s in pie["slices"]]

    def test_one_pitch_is_a_full_circle(self):
        """An arc from a point back to itself draws nothing."""
        pie = charts.pie_layout([{"pitch": "FF", "share": 1.0}])
        only = pie["slices"][0]
        assert only["full"] and only["d"] is None
        assert (only["label_x"], only["label_y"]) == (60.0, 60.0)

    def test_a_thin_slice_is_drawn_but_not_labelled(self):
        pie = charts.pie_layout([
            {"pitch": "FF", "share": 0.93}, {"pitch": "CU", "share": 0.07}])
        thin = pie["slices"][1]
        assert thin["d"] and thin["labelled"] is False

    def test_a_small_slice_is_labelled_nearer_the_rim(self):
        """Two neighbouring 13% slices printed their labels on each other."""
        pie = charts.pie_layout(self.rows())
        big, _, small = pie["slices"]
        def radius(s):
            return ((s["label_x"] - 60) ** 2 + (s["label_y"] - 60) ** 2) ** 0.5
        assert small["small"] and not big["small"]
        assert radius(small) > radius(big)

    def test_each_slice_keeps_its_pitch_colour_and_its_count(self):
        pie = charts.pie_layout(self.rows())
        assert pie["slices"][0]["color"] == charts.color("FF")
        assert pie["slices"][0]["n"] == 55

    def test_no_pitches_is_an_empty_pie(self):
        assert charts.pie_layout([])["slices"] == []


class TestZoneCell:
    def test_tint_runs_from_white_to_the_colour(self):
        assert charts.tint("#2f6fb2", 0) == "rgb(255,255,255)"
        assert charts.tint("#2f6fb2", 1) == "rgb(47,111,178)"
        assert charts.tint("#2f6fb2", 5) == "rgb(47,111,178)"

    def test_a_cell_is_its_pitchs_colour(self):
        """So a map is the colour its pitch has in the pies beside it."""
        full = charts.zone_cell("FS", charts.ZONE_FULL_SHARE)
        assert full["bg"] == charts.tint(charts.color("FS"), 1)

    def test_the_same_share_is_the_same_depth_in_every_map(self):
        """The scale is fixed, not fitted to each map."""
        assert charts.zone_cell("FF", 0.15)["bg"] == charts.tint(charts.color("FF"), 0.5)
        assert charts.zone_cell("FF", 0.90)["bg"] == charts.zone_cell("FF", 0.30)["bg"]

    def test_a_deep_cell_prints_its_number_in_white(self):
        assert charts.zone_cell("FF", 0.25)["dark"] is True
        assert charts.zone_cell("FF", 0.05)["dark"] is False
