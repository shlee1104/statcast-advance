"""Tests for src/report.py.

The failure that matters here is silent. A NaN reaching the embedded JSON
produces a file that looks fine on disk, passes every Python assertion, and
renders a blank page in the browser — because bare NaN is a syntax error in
JSON and takes the whole script block down with it. Several tests exist only
to make that impossible.
"""

from __future__ import annotations

import json
import math
import re

import pandas as pd
import pytest

from src import report


def make_frame(specs: list[dict]) -> pd.DataFrame:
    """A minimal frame carrying every column the report touches."""
    rows = []
    at_bat = 0
    for spec in specs:
        for _ in range(spec["n"]):
            at_bat += 1
            balls = spec.get("balls", 0)
            strikes = spec.get("strikes", 0)
            rows.append({
                "game_pk": spec.get("game_pk", 1),
                "at_bat_number": at_bat,
                "pitch_number": 1,
                "balls": balls,
                "strikes": strikes,
                "count": f"{balls}-{strikes}",
                "is_first_pitch": balls == 0 and strikes == 0,
                "is_two_strike": strikes == 2,
                "pitch_type": spec["pitch_type"],
                "pitch_display": spec.get("display", spec["pitch_type"]),
                "p_throws": "R",
                "stand": spec.get("stand", "R"),
                "release_speed": spec.get("velo", 95.0),
                "release_spin_rate": 2200.0,
                "release_extension": 6.5,
                "release_pos_x": -1.9,
                "release_pos_z": 5.4,
                "arm_angle": 42.0,
                "pfx_x": -0.5,
                "pfx_z": 1.3,
                "plate_x": spec.get("plate_x", 0.0),
                "plate_z": spec.get("plate_z", 2.5),
                "sz_top": 3.5,
                "sz_bot": 1.5,
                "zone": spec.get("zone", 5),
                "description": spec.get("description", "called_strike"),
                "events": None,
                "type": "S",
                "estimated_woba_using_speedangle": spec.get("xwoba", 0.300),
                "delta_run_exp": 0.0,
                "n_thruorder_pitcher": spec.get("tto", 1),
                "is_swing": spec.get("is_swing", False),
                "is_whiff": spec.get("is_whiff", False),
                "is_called_strike": True,
                "is_in_zone": spec.get("zone", 5) in range(1, 10),
                "is_chase": False,
                "is_contact": False,
            })
    return pd.DataFrame(rows)


def sample_frame() -> pd.DataFrame:
    return make_frame([
        {"pitch_type": "FF", "n": 120, "plate_z": 3.3, "stand": "L"},
        {"pitch_type": "SL", "n": 80, "plate_z": 1.7, "stand": "L", "zone": 13},
        {"pitch_type": "FF", "n": 100, "plate_z": 3.3, "stand": "R"},
        {"pitch_type": "SL", "n": 100, "plate_z": 1.7, "stand": "R", "zone": 13},
    ])


class TestCleanValue:
    def test_nan_becomes_none(self):
        assert report._clean(float("nan")) is None

    def test_infinity_becomes_none(self):
        assert report._clean(float("inf")) is None
        assert report._clean(float("-inf")) is None

    def test_ordinary_numbers_survive(self):
        assert report._clean(0.5) == 0.5
        assert report._clean(7) == 7

    def test_strings_and_none_survive(self):
        assert report._clean("FF") == "FF"
        assert report._clean(None) is None


class TestBuildPayload:
    def test_rejects_an_empty_frame(self):
        with pytest.raises(ValueError):
            report.build_payload(make_frame([{"pitch_type": "FF", "n": 1}]).iloc[0:0],
                                 "Nobody", 2025)

    def test_meta_describes_the_sample(self):
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        assert payload["meta"]["pitcher"] == "Test Pitcher"
        assert payload["meta"]["season"] == 2025
        assert payload["meta"]["pitches"] == 400
        assert payload["meta"]["hand"] == "R"

    def test_works_without_league_data(self):
        """A report generated before build_baselines.py has ever run must be
        thinner, not broken."""
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        assert payload["meta"]["has_league"] is False
        assert payload["arsenal"]
        assert payload["count_grid"]["pitches"]

    def test_payload_is_json_serializable_with_no_nan(self):
        """The one that stops a blank page in the browser."""
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        blob = json.dumps(payload, allow_nan=False)
        assert not re.search(r'(?<![\w"])(NaN|Infinity)(?![\w"])', blob)

    def test_every_line_of_the_plan_carries_its_sample_size(self):
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        plan = payload["card"]["plan"]
        lines = plan["L"] + plan["R"] + plan["both"]
        assert lines
        assert all(line["n"] > 0 for line in lines)

    def test_zone_grid_counts_every_located_pitch(self):
        """The grid is the only zone map now, so it has to hold everything the
        single-pitch map did: every pitch of every type, by zone."""
        frame = sample_frame()
        payload = report.build_payload(frame, "Test Pitcher", 2025)
        slices = payload["zone_slices"]
        for pitch in slices["pitches"]:
            counted = sum(c["n"] for c in slices["cells"] if c["p"] == pitch)
            assert counted == int((frame["pitch_type"] == pitch).sum())

    def test_count_grid_covers_all_twelve_counts(self):
        """The grid draws a fixed 4x3 lattice, so every pitch needs every
        count, including counts where he never threw it."""
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        for pitch in payload["count_grid"]["pitches"]:
            assert len(pitch["cells"]) == 12

    def test_stacked_count_chart_is_gone(self):
        """Its data would be a field nothing reads."""
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        assert "count_mix" not in payload
        assert "countChart" not in report.render(payload)


class TestArsenalDetailTable:
    """One table for what the card's arsenal table leaves out. It replaced a
    second arsenal table, a platoon table and a usage bar chart, each of
    which held one thing shown nowhere else."""

    def swinging_frame(self):
        return make_frame([
            {"pitch_type": "FF", "n": 120, "stand": "L"},
            {"pitch_type": "SL", "n": 60, "stand": "L", "is_swing": True, "is_whiff": True},
            {"pitch_type": "SL", "n": 20, "stand": "L", "is_swing": True},
            {"pitch_type": "FF", "n": 100, "stand": "R"},
            {"pitch_type": "SL", "n": 50, "stand": "R", "is_swing": True, "is_whiff": True},
            {"pitch_type": "SL", "n": 50, "stand": "R", "is_swing": True},
        ])

    def test_carries_whiff_rate_by_batter_side(self):
        payload = report.build_payload(self.swinging_frame(), "Test Pitcher", 2025)
        slider = next(r for r in payload["arsenal"] if r["pitch_type"] == "SL")
        assert slider["whiff_L"] == pytest.approx(0.75)
        assert slider["whiff_R"] == pytest.approx(0.50)
        assert (slider["n_L"], slider["n_R"]) == (80, 100)

    def test_a_thin_side_is_blank_not_a_rate(self):
        """Nine sliders to lefties give a whiff rate worth nothing. The usage
        stays on the card; the rate is left empty."""
        frame = make_frame([
            {"pitch_type": "FF", "n": 191, "stand": "L"},
            {"pitch_type": "SL", "n": 9, "stand": "L", "is_swing": True, "is_whiff": True},
            {"pitch_type": "FF", "n": 100, "stand": "R"},
            {"pitch_type": "SL", "n": 100, "stand": "R", "is_swing": True},
        ])
        payload = report.build_payload(frame, "Test Pitcher", 2025)
        slider = next(r for r in payload["arsenal"] if r["pitch_type"] == "SL")
        assert slider["whiff_L"] is None
        assert slider["n_L"] == 9

    def test_side_columns_are_left_off_when_one_side_was_barely_faced(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 200, "stand": "R"},
            {"pitch_type": "SL", "n": 100, "stand": "R"},
            {"pitch_type": "FF", "n": 5, "stand": "L"},
        ])
        payload = report.build_payload(frame, "Test Pitcher", 2025)
        assert payload["arsenal"]
        assert all("whiff_L" not in row for row in payload["arsenal"])

    def test_page_draws_the_side_columns(self):
        html = report.render(
            report.build_payload(self.swinging_frame(), "Test Pitcher", 2025)
        )
        assert "{ f: 'whiff_L', l: 'Whiff vs L'" in html
        assert "{ f: 'zone_rate', l: 'Zone%'" in html


class TestCutSections:
    """Sections removed after the split-half check, and what stayed because
    later work reads it. A removed section takes its payload field with it:
    a field nothing draws is a number nobody checks."""

    REMOVED_FIELDS = ("findings", "count_lifts", "quadrants", "first_pitch",
                      "zone", "platoon", "tto_decomposition",
                      # The ranked findings list and the entropy table.
                      "takeaways", "finding_count", "resolved_count",
                      "comparisons", "flag_labels", "predictability")
    REMOVED_FROM_PAGE = ("arsenalChart", "platoonTable", "liftTable",
                         "findingsTable", "zonePick", "All findings",
                         "run cost unresolved",
                         "Findings, with their evidence", 'class="take"',
                         "Count tendencies", "predTable", "Predictability")
    KEPT_FIELDS = ("release", "setups", "putaway", "location_profile",
                   "band_information", "zone_slices", "movement", "tto")

    def build(self):
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        return payload, report.render(payload)

    def test_removed_sections_leave_no_field_behind(self):
        payload, _ = self.build()
        for field in self.REMOVED_FIELDS:
            assert field not in payload, field

    def test_removed_sections_are_not_on_the_page(self):
        _, html = self.build()
        for text in self.REMOVED_FROM_PAGE:
            assert text not in html, text

    def test_what_tunneling_and_sequencing_read_is_still_there(self):
        payload, html = self.build()
        for field in self.KEPT_FIELDS:
            assert field in payload, field
        for table in ("releaseTable", "putawayTable", "zgrid"):
            assert f'id="{table}"' in html, table
        assert html.index("<h2>Sequencing</h2>") < html.index("<h2>Release consistency</h2>")
        assert "<h2>Splits</h2>" not in html

    def test_by_trip_table_sits_with_the_velocity_chart(self):
        payload, html = self.build()
        assert payload["tto"] and "velo_delta" in payload["tto"][0]
        chart = html.index('id="fatigueChart"')
        trips = html.index('id="ttoTable"')
        assert chart < trips < html.index("<h2>Location</h2>")
        assert "Third time through" not in html
        # The change and the outings it rests on are printed in one cell.
        assert "{ f: 'velo_delta', l: 'vs 1st trip'" in html
        assert "r.velo_pairs" in html

    def test_no_table_carries_an_uncorrected_run_estimate(self):
        """One run estimate in 287 survived correction, so the page prints
        none. The setup and height tables used to print every one, in colour."""
        frame = make_frame([
            {"pitch_type": "FF", "n": 120, "plate_z": 3.3},
            {"pitch_type": "SL", "n": 30, "plate_z": 3.3},
            {"pitch_type": "SL", "n": 120, "plate_z": 1.7},
            {"pitch_type": "FF", "n": 30, "plate_z": 1.7},
        ])
        payload = report.build_payload(frame, "Test Pitcher", 2025)
        assert payload["location_tells"]
        for row in payload["location_tells"] + payload["setups"]:
            assert not set(report.RUN_COLUMNS) & set(row)
            assert {"n", "xwoba_when"} <= set(row)
        html = report.render(payload)
        assert "l: 'Runs'" not in html

    def test_height_note_moved_to_the_height_table(self):
        frame = make_frame([
            {"pitch_type": "FF", "n": 120, "plate_z": 3.3},
            {"pitch_type": "SL", "n": 30, "plate_z": 3.3},
            {"pitch_type": "SL", "n": 120, "plate_z": 1.7},
            {"pitch_type": "FF", "n": 30, "plate_z": 1.7},
        ])
        html = report.render(report.build_payload(frame, "Test Pitcher", 2025))
        heading = html.index("What the pitch's height gives away")
        assert heading < html.index('id="bandNote"') < html.index('id="tellsTable"')


def two_pitch_frame(groups: list[tuple[int, str, str]], stand: str = "R") -> pd.DataFrame:
    """Two-pitch plate appearances: (how many, first pitch, second pitch)."""
    first = make_frame([{"pitch_type": a, "n": n, "stand": stand} for n, a, _ in groups])
    second = make_frame([{"pitch_type": b, "n": n, "stand": stand, "strikes": 1}
                         for n, _, b in groups])
    second["pitch_number"] = 2
    return pd.concat([first, second], ignore_index=True)


class TestSequenceGraph:
    """The Sequencing section opens with a graph of what follows each pitch."""

    def doubling_up(self) -> pd.DataFrame:
        """To righties: another slider 70 times in 100 after one, 20 in 100
        after a four-seam."""
        return two_pitch_frame([(70, "SL", "SL"), (30, "SL", "FF"),
                                (20, "FF", "SL"), (80, "FF", "FF")])

    def build(self):
        payload = report.build_payload(self.doubling_up(), "Test Pitcher", 2025)
        return payload, report.render(payload)

    def test_one_panel_per_batter_side(self):
        payload, _ = self.build()
        panels = payload["sequence_graph"]["panels"]
        assert [(p["side"], p["label"]) for p in panels] == [
            ("L", "To lefties"), ("R", "To righties")]
        assert panels[1]["total"] == 200 and panels[1]["enough"]

    def test_a_side_he_barely_faced_says_so_instead_of_drawing(self):
        payload, html = self.build()
        left = payload["sequence_graph"]["panels"][0]
        assert not left["enough"] and "lines" not in left
        assert "too few to read" in html

    def test_colours_the_follow_ups_that_depart_from_his_usual(self):
        payload, _ = self.build()
        lines = payload["sequence_graph"]["panels"][1]["lines"]
        kinds = {(l["from"], l["to"]): l["kind"] for l in lines}
        assert kinds[("SL", "SL")] == "more"
        assert kinds[("FF", "SL")] == "less"

    def test_headline_states_the_pattern_with_its_count(self):
        payload, html = self.build()
        headline = payload["sequence_graph"]["headline"]
        assert "doubles up" in headline or "goes away" in headline
        assert " of 100)" in headline
        assert headline in html

    def test_drawn_by_the_template_so_it_prints_without_script(self):
        _, html = self.build()
        graph = html.index('class="svgchart seq"')
        assert graph < html.index('<script id="payload"')
        assert 'marker-end="url(#seq-more-R)"' in html
        assert 'stroke-dasharray="6 4"' in html          # less than usual

    def test_every_line_says_what_it_is_made_of(self):
        """Hovering a line gives the count behind it; the label gives the
        share and his usual."""
        _, html = self.build()
        assert "followed 70 of 100 times (70%). His usual is 45%." in html
        assert "70% · usual 45%" in html

    def test_sits_above_the_setup_table_and_release_consistency(self):
        _, html = self.build()
        order = [html.index(text) for text in (
            "<h2>Sequencing</h2>", 'class="svgchart seq"',
            "Setups by pitch and height", "Two-strike approach",
            "<h2>Release consistency</h2>")]
        assert order == sorted(order)

    def test_says_so_when_nothing_stands_out(self):
        frame = two_pitch_frame([(50, "SL", "SL"), (50, "SL", "FF"),
                                 (50, "FF", "SL"), (50, "FF", "FF")])
        payload = report.build_payload(frame, "Test Pitcher", 2025)
        html = report.render(payload)
        assert payload["sequence_graph"]["panels"][1]["standouts"] == 0
        assert payload["sequence_graph"]["headline"].startswith("No pitch-to-pitch")
        assert "Nothing stands out to this side" in html

    def test_payload_stays_json_safe(self):
        payload, _ = self.build()
        json.dumps(payload, allow_nan=False)

    def test_a_frame_with_no_follow_ups_still_renders(self):
        """Every plate appearance one pitch long: nothing follows anything."""
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        assert all(not p["enough"] for p in payload["sequence_graph"]["panels"])
        assert "<h2>Sequencing</h2>" in report.render(payload)


class TestDataGaps:
    def test_reports_a_missing_optional_column(self):
        frame = sample_frame().drop(columns=["n_thruorder_pitcher"])
        payload = report.build_payload(frame, "Test Pitcher", 2025)
        assert any(g["column"] == "n_thruorder_pitcher"
                   for g in payload["data_gaps"])

    def test_silent_when_everything_is_present(self):
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        assert payload["data_gaps"] == []

    def test_display_names_are_arsenal_only(self):
        """A lookup built from raw labels advertises pitches the arsenal table
        has never heard of."""
        frame = pd.concat([
            sample_frame(),
            make_frame([{"pitch_type": "ST", "n": 1, "display": "Sweeper"}]),
        ]).reset_index(drop=True)
        payload = report.build_payload(frame, "Test Pitcher", 2025)
        assert "ST" not in payload["display_names"]
        assert "ST" not in [p["pitch"] for p in payload["count_grid"]["pitches"]]


class TestFatigueReliability:
    def test_marks_buckets_backed_by_too_few_outings(self):
        """A late bucket holds only the starts that got that deep, so its rise
        is selection rather than recovery."""
        many = make_frame([
            {"pitch_type": "FF", "n": 15, "game_pk": g} for g in range(1, 21)
        ])
        few = make_frame([
            {"pitch_type": "FF", "n": 20, "game_pk": g} for g in range(1, 3)
        ])
        frame = pd.concat([many, few]).reset_index(drop=True)
        payload = report.build_payload(frame, "Test Pitcher", 2025)
        assert payload["fatigue"]["reliable"]
        assert not all(payload["fatigue"]["reliable"])

    def test_unreliable_buckets_are_kept_not_dropped(self):
        """Trimming them would make the curve look better supported than the
        evidence, and the reader would not know."""
        many = make_frame([
            {"pitch_type": "FF", "n": 15, "game_pk": g} for g in range(1, 21)
        ])
        few = make_frame([
            {"pitch_type": "FF", "n": 20, "game_pk": g} for g in range(1, 3)
        ])
        frame = pd.concat([many, few]).reset_index(drop=True)
        payload = report.build_payload(frame, "Test Pitcher", 2025)
        f = payload["fatigue"]
        assert len(f["buckets"]) == len(f["reliable"]) == len(f["velo"])


def card_html(html: str) -> str:
    """The dugout card's own markup, without the pages behind it or the
    embedded payload, which still carries everything the card leaves off."""
    return html[html.index('<section class="dugout"'):html.index('<header class="top">')]


def thin_side_frame() -> pd.DataFrame:
    """Plenty to righties, a handful of sliders to lefties, and a hitter's
    count to righties in which he mixes."""
    return make_frame([
        {"pitch_type": "FF", "n": 120, "stand": "L"},
        {"pitch_type": "SL", "n": 10, "stand": "L", "zone": 13},
        {"pitch_type": "FF", "n": 100, "stand": "R"},
        {"pitch_type": "SL", "n": 100, "stand": "R", "zone": 13},
        {"pitch_type": "FF", "n": 45, "stand": "R", "balls": 2},
        {"pitch_type": "SL", "n": 55, "stand": "R", "balls": 2, "zone": 13},
    ])


class TestDugoutCard:
    def test_payload_carries_the_card(self):
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        card = payload["card"]
        assert {"season", "arsenal", "recent", "pies", "zones", "plan",
                "key_lines", "order"} <= set(card)

    def test_headline_strip_is_gone(self):
        """It led with "most predictable count: 3-0, four-seam 82%" — 32
        pitches, a count where every pitcher throws a fastball."""
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        assert "kpis" not in payload

    def test_card_renders_first_and_without_javascript(self):
        """The card is the page that gets printed and folded, so it is drawn by
        the template, not by scripts."""
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        html = report.render(payload)
        card_at = html.find('<section class="dugout"')
        behind_at = html.find("Behind the card")
        evidence_at = html.find("<h2>Location</h2>")
        assert 0 < card_at < behind_at < evidence_at

        card = card_html(html)
        assert "<script" not in card
        for heading in ("Arsenal", "Pitch mix by situation", "Location by pitch",
                        "Keys vs LHH", "Keys vs RHH", "Either side",
                        "Running game", "Scout notes"):
            assert heading in card

    def test_card_is_one_printed_page(self):
        """Sized to a letter page, and the next section starts on a new one.
        That the content fits is checked on real pitchers, not here: every
        size on the card is relative to its width, so it is the same picture
        on paper as on screen."""
        html = report.render(report.build_payload(sample_frame(), "Test Pitcher", 2025))
        assert "@page{size:letter;margin:.35in}" in html
        assert "break-after:page" in html
        assert "font-size:1.14cqw" in html

    def test_tendency_rows_cover_every_situation(self):
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        situations = [t["situation"] for t in payload["card"]["tendencies"]]
        from src.metrics import counts
        assert situations == counts.SITUATIONS


class TestCardPies:
    def pies(self, frame=None):
        payload = report.build_payload(
            sample_frame() if frame is None else frame, "Test Pitcher", 2025)
        return payload["card"]

    def test_five_situations_for_each_side(self):
        from src.metrics import counts
        card = self.pies()
        assert [row["side"] for row in card["pies"]] == ["L", "R"]
        for row in card["pies"]:
            assert [p["situation"] for p in row["pies"]] == counts.SITUATIONS

    def test_each_pie_is_the_whole_mix(self):
        for row in self.pies()["pies"]:
            for pie in row["pies"]:
                if pie["n"]:
                    assert sum(s["share"] for s in pie["slices"]) == pytest.approx(1.0)
                    assert sum(s["n"] for s in pie["slices"]) == pie["n"]

    def test_pitches_go_round_every_pie_in_the_same_order(self):
        card = self.pies()
        rank = {p: i for i, p in enumerate(card["order"])}
        for row in card["pies"]:
            for pie in row["pies"]:
                ranks = [rank[s["pitch"]] for s in pie["slices"]]
                assert ranks == sorted(ranks)

    def test_a_situation_never_seen_is_an_empty_pie(self):
        """Every pitch in the sample frame is a first pitch."""
        card = self.pies()
        two_strikes = card["pies"][0]["pies"][-1]
        assert two_strikes["n"] == 0 and two_strikes["slices"] == []
        html = report.render(report.build_payload(sample_frame(), "Test Pitcher", 2025))
        assert "no pitches" in card_html(html)

    def test_a_thin_situation_is_marked_for_fading(self):
        card = self.pies(thin_side_frame())
        first_to_lefties = card["pies"][0]["pies"][0]
        assert first_to_lefties["n"] == 130 and first_to_lefties["reliable"] is True
        thin = make_frame([
            {"pitch_type": "FF", "n": 15, "stand": "L"},
            {"pitch_type": "SL", "n": 5, "stand": "L"},
            {"pitch_type": "FF", "n": 200, "stand": "R"},
        ])
        assert self.pies(thin)["pies"][0]["pies"][0]["reliable"] is False


class TestCardZones:
    def zones(self, frame) -> dict:
        return report.build_payload(frame, "Test Pitcher", 2025)["card"]["zones"]

    def test_one_map_per_pitch_per_side_in_arsenal_order(self):
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        zones = payload["card"]["zones"]
        for row in zones["sides"]:
            assert [m["pitch"] for m in row["maps"]] == payload["card"]["order"]

    def test_a_map_is_the_whole_of_that_pitch_to_that_side(self):
        for row in self.zones(sample_frame())["sides"]:
            for m in row["maps"]:
                cells = m["inner"] + m["outer"]
                assert len(cells) == 13
                assert sum(c["n"] for c in cells) == m["n"]
                assert sum(c["share"] for c in cells) == pytest.approx(1.0)

    def test_too_few_pitches_to_a_side_is_not_mapped(self):
        """Ten sliders over thirteen zones is a picture of noise. The pitch
        keeps its place in the row and says how many there were."""
        lefties = self.zones(thin_side_frame())["sides"][0]
        slider = next(m for m in lefties["maps"] if m["pitch"] == "SL")
        assert slider["n"] == 10 and slider["enough"] is False
        assert "inner" not in slider
        html = report.render(report.build_payload(thin_side_frame(), "Test Pitcher", 2025))
        assert "too few" in card_html(html)

    def test_an_empty_zone_prints_no_number(self):
        lefties = self.zones(sample_frame())["sides"][0]
        fastball = next(m for m in lefties["maps"] if m["pitch"] == "FF")
        by_zone = {c["zone"]: c for c in fastball["inner"] + fastball["outer"]}
        assert by_zone[5]["pct"] == 100
        assert by_zone[1]["pct"] is None


class TestCardKeepsWhatRepeats:
    """The card prints what a hitter can act on. Everything it used to print
    is still computed and still in the payload."""

    def test_he_mixes_is_in_the_plan_and_not_on_the_card(self):
        payload = report.build_payload(thin_side_frame(), "Test Pitcher", 2025)
        plan = [k["text"] for k in payload["card"]["plan"]["R"]]
        assert any(text.startswith("He still mixes") for text in plan)
        assert "He still mixes" not in card_html(report.render(payload))

    def test_card_lines_are_all_advice(self):
        payload = report.build_payload(thin_side_frame(), "Test Pitcher", 2025)
        for side in ("L", "R", "both"):
            assert all(k["card"] for k in payload["card"]["key_lines"][side])

    def test_a_side_with_nothing_to_say_says_so(self):
        payload = report.build_payload(thin_side_frame(), "Test Pitcher", 2025)
        payload["card"]["key_lines"]["R"] = []
        assert "Nothing to this side clears its bar" in card_html(report.render(payload))

    def test_results_and_runs_stay_off_the_card(self):
        """A pitch's xwOBA in one half of a season says little about the
        other, so the card's arsenal table does not carry it."""
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        card = card_html(report.render(payload))
        arsenal_table = card[card.index("<table"):card.index("</table>")]
        assert "xwOBA" not in arsenal_table and "Spin" not in arsenal_table
        assert "K−BB" not in card

    def test_what_left_the_card_is_still_in_the_payload(self):
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        card = payload["card"]
        assert "k_minus_bb" in card["season"]
        assert {"spin", "xwoba", "usage"} <= set(card["arsenal"][0])
        assert card["tendencies"] and card["plan"]["basis"]

    def test_spin_moved_to_the_detail_table(self):
        html = report.render(report.build_payload(sample_frame(), "Test Pitcher", 2025))
        assert "{ f: 'spin', l: 'Spin'" in html

    def test_a_thin_rate_is_flagged_by_its_sample(self):
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        card = payload["card"]
        assert card["rate_min"] == 30
        assert all(row["swings"] < card["rate_min"] for row in card["arsenal"])
        assert 'class="dim"' in card_html(report.render(payload))


class TestZoneMapLayout:
    """Two layout bugs that hid real numbers.

    The strike-zone box is drawn over the middle of the four outside quadrants.
    Labels placed in a quadrant's centre, or in the corner nearest the zone,
    end up underneath it — which hid all four outside labels on the grids,
    including the 35% of Skubal's changeups that finish low and to the
    catcher's right.
    """

    def render(self):
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        return report.render(payload)

    def test_outside_labels_are_pinned_to_outer_corners(self):
        html = self.render()
        assert ".oz-11 .lab{top:0;left:0" in html
        assert ".oz-12 .lab{top:0;right:0" in html
        assert ".oz-13 .lab{bottom:0;left:0" in html
        assert ".oz-14 .lab{bottom:0;right:0" in html

    def test_the_grids_use_the_pinned_labels(self):
        html = self.render()
        assert "function outerCell(" in html
        assert "return outerCell(z, bg, text" in html

    def test_small_grids_open_on_share_not_lift(self):
        """Lift against the whole season is 1.0x in every cell when nothing is
        sliced, which reads as a finding and is only arithmetic."""
        html = self.render()
        select = html[html.index('<select id="sliceMode">'):]
        first_option = select[:select.index("</option>")]
        assert 'value="share"' in first_option


class TestRender:
    def test_produces_a_complete_document(self):
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        html = report.render(payload)
        assert html.lstrip().startswith("<!DOCTYPE html>")
        assert html.rstrip().endswith("</html>")

    def test_leaves_no_unrendered_template_syntax(self):
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        html = report.render(payload)
        assert "{{" not in html
        assert "{%" not in html

    def test_embedded_payload_parses_as_json(self):
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        html = report.render(payload)
        match = re.search(
            r'<script id="payload" type="application/json">(.*?)</script>',
            html, re.S,
        )
        assert match
        parsed = json.loads(match.group(1).replace("<\\/", "</"))
        assert parsed["meta"]["pitcher"] == "Test Pitcher"

    def test_escapes_a_closing_script_tag_in_the_data(self):
        """Pitch names will never contain markup. The payload is built from an
        upstream feed, so one day something will."""
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        payload["meta"]["pitcher"] = "</script><script>alert(1)</script>"
        html = report.render(payload)
        assert "</script><script>alert(1)" not in html

    def test_pitcher_name_is_escaped_in_the_body(self):
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        payload["meta"]["pitcher"] = "<b>Bold</b>"
        html = report.render(payload)
        assert "<b>Bold</b>" not in html
        assert "&lt;b&gt;Bold&lt;/b&gt;" in html

    def test_writes_a_file(self, tmp_path=None):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.html"
            written = report.write(sample_frame(), "Test Pitcher", 2025, out_path=out)
            assert written.exists()
            assert written.stat().st_size > 5000
