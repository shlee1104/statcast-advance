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
                "is_whiff": False,
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

    def test_every_claim_carries_its_sample_size(self):
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        for finding in payload["findings"]:
            assert "n=" in finding["claim"]

    def test_zone_rows_sum_to_one_per_pitch(self):
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        zones = payload["zone"]["inner"] + payload["zone"]["outer"]
        for row in payload["zone"]["rows"]:
            total = sum(row.get(str(z)) or 0 for z in zones)
            assert total == pytest.approx(1.0)

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

    def test_takeaways_are_a_subset_of_findings(self):
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        claims = {f["claim"] for f in payload["findings"]}
        for t in payload["takeaways"]:
            assert t["claim"] in claims


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


class TestDugoutCard:
    def test_payload_carries_the_card(self):
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        card = payload["card"]
        assert {"season", "arsenal", "tendencies", "recent", "plan"} <= set(card)

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
        card_at = html.find("Dugout card")
        findings_at = html.find("Findings, with their evidence")
        assert 0 < card_at < findings_at
        assert "Lefty hitting plan" in html
        assert "Righty hitting plan" in html
        assert "Not covered:" in html

    def test_tendency_rows_cover_every_situation(self):
        payload = report.build_payload(sample_frame(), "Test Pitcher", 2025)
        situations = [t["situation"] for t in payload["card"]["tendencies"]]
        from src.metrics import counts
        assert situations == counts.SITUATIONS


class TestZoneMapLayout:
    """Two layout bugs that hid real numbers.

    The strike-zone box is drawn over the middle of the four outside quadrants.
    Labels placed in a quadrant's centre, or in the corner nearest the zone,
    end up underneath it — which hid zone 14 on the single map and all four
    outside labels on the small grids, including the 35% of Skubal's changeups
    that finish low and to the catcher's right.
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

    def test_both_maps_use_the_pinned_labels(self):
        html = self.render()
        assert "outerCell(z, shade(" in html          # single map
        assert "return outerCell(z, bg, text" in html  # small grids

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
