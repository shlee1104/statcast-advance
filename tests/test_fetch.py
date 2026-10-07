"""Tests for name resolution.

The calibration run lost four pitchers to one bug: the player register keeps
accents ("Díaz") and the lookup compares exactly, so a typed "Edwin Diaz"
matched nothing.
"""

from __future__ import annotations

import pandas as pd

from src import fetch


def register() -> pd.DataFrame:
    return pd.DataFrame([
        {"name_first": "Edwin", "name_last": "Díaz", "key_mlbam": 621242.0},
        {"name_first": "Andrés", "name_last": "Muñoz", "key_mlbam": 662253.0},
        {"name_first": "Cristopher", "name_last": "Sánchez", "key_mlbam": 650911.0},
        {"name_first": "Old", "name_last": "Timer", "key_mlbam": None},
    ])


class TestMatchWithoutAccents:
    def test_unaccented_name_finds_accented_register_entry(self):
        hit = fetch.match_without_accents(register(), "Edwin", "Diaz")
        assert hit["key_mlbam"].tolist() == [621242.0]

    def test_accents_in_both_first_and_last_name(self):
        hit = fetch.match_without_accents(register(), "andres", "munoz")
        assert hit["key_mlbam"].tolist() == [662253.0]

    def test_accented_input_still_matches(self):
        hit = fetch.match_without_accents(register(), "Cristopher", "Sánchez")
        assert len(hit) == 1

    def test_rows_without_an_mlbam_id_are_dropped(self):
        assert len(fetch.match_without_accents(register(), "Old", "Timer")) == 0

    def test_no_match_is_empty(self):
        assert len(fetch.match_without_accents(register(), "Edwin", "Diazz")) == 0
