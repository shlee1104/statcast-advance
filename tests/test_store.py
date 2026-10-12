"""Tests for src/store.py: when a cached season can be trusted without re-fetching."""

from __future__ import annotations

import datetime as dt

import pytest

from src import store


@pytest.fixture
def conn(tmp_path):
    connection = store.connect(tmp_path / "cache.duckdb")
    yield connection
    connection.close()


def log_fetch(conn, season: int, fetched_at: dt.datetime, rows: int = 2500) -> None:
    conn.execute("INSERT INTO fetch_log VALUES (?, ?, ?, ?, ?)",
                 [1, season, "pitcher", rows, fetched_at])


class TestIsCached:
    def test_never_fetched(self, conn):
        assert not store.is_cached(conn, 1, 2024)

    def test_a_finished_season_fetched_after_it_ended_is_kept(self, conn):
        log_fetch(conn, 2024, dt.datetime(2025, 3, 1))
        assert store.is_cached(conn, 1, 2024)

    def test_a_finished_season_fetched_mid_season_is_not_complete(self, conn):
        """Fetched in July, never touched again. Once the year turned this was
        reported as a complete, permanent copy of the season, with August
        onward missing and nothing that would ever fetch it."""
        log_fetch(conn, 2024, dt.datetime(2024, 7, 4))
        assert not store.is_cached(conn, 1, 2024)

    def test_an_empty_fetch_is_not_a_cache_hit(self, conn):
        log_fetch(conn, 2024, dt.datetime(2025, 3, 1), rows=0)
        assert not store.is_cached(conn, 1, 2024)

    def test_the_current_season_expires(self, conn):
        this_year = dt.date.today().year
        log_fetch(conn, this_year, dt.datetime.now() - dt.timedelta(days=30))
        assert not store.is_cached(conn, 1, this_year)

    def test_the_current_season_is_fresh_for_a_while(self, conn):
        this_year = dt.date.today().year
        log_fetch(conn, this_year, dt.datetime.now() - dt.timedelta(minutes=5))
        assert store.is_cached(conn, 1, this_year)

    def test_a_season_is_over_by_mid_november(self):
        assert store.season_complete_after(2025) == dt.datetime(2025, 11, 15)
