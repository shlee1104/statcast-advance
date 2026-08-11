"""Local DuckDB cache for pitch-level data.

Savant is slow enough (5-20s per player-season) that re-fetching on every run
makes iterating on the report painful. This module persists what we pull so a
repeat request is instant, and gives the analysis layer a real SQL surface to
query instead of a pandas DataFrame passed around in memory.

DuckDB rather than SQLite or Postgres: it is a single file with no server to
run, it reads pandas DataFrames natively, and it is columnar, which is the
right shape for "average velocity grouped by count and pitch type" queries.

Two tables:
  pitches       one row per pitch
  fetch_log     what we have already downloaded, and when
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import duckdb
import pandas as pd

from src import config

log = logging.getLogger(__name__)

# Columns the project actually uses, with their target types. Savant returns
# 118+ columns; storing all of them wastes space and hides which fields the
# analysis genuinely depends on. Anything added here must also be handled in
# clean.py.
PITCH_SCHEMA: dict[str, str] = {
    # Identity and game context
    "game_pk": "BIGINT",
    "game_date": "DATE",
    "game_type": "VARCHAR",
    "game_year": "INTEGER",
    "at_bat_number": "INTEGER",
    "pitch_number": "INTEGER",
    "inning": "INTEGER",
    "inning_topbot": "VARCHAR",
    # Participants
    "pitcher": "BIGINT",
    "batter": "BIGINT",
    "player_name": "VARCHAR",
    "p_throws": "VARCHAR",
    "stand": "VARCHAR",
    # The pitch itself
    "pitch_type": "VARCHAR",
    "pitch_name": "VARCHAR",
    "release_speed": "DOUBLE",
    "release_spin_rate": "DOUBLE",
    "spin_axis": "DOUBLE",
    "release_extension": "DOUBLE",
    "release_pos_x": "DOUBLE",
    "release_pos_z": "DOUBLE",
    "pfx_x": "DOUBLE",
    "pfx_z": "DOUBLE",
    "plate_x": "DOUBLE",
    "plate_z": "DOUBLE",
    "sz_top": "DOUBLE",
    "sz_bot": "DOUBLE",
    "zone": "INTEGER",
    # Count state, the backbone of the sequencing analysis
    "balls": "INTEGER",
    "strikes": "INTEGER",
    "outs_when_up": "INTEGER",
    # Outcome
    "description": "VARCHAR",
    "events": "VARCHAR",
    "type": "VARCHAR",
    "bb_type": "VARCHAR",
    "launch_speed": "DOUBLE",
    "launch_angle": "DOUBLE",
    "estimated_woba_using_speedangle": "DOUBLE",
    "woba_value": "DOUBLE",
    "delta_run_exp": "DOUBLE",
}

# A pitch is uniquely identified by game, plate appearance, and its number
# within that plate appearance.
PRIMARY_KEY = ["game_pk", "at_bat_number", "pitch_number"]


def connect(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Open the cache, creating the file and schema if needed."""
    target = db_path or config.resolve_path("data.db_path")
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(target))
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    columns = ",\n    ".join(f"{name} {sql_type}" for name, sql_type in PITCH_SCHEMA.items())
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS pitches (
            {columns},
            PRIMARY KEY ({", ".join(PRIMARY_KEY)})
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fetch_log (
            mlbam_id     BIGINT,
            season       INTEGER,
            player_type  VARCHAR,
            row_count    INTEGER,
            fetched_at   TIMESTAMP,
            PRIMARY KEY (mlbam_id, season, player_type)
        )
        """
    )


def is_cached(
    conn: duckdb.DuckDBPyConnection,
    mlbam_id: int,
    season: int,
    player_type: str = "pitcher",
) -> bool:
    """Has this player-season been fetched recently enough to reuse?

    Completed seasons never change, but an in-progress season gains games
    daily, so cached data expires after config's cache_ttl_days.
    """
    row = conn.execute(
        """
        SELECT fetched_at, row_count
        FROM fetch_log
        WHERE mlbam_id = ? AND season = ? AND player_type = ?
        """,
        [mlbam_id, season, player_type],
    ).fetchone()

    if row is None or row[1] == 0:
        return False

    fetched_at: dt.datetime = row[0]
    current_season = dt.date.today().year

    # A finished season is immutable, so the cache never goes stale.
    if season < current_season:
        return True

    ttl_days = float(config.get("data.cache_ttl_days", 1))
    age = dt.datetime.now() - fetched_at
    return age < dt.timedelta(days=ttl_days)


def coerce_schema_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Force each column to a dtype DuckDB will accept for its declared type.

    Savant returns everything loosely typed, and pandas widens any column that
    has ever held a null to float64. Inserting a float64 column into an INTEGER
    column fails, so integer fields are converted to pandas' nullable "Int64",
    which DuckDB maps cleanly onto a nullable integer.

    `zone` is the column that makes this necessary: it is genuinely an integer,
    but it is null on pitches Statcast could not locate, so it always arrives
    as a float.
    """
    result = frame.copy()

    for column, sql_type in PITCH_SCHEMA.items():
        if column not in result.columns:
            continue

        if sql_type in ("INTEGER", "BIGINT"):
            # round() first: a float that is 3.0 converts cleanly, but a
            # value like 2.9999 from a lossy round-trip would truncate to 2.
            result[column] = (
                pd.to_numeric(result[column], errors="coerce").round().astype("Int64")
            )
        elif sql_type == "DOUBLE":
            result[column] = pd.to_numeric(result[column], errors="coerce")
        elif sql_type == "DATE":
            result[column] = pd.to_datetime(result[column], errors="coerce")
        elif sql_type == "VARCHAR":
            # Guard against NaN becoming the literal string "nan".
            column_data = result[column]
            result[column] = column_data.where(column_data.notna(), None).astype(object)

    return result


def save_pitches(
    conn: duckdb.DuckDBPyConnection,
    frame: pd.DataFrame,
    mlbam_id: int,
    season: int,
    player_type: str = "pitcher",
) -> int:
    """Write pitches to the cache, replacing any rows we already held.

    Uses delete-then-insert on the primary key rather than a bare INSERT so
    that re-fetching an in-progress season updates existing rows instead of
    failing on a key collision.
    """
    if frame.empty:
        _log_fetch(conn, mlbam_id, season, player_type, 0)
        return 0

    missing = [c for c in PITCH_SCHEMA if c not in frame.columns]
    if missing:
        raise ValueError(
            f"Frame is missing expected columns: {missing}. "
            f"Run it through clean.prepare_for_storage() first."
        )

    staged = coerce_schema_types(frame[list(PITCH_SCHEMA.keys())])

    # Savant occasionally returns the same pitch twice. Duplicates would
    # violate the primary key, so collapse them before insert.
    before = len(staged)
    staged = staged.drop_duplicates(subset=PRIMARY_KEY, keep="last")
    if len(staged) < before:
        log.warning("Dropped %s duplicate pitch rows from Savant", before - len(staged))

    conn.register("staged_pitches", staged)
    try:
        conn.execute("BEGIN TRANSACTION")
        conn.execute(
            f"""
            DELETE FROM pitches
            WHERE ({", ".join(PRIMARY_KEY)}) IN (
                SELECT {", ".join(PRIMARY_KEY)} FROM staged_pitches
            )
            """
        )
        conn.execute("INSERT INTO pitches SELECT * FROM staged_pitches")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.unregister("staged_pitches")

    _log_fetch(conn, mlbam_id, season, player_type, len(staged))
    log.info("Cached %s pitches for %s (%s)", len(staged), mlbam_id, season)
    return len(staged)


def _log_fetch(
    conn: duckdb.DuckDBPyConnection,
    mlbam_id: int,
    season: int,
    player_type: str,
    row_count: int,
) -> None:
    conn.execute(
        """
        DELETE FROM fetch_log
        WHERE mlbam_id = ? AND season = ? AND player_type = ?
        """,
        [mlbam_id, season, player_type],
    )
    conn.execute(
        "INSERT INTO fetch_log VALUES (?, ?, ?, ?, ?)",
        [mlbam_id, season, player_type, row_count, dt.datetime.now()],
    )


def load_pitcher_season(
    conn: duckdb.DuckDBPyConnection,
    mlbam_id: int,
    season: int,
) -> pd.DataFrame:
    """Read one pitcher-season back out, in true chronological pitch order.

    Ordering matters: the sequencing metrics read consecutive rows to build
    transition matrices, so an unordered result would silently produce
    meaningless numbers.
    """
    return conn.execute(
        """
        SELECT *
        FROM pitches
        WHERE pitcher = ? AND game_year = ?
        ORDER BY game_date, game_pk, at_bat_number, pitch_number
        """,
        [mlbam_id, season],
    ).df()


def cache_summary(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """What is currently in the cache. Useful for debugging and the CLI."""
    return conn.execute(
        """
        SELECT
            f.mlbam_id,
            f.season,
            f.player_type,
            f.row_count,
            f.fetched_at,
            MAX(p.player_name) AS player_name
        FROM fetch_log f
        LEFT JOIN pitches p
            ON p.pitcher = f.mlbam_id AND p.game_year = f.season
        GROUP BY ALL
        ORDER BY f.fetched_at DESC
        """
    ).df()
