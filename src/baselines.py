"""League reference rates, for turning raw numbers into findings.

A scouting report that says "throws a fastball 82% of the time in 3-0" has
stated a fact and communicated nothing, because nearly every pitcher does.
The same line against a league rate of 87% says he is slightly *less*
predictable there than his peers — which is the opposite conclusion, and the
useful one.

This module builds that comparison layer by sampling days across a season
rather than downloading it whole. One date is roughly 4,400 pitches, so thirty
dates give around 130,000 — far more than needed for stable count-level and
pitch-type-level rates, at a thirtieth of the requests.

Sampling is stratified: the season is divided into equal intervals and one date
is drawn from each, so coverage is guaranteed across April through September
rather than left to chance. The draw within each interval is randomized (from a
fixed seed, so results reproduce) to avoid landing on a fixed weekday spacing,
which would bias the sample toward whichever days carry more games.

Baselines are computed separately by pitcher handedness. Measuring a
right-hander against a pool that includes left-handers understates how unusual
his mix is, since the underlying platoon logic differs.

Aggregation is written in SQL rather than pandas. Grouping, ranking, and
joining across a hundred thousand rows is what a columnar database is for, and
the percentile calculations are far clearer as window functions.
"""

from __future__ import annotations

import datetime as dt
import logging
import random

import duckdb
import pandas as pd

from src import config

log = logging.getLogger(__name__)


def sample_dates(
    season: int,
    n_dates: int | None = None,
    seed: int | None = None,
) -> list[str]:
    """Draw a stratified sample of dates spread across the regular season.

    Splits the season window into `n_dates` equal intervals and draws one date
    from each. Deterministic given the seed, so a rebuild reproduces the same
    sample and baselines do not drift between runs for no reason.

    Returns ISO date strings in chronological order.
    """
    if n_dates is None:
        n_dates = int(config.get("baselines.n_dates", 30))
    if seed is None:
        seed = int(config.get("baselines.seed", 17))

    start_md = str(config.get("baselines.season_start", "03-20"))
    end_md = str(config.get("baselines.season_end", "10-01"))

    start = dt.date.fromisoformat(f"{season}-{start_md}")
    end = dt.date.fromisoformat(f"{season}-{end_md}")
    span = (end - start).days

    if n_dates < 1:
        raise ValueError("n_dates must be at least 1")
    if n_dates > span:
        raise ValueError(f"Cannot draw {n_dates} dates from a {span}-day season")

    rng = random.Random(seed)
    width = span / n_dates

    dates = []
    for i in range(n_dates):
        lower = int(i * width)
        upper = max(lower, int((i + 1) * width) - 1)
        dates.append((start + dt.timedelta(days=rng.randint(lower, upper))).isoformat())

    return sorted(dates)


def build(
    conn: duckdb.DuckDBPyConnection,
    season: int,
    n_dates: int | None = None,
    progress: bool = True,
) -> int:
    """Fetch and store the league sample for a season.

    Skips dates already present, so an interrupted build resumes rather than
    starting over. Returns the number of pitches added on this run.
    """
    from src import clean, fetch, store

    wanted = sample_dates(season, n_dates)
    already = store.cached_league_dates(conn, season)
    pending = [d for d in wanted if d not in already]

    if progress:
        print(f"League sample for {season}: {len(wanted)} dates, "
              f"{len(already)} cached, {len(pending)} to fetch")

    added = 0
    for i, game_date in enumerate(pending, start=1):
        if progress:
            print(f"  [{i}/{len(pending)}] {game_date} ... ", end="", flush=True)

        raw = fetch.fetch_date_range(game_date, game_date)
        if raw.empty:
            # An off day. Log it so the date is not retried on every rebuild.
            store._log_league_sample(conn, game_date, season, 0)
            if progress:
                print("no games")
            continue

        cleaned = clean.prepare_for_storage(raw)
        n = store.save_league_pitches(conn, cleaned, game_date, season)
        added += n
        if progress:
            print(f"{n:,} pitches")

    return added


def sample_summary(conn: duckdb.DuckDBPyConnection, season: int) -> pd.DataFrame:
    """What the league sample currently contains."""
    return conn.execute(
        """
        SELECT
            COUNT(DISTINCT game_date)                         AS dates,
            COUNT(*)                                          AS pitches,
            COUNT(DISTINCT pitcher)                           AS pitchers,
            MIN(game_date)                                    AS first_date,
            MAX(game_date)                                    AS last_date
        FROM league_pitches
        WHERE game_year = ?
        """,
        [season],
    ).df()


def count_mix(
    conn: duckdb.DuckDBPyConnection,
    season: int,
    p_throws: str = "R",
    min_pitches: int = 200,
) -> pd.DataFrame:
    """League pitch-type usage by count, for one pitcher handedness.

    Returns one row per (count, pitch_type) with the league share and that
    pitch's rank within the count. This is the table a pitcher's own
    `mix_by_count` gets compared against.
    """
    return conn.execute(
        """
        WITH eligible AS (
            SELECT
                balls || '-' || strikes AS count,
                pitch_type
            FROM league_pitches
            WHERE game_year = ?
              AND p_throws = ?
              AND pitch_type IS NOT NULL
              AND balls BETWEEN 0 AND 3
              AND strikes BETWEEN 0 AND 2
        ),
        by_count_pitch AS (
            SELECT count, pitch_type, COUNT(*) AS n
            FROM eligible
            GROUP BY count, pitch_type
        ),
        count_totals AS (
            SELECT count, SUM(n) AS total
            FROM by_count_pitch
            GROUP BY count
        )
        SELECT
            b.count,
            b.pitch_type,
            b.n,
            t.total                                              AS count_total,
            b.n::DOUBLE / t.total                                AS league_share,
            ROW_NUMBER() OVER (
                PARTITION BY b.count ORDER BY b.n DESC
            )                                                    AS rank_in_count
        FROM by_count_pitch b
        JOIN count_totals t USING (count)
        WHERE t.total >= ?
        ORDER BY b.count, league_share DESC
        """,
        [season, p_throws, min_pitches],
    ).df()


def pitch_outcomes(
    conn: duckdb.DuckDBPyConnection,
    season: int,
    p_throws: str = "R",
    min_pitches: int = 200,
) -> pd.DataFrame:
    """League usage, whiff rate, and contact quality by pitch type.

    Whiff rate uses swings as the denominator, matching metrics.events, so the
    two are directly comparable. Mismatched denominators here would produce
    percentile ranks that look plausible and mean nothing.
    """
    return conn.execute(
        """
        WITH tagged AS (
            SELECT
                pitch_type,
                description IN (
                    'foul', 'foul_tip', 'bunt_foul_tip', 'foul_bunt',
                    'hit_into_play', 'missed_bunt',
                    'swinging_strike', 'swinging_strike_blocked'
                )                                               AS is_swing,
                description IN (
                    'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'
                )                                               AS is_whiff,
                zone BETWEEN 1 AND 9                            AS in_zone,
                estimated_woba_using_speedangle                 AS xwoba,
                release_speed,
                delta_run_exp
            FROM league_pitches
            WHERE game_year = ? AND p_throws = ? AND pitch_type IS NOT NULL
        )
        SELECT
            pitch_type,
            COUNT(*)                                            AS n,
            COUNT(*)::DOUBLE / SUM(COUNT(*)) OVER ()            AS league_usage,
            AVG(release_speed)                                  AS avg_velo,
            SUM(in_zone::INT)::DOUBLE / COUNT(*)                AS zone_rate,
            SUM(is_swing::INT)::DOUBLE / COUNT(*)               AS swing_rate,
            CASE WHEN SUM(is_swing::INT) > 0
                 THEN SUM(is_whiff::INT)::DOUBLE / SUM(is_swing::INT)
            END                                                 AS whiff_rate,
            AVG(xwoba)                                          AS avg_xwoba,
            AVG(delta_run_exp)                                  AS avg_run_value
        FROM tagged
        GROUP BY pitch_type
        HAVING COUNT(*) >= ?
        ORDER BY n DESC
        """,
        [season, p_throws, min_pitches],
    ).df()


def putaway_rates(
    conn: duckdb.DuckDBPyConnection,
    season: int,
    p_throws: str = "R",
    min_pitches: int = 100,
) -> pd.DataFrame:
    """League two-strike usage and putaway rate by pitch type.

    The comparison layer for metrics.sequencing.putaway(). A 24% putaway rate
    means nothing until you know whether the league finishes at 18% or 30%
    with that pitch.
    """
    return conn.execute(
        """
        WITH two_strike AS (
            SELECT
                pitch_type,
                events IN ('strikeout', 'strikeout_double_play')  AS is_strikeout,
                description IN (
                    'foul', 'foul_tip', 'bunt_foul_tip', 'foul_bunt',
                    'hit_into_play', 'missed_bunt',
                    'swinging_strike', 'swinging_strike_blocked'
                )                                                 AS is_swing,
                description IN (
                    'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'
                )                                                 AS is_whiff
            FROM league_pitches
            WHERE game_year = ? AND p_throws = ? AND strikes = 2
              AND pitch_type IS NOT NULL
        )
        SELECT
            pitch_type,
            COUNT(*)                                              AS n,
            COUNT(*)::DOUBLE / SUM(COUNT(*)) OVER ()              AS league_usage,
            SUM(is_strikeout::INT)::DOUBLE / COUNT(*)             AS putaway_rate,
            CASE WHEN SUM(is_swing::INT) > 0
                 THEN SUM(is_whiff::INT)::DOUBLE / SUM(is_swing::INT)
            END                                                   AS whiff_rate
        FROM two_strike
        GROUP BY pitch_type
        HAVING COUNT(*) >= ?
        ORDER BY n DESC
        """,
        [season, p_throws, min_pitches],
    ).df()


def compare_count_mix(
    pitcher_mix: pd.DataFrame,
    league: pd.DataFrame,
) -> pd.DataFrame:
    """Join a pitcher's mix_by_count table to league rates and difference them.

    `pitcher_mix` is the wide table from metrics.counts.mix_by_count(); it is
    melted to long form so the join is a plain merge on (count, pitch_type).

    The `delta` column is what a finding is built from: not "throws it 40% of
    the time" but "throws it 18 points more often than the league does".
    """
    long = (
        pitcher_mix.reset_index()
        .melt(id_vars="count", var_name="pitch_type", value_name="pitcher_share")
    )
    merged = long.merge(
        league[["count", "pitch_type", "league_share"]],
        on=["count", "pitch_type"],
        how="left",
    )
    merged["delta"] = merged["pitcher_share"] - merged["league_share"]
    return merged.sort_values("delta", ascending=False, key=abs).reset_index(drop=True)
