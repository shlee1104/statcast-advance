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


def proportion_z(observed_share: float, n: int, league_share: float) -> float:
    """How many standard errors a rate sits from the league rate.

    Uses the league proportion for the standard error, treating it as known
    rather than estimated — the league sample is orders of magnitude larger
    than any pitcher's, so its sampling error is negligible by comparison.

    Roughly |z| > 2 is worth reporting. This is what stops a 16-point deviation
    measured over 39 pitches from being presented with the same confidence as
    the same deviation measured over 900.
    """
    if n is None or n <= 0:
        return float("nan")
    if not (0 < league_share < 1) or pd.isna(observed_share):
        return float("nan")

    standard_error = (league_share * (1 - league_share) / n) ** 0.5
    if standard_error == 0:
        return float("nan")
    return (observed_share - league_share) / standard_error


def compare_arsenal(pitcher_mix: pd.Series, league: pd.DataFrame) -> pd.DataFrame:
    """Pitch-type usage against league usage for the same handedness.

    `pitcher_mix` is the Series from metrics.counts.pitch_mix(); `league` is
    the frame from pitch_outcomes(). The ratio column is what surfaces an
    unusual repertoire — a pitch thrown at six times the league rate is the
    defining fact about a pitcher, and a raw percentage never says so.
    """
    frame = pitcher_mix.rename("usage").rename_axis("pitch_type").reset_index()
    merged = frame.merge(
        league[["pitch_type", "league_usage", "avg_velo", "whiff_rate", "avg_xwoba"]]
        .rename(columns={
            "avg_velo": "league_velo",
            "whiff_rate": "league_whiff",
            "avg_xwoba": "league_xwoba",
        }),
        on="pitch_type",
        how="left",
    )
    merged["usage_delta"] = merged["usage"] - merged["league_usage"]
    merged["usage_ratio"] = merged["usage"] / merged["league_usage"]
    return merged.sort_values("usage", ascending=False).reset_index(drop=True)


def compare_profile(pitcher: pd.DataFrame, league: pd.DataFrame) -> pd.DataFrame:
    """A pitcher's per-pitch stuff and results against league, pitch by pitch.

    `pitcher` is the frame from metrics.arsenal.profile(); `league` is the frame
    from pitch_outcomes(). Where compare_arsenal answers "how unusual is his
    mix", this answers "how good is each pitch in it" — velocity, whiff rate,
    and contact quality, each differenced against pitchers of the same hand.

    `xwoba_delta` is the column the hittable_pitch flag reads. It is signed so
    that positive is bad for the pitcher: hitters are doing more damage against
    this offering than against the league's version of it. A pitch that is both
    heavily used and worse than league is the clearest "sit on this" finding a
    report can make.
    """
    frame = pitcher.rename_axis("pitch_type").reset_index()
    merged = frame.merge(
        league[["pitch_type", "league_usage", "avg_velo", "whiff_rate", "avg_xwoba"]]
        .rename(columns={
            "avg_velo": "league_velo",
            "whiff_rate": "league_whiff",
            "avg_xwoba": "league_xwoba",
        }),
        on="pitch_type",
        how="left",
    )

    merged["velo_delta"] = merged["velo"] - merged["league_velo"]
    merged["whiff_delta"] = merged["whiff_rate"] - merged["league_whiff"]
    merged["xwoba_delta"] = merged["xwoba"] - merged["league_xwoba"]
    merged["usage_ratio"] = merged["usage"] / merged["league_usage"]

    return merged.sort_values("usage", ascending=False).reset_index(drop=True)


def compare_putaway(pitcher: pd.DataFrame, league: pd.DataFrame) -> pd.DataFrame:
    """Two-strike behaviour against league rates for the same handedness.

    Reports usage and putaway separately, and differences them against league,
    because the interesting case is a pitcher leaning hardest on the pitch with
    the smallest edge over league.
    """
    frame = pitcher.rename_axis("pitch_type").reset_index()
    merged = frame.merge(
        league[["pitch_type", "league_usage", "putaway_rate", "whiff_rate"]].rename(
            columns={
                "putaway_rate": "league_putaway",
                "whiff_rate": "league_whiff",
            }
        ),
        on="pitch_type",
        how="left",
    )

    merged["usage_ratio"] = merged["usage"] / merged["league_usage"]
    merged["putaway_delta"] = merged["putaway_rate"] - merged["league_putaway"]
    merged["whiff_delta"] = merged["whiff_rate"] - merged["league_whiff"]
    merged["putaway_z"] = [
        proportion_z(row.putaway_rate, row.n, row.league_putaway)
        for row in merged.itertuples()
    ]
    return merged.sort_values("usage", ascending=False).reset_index(drop=True)


def compare_predictability(
    pitcher: pd.DataFrame,
    league_mix: pd.DataFrame,
) -> pd.DataFrame:
    """A pitcher's top pitch per count against how often the league throws it.

    This is the comparison that rescues the 3-0 problem. "82% four-seams in
    3-0" reads as a finding until you know the league rate; against 65.8% it
    genuinely is one, and against 90% it would be the opposite.

    `pitcher` is the frame from metrics.counts.predictability().
    """
    frame = pitcher.rename_axis("count").reset_index()
    merged = frame.merge(
        league_mix[["count", "pitch_type", "league_share"]].rename(
            columns={"pitch_type": "top_pitch", "league_share": "league_top_share"}
        ),
        on=["count", "top_pitch"],
        how="left",
    )
    merged["share_delta"] = merged["top_share"] - merged["league_top_share"]
    merged["share_z"] = [
        proportion_z(row.top_share, row.n, row.league_top_share)
        for row in merged.itertuples()
    ]
    return merged.sort_values("share_delta", ascending=False, key=abs).reset_index(
        drop=True
    )


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
