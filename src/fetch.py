"""Baseball Savant client.

Two responsibilities:
  1. Resolve a human-typed player name to an MLBAM id.
  2. Pull pitch-level rows for one pitcher-season from Savant.

Design note: we call the Savant CSV endpoint directly rather than relying
on pybaseball for the data pull. pybaseball is a thin wrapper over this same
endpoint, and owning the request means we control retries, timeouts, rate
limiting, and error messages. pybaseball is still used for name -> id
resolution, where it maintains a player register that is not worth rebuilding,
and is kept as a fallback data path in case Savant changes its query
parameters (which it has done before).
"""

from __future__ import annotations

import logging
import time
import unicodedata
from dataclasses import dataclass
from io import StringIO

import pandas as pd
import requests

from src import config

log = logging.getLogger(__name__)

SAVANT_CSV_URL = "https://baseballsavant.mlb.com/statcast_search/csv"

# Savant blocks requests without a plausible user agent. Identifying the
# project is the polite thing to do.
HEADERS = {
    "User-Agent": "statcast-advance/0.1 (personal analytics project)",
    "Accept": "text/csv,*/*",
}

# Timestamp of the last outbound request, used to enforce the configured
# delay between calls across the whole process.
_last_request_at: float = 0.0


class PlayerNotFound(Exception):
    """Raised when a name cannot be resolved to exactly one MLBAM id."""


class SavantError(Exception):
    """Raised when Savant cannot be reached or returns something unusable."""


@dataclass(frozen=True)
class Player:
    """A resolved player identity."""

    mlbam_id: int
    first_name: str
    last_name: str
    first_season: int | None = None
    last_season: int | None = None

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".title()


def _strip_accents(text: str) -> str:
    """Normalize names so 'Jose' matches 'José'."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def resolve_player(name: str) -> Player:
    """Turn a typed name into a single Player.

    Raises PlayerNotFound if there is no match, or if there are several and
    we cannot pick confidently. Ambiguity is surfaced rather than guessed at:
    silently scouting the wrong Luis Garcia would be worse than an error.
    """
    from pybaseball import playerid_lookup

    cleaned = _strip_accents(name.strip())
    parts = cleaned.split()
    if len(parts) < 2:
        raise PlayerNotFound(
            f"Need a first and last name, got {name!r}. Try 'Tarik Skubal'."
        )

    first, last = parts[0], " ".join(parts[1:])

    try:
        matches = playerid_lookup(last, first, fuzzy=False)
    except Exception as exc:  # noqa: BLE001 - pybaseball raises broadly
        raise PlayerNotFound(f"Player lookup failed for {name!r}: {exc}") from exc

    matches = matches[matches["key_mlbam"].notna()]

    if len(matches) == 0:
        raise PlayerNotFound(
            f"No player found matching {name!r}. Check the spelling, and note "
            f"that Savant uses given names (e.g. 'Michael King', not 'Mike King')."
        )

    if len(matches) > 1:
        # Prefer the most recently active player. Two players sharing a name
        # is common; one of them being active in the last decade usually is not.
        matches = matches.sort_values("mlb_played_last", ascending=False)
        top = matches.iloc[0]
        runner_up = matches.iloc[1]
        if pd.isna(top["mlb_played_last"]) or (
            top["mlb_played_last"] == runner_up["mlb_played_last"]
        ):
            options = ", ".join(
                f"{r.name_first} {r.name_last} "
                f"({int(r.mlb_played_first) if pd.notna(r.mlb_played_first) else '?'}"
                f"-{int(r.mlb_played_last) if pd.notna(r.mlb_played_last) else '?'}, "
                f"id={int(r.key_mlbam)})"
                for r in matches.head(5).itertuples()
            )
            raise PlayerNotFound(
                f"{name!r} is ambiguous. Candidates: {options}. "
                f"Pass --player-id to disambiguate."
            )
        row = top
    else:
        row = matches.iloc[0]

    return Player(
        mlbam_id=int(row["key_mlbam"]),
        first_name=str(row["name_first"]),
        last_name=str(row["name_last"]),
        first_season=int(row["mlb_played_first"]) if pd.notna(row["mlb_played_first"]) else None,
        last_season=int(row["mlb_played_last"]) if pd.notna(row["mlb_played_last"]) else None,
    )


def _respect_rate_limit() -> None:
    """Sleep if the previous request was too recent.

    Savant is a free public endpoint with no auth. Rate limiting ourselves is
    the cost of keeping it that way.
    """
    global _last_request_at
    min_gap = float(config.get("data.seconds_between_requests", 3))
    elapsed = time.time() - _last_request_at
    if elapsed < min_gap:
        time.sleep(min_gap - elapsed)
    _last_request_at = time.time()


def _build_params(mlbam_id: int, season: int, player_type: str) -> dict[str, str]:
    """Assemble the Savant search query.

    The endpoint expects the full set of filter parameters even when most are
    empty; omitting them can silently change the result set. Regular season and
    all postseason rounds are requested here, and narrowed later in clean.py.
    """
    lookup_key = "pitchers_lookup[]" if player_type == "pitcher" else "batters_lookup[]"
    return {
        "all": "true",
        "hfPT": "",
        "hfAB": "",
        "hfGT": "R|PO|",
        "hfPR": "",
        "hfZ": "",
        "hfStadium": "",
        "hfBBL": "",
        "hfNewZones": "",
        "hfPull": "",
        "hfC": "",
        "hfSea": f"{season}|",
        "hfSit": "",
        "player_type": player_type,
        "hfOuts": "",
        "hfOpponent": "",
        "pitcher_throws": "",
        "batter_stands": "",
        "hfSA": "",
        "game_date_gt": "",
        "game_date_lt": "",
        "hfMo": "",
        "hfTeam": "",
        "home_road": "",
        lookup_key: str(mlbam_id),
        "hfRO": "",
        "position": "",
        "hfInfield": "",
        "hfOutfield": "",
        "hfInn": "",
        "hfBBT": "",
        "min_pitches": "0",
        "min_results": "0",
        "group_by": "name",
        "sort_col": "pitches",
        "player_event_sort": "api_p_release_speed",
        "sort_order": "desc",
        "min_pas": "0",
        "type": "details",
    }


def fetch_player_season(
    mlbam_id: int,
    season: int,
    player_type: str = "pitcher",
) -> pd.DataFrame:
    """Download every pitch for one player-season.

    Returns the raw Savant frame, uncleaned. Filtering and type normalization
    are deliberately left to clean.py so that exactly what the API returned is
    what lands in the cache.
    """
    if player_type not in {"pitcher", "batter"}:
        raise ValueError(f"player_type must be 'pitcher' or 'batter', got {player_type!r}")

    params = _build_params(mlbam_id, season, player_type)
    timeout = int(config.get("data.request_timeout_seconds", 120))
    max_retries = int(config.get("data.max_retries", 3))

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        _respect_rate_limit()
        try:
            log.info(
                "Fetching %s %s season %s from Savant (attempt %s/%s)",
                player_type, mlbam_id, season, attempt, max_retries,
            )
            response = requests.get(
                SAVANT_CSV_URL, params=params, headers=HEADERS, timeout=timeout
            )
            response.raise_for_status()

            text = response.text
            # Savant returns a 200 with an HTML error page under load.
            if text.lstrip().startswith("<"):
                raise SavantError("Savant returned HTML instead of CSV (likely overloaded)")

            frame = pd.read_csv(StringIO(text), low_memory=False)
            if frame.empty:
                log.warning("Savant returned 0 rows for %s in %s", mlbam_id, season)
                return frame

            log.info("Retrieved %s pitches", len(frame))
            return frame

        except Exception as exc:  # noqa: BLE001 - retry on anything transient
            last_error = exc
            if attempt < max_retries:
                backoff = 2 ** attempt
                log.warning("Attempt %s failed (%s). Retrying in %ss.", attempt, exc, backoff)
                time.sleep(backoff)

    # Direct endpoint exhausted its retries. Fall back to pybaseball, which
    # may be tracking a parameter change we have not caught up with yet.
    log.warning("Direct Savant fetch failed; falling back to pybaseball.")
    try:
        return _fetch_via_pybaseball(mlbam_id, season, player_type)
    except Exception as exc:  # noqa: BLE001
        raise SavantError(
            f"Could not retrieve data for {player_type} {mlbam_id}, season {season}. "
            f"Direct endpoint error: {last_error}. Fallback error: {exc}"
        ) from exc


def fetch_date_range(start_date: str, end_date: str) -> pd.DataFrame:
    """Download every pitch thrown league-wide between two dates, inclusive.

    Dates are ISO strings, "YYYY-MM-DD". Used to assemble league baselines by
    sampling days rather than pulling a whole season, which would be hundreds
    of thousands of rows and a great many requests.

    A single day is roughly 4,400 pitches and takes about ten seconds, so keep
    ranges short and let the caller loop with the rate limiter in between.
    """
    params = {
        "all": "true",
        "hfGT": "R|PO|",
        "game_date_gt": start_date,
        "game_date_lt": end_date,
        "type": "details",
        "min_pitches": "0",
        "min_results": "0",
        "group_by": "name",
        "sort_col": "pitches",
        "sort_order": "desc",
    }

    timeout = int(config.get("data.request_timeout_seconds", 120))
    max_retries = int(config.get("data.max_retries", 3))
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        _respect_rate_limit()
        try:
            log.info("Fetching league pitches %s..%s (attempt %s/%s)",
                     start_date, end_date, attempt, max_retries)
            response = requests.get(
                SAVANT_CSV_URL, params=params, headers=HEADERS, timeout=timeout
            )
            response.raise_for_status()

            text = response.text
            if text.lstrip().startswith("<"):
                raise SavantError("Savant returned HTML instead of CSV (likely overloaded)")

            return pd.read_csv(StringIO(text), low_memory=False)

        except Exception as exc:  # noqa: BLE001 - retry on anything transient
            last_error = exc
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    raise SavantError(
        f"Could not retrieve league pitches for {start_date}..{end_date}: {last_error}"
    )


def _fetch_via_pybaseball(mlbam_id: int, season: int, player_type: str) -> pd.DataFrame:
    """Fallback path using pybaseball's own Savant wrapper."""
    from pybaseball import statcast_batter, statcast_pitcher

    start, end = f"{season}-01-01", f"{season}-12-31"
    if player_type == "pitcher":
        return statcast_pitcher(start, end, mlbam_id)
    return statcast_batter(start, end, mlbam_id)
