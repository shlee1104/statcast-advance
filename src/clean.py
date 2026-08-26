"""Cleaning and normalization of raw Savant data.

Sits between the API client and the database. Savant returns 119 loosely typed
columns including deprecated pitch classifications, non-competitive pitches,
spring training games, and the occasional impossible count. This module decides
what counts as a pitch worth analyzing.

Those decisions are not cosmetic. Every rate statistic in the final report is a
fraction, and this module determines the denominator. A report stating "throws
his fastball 52% of the time" is wrong if four intentional balls are sitting in
the sample, and wrong in a way no downstream check would catch.

Decisions that are genuinely debatable are marked JUDGMENT CALL below, with the
reasoning for the choice made.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.store import PITCH_SCHEMA

# ---------------------------------------------------------------------------
# Reference tables
# ---------------------------------------------------------------------------

# Savant's raw pitch_type codes, and what we fold them into.
#
# Rationale for each:
#   FT -> SI  MLB retired the two-seam classification in 2020 and folded it
#             into sinker. Data spanning that change needs them unified or
#             the same pitch appears as two different offerings.
#   FA -> FF  "FA" is a generic unclassified fastball, almost always a
#             four-seam.
#   CS -> CU  "Slow curve" is a curveball. Separating it fragments an
#             already small sample.
#   KC -> CU  JUDGMENT CALL. Knuckle curve is mechanically distinct, but few
#             pitchers throw both it and a standard curve, so folding avoids
#             splitting one pitcher's breaking ball into two thin buckets.
#   SV -> SL  JUDGMENT CALL. Slurve sits between slider and curve; folding
#             into slider matches how most pitchers who throw it use it.
#   ST        JUDGMENT CALL - kept SEPARATE. The sweeper is a genuinely
#             different pitch from a slider in movement and usage, and it is
#             central to modern scouting. Folding it into SL would erase the
#             single most important arsenal change of the last five years.
#   FO -> FS  Forkball and splitter are the same family and rarely coexist.
PITCH_TYPE_MAP: dict[str, str] = {
    "FT": "SI",
    "FA": "FF",
    "CS": "CU",
    "KC": "CU",
    "SV": "SL",
    "FO": "FS",
}

# Pitch codes that are not competitive pitches and must be removed entirely.
#   PO / IN  pitchout and intentional ball - thrown with no intent to get an
#            out, so including them distorts usage rates and location maps.
#   AB / AS  automatic ball/strike (pitch timer violations) - no pitch thrown.
#   UN       unknown classification.
NON_COMPETITIVE_PITCH_CODES: frozenset[str] = frozenset({"PO", "IN", "AB", "AS", "UN"})

# Descriptions marking a pitch as non-competitive even when pitch_type looks
# ordinary. Savant sometimes classifies an intentional ball as a normal
# fastball, so filtering on description as well as pitch_type catches both.
NON_COMPETITIVE_DESCRIPTIONS: frozenset[str] = frozenset(
    {"intent_ball", "pitchout", "automatic_ball", "automatic_strike"}
)

# Human-readable names for the consolidated codes, used in report labels.
PITCH_DISPLAY_NAMES: dict[str, str] = {
    "FF": "Four-Seam",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "ST": "Sweeper",
    "CU": "Curveball",
    "CH": "Changeup",
    "FS": "Splitter",
    "KN": "Knuckleball",
    "EP": "Eephus",
    "SC": "Screwball",
}


# ---------------------------------------------------------------------------
# Functions to implement
# ---------------------------------------------------------------------------


def consolidate_pitch_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Fold deprecated and near-duplicate pitch codes together.

    Apply PITCH_TYPE_MAP to the `pitch_type` column. Codes not in the map are
    left alone. Add a `pitch_display` column with the readable name from
    PITCH_DISPLAY_NAMES, falling back to the raw code when unmapped.

    Rows where `pitch_type` is null should keep a null pitch_type here; the
    dropping happens in drop_non_competitive().

    Must not mutate the input frame. Return a new one.
    """
    result = frame.copy()

    result['pitch_type'] = result['pitch_type'].replace(PITCH_TYPE_MAP)

    result['pitch_display'] = result['pitch_type'].map(
        lambda code: PITCH_DISPLAY_NAMES.get(code, code)
    )
    
    return result                 


def drop_non_competitive(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove pitches that should not count toward any rate statistic.

    Drop a row if any of these hold:
      - `pitch_type` is null or empty
      - `pitch_type` is in NON_COMPETITIVE_PITCH_CODES
      - `description` is in NON_COMPETITIVE_DESCRIPTIONS

    Why this matters: a scouting report saying "throws his fastball 52% of the
    time" is wrong if four intentional balls are sitting in the denominator.

    Must not mutate the input frame.
    """

    result = frame.copy()
    
    bad = (
        result['pitch_type'].isna() 
        | (result['pitch_type'] == '')  
        | result['pitch_type'].isin(NON_COMPETITIVE_PITCH_CODES) 
        | result['description'].isin(NON_COMPETITIVE_DESCRIPTIONS)
    )
    return result[~bad].reset_index(drop=True)


def filter_game_types(frame: pd.DataFrame, game_types: list[str] | None = None) -> pd.DataFrame:
    """Keep only the game types configured in config.yaml.

    When `game_types` is None, read `data.game_types` from config, which
    defaults to regular season plus all postseason rounds and excludes spring
    training ("S") and exhibitions ("E").

    Spring training is excluded because pitchers work on pitches they will not
    throw in a real game, which is exactly the kind of noise that produces a
    confident, wrong scouting report.

    Must not mutate the input frame.
    """

    if game_types is None:
        game_types = config.get('data.game_types')

    result = frame.copy()

    return result[result['game_type'].isin(game_types)].reset_index(drop=True)


def add_count_state(frame: pd.DataFrame) -> pd.DataFrame:
    """Add columns describing the count, the backbone of the whole analysis.

    Add:
      count       string "balls-strikes", e.g. "0-0", "3-2"
      is_ahead    bool, pitcher ahead: strikes > balls
      is_behind   bool, pitcher behind: balls > strikes
      is_two_strike  bool, strikes == 2
      is_first_pitch bool, balls == 0 and strikes == 0

    Drop any row with an impossible count (balls outside 0-3 or strikes
    outside 0-2). These appear occasionally from data errors, and a "4-2
    count" bucket in the report would be embarrassing.

    Must not mutate the input frame.
    """

    result = frame.copy()

    valid = result['balls'].between(0, 3) & result['strikes'].between(0, 2)
    result = result[valid].reset_index(drop=True)

    result['count'] = (
        result['balls'].astype(int).astype(str)
        + "-"
        + result['strikes'].astype(int).astype(str)
    )

    result['is_ahead'] = (
        (result['strikes'] > result['balls'])
    )

    result['is_behind'] = (
        (result['balls'] > result['strikes'])
    )

    result['is_two_strike'] = (
        (result['strikes'] == 2)
    )

    result['is_first_pitch'] = (
        (result['balls'] == 0) & (result['strikes'] == 0)
    )

    return result


def prepare_for_storage(frame: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline and shape the frame for store.py.

    Order:
      1. consolidate_pitch_types
      2. drop_non_competitive
      3. filter_game_types
      4. add_count_state
      5. coerce `game_date` to datetime
      6. select exactly the columns in store.PITCH_SCHEMA, in that order,
         creating any missing ones as null

    Step 6 exists because store.save_pitches() requires the frame to match its
    schema exactly. Import PITCH_SCHEMA from src.store.

    Return a frame ready to hand straight to store.save_pitches().
    """

    result = consolidate_pitch_types(frame)
    result = drop_non_competitive(result)
    result = filter_game_types(result)
    result = add_count_state(result)

    result['game_date'] = pd.to_datetime(result['game_date'])

    return result.reindex(columns=list(PITCH_SCHEMA.keys()))
