"""Pitch sequencing: what follows what.

Pitches are not thrown independently. A slider after a fastball up is a
different pitch, competitively, than the same slider thrown cold — the hitter
has been given a reason to expect something else. Usage rates alone cannot see
this, because they treat every pitch as an isolated draw from a distribution.

This module models the dependence: transition probabilities between
consecutive pitches, the same transitions conditioned on what the previous
pitch did, and two-strike putaway behavior.

All pairing happens strictly within a plate appearance. The final pitch of one
PA never sets up the first pitch of the next.

Inputs are expected to have passed through `clean.py` and
`events.add_event_flags()`.
"""

from __future__ import annotations

import pandas as pd

# A plate appearance is uniquely identified by game and at-bat number.
PA_KEYS = ["game_pk", "at_bat_number"]


def transition_matrix(frame: pd.DataFrame, min_transitions: int = 1) -> pd.DataFrame:
    """P(next pitch type | current pitch type), within a plate appearance.

    Returns a DataFrame where the index is the current pitch, the columns are
    the next pitch, and each ROW sums to 1. Cell (FF, SL) reads: "given he
    just threw a four-seam, how often does a slider come next".

    Pairs are formed only within the same plate appearance, in `pitch_number`
    order rather than row order. The final pitch of a PA has no successor and
    contributes nothing. Rows whose total transition count falls below
    `min_transitions` are excluded, and missing combinations are 0.0, not NaN.

    The plate-appearance boundary is the critical constraint. A starter faces
    roughly 25 batters a game, so pairing across boundaries injects one
    fabricated transition per PA — a few percent of the sample, every one of
    them a spurious "he follows X with Y" claim. The resulting matrix looks
    entirely reasonable, which is what makes the error dangerous.
    """
    raise NotImplementedError


def transition_after_outcome(
    frame: pd.DataFrame,
    outcome: str,
    min_transitions: int = 1,
) -> pd.DataFrame:
    """Transition matrix conditioned on what the previous pitch did.

    Same output shape as transition_matrix(), but only counts pairs where the
    FIRST pitch of the pair had the given outcome.

    `outcome` is one of:
      "whiff"   the batter swung and missed
      "take"    the batter did not swing
      "foul"    the batter fouled it off
      "contact" the batter made contact (fair or foul)

    Derived from the boolean columns added by `events.add_event_flags()`,
    except "foul", which reads the raw `description` column.

    Conditioning captures intent, which is what makes this the more actionable
    view. "After a swinging strike on the slider, he goes back to it 61% of the
    time" is a pattern a hitter can sit on; the unconditional matrix averages
    that together with every other situation and washes it out.
    """
    raise NotImplementedError


def putaway(frame: pd.DataFrame, min_pitches: int = 10) -> pd.DataFrame:
    """Two-strike approach: what he goes to, and how often it finishes the job.

    Considers only pitches thrown with two strikes. Returns a DataFrame indexed
    by pitch_type, sorted by usage descending, with columns:

      n              int, two-strike pitches of this type
      usage          float, share of all two-strike pitches
      strikeouts     int, how many ended the PA in a strikeout
      putaway_rate   float, strikeouts / n
      whiff_rate     float, whiffs / swings on this pitch in two-strike counts

    A strikeout is a row where the `events` column is "strikeout" or
    "strikeout_double_play".

    Pitch types with fewer than `min_pitches` two-strike appearances are
    excluded; a putaway rate over four pitches is not worth printing.

    Usage and putaway_rate answer different questions and frequently disagree.
    A pitcher may go to his fastball most often with two strikes while his
    slider finishes a far higher share of the plate appearances it appears in.
    That gap is itself the exploitable finding, so both are reported.
    """
    raise NotImplementedError
