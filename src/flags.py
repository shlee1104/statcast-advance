"""Weakness flags: the layer that turns tables into findings.

Everything below this module measures. This module decides what is worth
telling a coach, which is a different and harder question. A table of twelve
counts is not a finding; "he throws the splitter 48% of the time in 0-2,
against a league rate of 24%, on 61 pitches" is.

Three rules govern every flag here.

**Every claim carries its sample size.** Not in a footnote — in the sentence.
A coach who reads a tendency and acts on it is entitled to know whether it
rests on 300 pitches or 9, and a report that hides the denominator is worse
than no report because it converts noise into instructions.

**Nothing fires below its gate.** The minimum sample for each flag lives in
`config.yaml` under `flags.min_n`, and a finding that cannot clear its gate is
discarded rather than reported with a caveat. Caveats do not survive the trip
from a PDF to a dugout.

**Severity is ranked on one axis, and the axis is named.** Findings arrive in
incompatible units. Where a league proportion exists, severity is standard
errors from it, which is a measurement. Where none exists — fatigue,
sequencing lift, location lift — severity is how far past its own threshold
the finding sits, scaled by a weight in `report.severity_weights`, which is a
judgment call. Every finding records which of the two produced its number in
`severity_basis`, so a weighted guess is never mistaken for a z-score.

League comparison is optional throughout. Passing `league=None` still produces
findings from the rules that need no baseline, so the engine works before
`scripts/build_baselines.py` has ever been run.
"""

from __future__ import annotations

import pandas as pd

from src import baselines, config
from src.gameplan import PITCH_FAMILY
from src.gameplan import name as pitch_name
from src.metrics import arsenal, counts, location, sequencing, splits

# Height bands as a hitter says them, for "after a low slider" rather than
# "after a SL down".
HEIGHT_WORDS: dict[str, str] = {
    "UP": "high", "MID": "belt-high", "MIDDLE": "belt-high", "DOWN": "low",
}

# Where a pitch is headed, as the plan says it.
WHERE_WORDS: dict[str, str] = {
    "UP": "up in the zone", "MIDDLE": "at the belt", "MID": "at the belt", "DOWN": "down",
}

# Findings carry these columns in this order, whatever rule produced them.
FINDING_COLUMNS: list[str] = [
    "flag", "severity", "severity_basis", "n", "runs_cost", "resolved",
    "claim", "observed", "reference", "delta", "runs_lo", "runs_hi", "rv_p",
]

# The two ways a severity number can come to exist. Recorded per finding
# rather than inferred from the flag type, so the distinction survives into
# the report.
BASIS_Z = "z_vs_league"
BASIS_THRESHOLD = "threshold_multiple"


def _finding(
    flag: str,
    severity: float,
    basis: str,
    n: int,
    claim: str,
    observed: float,
    reference: float | None = None,
    runs_cost: float | None = None,
    runs_lo: float | None = None,
    runs_hi: float | None = None,
    rv_p: float | None = None,
) -> dict:
    """Assemble one finding row.

    `runs_cost` is the estimated runs the pattern costs him over the season,
    where it can be computed. It is the only column in this table that is
    comparable across flag types, which is why it is carried even for the
    rules that cannot produce it.
    """
    return {
        "flag": flag,
        "severity": float(severity),
        "severity_basis": basis,
        "n": int(n),
        "claim": claim,
        "observed": float(observed),
        "reference": float(reference) if reference is not None else float("nan"),
        "delta": (
            float(observed - reference) if reference is not None else float("nan")
        ),
        "runs_cost": (
            float(runs_cost) if runs_cost is not None else float("nan")
        ),
        "runs_lo": float(runs_lo) if runs_lo is not None else float("nan"),
        "runs_hi": float(runs_hi) if runs_hi is not None else float("nan"),
        # Uncorrected, and never read directly. evaluate() corrects across the
        # whole family and writes `resolved`.
        "rv_p": float(rv_p) if rv_p is not None else float("nan"),
        "resolved": False,
    }


def _consequence(
    runs_cost: float,
    min_runs: float,
    significant: bool = False,
    runs_lo: float = float("nan"),
    runs_hi: float = float("nan"),
) -> tuple[bool, str]:
    """Whether a tell has a demonstrable run consequence, and how to say it.

    Returns (is_exploitable, phrase).

    The significance test is what stops this from overclaiming, and it fails
    almost always. Run value per pitch has a standard deviation near 0.19
    while the differences here are 0.01-0.02, so a season of one pitcher
    cannot resolve them: the up-fastball tell's "3.4 runs" carries a 95%
    interval of about [-4, +11].

    So the honest report says the frequency pattern is real and its run
    consequence is unresolved, rather than printing a run figure to one
    decimal and letting a reader treat it as measured. A pattern is only
    called costly or working when the interval excludes zero.

    A point estimate is still shown, with its interval, because the direction
    is weak evidence and worth having — it just cannot carry a conclusion.
    """
    if pd.isna(runs_cost):
        return True, ""

    interval = ""
    if runs_lo == runs_lo and runs_hi == runs_hi:
        interval = f" [95% CI {runs_lo:+.0f} to {runs_hi:+.0f}]"

    if not significant:
        return True, (
            f" Run consequence is unresolved at this sample: the point "
            f"estimate is {runs_cost:+.1f} runs{interval}, which does not "
            f"exclude zero. Treat the frequency pattern as the finding."
        )

    if runs_cost >= min_runs:
        return True, (
            f" It costs him about {runs_cost:.1f} runs over this sample"
            f"{interval}."
        )
    if runs_cost <= -min_runs:
        return False, (
            f" But it is working: the pattern saves him about "
            f"{abs(runs_cost):.1f} runs{interval}, so this is a tendency "
            f"rather than a weakness."
        )
    return False, (
        f" The pattern has no run consequence either way ({runs_cost:+.1f}"
        f"{interval})."
    )


def benjamini_hochberg(p_values: list[float], q: float = 0.05) -> list[bool]:
    """Which comparisons survive a false-discovery-rate correction.

    Sort the p-values, find the largest rank k where p(k) <= q * k / m, and
    reject everything up to it. Missing p-values never survive.

    This has to happen where the whole family is visible, which is why it lives
    in the engine rather than in the metric modules. Each module sees its own
    rows; only `evaluate()` sees that a report runs 120 comparisons across
    counts, locations and sequences before picking five things to say.

    The correction matters more here than the raw threshold ever did. On
    Yamamoto, 9 comparisons clear an uncorrected p < .05 — against 6 expected
    from pure noise at that volume — and **none** survive at q = .05. A report
    that printed "9 significant" would be reporting coin flips with a decimal
    point on them.

    Benjamini-Hochberg rather than Bonferroni because these hypotheses are
    heavily dependent — the same pitch appears in a dozen counts — and
    controlling the false-discovery rate is the right question for a screening
    exercise anyway: of the things I am about to tell a coach, what share are
    likely to be nothing.
    """
    indexed = [
        (i, p) for i, p in enumerate(p_values)
        if p is not None and p == p
    ]
    if not indexed:
        return [False] * len(p_values)

    m = len(indexed)
    indexed.sort(key=lambda pair: pair[1])

    cutoff_rank = 0
    for rank, (_, p) in enumerate(indexed, start=1):
        if p <= q * rank / m:
            cutoff_rank = rank

    survivors = {i for i, _ in indexed[:cutoff_rank]}
    return [i in survivors for i in range(len(p_values))]


def _threshold_severity(
    observed: float,
    threshold: float,
    weight: float,
    n: int,
    gate_n: int,
) -> float:
    """Scale a finding with no league proportion onto the severity axis.

        severity = weight * (observed / threshold) * sqrt(n / gate_n)

    Two factors, and the second is the important one. Effect size alone ranks
    a 2.2x lift measured over 27 pitches above a 1.8x lift measured over 393,
    which inverts the order a scouting report needs: the whole premise of this
    project is that a moderate tendency you can trust beats a dramatic one you
    cannot. The sample term fixes that, and it is a square root because that is
    how a standard error scales with n — which keeps this axis dimensionally
    analogous to the z-scores it is being ranked alongside, rather than
    arbitrarily chosen to look similar.

    Both factors are ratios against the flag's own configured bar, so a finding
    sitting exactly at its threshold with exactly its minimum sample scores the
    flag's weight, and everything better scores higher. Nothing that fired can
    score zero or negative, which was a bug in the first version of this
    function: a familiarity finding scored on how little velocity had dropped
    ranked below every other finding precisely because it was the clean case.

    The choice of weight is still a judgment call, and `severity_basis` records
    that this path produced the number.
    """
    if threshold == 0 or gate_n <= 0 or n <= 0:
        return float("nan")
    return float(weight * (observed / threshold) * (n / gate_n) ** 0.5)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def predictable_counts(frame: pd.DataFrame, league_mix: pd.DataFrame | None) -> list[dict]:
    """Counts where he changes his mind, measured against his own overall mix.

    Works with or without league data. The comparison that fires this flag is
    the pitcher against himself, because that is the question a hitter is
    asking: not "is this unusual for a major leaguer" but "is he about to do
    something different from what he normally does".

    This replaced a league-relative version, and the failure is worth keeping
    on the record. Against league rates, Yamamoto's splitter runs 5-7x in every
    count he uses it in, so the report produced five findings that were one
    sentence with the count swapped — "he throws a splitter", which any coach
    already knew. Against his own rate the splitter compresses to 1.2-1.9x, and
    a cutter appears at 2.4x in 2-1 and 3-1 that he then abandons with two
    strikes. That is a pattern a hitter can use, and the league comparison
    could never have surfaced it, because his overall cutter usage is only
    1.5x league.

    Findings are deduplicated by PITCH, not by count. A pitch elevated in four
    related counts is one tendency, and reporting it four times crowds out
    three other pitches. The best count survives and the others are counted in
    the claim.

    League rates, when supplied, are reported as context inside the sentence —
    they say who this pitcher is — but they do not decide what fires or how it
    ranks.
    """
    min_n = int(config.get("flags.min_n.predictable_count", 20))
    min_lift = float(
        config.get("flags.thresholds.predictable_count_own_lift", 1.50)
    )
    min_runs = float(config.get("flags.thresholds.min_runs_cost", 1.0))
    weight = float(config.get("report.severity_weights.predictable_count", 2.0))

    table = counts.count_lifts(frame, min_n=min_n)
    if len(table) == 0:
        return []

    qualifying = table[table["lift"] >= min_lift]
    if len(qualifying) == 0:
        return []

    league_lookup = {}
    if league_mix is not None:
        for row in league_mix.itertuples():
            league_lookup[(row.count, row.pitch_type)] = row.league_share

    findings = []
    # Sorted by score already, so the first row per pitch is its best count.
    for pitch_type, group in qualifying.groupby("pitch_type", sort=False):
        best = group.iloc[0]
        others = len(group) - 1

        also = ""
        if others:
            extra = ", ".join(group["count"].iloc[1:4])
            # "Leans on", not "elevated": in baseball an elevated pitch is a
            # high one, and this is about how often, not where.
            also = f" He also leans on it in {extra}."

        league_share = league_lookup.get((best["count"], pitch_type))
        context = ""
        if league_share:
            context = (f" League rate in {best['count']} is "
                       f"{league_share:.0%}.")

        severity = _threshold_severity(
            best["lift"], min_lift, weight, int(best["n"]), min_n
        )

        findings.append(_finding(
            flag="predictable_count",
            severity=severity,
            basis=BASIS_THRESHOLD,
            n=int(best["n"]),
            runs_cost=best["runs_cost"],
            runs_lo=best.get("runs_lo"),
            runs_hi=best.get("runs_hi"),
            rv_p=best.get("rv_p"),
            claim=(
                f"In {best['count']} he goes to the {pitch_name(pitch_type)} "
                f"{best['share']:.0%} of the time against his own "
                f"{best['own_rate']:.0%} overall — {best['lift']:.1f}x "
                f"(n={int(best['n'])} of {int(best['count_n'])} pitches in that "
                f"count).{also}{context}"
            ),
            observed=float(best["share"]),
            reference=float(best["own_rate"]),
        ))

    return findings


def _count_or_zero(value) -> float:
    return 0.0 if value is None or pd.isna(value) else float(value)


def handedness_gaps(frame: pd.DataFrame) -> list[dict]:
    """Pitches shown to one side far more than the other.

    Needs no league data: the comparison is the pitcher against himself, which
    is what makes this rule work before any baseline exists.
    """
    min_n = int(config.get("flags.min_n.handedness_gap", 40))
    min_gap = float(config.get("flags.thresholds.handedness_usage_gap", 0.20))

    table = splits.platoon_gaps(frame)
    if len(table) == 0:
        return []

    findings = []
    for row in table.itertuples():
        gap = row.gap
        if pd.isna(gap) or abs(gap) < min_gap:
            continue

        favored, starved = ("lefties", "righties") if gap > 0 else ("righties", "lefties")
        # A pitch never thrown to a side has no usage there, not an unknown
        # one: it is 0%. Left as NaN it printed "only nan% to lefties".
        usage_L, usage_R = _count_or_zero(row.usage_L), _count_or_zero(row.usage_R)
        high, low = (usage_L, usage_R) if gap > 0 else (usage_R, usage_L)
        # A pitch never thrown to one side has a missing count there, and
        # `NaN or 0` is NaN — NaN is truthy — so missing is replaced explicitly.
        n_side = int(max(_count_or_zero(row.n_L), _count_or_zero(row.n_R)))
        if n_side < min_n:
            continue

        findings.append(_finding(
            flag="handedness_gap",
            severity=_threshold_severity(
                abs(gap), min_gap,
                float(config.get("report.severity_weights.handedness_gap", 2.0)),
                n_side, min_n,
            ),
            basis=BASIS_THRESHOLD,
            n=n_side,
            claim=(
                f"He throws the {pitch_name(row.pitch_type)} {high:.0%} of the "
                f"time to {favored} but only {low:.0%} to {starved} — "
                f"{abs(high - low) * 100:.0f} points more (n={n_side} "
                f"{pitch_name(row.pitch_type)}s to {favored})."
            ),
            observed=high,
            reference=low,
        ))

    return findings


def hittable_pitches(
    frame: pd.DataFrame,
    league_outcomes: pd.DataFrame | None,
) -> list[dict]:
    """Offerings hitters do more damage against than the league's version.

    Requires league data, since "an xwOBA of .350 on his slider" means nothing
    without knowing what sliders generally allow.
    """
    min_n = int(config.get("flags.min_n.hittable_pitch", 50))
    min_delta = float(config.get("flags.thresholds.hittable_pitch_xwoba_delta", 0.060))

    if league_outcomes is None:
        return []

    profile = arsenal.profile(frame)
    if len(profile) == 0:
        return []

    compared = baselines.compare_profile(profile, league_outcomes)

    findings = []
    for row in compared.itertuples():
        if row.n < min_n or pd.isna(row.xwoba_delta):
            continue
        if row.xwoba_delta < min_delta:
            continue

        # xwOBA is a mean, not a proportion, so proportion_z does not apply.
        # Scaled against its own threshold instead, and labelled as such.
        findings.append(_finding(
            flag="hittable_pitch",
            severity=_threshold_severity(
                row.xwoba_delta, min_delta,
                float(config.get("report.severity_weights.hittable_pitch", 2.0)),
                int(row.n), min_n,
            ),
            basis=BASIS_THRESHOLD,
            n=int(row.n),
            claim=(
                f"Hitters post a {row.xwoba:.3f} xwOBA against his "
                f"{pitch_name(row.pitch_type)}, {row.xwoba_delta:+.3f} above the league's "
                f"{row.league_xwoba:.3f} for that pitch, and he throws it "
                f"{row.usage:.0%} of the time (n={int(row.n)})."
            ),
            observed=row.xwoba,
            reference=row.league_xwoba,
        ))

    return findings


def fatigue_decline(frame: pd.DataFrame) -> list[dict]:
    """Velocity loss late in outings, and what it is attributable to.

    Two findings can come out of this: the raw decline, and the
    times-through-order attribution, which points in opposite tactical
    directions depending on whether velocity holds.
    """
    min_outings = int(config.get("flags.min_n.fatigue_outings", 10))
    min_drop = float(config.get("flags.thresholds.fatigue_velo_drop_mph", 1.0))
    min_tto_n = int(config.get("flags.min_n.times_through_order", 150))
    xwoba_rise = float(config.get("flags.thresholds.tto_xwoba_rise", 0.020))
    weight = float(config.get("report.severity_weights.fatigue", 2.0))

    findings = []

    table = splits.fatigue(frame)
    if len(table) > 0:
        # Only buckets backed by enough separate outings can support a claim;
        # a 2 mph drop seen in two starts is two starts, not a pattern.
        trusted = table[table["outings"] >= min_outings]
        if len(trusted) > 0:
            worst = trusted.loc[trusted["velo_delta"].idxmin()]
            drop = -float(worst["velo_delta"])
            if drop >= min_drop:
                findings.append(_finding(
                    flag="fatigue",
                    # Sample size here is outings, not pitches: the claim is
                    # about a pattern across starts, and thirty pitches drawn
                    # from three of them is three data points.
                    severity=_threshold_severity(
                        drop, min_drop, weight,
                        int(worst["outings"]), min_outings,
                    ),
                    basis=BASIS_THRESHOLD,
                    n=int(worst["n"]),
                    claim=(
                        f"Fastball velocity falls {drop:.1f} mph by pitch "
                        f"{worst['bucket']} of an outing, measured across "
                        f"{int(worst['outings'])} starts (n={int(worst['n'])} "
                        f"pitches)."
                    ),
                    observed=float(worst["velo"]),
                    reference=float(table.iloc[0]["velo"]),
                ))

    decomposition = splits.decompose_tto(frame)
    if decomposition["attribution"] in ("velocity_declines", "results_decay", "both"):
        tto = splits.times_through_order(frame)
        third = tto[tto["tto"] == 3]
        n_third = int(third.iloc[0]["n"]) if len(third) else 0

        # Both trips must clear the outings gate. A third trip drawn from a
        # handful of starts is the same survivorship artifact as a late
        # fatigue bucket.
        if n_third >= min_tto_n and decomposition["reliable"]:
            velo_delta = decomposition["velo_delta"]
            xwoba_delta = decomposition["xwoba_delta"]

            # Score whichever signal actually cleared its threshold. The first
            # version scored the magnitude of the velocity change in every
            # case, which ranked a results-only finding lowest of anything in
            # the report, exactly because velocity had held steady.
            if decomposition["attribution"] == "results_decay":
                observed, threshold = xwoba_delta, xwoba_rise
            else:
                observed, threshold = -velo_delta, min_drop

            findings.append(_finding(
                flag="times_through_order",
                severity=_threshold_severity(
                    observed, threshold, weight, n_third, min_tto_n
                ),
                basis=BASIS_THRESHOLD,
                n=n_third,
                # The caveat travels with the claim rather than sitting in a
                # footnote, because the claim is the part that gets quoted.
                claim=(
                    f"{decomposition['note'].rstrip('.')} (n={n_third} pitches). "
                    f"{decomposition['caveat']}"
                ),
                observed=velo_delta,
            ))

    return findings


def sequencing_tells(frame: pd.DataFrame) -> list[dict]:
    """Setup pitches that make the next pitch predictable without helping.

    The most exploitable finding available, and invisible to any metric that
    merges frequency with effectiveness. A sequence the pitcher repeats that
    buys him nothing is a sequence a hitter can sit on for free.
    """
    min_n = int(config.get("flags.min_n.sequencing_tell", 25))
    min_prob = float(config.get("flags.thresholds.sequencing_tell_prob", 0.65))
    min_lift = float(config.get("flags.thresholds.sequencing_tell_lift", 1.50))
    min_runs = float(config.get("flags.thresholds.min_runs_cost", 1.0))
    weight = float(config.get("report.severity_weights.sequencing_tell", 2.0))

    table = sequencing.setup_pairs(frame, min_n=min_n)
    if len(table) == 0:
        return []

    findings = []
    for row in table.itertuples():
        if row.n < min_n or pd.isna(row.freq_lift):
            continue
        # Either route qualifies: an outright majority follow-up, or one made
        # substantially more likely than baseline without reaching a majority.
        if row.p_next < min_prob and row.freq_lift < min_lift:
            continue

        severity = _threshold_severity(
            row.freq_lift, min_lift, weight, int(row.n), min_n
        )

        findings.append(_finding(
            flag="sequencing_tell",
            severity=severity,
            basis=BASIS_THRESHOLD,
            n=int(row.n),
            claim=(
                f"After a {HEIGHT_WORDS.get(row.setup_band, row.setup_band.lower())} "
                f"{pitch_name(row.setup_pitch)}, the next pitch is "
                f"{'another' if row.next_pitch == row.setup_pitch else 'a'} "
                f"{pitch_name(row.next_pitch)} {row.p_next:.0%} of the time "
                f"against a baseline of {row.baseline_p:.0%} — "
                f"{row.freq_lift:.2f}x (n={int(row.n)})."
            ),
            observed=row.p_next,
            reference=row.baseline_p,
            runs_cost=row.runs_cost,
            runs_lo=getattr(row, "runs_lo", None),
            runs_hi=getattr(row, "runs_hi", None),
            rv_p=getattr(row, "rv_p", None),
        ))

    return findings


def location_tells(frame: pd.DataFrame) -> list[dict]:
    """Location bands that narrow down which pitch is coming.

    Scored on lift rather than on the conditional-entropy summary, because the
    summary came out modest for both pitchers tested while the per-band lifts
    were large. See the PROVISIONAL note on `location_tell_info` in config.
    """
    min_n = int(config.get("flags.min_n.location_tell", 40))
    min_lift = float(config.get("flags.thresholds.location_tell_lift", 1.35))
    min_share = float(config.get("flags.thresholds.location_tell_share", 0.45))
    min_runs = float(config.get("flags.thresholds.min_runs_cost", 1.0))
    weight = float(config.get("report.severity_weights.location_tell", 2.0))

    table = location.location_tells(frame, min_n=min_n)
    if len(table) == 0:
        return []

    findings = []
    for row in table.itertuples():
        if row.n < min_n or pd.isna(row.lift) or row.lift < min_lift:
            continue
        # Lift and absolute share must both clear. A large lift onto a 37%
        # share is a real effect that no hitter can act on.
        if row.p_pitch < min_share:
            continue
        # A fastball up is what nearly every pitcher does; it ranked as the
        # top finding in 28 of 32 reports while telling a hitter nothing. The
        # plan's height line leaves it out for the same reason.
        if row.band == "UP" and PITCH_FAMILY.get(row.pitch_type) == "hard":
            continue

        where = WHERE_WORDS.get(row.band, row.band.lower())
        severity = _threshold_severity(
            row.lift, min_lift, weight, int(row.n), min_n
        )

        findings.append(_finding(
            flag="location_tell",
            severity=severity,
            basis=BASIS_THRESHOLD,
            n=int(row.n),
            claim=(
                f"When the pitch is headed {where}, it is the "
                f"{pitch_name(row.pitch_type)} {row.p_pitch:.0%} of the time, against his "
                f"overall {row.baseline_p:.0%} — {row.lift:.2f}x "
                f"(n={int(row.n)})."
            ),
            observed=row.p_pitch,
            reference=row.baseline_p,
            runs_cost=row.runs_cost,
            runs_lo=getattr(row, "runs_lo", None),
            runs_hi=getattr(row, "runs_hi", None),
            rv_p=getattr(row, "rv_p", None),
        ))

    return findings


def first_pitch_tendency(
    frame: pd.DataFrame,
    league_mix: pd.DataFrame | None = None,
) -> list[dict]:
    """How he opens a plate appearance, when it is exploitable.

    Two separate findings: a predictable first pitch, and taken first pitches
    called strikes rarely enough that taking is the correct approach.

    The second uses the same measure and cutoff as the plan's first-pitch line
    — called strikes on TAKEN first pitches — so the two can never disagree.
    The usual first-pitch strike rate counts every swing as a strike, which
    makes it look like taking costs a strike far more often than it does.
    """
    min_n = int(config.get("flags.min_n.first_pitch", 30))
    min_share = float(config.get("flags.thresholds.first_pitch_share", 0.70))
    take_below = float(config.get("plan.first_pitch_take_below", 0.40))
    weight = float(config.get("report.severity_weights.first_pitch", 2.0))

    summary = counts.first_pitch_tendencies(frame)
    if summary["n"] < min_n:
        return []

    findings = []

    if summary["primary_share"] >= min_share:
        findings.append(_finding(
            flag="first_pitch",
            severity=_threshold_severity(
                summary["primary_share"], min_share, weight,
                int(summary["n"]), min_n,
            ),
            basis=BASIS_THRESHOLD,
            n=int(summary["n"]),
            claim=(
                f"He opens with the {pitch_name(summary['primary_pitch'])} "
                f"{summary['primary_share']:.0%} of the time "
                f"(n={int(summary['n'])})."
            ),
            observed=summary["primary_share"],
            reference=min_share,
        ))

    first = frame[(frame["balls"] == 0) & (frame["strikes"] == 0)]
    taken = first[~first["is_swing"].eq(True)]
    if len(taken) >= min_n:
        called = float(taken["is_called_strike"].eq(True).mean())
        if called <= take_below:
            findings.append(_finding(
                flag="first_pitch",
                # Inverted so the observed quantity rises as the finding gets
                # stronger: a lower called-strike rate is a bigger finding, and
                # the severity formula expects observed/threshold to grow.
                severity=_threshold_severity(
                    take_below / max(called, 1e-9), 1.0, weight,
                    int(len(taken)), min_n,
                ),
                basis=BASIS_THRESHOLD,
                n=int(len(taken)),
                claim=(
                    f"Taken first pitches are called strikes only {called:.0%} "
                    f"of the time (n={len(taken)} taken), so taking the first "
                    f"pitch is the percentage play."
                ),
                observed=called,
                reference=take_below,
            ))

    return findings


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def evaluate(
    frame: pd.DataFrame,
    league_outcomes: pd.DataFrame | None = None,
    league_mix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run every rule and return the findings, ranked by severity.

    Returns a DataFrame with FINDING_COLUMNS, sorted by severity descending.
    Rules requiring league data are skipped when it is absent, so this works
    against a bare fixture as well as a fully built cache.

    Ranking mixes two severity bases by design, and `severity_basis` names
    which one produced each row. See the module docstring, and the comment on
    `report.severity_weights` in config.yaml, for why that is a deliberate
    compromise rather than an oversight.
    """
    if len(frame) == 0:
        return pd.DataFrame(columns=FINDING_COLUMNS)

    findings: list[dict] = []
    findings.extend(predictable_counts(frame, league_mix))
    findings.extend(handedness_gaps(frame))
    findings.extend(hittable_pitches(frame, league_outcomes))
    findings.extend(fatigue_decline(frame))
    findings.extend(sequencing_tells(frame))
    findings.extend(location_tells(frame))
    findings.extend(first_pitch_tendency(frame, league_mix))

    if not findings:
        return pd.DataFrame(columns=FINDING_COLUMNS)

    # The false-discovery correction runs here because here is the only place
    # the whole family is visible. A rule that corrected its own rows would be
    # correcting for a fraction of the comparisons actually made.
    resolved = benjamini_hochberg([f["rv_p"] for f in findings])
    min_runs = float(config.get("flags.thresholds.min_runs_cost", 1.0))

    for finding, is_resolved in zip(findings, resolved):
        finding["resolved"] = bool(is_resolved)
        exploitable, phrase = _consequence(
            finding["runs_cost"], min_runs, is_resolved,
            finding["runs_lo"], finding["runs_hi"],
        )
        finding["claim"] = finding["claim"] + phrase
        # Only a demonstrated benefit to the pitcher demotes a finding.
        # "Unresolved" is not evidence of harmlessness, so it must not.
        if not exploitable:
            finding["severity"] = finding["severity"] / 2.0

    table = pd.DataFrame(findings)
    table = table.sort_values("severity", ascending=False, na_position="last")
    return table.reset_index(drop=True)[FINDING_COLUMNS]


def comparison_count(
    frame: pd.DataFrame,
    league_mix: pd.DataFrame | None = None,
) -> dict:
    """How many comparisons the report ran to produce its findings.

    Returns counts per family plus the total, and how many would be expected to
    clear an uncorrected p < .05 by chance alone.

    This belongs in the report because "8 findings" means something completely
    different depending on whether it came out of 15 comparisons or 400. On
    Yamamoto the denominator is 120, the uncorrected hit count is 9, and 6 is
    what pure noise produces — a difference a reader cannot assess without
    being told the denominator.
    """
    families = {
        "count": len(counts.count_lifts(frame, min_n=20)),
        "location": sum(
            len(location.location_tells(frame, by=band, min_n=40))
            for band in location.BAND_COLUMNS
        ),
        "sequencing": len(sequencing.setup_pairs(frame, min_n=25)),
    }
    total = sum(families.values())
    return {
        "families": families,
        "total": total,
        "expected_by_chance": round(0.05 * total, 1),
    }


def takeaways(
    frame: pd.DataFrame,
    league_outcomes: pd.DataFrame | None = None,
    league_mix: pd.DataFrame | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """The top findings, one per flag type, capped at `report.max_takeaways`.

    Deduplicated by flag before truncation. Five takeaways that are five
    variations of the same location tell tell a coach one thing; five
    takeaways drawn from five different rules tell him five. Within a flag the
    highest-severity finding survives, so nothing is lost except repetition —
    the full set remains available from evaluate().
    """
    if limit is None:
        limit = int(config.get("report.max_takeaways", 5))

    table = evaluate(frame, league_outcomes, league_mix)
    if len(table) == 0:
        return table

    best = table.drop_duplicates(subset="flag", keep="first")
    return best.head(limit).reset_index(drop=True)
