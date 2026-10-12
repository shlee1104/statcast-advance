# statcast-advance

Automated advance scouting reports from MLB pitch-level data.

Player name in, coach-ready interactive report out.

## What this is

As an analytics intern with the UC Davis D1 baseball team, I produced advance
scouting reports on assigned opponents by hand: pull the player up in TruMedia,
work through usage, sequencing, and situational splits, separate real tendencies
from small-sample noise, and write up the handful of findings most likely to
change an in-game decision.

This project rebuilds that workflow in code. It ingests pitch-level Statcast
data, computes count and sequencing tendencies against league baselines, flags
exploitable patterns with explicit minimum-sample gates, and renders a
self-contained interactive HTML report.

D1 Trackman data is proprietary, so this uses MLB Statcast — the same category of
pitch-level measurement (release point, velocity, spin, movement, plate location,
outcome), which makes the methodology directly transferable.

## Status

Working end to end against real season data. A pitcher name goes in and a
self-contained interactive HTML report comes out.

**Complete**

- Baseball Savant API client — retry, rate limiting, name-to-MLBAM resolution
- DuckDB cache — typed 40-column schema, primary-key upserts, freshness checks
- Cleaning layer — pitch-type consolidation, non-competitive pitch removal,
  game-type filtering, count validation
- Count metrics — usage by count, first-pitch tendencies, normalized-entropy
  predictability scoring
- Sequencing metrics — transition matrices, outcome-conditioned transitions,
  two-strike putaway, setup-pitch lift analysis
- League baselines — stratified date sampling (~100k pitches), handedness-split
  reference rates computed in SQL, and z-tested comparisons of a pitcher's
  arsenal, count mix, predictability, and putaway rates against them
- Arsenal metrics — velocity, spin, movement and results by pitch type, plus a
  release-point consistency check that flags offerings thrown from a different
  slot than the fastball
- Splits — platoon usage gaps, in-outing velocity decay, and a times-through-order
  comparison that reports what changes late without claiming why, since trip
  number and pitch count are collinear
- Location — plate coordinates normalized to the pitcher's arm side and to the
  batter's own strike zone, with per-pitch quadrant concentration, command
  spread, and a conditional-entropy score for how much a location band gives
  away about which pitch is coming
- Weakness-flag rules engine — seven rules over the metrics layer, each finding
  carrying its sample size in the sentence, ranked on a single severity axis
  that records whether it came from a league z-score or a weighted threshold.
  A usage lift has to clear its minimum sample and stand at least three
  standard errors from no lift before it is a finding
- Interactive HTML report — takeaways first, then arsenal, location, count,
  sequencing and splits evidence; a normalized zone heat map; self-contained in
  one file with no server and no build step
- Single-command CLI — `python -m src.cli --pitcher "Name"` from name to report
- Outcome coupling — every tell reports what the pattern costs him in runs, with
  a confidence interval, and says so when the interval does not exclude zero
- Multiple-comparison correction — Benjamini-Hochberg across every run-value
  comparison the report makes (not just the ones that became findings), with
  the number of comparisons printed next to the number of findings
- Dugout card — page one is laid out like a pro advance report: a short season
  line, a plain-language hitting plan split by batter side, the arsenal with
  velocity ranges and usage against each side, count tendencies by side, and
  the pitcher's last five outings against the rest of his season
- Count grid — one balls-by-strikes lattice per pitch, coloured by how far each
  count moves him from his own habit, with an auto-written headline stating
  what it shows; it replaced a stacked bar chart that hid the count's structure
- Platoon slope chart and movement plot — usage against each side drawn as one
  line per pitch, and ride against arm-side run with league-average markers;
  both titled with an auto-written sentence stating what they show
- Calibration run — `scripts/batch_reports.py` runs the report across 30
  deliberately varied pitchers and prints, for every hitting-plan cutoff, how the
  measured values spread and how often the cutoff fires
- Calibrated hitting plan — cutoffs reset from that run so each line fires for
  a minority of pitchers rather than describing the league: "sit hard" now
  needs clearly more fastballs than the league's 63%, "sit <pitch>" needs a
  majority, and lines that were true of nearly everyone ("if it's up, it's the
  four-seam") are no longer written
- Combined seasons — `--seasons 2025 2026` pools a short season with the one
  before it for a usable sample, compares the two to name what changed (new or
  shelved pitches, usage shifts beyond noise, velocity), and leaves a shelved
  pitch out of the plan rather than telling a hitter to look for it
- Holdout check — `scripts/holdout.py` builds the plan from the first half of
  each pitcher's season and tests every call against the second half (or the
  regular season against the postseason): how often the advice held, how often
  it flipped to the opposite advice, and whether the pitcher's own numbers
  predict better than an average pitcher's would, each with an interval. Run on
  2026, a season no cutoff was set on
- 481 unit tests, plus an end-to-end validation suite over live data

**Next**

- Pitch tunneling (see [docs/tunneling_design.md](docs/tunneling_design.md))
- Hitter reports
- Swing-disruption metrics from the bat-tracking fields

## Setup

Requires Python 3.11 or 3.12.

```bash
git clone https://github.com/shlee1104/statcast-advance.git
cd statcast-advance

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python scripts/check_setup.py
```

`check_setup.py` verifies the Python version, installed packages, config
parsing, and live connectivity to Baseball Savant. It should end with
"Everything checks out."

## Usage

Fetch a pitcher-season and cache it locally:

```bash
python scripts/fetch_fixture.py --pitcher "Yoshinobu Yamamoto" --season 2025
```

Verify the pipeline end to end — fetch, clean, store, read back, with ten
data-quality checks:

```bash
python scripts/smoke_test.py
```

Run the metrics layer and print every current output:

```bash
python scripts/explore.py --pitcher yamamoto
```

Run the calibration list — 30 pitchers chosen to cover starters and
relievers, both hands, sinkerballers, splitter and forkball pitchers, two-pitch
closers, soft-tossers and short seasons — and see where each plan cutoff lands:

```bash
python scripts/batch_reports.py
```

Reports and three CSVs (`summary.csv`, `plan_lines.csv`, `thresholds.csv`) go to
`reports/calibration/`. A pitcher that fails is recorded and the run carries on.

Check whether the plan holds up on games it never saw — built from the first
half of each season, tested on the second, or built from the regular season and
tested on the postseason:

```bash
python scripts/holdout.py --season 2025
python scripts/holdout.py --season 2025 --split postseason
```

Generate a report. Name in, self-contained HTML out:

```bash
python -m src.cli --pitcher "Yoshinobu Yamamoto" --season 2025
```

For a pitcher whose latest season is short — an injury, a midseason call-up —
combine it with the one before:

```bash
python -m src.cli --pitcher "Blake Snell" --seasons 2025 2026
```

Tendencies pool both seasons; a "Since last season" line on the card says what
changed, and any pitch he has stopped throwing is left out. A single season
under 1,500 pitches prints this suggestion on its own.

If two active players share the name, the lookup stops rather than guess;
pass the MLBAM id it lists with `--player-id`.

It resolves the name, reads the cache or fetches, cleans, computes every
metric, runs the flags, and writes `reports/<name>_<season>.html`. League
baselines are used if they exist and skipped if they do not. To work offline
from a committed fixture:

```bash
python -m src.cli --pitcher "Yoshinobu Yamamoto" --fixture yamamoto
```

## Sample output

Two-strike approach, Yamamoto 2025:

| Pitch | Usage | Putaway rate | Whiff rate |
|---|---|---|---|
| Splitter | 39.9% | 24.0% | 35.2% |
| Four-seam | 28.6% | 23.1% | 22.5% |
| Curveball | 18.0% | 25.3% | 32.1% |

The curveball finishes plate appearances at a higher rate than the splitter
while being thrown less than half as often — and in two-strike counts the
splitter reaches 46–50% usage, among his most predictable spots. Usage and
effectiveness are computed separately precisely so gaps like this surface.

Setup-pitch detection on the same season finds that a four-seam up in the zone
raises curveball frequency 59% above baseline (n=89). The lift is still there
when the count the curveball is thrown in is held fixed — 1.8x with the pitcher
ahead, 1.6x even, 1.4x behind, on 26 to 33 pitches each — which points to
sequencing intent rather than count logic. Three count states is a coarse
control and those are small samples, so this is support, not proof.

## Report contents

The structure follows how teams describe their own advance reports: a short
dugout version first, and the detail behind it after.

- **Dugout card** — one printable page. Season line (K%, BB%, whiff, chase,
  zone, xwOBA). A hitting plan for lefties and for righties: what he starts
  hitters with and whether taking is a good idea, what to sit on when ahead in
  the count, his two-strike pitch and where it finishes, and which pitches he
  essentially never shows that side. Every line carries its sample size. Below
  that, the arsenal as a scout writes it ("sits 94–97, touches 98", ride,
  arm-side run, usage vs each side), count tendencies by side, and his last
  five outings. The running game — time to the plate, pickoff move, tipping —
  is named as not covered, because it needs video rather than pitch data.
- **Arsenal** — velocity, spin, movement, usage, and whiff rate by pitch type
- **Count tendencies** — pitch mix across all 12 counts, plus a normalized-entropy
  predictability score identifying the counts where selection is most anticipatable
- **Sequencing** — within-plate-appearance pitch transition matrices, variants
  conditioned on the previous pitch outcome, two-strike putaway mix, and
  setup-pitch detection keyed on pitch type and location band
- **Location** — where each pitch lives in coordinates normalized to the pitcher's
  arm side and the batter's own zone, command spread, and how much knowing the
  region narrows down which pitch is coming
- **Splits** — platoon splits, velocity decay within outings, times-through-order
- **Findings** — the rule-based findings behind the plan, ranked, each carrying
  its sample size, with run consequences stated as unresolved where they do not
  survive correction

## Does the plan hold up?

A scouting report is a prediction: it says what a pitcher did, and the hitter
acts as if he will keep doing it. `scripts/holdout.py` tests that directly. It
builds the plan from the first half of each pitcher's season and checks every
call against the second half, which the plan never saw.

The test that counts is 2026. Every cutoff in the plan was set on 2025 data, so
2026 is a season none of them has seen. Twenty-eight of the thirty calibration
pitchers threw in it:

| Plan line (2026) | Advice given | Held (95% CI) | Same line by the plan's own cutoff | Weaker | Flipped (95% CI) | Skill (95% CI) |
|---|---|---|---|---|---|---|
| First pitch: take or swing | 12 | 58% (32–81%) | 58% | 33% | 8% (1–35%) | +0.01 (−0.53 to +0.35) |
| Hitter's count: what to sit on | 24 | 67% (47–82%) | 67% | 29% | 4% (1–20%) | +0.29 (+0.01 to +0.50) |
| Two strikes: the put-away pitch | 46 | 70% (55–81%) | 70% | 30% | 0% (0–8%) | +0.19 (−0.09 to +0.39) |
| Two strikes: lay off it low | 21 | 76% (55–89%) | 76% | 24% | 0% (0–15%) | +0.68 (+0.53 to +0.78) |
| Won't see | 32 | 97% (84–99%) | 91% | 3% | 0% (0–11%) | — |
| Count tell | 6 | 100% (61–100%) | 50% | 0% | 0% (0–39%) | — |
| Read the height | 11 | 82% (52–95%) | 82% | 18% | 0% (0–26%) | — |

*Held* means the advice was still usable in the second half. *Weaker* means it
said less — "he mixes", "no strong case" — which costs a hitter nothing he
would have had without the report. *Flipped* means it pointed the other way,
the failure that matters. *Skill* is how much better the pitcher's own
first-half number predicts his second half than the average pitcher's does:
+0.29 means 29% less error, 0 would mean knowing the pitcher adds nothing.
Rate intervals are Wilson intervals; skill intervals are a bootstrap over
pitchers.

What that table says:

- **The plan rarely points a hitter the wrong way.** Two flips in 152 pieces
  of advice, and none on the two-strike, won't-see, count or height lines.
- **"Lay off it low" and "won't see" are the strongest lines.** Where a
  pitcher's two-strike pitch finishes is his own habit and it carries over
  (+0.68); a pitch one side never saw in the first half stays unseen.
- **The first-pitch take-or-swing advice does not hold up.** Skill of +0.01
  with an interval from −0.53 to +0.35, the lowest hold rate, and one of the
  two flips. Pitchers differ too little on that rate for half a season to
  separate them. It is the line to drop.
- **Naming the put-away pitch is safe, and less informative than it looked.**
  It never flipped, but its skill interval reaches zero: which pitch he goes
  to is stable, how much he leans on it is not much better predicted by his
  own number than by a typical pitcher's.
- **"Held" is a looser bar than the plan's own for three lines.** A "won't
  see" pitch counts as held under 5%, where the plan writes the line under 3%;
  a count tell at 1.5 times his rate, where the plan needs 2 times and a 25%
  share; a height line at a 45% share. The fourth column is the strict
  version, and for the count tell it halves the number.

### The 2025 run, and why it is not the headline

The same check on 2025 (30 pitchers), where the cutoffs were set:

| Plan line (2025) | Advice given | Held (95% CI) | Same line by the plan's own cutoff | Weaker | Flipped (95% CI) | Skill (95% CI) |
|---|---|---|---|---|---|---|
| First pitch: take or swing | 11 | 73% (43–90%) | 73% | 27% | 0% (0–26%) | +0.11 (−0.33 to +0.32) |
| Hitter's count: what to sit on | 36 | 72% (56–84%) | 72% | 25% | 3% (0–14%) | +0.45 (+0.20 to +0.60) |
| Two strikes: the put-away pitch | 57 | 68% (56–79%) | 68% | 30% | 2% (0–9%) | +0.43 (+0.27 to +0.55) |
| Two strikes: lay off it low | 21 | 76% (55–89%) | 76% | 24% | 0% (0–15%) | +0.70 (+0.54 to +0.80) |
| Won't see | 44 | 91% (79–96%) | 77% | 9% | 0% (0–8%) | — |
| Count tell | 7 | 100% (65–100%) | 57% | 0% | 0% (0–35%) | — |
| Read the height | 17 | 88% (66–97%) | 76% | 12% | 0% (0–18%) | — |

The first run of this check changed the plan. "Sit <pitch>" flipped 28% of
the time, always on two-pitch relievers near 50/50 where the leader in one half
was the runner-up in the other; it now needs a 15-point lead. Every
take-or-swing call that flipped rested on fewer than 60 taken first pitches;
that advice now needs 80. Rerun on the same pitchers and season, those two
flip rates went from 12% to 3% and from 14% to 0% — but the cutoffs were chosen
to remove exactly those flips, so the 2025 table measures how well they fit
2025, not how they do on new data. Hold rates came through to 2026 about
where they were; the skill scores did not, falling from +0.45 to +0.29 for the
hitter's count and from +0.43 to +0.19 for the put-away pitch.

Skill also depends on who is in the run. It compares each pitcher's own number
with the average across the list, and the calibration list was built to span
sinkerballers, splitter specialists and two-pitch closers. The more different
the pitchers, the worse the average predicts any one of them, so skill here is
higher than it would be across a random sample of the league.

```bash
python scripts/holdout.py --season 2026 --out reports/holdout/2026
python scripts/holdout.py --season 2025 --out reports/holdout/2025
```

The same check from regular season to postseason (2025, 19 pitchers) is
directional only — October samples are a few dozen pitches per side — but
points the same way, with one pattern worth following: some pitchers lean
harder on their best put-away pitch in October.

## Design notes

**Sample gates are a feature.** Every flag has a minimum-n threshold defined in
`config.yaml`, and every claim in the report states its sample size. A scouting
report that confidently asserts a tendency off 6 pitches is worse than no report,
because a coach may act on it.

**Two baselines, answering two questions.** League rates say who a pitcher is;
his own rates say what he is about to throw. Only the second is a hitter's
question, and conflating them buries findings. A splitter specialist beats the
league rate in every count he uses the pitch in, which reads as twelve findings
and is one fact — while a pitch he elevates in two specific counts never crosses
a league threshold at all, because his overall usage of it is ordinary. Counts
and sequences are therefore measured against his own mix, with league rates
carried as context.

**The denominator is part of the finding.** A report that surfaces eight
findings has run far more comparisons than eight. This one runs 120 — 44 count,
44 location, 32 sequencing — of which about six clear an uncorrected p < .05 by
chance alone. Benjamini-Hochberg is applied across all 120 in the engine,
where the whole family is visible, and the comparison count is printed next to
the finding count. On Yamamoto, sixteen comparisons clear the raw threshold and
none survive the correction, which is why every run consequence in that report
is stated as unresolved. (More than six clear it because the null is not
exactly true: where a pitch is thrown, and in what count, moves its run value
whether or not anyone saw it coming. A comparison that does survive says the
pitch does better or worse in that spot than it does for him generally; it
cannot say the predictability is the reason.)

**A minimum sample is not a noise filter.** "At least 20 of that pitch, at 1.5
times his usual rate" sounds like a guard and admits 20 pitches where 13 were
expected, which is under two standard errors — and a report looks at forty
such cells per pitcher. So every usage lift also has to stand three standard
errors clear of no lift, measured against his pitches outside that count or
location rather than against an overall rate that already contains them.

**Compare an outing with itself.** Pitches 90 to 105 only exist in the starts
that lasted, and those are the starts where he had his best stuff, so the
plain average of a late bucket can sit above the first one and read as
velocity recovering. The velocity curve is therefore the change within each
outing, averaged across outings, on one pitch: his most-used fastball. Blending
the fastball family moved the curve whenever his mix did — Yamamoto throws more
cutters late, four mph under his four-seam, and the blend called that a
one-mph drop.

**A tell is not a weakness.** Predictable and exploitable are different claims.
Hitters know Yamamoto's splitter is coming with two strikes and still manage
only a .163 xwOBA against it there, against .203 on everything else he throws
with two strikes. (xwOBA here is per plate appearance, strikeouts counted as
zero, so any two-strike number runs low; the comparison is what means
something.) So every tell reports what the pattern costs him in runs
— and reports honestly that one season of one pitcher usually cannot resolve
that: run value per pitch has a standard deviation near 0.19 against
pattern-level differences of 0.01 to 0.02. Where the interval spans zero the
report says the frequency pattern is the finding and the run consequence is
unresolved, rather than printing a run figure that implies precision it does
not have.

**The cache is a real data layer.** Pitch data is fetched on demand per player,
then persisted to a local DuckDB store. Repeat requests skip the network, and the
analytical work is written as SQL against pitch-level tables.

**Frequency and effectiveness are measured separately.** A sequence being
predictable and a sequence working are different claims, and they diverge often
enough to matter. A pattern the pitcher repeats without gaining anything is the
most exploitable finding a report can surface, and it is invisible to any metric
that collapses the two.

**Conditioning is confounded with the count.** The outcome of one pitch
determines the count for the next, so comparing "what he throws after a whiff"
against his overall mix conflates sequencing intent with ordinary count logic.
Setup analysis therefore supports holding the count state fixed.

## Data source

[Baseball Savant](https://baseballsavant.mlb.com/) (MLB Statcast), accessed via
its public CSV search endpoint. Player identity resolution uses
[pybaseball](https://github.com/jldbc/pybaseball). Requests are rate-limited;
please do not remove the delays in `config.yaml`.

## License

MIT
