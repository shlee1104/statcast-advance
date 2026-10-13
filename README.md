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
- Sequencing metrics — pitch-to-pitch transitions by batter side with their
  counts, outcome-conditioned transitions, two-strike putaway, setup-pitch lift
  analysis
- League baselines — stratified date sampling (~100k pitches), handedness-split
  reference rates computed in SQL, and z-tested comparisons of a pitcher's
  arsenal, count mix, predictability, and putaway rates against them
- Arsenal metrics — velocity, spin, movement and results by pitch type, plus a
  release-point consistency check that flags offerings thrown from a different
  slot than the fastball
- Splits — platoon usage gaps, in-outing velocity decay, and a by-trip table
  whose velocity change is measured inside each outing. It reports what changes
  late without claiming why, since trip number and pitch count are collinear
- Location — plate coordinates normalized to the pitcher's arm side and to the
  batter's own strike zone, with per-pitch quadrant concentration, command
  spread, and a conditional-entropy score for how much a location band gives
  away about which pitch is coming
- Weakness-flag rules engine — six rules over the metrics layer, each finding
  carrying its sample size in the sentence, ranked on a single severity axis
  that records whether it came from a league z-score or a weighted threshold.
  A usage lift has to clear its minimum sample and stand at least three
  standard errors from no lift before it is a finding
- Interactive HTML report — the dugout card and findings first, then the count,
  movement, location, arsenal, sequencing and release evidence; zone heat maps
  sliceable by count and batter side; self-contained in one file with no server
  and no build step
- Single-command CLI — `python -m src.cli --pitcher "Name"` from name to report
- Outcome coupling — every tell's cost in runs is estimated with a confidence
  interval, and printed only if it survives the correction below; the page says
  once how many comparisons were run and how many survived
- Multiple-comparison correction — Benjamini-Hochberg across every run-value
  comparison the report makes (not just the ones that became findings), with
  the number of comparisons printed next to the number of findings
- Dugout card — page one is a one-page advance sheet, laid out the way clubs
  lay theirs out: season line, arsenal with velocity ranges, his last five
  outings, pitch-mix pies for five situations and a zone map for every pitch,
  both split by batter side, and the keys. It prints on exactly one letter
  page, for a two-pitch closer and a seven-pitch starter alike
- One page means choosing — the card prints what repeats between halves of a
  season and the plan lines a hitter can act on, in a scout's clipped wording;
  "he mixes", per-pitch xwOBA and anything in runs stay in the pages behind it
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
- Replication check — `scripts/replication.py` re-measures every finding the
  rules make in the first half of a season on the second half, and correlates
  every table column between halves, so each section of the report can be
  judged on whether it repeats
- Pruning — what fails those two checks comes out of the report, with its
  cutoffs and its code: the first-pitch "take or swing" advice, a verdict on
  the third trip through the order, a "hittable pitch" finding, and a run
  estimate printed on every finding (see "Does the plan hold up?" and "Do the
  findings repeat?")
- Sequence graph — what follows each pitch, one panel per batter side, read
  left to right: the pitch he just threw, then the next one. Lines are as thick
  as the follow-up is common, and coloured only where it comes clearly more or
  less often than usual; titled with an auto-written sentence ("he doubles up
  on the slider", "the curveball sets up the sinker") or one saying nothing
  stands out
- 624 unit tests, plus an end-to-end validation suite over live data

**Next**

- Hitter reports
- Pitch tunneling (see [docs/tunneling_design.md](docs/tunneling_design.md))
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

- **Dugout card** — one printed page, top to bottom:
  - *Season line* — K%, BB%, whiff, chase, zone, first-pitch strike, xwOBA.
  - *Arsenal* — each pitch as a scout writes it: where the velocity sits and
    its max, usage against lefties and against righties, inches of break,
    whiff and chase rates. A rate on too few swings is greyed.
  - *Last five outings* — pitches, fastball velocity, whiff rate, strikeouts,
    walks.
  - *Pitch mix by situation* — ten pies: first pitch, pitcher's count, even,
    hitter's count and two strikes, to each side. A pie on fewer than 30
    pitches is faded.
  - *Location by pitch* — a thirteen-zone map for every pitch to each side, in
    that pitch's own colour. A pitch thrown fewer than 40 times to a side is
    not mapped.
  - *Keys* — to lefties, to righties and to either side: what to sit on in a
    hitter's count, his two-strike pitch and whether to lay off it low, what a
    side will not see, the follow-up that stands out most, a count or height
    tell, and what has changed lately. Each line carries its sample size.
  - *Running game and scout notes* — blank, for a pen. Time to the plate,
    pickoff move and tipping come from video and a stopwatch, not from pitch
    tracking.

  The card prints a plan line only when it tells the hitter something to do.
  The full plan — including the first-pitch mix, "he mixes" and "look
  fastball" — is still built and still in the report's data, and the lines on
  the card are the same sentences cut down (`gameplan.card_keys`).
- **Findings** — the rule-based findings behind the plan, the strongest in each
  category, each carrying its sample size, with one note on how many
  comparisons produced them and how many run-cost estimates survived
  correction
- **Count grid** — one balls-by-strikes grid per pitch: how far each count
  moves him from his own usual rate, with the share and pitch count in every
  cell
- **Movement** — ride against arm-side run for each pitch, with the league's
  version of the pitch marked behind it
- **Velocity through the outing** — his main fastball by pitch count, each
  outing compared with its own first pitches, and a by-trip table underneath
- **Usage by batter side** — one line per pitch from its share against lefties
  to its share against righties
- **Location** — a zone grid for every pitch, sliceable by count state and
  batter side, as share or as lift over his usual; and what a pitch's height
  gives away about which pitch it is
- **Arsenal against the league, by batter side** — usage, velocity, whiff rate
  and xwOBA for each pitch next to the league's version of it, zone rate, and
  whiff rate against each side
- **Count tendencies** — a normalized-entropy predictability score for each
  count, with the league's share of the same pitch as context
- **Sequencing** — a graph of what follows each pitch to lefties and to
  righties, coloured where a follow-up departs from his usual; then setup
  pairs keyed on pitch type and height, with how much more likely the
  follow-up becomes and whether it then misses more bats; and his two-strike
  mix and put-away rates
- **Release consistency** — where each pitch leaves his hand compared with his
  fastball, the first step of the tunneling design

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
| Hitter's count: what to sit on | 24 | 67% (47–82%) | 67% | 29% | 4% (1–20%) | +0.29 (+0.01 to +0.50) |
| Two strikes: the put-away pitch | 46 | 70% (55–81%) | 70% | 30% | 0% (0–8%) | +0.19 (−0.09 to +0.39) |
| Two strikes: lay off it low | 21 | 76% (55–89%) | 76% | 24% | 0% (0–15%) | +0.68 (+0.53 to +0.78) |
| Won't see | 32 | 97% (84–99%) | 91% | 3% | 0% (0–11%) | — |
| Count tell | 6 | 100% (61–100%) | 50% | 0% | 0% (0–39%) | — |
| Read the height | 11 | 82% (52–95%) | 82% | 18% | 0% (0–26%) | — |

*Held* means the advice was still usable in the second half. *Weaker* means it
said less — "he mixes", "look fastball" — which costs a hitter nothing he
would have had without the report. *Flipped* means it pointed the other way,
the failure that matters. *Skill* is how much better the pitcher's own
first-half number predicts his second half than the average pitcher's does:
+0.29 means 29% less error, 0 would mean knowing the pitcher adds nothing.
Rate intervals are Wilson intervals; skill intervals are a bootstrap over
pitchers.

What that table says:

- **The plan rarely points a hitter the wrong way.** One flip in 140 pieces
  of advice, and none on the two-strike, won't-see, count or height lines.
- **"Lay off it low" and "won't see" are the strongest lines.** Where a
  pitcher's two-strike pitch finishes is his own habit and it carries over
  (+0.68); a pitch one side never saw in the first half stays unseen.
- **The first-pitch take-or-swing advice did not hold up, and is gone.** The
  plan used to tell a hitter to take or swing at the first pitch, from how
  often his taken first pitches were called strikes. On this run it gave that
  advice 12 times and it held 7 (58%, interval 32–81%), flipped once, and
  showed a skill of +0.01 (−0.53 to +0.35): his own rate predicted his second
  half no better than the average pitcher's. Pitchers differ too little on
  that rate for half a season to separate them. The line now states his
  first-pitch mix and nothing else, and its three cutoffs are out of the
  config. The table above is the plan as it stands; the row that was removed
  is reproducible at commit `cc5a2ed`.
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
| Hitter's count: what to sit on | 36 | 72% (56–84%) | 72% | 25% | 3% (0–14%) | +0.45 (+0.20 to +0.60) |
| Two strikes: the put-away pitch | 57 | 68% (56–79%) | 68% | 30% | 2% (0–9%) | +0.43 (+0.27 to +0.55) |
| Two strikes: lay off it low | 21 | 76% (55–89%) | 76% | 24% | 0% (0–15%) | +0.70 (+0.54 to +0.80) |
| Won't see | 44 | 91% (79–96%) | 77% | 9% | 0% (0–8%) | — |
| Count tell | 7 | 100% (65–100%) | 57% | 0% | 0% (0–35%) | — |
| Read the height | 17 | 88% (66–97%) | 76% | 12% | 0% (0–18%) | — |

The first run of this check changed the plan. "Sit <pitch>" flipped 28% of
the time, always on two-pitch relievers near 50/50 where the leader in one half
was the runner-up in the other; it now needs a 15-point lead. Rerun on the same
pitchers and season, the hitter's-count flip rate went from 12% to 3% — but the
cutoff was chosen to remove exactly those flips, so the 2025 table measures how
well it fits 2025, not how it does on new data. Hold rates came through to 2026
about where they were; the skill scores did not, falling from +0.45 to +0.29
for the hitter's count and from +0.43 to +0.19 for the put-away pitch.

The take-or-swing line is the cautionary case. It got the same treatment — a
floor of 80 taken first pitches, chosen because every flip sat below 60 — and
on 2025 its flip rate went from 14% to 0%. On 2026 it had no skill at all.
A cutoff tuned on the season it is tested on will look fixed.

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

## Do the findings repeat?

The holdout check tests the plan's lines. `scripts/replication.py` asks the same
question of everything else in the report: each finding the rules make from the
first half of a pitcher's season is re-measured in the second half, at the
rule's own bar.

| Finding | 2025: found | Held (95% CI) | Same direction | 2026: found | Held (95% CI) | Same direction |
|---|---|---|---|---|---|---|
| Count tendency | 56 | 82% (70–90%) | 98% | 47 | 51% (37–65%) | 96% |
| Height tell | 19 | 74% (51–88%) | 100% | 11 | 82% (52–95%) | 100% |
| Lefty/righty split | 15 | 67% (42–85%) | 100% | 17 | 71% (47–87%) | 100% |
| Sequencing (setup by pitch and height) | 10 | 40% (17–69%) | 90% | 11 | 55% (28–79%) | 91% |
| Sequence graph: follow-up more than usual | 17 | 53% (31–74%) | 88% | 19 | 68% (46–85%) | 95% |
| Sequence graph: follow-up less than usual | 16 | 56% (33–77%) | 94% | 3 | 33% (6–79%) | 100% |
| Velocity late in starts | 6 | 33% (10–70%) | 100% | 4 | 75% (30–95%) | 100% |
| Third time through (verdict), since removed | 2 | 0% (0–66%) | 0% | 5 | 20% (4–62%) | 40% |

*Held* means the second half cleared the same bar on its own. *Same direction*
means it at least still pointed the same way. A second half too thin to
measure a finding in is left out, not counted as a miss. Thirty pitchers in
2025, twenty-eight in 2026.

The usage findings — counts, height, batter side — almost never reverse, and
hold their full size half to four-fifths of the time. Sequencing keeps its
direction but holds its size about half the time, on few cases. That is true
of the lines the sequence graph colours as well: across both seasons 55 were
found, 93% pointed the same way in the other half and 58% were as strong. Of
those, doubling up on a pitch held 72% of the time (25 cases) and one pitch
setting up a different one 36% (11 cases), which is why the graph's note tells
the reader how far to trust a coloured line. Its bars were chosen after
looking at these two seasons, so those rates describe the fit. The verdict on
the third trip through the order came back once in seven, and was removed; the
by-trip table it was read from is still in the report.

The same script computes every table column in each half and correlates them
across pitcher-pitches (86 in 2025, 77 in 2026):

| Number | 2025 r | 2026 r |
|---|---|---|
| Velocity, horizontal and vertical break | 0.99–1.00 | 0.99–1.00 |
| Usage | 0.96 | 0.92 |
| Location: height | 0.96 | 0.97 |
| Release: distance from his fastball slot | 0.84 | 0.88 |
| Zone rate | 0.85 | 0.91 |
| Whiff rate | 0.81 | 0.82 |
| Chase rate | 0.77 | 0.73 |
| xwOBA | 0.61 | 0.59 |
| xwOBA against the league's for that pitch | 0.44 | 0.58 |
| Run value per pitch | 0.28 | 0.37 |
| Third trip: change in fastball velocity (per pitcher) | 0.86 | 0.76 |
| Third trip: change in xwOBA (per pitcher) | 0.06 | 0.29 |

What he throws, how it moves and where it goes are properties of the pitcher.
What happened to it is mostly a property of the half: a pitch's run value in
one half says little about the other, and the change in results on a third
trip says close to nothing, while the change in velocity on that trip carries
over.

Across the 58 full-season reports the rules made 287 findings, and the run
consequence of one of them survived correction, so the run estimate came off
every finding and is stated once. The hittable-pitch finding fired for 2 of 30
pitchers in 2025 and 3 of 28 in 2026, and was removed.

The script as it stands measures the rules that remain (260 findings across
the same 58 reports). The figures for the removed rules are reproducible at
commit `432abb0`, where those rules still exist.

The dugout card's two pictures are split by batter side, which none of the
columns above are, so the script checks them directly: every pie slice and
every zone-map cell the card would draw in both halves.

| On the card | Season | r | Median move | 90% within | Same top |
|---|---|---|---|---|---|
| Pie slice: a pitch's share of a situation, by side | 2025 | 0.93 | 3.7 pts | 12.0 pts | 76% of 282 pies |
| | 2026 | 0.90 | 3.8 pts | 12.2 pts | 78% of 228 pies |
| Zone-map cell: a pitch's share of a zone, by side | 2025 | 0.88 | 1.8 pts | 6.4 pts | 74% of 164 maps |
| | 2026 | 0.89 | 1.9 pts | 6.4 pts | 75% of 146 maps |

*Move* is the change in that number from the first half to the second, in
percentage points, over 1,199 slices and 2,132 cells in 2025 and 1,075 and
1,898 in 2026, from the pitchers with 200 pitches in each half (30 and 25).
*Same top* is how often a pie's largest slice, or a map's busiest zone, is
the same one in both halves. The correlation flatters both pictures — a 40%
pitch and a 4% pitch are far apart in either half — so the move is the figure
to read: a pie slice is good to about four points and a zone cell to about
two, and one slice in ten moves twelve points or more. The card says so in
its footnote. Each half is half the sample the full-season card is drawn
from, so this is the harder version of the test.

```bash
python scripts/replication.py --season 2025
python scripts/replication.py --season 2026
python scripts/replication.py --season 2026 --cached   # local cache only, no network
```

These correlations are across a list built to span very different pitchers,
so they are higher than they would be across a random sample of the league.

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
none survive the correction, which is why that report prints no run figure for
any finding. (More than six clear it because the null is not
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
something.) So the cost of every tell is estimated in runs — and one season of
one pitcher usually cannot resolve it: run value per pitch has a standard
deviation near 0.19 against pattern-level differences of 0.01 to 0.02. A run
figure is printed only when it survives the correction. Otherwise the
frequency pattern is the finding and the report prints no run figure, rather
than one that implies precision it does not have. It says so once, under the
findings, with the number of comparisons behind it; it used to say so in a
sentence on every finding, and a caveat attached to everything is read by no
one.

**What does not repeat comes out.** A line in the plan is a prediction, and a
finding type is only worth its space if it comes back in games it was not
found in. The first-pitch take-or-swing advice was calibrated, gated on
sample size and unit-tested, and still predicted nothing on a new season, so
it was deleted rather than hedged. A verdict on the third trip through the
order (back in 1 of 7 cases) and a "hittable pitch" finding (fired in 5 of 58
pitcher-seasons) were removed the same way after being re-measured on the other half
of the season; the tables they were read from are still in the report, without
the conclusion.

**One page means choosing, and the choice has a rule.** Clubs pare a phone
book down to a page, and the page is short because someone decided what a
hitter needs. Here the decision is the two checks above. Usage, velocity,
movement and location are nearly the same pitcher in both halves of a season
(r of 0.92 and up), whiff and chase rates mostly (0.73 to 0.82), so they are
on the card. A pitch's xwOBA (0.6) and anything in runs are not. Of the plan's
lines, the card takes the ones the holdout check scores as advice and leaves
"he mixes" and "look fastball", which that check calls what a hitter would
have had without a report. Across the 58 calibration reports that put 367
lines on cards and left 171 off, 113 of them first-pitch mixes the pies now
draw. Nothing was deleted to do it: the full plan is still built, and the
card's lines are the same sentences in a scout's clipped form.

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
