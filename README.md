# statcast-advance

Automated advance scouting reports from MLB pitch-level data.

Player name in, a one-page advance card out, with the evidence one link
behind it.

**Sample (Yoshinobu Yamamoto, 2025):**
[the card](https://shlee1104.github.io/statcast-advance/sample_report.html) ·
[the evidence behind it](https://shlee1104.github.io/statcast-advance/sample_report_detail.html)

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

Working end to end against real season data. A pitcher name goes in and two
self-contained HTML files come out: a one-page card, and the evidence for it.

**Complete**

- Baseball Savant API client — retry, rate limiting, name-to-MLBAM resolution
- DuckDB cache — typed 52-column schema, primary-key upserts, freshness checks
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
- Findings engine — six rules over the metrics layer, each finding carrying
  its sample size in the sentence. A usage lift has to clear its minimum
  sample and stand at least three standard errors from no lift before it is a
  finding. The replication check below is run through it; its ranked list is
  no longer printed in the report (see Pruning)
- Two-file HTML report — the dugout card on its own, static, one printed
  page; and beside it the evidence: count, movement, location, arsenal,
  sequencing and release, with zone heat maps sliceable by count and batter
  side. Each file is self-contained, with no server and no build step, and
  links to the other
- Single-command CLI — `python -m src.cli --pitcher "Name"` from name to report
- Outcome coupling — every tell's cost in runs is estimated with a confidence
  interval and held to the correction below. Across 58 reports one estimate in
  287 survived it, so the report prints no run figure
- Multiple-comparison correction — Benjamini-Hochberg across every run-value
  comparison a report runs (a median of 74 to 80 per pitcher, not just the ones that
  became findings); `scripts/replication.py` prints how many were run and how
  many survive
- Dugout card — the default output is a one-page advance sheet, laid out the
  way clubs lay theirs out: season line, arsenal with velocity ranges, his last five
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
- Sample-size check — `scripts/sample_size.py` holds out half of a starter's
  games and builds reports from 3, 5 and 10 of the others, to see how much the
  report says, and how often it is right, on what a college staff would have
  on a pitcher (see "How few starts does it need?")
- Pruning — what fails those two checks comes out of the report, with its
  cutoffs and its code: the first-pitch "take or swing" advice, a verdict on
  the third trip through the order, a "hittable pitch" finding, and a run
  estimate printed on every finding (see "Does the plan hold up?" and "Do the
  findings repeat?"). Two sections left the page later and kept their code: a
  ranked findings list, whose "strength" score rested on weights that were a
  judgment call, and a per-count entropy table that no rule read
- Sequence graph — what follows each pitch, one panel per batter side, read
  left to right: the pitch he just threw, then the next one. Lines are as thick
  as the follow-up is common, and coloured only where it comes clearly more or
  less often than usual; titled with an auto-written sentence ("he doubles up
  on the slider", "the curveball sets up the sinker") or one saying nothing
  stands out
- Pitch tunneling — Savant's trajectory fit rebuilt for every pitch (run to
  the plate it lands on Savant's own location within 0.07 inch), then each
  pitch compared with the one that followed it: out of the hand, at the
  23.8-foot decision point, and in how the two move after it. Read as offsets
  against the pitcher's own scatter, which says whether a hitter could tell
  the two apart from where the ball is. In the evidence file, under the
  sequence graph, and not on the card: its central number repeats between
  halves at r of 0.5 to 0.7, below the card's bar
- 764 unit tests, plus an end-to-end validation suite over live data

**Next**

- Hitter reports
- League context for tunneling: where a pair sits among every pitcher's
  version of that pair (see [docs/tunneling_design.md](docs/tunneling_design.md))
- Swing-disruption metrics from the bat-tracking fields
- The checks on pitchers below the majors: the sample-size check cuts
  major-league starters down to a college-sized sample, which answers the
  sample question and not whether a Triple-A or college pitcher is as readable

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

Check what the report says, and how often it is right, when it is built from
3, 5 or 10 starts instead of half a season:

```bash
python scripts/sample_size.py --season 2025 2026 --cached
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
metric, builds the hitting plan, and writes two files:
`reports/<name>_<season>.html`, the one-page card, and
`reports/<name>_<season>_detail.html`, the evidence behind it. League
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
dugout version, and the detail behind it. They are two files. The card is the
one a coach is sent; it is static, needs no script, and prints on one page.
The second file is for whoever wants to check a line. They were one file,
card first, which made a one-page report twelve pages long.

**The card** (`<name>_<season>.html`)

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

**The evidence** (`<name>_<season>_detail.html`)

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
- **Sequencing** — a graph of what follows each pitch to lefties and to
  righties, coloured where a follow-up departs from his usual; then
  tunneling for the same follow-ups: how far apart the two pitches leave his
  hand, where the second is at the decision point relative to the first
  ("4 inches lower"), whether that reads as the same window or apart early,
  how differently they move after it, and how much later the second arrives;
  then setup pairs keyed on pitch type and height, with how much more likely
  the follow-up becomes and whether it then misses more bats; and his
  two-strike mix and put-away rates
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
pitchers. Wilson counts each piece of advice as its own fact, and a pitcher
gives up to two on a line, one to each batter side. The script prints the held
interval again from resampling pitchers; on both seasons it lands within about
six points of the one in the table.

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
| Tunneling: read as "same window" | 40 | 72% (57–84%) | 90% | 34 | 65% (48–79%) | 91% |
| Tunneling: read as "apart early" | 32 | 66% (48–80%) | 81% | 22 | 45% (27–65%) | 73% |
| Velocity late in starts | 6 | 33% (10–70%) | 100% | 4 | 75% (30–95%) | 100% |
| Third time through (verdict), since removed | 2 | 0% (0–66%) | 0% | 5 | 20% (4–62%) | 40% |

*Held* means the second half cleared the same bar on its own. *Same direction*
means it at least still pointed the same way. A second half too thin to
measure a finding in is left out, not counted as a miss. Thirty pitchers in
2025, twenty-eight in 2026.

The intervals are Wilson intervals, which count every finding as independent.
They are not quite: one pitcher can give two or three. The script prints the
same interval from resampling pitchers beside each, and for most rows the two
agree to within a few points. Two rows are wider than they are printed. The
"apart early" reading in 2026 is 22 findings from 8 pitchers, and by pitcher
its interval is 18–81%, not 27–65%; the follow-ups that came more than usual
in 2025 are 24–80%, not 31–74%.

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
consequence of one of them survived correction, so the report prints no run
estimate. The hittable-pitch finding fired for 2 of 30
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

Tunneling gets the same treatment, for every pair of pitch types thrown back
to back 30 times to one side in each half (116 pairs from 27 pitchers in
2025, 91 from 18 in 2026):

| A pair's | 2025 r | within pair type | Median move | 2026 r | within pair type | Median move |
|---|---|---|---|---|---|---|
| Offset out of the hand | 0.83 | 0.78 | 0.3 in | 0.93 | 0.93 | 0.3 in |
| Offset at the decision point | 0.72 | 0.57 | 1.3 in | 0.52 | 0.42 | 1.2 in |
| The same, over his scatter (d′) | 0.71 | 0.58 | 0.2 | 0.53 | 0.45 | 0.2 |
| Difference in late movement | 0.99 | 0.94 | 0.2 in | 0.98 | 0.94 | 0.2 in |
| Difference in flight time | 1.00 | 0.97 | 2.2 ms | 0.99 | 0.93 | 2.3 ms |
| Published form: mean distance at the decision point, above his floor | 0.57 | 0.43 | 0.9 in | 0.42 | 0.35 | 1.0 in |

*Within pair type* is the correlation left after each pair type's average
across pitchers is subtracted from both halves. A four-seam and a curveball
are far apart late for every pitcher alive, so a table of pairs correlates
with itself however little it says about anyone; the second column asks
whether this pitcher's four-seam and slider are tighter than other pitchers'
in both halves.

What that says: the parts of tunneling that repeat are the parts that were
already known. Where he lets each pitch go, how it moves and how fast it is
are his, and barely change. The part that is tunneling's own — how close two
pitches are when the hitter has to decide — repeats about as well as a
pitch's xwOBA does, because it depends on where he happens to throw them. It
is good to about an inch and a quarter. So it is in the evidence file with
its hold rates beside it, and not on the card.

```bash
python scripts/replication.py --season 2025
python scripts/replication.py --season 2026
python scripts/replication.py --season 2026 --cached   # local cache only, no network
```

These correlations are across a list built to span very different pitchers,
so they are higher than they would be across a random sample of the league.

## How few starts does it need?

Both checks above cut a major-league season in half, which hands each side
about 1,400 pitches from a starter. That is a college weekend starter's whole
year, and far more than anyone has on him in March. `scripts/sample_size.py`
asks what the report does with less.

Half of a starter's games are held out as the answer key. Reports are built
from 3, 5 and 10 of his other starts, and then from all of them, by the same
functions and at the same bars as the real report, and everything each one
would print is scored against the held-out games. The games are drawn at
random, 20 shuffles a season, so the only thing that changes from one column
to the next is the size. Run on every cached season of 20 or more starts: 33
seasons from 19 starters, 2025 and 2026 pooled.

**What one report prints**

| | 3 starts | 5 starts | 10 starts | Half season |
|---|---|---|---|---|
| Pitches | 277 | 461 | 925 | 1,348 |
| Lines on the card | 2.8 | 4.0 | 5.2 | 5.7 |
| Sequence graph: rows it can read | 1.5 | 3.3 | 5.9 | 6.9 |
| Reports that colour any follow-up | 4% | 12% | 40% | 59% |
| Reports that put a word on any tunneling pair | 7% | 32% | 84% | 96% |
| Reports where the count rule finds a tendency | 15% | 48% | 84% | 93% |

**How often what it printed held** (95% interval, resampling pitchers)

| | 3 starts | 5 starts | 10 starts | Half season |
|---|---|---|---|---|
| Hitter's count: what to sit on | 63% (49–74%) | 69% (59–78%) | 76% (66–84%) | 78% (68–86%) |
| Two strikes: the put-away pitch | 66% (60–72%) | 68% (63–75%) | 76% (70–81%) | 78% (72–84%) |
| Two strikes: lay off it low | 91% (80–97%) | 87% (76–93%) | 86% (77–93%) | 86% (76–92%) |
| Won't see | 89% (83–94%) | 94% (91–96%) | 96% (95–98%) | 97% (95–99%) |
| Read the height | 84% (67–96%) | 85% (70–95%) | 87% (72–97%) | 88% (74–97%) |
| Count tell | 90% (59–100%) | 90% (71–97%) | 92% (82–97%) | 95% (89–99%) |
| Sequence graph: more than usual | 12% (0–33%) | 35% (20–56%) | 57% (40–69%) | 67% (55–76%) |
| Sequence graph: less than usual | — | 33% (6–55%) | 54% (28–70%) | 61% (51–68%) |
| Tunneling: same window | — | 60% (27–81%) | 69% (52–81%) | 71% (59–80%) |
| Tunneling: apart early | — | 66% (29–84%) | 73% (60–82%) | 71% (59–80%) |
| Rule: count tendency | 36% (11–60%) | 58% (42–68%) | 67% (58–73%) | 69% (61–75%) |
| Rule: setup by pitch and height | 25% (0–56%) | 40% (12–56%) | 59% (18–74%) | 63% (43–74%) |

A dash is a claim fewer than six pitchers made at that size: 10 to 30 claims
out of 660 reports, which says something about those pitchers and nothing
about the rule. The two rows marked *Rule* are not lines the report prints.
They are rules of the findings engine — a pitch at 1.5 times his own rate in a
count, and a setup by pitch and height, each at three standard errors — which
the report still computes and no longer lists. They are scored here because
they are the best-measured answer to what a three-standard-error bar lets
through on a small sample: the count rule makes several times the claims the
sequence graph does.

**How big it looked, and how big it was** (median, sample → held-out games;
1.0 is his usual rate)

| | 3 starts | 5 starts | 10 starts | Half season |
|---|---|---|---|---|
| Rule: count tendency | 1.71 → 1.42 | 1.82 → 1.57 | 1.80 → 1.64 | 1.80 → 1.66 |
| Sequence graph: more than usual | 1.43 → 1.17 | 1.42 → 1.21 | 1.47 → 1.28 | 1.57 → 1.34 |

What that says, in three parts.

*The plan's lines survive a small sample.* A three-start card has half the
lines of a half-season one, and they are worth reading: the two that name a
pitch hold about two times in three against 78% (pooled; see the note on
seasons below), and the rest hold 84% to 91%. At no size did a plan line send the hitter to a different pitch more than
2% of the time (3% in the date-order control below). These lines are shares of
a situation he reaches in every start, so they settle early.

*What clears three standard errors does not.* A count tendency found on three
starts holds 36% of the time and a coloured follow-up 12%, against 69% and 67%
on a half season. The direction is nearly always right: at most 2% of count
tendencies and 7% of follow-ups came back on the other side of his usual rate.
What fails is the size. A count tendency found on three starts looks like one
found on a half season, 1.7 to 1.8 times his usual rate, but it comes back at
1.4 where the half-season one comes back near 1.7. On 280 pitches, what clears three
standard errors is a mix of real tendencies and ordinary ones caught on a good
run, and nothing in the sample tells them apart. A bar set in standard errors
keeps a small sample quiet. It cannot make the few things it does say the
right size.

*Sequencing needs about ten starts.* Before that the graph mostly says nothing:
4% of three-start reports and 12% of five-start reports colour a follow-up at
all. The shares it draws are another matter. A row is drawn only once the
pitch has 40 follow-ups to that side, so a share on a three-start graph sits
a median 3.7 points from the held-out share, against 3.0 on a half season;
there are just 1.5 rows of it where a half season has 6.9. Tunneling behaves
the same way and for the same reason: few pairs reach their 30 occurrences
before ten starts (0.2 a report on three, 0.9 on five, 3.8 on ten), and from
ten on the words hold as often as they ever do.

So for a staff with three starts on a pitcher: read the card's lines, read the
shares on the sequence graph, and treat anything coloured as a direction and
not a size. The colours start to mean something around his tenth start, which
is late in a college regular season.

**The control.** A real report is built from his first games and read against
his later ones, and he changes in between; shuffling hides that. So the
half-season sample was scored once more in date order, first half against
second, for the same 33 seasons.

| | Shuffled: held | By date: held | By date: claims |
|---|---|---|---|
| Hitter's count: what to sit on | 78% | 72% | 36 |
| Two strikes: the put-away pitch | 78% | 72% | 65 |
| Won't see | 97% | 94% | 49 |
| Sequence graph: more than usual | 67% | 62% | 34 |
| Tunneling: same window | 71% | 72% | 58 |
| Tunneling: apart early | 71% | 57% | 47 |
| Rule: count tendency | 69% | 69% | 90 |

On these pitchers a season's drift costs the plan's lines about six points,
from one split a season and a few dozen claims a row. That is the allowance to
make on the small-sample columns above, which were shuffled.

**Two seasons, one of them tuned.** Every cutoff was set on 2025, and the
tables above pool it with 2026 to have enough pitchers at the small sizes.
The climb with sample size is the same in both seasons; the level is not.

| Held | Season | 3 starts | 5 starts | 10 starts | Half season |
|---|---|---|---|---|---|
| Hitter's count: what to sit on | 2025 | 68% | 77% | 84% | 86% |
| | 2026 | 56% | 60% | 63% | 65% |
| Two strikes: the put-away pitch | 2025 | 68% | 72% | 77% | 79% |
| | 2026 | 63% | 65% | 75% | 77% |
| Won't see | 2025 | 89% | 94% | 97% | 97% |
| | 2026 | 89% | 93% | 96% | 97% |
| Rule: count tendency | 2025 | 41% | 61% | 72% | 72% |
| | 2026 | 26% | 50% | 59% | 64% |

So read the shape from the pooled tables and the level from 2026, the season
no cutoff had seen: on three starts the hitter's-count line holds a little
over half the time there, not two times in three.

What this does not show. These are major-league starters cut down to a
college-sized sample, not college pitchers. A college starter may be easier to
read — fewer pitches, less command, a simpler game called behind the plate —
or harder, and nothing here measures it. Relievers are left out, because ten
of their games are 150 pitches. And the bars were not moved for small samples:
this is the report as it stands, not the best report that could be written
from three starts.

```bash
python scripts/sample_size.py --season 2025 2026 --cached   # about five minutes
python scripts/sample_size.py --season 2026 --cached        # the untuned season alone
```

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
where the whole family is visible, and the replication script prints the
comparison count next to the finding count. On Yamamoto, sixteen comparisons
clear the raw threshold and none survive the correction, which is why the
report prints no run figure. (More than six clear it because the null is not
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

**Standard errors do not rescue a small sample.** The three-standard-error
bar was added so that a report checking 40 cells would not print the two that
cleared a lower bar by chance, and on a half season it does that. On three
starts it cannot do the rest of the job: what clears it is a mix of real
tendencies and ordinary ones on a good run. Scored against held-out games, a
count tendency keeps its direction and gives back about 40% of its distance
from his usual rate, against under 20% for one found on a half season
(see "How few starts does it need?"). The bar controls how often the report
speaks. It does not correct the size of what it says.

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
deviation near 0.19 against pattern-level differences of 0.01 to 0.02. Across
the 58 calibration reports one estimate in 287 survived the correction. So
the frequency pattern is the finding, and the report prints no run figure
rather than one that implies precision it does not have. It used to say so in
a sentence on every finding, and a caveat attached to everything is read by
no one.

**What does not repeat comes out.** A line in the plan is a prediction, and a
finding type is only worth its space if it comes back in games it was not
found in. The first-pitch take-or-swing advice was calibrated, gated on
sample size and unit-tested, and still predicted nothing on a new season, so
it was deleted rather than hedged. A verdict on the third trip through the
order (back in 1 of 7 cases) and a "hittable pitch" finding (fired in 5 of 58
pitcher-seasons) were removed the same way after being re-measured on the other half
of the season; the tables they were read from are still in the report, without
the conclusion. The ranked findings list went for a different reason: its
patterns repeat, but they are the patterns the card and the tables already
show, and the order it put them in came from a severity score with
hand-chosen weights. A number on a page that nobody can defend is worse than
no number, so the list is off the page and the engine behind it is kept for
the replication check.

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
