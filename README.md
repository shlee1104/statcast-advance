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

In development. The data pipeline and metrics layer are working end to end
against real season data; the report layer is not yet built.

**Complete**

- Baseball Savant API client — retry, rate limiting, name-to-MLBAM resolution
- DuckDB cache — typed 40-column schema, primary-key upserts, freshness checks
- Cleaning layer — pitch-type consolidation, non-competitive pitch removal,
  game-type filtering, count validation
- Count metrics — usage by count, first-pitch tendencies, normalized-entropy
  predictability scoring
- Sequencing metrics — transition matrices, outcome-conditioned transitions,
  two-strike putaway, setup-pitch lift analysis
- 79 unit tests, plus an end-to-end validation suite over live data

**Next**

- League baselines via stratified date sampling, so findings can be stated as
  deviations from league norms rather than raw rates
- Arsenal and platoon/fatigue splits
- Weakness-flag rules engine
- Single-file interactive HTML report
- Pitch tunneling (see [docs/tunneling_design.md](docs/tunneling_design.md))

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

The single-command report generator is the next milestone:

```bash
python -m src.cli --pitcher "Yoshinobu Yamamoto" --season 2025    # not yet built
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
raises curveball frequency 59% above baseline (n=89), and the effect holds when
the count is held fixed — so it reflects sequencing intent rather than count
logic.

## Report contents

- **Arsenal** — velocity, spin, movement, usage, and whiff rate by pitch type
- **Count tendencies** — pitch mix across all 12 counts, plus a normalized-entropy
  predictability score identifying the counts where selection is most anticipatable
- **Sequencing** — within-plate-appearance pitch transition matrices, variants
  conditioned on the previous pitch outcome, two-strike putaway mix, and
  setup-pitch detection keyed on pitch type and location band
- **Splits** — platoon splits, velocity decay within outings, times-through-order
- **Key takeaways** — up to five auto-generated findings, ranked by severity, each
  carrying its sample size and the league baseline it deviates from

## Design notes

**Sample gates are a feature.** Every flag has a minimum-n threshold defined in
`config.yaml`, and every claim in the report states its sample size. A scouting
report that confidently asserts a tendency off 6 pitches is worse than no report,
because a coach may act on it.

**League baselines are the comparison layer.** "Throws a slider 40% in 1-2" means
nothing alone. "Throws a slider 40% in 1-2, against a league rate of 22%" is
actionable.

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
