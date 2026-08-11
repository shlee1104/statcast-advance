# statcast-advance

Automated advance scouting reports from MLB pitch-level data.

Player name in, coach-ready interactive report out.

## What this is

As a analytics intern with the UC Davis D1 baseball team, I produced advance
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

In development. Phase 0 (setup) complete.

## Setup

Requires Python 3.11 or 3.12.

```bash
git clone https://github.com/<your-username>/statcast-advance.git
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

Not yet implemented. Target interface:

```bash
python -m src.cli --pitcher "Tarik Skubal" --season 2025
```

## Report contents

- **Arsenal** — velocity, spin, movement, usage, and whiff rate by pitch type
- **Count tendencies** — pitch mix across all 12 counts, plus a normalized-entropy
  predictability score identifying the counts where selection is most anticipatable
- **Sequencing** — within-plate-appearance pitch transition matrices, including
  variants conditioned on the previous pitch outcome, and two-strike putaway mix
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

## Data source

[Baseball Savant](https://baseballsavant.mlb.com/) (MLB Statcast), accessed via
its public CSV search endpoint. Player identity resolution uses
[pybaseball](https://github.com/jldbc/pybaseball). Requests are rate-limited;
please do not remove the delays in `config.yaml`.

## License

MIT
