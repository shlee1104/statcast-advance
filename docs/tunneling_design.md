# Pitch Tunneling — Design Notes (Phase 2.5)

Status: built, except league context. Release consistency is in the report.
The trajectory reconstruction and the pair metrics are in
`src/metrics/tunneling.py`. The split-half check is in
`scripts/replication.py`, and it decided where the result goes: a table in
the evidence file, under the sequence graph, and nothing on the card.

The cache schema (`store.PITCH_SCHEMA`) has the seven columns the
reconstruction reads: `vx0`, `vy0`, `vz0`, `ax`, `ay`, `az` and
`release_pos_y`. A player-season cached before they were added holds nulls
there until it is fetched again (`--refresh`), and the report then says so in
place of the table.

---

## The idea

A hitter must commit to his swing roughly 150–175 ms before contact, which is
about **23.8 feet** from home plate. Two pitches that are still travelling
together at that moment are indistinguishable at the only instant the decision
can be made. Whatever they do afterward is free deception.

Tunneling measures that: how similar a pitch pair looks at the commit point,
relative to how differently it finishes.

---

## What Statcast gives us

Savant ships a **9-parameter trajectory fit** per pitch, not just summary
statistics. Confirmed present in our fixture:

| Field | Meaning |
|---|---|
| `release_pos_x`, `release_pos_y`, `release_pos_z` | release point, feet |
| `vx0`, `vy0`, `vz0` | velocity components at the y = 50 ft plane |
| `ax`, `ay`, `az` | acceleration components (drag + Magnus + gravity) |
| `plate_x`, `plate_z` | location at the plate |
| `arm_angle` | arm slot in degrees |
| `release_extension` | how far toward the plate he releases |

With constant acceleration, this is a closed-form trajectory. We can evaluate
the ball's position at **any** distance from the plate, which is the whole
requirement.

### Coordinate system

- `y` is measured from the back of home plate, increasing toward the mound
- The reference plane for the velocity and acceleration terms is `y = 50`
- `vy0` is negative (the ball moves toward decreasing `y`)
- `x` is horizontal (positive toward the first-base side from the catcher's view)
- `z` is vertical

### Position at time t

$$p(t) = p_0 + v_0 t + \tfrac{1}{2} a t^2$$

### Solving for the time at a given distance

To find when the ball reaches a plane `y = y*`:

$$\tfrac{1}{2} a_y t^2 + v_{y0} t + (50 - y^*) = 0$$

$$t = \frac{-v_{y0} - \sqrt{v_{y0}^2 - 2 a_y (50 - y^*)}}{a_y}$$

Take the root giving `t > 0` for `y* < 50`.

### Recovering x0 and z0

Savant does not ship the `x0` / `z0` columns, so the position at the `y = 50`
plane has to be reconstructed from the release point.

Release happens at `y = release_pos_y` (roughly 54 ft), which is *before* the
reference plane, so the solved time is negative:

$$x_0 = \text{release\_pos\_x} - v_{x0} t_r - \tfrac{1}{2} a_x t_r^2$$

Same form for `z0`. Do this once per pitch and cache it.

---

## Metrics

All are computed for a **pair** of pitches: two consecutive pitches within the
same plate appearance. Each is the distance between the two in the `(x, z)`
plane the hitter looks into, in inches.

| Metric | Evaluated at | What it says |
|---|---|---|
| **Release** | the release point, y ≈ 54 ft | did they leave the same slot |
| **Tunnel** | y = 23.8 ft | how far apart at the decision point |
| **Plate** | y = 1.417 ft (front of plate) | how far apart they finished |
| **Late movement** | decision point to plate | how differently they moved after the decision point |

Tunnel, plate and the 23.8-foot decision point are Baseball Prospectus's 2017
definitions. Late movement is this project's reading of their "break
differential": for each pitch, where it finished against where it was headed
in a straight line from the decision point, which under constant acceleration
is ½·a·Δt²; then the distance between those two for the pair.

### What was planned and did not survive the data

This document first made **plate ÷ tunnel** the headline, as a "break:tunnel
ratio". Two things were wrong with that.

It is not Baseball Prospectus's break:tunnel ratio, which divides the break
differential, not the plate differential, by the tunnel differential.

And it is mostly geometry. For the *same pitch* thrown twice in a row —
Yamamoto's four-seam after a four-seam, 254 pairs — the ratio is 1.7. Two
straight lines from one release point are 1.7 times further apart after 53
feet than after 30, so 1.7 is what no deception at all looks like, and every
pair's ratio is that number give or take (1.1 to 2.6 on the fixtures).

### The floor

What replaced it is a reference the pitcher supplies himself. Two consecutive
pitches of the same type are about 8 inches apart at the decision point (8.1
for Yamamoto to either side, 9.0 and 9.7 for Skubal), because he does not
throw to one spot. That is what "the same pitch" looks like to a hitter. A
different pitch that comes through 9 inches from the last one is, at that
moment, inside the spread of the pitch he just saw.

So a pair is read as two numbers: how far its tunnel separation sits above
that floor, and how much it moves late.

| Yamamoto 2025, to lefties | n | Release | Tunnel | vs floor | Late | Slower by |
|---|---|---|---|---|---|---|
| Four-seam → cutter | 72 | 1.5 in | 8.3 in | +0.2 | 2.9 in | 16 ms |
| Four-seam → splitter | 141 | 1.6 in | 9.2 in | +1.1 | 3.5 in | 18 ms |
| Four-seam → curveball | 119 | 3.4 in | 11.8 in | +3.7 | 10.5 in | 97 ms |
| Curveball → splitter | 82 | 4.0 in | 14.3 in | +6.2 | 8.0 in | −79 ms |

The cutter and the splitter come through the fastball's window and then leave
its line by three inches. The curveball moves the most late, and is the
easiest to see coming: twice as far from the fastball at release, four inches
outside the floor at the decision point, and a tenth of a second slower.

### Flight-time differential

A secondary term worth computing: the difference in total flight time between
the two pitches. Two offerings can tunnel perfectly in space and still be
easy to separate if one arrives 80 ms later. This is why a slow curve rarely
tunnels effectively regardless of its path.

---

## Integration with sequencing

This is what makes the module worth building rather than a novelty.

`sequencing.transitions()` answers **which** pairs he throws, to each batter
side, with the count behind each share — it is what the report's sequence
graph draws. Tunneling answers **which of those pairs are deceptive**. Joined:

> After a four-seam, lefties get the splitter 29% of the time, and it comes
> through an inch outside the fastball's own spread.
> They get the curveball 25% of the time, four inches outside it and a tenth
> of a second slower — that one is readable.

Neither number means much alone. A high-frequency sequence that tunnels
poorly is an exploitable tell; a beautifully tunneled sequence he throws twice
a year is trivia. **Report pairs ranked by frequency × tunnel quality**, not
by tunnel quality alone.

---

## Baselines

Raw ratios are uninterpretable without context. The league-baseline sample
(stratified dates across the season) should also aggregate tunnel metrics by
pitch-type pair, so the report can say "88th percentile among FF→FS pairs"
rather than "14.2".

---

## Build order

**Step 1 — release consistency (cheap, do first).**
Standard deviation of `release_pos_x`, `release_pos_z`, and `arm_angle`
grouped by pitch type. If the slider releases two inches lower than the
fastball, he is tipping. Roughly five lines, immediate scouting value, and no
trajectory reconstruction required.

**Step 2 — trajectory reconstruction. Built.**
`tunneling.position_at(frame, y)` and `tunneling.flight(frame)`. The
self-check holds: run to the front of the plate, the fit lands on Savant's
`plate_x` / `plate_z` within 0.07 of an inch on all 6,400 fixture pitches,
which is the rounding of the published values. Reading the release columns as
positions on the 50-foot plane instead misses by a median of two inches, so
the check also settles which plane they are on.

**Step 3 — pairwise metrics. Built.**
`tunneling.pairs(frame)` reuses the plate-appearance pairing in
`sequencing.py` and returns the four separations and the flight-time
difference for every consecutive pair. `tunneling.pair_table(frame)` averages
them by batter side and pitch-type pair, gates on 30 pairs, and attaches the
floor.

**Step 4 — the check, and the report. Built; league context is not.**

1. The 58 cached pitcher-seasons were fetched again with the seven columns.
2. `scripts/replication.py` re-measures every pair that occurs 30 times to
   one side in each half (116 pairs from 27 pitchers in 2025, 91 from 18 in
   2026). See "What the check said" below.
3. The table is in the evidence file: `report._build_tunneling`, with its
   headline from `gameplan.tunnel_headline`.
4. Not built: where a pair sits among every pitcher's version of that pair,
   from the league sample. It needs the league sample fetched again too.

---

## From mean distances to offsets

The table above averages distances, as the published metrics do, and that
average is mostly scatter. The mean distance from Yamamoto's four-seam to the
splitter after it is 9.2 inches against a floor of 8.1, which reads as "the
same".

Averaging the *difference* first — across and up separately — and then taking
its length cancels the scatter and leaves where the follow-up sits relative
to the pitch before it. For that pair it is 4.4 inches, nearly all of it
down. For the four-seam and the curveball it is 7.7 inches *up*: the hump out
of the hand, before the curveball finishes 10 inches below.

The scatter then gets its own number: how far one of his pitches lands from
where that pitch type usually is at the decision point, 4.6 inches along one
axis for Yamamoto. Offset over scatter is a discriminability index, d′. At
1.0, where the ball is at that moment picks the right pitch of the two about
69% of the time; at 0 it is a coin flip. The report calls a pair **same
window** at 0.75 or less and **apart early** at 1.25 or more, and gives the
ones between no word.

## What the check said

Between halves of a season, for the same pair to the same side:

| A pair's | 2025 r | within pair type | 2026 r | within pair type | Median move |
|---|---|---|---|---|---|
| Offset out of the hand | 0.83 | 0.78 | 0.93 | 0.93 | 0.3 in |
| Offset at the decision point | 0.72 | 0.57 | 0.52 | 0.42 | 1.2–1.3 in |
| The same, over his scatter (d′) | 0.71 | 0.58 | 0.53 | 0.45 | 0.2 |
| Difference in late movement | 0.99 | 0.94 | 0.98 | 0.94 | 0.2 in |
| Difference in flight time | 1.00 | 0.97 | 0.99 | 0.93 | 2.2 ms |
| Mean distance at the decision point, above his floor | 0.57 | 0.43 | 0.42 | 0.35 | 0.9–1.0 in |

"Within pair type" is the correlation after each pair type's average across
pitchers is removed, which is the test of whether a number is about this
pitcher and not about fastballs and curveballs in general.

Three things follow.

**The offsets repeat better than the published mean distances** (0.52–0.72
against 0.42–0.57), which is the reason the report uses them.

**What repeats is what was already known.** Where he lets each pitch go, how
it moves and how fast it is are his, and hardly change. The number that is
tunneling's own — how close the two are when the hitter decides — repeats
about as well as a pitch's xwOBA, because it depends on where he throws them
as well as how.

**So it is not on the card.** The card's numbers repeat at 0.73 and up. The
words do better than the number: of 74 pairs read as "same window" in a first
half, 91% stayed on that side in the second and 69% earned the word again; of
54 read as "apart early", 78% and 57%. Both bars were set after looking at
both seasons, so those rates describe the fit.

One alternative was tried and not kept: comparing each pitch with his main
fastball over *all* pitches to a side, not only consecutive ones. In a
one-off run during development the decision-point offset repeated at 0.73
across 173 rows, a little better and not enough to give up the link to the
sequence graph. That run is not in the replication script.

---

## Honest limitations

**Predictive value is contested.** Tunneling metrics correlate with outcomes
more weakly than their intuitive appeal suggests. A pair that tunnels
perfectly is worthless if both pitches are hittable. This belongs in the
report as a descriptive section, not as a verdict.

**The commit point is a modeling assumption, and it is a distance where the
hitter's deadline is a time.** 23.8 feet is 172 ms before the plate for
Yamamoto's four-seam and 215 ms for his curveball. The other reading compares
both pitches at one instant: 175 ms before the *faster* of the two arrives,
since that is the pitch the hitter has to be on time for. Across the 51 pair
types on the fixtures the two readings rank pairs almost identically
(Spearman 0.96). They differ where it would be expected: curveball pairs are
3 to 4 inches further apart under the timed reading, and changeup pairs about
an inch closer. The standard distance is kept so the numbers mean what they
mean elsewhere, and it understates how readable a slow curveball is.

**Deception is not only geometric.** Arm speed, grip visibility, and prior
familiarity all matter and none are in this data.

**Consecutive-pitch pairing is a simplification.** A hitter's expectation is
shaped by the whole plate appearance and by previous at-bats, not just the
last pitch. Pairwise is the tractable version, not the complete one.

---

## Sources

The tunnel point (23.8 feet, about 175 ms before contact), tunnel
differential, plate differential and break:tunnel ratio are Baseball
Prospectus's, from their 2017 pitch-tunnel work:

- https://www.baseballprospectus.com/?p=31030
- https://baseballprospectus.com/?p=31032

Their exact formula for the break differential is not reproduced here; "late
movement" above is this project's own definition and is named differently for
that reason.
