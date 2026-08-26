# Pitch Tunneling — Design Notes (Phase 2.5)

Status: designed, not yet built. Blocked on `metrics/sequencing.py`, which
supplies the plate-appearance pairing logic this module reuses.

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

All are computed for a **pair** of pitches — in practice, two consecutive
pitches within the same plate appearance.

| Metric | Evaluated at | Direction |
|---|---|---|
| **Release differential** | y ≈ 54 ft | smaller is better — same slot |
| **Tunnel differential** | y = 23.8 ft | **smaller is better** |
| **Plate differential** | y = 1.417 ft (front of plate) | **larger is better** |
| **Break:tunnel ratio** | plate ÷ tunnel | **higher is better** |

Each differential is the Euclidean distance in the `(x, z)` plane at that
depth:

$$d = \sqrt{(x_1 - x_2)^2 + (z_1 - z_2)^2}$$

**Break:tunnel ratio is the headline number.** It captures the actual
question — do these two pitches look the same when it matters and end up in
different places.

### Flight-time differential

A secondary term worth computing: the difference in total flight time between
the two pitches. Two offerings can tunnel perfectly in space and still be
easy to separate if one arrives 80 ms later. This is why a slow curve rarely
tunnels effectively regardless of its path.

---

## Integration with sequencing

This is what makes the module worth building rather than a novelty.

`sequencing.transition_matrix()` answers **which** pairs he throws.
Tunneling answers **which of those pairs are deceptive**. Joined:

> FF → FS is 34% of his sequences and tunnels at 14:1.
> FF → CU is 11% and tunnels at 3:1 — that one is readable.

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

**Step 2 — trajectory reconstruction.**
`reconstruct_trajectory()` returning `(x0, z0)` and a `position_at(y)`
evaluator. Unit-test against `plate_x` / `plate_z`: evaluating at the plate
must reproduce Savant's own reported values to within a few thousandths of a
foot. That is a strong self-check — if it matches, the physics is right.

**Step 3 — pairwise metrics.**
Reuse the PA-pairing logic from `sequencing.py`. Compute the four
differentials per pair.

**Step 4 — aggregation and ranking.**
Group by pitch-type pair, gate on minimum sample, rank by frequency ×
quality, join to league baselines.

---

## Honest limitations

**Predictive value is contested.** Tunneling metrics correlate with outcomes
more weakly than their intuitive appeal suggests. A pair that tunnels
perfectly is worthless if both pitches are hittable. This belongs in the
report as a descriptive section, not as a verdict.

**The commit point is a modeling assumption.** 23.8 ft is a reasonable
league-average approximation of when a hitter must decide. It varies by
hitter, by pitch velocity, and by count. Treat it as a convention, and state
it in the report rather than hiding it.

**Deception is not only geometric.** Arm speed, grip visibility, and prior
familiarity all matter and none are in this data.

**Consecutive-pitch pairing is a simplification.** A hitter's expectation is
shaped by the whole plate appearance and by previous at-bats, not just the
last pitch. Pairwise is the tractable version, not the complete one.
