# heat_bench: stress-driven failure & degradation system

Design reference for the next phase of heat_bench work, written up after an
extended design discussion. Nothing here is implemented yet — this is a
plan to come back to session by session, not a description of current
code. See `heat_bench/README.md` for what's actually built today.

## Context

heat_bench currently *observes* thermal, battery, and torque state
passively against already-trained checkpoints — it never changes what the
robot can actually do, and stress conditions (terrain, disturbance,
payload) exist as separate, disconnected knobs. The goal is to close the
loop: build a system where varied environmental stress produces real,
physically-grounded actuator degradation and failure, emerging from
tracked state rather than being randomly injected — and to eventually
support training policies against this system, not just evaluating them,
so a policy can learn proactive protective behavior (e.g. backing off a
joint approaching its thermal limit) instead of reacting to a joint
that's already gone. The batched, GPU-native design discipline heat_bench
has followed since day one exists specifically so this remains possible.

## Design decisions already settled

- **Causes vs. consequence states.** Stressors (thermal runaway,
  mechanical impact, battery brownout) are *causes*; they converge onto a
  shared, small set of per-joint *consequence states*: `healthy` →
  `derated` (continuous, reversible) → `free` / `locked` (discrete,
  triggered) → `dead` (terminal, irreversible — the insulation-melt
  equivalent). Causes decide *which* state to enter and *when*; the state
  machine and its effect on the actuator is one shared mechanism.
- **Mechanism confirmed feasible, no new mjlab infrastructure needed.**
  `actuator_gainprm`/`actuator_biasprm` (stiffness — `free` = near-zero
  gains, `locked` = maxed gains + frozen `joint_pos_target`) and
  `actuator_forcerange` (torque ceiling — continuous derate, or zero for
  `dead`) are all per-env, per-step writable MuJoCo model fields at the
  cheapest recompute tier (`RecomputeLevel.none`) — the same pattern
  `dr.pd_gains`/`dr.effort_limits` already use
  (`src/mjlab/envs/mdp/dr/actuator.py`).
- **Purely additive architecture.** `ThermalEnergyObservation`
  (`heat_bench/envs_mjlab/eval_observations.py`) does **not change** for
  the failure mechanism itself. A **new, separate `EventTerm`** reads its
  cached `last_joint_temps` (same pattern the existing metric functions
  already use via `observation_manager.get_term_cfg(...).func`) and owns
  the actuator writes. This composes correctly with vectorized training
  rollouts, resets, and existing DR events, since it's a normal
  `EventTerm`, not a monkeypatch extension.
- **Deterministic thresholds, not injected hazard** — failure state is a
  function of tracked state, not a sampled random event. This is the
  core philosophical point from the original terrain discussion: failure
  should be a *consequence* of stress, not a scripted/randomly-timed
  event, so the platform can be used to study *prediction and prevention*
  of failure rather than just reaction to it.
- **`qfrc_actuator` is read-only.** All actuator intervention must happen
  *before* `sim.step()`, never as a post-hoc correction — confirmed via
  code investigation, no write path exists on that field.
- **Torque saturation today is already correct.** Go1's `effort_limit`
  (`go1_constants.py`, real Unitree spec numbers: 23.7 N·m hip/thigh,
  35.55 N·m knee) already clamps via `actuator_forcerange` natively in
  MuJoCo, and `qfrc_actuator` already reflects the *realized*, clamped
  torque — not what the policy asked for. Nothing to fix there; the new
  work is making that ceiling *dynamic* (temperature-dependent) instead
  of fixed.
- **`BuiltinPositionActuatorCfg` (what Go1/G1/YAM all use, unanimously —
  no mjlab robot uses `BuiltinDcMotorActuatorCfg`) has no native
  electrical or thermal model.** Adopting the DC-motor actuator would be
  heat_bench leading, not following existing convention, and needs Go2
  electrical parameters (resistance, inductance) we don't have. **Decision:
  stay on `BuiltinPositionActuatorCfg`** and implement all
  temperature-dependent effects on heat_bench's own Python side, reusing
  the existing `effort_limit`/`forcerange` lever rather than switching
  actuator families.

## The two temperature-dependent effects that must both be added

Both currently fixed constants in `_accumulate_substep`
(`eval_observations.py:64,114`: `current = tau / (gear_ratio * self._kt)`,
`joule_heat = current² * self._rd`) — neither depends on temperature
today. Confirmed via code read; this is not yet implemented anywhere.
These are two *distinct* physical mechanisms, easy to conflate:

- **`Rd(T)` — copper resistance rise.** Makes the *same current* generate
  *more heat* (`heat = I² · Rd(T)`). Does not by itself change how much
  current a given torque needs.
- **`Kt(T)` — magnet demagnetization fade** (NdFeB, above ~80°C). Makes
  the *same torque* require *more current* (`I = τ / (N · Kt(T))`), since
  the motor is less efficient at converting current to torque.

Together they form the actual runaway loop: hotter → weaker magnets
(`Kt` drops) → more current needed to hold the same commanded torque →
that larger current meets *higher* resistance (`Rd` up) → disproportionately
more heat → hotter still. This is a genuine positive-feedback loop with
**no built-in brake** — the only thing that can arrest it is the
derating `EventTerm` forcing `τ` down via `forcerange`, which forces
`current` down even as `Kt(T)` keeps falling. This is the mechanistic
reason the derating system isn't optional polish — without it, a joint
that starts losing `Kt` has no way to stop accelerating toward failure
in this model.

Both `Rd(T)` and `Kt(T)` are read-modify additions to the *existing*
`_accumulate_substep` current/heat computation (fixed constant →
temperature-dependent lookup against `self.thermal`'s last-known temps),
not part of the new EventTerm. Placeholder-tag both coefficients the same
way existing config placeholders are tagged, pending real Go2 data.

## Closed-loop behavior to verify (both directions)

The point of this whole system is that these two loops actually close
end-to-end without extra plumbing, because the monitor and the actuator
both read/write through the same realized-torque path
(`qfrc_actuator` is computed *after* any actuator override, and the
monitor reads it *after* the step). Concretely, verify both:

1. **Protective loop (Phase 1 derating).** Hot joint → EventTerm reduces
   `forcerange` → policy's commanded torque gets clamped lower →
   `_accumulate_substep` reads the now-smaller realized torque → smaller
   computed current/heat → LPTN's existing passive joint→chassis
   conduction (always active, proportional to `(T_joint - T_chassis)/Rth`,
   independent of current activity) cools the joint → temperature drops →
   derate factor relaxes (since Phase 1 is re-evaluated live every step,
   not latched) → torque headroom returns. This is the "back off a hot
   joint, get it back later" behavior the project is meant to enable.
2. **Runaway loop (no intervention, or intervention arrives too late).**
   Hot joint → `Kt(T)` drops → more current for same commanded torque →
   `Rd(T)` up → disproportionately more heat → hotter → repeat,
   accelerating, until either derating catches it (loop 1 kicks in) or
   the terminal `dead` threshold is crossed (Phase 2, permanent).
3. **Inactivity cooldown**, independent of both loops above: a joint
   given zero commanded torque (or forced to ~zero via derate/`free`)
   cools via the same always-on passive conduction path — no special
   case needed, this already falls out of the existing LPTN topology.

Verification approach (manual, via the viewer, not automated at first):
hold a joint artificially hot (e.g. inject a high joule_heat manually or
run a high-stress scenario) and watch, in the Robot Health tab, that (a)
current visibly rises for the same commanded torque as `Kt(T)` drops,
(b) `forcerange` derating visibly caps torque and current once the
threshold is crossed, and (c) temperature relaxes and torque headroom
returns once the stressor is removed and the joint idles.

## Difficulty & terrain design

From the original "level design" discussion: terrain, disturbance, and
payload should compose into named, graduated difficulty presets, since
failure only means something once stress is actually varied — a policy
run on flat ground with no disturbance will rarely generate the
thermal/mechanical stress needed to exercise any of the above.

- **Terrain**: mjlab already has `ROUGH_TERRAINS_CFG` (7 sub-terrain
  types), `STAIRS_TERRAINS_CFG`, `ALL_TERRAINS_CFG`, and the
  `terrain_levels_vel` curriculum (promotes/demotes per-env terrain
  difficulty based on distance walked vs. commanded). heat_bench doesn't
  currently correlate per-env `terrain_types`/`terrain_levels` with
  failure outcomes — worth logging alongside failure metrics once Phase 5
  (observation exposure) lands, so "which terrain caused this joint to
  fail" is answerable.
  - Ties into the *deterministic-thresholds-not-injected-hazard*
    principle: rough terrain doesn't need to "inject" impacts — it
    naturally produces the torque spikes that feed both the current
    thermal model and any future impact-triggered mechanical failure
    (Phase 3), for free, without any new mechanism.
- **Disturbance**: already implemented — `push_by_setting_velocity`
  (instantaneous qvel kick) and `apply_body_impulse` (real force+torque
  wrench, respects mass/inertia, debug-vis arrow). Impulse disturbance is
  the natural trigger source for Phase 3's mechanical failure states
  (sufficiently large impact → `free`/`locked` on the affected joint).
- **Payload**: already implemented via `dr.pseudo_inertia` (mass/inertia/
  COM jointly randomized). Increases baseline torque demand across all
  joints, indirectly raising baseline current/heat — a difficulty axis
  that stresses the thermal system without any impact events at all.
- **Difficulty presets (new work)**: bundle terrain config + disturbance
  parameters + payload settings into named tiers (e.g. `easy`/`moderate`/
  `severe`), building entirely on existing config sections — no new
  physics, just composition and a config/CLI way to select a tier. This
  is what makes A/B comparisons ("does the policy survive severe stress
  longer with derating enabled vs. disabled") meaningful and repeatable.

## Event design (failure injection)

- **New `EventTerm`**, e.g. `apply_actuator_health`, separate from
  `ThermalEnergyObservation`. Runs every step (granularity — physics
  substep vs. control step — decide during implementation; control step
  is likely sufficient since failure state shouldn't flicker at 200Hz).
- **Reads**: the observation term's cached `last_joint_temps` (read-only,
  no changes to the observation term).
- **Owns**: a new per-(env, joint) failure-state buffer — continuous
  derate factor (float, `[0,1]`) + discrete enum
  (`healthy`/`derated`/`free`/`locked`/`dead`) — and the actual
  `actuator_gainprm`/`biasprm`/`forcerange` writes.
- **Reset semantics**: buffer clears/rerandomizes on episode reset, same
  as `ThermalEnergyObservation`'s own temperature reset — needs explicit
  handling, easy to forget.
- **Write-ordering risk to check during implementation**: `dr.pd_gains`/
  `dr.effort_limits` already write these same fields for training-time
  domain randomization. If both a DR event and the failure event touch
  the same field for the same joint in the same step, whichever runs
  last in the event pipeline wins silently. Needs verification against
  mjlab's actual event ordering once this becomes relevant for training
  (not blocking for eval-only use, where DR is typically off).
- **Numerical stability**: discontinuous jumps (gains 0 → max, forcerange
  full → zero) are step-function changes to the dynamics. MuJoCo-warp is
  presumed robust to this but untested here — sanity-check for
  instability/NaN/contact-force spikes once implemented.

## Step-by-step phases

### Phase 0 — New EventTerm scaffold (additive only)
- Add `apply_actuator_health` EventTerm, wired into
  `go2_eval_env_cfg.py` alongside the existing `"thermal"` observation
  group. No changes to `ThermalEnergyObservation`.
- Add the per-(env, joint) failure-state buffer (derate factor + enum),
  with reset-on-episode-reset handling.
- No actual derating logic yet — just prove the plumbing (buffer exists,
  resets correctly, event runs every step, can write a no-op/identity
  `forcerange` equal to the existing `effort_limit`).

### Phase 1 — Reversible thermal derating + the two temperature effects
- Add `Rd(T)` and `Kt(T)` to `_accumulate_substep`'s current/heat
  computation (placeholder-tagged coefficients).
- Add the derate decision function in the new EventTerm: temperature →
  `forcerange` scale, flat below a threshold, tapering above it
  (placeholder curve).
- Verify the closed loop per the "Closed-loop behavior to verify"
  section above, via the Robot Health viewer tab.

### Phase 2 — Terminal thermal failure
- Insulation-melt-analogue threshold: once crossed, latch `dead`
  permanently for that (env, joint) — `forcerange = [0, 0]` reapplied
  every step regardless of temperature afterward.
- Metric: dead-joint count/flag per episode.

### Phase 3 — Mechanical failure states from disturbance
- Extend impulse disturbance (or add a new event) so a sufficiently
  large impact can trigger `free` or `locked` on the affected joint(s),
  reusing the same state buffer/write path (gain-zeroing for `free`,
  gain-maxing + frozen target for `locked`).
- Open question, not blocking Phases 0-2: are mechanical states
  reversible or also terminal? Decide when this phase starts.

### Phase 4 — Difficulty presets
- Named tiers bundling terrain config + payload + push/impulse
  disturbance parameters, building on existing config sections
  (`ROUGH_TERRAINS_CFG`/`STAIRS_TERRAINS_CFG`, `payload`,
  `impulse_disturbance`). Config/CLI composition only, no new physics.
- Log per-env terrain type/level alongside failure outcomes, so failure
  causes are attributable to specific stress conditions.

### Phase 5 — Observation exposure
- Add failure-state fields (derate factor, discrete state, dead-joint
  count) to the existing `"thermal"` observation group, batched, no
  Python loops — this is the fork point for eventual policy training.

## Non-goals (this pass)

- Velocity-dependent torque-speed derating (voltage-saturation curve).
- Native electrical actuator model (`BuiltinDcMotorActuatorCfg`
  adoption) — explicitly decided against; see Design decisions.
- Calibrating thresholds/curves against real hardware telemetry — ships
  with clearly `# PLACEHOLDER`-tagged values, same standard as existing
  config.
- Actually training a policy against this system — this phase builds and
  validates the mechanism under eval; training is a follow-on effort.

## Verification checklist (per phase, before moving on)

- `uv run pytest heat_bench/tests/ -q`, `uv run ty check`, `make format`.
- Manual check via `heat_bench/scripts/play.py` + Robot Health tab: induce
  the relevant stress (hold a joint hot, trigger a large impulse) and
  visually confirm the new state responds and, where reversible, recovers.
- No full training run in this phase — eval/viewer verification only.

## Open questions to resolve when each phase starts

- Mechanical failure states (Phase 3): reversible or terminal?
- Failure-decision granularity: physics substep or control step?
- Exact placeholder curve shapes for `Rd(T)`, `Kt(T)`, and the derate
  function — need at least a plausible shape before Phase 1 can be
  verified qualitatively, even without real calibration data.
