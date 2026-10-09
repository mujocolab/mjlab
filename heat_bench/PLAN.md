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

## The two temperature-dependent effects (implemented in Phase 1)

Implemented in `heat_bench/physics/motor_thermal.py` and used by
`ThermalEnergyObservation._electrical` (previously fixed constants). These
are two *distinct* physical mechanisms, easy to conflate:

- **`Rd(T)` — copper resistance rise.** Makes the *same current* generate
  *more heat* (`heat = I² · Rd(T)`). Does not by itself change how much
  current a given torque needs.
- **`Kt(T)` — magnet remanence fade.** Makes the *same torque* require
  *more current* (`I = τ / (N · Kt(T))`). Two separate NdFeB effects,
  corrected from an earlier draft that merged them as "fade above ~80°C":
  - *Reversible* fade at **all** temperatures, −0.12 %/°C for N42 (Arnold
    datasheet, measured 20–80°C), recovered on cooling. **Phase 1.**
  - *Irreversible* demagnetization above a grade's max operating
    temperature (~80°C for standard N grades). Accumulating damage —
    **Phase 2**, not modeled yet.

Since a real driver limits *current*, `Kt(T)` also lowers the joint's torque
ceiling: Phase 1's thermal derate is exactly `Kt(T)/Kt_spec` (≈0.93 at
80°C) — physics only, no invented protection curve.

Together they form a positive-feedback loop: hotter → `Kt` drops → more
current for the same torque → that current meets higher `Rd` → heat for the
same torque scales as `Rd(T)/Kt(T)²` (≈×1.4 at 80°C) → hotter still.
**This raises the equilibrium temperature but is only a true runaway above
a critical load**: the loop diverges when the extra heat per °C exceeds the
joint→chassis conductance, i.e. sustained per-joint heat
`P > 1/(Rth·(α_Rd + 2|α_Kt|)) ≈ 1/(2·0.0063) ≈ 80 W` (≈18 N·m continuous on
one joint). Normal walking is ~25 W on the hottest knee. Stopping a true
runaway needs torque reduction beyond physics — the user's planned Unitree
80°C shutdown condition, or a protective policy.

## Closed-loop behavior to verify (both directions)

The point of this whole system is that these two loops actually close
end-to-end without extra plumbing, because the monitor and the actuator
both read/write through the same realized-torque path
(`qfrc_actuator` is computed *after* any actuator override, and the
monitor reads it *after* the step). Concretely, verify both:

1. **Protective loop.** Hot joint → EventTerm reduces `forcerange` (Phase
   1: only by the physical `Kt` ratio, a weak brake; stronger reduction
   comes from a future shutdown condition) → policy's commanded torque
   gets clamped lower →
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
   — settling at a higher equilibrium below ~80 W/joint, accelerating
   only above it, until a shutdown/derating condition catches it or the
   terminal `dead` threshold is crossed (Phase 2, permanent).
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

### Phase 0 — New EventTerm scaffold (additive only) — DONE
Implemented in `heat_bench/envs_mjlab/actuator_health.py`, gated by
`actuator_health.enabled` (default on). Decisions made while building it:
- **Granularity: control step.** Step events run after the decimation loop
  and before observation compute, so the event reads temps from the end of
  the previous control step (20 ms lag) and its write governs every
  substep of the next one.
- **Baseline snapshotted at reset.** `reset()` runs after reset-mode DR
  events, so it copies the live per-env `forcerange` and each step writes
  `baseline × derate`. This resolves the DR write-ordering risk under
  "Event design" for `effort_limits`. `pd_gains` vs. gain writes for
  `free`/`locked` still needs the same treatment in Phase 3.
- The thermal obs term is resolved lazily on first call (the
  EventManager is built before the ObservationManager).

Original scope:
- Add `apply_actuator_health` EventTerm, wired into
  `go2_eval_env_cfg.py` alongside the existing `"thermal"` observation
  group. No changes to `ThermalEnergyObservation`.
- Add the per-(env, joint) failure-state buffer (derate factor + enum),
  with reset-on-episode-reset handling.
- No actual derating logic yet — just prove the plumbing (buffer exists,
  resets correctly, event runs every step, can write a no-op/identity
  `forcerange` equal to the existing `effort_limit`).

### Phase 1 — Temperature-dependent motor physics — DONE
- `Rd(T)` and `Kt(T)` (`physics/motor_thermal.py`), linear, cited
  coefficients (copper 0.00393/°C, NBS Misc. Pub. 17; NdFeB N42
  −0.12 %/°C, Arnold datasheet), both referenced to 20°C; nominal Rd/Kt
  assumed specified at 25°C (placeholder). Zeroing both coefficients
  recovers the old constant model exactly.
- Event writes `baseline × derate × thermal_derate`, with
  `thermal_derate = min(1, Kt(T)/Kt_spec)` recomputed every step from the
  previous step's joint temps. `derate` stays the external fault factor
  (scripted faults, future shutdown) so the two compose. Thermal fade never
  changes `state`.
- Decided against the originally planned taper curve: the user wants
  degradation physically faithful. Unitree's 80°C joint shutdown is a
  separate condition the user will add later.
- Target is long eval runs (temps rise over minutes; a 20 s episode barely
  warms a joint). Hot-start / short-episode support deferred to training.

### Battery → torque (voltage-limited torque-speed curve) — DONE
- Before this, the battery was one-way: a 2%-SoC run walked exactly like
  a full one. Now `apply_actuator_health` narrows each joint's
  current-limited range to what the bus voltage can drive at its speed:
  `upper = N·Kt·(+V − Kt·N·q̇)/Rd`, `lower = N·Kt·(−V − Kt·N·q̇)/Rd`
  (DC-equivalent motor, Ke = Kt in SI — energy-consistent, no new
  constant, inherits Phase 1's `Kt(T)`/`Rd(T)`). One-sided: back-EMF
  opposes torque in the direction of motion and helps braking.
- Voltage bounds are clamped *into* the current-limited range, so the
  written range is always valid (overspeed collapses to an edge; dead
  joints stay [0, 0]). Uses the battery model's sagged `bus_voltage`, so
  heavy load → sag → lower ceiling is a second feedback loop.
- Updated per control step from the post-decimation joint speed (lags
  fast swings by ≤ 20 ms; measured, see commit). On by default;
  `battery.voltage_limited_torque: false` restores current-limit-only.
- Paper constants unchanged (N 6.22, Kt 0.26, Rd 0.66): full torque up to
  ~14.8 rad/s at 33.6 V vs ~8.9 rad/s at 24 V.
- Deferred: a battery safety limit (BMS current/power cap or low-voltage
  cutoff), like the user's planned 80°C shutdown.

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
- Unitree 80°C shutdown condition (user-owned): trip/re-enable hysteresis
  and where it writes (`derate`/`state`), plus a cited source.
- Resolved: failure-decision granularity is the control step (Phase 0);
  `Rd(T)`/`Kt(T)` are linear with cited coefficients and the thermal derate
  is the physical `Kt` ratio (Phase 1).
