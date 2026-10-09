# heat_bench

A standalone benchmark for passively measuring thermal accumulation and
battery energy usage of **existing, already-trained** quadruped locomotion
policies running in [mjlab](https://github.com/mujocolab/mjlab). No new
training, residual policies, or reward terms — this package only reads
actuator state each control step, runs it through a thermal and a battery
model, and exposes the result as an mjlab observation group and a set of
episode metrics.

mjlab has no Go2 MJCF asset yet, so `heat_bench` simulates the existing
Unitree Go1 robot body while modeling **Go2's** electro-thermal hardware
constants on top of it (see `configs/go2_eval_config.yaml`). Swap in a real
Go2 MJCF later without touching the physics engines.

## What it does

- **`physics/lptn_engine.py`** — `BatchedLPTNEngine`: a batched 14-node
  lumped-parameter thermal network (12 actuators + 1 chassis + 1 ambient
  boundary node), integrated with forward Euler.
- **`physics/motor_thermal.py`** — `MotorThermalModel`: temperature-
  dependent phase resistance `Rd(T)` (copper, +0.393%/°C [[7]](#references))
  and torque constant `Kt(T)` (reversible NdFeB magnet fade, −0.12%/°C
  [[8]](#references)). A hot joint draws more current and makes more heat
  for the same torque (≈×1.4 heat at 80°C), and — since a motor driver
  limits current — has a lower torque ceiling. Set both
  `*_temp_coeff_per_c` values in the yaml to 0 to recover constant Rd/Kt.
- **`physics/battery_ecm.py`** — two swappable battery models, selected via
  `battery.model` in `configs/go2_eval_config.yaml`: `BatchedBatteryECM`
  ("rint", default), a fixed-resistance Rint model, and `AdvancedBatteryECM`
  ("rint_soc_aging"), which adds SoC-dependent internal resistance and a
  chassis-temperature-coupled capacity-loss/aging estimate. See "Related
  work" below for citations.
- **`envs_mjlab/eval_observations.py`** — `ThermalEnergyObservation`, the
  mjlab observation term wiring both engines into the sim loop. Joule heat,
  current, and mechanical power are accumulated at *physics-substep*
  resolution (every 0.005s physics step), then averaged before being fed
  to the engines once per 0.02s control step — not sampled once after the
  fact. See below for why.
- **`envs_mjlab/go2_eval_env_cfg.py`** — wraps the stock Go1 rough-terrain
  task, adding a read-only `"thermal"` observation group and metrics
  (joint temps, battery energy/SoC, distance) without touching rewards,
  actions, or the actor/critic observation groups a trained checkpoint
  depends on.
- **`envs_mjlab/actuator_health.py`** — `apply_actuator_health`, a
  `mode="step"` event term that owns per-(env, joint) actuator health
  (fault derate factor + discrete `ActuatorState`) and writes
  `actuator_forcerange` every control step as
  `baseline × derate × thermal_derate`. `thermal_derate = min(1,
  Kt(T)/Kt_spec)` is the physical torque-ceiling loss from magnet fade
  (≈0.93 at 80°C), recomputed every step and recovered on cooling;
  `derate` is the external fault factor (e.g. `--joint-fault`). With
  `battery.voltage_limited_torque` on (default), that range is narrowed to
  what the battery's bus voltage can drive at each joint's speed
  (back-EMF, `V = I·Rd + Kt·ω`): a drained or sagging pack clips fast
  motions first, while braking and low-speed torque are unaffected. The
  baseline is refreshed at reset (so it composes with effort-limit domain
  randomization). A joint that reaches `actuator_health.dead_temp_c`
  (85°C by default; `null` disables it) latches `DEAD` (zero torque) until
  the episode resets, even after cooling — a user-set physical-death
  threshold, not Go2's 80–85°C software shutdown. Toggle the whole event
  with `actuator_health.enabled` in the yaml. See `PLAN.md` for the phases
  that build on it.
- **`scripts/run_eval.py`** — headless batch evaluation of a checkpoint,
  dumping per-episode results to CSV/JSON (including `dead_joints` and
  `first_death_s` when a death threshold is set).
- **`scripts/play.py`** + **`viewer/`** — an interactive Viser-based viewer
  with a live "Robot Health" tab: per-node thermal, current, and torque
  charts (all nodes on shared axes, not just aggregates) plus a battery
  panel, for eyeballing what failure conditions actually look like before
  building any automatic detection.

## Why physics-substep heat averaging

A single post-decimation sample of `qfrc_actuator` misses torque
transients that happen mid-control-step — e.g. a foot impact on rough
terrain that spikes and relaxes within one 0.02s control step's four
0.005s physics substeps. Measured against a trained Go1 policy on rough
terrain (32 envs × 200 control steps), sampling only the last substep
understated mean I²R heat by ~7% and missed tail spikes by up to ~3x
(p99 relative error ~170%). `ThermalEnergyObservation` instead accumulates
heat and mechanical power every physics substep (by wrapping
`Simulation.step`, since mjlab has no official per-substep hook for
observation terms) and averages over the control step before updating
either engine.

## Related work

The 14-node topology, the 50Hz (control step) / 200Hz (physics substep)
update split, and the substep-averaged heat-input approach were
cross-checked against published thermal-aware quadruped locomotion
research, using the same Unitree-A1-class robot: [[1]](#references),
[[2]](#references). Both papers use the same 14-node LPTN (12 motors + 1
non-actuator node + ambient), updated at 50Hz synchronized with a 200Hz
PD/physics loop — matching this package's `env.step_dt`/`physics_dt`
split exactly. Their heat input is RMS torque over the 200Hz samples
inside each 50Hz interval; since heat ∝ I² and RMS(x)² = mean(x²), that's
mathematically the same operation as this package's per-substep
mean-of-I² accumulation. The one difference is discretization method:
both papers use zero-order-hold (exact, matrix-exponential)
discretization, kept here as forward Euler instead — benchmarked at
~140x the per-step cost of the current `bmm`-based Euler update for no
accuracy benefit, since the thermal time constant (`Rth*Cth ≈ 800s`) is
~40,000x the control step. Both papers' underlying thermal model
parameters (not just topology) come from a separate companion paper,
[[3]](#references) (not on arXiv).

The two battery models (`physics/battery_ecm.py`) were similarly
cross-checked. Model 1 ("rint") matches the battery topology used in the
one legged-robot-specific battery paper found, [[4]](#references) — their
Eq. 12–13 use the same voltage-source-plus-series-resistance topology
(solved in the opposite direction: given power demand, solve for current,
since their use case is MPC power allocation rather than passive
observation). Model 2's capacity-loss/aging term
(`Q_loss = B·exp((-Ea+α|I|)/(R·T))·(Ah)^z`) is their Eq. 18, which is
itself drawn from [[5]](#references). Model 2's battery constants
(capacity, voltage, series cell count, full-charge OCV) come from
Unitree's official Go2 spec, [[6]](#references).

## References

1. Qian et al., *"Learning Thermal-Aware Locomotion Policies for an
   Electrically-Actuated Quadruped Robot,"* [arXiv:2603.01631](https://arxiv.org/abs/2603.01631).
2. Wan et al., *"Learning to Balance Motor Thermal Safety and Quadrupedal
   Locomotion Performance with Residual Policy,"* [arXiv:2605.27046](https://arxiv.org/abs/2605.27046).
3. Lin, Qian, Luo, Liang, *"Temperature Distribution Prediction of the
   Quadruped Robot Based on the Lumped-parameter Thermal Networks,"*
   ROBOT journal, 2025 (not on arXiv).
4. Shu, Huang, Ren, Wu, Li, *"Learning-Based Model Predictive Control for
   Legged Robots with Battery–Supercapacitor Hybrid Energy Storage
   System,"* Appl. Sci. 2025, 15, 382, [10.3390/app15010382](https://doi.org/10.3390/app15010382).
5. Petit, Prada, Sauvant-Moynot, *"Development of an empirical aging model
   for Li-ion batteries and application to assess the impact of
   Vehicle-to-Grid strategies on battery lifetime,"* Appl. Energy 2016,
   172, 398–407.
6. Unitree, *Go2 battery specification* (BT2-05 "Standard Version"),
   <https://www.unitree.com/go2/battery> — data source, not a paper: 8S
   Li-ion, 8000mAh (236.8Wh), 29.6V nominal / 33.6V charge limit.
7. U.S. Bureau of Standards, *Copper Wire Card*, Miscellaneous
   Publication No. 17, 1919,
   <https://nvlpubs.nist.gov/nistpubs/Legacy/MP/nbsmiscellaneouspub17.pdf>
   — data source: annealed-copper resistance temperature coefficient
   0.00393 /°C at 20°C.
8. Arnold Magnetic Technologies, *N42 Sintered Neodymium-Iron-Boron
   Magnets* datasheet,
   <https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42-151021.pdf>
   — data source: reversible temperature coefficient of induction
   α(Br) = −0.12 %/°C, measured 20–80°C. Go2's actual magnet grade is
   unknown; N42 is a representative standard grade.

## Known placeholders

The motor/thermal constants (gear ratio, torque constant, phase
resistance, joint thermal capacitance/resistance) are given, real values
from the papers this thermal model is based on ([[1]](#references)–[[3]](#references))
— a Unitree-A1-class parameter set, not Go2 motor specs. They're kept as
one consistent set on purpose: the 12-joint topology is shared, so the
model's behavior carries over, and the thermal/battery models are meant to
be swapped as whole modules if upgraded. Don't replace single values with
Go2 datasheet numbers (Unitree's GO-M8010-6 manual lists ratio 6.33 and an
output-side torque constant of 0.639 N·m/A vs this set's 6.22 × 0.26 =
1.617): mixing sources while keeping the set's phase resistance gave ~6×
the Joule heat and a joint at 179°C within 3 minutes.
The battery pack's nominal voltage, capacity, series cell count, and
full-charge OCV are now also given, from Unitree's official Go2 battery
spec ([[6]](#references), BT2-05 "Standard Version"): an 8S Li-ion pack,
8000mAh (236.8Wh), 29.6V nominal / 33.6V charge limit.
Still placeholder, clearly tagged `# PLACEHOLDER` in
`configs/go2_eval_config.yaml`: the temperature at which Go2's Rd/Kt were
specified (assumed 25°C; the Rd(T)/Kt(T) coefficients themselves are cited
material constants [[7]](#references), [[8]](#references)), internal
resistance (not published by Unitree), the empty-pack OCV (standard 3.0V/cell Li-ion cutoff, not
Go2-specific), chassis thermal capacitance, and the convection-vs-velocity
coefficients -- pending either further datasheet digging or calibration
against real hardware telemetry (e.g. Unitree's per-motor
`MotorState.temperature`, a direct sensor reading and the best available
ground truth for tuning `Rth`/`Cth` once real logs exist).

## Optional scenario features

All disabled/default off unless you opt in, so existing eval/play runs are
unaffected. Config values live in `configs/go2_eval_config.yaml` unless
noted otherwise.

- **Payload** (`payload:` section) — a simulated backpack/load, via
  `dr.pseudo_inertia` (jointly randomizes mass, inertia, and COM; see
  `heat_bench/envs_mjlab/go2_eval_env_cfg.py`). Set `enabled: true` and an
  `alpha_range` (log mass-scale; see the inline comment for the kg→alpha
  formula). `mode: "reset"` samples a new payload every episode, `"startup"`
  fixes one for the whole run. Config-only right now — no CLI flag.
- **Push disturbance** (`--push-disturbance` on `play.py`) — re-enables
  mjlab's standard training-time push event (`push_by_setting_velocity`):
  an instantaneous, mass-independent `qvel` overwrite every 1–3s. No real
  force is computed, so there's nothing to visualize as an arrow.
- **Impulse disturbance** (`--impulse-disturbance` on `play.py`,
  `impulse_disturbance:` section for magnitude/timing) — a real force+
  torque wrench (`apply_body_impulse`) held for a sampled duration,
  respecting the robot's mass/inertia/contacts. Renders as a visible arrow
  in the viewer automatically (mjlab's built-in debug-vis, no extra code
  needed). Both disturbances can be combined.
- **Scripted joint fault** (`--joint-fault JOINT [JOINT ...]` on
  `play.py`) — drives the named joints through healthy → derated (torque
  limit ramps to 30% over 3–6s) → dead (zero torque at 10s), timed per
  episode, via `scripted_joint_fault` writing the actuator-health event's
  buffers. A controlled demo of the torque-limit path, not a physical
  failure model — Phase 1+ derive failure from tracked temperature.
- **Battery-limited torque** (`battery.voltage_limited_torque`, on by
  default) — set `false` to compare against the current-limit-only
  behavior, e.g. alongside a low `battery.initial_soc` to see what a
  depleted pack costs.
- **Battery model A/B comparison** (`--battery-model rint|rint_soc_aging`
  plus `--port` on `play.py`) — override `battery.model` for a single run
  without editing the yaml; launch two instances on different ports to
  compare side by side.

## Usage

```sh
# Headless batch eval against a trained checkpoint.
uv run python -m heat_bench.scripts.run_eval \
  --checkpoint-file logs/rsl_rl/go1_velocity/<run>/model_<N>.pt \
  --num-envs 64 --output-file results.csv

# Interactive viewer with live thermal/battery/torque monitoring.
uv run python -m heat_bench.scripts.play \
  --checkpoint-file logs/rsl_rl/go1_velocity/<run>/model_<N>.pt \
  --num-envs 4

# With disturbances and a specific battery model.
uv run python -m heat_bench.scripts.play \
  --checkpoint-file logs/rsl_rl/go1_velocity/<run>/model_<N>.pt \
  --num-envs 4 --push-disturbance --impulse-disturbance \
  --battery-model rint_soc_aging
```

## Tests

```sh
uv run pytest heat_bench/tests/ -v
```
