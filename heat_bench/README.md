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
- **`physics/battery_ecm.py`** — `BatchedBatteryECM`: Coulomb counting,
  state of charge, and voltage sag under load.
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
- **`scripts/run_eval.py`** — headless batch evaluation of a checkpoint,
  dumping per-episode results to CSV/JSON.
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
research, using the same Unitree-A1-class robot:

- Qian et al., *"Learning Thermal-Aware Locomotion Policies for an
  Electrically-Actuated Quadruped Robot,"* [arXiv:2603.01631](https://arxiv.org/abs/2603.01631).
- Wan et al., *"Learning to Balance Motor Thermal Safety and Quadrupedal
  Locomotion Performance with Residual Policy,"* [arXiv:2605.27046](https://arxiv.org/abs/2605.27046).

Both papers use the same 14-node LPTN (12 motors + 1 non-actuator node +
ambient), updated at 50Hz synchronized with a 200Hz PD/physics loop —
matching this package's `env.step_dt`/`physics_dt` split exactly. Their
heat input is RMS torque over the 200Hz samples inside each 50Hz interval;
since heat ∝ I² and RMS(x)² = mean(x²), that's mathematically the same
operation as this package's per-substep mean-of-I² accumulation. The one
difference is discretization method: both papers use zero-order-hold
(exact, matrix-exponential) discretization, kept here as forward Euler
instead — benchmarked at ~140x the per-step cost of the current
`bmm`-based Euler update for no accuracy benefit, since the thermal time
constant (`Rth*Cth ≈ 800s`) is ~40,000x the control step. Both papers'
underlying thermal model parameters (not just topology) come from a
separate companion paper, Lin, Qian, Luo, Liang, *"Temperature
Distribution Prediction of the Quadruped Robot Based on the
Lumped-parameter Thermal Networks,"* ROBOT journal, 2025 (not on arXiv).

## Known placeholders

Go2's motor/electrical constants in `configs/go2_eval_config.yaml`
(gear ratio, torque constant, phase resistance, joint thermal
capacitance/resistance) are given, real values. Everything else —
battery pack capacity/OCV/internal resistance, chassis thermal
capacitance, and the convection-vs-velocity coefficients — is a
placeholder, clearly tagged `# PLACEHOLDER` in the config, pending either
a real Go2 datasheet or calibration against real hardware telemetry
(e.g. Unitree's per-motor `MotorState.temperature`, which is a direct
sensor reading and the best available ground truth for tuning `Rth`/`Cth`
once real logs exist).

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
```

## Tests

```sh
uv run pytest heat_bench/tests/ -v
```
