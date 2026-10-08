# Can sampling MPC walk G1 at 1.5 m/s? (gate before MPC → PPO on G1)

Date: 2026-09-29. Branch `mpc-stage1`.

MPC-DAgger and MPC-Inject need an MPC teacher that can do the task. Before
comparing them with PPO on `Mjlab-Velocity-Flat-Unitree-G1` (random velocity
commands that include 1.5 m/s), this checks the MPC alone.

## Prerequisites done

- **State sync** (`mpopi_train.mpc.state_sync`): the planner copies commands and
  timers, previous actions, stateful reward terms, sensor histories, entity
  data and domain-randomized model fields. On G1 the planner's per-term
  rewards match the real env to about 1e-5; copying only the simulator state
  gave errors of order 1 (`tests/test_mpc_state_sync.py`).
- **Smooth planner noise** (`num_knots`): with independent per-step noise every
  perturbed sequence scored far below the unperturbed plan (0.24 vs 0.72 over
  10 steps at noise 0.3), mainly from the action-rate penalty over 29 joints.
  Knot noise and a smaller scale narrow the gap (best perturbed 0.70 vs 0.72 at
  noise 0.1, 3 knots, 16 samples, CPU). No perturbed sequence beat standing
  still in that CPU probe: starting to walk first costs posture and upright
  reward before velocity tracking pays off, a local optimum for a short
  horizon.

## Pre-registration (written before the GPU run)

`src/mpopi_train/scripts/eval_velocity.py` (`mpopi-eval`): `play` env, commanded forward speeds 0.5,
1.0 and 1.5 m/s (no lateral or yaw command), 8 envs × 250 steps (5 s), eval
seed 10000; speed averaged after the first 50 steps.

| Planner | Samples × iterations | Horizon | Noise | Knots |
|---|---|---|---|---|
| P1 MPPI | 64 × 1 | 20 | 0.3 | none (per step) |
| P2 MPPI | 64 × 1 | 20 | 0.1 | 4 |
| P3 MPPI | 128 × 1 | 40 | 0.2 | 4 |
| P4 MPOPI | 32 × 4 | 40 | 0.2 | 4 |

Baseline: zero actions (default standing pose).

**Pass criterion (MPC is usable as a teacher):** some planner reaches a mean
forward speed of at least 1.2 m/s at the 1.5 m/s command with at most 0.25
falls per env in 5 s. If none passes, MPC → PPO on G1 is not pursued with this
planner; the options are then a better planner (longer horizon, more samples,
MPC-specific cost) or a different teacher.
