# Sampling MPC in mjlab: stage 1 (MPC as a controller)

Date: 2026-09-26. Branch `mpc-stage1`. CPU only.

Stage 1 of the MPC → PPO plan: a sampling-based MPC (MPPI, with an MPOPI
option) that runs on a batched copy of an mjlab task, checked as a standalone
controller before it is connected to PPO.

## How it works

`mpopi_train.mpc.SamplingMpc` keeps a planning copy of the task with
`num_real × num_samples` worlds (auto-reset and terminations off). At every
control step it:

1. copies `qpos`, `qvel`, `act`, `qacc_warmstart` and `ctrl` of each real env
   into that env's `num_samples` planning worlds;
2. rolls out `num_samples` perturbed action sequences (sample 0 is the
   noise-free nominal plan) for `horizon` steps and sums the task's own
   rewards;
3. weights the sequences with MPPI weights `w ∝ exp((R − max R)/λ)`, where λ
   is chosen per env by bisection so that the normalized ESS equals
   `target_ess`, and averages them;
4. with `iterations > 1` (MPOPI), also adapts a per-dimension std between
   batches (clamped to `[0.2, 3] × noise_std`);
5. returns the first action and shifts the plan one step for warm start.

Actions are in the policy action space, so MPC and PPO actions are comparable.
The planner uses its own random generator and restores the global RNG state
around planning-env construction (mjlab env construction reseeds it).

For Cartpole, copying those five fields reproduces the real env **exactly**:
the planning worlds match the real env's `qpos` and rewards bit for bit over
several steps (`tests/test_mpc_sampling.py`). Tasks with stateful managers
(commands, action history, actuator delays) will need more state copied.

## Result (criterion fixed before running)

Criterion: normalized score ≥ 0.9 on 16 play envs × 200 steps (10 s), eval
seed 10000, the same starts for every controller. Score = mean per-step
reward / 0.05.

| Controller | Normalized score |
|---|---|
| **MPC (MPPI, K = 64, H = 20, noise 0.5, target ESS 0.1)** | **0.973** ✅ |
| Zero action | 0.695 |
| Uniform random action | 0.300 |

For reference, a trained PPO policy scores about 1.0 on the same evaluation.
Planning took 1.56 s per control step on this CPU (1024 planning worlds ×
20 steps). The MPOPI option (`iterations > 1`) is implemented and unit-tested
but has not been evaluated as a controller yet.

## Reproduce

```bash
uv run --extra cpu python -m mpopi_train.scripts.eval_mpc --num-envs 16 --steps 200 --mpc.num-samples 64 --mpc.horizon 20
```

## Next

Stages 2 and 3 (MPC data collection and `mpc_ppo`) are in
[`mpc_ppo_results.md`](mpc_ppo_results.md). `SamplingMpc` now takes an env
config instead of a task id.
