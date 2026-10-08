# MPC-generated data for PPO on Cartpole (stages 2 and 3)

Date: 2026-09-26. Branch `mpc-stage1`. CPU only.

## Pre-registration (written before the main run)

Task `Mjlab-Cartpole-Balance`, 64 PPO envs, task PPO hyperparameters, 100
iterations, deterministic-policy evaluation every 5 iterations (16 play envs ×
200 steps, eval seed 10000). Fresh seeds 300–304, the same seeds in every arm.
Score = mean per-step reward (maximum 0.05; 0.0348 is the zero-action score).

MPC data (`BenchmarkCfg.mpc`): 8 MPC envs, 16 steps per segment, one segment
per iteration for the first 20 iterations, execution std 0.3, buffer of 8
segments, max age 10, BC weight 1.0 decaying to 0 over 30 iterations, planner
MPPI K = 32, H = 20 (0.92 of the maximum as a controller).

Arms:

| Arm | MPC data in PPO loss | Importance correction | BC term |
|---|---|---|---|
| `A_ppo` | – | – | – |
| `D_mpc_ppo` | yes | yes | yes |
| `D_mpc_naive` | yes | no (ratio 1) | yes |
| `D_mpc_bc_only` | no | – | yes |
| `D_mpc_no_bc` | yes | yes | no |

Hypotheses, tested on AUC (mean eval score over training) with a paired
Wilcoxon test over the 5 seeds (smallest possible two-sided p = 0.0625, so
"significant" is out of reach; direction consistency across seeds is what
can be shown):

- **H1 (primary):** `D_mpc_ppo` > `A_ppo` per PPO iteration.
- **H2:** `D_mpc_ppo` vs `D_mpc_bc_only`: does adding MPC samples to the PPO
  loss help beyond behavior cloning?
- **H3:** `D_mpc_ppo` vs `D_mpc_naive`: does the importance correction matter?
- **H4:** `D_mpc_no_bc` > `A_ppo`: do corrected MPC samples help without BC?

Also reported, because MPC planning is expensive: wall-clock time, and the
iteration and wall-clock time at which the eval score first reaches 0.045
(0.9 of the maximum).

A 1-seed smoke test (seed 0, 30 iterations) was run before this
registration to check the pipeline: `D_mpc_ppo` scored 0.049 from iteration
10 on, `A_ppo` 0.014 at iteration 30; MPC collection took 210 s of the 252 s
run (PPO alone: 38 s).

## How it works

- `mpopi_train.mpc.collector.MpcCollector` drives `num_envs` envs of the training
  task with `SamplingMpc` and executes `a = u0 + σ ε` around the MPC action
  `u0`. It stores `log μ(a|s) = log N(a; u0, σ²)` and `(u0, σ)` as the
  behavior distribution parameters, in the `ReplayBuffer` segment layout.
- In mode `mpc_ppo`, `MpopiPpo` collects one segment per scheduled
  iteration into its buffer, drops segments older than `max_age`, and runs
  the same MPOPI processing as for PPO replay (weights `min(1, π_old/μ)`,
  V-trace targets) over every MPC segment. The samples are appended to the
  fresh rollout in PPO's surrogate and value losses (`use_in_ppo`), and a
  behavior-cloning term `bc_weight(it) · ||mean_π(s) − u0(s)||²` is added
  on MPC samples, with `bc_weight` decaying linearly to 0.
- The runner builds the collector from the training env's config
  (`MjlabOnPolicyRunner._attach_mpc_collector`). The actor must be Gaussian.
- State copying into the planner is exact for Cartpole only; tasks with
  stateful managers (commands, action history, delays) need more state.

## Results (seeds 300–304, 100 iterations)

| Arm | AUC | Final | Iteration reaching 0.045 (per seed) | Est. seconds to 0.045 | Run s | MPC s |
|---|---|---|---|---|---|---|
| `A_ppo` | 0.0205 | 0.0295 | – – 80 – 55 | – – 182 – 118 | 221 | 0 |
| `D_mpc_ppo` | **0.0472** | **0.0498** | 5 10 10 10 10 | 96 183 162 164 159 | 476 | 268 |
| `D_mpc_naive` | 0.0471 | 0.0500 | 10 10 15 10 10 | 178 185 236 164 160 | 477 | 269 |
| `D_mpc_bc_only` | 0.0471 | 0.0499 | 5 15 10 10 5 | 98 274 167 167 89 | 485 | 274 |
| `D_mpc_no_bc` | 0.0256 | 0.0418 | – – 80 85 60 | – – 406 429 372 | 478 | 268 |

"–" = never reached within 100 iterations. The five arms ran as parallel
processes on one 12-core CPU, so absolute times are inflated by contention;
"est. seconds" adds the measured MPC collection time up to that iteration to
the run's average PPO time per iteration.

Paired over seeds (Wilcoxon, two-sided; 0.0625 is the smallest possible p):

| | AUC diff | Seeds positive | p |
|---|---|---|---|
| H1 `D_mpc_ppo − A_ppo` | +0.0267 | 5/5 | 0.0625 |
| H2 `D_mpc_ppo − D_mpc_bc_only` | +0.0001 | 1/5 | 0.63 |
| H3 `D_mpc_ppo − D_mpc_naive` | +0.0001 | 4/5 | 0.63 |
| H4 `D_mpc_no_bc − A_ppo` | +0.0051 | 4/5 | 0.13 |

## Reading

- **H1 supported in direction on all 5 seeds:** with MPC data, PPO reached
  0.9 of the maximum score within 5–15 iterations on every seed; plain PPO
  reached it on 2 of 5 seeds within 100 iterations. Even counting MPC
  planning time, `D_mpc_ppo` reached the threshold in about 100–180 s on
  every seed, while `A_ppo` reached it on 2 seeds (118 s, 182 s).
- **The effect comes from behavior cloning.** BC alone is as good as the
  full method (H2), and turning the importance correction off changes
  nothing measurable (H3). All BC arms hit the score ceiling by iteration
  ~10, so this setup cannot discriminate between them: a ceiling effect, not
  evidence that the correction is useless.
- **Corrected MPC samples without BC help only weakly** (H4: 4/5 seeds,
  not significant). The mean weight `min(1, π/μ)` was about 0.44–0.6, and the
  behavior KL between MPC and policy was about 0.5–0.9, so the correction
  down-weights most MPC samples early on.
- Cost: MPC collection took ~270 s per run (20 segments of 8 envs × 16
  steps), more than the whole PPO run. Cartpole is also an easy task that MPC
  nearly solves by itself; nothing here transfers to G1 without a harder task
  and exact state copying for stateful managers.

Next useful experiments: a harder task where MPC is imperfect (so BC's
ceiling does not hide the other terms), the same arms with a smaller BC
weight, and seeds ≥ 10.

## Reproduce

```bash
uv run --extra cpu python -m mpopi_train.scripts.benchmark --task Mjlab-Cartpole-Balance --num-envs 64 --iterations 100 --eval-every 5 --seeds 5 --seed-offset 300 --arms A_ppo D_mpc_ppo D_mpc_naive D_mpc_bc_only D_mpc_no_bc
```
