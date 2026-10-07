# MPC-generated data for PPO on Cartpole swing-up

Date: 2026-09-26. Branch `mpc-stage1`. CPU only. Follows
[`mpc_ppo_results.md`](mpc_ppo_results.md), where on Cartpole balance every
arm with behavior cloning hit the score ceiling within ~10 iterations, so the
importance correction could not be assessed.

## Why swing-up

- Same model as balance, so copying the state into the planner stays exact.
- The MPC teacher is imperfect here. As a controller (8 envs × 400 steps,
  eval seed 10000; score = mean per-step reward, maximum 0.05):

  | Controller | Score | Normalized |
  |---|---|---|
  | MPPI K = 32, H = 20 (used below) | 0.0387 | 0.773 |
  | MPPI K = 32, H = 40 | 0.0407 | 0.814 |
  | Random action | 0.0053 | 0.105 |
  | Zero action | 0.0000 | 0.000 |

- Plain PPO learns slowly but can surpass the teacher. In a probe (seeds 0–1,
  64 envs, 200 iterations), eval scores reached about 0.043 after 120–160
  iterations, with large swings (seed 0 fell back to 0.017 at iteration 200).

## Pre-registration (written before the main run)

`Mjlab-Cartpole-Swingup`, 64 PPO envs, task PPO hyperparameters, 200
iterations, eval every 10 iterations over 400 steps (16 play envs, eval seed
10000). Fresh seeds 400–404, the same in every arm. MPC data settings are
unchanged from the balance study (8 MPC envs, 16 steps per segment, collect
for 20 iterations, execution std 0.3, BC weight 1.0 → 0 over 30 iterations,
K = 32, H = 20). Same five arms: `A_ppo`, `D_mpc_ppo`, `D_mpc_naive`,
`D_mpc_bc_only`, `D_mpc_no_bc`.

Hypotheses, paired Wilcoxon over 5 seeds on AUC (mean eval score over
training; the smallest possible two-sided p is 0.0625):

- **H1:** `D_mpc_ppo` > `A_ppo`.
- **H2:** `D_mpc_ppo` > `D_mpc_bc_only`. With an imperfect teacher, adding
  the MPC samples to PPO's loss should help beyond imitation.
- **H3:** `D_mpc_ppo` > `D_mpc_naive`. With an imperfect teacher, the
  importance correction should matter.
- **H4:** `D_mpc_no_bc` > `A_ppo`.

Also reported: final score (last 10% of evaluations), whether each arm ends
above the teacher's 0.0387, the iteration at which the eval score first
reaches 0.040, and wall-clock time (arms run in parallel on one CPU, so
times include contention).

## Results (seeds 400–404, 200 iterations)

| Arm | AUC | Final | Iteration reaching 0.040 (per seed) | Est. seconds to 0.040 | Run s | MPC s |
|---|---|---|---|---|---|---|
| `A_ppo` | 0.0271 | 0.0316 | 60 120 50 60 – | 138 292 117 142 – | 469 | 0 |
| `D_mpc_ppo` | 0.0318 | 0.0344 | 50 30 30 50 50 | 416 384 392 401 384 | 766 | 294 |
| `D_mpc_naive` | 0.0220 | 0.0290 | 30 170 – 80 – | 369 717 – 471 – | 773 | 294 |
| `D_mpc_bc_only` | **0.0322** | **0.0346** | 30 20 30 20 30 | 364 360 392 328 341 | 762 | 293 |
| `D_mpc_no_bc` | 0.0220 | 0.0283 | – – 110 110 130 | – – 590 529 564 | 762 | 293 |

Mean eval score across seeds at selected iterations:

| Arm | 0 | 10 | 20 | 30 | 50 | 80 | 120 | 160 | 199 |
|---|---|---|---|---|---|---|---|---|---|
| `A_ppo` | 0.003 | 0.007 | 0.008 | 0.015 | 0.028 | 0.034 | 0.031 | 0.031 | 0.032 |
| `D_mpc_ppo` | 0.008 | 0.022 | 0.030 | 0.037 | 0.037 | 0.028 | 0.025 | 0.038 | 0.035 |
| `D_mpc_naive` | 0.008 | 0.018 | 0.022 | 0.027 | 0.016 | 0.019 | 0.018 | 0.023 | 0.027 |
| `D_mpc_bc_only` | 0.008 | 0.014 | 0.033 | 0.045 | 0.025 | 0.038 | 0.038 | 0.031 | 0.031 |
| `D_mpc_no_bc` | 0.003 | 0.008 | 0.009 | 0.010 | 0.015 | 0.026 | 0.031 | 0.028 | 0.025 |

Paired over seeds (Wilcoxon, two-sided):

| | AUC diff | Seeds positive | p |
|---|---|---|---|
| H1 `D_mpc_ppo − A_ppo` | +0.0048 | 3/5 | 0.63 |
| H2 `D_mpc_ppo − D_mpc_bc_only` | −0.0003 | 3/5 | 1.0 |
| H3 `D_mpc_ppo − D_mpc_naive` | +0.0099 | 4/5 | 0.13 |
| H4 `D_mpc_no_bc − A_ppo` | −0.0051 | 2/5 | 0.31 |

## Reading

- **H1 not supported on AUC.** MPC data makes the start faster: every
  `D_mpc_ppo` seed reached 0.040 by iteration 30–50, versus 4 of 5 `A_ppo`
  seeds at 50–120. But once BC fades and collection stops (iterations
  20–30), the mean score drops (0.037 at iteration 50 to 0.025 at 120) and
  PPO has to recover on its own. In wall-clock time plain PPO is faster to
  0.040 (about 140 s vs about 390 s), because of the MPC planning cost.
- **H2: no gain from putting MPC samples in PPO's loss** beyond behavior
  cloning, even with an imperfect teacher. BC alone was as good or better.
- **H3: the correction mainly prevents harm.** Uncorrected MPC samples in
  PPO's loss (`D_mpc_naive`) were worse than plain PPO (AUC 0.0220 vs
  0.0271) and than BC alone; the correction removed most of that damage (4/5
  seeds, p = 0.13, not significant). This matches the toy study, where the
  correction prevented collapse rather than speeding learning up.
- **H4: corrected MPC data without BC did not help** (2/5 seeds, below
  plain PPO on average). With mean weights around 0.4–0.6 early on, most MPC
  samples are down-weighted, and the ones kept come from states PPO's own
  rollouts rarely visit.
- Swing-up is unstable for every arm: policies near 0.045 often drop back
  within a few evaluations, so final scores at a single iteration are noisy.

Overall, on both Cartpole tasks the useful channel from MPC to PPO was
behavior cloning. The MPOPI-corrected samples did not add sample efficiency;
their correction mattered only in making MPC samples safe to include.
Five seeds cannot show significance; the directions above are what the data
supports.

## Reproduce

```bash
uv run --extra cpu python -m mpopi_train.scripts.benchmark --task Mjlab-Cartpole-Swingup --num-envs 64 --iterations 200 --eval-every 10 --eval-steps 400 --seeds 5 --seed-offset 400 --arms A_ppo D_mpc_ppo D_mpc_naive D_mpc_bc_only D_mpc_no_bc
```

## MPOPI as the teacher (pre-registration)

All results above used MPPI (one sampling iteration per control step) as the
teacher. MPOPI (`iterations = L > 1`) refines the mean and per-dimension std
over L batches within a control step. Planned before running:

**Part 1: controller comparison at equal simulation budget** (K × L rollouts
of H = 20 steps per control step), swing-up, 8 envs × 400 steps, eval seeds
10000 and 10001:

| Budget K·L | MPPI | MPOPI |
|---|---|---|
| 32 | K = 32, L = 1 | K = 16, L = 2; K = 8, L = 4 |
| 96 | K = 96, L = 1 | K = 32, L = 3 |

MPOPI "wins" a budget if its best configuration scores higher than MPPI on
both eval seeds.

**Part 2: MPOPI teacher for PPO.** `D_mpc_bc_only` and `D_mpc_ppo` rerun on
seeds 400–404 with every setting unchanged except the planner, which becomes
the better MPOPI configuration at budget 32. Compared, paired by seed, with the
MPPI-teacher runs above on AUC. Run regardless of the Part 1 outcome.

## MPOPI as the teacher: results

**Part 1 (controller, 8 envs × 400 steps):**

| Budget K·L | Planner | Seed 10000 | Seed 10001 | Planning s/step* |
|---|---|---|---|---|
| 32 | MPPI K = 32 | 0.773 | 0.788 | 1.5 |
| 32 | MPOPI K = 16, L = 2 | 0.817 | 0.808 | 2.2 |
| 32 | **MPOPI K = 8, L = 4** | **0.851** | **0.828** | 3.2 |
| 96 | MPPI K = 96 | 0.871 | 0.873 | 2.3 |
| 96 | **MPOPI K = 32, L = 3** | **0.905** | **0.904** | 3.3 |

\*Ten runs in parallel on one CPU. MPOPI wins both budgets on both seeds
by the registered criterion. At the same number of simulated steps it is
slower in wall-clock time because its L batches run one after another. The
gain is in the balancing phase: after 100 steps (swing-up) the two are equal
(0.59/0.58 vs 0.60/0.55), after 200 steps MPOPI leads (0.74/0.73 vs
0.69/0.71).

**Part 2 (MPOPI K = 8, L = 4 as the teacher, seeds 400–404, paired with the
MPPI-teacher runs above):**

| Arm | Teacher | AUC | Final | Iteration reaching 0.040 | Mean reward of collected MPC data |
|---|---|---|---|---|---|
| `D_mpc_bc_only` | MPPI | 0.0322 | 0.0346 | 30 20 30 20 30 | 0.0249 |
| `D_mpc_bc_only` | MPOPI | 0.0274 | 0.0274 | 50 30 30 30 30 | 0.0219 |
| `D_mpc_ppo` | MPPI | 0.0318 | 0.0344 | 50 30 30 50 50 | 0.0249 |
| `D_mpc_ppo` | MPOPI | 0.0278 | 0.0351 | 50 40 190 40 40 | 0.0219 |

MPOPI − MPPI teacher, AUC: `D_mpc_bc_only` −0.0047 (1/5 seeds positive,
p = 0.31); `D_mpc_ppo` −0.0040 (1/5, p = 0.44).

**Reading.** The better controller did not make a better teacher. The data
MPOPI collected had a *lower* mean reward than MPPI's (0.0219 vs 0.0249),
even though MPOPI scored higher as a controller. The difference is that
collection adds execution noise (σ = 0.3) to every action. An untested
explanation: with only 8 samples per batch and a std that MPOPI shrinks
(down to 0.1), the planner explores narrowly and recovers worse from the
injected noise. Checking it would need MPOPI evaluated as a controller under
the same execution noise, or a smaller execution std. With 5 seeds none of
the differences is significant; the direction is against the MPOPI teacher.
