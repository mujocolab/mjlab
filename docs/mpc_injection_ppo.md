# MPC-Injection for PPO on Cartpole swing-up

Date: 2026-09-28. Branch `mpc-stage1`. CPU only.

MPC-Injection (arXiv 2606.26392) mixes MPC transitions into the replay buffer
of an off-policy learner (SAC / TD3) at a fixed fraction p, with no
importance correction and no behavior-cloning loss. This study transfers the
idea to PPO: every PPO update draws MPC samples uniformly from a fixed MPC
dataset so that they make up a fraction p of the batch, for the whole run.

For PPO this is not an unbiased estimator. Advantages are computed with GAE
along the MPC's trajectories and the samples are treated as on-policy, so
the update behaves like advantage-weighted imitation of the MPC. The
off-policy learners in the paper avoid this because their critic is
Q(s, a).

## Setup

`mpc_ppo` with `use_in_ppo=True`, `correction=False`, `bc_coef=0`,
`execution_std=0` (the MPC action itself, as in the paper),
`inject_fraction=p`, MPOPI teacher K = 8, L = 4, H = 20. MPC data: 8 envs × 16
steps per iteration for the first 20 iterations, all kept (`max_age=None`,
20 buffer segments, 2560 samples), then reused until the end of training like
the paper's offline MPC dataset.

Everything else matches the swing-up recipe study
([`mpc_ppo_swingup_recipe.md`](mpc_ppo_swingup_recipe.md)): 64 PPO envs, 200
iterations, eval every 10 iterations over 400 steps, seeds 400–404, one CPU
thread per process. `PPO` and `MPC-DAgger` are taken from that study (same
seeds and settings), not rerun.

## Pre-registration (written before running)

Arms: `MPC-Inject` with p = 0.25 (primary; the fraction the paper reports on
the Go2), p = 0.10 and p = 0.50 (secondary).

- **H1:** `MPC-Inject` (p = 0.25) has a higher AUC than `PPO`, paired by seed.
  With 5 seeds the smallest two-sided Wilcoxon p is 0.0625, so this can only
  show a direction (at least 4 of 5 seeds).
- **H2 (descriptive):** `MPC-Inject` (p = 0.25) compared with `MPC-DAgger`.
- Also reported: final score, whether the score still drops after the MPC
  phase (the MPC data stays in every batch here), and wall-clock time.

## Results (CPU)

| Arm | Runs finished | AUC (finished runs) | Paired with `PPO` |
|---|---|---|---|
| `PPO` (recipe study) | 5/5 | 0.0273 | – |
| `MPC-DAgger` (recipe study) | 5/5 | 0.0366 | +0.0093, 4/5 seeds |
| `MPC-Inject` p = 0.10 | 2/5 | 0.0172 | −0.0101, 0/2 seeds |
| `MPC-Inject` p = 0.25 | 3/5 | 0.0172 | −0.0147, 0/3 seeds |
| `MPC-Inject` p = 0.50 | **0/5** | – | – |

Mean normalized score of the finished runs: p = 0.25 reached 0.24 at
iteration 20, 0.32 at 50 and 0.40 at 199 (`PPO`: 0.21, 0.51, 0.68;
`MPC-DAgger`: 0.34, 0.84, 0.78). Injection did not speed up the start either.

**10 of 15 runs crashed** with `RuntimeError: normal expects all elements of
std >= 0.0` between iterations 120 and 180, the same failure as the BC-floor
runs. The crash rate grows with the injected fraction (3/5, 2/5, 5/5). A
plausible mechanism, not verified: the injected actions are the noise-free
MPC action, and PPO's surrogate raises `log π(u0|o)` for samples with positive
advantage; for a Gaussian this pushes the std down (the gradient of the log
density with respect to σ grows like 1/σ as the mean approaches `u0`), and the
Cartpole actor's directly parameterized std (`std_type="scalar"`) is pushed
below zero.

**H1 not supported.** Every finished `MPC-Inject` run had a lower AUC than
`PPO` on the same seed, and most runs did not finish. Injecting noise-free MPC
data into PPO as if it were on-policy both hurt learning and destabilized the
policy's exploration noise on this task.
