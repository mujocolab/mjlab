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
