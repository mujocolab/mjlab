# MPOPI on Mjlab-Cartpole-Balance (Phase 9)

Date: 2026-09-25. Branch `mpopi-ppo`. CPU only (12 cores, no GPU), torch
2.10.0+cpu. Humanoid (G1) runs were not possible on this machine.

**Bottom line.** On Cartpole-Balance, replay did not improve sample
efficiency. Both pre-registered hypotheses (MPOPI + PPO learns faster than
PPO, and faster than a compute-matched PPO) were **not supported**. In an
exploratory paired comparison, MPOPI + PPO was *slower* than both plain PPO and
naive replay in 5 of 5 seeds, although n = 5 cannot reach significance. Every
arm eventually solves the task. Unlike the toy task, naive replay never
collapsed here. The toy result (correction prevents collapse under stale data)
does **not** transfer to a benefit on this task at the default settings.

## Setup

- Task: `Mjlab-Cartpole-Balance` with the task's registered PPO settings
  (5 epochs × 4 minibatches, lr 1e-3 adaptive, 32 steps/env). The maximum
  per-step reward is 0.05 (reward × dt); scores below are normalized by it,
  so 1.0 = solved.
- Score: deterministic (mean-action) policy on a separate `play` env, 16 envs
  × 200 steps, fixed eval seed 10000. The global Python / NumPy / torch RNG
  state is saved and restored around evaluation and eval-env construction.
  It was verified that evaluating every iteration vs. only at the end gives
  bit-identical training curves.
- **AUC** = mean score over evaluated iterations (speed of solving).
  **Final** = mean over the last 10% of evaluated iterations.
- Replay defaults: buffer 4, ratio 1.0, ρ̄ = c̄ = 1.
- `A_ppo_bigmb` halves `num_mini_batches`, which gives PPO the same samples
  per gradient step as the replay arms (with half the optimizer steps).
- Script: `scripts/benchmarks/mpopi_benchmark.py --task ...` (the Phase 8
  script, generalized; toy results are reproduced exactly).

## Pilot (regime choice, single seed 0)

| envs | iterations | score over training |
|---|---|---|
| 256 | 300 | solved (1.0) by iteration 40 |
| 64 | 300 | stuck around 0.33 |

The 64-env "stall" later turned out to be seed-specific (see Study 2).

## Study 1: confirmatory (pre-registered)

Written before any MPOPI data on this task: 256 envs, 80 iterations, eval
every 5, fresh seeds 100–104. H1: C vs A on AUC. H2: C vs A′ on AUC.
Two-sided Mann-Whitney U, α = 0.025 each.

| Arm | AUC | Final | Per-seed AUC (seeds 100–104) |
|---|---|---|---|
| A `ppo` | 0.679 ± 0.305 | 0.863 ± 0.378 | 0.65 0.86 0.82 0.80 0.26 |
| A′ `ppo_bigmb` | 0.540 ± 0.330 | 0.753 ± 0.429 | 0.72 0.78 0.69 0.29 0.22 |
| B `naive_replay_ppo` | 0.692 ± 0.281 | 0.873 ± 0.343 | 0.76 0.87 0.71 0.82 0.30 |
| C `mpopi_ppo` | 0.554 ± 0.246 | 0.845 ± 0.386 | 0.57 0.74 0.56 0.67 0.22 |

| Test | Diff | MWU p | Result |
|---|---|---|---|
| **H1** C vs A, AUC | −0.125 | 0.22 | **Not supported** (opposite direction) |
| **H2** C vs A′, AUC | +0.014 | 0.84 | **Not supported** |

**Exploratory, paired by seed.** Seeds share network init and env, so
per-seed differences remove the large between-seed variance. Seed 104 is hard
for every arm.

| Pair | Per-seed AUC diff | Negative | Exact Wilcoxon p |
|---|---|---|---|
| C − A | −0.08 −0.12 −0.25 −0.13 −0.04 | 5/5 | 0.0625 |
| C − B | −0.19 −0.13 −0.14 −0.15 −0.08 | 5/5 | 0.0625 |
| C − A′ | −0.15 −0.04 −0.13 +0.39 +0.01 | 3/5 | 0.81 |
| B − A | +0.11 +0.00 −0.11 +0.03 +0.04 | 1/5 | 0.63 |

0.0625 is the smallest two-sided p attainable with n = 5, so a consistent
direction is all that 5 seeds can show.

Diagnostics (means over training): ESS 0.95–0.96, KL(μ‖π_old) ≈ 0.07,
policy age ≈ 2.5, PPO KL 0.014–0.016. The replay is only mildly off-policy.
With ρ̄ = 1 about 53% of weights are truncated and the mean weight is 0.89.

## Exploratory: no truncation (added after Study 1)

Same settings and seeds as Study 1, `importance_weight_clip_max=None`.

| Arm | AUC | Final |
|---|---|---|
| C (ρ̄ = 1) | 0.554 ± 0.246 | 0.845 |
| C′ (no truncation) | 0.611 ± 0.273 | 0.844 |

Removing truncation recovers part of the gap (+0.057, MWU p = 0.42) but stays
below A (0.679) and B (0.692). A plausible mechanism is that truncation shrinks
the replay part of the gradient (mean weight 0.89) and biases it toward the
behavior policy, which slows learning when the data is nearly on-policy
anyway. This is a hypothesis, not a tested explanation.

## Study 2: 64 envs (exploratory)

Registered before running as exploratory: 64 envs, 200 iterations, eval every
20, seeds 200–202, arms A and C.

| Arm | AUC | Final | Per-seed AUC |
|---|---|---|---|
| A `ppo` | 0.848 ± 0.101 | 0.998 | 0.84 0.81 0.89 |
| C `mpopi_ppo` | 0.624 ± 0.731 | 0.865 | 0.62 0.33 0.92 |

The premise ("PPO stalls at 64 envs") was wrong: PPO solved the task by about
iteration 60 on all three fresh seeds. C was faster than A on seed 202, but on
seed 201 it had only reached 0.61 after 200 iterations. There is no
significant difference with n = 3 (MWU p = 0.7 AUC, 0.2 final).

## Interpretation and recommendations

1. At default settings, MPOPI-corrected replay of *past PPO data* is not
   worth enabling for Cartpole-Balance. It adds computation and, if anything,
   slows learning.
2. Toy (Phase 8) and Cartpole agree on one point: the correction matters
   when the replay data is far off-policy (toy, buffer 8: naive replay
   collapsed). With data this close to on-policy (ESS ≈ 0.95), truncation
   costs more than it saves.
3. Robot-scale runs are **not recommended** with the current defaults. Worth
   testing first, at small scale:
   - more seeds (≥ 10) so paired effects can reach significance;
   - ρ̄ > 1 or self-normalized weights, so the replay gradient is not shrunk;
   - staler or genuinely external data (larger buffer, MPC-generated data),
     where the correction is expected to matter.
4. For the MPC → PPO direction, the behavior policy (an MPC controller) is far
   from π, which is the regime where correction is needed. The components
   built here (behavior log-prob storage, truncated IS, V-trace, ESS gating)
   apply there once the MPC side records log μ(a|s).

## Reproduce

```bash
uv run --extra cpu python scripts/benchmarks/mpopi_benchmark.py --task Mjlab-Cartpole-Balance --num-envs 256 --seeds 5 --seed-offset 100 --iterations 80 --eval-every 5 --arms A_ppo A_ppo_bigmb B_naive_replay C_mpopi C_mpopi_noclip --out-dir logs/mpopi_bench/cartpole_256
uv run --extra cpu python scripts/benchmarks/mpopi_benchmark.py --task Mjlab-Cartpole-Balance --num-envs 64 --seeds 3 --seed-offset 200 --iterations 200 --eval-every 20 --arms A_ppo C_mpopi --out-dir logs/mpopi_bench/cartpole_64
```

A 256-env run takes about 4 minutes on this CPU (4 runs in parallel). Runs are
deterministic per seed on the same machine: two processes with the same seed
produced identical curves.
