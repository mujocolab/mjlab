# MPOPI toy benchmark results (Phase 8)

Date: 2026-09-25. Branch `mpopi-ppo`. CPU only, no GPU available.

**Bottom line.** On a 1-D point-mass toy task, importance-corrected replay
(MPOPI + PPO) never destabilised in 80 runs, while uncorrected replay
collapsed in 11 of 70. MPOPI finished with a small but confirmed improvement
in final return over plain PPO: +0.58 on a return of about −8.4, roughly 7%.
It did **not** show a significant sample-efficiency (AUC) gain over PPO. This
is one toy task. It is not evidence that MPOPI helps on robots.

## Setup

- Env: `mpopi_train.algorithms.toy_env.PointMassVecEnv`. Dynamics `x' = x + 0.1·clip(a)`,
  reward `−x² − 0.01a²`, 50-step episodes, terminate at `|x| > 3`.
- Data per iteration: 32 envs × 16 steps = 512 env steps. This is deliberately
  scarce, the regime where replay could help. 150 iterations per run.
- PPO: actor and critic (32, 32), 5 epochs × 4 minibatches, lr 1e-3 adaptive
  (desired KL 0.01), clip 0.2. Everything else is the mjlab default.
- Replay defaults: buffer 4 segments, replay ratio 1.0, ρ̄ = c̄ = 1.
- Eval: after every update, the deterministic (mean-action) policy runs from 16
  fixed starts `x0 ∈ linspace(−2, 2)`. It uses no training RNG.
- Metrics: **AUC** = mean eval return over all iterations (sample efficiency).
  **Final** = mean over the last 10% of iterations. Higher is better.
- Runs go through the real `MjlabOnPolicyRunner` mode switch.
- Script: `src/mpopi_train/scripts/benchmark.py`.

Arm C and arm B use 2× the samples per gradient step at the same number of
optimizer steps as A. This is inherent to replay, but it means "C beats A"
cannot separate *more samples per step* from *reusing data*.

## 1. Exploratory run (seeds 0–9)

| Arm | AUC | Final |
|---|---|---|
| A `ppo` | −8.86 ± 0.44 | −8.25 ± 0.34 |
| B `naive_replay_ppo` | −10.08 ± 1.31 | −9.32 ± 1.77 |
| C `mpopi_ppo` (ρ̄ = 1) | −8.44 ± 0.28 | −7.82 ± 0.17 |
| C′ `mpopi_ppo` (no truncation) | −8.51 ± 0.31 | −7.89 ± 0.26 |

Values are mean ± 95% t-interval over seeds. Eight Welch tests were run. The
smallest p is 0.020, which is not significant after a Holm correction, so this
run was treated as hypothesis-generating only. C vs C′ shows no difference
(p ≥ 0.59). The replay here was only mildly off-policy (ESS 0.90–0.96,
KL(μ‖π_old) ≈ 0.06), which is a regime where truncation barely matters.

## 2. Confirmation run (fresh seeds 10–29)

Hypotheses fixed before the run: **H1** C > A on final return, **H2** C > B on
AUC. The significance level is 0.025 per test (Bonferroni over 2).

| Arm | AUC | Final |
|---|---|---|
| A `ppo` | −9.90 ± 0.45 | −8.48 ± 0.21 |
| B `naive_replay_ppo` | −13.37 ± 5.47 | −17.22 ± 16.27 |
| C `mpopi_ppo` | −9.45 ± 0.63 | −7.90 ± 0.16 |

| Test (Welch) | Diff | p | Result |
|---|---|---|---|
| **H1** C vs A, final | +0.58 | 5.5e-5 | **Confirmed** |
| **H2** C vs B, AUC | +3.92 | 0.15 | Not confirmed |
| C vs A, AUC (secondary) | +0.45 | 0.23 | Not significant |

H2 fails because B's variance is enormous, not because B is competitive. See §3.

## 3. Exploratory: collapse rate and ablations

A run counts as *collapsed* if its final return is below −12. That threshold
was picked after seeing the data. For reference, A's worst run over all 30
seeds was −9.4. The Mann-Whitney tests are also post hoc.

| Study | B collapsed | C collapsed | B worst | C worst | C vs B final (Welch p) | Mann-Whitney p |
|---|---|---|---|---|---|---|
| Main (buffer 4, ratio 1) | 1/10 | 0/10 | −15.7 | −8.4 | 0.087 | 0.0022 |
| Confirmation | 4/20 | 0/20 | −164.6 | −9.0 | 0.25 | 0.00069 |
| Ratio 0.5 | 1/10 | 0/10 | −12.3 | −9.3 | 0.19 | 0.43 |
| Ratio 2.0 | 2/10 | 0/10 | −71.8 | −8.2 | 0.15 | 0.0010 |
| Buffer 1 (age 1 only) | 0/10 | 0/10 | −9.3 | −8.8 | 0.49 | 0.57 |
| Buffer 8 | 3/10 | 0/10 | −15.6 | −8.8 | 0.0097 | 0.0010 |

Totals: B collapsed in 11 of 70 runs, C in 0 of 80 (counting C′), A in 0 of 30.
C's final return stayed between −7.8 and −8.1 in every configuration.

The pattern matches the theory in `mpopi_design.md` §3–4. With age-1 data
(buffer 1), μ ≈ π_old and correction makes no difference. As data gets staler
(buffer 8) or more of the batch is replay (ratio 2), uncorrected replay
increasingly destabilises PPO, and the importance correction removes that
failure mode.

## What this does and does not show

Shown on this toy task:
- Naive replay is unsafe.
- MPOPI's correction removes the instability.
- MPOPI ends slightly better than PPO.

Not shown:
- Faster learning per env step (AUC vs PPO was not significant twice).
- Any benefit on a MuJoCo task or a robot.
- A benefit that is independent of the larger per-step batch.
- That ρ̄ = 1 truncation matters (C vs C′ was indistinguishable here).

## Recommended next steps

1. Repeat A/B/C on `Mjlab-Cartpole-Balance`, then `Mjlab-Velocity-Flat-Unitree-G1`
   (GPU), with at least 5 seeds each and the same primary hypotheses fixed
   beforehand.
2. Add a compute-matched control: A with `num_mini_batches` halved, giving
   equal samples per gradient step. This separates "reuse" from "bigger batch".
3. Push staleness harder (buffer 16, `max_policy_age`) to find where C itself
   degrades and whether `min_ess` or truncation then matters.

## Reproduce

```bash
uv run --extra cpu python -m mpopi_train.scripts.benchmark --seeds 10 --iterations 150 --out-dir logs/mpopi_toy/main
uv run --extra cpu python -m mpopi_train.scripts.benchmark --seeds 20 --seed-offset 10 --iterations 150 --arms A_ppo B_naive_replay C_mpopi --out-dir logs/mpopi_toy/confirm
uv run --extra cpu python -m mpopi_train.scripts.benchmark --seeds 10 --iterations 150 --arms B_naive_replay C_mpopi --replay-ratio 2.0 --out-dir logs/mpopi_toy/ratio_2
uv run --extra cpu python -m mpopi_train.scripts.benchmark --seeds 10 --iterations 150 --arms B_naive_replay C_mpopi --replay-buffer-size 8 --out-dir logs/mpopi_toy/buf_8
```

Each run takes about 14 s on CPU. Results are deterministic per seed on the
same machine and torch build.
