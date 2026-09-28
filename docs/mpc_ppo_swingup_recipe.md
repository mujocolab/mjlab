# Choosing the MPC → PPO recipe on Cartpole swing-up

Date: 2026-09-28. Branch `mpc-stage1`. CPU only. Follows
[`mpc_ppo_swingup_results.md`](mpc_ppo_swingup_results.md), where behavior
cloning (BC) toward the MPC was the only useful channel, the score dropped
once BC faded, and the MPOPI teacher collected worse data than MPPI under
execution noise.

## Pre-registration (written before running)

Common settings: `Mjlab-Cartpole-Swingup`, 64 PPO envs, task PPO
hyperparameters, 200 iterations, eval every 10 iterations over 400 steps (16
play envs, eval seed 10000), seeds 400–404. MPC data from 8 envs × 16 steps,
collected during the first 20 iterations. All `D` arms are BC only
(`use_in_ppo=False`): BC weight 1.0 decaying over 30 iterations. Primary
metric: AUC (mean eval score over training), paired by seed.

**Round 1: teacher and execution noise.**

| Arm | Teacher | Execution std |
|---|---|---|
| `A_ppo` | – | – |
| `R1_mppi_noise` (rerun of the earlier `D_mpc_bc_only`) | MPPI K = 32 | 0.3 |
| `R1_mppi_clean` | MPPI K = 32 | 0 |
| `R1_mpopi_clean` | MPOPI K = 8, L = 4 | 0 |

`A_ppo` and `R1_mppi_noise` repeat earlier runs with the same seeds; they
should reproduce the earlier AUCs (0.0271 and 0.0322) if runs are
deterministic per seed on this machine.

Questions: does removing execution noise help BC (`R1_mppi_clean` vs
`R1_mppi_noise`)? With clean data, is MPOPI a better teacher than MPPI
(`R1_mpopi_clean` vs `R1_mppi_clean`)?

Selection rule: the teacher and noise setting for round 2 is the round-1
`D` arm with the highest mean AUC.

**Round 2: keeping the teacher's pull**, with the round-1 winner's teacher
and noise:

| Arm | Change |
|---|---|
| `R2_floor` | BC weight floor 0.1 for the whole run; all 20 segments kept (`max_age=None`, 20 buffer segments) |
| `R2_dagger` | DAgger: the policy acts in the MPC envs and the MPC only labels the visited states |
| `R2_dagger_floor` | Both |

Questions: does a BC floor remove the drop after BC fades? Does DAgger
labeling beat cloning on the MPC's own states?

**Final choice.** The recipe is the arm with the highest mean AUC over both
rounds. It is called the chosen recipe only if it beats `A_ppo` on AUC in at
least 4 of 5 seeds; otherwise the result is "no clear winner". Wall-clock
cost is reported alongside. With 5 seeds no difference can reach
significance (smallest two-sided Wilcoxon p = 0.0625).

## Results

All runs: 1 CPU thread per process, runs in parallel. Score = mean per-step
reward (maximum 0.05); AUC over 200 iterations.

| Arm | Runs finished | AUC | Final | Iteration reaching 0.040 | Run s | MPC s |
|---|---|---|---|---|---|---|
| `A_ppo` | 5/5 | 0.0273 | 0.0372 | 70 180 80 – 100 | 1517 | 0 |
| `R1_mppi_noise` | 5/5 | 0.0288 | 0.0303 | 20 30 30 30 30 | 2257 | 969 |
| `R1_mppi_clean` | 5/5 | 0.0257 | 0.0275 | 40 90 40 30 30 | 2298 | 998 |
| `R1_mpopi_clean` | 5/5 | 0.0315 | 0.0344 | 40 40 40 30 30 | 2783 | 2237 |
| **`R2_dagger`** | 5/5 | **0.0366** | **0.0412** | 60 40 50 40 60 | 2794 | 1976 |
| `R2_floor` | **1/5** | (0.0241, 1 run) | – | – | – | – |
| `R2_dagger_floor` | **0/5** | – | – | – | – | – |

Mean normalized score at selected iterations:

| Arm | 0 | 20 | 30 | 40 | 50 | 80 | 120 | 160 | 199 |
|---|---|---|---|---|---|---|---|---|---|
| `A_ppo` | 0.05 | 0.21 | 0.29 | 0.47 | 0.51 | 0.62 | 0.62 | 0.63 | 0.68 |
| `R1_mppi_noise` | 0.16 | 0.60 | 0.87 | 0.66 | 0.60 | 0.63 | 0.63 | 0.53 | 0.65 |
| `R1_mppi_clean` | 0.17 | 0.39 | 0.54 | 0.71 | 0.52 | 0.49 | 0.56 | 0.52 | 0.52 |
| `R1_mpopi_clean` | 0.16 | 0.48 | 0.70 | 0.84 | 0.65 | 0.74 | 0.71 | 0.66 | 0.71 |
| `R2_dagger` | 0.11 | 0.34 | 0.57 | 0.74 | 0.84 | 0.75 | 0.84 | 0.86 | 0.78 |

Paired AUC differences (Wilcoxon, two-sided):

| Pair | Mean diff | Seeds positive | p |
|---|---|---|---|
| `R2_dagger − A_ppo` | +0.0093 | 4/5 | 0.13 |
| `R2_dagger − R1_mpopi_clean` | +0.0051 | 4/5 | 0.13 |
| `R1_mpopi_clean − A_ppo` | +0.0042 | 3/5 | 0.31 |
| `R1_mpopi_clean − R1_mppi_clean` | +0.0058 | 4/5 | 0.19 |
| `R1_mppi_noise − A_ppo` | +0.0015 | 3/5 | 0.81 |
| `R1_mppi_clean − A_ppo` | −0.0016 | 2/5 | 0.81 |

**Floor arms crashed.** 9 of the 10 runs with a BC floor stopped with
`RuntimeError: normal expects all elements of std >= 0.0` between
iterations 80 and 140: the Cartpole actor uses a directly parameterized
(`std_type="scalar"`) standard deviation, and a gradient step pushed it
below zero. The mechanism is not verified; a plausible one is that the
persistent BC pull keeps the policy change per update small, the adaptive
KL schedule then raises the learning rate toward its 1e-2 cap, and one large
step overshoots the std. A floor would need a log-parameterized std or a
std clamp, which changes PPO itself and was not part of this study.

**Reproducibility.** `A_ppo` and `R1_mppi_noise` repeat the earlier study's
seeds but not its per-seed curves (earlier AUC 0.0271 and 0.0322, now
0.0273 and 0.0288, with different per-seed values). The earlier runs used 2
CPU threads per process, these 1, so floating-point reduction order differs
and swing-up training diverges from there. Run-to-run noise at a fixed seed
is therefore of the same size as the differences measured here.

## Decision

By the pre-registered rule, round 1 picked the MPOPI teacher without
execution noise, and the highest-AUC arm overall is `R2_dagger`. It beats
`A_ppo` on AUC in 4 of 5 seeds, which meets the bar, so the chosen recipe is:

- BC only (`use_in_ppo=False`): MPC samples are not used in PPO's loss and
  no importance correction is needed;
- teacher MPOPI, K = 8, L = 4, H = 20;
- DAgger labeling (`driver="policy"`, `execution_std=0`): the policy acts in
  8 extra envs and the MPC labels the visited states, 16 steps per
  iteration for the first 20 iterations;
- BC weight 1.0 decaying to 0 over 30 iterations, then plain PPO.

Unlike cloning on the MPC's own states, DAgger showed no drop after BC
ended: its mean score kept rising to about 0.84–0.86 of the maximum, close
to the teacher's own score as a controller (about 0.84). Caveats: 5 seeds,
p = 0.13, and in wall-clock time plain PPO still reaches 0.040 sooner on
this CPU because MPOPI labeling costs about 2000 s per run here. Worth
checking next: more seeds, GPU cost, and whether the drop-free behavior
holds on a harder task.
