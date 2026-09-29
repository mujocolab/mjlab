# Gymnasium MuJoCo v5 benchmark results

This is a measured checkpoint comparison for mjlab's eleven manager-based MuJoCo
tasks, not a claim that their trajectories reproduce Gymnasium step for step.
The evaluation uses deterministic policies, 1,000 complete episodes per task,
evaluation seed 40001, and training seed 42. Scores are raw episode returns.
The results were recorded on 2026-09-21 and 2026-09-22. The selected checkpoint
is the final checkpoint of its listed training stage. Some selected policies were
trained in multiple stages; the full lineage is counted below.

## Selected checkpoint returns

The reference column preserves the source algorithm and environment version.
It is a useful scale check, not a controlled comparison: the policies, simulator
implementation, evaluation episode counts, and sometimes environment versions
differ. In particular, the RL Zoo PPO rows use v3; the mjlab tasks use v5.

| Task (mjlab v5) | Return mean ± episode std | Timeout | Public reference mean | Reference |
| --- | ---: | ---: | ---: | --- |
| Ant | 6,294.429 ± 754.161 | 97.5% | 1,327.158 | RL Zoo PPO, Ant-v3 |
| HalfCheetah | 6,489.964 ± 43.753 | 100.0% | 5,819.099 | RL Zoo PPO, HalfCheetah-v3 |
| Hopper | 3,100.010 ± 13.029 | 100.0% | 2,410.435 | RL Zoo PPO, Hopper-v3 |
| Humanoid | 12,049.012 ± 1,821.074 | 95.8% | 8,127.004 | Minari SAC, Humanoid-v5 |
| HumanoidStandup | 395,345.155 ± 15,591.305 | 100.0% | 129,867.182 | Minari PPO, HumanoidStandup-v5 |
| InvertedDoublePendulum | 9,295.498 ± 374.034 | 99.5% | 9,356.023 | Minari SAC, InvertedDoublePendulum-v5 |
| InvertedPendulum | 1,000.000 ± 0.000 | 100.0% | 1,000.000 | Minari SAC, InvertedPendulum-v5 |
| Pusher | -28.900 ± 4.100 | 100.0% | -22.053 | Minari SAC, Pusher-v5 |
| Reacher | -3.468 ± 1.367 | 100.0% | -3.281 | Minari SAC, Reacher-v5 |
| Swimmer | 341.757 ± 2.754 | 100.0% | 281.561 | RL Zoo PPO, Swimmer-v3 |
| Walker2d | 7,045.781 ± 337.195 | 96.1% | 3,478.798 | RL Zoo PPO, Walker2d-v3 |

Seven means exceed the listed reference, InvertedPendulum matches its maximum
return, and InvertedDoublePendulum, Pusher, and Reacher fall below their listed
SAC references. Reacher's result is close to the separate
[RAPID PPO teacher](https://github.com/eastha10/RAPID-Policy-Distillation)
result of -3.5086 over 100 episodes. Pusher is also below RAPID's
PPO mean of -26.1933 across five training seeds. The RAPID Pusher standard
deviation is across training seed means,
whereas this table reports variation across episodes from one training seed.
None of these score differences establishes statistical superiority.

## Training cost through the selected checkpoint

`E × R × epoch × mini` means environments, rollout steps per environment, PPO
epochs, and minibatches per epoch. `Iterations` counts completed data-collection
updates, not the checkpoint filename. Each update gathers `E × R` new transitions
and performs `epoch × mini` optimizer steps. PPO epochs reuse those samples.
Active time excludes evaluation pauses; process time includes them. Both times
cover training through the selected checkpoint, not the earliest point at which
a score threshold was crossed.

| Task | Selected checkpoint | E × R × epoch × mini | Iterations, stage / lineage | New transitions, lineage (M) | Optimizer steps, lineage | Active / process, lineage (min) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Ant | `model_1500.pt` | 4096 × 24 × 5 × 4 | 1,501 / 1,501 | 147.554 | 30,020 | 13.52 / 17.32 |
| HalfCheetah | `model_1700.pt` | 4096 × 24 × 5 × 4 | 1,701 / 1,701 | 167.215 | 34,020 | 6.19 / 9.23 |
| Hopper | `model_1500.pt` | 4096 × 24 × 5 × 4 | 1,501 / 1,501 | 147.554 | 30,020 | 7.73 / 10.60 |
| Humanoid | `model_5525.pt` | 2048 × 128 × 10 × 32 | 115 / 5,529 | 627.900 | 265,080 | 82.36 / 100.45 |
| HumanoidStandup | `model_2500.pt` | 4096 × 24 × 5 × 4 | 2,501 / 2,501 | 245.858 | 50,020 | 28.04 / 40.14 |
| InvertedDoublePendulum | `model_500.pt` | 4096 × 24 × 5 × 4 | 501 / 501 | 49.250 | 10,020 | 2.72 / 4.05 |
| InvertedPendulum | `model_300.pt` | 4096 × 24 × 5 × 4 | 301 / 301 | 29.590 | 6,020 | 1.02 / 1.78 |
| Pusher | `model_1598.pt` | 4096 × 32 × 5 × 16 | 400 / 1,600 | 209.715 | 56,000 | 11.28 / 16.11 |
| Reacher | `model_175.pt` | 4096 × 128 × 5 × 4 | 176 / 176 | 92.275 | 3,520 | 2.70 / 3.89 |
| Swimmer | `model_275.pt` | 4096 × 128 × 5 × 4 | 276 / 276 | 144.703 | 5,520 | 5.16 / 7.31 |
| Walker2d | `model_1300.pt` | 4096 × 24 × 5 × 4 | 1,301 / 1,301 | 127.894 | 26,020 | 7.80 / 10.70 |

The default mjlab RSL runner uses 24 rollout steps, five epochs, and four
minibatches. That is a valid starting point, not a required setting for every
task. Swimmer's fresh 4096 × 128 run reached 341.757 after 276 updates and
144.703 million transitions; the separate 4096 × 24 run exhausted 3,000 updates,
294.912 million transitions, and returned 122.24. Longer rollout also increases
the samples per update, so iteration counts alone do not measure training cost.
The selected Humanoid policy includes 5,414 parent updates before its final
115-update stage. Pusher's final 400-update stage followed 1,200 updates with
four minibatches; the final stage used sixteen. These continuation results do
not establish equivalent performance from a fresh run using the final settings.

## Sources and reproducibility

- [RL Zoo PPO benchmark](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md)
  supplies the five v3 reference means. RL Zoo describes this as a single-run
  performance check rather than a quantitative multi-seed benchmark.
- Farama-Minari expert `results.json` supplies the v5 references:
  [Humanoid SAC](https://huggingface.co/farama-minari/Humanoid-v5-SAC-expert/blob/main/results.json),
  [HumanoidStandup PPO](https://huggingface.co/farama-minari/HumanoidStandup-v5-PPO-expert/blob/main/results.json),
  [InvertedDoublePendulum SAC](https://huggingface.co/farama-minari/InvertedDoublePendulum-v5-SAC-expert/blob/main/results.json),
  [InvertedPendulum SAC](https://huggingface.co/farama-minari/InvertedPendulum-v5-SAC-expert/blob/main/results.json),
  [Pusher SAC](https://huggingface.co/farama-minari/Pusher-v5-SAC-expert/blob/main/results.json),
  and [Reacher SAC](https://huggingface.co/farama-minari/Reacher-v5-SAC-expert/blob/main/results.json).
- The table is transcribed from `logs/gym_mujoco_short_rollout_20260922/pr_metrics.json`.
  The 11 selected rows were checked against their saved evaluation JSON, run
  status, and checkpoint path. Raw checkpoints, episode records, TensorBoard
  events, and the original result JSON remain under ignored `logs/`; this PR
  includes the summary and protocol, not those large artifacts.

The registered PPO configurations are in `config/rl_cfg.py`. This table records
selected experiments, which sometimes used CLI overrides or warm starts; it does
not claim that every registered default independently reproduced the listed
result. Training uses native `train`; Gymnasium is required only for reference
tests. The MuJoCo XML assets and their license are under `assets/`.
