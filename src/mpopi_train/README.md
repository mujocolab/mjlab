# mpopi_train

Training methods that reuse past and teacher data in PPO, built on mjlab as a
library. Nothing under `src/mjlab` is modified: this package adds its own
algorithm, runner, tasks and commands.

## Methods

| Task | Method |
|---|---|
| `Mpopi-G1-2k-PPO` | RSL-RL PPO (baseline) |
| `Mpopi-G1-2k-Replay-IS` | PPO plus the last 4 rollouts, importance-corrected (clipped weights, V-trace) |
| `Mpopi-G1-2k-DAgger` | PPO plus behavior cloning toward an MPOPI planner that labels the policy's own states |
| `Mpopi-G1-2k-Replay-IS-DAgger` | Both: replay of PPO's rollouts and DAgger labels in separate buffers |

All tasks are mjlab's flat G1 velocity task for 2000 iterations, with a command
curriculum that moves from (-1, 1) m/s to (-1, 1.5) m/s at iteration 500.
Every method bounds the policy std below by 0.05. The settings are in
`presets.py`.

## Commands

```bash
uv run mpopi-train Mpopi-G1-2k-Replay-IS-DAgger --agent.seed 1 --agent.run-name Replay-IS-DAgger_s1
uv run mpopi-play Mpopi-G1-2k-Replay-IS-DAgger --checkpoint-file logs/rsl_rl/g1_velocity_2k/<run>/model_1999.pt
uv run mpopi-eval --task Mpopi-G1-2k-Replay-IS-DAgger --controllers policy --checkpoint logs/rsl_rl/g1_velocity_2k/<run>/model_1999.pt
```

`mpopi-train` and `mpopi-play` are mjlab's `train` and `play` with these tasks
registered, so every mjlab option works (`--env.scene.num-envs`,
`--agent.logger`, ...). Preset settings can be overridden with
`--agent.algorithm.mpopi.*`, for example
`--agent.algorithm.mpopi.mpc.num-envs 32`. Several seeds are several runs:

```bash
for s in 1 2 3; do uv run mpopi-train Mpopi-G1-2k-PPO --agent.seed $s --agent.run-name PPO_s$s; done
```

Multi-GPU training of one run (`--gpu-ids` with more than one GPU) is not
supported: the worker processes do not register these tasks.

## Layout

| Path | Content |
|---|---|
| `algorithms/` | `MpopiPpo` (PPO whose batch adds corrected replay and BC samples), MPOPI estimators, replay buffer, toy env |
| `mpc/` | Sampling MPC (MPPI / MPOPI) on a batched copy of an mjlab task, state copy, MPC data collector |
| `config.py` | `MpopiRunnerCfg`: mjlab's runner config with `algorithm.mpopi`; `with_mpopi()` adds it to an existing config |
| `runner.py` | `MpopiOnPolicyRunner` and `MpopiVelocityRunner`: mjlab runners that build `MpopiPpo` and attach the MPC collector |
| `presets.py` | Settings of the four methods |
| `tasks.py` | The `Mpopi-G1-2k-*` tasks |
| `scripts/` | `train`, `play`, `eval_velocity` (fixed-speed evaluation), `eval_mpc`, `benchmark` (toy and Cartpole) |

Design notes and earlier results are in `docs/mpopi/`.

## `algorithm.mpopi` options

- `mode`: `ppo` (RSL-RL's `PPO`, unchanged), `naive_replay_ppo`, `mpopi_ppo`
  (replay with importance correction), or `mpc_ppo` (MPC teacher data).
- `min_action_std`: lower bound on the Gaussian actor's std, in every mode.
- `mpc.*` (mode `mpc_ppo`): `driver="policy"` lets the policy act while the
  planner labels its states (DAgger); `execution_std=0` clones the MPC action
  without noise; `use_in_ppo=False` keeps MPC samples out of the PPO losses;
  `bc_coef`, `bc_iterations` and `bc_floor` set the cloning weight;
  `inject_fraction` makes MPC samples a fixed fraction of every batch;
  `replay_own_rollouts` also replays PPO's own rollouts;
  `teacher_gap_every` logs how much the planner's plan beats the policy.
