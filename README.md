![Project banner](https://raw.githubusercontent.com/mujocolab/mjlab/main/docs/source/_static/mjlab-banner.jpg)

# mjlab_MPOPI

[![Based on mjlab](https://img.shields.io/badge/based_on-mjlab-blue)](https://github.com/mujocolab/mjlab)
[![MuJoCo Warp](https://img.shields.io/badge/MuJoCo_Warp-3.11.0-blue)](https://github.com/google-deepmind/mujoco_warp/releases/tag/v3.11.0)
[![License](https://img.shields.io/github/license/mujocolab/mjlab)](LICENSE)

This repository is [mjlab](https://github.com/mujocolab/mjlab) plus a research package,
**`mpopi_train`**, that tries to **speed up policy training with control algorithms**: PPO reuses
its own past rollouts with importance correction (**Replay-IS**), and a sampling MPC planner
(MPPI / MPOPI) acts as a teacher that labels the policy's states (**DAgger**).

mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s manager-based API with
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), a GPU-accelerated version of
[MuJoCo](https://github.com/google-deepmind/mujoco). `mpopi_train` uses it as a library:
**nothing under `src/mjlab` is modified**, so upstream mjlab updates merge cleanly.

> **New to the project?** Follow the step-by-step guide in Vietnamese: **[GUIDE.md](GUIDE.md)**.

## Getting Started

Training requires an NVIDIA GPU (about 12 GB of memory for 4096 robots). Without one, use the
Kaggle notebook (see [Notebooks](#notebooks)).

The research code is on the **`mpc-stage1`** branch:

```bash
git clone --branch mpc-stage1 https://github.com/TamasTran/mjlab_MPOPI.git && cd mjlab_MPOPI
uv sync --extra cu128
```

mjlab's own demo still works:

```bash
uv run --extra cu128 demo
```

## Training Examples

### 1. Velocity Tracking with the Compared Methods

The task is mjlab's flat Unitree G1 velocity task for **2000 iterations**, with a command
curriculum that moves from (-1, 1) m/s to (-1, 1.5) m/s at iteration 500. Each method is a task:

| Task | Method |
|---|---|
| `Mpopi-G1-2k-PPO` | RSL-RL PPO (baseline) |
| `Mpopi-G1-2k-Replay-IS` | PPO plus its last 4 rollouts, importance-corrected (clipped weights, V-trace) |
| `Mpopi-G1-2k-DAgger` | PPO plus behavior cloning toward an MPOPI planner that labels the policy's own states |
| `Mpopi-G1-2k-Replay-IS-DAgger` | Both, in separate buffers |

```bash
uv run --extra cu128 mpopi-train Mpopi-G1-2k-Replay-IS-DAgger --agent.logger tensorboard \
  --agent.seed 1 --agent.run-name Replay-IS-DAgger_s1
```

`mpopi-train` is mjlab's `train` with these tasks registered, so every mjlab option works. Method
settings are in [`src/mpopi_train/presets.py`](src/mpopi_train/presets.py) and can be overridden,
for example `--agent.algorithm.mpopi.mpc.num-envs 32`.

**Several seeds** are several runs (results vary noticeably between runs, so compare at least two):

```bash
for s in 1 2 3; do
  uv run --extra cu128 mpopi-train Mpopi-G1-2k-PPO --agent.logger tensorboard \
    --agent.seed $s --agent.run-name PPO_s$s
done
```

The tasks are registered through the `mjlab.tasks` entry point, so mjlab's own `train` and
`play` commands (and the worker processes of multi-GPU training) see them too. Multi-GPU training
of one run (`--gpu-ids` with more than one GPU) is untested; the experiments ran one training per
GPU with `CUDA_VISIBLE_DEVICES`.

**Evaluate** a checkpoint at fixed forward speeds (64 robots, measured over 10 s after 1 s of
acceleration). The PPO task is used for every method: same robot and network, and no MPC teacher
is built:

```bash
uv run --extra cu128 mpopi-eval --task Mpopi-G1-2k-PPO --controllers policy \
  --num-envs 64 --steps 550 --settle-steps 50 \
  --checkpoint logs/rsl_rl/g1_velocity_2k/<run>/model_1999.pt
```

**Watch** a trained policy:

```bash
uv run --extra cu128 mpopi-play Mpopi-G1-2k-PPO \
  --checkpoint-file logs/rsl_rl/g1_velocity_2k/<run>/model_1999.pt
```

### 2. mjlab's Original Velocity Task

Train a Unitree G1 humanoid to follow velocity commands on flat terrain (PPO, 30,000 iterations):

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096
```

**Multi-GPU Training:** Scale to multiple GPUs using `--gpu-ids`:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --gpu-ids "[0, 1]" \
  --env.scene.num-envs 4096
```

See the [Distributed Training guide](https://mujocolab.github.io/mjlab/main/source/training/distributed_training.html) for details.

Evaluate a policy while training (fetches latest checkpoint from Weights & Biases):

```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

### 3. Motion Imitation

Train a humanoid to mimic reference motions. See the [motion imitation guide](https://mujocolab.github.io/mjlab/main/source/training/motion_imitation.html) for preprocessing setup.

```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1 --registry-name your-org/motions/motion-name --env.scene.num-envs 4096
uv run play Mjlab-Tracking-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

### 4. Sanity-check with Dummy Agents

Use built-in agents to sanity check your MDP before training:

```bash
uv run play Mjlab-Your-Task-Id --agent zero  # Sends zero actions
uv run play Mjlab-Your-Task-Id --agent random  # Sends uniform random actions
```

When running motion-tracking tasks, add `--registry-name your-org/motions/motion-name` to the command.

## Project Layout

| Path | Content |
|---|---|
| [`src/mpopi_train/`](src/mpopi_train/) | The research package: algorithms, MPC, runner, presets, tasks, scripts ([README](src/mpopi_train/README.md)) |
| [`docs/mpopi/`](docs/mpopi/) | Design notes |
| [`notebooks/`](notebooks/) | Colab and Kaggle notebooks |
| [`tests/`](tests/) (`test_mpopi_*.py`, `test_mpc_*.py`) | Tests of the package |
| `src/mjlab/` | Unmodified mjlab |

### Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/mpc_g1_train_kaggle.ipynb` | Train and evaluate the four methods on Kaggle (2 T4 GPUs, up to 12 h) |
| `notebooks/mpc_g1_eval_kaggle.ipynb` | Re-evaluate checkpoints of earlier Kaggle runs |
| `notebooks/mpc_g1_bc_kaggle.ipynb` | How long DAgger should clone the MPC teacher (3 variants, 2 seeds) |
| Other `mpc_*` and `mpopi_*` notebooks | Earlier experiments; they clone the tag `mpopi-before-module` |

## Documentation

- Step-by-step guide (Vietnamese): **[GUIDE.md](GUIDE.md)**
- Package reference: [`src/mpopi_train/README.md`](src/mpopi_train/README.md)
- mjlab documentation: **[mujocolab.github.io/mjlab](https://mujocolab.github.io/mjlab/)**

## Development

```bash
make test          # Run all tests
make test-fast     # Skip slow tests
make format        # Format and lint
make docs          # Build docs locally
```

Tests of the research package only:

```bash
uv run --extra cu128 pytest tests/test_mpopi_*.py tests/test_mpc_*.py
```

For development setup: `uvx pre-commit install`

## Citation

mjlab is used in published research and open-source robotics projects. See the [Research](https://mujocolab.github.io/mjlab/main/source/research.html) page for publications and projects, or share your own in [Show and Tell](https://github.com/mujocolab/mjlab/discussions/categories/show-and-tell).

If you use mjlab in your research, please consider citing:

```bibtex
@misc{zakka2026mjlablightweightframeworkgpuaccelerated,
  title={mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning},
  author={Kevin Zakka and Qiayuan Liao and Brent Yi and Louis Le Lay and Koushil Sreenath and Pieter Abbeel},
  year={2026},
  eprint={2601.22074},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2601.22074},
}
```

The methods in `mpopi_train` build on PPO (Schulman et al., 2017,
[arXiv:1707.06347](https://arxiv.org/abs/1707.06347)), V-trace (Espeholt et al., 2018,
[arXiv:1802.01561](https://arxiv.org/abs/1802.01561)), DAgger (Ross et al., 2011,
[arXiv:1011.0686](https://arxiv.org/abs/1011.0686)) and MPPI (Williams et al., 2017).

## License

mjlab is licensed under the [Apache License, Version 2.0](LICENSE).

### Third-Party Code

Some portions of mjlab are forked from external projects:

- **`src/mjlab/utils/lab_api/`** — Utilities forked from [NVIDIA Isaac
  Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause license, see file
  headers)

Forked components retain their original licenses. See file headers for details.

## Acknowledgments

mjlab wouldn't exist without the excellent work of the Isaac Lab team, whose API
design and abstractions mjlab builds upon.

Thanks to the MuJoCo Warp team — especially Erik Frey and Taylor Howell — for
answering our questions, giving helpful feedback, and implementing features
based on our requests countless times.
