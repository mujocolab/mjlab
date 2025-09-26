# mjlab

<p align="left">
  <img alt="tests" src="https://github.com/mujocolab/mjlab/actions/workflows/ci.yml/badge.svg" />
</p>

> **⚠️ EXPERIMENTAL PREVIEW** 
> 
> This project is in very early experimental stages. APIs, features, and documentation are subject to significant changes. Use at your own risk and expect frequent breaking changes.

[Isaac Lab](https://github.com/isaac-sim/IsaacLab) API with [MJWarp](https://github.com/google-deepmind/mujoco_warp) backend.

## Development Guide

Clone `mjlab`:

```bash
git clone git@github.com:mujocolab/mjlab.git && cd mjlab
```

### Using uv

Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Once installed, you can verify it works by running:

```bash
uv run scripts/list_envs.py
```

## Reinforcement Learning

### Velocity training

Train a Unitree G1 to follow velocity commands (headless, large batch):

```bash
MUJOCO_GL=egl uv run scripts/rl/train.py \
  Mjlab-Velocity-Flat-Unitree-G1 \
  --env.scene.num-envs 4096
```

Play the trained policy:

```bash
uv run scripts/rl/play.py \
  --task Mjlab-Velocity-Flat-Unitree-G1-Play
```

### Motion mimicking

Run a pre-trained motion-mimic policy on the G1:

```bash
uv run scripts/rl/play.py \
  --task Mjlab-Tracking-Flat-Unitree-G1-Play \
  --wandb-run-path gcbc_researchers/mjlab_alpha/rfdej55h
```

Train the same motion-mimic policy (headless, large batch):

```bash
MUJOCO_GL=egl uv run scripts/rl/train.py \
  Mjlab-Tracking-Flat-Unitree-G1 \
  --registry-name gcbc_researchers/csv_to_npz/lafan_cartwheel \
  --env.scene.num-envs 4096
```

Add a new motion to the WandB registry from a CSV:

```bash
MUJOCO_GL=egl uv run scripts/csv_to_npz.py \
  --input-file /path/to/motion.csv \
  --output-name side_kick \
  --input-fps 30 \
  --output-fps 50 \
  --render
```

### Debugging

Use dummy agents for quick environment checks (velocity envs only):

```bash
uv run scripts/velocity/random_agent.py --task Mjlab-Velocity-Flat-Unitree-G1
```

```bash
uv run scripts/velocity/zero_agent.py --task Mjlab-Velocity-Flat-Unitree-G1
```

## Running tests

```bash
make test
```

## Code formatting and linting

You can install a pre-commit hook:

```bash
uvx pre-commit install
```

or manually format with:

```
make format
```

## Troubleshooting

**CUDA Compatibility**: Not all CUDA versions are supported. Check [mujoco_warp#101](https://github.com/google-deepmind/mujoco_warp/issues/101) for your CUDA version compatibility.

## License

This project, **mjlab**, is licensed under the [Apache License, Version 2.0](LICENSE).

### Third-Party Code

The `third_party/` directory contains selected files from external projects.  
Each such subdirectory includes its own original `LICENSE` file from the upstream source.  
These files are used under the terms of their respective licenses.

Currently, `third_party/` contains:

- **isaaclab/** — Selected files from [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab),  
  licensed under the [BSD-3-Clause](src/mjlab/third_party/isaaclab/LICENSE).

When distributing or modifying this project, you must comply with both:

1. The **Apache-2.0 license** of mjlab (applies to all original code in this repository).
2. The licenses of any code in `third_party/` (applies only to the files from those projects).

See the individual `LICENSE` files for the complete terms.
