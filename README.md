# mjlab

<p align="left">
  <img alt="tests" src="https://github.com/mujocolab/mjlab/actions/workflows/ci.yml/badge.svg" />
  <img alt="license" src="https://img.shields.io/github/license/mujocolab/mjlab" />
</p>

> **⚠️ EXPERIMENTAL PREVIEW** 
> 
> This project is in very early experimental stages. APIs, features, and documentation are subject to significant changes. Use at your own risk and expect frequent breaking changes.

IsaacLab API with [MJWarp](https://github.com/google-deepmind/mujoco_warp) backend.

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

Run a pre-trained motion mimic policy on the Unitree G1 humanoid:

```bash
uv run scripts/tracking/rl/play.py \
  --task Mjlab-Tracking-Flat-G1-Play \
  --wandb-run-path gcbc_researchers/mjlab_alpha/rfdej55h
```

You can train this exact motion mimic policy using the following command:

```bash
MUJOCO_GL=egl uv run scripts/tracking/rl/train.py \
  Mjlab-Tracking-Flat-G1 \
  --registry-name gcbc_researchers/csv_to_npz/lafan_cartwheel \
  --env.scene.num-envs 4096
```

To add a new motion to the wandb registry, run:

```bash
MUJOCO_GL=egl uv run scripts/tracking/csv_to_npz.py \
  --input-file /path/to/motion.csv \
  --output-name side_kick \
  --input-fps 30 \
  --output-fps 50 \
  --render
```

### Running tests

```bash
make test
```

### Code formatting and linting

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
