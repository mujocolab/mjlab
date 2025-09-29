# mjlab

<p align="left">
  <img alt="tests" src="https://github.com/mujocolab/mjlab/actions/workflows/ci.yml/badge.svg" />
</p>

mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s proven API with best-in-class [MuJoCo](https://github.com/google-deepmind/mujoco_warp) physics to provide lightweight, modular abstractions for RL robotics research and sim-to-real deployment.

```bash
uvx --from mjlab --with "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp" demo
```

> **⚠️ BETA PREVIEW** 
> 
> This project is in beta. There might be breaking changes and missing features.

## Why mjlab?

- **Familiar APIs**: If you know Isaac Lab or MuJoCo, you already know mjlab
- **Instant Feedback**: Fast startup and kernel caching. Drop a breakpoint anywhere and debug immediately
- **Massively Parallel**: MuJoCo Warp enables efficient GPU-accelerated simulation at scale
- **Zero Friction**: Pure Python, minimal dependencies. Just `uv run` and go

**[Read the full comparison →](docs/motivation.md)**

## Documentation

- **[Why mjlab?](docs/motivation.md)** - When to use mjlab (and when to use Isaac Lab, Newton, etc.)
- **[Migration Guide](docs/migration_guide.md)** - Moving from Isaac Lab
- **[FAQ & Troubleshooting](docs/faq.md)** - Common questions and answers

## Quick Start

### Option 1: Install from PyPI

Install [uv](https://docs.astral.sh/uv/) if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Run the demo directly:

```bash
uvx --from mjlab --with "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp" demo
```

### Option 2: Install from Source (Recommended)

Clone the repository:

```bash
git clone git@github.com:mujocolab/mjlab.git && cd mjlab
```

Then either:

- **Run commands directly** (recommended for development):
  ```bash
  uv run demo
  ```

- **Install as editable package** (if you need to import mjlab elsewhere):
  ```bash
  uv pip install -e . "mujoco-warp @ git+https://github.com/google-deepmind/mujoco_warp"
  ```

## Training Examples

### 1. Velocity Tracking

Train a Unitree G1 humanoid to follow velocity commands on flat terrain:

```bash
MUJOCO_GL=egl uv run train \
  Mjlab-Velocity-Flat-Unitree-G1 \
  --env.scene.num-envs 4096

# NOTE: You can evaluate a policy while your training is still
# in progress. This will grab the latest checkpoint from wandb.
uv run play \
  --task Mjlab-Velocity-Flat-Unitree-G1-Play \
  --wandb-run-path your-org/mjlab/run-id
```

### 2. Motion Imitation

Train a Unitree G1 to mimic reference motions. mjlab uses [WandB](https://wandb.ai) to manage reference motion datasets:

1. **Create a registry collection** in your WandB workspace named `Motions`

2. **Set your WandB entity**:
   ```bash
   export WANDB_ENTITY=your-organization-name
   ```

3. **Process and upload motion files**:
   ```bash
   MUJOCO_GL=egl uv run scripts/tracking/csv_to_npz.py \
     --input-file /path/to/motion.csv \
     --output-name motion_name \
     --input-fps 30 \
     --output-fps 50 \
     --render  # Optional: generates preview video
   ```

> **Note**: For detailed motion preprocessing instructions, see the [BeyondMimic documentation](https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup).

#### Train and Play

```bash
MUJOCO_GL=egl uv run train \
  Mjlab-Tracking-Flat-Unitree-G1 \
  --registry-name your-org/motions/motion-name \
  --env.scene.num-envs 4096

uv run play \
  --task Mjlab-Tracking-Flat-Unitree-G1-Play \
  --wandb-run-path your-org/mjlab/run-id
```

## Development

### Running Tests

```bash
make test
```

### Code Formatting

```bash
# Install pre-commit hook.
uvx pre-commit install

# Format manually.
make format
```

## License

mjlab is licensed under the [Apache License, Version 2.0](LICENSE).

### Third-Party Code

The `third_party/` directory contains files from external projects, each with its own license:

- **isaaclab/** — Selected files from [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) ([BSD-3-Clause](src/mjlab/third_party/isaaclab/LICENSE))

When distributing or modifying mjlab, comply with:
1. The Apache-2.0 license for mjlab's original code
2. The respective licenses in `third_party/` for those files

See individual `LICENSE` files for complete terms.
