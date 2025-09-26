# mjlab

<p align="left">
  <img alt="tests" src="https://github.com/mujocolab/mjlab/actions/workflows/ci.yml/badge.svg" />
</p>

```python
mjlab = isaaclab - isaacsim + mjwarp
```
**Keep the API, ditch the complexity, add the speed**

mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s proven API with best-in-class [MuJoCo](https://github.com/google-deepmind/mujoco_warp) physics to provide lightweight, modular abstractions for RL robotics research and sim-to-real deployment.

> **⚠️ EXPERIMENTAL PREVIEW** 
> 
> This project is in very early experimental stages. APIs, features, and documentation are subject to significant changes. Use at your own risk and expect frequent breaking changes.

## Documentation

- **[Getting Started](docs/getting_started.md)** - Install and run your first training in minutes
- **[Why mjlab?](docs/motivation.md)** - Comparison with Isaac Lab, Newton, and MuJoCo
- **[Migration Guide](docs/migration_guide.md)** - Moving from Isaac Lab (mostly copy-paste configs!)
- **[Architecture](docs/architecture.md)** - Understanding mjlab's design and MuJoCo integration
- **[FAQ & Troubleshooting](docs/faq.md)** - Common questions and answers

## Quick Start

### Installation

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

### Velocity tracking

Train a Unitree G1 to follow velocity commands (headless, large batch):

```bash
MUJOCO_GL=egl uv run scripts/velocity/rl/train.py \
  Mjlab-Velocity-Flat-G1 \
  --env.scene.num-envs 4096
```

Play the trained policy:

```bash
uv run scripts/velocity/rl/play.py \
  --task Mjlab-Velocity-Flat-G1-Play
```

### Motion imitation

Before running motion mimicking, you'll need to set up a WandB registry for reference motions. Follow the detailed instructions here: [Motion Preprocessing & Registry Setup](https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup)

**Quick setup summary:**
1. Create a WandB registry collection named "Motions"
2. Process and upload your motion files:
   ```bash
   MUJOCO_GL=egl uv run scripts/tracking/csv_to_npz.py \
     --input-file /path/to/motion.csv \
     --output-name motion_name \
     --input-fps 30 \
     --output-fps 50 \ 
     --render
   ```
3. Set your WandB entity: `export WANDB_ENTITY=your-organization-name`

The `--render` flag will output a video of the motion which you can inspect for quick debugging.

#### Training and Playing

Run a pre-trained motion-mimic policy on the G1:

```bash
uv run scripts/tracking/rl/play.py \
  --task Mjlab-Tracking-Flat-G1-Play \
  --wandb-run-path your-org/mjlab/run-id
```

Train a motion-mimic policy (headless, large batch):

```bash
MUJOCO_GL=egl uv run scripts/tracking/rl/train.py \
  Mjlab-Tracking-Flat-G1 \
  --registry-name your-org/motions/motion-name \
  --env.scene.num-envs 4096
```

### Debugging

Use dummy agents for quick environment checks (velocity envs only):

```bash
uv run scripts/velocity/random_agent.py --task Mjlab-Velocity-Flat-G1
```

```bash
uv run scripts/velocity/zero_agent.py --task Mjlab-Velocity-Flat-G1
```

## 🛠️ Development

### Running Tests
```bash
make test
```

### Code Formatting
```bash
# Install pre-commit hook.
uvx pre-commit install

# Manual formatting.
make format
```

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
