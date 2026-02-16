![Project banner](https://raw.githubusercontent.com/mujocolab/mjlab/main/docs/source/_static/mjlab-banner.jpg)

# mjlab

[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/mujocolab/mjlab/ci.yml?branch=main)](https://github.com/mujocolab/mjlab/actions/workflows/ci.yml?query=branch%3Amain)
[![Documentation](https://github.com/mujocolab/mjlab/actions/workflows/docs.yml/badge.svg)](https://mujocolab.github.io/mjlab/)
[![License](https://img.shields.io/github/license/mujocolab/mjlab)](https://github.com/mujocolab/mjlab/blob/main/LICENSE)
[![Nightly Benchmarks](https://img.shields.io/badge/Nightly-Benchmarks-blue)](https://mujocolab.github.io/mjlab/nightly/)
[![PyPI](https://img.shields.io/pypi/v/mjlab)](https://pypi.org/project/mjlab/)

mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s manager-based API with [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), a GPU-accelerated version of [MuJoCo](https://github.com/google-deepmind/mujoco).
The framework provides composable building blocks for environment design,
with minimal dependencies and direct access to native MuJoCo data structures.

## Getting Started

mjlab requires an NVIDIA GPU for training. macOS is supported for evaluation only.

**Try it now:**

Run the demo (no installation needed):

```bash
uvx --from mjlab --refresh demo
```

Or try in [Google Colab](https://colab.research.google.com/github/mujocolab/mjlab/blob/main/notebooks/demo.ipynb) (no local setup required).

**Install from source:**

```bash
git clone https://github.com/mujocolab/mjlab.git && cd mjlab
uv run demo
```

For alternative installation methods (PyPI, Docker), see the [Installation Guide](https://mujocolab.github.io/mjlab/source/installation.html).

## Training Examples

### 1. Velocity Tracking

Train a Unitree G1 humanoid to follow velocity commands on flat terrain:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096
```

**Multi-GPU Training:** Scale to multiple GPUs using `--gpu-ids`:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --gpu-ids 0 1 \
  --env.scene.num-envs 4096
```

See the [Distributed Training guide](https://mujocolab.github.io/mjlab/source/distributed_training.html) for details.

Evaluate a policy while training (fetches latest checkpoint from Weights & Biases):

```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

### 2. Motion Imitation

Train a humanoid to mimic reference motions. mjlab uses WandB to manage motion datasets.
See the [motion preprocessing documentation](https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup) for setup instructions.

```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1 --registry-name your-org/motions/motion-name --env.scene.num-envs 4096
uv run play Mjlab-Tracking-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

### 3. Sanity-check with Dummy Agents

Use built-in agents to sanity check your MDP before training:

```bash
uv run play Mjlab-Your-Task-Id --agent zero  # Sends zero actions
uv run play Mjlab-Your-Task-Id --agent random  # Sends uniform random actions
```

When running motion-tracking tasks, add `--registry-name your-org/motions/motion-name` to the command.


## Community Projects

mjlab is used for research and robotics applications around the world. Examples:

<table>
  <tr>
    <td>
      <a href="https://github.com/menloresearch/asimov-mjlab">
        menloresearch/asimov-mjlab
        <br /><img
          alt="GitHub stars"
          src="https://img.shields.io/github/stars/menloresearch/asimov-mjlab?style=social"
        />
      </a>
    </td>
    <td>Locomotion fork for the Asimov bipedal robot.</td>
  </tr>
  <tr>
    <td>
      <a href="http://husky-humanoid.github.io/">
        HUSKY
      </a>
      <br />
      <a href="https://github.com/mujocolab/mjlab/discussions/572">#572</a>
      ·
      <a href="https://arxiv.org/abs/2602.03205">Paper</a>
    </td>
    <td>
      Humanoid skateboarding with dynamic balance control.
    </td>
  </tr>
  <tr>
    <td>
      <a href="https://github.com/Nagi-ovo/mjlab-homierl">
        Nagi-ovo/mjlab-homierl
        <br /><img
          alt="GitHub stars"
          src="https://img.shields.io/github/stars/Nagi-ovo/mjlab-homierl?style=social"
        />
      </a>
    </td>
    <td>Multi-task H1 locomotion (walk/squat/stand) with upper-body disturbance robustness.</td>
  </tr>
  <tr>
    <td>
      <a href="https://github.com/MyoHub/mjlab_myosuite">
        MyoHub/mjlab_myosuite
        <br /><img
          alt="GitHub stars"
          src="https://img.shields.io/github/stars/MyoHub/mjlab_myosuite?style=social"
        />
      </a>
    </td>
    <td>Musculoskeletal simulation integration with MyoSuite.</td>
  </tr>
  <tr>
    <td>
      <a href="https://github.com/MarcDcls/mjlab_upkie">
        MarcDcls/mjlab_upkie
        <br /><img
          alt="GitHub stars"
          src="https://img.shields.io/github/stars/MarcDcls/mjlab_upkie?style=social"
        />
      </a>
    </td>
    <td>Velocity control for the Upkie wheeled biped.</td>
  </tr>
  <tr>
    <td>
      <a href="https://github.com/unitreerobotics/unitree_rl_mjlab">
        unitreerobotics/unitree_rl_mjlab
        <br /><img
          alt="GitHub stars"
          src="https://img.shields.io/github/stars/unitreerobotics/unitree_rl_mjlab?style=social"
        />
      </a>
    </td>
    <td>Official Unitree RL environments for Go2, G1, and H1_2.</td>
  </tr>
</table>

Want to share your project? Post in [Show and Tell](https://github.com/mujocolab/mjlab/discussions/categories/show-and-tell)!

## Documentation

Full documentation is available at **[mujocolab.github.io/mjlab](https://mujocolab.github.io/mjlab/)**.

## Development

```bash
make test          # Run all tests
make test-fast     # Skip slow tests
make format        # Format and lint
make docs          # Build docs locally
```

For development setup: `uvx pre-commit install`

## Citation

If you use mjlab in your research, please cite:

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
