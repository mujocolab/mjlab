# mjlab

IsaacLab API with [MJWarp](https://github.com/google-deepmind/mujoco_warp) backend.

## Getting started

### Motion Mimic

```bash
# Train.
MUJOCO_GL=egl uv run --extra rl scripts/tracking/rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --registry-name gcbc_researchers/csv_to_npz/run1_subject5:v0 \
  --num_envs 4096
```

```bash
# Play.
uv run --extra rl scripts/tracking/rl/play.py \
  --task Tracking-Flat-G1-Play-v0 \
  --wandb-run-path gcbc_researchers/mjlab_alpha/rfdej55h
```

## TODO

**P0**.

- [ ] Contact sensor
  - [ ] Clean up, super gross right now
  - [ ] Investiate cause of NaNs
- [x] Domain randomization
- [ ] Terrain generation
- [ ] Observation history and modifiers
- [ ] Rename entities to be consistent with isaaclab
- [ ] Actuation
  - [x] Joint PD control
  - [ ] Learned actuator models
  - [ ] Cleanup to be consistent with isaaclab
- [ ] Managers
  - [x] Action manager
  - [x] Command manager
  - [x] Curriculum manager
  - [x] Event manager
    - [x] startup
    - [x] reset
    - [x] interval
  - [x] Observation manager
  - [ ] Recorder manager
  - [x] Reward manager
  - [x] Termination Manager
- [ ] Add rendering
  - [x] `rgb_array` rendering
  - [ ] `human` rendering
- [x] `rsl_rl` integration
  - [x] `VecEnv` wrapper
  - [x] `train.py` and `play.py`
- [x] Add __str__ method for all important classes
- [x] Add some form of contact sensor and sensor API
- [x] Inherit from gym

**P1**

aka make Brent proud

- [ ] Unit test everything
- [ ] Redo dataclass config correctly
- [ ] Make tyro configure everything in rsl_rl train and eval script
- [ ] Improve type checking across the board
- [ ] Switch to `uv`

**P2**

- [ ] Make it possible to seamlessly switch to CPU mujoco

## Debug

### Reset warp cache

```bash
bash scripts/clear_wp_cache.sh
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
