# mjlab

IsaacLab API with [MJWarp](https://github.com/google-deepmind/mujoco_warp) backend.

## Installation

1. Install `mjwarp`.

```bash
pip install warp-lang --pre --upgrade -f https://pypi.nvidia.com/warp-lang/
pip install mujoco --pre --upgrade -f https://py.mujoco.org/
git clone https://github.com/google-deepmind/mujoco_warp.git && cd mujoco_warp
git checkout 303de9f  # IMPORTANT!
pip install -e .[dev,cuda] --config-settings editable_mode=strict
```

2. [Install](https://pytorch.org/get-started/locally/) `pytorch`. On a 5090, I used the below command:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

3. Install mjlab: `pip install -e .[test,rl]`

## Getting started

### Motion Mimic

```bash
# Train.
MUJOCO_GL=egl python scripts/tracking/rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --registry-name gcbc_researchers/csv_to_npz/run1_subject5:v0 \
  --num_envs 4096
```

```bash
# Play.
python scripts/tracking/rl/play.py \
  --task Tracking-Flat-G1-v0 \
  --wandb-run-path gcbc_researchers/mjlab_alpha/7nm9duu4
```

## TODO

**P0**.

- [ ] Domain randomization
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
  - [ ] Curriculum manager
  - [ ] Event manager
    - [ ] startup
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
python -c "import warp; warp.clear_kernel_cache()"
```
