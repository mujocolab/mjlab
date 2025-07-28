# mjlab

IsaacLab API with [MJWarp](https://github.com/google-deepmind/mujoco_warp) backend.

## Installation

1. Install `mjwarp`.

```bash
pip install warp-lang --pre --upgrade -f https://pypi.nvidia.com/warp-lang/
pip install mujoco --pre --upgrade -f https://py.mujoco.org/
git clone https://github.com/google-deepmind/mujoco_warp.git
pip install -e .[dev,cuda] --config-settings editable_mode=strict
```

2. [Install](https://pytorch.org/get-started/locally/) `pytorch`. On a 5090, I used the below command:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

3. Install mjlab: `pip install -e .[test]`

## TODO

- [ ] Domain randomization
- [ ] Observation history and modifiers
- [ ] Managers
  - [x] Action manager
  - [x] Command manager
  - [ ] Curriculum manager
  - [ ] Event manager
    - [ ] startup
    - [x] reset
    - [ ] interval 
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
- [ ] Cosmetic improvements
  - [ ] Redo dataclass config correctly

## Debug

### Reset warp cache

```bash
rm -r /home/kevin/.cache/warp
```