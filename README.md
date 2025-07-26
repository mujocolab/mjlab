# mjlab

This is extremely bad code and very much work in progress. I'm just exploring different design patterns for physics simulation + RL workflows.

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
- [ ] Add rendering
- [ ] `rsl_rl` integration
  - [x] `VecEnv` wrapper
  - [ ] `train.py` and `play.py`
- [ ] Add __str__ method for all important classes
- [x] Add some form of contact sensor and sensor API
- [x] Inherit from gym

## Debug

### Reset warp cache

```bash
rm -r /home/kevin/.cache/warp
```