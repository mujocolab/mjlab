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
- [ ] `rsl_rl` integration
  - [ ] `VecEnv` wrapper
  - [ ] `train.py` and `play.py`
- [ ] Observation history and modifiers