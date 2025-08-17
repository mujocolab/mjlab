**P0**.

- [ ] Contact sensor
  - [ ] Clean up, super gross right now
  - [ ] Investiate cause of NaNs
- [ ] Improve viewer
  - [ ] Package into class
  - [ ] 2 backends: mujoco viewer, viser
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
  - [ ] Reconsider `term` helper
- [ ] Make tyro configure everything in rsl_rl train and eval script
- [ ] Improve type checking across the board
  - [ ] pylance
  - [x] ty
- [x] Switch to `uv`

**P2**

- [ ] Make it possible to seamlessly switch to CPU mujoco