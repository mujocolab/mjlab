**P0**.

- [ ] Sensor API re-design
  - [ ] Contact sensor
  - [ ] Arbitrary XML sensors
- [ ] Think about how to randomize collision geometry
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
  - [x] Observation manager
  - [ ] Recorder manager
  - [x] Reward manager
  - [x] Termination Manager
- [ ] Add rendering
  - [x] `rgb_array` rendering
  - [ ] `human` rendering
- [x] `rsl_rl` integration
- [x] Add __str__ method for all important classes
- [x] Add some form of contact sensor and sensor API
- [x] Inherit from gym
- [x] Improve viewer
- [x] Domain randomization

**P1**

- [ ] Unit test everything
- [ ] Improve dataclass configs across the board
- [ ] Make tyro configure everything in rsl_rl train and eval script
- [x] Improve type checking across the board
- [x] Switch to `uv`
