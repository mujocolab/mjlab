# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is mjlab?

mjlab is a GPU-accelerated reinforcement learning framework for robotics. It combines Isaac Lab's manager-based API with MuJoCo-Warp (GPU-parallel MuJoCo). Training requires an NVIDIA GPU; macOS is supported for evaluation only.

## Development Workflow

**Always use `uv run`, not python**.

```sh
# Type check (fast)
uv run ty check

# Type check (thorough)
uv run pyright

# Run a specific test file
uv run pytest tests/<test_file>.py

# Format and lint before committing
uv run ruff format && uv run ruff check --fix
```

Makefile shortcuts:

```sh
make format     # Format and lint
make type       # Type-check (ty + pyright)
make check      # make format && make type
make test-fast  # Run tests excluding @pytest.mark.slow
make test       # Run the full test suite
```

Before creating a PR, ensure all checks pass with `make test`.

## Style Guidelines

- Line length limit is 88 columns (code, comments, and docstrings).
- Indent width is 2 spaces (configured in ruff).
- Avoid local imports unless strictly necessary (e.g. circular imports).
- Tests: use functions and fixtures, not test classes. Favor targeted tests over exhaustive edge-case coverage.

## Architecture

### Core Design Pattern: Cfg + Build

Nearly every component is a paired `FooCfg` dataclass and `Foo` implementation class. Configs are pure data; the `build()` method (or constructor) creates the runtime object. This applies to actuators, sensors, actions, commands, and more.

### Source Layout (`src/mjlab/`)

- **`sim/`** — GPU simulation bridge. `Simulation` owns MuJoCo model/data and their MJWarp GPU counterparts. Uses CUDA graph capture for `step()`/`forward()`/`reset()`/`sense()`. `WarpBridge` and `TorchArray` provide zero-copy Warp-to-PyTorch interop.
- **`scene/`** — Scene composition. `Scene` builds an `MjSpec` from terrain + entities + sensors, compiles it, then initializes everything against the GPU simulation.
- **`entity/`** — Physical objects (robots, objects). `Entity` handles spec construction, actuator attachment, and provides `EntityData` for read/write access to root/body/joint state.
- **`actuator/`** — Actuator models (builtin position/velocity/motor, ideal PD, DC motor, learned MLP, delayed). Builtin actuators are batched; custom actuators are looped individually.
- **`sensor/`** — Sensors (builtin MuJoCo sensors, contact, camera, raycast). Camera and raycast sensors require a shared `SensorContext`.
- **`managers/`** — The manager system that drives the RL loop. Each manager type handles one MDP concern:
  - `ActionManager` — splits policy actions and routes to entities
  - `ObservationManager` — computes obs with noise/clip/scale/delay/history, organized into groups ("actor", "critic")
  - `RewardManager` — weighted reward aggregation
  - `TerminationManager` — terminated/truncated signals
  - `EventManager` — domain randomization and state resets (startup/reset/interval modes)
  - `CommandManager` — goal command generation
  - `CurriculumManager` — adaptive difficulty
  - `MetricsManager` — custom per-step metrics logging
- **`envs/`** — `ManagerBasedRlEnv` is the central RL environment class. `envs/mdp/` contains reusable MDP building blocks (observations, rewards, events, actions, terminations).
- **`tasks/`** — Concrete task definitions. Tasks register via `register_mjlab_task()` at import time. Auto-discovered via recursive import. External packages can register tasks via `mjlab.tasks` entry points.
- **`rl/`** — RSL-RL integration. `RslRlVecEnvWrapper` adapts the env to RSL-RL's interface. `MjlabOnPolicyRunner` extends RSL-RL's `OnPolicyRunner`.
- **`terrains/`** — Procedural terrain generation and import.
- **`asset_zoo/`** — Pre-configured robot assets.
- **`scripts/`** — CLI entry points: `train`, `play`, `demo`, `list_envs`, `viz-nan`.
- **`utils/lab_api/`** — Utilities ported from Isaac Lab (excluded from ruff/pyright).

### Manager Term System

Manager terms (`func` in `*TermCfg`) can be:
- **Functions**: stateless, e.g. `func=mdp.joint_vel_l2`
- **Classes**: stateful, instantiated with `(cfg, env)`, must implement `__call__` and optionally `reset(env_ids)`

When a term needs to reference parts of an entity (specific joints, bodies, geoms), it uses `SceneEntityCfg` in `params`. The manager base class auto-resolves these at initialization, converting regex name patterns to integer IDs.

### Domain Randomization

The `@requires_model_fields` decorator on event functions tells the event manager which MuJoCo model fields need per-environment copies. The environment calls `sim.expand_model_fields()` to allocate them. This must happen before CUDA graphs are captured.

### Configuration via tyro

All configuration uses Python dataclasses, made CLI-configurable via tyro. The train/play scripts use a two-pass pattern: first select task name, then configure with defaults from the registered task. Any config field can be overridden from CLI:
```bash
uv run train Mjlab-Velocity-Rough-Unitree-Go1 --env.scene.num-envs 4096
```

### Task Configuration Pattern

Each task follows: factory function creates base config → robot-specific function customizes it → `register_mjlab_task()` registers it. Tasks live in `tasks/<type>/config/<robot>/`.

### `ManagerBasedRlEnv.step()` Flow

1. Process actions → 2. Decimation loop (apply actions → write to sim → `sim.step()` → update scene) → 3. Compute terminations → 4. Compute rewards → 5. Reset terminated envs → 6. `sim.forward()` → 7. Update commands → 8. Fire interval events → 9. `sim.sense()` → 10. Compute observations.

Manager load order is critical: EventManager first (expands model fields), then CommandManager, then Action/Observation, then the rest.
