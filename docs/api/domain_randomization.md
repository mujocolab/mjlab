# Domain Randomization

Domain randomization varies physical parameters during training so that policies
are robust to modeling errors and real-world variation. This guide shows
how to attach randomization terms to your environment using `EventTerm` and
`mdp.randomize_field`.

## TL;DR

Use an `EventTerm` that calls `mdp.randomize_field` with a target **field**, a
**value range** (or per-axis ranges), and an **operation** describing how to
apply the draw.

```python
from mjlab.managers.manager_term_config import EventTermCfg as EventTerm
from mjlab.managers.manager_term_config import term
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs import mdp

foot_friction: EventTerm = term(
    EventTerm,
    mode="reset",  # randomize each episode
    func=mdp.randomize_field,
    params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=[".*_foot.*"]),
        "field": "geom_friction",
        "ranges": (0.3, 1.2),
        "operation": "abs",
    },
)
```

## Event Modes

* `"startup"` - randomize once at initialization
* `"reset"` - randomize at every episode reset
* `"interval"` - randomize at regular time intervals

## Available Fields

**Joint/DOF:** `dof_armature`, `dof_frictionloss`, `dof_damping`, `jnt_range`,
`jnt_stiffness`, `qpos0`

**Body:** `body_mass`, `body_ipos`, `body_iquat`, `body_inertia`, `body_pos`,
`body_quat`

**Geom:** `geom_friction`, `geom_pos`, `geom_quat`, `geom_rgba`

**Site:** `site_pos`, `site_quat`

## Randomization Parameters

**Distribution:** `"uniform"` (default), `"log_uniform"` (values must be > 0),
`"gaussian"` (`mean, std`)

**Operation:** `"abs"` (default, set), `"scale"` (multiply), `"add"` (offset)

### Axis selection

Multi-dimensional fields can be randomized per-axis.

**Friction.** Geoms have three coefficients `[tangential, torsional, rolling]`.
For `condim=3` (standard frictional contact), only **axis 0 (tangential)**
affects contact behavior:

```python
# Tangential friction (affects condim=3)
params={"field": "geom_friction", "ranges": {0: (0.3, 1.2)}}

# Tangential + torsional (torsional matters for condim >= 4)
params={"field": "geom_friction", "ranges": {0: (0.5, 1.0), 1: (0.001, 0.01)}}

# X and Y position
params={"field": "body_pos", "axes": [0, 1], "ranges": (-0.1, 0.1)}
```

## Examples

### Friction (reset)

```python
foot_friction: EventTerm = term(
    EventTerm,
    mode="reset",
    func=mdp.randomize_field,
    params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=[".*_foot.*"]),
        "field": "geom_friction",
        "ranges": (0.3, 1.2),
        "operation": "abs",
    },
)
```

> **Tip:** Give your robot's collision geoms higher **priority** than terrain
> (geom priority defaults to 0). Then you only need to randomize robot friction.
> MuJoCo will use the higher-priority geom's friction in (robot, terrain)
> contacts.

```python
from mjlab.utils.spec_config import CollisionCfg

robot_collision = CollisionCfg(
    geom_names_expr=[".*_foot.*"],
    priority=1,
    friction=(0.6,),
    condim=3,
)
```

### Joint Offset (startup)

Randomize default joint positions to simulate joint offset calibration errors:

```python
joint_offset: EventTerm = term(
    EventTerm,
    mode="startup",
    func=mdp.randomize_field,
    params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        "field": "qpos0",
        "ranges": (-0.01, 0.01),
        "operation": "add",
    },
)
```

### Center of Mass (COM) (startup)

```python
com: EventTerm = term(
    EventTerm,
    mode="startup",
    func=mdp.randomize_field,
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=["torso"]),
        "field": "body_ipos",
        "ranges": {0: (-0.02, 0.02), 1: (-0.02, 0.02)},
        "operation": "add",
    },
)
```

## Migrating from Isaac Lab

Isaac Lab exposes explicit friction combination modes (`multiply`, `average`,
`min`, `max`). MuJoCo instead uses **priority-based selection**: if one
contacting geom has higher `priority`, its friction is used; otherwise the
**element-wise maximum** is used. See the
[MuJoCo contact documentation](https://mujoco.readthedocs.io/en/stable/computation/index.html#contact)
for details.
