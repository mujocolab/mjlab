# Spec Configuration Classes

## Overview

The spec configuration system provides a declarative, pattern-based interface
for configuring MuJoCo models. Define configurations once and apply them to
multiple joints or geoms using regex patterns, rather than writing loops to
modify individual elements.

Configuration classes are applied during `Entity` initialization, after your
custom spec function (if provided) returns the base spec.

## Quick Start

```python
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg, Entity
from mjlab.utils.spec_config import ActuatorCfg, CollisionCfg

# Define a robot with actuators and collision settings.
robot_cfg = EntityCfg(
  spec_fn=get_robot_spec,
  articulation=EntityArticulationInfoCfg(
    actuators=(
      ActuatorCfg(
        joint_names_expr=[".*_joint"],
        effort_limit=10.0,
        stiffness=15.0,
        damping=2.0
      ),
    )
  ),
  collisions=(
    CollisionCfg(
      geom_names_expr=[".*_collision"],
      contype=1,
      conaffinity=1
    ),
  )
)

# Configuration is automatically applied during entity creation.
robot = Entity(robot_cfg)
```

---

## Configuration Classes

### CollisionCfg

Controls collision properties for geoms using pattern matching. Uses regex
patterns to select geoms, then applies collision attributes. For dict-based
parameters, the first matching pattern wins for each geom.

#### Real-World Examples

**Go1 Quadruped - Feet Only Collision:**
```python
# From go1_constants.py
_foot_regex = "^[FR][LR]_foot_collision$"

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=[_foot_regex],  # Only match foot collision geoms.
  contype=0, conaffinity=1,       # Disable self-collisions.
  condim=3,
  priority=1,                     # Higher priority for foot contacts.
  friction=(0.6,),                # Sliding friction coefficient.
  solimp=(0.9, 0.95, 0.023),      # Solver impedance parameters.
)
```

**G1 Humanoid - Full Collision with special foot configuration:**
```python
# From g1_constants.py
FULL_COLLISION = CollisionCfg(
  geom_names_expr=[".*_collision"],  # Match ALL collision geoms.
  condim={
    r"^(left|right)_foot[1-7]_collision$": 3,
    ".*_collision": 1,
  },
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={
    r"^(left|right)_foot[1-7]_collision$": (0.6,)  # Custom friction for feet.
  }
)
```

**G1 Humanoid - No Self-Collision:**
```python
FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=[".*_collision"],
  contype=0,
  conaffinity=1,
  condim={
    r"^(left|right)_foot[1-7]_collision$": 3,
    ".*_collision": 1,
  },
  priority={
    r"^(left|right)_foot[1-7]_collision$": 1
  },
  friction={
    r"^(left|right)_foot[1-7]_collision$": (0.6,)
  }
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `geom_names_expr` | `list[str]` | *required* | Regex patterns to match geom names |
| `contype` | `int` or `dict[str, int]` | `1` | Collision type bitmask |
| `conaffinity` | `int` or `dict[str, int]` | `1` | Collision affinity bitmask |
| `condim` | `int` or `dict[str, int]` | `3` | Contact dimensions: `1`, `3`, `4`, `6` |
| `priority` | `int` or `dict[str, int]` | `0` | Collision priority |
| `friction` | `tuple` or `dict[str, tuple]` | `None` | Friction coefficients (sliding, torsional, rolling) |
| `solref` | `tuple` or `dict[str, tuple]` | `None` | Solver reference parameters |
| `solimp` | `tuple` or `dict[str, tuple]` | `None` | Solver impedance parameters |
| `disable_other_geoms` | `bool` | `True` | Disables collisions for all non-matching geoms |

For a detailed explanation of the above collision parameters,
see the [MuJoCo Contact Documentation](https://mujoco.readthedocs.io/en/stable/computation/index.html#contact).

---

### ActuatorCfg

Configures PD-controlled position actuators for joints.

#### Basic Example
```python
ActuatorCfg(
  joint_names_expr=["shoulder_joint", "elbow_joint"],
  effort_limit=25.0,   # Maximum torque (Nm).
  stiffness=100.0,     # Position gain (P).
  damping=10.0         # Velocity gain (D).
)
```

#### Pattern Matching Examples

**Match all arm joints:**
```python
ActuatorCfg(
  joint_names_expr=[".*_arm_.*"],
  effort_limit=50.0,
  stiffness=200.0,
  damping=20.0
)
```

**Multiple patterns:**
```python
ActuatorCfg(
  joint_names_expr=[
    ".*_elbow_joint",
    ".*_shoulder_pitch_joint",
    ".*_wrist_.*"
  ],
  effort_limit=25.0,
  stiffness=100.0,
  damping=10.0
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `joint_names_expr` | `list[str]` | *required* | Regex patterns to match joint names |
| `effort_limit` | `float` | *required* | Maximum force/torque (must be > 0) |
| `stiffness` | `float` | *required* | Position gain for PD control (>= 0) |
| `damping` | `float` | *required* | Velocity gain for PD control (>= 0) |
| `frictionloss` | `float` | `0.0` | Joint friction coefficient (>= 0) |
| `armature` | `float` | `0.0` | Reflected rotor inertia (>= 0) |

**Important:** Joints must have position limits defined to use position control.

## Actuator Configuration Approach

MJLab uses a per-actuator configuration model where you define one `ActuatorCfg`
per actuator type on your robot.

**Why this approach:**

In real robots, each joint is driven by a specific actuator model with fixed
electromechanical properties. The motor's torque constant, back-EMF constant,
rotor inertia, and gearbox ratio determine the joint-level effort limits,
reflected inertia (armature), and appropriate control gains. These are physical
constants of the actuator, not free parameters to tune per joint.

**The MJLab approach:**

Define one `ActuatorCfg` per actuator type, matching joints to their physical
actuators using regex patterns:

```python
# Taken from the Unitree G1/
ACTUATOR_7520_22 = ActuatorCfg(
  joint_names_expr=[".*_hip_roll_joint", ".*_knee_joint"],
  effort_limit=139.0,
  armature=ARMATURE_7520_22,
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
)

ACTUATOR_5020 = ActuatorCfg(
  joint_names_expr=[".*_elbow_joint", ".*_shoulder_.*_joint"],
  effort_limit=25.0,
  armature=ARMATURE_5020,
  stiffness=STIFFNESS_5020,
  damping=DAMPING_5020,
)
```

This makes the configuration more maintainable and physically grounded. If you
know your robot uses Unitree 7520 motors with 22:1 gearboxes on the hips and
knees, you define one configuration for those actuators rather than manually
specifying parameters for each individual joint.

**Special case: Parallel linkages**

Some joints are driven by parallel mechanisms (e.g., four-bar linkages with dual
actuators). For the G1's waist and ankle joints, which use two 5020 actuators in
parallel, the effective joint-level properties are configuration-dependent. As a
first-order approximation, we assume a nominal 1:1 gear ratio and approximate
the armature as the sum of both actuators' reflected inertias, and similarly
double the effort limits and gains:

```python
G1_ACTUATOR_ANKLE = ActuatorCfg(
  joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.armature * 2,
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
)
```

---

### ActuatorSetCfg

Applies multiple actuator configurations. Actuators are created in spec joint
order for deterministic behavior.

#### Example
```python
EntityArticulationInfoCfg(actuators=(
  # Unitree 7520 motors (hips and knees).
  ActuatorCfg(
    joint_names_expr=[".*_hip_.*", ".*_knee_.*"],
    effort_limit=139.0,
    stiffness=150.0,
    damping=15.0
  ),
  # Unitree 5020 motors (shoulders and elbows).
  ActuatorCfg(
    joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*"],
    effort_limit=25.0,
    stiffness=50.0,
    damping=5.0
  ),
))
```

If no joints match any pattern, an error is raised with available joint names.

---

### SensorCfg

Adds sensors to measure physical quantities.

#### Examples

**IMU sensor:**
```python
SensorCfg(
  name="base_imu",
  sensor_type="accelerometer",
  objtype="body",
  objname="base_link"
)
```

**Gyroscope:**
```python
SensorCfg(
  name="base_gyro",
  sensor_type="gyro",
  objtype="body",
  objname="base_link"
)
```

#### Available Sensor Types

| Type | Measures | Dimensions |
|------|----------|------------|
| `accelerometer` | Linear acceleration | 3 |
| `gyro` | Angular velocity | 3 |
| `velocimeter` | Linear velocity | 3 |
| `framequat` | Orientation (quaternion) | 4 |
| `framepos` | Position | 3 |
| `framelinvel` | Linear velocity | 3 |
| `frameangvel` | Angular velocity | 3 |
| `upvector` | Up vector | 3 |
| `framezaxis` | Up vector | 3 |
| `subtreeangmom` | Angular momentum | 3 |

---

### ContactSensorCfg

Detects and measures contact forces between objects.

#### Examples

**Any contact with a body:**
```python
ContactSensorCfg(
  name="hand_contacts",
  body1="hand"
)
```

**Contact between two specific geoms:**
```python
ContactSensorCfg(
  name="gripper_object",
  geom1="gripper_left",
  geom2="object"
)
```

**Self-collisions within a subtree:**
```python
ContactSensorCfg(
  name="arm_self_collision",
  subtree1="left_arm",
  subtree2="left_arm"
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Sensor name |
| **Primary object (exactly one):** ||||
| `geom1` | `str` | `None` | Primary geom name |
| `body1` | `str` | `None` | Primary body name |
| `subtree1` | `str` | `None` | Primary subtree name |
| `site` | `str` | `None` | Site volume (requires secondary object) |
| **Secondary object (optional, at most one):** ||||
| `geom2` | `str` | `None` | Secondary geom name |
| `body2` | `str` | `None` | Secondary body name |
| `subtree2` | `str` | `None` | Secondary subtree name |
| **Options:** ||||
| `num` | `int` | `1` | Max contacts to track |
| `data` | `tuple` | `("found",)` | Data to extract: `"found"`, `"force"`, `"torque"`, `"dist"`, `"pos"`, `"normal"`, `"tangent"` |
| `reduce` | `str` | `"none"` | Combine method: `"none"`, `"mindist"`, `"maxforce"`, `"netforce"` |

---

### Visual Elements

#### TextureCfg

Creates textures for materials.

```python
TextureCfg(
  name="checker",
  type="2d",
  builtin="checker",
  rgb1=(0.2, 0.3, 0.4),
  rgb2=(0.8, 0.8, 0.8),
  width=256,
  height=256
)
```

#### MaterialCfg

Defines surface materials with optional textures.

```python
MaterialCfg(
  name="rubber",
  texuniform=True,
  texrepeat=(2, 2),
  reflectance=0.5,
  texture="checker"  # References texture by name.
)
```

#### LightCfg

Adds lighting to the scene.

```python
LightCfg(
  name="spotlight",
  body="world",
  type="spot",
  pos=(2.0, 2.0, 3.0),
  dir=(0.0, 0.0, -1.0),
  cutoff=45.0
)
```

#### CameraCfg

Adds a camera.

```python
CameraCfg(
  name="front_cam",
  # Adds the camera to the world body.
  body="world",
  fovy=60.0,
  pos=(3.0, 0.0, 1.5),
  quat=(0.924, 0.383, 0.0, 0.0)
)
```

---

## Custom Spec Functions

For modifications beyond what the configuration classes support, provide a
custom `spec_fn` that returns a modified `MjSpec`. Configuration classes are
applied after your custom function runs.

### Basic Custom Spec Function

```python
def get_custom_spec() -> mujoco.MjSpec:
  """Create a spec with custom modifications."""
  spec = mujoco.MjSpec.from_file("robot.xml")

  # Custom modifications.
  for geom in spec.geoms:
    if "foot" in geom.name:
      geom.friction = (0.8, 0.1, 0.005)
      geom.condim = 4

  return spec

robot_cfg = EntityCfg(
  spec_fn=get_custom_spec,
  collisions=(CollisionCfg(...),),  # Applied after custom function.
)
```

---

## Architecture

```
┌─────────────────────┐
│  EntityCfg          │  High-level robot configuration
│  - spec_fn          │
│  - actuators        │
│  - collisions       │
│  - sensors          │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│  SpecCfg classes    │  Declarative configuration
│  - CollisionCfg     │
│  - ActuatorSetCfg   │
│  - SensorCfg        │
└──────────┬──────────┘
           │ .edit_spec()
           v
┌─────────────────────┐
│  MjSpec             │  MuJoCo specification
│  (low-level API)    │
└─────────────────────┘
```

The configuration classes provide a declarative interface to the MjSpec API,
letting you define configurations with regex patterns rather than writing loops
to modify individual elements.
