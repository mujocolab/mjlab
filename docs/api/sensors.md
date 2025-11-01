# Sensors

Sensors provide a configurable way to measure physical quantities in your simulation. They live at the scene level alongside entities and terrain, can reference multiple entities (e.g., detect contact between robot and terrain), and return structured data for use in rewards, terminations, and observations.

## Quick Note: Entity Data vs Sensors

> **Before diving into sensors**, it's helpful to understand that mjlab provides two complementary ways to access simulation data:

**Entity Data** (`entity.data.*`)
- Common quantities are available out of the box with zero configuration
- Provides convenient coordinate frame transformations between world and body frames, COM and link frames
- Offers a familiar API for users coming from Isaac Lab
- Example: `robot.data.root_link_lin_vel_b`, `robot.data.joint_pos`

**Sensors** (this system)
- Provides reusable, configurable sensor definitions that can be shared across tasks
- Extensible through subclassing to add custom logic like noise, filtering, or processing
- Maps directly to real robot sensors like IMUs, force sensors, and cameras
- Example: `env.scene["feet_contact"].data`, `env.scene["robot/imu"].data`

**Use them together:**
- Use entity data for quick access to state and transforms
- Use sensors for measurements that span entities or need configuration
- Mix both approaches freely in your rewards and observations based on your needs

```python
from mjlab.sensor import BuiltinSensorCfg, ContactSensorCfg, ContactMatch, ObjRef

scene_cfg = SceneCfg(
  entities={"robot": robot_cfg},
  sensors=(
    BuiltinSensorCfg(
      name="imu_acc",
      sensor_type="accelerometer",
      obj=ObjRef(type="site", name="imu_site", entity="robot"),
    ),
    ContactSensorCfg(
      name="feet_contact",
      primary=ContactMatch(mode="geom", pattern=r".*_foot$", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
    ),
  ),
)

# Access at runtime.
imu_acc_data = env.scene["robot/imu_acc"].data  # [B, 3] acceleration
feet_contact = env.scene["feet_contact"].data   # ContactData with .found, .force
```

## Sensor Types

mjlab currently provides two sensor implementations:

### BuiltinSensor

Wraps MuJoCo's native sensor types (57 total) for measuring forces, positions,
velocities, and other physical quantities. Returns raw `torch.Tensor` data.

### ContactSensor

Detects contacts between bodies, geoms, or subtrees. Returns structured
`ContactData` with forces, positions, air time metrics, etc.

---

## BuiltinSensor

### Sensor Types

| Category | Available Sensors |
|----------|------------------|
| **Site** | `accelerometer`, `velocimeter`, `gyro`, `force`, `torque`, `magnetometer`, `rangefinder` |
| **Joint** | `jointpos`, `jointvel`, `jointlimitpos`, `jointlimitvel`, `jointlimitfrc`, `jointactuatorfrc` |
| **Frame** | `framepos`, `framequat`, `framexaxis`, `frameyaxis`, `framezaxis`, `framelinvel`, `frameangvel`, `framelinacc`, `frameangacc` |
| **Other** | `actuatorpos`, `actuatorvel`, `actuatorfrc`, `subtreecom`, `subtreelinvel`, `subtreeangmom`, `clock`, `e_potential`, `e_kinetic` |

### Usage

BuiltinSensor returns a `torch.Tensor` with shape `[N_envs, dim]` where dim depends on the sensor type (e.g., 3 for vectors, 4 for quaternions). Configure with `BuiltinSensorCfg`, specifying the sensor type, attached object via `ObjRef`, and optional parameters like `cutoff` to limit output magnitude or `ref` for frame sensors.

### Examples

```python
# Accelerometer.
imu_acc = BuiltinSensorCfg(
  name="imu_acc",
  sensor_type="accelerometer",
  obj=ObjRef(type="site", name="imu_site", entity="robot"),
)

# Joint limits.
joint_limit = BuiltinSensorCfg(
  name="knee_limit",
  sensor_type="jointlimitpos",
  obj=ObjRef(type="joint", name="knee_joint", entity="robot"),
  cutoff=0.1,
)

# Frame tracking (relative position).
ee_pos = BuiltinSensorCfg(
  name="ee_pos",
  sensor_type="framepos",
  obj=ObjRef(type="body", name="end_effector", entity="robot"),
  ref=ObjRef(type="body", name="base", entity="robot"),
)
```

---

## ContactSensor

ContactSensor detects and reports contact between bodies, geoms, or entire subtrees in your simulation. It's particularly useful for foot contact detection, self-collision monitoring, and measuring ground reaction forces. The sensor tracks contacts between a "primary" set of objects (e.g., robot feet) and an optional "secondary" set (e.g., terrain), returning structured data including forces, positions, and timing information.

### Pattern Matching

Use `ContactMatch` to specify what to track:

```python
ContactMatch(
  mode="geom",              # "geom", "body", or "subtree"
  pattern=r".*_foot$",      # Regex or list of names
  entity="robot",           # Optional entity scope
  exclude=(r".*_heel$",),   # Optional exclusions
)
```

Patterns can be:
- **List of exact names:** `["left_foot", "right_foot"]`
- **Regex:** `r".*_collision$"` (expands to all matches)
- **With exclusions:** Filter out specific matches

### Configuration

```python
ContactSensorCfg(
  name="feet_ground",
  primary=ContactMatch(...),     # What to track
  secondary=ContactMatch(...),   # Optional filter
  fields=("found", "force"),     # Data to extract
  reduce="maxforce",              # Contact selection
  num_slots=1,                    # Contacts per primary
  track_air_time=False,           # Landing/takeoff tracking
  global_frame=False,             # Force frame
)
```

**Fields:** `"found"`, `"force"`, `"torque"`, `"dist"`, `"pos"`, `"normal"`, `"tangent"`

**Reduction modes:**
- `"none"` - Fast, non-deterministic
- `"mindist"` - Closest contacts
- `"maxforce"` - Strongest contacts
- `"netforce"` - Returns single synthetic contact at force-weighted centroid with net wrench

### Output: ContactData

```python
@dataclass
class ContactData:
  found: Tensor | None        # [B, N] contact count
  force: Tensor | None        # [B, N, 3]
  torque: Tensor | None       # [B, N, 3]
  dist: Tensor | None         # [B, N] penetration
  pos: Tensor | None          # [B, N, 3] position
  normal: Tensor | None       # [B, N, 3] primary→secondary
  tangent: Tensor | None      # [B, N, 3]

  # With track_air_time=True.
  current_air_time: Tensor | None
  last_air_time: Tensor | None
  current_contact_time: Tensor | None
  last_contact_time: Tensor | None
```

Shape: `[B, N * num_slots]` where N = number of primary matches

### Understanding num_slots

- **`num_slots=1`** (most common): Single representative contact per match
- **`num_slots > 1`**: Multiple contact points per geom/body
- **`reduce="netforce"`**: Always returns exactly one contact regardless of num_slots

```python
# 4 feet, 1 contact each → [B, 4].
ContactSensorCfg(primary=ContactMatch(pattern=["LF", "RF", "LH", "RH"]), num_slots=1)

# 4 feet, 3 contacts each → [B, 12].
ContactSensorCfg(primary=ContactMatch(pattern=["LF", "RF", "LH", "RH"]), num_slots=3)
```

### Examples

```python
# Foot contacts with forces.
feet = ContactSensorCfg(
  name="feet_ground",
  primary=ContactMatch(mode="geom", pattern=r".*_foot$", entity="robot"),
  secondary=ContactMatch(mode="body", pattern="terrain"),
  fields=("found", "force", "pos"),
  reduce="maxforce",
)

# Self-collision detection.
self_collision = ContactSensorCfg(
  name="self_collision",
  primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
  secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
  fields=("found",),
)

# Air time tracking for gait analysis.
feet_air = ContactSensorCfg(
  name="feet_air",
  primary=ContactMatch(pattern=["LF", "RF", "LH", "RH"], entity="robot"),
  track_air_time=True,
  fields=("found",),
)

# Net ground reaction force.
grf = ContactSensorCfg(
  name="grf",
  primary=ContactMatch(mode="subtree", pattern=["left_ankle", "right_ankle"], entity="robot"),
  secondary=ContactMatch(mode="body", pattern="terrain"),
  fields=("force",),
  reduce="netforce",
)
```

---

## Auto-discovery

Sensors defined in an entity's XML are automatically discovered and prefixed with the entity's name.

```xml
<!-- In robot.xml -->
<sensor>
  <accelerometer name="trunk_imu" site="imu_site"/>
  <jointpos name="hip_sensor" joint="hip_joint"/>
</sensor>
```

```python
imu = env.scene["robot/trunk_imu"]
hip = env.scene["robot/hip_sensor"]
```

## Usage Patterns

**In observations**

```python
def imu_acc_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
  sensor = env.scene["robot/imu_acc"]
  return sensor.data  # [N_envs, 3]
```

**In rewards**

```python
def foot_slip(env: ManagerBasedRlEnv) -> torch.Tensor:
  sensor = env.scene["feet_ground"]
  vel = sensor.data.force[..., :2].norm(dim=-1)
  in_contact = sensor.data.found > 0
  return -torch.where(in_contact, vel, 0.0).mean(dim=1)
```

**In terminations**

```python
def illegal_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
  sensor = env.scene["nonfoot_contact"]
  return torch.any(sensor.data.found, dim=-1)  # [B]
```

**Air time helpers**

```python
sensor = env.scene["feet_air"]
first_contact = sensor.compute_first_contact(dt=0.01)  # Just landed
first_air = sensor.compute_first_air(dt=0.01)          # Just took off
```
