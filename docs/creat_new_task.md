# Create New Task

> **🚧 Work in Progress**  
> This guide walks you step by step through creating a new task, training it, and performing inference.

## 1. Prepare Your Robot 
For a minimal demonstration, we’ll create a simple CartPole environment without requiring any .stl files.

First, navigate to the directory `mjlab\src\mjlab\asset_zoo\robots`, and create the following structure:

```
cartpole
 │ cartpole_constants.py  # Defines the CartPole robot configuration and loads the MuJoCo model.
 │ __init__.py            # Marks the folder as a Python package.
 │
 └─xmls
      cartpole.xml        # Describes the CartPole's physical structure and actuator setup.
```

### 1) cartpole.xml

```xml
<mujoco model="inverted_pendulum">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option timestep="0.02" gravity="0 0 -9.81"/>

  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="5 5 0.1" rgba="0.8 0.9 0.8 1"/>

    <body name="cart" pos="0 0 0.1">
      <geom type="box" size="0.2 0.1 0.1" rgba="0.2 0.2 0.8 1"/>
      <joint name="slide" type="slide" axis="1 0 0" limited="true" range="-2 2"/>

      <body name="pole" pos="0 0 0.1">
        <geom type="capsule" size="0.05 0.5" fromto="0 0 0 0 0 1" rgba="0.8 0.2 0.2 1"/>
        <joint name="hinge" type="hinge" axis="0 1 0" range="-90 90"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <velocity name="slide_velocity" joint="slide" ctrlrange="-20 20" kv="40"/>
  </actuator>

  <keyframe>
    <key name="pendulum_init" qpos="0 0" qvel="0 0" ctrl="0 0"/>
  </keyframe>
</mujoco>

```

### 2) cartpole_constants.py

```python
from pathlib import Path
import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import Entity, EntityCfg, EntityArticulationInfoCfg
from mjlab.utils.spec_config import ActuatorCfg

# Path to the MuJoCo XML file
CARTPOLE_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "robots" / "inverted_pendulum" / "xmls" / "cartpole.xml"
)
assert CARTPOLE_XML.exists(), f"XML not found: {CARTPOLE_XML}"


def get_spec() -> mujoco.MjSpec:
    """Load the MuJoCo model specification."""
    return mujoco.MjSpec.from_file(str(CARTPOLE_XML))


# Actuator configuration: controls the slide joint (the cart)
CARTPOLE_ACTUATOR = ActuatorCfg(
    joint_names_expr=["slide"],   # Joint to control
    effort_limit=5.0,             # Maximum force/effort
    stiffness=0.0,                # No position stiffness (velocity control)
    damping=0.1,                  # Damping factor
)

# Articulation configuration
CARTPOLE_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(CARTPOLE_ACTUATOR,),
)

# Entity configuration
CARTPOLE_ROBOT_CFG = EntityCfg(
    spec_fn=get_spec,
    articulation=CARTPOLE_ARTICULATION,
)

if __name__ == "__main__":
    import mujoco.viewer as viewer

    # Initialize the entity and launch the MuJoCo viewer
    robot = Entity(CARTPOLE_ROBOT_CFG)
    viewer.launch(robot.spec.compile())
```

### 3) __init__.py
It’s just an empty file used to mark the folder as a Python package.

## 2. Register Robot

Open file `mjlab\src\mjlab\asset_zoo\robots\__init__.py`, add cartpole robot item.

```python
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ROBOT_CFG
from mjlab.asset_zoo.robots.unitree_go1.go1_constants import GO1_ROBOT_CFG
from mjlab.asset_zoo.robots.cartpole.cartpole_constants import CARTPOLE_ROBOT_CFG # New robot item

__all__ = (
  "G1_ROBOT_CFG",
  "GO1_ROBOT_CFG",
  "CARTPOLE_ROBOT_CFG", # New robot item
)
```

## 3. Train

## 4. Inference


**Note:** We use consistent `CamelCase` naming conventions (e.g., `RlEnv` instead of `RLEnv`).

### 2. Configuration Classes

Isaac Lab uses `@configclass`, mjlab uses Python's standard `@dataclass` with a `term()` helper.

**Isaac Lab:**
```python
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
```

**mjlab:**
```python
@dataclass
class RewardCfg:
    motion_global_root_pos: RewTerm = term(
        RewTerm,
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_root_ori: RewTerm = term(
        RewTerm,
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
```

### 3. Scene Configuration

Scene setup is more streamlined in mjlab—no Omniverse/USD scene graphs. Instead, you configure materials, lights, and textures directly through MuJoCo's MjSpec modifiers.

**Isaac Lab:**
```python
from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

**mjlab:**
```python
from mjlab.scene import SceneCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ROBOT_CFG
from mjlab.utils.spec_config import ContactSensorCfg
from mjlab.terrains import TerrainImporterCfg

# Configure contact sensor
self_collision_sensor = ContactSensorCfg(
    name="self_collision",
    subtree1="pelvis",
    subtree2="pelvis",
    data=("found",),
    reduce="netforce",
    num=10,  # Report up to 10 contacts
)

# Add sensor to robot config
g1_cfg = replace(G1_ROBOT_CFG, sensors=(self_collision_sensor,))

# Create scene
SCENE_CFG = SceneCfg(
    terrain=TerrainImporterCfg(terrain_type="plane"),
    entities={"robot": g1_cfg}
)
```

**Key changes:**
- No USD scene graph or `prim_path` management
- Materials, lights, and textures configured via MuJoCo's MjSpec. See our [`spec_config.py`](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/utils/spec_config.py) for dataclass-based modifiers that handle MjSpec changes for you.

## Complete Example Comparison

Everything else—rewards, observations, commands, terminations—works almost identically!

**Isaac Lab implementation** (Beyond Mimic):  
https://github.com/HybridRobotics/whole_body_tracking/blob/main/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py

**mjlab implementation**:  
https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/tracking/tracking_env_cfg.py

Compare these to see how similar the APIs are in practice.

## Tips for Migration

1. **Check the examples** - Look at our reference tasks in `src/mjlab/tasks/`
2. **Ask questions** - [Open a discussion](https://github.com/mujocolab/mjlab/discussions) if you get stuck
3. **MuJoCo differences** - Some Isaac Sim features (fancy rendering, USD workflows) don't have direct equivalents
