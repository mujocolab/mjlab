# Migrating from Isaac Lab

> **🚧 Work in Progress**  
> This guide is actively being improved as more users onboard and find issues. If something isn't covered here, please [open an issue](https://github.com/mujocolab/mjlab/issues) or [start a discussion](https://github.com/mujocolab/mjlab/discussions). Your questions help us improve the docs!

## TL;DR

Most Isaac Lab task configs work in mjlab with only minor tweaks! The manager-based API is nearly identical; just a few syntax changes.

## Key Differences

### 1. Import Paths

Isaac Lab:
```python
from isaaclab.envs import ManagerBasedRLEnv
```

mjlab:
```python
from mjlab.envs import ManagerBasedRlEnvCfg
```

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

## Need Help?

If something in your Isaac Lab config doesn't translate cleanly, please [open an issue](https://github.com/mujocolab/mjlab/issues) or [start a discussion](https://github.com/mujocolab/mjlab/discussions). We're actively improving migration support!
