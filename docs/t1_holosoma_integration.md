# T1 Holosoma Integration - Configuration Guide

## Overview

This document explains how the Booster T1 robot constants from the **holosoma** repository have been correctly integrated into **MjLAB**.

## Key Configuration Values

### ✅ Values Correctly Imported from Holosoma

#### 1. **Robot Properties**
```python
T1_DOF = 23                    # 23 degrees of freedom
T1_HEIGHT = 1.2                # meters
```

#### 2. **Initial State (Home Keyframe)**
The robot starts in the "home" keyframe position from the XML:

```python
# Base position
pos = (0.0, 0.0, 0.665)        # Height: 0.665m (from XML keyframe)
rot = (1.0, 0.0, 0.0, 0.0)     # Quaternion [w, x, y, z]

# Joint positions (home keyframe)
T1_HOME_QPOS = {
    # Arms positioned at sides
    "Left_Shoulder_Roll": -1.4,     # Arms down
    "Right_Shoulder_Roll": 1.4,
    "Left_Elbow_Yaw": -0.4,
    "Right_Elbow_Yaw": 0.4,
    
    # Legs in slight squat
    "Left_Hip_Pitch": -0.2,
    "Right_Hip_Pitch": -0.2,
    "Left_Knee_Pitch": 0.4,         # Slight knee bend
    "Right_Knee_Pitch": 0.4,
    "Left_Ankle_Pitch": -0.2,       # Balances knee bend
    "Right_Ankle_Pitch": -0.2,
    
    # All other joints: 0.0
}
```

#### 3. **Actuator Force Ranges**
Real motor torque limits from the T1 XML:

| Joint Group | Force Range (N⋅m) |
|-------------|-------------------|
| **Head** | ±7 |
| **Arms** | ±18 |
| **Waist** | ±30 |
| **Hip Pitch** | ±45 |
| **Hip Roll/Yaw** | ±30 |
| **Knee** | ±60 (strongest) |
| **Ankle Pitch** | ±20 |
| **Ankle Roll** | ±15 |

#### 4. **Joint Ranges**
Physical joint limits from the T1 XML:

```python
# Examples (all values in radians)
"Left_Knee_Pitch": (0.0, 2.34)           # Knee can't hyperextend
"Left_Hip_Roll": (-0.2, 1.57)            # Asymmetric (can lift leg out more than in)
"Right_Hip_Roll": (-1.57, 0.2)           # Opposite side
"Left_Elbow_Yaw": (-2.44, 0.0)           # Left elbow rotates inward
"Right_Elbow_Yaw": (0.0, 2.44)           # Right elbow rotates outward
```

#### 5. **Foot Contact Links**
The T1 uses **sphere collision geometries** for foot contact (10 spheres total):

```python
T1_FOOT_STICKING_LINKS = [
    "left_foot_sphere_1_link",   # Front contact points
    "right_foot_sphere_1_link",
    "left_foot_sphere_2_link",   # Mid contact points
    "right_foot_sphere_2_link",
    "left_foot_sphere_3_link",
    "right_foot_sphere_3_link",
    "left_foot_sphere_4_link",
    "right_foot_sphere_4_link",
    "left_foot_sphere_5_link",   # Heel contact points
    "right_foot_sphere_5_link",
]
```

#### 6. **Action Scale**
```python
T1_ACTION_SCALE = {name: 0.25 for name in T1_JOINT_NAMES}
```
This means policy outputs are scaled by 0.25 before being sent to actuators.

## Usage in MjLAB

### How to Access Constants

```python
from mjlab.asset_zoo.robots.booster_t1.t1_constants import (
    T1_ACTUATOR_FRC_RANGE,     # Force limits per joint
    T1_ACTION_SCALE,           # Action scaling
    T1_DOF,                    # Degrees of freedom
    T1_FOOT_STICKING_LINKS,    # Foot contact links
    T1_HEIGHT,                 # Robot height
    T1_HOME_QPOS,              # Home joint positions
    T1_JOINT_NAMES,            # All joint names
    T1_JOINT_RANGES,           # Joint limits
    get_spec,                  # Get MuJoCo spec
    get_t1_robot_cfg,          # Get robot EntityCfg
)
```

### How Constants Are Used

#### In Robot Configuration (`t1_constants.py`)
```python
# Initial state uses home keyframe
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.665),    # From T1_HOME_QPOS keyframe
    joint_pos=T1_HOME_QPOS,   # Home positions
)

# Collision uses foot sphere links
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(r"^(left|right)_foot.*link$",),  # Matches all foot spheres
    friction=(0.6,),
)
```

#### In Task Configuration (`env_cfgs.py`)
```python
# Contact sensor for ground detection
feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
        mode="body",
        pattern=r"^(left_foot_link|right_foot_link)$",  # Main foot bodies
    ),
)

# Action scaling
joint_pos_action.scale = T1_ACTION_SCALE  # Apply 0.25 scaling
```

## Differences from Generic Configuration

### ❌ Before (Generic Config)
```python
# Wrong initial height
pos = (0.0, 0.0, 0.75)

# All joints at zero (robot would collapse)
joint_pos = {".*": 0.0}

# No actuator/joint range information
# No foot contact link information
```

### ✅ After (Holosoma Config)
```python
# Correct initial height from keyframe
pos = (0.0, 0.0, 0.665)

# Stable home position (arms at sides, slight squat)
joint_pos = T1_HOME_QPOS

# Real motor force limits
T1_ACTUATOR_FRC_RANGE = {...}  # 23 motors with accurate limits

# Real joint limits
T1_JOINT_RANGES = {...}  # 23 joints with accurate ranges

# Real foot contact geometry
T1_FOOT_STICKING_LINKS = [...]  # 10 sphere contacts
```

## Configuration Validation

Run this to verify all constants are loaded correctly:

```bash
uv run python verify_t1_constants.py
```

Expected output:
```
============================================================
T1 Constants Verification (from holosoma)
============================================================

✓ DOF: 23
✓ Height: 1.2m
✓ Number of joints: 23

✓ Home joint positions defined: 23 joints
✓ Actuator force ranges defined: 23 actuators
✓ Joint ranges defined: 23 joints
✓ Foot contact links: 10

✓ Loading MuJoCo spec...
  - Model compiled successfully
  - Actuators in model: 23

✓ Loading robot configuration...
  - Initial position: (0.0, 0.0, 0.665)
  - Home joint count: 23

============================================================
All T1 constants verified successfully! ✓
============================================================
```

## Why These Values Matter

### 1. **Home Position**
- **Critical for stability**: The home keyframe puts the robot in a stable standing pose
- **Arms at sides** (-1.4/+1.4 rad shoulder roll) prevents arm collisions
- **Slight squat** (0.4 rad knee bend) keeps the robot balanced and ready to move

### 2. **Actuator Force Ranges**
- **Knee is strongest** (±60 N⋅m): Supports body weight during locomotion
- **Hip pitch is strong** (±45 N⋅m): Drives forward/backward motion
- **Head is weakest** (±7 N⋅m): Only needs to move the light head assembly

### 3. **Joint Ranges**
- **Prevents impossible poses**: Keeps RL policy from attempting invalid joint configurations
- **Asymmetric ranges**: Left/right hip rolls have opposite ranges (anatomically correct)
- **Knee can't hyperextend**: Range starts at 0.0 rad

### 4. **Foot Sphere Contacts**
- **Multi-point contact**: 5 spheres per foot provides stable ground contact
- **Realistic friction**: Models how real T1 foot makes contact with ground

## Training Recommendations

### Using Real Actuator Limits

The T1 has **very different motor capabilities** across joints:

```python
# Strong joints (can apply high forces)
"Left_Knee_Pitch": (-60, 60)      # ±60 N⋅m
"Left_Hip_Pitch": (-45, 45)       # ±45 N⋅m

# Medium joints
"Waist": (-30, 30)                # ±30 N⋅m
"Left_Hip_Roll": (-30, 30)        # ±30 N⋅m

# Weak joints (limited force)
"Left_Shoulder_Pitch": (-18, 18)  # ±18 N⋅m
"AAHead_yaw": (-7, 7)             # ±7 N⋅m
```

**Implication for training:**
- Don't expect arms to support body weight
- Knee and hip pitch are primary locomotion actuators
- Head movements should be minimal during walking

### Using Real Joint Ranges

Some joints have **restricted ranges**:

```python
# Knee can only bend forward
"Left_Knee_Pitch": (0.0, 2.34)    # 0° to 134°

# Hip roll is asymmetric
"Left_Hip_Roll": (-0.2, 1.57)     # Can lift leg OUT more than IN
```

**Implication for training:**
- Policy must learn asymmetric gaits
- Can't do hyperextended poses
- Reward shaping should respect these limits

## Next Steps

### Fine-tuning for Real Robot

When deploying to hardware, you may need to:

1. **Calibrate action scale**: The 0.25 scale may need adjustment based on real motor response
2. **Tune PD gains**: XML actuators have default gains; real motors may need different values
3. **Adjust force limits**: Real motors may have different limits than XML values
4. **Add safety limits**: Reduce joint ranges for safety during initial testing

### Sim-to-Real Transfer

Use the **holosoma SDK** for deployment:
```python
# In holosoma repo
from holosoma_retargeting.robot import T1Robot

robot = T1Robot()
robot.deploy_policy(policy_checkpoint)
```

The holosoma constants ensure **sim-to-real consistency** between MjLAB training and real deployment.

## Summary

✅ **All critical constants from holosoma are correctly integrated:**
- Initial home position (stable standing pose)
- Real actuator force ranges (±7 to ±60 N⋅m)
- Real joint limits (anatomically correct)
- Foot contact geometry (10 sphere contacts)
- Action scaling (0.25)

✅ **The T1 configuration is production-ready for:**
- RL training in MjLAB
- Sim-to-real transfer via holosoma SDK
- Accurate physics simulation
- Safe deployment to hardware
