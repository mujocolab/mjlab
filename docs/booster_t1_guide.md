# Booster T1 Integration Guide

This guide explains how to use the Booster T1 humanoid robot in MjLAB for RL training.

## Overview

The Booster T1 is a 23 DOF humanoid robot from Booster Robotics. It has been successfully integrated into MjLAB following the same pattern as the Unitree G1 and Go1 robots.

## Robot Configuration

The T1 robot configuration is located at:
- `src/mjlab/asset_zoo/robots/booster_t1/t1_constants.py`
- MJCF model: `src/mjlab/asset_zoo/robots/booster_t1/xmls/t1_23dof.xml`

### Key Features
- **23 Degrees of Freedom:**
  - Lower body: 6 DOF per leg (Hip Pitch/Roll/Yaw, Knee Pitch, Ankle Pitch/Roll)
  - Waist: 1 DOF
  - Arms: 4 DOF per arm (Shoulder Pitch/Roll, Elbow Pitch/Yaw)
  - Head: 2 DOF (AAHead_yaw, Head_pitch)

- **Actuators:** Uses XML-defined position actuators (XmlPositionActuatorCfg)
- **Action Scale:** 0.25 for all joints (configured in `T1_ACTION_SCALE`)

## Available Tasks

Two velocity tracking tasks are available for the T1:

1. **Flat Terrain:** `Mjlab-Velocity-Flat-Booster-T1`
2. **Rough Terrain:** `Mjlab-Velocity-Rough-Booster-T1`

## Training

### Start Training on Flat Terrain

```bash
uv run python src/mjlab/scripts/train.py Mjlab-Velocity-Flat-Booster-T1 --env.num_envs 4096
```

### Start Training on Rough Terrain

```bash
uv run python src/mjlab/scripts/train.py Mjlab-Velocity-Rough-Booster-T1 --env.num_envs 4096
```

### Multi-GPU Training

```bash
uv run python src/mjlab/scripts/train.py Mjlab-Velocity-Flat-Booster-T1 \
  --env.num_envs 4096 \
  --gpu-ids all
```

## Playing with Trained Policy

After training, you can visualize the policy:

```bash
uv run python src/mjlab/scripts/play.py Mjlab-Velocity-Flat-Booster-T1 \
  --checkpoint path/to/checkpoint.pt \
  --num-envs 1
```

## Viewing the Robot

To view the T1 robot model in MuJoCo:

```bash
uv run python src/mjlab/asset_zoo/robots/booster_t1/t1_constants.py
```

## Configuration Details

### Contact Sensors
- **Feet ground contact:** Detects contact between ankle subtrees and terrain
- **Self collision:** Monitors collisions within the trunk subtree

### Observations
The policy observes:
- Base linear velocity (IMU)
- Base angular velocity (IMU)
- Projected gravity
- Joint positions (relative)
- Joint velocities (relative)
- Last action
- Velocity command

The critic additionally observes:
- Foot air time
- Foot contact state
- Foot contact forces

### Rewards
Key reward terms include:
- Tracking linear/angular velocity
- Maintaining upright posture
- Joint pose regularization (standing/walking/running)
- Penalizing high body angular velocity
- Penalizing angular momentum
- Limiting joint positions
- Penalizing action rate changes
- Soft landing
- Self-collision penalty

## Differences from Holosoma

The holosoma repository uses a different framework structure:
- **Holosoma:** Uses RobotConfig with SDK integration for real hardware deployment
- **MjLAB:** Uses EntityCfg with pure MuJoCo simulation

Key adaptations:
1. Converted RobotConfig to EntityCfg format
2. Used XmlPositionActuatorCfg to wrap existing XML actuators (avoiding duplication)
3. Configured contact sensors using MjLAB's ContactSensorCfg
4. Adapted observations and rewards to MjLAB's MDP system
5. Removed foot sites (not present in T1 XML) and related observations

## Fine-tuning

You may want to adjust these parameters based on your needs:

1. **Action scale** (`t1_constants.py`): Currently 0.25 for all joints
2. **Reward weights** (`env_cfgs.py`): Tune tracking vs regularization balance
3. **Pose standards** (`env_cfgs.py`): Adjust standing/walking/running pose tolerances
4. **RL hyperparameters** (`rl_cfg.py`): Learning rate, network architecture, etc.

## Next Steps

1. **Tune actuator parameters:** The XML actuators use default MuJoCo settings. You may want to adjust stiffness, damping, and force limits based on real T1 motor specs.
2. **Add foot sites:** For better foot tracking, add site elements to the XML at foot locations and enable foot_height, foot_clearance, foot_swing_height observations.
3. **Create custom tasks:** Follow `docs/create_new_task.md` to create specialized tasks.
4. **Deploy to real robot:** Use the holosoma SDK for sim-to-real transfer.

## Files Created

```
src/mjlab/asset_zoo/robots/booster_t1/
├── __init__.py
├── t1_constants.py
└── xmls/
    ├── t1_23dof.xml
    ├── t1_23dof.urdf
    └── assets/...

src/mjlab/tasks/velocity/config/t1/
├── __init__.py
├── env_cfgs.py
└── rl_cfg.py
```

## Troubleshooting

### Issue: "repeated name 'AAHead_yaw' in actuator"
**Solution:** This was fixed by using `XmlPositionActuatorCfg` instead of `BuiltinPositionActuatorCfg`. The T1 XML already defines actuators.

### Issue: Body/geom names not found
**Solution:** Use `grep` to extract names from the XML:
```bash
grep -o '<body name="[^"]*"' src/mjlab/asset_zoo/robots/booster_t1/xmls/t1_23dof.xml
grep -o '<geom name="[^"]*"' src/mjlab/asset_zoo/robots/booster_t1/xmls/t1_23dof.xml
```

### Issue: Missing observations
**Solution:** Some observations (like foot_height) require sites in the XML. Either add them or remove those observation terms.
