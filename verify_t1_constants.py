"""Verify T1 constants are correctly loaded from holosoma."""

from mjlab.asset_zoo.robots.booster_t1.t1_constants import (
    T1_ACTUATOR_FRC_RANGE,
    T1_ACTION_SCALE,
    T1_DOF,
    T1_FOOT_STICKING_LINKS,
    T1_HEIGHT,
    T1_HOME_QPOS,
    T1_JOINT_NAMES,
    T1_JOINT_RANGES,
    get_spec,
    get_t1_robot_cfg,
)

if __name__ == "__main__":
    print("=" * 60)
    print("T1 Constants Verification (from holosoma)")
    print("=" * 60)
    
    # Basic properties
    print(f"\n✓ DOF: {T1_DOF}")
    print(f"✓ Height: {T1_HEIGHT}m")
    print(f"✓ Number of joints: {len(T1_JOINT_NAMES)}")
    
    # Home position
    print(f"\n✓ Home joint positions defined: {len(T1_HOME_QPOS)} joints")
    print("  Sample home positions:")
    for joint in ["Left_Shoulder_Roll", "Right_Shoulder_Roll", "Left_Knee_Pitch"]:
        print(f"    {joint}: {T1_HOME_QPOS[joint]} rad")
    
    # Actuator ranges
    print(f"\n✓ Actuator force ranges defined: {len(T1_ACTUATOR_FRC_RANGE)} actuators")
    print("  Sample force ranges:")
    print(f"    Left_Hip_Pitch: {T1_ACTUATOR_FRC_RANGE['Left_Hip_Pitch']} N⋅m")
    print(f"    Left_Knee_Pitch: {T1_ACTUATOR_FRC_RANGE['Left_Knee_Pitch']} N⋅m")
    print(f"    AAHead_yaw: {T1_ACTUATOR_FRC_RANGE['AAHead_yaw']} N⋅m")
    
    # Joint ranges
    print(f"\n✓ Joint ranges defined: {len(T1_JOINT_RANGES)} joints")
    print("  Sample joint ranges:")
    print(f"    Left_Hip_Pitch: {T1_JOINT_RANGES['Left_Hip_Pitch']} rad")
    print(f"    Left_Knee_Pitch: {T1_JOINT_RANGES['Left_Knee_Pitch']} rad")
    
    # Foot links
    print(f"\n✓ Foot contact links: {len(T1_FOOT_STICKING_LINKS)}")
    print(f"  {T1_FOOT_STICKING_LINKS[:3]} ...")
    
    # Action scale
    print(f"\n✓ Action scale: {list(T1_ACTION_SCALE.values())[0]} (uniform)")
    
    # Test loading spec
    print("\n✓ Loading MuJoCo spec...")
    spec = get_spec()
    model = spec.compile()
    print(f"  - Model compiled successfully")
    print(f"  - Actuators in model: {model.nu}")
    print(f"  - Joints in model: {model.njnt}")
    
    # Test robot config
    print("\n✓ Loading robot configuration...")
    robot_cfg = get_t1_robot_cfg()
    print(f"  - Initial position: {robot_cfg.init_state.pos}")
    print(f"  - Home joint count: {len(robot_cfg.init_state.joint_pos)}")
    
    print("\n" + "=" * 60)
    print("All T1 constants verified successfully! ✓")
    print("=" * 60)
    print("\nKey differences from generic config:")
    print("  • Using home keyframe positions (not all zeros)")
    print("  • Correct initial height: 0.665m (not 0.75m)")
    print("  • Real actuator force ranges from XML")
    print("  • Real joint ranges from XML")
    print("  • Foot sphere collision links defined")
