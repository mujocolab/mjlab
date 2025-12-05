#!/usr/bin/env python3
"""
Complete end-to-end test of T1 integration with holosoma constants.
This verifies that the robot can be loaded, configured, and is ready for training.
"""

import sys


def test_constants_import():
    """Test that all constants can be imported."""
    print("Testing constant imports...")
    try:
        from mjlab.asset_zoo.robots import (
            T1_ACTION_SCALE,
            T1_ACTUATOR_FRC_RANGE,
            T1_FOOT_STICKING_LINKS,
            T1_HOME_QPOS,
            T1_JOINT_RANGES,
            get_t1_robot_cfg,
        )
        assert len(T1_ACTION_SCALE) == 23
        assert len(T1_ACTUATOR_FRC_RANGE) == 23
        assert len(T1_JOINT_RANGES) == 23
        assert len(T1_HOME_QPOS) == 23
        assert len(T1_FOOT_STICKING_LINKS) == 10
        print("  ✓ All constants imported and validated")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_robot_config():
    """Test that robot configuration can be created."""
    print("\nTesting robot configuration...")
    try:
        from mjlab.asset_zoo.robots import get_t1_robot_cfg
        
        cfg = get_t1_robot_cfg()
        assert cfg.init_state.pos == (0.0, 0.0, 0.665), "Incorrect initial height"
        assert len(cfg.init_state.joint_pos) == 23, "Incorrect number of home positions"
        assert cfg.init_state.joint_pos["Left_Shoulder_Roll"] == -1.4
        assert cfg.init_state.joint_pos["Left_Knee_Pitch"] == 0.4
        print("  ✓ Robot configuration validated")
        return True
    except Exception as e:
        print(f"  ✗ Configuration failed: {e}")
        return False


def test_spec_compilation():
    """Test that MuJoCo spec compiles."""
    print("\nTesting MuJoCo spec compilation...")
    try:
        from mjlab.asset_zoo.robots.booster_t1.t1_constants import get_spec
        
        spec = get_spec()
        model = spec.compile()
        assert model.nu == 23, f"Expected 23 actuators, got {model.nu}"
        print(f"  ✓ Spec compiled successfully ({model.nu} actuators)")
        return True
    except Exception as e:
        print(f"  ✗ Compilation failed: {e}")
        return False


def test_task_registration():
    """Test that T1 tasks are registered."""
    print("\nTesting task registration...")
    try:
        import mjlab.tasks  # noqa: F401
        from mjlab.tasks.registry import list_tasks
        
        tasks = list_tasks()
        t1_tasks = [t for t in tasks if "T1" in t]
        
        expected_tasks = [
            "Mjlab-Velocity-Flat-Booster-T1",
            "Mjlab-Velocity-Rough-Booster-T1",
        ]
        
        for expected in expected_tasks:
            assert expected in t1_tasks, f"Task {expected} not registered"
        
        print(f"  ✓ {len(t1_tasks)} T1 tasks registered:")
        for task in t1_tasks:
            print(f"    - {task}")
        return True
    except Exception as e:
        print(f"  ✗ Task registration failed: {e}")
        return False


def test_task_config():
    """Test that task configuration can be created."""
    print("\nTesting task configuration...")
    try:
        from mjlab.tasks.velocity.config.t1 import booster_t1_flat_env_cfg
        
        cfg = booster_t1_flat_env_cfg()
        assert "robot" in cfg.scene.entities
        assert cfg.scene.sensors is not None
        assert len(cfg.scene.sensors) == 2  # feet_ground + self_collision
        print("  ✓ Task configuration validated")
        print(f"    - Sensors: {[s.name for s in cfg.scene.sensors]}")
        print(f"    - Observations: {list(cfg.observations.keys())}")
        print(f"    - Rewards: {len(cfg.rewards)} terms")
        return True
    except Exception as e:
        print(f"  ✗ Task config failed: {e}")
        return False


def test_holosoma_values():
    """Test that specific holosoma values are correctly set."""
    print("\nTesting holosoma-specific values...")
    try:
        from mjlab.asset_zoo.robots import (
            T1_ACTUATOR_FRC_RANGE,
            T1_HOME_QPOS,
            T1_JOINT_RANGES,
        )
        
        # Check key values from holosoma
        tests = [
            (T1_HOME_QPOS["Left_Shoulder_Roll"], -1.4, "Home shoulder roll"),
            (T1_HOME_QPOS["Left_Knee_Pitch"], 0.4, "Home knee pitch"),
            (T1_ACTUATOR_FRC_RANGE["Left_Knee_Pitch"], (-60, 60), "Knee force range"),
            (T1_ACTUATOR_FRC_RANGE["AAHead_yaw"], (-7, 7), "Head force range"),
            (T1_JOINT_RANGES["Left_Knee_Pitch"], (0.0, 2.34), "Knee joint range"),
        ]
        
        for actual, expected, name in tests:
            assert actual == expected, f"{name}: expected {expected}, got {actual}"
            print(f"  ✓ {name}: {actual}")
        
        return True
    except Exception as e:
        print(f"  ✗ Holosoma values check failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("T1 Holosoma Integration - End-to-End Test")
    print("=" * 70)
    
    tests = [
        ("Constants Import", test_constants_import),
        ("Robot Configuration", test_robot_config),
        ("Spec Compilation", test_spec_compilation),
        ("Task Registration", test_task_registration),
        ("Task Configuration", test_task_config),
        ("Holosoma Values", test_holosoma_values),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 All tests passed! T1 integration is complete and ready.")
        print("\nNext steps:")
        print("  1. Train: uv run python src/mjlab/scripts/train.py Mjlab-Velocity-Flat-Booster-T1")
        print("  2. Read: docs/t1_holosoma_integration.md")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
