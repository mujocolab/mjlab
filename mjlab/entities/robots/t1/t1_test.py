import numpy as np
from absl.testing import absltest
import mujoco

from mjlab.entities.robots.t1 import t1_constants as consts
from mjlab.entities.robots.t1.t1 import BoosterT1


class TestT1Robot(absltest.TestCase):
  def setUp(self):
    self.model = BoosterT1().compile()

  def test_can_compile_and_step(self):
    data = mujoco.MjData(self.model)
    for _ in range(5):
      mujoco.mj_step(self.model, data)

  def test_constants(self):
    self.assertEqual(self.model.nq, consts.NQ)
    self.assertEqual(self.model.nu, consts.NU)
    self.assertEqual(self.model.nv, consts.NV)

    # Check options are the default. The idea here is that these parameters should be
    # specified or set during task creation.
    self.assertEqual(self.model.opt.timestep, 0.002)
    self.assertEqual(self.model.opt.integrator, mujoco.mjtIntegrator.mjINT_EULER)

    # Bodies.
    bodies = [
      consts.ROOT_BODY,
      consts.TORSO_BODY,
    ]
    for body in bodies:
      body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
      self.assertGreaterEqual(body_id, 0)

    # Sites.
    sites = [
      *consts.FEET_SITES,
      *consts.HAND_SITES,
      consts.LEFT_FOOT_SITE,
      consts.RIGHT_FOOT_SITE,
      consts.IMU_SITE,
    ]
    for site in sites:
      site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
      self.assertGreaterEqual(site_id, 0)

    # Geoms.
    geoms = [*consts.FEET_GEOMS]
    for geom in geoms:
      geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom)
      self.assertGreaterEqual(geom_id, 0)

  def test_sensor_properties(self):
    expected_nsensor = len(consts.SENSOR_CONFIG)
    self.assertEqual(self.model.nsensor, expected_nsensor)
    expected_nsensordata = 3 + 3 + 3
    self.assertEqual(self.model.nsensordata, expected_nsensordata)

  def test_actuator_order_matches_joint_order(self):
    actuator_names = [self.model.actuator(i).name for i in range(self.model.nu)]
    joint_names = [self.model.joint(i).name for i in range(1, self.model.njnt)]
    self.assertEqual(actuator_names, joint_names)

  def test_keyframe_properties(self):
    expected_nkey = len(consts.KEYFRAME_CONFIG)
    self.assertEqual(self.model.nkey, expected_nkey)
    for key in consts.KEYFRAME_CONFIG:
      expected_qpos = key.qpos
      np.testing.assert_array_equal(self.model.key(key.name).qpos, expected_qpos)

  def test_joint_properties(self):
    # Damping, frictionloss and armature should be 0.
    np.testing.assert_array_equal(
      self.model.dof_damping[6:],  # Ignore freejoint.
      np.zeros(consts.NU),
    )
    np.testing.assert_array_equal(
      self.model.dof_frictionloss[6:],  # Ignore freejoint.
      np.zeros(consts.NU),
    )
    np.testing.assert_array_equal(
      self.model.dof_armature[6:],  # Ignore freejoint.
      np.zeros(consts.NU),
    )

    # Joint ranges.
    head_ranges = [
      (-1.57, 1.57),
      (-0.35, 1.22),
    ]
    arm_ranges = [
      (-3.31, 1.22),
      (-1.74, 1.57),
      (-2.27, 2.27),
      (-2.44, 0),
    ]
    waist_ranges = [
      (-1.57, 1.57),
    ]
    leg_ranges = [
      (-1.8, 1.57),
      (-0.2, 1.57),
      (-1, 1),
      (0, 2.34),
      (-0.87, 0.35),
      (-0.44, 0.44),
    ]
    arm_symmetry = [1, -1, 1, -1]
    leg_symmetry = [1, -1, 1, 1, 1, 1]
    expected_ranges = []
    # Head.
    expected_ranges.extend(head_ranges)
    # Left arm.
    expected_ranges.extend(arm_ranges)
    # Right arm.
    for range_list, symmetry in zip(arm_ranges, arm_symmetry):
      if symmetry == -1:
        range = (-range_list[1], -range_list[0])
      else:
        range = range_list
      expected_ranges.append(range)
    # Waist.
    expected_ranges.extend(waist_ranges)
    # Left leg.
    expected_ranges.extend(leg_ranges)
    # Right leg.
    for range_list, symmetry in zip(leg_ranges, leg_symmetry):
      if symmetry == -1:
        range = (-range_list[1], -range_list[0])
      else:
        range = range_list
      expected_ranges.append(range)
    expected_joint_ranges = np.array(expected_ranges)
    np.testing.assert_array_almost_equal(
      self.model.jnt_range[1:],
      expected_joint_ranges,
    )

  def test_actuator_properties(self):
    head_kps = [20, 20]
    arm_kps = [20, 20, 20, 20]
    waist_kps = [50]
    leg_kps = [50, 50, 50, 50, 20, 20]
    expected_kp = np.array(head_kps + arm_kps * 2 + waist_kps + leg_kps * 2)
    head_kvs = [5, 5]
    arm_kvs = [2, 2, 2, 2]
    waist_kvs = [5]
    leg_kvs = [5, 5, 5, 5, 2, 2]
    expected_kv = np.array(head_kvs + arm_kvs * 2 + waist_kvs + leg_kvs * 2)
    np.testing.assert_array_equal(self.model.actuator_gainprm[:, 0], expected_kp)
    np.testing.assert_array_equal(self.model.actuator_biasprm[:, 1], -expected_kp)
    np.testing.assert_array_equal(self.model.actuator_biasprm[:, 2], -expected_kv)

    head_forcerange = [7, 7]
    arm_forcerange = [18, 18, 18, 18]
    waist_forcerange = [30]
    leg_forcerange = [45, 30, 30, 60, 20, 15]
    expected_frcrange = np.array(
      head_forcerange + arm_forcerange * 2 + waist_forcerange + leg_forcerange * 2
    )
    self.assertTrue(np.all(self.model.actuator_forcelimited))
    np.testing.assert_array_equal(
      self.model.actuator_forcerange[:, 0], -expected_frcrange
    )
    np.testing.assert_array_equal(
      self.model.actuator_forcerange[:, 1], expected_frcrange
    )


if __name__ == "__main__":
  absltest.main()
