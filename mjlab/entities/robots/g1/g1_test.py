import numpy as np
from absl.testing import absltest
import mujoco

from mjlab.entities.robots.g1 import g1_constants as consts
from mjlab.entities.robots.g1.g1 import UnitreeG1


class TestG1Robot(absltest.TestCase):
  def setUp(self):
    self.model = UnitreeG1().compile()

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
      consts.PELVIS_BODY,
      *consts.BODY_NAMES,
      *consts.BODY_NAMES_MINUS_END_EFFECTORS,
      *consts.END_EFFECTOR_NAMES,
    ]
    for body in bodies:
      body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
      self.assertGreaterEqual(body_id, 0)

    # Sites.
    sites = [
      *consts.FEET_SITES,
      consts.PELVIS_IMU_SITE,
      consts.TORSO_IMU_SITE,
      *consts.HAND_SITES,
      consts.LEFT_FOOT_SITE,
      consts.RIGHT_FOOT_SITE,
    ]
    for site in sites:
      site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
      self.assertGreaterEqual(site_id, 0)

    # Geoms.
    geoms = [*consts.FEET_GEOMS, *consts.LEFT_FEET_GEOMS, *consts.RIGHT_FEET_GEOMS]
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
    # Armature.
    leg_armature = [
      0.01017752004,
      0.025101925,
      0.01017752004,
      0.025101925,
      0.00721945,
      0.00721945,
    ]
    waist_armature = [
      0.01017752004,
      0.00721945,
      0.00721945,
    ]
    arm_armature = [
      0.003609725,
      0.003609725,
      0.003609725,
      0.003609725,
      0.003609725,
      0.00425,
      0.00425,
    ]
    expected_armature = np.array(leg_armature * 2 + waist_armature + arm_armature * 2)
    np.testing.assert_array_almost_equal(
      self.model.dof_armature[6:],  # Ignore freejoint.
      expected_armature,
    )

    # Damping and frictionloss should be 0.
    np.testing.assert_array_equal(
      self.model.dof_damping[6:],  # Ignore freejoint.
      np.zeros(consts.NU),
    )
    np.testing.assert_array_equal(
      self.model.dof_frictionloss[6:],  # Ignore freejoint.
      np.zeros(consts.NU),
    )

    # Joint ranges.
    leg_ranges = [
      (-2.5307, 2.8798),
      (-0.5236, 2.9671),
      (-2.7576, 2.7576),
      (-0.087267, 2.8798),
      (-0.87267, 0.5236),
      (-0.2618, 0.2618),
    ]
    waist_ranges = [
      (-2.618, 2.618),
      (-0.52, 0.52),
      (-0.52, 0.52),
    ]
    arm_ranges = [
      (-3.0892, 2.6704),
      (-1.5882, 2.2515),
      (-2.618, 2.618),
      (-1.0472, 2.0944),
      (-1.97222, 1.97222),
      (-1.61443, 1.61443),
      (-1.61443, 1.61443),
    ]
    leg_symmetry = [1, -1, 1, 1, 1, 1]
    arm_symmetry = [1, -1, 1, 1, 1, 1, 1]
    expected_ranges = []
    # Leg leg.
    expected_ranges.extend(leg_ranges)
    # Right leg.
    for range_list, symmetry in zip(leg_ranges, leg_symmetry):
      if symmetry == -1:
        range = (-range_list[1], -range_list[0])
      else:
        range = range_list
      expected_ranges.append(range)
    # Waist.
    expected_ranges.extend(waist_ranges)
    # Left arm.
    expected_ranges.extend(arm_ranges)
    for range_list, symmetry in zip(arm_ranges, arm_symmetry):
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
    leg_kps = [75, 75, 75, 75, 20, 20]
    waist_kps = [75, 75, 75]
    arm_kps = [75, 75, 75, 75, 20, 20, 20]
    leg_kvs = [2, 2, 2, 2, 2, 2]
    waist_kvs = [2, 2, 2]
    arm_kvs = [2, 2, 2, 2, 2, 2, 2]
    expected_kp = np.array(leg_kps * 2 + waist_kps + arm_kps * 2)
    expected_kv = np.array(leg_kvs * 2 + waist_kvs + arm_kvs * 2)
    np.testing.assert_array_equal(self.model.actuator_gainprm[:, 0], expected_kp)
    np.testing.assert_array_equal(self.model.actuator_biasprm[:, 1], -expected_kp)
    np.testing.assert_array_equal(self.model.actuator_biasprm[:, 2], -expected_kv)

    leg_forcerange = [88, 139, 88, 139, 50, 50]
    waist_forcerange = [88, 50, 50]
    arm_forcerange = [25, 25, 25, 25, 25, 5, 5]
    expected_frcrange = np.array(
      leg_forcerange * 2 + waist_forcerange + arm_forcerange * 2
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
