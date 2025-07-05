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


if __name__ == "__main__":
  absltest.main()
