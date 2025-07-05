import numpy as np
from absl.testing import absltest
import mujoco

from mjlab.entities.go1 import go1_constants as consts
from mjlab.entities.go1.go1 import UnitreeGo1
from mjlab.entities.robot_config import RobotConfig, Actuator, Joint


class TestGo1Robot(absltest.TestCase):
  def setUp(self):
    self.frictionloss = [0.1, 0.3, 1.0]
    self.armature = [0.005, 0.0025, 0.001]
    jnt_cfg = (
      Joint(
        joint_name="*hip_joint",
        frictionloss=self.frictionloss[0],
        armature=self.armature[0],
      ),
      Joint(
        joint_name="*thigh_joint",
        frictionloss=self.frictionloss[1],
        armature=self.armature[1],
      ),
      Joint(
        joint_name="*calf_joint",
        frictionloss=self.frictionloss[2],
        armature=self.armature[2],
      ),
    )
    self.kps = [35, 15, 5]
    self.kvs = [0.5, 0.25, 0.1]
    act_cfg = (
      Actuator(
        joint_name="*hip_joint",
        kp=self.kps[0],
        kv=self.kvs[0],
        torque_limit=consts.ACTUATOR_HIP_TORQUE_LIMIT,
      ),
      Actuator(
        joint_name="*thigh_joint",
        kp=self.kps[1],
        kv=self.kvs[1],
        torque_limit=consts.ACTUATOR_HIP_TORQUE_LIMIT,
      ),
      Actuator(
        joint_name="*calf_joint",
        kp=self.kps[2],
        kv=self.kvs[2],
        torque_limit=consts.ACTUATOR_KNEE_TORQUE_LIMIT,
      ),
    )
    cfg = RobotConfig(
      joints=jnt_cfg,
      actuators=act_cfg,
      sensors=consts.SENSOR_CONFIG,
      keyframes=consts.KEYFRAME_CONFIG,
    )
    go1 = UnitreeGo1.from_file(consts.GO1_XML, config=cfg, assets=consts.get_assets())
    self.model = go1.compile()

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
    bodies = [consts.ROOT_BODY]
    for body in bodies:
      body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
      self.assertGreaterEqual(body_id, 0)

    # Sites.
    sites = [*consts.FEET_SITES, consts.IMU_SITE]
    for site in sites:
      site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
      self.assertGreaterEqual(site_id, 0)

    # Geoms.
    geoms = [*consts.FEET_GEOMS]
    for geom in geoms:
      geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom)
      self.assertGreaterEqual(geom_id, 0)

  def test_joint_properties(self):
    # Check joint ranges get correctly set from default class.
    leg_joint_range = [
      (-0.863, 0.863),
      (-0.686, 4.501),
      (-2.818, -0.888),
    ]
    expected_joint_ranges = np.array(leg_joint_range * 4)
    np.testing.assert_array_almost_equal(
      self.model.jnt_range[1:],
      expected_joint_ranges,
    )

    # Armature.
    expected_armature = np.array(self.armature * 4)
    np.testing.assert_array_almost_equal(
      self.model.dof_armature[6:],  # Ignore freejoint.
      expected_armature,
    )

    # Damping should be 0.0.
    np.testing.assert_array_equal(
      self.model.dof_damping[6:],  # Ignore freejoint.
      np.zeros(consts.NU),
    )

    # Frictionloss.
    expected_frictionloss = np.array(self.frictionloss * 4)
    np.testing.assert_array_equal(
      self.model.dof_frictionloss[6:],  # Ignore freejoint.
      expected_frictionloss,
    )

  def test_sensor_properties(self):
    expected_nsensor = len(consts.SENSOR_CONFIG)
    self.assertEqual(self.model.nsensor, expected_nsensor)
    expected_nsensordata = 3 + 3 + 3
    self.assertEqual(self.model.nsensordata, expected_nsensordata)

  def test_actuator_properties(self):
    expected_kp = np.array(self.kps * 4)
    expected_kv = np.array(self.kvs * 4)
    np.testing.assert_array_equal(self.model.actuator_gainprm[:, 0], expected_kp)
    np.testing.assert_array_equal(self.model.actuator_biasprm[:, 1], -expected_kp)
    np.testing.assert_array_equal(self.model.actuator_biasprm[:, 2], -expected_kv)

    forcerange = [
      consts.ACTUATOR_HIP_TORQUE_LIMIT,
      consts.ACTUATOR_HIP_TORQUE_LIMIT,
      consts.ACTUATOR_KNEE_TORQUE_LIMIT,
    ]
    expected_frcrange = np.array(forcerange * 4)
    self.assertTrue(np.all(self.model.actuator_forcelimited))
    np.testing.assert_array_equal(
      self.model.actuator_forcerange[:, 0], -expected_frcrange
    )
    np.testing.assert_array_equal(
      self.model.actuator_forcerange[:, 1], expected_frcrange
    )

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
