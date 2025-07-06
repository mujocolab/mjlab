from absl.testing import absltest
import mujoco

from mjlab.entities.robots.mujoco_humanoid import mujoco_humanoid_constants as consts
from mjlab.entities.robots.mujoco_humanoid.mujoco_humanoid import MujocoHumanoid


class TestMujocoHumanoid(absltest.TestCase):
  def setUp(self):
    self.model = MujocoHumanoid().compile()

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
    bodies = [consts.TORSO_BODY, consts.HEAD_BODY]
    for body in bodies:
      body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
      self.assertGreaterEqual(body_id, 0)


if __name__ == "__main__":
  absltest.main()
