import numpy as np
from absl.testing import absltest
import mujoco

from mjlab.entities.go1 import go1_constants as consts
from mjlab.entities.go1.go1 import UnitreeGo1


class TestGo1Robot(absltest.TestCase):
  def setUp(self):
    self.assets = consts.get_assets()
    self.go1 = UnitreeGo1.from_default_config()

  def test_can_compile_and_step(self):
    model = self.go1.compile()
    data = mujoco.MjData(model)
    for _ in range(5):
      mujoco.mj_step(model, data)

  def test_joint_properties(self):
    model = self.go1.compile()

    # Check joint ranges get correctly set from default class.
    leg_joint_range = [
      (-0.863, 0.863),
      (-0.686, 4.501),
      (-2.818, -0.888),
    ]
    expected_joint_ranges = np.array(leg_joint_range * 4)
    np.testing.assert_array_almost_equal(
      model.jnt_range[1:],
      expected_joint_ranges,
    )

    # Armature.
    leg_actuator_armature = [
      consts.ACTUATOR_HIP_ARMATURE,
      consts.ACTUATOR_HIP_ARMATURE,
      consts.ACTUATOR_KNEE_ARMATURE,
    ]
    expected_armature = np.array(leg_actuator_armature * 4)
    np.testing.assert_array_almost_equal(
      model.dof_armature[6:],  # Ignore freejoint.
      expected_armature,
    )

    # Damping and frictionloss should be 0.0
    np.testing.assert_array_equal(
      model.dof_damping[6:],  # Ignore freejoint.
      np.zeros(12),
    )
    np.testing.assert_array_equal(
      model.dof_frictionloss[6:],  # Ignore freejoint.
      np.zeros(12),
    )


if __name__ == "__main__":
  absltest.main()
