from absl.testing import parameterized, absltest

import mujoco
from mjlab.asset_zoo import robots
from mjlab.entities import RobotCfg, Robot


@parameterized.parameters(
  ("T1", robots.T1_ROBOT_CFG),
  ("G1", robots.G1_ROBOT_CFG),
  ("GO1", robots.GO1_ROBOT_CFG),
)
class RobotTest(parameterized.TestCase):
  def test_compiles(self, robot_name: str, robot_cfg: RobotCfg) -> None:
    """Tests that all robots in the asset zoo compile without errors."""
    with self.subTest(robot=robot_name):
      self.assertIsInstance(Robot(robot_cfg).compile(), mujoco.MjModel)


if __name__ == "__main__":
  absltest.main()
