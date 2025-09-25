import mujoco
import pytest

from mjlab.asset_zoo import robots
from mjlab.entity import Entity, EntityCfg


@pytest.mark.parametrize(
  "robot_name,robot_cfg",
  [
    ("G1", robots.G1_ROBOT_CFG),
    ("GO1", robots.GO1_ROBOT_CFG),
  ],
)
def test_robot_compiles_parametrized(robot_name: str, robot_cfg: EntityCfg) -> None:
  """Tests that all robots in the asset zoo compile without errors."""
  assert isinstance(Entity(robot_cfg).compile(), mujoco.MjModel)
