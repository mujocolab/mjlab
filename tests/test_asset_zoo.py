import mujoco
import pytest

from mjlab.asset_zoo.robots import (
  get_d1_robot_cfg,
  get_g1_robot_cfg,
  get_go1_robot_cfg,
  get_go2_d1_robot_cfg,
  get_go2_robot_cfg,
)
from mjlab.entity import Entity


@pytest.mark.parametrize(
  "robot_name,robot_cfg_fn",
  [
    ("D1", get_d1_robot_cfg),
    ("G1", get_g1_robot_cfg),
    ("GO1", get_go1_robot_cfg),
    ("GO2", get_go2_robot_cfg),
    ("GO2_D1", get_go2_d1_robot_cfg),
  ],
)
def test_robot_compiles_parametrized(robot_name: str, robot_cfg_fn) -> None:
  """Tests that all robots in the asset zoo compile without errors."""
  robot_cfg = robot_cfg_fn()
  assert isinstance(Entity(robot_cfg).compile(), mujoco.MjModel)
