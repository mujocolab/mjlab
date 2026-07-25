import mujoco
import pytest

from mjlab.asset_zoo.robots import get_go2_d1_robot_cfg, get_go2_robot_cfg
from mjlab.scripts.visualize_terrain import _apply_initial_joint_positions


@pytest.mark.parametrize(
  "robot_cfg_fn",
  [
    get_go2_robot_cfg,
    get_go2_d1_robot_cfg,
  ],
)
def test_visualizer_initial_joint_positions_skip_unnamed_joints(robot_cfg_fn) -> None:
  robot_cfg = robot_cfg_fn()
  robot_model = robot_cfg.spec_fn().compile()
  robot_data = mujoco.MjData(robot_model)

  assert any(
    mujoco.mj_id2name(robot_model, mujoco.mjtObj.mjOBJ_JOINT, i) is None
    for i in range(robot_model.njnt)
  )

  _apply_initial_joint_positions(
    robot_model, robot_data, robot_cfg.init_state.joint_pos
  )
