import tyro
import mujoco.viewer
import enum

from mjlab.entities.robots.robot import Robot
from mjlab.entities.robots.go1 import go1_constants
from mjlab.entities.robots.g1 import g1_constants
from mjlab.entities.robots.t1 import t1_constants


class RobotType(enum.Enum):
  T1 = t1_constants.T1_ROBOT_CFG
  G1 = g1_constants.G1_ROBOT_CFG
  GO1 = go1_constants.GO1_ROBOT_CFG


def load_robot(robot_type: RobotType):
  robot = Robot(robot_type.value)
  robot.spec.option.gravity[2] = 0.0
  mujoco.viewer.launch(robot.compile())


if __name__ == "__main__":
  tyro.cli(load_robot)
