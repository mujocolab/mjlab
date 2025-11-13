"""CartPole robot constants and configuration for testing."""

from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg

CARTPOLE_XML: Path = Path(__file__).parent / "xmls" / "cartpole.xml"
assert CARTPOLE_XML.exists(), f"XML not found: {CARTPOLE_XML}"


def get_spec() -> mujoco.MjSpec:
  """Get CartPole MuJoCo spec."""
  return mujoco.MjSpec.from_file(str(CARTPOLE_XML))


def get_cartpole_robot_cfg() -> EntityCfg:
  """Get a fresh CartPole robot configuration instance."""
  INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.1),  # Cart start position
  )
  return EntityCfg(spec_fn=get_spec, init_state=INIT_STATE)


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity import Entity

  robot = Entity(get_cartpole_robot_cfg())
  viewer.launch(robot.spec.compile())

