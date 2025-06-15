from typing import Union
from pathlib import Path

# Import tasks to register them.
from mjlab import control_suite

# Import configs and add them to ConfigUnion.
from mjlab.control_suite.cartpole import CartpoleConfig
from mjlab.control_suite.go1 import Go1Config
from mjlab.control_suite.humanoid import (
  HumanoidStandConfig,
  HumanoidWalkConfig,
  HumanoidRunConfig,
)
from mjlab.control_suite.g1 import G1Config

TaskConfigUnion = Union[
  CartpoleConfig,
  Go1Config,
  HumanoidStandConfig,
  HumanoidWalkConfig,
  HumanoidRunConfig,
  G1Config,
]

_HERE = Path(__file__).parent
MJLAB_ROOT_PATH = _HERE.parent


__all__ = (
  "control_suite",
  "TaskConfigUnion",
  "MJLAB_ROOT_PATH",
)
