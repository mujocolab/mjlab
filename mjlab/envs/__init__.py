"""mjlab envs."""

from typing import Union

# Toy envs.
from mjlab.envs.toy.cartpole import CartpoleConfig
from mjlab.envs.toy.humanoid import (
  HumanoidStandConfig,
  HumanoidWalkConfig,
  HumanoidRunConfig,
)

# Playground envs.
from mjlab.envs.playground.go1_joystick import Go1JoystickConfig
from mjlab.envs.playground.go1_getup import Go1GetupConfig
from mjlab.envs.playground.g1_joystick import G1Config

# Motion imitation envs.
from mjlab.envs.motion_imitation import G1MotionTrackingDanceConfig

TaskConfigUnion = Union[
  CartpoleConfig,
  Go1JoystickConfig,
  Go1GetupConfig,
  HumanoidStandConfig,
  HumanoidWalkConfig,
  HumanoidRunConfig,
  G1Config,
  G1MotionTrackingDanceConfig,
]
