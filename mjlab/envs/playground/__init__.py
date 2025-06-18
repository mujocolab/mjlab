"""MuJoCo Playground task suite."""

from mjlab.envs.registry import _REGISTRY
from mjlab.envs.playground import go1_joystick
from mjlab.envs.playground import g1_joystick
from mjlab.envs.playground import go1_getup

_REGISTRY.register_task(
  "Velocity-Flat-G1-v0",
  g1_joystick.G1Env,
  g1_joystick.G1Config,
)

_REGISTRY.register_task(
  "Velocity-Flat-Unitree-Go1-v0",
  go1_joystick.Go1JoystickEnv,
  go1_joystick.Go1JoystickConfig,
)

_REGISTRY.register_task(
  "Getup-Unitree-Go1-v0",
  go1_getup.Go1GetupEnv,
  go1_getup.Go1GetupConfig,
)
