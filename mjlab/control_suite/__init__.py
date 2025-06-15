"""Control suite."""

from mjlab._src.registry import _REGISTRY
from mjlab.control_suite import cartpole
from mjlab.control_suite import go1
from mjlab.control_suite import humanoid
from mjlab.control_suite import g1

_REGISTRY.register_task(
  "Cartpole-v0",
  cartpole.Swingup,
  cartpole.CartpoleConfig,
)

_REGISTRY.register_task(
  "Velocity-Flat-Unitree-Go1-v0",
  go1.Go1Env,
  go1.Go1Config,
)

_REGISTRY.register_task(
  "Humanoid-Stand-v0",
  humanoid.HumanoidStand,
  humanoid.HumanoidStandConfig,
)

_REGISTRY.register_task(
  "Humanoid-Walk-v0",
  humanoid.HumanoidWalk,
  humanoid.HumanoidWalkConfig,
)

_REGISTRY.register_task(
  "Humanoid-Run-v0",
  humanoid.HumanoidRun,
  humanoid.HumanoidRunConfig,
)

_REGISTRY.register_task(
  "Velocity-Flat-G1-v0",
  g1.G1Env,
  g1.G1Config,
)
