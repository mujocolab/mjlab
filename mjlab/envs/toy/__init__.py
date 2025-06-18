"""DM control suite."""

from mjlab.envs.registry import _REGISTRY
from mjlab.envs.toy import cartpole
from mjlab.envs.toy import humanoid

_REGISTRY.register_task(
  "Cartpole-v0",
  cartpole.Swingup,
  cartpole.CartpoleConfig,
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
