"""Control suite."""

from mjlab._src.registry import _REGISTRY
from mjlab.control_suite import cartpole
from mjlab.control_suite import go1

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
