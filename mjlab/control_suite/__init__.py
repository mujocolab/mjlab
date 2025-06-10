"""Control suite."""

from mjlab._src.registry import _REGISTRY
from mjlab.control_suite import cartpole

_REGISTRY.register_task(
    "Cartpole-v0",
    cartpole.Swingup,
    cartpole.CartpoleConfig,
)
