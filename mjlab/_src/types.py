from typing import Any, Dict, Mapping, Tuple, Union

import jax
from flax import struct
from mujoco import mjx

# Type aliases.
Observation = Union[jax.Array, Mapping[str, jax.Array]]
ObservationSize = Union[int, Mapping[str, Union[Tuple[int, ...], int]]]
Metrics = Dict[str, jax.Array]
Info = Dict[str, Any]


@struct.dataclass
class State:
    """Environment state for training and inference."""

    data: mjx.Data
    obs: Observation
    reward: jax.Array
    done: jax.Array
    metrics: Metrics
    info: Info

    def tree_replace(self, params: Dict[str, jax.typing.ArrayLike]) -> "State":
        """Replace nested attributes using dot notation."""
        new = self
        for k, v in params.items():
            new = _tree_replace(new, k.split("."), v)
        return new


def _tree_replace(base: Any, attr: list[str], val: jax.typing.ArrayLike) -> Any:
    """Helper for tree_replace."""
    if not attr:
        return base
    if len(attr) == 1:
        return base.replace(**{attr[0]: val})
    return base.replace(
        **{attr[0]: _tree_replace(getattr(base, attr[0]), attr[1:], val)}
    )
