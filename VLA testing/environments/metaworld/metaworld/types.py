from __future__ import annotations

from typing import Any, NamedTuple, Tuple

import numpy as np
import numpy.typing as npt
from typing_extensions import NotRequired, TypeAlias, TypedDict


class Task(NamedTuple):
    """All data necessary to describe a single MDP.

    Should be passed into a `MetaWorldEnv`'s `set_task` method.
    """

    env_name: str
    data: bytes  # Contains env parameters like random_init and *a* goal


XYZ: TypeAlias = "Tuple[float, float, float]"
"""A 3D coordinate."""


class EnvironmentStateDict(TypedDict):
    state: dict[str, Any]
    mjb: str
    mocap: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]


class ObservationDict(TypedDict):
    state_observation: npt.NDArray[np.float64]
    state_desired_goal: npt.NDArray[np.float64]
    state_achieved_goal: npt.NDArray[np.float64]


class InitConfigDict(TypedDict):
    obj_init_angle: NotRequired[float]
    obj_init_pos: npt.NDArray[Any]
    hand_init_pos: npt.NDArray[Any]


class HammerInitConfigDict(TypedDict):
    hammer_init_pos: npt.NDArray[Any]
    hand_init_pos: npt.NDArray[Any]


class StickInitConfigDict(TypedDict):
    stick_init_pos: npt.NDArray[Any]
    hand_init_pos: npt.NDArray[Any]
