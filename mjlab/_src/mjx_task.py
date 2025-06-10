import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

from mjlab._src import entity
from mjlab._src.types import Observation, State


@dataclass
class TaskConfig:
    """Base class for task configs."""

    sim_dt: float
    """Simulation time step, in seconds."""
    ctrl_dt: float
    """Control time step, in seconds."""
    solver_iterations: int
    """Number of solver iterations."""
    solver_ls_iterations: int
    """Number of solver line search iterations."""
    euler_damping: bool
    """Whether to use implicit integration with resepect to joint damping when using
  the Euler integrator. Typically set to `False` for better performance in MJX."""
    max_episode_length: int
    """Maximum number of steps per episode."""


ConfigT = TypeVar("ConfigT", bound=TaskConfig)


class MjxTask(abc.ABC, Generic[ConfigT]):
    """Base class for all tasks."""

    @classmethod
    def from_xml_str(
        cls, config: TaskConfig, xml: str, assets: Optional[Dict[str, bytes]] = None
    ) -> "MjxTask":
        """Instantiates the task from an xml string."""
        spec = mujoco.MjSpec.from_string(xml, assets=assets)
        return cls(config, spec)

    @classmethod
    def from_xml_path(
        cls,
        config: TaskConfig,
        xml_path: Path,
        assets: Optional[Dict[str, bytes]] = None,
    ) -> "MjxTask":
        """Instantiates the task from an xml file."""
        with open(xml_path, "r") as f:
            xml = f.read()
        return cls.from_xml_str(config, xml, assets)

    def __init__(
        self,
        config: TaskConfig,
        spec: mujoco.MjSpec,
        entities: Optional[Dict[str, entity.Entity]] = None,
    ):
        self._config = config
        self._spec = spec
        self._entities = entities

        # Compile model and transfer to mjx.
        self._mj_model = self._spec.compile()
        self._mj_model.opt.timestep = self._config.sim_dt
        self._mj_model.opt.iterations = self._config.solver_iterations
        self._mj_model.opt.ls_iterations = self._config.solver_ls_iterations
        if self._config.euler_damping:
            self._mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
        self._mjx_model = mjx.put_model(self._mj_model)

        self.after_compile()

    def after_compile(self) -> None:
        """Callback executed after construction.

        The default implementation binds the entities to compiled model.
        """
        for entity in self._entities.values():
            entity.bind_model(self.mjx_model)

    def before_step(self, action: jax.Array, state: State) -> mjx.Data:
        """Callback executed before the physics step.

        The default implementation sets the control signal for the actuators in the
        model to be equal to `action`. Subclasses can override this method to implement
        more complex control schemes (e.g. inverse kinematics).
        """
        return state.data.replace(ctrl=action)

    def before_substep(
        self, data: mjx.Data, action: jax.Array, state: State
    ) -> mjx.Data:
        """Callback executed before the physics substep.

        The default implementation does nothing.
        """
        del action, state  # Unused.
        return data

    def after_substep(
        self, data: mjx.Data, action: jax.Array, state: State
    ) -> mjx.Data:
        """Callback executed after the physics substep.

        The default implementation does nothing.
        """
        del action, state  # Unused.
        return data

    def after_step(self, data: mjx.Data, state: State) -> mjx.Data:
        """Callback executed after the physics step.

        The default implementation does nothing.
        """
        del state  # Unused.
        return data

    def should_terminate_episode(self, data: mjx.Data, state: State) -> jax.Array:
        """Determines whether the episode should terminate."""
        del data, state  # Unused.
        return jp.array(False)

    @abc.abstractmethod
    def initialize_episode(
        self, data: mjx.Data, rng: jax.Array
    ) -> Tuple[mjx.Data, Dict[str, Any], Dict[str, Any]]:
        """Modifies the physics state before the next episode begins."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_observation(self, data: mjx.Data, state: State) -> Observation:
        """Calculates the observation given the physics state."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_reward(
        self,
        data: mjx.Data,
        state: State,
        action: jax.Array,
        done: jax.Array,
    ) -> jax.Array:
        """Calculates the reward given the physics state."""
        raise NotImplementedError

    # Properties.

    @property
    def cfg(self) -> ConfigT:
        """Returns the task configuration."""
        return self._config

    @property
    def mj_model(self) -> mujoco.MjModel:
        """Returns the compiled C mujoco model."""
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        """Returns the compiled MJX model."""
        return self._mjx_model

    @property
    def action_size(self) -> int:
        """Returns the size of the action space.

        Subclasses should override this property if overriding `apply_action`. By
        default, it returns the number of actuators in the model.
        """
        return self._mjx_model.nu

    @property
    def dt(self) -> float:
        """Returns the control time step."""
        return self._config.ctrl_dt

    @property
    def sim_dt(self) -> float:
        """Returns the simulation time step."""
        return self._mj_model.opt.timestep

    @property
    def n_substeps(self) -> int:
        """Returns the number of simulation substeps per control step."""
        return int(round(self.dt / self.sim_dt))
