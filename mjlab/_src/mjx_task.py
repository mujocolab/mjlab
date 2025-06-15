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

_integrator_map = {
  "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
  "euler": mujoco.mjtIntegrator.mjINT_EULER,
}


@dataclass(frozen=True)
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
  integrator: str
  """Integrator to use for the simulation."""

  def apply_defaults(self, spec: mujoco.MjSpec) -> mujoco.MjSpec:
    # TODO(kevin): Should we make a copy?
    spec.option.integrator = _integrator_map[self.integrator]
    spec.option.timestep = self.sim_dt
    spec.option.iterations = self.solver_iterations
    spec.option.ls_iterations = self.solver_ls_iterations
    if not self.euler_damping:
      spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    return spec


ConfigT = TypeVar("ConfigT", bound=TaskConfig)


class MjxTask(abc.ABC, Generic[ConfigT]):
  """Base class for all tasks."""

  def __init__(
    self,
    config: TaskConfig,
    spec: mujoco.MjSpec,
    entities: Optional[Dict[str, entity.Entity]] = None,
  ):
    self._config = config
    self._entities = entities
    self._spec = config.apply_defaults(spec)
    self._model = self._spec.compile()
    self._mjx_model = mjx.put_model(self._model)

  @classmethod
  def from_xml_str(
    cls, config: TaskConfig, xml: str, assets: Optional[Dict[str, bytes]] = None
  ) -> "MjxTask":
    """Instantiates the task from an xml string."""
    spec = mujoco.MjSpec.from_string(xml, assets=assets)
    return cls(config, entity.Entity(spec))

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

  def domain_randomize(self, model: mjx.Model, rng: jax.Array) -> Tuple[mjx.Model, Any]:
    """Applies domain randomization to the model."""
    del rng  # Unused.
    in_axes = jax.tree.map(lambda x: None, model)
    return model, in_axes

  def before_step(self, action: jax.Array, state: State) -> mjx.Data:
    """Callback executed before the physics step.

    The default implementation sets the control signal for the actuators in the
    model to be equal to `action`. Subclasses can override this method to implement
    more complex control schemes (e.g. inverse kinematics).
    """
    return state.data.replace(ctrl=action)

  def before_substep(self, data: mjx.Data, action: jax.Array, state: State) -> mjx.Data:
    """Callback executed before the physics substep.

    The default implementation does nothing.
    """
    del action, state  # Unused.
    return data

  def after_substep(self, data: mjx.Data, action: jax.Array, state: State) -> mjx.Data:
    """Callback executed after the physics substep.

    The default implementation does nothing.
    """
    del action, state  # Unused.
    return data

  def after_step(
    self, data: mjx.Data, state: State, action: jax.Array, done: jax.Array
  ) -> mjx.Data:
    """Callback executed after the physics step.

    The default implementation does nothing.
    """
    del action, state, done  # Unused.
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
  def spec(self) -> mujoco.MjSpec:
    """Returns the root spec."""
    return self._spec

  @property
  def cfg(self) -> ConfigT:
    """Returns the task configuration."""
    return self._config

  @property
  def entities(self) -> Dict[str, entity.Entity]:
    """Returns the entities in the task."""
    return self._entities

  @property
  def model(self) -> mujoco.MjModel:
    """Returns the compiled C mujoco model."""
    return self._model

  @property
  def mjx_model(self) -> mjx.Model:
    """Returns the mjx model."""
    return self._mjx_model

  @property
  def action_size(self) -> int:
    """Returns the size of the action space.

    Subclasses should override this property if overriding `apply_action`. By
    default, it returns the number of actuators in the model.
    """
    return self._model.nu

  @property
  def dt(self) -> float:
    """Returns the control time step."""
    return self._config.ctrl_dt

  @property
  def sim_dt(self) -> float:
    """Returns the simulation time step."""
    return self._config.sim_dt

  @property
  def n_substeps(self) -> int:
    """Returns the number of simulation substeps per control step."""
    return int(round(self.dt / self.sim_dt))
