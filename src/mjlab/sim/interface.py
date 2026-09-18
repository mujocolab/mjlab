"""Common public interface shared by the simulation backends.

mjlab ships three simulation backends:

* :class:`mjlab.sim.sim.Simulation` -- GPU-accelerated, powered by MJWarp.
* :class:`mjlab.sim.mujoco_sim.MujocoSimulation` -- CPU physics via
  ``mujoco.rollout``.
* :class:`mjlab.sim.mujoco_sim.MujocoSimulationWithKinematics` -- the MuJoCo
  backend extended to also expose derived body kinematics in ``data``. Refreshing
  those kinematics on every step/reset/forward makes it noticeably slower than the
  plain ``MujocoSimulation`` backend, so prefer ``MujocoSimulation`` unless you need the
  derived kinematics.

:class:`mjlab.envs.ManagerBasedRlEnv` and the viewer drive whichever backend
the configured ``SimulationCfg.backend`` produced, touching only the members
all backends implement. This module extrapolates that shared surface into
explicit :class:`typing.Protocol` definitions, so the contract is checked
statically.

The MuJoCo backend is the limiting one: it deliberately implements a subset of
MJWarp ``Simulation`` (no heterogeneous worlds / mesh variants or raycast
sensors), so these protocols describe exactly its public surface.
``Simulation`` exposes additional members (``wp_model``, ``get_default_field``,
``create_graph``, ``world_to_variant``, ...) that only apply to the MJWarp path
and are therefore intentionally absent here; code that uses them is, by
construction, MJWarp-only and should depend on the concrete class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
  import mujoco
  import torch

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
  from mjlab.managers.event_manager import RecomputeLevel
  from mjlab.sensor.sensor_context import SensorContext
  from mjlab.sim.sim import SimulationCfg


class SimModelProtocol(Protocol): ...


class SimDataProtocol(Protocol): ...


class SimulationProtocol(Protocol):
  """Public interface common to all simulation backends.

  This is the surface :class:`mjlab.envs.ManagerBasedRlEnv` and the viewer rely
  on when driving the simulation without knowing which backend is active. The
  built-in :class:`mjlab.sim.sim.Simulation`,
  :class:`mjlab.sim.mujoco_sim.MujocoSimulation`, and
  :class:`mjlab.sim.mujoco_sim.MujocoSimulationWithKinematics` backends all
  structurally satisfy it, as must any backend registered via
  :mod:`mjlab.sim.registry`.
  """

  cfg: SimulationCfg
  num_envs: int
  device: str

  # Constructor.

  def __init__(
    self,
    num_envs: int,
    cfg: SimulationCfg,
    model: mujoco.MjModel | None = None,
    device: str = "cuda:0",
    *,
    spec: mujoco.MjSpec | None = None,
  ):
    """Construct a simulation backend.

    Args:
        num_envs: Number of parallel environments to simulate.
        cfg: Simulation configuration.
        model: Pre-compiled MuJoCo model to use (optional).
        device: Device to place the simulation on.
        spec: Pre-compiled MuJoCo spec to use (optional).
    """
    ...

  # Properties.

  @property
  def mj_model(self) -> mujoco.MjModel: ...

  @property
  def mj_data(self) -> mujoco.MjData: ...

  @property
  def data(self) -> SimDataProtocol: ...

  @property
  def model(self) -> SimModelProtocol: ...

  @property
  def expanded_fields(self) -> set[str]:
    """Names of model fields expanded for per-env parameters (empty on MuJoCo)."""
    ...

  # Methods.

  def step(self) -> None:
    """Advance physics by one step for all envs."""
    ...

  def forward(self) -> None:
    """Recompute derived quantities (e.g. sensordata) without advancing time."""
    ...

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Reset the given envs to their default state, or all envs when ``None``."""
    ...

  def sense(self) -> None:
    """Run the sensing pipeline (e.g. render camera sensors)."""
    ...

  def set_sensor_context(self, ctx: SensorContext) -> None:
    """Wire a sensor context."""
    ...

  def expand_model_fields(self, fields: tuple[str, ...]) -> None:
    """Expand model fields for per-env parameters."""
    ...

  def recompute_constants(self, level: RecomputeLevel) -> None:
    """Recompute derived model constants (e.g. after domain randomization)."""
    ...

  # Environment and Model setup methods

  @classmethod
  def setup_cfg(cls, cfg: ManagerBasedRlEnvCfg) -> None:
    """Apply sim backend-specific fixups to the environment configuration."""
    ...

  def setup_model(self, model: mujoco.MjModel) -> None:
    """Apply sim backend-specific fixups to the model."""
    ...


if TYPE_CHECKING:
  # Static conformance checks: both concrete backends must satisfy the protocol.
  # These assignments fail type checking if either backend's public surface
  # drifts from the interface above. Never executed at runtime.
  from mjlab.sim.mujoco_sim import MujocoSimulation
  from mjlab.sim.sim import Simulation

  def _assert_backends_conform(
    warp_sim: Simulation, mujoco_sim: MujocoSimulation
  ) -> None:
    _warp: SimulationProtocol = warp_sim
    _mujoco: SimulationProtocol = mujoco_sim
