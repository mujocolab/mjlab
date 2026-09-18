"""Registry for pluggable simulation backends.

mjlab ships three built-in backends, ``mjwarp`` (:class:`mjlab.sim.sim.Simulation`),
``mujoco`` (:class:`mjlab.sim.mujoco_sim.MujocoSimulation`), and
``mujoco_with_kinematics``
(:class:`mjlab.sim.mujoco_sim.MujocoSimulationWithKinematics`), which
:class:`mjlab.envs.ManagerBasedRlEnv` dispatches to directly. This module lets
*external* code add further backends (selected by ``SimulationCfg.backend``)
without mjlab importing them: a backend registers its class here, and the env
looks it up by name when ``backend`` is none of the built-ins.

A backend class is any type satisfying
:class:`mjlab.sim.interface.SimulationProtocol`; the env constructs it with
``num_envs``, ``cfg``, ``spec`` and ``device`` keyword arguments. Registration
is import-triggered, so the consumer must import the module that calls
:func:`register_simulation_backend` before building the env.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.sim.mujoco_sim import MujocoSimulation, MujocoSimulationWithKinematics
from mjlab.sim.sim import Simulation, SimulationCfg

if TYPE_CHECKING:
  from mjlab.sim.interface import SimulationProtocol


# Names of the built-in backends, dispatched directly by ManagerBasedRlEnv. These
# names cannot be used for externally-registered backends.
BACKEND_MJWARP = "mjwarp"
BACKEND_MUJOCO = "mujoco"
BACKEND_MUJOCO_WITH_KINEMATICS = "mujoco_with_kinematics"
BUILT_IN_BACKENDS = (BACKEND_MJWARP, BACKEND_MUJOCO, BACKEND_MUJOCO_WITH_KINEMATICS)
_BACKEND_REGISTRY: dict[str, type[SimulationProtocol]] = {
  SimulationCfg.backend: Simulation,
  BACKEND_MUJOCO: MujocoSimulation,
  BACKEND_MUJOCO_WITH_KINEMATICS: MujocoSimulationWithKinematics,
}


def register_simulation_backend(name: str, sim_cls: type[SimulationProtocol]) -> None:
  """Register a simulation backend class under ``name``.

  Args:
    name: The ``SimulationCfg.backend`` value that selects this backend. Must
      not collide with the built-in ``"mjwarp"``, ``"mujoco"``, or
      ``"mujoco_with_kinematics"`` names.
    sim_cls: The class implementing :class:`SimulationProtocol` for this
      backend; the env constructs it as
      ``sim_cls(num_envs=..., cfg=..., spec=..., device=...)``.

  Raises:
    ValueError: If ``name`` is a built-in backend or already registered with a
      different class.
  """
  existing = _BACKEND_REGISTRY.get(name)
  if existing is not None and existing != sim_cls:
    msg = f"Backend {name!r} is already registered with a different class."
    raise ValueError(msg)
  _BACKEND_REGISTRY[name] = sim_cls


def get_simulation_backend(name: str) -> type[SimulationProtocol]:
  """Look up a registered backend class by name.

  Returns:
    The backend class.

  Raises:
    ValueError: If no backend is registered under ``name``, listing the
      registered names to aid debugging.
  """
  try:
    return _BACKEND_REGISTRY[name]
  except KeyError:
    known = sorted(_BACKEND_REGISTRY)
    msg = (
      f"Unknown simulation backend {name!r}. Registered backends are {known}. "
      "Did you forget to import the module that registers it?"
    )
    raise ValueError(msg) from None
