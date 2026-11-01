"""Constant graph replay must observe new inputs and preserve physical state."""

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.sim import Simulation, SimulationCfg

_XML = """
<mujoco><option gravity="0 0 0"/><worldbody>
  <body><freejoint/>
    <geom type="box" size="0.1 0.2 0.3" mass="1"/>
    <body pos="0.3 0.1 0.2"><joint axis="0 0 1"/>
      <geom type="sphere" size="0.1" mass="2"/>
    </body>
  </body>
</worldbody></mujoco>
"""


@pytest.mark.parametrize(
  "level",
  [
    RecomputeLevel.set_const_fixed,
    RecomputeLevel.set_const_0,
    RecomputeLevel.set_const,
  ],
)
def test_recompute_replay_uses_current_inputs_and_restores_current_state(level):
  device = get_test_device()
  sims = [
    Simulation(
      2, SimulationCfg(), model=mujoco.MjModel.from_xml_string(_XML), device=device
    )
    for _ in range(2)
  ]
  fields = tuple(getattr(dr.body_mass, "model_fields"))  # noqa: B009
  for sim in sims:
    sim.expand_model_fields(fields)
  cached, reference = sims
  reference.use_cuda_graph = False
  previous_graph = None
  physical_qpos = []
  for generation in (1, 2, 3):
    masses = torch.tensor([generation + 1.0, generation + 3.0], device=device)
    physical_qpos = []
    for sim in sims:
      sim.model.body_mass[:, 1] = masses
      sim.data.qpos[:, 0] = generation * 0.17
      sim.data.qvel[:, 0] = generation * 0.2
      physical_qpos.append(sim.data.qpos[:].clone())
      sim.recompute_constants(level)
    expected_fields = (
      ("body_subtreemass",)
      if level == RecomputeLevel.set_const_fixed
      else ("dof_invweight0", "body_invweight0", "tendon_length0", "actuator_acc0")
    )
    for name in expected_fields:
      torch.testing.assert_close(
        getattr(cached.model, name)[:], getattr(reference.model, name)[:]
      )
    for sim, before in zip(sims, physical_qpos, strict=True):
      torch.testing.assert_close(sim.data.qpos[:], before, atol=0, rtol=0)
      torch.testing.assert_close(
        sim.data.qvel[:, 0],
        torch.full((2,), generation * 0.2, device=device),
        atol=0,
        rtol=0,
      )
    if cached.use_cuda_graph:
      graph = cached._recompute_graphs[level]
      if previous_graph is not None:
        assert graph is previous_graph
      previous_graph = graph
  cached.recompute_constants(RecomputeLevel.none)
  if cached.use_cuda_graph:
    cached.expand_model_fields(("geom_friction",))
    assert not cached._recompute_graphs
    cached.recompute_constants(level)
    assert cached._recompute_graphs[level] is not previous_graph
    torch.testing.assert_close(cached.data.qpos[:], physical_qpos[0], atol=0, rtol=0)


@pytest.mark.parametrize("num_envs", [1, 2])
def test_repeated_field_expansion_preserves_storage_and_graphs(num_envs):
  sim = Simulation(
    num_envs,
    SimulationCfg(),
    model=mujoco.MjModel.from_xml_string(_XML),
    device=get_test_device(),
  )
  fields = tuple(getattr(dr.body_mass, "model_fields"))  # noqa: B009
  sim.expand_model_fields(fields)
  sim.recompute_constants(RecomputeLevel.set_const)
  pointers = {field: getattr(sim.wp_model, field).ptr for field in fields}
  graph, constants = sim.step_graph, dict(sim._recompute_graphs)
  sim.expand_model_fields(fields + fields)
  assert sim.step_graph is graph
  assert sim._recompute_graphs == constants
  assert {field: getattr(sim.wp_model, field).ptr for field in fields} == pointers
  assert set(fields).issubset(sim.expanded_fields)
