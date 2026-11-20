from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, cast

import mujoco
import numpy as np
import pytest
import torch

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, mdp
from mjlab.managers import EventTermCfg, SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer.viser.scene import MjlabViserScene, _PerWorldMeshGroup


@dataclass
class _Handle:
  """Like viser's handles: a dataclass with array fields, so == is ambiguous."""

  batched_positions: np.ndarray = field(default_factory=lambda: np.zeros((1, 3)))
  mesh: Any = None
  visible: bool = True
  batched_wxyzs: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
  position: np.ndarray = field(default_factory=lambda: np.zeros(3))
  wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))

  def remove(self) -> None:
    pass


class _Scene:
  """Counts body meshes baked; every other add_* call returns a dummy handle."""

  def __init__(self):
    self.baked = 0

  def __getattr__(self, name):
    return lambda *args, **kwargs: _Handle()

  def add_batched_meshes_trimesh(self, path, mesh, **kwargs) -> _Handle:
    self.baked += 1
    return _Handle(mesh=mesh, visible=kwargs.get("visible", True))


class _Server:
  def __init__(self):
    self.scene = _Scene()

  def atomic(self):
    return nullcontext()

  def flush(self) -> None:
    pass


def _spec(geom_type: mujoco.mjtGeom) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  spec.add_material(name="paint", rgba=[1, 0, 0, 1])
  body = spec.worldbody.add_body(name="body")
  body.add_freejoint()
  body.add_geom(name="geom", type=geom_type, size=[0.1] * 3, material="paint")
  return spec


@pytest.fixture
def make_scene(monkeypatch):
  """Build a box-and-ball env and a viser scene drawing it on a fake server."""
  envs = []

  def make(events=None, expand=()):
    env = ManagerBasedRlEnv(
      ManagerBasedRlEnvCfg(
        decimation=1,
        scene=SceneCfg(
          terrain=TerrainEntityCfg(terrain_type="plane"),
          num_envs=2,
          # Ball first, so the box's mesh groups aren't first in the list.
          entities={
            "ball": EntityCfg(spec_fn=lambda: _spec(mujoco.mjtGeom.mjGEOM_SPHERE)),
            "box": EntityCfg(spec_fn=lambda: _spec(mujoco.mjtGeom.mjGEOM_BOX)),
          },
        ),
        events=events or {},
      ),
      device="cpu",
    )
    envs.append(env)
    env.sim.expand_model_fields(expand)
    scene = MjlabViserScene(
      cast(Any, _Server()),
      env.sim.mj_model,
      env.num_envs,
      sim_model=env.sim.model,
      expanded_fields=env.sim.expanded_fields,
    )
    monkeypatch.setattr(
      scene, "rebuild_visual_handles", lambda: pytest.fail("full rebuild")
    )
    return env, scene

  yield make
  for env in envs:
    env.close()


def _render(env: ManagerBasedRlEnv, scene: MjlabViserScene) -> int:
  """Draw a frame and return how many meshes it baked."""
  server = cast(_Server, scene.server)
  baked = server.scene.baked
  xpos = env.sim.data.xpos.cpu().numpy()
  xmat = env.sim.data.xmat.cpu().numpy()
  scene.update_from_arrays(xpos, xmat, env_idx=0)
  return server.scene.baked - baked


def _groups(scene: MjlabViserScene, env: ManagerBasedRlEnv, name: str):
  body = env.sim.mj_model.body(name).id
  return [
    mg
    for mg in scene._mesh_groups
    if isinstance(mg, _PerWorldMeshGroup) and np.any(mg.body_ids == body)
  ]


def _drawn_mesh_per_env(scene, env, name: str) -> dict[int, Any]:
  """The mesh drawn for a body in each env, checking it's drawn exactly once."""
  drawn = {}
  for mg in _groups(scene, env, name):
    for env_id in mg.env_ids.tolist():
      assert env_id not in drawn
      drawn[env_id] = cast(_Handle, mg.handle).mesh
  assert sorted(drawn) == list(range(env.num_envs))
  return drawn


def test_reset_that_resizes_a_geom_rebakes_only_that_mesh(make_scene):
  """Regression for #1278: dr.geom_size on reset left the old mesh size."""
  resize = EventTermCfg(
    mode="reset",
    func=mdp.dr.geom_size,
    params={
      "asset_cfg": SceneEntityCfg("box", geom_names=("geom",)),
      "ranges": (0.15, 0.5),
      "operation": "abs",
    },
  )
  env, scene = make_scene(events={"resize": resize})
  _render(env, scene)
  ball_handles = [mg.handle for mg in _groups(scene, env, "ball/body")]

  env.reset()
  assert _render(env, scene) > 0
  geom = env.sim.mj_model.geom("box/geom").id
  sim_widths = 2 * env.sim.model.geom_size[:, geom, 0].cpu().numpy()
  drawn = _drawn_mesh_per_env(scene, env, "box/body")
  widths = [np.ptp(drawn[e].vertices[:, 0]) for e in range(env.num_envs)]
  np.testing.assert_allclose(widths, sim_widths, rtol=1e-5)
  assert [mg.handle for mg in _groups(scene, env, "ball/body")] == ball_handles

  assert _render(env, scene) == 0  # Nothing changed, so nothing is baked.


def test_material_change_rebakes_only_the_env_that_uses_it(make_scene):
  env, scene = make_scene(expand=("mat_rgba",))
  _render(env, scene)
  material = env.sim.mj_model.material("box/paint").id
  env.sim.model.mat_rgba[1, material] = torch.tensor([0.0, 0.0, 1.0, 1.0])

  assert _render(env, scene) == 1
  drawn = _drawn_mesh_per_env(scene, env, "box/body")
  red, blue = (255, 0, 0, 255), (0, 0, 255, 255)
  assert tuple(drawn[0].visual.vertex_colors[0]) == red
  assert tuple(drawn[1].visual.vertex_colors[0]) == blue
