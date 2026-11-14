"""Tests for terrain generation."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
import trimesh

from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sim import Simulation, SimulationCfg
from mjlab.terrains import MeshCollisionMode, MeshTerrainCfg, TerrainEntityCfg
from mjlab.terrains.config import ALL_TERRAIN_PRESETS
from mjlab.terrains.primitive_terrains import (
  _MIN_BORDER_HEIGHT,
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
  BoxSteppingStonesTerrainCfg,
)
from mjlab.terrains.terrain_generator import (
  FlatPatchSamplingCfg,
  SubTerrainCfg,
  TerrainGenerator,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)

_CFG = BoxSteppingStonesTerrainCfg(
  proportion=1.0,
  size=(8.0, 8.0),
  stone_size_range=(0.2, 0.6),
  stone_distance_range=(0.05, 0.25),
  stone_height=0.2,
  stone_height_variation=0.05,
  stone_size_variation=0.05,
  displacement_range=0.1,
  floor_depth=2.0,
  platform_width=1.5,
  border_width=0.25,
)


def _generate_stones(
  cfg: BoxSteppingStonesTerrainCfg,
  difficulty: float,
  rng: np.random.Generator,
) -> list[tuple[float, float, float, float]]:
  """Generate terrain and return stone (cx, cy, half_x, half_y) tuples."""
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(difficulty=difficulty, spec=spec, rng=rng)

  center = cfg.size[0] / 2
  stones = []
  for geom_info in output.geometries:
    geom = geom_info.geom
    if geom is None:
      continue
    pos, size = geom.pos, geom.size
    # Skip platform, floor, and border geoms. The platform is the geom centered
    # exactly at the patch center (its size is grid-snapped, not the configured
    # width, so it is identified by position alone).
    is_platform = np.isclose(pos[0], center) and np.isclose(pos[1], center)
    is_full_span = np.isclose(size[0], cfg.size[0] / 2) or np.isclose(
      size[1], cfg.size[1] / 2
    )
    if is_platform or is_full_span:
      continue
    stones.append((pos[0], pos[1], size[0], size[1]))
  return stones


def test_no_stone_centers_inside_platform():
  """No stone center should fall inside the platform."""
  center = _CFG.size[0] / 2
  p_half = _CFG.platform_width / 2
  p_min, p_max = center - p_half, center + p_half

  for difficulty in [0.0, 0.5, 1.0]:
    stones = _generate_stones(_CFG, difficulty, np.random.default_rng(42))
    for cx, cy, _, _ in stones:
      assert not (p_min <= cx <= p_max and p_min <= cy <= p_max), (
        f"Stone at ({cx:.3f}, {cy:.3f}) inside platform at difficulty={difficulty}"
      )


def test_stone_size_decreases_with_difficulty():
  """Average stone size should be smaller at higher difficulty."""
  sizes = {}
  for difficulty in [0.0, 1.0]:
    stones = _generate_stones(_CFG, difficulty, np.random.default_rng(42))
    sizes[difficulty] = np.mean([hx + hy for _, _, hx, hy in stones])

  assert sizes[0.0] > sizes[1.0]


@pytest.mark.parametrize(
  "cfg_cls", [BoxPyramidStairsTerrainCfg, BoxInvertedPyramidStairsTerrainCfg]
)
def test_pyramid_stairs_border_present_at_zero_difficulty(cfg_cls):
  """At difficulty 0 the step height collapses to 0, but the flat border frame
  must still be generated as solid, non-degenerate geometry (regression for the
  empty-boundary bug, issue #1033)."""
  cfg = cfg_cls(
    size=(8.0, 8.0),
    step_height_range=(0.0, 0.2),
    step_width=0.3,
    platform_width=3.0,
    border_width=1.0,
  )
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(difficulty=0.0, spec=spec, rng=np.random.default_rng(0))

  # The border frame sits below z=0 (top flush at ground level); inner step
  # boxes are centered at z=0. Identify the frame by its downward offset.
  border_geoms = [
    g.geom for g in output.geometries if g.geom is not None and g.geom.pos[2] < -1e-4
  ]
  assert len(border_geoms) == 4, "Expected four border frame boxes."
  for geom in border_geoms:
    # Each frame box must be solid, not a degenerate zero-height geom, and its
    # top must be flush with the ground plane at z=0.
    assert geom.size[2] >= _MIN_BORDER_HEIGHT / 2 - 1e-9
    assert np.isclose(geom.pos[2] + geom.size[2], 0.0, atol=1e-6)


@pytest.mark.parametrize("preset_name", sorted(ALL_TERRAIN_PRESETS))
@pytest.mark.parametrize("difficulty", [0.0, 1.0])
def test_preset_compiles_across_difficulty(preset_name, difficulty):
  """Every terrain preset must generate compilable MuJoCo geometry across the
  full difficulty range. Difficulty 0 is exercised explicitly because curriculum
  row 0 lands there deterministically, which previously produced degenerate
  geometry (zero-height hfields, NaN colors, missing borders)."""
  cfg = ALL_TERRAIN_PRESETS[preset_name](size=(8.0, 8.0))
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  cfg.function(difficulty=difficulty, spec=spec, rng=np.random.default_rng(0))
  # Compiling validates geom/hfield sizes and rgba values (catches NaNs and
  # non-positive sizes that MuJoCo rejects).
  spec.compile()


@dataclass(kw_only=True)
class _BoxesTerrainCfg(SubTerrainCfg):
  """Test terrain that emits two boxes, named per `names` (None leaves unnamed)."""

  names: tuple[str | None, str | None] = (None, None)

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del difficulty, rng
    body = spec.body("terrain")
    geometries = []
    for i in range(2):
      geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.5, 0.5, 0.5),
        pos=(i, 0, 0),
        name=self.names[i],
      )
      geometries.append(TerrainGeometry(geom=geom))
    return TerrainOutput(origin=np.zeros(3), geometries=geometries)


def test_compile_preserves_custom_names_and_defaults_unnamed_geoms():
  """Geoms with a custom name from a sub-terrain must be preserved, while
  unnamed geoms must fall back to the generator's default "terrain_{i}" name."""
  cfg = TerrainGeneratorCfg(
    size=(4.0, 4.0),
    curriculum=True,
    sub_terrains={
      "named": _BoxesTerrainCfg(size=(4.0, 4.0), names=("foo_0", "foo_1")),
      "unnamed": _BoxesTerrainCfg(size=(4.0, 4.0)),
    },
  )
  generator = TerrainGenerator(cfg)
  spec = mujoco.MjSpec()
  generator.compile(spec)
  model = spec.compile()

  names = [model.geom(i).name for i in range(model.ngeom)]
  assert names == ["foo_0", "foo_1", "terrain_2", "terrain_3"]


@pytest.mark.parametrize(
  "preset_name",
  [
    "hf_pyramid_slope",
    "random_rough",
    "wave_terrain",
    "discrete_obstacles",
    "perlin_noise",
  ],
)
def test_heightfield_rows_run_along_y(preset_name):
  """MuJoCo lays hfield rows along y and columns along x. Otherwise a non-square
  terrain is stretched and its flat patches land off the sub-terrain."""
  size = (4.0, 2.0)
  patch_cfg = FlatPatchSamplingCfg(num_patches=20, patch_radius=0.2)
  cfg = ALL_TERRAIN_PRESETS[preset_name](
    size=size, flat_patch_sampling={"init": patch_cfg}
  )
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  out = cfg.function(difficulty=1.0, spec=spec, rng=np.random.default_rng(0))

  hfield = next(g.hfield for g in out.geometries if g.hfield is not None)
  assert hfield.ncol == 2 * hfield.nrow  # Columns span x, twice as long as y.
  assert out.flat_patches is not None
  x, y = out.flat_patches["init"][:, 0], out.flat_patches["init"][:, 1]
  assert np.all((x >= 0) & (x <= size[0]) & (y >= 0) & (y <= size[1]))


def _export_concave_step_mesh(path: Path) -> Path:
  """Create a concave two-level step mesh: low shelf top z=0.0, high wall top z=1.0."""
  low = trimesh.creation.box(extents=(4.0, 4.0, 0.2))
  low.apply_translation((0.0, 0.0, -0.1))  # x,y in [-2, 2], top z = 0.0
  high = trimesh.creation.box(extents=(1.0, 4.0, 1.0))
  high.apply_translation((1.5, 0.0, 0.5))  # x in [1, 2], top z = 1.0
  combined = trimesh.util.concatenate([low, high])
  combined.export(path)
  return path


def _falling_ball_spec(radius: float = 0.05) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="ball")
  body.add_freejoint()
  body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=[radius, 0.0, 0.0],
    mass=0.5,
    group=3,
  )
  return spec


def test_mesh_terrain_subterrain_generator_and_flat_patches(tmp_path: Path):
  """MeshTerrainCfg compiles in TerrainGenerator, scales Z by difficulty, and finds
  flat patches on horizontal plateaus."""
  mesh_file = _export_concave_step_mesh(tmp_path / "stepped.obj")

  patch_cfg = FlatPatchSamplingCfg(
    num_patches=12,
    patch_radius=0.25,
    max_height_diff=0.02,
  )
  gen_cfg = TerrainGeneratorCfg(
    size=(4.0, 4.0),
    num_rows=2,
    curriculum=True,
    seed=7,
    sub_terrains={
      "mesh_step": MeshTerrainCfg(
        mesh_path=mesh_file,
        size=(4.0, 4.0),
        z_scale_range=(0.5, 1.0),
        origin=(3.5, 2.0, 1.0),
        flat_patch_sampling={"spawn": patch_cfg},
      ),
    },
  )
  # Also check automatic center surface Z when origin is None.
  auto_origin_cfg = MeshTerrainCfg(
    mesh_path=mesh_file,
    size=(4.0, 4.0),
    z_scale_range=(0.5, 1.0),
    flat_patch_sampling={"spawn": patch_cfg},
  )
  gen_cfg_auto = TerrainGeneratorCfg(
    size=(4.0, 4.0),
    num_rows=2,
    curriculum=True,
    seed=7,
    sub_terrains={"mesh_step": auto_origin_cfg},
  )
  del gen_cfg

  generator = TerrainGenerator(gen_cfg_auto)
  spec = mujoco.MjSpec()
  generator.compile(spec)
  model = spec.compile()
  assert model.nmesh == 2
  assert model.nhfield == 2

  patches = generator.flat_patches["spawn"]
  assert patches.shape == (2, 1, 12, 3)
  # Row 0 (difficulty=0 -> z_scale=0.5): flat plateaus are at z=0.0 or z=0.5.
  row0_z = patches[0, 0, :, 2]
  assert np.all(np.isclose(row0_z, 0.0, atol=1e-2) | np.isclose(row0_z, 0.5, atol=1e-2))
  # Row 1 (difficulty=1 -> z_scale=1.0): flat plateaus are at z=0.0 or z=1.0.
  row1_z = patches[1, 0, :, 2]
  assert np.all(np.isclose(row1_z, 0.0, atol=1e-2) | np.isclose(row1_z, 1.0, atol=1e-2))


@pytest.mark.parametrize("collision_mode", ["heightfield", "parts"])
def test_mesh_terrain_entity_concave_collision_and_raycast(
  tmp_path: Path, collision_mode: MeshCollisionMode
):
  """Standalone mesh terrain collides concavely (no convex-hull tenting) and works
  with terrain-only Group 0 RayCastSensor."""
  mesh_file = _export_concave_step_mesh(tmp_path / "concave_step.stl")
  ball_radius = 0.05

  def _add_probe_sites(spec: mujoco.MjSpec) -> None:
    spec.worldbody.add_site(name="probe_low", pos=(-0.5, 0.0, 2.0))
    spec.worldbody.add_site(name="probe_high", pos=(1.5, 0.0, 2.0))

  scene_cfg = SceneCfg(
    num_envs=2,
    env_spacing=1.0,
    terrain=TerrainEntityCfg(
      terrain_type="mesh",
      mesh_path=mesh_file,
      mesh_center_xy=False,
      mesh_collision_mode=collision_mode,
      mesh_horizontal_scale=0.05,
    ),
    entities={"ball": EntityCfg(spec_fn=lambda: _falling_ball_spec(ball_radius))},
    sensors=(
      RayCastSensorCfg(
        name="scan_low",
        frame=ObjRef(type="site", name="probe_low"),
        pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1),
        include_geom_groups=(0,),
        exclude_parent_body=False,
      ),
      RayCastSensorCfg(
        name="scan_high",
        frame=ObjRef(type="site", name="probe_high"),
        pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1),
        include_geom_groups=(0,),
        exclude_parent_body=False,
      ),
    ),
    spec_fn=_add_probe_sites,
  )
  scene = Scene(scene_cfg, device="cpu")
  assert torch.allclose(scene.env_origins[:, 2], torch.zeros(2), atol=1e-5)

  sim = Simulation(num_envs=2, cfg=SimulationCfg(), model=scene.compile(), device="cpu")
  scene.initialize(sim.mj_model, sim.model, sim.data)
  assert scene.sensor_context is not None
  sim.set_sensor_context(scene.sensor_context)

  # Drop ball 0 onto the concave low shelf (x=-0.5 -> z=0.0) and ball 1 onto the
  # elevated step (x=1.5 -> z=1.0). A naive single convex hull would ramp through
  # z=0.5 at x=-0.5 instead of resting at z=0.05.
  ball = scene["ball"]
  init_states = torch.zeros((2, 13), dtype=torch.float32)
  init_states[0, :7] = torch.tensor([-0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0])
  init_states[1, :7] = torch.tensor([1.5, 0.0, 1.3, 1.0, 0.0, 0.0, 0.0])
  ball.write_root_state_to_sim(init_states)
  sim.forward()
  sim.sense()

  # RayCastSensor in group 0 measures exact distances from z=2.0: 2.0m and 1.0m.
  torch.testing.assert_close(
    scene["scan_low"].data.distances,
    torch.full((2, 1), 2.0),
    atol=1e-4,
    rtol=1e-4,
  )
  torch.testing.assert_close(
    scene["scan_high"].data.distances,
    torch.full((2, 1), 1.0),
    atol=1e-4,
    rtol=1e-4,
  )

  for _ in range(400):
    sim.step()
  sim.forward()

  ball_z = ball.data.root_link_pos_w[:, 2]
  assert abs(float(ball_z[0]) - ball_radius) < 0.02
  assert abs(float(ball_z[1]) - (1.0 + ball_radius)) < 0.02


def test_mesh_terrain_convex_mode_and_validation(tmp_path: Path):
  """Convex mode compiles cleanly and invalid configs raise descriptive errors."""
  box = trimesh.creation.box(extents=(3.0, 3.0, 0.4))
  mesh_path = tmp_path / "flat_slab.ply"
  box.export(mesh_path)

  scene_cfg = SceneCfg(
    num_envs=1,
    terrain=TerrainEntityCfg(
      terrain_type="mesh",
      mesh_path=mesh_path,
      mesh_collision_mode="convex",
      mesh_align_min_z=True,
    ),
  )
  scene = Scene(scene_cfg, device="cpu")
  model = scene.compile()
  assert model.nmesh == 1
  assert abs(float(scene.env_origins[0, 2]) - 0.4) < 1e-5

  # Zero-thickness coplanar quad (.obj) also compiles via thin extrusion guard.
  quad_path = tmp_path / "flat_quad.obj"
  quad = trimesh.Trimesh(
    vertices=[
      [-1.0, -1.0, 0.25],
      [1.0, -1.0, 0.25],
      [1.0, 1.0, 0.25],
      [-1.0, 1.0, 0.25],
    ],
    faces=[[0, 1, 2], [0, 2, 3]],
  )
  quad.export(quad_path)
  quad_scene = Scene(
    SceneCfg(
      num_envs=1,
      terrain=TerrainEntityCfg(
        terrain_type="mesh",
        mesh_path=quad_path,
        mesh_collision_mode="convex",
      ),
    ),
    device="cpu",
  )
  assert quad_scene.compile().nmesh == 1
  assert abs(float(quad_scene.env_origins[0, 2]) - 0.25) < 1e-5

  with pytest.raises(ValueError, match="mesh_path"):
    Scene(
      SceneCfg(num_envs=1, terrain=TerrainEntityCfg(terrain_type="mesh")),
      device="cpu",
    )

  with pytest.raises(FileNotFoundError):
    Scene(
      SceneCfg(
        num_envs=1,
        terrain=TerrainEntityCfg(
          terrain_type="mesh", mesh_path=tmp_path / "missing.obj"
        ),
      ),
      device="cpu",
    )

  with pytest.raises(ValueError, match="positive"):
    Scene(
      SceneCfg(
        num_envs=1,
        terrain=TerrainEntityCfg(
          terrain_type="mesh",
          mesh_path=mesh_path,
          mesh_scale=(1.0, 0.0, 1.0),
        ),
      ),
      device="cpu",
    )
