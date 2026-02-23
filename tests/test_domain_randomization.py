"""Tests for the dr.* model-field randomization engine."""

import math

import mujoco
import pytest
import torch
from conftest import get_test_device

from mjlab.entity import EntityCfg
from mjlab.envs.mdp import dr
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

pytestmark = pytest.mark.filterwarnings(
  "ignore:Use of index_put_ on expanded tensors is deprecated:UserWarning"
)

# Shared helpers.

ROBOT_XML = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom name="base_geom" type="box" size="0.1 0.1 0.1" mass="1.0"
        friction="0.5 0.01 0.005"/>
      <site name="base_site" pos="0 0 0.1"/>
      <camera name="front_cam" pos="0.5 0 0.3" xyaxes="0 -1 0 0 0 1" fovy="60"/>
      <camera name="side_cam" pos="0 0.5 0.3" xyaxes="1 0 0 0 0 1" fovy="45"/>
      <light name="top_light" pos="0 0 2" dir="0 0 -1"/>
      <body name="foot1" pos="0.2 0 0">
        <joint name="joint1" type="hinge" axis="0 0 1" range="0 1.57"/>
        <geom name="foot1_geom" type="box" size="0.05 0.05 0.05" mass="0.1"
          friction="0.5 0.01 0.005"/>
      </body>
      <body name="foot2" pos="-0.2 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1" range="0 1.57"/>
        <geom name="foot2_geom" type="box" size="0.05 0.05 0.05" mass="0.1"
          friction="0.5 0.01 0.005"/>
      </body>
      <body name="offset_body" pos="0 0.3 0">
        <joint name="joint3" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
        <inertial pos="0.05 -0.02 0.03" mass="0.5"
          fullinertia="0.001 0.002 0.0015 0.0001 -0.00005 0.00008"/>
        <geom name="offset_geom" type="sphere" size="0.04"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

NUM_ENVS = 4


@pytest.fixture(scope="module")
def device():
  return get_test_device()


class Env:
  def __init__(self, scene, sim, device):
    self.scene = scene
    self.sim = sim
    self.num_envs = scene.num_envs
    self.device = device


def create_test_env(
  device, num_envs=NUM_ENVS, expand_fields=("geom_friction", "dof_damping")
):
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML))
  scene_cfg = SceneCfg(num_envs=num_envs, entities={"robot": entity_cfg})
  scene = Scene(scene_cfg, device)
  model = scene.compile()

  sim = Simulation(num_envs=num_envs, cfg=SimulationCfg(), model=model, device=device)
  scene.initialize(model, sim.model, sim.data)
  if expand_fields:
    sim.expand_model_fields(expand_fields)

  return Env(scene, sim, device)


# Operations: abs, scale, add.


def test_abs_operation(device):
  """Values set directly within range, diversity across envs."""
  torch.manual_seed(42)
  env = create_test_env(device)
  robot = env.scene["robot"]

  dr.geom_friction(
    env,
    env_ids=None,
    ranges=(0.3, 1.2),
    operation="abs",
    asset_cfg=SceneEntityCfg("robot", geom_names=(".*",)),
    axes=[0],
  )

  friction = env.sim.model.geom_friction[:, robot.indexing.geom_ids, 0]
  assert torch.all((friction >= 0.3) & (friction <= 1.2))
  assert len(torch.unique(friction)) >= 2


def test_scale_operation(device):
  """Values = default * sample, within expected bounds."""
  torch.manual_seed(42)
  env = create_test_env(device)
  robot = env.scene["robot"]
  geom_idx = robot.indexing.geom_ids[0]

  default_val = env.sim.get_default_field("geom_friction")[geom_idx, 0].item()

  dr.geom_friction(
    env,
    env_ids=None,
    ranges=(0.5, 2.0),
    operation="scale",
    asset_cfg=SceneEntityCfg("robot", geom_ids=[0]),
    axes=[0],
  )

  result = env.sim.model.geom_friction[:, geom_idx, 0]
  assert torch.all(
    (result >= default_val * 0.5 - 1e-5) & (result <= default_val * 2.0 + 1e-5)
  )


def test_add_operation(device):
  """Values = default + sample, within expected bounds."""
  torch.manual_seed(42)
  env = create_test_env(device)
  robot = env.scene["robot"]
  geom_idx = robot.indexing.geom_ids[0]

  default_val = env.sim.get_default_field("geom_friction")[geom_idx, 0].item()

  dr.geom_friction(
    env,
    env_ids=None,
    ranges=(-0.1, 0.1),
    operation="add",
    asset_cfg=SceneEntityCfg("robot", geom_ids=[0]),
    axes=[0],
  )

  result = env.sim.model.geom_friction[:, geom_idx, 0]
  assert torch.all(
    (result >= default_val - 0.1 - 1e-5) & (result <= default_val + 0.1 + 1e-5)
  )


# No accumulation (scale and add share the same mechanism).


@pytest.mark.parametrize(
  "operation, fixed_val, expected_fn",
  [
    ("scale", (2.0, 2.0), lambda d: d * 2.0),
    ("add", (0.1, 0.1), lambda d: d + 0.1),
  ],
)
def test_no_accumulation(device, operation, fixed_val, expected_fn):
  """3x with fixed value produces same result as 1x (uses defaults)."""
  env = create_test_env(device, num_envs=2)
  robot = env.scene["robot"]
  geom_idx = robot.indexing.geom_ids[0]

  default_friction = env.sim.get_default_field("geom_friction")[geom_idx, 0].item()

  for _ in range(3):
    dr.geom_friction(
      env,
      env_ids=None,
      ranges=fixed_val,
      operation=operation,
      asset_cfg=SceneEntityCfg("robot", geom_ids=[0]),
      axes=[0],
    )

  final = env.sim.model.geom_friction[0, geom_idx, 0].item()
  assert abs(final - expected_fn(default_friction)) < 1e-5


# Distributions.


def test_log_uniform_distribution(device):
  """Log-uniform produces values in [lo, hi], skewed toward lower."""
  torch.manual_seed(42)
  env = create_test_env(device, num_envs=64)

  dr.geom_friction(
    env,
    env_ids=None,
    ranges=(0.1, 10.0),
    operation="abs",
    distribution="log_uniform",
    asset_cfg=SceneEntityCfg("robot", geom_ids=[0]),
    axes=[0],
  )

  vals = env.sim.model.geom_friction[:, env.scene["robot"].indexing.geom_ids[0], 0]
  assert torch.all((vals >= 0.1 - 1e-5) & (vals <= 10.0 + 1e-5))
  # Geometric mean of log-uniform(0.1, 10) is 1.0, well below (0.1+10)/2=5.05.
  assert vals.median() < (0.1 + 10.0) / 2


def test_gaussian_distribution(device):
  """Gaussian: mean/std interpretation, values cluster around mean."""
  torch.manual_seed(42)
  env = create_test_env(device, num_envs=128)

  mean, std = 0.5, 0.05
  dr.geom_friction(
    env,
    env_ids=None,
    ranges=(mean, std),
    operation="abs",
    distribution="gaussian",
    asset_cfg=SceneEntityCfg("robot", geom_ids=[0]),
    axes=[0],
  )

  vals = env.sim.model.geom_friction[:, env.scene["robot"].indexing.geom_ids[0], 0]
  assert abs(vals.mean().item() - mean) < 0.05


# Axes.


@pytest.mark.parametrize(
  "axes, changed, unchanged",
  [
    (None, [0], [1, 2]),  # Default axes for geom_friction = [0]
    ([1, 2], [1, 2], [0]),  # Explicit axes
  ],
)
def test_axes_selectivity(device, axes, changed, unchanged):
  """Only the specified axes are modified; others stay at defaults."""
  env = create_test_env(device, num_envs=2)
  robot = env.scene["robot"]
  geom_idx = robot.indexing.geom_ids[0]

  default_friction = env.sim.get_default_field("geom_friction")[geom_idx].clone()

  dr.geom_friction(
    env,
    env_ids=None,
    ranges=(2.0, 2.0),
    operation="scale",
    asset_cfg=SceneEntityCfg("robot", geom_ids=[0]),
    axes=axes,
  )

  final = env.sim.model.geom_friction[0, geom_idx]
  for ax in changed:
    assert abs(final[ax] - default_friction[ax] * 2.0) < 1e-5
  for ax in unchanged:
    assert abs(final[ax] - default_friction[ax]) < 1e-5


def test_dict_int_axis_ranges(device):
  """{0: (lo, hi), 1: (lo2, hi2)} per-axis ranges."""
  torch.manual_seed(42)
  env = create_test_env(device, num_envs=2)
  robot = env.scene["robot"]
  geom_idx = robot.indexing.geom_ids[0]

  dr.geom_friction(
    env,
    env_ids=None,
    ranges={0: (0.3, 0.4), 1: (0.001, 0.002)},
    operation="abs",
    asset_cfg=SceneEntityCfg("robot", geom_ids=[0]),
  )

  final = env.sim.model.geom_friction[0, geom_idx]
  assert 0.3 - 1e-5 <= final[0].item() <= 0.4 + 1e-5
  assert 0.001 - 1e-5 <= final[1].item() <= 0.002 + 1e-5


def test_invalid_axes_raises(device):
  """axes outside valid_axes raises ValueError."""
  env = create_test_env(device)

  with pytest.raises(ValueError, match="Invalid axes"):
    dr.geom_friction(
      env,
      env_ids=None,
      ranges=(0.3, 1.2),
      operation="abs",
      asset_cfg=SceneEntityCfg("robot", geom_names=(".*",)),
      axes=[3],  # geom_friction valid_axes=[0,1,2]
    )


# String-keyed ranges.


def test_string_keyed_ranges(device):
  """Per-component DR with regex patterns."""
  torch.manual_seed(42)
  env = create_test_env(device)

  dr.joint_damping(
    env,
    env_ids=None,
    ranges={".*joint1": (0.5, 0.5), ".*joint2": (1.5, 1.5)},
    operation="abs",
    asset_cfg=SceneEntityCfg("robot", joint_names=(".*",)),
  )

  robot = env.scene["robot"]
  dof_adr = robot.indexing.joint_v_adr
  damping = env.sim.model.dof_damping[0, dof_adr]
  assert abs(damping[0].item() - 0.5) < 1e-5
  assert abs(damping[1].item() - 1.5) < 1e-5


def test_string_keyed_ranges_no_match_raises(device):
  """Pattern matching no names raises ValueError."""
  env = create_test_env(device)

  with pytest.raises(ValueError, match="matched no"):
    dr.joint_damping(
      env,
      env_ids=None,
      ranges={"nonexistent_joint": (0.5, 1.5)},
      operation="abs",
      asset_cfg=SceneEntityCfg("robot", joint_names=(".*",)),
    )


# Shared random.


def test_shared_random(device):
  """All entities within same env get same value, envs differ."""
  torch.manual_seed(42)
  env = create_test_env(device, num_envs=4)
  robot = env.scene["robot"]
  geom_ids = robot.indexing.geom_ids

  dr.geom_friction(
    env,
    env_ids=None,
    ranges=(0.3, 1.2),
    operation="abs",
    asset_cfg=SceneEntityCfg("robot", geom_names=(".*",)),
    axes=[0],
    shared_random=True,
  )

  friction = env.sim.model.geom_friction[:, geom_ids, 0]

  for env_idx in range(env.num_envs):
    env_friction = friction[env_idx]
    assert torch.allclose(env_friction, env_friction[0].expand_as(env_friction))

  env_frictions = friction[:, 0]
  assert len(torch.unique(env_frictions)) > 1
  assert torch.all((friction >= 0.3) & (friction <= 1.2))


# Edge cases.


def test_single_env_without_expand(device):
  """num_envs=1 works without expand_model_fields, no accumulation."""
  env = create_test_env(device, num_envs=1, expand_fields=())
  robot = env.scene["robot"]
  geom_idx = robot.indexing.geom_ids[0]

  original_friction = env.sim.model.geom_friction[0, geom_idx, 0].item()

  for _ in range(2):
    dr.geom_friction(
      env,
      env_ids=None,
      ranges=(2.0, 2.0),
      operation="scale",
      asset_cfg=SceneEntityCfg("robot", geom_ids=[0]),
      axes=[0],
    )

  final = env.sim.model.geom_friction[0, geom_idx, 0].item()
  assert abs(final - original_friction * 2.0) < 1e-5


def test_partial_env_ids(device):
  """Randomizing subset of envs leaves others unchanged."""
  torch.manual_seed(42)
  env = create_test_env(device, num_envs=4)
  robot = env.scene["robot"]
  geom_idx = robot.indexing.geom_ids[0]

  original = env.sim.model.geom_friction[:, geom_idx, 0].clone()

  dr.geom_friction(
    env,
    env_ids=torch.tensor([0, 2], device=env.device),
    ranges=(0.1, 0.2),
    operation="abs",
    asset_cfg=SceneEntityCfg("robot", geom_ids=[0]),
    axes=[0],
  )

  result = env.sim.model.geom_friction[:, geom_idx, 0]
  # Envs 0,2 should have changed.
  assert torch.all((result[0] >= 0.1) & (result[0] <= 0.2))
  assert torch.all((result[2] >= 0.1) & (result[2] <= 0.2))
  # Envs 1,3 should be unchanged.
  assert result[1] == original[1]
  assert result[3] == original[3]


# CUDA graph.


@pytest.mark.skipif(
  not torch.cuda.is_available(), reason="CUDA required for graph capture"
)
def test_expand_model_fields_recreates_cuda_graph(device):
  """Verify CUDA graph is recreated after expand_model_fields."""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(ROBOT_XML))
  scene_cfg = SceneCfg(num_envs=NUM_ENVS, entities={"robot": entity_cfg})
  scene = Scene(scene_cfg, device)
  model = scene.compile()

  sim = Simulation(num_envs=NUM_ENVS, cfg=SimulationCfg(), model=model, device=device)
  scene.initialize(model, sim.model, sim.data)

  if not sim.use_cuda_graph:
    pytest.skip("CUDA graph capture not enabled on this device")

  original_step_graph = sim.step_graph

  sim.expand_model_fields(("geom_friction",))

  assert sim.step_graph is not original_step_graph, (
    "CUDA graph was not recreated after expand_model_fields"
  )


# Integration test.


@pytest.mark.slow
def test_g1_foot_friction_shared_across_geoms(device):
  """G1 velocity env has uniform foot friction across all collision geoms."""
  import io
  import warnings
  from contextlib import redirect_stderr, redirect_stdout

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

  cfg = unitree_g1_flat_env_cfg()

  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
      env = ManagerBasedRlEnv(cfg, device=device)

  try:
    robot = env.scene["robot"]

    foot_geom_names = [
      f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
    ]
    foot_geom_ids, _ = robot.find_geoms(foot_geom_names)
    foot_geom_indices = robot.indexing.geom_ids[foot_geom_ids]

    friction = env.sim.model.geom_friction[:, foot_geom_indices, 0]

    for env_idx in range(env.num_envs):
      env_friction = friction[env_idx]
      assert torch.allclose(env_friction, env_friction[0].expand_as(env_friction))

    if env.num_envs > 1:
      env_frictions = friction[:, 0]
      assert len(torch.unique(env_frictions)) > 1
  finally:
    env.close()


# Quaternion DR tests.


def _make_quat_env(device, num_envs=NUM_ENVS):
  """Create an env with body_quat, geom_quat, site_quat expanded."""
  return create_test_env(
    device,
    num_envs=num_envs,
    expand_fields=("body_quat", "geom_quat", "site_quat"),
  )


_QUAT_CASES = [
  pytest.param(
    dr.body_quat,
    "body_quat",
    SceneEntityCfg("robot", body_names=(".*",)),
    "body_ids",
    id="body",
  ),
  pytest.param(
    dr.geom_quat,
    "geom_quat",
    SceneEntityCfg("robot", geom_names=(".*",)),
    "geom_ids",
    id="geom",
  ),
  pytest.param(
    dr.site_quat,
    "site_quat",
    SceneEntityCfg("robot", site_names=(".*",)),
    "site_ids",
    id="site",
  ),
]


@pytest.mark.parametrize("dr_func, field, asset_cfg, ids_attr", _QUAT_CASES)
def test_quat_is_unit_quaternion(device, dr_func, field, asset_cfg, ids_attr):
  """All output quaternions must have unit norm (within 1e-6)."""
  torch.manual_seed(0)
  env = _make_quat_env(device)
  robot = env.scene["robot"]

  dr_func(
    env,
    env_ids=None,
    roll_range=(-0.5, 0.5),
    pitch_range=(-0.5, 0.5),
    yaw_range=(-math.pi, math.pi),
    asset_cfg=asset_cfg,
  )

  ids = getattr(robot.indexing, ids_attr)
  quats = getattr(env.sim.model, field)[:, ids, :]
  norms = quats.norm(dim=-1)
  assert torch.all((norms - 1.0).abs() < 1e-6), (
    f"Non-unit quaternion found; max deviation: {(norms - 1.0).abs().max()}"
  )


@pytest.mark.parametrize("dr_func, field, asset_cfg, ids_attr", _QUAT_CASES)
def test_quat_zero_range_unchanged(device, dr_func, field, asset_cfg, ids_attr):
  """All ranges (0, 0): result equals default quaternion."""
  env = _make_quat_env(device)
  robot = env.scene["robot"]
  ids = getattr(robot.indexing, ids_attr)
  default_quat = env.sim.get_default_field(field)[ids].clone()

  dr_func(env, env_ids=None, asset_cfg=asset_cfg)

  result = getattr(env.sim.model, field)[:, ids, :]
  assert torch.allclose(result, default_quat.unsqueeze(0).expand_as(result), atol=1e-6)


@pytest.mark.parametrize("dr_func, field, asset_cfg, ids_attr", _QUAT_CASES)
def test_quat_composes_with_default(device, dr_func, field, asset_cfg, ids_attr):
  """Fixed yaw=pi/4 matches manual quat_mul(q_yaw, q_default)."""
  env = _make_quat_env(device)
  robot = env.scene["robot"]
  ids = getattr(robot.indexing, ids_attr)
  n = len(ids)

  yaw = math.pi / 4
  dr_func(env, env_ids=None, yaw_range=(yaw, yaw), asset_cfg=asset_cfg)

  result = getattr(env.sim.model, field)[:, ids, :]
  q_default = env.sim.get_default_field(field)[ids]
  zeros = torch.zeros(n, device=device)
  q_yaw = quat_from_euler_xyz(zeros, zeros, zeros + yaw)
  q_expected = quat_mul(q_yaw, q_default)
  assert torch.allclose(result, q_expected.unsqueeze(0).expand_as(result), atol=1e-6)


def test_body_quat_only_specified_axes(device):
  """Yaw-only perturbation: roll/pitch of result match default."""
  torch.manual_seed(1)
  env = _make_quat_env(device)
  robot = env.scene["robot"]
  body_cfg = SceneEntityCfg("robot", body_names=("base",))
  body_cfg.resolve(env.scene)
  body_ids = robot.indexing.body_ids[body_cfg.body_ids]

  dr.body_quat(
    env,
    env_ids=None,
    yaw_range=(-0.3, 0.3),
    asset_cfg=SceneEntityCfg("robot", body_names=("base",)),
  )

  result = env.sim.model.body_quat[:, body_ids, :]
  # For the default quat [1,0,0,0] composed with a yaw-only rotation,
  # qx and qy should remain 0 (pure yaw -> no roll/pitch).
  q_default = env.sim.get_default_field("body_quat")[body_ids]
  if torch.allclose(q_default, torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)):
    assert torch.allclose(result[..., 1], torch.zeros_like(result[..., 1]), atol=1e-6)
    assert torch.allclose(result[..., 2], torch.zeros_like(result[..., 2]), atol=1e-6)


# pseudo_inertia tests.


def _make_inertia_env(device, num_envs=NUM_ENVS):
  """Create an env with all inertia-related fields expanded."""
  return create_test_env(
    device,
    num_envs=num_envs,
    expand_fields=("body_mass", "body_ipos", "body_inertia", "body_iquat"),
  )


def _matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
  """Convert wxyz quaternion(s) to 3×3 rotation matrix. Shape: (..., 4) → (..., 3, 3)."""
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  return torch.stack(
    [
      torch.stack(
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1
      ),
      torch.stack(
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=-1
      ),
      torch.stack(
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1
      ),
    ],
    dim=-2,
  )


def _reconstruct_J(mass, ipos, inertia, iquat):
  """Reconstruct pseudo-inertia matrix from MuJoCo fields (test helper).

  Correctly accounts for body_iquat (the principal-frame rotation) and the
  parallel-axis theorem (body_ipos) so that the reconstructed J is exact
  regardless of shear magnitude.
  """
  I3 = torch.eye(3, device=mass.device, dtype=mass.dtype)
  # MuJoCo body_iquat maps principal→body.
  R = _matrix_from_quat(iquat)  # (..., 3, 3)
  # Inertia tensor at COM in body frame: I_com = R @ diag(inertia) @ R^T
  I_com = R @ torch.diag_embed(inertia) @ R.mT  # (..., 3, 3)

  # Parallel-axis theorem: shift from COM to body origin.
  c = ipos  # (..., 3)
  c_sq = (c * c).sum(dim=-1)  # (...,)
  c_outer = c.unsqueeze(-1) * c.unsqueeze(-2)  # (..., 3, 3)
  m = mass.unsqueeze(-1).unsqueeze(-1)  # (..., 1, 1)
  I_origin = I_com + m * (c_sq.unsqueeze(-1).unsqueeze(-1) * I3 - c_outer)

  trace = I_origin.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
  sigma = 0.5 * trace.unsqueeze(-1).unsqueeze(-1) * I3 - I_origin
  h = mass.unsqueeze(-1) * ipos
  batch = mass.shape
  J = torch.zeros(*batch, 4, 4, device=mass.device, dtype=mass.dtype)
  J[..., :3, :3] = sigma
  J[..., :3, 3] = h
  J[..., 3, :3] = h
  J[..., 3, 3] = mass
  return J


def test_pseudo_inertia_physical_consistency(device):
  """After wide perturbation, J ≻ 0, mass > 0, triangle inequality holds."""
  torch.manual_seed(10)
  env = _make_inertia_env(device)
  robot = env.scene["robot"]
  body_cfg = SceneEntityCfg("robot", body_names=("base", "foot1", "foot2"))
  body_cfg.resolve(env.scene)

  dr.pseudo_inertia(
    env,
    env_ids=None,
    alpha_range=(-0.5, 0.5),
    d_range=(-0.3, 0.3),
    s12_range=(-0.1, 0.1),
    s13_range=(-0.1, 0.1),
    s23_range=(-0.1, 0.1),
    t_range=(-0.05, 0.05),
    asset_cfg=body_cfg,
  )

  body_ids = robot.indexing.body_ids[body_cfg.body_ids]
  mass = env.sim.model.body_mass[:, body_ids]
  ipos = env.sim.model.body_ipos[:, body_ids, :]
  inertia = env.sim.model.body_inertia[:, body_ids, :]
  iquat = env.sim.model.body_iquat[:, body_ids, :]

  # Mass must be positive.
  assert torch.all(mass > 0), f"Non-positive mass: {mass.min()}"

  # Principal moments must all be positive.
  assert torch.all(inertia > 0), f"Non-positive principal moment: {inertia.min()}"

  # body_iquat must be a unit quaternion.
  iquat_norms = iquat.norm(dim=-1)
  assert torch.all((iquat_norms - 1.0).abs() < 1e-5), (
    f"Non-unit body_iquat; max deviation: {(iquat_norms - 1.0).abs().max()}"
  )

  # Triangle inequality on principal moments: D_i + D_j >= D_k.
  d1, d2, d3 = inertia[..., 0], inertia[..., 1], inertia[..., 2]
  assert torch.all(d1 + d2 >= d3 - 1e-6)
  assert torch.all(d1 + d3 >= d2 - 1e-6)
  assert torch.all(d2 + d3 >= d1 - 1e-6)

  # Round-trip check: reconstruct J from all four stored fields (including
  # body_iquat) and verify it is positive definite. This is exact because
  # the implementation writes the eigendecomposition of J'.
  J = _reconstruct_J(mass, ipos, inertia, iquat)
  eigvals = torch.linalg.eigvalsh(J)
  assert torch.all(eigvals > -1e-5), f"Non-positive J eigenvalue: {eigvals.min()}"


def test_pseudo_inertia_zero_perturbation_unchanged(device):
  """All ranges (0, 0): mass, ipos, inertia, iquat equal defaults (1e-5)."""
  env = _make_inertia_env(device)
  robot = env.scene["robot"]
  body_cfg = SceneEntityCfg("robot", body_names=("base", "foot1", "foot2"))
  body_cfg.resolve(env.scene)
  body_ids = robot.indexing.body_ids[body_cfg.body_ids]

  def_mass = env.sim.get_default_field("body_mass")[body_ids].clone()
  def_ipos = env.sim.get_default_field("body_ipos")[body_ids].clone()
  def_inertia = env.sim.get_default_field("body_inertia")[body_ids].clone()
  def_iquat = env.sim.get_default_field("body_iquat")[body_ids].clone()

  dr.pseudo_inertia(env, env_ids=None, asset_cfg=body_cfg)

  mass = env.sim.model.body_mass[:, body_ids]
  ipos = env.sim.model.body_ipos[:, body_ids, :]
  inertia = env.sim.model.body_inertia[:, body_ids, :]
  iquat = env.sim.model.body_iquat[:, body_ids, :]

  assert torch.allclose(mass, def_mass.unsqueeze(0).expand_as(mass), atol=1e-5)
  assert torch.allclose(ipos, def_ipos.unsqueeze(0).expand_as(ipos), atol=1e-5)
  assert torch.allclose(inertia, def_inertia.unsqueeze(0).expand_as(inertia), atol=1e-5)
  # q and -q represent the same rotation; check both signs.
  def_exp = def_iquat.unsqueeze(0).expand_as(iquat)
  assert torch.allclose(iquat, def_exp, atol=1e-5) or torch.allclose(
    iquat, -def_exp, atol=1e-5
  ), "body_iquat changed unexpectedly under zero perturbation"


def test_pseudo_inertia_zero_perturbation_offset_body(device):
  """Zero perturbation preserves physics for a body with non-trivial ipos/iquat.

  Since ``eigh`` returns eigenvalues in ascending order while MuJoCo may
  store them differently, we compare the reconstructed pseudo-inertia
  matrix J (which is representation-independent) rather than the raw
  principal moments and iquat individually.
  """
  env = _make_inertia_env(device)
  robot = env.scene["robot"]
  body_cfg = SceneEntityCfg("robot", body_names=("offset_body",))
  body_cfg.resolve(env.scene)
  body_ids = robot.indexing.body_ids[body_cfg.body_ids]

  def_mass = env.sim.get_default_field("body_mass")[body_ids].clone()
  def_ipos = env.sim.get_default_field("body_ipos")[body_ids].clone()
  def_inertia = env.sim.get_default_field("body_inertia")[body_ids].clone()
  def_iquat = env.sim.get_default_field("body_iquat")[body_ids].clone()

  # Verify the test body actually has non-trivial ipos and iquat.
  assert not torch.allclose(def_ipos, torch.zeros_like(def_ipos), atol=1e-6), (
    "Test body should have non-zero ipos"
  )
  identity_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
  assert not (
    torch.allclose(def_iquat, identity_q.expand_as(def_iquat), atol=1e-6)
    or torch.allclose(def_iquat, -identity_q.expand_as(def_iquat), atol=1e-6)
  ), "Test body should have non-identity iquat"

  dr.pseudo_inertia(env, env_ids=None, asset_cfg=body_cfg)

  mass = env.sim.model.body_mass[:, body_ids]
  ipos = env.sim.model.body_ipos[:, body_ids, :]
  inertia = env.sim.model.body_inertia[:, body_ids, :]
  iquat = env.sim.model.body_iquat[:, body_ids, :]

  # Mass and ipos must be exactly preserved.
  assert torch.allclose(mass, def_mass.unsqueeze(0).expand_as(mass), atol=1e-5)
  assert torch.allclose(ipos, def_ipos.unsqueeze(0).expand_as(ipos), atol=1e-5)

  # Inertia and iquat may have different eigenvalue ordering, so compare
  # the reconstructed J matrix instead.
  J_default = _reconstruct_J(def_mass, def_ipos, def_inertia, def_iquat)
  J_result = _reconstruct_J(mass, ipos, inertia, iquat)
  assert torch.allclose(
    J_result, J_default.unsqueeze(0).expand_as(J_result), atol=1e-5
  ), (
    f"Reconstructed J differs under zero perturbation.\n"
    f"Max deviation: {(J_result - J_default.unsqueeze(0)).abs().max()}"
  )


def test_pseudo_inertia_alpha_only_scales_mass(device):
  """alpha-only: mass and inertia scale by e^{2α}, ipos unchanged."""
  env = _make_inertia_env(device)
  robot = env.scene["robot"]
  body_cfg = SceneEntityCfg("robot", body_names=("base", "foot1", "foot2"))
  body_cfg.resolve(env.scene)
  body_ids = robot.indexing.body_ids[body_cfg.body_ids]

  a = 0.3
  scale = math.exp(2 * a)

  def_mass = env.sim.get_default_field("body_mass")[body_ids].clone()
  def_ipos = env.sim.get_default_field("body_ipos")[body_ids].clone()
  def_inertia = env.sim.get_default_field("body_inertia")[body_ids].clone()

  dr.pseudo_inertia(env, env_ids=None, alpha_range=(a, a), asset_cfg=body_cfg)

  mass = env.sim.model.body_mass[:, body_ids]
  ipos = env.sim.model.body_ipos[:, body_ids, :]
  inertia = env.sim.model.body_inertia[:, body_ids, :]

  expected_mass = def_mass * scale
  expected_inertia = def_inertia * scale

  assert torch.allclose(mass, expected_mass.unsqueeze(0).expand_as(mass), atol=1e-5)
  assert torch.allclose(
    inertia, expected_inertia.unsqueeze(0).expand_as(inertia), atol=1e-5
  )
  assert torch.allclose(ipos, def_ipos.unsqueeze(0).expand_as(ipos), atol=1e-5)


def test_pseudo_inertia_t1_shifts_com(device):
  """t1-only: mass unchanged, ipos[0] shifts by exactly t1."""
  env = _make_inertia_env(device)
  robot = env.scene["robot"]
  body_cfg = SceneEntityCfg("robot", body_names=("base", "foot1", "foot2"))
  body_cfg.resolve(env.scene)
  body_ids = robot.indexing.body_ids[body_cfg.body_ids]

  t = 0.05
  def_mass = env.sim.get_default_field("body_mass")[body_ids].clone()
  def_ipos = env.sim.get_default_field("body_ipos")[body_ids].clone()

  dr.pseudo_inertia(env, env_ids=None, t1_range=(t, t), asset_cfg=body_cfg)

  mass = env.sim.model.body_mass[:, body_ids]
  ipos = env.sim.model.body_ipos[:, body_ids, :]

  assert torch.allclose(mass, def_mass.unsqueeze(0).expand_as(mass), atol=1e-5)
  # ipos x-component shifts by t; y and z are unchanged.
  assert torch.allclose(
    ipos[..., 0], (def_ipos[..., 0] + t).unsqueeze(0).expand_as(ipos[..., 0]), atol=1e-5
  )
  assert torch.allclose(
    ipos[..., 1:], def_ipos[..., 1:].unsqueeze(0).expand_as(ipos[..., 1:]), atol=1e-5
  )


def test_pseudo_inertia_d_isotropic_vs_per_axis(device):
  """d_range=(r,r) gives same result as d1=d2=d3=(r,r)."""
  r = 0.2
  seed = 42

  torch.manual_seed(seed)
  env1 = _make_inertia_env(device)
  body_cfg = SceneEntityCfg("robot", body_names=("base", "foot1", "foot2"))
  body_cfg.resolve(env1.scene)
  dr.pseudo_inertia(env1, env_ids=None, d_range=(r, r), asset_cfg=body_cfg)

  torch.manual_seed(seed)
  env2 = _make_inertia_env(device)
  dr.pseudo_inertia(
    env2,
    env_ids=None,
    d1_range=(r, r),
    d2_range=(r, r),
    d3_range=(r, r),
    asset_cfg=body_cfg,
  )

  robot1 = env1.scene["robot"]
  body_ids = robot1.indexing.body_ids[body_cfg.body_ids]

  assert torch.allclose(
    env1.sim.model.body_mass[:, body_ids],
    env2.sim.model.body_mass[:, body_ids],
    atol=1e-6,
  )
  assert torch.allclose(
    env1.sim.model.body_inertia[:, body_ids, :],
    env2.sim.model.body_inertia[:, body_ids, :],
    atol=1e-6,
  )


def test_pseudo_inertia_no_accumulation(device):
  """3x with alpha=0.1 gives same result as 1x (uses defaults each time)."""
  env = _make_inertia_env(device)
  robot = env.scene["robot"]
  body_cfg = SceneEntityCfg("robot", body_names=("base",))
  body_cfg.resolve(env.scene)
  body_ids = robot.indexing.body_ids[body_cfg.body_ids]

  for _ in range(3):
    dr.pseudo_inertia(env, env_ids=None, alpha_range=(0.1, 0.1), asset_cfg=body_cfg)

  mass_3x = env.sim.model.body_mass[:, body_ids].clone()

  env2 = _make_inertia_env(device)
  dr.pseudo_inertia(env2, env_ids=None, alpha_range=(0.1, 0.1), asset_cfg=body_cfg)
  mass_1x = env2.sim.model.body_mass[:, body_ids]

  assert torch.allclose(mass_3x, mass_1x, atol=1e-5)


def test_pseudo_inertia_partial_env_ids(device):
  """Randomizing only env 0 leaves env 1 unchanged."""
  env = _make_inertia_env(device, num_envs=2)
  robot = env.scene["robot"]
  body_cfg = SceneEntityCfg("robot", body_names=("base",))
  body_cfg.resolve(env.scene)
  body_ids = robot.indexing.body_ids[body_cfg.body_ids]

  mass_before = env.sim.model.body_mass[:, body_ids].clone()

  dr.pseudo_inertia(
    env,
    env_ids=torch.tensor([0], device=device),
    alpha_range=(0.5, 0.5),
    asset_cfg=body_cfg,
  )

  mass_after = env.sim.model.body_mass[:, body_ids]
  # Env 0 changed.
  assert not torch.allclose(mass_after[0], mass_before[0], atol=1e-6)
  # Env 1 unchanged.
  assert torch.allclose(mass_after[1], mass_before[1], atol=1e-6)


# Camera / Light DR tests.


def _make_cam_light_env(device, num_envs=NUM_ENVS):
  """Create an env with camera and light fields expanded."""
  return create_test_env(
    device,
    num_envs=num_envs,
    expand_fields=(
      "cam_fovy",
      "cam_pos",
      "cam_quat",
      "cam_intrinsic",
      "light_pos",
      "light_dir",
    ),
  )


def test_cam_fovy_abs(device):
  """Set fovy to absolute range, check bounds."""
  torch.manual_seed(42)
  env = _make_cam_light_env(device)
  robot = env.scene["robot"]
  cam_cfg = SceneEntityCfg("robot", camera_names=(".*",))
  cam_cfg.resolve(env.scene)
  cam_ids = robot.indexing.cam_ids[cam_cfg.camera_ids]

  dr.cam_fovy(
    env,
    env_ids=None,
    ranges=(30.0, 90.0),
    operation="abs",
    asset_cfg=cam_cfg,
  )

  fovy = env.sim.model.cam_fovy[:, cam_ids]
  assert torch.all((fovy >= 30.0 - 1e-3) & (fovy <= 90.0 + 1e-3))


def test_cam_pos_add(device):
  """Add offset to camera positions, check default + offset."""
  torch.manual_seed(42)
  env = _make_cam_light_env(device)
  robot = env.scene["robot"]
  cam_cfg = SceneEntityCfg("robot", camera_names=("front_cam",))
  cam_cfg.resolve(env.scene)
  cam_ids = robot.indexing.cam_ids[cam_cfg.camera_ids]

  default_pos = env.sim.get_default_field("cam_pos")[cam_ids].clone()

  dr.cam_pos(
    env,
    env_ids=None,
    ranges=(-0.1, 0.1),
    operation="add",
    asset_cfg=cam_cfg,
  )

  result = env.sim.model.cam_pos[:, cam_ids, :]
  for ax in range(3):
    lo = default_pos[..., ax] - 0.1 - 1e-5
    hi = default_pos[..., ax] + 0.1 + 1e-5
    assert torch.all((result[..., ax] >= lo) & (result[..., ax] <= hi))


def test_cam_quat_zero_unchanged(device):
  """Zero RPY range preserves default quaternion."""
  env = _make_cam_light_env(device)
  robot = env.scene["robot"]
  cam_cfg = SceneEntityCfg("robot", camera_names=(".*",))
  cam_cfg.resolve(env.scene)
  cam_ids = robot.indexing.cam_ids[cam_cfg.camera_ids]

  default_quat = env.sim.get_default_field("cam_quat")[cam_ids].clone()

  dr.cam_quat(env, env_ids=None, asset_cfg=cam_cfg)

  result = env.sim.model.cam_quat[:, cam_ids, :]
  assert torch.allclose(
    result,
    default_quat.unsqueeze(0).expand_as(result),
    atol=1e-6,
  )


def test_light_pos_abs(device):
  """Set light position to absolute range, check bounds."""
  torch.manual_seed(42)
  env = _make_cam_light_env(device)
  robot = env.scene["robot"]
  light_cfg = SceneEntityCfg("robot", light_names=(".*",))
  light_cfg.resolve(env.scene)
  light_ids = robot.indexing.light_ids[light_cfg.light_ids]

  dr.light_pos(
    env,
    env_ids=None,
    ranges=(-5.0, 5.0),
    operation="abs",
    asset_cfg=light_cfg,
  )

  pos = env.sim.model.light_pos[:, light_ids, :]
  assert torch.all((pos >= -5.0 - 1e-3) & (pos <= 5.0 + 1e-3))


def test_light_dir_add(device):
  """Add offset to light direction, check default + offset."""
  torch.manual_seed(42)
  env = _make_cam_light_env(device)
  robot = env.scene["robot"]
  light_cfg = SceneEntityCfg("robot", light_names=(".*",))
  light_cfg.resolve(env.scene)
  light_ids = robot.indexing.light_ids[light_cfg.light_ids]

  default_dir = env.sim.get_default_field("light_dir")[light_ids].clone()

  dr.light_dir(
    env,
    env_ids=None,
    ranges=(-0.5, 0.5),
    operation="add",
    asset_cfg=light_cfg,
  )

  result = env.sim.model.light_dir[:, light_ids, :]
  for ax in range(3):
    lo = default_dir[..., ax] - 0.5 - 1e-5
    hi = default_dir[..., ax] + 0.5 + 1e-5
    assert torch.all((result[..., ax] >= lo) & (result[..., ax] <= hi))


def test_camera_partial_env_ids(device):
  """Subset of envs randomized, others unchanged."""
  torch.manual_seed(42)
  env = _make_cam_light_env(device, num_envs=4)
  robot = env.scene["robot"]
  cam_cfg = SceneEntityCfg("robot", camera_names=(".*",))
  cam_cfg.resolve(env.scene)
  cam_ids = robot.indexing.cam_ids[cam_cfg.camera_ids]

  original_fovy = env.sim.model.cam_fovy[:, cam_ids].clone()

  dr.cam_fovy(
    env,
    env_ids=torch.tensor([0, 2], device=env.device),
    ranges=(10.0, 20.0),
    operation="abs",
    asset_cfg=cam_cfg,
  )

  result = env.sim.model.cam_fovy[:, cam_ids]
  # Envs 0, 2 should have changed.
  assert torch.all((result[0] >= 10.0 - 1e-3) & (result[0] <= 20.0 + 1e-3))
  assert torch.all((result[2] >= 10.0 - 1e-3) & (result[2] <= 20.0 + 1e-3))
  # Envs 1, 3 should be unchanged.
  assert torch.allclose(result[1], original_fovy[1])
  assert torch.allclose(result[3], original_fovy[3])


# geom_size DR tests.

GEOM_SIZE_XML = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom name="box_geom" type="box" size="0.1 0.2 0.3" mass="1.0"/>
      <geom name="sphere_geom" type="sphere" size="0.15" mass="0.5"/>
      <geom name="capsule_geom" type="capsule" size="0.05 0.2" mass="0.3"/>
      <geom name="cylinder_geom" type="cylinder" size="0.08 0.15" mass="0.4"/>
      <geom name="ellipsoid_geom" type="ellipsoid" size="0.1 0.2 0.05"
        mass="0.2"/>
    </body>
  </worldbody>
</mujoco>
"""


def _make_geom_size_env(device, num_envs=NUM_ENVS):
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(GEOM_SIZE_XML))
  scene_cfg = SceneCfg(num_envs=num_envs, entities={"robot": entity_cfg})
  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim = Simulation(num_envs=num_envs, cfg=SimulationCfg(), model=model, device=device)
  scene.initialize(model, sim.model, sim.data)
  sim.expand_model_fields(("geom_size", "geom_rbound", "geom_aabb"))
  return Env(scene, sim, device)


def _expected_rbound(geom_type, s0, s1, s2):
  """Reference rbound computation matching MuJoCo GetRBound."""
  if geom_type == "sphere":
    return s0
  elif geom_type == "capsule":
    return s0 + s1
  elif geom_type == "cylinder":
    return math.sqrt(s0**2 + s1**2)
  elif geom_type == "ellipsoid":
    return max(s0, s1, s2)
  elif geom_type == "box":
    return math.sqrt(s0**2 + s1**2 + s2**2)
  raise ValueError(geom_type)


def _expected_aabb_half(geom_type, s0, s1, s2):
  """Reference aabb half-size computation matching MuJoCo ComputeAABB."""
  if geom_type == "sphere":
    return (s0, s0, s0)
  elif geom_type == "capsule":
    return (s0, s0, s0 + s1)
  elif geom_type == "cylinder":
    return (s0, s0, s1)
  elif geom_type == "ellipsoid":
    return (s0, s1, s2)
  elif geom_type == "box":
    return (s0, s1, s2)
  raise ValueError(geom_type)


def test_geom_size_scale_updates_bounds(device):
  """Scaling geom_size updates rbound and aabb consistently."""
  torch.manual_seed(42)
  env = _make_geom_size_env(device)
  robot = env.scene["robot"]
  geom_cfg = SceneEntityCfg("robot", geom_names=(".*",))
  geom_cfg.resolve(env.scene)

  dr.geom_size(
    env,
    env_ids=None,
    ranges=(0.8, 1.5),
    operation="scale",
    asset_cfg=geom_cfg,
  )

  geom_ids = robot.indexing.geom_ids[geom_cfg.geom_ids]
  size = env.sim.model.geom_size[:, geom_ids]  # (E, G, 3)
  rbound = env.sim.model.geom_rbound[:, geom_ids]  # (E, G)
  aabb = env.sim.model.geom_aabb[:, geom_ids]  # (E, G, 2, 3)

  geom_names = list(robot.geom_names)
  type_names = {
    "box_geom": "box",
    "sphere_geom": "sphere",
    "capsule_geom": "capsule",
    "cylinder_geom": "cylinder",
    "ellipsoid_geom": "ellipsoid",
  }

  for g_local, gname in enumerate(geom_names):
    tname = type_names[gname]
    for e in range(env.num_envs):
      s0 = size[e, g_local, 0].item()
      s1 = size[e, g_local, 1].item()
      s2 = size[e, g_local, 2].item()

      expected_rb = _expected_rbound(tname, s0, s1, s2)
      actual_rb = rbound[e, g_local].item()
      assert abs(actual_rb - expected_rb) < 1e-5, (
        f"{gname} env {e}: rbound {actual_rb} != {expected_rb}"
      )

      ex, ey, ez = _expected_aabb_half(tname, s0, s1, s2)
      actual_half = aabb[e, g_local, 1]  # half-size is index 1
      assert abs(actual_half[0].item() - ex) < 1e-5
      assert abs(actual_half[1].item() - ey) < 1e-5
      assert abs(actual_half[2].item() - ez) < 1e-5

      # Center should be zero for all primitives.
      center = aabb[e, g_local, 0]
      assert torch.allclose(center, torch.zeros(3, device=device), atol=1e-6)


def test_geom_size_no_accumulation(device):
  """Repeated scale operations don't accumulate."""
  env = _make_geom_size_env(device, num_envs=2)
  robot = env.scene["robot"]
  geom_cfg = SceneEntityCfg("robot", geom_names=("box_geom",))
  geom_cfg.resolve(env.scene)
  geom_ids = robot.indexing.geom_ids[geom_cfg.geom_ids]

  default_size = env.sim.get_default_field("geom_size")[geom_ids].clone()

  for _ in range(3):
    dr.geom_size(
      env,
      env_ids=None,
      ranges=(2.0, 2.0),
      operation="scale",
      asset_cfg=geom_cfg,
    )

  result = env.sim.model.geom_size[0, geom_ids]
  expected = default_size * 2.0
  assert torch.allclose(result, expected, atol=1e-5)


def test_geom_size_partial_env_ids(device):
  """Only specified envs are updated."""
  env = _make_geom_size_env(device, num_envs=4)
  robot = env.scene["robot"]
  geom_cfg = SceneEntityCfg("robot", geom_names=("sphere_geom",))
  geom_cfg.resolve(env.scene)
  geom_ids = robot.indexing.geom_ids[geom_cfg.geom_ids]

  original_rbound = env.sim.model.geom_rbound[:, geom_ids].clone()

  dr.geom_size(
    env,
    env_ids=torch.tensor([0, 2], device=device),
    ranges=(2.0, 2.0),
    operation="scale",
    asset_cfg=geom_cfg,
  )

  rbound = env.sim.model.geom_rbound[:, geom_ids]
  # Envs 0, 2 should have changed.
  assert not torch.allclose(rbound[0], original_rbound[0])
  assert not torch.allclose(rbound[2], original_rbound[2])
  # Envs 1, 3 should be unchanged.
  assert torch.allclose(rbound[1], original_rbound[1])
  assert torch.allclose(rbound[3], original_rbound[3])


def test_geom_size_raises_on_unsupported_type(device):
  """dr.geom_size raises ValueError when a non-primitive geom type is selected."""
  PLANE_XML = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="base" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom name="box_geom" type="box" size="0.1 0.1 0.1" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(PLANE_XML))
  scene_cfg = SceneCfg(num_envs=2, entities={"robot": entity_cfg})
  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim = Simulation(num_envs=2, cfg=SimulationCfg(), model=model, device=device)
  scene.initialize(model, sim.model, sim.data)
  sim.expand_model_fields(("geom_size", "geom_rbound", "geom_aabb"))
  env = Env(scene, sim, device)

  plane_cfg = SceneEntityCfg("robot", geom_names=("floor",))
  plane_cfg.resolve(env.scene)

  with pytest.raises(ValueError, match="unsupported types"):
    dr.geom_size(env, env_ids=None, ranges=(1.0, 2.0), asset_cfg=plane_cfg)


# Tendon DR tests.

TENDON_XML = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom name="base_geom" type="box" size="0.1 0.1 0.1" mass="1.0"/>
      <site name="anchor" pos="0 0 0.1"/>
      <body name="child" pos="0.3 0 0">
        <joint name="hinge1" type="hinge" axis="0 0 1"/>
        <geom name="child_geom" type="sphere" size="0.05" mass="0.2"/>
        <site name="hook" pos="0 0 -0.05"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <spatial name="wire" limited="true" range="0 0.5" width="0.003"
      damping="2.0" stiffness="10.0" frictionloss="0.5" armature="0.1">
      <site site="anchor"/>
      <site site="hook"/>
    </spatial>
  </tendon>
</mujoco>
"""


def _make_tendon_env(device, num_envs=NUM_ENVS):
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(TENDON_XML))
  scene_cfg = SceneCfg(num_envs=num_envs, entities={"robot": entity_cfg})
  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim = Simulation(num_envs=num_envs, cfg=SimulationCfg(), model=model, device=device)
  scene.initialize(model, sim.model, sim.data)
  sim.expand_model_fields(
    (
      "tendon_damping",
      "tendon_stiffness",
      "tendon_frictionloss",
      "tendon_armature",
    )
  )

  return Env(scene, sim, device)


@pytest.mark.parametrize(
  "dr_func, field",
  [
    (dr.tendon_damping, "tendon_damping"),
    (dr.tendon_stiffness, "tendon_stiffness"),
    (dr.tendon_friction, "tendon_frictionloss"),
    (dr.tendon_armature, "tendon_armature"),
  ],
)
def test_tendon_field_scale(device, dr_func, field):
  """Scale operation on tendon fields: result = default * sample."""
  torch.manual_seed(42)
  env = _make_tendon_env(device)
  robot = env.scene["robot"]
  tendon_cfg = SceneEntityCfg("robot", tendon_names=(".*",))
  tendon_cfg.resolve(env.scene)
  tendon_ids = robot.indexing.tendon_ids[tendon_cfg.tendon_ids]

  default_val = env.sim.get_default_field(field)[tendon_ids].clone()

  dr_func(
    env,
    env_ids=None,
    ranges=(0.5, 2.0),
    operation="scale",
    asset_cfg=tendon_cfg,
  )

  result = getattr(env.sim.model, field)[:, tendon_ids]
  lo = default_val * 0.5 - 1e-5
  hi = default_val * 2.0 + 1e-5
  assert torch.all((result >= lo) & (result <= hi))


def test_tendon_armature_no_accumulation(device):
  """Repeated scale does not accumulate."""
  env = _make_tendon_env(device, num_envs=2)
  robot = env.scene["robot"]
  tendon_cfg = SceneEntityCfg("robot", tendon_names=(".*",))
  tendon_cfg.resolve(env.scene)
  tendon_ids = robot.indexing.tendon_ids[tendon_cfg.tendon_ids]

  default_val = env.sim.get_default_field("tendon_armature")[tendon_ids]

  for _ in range(3):
    dr.tendon_armature(
      env,
      env_ids=None,
      ranges=(2.0, 2.0),
      operation="scale",
      asset_cfg=tendon_cfg,
    )

  result = env.sim.model.tendon_armature[0, tendon_ids]
  assert torch.allclose(result, default_val * 2.0, atol=1e-5)


# Extensible Operation / Distribution types.


def test_operation_instance_abs(device):
  """Passing an Operation instance works identically to the string."""
  from mjlab.envs.mdp.dr._types import abs as abs_op

  torch.manual_seed(42)
  env1 = create_test_env(device)
  robot = env1.scene["robot"]

  dr.geom_friction(
    env1,
    env_ids=None,
    ranges=(0.3, 1.2),
    operation="abs",
    asset_cfg=SceneEntityCfg("robot", geom_names=(".*",)),
    axes=[0],
  )
  result_str = env1.sim.model.geom_friction[:, robot.indexing.geom_ids, 0].clone()

  torch.manual_seed(42)
  env2 = create_test_env(device)

  dr.geom_friction(
    env2,
    env_ids=None,
    ranges=(0.3, 1.2),
    operation=abs_op,
    asset_cfg=SceneEntityCfg("robot", geom_names=(".*",)),
    axes=[0],
  )
  result_inst = env2.sim.model.geom_friction[:, robot.indexing.geom_ids, 0]

  assert torch.allclose(result_str, result_inst)


def test_distribution_instance_uniform(device):
  """Passing a Distribution instance works identically to the string."""
  from mjlab.envs.mdp.dr._types import uniform as uniform_dist

  torch.manual_seed(42)
  env1 = create_test_env(device)
  robot = env1.scene["robot"]

  dr.geom_friction(
    env1,
    env_ids=None,
    ranges=(0.3, 1.2),
    distribution="uniform",
    operation="abs",
    asset_cfg=SceneEntityCfg("robot", geom_names=(".*",)),
    axes=[0],
  )
  result_str = env1.sim.model.geom_friction[:, robot.indexing.geom_ids, 0].clone()

  torch.manual_seed(42)
  env2 = create_test_env(device)

  dr.geom_friction(
    env2,
    env_ids=None,
    ranges=(0.3, 1.2),
    distribution=uniform_dist,
    operation="abs",
    asset_cfg=SceneEntityCfg("robot", geom_names=(".*",)),
    axes=[0],
  )
  result_inst = env2.sim.model.geom_friction[:, robot.indexing.geom_ids, 0]

  assert torch.allclose(result_str, result_inst)


def test_custom_operation(device):
  """A user-defined Operation works end-to-end."""
  from mjlab.envs.mdp.dr._types import Operation

  clamp_op = Operation(
    name="clamp",
    initialize=torch.Tensor.clone,
    combine=lambda base, random: torch.clamp(random, min=0.4, max=0.9),
    uses_defaults=False,
  )

  torch.manual_seed(42)
  env = create_test_env(device)
  robot = env.scene["robot"]

  dr.geom_friction(
    env,
    env_ids=None,
    ranges=(0.1, 1.5),
    operation=clamp_op,
    asset_cfg=SceneEntityCfg("robot", geom_names=(".*",)),
    axes=[0],
  )

  friction = env.sim.model.geom_friction[:, robot.indexing.geom_ids, 0]
  assert torch.all((friction >= 0.4 - 1e-5) & (friction <= 0.9 + 1e-5))


def test_custom_distribution(device):
  """A user-defined Distribution works end-to-end."""
  from mjlab.envs.mdp.dr._types import Distribution

  # Distribution that always returns the midpoint.
  midpoint_dist = Distribution(
    name="midpoint",
    sample=lambda lo, hi, shape, device: ((lo + hi) / 2).expand(shape),
  )

  env = create_test_env(device)
  robot = env.scene["robot"]

  dr.geom_friction(
    env,
    env_ids=None,
    ranges=(0.2, 0.8),
    distribution=midpoint_dist,
    operation="abs",
    asset_cfg=SceneEntityCfg("robot", geom_names=(".*",)),
    axes=[0],
  )

  friction = env.sim.model.geom_friction[:, robot.indexing.geom_ids, 0]
  assert torch.allclose(friction, torch.tensor(0.5, device=device), atol=1e-5)


def test_resolve_unknown_operation_raises(device):
  """Unknown operation string raises ValueError."""
  from mjlab.envs.mdp.dr._types import resolve_operation

  with pytest.raises(ValueError, match="Unknown operation"):
    resolve_operation("nonexistent")


def test_resolve_unknown_distribution_raises(device):
  """Unknown distribution string raises ValueError."""
  from mjlab.envs.mdp.dr._types import resolve_distribution

  with pytest.raises(ValueError, match="Unknown distribution"):
    resolve_distribution("nonexistent")


# mat_rgba tests.

MAT_XML = """
<mujoco>
  <asset>
    <material name="test_mat" rgba="1 1 1 1"/>
  </asset>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="free_joint"/>
      <geom name="base_geom" type="box" size="0.1 0.1 0.1" mass="1.0"
        material="test_mat"/>
    </body>
  </worldbody>
</mujoco>
"""


def _make_mat_env(device, num_envs=NUM_ENVS):
  entity_cfg = EntityCfg(spec_fn=lambda: mujoco.MjSpec.from_string(MAT_XML))
  scene_cfg = SceneCfg(num_envs=num_envs, entities={"robot": entity_cfg})
  scene = Scene(scene_cfg, device)
  model = scene.compile()
  sim = Simulation(num_envs=num_envs, cfg=SimulationCfg(), model=model, device=device)
  scene.initialize(model, sim.model, sim.data)
  sim.expand_model_fields(("mat_rgba",))
  return Env(scene, sim, device)


def test_mat_rgba_abs(device):
  """Values within range, diversity across envs."""
  torch.manual_seed(42)
  env = _make_mat_env(device)

  dr.mat_rgba(
    env,
    env_ids=None,
    ranges=(0.2, 0.8),
    asset_cfg=SceneEntityCfg("robot", material_names=("test_mat",)),
    operation="abs",
  )

  mat_id = mujoco.mj_name2id(
    env.sim.mj_model, mujoco.mjtObj.mjOBJ_MATERIAL, "robot/test_mat"
  )
  rgba = env.sim.model.mat_rgba[:, mat_id, :]
  assert torch.all((rgba >= 0.2 - 1e-5) & (rgba <= 0.8 + 1e-5))
  assert len(torch.unique(rgba[:, 0])) >= 2


def test_mat_rgba_invalid_name(device):
  """ValueError for unknown material name."""
  env = _make_mat_env(device)

  with pytest.raises(ValueError, match="nonexistent_material"):
    cfg = SceneEntityCfg("robot", material_names=("nonexistent_material",))
    cfg.resolve(env.scene)
    dr.mat_rgba(
      env,
      env_ids=None,
      ranges=(0.2, 0.8),
      asset_cfg=cfg,
    )
