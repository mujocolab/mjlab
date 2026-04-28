import colorsys
from typing import Any, Literal

import mujoco
import numpy as np

from mjlab.asset_zoo.robots import (
  YAM_ACTION_SCALE,
  get_yam_robot_cfg,
)
from mjlab.entity import EntityCfg, VariantCfg, VariantEntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import CameraSensorCfg, ContactSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.manipulation.mdp import MultiCubeLiftingCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise


def get_cube_spec(
  cube_size: float = 0.02,
  mass: float = 0.05,
  rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(cube_size,) * 3,
    mass=mass,
    rgba=rgba,
  )
  return spec


_BOX_FACES = np.array(
  [
    [0, 3, 2],
    [0, 1, 3],  # -x
    [4, 7, 5],
    [4, 6, 7],  # +x
    [0, 5, 1],
    [0, 4, 5],  # -y
    [2, 7, 6],
    [2, 3, 7],  # +y
    [0, 6, 4],
    [0, 2, 6],  # -z
    [1, 7, 3],
    [1, 5, 7],  # +z
  ],
  dtype=np.int32,
)
_SPHERE_SUBDIVISION = 2
# Icosphere subdivisions trade accuracy for mesh-collision cost. Numbers below
# are vs. analytic primitives:
#   sub=2 (320 faces): mass within ~3%, inertia within ~10% (sorted).
#   sub=4 (5120 faces): mass within ~0.5%, inertia within ~0.5%.
# At ~4k envs, sub=4 mesh-vs-plane contact dominates physics step time
# (~3x slower than primitive boxes); sub=2 is the training-friendly default.


def _add_box_mesh(
  spec: mujoco.MjSpec,
  name: str,
  half_extents: tuple[float, float, float],
) -> None:
  sx, sy, sz = half_extents
  verts = np.array(
    [[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)],
    dtype=np.float32,
  )
  mesh = spec.add_mesh()
  mesh.name = name
  mesh.uservert = verts.flatten()
  mesh.userface = _BOX_FACES.flatten()


def _make_object_spec(
  build_mesh: Any,
  *,
  mesh_name: str,
  density: float,
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  build_mesh(spec, mesh_name)
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname=mesh_name,
    density=density,
    rgba=(0.8, 0.2, 0.2, 1.0),
  )
  return spec


def make_box_variant_spec(
  half_extents: tuple[float, float, float],
  density: float = 300.0,
) -> mujoco.MjSpec:
  return _make_object_spec(
    lambda spec, name: _add_box_mesh(spec, name, half_extents),
    mesh_name="box",
    density=density,
  )


def make_sphere_variant_spec(
  radius: float,
  density: float = 300.0,
) -> mujoco.MjSpec:
  def _build(spec: mujoco.MjSpec, name: str) -> None:
    m = spec.add_mesh()
    m.name = name
    m.make_sphere(_SPHERE_SUBDIVISION)
    m.scale[:] = (radius, radius, radius)

  return _make_object_spec(_build, mesh_name="sphere", density=density)


def make_ellipsoid_variant_spec(
  semi_axes: tuple[float, float, float],
  density: float = 300.0,
) -> mujoco.MjSpec:
  def _build(spec: mujoco.MjSpec, name: str) -> None:
    m = spec.add_mesh()
    m.name = name
    m.make_sphere(_SPHERE_SUBDIVISION)
    m.scale[:] = semi_axes

  return _make_object_spec(_build, mesh_name="ellipsoid", density=density)


def yam_lift_cube_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_lift_cube_env_cfg()

  cfg.scene.entities = {
    "robot": get_yam_robot_cfg(),
    "cube": EntityCfg(spec_fn=get_cube_spec),
  }

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = YAM_ACTION_SCALE

  cfg.observations["actor"].terms["ee_to_cube"].params["asset_cfg"].site_names = (
    "grasp_site",
  )
  cfg.rewards["lift"].params["asset_cfg"].site_names = ("grasp_site",)

  fingertip_geoms = r"[lr]f_down(6|7|8|9|10|11)_collision"
  cfg.events["fingertip_friction_slide"].params[
    "asset_cfg"
  ].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  # Configure collision sensor pattern.
  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link_6"

  cfg.viewer.body_name = "arm"

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}

    # Higher command resampling frequency for more dynamic play.
    assert cfg.commands is not None
    cfg.commands["lift_height"].resampling_time_range = (4.0, 4.0)

  return cfg


# Default variant set: 2x3 grid of (sharp box / smooth ellipsoid) x
# (isotropic / long-thin / flat-wide). Density 300 kg/m^3 keeps masses
# light enough for YAM (~14 - 65 g across variants).
_BOX_LONG_HALF_EXTENTS = (0.050, 0.015, 0.015)
_BOX_FLAT_HALF_EXTENTS = (0.035, 0.035, 0.012)
# Ellipsoids are slightly thicker on their short axes than the box pair so
# their fingertip contacts are stable (a 1.5 cm ellipsoid radius gives the
# fingers very little to grip).
_PENCIL_HALF_EXTENTS = (0.050, 0.020, 0.020)
_PLATE_HALF_EXTENTS = (0.035, 0.035, 0.018)

LIFT_VARIANT_DEFAULTS: dict[str, VariantCfg] = {
  # --- Sharp (boxes) ---
  "cube": VariantCfg(
    spec_fn=lambda: make_box_variant_spec((0.030, 0.030, 0.030)),
    weight=1.0,
  ),
  "box_long": VariantCfg(
    spec_fn=lambda: make_box_variant_spec(_BOX_LONG_HALF_EXTENTS),
    weight=1.0,
  ),
  "box_flat": VariantCfg(
    spec_fn=lambda: make_box_variant_spec(_BOX_FLAT_HALF_EXTENTS),
    weight=1.0,
  ),
  # --- Smooth (ellipsoids) ---
  "sphere": VariantCfg(
    spec_fn=lambda: make_sphere_variant_spec(0.025),
    weight=1.0,
  ),
  "pencil": VariantCfg(
    spec_fn=lambda: make_ellipsoid_variant_spec(_PENCIL_HALF_EXTENTS),
    weight=1.0,
  ),
  "plate": VariantCfg(
    spec_fn=lambda: make_ellipsoid_variant_spec(_PLATE_HALF_EXTENTS),
    weight=1.0,
  ),
}


def yam_lift_variant_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Lift task with per-world mesh variants: shape, size, color randomized."""
  cfg = yam_lift_cube_env_cfg(play=play)

  cfg.scene.entities["cube"] = VariantEntityCfg(variants=LIFT_VARIANT_DEFAULTS)

  # TEMP: disable actor observation noise to test whether the per-step Unoise
  # on joint_pos / joint_vel / ee_to_cube / cube_to_goal is hurting precision
  # on the anisotropic variants.
  cfg.observations["actor"].enable_corruption = False

  # Anisotropic shapes (pencil, plate) require the policy to know object
  # orientation and extent.
  actor_terms = cfg.observations["actor"].terms
  actor_terms["object_pose"] = ObservationTermCfg(
    func=manipulation_mdp.object_pose_in_base,
    params={"object_name": "cube"},
    noise=Unoise(n_min=-0.01, n_max=0.01),
  )
  actor_terms["object_extents"] = ObservationTermCfg(
    func=manipulation_mdp.object_sorted_extents,
    params={"object_name": "cube"},
  )
  cfg.observations["critic"].terms["object_pose"] = actor_terms["object_pose"]
  cfg.observations["critic"].terms["object_extents"] = actor_terms["object_extents"]

  # Spheres and ellipsoids can roll out of reach; terminate before they escape.
  cfg.terminations["object_out_of_reach"] = TerminationTermCfg(
    func=manipulation_mdp.object_out_of_reach,
    params={
      "object_name": "cube",
      "asset_cfg": SceneEntityCfg("robot"),
      "threshold": 0.7,
    },
  )

  # Catch pre-NaN solver divergence: penetration impulses launch the object
  # with extreme spin (hundreds of rad/s) before qpos/qvel go NaN. Normal
  # tumbles stay well under 20 rad/s.
  cfg.terminations["object_spinning"] = TerminationTermCfg(
    func=manipulation_mdp.object_spinning_too_fast,
    params={"object_name": "cube", "threshold": 30.0},
  )

  return cfg


def yam_lift_cube_vision_env_cfg(
  cam_type: Literal["rgb", "depth"],
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = yam_lift_cube_env_cfg(play=play)

  camera_names = ["robot/camera_d405"]
  cam_kwargs = {
    "robot/camera_d405": {
      "height": 32,
      "width": 32,
    },
  }
  shared_cam_kwargs = dict(
    data_types=(cam_type,),
    enabled_geom_groups=(0, 3),
    use_shadows=False,
    use_textures=True,
  )

  cam_terms = {}
  for cam_name in camera_names:
    cam_cfg = CameraSensorCfg(
      name=cam_name.split("/")[-1],
      camera_name=cam_name,
      **cam_kwargs[cam_name],  # type: ignore[invalid-argument-type]
      **shared_cam_kwargs,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)
    param_kwargs: dict[str, Any] = {"sensor_name": cam_cfg.name}
    if cam_type == "depth":
      param_kwargs["cutoff_distance"] = 0.5
      func = manipulation_mdp.camera_depth
    else:
      func = manipulation_mdp.camera_rgb
    cam_terms[f"{cam_name.split('/')[-1]}_{cam_type}"] = ObservationTermCfg(
      func=func, params=param_kwargs
    )

  camera_obs = ObservationGroupCfg(
    terms=cam_terms, enable_corruption=False, concatenate_terms=True
  )
  cfg.observations["camera"] = camera_obs

  if cam_type == "rgb":
    cfg.events["cube_color"] = EventTermCfg(
      func=dr.geom_rgba,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("cube", geom_names=(".*",)),
        "operation": "abs",
        "distribution": "uniform",
        "axes": [0, 1, 2],
        "ranges": (0.0, 1.0),
      },
    )

  # Pop privileged info from actor observations.
  actor_obs = cfg.observations["actor"]
  actor_obs.terms.pop("ee_to_cube")
  actor_obs.terms.pop("cube_to_goal")

  # Add goal_position to actor observations.
  actor_obs.terms["goal_position"] = ObservationTermCfg(
    func=manipulation_mdp.target_position,
    params={
      "command_name": "lift_height",
      "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
    },
    # NOTE: No noise for goal position.
  )

  return cfg


def _cube_color(i: int, n: int) -> tuple[float, float, float, float]:
  """Generate a distinct color for cube i of n using HSV hue rotation."""
  h = i / max(n, 1)
  r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.9)
  return (r, g, b, 1.0)


def yam_multi_cube_seg_env_cfg(
  num_cubes: int = 3,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Multi-cube task: depth + segmentation mask for goal conditioning."""
  cfg = make_lift_cube_env_cfg()

  cube_names = [f"cube_{i}" for i in range(num_cubes)]
  entities: dict[str, EntityCfg] = {"robot": get_yam_robot_cfg()}
  for i, name in enumerate(cube_names):
    color = _cube_color(i, num_cubes)
    entities[name] = EntityCfg(
      spec_fn=lambda c=color: get_cube_spec(rgba=c),
    )
  cfg.scene.entities = entities

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = YAM_ACTION_SCALE

  cfg.commands = {
    "lift_height": MultiCubeLiftingCommandCfg(
      entity_names=tuple(cube_names),
      resampling_time_range=(8.0, 12.0),
      debug_vis=True,
      difficulty="dynamic",
    ),
  }

  cfg.rewards["lift"] = RewardTermCfg(
    func=manipulation_mdp.multi_cube_staged_position_reward,
    weight=1.0,
    params={
      "command_name": "lift_height",
      "reaching_std": 0.2,
      "bringing_std": 0.3,
      "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
    },
  )
  cfg.rewards["lift_precise"] = RewardTermCfg(
    func=manipulation_mdp.multi_cube_bring_object_reward,
    weight=1.0,
    params={
      "command_name": "lift_height",
      "std": 0.05,
    },
  )

  fingertip_geoms = r"[lr]f_down(6|7|8|9|10|11)_collision"
  cfg.events["fingertip_friction_slide"].params[
    "asset_cfg"
  ].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link_6"

  cfg.viewer.body_name = "arm"
  cfg.sim.nconmax = max(cfg.sim.nconmax or 55, 55 + num_cubes * 120)

  cam_cfg = CameraSensorCfg(
    name="camera_d405",
    camera_name="robot/camera_d405",
    height=32,
    width=32,
    data_types=("depth", "segmentation"),
    enabled_geom_groups=(0, 3),
    use_shadows=False,
    use_textures=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)

  cam_terms = {
    "depth": ObservationTermCfg(
      func=manipulation_mdp.camera_depth,
      params={
        "sensor_name": "camera_d405",
        "cutoff_distance": 0.5,
      },
    ),
    "target_mask": ObservationTermCfg(
      func=manipulation_mdp.camera_target_cube_mask,
      params={
        "sensor_name": "camera_d405",
        "command_name": "lift_height",
      },
    ),
  }
  cfg.observations["camera"] = ObservationGroupCfg(
    terms=cam_terms,
    enable_corruption=False,
    concatenate_terms=True,
    concatenate_dim=0,
  )

  for group_name in ("actor", "critic"):
    obs = cfg.observations[group_name]
    obs.terms.pop("ee_to_cube", None)
    obs.terms.pop("cube_to_goal", None)
    obs.terms["goal_position"] = ObservationTermCfg(
      func=manipulation_mdp.target_position,
      params={
        "command_name": "lift_height",
        "asset_cfg": SceneEntityCfg("robot", site_names=("grasp_site",)),
      },
    )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    assert cfg.commands is not None
    cfg.commands["lift_height"].resampling_time_range = (
      4.0,
      4.0,
    )

  return cfg
