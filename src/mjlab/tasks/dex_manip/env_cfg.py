from __future__ import annotations

import mujoco

from mjlab.asset_zoo.robots.leap_hand import get_leap_left_custom_hand_cfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from . import mdp as dex_mdp
from . import metrics as dex_metrics
from .objects import (
  DEFAULT_DEX_OBJECTS,
  object_names_to_mesh_files,
  object_names_to_mesh_names,
  parse_object_selection,
)


def get_multi_object_spec() -> mujoco.MjSpec:
  mesh_names = object_names_to_mesh_names(DEFAULT_DEX_OBJECTS)
  mesh_files = object_names_to_mesh_files(DEFAULT_DEX_OBJECTS)
  mesh_xml = "\n".join(
    f'      <mesh name="{mesh_name}" file="{mesh_file.as_posix()}"/>'
    for mesh_name, mesh_file in zip(mesh_names, mesh_files, strict=True)
  )
  xml = """
  <mujoco>
    <asset>
__MESH_XML__
    </asset>
    <worldbody>
      <body name="object">
        <freejoint name="object_joint"/>
        <geom name="object_geom" type="mesh" mesh="water_bottle_mesh" mass="0.1" rgba="0.8 0.2 0.2 1.0"/>
      </body>
    </worldbody>
  </mujoco>
  """
  return mujoco.MjSpec.from_string(xml.replace("__MESH_XML__", mesh_xml))


LEAP_ACTION_SCALE = 1.0 / 24.0
PALM_CENTER_GEOM_EXPR = "palm_collision_.*"


def make_dex_manip_base_env_cfg() -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=dex_mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "biased": True,
      },
    ),
    "prev_commanded_joint_pos": ObservationTermCfg(
      func=dex_mdp.joint_pos_commanded,
      params={
        "action_name": "joint_pos",
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }

  critic_terms = {
    "joint_pos": ObservationTermCfg(
      func=dex_mdp.joint_pos_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "joint_vel": ObservationTermCfg(
      func=dex_mdp.joint_vel_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "joint_pos_error": ObservationTermCfg(
      func=dex_mdp.joint_pos_command_error,
      params={
        "action_name": "joint_pos",
        "biased": True,
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "prev_commanded_joint_pos": ObservationTermCfg(
      func=dex_mdp.joint_pos_commanded,
      params={
        "action_name": "joint_pos",
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "object_pose_palm": ObservationTermCfg(
      func=dex_mdp.object_pose_in_palm_frame,
      params={
        "object_name": "object",
        "hand_cfg": SceneEntityCfg("robot", body_names=("palm",)),
        "palm_center_geom_expr": PALM_CENTER_GEOM_EXPR,
      },
    ),
    "object_lin_vel_palm": ObservationTermCfg(
      func=dex_mdp.object_lin_vel_in_palm_frame,
      params={
        "object_name": "object",
        "hand_cfg": SceneEntityCfg("robot", body_names=("palm",)),
        "palm_center_geom_expr": PALM_CENTER_GEOM_EXPR,
      },
    ),
    "object_ang_vel_palm": ObservationTermCfg(
      func=dex_mdp.object_ang_vel_in_palm_frame,
      params={
        "object_name": "object",
        "hand_cfg": SceneEntityCfg("robot", body_names=("palm",)),
        "palm_center_geom_expr": PALM_CENTER_GEOM_EXPR,
      },
    ),
    "object_size": ObservationTermCfg(
      func=dex_mdp.object_size,
      params={"object_name": "object", "geom_name": "object_geom"},
    ),
    "object_mass": ObservationTermCfg(
      func=dex_mdp.object_mass,
      params={"object_name": "object", "body_name": "object"},
    ),
    "object_com_offset_b": ObservationTermCfg(
      func=dex_mdp.object_com_offset_b,
      params={"object_name": "object", "body_name": "object"},
    ),
    "object_friction": ObservationTermCfg(
      func=dex_mdp.object_friction_coeff,
      params={"object_name": "object", "geom_name": "object_geom", "axis": 0},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      actor_terms,
      enable_corruption=True,
      history_length=10,
      flatten_history_dim=True,
    ),
    "critic": ObservationGroupCfg(
      critic_terms,
      enable_corruption=False,
      history_length=1,
      flatten_history_dim=True,
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": dex_mdp.JointPositionDeltaActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=1.0,
      offset=0.0,
      use_default_offset=False,
      clip_to_joint_limits=True,
      use_soft_joint_pos_limits=True,
      delta_min=-LEAP_ACTION_SCALE,
      delta_max=LEAP_ACTION_SCALE,
      interpolate_decimation=True,
    )
  }

  sensors = (
    ContactSensorCfg(
      name="fingertip_object_contact",
      primary=ContactMatch(mode="geom", pattern=".*_tip", entity="robot"),
      secondary=ContactMatch(mode="geom", pattern="object_geom", entity="object"),
      fields=("found",),
      reduce="none",
      num_slots=1,
    ),
  )

  events = {
    "reset_base": EventTermCfg(
      func=dex_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {},
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=dex_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.03, 0.03),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "reset_object_pose": EventTermCfg(
      func=dex_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("object"),
        "pose_range": {
          "x": (-0.006, 0.006),
          "y": (-0.006, 0.006),
          "z": (-0.005, 0.005),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "randomize_object_mesh": EventTermCfg(
      mode="reset",
      func=dr.geom_dataid,
      params={
        "mesh_ids": object_names_to_mesh_names(DEFAULT_DEX_OBJECTS),
        "assignment_mode": "cycle",
        "shared_random": True,
        "asset_cfg": SceneEntityCfg("object", geom_names=("object_geom",)),
      },
    ),
    "dr_shared_contact_friction": EventTermCfg(
      func=dex_mdp.randomize_shared_contact_friction,
      mode="reset",
      params={
        "friction_range": (0.6, 1.4),
        "hand_cfg": SceneEntityCfg("robot", geom_names=(".*",)),
        "object_cfg": SceneEntityCfg("object", geom_names=("object_geom",)),
        "axes": (0,),
      },
    ),
    "dr_object_mass": EventTermCfg(
      func=dex_mdp.randomize_body_mass,
      mode="reset",
      params={
        "mass_range": (0.7, 1.4),
        "distribution": "uniform",
        "operation": "scale",
        "asset_cfg": SceneEntityCfg("object", body_names=("object",)),
      },
    ),
    "dr_robot_link_masses": EventTermCfg(
      func=dex_mdp.randomize_body_mass,
      mode="reset",
      params={
        "mass_range": (0.8, 1.2),
        "distribution": "uniform",
        "operation": "scale",
        "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
      },
    ),
  }

  rewards = {
    "rotate_finite_diff": RewardTermCfg(
      func=dex_mdp.object_yaw_finite_diff_clipped,
      weight=1.25,
      params={
        "object_name": "object",
        "clip_min": -0.25,
        "clip_max": 0.25,
        "history_steps": 4,
        "negate_yaw_rate": True,
        "drift_position_threshold": 0.02,
        "drift_tilt_threshold": 0.35,
        "drift_mode": "step",
        "drift_inside_factor": 1.0,
        "drift_outside_factor": 0.1,
      },
    ),
    "object_linvel_penalty": RewardTermCfg(
      func=dex_mdp.object_linvel_l1,
      weight=-0.3,
      params={"object_name": "object"},
    ),
    "pose_diff_penalty": RewardTermCfg(
      func=dex_mdp.pose_diff_l2_from_reset,
      weight=-0.1,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "average_per_joint": False,
        "joint_tolerance": 0.4,
      },
    ),
    "torque_penalty": RewardTermCfg(
      func=dex_mdp.joint_torque_l2,
      weight=-0.1,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "work_penalty": RewardTermCfg(
      func=dex_mdp.actuator_work_l2_penalty,
      weight=-0.05,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "object_fallen": RewardTermCfg(
      func=dex_mdp.object_fallen,
      weight=-10.0,
      params={"object_name": "object", "minimum_height": 0.2},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=dex_mdp.time_out, time_out=True),
    "object_fell": TerminationTermCfg(
      func=dex_mdp.root_height_below_minimum,
      params={"minimum_height": 0.2, "asset_cfg": SceneEntityCfg("object")},
    ),
    "object_too_fast": TerminationTermCfg(
      func=dex_mdp.object_linear_speed_above,
      params={"max_linear_speed": 1.0, "asset_cfg": SceneEntityCfg("object")},
    ),
    "object_pose_deviation": TerminationTermCfg(
      func=dex_mdp.object_pose_rp_position_deviation_from_reset,
      params={
        "max_position_error": 0.08,
        "max_tilt_error": 0.8,
        "asset_cfg": SceneEntityCfg("object"),
      },
    ),
    "nan": TerminationTermCfg(func=dex_mdp.nan_detection),
  }

  metrics = {
    "reward_mean": MetricsTermCfg(func=dex_metrics.reward_mean),
    "rotation_progress": MetricsTermCfg(
      func=dex_metrics.object_rotation_progress,
      params={
        "asset_cfg": SceneEntityCfg("object"),
        "target_yaw_rate": 0.20,
        "position_threshold": 0.02,
        "tilt_threshold": 0.35,
      },
    ),
    "linear_speed": MetricsTermCfg(
      func=dex_metrics.object_linear_speed,
      params={"asset_cfg": SceneEntityCfg("object")},
    ),
    "position_error": MetricsTermCfg(
      func=dex_metrics.object_pose_rp_error_from_reset,
      params={"component": "position", "asset_cfg": SceneEntityCfg("object")},
    ),
    "tilt_error": MetricsTermCfg(
      func=dex_metrics.object_pose_rp_error_from_reset,
      params={"component": "tilt", "asset_cfg": SceneEntityCfg("object")},
    ),
    "success": MetricsTermCfg(
      func=dex_metrics.object_rotation_success,
      params={
        "asset_cfg": SceneEntityCfg("object"),
        "target_yaw_rate": 0.20,
        "position_threshold": 0.02,
        "tilt_threshold": 0.35,
      },
    ),
  }

  curriculum = {
    "object_linvel_penalty_weight": CurriculumTermCfg(
      func=dex_mdp.reward_weight_by_metric_progress,
      params={
        "reward_name": "object_linvel_penalty",
        "metric_name": "rotation_progress",
        "progress_min": 0.05,
        "progress_max": 0.25,
        "weight_min": -0.03,
        "weight_max": -0.3,
        "ema_alpha": 0.08,
        "weight_lerp": 0.15,
      },
    ),
    "pose_diff_penalty_weight": CurriculumTermCfg(
      func=dex_mdp.reward_weight_by_metric_progress,
      params={
        "reward_name": "pose_diff_penalty",
        "metric_name": "rotation_progress",
        "progress_min": 0.05,
        "progress_max": 0.25,
        "weight_min": -0.01,
        "weight_max": -0.1,
        "ema_alpha": 0.08,
        "weight_lerp": 0.15,
      },
    ),
    "torque_penalty_weight": CurriculumTermCfg(
      func=dex_mdp.reward_weight_by_metric_progress,
      params={
        "reward_name": "torque_penalty",
        "metric_name": "rotation_progress",
        "progress_min": 0.05,
        "progress_max": 0.25,
        "weight_min": -0.1,
        "weight_max": -1.0,
        "ema_alpha": 0.08,
        "weight_lerp": 0.15,
      },
    ),
    "work_penalty_weight": CurriculumTermCfg(
      func=dex_mdp.reward_weight_by_metric_progress,
      params={
        "reward_name": "work_penalty",
        "metric_name": "rotation_progress",
        "progress_min": 0.05,
        "progress_max": 0.25,
        "weight_min": -0.01,
        "weight_max": -0.1,
        "ema_alpha": 0.08,
        "weight_lerp": 0.15,
      },
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={},
      sensors=sensors,
      num_envs=1,
      env_spacing=0.6,
    ),
    observations=observations,
    actions=actions,
    commands={},
    events=events,
    rewards=rewards,
    terminations=terminations,
    metrics=metrics,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="palm",
      distance=0.45,
      elevation=-25,
      azimuth=110,
    ),
    sim=SimulationCfg(
      nconmax=55,
      njmax=600,
      contact_sensor_maxmatch=256,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=10,
    episode_length_s=20.0,
    scale_rewards_by_dt=True,
  )


def apply_dex_manip_overrides(
  cfg: ManagerBasedRlEnvCfg,
  *,
  objects: str | None,
  envs_per_object: int | None,
  assignment_mode: str | None = None,
) -> tuple[str, ...]:
  object_names = parse_object_selection(objects)
  mesh_names = object_names_to_mesh_names(object_names)

  event = cfg.events.get("randomize_object_mesh")
  if event is None:
    raise ValueError("Dex manip cfg is missing 'randomize_object_mesh' event.")

  event.params["mesh_ids"] = mesh_names
  if assignment_mode is not None:
    event.params["assignment_mode"] = assignment_mode

  if envs_per_object is not None:
    if envs_per_object <= 0:
      raise ValueError(f"envs_per_object must be > 0. Got {envs_per_object}.")
    cfg.scene.num_envs = len(object_names) * envs_per_object

  return object_names


def dex_manip_env_cfg(
  play: bool = False,
  objects: str | None = None,
  envs_per_object: int | None = 8,
) -> ManagerBasedRlEnvCfg:
  cfg = make_dex_manip_base_env_cfg()
  cfg.scene.entities = {
    "robot": get_leap_left_custom_hand_cfg(),
    "object": EntityCfg(spec_fn=get_multi_object_spec),
  }

  selected_objects = apply_dex_manip_overrides(
    cfg,
    objects=objects,
    envs_per_object=envs_per_object,
  )
  selected_meshes = object_names_to_mesh_names(selected_objects)
  for object_name, mesh_name in zip(selected_objects, selected_meshes, strict=True):
    cfg.metrics[f"reward_{object_name.replace('-', '_')}"] = MetricsTermCfg(
      func=dex_metrics.reward_for_mesh,
      params={"mesh_name": mesh_name},
    )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}

  return cfg
