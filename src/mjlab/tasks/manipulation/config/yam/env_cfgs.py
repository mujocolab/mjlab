from typing import Any, Literal

import mujoco

from mjlab.asset_zoo.robots import (
  YAM_ACTION_SCALE,
  get_yam_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg, ContactSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg


def get_cube_spec(cube_size: float = 0.02, mass: float = 0.05) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(cube_size,) * 3,
    mass=mass,
    rgba=(0.8, 0.2, 0.2, 1.0),
  )
  return spec


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
