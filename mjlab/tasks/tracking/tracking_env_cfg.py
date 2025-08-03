from dataclasses import dataclass, field, replace

from mjlab.scene.scene_config import SceneCfg
from mjlab.utils.spec_editor import TextureCfg, LightCfg
from mjlab.asset_zoo.terrains.flat_terrain import FLAT_TERRAIN_CFG

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.manager_term_config import ActionTermCfg as ActionTerm
from mjlab.managers.manager_term_config import ObservationGroupCfg as ObsGroup
from mjlab.managers.manager_term_config import ObservationTermCfg as ObsTerm
from mjlab.managers.manager_term_config import term
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.managers.manager_term_config import TerminationTermCfg as DoneTerm
from mjlab.managers.manager_term_config import EventTermCfg as EventTerm
from mjlab.managers.manager_term_config import RewardTermCfg as RewTerm

from mjlab.tasks.tracking import mdp

# fmt: off
VELOCITY_RANGE = {
  "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2),
  "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78),
}
# fmt: on

##
# Scene.
##

terrain_cfg = replace(FLAT_TERRAIN_CFG)
terrain_cfg.textures = terrain_cfg.textures + (
  TextureCfg(
    name="skybox",
    type="skybox",
    builtin="gradient",
    rgb1=(0.3, 0.5, 0.7),
    rgb2=(0.1, 0.2, 0.3),
    width=512,
    height=3072,
  ),
)
terrain_cfg.lights = terrain_cfg.lights + (
  LightCfg(pos=(0, 0, 1.5), type="directional"),
)

SCENE_CFG = SceneCfg(
  terrains={"floor": terrain_cfg},
  # sensors={
  #   "contact_forces": ContactSensorCfg(
  #     entity_name="robot",
  #     history_length=3,
  #     track_air_time=True,
  #     force_threshold=10.0,
  #   ),
  # },
)


@dataclass
class CommandsCfg:
  motion: mdp.MotionCommandCfg = term(
    mdp.MotionCommandCfg,
    asset_name="robot",
    resampling_time_range=(1.0e9, 1.0e9),
    debug_vis=True,
    pose_range={
      "x": (-0.05, 0.05),
      "y": (-0.05, 0.05),
      "z": (-0.01, 0.01),
      "roll": (-0.1, 0.1),
      "pitch": (-0.1, 0.1),
      "yaw": (-0.2, 0.2),
    },
    velocity_range=VELOCITY_RANGE,
    joint_position_range=(-0.1, 0.1),
    reference_body="torso_link",
    body_names=[
      "pelvis",
      "left_hip_roll_link",
      "left_knee_link",
      "left_ankle_roll_link",
      "right_hip_roll_link",
      "right_knee_link",
      "right_ankle_roll_link",
      "torso_link",
      "left_shoulder_roll_link",
      "left_elbow_link",
      "left_wrist_yaw_link",
      "right_shoulder_roll_link",
      "right_elbow_link",
      "right_wrist_yaw_link",
    ],
    motion_file="/home/kevin/dev/mjlab/motions/motion.npz",
  )


@dataclass
class ActionCfg:
  joint_pos: ActionTerm = term(
    mdp.JointPositionActionCfg,
    asset_name="robot",
    actuator_names=[".*"],
    scale=0.5,
    use_default_offset=True,
  )


@dataclass
class ObservationCfg:
  @dataclass
  class PolicyCfg(ObsGroup):
    command: ObsTerm = term(
      ObsTerm, func=mdp.generated_commands, params={"command_name": "motion"}
    )
    motion_ref_pos_b: ObsTerm = term(
      ObsTerm,
      func=mdp.motion_ref_pos_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.25, n_max=0.25),
    )
    motion_ref_ori_b: ObsTerm = term(
      ObsTerm,
      func=mdp.motion_ref_ori_b,
      params={"command_name": "motion"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    base_lin_vel: ObsTerm = term(
      ObsTerm, func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5)
    )
    base_ang_vel: ObsTerm = term(
      ObsTerm, func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
    )
    joint_pos: ObsTerm = term(
      ObsTerm, func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
    )
    joint_vel: ObsTerm = term(
      ObsTerm, func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
    )
    actions: ObsTerm = term(ObsTerm, func=mdp.last_action)

    def __post_init__(self):
      self.enable_corruption = True

  @dataclass
  class PrivilegedCfg(ObsGroup):
    command: ObsTerm = term(
      ObsTerm, func=mdp.generated_commands, params={"command_name": "motion"}
    )
    motion_ref_pos_b: ObsTerm = term(
      ObsTerm, func=mdp.motion_ref_pos_b, params={"command_name": "motion"}
    )
    motion_ref_ori_b: ObsTerm = term(
      ObsTerm, func=mdp.motion_ref_ori_b, params={"command_name": "motion"}
    )
    body_pos: ObsTerm = term(
      ObsTerm, func=mdp.robot_body_pos_b, params={"command_name": "motion"}
    )
    body_ori: ObsTerm = term(
      ObsTerm, func=mdp.robot_body_ori_b, params={"command_name": "motion"}
    )
    base_lin_vel: ObsTerm = term(ObsTerm, func=mdp.base_lin_vel)
    base_ang_vel: ObsTerm = term(ObsTerm, func=mdp.base_ang_vel)
    joint_pos: ObsTerm = term(ObsTerm, func=mdp.joint_pos_rel)
    joint_vel: ObsTerm = term(ObsTerm, func=mdp.joint_vel_rel)
    actions: ObsTerm = term(ObsTerm, func=mdp.last_action)

  policy: PolicyCfg = field(default_factory=PolicyCfg)
  critic: PrivilegedCfg = field(default_factory=PrivilegedCfg)


@dataclass
class EventCfg:
  push_robot: EventTerm = term(
    EventTerm,
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(1.0, 3.0),
    params={"velocity_range": VELOCITY_RANGE},
  )

  # MISSING: base_com, add_joint_default_pos, physics_material


@dataclass
class RewardCfg:
  motion_global_root_pos: RewTerm = term(
    RewTerm,
    func=mdp.motion_global_ref_position_error_exp,
    weight=0.5,
    params={"command_name": "motion", "std": 0.3},
  )
  motion_global_root_ori: RewTerm = term(
    RewTerm,
    func=mdp.motion_global_ref_orientation_error_exp,
    weight=0.5,
    params={"command_name": "motion", "std": 0.4},
  )
  motion_body_pos: RewTerm = term(
    RewTerm,
    func=mdp.motion_relative_body_position_error_exp,
    weight=1.0,
    params={"command_name": "motion", "std": 0.3},
  )
  motion_body_ori: RewTerm = term(
    RewTerm,
    func=mdp.motion_relative_body_orientation_error_exp,
    weight=1.0,
    params={"command_name": "motion", "std": 0.4},
  )
  motion_body_lin_vel: RewTerm = term(
    RewTerm,
    func=mdp.motion_global_body_linear_velocity_error_exp,
    weight=1.0,
    params={"command_name": "motion", "std": 1.0},
  )
  motion_body_ang_vel: RewTerm = term(
    RewTerm,
    func=mdp.motion_global_body_angular_velocity_error_exp,
    weight=1.0,
    params={"command_name": "motion", "std": 3.14},
  )

  action_rate_l2: RewTerm = term(RewTerm, func=mdp.action_rate_l2, weight=-1e-1)
  joint_limit: RewTerm = term(
    RewTerm,
    func=mdp.joint_pos_limits,
    weight=-100.0,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
  )

  # MISSING: undesired_contacts


@dataclass
class TerminationsCfg:
  time_out: DoneTerm = term(DoneTerm, func=mdp.time_out, time_out=True)

  ref_pos: DoneTerm = term(
    DoneTerm,
    func=mdp.bad_ref_pos_z_only,
    params={"command_name": "motion", "threshold": 0.25},
  )

  ref_ori: DoneTerm = term(
    DoneTerm,
    func=mdp.bad_ref_ori,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "command_name": "motion",
      "threshold": 0.8,
    },
  )

  ee_body_pos: DoneTerm = term(
    DoneTerm,
    func=mdp.bad_motion_body_pos,
    params={
      "command_name": "motion",
      "threshold": 0.25,
      "body_names": [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
      ],
    },
  )


@dataclass
class TrackingEnvCfg(ManagerBasedRlEnvCfg):
  scene: SceneCfg = field(default_factory=lambda: SCENE_CFG)
  observations: ObservationCfg = field(default_factory=ObservationCfg)
  actions: ActionCfg = field(default_factory=ActionCfg)
  commands: CommandsCfg = field(default_factory=CommandsCfg)
  rewards: RewardCfg = field(default_factory=RewardCfg)
  terminations: TerminationsCfg = field(default_factory=TerminationsCfg)
  events: EventCfg = field(default_factory=EventCfg)
  decimation: int = 4
  episode_length_s: float = 10.0

  def __post_init__(self) -> None:
    self.sim.mujoco.integrator = "implicitfast"
    self.sim.mujoco.cone = "pyramidal"
    self.sim.mujoco.timestep = 0.005
    self.sim.num_envs = 1
    self.sim.nconmax = 50000
    self.sim.njmax = 250
    self.sim.mujoco.iterations = 10
    self.sim.mujoco.ls_iterations = 20
