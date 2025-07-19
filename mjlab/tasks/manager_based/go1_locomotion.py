from dataclasses import dataclass, field

from mjlab.entities.scene.scene_config import SceneCfg, LightCfg
from mjlab.entities.common.config import TextureCfg
from mjlab.entities.robots.go1.go1_constants import GO1_ROBOT_CFG
from mjlab.entities.terrains.flat_terrain import FLAT_TERRAIN_CFG

from mjlab.managers.manager_term_config import ObservationGroupCfg as ObsGroup
from mjlab.managers.manager_term_config import ObservationTermCfg as ObsTerm
from mjlab.managers.manager_term_config import RewardTermCfg as RewardTerm
from mjlab.managers.manager_term_config import ActionTermCfg as ActionTerm
from mjlab.managers.manager_term_config import term
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs.mdp import (
  rewards,
  observations,
  actions,
  terminations,
  events,
  commands,
)
from mjlab.envs.manager_based_rl_env import ManagerBasedRLEnv
from mjlab.envs.manager_based_rl_env_config import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import TerminationTermCfg as DoneTerm
from mjlab.managers.manager_term_config import EventTermCfg as EventTerm
import math

from mjlab.tasks.manager_based.mdp import terminations as custom_terminations
from mjlab.utils.noise import UniformNoiseCfg as Unoise


##
# Scene.
##


SCENE_CFG = SceneCfg(
  terrains={"floor": FLAT_TERRAIN_CFG},
  robots={"robot": GO1_ROBOT_CFG},
  lights=(LightCfg(pos=(0, 0, 1.5), type="directional"),),
  skybox=TextureCfg(
    name="skybox",
    type="skybox",
    builtin="gradient",
    rgb1=(0.3, 0.5, 0.7),
    rgb2=(0.1, 0.2, 0.3),
    width=512,
    height=3072,
  ),
)

##
# MDP.
##

# Commands.


@dataclass
class CommandsCfg:
  base_velocity: commands.UniformVelocityCommandCfg = term(
    commands.UniformVelocityCommandCfg,
    asset_name="robot",
    resampling_time_range=(10.0, 10.0),
    rel_standing_envs=0.02,
    rel_heading_envs=1.0,
    heading_command=True,
    heading_control_stiffness=0.5,
    debug_vis=True,
    ranges=commands.UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.0),
      lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-1.0, 1.0),
      heading=(-math.pi, math.pi),
    ),
  )


# Observations.


@dataclass
class ObservationCfg:
  @dataclass
  class PolicyCfg(ObsGroup):
    base_lin_vel: ObsTerm = term(
      ObsTerm,
      func=observations.base_lin_vel,
      noise=Unoise(n_min=-0.1, n_max=0.1),
    )
    base_ang_vel: ObsTerm = term(
      ObsTerm,
      func=observations.base_ang_vel,
      noise=Unoise(n_min=-0.2, n_max=0.2),
    )
    projected_gravity: ObsTerm = term(
      ObsTerm,
      func=observations.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    joint_pos: ObsTerm = term(
      ObsTerm,
      func=observations.joint_pos_rel,
    noise = Unoise(n_min=-0.01, n_max=0.01),
    )
    joint_vel: ObsTerm = term(
      ObsTerm,
      func=observations.joint_vel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    )
    actions: ObsTerm = term(
      ObsTerm,
      func=observations.last_action,
    )
    velocity_commands: ObsTerm = term(
      ObsTerm,
      func=observations.generated_commands,
      params={"command_name": "base_velocity"},
    )

    def __post_init__(self):
      self.enable_corruption = True
      self.concatenate_terms = True

  @dataclass
  class CriticCfg(PolicyCfg):
    pass
  
    def __post_init__(self):
      super().__post_init__()
      self.enable_corruption = False

  policy: PolicyCfg = field(default_factory=PolicyCfg)
  critic: CriticCfg = field(default_factory=CriticCfg)


# Events.


@dataclass
class EventCfg:
  reset_base: EventTerm = term(
    EventTerm,
    func=events.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
      "velocity_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.5, 0.5),
        "roll": (-0.5, 0.5),
        "pitch": (-0.5, 0.5),
        "yaw": (-0.5, 0.5),
      },
    },
  )
  reset_robot_joints: EventTerm = term(
    EventTerm,
    func=events.reset_joints_by_scale,
    mode="reset",
    params={
      "position_range": (0.5, 1.5),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
    },
  )
  
  # Add push.
  # Add domain randomization.


# Rewards.


@dataclass
class RewardCfg:
  pass
  # # Task.
  # track_lin_vel_xy_exp: RewardTerm = term(
  #   RewardTerm,
  #   func=rewards.track_lin_vel_xy_exp,
  #   weight=1.0,
  #   params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
  # )
  # track_ang_vel_z_exp: RewardTerm = term(
  #   RewardTerm,
  #   func=rewards.track_ang_vel_z_exp,
  #   weight=0.5,
  #   params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
  # )
  # # Penalties.
  # lin_vel_z_l2: RewardTerm = term(RewardTerm, func=rewards.lin_vel_z_l2, weight=-2.0)
  # ang_vel_xy_l2: RewardTerm = term(RewardTerm, func=rewards.ang_vel_xy_l2, weight=-0.05)
  # dof_torques_l2: RewardTerm = term(RewardTerm, func=rewards.joint_torques_l2, weight=-1.0e-5)
  # dof_acc_l2: RewardTerm = term(RewardTerm, func=rewards.joint_acc_l2, weight=-2.5e-7)
  # action_rate_l2: RewardTerm = term(RewardTerm, func=rewards.action_rate_l2, weight=-0.01)
  # flat_orientation_l2: RewardTerm = term(RewardTerm, func=rewards.flat_orientation_l2, weight=0.0)
  # dof_pos_limits: RewardTerm = term(RewardTerm, func=rewards.joint_pos_limits, weight=0.0)


# Terminations.


@dataclass
class TerminationCfg:
  time_out: DoneTerm = term(DoneTerm, func=terminations.time_out, time_out=True)
  fell_over: DoneTerm = term(
    DoneTerm,
    func=custom_terminations.bad_orientation,
    params={"threshold": 0.5},
  )


# Curriculum.


@dataclass
class CurriculumCfg:
  pass


# Actions.


@dataclass
class ActionCfg:
  joint_pos: ActionTerm = term(
    actions.JointPositionActionCfg,
    asset_name="robot",
    joint_names=[".*"],
    scale=0.5,
    use_default_offset=True,
  )


##
# Environment.
##

# Put everything together.


@dataclass
class Go1LocomotionFlatEnvCfg(ManagerBasedRlEnvCfg):
  scene: SceneCfg = field(default_factory=lambda: SCENE_CFG)
  observations: ObservationCfg = field(default_factory=ObservationCfg)
  actions: ActionCfg = field(default_factory=ActionCfg)
  decimation: int = 5
  rewards: RewardCfg = field(default_factory=RewardCfg)
  episode_length_s: float = 10.0
  events: EventCfg = field(default_factory=EventCfg)
  terminations: TerminationCfg = field(default_factory=TerminationCfg)
  commands: CommandsCfg = field(default_factory=CommandsCfg)

  def __post_init__(self):
    self.sim.mujoco.integrator = "implicitfast"
    self.sim.mujoco.cone = "pyramidal"
    self.sim.mujoco.timestep = 0.004
    self.sim.mujoco.iterations = 10
    self.sim.mujoco.ls_iterations = 20
    self.sim.num_envs = 2
    # self.sim.njmax = 1028


if __name__ == "__main__":
  import torch
  import mujoco.viewer
  import time
  
  VIZ_OTHERS = False

  # Setup environment
  env_cfg = Go1LocomotionFlatEnvCfg()
  env = ManagerBasedRLEnv(cfg=env_cfg)

  mjm = env.sim.mj_model
  mjd = env.sim.mj_data
  
  # # For visualizing other envs.
  # if VIZ_OTHERS:
  #   vd = mujoco.MjData(mjm)
  #   pert = mujoco.MjvPerturb()
  #   vopt = mujoco.MjvOption()
  #   catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC

  def copy_env_to_viewer():
    mjd.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
    mjd.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
    mujoco.mj_forward(mjm, mjd)

  def copy_viewer_to_env():
    xfrc_applied = torch.tensor(mjd.xfrc_applied, dtype=torch.float, device=env.device)
    env.sim.data.xfrc_applied[:] = xfrc_applied[None]

  @torch.no_grad()
  def get_zero_action():
    return torch.rand(
      (env.num_envs, env.action_manager.total_action_dim),
      device="cuda:0",
    )

  FRAME_TIME = 1.0 / 60.0

  KEY_BACKSPACE = 259
  KEY_ENTER = 257

  def key_callback(key: int) -> None:
    if key == KEY_ENTER:
      print("RESET KEY DETECTED")
      env.reset()

  viewer = mujoco.viewer.launch_passive(mjm, mjd, key_callback=key_callback)
  with viewer:
    obs, extras = env.reset()
    copy_env_to_viewer()

    last_frame_time = time.perf_counter()

    step = 0
    while viewer.is_running():
      frame_start = time.perf_counter()

      copy_viewer_to_env()
      env.step(get_zero_action())

      viewer.user_scn.ngeom = 0
      env.update_visualizers(viewer.user_scn)
      
      # if VIZ_OTHERS:
      #   for i in range(1, env.num_envs):
      #     vd.qpos[:] = env.sim.data.qpos[i].cpu().numpy()
      #     vd.qvel[:] = env.sim.data.qvel[i].cpu().numpy()
      #     mujoco.mj_forward(mjm, vd)
      #     mujoco.mjv_addGeoms(mjm, vd, vopt, pert, catmask, viewer.user_scn)

      copy_env_to_viewer()
      viewer.sync(state_only=True)

      elapsed = time.perf_counter() - frame_start
      remaining_time = FRAME_TIME - elapsed
      if remaining_time > 0.005:
        time.sleep(remaining_time - 0.003)
      while (time.perf_counter() - frame_start) < FRAME_TIME:
        pass

      # Debug FPS every 60 frames
      step += 1
      current_time = time.perf_counter()
      if step % 60 == 0 and step > 0:
        actual_fps = 1.0 / (current_time - last_frame_time)
        print(f"Step {step}: FPS={actual_fps:.1f}")
      last_frame_time = current_time
