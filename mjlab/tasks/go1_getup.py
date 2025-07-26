from dataclasses import dataclass, field, replace

from mjlab.scene.scene_config import SceneCfg
from mjlab.asset_zoo.robots.unitree_go1.go1_constants import GO1_ROBOT_CFG
from mjlab.asset_zoo.terrains.flat_terrain import FLAT_TERRAIN_CFG
from mjlab.utils.spec_editor.spec_editor_config import TextureCfg, LightCfg

from mjlab.managers.manager_term_config import ObservationGroupCfg as ObsGroup
from mjlab.managers.manager_term_config import ObservationTermCfg as ObsTerm
from mjlab.managers.manager_term_config import ActionTermCfg as ActionTerm
from mjlab.managers.manager_term_config import RewardTermCfg as RewardTerm
from mjlab.managers.manager_term_config import term
from mjlab.envs.mdp import (
  observations,
  actions,
  terminations,
  events,
  rewards,
)
from mjlab.envs.manager_based_rl_env_config import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import TerminationTermCfg as DoneTerm
from mjlab.managers.manager_term_config import EventTermCfg as EventTerm

from mjlab.utils.noise import UniformNoiseCfg as Unoise


##
# Scene.
##

terrain_cfg = replace(FLAT_TERRAIN_CFG)
terrain_cfg.textures.append(
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
terrain_cfg.lights.append(
  LightCfg(pos=(0, 0, 1.5), type="directional"),
)

SCENE_CFG = SceneCfg(
  terrains={"floor": FLAT_TERRAIN_CFG},
  robots={"robot": GO1_ROBOT_CFG},
)

##
# MDP.
##

# Actions.


@dataclass
class ActionCfg:
  joint_pos: ActionTerm = term(
    actions.JointPositionActionCfg,
    asset_name="robot",
    actuator_names=[".*"],
    scale=0.0,
    use_default_offset=True,
  )


# Observations.


@dataclass
class ObservationCfg:
  @dataclass
  class PolicyCfg(ObsGroup):
    projected_gravity: ObsTerm = term(
      ObsTerm,
      func=observations.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    joint_pos: ObsTerm = term(
      ObsTerm,
      func=observations.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
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

    def __post_init__(self):
      self.enable_corruption = False
      self.concatenate_terms = True

  # @dataclass
  # class CriticCfg(PolicyCfg):
  #   pass
  #
  #   def __post_init__(self):
  #     super().__post_init__()
  #     self.enable_corruption = False

  policy: PolicyCfg = field(default_factory=PolicyCfg)
  # critic: CriticCfg = field(default_factory=CriticCfg)


# Events.


@dataclass
class EventCfg:
  reset_base: EventTerm = term(
    EventTerm,
    func=events.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {
        # "x": (-0.5, 0.5),
        # "y": (-0.5, 0.5),
        "z": (0.5, 0.5),
        # "yaw": (-3.14, 3.14),
      },
      # "velocity_range": {
      #   "x": (-0.5, 0.5),
      #   "y": (-0.5, 0.5),
      #   "z": (-0.5, 0.5),
      #   "roll": (-0.5, 0.5),
      #   "pitch": (-0.5, 0.5),
      #   "yaw": (-0.5, 0.5),
      # },
      "velocity_range": {},
    },
  )
  # reset_robot_joints: EventTerm = term(
  #   EventTerm,
  #   func=events.reset_joints_by_scale,
  #   mode="reset",
  #   params={
  #     "position_range": (0.5, 1.5),
  #     "velocity_range": (0.0, 0.0),
  #     # "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
  #   },
  # )

  # Add push.
  # Add domain randomization.


# Rewards.


@dataclass
class RewardCfg:
  orientation: RewardTerm = term(RewardTerm, func=rewards.upright, weight=1.0)
  # height: RewardTerm = term(RewardTerm, func=rewards.torso_desired_height, weight=1.0)
  posture: RewardTerm = term(RewardTerm, func=rewards.posture, weight=1.0)

  # Penalties.
  dof_torques_l2: RewardTerm = term(
    RewardTerm, func=rewards.joint_torques_l2, weight=-1.0e-5
  )
  # dof_acc_l2: RewardTerm = term(RewardTerm, func=rewards.joint_acc_l2, weight=-2.5e-7)
  action_rate_l2: RewardTerm = term(
    RewardTerm, func=rewards.action_rate_l2, weight=-0.01
  )
  # flat_orientation_l2: RewardTerm = term(
  #   RewardTerm, func=rewards.flat_orientation_l2, weight=-2.5
  # )
  # dof_pos_limits: RewardTerm = term(
  #   RewardTerm, func=rewards.joint_pos_limits, weight=0.0
  # )


# Terminations.


@dataclass
class TerminationCfg:
  time_out: DoneTerm = term(DoneTerm, func=terminations.time_out, time_out=True)


# Curriculum.


@dataclass
class CurriculumCfg:
  pass


@dataclass
class CommandCfg:
  pass


##
# Environment.
##

# Put everything together.


@dataclass
class Go1GetupFlatEnvCfg(ManagerBasedRlEnvCfg):
  scene: SceneCfg = field(default_factory=lambda: SCENE_CFG)
  observations: ObservationCfg = field(default_factory=ObservationCfg)
  actions: ActionCfg = field(default_factory=ActionCfg)
  decimation: int = 5
  rewards: RewardCfg = field(default_factory=RewardCfg)
  episode_length_s: float = 1.0
  events: EventCfg = field(default_factory=EventCfg)
  terminations: TerminationCfg = field(default_factory=TerminationCfg)
  commands: CommandCfg = field(default_factory=CommandCfg)

  def __post_init__(self):
    self.sim.mujoco.integrator = "implicitfast"
    self.sim.mujoco.cone = "pyramidal"
    self.sim.mujoco.timestep = 0.004
    self.sim.mujoco.iterations = 10
    self.sim.mujoco.ls_iterations = 20
    self.sim.num_envs = 64
    self.sim.njmax = 81920
    self.sim.nconmax = 25000


if __name__ == "__main__":
  import torch
  import mujoco.viewer
  import time
  from mjlab.scene import Scene
  import mujoco_warp as mjwarp
  import warp as wp
  from mjlab.sim.sim_data import WarpBridge

  env_cfg = Go1GetupFlatEnvCfg()

  scene = Scene(SCENE_CFG)
  scene.configure_sim_options(env_cfg.sim.mujoco)

  #   xml_string = """
  # <mujoco>
  #   <worldbody>
  #     <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
  #     <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
  #     <body pos="0 0 1">
  #       <joint type="free"/>
  #       <geom type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
  #     </body>
  #   </worldbody>
  # </mujoco>
  #   """

  mjm = scene.compile()
  # mjm = mujoco.MjModel.from_xml_string(xml_string)
  mjd = mujoco.MjData(mjm)

  mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
  mujoco.mj_forward(mjm, mjd)
  wp_model = mjwarp.put_model(mjm)
  wp_data = mjwarp.put_data(
    mjm,
    mjd,
    nworld=1,
  )

  data = WarpBridge(wp_data)

  with wp.ScopedCapture() as capture:
    mjwarp.step(wp_model, wp_data)
  step_graph = capture.graph

  with wp.ScopedCapture() as capture:
    mjwarp.forward(wp_model, wp_data)
  forward_graph = capture.graph

  def copy_env_to_viewer():
    mjd.qpos[:] = data.qpos[0].cpu().numpy()
    mjd.qvel[:] = data.qvel[0].cpu().numpy()
    mujoco.mj_forward(mjm, mjd)

  def copy_viewer_to_env():
    xfrc_applied = torch.tensor(mjd.xfrc_applied, dtype=torch.float, device="cuda:0")
    data.xfrc_applied[:] = xfrc_applied[None]

  # @torch.inference_mode()
  # def get_zero_action():
  #   return torch.zeros(12, device="cuda:0")

  FRAME_TIME = 1.0 / 60.0

  KEY_BACKSPACE = 259
  KEY_ENTER = 257

  def key_callback(key: int) -> None:
    if key == KEY_ENTER:
      print("RESET KEY DETECTED")
      data.qpos[:, 2] = 0.5
      wp.capture_launch(forward_graph)

  viewer = mujoco.viewer.launch_passive(mjm, mjd, key_callback=key_callback)
  with viewer:
    with viewer.lock():
      copy_env_to_viewer()
    viewer.sync()

    last_frame_time = time.perf_counter()
    step = 0
    while viewer.is_running():
      frame_start = time.perf_counter()

      copy_viewer_to_env()
      for i in range(env_cfg.decimation):
        wp.capture_launch(step_graph)

      with viewer.lock():
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
