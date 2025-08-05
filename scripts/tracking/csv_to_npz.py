import numpy as np
import torch
import tyro
from dataclasses import replace

from mjlab.asset_zoo.robots.unitree_g1 import g1_constants
from mjlab.asset_zoo.robots.booster_t1 import t1_constants
from mjlab.entities import Robot
from mjlab.utils.math import (
  quat_mul,
  quat_conjugate,
  axis_angle_from_quat,
  quat_slerp,
  quat_apply_inverse,
)
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg

from mjlab.scene.scene_config import SceneCfg
from mjlab.asset_zoo.terrains.flat_terrain import FLAT_TERRAIN_CFG

terrain_cfg = replace(FLAT_TERRAIN_CFG)
# terrain_cfg.textures.append(
#   TextureCfg(
#     name="skybox",
#     type="skybox",
#     builtin="gradient",
#     rgb1=(0.3, 0.5, 0.7),
#     rgb2=(0.1, 0.2, 0.3),
#     width=512,
#     height=3072,
#   ),
# )
# terrain_cfg.lights.append(
#   LightCfg(pos=(0, 0, 1.5), type="directional"),
# )

SCENE_CFG = SceneCfg(
  terrains={"floor": terrain_cfg},
  # robots={"robot": replace(g1_constants.G1_ROBOT_CFG)},
  robots={"robot": replace(t1_constants.T1_ROBOT_CFG)},
)


class MotionLoader:
  def __init__(
    self,
    motion_file: str,
    input_fps: int,
    output_fps: int,
    device: torch.device,
    line_range: tuple[int, int] | None = None,
  ):
    self.motion_file = motion_file
    self.input_fps = input_fps
    self.output_fps = output_fps
    self.input_dt = 1.0 / self.input_fps
    self.output_dt = 1.0 / self.output_fps
    self.current_idx = 0
    self.device = device
    self.line_range = line_range
    self._load_motion()
    self._interpolate_motion()
    self._compute_velocities()

  def _load_motion(self):
    """Loads the motion from the csv file."""
    if self.line_range is None:
      motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
    else:
      motion = torch.from_numpy(
        np.loadtxt(
          self.motion_file,
          delimiter=",",
          skiprows=self.line_range[0] - 1,
          max_rows=self.line_range[1] - self.line_range[0] + 1,
        )
      )
    motion = motion.to(torch.float32).to(self.device)
    self.motion_base_poss_input = motion[:, :3]
    self.motion_base_rots_input = motion[:, 3:7]
    self.motion_base_rots_input = self.motion_base_rots_input[
      :, [3, 0, 1, 2]
    ]  # convert to wxyz
    self.motion_dof_poss_input = motion[:, 7:]

    self.input_frames = motion.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt
    print(
      f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}"
    )

  def _interpolate_motion(self):
    """Interpolates the motion to the output fps."""
    times = torch.arange(
      0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
    )
    self.output_frames = times.shape[0]
    index_0, index_1, blend = self._compute_frame_blend(times)
    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[index_0],
      self.motion_base_poss_input[index_1],
      blend.unsqueeze(1),
    )
    self.motion_base_rots = self._slerp(
      self.motion_base_rots_input[index_0],
      self.motion_base_rots_input[index_1],
      blend,
    )
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0],
      self.motion_dof_poss_input[index_1],
      blend.unsqueeze(1),
    )
    print(
      f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames: {self.output_frames}, output fps: {self.output_fps}"
    )

  def _lerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Linear interpolation between two tensors."""
    return a * (1 - blend) + b * blend

  def _slerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Spherical linear interpolation between two quaternions."""
    slerped_quats = torch.zeros_like(a)
    for i in range(a.shape[0]):
      slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
    return slerped_quats

  def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
    """Computes the frame blend for the motion."""
    phase = times / self.duration
    index_0 = (phase * (self.input_frames - 1)).floor().long()
    index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
    blend = phase * (self.input_frames - 1) - index_0
    return index_0, index_1, blend

  def _compute_velocities(self):
    """Computes the velocities of the motion."""
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_base_ang_vels = self._so3_derivative(
      self.motion_base_rots, self.output_dt
    )

  def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
    """Computes the derivative of a sequence of SO3 rotations.

    Args:
        rotations: shape (B, 4).
        dt: time step.
    Returns:
        shape (B, 3).
    """
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
    omega = torch.cat(
      [omega[:1], omega, omega[-1:]], dim=0
    )  # repeat first and last sample
    return omega

  def get_next_state(
    self,
  ) -> tuple[
    tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
    ],
    bool,
  ]:
    """Gets the next state of the motion."""
    state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    self.current_idx += 1
    reset_flag = False
    if self.current_idx >= self.output_frames:
      self.current_idx = 0
      reset_flag = True
    return state, reset_flag


def run_sim(
  sim: Simulation,
  scene,
  joint_names,
  input_file,
  input_fps,
  output_fps,
  output_name,
):
  # Load motion
  motion = MotionLoader(
    motion_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
  )

  robot: Robot = scene["robot"]
  robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

  # ------- data logger -------------------------------------------------------
  log = {
    "fps": [output_fps],
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }
  file_saved = False
  # --------------------------------------------------------------------------

  # frames = []
  scene.reset()

  while not file_saved:
    (
      (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
      ),
      reset_flag,
    ) = motion.get_next_state()

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = motion_base_pos
    root_states[:, 3:7] = motion_base_rot
    root_states[:, 7:10] = motion_base_lin_vel
    # root_states[:, 10:] = motion_base_ang_vel
    root_states[:, 10:] = quat_apply_inverse(motion_base_rot, motion_base_ang_vel)
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = motion_dof_pos
    joint_vel[:, robot_joint_indexes] = motion_dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)
    # sim.update_render()
    # frames.append(sim.render())

    if not file_saved:
      log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
      log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
      log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
      log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
      log["body_lin_vel_w"].append(
        robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
      )
      log["body_ang_vel_w"].append(
        robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
      )

      torch.testing.assert_close(
        robot.data.body_link_lin_vel_w[0, 0], motion_base_lin_vel[0]
      )
      torch.testing.assert_close(
        robot.data.body_link_ang_vel_w[0, 0], motion_base_ang_vel[0]
      )

      if reset_flag and not file_saved:
        file_saved = True
        for k in (
          "joint_pos",
          "joint_vel",
          "body_pos_w",
          "body_quat_w",
          "body_lin_vel_w",
          "body_ang_vel_w",
        ):
          log[k] = np.stack(log[k], axis=0)

        np.savez(f"./motions/{output_name}.npz", **log)

        # import wandb
        # COLLECTION = output_name
        # run = wandb.init(project="csv_to_npz", name=COLLECTION, entity="kzakka")
        # print(f"[INFO]: Logging motion to wandb: {COLLECTION}")
        # REGISTRY = "motions"
        # logged_artifact = run.log_artifact(artifact_or_path="/tmp/motion.npz",
        #                                    name=COLLECTION, type=REGISTRY)
        # run.link_artifact(artifact=logged_artifact,
        #                   target_path=f"wandb-registry-{REGISTRY}/{COLLECTION}")
        # print(f"[INFO]: Motion saved to wandb registry: {REGISTRY}/{COLLECTION}")

  # import mediapy as media
  # media.write_video("./motion.mp4", frames, fps=output_fps)
  # print(f"done")


def main(
  input_file: str,
  output_name: str,
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
):
  """Replay motion from CSV file and output to npz file.

  Args:
    input_file: Path to the input CSV file.
    output_name: Path to the output npz file.
    input_fps: Frame rate of the CSV file.
    output_fps: Desired output frame rate.
    device: Device to use.
  """
  sim_cfg = SimulationCfg(device=device)
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  # sim_cfg.render.camera = "robot/tracking"
  # sim_cfg.render.height = 480 * 2
  # sim_cfg.render.width = 640 * 2

  scene = Scene(SCENE_CFG)
  model = scene.compile()

  sim = Simulation(cfg=sim_cfg, model=model)
  scene.initialize(sim.mj_model, sim.data, device, sim.wp_model)

  run_sim(
    sim=sim,
    scene=scene,
    # joint_names=[
    #   "left_hip_pitch_joint",
    #   "left_hip_roll_joint",
    #   "left_hip_yaw_joint",
    #   "left_knee_joint",
    #   "left_ankle_pitch_joint",
    #   "left_ankle_roll_joint",
    #   "right_hip_pitch_joint",
    #   "right_hip_roll_joint",
    #   "right_hip_yaw_joint",
    #   "right_knee_joint",
    #   "right_ankle_pitch_joint",
    #   "right_ankle_roll_joint",
    #   "waist_yaw_joint",
    #   "waist_roll_joint",
    #   "waist_pitch_joint",
    #   "left_shoulder_pitch_joint",
    #   "left_shoulder_roll_joint",
    #   "left_shoulder_yaw_joint",
    #   "left_elbow_joint",
    #   "left_wrist_roll_joint",
    #   "left_wrist_pitch_joint",
    #   "left_wrist_yaw_joint",
    #   "right_shoulder_pitch_joint",
    #   "right_shoulder_roll_joint",
    #   "right_shoulder_yaw_joint",
    #   "right_elbow_joint",
    #   "right_wrist_roll_joint",
    #   "right_wrist_pitch_joint",
    #   "right_wrist_yaw_joint",
    # ],
    joint_names=[
      "AAHead_yaw",
      "Head_pitch",
      "Left_Shoulder_Pitch",
      "Left_Shoulder_Roll",
      "Left_Elbow_Pitch",
      "Left_Elbow_Yaw",
      "Right_Shoulder_Pitch",
      "Right_Shoulder_Roll",
      "Right_Elbow_Pitch",
      "Right_Elbow_Yaw",
      "Waist",
      "Left_Hip_Pitch",
      "Left_Hip_Roll",
      "Left_Hip_Yaw",
      "Left_Knee_Pitch",
      "Left_Ankle_Pitch",
      "Left_Ankle_Roll",
      "Right_Hip_Pitch",
      "Right_Hip_Roll",
      "Right_Hip_Yaw",
      "Right_Knee_Pitch",
      "Right_Ankle_Pitch",
      "Right_Ankle_Roll",
    ],
    input_fps=input_fps,
    input_file=input_file,
    output_fps=output_fps,
    output_name=output_name,
  )


if __name__ == "__main__":
  tyro.cli(main)
