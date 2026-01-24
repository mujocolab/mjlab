"""
Convert CSV motion data to NPZ format for Unitree G1.
Supports optional local rendering and optional WandB logging.
"""

import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import tyro
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.utils.lab_api.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    quat_slerp,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

# --- Constants ---

JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", 
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", 
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", 
    "right_wrist_yaw_joint",
]


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device | str,
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
        # convert to wxyz: input likely [x, y, z, w] -> [w, x, y, z]
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt

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
            f"Motion interpolated, input frames: {self.input_frames}, "
            f"input fps: {self.input_fps}, "
            f"output frames: {self.output_frames}, "
            f"output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], float(blend[i]))
        return slerped_quats

    def _compute_frame_blend(
        self, times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        """Computes the derivative of a sequence of SO3 rotations."""
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat(
            [omega[:1], omega, omega[-1:]], dim=0
        )  # repeat first and last sample
        return omega

    def get_next_state(self) -> tuple[tuple[torch.Tensor, ...], bool]:
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


def run_simulation_loop(
    sim: Simulation,
    scene: Scene,
    joint_names: List[str],
    motion: MotionLoader,
    render: bool,
    renderer: Optional[OffscreenRenderer] = None,
) -> Tuple[dict[str, Any], List[np.ndarray]]:
    """Runs the simulation and collects data and frames."""
    robot: Entity = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    log: dict[str, Any] = {
        "fps": [motion.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    frames = []
    scene.reset()

    print(f"\nStarting simulation with {motion.output_frames} frames...")
    if render:
        print("Rendering enabled - generating video frames...")

    pbar = tqdm(
        total=motion.output_frames,
        desc="Processing frames",
        unit="frame",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    reset_flag = False
    frame_count = 0

    while not reset_flag:
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

        # Set simulation state from motion
        root_states = robot.data.default_root_state.clone()
        root_states[:, 0:3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.forward()
        scene.update(sim.mj_model.opt.timestep)
        
        if render and renderer is not None:
            renderer.update(sim.data)
            frames.append(renderer.render())

        # Logging
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

        # Sanity check
        torch.testing.assert_close(
            robot.data.body_link_lin_vel_w[0, 0], motion_base_lin_vel[0]
        )
        torch.testing.assert_close(
            robot.data.body_link_ang_vel_w[0, 0], motion_base_ang_vel[0]
        )

        frame_count += 1
        pbar.update(1)

        if frame_count % 100 == 0:
            elapsed_time = frame_count / motion.output_fps
            pbar.set_description(f"Processing frames (t={elapsed_time:.1f}s)")

    pbar.close()

    # Stack lists into arrays
    print("\nStacking arrays...")
    for k in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ):
        log[k] = np.stack(log[k], axis=0)

    return log, frames


def save_local_files(
    log_data: dict, 
    frames: List[np.ndarray], 
    output_dir: Path, 
    output_name: str, 
    fps: float
) -> Tuple[Path, Optional[Path]]:
    """Saves NPZ and optionally MP4 to the local filesystem."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save NPZ
    npz_path = output_dir / f"{output_name}.npz"
    print(f"Saving data to {npz_path}...")
    np.savez(npz_path, **log_data)

    # Save Video
    mp4_path = None
    if frames:
        from moviepy import ImageSequenceClip
        mp4_path = output_dir / f"{output_name}.mp4"
        print(f"Creating video at {mp4_path}...")
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(str(mp4_path), logger=None)
        print("Video saved.")

    return npz_path, mp4_path


def upload_to_wandb(
    npz_path: Path, 
    mp4_path: Optional[Path], 
    output_name: str,
    project_name: str = "csv_to_npz"
):
    """Uploads artifacts to Weights & Biases."""
    if wandb is None:
        print("[ERROR]: wandb is not installed. Skipping upload.")
        return

    print(f"Initializing WandB project: {project_name}")
    run = wandb.init(project=project_name, name=output_name)
    
    print(f"[INFO]: Logging motion to wandb: {output_name}")
    REGISTRY = "motions"
    
    # Log NPZ
    logged_artifact = run.log_artifact(
        artifact_or_path=str(npz_path), name=output_name, type=REGISTRY
    )
    run.link_artifact(
        artifact=logged_artifact,
        target_path=f"wandb-registry-{REGISTRY}/{output_name}",
    )
    print(f"[INFO]: Motion saved to wandb registry: {REGISTRY}/{output_name}")

    # Log Video
    if mp4_path and mp4_path.exists():
        print("Logging video to wandb...")
        wandb.log({"motion_video": wandb.Video(str(mp4_path), format="mp4")})

    wandb.finish()


def main(
    input_file: str,
    output_name: str,
    output_dir: str = "./outputs",
    input_fps: float = 30.0,
    output_fps: float = 50.0,
    device: str = "cuda:0",
    render: bool = False,
    use_wandb: bool = True,
    wandb_project: str = "Mjlab-Spinkick-Unitree-G1",
    line_range: tuple[int, int] | None = None,
):
    """Replay motion from CSV file and output to npz file.

    Args:
        input_file: Path to the input CSV file.
        output_name: Name of the output file (without extension).
        output_dir: Directory to save local files.
        input_fps: Frame rate of the CSV file.
        output_fps: Desired output frame rate.
        device: Device to use.
        render: Whether to render the simulation and save a video.
        use_wandb: Whether to upload results to Weights & Biases.
        wandb_project: WandB project name.
        line_range: Range of lines to process from the CSV file.
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARNING]: CUDA is not available. Falling back to CPU. This may be slow.")
        device = "cpu"

    output_path = Path(output_dir)

    # 1. Setup Simulation
    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / output_fps

    scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)

    # 2. Setup Renderer
    renderer = None
    if render:
        viewer_cfg = ViewerConfig(
            height=480,
            width=640,
            origin_type=ViewerConfig.OriginType.ASSET_ROOT,
            distance=2.0,
            elevation=-5.0,
            azimuth=20,
        )
        renderer = OffscreenRenderer(
            model=sim.mj_model,
            cfg=viewer_cfg,
            scene=scene,
        )
        renderer.initialize()

    # 3. Load Motion
    motion = MotionLoader(
        motion_file=input_file,
        input_fps=input_fps,
        output_fps=output_fps,
        device=sim.device,
        line_range=line_range,
    )

    # 4. Run Simulation
    log_data, frames = run_simulation_loop(
        sim=sim,
        scene=scene,
        joint_names=JOINT_NAMES,
        motion=motion,
        render=render,
        renderer=renderer,
    )

    # 5. Save Local Files
    npz_file, mp4_file = save_local_files(
        log_data=log_data,
        frames=frames,
        output_dir=output_path,
        output_name=output_name,
        fps=output_fps
    )

    # 6. Upload to WandB (Optional)
    if use_wandb:
        upload_to_wandb(
            npz_path=npz_file, 
            mp4_path=mp4_file, 
            output_name=output_name,
            project_name=wandb_project
        )
    else:
        print("\nWandB logging disabled. Files are available locally.")
        print(f"NPZ: {npz_file}")
        if mp4_file:
            print(f"Video: {mp4_file}")


if __name__ == "__main__":
    tyro.cli(main)