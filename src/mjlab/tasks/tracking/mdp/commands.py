from __future__ import annotations

import copy
import glob
import json
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.forward_kinematics import ForwardKinematics
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
  from collections.abc import Callable
  from typing import Any

  import viser

  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))

# Per-frame arrays of a motion file, which also holds ``joint_names`` and ``fps``.
MOTION_FIELDS = (
  "root_pos_w",
  "root_quat_w",
  "root_lin_vel_w",
  "root_ang_vel_w",
  "joint_pos",
  "joint_vel",
)


def resolve_motion_file(source: str | list[str]) -> list[str]:
  """The file paths of a motion source, with ``~`` expanded: a list's paths, an
  existing path as it is, or the sorted matches of a glob, which must have some."""
  if isinstance(source, list):
    return [os.path.expanduser(path) for path in source]
  source = os.path.expanduser(source)
  if not os.path.exists(source) and any(c in source for c in "*?[]"):
    files = sorted(glob.glob(source))
    if not files:
      raise ValueError(f"No files matched pattern: {source}")
    return files
  return [source]


def load_clip(path: str) -> dict[str, np.ndarray]:
  """The arrays of the motion ``.npz`` file at ``path``."""
  keys = (*MOTION_FIELDS, "joint_names", "fps")
  with np.load(path) as data:
    missing = [key for key in keys if key not in data.files]
    if missing:
      raise ValueError(
        f"{path}: the motion file lacks {missing}; convert the motion with "
        "mjlab.scripts.csv_to_npz"
      )
    return {key: data[key] for key in keys}


class MotionLoader:
  """Reference motion as per-frame tensors on ``device``: the root link's world pose
  and twist and the joint positions and velocities of all clips, concatenated.
  ``body_state`` computes the tracked bodies from them with ``fk``.

  ``motion_file`` is ``.npz`` files (one file, a list or a glob) or a directory
  written by ``scripts/pack_motions.py``. ``motion_ends[i]`` is the exclusive end
  frame of clip ``i`` and ``motion_names[i]`` its name. The clips share ``fps``,
  have ``fk.joint_names`` as their joints and at least two frames each.
  """

  def __init__(
    self, motion_file: str | list[str], fk: ForwardKinematics, device: str = "cpu"
  ) -> None:
    files = resolve_motion_file(motion_file)
    if len(files) == 1 and os.path.isdir(files[0]):
      root = files[0]
      clips = []
      data = {key: np.load(os.path.join(root, f"{key}.npy")) for key in MOTION_FIELDS}
      motion_ends = np.load(os.path.join(root, "motion_ends.npy"))
      fps = float(np.load(os.path.join(root, "fps.npy")).item())
      with open(os.path.join(root, "joint_names.json")) as fh:
        clip_joints = [json.load(fh)]
      with open(os.path.join(root, "motion_names.json")) as fh:
        names = json.load(fh)
    else:
      clips = [load_clip(path) for path in files]
      motion_ends = np.cumsum([clip["joint_pos"].shape[0] for clip in clips])
      fps = float(clips[0]["fps"].item())
      clip_joints = [clip["joint_names"].tolist() for clip in clips]
      names = [os.path.splitext(os.path.basename(p))[0] for p in files]
      for path, clip in zip(files, clips, strict=True):
        clip_fps = float(clip["fps"].item())
        if clip_fps != fps:
          raise ValueError(f"{path}: fps {clip_fps} is not the {fps} of {files[0]}")

    for path, joints in zip(files, clip_joints, strict=True):
      if joints != fk.joint_names:
        raise ValueError(
          f"{path}: motion joints {joints} are not the robot's {fk.joint_names}"
        )

    # A reset frame comes before its clip's last frame, so a clip needs two frames.
    for name, length in zip(names, np.diff(motion_ends, prepend=0), strict=True):
      if length < 2:
        raise ValueError(f"motion clip {name} has {length} frame(s), fewer than 2")

    # The clips join after the checks, whose errors name the file at fault.
    if clips:
      data = {
        key: np.concatenate([clip[key] for clip in clips]) for key in MOTION_FIELDS
      }
    self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
    self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
    self.root_pos_w = torch.tensor(
      data["root_pos_w"], dtype=torch.float32, device=device
    )
    self.root_quat_w = torch.tensor(
      data["root_quat_w"], dtype=torch.float32, device=device
    )
    self.root_lin_vel_w = torch.tensor(
      data["root_lin_vel_w"], dtype=torch.float32, device=device
    )
    self.root_ang_vel_w = torch.tensor(
      data["root_ang_vel_w"], dtype=torch.float32, device=device
    )
    self.fk = fk
    self.fps = fps
    self.time_step_total = self.joint_pos.shape[0]
    self.motion_ends = torch.as_tensor(motion_ends, dtype=torch.long, device=device)
    self.motion_names: list[str] = names

  def body_state(
    self, frames: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """World positions, wxyz quaternions, linear and angular velocities of the
    tracked bodies at ``frames``, without env origins."""
    return self.fk(
      self.root_pos_w[frames],
      self.root_quat_w[frames],
      self.joint_pos[frames],
      self.root_lin_vel_w[frames],
      self.root_ang_vel_w[frames],
      self.joint_vel[frames],
    )


class MotionCommand(CommandTerm):
  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    fk = ForwardKinematics(
      env.sim.mj_model,
      self.robot.indexing.body_ids,
      self.robot.indexing.joint_q_adr,
      self.body_indexes,
      device=self.device,
    )
    self.motion = MotionLoader(self.cfg.motion_file, fk, device=self.device)
    # One fused kernel per call; env subsets of two or more share one graph.
    self._body_state = torch.compile(
      self.motion.body_state,
      dynamic=True,
      options={"max_fusion_size": 512},
      disable=not self.device.startswith("cuda"),
    )
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    # The tracked bodies' reference states at ``time_steps``, without env origins.
    (
      self._body_pos_w,
      self._body_quat_w,
      self._body_lin_vel_w,
      self._body_ang_vel_w,
    ) = self._body_state(self.time_steps)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    self.bin_count = int(self.motion.time_step_total // (1 / env.step_dt)) + 1
    self.bin_failed_count = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self._current_bin_failed = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self.kernel = torch.tensor(
      [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)
    self._pending_forward = False

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.time_steps]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self._body_pos_w + self._env.scene.env_origins[:, None, :]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._body_quat_w

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self._body_lin_vel_w

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self._body_ang_vel_w

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self._body_pos_w[:, self.motion_anchor_body_index] + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self._body_quat_w[:, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self._body_lin_vel_w[:, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self._body_ang_vel_w[:, self.motion_anchor_body_index]

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  @property
  def yaw_offset_quat_w(self) -> torch.Tensor:
    """Yaw offset from the reference anchor to the robot anchor."""
    return yaw_quat(quat_mul(self.robot_anchor_quat_w, quat_inv(self.anchor_quat_w)))

  def _update_metrics(self):
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    episode_failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(episode_failed):
      current_bin_index = torch.clamp(
        (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1),
        0,
        self.bin_count - 1,
      )
      fail_bins = current_bin_index[env_ids][episode_failed]
      self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

    # Sample.
    sampling_probabilities = (
      self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
    )
    sampling_probabilities = torch.nn.functional.pad(
      sampling_probabilities.unsqueeze(0).unsqueeze(0),
      (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
      mode="replicate",
    )
    sampling_probabilities = torch.nn.functional.conv1d(
      sampling_probabilities, self.kernel.view(1, 1, -1)
    ).view(-1)

    sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

    sampled_bins = torch.multinomial(
      sampling_probabilities, len(env_ids), replacement=True
    )
    # Bins map to frames with int64 multiply-first math, which is exact for any
    # bin count. Frames stay before ``total``, the last frame.
    total = self.motion.time_step_total - 1
    lo = (sampled_bins * total) // self.bin_count
    hi = ((sampled_bins + 1) * total) // self.bin_count
    jitter = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
    self.time_steps[env_ids] = lo + (jitter * (hi - lo)).long()

    # Update metrics.
    H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
    H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else 1.0
    pmax, imax = sampling_probabilities.max(dim=0)
    self.metrics["sampling_entropy"][:] = H_norm
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

  def _uniform_sampling(self, env_ids: torch.Tensor):
    self.time_steps[env_ids] = torch.randint(
      0, self.motion.time_step_total - 1, (len(env_ids),), device=self.device
    )
    self.metrics["sampling_entropy"][:] = 1.0  # Maximum entropy for uniform.
    self.metrics["sampling_top1_prob"][:] = 1.0 / self.bin_count
    self.metrics["sampling_top1_bin"][:] = 0.5  # No specific bin preference.

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    """Clip joint positions and write root + joint state to sim."""
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
      ends = self.motion.motion_ends
      if len(ends) > 1:
        # The first frame of a clip drawn uniformly at random.
        starts = torch.cat([ends.new_zeros(1), ends[:-1]])
        clips = torch.randint(len(starts), (len(env_ids),), device=self.device)
        self.time_steps[env_ids] = starts[clips]
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

    # An env reset advances the command by one frame before the first observation,
    # so a reset frame is at most the frame before its clip's last frame.
    frames = self.time_steps[env_ids]
    on_last = torch.isin(frames, self.motion.motion_ends - 1)
    frames = torch.where(on_last, frames - 1, frames)
    self.set_time_steps(env_ids, frames)

    root_pos = self.body_pos_w[env_ids, 0].clone()
    root_ori = self.body_quat_w[env_ids, 0].clone()
    root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
    root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori = quat_mul(orientations_delta, root_ori)
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel += rand_samples[:, :3]
    root_ang_vel += rand_samples[:, 3:]

    joint_pos = self.joint_pos[env_ids].clone()
    joint_vel = self.joint_vel[env_ids]

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,
    )

    self._write_reference_state_to_sim(
      env_ids,
      root_pos,
      root_ori,
      root_lin_vel,
      root_ang_vel,
      joint_pos,
      joint_vel,
    )
    self._pending_forward = True

  def set_time_steps(self, env_ids: torch.Tensor | slice, frames: torch.Tensor) -> None:
    """Move ``env_ids`` to ``frames`` and compute their tracked bodies there. A
    direct write to ``time_steps`` leaves the bodies at their previous frames."""
    self.time_steps[env_ids] = frames
    (
      self._body_pos_w[env_ids],
      self._body_quat_w[env_ids],
      self._body_lin_vel_w[env_ids],
      self._body_ang_vel_w[env_ids],
    ) = self._body_state(frames)

  def update_relative_body_poses(self) -> None:
    """Recompute ``body_pos_relative_w`` and ``body_quat_relative_w``.

    Called after ``reset_to_frame`` so that termination checks that
    compare relative body positions see the correct state.
    """
    delta_pos_w = self.robot_anchor_pos_w.clone()
    delta_pos_w[:, 2] = self.anchor_pos_w[:, 2]
    yaw = self.yaw_offset_quat_w[:, None].expand(-1, len(self.cfg.body_names), -1)
    self.body_quat_relative_w = quat_mul(yaw, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w[:, None] + quat_apply(
      yaw, self.body_pos_w - self.anchor_pos_w[:, None]
    )

  def to_robot_heading(self, vec_w: torch.Tensor) -> torch.Tensor:
    """Rotate reference body vectors by the yaw offset between the anchors."""
    yaw = self.yaw_offset_quat_w[:, None].expand(-1, vec_w.shape[1], -1)
    return quat_apply(yaw, vec_w)

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    extras = super().reset(env_ids)
    # Reset-path resamples are followed by the env's own forward(); only
    # compute-path resamples (wraparound or timer expiry) need our refresh.
    self._pending_forward = False
    return extras

  def _update_command(self, env_ids: torch.Tensor | None = None):
    ids = env_ids
    if ids is None:
      ids = torch.arange(self.num_envs, device=self.device)
    self.time_steps[ids] += 1
    # A frame that reaches a clip end has left its clip.
    wrap_ids = ids[torch.isin(self.time_steps[ids], self.motion.motion_ends)]
    if wrap_ids.numel() > 0:
      self._resample_command(wrap_ids)
    self.set_time_steps(ids, self.time_steps[ids])

    # _resample_command writes qpos/qvel but does not refresh derived
    # quantities; forward() so update_relative_body_poses reads the
    # post-teleport robot anchor instead of the stale pre-resample pose.
    # Covers both the wraparound above and a timer-expiry resample that ran
    # in compute before this call.
    if self._pending_forward:
      self._pending_forward = False
      self._env.sim.forward()

    self.update_relative_body_poses()

    # Fold failure counts into the EMA only on the per-step update so
    # manual resets do not decay it faster.
    if env_ids is None and self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw ghost robot or frames based on visualization mode."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        # Build a ghost model with only visual geoms visible. Collision geoms (nonzero
        # contype/conaffinity) get alpha=0 so the viewer's alpha filter excludes them.
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        for gi in range(self._ghost_model.ngeom):
          if (
            self._ghost_model.geom_contype[gi] != 0
            or self._ghost_model.geom_conaffinity[gi] != 0
          ):
            self._ghost_model.geom_rgba[gi, 3] = 0
          else:
            self._ghost_model.geom_rgba[gi] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.entity_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

        visualizer.add_ghost_mesh(
          qpos,
          model=self._ghost_model,
          label=f"ghost_{batch}",
        )

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_quat = self.body_quat_w[batch]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[batch]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

        for i, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_body_pos[i],
            rotation_matrix=desired_body_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[i],
            rotation_matrix=current_body_rotm[i],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
          )

        desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
        desired_anchor_quat = self.anchor_quat_w[batch]
        desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=desired_anchor_pos,
          rotation_matrix=desired_rotation_matrix,
          scale=0.1,
          label=f"desired_anchor_{batch}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )

        current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
        current_anchor_quat = self.robot_anchor_quat_w[batch]
        current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=current_anchor_pos,
          rotation_matrix=current_rotation_matrix,
          scale=0.15,
          label=f"current_anchor_{batch}",
        )

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create motion scrubber controls in the Viser viewer."""
    max_frame = int(self.motion.time_step_total) - 1

    with server.gui.add_folder(name.capitalize()):
      scrubber = server.gui.add_slider(
        "Frame",
        min=0,
        max=max_frame,
        step=1,
        initial_value=0,
      )

      @scrubber.on_update
      def _(_) -> None:
        env_ids = torch.tensor([get_env_idx()], device=self.device)
        self.set_time_steps(env_ids, torch.full_like(env_ids, int(scrubber.value)))
        if on_change is not None:
          on_change()

      all_envs_cb = server.gui.add_checkbox("All envs", initial_value=True)
      start_btn = server.gui.add_button("Start Here")

      @start_btn.on_click
      def _(_) -> None:
        if request_action is not None:
          request_action(
            "CUSTOM",
            {"type": "gui_reset", "all_envs": all_envs_cb.value},
          )

    self._scrubber_handles = (scrubber, all_envs_cb, start_btn)
    self._set_scrubber_disabled(True)

  def _set_scrubber_disabled(self, disabled: bool) -> None:
    """Enable or disable the motion scrubber GUI controls."""
    for handle in self._scrubber_handles:
      handle.disabled = disabled

  def on_viewer_pause(self, paused: bool) -> None:
    if hasattr(self, "_scrubber_handles"):
      self._set_scrubber_disabled(not paused)

  def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
    if not hasattr(self, "_scrubber_handles"):
      return False
    frame = int(self._scrubber_handles[0].value)
    self.reset_to_frame(env_ids, frame)
    # reset_to_frame writes qpos/qvel; forward so update_relative_body_poses
    # reads the scrubbed pose instead of the stale pre-scrub kinematics.
    self._env.sim.forward()
    self.update_relative_body_poses()
    return True

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
    """Reset to exact reference state at a specific frame.

    Like ``_resample_command`` but deterministic: no random
    perturbations to pose, velocity, or joint positions.
    """
    self.set_time_steps(env_ids, torch.full_like(env_ids, frame))
    self._write_reference_state_to_sim(
      env_ids,
      self.body_pos_w[env_ids, 0],
      self.body_quat_w[env_ids, 0],
      self.body_lin_vel_w[env_ids, 0],
      self.body_ang_vel_w[env_ids, 0],
      self.joint_pos[env_ids],
      self.joint_vel[env_ids],
    )


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
  motion_file: str | list[str]
  anchor_body_name: str
  body_names: tuple[str, ...]
  entity_name: str
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MotionCommand(self, env)
